---
title: "Megatron-LM Optimizer Step 内部机制深度解析"
---

# Megatron-LM Optimizer Step 内部机制深度解析

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **学习前置**：[[16_megatron_distributed_optimizer_analysis]]；数值格式细节见 [[23_megatron_precision_cudagraph_fusion_analysis]]。
> **回答的问题**：参数组怎样进入具体 optimizer，混合精度 step 如何完成 unscale/overflow/clip/update/copy-back，scheduler 与 μP 又怎样改写各组状态？
> **不覆盖**：DDP buffer、ZeRO/HSDP 与 FSDP 方案选型归 16；通用 Muon 数学见 [[11_muon_analysis]]。
> **叙事顺序**：边界 → factory → μP 参数组 → mixed-precision step → scheduler/offload/Muon → 约束与趋势。
> **最后复核**：2026-09-03。

---

## 1. 背景：分片回答“状态在哪”，step 回答“状态怎样变”

一次训练迭代在 `finalize_model_grads` 后仍有四件事没有完成：把模型梯度转成 optimizer 可消费的表示、处理 loss scale 与非有限值、执行裁剪和参数更新、推进 LR/WD。16 号页负责参数/梯度/state 沿 DP 怎样分片；本页只追更新边界内的状态变化。

`setup_model_and_optimizer()` 先生成 standard param-group overrides；启用 μP 时再从模型配置取得 `mup_width_mult`、合并 μP overrides，随后调用 `get_megatron_optimizer()` 并创建 `OptimizerParamScheduler`（`megatron/training/training.py:2799-2827`）。这条顺序是本页的入口合同：**先确定每个参数组的 LR/WD/eps 语义，再选择 optimizer wrapper，最后让 scheduler 驱动这些组**。

## 2. 工厂与类层次：先决定参数组，再选择 wrapper

`get_megatron_optimizer` 不把所有场景塞进一个巨型 optimizer：入口先补齐 overrides、做一致性检查，再把非 Adam/SGD 交给 emerging 路径（`megatron/core/optimizer/__init__.py:975-1031`）；standard 路径组织参数组和进程组后，必须继续调用 `_get_megatron_optimizer_based_on_param_groups`，才真正选中 raw optimizer 与 wrapper。也就是说，入口分流本身不是 concrete wrapper 的构造点。

`megatron/core/optimizer/optimizer.py` 的继承链:

```
MegatronOptimizer (ABC, :134)              抽象基类:clip_grad_norm / get_loss_scale / scale_loss / step
   │
   ├── MixedPrecisionOptimizer (:717)      混合精度:fp32 master 副本 + grad scaler
   │      ├── Float16OptimizerWithFloat16Params (:929)   fp16/bf16 模型参数的具体实现
   │      └── DistributedOptimizer          ← ZeRO-1,见 [[16_megatron_distributed_optimizer_analysis]] §5 阶段②
   │
   ├── FP32Optimizer (:1286)               纯 fp32,无 scaling、无 master 副本
   │
   └── ChainedOptimizer (:1478)            把多个优化器串成一个(见 §2.1)
```

> [!update] 该特性自提交 `232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> **行号基线刷新**:`megatron/core/optimizer/optimizer.py` 自 `ee3f1ff` 起明显增长(emerging-optimizer / MXFP8 / layer-wise 相关代码),上面继承链的行号整体上移。新基线 `71092579` 锚点(`megatron/core/optimizer/optimizer.py`):
> - `MegatronOptimizer` `:134`(`ee3f1ff` `:100` → `232c478d4` `:133`)、`MixedPrecisionOptimizer` `:717`(`:465` → `:589`)、`Float16OptimizerWithFloat16Params` `:929`(`:654` → `:779`)、`FP32Optimizer` `:1286`(`:918` → `:1042`)、`ChainedOptimizer` `:1478`(`:1104` → `:1229`)。
> - `MixedPrecisionOptimizer.step()` `:895`(`:621` → `:745`);`prepare_grads` `:807`、`step_with_ready_grads` `:852`(`232c478d4` 时为 `:676`/`:712`)。
> 类层次与五步流程本身**未变**,仅行号漂移。

standard 路径从入口到 concrete wrapper 的零搜索 hop 是：

```
get_megatron_optimizer                                      :975-1031
  ├─ Megatron-FSDP 参数组 → helper 调用                    :1068-1100
  └─ 普通 dense 参数组    → helper 调用                    :1126-1163
       └─ _get_megatron_optimizer_based_on_param_groups    :451-695
            ├─ raw optimizer（CPU offload / Adam / SGD）   :502-630
            ├─ grad scaler                                 :635-665
            └─ Distributed / Float16 / FP32 wrapper        :666-686
```

emerging optimizer 则由入口 `:1025-1031` 转入 `_get_megatron_emerging_optimizer`（`:725-972`），在那里构造 LayerWise/子 optimizer，必要时以 `ChainedOptimizer` 收口。这个两段式 factory 让 base optimizer 只负责更新数学，mixed-precision、DistributedOptimizer 与 LayerWise wrapper 分别负责数值状态和通信；“不把职责揉进一个类”的收益是本文基于上述分支作出的设计归纳。

### 2.1 `ChainedOptimizer` 为什么需要

一个模型常需要**多个优化器实例**:
- **稠密参数 vs 专家参数**:MoE 模型里专家参数走 EP 组、稠密参数走普通 DP 组,分片域不同 → 各用一个 `DistributedOptimizer`。
- `num_distributed_optimizer_instances > 1`(HSDP)。

`ChainedOptimizer` 把它们包成一个对外统一的优化器:`step()` 时依次驱动每个子优化器,`get_loss_scale` / `clip_grad` 跨子优化器协调。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — ChainedOptimizer 的 MXFP8 defer-sync 门控修正(#4982,`megatron/core/optimizer/optimizer.py:1806` `_should_defer_mxfp8_param_sync`)
> 当 `reuse_grad_buf_for_mxfp8_param_ag=True`(MXFP8 参数 all-gather 复用梯度 buffer)且 DDP 层 **未** 开 `overlap_param_gather` 时,链式 step 间会有参数 buffer 复用竞态,需把 MXFP8 参数同步**延迟**到所有子优化器 step 完成后再做。原实现用 `self.config.overlap_param_gather` 作为判据,但 `OptimizerConfig` 与 DDP config 的该字段**可能不一致**;修复后改为**直接探测每个子 `DistributedOptimizer.ddp_config.overlap_param_gather`**,任一为 False 即触发延迟同步。这是 ChainedOptimizer 与 DDP 层耦合的一个隐蔽点。

### 2.2 param group:weight decay 的区分

`get_megatron_optimizer` 建 param group 时把参数分两组:**该用 weight decay 的**(线性层权重)和**不该用的**(bias、LayerNorm 的 `weight`/`bias`)。后者 `weight_decay=0`。这是标准做法,避免对归一化/偏置施加权重衰减。

---

## 3. μP：宽度变化怎样落到初始化与 param group

μP 不是一种新的 optimizer 类，而是一组在模型配置和 optimizer param group 两端同时生效的缩放规则。只改初始化、不改 LR/eps，或只改 optimizer、不改 attention/output scale，都会破坏它试图保持的跨宽度尺度关系。

### 3.1 模型配置先产生宽度乘数

`TransformerConfig.__post_init__` 在 `use_mup=True` 时令 `mup_width_mult = hidden_size / mup_base_hidden_size`，据此设置 attention `softmax_scale` 与默认 `mup_output_mult`；默认 hidden-layer 初始化标准差除以 `sqrt(width_mult)`，output layer 还同时按 depth 与 width 缩放（`megatron/core/transformer/transformer_config.py:2893-2969`）。embedding 初始化刻意在这两段之间建立，保留未缩放的基准标准差。自定义 `init_method` 或 `output_layer_init_method` 不会被静默覆盖，而是发 warning，因为它可能破坏上述假设（`:2913-2930`）。

### 3.2 训练入口把缩放规则合并进 optimizer 参数组

`setup_model_and_optimizer` 先调用 `get_standard_config_overrides()` 建 weight-decay/decoupled-LR 规则，再调用 `get_mup_config_overrides(config, width_mult, optimizer_type)`，把非空结果合并后交给 optimizer factory（`megatron/training/training.py:2665-2676,2799-2824`）。工厂在构造具体 optimizer 前执行一致性检查并按 standard/emerging 路径分流（`megatron/core/optimizer/__init__.py:984-1031`）。这闭合了 `TransformerConfig → width_mult → MuP overrides → param groups → optimizer` 的执行 hop。

| 参数/optimizer 类别 | 当前实现的缩放 | 边界 |
|---|---|---|
| Adam/AdamW hidden matrix | `max_lr`、`min_lr` 与 `eps` 均除以 `width_mult` | vector-like 参数保留基准 LR/eps |
| SGD vector-like | LR 乘 `width_mult` | hidden matrix 在当前 uniform-width 实现中保持基准 LR |
| 其他非 Adam optimizer | hidden matrix LR 除以 `width_mult` | 不覆盖 eps |
| Muon 管理的 matrix | 不套 Adam 风格 μP override | 继续由 Muon 自身 scale mode 管理；spectral 模式会告警 |
| decoupled embedding/output | 保留显式 decoupled LR | μP 不覆盖这些绝对值 |

这些分类、predicate、decoupled 与非 decoupled 分支以及最终返回的 `ParamGroupOverride` 均在 `get_mup_config_overrides()` 中实现（`megatron/core/optimizer/__init__.py:131-297`）；`width_mult==1` 时直接返回空字典（`:193-195`）。运行时，embedding 输出乘 `mup_embedding_mult`（`megatron/core/models/common/embeddings/language_model_embedding.py:131-132`），logits 则由 `_scale_logits` 应用 `mup_output_mult`（`megatron/core/models/common/language_module/language_module.py:312-325`）。

## 4. 混合精度优化器:fp32 master 副本

### 4.1 动机

模型用 bf16/fp16 做前向反向(省显存、快)。但**优化器更新若也用 bf16**:`param += lr · update`,当 `update` 比 `param` 小几个数量级时,bf16 的尾数位不够,加法**直接丢失** → 训练停滞。

**解法:fp32 master 副本**。优化器维护一份 fp32 的参数主拷贝,所有 Adam 更新在 fp32 上做;每步结束把 fp32 master **拷回** bf16 模型参数供下一步前向用。

### 4.2 这就是"18 bytes/param"的来源

`MixedPrecisionOptimizer` 持有的东西,正是 [[16_megatron_distributed_optimizer_analysis]] §§1.4、9.1 ZeRO 显存表里的 `18Ψ`:

| 张量 | 精度 | bytes/param | 谁持有 |
|------|------|-------------|--------|
| 模型权重 | bf16 | 2 | 模型 |
| 模型梯度 | **fp32** | **4** | DDP grad buffer(bf16 训练强制 fp32 累加) |
| **master 权重** | fp32 | 4 | **优化器** |
| **Adam 动量 m** | fp32 | 4 | **优化器** |
| **Adam 方差 v** | fp32 | 4 | **优化器** |
| | | **合计 18** | |

> 梯度为 **fp32(4 字节)** 而非 bf16(2 字节):bf16 尾数仅 7 位,跨 microbatch 累加会丢精度,Megatron 对 bf16 训练强制 fp32 梯度累积(`megatron/training/arguments.py:1319-1333`,见 [[16_megatron_distributed_optimizer_analysis]] §3.5)。仅 `--grad-reduce-in-bf16` 时梯度为 2 字节、合计 16。

[[16_megatron_distributed_optimizer_analysis]] §§1.4、5 阶段②说 ZeRO-1 把"优化器状态 `12Ψ`"切成 `1/dp` —— 切的就是这里的 master + m + v。

`FP32Optimizer` 则相反:模型本身就是 fp32,无需 master 副本、无需 scaler。

### 4.3 精度感知优化器:decoupled_grad

见 [[16_megatron_distributed_optimizer_analysis]] §3.11——`use_precision_aware_optimizer: True` 时,master 权重、exp_avg、exp_avg_sq 可采用不同的低精度格式,用 `.decoupled_grad` 解耦模型参数 dtype 和优化器 state dtype,而非本节默认的固定 fp32 master。

---

## 5. `optimizer.step()` 流程

在展开 `optimizer.step()` 内部五步之前,先看它在**整个训练迭代**里的位置（补充，2026-07-31 由原 16 §3.5 并入）——这条流程串起了 [[16_megatron_distributed_optimizer_analysis]] §3(DP 通信)与本节(优化器内部):

```
Forward（参数 AG 可 overlap）
  → Backward + finalize_model_grads（梯度 RS/AR 可 overlap）
    → optimizer.step()
        ├─ prepare_grads()
        │    ├─ optimizer-state/master 预取（启用 chunked offload 时）
        │    ├─ _copy_model_grads_to_main_grads() [model grad → main grad]
        │    └─ unscale + non-finite check
        ├─ clip_grad_norm() + count_zeros()
        └─ step_with_ready_grads()
             ├─ raw optimizer / offloader.step()
             ├─ _copy_main_params_to_model_params() 或写 param buffer
             └─ DistributedOptimizer 同步参数，或登记给下一轮 pre-hook 发起 AG
      → 跨 MP rank 合并 update_successful
        → 成功才 scheduler.step()；失败则 LR/WD 不前进
          → 下一轮迭代
```

这里两个 copy 都是 `optimizer.step()` 的内部子调用：`prepare_grads()` 在 `megatron/core/optimizer/optimizer.py:807-849` 做 model-grad→main-grad，`step_with_ready_grads()` 在 `:852-892` 做更新与 main-param 回写，`step()` 本身在 `:895-926` 串起它们。DistributedOptimizer 再在 override `megatron/core/optimizer/distrib_optimizer.py:3251-3283` 中处理参数同步。DP 通信的前后边界见 [[16_megatron_distributed_optimizer_analysis]] §3。

`MixedPrecisionOptimizer.step()`(`megatron/core/optimizer/optimizer.py:895`;`prepare_grads` 在 `:807`、`step_with_ready_grads` 在 `:852`):

```python
def step(self):
    found_inf_flag = self.prepare_grads()          # ① 收梯度 + unscale + 查 inf/nan
    if found_inf_flag:
        return False, None, None                   #    有 inf/nan → 跳过本步
    grad_norm = 0.0
    if self.config.clip_grad > 0.0:
        grad_norm = self.clip_grad_norm(self.config.clip_grad)   # ② 全局梯度裁剪
    num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else 0  # ③ 可选统计
    success = self.step_with_ready_grads()          # ④ Adam 更新 + master→bf16 回拷
    return success, grad_norm, num_zeros_in_grad
```

五步:

```
① prepare_grads     bf16 模型梯度 ──拷贝/累加──► fp32 main grad
                     除以 loss scale(unscale)
                     扫描 inf/nan → found_inf_flag

   found_inf_flag?  ──是──► return False(跳过本步,dynamic scaler 随后降 scale)
        │否
② clip_grad_norm    按 wrapper 注入的统计组规约；FSDP DTensor 另补 shard 组(§7)
        │
③ count_zeros       (可选)统计零梯度数,日志用
        │
④ step_with_ready_grads
        │           base optimizer(FusedAdam)在 fp32 master 上做 Adam 更新
        │           fp32 master ──拷回──► bf16 模型参数
        ▼
   返回 (success, grad_norm, num_zeros)
```

关键:**inf/nan 检查在最前面**。一旦发现非有限梯度,整步丢弃(参数不动),交给 dynamic scaler 调整(§6)。这是 fp16 训练能稳住的安全阀。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — `count_zeros` 兼容解耦梯度 / Megatron-FSDP(#4802,`megatron/core/optimizer/clip_grads.py:199` `count_zeros_fp32`,新基线下行号未变)
> 第 ③ 步 `count_zeros`(统计零梯度)原来固定读 `param.grad`。但两种新路径下梯度不在 `.grad`:① **precision-aware / 解耦优化器**(`use_decoupled_grad=True`)梯度在 `param.decoupled_grad`(见 §4.3 与 [[16_megatron_distributed_optimizer_analysis]] §3.11);② **Megatron-FSDP** 管理的参数梯度是 FSDP 分片后的 DTensor,需取 `._local_tensor`。修复后 `count_zeros_fp32` 先按 `use_decoupled_grad` 选 `decoupled_grad`/`grad` 属性,再对 `__fsdp_param__` 参数取 local shard,避免漏统计或读到 `None`。

---

## 6. Loss Scaling 与 GradScaler

### 6.1 动机:fp16 的下溢

fp16 动态范围窄(最小正规数 ~6e-5)。反向里很多梯度比这还小 → **下溢成 0** → 参数收不到更新。

**Loss scaling**:前向后把 loss 乘一个大数 `S`(`scale_loss`),反向链式法则使所有梯度同样 ×`S`,把小梯度抬进 fp16 可表示区间;`prepare_grads` 里再 ÷`S` 还原(unscale)。

> bf16 动态范围与 fp32 几乎一样宽,通常**不需要** scaler(或用 `ConstantGradScaler(1.0)`)。loss scaling 主要为 fp16。

### 6.2 两种 scaler(`megatron/core/optimizer/grad_scaler.py`)

**`ConstantGradScaler`**:固定 `S`。简单,适合 bf16 或已知稳定的场景。

**`DynamicGradScaler`(`megatron/core/optimizer/grad_scaler.py:64`)**:自适应。
- 连续 `growth_interval` 步无 inf/nan → `S ×= growth_factor`(往上试探,尽量大)。
- 连续 `hysteresis` 步检测到 inf/nan → `S ×= backoff_factor`(`<1`,减半之类)、且这些步**跳过更新**。
- `min_scale` 兜底。

直觉:`S` 越大越不下溢,但太大会上溢成 inf。dynamic scaler 在"尽量大"和"不溢出"之间自动平衡 —— 不断试着调大,溢出了就回退。

---

## 7. 梯度裁剪(`megatron/core/optimizer/clip_grads.py`)

`clip_grad_norm(clip_grad)`:计算**全局梯度范数** `‖g‖`(所有参数拼起来的 L2 范数),若 `‖g‖ > clip_grad`(默认 1.0),把所有梯度等比缩放到 `clip_grad`:`g ← g · clip_grad / ‖g‖`。

并行要点不是统一写死“跨 TP×PP×DP”，而是由 wrapper 注入统计通信组。`MegatronOptimizer.clip_grad_norm` 把 `self.get_grad_stats_parallel_group()` 传给 `get_grad_norm_fp32`（`megatron/core/optimizer/optimizer.py:444-458`），后者再执行规约（`megatron/core/optimizer/clip_grads.py:55-138`）：

| 参数表示 | factory 注入的统计组 | 为什么不会重复计算 |
|---|---|---|
| replicated / 非分片 Float16 或 FP32 wrapper | `model_parallel_group`（`megatron/core/optimizer/__init__.py:680-686`） | DP 梯度同步后各 DP rank 已有相同值，只补模型并行分片 |
| `DistributedOptimizer` | `intra_dist_opt_group`（`:666-679`） | 梯度/状态按 distributed-optimizer 域分片，统计必须覆盖该实例域 |
| Megatron-FSDP | `no_shard` 注入 `mp_group`，其余策略注入 `intra_dist_opt_group`（`:1068-1075`） | `get_grad_norm_fp32` 先从 DTensor 取 local shard，并在检测到的 data-parallel shard group 上规约，再沿注入组补齐其余分片（`clip_grads.py:83-87,97-104,131-138`） |

算出总范数后，`clip_grad_by_total_norm_fp32`（`megatron/core/optimizer/clip_grads.py:147`）才在本 rank 的梯度上等比缩放。这个分层既挡住梯度尖峰，也避免对已经复制的 DP 梯度重复求和。

---

## 8. LR / WD 调度(`OptimizerParamScheduler`)

### 8.1 曲线与参数组状态

`megatron/core/optimizer_param_scheduler.py:102`(`class OptimizerParamScheduler`)。每步更新 base optimizer 各 param group 的 **learning rate** 和 **weight decay**。

典型曲线 = **warmup + decay**:
- **warmup**:前 `lr_warmup_steps` 步,LR 从 0(或 `lr_warmup_init`)线性升到峰值 —— 避免初期大步长震荡。
- **decay**:之后按 `lr_decay_style` 衰减到 `min_lr`。可选:
  - `cosine` —— 余弦衰减,最常用。
  - `linear` —— 线性。
  - `constant` —— 不衰减。
  - `WSD`(Warmup-Stable-Decay)—— 先 warmup、再长时间**恒定**、最后 `wsd_decay_steps` 步快速衰减;便于在"stable"段任意点取 checkpoint 续训。
- weight decay 也可独立调度。

它不在 `optimizer.step()` 内部，而由训练循环根据**全局一致的更新结果**决定是否推进。

### 8.2 `optimizer.step → success → scheduler.step` 完成链

`train_step` 先调用 `optimizer.step()`（`megatron/training/training.py:3298-3302`），再把各 model-parallel rank 的 `update_successful` 做逻辑与（`:3315-3328`）。只有所有相关 rank 都成功，才按 `get_num_microbatches() × micro_batch_size × data_parallel_size` 计算本轮实际消费的样本数，并调用 `opt_param_scheduler.step(increment=increment)`；overflow 或其它跳步则只记 `skipped_iter=1`，不推进调度（`:3342-3348`）。scheduler 内部随后累加 `num_steps`，逐 param group 写回 LR 与 weight decay（`megatron/core/optimizer_param_scheduler.py:294-310`）。

**被否掉的替代：把 scheduler 调用塞进 `optimizer.step()`。** 判据不是代码风格，而是更新原子性：某 rank 检出 overflow 时，wrapper 会返回 `False`；训练循环还必须先跨 MP 汇总成功状态。若各 optimizer 自己先推进 scheduler，失败 rank、成功 rank与参数更新状态会失配，也会让“跳过一次参数更新”仍然消耗 schedule。因而 scheduler 必须位于全局 success 判定之后。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — per-param-group 调度覆盖值的 resume 修复(#5213,`megatron/core/optimizer_param_scheduler.py:102/151/351`,新基线下三处行号均未变)
> `OptimizerParamScheduler` 支持 **per-param-group 覆盖**:某个 param group 可以带自己的 `max_lr`/`min_lr`/`start_wd`/`end_wd`(`_OPT_PARAM_SCHEDULER_OVERRIDE_KEYS`),它们在 `get_lr()`/`get_wd()` 中**优先于** scheduler 的类级值。两个 bug 被修:
> 1. **`override_opt_param_scheduler` 模式下 resume 丢失覆盖值**:checkpoint 里 param group 携带的 max_lr/min_lr 会覆盖当前 run 的命令行参数。修复:`__init__` 时用当前 run 的参数快照各 group 的覆盖值(`self._param_group_scheduler_overrides`),`load_state_dict` 里新增 `_restore_param_group_scheduler_overrides()` 在重放 schedule 前还原。
> 2. **`step(increment=num_steps)` 时机错误**:原来在还原 `start_wd`/`wd_incr_style` 等 WD 字段**之前**就调了 `self.step()`,导致 resume 后第一步用了旧 WD 状态。修复:把 `step(increment=num_steps)` 移到所有字段还原(含覆盖值还原)**之后**。

---

## 9. CPU Offloading 机制(补充,2026-07-31 · 由原 `16_megatron_distributed_optimizer_analysis.md` §4 并入)

### 9.1 HybridDeviceOptimizer

`megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py:14` — 将参数按比例拆分到 GPU 和 CPU:

- `offload_fraction`(默认 0.5):控制多少参数放在 CPU
- 双流 Overlap:`_d2h_stream` 传梯度到 CPU,`_h2d_stream` 传参数回 GPU
- 支持 `param_update_in_fp32`:CPU 上做 FP32 更新
- 通过 step hooks 自动化参数回拷

名字里的 "Hybrid":一部分参数的优化器状态/更新在 GPU、一部分在 CPU,按显存压力混合 —— 用 PCIe 带宽 + CPU 算力换 GPU 显存(类比激活 offload 的思路,见 `18_megatron_recompute_analysis.md` §1.2)。

### 9.2 ChunkedOptimizerStateOffloader：CPU canonical state，逐块回到 GPU 更新

当前冻结源码没有 `sync_before_step()`，也不是“step 后整块状态 `resize_(0)`、下步整块 reload”的旧生命周期。`ChunkedOptimizerStateOffloader` 把选中参数的 optimizer state 以 CPU 副本为 canonical，并把参数作为不可切分原子装入受 `chunk_size_bytes` 约束的 chunk；超大单参数允许独占超限 chunk，master weights 则在 step 前整窗恢复（`megatron/core/optimizer/cpu_offloading/chunked_optimizer_state_offload.py:57-87`）。当前 hop 是：

1. `prefetch_for_step()` 异步恢复全部所选 master 和第一个 state chunk；只需要 master 的延迟路径调用 `prefetch_master_for_step()`（`:785-798`）。训练入口把预取挂到 final-gradient 阶段以覆盖 H2D（`megatron/training/training.py:3064-3098`），`MixedPrecisionOptimizer.prepare_grads()` 对直接调用场景还提供幂等 fallback（`megatron/core/optimizer/optimizer.py:807-819`）。
2. `step()` 等 master H2D，先让常驻参数更新以覆盖首块预取，再逐 chunk 执行“等当前 H2D → 预取下一块 → `_step_subset` → 当前 state D2H”；首次懒建 state 时额外同步一次以守住峰值内存界限（`chunked_optimizer_state_offload.py:957-1006`）。
3. 到 optimizer→forward 生命周期边界，`offload_for_forward()` 把仍驻留的 state 与可选 master 排队 D2H，并释放 staging-slot 所有权（`:808-830`）；训练循环在 zero-grad/下一次 forward 前触发它，并为 MXFP8 参数 buffer 保留延迟 master-offload 分支（`megatron/training/training.py:3121-3186`），对外代理在 `megatron/core/optimizer/optimizer.py:228-249`。

**与 §9.1 的选择判据：**`optimizer_cpu_offload` 在 factory 里直接选择 `HybridDeviceOptimizer`，把一部分参数交给 CPU optimizer **计算更新**，另一部分留给 GPU optimizer（`megatron/core/optimizer/__init__.py:502-543`；`hybrid_optimizer.py:150-179`）。chunked state offload 则保留外部 optimizer 的 GPU 更新语义，只让选中 state/master 在非使用期以 CPU canonical 形式驻留，并逐块短暂回 GPU。前者适合明确要把部分计算也移到 CPU 的场景；后者适合主要目标是限制 GPU optimizer-state 峰值、仍希望沿用 GPU optimizer kernel 的场景。两者互斥，约束见 §11。

---

## 10. Layer-Wise 分布式优化器与 Muon 集成

（2026-07-31 由原 16 附录 A.7 与原 `megatron_optimizer_internals_analysis.md` §7 合并；前者讲分片布局，后者讲优化器实现，二者互补。）

### 10.1 ChainedOptimizer 分片布局整合(原 §A.7)

`Layer-Wise Distributed Optimizer`(`--layer-wise-distributed-optimizer`)将参数按**层**分配到 DP rank,而非按扁平的参数列表:

**解决的问题**:
- 支持**多个优化器组合**(如 Muon 处理 ≥2D 矩阵参数,AdamW 处理 vector/bias 参数),普通 distributed optimizer 难以优雅支持 per-parameter optimizer 切换
- 更细粒度的 all-gather overlap:可在计算第 L 层 forward 的同时,异步 all-gather 第 L+1 层的参数

**ChainedOptimizer 分配规则**:
- 通过 `param_group` 的 `optimizer_name` 或 `foreach` 映射规则路由不同参数到不同底层优化器
- 例如:所有 `weight` 矩阵参数(≥2D)→ `MuonOptimizer`,所有 `bias`、`norm`、`embedding` 参数 → `AdamWOptimizer`

**选择场景**:使用混合优化器(如 Muon + AdamW)或超大模型需要极致 per-layer overlap 时。

> [!deprecated] 2026-06-16:**触发方式更正**。不存在 `--layer-wise-distributed-optimizer` 这个 flag。Layer-wise 分布式优化器通过 **`--optimizer muon`(或其它 emerging 优化器)+ `--use-distributed-optimizer`** 触发:`megatron/training/arguments.py:1853-1866` 在 optimizer 非 `sgd`/`adam` 且开了 distributed optimizer 时,把 `use_layer_wise_distributed_optimizer` 置 True、并关掉普通 `use_distributed_optimizer`。`--optimizer dist_muon` 是旧写法,已弃用。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — LayerWise 与 DDP buffer 基建整合 + 非-Muon 参数改走真正的 DistributedOptimizer(#4509 / #4771,`megatron/core/optimizer/layer_wise_optimizer.py`、`megatron/core/optimizer/__init__.py:725-972` 的 `_get_megatron_emerging_optimizer`、`megatron/core/optimizer/distrib_optimizer.py:3223`)
>
> 这组 PR 实质性改写了 layer-wise 的实现,并**修正了上文"普通 distributed optimizer 难以优雅支持 per-parameter optimizer 切换"的暗示** —— 现在两者是**链式协作**,而非二选一:
>
> **① LayerWise 不再用独立 ping-pong 路径,而是建在 DDP 的 grad/param buffer 之上**(#4509)。它预计算一个 shard-aligned 的 `FullParamLayout`/`PerBufferParamLayout`(`megatron/core/optimizer/param_layout.py`),把参数按 backprop 顺序装进**对齐到 shard 边界**的 bucket,使任何参数都不跨 shard 边界,从而能直接复用 DDP 的 reduce-scatter(`use_distributed_optimizer=True` 时,见 [[16_megatron_distributed_optimizer_analysis]] §3)/ all-gather 通信与 `overlap_grad_reduce`/`overlap_param_gather` 重叠语义(见 [[16_megatron_distributed_optimizer_analysis]] §3.2/[[16_megatron_distributed_optimizer_analysis]] §3.3)。装箱算法在 #4771 中从"同尺寸配对(size-matching)"换成 **LPT 贪心装箱**(按 numel 降序塞进当前负载最小的 shard),在保证 bucket 连续 backprop 区间的同时让各 shard 尽量均衡。
>
> **② 非-Muon 参数改由独立的 `DistributedOptimizer` 按字节级分片管理**(#4771)。新增 `is_managed_by_layer_wise_optimizer(param)`(`megatron/core/optimizer/layer_wise_optimizer.py:43`):2D 矩阵权重且非 embedding/output → Muon/LayerWise 接管;embedding、bias、LayerNorm 等 → **路由到一个独立的 `DistributedOptimizer`**(真正的 ZeRO 字节级分片,见 [[16_megatron_distributed_optimizer_analysis]] §5 阶段②)。`BufferKey` 增加 `is_managed_by_layer_wise_optimizer` 维度(新基线上 `BufferKey` 已移出 `param_and_grad_buffer.py`,现为 `megatron/core/optimizer/param_layout.py:46`,该字段在 `:66`;`param_and_grad_buffer.py` 侧改为在 `group_params_for_buffers` 里 `from ..optimizer.param_layout import BufferKey` 导入,`megatron/core/distributed/param_and_grad_buffer.py:938-964`),让两类参数落进不同 buffer;`DistributedOptimizer.start_param_sync_for_bucket_group_subset()`(`megatron/core/optimizer/distrib_optimizer.py:3223`)只同步自己那批 bucket group,避免与 sibling LayerWise 重复 all-gather。最终 `LayerWiseDistributedOptimizer`(Muon)+ `DistributedOptimizer`(Adam)由 `ChainedOptimizer`(§2.1)串成一个。
>
> **结论(对上文 Muon/ZeRO 框架的修正)**:Muon 现在**可以与 ZeRO 分片共存**。Muon 管的矩阵权重经 LayerWise 走 shard-aligned 的 reduce-scatter/all-gather(等效 ZeRO-1/2 沿 DP 分片优化器状态与梯度),非-Muon 参数走标准 `DistributedOptimizer`。早期"Muon 对 ZeRO 切分的根本性挑战"指的是 Newton-Schulz 正交化需要**整块矩阵**、无法像 Adam 那样按字节随意切;LayerWise 的解法正是 **shard-aligned bucket + 按层/按整参数分配**,让每个矩阵整体落在某个 shard 内,从而既正交化又分片(跨框架的 Muon/ZeRO 张力综述见 [[32_distributed_optimizer_deepdive]] §六)。
>
> **限制**:此 split 路径要求 `use_layer_wise_param_layout=True`(默认开;`--no-use-layer-wise-param-layout` 回退到 legacy ping-pong)、`num_distributed_optimizer_instances == 1`、且不支持 expert-parallel 的非-Muon 参数组与 `overlap_param_gather_with_optimizer_step`(`megatron/core/optimizer/__init__.py:761` 断言)。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — MTP-stage word_embeddings 必须打 `is_embedding_or_output_parameter` 标签(#5034,`megatron/core/models/common/language_module/language_module.py:205-213`,新基线下行号未变)
> `is_embedding_or_output_parameter` 标签决定参数被 Muon/LayerWise 接管还是路由给 Adam/DistOpt(见上)。MTP(Multi-Token Prediction)阶段的 `word_embeddings.weight` 是 pre_process embedding 的**副本**(靠跨 stage all-reduce 同步),原来漏打此标签 → 被 LayerWise 当作 2D 矩阵接管、且因 `shared_embedding=True` 在 `_emit_bucket` 里把整个 `(vocab × hidden)` 张量**复制到全部 `dp_size` 个 shard**,使该 chunk 的 buffer 膨胀约 8×。修复:`pre_process` 或 `mtp_process` 任一为真就打标签,让 MTP embedding 正确归 Adam/DistOpt 管理。

### 10.2 Emerging optimizer 与 Muon 版本更新

| 优化器 | 文件 | 一句话 |
|--------|------|--------|
| **Muon** | `megatron/core/optimizer/muon.py` | 新型优化器,对矩阵参数用 Newton-Schulz 正交化更新方向;v0.16 引入,配 layer-wise 分布式优化器(§10.1) |
| **layer-wise 分布式优化器** | `megatron/core/optimizer/layer_wise_optimizer.py` | 按层组织优化器状态/通信,降低峰值显存(§10.1) |
| **CPU offload** | `megatron/core/optimizer/cpu_offloading/`(`HybridDeviceOptimizer`) | 把优化器状态与 step 计算放 CPU,`--optimizer-cpu-offload`,GPU 显存极紧时用,详见 §9 |
| emerging optimizers | `megatron/core/optimizer/emerging_optimizers.py` | 其他较新优化器 |

> [!deprecated] 2026-06-16:**Muon 的真正实现不在 `megatron/core/optimizer/muon.py`**。`megatron/core/optimizer/muon.py` 已是一个 28 行的 *backward-compatible shim*(`get_megatron_muon_optimizer` 在 `megatron/core/optimizer/muon.py:8`,仅转调 `get_megatron_optimizer`,且 `dist_muon` 已弃用;新基线 `71092579` 下该文件仍为 28 行)。Muon / AdaptiveMuon 的实际实现是 `megatron/core/optimizer/emerging_optimizers.py` 里的 `TensorParallelMuon`(`:160`)/ `TensorParallelAdaptiveMuon`(`:294`),经 `_EMERGING_OPTIMIZERS` 注册表(声明 `megatron/core/optimizer/emerging_optimizers.py:152`、填充 `:429`)接入,并依赖外部包 `emerging-optimizers`。注:此 shim 在 `ee3f1ff` 已存在,原表项的文件归属一直是错的。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — emerging optimizers / Muon 一组更新
> **① 升级到 v0.3.0**(#5320,`pyproject.toml`、`megatron/core/optimizer/emerging_optimizers.py`):外部 `emerging-optimizers` 包由 v0.2.0 → **v0.3.0**;`TensorParallelAdaptiveMuon` 新增暴露 `scale_mode` / `extra_scale_factor`;`OptimizerConfig` 删除 `soap_precondition_frequency` 字段。注册表当前内建 `muon`、`adaptive_muon`(本地 TP 版),并自动收编上游包注册的其它优化器(如 SOAP)。
> **② 触发方式**:emerging 优化器通过 `--optimizer muon`(或 `adaptive_muon`/`soap` 等,即非 `sgd`/`adam`)选择;若同时 `--use-distributed-optimizer`,会自动转成 **layer-wise distributed optimizer**(`megatron/training/arguments.py:1853-1866`,`use_layer_wise_distributed_optimizer=True`)。`--optimizer dist_muon` 已弃用。emerging 优化器目前**不支持** Torch-FSDP2 / Megatron-FSDP(`megatron/training/arguments.py:1879-1882` 断言)。
> **③ Muon 参数路由(关键)**:默认 override 规则把 **非线性/embedding/output 参数路由给 Adam**(`_is_nonlinear_or_embedding`,定义在 `megatron/core/optimizer/emerging_optimizers.py:128-130`、作为默认 override 注册在 `:81-87`),Muon 只接管 2D 矩阵权重。配合 #4509/#4771,Muon 矩阵权重走 `LayerWiseDistributedOptimizer`、其余 Adam 参数走独立 `DistributedOptimizer`,二者由 `ChainedOptimizer` 串起(详见 §10.1 的 2026-06-16 更新)。
> **④ Muon QKV split 支持 gated attention**(#4728,`megatron/core/optimizer/emerging_optimizers.py:133` `_get_qkv_split_shapes`、`megatron/core/optimizer/__init__.py:779-797`):Muon 对 fused `linear_qkv.weight` 需按 Q/K/V 分块各自做 Newton-Schulz 正交化。当 `attention_output_gate=True`(门控注意力)时,QKV 切分形状由 3 段变为 **4 段** `[q, q_gate, k, v]`;并改为**逐参数**携带 `param.qkv_split_shapes`,且对 `shape[0] % sum(splits) != 0` 的参数跳过 QKV 标记(避免误切)。
> **⑤ QK-Clip**:`megatron/core/optimizer/qk_clip.py:31`(`clip_qk`)对注意力 QK logits 做裁剪以稳住数值,是 Muon 训练注意力稳定性的配套件(该文件在 `ee3f1ff` 已存在,此前表中漏列)。

---

## 11. 约束

本页只列 optimizer-step、offload、precision-aware 与 emerging-optimizer 的边界；DDP/ZeRO/FSDP 约束见 [[16_megatron_distributed_optimizer_analysis]] §12。

| # | 前提 / 不变量 | 源码落点 | 破坏后的表现 |
|---|---|---|---|
| 6 | `overlap_param_gather_with_optimizer_step` 与 `reuse_grad_buf_for_mxfp8_param_ag`（[[16_megatron_distributed_optimizer_analysis]] §3.10）互斥 | `megatron/core/optimizer/optimizer_config.py:513-517`,抛 `ValueError` | 构造期直接失败 —— 共享 buffer 一旦复用,参数 AG 就不能再提前塞进 step |
| 7 | 精度感知优化器（[[16_megatron_distributed_optimizer_analysis]] §3.11 / 本页 §4.3）只支持 `adam`,且必须同时开 distributed optimizer | `megatron/core/optimizer/optimizer_config.py:519-525` | assert 失败 |
| 8 | chunked optimizer-state offload(§9)限 Adam/Muon、与 `optimizer_cpu_offload` 互斥、且不支持 optimizer CUDA graph | `megatron/core/optimizer/optimizer_config.py:445-448`、`:450-452`、`:453-455` | assert 失败 —— §9.1 的 `HybridDeviceOptimizer` 与 chunked offload 是两条不能同开的路 |
| 9 | emerging 优化器(Muon 等,§10)不能配 `overlap_param_gather_with_optimizer_step`,也不支持 fp16 | `megatron/core/optimizer/__init__.py:761-766`、`:769-770` | 前者 assert 失败,断言文本自己给了理由 ——「the emerging-optimizer path does not split model_chunks into (first, rest) groups, so the per-chunk param-gather dispatch never fires」;后者 `ValueError` |

- **optimizer 不负责构造进程组。** `get_megatron_optimizer` 只在 `pg_collection.tp` 缺失时回退全局 getter，并用 `setattr` 挂到实例；源码 TODO 要求以后把 TP group 直接贯穿 constructors（`megatron/core/optimizer/__init__.py:688-693`）。组的构造归 [[17_megatron_parallelism_orchestration_analysis]]。
- **μP 必须两端同时启用。** 模型侧的初始化/attention/output 缩放与 optimizer 侧的 param-group overrides 是同一合同；`width_mult=1` 只会让 optimizer override 为空，不会替代配置合法性检查。

---

## 12. 发展趋势

> [!note] 推断：以下判断基于冻结基线中的弃用标记与 TODO，不是源码给出的时间表。

**一、CPU state offload 已完成从旧整块接口到分块执行器的迁移。**
§9.2 展开的 `ChunkedOptimizerStateOffloader` 已是当前机制；`megatron/core/optimizer/cpu_offloading/` 在基线下只保留 `__init__.py`、`README.md`、`hybrid_optimizer.py` 与 `chunked_optimizer_state_offload.py`，后者由提交 `9050d4c5f`（commit message「[dev] Add chunked optimizer-state and master-weight offload (#6244)」）引入，并被 `MegatronOptimizer` 直接持有（`megatron/core/optimizer/optimizer.py:52,149,172-207`）。配置侧 `offload_optimizer_states` 只是 `chunked_optimizer_state_offload` 的 deprecated alias（`megatron/core/optimizer/optimizer_config.py:385-386`），`__post_init__` 会发 `FutureWarning` 并改写为新开关（`:418-429`）。**由此可推断**：后续调优的有效旋钮是 chunk 大小、offload fraction 与预取重叠，而不是已经不存在的旧类或整块 reload 生命周期。

**二、参数 layout 正在从 DDP buffer 里独立出去,成为多个优化器共享的第三方描述。**
§10.1 已经记录:`BufferKey` 不再定义在 `param_and_grad_buffer.py`,而是搬去 `megatron/core/optimizer/param_layout.py:46`,再由 `group_params_for_buffers` 反向导入(`megatron/core/distributed/param_and_grad_buffer.py:938-964`);[[16_megatron_distributed_optimizer_analysis]] §3.7 那条更正里的 bucket 末端对齐 divisor 同样集中到了 `megatron/core/optimizer/param_layout.py:29`。**由此可推断**:"谁决定参数怎么装桶"正在从 DDP 侧移到优化器侧 —— 因为 LayerWise(Muon)与 `DistributedOptimizer` 必须对同一份 layout 达成一致(§10.1);后续读分桶代码,应先看 `param_layout.py` 再看 `param_and_grad_buffer.py`。

**三、进程组正在从全局单例改成显式传入,而优化器这一层还没走完。**
`get_megatron_optimizer` 仍在 `pg_collection` 缺 `tp` 时回落到 `parallel_state.get_tensor_model_parallel_group()`,并用 `setattr` 把 tp_group 挂到优化器实例上,旁边写着「TODO(M4): plumb tp_group through optimizer constructors so this setattr disappears」(`megatron/core/optimizer/__init__.py:688-692`);工厂入口另有一条「TODO: the standard and emerging optimizer paths handle pg_collection differently; unify them so both use a single pg_collection-based flow」(`:1023-1024`)。**由此可推断**:§2 的类层次短期内不会变,但"优化器从哪里拿 DP/TP 组"会变;跨版本对照优化器代码时,通信组的来源是最容易漂移的一处(编排侧的同向变化见 [[17_megatron_parallelism_orchestration_analysis]])。

---

## 13. 小结

- optimizer factory 先确定 param-group overrides，再选择 standard/emerging 与 mixed-precision/distributed wrapper。
- mixed-precision step 的硬顺序是 `prepare_grads → overflow gate → clip/count → base step → master copy-back`；发现非有限值时整步跳过。
- scheduler 在 step 外按 param group 推进 LR 与 weight decay，checkpoint/override 规则决定恢复时谁覆盖谁。
- μP 同时改写模型初始化/attention/output scale 与 optimizer param-group LR/eps；Muon、SGD、decoupled LR 有各自例外。
- CPU offload 与 LayerWise/Muon 继续复用同一 step 接口，但受互斥开关、参数布局与进程组来源约束。

---

## 配置契约：`SchedulerConfig`

本页 §8 讲 LR/WD 调度的**机制**（`OptimizerParamScheduler` 的 warmup、decay、param-group override 合并）。本节给它的**配置面**。

`SchedulerConfig` 经 [[41_megatron_config_surface_analysis]] §2 的工厂转成 CLI（`megatron/training/arguments.py:3897`），且带一个 `exclude=["no_weight_decay_cond_type"]`——**那个字段被刻意排除在 CLI 之外**，因为它需要传一个条件函数而非标量，属于 §2.4 说的「dataclass 字段 ≠ 用户可配 flag」那类人工划线。**下表直接取自 `megatron/training/config/training_config.py` 的 `SchedulerConfig` 类体**。


### `SchedulerConfig`（`megatron/training/config/training_config.py`，14 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `lr_wsd_decay_style` | `Literal['exponential', 'linear', 'cosine', 'minus_sqrt']` | `'exponential'` | Decay style for the annealing phase of WSD | `:173` |
| `lr_decay_iters` | `int \| None` | `None` | number of iterations to decay learning rate over, If None defaults to train iters | `:176` |
| `lr_decay_samples` | `int \| None` | `None` | number of samples to decay learning rate over, If None defaults to train samples | `:179` |
| `lr_wsd_decay_iters` | `int \| None` | `None` | number of iterations for the annealing phase in the wsd schedule | `:182` |
| `lr_wsd_decay_samples` | `int \| None` | `None` | number of samples for the annealing phase in the wsd schedule | `:185` |
| `lr_warmup_fraction` | `float \| None` | `None` | fraction of lr-warmup-(iters/samples) to use for warmup (as a float) | `:188` |
| `lr_warmup_iters` | `int` | `0` | number of iterations to linearly warmup learning rate over. | `:191` |
| `lr_warmup_samples` | `int` | `0` | number of samples to linearly warmup learning rate over. | `:194` |
| `lr_decay_steps` | `int \| None` | `field(init=False, default=None)` | number of samples to decay learning rate over. Calculated at runtime from lr_decay_iters or lr_decay_samples. | `:200` |
| `use_checkpoint_opt_param_scheduler` | `bool` | `field(default=False, metadata={'argpa…` | Use checkpoint to set the values of the scheduler (learning rate, warmup iterations, minimum learning rate, maximum number of iterations, and decay style) fr… | `:222` |
| `start_weight_decay` | `float \| None` | `None` | Initial weight decay coefficient for L2 regularization. | `:239` |
| `end_weight_decay` | `float \| None` | `None` | End of run weight decay coefficient for L2 regularization. | `:242` |
| `weight_decay_incr_style` | `Literal['constant', 'linear', 'cosine']` | `'constant'` | Weight decay increment function. | `:245` |
| `wd_incr_steps` | `int \| None` | `field(init=False, default=None)` | Number of samples to increment weight decay over. Calculated at runtime. | `:254` |

> 该类共 20 个字段，本表收 14 项；其余 6 项已在别处归属：`lr_decay_style`、`lr_warmup_init`、`lr_warmup_steps`、`override_opt_param_scheduler`、`no_weight_decay_cond_type`、`wsd_decay_steps` → 本页 §8。

---

## 配置契约：μP（Maximal Update Parameterization）

**这是本知识库此前完全没有覆盖过的一块。** `megatron/core/transformer/transformer_config.py` 有一个专门的 `# MuP (Maximal Update Parameterization)` 段，`megatron/core/optimizer/__init__.py` 有配套的 `get_mup_config_overrides`（与 `get_standard_config_overrides` 并列，由 `check_config_overrides_consistency` 校验）。在 2026-09-02 配置面对账前，本域页面**无一字提及**——它是 [[40_megatron_feature_tree_analysis]] §3.2 点名的那个真实盲区。

μP 之所以落在本页而非模型结构页：它改变的不是模型**结构**，而是**各参数组的学习率与初始化缩放**——作用点在 optimizer 的 param group 组织上，正是本页 §2-§8 的领域。其目标是让超参（尤其学习率）在模型宽度变化时**可迁移**：小模型上调好的 lr 直接用在大模型上。

**下表直接取自类体**。`mup_base_hidden_size` / `mup_base_head_dim` 是「调参时那个小模型」的尺寸，`mup_width_mult` 由当前尺寸与基准尺寸之比推出，其余三项是各处的缩放指数与乘子。




### `TransformerConfig`（`megatron/core/transformer/transformer_config.py`，7 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `use_mup` | `bool` | `False` | Enable Maximal Update Parameterization (MuP) for hyperparameter transfer across model widths. When enabled, learning rates and initialization are scaled acco… | `:499` |
| `mup_width_mult` | `float` | `1.0` | Width multiplier for MuP scaling, computed as hidden_size / mup_base_hidden_size. This value is automatically computed in __post_init__ when use_mup is enabled. | `:506` |
| `mup_base_hidden_size` | `Optional[int]` | `None` | Base hidden size for MuP width scaling. This is the reference width from which scaling factors are computed. Defaults to hidden_size if not specified (base m… | `:512` |
| `mup_embedding_mult` | `float` | `1.0` | Multiplier for embedding layer output. Applied after the embedding lookup. Default: 1.0 (no scaling). | `:520` |
| `mup_output_mult` | `float` | `1.0` | Multiplier for output logits before softmax. When MuP is enabled and this is left at 1.0, it is auto-set to 1/mup_width_mult to keep output variance stable a… | `:526` |
| `mup_base_head_dim` | `Optional[float]` | `None` | Base head dimension for MuP attention scaling. When set, softmax_scale = sqrt(mup_base_head_dim) / (kv_channels ** mup_attn_scale_power). Set to base model's… | `:534` |
| `mup_attn_scale_power` | `float` | `1.0` | Power for attention scaling: softmax_scale = 1 / (kv_channels ** mup_attn_scale_power). 0.5 = standard attention (1/sqrt(d_head)), 1.0 = MuP attention (1/d_h… | `:542` |

> 该类共 266 个字段，本表收 7 项；其余 259 项已在别处归属：主要归 [[10_megatron_model_structure_analysis]] 92 项、[[14_megatron_ep_analysis]] 38 项、[[23_megatron_precision_cudagraph_fusion_analysis]] 38 项、[[21_megatron_fusion_operators_analysis]] 26 项，另散见 20 页（完整归属见 `docs/coverage/megatron-lm.yaml`）。

> [!done] 机制闭环（2026-09-03）
> 本页 §3 已补齐 `TransformerConfig.__post_init__ → setup_model_and_optimizer → get_mup_config_overrides → param groups → runtime embedding/logit scaling`；关键 locator 为 `megatron/core/transformer/transformer_config.py:2893-2969`、`megatron/training/training.py:2799-2824` 与 `megatron/core/optimizer/__init__.py:131-297`。

## Related Pages

- [[16_megatron_distributed_optimizer_analysis]] —— 解释本页 optimizer state/gradient/parameter 在 DP 上怎样分片与同步。
- [[23_megatron_precision_cudagraph_fusion_analysis]] —— 拥有 bf16/fp16/FP8 参数精度、recipe 与 CUDA Graph 配置。
- [[28_megatron_training_stability_observability_analysis]] —— 从稳定性和可观测性解释 overflow、grad norm 与异常定位。
- [[11_muon_analysis]] —— 解释 Muon 的 Newton–Schulz 数学；本页只拥有 Megatron 集成。
- [[19_megatron_dist_checkpointing_analysis]] —— 解释 optimizer/scheduler state 的持久化与恢复。
- [[17_megatron_parallelism_orchestration_analysis]] —— 提供 optimizer 消费的 TP/DP/expert 进程组。
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]] —— 返回本域索引。
