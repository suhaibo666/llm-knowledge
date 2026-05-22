# Megatron-LM 优化器内部深度解析(Optimizer Internals)

> 代码基准:`Megatron-LM/` 子仓库 `dev` 分支,commit `ee3f1ff`
> 核心文件:`megatron/core/optimizer/optimizer.py`(1516 行)、`grad_scaler.py`、`clip_grads.py`、`optimizer/__init__.py`、`optimizer_param_scheduler.py`
> 配套阅读:`ddp_optimizer_analysis.md`(分布式优化器 / ZeRO 那一层)
> 定位:"第二层补遗"第②份。`ddp_optimizer_analysis.md` 讲的是"优化器状态怎么沿 DP 切分(ZeRO)";本文讲**单个优化器实例内部**怎么运作 —— 混合精度、loss scaling、梯度裁剪、LR 调度。

---

## 0. 总览

一个训练步的尾部:反向算完梯度 → `finalize_model_grads`(DP/PP/SP 规约,见各并行文档)→ **`optimizer.step()`**。本文拆开这个 `step()`。

它要解决的不只是"调 Adam":bf16/fp16 训练有**数值精度**问题(梯度下溢、累加误差),需要 fp32 master 副本 + loss scaling;还要做**梯度裁剪**稳定训练、**LR/WD 调度**。这些都封装在 Megatron 的优化器类里。

`ddp_optimizer_analysis.md` 的 `DistributedOptimizer`(ZeRO-1)是本文 `MixedPrecisionOptimizer` 的**子类** —— 分布式分片是"在哪算",本文是"算什么"。

---

## 1. 优化器类层次

`optimizer.py` 的继承链:

```
MegatronOptimizer (ABC, :100)              抽象基类:clip_grad_norm / get_loss_scale / scale_loss / step
   │
   ├── MixedPrecisionOptimizer (:465)      混合精度:fp32 master 副本 + grad scaler
   │      ├── Float16OptimizerWithFloat16Params (:654)   fp16/bf16 模型参数的具体实现
   │      └── DistributedOptimizer          ← ZeRO-1,见 ddp_optimizer_analysis.md
   │
   ├── FP32Optimizer (:918)                纯 fp32,无 scaling、无 master 副本
   │
   └── ChainedOptimizer (:1104)            把多个优化器串成一个(见 §1.1)
```

`get_megatron_optimizer`(`optimizer/__init__.py`)是工厂:按配置(`bf16`/`fp16`/`fp32`、是否分布式、是否 MoE)挑类、切 param group、装配。

### 1.1 `ChainedOptimizer` 为什么需要

一个模型常需要**多个优化器实例**:
- **稠密参数 vs 专家参数**:MoE 模型里专家参数走 EP 组、稠密参数走普通 DP 组,分片域不同 → 各用一个 `DistributedOptimizer`。
- `num_distributed_optimizer_instances > 1`(HSDP)。

`ChainedOptimizer` 把它们包成一个对外统一的优化器:`step()` 时依次驱动每个子优化器,`get_loss_scale` / `clip_grad` 跨子优化器协调。

### 1.2 param group:weight decay 的区分

`get_megatron_optimizer` 建 param group 时把参数分两组:**该用 weight decay 的**(线性层权重)和**不该用的**(bias、LayerNorm 的 `weight`/`bias`)。后者 `weight_decay=0`。这是标准做法,避免对归一化/偏置施加权重衰减。

---

## 2. 混合精度优化器:fp32 master 副本

### 2.1 动机

模型用 bf16/fp16 做前向反向(省显存、快)。但**优化器更新若也用 bf16**:`param += lr · update`,当 `update` 比 `param` 小几个数量级时,bf16 的尾数位不够,加法**直接丢失** → 训练停滞。

**解法:fp32 master 副本**。优化器维护一份 fp32 的参数主拷贝,所有 Adam 更新在 fp32 上做;每步结束把 fp32 master **拷回** bf16 模型参数供下一步前向用。

### 2.2 这就是"18 bytes/param"的来源

`MixedPrecisionOptimizer` 持有的东西,正是 `ddp_optimizer_analysis.md` ZeRO 显存表里的 `18Ψ`:

| 张量 | 精度 | bytes/param | 谁持有 |
|------|------|-------------|--------|
| 模型权重 | bf16 | 2 | 模型 |
| 模型梯度 | **fp32** | **4** | DDP grad buffer(bf16 训练强制 fp32 累加) |
| **master 权重** | fp32 | 4 | **优化器** |
| **Adam 动量 m** | fp32 | 4 | **优化器** |
| **Adam 方差 v** | fp32 | 4 | **优化器** |
| | | **合计 18** | |

> 梯度为 **fp32(4 字节)** 而非 bf16(2 字节):bf16 尾数仅 7 位,跨 microbatch 累加会丢精度,Megatron 对 bf16 训练强制 fp32 梯度累积(`arguments.py:1296-1310`)。仅 `--grad-reduce-in-bf16` 时梯度为 2 字节、合计 16。

`ddp_optimizer_analysis.md` 说 ZeRO-1 把"优化器状态 `12Ψ`"切成 `1/dp` —— 切的就是这里的 master + m + v。

`FP32Optimizer` 则相反:模型本身就是 fp32,无需 master 副本、无需 scaler。

---

## 3. `optimizer.step()` 流程

`MixedPrecisionOptimizer.step()`(`optimizer.py:621`):

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
② clip_grad_norm    跨 TP/PP/DP 算全局梯度范数,超阈值则等比缩放(§5)
        │
③ count_zeros       (可选)统计零梯度数,日志用
        │
④ step_with_ready_grads
        │           base optimizer(FusedAdam)在 fp32 master 上做 Adam 更新
        │           fp32 master ──拷回──► bf16 模型参数
        ▼
   返回 (success, grad_norm, num_zeros)
```

关键:**inf/nan 检查在最前面**。一旦发现非有限梯度,整步丢弃(参数不动),交给 dynamic scaler 调整(§4)。这是 fp16 训练能稳住的安全阀。

---

## 4. Loss Scaling 与 GradScaler

### 4.1 动机:fp16 的下溢

fp16 动态范围窄(最小正规数 ~6e-5)。反向里很多梯度比这还小 → **下溢成 0** → 参数收不到更新。

**Loss scaling**:前向后把 loss 乘一个大数 `S`(`scale_loss`),反向链式法则使所有梯度同样 ×`S`,把小梯度抬进 fp16 可表示区间;`prepare_grads` 里再 ÷`S` 还原(unscale)。

> bf16 动态范围与 fp32 几乎一样宽,通常**不需要** scaler(或用 `ConstantGradScaler(1.0)`)。loss scaling 主要为 fp16。

### 4.2 两种 scaler(`grad_scaler.py`)

**`ConstantGradScaler`**:固定 `S`。简单,适合 bf16 或已知稳定的场景。

**`DynamicGradScaler`(`:64`)**:自适应。
- 连续 `growth_interval` 步无 inf/nan → `S ×= growth_factor`(往上试探,尽量大)。
- 连续 `hysteresis` 步检测到 inf/nan → `S ×= backoff_factor`(`<1`,减半之类)、且这些步**跳过更新**。
- `min_scale` 兜底。

直觉:`S` 越大越不下溢,但太大会上溢成 inf。dynamic scaler 在"尽量大"和"不溢出"之间自动平衡 —— 不断试着调大,溢出了就回退。

---

## 5. 梯度裁剪(`clip_grads.py`)

`clip_grad_norm(clip_grad)`:计算**全局梯度范数** `‖g‖`(所有参数拼起来的 L2 范数),若 `‖g‖ > clip_grad`(默认 1.0),把所有梯度等比缩放到 `clip_grad`:`g ← g · clip_grad / ‖g‖`。

并行要点:参数被 TP/PP 切散在多卡,所以"全局范数"要**跨 TP×PP 组 all-reduce** 各分片的范数平方和再开根。`MegatronOptimizer.clip_grad_norm`(`:220`)负责协调这次跨并行的范数规约。作用:挡住偶发的梯度尖峰,稳定训练。

---

## 6. LR / WD 调度(`OptimizerParamScheduler`)

`optimizer_param_scheduler.py:100`。每步更新 base optimizer 各 param group 的 **learning rate** 和 **weight decay**。

典型曲线 = **warmup + decay**:
- **warmup**:前 `lr_warmup_steps` 步,LR 从 0(或 `lr_warmup_init`)线性升到峰值 —— 避免初期大步长震荡。
- **decay**:之后按 `lr_decay_style` 衰减到 `min_lr`。可选:
  - `cosine` —— 余弦衰减,最常用。
  - `linear` —— 线性。
  - `constant` —— 不衰减。
  - `WSD`(Warmup-Stable-Decay)—— 先 warmup、再长时间**恒定**、最后 `wsd_decay_steps` 步快速衰减;便于在"stable"段任意点取 checkpoint 续训。
- weight decay 也可独立调度。

它不在 `optimizer.step()` 内部,而是训练循环每步单独调一次。

---

## 7. 其他优化器

| 优化器 | 文件 | 一句话 |
|--------|------|--------|
| **Muon** | `optimizer/muon.py` | 新型优化器,对矩阵参数用 Newton-Schulz 正交化更新方向;v0.16 引入,配 layer-wise 分布式优化器 |
| **layer-wise 分布式优化器** | `optimizer/layer_wise_optimizer.py` | 按层组织优化器状态/通信,降低峰值显存 |
| **CPU offload** | `optimizer/cpu_offloading/`(`HybridDeviceOptimizer`) | 把优化器状态与 step 计算放 CPU,`--optimizer-cpu-offload`,GPU 显存极紧时用 |
| emerging optimizers | `optimizer/emerging_optimizers.py` | 其他较新优化器 |

`HybridDeviceOptimizer` 名字里的 "Hybrid":一部分参数的优化器状态/更新在 GPU、一部分在 CPU,按显存压力混合 —— 用 PCIe 带宽 + CPU 算力换 GPU 显存(类比激活 offload 的思路,见 `recompute_analysis.md` §0.2)。

---

## 8. 小结

- **优化器类层次**:`MegatronOptimizer` → `MixedPrecisionOptimizer`(fp32 master)→ `Float16OptimizerWithFloat16Params` / `FP32Optimizer` / `DistributedOptimizer`;多实例用 `ChainedOptimizer` 串。
- **混合精度核心**:模型 bf16、优化器持 fp32 master + m + v —— 这就是 ZeRO 显存表 `18 bytes/param` 的来源。
- **`step()` 五步**:prepare_grads(unscale + 查 inf/nan)→ 裁剪 → count_zeros → Adam 更新 + master 回拷;**inf/nan 即跳步**是 fp16 训练的安全阀。
- **Loss scaling**:把小梯度抬出 fp16 下溢区;`DynamicGradScaler` 自适应"尽量大又不溢出"。
- **梯度裁剪**:全局范数裁剪,范数需跨 TP×PP all-reduce。
- **LR/WD 调度**:warmup + decay(cosine / WSD …),`OptimizerParamScheduler` 每步更新。
- 与 `ddp_optimizer_analysis.md` 的关系:那份讲"优化器状态怎么沿 DP 分片(ZeRO)",本文讲"单个优化器实例内部算什么"。

---

*生成依据:`Megatron-LM` `dev` 分支 `ee3f1ff`。源码行号以该 commit 为准。本文是"第二层补遗"3 份之②(优化器内部),后续:③ FP8 精度 + CUDA Graph + 算子融合。*

## Related Pages

- [[ddp_optimizer_analysis]] · [[precision_cudagraph_fusion_analysis]]
- [[megatron_distributed_optimizer_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
