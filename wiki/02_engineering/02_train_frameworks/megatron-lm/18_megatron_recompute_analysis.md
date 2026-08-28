---
title: "Megatron-LM 激活重计算(Activation Recomputation / Checkpointing)深度解析"
---

# Megatron-LM 激活重计算(Activation Recomputation / Checkpointing)深度解析

> **源码基线**：`NVIDIA/Megatron-LM@71092579522a12522d9f323ae180c9825d01928a`（`dev`，2026-08-27）
> **重定基线**：2026-08-28 由 `ee3f1ffa…`（2026-05-19）推进，跨 578 个提交；本页全部 `path:line` 已在新基线下逐条重核。
> 核心文件:`megatron/core/transformer/transformer_block.py`(`_checkpointed_forward`)、`megatron/core/recompute.py`(`checkpointed_forward`,整层重计算的共享实现)、`megatron/core/tensor_parallel/random.py`(`checkpoint`)、`megatron/core/transformer/transformer_config.py`(`recompute_*` 配置)
> 配套阅读:`22_megatron_memory_optimization_analysis.md` §2.3(激活换出 offloading)、五份并行文档
> 定位:"第二层补遗"第①份。激活重计算是与并行轴正交的**省显存**手段。

---

## 0. 总览

### 0.1 重计算是什么

反向传播需要前向算出的**中间激活**。默认情况下,前向把所有中间激活都留在显存里等反向用 —— 这是训练显存的最大头。**激活重计算(recomputation / checkpointing)**:前向时**只保留少量"检查点"张量、丢弃其余激活**;反向需要某段激活时,**从最近的检查点重新跑一遍前向**把它算出来。

**用算力换显存**:多付一遍(部分)前向计算,换回大量激活显存。

### 0.2 与"激活换出"的关系

| 手段 | 换什么 | 文档 |
|------|--------|------|
| **重计算 recompute** | 用**算力**换显存(反向多跑一遍前向) | 本文 |
| **换出 offload** | 用 **PCIe 带宽**换显存(激活搬 CPU) | `22_megatron_memory_optimization_analysis.md` §2.3 |

二者正交,可叠加。算力有余量、显存紧 → recompute;PCIe 有余量 → offload。

### 0.3 两个维度:粒度 × 方法

```
recompute_granularity ─┬─ "full"      整层重计算 ──┬─ recompute_method "uniform"
                       │                           └─ recompute_method "block"
                       └─ "selective" 按子模块重计算 ── recompute_modules=[core_attn, moe_act, ...]
```

---

## 1. 动机:激活显存墙

一个 transformer 层前向产生大量中间激活:QKV、attention 分数矩阵 `[s,s]`、softmax 输出、FFN 中间 `[s,b,H]`……总量随**层数 `L`** 线性、随**序列 `s`** 线性(attention 分数甚至 `O(s²)`)。

并行轴只能部分缓解:TP 切 `1/t`(配 SP)、CP 切 `1/cp`、PP 用 1F1B 把在世激活压到 `O(p)`(见 `15_megatron_pp_schedulers_analysis.md`)。但当这些都用上仍然 OOM,就需要**直接丢弃激活、反向重算** —— 这是与并行正交的最后一道省显存手段。

**代价量化**:反向计算量 ≈ 2× 前向。不重计算时一步 = 前向 1 + 反向 2 = 3 个单位;**整层重计算**时反向里要重跑一遍前向 → 1 + (2+1) = 4 个单位 ≈ **+33% 计算**。所以重计算不是免费的,要按需用。

---

## 2. 全量重计算(`recompute_granularity='full'`)

整个 transformer 层作为重计算单元:前向只存层的**输入** `[s,b,h]`,层内所有激活全丢;反向重跑整层前向。由 `_checkpointed_forward`(`megatron/core/transformer/transformer_block.py:581`)实现,`recompute_method` 选两种切法。

### 2.1 `recompute_method='uniform'`

把本 PP stage 的层均匀分成若干**块**,每块 `recompute_num_layers` 层,对每块的**输入**做一个检查点:

```python
# megatron/core/transformer/transformer_block.py:702
while layer_idx < self.num_layers_per_pipeline_rank:
    chunk_end = min(layer_idx + recompute_num_layers, num_layers_per_pipeline_rank)
    hidden_states, context = checkpoint_handler(custom(layer_idx, chunk_end))   # 整块作一个检查点
    layer_idx += recompute_num_layers
```

特点:**所有层都重计算**,检查点数 = `层数 / recompute_num_layers`。`recompute_num_layers` 越大,检查点越少、省得越多,但反向重算的粒度越粗。极端 `recompute_num_layers = 全部层` → 整个 stage 只留一个输入检查点,省到极致。

### 2.2 `recompute_method='block'`

只对**前 `recompute_num_layers` 层**做检查点(重计算),其余层正常跑(留全部激活):

```python
# megatron/core/transformer/transformer_block.py:724
for layer_idx in range(num_layers_per_pipeline_rank):
    if recompute_skip_num_layers <= layer_idx < recompute_num_layers + recompute_skip_num_layers:
        hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))  # 这些层重计算
    else:
        hidden_states, context = custom(layer_idx, layer_idx + 1)(...)                 # 这些层不重计算
```

特点:**部分层重计算**。动机 —— 配合 1F1B,越靠前的 PP stage 在世 microbatch 越多(`15_megatron_pp_schedulers_analysis.md` §②.4),激活压力越大;`block` 让你**只把恰好装不下的那几层重计算**,其余层省下重算开销。"fully use the device memory removing redundant re-computation"(源码注释)。

### 2.3 `uniform` vs `block`

| | uniform | block |
|--|---------|-------|
| 重计算范围 | 所有层 | 前 `recompute_num_layers` 层 |
| 省显存 | 多 | 可调(按需) |
| 多余重算 | 可能有(全算) | 少(只算装不下的) |
| 适合 | 显存极紧、要全省 | 显存差一点点、精打细算 |

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> **全量重计算逻辑抽成共享实现，并支持 HybridModel（#4496，118933a85）**
> - 新增 `megatron/core/recompute.py`，把整层重计算的 `checkpointed_forward`（含 `custom` / `chunk_runner` / `uniform`+`block` 两种切法）抽成**可复用函数**（`megatron/core/recompute.py:21`；`custom` 在 `:54`、`chunk_runner` 在 `:116`）；Hybrid（Mamba/GDN + attention 混合）模型的 `megatron/core/models/hybrid/hybrid_block.py` 现也走同一套：`recompute_granularity == 'full' and self.training` 时调 `checkpointed_forward(...)`（`megatron/core/models/hybrid/hybrid_block.py:1009-1010`，import 在 `:28`）。此前 full 重计算只在 GPT `transformer_block` 内可用，现 HybridModel 同样支持。
> - **GPT 路径行号已在新基线下重核**（机制仍有效）：`megatron/core/transformer/transformer_block.py` 的 `_checkpointed_forward` 在 `:581`、`checkpoint_handler` 在 `:674`、`uniform` 分支在 `:702`、`block` 分支在 `:724`（旧基线 `ee3f1ffa…` 依次为 `:464` / `:542` / `:570` / `:592`）。GPT 的 `transformer_block._checkpointed_forward` 仍保留自有实现（与 `megatron/core/recompute.py` 的共享版并存），故 §2 的描述对 GPT 依然成立。

---

## 3. 选择性重计算(`recompute_granularity='selective'`)

### 3.1 动机:不是所有激活都"值得"重计算

各类激活的"显存占用 / 重算代价"比差异巨大:
- attention 分数矩阵 `[s,s]` —— 显存巨大(`O(s²)`),但 softmax/dropout 重算很**便宜**。
- 大矩阵乘的输入 —— 显存中等,但重算要再做一次 **GEMM**,贵。

**选择性重计算**只挑"显存大、重算便宜"的子模块下手 —— 这是 Megatron 2022 论文(Korthikanti et al.)的核心洞察:**选择性重计算注意力,能省掉大部分激活显存,而计算开销仅 ~1-2%**(远低于全量重计算的 +33%)。

### 3.2 `recompute_modules`

`recompute_modules` 列表选要重计算的子模块(字段定义 `megatron/core/transformer/transformer_config.py:629-633`;合法取值集合 `megatron/core/transformer/transformer_config.py:2315-2326`),可选:

| 子模块 | 说明 |
|--------|------|
| `core_attn` | 注意力核(softmax/dropout)—— **默认目标**,显存最重、重算最便宜 |
| `moe_act` | GroupedMLP 的激活函数 |
| `layernorm` | input_layernorm / pre_mlp_layernorm |
| `mla_up_proj` | MLA 上投影 + RoPE |
| `mlp` | dense MLP 子模块 |
| `moe` | 整个 MoE 层 |
| `shared_experts` | 共享专家 |
| `mhc` | hyper connections |
| `gdn` | 整个 GatedDeltaNet 模块（**新增**，见下方 [!update]） |
| `gdn_norm_out` | GDN 输出 norm 的输出丢弃重计算 —— 在新基线下**已恢复**且与 `gdn` 互斥，见下方 [!contradiction] |

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> **新增 `gdn`：GatedDeltaNet 整模块 selective 重计算（#5296，8dc6e6676）**
> 新基线的合法 `recompute_modules` 集合（`megatron/core/transformer/transformer_config.py:2315-2326`）含 `"gdn"`：core_attn, moe_act, layernorm, mla_up_proj, mlp, moe, shared_experts, mhc, **gdn**, gdn_norm_out。
> - 语义：`gdn` 在 `recompute_granularity='selective'` 下把**整个 GatedDeltaNet**（in_proj → conv1d → gated delta rule → gated norm → out_proj）包进一次 checkpoint 整体重算，用**标准 checkpointing**（非输出丢弃）。实现：开关字段在 `megatron/core/ssm/gated_delta_net/common.py:304-307`（`self.recompute_gdn`），重算包装在 `megatron/core/ssm/gated_delta_net/gdn.py:251-267`（`tensor_parallel.checkpoint(_checkpointed_compute, False, hidden_states)`，核心计算抽到 `_forward_compute`）；KDA 变体共用同一开关（`megatron/core/ssm/gated_delta_net/kda.py:321`）。
> - 约束：仅当模型为 hybrid、或 `experimental_attention_variant` 属 GDN 家族时可用（`megatron/core/transformer/transformer_config.py:2354-2362`）。
> - **历史更正**：先前 #4715（ff5264c33）曾引入过更细的 `gdn_norm_out`（只对 GDN 输出 norm + HP→CP all-to-all 做**输出丢弃** checkpointing），在 `dev@232c478d4` 时点它确实已被 #5296 的整模块 `gdn` 取代、全仓查无此项。

> [!contradiction] 上一条「`gdn_norm_out` 已不存在」在基线 `71092579` 下**不再成立**：`3549dc62a`（#6088「Refactor: extract and split common logic between GDN & GDN2」，cherry-pick #5843）把 `gdn_norm_out` **重新加了回来**。新基线上二者**并存且互斥**——合法集合同时含 `"gdn"` 与 `"gdn_norm_out"`（`megatron/core/transformer/transformer_config.py:2315-2326`），并显式禁止同时指定（`megatron/core/transformer/transformer_config.py:2364-2368`，报错文案为 “'gdn' recomputes the full GDN-family layer, including gated norm.”）；`gdn_norm_out` 的开关字段 `self.recompute_norm_out` 也回到了 `megatron/core/ssm/gated_delta_net/common.py:302-306`。同一次重构还把原来的单文件 `megatron/core/ssm/gated_delta_net.py` 拆成了包 `megatron/core/ssm/gated_delta_net/`（`common.py` / `gdn.py` / `kda.py`），故本节原先指向 `gated_delta_net.py:267-269`、`:374-387` 的两处 locator 已按新布局改写。

### 3.3 标准 checkpointing vs 输出丢弃 checkpointing

selective 下有**两种**底层机制(README MoE §Fine-grained Recomputation 区分):

- **标准 checkpointing**(`core_attn`、`mlp`、`moe`):torch 风格 —— 存子模块**输入**,丢内部激活,反向重跑子模块前向。
- **输出丢弃 checkpointing**(`moe_act`、`layernorm`、`mla_up_proj`):更激进 —— 连子模块的**输出**也在前向丢掉,反向时重算。对这类"输出大、可由输入廉价重算"的子模块,比标准 checkpointing 省得更多。

### 3.4 `te_checkpoint` vs `tensor_parallel.checkpoint`

`checkpoint_handler`(`megatron/core/transformer/transformer_block.py:674`)按精度选实现:
- `fp8` / `fp4` → `te_checkpoint`(TransformerEngine 的版本,正确处理 fp8 量化上下文与 RNG;分支在 `megatron/core/transformer/transformer_block.py:677-689`)。
- 否则 → `tensor_parallel.checkpoint`(`megatron/core/tensor_parallel/random.py:718`,Megatron 自带,正确处理 TP 的 RNG tracker —— 保证重算的 dropout 与原前向用同一随机种子)。

> 重计算必须保证 dropout 等随机算子**重算与原前向结果一致**,否则梯度错。这就是为什么 checkpoint 要接管 CUDA RNG tracker。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> **两个 checkpoint 健壮性修复**（注：上文 `checkpoint_handler` 在新基线为 `megatron/core/transformer/transformer_block.py:674`）
> - **重计算 × 训练 CUDA graph（#3919，ada8dfe6f）**：`tensor_parallel.checkpoint`（`megatron/core/tensor_parallel/random.py:718`）在函数入口新增——`if is_graph_warmup() or is_graph_capturing(): return function(*args)`（`megatron/core/tensor_parallel/random.py:728-729`）。**CG warmup/capture 期间直接跳过 checkpoint 包装**，因为被捕获的 graph 已记录全部算子、反向无法在已捕获 graph 内再跑一遍重算（与 `CheckpointWithoutOutput` 行为一致）。否则 recompute + 训练 CG 会冲突。
> - **MTP + packed sequence 重计算崩溃（#4593，2b77d32b1）**：`MultiTokenPredictionLayer` 新增自己的 `_checkpointed_forward`（`megatron/core/transformer/multi_token_prediction.py:2360`），仿照 `transformer_block`：把 `attention_mask` / `rotary_pos_emb` / `packed_seq_params` 等 **kwarg 张量经 `custom_forward` 闭包捕获**（`tensor_parallel.checkpoint` 只 stash 位置参数、`te_checkpoint` reentrant 实现也只追踪位置张量输入，kwarg 张量不进重算反向路径），并按精度路由 fp8/fp4 → `te_checkpoint`、否则 `tensor_parallel.checkpoint`。修复 packed-sequence 下 MTP 重计算 crash。

---

## 4. 与其他机制的配合

### 4.1 PP 的部分激活检查点

`num_microbatches_with_partial_activation_checkpoints`(见 `15_megatron_pp_schedulers_analysis.md` §②.2 `max_outstanding_backprops`):1F1B 里**早期 microbatch 多、激活压力大**,可只对前若干个 microbatch 做(部分)重计算,后面的不做。这是"重计算 × microbatch 维度"的精细调度。

### 4.2 与并行轴的关系

- 重计算与 TP/PP/CP/EP/DP **完全正交**,叠加使用。
- selective `moe` 重计算与 MoE 的 `moe_act`/`expert_fc1`(新基线另加 `fused_group_mlp`)**换出互斥**(`megatron/core/transformer/transformer_config.py:2514-2524`:整层重算了就不能再换出层内子模块)。
- `fp8` delayed scaling 不支持 `moe_act`/`layernorm` 重计算(`megatron/core/transformer/transformer_config.py:2386-2392`)。
- 整层 `full` 重计算与 `cuda_graph_impl` 有约束(`megatron/core/transformer/transformer_config.py:3413-3417`,非 selective 时需 `full_iteration`)。

### 4.3 与 offload 叠加

重计算(换算力)+ offload(换带宽)可同时开,各管一部分激活 —— 当算力和 PCIe 都还有余量时,二者分摊省显存压力。

---

## 5. 开销分析与选型

### 5.1 开销总表

| 方案 | 省激活显存 | 计算开销 | 适合 |
|------|-----------|---------|------|
| 不重计算 | —— | 基准 | 显存够 |
| selective(`core_attn`) | 大(去掉 `O(s²)` 的注意力激活) | **~1-2%** | **首选**:几乎免费,优先开 |
| full `block`(部分层) | 可调 | 按重算层数,`<33%` | selective 还不够、差一点 |
| full `uniform`(全层) | 最大 | **~+33%** | 显存极度紧张 |

### 5.2 选型决策

```
激活显存 OOM?
├─ 否 ──► 不重计算(最快)
└─ 是 ──► 先开 selective recompute(--recompute-granularity selective
          --recompute-modules core_attn)—— ~1-2% 开销换大块显存,几乎必开
          │
          ├─ 还差一点 ──► full + block,只重算装不下的那几层
          │
          └─ 还不够 ──► full + uniform(全层重算,+33%);
                        或叠加激活 offload(pp_supplements §2)
```

### 5.3 一句话总结

- **重计算 = 用算力换激活显存**:前向丢激活,反向从检查点重跑前向补回。
- **粒度**:`full`(整层,uniform 全算 / block 部分算)vs `selective`(按子模块)。
- **关键洞察**:selective 重计算 `core_attn` —— 去掉 `O(s²)` 的注意力激活,开销仅 ~1-2%,是性价比最高的一档,优先开;全量 `uniform` 才到 +33%。
- **正交性**:与所有并行轴正交;与 offload 互补(算力 vs 带宽)。

---

*生成依据:`Megatron-LM` `dev` 分支 `71092579`(2026-08-28 由 `ee3f1ff` 重定基线)。源码行号以该 commit 为准。本文是"第二层补遗"3 份之①(激活重计算),后续:② 优化器内部、③ FP8 精度 + CUDA Graph + 算子融合。*

## Related Pages

- [[23_megatron_precision_cudagraph_fusion_analysis]] · [[15_megatron_pp_schedulers_analysis]]
- [[22_megatron_memory_optimization_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
