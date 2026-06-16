# Megatron-LM 融合算子优化 深度分析

**Date**: 2026-05-12
**Status**: Complete
**Source**: `megatron/core/fusions/`, `megatron/core/transformer/moe/fused_a2a.py`

## 1. 优化点是什么？

融合算子的核心思想是将多个连续的内存受限（memory-bound）操作合并为单个 CUDA kernel，消除中间张量的显存读写和 kernel launch 开销。Megatron-LM 的融合算子覆盖三大类：

| 类型 | 融合内容 | 典型文件 |
|------|---------|---------|
| **激活函数融合** | Bias + GELU/SwiGLU/GEGLU + Dropout + Residual | `fused_bias_geglu.py`, `fused_bias_swiglu.py`, `fused_bias_gelu.py` |
| **归一化融合** | Mean/Var + Normalize + Affine | `fused_layer_norm.py` |
| **通信融合** | Attention Softmax 多 kernel 合并、MoE All-to-All + Permute、Linear + CrossEntropy | `fused_softmax.py`, `fused_a2a.py`, `fused_linear_cross_entropy.py` |

## 2. 为什么有效？

### 2.1 消除中间张量

标准流程：`x → +bias → tmp1 → GELU → tmp2 → *gate → output`（3 个 kernel，2 个中间张量）

融合后：`x, bias → bias_geglu_kernel → output`（1 个 kernel，0 个中间张量）

每个中间张量意味着：
- 一次 HBM 写入 + 一次 HBM 读取
- GPU 显存分配开销
- 额外的显存占用（对 10K tokens × 8K hidden × 2 bytes = 160MB/层）

### 2.2 减少 Kernel Launch 开销

现代 GPU 有数万个 CUDA core。每个 kernel launch 有 ~5-10μs CPU 开销。对于有 80 层、每层 5+ 个激活操作的大模型，每步训练可节省 `80 × 5 × 10μs × 2(fwd+bwd) ≈ 8ms`。

### 2.3 自动微分融合

自定义 `torch.autograd.Function` 将 forward 和 backward 一并融合：

```python
# fused_bias_geglu.py:84
class BiasGeGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_geglu(input, bias)  # 单 kernel

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        return bias_geglu_back(grad_output, input, bias), ...  # 单 kernel
```

Backward 中不仅计算 input grad，同时计算 bias grad，避免重复加载 forward 的输入。

## 3. 关键实现技术

### 3.1 `@jit_fuser` — 统一编译入口

`megatron/core/jit.py:7,33`：

- PyTorch < 2.2：`@jit_fuser` = `torch.jit.script`（TorchScript 图融合）
- PyTorch ≥ 2.2：`@jit_fuser` = `torch.compile`（Dynamo + Inductor）

所有激活融合算子（GEGLU, SwiGLU, GELU, SquaredReLU）通过此装饰器自动获得 JIT 编译优化。

### 3.2 快速近似激活函数

GEGLU 使用 tanh 近似替代精确 erf：
```python
# fused_bias_geglu.py:27
# Exact: x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
# Tanh approx:
y_1 * 0.5 * (1.0 + torch.tanh(0.79788456 * y_1 * (1 + 0.044715 * y_1 * y_1))) * y_2
```

Quick-GEGLU 使用更快的 sigmoid 近似：
```python
# fused_bias_geglu.py:187
y * torch.sigmoid(1.702 * y)
```

精度损失通常 <0.1%，速度提升 ~15-20%。

### 3.3 FP8 Input Store — 用精度换显存

三个融合算子支持 `fp8_input_store`：

| 算子 | 文件:行 |
|------|---------|
| WeightedQuickGeGLU | `fused_bias_geglu.py:326` |
| WeightedBiasQuickGeGLU | `fused_bias_geglu.py:376` |
| BiasSwiGLU | `fused_bias_swiglu.py:164` |
| SwiGLU | `fused_bias_swiglu.py:210` |
| WeightedSwiGLU | `fused_bias_swiglu.py:241` |

```python
# Forward 中将 input 转为 FP8 保存 (1 byte vs 2 bytes for BF16)
input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
ctx.save_for_backward(input_for_backward, ...)

# Backward 中恢复精度
input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
```

对 [10K tokens, 2×8192 hidden] 的 GEGLU 输入，节省 `10000 × 16384 × 1 byte = 164MB` 显存/层。

### 3.4 Weighted Variants — MoE 专属优化

`fused_bias_geglu.py:410-442` 的 `weighted_bias_quick_geglu_impl`：

```python
# MoE routing: 每个 token 有 per-expert weight
# output = GEGLU(x) * weights  (weights: [num_tokens, 1])
```

Fused kernel 将 activation + gating + weighting 三步合并，对于 MoE 中每个 token 被路由到多个 expert 的场景，避免了 per-expert 的多余显存分配。

### 3.5 Communication Fusion

**Fused Cross-Entropy** (`fused_cross_entropy.py`)：
- 将 logits_max 和 sum_exp_logits 拼为一个 tensor：只需 1 次 AllReduce 而不是 2 次
- 减少 50% 的 CE loss 通信量

**Fused All-to-All** (`moe/fused_a2a.py`)：
- DeepEP backend：`FusedDispatch` 合并 Layout 计算 + Permute + A2A + Unpermute
- HybridEP backend：`HybridEPDispatch` 进一步支持 `fuse_permute_dispatch`
- Async execution 支持：dispatch 可与 expert compute 在不同 CUDA stream 上 overlap

### 3.6 Triton 和 CUTLASS Kernel

不同实现层次的融合：

| 层次 | 代表性融合 | 文件 |
|------|-----------|------|
| `@jit_fuser` (TorchScript/compile) | GEGLU, SwiGLU, GELU | `fused_bias_*.py` |
| Apex CUDA | LayerNorm | `fused_layer_norm.py:30` |
| Custom CUDA | Softmax (Scaled + Masked) | `fused_softmax.py:11-152` |
| Triton | Pad Routing Map, Indices Converter, MLA RoPE | `fused_pad_routing_map.py`, `fused_mla_yarn_rope_apply.py` |
| CUTLASS/cuTile | Linear + Cross-Entropy, MHC | `linear_cross_entropy/blackwell/`, `fused_mhc_kernels.py` |

## 4. 详细融合算子清单

### 4.1 Bias + Dropout + Residual (`fused_bias_dropout.py`)

**融合**: `x + bias → dropout → + residual`
**场景**: 每个 Transformer Block 的残差连接（pre-norm 或 post-norm）
**优化**: 训练和推理分开的融合路径，推理使用 in-place 操作彻底消除分配

### 4.2 Bias + GEGLU/SwiGLU (`fused_bias_geglu.py`, `fused_bias_swiglu.py`)

**融合**: `chunk(y, 2) → activation(y1) → *y2 [+ bias] [+ weighting]`
**场景**: FFN 的门控激活（SwiGLU = LLaMA/Mistral/Qwen 默认，GEGLU = PaLM）
**优化**: 
- Clamped variant（`fused_bias_swiglu.py:52-65`）：对数值进行夹持防止 FP8 溢出
- MoE 加权变体：将 routing probability 直接融入激活计算
- `cpu_offload_input`: 将 backward 需要的 input 暂存 CPU（`fused_bias_swiglu.py:165`）

### 4.3 Bias + GELU (`fused_bias_gelu.py`)

**融合**: `x + bias → GELU(x)`
**场景**: 传统 Transformer（GPT-2, BERT）
**优化**: tanh 近似替代精确 erf

### 4.4 Fused LayerNorm (`fused_layer_norm.py`)

**融合**: `mean/variance → normalize → affine(gamma, beta)`
**场景**: 每个 Transformer Block 的归一化层
**实现**: Apex CUDA kernel（persistent kernel），支持 zero-centered gamma
**限制**: 只支持特定的 hidden size（`fused_layer_norm.py:73-98`：1024, 1536, 2048, ..., 65536）

### 4.5 Fused Softmax (`fused_softmax.py`)

**融合**: `scale → +mask → softmax`（三种变体：causal mask / arbitrary mask / no mask）
**场景**: 所有 Attention 的 softmax 计算
**优化**: Causal mask 在寄存器中即时生成，无需显存分配完整的 `[b, np, sq, sk]` mask tensor

### 4.6 MoE 专用融合

| 融合 | 文件 | 内容 |
|------|------|------|
| Pad Routing Map | `fused_pad_routing_map.py` | 单 Triton kernel 完成 count zeros → compute padding → flip zeros |
| Indices Converter | `fused_indices_converter.py` | `[num_tokens, topk]` ↔ `[num_tokens, num_experts]` 双向转换 |
| Weighted Squared ReLU | `fused_weighted_squared_relu.py` | ReLU² × routing weights（MoE routing） |
| Fused A2A | `moe/fused_a2a.py` | DeepEP/HybridEP：Dispatch/Combine 通信融合 |
| Router Fusion | router path | Gate 线性投影 → Top-k 选择 → Softmax/SqrtSoftplus → Aux loss 计算融合为单个 compute-bound kernel |
| Grouped GEMM | cuBLAS / custom kernel | 将多个 expert 的小 GEMM 合并为一次 `grouped GEMM` launch，消除多次 kernel 调度开销，提升 SM 占用率和 Tensor Core 效率 |

## 5. 配置决策树

| 场景 | 推荐融合配置 |
|------|------------|
| 标准训练 | 所有激活融合自动生效（通过 `@jit_fuser`） |
| MoE 训练 | `--moe-token-dispatcher-type alltoall` + DeepEP/HybridEP |
| 大 vocab 训练 | Fused CrossEntropy（自动选择） |
| Blackwell GPU | Linear+CrossEntropy CUTLASS fusion 自动启用 |
| 显存紧张的 MoE | `fp8_input_store=True`（节省激活显存） |
| FP8 训练 | `clamp_value` 防止 SwiGLU/GEGLU 溢出 |

## 6. 何时适用

- ✓ 所有 Transformer 训练/推理（激活融合）
- ✓ MoE 专家数量 > 1（MoE 专用融合）
- ✓ 大 vocabulary（CrossEntropy 融合）
- ✗ 极小的 hidden size（LayerNorm 融合不支持则回退）
- ✗ 非 Blackwell GPU（Linear+CrossEntropy CUTLASS fusion 不可用）

## 7. 增量更新（ee3f1ff → dev@232c478d4）

> 以下为 wiki 基线 commit `ee3f1ff`（2026-05-19）之后、当前 `dev@232c478d4`（2026-06-16）的新增/变更融合机制。原文（§1–§6）的论断经核对仍然成立，下方为补充与勘误。

### 7.1 TE Op-Fuser：把 grouped MLP 的两次 GEMM + 激活融成一条算子链

> [!update] 2026-06-16 · dev@232c478d4
> 新增 `use_transformer_engine_op_fuser`（`transformer_config.py:516`，#4636）。这是一条全新的 MoE grouped MLP 融合**实现路径**：不再用 `@jit_fuser` 包激活，而是借 **Transformer Engine 的 op-fuser API**（`te.pytorch.ops.Sequential`）把 `FC1 GroupedLinear → 激活 → FC2 GroupedLinear` 整条链交给 TE 一并融合（含跨 GEMM 的 epilogue 融合）。
> - 入口：`TEGroupedMLP._is_fused_impl_supported`（`moe/experts.py:315`）做能力探测，`_make_fused_ops`（`moe/experts.py:408`）构造融合算子链；`_with_fused_impl`（`:273`）为开关。
> - 支持条件（任一不满足回退非融合路径）：TE ≥ 2.14.0、`tp_group.size()==1`(不支持 TP)、非 fine-grained activation offloading、非 `moe_apply_probs_on_input`、FC1/FC2 均为 `te.pytorch.GroupedLinear`。
> - 激活映射：SwiGLU → `ScaledSwiGLU`；quick-GEGLU → `ScaledClampedQGeGLU`（需 TE ≥ 2.15）。
> - 配套新配置：`moe_single_grouped_weight` / `moe_single_grouped_bias`（`transformer_config.py`，把每个 expert 的权重/bias 存为 TE `GroupedTensor` 单参数，要求 `moe_grouped_gemm=True` + TE ≥ 2.14.0；`single_grouped_weight` 目前仅在 fp8+mxfp8 下验证过数值正确）；`moe_mlp_glu_interleave_size`（`transformer_config.py:962`，GLU 输入改为 gate/linear **交错块**布局，专为高级融合 kernel 设计）。

### 7.2 Op-Fuser 激活扩展：ScaledSReLU 与 Clamped-SwiGLU

> [!update] 2026-06-16 · dev@232c478d4
> 在 §7.1 的 op-fuser 链上新增两种激活：
> - **ScaledSReLU（加权 squared-ReLU）**（#4859，`moe/experts.py:389/527`）：当 `activation_func==squared_relu` 且 `use_fused_weighted_squared_relu=True` 且非 GLU 时，用 `te.pytorch.ops.ScaledSReLU` 融合。这把 §4.6 的 `fused_weighted_squared_relu`（jit_fuser 版）进一步纳入了 TE op-fuser 路径。同时引入 `activation_recompute_in_mlp` 透传（按 TE 签名探测）。
> - **Clamped-SwiGLU（DSv4）**（#5130，`moe/experts.py:360`）：带 `activation_func_clamp_value` 的 SwiGLU 复用 `ScaledClampedQGeGLU(alpha=1.0, limit=clamp, glu_linear_offset=0.0)`——因为 cuDNN 的 geglu kernel 是 swiglu 的超集（`sigmoid(alpha·x)·x`，`alpha=1.0` 即 SiLU，`alpha=1.702` 即 quick-gelu）。需 **TE ≥ 2.17.0.dev0**。这与 §4.2 提到的 "Clamped variant 防 FP8 溢出" 一脉相承，但实现从手写 clamp 改为融合 kernel 的 runtime 参数。

### 7.3 TEFusedDenseMLP：Dense MLP 也走 Grouped GEMM 以触发 SM100 融合

> [!update] 2026-06-16 · dev@232c478d4
> 新增 `TEFusedMLPWithGroupedLinear`（别名 `TEFusedDenseMLP`，`extensions/transformer_engine.py:2852`，#4318）与开关 `use_grouped_gemm_for_dense_mlp`（`transformer_config.py:830`）。**把稠密 MLP 也用 `GroupedLinear(num_groups=1)` 实现**，目的是在 **SM100+（Blackwell）+ MXFP8 recipe** 下触发 TE 的 `ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8` 融合 kernel（FC1 GEMM + SwiGLU + 量化一体）。要求 `use_te_op_fuser=True` 且 SwiGLU 激活；spec 选择见 `gpt_layer_specs.py:get_mlp_module_spec_for_backend`。

### 7.4 Frozen Linear dgrad 折叠

> [!update] 2026-06-16 · dev@232c478d4
> `LinearWithFrozenWeight.backward`（`tensor_parallel/layers.py:375`，#5092）：当 `grad_output.dim()>2` 时先 `reshape(-1, k)` 折成 2D 再 `matmul`，绕过 PyTorch matmul 对 size-1 前导维不折叠为 `mm` 的问题（pytorch#186148）。对冻结（如 LoRA base、frozen embedding）线性层的反向 dgrad 是一次纯吞吐优化，不改数值。

### 7.5 mHC 融合 kernel 重写：多后端自动选择

> [!deprecated] 2026-06-16：§3.6 与 §4.6 提到的 `fused_mhc_kernels` 原描述为"cuTile/cuTile fused kernels"。#4624（`fusions/fused_mhc_kernels.py`，`transformer_config.py:1103`）已重写为**多后端自动选择**：Sinkhorn 与 H_post_bda 反向优先 Triton，其余 fused kernel 用 cuTile，否则回退原生 torch。`use_fused_mhc` 不再因 cuTile 缺失而**静默重置为 False**（旧逻辑已删除），全 native 回退时保持开启并只发一条 rank-0 warning。hyper-connection 模块（SinkhornKnopp / H_aggregate / H_post_bda / ProjRms）相应改写。

### 7.6 Fused MLA：补齐 delayed weight-grad 钩子

> [!update] 2026-06-16 · dev@232c478d4
> `FusedMLASelfAttention` 新增 `backward_dw()` 与 `set_for_recompute_input_layernorm()`（`transformer/multi_latent_attention.py:1352`，#5273）。修复融合 MLA 在 `delay_wgrad_compute`（延迟权重梯度，与 dispatch/通信 overlap 配合）下缺失 wgrad 钩子导致权重梯度不被触发的 bug；逐个对 `linear_kv_up_proj / linear_qkv_down_proj / linear_q_up_proj / output_proj` 调 `backward_dw()`。

### 7.7 DSv4 Hybrid Attention 融合 kernel

> [!update] 2026-06-16 · dev@232c478d4
> 新增 `apply_dsa_kernel_fusion`（`transformer_config.py:346`，#4894）与 `experimental_attention_variant/dsa_kernels.py`（DeepSeek Sparse Attention 融合 kernel 封装）。基于 **FlashMLA 稀疏前向**（`flash_mla_sparse_fwd`）+ **cuDNN-Frontend DSA**（CuTe-DSL 反向 + indexer 评分 + TRT-LLM radix top-K），覆盖三条集成路径：可微稀疏注意力 `dsa_sparse_attn`、推理 indexer 评分+top-K `indexer_topk`、训练融合 indexer-loss+稀疏注意力共享反向 `fused_indexer_sparse_attn`。**仅 SM100+（Blackwell）**；缺包时报错或可 `--no-dsa-kernel-fusion` 回退非融合 PyTorch 实现。

### 7.8 勘误：GDN 统一 A2A 已被回退

> [!contradiction] 2026-06-16：#4913（48032d7b3）曾把 GDN forward 里 CP→HP 的**逐序列 All-to-All 循环**合并为**单次统一 A2A**（引入 `_build_head_perm_for_split_sections` / `_build_thd_cp_a2a_perm` 把分段与负载均衡置换折进一次 A2A）。但**当前 `dev@232c478d4` 源码中该优化已不存在**——`ssm/gated_delta_net.py:413-447` 仍是按 `split_sections` 的逐序列/分段 A2A 循环（`_build_thd_cp_a2a_perm` 在 HEAD 已不可见，被后续 dev↔main 合并回退/取代）。因此 §3.5 关于 MoE A2A 融合的论述对 GDN **不适用**当前代码；记录此 PR 仅为追溯历史。

### 7.9 TE 版本依赖（影响融合可用性）

> [!update] 2026-06-16 · dev@232c478d4
> TE 依赖经 #4682（2.15.0）、#4992（2.16.0）多次 bump（`pyproject.toml`）。对融合的实际影响：op-fuser 路径需 **TE ≥ 2.14.0**；`ScaledClampedQGeGLU`（quick-GEGLU 融合）需 **TE ≥ 2.15**；Clamped-SwiGLU 走 `ScaledClampedQGeGLU` 需 **TE ≥ 2.17.0.dev0**（见 §7.2）。

## Related Pages

- [[megatron_precision_cudagraph_fusion_analysis]]
- [[megatron_memory_optimization_analysis]]
- [[megatron_distributed_optimizer_analysis]]
- [[megatron_comm_overlap_analysis]]
- [[megatron_ep_analysis]]
