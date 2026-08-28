---
title: "Megatron-LM 融合算子优化 深度分析"
---

# Megatron-LM 融合算子优化 深度分析

> **源码基线**：`NVIDIA/Megatron-LM@71092579522a12522d9f323ae180c9825d01928a`（`dev`，2026-08-27）。
> **重定基线**：2026-08-28 由 `ee3f1ffa2acd18131ab67cabab4cec45283512ab`（2026-05-19）推进，跨 578 个提交；本页全部 `path:line` 已在新基线下逐条重核。

**Date**: 2026-05-12
**Status**: Complete
**Source**: `megatron/core/fusions/`, `megatron/core/transformer/moe/fused_a2a.py`

## 1. 优化点是什么？

融合算子的核心思想是将多个连续的内存受限（memory-bound）操作合并为单个 CUDA kernel，消除中间张量的显存读写和 kernel launch 开销。Megatron-LM 的融合算子覆盖三大类：

| 类型 | 融合内容 | 典型文件 |
|------|---------|---------|
| **激活函数融合** | Bias + GELU/SwiGLU/GEGLU + Dropout + Residual | `megatron/core/fusions/fused_bias_geglu.py`, `megatron/core/fusions/fused_bias_swiglu.py`, `megatron/core/fusions/fused_bias_gelu.py` |
| **归一化融合** | Mean/Var + Normalize + Affine | `megatron/core/fusions/fused_layer_norm.py` |
| **通信融合** | Attention Softmax 多 kernel 合并、MoE All-to-All + Permute、Linear + CrossEntropy | `megatron/core/fusions/fused_softmax.py`, `megatron/core/transformer/moe/fused_a2a.py`, `megatron/core/fusions/fused_linear_cross_entropy.py` |

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
# megatron/core/fusions/fused_bias_geglu.py:84
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
# megatron/core/fusions/fused_bias_geglu.py:27
# Exact: x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
# Tanh approx:
y_1 * 0.5 * (1.0 + torch.tanh(0.79788456 * y_1 * (1 + 0.044715 * y_1 * y_1))) * y_2
```

Quick-GEGLU 使用更快的 sigmoid 近似：
```python
# megatron/core/fusions/fused_bias_geglu.py:187
y * torch.sigmoid(1.702 * y)
```

精度损失通常 <0.1%，速度提升 ~15-20%。

### 3.3 FP8 Input Store — 用精度换显存

三个融合算子支持 `fp8_input_store`：

| 算子 | 文件:行 |
|------|---------|
| WeightedQuickGeGLU | `megatron/core/fusions/fused_bias_geglu.py:326` |
| WeightedBiasQuickGeGLU | `megatron/core/fusions/fused_bias_geglu.py:376` |
| BiasSwiGLU | `megatron/core/fusions/fused_bias_swiglu.py:164` |
| SwiGLU | `megatron/core/fusions/fused_bias_swiglu.py:223` |
| WeightedSwiGLU | `megatron/core/fusions/fused_bias_swiglu.py:276` |

```python
# Forward 中将 input 转为 FP8 保存 (1 byte vs 2 bytes for BF16)
input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
ctx.save_for_backward(input_for_backward, ...)

# Backward 中恢复精度
input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
```

对 [10K tokens, 2×8192 hidden] 的 GEGLU 输入，节省 `10000 × 16384 × 1 byte = 164MB` 显存/层。

### 3.4 Weighted Variants — MoE 专属优化

`megatron/core/fusions/fused_bias_geglu.py:410-442` 的 `weighted_bias_quick_geglu_impl`：

```python
# MoE routing: 每个 token 有 per-expert weight
# output = GEGLU(x) * weights  (weights: [num_tokens, 1])
```

Fused kernel 将 activation + gating + weighting 三步合并，对于 MoE 中每个 token 被路由到多个 expert 的场景，避免了 per-expert 的多余显存分配。

### 3.5 Communication Fusion

**Fused Cross-Entropy** (`megatron/core/fusions/fused_cross_entropy.py`)：
- 将 logits_max 和 sum_exp_logits 拼为一个 tensor：只需 1 次 AllReduce 而不是 2 次
- 减少 50% 的 CE loss 通信量

**Fused All-to-All** (`megatron/core/transformer/moe/fused_a2a.py`)：
- DeepEP backend：`FusedDispatch` 合并 Layout 计算 + Permute + A2A + Unpermute
- HybridEP backend：`HybridEPDispatch` 进一步支持 `fuse_permute_dispatch`
- Async execution 支持：dispatch 可与 expert compute 在不同 CUDA stream 上 overlap

### 3.6 Triton 和 CUTLASS Kernel

不同实现层次的融合：

| 层次 | 代表性融合 | 文件 |
|------|-----------|------|
| `@jit_fuser` (TorchScript/compile) | GEGLU, SwiGLU, GELU | `megatron/core/fusions/fused_bias_*.py` |
| Apex CUDA | LayerNorm | `megatron/core/fusions/fused_layer_norm.py:30` |
| Custom CUDA | Softmax (Scaled + Masked) | `megatron/core/fusions/fused_softmax.py:11-152` |
| Triton | Pad Routing Map, Indices Converter, MLA RoPE | `megatron/core/fusions/fused_pad_routing_map.py`, `megatron/core/fusions/fused_mla_yarn_rope_apply.py` |
| CUTLASS/cuTile | Linear + Cross-Entropy, MHC（mHC 实为多后端，见 §7.5） | `megatron/core/fusions/linear_cross_entropy/blackwell/`, `megatron/core/fusions/fused_mhc_kernels.py` |

## 4. 详细融合算子清单

### 4.1 Bias + Dropout + Residual (`megatron/core/fusions/fused_bias_dropout.py`)

**融合**: `x + bias → dropout → + residual`
**场景**: 每个 Transformer Block 的残差连接（pre-norm 或 post-norm）
**优化**: 训练和推理分开的融合路径，推理使用 in-place 操作彻底消除分配

### 4.2 Bias + GEGLU/SwiGLU (`megatron/core/fusions/fused_bias_geglu.py`, `megatron/core/fusions/fused_bias_swiglu.py`)

**融合**: `chunk(y, 2) → activation(y1) → *y2 [+ bias] [+ weighting]`
**场景**: FFN 的门控激活（SwiGLU = LLaMA/Mistral/Qwen 默认，GEGLU = PaLM）
**优化**: 
- Clamped variant（`megatron/core/fusions/fused_bias_swiglu.py:52-65`）：夹紧数值，防止 FP8 溢出
- MoE 加权变体：将 routing probability 直接融入激活计算
- `cpu_offload_input`: 将 backward 需要的 input 暂存 CPU（`megatron/core/fusions/fused_bias_swiglu.py:165`）

### 4.3 Bias + GELU (`megatron/core/fusions/fused_bias_gelu.py`)

**融合**: `x + bias → GELU(x)`
**场景**: 传统 Transformer（GPT-2, BERT）
**优化**: tanh 近似替代精确 erf

### 4.4 Fused LayerNorm (`megatron/core/fusions/fused_layer_norm.py`)

**融合**: `mean/variance → normalize → affine(gamma, beta)`
**场景**: 每个 Transformer Block 的归一化层
**实现**: Apex CUDA kernel（persistent kernel），支持 zero-centered gamma
**限制**: 只支持特定的 hidden size（`megatron/core/fusions/fused_layer_norm.py:73-98`：1024, 1536, 2048, ..., 65536）

### 4.5 Fused Softmax (`megatron/core/fusions/fused_softmax.py`)

**融合**: `scale → +mask → softmax`（三种变体：causal mask / arbitrary mask / no mask）
**场景**: 所有 Attention 的 softmax 计算
**优化**: Causal mask 在寄存器中即时生成，无需显存分配完整的 `[b, np, sq, sk]` mask tensor

### 4.6 MoE 专用融合

| 融合 | 文件 | 内容 |
|------|------|------|
| Pad Routing Map | `megatron/core/fusions/fused_pad_routing_map.py` | 单 Triton kernel 完成 count zeros → compute padding → flip zeros |
| Indices Converter | `megatron/core/fusions/fused_indices_converter.py` | `[num_tokens, topk]` ↔ `[num_tokens, num_experts]` 双向转换 |
| Weighted Squared ReLU | `megatron/core/fusions/fused_weighted_squared_relu.py` | ReLU² × routing weights（MoE routing） |
| Fused A2A | `megatron/core/transformer/moe/fused_a2a.py` | DeepEP/HybridEP：Dispatch/Combine 通信融合 |
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

> 以下为 wiki 基线 commit `ee3f1ff`（2026-05-19）之后、当前 `dev@232c478d4`（2026-06-16）的新增/变更融合机制。原文（§1–§6）的论断经核对仍然成立，下方为补充与勘误。本节所有 `path:line` 已于 2026-08-28 重核至基线 `71092579`。

### 7.1 TE Op-Fuser：把 grouped MLP 的两次 GEMM + 激活融成一条算子链

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> 新增 `use_transformer_engine_op_fuser`（`megatron/core/transformer/transformer_config.py:591`，#4636）。这是一条全新的 MoE grouped MLP 融合**实现路径**：不再用 `@jit_fuser` 包激活，而是借 **Transformer Engine 的 op-fuser API**（`te.pytorch.ops.Sequential`）把 `FC1 GroupedLinear → 激活 → FC2 GroupedLinear` 整条链交给 TE 一并融合（含跨 GEMM 的 epilogue 融合）。
> - 入口：`TEGroupedMLP._is_fused_impl_supported`（`megatron/core/transformer/moe/experts.py:323`）做能力探测，`_make_fused_ops`（`megatron/core/transformer/moe/experts.py:410`）构造融合算子链；`_with_fused_impl`（`:281`）为开关。
> - 支持条件（任一不满足回退非融合路径）：TE ≥ 2.14.0、`tp_group.size()==1`(不支持 TP)、非 fine-grained activation offloading、非 `moe_apply_probs_on_input`、FC1/FC2 均为 `te.pytorch.GroupedLinear`。
> - 激活映射：SwiGLU → `ScaledSwiGLU`；quick-GEGLU → `ScaledClampedQGeGLU`（需 TE ≥ 2.15）。
> - 配套新配置：`moe_single_grouped_weight` / `moe_single_grouped_bias`（`megatron/core/transformer/transformer_config.py`，把每个 expert 的权重/bias 存为 TE `GroupedTensor` 单参数，要求 `moe_grouped_gemm=True` + TE ≥ 2.14.0；`single_grouped_weight` 目前仅在 fp8+mxfp8 下验证过数值正确）；`moe_mlp_glu_interleave_size`（`megatron/core/transformer/transformer_config.py:1079`，GLU 输入改为 gate/linear **交错块**布局，专为高级融合 kernel 设计）。

### 7.2 Op-Fuser 激活扩展：ScaledSReLU 与 Clamped-SwiGLU

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> 在 §7.1 的 op-fuser 链上新增两种激活：
> - **ScaledSReLU（加权 squared-ReLU）**（#4859，能力探测 `megatron/core/transformer/moe/experts.py:396-398`，构造 `:554-562`）：当 `activation_func==squared_relu` 且 `use_fused_weighted_squared_relu=True` 且非 GLU 时，用 `te.pytorch.ops.ScaledSReLU` 融合。这把 §4.6 的 `fused_weighted_squared_relu`（jit_fuser 版）进一步纳入了 TE op-fuser 路径。同时引入 `activation_recompute_in_mlp` 透传（按 TE 签名探测）。
> - **Clamped-SwiGLU（DSv4）**（#5130，注释 `megatron/core/transformer/moe/experts.py:368-369`，门槛判定 `:387-392`，构造 `:526-536`）：带 `activation_func_clamp_value` 的 SwiGLU 复用 `ScaledClampedQGeGLU(alpha=1.0, limit=clamp, glu_linear_offset=0.0)`——因为 cuDNN 的 geglu kernel 是 swiglu 的超集（`sigmoid(alpha·x)·x`，`alpha=1.0` 即 SiLU，`alpha=1.702` 即 quick-gelu）。需 **TE ≥ 2.17.0.dev0**。这与 §4.2 提到的 "Clamped variant 防 FP8 溢出" 一脉相承，但实现从手写 clamp 改为融合 kernel 的 runtime 参数。

### 7.3 TEFusedDenseMLP：Dense MLP 也走 Grouped GEMM 以触发 SM100 融合

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> 新增 `TEFusedMLPWithGroupedLinear`（类定义 `megatron/core/extensions/transformer_engine.py:2892`，别名 `TEFusedDenseMLP` 在 `:3073` 赋值，#4318）与开关 `use_grouped_gemm_for_dense_mlp`（`megatron/core/transformer/transformer_config.py:920`）。**把稠密 MLP 也用 `GroupedLinear(num_groups=1)` 实现**，目的是在 **SM100+（Blackwell）+ MXFP8 recipe** 下触发 TE 的 `ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8` 融合 kernel（FC1 GEMM + SwiGLU + 量化一体）。要求 `use_te_op_fuser=True` 且 SwiGLU 激活；spec 选择见 `megatron/core/models/gpt/gpt_layer_specs.py:554` 的 `get_mlp_module_spec_for_backend`（分支 `:573`）。

### 7.4 Frozen Linear dgrad 折叠

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> `LinearWithFrozenWeight.backward`（`megatron/core/tensor_parallel/layers.py:380-390`，类定义 `:357`，#5092）：当 `grad_output.dim()>2` 时先 `reshape(-1, k)` 折成 2D 再 `matmul`，绕过 PyTorch matmul 对 size-1 前导维不折叠为 `mm` 的问题（pytorch#186148）。对冻结（如 LoRA base、frozen embedding）线性层的反向 dgrad 是一次纯吞吐优化，不改数值。

### 7.5 mHC 融合 kernel 重写：多后端自动选择

> [!deprecated] 2026-06-16：§3.6 与 §4.6 提到的 `fused_mhc_kernels` 原描述为"cuTile/cuTile fused kernels"。#4624（`9d46c924d`，`megatron/core/fusions/fused_mhc_kernels.py`，`megatron/core/transformer/transformer_config.py:1255`）已重写为**多后端自动选择**：Sinkhorn 与 H_post_bda 反向优先 Triton，其余 fused kernel 用 cuTile，否则回退原生 torch。`use_fused_mhc` 不再因 cuTile 缺失而**静默重置为 False**（旧逻辑已删除），全 native 回退时保持开启并只发一条 rank-0 warning。hyper-connection 模块（SinkhornKnopp / H_aggregate / H_post_bda / ProjRms）相应改写。
>
> **2026-08-28 重核（基线 `71092579`）：多后端结论仍然成立，§3.6 表格把 mHC 归为「CUTLASS/cuTile」单后端的写法已不准确。**证据：`megatron/core/fusions/fused_mhc_kernels.py:5`（模块 docstring 自陈 "Uses Triton and cuda.tile (cuTile) kernels when available, with PyTorch" 回退）；`:38-58` 的 `MHC_FORCE_BACKEND=auto|native|triton|cutile` 强制选择；`:66-97` 的 `_CUTILE_AVAILABLE` / `_TRITON_AVAILABLE` 双探测；`:134-147` 的 `MHC_DISABLE_TRITON` / `MHC_DISABLE_CUTILE` 关断。`use_fused_mhc` 的 docstring（`megatron/core/transformer/transformer_config.py:1255-1268`）逐字写明 "Triton for Sinkhorn and H_post_bda backward when available, cuTile for the remaining fused kernels when available, then native torch fallback"，且全 native 回退时 `use_fused_mhc` 保持开启、只发一条 rank-0 warning。该文件在旧基线之后又有两次改动：#6172（`d8b71082e`，cuTile 路径的 mHC mapping 计算保持 fp32）与 #5841（`2f2f8ebae`，EP a2a overlap 下的 mHC selective recompute + CUDA graph）。

### 7.6 Fused MLA：补齐 delayed weight-grad 钩子

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> `FusedMLASelfAttention`（`megatron/core/transformer/multi_latent_attention.py:1359`）新增 `backward_dw()`（`:1503`）与 `set_for_recompute_input_layernorm()`（`:1510`），#5273。修复融合 MLA 在 `delay_wgrad_compute`（延迟权重梯度，与 dispatch/通信 overlap 配合）下缺失 wgrad 钩子导致权重梯度不被触发的 bug；逐个对 `linear_kv_up_proj / linear_qkv_down_proj / linear_q_up_proj` 调 `backward_dw()`（`:1505-1507`），输出投影走 `self._backward_output_proj()`（`:1508`）。

### 7.7 DSv4 Hybrid Attention 融合 kernel

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> 新增 `apply_dsa_kernel_fusion`（`megatron/core/transformer/transformer_config.py:408`，#4894）与 `megatron/core/transformer/experimental_attention_variant/dsa_kernels.py`（DeepSeek Sparse Attention 融合 kernel 封装）。基于 **FlashMLA 稀疏前向**（`flash_mla_sparse_fwd`）+ **cuDNN-Frontend DSA**（CuTe-DSL 反向 + indexer 评分 + TRT-LLM radix top-K），覆盖三条集成路径：可微稀疏注意力 `dsa_sparse_attn`、推理 indexer 评分+top-K `indexer_topk`、训练融合 indexer-loss+稀疏注意力共享反向 `fused_indexer_sparse_attn`。**仅 SM100+（Blackwell）**；缺包时报错或可 `--no-dsa-kernel-fusion` 回退非融合 PyTorch 实现。
>
> [!contradiction] 2026-08-28（基线 `71092579`）：`apply_dsa_kernel_fusion` 这个开关本身**已被标记废弃**，语义从"布尔总开关"退化为"旧名映射"。现在的主开关是 `dsa_kernel_backend: Literal["none", "tilelang", "cudnn"] = "none"`（`megatron/core/transformer/transformer_config.py:361`）；`apply_dsa_kernel_fusion` 的类型也从 `bool` 改成 `Optional[bool] = None`（`:408`，docstring `:409-410` 自陈 "Deprecated DSv4 fused-kernel switch. Use ``dsa_kernel_backend`` instead."）。归一化逻辑在 `:1697-1723`：非 `dsv4_hybrid` 时只发 warning 并忽略；`dsv4_hybrid` 时 `True→"cudnn"`、`False→"none"`，与显式 `dsa_kernel_backend` 冲突则报错，并警告该字段未来会被删除。后端实现也已按 backend 拆成 `megatron/core/transformer/experimental_attention_variant/dsa_cudnn_kernels.py` 与 `dsa_tilelang_kernels.py`（`dsa_kernels.py` 仍在）。

### 7.8 勘误：GDN 统一 A2A 已被回退

> [!contradiction] 2026-06-16 原记（**已被 2026-08-28 重核推翻**，保留以追溯）：#4913（48032d7b3）曾把 GDN forward 里 CP→HP 的**逐序列 All-to-All 循环**合并为**单次统一 A2A**（引入 `_build_head_perm_for_split_sections` / `_build_thd_cp_a2a_perm` 把分段与负载均衡置换折进一次 A2A）。当时判定为"`dev@232c478d4` 源码中该优化已不存在，`megatron/core/ssm/gated_delta_net.py:413-447` 仍是按 `split_sections` 的逐序列/分段 A2A 循环"。
>
> **2026-08-28 重核（基线 `71092579`）：该统一 A2A 已经回来，而且是默认路径。**首先原 locator 已彻底失效——`megatron/core/ssm/gated_delta_net.py` 这个**文件不再存在**，被拆成包 `megatron/core/ssm/gated_delta_net/`（`common.py` / `gdn.py` / `kda.py`，见 #6088 `3549dc62a`「extract and split common logic between GDN & GDN2」）。新实现：`a2a_cp_to_hp`（`megatron/core/ssm/gated_delta_net/common.py:838`）在 `cp_size>1` 时先用 `_build_head_perm_for_split_sections`（`:640`）对 head 维预置换，thd 路径再用 `_build_thd_cp_a2a_perm`（`:599`）对序列维预置换、并**把 `_undo_attention_load_balancing` 折进同一次置换**；源码注释原话是 "Pre-permute head dim so a single unsectioned a2a is equivalent to per-section a2a"（`:864-865`）——即用**一次不分段的 A2A** 取代逐段循环，返回的逆置换交给 `a2a_hp_to_cp`（`:888`）在回程还原。调用点：`gdn.py:308` / `:451`、`kda.py:380` / `:389` / `:515`。逐段循环只作为 `tensor_a2a_cp2hp` / `tensor_a2a_hp2cp` 的 `split_sections` 分支保留（`common.py:750-764` / `:819-833`），而 `a2a_cp_to_hp` 调它们时传的正是不分段形式。**结论反转：§3.5 关于 A2A 融合的论述现在对 GDN 同样适用。**

### 7.9 TE 版本依赖（影响融合可用性）

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> TE 依赖经 #4682（2.15.0，`815c83d9b`）、#4992（2.16.0，`6e091e1b6`）多次 bump。对融合的实际影响：op-fuser 路径需 **TE ≥ 2.14.0**（`megatron/core/transformer/moe/experts.py:350`）；`ScaledClampedQGeGLU`（quick-GEGLU 融合）需 **TE ≥ 2.15**（`:393-395`，以 `hasattr(te_ops, "ScaledClampedQGeGLU")` 探测，报错文案写的就是 "quick_gelu needs TE >= 2.15"）；Clamped-SwiGLU 走 `ScaledClampedQGeGLU` 需 **TE ≥ 2.17.0.dev0**（`:389-390` 的 `is_te_min_version("2.17.0.dev0")`，见 §7.2）。
>
> [!contradiction] 2026-08-28："bump 写在 `pyproject.toml`" 这半句在基线 `71092579` 下**核不上**——`pyproject.toml:87` 仍是 `"transformer-engine[pytorch,core_cu13]>=2.9.0a0,<2.12.0"`，与 `ee3f1ff`、`232c478d4` **逐字相同**；`[tool.uv.sources]` 的 TE git rev（`pyproject.toml:227`）在三处基线也都是 `f031cf87bd054c7558b887df7bed93975456667f`。#4682/#4992 确实是 `71092579` 的祖先、当时改的也确实是 `pyproject.toml`，但其效果已被后续 dev↔main 合并覆盖回去。因此**当前基线下可核验的 TE 版本门槛只有代码里的 `is_te_min_version` / `hasattr` 探测**（上一段的 `experts.py` 三处），不是 `pyproject.toml` 的 pin。

## Related Pages

- [[23_megatron_precision_cudagraph_fusion_analysis]]
- [[22_megatron_memory_optimization_analysis]]
- [[16_megatron_distributed_optimizer_analysis]]
- [[20_megatron_comm_overlap_analysis]]
- [[14_megatron_ep_analysis]]
