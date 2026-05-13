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

## Related Pages

- [[megatron_memory_optimization_analysis]]
- [[megatron_distributed_optimizer_analysis]]
- [[megatron_comm_overlap_analysis]]
- [[Megatron-LM_MoE_Zero_Redundancy_Analysis]]
