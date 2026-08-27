---
title: "FlexAttention：可组合注意力的编译范式"
---

# FlexAttention：可组合注意力的编译范式

> 从 hardcoded pattern matching 到语义驱动代码生成的范式转移
> 最后更新: 2026-05-12

---

## 1. 背景：_sfdp_init 的局限

torch.compile 的 `joint_graph_passes` 中，FlashAttention 融合由 `_sfdp_init` 实现。其工作原理是 **Pattern Matching**：

```python
# _sfdp_init 识别这个固定模式：
scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
attn_weights = F.softmax(scores + mask, dim=-1)
out = attn_weights @ v
# → 替换为 scaled_dot_product_attention(q, k, v, attn_mask)
```

**根本局限**：

| 场景 | _sfdp_init 能处理 | 原因 |
|------|-----------------|------|
| 标准 MHA | ✅ | Pattern 固定 |
| GQA / MQA | ✅ | 已有专用 Pattern |
| Sliding Window Attention | ❌ | Mask 逻辑不在已知 Pattern 集合 |
| Block Sparse Attention | ❌ | 需要跳过空 block，结构不匹配 |
| ALiBi / RoPE fused | ❌ | Score 修改不在 Pattern 中 |
| Document-level Attention | ❌ | 多文档拼接 mask 逻辑复杂 |

随着 LLM 变体爆炸（Mistral Sliding Window、Mamba 混合、MoBA、Flash-Linear-Attention），固定 Pattern 维护成本极高。

---

## 2. FlexAttention：语义驱动的解法

**PyTorch 2.4+** 引入 FlexAttention API，核心思路：

> 用户描述 **注意力的计算语义**（What），编译器决定 **最优实现**（How）。

```python
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    and_masks,
    or_masks,
)

# ① 用户定义 mask 逻辑（纯 Python 函数）
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def sliding_window(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx).abs() <= 512

# ② 组合 mask
combined = and_masks(causal_mask, sliding_window)

# ③ 创建 BlockMask（编译时分析稀疏结构）
block_mask = create_block_mask(
    combined,
    B=batch_size, H=num_heads,
    Q_LEN=seq_len, KV_LEN=seq_len,
    device="cuda",
    BLOCK_SIZE=128,
)

# ④ 调用（编译器自动生成融合 Triton kernel）
out = flex_attention(q, k, v, block_mask=block_mask)
```

---

## 3. 核心机制

### 3.1 BlockMask：稀疏结构的编译时分析

```
BlockMask 预计算：将 mask 函数在 block 粒度上求值

序列长度 = 4096, Block Size = 128 → 32×32 的 block grid

对每个 (Q_block, KV_block) 组合，判断：
  FULL:  整个 block 都可见 → 完整执行 GEMM
  PARTIAL: 部分可见 → 执行后 apply mask
  EMPTY: 完全不可见 → 跳过（skip）

BlockMask 本质上是一个 compressed 稀疏表示：
  - full_kv_num: 每个 Q block 完整可见的 KV block 数
  - partial_kv_indices: 需要 masking 的 KV block 列表
```

**节省计算**：对于 Causal + Sliding Window（窗口=512, 序列=4096）：

```
标准 Attention:  4096×4096 = 16M 个元素需要计算
FlexAttention:   只有 ~25% 的 block 是 FULL + PARTIAL
                 EMPTY block 直接跳过 → ~75% 计算量节省
```

### 3.2 score_mod：自定义注意力权重修改

除了 mask，FlexAttention 还支持 `score_mod`——在 softmax 之前修改原始 score：

```python
# ALiBi 位置偏置
def alibi_bias(score, b, h, q_idx, kv_idx):
    bias = -torch.abs(q_idx - kv_idx) * alibi_slopes[h]
    return score + bias

# Temperature scaling
def temperature_scale(score, b, h, q_idx, kv_idx):
    return score * (1.0 / math.sqrt(d_k))

# Soft-capping（Gemma2 使用）
def soft_cap(score, b, h, q_idx, kv_idx):
    return torch.tanh(score / cap) * cap

out = flex_attention(q, k, v, score_mod=alibi_bias, block_mask=block_mask)
```

`score_mod` 被 JIT 编译进 Triton kernel 的 inner loop——无需额外 pass，直接内联。

### 3.3 编译流程

```
flex_attention(q, k, v, block_mask, score_mod)
        ↓
TorchDynamo 捕获 FlexAttention 为高阶算子
        ↓
torch.compile 特殊处理路径（非 _sfdp_init pattern matching）:
  1. 分析 block_mask 的稀疏结构（FULL/PARTIAL/EMPTY block 比例）
  2. 编译 score_mod 为 Triton lambda（内联到 inner loop）
  3. 根据稀疏比例选择 kernel 策略：
     - 高稀疏度 → block-sparse Triton kernel（跳过 EMPTY block）
     - 低稀疏度 → dense FlashAttention kernel（低 overhead）
        ↓
生成的 Triton kernel 包含：
  - Flash Attention 2 的分块 softmax 算法
  - block_mask 驱动的条件跳过逻辑
  - score_mod 的内联计算
```

---

## 4. 与 _sfdp_init 的对比

| 特性 | _sfdp_init（Pattern Matching） | FlexAttention（语义驱动） |
|------|-------------------------------|------------------------|
| 支持范围 | 固定 Pattern 集合 | 任意可表达语义 |
| 新 Attention 变体 | 需修改编译器源码添加 Pattern | 用户直接描述，无需改编译器 |
| Block sparse 支持 | ❌ | ✅（BlockMask 机制） |
| score_mod 支持 | ❌ | ✅（内联进 kernel） |
| 维护成本 | 高（每种变体一个 Pattern） | 低（通用框架） |
| 生成 kernel 质量 | cuDNN/FlashAttention 专用库 | Triton 自动生成 |
| 峰值性能 | 专用库更高（H100 FA3 等） | 通用 kernel 略逊，但差距收窄 |

> [!note]
> _sfdp_init 并未废弃。对于标准 MHA/GQA，专用 FlashAttention 库（FA2/FA3）仍比 FlexAttention 快。
> FlexAttention 的价值在于覆盖所有"非标准"场景，且随着 Triton 成熟差距在缩小。

---

## 5. 典型模型与 FlexAttention

| 模型 | 注意力类型 | FlexAttention 实现方式 |
|------|-----------|----------------------|
| Mistral / Mixtral | Sliding Window | `sliding_window` mask function |
| Llama 3 | Causal Dense | 标准 causal mask |
| Gemma 2 | Soft-cap + Causal | `soft_cap` score_mod |
| DeepSeek V4 CSA | 稀疏 Top-K block 选择 | block_mask 自定义稀疏选择 |
| MoBA | 块级稀疏 + 全局 token | 混合 FULL/PARTIAL block |
| Ring Attention (CP) | 分块因果 | 跨 rank 的 block_mask 拼接 |

---

## 6. FlexAttention 的编译器意义

**范式转变**：

```
旧范式（规则驱动）：
  编译器知道几种特定 attention 的最优实现
  用户写的代码如果不匹配 → 退化为低效实现

新范式（语义驱动）：
  用户描述 "注意力语义"（mask 逻辑 + score 变换）
  编译器根据语义 + 稀疏结构 → 生成最优 kernel

类比：
  旧：SQL 优化器只认识几种固定查询模式
  新：SQL 优化器理解查询语义，对任意查询生成最优执行计划
```

**对 Pass 体系的影响**：

1. `_sfdp_init` 的 Pattern 集合不再需要扩展
2. 新 Attention 变体的支持成本从"修改编译器"降为"写几十行 Python"
3. Block-sparse 计算图的 Pass 优化（跳过 EMPTY block）被 BlockMask 统一处理
4. 与 `auto_chunker` 的协同：FlexAttention 的 block 粒度与 auto_chunker 的 chunk 边界天然对齐

---

## 7. 局限与未来方向

**当前局限**：
- `score_mod` 必须是 element-wise（不能访问其他 token 的信息）
- BlockMask 的预计算开销在动态 mask 场景下较高
- 与 CUDA Graphs 的集成需要静态 BlockMask

**未来方向**：
- `flex_attention` 与 `torch.export` 集成：支持 AOT 编译 + 静态部署
- 扩展 `score_mod` 到行级别操作（如 per-head softmax temperature）
- NPU/XPU 后端支持（目前主要支持 CUDA）
- 与 Ring Attention（CP）的深度集成：跨 rank 的 block_mask 拼接协议

---

## Related Pages

- [[31_joint_graph_passes_guide]] — _sfdp_init 的实现（被 FlexAttention 补充）
- [[30_pre_grad_passes_guide]] — batch_linear_lhs（QKV 融合的上游 Pass）
- [[30_triton_vs_mlir_backend_analysis]] — Triton codegen 路径
- [[31_comm_compute_fusion_guide]] — FlexAttention 在 Context Parallel 中的 WaveEP 集成
- [[23_tilelang_analysis]] — Tile-Level IR：FlexAttention 的 block 粒度与 tile 概念的关系
