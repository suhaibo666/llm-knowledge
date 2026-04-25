# DeepSeek-V4 Compressed Key-Value Entries 详细解析

## 1. 什么是 Compressed Key-Value Entries？

Compressed Key-Value Entries（压缩的 Key-Value 条目）是 DeepSeek-V4 中 CSA（Compressed Sparse Attention）和 HCA（Heavily Compressed Attention）的核心技术，用于在长序列场景下大幅减少 KV Cache 的内存占用和计算量。

## 2. 基本概念

### 2.1 传统 Attention 的 KV Cache 问题

在传统的 Transformer Attention 中：
- 每个 token 都需要存储一个 Key 向量和一个 Value 向量
- 对于长度为 `n` 的序列，需要存储 `n` 个 KV 对
- 当 `n` 达到 1M（百万）时，KV Cache 的内存占用成为瓶颈

### 2.2 Compressed KV Entries 的核心思想

**将多个 token 的 KV 合并为一个压缩的 KV 条目**：
- 原始：`n` 个 token → `n` 个 KV 对
- 压缩后：`n` 个 token → `n/m` 个压缩 KV 条目（m 是压缩率）

**压缩率示例**：
- 如果 `m = 8`，则压缩后 KV 数量减少到原来的 `1/8`
- 对于 1M 长度的序列，压缩后只需要 `125K` 个 KV 条目

## 3. CSA 中的 Compressed KV Entries 计算细节

### 3.1 计算流程

#### 步骤 1：计算原始 KV 和压缩权重

给定输入隐藏状态 `H ∈ R^(n×d)`（n 个 token，d 维隐藏层）：

```python
# 计算原始 KV
C_a = H · W_a_KV  # ∈ R^(n×c)
C_b = H · W_b_KV  # ∈ R^(n×c)

# 计算压缩权重
Z_a = H · W_a_Z   # ∈ R^(n×c)
Z_b = H · W_b_Z   # ∈ R^(n×c)
```

其中：
- `W_a_KV, W_b_KV ∈ R^(d×c)`：KV 投影矩阵
- `W_a_Z, W_b_Z ∈ R^(d×c)`：压缩权重投影矩阵
- `c`：压缩后的 KV 维度（通常远小于 d）

#### 步骤 2：压缩每 m 个 token 的 KV

对于第 `i` 个压缩后的 KV 条目 `C_Comp[i]`：

```python
# 公式 (11)：计算压缩权重（Softmax 归一化）
S_a[m*i : m*(i+1)-1] = Softmax_row(Z_a[m*i : m*(i+1)-1] + B_a)
S_b[m*(i-1) : m*i-1] = Softmax_row(Z_b[m*(i-1) : m*i-1] + B_b)

# 公式 (12)：加权合并 KV
C_Comp[i] = Σ_{j=m*i}^{m*(i+1)-1} S_a[j] ⊙ C_a[j] + Σ_{j=m*(i-1)}^{m*i-1} S_b[j] ⊙ C_b[j]
```

其中：
- `B_a, B_b ∈ R^(m×c)`：可学习的位置偏置
- `S_a, S_b ∈ R^(m×c)`：压缩权重（经过 Softmax 归一化）
- `⊙`：Hadamard 积（逐元素相乘）

### 3.2 关键特性

#### 特性 1：重叠压缩（Overlapped Compression）

**重要**：CSA 使用**重叠压缩**策略：

```
Token 索引:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
            └─── m=4 ───┘
            └────── m=4 ───────┘
            └───────── m=4 ─────────┘

C_Comp[0] 来自: C_a[0:3]      (4 个 token)
C_Comp[1] 来自: C_a[4:7] + C_b[0:3]  (4 + 4 个 token，重叠!)
C_Comp[2] 来自: C_a[8:11] + C_b[4:7]  (4 + 4 个 token，重叠!)
C_Comp[3] 来自: C_a[12:15] + C_b[8:11] (4 + 4 个 token，重叠!)
```

**重叠的好处**：
- `C_Comp[1]` 使用了 `C_b[0:3]`，这些 token 也参与了 `C_Comp[0]` 的计算
- 这样可以保留局部细粒度的依赖关系
- 避免了信息丢失

#### 特性 2：双路径压缩（Dual-Path Compression）

CSA 同时使用 `C_a` 和 `C_b` 两条路径：
- `C_a`：主路径，用于当前块的压缩
- `C_b`：辅助路径，与前一个块重叠，提供局部信息

#### 特性 3：Softmax 归一化

```python
# 对每个压缩块内的权重进行 row-wise Softmax
S = Softmax_row([Z_a[block] + B_a; Z_b[block] + B_b])
```

这确保了：
- 每个压缩块内的权重和为 1
- 权重是概率分布，避免数值不稳定

### 3.4 完整计算示例

假设：
- `n = 16`（16 个 tokens）
- `m = 4`（每 4 个 tokens 压缩为 1 个）
- `c = 512`（压缩后的 KV 维度）

#### 计算过程：

```python
# 1. 原始 KV 计算
H.shape = (16, d)  # 16 个 tokens
C_a.shape = (16, 512)
C_b.shape = (16, 512)
Z_a.shape = (16, 512)
Z_b.shape = (16, 512)

# 2. 压缩权重计算
B_a.shape = (4, 512)  # 每个块的位置偏置
B_b.shape = (4, 512)

# C_Comp[0]
S_a[0:3] = Softmax(Z_a[0:3] + B_a)  # shape: (4, 512)
C_Comp[0] = Σ_{j=0}^{3} S_a[j] ⊙ C_a[j]  # shape: (512,)

# C_Comp[1]
S_a[4:7] = Softmax(Z_a[4:7] + B_a)  # shape: (4, 512)
S_b[0:3] = Softmax(Z_b[0:3] + B_b)  # shape: (4, 512)
C_Comp[1] = Σ_{j=4}^{7} S_a[j] ⊙ C_a[j] + Σ_{j=0}^{3} S_b[j] ⊙ C_b[j]

# C_Comp[2]
S_a[8:11] = Softmax(Z_a[8:11] + B_a)  # shape: (4, 512)
S_b[4:7] = Softmax(Z_b[4:7] + B_b)  # shape: (4, 512)
C_Comp[2] = Σ_{j=8}^{11} S_a[j] ⊙ C_a[j] + Σ_{j=4}^{7} S_b[j] ⊙ C_b[j]

# C_Comp[3]
S_a[12:15] = Softmax(Z_a[12:15] + B_a)  # shape: (4, 512)
S_b[8:11] = Softmax(Z_b[8:11] + B_b)  # shape: (4, 512)
C_Comp[3] = Σ_{j=12}^{15} S_a[j] ⊙ C_a[j] + Σ_{j=8}^{11} S_b[j] ⊙ C_b[j]

# 3. 最终结果
C_Comp.shape = (4, 512)  # 16 个 tokens 压缩为 4 个压缩 KV 条目
```

### 3.5 是否是 Token-Level 计算？

**是的！** Compressed KV Entries 是 **token-level** 的计算：

1. **输入**：每个 token 的隐藏状态 `H[i] ∈ R^d`
2. **处理**：将 `m` 个连续 tokens 的 KV 合并为一个压缩条目
3. **输出**：每个压缩条目对应 `m` 个原始 tokens

**关键点**：
- 压缩是在**token 序列维度**上进行的
- 每个压缩条目可以看作是 `m` 个 tokens 的"聚合表示"
- 压缩后的每个条目仍然保持 `c` 维的 KV 表示

## 4. HCA 中的 Compressed KV Entries

HCA 的压缩策略与 CSA 类似，但有一些重要区别：

### 4.1 计算公式

```python
# 公式 (20-21)：计算原始 KV
C = H · W_KV  # ∈ R^(n×c)
Z = H · W_Z   # ∈ R^(n×c)

# 公式 (22-23)：压缩
S[m'*i : m'*(i+1)-1] = Softmax_row(Z[m'*i : m'*(i+1)-1] + B)
C_Comp[i] = Σ_{j=m'*i}^{m'*(i+1)-1} S[j] ⊙ C[j]
```

### 4.2 CSA vs HCA 的区别

| 特性 | CSA | HCA |
|------|-----|-----|
| **压缩率** | `m`（较小，如 4-8） | `m'`（较大，`m' ≫ m`，如 32-64） |
| **重叠** | 有重叠（`C_b` 路径） | 无重叠 |
| **稀疏注意力** | 有（只选择 top-k 压缩条目） | 无（全注意力） |
| **局部信息** | 通过重叠保留 | 通过滑动窗口保留 |

### 4.3 HCA 计算示例

假设：
- `n = 64`（64 个 tokens）
- `m' = 16`（每 16 个 tokens 压缩为 1 个）

```python
# 1. 原始 KV 计算
C.shape = (64, 512)
Z.shape = (64, 512)

# 2. 压缩
B.shape = (16, 512)

C_Comp[0] = Σ_{j=0}^{15} S[j] ⊙ C[j]  # tokens 0-15
C_Comp[1] = Σ_{j=16}^{31} S[j] ⊙ C[j]  # tokens 16-31
C_Comp[2] = Σ_{j=32}^{47} S[j] ⊙ C[j]  # tokens 32-47
C_Comp[3] = Σ_{j=48}^{63} S[j] ⊙ C[j]  # tokens 48-63

# 3. 结果
C_Comp.shape = (4, 512)  # 64 个 tokens 压缩为 4 个
```

## 5. 为什么这样设计？

### 5.1 内存效率

| 场景 | 原始 KV | CSA (m=8) | HCA (m'=32) |
|------|---------|-----------|-------------|
| 1K tokens | 1K × 2 × 512 × 2B | 128 × 2 × 512 × 2B | 32 × 2 × 512 × 2B |
| 1M tokens | 1M × 2 × 512 × 2B | 125K × 2 × 512 × 2B | 31.25K × 2 × 512 × 2B |

**节省比例**：
- CSA (m=8)：减少 **87.5%** 的 KV 内存
- HCA (m'=32)：减少 **96.875%** 的 KV 内存

### 5.2 计算效率

压缩后的 KV 可以：
1. **减少注意力计算量**：从 O(n²) 降到 O((n/m)²)
2. **提高缓存命中率**：更少的 KV 条目意味着更好的缓存局部性
3. **支持超长上下文**：1M tokens 的上下文变得可行

### 5.3 信息保留

通过以下机制保留信息：
1. **Softmax 归一化**：确保权重是概率分布，避免信息丢失
2. **可学习的位置偏置**：`B` 允许模型学习不同位置的重要性
3. **重叠压缩（CSA）**：通过 `C_b` 路径保留局部依赖
4. **滑动窗口（HCA）**：额外的小窗口 KV 保留局部细粒度信息

## 6. 完整的 CSA 实现伪代码

```python
def compressed_sparse_attention(H, m=8, c=512):
    """
    H: 输入隐藏状态，shape = (n, d)
    m: 压缩率
    c: 压缩后的 KV 维度
    """
    n, d = H.shape
    
    # 1. 计算原始 KV 和压缩权重
    C_a = H @ W_a_KV  # (n, c)
    C_b = H @ W_b_KV  # (n, c)
    Z_a = H @ W_a_Z   # (n, c)
    Z_b = H @ W_b_Z   # (n, c)
    
    # 2. 计算压缩后的 KV
    num_compressed = n // m
    C_Comp = []
    
    for i in range(num_compressed):
        # 定义当前块的 token 索引
        start_a = i * m
        end_a = min((i + 1) * m, n)
        
        start_b = max(0, (i - 1) * m)
        end_b = i * m
        
        # 计算压缩权重（Softmax 归一化）
        S_a = softmax(Z_a[start_a:end_a] + B_a, dim=0)  # (m, c)
        S_b = softmax(Z_b[start_b:end_b] + B_b, dim=0)  # (m, c)
        
        # 加权合并 KV
        compressed = (
            torch.sum(S_a * C_a[start_a:end_a], dim=0) +  # C_a 路径
            torch.sum(S_b * C_b[start_b:end_b], dim=0)    # C_b 路径（重叠）
        )
        
        C_Comp.append(compressed)
    
    C_Comp = torch.stack(C_Comp)  # (num_compressed, c)
    
    return C_Comp
```

## 7. 总结

Compressed Key-Value Entries 是 DeepSeek-V4 实现百万 token 上下文的关键技术：

1. **Token-Level 压缩**：将 `m` 个连续 tokens 的 KV 合并为一个压缩条目
2. **可学习的压缩权重**：通过 Softmax 和可学习的位置偏置，确保信息保留
3. **重叠策略（CSA）**：通过双路径设计保留局部依赖
4. **高效计算**：大幅减少 KV Cache 内存和注意力计算量

这项技术使得 DeepSeek-V4 能够高效处理 1M tokens 的超长上下文，同时保持模型的表达能力。