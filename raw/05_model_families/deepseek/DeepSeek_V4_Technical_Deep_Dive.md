# DeepSeek-V4 核心技术深度解析

## 一、CSA、HCA、DSA 和 MLA 的区别

### 1.1 各种注意力机制的对比

#### 传统注意力机制

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    传统多头注意力 (MHA)                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  计算复杂度：O(n²)                                                               │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │  Token 1 ──┬───────────────────────────────────────────────────────────────┐ │ │
│  │            │                                                               │ │ │
│  │  Token 2 ──┼───┬─────────────────────────────────────────────────────────┐ │ │ │
│  │            │   │                                                         │ │ │ │
│  │  Token 3 ──┼───┼───┬───────────────────────────────────────────────────┐ │ │ │ │
│  │            │   │   │                                                   │ │ │ │ │
│  │  Token 4 ──┼───┼───┼───┬─────────────────────────────────────────────┐ │ │ │ │ │
│  │            │   │   │   │                                             │ │ │ │ │ │
│  │  Token 5 ──┼───┼───┼───┼───┬───────────────────────────────────────┐ │ │ │ │ │ │
│  │            │   │   │   │   │                                       │ │ │ │ │ │ │
│  │  Token 6 ──┼───┼───┼───┼───┼───┬─────────────────────────────────┐ │ │ │ │ │ │ │
│  │            │   │   │   │   │   │                                 │ │ │ │ │ │ │ │
│  │  Token 7 ──┼───┼───┼───┼───┼───┼───┬───────────────────────────┐ │ │ │ │ │ │ │ │
│  │            │   │   │   │   │   │   │                           │ │ │ │ │ │ │ │ │
│  │  Token 8 ──┼───┼───┼───┼───┼───┼───┼───┬─────────────────────┐ │ │ │ │ │ │ │ │ │
│  │            │   │   │   │   │   │   │   │                     │ │ │ │ │ │ │ │ │ │
│  │  ...       │   │   │   │   │   │   │   │         ...         │ │ │ │ │ │ │ │ │ │
│  │            │   │   │   │   │   │   │   │                     │ │ │ │ │ │ │ │ │ │
│  │  Token N ──┴───┴───┴───┴───┴───┴───┴───┴─────────────────────┴─┴─┴─┴─┴─┴─┴─┘
│  │                                                                                      │
│  │  所有 token 两两计算注意力 (N×N = O(n²))                                             │
│  │                                                                                      │
│  │  问题：上下文越长，计算量呈平方级增长                                                │
│  └──────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 DeepSeek-V4 的注意力机制

#### DSA (DeepSeek Sparse Attention)

DSA 是 DeepSeek-V4 的**稀疏注意力框架**，包含两个子机制：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DSA 稀疏注意力框架                                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  DSA = CSA + HCA                                                                 │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │  CSA (Compressed Sparse Attention) - 压缩稀疏注意力                        │ │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐  │ │ │
│  │  │  压缩比：25% (保留 25% 的 token)                                       │  │ │ │
│  │  │  适用场景：浅层网络，需要保留较多上下文信息                            │  │ │ │
│  │  │  计算复杂度：O(n log n)                                                │  │ │ │
│  │  └───────────────────────────────────────────────────────────────────────┘  │ │ │
│  │                                                                              │ │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐  │ │ │
│  │  │  HCA (Highly Compressed Attention) - 高度压缩注意力                    │  │ │ │
│  │  │  压缩比：10% (仅保留 10% 的 token)                                     │  │ │ │
│  │  │  适用场景：深层网络，只需要关键 token 的长距离依赖                     │  │ │ │
│  │  │  计算复杂度：O(n log n)                                                │  │ │ │
│  │  └───────────────────────────────────────────────────────────────────────┘  │ │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 MLA (Multi-Head Latent Attention)

MLA 是 DeepSeek-V3 引入的注意力机制，与 DeepSeek-V4 的 DSA 有本质区别：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MLA vs DSA 对比                                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │  MLA (Multi-Head Latent Attention) - V3 使用                               │ │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐  │ │ │
│  │  │  核心思想：                                                            │  │ │ │
│  │  │  - 将 QKV 映射到低维潜在空间                                           │  │ │ │
│  │  │  - 在潜在空间进行注意力计算                                            │  │ │ │
│  │  │  - 然后投影回原始维度                                                  │  │ │ │
│  │  │                                                                         │  │ │ │
│  │  │  计算流程：                                                            │  │ │ │
│  │  │  1. Q, K, V → 投影到低维潜在空间 (d' << d)                            │  │ │ │
│  │  │  2. 在潜在空间计算注意力: Attention(Q', K', V')                       │  │ │ │
│  │  │  3. 投影回原始维度                                                     │  │ │ │
│  │  │                                                                         │  │ │ │
│  │  │  优势：                                                                │  │ │ │
│  │  │  - 减少注意力计算的维度                                                │  │ │ │
│  │  │  - 降低 FLOPs                                                         │  │ │ │
│  │  │  - 保持模型容量                                                        │  │ │ │
│  │  └───────────────────────────────────────────────────────────────────────┘  │ │ │
│  │                                                                              │ │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐  │ │ │
│  │  │  DSA (DeepSeek Sparse Attention) - V4 使用                             │  │ │ │
│  │  │  ┌─────────────────────────────────────────────────────────────────┐  │ │ │ │
│  │  │  │  核心思想：                                                      │  │ │ │ │
│  │  │  │  - 在 token 维度进行稀疏化                                      │  │ │ │ │
│  │  │  │  - 动态选择关键 token 进行注意力计算                            │  │ │ │ │
│  │  │  │  - 避免计算所有 token 对之间的注意力                            │  │ │ │ │
│  │  │  │                                                                  │  │ │ │ │
│  │  │  │  计算流程：                                                      │  │ │ │ │
│  │  │  │  1. 计算每个 token 的重要性分数                                 │  │ │ │ │
│  │  │  │  2. 选择 Top-K 个关键 token                                     │  │ │ │ │
│  │  │  │  3. 仅在关键 token 之间计算注意力                               │  │ │ │ │
│  │  │  │                                                                  │  │ │ │ │
│  │  │  │  优势：                                                          │  │ │ │ │
│  │  │  │  - 将复杂度从 O(n²) 降低到 O(n log n)                           │  │ │ │ │
│  │  │  │  - 支持百万 Token 上下文                                        │  │ │ │ │
│  │  │  │  - 显存占用大幅降低                                             │  │ │ │ │
│  │  │  └─────────────────────────────────────────────────────────────────┘  │ │ │ │
│  │  └───────────────────────────────────────────────────────────────────────┘  │ │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 四种注意力机制的详细对比

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    四种注意力机制对比表                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐   │
│  │              │     MHA      │      MLA     │     CSA      │     HCA      │   │
│  ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤   │
│  │  复杂度      │    O(n²)     │   O(n·d'²)   │  O(n log n)  │  O(n log n)  │   │
│  │              │              │  (d'<<d)     │              │              │   │
│  ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤   │
│  │  稀疏化维度  │     无         │  维度维度     │  Token 维度   │  Token 维度   │   │
│  ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤   │
│  │  压缩比      │     100%      │   ~50%       │    25%       │    10%       │   │
│  ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤   │
│  │  适用场景    │  短上下文     │  中等上下文    │  浅层网络     │  深层网络     │   │
│  ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤   │
│  │  上下文长度  │   ≤ 4K        │   ≤ 32K      │   ≤ 128K      │   ≤ 1M        │   │
│  ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤   │
│  │  显存占用    │    100%       │    ~50%      │    ~25%       │    ~10%       │   │
│  ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤   │
│  │  计算量      │    100%       │    ~50%      │    ~25%       │    ~10%       │   │
│  └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘   │
│                                                                                   │
│  说明：                                                                           │
│  - MHA (Multi-Head Attention): 传统多头注意力                                    │
│  - MLA (Multi-Head Latent Attention): 多头潜在注意力 (V3)                       │
│  - CSA (Compressed Sparse Attention): 压缩稀疏注意力 (V4)                       │
│  - HCA (Highly Compressed Attention): 高度压缩注意力 (V4)                       │
│  - DSA (DeepSeek Sparse Attention): DeepSeek 稀疏注意力框架 (V4)                │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.5 CSA 和 HCA 的具体实现

#### CSA (Compressed Sparse Attention)

```python
# CSA 实现
class CompressedSparseAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, compression_ratio=0.25):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.compression_ratio = compression_ratio  # 25%
        
        # QKV 投影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # 压缩采样器
        self.sampler = CompressedSampler(compression_ratio)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV 投影
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # 压缩采样 (仅选择 25% 的关键 token)
        compressed_query, compressed_indices = self.sampler(query)
        compressed_key = key[:, compressed_indices, :]
        compressed_value = value[:, compressed_indices, :]
        
        # 重塑为多头格式
        query = self.reshape_to_heads(compressed_query)
        key = self.reshape_to_heads(compressed_key)
        value = self.reshape_to_heads(compressed_value)
        
        # 注意力计算 (仅在压缩后的 token 之间)
        attention_weights = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = attention_weights / math.sqrt(self.hidden_size // self.num_heads)
        
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask
        
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # 注意力加权求和
        attention_output = torch.matmul(attention_weights, value)
        
        # 重塑回原始格式
        attention_output = self.reshape_from_heads(attention_output)
        
        # 输出投影
        output = self.o_proj(attention_output)
        
        return output
```

#### HCA (Highly Compressed Attention)

```python
# HCA 实现
class HighlyCompressedAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, compression_ratio=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.compression_ratio = compression_ratio  # 10%
        
        # QKV 投影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # 高度压缩采样器
        self.sampler = HighlyCompressedSampler(compression_ratio)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV 投影
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # 高度压缩采样 (仅选择 10% 的关键 token)
        compressed_query, compressed_indices = self.sampler(query)
        compressed_key = key[:, compressed_indices, :]
        compressed_value = value[:, compressed_indices, :]
        
        # 注意力计算 (仅在压缩后的 token 之间)
        attention_weights = torch.matmul(query, compressed_key.transpose(-2, -1))
        attention_weights = attention_weights / math.sqrt(self.hidden_size // self.num_heads)
        
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # 注意力加权求和
        attention_output = torch.matmul(attention_weights, compressed_value)
        
        # 输出投影
        output = self.o_proj(attention_output)
        
        return output
```

### 1.6 DSA 的动态调度策略

DSA 会根据**层深度**和**任务类型**动态选择 CSA 或 HCA：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DSA 动态调度策略                                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │  层感知的稀疏模式：                                                        │ │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐  │ │ │
│  │  │  浅层 (Layer 0-3):                                                     │  │ │ │
│  │  │  - 使用 CSA (压缩比 25%)                                               │  │ │ │
│  │  │  - 保留更多上下文信息                                                  │  │ │ │
│  │  │  - 处理局部特征和术语识别                                              │  │ │ │
│  │  └───────────────────────────────────────────────────────────────────────┘  │ │ │
│  │                                                                              │ │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐  │ │ │
│  │  │  中层 (Layer 4-12):                                                    │  │ │ │
│  │  │  - 使用 CSA (压缩比 15%)                                               │  │ │ │
│  │  │  - 平衡上下文信息和计算效率                                            │  │ │ │
│  │  │  - 处理中距离依赖                                                      │  │ │ │
│  │  └───────────────────────────────────────────────────────────────────────┘  │ │ │
│  │                                                                              │ │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐  │ │ │
│  │  │  深层 (Layer 13+):                                                     │  │ │ │
│  │  │  - 使用 HCA (压缩比 10%)                                               │  │ │ │
│  │  │  - 仅保留关键 token 的长距离依赖                                       │  │ │ │
│  │  └───────────────────────────────────────────────────────────────────────┘  │ │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、mHC 流形约束超连接

### 2.1 问题背景

传统 Transformer 使用简单的残差连接：
```
output = input + F(input)
```

当模型参数规模达到万亿级别时，这种连接方式可能导致信号传播不稳定。

### 2.2 Hyper-Connections (HC)

HC 引入可学习的权重矩阵：
```
output = W₁·input + W₂·F(input) + W₃·input
```

**问题**：信号放大可能失控，达到 3000 倍，导致训练不稳定。

### 2.3 mHC (Manifold-Constrained Hyper-Connections)

**核心思想**：使用 Sinkhorn-Knopp 算法将连接矩阵投影到数学流形上。

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      mHC 架构                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  传统残差连接:                                                                     │
│     output = input + F(input)                                                    │
│                                                                                   │
│  Hyper-Connections (HC):                                                         │
│     output = W₁·input + W₂·F(input) + W₃·input                                  │
│     │                                                                             │
│     └─ 问题：信号放大失控，训练不稳定                                           │
│                                                                                   │
│  mHC (流形约束):                                                                  │
│     │                                                                             │
│     ├─ 使用 Sinkhorn-Knopp 算法                                                │
│     │                                                                             │
│     └─ 将连接矩阵投影到数学流形上                                               │
│                                                                                   │
│     → 信号放大控制在 1.6-2 倍以内                                                │
│                                                                                   │
│  效果：                                                                           │
│  - 训练效率提升约 30%                                                           │
│  - 仅增加 6.7% 额外计算开销                                                     │
│  - 使万亿参数模型稳定训练成为可能                                               │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Sinkhorn-Knopp 算法实现

```python
class SinkhornKnoppAlgorithm:
    def __init__(self, max_iter=100, epsilon=1e-6):
        self.max_iter = max_iter
        self.epsilon = epsilon
    
    def project_to_manifold(self, matrix):
        """
        将连接矩阵投影到数学流形上
        
        约束条件:
        - 行和 = 1 (概率分布)
        - 列和 = 1 (概率分布)
        - 所有元素 >= 0 (非负)
        """
        # 确保非负
        matrix = torch.clamp(matrix, min=0)
        
        # 迭代投影
        for _ in range(self.max_iter):
            # 行归一化
            row_sum = torch.sum(matrix, dim=1, keepdim=True)
            matrix = matrix / (row_sum + self.epsilon)
            
            # 列归一化
            col_sum = torch.sum(matrix, dim=0, keepdim=True)
            matrix = matrix / (col_sum + self.epsilon)
            
            # 检查收敛
            if torch.max(torch.abs(torch.sum(matrix, dim=1) - 1)) < self.epsilon:
                if torch.max(torch.abs(torch.sum(matrix, dim=0) - 1)) < self.epsilon:
                    break
        
        return matrix
```

---

## 三、DualPath 推理框架

### 3.1 问题背景

在智能体场景下，KV-Cache 加载成为 I/O 瓶颈：
- 所有请求都通过 Prefill 引擎，导致存储网卡饱和
- Decode 引擎的网络带宽闲置

### 3.2 DualPath 架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     DualPath 架构                                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  传统架构瓶颈：                                                                   │
│     所有 KV-Cache → Prefill 引擎 (存储网卡饱和)                                  │
│     Decode 引擎网卡闲置                                                           │
│                                                                                   │
│  DualPath 解决方案：                                                              │
│                                                                                   │
│  路径A (传统):                                                                    │
│     存储 → Prefill 引擎 (SNIC) → GPU 显存 → Decode 引擎                         │
│                                                                                   │
│  路径B (新增):                                                                    │
│     存储 → Decode 引擎 (SNIC) → RDMA → Prefill 引擎 (CNIC)                      │
│                                                                                   │
│  动态调度器：                                                                     │
│     根据实时负载均衡选择最优路径                                                  │
│                                                                                   │
│  效果：                                                                           │
│  - 离线推理吞吐量提升 1.87 倍                                                    │
│  - 在线服务吞吐量提升 1.96 倍                                                    │
│  - 利用原本闲置的 Decode 节点网络带宽                                            │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 技术实现

```python
class DualPathInference:
    def __init__(self, num_decode_nodes, num_prefill_nodes):
        self.num_decode_nodes = num_decode_nodes
        self.num_prefill_nodes = num_prefill_nodes
        
        # 路径 A: 传统路径
        self.path_a = TraditionalPath()
        
        # 路径 B: 新增路径
        self.path_b = EnhancedPath()
        
        # 动态调度器
        self.scheduler = DynamicScheduler()
    
    def load_kv_cache(self, kv_cache_requests):
        """加载 KV-Cache"""
        # 动态选择最优路径
        selected_path = self.scheduler.select_path(kv_cache_requests)
        
        if selected_path == "path_a":
            return self.path_a.load(kv_cache_requests)
        else:
            return self.path_b.load(kv_cache_requests)
```

---

## 四、MoE 专家路由 v2

### 4.1 路由机制对比

**V3 版本**：固定专家激活数量
**V4 版本**：根据任务类型动态调整

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MoE 路由 v2                                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  输入 Token → 路由网络 → 专家选择                                                │
│     │                                                                             │
│  ┌──┴──┐                                                                          │
│  │     │                                                                          │
│  ▼     ▼                                                                          │
│ 简单   复杂                                                                       │
│ 任务   任务                                                                       │
│  │     │                                                                          │
│  ▼     ▼                                                                          │
│ 激活  激活                                                                        │
│ 5%   35%                                                                          │
│ 参数   参数                                                                       │
│ (~13B) (~49B)                                                                    │
│                                                                                   │
│  优势：动态调整激活专家组合，根据任务类型按需调用                              │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 实现细节

```python
class RouterNetwork(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 路由层
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # 任务类型分类器 (用于动态调整激活专家)
        self.task_classifier = nn.Linear(hidden_size, 3)  # simple, moderate, complex
    
    def forward(self, hidden_states):
        # 1. 任务类型判断
        task_logits = self.task_classifier(torch.mean(hidden_states, dim=1))
        task_type = torch.argmax(task_logits, dim=-1)
        
        # 2. 根据任务类型调整激活专家数量
        if task_type == 0:  # 简单任务
            k = max(1, self.num_experts // 20)  # 5% 参数
        elif task_type == 1:  # 中等任务
            k = max(4, self.num_experts // 10)  # 10% 参数
        else:  # 复杂推理
            k = max(8, self.num_experts // 4)  # 25% 参数
        
        # 3. 计算路由权重
        routing_logits = self.router(hidden_states)
        
        # 4. Top-K 选择
        routing_weights = F.softmax(routing_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, k=k, dim=-1)
        
        return top_k_weights, top_k_indices, task_type
```

---

## 五、性能对比总结

### 5.1 与前代模型对比 (V4-Pro vs V3.2)

| 指标 | V3.2 | V4-Pro | 提升 |
|------|------|--------|------|
| FLOPs | 100% | 27% | ↓ 73% |
| KV Cache | 100% | 10% | ↓ 90% |
| 上下文长度 | 128K | 1M | ↑ 7.8x |
| 推理成本 | 1x | 0.1x | ↓ 90% |

### 5.2 技术创新点总结

1. **DSA 稀疏注意力**：将注意力计算复杂度从 O(n²) 降低到 O(n log n)
2. **mHC 流形约束**：将信号放大控制在 1.6-2 倍以内，使万亿参数模型稳定训练
3. **DualPath 推理**：双路径 KV-Cache 加载，吞吐量提升 1.87-1.96 倍
4. **MoE 路由 v2**：根据任务类型动态调整激活专家，推理成本降低 40%

---

## 六、参考文献

1. mHC: Manifold-Constrained Hyper-Connections - arXiv:2512.24880
2. DualPath: Breaking the Storage Bandwidth Bottleneck - arXiv:2602.21548
3. DeepSeekMath: DeepSeek-GRPO - arXiv:2402.03300
