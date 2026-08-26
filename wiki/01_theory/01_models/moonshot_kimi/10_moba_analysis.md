# MoBA: Mixture of Block Attention 长上下文注意力机制

> **论文信息**: MoBA: Mixture of Block Attention for Long-Context LLMs
> **作者**: Enzhe Lu, Zhejun Jiang, Jingyuan Liu 等 (Moonshot AI, Tsinghua University, Zhejiang Lab)
> **arXiv**: 2502.13189 (2025-02)
> **开源**: https://github.com/MoonshotAI/MoBA

---

## 一、核心问题：长上下文注意力效率瓶颈

标准 Transformer 注意力复杂度为 $O(N^2)$，上下文从 8K 扩展到 1M/10M 时计算开销平方级增长。

| 方案类型 | 代表方法 | 核心问题 |
|---------|---------|---------|
| 强偏置结构 | Sink, Sliding Window | 任务特定，泛化差 |
| 推理时动态稀疏 | Quest, Minference | 无法降低训练成本 |
| 线性注意力近似 | Mamba, RWKV, RetNet | 与标准注意力差异大，复杂推理未验证 |

**MoBA 设计哲学**："less structure" — 不引入预定义偏置，让模型自主决定注意力分布。

---

## 二、架构：将 MoE 原理应用于注意力

```
传统 MoE: 每个 token → 选择性地路由到不同的 FFN Expert
MoBA:     每个 Query → 选择性地路由到不同的 KV Block
```

### 2.1 核心公式

标准注意力：
$$
\text{Attn}(q, K, V) = \text{Softmax}(qK^\top)V
$$

MoBA 注意力：
$$
\text{MoBA}(q, K, V) = \text{Softmax}\left(qK[I]^\top\right)V[I]
$$

其中 $I \subseteq [N]$ 是被选中的 key-value 索引集合。

### 2.2 块划分

- 块大小：$B = \frac{N}{n}$
- 第 $i$ 个块：$I_i = [(i-1) \times B + 1, i \times B]$

### 2.3 架构图

```
Query Tokens                    KV Blocks
┌──────┐                        ┌─────┬─────┬─────┬─────┐
│  q1  │─── top-2 ────────────→│ B1  │ B2  │     │     │
│      │                       │ ███ │ ███ │     │     │
│      │─── top-2 ────────────────────────┬─────┬─────┐
│  q2  │                                 │ B3  │ B4  │
└──────┘                                 │ ███ │ ███ │
                                         └─────┴─────┴─────┘

Router: s_i = ⟨q, mean_pool(K[I_i])⟩ → top-k selection
```

---

## 三、块路由机制

### 3.1 门控网络

亲和度分数：
$$
s_i = \langle q, \text{mean\_pool}(K[I_i]) \rangle
$$

Top-K 选择：
$$
g_i = \begin{cases} 1 & \text{if } s_i \in \text{Top}_k(\{s_j\}, k) \\ 0 & \text{otherwise} \end{cases}
$$

### 3.2 因果性保证

1. **不关注未来块**：$\text{pos}(q) < i \times B \implies s_i = -\infty$
2. **当前块强制路由**：每个 token 必须路由到所在块（类似 MoE shared expert），应用因果掩码

### 3.3 细粒度块分割

| 配置 (top-k / 总块数) | 稀疏度 | LM Loss |
|----------------------|--------|---------|
| 2 / 8 | 75% | ~2.255 |
| 4 / 16 | 75% | ~2.248 |
| 8 / 32 | 75% | ~2.242 |
| 32 / 128 | 75% | ~2.235 |

**结论**：相同稀疏度下，块越细粒度性能越好。

---

## 四、训练方法

### 4.1 MoBA/Full 混合预训练

```
Stage 1 (90% tokens): MoBA — 高效训练，学习动态块选择
Stage 2 (10% tokens): Full Attention — 恢复全注意力能力
```

**关键发现**：MoBA ↔ Full 切换时**无 loss spike**。

### 4.2 层级混合 (SFT 阶段)

```
Layer 1-29:  MoBA          ← 高效处理长上下文
Layer 30-32: Full Attention ← 保证梯度传播和性能
```

### 4.3 Scaling Law

| 指标 | MoBA | Full Attention |
|------|------|----------------|
| LM Loss (8K) | $2.625 \times C^{-0.063}$ | $2.622 \times C^{-0.063}$ |
| Trailing Loss (32K) | $1.546 \times C^{-0.108}$ | $1.464 \times C^{-0.097}$ |

---

## 五、性能基准

### 5.1 速度对比

| 序列长度 | Flash Attention | MoBA | 加速比 |
|---------|----------------|------|--------|
| 128K | ~80ms | ~40ms | ~2x |
| 512K | ~500ms | ~100ms | ~5x |
| **1M** | **~800ms** | **~120ms** | **~6.5x** |
| **10M** | **~10s** | **~0.6s** | **~16x** |

### 5.2 下游任务 (Llama-8B-1M)

| Benchmark | MoBA | Full | 差异 |
|-----------|------|------|------|
| CEval | **0.6273** | 0.6165 | **+0.0108** |
| GSM8K | **0.7278** | 0.7142 | **+0.0136** |
| LongBench @32K | **0.4828** | 0.4821 | +0.0007 |
| RULER @128K | 0.7818 | 0.7849 | -0.0031 |

**关键**：多数任务性能相当，部分任务 MoBA 更优。

---

## 六、MoBA 的统一视角

```
MoBA 统一框架
    │
    ├── 固定选最近块 → Sliding Window Attention
    ├── 固定选首尾块 → Attention Sink
    ├── top-1 gating → LongHeads
    ├── 特殊块表示函数 → Quest
    └── 动态 top-k → MoBA (通用形式)
```

---

## 七、在 Kimi 技术路线中的地位

```
Kimi 长上下文技术栈:
┌─────────────────────────────────────────┐
│ 应用层: Kimi Chat / API                 │
│   │                                     │
│   ▼                                     │
│ 推理优化: MoBA (Prefill 阶段)            │
│   - 动态块选择，高效处理长 prompt         │
│   │                                     │
│   ▼                                     │
│ 训练优化: MoBA/Full 混合预训练           │
│   - 90% MoBA + 10% Full                 │
│   │                                     │
│   ▼                                     │
│ 基础架构: Transformer + FlashAttention   │
└─────────────────────────────────────────┘
```

**已部署**：论文明确指出 "MoBA has already been deployed to support Kimi's long-context requests"

---

## 八、关键超参数建议

| 场景 | Block Size | Top-K | 稀疏度 |
|------|-----------|-------|--------|
| 短序列 (8K-32K) | 512 | 3 | 75-95% |
| 中序列 (128K) | 2048 | 3 | ~95% |
| 长序列 (1M) | 4096 | 12 | ~95% |
| 超长序列 (10M) | 可变 | 3 (固定) | ~95% |

---

## Related Pages

- [[01_theory/index]]
- [[01_theory/01_models/attention_is_all_you_need_analysis]]
- [[02_engineering/03_infer_frameworks/mooncake_analysis]]
