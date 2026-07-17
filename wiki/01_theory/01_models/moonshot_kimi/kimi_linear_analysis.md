# Kimi Linear: 高效线性注意力架构

> **论文信息**: Kimi Linear: An Expressive, Efficient Attention Architecture
> **作者**: Kimi Team (Yu Zhang, Zongyu Lin, Xingcheng Yao 等 60+ 作者)
> **arXiv**: 2510.26692 (2025-10)
> **开源**: https://github.com/MoonshotAI/Kimi-Linear
> **模型**: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
> **机制与 Kernel 深挖（2026-07-17）**: [[gdn_kda_linear_attention_analysis]] · [[gdn_kda_kernel_implementation_analysis]]

---

## 一、核心问题：长序列注意力效率瓶颈

标准 Softmax 注意力复杂度 $O(T^2)$，在智能体 (Agentic) 和测试时扩展 (Test-Time Scaling) 场景下成为瓶颈：
- 长轨迹处理 (RL 测试时扩展)
- 工具使用 (长时间上下文维护)
- 复杂决策空间 (大量中间状态)

**设计目标**：质量匹配或超越全注意力，同时速度和内存显著增益。

---

## 二、架构：3:1 KDA-MLA 混合

```
Layer 1:  [KDA]  ← Kimi Delta Attention (线性注意力)
Layer 2:  [KDA]
Layer 3:  [KDA]
Layer 4:  [MLA]  ← Full Multi-Head Latent Attention (全局)
Layer 5:  [KDA]
Layer 6:  [KDA]
Layer 7:  [KDA]
Layer 8:  [MLA]
   ...   ← 3:1 比例重复
```

### 2.1 KDA 核心公式

$$\mathbf{S}_t = \left(\mathbf{I}-\beta_t\bm{k}_{t}\bm{k}_{t}^{\top}\right)\operatorname{Diag}\left(\bm{\alpha}_t \right)\mathbf{S}_{t-1} + \beta_t\bm{k}_{t}\bm{v}_{t}^{\top}$$

$$\bm{o}_t = \mathbf{S}_t^\top \bm{q}_t$$

| 符号 | 含义 |
|------|------|
| $\mathbf{S}_t \in \mathbb{R}^{d_k \times d_v}$ | 矩阵值循环状态 (快速权重记忆) |
| $\bm{\alpha}_t \in [0,1]^{d_k}$ | **通道级细粒度遗忘门** (核心创新) |
| $\beta_t \in [0,1]$ | 学习率/更新强度 |

### 2.2 混合比例消融

| 混合比例 (KDA:MLA) | 训练 Loss | 验证 Loss | 推理开销 |
|-------------------|-----------|-----------|---------|
| 7:1 | 低 | **高** (泛化差) | 最低 |
| **3:1** | **低** | **低** | **平衡** |
| 1:1 | 中 | 中 | 高 |
| 0:1 (纯 MLA) | 高 | 高 | 最高 |

---

## 三、关键技术创新

### 3.1 通道级细粒度门控

```
GDN:  α_t (标量) → 整个头共享同一遗忘率
      ┌─────────────────────────────────┐
      │  h1  h2  h3  ...  hd           │
      │  α   α   α   ...  α   ← 相同   │
      └─────────────────────────────────┘

KDA:  α_t (向量) → 每个通道独立遗忘率
      ┌─────────────────────────────────┐
      │  h1   h2   h3   ...  hd        │
      │  α1   α2   α3   ...  αd ← 独立 │
      └─────────────────────────────────┘
```

**优势**：更精确的记忆管理，选择性保留某些特征维度而遗忘其他。

### 3.2 Delta Rule + DPLR 优化

KDA 的约束 DPLR 变体：
$$\mathbf{S}_t = \left(\operatorname{Diag}(\bm{\alpha}_t) - \beta_t \bm{k}_t \bm{k}_t^{\top} \operatorname{Diag}(\bm{\alpha}_t)\right)\mathbf{S}_{t-1} + \beta_t \bm{k}_t \bm{v}_t^{\top}$$

| 操作 | 通用 DPLR | KDA | 改进 |
|------|-----------|-----|------|
| 二次 chunking 矩阵 | 4 个 | 2 个 | **减少 50%** |
| 额外矩阵乘法 | 3 个 | 0 个 | **消除 100%** |
| Kernel 速度 | 基准 | **~2x** | **提升 100%** |

### 3.3 KDA 作为可学习位置编码

KDA 的转移矩阵是**数据依赖且可学习的**，放松了 RoPE 的正交性约束：
- 解决了 RoPE 的外推问题
- MLA 层可使用 **NoPE**，简化长上下文训练

---

## 四、训练方法

### 4.1 预训练配置

| 参数 | 值 |
|------|-----|
| 总参数量 | 48B |
| 激活参数量 | 3B |
| 架构 | MoE (8/256 experts + 1 shared) |
| 训练 Token | 5.7T |
| 上下文窗口 | 4,096 tokens |
| 优化器 | MuonClip |
| 学习率 | $1.1 \times 10^{-3}$ |
| Batch Size | 32M tokens |

### 4.2 RL 配置

- 数据来源：数学、代码、STEM
- Truncated Importance Sampling：缓解策略不匹配
- 动态 KL penalty：避免熵崩溃
- PTX Loss：防止通用能力退化

---

## 五、性能基准

### 5.1 预训练结果 (1.4T Tokens)

| Benchmark | MLA | GDN-H | **Kimi Linear** |
|-----------|-----|-------|-----------------|
| MMLU | 71.6 | 72.2 | **73.8** |
| MMLU-Pro | 47.2 | 47.9 | **51.0** |
| GSM8K | 83.7 | 81.7 | **83.9** |
| MATH | **54.7** | 54.1 | **54.7** |
| CEval | 79.3 | 79.1 | **79.5** |

### 5.2 长上下文性能 (128K)

| Benchmark | MLA | GDN-H | **Kimi Linear** |
|-----------|-----|-------|-----------------|
| RULER | ~75 | ~70 | **84.3** |
| RepoQA | ~55 | ~50 | **68.5** |

### 5.3 效率对比

| 指标 | 结果 |
|------|------|
| 1M 上下文 Prefill 加速 | **2.9x** vs MLA |
| 1M 上下文 Decoding 加速 | **6x** vs MLA |
| KV Cache 减少 | **75%** |

---

## 六、线性注意力演进

```
2020  Linear Attention (Katharopoulos)
      └── 核函数替代 Softmax
2023  RetNet → 标量衰减
2023  Mamba → 数据依赖选择性
2024  Mamba2 → 标量衰减 + 矩阵状态
2024  GLA → 通道级对角门控
2025  GDN → 标量门控 + Delta Rule
2025  KDA → 通道级门控 + Delta Rule + DPLR 优化
```

---

## 七、与 MoBA 的互补关系

| 维度 | MoBA | Kimi Linear (KDA) |
|------|------|-------------------|
| 注意力形式 | 保留 Softmax，稀疏化 | 线性近似 + Delta 规则 |
| 计算复杂度 | 亚二次 $O(N \cdot B \cdot k)$ | 线性 $O(T)$ |
| 状态大小 | 完整 KV Cache (稀疏) | 固定 $d_k \times d_v$ |
| 适用场景 | 长上下文 prefill | 超长序列/流式/Agent |
| 稀疏度 | 显式可控 (top-k) | 隐式 |

**协同可能性**：
1. 分层：浅层 KDA，深层 MoBA
2. 阶段：Pre-training MoBA，继续预训练引入 KDA
3. 混合：同层内部分头使用不同机制

---

## 八、在 Kimi 技术路线中的地位

```
Kimi 技术栈:
┌─────────────────────────────────────────┐
│ 应用层: Kimi Chat / API / Agent         │
│   │                                     │
│   ▼                                     │
│ 推理优化:                                │
│   ├── MoBA (Prefill 稀疏化)              │
│   ├── Kimi Linear (混合架构)             │
│   └── vLLM 集成                         │
│   │                                     │
│   ▼                                     │
│ 模型架构:                                │
│   ├── KDA (线性注意力)                   │
│   ├── MLA (全局注意力)                   │
│   └── 3:1 混合策略                       │
│   │                                     │
│   ▼                                     │
│ 基础组件:                                │
│   ├── KDA Kernel (开源)                  │
│   └── 定制 Chunkwise 算法               │
└─────────────────────────────────────────┘
```

Kimi Linear 基于 **Moonlight** (K2 基础架构)，面向 K2.5 及后续模型的效率优化。

> [!note] 后继(2026-07-17 更新)
> KDA + 3:1 混合架构已在 **Kimi K3**(2.8T,2026-07-16 发布)成为注意力主干,且论文遗留的"MLA 加输出门"在 K3 以 **Gated MLA** 兑现——见 [[kimi_k3_architecture_deepdive]] §2-3;KDA 引发的 prefix caching 重做与 FlashKDA kernel 见 [[kimi_k3_infra_deepdive]] §3.2-3.3。

---

## Related Pages

- [[01_theory/index]]
- [[01_theory/01_models/attention_is_all_you_need_analysis]]
- [[moba_analysis]]
- [[gdn_kda_linear_attention_analysis]] — GDN/KDA 的 QKVABZ、RNN 递推与 chunk 数学等价性
- [[gdn_kda_kernel_implementation_analysis]] — 训练、Prefill、Decode 融合 kernel
- [[kimi_k3_analysis]] / [[kimi_k3_architecture_deepdive]] — KDA 在 K3 的产品化落地
