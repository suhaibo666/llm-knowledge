# Stem: Rethinking Causal Information Flow in Sparse Attention

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2603.06274](https://arxiv.org/abs/2603.06274) |
| PDF | https://arxiv.org/pdf/2603.06274 |
| 提交日期 | 2026-03-06 |
| 最后更新 | 2026-07-31 |
| 主分类 | cs.LG |
| 作者 | Lin Niu、Xin Luo、Linchuan Xie　等 7 人 |
| 原文件名 | `Stem_Sparse_Attention-2603.06274.pdf` |

## 摘要

The quadratic computational complexity of self-attention remains a fundamental bottleneck for scaling Large Language Models (LLMs) to long contexts, particularly during the pre-filling phase. In this paper, we rethink the causal attention mechanism from the perspective of information flow. Due to causal constraints, tokens at initial positions participate in the aggregation of every subsequent token. However, existing sparse methods typically apply a uniform top-k selection across all token positions within a layer, ignoring the cumulative dependency of token information inherent in causal architectures. To address this, we propose Stem, a novel, plug-and-play sparsity module aligned with information flow. First, Stem employs the Token Position-Decay strategy, applying position-dependent top-k within each layer to retain initial tokens for recursive dependencies. Second, to preserve information-rich tokens, Stem utilizes the Output-Aware Metric. It prioritizes high-impact tokens based on approximate output magnitude. Extensive evaluations demonstrate that Stem achieves superior accuracy with reduced computation and pre-filling latency.
