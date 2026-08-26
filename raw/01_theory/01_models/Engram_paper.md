# Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

> 原文件名未含 arXiv ID；经正文标题核对为 DeepSeek-AI × 北京大学的 Engram 论文。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2601.07372](https://arxiv.org/abs/2601.07372) |
| PDF | https://arxiv.org/pdf/2601.07372 |
| 提交日期 | 2026-01-12 |
| 最后更新 | 2026-07-12 |
| 主分类 | cs.CL |
| 作者 | Xin Cheng、Rui Tian、Wangding Zeng　等 21 人 |
| 原文件名 | `Engram_paper.pdf` |
| 代码 | https://github.com/deepseek-ai/Engram |

## 摘要

While Mixture-of-Experts (MoE) scales capacity via conditional computation, Transformers lack a native primitive for knowledge lookup, forcing them to inefficiently simulate retrieval through computation. To address this, we introduce conditional memory as a complementary sparsity axis, instantiated via Engram, a module that modernizes classic $N$-gram embedding for O(1) lookup. By formulating the Sparsity Allocation problem, we uncover a U-shaped scaling law that optimizes the trade-off between neural computation (MoE) and static memory (Engram). Guided by this law, we scale Engram to 27B parameters, achieving superior performance over a strictly iso-parameter and iso-FLOPs MoE baseline. Most notably, while the memory module is expected to aid knowledge retrieval (e.g., MMLU +3.4; CMMLU +4.0), we observe even larger gains in general reasoning (e.g., BBH +5.0; ARC-Challenge +3.7) and code/math domains~(HumanEval +3.0; MATH +2.4). Mechanistic analyses reveal that Engram relieves the backbone's early layers from static reconstruction, effectively deepening the network for complex reasoning. Furthermore, by delegating local dependencies to lookups, it frees up attention capacity for global context, substantially boosting long-context retrieval (e.g., Multi-Query NIAH: 84.2 to 97.0). Finally, Engram establishes infrastructure-aware efficiency: its deterministic addressing enables runtime prefetching from host memory, incurring negligible overhead. We envision conditional memory as an indispensable modeling primitive for next-generation sparse models.
