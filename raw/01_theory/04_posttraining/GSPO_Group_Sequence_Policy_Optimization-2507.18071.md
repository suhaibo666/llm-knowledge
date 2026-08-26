# Group Sequence Policy Optimization

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2507.18071](https://arxiv.org/abs/2507.18071) |
| PDF | https://arxiv.org/pdf/2507.18071 |
| 提交日期 | 2025-07-24 |
| 最后更新 | 2025-07-28 |
| 主分类 | cs.LG |
| 作者 | Chujie Zheng、Shixuan Liu、Mingze Li　等 12 人 |
| 原文件名 | `GSPO_Group_Sequence_Policy_Optimization-2507.18071.pdf` |

## 摘要

This paper introduces Group Sequence Policy Optimization (GSPO), our stable, efficient, and performant reinforcement learning algorithm for training large language models. Unlike previous algorithms that adopt token-level importance ratios, GSPO defines the importance ratio based on sequence likelihood and performs sequence-level clipping, rewarding, and optimization. We demonstrate that GSPO achieves superior training efficiency and performance compared to the GRPO algorithm, notably stabilizes Mixture-of-Experts (MoE) RL training, and has the potential for simplifying the design of RL infrastructure. These merits of GSPO have contributed to the remarkable improvements in the latest Qwen3 models.
