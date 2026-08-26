# Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2402.14740](https://arxiv.org/abs/2402.14740) |
| PDF | https://arxiv.org/pdf/2402.14740 |
| 提交日期 | 2024-02-22 |
| 最后更新 | 2024-02-26 |
| 主分类 | cs.LG |
| 作者 | Arash Ahmadian、Chris Cremer、Matthias Gallé　等 8 人 |
| 原文件名 | `RLOO_REINFORCE_Leave_One_Out-2402.14740.pdf` |

## 摘要

AI alignment in the shape of Reinforcement Learning from Human Feedback (RLHF) is increasingly treated as a crucial ingredient for high performance large language models. Proximal Policy Optimization (PPO) has been positioned by recent literature as the canonical method for the RL part of RLHF. However, it involves both high computational cost and sensitive hyperparameter tuning. We posit that most of the motivational principles that led to the development of PPO are less of a practical concern in RLHF and advocate for a less computationally expensive method that preserves and even increases performance. We revisit the formulation of alignment from human preferences in the context of RL. Keeping simplicity as a guiding principle, we show that many components of PPO are unnecessary in an RLHF context and that far simpler REINFORCE-style optimization variants outperform both PPO and newly proposed "RL-free" methods such as DPO and RAFT. Our work suggests that careful adaptation to LLMs alignment characteristics enables benefiting from online RL optimization at low cost.
