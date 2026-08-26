# A General Theoretical Paradigm to Understand Learning from Human Preferences

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2310.12036](https://arxiv.org/abs/2310.12036) |
| PDF | https://arxiv.org/pdf/2310.12036 |
| 提交日期 | 2023-10-18 |
| 最后更新 | 2023-11-22 |
| 主分类 | cs.AI |
| 作者 | Mohammad Gheshlaghi Azar、Mark Rowland、Bilal Piot　等 7 人 |
| 原文件名 | `IPO_Identity_Preference_Optimization-2310.12036.pdf` |

## 摘要

The prevalent deployment of learning from human preferences through reinforcement learning (RLHF) relies on two important approximations: the first assumes that pairwise preferences can be substituted with pointwise rewards. The second assumes that a reward model trained on these pointwise rewards can generalize from collected data to out-of-distribution data sampled by the policy. Recently, Direct Preference Optimisation (DPO) has been proposed as an approach that bypasses the second approximation and learn directly a policy from collected data without the reward modelling stage. However, this method still heavily relies on the first approximation. In this paper we try to gain a deeper theoretical understanding of these practical algorithms. In particular we derive a new general objective called $Ψ$PO for learning from human preferences that is expressed in terms of pairwise preferences and therefore bypasses both approximations. This new general objective allows us to perform an in-depth analysis of the behavior of RLHF and DPO (as special cases of $Ψ$PO) and to identify their potential pitfalls. We then consider another special case for $Ψ$PO by setting $Ψ$ simply to Identity, for which we can derive an efficient optimisation procedure, prove performance guarantees and demonstrate its empirical superiority to DPO on some illustrative examples.
