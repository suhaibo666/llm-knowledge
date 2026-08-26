# Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2310.03708](https://arxiv.org/abs/2310.03708) |
| PDF | https://arxiv.org/pdf/2310.03708 |
| 提交日期 | 2023-10-05 |
| 最后更新 | 2024-08-17 |
| 主分类 | cs.LG |
| 作者 | Zhanhui Zhou、Jie Liu、Jing Shao　等 7 人 |
| 原文件名 | `MODPO_Multi_Objective_DPO-2310.03708.pdf` |

## 摘要

A single language model, even when aligned with labelers through reinforcement learning from human feedback (RLHF), may not suit all human preferences. Recent approaches therefore prefer customization, gathering multi-dimensional feedback, and creating distinct reward models for each dimension. Different language models are then optimized for various preferences using multi-objective RLHF (MORLHF) with varying reward weights. However, RL fine-tuning is unstable and resource-heavy, especially with diverse and usually conflicting objectives. In this paper, we present Multi-Objective Direct Preference Optimization (MODPO), an RL-free extension of Direct Preference Optimization (DPO) for multiple alignment objectives. Essentially, MODPO folds language modeling directly into reward modeling, training language models as implicit collective reward models that combine all objectives with specific weights. MODPO theoretically yields the same optimal solutions as MORLHF but is practically more stable and efficient. Empirical results in safety alignment and long-form question answering show that MODPO matches or outperforms existing methods, producing a Pareto front of language models catering to diverse preferences with three times less computational resources compared to MORLHF. Code is available at https://github.com/ZHZisZZ/modpo.
