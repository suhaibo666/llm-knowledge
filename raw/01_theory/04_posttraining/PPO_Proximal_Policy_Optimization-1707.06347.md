# Proximal Policy Optimization Algorithms

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| PDF | https://arxiv.org/pdf/1707.06347 |
| 提交日期 | 2017-07-20 |
| 最后更新 | 2017-08-28 |
| 主分类 | cs.LG |
| 作者 | John Schulman、Filip Wolski、Prafulla Dhariwal　等 5 人 |
| 原文件名 | `PPO_Proximal_Policy_Optimization-1707.06347.pdf` |

## 摘要

We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a "surrogate" objective function using stochastic gradient ascent. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates. The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically). Our experiments test PPO on a collection of benchmark tasks, including simulated robotic locomotion and Atari game playing, and we show that PPO outperforms other online policy gradient methods, and overall strikes a favorable balance between sample complexity, simplicity, and wall-time.
