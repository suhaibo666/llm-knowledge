# VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2504.05118](https://arxiv.org/abs/2504.05118) |
| PDF | https://arxiv.org/pdf/2504.05118 |
| 提交日期 | 2025-04-07 |
| 最后更新 | 2025-04-11 |
| 主分类 | cs.AI |
| 作者 | Yu Yue、Yufeng Yuan、Qiying Yu　等 27 人 |
| 原文件名 | `VAPO_Value_Augmented_Proximal_Policy_Optimization-2504.05118.pdf` |

## 摘要

We present VAPO, Value-based Augmented Proximal Policy Optimization framework for reasoning models., a novel framework tailored for reasoning models within the value-based paradigm. Benchmarked the AIME 2024 dataset, VAPO, built on the Qwen 32B pre-trained model, attains a state-of-the-art score of $\mathbf{60.4}$. In direct comparison under identical experimental settings, VAPO outperforms the previously reported results of DeepSeek-R1-Zero-Qwen-32B and DAPO by more than 10 points. The training process of VAPO stands out for its stability and efficiency. It reaches state-of-the-art performance within a mere 5,000 steps. Moreover, across multiple independent runs, no training crashes occur, underscoring its reliability. This research delves into long chain-of-thought (long-CoT) reasoning using a value-based reinforcement learning framework. We pinpoint three key challenges that plague value-based methods: value model bias, the presence of heterogeneous sequence lengths, and the sparsity of reward signals. Through systematic design, VAPO offers an integrated solution that effectively alleviates these challenges, enabling enhanced performance in long-CoT reasoning tasks.
