# Fine-Tuning Language Models from Human Preferences

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:1909.08593](https://arxiv.org/abs/1909.08593) |
| PDF | https://arxiv.org/pdf/1909.08593 |
| 提交日期 | 2019-09-18 |
| 最后更新 | 2020-01-08 |
| 主分类 | cs.CL |
| 作者 | Daniel M. Ziegler、Nisan Stiennon、Jeffrey Wu　等 8 人 |
| 原文件名 | `Fine_Tuning_Language_Models_from_Human_Preferences-1909.08593.pdf` |

## 摘要

Reward learning enables the application of reinforcement learning (RL) to tasks where reward is defined by human judgment, building a model of reward by asking humans questions. Most work on reward learning has used simulated environments, but complex information about values is often expressed in natural language, and we believe reward learning for language is a key to making RL practical and safe for real-world tasks. In this paper, we build on advances in generative pretraining of language models to apply reward learning to four natural language tasks: continuing text with positive sentiment or physically descriptive language, and summarization tasks on the TL;DR and CNN/Daily Mail datasets. For stylistic continuation we achieve good results with only 5,000 comparisons evaluated by humans. For summarization, models trained with 60,000 comparisons copy whole sentences from the input but skip irrelevant preamble; this leads to reasonable ROUGE scores and very good performance according to our human labelers, but may be exploiting the fact that labelers rely on simple heuristics.
