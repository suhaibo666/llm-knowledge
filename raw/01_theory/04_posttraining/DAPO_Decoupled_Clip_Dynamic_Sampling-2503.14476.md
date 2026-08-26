# DAPO: An Open-Source LLM Reinforcement Learning System at Scale

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| PDF | https://arxiv.org/pdf/2503.14476 |
| 提交日期 | 2025-03-18 |
| 最后更新 | 2025-05-20 |
| 主分类 | cs.LG |
| 作者 | Qiying Yu、Zheng Zhang、Ruofei Zhu　等 35 人 |
| 原文件名 | `DAPO_Decoupled_Clip_Dynamic_Sampling-2503.14476.pdf` |

## 摘要

Inference scaling empowers LLMs with unprecedented reasoning ability, with reinforcement learning as the core technique to elicit complex reasoning. However, key technical details of state-of-the-art reasoning LLMs are concealed (such as in OpenAI o1 blog and DeepSeek R1 technical report), thus the community still struggles to reproduce their RL training results. We propose the $\textbf{D}$ecoupled Clip and $\textbf{D}$ynamic s$\textbf{A}$mpling $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{DAPO}$) algorithm, and fully open-source a state-of-the-art large-scale RL system that achieves 50 points on AIME 2024 using Qwen2.5-32B base model. Unlike previous works that withhold training details, we introduce four key techniques of our algorithm that make large-scale LLM RL a success. In addition, we open-source our training code, which is built on the verl framework, along with a carefully curated and processed dataset. These components of our open-source system enhance reproducibility and support future research in large-scale LLM RL.
