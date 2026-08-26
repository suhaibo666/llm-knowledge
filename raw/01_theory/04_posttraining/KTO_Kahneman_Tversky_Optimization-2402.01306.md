# KTO: Model Alignment as Prospect Theoretic Optimization

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2402.01306](https://arxiv.org/abs/2402.01306) |
| PDF | https://arxiv.org/pdf/2402.01306 |
| 提交日期 | 2024-02-02 |
| 最后更新 | 2024-11-19 |
| 主分类 | cs.LG |
| 作者 | Kawin Ethayarajh、Winnie Xu、Niklas Muennighoff　等 5 人 |
| 原文件名 | `KTO_Kahneman_Tversky_Optimization-2402.01306.pdf` |

## 摘要

Kahneman & Tversky's $\textit{prospect theory}$ tells us that humans perceive random variables in a biased but well-defined manner (1992); for example, humans are famously loss-averse. We show that objectives for aligning LLMs with human feedback implicitly incorporate many of these biases -- the success of these objectives (e.g., DPO) over cross-entropy minimization can partly be ascribed to them belonging to a family of loss functions that we call $\textit{human-aware losses}$ (HALOs). However, the utility functions these methods attribute to humans still differ from those in the prospect theory literature. Using a Kahneman-Tversky model of human utility, we propose a HALO that directly maximizes the utility of generations instead of maximizing the log-likelihood of preferences, as current methods do. We call this approach KTO, and it matches or exceeds the performance of preference-based methods at scales from 1B to 30B, despite only learning from a binary signal of whether an output is desirable. More broadly, our work suggests that there is no one HALO that is universally superior; the best loss depends on the inductive biases most appropriate for a given setting, an oft-overlooked consideration.
