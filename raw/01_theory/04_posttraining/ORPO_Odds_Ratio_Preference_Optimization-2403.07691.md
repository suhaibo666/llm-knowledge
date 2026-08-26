# ORPO: Monolithic Preference Optimization without Reference Model

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2403.07691](https://arxiv.org/abs/2403.07691) |
| PDF | https://arxiv.org/pdf/2403.07691 |
| 提交日期 | 2024-03-12 |
| 最后更新 | 2024-03-14 |
| 主分类 | cs.CL |
| 作者 | Jiwoo Hong、Noah Lee、James Thorne |
| 原文件名 | `ORPO_Odds_Ratio_Preference_Optimization-2403.07691.pdf` |

## 摘要

While recent preference alignment algorithms for language models have demonstrated promising results, supervised fine-tuning (SFT) remains imperative for achieving successful convergence. In this paper, we study the crucial role of SFT within the context of preference alignment, emphasizing that a minor penalty for the disfavored generation style is sufficient for preference-aligned SFT. Building on this foundation, we introduce a straightforward and innovative reference model-free monolithic odds ratio preference optimization algorithm, ORPO, eliminating the necessity for an additional preference alignment phase. We demonstrate, both empirically and theoretically, that the odds ratio is a sensible choice for contrasting favored and disfavored styles during SFT across the diverse sizes from 125M to 7B. Specifically, fine-tuning Phi-2 (2.7B), Llama-2 (7B), and Mistral (7B) with ORPO on the UltraFeedback alone surpasses the performance of state-of-the-art language models with more than 7B and 13B parameters: achieving up to 12.20% on $\text{AlpacaEval}_{2.0}$ (Figure 1), 66.19% on IFEval (instruction-level loose, Table 6), and 7.32 in MT-Bench (Figure 12). We release code and model checkpoints for Mistral-ORPO-$α$ (7B) and Mistral-ORPO-$β$ (7B).
