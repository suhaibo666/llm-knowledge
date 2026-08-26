# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| PDF | https://arxiv.org/pdf/2402.03300 |
| 提交日期 | 2024-02-05 |
| 最后更新 | 2024-04-27 |
| 主分类 | cs.CL |
| 作者 | Zhihong Shao、Peiyi Wang、Qihao Zhu　等 11 人 |
| 原文件名 | `DeepSeek_Math-2402.03300.pdf` |

## 摘要

Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH. The mathematical reasoning capability of DeepSeekMath is attributed to two key factors: First, we harness the significant potential of publicly available web data through a meticulously engineered data selection pipeline. Second, we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO.
