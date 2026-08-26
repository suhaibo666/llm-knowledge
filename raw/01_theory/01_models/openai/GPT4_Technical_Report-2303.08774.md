# GPT-4 Technical Report

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2303.08774](https://arxiv.org/abs/2303.08774) |
| PDF | https://arxiv.org/pdf/2303.08774 |
| 提交日期 | 2023-03-15 |
| 最后更新 | 2024-03-04 |
| 主分类 | cs.CL |
| 作者 |  OpenAI、Josh Achiam、Steven Adler　等 281 人 |
| 原文件名 | `GPT4_Technical_Report-2303.08774.pdf` |

## 摘要

We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer-based model pre-trained to predict the next token in a document. The post-training alignment process results in improved performance on measures of factuality and adherence to desired behavior. A core component of this project was developing infrastructure and optimization methods that behave predictably across a wide range of scales. This allowed us to accurately predict some aspects of GPT-4's performance based on models trained with no more than 1/1,000th the compute of GPT-4.
