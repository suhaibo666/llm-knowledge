# Let's Verify Step by Step

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) |
| PDF | https://arxiv.org/pdf/2305.20050 |
| 提交日期 | 2023-05-31 |
| 最后更新 | 2023-05-31 |
| 主分类 | cs.LG |
| 作者 | Hunter Lightman、Vineet Kosaraju、Yura Burda　等 10 人 |
| 原文件名 | `Lets_Verify_Step_by_Step-2305.20050.pdf` |

## 摘要

In recent years, large language models have greatly improved in their ability to perform complex multi-step reasoning. However, even state-of-the-art models still regularly produce logical mistakes. To train more reliable models, we can turn either to outcome supervision, which provides feedback for a final result, or process supervision, which provides feedback for each intermediate reasoning step. Given the importance of training reliable models, and given the high cost of human feedback, it is important to carefully compare the both methods. Recent work has already begun this comparison, but many questions still remain. We conduct our own investigation, finding that process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset. Our process-supervised model solves 78% of problems from a representative subset of the MATH test set. Additionally, we show that active learning significantly improves the efficacy of process supervision. To support related research, we also release PRM800K, the complete dataset of 800,000 step-level human feedback labels used to train our best reward model.
