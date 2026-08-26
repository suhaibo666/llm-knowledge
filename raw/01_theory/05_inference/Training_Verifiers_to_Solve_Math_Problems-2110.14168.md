# Training Verifiers to Solve Math Word Problems

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) |
| PDF | https://arxiv.org/pdf/2110.14168 |
| 提交日期 | 2021-10-27 |
| 最后更新 | 2021-11-18 |
| 主分类 | cs.LG |
| 作者 | Karl Cobbe、Vineet Kosaraju、Mohammad Bavarian　等 12 人 |
| 原文件名 | `Training_Verifiers_to_Solve_Math_Problems-2110.14168.pdf` |

## 摘要

State-of-the-art language models can match human performance on many tasks, but they still struggle to robustly perform multi-step mathematical reasoning. To diagnose the failures of current models and support research, we introduce GSM8K, a dataset of 8.5K high quality linguistically diverse grade school math word problems. We find that even the largest transformer models fail to achieve high test performance, despite the conceptual simplicity of this problem distribution. To increase performance, we propose training verifiers to judge the correctness of model completions. At test time, we generate many candidate solutions and select the one ranked highest by the verifier. We demonstrate that verification significantly improves performance on GSM8K, and we provide strong empirical evidence that verification scales more effectively with increased data than a finetuning baseline.
