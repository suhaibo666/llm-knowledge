# L$^2$M: Mutual Information Scaling Law for Long-Context Language Modeling

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2503.04725](https://arxiv.org/abs/2503.04725) |
| PDF | https://arxiv.org/pdf/2503.04725 |
| 提交日期 | 2025-03-06 |
| 最后更新 | 2025-10-24 |
| 主分类 | cs.CL |
| 作者 | Zhuo Chen、Oriol Mayné i Comas、Zhuotao Jin　等 5 人 |
| 原文件名 | `Long_Context_Scaling_Law-2503.04725.pdf` |

## 摘要

We present a universal theoretical framework for understanding long-context language modeling based on a bipartite mutual information scaling law that we rigorously verify in natural language. We demonstrate that bipartite mutual information captures multi-token interactions distinct from and scaling independently of conventional two-point mutual information, and show that this provides a more complete characterization of the dependencies needed for accurately modeling long sequences. Leveraging this scaling law, we formulate the Long-context Language Modeling (L$^2$M) condition, which lower bounds the necessary scaling of a model's history state -- the latent variables responsible for storing past information -- for effective long-context modeling. We validate the framework and its predictions on transformer and state-space models. Our work provides a principled foundation to understand long-context modeling and to design more efficient architectures with stronger long-context capabilities, with potential applications beyond natural language.
