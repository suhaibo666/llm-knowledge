# Text and Code Embeddings by Contrastive Pre-Training

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2201.10005](https://arxiv.org/abs/2201.10005) |
| PDF | https://arxiv.org/pdf/2201.10005 |
| 提交日期 | 2022-01-24 |
| 最后更新 | 2022-01-24 |
| 主分类 | cs.CL |
| 作者 | Arvind Neelakantan、Tao Xu、Raul Puri　等 25 人 |
| 原文件名 | `Text_and_Code_Embeddings_Contrastive_Pretraining-2201.10005.pdf` |

## 摘要

Text embeddings are useful features in many applications such as semantic search and computing text similarity. Previous work typically trains models customized for different use cases, varying in dataset choice, training objective and model architecture. In this work, we show that contrastive pre-training on unsupervised data at scale leads to high quality vector representations of text and code. The same unsupervised text embeddings that achieve new state-of-the-art results in linear-probe classification also display impressive semantic search capabilities and sometimes even perform competitively with fine-tuned models. On linear-probe classification accuracy averaging over 7 tasks, our best unsupervised model achieves a relative improvement of 4% and 1.8% over previous best unsupervised and supervised text embedding models respectively. The same text embeddings when evaluated on large-scale semantic search attains a relative improvement of 23.4%, 14.7%, and 10.6% over previous best unsupervised methods on MSMARCO, Natural Questions and TriviaQA benchmarks, respectively. Similarly to text embeddings, we train code embedding models on (text, code) pairs, obtaining a 20.8% relative improvement over prior best work on code search.
