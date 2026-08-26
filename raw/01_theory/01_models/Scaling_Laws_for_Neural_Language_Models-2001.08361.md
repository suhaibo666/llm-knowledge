# Scaling Laws for Neural Language Models

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) |
| PDF | https://arxiv.org/pdf/2001.08361 |
| 提交日期 | 2020-01-23 |
| 最后更新 | 2020-01-23 |
| 主分类 | cs.LG |
| 作者 | Jared Kaplan、Sam McCandlish、Tom Henighan　等 10 人 |
| 原文件名 | `Scaling_Laws_for_Neural_Language_Models-2001.08361.pdf` |

## 摘要

We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude. Other architectural details such as network width or depth have minimal effects within a wide range. Simple equations govern the dependence of overfitting on model/dataset size and the dependence of training speed on model size. These relationships allow us to determine the optimal allocation of a fixed compute budget. Larger models are significantly more sample-efficient, such that optimally compute-efficient training involves training very large models on a relatively modest amount of data and stopping significantly before convergence.
