# Learning Transferable Visual Models From Natural Language Supervision

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |
| PDF | https://arxiv.org/pdf/2103.00020 |
| 提交日期 | 2021-02-26 |
| 最后更新 | 2021-02-26 |
| 主分类 | cs.CV |
| 作者 | Alec Radford、Jong Wook Kim、Chris Hallacy　等 12 人 |
| 原文件名 | `CLIP_Learning_Transferable_Visual_Models-2103.00020.pdf` |

## 摘要

State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained model weights at https://github.com/OpenAI/CLIP.
