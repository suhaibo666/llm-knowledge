# CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

> ⚠️ **ID 更正**：原文件名中的 `2204.14230` 实际对应无关论文《Cohomological boundedness for flat bundles on surfaces and applications》，正确 arXiv ID 为 `2205.15868`（已按论文标题在 arXiv 检索核对）。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2205.15868](https://arxiv.org/abs/2205.15868) |
| PDF | https://arxiv.org/pdf/2205.15868 |
| 提交日期 | 2022-05-29 |
| 最后更新 | 2022-05-29 |
| 主分类 | cs.CV |
| 作者 | Wenyi Hong、Ming Ding、Wendi Zheng　等 5 人 |
| 原文件名 | `CogVideo-2204.14230.pdf` |

## 摘要

Large-scale pretrained transformers have created milestones in text (GPT-3) and text-to-image (DALL-E and CogView) generation. Its application to video generation is still facing many challenges: The potential huge computation cost makes the training from scratch unaffordable; The scarcity and weak relevance of text-video datasets hinder the model understanding complex movement semantics. In this work, we present 9B-parameter transformer CogVideo, trained by inheriting a pretrained text-to-image model, CogView2. We also propose multi-frame-rate hierarchical training strategy to better align text and video clips. As (probably) the first open-source large-scale pretrained text-to-video model, CogVideo outperforms all publicly available models at a large margin in machine and human evaluations.
