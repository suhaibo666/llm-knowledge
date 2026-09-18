# Visual Instruction Tuning (LLaVA)

> 本库仅保留来源索引；论文原文由 arXiv 提供。取证日期：2026-09-17。

| 项 | 值 |
|---|---|
| 版本 | [arXiv:2304.08485v2](https://arxiv.org/abs/2304.08485v2)，2023-12-11 |
| 原始 PDF | [官方 PDF](https://arxiv.org/pdf/2304.08485v2) |
| 本库取证范围 | §4.1、Fig. 1、Eq. (1)：CLIP 视觉编码器产生网格特征，线性投影为语言 embedding 维度的视觉 token 序列，作为视觉与语言连接的一种架构。 |

原论文的线性投影不在序列维压缩网格；单个文本图像标记不能代表视觉编码器的网格特征数。
