# Flamingo: a Visual Language Model for Few-Shot Learning

> 本库仅保留来源索引；论文原文由 arXiv 提供。取证日期：2026-09-17。

| 项 | 值 |
|---|---|
| 版本 | [arXiv:2204.14198v1](https://arxiv.org/abs/2204.14198v1)，2022-04-29 |
| 原始 PDF | [官方 PDF](https://arxiv.org/pdf/2204.14198v1) |
| 本库取证范围 | §3.1.1、Fig. 4 的视觉特征与 Perceiver Resampler；§3.1.2、Fig. 5 的门控 cross-attention；§3.1.3、Fig. 6 的交错媒体/文本可见性和图像位置映射。 |

Flamingo 的视觉 latent 作为 cross-attention 的 K/V，不应机械地计为普通文本 decoder 自注意力的前缀 token。
