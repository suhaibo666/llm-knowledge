# DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

> ⚠️ **ID 更正**：原文件名中的 `2412.10322` 实际对应无关论文《Investigating SU(3) with Nf=8 fundamental fermions at strong renormalized coupling》，正确 arXiv ID 为 `2412.10302`（已按论文标题在 arXiv 检索核对）。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2412.10302](https://arxiv.org/abs/2412.10302) |
| PDF | https://arxiv.org/pdf/2412.10302 |
| 提交日期 | 2024-12-13 |
| 最后更新 | 2024-12-13 |
| 主分类 | cs.CV |
| 作者 | Zhiyu Wu、Xiaokang Chen、Zizheng Pan　等 27 人 |
| 原文件名 | `DeepSeek_VL2-2412.10322.pdf` |

## 摘要

We present DeepSeek-VL2, an advanced series of large Mixture-of-Experts (MoE) Vision-Language Models that significantly improves upon its predecessor, DeepSeek-VL, through two key major upgrades. For the vision component, we incorporate a dynamic tiling vision encoding strategy designed for processing high-resolution images with different aspect ratios. For the language component, we leverage DeepSeekMoE models with the Multi-head Latent Attention mechanism, which compresses Key-Value cache into latent vectors, to enable efficient inference and high throughput. Trained on an improved vision-language dataset, DeepSeek-VL2 demonstrates superior capabilities across various tasks, including but not limited to visual question answering, optical character recognition, document/table/chart understanding, and visual grounding. Our model series is composed of three variants: DeepSeek-VL2-Tiny, DeepSeek-VL2-Small and DeepSeek-VL2, with 1.0B, 2.8B and 4.5B activated parameters respectively. DeepSeek-VL2 achieves competitive or state-of-the-art performance with similar or fewer activated parameters compared to existing open-source dense and MoE-based models. Codes and pre-trained models are publicly accessible at https://github.com/deepseek-ai/DeepSeek-VL2.
