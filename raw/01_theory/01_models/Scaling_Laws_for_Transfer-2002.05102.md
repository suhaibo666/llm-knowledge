# Scaling Laws for Transfer

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

> ⚠️ **ID 更正**：原文件名中的 `2002.05102` 实际对应无关论文《Hurwitz Actions on Reflection Factorizations in Complex Reflection Group $G_6$》，正确 arXiv ID 为 `2102.01293`（已按论文标题在 arXiv 检索核对）。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2102.01293](https://arxiv.org/abs/2102.01293) |
| PDF | https://arxiv.org/pdf/2102.01293 |
| 提交日期 | 2021-02-02 |
| 最后更新 | 2021-02-02 |
| 主分类 | cs.LG |
| 作者 | Danny Hernandez、Jared Kaplan、Tom Henighan　等 4 人 |
| 原文件名 | `Scaling_Laws_for_Transfer-2002.05102.pdf` |

## 摘要

We study empirical scaling laws for transfer learning between distributions in an unsupervised, fine-tuning setting. When we train increasingly large neural networks from-scratch on a fixed-size dataset, they eventually become data-limited and stop improving in performance (cross-entropy loss). When we do the same for models pre-trained on a large language dataset, the slope in performance gains is merely reduced rather than going to zero. We calculate the effective data "transferred" from pre-training by determining how much data a transformer of the same size would have required to achieve the same loss when training from scratch. In other words, we focus on units of data while holding everything else fixed. We find that the effective data transferred is described well in the low data regime by a power-law of parameter count and fine-tuning dataset size. We believe the exponents in these power-laws correspond to measures of the generality of a model and proximity of distributions (in a directed rather than symmetric sense). We find that pre-training effectively multiplies the fine-tuning dataset size. Transfer, like overall performance, scales predictably in terms of parameters, data, and compute.
