# Muon is Scalable for LLM Training

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2502.16982](https://arxiv.org/abs/2502.16982) |
| PDF | https://arxiv.org/pdf/2502.16982 |
| 提交日期 | 2025-02-24 |
| 最后更新 | 2025-02-24 |
| 主分类 | cs.LG |
| 作者 | Jingyuan Liu、Jianlin Su、Xingcheng Yao　等 28 人 |
| 原文件名 | `MUON IS SCALABLE FOR LLM TRAINING-2502.16982v1.pdf` |

## 摘要

Recently, the Muon optimizer based on matrix orthogonalization has demonstrated strong results in training small-scale language models, but the scalability to larger models has not been proven. We identify two crucial techniques for scaling up Muon: (1) adding weight decay and (2) carefully adjusting the per-parameter update scale. These techniques allow Muon to work out-of-the-box on large-scale training without the need of hyper-parameter tuning. Scaling law experiments indicate that Muon achieves $\sim\!2\times$ computational efficiency compared to AdamW with compute optimal training. Based on these improvements, we introduce Moonlight, a 3B/16B-parameter Mixture-of-Expert (MoE) model trained with 5.7T tokens using Muon. Our model improves the current Pareto frontier, achieving better performance with much fewer training FLOPs compared to prior models. We open-source our distributed Muon implementation that is memory optimal and communication efficient. We also release the pretrained, instruction-tuned, and intermediate checkpoints to support future research.
