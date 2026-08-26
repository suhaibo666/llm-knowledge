# Gated Delta Networks: Improving Mamba2 with Delta Rule

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) |
| PDF | https://arxiv.org/pdf/2412.06464 |
| 提交日期 | 2024-12-09 |
| 最后更新 | 2025-03-06 |
| 主分类 | cs.CL |
| 作者 | Songlin Yang、Jan Kautz、Ali Hatamizadeh |
| 原文件名 | `Gated_Delta_Networks-2412.06464v3.pdf` |

## 摘要

Linear Transformers have gained attention as efficient alternatives to standard Transformers, but their performance in retrieval and long-context tasks has been limited. To address these limitations, recent work has explored two distinct mechanisms: gating for adaptive memory control and the delta update rule for precise memory modifications. We observe that these mechanisms are complementary: gating enables rapid memory erasure while the delta rule facilitates targeted updates. Building on this insight, we introduce the gated delta rule and develop a parallel training algorithm optimized for modern hardware. Our proposed architecture, Gated DeltaNet, consistently surpasses existing models like Mamba2 and DeltaNet across multiple benchmarks, including language modeling, common-sense reasoning, in-context retrieval, length extrapolation, and long-context understanding. We further enhance performance by developing hybrid architectures that combine Gated DeltaNet layers with sliding window attention or Mamba2 layers, achieving both improved training efficiency and superior task performance.
