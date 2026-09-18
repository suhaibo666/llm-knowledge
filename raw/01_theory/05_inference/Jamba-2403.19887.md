# Jamba: A Hybrid Transformer-Mamba Language Model

| 项 | 值 |
|---|---|
| 冻结原文 | [arXiv:2403.19887v1 PDF](https://arxiv.org/pdf/2403.19887v1) |
| 版本日期 | 2024-03-28 |
| 作者 | Opher Lieber 等 |
| 本库访问 | 2026-09-17 |

## 本批核验位置

- §1–2、Fig. 1：Transformer 注意力层与 Mamba 层按层混排；论文的 Jamba 配置还含 MoE，本批只取混合运行状态，不把教学两层模型说成发布模型。
- §2、Table 1：论文所报告的 KV cache 容量比较有模型、256K 上下文和 16-bit 条件；本批不把其容量比套用于任意混合模型。
- §3.2、Fig. 3：论文吞吐结论依赖所列模型、硬件、上下文和 batch，不把该实测用于教学状态账本。

本文件只做来源索引；不存放第三方论文 PDF。对应原理页为 T34。
