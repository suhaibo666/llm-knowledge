# Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2407.00079](https://arxiv.org/abs/2407.00079) |
| PDF | https://arxiv.org/pdf/2407.00079 |
| 提交日期 | 2024-06-24 |
| 最后更新 | 2025-09-03 |
| 主分类 | cs.DC |
| 作者 | Ruoyu Qin、Zheming Li、Weiran He　等 7 人 |
| 原文件名 | `Mooncake_KVCache_Disaggregated-2407.00079.pdf` |

## 摘要

Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI. It features a KVCache-centric disaggregated architecture that separates the prefill and decoding clusters. It also leverages the underutilized CPU, DRAM, and SSD resources of the GPU cluster to implement a disaggregated cache of KVCache. The core of Mooncake is its KVCache-centric scheduler, which balances maximizing overall effective throughput while meeting latency-related Service Level Objectives (SLOs). Unlike traditional studies that assume all requests will be processed, Mooncake faces challenges due to highly overloaded scenarios. To mitigate these, we developed a prediction-based early rejection policy. Experiments show that Mooncake excels in long-context scenarios. Compared to the baseline method, Mooncake can achieve up to a 525% increase in throughput in certain simulated scenarios while adhering to SLOs. Under real workloads, Mooncake's innovative architecture enables Kimi to handle 75% more requests.
