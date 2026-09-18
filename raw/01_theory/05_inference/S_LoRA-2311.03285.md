# S-LoRA: Serving Thousands of Concurrent LoRA Adapters

> 本库仅保留来源索引；论文原文由 arXiv 提供。取证日期：2026-09-17。

| 项 | 值 |
|---|---|
| 作者 | Ying Sheng、Shiyi Cao、Dacheng Li、Coleman Hooper、Nicholas Lee、Shuo Yang、Christopher Chou、Banghua Zhu、Lianmin Zheng、Kurt Keutzer、Joseph E. Gonzalez、Ion Stoica |
| 版本 | [arXiv:2311.03285v1](https://arxiv.org/abs/2311.03285v1)，2023-11-06 |
| 原始 PDF | [官方 PDF](https://arxiv.org/pdf/2311.03285v1) |
| 本库取证范围 | §2 的 LoRA 行向量表示；§4.1、Fig. 1–2 的共享基座、分离低秩增量和主存／GPU 活跃 adapter；§4.2 的聚类取舍；§5.1–5.2 的 Unified Paging、预取和驻留；§5.3 的异 rank、非连续内存 MBGMM/MBGMV。 |

本索引不代替原文；S-LoRA 的统一分页属于该系统方案，不代表其他引擎都共同分页 adapter 与 KV。
