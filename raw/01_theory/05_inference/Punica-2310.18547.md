# Punica: Multi-Tenant LoRA Serving

> 本库仅保留来源索引；论文原文由 arXiv 提供。取证日期：2026-09-17。

| 项 | 值 |
|---|---|
| 作者 | Lequn Chen、Zihao Ye、Yongji Wu、Danyang Zhuo、Luis Ceze、Arvind Krishnamurthy |
| 版本 | [arXiv:2310.18547v1](https://arxiv.org/abs/2310.18547v1)，2023-10-28 |
| 原始 PDF | [官方 PDF](https://arxiv.org/pdf/2310.18547v1) |
| 本库取证范围 | §2.2 的低秩表示；§3 的共享基座和请求 adapter 身份；§4、Fig. 3–4 的 SGMV、shrink/expand 与分组；§5.2 的按需异步装载；§6 的 batch 输入重排和 segment 索引。 |

本索引不代替原文；论文的 kernel 和调度选择是 Punica 实现，不是所有 Multi-LoRA 服务的通用 API。
