# Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills

> 本库仅保留来源索引；论文原文由 arXiv 提供。访问日期：2026-09-17。

| 项 | 值 |
|---|---|
| 作者 | Amey Agrawal、Ashish Panwar、Jayashree Mohan、Nipun Kwatra、Bhargav S. Gulavani、Ramachandran Ramjee |
| 版本 | [arXiv:2308.16369v1](https://arxiv.org/abs/2308.16369v1)，2023-08-31，已读 arXiv HTML v1 |
| 原始 HTML | [Sarathi arXiv HTML v1](https://arxiv.org/html/2308.16369v1) |
| 原始 PDF | [Sarathi arXiv PDF v1](https://arxiv.org/pdf/2308.16369v1) |
| 本库取证范围 | §3.1–3.3：本论文实验下 prefill/decode 的利用率差异及其边界；§4.2、Fig. 6：chunk 间因果 mask、数学等价和重读历史 KV 的开销；§4.3：decode-maximal batching；§4.4：chunk 大小与 prefill 效率、可搭载 decode 数的取舍；§5：本论文实测的条件和范围。 |

本索引不代替论文正文。T16 的通用分块语义与论文系统策略分开书写；本文论文的 A6000、LLaMA-13B 和吞吐数字不可泛化到其他模型、硬件或引擎。
