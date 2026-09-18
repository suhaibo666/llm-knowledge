# Fast Inference from Transformers via Speculative Decoding

> 本库仅保留来源索引；论文原文由 arXiv 提供。取证日期：2026-09-17。

| 项 | 值 |
|---|---|
| 作者 | Yaniv Leviathan、Matan Kalman、Yossi Matias |
| 版本 | [arXiv:2211.17192v2](https://arxiv.org/abs/2211.17192v2)，2023-05-18 |
| 原始 PDF | [官方 PDF](https://arxiv.org/pdf/2211.17192v2) |
| 本库取证范围 | §2.1 的 draft/target 两阶段与额外 token；§2.2 的采样分布标准化；§2.3、Algorithm 1 的顺序接受、首次拒绝、正差残差和全接受 bonus；§3.1–3.4 的接受率、输出长度和理想墙钟模型；§3.6 的小模型和 n-gram 草稿；Appendix A.1 的分布保持证明。 |

本索引不代替原文；T23 的 KV 逻辑提交边界还需要结合自回归因果语义推导，不能误称论文给出了特定引擎的缓存 API。
