# Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding

> 本库仅保留来源索引；论文原文由 arXiv 提供。取证日期：2026-09-17。

| 项 | 值 |
|---|---|
| 作者 | Jun Zhang、Jue Wang、Huan Li、Lidan Shou、Ke Chen、Gang Chen、Sharad Mehrotra |
| 版本 | [arXiv:2309.08168v2](https://arxiv.org/abs/2309.08168v2)，2024-05-20 |
| 原始 PDF | [官方 PDF](https://arxiv.org/pdf/2309.08168v2) |
| 本库取证范围 | §3.2、Algorithm 2 的跳层自草拟和完整模型 greedy 校验；§3.3 的跳层选择；§3.4 的自适应草稿提前停止。 |

Algorithm 2 标题明确为 **Greedy**；不能把该版本的确定性一致性直接扩展为任意随机采样分布保持。
