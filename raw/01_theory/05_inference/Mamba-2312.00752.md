# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

| 项 | 值 |
|---|---|
| 冻结原文 | [arXiv:2312.00752v1 PDF](https://arxiv.org/pdf/2312.00752v1) |
| 版本日期 | 2023-12-01 |
| 作者 | Albert Gu、Tri Dao |
| 本库访问 | 2026-09-17 |

## 本批核验位置

- §2、Eqs. (2a)–(2b)：离散循环状态按输入逐步更新；自回归推理使用 recurrent mode。
- §3.2、Algorithm 2：选择性 SSM 的参数依赖输入，整个序列的更新是 time-varying recurrence/scan。
- §3.3.2、Fig. 1：scan 在训练/长序列计算中避免向 HBM 写出每个中间扩展状态；这不自动提供任意 token 的回退快照。
- §1、§3.4、Fig. 3：纯 Mamba 每步不需要随历史线性增长的 KV；Mamba block 还包含短卷积路径，完整运行状态不能只用教学标量表示。

本文件只做来源索引；不存放第三方论文 PDF。对应原理页为 T34。
