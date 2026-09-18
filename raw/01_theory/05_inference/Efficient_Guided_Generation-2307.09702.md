# Efficient Guided Generation for LLMs

> 本库仅保留来源索引；论文原文由 arXiv 提供。取证日期：2026-09-17。

| 项 | 值 |
|---|---|
| 作者 | Brandon T. Willard、Rémi Louf |
| 版本 | [arXiv:2307.09702v1](https://arxiv.org/abs/2307.09702v1)，2023-07-19 |
| 原始 PDF | [官方 PDF](https://arxiv.org/pdf/2307.09702v1) |
| 本库取证范围 | §1 的格式约束问题；§2 的 token 抽样与引导生成；§3 的正则表达式有限状态机、词表索引和状态转移；§4 的迭代解析/CFG 扩展。 |

正则/FSM 路径只用于适用的正规语言；递归结构所需的 CFG/PDA 不能被简单等同于一个普通有限状态机。
