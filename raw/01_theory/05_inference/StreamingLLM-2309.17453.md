# Efficient Streaming Language Models with Attention Sinks

| 项 | 值 |
|---|---|
| 冻结原文 | [arXiv:2309.17453v2 HTML](https://arxiv.org/html/2309.17453v2) |
| 版本日期 | 2023-11-21 |
| 作者 | Guangxuan Xiao、Yuandong Tian、Beidi Chen、Song Han、Mike Lewis |
| 本库访问 | 2026-09-17 |

## 本批核验位置

- §3.1、Fig. 1–3：只保留近期窗口会在移走起始 token 后出现质量问题；attention sink 的观察。
- §3.2：保留初始 sink 与最近窗口的滚动 KV，位置语义仍须一致。
- §4.1、§4.3：语言建模与流式问答各自的评价口径；无限流处理不等于检索任意久远事实。

本文件只做来源索引；不存放第三方论文 PDF。对应原理页为 T21。
