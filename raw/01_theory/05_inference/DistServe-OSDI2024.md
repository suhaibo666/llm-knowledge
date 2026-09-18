# DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

| 项 | 值 |
|---|---|
| 正式版本 | [OSDI 2024，pp. 193–210，USENIX 原文](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) |
| 作者 | Yinmin Zhong、Shengyu Liu、Junda Chen、Jianbo Hu、Yibo Zhu、Xuanzhe Liu、Xin Jin、Hao Zhang |
| 本库访问 | 2026-09-17 |

## 本批核验位置

- §2.3：同卡 prefill/decode 干扰、分块与分离的决策条件。
- §3.1–3.3：按 TTFT/TPOT 目标分别选择资源与并行方案；真实长度变化和传输。
- §4.1–4.3：跨节点与节点内放置、decode 侧 pull KV、突发时缓冲。
- §5、Fig. 6：运行时编排、KV 传输和两个计算池。
- §6.3、Fig. 10：论文给定模型、工作负载和网络下的传输成本，不能外推为普遍可忽略。

本文件只做来源索引；不存放第三方论文 PDF。对应原理页为 T26。
