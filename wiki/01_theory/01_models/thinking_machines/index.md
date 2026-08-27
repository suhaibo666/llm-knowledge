---
title: "Thinking Machines Lab — 目录索引"
---

# Thinking Machines Lab — 目录索引

> Mira Murati(前 OpenAI CTO)创立的 Thinking Machines Lab 的模型技术分析页。当前覆盖 2026-07-15 首发的开源模型 Inkling。
> 最后更新: 2026-07-16

---

## 分析页

| 页面 | 核心主题 |
|------|---------|
| [[inkling_analysis]] | Inkling 975B/41B 多模态 MoE:抛 RoPE(学习相对 PE)/抛 MLA(滑窗-全局 5:1 交错)/掺 SConv/encoder-free 四模态/8 层 MTP/Muon 训练;Apache 2.0,赌"Tinker 可定制底座"而非榜首 |

## 家族说明

- **Inkling**(2026-07-15): 潜行 18 个月后的首个模型,开源权重(BF16 + NVFP4)。旗舰 975B/41B,另有 **Inkling-Small** 预览(276B 总参 / 12B 激活)。见 [[inkling_analysis]]
- 变现路径 = **Tinker** 微调服务,非模型本身

## 原始文档

`raw/01_theory/01_models/thinking_machines/` — config.json、HF 模型卡、官方公告快照(清单见 [[inkling_analysis]] 附录)

## 关联域

- [[../index]] — 模型架构与模型家族总索引
- [[../deepseek/index]] — Inkling 路由配方的部分技术上游
