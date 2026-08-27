---
title: "腾讯混元 (Tencent Hunyuan / Hy) — 目录索引"
---

# 腾讯混元 (Tencent Hunyuan / Hy) — 目录索引

> 腾讯混元大模型家族的技术分析页。当前覆盖 2026-07 正式开源的 Hy3。
> 最后更新: 2026-07-14

---

## 分析页

| 页面 | 核心主题 |
|------|---------|
| [[hy3_analysis]] | Hy3 295B/21B MoE,架构冻结 + 纯后训练迭代,sigmoid 免辅助损失路由,三档推理模式,Apache 2.0 |
| [[stem_sparse_attention_analysis]] | Stem 免训练稀疏注意力 (arXiv 2603.06274): TPD 位置衰减预算 + OAM 输出感知选块,25% 算力近稠密精度,128K prefill 3.7× |

## 家族时间线(简)

- **Hunyuan-Large**(2024-11, arXiv 2411.02265): 389B/52B MoE,本库未单独收录
- **Hy3-preview**(2026-04 下旬): 重建基础设施后的首个模型,限制性许可
- **Hy3**(2026-07-06): 与 preview 架构逐字段一致,纯后训练升级,Apache 2.0 —— 见 [[hy3_analysis]]

## 原始文档

`raw/01_theory/01_models/tencent_hunyuan/` — README(EN/CN)、config.json、chat_template.jinja、transformers 建模代码、官方榜图 ×2(清单见 [[hy3_analysis]] 附录)

## 关联域

- [[../index]] — 模型架构与模型家族总索引
- [[../deepseek/index]] — Hy3 路由/MTP 配方的技术上游
