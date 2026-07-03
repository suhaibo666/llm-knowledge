# 美团 LongCat 技术路线总览

> LongCat 是美团（Meituan）开发的大语言模型系列，主打**大规模 MoE + 稀疏注意力 + 国产 AI ASIC 全栈自研**，2026 年在 Agentic Coding 方向做到近前沿。
> 最后更新: 2026-07-02

---

## 一、LongCat 模型家族

| 模型 | 发布 | 参数量（总/激活） | 核心特点 | 状态 |
|------|------|------------------|----------|------|
| **LongCat-Flash** | 2025 | — MoE（含 **zero-compute experts** 动态激活） | 计算-通信重叠、动态算力分配 | 架构前身（待摄入） |
| **LongCat-2.0** | 2026-06 | **1.6T / ~48B** | LSA 稀疏注意力 · N-gram Embedding(135B) · ScMoE · MOPD 多教师蒸馏 · 全栈国产 ASIC | [[longcat_2_analysis]] |

---

## 二、LongCat-2.0 一页速览

**主线**：证明「前沿规模训练 + 近前沿 Agentic Coding」可完全建立在**国产 AI ASIC superpod**上（50K+ 卡、非 NVIDIA），>35T tokens 预训练**零回滚**。详见 [[longcat_2_analysis]]。

```
架构三创新                    训练/后训练               Infra/可靠性
─────────────                 ───────────               ────────────
LSA 稀疏注意力 (SI/CLI/HI)     Muon 大规模 (TP/DP/kernel) 6D 并行 (+EMBP)
N-gram Embedding 135B/n=5      原生 1M (CP 512+)          PD 分离 + EP128
ScMoE (per-core 全并行)        MOPD 三教师蒸馏            确定性算子 + 二叉树累加
MTP 3-step (复用 LSA 索引)     (Agent/Reasoning/Interact) bit-flip 检测 + 自动容错
```

| 维度 | 一句话 |
|------|--------|
| **架构** | 1.6T/48B MoE；LSA 三正交索引压 1M 上下文；N-gram 在「稀疏维」廉价扩参 135B；ScMoE 把计算-通信从重叠推到全并行 |
| **预训练** | >35T tokens；Muon 大规模（TP 适配 + DP 状态去冗 + 对称矩阵乘 kernel）；数百亿 token 原生 1M |
| **后训练** | MOPD：训 Agent/Reasoning/Interaction 三组 teacher，蒸馏融合最强能力 |
| **AI Infra** | 6D 并行（5D + EMBP 专并行 N-gram）；superpod ≤48 机 + RoCE；推理 PD 分离（CPP+SP / KVP+EP128） |
| **低精度** | **不**讲 FP8/FP4；主打国产 ASIC 上的**数值可靠性**（确定性算子 + 二叉树分段累加 + 精度对齐验证） |
| **稳定性** | >35T 零回滚/无不可恢复 spike；bit-flip 检测 + 端到端自动容错 |
| **效果** | 开源近前沿；SWE-bench Pro 59.5 > GPT-5.5，整体落后 Claude Opus 4.8 |

---

## 三、与同类模型的定位

| 模型 | 稀疏注意力 | 优化器 | 低精度路线 | 硬件 | Wiki |
|------|-----------|--------|-----------|------|------|
| **LongCat-2.0** | LSA (SI/CLI/HI) | Muon | 数值可靠性（非量化） | **国产 ASIC** | [[longcat_2_analysis]] |
| GLM-5 | DSA | Muon Split | INT4 QAT | 多家国产 GPU | [[glm_5_analysis]] |
| DeepSeek-V3 | MLA | AdamW | **FP8 训练** | NVIDIA | [[deepseek_v3_analysis]] |
| DeepSeek-V4 | CSA/HCA | Muon | FP4 QAT | NVIDIA | [[deepseek_v4_analysis]] |
| Kimi K2 | MLA | MuonClip | — | NVIDIA | [[kimi_k2_analysis]] |

---

## 四、知识缺口

- **LongCat-Flash**（架构前身，含 zero-compute experts 动态激活机制）尚未摄入——待补技术报告。
- **LongCat-2.0** 的层数/隐藏维/每层专家数/top-k、训练课程、是否 FP8、MOPD 具体 RL 算法**均未由官方博客披露**；权重与 config.json「coming soon」、正式技术报告未见。待 raw 源到位后回填精确基线（见 [[longcat_2_analysis]] §9）。

---

## Related Pages

- [[01_theory/01_models/index]] — 模型架构与家族总索引
- [[longcat_2_analysis]] — LongCat-2.0 深度分析
- [[glm_5_analysis]] · [[deepseek_v3_analysis]] · [[kimi_k2_analysis]] — 同期大模型对照
- [[01_theory/index]] — 理论研究总览
