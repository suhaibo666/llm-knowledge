---
title: "美团 LongCat 技术路线总览"
---

# 美团 LongCat 技术路线总览

> LongCat 是美团（Meituan）开发的大语言模型系列，主打**大规模 MoE + 稀疏注意力 + 国产 AI ASIC 全栈自研**，2026 年在 Agentic Coding 方向做到近前沿。
> 最后更新: 2026-07-06（LongCat-2.0 已开源权重 + SGLang 推理码，架构描述升级为代码一手）

---

## 一、LongCat 模型家族

| 模型 | 发布 | 参数量（总/激活） | 核心特点 | 状态 |
|------|------|------------------|----------|------|
| **LongCat-Flash** | 2025-09 | **560B / ~27B**（18.6–31.3B 动态） | ScMoE 短路 + **零计算专家**首创 · MLA · MTP · H800（arXiv 2509.01322） | [[longcat_flash_analysis]] |
| **LongCat-2.0** | 2026-06 | **1.6T / ~48B**（动态） | **38 层** · MLA+LSA 稀疏注意力 · N-gram Embedding(135B) · **ScMoE 短路 + 128 零计算专家** · MOPD 多教师蒸馏 · 国产 ASIC | ✅ 已开源（权重+config+SGLang 推理码）· [[longcat_2_analysis]] |

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
| **架构** | 1.6T/48B MoE · **38 层**；注意力 = **MLA + LSA**（LSA=MLA 骨干 + DSA 式稀疏索引，index_topk 2048、SI 保 16 sink+1024 local、CLI 每 2 层复用）；N-gram 135B(16 路哈希)；**ScMoE 短路**：每层 2×(MLA+稠密FFN 12288) ∥ MoE(768 路由+128 零计算, top-12) 并行相加 |
| **预训练** | >35T tokens；Muon 大规模（TP 适配 + DP 状态去冗 + 对称矩阵乘 kernel）；数百亿 token 原生 1M |
| **后训练** | MOPD（**Multi-Teacher On-Policy Distillation**，多教师在线策略蒸馏）：训 Agent/Reasoning/Interaction 三组 teacher，对学生自身轨迹 on-policy 蒸馏融合 |
| **AI Infra** | 6D 并行（5D + EMBP 专并行 N-gram）；superpod ≤48 机 + RoCE；推理 PD 分离（CPP+SP / KVP+EP128） |
| **低精度** | **推理 FP8**（`LongCat-2.0-FP8` + bf16 KV）；训练精度未披露。国产 ASIC 侧主打**数值可靠性**（确定性算子 + 二叉树分段累加 + 精度对齐验证） |
| **稳定性** | >35T 零回滚/无不可恢复 spike；bit-flip 检测 + 端到端自动容错 |
| **效果** | 开源近前沿；SWE-bench Pro 59.5 > GPT-5.5，整体落后 Claude Opus 4.8 |

---

## 三、与同类模型的定位

| 模型 | 稀疏注意力 | 优化器 | 低精度路线 | 硬件 | Wiki |
|------|-----------|--------|-----------|------|------|
| **LongCat-2.0** | **MLA + LSA** (SI/CLI/HI) | Muon | 推理 FP8 + 数值可靠性 | **国产 ASIC** | [[longcat_2_analysis]] |
| GLM-5 | DSA | Muon Split | INT4 QAT | 多家国产 GPU | [[01_glm_5_analysis]] |
| DeepSeek-V3 | MLA | AdamW | **FP8 训练** | NVIDIA | [[12_deepseek_v3_analysis]] |
| DeepSeek-V4 | CSA/HCA | Muon | FP4 QAT | NVIDIA | [[13_deepseek_v4_analysis]] |
| Kimi K2 | MLA | MuonClip | — | NVIDIA | [[11_kimi_k2_analysis]] |

---

## 四、开源状态（截至 2026-07-06，已核实——已翻篇）

| 项 | 状态 | 说明 |
|----|------|------|
| **LongCat-2.0 权重 + config** | ✅ **已开源** | HF `meituan-longcat/LongCat-2.0`：`config.json` + **194 分片权重** + tokenizer；另有 **`LongCat-2.0-FP8`**。MIT。 |
| **推理代码** | ✅ **SGLang** | GPU 经 SGLang [PR #30042](https://github.com/sgl-project/sglang/pull/30042)（`longcat_flash.py` / `nsa_indexer.py` / `n_gram_embedding.py`）；NPU 见 SGLang-FluentLLM。注意 **HI 未在 SGLang 实现**（for simplicity）。 |
| **LongCat-Flash-Chat**（前身） | ✅ 已开源 | 79K+ 下载；ScMoE / zero-compute experts 的更早参考实现；N-gram Embedding 承袭自 **LongCat-Flash-Lite**。 |

**结论（较 2026-07-03 已翻篇）**：**LongCat-2.0 已完整开源**（权重 + `config.json` + SGLang 推理码）。本页架构描述据此**从博客二手升级为代码一手**（§1.1 参数带 config 行号、§2 结构带 `文件:行`、图 1-3 据代码绘制）。注意：**建模 forward 代码在 SGLang**（HF 仓库只有 config + 权重 + tokenizer，无 `modeling_*.py`）。

## 五、知识缺口

- ✅ **LongCat-Flash 已摄入** → [[longcat_flash_analysis]]。仍待补：**Flash-Lite**（N-gram Embedding 出处）、**Flash-Thinking**（arXiv 2509.18883，推理专精）、**Flash-Omni**（2511.00279，多模态）。
- **LongCat-2.0 训练侧仍未披露**：学习率/batch/课程、数据配比、MOPD 蒸馏损失、**是否 FP8 训练**（推理已确认 FP8）——待正式技术报告。模型**结构硬参数已由 `config.json` 补全**（见 [[longcat_2_analysis]] §1.1/§9.2）。

---

## Related Pages

- [[01_theory/01_models/index]] — 模型架构与家族总索引
- [[longcat_2_analysis]] — LongCat-2.0 深度分析
- [[01_glm_5_analysis]] · [[12_deepseek_v3_analysis]] · [[11_kimi_k2_analysis]] — 同期大模型对照
- [[01_theory/index]] — 理论研究总览
