---
title: "模型架构与模型家族 — 目录索引"
---

# 模型架构与模型家族 — 目录索引

> 覆盖 Transformer 架构、缩放定律、模型技术报告
> 最后更新: 2026-08-13

---

## 架构论文

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[attention_is_all_you_need_analysis]] | Attention Is All You Need | Transformer 架构、多头注意力、位置编码 |
| [[scaling_laws_analysis]] | Scaling Laws | 幂律缩放、计算最优训练、临界批次大小 |
| [[long_context_scaling_law_analysis]] | Long Context Scaling Law | 长上下文互信息缩放、Transformer vs SSM |

## 模型家族

### DeepSeek

| 页面 | 核心主题 |
|------|---------|
| [[deepseek/index]] | DeepSeek 模型家族总览 |
| [[10_deepseek_llm_analysis]] | DeepSeek LLM (7B/67B), GQA, 双语预训练 |
| [[11_deepseek_v2_analysis]] | MLA, DeepSeekMoE, 236B/21B |
| [[12_deepseek_v3_analysis]] | FP8 训练, 671B MoE, MTP |
| [[13_deepseek_v4_analysis]] | CSA/HCA, mHC, Muon, 1.6T MoE |
| [[14_deepseek_r1_analysis]] | 纯 RL 推理, GRPO, 冷启动蒸馏 |
| [[15_deepseek_coder_analysis]] | 代码预训练, 2T tokens, 项目级 FIM |
| [[16_deepseek_coder_v2_analysis]] | MoE 代码模型, 338 语言, GRPO |
| [[17_deepseek_math_analysis]] | DeepSeekMath, 120B 数学 tokens, GRPO |
| [[18_deepseek_math_v2_analysis]] | 自验证, Generator-Verifier, RL 微调 |
| [[20_deepseek_moe_analysis]] | MoE 架构, 专家路由, 负载均衡 |
| [[22_deepseek_prover_analysis]] | Lean 4 定理证明, truncate-and-resume |
| [[21_deepseek_vl_analysis]] | 视觉-语言对齐, 混合编码器 |

### Kimi / Moonshot AI

| 页面 | 核心主题 |
|------|---------|
| [[moonshot_kimi/index]] | Kimi 技术路线总览 |
| [[29_kimi_k1_5_analysis]] | k1.5 RL 缩放定律, 128K 上下文 RL |
| [[11_kimi_k2_analysis]] | K2 1.04T MoE, MuonClip, Agentic RL |
| [[13_kimi_k2_5_analysis]] | K2.5 视觉 Agent, MoonViT-3D, Agent Swarm |
| [[12_kimi_linear_analysis]] | Kimi Linear 48B/3B, KDA 线性注意力, 3:1 混合, K3 注意力前身 |
| [[20_gdn_kda_linear_attention_analysis]] | GDN/KDA: QKVABZ、矩阵 RNN、门控 Delta Rule、chunk 数学等价性 |
| [[21_gdn_kda_kernel_implementation_analysis]] | GDN/KDA 训练、Prefill、Decode 融合 kernel 与源码调用链 |
| [[14_kimi_k3_analysis]] | K3 2.8T MoE(896 选 16), 首个开源 3T 级, KDA+AttnRes, 1M 上下文, 2.5× scaling 效率 |
| [[22_kimi_k3_architecture_deepdive]] | K3 六大结构变化点: KDA 3:1 / Gated MLA / AttnRes / Stable LatentMoE / SiTU(源码级溯源) |
| [[23_kimi_k3_infra_deepdive]] | K3 训推 infra: Per-Head Muon, MXFP4/MXFP8 QAT, 全平衡 EP, Mooncake, KDA prefix cache, 64+ 卡超节点 |

### GLM / Zhipu AI

| 页面 | 核心主题 |
|------|---------|
| [[zhipu_glm/index]] | GLM 技术路线总览 |
| [[01_glm_5_analysis]] | GLM-5 744B/40B MoE, Muon Split, DSA |
| [[10_glm_5v_turbo_analysis]] | GLM-5V-Turbo, CogViT, MMTP, 多模态 Agent |

### Qwen / 阿里巴巴

| 页面 | 核心主题 |
|------|---------|
| [[01_theory/01_models/alibaba_qwen/index|Qwen 技术路线总览]] | Qwen 家族报告分析与开放权重审计入口 |
| [[10_qwen3_8_analysis]] | Qwen3.8 2.4T/95B，3:1 Gated DeltaNet/Attention，512 选 10+1 MoE，真实工作 RL，Max endpoint 与开放 checkpoint 边界 |

### Hunyuan / 腾讯混元

| 页面 | 核心主题 |
|------|---------|
| [[tencent_hunyuan/index]] | 腾讯混元模型家族总览 |
| [[hy3_analysis]] | Hy3 295B/21B MoE,架构冻结+纯后训练迭代,sigmoid 免辅助损失路由,三档推理模式 |
| [[stem_sparse_attention_analysis]] | Stem 免训练稀疏注意力,TPD+OAM,25% 算力近稠密,128K prefill 3.7× |

### Thinking Machines Lab

| 页面 | 核心主题 |
|------|---------|
| [[thinking_machines/index]] | Thinking Machines(Mira Murati)模型家族总览 |
| [[inkling_analysis]] | Inkling 975B/41B 多模态 MoE,抛 RoPE/MLA、滑窗-全局 5:1 交错、SConv、encoder-free 四模态、8 层 MTP、Muon、Apache 2.0 |

### LongCat / Meituan

| 页面 | 核心主题 |
|------|---------|
| [[meituan_longcat/index]] | 美团 LongCat 技术路线总览 |
| [[longcat_flash_analysis]] | LongCat-Flash 560B/27B MoE, ScMoE 短路 + 零计算专家(首创), MLA, H800 |
| [[longcat_2_analysis]] | LongCat-2.0 1.6T/48B MoE, LSA 稀疏注意力, N-gram Embedding, ScMoE, MOPD, 国产 ASIC 全栈 |

---

## Knowledge Gaps

> 本节记录已确认在当前 wiki 中缺失的一手来源摄入项。按 CLAUDE.md 的 Query Workflow，这些缺口不应用推测填补；若日后摄入对应 raw 材料，应先建页再补链接。

- scaling_laws_for_transfer_analysis — 待摄入(raw 有源)

---

## 关联域

- [[../02_pretraining/index]] — 预训练技术
- [[../04_posttraining/index]] — 后训练对齐
- [[../../02_engineering/02_train_frameworks/index]] — 训练框架
