# 模型架构与模型家族 — 目录索引

> 覆盖 Transformer 架构、缩放定律、模型技术报告
> 最后更新: 2026-05-08

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
| [[deepseek_llm_analysis]] | DeepSeek LLM (7B/67B), GQA, 双语预训练 |
| [[deepseek_v2_analysis]] | MLA, DeepSeekMoE, 236B/21B |
| [[deepseek_v3_analysis]] | FP8 训练, 671B MoE, MTP |
| [[deepseek_v4_analysis]] | CSA/HCA, mHC, Muon, 1.6T MoE |
| [[deepseek_r1_analysis]] | 纯 RL 推理, GRPO, 冷启动蒸馏 |
| [[deepseek_coder_analysis]] | 代码预训练, 2T tokens, 项目级 FIM |
| [[deepseek_coder_v2_analysis]] | MoE 代码模型, 338 语言, GRPO |
| [[deepseek_math_analysis]] | DeepSeekMath, 120B 数学 tokens, GRPO |
| [[deepseek_math_v2]] | 自验证, Generator-Verifier, RL 微调 |
| [[deepseek_moe_analysis]] | MoE 架构, 专家路由, 负载均衡 |
| [[deepseek_prover_analysis]] | Lean 4 定理证明, truncate-and-resume |
| [[deepseek_vl_analysis]] | 视觉-语言对齐, 混合编码器 |

### Kimi / Moonshot AI

| 页面 | 核心主题 |
|------|---------|
| [[moonshot_kimi/index]] | Kimi 技术路线总览 |
| [[kimi_k1.5_analysis]] | k1.5 RL 缩放定律, 128K 上下文 RL |
| [[kimi_k2_analysis]] | K2 1.04T MoE, MuonClip, Agentic RL |
| [[kimi_k2.5_analysis]] | K2.5 视觉 Agent, MoonViT-3D, Agent Swarm |

### GLM / Zhipu AI

| 页面 | 核心主题 |
|------|---------|
| [[zhipu_glm/index]] | GLM 技术路线总览 |
| [[glm_5_analysis]] | GLM-5 744B/40B MoE, Muon Split, DSA |
| [[glm_5v_turbo_analysis]] | GLM-5V-Turbo, CogViT, MMTP, 多模态 Agent |

### LongCat / Meituan

| 页面 | 核心主题 |
|------|---------|
| [[meituan_longcat/index]] | 美团 LongCat 技术路线总览 |
| [[longcat_2_analysis]] | LongCat-2.0 1.6T/48B MoE, LSA 稀疏注意力, N-gram Embedding, ScMoE, MOPD, 国产 ASIC 全栈 |

---

## 关联域

- [[../02_pretraining/index]] — 预训练技术
- [[../04_posttraining/index]] — 后训练对齐
- [[../../02_engineering/02_train_frameworks/index]] — 训练框架
