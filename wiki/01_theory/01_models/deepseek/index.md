# DeepSeek 模型家族 — 目录索引

> 覆盖 DeepSeek LLM、V2、V3、V4、R1、Coder、Math、MoE、VL 等全系列
> 最后更新: 2026-06-25（V4 全系列对正式版 arXiv:2606.19348v1 审计/订正）

---

## 页面列表

### 核心 LLM 系列

| 页面 | 核心主题 |
|------|---------|
| [[deepseek_llm_analysis]] | DeepSeek LLM (7B/67B), 缩放定律, 多步 LR, GQA |
| [[deepseek_v2_analysis]] | DeepSeek-V2, MLA, DeepSeekMoE, 236B/21B active |
| [[deepseek_v3_analysis]] | DeepSeek-V3, FP8 训练, 671B MoE, DualPipe, MTP |
| [[deepseek_v4_analysis]] | DeepSeek-V4, CSA/HCA 混合注意力, 百万 token 上下文, 1.6T MoE |

### V4 专题分析

> **基线**: 已全部对正式发表版 **arXiv:2606.19348v1 (2026-04-26)** 核对 —— 见 [[deepseek_v4_audit_report]]（2026-06-25）。

| 页面 | 核心主题 | 核对状态 |
|------|---------|---------|
| [[deepseek_v4_audit_report]] | **审计报告**：以正式版核对全部 V4 页面，列出一致项与订正项 | ★ 入口 |
| [[deepseek_v4_cp_analysis]] | V4 Context Parallelism + packed sequences | ✅ 已订正章节号 |
| [[deepseek_v4_fp4_qat_analysis]] | V4 FP4 量化感知训练（后训练 §5.2.1） | ✅ 已订正出处/口径 |
| [[mHC]] | 流形约束超连接（源自 mHC 论文，V4 §2.2 采用） | ✅ 一致 |
| [[deepseek_v4_technical_deep_dive]] | CSA/HCA/DSA/MLA 对比深度解析 | ✅ 已据正式版整页重写 |
| [[deepseek_v4_implementation_details]] | V4 核心组件伪代码（CSA/HCA/mHC/Muon/MoE，逐方程） | ✅ 已据正式版整页重写 |
| [[deepseek_v4_architecture_diagrams]] | V4 结构图（Figure 2/3/4 复刻） | ✅ 已据正式版整页重画 |

### 推理与代码模型

| 页面 | 核心主题 |
|------|---------|
| [[deepseek_r1_analysis]] | DeepSeek-R1, 纯 RL 推理, GRPO, cold start, 蒸馏 |
| [[deepseek_coder_analysis]] | DeepSeek-Coder, 代码预训练, 2T tokens, FIM |
| [[deepseek_coder_v2_analysis]] | DeepSeek-Coder-V2, MoE 代码模型, 338 语言, 128K |
| [[deepseek_math_analysis]] | DeepSeekMath 7B, 120B 数学 tokens, GRPO 起源 |
| [[deepseek_math_v2]] | DeepSeekMath-V2, 自验证, Generator-Verifier 循环 |

### 架构与多模态

| 页面 | 核心主题 |
|------|---------|
| [[deepseek_moe_analysis]] | DeepSeek-MoE 架构, 细粒度专家, 负载均衡 |
| [[deepseek_prover_analysis]] | DeepSeek-Prover-V1.5, Lean 4 定理证明 |
| [[deepseek_vl_analysis]] | DeepSeek-VL, 视觉语言对齐 |
| [[Engram_Analysis]] | 条件记忆, N-gram 哈希, 记忆稀疏性 |

---

## 关联域

- [[06_infra/megatron-lm/index]] — Megatron-LM（DeepSeek 使用 Megatron 训练）
- [[03_alignment/index]] — 对齐方法（GRPO, DPO 等）
- [[02_training/index]] — 低精度训练（FP8, FP4 QAT）
