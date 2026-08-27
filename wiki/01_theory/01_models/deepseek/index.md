# DeepSeek 模型家族 — 目录索引

> 覆盖 DeepSeek LLM、V2、V3、V4、R1、Coder、Math、MoE、VL 等全系列
> 最后更新: 2026-07-31（kb-reorg P7 Task 7:目录内分段编号 + 命名统一——`mHC`→`mhc_analysis`、`Engram_Analysis`→`engram_analysis`、`deepseek_math_v2`补 `_analysis` 后缀、`_deep_dive`→`_deepdive`、`_report`/`_diagrams`/`_details` 非法后缀就近改 `_analysis`/`_deepdive`）

---

## 页面列表

> **段位与阅读顺序**(kb-reorg P7 Task 7,2026-07-31):段 1(10-19)核心机制主线——DeepSeek 旗舰模型按发布序排列(基础 LLM 世系 LLM→V2→V3→V4,再接同代衍生旗舰 R1/Coder/Coder-V2/Math/Math-V2);段 2(20-29)深潜/专题——架构组件与衍生变体的专项深挖(MoE/VL/Prover 等特化模型、V4 专属机制 CP/FP4-QAT/mHC/结构图/伪代码/技术对比);段 3(30-39)方法论——V4 全系列审计报告(核对正式版 arXiv,是 V4 专题分析的入口锚点)。

### 段 1 · 核心 LLM 系列(10-19)

| 页面 | 核心主题 |
|------|---------|
| [[10_deepseek_llm_analysis]] | DeepSeek LLM (7B/67B), 缩放定律, 多步 LR, GQA |
| [[11_deepseek_v2_analysis]] | DeepSeek-V2, MLA, DeepSeekMoE, 236B/21B active |
| [[12_deepseek_v3_analysis]] | DeepSeek-V3, FP8 训练, 671B MoE, DualPipe, MTP |
| [[13_deepseek_v4_analysis]] | DeepSeek-V4, CSA/HCA 混合注意力, 百万 token 上下文, 1.6T MoE |
| [[14_deepseek_r1_analysis]] | DeepSeek-R1, 纯 RL 推理, GRPO, cold start, 蒸馏 |
| [[15_deepseek_coder_analysis]] | DeepSeek-Coder, 代码预训练, 2T tokens, FIM |
| [[16_deepseek_coder_v2_analysis]] | DeepSeek-Coder-V2, MoE 代码模型, 338 语言, 128K |
| [[17_deepseek_math_analysis]] | DeepSeekMath 7B, 120B 数学 tokens, GRPO 起源 |
| [[18_deepseek_math_v2_analysis]] | DeepSeekMath-V2, 自验证, Generator-Verifier 循环 |

### 段 2 · 架构组件与 V4 专题深挖(20-29)

> V4 专题分析基线:已全部对正式发表版 **arXiv:2606.19348v1 (2026-04-26)** 核对 —— 见 [[30_deepseek_v4_audit_analysis]]（2026-06-25）。
> **并已进一步对发布权重核对**：`DeepSeek-V4-Pro-0813` / `DeepSeek-V4-Flash-0731` 的 `config.json` 与论文 §4.2.1 超参**十四项全中**，见 [[31_deepseek_v4_released_checkpoints_analysis]]（2026-08-27）。

| 页面 | 核心主题 | 核对状态 |
|------|---------|---------|
| [[20_deepseek_moe_analysis]] | DeepSeek-MoE 架构, 细粒度专家, 负载均衡 | — |
| [[21_deepseek_vl_analysis]] | DeepSeek-VL, 视觉语言对齐 | — |
| [[22_deepseek_prover_analysis]] | DeepSeek-Prover-V1.5, Lean 4 定理证明 | — |
| [[23_deepseek_v4_cp_analysis]] | V4 Context Parallelism + packed sequences | ✅ 已订正章节号 |
| [[24_deepseek_v4_fp4_qat_analysis]] | V4 FP4 量化感知训练（后训练 §5.2.1） | ✅ 已订正出处/口径 |
| [[25_mhc_analysis]] | 流形约束超连接（源自 mHC 论文，V4 §2.2 采用） | ✅ 一致 |
| [[26_deepseek_v4_technical_deepdive]] | CSA/HCA/DSA/MLA 对比深度解析 | ✅ 已据正式版整页重写 |
| [[27_deepseek_v4_implementation_deepdive]] | V4 核心组件伪代码（CSA/HCA/mHC/Muon/MoE，逐方程） | ✅ 已据正式版整页重写 |
| [[28_deepseek_v4_architecture_analysis]] | V4 结构图（Figure 2/3/4 复刻） | ✅ 已据正式版整页重画 |
| [[29_engram_analysis]] | 条件记忆, N-gram 哈希, 记忆稀疏性 | — |

### 段 3 · 方法论(30-39)

| 页面 | 核心主题 | 核对状态 |
|------|---------|---------|
| [[30_deepseek_v4_audit_analysis]] | **审计报告**：以正式版核对全部 V4 页面，列出一致项与订正项 | ★ 入口 |
| [[31_deepseek_v4_released_checkpoints_analysis]] | **权重对账**：以发布 checkpoint（Pro-0813 / Flash-0731）核对论文超参；新增 DSpark/FP4/参数量三项事实 | ★ 续篇 |

---

## 关联域

- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM]] — Megatron-LM（DeepSeek 使用 Megatron 训练）
- [[01_theory/04_posttraining/index|对齐/后训练]] — 对齐方法（GRPO, DPO 等）
- [[01_theory/02_pretraining/index|预训练/低精度]] — 低精度训练（FP8, FP4 QAT）
