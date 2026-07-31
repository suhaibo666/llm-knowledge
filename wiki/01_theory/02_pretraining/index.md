# LLM 训练技术 — 目录索引

> 覆盖优化器、权重初始化、低精度训练、激活重计算、数值精度一致性
> 最后更新: 2026-07-31（kb-reorg P7 Task 7:目录内分段编号 + `RL_Training_Inference_Precision_Analysis` 改小写）

---

## 页面列表

> **段位与阅读顺序**(kb-reorg P7 Task 7,2026-07-31):段 1(10-19)核心机制主线——按训练流水线顺序:初始化→优化器→激活重计算→低精度→TE 落地实现;段 2(20-29)深潜/专题——RL 训练/推理数值一致性(跨训练-推理两个系统的专项交叉话题)。

| 页面 | 层次 | 来源 | 核心主题 |
|------|------|------|---------|
| [[10_llm_initiliaze_analysis]] | 核心机制(段 1) | 综合（代码分析） | 权重初始化 (Xavier/He), 残差缩放, MoE 专家初始化 |
| [[11_muon_analysis]] | 核心机制(段 1) | Muon Is Scalable for LLM Training (2502.16982) | Muon 优化器, Newton-Schulz 迭代, 谱范数正交化 |
| [[12_activation_checkpointing_analysis]] | 核心机制(段 1) | PyTorch + Megatron-LM 源码 | 激活重计算, CheckpointFunction, Full/Selective 策略 |
| [[13_low_precision_training_analysis]] | 核心机制(段 1) | Megatron-LM + DeepSeek-V4 | FP8 Recipe 体系, FP4 QAT, MoE + 低精度 |
| [[14_transformer_engine_analysis]] | 核心机制(段 1) | NVIDIA TE GitHub 仓库 | TE 架构, 精度格式, Quantizer, CommOverlap |
| [[20_rl_training_inference_precision_analysis]] | 深潜(段 2) | TorchTitan + vLLM | RL 训练推理数值一致性, 确定性 Forward |

---

## 关联域

- [[01_theory/01_models/index|模型家族]] — 基础架构（Transformer, 缩放定律）
- [[01_theory/04_posttraining/index|对齐/后训练]] — 对齐方法（RLHF, DPO, GRPO）
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM]] — Megatron-LM 训练基础设施
- [[02_engineering/01_ai_frameworks/index|AI 框架]] — PyTorch 编译栈
