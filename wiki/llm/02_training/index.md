# LLM 训练技术 — 目录索引

> 覆盖优化器、权重初始化、低精度训练、激活重计算、数值精度一致性
> 最后更新: 2026-05-07

---

## 页面列表

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[muon_analysis]] | Muon Is Scalable for LLM Training (2502.16982) | Muon 优化器, Newton-Schulz 迭代, 谱范数正交化 |
| [[llm_initiliaze_analysis]] | 综合（代码分析） | 权重初始化 (Xavier/He), 残差缩放, MoE 专家初始化 |
| [[activation_checkpointing_analysis]] | PyTorch + Megatron-LM 源码 | 激活重计算, CheckpointFunction, Full/Selective 策略 |
| [[low_precision_training_analysis]] | Megatron-LM + DeepSeek-V4 | FP8 Recipe 体系, FP4 QAT, MoE + 低精度 |
| [[transformer_engine_analysis]] | NVIDIA TE GitHub 仓库 | TE 架构, 精度格式, Quantizer, CommOverlap |
| [[RL_Training_Inference_Precision_Analysis]] | TorchTitan + vLLM | RL 训练推理数值一致性, 确定性 Forward |

---

## 关联域

- [[../01_architecture/index]] — 基础架构（Transformer, 缩放定律）
- [[../03_alignment/index]] — 对齐方法（RLHF, DPO, GRPO）
- [[../06_infra/megatron-lm/index]] — Megatron-LM 训练基础设施
- [[../../torch_compile/index]] — PyTorch 编译栈
