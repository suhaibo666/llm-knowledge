# 分布式训练基础设施 — 目录索引

> 覆盖 Megatron-LM、并行策略、通信优化、分布式系统
> 最后更新: 2026-05-07

---

## 子目录

| 目录 | 核心主题 |
|------|---------|
| [[megatron-lm/index]] | NVIDIA Megatron-LM, 5D 并行, MoE, TFLOPS, 通信掩盖 |

## 页面列表

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[llm_parallelism_analysis]] | Megatron-LM 源码验证 | 正反向 DAG, TP/SP/EP/CP 通信依赖, 计算通信重叠 |
| [[mooncake_analysis]] | Mooncake (Moonshot/Kimi) | 分离式推理架构, 分布式 KV Cache, RDMA |

---

## 关联域

- [[../02_training/low_precision_training_analysis]] — 低精度训练
- [[../02_training/transformer_engine_analysis]] — Transformer Engine
- [[../03_alignment/index]] — 对齐方法
- [[../../torch_compile/index]] — PyTorch 编译栈
