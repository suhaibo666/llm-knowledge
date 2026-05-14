# 训练框架 — 目录索引

> 覆盖分布式训练框架、并行策略、通信优化
> 最后更新: 2026-05-08

---

## 子目录

| 目录 | 核心主题 |
|------|---------|
| [[megatron-lm/index]] | NVIDIA Megatron-LM, 5D 并行, MoE, TFLOPS, 通信掩盖 |

## 页面列表

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[megatron-lm/index]] | Megatron-LM 源码 | 分布式并行、通信优化、MoE |
| [[llm_parallelism_analysis]] | Megatron-LM 源码验证 | 正反向 DAG, TP/SP/EP/CP 通信依赖 |
| [[mooncake_analysis]] | Mooncake (Kimi) | 分离式推理架构, 分布式 KV Cache, RDMA |
| [[distributed_optimizer_deep_dive.html]] | 综合深度分析 | FSDP2/ZeRO/MindSpeed 对比, 梯度累积, Adam vs Muon |
| [[deepseek_v4_tensor_parallel_analysis.html]] | Megatron-LM dev 源码 | DeepSeek-V4 TP 切分方案, CSA/HCA/MoE/mHC 通信量与 Overlap |

---

## 原始素材

`raw/02_engineering/02_train_frameworks/`:

| 文件 | 主题 |
|------|------|
| `megatron.eddx` | Megatron 训练框架架构图 |
| `mindformers.eddx` | MindFormers 训练框架架构图 |

---

## 关联域

- [[../01_ai_frameworks/index]] — AI框架 (PyTorch 编译栈)
- [[../03_infer_frameworks/index]] — 推理框架
- [[../../01_theory/02_pretraining/index]] — 预训练技术
- [[../../01_theory/04_posttraining/index]] — 后训练算法
