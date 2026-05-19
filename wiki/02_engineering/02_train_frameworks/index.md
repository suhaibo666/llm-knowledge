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
| [[comm_compute_fusion_guide]] | 综合深度分析 | 通算融合: WaveEP、DeepEP、TP/DP/PP/CP 各维度重叠, 自动化路线图 |
| [[distributed_optimizer_deep_dive.html]] | 综合深度分析 | FSDP2/ZeRO/MindSpeed 对比, 梯度累积, Adam vs Muon |
| [[deepseek_v4_tensor_parallel_analysis.html]] | Megatron-LM dev 源码 | DeepSeek-V4 TP 切分方案, CSA/HCA/MoE/mHC 通信量与 Overlap |
| [[deepseek_v4_context_parallel_analysis.html]] | Megatron-LM dev 源码 | DeepSeek-V4 CP 实现, 4 种通信类型, Native/TE CP, Dynamic CP, MLA 通信量优化 |
| [[megatron_pp_parallelism_analysis.html]] | Megatron-LM 源码 | PP 并行: 1F1B/VPP/Combined 调度, P2P 通信, Bubble 分析, 激活优化与卸载 |
| [[comm_compute_overlap_analysis.html]] | Megatron-LM / torchtitan 源码 | 计算通信掩盖: combined_1f1b vs ZBV/DualPipe, sub-layer 级调度, DeepEP/HybridEP |
| [[async_collective_tensor_deep_dive.html]] | PyTorch 源码 (_functional_collectives.py) | ACT 源码追踪: __torch_dispatch__, wait_tensor, stream 级执行过程, 与 Megatron 手动 stream 对比 |
| [[muon_sharded_hsdp_report.html]] | Cursor Composer 2.5 博客 | 分片 Muon + 双网格 HSDP: all-to-all N-S、EP/CP 解耦、异步流水线、非专家分工优化 |

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
