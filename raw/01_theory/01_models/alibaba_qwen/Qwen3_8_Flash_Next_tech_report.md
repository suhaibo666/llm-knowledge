# On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability

> **本库不随仓分发第三方论文原文**：迁至 GitHub 公开仓库后，`raw/` 只保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| 标题 | On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability |
| 作者 | Qwen Team（Core Contributors: Zihan Qiu、Zekun Wang、Xiao Li、Yanpeng Li、Yang Xu、Yixuan Wang、Huaqing Zhang、Rui Men、Bo Zheng、Dayiheng Liu，另 24 位 Contributors） |
| 日期 | 2026-08-26（PDF 首页页眉） |
| PDF | https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf |
| 官方博客 | https://qwen.ai/blog?id=qwen3.8-flash-next |
| 模型卡 | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)（快照见 `Qwen3_8_Flash_Next_model_card_f5d08274.md`） |
| 无 arXiv 编号 | 该报告以 PDF 形式随 GitHub 仓库发布，未上 arXiv |
| 篇幅 | 28 页；11 张表、13 张图 |
| 摄入日期 | 2026-08-27 |

> [!note] 标题与模型名不一致
> 报告标题写的是 **Qwen3.8-**Next**** 架构，正文与模型卡则一律称 **Qwen3.8-Flash-**Next****。
> 前者指这套将支撑 Qwen4 的架构本身，后者是该架构下的首个开放权重模型（正文 p1：
> "the first open-weight release under this architecture"）。

## 摘要

We describe the architecture and ablations of Qwen3.8-Flash-Next, a sparse mixture-of-experts model with
125B parameters, 6B activated per token, and additional 51B parameters of n-gram embedding tables held
off the accelerator. On fourteen pre-training benchmarks the model leads the 397B-A17B predecessor on
eight and trails it on the rest by at most 2.6 points, at 1/3 the activated parameters, 1/3 the training tokens,
and roughly 1/9 the training FLOPs. Token mixing uses a layer-wise hybrid of Gated DeltaNet (GDN) and
global attention, with one full-attention layer in every four; at continued-pretraining time those full-attention
layers are replaced by Qwen Sparse Attention (QSA), which scores context at micro-block granularity with a
compressed lightweight indexer. The residual stream is widened to four branches and read through an
elementwise gate, a design we call the Gated Residual (GR). Capacity is added outside the backbone by a
single n-gram embedding layer whose tables are prefetched from host memory. We evaluate every candidate
change along three axes: loss together with downstream benchmarks; the cost of the change in training,
prefill and decode; and its effect on the optimal hyperparameters and training stability. Loss and downstream
accuracy do not always move together: enlarging the n-gram vocabulary lowers loss monotonically while
downstream accuracy saturates. The architecture and the Muon optimizer together shift the optimal learning
rate and batch size upwards, render batch-size warmup unnecessary, and substantially improve stability
under stress tests. Loss, benchmarks, efficiency and stability form one design problem. Solved jointly, they
yield a recipe that is simultaneously more efficient, more capable and more stable.

## 章节结构（供定位）

| 章节 | 页 | 内容 |
|---|---|---|
| 1 Introduction | 1–3 | 三轴评估方法论 |
| 2.1.1 GDN Hybrid Architecture | 3–5 | GDN 机制、参数化、RoPE/NoPE 取舍、Table 1 架构消融、FlashQLA |
| 2.1.2 Qwen Sparse Attention | 5–10 | QSA 索引器、两阶段 CPT 训练、Tables 2–4、Figs 4–6 |
| 2.2 Residual | 10–14 | HC/mHC 形式化、Table 5 消融、GR 定义、Fig 7 跨层路径分析、推理效率 |
| 2.3 N-gram Embedding | 14–15 | Tables 7–9：放置位置、固定预算配比、词表缩放 |
| 3.1 Optimizer | 16 | Muon 参数归属、拆分融合参数、Canzona |
| 3.2 Hyperparameter Scaling | 16–19 | Figs 8–9、Table 10；batch-size warmup 不再必要 |
| 3.3 Stability Stress Test | 19–22 | Figs 10–13：压力测试与门控机制 |
| 4 Evaluation | 22–23 | Table 11：14 项基准三方对比 |
| 5 Conclusion | 23 | — |
| References | 24–28 | — |

## 关键外部引用（本页核实自参考文献页 24–28）

| 简称 | 完整出处 |
|---|---|
| **IndexShare / IndexCache** | Bai et al. *IndexCache: Accelerating sparse attention via cross-layer index reuse.* arXiv:**2603.12201**, 2026 —— **注意**：GLM-5.2 模型卡称其为 "IndexShare"，同一 arXiv 号 |
| **mHC** | Xie et al. *mHC: Manifold-constrained hyper-connections.* arXiv:**2512.24880**, 2025 |
| **HC** | Zhu et al. *Hyper-Connections.* arXiv:2409.19606, 2024 |
| **xHC** | Zhang et al. *xHC: Expanded hyper-connections.* arXiv:2607.14530, 2026 |
| **VWN** | Seed. *Virtual width networks.* arXiv:2511.11238, 2025 |
| **Canzona** | Wang et al. *Canzona: A unified, asynchronous, and load-balanced framework for distributed matrix-based optimizers.* arXiv:2602.06079, 2026 |
| **Gated Attention** | Qiu et al. *Gated attention for large language models.* arXiv:2505.06708, 2025 |
| **条件记忆查表（Engram 系）** | Cheng et al. *Conditional memory via scalable lookup: A new axis of sparsity for large language models.* 2026（DeepSeek） |
| **嵌入缩放优于专家缩放** | Liu et al. arXiv:2601.21204, 2026（美团） |
| **Polar Express** | Amsel et al.（Muon 的 Newton–Schulz 系数调度） |

## 已摄入的 wiki 页面

- [[20_qwen3_8_flash_next_architecture_deepdive]] — §2 架构逐节深挖
- [[21_qwen3_8_flash_next_optimization_deepdive]] — §3–§4 优化、稳定性与评测
- [[12_qwen3_8_flash_next_analysis]] — 发布总览（模型卡 + config 对账）
