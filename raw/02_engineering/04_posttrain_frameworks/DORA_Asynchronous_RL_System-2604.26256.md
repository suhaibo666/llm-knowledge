# DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training

> 本库不随库分发第三方论文原文，仅保留来源链接与元数据。原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2604.26256](https://arxiv.org/abs/2604.26256) |
| PDF | https://arxiv.org/pdf/2604.26256 |
| 提交日期 | 2026-04-29（v1） |
| 最后更新 | 2026-07-20（v2） |
| 主分类 | cs.LG |
| 作者 | Tianhao Hu、Xiangcheng Liu、Yuchun Miao、Youshao Xiao　等 19 人（Meituan LongCat Team） |
| 代码 | 未开源；[meituan-longcat/LongCat-Flash-Thinking](https://github.com/meituan-longcat/LongCat-Flash-Thinking) 仅含模型权重与推理说明 |
| 入库 | 2026-09-16，分析页 `wiki/02_engineering/04_posttrain_frameworks/23_dora_multi_version_rollout_analysis.md` |

## 摘要

Reinforcement learning (RL) has become a critical paradigm for LLM post-training, yet the rollout phase -- accounting for 50--80% of total step time -- is bottlenecked by skewed generation: long-tailed trajectories indispensable for model performance block the entire training pipeline. Asynchronous training offers a natural remedy by overlapping generation with training, but introduces a fundamental tension between efficiency and algorithmic correctness. We identify three constraints in asynchronous training to preserve convergence: intra-trajectory policy consistency, data integrity, and bounded staleness. Existing approaches fail to intrinsically address the long-tailed trajectory problem, which is further exacerbated by the imbalance characteristic of Mix-of-Experts models, or deviate from the standard RL training formulation, thereby hindering model convergence. Therefore, we propose DORA (Dynamic ORchestration for Asynchronous Rollout), which addresses this challenge through algorithm-system co-design. DORA introduces multi-version streaming rollout, a novel asynchronous paradigm that maintains multiple policy versions concurrently -- simultaneously achieving full bubble elimination without compromising algorithmic constraints. Experimental results demonstrate that our DORA system achieves substantial improvements in throughput -- up to 2--3 times higher than state-of-the-art systems on open-source benchmarks -- without compromising convergence. Furthermore, in large-scale industrial applications with tens of thousands of accelerators, DORA accelerates RL training by 2--4 times compared to synchronous training across various scenarios. The resultant open-source models, LongCat-Flash-Thinking, exhibit competitive performance on complex reasoning benchmarks, matching the capability of most advanced LLMs.
