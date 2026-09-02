---
title: "Megatron-LM 阅读路径"
---

# Megatron-LM 阅读路径

> **这是一张路线图,不是一篇分析。** 每条只给"读它要先会什么、读完能回答什么",机制一律在目标页里。
> 目标域:[[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]](34 篇内容页)。
> 全域统一基线 `NVIDIA/Megatron-LM@85902ef59`(`dev`,2026-09-01),各页页头自带。

**这条路径解决什么**:域里 34 篇页按主题编号,但主题目录回答不了"我该按什么顺序读、读到哪一篇时该已经会了什么"。
下面按**理解的依赖顺序**排,不按编号顺序——个别地方会明确让你**提前读**某一篇。

**读者假设**:懂 Transformer 结构与朴素数据并行;不假设你了解 TP/PP/ZeRO。

---

## 站 0 · 先建立坐标(两篇,后面全部依赖它)

| # | 页 | 读它要先会什么 · 读完能回答什么 |
|---|---|---|
| 1 | [[01_megatron_architecture_analysis]] | 无前置。读完能回答:一次训练从命令行走到参数提交经过哪五层、状态按什么顺序固化——**后面每一篇都是在给这条主链的某一环补细节** |
| 2 | [[40_megatron_feature_tree_analysis]] | 无前置。**不必通读**,当地图用:想确认"Megatron 到底有没有 X 能力、归哪页管"时回来查 |

---

## 站 1 · 单卡上的模型与数据

| # | 页 | 读它要先会什么 · 读完能回答什么 |
|---|---|---|
| 3 | [[10_megatron_model_structure_analysis]] | 前置:站 0。读完能回答:模型怎么由 Spec 装配出来、注意力家族(MHA/GQA/MLA)与 MoE Router 各是什么 |
| 4 | [[11_megatron_dataset_analysis]] | 前置:无。读完能回答:语料怎么变成按 rank 对齐的 batch、`.bin`/`.idx` 三级索引怎么工作 |

---

## 站 2 · 切开:**先有几何,再有轴**

> ⚠ **这里有一个反直觉的顺序建议**:[[17_megatron_parallelism_orchestration_analysis]] 在域里定位为"收口文档",
> 按编号排在并行轴之后。但每条轴都默认"每张卡已经知道自己在各维度上的身份"——那正是 17 号页讲的。
> **第一次读建议先扫一遍 17 的 §1-§2 建立几何直觉,再回来逐轴细读**,轴读完再回 17 补完。

| # | 页 | 读它要先会什么 · 读完能回答什么 |
|---|---|---|
| 5 | [[17_megatron_parallelism_orchestration_analysis]] | 前置:站 0。读完能回答:`world_size` 个裸 GPU 怎么被切成 TP/PP/CP/EP/DP 各维度的进程组 |
| 6 | [[12_megatron_tp_analysis]] | 前置:站 1 的模型结构。读完能回答:单层的四个大矩阵乘怎么切、`f`/`g` 共轭算子与序列并行 |
| 7 | [[13_megatron_cp_analysis]] | 前置:12。读完能回答:序列维怎么切、`cp_comm_type` 四种通信方式怎么选 |
| 8 | [[14_megatron_ep_analysis]] | 前置:10 的 MoE Router 一节。读完能回答:三种 token dispatcher、MoE Parallel Folding、通信量账本 |
| 9 | [[02_megatron_moe_training_optimization_analysis]] | 前置:14。读完能回答:MoE 的各项优化按"谁拥有什么"怎么归类、选型边界在哪——**它是 MoE 的总纲,放在 14 之后读才有参照** |
| 10 | [[15_megatron_pp_schedulers_analysis]] | 前置:站 0 的训练主链。读完能回答:五种流水线调度器、气泡从哪来、怎么被 VPP 与 combined-1F1B 压小 |

---

## 站 3 · 谁持有参数、梯度与优化器状态

| # | 页 | 读它要先会什么 · 读完能回答什么 |
|---|---|---|
| 11 | [[16_megatron_distributed_optimizer_analysis]] | 前置:站 2(尤其 17 的 DP 组)。**本域最长的一页**,可分两次读:§1-§10 是 DP/ZeRO 四阶段,§11-§16 是优化器 step 内部 |
| 12 | [[36_megatron_fsdp_analysis]] | 前置:16 的 ZeRO 阶梯。读完能回答:Megatron-FSDP 为什么把分片切在扁平桶而不是切参数、与 EP/TP 怎么叠 |

---

## 站 4 · 跑快:性能基建

> 这一站六篇彼此独立,**可按需挑读**;若通读,建议按"先省显存、再省时间"的顺序。

| # | 页 | 读它要先会什么 · 读完能回答什么 |
|---|---|---|
| 13 | [[18_megatron_recompute_analysis]] | 前置:站 1。读完能回答:full 与 selective 重计算各省什么、代价是什么 |
| 14 | [[22_megatron_memory_optimization_analysis]] | 前置:18。读完能回答:显存手段全景——NCCL 池、Paged Stash、激活 offload、buffer 复用 |
| 15 | [[20_megatron_comm_overlap_analysis]] | 前置:站 2 全部。读完能回答:六个维度的通信怎么藏进计算空隙、各自的前提是什么 |
| 16 | [[21_megatron_fusion_operators_analysis]] | 前置:站 1。读完能回答:融合算子全目录与各自的触发条件 |
| 17 | [[23_megatron_precision_cudagraph_fusion_analysis]] | 前置:21。读完能回答:FP8/FP4 四种 recipe、CUDA Graph 三种 impl 的边界 |
| 18 | [[24_megatron_linear_cross_entropy_analysis]] | 前置:12(词表并行)。读完能回答:LM-head 的 logits 怎么做到从不物化 |

---

## 站 5 · 跑完、跑对:存档与可靠性

| # | 页 | 读它要先会什么 · 读完能回答什么 |
|---|---|---|
| 19 | [[19_megatron_dist_checkpointing_analysis]] | 前置:站 2、站 3。读完能回答:并行无关的分片存档怎么做到换并行度也能加载 |
| 20 | [[28_megatron_training_stability_observability_analysis]] | 前置:站 3。读完能回答:**数值层面**——loss 可不可信、SDC 怎么归因、哪张卡慢 |
| 21 | [[43_megatron_job_resilience_analysis]] | 前置:20。读完能回答:**作业层面**——进程挂了怎么原地恢复而不重排队。与 20 是同一主题的两半 |
| 22 | [[25_megatron_nonuniform_tp_analysis]] | 前置:12、站 3。读完能回答:TP 组少了 rank 还怎么训(冷重启容错) |
| 23 | [[29_megatron_packed_dataset_dynamic_cp_analysis]] | 前置:11、13。读完能回答:变长序列怎么打包、动态 CP 怎么按 batch 调整 |

---

## 站 6 · 训练之外:推理、RL、权重交付

| # | 页 | 读它要先会什么 · 读完能回答什么 |
|---|---|---|
| 24 | [[31_megatron_inference_engine_analysis]] | 前置:站 1、站 2。读完能回答:同一份权重怎么当推理引擎用——KV cache、连续批处理、chunked prefill |
| 25 | [[42_megatron_rl_runtime_analysis]] | 前置:24。读完能回答:GRPO 全链路的**实现层**——rollout 粒度、Agent 协议、损失、训推态切换 |
| 26 | [[30_megatron_rl_posttraining_consistency_analysis]] | 前置:25。读完能回答:训练与推理算出的 logprob 为什么不一致、怎么修——**与 25 是同一主题的两半**(25 讲实现,26 讲算法正确性) |
| 27 | [[33_megatron_vllm_weight_sync_analysis]] | 前置:26。读完能回答:verl 怎么把 Megatron 权重同步给 vLLM(**跨仓页**,钉 verl 基线) |
| 28 | [[44_megatron_tokenizer_and_export_analysis]] | 前置:11(分词侧)、站 2(导出侧)。读完能回答:分词器怎么装配、训练权重怎么导给 TRT-LLM |

---

## 站 7 · 参考层与案例(按需查阅,不必通读)

| # | 页 | 什么时候来看 |
|---|---|---|
| 29 | [[41_megatron_config_surface_analysis]] | 想知道某个 flag 从哪来、YAML 与 CLI 怎么保持一致时 |
| 30 | [[32_megatron_tflops_analysis]] | 要算 MFU / 吞吐,或想知道 FLOPs 数字怎么来的时 |
| 31 | [[27_megatron_tp_fsdp_resharding_supplements_analysis]] | 补遗页,三块内容的锚点与指针 |
| 32 | [[34_deepseek_v4_tensor_parallel_analysis]] | 案例:DSv4 为什么强制 TP=1 |
| 33 | [[35_deepseek_v4_context_parallel_analysis]] | 案例:MLA 怎么把 CP 通信量降两个数量级 |
| 34 | [[45_megatron_logits_distillation_analysis]] | 专题:离线蒸馏——教师前向怎么从「每步一次」变成「整数据集一次」 |

---

## 三条读法建议

1. **第一遍别读全**。站 0 → 站 2 → 站 3 是主干(约 12 篇),读完就能看懂一份真实的 Megatron 训练脚本在配什么。站 4 之后按遇到的问题挑读。
2. **遇到"这个开关是什么"就查 [[41_megatron_config_surface_analysis]] 或各页的「配置契约」小节**,不要在主线里追 flag。
3. **遇到"Megatron 有没有 X"就查 [[40_megatron_feature_tree_analysis]] 的覆盖仪表盘**,它会告诉你归哪页管、或者本库确实没写。

## 已知空白

- 各页内以 `[!note] 待展开` 标注的部分(`validate_args` 校验网、μP 机制、张量转储落盘格式等)。

## Related Pages

- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]] — 本路径的目标域,按主题组织的完整目录
- [[02_engineering/02_train_frameworks/index|训练框架]] — Megatron 在四个训练框架里的定位与覆盖度对比
- [[01_theory/06_distributed_parallelism/index|分布式并行原理]] — 各并行维度**为什么成立**的理论侧;本路径讲的是工程实现
- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan]] — PyTorch-native 的另一条路线,与 Megatron 手工并行形成对照
