---
title: "Megatron-LM 阅读路径"
---

# Megatron-LM 阅读路径

> 目标域：[[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]。事实与源码证据以目标页为准。
> 源码基线：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）。

默认读者理解 Transformer 与朴素数据并行，不要求预先掌握 Megatron 的进程组、TP、PP、CP、EP 或 ZeRO。

## 30–60 分钟入门

| 页面 | 前置 | 学习产出 |
|---|---|---|
| [[01_megatron_architecture_analysis]] | 无 | 能把一次训练从任务入口映射到配置、并行执行、模型组合、后端原语与参数提交。 |
| [[02_megatron_training_quickstart]] | 01 | 能沿官方两卡脚本走通初始化、forward-backward、梯度完成、更新和 checkpoint 回读。 |
| [[03_megatron_parallelism_geometry_quickstart]] | 02 | 能由 world size 与 TP/PP/CP/EP 推导 DP，并判断 rank 的进程组归属与配置合法性。 |

## Dense 模型核心路径

本段七页与三页入门合计十页；完成后应能阅读一份常规 Megatron 训练配置。

| 页面 | 前置 | 学习产出 |
|---|---|---|
| [[10_megatron_model_structure_analysis]] | 01 | 能定位 GPTModel、ModuleSpec、attention、MLP 与输出层的装配边界。 |
| [[11_megatron_dataset_analysis]] | 02 | 能从 tokenizer 与词表 padding 追到 IndexedDataset、GPT 取样和 packed batch。 |
| [[12_megatron_tp_analysis]] | 03、10 | 能解释线性层切分、sequence parallel 与 TP overlap 的触发边界。 |
| [[15_megatron_pp_schedulers_analysis]] | 02、03 | 能解释 microbatch、1F1B、VPP 与 pipeline bubble；首次阅读可停在标准/交错 1F1B。 |
| [[16_megatron_distributed_optimizer_analysis]] | 03 | 能推导 DDP buffer、ZeRO 0–3、HSDP 与参数/梯度/optimizer state 的所有权。 |
| [[17_megatron_parallelism_orchestration_analysis]] | 12、15、16 | 能从 RankGenerator 走到真实 ProcessGroup 与显式组注入。 |
| [[19_megatron_dist_checkpointing_analysis]] | 16、17 | 能解释 sharded state 如何跨并行配置保存、加载与重分片。 |

## 分支一：长上下文与变长序列

| 页面 | 前置 | 学习产出 |
|---|---|---|
| [[13_megatron_cp_analysis]] | 03、12 | 能比较 P2P、AllGather、A2A 与 hierarchical CP 的适用边界。 |
| [[29_megatron_packed_dataset_dynamic_cp_analysis]] | 11、13 | 能解释 packed sequence 怎样进入调度器，以及怎样按 batch 改变 CP 度。 |
| [[35_deepseek_v4_context_parallel_analysis]] | 13、29 | 能沿 boundary hidden P2P、compressed gather 与本地投影追完 DSv4 CP live path。 |

## 分支二：MoE

| 页面 | 前置 | 学习产出 |
|---|---|---|
| [[14_megatron_ep_analysis]] | 03、10 | 能追踪 route、dispatch、expert compute 与 combine，并识别 EP 硬约束。 |
| [[39_megatron_moe_training_optimization_analysis]] | 14 | 能按 token、专家参数、激活/optimizer state 与时间窗口做 MoE 工程选型。 |

## 分支三：性能与内存

| 页面 | 前置 | 学习产出 |
|---|---|---|
| [[18_megatron_recompute_analysis]] | 10 | 能比较 full/selective recompute 的显存与计算代价。 |
| [[20_megatron_comm_overlap_analysis]] | 按配置选读 12–16 | 能画出跨轴 dispatch/wait 时间线，并诊断资源竞争与静默无收益。 |
| [[21_megatron_fusion_operators_analysis]] | 20 | 能说出每一级融合吃掉的是哪条 kernel 边界、付出什么、不可用时在哪一层失败。 |
| [[22_megatron_memory_optimization_analysis]] | 18、20、21 | 能画出一块激活或优化器状态在整层换出、子模块换出、分页暂存与分块换出下的搬运时序与等待点。 |
| [[23_megatron_precision_cudagraph_fusion_analysis]] | 21 | 能比较 FP8/FP4 recipe、CUDA Graph 与 fusion 的组合边界。 |
| [[24_megatron_linear_cross_entropy_analysis]] | 12 | 能解释 LM head 与 cross-entropy 怎样避免完整 logits 物化。 |
| [[26_megatron_optimizer_step_internals_deepdive]] | 16、23 | 能走通 optimizer factory、混合精度 step、LR/WD、CPU offload 与 Muon/μP。 |
| [[32_megatron_tflops_analysis]] | 10、14 | 能核对 dense/MoE FLOPs、吞吐与 MFU 的统计口径。 |
| [[36_megatron_fsdp_analysis]] | 16 | 能追踪 MegatronFSDP buffer、hook、mesh 与预取流水线。 |

## 分支四：可靠性

| 页面 | 前置 | 学习产出 |
|---|---|---|
| [[28_megatron_training_stability_observability_analysis]] | 16 | 能区分 loss、NaN、SDC 与 straggler 的数值和观测边界。 |
| [[27_megatron_job_resilience_analysis]] | 28 | 能解释进程内重启、退出策略、GPU sniff test 与 tensor dump。 |
| [[25_megatron_nonuniform_tp_analysis]] | 12、16 | 能理解预定义混合 TP 布局、梯度重共享与冷重启边界。 |

## 分支五：推理与权重导出

| 页面 | 前置 | 学习产出 |
|---|---|---|
| [[31_megatron_inference_engine_analysis]] | 10、17 | 能解释 KV cache、连续批处理、prefix caching 与 chunked prefill。 |
| [[37_megatron_trtllm_export_analysis]] | 11、19、31 | 能把 checkpoint state dict 转成逐 rank TRT-LLM weights/config，并接到 engine build。 |

## 分支六：RL 与训推一致性

| 页面 | 前置 | 学习产出 |
|---|---|---|
| [[31_megatron_inference_engine_analysis]] | Dense 核心路径 | 能建立 rollout 推理侧的请求、批处理与 KV 状态模型。 |
| [[30_megatron_rl_posttraining_consistency_analysis]] | 31 | 能解释 logprob 一致性、importance sampling 与 reshard/refit 权重搬运。 |
| [[33_megatron_rl_runtime_analysis]] | 30 | 能追踪 rollout 粒度、Agent 协议、GRPO loss 与训推态切换。 |

## 分支七：特殊训练、案例与参考

| 页面 | 前置 | 学习产出 |
|---|---|---|
| [[38_megatron_logits_distillation_analysis]] | 11、12 | 能区分离线 top-K 缓存的 producer、未接线 writer、consumer 与 TP-aware sparse KL。 |
| [[34_deepseek_v4_tensor_parallel_analysis]] | 12、14 | 能区分 DSv4 TP=1 硬边界、duplicated 参数与仅保留 TP 接口的投影。 |
| [[40_megatron_feature_tree_analysis]] | 完成任一机制分支 | 能确认仓库能力是否被页面覆盖，以及 A–Q 模块各归谁负责。 |
| [[41_megatron_config_surface_analysis]] | 40 | 能从 config dataclass 追到 CLI/YAML 入口与机制 owner。 |

## Related Pages

- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]] —— 按编号和主题检索全部 35 篇内容页。
- [[02_engineering/02_train_frameworks/index|训练框架]] —— 对照 Megatron、TorchTitan、MindSpeed 与跨框架专题。
- [[01_theory/06_distributed_parallelism/index|分布式并行原理]] —— 补齐 TP、PP、CP、EP 与 collective 的理论前置。
- [[changelog]] —— 查询页面迁移、历史结论与基线纠正记录。
