---
title: "Megatron-LM 知识地图"
---

# Megatron-LM 知识地图

本域以 `NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）为统一源码基线，共 **35 篇内容页**。

第一次系统学习请走 [[courses/megatron_lm|Megatron-LM 阅读路径]]；查询仓库能力用 [[40_megatron_feature_tree_analysis]]，查询配置字段 owner 用 [[41_megatron_config_surface_analysis]]，迁移与历史纠正见 [[changelog]]。

## 五个内容段

| 段 | 编号 | 内容页 | 作用 |
|---|---|---:|---|
| 入门 | 01–03 | 3 | 建立系统全貌、跑通最小训练、掌握并行几何。 |
| 核心训练 | 10–19 | 10 | 覆盖模型、数据、五维并行、重计算与 checkpoint 主链。 |
| 优化与可靠性 | 20–29 | 10 | 覆盖 overlap、融合、内存、精度、optimizer、韧性与动态 CP。 |
| Runtime、集成与案例 | 30–39 | 10 | 覆盖 RL、推理、导出、FSDP、DSv4 案例、蒸馏与 MoE 选型。 |
| 参考 | 40–41 | 2 | 提供代码仓功能树与配置面双向对账。 |

## 全部页面

| 编号 | 页面 | 一行定位 |
|---:|---|---|
| 01 | [[01_megatron_architecture_analysis]] | 从任务入口到一次参数提交的五层系统架构。 |
| 02 | [[02_megatron_training_quickstart]] | 用官方两卡脚本走通最小训练与 checkpoint 回读。 |
| 03 | [[03_megatron_parallelism_geometry_quickstart]] | 从 world size 推导 TP/PP/CP/EP/DP 坐标与进程组。 |
| 10 | [[10_megatron_model_structure_analysis]] | GPTModel、ModuleSpec、attention、MLP 与输出层的装配边界。 |
| 11 | [[11_megatron_dataset_analysis]] | tokenizer、IndexedDataset、GPT 取样与 sequence packing 数据入口。 |
| 12 | [[12_megatron_tp_analysis]] | tensor/sequence parallel 的切分、collective 与 overlap。 |
| 13 | [[13_megatron_cp_analysis]] | context parallel 的通信模式、分层组与 Dynamic CP 接口。 |
| 14 | [[14_megatron_ep_analysis]] | MoE route、dispatch、expert compute、combine 与 EP 硬约束。 |
| 15 | [[15_megatron_pp_schedulers_analysis]] | microbatch、1F1B、VPP 与高级 pipeline scheduler。 |
| 16 | [[16_megatron_distributed_optimizer_analysis]] | 连续 range 与整参 owner、梯度规约、更新及参数可见性闭环。 |
| 17 | [[17_megatron_parallelism_orchestration_analysis]] | dense/expert 分组、进程组生命周期与显式组注入。 |
| 18 | [[18_megatron_recompute_analysis]] | 激活与 RNG 回放、full/selective 及调度器重计算的数据路径。 |
| 19 | [[19_megatron_dist_checkpointing_analysis]] | sharded state、并行无关保存、加载与重分片。 |
| 20 | [[20_megatron_comm_overlap_analysis]] | 跨 TP/CP/EP/PP/DP 的时间线、资源竞争与诊断。 |
| 21 | [[21_megatron_fusion_operators_analysis]] | 融合算子的触发条件、后端实现与回退目录。 |
| 22 | [[22_megatron_memory_optimization_analysis]] | offload、paged stash、buffer reuse 与通信内存池。 |
| 23 | [[23_megatron_precision_cudagraph_fusion_analysis]] | FP8/FP4、CUDA Graph 与算子融合的组合边界。 |
| 24 | [[24_megatron_linear_cross_entropy_analysis]] | 不物化完整 logits 的 fused linear cross-entropy。 |
| 25 | [[25_megatron_nonuniform_tp_analysis]] | 预定义混合 TP 布局、梯度重共享与冷重启边界。 |
| 26 | [[26_megatron_optimizer_step_internals_deepdive]] | optimizer factory、混合精度 step、LR/WD、offload 与 Muon/μP。 |
| 27 | [[27_megatron_job_resilience_analysis]] | 进程内重启、退出策略、GPU 检测与 tensor dump。 |
| 28 | [[28_megatron_training_stability_observability_analysis]] | loss、NaN、SDC、straggler 与观测控制面。 |
| 29 | [[29_megatron_packed_dataset_dynamic_cp_analysis]] | packing scheduler 与按 microbatch 改变 CP 度的流水线。 |
| 30 | [[30_megatron_rl_posttraining_consistency_analysis]] | RL logprob 一致性、importance sampling 与 reshard/refit。 |
| 31 | [[31_megatron_inference_engine_analysis]] | continuous batching、KV cache、prefix caching 与 chunked prefill。 |
| 32 | [[32_megatron_tflops_analysis]] | dense/MoE FLOPs、吞吐与 MFU 的统计口径。 |
| 33 | [[33_megatron_rl_runtime_analysis]] | Agent 协议、rollout 粒度、GRPO 与训推态切换。 |
| 34 | [[34_deepseek_v4_tensor_parallel_analysis]] | DSv4 TP=1 硬边界、duplicated 投影与多 rank 证明义务。 |
| 35 | [[35_deepseek_v4_context_parallel_analysis]] | DSv4 boundary hidden P2P 与 compressed gather 两阶段案例。 |
| 36 | [[36_megatron_fsdp_analysis]] | MegatronFSDP 的 buffer、hook、mesh 与预取流水线。 |
| 37 | [[37_megatron_trtllm_export_analysis]] | checkpoint 到逐 rank TRT-LLM 权重/config 与 engine build。 |
| 38 | [[38_megatron_logits_distillation_analysis]] | 离线 top-K 缓存协议、writer 接线边界与 sparse KL。 |
| 39 | [[39_megatron_moe_training_optimization_analysis]] | 按四种所有权组织的 MoE 工程选型地图。 |
| 40 | [[40_megatron_feature_tree_analysis]] | A–Q 功能树、600 个源文件与页面覆盖对账。 |
| 41 | [[41_megatron_config_surface_analysis]] | dataclass 到 CLI/YAML 的配置生成、校验与 owner 追踪。 |

## Related Pages

- [[courses/megatron_lm|Megatron-LM 阅读路径]] —— 按理解依赖组织三页入门、Dense 核心与七个问题分支。
- [[02_engineering/02_train_frameworks/index|训练框架]] —— 查看 Megatron 与其它训练框架在父域中的位置。
- [[40_megatron_feature_tree_analysis]] —— 从代码仓 A–Q 功能树反查页面覆盖与空白。
- [[41_megatron_config_surface_analysis]] —— 从配置字段反查声明、CLI/YAML 入口与机制 owner。
- [[changelog]] —— 查询本域重编号、拆并页和失效结论的历史记录。
