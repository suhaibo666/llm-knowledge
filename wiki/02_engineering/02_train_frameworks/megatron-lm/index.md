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
| 01 | [[01_megatron_architecture_analysis]] | 七层系统架构：一次预训练怎样穿过它们，以及十一个顶层场景。 |
| 02 | [[02_megatron_training_quickstart]] | 用官方两卡脚本走通最小训练与 checkpoint 回读。 |
| 03 | [[03_megatron_parallelism_geometry_quickstart]] | 从 world size 推导 TP/PP/CP/EP/DP 坐标与进程组。 |
| 10 | [[10_megatron_model_structure_analysis]] | GPTModel、ModuleSpec、attention、MLP 与输出层的装配边界。 |
| 11 | [[11_megatron_dataset_analysis]] | tokenizer、IndexedDataset、GPT 取样与 sequence packing 数据入口。 |
| 12 | [[12_megatron_tp_analysis]] | tensor/sequence parallel 的切分、collective 与 overlap。 |
| 13 | [[13_megatron_cp_analysis]] | 标准 attention 的 `cp_comm_type` 接入面，线性 attention 与 Mamba 两条非标准 CP 数据面，分层组与 Dynamic CP。 |
| 14 | [[14_megatron_ep_analysis]] | 从 AllGather、AllToAll 到 DeepEP、HybridEP 的分发方案与训练闭环。 |
| 15 | [[15_megatron_pp_schedulers_analysis]] | 从 GPipe、1F1B 到 VPP 与通信重叠，兼述 MoE 和多模块调度。 |
| 16 | [[16_megatron_distributed_optimizer_analysis]] | 连续 range 与整参 owner、梯度规约、更新及参数可见性闭环。 |
| 17 | [[17_megatron_parallelism_orchestration_analysis]] | dense/expert 分组、进程组生命周期与显式组注入。 |
| 18 | [[18_megatron_recompute_analysis]] | 从反向依赖推导 full/selective 与输出丢弃，按容量预算选型，再解读源码装配和系统组合。 |
| 19 | [[19_megatron_dist_checkpointing_analysis]] | 描述子、访问校验与并行无关的存取：存时 TP4、载时 TP2 也能对上。 |
| 20 | [[20_megatron_comm_overlap_analysis]] | TP/CP/EP/PP/DP/FSDP 六条掩盖对、跨轴时间线、资源竞争与诊断梯子。 |
| 21 | [[21_megatron_fusion_operators_analysis]] | 从逐点 JIT 融合到 GEMM 链与通信融合的四级阶梯，及各级的后端选择与失败边界。 |
| 22 | [[22_megatron_memory_optimization_analysis]] | 从整层换出、子模块换出到分页暂存与分块优化器换出，及常驻通信池的反向取舍。 |
| 23 | [[23_megatron_precision_cudagraph_fusion_analysis]] | 从计算、访存与提交瓶颈推导低精度、融合和 CUDA Graph，再展开整层收益、源码前后向及使用约束。 |
| 24 | [[24_megatron_linear_cross_entropy_analysis]] | 从普通词表 CE 递进到 linear 分块重计算，核算前反向显存与 TP/SP 通信。 |
| 25 | [[25_megatron_nonuniform_tp_analysis]] | 预定义混合 TP 布局、梯度重共享与冷重启边界。 |
| 26 | [[26_megatron_optimizer_step_internals_deepdive]] | fp32 master 换来什么、五步固定顺序与闸门位置，四条 wrapper 的字节账。 |
| 27 | [[27_megatron_job_resilience_analysis]] | 故障五段各自的 deadline：心跳闸门、清理超时、一致退出与主动探测。 |
| 28 | [[28_megatron_training_stability_observability_analysis]] | 每条判据连同它的失效条件：梯度三闸、尖峰判据、SDC 归因与观测面。 |
| 29 | [[29_megatron_packed_dataset_dynamic_cp_analysis]] | 一条九步流水线加一个可换的分组步：固定 CP 与按长度定 CP 的同批对照。 |
| 30 | [[30_megatron_rl_posttraining_consistency_analysis]] | 训推一致性的五条来源与五环收敛，残差交给 importance sampling。 |
| 31 | [[31_megatron_inference_engine_analysis]] | 块级 KV cache 上的连续批处理与背压、`InferenceMode` 单一开关，再到 chunked prefill、prefix caching 与图尺寸枚举。 |
| 32 | [[32_megatron_tflops_analysis]] | 上报 TFLOPS 是 GEMM 的闭式计数乘两个批级统计量：dense/THD/MoE/DSA 各口径偏向哪边、偏多少。 |
| 33 | [[33_megatron_rl_runtime_analysis]] | Agent 协议、rollout 粒度、GRPO 与训推态切换。 |
| 34 | [[34_deepseek_v4_tensor_parallel_analysis]] | DSv4 Hybrid Attention 的单卡执行面：两道 TP=1 守卫与参数所有权账本，CSA 索引集合、FlashMLA / cuDNN / cudnn-frontend 融合内核族的分派与回退、FP8 下的精度驻留、up-proj 重算与 mHC 交界。 |
| 35 | [[35_deepseek_v4_context_parallel_analysis]] | DSv4 的 CP 数据面：contiguous THD 分片、左边界 hidden 的 autograd P2P、定容压紧与 rank-major 压缩行、两个异步 AllGather 与本地投影重叠、反向延迟 reduce-scatter，以及 CP 下的 indexer loss 与 CUDA graph 共存。 |
| 36 | [[36_megatron_fsdp_analysis]] | Megatron-FSDP 把分片切在 FSDP unit 的扁平桶上：四步分组与 DP-LCM 网格、四类缓冲与「四档 = 三个布尔量」、hook 状态机与 AG / RS 两条流水线、持久池与 HSDP / HFSDP、接入层；并用同一算例复演基线内并存的 v2 数据面（`DBuffer` / placement 变换）。 |
| 37 | [[37_megatron_trtllm_export_analysis]] | checkpoint 到逐 rank TRT-LLM 权重/config 与 engine build。 |
| 38 | [[38_megatron_logits_distillation_analysis]] | 离线 top-K 缓存协议、writer 接线边界与 sparse KL。 |
| 39 | [[39_megatron_moe_training_optimization_analysis]] | MoE 负载建模、并行策略与调优顺序；结合 NVIDIA 2026 技术报告核对收益条件。 |
| 40 | [[40_megatron_feature_tree_analysis]] | A–Q 功能树、600 个源文件与页面覆盖对账。 |
| 41 | [[41_megatron_config_surface_analysis]] | dataclass 到 CLI/YAML 的配置生成、校验与 owner 追踪。 |

## Related Pages

- [[courses/megatron_lm|Megatron-LM 阅读路径]] —— 按理解依赖组织三页入门、Dense 核心与七个问题分支。
- [[02_engineering/02_train_frameworks/index|训练框架]] —— 查看 Megatron 与其它训练框架在父域中的位置。
- [[40_megatron_feature_tree_analysis]] —— 从代码仓 A–Q 功能树反查页面覆盖与空白。
- [[41_megatron_config_surface_analysis]] —— 从配置字段反查声明、CLI/YAML 入口与机制 owner。
- [[changelog]] —— 查询本域重编号、拆并页和失效结论的历史记录。
