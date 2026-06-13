# Megatron-LM — Knowledge Map

This domain covers NVIDIA Megatron-LM distributed training framework, including parallelism strategies, MoE implementation, performance measurement, and integration with inference engines.

## Core Topics

### Comprehensive Reports

| Page | Key Concepts |
|------|-------------|
| [[moe_training_optimization_report]] | 7-dimension MoE training optimization panorama: TP/PP/EP/CP, distributed optimizer, activation checkpointing, low-precision (FP8/FP4), comm-compute overlap, memory optimization, fusion operators. Source-code-level analysis based on Megatron-LM. Covers 50B → 1.xT MoE model scales |

### Distributed Parallelism

| Page | Key Concepts |
|------|-------------|
| [[llm_parallelism_analysis]] | LLM 正反向计算依赖 DAG, TP/SP/EP/CP 通信依赖分析, 计算通信重叠时序, Megatron-LM 源码级验证 |
| [[moe_training_optimization_report]] | 7 维 MoE 训练优化全景：并行策略（含 SP/TP 边界、Dynamic CP、MoE Router/Folding、autograd Function 设计）、分布式优化器/FSDP2、重计算、低精度、通信 Overlap、显存优化、融合算子 |

### MoE Implementation

| Page | Key Concepts |
|------|-------------|
| [[Megatron-LM_MoE_Zero_Redundancy_Analysis]] | Zero-redundancy communication, Expert Parallelism, AlltoAll token dispatch, token routing strategies, memory vs communication tradeoff |

### Memory & Compute Optimization

| Page | Key Concepts |
|------|-------------|
| [[megatron_distributed_optimizer_analysis]] | ZeRO-1/2 optimizer state sharding, Reduce-Scatter + All-Gather communication, FP8/FP4 quantized params, CPU offloading (HybridDeviceOptimizer, StateOffloader), precision-aware optimizer |
| [[../distributed_optimizer_deep_dive\|distributed_optimizer_deep_dive]] | FSDP2/ZeRO/MindSpeed 三方对比, 梯度累积通信量 (K×P), param 临时化与 zero-copy, Adam vs Muon 内存估算 (18→14 bytes/param), Muon 对 ZeRO 切分的根本性挑战 |
| [[megatron_memory_optimization_analysis]] | NCCL memory pool, MoE Paged Stash (3-tier overflow), fine-grained activation offloading, param/grad buffer reuse (MXFP8 shared, NVFP4 dual), FP8/FP4 parameter precision, CUDA graph buffer reuse, resharding |
| [[megatron_fusion_operators_analysis]] | Bias+Activation fusion (GEGLU/SwiGLU/GELU), fused LayerNorm, fused Softmax, MoE-specific fusions (pad routing map, indices converter, weighted squared ReLU), fused Cross-Entropy, fused All-to-All (DeepEP/HybridEP), FP8 input store, Triton/CUTLASS/cuTile kernels |

### Performance & Measurement

| Page | Key Concepts |
|------|-------------|
| [[Megatron_LM_TFLOPS_Analysis]] | Theoretical FLOPS estimation, forward/backward FLOP counting, MoE dropless vs dropout accuracy, throughput calculation formula |
| [[megatron_nonuniform_tp_analysis]] | GPU fault tolerance via mixed-size TP groups across DP replicas, spare→core→extra gradient all-to-all resharing, process group reconfiguration, cold-restart mechanism |
| [[megatron_comm_overlap_analysis]] | 6-dimension communication-computation overlap (TP/DP/PP/EP/CP), bulk & pipelined overlap, delay-wgrad, DeepEP/HybridEP |

### Training-Inference Integration

| Page | Key Concepts |
|------|-------------|
| [[Megatron_vLLM_Weight_Sync_Analysis]] | verl framework, Megatron-to-vLLM weight synchronization, Gather-Broadcast-Load pattern, colocation scenario, HuggingFace format reassembly |
| [[mooncake_analysis]] | Mooncake: KVCache 中心化分离式服务架构，Prefill/Decode 池分离，缓存感知调度，Moonshot AI 推理基础设施 |

## 源码级系统分析系列(Megatron-LM `dev` @ `ee3f1ff`, 2026-05)

> 一套 18 篇源码级系统分析,基于 Megatron-LM `dev` 分支 commit `ee3f1ff`,逐层覆盖并行体系、性能基建、训练稳定性、数据、推理与 RL、模型结构与存档。各篇互为 `[[wiki link]]` 交叉引用,自成体系。

### 并行轴(5)

| Page | Key Concepts |
|------|-------------|
| [[pp_schedulers_analysis]] | 流水线并行 5 调度器:无流水线 / 1F1B / 交错 VPP / P2P-overlap / combined-1F1B;气泡公式推导、流水线模拟图(附调度模拟器 `_pp_sim.py`) |
| [[ep_analysis]] | 专家并行:AllGather / AllToAll / Flex(DeepEP/HybridEP)三种 token dispatcher;MoE Parallel Folding;通信量与负载均衡 |
| [[tp_analysis]] | 张量并行:ColumnParallel/RowParallel 共轭算子 f/g、MLP/Attention 切分;Sequence Parallelism |
| [[cp_analysis]] | 上下文并行:p2p(ring)/ all_gather / a2a(Ulysses)/ a2a+p2p 四种 cp_comm_type;因果 zigzag 负载均衡 |
| [[ddp_optimizer_analysis]] | 数据并行 + 分布式优化器:ZeRO-0/1/2/3 四阶段、梯度分桶重叠、HSDP |

### 编排与补遗(3)

| Page | Key Concepts |
|------|-------------|
| [[parallelism_orchestration_analysis]] | 进程组编排 capstone:RankGenerator、order 字符串、正交分组数学、双 RankGenerator(MoE Folding)、parallel_state/ProcessGroupCollection/HyperCommGrid 三层抽象 |
| [[pp_supplements_analysis]] | PP 补遗:P2P 通信内部、激活换出、混合 CP 动态调度、多模块/多模态流水线 |
| [[tp_fsdp_resharding_supplements_analysis]] | Megatron-FSDP 内部(ZeRO-2/3 流水线)、Nonuniform TP 容错、Resharding/Refit |

### 性能基建(3)

| Page | Key Concepts |
|------|-------------|
| [[recompute_analysis]] | 激活重计算:full(uniform/block)vs selective;标准 vs 输出丢弃 checkpointing |
| [[optimizer_internals_analysis]] | 优化器内部:混合精度 fp32 master、step 五步、Loss Scaling、梯度裁剪、LR 调度 |
| [[precision_cudagraph_fusion_analysis]] | FP8/FP4 四 recipe、CUDA Graph 三 impl、算子融合 |

### 系统专题(3)

| Page | Key Concepts |
|------|-------------|
| [[training_stability_observability_analysis]] | 训练稳定性与可观测:loss scaling / 梯度裁剪 / RerunStateMachine(SDC 归因)/ QK-clip;Timer、MoE 逐层指标、指标目录 |
| [[rl_posttraining_consistency_analysis]] | RL 后训练适配与训推一致性:Refit、布局/格式保真、inference_optimized、重算 logprob、importance sampling |
| [[inference_engine_analysis]] | 推理引擎:Static/Dynamic 引擎、连续批处理、块级 KV cache(paged)、prefix caching、chunked prefill |

### 数据 / 模型 / 存档(4)

| Page | Key Concepts |
|------|-------------|
| [[dataset_analysis]] | GPT 数据集:IndexedDataset(.bin/.idx)、三级索引、隐式打包 vs 显式打包 |
| [[packed_dataset_dynamic_cp_analysis]] | 序列打包与动态 CP 统一流水线:BasePackingScheduler→DpBalanced→DefaultDynamicCP 继承链 |
| [[model_structure_analysis]] | 模型结构:Spec 系统、TransformerLayer、注意力家族(MHA/GQA/MLA)、MoE Router 算法、MTP、SSM/Mamba |
| [[dist_checkpointing_analysis]] | 分布式 checkpoint:ShardedTensor(local↔global 映射)、并行无关存档、sharded_state_dict、fully-parallel/async 策略 |

## Cross-Domain Links

- Distributed parallelism concepts connect to [[muon_analysis]] in the LLM domain (Muon's ZeRO incompatibility)
- MoE zero-redundancy dispatch relates to [[llm_initiliaze_analysis]] (MoE expert initialization)
- CUDA Graphs usage in Megatron connects to [[06_graphs/cuda/README]] in the CUDA Graphs sub-domain
- Weight sync patterns relate to inference optimization topics in the torch_compile domain
- FP8/low-precision training connects to [[low_precision_training_analysis]] and [[transformer_engine_analysis]] in the training domain
- Distributed optimizer state sharding relates to ZeRO strategies in [[megatron_distributed_optimizer_analysis]]
- Memory optimization techniques coalesce around [[megatron_memory_optimization_analysis]]
- Fusion operators complement communication overlap patterns in [[megatron_comm_overlap_analysis]] and [[megatron_fusion_operators_analysis]]
- NTP fault tolerance connects to TP/DP/CP topology concepts in [[megatron_nonuniform_tp_analysis]]

## Knowledge Gaps

These topics are referenced but lack dedicated wiki pages:

- ~~**Context Parallelism deep dive**~~ → 专文 [[cp_analysis]](4 种 cp_comm_type + zigzag 负载均衡);另见 [[llm_parallelism_analysis]]
- **TransformerEngine integration** — ~~referenced but not documented~~ → addressed by [[transformer_engine_analysis]]
- **Low-precision training** — ~~FP8/FP4 scattered across exam and V4 pages~~ → consolidated in [[low_precision_training_analysis]];另见 [[precision_cudagraph_fusion_analysis]]
- ~~**Megatron-LM checkpoint format**~~ → 专文 [[dist_checkpointing_analysis]](ShardedTensor、并行无关存档)
- ~~**Distributed optimizer** — no standalone analysis~~ → addressed by [[megatron_distributed_optimizer_analysis]] 与 [[ddp_optimizer_analysis]]
- ~~**Memory optimization panorama** — scattered across multiple pages~~ → addressed by [[megatron_memory_optimization_analysis]];另见 [[recompute_analysis]]
- ~~**Fusion operators** — no dedicated page~~ → addressed by [[megatron_fusion_operators_analysis]]
- ~~**Sequence Parallelism implementation details**~~ → 专门章节见 [[tp_analysis]] §4;另见 [[llm_parallelism_analysis]]、[[megatron_comm_overlap_analysis]]
- ~~**Fault tolerance / GPU failure recovery** — not covered~~ → addressed by [[megatron_nonuniform_tp_analysis]];另见 [[tp_fsdp_resharding_supplements_analysis]] §2
- **NTP checkpoint conversion tooling** — gradient resharing is implemented, but parameter/optimizer state resharding for checkpoint restore is not yet in the codebase

## Related Pages

- [[Megatron-LM_MoE_Zero_Redundancy_Analysis]]
- [[llm_parallelism_analysis]]
- [[megatron_nonuniform_tp_analysis]]
- [[pp_schedulers_analysis]] · [[parallelism_orchestration_analysis]] · [[model_structure_analysis]]
- [[02_engineering/01_ai_frameworks/index]]
