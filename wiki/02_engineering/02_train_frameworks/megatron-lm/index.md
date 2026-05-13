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
| [[megatron_memory_optimization_analysis]] | NCCL memory pool, MoE Paged Stash (3-tier overflow), fine-grained activation offloading, param/grad buffer reuse (MXFP8 shared, NVFP4 dual), FP8/FP4 parameter precision, CUDA graph buffer reuse, resharding |
| [[megatron_fusion_operators_analysis]] | Bias+Activation fusion (GEGLU/SwiGLU/GELU), fused LayerNorm, fused Softmax, MoE-specific fusions (pad routing map, indices converter, weighted squared ReLU), fused Cross-Entropy, fused All-to-All (DeepEP/HybridEP), FP8 input store, Triton/CUTLASS/cuTile kernels |

### Performance & Measurement

| Page | Key Concepts |
|------|-------------|
| [[Megatron_LM_TFLOPS_Analysis]] | Theoretical FLOPS estimation, forward/backward FLOP counting, MoE dropless vs dropout accuracy, throughput calculation formula |
| [[megatron_comm_overlap_analysis]] | 6-dimension communication-computation overlap (TP/DP/PP/EP/CP), bulk & pipelined overlap, delay-wgrad, DeepEP/HybridEP |

### Training-Inference Integration

| Page | Key Concepts |
|------|-------------|
| [[Megatron_vLLM_Weight_Sync_Analysis]] | verl framework, Megatron-to-vLLM weight synchronization, Gather-Broadcast-Load pattern, colocation scenario, HuggingFace format reassembly |
| [[mooncake_analysis]] | Mooncake: KVCache 中心化分离式服务架构，Prefill/Decode 池分离，缓存感知调度，Moonshot AI 推理基础设施 |

## Cross-Domain Links

- Distributed parallelism concepts connect to [[muon_analysis]] in the LLM domain (Muon's ZeRO incompatibility)
- MoE zero-redundancy dispatch relates to [[llm_initiliaze_analysis]] (MoE expert initialization)
- CUDA Graphs usage in Megatron connects to [[SUMMARY]] in the CUDA Graphs sub-domain
- Weight sync patterns relate to inference optimization topics in the torch_compile domain
- FP8/low-precision training connects to [[low_precision_training_analysis]] and [[transformer_engine_analysis]] in the training domain
- Distributed optimizer state sharding relates to ZeRO strategies in [[megatron_distributed_optimizer_analysis]]
- Memory optimization techniques coalesce around [[megatron_memory_optimization_analysis]]
- Fusion operators complement communication overlap patterns in [[megatron_comm_overlap_analysis]] and [[megatron_fusion_operators_analysis]]

## Knowledge Gaps

These topics are referenced but lack dedicated wiki pages:

- **Context Parallelism deep dive** — ~~mentioned in the exam but no standalone analysis~~ → partially addressed by [[llm_parallelism_analysis]]
- **TransformerEngine integration** — ~~referenced but not documented~~ → addressed by [[transformer_engine_analysis]]
- **Low-precision training** — ~~FP8/FP4 scattered across exam and V4 pages~~ → consolidated in [[low_precision_training_analysis]]
- **Megatron-LM checkpoint format** — important for weight sync understanding
- ~~**Distributed optimizer** — no standalone analysis~~ → addressed by [[megatron_distributed_optimizer_analysis]]
- ~~**Memory optimization panorama** — scattered across multiple pages~~ → addressed by [[megatron_memory_optimization_analysis]]
- ~~**Fusion operators** — no dedicated page~~ → addressed by [[megatron_fusion_operators_analysis]]
- **Sequence Parallelism implementation details** — ~~distinct from TP but not yet covered~~ → addressed by [[llm_parallelism_analysis]] and [[megatron_comm_overlap_analysis]]

## Related Pages

- [[Megatron-LM_MoE_Zero_Redundancy_Analysis]]
- [[llm_parallelism_analysis]]
- [[02_engineering/01_ai_frameworks/index]]
