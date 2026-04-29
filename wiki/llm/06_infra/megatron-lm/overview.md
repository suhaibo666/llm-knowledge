# Megatron-LM — Knowledge Map

This domain covers NVIDIA Megatron-LM distributed training framework, including parallelism strategies, MoE implementation, performance measurement, and integration with inference engines.

## Core Topics

### Distributed Parallelism

| Page | Key Concepts |
|------|-------------|
| [[Megatron-LM_Distributed_Parallel_Exam]] | 3D/4D/5D parallelism (DP, TP, PP, SP, CP), Expert Parallelism (EP/ETP), communication volume analysis, activation checkpointing, FP8, CUDA Graphs in Megatron |
| [[llm_parallelism_analysis]] | LLM 正反向计算依赖 DAG, TP/SP/EP/CP 通信依赖分析, 计算通信重叠时序, Megatron-LM 源码级验证 |

### MoE Implementation

| Page | Key Concepts |
|------|-------------|
| [[Megatron-LM_MoE_Zero_Redundancy_Analysis]] | Zero-redundancy communication, Expert Parallelism, AlltoAll token dispatch, token routing strategies, memory vs communication tradeoff |

### Performance & Measurement

| Page | Key Concepts |
|------|-------------|
| [[Megatron_LM_TFLOPS_Analysis]] | Theoretical FLOPS estimation, forward/backward FLOP counting, MoE dropless vs dropout accuracy, throughput calculation formula |

### Training-Inference Integration

| Page | Key Concepts |
|------|-------------|
| [[Megatron_vLLM_Weight_Sync_Analysis]] | verl framework, Megatron-to-vLLM weight synchronization, Gather-Broadcast-Load pattern, colocation scenario, HuggingFace format reassembly |

## Cross-Domain Links

- Distributed parallelism concepts connect to [[muon_analysis]] in the LLM domain (Muon's ZeRO incompatibility)
- MoE zero-redundancy dispatch relates to [[llm_initiliaze_analysis]] (MoE expert initialization)
- CUDA Graphs usage in Megatron connects to [[SUMMARY]] in the CUDA Graphs sub-domain
- Weight sync patterns relate to inference optimization topics in the torch_compile domain

## Knowledge Gaps

These topics are referenced but lack dedicated wiki pages:

- **Context Parallelism deep dive** — ~~mentioned in the exam but no standalone analysis~~ → partially addressed by [[llm_parallelism_analysis]]
- **TransformerEngine integration** — referenced but not documented
- **Megatron-LM checkpoint format** — important for weight sync understanding
- **Sequence Parallelism implementation details** — ~~distinct from TP but not yet covered~~ → addressed by [[llm_parallelism_analysis]] and [[megatron_comm_overlap_analysis]]

## Related Pages

- [[Megatron-LM_Distributed_Parallel_Exam]]
- [[Megatron-LM_MoE_Zero_Redundancy_Analysis]]
- [[llm_parallelism_analysis]]
- [[llm/02_training/torch_compile/overview]]
