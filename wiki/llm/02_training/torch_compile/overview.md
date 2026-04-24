# PyTorch Compilation Stack — Knowledge Map

This domain covers the PyTorch compilation pipeline (`torch.compile`), including Dynamo graph capture, AOT Autograd, TorchInductor code generation, and CUDA/NPU Graphs optimization.

## Architecture Overview

```
User Code → @torch.compile
              ↓
         TorchDynamo (graph capture via frame evaluation hook)
              ↓
         AOT Autograd (forward/backward decomposition)
              ↓
         TorchInductor (IR lowering → codegen)
              ↓
         Optimized Kernel Execution
```

See [[torch_compile_architecture]] for the full pipeline breakdown.

## Sub-Domains

### TorchInductor

| Page | Key Concepts |
|------|-------------|
| [[torch_compile_source_analysis]] | Source code structure, module organization |
| [[torch_compile_architecture]] | End-to-end pipeline: Dynamo → AOT Autograd → Inductor |
| [[PyTorch_Dynamo_Technical_Analysis]] | Frame evaluation API, bytecode symbolic execution, guard generation, FX graph construction |
| [[PyTorch_Inductor_Technical_Analysis]] | Inductor IR, scheduling, codegen backend architecture |
| [[aotautograd_analysis]] | Forward/backward graph decomposition, joint graph passes |
| [[lowering_analysis]] | FX graph → Inductor IR lowering process |
| [[inductor_codegen_analysis]] | Code generation strategy, kernel fusion |
| [[inductor_codegen_dynamic_shape_analysis]] | Dynamic shape handling in codegen |
| [[scheduler_analysis]] | Operator scheduling, fusion decisions |
| [[pre_grad_passes_guide]] | Pre-grad optimization passes |
| [[post_grad_passes_guide]] | Post-grad optimization passes |
| [[joint_graph_passes_guide]] | Joint forward-backward graph optimization |
| [[npu_lowering_guide]] | NPU-specific lowering steps |
| [[npu_compile]] | NPU compilation workflow |
| [[Pytorch_Compile_Debug_Analysis]] | Debugging techniques, log interpretation |
| [[NPU_Inductor_Backend_Analysis]] | NPU backend integration architecture |
| [[NPU_Inductor_Backend_Mechanism]] | NPU backend internal mechanisms |
| [[NPU_MLIR_Backend_Technical_Analysis]] | MLIR-based NPU backend technical details |

### CUDA & NPU Graphs

| Page | Key Concepts |
|------|-------------|
| [[SUMMARY]] | Index of all CUDA/NPU Graphs documentation |
| [[PyTorch_CUDA_Graphs_Complete_Guide]] | Complete CUDA Graphs usage guide |
| [[CUDA_Graphs_Timing_Diagrams]] | Timing diagrams for graph capture & replay |
| [[llm/02_training/torch_compile/cudagraphs/npugraphs/README]] | NPU Graphs overview |
| [[comparison]] | CUDA Graphs vs NPU Graphs feature comparison |
| [[torch_compile_npugraphs_deep_dive]] | NPU Graphs integration with torch.compile |
| [[npugraphs_make_graphed_callables_deep_dive]] | make_graphed_callables API deep dive |
| [[npugraphs_memory_management_analysis]] | Memory management in NPU Graphs |
| [[npugraphs_memory_reuse_analysis]] | Memory reuse strategies |
| [[torch_compile_mode_reduce_overhead_vs_backend_npugraphs]] | reduce_overhead mode vs backend npugraphs |
| [[aclgraph]] | ACL Graph (Ascend Computing Language) integration |

## Cross-Domain Links

- CUDA Graphs usage in Megatron-LM connects to [[Megatron-LM_Distributed_Parallel_Exam]]
- NPU compilation relates to hardware-specific optimization in the Megatron-LM domain
- Graph capture mechanisms relate to training pipeline optimization

## Knowledge Gaps

These topics are referenced but lack dedicated wiki pages:

- **TorchDynamo guard failure debugging** — common but not documented
- **Inductor autotuning** — Triton kernel autotuning strategy
- **Dynamic shape support completeness** — known limitations not catalogued
- **Multi-backend dispatch** — how Inductor selects between CUDA/NPU backends

## Related Pages

- [[torch_compile_architecture]]
- [[PyTorch_CUDA_Graphs_Complete_Guide]]
- [[llm/06_infra/megatron-lm/overview]]
