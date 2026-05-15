# MoE Training Optimization Technical Report — Design Spec

**Date**: 2026-05-12
**Status**: Approved
**Type**: Research report + Wiki knowledge base expansion

## Goal

Produce a comprehensive technical report analyzing MoE (Mixture of Experts) training optimization techniques in Megatron-LM, covering all 7 dimensions at source-code depth.

## Architecture: Two-Layer Delivery

1. **Wiki pages** (reusable knowledge): Each dimension has a standalone wiki page with source-code analysis. Three new pages fill identified Knowledge Gaps; existing pages are referenced.
2. **Consolidated report**: A single report weaving all 7 dimensions with core insights, cross-referencing wiki pages via `[[links]]`.

## Report Structure (9 Chapters)

| Chapter | Topic | Wiki Strategy |
|---------|-------|---------------|
| 1 | Introduction: MoE Training Challenges | Reference existing deepseek MoE analyses |
| 2 | Multi-Dimensional Parallelism (TP/PP/EP/CP) | Integrate existing + supplement CP |
| 3 | Distributed Optimizer & FSDP2 | **New page**: `megatron_distributed_optimizer_analysis.md` (includes 3-strategy comparison: DistributedOptimizer vs TorchFullyShardedDataParallel vs MegatronFSDP) |
| 4 | Activation Checkpointing | Integrate from `activation_checkpointing_analysis.md` |
| 5 | Low-Precision Training | Integrate from `low_precision_training_analysis.md` |
| 6 | Communication-Computation Overlap | Integrate from `megatron_comm_overlap_analysis.md` |
| 7 | Memory Optimization | **New page**: `megatron_memory_optimization_analysis.md` |
| 8 | Fusion Operator Optimization | **New page**: `megatron_fusion_operators_analysis.md` |
| 9 | Summary: Technique Matrix & Decision Tree | Scale tiers: 50B → 300B → 671B → 1.x TB |

## Per-Chapter Format

- What is the optimization point?
- Why is it effective?
- When is it applicable (hardware/scale conditions)?
- Key implementation details (source code references)
- Applicability summary

## Source Code Entry Points

| Dimension | Key Files in `megatron/core/` |
|-----------|-------------------------------|
| TP | `tensor_parallel/layers.py`, `mappings.py` |
| PP | `pipeline_parallel/schedules.py`, `p2p_communication.py` |
| EP | `transformer/moe/token_dispatcher.py`, `router.py` |
| CP | `tensor_parallel/cross_entropy.py`, `dot_product_attention_context_parallel.py` |
| Distributed Optimizer | `optimizer/distrib_optimizer.py`, `param_and_grad_buffer.py`, `cpu_offloading/` |
| FSDP2 (TorchFSDP2 + MegatronFSDP) | `distributed/torch_fully_sharded_data_parallel.py`, `distributed/fsdp/mcore_fsdp_adapter.py`, `distributed/fsdp/src/megatron_fsdp/` |
| Recompute | `transformer/transformer_block.py`, `pipeline_parallel/schedules.py` |
| Low Precision | `fp8_utils.py`, `fp4_utils.py` |
| Comm Overlap | `distributed/distributed_data_parallel.py`, `transformer/moe/token_dispatcher.py` |
| Memory | `nccl_allocator.py`, `transformer/moe/paged_stash.py`, `resharding/` |
| Fusions | `fusions/` (10+ operators), `transformer/moe/fused_a2a.py` |

## Success Criteria

- [x] 3 new wiki pages created (distributed optimizer, memory optimization, fusion operators)
- [x] CP source code analysis supplemented to existing coverage
- [x] FSDP2 analysis supplemented — Megatron 三种梯度/参数分片方案对比 (DistributedOptimizer / TorchFullyShardedDataParallel / MegatronFSDP)
- [x] 9-chapter report completed with core insights per chapter, including FSDP2 §3.5
- [x] All cross-references consistent, megatron-lm/index.md updated
- [x] Changelog updated with all operations
