# LLM Training & Optimization — Knowledge Map

This domain covers large language model training algorithms, optimization techniques, and novel architectural proposals.

## Core Topics

### Optimizers & Training Algorithms

| Page | Key Concepts |
|------|-------------|
| [[muon_analysis]] | Muon optimizer, Newton-Schulz iteration, spectral norm orthogonalization, ZeRO incompatibility, Megatron-LM layer-wise partitioning |
| [[llm_initiliaze_analysis]] | Weight initialization (Xavier/He/Kaiming), residual scaling, pre-LN, MoE expert & router initialization, meta-device lazy init |

### Model Architecture Innovations

| Page | Key Concepts |
|------|-------------|
| [[Engram_Analysis]] | DeepSeek Engram, memory sparsity, conditional memory, static N-gram lookup, deterministic multi-head hashing, U-type scaling law, compute-memory tradeoff |
| [[mHC]] | Manifold-Constrained Hyper-Connections, doubly stochastic matrix, Sinkhorn-Knopp projection, residual stream expansion, DeepSeek-V3 MoE |

### DeepSeek Model Family

| Page | Key Concepts |
|------|-------------|
| [[deepseek_llm_analysis]] | DeepSeek LLM (7B/67B), scaling laws, multi-step LR, GQA, bilingual pre-training |
| [[deepseek_v2_analysis]] | DeepSeek-V2, MLA (Multi-head Latent Attention), DeepSeekMoE, 236B total / 21B active |
| [[deepseek_v3_analysis]] | DeepSeek-V3, FP8 training, 671B MoE, dual-pipeline parallelism, MTP |
| [[deepseek_r1_analysis]] | DeepSeek-R1, pure RL reasoning, GRPO, cold start, distillation |
| [[deepseek_coder_analysis]] | DeepSeek-Coder, code-specific pre-training, 2T tokens, project-level FIM |
| [[deepseek_coder_v2_analysis]] | DeepSeek-Coder-V2, MoE code model, 338 languages, 128K context, GRPO alignment |
| [[deepseek_math_analysis]] | DeepSeekMath 7B, 120B math tokens from Common Crawl, GRPO origin, MATH 51.7% |
| [[deepseek_moe_analysis]] | DeepSeek-MoE architecture, expert routing, load balancing, fine-grained experts |
| [[deepseek_prover_analysis]] | DeepSeek-Prover-V1.5, Lean 4 theorem proving, truncate-and-resume, RMaxTS |
| [[deepseek_math_v2]] | DeepSeekMath-V2, self-verification, Generator-Verifier loop, proof generation, proof refinement, RL fine-tuning for math |
| [[deepseek_vl_analysis]] | DeepSeek-VL, vision-language alignment, hybrid encoder |

### Reasoning & Verification

| Page | Key Concepts |
|------|-------------|
| [[deepseek_math_v2]] | DeepSeekMath-V2, self-verification, Generator-Verifier loop, proof generation, proof refinement, RL fine-tuning for math |

## Cross-Domain Links

- Muon optimizer's distributed training challenges relate directly to [[Megatron-LM_Distributed_Parallel_Exam]] in the Megatron-LM domain
- MoE initialization in [[llm_initiliaze_analysis]] connects to [[Megatron-LM_MoE_Zero_Redundancy_Analysis]]
- Engram's compute-memory separation relates to training efficiency metrics in [[Megatron_LM_TFLOPS_Analysis]]
- mHC's residual stream scaling and MoE training connect to [[Megatron-LM_MoE_Zero_Redundancy_Analysis]] and [[llm_initiliaze_analysis]]

## Knowledge Gaps

These topics are referenced but lack dedicated wiki pages:

- **AdamW vs Muon benchmarking** — quantitative comparison data
- **Distributed optimizer state management** — general patterns beyond Muon
- **DeepSeek-VL2** — `raw/DeepSeek_VL2-2412.10322.pdf` is not a genuine DeepSeek-VL2 paper (contains unrelated physics content)

## Related Pages

- [[muon_analysis]]
- [[llm_initiliaze_analysis]]
- [[megatron-lm/overview]]
