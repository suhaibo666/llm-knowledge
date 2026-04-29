# LLM Training & Optimization — Knowledge Map

This domain covers large language model training algorithms, optimization techniques, and novel architectural proposals.

## Core Topics

### Foundational Architecture

| Page | Key Concepts |
|------|-------------|
| [[attention_is_all_you_need_analysis]] | Transformer architecture, scaled dot-product attention, multi-head attention, positional encoding, encoder-decoder structure, self-attention vs RNN/CNN |

### Scaling Laws

| Page | Key Concepts |
|------|-------------|
| [[scaling_laws_analysis]] | Power-law scaling of loss with N/D/C, compute-optimal training (N~C^0.73), sub-linear data scaling (D~N^0.74), early stopping, critical batch size, architecture independence |
| [[long_context_scaling_law_analysis]] | Bipartite mutual information scaling (I_BP ~ L^beta), L2M condition, history state requirements, Transformer vs SSM long-context capability |

### Alignment & Preference Optimization

| Page | Key Concepts |
|------|-------------|
| [[instructgpt_rlhf_analysis]] | Three-step RLHF pipeline (SFT → RM → PPO), KL penalty against SFT, 1.3B > 175B GPT-3 |
| [[ppo_analysis]] | PPO-Clip objective, surrogate loss, multiple epochs on same data, GAE advantage estimation |
| [[dpo_analysis]] | Direct preference optimization, closed-form policy-reward relationship, binary cross-entropy replaces RLHF |
| [[preference_optimization_analysis]] | DPO family comparison: IPO, SimPO (no ref model), ORPO (monolithic), KTO (binary labels), MODPO |
| [[grpo_analysis]] | Group-relative advantages, no value function, pure RL for reasoning, DeepSeek-R1 training |
| [[dapo_analysis]] | Decoupled clipping (eps_high=0.28), dynamic sampling, token-level loss, overlong reward shaping, AIME 50 |
| [[gspo_analysis]] | Sequence-level importance ratio, fixes GRPO's token-level instability, stabilizes MoE RL training |
| [[rloo_analysis]] | REINFORCE with leave-one-out baseline, theoretical foundation for GRPO, simpler than PPO |
| [[vapo_analysis]] | Value-model-based RL, AIME 60.4, addresses value bias/length heterogeneity/reward sparsity |
| [[rlhf_foundations_analysis]] | ReMax, Weak-to-Strong, RM Overoptimization, Learning to Summarize, Fine-Tuning from Preferences, RigorLLM |

### Optimizers & Training Algorithms

| Page | Key Concepts |
|------|-------------|
| [[muon_analysis]] | Muon optimizer, Newton-Schulz iteration, spectral norm orthogonalization, ZeRO incompatibility, Megatron-LM layer-wise partitioning |
| [[llm_initiliaze_analysis]] | Weight initialization (Xavier/He/Kaiming), residual scaling, pre-LN, MoE expert & router initialization, meta-device lazy init |
| [[activation_checkpointing_analysis]] | 激活重计算完整分析：autograd ctx 保存机制 → CheckpointFunction 源码 → Full/Selective 策略 → view/cast/slice 的 ctx 特性 → 理论显存评估 |

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
| [[deepseek_v4_analysis]] | DeepSeek-V4, CSA/HCA hybrid attention, million-token context, mHC, Muon, 1.6T MoE |
| [[deepseek_v4_cp_analysis]] | V4 Context Parallelism 适配 packed sequences 的两阶段通信方案 |
| [[deepseek_v4_fp4_qat_analysis]] | V4 FP4 量化感知训练（MoE 专家权重、CSA QK path、index scores） |
| [[deepseek_v4_architecture_diagrams]] | V4 架构 ASCII 结构图（补充参考） |
| [[deepseek_v4_implementation_details]] | V4 核心组件伪代码实现（补充参考） |
| [[deepseek_v4_technical_deep_dive]] | V4 CSA/HCA/DSA/MLA 对比深度解析（补充参考） |
| [[mHC]] | 流形约束超连接（V4 残差连接改进核心） |
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
- [[llm/06_infra/megatron-lm/overview]]
