# Knowledge Base Changelog

All source ingestions and significant wiki updates are logged here.

---

## 2026-04-29: DeepSeek-V4 Raw → Wiki 知识整合

**Type**: Knowledge Integration（Raw MD 文件与 Wiki 合并/去重）

将 `raw/05_model_families/deepseek/` 下 9 个 V4 相关 MD 文件与 Wiki 现有内容整合：

- **Created**: `wiki/llm/05_model_families/deepseek/deepseek_v4_fp4_qat_analysis.md` — FP4 QAT 完整分析（全新主题）
- **Moved (3 files)**:
  - `deepseek_v4_architecture_diagrams.md` — V4 架构 ASCII 结构图（50KB 补充参考）
  - `deepseek_v4_implementation_details.md` — V4 核心组件伪代码实现（34KB 补充参考）
  - `deepseek_v4_technical_deep_dive.md` — CSA/HCA/DSA/MLA 对比深度解析（42KB 补充参考）
- **Updated (merged unique content)**:
  - `deepseek_v4_analysis.md` — 新增 §Compressed KV 数值示例、DualPath 推理框架、Think Modes、Pro-Max 评测
  - `mHC.md` — 扩展 §动态与静态系数（完整公式 3-8、对比表、训练细节）
  - `deepseek_v4_cp_analysis.md` — 新增 §9 实现细节（Fused Select-and-Pad、Top-K Selector、传统 CP 对比表）
- **Cross-references**: 所有新/更新页面双向链接已更新

---

## 2026-04-29: Activation Checkpointing（重计算）完整分析

**Type**: Knowledge Synthesis（PyTorch autograd 机制 + Megatron-LM 源码分析）

- **Created**: `wiki/llm/02_training/activation_checkpointing_analysis.md` — 激活重计算完整分析（中文）
- **Updated**: `wiki/llm/overview.md` — Optimizers & Training Algorithms 表格新增条目
- **Updated**: `wiki/llm/06_infra/megatron-lm/Megatron-LM_Distributed_Parallel_Exam.md` — Q12 考点添加交叉引用
- **Key topics**:
  - autograd `ctx.save_for_backward` 机制与 `torch.no_grad` 干预原理
  - ctx 中 tensor 激活值 vs 元信息的二分法（重计算只消除前者）
  - View/Cast/Slice 算子的反向机制：仅依赖元信息，ctx 不存储 tensor
  - View chain 问题与 Megatron `make_viewless_tensor` 的切断方案
  - Megatron 三层 checkpoint 架构：CheckpointFunction → CheckpointWithoutOutput/te_checkpoint → TransformerBlock 调度
  - `distribute_saved_activations` 的 TP 切分/聚合机制
  - `CheckpointWithoutOutput` 的 zero-copy storage sharing 和 `CheckpointManager`
  - Uniform vs Block 调度策略、逐层 checkpoint 的必要性（vs 整 model 一层）
  - Selective recomputation 的子模块级选择依据与 Decoder 层激活值依赖全景
  - 理论激活值开销公式与估算范例

---

## 2026-04-28: DeepSeek-V4 CP 深度分析

---

## 2026-04-28: DeepSeek-V4 CP 深度分析

**Type**: Source Ingestion (扩展已有 V4 分析)

- **Source**: `raw/05_model_families/deepseek/DeepSeek_V4.pdf` §3.5.3, §3.6, §4.1
- **Created**: `wiki/llm/05_model_families/deepseek/deepseek_v4_cp_analysis.md` — DeepSeek-V4 Context Parallelism 深度分析（中文）
- **Updated**: `wiki/llm/05_model_families/deepseek/deepseek_v4_analysis.md` — CP 节扩展并添加指向新页面链接
- **Key topics**:
  - Packed sequences 数据格式与 CP 的三个矛盾（跨 rank 文档切断、压缩窗口跨边界、压缩输出长度不可预测）
  - 两阶段通信协议形式化描述（Stage 1 P2P O(c) 常数通信 + Stage 2 All-Gather 压缩 KV）
  - 通信量开销公式推导与数值估算（CSA ~51× 减少, HCA ~2048× 减少 vs 标准 CP）
  - 三层 sample 可见性控制（sample-level attention mask → block-level causal → precomputed rules / Top-K selector）
  - 训练 vs 推理尾部 token 处理策略对比（丢弃 vs State Cache vs 重计算）
  - CSA 重叠窗口对 CP 边界的额外影响
  - 完整 packed sequences × CP × 压缩的数值示例

---

## 2026-04-24: Wiki Directory Restructure

**Type**: Infrastructure

Restructured `wiki/llm/` to mirror `raw/` classification (01-08), consolidating related content:

- **Created** subdirectories under `wiki/llm/`:
  - `01_architecture/` — Transformer, scaling laws, memory architectures
  - `02_training/` — Optimizers, initialization, training precision
  - `03_alignment/` — RLHF, DPO, GRPO, PPO, and related methods
  - `04_reasoning_and_retrieval/` — Reserved for CoT, verification, RAG
  - `05_model_families/deepseek/` — All DeepSeek model analyses
  - `06_infra/megatron-lm/` — Distributed training, MoE infrastructure
  - `07_multimodal/` — Reserved for vision-language, audio-language
  - `08_agents/` — Reserved for agentic AI, tool use
- **Moved** `wiki/torch_compile/` → `wiki/llm/02_training/torch_compile/`
- **Moved** `wiki/megatron-lm/` → `wiki/llm/06_infra/megatron-lm/`
- **Moved** `mHC.md` → `wiki/llm/05_model_families/deepseek/mHC.md`
- **Updated** all path-based wiki links across the entire wiki

---

## 2026-04-16: Wiki Schema & Structure Initialization

**Type**: Infrastructure

Created the wiki schema and structural pages:

- Created `CLAUDE.md` — wiki maintenance schema and rules
- Created `wiki/llm/overview.md` — LLM domain knowledge map
- Created `wiki/megatron-lm/overview.md` — Megatron-LM domain knowledge map
- Created `wiki/torch_compile/overview.md` — torch compile domain knowledge map
- Created `wiki/changelog.md` — this file

---

## Pre-Changelog Entries (Historical Reconstruction)

The following pages were created before the changelog was established. Dates are approximate.

### ~2026-03: MoE & Distributed Training

- Created `wiki/megatron-lm/Megatron-LM_MoE_Zero_Redundancy_Analysis.md` — Source: `raw/Scalable Training of Moe Models with Megatron core-2603.07685v2.pdf`
- Created `wiki/megatron-lm/Megatron-LM_Distributed_Parallel_Exam.md` — Comprehensive exam covering 5D parallelism

### ~2026-02: Muon Optimizer

- Created `wiki/llm/muon_analysis.md` — Source: `raw/MUON IS SCALABLE FOR LLM TRAINING-2502.16982v1.pdf`
- Created `wiki/megatron-lm/Megatron_LM_TFLOPS_Analysis.md` — TFLOPS estimation methodology

### ~2026-01: DeepSeek & Memory Architectures

- Created `wiki/llm/Engram_Analysis.md` — Source: `raw/Engram_paper.pdf`
- Created `wiki/llm/deepseek_math_v2.md` — Self-verifiable math reasoning

### ~2025-12: Weight Initialization & KIMI

- Created `wiki/llm/llm_initiliaze_analysis.md` — Dense & MoE initialization

---

## 2026-04-17: mHC Source Ingestion

**Type**: Source Ingestion

- **Source**: `raw/mHC-2512.24880v2.pdf` (DeepSeek-AI, arXiv:2512.24880v2)
- **Created**: `wiki/llm/mHC.md` — Manifold-Constrained Hyper-Connections analysis (in Chinese)
- **Updated**: `wiki/llm/overview.md` — Added mHC entry and cross-domain links
- **Cross-referenced**: Added backlinks to `muon_analysis.md`, `llm_initiliaze_analysis.md`, `Megatron-LM_MoE_Zero_Redundancy_Analysis.md`
- **Key topics**: doubly stochastic matrix, Sinkhorn-Knopp projection, residual stream expansion, DeepSeek-V3 MoE, kernel fusion, selective recomputing

### ~2025-11: Training-Inference Integration

- Created `wiki/megatron-lm/Megatron_vLLM_Weight_Sync_Analysis.md` — verl Megatron + vLLM weight sync

### ~2025-10: Torch Compile & NPU

- Created `wiki/torch_compile/inductor/` — 17 pages covering Dynamo, AOT Autograd, Inductor, NPU backends
- Created `wiki/torch_compile/cudagraphs/` — CUDA Graphs guides and NPU Graphs deep dives

## 2026-04-20: DeepSeek Model Family Batch Ingestion (Part 1/4)

**Type**: Source Ingestion

- **Source**: `raw/05_model_families/deepseek/DeepSeek_LLM-2401.02954.pdf` (DeepSeek-AI, arXiv:2401.02954)
- **Created**: `wiki/llm/deepseek_llm_analysis.md` — DeepSeek LLM analysis
- **Updated**: `wiki/llm/overview.md` — Added DeepSeek model family section
- **Key topics**: scaling laws with non-embedding FLOPs/token representation, data quality impact on model/data allocation, multi-step LR scheduler, GQA, bilingual pre-training, SFT+DPO alignment

- **Source**: `raw/05_model_families/deepseek/DeepSeek_V2-2405.04434.pdf` (DeepSeek-AI, arXiv:2405.04434)
- **Created**: `wiki/llm/deepseek_v2_analysis.md` — DeepSeek-V2 analysis
- **Key topics**: MLA (Multi-head Latent Attention), low-rank KV joint compression, decoupled RoPE, DeepSeekMoE, device-limited routing, three-level auxiliary losses, token dropping, GRPO, two-stage RL

- **Source**: `raw/05_model_families/deepseek/DeepSeek_V3-2412.19437.pdf` (DeepSeek-AI, arXiv:2412.19437)
- **Created**: `wiki/llm/deepseek_v3_analysis.md` — DeepSeek-V3 analysis
- **Key topics**: FP8 mixed precision training, fine-grained quantization (tile/block-wise), DualPipe pipeline parallelism, auxiliary-loss-free load balancing, Multi-Token Prediction (MTP), cross-node all-to-all communication kernels, inference deployment with redundant experts, R1 distillation

- **Source**: `raw/05_model_families/deepseek/DeepSeek_R1-2501.12948.pdf` (DeepSeek-AI, arXiv:2501.12948)
- **Created**: `wiki/llm/deepseek_r1_analysis.md` — DeepSeek-R1 analysis
- **Key topics**: pure RL reasoning without SFT, GRPO, emergent self-verification/reflection, "aha moment", multi-stage pipeline (cold start → RL → SFT → RL), distillation to Qwen/Llama, rule-based rewards, language consistency reward

**Remaining**: Coder, Coder-V2, Math, MoE, Prover, VL

---

## 2026-04-24: DeepSeek-V4 Source Ingestion

**Type**: Source Ingestion

- **Source**: `raw/05_model_families/deepseek/DeepSeek_V4.pdf` (DeepSeek-AI, 2025)
- **Created**: `wiki/llm/deepseek_v4_analysis.md` — DeepSeek-V4 analysis (in Chinese)
- **Updated**: `wiki/llm/overview.md` — Added V4 to DeepSeek model family section
- **Updated**: `wiki/llm/deepseek_v3_analysis.md` — Added backlink to V4
- **Updated**: `wiki/llm/deepseek_v2_analysis.md` — Added backlink to V4
- **Cross-referenced**: `mHC.md`, `muon_analysis.md`, `deepseek_v3_analysis.md`, `deepseek_v2_analysis.md`
- **Key topics**: CSA (Compressed Sparse Attention), HCA (Heavily Compressed Attention), hybrid attention architecture, DSA (DeepSeek Sparse Attention), Lightning Indexer, million-token context, mHC integration, Muon optimizer, Anticipatory Routing, SwiGLU clamping, wave-based EP overlap, TileLang kernels, FP4 QAT, heterogeneous KV cache management, on-disk KV cache storage

---

## 2026-04-21: DeepSeek Model Family Batch Ingestion (Part 2/4)

**Type**: Source Ingestion

- **Source**: `raw/05_model_families/deepseek/DeepSeek_Coder-2401.14196.pdf` (DeepSeek-AI, arXiv:2401.14196)
- **Created**: `wiki/llm/deepseek_coder_analysis.md` — DeepSeek-Coder analysis
- **Key topics**: repository-level code corpus, dependency parsing, topological sort, Fill-in-the-Middle (FIM), 87 programming languages, 16K context, GQA

- **Source**: `raw/05_model_families/deepseek/DeepSeek_Coder_V2-2406.11931.pdf` (DeepSeek-AI, arXiv:2406.11931)
- **Created**: `wiki/llm/deepseek_coder_v2_analysis.md` — DeepSeek-Coder-V2 analysis
- **Key topics**: MoE code model, 338 languages, 128K context, 6T additional tokens, YaRN extension, GRPO with reward model, SWE-bench >10%

- **Source**: `raw/05_model_families/deepseek/DeepSeek_Math-2402.03300.pdf` (DeepSeek-AI, arXiv:2402.03300)
- **Created**: `wiki/llm/deepseek_math_analysis.md` — DeepSeekMath analysis
- **Key topics**: 120B math tokens from Common Crawl, iterative fastText pipeline, GRPO origin, unified RL paradigm, MATH 51.7%

- **Source**: `raw/05_model_families/deepseek/DeepSeek_MoE-2401.06066.pdf` (DeepSeek-AI, arXiv:2401.06066)
- **Created**: `wiki/llm/deepseek_moe_analysis.md` — DeepSeekMoE architecture analysis
- **Key topics**: fine-grained expert segmentation, shared expert isolation, expert-level/device-level balance loss, 2B/16B/145B scales

- **Source**: `raw/05_model_families/deepseek/DeepSeek_Prover-2408.08152.pdf` (DeepSeek-AI, arXiv:2408.08152)
- **Created**: `wiki/llm/deepseek_prover_analysis.md` — DeepSeek-Prover-V1.5 analysis
- **Key topics**: Lean 4 theorem proving, truncate-and-resume mechanism, RMaxTS Monte-Carlo tree search, thought-augmented proofs, RLPAF

- **Source**: `raw/05_model_families/deepseek/DeepSeek_VL-2403.05525.pdf` (DeepSeek-AI, arXiv:2403.05525)
- **Created**: `wiki/llm/deepseek_vl_analysis.md` — DeepSeek-VL analysis
- **Key topics**: hybrid vision encoder (SigLIP + SAM), 576 visual tokens, modality warm-up, 70% text preservation, real-world VL taxonomy

- **Note**: `raw/05_model_families/deepseek/DeepSeek_VL2-2412.10322.pdf` was identified as an unrelated physics paper (arXiv:2412.10322v1, hep-lat). No genuine DeepSeek-VL2 source was found.

**Remaining**: None (DeepSeek model family complete)

---

## 2026-04-21: Architecture Foundations & Alignment Methods Batch Ingestion

**Type**: Source Ingestion

### Architecture Foundations (01_architecture/)

- **Source**: `raw/01_architecture/Attention_Is_All_You_Need-1706.03762.pdf` (Vaswani et al., Google, NIPS 2017)
- **Created**: `wiki/llm/attention_is_all_you_need_analysis.md` — Transformer architecture analysis
- **Key topics**: scaled dot-product attention, multi-head attention, positional encoding, encoder-decoder structure, self-attention vs RNN/CNN complexity, O(1) path length

- **Source**: `raw/01_architecture/Scaling_Laws_for_Neural_Language_Models-2001.08361.pdf` (Kaplan et al., OpenAI, 2020)
- **Created**: `wiki/llm/scaling_laws_analysis.md` — Neural scaling laws analysis
- **Key topics**: power-law scaling (L ~ N^-0.076, D^-0.095, C^-0.050), compute-optimal training (N~C^0.73), sub-linear data scaling (D~N^0.74), early stopping, critical batch size, architecture independence

- **Source**: `raw/01_architecture/Long_Context_Scaling_Law-2503.04725.pdf` (Chen et al., MIT, NeurIPS 2025)
- **Created**: `wiki/llm/long_context_scaling_law_analysis.md` — Long-context mutual information scaling
- **Key topics**: bipartite mutual information (I_BP ~ L^beta), L2M condition, history state requirements, Transformer vs SSM long-context capability

- **Skipped**: `raw/01_architecture/Scaling_Laws_for_Transfer-2002.05102.pdf` — PDF contains unrelated mathematics paper (Hurwitz actions on reflection groups)

### Alignment & Preference Optimization (03_alignment/)

- **Source**: `raw/03_alignment/PPO_Proximal_Policy_Optimization-1707.06347.pdf` (Schulman et al., OpenAI, 2017)
- **Created**: `wiki/llm/ppo_analysis.md` — PPO algorithm analysis
- **Key topics**: PPO-Clip objective, surrogate loss, multiple epochs on same data, GAE advantage estimation, KL constraint

- **Source**: `raw/03_alignment/InstructGPT_RLHF-2203.02155.pdf` (Ouyang et al., OpenAI, 2022)
- **Created**: `wiki/llm/instructgpt_rlhf_analysis.md` — RLHF pipeline analysis
- **Key topics**: three-step RLHF (SFT→RM→PPO), KL penalty against SFT, 1.3B > 175B GPT-3, helpful/honest/harmless criteria

- **Source**: `raw/03_alignment/DPO_Direct_Preference_Optimization-2305.18290.pdf` (Rafailov et al., Stanford, 2023)
- **Created**: `wiki/llm/dpo_analysis.md` — DPO algorithm analysis
- **Key topics**: closed-form policy-reward relationship, binary cross-entropy replaces RLHF, no sampling during training

- **Created**: `wiki/llm/preference_optimization_analysis.md` — DPO family comparison
- **Covers**: IPO (squared loss), SimPO (no ref model, length-normalized), ORPO (monolithic), KTO (binary labels, prospect theory), MODPO (multi-objective)

- **Source**: `raw/03_alignment/DeepSeek_R1_Reasoning_via_RL-2501.12948.pdf` (DeepSeek-AI, 2025)
- **Created**: `wiki/llm/grpo_analysis.md` — GRPO algorithm analysis
- **Key topics**: group-relative advantages, no value function, pure RL for reasoning, DeepSeek-R1-Zero emergent behaviors

**Updated**: `wiki/llm/overview.md` — Added Architecture Foundations, Scaling Laws, and Alignment sections

---

## 2026-04-21: Alignment Methods Batch Ingestion (Part 2)

**Type**: Source Ingestion

### Advanced RL Algorithms

- **Source**: `raw/03_alignment/DAPO_Decoupled_Clip_Dynamic_Sampling-2503.14476.pdf` (ByteDance Seed, Tsinghua AIR, 2025)
- **Created**: `wiki/llm/dapo_analysis.md` — DAPO algorithm analysis
- **Key topics**: decoupled clipping (eps_low=0.2, eps_high=0.28), dynamic sampling (filter accuracy 0/1), token-level policy gradient loss, soft overlong punishment, AIME 50 with Qwen2.5-32B, open-source RL system

- **Source**: `raw/03_alignment/GSPO_Group_Sequence_Policy_Optimization-2507.18071.pdf` (Qwen Team, Alibaba, 2025)
- **Created**: `wiki/llm/gspo_analysis.md` — GSPO algorithm analysis
- **Key topics**: sequence-level importance ratio, fixes GRPO's token-level instability, length-normalized sequence likelihood, stabilizes MoE RL training, Qwen3 improvements

- **Source**: `raw/03_alignment/RLOO_REINFORCE_Leave_One_Out-2402.14740.pdf` (Cohere For AI, 2024)
- **Created**: `wiki/llm/rloo_analysis.md` — RLOO algorithm analysis
- **Key topics**: REINFORCE with leave-one-out baseline, no value function needed, theoretical foundation for GRPO, 2.5x faster than PPO

- **Source**: `raw/03_alignment/VAPO_Value_Augmented_Proximal_Policy_Optimization-2504.05118.pdf` (ByteDance Seed, 2025)
- **Created**: `wiki/llm/vapo_analysis.md` — VAPO framework analysis
- **Key topics**: value-model-based RL, AIME 60.4 (SOTA), addresses value bias/length heterogeneity/reward sparsity, 5000 steps to SOTA, zero crashes

### RLHF Foundations & Advanced Methods

- **Created**: `wiki/llm/rlhf_foundations_analysis.md` — Comprehensive coverage of:
  - **ReMax** (arXiv:2310.10505): Simplified RLHF using REINFORCE, exploits fast simulation/deterministic transitions/trajectory rewards
  - **Weak-to-Strong Generalization** (OpenAI, arXiv:2312.09390): Can weak model supervision elicit strong model capabilities? Analogy to superhuman alignment
  - **Scaling Laws for RM Overoptimization** (OpenAI, arXiv:2210.10760): Goodhart's Law in RLHF, predictable scaling of overoptimization, best-of-n vs RL
  - **Learning to Summarize** (OpenAI, arXiv:2009.01325): First RLHF for summarization, precursor to InstructGPT
  - **Fine-Tuning from Human Preferences** (OpenAI, arXiv:1909.08593): Earliest RLHF work, stylistic control and summarization
  - **RigorLLM** (arXiv:2403.13031): Resilient guardrails against adversarial attacks, energy-based data generation, minimax optimization

**Updated**: `wiki/llm/overview.md` — Added DAPO, GSPO, RLOO, VAPO, and RLHF Foundations entries

**Digestion progress**: 3/4 architecture papers, **20/20 alignment papers digested** (complete)

## Related Pages

- [[llm/overview]]
- [[llm/06_infra/megatron-lm/overview]]
- [[llm/02_training/torch_compile/overview]]
