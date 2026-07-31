# DeepSeek LLM Analysis

**Source:** `raw/05_model_families/deepseek/DeepSeek_LLM-2401.02954.pdf`  
**Date Ingested:** 2026-04-20  
**Authors:** DeepSeek-AI  
**Published:** January 2024

---

## Overview

DeepSeek LLM is the first open-source large language model from DeepSeek-AI, released in January 2024. It introduces a rigorous empirical study of scaling laws and trains competitive 7B and 67B parameter models that surpass LLaMA-2 70B on code, math, and reasoning benchmarks.

---

## Key Contributions

1. **Scaling Laws Revisited**: Discovered that data quality significantly affects the optimal model/data allocation strategy when scaling compute.
2. **New Model Scale Representation**: Proposed using **non-embedding FLOPs/token ($M$)** instead of raw parameter count ($N$) to more accurately estimate compute budget $C = M \cdot D$.
3. **Multi-Step LR Scheduler**: Replaced cosine decay with a multi-step scheduler to facilitate continual training without performance loss.
4. **Competitive Performance**: DeepSeek 67B Chat outperforms GPT-3.5 in both Chinese and English open-ended evaluations.

---

## Architecture

The micro-design follows LLaMA:

- **Pre-Norm** with RMSNorm
- **SwiGLU** activation (FFN intermediate = $\frac{8}{3} d_{model}$)
- **RoPE** (Rotary Positional Embedding)
- **Grouped-Query Attention (GQA)** on 67B model (8 KV heads) to reduce inference cost

The macro-design differs in depth scaling:

| Model | Layers | $d_{model}$ | Heads | KV Heads | Context | Batch Size | LR | Tokens |
|-------|--------|-------------|-------|----------|---------|------------|-----|--------|
| 7B  | 30 | 4096 | 32 | 32 | 4096 | 2304 | $4.2 \times 10^{-4}$ | 2.0T |
| 67B | 95 | 8192 | 64 | 8  | 4096 | 4608 | $3.2 \times 10^{-4}$ | 2.0T |

Notably, the 67B model scales **depth** (95 layers) rather than FFN width, which is an uncommon design choice aimed at better performance.

---

## Pre-Training

### Data

- **2 trillion tokens**, primarily Chinese and English
- Pipeline: **deduplication -> filtering -> remixing**
- Aggressive cross-dump deduplication (91 dumps -> 89.8% dedup rate)
- Quality assessment combines linguistic and semantic evaluation

### Tokenizer

- Byte-level BPE (BBPE) with pre-tokenization
- Vocabulary: 100,015 tokens (100,000 conventional + 15 special)
- Model vocabulary size configured to 102,400 for computational efficiency

### Hyperparameters

- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.95$, weight_decay=0.1)
- **Init std**: 0.006
- **Gradient clipping**: 1.0
- **Precision**: bf16 training with fp32 gradient accumulation

### Multi-Step Learning Rate Scheduler

| Stage | Token Progression | LR Level |
|-------|-------------------|----------|
| Warmup | First 2,000 steps | Linear ramp to max |
| Phase 1 | 0% - 80% | Max LR |
| Phase 2 | 80% - 90% | 31.6% of max |
| Phase 3 | 90% - 100% | 10% of max |

This scheduler matches cosine final performance while allowing **seamless continuation** from Phase 1 checkpoints.

---

## Scaling Laws

### Hyperparameter Scaling

Through IsoFLOP profiling on compute budgets from $10^{17}$ to $2 \times 10^{19}$ FLOPs:

$$
\eta_{opt} = 0.3118 \cdot C^{-0.1250}
$$

$$
B_{opt} = 0.2920 \cdot C^{0.3271}
$$

### Model/Data Allocation

Using $M$ (non-embedding FLOPs/token) as model scale:

$$
M_{opt} = 0.1715 \cdot C^{0.5243}
$$

$$
D_{opt} = 5.8316 \cdot C^{0.4757}
$$

Where $M = 72 \cdot n_{layer} \cdot d_{model}^2 + 12 \cdot n_{layer} \cdot d_{model} \cdot l_{seq}$

### Key Finding: Data Quality Impact

| Dataset | Model Exponent $a$ | Data Exponent $b$ |
|---------|-------------------|-------------------|
| OpenAI (OpenWebText2) | 0.73 | 0.27 |
| Chinchilla (MassiveText) | 0.49 | 0.51 |
| DeepSeek (Early Data) | 0.450 | 0.550 |
| DeepSeek (Current Data) | 0.524 | 0.476 |
| DeepSeek (OpenWebText2) | 0.578 | 0.422 |

**Insight:** Higher data quality shifts the optimal allocation toward **model scaling** rather than data scaling. This helps explain divergent conclusions in prior scaling law literature.

---

## Alignment Pipeline

### SFT (Supervised Fine-Tuning)

- ~1.5M instruction instances (1.2M helpful + 300K safety)
- Helpful data: 31.2% general language, 46.6% math, 22.2% code
- 7B: 4 epochs | 67B: 2 epochs (overfitting observed on larger model)
- **Repetition issue**: Math SFT data causes repetitive outputs; mitigated via two-stage SFT (exclude math/code in stage 2)

### DPO (Direct Preference Optimization)

- 1 epoch, LR=$5 \times 10^{-6}$, batch size=512
- Preference data for helpfulness and harmlessness
- DPO improves open-ended generation skills with minimal benchmark impact

---

## Evaluation Highlights

### Base Model vs LLaMA-2

DeepSeek 67B surpasses LLaMA-2 70B on:
- **MATH**: 18.7 vs 13.5
- **GSM8K**: 63.4 vs 58.4
- **HumanEval**: 42.7 vs 28.7
- **MBPP**: 57.4 vs 45.6
- **BBH**: 68.7 vs 62.9
- **Chinese benchmarks**: Significant lead on C-Eval, CMMLU, CHID, CMath

### Chat Model Performance

| Benchmark | DeepSeek 67B Chat | GPT-3.5-turbo |
|-----------|-------------------|---------------|
| MT-Bench | 8.35 (8.76 w/ DPO) | 8.39 |
| AlignBench (Chinese) | 6.43 (6.69 w/ DPO) | 6.08 |

The 67B Chat model with DPO achieves **MT-Bench 8.76**, approaching GPT-4 territory.

---

## Infrastructure

- **Framework**: HAI-LLM (in-house lightweight framework)
- **Parallelism**: Data + Tensor + Sequence + 1F1B Pipeline parallelism
- **Optimizations**:
  - Flash Attention
  - ZeRO-1 for optimizer state partitioning
  - Computation/communication overlapping
  - Fused kernels (LayerNorm, GEMM, Adam)
  - In-place cross-entropy (bf16->fp32 on-the-fly in CUDA kernel)
- **Checkpointing**: Async save every 5 minutes
- **Inference**: vLLM for generative tasks, continuous batching for non-generative

---

## Related Pages

- [[11_deepseek_v2_analysis]] — Successor with MLA attention and MoE architecture
- [[12_deepseek_v3_analysis]] — Further scaling with FP8 training and expert parallelism
- [[14_deepseek_r1_analysis]] — RL-based reasoning model built on DeepSeek-V3-Base
- [[18_deepseek_math_v2_analysis]] — Mathematical reasoning specialization
- [[15_deepseek_coder_analysis]] — Code intelligence specialization
- [[16_deepseek_coder_v2_analysis]] — MoE code model with 338 languages
- [[17_deepseek_math_analysis]] — Math reasoning with GRPO origin
- [[22_deepseek_prover_analysis]] — Formal theorem proving in Lean 4
- [[21_deepseek_vl_analysis]] — Vision-language model with hybrid encoder
- [[20_deepseek_moe_analysis]] — MoE architecture details
- [[25_mhc_analysis]] — Manifold-Constrained Hyper-Connections used in DeepSeek-V3 MoE
- [[29_engram_analysis]] — DeepSeek Engram memory mechanism
- [[llm_initiliaze_analysis]] — Weight initialization practices referenced in DeepSeek training
