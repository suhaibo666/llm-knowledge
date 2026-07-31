# DeepSeek-Coder-V2 Analysis

**Source:** `raw/05_model_families/deepseek/DeepSeek_Coder_V2-2406.11931.pdf`  
**Date Ingested:** 2026-04-21  
**Authors:** DeepSeek-AI  
**Published:** June 2024

---

## Overview

DeepSeek-Coder-V2 is an open-source **Mixture-of-Experts (MoE)** code language model that achieves performance comparable to GPT-4 Turbo in code-specific tasks. Built on the DeepSeek-V2 architecture, it is further pre-trained from a DeepSeek-V2 intermediate checkpoint with an additional **6 trillion tokens** of code and math data. It expands language support from 87 to **338 programming languages** and context length from 16K to **128K tokens**.

Key achievements:
- **HumanEval**: 90.2% (new SOTA for open-source)
- **MBPP+**: 76.2%
- **MATH**: 75.7%
- **SWE-bench**: >10% (first open-source model to exceed this threshold)

---

## Model Specifications

| Attribute | DeepSeek-Coder-V2-Lite | DeepSeek-Coder-V2 |
|-----------|------------------------|-------------------|
| Total Parameters | 16B | 236B |
| Activated Parameters | 2.4B | 21B |
| Architecture | DeepSeek-V2-Lite | DeepSeek-V2 |
| Context Length | 128K | 128K |
| Supported Languages | 338 | 338 |
| Pre-training Tokens | 4.2T + 6T | 4.2T + 6T |
| FIM Training | Enabled | Disabled |

---

## Data Collection

### Composition

- **60%** source code
- **10%** math corpus
- **30%** natural language corpus (from DeepSeek-V2)

### Code Corpus Expansion

| Source | Tokens | Description |
|--------|--------|-------------|
| GitHub + Common Crawl (code) | 1,170B | 338 programming languages |
| Code-related text (markdown, issues) | 185B | |
| Math-related web text | 221B | ~2x DeepSeekMath corpus |
| DeepSeek-V2 original corpus | 4,200B | General language |
| **Total exposure** | **~10.2T** | |

**Data quality validation**: A 1B parameter model trained on the new corpus achieves 37.2% on HumanEval (vs. 30.5% with old corpus) and 54.0% on MBPP (vs. 44.6%).

### Collection Pipeline

Follows the same iterative fastText-based pipeline as DeepSeekMath:
1. Seed corpus: StackOverflow, PyTorch docs, StackExchange
2. Train fastText classifier with BPE tokenization
3. Recall code/math-related web pages from Common Crawl
4. Domain annotation and iterative refinement (3 iterations)
5. Additional GitHub-specific collection (2 iterations)

---

## Training Policy

### Architecture

Identical to DeepSeek-V2/V2-Lite with MLA (Multi-head Latent Attention) and DeepSeekMoE routing.

**Notable modification**: Reverted from exponential normalization to conventional normalization due to training instability and gradient spikes observed with the exponential variant.

### Training Objectives

- **16B model**: Next-Token-Prediction + Fill-in-the-Middle (FIM) at 0.5 rate (PSM mode)
- **236B model**: Next-Token-Prediction only

FIM format:
```
<｜fim_begin｜> f_pre <｜fim_hole｜> f_suf <｜fim_end｜> f_middle <|eos_token|>
```

### Hyperparameters

- Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.95$, weight_decay=0.1)
- LR scheduler: Cosine decay with 2,000 warm-up steps, final LR = 10% of initial
- Total pre-training tokens: 10.2T (4.2T from DeepSeek-V2 + 6T new)

### Long Context Extension

Using **YaRN** with same hyperparameters as DeepSeek-V2 ($s=40, \alpha=1, \beta=32$):

| Stage | Sequence Length | Batch Size | Steps |
|-------|----------------|------------|-------|
| 1 | 32K | 1152 | 1,000 |
| 2 | 128K | 288 | 1,000 |

Passes "Needle In A Haystack" at all context lengths up to 128K.

---

## Alignment

### Supervised Fine-Tuning

- 20K code instruction data + 30K math data (from DeepSeek-Coder and DeepSeek-Math)
- General instruction data from DeepSeek-V2
- Total: 300M tokens
- LR: $5 \times 10^{-6}$, cosine schedule, 100 warm-up steps
- Batch size: 1M tokens, total 1B tokens

### Reinforcement Learning

**GRPO** (Group Relative Policy Optimization) is employed for RL alignment.

**Reward modeling**: For code tasks, a trained reward model is used rather than raw compiler 0-1 feedback, because some prompts have limited test case coverage. The reward model provides more robust signals with better generalization.

**Prompts**: ~40K filtered code and math prompts with test cases.

---

## Evaluation Highlights

### Code Generation

| Benchmark | DS-Coder-V2-Instruct (236B) | GPT-4-Turbo | Claude-3-Opus |
|-----------|----------------------------|-------------|---------------|
| HumanEval | **90.2%** | 88.2% | 84.2% |
| MBPP+ | **76.2%** | 72.2% | 72.0% |
| LiveCodeBench | **43.4%** | 45.7% | 34.6% |
| SWE-bench | **12.7%** | 18.3% | 11.7% |

### Competitive Programming

| Benchmark | DS-Coder-V2-Instruct | GPT-4-Turbo | GPT-4o |
|-----------|---------------------|-------------|--------|
| LiveCodeBench (Overall) | **43.4%** | 45.7% | 43.4% |
| USACO | 12.1% | 12.3% | **18.8%** |

### Math Reasoning

| Benchmark | DS-Coder-V2-Instruct | GPT-4o | Gemini-1.5-Pro |
|-----------|---------------------|--------|----------------|
| MATH | **75.7%** | **76.6%** | 73.4% |
| GSM8K | **94.9%** | 93.7% | 93.0% |
| AIME 2024 | **43.4%** | 40.2% | 35.8% |

### Repository-Level Code Completion (RepoBench)

DeepSeek-Coder-V2-Lite-Base (16B, 2.4B active) matches DeepSeek-Coder-Base 33B in Python repo-level completion, despite having only ~7% of the active parameters.

---

## Related Pages

- [[15_deepseek_coder_analysis]] — Predecessor dense code model (1.3B-33B)
- [[11_deepseek_v2_analysis]] — Base MoE architecture (MLA + DeepSeekMoE)
- [[17_deepseek_math_analysis]] — Math data pipeline and GRPO origin
- [[14_deepseek_r1_analysis]] — GRPO algorithm used for RL alignment
- [[20_deepseek_moe_analysis]] — DeepSeekMoE architecture deep dive

