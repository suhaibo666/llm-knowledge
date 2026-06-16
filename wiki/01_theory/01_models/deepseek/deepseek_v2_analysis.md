# DeepSeek-V2 Analysis

**Source:** `raw/05_model_families/deepseek/DeepSeek_V2-2405.04434.pdf`  
**Date Ingested:** 2026-04-20  
**Authors:** DeepSeek-AI  
**Published:** May 2024

---

## Overview

DeepSeek-V2 is a Mixture-of-Experts (MoE) language model that introduces two major architectural innovations: **Multi-head Latent Attention (MLA)** for efficient inference and **DeepSeekMoE** for economical training. With 236B total parameters but only 21B activated per token, it achieves stronger performance than its dense predecessor DeepSeek 67B while saving 42.5% training costs and boosting inference throughput by 5.76x.

---

## Model Specifications

| Attribute | Value |
|-----------|-------|
| Total Parameters | 236B |
| Activated Parameters | 21B per token |
| Context Length | 128K tokens (extended from 4K via YaRN) |
| Transformer Layers | 60 |
| Hidden Dimension | 5120 |
| Attention Heads | 128 ($d_h = 128$) |
| KV Compression Dim ($d_c$) | 512 |
| Query Compression Dim ($d'_c$) | 1536 |
| Decoupled RoPE Dim ($d^R_h$) | 64 |
| Shared Experts | 2 |
| Routed Experts | 160 |
| Activated Routed Experts | 6 |
| Expert Intermediate Dim | 1536 |

---

## Multi-Head Latent Attention (MLA)

### Motivation

Standard Multi-Head Attention (MHA) requires caching $2 n_h d_h l$ elements per token during inference, which becomes a severe bottleneck. GQA and MQA reduce cache at the cost of performance. MLA achieves the best of both worlds: **stronger than MHA** with **significantly less KV cache**.

### Low-Rank Key-Value Joint Compression

The core idea is to jointly compress keys and values into a single low-dimensional latent vector:

$$
c^{KV}_t = W^{DKV} h_t \quad \text{(down-projection)}
$$

$$
k^C_t = W^{UK} c^{KV}_t, \quad v^C_t = W^{UV} c^{KV}_t \quad \text{(up-projections)}
$$

During inference, only $c^{KV}_t \in \mathbb{R}^{d_c}$ needs to be cached (where $d_c \ll d_h n_h$). The up-projection matrices $W^{UK}$ and $W^{UV}$ can be **absorbed** into $W^Q$ and $W^O$ respectively, so no additional computation is needed.

For DeepSeek-V2, $d_c = 4 d_h$ and $d^R_h = d_h / 2$, yielding KV cache equivalent to **GQA with only 2.25 groups**, but with stronger performance than MHA.

### Query Compression (Training Only)

To reduce activation memory during training (not inference), queries are also low-rank compressed:

$$
c^Q_t = W^{DQ} h_t, \quad q^C_t = W^{UQ} c^Q_t
$$

### Decoupled Rotary Position Embedding (RoPE)

Standard RoPE is incompatible with low-rank KV compression because the position-sensitive RoPE matrix would prevent $W^{UK}$ from being absorbed into $W^Q$.

**Solution**: Use additional small-dimensional queries $q^R_{t,i}$ and a shared key $k^R_t$ to carry RoPE, while the compressed components remain position-agnostic:

$$
q_{t,i} = [q^C_{t,i};\; q^R_{t,i}], \quad k_{t,i} = [k^C_{t,i};\; k^R_t]
$$

The attention computation becomes:

$$
o_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j \left( \frac{q_{t,i}^T k_{j,i}}{\sqrt{d_h + d^R_h}} \right) v^C_{j,i}
$$

### KV Cache Comparison

| Mechanism | Cache per Token | Capability |
|-----------|----------------|------------|
| MHA | $2 n_h d_h l$ | Strong |
| GQA | $2 n_g d_h l$ | Moderate |
| MQA | $2 d_h l$ | Weak |
| **MLA** | $(d_c + d^R_h) l \approx 9 d_h l$ | **Stronger** |

---

## DeepSeekMoE Architecture

### Core Design

1. **Fine-grained expert segmentation**: More experts with smaller intermediate dimensions enable higher specialization
2. **Shared expert isolation**: Some experts are always activated to capture common knowledge, reducing redundancy among routed experts

### Computation

$$
h'_t = u_t + \sum_{i=1}^{N_s} \text{FFN}^{(s)}_i(u_t) + \sum_{i=1}^{N_r} g_{i,t} \cdot \text{FFN}^{(r)}_i(u_t)
$$

where $g_{i,t}$ is the top-$K_r$ gated value based on token-to-expert affinity $s_{i,t} = \text{Softmax}_i(u_t^T e_i)$.

### Device-Limited Routing

To bound MoE communication costs under expert parallelism:

- Each token's target experts are distributed on at most $M$ devices ($M=3$ for DeepSeek-V2)
- First select $M$ devices with highest affinity scores
- Then perform top-$K$ selection among experts on those $M$ devices only

### Auxiliary Losses for Load Balance

Three complementary balance losses:

1. **Expert-Level Balance Loss** ($\mathcal{L}_{\text{ExpBal}}$): Prevents routing collapse using $f_i \cdot P_i$ where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the mean routing probability

2. **Device-Level Balance Loss** ($\mathcal{L}_{\text{DevBal}}$): Ensures balanced computation across devices

3. **Communication Balance Loss** ($\mathcal{L}_{\text{CommBal}}$): Ensures each device receives roughly equal amounts of tokens from others

Hyper-parameters: $\alpha_1 = 0.003$, $\alpha_2 = 0.05$, $\alpha_3 = 0.02$

### Token-Dropping Strategy

During training, tokens with the lowest affinity scores on each device are dropped until reaching the computational budget (capacity factor = 1.0). However, ~10% of training sequences are protected from dropping to maintain consistency.

---

## Pre-Training

### Data

- **8.1T tokens** (vs 2T for DeepSeek 67B)
- Increased Chinese data (~12% more Chinese than English tokens)
- Improved quality-based filtering
- Contentious content filtered to mitigate regional bias

### Hyperparameters

- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.95$, weight_decay=0.1)
- **Learning Rate**: Multi-step decay, max $2.4 \times 10^{-4}$
  - Warmup: 2K steps
  - Decay to 31.6% at ~60% tokens
  - Decay to 10% at ~90% tokens
- **Batch Size**: Gradually increased from 2304 to 9216 over first 225B tokens, then fixed at 9216
- **Sequence Length**: 4K (training), 32K (YaRN extension), 128K (inference)
- **Gradient Clipping**: 1.0

### Long Context Extension (YaRN)

- Applied to the decoupled shared key $k^R_t$ (which carries RoPE)
- Scale $s = 40$, $\alpha = 1$, $\beta = 32$, target max length = 160K
- Length scaling factor: $\sqrt{t} = 0.0707 \ln s + 1$
- Additional 1000 steps training at 32K sequence length
- Passes "Needle In A Haystack" at 128K context

### Training Efficiency

| Metric | DeepSeek 67B (Dense) | DeepSeek-V2 (MoE) | Improvement |
|--------|---------------------|-------------------|-------------|
| Training Cost (per 1T tokens) | 300.6K GPU hours | 172.8K GPU hours | **-42.5%** |
| KV Cache | Baseline | -93.3% | |
| Max Generation Throughput | Baseline | 5.76x | |
| Prompt Throughput | Baseline | >100K tokens/sec | |

**Infrastructure**: 16-way zero-bubble pipeline parallelism + 8-way expert parallelism + ZeRO-1 data parallelism. **No tensor parallelism needed** due to small activated parameter count.

---

## Alignment Pipeline

### SFT

- 1.5M instances (1.2M helpful + 0.3M safety)
- Improved data quality vs DeepSeek 67B to reduce hallucination
- 2 epochs, LR = $5 \times 10^{-6}$

### RL: Group Relative Policy Optimization (GRPO)

DeepSeek-V2 adopts GRPO (from DeepSeekMath) which **eliminates the critic model**:

$$
J_{GRPO}(\theta) = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right) A_i \right) - \beta D_{KL}(\pi_\theta \| \pi_{ref}) \right]
$$

Advantage is computed from group rewards:

$$
A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}
$$

### Two-Stage RL Strategy

1. **Reasoning Alignment**: Train reward model $RM_{reasoning}$ on code/math, optimize policy with reasoning feedback
2. **Human Preference Alignment**: Multi-reward framework combining helpfulness, safety, and rule-based rewards

### Engineering Optimizations for RL

1. Hybrid engine with different parallel strategies for training vs inference
2. vLLM backend with large batch sizes for fast inference
3. CPU offloading scheduling for model loading/unloading

---

## Evaluation Highlights

### Base Model

With only 21B activated parameters, DeepSeek-V2 achieves top-tier open-source performance:

| Benchmark | DeepSeek-V2 | LLaMA3 70B | Mixtral 8x22B | Qwen1.5 72B |
|-----------|------------|------------|---------------|-------------|
| MMLU | 78.5 | 78.9 | 77.6 | 77.2 |
| BBH | 78.9 | 81.0 | 78.9 | 59.9 |
| HumanEval | 48.8 | 48.2 | 53.1 | 43.9 |
| MATH | 43.6 | 42.2 | 42.5 | 41.4 |
| GSM8K | 79.2 | 83.0 | 80.3 | 77.9 |
| C-Eval | 81.7 | 67.5 | 59.6 | 83.7 |

### Chat Model (RL)

| Benchmark | DeepSeek-V2 Chat (RL) | LLaMA3 70B Inst. |
|-----------|----------------------|------------------|
| MT-Bench | 8.97 | 8.95 |
| AlpacaEval 2.0 (LC Win Rate) | 38.9 | 34.4 |
| AlignBench (Chinese) | 7.91 | - |
| HumanEval | 81.1 | 76.2 |
| MATH | 53.9 | 48.5 |
| GSM8K | 92.2 | 93.2 |

---

## Related Pages

- [[deepseek_llm_analysis]] — Predecessor dense model (7B/67B) with scaling law studies
- [[deepseek_v3_analysis]] — Successor scaling to 671B MoE with FP8 training
- [[deepseek_v4_analysis]] — Successor with CSA/HCA hybrid attention replacing MLA, million-token context
- [[deepseek_r1_analysis]] — RL-based reasoning model built on DeepSeek-V3-Base
- [[deepseek_moe_analysis]] — Detailed MoE architecture analysis
- [[deepseek_coder_analysis]] — Code-specific model based on DeepSeek-V2 intermediate checkpoint
- [[deepseek_coder_v2_analysis]] — Successor MoE code model (236B) continuing from DeepSeek-V2
- [[deepseek_math_analysis]] — DeepSeekMath base model and GRPO algorithm
- [[deepseek_prover_analysis]] — Formal theorem proving with GRPO and proof assistant feedback
- [[deepseek_vl_analysis]] — Vision-language model built on DeepSeek LLM
- [[deepseek_math_v2]] — Mathematical reasoning with self-verification
- [[mHC]] — Manifold-Constrained Hyper-Connections used in later DeepSeek-V3
- [[megatron_ep_analysis]] — Expert parallelism and MoE training infrastructure
