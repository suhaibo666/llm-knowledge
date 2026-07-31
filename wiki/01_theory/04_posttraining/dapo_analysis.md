# DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization — Analysis

**Source**: `raw/03_alignment/DAPO_Decoupled_Clip_Dynamic_Sampling-2503.14476.pdf`
**Authors**: ByteDance Seed, Tsinghua AIR, HKU, SIA-Lab
**Published**: arXiv:2503.14476 | Mar 2025

---

## Core Contribution

Proposes **DAPO**, an open-source large-scale RL system for LLM reasoning that achieves **50 points on AIME 2024** using Qwen2.5-32B base model — outperforming DeepSeek-R1-Zero-Qwen-32B (47 points) with only 50% of the training steps. Reveals **four key techniques** that make large-scale LLM RL work, which were previously concealed by OpenAI and DeepSeek.

## Problem: Why Naive GRPO Fails at Scale

Starting from naive GRPO on Qwen2.5-32B, initial results achieved only **30 points on AIME** — far below DeepSeek's 47 points. Analysis revealed three key issues:

1. **Entropy collapse**: Policy becomes deterministic too early, limiting exploration
2. **Reward noise**: Truncated samples receive misleading punishment
3. **Training instability**: Gradient signals degrade as accuracy improves

## Four Key Techniques

### 1. Clip-Higher (Decoupled Clipping)

Standard PPO/GRPO's symmetric clip (`eps=0.2`) leaves low-probability "exploration" tokens almost no room to increase (0.01 → 0.012) while high-probability tokens saturate easily (0.9 → 1.08). DAPO decouples the bounds — `clip(r_t(theta), 1-eps_low, 1+eps_high)` with `eps_low=0.2`, `eps_high=0.28` (see Training Configuration below) — to keep entropy up and delay premature determinism. This asymmetric-clip notation is exactly D02 §2's unified $\epsilon_l/\epsilon_h$; the systemic risk (a higher upper bound does not always prevent entropy collapse) is analyzed in [[reasoning_rl_algorithm_evolution_analysis|D02]] §3.2.

### 2. Dynamic Sampling

All-correct or all-wrong response groups give a zero group-relative advantage — no gradient signal — and grow more common as training progresses, shrinking the effective batch. DAPO oversamples and filters to keep only prompts with a mix of correct/incorrect outputs (`0 < |{o_i correct}| < G`), trading extra rollout compute for a consistent effective batch size and lower gradient variance. The infra requirements this creates (buffer, resample budget, group-completeness bookkeeping) and its risk (the training distribution is now filtered by current learnability) are analyzed in [[reasoning_rl_algorithm_evolution_analysis|D02]] §3.2.

### 3. Token-Level Policy Gradient Loss

**Problem**: GRPO uses **sample-level** loss — average tokens within each sample, then average across samples. This gives equal weight to short and long responses, causing:
- Long responses (often low quality, repetitive) get same weight as short ones
- Unhealthy increase in response length and entropy

**Solution**: Compute loss at **token level** — all tokens contribute equally regardless of which sample they belong to:

```
J_DAPO = E[1 / (sum_i |o_i|) * sum_i sum_t min(r_t * A_t, clip(...) * A_t)]
```

vs GRPO's sample-level:

```
J_GRPO = E[1/G * sum_i (1/|o_i|) sum_t min(r_t * A_t, clip(...) * A_t)]
```

**Effect**:
- Longer sequences have proportionally more influence
- Each token is equally rewarded/suppressed regardless of response length
- Healthier length growth, more stable entropy

### 4. Overlong Reward Shaping

**Problem**: Truncated samples (exceeding max length) receive a flat -1 reward, even if the reasoning process was correct but just too long. This introduces **reward noise** that confuses the model.

**Solution**: Two-stage approach:

1. **Overlong Filtering**: Mask loss for truncated samples entirely (don't penalize)
2. **Soft Overlong Punishment**: Length-aware penalty within a cache window:

```
R_length(y) = 0,                          if |y| <= L_max - L_cache
            = (L_max - L_cache - |y|) / L_cache, if L_max - L_cache < |y| <= L_max
            = -1,                         if |y| > L_max
```

**Effect**:
- Stable training without reward noise
- Gradual penalty encourages concise responses
- Correct reasoning not unfairly punished for length

## DAPO Algorithm

```
Input: initial policy pi_theta, reward model R, prompts D, eps_low, eps_high
for step = 1, ..., M:
    Sample batch D_b from D
    Update old policy: pi_theta_old <- pi_theta
    Sample G outputs {o_i} for each question q in D_b
    Compute rewards {r_i} for each output
    Filter out outputs with accuracy 0 or 1 (Dynamic Sampling)
    if buffer size < N: continue
    Compute advantages A_i,t = (r_i - mean) / std
    for iteration = 1, ..., mu:
        Update pi_theta by maximizing DAPO objective
Output: pi_theta
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-32B |
| Prompt batch size | 512 |
| Group size (G) | 16 |
| Mini-batch size | 512 (16 gradient updates per rollout) |
| Learning rate | 1e-6 (AdamW, constant) |
| eps_low | 0.2 |
| eps_high | 0.28 |
| Max generation length | 20,480 tokens |
| Soft punish cache | 4,096 tokens |
| KL coefficient | **0** (removed) |

## Why Remove KL Penalty?

In RLHF, KL penalty keeps the policy close to the SFT model. But for long-CoT reasoning:
- The model distribution naturally diverges significantly from the initial model
- The KL restriction is unnecessary and potentially harmful
- Rule-based rewards provide sufficient regularization

## Dataset: DAPO-Math-17K

- 17K math prompts with integer answers
- Transformed from various formats (expressions, formulas) to integers for easy parsing
- Example: if answer is (a + sqrt(b))/c, modify question so answer becomes a+b+c

## Progressive Results

| Configuration | AIME 2024 avg@32 |
|--------------|-----------------|
| DeepSeek-R1-Zero-Qwen-32B | 47 |
| Naive GRPO | 30 |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-level Loss | 42 |
| + Dynamic Sampling (**DAPO**) | **50** |

## Key Insights for Large-Scale RL

1. **RL is complex systems engineering**: Changes to any subsystem propagate unpredictably
2. **Monitor intermediate metrics**: Length, reward, entropy, generation probability
3. **Length doesn't always increase**: Can stagnate or decline during training — this is normal
4. **Training reward != validation accuracy**: Models can overfit to training set rewards
5. **Entropy must be balanced**: Too low = no exploration, too high = no convergence

## Relationship to Other Methods

DAPO's position relative to PPO/GRPO/Dr. GRPO/GSPO/SAO across optimization unit, advantage estimator, ratio/clip and sampling structure is tabulated in [[reasoning_rl_algorithm_evolution_analysis|D02]] §4 — D02's table does not track a reward-shaping dimension. On that dimension DAPO is the outlier among these methods: it is the only one that applies **soft overlong reward shaping** (a length-aware penalty inside a cache window near the max-length cutoff, see §4 above) rather than a flat truncation penalty or no shaping at all.

## Impact

DAPO is the **first fully open-source** large-scale RL system for LLM reasoning that matches/exceeds DeepSeek-R1 results. All code, data, and training details are released, enabling reproducibility and further research.

> **2026-07-27 更新说明**：上面的 AIME 数字属于论文完整 recipe 与固定模型/采样条件，不应外推为单一组件收益。DAPO 与 Dr. GRPO、GSPO、SAO 的统计单位和系统约束对照见 [[reasoning_rl_algorithm_evolution_analysis|D02]]；dynamic sampling 的 buffer/backpressure 见 [[03_posttraining/05_posttraining_infra_mechanism_analysis|D05]]。

## Related Pages

- [[reasoning_rl_algorithm_evolution_analysis|D02 演进权威页]] — 公式演进、系统约束与跨算法对照
- [[grpo_analysis]] — GRPO algorithm that DAPO improves upon
- [[ppo_analysis]] — PPO foundation
- [[deepseek_r1_analysis]] — DeepSeek-R1 comparison
- [[verl_rl_algorithms_analysis]] — verl 源码级实现(注册表 + config key→代码锚点)
- [[03_posttraining/index]] — D00–D11 后训练统一学习域
- [[01_theory/index]]
