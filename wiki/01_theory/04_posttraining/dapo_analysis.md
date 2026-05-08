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

**Problem**: Standard PPO/GRPO uses symmetric clipping (epsilon = 0.2). The upper clip restricts exploration tokens:
- For a token with pi_old = 0.01, max increase is to 0.012 (barely any room)
- For a token with pi_old = 0.9, max increase is to 1.08 (easily saturated)

**Solution**: Decouple lower and upper clipping bounds:

```
clip(r_t(theta), 1 - eps_low, 1 + eps_high)
```

where `eps_low = 0.2` (unchanged) and `eps_high = 0.28` (increased).

**Effect**:
- Maintains entropy throughout training
- Allows low-probability "exploration" tokens to increase more freely
- Prevents premature determinism

### 2. Dynamic Sampling

**Problem**: When all G outputs for a prompt are correct (accuracy = 1) or all wrong (accuracy = 0), the group-relative advantage becomes zero — providing **no gradient signal**. As training progresses, more prompts reach accuracy = 1, reducing effective batch size.

**Solution**: Oversample and filter out prompts with accuracy = 0 or 1:

```
Constraint: 0 < |{o_i | is_equivalent(a, o_i)}| < G
```

Only keep prompts where some outputs are correct and some are wrong — these provide the most informative gradients.

**Effect**:
- Consistent number of effective prompts per batch
- Lower gradient variance
- Faster convergence (fewer training steps needed)

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

| Method | Clipping | Sampling | Loss Level | Reward Shaping |
|--------|----------|----------|------------|----------------|
| PPO | Symmetric | Fixed | Sample | None |
| GRPO | Symmetric | Fixed | Sample | None |
| **DAPO** | **Decoupled** | **Dynamic** | **Token** | **Soft overlong** |

## Impact

DAPO is the **first fully open-source** large-scale RL system for LLM reasoning that matches/exceeds DeepSeek-R1 results. All code, data, and training details are released, enabling reproducibility and further research.

## Related Pages

- [[grpo_analysis]] — GRPO algorithm that DAPO improves upon
- [[ppo_analysis]] — PPO foundation
- [[deepseek_r1_analysis]] — DeepSeek-R1 comparison
- [[01_theory/index]]
