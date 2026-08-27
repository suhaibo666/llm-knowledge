---
title: "RLOO: REINFORCE Leave-One-Out — Analysis"
---

# RLOO: REINFORCE Leave-One-Out — Analysis

**Source**: `raw/01_theory/04_posttraining/RLOO_REINFORCE_Leave_One_Out-2402.14740.md`
**Authors**: Ahmadian, Cremer, Gallé, Fadaee, Kreutzer, Pietquin, Üstün, Hooker (Cohere For AI)
**Published**: arXiv:2402.14740 | Feb 2024

---

## Core Contribution

Revisits **REINFORCE-style optimization** for RLHF, showing that simple REINFORCE with a leave-one-out baseline matches or exceeds PPO performance while being significantly simpler and cheaper. Challenges the assumption that PPO is necessary for RLHF.

## Key Insight: PPO is Over-Engineered for RLHF

PPO was designed for general RL tasks with:
- Stochastic environments
- Unknown reward functions
- Need for value function estimation

But RLHF has unique properties:
- **Fast simulation**: Generating text is fast and deterministic
- **Deterministic transitions**: Given a prompt, the environment doesn't change
- **Trajectory-level rewards**: Reward is given for the entire sequence, not per-token

These properties make PPO's complexity unnecessary.

## REINFORCE with Leave-One-Out Baseline

### Standard REINFORCE

```
nabla J = E[sum_t nabla log pi(a_t|s_t) * R]
```

High variance because R is a single scalar.

### REINFORCE with Baseline

```
nabla J = E[sum_t nabla log pi(a_t|s_t) * (R - b)]
```

where b is a baseline (usually value function) to reduce variance.

### Leave-One-Out Baseline (RLOO)

For a group of G samples from the same prompt:

```
A_i = R_i - (1/(G-1)) * sum_{j!=i} R_j
```

**No value function needed** — the baseline is computed from other samples in the group.

This is essentially the same as GRPO's advantage computation, but derived from a different perspective (REINFORCE vs. PPO).

## RLOO vs GRPO

| Aspect | RLOO | GRPO |
|--------|------|------|
| Derivation | REINFORCE + leave-one-out baseline | PPO without value function |
| Advantage | A_i = R_i - mean(R_{-i}) | A_i = (R_i - mean(R)) / std(R) |
| Clipping | None (original) | Yes |
| Normalization | None | Standard deviation |

**RLOO 与 GRPO 是同期独立工作**，共享「用组内其余样本构造 baseline、免去 value model」这一思路，但不存在先后继承关系。

> [!deprecated] 2026-08-10 更正（时序核对）
> 本行此前写作「**RLOO is the theoretical foundation for GRPO**」，与两页自印的 arXiv 编号冲突：GRPO 出自 DeepSeekMath **arXiv:2402.03300**，RLOO 为 **arXiv:2402.14740**——同为 2024 年 2 月，但 arXiv 编号按投稿顺序递增，`03300 < 14740`，即 GRPO 的提出**早于** RLOO 公开。两文各自把 leave-one-out baseline 追溯到更早的文献，不互为基础。下文 Impact 节的「paving the way for GRPO」同此更正。

## Experimental Results

- RLOO matches or exceeds PPO on multiple RLHF tasks
- 2.5x faster training (no value function, no critic)
- Fewer hyperparameters to tune
- More stable training dynamics

## Impact

RLOO demonstrated that **simple REINFORCE with a good baseline** is sufficient for RLHF。它与下列工作共同构成 2024 年「去 critic 化」这条线（**并列关系，非因果**，见上文时序更正）：
- GRPO（arXiv:2402.03300，早于本文公开；用组均值/标准差归一化 + clipping）
- DAPO（arXiv:2503.14476，在 GRPO 上改 clipping 与采样）
- The broader trend toward simpler RL algorithms for LLMs

## Related Pages

- [[20_grpo_analysis]] — GRPO builds on RLOO's insight
- [[11_ppo_analysis]] — PPO that RLOO replaces
- [[21_dapo_analysis]] — DAPO further improvements
- [[01_theory/index]]
