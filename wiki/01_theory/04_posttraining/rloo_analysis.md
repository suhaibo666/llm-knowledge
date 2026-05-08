# RLOO: REINFORCE Leave-One-Out — Analysis

**Source**: `raw/03_alignment/RLOO_REINFORCE_Leave_One_Out-2402.14740.pdf`
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

**RLOO is the theoretical foundation for GRPO** — both use group-relative advantages without a value function.

## Experimental Results

- RLOO matches or exceeds PPO on multiple RLHF tasks
- 2.5x faster training (no value function, no critic)
- Fewer hyperparameters to tune
- More stable training dynamics

## Impact

RLOO demonstrated that **simple REINFORCE with a good baseline** is sufficient for RLHF, paving the way for:
- GRPO (which added clipping and normalization)
- DAPO (which further improved clipping and sampling)
- The broader trend toward simpler RL algorithms for LLMs

## Related Pages

- [[grpo_analysis]] — GRPO builds on RLOO's insight
- [[ppo_analysis]] — PPO that RLOO replaces
- [[dapo_analysis]] — DAPO further improvements
- [[01_theory/index]]
