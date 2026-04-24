# GSPO: Group Sequence Policy Optimization — Analysis

**Source**: `raw/03_alignment/GSPO_Group_Sequence_Policy_Optimization-2507.18071.pdf`
**Authors**: Qwen Team, Alibaba Inc.
**Published**: arXiv:2507.18071 | Jul 2025

---

## Core Contribution

Proposes **GSPO**, a sequence-level RL algorithm that fixes the fundamental instability of GRPO by using **sequence-level importance ratios** instead of token-level ratios. GSPO stabilizes MoE RL training, achieves superior performance, and contributed to Qwen3 model improvements.

## Problem: Why GRPO is Fundamentally Ill-Posed

GRPO applies token-level importance weights:

```
w_i,t(theta) = pi_theta(y_i,t | x, y_i,<t) / pi_theta_old(y_i,t | x, y_i,<t)
```

**Critical flaw**: This weight is based on a **single sample** from each next-token distribution. Importance sampling requires averaging over multiple samples (N >> 1) from the behavior distribution to correct for distributional mismatch. With a single sample, the token-level weight:

1. Fails to perform distribution correction
2. Introduces **high-variance noise** into gradients
3. Noise **accumulates over long sequences**
4. Exacerbated by clipping mechanism
5. Leads to **irreversible model collapse** on large models

**Core principle violated**: The unit of optimization objective should match the unit of reward. Since reward is granted to the entire sequence, token-level off-policy correction is problematic.

## GSPO Algorithm

### Sequence-Level Importance Ratio

```
s_i(theta) = (pi_theta(y_i | x) / pi_theta_old(y_i | x))^(1/|y_i|)
           = exp(1/|y_i| * sum_t log(pi_theta(y_i,t | x, y_i,<t) / pi_theta_old(y_i,t | x, y_i,<t)))
```

**Length normalization** is critical:
- Reduces variance
- Keeps ratios in a unified numerical range
- Without it, a few token likelihood changes cause dramatic fluctuations

### GSPO Objective

```
J_GSPO(theta) = E[1/G * sum_i min(s_i(theta) * A_i, clip(s_i(theta), 1-eps, 1+eps) * A_i)]
```

where:
- `s_i(theta)` = sequence-level importance ratio (length-normalized)
- `A_i = (r(x, y_i) - mean({r})) / std({r})` = group-relative advantage
- All tokens in a response share the same advantage

### Key Difference from GRPO

| Aspect | GRPO | GSPO |
|--------|------|------|
| Importance ratio | Token-level: `w_i,t` | Sequence-level: `s_i` |
| Clipping | Per token | Per sequence |
| Gradient weighting | Unequal token weights | Equal token weights |
| Stability | Unstable on large models | Stable |

### Gradient Comparison

**GRPO gradient**:
```
nabla J_GRPO = E[1/G * sum_i A_i * 1/|y_i| * sum_t w_i,t * nabla log pi(y_i,t)]
```
Tokens weighted unequally by `w_i,t`, which varies among (0, 1+eps] or [1-eps, +inf).

**GSPO gradient**:
```
nabla J_GSPO = E[1/G * sum_i s_i * A_i * 1/|y_i| * sum_t nabla log pi(y_i,t)]
```
All tokens in a response weighted **equally** — eliminates GRPO's instability factor.

### GSPO-token Variant

For scenarios requiring token-level advantage customization (e.g., multi-turn RL):

```
s_i,t(theta) = sg[s_i(theta)] * pi_theta(y_i,t) / sg[pi_theta(y_i,t)]
```

where `sg[·]` is stop-gradient (detach). Numerically equal to `s_i` but allows per-token advantage adjustment.

## Clipping Range Difference

GSPO and GRPO use **different orders of magnitude** for clipping:

| Algorithm | eps_low | eps_high |
|-----------|---------|----------|
| GRPO | 0.2 | 0.27 |
| **GSPO** | **3e-4** | **4e-4** |

This is because GSPO's sequence-level ratio has fundamentally different numerical properties than GRPO's token-level ratio.

## Empirical Results

GSPO was tested on Qwen3-30B-A3B-Base (MoE model):

- **Superior training reward curves** compared to GRPO
- **Better performance** on AIME'24, LiveCodeBench, CodeForces
- **Stable MoE RL training** — GRPO often collapses on MoE models
- Contributed to **Qwen3 model improvements**

## Why GSPO Matters

1. **Fixes fundamental flaw**: Token-level importance sampling in GRPO is theoretically unsound
2. **Stabilizes MoE training**: MoE models are particularly sensitive to gradient noise
3. **Simplifies infrastructure**: No need for complex stabilization strategies
4. **Scales better**: Sequence-level optimization is more principled for long responses

## Relationship to Other Methods

| Method | Importance Ratio | Clipping Level | Stability |
|--------|-----------------|----------------|-----------|
| PPO | Token-level | Token | Moderate |
| GRPO | Token-level | Token | Poor (large models) |
| DAPO | Token-level | Token (decoupled) | Good |
| **GSPO** | **Sequence-level** | **Sequence** | **Excellent** |

## Related Pages

- [[grpo_analysis]] — GRPO algorithm that GSPO fixes
- [[dapo_analysis]] — DAPO improvements to GRPO
- [[ppo_analysis]] — PPO foundation
- [[llm/overview]]
