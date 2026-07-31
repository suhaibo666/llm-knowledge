# GSPO: Group Sequence Policy Optimization — Analysis

**Source**: `raw/03_alignment/GSPO_Group_Sequence_Policy_Optimization-2507.18071.pdf`
**Authors**: Qwen Team, Alibaba Inc.
**Published**: arXiv:2507.18071 | Jul 2025

---

## Core Contribution

Proposes **GSPO**, a sequence-level RL algorithm that fixes the fundamental instability of GRPO by using **sequence-level importance ratios** instead of token-level ratios. GSPO stabilizes MoE RL training, achieves superior performance, and contributed to Qwen3 model improvements.

## Problem: Why GRPO is Fundamentally Ill-Posed

GRPO's token-level importance weight `w_i,t = pi_theta(y_i,t|...) / pi_theta_old(y_i,t|...)` is based on a **single sample** from each next-token distribution, so it cannot perform the distribution correction importance sampling normally requires (N >> 1 samples). The resulting high-variance noise accumulates over long sequences, is exacerbated by clipping, and can produce irreversible collapse on large models — the core principle violated is that the unit of the optimization objective should match the unit that receives reward (the whole sequence, not each token). This diagnosis and GSPO's fix are unified with GRPO/DAPO/Dr. GRPO/SAO's own statistical-unit issues in [[reasoning_rl_algorithm_evolution_analysis|D02]] §3.4.

## GSPO Algorithm

### Sequence-Level Importance Ratio, Objective and Key Difference from GRPO

GSPO replaces the token-level ratio with a length-normalized **sequence-level** ratio `s_i(theta) = (pi_theta(y_i|x)/pi_theta_old(y_i|x))^(1/|y_i|)`, applied inside the same clipped-surrogate template as GRPO (all tokens in a response now share both the advantage and the clip decision). The exact formula, why length normalization is critical, and the full GRPO-vs-GSPO comparison (importance ratio/clipping granularity/gradient weighting/stability) are unified in [[reasoning_rl_algorithm_evolution_analysis|D02]] §3.4, §4 — this notation is D02 §2's general clipped surrogate with `s_i` substituted for the token ratio.

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

> **2026-07-27 更新说明**：本文的“稳定性”表述代表论文固定实验条件，不是所有模型/框架的无条件结论。sequence ratio/clip 的公式—batch schema 对照见 [[reasoning_rl_algorithm_evolution_analysis|D02]]，与 staleness/TIM 的组合语义见 [[on_policy_off_policy_staleness_analysis|D04]]。

## Relationship to Other Methods

GSPO's position relative to PPO/GRPO/DAPO/Dr. GRPO/SAO across importance-ratio granularity, clipping level and the systemic invariant each preserves is tabulated in [[reasoning_rl_algorithm_evolution_analysis|D02]] §4.

## Related Pages

- [[reasoning_rl_algorithm_evolution_analysis|D02 演进权威页]] — 公式演进、系统约束与跨算法对照
- [[grpo_analysis]] — GRPO algorithm that GSPO fixes
- [[dapo_analysis]] — DAPO improvements to GRPO
- [[ppo_analysis]] — PPO foundation
- [[verl_rl_algorithms_analysis]] — verl 源码级实现(注册表 + config key→代码锚点)
- [[03_posttraining/index]] — D00–D11 后训练统一学习域
- [[01_theory/index]]
