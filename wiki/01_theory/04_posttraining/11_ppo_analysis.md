---
title: "PPO: Proximal Policy Optimization — Analysis"
---

# PPO: Proximal Policy Optimization — Analysis

**Source**: `raw/01_theory/04_posttraining/PPO_Proximal_Policy_Optimization-1707.06347.md`
**Authors**: Schulman, Wolski, Dhariwal, Radford, Klimov (OpenAI)
**Published**: arXiv:1707.06347 | Jul 2017

---

## Core Contribution

Proposes **Proximal Policy Optimization (PPO)**, a family of policy gradient methods that alternate between sampling data from the environment and optimizing a **surrogate objective** using stochastic gradient ascent. PPO achieves the stability of trust region methods (TRPO) while being much simpler to implement and more general.

## Problem with Standard Policy Gradient

Standard policy gradient computes:

```
L_PG(theta) = E_t[log pi_theta(a_t|s_t) * A_t]
```

**Problem**: Performing multiple optimization steps on the same trajectory leads to destructively large policy updates. The data becomes "off-policy" after the first update, making subsequent gradients unreliable.

## TRPO Background

TRPO addresses this by maximizing a surrogate objective subject to a KL divergence constraint:

```
max_theta E_t[pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t) * A_t]
subject to: E_t[KL(pi_theta_old(·|s_t), pi_theta(·|s_t))] <= delta
```

**Problem**: TRPO is complex to implement (requires conjugate gradient + line search), incompatible with architectures with noise (dropout) or parameter sharing, and doesn't work well with multi-task learning.

## PPO Solutions

### 1. Clipped Surrogate Objective (PPO-Clip)

The main contribution:

```
L_CLIP(theta) = E_t[min(r_t(theta) * A_t, clip(r_t(theta), 1-epsilon, 1+epsilon) * A_t)]
```

where `r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)` is the probability ratio.

**How clipping works**:
- When A_t > 0 (action was good): clip at 1+epsilon, preventing overly large updates that increase probability too much
- When A_t < 0 (action was bad): clip at 1-epsilon, preventing overly large updates that decrease probability too much
- Takes the **minimum** of clipped and unclipped — a pessimistic lower bound

**Key property**: L_CLIP(theta) = L_CPI(theta) to first order around theta_old, but diverges as theta moves away.

**Default epsilon**: 0.2

### 2. Adaptive KL Penalty (PPO-Penalty)

Alternative approach using KL penalty instead of clipping:

```
L_KLPEN(theta) = E_t[r_t(theta) * A_t - beta * KL(pi_theta_old(·|s_t), pi_theta(·|s_t))]
```

With adaptive beta:
- If KL < KL_target / 1.5: beta = beta / 2 (penalty too strong)
- If KL > KL_target * 1.5: beta = beta * 2 (penalty too weak)

**Default KL_target**: 0.01

## PPO Algorithm (Actor-Critic Style)

```
for iteration = 1, 2, ...:
    for actor = 1, 2, ..., N:
        Run policy pi_theta_old for T timesteps
        Compute advantage estimates A_1, ..., A_T
    Optimize surrogate L w.r.t. theta, K epochs, minibatch size M <= NT
    theta_old <- theta
```

**Typical hyperparameters**:
- Horizon T: 2048 (continuous), 512 (Atari)
- Epochs K: 10-15
- Minibatch size: 64 (continuous), 4096 (Atari)
- Adam learning rate: 3e-4
- Discount gamma: 0.99
- GAE lambda: 0.95

## Advantage Estimation

PPO uses **Generalized Advantage Estimation (GAE)**:

```
A_t^GAE(gamma, lambda) = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}
```

where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) is the TD error.

GAE interpolates between:
- lambda = 0: 1-step TD (low variance, high bias)
- lambda = 1: Monte Carlo (high variance, low bias)

## Comparison Results

### Surrogate Objective Comparison (7 MuJoCo tasks)

| Method | Avg Normalized Score |
|--------|---------------------|
| No clipping/penalty | -0.39 (fails catastrophically) |
| Clipping, epsilon=0.2 | **0.82** (best) |
| Clipping, epsilon=0.1 | 0.76 |
| Adaptive KL, dtarg=0.01 | 0.74 |
| Fixed KL, beta=3 | 0.72 |

### vs Other Algorithms

- PPO outperforms TRPO, A2C, CEM, vanilla PG on most continuous control tasks
- On Atari: PPO wins 30/49 games on learning speed, 19/49 on final performance
- Overall: **best balance of sample complexity, simplicity, and wall-time**

## Why PPO Works

1. **Multiple epochs on same data**: Clipping prevents destructive updates, enabling K epochs of minibatch SGD
2. **Simple implementation**: Only a few lines of code change from vanilla policy gradient
3. **General applicability**: Works with dropout, parameter sharing, multi-task learning
4. **Robust hyperparameters**: epsilon=0.2 works well across diverse tasks

## PPO in LLM Training

PPO became the **standard algorithm for RLHF** (Reinforcement Learning from Human Feedback):
- InstructGPT (Ouyang et al., 2022) uses PPO to align LLMs with human preferences
- The KL penalty in RLHF PPO is between the policy and a reference model (not between old and new policy)
- DeepSeek-R1 later replaced PPO with GRPO for more efficient reasoning training

## Related Pages

- [[14_deepseek_r1_analysis]] — GRPO replaced PPO for reasoning training
- [[11_ppo_analysis]] — this page
- [[01_theory/index]]
