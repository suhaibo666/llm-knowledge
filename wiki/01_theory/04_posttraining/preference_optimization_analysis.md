# Preference Optimization Methods — DPO Family Analysis

**Covers**: DPO, IPO, SimPO, ORPO, KTO, MODPO
**Domain**: LLM Alignment & Preference Optimization

---

## Overview

After RLHF (SFT → RM → PPO) was established by InstructGPT, a family of **direct preference optimization** methods emerged that simplify or improve upon DPO. All share the goal of replacing the complex RLHF pipeline with simpler supervised losses.

---

## DPO (Direct Preference Optimization)

**Source**: Rafailov et al., Stanford, 2023 | arXiv:2305.18290

### Core Idea

Exploits the closed-form relationship between reward and optimal policy in RLHF:

```
r(x, y) = beta * log(pi(y|x) / pi_ref(y|x))
```

### Loss Function

```
L_DPO = -E[log(sigma(beta * log(pi(y_w|x)/pi_ref(y_w|x)) - beta * log(pi(y_l|x)/pi_ref(y_l|x))))]
```

### Key Properties

- Requires reference model pi_ref
- Only hyperparameter: beta (KL strength)
- Binary cross-entropy loss — no sampling, no value function
- Matches or exceeds PPO performance

### Limitations

- Needs a good reference model (usually SFT)
- Beta tuning can be tricky
- Performance sensitive to data quality

---

## IPO (Identity Preference Optimization)

**Source**: Azar et al., 2023 | arXiv:2310.12036

### Core Idea

Addresses DPO's theoretical limitation: DPO's implicit reward is not well-calibrated for out-of-distribution data. IPO uses a **regularized objective** that directly optimizes the policy without the reward reparameterization.

### Loss Function

```
L_IPO = E[(log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x)) - 1/(2*tau))^2]
```

where tau is a temperature parameter.

### Key Properties

- Squared loss instead of logistic loss
- Better theoretical guarantees
- Less sensitive to beta
- More robust to distributional shift

---

## SimPO (Simple Preference Optimization)

**Source**: Meng, Xia, Chen (UVA, Princeton), 2024 | arXiv:2405.14734

### Core Idea

**Eliminates the reference model entirely** by using average log probability as the implicit reward:

```
r(x, y) = (1/|y|) * log(pi(y|x))
```

### Loss Function

```
L_SimPO = -E[log(sigma(beta/|y_w| * log(pi(y_w|x)) - beta/|y_l| * log(pi(y_l|x)) - gamma))]
```

where gamma is a target reward margin.

### Key Properties

- **No reference model needed** — saves memory and compute
- Length normalization (average log prob) handles variable-length responses
- gamma controls the margin between chosen and rejected
- Simpler than DPO, often better performance

### Why It Works

- Average log probability better aligns with autoregressive generation
- Eliminates the need to store and compute reference model logits
- Length normalization prevents bias toward longer responses

---

## ORPO (Odds Ratio Preference Optimization)

**Source**: Hong, Lee, Thorne (KAIST), 2024 | arXiv:2403.07691

### Core Idea

**Monolithic preference optimization** — combines SFT and preference alignment into a single step, no reference model needed.

### Loss Function

```
L_ORPO = L_SFT + lambda * L_ODDS
```

where:

```
L_ODDS = -E[log(sigma(log(odds(y_w|x)) - log(odds(y_l|x))))]
```

and odds(y|x) = pi(y|x) / (1 - pi(y|x))

### Key Properties

- **Single training phase** — no separate SFT needed
- No reference model
- lambda controls preference strength
- Minor penalty for dispreferred style is sufficient

### Why It Works

- SFT naturally encourages preferred responses
- Odds ratio penalizes dispreferred responses
- Combined objective achieves both imitation and alignment

---

## KTO (Kahneman-Tversky Optimization)

**Source**: Ethayarajh, Xu, Muennighoff, Jurafsky, Kiela (Stanford, HuggingFace), 2024 | arXiv:2402.01306

### Core Idea

Based on **prospect theory** (Kahneman & Tversky, 1992): humans perceive gains and losses asymmetrically (loss aversion). Uses **binary preference labels** (good/bad) instead of pairwise comparisons.

### Loss Function

```
L_KTO = -E[z(x, y) * lambda(x) * sigma(beta * (r(x, y) - r_ref))]
```

where:
- z(x, y) = +1 for desirable outputs, -1 for undesirable
- lambda(x) = loss aversion coefficient (different for gains vs losses)
- r_ref = reference reward (KL term)

### Key Properties

- **Only needs binary labels** (good/bad), not pairwise comparisons
- Models human loss aversion (losses hurt more than equivalent gains help)
- More data-efficient — each sample provides a signal
- Better aligns with how humans actually evaluate outputs

### Data Collection Advantage

- Pairwise methods need 2+ outputs per prompt
- KTO needs only 1 output + binary label
- Easier to collect at scale

---

## MODPO (Multi-Objective DPO)

**Source**: 2023 | arXiv:2310.03708

### Core Idea

Extends DPO to **multiple objectives** simultaneously (e.g., helpfulness + harmlessness + truthfulness).

### Loss Function

```
L_MODPO = sum_k w_k * L_DPO_k
```

where each L_DPO_k is a DPO loss for objective k, and w_k are weights.

### Key Properties

- Handles multiple preference dimensions
- Can trade off between objectives
- Useful for complex alignment scenarios

---

## Comparison Summary

| Method | Reference Model | Data Format | Training Phases | Key Innovation |
|--------|----------------|-------------|-----------------|----------------|
| **RLHF/PPO** | Yes (RM) | Pairwise rankings | 3 (SFT→RM→PPO) | Original pipeline |
| **DPO** | Yes | Pairwise | 1 | Closed-form policy-reward |
| **IPO** | Yes | Pairwise | 1 | Squared loss, better theory |
| **SimPO** | **No** | Pairwise | 1 | Length-normalized avg log prob |
| **ORPO** | **No** | Pairwise | **1 (monolithic)** | Combines SFT + alignment |
| **KTO** | Yes | **Binary** | 1 | Prospect theory, loss aversion |
| **MODPO** | Yes | Pairwise (multi) | 1 | Multiple objectives |

## Practical Recommendations

| Scenario | Recommended Method |
|----------|-------------------|
| Standard alignment, have SFT model | DPO or SimPO |
| Want simplest pipeline | SimPO or ORPO |
| Limited compute (no ref model) | SimPO or ORPO |
| Binary preference data available | KTO |
| Multiple alignment objectives | MODPO |
| Theoretical guarantees important | IPO |

## Relationship to GRPO

GRPO (Group Relative Policy Optimization), used by DeepSeek-R1, is different from the DPO family:
- GRPO is still an **RL-based** method (like PPO)
- Uses group-relative advantages instead of a value function
- More sample-efficient than PPO for reasoning tasks
- DPO family methods are **offline** (fixed dataset), GRPO is **online** (samples during training)

## Related Pages

- [[dpo_analysis]] — Detailed DPO analysis
- [[instructgpt_rlhf_analysis]] — Original RLHF pipeline
- [[ppo_analysis]] — PPO algorithm
- [[deepseek_r1_analysis]] — GRPO usage in DeepSeek-R1
- [[01_theory/index]]
