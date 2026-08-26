# GRPO: Group Relative Policy Optimization — Analysis

**Source**: DeepSeek R1 paper (arXiv:2501.12948) + DeepSeekMath paper (arXiv:2402.03300)
**Authors**: DeepSeek-AI
**Published**: Jan 2025 (R1), Feb 2024 (Math)

---

## Core Contribution

**GRPO (Group Relative Policy Optimization)** simplifies PPO by eliminating the value function and computing advantages from a group of outputs sampled from the same prompt. Used to train DeepSeek-R1-Zero through **pure RL without SFT**, enabling emergent reasoning capabilities.

## GRPO Algorithm

### Key Innovation: Group-Relative Advantages

GRPO replaces PPO's separate value function with a group-relative baseline: sample $G$ outputs per question, then normalize each output's reward by the group mean/std to get its advantage — the group mean itself serves as the baseline. The full formula, the two systemic constraints this creates (group is an indivisible statistical unit; all-0/all-1 reward groups carry no signal), and how DAPO/Dr. GRPO/GSPO/SAO each modify this estimator are unified in [[13_reasoning_rl_algorithm_evolution_analysis|D02 Reasoning RL 算法演进]] §3.1.

### GRPO Objective

```
J_GRPO(theta) = E[ (1/G) * sum_i min(r_i * A_i, clip(r_i, 1-eps, 1+eps) * A_i) ] - beta * D_KL(pi_theta || pi_ref)
```

The clip-and-advantage core matches D02 §2's unified clipped-surrogate shape (min of raw and clipped ratio×advantage) — see [[13_reasoning_rl_algorithm_evolution_analysis|D02]] for the general form and how later methods modify it. Two things D02's generic template does **not** carry, both original to GRPO's own formulation: the **`-beta * D_KL(pi_theta || pi_ref)`** regularization term added outside the clip (D02 §2 has no KL term at all), and the fact that GRPO's objective is itself already **token-level**: the ratio is `pi_theta(o_{i,t} | q, o_{i,<t}) / pi_theta_old(o_{i,t} | q, o_{i,<t})`, summed over `t = 1..|o_i|` under a `1/|o_i|` length normalization and a **symmetric** clip `[1-eps, 1+eps]` (DeepSeekMath arXiv:2402.03300v3 §4.1.1 Eq. 3). D02 §2's `r_{i,t}` notation is therefore faithful to the source, not a re-interpretation.

> [!deprecated] 2026-08-10 更正（回原文裁决）
> 本段此前称该比值是「a **whole-output** ratio with no token subscript」，并称论文伪代码「treats each sampled output `o_i` atomically」。对照 `raw/01_theory/04_posttraining/GRPO_Group_Relative_Policy_Optimization-2402.03300.md` **§4.1.1 式 (3)**，该说法**不成立**：式 (3) 的重要性比分子分母均为 `pi(o_{i,t} | q, o_{i,<t})`，外层为 `1/G Σ_{i=1..G} 1/|o_i| Σ_{t=1..|o_i|}`；**Algorithm 1 第 9 行**亦明写 "Compute `Â_{i,t}` for the **t-th token** of `o_i`"。该错误此前与 [[22_gspo_analysis]]（其全部论证建立在 GRPO 为 token 级之上）及 [[13_reasoning_rl_algorithm_evolution_analysis]] 的记号直接冲突，现以一手原文为准统一为 token 级。 GRPO's KL term uses a **low-variance unbiased estimator** (Schulman's k3 form):

```
D_KL(pi_theta || pi_ref) = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1
```

### Comparison to PPO

| Aspect | PPO | GRPO |
|--------|-----|-------|
| Value function | Required (separate model) | **Not needed** |
| Advantage computation | GAE from value function | Group-relative (mean/std) |
| Memory | Policy + Value + RM | Policy + RM only |
| Training stability | Can be unstable | More stable |
| Sample efficiency | Good | Better for reasoning |

## DeepSeek-R1-Zero Training

### Key Design Choices

1. **No SFT before RL** — bypass supervised fine-tuning to avoid limiting exploration
2. **Rule-based rewards** — reward only based on final answer correctness, not reasoning process
3. **Pure RL** — incentivize reasoning through self-evolution
4. **Long context** — 32K tokens initially, 65K tokens after step 8.2k

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-6 |
| KL coefficient (beta) | 0.001 |
| Sampling temperature | 1.0 |
| Group size (G) | 16 outputs per question |
| Batch size | 512 (32 questions x 16 outputs) |
| Max response length | 32K → 65K tokens |
| Total steps | 10,400 (1.6 epochs) |
| Reference model update | Every 400 steps |

### Emergent Behaviors

DeepSeek-R1-Zero spontaneously developed:
- **Self-reflection**: "Wait, let me think about this again..."
- **Verification**: Checking intermediate results
- **Alternative exploration**: "Another approach would be..."
- **Longer responses**: Response length jumped at step 8.2k

### Performance

- Surpassed models trained on human-labeled reasoning trajectories
- Achieved strong performance on math, coding, STEM tasks
- Issues: poor readability, language mixing (English + Chinese)

## DeepSeek-R1 (Full Pipeline)

To address R1-Zero's limitations, DeepSeek-R1 uses a multi-stage pipeline:

1. **RL on reasoning data** → R1-Zero (emergent reasoning)
2. **Rejection sampling** → collect high-quality reasoning trajectories
3. **SFT on reasoning + non-reasoning data** → improve readability and general capabilities
4. **RL on all data** → final alignment

## Why GRPO Works Well for Reasoning

Verifiable rule-based rewards, an informative group-relative comparison signal, freedom from hard-to-estimate value functions on long reasoning trajectories, and skipping SFT so pure RL is free to explore reasoning strategies a human-labeled curriculum might never have shown — together make GRPO well suited to reasoning tasks. The systemic constraints this creates — and where later methods (DAPO, Dr. GRPO, GSPO, SAO) relax them — are analyzed in [[13_reasoning_rl_algorithm_evolution_analysis|D02]] §1, §3.1.

## GRPO vs DPO Family

| Aspect | GRPO | DPO/SimPO/ORPO |
|--------|------|----------------|
| Training type | **Online RL** (samples during training) | **Offline** (fixed dataset) |
| Reward source | Rule-based (correctness) | Human/AI preferences |
| Value function | Not needed | Not needed |
| Data | Verifiable tasks (math, code) | General preference data |
| Best for | Reasoning capability | General alignment |

## Practical Implementation

The computation above maps directly to production code in verl's `core_algos.py`: group-relative normalization is `compute_grpo_outcome_advantage` (`core_algos.py:268`), and the clipped surrogate (with an added dual-clip safeguard for negative advantages) is `compute_policy_loss_vanilla` (`core_algos.py:1279`). See [[15_verl_rl_algorithms_analysis]] §3.2, §4.1 for the source-level walkthrough and the registry keys (`adv_estimator=grpo`, `loss_mode=vanilla`) that select this path.

## Impact

GRPO demonstrated that:
- **Pure RL can induce reasoning** without human-labeled trajectories
- **Value functions are not necessary** for stable RL training
- **Emergent behaviors** can surpass human-designed reasoning patterns
- **Small models can be distilled** from large reasoning models

> **2026-07-27 更新说明**：本文保留 GRPO 的历史原理介绍。关于 response-length/group-std 偏置、DAPO/GSPO/SAO 演进，以及 group rollout 是否“过时”的当前判断，统一阅读 [[13_reasoning_rl_algorithm_evolution_analysis|D02 Reasoning RL 算法演进]]；on/off-policy 与 TIM 见 [[25_on_policy_off_policy_staleness_analysis|D04]]。

## Related Pages

- [[13_reasoning_rl_algorithm_evolution_analysis|D02 演进权威页]] — 公式演进、系统约束与跨算法对照
- [[11_ppo_analysis]] — PPO algorithm that GRPO simplifies
- [[14_deepseek_r1_analysis]] — DeepSeek-R1 training details
- [[17_deepseek_math_analysis]] — DeepSeekMath where GRPO was first proposed
- [[30_preference_optimization_analysis]] — DPO family (offline alternative)
- [[15_verl_rl_algorithms_analysis]] — verl 源码级实现(注册表 + config key→代码锚点)
- [[courses/posttraining_frontier]] — 后训练前沿阅读课程(原 D00–D12 学习域已解散,内容归位至功能树)
- [[01_theory/index]]
