# GRPO: Group Relative Policy Optimization — Analysis

**Source**: DeepSeek R1 paper (arXiv:2501.12948) + DeepSeekMath paper (arXiv:2402.03300)
**Authors**: DeepSeek-AI
**Published**: Jan 2025 (R1), Feb 2024 (Math)

---

## Core Contribution

**GRPO (Group Relative Policy Optimization)** simplifies PPO by eliminating the value function and computing advantages from a group of outputs sampled from the same prompt. Used to train DeepSeek-R1-Zero through **pure RL without SFT**, enabling emergent reasoning capabilities.

## GRPO Algorithm

### Key Innovation: Group-Relative Advantages

Instead of training a separate value function (like PPO), GRPO computes advantages from a group of outputs:

```
For each question q:
    Sample G outputs {o_1, o_2, ..., o_G} from old policy pi_theta_old
    Compute rewards {r_1, r_2, ..., r_G} for each output
    Compute advantage: A_i = (r_i - mean({r})) / std({r})
```

**This eliminates the need for a value function** — the group mean serves as the baseline.

### GRPO Objective

```
J_GRPO(theta) = E[q~P(Q), {o_i}~pi_theta_old] [
    (1/G) * sum_i min(
        (pi_theta(o_i|q) / pi_theta_old(o_i|q)) * A_i,
        clip(pi_theta(o_i|q) / pi_theta_old(o_i|q), 1-eps, 1+eps) * A_i
    )
    - beta * D_KL(pi_theta || pi_ref)
]
```

where:

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

1. **Verifiable rewards**: Math/code problems have clear correct/incorrect answers
2. **Group comparison**: Relative ranking within a group is more informative than absolute rewards
3. **No value function**: Reasoning trajectories are long and complex; value function estimation is difficult
4. **Exploration**: Pure RL without SFT allows discovery of non-human reasoning strategies

## GRPO vs DPO Family

| Aspect | GRPO | DPO/SimPO/ORPO |
|--------|------|----------------|
| Training type | **Online RL** (samples during training) | **Offline** (fixed dataset) |
| Reward source | Rule-based (correctness) | Human/AI preferences |
| Value function | Not needed | Not needed |
| Data | Verifiable tasks (math, code) | General preference data |
| Best for | Reasoning capability | General alignment |

## Practical Implementation

```python
# Pseudocode for GRPO
for q in questions:
    # Sample group of outputs
    outputs = [model.generate(q) for _ in range(G)]
    
    # Compute rewards (e.g., correctness)
    rewards = [compute_reward(o, ground_truth) for o in outputs]
    
    # Compute group-relative advantages
    mean_r = mean(rewards)
    std_r = std(rewards)
    advantages = [(r - mean_r) / std_r for r in rewards]
    
    # Compute GRPO loss
    for o_i, adv_i in zip(outputs, advantages):
        ratio = pi_theta(o_i|q) / pi_theta_old(o_i|q)
        clipped_ratio = clip(ratio, 1-eps, 1+eps)
        loss += -min(ratio * adv_i, clipped_ratio * adv_i)
    
    # Add KL penalty
    loss += beta * KL(pi_theta || pi_ref)
    
    # Update policy
    loss.backward()
    optimizer.step()
```

## Impact

GRPO demonstrated that:
- **Pure RL can induce reasoning** without human-labeled trajectories
- **Value functions are not necessary** for stable RL training
- **Emergent behaviors** can surpass human-designed reasoning patterns
- **Small models can be distilled** from large reasoning models

## Related Pages

- [[ppo_analysis]] — PPO algorithm that GRPO simplifies
- [[deepseek_r1_analysis]] — DeepSeek-R1 training details
- [[deepseek_math_analysis]] — DeepSeekMath where GRPO was first proposed
- [[preference_optimization_analysis]] — DPO family (offline alternative)
- [[01_theory/index]]
