# InstructGPT: Training Language Models to Follow Instructions with Human Feedback — Analysis

**Source**: `raw/03_alignment/InstructGPT_RLHF-2203.02155.pdf`
**Authors**: Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin, Zhang, Agarwal, Slama, Ray, Schulman, Hilton, Kelton, Miller, Simens, Askell, Welinder, Christiano, Leike, Lowe (OpenAI)
**Published**: arXiv:2203.02155 | Mar 2022

---

## Core Contribution

Demonstrates **RLHF (Reinforcement Learning from Human Feedback)** for aligning large language models with user intent. Shows that a 1.3B parameter InstructGPT model preferred over 175B GPT-3 despite 100x fewer parameters. Establishes the three-step alignment pipeline that became the standard for LLM training.

## The Alignment Problem

**Key insight**: Making language models bigger does not inherently make them better at following user intent. Large LMs can generate:
- Untruthful outputs (hallucinations)
- Toxic content
- Unhelpful responses

The pretraining objective (predict next token) is **misaligned** with "follow user instructions helpfully and safely".

**Three alignment criteria** (Askell et al., 2021):
1. **Helpful**: Help the user solve their task
2. **Honest**: Don't fabricate information or mislead
3. **Harmless**: Don't cause physical, psychological, or social harm

## Three-Step RLHF Pipeline

### Step 1: Supervised Fine-Tuning (SFT)

- Collect ~13k labeler demonstrations on diverse prompts (API prompts + labeler-written)
- Fine-tune GPT-3 pretrained model using supervised learning
- 16 epochs, cosine LR decay, residual dropout 0.2
- Model selection based on RM score on validation set (not validation loss — SFT overfits on loss after 1 epoch but continues improving on RM score)

**Prompt categories**:
| Use-case | Percentage |
|----------|-----------|
| Generation | 45.6% |
| Open QA | 12.4% |
| Brainstorming | 11.2% |
| Chat | 8.4% |
| Rewrite | 6.6% |
| Summarization | 4.2% |
| Classification | 3.5% |
| Closed QA | 2.6% |
| Extract | 1.9% |

### Step 2: Reward Model (RM) Training

- Collect ~33k prompts with rankings of K=4-9 model outputs per prompt
- Train RM on SFT model (with final unembedding layer removed)
- Use 6B RM (175B RM was unstable and less suitable as value function)

**RM Loss Function**:

```
loss(theta) = -1/C(K,2) * E[(x, y_w, y_l)~D] [log(sigma(r_theta(x, y_w) - r_theta(x, y_l)))]
```

where:
- r_theta(x, y) = scalar reward for prompt x and completion y
- y_w = preferred completion, y_l = less preferred
- sigma = sigmoid function

**Key optimization**: Train on all C(K,2) comparisons from each prompt as a single batch element — requires only 1 forward pass per completion instead of C(K,2), and prevents overfitting.

### Step 3: PPO Optimization

Fine-tune SFT policy to maximize RM reward using PPO:

**PPO Objective for RLHF**:

```
L_PPO = E[(x, y)~D_pi_old] [min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)] - beta * KL(pi_RL || pi_SFT)
```

where:
- r_t = pi_RL(y|x) / pi_RL_old(y|x) (probability ratio)
- A_t = advantage estimate from value function
- KL penalty is against the **SFT model** (not old policy) to prevent drift from human demonstrations
- beta = KL coefficient (controls how much the policy can deviate from SFT)

**Implementation details**:
- Value function initialized from RM weights
- Pretraining mixture: mix PPO loss with language modeling loss on unlabeled data to preserve capabilities
- 31k prompts from API used for PPO training

## Key Results

### Human Preference

| Model | Win Rate vs SFT 175B |
|-------|---------------------|
| GPT 175B | Baseline |
| GPT 175B (few-shot) | Lower than SFT 175B |
| SFT 175B | Baseline |
| **PPO-ptx 1.3B** | **Preferred over 175B GPT-3** |
| PPO-ptx 6B | Higher |
| PPO-ptx 175B | **85% win rate vs 175B GPT-3** |

### Truthfulness

- On TruthfulQA: InstructGPT generates truthful + informative answers **2x more often** than GPT-3
- On closed-domain tasks: Hallucination rate **21% vs 41%** for GPT-3 (half as often)

### Toxicity

- Small improvements in toxicity over GPT-3
- On RealToxicityPrompts: InstructGPT generates toxic outputs slightly less often

### Public NLP Datasets

- Minimal performance regression on most public NLP datasets
- Some degradation on translation and code tasks (expected, as these weren't in the instruction distribution)

## Human Data Collection

- ~40 contractors on Upwork and ScaleAI
- Screening test to measure sensitivity to demographic preferences and harmful outputs
- Inter-annotator agreement: **72.6%** (training labelers), **77.3%** (held-out labelers)
- During training: prioritize helpfulness; during evaluation: prioritize truthfulness + harmlessness

## Important Design Choices

1. **RM trained on SFT outputs** (not GPT-3 outputs) — higher quality comparisons
2. **KL penalty against SFT model** — prevents policy from drifting too far from human demonstrations
3. **Pretraining mix in PPO** — preserves capabilities on tasks outside the instruction distribution
4. **Model selection by RM score** — not by validation loss (SFT overfits on loss but continues improving on RM score)
5. **6B RM instead of 175B** — more stable, better as value function

## Limitations

- InstructGPT still makes simple mistakes
- Prioritizes helpfulness over safety during training (design choice)
- Performance degrades on tasks outside the instruction distribution (translation, code)
- Human labelers' preferences may not represent all users

## Historical Impact

InstructGPT established the **standard alignment pipeline** used by virtually all modern LLMs:
1. Pretrain on internet data
2. SFT on instruction demonstrations
3. Train reward model on human preferences
4. PPO optimization with KL penalty

This pipeline was later adopted (with variations) by:
- ChatGPT (GPT-3.5, GPT-4)
- Claude (Anthropic)
- LLaMA-2/3 (Meta)
- Qwen (Alibaba)
- DeepSeek

## Related Pages

- [[ppo_analysis]] — PPO algorithm used in Step 3
- [[deepseek_r1_analysis]] — DeepSeek-R1 replaced PPO with GRPO
- [[dpo_analysis]] — DPO is a simpler alternative to RLHF
- [[llm/overview]]
