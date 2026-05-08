# RLHF Foundations & Advanced Methods — Analysis

**Covers**: ReMax, Weak-to-Strong, Scaling Laws for Reward Model Overoptimization, Learning to Summarize, Fine-Tuning from Human Preferences, RigorLLM
**Domain**: LLM Alignment & Preference Optimization

---

## ReMax: Reward Maximization RLHF

**Source**: Li et al., arXiv:2310.10505 | Oct 2023

### Core Idea

Simplifies RLHF by exploiting three properties that PPO ignores:
1. **Fast simulation**: Text generation is fast and deterministic
2. **Deterministic transitions**: Given a prompt, the environment doesn't change
3. **Trajectory-level rewards**: Reward is given for the entire sequence

### Method

Built on REINFORCE with a novel variance reduction technique:
- No value model needed (saves compute)
- Uses baseline from group samples
- Simpler hyperparameter tuning than PPO

### Results

- Matches PPO performance on alignment tasks
- Significantly lower computational cost
- More stable training

---

## Weak-to-Strong Generalization

**Source**: Burns, Izmailov, Kirchner, et al. (OpenAI), arXiv:2312.09390 | Dec 2023

### Core Question

Can **weak model supervision** elicit the full capabilities of a much **stronger model**?

This is an analogy to the superhuman alignment problem: humans (weak) need to supervise superhuman models (strong).

### Method

1. Train a weak model (e.g., GPT-2) on a task
2. Use the weak model's outputs to fine-tune a strong model (e.g., GPT-4)
3. Measure how much of the strong model's capability is recovered

### Key Findings

- **Naive fine-tuning** on weak model outputs significantly degrades strong model performance
- **Weak-to-strong fine-tuning** recovers a substantial portion of strong model capability
- The strong model can **extrapolate beyond** the weak model's performance level
- Works across NLP, chess, and reward modeling tasks

### Implications

- Humans may be able to supervise superhuman models through weak supervision
- Strong models can "fill in the gaps" left by weak supervisors
- Provides a potential pathway for aligning superhuman AI

---

## Scaling Laws for Reward Model Overoptimization

**Source**: Gao, Schulman, Hilton (OpenAI), arXiv:2210.10760 | Oct 2022

### Core Problem

**Goodhart's Law**: "When a measure becomes a target, it ceases to be a good measure."

In RLHF, optimizing against an imperfect reward model leads to **overoptimization** — the policy maximizes proxy reward while degrading true quality.

### Method

- Use a fixed "gold-standard" reward model as ground truth
- Train a proxy reward model on limited preference data
- Optimize policy against proxy reward using RL or best-of-n sampling
- Measure how gold-standard reward changes as proxy reward increases

### Key Findings

1. **Overoptimization follows predictable scaling laws**:
   - RL optimization: different functional form than best-of-n
   - Both methods show smooth scaling with model size and dataset size

2. **Best-of-n is more robust** to overoptimization than RL

3. **Larger reward models** are more sample-efficient but also more prone to overoptimization

4. **The relationship between proxy and true reward** can be predicted from scaling laws

### Implications

- Provides a framework for understanding RLHF failure modes
- Helps determine when to stop optimization to avoid overoptimization
- Informs reward model design and training data requirements

---

## Learning to Summarize from Human Feedback

**Source**: Stiennon, Ouyang, Wu, et al. (OpenAI), arXiv:2009.01325 | Sep 2020

### Core Contribution

First demonstration of **RLHF for summarization** — a precursor to InstructGPT.

### Method

1. Collect human comparisons between summaries
2. Train a reward model to predict human preferences
3. Fine-tune a summarization policy using PPO

### Key Findings

- RLHF produces **higher quality summaries** than supervised learning alone
- Human evaluators strongly prefer RLHF summaries
- ROUGE scores (automatic metric) don't correlate well with human preference
- Demonstrates the value of **human preference optimization** over imitation learning

### Historical Significance

- Established the RLHF pipeline later used in InstructGPT and ChatGPT
- Showed that reward models can capture nuanced human preferences
- Demonstrated PPO's effectiveness for language model fine-tuning

---

## Fine-Tuning Language Models from Human Preferences

**Source**: Ziegler, Stiennon, Wu, et al. (OpenAI), arXiv:1909.08593 | Sep 2019

### Core Contribution

**Earliest RLHF work** — applied reward learning to four natural language tasks:
1. Continuing text with positive sentiment
2. Continuing text with physically descriptive language
3. Summarization (TL;DR)
4. Summarization (CNN/Daily Mail)

### Method

1. Build reward model from human preference data
2. Fine-tune language model using PPO to maximize reward

### Key Findings

- Reward learning works for **stylistic control** (sentiment, descriptiveness)
- More challenging for **content-based tasks** (summarization)
- Established the foundation for all subsequent RLHF work

### Historical Significance

- First paper to combine reward learning with language model fine-tuning
- Introduced the RLHF paradigm that became standard
- Paved the way for InstructGPT, ChatGPT, and modern alignment

---

## RigorLLM: Resilient Guardrails

**Source**: Yuan, Xiong, Zeng, et al., arXiv:2403.13031 | Mar 2024

### Core Problem

Current LLM safety guardrails are **not resilient under adversarial attacks**.

### Method

Multi-faceted approach:
1. **Energy-based training data generation** via Langevin dynamics
2. **Safe suffix optimization** via minimax optimization
3. **Fusion-based model** combining robust KNN with LLMs

### Key Contributions

- Generates adversarial training data systematically
- Optimizes safety prompts for maximum robustness
- Combines multiple defense mechanisms for resilience

### Significance

- Addresses the growing problem of **jailbreak attacks** on LLMs
- Provides a framework for building robust safety systems
- Important for deployment of aligned models in production

---

## Timeline of RLHF Development

| Year | Paper | Contribution |
|------|-------|-------------|
| 2019 | Fine-Tuning from Human Preferences | First RLHF demonstration |
| 2020 | Learning to Summarize | RLHF for summarization |
| 2022 | Scaling Laws for RM Overoptimization | Understanding Goodhart's Law in RLHF |
| 2022 | InstructGPT | Full RLHF pipeline for instruction following |
| 2023 | DPO | Direct preference optimization (no RL) |
| 2023 | ReMax | Simplified RLHF using REINFORCE |
| 2023 | Weak-to-Strong | Supervising strong models with weak labels |
| 2024 | RigorLLM | Resilient safety guardrails |

---

## Related Pages

- [[instructgpt_rlhf_analysis]] — Full RLHF pipeline
- [[ppo_analysis]] — PPO algorithm used in RLHF
- [[dpo_analysis]] — DPO (alternative to RLHF)
- [[preference_optimization_analysis]] — DPO family comparison
- [[01_theory/index]]
