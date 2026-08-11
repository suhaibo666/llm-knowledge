# VAPO: Value Augmented Proximal Policy Optimization — Analysis

**Source**: `raw/03_alignment/VAPO_Value_Augmented_Proximal_Policy_Optimization-2504.05118.pdf`
**Authors**: ByteDance Seed
**Published**: arXiv:2504.05118 | Apr 2025

---

## Core Contribution

Proposes **VAPO**, a value-model-based RL framework for reasoning that achieves **60.4 on AIME 2024** with Qwen2.5-32B — outperforming DAPO (50) and DeepSeek-R1-Zero-Qwen-32B (47) by more than 10 points. Reaches SOTA in only **5,000 steps** with zero training crashes across multiple independent runs（摘要 + §5）。

## Why Value-Model-Based Methods Still Matter

Despite the success of value-model-free methods (GRPO, DAPO), value-model-based approaches have a **higher performance ceiling**:

1. **Precise credit assignment**: Value models trace the impact of each action on subsequent returns — critical for complex reasoning where single-step errors cause catastrophic failures
2. **Lower variance**: Value estimates have lower variance than Monte Carlo group averages
3. **Better generalization**: Well-trained value models generalize across samples, improving sample efficiency

## Three Challenges in Value-Model-Based RL for Long-CoT

### 1. Value Model Bias

Long trajectories make bootstrapped value learning unstable. The value model accumulates errors over long sequences.

**Solution（§4.1，原文机制）**：
- **Value-Pretraining** —— 正式策略训练开始前，先用 reward model 对 value network 做 **50 步 warmup**，返回值以 `λ = 1.0` 估计，消除 value 初始化偏差。
- **Decoupled-GAE**（承自 VC-PPO）—— value network 用 `λ = 1.0` 的 return 学习，policy network 用另一个独立的 λ 得到的 advantage 学习，两者解耦。

### 2. Heterogeneous Sequence Lengths

Short and long responses have very different value distributions. A single value model struggles to handle both.

**Solution（§4.2，原文机制）**：
- **Length-Adaptive GAE** —— 令 λ 随输出长度 `l` 自适应：

  $$\lambda_{\text{policy}} = 1 - \frac{1}{\alpha l},\qquad \alpha = 0.05$$

  动机（原文）：VC-PPO 固定 `λ_policy = 0.95`，当 `l > 100` 时 reward 对应 TD-error 的系数 `0.95^100 ≈ 0.006`，实际为零，GAE 被可能有偏的 bootstrapping TD-error 主导。自适应 λ 使 TD-error 在长短序列上分布更均匀。
- **Token-level policy gradient loss**（承自 DAPO）—— 用 token 级取代 sample 级聚合，避免长序列在梯度中被稀释。

### 3. Sparse Reward Signals

In reasoning tasks, reward is only given at the end (correct/incorrect). Most tokens receive no direct reward signal.

**Solution（§4.3，原文机制）**：针对 verifier-based 任务的稀疏奖励，采用三件套 —— **Clip-Higher**（承自 DAPO，解耦上下 clip 界以缓解熵坍塌）、**Positive Example LM Loss**（提高正样本利用效率）、**Group-Sampling**（承自 GRPO）。

## VAPO Framework：论文自列的七项修改

§5.1 逐条列出了 VAPO 相对基线的**七项**修改，消融实验即逐项移除这七项（`raw/.../VAPO_...-2504.05118.pdf` §5.1）：

| # | 修改 | 关键参数 | 来源 |
|---|------|---------|------|
| 1 | Value network warmup（Value-Pretraining） | 基于 RM 预热 **50 步**，再开始策略训练 | VC-PPO |
| 2 | Decoupled GAE | value 用 `λ = 1.0` 的 return；policy 用独立 λ | VC-PPO |
| 3 | Length-Adaptive GAE | `λ_policy = 1 − 1/(αl)`，`α = 0.05` | **VAPO 原创** |
| 4 | Clip-Higher | `ε_high = 0.28`、`ε_low = 0.2` | DAPO |
| 5 | Token-level policy gradient loss | — | DAPO |
| 6 | Positive-example LM loss | 权重 **0.1** | SIL（self-imitation learning） |
| 7 | Group-Sampling | 每次采样 **512 prompts × 16 次**，mini-batch **512** | GRPO |

> [!deprecated] 2026-08-10 更正（回原文补齐）
> 本节此前是四条名词短语（"Improved value model training / Length-aware value normalization / Dense reward shaping / Stable clipping"），既非论文用语，也无参数与出处，等同占位。现按 §5.1 的七项原文清单重写；上方三个 Challenge 的 "Solution" 亦由「VAPO uses careful … techniques」这类空句替换为 §4.1–4.3 的具体机制与参数。

## Results

| Method | AIME 2024 | Training Steps | Crashes |
|--------|-----------|---------------|---------|
| DeepSeek-R1-Zero-Qwen-32B（GRPO） | 47 | 论文未给绝对值 | 论文未评述 |
| DAPO | 50 | R1-Zero 的 **50%** update steps | 论文未评述 |
| **VAPO** | **60.4** | 达到 60.4 用 **5,000 步**；追平 DAPO 的 50 分只用 DAPO 的 **60%** 步数 | **No**（多次独立运行零崩溃） |

论文原文（§5，`raw/.../VAPO_...-2504.05118.pdf`）：

> "On Qwen-32b, DeepSeek R1 using GRPO achieves 47 points on AIME24, while DAPO reaches 50 points with 50% of the update steps. In Figure 1, our proposed VAPO matches this performance using only **60% of DAPO's steps** and achieves a new SOTA score of **60.4 within just 5,000 steps**."

VAPO achieves:
- **10+ points improvement** over previous SOTA（摘要：比 DeepSeek-R1-Zero-Qwen-32B 与 DAPO 高 10 分以上）
- 追平 DAPO 只需其 **60% 的步数**（即少 40%）；60.4 分在 **5,000 步**内达成
- **Zero crashes** across multiple independent runs

> [!deprecated] 2026-08-10 更正（回原文核对）
> 三处更正：① 此前写「**50% fewer training steps** than DAPO」，论文实为「using only **60% of DAPO's steps**」（少 40%，且指的是**追平 DAPO 的 50 分**，不是达到 60.4）；② 表中 DAPO `~8,000` 与 R1-Zero `~10,400` 是页面自行换算的绝对步数，VAPO 论文只给相对比例（DAPO = R1-Zero 的 50%），且 `~10,400` 出自 [[20_grpo_analysis]] 中 **DeepSeek-R1-Zero（V3-base）** 的训练步数，与此处的 **DeepSeek-R1-Zero-Qwen-32B** 不是同一次运行，不可混用；③ `Crashes: Yes` 是对他方系统的**无源断言**——论文只声明 VAPO 自身多次独立运行零崩溃，未评述其它方法是否崩溃。

## Significance

VAPO demonstrates that **value-model-based methods are not dead** — with proper engineering, they can outperform value-model-free methods by a significant margin. The key is addressing the three challenges of bias, length heterogeneity, and reward sparsity.

## Related Pages

- [[21_dapo_analysis]] — DAPO (value-model-free baseline)
- [[11_ppo_analysis]] — PPO foundation
- [[20_grpo_analysis]] — GRPO (value-model-free)
- [[01_theory/index]]
