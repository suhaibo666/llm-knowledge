# GSPO: Group Sequence Policy Optimization — Analysis

**Source**: `raw/03_alignment/GSPO_Group_Sequence_Policy_Optimization-2507.18071.pdf`
**Authors**: Qwen Team, Alibaba Inc.
**Published**: arXiv:2507.18071 | Jul 2025

---

## Core Contribution

Proposes **GSPO**, a sequence-level RL algorithm that fixes the fundamental instability of GRPO by using **sequence-level importance ratios** instead of token-level ratios. GSPO stabilizes MoE RL training, achieves superior performance, and contributed to Qwen3 model improvements.

## Problem: Why GRPO is Fundamentally Ill-Posed

GRPO's token-level importance weight `w_i,t = pi_theta(y_i,t|...) / pi_theta_old(y_i,t|...)` is based on a **single sample** from each next-token distribution, so it cannot perform the distribution correction importance sampling normally requires (N >> 1 samples). The resulting high-variance noise accumulates over long sequences, is exacerbated by clipping, and can produce irreversible collapse on large models — the core principle violated is that the unit of the optimization objective should match the unit that receives reward (the whole sequence, not each token). This diagnosis and GSPO's fix are unified with GRPO/DAPO/Dr. GRPO/SAO's own statistical-unit issues in [[13_reasoning_rl_algorithm_evolution_analysis|D02]] §3.4.

## GSPO Algorithm

### Sequence-Level Importance Ratio, Objective and Key Difference from GRPO

GSPO replaces the token-level ratio with a length-normalized **sequence-level** ratio `s_i(theta) = (pi_theta(y_i|x)/pi_theta_old(y_i|x))^(1/|y_i|)`, applied inside the same clipped-surrogate template as GRPO (all tokens in a response now share both the advantage and the clip decision). D02 §3.4 给出 `s_i(theta)` 的公式，§4 的跨算法表覆盖 importance ratio 与 clipping granularity 两轴；这一记号即 D02 §2 的通用 clipped surrogate 把 token 比值换成 `s_i`。

**长度归一化为什么关键（§4.1，本页自留，勿再外指）**：

> "we adopt length normalization in `s_i(θ)` to reduce the variance and to control `s_i(θ)` within a **unified numerical range**. Otherwise, the likelihood changes of a few tokens can result in **dramatic fluctuations** of the sequence-level importance ratio, and the importance ratios of responses with **different lengths will require varying clipping ranges**."

即：不做长度归一化，`s_i` 是 `|y_i|` 个 token 概率比的连乘，少数 token 的似然抖动会被指数放大；且不同长度的响应会各自需要不同的 clip 区间，无法用单一超参覆盖。

> [!warning] 2026-08-10 更正：转指曾指向不存在的内容
> 本段此前写「why length normalization is critical, and the full GRPO-vs-GSPO comparison (importance ratio / clipping granularity / **gradient weighting** / **stability**) are unified in D02 §3.4, §4」。核对后：D02 §4 的跨算法表只有 6 列（优化单位 / advantage / ratio-clip / 采样结构 / 系统不变量等），**没有 gradient weighting 列，也没有 stability 列**；D02 §3.4 只给公式，不含长度归一化的必要性推导。本页据此把长度归一化的论证**收回本地**（上方引文），并把转指范围缩到 D02 确实承载的两轴。本页下文自留的稳定性排序同理——D02 明确不做定性稳定性排名。

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

| Algorithm | eps_low | eps_high | 出处与性质 |
|-----------|---------|----------|-----------|
| GRPO（本文实验基线） | 0.2 | 0.27 | GSPO 论文 §5.1 自设并「carefully tuned」的对照基线，**非 GRPO 的默认或固有配置** |
| **GSPO** | **3e-4** | **4e-4** | GSPO 论文 §5.1，作用于 Eq. (5) 的序列级比值 |

This is because GSPO's sequence-level ratio has fundamentally different numerical properties than GRPO's token-level ratio. 论文原话：clipping ranges "typically differ in **order of magnitude** due to the distinct definitions of importance ratios"（§4.1 末）。

> [!note] 三个 0.2/0.2x 别混（2026-08-10 回原文核对）
> - **GRPO 本身是对称 clip**：DeepSeekMath arXiv:2402.03300v3 §4.1.1 式 (3) 写的是 `clip(·, 1-eps, 1+eps)`，只有一个 `eps`，没有非对称上下界。见 [[20_grpo_analysis]]。
> - **上表的 0.2 / 0.27** 出自 GSPO 论文 §5.1：「We compare against GRPO as the baseline and set the left and right clipping ranges in Equation (2) to 0.2 and 0.27, respectively, which we have carefully tuned to ensure a fair comparison.」——这是 GSPO 作者为公平对比而调的**实验设定**。
> - **DAPO 的 Clip-Higher 是 0.2 / 0.28**（DAPO arXiv:2503.14476 §5：`ε_low = 0.2`、`ε_high = 0.28`），与上表的 0.27 是**两篇不同论文的不同数字**，不是同一个值的笔误。见 [[21_dapo_analysis]]。

## Empirical Results

实验设定（§5.1）：cold-start 模型自 **Qwen3-30B-A3B-Base**（MoE）微调；每 batch rollout 数据切成 **4 个 mini-batch** 做梯度更新；评测 AIME'24（32 次采样的平均 Pass@1）、LiveCodeBench（202410–202502，8 次采样平均 Pass@1）、CodeForces（Elo Rating）。

- **训练效率显著高于 GRPO**（Figure 1 的训练 reward 曲线与三项基准曲线）
- **Stable MoE RL training** —— GRPO 需要 Routing Replay 才能正常收敛，GSPO 免除该依赖（§5.3）
- Contributed to **Qwen3 model improvements**

**§5.2 唯一的硬量化结论 —— clipping fraction 相差两个数量级**：

> "we observe a difference of **two orders of magnitude** in the fractions of clipped tokens between GSPO and GRPO (while adjusting the clipping ranges does not alter the disparity in magnitude). However, despite clipping significantly more tokens and consequently using fewer for training (or gradient estimation), **GSPO still achieves higher training efficiency than GRPO**."

论文把这个反直觉现象（裁掉的 token 多得多，训练效率反而更高）当作 GRPO token 级梯度估计「inherently noisy and inefficient for sample exploitation」的旁证。

> [!note] 关于本节此前「零数字」的说明（2026-08-10）
> 本节先前只有四条定性 bullet。回原文核对后确认：**GSPO 论文本身不提供基准分数表**，AIME'24 / LiveCodeBench / CodeForces 的结果全部以 Figure 1 的曲线形式给出，正文未复述任何终点数值；唯一可引用的量化结论是 §5.2 的 clipping fraction 两个数量级差（Figure 2，纵轴上限 0.15）。因此本节补的是**实验设定 + §5.2 结论**，而非从图上目测读数。

## Why GSPO Matters

1. **Fixes fundamental flaw**: Token-level importance sampling in GRPO is theoretically unsound
2. **Stabilizes MoE training**: MoE models are particularly sensitive to gradient noise
3. **Simplifies infrastructure**: No need for complex stabilization strategies
4. **Scales better**: Sequence-level optimization is more principled for long responses

> **2026-07-27 更新说明**：本文的“稳定性”表述代表论文固定实验条件，不是所有模型/框架的无条件结论。sequence ratio/clip 的公式—batch schema 对照见 [[13_reasoning_rl_algorithm_evolution_analysis|D02]]，与 staleness/TIM 的组合语义见 [[25_on_policy_off_policy_staleness_analysis|D04]]。

## Relationship to Other Methods

GSPO's position relative to PPO/GRPO/DAPO/Dr. GRPO/SAO across importance-ratio granularity, clipping level and the systemic invariant each preserves is tabulated in [[13_reasoning_rl_algorithm_evolution_analysis|D02]] §4 — D02's table does not rank qualitative training stability. This paper's own stability ordering across that same method set is: PPO moderate, GRPO poor at scale, DAPO good, **GSPO excellent**.

## Related Pages

- [[13_reasoning_rl_algorithm_evolution_analysis|D02 演进权威页]] — 公式演进、系统约束与跨算法对照
- [[20_grpo_analysis]] — GRPO algorithm that GSPO fixes
- [[21_dapo_analysis]] — DAPO improvements to GRPO
- [[11_ppo_analysis]] — PPO foundation
- [[15_verl_rl_algorithms_analysis]] — verl 源码级实现(注册表 + config key→代码锚点)
- [[courses/posttraining_frontier]] — 后训练前沿阅读课程(原 D00–D12 学习域已解散,内容归位至功能树)
- [[01_theory/index]]
