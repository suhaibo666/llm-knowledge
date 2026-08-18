# slime RL Loss、Reducer 与并行调度实现分析

> **源码基线**：slime `main@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`
> **核验日期**：2026-08-18 · **系列**：[[slime/index]]
> **结论先行**：slime 算法层的核心不是“支持几个 loss 名字”，而是把算法的统计单位从物理 token/sample/mbs/rank 中剥离出来。reward/advantage 先在完整 rollout 上建立，`rollout_id + rollout_mask_sums` 定义逻辑目标，DP scheduler 保证 PP/VPP liveness，CP reducer 再让分片前后的 loss/metrics 等价。

## 1. 算法管线

```mermaid
flowchart LR
    RW["group-normalized reward"] --> KL["reference KL shaping"]
    KL --> ADV["GRPO/GSPO/CISPO/PPO/REINFORCE++ advantage"]
    ADV --> LP["current/old/rollout logprob"]
    LP --> OBJ["PPO or CISPO objective"]
    OBJ --> CORR["optional OPSM / TIS / custom correction"]
    CORR --> RED["per-rollout or per-token reducer"]
    RED --> SCALE["Megatron DP×CP×mb scaling"]
```

## 2. Reward 与 advantage 的统计时点

GRPO/GSPO/CISPO/reinforce++ baseline 的 group reward normalization 在 RolloutManager flatten 后、DP split 前执行，因此能看见完整 prompt group。[`rollout.py:722-747`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L722-L747)

训练侧 `compute_advantages_and_returns` 支持：

- GRPO/GSPO/CISPO：normalized scalar reward扩展到 response tokens；
- PPO：把 terminal reward 加到 KL-shaped token rewards 后做 GAE；
- REINFORCE++ 与 baseline：分别构造 discounted/token return 或 group baseline advantage；
- custom function：在 KL 计算后写回 advantages/returns；
- OPD：正交地把 teacher KL 施加到 advantages。

实现见 [`loss.py:704-817`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L704-L817)。

### 2.1 PPO 中 $t$ 与 $t+1$ 数的是相邻 token 决策

对一条生成轨迹

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots,s_T),
$$

$s_t$ 是“prompt 加上前 $t$ 个已生成 response tokens”的 prefix 状态，$a_t$ 是接下来选择的第 $t$ 个 response token。于是 $s_{t+1}$ 就是把 $a_t$ 追加到 prefix 后的状态。这里的 $t/t+1$ 不是 optimizer step、rollout cycle 或 sample 编号。

slime 的 causal 对齐正是这个定义：对 response token $a_t$，取前一 token 位置的 policy/value output；`get_values` 最终为每条 sample 返回长度为 `response_length` 的 value 向量。[`loss.py:97-167`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L97-L167) [`loss.py:607-650`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L607-L650)

### 2.2 TD residual 只是 GAE 的一块，不是 GAE 本身

一步 TD residual 是：

$$
\delta_t
=r_t+\gamma V_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t).
$$

它衡量“实际收到的一步 reward 加下一状态预期”相对当前 Critic 预期有多意外。GAE 再把当前及未来 residual 反向累积：

$$
\begin{aligned}
A_t^{\mathrm{GAE}}
&=\delta_t+\gamma\lambda A_{t+1}^{\mathrm{GAE}} \\
&=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}.
\end{aligned}
$$

所以“$\delta_t$ 是简化后的 GAE”不够准确：它是 GAE 递推中的单步误差信号；只有 $\lambda=0$ 时，GAE 才退化成当前这一个 $\delta_t$。$\lambda$ 越大，越多远期 reward 信号会传回较早 token，方差通常更大、对 Critic bootstrap 偏差的依赖更小。源码从 response 尾部向前执行这条递推，并令 terminal 的 $V(s_T)=0$。[`ppo_utils.py:586-607`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/ppo_utils.py#L586-L607)

slime 的 PPO token reward 先取 $-\texttt{kl\_coef}\cdot\mathrm{KL}_t$，再只把 rollout 产生的 sample-level scalar reward 加到最后一个 response token；因此即使任务只在答案结束时打一次分，GAE 也能把终局结果向前传播给早期 token。[`loss.py:769-781`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L769-L781)

### 2.3 为什么 Critic target 是 $A_t+V_{\mathrm{old}}(s_t)$

源码构造：

$$
\widehat R_t=A_t^{\mathrm{GAE}}+V_{\mathrm{old}}(s_t).
$$

依据不是人为把两个数凑在一起，而是 advantage 的定义本来就是“这个 action 的回报比状态基线高多少”：$A(s,a)=Q(s,a)-V(s)$；加回同一份 old baseline，才能从相对量恢复 Critic 要拟合的绝对 return target。

两端特例能看得更清楚：

- $\lambda=0$ 时，$\widehat R_t=\delta_t+V_{\mathrm{old}}(s_t)=r_t+\gamma V_{\mathrm{old}}(s_{t+1})$，就是一步 TD target；
- $\lambda=1$ 且轨迹终止时，递推中的相邻 values 望远镜消去，$\widehat R_t$ 变成从 $t$ 开始的完整 discounted reward sum。

中间的 $\lambda$ 则给出 multi-step $\lambda$-return。实现把 advantage/return 计算放在 `torch.no_grad()` 中，且 Critic 先 forward old values、再训练；这是为了让 target 在本批更新中固定。若把公式右边换成正在反向更新的 current value，target 会跟预测一起移动，甚至部分抵消误差。[`ppo_utils.py:478-506`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/ppo_utils.py#L478-L506) [`actor.py:396-421`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/actor.py#L396-L421)

单条轨迹的 $\widehat R_t$ 当然很噪；Critic 不是被要求记住这一次结果。跨大量从相似 prefix 出发的轨迹做 squared-error regression 时，MSE 的总体最优解是条件均值 $V^\pi(s_t)=\mathbb E[\widehat R_t\mid s_t]$，即“当前策略从这个 prefix 继续生成的平均未来回报”。

### 2.4 这些量到底是 sample-level 还是 token-level

| 阶段 | 典型量 | 粒度 |
|---|---|---|
| rollout/RM | `Sample.reward` | completion/sample-level scalar；prompt group 内有 $K$ 个 samples |
| PPO reward shaping | $r_t$ | token-level；终局 scalar reward 放在最后 token，KL 可逐 token 注入 |
| Critic/GAE | $V_t$、$\delta_t$、$A_t$、$\widehat R_t$ | token-level，单条 shape 为 `response_length` |
| Actor/Critic raw loss | $\ell_t$ | token-level，prompt/padding/tool 等位置由 mask 排除 |
| 默认 reducer | $L_g$ | logical-rollout-level token-weighted mean；fanout siblings 共享完整 denominator |
| Megatron step | $L$ | 聚合 micro-batches、DP/CP 后用于 backward 的 scalar |

RolloutManager 先保留 sample-level reward 与 response mask，再在完整 step 上为每个 logical rollout 计算 `rollout_mask_sums`；loss 侧把 token losses 乘 mask、除以该 rollout 的完整 denominator。开启 `calculate_per_token_loss` 时才改为全局 token-sum/总 token 数口径。[`rollout.py:749-814`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L749-L814) [`cp_utils.py:47-124`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/cp_utils.py#L47-L124) 因此前面手算的 TD、GAE、Actor/Critic 单项都是 **token-level**；最终交给 optimizer 的 loss 已经是按配置约化后的 scalar，不能简单称为“sample-level loss”。

## 3. PPO、GSPO 与 CISPO 目标

### 3.1 Actor loss：把“相对预期好坏”变成 token 概率更新

对 rollout 中实际采到的 token $a_t$，概率比为：

$$
\rho_t(\theta)
=\exp\!\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t)\right).
$$

slime 以最小化形式实现 PPO-Clip：

$$
\ell_{\mathrm{actor},t}
=\max\!\left(
-\rho_t A_t,
-\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon_{\mathrm{high}})A_t
\right).
$$

- $A_t>0$ 表示这个 token 比 Critic 对该 prefix 的平均预期更好，梯度提高它的概率；但 $\rho_t$ 超过上界后不再因继续增大而获得更好 surrogate；
- $A_t<0$ 表示它比预期差，梯度降低它的概率；下界阻止一次 noisy batch 把概率压得过猛；
- reward/verifier 通常不可微，policy gradient 的 log-prob trick 让 reward 无需反向穿过采样或 verifier，只用 advantage 给已选 token 的 logprob 加权。

源码先算 `ppo_kl = old_log_prob - new_log_prob`，再令 ratio 为 `exp(-ppo_kl)`；因为代码在最小化 loss，所以取 clipped/unclipped 两者的最大值，等价于最大化形式取较悲观的最小 surrogate。可选 dual clip 只作用于负 advantage。[`loss.py:1020-1044`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L1020-L1044) [`ppo_utils.py:124-148`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/ppo_utils.py#L124-L148)

### 3.2 Critic loss：拟合绝对 return，但限制相对 old value 的单批漂移

Critic 当前 forward 得到 $V_\phi(s_t)$，本轮 forward 前保存的 prediction 是 $V_{\mathrm{old}}(s_t)$。先构造 clipped prediction：

$$
V_{\mathrm{clip},t}
=V_{\mathrm{old}}(s_t)
+\operatorname{clip}\!\left(
V_\phi(s_t)-V_{\mathrm{old}}(s_t),
-\epsilon_v,
\epsilon_v
\right),
$$

再取：

$$
\ell_{\mathrm{critic},t}
=\max\!\left(
\left(V_\phi(s_t)-\widehat R_t\right)^2,
\left(V_{\mathrm{clip},t}-\widehat R_t\right)^2
\right).
$$

普通 MSE 回答“Critic 应往哪里拟合”；old-value clip 回答“同一批 noisy returns 最多允许 Critic 一次获益多少”。若新 value 跨出 clip 区间，即使它碰巧更接近本批 target，clipped 分支仍可能给出更大的误差，避免 value baseline 一批跳得太远。源码没有额外乘 $\frac{1}{2}$。[`loss.py:1176-1230`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L1176-L1230)

Actor 和 Critic 的数值通过 $A_t/\widehat R_t$ 耦合，但梯度不相互穿透：Critic loss只更新 value model，Actor loss只更新 policy model。Critic 降低“同一个最终 reward 应怎样分摊到不同 prefix”的方差；Actor 才真正改变生成分布。

### 3.3 用一条实际 $\tau$ 手算两种 loss

考虑 prompt `1+1=`，Actor 生成三个 response tokens：`2`、`。`、`<eos>`。为突出主链，设 $\gamma=1$、$\lambda=1$、`kl_coef=0`、所有 loss mask 为 1；终局 verifier reward 为 1，因此 token rewards 是 $[0,0,1]$。这也恰好是 slime 当前 PPO 的默认 $\gamma/\lambda$。[`arguments.py:899-907`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/arguments.py#L899-L907) [`arguments.py:1001-1004`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/arguments.py#L1001-L1004)

Critic 训练前预测：

| $t$ | 状态与动作 | $r_t$ | $V_{\mathrm{old}}(s_t)$ | next value | $\delta_t$ | $A_t$ | $\widehat R_t$ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `1+1=` → `2` | 0 | 0.40 | 0.70 | 0.30 | 0.60 | 1.00 |
| 1 | `1+1=2` → `。` | 0 | 0.70 | 0.90 | 0.20 | 0.30 | 1.00 |
| 2 | `1+1=2。` → `<eos>` | 1 | 0.90 | 0 | 0.10 | 0.10 | 1.00 |

反向递推是：$A_2=0.10$，$A_1=0.20+A_2=0.30$，$A_0=0.30+A_1=0.60$；再加回 old values，三个 returns 都是 1。其业务含义是：在这条成功轨迹上，早期选择 `2` 比 Critic 原先 0.40 的成功预期高很多；到 `<eos>` 前 Critic 已预期 0.90，终局成功只带来 0.10 的惊喜。

假设 Critic 一次 forward/backward 中的新预测为 $[0.70,0.85,0.95]$，`value_clip=0.2`：

| $t$ | new value | clipped value | unclipped error² | clipped error² | 取 max |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.70 | 0.60 | 0.09 | 0.16 | 0.16 |
| 1 | 0.85 | 0.85 | 0.0225 | 0.0225 | 0.0225 |
| 2 | 0.95 | 0.95 | 0.0025 | 0.0025 | 0.0025 |

在这一个三 token logical rollout 上，默认 token-weighted mean 的 Critic loss 是 $\frac{0.16+0.0225+0.0025}{3}\approx0.0617$，value clip fraction 是 $\frac{1}{3}$。注意“一条成功轨迹的 return 都是 1”不表示 Critic 最终应对所有相同 prefix 输出 1：若从某 prefix 采样 10 次只成功 4 次，跨样本 MSE 会把 value 推向约 0.4。

再看 Actor。设 old policy 对三个已选 token 的概率为 $[0.50,0.80,0.90]$，当前 policy 为 $[0.65,0.72,0.945]$；概率 ratios 为 $[1.30,0.90,1.05]$。取 $\epsilon=\epsilon_{\mathrm{high}}=0.2$：

| $t$ | $A_t$ | ratio | clipped ratio | PPO token loss |
|---:|---:|---:|---:|---:|
| 0 | 0.60 | 1.30 | 1.20 | $\max(-0.78,-0.72)=-0.72$ |
| 1 | 0.30 | 0.90 | 0.90 | $-0.27$ |
| 2 | 0.10 | 1.05 | 1.05 | $-0.105$ |

默认 mean 为 $(-0.72-0.27-0.105)/3=-0.365$。第一个 token 虽然最值得鼓励，但 probability ratio 已到 1.30，PPO 只按 1.20 计算 surrogate，防止一批数据把 `2` 的概率继续猛推；`<eos>` 虽来自成功答案，但 advantage 只有 0.10，因为 Critic 早已预期大概率成功。若这条轨迹最终 reward 为 0，在同样 $\gamma=\lambda=1$ 下 returns 会变成 0，advantages 则对各 prefix 为负，Actor 会降低这些已选 token 的概率。

这说明两个 loss 的分工不是“Actor 学正确答案、Critic 再判一次对错”：Critic 把稀疏终局 reward 变成每个 prefix 的期望与 surprise，Actor 再依据 surprise 调整每个已选 token 的概率；两边各自的 clip 都以 old snapshot 为锚，抑制单批更新过猛。

### 3.4 GSPO 与 CISPO 的差异

GSPO 先在完整 sequence 上计算 masked mean log-ratio，再把同一 sequence-level KL 展开回本地 tokens；CP 情况必须先 all-gather full logprobs。[`ppo_utils.py:95-121`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/ppo_utils.py#L95-L121) [`loss.py:991-1033`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L991-L1033)

CISPO 截断 stop-gradient ratio，但梯度流经 logprob，所以被 clip token 仍产生梯度；canonical single-sided setting 需要关闭实际 lower bound。[`ppo_utils.py:151-171`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/ppo_utils.py#L151-L171)

## 4. KL 有两个不同位置

1. `kl_coef`：在 advantage 前做 reward shaping；PPO 把 KL 当 token reward，GRPO 类 estimator 也通过 returns 消费。
2. `use_kl_loss + kl_loss_coef`：在 policy loss 后直接加显式 KL regularizer。

参数校验禁止两者同时非零，避免同一 reference KL 被重复施加。[`arguments.py:1841-1842`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/arguments.py#L1841-L1842) KL estimator 支持 k1/k2/k3/low-var，后者做 clamp；unbiased KL 可再乘 importance ratio。[`ppo_utils.py:11-51`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/ppo_utils.py#L11-L51)

## 5. Off-policy correction 与 mask

`vanilla_tis_function` 用 training old logprob 与 rollout behavior logprob 计算 ratio，并 clamp 到上下界后乘 policy loss；ICEPOP 则把区间外 ratio 置零。[`loss.py:884-931`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L884-L931)

OPSM 按 sequence-level KL 与 advantage sign 决定整序列 mask：负 advantage 且 divergence 超阈值时屏蔽。[`ppo_utils.py:54-92`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/ppo_utils.py#L54-L92)

TIS/custom rejection 修改 response masks 后，源码重建 pg-loss reducer，但 denominator 仍使用原先 step-global `rollout_mask_sums`；mismatch metrics 则保留 pre-rejection reducer，防止被拒 tokens 同时从指标分子和分母消失而把 truncate fraction 人为压到 0。[`loss.py:1049-1107`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L1049-L1107)

## 6. Rollout-aware reducer

对于 rollout $g$ 的 fragments $i$，目标是：

$$
\begin{aligned}
L_g
&=\frac{\sum_{i\in g}\sum_t m_{it}\ell_{it}}
        {\max\!\left(1,\sum_{i\in g}\sum_t m_{it}\right)}, \\
L&=\sum_g L_g.
\end{aligned}
$$

RolloutManager 在完整 step 上算 denominator 并复制给 siblings。[`rollout.py:799-814`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L799-L814) `get_sum_of_sample_mean` 让每个 micro-batch 只贡献本地 numerator，却除以同一个完整 rollout denominator；CP 版本对本 rank 两段 zigzag response mask 做同样处理。[`cp_utils.py:47-124`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/cp_utils.py#L47-L124)

若改成 per-sample mean，agent fanout 越多的 rollout 权重越大；若在 mbs 内重算 denominator，同一 rollout 被拆到多个 mbs 后每片都会被独立归一化。

## 7. DP schedule：pack first, distribute second

调度分四步：按 rollout id 划固定 logical rollout steps → 在每 step 内 dynamic first-fit 或 static chunk → 把 mbs 数对齐到 `dp_size × VPP mb_group` → 按 round-robin 或估算 FLOPs 平衡分给 ranks。[`dp_schedule.py:1-37`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/dp_schedule.py#L1-L37)

关键不变量：

- 同一 rollout 的所有 fragments 留在一个 optimizer step；
- 每个 DP rank 每 step 的 mbs 数完全相同，避免 PP collective desync；
- dynamic token cap 按 `max_tokens_per_gpu × cp_size`；
- VPP mbs count 满足 microbatch group 对齐；
- 每个 retained sample 恰好放置一次。

实现见 [`dp_schedule.py:82-209`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/dp_schedule.py#L82-L209)。`balance_by_flops` 明确不保证 token cap，tight-memory recipe 可能因此 OOM。[`dp_schedule.py:55-79`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/dp_schedule.py#L55-L79)

## 8. CP-correct advantage whitening

normalize advantages 时，slime 根据 CP offset 取本 rank 拥有的 response masks，再在 `data_parallel_group(with_context_parallel=True)` 上聚合 masked statistics。即使某 CP rank 本地 response token 数为 0，也必须参与 collective。[`loss.py:818-878`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L818-L878)

否则会出现两类问题：各 CP slice 使用不同 mean/var，改变优化目标；空 rank跳过 collective，其他 ranks deadlock。

## 9. Megatron loss scaling 与 metric 口径

`loss_function` 按 `loss_type` 分派 policy/value/SFT/custom，构造 reducer，再按 actual `step_global_batch_size`、micro-batches 与 DP×CP world size重缩放，替代“每 DP rank sample 数相同”的旧假设。[`loss.py:1283-1382`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L1283-L1382)

allgather-CP 下即使某 rank 无 loss token，也添加 `0*logits.sum()` 强制 autograd 走完整 CP gather backward，保证 collective liveness。[`loss.py:1344-1350`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L1344-L1350)

metrics reducer区分：per-token 用 all-reduced token count 并抵消 CP 重复；per-rollout 直接使用 rollout side 的 step-global batch 常量，不让 DP/CP shard 数改变日志口径。[`cp_utils.py:127-168`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/cp_utils.py#L127-L168)

## 10. Policy loss 组成

最终 policy loss依次组合 PPO/CISPO、OPSM、TIS/custom correction、entropy bonus、可选显式 KL；空 local logprob 时用 zero-connected logits 保持 gradient graph。[`loss.py:1035-1173`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L1035-L1173) value loss则使用 PPO-style clipped/unclipped squared error 的最大值。[`loss.py:1176-1230`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L1176-L1230)

## Related Pages

- [[12_slime_sample_datasource_analysis]] — rollout identity 与 denominator 的来源
- [[14_slime_megatron_training_analysis]] — loss callback 所在的 pipeline forward/backward
- [[17_slime_train_inference_consistency_analysis]] — TIS/top-p/routing 的 behavior policy 语义
- [[31_slime_posttraining_stability_analysis]] — 本页不变量的稳定性视角
- [[11_ppo_analysis]] — PPO/GAE 的算法来源与经典目标
