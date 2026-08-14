# slime RL Loss、Reducer 与并行调度实现分析

> **源码基线**：slime `main@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`
> **核验日期**：2026-08-14 · **系列**：[[slime/index]]
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

## 3. PPO、GSPO 与 CISPO 目标

普通 PPO 使用 token ratio $r=\exp(\log\pi_\theta-\log\pi_{\mathrm{old}})$，取 unclipped/clipped surrogate 的最大值；可选 dual clip 只作用于负 advantage。[`ppo_utils.py:124-148`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/ppo_utils.py#L124-L148)

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
