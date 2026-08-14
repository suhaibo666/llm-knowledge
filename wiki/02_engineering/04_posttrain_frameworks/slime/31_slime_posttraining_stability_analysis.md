# slime 后训练稳定性深潜

> **文档类型**：slime 段 3 横切专题 · Post-training Stability
> **源码基线**：slime `main@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`
> **基线提交时间**：2026-08-12T16:50:12+07:00
> **核验日期**：2026-08-14
> **系列入口**：[[slime/index]]
> **结论先行**：slime 的稳定性主要来自“让统计单位在并行与异步变换后仍不变”，而不是靠某个 optimizer trick。完整链条是：group reward normalization → rollout-aware denominator → DP×CP advantage whitening → PPO/GSPO/CISPO/TIS/OPSM 的显式 clip/mask → pause/versioned weight commit → debug replay 与 rollout-engine recovery。任何一环的 denominator、mask 或 version 错位，都可能表现成不报错但训练悄悄漂移。

---

## 1. 稳定性分四层

| 层 | 核心问题 | 常见静默故障 |
|---|---|---|
| 数据语义 | 哪些 token、samples、rollouts 参与目标 | tool token 被训练、compact fanout 被重复计数 |
| 估计量/目标 | reward、advantage、ratio、KL 如何归一/截断 | 小组 std 爆炸、off-policy ratio 高方差 |
| 分布式执行 | DP/CP/PP/mbs 改变是否改 loss | CP slice 各自 whitening、DP rank mbs 数不同导致 deadlock |
| 数值/运维 | 低精度、权重更新、engine failure 如何收敛到一致状态 | FP8 zero block NaN、半套权重、重启 engine 用旧参数 |

slime 当前改动最值得注意的是第二、三层的结合：它不再假设“一条 rollout 等于一个 training sample、每个 DP rank 拿相同样本数”。`rollout_id`、`rollout_mask_sums` 与 `step_global_batch_size` 把训练目标从物理 sample/mbs 重新锚定回逻辑 rollout。

## 2. Reward normalization：先明确组统计单位

GRPO/GSPO/CISPO 与 REINFORCE++ baseline 的默认 reward postprocess 先按 `n_samples_per_prompt` reshape group，减 group mean；GRPO/GSPO/CISPO 可再除 `std + 1e-6`。如果 sample 数不规则，则当前实现会把全部 reward 视成一个 group。[`rollout.py:722-747`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L722-L747)

\[
\hat r_i = r_i-\bar r_g,
\qquad
\hat r_i^{std}=\frac{r_i-\bar r_g}{s_g+10^{-6}}.
\]

这里有三个稳定性旋钮：

1. 全对/全错组 std 为 0，epsilon 防除零，但 normalized advantage 仍几乎没有相对学习信号；可在 rollout 侧动态过滤。
2. `disable_grpo_std_normalization` 对应 Dr.GRPO 风格：减少 group std 对不同难度 prompt 的重加权。[`arguments.py:1004-1016`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/arguments.py#L1004-L1016)
3. custom/compact fanout 若 group size 不规则，必须自定义 reward postprocess；否则“全部剩余 sample 一个组”的 fallback 可能不是想要的统计语义。

## 3. Advantage：算法分支与 CP-correct whitening

`compute_advantages_and_returns` 支持 GRPO、GSPO、CISPO、PPO、REINFORCE++ 与 baseline 版本；KL reward shaping 在 estimator 前进入 returns，OPD KL 可正交地再作用于 advantages。[`loss.py:704-818`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L704-L818)

当 `normalize_advantages` 开启时，源码不是让每个 GPU 对自己 token slice whitening，而是：

1. 按 CP layout 从 full loss mask取出本 rank 真正拥有的 response token mask；
2. 使用 `get_data_parallel_group(with_context_parallel=True)`；
3. 即使本 CP rank 本地 response token 数为 0，也无条件参与 collective；
4. 对全局 masked mean/variance whitening，再按原 sample chunk split 回去。

对应实现见 [`loss.py:819-878`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L819-L878)。如果排除 CP，每个 zigzag slice 会得到不同 affine transform；如果空 rank跳过 all-reduce，其他 rank会 deadlock。这个修复同时保证数值语义和 collective liveness。

`distributed_masked_whiten` 用 FP32 聚合 sum、sum-square、mask count，做 Bessel correction，并以 `epsilon=1e-8` 稳定 rsqrt；全局 mask sum 为 0 时显式报错。[`distributed_utils.py:111-169`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/distributed_utils.py#L111-L169) 多进程测试验证 DP/CP 组合下 whitened advantages invariant。[`test_advantage_whiten_cp.py:177-188`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/tests/test_advantage_whiten_cp.py#L177-L188)

参数校验要求 REINFORCE++ 两个 estimator 必须开启 advantage normalization，防止用户组合出算法未定义的配置。[`arguments.py:1841-1847`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/arguments.py#L1841-L1847)

## 4. Rollout-aware reducer：稳定性的核心修正

### 4.1 为什么 sample mean 会错

一条 agent rollout 可能 fanout 成 N 个 prefix-chained training samples。若逐 sample 平均，它的权重会变成普通 rollout 的 N 倍；若 sibling 被 first-fit 放到不同 micro-batch，每个 mbs 只看局部 sample，又无法知道整个 rollout 的 denominator。

RolloutManager 因此先在完整 step 上累计每个 `rollout_id` 的所有 loss-mask totals，再把同一个 `rollout_mask_sums` denominator复制到该 rollout 的每个 sibling。[`rollout.py:799-815`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L799-L815)

对 rollout \(g\) 的目标是：

\[
L_g=\frac{\sum_{i\in g}\sum_t m_{it}\ell_{it}}
          {\max(1,\sum_{i\in g}\sum_t m_{it})},
\qquad
L=\sum_g L_g.
\]

每个 mbs 只贡献自己持有的 numerator，但都除以完整 rollout denominator；跨 mbs/DP 累加后恰好恢复一次 \(L_g\)。`get_sum_of_sample_mean` 的 docstring明确说明为什么 denominator 必须在 step 级预计算，而不能在 mbs 内临时算。[`cp_utils.py:47-81`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/cp_utils.py#L47-L81) CP 路径对本 rank 的 zigzag mask slice 做同一 denominator 的 partial contribution。[`cp_utils.py:91-124`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/cp_utils.py#L91-L124)

### 4.2 Step normalization 不再依赖每 rank 相同 N

`loss_function` 接收 rollout side 给出的 `step_global_batch_size`，在 Megatron gradient accumulation 中按实际 micro-batch 数、全 DP×CP world size 与逻辑 rollout batch重缩放；它明确替代“每个 DP rank 持有相同样本数”的旧假设。[`loss.py:1283-1325`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L1283-L1325) [`loss.py:1352-1382`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L1352-L1382)

metrics reducer 同样区分 per-token 与 per-rollout：前者 all-reduce token count并抵消 CP 重复计数，后者直接用 rollout side 的常量 `step_global_batch_size`，不让物理 DP/CP shard 数改变报告口径。[`cp_utils.py:127-168`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/cp_utils.py#L127-L168)

## 5. 动态调度：吞吐优化必须保持 PP 同步

`build_dp_schedule` 采用“先 pack、后 distribute”：先按 rollout id 把逻辑 rollouts 切 step，compact siblings留在同一步；再 first-fit/static pack；把 micro-batch 数 K 补到 `dp_size × VPP group` 的倍数；最后按 round-robin 或估计 FLOPs分给 DP rank。[`dp_schedule.py:1-37`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/dp_schedule.py#L1-L37)

硬 invariants 包括：

- 每个 DP rank 每步 micro-batch 数相同，满足 PP sync；
- dynamic first-fit 通常不超过 `max_tokens_per_gpu × cp_size`，仅允许单个超长 sample 独占并超 cap；
- 所有保留 sample 恰放一次；
- 每 rank 的 local mbs schedule恰覆盖本地 partition一次。

这些 invariants 在 module docstring 与实现 assertions 中给出。[`dp_schedule.py:25-37`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/dp_schedule.py#L25-L37) [`dp_schedule.py:122-207`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/dp_schedule.py#L122-L207) CPU tests覆盖 static/dynamic、超长 sample、VPP 对齐、compact grouping 与 trailing rollout trim。[`test_dp_schedule.py:97-326`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/tests/test_dp_schedule.py#L97-L326)

`balance_by_flops` 是明确的风险交换：它考虑 attention 二次项但不保证 token cap，可能 OOM；因此稳定性优先时应先用 first-fit token cap，再逐步验证 FLOPs balance 的显存 headroom。[`dp_schedule.py:65-76`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/dp_schedule.py#L65-L76)

## 6. Policy objective 的方差/偏差防线

### 6.1 PPO asymmetric 与 dual clip

普通 PPO 用 \(r=\exp(-\mathrm{ppo\_kl})\) 和 `[1-eps_low, 1+eps_high]` 非对称区间；设置 `eps_clip_c>1` 后，对负 advantage 增加 dual-clip 下界，限制极端 ratio 让负优势 loss 继续放大。[`ppo_utils.py:124-148`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/ppo_utils.py#L124-L148) `loss.py` 确实把 `eps_clip_c` 转发给 policy loss。[`loss.py:1035-1044`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L1035-L1044)

### 6.2 CISPO 与 GSPO

CISPO 在 stop-gradient 下 clip ratio，梯度仍流经 `log_probs`，所以 clipped token 仍有梯度；源码提示 canonical CISPO 常关闭 lower bound，即 `eps_clip >= 1`。[`ppo_utils.py:151-171`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/ppo_utils.py#L151-L171) 参数校验不会强制，但会在 lower bound仍开启时 warning。[`arguments.py:1871-1880`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/arguments.py#L1871-L1880)

GSPO先跨 CP gather完整序列 logprob，计算 sequence-level mean KL，再扩展到 local token形状；避免把 sequence-level objective 误实现成独立 token ratio。[`ppo_utils.py:95-121`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/ppo_utils.py#L95-L121) [`loss.py:991-1033`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L991-L1033)

### 6.3 KL 防线

`compute_approx_kl` 支持 k1/k2/k3/low_var_kl；k3形式非负，`low_var_kl` 额外 clamp 到 `[-10,10]`，并可乘 importance ratio做 unbiased KL estimator。[`ppo_utils.py:11-51`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/ppo_utils.py#L11-L51) 参数校验禁止 reward-shaping `kl_coef` 与 loss-level `kl_loss_coef` 同时非零，避免双重 KL 约束语义不清。[`arguments.py:1841-1842`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/arguments.py#L1841-L1842)

### 6.4 TIS、ICEPOP、OPSM

| 机制 | 稳定对象 | 代价 |
|---|---|---|
| TIS clip | train-old / rollout ratio 高方差 | clip 后有偏 |
| ICEPOP reject | 区间外严重 mismatch token | 有效 batch/信号减少 |
| OPSM | 负 advantage 且 sequence KL 高的轨迹 | 序列级 selection bias |

TIS 与 ICEPOP 实现见 [`loss.py:884-931`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L884-L931)，OPSM 见 [`ppo_utils.py:54-92`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/ppo_utils.py#L54-L92)。三者都在 mismatch 已量化后才有意义；token/version 错位不是可接受的 off-policy，而是数据 bug。

## 7. 空 token、mask 与 autograd liveness

CP/packing 下某 rank可能没有任何 loss-contributing response token。slime 有两处显式保护：

1. policy loss 若 `log_probs.numel()==0`，加 `0 * logits.sum()`，保持 gradient graph；
2. allgather-CP 下无论本 rank 是否有有效 token，都再加 `0 * logits.sum()`，强制 backward 经过 CP gather/reduce-scatter，避免其他 rank等待 collective 而 deadlock。

对应源码是 [`loss.py:1132-1135`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L1132-L1135) 与 [`loss.py:1344-1350`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L1344-L1350)。这类 zero-value graph edge 是分布式稳定性机制，不是多余计算。

## 8. 低精度数值防线

低精度路径最危险的是“静默 NaN/Inf 继续传播”。block-FP8 转换把每个 block 的绝对最大值 clamp 到至少 `1e-12` 后再除 `FP8_MAX`，防止全零 padding/unused expert block 得到零 scale、随后 `0/0`。[`convert_hf_to_fp8.py:51-68`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/tools/convert_hf_to_fp8.py#L51-L68) CPU test显式断言全零 block无 NaN且 roundtrip仍为零。[`test_block_fp8_zero_block.py:39-70`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/tests/test_block_fp8_zero_block.py#L39-L70)

训练侧 logprob函数要求输入 logits 已是 FP32，softmax 前减全局 max，再用 TP all-reduce聚合 exp sum；这是大词表分布式 softmax的基本数值稳定路径。[`ppo_utils.py:187-229`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/ppo_utils.py#L187-L229)

不过“训练稳定”不等于“训推一致”：BF16 train + FP8 rollout是吞吐/显存取舍，仍要监控 mismatch；详见 [[17_slime_train_inference_consistency_analysis]]。

## 9. 权重更新与 engine recovery

update 前若开启 fault tolerance，actor rank 0先让 RolloutManager恢复 updatable engines，再用 Gloo barrier让所有 train ranks一致进入 reconnect/update。[`actor.py:592-630`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/actor.py#L592-L630) health monitor周期性调用 engine health endpoint，失败则 shutdown/kill并把对应 handle置 `None`。[`health_monitor.py:137-177`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/health_monitor.py#L137-L177)

recovery 会先找出所有 dead index，并发重启，再恢复需要 offload的 memory；updatable engine随后由下一次 weight update覆盖为当前参数，frozen model从自己的 model path/CPU backup恢复。[`rollout.py:384-425`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L384-L425) `recover_updatable_engines` 在 weight update前暂停 health monitoring并只恢复当前可更新 model。[`rollout.py:641-658`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L641-L658)

边界也很清楚：内置 fault tolerance 聚焦 rollout engine；trainer rank failure、集群抢占与 full-job resume仍依赖调度器、Ray policy与 checkpoint。[`fault-tolerance.md:7-27`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/zh/advanced/fault-tolerance.md#L7-L27)

## 10. Debug replay：把随机 rollout 与训练故障解耦

slime 支持：

- rollout-only：只初始化 SGLang并保存 samples；
- train-only/load-debug-rollout-data：固定同一批 rollout，重复调 train parallelism/loss；
- train debug dump：把 CP response tensor还原成完整 per-sample 字段，再按 sample index与 rollout dump join；
- DataSource save/load：保存 epoch、sample offset与 metadata。

debug 路径与 train dump schema见 [`debug.md:26-55`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/zh/developer_guide/debug.md#L26-L55)。RolloutManager加载 debug data会跳过 generation并可稳定 subsample首尾，保存时序列化 `Sample.to_dict()`。[`rollout.py:671-720`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L671-L720) DataSource state persistence见 [`data_source.py:123-160`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/data_source.py#L123-L160)。

这条路径对稳定性非常关键：同一 batch 在 CP1/CP2、不同 mbs packing、不同 TP/EP 下重复训练，才能区分“算法真的发散”与“输入每次随机不同”。

## 11. 稳定性监控面板

| 类别 | 必看指标 | 触发调查 |
|---|---|---|
| reward/group | group reward mean/std、zero-std group fraction、filter reason | zero-std 长期高、某来源全被过滤 |
| advantage | masked mean/std、whiten count、per-source quantile | CP/DP 配置变化时统计跳变 |
| policy | pg clipfrac、KL、entropy、grad norm | clipfrac/KL突增、entropy塌缩 |
| mismatch | train-rollout abs diff、OIS/TIS p50/p95/p99、reject fraction | ratio heavy-tail扩大 |
| data | loss-mask tokens、compact siblings/rollout、truncated/aborted | denominator或有效 token骤变 |
| systems | mbs tokens/FLOPs、rank imbalance、queue age、weight sync time | 某 DP rank尾部、sync占比上升 |
| fault | engine version、restart count、health timeout、debug dump id | 重启后版本不一致或反复失败 |

## 12. 最小稳定性验收矩阵

1. **统计单位**：普通 rollout与 compact fanout在总 token/reward相同条件下 loss权重相同。
2. **mbs invariant**：static、dynamic first-fit、不同 sample顺序得到同一 rollout-level loss/grad容差。
3. **DP/CP invariant**：DP1/2 × CP1/2 的 advantage、loss、grad一致；包含某 CP rank无 response token的样例。
4. **off-policy stress**：人工注入 rollout logprob偏差，检查 TIS clip/ICEPOP reject/OPSM mask与 metrics denominator正确。
5. **低精度**：全零 FP8 block、极端 logits、全 mask/空 local token都不产生 NaN或 collective hang。
6. **weight fault injection**：传输中断不得 resume半套新权重；重启 engine必须先恢复当前 version再接新请求。
7. **replay**：同一 rollout dump在不同并行配置重复训练，sample join与 per-sample tensor完全对应。

当前仓库已有关键 CPU/多进程测试：DP schedule、compact/local reward alignment、CP loss/grad invariant、CP advantage whitening、fully-async queue contract与 GLM-5 e2e consistency gate。[`test_process_rollout_data.py:90-163`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/tests/test_process_rollout_data.py#L90-L163) [`test_loss_cp_invariance.py:231-256`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/tests/test_loss_cp_invariance.py#L231-L256)

## Related Pages

- [[01_slime_architecture_overview_analysis|D08 slime 架构与端到端闭环]]
- [[30_slime_rollout_optimization_analysis|slime Rollout 优化深潜]]
- [[17_slime_train_inference_consistency_analysis|slime 训推一致性深潜]]
- [[12_training_dynamics_stability_analysis|训练动力学稳定性]]
- [[11_fault_tolerance_and_recovery_analysis|故障容错与恢复]]
- [[10_determinism_and_numerical_reliability_analysis|确定性与数值可靠性]]
