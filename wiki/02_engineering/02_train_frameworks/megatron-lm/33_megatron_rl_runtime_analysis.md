---
title: "Megatron-LM RL 运行时：GRPO 全链路的实现层"
---

# Megatron-LM RL 运行时：GRPO 全链路的实现层

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **维度**：功能树模块 P。[[30_megatron_rl_posttraining_consistency_analysis]] 覆盖的是**训推一致性的算法层**（refit、`inference_optimized`、logprob 重算的正确性论证）；本页覆盖 `megatron/rl` 的**实现层**——25 个文件里有 20 个此前无任何页面引用。
> **核心文件**：`megatron/rl/{rl_utils,rollout_granularity,sequence_packing_utils,parallel_utils,rl_profiling}.py`、`megatron/rl/agent/**`、`megatron/rl/server/**`、`megatron/rl/inference/**`
> **最近更新**：2026-09-03。由参考段 42 号迁入 Runtime/工程集成段 33 号；内容 owner 不变。

---

## 1. RL 后训练与预训练在结构上的根本差别

预训练 step 本身当然不是纯函数：它会推进参数、optimizer、scheduler、RNG 与 data iterator 状态。这里真正要对比的是**训练样本的依赖关系**：常规预训练样本可在 step 前由既有数据管线供给，不需要先用当前策略在线生成并计算 reward。

RL 后训练的一次迭代则要先**生成数据**——拿当前策略去跑若干条 rollout，拿到奖励，才有东西可训。在典型共置路径中，同一批 GPU 会先后扮演推理与训练角色；仓内也支持通过 `rank_offset` 为 non-collocated inference model 建独立 rank 网格，跨 rank-set refit 机制由 [[30_megatron_rl_posttraining_consistency_analysis]] 负责（`megatron/rl/parallel_utils.py:16-42`）。这带来三个常规预训练里没有的运行时问题：

1. **共置时两种角色争用显存**。训练态要持有优化器状态与梯度缓冲，推理态要持有 KV cache，因此需要显式的 offload、suspend/resume 与 cache 生命周期。
2. **生成是变长且不可预测的**。一条 rollout 生成多少 token 事先不知道，而训练需要规整的 batch。
3. **生成可以领先训练若干步**。严格同步（生成完全部再训练）会让 GPU 在生成期间空转训练能力；但领先太多就变成 off-policy，策略梯度不再无偏。

`megatron/rl` 的结构基本就是这三个问题的答案。本页按这个顺序展开。

**本页不覆盖**：训推一致性的算法正确性论证、core 侧 Resharding/Refit 的权重搬运机制、importance sampling 的偏差分析 → [[30_megatron_rl_posttraining_consistency_analysis]]；推理引擎本体（KV cache、连续批处理、采样） → [[31_megatron_inference_engine_analysis]]；序列打包的 core 侧调度器 → [[29_megatron_packed_dataset_dynamic_cp_analysis]]。

> [!note] 证据口径
> 下文带 locator 的公式、guard、类型关系与调用顺序是冻结源码事实；由这些事实推出的“为什么这样设计”不是上游作者自述。具体包括 §2.2 的 capacity 分层判据、§3.1 的路由替代方案、§4.2 的估计量与 IS 选择解释、§6 的并行度/校验成本取舍，以及 §8 的 profiler 视角比较，均是本文的分析重建。源码自己明确写出的 allowlist 防任意代码执行与多任务余数分配目标不在此列。

---

## 2. Rollout 的提交与消费粒度：容量和完成边界

这是 `megatron/rl` 里最值得先看的设计：`rl_generation_lag` 控制 policy staleness 上限，提交/消费粒度则决定同一上限如何换算为 slot，以及 slot 在哪一个完成状态归还。

### 2.1 两个轴、五种取值

`megatron/rl/rollout_granularity.py` 用两个 `Literal` 定义了整个设计空间（`:7-8`）：

- `SubmissionGranularity = Literal["R", "G", "B"]` —— **提交**粒度：单条 rollout（R）、一个组（G）、整批（B）
- `ConsumptionGranularity = Literal["G", "B"]` —— **消费**粒度：组或批

配套还有一个 `ReleaseState = Literal["inferred", "assembled", "consumed"]`（`:9`），并用一张表把提交粒度映射到释放时机（`:12-16`）：

| 提交粒度 | 释放状态 | 含义 |
|---|---|---|
| `R` | `inferred` | 单条 rollout 一推理完就释放槽位 |
| `G` | `assembled` | 整组装配完才释放 |
| `B` | `consumed` | 整批被训练消费掉才释放 |

**这张表决定一个已占用槽位何时归还**。`R` 在单条推理完成后即可归还，`B` 要等整批被消费后才归还；但 `B` 不必然等价于批同步。只有 `rl_generation_lag=0` 时它只有一个 batch slot；例如冻结测试明确验证 `B + lag=2` 会得到 3 个 batch slots（`tests/unit_tests/rl/test_rl_utils.py:170-200`）。

### 2.2 并发槽位是算出来的，不是配的

`get_rl_parallel_generation_tasks(args)`（`:19-25`）从三个 flag 推出并发生成槽位数：

```
parallel_generation_tasks = rl_generation_lag + 1
if submission_granularity != "B":  parallel_generation_tasks *= grpo_prompts_per_step
if submission_granularity == "R":  parallel_generation_tasks *= grpo_group_size
```

读法：`--rl-generation-lag` 给出允许生成领先 trainer 的批数，in-flight batch 数为 `lag + 1`；提交粒度越细，一个 batch slot 就要换算成越多的 group/rollout slots，于是逐级乘上 `grpo_prompts_per_step` 与 `grpo_group_size`（`megatron/training/arguments.py:3388-3415`；`megatron/rl/rollout_granularity.py:19-25`）。

从这些事实可重建出的设计判据，是把“策略陈旧度上限”与“调度计量单位”分开：`rl_generation_lag` 控制 rollout 最多可落后多少个 trainer batch；R/G/B 与两个 GRPO 尺寸只把这些 batch slots 换算成 gate capacity，并不额外增加允许的 policy-version lag（`megatron/training/arguments.py:3500-3506`）。这个 capacity 是上限而非实测并发或吞吐；实际驻留还取决于推理、装配和训练各阶段的运行时间。

### 2.3 流水线与闸门

`megatron/rl/agent/api.py` 里 `_GranularityConfig`（`:199`）从请求构造（`from_request`，`:205`）并自校验（`_validate`，`:218`）；`_SubmissionGate`（`:229`）按 `capacity` 与提交粒度控制放行（`release_after(state)`，`:254`）；`_RolloutPipeline`（`:286`）把工作项（`_InferWorkItem`，`:261`）推过推理、装配、消费三态（产出 `_InferredItem`，`:278`）。

`_GranularityConfig.prevent_dataset_reorder` 并不是泛指“禁止数据集重排”，它精确等于 `consumption == "B"`：B-consumption 在输出队列后按 `batch_id/index_in_batch` 缓冲与排序，只放行完整的下一批；G-consumption 则按 group 完成顺序直接 yield（`megatron/rl/agent/api.py:213-215,462-487`）。因此该属性拥有的是**消费端批序恢复**，不是所有细粒度组合的全局确定性保证。

两个 `Literal` 也不构成任意笛卡尔积。pipeline 构造时拒绝 B-submit/G-consume，并暂时完全拒绝 `filter_groups_with_same_reward`，因为丢组后没有再生成会让 batch consumer 卡在不完整批次；全局参数校验还要求 `lag>0` 和 R-submit 必须开启 partial rollouts（`megatron/rl/agent/api.py:218-226`；`megatron/training/arguments.py:562-574`）。此外 remote `FastAPIEnvServer` 覆盖了本地 pipeline 入口，明确拒绝 R-submit 与 streaming，所以最细的 R 流水只适用于本地 agent 路径（`megatron/rl/server/agent/fastapi_env_server.py:121-153`）。

> [!note] 阅读边界
> 下方只追踪状态所有权与完成边界；`_RolloutPipeline.__init__` 的 queue 计数、每项 dwell metric 及异常传播策略不在本页展开。

### 2.4 从训练入口到全 rank 可见的 live hop

真实入口不是 pipeline 类本身，而是训练侧 `get_environment_rollouts()`。它完成 separate-model refit 与推理态切换后，调用 `get_rollout_generator()`：后者计算 capacity、构造包含 `n_prompts` 个 group 与 `samples_per_group` 条 rollout 的 `GroupedRolloutRequest`，再从 agent 取得异步 generator（`megatron/rl/rl_utils.py:585-609,612-679`）。当前多任务入口 `WeightedMultiTask.get_grouped_rollouts()` 始终按权重拆 group；G/R 模式也按权重拆 `parallel_generation_tasks`，而 B 模式把完整的 local-batch pgt 复制给每个 active agent，因为此时 pgt 计量的是各 agent 本地可在途 batch。随后它从多个子 generator 平衡地产出完整 group（`megatron/rl/agent/weighted_multi_task.py:192-285`）。

走基类本地实现的 concrete `GroupedRolloutGenerator` 会启动 prepare、infer、assemble 三个 task，同时由 consume stage 向外 yield；remote `FastAPIEnvServer` 是上文所述的覆盖实现，不走这条 pipeline（`megatron/rl/agent/api.py:490-534`）：

1. `stage_prepare` 按配置的 B/G/R 单元取得 gate slot，并为组内每条 rollout 建 `_InferWorkItem`；
2. infer worker 调 `get_rollout_response()`，在 `inferred` 状态归还 R slot；
3. `stage_assemble` 等齐同组 response、构造 `RolloutGroup`，在 `assembled` 状态归还 G slot；
4. `stage_consume` 对 G-consumption 直接 yield，对 B-consumption 则等齐并按 batch 内索引排序，整批 yield 后在 `consumed` 状态归还 B slot（`megatron/rl/agent/api.py:337-487`）。

完成边界在 pipeline 外：rank 0 从 generator 取满 `n_prompts` 个完整 group；非 streaming 模式还要求 generator 随后耗尽。退出推理态后，`broadcast_object_list` 把同一组 rollouts 发到所有 rank，函数返回才表示本轮训练输入对全 rank 可见（`megatron/rl/rl_utils.py:681-727`）。因此 slot release 只表示相应提交单元可复用，不等于这批数据已经完成全 rank 交接。

---

## 3. Agent 协议：环境侧的契约面

RL 需要"环境"——给 prompt、收答案、算奖励。Megatron 把这层做成了可插拔的 Agent。

### 3.1 能力用继承表达，端点按能力注册

`megatron/rl/agent/api.py` 定义了一族 ABC：`Agent`（`:160`）是基类，`RolloutGenerator`（`:172`）、`ContrastiveRolloutGenerator`（`:179`）、`TokenizedRolloutGenerator`（`:188`）、`GroupedRolloutGenerator`（`:490`）、`EvaluationAgent`（`:537`）各表达一种能力。

有意思的是这套继承**直接决定了 HTTP 服务面的形状**。`megatron/rl/server/agent/fastapi_env_server.py` 的 `launch`（`:51`）逐个 `issubclass` 检查后**条件注册**端点（`:55-86`）：

| 端点 | 注册条件 |
|---|---|
| `/grouped_rollouts/` | `issubclass(env_cls, GroupedRolloutGenerator)` |
| `/contrastive_rollouts/` | `issubclass(env_cls, ContrastiveRolloutGenerator)` |
| `/rollouts/` | `issubclass(env_cls, RolloutGenerator)` |
| `/evaluation/` | `issubclass(env_cls, EvaluationAgent)` |

**一个 agent 支持什么，它的服务就暴露什么**——不实现 `EvaluationAgent` 的 agent 启动后，路由表里就没有 `/evaluation/` 处理器。由此可重建出的取舍是：相比“注册全部端点、在处理函数里再返回不支持”，条件注册把能力边界前移到路由构造；这里的替代方案与服务发现收益是本文分析，不是源码注释声明。

数据契约是 pydantic 模型：`RolloutRequest`（`:34`）、`GroupedRolloutRequest`（`:42`）、`Rollout`（`:55`）、`TokenRollout`（`:68`）、`RolloutGroup`（`:85`，实现了 `__iter__`/`__len__`/`__getitem__`，`:92-98`）、`ContrastiveRollout`（`:115`）、`EvaluationRequest`（`:128`）、`EvaluationResult`（`:139`）与 `RewardEvaluationResult`（`:144`）、`EvaluationResponse`（`:152`）。基类 `AgentBaseModel` 声明了 `extra='allow'`（`:30`）——**允许携带未声明字段**，这样自定义 agent 可以在标准契约上附加自己的元数据而不必改基类。

### 3.2 注册表白名单：一处显式的 RCE 防护

`megatron/rl/agent/registry.py` 的 `AGENT_REGISTRY` 是一个 `dict[str, str]`，把 9 个 agent 名映射到导入路径（`RemoteAgent`、`CountdownAgent`、`OpenMathInstructAgent`、`BigMathAgent`、`DAPOAgent`、`GSM8KAgent`、`AIMEAgent`、`NemoGymAgent`、`AceMathAgent`）。`get_agent_class` 在 `KeyError` 时把已知名字列进报错信息。

为什么不直接让 YAML 里写全限定名然后 import？因为 `--langrl-env-config` 指向的是一份**用户提供的 YAML**，允许任意导入路径等于允许任意代码执行。

> 这与 [[41_megatron_config_surface_analysis]] §3.2 的 `TargetAllowlist` 是**同一类防护的两个独立实例**：配置文件里凡是能指定"用哪个类"的地方，都要有一道白名单。两处各自实现、互不知情，说明这在 Megatron 里是一条被反复应用的约束，不是某处的偶然。

### 3.3 内置 agent 与多任务加权

`reward_only_agent.py` 的 `RewardOnlyAgent` 是训练 rollout 的最小实现基类：子类要实现取数据集、算奖励、取 prompt 三项；由于它还继承 `PassAtEvaluationAgent`，要让已注册的 evaluation 端点可用，子类还必须覆盖默认抛 `NotImplementedError` 的 `evaluation_prompts()`（`megatron/rl/agent/reward_only_agent.py:40-61`）。`pass_at_evaluation_agent.py` 提供 pass@k 评估；`huggingface_dataset_agent.py:HFDatasetAgent` 是接 HF 数据集的混入；`remote_agent.py:RemoteAgent` 把请求转发到远端 `FastAPIEnvServer`。

`weighted_multi_task.py:WeightedMultiTask` 按权重在多个任务间分配采样数，其 `_distribute_counts` 用**最大余数法**——整数配额下按权重分配必然有小数部分，最大余数法保证总数精确等于目标且分配偏差最小。若简单地对每个权重取整再相加，总数会漂移。

---

## 4. GRPO 目标：优势与损失

### 4.1 组内标准化与多轮展开

`megatron/rl/rl_utils.py:853` 的 `calculate_grpo_advantages(rewards, num_turns)` 做的是 GRPO 的核心——**用同一 prompt 的一组回答互相做基线**，不需要单独的 value 网络：

$$
A_{i} = \frac{r_{i} - \mathrm{mean}(r_{\text{group}})}{10^{-4} + \mathrm{std}(r_{\text{group}})}
$$

分母那个 `1e-4`（`:877`）是防除零：一组回答**奖励全同**时标准差为 0（比如全对或全错），此时优势应当趋近 0 而不是 NaN。

多轮轨迹要多做一步。docstring 写明了语义（`:855-860`）："if [[a,b],[c,d,e]] trajectory has reward 1.0, we will get [a,b] with 1.0 and [c,d,e] with 1.0 when doing updates"——**一条多轮轨迹的奖励要复制给它的每一轮**。实现上先按组求 `num_turns.sum(axis=-1)` 得到每组总轮数，用它 `repeat` 均值与标准差（`:868-870`），再把奖励按 `num_turns.flatten()` 展开（`:874`）。

代码里留了一条自陈的限制（`:872-873`）："Making an assumption that all groups are of the same size! @vitalyk: this will go away when we start sending env-based sample reqs."——**当前实现要求组等大**，作者已标注这是待解的临时约束。

### 4.2 损失的四项

`calculate_grpo_loss`（`:2015`）返回的损失是（`:2100-2104`）：

$$
\mathcal{L} = -w_{\text{IS}} \cdot \min(\rho A,\; \bar{\rho} A) + \beta_{\text{KL}} \cdot k_{3} - w_{H} \cdot H
$$

逐项对应源码：

| 项 | 实现 | 控制 flag |
|---|---|---|
| 比值 $\rho$ | `ratios = (current_logprobs - old_logprobs).exp()`（`:2062`） | — |
| 裁剪 $\bar{\rho}$ | `ratios.clamp(1 - clamp_eps_lower, 1 + clamp_eps_upper)`（`:2063`） | `--grpo-clamp-eps-lower` / `--grpo-clamp-eps-upper` |
| KL 项 $k_3$ | `ref_diff.exp() - ref_diff - 1`，其中 `ref_diff = ref_logprobs - current_logprobs`（`:2087-2088`） | `--grpo-kl-beta` |
| 熵项 $H$ | `-current_logprobs.exp() * current_logprobs`（`:2089`） | `--grpo-entropy-term-weight` |
| IS 权重 $w_\text{IS}$ | `(old_logprobs - inference_logprobs).exp()`，再 `torch.min` 截断（`:2093-2099`） | `--rl-inference-logprobs-is-correction` / `--rl-importance-sampling-truncation-coef` |

**KL 用的是 $k_3$ 形式**（$e^{d} - d - 1$，其中 $d = \log\pi_{\text{ref}} - \log\pi$）而不是直接的 $d$。由 $e^d \ge 1+d$ 可知这个逐 token 项恒非负；直接的 $d$ 则可正可负。源码证明的是所选公式，关于为何偏好非负逐样本惩罚的解释属于本文分析；本页不在缺少采样分布前提时宣称其无偏性或统一的方差优势。

**IS 权重的两个输入揭示了它修正的边界**。$w_\text{IS} = \pi_{\text{old}} / \pi_{\text{inference}}$ 比较训练侧保存的 old logprob 与 rollout 返回的 inference logprob。不同 kernel、精度或并行布局可能造成偏差，是启用这项修正的一种工程动机；源码没有证明偏差在每次运行都必然存在，开关也不是强制开启。截断系数限制过大的权重；实现用 `torch.min` 只截上界（`:2093-2099`）。为何把极大比值视为高方差样本，是从公式作出的稳定性分析而非源码自述。

PPO ratio 的上下界命中被显式返回（`truncated_from_above` / `truncated_from_below`，`:2064-2065`），供指标统计。裁剪率是诊断策略漂移的信号之一，但不能单独等同于 policy-version lag；还要结合 `rl_generation_lag` 与 IS 指标判断。

### 4.3 packed 与 unpacked 两种形状

同一个函数要处理两种张量布局（docstring `:2032-2035`）：unpacked 是 `[batch, seq]`、优势 `[batch,]`；packed 是 `[1, bin_size]`、优势 `[num_sequences_in_bin,]`。

差别只在优势怎么铺开（`:2068-2085`）：unpacked 直接 `advantages.view(-1, 1)` 靠广播；packed 则按 `seq_starts` / `seq_lengths` 逐条写进 `packed_advantages[0, start:end]`——**打包后一个"行"里挤着多条序列，广播会把优势串到隔壁序列上**。

---

## 5. 序列打包：把变长生成塞进规整 batch

对应 §1 的第 2 个问题。由 `--rl-use-sequence-packing` 打开，实现在 `megatron/rl/sequence_packing_utils.py`。

- `SequencePacker.pack_sequences`（`:605`）装箱，`create_empty_bins`（`:303`）建空箱，`get_actual_sequence_lengths`（`:274`）按 pad token 反推真实长度
- `distribute_packed_bins`（`:754`）跨 rank 分发，算法由 `--rl-sequence-packing-algo` 选 `fifo` 或 `round-robin`
- `create_packed_seq_params`（`:407`）/ `create_packed_seq_params_for_bin`（`:425`）构造 THD 所需的 `PackedSeqParams`
- `pack_inference_logprobs`（`:485`）与 `compute_packed_inference_logprobs_stats`（`:556`）把 §4.2 的 IS 修正搬到打包布局下
- `log_packing_efficiency`（`:144`）产出效率指标，内部走 `all_gather`（`:211`）汇总

### 5.1 它逼训练主循环让了一步

打包最外溢的后果在 `megatron/training/training.py`。主循环原本有一条不变量：微批数只能增不能减，否则断言失败（`:4609-4612`，"Number of microbatches should not decrease"）——这是 rampup batch size 的正常行为。

但打包**会按当前这批序列的实际长度重新算微批数**，增减都可能。于是主循环为它开了一个口子（`:4600-4608`）：`args.rl_use_sequence_packing or args.sequence_packing_scheduler is not None` 时跳过自动存档与那条断言，只打印一行说明。代码注释写得很直白——"Skip automatic checkpoint on microbatch changes when sequence packing is active as it intentionally reconfigures microbatches."

**这是一处真实的抽象泄漏，值得记住**：`megatron/rl` 的一个 flag 改变了 `megatron/training` 主循环的不变量。紧邻的注释（`:4598`）还写明 "Standard microbatch update (sequence packing overrides this in rl_utils.py)"。想读懂打包路径下的微批语义，只看 `training.py` 是不够的。

---

## 6. 训推态切换：显存腾挪

对应 §1 的第 1 个问题。`megatron/rl/rl_utils.py:2110` 的 `megatron_rl_inference_mode` 是一个上下文管理器，进入时把训练态换成推理态，退出时换回来。

它管的东西比"调 `model.eval()`"多得多：

| 搬什么 | 触发 |
|---|---|
| 优化器状态与梯度 buffer offload 到 CPU | `--rl-offload-optimizer-during-inference` |
| 独立推理模型权重（UVM / torch_memory_saver 双后端） | `--rl-inference-model-unified-memory-level`、`--rl-offload-inference-model-weights-when-idle` |
| KV cache 的处置 | `--rl-kv-cache-management-mode {persist,offload,recompute}` |
| CUDA graph 形态切换 | `--rl-persist-cuda-graphs` / `--rl-training-cuda-graphs` |
| rotary embedding 的 `lru_cache` 清理 | 自动 |

最后一条容易被忽略但很关键：进入与退出 inference mode 都会对 vanilla `RotaryEmbedding.forward` 的 LRU cache 执行 `cache_clear()`。源码明确给出的 correctness 风险是**训练阶段复用 inference mode 缓存的 frequency tensor 会破坏 RL 训练**；因此这是双向切相时的状态隔离，不应只解释成回收显存（`megatron/rl/rl_utils.py:2157-2162,2231-2238`）。

`--rl-verify-model-weights-swap` 提供了一道调试自检：只要启用该 flag，separate-model 路径就在**每次** `swap_model_weights()` 后分别跑训练模型与推理模型的前向并比较输出；当前调用链没有 once-only guard（`megatron/rl/rl_utils.py:187-211,645-657`）。**权重搬运出错是最难查的一类 bug**——训练不崩、loss 不炸，只是模型悄悄用了错的权重。这个判据能换来确定性，但也会给每次 rollout 收集增加两次调试前向，因此不应被理解成只付一次成本。

推理侧还需要自己的并行拓扑：`megatron/rl/parallel_utils.py:build_inference_pg_collection` 用 `HyperCommGrid` 建两套网格（decoder 与 expert），由 `--rl-inference-{tensor,pipeline,expert,expert-tensor}-model-parallel-size` 指定。独立配置证明训推拓扑可以不同；“推理无梯度/优化器状态时可评估更小 TP 以减少通信”是一个候选选型判据，不是源码承诺的最优配置。相关的进程组机制见 [[17_megatron_parallelism_orchestration_analysis]]。

---

## 7. 服务面与协议

| 组件 | 端点 / 端口 | 文件 |
|---|---|---|
| 环境服务 | `/rollouts/` `/grouped_rollouts/` `/contrastive_rollouts/` `/evaluation/`（uvicorn `0.0.0.0:<port>`） | `megatron/rl/server/agent/fastapi_env_server.py` |
| 推理服务 | `/base_generate/`（端口取自环境变量 `MEGATRON_RL_INFERENCE_SERVER_PORT`，默认 8294） | `megatron/rl/server/inference/inference_interface_server.py` |
| 本地引擎适配 | 内嵌 text-gen server；对外用 OpenAI 兼容 client | `megatron/rl/inference/megatron.py:MegatronLocal` |

`megatron/rl/inference/inference_interface.py` 声明了 `ReturnsRaw` / `ReturnsTokens` / `ReturnsLogProbs` 三个能力 mixin；前两个确实用于 agent 的 `isinstance` 分支。但在冻结源码中，`ReturnsLogProbs` 只有定义，没有实现者或运行时检查，因而只是**尚未闭合的接口意图**，不能证明任意新后端都返回 logprob（`megatron/rl/inference/inference_interface.py:45-60`；`megatron/rl/agent/reward_only_agent.py:87-110`）。

当前 `MegatronLocal` 的 logprob 闭环来自具体实现而非该 marker：它继承 `ReturnsTokens` / `ReturnsRaw`，`launch()` 强制 `return_log_probs=True` 与 `skip_prompt_log_probs=True`，`base_generate()` 又显式请求 `logprobs=True` 并把 `generation_log_probs` 写入 `InferenceResponse.logprobs`（`megatron/rl/inference/megatron.py:39-40,50-86,88-105`）。因此 §4.2 的 IS 在这条当前路径上有输入；接入其它 inference backend 时，必须逐实现验证 response，而不能仅凭类型名放行。

`megatron/rl/__init__.py` 的 `TypeLookupable` 提供跨进程多态反序列化（`register_subclass` / `unwrap`）：请求要跨 HTTP 传，收端必须能从 JSON 恢复出正确的子类。

`MegatronLocal` 把动态推理引擎包成这个接口，并提供 `suspend` / `resume` 供 §6 的切换调用；`set_generation_epoch` 用于 §4.2 的权重时效追踪。

> [!note] 待展开
> `MegatronLocal.launch` 里 inference coordinator 的端口约定、并发连接上限的具体计算式，本页只给了定位，未逐行核对。

---

## 8. 可观测性

`megatron/rl/rl_profiling.py`（902 行）提供**阶段级** profiler：`RLProfiler` / `IterationProfile` / `RunSummary`，由 `--rl-profile` 与 `--rl-profile-dir`（默认 `{save}/profiles`）触发，产出 JSONL 加 CSV，可同步到 wandb/TensorBoard。

同文件带一个独立 CLI：`python -m megatron.rl.rl_profiling {analyze,compare,list}`——`analyze` 找瓶颈、`compare` 对比多次运行、`list` 列最近若干次迭代。

从该 profiler 的数据模型可得到一个运维判据：RL step 同时包含生成、装配与训练，且比例会随 §2 的粒度配置变化；定位瓶颈时应先看阶段驻留，再下钻 kernel。这里与通用 `Timer` 的比较是本文分析，不表示普通预训练只需某一种打点工具。

训练侧指标由 `training_log` 汇合：sequence-packing 指标分别进入 TensorBoard 与 stdout，RL throughput 则由 `log_rl_throughput_metrics` 独立接入（`megatron/training/training.py:3534-3539,3712-3713,3764-3768`）。

---

## 9. 约束与边界

| 边界 | 表现 | 证据 |
|---|---|---|
| GRPO 组必须等大 | 优势计算显式假设，作者已标注为待解 | `megatron/rl/rl_utils.py:868-877` |
| 打包会打破主循环的微批不变量 | `training.py` 为它跳过断言与自动存档 | `megatron/training/training.py:4598-4612` |
| IS 修正要求推理后端能返回 logprob | 当前 `MegatronLocal` 显式请求并写回；`ReturnsLogProbs` marker 尚未接线，不能充当通用保证 | `megatron/rl/inference/megatron.py:39-40,50-86,103-105` |
| 自定义 agent 必须进白名单 | 否则 `get_agent_class` 抛 `ValueError` | `megatron/rl/agent/registry.py` |
| 端点随 agent 能力增减 | 不实现对应 ABC 即无该路由 | `megatron/rl/server/agent/fastapi_env_server.py:55-86` |
| B-submit/G-consume 与 reward-equal filtering 当前不可用 | pipeline 构造即 assert；过滤被拒绝是因为缺少丢组再生成 | `megatron/rl/agent/api.py:218-226` |
| `lag>0` 或 R-submit 的前置 | 都必须启用 partial rollouts | `megatron/training/arguments.py:562-574` |
| remote environment 的细粒度边界 | `FastAPIEnvServer` 拒绝 R-submit 与 streaming，不走本地 `_RolloutPipeline` | `megatron/rl/server/agent/fastapi_env_server.py:121-153` |
| B-consumption 的顺序保证 | 按 batch/id 缓冲并排序完整下一批；G-consumption 按完成顺序 yield | `megatron/rl/agent/api.py:213-215,462-487` |
| policy staleness 与调度 capacity 不同 | `rl_generation_lag` 控制最多落后多少批；R/G/B 与 GRPO 尺寸只把 batch slots 换算成 gate capacity | `megatron/training/arguments.py:3388-3415,3500-3506`；`megatron/rl/rollout_granularity.py:19-25` |

---

## Related Pages

- [[30_megatron_rl_posttraining_consistency_analysis]] — 训推一致性的算法层（refit、`inference_optimized`、logprob 重算的正确性）；与本页的实现层互补，§4.2 的 IS 修正在那里有偏差分析
- [[40_megatron_feature_tree_analysis]] — 功能树总览；本页接管它 §4 仪表盘中重构前规模最大的一块零覆盖（模块 P）
- [[41_megatron_config_surface_analysis]] — §3.2 的 agent 白名单与那里的 `TargetAllowlist` 是同一类防护的两个独立实例
- [[31_megatron_inference_engine_analysis]] — §7 的 `MegatronLocal` 包装的推理引擎本体在那里
- [[17_megatron_parallelism_orchestration_analysis]] — 推理侧独立并行度使用的 `HyperCommGrid` 与进程组机制。
- [[29_megatron_packed_dataset_dynamic_cp_analysis]] — core 侧的序列打包调度器；与本页 §5 的 RL 侧打包是两套实现
