---
title: "Megatron-LM RL 运行时：GRPO 全链路的实现层"
---

# Megatron-LM RL 运行时：GRPO 全链路的实现层

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **维度**：功能树模块 P。[[30_megatron_rl_posttraining_consistency_analysis]] 覆盖的是**训推一致性的算法层**（refit、`inference_optimized`、logprob 重算的正确性论证）；本页覆盖 `megatron/rl` 的**实现层**——25 个文件里有 20 个此前无任何页面引用。
> **核心文件**：`megatron/rl/{rl_utils,rollout_granularity,sequence_packing_utils,parallel_utils,rl_profiling}.py`、`megatron/rl/agent/**`、`megatron/rl/server/**`、`megatron/rl/inference/**`
> **最近更新**：2026-09-02 首建。

---

## 1. RL 后训练与预训练在结构上的根本差别

预训练的一次迭代是**纯函数**：数据从磁盘来，前向、反向、更新，结束。数据在训练开始前就全部存在。

RL 后训练不是。它的一次迭代要先**生成数据**——拿当前策略去跑若干条 rollout，拿到奖励，才有东西可训。于是同一批 GPU 在一次迭代内要扮演两个角色：先当推理引擎生成，再当训练引擎更新。这带来三个预训练里不存在的问题：

1. **两种角色抢同一批显存**。训练态要持有优化器状态与梯度缓冲，推理态要持有 KV cache。同时留着放不下，于是必须有一套搬进搬出的机制。
2. **生成是变长且不可预测的**。一条 rollout 生成多少 token 事先不知道，而训练需要规整的 batch。
3. **生成可以领先训练若干步**。严格同步（生成完全部再训练）会让 GPU 在生成期间空转训练能力；但领先太多就变成 off-policy，策略梯度不再无偏。

`megatron/rl` 的结构基本就是这三个问题的答案。本页按这个顺序展开。

**本页不覆盖**：训推一致性的算法正确性论证、refit 的权重搬运机制、importance sampling 的偏差分析 → [[30_megatron_rl_posttraining_consistency_analysis]]；推理引擎本体（KV cache、连续批处理、采样） → [[31_megatron_inference_engine_analysis]]；core 侧的 resharding → [[27_megatron_tp_fsdp_resharding_supplements_analysis]]；序列打包的 core 侧调度器 → [[29_megatron_packed_dataset_dynamic_cp_analysis]]。

---

## 2. Rollout 的提交与消费粒度：off-policy 程度的旋钮

这是 `megatron/rl` 里最值得先看的设计，因为它直接控制上面第 3 个问题。

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

**这张表是"流水线开多深"的开关**。`R` 最激进：一条完成立刻腾出槽位放下一条，流水线最满；`B` 最保守，等价于批同步。

### 2.2 并发槽位是算出来的，不是配的

`get_rl_parallel_generation_tasks(args)`（`:19-25`）从三个 flag 推出并发生成槽位数：

```
parallel_generation_tasks = rl_generation_lag + 1
if submission_granularity != "B":  parallel_generation_tasks *= grpo_prompts_per_step
if submission_granularity == "R":  parallel_generation_tasks *= grpo_group_size
```

读法：`--rl-generation-lag` 给出**允许在途的批数**（0 表示严格同步，+1 是当前这批）；提交粒度越细，同一批里能并发的单元越多，于是逐级乘上 `grpo_prompts_per_step` 与 `grpo_group_size`。

**设计要点是"用户配语义、系统算数量"**。用户表达的是"我能容忍落后几批"和"以什么为单位提交"，并发度由这两者推导。反过来让用户直接配并发数会让他必须自己心算这个乘积，而且配错了不会报错——只会静默地改变 off-policy 程度。

### 2.3 流水线与闸门

`megatron/rl/agent/api.py` 里 `_GranularityConfig`（`:199`）从请求构造（`from_request`，`:205`）并自校验（`_validate`，`:218`）；`_SubmissionGate`（`:229`）按 `capacity` 与提交粒度控制放行（`release_after(state)`，`:254`）；`_RolloutPipeline`（`:286`）把工作项（`_InferWorkItem`，`:261`）推过推理、装配、消费三态（产出 `_InferredItem`，`:278`）。

`_GranularityConfig.prevent_dataset_reorder`（`:214`）是个值得注意的属性——**某些粒度组合下不允许重排数据集**。细粒度提交天然会让完成顺序偏离提交顺序，如果数据集本身还要重排，两层乱序叠加就无法复现。

> [!note] 待展开
> `_RolloutPipeline.__init__`（`:289-454`）的状态机细节与 `_record_output_dwell`（`:454`）的驻留统计本页未逐行展开，只覆盖了它的接口与粒度语义。

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

**一个 agent 支持什么，它的服务就暴露什么**——不实现 `EvaluationAgent` 的 agent 起的服务上根本没有 `/evaluation/` 路由，请求得到的是 404 而不是运行时错误。**被否掉的替代是注册全部端点、在处理函数里判断能力后返回 501**：那样错误发生在调用之后，而条件注册让不支持的能力在**路由表层面**就不存在，客户端的服务发现能直接看出来。

数据契约是 pydantic 模型：`RolloutRequest`（`:34`）、`GroupedRolloutRequest`（`:42`）、`Rollout`（`:55`）、`TokenRollout`（`:68`）、`RolloutGroup`（`:85`，实现了 `__iter__`/`__len__`/`__getitem__`，`:92-98`）、`ContrastiveRollout`（`:115`）、`EvaluationRequest`（`:128`）、`EvaluationResult`（`:139`）与 `RewardEvaluationResult`（`:144`）、`EvaluationResponse`（`:152`）。基类 `AgentBaseModel` 声明了 `extra='allow'`（`:30`）——**允许携带未声明字段**，这样自定义 agent 可以在标准契约上附加自己的元数据而不必改基类。

### 3.2 注册表白名单：一处显式的 RCE 防护

`megatron/rl/agent/registry.py` 的 `AGENT_REGISTRY` 是一个 `dict[str, str]`，把 9 个 agent 名映射到导入路径（`RemoteAgent`、`CountdownAgent`、`OpenMathInstructAgent`、`BigMathAgent`、`DAPOAgent`、`GSM8KAgent`、`AIMEAgent`、`NemoGymAgent`、`AceMathAgent`）。`get_agent_class` 在 `KeyError` 时把已知名字列进报错信息。

为什么不直接让 YAML 里写全限定名然后 import？因为 `--langrl-env-config` 指向的是一份**用户提供的 YAML**，允许任意导入路径等于允许任意代码执行。

> 这与 [[41_megatron_config_surface_analysis]] §3.2 的 `TargetAllowlist` 是**同一类防护的两个独立实例**：配置文件里凡是能指定"用哪个类"的地方，都要有一道白名单。两处各自实现、互不知情，说明这在 Megatron 里是一条被反复应用的约束，不是某处的偶然。

### 3.3 内置 agent 与多任务加权

`reward_only_agent.py` 的 `RewardOnlyAgent` 是最小实现基类（用户只需实现取数据集、算奖励、取 prompt 三件事）；`pass_at_evaluation_agent.py` 提供 pass@k 评估；`huggingface_dataset_agent.py:HFDatasetAgent` 是接 HF 数据集的混入；`remote_agent.py:RemoteAgent` 把请求转发到远端 `FastAPIEnvServer`。

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

**KL 用的是 $k_3$ 估计量**（$e^{d} - d - 1$，其中 $d = \log\pi_{\text{ref}} - \log\pi$）而不是直接的 $d$。$k_3$ 恒非负且方差更低——直接用 $d$ 作为 KL 的单样本估计虽然无偏，但可正可负，训练早期会给出"负 KL 惩罚"这种没有意义的梯度。

**IS 权重的存在本身是一个设计信号**。$w_\text{IS} = \pi_{\text{old}} / \pi_{\text{inference}}$ 修正的是**训练引擎与推理引擎算出的 logprob 不一致**——同一份权重、同一个 token，两条实现路径（不同 kernel、不同精度、不同并行度）会给出略微不同的概率。这个差异在 §1 说的"同一批 GPU 扮演两个角色"的架构下不可避免，于是只能在损失里修正。截断系数则防止比值爆炸。上界截断用 `torch.min`（`:2096-2099`），**只截上不截下**——比值过大意味着推理策略给该 token 的概率过低，这类样本的梯度贡献不可信。

裁剪与否被显式返回（`truncated_from_above` / `truncated_from_below`，`:2064-2065`），供指标统计——裁剪率是判断 off-policy 是否过头的直接观测量。

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

但打包**会按当前这批序列的实际长度重新算微批数**，增减都可能。于是主循环为它开了一个口子（`:4603-4608`）：`args.rl_use_sequence_packing or args.sequence_packing_scheduler is not None` 时跳过自动存档与那条断言，只打印一行说明。代码注释写得很直白——"Skip automatic checkpoint on microbatch changes when sequence packing is active as it intentionally reconfigures microbatches."

**这是一处真实的抽象泄漏，值得记住**：`megatron/rl` 的一个 flag 改变了 `megatron/training` 主循环的不变量。§3 的注释还标了另一处（`:4600`）——"Standard microbatch update (sequence packing overrides this in rl_utils.py)"。想读懂打包路径下的微批语义，只看 `training.py` 是不够的。

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

最后一条容易被忽略但很关键：**rotary embedding 会按序列长度缓存**，训练与推理的序列长度分布完全不同，不清缓存会让推理阶段命中一堆训练期长度的条目，白占显存。

`--rl-verify-model-weights-swap` 提供了一道自检：首次 swap 时比对前后的前向输出。**权重搬运出错是最难查的一类 bug**——训练不崩、loss 不炸，只是模型悄悄用了错的权重。花一次前向换一个确定性判据是划算的。

推理侧还需要自己的并行拓扑：`megatron/rl/parallel_utils.py:build_inference_pg_collection` 用 `HyperCommGrid` 建两套网格（decoder 与 expert），由 `--rl-inference-{tensor,pipeline,expert,expert-tensor}-model-parallel-size` 指定——**推理的最优并行度与训练不同**（推理没有梯度与优化器状态，可以用更小的 TP 换更低的通信延迟）。相关的进程组机制见 [[17_megatron_parallelism_orchestration_analysis]]。

---

## 7. 服务面与协议

| 组件 | 端点 / 端口 | 文件 |
|---|---|---|
| 环境服务 | `/rollouts/` `/grouped_rollouts/` `/contrastive_rollouts/` `/evaluation/`（uvicorn `0.0.0.0:<port>`） | `megatron/rl/server/agent/fastapi_env_server.py` |
| 推理服务 | `/base_generate/`（端口取自环境变量 `MEGATRON_RL_INFERENCE_SERVER_PORT`，默认 8294） | `megatron/rl/server/inference/inference_interface_server.py` |
| 本地引擎适配 | 内嵌 text-gen server；对外用 OpenAI 兼容 client | `megatron/rl/inference/megatron.py:MegatronLocal` |

`megatron/rl/inference/inference_interface.py` 的 `InferenceInterface` 按**能力分层**：`ReturnsRaw` / `ReturnsTokens` / `ReturnsLogProbs` 三个 mixin。这与 §3.1 的 agent 能力继承是同一手法——**用类型系统表达"这个后端能给什么"**，而不是用可选返回值或运行时判断。`ReturnsLogProbs` 尤其关键：§4.2 的 IS 修正只有在推理后端能返回 logprob 时才可用，这个前提由类型而非文档保证。

`megatron/rl/__init__.py` 的 `TypeLookupable` 提供跨进程多态反序列化（`register_subclass` / `unwrap`）：请求要跨 HTTP 传，收端必须能从 JSON 恢复出正确的子类。

`MegatronLocal` 把动态推理引擎包成这个接口，并提供 `suspend` / `resume` 供 §6 的切换调用；`set_generation_epoch` 用于 §4.2 的权重时效追踪。

> [!note] 待展开
> `MegatronLocal.launch` 里 inference coordinator 的端口约定、并发连接上限的具体计算式，本页只给了定位，未逐行核对。

---

## 8. 可观测性

`megatron/rl/rl_profiling.py`（902 行）提供**阶段级** profiler：`RLProfiler` / `IterationProfile` / `RunSummary`，由 `--rl-profile` 与 `--rl-profile-dir`（默认 `{save}/profiles`）触发，产出 JSONL 加 CSV，可同步到 wandb/TensorBoard。

同文件带一个独立 CLI：`python -m megatron.rl.rl_profiling {analyze,compare,list}`——`analyze` 找瓶颈、`compare` 对比多次运行、`list` 列最近若干次迭代。

**为什么 RL 需要自己的 profiler**：预训练的一步是同构的，`Timer` 打点就够；RL 的一步由生成、装配、训练三个异质阶段组成，且它们的时间占比随 §2 的粒度配置剧烈变化。要回答"这次跑得慢是因为生成慢还是训练慢"，需要按阶段而非按 kernel 的视图。

训练侧指标由 `log_rl_throughput_metrics` 等并入 `training_log`（`megatron/training/training.py:3534`、`:3712` 处的 `args.rl_use_sequence_packing` 分支即打包效率指标的接入点）。

---

## 9. 约束与边界

| 边界 | 表现 | 证据 |
|---|---|---|
| GRPO 组必须等大 | 优势计算显式假设，作者已标注为待解 | `megatron/rl/rl_utils.py:2072-2073` |
| 打包会打破主循环的微批不变量 | `training.py` 为它跳过断言与自动存档 | `megatron/training/training.py:4603-4612` |
| IS 修正要求推理后端能返回 logprob | 由 `ReturnsLogProbs` 类型保证 | `megatron/rl/inference/inference_interface.py` |
| 自定义 agent 必须进白名单 | 否则 `get_agent_class` 抛 `ValueError` | `megatron/rl/agent/registry.py` |
| 端点随 agent 能力增减 | 不实现对应 ABC 即无该路由 | `megatron/rl/server/agent/fastapi_env_server.py:55-86` |
| 细粒度提交下不允许重排数据集 | 两层乱序叠加会无法复现 | `megatron/rl/agent/api.py:214` |
| off-policy 程度由三个 flag 的乘积决定 | 配错不报错，只静默改变分布 | `megatron/rl/rollout_granularity.py:19-25` |

---

## Related Pages

- [[30_megatron_rl_posttraining_consistency_analysis]] — 训推一致性的算法层（refit、`inference_optimized`、logprob 重算的正确性）；与本页的实现层互补，§4.2 的 IS 修正在那里有偏差分析
- [[40_megatron_feature_tree_analysis]] — 功能树总览；本页覆盖它 §4 仪表盘里规模最大的一块零覆盖（模块 P）
- [[41_megatron_config_surface_analysis]] — §3.2 的 agent 白名单与那里的 `TargetAllowlist` 是同一类防护的两个独立实例
- [[31_megatron_inference_engine_analysis]] — §7 的 `MegatronLocal` 包装的推理引擎本体在那里
- [[17_megatron_parallelism_orchestration_analysis]] — §6 推理侧独立并行度用的 `HyperCommGrid` 与进程组机制
- [[29_megatron_packed_dataset_dynamic_cp_analysis]] — core 侧的序列打包调度器；与本页 §5 的 RL 侧打包是两套实现
