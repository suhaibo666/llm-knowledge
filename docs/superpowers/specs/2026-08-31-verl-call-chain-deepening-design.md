# Verl 调用链深潜补充设计

> 批准背景：用户在 2026-08-31 复审中指出，现有页面虽有“是什么、怎么做、为什么”，但“怎么做”仍停留在功能点和阶段顺序，读者无法据此跟踪实现。
> 源码基线：`verl` `main` @ `254a23edc62f25ebfae626e3932ae285d6f86009`
> 上位设计：`docs/superpowers/specs/2026-08-31-verl-analysis-domain-refactor-design.md`
> 文档范围：`wiki/02_engineering/04_posttrain_frameworks/verl/`

## 1. 问题定义

上一轮重构解决了概念 owner、模块定位和设计理由，但没有把“实现思路与细节”落实为真实符号链。当前常见形态是：

- 用“Trainer → Worker → Engine”代替实际方法调用；
- 用阶段表说明先后，却不解释一次调用怎样跨 decorator、Ray、TransferQueue 和 backend；
- 一个大范围 locator 覆盖整段叙述，读者无法把每个 hop 对回源码；
- 说明输入输出字段，却不记录字段在哪个 owner 被读取、改名、补齐和写回；
- 写了 async/future，却没有标出 `ray.get`、`asyncio.create_task`、collect 或 publication barrier 的阻塞点。

因此第二轮的中心命题是：**“怎么做”必须是一条可执行的源码阅读路径，而不是组件关系摘要。** 读者应能从页面给出的第一个符号设置断点，沿调用、数据和状态变化走到该机制的完成点。

## 2. 范围与不变项

- 修改 7 篇现有页面：01、10、11、12、13、15、18。
- 不新增、删除、重命名、拆分或合并 wiki 页面。
- 不改变既有 concept ownership；生命周期页可以展示跨 owner 的调用边界，但不能复制 owner 页的内部推导。
- 冻结 Verl 源码提交不变，不修改或移动上游源码工作区及其用户文件 `GRPO_Analysis.md`。
- 不追求穷举函数；每页选择 1–4 条最能承载页面 thesis 的代表性 trace。

## 3. Trace 单元合同

每条承重调用链必须依次回答以下内容：

1. **触发条件与入口**：哪个配置、runner、hook 或公开方法进入这条路径。
2. **真实符号链**：使用源码中的类名和方法名，逐 hop 给出已打开验证的 `file:line`。
3. **数据与状态账本**：每一 hop 消费什么对象/字段，修改谁拥有的状态，产出什么对象/字段。
4. **执行语义**：标出本地调用、动态 wrapper、Ray remote、coroutine、background task、future、collect 与显式 barrier。
5. **分支与失败**：至少覆盖决定路径形态的 guard，以及异常在哪一层被传播、聚合或转成状态。
6. **设计解释**：这条链为什么在此处分层，最接近的替代方案会造成什么耦合或错误。
7. **完成定义**：函数返回、TQ 字段可见、group 状态 terminal、参数更新完成和 rollout 新版本可见必须区分。

推荐用两种紧邻结构表达：

```text
Caller.method
  → Boundary.wrapper
    → RemoteWorker.method
      → StateOwner.method
```

| Hop | 输入/前态 | 动作与 owner | 输出/后态 | 执行语义 |
|---|---|---|---|---|
| 1 | ... | ... | ... | local / remote / async / blocking |

禁止只画组件名、只罗列函数、逐行转述源码或用一段宽泛 locator 代替逐 hop 证据。

## 4. 七页交付合同

### 4.1 `10_verl_end_to_end_iteration_analysis.md`：样板页

该页先建立全域 exemplar，至少包含四条 trace：

1. **启动链**：`TaskRunnerV1.run` → trainer registry → `trainer.init()` → `AgentLoopManagerTQ.create` → `trainer.fit()` → TQ close。
2. **prompt 到 finished group**：`step()` → `prepare_step()` → `_submit_batch_to_rollout()` → `AgentLoopManagerTQ.generate_sequences()` → `AgentLoopWorkerTQ.generate_sequences()` → `_run_prompt()` → `_run_agent_loop()` → TQ `running/finished/failure` → `ReplayBuffer.sample()`。
3. **old log-prob 计算链**：`_compute_old_log_prob()` → 动态绑定的 `actor_rollout_wg.compute_log_prob()` → WorkerGroup dispatch/Ray remote → `tqbridge` 物化 → `ActorRolloutRefWorker.compute_log_prob()` → `TrainingWorker.infer_batch()` → `BaseEngine.infer_batch()` → TQ 回写与 `old_log_probs` 重命名。
4. **actor update 到下一版本可见**：`_update_actor()` → worker mini-batch/update → Engine forward/backward/optimizer → mode `on_step_end()` → CheckpointEngine publication → rollout resume。

九阶段表降级为导航摘要；上述 trace 才是正文骨架。对 critic-first 必须区分“当前固定源码顺序”与“数学上不可交换”。

### 4.2 `11_verl_single_controller_analysis.md`

- 用一条真实的 `actor_rollout_wg.compute_log_prob(batch)` 贯穿 `_bind_worker_method`、method metadata、dispatch、Ray refs、worker method 和 collect。
- 单独追踪 `blocking=False` 怎样返回 `DataProtoFuture`，何时 `ray.get`，为什么它不构成跨 rank 事务。
- 记录 DP mesh 怎样决定 shard，而不是只列出 Dispatch 枚举。

### 4.3 `12_verl_dataproto_analysis.md`

- 追踪一份 V0/local `DataProto` 从构造、索引/reorder、chunk、worker 输出到 concat/union 的样本身份变化。
- 追踪 `DataProtoFuture` 的 refs、collect、可选 dispatch 和 materialize 路径。
- 对每个操作记录 tensor、non-tensor、meta 三类字段怎样同步变化；不能退化成 API 清单。

### 4.4 `13_verl_workers_engine_analysis.md`

- **infer trace**：组合 Worker RPC → `TrainingWorker.infer_batch()` → Engine eval context → backend `infer_batch/forward_backward_batch` → postprocess。
- **train trace**：actor update → `train_mini_batch()` 切 epoch/mini-batch → `TrainingWorker.train_batch()` → Engine train mode → zero-grad/forward-backward/optimizer-step → metrics。
- **export trace**：actor Worker 如何在 naive/full/delta 路径选择 Engine export 与 CheckpointEngine 边界。
- registry 表继续保留，但必须展示一次 `model_type × backend × device/vendor` 的实际解析顺序。

### 4.5 `15_verl_rl_algorithms_analysis.md`

- **advantage trace**：Trainer 从 TQ 取字段 → padded `DataProto` → KL/rollout correction → V1 multi-trajectory adapter → `get_adv_estimator_fn()` → estimator → nested advantages/returns 写回。
- **policy-loss trace**：actor update → Worker/Engine forward → `actor_loss()` → `get_policy_loss_fn()` → 具体 loss → entropy/KL → `agg_loss()` → backward。
- 算法公式与设计家族保留，但每个家族不再孤立于实际调用和张量 shape/mask 变化。

### 4.6 `18_verl_agent_loop_reward_runtime_analysis.md`

- 将抽象 sequence diagram 改为真实方法名，区分普通 `AgentLoopManager.generate_sequences()` 的 gather 路径与 V1 `AgentLoopManagerTQ` 的 fire-and-forget 路径。
- V1 trace 必须覆盖 `generate_sequences()` → per-sample background task → `_run_prompt()` → rollout.n session tasks → `_run_agent_loop()` → concrete `run()` → postprocess → TQ trajectory keys → group terminal status。
- reward trace 必须展示 async worker handle 与 colocated trainer 两条实际分叉，以及 final-output reward 如何广播到 sibling outputs。

### 4.7 `01_verl_architecture_overview_analysis.md`

- 最后反向收敛，不先凭概念重写。
- 增加一条从 `TaskRunnerV1.run()` 到下一 actor version rollout 可见的真实顶层链；每个 hop 只解释边界和 owner，内部细节链接上述 owner 页。
- 静态能力表保留；动态章节不得再用 `Manager/Worker/Engine` 泛称替代承重入口符号。

## 5. 证据规则

- 每个调用 hop 的 locator 必须在冻结源码检出中重新打开；不能仅确认文件存在或行号未越界。
- 同一个大范围可以证明函数内部机制，但跨文件调用必须分别引用 caller 与 callee。
- decorator、registry 和动态绑定必须同时引用“元数据写入处”和“元数据消费处”。
- 设计理由若源码/提交历史没有直接说明，继续标记 `【分析推断】`。
- 对外部 TransferQueue 内部、真实多节点性能和 crash 原子性保持既有证据边界。

## 6. 验收标准

逐页人工回答以下问题，任何一项为“否”则不得完成：

1. 读者能否从页面给出的第一个符号开始设置断点并继续跟踪？
2. 是否至少跨过一次该页负责解释的真实边界，而不是只有组件箭头？
3. 是否知道每一 hop 的输入、输出和状态 owner？
4. 是否看得出哪些调用立即返回、哪些在 background 执行、哪些显式等待？
5. 是否区分函数返回、字段持久化、参数更新和外部可见四种完成？
6. 是否覆盖决定路径形态的 guard 和一个真实失败边界？
7. 删除所有代码块后，因果解释是否仍成立；只看代码块时，调用顺序是否仍可追踪？

机械门禁保持不变：

- `python tools/check_links.py --strict`
- `python tools/check_math.py --changed --strict`
- `python -m pytest tools/`
- `npm run docs:test`
- `git diff --check`
- 7 篇页面均不超过 500 行，`## Related Pages` 保持 3–7 条。

## 7. 实施顺序

1. 重做 10，形成 trace 表达样板并人工复核。
2. 按控制与计算路径依赖重做 11、13、15。
3. 重做数据和 trajectory 路径 12、18。
4. 用已验证的 owner trace 反向收敛 01。
5. 更新 changelog，不修改 index；执行全套质量门和最终 diff 审查。
