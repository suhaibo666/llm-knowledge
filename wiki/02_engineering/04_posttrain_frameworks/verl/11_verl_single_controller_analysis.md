---
title: "verl single-controller：当前 V0/V1 共用的 Ray 控制基座"
---

# verl single-controller：当前 V0/V1 共用的 Ray 控制基座

> **代码基准**：verl `main` @ `254a23edc62f25ebfae626e3932ae285d6f86009`（2026-08-28）
> **最后更新**：2026-08-31 · **定位**：Ray/SPMD 控制基础设施唯一机制 owner
>
> **核心结论**：`single_controller` 不是只属于 V0 `RayPPOTrainer` 的旧控制面。当前 V1 trainer、
> rollout server 和 CheckpointEngine 仍共同使用 `RayResourcePool`、`RayWorkerGroup`、`Worker` 与
> `@register`。它拥有的是“把一组 SPMD rank 暴露成一个 controller 对象”的 RPC 语义；
> Trainer 生命周期、批数据语义和模型计算分别属于其它层。

---

## 1. 先划清当前边界

V1 trainer 直接导入 `RayWorkerGroup`、`ResourcePoolManager` 和 `create_colocated_worker_cls`，并在初始化时
按 resource pool 创建 colocated worker class 与 group（`verl/trainer/ppo/v1/trainer_base.py:44-48,277-306`）。
rollout manager 同样接收 `RayWorkerGroup`/`RayResourcePool`（`verl/workers/rollout/llm_server.py:32,475-509`）；
CheckpointEngine worker 继承 `Worker`，manager 还会临时用 rollout handles 组装 `RayWorkerGroup`
（`verl/checkpoint_engine/base.py:22-24,303-354,381-425,521-525`）。

所以本页不能再写成“V0 控制面”。准确边界是：

| 本页拥有 | 本页不拥有 |
|---|---|
| rank 元信息、资源池、placement group、worker 生命周期 | V1 sync/async 的 step 状态机 |
| group method 动态绑定、dispatch、execute、collect | `DataProto` 字段与 batch 不变量 |
| Ray actor 拉起、colocation、同步/异步返回 | Engine 的模型、优化器与并行布局 |
| `DataProtoFuture` 在 RPC 层的延迟物化入口 | TransferQueue 的 key、tag 与存储 |

V0 与 V1 的区别不是“是否使用 single-controller”，而是 controller 传什么：V0 主循环大量传递完整
`DataProto`；V1 controller 主要传 `KVBatchMeta`，worker 在 `tqbridge` 边界才取实际 TensorDict。

这里最关键的对象拆分是 `Worker` 与 `WorkerGroup`。**是什么**：前者表示一个 rank 的进程内执行环境，后者表示 controller 眼中的一组 SPMD ranks。**怎么做**：`Worker` 从环境构造 rank/mesh 元数据并实现 rank-local 方法；`WorkerGroup` 保存 handles，扫描这些方法的注册元数据，生成 dispatch/remote/collect wrapper。**为什么**：rank-local 业务代码不应同时承担 Ray 拓扑和聚合，controller 也不应直接管理每个 rank 的模型状态。`【分析推断】` 若把二者合成一个对象，每个业务方法都要重复处理“我是谁、发给谁、怎样收集”；若 controller 逐 handle 调用，则 mesh/collect 语义会散落到上层。代价是一次看似普通的方法调用背后存在动态绑定，排错必须同时检查定义端 metadata 与 group 绑定结果。

## 2. 一次 group method call 的五段路径

```mermaid
flowchart LR
    C["Controller 调用 group method"] --> B["动态绑定 wrapper"]
    B --> D["dispatch 切分参数"]
    D --> E["Ray remote 并发执行"]
    E --> K["collect 聚合结果"]
    K --> R["同步值或 DataProtoFuture"]
```

这五段分别由 `WorkerGroup._bind_worker_method`、Ray `func_generator`、decorator registry、Ray actor
handle 和 collect function 承担（`verl/single_controller/base/worker_group.py:185-240`；
`verl/single_controller/ray/base.py:49-68`；`verl/single_controller/base/decorator.py:300-442`）。

这条路径的定位是把“一次逻辑 SPMD 调用”变成 N 个物理 RPC，而不是隐藏算法阶段。实现把切分与收集留在 wrapper，把单 rank 计算留给原始 method。`【分析推断】` 这样做使同一个 actor/ref/critic 方法可以复用 one-to-all、DP 或 mesh-aware 形状，而无需为 Ray 另写一套业务 API；相对地，wrapper 不会替 Trainer 推断跨方法依赖，也不会把 N 个 actor 的副作用合成原子事务。

### 2.1 先看一条真实调用：`compute_log_prob`

下面这条链以 V1 controller 的 `actor_rollout_wg.compute_log_prob(batch)` 为例。调用点只拿着 `KVBatchMeta`，但远端函数签名接收的是 TensorDict；中间转换不是业务代码显式写出的，而是注册与绑定机制共同插入的：

```text
PPOTrainer._compute_old_log_prob
  → RayWorkerGroup.compute_log_prob(KVBatchMeta)            # 动态绑定的方法
      → dispatch_lazy_compute_data_proto("actor", ...)
          → query actor dp_rank_mapping
          → BatchData(KVBatchMeta).chunk(dp_size)
              → KVBatchMeta → BatchMeta → dp shards
          → map each DP shard to every TP/PP/CP rank in that DP replica
      → RayWorkerGroup.execute_all("compute_log_prob", shard_per_rank)
          → worker_handle.compute_log_prob.remote(shard)
              → register wrapper materializes incoming future if any
              → tqbridge: BatchMeta → TensorDict
              → ActorRolloutRefWorker.compute_log_prob(TensorDict)
      → ray.get(all ObjectRefs)                              # blocking=True
      → collect_lazy_compute_data_proto("actor", ...)
          → keep only ranks designated by collect_mask
          → concat BatchMeta/TensorDict result
  → updated KVBatchMeta
```

定义端从 `ActorRolloutRefWorker.compute_log_prob` 的 `@register(make_nd_compute_dataproto_dispatch_fn("actor"))` 开始（`verl/workers/engine_workers.py:699-705`）。`register` 先用 `tqbridge` 包住原函数，再把 dispatch、execute 和 blocking 写入 `MAGIC_ATTR`（`verl/single_controller/base/decorator.py:398-442`）。group 初始化时 `_bind_worker_method` 读取这些属性，解析 dispatch/collect 与 execute 方法，最后调用 Ray `func_generator` 把新方法挂到实例上（`verl/single_controller/base/worker_group.py:185-250`；`verl/single_controller/ray/base.py:770-776`）。

| Hop | 输入/前态 | 动作与 owner | 输出/后态 | 执行语义 |
|---|---|---|---|---|
| method definition | 原始 `compute_log_prob(TensorDict)` | `register` 套 `tqbridge` 并写 metadata | 可被 group 发现的 worker method | class 定义时完成，不发 RPC |
| group binding | worker class + `MAGIC_ATTR` | `_bind_worker_method` 解析形状并调用 `func_generator` | `RayWorkerGroup.compute_log_prob` | group 初始化时一次性绑定 |
| dispatch | controller 的 `KVBatchMeta` | actor-mesh dispatch 查询 DP mapping，转换/切分为 `BatchMeta` shards | 与 worker 数等长的参数列表 | 本地；同一 DP shard 可映射到多个模型并行 rank |
| execute | per-rank shard | `execute_all_async` 对每个 handle 调 `.remote()` | `list[ObjectRef]` | N 个 Ray RPC 已开始执行 |
| worker entry | `BatchMeta` | `tqbridge` 从 TQ 取实际字段，再调用 rank-local method | TensorDict 输出或非 source rank 的 `None` | 远端；TQ get 是函数体前置条件 |
| blocking collect | `list[ObjectRef]` | `func_generator` 先 `ray.get`，collect mask 再选 source ranks 并拼接 | 更新后的 meta/聚合结果 | controller 阻塞到全部 refs resolve |

DP-mesh 切分与收集位于 `verl/single_controller/base/decorator.py:202-304`；`execute_all_async` 将等长参数列表逐项送到对应 worker handle（`verl/single_controller/ray/base.py:778-795,862-890`）；`func_generator` 的固定顺序是 dispatch → execute → 可选 `ray.get` → collect（`verl/single_controller/ray/base.py:49-67`）。`BatchData.chunk` 在 V1 路径把 `KVBatchMeta` 提前翻译成 `BatchMeta`，避免每个 rank 重复向 controller 查询 key（`verl/protocol.py:1268-1286`）。

### 2.2 `Worker` 只描述一个 rank

`Worker` 从环境变量读取 world size、rank、local rank、master address 和设备可见性，并保存 mesh-aware
dispatch/collect 元数据（`verl/single_controller/base/worker.py:76-147,169-231,283-321`）。它不决定
actor/ref/critic 的业务角色，也不实现 PPO；具体 worker class 在其上提供被远程调用的方法。

`execute_with_func_generator` 和 `execute_func_rank_zero` 是少量允许 controller 向 rank 注入函数的通用入口
（`verl/single_controller/base/worker.py:321-344`）。这类能力仍属于 RPC 基座，不能据此推断业务方法的
同步顺序。

### 2.3 `WorkerGroup` 把 N 个 rank 伪装成一个对象

`ResourcePool` 只声明每节点进程数、world size 和 local-rank 分布；`ClassWithInitArgs` 延迟保存 class 与
构造参数（`verl/single_controller/base/worker_group.py:27-101`）。`WorkerGroup` 持有实际 worker handles，
在 `_bind_worker_method` 中扫描用户 worker class 的方法，把带 dispatch metadata 的方法动态绑到 group
实例（`verl/single_controller/base/worker_group.py:123-240`）。

因此 `actor_wg.update_actor(...)` 看起来是一条普通 Python 调用，实际 wrapper 已经固定：

1. 用 method 上的 metadata 选择 dispatch、collect、execute 与 blocking；
2. 将输入按数据并行 mesh 切给目标 ranks；
3. 对每个 actor handle 调相同 method；
4. 同步收集，或返回可延迟 materialize 的 future。

动态绑定只提供调用形状，不保证不同业务调用之间的事务性；Trainer 仍必须显式编排先后关系。

把调用形状写在 method metadata 而不是手写每个 Ray proxy，还有一个直接收益：定义业务方法时即可声明 dispatch、execute、blocking 与 collect，`WorkerGroup` 能对不同 role 统一生成入口（`verl/single_controller/base/decorator.py:398-444`）。`【分析推断】` 替代方案是为每个 worker method 维护一份 controller 代理，二者容易在参数切分或返回聚合上漂移。当前设计的成本也很明确：普通调用点看不见 RPC 形状，错误 metadata 可能直到 group 绑定或远程执行时才暴露。

## 3. `@register` 决定 RPC 形状，不决定算法

`Dispatch` 定义 one-to-all、all-to-all、DP compute、Megatron compute 等预设；`Execute` 决定所有 rank
或 rank zero 执行（`verl/single_controller/base/decorator.py:40-68`）。`register()` 把 dispatch mode、
execute mode 与 blocking 写到函数属性，绑定 group method 时再读取；`materialize_futures` 则保留在 wrapper 闭包里，控制远端函数体执行前是否调用 `.get()`
（`verl/single_controller/base/decorator.py:383-442`）。

常见语义可以分成三类：

| 形状 | 适用输入 | collect 后的含义 |
|---|---|---|
| one-to-all | 配置、控制信号 | 每个 rank 收到同一对象 |
| DP compute | 按 batch 维可切的数据 | 每个 DP rank 处理自己的 shard，再按 batch 语义聚合 |
| Megatron/mesh compute | 只在 replica master 接收 controller shard | replica 内部再由模型并行代码同步 |

`make_nd_compute_dataproto_dispatch_fn` 根据 worker 注册的 mesh dispatch 信息构造 N-D 数据并行切分
（`verl/single_controller/base/decorator.py:300-332`）。这说明“均匀切给所有 Ray actors”不是通用保证：
实际切分必须服从 worker 声明的 DP mesh，而 TP/PP/CP rank 可能共享同一 controller shard。

`DataProto` 如何检查、切分和拼接由 [[12_verl_dataproto_analysis]] 负责；本页只说明谁调用这些操作。

## 4. Ray backend：资源声明到活 actor

`RayResourcePool` 把每节点进程数翻译成 placement group bundles；`ResourcePoolManager` 根据 role mapping
创建和查询 pool（`verl/single_controller/ray/base.py:113-226`）。`RayWorkerGroup` 有两种主要来源：

- 从 resource pool 拉起新 workers，分配 global/local rank、master address 和环境；
- 从已有 handles 组装 group，例如 CheckpointEngine 每次 publication 会话临时包住 rollout CE workers。

对应构造与拉起路径位于 `verl/single_controller/ray/base.py:418-682`。`spawn`、`spawn_fused` 和 `fuse`
把多个逻辑角色暴露成不同前缀的方法，而不是启动互不相关的重复 actor
（`verl/single_controller/ray/base.py:714-776`）。V1 `_setup()` 用
`create_colocated_worker_cls` 把同 pool 的 actor/ref/critic 组合到一组 Ray actors
（`verl/trainer/ppo/v1/trainer_base.py:235-306`）。

colocation 只表示共享 Ray actor/设备资源，不表示业务状态自动一致。不同角色仍各自持有 Engine；切换
train/eval、sleep/wake 和更新权重都需要上层明确调用。

它的设计目标是让同一 resource pool 上的多个逻辑角色复用 actor 与 placement，而不是复制一套常驻进程/设备占用；`spawn_fused` 和前缀方法保留了角色可寻址性。`【分析推断】` 若把 colocation 写成“状态共享”，上层就可能漏掉 offload、模式切换或权重发布；共享地址空间只减少资源与 actor 数量，不会自动给多个 Engine 建立一致性协议。

## 5. 同步、异步与失败边界

Ray group 暴露 rank-zero 与 all-worker 的 sync/async 执行入口（`verl/single_controller/ray/base.py:778-862`）。
`blocking=False` 可以把 refs 包成 `DataProtoFuture`，但 future 只是延迟 `ray.get` 与 collect，不会建立
跨 RPC 的 commit protocol（`verl/protocol.py:1171-1226`）。

### 5.1 `blocking=False` 到底把哪一步推迟了

`TrainingWorker.train_mini_batch` 是一个具体的非阻塞方法：它用 actor/critic 的 train mesh dispatch，同时把 `blocking=False` 写进注册信息（`verl/workers/engine_workers.py:241-242`）。以 V1 critic 更新为例：

```text
PPOTrainer._update_critic
  → critic_wg.train_mini_batch(KVBatchMeta)
      → dispatch_fn(...)                                  # 已完成
      → execute_all_async(...)
          → worker.train_mini_batch.remote(...)            # RPC 已提交并可能正在改参数
      → skip ray.get because blocking=False
      → collect_fn(list[ObjectRef])
          → BatchData.concat(ObjectRefs)
          → DataProtoFuture(collect_fn, futures)
  ← DataProtoFuture
  → output.get()
      → ray.get(futures)                                  # 第一处真正等待
      → concat TensorDict results
  → reduce critic metrics
```

`func_generator` 即使不阻塞也会先调用 execute；它只跳过第 55–56 行的 `ray.get`，随后 collect 把 ObjectRefs 识别为 `DataProtoFuture`（`verl/single_controller/ray/base.py:49-64`；`verl/single_controller/base/decorator.py:138-145,191-199`；`verl/protocol.py:1288-1302`）。当前 `_update_critic` 在下一行就显式 `get()`，第一处真实等待位于 `DataProtoFuture.get → ray.get(self.futures)`（`verl/trainer/ppo/v1/trainer_base.py:1711-1732`；`verl/protocol.py:1209-1225`），所以这条具体路径没有把 critic update 与后续工作重叠起来。

future 也可以直接作为另一个已注册 worker method 的输入；默认 `materialize_futures=True` 会在远端函数体前调用 `_materialize_futures`（`verl/single_controller/base/decorator.py:383-395,428-437`）。这使 controller 可以不把实际 TensorDict 拉回本进程，但不改变三个事实：RPC 在 future 创建前已经提交；远端 optimizer side effect 可能已经发生；消费端第一次 `.get()`/materialize 仍会暴露上游异常。

| 事件 | 已发生 | 尚未发生/不保证 |
|---|---|---|
| group method 返回 `DataProtoFuture` | 参数已 dispatch，所有 remote calls 已提交 | 结果已成功、collect 已完成 |
| 某个 worker ref resolve | 该 rank 的方法已返回 | 其它 rank 成功；SPMD 结果可拼接 |
| `DataProtoFuture.get()` 返回 | 所有 refs 已 `ray.get`，结果已按 batch 语义聚合 | 跨 RPC 原子提交或失败回滚 |
| 下游 wrapper materialize 完成 | future 的目标 shard 已可供函数体使用 | 上游参数副作用可撤销 |

需要守住四个边界：

1. **dispatch 对齐**：可切输入的 batch 维必须一致，否则问题在发 RPC 前就已产生；
2. **SPMD lockstep**：mesh 内 collective 要求所有参与 rank 走同一调用序列；controller 的异常分支可能导致挂起；
3. **partial failure**：一部分 Ray actor 已执行、一部分失败时，group wrapper 没有通用 rollback；
4. **liveness 不等于 correctness**：worker aliveness check 只能发现 actor 死亡，不能证明参数、KV 或 batch 已一致。

Trainer、CheckpointEngine 和 rollout manager 必须分别定义自己的失败恢复；不能把 group method 返回 future 写成“远端已完成”，也不能把 future materialize 写成“整个 RL step 已提交”。例如训练 RPC 在多个 rank 上部分执行后失败时，single-controller 只传播异常，没有通用 optimizer rollback；是否可重试必须由训练/恢复状态机另行定义。

## 6. 当前调用定位

| 当前调用方 | 使用 single-controller 做什么 | 机制 owner |
|---|---|---|
| V1 trainer | 建资源池、colocate 角色、调用 actor/ref/critic workers | [[10_verl_end_to_end_iteration_analysis]]、[[17_verl_v1_async_trainer_analysis]] |
| V0 trainer | driver-centric DataProto RPC lifecycle | [[20_verl_ray_trainer_analysis]] |
| Worker/Engine | 把模型计算方法暴露为 group RPC | [[13_verl_workers_engine_analysis]] |
| LLM server manager | 复用 hybrid worker group 或独立 rollout pool | [[14_verl_rollout_runtime_analysis]] |
| CheckpointEngine | 组 actor/rollout CE workers，执行 publication session | [[21_verl_weight_publication_analysis]] |

## Related Pages

- [[01_verl_architecture_overview_analysis]] —— 当前各状态 owner 与模式路由总览。
- [[12_verl_dataproto_analysis]] —— dispatch/collect 可能承载的本地批容器及其不变量。
- [[13_verl_workers_engine_analysis]] —— group RPC 进入模型计算后的 Worker/Engine 边界。
- [[16_verl_v1_transfer_queue_analysis]] —— V1 controller 不再搬完整 batch 的数据面。
- [[20_verl_ray_trainer_analysis]] —— 使用该基座的 V0 legacy driver lifecycle。
- [[21_verl_weight_publication_analysis]] —— 临时 CE worker group 的权重发布状态机。
