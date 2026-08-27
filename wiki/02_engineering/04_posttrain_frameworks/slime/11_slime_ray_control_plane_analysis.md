---
title: "slime Ray 控制面分析：按职责边界编排训练与推理"
---

# slime Ray 控制面分析：按职责边界编排训练与推理

> **源码基线**：slime `main@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`
> **文档与测试基线**：同一提交下 `docs/en/` 与 `tests/`
> **Ray 语义参考**：[Ray Actors](https://docs.ray.io/en/latest/ray-core/actors.html) 与 [Placement Groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html)，核验于 2026-08-19
> **核验日期**：2026-08-19 · **系列**：[[02_engineering/04_posttrain_frameworks/slime/index|slime 源码分析]]
> **结论先行**：slime 没有把分布式后训练塞进一个“总协调器”，也没有把每个概念都实现成 Ray actor。它按资源、训练角色、模型服务、同构推理引擎组、服务进程和权重传输拆分职责：placement group 预留并排序 GPU 资源，`RayTrainGroup` 向训练 rank 广播调用，`RolloutManager` 管理生成侧控制状态，`RolloutServer`/`ServerGroup` 描述服务拓扑，`SGLangEngine` actor 管理服务进程，trainer actor 执行训练，其内部 weight updater 负责提交权重。分层增加了委托链和 RPC，但避免把不同生命周期、故障域和并行语义绑在同一个对象中。

本文把带 fixed-commit 定位符的源码/官方文档事实与“设计分析”分开；后者是根据对象边界、调用方向和失败路径作出的推断，不代表项目作者原话。

## 1. 问题不是“怎么调用 Ray”，而是各组件分别负责什么

训练 rank、rollout 服务和主控进程面临三套不同约束：Megatron 要求每个 rank 对应一个长期运行的分布式进程，并持有模型与优化器；SGLang 要长期监听请求、维护 KV cache，还可能按 prefill/decode/encoder 拆分拓扑；主控进程只应推进训练阶段，不应承载所有服务内部状态。源码因此先创建资源布局，再创建不占 GPU 的 Manager，最后创建训练 rank；同步主循环只持有 Manager 句柄和两个训练组封装对象。[`train.py:13-27`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/train.py#L13-L27) [`placement_group.py:227-253`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/placement_group.py#L227-L253)

这形成四类不能混为一谈的状态归属：

| 不变量 | 长期责任主体 | 若交给一个总对象的具体风险 |
|---|---|---|
| 资源位置 | placement group 及其有序 bundle/GPU 映射 | 逻辑 rank、physical GPU 与 colocate 区间在各模块重复推导 |
| 角色状态 | 每个 trainer actor 的模型、optimizer、并行进程组 | 一个进程无法代替 Megatron 的逐 rank SPMD 身份 |
| 生成状态 | Manager 的 DataSource、rollout 函数、服务注册表、健康监控 | 训练 rank 被迫理解请求、数据恢复与服务故障 |
| 服务状态 | router 后的 SGLang 进程、KV cache、权重版本 | 主控进程或 Manager 变成实际推理执行器，故障与资源占用耦合 |

`MegatronTrainRayActor.init` 在各 rank 内初始化模型、optimizer 和训练并行配置；`RolloutManager.__init__` 则加载 DataSource、rollout/eval 函数和转换 hooks，并创建 engine lock。两者的状态根本不同。[`actor.py:79-129`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/actor.py#L79-L129) [`rollout.py:468-515`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L468-L515)

> **设计分析**：这里的关键不是“多建几个类”，而是让资源布局、控制状态、服务存活状态和计算状态分别只有一个权威责任主体。调用另一个对象的方法只是**委托**；持有可恢复状态并决定其生命周期，才表示该对象真正负责这部分状态。

## 2. 九个概念不在同一抽象层

本文用三个层面区分职责：**控制面**决定资源放置、拓扑、阶段和生命周期；**服务面**提供长期可寻址的请求端点；**数据面**执行训练/推理并移动 token、tensor 或权重字节。一个对象可以位于交界处，但仍应明确其主要职责。

```mermaid
flowchart TB
    D["主控进程<br/>推进训练阶段"]
    PG["PlacementGroup<br/>有序 GPU 资源槽位"]
    TG["RayTrainGroup<br/>训练角色扇出器"]
    RM["RolloutManager<br/>生成状态责任主体"]
    RS["RolloutServer<br/>单模型服务视图"]
    SG["ServerGroup<br/>同构 engine 组"]
    TA["trainer Ray actors<br/>每 rank 一个训练进程"]
    EA["SGLangEngine Ray actors<br/>服务进程控制壳"]
    RT["router 与 HTTP server<br/>长期服务端点"]
    WU["weight updater<br/>权重提交器"]
    D --> PG
    D --> TG --> TA
    D --> RM --> RS --> SG --> EA --> RT
    PG -.->|提供有序 bundle| TA
    PG -.->|提供有序 bundle| EA
    TA --> WU
    WU -->|权重更新 RPC| EA
    RM -->|返回 handles 与 lock| WU
```

图中实线表示创建、持有或主要委托，虚线只表示资源约束；placement group 不调用训练或推理。对象的准确身份如下。

| 概念 | 对象身份与负责的状态 | 何时存在/参与，向谁发调用 | 主要层面 |
|---|---|---|---|
| placement group | Ray 原生资源对象；slime 另保存重排后的 bundle indices 与 physical GPU ids | 最先创建，整个任务存活；被训练 actor 与 engine actor 的 scheduling strategy 引用，不发业务调用 | 控制面 |
| `RayTrainGroup` | 主控进程内的普通 Python 封装对象；持有一个 actor 或 critic 角色的全部 rank handles 与角色参数 | training models 创建后存在；向所有 trainer actors 广播 `init/train/save/update/sleep/wake` | 控制面 |
| `RolloutManager` | 单个 `num_gpus=0` Ray actor；持有 DataSource、rollout 钩子、servers、engine lock 和 monitor | 服务启动前创建并贯穿任务；向服务聚合对象委托生命周期管理，向 rollout 函数委托生成 | 控制面 |
| `RolloutServer` | Manager 内普通 dataclass；一个模型、一个 router、若干 groups 与 `update_weights` 标记 | 模型服务装配时创建；聚合 `recover/offload/onload`，暴露 engine handles | 控制面中的服务视图 |
| `ServerGroup` | `RolloutServer` 内普通 dataclass；同构 engines、worker type、offset、并行覆盖、offload 标记 | 拓扑装配与恢复时参与；创建 engine actors，向组内 node-0 engines 扇出显存生命周期 RPC | 控制面中的拓扑单元 |
| `SGLangEngine` | 动态 `ray.remote` 包装的 Ray actor；持有 rank、端口、base GPU id、服务参数和子进程句柄 | group 启动/恢复时创建；spawn 或接入 SGLang server，并接收权重、cache、pause/continue RPC | 控制面到服务面的适配器 |
| Ray actor | 本页不是一个额外组件，而是远程进程/handle 形态；Manager、trainer、engine 都可以采用这种形态，但职责由具体实例决定 | 需要进程隔离、GPU 绑定或远程 RPC 时使用；`RolloutServer`/`ServerGroup` 并不是 actors | 不是职责层面，而是运行时原语 |
| trainer actor | 每个 Megatron world rank 一个 `MegatronTrainRayActor`；持有 rank-local 模型、optimizer、scheduler、backups 与 updater | 训练模型创建后存在或在 release 模式重建；执行 train，并向 Manager 查询 engines、向 updater 委托同步 | 数据面，带控制端点 |
| weight updater | trainer actor 内的普通对象；持有传输实现、版本、bucket/连接状态和 engine handles | trainer 初始化时按 mode/transport 选择；权重提交时 pause/flush、传输、continue | 控制/数据交界 |

这些身份由创建点而不是名称决定：`RayTrainGroup` 在循环中逐 rank 创建 `TrainRayActor`；`ServerGroup.start_engines` 才把 `SGLangEngine` 包成 Ray actor；`RolloutServer` 和 `ServerGroup` 的定义只是 dataclass。[`actor_group.py:57-129`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/actor_group.py#L57-L129) [`rollout.py:144-219`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L144-L219) [`rollout.py:320-338`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L320-L338)

`SGLangEngine` 也不是 token decoding engine 本体：普通路径在 actor 内 spawn SGLang HTTP server process，等待健康后才注册到 router；external 路径只核验已存在服务参数并注册。[`sglang_engine.py:48-81`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/sglang_utils/sglang_engine.py#L48-L81) [`sglang_engine.py:122-192`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/sglang_utils/sglang_engine.py#L122-L192)

### 2.1 一个训练角色为何对应多个 Ray actors

先纠正一个容易造成后续误读的术语：SPMD 是 **single program, multiple data**，意为多个进程执行同一套程序、各自处理不同数据或模型分片；它不是“single process, multiple data”。在常见的一卡一进程训练中，一个逻辑训练任务本来就由多个 OS 进程组成，每个进程拥有一个 distributed rank。DP、TP、PP、CP、EP 决定这些 rank 如何分工，Ray 并不替 Megatron 实现这些并行算法。

slime 的对象映射如下：

```mermaid
flowchart TB
    D["主控进程中的 RayTrainGroup<br/>普通 Python 对象"]
    A0["TrainRayActor<br/>进程 / rank 0"]
    A1["TrainRayActor<br/>进程 / rank 1"]
    AN["TrainRayActor<br/>进程 / rank N-1"]
    PG["同一个 torch.distributed process group<br/>Megatron TP/PP/DP/CP/EP"]
    D -->|"train.remote()"| A0
    D -->|"train.remote()"| A1
    D -->|"train.remote()"| AN
    A0 <--> PG
    A1 <--> PG
    AN <--> PG
```

`RayTrainGroup` 先计算 `world_size = num_nodes × num_gpus_per_node`，随后按 rank 循环创建同样数量的 `TrainRayActor`；rank 0 提供 rendezvous 地址和端口，其余 actors 使用同一组参数加入训练 world。[`actor_group.py:57-62`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/actor_group.py#L57-L62) [`actor_group.py:113-129`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/actor_group.py#L113-L129) 每个 actor 进程设置自己的 `RANK`、`WORLD_SIZE`、`LOCAL_RANK`、`MASTER_ADDR` 和 `MASTER_PORT`，再调用 `torch.distributed.init_process_group()`；至此 Ray actor 的进程身份才变成 Megatron 可使用的 distributed rank。[`train_actor.py:28-70`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/train_actor.py#L28-L70)

因此，这里的对应关系不是“一个训练任务 = 一个 Ray actor”，而是：

| 观察粒度 | 在 slime 中是什么 | 负责什么 |
|---|---|---|
| 逻辑训练角色 | 一个 `RayTrainGroup`，例如 actor 或 critic | 保存全部 rank handles，统一发起一轮训练、保存或权重更新 |
| 训练进程 | 一个 `TrainRayActor` | 承载一个 distributed rank 的长期状态，并绑定 Ray 分配的资源槽位 |
| 分布式计算组 | 全部 trainer actors 加入的 `torch.distributed` process group | 执行 Megatron 的 collective、流水线通信和参数/梯度同步 |
| GPU 资源布局 | placement group 的 bundles | 决定这些进程能否成组启动以及分别落到哪个资源槽位 |

例如 2 个节点、每节点 4 张 GPU 时，训练 `world_size` 是 8，slime 会创建 8 个 `TrainRayActor`。假设 Megatron 配置为 TP=2、PP=2、DP=2，这 8 个 actors 仍属于同一个逻辑 actor 训练角色；Megatron 再把它们组织进不同的 TP、PP、DP process groups。这里不是“一个 Ray actor 同时处理 8 份数据”，而是“8 个 Ray actor 进程执行同一训练程序，各自以不同 rank 处理其数据或模型分片”。

Ray actor 在这里是**进程容器和远程控制端点**，distributed rank 是**集合通信身份**；两者通常一一对应，但属于不同系统的概念。某个 rank 最终是 DP rank、TP rank 还是 PP rank，由 Megatron 在 process group 内建立的并行拓扑决定，不由 Ray actor 本身决定。

## 3. 启动时间线：先确定资源布局，再初始化各组件状态

```mermaid
sequenceDiagram
    participant D as 主控进程
    participant PG as placement group
    participant RM as RolloutManager
    participant SG as server groups
    participant EA as engine actors
    participant TA as trainer actors
    D->>PG: 创建 bundles 并探测 physical GPU
    D->>RM: 创建零 GPU actor
    RM->>SG: 按模型与 worker type 建组
    SG->>EA: 分配 bundle 端口并调用 init
    EA->>EA: 拉起服务进程并注册 router
    RM->>RM: 等待 engine 健康后完成初始化
    D->>TA: 逐 rank 创建并调用 init
    TA->>RM: rank 0 回报训练并行配置
    D->>TA: 首次 update weights
```

### 3.1 Placement group 只预留资源，不管理 Actor

每张 GPU 对应一个 `{GPU:1, CPU:1}` bundle；Ray 排出的 bundle index 不被假定为 physical 拓扑顺序，因此临时 `InfoActor` 读取 node/GPU id，slime 再按 node、GPU 排序，得到 logical index 到 bundle/GPU 的稳定映射。[`placement_group.py:42-97`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/placement_group.py#L42-L97)

布局函数返回的是“资源池总量 + rollout 起点”，下游再把逻辑位置分配给具体 actor：

| 模式 | placement group GPU 数 | rollout offset | 资源含义 |
|---|---:|---:|---|
| train-only | actor GPUs | 0 | 无本地 rollout engine |
| rollout-only | rollout GPUs | 0 | 无 trainer actor |
| colocate | `max(actor, rollout)` | 0 | 训练与 rollout 的前缀 GPU 区间重叠 |
| disaggregate | actor + rollout GPUs | actor GPUs | 两个连续且不重叠的区间 |
| external rollout | actor GPUs | actor GPUs | serving 不占本任务的本地 rollout bundles |

源码分支和单测同时钉住这些返回值，包括 colocate 下 rollout GPU 少于、等于或多于 actor，以及 rollout GPU 为零的路径。[`placement_group.py:100-137`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/placement_group.py#L100-L137) [`tests/test_placement_group.py:30-50`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/tests/test_placement_group.py#L30-L50)

### 3.2 Manager 先于 trainer 创建，因为它负责生成侧装配

主控进程先创建 Manager，是因为 `num_rollout_per_epoch` 可能需要根据其 DataSource 计算；Manager 内部先启动路由器、服务组和推理引擎，并等待尚未完成的 `engine.init` 引用，再创建指标跟踪、锁和监控线程。之后 `create_training_models` 才建立 actor/critic 训练组，并在 rank 初始化后把 Manager 句柄下发给训练 actor。[`placement_group.py:227-253`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/placement_group.py#L227-L253) [`rollout.py:474-515`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L474-L515) [`actor_group.py:188-215`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/actor_group.py#L188-L215)

### 3.3 Server 与 group 分层，因为模型级配置和同构拓扑不是一回事

一个 `RolloutServer` 对应一个模型和一个 router，但可含多个 `ServerGroup`；group 才统一 `worker_type`、GPU 数、engine 大小和 SGLang overrides。官方配置文档也明确规定“每模型一个 router”“模型内 groups 可异构”“权重更新按模型选择”。[`docs/en/advanced/sglang-config.md:17-21`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/en/advanced/sglang-config.md#L17-L21) [`rollout.py:1132-1212`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L1132-L1212)

因此 server 适合回答“哪个模型、哪个 router、是否接收训练权重”，group 适合回答“这一批 engines 是 prefill、decode、regular、encoder 还是 placeholder，用几张卡，是否与训练重叠”。placeholder 甚至只推进 GPU offset 而不创建 engine，证明 group 首先是拓扑单元，不是服务进程的别名。[`sglang_config.py:11-40`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/sglang_utils/sglang_config.py#L11-L40) [`rollout.py:188-217`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L188-L217)

## 4. 每轮时间线：主控进程编排阶段，各组件处理本地状态

同步路径的真实顺序是生成 → 必要时释放 rollout 显存 → critic/actor 训练 → 保存/释放训练内存 → 恢复 rollout weights → 提交新权重 → 恢复 KV cache。主控进程决定阶段顺序，但不接管任何 rank 内 optimizer、DataSource 或 KV cache。[`train.py:48-91`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/train.py#L48-L91)

```mermaid
sequenceDiagram
    participant D as 主控进程
    participant RM as RolloutManager
    participant TG as RayTrainGroup
    participant TA as trainer actors
    participant WU as weight updaters
    participant EA as engine actors
    D->>RM: generate rollout id
    RM->>RM: 调 rollout 函数并转换训练数据
    RM-->>D: 返回按 DP 切分的数据引用
    D->>RM: 必要时 offload rollout
    D->>TG: async train
    TG->>TA: 向全部 ranks 扇出 train
    TA->>TA: Megatron forward backward update
    D->>TG: update weights
    TG->>TA: 向全部 ranks 扇出 update
    TA->>RM: 查询 updatable engines 与 lock
    TA->>WU: 委托提交新版本
    WU->>EA: pause flush transfer continue
    D->>RM: 恢复 KV cache
```

`RolloutManager.generate` 设置 rollout id、恢复 health monitoring、调用 rollout 函数、记录与转换 samples，最后做 DP split；训练计算则由 group 对每个 rank 调 `train.remote`，trainer actor 在本进程取数据后分派 actor/critic 路径。[`rollout.py:590-604`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L590-L604) [`actor_group.py:131-149`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/actor_group.py#L131-L149) [`actor.py:374-422`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/actor.py#L374-L422)

异步入口没有删除版本边界，而是先发下一轮 `generate.remote` 使其与当前训练重叠；到更新间隔时，它先等待 pending generation 完成，再执行权重更新，且入口直接拒绝 colocate。[`train_async.py:9-11`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/train_async.py#L9-L11) [`train_async.py:31-70`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/train_async.py#L31-L70)

### 4.1 Ray 管的是进程调度与 RPC 并发，不是 Megatron 的并行算法

slime 中至少有四种“并发”，不能只用一个 actor 数量来理解：

| 并发层次 | Ray/slime 如何实现 | 等待点与边界 |
|---|---|---|
| 资源成组调度 | placement group 先按每卡一个 bundle 预留整组 GPU/CPU，再把各 actor 绑定到指定 bundle | 资源组未 ready 时不创建后续训练 world；这是 gang scheduling，不是训练计算 |
| 不同 actors 间的并行 | `RayTrainGroup.async_train()` 对每个 rank 分别调用 `train.remote()`，先收集全部 `ObjectRef` | 主控进程随后 `ray.get(refs)`，把“一轮所有 ranks 完成”设为阶段屏障 |
| 同一个 actor 内的方法顺序 | 本页涉及的 trainer 和 Manager 都是同步 Ray actors；按 Ray 默认语义，同一 actor 的方法调用排队执行，不同 actors 的方法才可并行 | Manager 正在执行 `generate()` 时，发给同一 Manager 的后续 RPC 不能假定会并行进入其可变状态 |
| rank 之间的计算同步 | 每个 trainer actor 进入 `train()` 后，由 Megatron/PyTorch distributed 执行 TP/PP/DP/CP/EP 通信 | collective、pipeline send/recv 和 optimizer 语义由 Megatron 保证，Ray 既不拆 batch，也不执行 all-reduce |

Ray 官方语义明确区分“不同 actors 可并行”和“同一同步 actor 的方法按调用顺序串行”；placement group 则是资源的原子预留和 actor 放置约束，而不是一个执行线程池。[Ray Actors](https://docs.ray.io/en/latest/ray-core/actors.html) [Ray Placement Groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html) slime 的训练扇出直接体现这一点：它先向所有 trainer actors 发出 RPC，再由调用方用 `ray.get` 等齐，因此 Ray 建立的是**跨进程并发与控制屏障**，Megatron 建立的是**进程内部进入哪组集合通信**。[`actor_group.py:131-149`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/actor_group.py#L131-L149) [`train.py:61-69`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/train.py#L61-L69)

异步训练的重叠也发生在**不同 actor 集合**之间：主控进程保留下一轮 `RolloutManager.generate.remote()` 的 future，同时让 trainer actors 训练当前轮；它不是让同一个 trainer actor 并发执行两轮 `train()`。[`train_async.py:31-53`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/train_async.py#L31-L53) 同理，推理请求的 continuous batching、KV cache 调度和 token generation 并发发生在 SGLang 服务进程内部；Ray 的 `SGLangEngine` actor 主要负责放置、启动、RPC 和故障隔离，不能把 Ray actor 并发等同于推理数据面的请求并发。

## 5. 权重更新最能说明“调用发起方不等于状态责任方”

权重同步的完整机制归 [[16_slime_weight_sync_analysis]]；本页只追踪控制权如何穿过对象边界：

1. 主控进程调用 `RayTrainGroup.update_weights()`；group 只负责把 RPC 广播到所有 trainer rank，不管理参数 bucket 或 engine cache。[`actor_group.py:162-174`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/actor_group.py#L162-L174)
2. 每个 actor trainer 初始化时根据 delta/full、disk/NCCL、colocate 选择 updater 实现；updater 是 trainer actor 内对象，拥有 weight version 与传输连接状态。[`actor.py:151-182`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/actor.py#L151-L182)
3. trainer actor 向 Manager 查询第一个可更新模型的 engine handles、lock、GPU offsets 和并行配置；Manager 拥有“哪些服务可更新”的注册事实，但不发送参数。[`rollout.py:555-584`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L555-L584) [`actor.py:592-629`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/actor.py#L592-L629)
4. updater 驱动 pause → flush → 传输 → continue；distributed 路径的 metadata 走 Ray，tensor bytes 走 NCCL，engine actor 执行接收侧 RPC。[`update_weight_from_distributed.py:102-134`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L102-L134) [`update_weight_from_distributed.py:326-353`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L326-L353)

> **设计分析**：主控进程决定“何时提交”，Manager 维护“提交给谁”的服务注册信息，trainer actor 持有源模型状态，updater 负责“如何提交”的协议状态，engine 提供目标服务的权重应用入口。这条链比 Manager 直接搬运权重更长，却把版本时序、服务发现、训练分片和传输协议放到了真正掌握相应信息的组件中。

固定基线还存在一个明确边界：Manager 只返回第一个 `update_weights=True` 的模型，源码注明多 updatable models 尚未支持。因此“多模型 serving”并不意味着“多模型联合权重提交”。[`rollout.py:555-584`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L555-L584)

## 6. 资源复用与生命周期：共享 GPU 不等于共享对象

slime 有两条独立的 GPU 复用轴：colocate 让 train 与 rollout 的逻辑 GPU 区间重叠；PPO 则令 critic 与 actor 指向同一个 train placement group。官方文档明确说 actor/critic 是两个独立训练 process groups，只是轮流占用同一组 GPU。[`placement_group.py:120-137`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/placement_group.py#L120-L137) [`docs/en/get_started/usage.md:241-246`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/en/get_started/usage.md#L241-L246)

ref、Megatron OPD teacher 与 old actor 也不是各自一组 Ray actors：它们由 actor trainer 内的 `TensorBackuper` 以不同 tag 保存/切换；critic 因为有独立可训练状态才建立第二个 `RayTrainGroup`。这再次说明 group 边界对应独立训练角色，而不等于每个模型身份都起一组进程。[`actor.py:120-143`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/actor.py#L120-L143) [`placement_group.py:186-224`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/placement_group.py#L186-L224)

训练 actor 以 0.4 个逻辑 GPU 的资源声明绑定到每个包含 1 张 GPU 的 bundle。这个数值只是 Ray 用于资源准入和调度的声明，不代表 40% 的显存配额；实际显存让渡由 sleep/wake、release 或 rollout 显存占用接口完成。[`placement_group.py:140-160`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/placement_group.py#L140-L160) [`actor_group.py:107-126`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/actor_group.py#L107-L126)

| 动作 | 对象是否仍存在 | 调用链与责任主体 | 语义 |
|---|---|---|---|
| rollout `offload/onload` | engine actor 与服务进程仍在 | Manager → server → group → engine | 只对 `needs_offload` groups 释放/恢复指定显存占用 |
| train `sleep/wake` | trainer actor 仍在 | group → trainer actor | 暂停/恢复训练 GPU 状态与进程组 |
| train `release/create` | trainer actor 被 kill/重建 | `RayTrainGroup` | 以 checkpoint 换取更彻底的资源释放 |
| rollout `recover` | 死 engine actor 被替换 | Manager → server → group | 重建可更新模型的缺失服务控制 actor，供 updater 重连 |

`ServerGroup.offload/onload` 会跳过 `needs_offload=False` 的组。固定基线的恢复入口只选择可更新模型：Manager 调该 server 的 `recover` 并重建 dead engines；trainer 随后根据 `num_new_engines` 让 updater 重连。冻结模型的完整恢复边界由故障专题页讨论，本页不把“有 health monitor”误写成“所有模型都会自动恢复”。[`rollout.py:299-317`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L299-L317) [`rollout.py:384-425`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L384-L425) [`rollout.py:641-658`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L641-L658) [`actor.py:622-632`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/actor.py#L622-L632)

## 7. 为什么不选两个直观替代方案

### 7.1 单体式总协调器

> **设计分析**：若主控进程或 Manager 同时负责资源映射、所有训练 rank、DataSource、服务子进程、KV cache 与权重分桶，它就必须同时处理 Ray 调度、Megatron 集合通信、SGLang 生命周期和 rollout 数据恢复。这样，任何服务重启都会影响训练状态，任何训练 rank 阻塞也会阻塞服务管理；共置与分离部署也很难只改变资源布局而不改变对象结构。当前分层让主控进程只推进阶段，让 Manager 只掌握生成侧控制状态，具体计算仍留在原生后端。

代价是跨责任主体的操作需要显式协议：例如权重更新要经过训练组 → 训练器 → Manager → 更新器 → 推理引擎，故障恢复要经过 Manager → 服务 → 服务组 → 推理引擎。代码中的句柄、锁、偏移量和版本号正是这种解耦的必要成本，而不是可以随意删除的样板代码。

### 7.2 把整个训练角色压进一个 Ray actor

这一节比较的是 **Ray 远程进程的划分粒度**，不是“一个算法模块该有几个类”，也不是说一个训练任务只能运行一份程序。更准确的反事实方案是：actor 训练角色只创建一个 Ray actor，再让这个 actor 在内部管理全部 GPU 和 ranks。

> **设计分析**：这种方案会让 Ray 只能放置和恢复那个总进程，无法分别把每个 rank 绑定到已排序的 GPU bundle；总进程还必须再自行 spawn 多个训练进程并向它们传递 rendezvous 信息，等于在 Ray 下面重新实现一层进程管理。固定基线选择逐 rank 创建 actors，由 rank 0 提供 master address/port，再由各 actor 初始化同一个 distributed world。[`actor_group.py:113-129`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/actor_group.py#L113-L129) [`train_actor.py:41-70`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/train_actor.py#L41-L70)

反方向也不能得出“每一层对象都应该是 actor”：`RolloutServer` 和 `ServerGroup` 只是 Manager 内的模型/拓扑描述，没有独立执行循环和独立资源需求；把它们改成 actors 只会为本地聚合增加序列化与 RPC。需要远程进程身份的，是每个 trainer rank、每个可独立放置和恢复的 engine 控制进程，以及持有跨轮生成状态的 Manager。是否使用 Ray actor，取决于是否需要**独立进程、资源放置、故障边界或远程串行状态**，而不取决于它是否被称为“子系统”。

这也解释了为什么 `group/manager/server/engine` 不能互换：group 聚合同类句柄，Manager 持有跨轮生成状态，server 建立模型级服务边界，engine 对应可独立放置和恢复的服务控制进程。名称相似不代表相同生命周期。

## 8. 常见误读与边界

| 误读 | 固定基线的实际行为 |
|---|---|
| placement group 是一组训练 actors | 它只预留和排序 resources；actors 后续绑定 bundles |
| `RayTrainGroup` 自己是 Ray actor | 它是主控进程内的 Python 封装对象，内部才持有一组 actor 句柄 |
| `RolloutServer` 就是监听 HTTP 的 server | 它是 Manager 内模型级 dataclass；HTTP server 是 engine actor 拉起的子进程 |
| 一个 `ServerGroup` 等于一个 engine | group 可含多个同构 engines；多节点 engine 的上层控制还只暴露 node-0 handles |
| `SGLangEngine` 执行全部 decoding | 它是进程控制与 RPC 壳，实际 forward/KV/token 服务在 SGLang 进程树 |
| Ray actor 就是训练并行 | Ray 负责进程放置与 RPC；Megatron collectives 在 trainer actors 内初始化和执行 |
| Manager 拿着 engine handles，所以拥有权重同步 | Manager 拥有服务注册与 lock；trainer 内 updater 拥有传输/版本状态并驱动提交 |
| `offload`、`sleep`、`release`、`recover` 都是“释放 GPU” | 它们分别改变服务显存、训练驻留、actor 存活或故障实例，恢复成本不同 |

本页不展开 rollout 请求、Sample 转换、Megatron forward/backward 或权重 tensor 拆分；它只界定这些机制由谁拥有、何时被调用。对应细节分别由下列权威页承接。

## Related Pages

- [[10_slime_end_to_end_iteration_analysis]] — 把本文的对象所有权放回同步与异步 iteration 的版本边界。
- [[12_slime_sample_datasource_analysis]] — 深入 Manager 所有的 DataSource、Sample 与 train-data conversion 契约。
- [[13_slime_sglang_rollout_engine_analysis]] — 深入 router 后的请求调度、生成状态与 SGLang 数据面。
- [[14_slime_megatron_training_analysis]] — 深入 trainer actor 内的 Megatron 初始化、数据迭代与 forward/backward。
- [[16_slime_weight_sync_analysis]] — 深入 weight updater 的提交协议、拓扑变换与 transport 数据面。
- [[18_slime_fault_tolerance_observability_analysis]] — 深入 health monitor、engine recover 与分层故障域。
