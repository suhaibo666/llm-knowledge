---
title: "FSDP 预取、流与显存：先选通信组，再决定重叠窗口"
---

# FSDP 预取、流与显存：先选通信组，再决定重叠窗口

> **TorchTitan 代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **上游机制辅助基准**：pytorch/pytorch `trunk` @ `ea5655fcebf726ec4cf1a859de75d2d0e6425805`（2026-07-21；所引 `_fully_shard/` 子树无本地修改）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页论点**：FSDP 的“通信能否藏住”和“峰值显存是多少”不是两个独立开关，而是同一条流水的两面：TorchTitan 先按 block、首尾模块与参数 mesh 划定通信组，再选择是否保留完整参数、是否显式提前下一组；PyTorch FSDP2 才负责 hook、stream、event、collective buffer 与 storage resize。当前 TorchTitan 只在 EP 路径建立显式一跳预取链，dense 路径仍依赖隐式 CPU run-ahead；因此旧页把固定五条流、固定双缓冲和“峰值恒为两组完整参数”写成 TorchTitan 保证，是过度泛化。
>
> 本页聚焦预取、计算通信重叠和参数显存生命周期。storage mesh、mixed precision、梯度语义见 [[11_torchtitan_fsdp_analysis]]；HSDP 的第二级 all-reduce 见 [[21_torchtitan_hsdp_backward_overlap_analysis]]；GraphTrainer/SimpleFSDP 的图级 collective 调度不在此页重复。

---

## 1. Overview

参数分片把长期存储降下来，却把每层计算前的 all-gather 和计算后的 reduce-scatter 放进关键路径。最直接的实现是“等本层参数聚齐，再计算，再立刻释放”；它最省瞬时显存，却让通信完整暴露。另一极端是一次性 gather 多层甚至全模型；它容易掩盖通信，却逐步失去分片带来的峰值优势。

TorchTitan 当前没有另造一套 FSDP runtime。模型 adapter 先调用 `fully_shard()` 划分单元，并把 reshard、CPU offload、per-param mesh 和可选 symmetric-memory 策略交给上游；FSDP2 的 module hooks 再驱动 `unshard -> compute -> reshard/reduce`（`torchtitan/distributed/fsdp.py:223`、`torchtitan/distributed/fsdp.py:228`；`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:110`、`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:131`）。

核心决策可压缩成四个旋钮：

| 决策 | TorchTitan 当前选择 | 直接代价 |
|---|---|---|
| 通信组边界 | decoder block 为主；embedding、norm/head、root 另行处理 | 组过大抬高峰值，组过小增加 collective/CPU 开销 |
| 预取来源 | dense 用上游隐式预取；EP 显式串下一单元 | 显式链更抗 host 同步，但错误顺序会提前占内存或预取错目标 |
| forward 后是否 reshard | 非 PP 默认 reshard；PP 默认保留；尾部组另有特例 | reshard 省内存但 backward 需 re-all-gather |
| 通信 buffer 后端 | 默认 allocator/collective；可选 FSDP symmetric memory | 需要硬件与 NCCL 条件，且不是所有模型 adapter 都已接通 |

```text
TorchTitan model adapter
  -> bottom-up fully_shard(unit)              划定通信/驻留粒度
  -> resolve reshard policy                   决定 forward 后保留或释放
  -> EP only: set next-unit prefetch edges    决定谁提前发起 all-gather
  -> optional: set_symm_mem_for_comm          替换 staging-buffer 通信路径

PyTorch FSDP2 runtime
  pre-forward:  copy-in -> all-gather -> copy-out -> module compute
  post-forward: reshard? -> record forward order
  pre-backward: re-all-gather if needed -> prefetch previous unit
  post-backward: reshard -> reduce-scatter -> event-delayed buffer release
```

### Quick Start：从配置追到真实调用链

```python
config.parallelism.fsdp_reshard_after_forward = "default"
config.parallelism.enable_fsdp_symm_mem = False
```

配置只公开 `default/always/never` 与 symmetric-memory 开关（`torchtitan/config/configs.py:147`、`torchtitan/config/configs.py:162`）；没有“显式预取层数”这一用户开关。以 Qwen3 为例，adapter 解析 dense/sparse mesh，再把 PP、EP、reshard 与 symm-mem 参数送入公共 decoder FSDP helper（`torchtitan/models/qwen3/parallelize.py:66`、`torchtitan/models/qwen3/parallelize.py:85`、`torchtitan/models/qwen3/parallelize.py:97`）。

```text
parallelize_qwen3
  -> resolve_fsdp_mesh / resolve_sparse_fsdp_mesh
  -> apply_fsdp_to_decoder
       -> fully_shard(embedding / norm+head / each block / root)
       -> optional enable_fsdp_symm_mem
       -> disable_fsdp_gradient_division
       -> if EP: wire explicit forward/backward prefetch edges
  -> first real forward triggers FSDP lazy init + runtime hooks
```

---

## 2. 通信组边界：为什么以 block 为主，而不是整模一组

### ① 背景/问题

FSDP 的 all-gather/reduce-scatter 以参数组为单位。整模型只包一次时，所有未分组参数会同时恢复完整形态；逐个小 tensor 分组又会产生大量短 collective。上游 API 因此把 grouping 设为一等语义，并明确要求 bottom-up：先包子模块，root 只接管尚未被子组认领的参数（`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:133`、`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:143`）。

### ② 为什么这么设计

**选中路线**是“每个 TransformerBlock 一个 module boundary，首尾小模块按共享关系合并”；**明显替代方案**是整模一组或每参数一组。决定性标准是让单组计算足以覆盖下一组通信，同时不让太多完整参数共同驻留。源码没有给出跨模型的最优 group 大小公式；这是由上游明确的 memory/overlap 取舍与 TorchTitan 分组形状共同得到的**知识库推断**。

特殊模块按复用而不是统一粒度处理：小型 vision encoder 被整体视为一组，因为注释认为一次 gather 比逐层分片更高效（`torchtitan/distributed/fsdp.py:139`、`torchtitan/distributed/fsdp.py:153`）；tied embedding/output 则必须放进同一组，避免共享参数被两个组重复管理与重复 gather（`torchtitan/distributed/fsdp.py:238`、`torchtitan/distributed/fsdp.py:250`）。

### ③ 实现思路与细节

非 tied decoder 先各自包 embedding 与 `[norm, lm_head]`，再遍历 `model.layers` 为每个 block 调 `fully_shard()`，最后包 root model（`torchtitan/distributed/fsdp.py:252`、`torchtitan/distributed/fsdp.py:265`；`torchtitan/distributed/fsdp.py:267`、`torchtitan/distributed/fsdp.py:368`）。root 调用只收纳尚未属于子组的剩余参数，这是 bottom-up 规则的实际落点。

MoE block 看似仍是一次 `fully_shard(block)`，但 `shard_placement_fn` 会把 routed experts 指到 sparse mesh、其余参数指到 dense mesh（`torchtitan/distributed/fsdp.py:318`、`torchtitan/distributed/fsdp.py:359`）。上游按 shard/replicate process-group 键重新分组，因此同一 module state 可以拥有多个 `FSDPParamGroup`（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_init.py:448`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_init.py:451`；`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_init.py:478`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_init.py:525`）。

实际通信单位因此是：

| TorchTitan 单元 | 常见上游 param group 数 | 当前目的 |
|---|---:|---|
| dense block | 1 | 层级 overlap/memory 平衡 |
| EP MoE block | 可能多于 1 | expert 与 dense 参数走不同 mesh/process group |
| tied embedding+norm+head | 1 个 grouped module state | 共享参数只归一个 FSDP group |
| 小型 vision encoder | 1 | 减少过细 collective |
| root model | 只含剩余参数 | 闭合所有权，不重复接管子组参数 |

### ④ 约束/代价/失败边界

- `fully_shard([a, b, ...])` 允许 forward 分时调用组内模块，但上游说明每个 standalone chunk invocation 都可能产生自己的 reduce-scatter；grouped 不等于永远只有一次 backward collective（`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:145`、`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:157`）。
- 上游 lazy init 会拒绝同一共享参数被两个 FSDP groups 管理，并明确要求 tied modules 同组（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_state.py:217`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_state.py:229`）。
- per-param mesh 不是“同一次 collective 混两个 process group”；上游先拆成多个 param groups。故 profiler 中一个 MoE block 出现多次 AG/RS 是当前设计，不是重复包装 bug。
- 模型可偏离公共 helper。Flux 逐个包线性层、double/single blocks，并固定 final layer 不 reshard（`torchtitan/models/flux/parallelize.py:118`、`torchtitan/models/flux/parallelize.py:123`、`torchtitan/models/flux/parallelize.py:131`、`torchtitan/models/flux/parallelize.py:138`）；不能把 decoder helper 的分组图套到所有模型。

### ⑤ 发展趋势（有锚点的推断）

提交 `d92336fee` 把 dense/MoE 两份 FSDP adapter 合并到公共 helper，提交正文的理由是 MoE 路径是 dense 的严格超集。**推断**：decoder 家族会继续共享分组策略，但 Flux 等非 decoder 结构仍需要模型专属边界，而不会被强行压成统一 block 列表。

---

## 3. 预取策略：dense 隐式，EP 显式一跳

### ① 背景/问题

默认隐式预取依赖 Python/CPU 在 GPU 前面继续发出下一层 hook；一旦执行中出现 device-to-host 同步，host run-ahead 会收缩，下一组 all-gather 可能直到当前计算快结束才发出。TorchTitan 当前源码直接把 EP 内的 D2H sync 识别为这个干扰源（`torchtitan/distributed/fsdp.py:384`、`torchtitan/distributed/fsdp.py:385`）。

### ② 为什么这么设计

**选中路线**是在 `ep_degree > 1` 时显式写出相邻 FSDP units 的一跳前向/反向边；**替代方案**是所有模型都显式预取，或一次预取两层以上。上游 API 说明单目标显式预取与默认 overlap 窗口相当，只是 CPU 更早发出；两个以上目标才更激进，并增加 reserved memory（`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:513`、`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:523`）。因此当前判据是“只为已知会破坏 run-ahead 的 EP 支付显式维护成本”，而不是把窗口无条件加深。

历史提交 `a54725cfab` 的标题即 “MoE explicit prefetching in FSDP”，正文称前后向均在 profiler 中恢复预期预取。这提供了选择显式 EP 链的演进证据；当前机制断言仍以下述现行代码为准。

### ③ 实现思路与细节

forward 链按执行方向连接：embedding 预取首 block，每个 block 预取下一 block，末 block 预取 `[norm, lm_head]`（`torchtitan/distributed/fsdp.py:389`、`torchtitan/distributed/fsdp.py:406`）。backward 链反向连接：`lm_head` 预取末 block，每个 reversed block 预取前一 block，最前 block 预取 embedding（`torchtitan/distributed/fsdp.py:408`、`torchtitan/distributed/fsdp.py:424`）。

上游保存这些 target states；forward 在当前 state 完成自身 pre-forward 后发出 targets，backward 则覆盖默认 reverse-post-forward prefetch（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_state.py:306`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_state.py:324`；`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_state.py:385`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_state.py:397`）。真正的 prefetch 仍只是给目标 param group 调 `unshard()`；目标随后进入 hook 时，该调用成为 no-op/pending wait（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:859`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:874`；`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:382`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:386`）。

### ④ 约束/代价/失败边界

- `ep_degree == 1` 直接返回，不建立任何显式边；dense 训练继续依赖上游默认机制（`torchtitan/distributed/fsdp.py:384`、`torchtitan/distributed/fsdp.py:387`）。
- 当前链来自 `model.layers.values()` 的顺序并假设 forward/backward 遵循相邻 block 顺序。动态跳层或模型自定义执行顺序需要重审 target graph；源码没有自动控制流分析。
- optional embedding/norm/head 和空 layer 集合都有 guard（`torchtitan/distributed/fsdp.py:393`、`torchtitan/distributed/fsdp.py:413`）；PP 裁剪后的 stage 因而不会盲目连接已删除模块。
- 当前公开配置没有控制显式窗口深度；若直接改成多目标列表，会增加 reserved memory，且必须以 profiler 与峰值测量验证，而不能只依据“发得更早”。
- 集成矩阵覆盖 DeepSeek FSDP+EP、FSDP+CP+PP+EP、HSDP+EP，以及 GPT-OSS FSDP+TP+EP（`tests/integration_tests/models.py:48`、`tests/integration_tests/models.py:70`；`tests/integration_tests/models.py:143`、`tests/integration_tests/models.py:152`），但这些是端到端数值/运行覆盖，不是预取时序断言测试。

### ⑤ 发展趋势（有锚点的推断）

当前上游默认 backward prefetch 已能处理一个 module 内多个 param groups，并按 forward 记录逆序找目标（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:821`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:856`）。**推断**：TorchTitan 保留显式 EP 链的价值主要是提前 host 发射，而不是修补“上游不认识多 param group”；未来若 EP 消除 D2H，同一条显式链是否仍获益需要重新 profile。

---

## 4. Stream 与 event：重叠由上游执行，TorchTitan 不拥有五流状态机

### ① 背景/问题

若 copy-in、all-gather、模型计算和 reduce-scatter 全进默认流，同流 FIFO 会把通信与计算串行化；若只把 collective 丢到别的流而不保留 buffer/event，又会出现生产者尚未完成、消费者或 allocator 已复用内存的竞态。

### ② 为什么这么设计

**选中路线**是上游共享通信上下文拥有独立高优先级 copy-in、all-gather、reduce-scatter 流，HSDP 再有普通优先级 all-reduce 流；**明显替代方案**是所有工作留在当前流。决定性标准是让“下一组参数通信、本组计算、上一组梯度规约”有并行执行线程，同时用 event 保证跨流生命周期（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:80`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:107`）。

上游也保留相反取舍：`_set_unshard_async_op(True)` 把 all-gather 分配放回默认流以减少跨流碎片，但要求显式预取，而且 dtype cast/copy-in 不再与计算重叠（`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:862`、`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:872`）。TorchTitan 当前 adapter 没有调用这个私有开关。

### ③ 实现思路与细节

默认 `unshard_async_op=False` 且处于 forward/pre-backward 时，FSDP 选择独立 copy-in/all-gather streams；其他情形回退 current stream（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:117`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:127`）。

一次 all-gather 的上游顺序是：

1. copy-in stream 收集参数分片、分配 staging output，并把本 rank 输入拷进 output 对应 slice（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:324`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:360`）。
2. all-gather stream 等 copy-in stream，再发 collective 并记录 event（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:362`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:377`）。
3. 当前 module 的 pre-forward 调 `unshard()` 与 `wait_for_unshard()`，随后才把完整参数交给计算（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:550`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:568`）。
4. backward 完成后，完整梯度交给独立 reduce-scatter stream；FSDP 保存输入 buffer 与 event，最后由 root callback 等待并释放（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:703`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:743`；`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_state.py:424`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_state.py:437`）。

### ④ 约束/代价/失败边界

- “有独立 stream”只提供重叠机会，不保证 collective 完全被藏住。首个 all-gather 没有前一层计算可遮挡；单层计算短于通信或多个 collective 争同一网络资源时仍会露尾。这是从依赖图得到的**知识库推断**，源码没有吞吐保证。
- HSDP all-reduce 能与 AG/RS 并发的前提是典型的节点内 shard、节点间 replicate 使用不同网络资源；上游注释把它写成 typical case，而非普适保证（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:100`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:103`）。
- forward 返回 view 后若用户做 in-place op，可能丢失 pre-backward hook，导致跳过 all-gather；上游现在显式告警要求 out-of-place 或 clone（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_state.py:440`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_state.py:454`）。
- TorchTitan 代码只声明 targets 与 policies；stream 优先级、event 和 buffer keepalive 都属于所用 PyTorch nightly。仓库自身承认依赖 moving nightly、不同环境可能每周漂移（`torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py:83`、`torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py:92`），所以本页给上游机制单列辅助 baseline。

### ⑤ 发展趋势（有锚点的推断）

上游 `_fully_shard` 已加入 reduce-scatter 输入 buffer 的可配置保留上限，并明确把它定义为 memory/overlap tradeoff（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:225`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:230`）。**推断**：未来 overlap 调优会继续从“固定几条流”转向“每种 buffer 各自控制在途窗口”；TorchTitan 当前没有暴露该上限。

---

## 5. 参数显存生命周期：不能再用“恒定 2p”作容量保证

### ① 背景/问题

峰值并不等于长期 sharded 参数大小。all-gather 需要 staging output，copy-out/参数视图需要可计算 storage，显式或隐式预取还会让相邻 groups 同时在途。旧页用单一等尺寸 block、单目标 implicit forward 推导出“两份完整参数 buffer”，却把这个局部稳态结论推广到 aggressive prefetch、MoE 多 param groups、RS buffers、CPU offload 与自定义通信后端；这不是当前 API 保证。

### ② 为什么这么设计

**选中路线**是按事件延迟释放必要 buffer，并用 `storage.resize_(0/full)` 复用逐参数 tensor 对象；**替代方案**是每次 CPU 同步后立即 free，或为全模型预留固定持久池。决定性标准是不能为了回收一块显存而破坏跨流重叠/正确性，同时又不让所有完整参数常驻。上游把 `all_gather_state` 与 `reduce_scatter_states` 明确定义为跨流 tensor 引用和同步 event（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:104`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:113`）。

### ③ 实现思路与细节

| 生命周期对象 | 当前上游行为 | 对峰值的意义 |
|---|---|---|
| sharded parameter | 长期注册为 DTensor shard | 基础常驻项 |
| flat all-gather output | collective backend `allocate()` 创建 staging output | 每个在途 group 都可能有一份 |
| per-param all-gather outputs | tensor 对象只初始化一次，之后恢复/释放 storage | 对象复用不等于 storage 永久常驻 |
| unsharded parameter | 默认是 all-gather output 的 `as_strided` 视图 | 参数对象本身不再复制一份数据 |
| reduce-scatter input | event 完成前由 `reduce_scatter_states` 持有 | backward 峰值还含梯度规约窗口 |

staging output 的分配发生在 copy-in stream，大小由当前 param group 的输入总量乘 world size 决定（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:337`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:353`）。逐参数 outputs 首次建立后直接早退复用对象（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param.py:831`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param.py:843`）；unsharded parameter 是其 `as_strided` view（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param.py:873`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param.py:895`）。reshard 时释放这些 outputs 的 storage，底层实现是 resize 到零（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param.py:1040`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param.py:1048`；`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param.py:1300`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param.py:1308`）。

implicit forward 为让本组 copy-out 与下一组 all-gather 重叠，会把当前 result 延迟到下一次 wait 再释放；其他情形在当前 copy-out event 排序后释放（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:427`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:442`；`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:486`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:499`）。这解释了常见一跳 ping-pong，但不构成跨所有策略的固定峰值上界。

### ④ 约束/代价/失败边界

- allocator 的 active、reserved 与逻辑 live tensors 不是同一个量。`resize_(0)` 结束逻辑 storage 占用，不保证 caching allocator 立即把 reserved bytes 还给系统。
- 显式 target list 可包含多个 modules；上游明确说长度至少 2 会使用更多 reserved memory（`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:519`、`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:523`）。因此容量规划必须用实际 group sizes 与 prefetch window，而不是统一的 `2p`。
- MoE 一个 block 可能包含多个 param groups，且每组大小、mesh/world size 不同；“一层参数大小”已不是单一 p。
- CPU offload 只通过 `CPUOffloadPolicy` 进入 FSDP config（`torchtitan/distributed/fsdp.py:228`、`torchtitan/distributed/fsdp.py:232`）；它改变基础驻留与传输，但当前 TorchTitan adapter 没有提供本页旧文所设想的持久 ping-pong pool。

### ⑤ 发展趋势（有锚点的推断）

上游通信接口已经允许 collective backend 自己 `allocate()`，symmetric-memory 路径也正是替换 staging allocation/collective，而不是在 TorchTitan 里手写持久池（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:351`、`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:828`）。**推断**：内存优化方向更可能继续下沉到 ProcessGroup/collective allocator，而非在模型 adapter 复制跨流 allocator 逻辑。

---

## 6. Reshard 与 symmetric memory：一个管驻留窗口，一个管通信实现

### ① 背景/问题

forward 后立即 reshard 可回收完整参数，但 backward 前必须再次 all-gather；保留参数则省掉这次通信，却扩大 activation 存活期间的参数驻留。另一方面，即使时序不变，staging buffer/collective 实现仍可能成为瓶颈。这两类问题分别由 reshard policy 与 symmetric-memory backend 处理，不能混成一个“FSDP overlap 开关”。

### ② 为什么这么设计

**选中路线**是场景化 reshard：普通非 PP decoder 默认释放，PP 默认保留；临近 backward 的尾部小组也默认保留。**替代方案**是全模型统一 always/never。决定性标准是下一次使用距离：PP 若每个 microbatch 都 reshard，会产生昂贵且无法充分重叠的重复 gather；尾部组刚释放就会被 backward 重新预取（`torchtitan/distributed/fsdp.py:112`、`torchtitan/distributed/fsdp.py:132`；`torchtitan/distributed/fsdp.py:258`、`torchtitan/distributed/fsdp.py:264`）。

symmetric memory 则是 opt-in：默认通信路径更通用；只有满足设备/拓扑条件时才把所有可见 FSDP modules 切到 symm-mem，并强制 sum reduction 以满足零拷贝 collective 的 reduction 约束（`torchtitan/distributed/fsdp.py:102`、`torchtitan/distributed/fsdp.py:109`；`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:659`、`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:665`）。

### ③ 实现思路与细节

`get_fsdp_reshard_after_forward_policy()` 将 `always/never/default` 映射为布尔值；default 等于 `not pp_enabled`（`torchtitan/distributed/fsdp.py:124`、`torchtitan/distributed/fsdp.py:136`）。但 helper 还有局部覆盖：tied 首尾组和非 tied `[norm, lm_head]` 只有显式 `always` 才 reshard（`torchtitan/distributed/fsdp.py:238`、`torchtitan/distributed/fsdp.py:264`）。root module 的上游默认也会在 lazy init 关闭 auto-reshard，避免 forward 结束即释放、backward 立刻再 gather（`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_state.py:203`、`[pt]torch/distributed/fsdp/_fully_shard/_fsdp_state.py:207`）。

启用 symm-mem 时，公共 decoder helper 在完成所有 `fully_shard` 后遍历 `FSDPModule`，先设 force-sum，再调用 `set_symm_mem_for_comm()`（`torchtitan/distributed/fsdp.py:102`、`torchtitan/distributed/fsdp.py:109`；`torchtitan/distributed/fsdp.py:368`、`torchtitan/distributed/fsdp.py:371`）。上游说明该 backend 为 all-gather staging buffer 提供 symmetric allocation；单机可用 Copy Engine AG，多机可用 symmetric-kernel AG，具体依赖拓扑（`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:828`、`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:835`）。

### ④ 约束/代价/失败边界

- TorchTitan 配置在 NVIDIA 上要求 CUDA 可用且 compute capability 至少 9.0；ROCm 不走这一 NVIDIA capability 拒绝分支（`torchtitan/config/configs.py:272`、`torchtitan/config/configs.py:281`）。
- 上游当前 symm-mem API 只接受 NCCL backend；Copy Engine AG 还要求 NCCL zero-CTA policy，并且不能与 custom all-gather/reduce-scatter 同时使用（`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:837`、`[pt]torch/distributed/fsdp/_fully_shard/_fully_shard.py:856`）。
- 当前 H100 suite 只有两卡 Llama3 symm-mem integration case，并跳过 ROCm（`tests/integration_tests/h100.py:30`、`tests/integration_tests/h100.py:36`）。这证明一条受测入口，不证明所有模型/拓扑受支持。
- **当前接线不一致**：Llama3 向公共 helper 传 `enable_fsdp_symm_mem`（`torchtitan/models/llama3/parallelize.py:68`、`torchtitan/models/llama3/parallelize.py:77`）；Qwen3.5 调同一 helper 时没有传该字段，vision helper 的签名也没有 symm-mem 参数（`torchtitan/models/qwen3_5/parallelize.py:115`、`torchtitan/models/qwen3_5/parallelize.py:138`；`torchtitan/distributed/fsdp.py:139`、`torchtitan/distributed/fsdp.py:148`）。因此配置注释所说“all FSDP modules”只能理解为**已接通 helper 的模型范围**，不是仓库全模型保证。

### ⑤ 发展趋势（有锚点的推断）

提交 `58b034444` 才把 FSDP symm-mem 接入默认 Trainer；提交正文报告 4×GB200 上多个模型约 1.1%–6.3% throughput 提升，并明确不覆盖 GraphTrainer 或 async-TP symm-mem kernels。结合当前 Qwen3.5 漏传，**推断**：这仍是逐模型补齐中的可选优化，不应成为无条件默认值。

---

## 7. 旧心智模型的明确修正

### ① 背景/问题

旧页把 PyTorch 2.9.1 的内部行号、TorchTitan 当时的 adapter 与若干容量推导揉成同一个“当前实现”，导致读者难以判断哪部分是 TorchTitan 保证、哪部分只是特定上游版本的实现细节。

### ② 为什么这么设计

本次选择双 baseline：TorchTitan baseline 决定“接了哪些策略”，单列 PyTorch 辅助 baseline 解释“hook/stream/buffer 怎样执行”；**替代方案**是继续用 `[pt]:NN` 无文件名短引用。决定性标准是让每个机制断言能回到唯一 commit 与完整路径。TorchTitan 自身明确说明它依赖 moving nightly 而非 pinned release（`torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py:83`、`torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py:96`）。

### ③ 实现思路与细节

本页保留的稳定因果链是：group boundary 决定单次通信/驻留粒度；prefetch edge 决定同时在途的 groups；reshard 决定 forward/backward 之间是否保留完整参数；stream/event/allocator 决定这些生命周期能否安全重叠。每个环节分别落到 TorchTitan adapter 或上游 FSDP2 locator，不再用一张固定时序图代替所有模式。

### ④ 约束/代价/失败边界

- PyTorch `2.9.1` 不再作为本页当前机制 baseline；旧 `[pt]:NN` 引用已全部删除。
- 本页没有声称 TorchTitan 锁死到辅助 PyTorch commit；辅助 baseline 只是本次可复核的上游实现快照。不同 nightly 必须重新核对内部行号与 buffer 策略。
- 没有运行 GPU profiler 或容量 benchmark；历史提交中的 profiler/GB200 数据只作为演进证据，不能替代当前硬件测量。
- GraphTrainer 的 graph bucketing/prefetch、SimpleFSDP collective IR 与 FlexShard optimizer reshard 都是独立 runtime，不应套用本页 eager FSDP2 hook 结论。

### ⑤ 发展趋势（有锚点的推断）

上游 buffer retention 已出现可配置上限，TorchTitan 又新增了 symm-mem adapter；两者都锚定“把固定策略变成显式资源窗口”。**推断**：后续文档更应记录可观测窗口与模型 adapter 覆盖矩阵，而不是继续维护一条看似精确、实则仅适用于单一版本的固定五流时间线。

## Related Pages

- [[11_torchtitan_fsdp_analysis]] —— 当前 FSDP storage mesh、分组所有权、mixed precision 与全局 token 梯度语义总览。
- [[14_torchtitan_pp_analysis]] —— PP microbatch 为什么改变默认 reshard 策略，以及 stage-local FSDP 的边界。
- [[15_torchtitan_ep_analysis]] —— EP token dispatch 的 host/device 同步与 per-param sparse mesh，是显式预取链的直接背景。
- [[21_torchtitan_hsdp_backward_overlap_analysis]] —— 展开 reduce-scatter 与 replicate all-reduce 的双层反向通信。
- [[24_torchtitan_comm_optimizations_overlap_analysis]] —— symmetric memory、compile overlap 与其他通信优化的横向地图。
- [[25_torchtitan_simple_fsdp_analysis]] —— 对比 eager hook 驱动 FSDP2 与编译图内 collective 的参数物化路径。
- [[27_torchtitan_graph_trainer_compiler_runtime_analysis]] —— GraphTrainer 如何在图级 passes 中重新安排 bucket 与 prefetch，而非复用本页的 eager hook 时序。
