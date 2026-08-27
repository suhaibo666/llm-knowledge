# SPMD Types：把并行布局从运行时容器变成可检查的模块契约

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-27）
> **维度**：Deep Dive（默认 SPMD 后端）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **主线**：新版 TorchTitan 不再要求每个前后向值都靠 DTensor 对象携带布局。模型配置先用 `SpmdType` 声明参数、输入和输出在命名 mesh 轴上的语义，`Module.parallelize()` 再把声明降成状态切分、边界 collective 与局部 kernel 区域；可选 type checker 检查整个前后向是否遵守这些契约。FSDP 仍管理参数存储，SPMD types 管的是运行时值的布局语义，两者通过 `ParallelDims` 的双平面 mesh 对接。

---

## 1. Overview

### 1.1 这次演进改变了什么

知识库旧基线 `61c010fcb` 仍把 `full_dtensor` 描述成未来 SPMD 路线。当前演进已经收敛为：

- `spmd_types` 在提交 `5ab3a0fd1` 中成为默认后端，配置默认值和全局运行时默认值均为 `spmd_types`（`torchtitan/config/configs.py:168-180`、`torchtitan/distributed/utils.py:36-47`）。
- `full_dtensor` 在提交 `601cf4d23` 中删除；现行后端只剩 `spmd_types` 与兼容用 `partial_dtensor`（`torchtitan/config/configs.py:261-266`）。
- 依赖已固定到 `spmd_types==0.2.5`（`pyproject.toml:30`），模型侧使用 `SpmdType` + `ShardingConfig`，而不是旧的 `SpmdLayout`。

> [!deprecated] `full_dtensor` 不是现行路径
> 它试图用 DTensor 对象把更多布局带进整网；当前默认路径改为“本地 tensor + 外部 SPMD 类型契约 + 显式边界 collective”。不要把二者视为只改了类名。

### 1.2 三层结构

| 层 | 核心对象 | 回答的问题 | 关键源码 |
|---|---|---|---|
| **语义层** | `SpmdType`、`PartitionSpec` | 每个 mesh 轴上，这个值是复制、切分、部分和、相同还是随 rank 变化？ | `torchtitan/models/common/decoder_sharding.py:24-119` |
| **模块契约层** | `ShardingConfig` | 参数如何放置？输入/输出从什么布局变到什么布局？kernel 是否进入 local region？ | `torchtitan/protocols/sharding.py:59-113` |
| **运行时层** | `Module.parallelize()`、`SpmdContext` | 何时切参数、发 collective、设置当前 mesh、启用检查器？ | `torchtitan/protocols/module.py:244-290`、`torchtitan/distributed/utils.py:397-425` |

### 1.3 Quick Start：从哪里开启与读源码

默认无需显式开关；`parallelism.spmd_backend` 已是 `spmd_types`。若要开启全局布局检查，设置 `debug.spmd_typechecking=true`，该选项只在 `spmd_types` 后端生效（`torchtitan/config/configs.py:348-354`）。

源码阅读顺序：

```text
模型 Config.update_from_config
  -> 写入每个子模块的 ShardingConfig
     models/llama3/model.py:69-83
  -> model.parallelize(parallel_dims)
     models/llama3/parallelize.py:40-41
  -> 递归切状态 + 包 forward 边界
     torchtitan/protocols/module.py:244-290
  -> train step 在 SpmdContext 中执行
     torchtitan/trainer.py:579-584, 710-715
```

---

## 2. `SpmdType` 同时描述“每轴语义”与“全局切片顺序”

### 2.1 为什么只有 DTensor `Placement` 不够

TorchTitan 的布局以**轴名**而非 mesh 位置声明。`resolve_placements()` 在真正使用某张 mesh 时才按轴顺序翻译成 DTensor `Placement`；缺少所需轴会直接报错，布局里多出的轴则可在较小子 mesh 上忽略（`torchtitan/protocols/sharding.py:120-159`）。因此，同一份模型配置可以在 `[tp]`、`[dp,cp,tp]` 或 sparse mesh 上解析，而不把物理轴序写死在模型代码中。

`SpmdType.local_type` 表达每根轴的局部语义；`PartitionSpec` 在多根轴切同一 tensor 维时保存全局切片顺序。典型 dense activation 定义把 token 维写成 `(DP, CP)`、feature 维写成 `TP`（`torchtitan/models/common/decoder_sharding.py:39-61`）：

```text
shape = [tokens, hidden]
PartitionSpec((DP, CP), TP)

先沿 DP 切 tokens，再在每个 DP slice 内沿 CP 切 tokens；
hidden 则沿 TP 切。
```

顺序不是装饰。`spmd_distribute_tensor()` 会严格按 `PartitionSpec` 中的轴顺序逐次 shard（`torchtitan/distributed/spmd_types.py:440-488`）；`(DP, CP)` 与 `(CP, DP)` 即使每轴都写了 shard，也不是同一物理布局。

### 2.2 参数与激活故意使用不同语义

Dense 参数在 DP/CP 上是 replicated，TP placement 由具体线性层指定（`torchtitan/models/common/decoder_sharding.py:24-36`）；激活则在 DP 上随 rank 变化，并可同时被 CP 切 token、被 TP 切 feature（`torchtitan/models/common/decoder_sharding.py:39-61`）。

这揭示了 SPMD types 与 FSDP 的分工：

- 前后向类型系统把逻辑 DP 看成“不同 rank 持有不同样本/激活”。
- 参数存储仍由 FSDP 的 `dp_replicate`/`dp_shard` 轴管理；模型语义只需声明参数在逻辑 DP/CP 上复制。
- DP 逻辑轴向存储轴展开的桥接发生在 `unfold_dp_axis()` 与 `resolve_placements()`，而不是散落在每个模型里（`torchtitan/distributed/parallel_dims.py:53-65`、`torchtitan/protocols/sharding.py:140-158`）。

### 2.3 一组可复用的布局积木

公共 decoder sharding 把常见模式收敛成函数：token-id、attention activation、sequence-parallel activation、colwise、rowwise、norm（`torchtitan/models/common/decoder_sharding.py:64-168`）。例如：

- colwise weight 是 TP `S(0)`，输出 feature 是 TP `S(-1)`（`torchtitan/models/common/decoder_sharding.py:122-130`）。
- rowwise weight 是 TP `S(1)`，局部 matmul 先产生 TP partial；有 SP 时 reduce-scatter 到 sequence shard，否则 all-reduce 成 invariant（`torchtitan/models/common/decoder_sharding.py:133-150`）。
- decoder 输入为 token 维 `(DP, CP)` 分片，labels 在 TP 上 invariant（`torchtitan/models/common/decoder_sharding.py:110-119`）。

模型只补充自己的结构差异。Llama 3 的 config 更新阶段调用 `set_llama3_sharding_config()`，而不是在 forward 内临时决定布局（`torchtitan/models/llama3/model.py:69-83`）。

---

## 3. `ShardingConfig`：把模块边界变成五段式契约

`ShardingConfig` 有五组字段（`torchtitan/protocols/sharding.py:59-113`）：

| 字段 | 作用 |
|---|---|
| `state_shardings` | 当前模块直属参数/缓冲的布局 |
| `in_src_shardings` | 输入到达模块边界时应有的布局 |
| `in_dst_shardings` | forward 真正需要的输入布局 |
| `out_src_shardings` | forward 刚产出的布局 |
| `out_dst_shardings` | 交给下个模块前的目标布局 |
| `local_map` | 把 DTensor/typed value 暂时变成本地 tensor 运行 kernel，再包回输出 |

源/目标成对声明不是冗余：类型检查必须先知道“现在应是什么”，才能证明 collective 的输入合法；只有目标布局会把上游错误悄悄洗掉。当前 `Module` 因而在 redistribution 前先 assert source type（`torchtitan/protocols/module.py:597-620`、`torchtitan/protocols/module.py:674-703`）。

### 3.1 一个真实配对：rowwise TP

rowwise 线性层声明：

```text
state.weight : TP S(1)
output src   : TP Partial
output dst   : TP Shard(sequence)  或 TP Invariant
```

配置来源是 `torchtitan/models/common/decoder_sharding.py:133-150`。运行时看到 `Partial -> Shard` 就发 reduce-scatter，看到 `Partial -> Invariant` 就发 all-reduce；collective 不是模型作者手写在 forward 里，而是边界布局差触发。

### 3.2 `local_map` 为什么仍需要

很多 attention/MoE kernel 只接受普通本地 tensor。`LocalMapConfig` 声明进入 kernel 前后的布局与输入梯度布局（`torchtitan/protocols/sharding.py:33-53`）。

- `partial_dtensor` 后端使用 PyTorch `local_map` 把 DTensor 解包/回包（`torchtitan/protocols/module.py:493-537`）。
- `spmd_types` 后端用 `spmd.no_typecheck` 标注一个受控的本地实现区域，外层仍以输入/输出类型约束它（`torchtitan/protocols/module.py:539-557`）。

GQA inner attention 的配置展示了完整例子：Q 保持 CP token shard，K/V 在 local-map 边界 all-gather 成 CP replicate，反向 K/V 梯度以 partial 规约回去（`torchtitan/models/common/decoder_sharding.py:275-314`）。

---

## 4. `Module.parallelize()`：声明如何变成真实执行

### 4.1 递归、切状态、包 forward

`Module.parallelize()` 先递归处理子模块，再对有 `ShardingConfig` 的模块执行三步（`torchtitan/protocols/module.py:244-290`）：

1. 校验当前 backend 能否表达配置中的 redistribution。
2. 根据 `state_shardings` 切本模块直属参数/缓冲。
3. 把 forward 包成 `redistribute inputs -> optional local region -> forward -> redistribute outputs`。

它只允许每个 module 调一次，避免 forward 被重复包装（`torchtitan/protocols/module.py:259-264`）。FSDP hook 挂在 `__call__` 外层，所以仍能包住这个已重写的 forward（`torchtitan/protocols/module.py:244-257`）。

### 4.2 参数为什么最终仍是物理分片

默认后端的 `_spmd_distribute_state()` 解析 layout 所需 mesh，按 `PartitionSpec` 调用 `spmd.shard()`，重新注册 parameter/buffer，再在当前 mesh context 中 `assert_type`（`torchtitan/protocols/module.py:292-325`）。因此“类型系统使用本地 tensor”不等于“参数不分片”；区别在于运行时值不必一直包装成 DTensor 对象。

每个参数都必须在 `state_shardings` 中显式出现，漏项会报错；同一模块不同参数还可以解析到不同 mesh（`torchtitan/protocols/module.py:366-398`）。这对 MoE 很重要：dense 状态和 expert 状态不必被一张 mesh 的统一 placement 绑死。

### 4.3 输入从数据层进入类型世界

Decoder 的 `preprocess_inputs()` 先建 mask，再做 CP input 准备，最后在 `spmd_types` 后端给 batch 中每个 tensor 加类型断言（`torchtitan/models/common/decoder.py:351-386`）。`annotate_input_spmd_types()` 要求每个顶层 tensor 都有布局，缺失时不是静默当 replicate，而是列出未声明输入并报错（`torchtitan/distributed/spmd_types.py:195-228`）。

Trainer 随后把前后向放进 `train_context`（`torchtitan/trainer.py:710-715`）。这个 context 注册 dense/sparse mesh、激活 dense current mesh，并在 debug 开关开启时进入全局 type checker（`torchtitan/distributed/utils.py:397-425`）。

---

## 5. Redistribution：布局差如何选择 collective

`spmd_redistribute_per_axis()` 比较 src/dst 的每轴类型，只在发生变化且轴 size > 1 时调用 `spmd.redistribute`；库根据 per-axis type 选择 all-reduce、reduce-scatter 或 all-gather（`torchtitan/distributed/spmd_types.py:398-437`）。

这条路径与 DTensor 的关键差异是：

- `partial_dtensor` 先把 plain tensor 包成 DTensor，校验 placements，再调用 `DTensor.redistribute(async_op=True)`（`torchtitan/protocols/module.py:624-655`）。
- `spmd_types` 保持 plain/typed tensor，只在边界根据源/目标语义发单轴 collective（`torchtitan/protocols/module.py:597-622`）。

这让 collective 的位置由模块契约显式决定，也让编译器看到普通 tensor + functional collective；代价是当前 helper 还没有 DTensor planner 那样的一般多轴重排能力。

---

## 6. Dense / sparse current mesh 是运行时状态机

`ParallelDims` 同时创建 dense 前后向 mesh `[dp, cp, tp]` 与可选 sparse mesh `[dp_replicate, efsdp, ep]`（`torchtitan/distributed/parallel_dims.py:229-279`）。两张 mesh 注册在线程局部状态中（`torchtitan/distributed/spmd_types.py:108-145`）：

```text
进入 train step       -> current = dense mesh
进入 routed experts   -> current = sparse mesh
退出 experts          -> pop，恢复 dense mesh
```

`set_current_spmd_mesh()` 用栈保证嵌套恢复，`maybe_set_sparse_mesh()` 在 EP 未开启或非 `spmd_types` 后端时退化为 no-op（`torchtitan/distributed/spmd_types.py:159-192`）。这比在每个 collective 调用点传完整 mesh 更简洁，但也带来一个运行时不变量：执行 collective 的线程必须能看到同一 TLS mesh 栈。

为满足该不变量，分布式初始化显式关闭 autograd multithreading；源码说明激活重计算的 BWD 线程否则拿不到 current mesh/进程组（`torchtitan/distributed/utils.py:446-463`）。这是该设计的真实成本，不应被“只是静态类型标注”掩盖。

---

## 7. 当前失败边界

### 7.1 只支持单轴边界 redistribution

当前校验器拒绝三类变换（`torchtitan/distributed/spmd_types.py:256-380`）：

- 一次边界同时改变多根 mesh 轴；
- 让 `spmd.V` 作为 redistribution 的源或目标；
- 仅 per-axis type 相同、但 `PartitionSpec` 把 shard 顺序从 `(DP, CP)` 改成 `(CP, DP)`。

对应单测把这些限制固定为失败契约（`tests/unit_tests/cpu/test_parallel_dims.py:301-375`）。这意味着复杂跨轴重排要拆成多个显式 collective，或继续借助 DTensor/专用 kernel，不能假设 planner 会自动找路径。

### 7.2 TP/EP 参数必须均匀切分

默认后端在切参数前检查 TP/EP 所切 tensor dimension 能否整除轴大小，不能则直接拒绝（`torchtitan/protocols/module.py:327-364`）；单测分别覆盖 uneven TP 与 uneven EP（`tests/unit_tests/cpu/test_module.py:391-436`）。这比运行后得到不规则 local shard 更保守，但能避免大量 kernel/状态同步对不等长分片的隐式假设。

### 7.3 输出结构仍不一般

当前 `_redistribute_outputs()` 的注释明确只完整支持单 tensor 输出；tuple/dict 等嵌套输出仍需扩展配置结构（`torchtitan/protocols/module.py:660-688`）。因此，复杂 MoE/多模态模块常在更内层放 local region 或显式 collective，而不是把任意返回树都交给通用边界包装器。

### 7.4 type checking 是可选调试能力

全局 checker 只有 `debug.spmd_typechecking=true` 时才进入（`torchtitan/distributed/utils.py:419-422`）。即使不开 checker，配置仍驱动状态切分和 collective；区别是 source layout 断言不会在每个边界完整执行（`torchtitan/protocols/module.py:606-610`、`torchtitan/protocols/module.py:690-694`）。所以它不是“关掉就退回普通 TP”，而是“同一路径少了全局语义验证”。

---

## 8. 为什么这条路线优于明显替代方案

以下是从当前实现得出的**工程推论**，不是源码作者的原话：

- 相比“每个值始终是 DTensor”，typed local tensor 更容易进入现有 eager/compile kernel；`local_map`/`no_typecheck` 把不理解分布式类型的内核隔离在明确边界内。
- 相比“模型 forward 手写 all-gather/reduce-scatter”，`ShardingConfig` 让参数布局、边界 collective 与检查契约共用一份声明，减少同一布局在初始化/前向/反向三处漂移。
- 代价是通用性暂时较弱：复杂多轴重排、嵌套输出与 TLS/autograd 线程问题都由 TorchTitan 自己承担，而不是交给 DTensor dispatcher 统一处理。

这些取舍可以从声明结构（`torchtitan/protocols/sharding.py:59-113`）、通用 forward 包装（`torchtitan/protocols/module.py:244-290`）和当前失败边界（`torchtitan/distributed/spmd_types.py:256-380`）共同验证。

---

## 9. 小结

- `spmd_types` 已是默认且正在替代 `full_dtensor` 方向；`partial_dtensor` 仍作为迁移期对照存在。
- `SpmdType` 用命名 mesh 轴声明局部语义，`PartitionSpec` 补上多轴切同一 tensor 维时的全局顺序。
- `ShardingConfig` 把 state、input、output、local region 组织成统一模块契约。
- `Module.parallelize()` 把契约降成参数物理分片、forward 边界 collective 和局部 kernel 包装；FSDP 存储平面与 SPMD 前后向平面并行存在。
- 当前限制是单轴 redistribution、均匀 TP/EP 参数分片、单 tensor 输出与 TLS mesh 对 autograd 线程的约束。

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 本系列入口与代码演进总览。
- [[10_torchtitan_parallel_dims_analysis]] —— 存储 mesh 与前后向 SPMD mesh 的双平面坐标系。
- [[11_torchtitan_fsdp_analysis]] —— 参数状态最终如何由 FSDP2 在 `dp_replicate`/`dp_shard` 上保存与取回。
- [[12_torchtitan_tp_analysis]] —— colwise/rowwise/SP 布局契约如何落成 TP collective。
- [[13_torchtitan_cp_analysis]] —— CP input preparation 与 `(DP, CP)` token shard 的边界。
- [[15_torchtitan_ep_analysis]] —— dense/sparse current mesh 切换与 routed-expert local region。
