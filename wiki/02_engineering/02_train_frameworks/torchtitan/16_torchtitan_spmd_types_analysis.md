---
title: "SPMD Types：把命名轴布局降成状态分片与模块边界 collective"
---

# SPMD Types：把命名轴布局降成状态分片与模块边界 collective

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页回答**：当前 `spmd_types` 与 `partial_dtensor` 各把哪些布局放进运行时值，`SpmdType + PartitionSpec` 怎样从逻辑轴降成具体 placement，`ShardingConfig` 的 state/input/output src-dst contract 怎样变成参数切分、边界 collective 与 local kernel region，以及 typechecking、singleton、uneven shard 的失败边界。
>
> **边界**：rank 预算和 dense/sparse mesh 的来源归 [[02_engineering/02_train_frameworks/torchtitan/10_torchtitan_parallel_dims_analysis|并行维度与进程网格]]；TP/CP/EP 页面负责各算法。本页只解释这些算法如何被表达成模块布局契约。文中的“布局契约”是知识库概括；上游实际公开对象名是 `SpmdType`、`PartitionSpec`、`ShardingConfig`。

---

## 1. Overview

### 背景与问题

“张量在 TP 上切 feature、在 CP 上切 token”不能只靠一组 `Placement` 列表表达：列表把语义绑在某张 mesh 的物理轴序上；而 TorchTitan 同一模型既会运行在 dense `[dp, cp, tp]` mesh，也可能进入 sparse `[dp_replicate, efsdp, ep]` mesh。参数存储还要把逻辑 `dp` 展开成 `dp_replicate + dp_shard`，这又不是前后向激活应看到的坐标（`torchtitan/distributed/parallel_dims.py:53-65`、`torchtitan/distributed/parallel_dims.py:229-279`）。

另一个问题是 collective 的正确性不只取决于目标布局。若只写“输出应变成 sequence shard”，框架无法检查 kernel 实际产生的是否为 TP partial，也无法区分 reduce-scatter、all-gather 或无操作。当前 `ShardingConfig` 因而显式声明 source 与 destination 两侧（`torchtitan/protocols/sharding.py:59-113`）。

### Thesis

TorchTitan 当前把模型并行写成两阶段编译：模型 config 先用命名逻辑轴上的 `SpmdType` 声明状态和边界值，再由 `Module.parallelize()` 按选定 backend 降成物理状态 shard、输入/输出 redistribution 和 local region。默认 `spmd_types` 让普通本地 tensor 携带外部类型语义并按单轴规则发 collective；`partial_dtensor` 则只把 TP/EP 放进 DTensor，DP/CP 继续留在带外（`torchtitan/distributed/parallel_dims.py:461-482`、`torchtitan/protocols/module.py:559-730`）。

这不是“DTensor 已完全消失”：`partial_dtensor` 仍是配置允许的 live backend；`spmd_types` 也在 state-dict bridge 和 placement 解析处借用 DTensor 表示（`torchtitan/config/configs.py:168-180`、`torchtitan/distributed/spmd_types.py:66-105`）。真正改变的是前后向值的权威语义从容器类型转到模块声明与 current-mesh context。

### 核心对象

| 对象 | 当前职责 | 不负责什么 |
|---|---|---|
| `SpmdType` | 记录命名 mesh 轴的 per-axis 类型，并可携带 `PartitionSpec`（`torchtitan/distributed/spmd_types.py:54-63`、`:231-253`） | 不自行选择 TorchTitan 的 DeviceMesh |
| `PartitionSpec` | 指定每个 tensor dim 被哪些逻辑轴、以何顺序切分（`torchtitan/distributed/spmd_types.py:440-488`） | 不等同于无序的 per-axis `S(dim)` 集合 |
| `ShardingConfig` | 统一 state、input src/dst、output src/dst、local-map 梯度声明（`torchtitan/protocols/sharding.py:59-113`） | 不直接执行 collective |
| `Module.parallelize()` | 递归切状态并包装 forward 边界（`torchtitan/protocols/module.py:244-290`） | 不替代 FSDP 的参数生命周期 |
| SPMD context | 注册 dense/sparse mesh、设置当前 mesh、可选启用 checker（`torchtitan/distributed/utils.py:397-425`） | 不是 `partial_dtensor` 的共同运行时状态机 |

### 从声明到执行

```text
model Config.update_from_config
  -> 子模块 ShardingConfig
  -> Module.parallelize
       -> validate redistributions
       -> distribute states
       -> wrap local region
       -> wrap input and output boundaries
  -> preprocess_inputs 标注顶层 tensor
  -> train_context 激活 dense current mesh 与可选 checker
  -> routed experts 暂时切 sparse current mesh
```

对应入口分别是公共 decoder 布局积木（`torchtitan/models/common/decoder_sharding.py:24-168`）、parallelize lowering（`torchtitan/protocols/module.py:244-290`）、输入标注（`torchtitan/models/common/decoder.py:351-386`）和训练上下文（`torchtitan/trainer.py:579-585`、`:703-720`）。

### Quick Start

默认 backend 已是 `spmd_types`；全局类型检查默认关闭：

```python
config.parallelism.spmd_backend = "spmd_types"
config.debug.spmd_typechecking = True
```

配置只接受 `spmd_types` 或 `partial_dtensor`（`torchtitan/config/configs.py:168-180`）。`spmd_typechecking` 只对前者生效（`torchtitan/config/configs.py:348-354`）；它是调试强度开关，不是布局/collective 开关。

---

## 2. 双后端边界与 `SpmdLayout` 迁移

### ① 背景/问题

TorchTitan 既需要一条让编译器看见普通 tensor 与 functional collective 的新路径，也不能在一次迁移中要求所有模块放弃 DTensor。旧基线还容易把 `full_dtensor`、`partial_dtensor`、`spmd_types` 当成三个并列现状，或把 TorchTitan 自己的 `SpmdLayout` 当作当前公共类型。

### ② 为什么这么设计

选中路线是共享同一份 `ShardingConfig[SpmdType]` 声明，在 lowering 时分叉；明显替代方案是两套模型配置，分别写 DTensor placement 与 SPMD 类型。**知识库推断**：共享声明减少了 backend 间语义漂移，决定性标准是 source/destination contract 与模型结构只保留一份，而容器和 collective 实现可以不同（`torchtitan/protocols/sharding.py:59-113`、`torchtitan/protocols/module.py:279-290`）。

### ③ 实现思路与细节

- 配置默认值与全局 runtime 默认值均为 `spmd_types`，但 `partial_dtensor` 仍是合法选项（`torchtitan/config/configs.py:168-180`、`torchtitan/distributed/utils.py:36-47`）。
- `spmd_types` 把 `dp/cp/tp/ep` 都视为 in-band 逻辑轴；`partial_dtensor` 的 mesh resolver 只保留 `tp/ep`，DP/CP 布局由其他机制处理（`torchtitan/distributed/parallel_dims.py:461-482`）。
- dense mesh 也随 backend 分叉：`spmd_types` 同时建 FSDP storage view `[dpR, dpS, cp, tp]` 与 fwd/bwd view `[dp, cp, tp]`；`partial_dtensor` 把 `dp_shard × cp` 折成 `fsdp`（`torchtitan/distributed/parallel_dims.py:229-260`）。
- state lowering 在 `spmd_types` 下调用 `spmd.shard()` 后重新注册本地 parameter/buffer 并断言类型；在 `partial_dtensor` 下调用 `distribute_tensor()` 产生 DTensor（`torchtitan/protocols/module.py:292-325`、`:366-452`）。
- input/output lowering 在 `spmd_types` 下检查声明并调用 `spmd_redistribute_per_axis()`；在 `partial_dtensor` 下用 `DTensor.from_local()` 和 `DTensor.redistribute(async_op=True)`（`torchtitan/protocols/module.py:559-730`）。

历史上，提交 `5ab3a0fd10a1` 把 `spmd_types` 设为默认，同时保留旧 backend；提交 `601cf4d2304e` 删除 `full_dtensor`，正文明确把 `partial_dtensor` 称为 fallback。当前依赖固定为 `spmd_types==0.2.5`（`pyproject.toml:30`）。提交 `8179a3a0df84` 随依赖升级删除 TorchTitan 临时 `SpmdLayout` dataclass，并把模型声明改成依赖提供的 `SpmdType + PartitionSpec`。该提交正文只有 ghstack 元数据，没有额外 Motivation；从其 exact diff 可验证旧类自称 temporary、等待依赖 API，而新代码直接 import `SpmdType`（`torchtitan/protocols/sharding.py:15-29`）。**基于提交差异的推断**：迁移判据是上游类型已能承载 per-axis type 与有序 PartitionSpec，继续维护 TorchTitan 重复适配层不再有收益，而不是两者只做了类名替换。

### ④ 约束/边界

`partial_dtensor` 是 live 兼容后端，不是已经删除的 legacy；`full_dtensor` 才不在当前配置枚举中（`torchtitan/config/configs.py:168-180`）。反过来，`spmd_types` 也不等于“全程无 DTensor”：checkpoint/state transfer 会把 plain local state 临时表示为 DTensor，并要求每个 tensor 有布局 metadata（`torchtitan/distributed/spmd_types.py:66-105`）。

当前 `ParallelDims.resolve_shared_mesh()` docstring 仍写着复数 `SpmdLayouts`，但函数签名是 `Iterable[SpmdType | None]`（`torchtitan/distributed/parallel_dims.py:484-512`）。这是遗留文字，不是现行类；不能据此复活已删 API。

### ⑤ 发展趋势（有锚点的推断）

mesh 构造留有“SPMD 不再与 DTensor/default backend 共享路径后清理”的 TODO（`torchtitan/distributed/parallel_dims.py:229-253`）。redistribution helper 也把“partial_dtensor 移除后改成 collective-based 声明”列为一种未来路线（`torchtitan/distributed/spmd_types.py:256-270`）。**推断**：代码在向单一语义前端收敛，但当前源码没有承诺删除 `partial_dtensor` 的版本或日期。

---

## 3. `SpmdType`：从逻辑轴语义到具体 placement

### ① 背景/问题

模型作者希望写“TP 切 hidden、CP 切 token”，而不是写“mesh 第 2 轴用 Shard(1)”。同一 tensor dim 还可能先按 DP、再按 CP、最后按 TP 连续切分；只保存 `{DP:S(0), CP:S(0)}` 会丢失全局 slice 的先后顺序。

### ② 为什么这么设计

选中路线是 `SpmdType.local_type` 记录命名轴语义，`PartitionSpec` 记录 tensor-dim 与有序轴元组；明显替代方案是直接保存按物理 mesh 顺序排列的 DTensor placements。前者可在解析时选择 dense/sparse 或 TP-only mesh，并能区分 `(DP, CP)` 与 `(CP, DP)` 的 rank-to-slice 映射（`torchtitan/distributed/spmd_types.py:440-488`、`tests/unit_tests/cpu/test_parallel_dims.py:380-424`）。

### ③ 实现思路与细节

TorchTitan 当前使用这些依赖类型；下表描述的是**源码中的使用方式**，不是知识库新造的类型：

| per-axis 类型 | TorchTitan 中的典型用途 | 当前证据 |
|---|---|---|
| `R` | dense 参数在 DP/CP 上复制；SP norm weight 在 TP 上复制 | `torchtitan/models/common/decoder_sharding.py:24-36`、`:153-168` |
| `I` | labels 或 all-reduce 后的激活在 TP 上保持相同 | `torchtitan/models/common/decoder_sharding.py:110-119`、`:133-150` |
| `P` | rowwise matmul 的待规约输出，或 local-map 的 K/V 输入梯度 | `torchtitan/models/common/decoder_sharding.py:133-150`、`:275-314` |
| `S(dim)` | 简单的一轴切分，如 colwise weight 的 TP `S(0)` | `torchtitan/models/common/decoder_sharding.py:122-130` |
| `V + PartitionSpec` | 值随 rank 变化，具体 tensor dim 和多轴切片顺序由 spec 给出 | `torchtitan/models/common/decoder_sharding.py:39-107` |

解析分两步：

1. `_per_axis_types()` 读取 `local_type`，再把 `PartitionSpec` 中每个轴覆盖为对应 `S(dim)`；轴必须是可转成 `MeshAxisName` 的字符串（`torchtitan/distributed/spmd_types.py:231-253`）。
2. `resolve_placements()` 把逻辑 `dp` 展开为 `dp_replicate + dp_shard`，按目标 DeviceMesh 的轴序调用依赖的 `spmd_type_to_dtensor_placement()`（`torchtitan/protocols/sharding.py:120-159`）。

布局可以声明目标 mesh 没有的额外轴，这些轴被忽略；但目标 mesh 的每一轴都必须在声明中有类型，否则立即报错（`torchtitan/protocols/sharding.py:124-154`）。这使同一 GQA local-map 声明能覆盖完整 DP/CP/TP 语义，而 `partial_dtensor` 的 TP-only mesh 只消费 TP 部分（`torchtitan/models/common/decoder_sharding.py:275-314`）。

### ④ 约束/边界

`PartitionSpec` 顺序是数据所有权，不是注释顺序。状态切分会严格依次执行轴 shard；CPU distributed test 验证 `(DP, CP)` 与 `(CP, DP)` 给相同 rank 分配不同 global slice（`torchtitan/distributed/spmd_types.py:440-488`、`tests/unit_tests/cpu/test_parallel_dims.py:380-424`）。

当前还有依赖 workaround：`PartitionSpec` 是 immutable variadic tuple subclass，TorchTitan 临时覆盖其 `__deepcopy__`，等待依赖修复（`torchtitan/distributed/spmd_types.py:29-31`）。因此 config deepcopy 可以共享 spec 对象，但这不是模型作者应依赖的 identity 契约。

---

## 4. `ShardingConfig`：state 与 input/output src-dst 契约

### ① 背景/问题

参数如何分片、输入进入 kernel 前需要什么布局、kernel 产出什么布局、交给下一层时应是什么布局，是四个不同问题。只声明目标 placement 会掩盖上游或 kernel 的错误：一个本应产生 `Partial` 的 rowwise matmul 若被误标成 replicated，后续仍可能发 collective，却无法证明语义正确。

### ② 为什么这么设计

选中路线是 source 与 destination 成对声明；明显替代方案是只写 destination，让运行时从容器猜 source。源码明确说显式 src 是为了统一 DTensor 与未来 erased-type system，并让每次 redistribute 两侧都有契约（`torchtitan/protocols/sharding.py:59-105`）。决定性标准是 collective 之前能验证现状，而不是让转换把错误“洗成”目标布局。

### ③ 实现思路与细节

| 字段 | 状态/边界含义 | lowering |
|---|---|---|
| `state_shardings` | 当前模块直属 parameter/buffer 的布局，按名字索引 | 初始化时物理切状态（`torchtitan/protocols/module.py:366-452`） |
| `in_src_shardings` | forward 参数抵达边界时的布局 | checker/DTensor placement 先验证（`torchtitan/protocols/module.py:591-646`） |
| `in_dst_shardings` | kernel 真正需要的输入布局 | src→dst redistribution（`torchtitan/protocols/module.py:597-655`） |
| `out_src_shardings` | kernel 刚产生的布局 | checker/DTensor placement 先验证（`torchtitan/protocols/module.py:660-721`） |
| `out_dst_shardings` | 交给下个模块前的布局 | src→dst redistribution（`torchtitan/protocols/module.py:696-730`） |
| `local_map` | local kernel 的输入、输出和输入梯度边界 | backend-specific local wrapper（`torchtitan/protocols/module.py:459-557`） |

真实 rowwise TP 例子把 weight 声明为 TP `S(1)`，把 matmul 输出声明为 TP `P`；开启 sequence parallel 时目标为 token/sequence shard，否则目标为 TP `I`。同一份 src/dst 差分别触发 reduce-scatter 或 all-reduce（`torchtitan/models/common/decoder_sharding.py:133-150`）。

`Module.parallelize()` 先递归处理子模块，再校验 redistribution、切直属状态、构造 local region，最后把 forward 改写为 `redistribute inputs → kernel → redistribute outputs`（`torchtitan/protocols/module.py:244-290`）。FSDP hook 位于 `__call__` 外层，因此仍包住改写后的 forward（`torchtitan/protocols/module.py:244-257`）。

### ④ 约束/边界

- 每个直属 parameter/buffer 必须在 `state_shardings` 有条目；漏项直接报错，`None` buffer 仅保留初始化槽位（`torchtitan/protocols/module.py:366-452`）。
- `spmd_types` 下只要声明 dst 就必须显式声明 src；输入和输出都有对应 guard（`torchtitan/protocols/module.py:597-620`、`:674-703`）。
- 同一 boundary 的所有非空布局必须具有完全相同的轴集合；placement 值可以不同，因为那正是 redistribution，轴集合不同则 assertion 失败（`torchtitan/distributed/parallel_dims.py:484-512`）。
- 一个模块实例只能 parallelize 一次，防止重复切状态和嵌套包装 forward（`torchtitan/protocols/module.py:259-264`）。

---

## 5. Redistribution 与 local region：契约如何变成 collective

### ① 背景/问题

声明 `P@TP → sequence shard@TP` 仍需落成真实通信。与此同时，FlashAttention、GroupedExperts 等 kernel 通常只接受普通本地 tensor；强迫每个 kernel 理解分布式容器会扩大改造面。

### ② 为什么这么设计

选中路线是把 collective 放在模块边界，并用受控 local region 隔离不理解布局类型的 kernel；明显替代方案是模型 forward 手写 all-gather/reduce-scatter，或让所有 kernel 原生接收 DTensor。**知识库推断**：边界 lowering 让状态、source 检查和通信共用一份声明，local region 则保留既有 kernel；决定性代价是通用 planner 能力弱于 DTensor。

### ③ 实现思路与细节

- `spmd_types` 在 parallelize 入口先验证 src/dst pair，当前只允许至多一个轴变化（`torchtitan/distributed/spmd_types.py:256-395`）。运行时遍历 dst axes，只对类型变化且 size>1 的轴调用 `spmd.redistribute()`（`torchtitan/distributed/spmd_types.py:398-437`）。
- `partial_dtensor` 先从 src layout 解析 shared mesh；plain tensor 用 `DTensor.from_local(run_check=False)` 包装，再按 dst placements 调用 `redistribute(async_op=True)`（`torchtitan/protocols/module.py:624-655`）。
- local region 的输入布局取 `in_dst`，输出布局取 `out_src`。`partial_dtensor` 使用 PyTorch experimental `local_map()`；`spmd_types` 把 `local_type + partition_spec` 传给 `spmd.no_typecheck()`（`torchtitan/protocols/module.py:459-557`）。
- GQA inner attention 是完整实例：Q 保留 CP token shard，K/V 在 kernel 前 all-gather 为 CP replicated，K/V 输入梯度以 CP partial 离开 local region（`torchtitan/models/common/decoder_sharding.py:275-314`）。

### ④ 约束/边界

当前通用 helper 明确拒绝：一次改变多根轴、`V` 作为变化轴的 src/dst、以及 `(DP, CP) → (CP, DP)` 这种 per-axis 类型相同但全局 shard 顺序变化的转换（`torchtitan/distributed/spmd_types.py:293-380`）。CPU tests 把三类错误都固定为失败契约（`tests/unit_tests/cpu/test_parallel_dims.py:301-375`）。复杂重排必须拆成显式 collective 或使用专用 kernel。

输出 redistribution 当前只完整处理单 tensor；非 tensor 原样返回，tuple `out_src` 在 `spmd_types` 路径直接拒绝，嵌套输出仍是 TODO（`torchtitan/protocols/module.py:660-703`）。local-map 虽支持 tuple-like output type tree，也不能据此推断通用 output redistribution 已支持任意 pytree。

### ⑤ 发展趋势（有锚点的推断）

helper TODO 提议把多轴排序能力移入 `spmd_types`，或在 `partial_dtensor` 删除后改成直接 collective 声明（`torchtitan/distributed/spmd_types.py:256-270`、`:398-412`）。decoder sharding 还把“collective 移进模块、边界 src→dst 消失”标为 transitional 方向（`torchtitan/models/common/decoder_sharding.py:255-271`）。**推断**：边界 contract 仍是当前权威机制，这些 TODO 只说明候选收敛方向，不是已经落地的调用链。

---

## 6. Dense/sparse current mesh 与 typechecking

### ① 背景/问题

同一逻辑轴名必须在运行时解析到正确 process group。Dense attention 使用 `[dp, cp, tp]`，MoE 专家窗口却使用 `[dp_replicate, efsdp, ep]`；若类型断言只依赖一个全局 mesh，进入 routed experts 后会在错误 group 上解释 EP/EFSDP。

### ② 为什么这么设计

选中路线是把 dense/sparse mesh 注册到线程局部状态，并用栈式 current-mesh context 描述运行时区域；明显替代方案是每个 collective 显式传完整 mesh，或永久绑定 dense mesh。**知识库推断**：TLS 减少了模块签名中的拓扑参数，决定性代价是执行 forward/backward 的线程必须看到同一栈。

### ③ 实现思路与细节

1. `get_spmd_context()` 注册 dense/sparse mesh，并把 dense mesh 压入 current stack；若调试开关为真，再进入全局 `spmd_typecheck(local=False)`（`torchtitan/distributed/utils.py:397-425`）。
2. `set_current_spmd_mesh()` 同时维护 TorchTitan TLS stack 和依赖的 `spmd.set_current_mesh()` context，退出时以 identity 断言成对 pop（`torchtitan/distributed/spmd_types.py:130-181`）。
3. RoutedExperts 只在 `inner_experts` 计算窗口调用 `maybe_set_sparse_mesh()`，dispatch/combine 位于该窗口外（`torchtitan/models/common/moe.py:150-169`）。非 `spmd_types` 或无 sparse mesh 时 context manager 是 no-op（`torchtitan/distributed/spmd_types.py:184-192`）。
4. Decoder preprocess 先准备 CP input，再要求每个顶层 tensor 有布局并在 dense mesh 下 `assert_type`；缺失条目会列名报错，容器内部 tensor 必须在构造点自行标注（`torchtitan/models/common/decoder.py:351-386`、`torchtitan/distributed/spmd_types.py:195-228`）。
5. 模块边界只在 `spmd.is_type_checking()` 为真时执行 src `assert_type`，避免 compile 因无全局 checker 而报错；布局仍然驱动 redistribution（`torchtitan/protocols/module.py:597-620`、`:674-703`）。

### ④ 约束/边界

autograd multithreading 被全局关闭：源码明确说 activation checkpoint backward/recompute 线程否则看不到 TLS DeviceMesh stack 与 process groups（`torchtitan/distributed/utils.py:446-463`）。这是当前设计的真实运行时成本。

typechecking 关闭不等于退回普通 local forward；parameter sharding、redistribution wrapper 和 local region 仍由 `Module.parallelize()` 安装（`torchtitan/protocols/module.py:244-290`）。区别是全局 checker 与边界 source assertions 不执行；输入 preprocessing 和 state registration 仍会用 `assert_type` 附着声明（`torchtitan/distributed/spmd_types.py:195-228`、`torchtitan/protocols/module.py:292-325`）。

`current_spmd_mesh()` 在非 `spmd_types` backend 或空栈时返回 `None`，`spmd_mesh_size()` 对缺失轴返回 1（`torchtitan/distributed/spmd_types.py:138-156`）。因此 TLS current mesh 不能作为 `partial_dtensor` 的共同抽象。

---

## 7. Singleton、uneven shard 与测试边界

### ① 背景/问题

degree=1 时 `Shard`、`Partial`、`Replicate` 在数据上都不发生通信，但 DTensor op rules 仍把 placements 当作不同类型。另一边，TP/EP 若不能整除参数维，局部 shard 会不等长；许多 kernel、state sync 和 type rule 并没有这种语义。

### ② 为什么这么设计

singleton 选择在 DTensor bridge 处规范化为 `Replicate`，而在 `spmd_types` state 注册时保留完整逻辑轴、交给依赖过滤；明显替代方案是模型 config 针对 degree=1 改写布局。当前路线保持配置拓扑无关，并规避 DTensor 严格 placement equality（`torchtitan/protocols/sharding.py:120-159`、`torchtitan/protocols/module.py:301-325`）。

uneven TP/EP 参数则选择早失败，不尝试 padding 或不规则 local shard。**知识库推断**：决定性标准是 model-parallel kernel 与参数状态必须维持规则形状；当前 guard 只针对 TP/EP parameter shard，不是所有 tensor 的一般整除策略（`torchtitan/protocols/module.py:327-364`）。

### ③ 实现思路与细节

- `resolve_placements()` 在 size-1 mesh axis 上把 DTensor `Shard`/`Partial` 转成 `Replicate`；TODO 说明待 FlexShard 替代 `fully_shard` 后再删除 workaround（`torchtitan/protocols/sharding.py:124-159`）。
- `_spmd_distribute_state()` 查询 mesh 时传 `include_singleton_axes=True`，让 `assert_type()` 仍看到声明中的所有轴；物理切分 helper 对 axis size 1 跳过 `spmd.shard()`（`torchtitan/protocols/module.py:301-325`、`torchtitan/distributed/spmd_types.py:470-488`）。
- runtime redistribution 同样跳过 size-1 axis（`torchtitan/distributed/spmd_types.py:417-437`）。
- 参数 guard 解析负 tensor dim，分别读取 TP/EP size；shape 不能整除就以模块名、参数名、tensor dim、mesh axis 和 degree 报错（`torchtitan/protocols/module.py:327-364`）。
- CPU tests 分别覆盖 shape `(4,5)` 在 TP2 上切 dim1，以及 shape `(3,4)` 在 EP2 上切 dim0 的拒绝（`tests/unit_tests/cpu/test_module.py:391-436`）。

### ④ 约束/边界

size-1 normalization 发生在 TorchTitan 到 DTensor 的 bridge，不表示 `R`、`I`、`P`、`S` 在 `spmd_types` 的语义已经相同；模型声明仍保留原类型。反过来，extra layout axes 可在较小 mesh 上忽略，但缺少目标 mesh 轴的声明会失败（`torchtitan/protocols/sharding.py:124-159`）。

uneven guard 只在 `spmd_types` parameter lowering 分支调用；`partial_dtensor` 走 `distribute_tensor()` 的现有行为（`torchtitan/protocols/module.py:377-420`）。不能把“当前拒绝 uneven TP/EP 参数”泛化成所有 backend、所有 activation shard 都有相同 guard。

当前 CPU tests 验证 PartitionSpec 顺序、all-gather 单 collective、三类 redistribution rejection 与 uneven parameter rejection（`tests/unit_tests/cpu/test_parallel_dims.py:236-452`、`tests/unit_tests/cpu/test_module.py:391-436`）。它们不覆盖真实 GPU 拓扑下的全部 TP+CP+EP 组合，也不证明全局 type checker 能发现任意 kernel 内部错误。

### ⑤ 发展趋势（有锚点的推断）

提交 `a23d97d193f8` 引入 uneven model-parallel parameter 拒绝；当前代码没有自动 padding TODO。singleton conversion 则有明确 FlexShard TODO（`torchtitan/protocols/sharding.py:137-138`）。**推断**：近期边界是继续早失败并维持规则参数形状，而 singleton workaround 有明确但无时间表的替代条件。

---

## 8. 旧心智模型纠偏与选择准则

### ① 背景/问题

SPMD 页面最容易把演进阶段当成现状：把 `SpmdLayout` 当当前类、把 `full_dtensor` 当第三 backend、把 typechecking 当通信开关，或把逻辑 `dp` 直接等同于 FSDP 的存储轴。

### ② 为什么这么设计

**知识库推断**：选择 backend 时应按值表示与工具链兼容性，而不是按“新路径必然更通用”。默认选择 `spmd_types` 可获得命名轴 contract、plain local tensor 与全局 checker；需要当前 DTensor model-parallel 路径时仍可显式选择 `partial_dtensor`。决定性标准是目标模型/训练器是否已满足单轴 redistribution、单输出和 TLS 约束，而不是配置是否能被解析（`torchtitan/config/configs.py:168-180`、`torchtitan/distributed/spmd_types.py:256-270`、`torchtitan/protocols/module.py:660-703`）。

### ③ 实现思路与细节

| 旧断言 | HEAD 事实 |
|---|---|
| `SpmdLayout` 是 TorchTitan 当前布局类 | `8179a3a0df84` 已删除临时类；当前声明直接使用依赖 `SpmdType`（`torchtitan/protocols/sharding.py:15-29`） |
| `full_dtensor`、`partial_dtensor`、`spmd_types` 三路并存 | 当前 config 只允许后两者（`torchtitan/config/configs.py:168-180`） |
| `spmd_types` 完全不用 DTensor | state-dict bridge 和 `resolve_placements()` 仍使用 DTensor 表示/placements（`torchtitan/distributed/spmd_types.py:66-105`、`torchtitan/protocols/sharding.py:120-159`） |
| typechecking=false 就不发边界 collective | lowering 总会包装 forward；开关只控制全局 checker 与条件式 boundary source assertion（`torchtitan/protocols/module.py:244-290`、`:597-620`） |
| 逻辑 `dp` 就是 `dp_shard` | bridge 会展开为 `dp_replicate + dp_shard`；fwd/bwd mesh 只看折叠后的 `dp`（`torchtitan/distributed/parallel_dims.py:53-65`、`:229-253`） |
| `PartitionSpec` 只是 shape 注释 | 它决定多轴重复切同一维的 global slice 顺序（`torchtitan/distributed/spmd_types.py:440-488`） |
| degree=1 无需布局类型 | bridge 需要显式规范化，state registration 仍保留 singleton 轴（`torchtitan/protocols/sharding.py:120-159`、`torchtitan/protocols/module.py:301-325`） |

### ④ 约束/边界

本页把 state ownership、value layout 和 boundary conversion 合称“布局契约”，便于跨 TP/CP/EP 推理；这不是 `spmd_types` 依赖里的一个上游类名。上游当前对象仍应按源码称为 `SpmdType`、`PartitionSpec` 与 `ShardingConfig`。

排障应依次检查：backend 的 in-band axes、src/dst 是否同轴集合、是否只变化一轴、PartitionSpec 顺序、current mesh 区域、singleton normalization、parameter shape 对 TP/EP 的整除性。每一层都有独立 guard，不能把最终 collective 错误都归因于 mesh 构造。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/10_torchtitan_parallel_dims_analysis|并行维度与进程网格]] — storage mesh、fwd/bwd mesh 与逻辑 `dp` 展开的坐标来源。
- [[02_engineering/02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis|FSDP 分片与生命周期]] — 参数存储所有权如何与 SPMD 值布局并行存在。
- [[02_engineering/02_train_frameworks/torchtitan/12_torchtitan_tp_analysis|张量并行 TP]] — colwise、rowwise、sequence-parallel contract 的主要消费者。
- [[02_engineering/02_train_frameworks/torchtitan/13_torchtitan_cp_analysis|上下文并行 CP]] — `(DP, CP)` token slicing 与 GQA local-map 的通信语义。
- [[02_engineering/02_train_frameworks/torchtitan/15_torchtitan_ep_analysis|专家并行 EP]] — dense/sparse current mesh transition 与 expert state placement。
- [[02_engineering/02_train_frameworks/torchtitan/24_torchtitan_comm_optimizations_overlap_analysis|通信优化与重叠]] — 边界 collective、编译与重叠窗口的全局视角。
