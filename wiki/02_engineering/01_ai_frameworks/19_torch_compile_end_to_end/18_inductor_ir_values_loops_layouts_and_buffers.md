# 18 · Inductor IR：Value、Loop、Layout 与 Buffer

> 前置：[[17_fx_lowering_to_inductor_ir]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. 为什么不是一个万能IRNode

Inductor要分别回答：

1. 计算什么value？
2. 如何按index计算？
3. 是否own storage或alias？
4. size/stride/offset是什么？
5. lazy还是realized？
6. template、external还是generated loop？
7. 是否作为schedulable operation？

这些维度正是子类/组合设计的来源。

这里最重要的设计选择不是“类很多”，而是把三种不能等价的关系拆开：

```text
计算关系：Loops.inner_fn / ExternKernel / Template
存储关系：Layout / Buffer / View / alias
调度关系：Operation → SchedulerNode → ReadWrites
```

一个对象可以同时参与多个维度，例如 `ComputedBuffer`同时是 `Buffer` 与 `Operation`；
但 `Pointwise`只有惰性计算，还不是具名存储；`ReinterpretView`可以代表已实现值，却不等于
产生一个新 operation。若用万能 IRNode 只放一组 `inputs/outputs` 边，就无法区分
“计算依赖”“共享 storage”与“必须按序执行”。

## 2. 顶层族

| 维度 | 代表类 |
|---|---|
| IR基础/provenance | `IRNode` |
| schedulable operation protocol | `Operation` |
| lazy loop | `Loops`, `Pointwise`, `Reduction` |
| view | `BaseView`, `ReinterpretView` |
| output/storage spec | `OutputSpec`, `Layout` |
| named storage | `Buffer` |
| buffer+operation | `OperationBuffer`, `ComputedBuffer` |
| template/extern | `TemplateBuffer`, `ExternKernel`, `FallbackKernel` |
| value wrapper | `TensorBox`, `StorageBox` |

关键声明位置：

- `IRNode`/`Operation`/`Loops`：`torch/_inductor/ir.py:589-655`、
  `torch/_inductor/ir.py:956-977`、`torch/_inductor/ir.py:979-1007`、
  `torch/_inductor/ir.py:1057-1086`、`torch/_inductor/ir.py:1090-1119`；
- `Pointwise`/`Reduction`/`Scan`：`torch/_inductor/ir.py:1219-1238`、
  `torch/_inductor/ir.py:1238-1256`、`torch/_inductor/ir.py:1384-1406`、
  `torch/_inductor/ir.py:2877-2946`；
- view/layout：`torch/_inductor/ir.py:3386-3412`、
  `torch/_inductor/ir.py:4061-4105`、`torch/_inductor/ir.py:4399-4412`、
  `torch/_inductor/ir.py:4416-4432`、`torch/_inductor/ir.py:4433-4448`、
  `torch/_inductor/ir.py:4726-4750`、`torch/_inductor/ir.py:4962-4976`、
  `torch/_inductor/ir.py:4978-4998`；
- buffer/template/extern/fallback：`torch/_inductor/ir.py:5160-5180`、
  `torch/_inductor/ir.py:5182-5194`、`torch/_inductor/ir.py:5306-5334`、
  `torch/_inductor/ir.py:5408-5435`、
  `torch/_inductor/ir.py:5876-5915`、`torch/_inductor/ir.py:6995-7035`、
  `torch/_inductor/ir.py:9314-9358`；
- value wrapper：`torch/_inductor/ir.py:10547-10607`。

## 3. TensorBox与StorageBox

`TensorBox`是张量语义wrapper；`TensorBox.create`把普通IRNode放入StorageBox。
StorageBox允许in-place替换其内部representation，例如lazy loop→ComputedBuffer。

这层indirection让：

- consumers持有同一logical tensor box；
- realization可更新storage implementation；
- view/mutation逻辑不必改所有引用。

源码中 `TensorBox.create(data)`直接返回 `TensorBox(StorageBox(data))`；
`StorageBox`的注释也明确指出它用于允许 Tensor 的原地 mutation
（`torch/_inductor/ir.py:10547-10565`）。因此这里的“同一 logical tensor”不是指
`StorageBox`有全局唯一值编号，而是多个 Python 引用可以共同观察到内部 `data` 被替换。

以 realization 为例，`StorageBox.realize()`不是返回一个全新的 box：它把自己的
`data` 从 lazy `Pointwise/Reduction/Scan/Sort`替换为 `ComputedBuffer`，然后登记 buffer
和 operation（`torch/_inductor/ir.py:10578-10607`）。这个可变间接层正是为了让已经持有
该 box 的消费者不必逐一重连。

## 4. Loops

`Loops`持有：

- device/dtype；
- iteration ranges；
- inner function；
- origin/trace。

`Loops`本身直接持有`inner_fn`；进入Scheduler后，`SchedulerNode._compute_attrs()`才执行并
提取`LoopBody`及其read/write信息（`torch/_inductor/ir.py:1057-1064`；
`torch/_inductor/scheduler.py:2232-2286`）。

`Pointwise`在output domain逐点计算；`Reduction`增加reduction domain/type。Scan、Sort、
MultiOutputReduction扩展同一方向（`torch/_inductor/ir.py:2291-2315`、
`torch/_inductor/ir.py:3091-3150`）。

源码责任可以进一步拆开：

- `Loops`只规定 `device`、`dtype`、`inner_fn` 与 `ranges`
  （`torch/_inductor/ir.py:1057-1072`）；
- `Pointwise.make_loader()`直接暴露 `inner_fn`，reduction size 为空
  （`torch/_inductor/ir.py:1219-1236`）；
- realization 后，`SchedulerNode._compute_attrs()`才对
  `ComputedBuffer/TemplateBuffer`执行 `simplify_and_reorder()`并得到 `LoopBody`
  （`torch/_inductor/scheduler.py:2232-2268`）。

这解释了为什么 lowering 时可以保留 Python callable 形式的 loop body：只有当它变成
可调度 operation 后，后端才需要把 callable 在符号索引环境中“执行”一次，抽取真正的
load/store 关系。

`Scan`并非普通Reduction的别名：它同时持有pointwise `ranges`、`scan_ranges`、
`combine_fn`、多输出`dtypes/inner_fns`与`output_index`，通过`ops.scan`生成对应输出
（`torch/_inductor/ir.py:2877-2946`）。这解释了为何前缀和/递推类计算需要单独子类：
它们共享loop协议，却需要保留有序combine语义和可能的多结果。

## 5. Index expression

一个逻辑load通常包含：

```text
buffer name
index expression
mask
indirect/indexing relation
```

broadcast dimension可映射常量0；transpose通过stride/index reorder；slice加入offset。fusion
的核心是组合producer inner expression与consumer indexing。

例如输入`x:[3,1]`、`y:[1,4]`产生`out:[3,4]`时，逻辑输出索引`(i,j)`会分别映射到
`x[i,0]`和`y[0,j]`。本系列Lab不只看shape：它从Scheduler `MemoryDep.index`中断言两个
read具有不同索引表达式，从而把“发生了broadcast”落实到寻址关系。

这里需要区分源码机制与运行观察。源码机制是：依赖抽取时，loop body 中的
`load(name, index)`经 canonicalize 后生成 `MemoryDep(name, index, ranges)`；
`store`同理生成 write dependency
（`torch/_inductor/dependencies.py:571-601`）。具体 broadcast case 中
`x[i,0]`、`y[0,j]`对应不同 `MemoryDep.index`，则是
`part4_ir_scheduler_analysis.py`实际运行后写入
`labs/artifacts/part4_ir/ir_matrix.json`的 `[R]` 级观察，不应标为只读源码即可推出的
`[S]` 事实。

## 6. Layout

`Layout`描述tensor-like output的device/dtype/size/stride/offset等metadata；它本身不等于
“一定own storage”，ownership还要结合Buffer、view与具体layout subclass判断：

- device/dtype；
- size；
- stride；
- offset；
- pinned状态等。

当前代表类：

- `FixedLayout`：stride确定；
- `FlexibleLayout`：后续可选择；
- `NoneLayout`：无普通Tensor storage；
- `NonOwningLayout`/view；
- `MultiOutputLayout`
  （`torch/_inductor/ir.py:4399-4412`;
  `torch/_inductor/ir.py:4416-4432`;
  `torch/_inductor/ir.py:4433-4448`;
  `torch/_inductor/ir.py:4726-4750`;
  `torch/_inductor/ir.py:4962-4976`;
  `torch/_inductor/ir.py:4978-4998`;
  `torch/_inductor/ir.py:5040-5085`;
  `torch/_inductor/ir.py:10144-10190`）。

`Layout.__init__()`实际保存 device、dtype、size、stride、offset 与 pinned 状态，并检查
rank、类型以及 pinned CPU 约束
（`torch/_inductor/ir.py:4416-4448`）。它描述的是“若这个值作为 tensor-like output，
其地址应如何解释”，而 storage ownership 由持有它的 `Buffer`、view subclass 或
non-owning layout 决定。于是相同 size/stride 元数据既可能属于新分配 buffer，也可能是
已有 storage 的视图。

## 7. Buffer与Operation

`Buffer`是具名storage/输出spec载体；`Operation`提供可调度operation接口。
`OperationBuffer(Buffer, Operation)`同时是具名buffer和schedulable computation。

`ComputedBuffer`包装Loops；`TemplateBuffer`包装template implementation
（`torch/_inductor/ir.py:5160-5180`;
`torch/_inductor/ir.py:5182-5194`;
`torch/_inductor/ir.py:5306-5334`;
`torch/_inductor/ir.py:5408-5435`;
`torch/_inductor/ir.py:5876-5905`;
`torch/_inductor/ir.py:5942-5964`;
`torch/_inductor/ir.py:6011-6043`;
`torch/_inductor/ir.py:6185-6193`;
`torch/_inductor/ir.py:6194-6219`;
`torch/_inductor/ir.py:6284-6305`;
`torch/_inductor/ir.py:6306-6315`）。

对应到源码：

- `Buffer`持有 `name` 与 `layout`，暴露 size/stride/offset，却默认
  `get_defining_op() -> None`
  （`torch/_inductor/ir.py:5160-5194`）；
- `Operation`持有 operation name、origin、stream 与 mempool 协议
  （`torch/_inductor/ir.py:956-987`）；
- `OperationBuffer`通过多继承声明“这个 buffer 自己就是其 defining operation”
  （`torch/_inductor/ir.py:5306-5319`）；
- `ComputedBuffer.data`必须是 `Loops`
  （`torch/_inductor/ir.py:5408-5420`）。

这些子类由图中的真实场景决定：graph input有 storage 但没有 defining operation；
lazy Pointwise有 computation 但尚无 storage name；realized loop 同时拥有两者；view 又可
引用 storage 而不增加 operation。

## 8. Input、Donated与Output

- InputBuffer代表graph input storage；
- DonatedBuffer主要表示满足alias/output条件、可供backward复用的saved-tensor输入资源；
  它不是所有input都可自动“捐赠”的通用标记；
- outputs必须满足用户/ABI layout与alias要求。

DonatedBuffer当前说明位于 `torch/_inductor/ir.py:5327-5338`。

## 9. ExternKernel

ExternKernel表示调用外部kernel/library；它仍是OperationBuffer，拥有inputs/layout/codegen
calling convention。FallbackKernel是其具体路径之一。

这类IR保留operator/library implementation语义，不被强行变成loop lambda。

## 10. TemplateBuffer与ChoiceCaller

template表示可生成专用kernel的实现。`ChoiceCaller`可产生output node并benchmark；多个choices
可由 `MultiTemplateBuffer`延迟决定
（`torch/_inductor/ir.py:6185-6193`;
`torch/_inductor/ir.py:6194-6219`;
`torch/_inductor/ir.py:6221-6250`;
`torch/_inductor/ir.py:6284-6305`;
`torch/_inductor/ir.py:6306-6315`）。

这让algorithm selection和fusion交互，而非lowering时过早固定。

## 11. Multi-output

Extern/template op可一次产生多个physical/logical outputs。`MultiOutputLayout`与`MultiOutput`
表达共享kernel调用与各output view/layout。它不同于FX output pytree或MultiOutputPattern。

## 12. Mutation与alias

IR reads/writes、mutation names、aliases与non-owning layout共同表达：

- 哪些values共享storage；
- 某operation写哪个buffer；
- 哪些views不能free；
- Scheduler如何建立依赖；
- wrapper能否in-place/reuse。

历史类名 `MutationLayoutSHOULDREMOVE`仍存在于当前源码
（`torch/_inductor/ir.py:5063-5085`），其名字本身说明这是实现遗留，不应当成理想公共
architecture层。

## 13. FX与IR非一一对应示例

```text
FX:
  a = x + 1
  b = relu(a)
  c = b.sum()

可能IR:
  a,b 保持lazy pointwise expression
  c要求reduction边界并realize ComputedBuffer
  Scheduler只获得realized operations
```

若a还被external op使用，它可能提前realize，改变operation/buffer数。

## 14. IR dependency入口

SchedulerNode从operation分析read/write dependencies，而不是读取FX args。LoopBody load/store、
Extern inputs/outputs、alias/mutation构成依赖信息。

对 `ComputedBuffer/TemplateBuffer`，真正的依赖入口在
`SchedulerNode._compute_attrs()`：它先得到 loop body，再调用
`dependencies.extract_read_writes()`或 template 自己的
`extract_read_writes()`，最后保存 read/write 集合
（`torch/_inductor/scheduler.py:2270-2285`）。`RecordLoadStore.load/store`
则把动态访问记录成 `MemoryDep`
（`torch/_inductor/dependencies.py:590-601`）。

Scheduler 的 operation 类型映射只决定创建哪一种 SchedulerNode；它不能单独证明依赖如何
构造。依赖边的证据必须追到上述 read/write 抽取路径。

## 15. 复杂度

- 构造一个 lazy wrapper 通常是局部常数工作，但整个 expression 的语义大小可能累积在
  `inner_fn`闭包中；
- realization 本身创建并登记 `ComputedBuffer`近似为 `O(1)`，真正展开/分析
  `inner_fn`主要发生在 SchedulerNode 的 loop-body/read-write 抽取；
- index simplification受SymPy expression复杂度影响；
- alias/layout propagation可造成额外copies；
- IR object数与FX Node数没有固定比例。

设 registered operation 数为 `N_ir`，每个 operation 被抽取的 loop body 操作数为
`B_i`，产生的 dependency 数为 `D_i`，则不计符号化简内部代价时，依赖抽取骨架近似：

```text
O(Σ_i (B_i + D_i))
```

SymPy canonicalization、loop reorder 与 template 特有分析另计，不能被压成
`O(N_fx)`或`O(N_ir)`。

## 16. 阅读IR dump

按顺序看：

1. logical size/dtype/device；
2. layout stride/offset；
3. lazy/realized；
4. loop ranges与inner loads；
5. buffer name；
6. operation type；
7. origins；
8. aliases/mutations；
9. template/extern choice。

## 17. 已验证 Lab

### 17.1 命令

```powershell
python wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/labs/part4_ir_scheduler_analysis.py `
  --output-dir wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/labs/artifacts/part4_ir
```

### 17.2 覆盖矩阵

脚本实际执行`make_fx → GraphLowering.run → Scheduler`，并对每个case保存FX、operation、
buffer、layout、read/write index与output：

| case | 已观察结果 |
|---|---|
| elementwise | realized buffer内部是`Pointwise` |
| broadcast | 两个input read具有不同`MemoryDep.index` |
| transpose | output是`ReinterpretView`，没有新增Scheduler operation |
| transpose + contiguous clone | 产生owning `ComputedBuffer` |
| reduction | buffer内部是`Reduction` |
| matmul | operation属于`ExternKernel*`路径 |

stdout中的六项`*_observed`必须全为`True`，否则脚本直接`AssertionError`。关键产物是：

- `labs/artifacts/part4_ir/ir_matrix.json`：完整case矩阵；
- `labs/artifacts/part4_ir/environment.json`：runtime与源码基线；
- `labs/artifacts/part4_ir/manifest.json`：命令、schema和证据边界。

### 17.3 证据边界

这是`[R]`级的GraphLowering/Scheduler运行时内部观察，不是手写示意IR。transpose与clone
对照在当前case中观察到view/copy边界。

该固定case不能推出所有view都永不materialize。

脚本有意不进入native codegen，所以没有证明C++/Triton kernel执行或性能。

## 学习顺序

- 上一篇：[[17_fx_lowering_to_inductor_ir]]
- 下一篇：[[19_buffer_liveness_memory_planning_and_reuse]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[17_fx_lowering_to_inductor_ir]]
- [[19_buffer_liveness_memory_planning_and_reuse]]
- [[20_scheduler_dependency_graph_fusion_and_ordering]]
- [[lowering_analysis]]
