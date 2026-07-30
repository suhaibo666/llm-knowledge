# 01 · 图 IR 的动机与分类

> 前置：无
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. 先回答“为什么要有图”

Eager Python 的优势是表达能力：控制流、对象、动态数据结构和任意副作用都能即时执行。
它的限制也来自同一点——单个算子被调用时，编译器通常只知道当前操作，看不到后续还会
读取什么、哪个中间值最终无用、两个操作能否融合、反向将需要哪些激活。

图 IR 把一次计算的部分语义冻结成可检查的数据结构，使以下问题可计算：

- producer 与 consumer 是谁；
- 哪些值必须保存，哪些可以重算；
- 哪些节点无用户且无副作用；
- 哪些节点可以重排、分区或融合；
- 输入在哪些 shape/dtype/device 条件下可复用；
- 一个生成 kernel 来自哪些源操作。

图不是为了“把 Python 画出来”，而是为了选择一个**比 Python 更受约束、但适合某类分析**
的表示层。约束越强，优化越容易；但能表达的程序也越少。因此 PyTorch 不只需要一张图。

## 2. 图的四个基本问题

看到任何“图”时先问：

| 问题 | 含义 |
|---|---|
| Node 是什么 | Python 调用、FX callsite、autograd Function、IR Buffer、kernel group，还是设备任务 |
| Edge 是什么 | Tensor 数据依赖、梯度 next edge、buffer read/write、控制/effect 顺序，还是流事件 |
| Value 是什么 | 运行时 Tensor、Node 引用、FakeTensor、saved activation、storage name，还是设备地址 |
| Order 是什么 | 源程序顺序、拓扑顺序、反向执行顺序、调度顺序，还是捕获回放顺序 |

同一个词在不同层的答案不同。若不先分类，“逆图”“dead node”“正反向连接”“图重排”
都会产生歧义。

## 3. PyTorch 中常见的六类“图”

### 3.1 Eager autograd tape

Eager 前向执行真实算子，同时根据需要创建 autograd `Node`，其 edge 指向梯度传播的
下一目标。它是运行时构造的反向依赖结构，不是 FX `Graph`，也不保存一份可直接重写的
Python forward program。

可观察入口是 Tensor 的 `grad_fn` 与其 `next_functions`。这张图中的 Node 表示反向函数，
而不是 `aten.add` 之类的 forward callsite。它属于 [[01_eager_runtime/05_autograd_engine/index]] 的运行时模型。

### 3.2 FX program graph

FX `Graph` 是一串共同构成有效 Python 函数的 `Node`
（`torch/fx/graph.py:1397-1443`）。普通 Node 的六种 opcode 是：
`placeholder`、`get_attr`、`call_function`、`call_module`、`call_method`、`output`
（`torch/fx/node.py:258-284`）。

FX 同时有两种结构：

- 双向链表保存程序顺序；
- `args/kwargs` 中的 Node 引用与反向 `users` 保存 use-def。

它通常是 DAG，但“图顺序”和“数据依赖”不是一个字段。详见
[[10_fx_graph_core_data_model_analysis]]。

#### 3.2.1 Program graph 与 dataflow graph

这里的 **program graph** 是完整 FX IR：除 use-def 外，还保存 placeholder/output、
`get_attr`、调用种类、节点链表顺序、target 和 metadata。**Dataflow graph**通常是从
program graph 投影出的 producer→consumer 关系；它只回答“这个值被谁使用”，不自动包含
两个没有 Tensor 依赖的 effect 顺序、GraphModule state 或用户签名。

因此二者不是必须各自拥有一套 Node 的两张运行时对象。对同一个 `fx.Graph`，可以：

```text
program view: 依次读取 Graph.nodes
dataflow view: 对每个 Node 展开 args/kwargs 中的 Node 引用，或反查 users
```

FX 选择同时保存这两种结构，正是因为只保存 dataflow adjacency 会丢掉可重新生成 Python
程序所需的顺序和调用语义；只保存线性程序又会让 producer/consumer 分析反复扫描。

### 3.3 AOT joint、forward 与 backward FX graph

AOTAutograd 先捕获 joint FX 图，再由 partitioner 创建两张新 `Graph`。提取函数会创建
fresh graph、fresh placeholders 和 old→new 环境映射；该 helper 返回两个 fresh
`fx.Graph`，caller 才包装为 GraphModule
（`torch/_functorch/partitioners.py:625-705`;
`torch/_functorch/partitioners.py:1577-1578`）。

因此：

- joint Node、fw Node、bw Node 是不同对象；
- fw 与 bw 没有跨 Graph Node edge；
- saved value 由 fw output + bw placeholder + runtime ABI 表达；
- recompute 是 forward 节点被复制到 bw，不是特殊 opcode。

详见 [[11_aotautograd_joint_forward_backward_graphs_analysis]] 与
[[12_saved_tensors_recompute_and_runtime_abi_analysis]]。

### 3.4 Inductor IR

GraphLowering 作为 FX Interpreter 执行 lowering。环境值可能是 lazy
`TensorBox/StorageBox`、view、constant、`ComputedBuffer`、template 或 extern operation。
它不是 FX 图的同构副本（`torch/_inductor/graph.py:386-386`;
`torch/_inductor/graph.py:1925-2000`）。

loop-level `Pointwise`/`Reduction` 是 fusion-friendly 核心，但不是唯一 IR；extern、
template、multi-output 等有不同表示。详见 [[10_fx_lowering_to_inductor_ir_analysis]]。

### 3.5 Scheduler dependency graph

Scheduler 从已经物化并注册的 operation 创建 SchedulerNode，然后根据 buffer reads/writes、
alias、mutation rename、weak ordering 等建立依赖
（`torch/_inductor/scheduler.py:4641-4651`;
`torch/_inductor/scheduler.py:4689-4693`;
`torch/_inductor/scheduler.py:4731-4745`;
`torch/_inductor/scheduler.py:4746-4761`;
`torch/_inductor/scheduler.py:4847-4865`;
`torch/_inductor/scheduler.py:4874-4898`;
`torch/_inductor/scheduler.py:4904-4918`）。

它的边不同于 FX users：

- 一个 FX Node 可能没有对应的 SchedulerNode；
- 多个 FX origin 可被 fusion 到一个 Scheduler group；
- view/alias/mutation 会改变 buffer 依赖；
- stream、mempool、collective 等可引入额外顺序限制。

详见 [[13_scheduler_dependency_graph_fusion_and_ordering_analysis]]。

### 3.6 Runtime CUDA Graph

CUDA Graph 捕获的是一段已经准备好的设备工作和内存地址关系，以便之后低开销回放。
它不是编译器 program IR，也不等价于 Scheduler graph。它关心固定地址、stream capture、
重放和内存池；FX 关心程序与值依赖。运行时图见 [[03_runtime_graphs/index]]。

## 源码跟读：为什么这些“图”不能合并成一个通用 Graph 类

分类不是术语偏好，而是由每一层必须回答的问题决定。最直接的源码判据不是类名里有没有
`Graph`，而是“节点保存什么状态、边指向什么对象、谁创建下一层表示”。

### 1. Eager autograd 的边指向 backward function 的某个输入槽

autograd 的 `Edge` 是显式结构，保存 `intrusive_ptr<Node> function` 和 `input_nr`
（`torch/csrc/autograd/edge.h:13-18`; `torch/csrc/autograd/edge.h:34-36`）。autograd
`Node` 构造时接收 `next_edges`，并据此更新拓扑编号
（`torch/csrc/autograd/node.h:112-139`）。

这套结构为反向执行器服务：一条边不仅要说“去哪个 backward function”，还要说“进入它的
第几个输入槽”。FX 的 `consumer.args` 已经包含调用位置，因此不需要复用这类 Edge。若强行
统一，两边都会携带对自己无意义的字段：autograd 不需要 FX 的 `target/kwargs/GraphModule`
代码生成语义，FX 也不需要 autograd 的 `input_nr` 和侵入式生命周期。

### 2. FX 的 Node 同时是 callsite 和值，边嵌在调用参数里

FX `Node` 的 opcode 决定 placeholder、state lookup、function/module/method call 和 output
语义（`torch/fx/node.py:258-284`）；其 `_input_nodes` 与 `users` 是 distinct
producer/consumer 索引，而真实位置和重复 use 仍保留在 `args/kwargs`
（`torch/fx/node.py:286-304`）。

所以 FX 选择的是“可重新生成 Python 调用”的 program IR。它不仅回答依赖，还必须保存
target、调用约定、Graph 顺序与 root state 的寻址方式。这些字段来自后续 pass 需要改写并
重新执行程序，而不是来自“所有图都应该有节点和边”的抽象美感。

### 3. AOT partition 不是在 joint Graph 上打两个视图标签，而是复制出新图

`_extract_graph_with_inputs_outputs` 明确创建 `new_graph = fx.Graph()` 和
`env: dict[old_node, new_node]`，再按指定 inputs 创建 fresh placeholder
（`torch/_functorch/partitioners.py:514-533`;
`torch/_functorch/partitioners.py:544-548`）。上层 `_extract_fwd_bwd_modules` 先从 joint
module 分离 forward/backward outputs，再收集 primal 与 tangent placeholders
（`torch/_functorch/partitioners.py:1343-1375`）。

因此 joint、fw、bw 的关系是：

```text
joint Graph Node
      │ old→new remap + node copy
      ├──────────────► fresh fw Graph Node
      └──────────────► fresh bw Graph Node

fw output slot ── runtime ABI ──► bw placeholder slot
```

最后一行是跨图值传递约定，不是 `fw_node.users` 中出现一个 bw Node。`Node.graph` ownership
要求恰恰阻止这种跨 Graph use-def。之所以这样设计，是为了让 fw/bw 可以独立编译、持有
各自合法的拓扑和签名；代价是必须显式维护 saved-value 的位置 ABI 与来源映射。

### 4. Inductor lowering 执行 FX Node，但产物不是 FX Node 的一对一副本

`GraphLowering` 直接继承 `torch.fx.Interpreter`，构造参数包括 FX `GraphModule`、
`ShapeEnv`、后端模式以及 graph-wide lowering/codegen 状态
（`torch/_inductor/graph.py:386-417`）。它的 `run_node` 接收一个 `torch.fx.Node`，
执行对应 lowering，并对产出的 IR 应用布局等决策
（`torch/_inductor/graph.py:1925-1955`）。

这里的“执行”是解释 IR 并建立下一层表示，不是运行原模型的真实 eager Tensor 计算。一个
FX Node 可能只产生 lazy 表达式、view 或常量，也可能触发 extern/template/multi-output
operation；反过来，一个后端 operation 也可以携带多个 FX origins。因此不能假设
`FX Node ↔ Inductor operation` 是双射。

### 5. Scheduler Node 只为需要调度的 realized operation 建立

Scheduler 的工厂按 Inductor operation 类型选择节点：no-op、computed/template buffer、
extern kernel 分别进入不同 `BaseSchedulerNode` 子类，未支持类型直接报错
（`torch/_inductor/scheduler.py:4641-4651`）。

这说明 Scheduler 图的节点集合由“需要安排执行、融合和内存生存期的 operation”决定，
而不是把所有 FX Node 再复制一遍。view、常量折叠、lazy 内联等没有独立执行体的值不必
拥有同构 SchedulerNode；buffer read/write、mutation、alias 和 effect 才是这一层建边的
依据。

### 源码给出的统一边界

这些实现共享“有节点、有依赖、需要顺序”的外观，但没有共享一种足够具体的运行语义。
真正稳定的统一框架是下面五问，而不是统一基类：

| 问题 | autograd | FX | AOT fw/bw | Inductor/Scheduler |
|---|---|---|---|---|
| 谁创建它 | eager forward | tracer/capture | partitioner copy | lowering / scheduler |
| Node 代表什么 | backward function | callsite/value | 独立 program callsite | IR value、operation、group |
| 依赖放哪里 | explicit `Edge` | args/kwargs + users | 各图内 use-def + 跨图 ABI | reads/writes/effect deps |
| 顺序为何存在 | backward ready order | 可生成程序的拓扑顺序 | 独立 fw/bw 拓扑 | 可执行调度顺序 |
| 何时结束生命周期 | backward graph 释放策略 | GraphModule/编译阶段持有 | 编译 artifact/runtime 持有 | codegen 后由 artifact 接管 |

源码并不能证明“任何 PyTorch 图都一定属于这几类”；它证明的是本系列讨论的这些具体实现
拥有不同的数据结构与转换边界。后续各篇都以固定版本源码中的真实类型和调用链为准。

## 4. 一张对照表

| 图 | 构造时间 | Node | Edge/依赖 | 主要用途 |
|---|---|---|---|---|
| eager autograd tape | forward 运行时 | backward Function | gradient next edge | 立即求导 |
| FX program graph | trace/capture 时 | 程序 callsite/value | args/kwargs use-def | 分析与改写程序 |
| AOT joint graph | 编译期 capture | forward/backward callsite | joint 数据依赖 | 选择 save/recompute |
| AOT fw/bw graph | partition 时 | fresh copied FX Node | 各自图内 use-def | 独立后端编译 |
| Inductor IR | lowering 时 | value/layout/buffer/operation | lazy composition 与 reads/writes | 表达实现与寻址 |
| Scheduler graph | 调度时 | realized operation/group | buffer/effect/order deps | 排序、融合、liveness |
| CUDA Graph | 运行时 capture | 设备操作节点 | stream/event/memory relation | 低开销回放 |

## 5. DAG、循环和控制流

### 5.1 普通 FX 图为什么常是 DAG

Node 的数据参数只能引用已经创建的 producer；`Graph.lint()`检查 producer-before-consumer
拓扑顺序和 ownership（`torch/fx/graph.py:2610-2649`）。普通 use-def 因而不能形成数据环。

### 5.2 循环如何表达

循环不一定意味着外层 FX use-def 有环：

- Python 循环可能在 tracing 时展开；
- 数据相关循环可能导致 graph break；
- `while_loop` 可作为 HigherOrderOperator Node，循环体放在嵌套子图；
- Inductor loop IR 表达的是迭代域，不是有环的 FX Node 指针。

所以“DAG 无法表达循环”并不精确。更准确的说法是：普通 FX use-def 是无环的，循环语义
可由展开、opaque call 或嵌套 graph region 表达。

### 5.3 控制流为何需要嵌套图

`torch.cond` 的外层 Node 代表一个 HigherOrderOperator，true/false branch 作为可捕获函数或
子图传入；FakeTensor 路径要求两分支输出 pytree 结构一致
（`torch/_higher_order_ops/cond.py:388-415`）。外层 DAG 因此通过“节点持有子程序”表达分支。

## 6. FX graph 与 autograd graph 的边界

最常见误解是把 FX `Node` 与 autograd `Node` 当成同一种对象：

```text
FX:
  Node add → Node relu
  表示 forward 程序值被使用

eager autograd:
  ReluBackward → AddBackward
  表示梯度应传播到哪个 backward Function
```

即使两者都能画成 DAG，它们的方向、节点类型、构造时机、生命周期和可编辑接口都不同。
AOTAutograd 的价值正是把原本由 eager autograd 动态执行的梯度计算，再捕获成可交给
后端的普通 FX program graph。

## 7. “反向图”与“逆序遍历”也不是一回事

- AOT backward graph：一张包含梯度计算的独立 FX Graph；
- reverse topological traversal：在同一图中从后向前访问，用于 DCE/liveness；
- PatternMatcher reverse order：把候选 root 按 FX 链表顺序逆序处理；
- reverse edge：某个算法显式构造的反向邻接关系。

这些概念没有自动联系。看到“逆图”应追问它指结构、遍历方向，还是 autograd backward。

## 8. 复杂度的第一层直觉

若图有 `V` 个节点、`E` 个依赖引用：

- 线性遍历通常 `O(V)`；
- use-def 构建/更新通常与被访问参数和边数相关，为 `O(V+E)`；
- DFS/Kahn 拓扑排序为 `O(V+E)`，若额外稳定排序会叠加排序成本；
- 保存每个节点的传递 ancestors 可达到 `Θ(V²)`；
- naive 子图 pattern 是候选数 × pattern 大小，实际 matcher 会用 root index 降低候选；
- fusion 若退化为 all-pairs/重复图可达性检查，最坏会明显超线性。

这里的 `V/E` 必须绑定具体图类型：FX 的边数、Scheduler dependency 数和 autograd next edge
数不是同一个量。

## 9. 已验证 Lab：同一函数的三种观察面

从知识库根目录运行：

```powershell
python -B tools\labs_torch_compile\part1_graph_taxonomy.py
python -B tools\labs_torch_compile\series_artifact_bundle.py `
  --output-dir tools\labs_torch_compile\artifacts\end_to_end
```

最小输入是一个可求导的 `add→relu→sum`函数；正例同时观察 eager backward tape 与 FX
forward program graph。边界例是二者虽然来自同一函数，却没有共享 Node/edge；贯穿
bundle 再通过自定义 Dynamo backend 保存真正的 compile-time FX 与 guards，避免把
`grad_fn`误称为编译图。

在 PyTorch `2.9.1+cpu` 实测：

```text
eager_grad_fn=SumBackward0
eager_next=ReluBackward0
fx_ops=placeholder,call_function,call_method,call_method,output
fx_call_targets=add,relu,sum
```

解释：

- `grad_fn`/`next_functions`展示 backward Function；
- FX graph 展示 forward callsite；
- 两者可追溯到同一用户函数，但不是同一张图，也不共享 Node。

持久 artifact 位于：

- `tools/labs_torch_compile/artifacts/end_to_end/dynamo_fx.py`；
- `tools/labs_torch_compile/artifacts/end_to_end/dynamo_guards.txt`；
- `tools/labs_torch_compile/artifacts/end_to_end/stage_node_mapping.json`。

源码结论仍绑定页头 pinned SHA；上述文件绑定 Lab runtime。完整环境和实际命令保存在
`tools/labs_torch_compile/artifacts/end_to_end/environment.json`及 [`tools/labs_torch_compile/README.md`](tools/labs_torch_compile/README.md)。

## 10. 本篇检查清单

遇到“图”先回答：

1. 这是什么阶段的图？
2. Node 的运行语义是什么？
3. edge 是显式对象、Node 引用、buffer 名称，还是隐藏 effect？
4. 节点列表顺序是否等于依赖顺序？
5. 这张图是否拥有 nested graph？
6. 它在何时构造、何时销毁？
7. 是否存在跨图 ABI，但不存在跨图 edge？

## 学习顺序

- 上一篇：[[00_pytorch_graph_series_index]]
- 下一篇：[[10_fx_graph_core_data_model_analysis]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[10_fx_graph_core_data_model_analysis]]
- [[01_eager_runtime/05_autograd_engine/index]]
- [[11_aotautograd_joint_forward_backward_graphs_analysis]]
- [[10_fx_lowering_to_inductor_ir_analysis]]
- [[13_scheduler_dependency_graph_fusion_and_ordering_analysis]]
- [[03_runtime_graphs/index]]
