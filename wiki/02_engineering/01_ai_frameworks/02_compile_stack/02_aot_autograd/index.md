# 02 · AOTAutograd — 目录索引

> 介于 Dynamo 与 Inductor 之间:functionalization、前/反向联合图(joint graph)生成与 min-cut partition。
> 知识分层:overview(本索引)→ quick start → deep dive(约定见 [[01_ai_frameworks/index]])。
> 最后更新: 2026-07-27

---

## 模块概述

AOTAutograd(ahead-of-time autograd)在编译期就把一段可微计算的**前向与反向拆成两张独立的 FX 图**,而不像 eager 那样在运行时由 autograd 引擎边执行边构建反向。这样做的意义在于:反向计算从此成为一张**显式的图**,可以和前向图一起接受图级优化(算子融合、重计算换显存、decomposition 等);否则反向只是运行时一连串 Tensor 调用,编译器无从插手。它解决的核心问题,就是「让梯度计算也能被编译优化」。

在 `torch.compile` 三段管线里它处于中段:**Dynamo** 从 Python 字节码捕获出带 guards 的前向 FX 图;**AOTAutograd** 接过这张前向图,ahead-of-time 地 trace 出前向 + 反向并加以切分;**Inductor**(或其他后端)再对切分结果做 lowering 与 codegen。一句话概括各自产出:Dynamo 产「捕获到的前向图」,AOTAutograd 产「切分好的前向图 + 反向图」,Inductor 产「可执行 kernel」。

为此 AOTAutograd 承担三大职责:**functionalization**——把原地写(in-place / mutation)、view 等改写成纯函数式 ATen 算子,消除别名与副作用,后端才好优化;**joint graph**——ahead-of-time 同时 trace 前向与反向,合成一张「前 + 反向联合图」;**partition**——用 partitioner 把联合图切成独立的 forward / backward 两张图,并决定哪些中间张量 save 给反向、哪些在反向里重计算(min-cut vs default,详见 quick start)。

## 当前系统化主线

下列五篇组成按当前固定源码基线核验的课程主线；专题页继续保留各自独有角色：

- [[graph_effects_alias_mutation_and_order_analysis]] — functionalization之前必须理解的alias、mutation与effect
- [[graph_normalization_decomposition_and_functionalization_analysis]] — functional ATen、decomposition、synthetic base与规范化顺序
- [[aotautograd_joint_forward_backward_graphs_analysis]] — joint graph 如何生成并切成两张互不持有对方 `Node` 的 fw/bw 图
- [[saved_tensors_recompute_and_runtime_abi_analysis]] — saved tensor ABI、bw placeholder、重计算节点复制与运行时拼接
- [[graph_stage_boundaries_identity_and_provenance_analysis]] — 各阶段的图身份、边界、metadata 与 provenance

### 课程主线与专题参考分工

| 页面 | 保留角色 | 当前审计口径 |
|---|---|---|
| [[aot_autograd_quickstart]] | API quick start与日志/config入口 | 示例需看代码块是否current-run；未统一默认视为未复跑 |
| [[dispatch_modes_proxytensor_faketensor_analysis]] | make_fx 捕获所依赖的 ProxyTensor/FakeTensor dispatch-mode 专题 | 2026-07-30 从原 `aotautograd_analysis` §13 独立成页，逐字保留 |
| [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]] | 当前系统课程与可执行Lab | 源码、runtime和mock证据分级 |

> `aotautograd_analysis`（1460 行全量 reference）与 `fx_graph_construction_and_transformation_analysis`（AOT 特有残留页）已于 2026-07-30（kb-reorg P4 Task 8）判重删除：joint/fw/bw 构图并入 [[aotautograd_joint_forward_backward_graphs_analysis]]，saved/recompute/runtime ABI 并入 [[saved_tensors_recompute_and_runtime_abi_analysis]]，ProxyTensor/FakeTensor 独立为 [[dispatch_modes_proxytensor_faketensor_analysis]]；逐节台账见对应 commit 与 changelog。

另见[[02_compile_stack/06_compile_cache/index]]：AOTAutograd cache命中可能复用functionalization、
joint/partition及其编译结果；必须先确认cache层级，才能解释为什么某次运行没有重新打印
fw/bw图。另见[[02_compile_stack/03_graph_ir_and_passes/index]]：FX `Graph`/`Node`数据模型、
PatternExpr/PatternMatcherPass、DCE与稳定拓扑排序、rewrite合法性与复杂度的当前系统主线，
本模块的joint graph/partition建立在这套底座之上。

## 页面列表(按层次)

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[aot_autograd_quickstart]] | **quick start** | 看前/反向图:`backend="aot_eager"` + `TORCH_LOGS=aot_graphs`;看联合图 `aot_joint_graph`;partitioner(min-cut vs default)与重计算;`aot_function` 最小用法;`AOT_PARTITIONER_DEBUG`/activation_memory_budget |
| [[aotautograd_joint_forward_backward_graphs_analysis]] | deep dive | metadata analysis、joint graph 构造(primals/tangents)、partition 如何抽取 fresh fw/bw Graph、输出/输入 ABI 分层 |
| [[saved_tensors_recompute_and_runtime_abi_analysis]] | deep dive | saved value 分类、min-cut rematerialization、recompute 节点复制与 reorder、运行时 autograd.Function 拼接 |
| [[dispatch_modes_proxytensor_faketensor_analysis]] | deep dive(专题) | `__torch_function__`/`__torch_dispatch__`/ProxyTensor/FakeTensor 四层分工，make_fx 怎样协同两套抽象执行状态捕获联合图 |
| [[activation_checkpoint_recompute_and_compile_analysis]] | deep dive(专题) | 用户层 `torch.utils.checkpoint`(reentrant/non-reentrant、Selective AC policy)如何与 partitioner 的 save/recompute 选择叠加；2026-07-30 迁入,与 [[saved_tensors_recompute_and_runtime_abi_analysis]] 互指划界(用户 API/策略层 vs partitioner 源码/runtime ABI 层) |

> joint graph 上的优化 pass 见 [[joint_graph_passes_guide]](实现于 Inductor `fx_passes/joint_graph.py`)。

---

## Related Pages

- [[19_torch_compile_end_to_end/00_torch_compile_end_to_end_index]] — 编号化端到端课程：AOTAutograd 位于卷 C，并由卷 B、D 衔接捕获与运行时
- [[01_eager_runtime/05_autograd_engine/index]] — **eager 对应物**:运行时动态磁带 + C++ 引擎(`.backward()`);本编译期模块的 eager 侧根源,两者对照见该页表格
- [[02_compile_stack/01_dynamo/index]] — 上游:图捕获
- [[02_compile_stack/04_inductor/index]] — 下游:lowering 与 codegen
- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]] — 图编译系统化主线
- [[02_compile_stack/06_compile_cache/index]] — AOTAutograd cache与其他编译缓存的边界
- [[01_ai_frameworks/index]] — 本域总索引
