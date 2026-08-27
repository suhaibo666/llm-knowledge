---
title: "03 · Graph IR 与 Passes — 目录索引"
---

# 03 · Graph IR 与 Passes — 目录索引

> FX 图的数据模型、捕获前端、改图原语、PatternMatcher 与 pass 安全改写机制；不含 AOTAutograd
> joint/正反向分图(见 [[02_compile_stack/02_aot_autograd/index]])、Inductor IR/scheduler/codegen
> (见 [[02_compile_stack/04_inductor/index]])。
> 知识分层:overview(本索引)→ deep dive(约定见 [[01_pytorch/index]])。
> 最后更新: 2026-07-30(P4 Task 7 全部四组归一收官:C 卷 21 篇中的 12 篇迁入并去前缀重命名,
> 吸收 4 篇旧页的独有内容后其中 2 篇删除、1 篇瘦身保留、1 篇小幅划界)

---

## 页面列表(按层次)

> **段位与阅读顺序**（kb-reorg P4 Task 9.5，2026-07-30）：本目录没有 quickstart，不占用段 0。段 1（10-19）介绍 IR 基础以及如何获得这张图（数据模型→值/元数据→effect/alias→结构化输出/HOP→捕获前端→规范化，即 C02/C03/C05/C06/C07/C08 的原始顺序）；段 2（20-29）介绍 pass 机制（节点身份/改图原语/pattern matcher/DCE/pass 流水线/合法性校验，即 C11-C16 的原始顺序），所有内容都建立在段 1 的 IR 基础之上。

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[10_fx_graph_core_data_model_analysis]] | deep dive(C02,段 1) | `Graph`/`Node`/`GraphModule` 五个核心对象、侵入式双向链表、`args/kwargs` 与 `users` 反向邻接、lint/recompile |
| [[11_graph_values_metadata_and_signatures_analysis]] | deep dive(C03,段 1) | Node 引用、`meta["val"]`、pytree 结构化输出与三类 graph signature |
| [[12_graph_effects_alias_mutation_and_order_analysis]] | deep dive(C05,段 1) | 数据边之外的 alias、mutation、functionalization 与 effect 顺序 |
| [[13_structured_outputs_higher_order_and_nested_graphs_analysis]] | deep dive(C06,段 1) | 多输出、HOP(`cond`/`while_loop`/`map`)与 nested GraphModule 如何扩展普通 DAG |
| [[14_graph_capture_frontends_and_tracing_analysis]] | deep dive(C07,段 1) | `symbolic_trace`、`make_fx`、Dynamo、`export` 四种捕获前端为何产生不同图 |
| [[15_graph_normalization_decomposition_and_functionalization_analysis]] | deep dive(C08,段 1) | schema normalization、decomposition 与 functionalization 为何必须分层 |
| [[20_graph_stage_boundaries_identity_and_provenance_analysis]] | deep dive(C11,段 2) | Node identity 跨阶段(capture→AOT→Inductor)断开后如何维持 provenance |
| [[21_fx_graph_editing_primitives_and_invariants_analysis]] | deep dive(C12,段 2) | replace/erase/copy/lint/recompile 组成的安全改图事务、10 项检查清单 |
| [[22_pattern_expression_and_matcher_engine_analysis]] | deep dive(C13,段 2) | PatternExpr AST、候选桶索引、逆序 matcher、三类 Entry 与序列化 pattern |
| [[23_dead_code_topology_and_effect_order_analysis]] | deep dive(C14,段 2) | FX DCE/Scheduler DCE、stable topological sort、拓扑正确 ≠ effect 正确 |
| [[24_graph_pass_pipeline_ordering_and_fixpoint_analysis]] | deep dive(C15,段 2) | pass stage、注册顺序、迭代与 fixpoint 如何决定改写结果 |
| [[25_graph_rewrite_legality_validation_and_complexity_analysis]] | deep dive(C16,段 2) | 结构命中后 shape/dtype/alias/autograd 的合法性验证、全链路复杂度 |

> 12 篇按 [[courses/torch_compile_end_to_end]] Part I–III 的固定顺序组织
> (Part IV 的 C17–C21 属 Inductor IR/Scheduler/Codegen,随 Task 8 迁入 `04_inductor/`)。课程化
> 阅读顺序、前置依赖表与配套 Demo 见该系列索引 §"四部分知识地图"与 §"每篇前置依赖与学习成果"。

## 与旧页的关系（P4 Task 7 四组判重归一，逐组 commit 台账见各自 commit message）

- **组 1**:`02_aot_autograd/fx_graph_construction_and_transformation_analysis.md`
  (2026-07-23 综合报告快照,601 行)的 FX 数据模型、PatternExpr/PatternMatcherPass、DCE、
  稳定拓扑排序、合法性/复杂度部分已核实并入上表对应页；该页已精简为 269 行，目前仅保留
  AOTAutograd joint→fw/bw 构图与 saved-tensor/recompute 专题,留给 Task 8 与 C09/C10 归一。
- **组 2**:`04_inductor/fx_pass_optimization_methodology.md`(349 行)+
  `04_inductor/torch_upstream_pass_deepdive.md`(232 行)判重后**均删除**,独有事实(八阶段
  placement 细节缺口、`fwd_only`/`joint_fwd_bwd`、`GraphTransformObserver`、
  `GroupBatchFusionBase`、四则 pass 注册取证式脚注、跨框架方法论对照)分别并入 C13/C15/C16。
- **组 3**:`04_inductor/decomposition_passes_guide.md`(163 行)vs C08 判重后**保留双页**——
  guide 采用开发者 API/checklist 视角，与 C08 对捕获期机制的说明体裁不同，独有内容 >50%。
- **组 4**:`02_compile_stack/01_dynamo/control_flow_capture_analysis.md`(204 行)vs C06
  判重后**基本保留**,只在 `torch.cond` 的 `FakeTensorMode`/`ProxyTorchDispatchMode` 重叠段
  收缩为相互引用（该页介绍 Dynamo 捕获前端，C06 介绍 IR 层结构，体裁不同，几乎没有重叠）。

---

## 附录：跨图类型对照（源自已删除的课程页 C01，逐字保留）

> 出处：原 19 号课程 `01_graph_ir_motivation_and_taxonomy.md`（见 git `e5cc60a`）。各类型的机制细节分别由本目录及关联页面承载；此处保留其**跨类型综合视图**——功能树中没有天然“同时介绍六类图”的页面，因此将这两张对照表作为附录保留。

### 四个基本问题(看任何"图"先问)

看到任何“图”时先问：

| 问题 | 含义 |
|---|---|
| Node 是什么 | Python 调用、FX callsite、autograd Function、IR Buffer、kernel group，还是设备任务 |
| Edge 是什么 | Tensor 数据依赖、梯度 next edge、buffer read/write、控制/effect 顺序，还是流事件 |
| Value 是什么 | 运行时 Tensor、Node 引用、FakeTensor、saved activation、storage name，还是设备地址 |
| Order 是什么 | 源程序顺序、拓扑顺序、反向执行顺序、调度顺序，还是捕获回放顺序 |

同一个词在不同层的答案不同。若不先分类，“逆图”“dead node”“正反向连接”“图重排”
都会产生歧义。

### 六类图的统一边界(五问对照)

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

### 七行速查对照表

| 图 | 构造时间 | Node | Edge/依赖 | 主要用途 |
|---|---|---|---|---|
| eager autograd tape | forward 运行时 | backward Function | gradient next edge | 立即求导 |
| FX program graph | trace/capture 时 | 程序 callsite/value | args/kwargs use-def | 分析与改写程序 |
| AOT joint graph | 编译期 capture | forward/backward callsite | joint 数据依赖 | 选择 save/recompute |
| AOT fw/bw graph | partition 时 | fresh copied FX Node | 各自图内 use-def | 独立后端编译 |
| Inductor IR | lowering 时 | value/layout/buffer/operation | lazy composition 与 reads/writes | 表达实现与寻址 |
| Scheduler graph | 调度时 | realized operation/group | buffer/effect/order deps | 排序、融合、liveness |
| CUDA Graph | 运行时 capture | 设备操作节点 | stream/event/memory relation | 低开销回放 |

## 关联域

- [[courses/torch_compile_end_to_end]] — 编号化课程主线(C01 动机与 C17–C21 Inductor IR 不在本目录),对应卷 C §5 部分行
- [[02_compile_stack/01_dynamo/index]] — 上游:Dynamo 捕获出 guarded FX graph 交给本目录的捕获/规范化机制
- [[02_compile_stack/02_aot_autograd/index]] — 下游承接:joint graph、正反向分图与 recompute
- [[02_compile_stack/04_inductor/index]] — 下游承接:FX lowering 为 Inductor IR、pass 阶段 guide 与 scheduler
- [[04_export_and_distributed/01_fx_export_extensibility/index]] — 另一条使用场景:`export`/自定义 op 对 FX IR 的复用
- [[01_pytorch/index]] — 本域总索引
