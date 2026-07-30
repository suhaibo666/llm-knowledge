# 03 · Graph IR 与 Passes — 目录索引

> FX 图的数据模型、捕获前端、改图原语、PatternMatcher 与 pass 安全改写机制;不含 AOTAutograd
> joint/正反向分图(见 [[02_compile_stack/02_aot_autograd/index]])、Inductor IR/scheduler/codegen
> (见 [[02_compile_stack/04_inductor/index]])。
> 知识分层:overview(本索引)→ deep dive(约定见 [[01_ai_frameworks/index]])。
> 最后更新: 2026-07-30(P4 Task 7 全部四组归一收官:C 卷 21 篇中的 12 篇迁入并去前缀重命名,
> 吸收 4 篇旧页的独有内容后其中 2 篇删除、1 篇瘦身保留、1 篇小幅划界)

---

## 页面列表(按层次)

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[fx_graph_core_data_model_analysis]] | deep dive(C02) | `Graph`/`Node`/`GraphModule` 五个核心对象、侵入式双向链表、`args/kwargs` 与 `users` 反向邻接、lint/recompile |
| [[graph_values_metadata_and_signatures_analysis]] | deep dive(C03) | Node 引用、`meta["val"]`、pytree 结构化输出与三类 graph signature |
| [[graph_effects_alias_mutation_and_order_analysis]] | deep dive(C05) | 数据边之外的 alias、mutation、functionalization 与 effect 顺序 |
| [[structured_outputs_higher_order_and_nested_graphs_analysis]] | deep dive(C06) | 多输出、HOP(`cond`/`while_loop`/`map`)与 nested GraphModule 如何扩展普通 DAG |
| [[graph_capture_frontends_and_tracing_analysis]] | deep dive(C07) | `symbolic_trace`、`make_fx`、Dynamo、`export` 四种捕获前端为何产生不同图 |
| [[graph_normalization_decomposition_and_functionalization_analysis]] | deep dive(C08) | schema normalization、decomposition 与 functionalization 为何必须分层 |
| [[graph_stage_boundaries_identity_and_provenance_analysis]] | deep dive(C11) | Node identity 跨阶段(capture→AOT→Inductor)断开后如何维持 provenance |
| [[fx_graph_editing_primitives_and_invariants_analysis]] | deep dive(C12) | replace/erase/copy/lint/recompile 组成的安全改图事务、10 项检查清单 |
| [[pattern_expression_and_matcher_engine_analysis]] | deep dive(C13) | PatternExpr AST、候选桶索引、逆序 matcher、三类 Entry 与序列化 pattern |
| [[dead_code_topology_and_effect_order_analysis]] | deep dive(C14) | FX DCE/Scheduler DCE、stable topological sort、拓扑正确 ≠ effect 正确 |
| [[graph_pass_pipeline_ordering_and_fixpoint_analysis]] | deep dive(C15) | pass stage、注册顺序、迭代与 fixpoint 如何决定改写结果 |
| [[graph_rewrite_legality_validation_and_complexity_analysis]] | deep dive(C16) | 结构命中后 shape/dtype/alias/autograd 的合法性验证、全链路复杂度 |

> 12 篇按 [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]] Part I–III 的固定顺序组织
> (Part IV 的 C17–C21 属 Inductor IR/Scheduler/Codegen,随 Task 8 迁入 `04_inductor/`)。课程化
> 阅读顺序、前置依赖表与配套 Demo 见该系列索引 §"四部分知识地图"与 §"每篇前置依赖与学习成果"。

## 与旧页的关系(P4 Task 7 四组判重归一,逐组 commit 台账见各自 commit message)

- **组 1**:`02_aot_autograd/fx_graph_construction_and_transformation_analysis.md`
  (2026-07-23 综合报告快照,601 行)的 FX 数据模型、PatternExpr/PatternMatcherPass、DCE、
  稳定拓扑排序、合法性/复杂度部分已核实并入上表对应页;该页瘦身为 269 行,现仅存
  AOTAutograd joint→fw/bw 构图与 saved-tensor/recompute 专题,留给 Task 8 与 C09/C10 归一。
- **组 2**:`04_inductor/fx_pass_optimization_methodology.md`(349 行)+
  `04_inductor/torch_upstream_pass_deepdive.md`(232 行)判重后**均删除**,独有事实(八阶段
  placement 细节缺口、`fwd_only`/`joint_fwd_bwd`、`GraphTransformObserver`、
  `GroupBatchFusionBase`、四则 pass 注册取证式脚注、跨框架方法论对照)分别并入 C13/C15/C16。
- **组 3**:`04_inductor/decomposition_passes_guide.md`(163 行)vs C08 判重后**保留双页**——
  guide 是开发者 API/checklist 视角,与 C08 的捕获期机制解释体裁不同,独有内容 >50%。
- **组 4**:`02_compile_stack/01_dynamo/control_flow_capture_analysis.md`(204 行)vs C06
  判重后**基本保留**,只在 `torch.cond` 的 `FakeTensorMode`/`ProxyTorchDispatchMode` 重叠段
  收缩为互指(该页讲 Dynamo 捕获前端,C06 讲 IR 层结构,体裁不同、几乎零重叠)。

---

## 关联域

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]] — 编号化课程主线,C01(动机)与 C17–C21(Inductor IR)不在本目录
- [[19_torch_compile_end_to_end/00_torch_compile_end_to_end_index]] — 六卷端到端总索引,本目录对应卷 C 的 §5 部分行
- [[02_compile_stack/01_dynamo/index]] — 上游:Dynamo 捕获出 guarded FX graph 交给本目录的捕获/规范化机制
- [[02_compile_stack/02_aot_autograd/index]] — 下游承接:joint graph、正反向分图与 recompute
- [[02_compile_stack/04_inductor/index]] — 下游承接:FX lowering 为 Inductor IR、pass 阶段 guide 与 scheduler
- [[04_export_and_distributed/01_fx_export_extensibility/index]] — 另一条使用场景:`export`/自定义 op 对 FX IR 的复用
- [[01_ai_frameworks/index]] — 本域总索引
