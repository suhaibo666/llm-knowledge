# PyTorch 图编译器学习系列重构设计

**日期**：2026-07-23
**状态**：设计已由用户确认，等待规格复审后进入实施计划
**目标仓库**：`llm-knowledge`
**源码主基线**：本地 `origin/main` detached checkout `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`，2026-07-23
**新系列目录**：`wiki/02_engineering/01_ai_frameworks/16_graph_compiler_foundations/`

## 1. 背景与问题

现有知识库已经包含 FX、export、AOTAutograd、Inductor pass、dynamic shape、lowering、scheduler 和 codegen 等大量材料，但它们按模块或单次问题逐步积累，存在四类结构性问题：

1. “图”在不同页面中可能指 eager autograd tape、FX program graph、AOT backward graph、Inductor IR 或 Scheduler dependency graph，缺少统一分类和边界。
2. 数据结构、图语义、构图阶段、pass 机制与后端执行被混在同一长文中，读者很难建立前置依赖。
3. effect、alias、mutation、symbolic shape、graph signature、nested graph、liveness 等横切机制散落在多个模块页面中。
4. 部分历史页面基于旧 PyTorch 版本或未声明的源码基线，行号、路径和结论可能已经漂移；现有页面不能直接作为新系列的事实来源。

本次重构不以缩短页面为目标，也不以固定篇数为目标。首要目标是完整、准确地建立从“为什么需要图”到“FX 如何成为 kernel”的学习体系。

## 2. 用户目标与学习终点

系列采用“概念与语义轴 × 编译生命周期轴”的双轴设计。

读者完成系列后应具备以下能力：

1. 区分 PyTorch 编译栈中不同种类的图、节点、边、值和顺序语义。
2. 阅读 FX/AOT/Inductor dump，并判断当前图所属阶段。
3. 从 Python 源码追踪一个计算到 Dynamo FX、AOT joint、fw/bw、Inductor IR、Scheduler group 和生成 kernel。
4. 理解 alias、mutation、effect、dynamic shape 和 graph signature 对改图合法性的约束。
5. 使用 FX 编辑 API 和 PatternMatcher 实现 pass。
6. 判断 pass 应放在哪个阶段，验证数值、梯度、alias、mutation 和 shape 语义。
7. 分析构图、匹配、DCE、排序、partition、scheduler 和 codegen 的主要复杂度。
8. 独立定位图捕获、分图、改写、内存和代码生成问题。

## 3. 设计原则

### 3.1 准确性第一

现有 wiki 只能作为审计线索，不能作为新文档的最终事实来源。所有非平凡实现断言必须回到固定源码基线、测试或官方一手资料核验。

事实来源优先级：

1. 固定 commit 的实现源码和测试；
2. 同一基线中的源码注释、类型和不变量检查；
3. PyTorch 官方文档、开发者文档和官方论文；
4. 现有 wiki 页面，仅用于发现主题和历史解释；
5. 推断必须显式标记，不得伪装成源码事实。

### 3.2 完整闭环

一篇文档的边界由独立学习问题决定，不由行数决定。一个主题必须完整回答：

```text
为什么需要
→ 概念模型
→ 数据结构
→ 不变量
→ 源码执行路径
→ 贯穿示例
→ 可运行实验
→ 失败模式
→ 复杂度
→ 与上下游的关系
```

如果完整解释需要长文，则保留长文。只有出现独立学习目标、不同前置依赖或可独立验证的子系统时才拆篇。

### 3.3 无损迁移

不删除现有正文来实现“去重”。先建立新系列，再用覆盖矩阵证明旧内容已被吸收。旧页面保持全文，并根据角色增加：

- 新系列入口；
- 逐节迁移去向；
- 历史源码基线和验证状态；
- 仍然独有的专题范围。

### 3.4 概念先于源码

每个机制先解释它解决的问题、若不存在会造成什么后果、为什么采用当前设计，再进入对象关系和源码路径。避免把 API 顺序当作原理。

### 3.5 每篇包含可验证 Lab

Lab 必须包含：

- 最小可运行输入；
- 明确的执行命令；
- 预期图结构或输出模式；
- 至少一个正确案例和一个错误/边界案例；
- dump、日志或生成 artifact 的定位方法；
- 源码基线和运行环境；
- 验证结果，不得把未运行示例写成已验证。

## 4. 范围

### 4.1 主线范围

```text
Python / nn.Module
→ FX 与 Dynamo 捕获
→ functional ATen 与规范化
→ AOTAutograd joint graph
→ forward / backward FX graphs
→ Inductor post-grad FX
→ GraphLowering
→ Inductor IR
→ Scheduler dependency graph
→ backend codegen
→ generated kernel 与 wrapper
```

### 4.2 延伸但不进入主线

- eager autograd engine；
- CUDA Graph；
- 分布式通信图和执行图；
- compile cache；
- activation checkpoint 用户 API；
- NPU/CUDA 后端专属 kernel 优化；
- custom operator 注册；
- 具体模型的领域级融合。

这些主题通过 Related Pages 接入，不与主线中的 program graph、IR 或 scheduler graph 混为一谈。

## 5. 系列信息架构

新建 1 个系列索引和 21 篇核心文档。21 是当前独立问题边界的结果，不是硬上限。

### 5.1 系列索引

`index.md`

包含：

- 图类型和完整编译生命周期总图；
- 四部分知识地图；
- 每篇前置依赖与学习成果；
- 基础、pass 开发、后端三条阅读路径；
- 贯穿模型和 Lab 环境；
- 源码基线、验证等级和术语入口；
- 与 FX/export、AOTAutograd、Inductor、runtime 和 cache 领域的导航。

### 5.2 Part I：图的概念与语义基础

#### 01 `01_graph_ir_motivation_and_taxonomy.md`

核心问题：为什么编译器需要图，各类“图”是否相同？

包含：

- Python/eager 对全局分析和变换的限制；
- 图作为 IR 的价值；
- program graph、dataflow graph、eager autograd tape、AOT backward graph、Scheduler dependency graph、runtime CUDA Graph；
- Node、Edge、Value、Order；
- DAG、循环和控制流；
- FX 图与 autograd graph 的边界。

Lab：同一函数对比 eager `grad_fn`、FX graph 和编译日志。

#### 02 `02_fx_graph_core_data_model.md`

核心问题：FX 如何同时保存程序顺序和数据依赖？

包含：

- `Graph`、`Node`、`GraphModule`、`Proxy`、`Tracer`；
- 侵入式双向链表；
- `args/kwargs`、`_input_nodes/users`；
- 没有独立 Edge 对象的原因；
- 重复 use 与 user 集合；
- root sentinel、插入点、find-nodes lookup table、ownership；
- create/replace/erase、lint、recompile 和生成代码。

Lab：手工创建、修改和破坏一张 Graph，观察 users、lint 和 recompile。

#### 03 `03_graph_values_metadata_and_signatures.md`

核心问题：图边上传递的值在编译期是什么，图签名如何对应用户程序？

包含：

- 运行时 Tensor、Proxy、FakeTensor 和 Node 引用；
- 常量、placeholder、get_attr、参数、Buffer 和 lifted state；
- 结构化 args/kwargs、pytree 和 `getitem`；
- `node.meta` 中的 val、tensor_meta、stack trace 和 provenance；
- 基础 Graph 输入输出；
- Export/AOT Graph Signature 的分层关系。

Lab：对比 Python 签名、FX placeholders、Node meta 和 ExportedProgram graph signature。

#### 04 `04_symbolic_shapes_guards_and_graph_reuse.md`

核心问题：一张图为什么只对某些形状成立，又如何安全复用？

包含：

- 静态特化；
- SymInt/SymBool 和 ShapeEnv；
- 符号表达式、shape op 和 FakeTensor propagation；
- Dynamo guards 与 export range constraints；
- guard 失败和重新编译；
- 符号恒等、可证明条件；
- dynamic shape 对 pattern、DCE、fusion 和 codegen 的影响。

Lab：多组 shape 下观察 guards、重编译和符号节点。

#### 05 `05_graph_effects_alias_mutation_and_order.md`

核心问题：为什么没有数据边的两个节点也不一定能交换或删除？

包含：

- 数据依赖与 effect/control dependency；
- FX 缺少通用显式控制边的含义；
- purity、RNG、collective 和 stream；
- storage、alias、view、in-place mutation；
- mutation region 和 effect token；
- 输入/参数 mutation；
- functionalization 的语义；
- post-grad reinplace 的合法边界。

Lab：构造错误 DCE、错误重排和 functionalization 前后对照。

#### 06 `06_structured_outputs_higher_order_and_nested_graphs.md`

核心问题：多值返回和图中子图如何扩展普通 DAG 模型？

包含：

- tuple/list/dict 和 graph output pytree；
- tuple Node、`getitem`；
- 单节点多值与 MultiOutputPattern；
- HigherOrderOperator；
- cond、while_loop、map 和 checkpoint 类结构；
- Node 参数中的子 GraphModule；
- 分支签名、自由变量；
- 递归 pass、lint、DCE 和 ownership 边界。

Lab：捕获 tuple 返回和 `torch.cond`，进入 branch GraphModule 检查边界。

### 5.3 Part II：图的捕获、规范化与正反向构造

#### 07 `07_graph_capture_frontends_and_tracing.md`

核心问题：symbolic_trace、make_fx、Dynamo 和 export 为什么产生不同图？

包含：

- Proxy 拦截、ProxyTensor/torch_dispatch、Dynamo 字节码执行、export AOT 捕获；
- leaf function/module；
- real/Fake/symbolic value；
- Python 控制流、数据分支、graph break；
- guards、strictness；
- 四种路径的输入、算子层级、能力和产物。

Lab：用同一模型执行四种捕获并比较节点、guards、break 和 meta。

#### 08 `08_graph_normalization_decomposition_and_functionalization.md`

核心问题：为什么捕获成功后仍需规范化？

包含：

- 高层图和 functional ATen 图；
- schema normalization；
- decomposition；
- view/mutation functionalization；
- synthetic base；
- mutation 的输入输出编码；
- 算子集收敛；
- canonicalization、constant folding 和 CSE；
- 规范化顺序对 pattern 和 partition 的影响。

Lab：逐步比较 capture、decomposition 和 functionalization 图。

#### 09 `09_aotautograd_joint_forward_backward_graphs.md`

核心问题：AOTAutograd 如何得到 backward 并生成两张独立图？

包含：

- primals、tangents 和 create_joint；
- forward 与 autograd.grad；
- forward/backward 节点标记；
- joint outputs；
- required-node closure 和 classify；
- partition_fn；
- old-to-new env、fresh placeholder；
- fw 用户输出、saved outputs、bw inputs 和 gradients；
- fw/bw 无跨图 Node 边。

Lab：用 aot 日志逐节点映射 joint、fw 和 bw。

#### 10 `10_saved_tensors_recompute_and_runtime_abi.md`

核心问题：正反向之间保存、重算和传递什么？

包含：

- saved-value boundary；
- default partition 与 min-cut rematerialization；
- flow network 和割边代价；
- activation memory budget；
- forward Node 复制到 bw；
- recompute metadata 和 backward reorder；
- saved tensor/SymInt、opaque object、tangent；
- runtime wrapper、autograd context 和版本检查。

Lab：切换 partition/memory budget，比较签名、recompute 和峰值激活。

#### 11 `11_graph_stage_boundaries_identity_and_provenance.md`

核心问题：各编译阶段看到的是不是同一张图，如何追踪节点演化？

包含：

- Dynamo FX 到 Scheduler graph 的阶段地图；
- Node identity 不连续；
- decomposition 一对多、fusion 多对一、partition 复制、lowering 重建 IR；
- node.meta、stack trace、source_fn_stack、debug handle；
- generated-code mapping；
- 每个 pass 的真实阶段和 dump 名称。

Lab：追踪一个 matmul+activation 穿过所有阶段。

### 5.4 Part III：安全改图、匹配、清理与验证

#### 12 `12_fx_graph_editing_primitives_and_invariants.md`

核心问题：最小改图需更新哪些结构？

包含：

- mutation-safe traversal；
- insertion points；
- create/call/node_copy/graph_copy；
- args/users 同步；
- replace 和 erase；
- output/placeholder signature；
- ownership、name、meta；
- recompile；
- 局部合法与阶段规范化。

Lab：实现插入、替换、复制三个 pass，并故意破坏不变量。

#### 13 `13_pattern_expression_and_matcher_engine.md`

核心问题：如何声明可表达 DAG 共享和稳定 handler ABI 的子图 pattern？

包含：

- PatternExpr 和各 TargetExpr；
- Arg、KeywordArg、Ignored；
- constants、users、多用户约束；
- Match/MatchContext 和失败回滚；
- MultiOutputPattern anchors；
- root registry、candidate index、reverse traversal；
- Lowering/Graph/Replacement entries；
- traced replacement 和 cache；
- mutation/stream/mempool boundaries。

Lab：逐步实现 unary、shared-input、kwargs、multi-output、replacement 和 lowering patterns。

#### 14 `14_dead_code_topology_and_effect_order.md`

核心问题：节点何时真正 dead，拓扑正确为何不等于语义正确？

包含：

- dead、unclaimed、unsaved 的区别；
- pure + no users；
- impurity；
- reverse cascading DCE；
- nested graph cleanup；
- list order 与 dependency order；
- stable topological sort；
- effect/mutation/RNG/collective/stream order；
- lint、DCE、sort、recompile 顺序。

Lab：dead chain、副作用节点、拓扑错误和 effect 重排反例。

#### 15 `15_graph_pass_pipeline_ordering_and_fixpoint.md`

核心问题：pass 为什么必须位于正确阶段并按正确顺序执行？

包含：

- Dynamo、pre-grad、decomposition、joint、post-grad、lowering 前置不变量；
- canonicalization 和 mutation-tail ordering；
- pass conflicts 和注册顺序；
- single round、repeat-until-stable、fixed point；
- idempotence；
- replacement candidate lifecycle；
- observer、counters、hooks 和 debug。

Lab：同一融合规则放入不同阶段，并构造非收敛 pass 对。

#### 16 `16_graph_rewrite_legality_validation_and_complexity.md`

核心问题：结构命中后如何证明改写合法、正确且值得做？

包含：

- shape、dtype、device、layout、stride、broadcast；
- alias、mutation、effect、autograd、dynamic shape；
- 数值、梯度、alias 和 mutation 等价；
- FakeTensor/meta checks 和 runtime guards；
- differential testing、随机 shape、边界值、gradcheck 和 determinism；
- 失败原子性；
- 构图、候选、pattern、replacement、DCE、sort、lint、recompile、partition 和 pipeline 复杂度。

Lab：修复一个结构命中但 broadcast/alias 不合法的融合。

### 5.5 Part IV：从 FX 到 Inductor IR、Scheduler 和 Kernel

#### 17 `17_fx_lowering_to_inductor_ir.md`

核心问题：为什么 FX 与代码生成之间需要 Inductor IR？

包含：

- FX operator 与 loop/memory implementation 的鸿沟；
- GraphLowering Interpreter；
- FX Node 到 TensorBox/StorageBox/IR value；
- register_lowering、fallback、ExternKernel、lowering-pattern；
- shape/stride/layout；
- output、mutation、device；
- decomposition、post-grad fusion 和 lowering 的职责。

Lab：追踪 pointwise、reduction、matmul、unsupported op，并增加 custom lowering。

#### 18 `18_inductor_ir_values_loops_layouts_and_buffers.md`

核心问题：Inductor IR 如何表达循环、寻址和存储？

包含：

- TensorBox、StorageBox、Buffer、Operation；
- Pointwise、Reduction、Scan、ExternKernel、TemplateBuffer；
- LoopBody、iteration domain 和 index expression；
- Fixed/Flexible/NonOwning layout；
- stride、offset、view/reinterpret；
- device、dtype、mutation、alias；
- IR dependency 入口；
- FX value 与 IR buffer 的非一对一关系。

Lab：对 elementwise、broadcast、transpose、reduction 和 matmul 核对 IR 与索引。

#### 19 `19_buffer_liveness_memory_planning_and_reuse.md`

核心问题：哪些中间值需要真实 Buffer，活到何时，如何复用？

包含：

- FX users、IR reads/writes、Scheduler dependencies；
- logical value 与 physical buffer；
- last use、liveness、free/reuse；
- alias/view/mutation/output/external limits；
- in-place reuse；
- memory planning 和 wrapper allocation；
- saved/recompute 对后端峰值的影响；
- fusion 和 materialization；
- peak-memory trade-off。

Lab：比较 fusion、save/recompute、view/copy 的 buffer 生命周期和峰值。

#### 20 `20_scheduler_dependency_graph_fusion_and_ordering.md`

核心问题：Scheduler graph 的边为何不同于 FX users？

包含：

- SchedulerNode 和 backend scheduling；
- reads/writes dependencies；
- MemoryDep、WeakDep、StarDep 和 mutation/collective dependencies；
- topological schedule；
- vertical/horizontal/foreach fusion；
- legality 和 profitability；
- reduction、template、extern、multi-output 和 device boundaries；
- locality reorder；
- scheduler graph 与 operator dataflow 的区别。

Lab：打印 dependency/fusion groups，关闭 fusion/reorder 比较 kernel 和性能。

#### 21 `21_codegen_kernel_mapping_autotuning_and_provenance.md`

核心问题：Scheduler group 如何成为 kernel，并如何映射回原始源码？

包含：

- backend codegen dispatch；
- Scheduler group 到 kernel；
- loop ordering、tiling、indexing、mask、reduction；
- Triton、C++、template/extern branches；
- algorithm choices、template selection 和 autotuning；
- wrapper、allocation、launch 和 stream；
- compile cache 和 artifacts；
- kernel → scheduler → IR → FX → Python provenance。

Lab：追踪一个 fusion group 的 kernel、wrapper、autotune choices 和源码映射。

## 6. 贯穿模型与实验结构

全系列使用一个逐步扩展的模型，包含：

- parameter 和 buffer；
- view 与可选 mutation；
- dynamic shape；
- 结构化输出；
- 可微 matmul、pointwise 和 reduction；
- 一个可由 higher-order operator 表达的分支。

Part I 用它建立语义；Part II 捕获和分图；Part III 实现 `add + matmul` 改写；Part IV 追踪到 Buffer、Scheduler 和 kernel。

最终实验生成一份端到端 artifact bundle：

- Python source；
- Dynamo/FX dump；
- functional ATen；
- AOT joint、fw、bw；
- post-grad FX；
- Inductor IR；
- Scheduler dependencies/fusion groups；
- generated kernel 和 wrapper；
- 节点拆分、复制、融合、liveness 和 provenance 表。

## 7. 历史资料准确性审计

### 7.1 审计对象

至少包括：

- `02_compile_stack/02_aot_autograd/aotautograd_analysis.md`
- `02_compile_stack/02_aot_autograd/aot_autograd_quickstart.md`
- `02_compile_stack/02_aot_autograd/fx_graph_construction_and_transformation_analysis.md`
- `04_export_and_distributed/01_fx_export_extensibility/fx_graph_export_and_custom_ops_analysis.md`
- `04_export_and_distributed/01_fx_export_extensibility/fx_export_custom_op_quickstart.md`
- `02_compile_stack/04_inductor/dynamic_shapes_full_analysis.md`
- `02_compile_stack/04_inductor/decomposition_passes_guide.md`
- `02_compile_stack/04_inductor/pre_grad_passes_guide.md`
- `02_compile_stack/04_inductor/joint_graph_passes_guide.md`
- `02_compile_stack/04_inductor/post_grad_passes_guide.md`
- `02_compile_stack/04_inductor/torch_upstream_pass_deepdive.md`
- `02_compile_stack/04_inductor/fx_pass_optimization_methodology.md`
- `02_compile_stack/04_inductor/lowering_analysis.md`
- `02_compile_stack/04_inductor/scheduler_analysis.md`
- `02_compile_stack/04_inductor/inductor_codegen_analysis.md`
- `02_compile_stack/04_inductor/inductor_autotuning_analysis.md`
- `02_compile_stack/04_inductor/PyTorch_Inductor_Technical_Analysis.md`
- `02_compile_stack/04_inductor/torch_compile_architecture.md`
- runtime memory、activation checkpointing 和 compile-cache 相关页面。

### 7.2 审计单位

为每个旧页面建立 coverage ledger，至少记录：

| 字段 | 含义 |
|---|---|
| legacy page | 旧页面 |
| section | H2/H3 或代码块 |
| claim | 非平凡断言 |
| locator | 旧定位符或来源 |
| claimed baseline | 页面声称或推定的版本 |
| current result | 当前源码核验结果 |
| status | verified-current / verified-historical / corrected / unresolved |
| destination | 新文档与章节 |
| action | 保留、重写、标版本、纠错或阻塞 |

旧页面中的每个 H2/H3、代码块、图表和实验都必须进入 ledger。只统计标题数量不足以证明无损。

### 7.3 验证状态

- **verified-current**：在本次固定 commit 上直接成立。
- **verified-historical**：在原页面基线上成立，但当前实现已变化；必须明确版本。
- **corrected**：原断言在其声称基线上也不成立，或与源码相冲突；新文档纠正，旧页增加 correction callout 和 changelog。
- **unresolved**：证据不足或环境无法验证；不得作为权威结论进入新系列。

### 7.4 审计方法

1. 固定当前 PyTorch commit、branch、dirty state 和日期。
2. 抽取旧页面标题、源码定位符、代码块、Mermaid 和外部引用。
3. 对每个 `file:line`：
   - 验证文件存在；
   - 验证符号仍存在；
   - 阅读完整函数/类上下文，而非只检查文本命中；
   - 核对调用者和被调用者；
   - 对关键路径核对测试。
4. 对代码示例：
   - 语法检查；
   - 在声明环境运行；
   - 核对输出结构和关键字段；
   - 动态 shape、mutation 和 backward 示例覆盖边界输入。
5. 对复杂度：
   - 从实际循环、数据结构和调用次数推导；
   - 区分常见条件、期望复杂度和严格上界；
   - 不用未经证实的算法实现假设填补 max-flow/autotune 等成本。
6. 对当前实现与历史页面冲突：
   - 当前行为以固定源码和测试为准；
   - 历史结论若当时正确，则标为 version-specific，不称为错误；
   - 历史结论若本身错误，保留纠错记录。

### 7.5 准确性门槛

新文档不得发布为 authoritative，除非满足：

- 100% 非平凡实现断言有固定基线定位符或明确推断标记；
- 100% 旧页面 H2/H3、代码块、图表和实验已映射；
- 所有新增 Lab 已运行，或明确标记为未验证且不进入正式正文；
- 关键图路径至少有一个真实 artifact 证明；
- 0 个悬空 wikilink；
- Mermaid 结构检查通过；
- 代码围栏平衡；
- 页面包含 Related Pages；
- 索引、反向链接和 changelog 同步；
- `git diff --check` 通过。

## 8. 迁移策略

### 8.1 拆分迁移但保留全文

`fx_graph_construction_and_transformation_analysis.md`

- 迁入 02、09、10、13、14、16；
- 原文保留；
- 增加“系统系列去向”表；
- 角色变为本次问答形成的综合报告快照。

`aotautograd_analysis.md`

- 通用内容迁入 05、08、09、10、11；
- 原文保留为 AOTAutograd 模块全量参考和 edge-case 集合；
- 增加基线、审计状态和课程导航。

### 8.2 保留领域专题

- FX/export 页面保留 export、custom-op、functorch；
- dynamic-shape 页面保留高级 shape 专题；
- decomposition/pre/joint/post 页面保留逐 pass 目录和注册 API；
- pass methodology 保留工程方法论；
- lowering/scheduler/codegen/autotuning 页面保留各子系统完整源码参考；
- runtime memory、compile cache、activation checkpointing 保持独立。

新系列吸收主线解释，但不搬空这些专题页面。

### 8.3 索引与回链

修改：

- `01_ai_frameworks/index.md`
- `02_compile_stack/02_aot_autograd/index.md`
- `02_compile_stack/04_inductor/index.md`
- `04_export_and_distributed/01_fx_export_extensibility/index.md`
- `01_eager_runtime/07_memory_amp_profiler/index.md`
- `02_compile_stack/06_compile_cache/index.md`
- `wiki/changelog.md`

相关旧页面增加反向链接和角色说明。

## 9. 实施批次

### Batch 0：基线与历史审计

- 固定源码基线；
- 建立 coverage ledger；
- 核验旧 locators、claims、code 和 labs；
- 输出 correction list；
- 未解决问题进入阻塞清单。

### Batch 1：Part I

- 创建 index 初版和 01–06；
- 运行 Part I labs；
- 固定术语、图类型和贯穿模型；
- 更新 FX/export、dynamic-shape 和 eager-autograd 链接。

### Batch 2：Part II

- 创建 07–11；
- 验证 capture、functionalization、joint、fw/bw 和 runtime artifacts；
- 更新 AOTAutograd 页面角色与映射。

### Batch 3：Part III

- 创建 12–16；
- 实现贯穿 pass；
- 验证 pattern、DCE、sort、stage placement、legality 和 complexity；
- 更新 pass 目录与方法论链接。

### Batch 4：Part IV

- 创建 17–21；
- 生成 IR、scheduler、kernel artifacts；
- 完成端到端 provenance；
- 更新 Inductor、runtime 和 cache 链接；
- 完成全系列 index 和 changelog。

每个 Batch 单独完成覆盖审计和验证，不等待全部文档写完后统一返工。

## 10. 文件改动范围

### 新建

- `wiki/02_engineering/01_ai_frameworks/16_graph_compiler_foundations/00_pytorch_graph_series_index.md`
- 21 篇核心文档；
- 实施阶段的 coverage ledger 和 correction report；
- 必要的最小 Lab 文件或 artifact，位置在实施计划中确定。

### 修改

- 6 个索引页面（框架总索引、AOTAutograd、Inductor、FX/export、runtime memory、compile cache）；
- changelog；
- 本规格第 7.1 节所列旧页面的导航、审计状态、correction callout 和 Related Pages。

### 不做

- 不删除现有正文；
- 不把未核验旧内容直接复制为新结论；
- 不因长度压缩细节；
- 不把 CUDA Graph 或分布式图并入主线；
- 不在没有 artifact 的情况下声称 Lab 已验证；
- 不一次性重写 21 篇后再做校验。

## 11. 风险与控制

| 风险 | 控制 |
|---|---|
| 当前 PyTorch 源码继续变化 | 每批固定 commit；页面页头记录基线；更新时重新审计 |
| 旧文档无明确基线 | 标为 baseline-unknown；只能作为线索，不能直接迁入 |
| 新旧页面重复 | 新页承担课程主线，旧页承担模块/阶段参考；页头明确角色 |
| 21 篇范围过大 | 分批交付，每批独立可读；不牺牲完整性 |
| Lab 环境不可运行 | 明确阻塞，不伪造输出；先完成源码事实审计 |
| line locator 漂移 | locator 与 commit 绑定；同时记录符号名和调用路径 |
| 推断被误认为事实 | 推断显式标注，并说明证据链 |
| 复杂度过度简化 | 参数化表达，区分期望、常见和严格上界 |

## 12. 验收标准

系列完成时必须满足：

1. 新索引能够把读者从概念引导至 codegen。
2. 21 篇各自形成完整学习闭环，没有依赖未定义术语。
3. 每个旧页面的 H2/H3、代码块、图表和实验都有 coverage ledger 去向。
4. 所有历史错漏都有 correction report 和 changelog 记录。
5. 所有权威实现断言绑定固定源码基线。
6. 所有正式 Lab 已真实运行并记录预期输出。
7. 贯穿模型可生成完整阶段 artifact bundle。
8. Part III 的贯穿 pass 通过数值、梯度、shape、alias/mutation 和重复运行验证。
9. Part IV 能从 Scheduler/kernel 反向定位至 FX 和用户源码。
10. 所有链接、Mermaid、代码围栏、Related Pages、索引、回链和 diff 检查通过。

## 13. 设计确认记录

用户已确认：

- 学习终点为从基础概念到能够读源码、写 pass 和判断合法性；
- 主线延伸至 Inductor IR、Scheduler 和 codegen；
- 对现有页面执行拆分、合并和迁移，而不是另建重复体系；
- 每篇包含概念背景、源码与可运行实验；
- 不以篇幅限制内容；
- 完整性与准确性优先；
- 旧资料必须重新核验，历史内容不能默认正确。
