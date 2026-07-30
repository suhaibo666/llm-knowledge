# 01 · TorchDynamo 图捕获 — 目录索引

> torch.compile 前端:Python 帧评估钩子、字节码符号执行、Guard 生成与守卫失败重编译。
> 知识分层:overview(本索引)→ quick start → deep dive(约定见 [[01_ai_frameworks/index]])。
> 最后更新: 2026-07-30(P4 Task 5:卷 B 十篇迁入并去前缀重命名,取代原 `PyTorch_Dynamo_Technical_Analysis` 单篇 deep dive)

---

## 页面列表(按层次)

> **段位与阅读顺序**(kb-reorg P4 Task 9.5,2026-07-30):段 0(01-09)入门;段 1(10-19)核心机制主线——按执行流水线 api→eval frame→translator→variable tracker→output graph→guards→graph break→dynamic shapes→backend contract 排列;段 2(20-29)深潜/专题——符号形状概念权威页、控制流捕获、后端选项面,均依赖段 1 对应机制页;段 3(30-39)方法论。

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[01_dynamo_quickstart]] | **quick start**(段 0) | 看捕获结果 `torch._dynamo.explain`、graph break 定位(`TORCH_LOGS=graph_breaks`)、`fullgraph=True`、guards/recompiles、`disable`/`allow_in_graph`/`reset` 逃生阀 |
| [[10_torch_compile_api_and_first_call_lifecycle_analysis]] | deep dive(B01,段 1) | wrapper 创建与第一次真正编译如何区分、生命周期状态机 |
| [[11_eval_frame_callback_and_code_cache_analysis]] | deep dive(B03,段 1) | eval-frame callback 三态协议、code cache 挂在 code object 上的原因 |
| [[12_instruction_translator_and_bytecode_state_machine_analysis]] | deep dive(B04,段 1) | 字节码符号执行状态机、`run → step → opcode handler`、图断点重启 |
| [[13_variable_tracker_source_and_python_object_model_analysis]] | deep dive(B05,段 1) | VariableTracker/VariableBuilder、Source 取值路径语法树 |
| [[14_output_graph_side_effects_and_graph_emission_analysis]] | deep dive(B06,段 1) | OutputGraph 所有权边界、SideEffects、FX graph 如何交给 backend |
| [[15_guards_cache_lookup_and_recompilation_analysis]] | deep dive(B07,段 1) | guard 树、cache lookup 精确顺序、重编译判定与两层上限 |
| [[16_graph_break_resume_functions_and_partial_graphs_analysis]] | deep dive(B08,段 1) | graph break 触发场景、resume function、partial graph 真实执行形态 |
| [[17_dynamic_shapes_generalization_and_fallback_analysis]] | deep dive(B09,段 1) | 动态形状泛化状态机、backed/unbacked symbol、泛化失败原因(Dynamo 侧行为面;符号系统本身见下方 20 号页) |
| [[18_backend_contract_and_custom_backend_analysis]] | deep dive(B10,段 1) | backend 最小契约、custom backend 与 AOTAutograd 的组合边界 |
| [[20_symbolic_shapes_guards_and_graph_reuse_analysis]] | deep dive(C04,段 2) | 符号形状/Guard/图复用**概念权威页**:ShapeEnv、SymNode、`DimDynamic` 分配策略、guard 生成、backed/unbacked 判定、matmul 端到端案例(2026-07-30 起并入原 `dynamic_shapes_full_analysis` 独有段) |
| [[21_control_flow_capture_analysis]] | deep dive(专题,段 2) | 控制流两路径:显式 HOP(`cond`/`while_loop`/`map`/`scan`)投机子图入图 vs 原生 `if`/`for`/`while` 字节码特化/展开/切图 |
| [[22_backend_modes_options_stances_and_fullgraph_analysis]] | deep dive(B02,段 2) | backend/mode/options/stance/fullgraph 五个控制面怎样互不覆盖 |
| [[30_dynamo_pass_methodology]] | **development guide**(段 3) | Dynamo backend 是什么/为什么、整图改写边界、`register_backend`/callable 注册示例、何时后移到 Inductor 各阶段 |

> 端到端流水线(Dynamo→AOTAutograd→Inductor)见 [[torch_compile_architecture]]。B01-B10 的课程化阅读顺序与配套 Demo 见 [[19_torch_compile_end_to_end/00_torch_compile_end_to_end_index]] §4。

---

## 关联域

- [[19_torch_compile_end_to_end/00_torch_compile_end_to_end_index]] — 编号化端到端课程：卷 B 系统展开 Dynamo，并连接卷 A、C–F
- [[02_compile_stack/02_aot_autograd/index]] — 下一阶段:前/反向分解
- [[02_compile_stack/04_inductor/index]] — 编译后端
- [[01_ai_frameworks/index]] — 本域总索引
