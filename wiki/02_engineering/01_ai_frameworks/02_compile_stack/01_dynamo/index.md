# 01 · TorchDynamo 图捕获 — 目录索引

> torch.compile 前端:Python 帧评估钩子、字节码符号执行、Guard 生成与守卫失败重编译。
> 知识分层:overview(本索引)→ quick start → deep dive(约定见 [[01_ai_frameworks/index]])。
> 最后更新: 2026-07-30(P4 Task 5:卷 B 十篇迁入并去前缀重命名,取代原 `PyTorch_Dynamo_Technical_Analysis` 单篇 deep dive)

---

## 页面列表(按层次)

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[dynamo_quickstart]] | **quick start** | 看捕获结果 `torch._dynamo.explain`、graph break 定位(`TORCH_LOGS=graph_breaks`)、`fullgraph=True`、guards/recompiles、`disable`/`allow_in_graph`/`reset` 逃生阀 |
| [[torch_compile_api_and_first_call_lifecycle_analysis]] | deep dive(B01) | wrapper 创建与第一次真正编译如何区分、生命周期状态机 |
| [[backend_modes_options_stances_and_fullgraph_analysis]] | deep dive(B02) | backend/mode/options/stance/fullgraph 五个控制面怎样互不覆盖 |
| [[eval_frame_callback_and_code_cache_analysis]] | deep dive(B03) | eval-frame callback 三态协议、code cache 挂在 code object 上的原因 |
| [[instruction_translator_and_bytecode_state_machine_analysis]] | deep dive(B04) | 字节码符号执行状态机、`run → step → opcode handler`、图断点重启 |
| [[variable_tracker_source_and_python_object_model_analysis]] | deep dive(B05) | VariableTracker/VariableBuilder、Source 取值路径语法树 |
| [[output_graph_side_effects_and_graph_emission_analysis]] | deep dive(B06) | OutputGraph 所有权边界、SideEffects、FX graph 如何交给 backend |
| [[guards_cache_lookup_and_recompilation_analysis]] | deep dive(B07) | guard 树、cache lookup 精确顺序、重编译判定与两层上限 |
| [[graph_break_resume_functions_and_partial_graphs_analysis]] | deep dive(B08) | graph break 触发场景、resume function、partial graph 真实执行形态 |
| [[dynamic_shapes_generalization_and_fallback_analysis]] | deep dive(B09) | 动态形状泛化状态机、backed/unbacked symbol、泛化失败原因 |
| [[backend_contract_and_custom_backend_analysis]] | deep dive(B10) | backend 最小契约、custom backend 与 AOTAutograd 的组合边界 |
| [[control_flow_capture_analysis]] | deep dive(专题) | 控制流两路径:显式 HOP(`cond`/`while_loop`/`map`/`scan`)投机子图入图 vs 原生 `if`/`for`/`while` 字节码特化/展开/切图 |
| [[dynamo_pass_methodology]] | **development guide** | Dynamo backend 是什么/为什么、整图改写边界、`register_backend`/callable 注册示例、何时后移到 Inductor 各阶段 |

> 端到端流水线(Dynamo→AOTAutograd→Inductor)见 [[torch_compile_architecture]]。B01-B10 的课程化阅读顺序与配套 Demo 见 [[19_torch_compile_end_to_end/00_torch_compile_end_to_end_index]] §4。

---

## 关联域

- [[19_torch_compile_end_to_end/00_torch_compile_end_to_end_index]] — 编号化端到端课程：卷 B 系统展开 Dynamo，并连接卷 A、C–F
- [[02_compile_stack/02_aot_autograd/index]] — 下一阶段:前/反向分解
- [[02_compile_stack/04_inductor/index]] — 编译后端
- [[01_ai_frameworks/index]] — 本域总索引
