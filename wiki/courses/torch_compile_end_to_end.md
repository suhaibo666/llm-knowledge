# `torch.compile` 端到端阅读课程

> **本页角色**:纯导读页,不承载正文——只给阅读顺序、链接与每篇一句话导读。全部技术内容
> 已归属 `01_ai_frameworks` 五层功能树对应模块;本页过时只改链接/顺序,不要在此加正文。
> 固定源码基线:PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`;Lab 环境:PyTorch `2.9.1+cpu`。
> 最后更新:2026-07-30(kb-reorg P4 Task 10:课程页化,原 19 号目录两个 00 索引 + C01 动机页
> + `02_torch_compile_architecture` overview 页并入本页后四页删除,19 号目录整体解散)

---

## 这门课是什么

`torch.compile` 不是"把 Module 立刻转成一个 kernel"的单步骤 API。一次调用要经过捕获、
分解、编译、缓存、运行、失效重编译等多个阶段,每阶段有自己的数据结构和状态机。这条
阅读路线把知识库里分散在 [[01_ai_frameworks/index|01_ai_frameworks 五层架构]]
各模块的相关页面,串成一条从 eager 地基到生产部署的顺序索引。

**前置**:能日常使用 `Tensor`/`nn.Module`/`Optimizer`/`.backward()` 的 PyTorch eager 基础。
不要求提前读完 eager 运行时的全部深潜页——用到具体机制时再回查即可。

**三段式流水线速览**(建立在 eager 运行时地基之上,细节见下方各节):

```text
eager 运行时地基(Tensor/Dispatcher/ATen/autograd/nn.Module/运行时横切)
  → Dynamo:Python 帧捕获 + Guards → FX 图
  → AOTAutograd:functionalize + joint trace → 前向/反向两张 FX 图
  → Inductor:decomposition → FX passes → lowering(Inductor IR)→ Scheduler(融合/调度)→ CodeGen
  → Triton / C++ kernel + wrapper,编译执行
```

编译产物随后进入跨阶段缓存(命中即可跳过对应阶段)、运行时图捕获(CUDA/NPU Graphs 消除
下发开销)、调试诊断与生产上线;训练场景另叠加 Compiled Autograd、activation checkpoint、
分布式(DDP/FSDP/DTensor)与 AOTInductor 部署几条扩展线。

---

## 阅读路线(按目录顺序,目录内按段位 0→1→2→3)

### 0. 前置:eager 运行时地基

`01_eager_runtime` 是编译栈的地基,按需查阅,不要求线性通读:

| 模块 | 一句话 |
|---|---|
| [[01_eager_runtime/01_tensor_and_storage/index]] | 张量表达机制:`Tensor=intrusive_ptr<TensorImpl>`、Storage/视图别名、sizes/strides/dtype、DispatchKeySet |
| [[01_eager_runtime/02_dispatcher_and_device/index]] | 算子分发(Dispatcher)、PrivateUse1 设备接入 |
| [[01_eager_runtime/03_op_registration/index]] | 算子接入供给侧:`TORCH_LIBRARY`/torchgen 通用注册机制、NPU op-plugin |
| [[01_eager_runtime/04_aten_op_execution/index]] | ATen 算子定义与执行:`native_functions.yaml`、torchgen 代码生成、结构化 kernel、boxing |
| [[01_eager_runtime/05_autograd_engine/index]] | eager 反向自动微分引擎:Node/Edge DAG、多线程 Engine、AccumulateGrad、SavedVariable |
| [[01_eager_runtime/06_nn_module_system/index]] | `torch.nn` 模块体系:Module/Parameter/Buffer 注册、state_dict、hooks、容器、lazy、Optimizer |
| [[01_eager_runtime/07_memory_amp_profiler/index]] | 横切运行时:缓存内存分配器、AMP/autocast + GradScaler、Kineto Profiler |

> 原课程卷 A(执行模型五篇回顾)已于 P4 Task 3 判重删除;其"编译器为什么在乎"的独有分析
> 逐字并入以下页面对应节,不重复整理:[[10_tensor_impl_and_storage_analysis]] §13、
> [[10_pytorch_dispatcher_analysis]] §12、[[11_eval_frame_callback_and_code_cache_analysis]] §13、
> [[12_instruction_translator_and_bytecode_state_machine_analysis]] §14、
> [[10_dispatch_modes_proxytensor_faketensor_analysis]]、
> [[10_torch_compile_api_and_first_call_lifecycle_analysis]] §12、
> [[17_compile_latency_cache_and_steady_state_performance_analysis]] §12-§16。

### 1. Dynamo 图捕获

| 页面 | 段位 | 一句话 |
|---|---|---|
| [[01_dynamo_quickstart]] | 0 | 看捕获结果 `torch._dynamo.explain`、graph break 定位、`fullgraph=True`、guards/recompiles、逃生阀 |
| [[10_torch_compile_api_and_first_call_lifecycle_analysis]] | 1 | wrapper 创建与第一次真正编译如何区分、生命周期状态机 |
| [[11_eval_frame_callback_and_code_cache_analysis]] | 1 | eval-frame callback 三态协议、code cache 挂在 code object 上的原因 |
| [[12_instruction_translator_and_bytecode_state_machine_analysis]] | 1 | 字节码符号执行状态机、`run → step → opcode handler`、图断点重启 |
| [[13_variable_tracker_source_and_python_object_model_analysis]] | 1 | VariableTracker/VariableBuilder、Source 取值路径语法树 |
| [[14_output_graph_side_effects_and_graph_emission_analysis]] | 1 | OutputGraph 所有权边界、SideEffects、FX graph 如何交给 backend |
| [[15_guards_cache_lookup_and_recompilation_analysis]] | 1 | guard 树、cache lookup 精确顺序、重编译判定与两层上限 |
| [[16_graph_break_resume_functions_and_partial_graphs_analysis]] | 1 | graph break 触发场景、resume function、partial graph 真实执行形态 |
| [[17_dynamic_shapes_generalization_and_fallback_analysis]] | 1 | 动态形状泛化状态机、backed/unbacked symbol、泛化失败原因(Dynamo 侧行为面) |
| [[18_backend_contract_and_custom_backend_analysis]] | 1 | backend 最小契约、custom backend 与 AOTAutograd 的组合边界 |
| [[20_symbolic_shapes_guards_and_graph_reuse_analysis]] | 2 | 符号形状/Guard/图复用**概念权威页**:ShapeEnv、SymNode、`DimDynamic`、guard 生成、backed/unbacked |
| [[21_control_flow_capture_analysis]] | 2 | 控制流两路径:显式 HOP(`cond`/`while_loop`/`map`/`scan`)子图入图 vs 原生 `if`/`for`/`while` 特化/展开 |
| [[22_backend_modes_options_stances_and_fullgraph_analysis]] | 2 | backend/mode/options/stance/fullgraph 五个控制面怎样互不覆盖 |
| [[30_dynamo_pass_methodology]] | 3 | Dynamo backend 是什么/为什么、整图改写边界、`register_backend` 注册示例 |

### 2. AOTAutograd 前反向分解

| 页面 | 段位 | 一句话 |
|---|---|---|
| [[01_aot_autograd_quickstart]] | 0 | 看前/反向图、看联合图、partitioner(min-cut vs default)、`aot_function` 最小用法 |
| [[10_dispatch_modes_proxytensor_faketensor_analysis]] | 1 | `__torch_function__`/`__torch_dispatch__`/ProxyTensor/FakeTensor 四层分工,`make_fx` 如何协同捕获联合图 |
| [[11_aotautograd_joint_forward_backward_graphs_analysis]] | 1 | joint graph 构造(primals/tangents)、partition 如何抽取 fresh fw/bw Graph、输出/输入 ABI 分层 |
| [[12_saved_tensors_recompute_and_runtime_abi_analysis]] | 1 | saved value 分类、min-cut rematerialization、recompute 节点复制、运行时 autograd.Function 拼接 |
| [[13_aot_runtime_wrappers_and_lazy_backward_compile_analysis]] | 1 | fw/bw callable 包回 eager autograd 协议的 runtime wrapper、lazy backward compile 触发时机 |
| [[20_activation_checkpoint_recompute_and_compile_analysis]] | 2 | `torch.utils.checkpoint`(reentrant/non-reentrant、Selective AC)如何与 partitioner 的 save/recompute 叠加 |

### 3. Graph IR 与 Passes

`03_graph_ir_and_passes` 不含 quickstart,段 1 是 IR 基础(如何得到这张图),段 2 是建立在
其上的 pass 机制层:

| 页面 | 段位 | 一句话 |
|---|---|---|
| [[10_fx_graph_core_data_model_analysis]] | 1 | `Graph`/`Node`/`GraphModule` 五个核心对象、侵入式双向链表、`args/kwargs` 与 `users` 反向邻接 |
| [[11_graph_values_metadata_and_signatures_analysis]] | 1 | Node 引用、`meta["val"]`、pytree 结构化输出与三类 graph signature |
| [[12_graph_effects_alias_mutation_and_order_analysis]] | 1 | 数据边之外的 alias、mutation、functionalization 与 effect 顺序 |
| [[13_structured_outputs_higher_order_and_nested_graphs_analysis]] | 1 | 多输出、HOP(`cond`/`while_loop`/`map`)与 nested GraphModule 如何扩展普通 DAG |
| [[14_graph_capture_frontends_and_tracing_analysis]] | 1 | `symbolic_trace`、`make_fx`、Dynamo、`export` 四种捕获前端为何产生不同图 |
| [[15_graph_normalization_decomposition_and_functionalization_analysis]] | 1 | schema normalization、decomposition 与 functionalization 为何必须分层 |
| [[20_graph_stage_boundaries_identity_and_provenance_analysis]] | 2 | Node identity 跨阶段(capture→AOT→Inductor)断开后如何维持 provenance |
| [[21_fx_graph_editing_primitives_and_invariants_analysis]] | 2 | replace/erase/copy/lint/recompile 组成的安全改图事务、10 项检查清单 |
| [[22_pattern_expression_and_matcher_engine_analysis]] | 2 | PatternExpr AST、候选桶索引、逆序 matcher、三类 Entry 与序列化 pattern |
| [[23_dead_code_topology_and_effect_order_analysis]] | 2 | FX DCE/Scheduler DCE、stable topological sort、拓扑正确 ≠ effect 正确 |
| [[24_graph_pass_pipeline_ordering_and_fixpoint_analysis]] | 2 | pass stage、注册顺序、迭代与 fixpoint 如何决定改写结果 |
| [[25_graph_rewrite_legality_validation_and_complexity_analysis]] | 2 | 结构命中后 shape/dtype/alias/autograd 的合法性验证、全链路复杂度 |

> **为什么需要这层 IR**(原 C01 动机页导读级要点,正文价值已由上表各页承载):eager 单个算子
> 被调用时编译器看不到后续读取、无用中间值、可融合的算子对与反向所需激活;图 IR 把部分语义
> 冻结成可检查结构,使 producer/consumer、save/recompute、DCE、重排/融合/分区、shape 复用性、
> kernel 溯源变得可计算。PyTorch 里"图"不止一种——eager autograd tape、FX program graph、
> AOT joint/fw/bw graph、Inductor IR、Scheduler dependency graph、CUDA Graph 各自的节点/边/
> 生命周期都不同,看到"图"先问是哪一层,不要默认它们可以互相替代。

### 4. Inductor 编译后端

Inductor 内部五阶段一览(原 overview 页要点,`compile_fx` 编排入口见下表 15 号页):

| 阶段 | 一句话 | 对应页面 |
|---|---|---|
| Decomposition | 把复合 ATen 算子拆解为原语算子,收敛后续 lowering 要处理的算子集 | [[33_decomposition_passes_guide]] |
| FX Graph Passes | pre-grad(高层重写)→ joint-graph(常量折叠/模式匹配)→ post-grad(底层融合/设备相关重写) | [[30_pre_grad_passes_guide]] / [[31_joint_graph_passes_guide]] / [[32_post_grad_passes_guide]] |
| Lowering | ATen 算子逐一翻译为 Inductor IR(`lowerings[target]` 注册表),`Pointwise`/`Reduction` 等循环原语 | [[10_fx_lowering_to_inductor_ir_analysis]] |
| Scheduler | IR 节点依赖分析,决定融合(水平/垂直)、执行顺序、内存规划与缓冲区复用 | [[13_scheduler_dependency_graph_fusion_and_ordering_analysis]] |
| CodeGen | 调度后的 IR 翻译成 Triton(GPU)/C++(CPU)kernel + 驱动 wrapper,期间 autotuning | [[20_inductor_codegen_analysis]] |

| 页面 | 段位 | 一句话 |
|---|---|---|
| [[01_inductor_quickstart]] | 0 | 最小 fwd+bwd 示例、`torch.compile` 参数与 `torch._inductor.config` 速查、怎么看生成代码 |
| [[10_fx_lowering_to_inductor_ir_analysis]] | 1 | FX → Inductor IR:注册/fallback/layout,为什么 FX 与代码生成之间还需要一层 IR |
| [[11_inductor_ir_values_loops_layouts_and_buffers_analysis]] | 1 | `TensorBox`/`Loops`/`Layout`/`Buffer`/`ExternKernel` 如何分工 |
| [[12_buffer_liveness_memory_planning_and_reuse_analysis]] | 1 | **内存分配管理权威页**:编译期 realize/last-use/reuse → 运行期 allocator 物理池 → CUDA Graphs 私有池 |
| [[13_scheduler_dependency_graph_fusion_and_ordering_analysis]] | 1 | Scheduler 依赖/融合决策、`_pre/_post_fusion_custom_pass` 真实签名 |
| [[14_codegen_kernel_mapping_autotuning_and_provenance_analysis]] | 1 | Scheduler group 怎样映射到 kernel、wrapper、autotune 和 provenance |
| [[15_inductor_compile_fx_orchestration_analysis]] | 1 | **compile_fx 编排入口**:为什么先调用 AOTAutograd、wrapper ABI 归一化、fw/bw compiler 分工 |
| [[20_inductor_codegen_analysis]] | 2 | 现有代码生成策略、kernel、wrapper 与调用链完整参考 |
| [[21_inductor_autotuning_analysis]] | 2 | Autotune 生命周期(`CachingAutotuner`)、config 启发式、`triton.compile`→PTX/cubin |
| [[22_inductor_reduction_codegen_deep_analysis]] | 2 | Reduction codegen:persistent / looped / split / cooperative(semaphore barrier) |
| [[23_inductor_gpu_kernel_dispatch_model]] | 2 | GPU kernel 骨架(`program_id→offset→index→mask`)、`IterationRanges` 树 |
| [[24_inductor_codegen_dynamic_shape_analysis]] | 2 | 代码生成中的动态形状,XBLOCK 选择与性能代价 |
| [[25_unbacked_symint_analysis]] | 2 | Unbacked SymInt:数据相关 shape、deferred_runtime_asserts、`torch._check()` |
| [[26_flex_attention_analysis]] | 2 | FlexAttention:可组合注意力融合、BlockMask、score_mod |
| [[27_async_compile_workers_and_module_loading_analysis]] | 2 | 编译任务如何异步完成并加载为 module |
| [[28_aotinductor_packaging_and_deployment_analysis]] | 2 | **AOTInductor 打包与部署**:JIT cache 与 AOT package 产物差异、PT2 archive/call spec/C ABI runner |
| [[30_pre_grad_passes_guide]] | 3 | Pre-Grad pass 开发指南 |
| [[31_joint_graph_passes_guide]] | 3 | Joint pass 开发指南 |
| [[32_post_grad_passes_guide]] | 3 | Post-Grad pass 开发指南 |
| [[33_decomposition_passes_guide]] | 3 | Decomposition 开发指南 |
| [[34_codegen_extension_guide]] | 3 | Codegen 扩展开发指南(`BaseScheduling`、Wrapper、`DeviceOpOverrides`) |
| [[35_inductor_memory_allocation_guide]] | 3 | 内存分配实战指南:分配全过程走查、分配器对照、越界/踩踏排查 |

> NPU Inductor 后端(Ascend 适配,非 upstream)见 [[02_compile_stack/04_inductor/npu/index]]。

### 5. 跨阶段编译缓存

| 页面 | 一句话 |
|---|---|
| [[02_compile_stack/06_compile_cache/index]] | overview:七层 cache(Dynamo code/PGO、AOTAutograd、Inductor FXGraphCache、source/module、Triton kernel、autotune、CUDA Graph runtime)为何不是一张统一的表 |
| [[10_dynamo_pgo_cache_analysis]] | PGO 画像的状态合并、local/remote/sticky key 与"soundly stale"边界 |
| [[11_aotautograd_cache_analysis]] | AOT 级 key、entry 形态、命中 wrapper 链与 bypass 条件 |
| [[12_fx_graph_cache_analysis]] | post-grad 图指纹、GuardedCache、`CompiledFxGraph` 与 Triton bundling |
| [[13_triton_autotune_cache_analysis]] | kernel 级 autotune winner、remote cache 与 Triton 磁盘 cache |

### 6. 调试与诊断

十篇按证据层级、失败分层定位、修复策略与生产上线组织,**建议阅读顺序** = 表格顺序:

| 页面 | 一句话 |
|---|---|
| [[10_observability_logs_counters_and_artifact_map_analysis]] | 建立证据层级:log、artifact、counter 分别能证明什么、不能证明什么 |
| [[11_dynamo_explain_and_graph_break_diagnosis_analysis]] | 用 `explain` 与 `graph_breaks` 定位捕获失败与切图原因 |
| [[12_guard_failure_and_recompile_diagnosis_analysis]] | 定位 recompile storm:cache entry 选择失败的根因分类与修复 |
| [[13_aotautograd_and_inductor_failure_localization_analysis]] | 用 backend 阶梯(eager→aot_eager→decomp/partition→inductor)做失败分层二分 |
| [[14_compiled_artifact_lifecycle_and_runtime_failures_analysis]] | 编译产物 build/serialize/load/post-compile/first-call/replay 状态机与六类 failure taxonomy |
| [[15_minifier_repro_and_compiler_bisector_analysis]] | Repro、Minifier 与 Compiler Bisector 三个正交工具的选用边界 |
| [[16_compiled_correctness_validation_methodology_analysis]] | 值、梯度、mutation、alias、effect 六维正确性验证方法论 |
| [[17_compile_latency_cache_and_steady_state_performance_analysis]] | 冷启动、cache hit、稳态三类场景分开测量与 break-even 分析 |
| [[18_kernel_fusion_memory_and_hardware_performance_analysis]] | 从图结构到 fusion、内存生命周期、硬件计数器的四层性能归因 |
| [[19_production_rollout_fallback_and_monitoring_analysis]] | 分阶段上线、fallback 分级、SLO 与自动回退、回滚演练 |

> 附:分布式 + torch.compile 全链路排查脚本包(`run_debug.sh`/`collect_artifacts.sh`/
> `diff_rank_logs.sh` 等)见 [[02_compile_stack/07_debugging/index]] 附录,九篇正文之外的
> 操作性附件,不在此页重复列出命令。

### 7. 运行时图捕获:CUDA Graphs

| 页面 | 段位 | 一句话 |
|---|---|---|
| [[01_PyTorch_CUDA_Graphs_Complete_Guide]] | 0 | CUDA Graphs 完整指南:四种用法、实现原理、时序图、综合比较、最佳实践 |
| [[10_cudagraph_trees_warmup_record_and_replay_analysis]] | 1 | `CUDAGraphNode`/`CUDAGraphTreeManager`、按静态输入地址分流的多份 recording、warmup→record→replay→fallback 状态机 |
| [[20_training_inference_cudagraph_and_freezing_analysis]] | 2 | training/inference/freezing/CUDA Graph 四轴组合边界、`freezing.py` 变换链、`cudagraphify` 地址不变式 |

> NPU Graphs(ACLGraph)是另一硬件线,见 [[03_runtime_graphs/npu/index]],本课程只覆盖
> upstream/CUDA 视角。

### 8. 图导出与算子扩展

| 页面 | 段位 | 一句话 |
|---|---|---|
| [[01_fx_export_custom_op_quickstart]] | 0 | `symbolic_trace` + 改写 `Graph`;`export` + `dynamic_shapes`;`torch.library.custom_op` 注册;`vmap`/`functional_call` |
| [[10_fx_graph_export_and_custom_ops_analysis]] | 1 | Proxy 拦截与 `TracerBase.create_proxy`、`GraphModule` 代码生成、`ExportedProgram` 的 lifted params/约束、`Library`/`custom_op` 分发与 autograd 桥接 |
| [[20_custom_operators_fake_kernels_and_decompositions_analysis]] | 2 | custom op 作为"编译器边界契约":fake kernel 正确性要求、mutation/version、decomposition/direct lowering/fallback 选择 |

### 9. 分布式原语

| 页面 | 段位 | 一句话 |
|---|---|---|
| [[01_distributed_primitives_quickstart]] | 0 | `init_process_group`/`init_device_mesh`;集合原语最小示例;DDP/FSDP2/DTensor/TP 最小用法 |
| [[10_c10d_ddp_fsdp_dtensor_analysis]] | 1 | 源码级机制:Backend 注册、DDP `Reducer` 分桶、FSDP `FlatParameter` shard/unshard、DTensor placement 传播 |
| [[20_ddp_compile_boundaries_and_optimizer_analysis]] | 2 | DDPOptimizer 为何按 DDP bucket 逆序切分 Dynamo forward 图、optimizer 是否入图的三种边界 |
| [[21_fsdp_dtensor_and_distributed_graphs_analysis]] | 2 | 分布式图不是一张全局 FX 图、FSDP1 `use_orig_params=True` 为何是编译前提、DTensor 在编译图中的形态 |

### 10. 训练扩展收尾

以上各表已收纳大多数扩展场景页面(activation checkpoint 见 §2、DDP/FSDP compile 边界
见 §9、custom op 见 §8、AOTInductor 部署见 §4);仍留在 `01_eager_runtime` 内、未被上述
任一目录表覆盖的两页在此补齐:

| 页面 | 一句话 |
|---|---|
| [[20_compiled_autograd_analysis]] | Compiled Autograd:同一 C++ engine 的第三种运行模式,把运行时反向"录制"成 FX 图交给 Dynamo/Inductor 编译执行 |
| [[20_custom_backends_and_device_integration_analysis]] | 设备接入 `torch.compile` 的另一层契约:Dynamo backend / Inductor device backend / dispatcher-custom-op backend 三者如何分层组合 |

### 三条捷径(按需选择,不必线性通读全部)

- **建立完整心智模型**:按上表 0→9 顺序通读。
- **捕获与编译器开发**:eager 地基(按需)→ §1 全部 → §3 全部 → §4 段 0-1 → §6 前三篇。
- **线上性能与稳定性**:§1(quickstart+guards+dynamic shapes)→ §4(内存/scheduler/codegen)→
  §5 全部 → §6 全部 → §7 全部。

---

## Labs 使用说明

六卷各一个统一入口,每个入口含多个可独立选择的 case;完整命令、状态语义与页面映射见
[`tools/labs_torch_compile/README.md`](../../tools/labs_torch_compile/README.md) 与
`tools/labs_torch_compile/demo_manifest.json`。

| 入口 | 主题 |
|---|---|
| `tools/labs_torch_compile/demo_a_execution_model.py` | Tensor/dispatcher/frame/Proxy-FakeTensor/成本(CPU 机制脚本,不再对应独立页面) |
| `tools/labs_torch_compile/demo_b_dynamo_capture.py` | lifecycle/cache/bytecode/guard/break/dynamic/backend,对应 §1 |
| `tools/labs_torch_compile/demo_c_graph_compiler.py` | FX/AOTAutograd/Inductor 图编译,对应 §2-§4 |
| `tools/labs_torch_compile/demo_d_artifact_runtime.py` | compile_fx/AOT wrapper/cache/module load/内存/CUDAGraph,对应 §4-§5、§7 |
| `tools/labs_torch_compile/demo_e_diagnostics.py` | explain/recompile/故障分层/repro/正确性/性能/回退,对应 §6 |
| `tools/labs_torch_compile/demo_f_advanced_topics.py` | compiled autograd/checkpoint/distributed/custom/AOTI,对应 §9-§10 |

统一运行契约:

```powershell
python -B tools\labs_torch_compile\demo_b_dynamo_capture.py `
  --case guards_recompile --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\b07
```

`--list --json` 查看某入口的全部 case;默认 `--device cuda`,无 CUDA 的机器可传
`--device cpu` 预检设备无关机制,但不能替代 GPU/Triton/CUDAGraph/多卡验收。

- `PASS`:case 正文真实执行,且内置断言全部通过;
- `BLOCKED`:声明的 CUDA/Triton/native compiler/多卡等能力缺失,正文没有执行;
- `FAIL`:正文已经执行,但编译阶段、子进程、断言或 runtime 失败。

不要把一次 `PASS` 外推到其他 PyTorch 版本、shape、dtype 或硬件。

---

## 与功能树的关系

`01_ai_frameworks` 五层功能树([[01_ai_frameworks/index]])是唯一的内容权威;本页(以及
`wiki/courses/` 下未来的其它课程页)只是这棵树之上的一条阅读顺序索引,不持有正文,也不会
成为第二份真相来源。发现内容缺失、过时或与功能树矛盾:去对应模块的 index.md 或深潜页修改,
再回到本页只更新链接、顺序或一句话导读。

原 `19_torch_compile_end_to_end/` 课程目录(A-F 六卷叙事、63 篇独立课程正文)已随 kb-reorg
P4 逐任务解散:A 卷判重删除(Task 3)、B/E 卷整体迁入 `01_dynamo`/`07_debugging`(Task 4-5)、
D/C/F 卷按内容拆散到 `02_aot_autograd`/`03_graph_ir_and_passes`/`04_inductor`/
`03_runtime_graphs/cuda`/`04_export_and_distributed`/`01_eager_runtime` 对应模块
(Task 6-9),两个 `00_*_index.md` 总索引与 C01 动机页、`04_inductor/02_torch_compile_architecture.md`
overview 页的导读价值最终并入本页(Task 10),19 号目录整体删除。

NPU 后端(Ascend 适配,非 upstream)不在本课程覆盖范围,见
[[02_compile_stack/04_inductor/npu/index]] 与 [[03_runtime_graphs/npu/index]]。

---

## Related Pages

- [[01_ai_frameworks/index]] — 功能树总索引(唯一权威)
- [[02_compile_stack/index]] — torch.compile 编译栈领域索引
- [[02_compile_stack/01_dynamo/index]] — Dynamo 领域索引
- [[02_compile_stack/02_aot_autograd/index]] — AOTAutograd 领域索引
- [[02_compile_stack/03_graph_ir_and_passes/index]] — Graph IR/Passes 领域索引
- [[02_compile_stack/04_inductor/index]] — Inductor 领域索引
- [[02_compile_stack/06_compile_cache/index]] — 编译缓存领域索引
- [[02_compile_stack/07_debugging/index]] — 调试诊断领域索引
- [[03_runtime_graphs/index]] — 运行时图捕获领域索引
- [[04_export_and_distributed/01_fx_export_extensibility/index]] — 图导出与算子扩展领域索引
- [[04_export_and_distributed/02_distributed_primitives/index]] — 分布式原语领域索引
- `tools/labs_torch_compile/README.md` — Demo 命令、状态与证据说明
