# 04 · TorchInductor — 目录索引

> PyTorch 编译全链路扩展面:Dynamo backend → Pre-Grad → AOT/Decomposition → Joint → Post-Grad → Lowering(FX→Inductor IR) → Scheduler(调度/融合) → CodeGen。
> 阅读路径 **overview → quick start → deep dive**(约定见 [[01_ai_frameworks/index]])。本目录为 **upstream**;**NPU Inductor 后端**单独见 [[02_compile_stack/04_inductor/npu/index]]。
> 最后更新: 2026-07-27

---

## overview(概览,先读这里)

| 页面 | 核心主题 |
|------|---------|
| [[torch_compile_architecture]] | **Inductor 概览**:是什么/为什么、在 torch.compile 中的位置、五阶段一览、核心概念(IR/Scheduler/CodeGen)、由浅入深导航 |
| [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]] | **当前图编译系统化主线**：从 FX IR、AOT 正反向图和改图合法性，一直走到 Inductor IR、调度、内存规划与 codegen；固定源码基线并配套可执行 Lab |

### 课程主线与子系统参考分工

| 需求 | 入口 |
|---|---|
| 建立“为什么这样设计”的连续心智模型并运行Lab | [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]] |
| FX→IR职责与fallback/custom lowering | [[19_torch_compile_end_to_end/17_fx_lowering_to_inductor_ir]] |
| IR value/loop/layout/buffer与index | [[19_torch_compile_end_to_end/18_inductor_ir_values_loops_layouts_and_buffers]] |
| liveness、reuse、静态peak与runtime边界 | [[19_torch_compile_end_to_end/19_buffer_liveness_memory_planning_and_reuse]] |
| Scheduler dependency/fusion/reorder | [[19_torch_compile_end_to_end/20_scheduler_dependency_graph_fusion_and_ordering]] |
| kernel/wrapper/autotune/provenance | [[19_torch_compile_end_to_end/21_codegen_kernel_mapping_autotuning_and_provenance]] |
| 查某个子系统的完整函数/API清单 | 本目录下`lowering_analysis`、`scheduler_analysis`、`inductor_codegen_analysis`等专题页 |

autotuning负责搜索/测量候选并选择winner；[[02_compile_stack/06_compile_cache/index]]讨论winner、graph
artifact与compiled module如何复用。两者相邻但不是同一机制：一次autotune可能写cache，
一次cache hit也可能完全跳过新的benchmark。

## quick start(上手)

| 页面 | 核心主题 |
|------|---------|
| [[inductor_quickstart]] | 最小 fwd+bwd 示例、`torch.compile` 参数(mode/dynamic/fullgraph/options)、关键 `torch._inductor.config` 与环境变量、mode 选型、看生成代码 |

---

## deep dive — 端到端与全局

| 页面 | 核心主题 |
|------|---------|
| [[inductor_compile_fx_orchestration_analysis]] | **compile_fx 编排入口**:为什么先调用 AOTAutograd、wrapper ABI 归一化、fw/bw compiler 分工、产物层次、`compile_fx→_compile_fx_main→AOTAutograd`源码跟读;§0 附带全链路全景图与各阶段深挖入口导航表(原"脊柱文档" `inductor_compiler_pipeline_analysis` 921 行的逐阶段走读已被本目录各专题页更深入覆盖,判重后删除,2026-07-30) |
| [[PyTorch_Inductor_Technical_Analysis]] | **后端选择 & IR 优化深度**:后端选择/配置、Inductor IR 数据结构、融合成本模型与坐标下降 autotune、常量折叠、内存规划/内存池、CUDA Graphs 集成、后端扩展(pipeline 未展开的纵深) |
| [[inductor_memory_management_analysis]] | **内存分配管理(全栈三层)**:编译期 buffer 复用/峰值重排/池化规划(`memory_plan_reuse`·`reorder_for_peak_memory`·`memory_planning.py`)→ 运行期 `CUDACachingAllocator` → CUDA Graphs `cudagraph_trees` 跨图共享私有池 + checkpoint;含**池大小如何确定**(§2.6)+ 段大小档位(§3) |
| [[inductor_memory_allocation_guide]] | **内存分配实战指南(guide)**:实际分配全过程走查 + 分配器对照(native/cudaMallocAsync/expandable)+ `memory_stats`/snapshot 实测复现 + **内存越界/踩踏排查**(mask/size_asserts 内置防护、compute-sanitizer)+ 实践建议 |
| [[wrapper_execution_memory_allocation_and_reuse_analysis]] | **Wrapper 执行与 buffer reuse(源码级)**:boxed calling convention、`MemoryPlanningLine` IR(Allocate/FreeIfNotReused/Reuse)、reuse key、view/alias 对 reuse 的限制;与上两页视角重叠,归一进行中(见页头互指) |

> `torch.compile` 源码入口、调用栈、函数签名、mode 对照见 [[02_compile_stack/01_dynamo/index]]（B01/B02，2026-07-30 起随 P4 判重并入,不再在本目录单列）。

## deep dive — 各编译阶段

| 页面 | 核心主题 |
|------|---------|
| [[decomposition_passes_guide]] | **Decomposition 开发**：是什么/为什么、AOT 注入位置、关键 API、注册示例、与 Graph Pattern/Lowering 的选择边界 |
| [[lowering_analysis]] | **FX → Inductor IR**：注册/IR/fallback/layout API、接入示例、为什么在此阶段 |
| [[scheduler_analysis]] | **调度与融合**：依赖/融合决策、`_pre/_post_fusion_custom_pass` 真实签名、错误旧接口辨析 |
| [[inductor_codegen_analysis]] | 现有代码生成策略、kernel、wrapper 与调用链 |
| [[codegen_extension_guide]] | **Codegen 开发**：`BaseScheduling`、Wrapper、`DeviceOpOverrides`、设备注册骨架与验证清单 |

## deep dive — codegen 派发与运行时（GPU 基线）

| 页面 | 核心主题 |
|------|---------|
| [[inductor_gpu_kernel_dispatch_model]] | GPU kernel 骨架（`program_id→offset→index→mask`，kernel 内无循环）、`IterationRanges` 树、stride-1 tiling、`Grid1D/2D/2DWithYZOverflow/CooperativeReductionGrid` |
| [[inductor_reduction_codegen_deep_analysis]] | Reduction codegen：persistent / looped / split / cooperative（semaphore barrier）、block ptr / TMA |
| [[inductor_autotuning_analysis]] | Autotune 生命周期（`CachingAutotuner`）、config 启发式、`config_of`/AttrsDescriptor、`make_launcher`、`triton.compile`→PTX/cubin、`DeviceProperties` |

## deep dive — FX Passes

| 页面 | 核心主题 |
|------|---------|
| [[02_compile_stack/03_graph_ir_and_passes/index]] | **FX 构图与改图底座 + Pass 方法论总纲**:Node/Graph 双向 use-def、PatternExpr AST(含 `fwd_only`/`joint_fwd_bwd`)、候选桶与逆序匹配、序列化 pattern 缓存、DCE、稳定拓扑排序、三阶段 driver 机制(`GraphTransformObserver`/`GroupBatchFusionBase`)、八阶段选型方法论、rewrite 合法性与复杂度、跨框架(torch_npu/vLLM/SGLang)对照(AOT fw/bw/recompute 见 [[02_compile_stack/02_aot_autograd/index]]) |
| [[pre_grad_passes_guide]] | Pre-Grad 真实顺序、主要 Pass、关键 API、custom/Pattern 注册示例与动态形状边界 |
| [[joint_graph_passes_guide]] | Joint 真实顺序、两轮 `pass_patterns`、切图前方法论与 custom hook 示例 |
| [[post_grad_passes_guide]] | Post-Grad 真实顺序、三轮 pattern、通信/mutation 尾部约束与 inference-aware hook |

## deep dive — 动态形状

> 符号形状系统本身(ShapeEnv/SymNode/guard 生成/DimDynamic/backed·unbacked 判定)的概念权威页见 [[02_compile_stack/01_dynamo/index]] 的 [[symbolic_shapes_guards_and_graph_reuse_analysis]](2026-07-30 起随 P4 判重并入,不再在本目录单列)。本目录只保留 Inductor 侧的两篇专项:

| 页面 | 核心主题 |
|------|---------|
| [[inductor_codegen_dynamic_shape_analysis]] | 代码生成中的动态形状,XBLOCK 选择与性能代价 |
| [[unbacked_symint_analysis]] | Unbacked SymInt:数据相关 shape、deferred_runtime_asserts、torch._check() |

## deep dive — 专题与调试

| 页面 | 核心主题 |
|------|---------|
| [[flex_attention_analysis]] | FlexAttention:可组合注意力融合、BlockMask、score_mod、语义驱动 codegen |
| [[02_compile_stack/07_debugging/index]] | torch.compile 调试:`TORCH_LOGS`/`TORCH_COMPILE_DEBUG`、九篇分层诊断 + 分布式排查脚本包(纯 upstream;NPU 调试见 [[npu_debug_guide]]) |

---

## 硬件子目录

| 目录 | 核心主题 |
|------|---------|
| [[02_compile_stack/04_inductor/npu/index]] | **NPU Inductor 后端**:三条 compile 路径、lowering/fallback、monkey patch、优化思想、NPU 调试 |

---

## Related Pages

- [[19_torch_compile_end_to_end/00_torch_compile_end_to_end_index]] — 编号化端到端课程：卷 C 的 lowering/codegen 与卷 D 的 artifact/runtime
- [[02_compile_stack/01_dynamo/index]] — 上游:图捕获
- [[02_compile_stack/02_aot_autograd/index]] — 上游:前/反向分解
- [[19_torch_compile_end_to_end/17_fx_lowering_to_inductor_ir]] — 当前基线的 FX → Inductor IR 边界
- [[19_torch_compile_end_to_end/18_inductor_ir_values_loops_layouts_and_buffers]] — 当前基线的IR值、循环、layout与buffer
- [[19_torch_compile_end_to_end/19_buffer_liveness_memory_planning_and_reuse]] — 当前基线的liveness、reuse与peak边界
- [[19_torch_compile_end_to_end/20_scheduler_dependency_graph_fusion_and_ordering]] — 当前基线的 scheduler 依赖图、融合与保序
- [[19_torch_compile_end_to_end/21_codegen_kernel_mapping_autotuning_and_provenance]] — 当前基线的 codegen、autotune 与 provenance
- [[02_compile_stack/06_compile_cache/index]] — 跨阶段cache与artifact复用
- [[02_compile_stack/05_codegen_backends/index]] — codegen 后端(MLIR/Triton)
- [[03_runtime_graphs/index]] — 运行时图捕获
- [[01_ai_frameworks/index]] — 本域总索引
