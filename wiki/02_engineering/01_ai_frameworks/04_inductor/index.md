# 04 · TorchInductor — 目录索引

> PyTorch 编译后端核心:Decomposition → FX Passes → Lowering(FX→Inductor IR)→ Scheduler(调度/融合)→ CodeGen。
> 阅读路径 **overview → quick start → deep dive**(约定见 [[01_ai_frameworks/index]])。本目录为 **upstream**;**NPU Inductor 后端**单独见 [[04_inductor/npu/index]]。
> 最后更新: 2026-06-17

---

## overview(概览,先读这里)

| 页面 | 核心主题 |
|------|---------|
| [[torch_compile_architecture]] | **Inductor 概览**:是什么/为什么、在 torch.compile 中的位置、五阶段一览、核心概念(IR/Scheduler/CodeGen)、由浅入深导航 |

## quick start(上手)

| 页面 | 核心主题 |
|------|---------|
| [[inductor_quickstart]] | 最小 fwd+bwd 示例、`torch.compile` 参数(mode/dynamic/fullgraph/options)、关键 `torch._inductor.config` 与环境变量、mode 选型、看生成代码 |

---

## deep dive — 端到端与全局

| 页面 | 核心主题 |
|------|---------|
| [[inductor_compiler_pipeline_analysis]] | **端到端编译管线**:Eager→Dynamo→AOT→Decomp→FX Passes→Lowering→Scheduler→CodeGen 逐阶段源码级走读(脊柱文档) |
| [[PyTorch_Inductor_Technical_Analysis]] | **后端选择 & IR 优化深度**:后端选择/配置、Inductor IR 数据结构、融合成本模型与坐标下降 autotune、常量折叠、内存规划/内存池、CUDA Graphs 集成、后端扩展(pipeline 未展开的纵深) |
| [[inductor_memory_management_analysis]] | **内存分配管理(全栈三层)**:编译期 buffer 复用/峰值重排/池化规划(`memory_plan_reuse`·`reorder_for_peak_memory`·`memory_planning.py`)→ 运行期 `CUDACachingAllocator` → CUDA Graphs `cudagraph_trees` 跨图共享私有池 + checkpoint;含**池大小如何确定**(§2.6)+ 段大小档位(§3) |
| [[inductor_memory_allocation_guide]] | **内存分配实战指南(guide)**:实际分配全过程走查 + 分配器对照(native/cudaMallocAsync/expandable)+ `memory_stats`/snapshot 实测复现 + **内存越界/踩踏排查**(mask/size_asserts 内置防护、compute-sanitizer)+ 实践建议 |
| [[torch_compile_source_analysis]] | torch.compile 源码入口、调用栈、函数签名、mode 对照、能力边界 |

## deep dive — 各编译阶段

| 页面 | 核心主题 |
|------|---------|
| [[lowering_analysis]] | FX → Inductor IR lowering(注册/API/优化) |
| [[scheduler_analysis]] | 算子调度器、融合决策;自定义融合 Pass 与排查;新设备 backend 注册(设备无关示例) |
| [[inductor_codegen_analysis]] | 代码生成策略、kernel 融合、wrapper |

## deep dive — codegen 派发与运行时（GPU 基线）

| 页面 | 核心主题 |
|------|---------|
| [[inductor_gpu_kernel_dispatch_model]] | GPU kernel 骨架（`program_id→offset→index→mask`，kernel 内无循环）、`IterationRanges` 树、stride-1 tiling、`Grid1D/2D/2DWithYZOverflow/CooperativeReductionGrid` |
| [[inductor_reduction_codegen_deep_analysis]] | Reduction codegen：persistent / looped / split / cooperative（semaphore barrier）、block ptr / TMA |
| [[inductor_autotuning_analysis]] | Autotune 生命周期（`CachingAutotuner`）、config 启发式、`config_of`/AttrsDescriptor、`make_launcher`、`triton.compile`→PTX/cubin、`DeviceProperties` |

## deep dive — FX Passes

| 页面 | 核心主题 |
|------|---------|
| [[pre_grad_passes_guide]] | 预梯度 passes(`fx_passes/pre_grad.py`) |
| [[joint_graph_passes_guide]] | 联合图 passes(`fx_passes/joint_graph.py`) |
| [[post_grad_passes_guide]] | 后梯度 passes(`fx_passes/post_grad.py`) |

## deep dive — 动态形状

| 页面 | 核心主题 |
|------|---------|
| [[dynamic_shapes_full_analysis]] | Dynamic Shape 全链路:静态特化→符号化→Guard→渐进动态化,ShapeEnv |
| [[inductor_codegen_dynamic_shape_analysis]] | 代码生成中的动态形状,XBLOCK 选择与性能代价 |
| [[unbacked_symint_analysis]] | Unbacked SymInt:数据相关 shape、deferred_runtime_asserts、torch._check() |

## deep dive — 专题与调试

| 页面 | 核心主题 |
|------|---------|
| [[flex_attention_analysis]] | FlexAttention:可组合注意力融合、BlockMask、score_mod、语义驱动 codegen |
| [[Pytorch_Compile_Debug_Analysis]] | torch.compile 调试:`TORCH_LOGS`/`TORCH_COMPILE_DEBUG`、日志解读(纯 upstream;NPU 调试见 [[npu_debug_guide]]) |

---

## 硬件子目录

| 目录 | 核心主题 |
|------|---------|
| [[04_inductor/npu/index]] | **NPU Inductor 后端**:三条 compile 路径、lowering/fallback、monkey patch、优化思想、NPU 调试 |

---

## 关联域

- [[02_dynamo/index]] — 上游:图捕获
- [[03_aot_autograd/index]] — 上游:前/反向分解
- [[05_codegen_backends/index]] — codegen 后端(MLIR/Triton)
- [[06_graphs/index]] — 运行时图捕获
- [[01_ai_frameworks/index]] — 本域总索引
