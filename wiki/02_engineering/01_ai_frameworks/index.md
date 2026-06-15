# PyTorch 编译与运行时架构 — 知识地图

> 本域(`01_ai_frameworks`)按 **PyTorch 编译/运行时架构**组织。每个功能目录下,**硬件无关的通用机制**置于该目录层级,**硬件特定实现**下沉到 `npu/`、`cuda/` 等硬件子目录。
> 最后更新: 2026-06-15

---

## 架构总览:两条主轴

PyTorch 可拆成相互支撑的两条主轴——**① eager 运行时地基**(默认即时执行所依赖的数据模型与机制)与 **② torch.compile 编译栈**(把 eager 计算捕获、分解、编译成 kernel)。编译栈始终建立在地基之上,二者共享同一套 Tensor/Dispatcher 底座。

```
                         ┌─────────────────── ② torch.compile 编译栈 ───────────────────┐
   算子供给侧            @torch.compile
 (07_op_registration)   User Code ─► TorchDynamo ─► AOTAutograd ─► TorchInductor ─► Kernel 执行
        │                              (02)            (03)            (04)            │
        │                            图捕获/Guard    前/反向分解     lowering→codegen   │
        │                                                              │               │
        ▼                                                     codegen 后端(05)   图捕获/回放(06)
 ┌──────────────────────────────── ① eager 运行时地基 ──────────────────────────────────┐
 │ 00 Tensor/Storage(张量表达) ── 01 Dispatcher(分发) ── 11 ATen 算子定义/执行          │
 │ 10 eager autograd 引擎(.backward())  12 torch.nn 模块/Optimizer                       │
 │ 13 运行时:缓存分配器 / AMP / Profiler    14 torch.fx/export/扩展   15 torch.distributed │
 └──────────────────────────────────────────────────────────────────────────────────────┘
```

> 关键对照:**10_eager_autograd**(运行时动态磁带 + C++ 引擎)是 **03_aot_autograd**(编译期前/反向联合 FX 图)的 eager 对应物;两者易混,见各自 index 的对照表。
> 编译栈端到端流水线详见 [[torch_compile_architecture]];eager 地基从 [[00_tensor_and_storage/index]] 读起。

---

## 功能目录(按架构分层)

### ① eager 运行时地基(2026-06-15 新增)

| 目录 | 功能 | 硬件子目录 |
|------|------|-----------|
| [[00_tensor_and_storage/index]] | 张量表达机制:`Tensor=intrusive_ptr<TensorImpl>`、Storage/视图别名、sizes/strides/dtype、张量上的 DispatchKeySet | 通用 |
| [[10_eager_autograd/index]] | eager 反向自动微分引擎:Node/Edge DAG、多线程 Engine、AccumulateGrad、SavedVariable、自定义 Function(**区别于 03 编译期**) | 通用 |
| [[11_aten_op_execution/index]] | ATen 算子定义与执行:`native_functions.yaml`、torchgen 代码生成、结构化 kernel(meta/impl)、boxing(**07 的上游通用版**) | 通用 |
| [[12_nn_module_system/index]] | torch.nn 模块体系:Module/Parameter/Buffer 注册、state_dict、hooks、容器、lazy、Optimizer | 通用 |
| [[13_runtime_memory_amp_profiler/index]] | 横切运行时:缓存内存分配器、AMP/autocast + GradScaler、Kineto Profiler | 通用 |

### ② torch.compile 编译栈 + 运行时分发(既有)

| 目录 | 功能 | 硬件子目录 |
|------|------|-----------|
| [[01_dispatcher_and_device/index]] | 运行时:算子分发(Dispatcher)、PrivateUse1 设备接入 | 通用 |
| [[02_dynamo/index]] | torch.compile 前端:帧评估图捕获、Guard | 通用 |
| [[03_aot_autograd/index]] | 前/反向图分解、partition、functionalization(**编译期 autograd**) | 通用 |
| [[04_inductor/index]] | 编译后端核心:lowering、调度、codegen、FX passes、动态形状 | `npu/` NPU Inductor 后端 |
| [[05_codegen_backends/index]] | codegen 后端:MLIR(及 Triton 对比) | `mlir/` + `mlir/npu/` |
| [[06_graphs/index]] | 运行时图捕获:CUDA Graphs / NPU Graphs(ACLGraph) | `cuda/` + `npu/` |
| [[07_op_registration/index]] | 算子接入供给侧:op-plugin 配置/注册/入图判别 | `npu/` |
| [[08_kernel_optimization/index]] | 算子调优(GPU/NPU Roofline、融合)、TileLang | 通用 |

### ③ 图/扩展与分布式

| 目录 | 功能 | 硬件子目录 |
|------|------|-----------|
| [[14_fx_export_and_extensibility/index]] | torch.fx(eager 图 IR)、torch.export(ExportedProgram)、torch.library/custom_op、functorch(vmap/grad) | 通用 |
| [[15_distributed_primitives/index]] | torch.distributed 原生原语:c10d/ProcessGroup、DDP Reducer、FSDP/FSDP2、DTensor/DeviceMesh、TP/PP(**[[02_train_frameworks/index]] 的底座**) | 通用 |
| [[09_other_frameworks/index]] | 非 PyTorch 框架对照:MindSpore 编译器 | — |

> 编号说明:数字前缀仅决定文件系统排序;**逻辑阅读顺序以本表分层为准**(地基 00/10–13 → 编译栈 01–08 → 扩展/分布式 14/15/09)。新模块用 `00_` 与 `10_+` 前缀,以**不重命名既有目录、不破坏既有 wikilink**。

---

## 硬件分层约定

- **通用页**(无硬件后缀)位于功能目录层级,描述 PyTorch 上游通用机制。
- **NPU 特定页**位于 `<功能>/npu/`,基于 `torch_npu` / `op-plugin`(当前核验基准 = **v2.7.1.post5**)。
- **CUDA 特定页**位于 `<功能>/cuda/`。
- 跨硬件对比页(如 NPU vs CUDA Graphs)放硬件目录内并双向 backlink。
- eager 地基模块(00/10–13)当前均为 upstream 通用页(核验基准 = PyTorch `E:\97-codes\pytorch\pytorch`,v2.13.0a0);其 NPU 特化(NPU 分配器、HCCL ProcessGroup、AutogradPrivateUse1 等)按需下沉到各自 `npu/`(见下方路线图)。

---

## 知识分层约定(overview → quick start → deep dive)

每个(子)模块内的知识按「深入浅出」分三层,索引页用 **层次** 列标注:

- **overview**(浅):该模块是什么 / 为什么 / 全景图 / 如何选型,5 分钟通读;通常即各目录的 `index.md` 或显式 overview 页。
- **quick start**(用):最小可用路径、关键 API、可跑示例、常见开关(并非每模块都有,按需)。
- **deep dive**(深):源码级深度分析(多为既有大文),按主题拆分,一概念一页。

阅读建议:先 overview 建立全局 → 需要动手看 quick start → 钻研细节看 deep dive。

---

## 规划路线图与知识空白

### ✅ 已补齐(2026-06-15 · eager 运行时地基,Workflow 编排 + 源码逐行核实)

针对「wiki 过度偏向 torch.compile 编译栈、缺 eager 运行时地基」的审计结论,新增 **7 个模块共 21 页**(P0+P1),均对照 `E:\97-codes\pytorch\pytorch`(v2.13.0a0)核实行号:
[[00_tensor_and_storage/index]]、[[10_eager_autograd/index]]、[[11_aten_op_execution/index]]、[[12_nn_module_system/index]]、[[13_runtime_memory_amp_profiler/index]]、[[14_fx_export_and_extensibility/index]]、[[15_distributed_primitives/index]]。

### 📋 规划任务(按优先级)

- **[P2] 持久化与遗留 `16_serialization_and_legacy`**(spec 已就绪,本轮暂缓):序列化(`torch.save/load` zip+pickle、`weights_only` 安全模型)、TorchScript/JIT(已废弃,迁移见 14)、C++/CUDA 扩展(`cpp_extension`)。建议三页:overview / `save_load_and_cpp_extension_quickstart` / `serialization_jit_and_cpp_extensions_analysis`。
- **[P2] 新模块的 NPU 特化下沉**:为 00/10/11/13/15 按需建 `npu/` 子目录——NPU 缓存分配器与内存复用(与 [[06_graphs/npu/index]] 联动)、`HCCL`/`ProcessGroupHCCL`、`AutogradPrivateUse1` 反向注册、aclnn 结构化 kernel 对照;upstream 页仅留指向 `npu/` 的指针。
- **[P1→补深] 11_aten**:补一条 `yaml → torchgen → 生成 C++(RegisterCPU.cpp)` 的**具体生成实例**走读(当前以机制为主)。
- **[P1→补深] 15_distributed**:`ProcessGroupNCCL` 具体实现、FSDP2(`_composable/fsdp`)与 FSDP1 差异、pipeline 调度(1F1B/interleaved)细节;与 [[02_train_frameworks/index]] 划清「原语 vs 应用」边界后双向补链。

### 既有空白(沿用)

- **TorchDynamo guard 失败调试** — 常见但未系统记录(可并入 [[02_dynamo/index]])
- **Inductor autotuning** — Triton kernel autotuning 策略
- **NPU Monkey Patch 演进追踪** — v2.7.1 → v2.9.0 → master,每次 PyTorch 升级需人工对齐内部接口
- **CATLASS/CK GEMM 模板库生态** — 社区 CUTLASS 与 NPU CATLASS 差异
- **IR 回溯机制通用性** — MLIR 路径 FX Graph 重建的泛化方案
- **Multi-backend dispatch** — Inductor 在 CUDA/NPU 间选择逻辑(部分覆盖,见 [[npu_compile_paths_overview]])
- **IREE 实际 Pass 细节** — Flow/Stream Dialect 具体 Pass 列表
- **TileLang 源码分析** — 实现未开源,当前分析基于论文
- **Triton 3.x MLIR 迁移进度** — TMA 以外特性支持状态

---

## 关联域

- [[02_train_frameworks/megatron-lm/index]] — Megatron-LM(CUDA Graphs 使用场景)
- [[02_train_frameworks/index]] — 训练框架:建立在 [[15_distributed_primitives/index]] 之上的并行应用层
- [[05_gpu_kernel/index]] — GPU Kernel 开发(执行层级、内存优化、NPU 差异)
- [[04_posttrain_frameworks/batch_invariance_guide]] — 批不变性与 torch.compile
- [[01_theory/index]] — 理论研究
