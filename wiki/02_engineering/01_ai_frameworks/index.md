# PyTorch 编译与运行时架构 — 知识地图

> 本域(`01_ai_frameworks`)按 **PyTorch 编译/运行时架构**组织。每个功能目录下,**硬件无关的通用机制**置于该目录层级,**硬件特定实现**下沉到 `npu/`、`cuda/` 等硬件子目录。
> 最后更新: 2026-07-30

---

## 架构总览:两条主轴

PyTorch 可拆成相互支撑的两条主轴——**① eager 运行时地基**(默认即时执行所依赖的数据模型与机制)与 **② torch.compile 编译栈**(把 eager 计算捕获、分解、编译成 kernel)。编译栈始终建立在地基之上,二者共享同一套 Tensor/Dispatcher 底座。

```
                         ┌─────────────────── ② torch.compile 编译栈(02_compile_stack) ───────────────────┐
   算子供给侧             @torch.compile
 (01_eager_runtime/      User Code ─► TorchDynamo ─► AOTAutograd ─► TorchInductor ─► Kernel 执行
  03_op_registration)                  (01_dynamo)   (02_aot_autograd) (04_inductor)          │
        │                             图捕获/Guard    前/反向分解      lowering→codegen        │
        │                                                                  │                  │
        ▼                                                    codegen 后端(05_codegen_backends)  图捕获/回放(03_runtime_graphs) │
 ┌──────────────────────────────── ① eager 运行时地基(01_eager_runtime) ──────────────────────────────┐
 │ 01 Tensor/Storage(张量表达) ── 02 Dispatcher(分发) ── 04 ATen 算子定义/执行                        │
 │ 05 eager autograd 引擎(.backward())  06 torch.nn 模块/Optimizer                                    │
 │ 07 运行时:缓存分配器 / AMP / Profiler                                                              │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
   (04_export_and_distributed:torch.fx/export/扩展、torch.distributed 原语 —— 见 ④ 图导出与分布式)
```

> 关键对照:**01_eager_runtime/05_autograd_engine**(运行时动态磁带 + C++ 引擎)是 **02_compile_stack/02_aot_autograd**(编译期前/反向联合 FX 图)的 eager 对应物;两者易混,见各自 index 的对照表。
> 编译栈端到端流水线详见 [[02_compile_stack/04_inductor/index]];eager 地基从 [[01_eager_runtime/01_tensor_and_storage/index]] 读起。

---

## 功能目录(五层架构导航)

`01_ai_frameworks` 按 5 个架构层两级组织,每层目录下再分功能子目录(逐子目录导航见各层自己的 `index.md`)。

| 层 | 目录 | 功能 |
|---|------|------|
| ① eager 运行时地基 | [[01_eager_runtime/index]] | 张量表达、Dispatcher/PrivateUse1 设备接入、算子接入供给侧(op-plugin)、ATen 算子定义与执行、eager 反向自动微分引擎、torch.nn 模块体系、运行时横切(缓存分配器/AMP/Profiler) |
| ② torch.compile 编译栈 | [[02_compile_stack/index]] | Dynamo 图捕获、AOTAutograd 前/反向分解、Graph IR/Passes(待填充)、TorchInductor lowering/调度/codegen(`npu/` NPU 后端)、codegen 后端(MLIR)、跨阶段编译缓存、调试诊断(待填充) |
| ③ 运行时图捕获 | [[03_runtime_graphs/index]] | CUDA Graphs / NPU Graphs(ACLGraph),与 Inductor `mode="reduce-overhead"` 经 cudagraph trees 集成 |
| ④ 图导出与分布式 | [[04_export_and_distributed/index]] | torch.fx(eager 图 IR)、torch.export(ExportedProgram)、torch.library/custom_op、functorch(vmap/grad);torch.distributed 原生原语:c10d/ProcessGroup、DDP、FSDP/FSDP2、DTensor/DeviceMesh、TP/PP(**[[02_train_frameworks/index]] 的底座**) |
| ⑤ 其它框架对照 | [[05_other_frameworks/index]] | 非 PyTorch 框架编译器/架构分析(MindSpore 等),与本域横向对照 |

缓存(② 层 `06_compile_cache`)不是 codegen 之后新增的一种 IR 阶段,而是横跨生命周期的"跳过既有工作"机制:

```text
Dynamo PGO → 复用动态行为/shape经验
AOTAutograd cache → 复用functionalization、joint/partition与编译结果
Inductor FX graph cache → 复用lowering/codegen artifact
Triton autotune/kernel cache → 复用候选winner与已编译kernel
```

阅读顺序仍是捕获→AOT→Inductor→codegen；随后从[[02_compile_stack/06_compile_cache/index]]反向检查每一层
cache hit究竟跳过了哪些阶段，不能把“命中cache”笼统理解成整条编译栈都未运行。

课程入口(跨 5 层的纯导读页,正文全部归属上表功能树,不计入上表):[[courses/torch_compile_end_to_end|torch.compile 端到端课程]] ——
从 eager 地基到 Dynamo 捕获、AOTAutograd 分解、Graph IR/Passes、Inductor 编译、跨阶段缓存、
调试诊断、运行时图捕获、导出与分布式的完整阅读路线 + labs 对应表(2026-07-30 kb-reorg P4
Task 10 起,原 19 号课程目录已整体解散,内容归并进上表 5 层)。

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
[[01_eager_runtime/01_tensor_and_storage/index]]、[[01_eager_runtime/05_autograd_engine/index]]、[[01_eager_runtime/04_aten_op_execution/index]]、[[01_eager_runtime/06_nn_module_system/index]]、[[01_eager_runtime/07_memory_amp_profiler/index]]、[[04_export_and_distributed/01_fx_export_extensibility/index]]、[[04_export_and_distributed/02_distributed_primitives/index]]。

### 📋 规划任务(按优先级)

- **[P2] 持久化与遗留 `18_serialization_and_legacy`**(spec 已就绪,本轮暂缓):序列化(`torch.save/load` zip+pickle、`weights_only` 安全模型)、TorchScript/JIT(已废弃,迁移见 [[04_export_and_distributed/01_fx_export_extensibility/index]])、C++/CUDA 扩展(`cpp_extension`)。编号 16 曾由图编译主线占用;该主线现已归并到 19。17_compile_cache 已随 P4 两级重组迁至 `02_compile_stack/06_compile_cache`(编号跟随新层结构变化);本条目若未来落地,按 P4 后的 5 层结构选址,不再沿用旧扁平编号。建议三页:overview / `save_load_and_cpp_extension_quickstart` / `serialization_jit_and_cpp_extensions_analysis`。
- **[P2] 新模块的 NPU 特化下沉**:为 `01_eager_runtime` 的 01/04/05/07(tensor_and_storage/aten_op_execution/autograd_engine/memory_amp_profiler)与 `04_export_and_distributed/02_distributed_primitives` 按需建 `npu/` 子目录——NPU 缓存分配器与内存复用(与 [[03_runtime_graphs/npu/index]] 联动)、`HCCL`/`ProcessGroupHCCL`、`AutogradPrivateUse1` 反向注册、aclnn 结构化 kernel 对照;upstream 页仅留指向 `npu/` 的指针。
- **[P1→补深] 01_eager_runtime/04_aten_op_execution**:补一条 `yaml → torchgen → 生成 C++(RegisterCPU.cpp)` 的**具体生成实例**走读(当前以机制为主)。
- **[P1→补深] 04_export_and_distributed/02_distributed_primitives**:`ProcessGroupNCCL` 具体实现、FSDP2(`_composable/fsdp`)与 FSDP1 差异、pipeline 调度(1F1B/interleaved)细节;与 [[02_train_frameworks/index]] 划清「原语 vs 应用」边界后双向补链。

### 既有空白(沿用)

- **TorchDynamo guard 失败调试** — 常见但未系统记录(可并入 [[02_compile_stack/01_dynamo/index]])
- **Inductor autotuning** — ✅ 已补 [[21_inductor_autotuning_analysis]]（CachingAutotuner / config_of / triton.compile）、[[23_inductor_gpu_kernel_dispatch_model]]、[[22_inductor_reduction_codegen_deep_analysis]]
- **NPU Monkey Patch 演进追踪** — ✅ 当前横向总览已补 [[21_torch_npu_upstream_adaptation_analysis]]，完成 v2.7.1 与 2026-07-15 upstream main 的标准插件面/硬件面/补丁面分类；逐 release 的 v2.7.1 → v2.9.0 → master 符号级增删仍待持续维护（v2.9.0 实验性 Linearize 后端见 [[23_npu_inductor_linearize_backend_analysis]]）
- **CATLASS/CK GEMM 模板库生态** — 社区 CUTLASS 与 NPU CATLASS 差异
- **IR 回溯机制通用性** — MLIR 路径 FX Graph 重建的泛化方案
- **Multi-backend dispatch** — Inductor 在 CUDA/NPU 间选择逻辑(部分覆盖,见 [[01_npu_compile_paths_overview]])
- **IREE 实际 Pass 细节** — Flow/Stream Dialect 具体 Pass 列表
- **TileLang 源码分析** — 实现未开源,当前分析基于论文
- **Triton 3.x MLIR 迁移进度** — TMA 以外特性支持状态

---

## Related Pages

- [[courses/torch_compile_end_to_end]] — torch.compile 端到端阅读课程:从 API 到生产运行,图编译系统化主线
- [[02_compile_stack/06_compile_cache/index]] — 跨阶段编译缓存地图
- [[02_train_frameworks/megatron-lm/index]] — Megatron-LM(CUDA Graphs 使用场景)
- [[02_train_frameworks/index]] — 训练框架:建立在 [[04_export_and_distributed/02_distributed_primitives/index]] 之上的并行应用层
- [[05_gpu_kernel/index]] — GPU Kernel 开发(执行层级、内存优化、NPU 差异)
- [[04_posttrain_frameworks/batch_invariance_guide]] — 批不变性与 torch.compile
- [[01_theory/index]] — 理论研究
