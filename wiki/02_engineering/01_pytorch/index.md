---
title: "PyTorch 编译与运行时架构 — 知识地图"
---

# PyTorch 编译与运行时架构 — 知识地图

> 本域（`01_pytorch`）按 **PyTorch 编译/运行时架构**组织。每个功能目录下，**硬件无关的通用机制**置于该目录层级，**硬件特定实现**下沉到 `npu/`、`cuda/` 等硬件子目录。
> 最后更新：2026-08-27（域目录 `01_ai_frameworks` → `01_pytorch`；原 ⑤ `05_other_frameworks`（MindSpore 编译器分析）已删除，本域现为纯 PyTorch 四层）

---

## 模块定位：做什么 · 提供什么能力 · 边界在哪

**一句话**：PyTorch 不是"和训练/推理框架并列的又一个框架"，而是**其余三域共同的底座**——它定义了张量是什么、算子如何派发到设备、反向如何求解、图如何被捕获与编译。[[02_engineering/02_train_frameworks/index|训练框架]]、[[02_engineering/03_infer_frameworks/index|推理框架]]、[[02_engineering/04_posttrain_frameworks/index|后训练框架]] 各自解决"怎么编排"，但它们编排的**对象**（`Tensor` / `Module` / `ProcessGroup` / `DTensor` / 编译产物）全部由本域定义。

**为什么必须独立成一层**：上层框架的差异是编排策略，本域的差异是**语义**。同一句 `a @ b` 要经过"张量表达 → dispatcher 选实现 → autograd 记带 → （可选）被 Dynamo 捕获 → AOTAutograd 分解 → Inductor 生成 kernel → 可能被 CUDA Graph 回放"这条链；任何一环的语义改变（新设备、新精度、新并行原语）都会同时影响上面三个域。把这条链单独成域，是为了让上层只依赖**它的契约**，而不是它的实现。

### 本域的来源构成与硬件分级

本域虽然只围绕一个框架，但**上游 PyTorch 与厂商适配层是两套代码库**，页面的可核验性不同：

| 来源 | 覆盖范围 | 本库页数 | 基线 |
|---|---|---|---|
| **上游 `pytorch/pytorch`** | 四层架构的硬件无关部分：张量/dispatcher/autograd/Dynamo/AOTAutograd/Inductor/export/distributed | ~116 篇（本域主体） | 侧车 checkout `ea5655fc`（2026-07-20）；**各页正文基线以页头为准**，部分页固定在更早的 2.x 版本 |
| **`torch_npu`（昇腾适配）** | 下沉在 `npu/` 子目录：PrivateUse1 算子接入、Inductor NPU 后端、ACLGraph、NPU MLIR 后端 | 28 篇 | 侧车 checkout `b3c8a815b`（2026-07-15），见各页头 |
| **CUDA 特定实现** | 下沉在 `cuda/` 子目录：CUDA Graphs 与 cudagraph trees 的具体接线 | 4 篇 | 随上游 |

> 目录约定：**硬件无关的通用机制放在功能目录本层，硬件特定实现下沉到 `npu/`、`cuda/`**。读某个机制时先读本层（这是语义），再按你的硬件读子目录（这是实现差异）。反过来读容易把某一家的实现细节误当成 PyTorch 的通用语义。

### 本域提供的能力

（源码锚点按侧车 checkout `pytorch/pytorch@ea5655fc`（2026-07-20）核对路径存在；各页正文的具体基线以页头为准）：

| 层 | 能力 | 具体提供什么 | 源码锚点 |
|---|---|---|---|
| ① eager 地基 | **张量与存储表达** | shape/stride/dtype/device/storage 的数据模型，view 与别名语义 | `c10/core/TensorImpl.h` |
| ① | **算子派发与设备接入** | DispatchKey 决定同一算子调哪个实现；PrivateUse1 是第三方设备（NPU 等）的接入点 | `aten/src/ATen/core/dispatch/Dispatcher.h` · `aten/src/ATen/native/native_functions.yaml` |
| ① | **eager 反向自动微分** | 运行时动态记带 + C++ 引擎按依赖拓扑回放 | `torch/csrc/autograd/engine.cpp` |
| ① | **模块与优化器体系** | 参数/缓冲区归属、state_dict 语义、优化器状态 | `torch/nn/` · `torch/optim/` |
| ① | **运行时横切** | 缓存分配器（显存复用与碎片）、AMP 自动混精、Profiler | `c10/core/CachingDeviceAllocator.h` · `torch/amp/` · `torch/profiler/` |
| ② 编译栈 | **图捕获** | 字节码层面捕获 Python 函数为 FX 图，用 Guard 保证复用条件 | `torch/_dynamo/convert_frame.py` · `torch/_dynamo/guards.py` |
| ② | **前/反向分解** | 把 forward 与 backward 联合成一张图再切分，决定哪些中间量重算而非保存 | `torch/_functorch/aot_autograd.py` · `torch/_functorch/partitioners.py` |
| ② | **图 IR 与改图原语** | FX 数据模型、PatternMatcher、pre/joint/post-grad pass 三个改图时机 | `torch/fx/` · `torch/_inductor/fx_passes/` |
| ② | **lowering·调度·codegen** | ATen 图降到 Inductor IR、按依赖图做融合与排序、生成 Triton/C++ kernel | `torch/_inductor/lowering.py` · `torch/_inductor/scheduler.py` |
| ② | **跨阶段编译缓存** | Dynamo PGO / AOTAutograd cache / FX graph cache，各自跳过不同阶段的既有工作 | `torch/_dynamo/pgo.py` · `torch/_functorch/_aot_autograd/autograd_cache.py` · `torch/_inductor/codecache.py` |
| ③ 运行时图 | **CUDA/NPU Graph 捕获与回放** | 消除 launch 开销；与 `mode="reduce-overhead"` 经 cudagraph trees 集成 | `torch/cuda/graphs.py` · `torch/_inductor/cudagraph_trees.py` |
| ④ 导出与分布式 | **导出与可扩展性** | `torch.export` 的 ExportedProgram、`torch.library` 自定义算子契约 | `torch/export/` · `torch/library.py` |
| ④ | **分布式原语** | c10d/ProcessGroup、DeviceMesh、DTensor 分片语义、FSDP2、pipelining——**这是 `02_train_frameworks` 的底座** | `torch/distributed/device_mesh.py` · `torch/distributed/tensor/` · `torch/distributed/fsdp/` · `torch/distributed/pipelining/` |

### 不属于本模块的

- 训练任务的编排（并行策略组合、通信掩盖、checkpoint 体系）→ [[02_engineering/02_train_frameworks/index|训练框架]]；本域只提供 DTensor/FSDP/PP **原语**，不负责怎么把它们组合成 5D 并行；
- kernel 内部的 tile 切分、寄存器账本、流水设计 → [[02_engineering/05_gpu_kernel/index|GPU Kernel 开发]]；本域到 codegen 出 Triton 源码为止；
- 自动求解并行策略 → [[02_engineering/06_auto_parallel/index|自动并行]]。

### 与兄弟域的关系
本域是全栈唯一的语义源头——`02` 用它的分布式原语搭并行，`03` 用它的编译栈与 CUDA Graph 搭推理引擎，`04` 同时消费 `02` 和 `03`。反过来，本域**不依赖**任何上层域；域内页面若需要举上层例子，链出去而不是把上层机制搬进来。

---

## 架构总览：两条主轴

PyTorch 可拆成两条相互支撑的主轴：**① eager 运行时地基**（默认即时执行所依赖的数据模型与机制）与 **② torch.compile 编译栈**（把 eager 计算捕获、分解并编译成 kernel）。编译栈始终建立在运行时地基之上，二者共享同一套 Tensor/Dispatcher 底座。

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

> 关键对照：**01_eager_runtime/05_autograd_engine**（运行时动态磁带 + C++ 引擎）是 **02_compile_stack/02_aot_autograd**（编译期前/反向联合 FX 图）的 eager 对应物。两者容易混淆，具体区别见各自 index 中的对照表。
> 编译栈端到端流水线详见 [[02_compile_stack/04_inductor/index]]；eager 地基从 [[01_eager_runtime/01_tensor_and_storage/index]] 读起。

---

## 功能目录（四层架构导航）

`01_pytorch` 按 4 个架构层、两级目录组织，每层目录下再划分功能子目录（逐级导航见各层的 `index.md`）。

| 层 | 目录 | 功能 |
|---|------|------|
| ① eager 运行时地基 | [[01_eager_runtime/index]] | 张量表达、Dispatcher/PrivateUse1 设备接入、算子接入供给侧(op-plugin)、ATen 算子定义与执行、eager 反向自动微分引擎、torch.nn 模块体系、运行时横切(缓存分配器/AMP/Profiler) |
| ② torch.compile 编译栈 | [[02_compile_stack/index]] | Dynamo 图捕获、AOTAutograd 前/反向分解、Graph IR/Passes(FX 数据模型/改图原语/PatternMatcher/DCE)、TorchInductor lowering/调度/codegen(`npu/` NPU 后端)、codegen 后端(MLIR)、跨阶段编译缓存、调试诊断(证据层级/失败分层定位/正确性与性能验收/生产上线) |
| ③ 运行时图捕获 | [[03_runtime_graphs/index]] | CUDA Graphs / NPU Graphs(ACLGraph),与 Inductor `mode="reduce-overhead"` 经 cudagraph trees 集成 |
| ④ 图导出与分布式 | [[04_export_and_distributed/index]] | torch.fx(eager 图 IR)、torch.export(ExportedProgram)、torch.library/custom_op、functorch(vmap/grad);torch.distributed 原生原语:c10d/ProcessGroup、DDP、FSDP/FSDP2、DTensor/DeviceMesh、TP/PP(**[[02_train_frameworks/index]] 的底座**) |

缓存（② 层 `06_compile_cache`）不是 codegen 之后新增的 IR 阶段，而是横跨整个生命周期、用于“跳过既有工作”的机制：

```text
Dynamo PGO → 复用动态行为/shape经验
AOTAutograd cache → 复用functionalization、joint/partition与编译结果
Inductor FX graph cache → 复用lowering/codegen artifact
Triton autotune/kernel cache → 复用候选winner与已编译kernel
```

阅读顺序仍是捕获 → AOT → Inductor → codegen；随后从 [[02_compile_stack/06_compile_cache/index]] 反向检查每一层的
cache hit 究竟跳过了哪些阶段，不能把“命中 cache”笼统理解为整条编译栈都未运行。

课程入口（跨 5 层的纯导读页，正文全部归属上表功能树，不计入上表）：[[courses/torch_compile_end_to_end|torch.compile 端到端课程]]——
提供从 eager 地基、Dynamo 捕获、AOTAutograd 分解、Graph IR/Passes、Inductor 编译、跨阶段缓存，
到调试诊断、运行时图捕获、导出与分布式的完整阅读路线及 labs 对应表（自 2026-07-30 kb-reorg P4
Task 10 起，原 19 号课程目录已整体解散，内容归并到上表 5 层）。

---

## 硬件分层约定

- **通用页**（无硬件后缀）位于功能目录层级，描述 PyTorch 上游通用机制。
- **NPU 特定页**位于 `<功能>/npu/`，基于 `torch_npu` / `op-plugin`（当前核验基准 = **v2.7.1.post5**）。
- **CUDA 特定页**位于 `<功能>/cuda/`。
- 跨硬件对比页（如 NPU vs CUDA Graphs）放在硬件目录内，并添加双向 backlink。
- eager 地基模块（00/10–13）当前均为 upstream 通用页（核验基准 = PyTorch `E:\97-codes\pytorch\pytorch`，v2.13.0a0）；其 NPU 特化内容（NPU 分配器、HCCL ProcessGroup、AutogradPrivateUse1 等）按需下沉到各自的 `npu/`（见下方路线图）。

---

## 知识分层约定（overview → quick start → deep dive）

每个（子）模块内的知识按由浅入深的顺序分为三层，索引页使用 **层次** 列标注：

- **overview**（浅）：说明模块是什么、为什么需要它、全景结构和选型方法，通常是各目录的 `index.md` 或显式 overview 页，约 5 分钟可读完。
- **quick start**（用）：提供最小可用路径、关键 API、可运行示例和常用开关；并非每个模块都需要。
- **deep dive**（深）：源码级深度分析（多为既有长文），按主题拆分，每页聚焦一个概念。

阅读建议：先通过 overview 建立全局认识，需要动手时看 quick start，钻研细节时再看 deep dive。

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
- [[20_batch_invariance_guide]] — 批不变性与 torch.compile
- [[01_theory/index]] — 理论研究
