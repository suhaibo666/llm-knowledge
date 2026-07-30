# torch_npu torch.compile 适配全景分析

003e Type: Analysis
003e Created: 2026-05-13
003e Source: torch_npu v2.7.1 源码级分析

# torch_npu torch.compile 适配情况深度分析

> 分析范围：Inductor 后端适配、ACLGraph 图执行适配  
> 基于版本：torch_npu v2.7.1（含 v2.9.0 / master 演进趋势）  
> 分析日期：2026-05-13  
> 最后更新：2026-07-15（订正“社区统一 Triton 路径”旧口径）

---

## 一、概述

torch_npu 对 `torch.compile()` 的适配横跨 Dynamo 图捕获、Inductor 编译优化、ACLGraph 图执行三个层面。与社区 CUDA/XPU 后端相比，NPU 的适配策略呈现出**"表面标准化、底层深度补丁"**的特征：虽然使用了 `register_backend_for_device` 等社区接口，但在 Inductor 内部执行了 35+ monkey patches（v2.7.1），维护了三条独立的 codegen 路径，并重建了整套 ACLGraph 机制来替代 CUDAGraph。

本文从差异对比、原因分析、收益评估、演进路径四个维度展开。

---

## 二、Inductor 层：torch_npu 与社区的差异

### 2.1 表面相似，底层 divergence 极大

torch_npu 使用了社区提供的设备注册接口：

```python
# torch_npu/_inductor/__init__.py:89
register_backend_for_device('npu', NPUCombinedScheduling, NPUWrapperCodeGen, CppWrapperNpu)
```

这看起来和 XPU 等新设备的接入方式一致，但 NPU 在注册的同时执行了 **30+ monkey patches**（v2.7.1），覆盖范围横跨：

| 被 Patch 的上游模块 | Patch 数量 | 典型目标 |
|---|---|---|
| `torch._inductor.codegen.*` | ~8 | wrapper 生成、Triton scheduling、IR indexing、loop body |
| `torch._inductor.scheduler.*` | ~2 | `are_long_distant_nodes` 阈值、CATLASS/Triton 模板互换 |
| `torch._inductor.lowering.*` | ~3 | `make_reduction`、`make_fallback`、NPU 专属 decomposition |
| `torch._inductor.runtime.*` | ~4 | autotune cache、triton heuristics、device properties |
| `torch._inductor.compile_fx.*` | ~2 | cudagraphify、CPP wrapper 路径 |
| `torch._inductor.utils.*` | ~4 | `is_gpu`、`has_triton`、`GPU_TYPES`、cudagraph 安全检查 |
| `torch._dynamo.eval_frame.*` | ~1 | Shape handling 拦截 |

**社区设备（如 XPU）的典型接入方式**：仅需实现 `DeviceOpOverrides` + `Scheduling` + `WrapperCodeGen`，无需 patch 上游内部函数。而 NPU 的 `NPUCombinedScheduling` 甚至直接继承自 `CUDACombinedScheduling`：

```python
# torch_npu/_inductor/codegen/npu_combined_scheduling.py:23
class NPUCombinedScheduling(CUDACombinedScheduling):
```

这说明 NPU 不是作为一个独立的、平等的后端接入，而是**寄生在 CUDA 代码路径之上做 delta 修改**。

### 2.2 三条编译路径 vs 社区的统一 Triton 路径

> [!deprecated] Updated by [[21_torch_npu_upstream_adaptation_analysis]]
> “社区统一 Triton 路径”是较早版本口径。2026-07-15 的 upstream `main` 中，CUDA/XPU 也已通过 combined scheduling 混合 Triton、CUTLASS、CuteDSL、C++ 等 codegen；当前核心差异应改看“公开 scheduling/wrapper/policy 接口”与“直接 patch 私有实现”的边界。本节保留原文作为 v2.7.1 时点记录。

NPU 的 Inductor 有三条互斥路径，由 `TORCHINDUCTOR_NPU_BACKEND` 控制：

| 路径 | 实现位置 | 技术栈 | 适用场景 |
|---|---|---|---|
| `default` | `torch_npu/_inductor/` | Triton-like codegen + CATLASS + C++ wrapper | 通用训练/推理 |
| `mlir` | `torch_npu/_inductor/ascend_npu_ir/` | torch-mlir + AKG compiler | 图模式融合 |
| `dvm` | `torch_npu/_inductor/dvm/` | DVM (Dynamic Virtual Machine) | 特定 kernel 场景 |

**社区差异**：CUDA/XPU 只有一条 Triton-based 路径。NPU 的 MLIR 路径是一个完全独立的 codegen 生态——它有自己的 `NpuMlirScheduling`、`NpuMlirWrapperCodeGen`、`NpuMlirCompiler`，甚至有自己的 IR patch（`ascend_npu_ir/npu/inductor_patch/ir.py`、`scheduler.py`）。

这意味着 torch_npu 实际维护的不是一个后端，而是 **三个独立后端 + 一个兼容层**。

### 2.3 Lowering 能力差距：约 963 个 op fallback

`torch_npu/_inductor/lowering_fallback_list.py` 的统计结果：

- **约 963 个算子**被强制 fallback 到 ACLNN（348 native + 615 npu-extra，截至 v2.7.1；不进入 Inductor lowering/fusion）
- **135 个 `prims.*` 算子**被强制 fallback
- 总计 **~994 个算子**无法被 Inductor 优化

这些 fallback 的类别包括：
- 所有分布式通信算子（`all_gather`、`all_reduce`、`broadcast` 等）
- 位运算（`__and__`、`__or__`、`__lshift__`、`__rshift__` 等）
- 大量 higher-order ops（`cond`、`while_loop`、`associative_scan`）
- 随机数/量化相关算子

**社区对比**：CUDA 的 fallback list 要小一个数量级，大多数标准 aten op 都有对应的 Triton lowering。

### 2.4 GEMM 算子：社区用 Triton，NPU 用 CATLASS + CK + ATen fallback

NPU 重写了核心 GEMM 算子的 lowering：

```python
# torch_npu/_inductor/kernel/mm.py
def tuned_mm(mat1, mat2, *, layout=None):
    # ...
    if use_catlass_template("mm", layout, m, n, k):
        CATLASS1xGemmTemplate.add_catlass_gemm_choices(...)
    if use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(...)
    if use_cpp_gemm_template(layout, mat1, mat2):
        CppGemmTemplate.add_choices(...)
```

社区 CUDA 路径优先使用 Triton GEMM template + CUTLASS。NPU 由于硬件是达芬奇架构（Cube Core + Vector Core），**Triton 无法直接生成高效 NPU kernel**，因此引入了：
- **CATLASS**（华为适配的 CUTLASS-like template）
- **CK Gemm Template**（Composable Kernel）
- **CppGemmTemplate**

这导致 autotune 的搜索空间和 benchmark 逻辑都与社区不同。

### 2.5 Monkey Patch 完整清单（v2.7.1）

以下是 `torch_npu/_inductor/__init__.py` 中所有活跃 patch 的完整清单：

| Patch 函数 | 目标上游函数 / 模块 | NPU 覆盖原因 |
|---|---|---|
| `patch_codegen_with_cpp_wrapper()` | `GraphLowering.codegen_with_cpp_wrapper` | 添加 `"npu"` 到支持 two-pass codegen 的设备列表 |
| `patch_get_cpp_torch_device_options()` | `cpp_builder.get_cpp_torch_device_options` | 注入 CANN 头文件/库路径 |
| `patch_device_to_aten()` | `DEVICE_TO_ATEN` | 映射 `"npu"` → `at::kPrivateUse1` |
| `patch_constant_fold_uniform_value()` | `fx_passes.joint_graph.constant_fold_uniform_value` | AOTInductor 死代码消除修复 |
| `patch_fallback_kernel_codegen()` | `ir.FallbackKernel.codegen` | NPU C++ wrapper 使用 proxy executor 而非 AOTI shim |
| `patch_aot_code_compiler_compile()` | `codecache.AotCodeCompiler.compile` | 生成 NPU extern kernel JSON，绕过 v2.6.0 bug |
| `patch_algorithm_selector()` | `select_algorithm.AlgorithmSelectorCache` | 替换为 NPU-aware autotune（CATLASS + profiling） |
| `patch_tuning_process()` | `autotune_process.CUDA_VISIBLE_DEVICES` | 改为 `ASCEND_RT_VISIBLE_DEVICES` |
| `patch_tuning_process_pool()` | `autotune_process.TuningProcessPool.get_device_list` | 使用 `torch.npu.device_count()` |
| `patch_async_compile()` | `async_compile.AsyncCompile.catlass` | 添加 CATLASS kernel 编译路径 |
| `patch_scheduler()` | `scheduler.Scheduler.are_long_distant_nodes` | Ascend950 proximity 阈值从 64 降为 20 |
| `patch_gen_common_triton_ext_imports()` | `codegen.triton.gen_common_triton_imports` | 注入 NPU Triton helper |
| `patch_simplify()` | `sizevars.SizeVarAllocator.simplify` | 使用 `sympy.expand` 避免 NPU 索引表达式排序问题 |
| `patch_num_splits()` | `ir.Reduction.num_splits` | 自定义 reduction split，禁用 scan/argmax/argmin split |
| `patch_loop_body()` | `loop_body.LoopBody.__call__` | 添加 indirect memory template（`index_select`、`gather_template` 等） |
| `patch_indexing()` | `loop_body.CaptureIndexing` | 注册 NPU 自定义索引 op |
| `patch_triton_scheduling()` | `codegen.triton.TritonScheduling` | 替换为 `NPUTritonScheduling`，支持 SIMD/SIMT/mixed kernel |
| `patch_create_device_properties()` | `runtime.hints.DeviceProperties.create` | 使用 `vector_core_num` 替代 `multi_processor_count` |
| `patch_load_cached_autotuning()` | `runtime.autotune_cache._load_cached_autotuning` | 移除 `time_taken_ms`，跳过冗余 autotune |
| `patch_triton_heuristics_cached_autotune()` | `runtime.triton_heuristics.CachingAutotuner` | 替换为 `NPUCachingAutotuner`，使用 NPU profiler benchmark |
| `pre_grad_custom_pass_fuc()` / `post_grad_custom_pass_fuc()` | FX graph passes | 注册 NPU attention fusion 等自定义 pass |
| `patch_pattern_mm_plus_mm()` | `fx_passes.post_grad` | `mm + mm` pattern fusion |
| `patch_cache_base_get_system()` | `codecache.CacheBase.get_system` | cache key 使用 CANN 版本和 NPU 设备名 |
| `patch_count_bytes()` | `graph.GraphLowering.count_bytes` | 自定义内存统计，捕获 `AssertionError` |
| `patch_run_node()` | `graph.GraphLowering.run_node` | 复刻上游逻辑，添加 NPU IR origin_node 跟踪 |
| `patch_is_gpu()` | `utils.GPU_TYPES` | 追加 `'npu'` |
| `patch_has_triton()` | `utils._triton.has_triton` | 注册 NPU 为 triton 兼容设备 |
| `patch_get_first_incompatible_cudagraph_node()` | `utils.get_first_incompatible_cudagraph_node` | 维护 NPU 专属禁止算子列表 |
| `patch_get_optimization_cflags()` | `cpp_builder._get_optimization_cflags` | 添加 `-fno-finite-math-only` |
| `patch_extract_read_writes()` | `dependencies.extract_read_writes` | 强制 `normalize=False` |
| `add_additional_op()` | `ops_handler.OpsHandler` | 注册 `index_select`、`gather_template`、`indexput_template`、`scatter_template` |

---

## 三、ACLGraph 层：与 CUDAGraph 的差异

### 3.1 架构映射关系

| CUDAGraph (社区) | ACLGraph (torch_npu) |
|---|---|
| `torch._C._cuda_graphPoolHandle()` | `c10_npu::MemPool` |
| `cudaStreamBeginCapture()` | `AclmdlRICaptureBegin()` |
| `cudaStreamEndCapture()` | `AclmdlRICaptureEnd()` |
| `cudaGraphLaunch()` | ACL Graph replay |
| `torch._inductor.cudagraph_trees` | `torch_npu.npu._graph_tree` |
| `CUDAGraph` 类 | `NPUGraph` 类 (`NPUGraph.cpp`) |

ACLGraph 在 `torch_npu/utils/_graph_tree.py` 中**完全复刻了 CUDAGraph tree 的 Python 层逻辑**（`npugraphify_impl`、`align_inputs_from_check_idxs`、`static_input` 等），但底层调用的是 **CANN 的 ACL API**（`aclmdlRI*` 系列）。

### 3.2 关键差异：Operator Handler 框架

这是 ACLGraph 与 CUDAGraph 最大的不同。NPU 有大量**手工融合算子**（如 `npu_fusion_attention_v3`、`npu_fused_infer_attention_score`、`npu_multi_head_latent_attention`），这些算子在 graph capture 时有特殊行为：

```python
# torch_npu/npu/_npugraph_handlers/_fa3_graph_handler.py
class FA3ForwardHandler(_FA3TensorListOutHandler):
    @classmethod
    def prepare_capture(cls, func, args, kwargs):
        # 1. 预分配 workspace
        workspace = torch_npu._npu_fusion_attention_v3_get_max_workspace(*args, **kwargs)
        # 2. 推断 output shapes
        attention_score, softmax_max, softmax_sum = torch_npu._npu_fusion_attention_v3_infer_output(...)
        # 3. 将 .default 调用切换为 .out 变体
        kwargs["workspace"] = workspace
        kwargs["out"] = [attention_score, softmax_max, softmax_sum, ...]
        return func_out, args, kwargs
```

NPU 为 graph capture 专门实现了 **Operator Handler Registry**（`torch_npu/npu/_npugraph_handlers/`），目前内置了：
- IFA v1/v2（推理融合 Attention）
- FA v3 forward/backward
- Paged Attention / MLA

**社区 CUDAGraph 没有这种概念**。CUDA 的 graph capture 对所有 kernel 是透明的，不需要针对特定算子做 pre-alloc + `.out` 变体切换。

### 3.3 差异原因

根本原因是 **CANN 的 Graph Capture 模型与 CUDA Graph 不同**：
- CUDA Graph：在 stream 级别做 API call recording，对单个 kernel transparent
- ACL Graph（`aclmdlRI`）：在 model level 做 capture，某些融合算子需要预先知道 workspace 大小、output tensor 地址，且必须使用 `.out` 变体才能被 capture

---

## 四、当前这样适配的原因

### 4.1 硬件架构根本不同

Ascend NPU 是**达芬奇架构**，核心特点是：
- **Cube Core**：专用于矩阵运算（类似 Tensor Core）
- **Vector Core**：用于 element-wise 运算
- **统一的 L0/L1 Buffer 层次**，而非 CUDA 的显存 + Shared Memory 模型

这导致：
1. **Triton 不直接适用**：Triton 的 block/thread/threadblock 编程模型基于 CUDA SIMT，NPU 需要完全不同的 tiling 策略（`tile_generator.py`、`split_tiling.py`）
2. **内存布局约束不同**：NPU 对 strides、alignment 有特定要求，因此需要 `patch_shape_handling`、`disable_comprehensive_padding`
3. **算子融合粒度不同**：NPU 的手工融合算子（如 `npu_fusion_attention_v3`）在性能上远超 Inductor 自动融合的结果

### 4.2 CANN/ACL 软件栈的封闭性

华为 CANN 是**闭源软件栈**（除部分头文件外），torch_npu 只能通过 ACL API 与底层交互。这与 ROCm（开源，可直接修改）或 XPU（Intel 积极参与社区标准化）不同。因此：
- ACL Graph API 无法标准化为 CUDA Graph 的子集
- 很多 kernel 只能以 pre-built binary 形式存在（通过 `op-plugin` 子模块）
- 没有社区驱动的 Triton backend for NPU

### 4.3 PyTorch Inductor 的设备抽象历史上不够成熟

从 knowledge base 的演进记录可以清晰看到这一点：

| 版本 | 活跃 Patch 数 | 关键变化 |
|---|---|---|
| v2.7.1 | 35+ | 大量 monkey patches |
| v2.9.0 | ~10 | 引入 `patch_torch_for_aoti()` 条件化管理，20+ patches 被删除（"已合并"） |
| master | ~8 | 引入 `torch_npu._compat.inductor` 兼容层 |

这说明 torch_npu 团队**主动推动了很多 patch 回馈社区**，但 v2.7.1 时期的 Inductor 内部确实存在大量 CUDA-hardcoded 逻辑（`GPU_TYPES = ['cuda']`、`is_gpu()`、`has_triton()` 等），不得不 patch。

### 4.4 torchair 提供了另一条更优路径

`torch_npu/dynamo/__init__.py` 中，Dynamo backend 可以走 `torchair`（华为的 Graph Compiler）：

```python
def _get_global_npu_backend(name, config=None):
    import torchair
    _global_npu_backend[name] = torchair.get_npu_backend(compiler_config=config)
```

torchair 本质上是一个**独立于 Inductor 的 AOT 图编译器**，它直接将 FX Graph 编译为 CANN 的 graph 表示。对于推理场景，这条路径的性能往往优于 Inductor + ACLGraph。因此 torch_npu 团队没有把所有精力都投入在"让 Inductor 完美支持 NPU"上，而是**维护多条编译路径，按场景选择**。

---

## 五、这么做的收益

### 5.1 快速跟进 PyTorch 版本

Monkey-patching 虽然 dirty，但**版本迁移成本最低**。当 PyTorch 从 2.5 → 2.7 → 2.9 升级时，torch_npu 只需要调整被 patch 的函数签名，而不需要重写整个 backend。这与 ROCm（需要维护一个完整的 fork）形成对比。

### 5.2 灵活的编译路径选择

三条 Inductor 路径 + torchair 路径，让 torch_npu 可以针对不同场景选择最优策略：
- **训练 eager**：ATen ACLNN path
- **训练 compile**：Inductor default path（Triton-like）
- **推理 compile**：torchair path（graph compiler，性能最优）
- **推理 low-latency**：ACLGraph path（类似 CUDA Graph）

### 5.3 充分利用 NPU 硬件特性

CATLASS、手工融合算子、ACL Graph 的 operator handler 等，都是**为了榨取 NPU 硬件性能**而做的特化。如果强行套用社区的 Triton/CUDAGraph 抽象，性能会差很多。

---

## 六、未来如何缩小与社区的差距

### 6.1 短期（v2.9.0 → master 已在做的）

从 `tools/compile_pr_radar/knowledge/` 可以看到 torch_npu 团队已经在执行的策略：

1. **将 monkey patches 转化为标准设备接口**
   - `NPUDeviceOpOverrides(DeviceOpOverrides)` — 使用 `register_device_op_overrides('npu', ...)` 替代 patch
   - `patch_is_gpu` / `patch_has_triton` → 推动社区将 `GPU_TYPES` 改为可扩展集合

2. **引入 `_compat` 兼容层**
   - `torch_npu._compat.inductor` 隔离上游版本差异，避免直接 patch 内部类

3. **条件化 patch 管理**
   - `patch_torch_for_aoti()` + `DISABLE_AOTI_PATCH=1`，将非核心 patch 与核心路径解耦

### 6.2 中期（建议方向）

#### A. 推动 PyTorch 社区完善设备抽象

当前仍需要 monkey patch 的核心原因，是 Inductor 内部存在**隐式的 CUDA 假设**。建议推动以下社区修改：

| 上游修改点 | 当前问题 | 建议方案 |
|---|---|---|
| `torch._inductor.utils.is_gpu()` | hardcode CUDA/XPU | 改为检查 backend feature |
| `torch._inductor.codegen.wrapper` | `write_triton_header_once` 假设 triton | 抽象为 `write_kernel_header` |
| `torch._inductor.cudagraph_utils` | 模块名和函数名含 `cuda` | 重命名为 `graph_utils` / `static_graph_utils` |
| `torch._inductor.runtime.triton_heuristics` | 假设所有 GPU 用 Triton | 引入 `BackendKernelHeuristics` 抽象 |

#### B. 减少 lowering_fallback_list 规模

当前约 963 个 op fallback 意味着 **Inductor 对 NPU 的价值被严重削弱**（大部分 op 走 eager ACLNN，无法融合）。建议：

1. **优先实现 bitwise / reduction 的 lowering**：这些 op 在 fusion 中非常常见（如 mask 计算后的 `__and__` + `addmm`）
2. **推动社区标准化 distributed op 的 Inductor 路径**：当前所有 `c10d_functional` 算子都 fallback，导致分布式 compile 几乎不可用
3. **利用 decomposition 减少工作量**：很多 fallback op 可以通过 decomposition 拆分为已支持的 primitive

#### C. 统一 ascend_npu_ir 与 default 路径

当前 MLIR 路径（`ascend_npu_ir/`）和 default 路径是**完全独立的代码库**，都有自己的 scheduler、wrapper、codecache。建议：

1. 将 MLIR compiler 抽象为 `torch._inductor.codegen.compiler_backend` 的标准插件
2. 复用 default 路径的 wrapper 和 runtime，仅替换 kernel compiler 部分

### 6.3 长期（架构演进）

#### A. 让 torchair 成为社区标准 Graph Compiler 接口

torchair 的路径设计（Dynamo → FX Graph → Graph Compiler → Binary）实际上是 **PyTorch 社区想要的标准化方向**（参考 `torch.compile(backend="openxla")`、`torch.compile(backend="tvm")`）。

建议 torch_npu 团队：
1. 将 torchair backend 注册为标准的 `torch.compile(backend="npugraph_ex")`
2. 推动社区定义 `torch._dynamo.backends.graph_compiler` 标准接口
3. 让 torchair 成为 Inductor 的替代而非补充，从而减少对 Inductor monkey-patching 的依赖

#### B. 标准化 ACLGraph 为 PyTorch 的 Device-Agnostic Graph API

当前 ACLGraph 通过 patch `torch._inductor.compile_fx.cudagraphify` 注入，这是非常 fragile 的。建议：

1. 推动社区将 `cudagraphs` 改名为 `static_graphs` 或 `device_graphs`
2. 将 `torch._dynamo.backends.cudagraphs` 中的 `check_multiple_devices_or_any_cpu_nodes`、`get_first_incompatible_cudagraph_node` 等设备无关逻辑提取到公共模块
3. 定义 `DeviceGraphCapture` 抽象接口，让 NPU/CUDA/XPU 各自实现

#### C. 投资 NPU Triton Backend

如果能像 Intel（XPU Triton）或 AMD（Triton ROCm backend）一样，**为 NPU 实现一个标准的 Triton backend**，那么：
- 大量 codegen patches（`patch_triton_scheduling`、`patch_gen_common_triton_ext_imports` 等）可以删除
- CATLASS/CK 的维护成本可以降低
- 社区贡献者可以为 NPU 写 Triton kernel

这是缩小差距的**终极方案**，但也是投入最大的方案，需要华为与 OpenAI Triton 团队合作。

---

## 七、版本演进趋势

从 `tools/compile_pr_radar/knowledge/` 中的版本对比可以清晰看到 torch_npu 的演进方向：

| 维度 | v2.7.1 | v2.9.0 | master |
|---|---|---|---|
| 活跃 Patch 数 | 35+ | ~10 | ~8 |
| Scheduling 类 | `NPUCombinedScheduling` | `NPUTritonScheduling` | `NPUTritonScheduling` |
| 条件化 patch | 无 | `patch_torch_for_aoti()` | `patch_torch_for_aoti()` |
| 兼容层 | 无 | 无 | `torch_npu._compat.inductor` |
| `DeviceOpOverrides` | 无 | `NPUDeviceOpOverrides` | `NPUDeviceOpOverrides` |
| 已合并上游的 patch | 少数 | 20+ | 几乎全部 |

**趋势判断**：torch_npu 团队正在积极地从 "monkey-patch 驱动" 向 "标准接口驱动" 转型。预计在未来 1-2 个版本中，核心 patch 数可以降至 5 个以内，但 lowering 覆盖率和 Triton backend 的缺失是更深层的结构性问题，需要更长时间的社区协作。

---

## 八、总结

| 维度 | 当前状态 | 与社区差距 | 缩小差距优先级 |
|---|---|---|---|
| **Inductor 接入方式** | `register_backend_for_device` + 35→10→8 patches | 中等（演进中） | 高 |
| **Lowering 覆盖率** | ~994 个 op fallback | **极大** | **最高** |
| **Codegen 路径** | 3 条独立路径（default/MLIR/DVM） | 大 | 中 |
| **GEMM kernel** | CATLASS + CK + ATen | 中等 | 中 |
| **Graph Capture** | ACLGraph，有 Operator Handler | 中等（功能等价，实现 diverge） | 高 |
| **Dynamo Backend** | torchair 作为独立 graph compiler | 小（设计先进） | 低 |

**最核心的问题**：`lowering_fallback_list.py` 中约 963 个 op 的 fallback 是 torch_npu compile 路径的**最大短板**。即使所有 monkey patches 都消失，如果大部分 op 无法进入 Inductor 的 fusion 管道，`torch.compile()` 对 NPU 的性能收益就会非常有限。建议将 **lowering 覆盖率提升** 作为首要攻关目标。

---

## 九、GPU vs NPU：Dynamic Shape 难易度对比

> Dynamic shape 在 GPU 和 NPU 上的难度存在本质差异：GPU 是软件/编译层问题，NPU 是硬件架构层问题。

### 9.1 GPU（CUDA + Triton）：Dynamic Shape 是"缓存命中率"问题

GPU SIMT 执行模型天然参数化——Triton kernel 接受运行时整数 `xnumel`，通过 mask 处理边界：

```triton
@triton.jit
def kernel(in_ptr, out_ptr, xnumel, XBLOCK: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    mask = xindex < xnumel      # 边界用 mask，无需 shape 固定
    ...
```

PyTorch 的 SymInt + ShapeEnv 方案与 GPU 硬件特性完美匹配：
- Dynamo 符号化变化的维度为 `s0`
- Inductor wrapper 生成 `xnumel = s0 * 128`（运行时求值）
- 同一个编译产物处理所有满足 guard 的 shape

主要代价仅有两处：① CUDA Graph 不兼容（需退出静态图模式）；② autotune 候选集被编译期 hint 截断（见 [[inductor_codegen_dynamic_shape_analysis]] §9）。

### 9.2 NPU（Ascend DSL）：Dynamic Shape 是"硬件架构"问题

NPU 的 dynamic shape 困难来自三层结构性约束：

**困难 1：硬件 tiling 的刚性对齐要求**

Ascend Cube Core 要求矩阵运算输入满足特定对齐（如 16×16 tile）。shape 不满足时必须 padding，而 padding 规则与 CUDA 不同且复杂：

```
GPU:  任意 shape + mask → 边界 threads idle，性能轻微下降
NPU:  shape 不对齐 → 必须 padding → 额外 reshape/pad kernel → 破坏 fusion 链
```

NPU 需要 disable 社区的 `comprehensive_padding`（通过 `patch_shape_handling`），因为 NPU 有自己独立的 padding 规则。

**困难 2：ACLGraph 需要预知 shape**

CUDA Graph 在 stream 级别透明录制 API call，对 shape 无感知。ACL Graph（CANN 的 `aclmdlRI`）在模型级别做 capture，某些融合 kernel 必须在 capture 时预知 shape：

```python
# FA3 fusion attention 的 handler（NPU 专有）
def prepare_capture(cls, func, args, kwargs):
    workspace = _npu_fusion_attention_v3_get_max_workspace(...)  # 必须预知 shape
    attention_score = _npu_fusion_attention_v3_infer_output(...)  # 推断 output shape
    kwargs["out"] = [attention_score, ...]   # 必须用 .out 变体
```

动态 shape 下 workspace 必须按最大可能 shape 预分配，且 captured graph 与具体 shape 绑定，不同 shape 无法复用同一 graph。

**困难 3：大量算子 fallback 打断 fusion 链**

GPU 几乎所有标准 aten op 都有 Triton lowering，dynamic shape 可端到端在 Inductor 管道内处理。NPU 有约 963 个 op fallback 到 ACLNN，每个 fallback 点是一个硬边界，ACLNN 有自己的 shape 约束且**完全绕过 SymInt 体系**，无法参与 ShapeEnv 的符号推导。

### 9.3 本质差异汇总

| 维度 | GPU (CUDA+Triton) | NPU (Ascend DSL) |
|------|-------------------|------------------|
| 执行模型友好度 | 高（SIMT + mask 边界） | 低（Cube Core 需对齐 tiling） |
| Graph Capture | 不兼容但可 fallback | 需预知 shape，结构性阻碍 |
| Kernel 参数化 | Triton 天然支持（runtime xnumel） | CATLASS/CK 需 shape-specific tuning |
| Fallback 算子 | 极少（大多数 op 有 lowering） | 约 963 op fallback，独立 shape 约束 |
| 工程成熟度 | SymInt/ShapeEnv 成熟 | 多路径各有妥协，仍在演进 |
| 核心问题定性 | 软件/编译层（可工程化解决） | 硬件架构层（需跨层协同解决） |

### 9.4 实践建议（NPU + Dynamic Shape）

1. **静态 shape 优先**：对 batch_size、seq_len 做 **bucketing**（预定义几组固定 shape），完全绕开 dynamic shape 复杂性
2. **torchair 路径**：推理场景用 `torchair.get_npu_backend()`，为每个 shape profile 编译一次，是 NPU 上目前最成熟的替代方案
3. **避免 dynamic=True + ACLGraph 组合**：两者是正交约束，同时启用几乎必然出问题

---

## Related Pages

- [[npu_vs_upstream_fusion_passes]] —— torch_npu vs 上游融合 Pass 全流程对照（本页 §2 差异的 pass 级逐条展开；订正本页 §2.5 `patch_pattern_mm_plus_mm` 与 §2.3 fallback 计数旧口径）
- [[21_torch_npu_upstream_adaptation_analysis]] —— 跨 eager/compile/graph/distributed 的 upstream 对照与补丁债分类，并订正“社区统一 Triton”旧口径
- [[npu_inductor_splittiling_backend_analysis]] — Triton/Inductor default 路径深度分析（本文的三条路径之一）
- [[NPU_MLIR_Backend_Technical_Analysis]] — MLIR 路径深度分析（本文的三条路径之一；含六阶段适配全景）
- [[aclgraph_deep_analysis]] — ACLGraph 路径深度分析（本文的三条路径之一）
- [[NPU_Inductor_Backend_Analysis]] — NPU Inductor 后端集成架构（已有页面）
- [[NPU_Inductor_Backend_Analysis]] — NPU 后端内部机制（已有页面）
- [[npu_compile]] — NPU 编译工作流与配置（已有页面）
- [[aclgraph]] — ACL Graph 基础集成（已有页面）
- [[20_symbolic_shapes_guards_and_graph_reuse_analysis]] — PyTorch Dynamic Shape 全链路（GPU 侧参考）
- [[inductor_codegen_dynamic_shape_analysis]] — Inductor codegen 的 XBLOCK 选择机制
