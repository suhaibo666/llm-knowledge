# NPU MLIR 编译流水线: 六阶段适配全景分析

> 对比 GPU Triton 路径，逐阶段分析 NPU MLIR 路径"改了什么、为什么改、怎么改的"
> 最后更新: 2026-05-09

---

## 概述

NPU (华为昇腾) 上不存在 Triton。唯一的编译路径是 MLIR → 毕昇编译器。这意味着 `torch.compile` 的 Inductor 后端必须从"生成 Triton 代码"切换为"重建 FX Graph → MLIR 导入 → 毕昇编译"。本文逐阶段对比两条路径，重点分析每个适配"为什么在这一层"。

```
═══════════════════════════════════════════════════════════════════
GPU (Triton 路径):
  Dynamo → AOT → Decomp → Lowering → Scheduler → Triton codegen
    → Triton Python 源码 → Triton JIT 编译器 → PTX → GPU binary

NPU (MLIR 路径):
  Dynamo → AOT → Decomp → Lowering → Scheduler → MLIR codegen
    → 重建 FX Graph → torch-mlir FxImporter → MLIR IR
    → 毕昇编译器(bishengir-compile) → NPU kernel (.o/.so)
═══════════════════════════════════════════════════════════════════
```

**架构两难的根源**: MLIR 的入口需要 FX Graph（通过 torch-mlir 的 `FxImporter.import_stateless_graph()`），但 Inductor 的 lowering 会把 FX Graph 丢弃——ATen op 变成 Inductor IR 节点后，原来的 op 语义不复存在。NPU 既要复用 Inductor 的融合优化能力，又必须在代码生成阶段交出 FX Graph 给 MLIR 编译器。

解决方案是 **TracedGraph 机制**：在 lowering 的每个函数中"偷偷"记录一份 FX Graph 的副本，融合后拼接还原。

---

## 阶段 1: Dynamo — 无改动

NPU 和 GPU 共用同一套 Dynamo（PEP 523 帧评估 → 符号执行 → FX Graph + Guards）。

**没有 NPU 特有的改动。**

---

## 阶段 2: AOT Autograd — 注入 FX 预处理

### 改了什么

Monkey-patch `AotAutograd.__call__`，在 fw_compiler/bw_compiler/inference_compiler 调用前注入 `npu_optimize_fx_graph()`。

```python
# npu_inductor_plugin.py:202-222
AotAutograd.__call__ = wrap_aot_autograd(AotAutograd.__call__)

# wrap_aot_autograd 内部:
# 1. 调用 _register_npu_inductor_fallbacks()  注册 NPU fallback 算子
# 2. 包装 fw/bw/inference compiler: 先调 npu_optimize_fx_graph(gm) 再调原 compiler
```

### 为什么在这一层

FX Graph 刚从 AOT 分区产出，尚未进入 Inductor lowering。此时做 NPU 硬件偏好的图优化（类型转换、算子替换），不影响 Inductor 的后续流程。

| NPU 预处理 | 作用 | 为什么 |
|-----------|------|--------|
| `iota int64 → int32` | 将索引生成的类型从 int64 转为 int32 | Ascend NPU 的标量计算单元偏好 int32 |
| `empty + copy → npu_dtype_cast` | 将 "分配空tensor + 拷贝" 合并为 NPU 原生 dtype cast | 减少内存分配 + NPU 有专用的 dtype cast 硬件指令 |

---

## 阶段 3: Decomposition — NPU 选择性禁用

### 改了什么

`npu_decomp.py` 禁用部分 ATen 分解规则。

### 为什么在这一层

GPU (Triton) 的策略是"尽可能分解为 pointwise → 最大化融合"。NPU 的策略是"保留部分复合算子 → MLIR 编译器直接优化"。

| 策略 | GPU (Triton) | NPU (MLIR) | 原因 |
|------|-------------|-----------|------|
| `aten.native_layer_norm` | 分解为 mean/var/sub/mul | **不分解**，在 lowering 中手动实现 | 保持对中间 op 的控制（确保都在白名单内），避免分解后的中间节点被 fallback |
| `aten.addmm` | 分解为 mm + add | **不分解**，交给毕昇编译器处理 | MLIR 编译器可以直接识别和优化 addmm 模式 |

---

## 阶段 4: Lowering — 最重的适配（也是最大的技术债务）

这里有**三个独立但耦合的适配**。必须理解它们之间的关系。

### 适配 A: 完整复制 lowering.py + 注入 TracedGraph

**问题**: 原生 `lowering.py` 的每个函数在创建 IR 节点后，FX Graph 信息就不复存在。但 NPU 的 codegen 阶段需要 FX Graph 作为 MLIR 导入的输入。

**方案**: 复制全部 `lowering.py`（~7440 行），在每个函数中额外记录 FX Graph：

```python
# 原生 (3 行)
@register_lowering(aten.view)
def view(x, sizes):
    return TensorBox(View.create(x.data, sizes))

# NPU (~10 行)
def view(x, sizes):
    input_graphs = fetch_graphs([x.data, sizes])        # 提取输入 traced_graph
    node_name = f'view_{next(node_id)}'
    new_graph = merge_traced_graphs(input_graphs,       # 合并为新的 FX subgraph
                                     aten.reshape, node_name)
    return TensorBox(View.create(x.data, sizes,
              traced_graph=new_graph,   # ← 夹带 FX 信息到 IR 节点
              node_name=node_name))
```

`TracedGraph` 类（`inductor_patch/lowering.py:187-203`）：

```python
class TracedGraph:
    def __init__(self):
        self.graph = torch.fx.Graph()                    # FX Graph 实例
        self.last_node: Optional[torch.fx.Node] = None   # 最后一个 FX 节点
        self.sym_nodes: Dict[str, torch.fx.Node] = {}    # 符号变量节点
```

### 适配 B: Monkey-patch IR 类注入 traced_graph 属性

Inductor 的 IR 类是 frozen dataclass，不能正常添加属性。NPU 通过 `_post_init_setattr` 绕过限制：

```python
# inductor_patch/ir.py
@classmethod
def _patch_loops_create(cls, *args, **kwargs):
    traced_graph = kwargs.pop("traced_graph", None)
    node_name = kwargs.pop("node_name", None)
    r = cls(*args, **kwargs)
    r._post_init_setattr("traced_graph", traced_graph)   # 注入
    r._post_init_setattr("node_name", node_name)         # 注入
    return ir.TensorBox.create(r)

ir.Loops.create   = _patch_loops_create
ir.Reduction.create = _patch_reduction_create
ir.Pointwise.constant_to_device = _patch_pointwise_constant_to_device
```

### 适配 C: 算子白名单/黑名单分流

不是所有算子都能走 MLIR 编译。NPU 通过双层配置控制：

```
旧芯片 (910B1): 白名单模式
  GENERATE_LIST (~94 个 op) → MLIR codegen
  其余 → FallbackKernel → AclNN 算子库

新芯片 (910_9391): 黑名单模式
  FALLBACK_LIST (~27 个 op) → FallbackKernel → AclNN
  其余 → MLIR codegen
```

| 白名单核心 op | 黑名单核心 op |
|-------------|-------------|
| add/sub/mul/div/exp/log/sqrt/relu/sigmoid/tanh | mm/bmm/addmm |
| sum/mean/amax/min/max/argmax | convolution/convolution_backward |
| cat/split/reshape/permute/expand/slice | max_pool2d/adaptive_avg_pool2d |
| where/clamp/full/arange | embedding/random/sort/topk |
| native_layer_norm/flex_attention | linalg_*/triangular_solve |

**策略哲学**: "保守 codegen，激进 fallback"——不确定能正确编译的 op 一律走 AclNN。正确性第一。

### 为什么三个适配在 lowering 层耦合

TracedGraph（适配 A）需要 IR 节点能携带额外属性（适配 B），而 NPU 专有 lowering 实现中的一部分 op（如 `npu_dtype_cast`）必须在白名单中（适配 C）才能走 codegen 而非回退。三层适配相互依赖，无法独立演进。

---

## 阶段 5: Scheduler — 融合策略放宽

### 改了什么

覆盖 5 个 Scheduler 方法（仅在 `enable_graph_trace=True` 时生效）：

```python
# npu_inductor_plugin.py:394-399
Scheduler.can_fuse_vertical    = npu_can_fuse_vertical    # 放宽融合条件
Scheduler._prune_redundant_deps = _npu_prune_redundant_deps # 自定义依赖剪枝
Scheduler.compute_ancestors    = npu_compute_ancestors    # 自定义祖先计算
Scheduler._get_unmet_dep_nodes = _npu_get_unmet_dep_nodes
Scheduler._codegen             = wrap_scheduler_codegen   # 重算 last_usage
```

`wrap_scheduler_codegen` 在融合前重算 `last_usage`，使用 traced_graph 的 placeholder 而非 Inductor buffer 名——因为 MLIR 编译需要正确的输入/输出集合。

`npu_can_fuse_vertical` 简化了原生实现中的部分限制条件（`WhyNoFuse` 的检查项更少）。

### 为什么在这一层

NPU 融合作了"去重"：Inductor Scheduler 做基础融合，剩下的交给毕昇编译器。原因有二：

1. **毕昇编译器有自己的 hfusion 水平融合**——如果 Scheduler 融合过度，可能破坏 MLIR 编译器可识别的算子模式
2. **NPU 的融合收益模型与 GPU 不同**——Ascend 的 Vector/Cube 单元的并行约束不同，Inductor 的 `score_fusion_memory` 成本模型是为 GPU HBM 带宽优化的

---

## 阶段 6: Codegen — 完全替换

这是 NPU 与 GPU 差异最大的阶段。

### 改了什么

GPU 的 `TritonScheduling` 被完全替换为 `NpuMlirScheduling`：

```
GPU (TritonScheduling):
  FusedSchedulerNode → TritonKernel.codegen_kernel()
    → 遍历 node.schedule, inner_fn → tl.load/tl.store
    → Triton Python 源码 → Triton JIT → PTX

NPU (NpuMlirScheduling):
  SchedulerNode (含 traced_graph) 
    → create_fx_from_snodes_by_traced_graph()
      → 合并各节点 traced_graph → 重建 FX Graph
      → make_fx() 标准化 → view_to_reshape()
    → FxImporter.import_stateless_graph()
      → FX Graph → MLIR Torch dialect IR
    → bishengir-compile
      → MLIR IR → NPU kernel (.so)
```

**重建 FX Graph 的核心步骤**（`codegen/mlir.py:225-270`）：

1. 遍历 SchedulerNode 列表，收集每个节点的 `traced_graph`
2. 如果多节点融合，调用 `merge_fx_graphs()` 合并子图
3. 提取输入/输出，处理 mutation/alias
4. 创建 `GraphModule`，调用 `make_fx()` 获得标准化版本
5. `view_to_reshape(gm)` —— 将 view 替换为 reshape（MLIR 兼容性要求）

**NPU 特有的 torch-mlir 补丁**：

| Patch | 文件 | 作用 |
|-------|------|------|
| `FxImporter.import_stateless_graph` | `torch_mlir_patch.py:51-124` | 注入符号形状范围约束 `[128, 1024]` 到 symbolic guards |
| `sympy_expr_to_semi_affine_expr` | `torch_mlir_patch.py:127-193` | 扩展对 `sympy.Pow` 和 `FloorDiv` 的 MLIR 仿射表达式转换支持 |

### 为什么在这一层

这些改动是 NPU 硬件特有的编译需求。Inductor 的 codegen 是"最后一道门"——在这之前一切改动都是与 GPU 共享的，在这之后就是 NPU 专属的编译器栈。Torch-mlir 补丁的存在是因为上游 torch-mlir 不支持 PyTorch 2.x 的动态形状（SymInt）和部分 sympy 表达式。

---

## 三层 Pass 架构

NPU MLIR 路径的优化 Pass 分布在三个层级，每层有不同的设计动机：

```
═══════════════════════════════════════════════════════════════════
第一层: FX Graph 预处理 (Python, Inductor lowering 之前)
  │
  ├── npu_optimize_fx_graph()          [utils.py:338-362]
  │   ├── replace_iota_int64_to_int32    NPU 硬件 int32 偏好
  │   └── empty + copy → dtype_cast     NPU 自定义算子
  │
  │  定位理由: NPU 硬件的类型/算子偏好必须在 lowering 前转换。
  │  一旦 ATen op 变成 Inductor IR，就无法做图级别的算子替换。
  │
═══════════════════════════════════════════════════════════════════
第二层: Inductor Scheduler 融合 (Python, IR 融合阶段)
  │
  ├── npu_can_fuse_vertical()          放宽垂直融合条件
  ├── _npu_prune_redundant_deps()      剪除冗余依赖
  ├── npu_compute_ancestors()          自定义祖先计算
  ├── _npu_get_unmet_dep_nodes()       自定义未满足依赖节点
  └── wrap_scheduler_codegen()         重算 last_usage (基于 traced_graph)
  │
  │  定位理由: ①放宽融合——毕昇编译器有 hfusion，NPU 只需基础融合；
  │  ②重算 last_usage——MLIR 编译需要正确的输入/输出参数集，
  │  traced_graph 的 placeholder 与 Inductor buffer 名不同。
  │
═══════════════════════════════════════════════════════════════════
第三层: 毕昇编译器优化 (C++, bishengir-compile 内部)
  │
  ├── hfusion (水平融合)                -enable-hfusion-compile=true (始终)
  ├── ops_reorder (算子重排)             --enable-ops-reorder (条件)
  ├── auto_multi_buffer (多缓冲流水线)    --enable-auto-multi-buffer (条件)
  ├── symbol_analysis (符号分析)          --enable-symbol-analysis (动态shape)
  └── tuning (autotune)                -enable-tuning-mode (条件)
  │
  │  定位理由: 这些优化依赖 NPU 微架构细节：
  │  • hfusion — Ascend Vector/Cube 单元的指令级并行约束
  │  • ops_reorder — 优化 NPU 流水线利用率
  │  • auto_multi_buffer — UB 大小固定 (910B1=192KB, 910_9391=256KB)
  │  • block_dim — 控制 AI Core 的 block 调度粒度 (默认48)
  │
  │  PyTorch 层不知道这些硬件细节，必须在编译器层做"最后一公里"优化。
═══════════════════════════════════════════════════════════════════
```

---

## 15 个 Monkey Patch 全景

这些 Patch 按目的分为五组。本质原因：PyTorch Inductor 没有为第三方后端预留足够的扩展点。

### 组 1: 路径控制 (2 个)

| Patch | 目的 |
|-------|------|
| `_triton.has_triton → False` | 强制禁用 Triton 路径 |
| `_TorchCompileInductorWrapper.__call__` | 恢复 compile_fx 入口（抵消 torch_npu 其他补丁影响） |

### 组 2: FX Graph 预处理 (2 个)

| Patch | 目的 |
|-------|------|
| `AotAutograd.__call__` | 注入 `npu_optimize_fx_graph` |
| `torch._dynamo.utils.run_node` | 处理 `npu_fusion_attention` 的参数类型 |

### 组 3: IR 扩展 — TracedGraph 支持 (3 个)

| Patch | 目的 |
|-------|------|
| `ir.Loops.create` | 注入 `traced_graph` + `node_name` |
| `ir.Pointwise.constant_to_device` | `traced_graph` 透传 |
| `ir.Reduction.create` | `traced_graph` 透传 + 附加 kept_idx/reduced_idx |

### 组 4: Scheduler 融合策略 (5 个)

| Patch | 目的 |
|-------|------|
| `Scheduler._codegen` | 重算 last_usage |
| `Scheduler.compute_ancestors` | 自定义祖先计算 |
| `scheduler._prune_redundant_deps` | 自定义依赖剪枝 |
| `Scheduler.can_fuse_vertical` | 放宽融合条件 |
| `Scheduler._get_unmet_dep_nodes` | 自定义未满足依赖节点获取 |

### 组 5: 算子兼容 + torch-mlir 兼容 (3 个)

| Patch | 目的 |
|-------|------|
| `F.avg_pool2d` | bf16→fp32→bf16 精度修正 |
| `FxImporter.import_stateless_graph` | 注入符号形状范围约束 |
| `sympy_expr_to_semi_affine_expr` | 扩展 sympy→MLIR 仿射表达式 |

---

## 编译模式状态机

NPU MLIR 路径有一个 GPU 没有的运行时决策树。

### 三种编译模式

```
每个 kernel 的 mode 决定:

  compile_mode = "auto_fallback" (默认)
    → MLIR 编译 → 成功? → 执行 MLIR kernel
                 → 失败? → 自动回退到 FX Graph eager 执行

  compile_mode = "default"
    → MLIR 编译 → 成功? → 执行 MLIR kernel
                 → 失败? → 抛异常 (无回退)

  compile_mode = "complete_fallback"
    → 跳过 MLIR，直接 FX Graph eager 执行
```

### 自动降级规则

即使配置了 `auto_fallback` 或 `default` 模式，以下情况强制降级为 `complete_fallback`：

```python
# codegen/mlir.py:273-392
num_call_functions = get_num_call_functions(mlir_kernel._gm)
if num_call_functions <= 1:
    mode = "complete_fallback"  # 单算子 kernel: 不值的 MLIR 编译开销

if kernel_name in anir_config.force_fallback_kernel_names:
    mode = "complete_fallback"  # 用户指定的回退 kernel
```

### GPU 不需要这个状态机

Triton JIT 编译要么成功要么报错。不存在"Triton 编译失败后自动回退到 ATen eager"的设计——Inductor 的 `FallbackKernel` 机制只作用于 lowering 阶段（标记"这个 op 我们不会编译"），而非运行时。

---

## Fallback 双通道

NPU 有两套并行的 fallback 机制：

```
═══════════════════════════════════════════════════════════════
Lowering 层 fallback (op 级别):
  某个 ATen op 不在 GENERATE_LIST → make_fallback(op)
    → lowerings[op] = fallback_handler(op)
    → 创建 ir.FallbackKernel → 调用 AclNN 算子库
    → 发生在编译期, 不可逆

Codegen 层 fallback (kernel 级别):
  某个 kernel 的 MLIR 编译失败 → auto_fallback 模式自动回退
    → 保存 FX Graph 到磁盘 → eager 执行
    → 发生在运行时, 自动恢复
═══════════════════════════════════════════════════════════════
```

| 维度 | Lowering fallback | Codegen fallback |
|------|------------------|-----------------|
| 触发条件 | op 不在白名单 | MLIR 编译失败 |
| 时机 | 编译期 | 首次执行时 |
| 目标 | AclNN 算子库 | FX Graph eager |
| 可恢复 | 不可逆 | 自动回退 |

---

## Autotune: 60 种编译配置

```python
# mlir_compiler.py:335-343
def get_autotune_config(self):
    compile_args = []
    for ops_reorder in [True, False]:           # 2 种
        for auto_db in [True, False]:           # 2 种
            for tiling_size in range(-10, 20, 2): # 15 种
                compile_args.append((tiling_size, ops_reorder, auto_db))
    return compile_args  # 15 × 2 × 2 = 60 种

# autotune=False: 仅默认参数 (None, True, True), 单次编译
```

| 参数 | 含义 | 影响 |
|------|------|------|
| `tiling_size` | tiling 大小参数 (-10~18) | 控制计算切分粒度 |
| `ops_reorder` | 算子重排序开关 | 调整指令顺序优化流水线 |
| `auto_db` | 自动多缓冲开关 | 插入双缓冲隐藏内存延迟 |

### GPU 的 autotune 区别

Triton 的 autotune 直接在 GPU 上 benchmark 实际 kernel 执行时间。NPU 的 autotune 是**编译时搜索**——尝试不同的编译器参数组合，编译出不同版本的 kernel，在首次执行时 benchmark 择优。

---

## 在线精度对比

NPU 特有的精度验证机制（环境变量 `ANIR_ONLINE_ACC_COMP=1`）：

```
MLIR kernel 执行
  ↓
同时对相同输入执行 FX Graph reference
  ↓
逐输出对比: torch.isclose(actual, expected, rtol, atol)
  ↓
精度不通过? → 用 FX 结果修正输出 + dump mismatch 报告
```

| dtype | rtol | atol |
|-------|------|------|
| float32 | 1.3e-6 | 1e-5 |
| float16 | 1e-3 | 1e-5 |
| bfloat16 | 1.6e-2 | 1e-5 |

---

## Related Pages

- [[NPU_MLIR_Backend_Technical_Analysis]] — TracedGraph 机制、融合规则、编译模式、Monkey Patch 详细代码
- [[npu_lowering_guide]] — NPU lowering op 分流策略、NPU 专有 IR 节点、配置体系
- [[npu_compile]] — 编译工作流、Autotune 细节、精度校验
- [[NPU_Inductor_Backend_Analysis]] — 多后端选择机制、执行流程差异
- [[NPU_Inductor_Backend_Mechanism]] — 后端混合使用 (MultiTemplateBuffer, Epilogue/Prologue Fusion)
- [[triton_vs_mlir_backend_analysis]] — Triton vs Torch-MLIR 通用对比（非 NPU 专属）
- [[mlir_core_concepts]] — MLIR Dialect/Pass/IR 注册基础概念
- [[02_engineering/01_ai_frameworks/index]]
