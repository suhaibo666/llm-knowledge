# Triton vs Torch-MLIR: 编译后端技术原理对比

> 从 Dynamo → AOT Eager → Decomposition → Lowering → Scheduler → Codegen 六个阶段，逐阶段做概念级对等映射
> 最后更新: 2026-05-11

---

## 社区活跃度 (2026-05)

`llvm/torch-mlir` 是 LLVM 孵化器下的活跃项目（截至 2026-05 观察；社区活跃，具体以上游为准）。当前维护方向：

- **TorchToLinalg lowering** 持续补全（反向传播算子、激活函数、插值）
- **FlexAttention / GQA** 支持
- **Float8** 新数据类型 (Float8E8M0FNU)
- **TorchToTosa** 路径
- **FX Importer** 符号形状修复

> [!note]
> 部分文档引用 `nod-ai/SHARK-Turbine` 仓库。其 README 声明所有代码已迁出至上���项目（FX/ONNX 导入器 → `llvm/torch-mlir`，IREE Turbine → `iree-org/iree-turbine`），该仓库现在是 CI 协调的 staging 空壳，不再活跃开发。评估 torch-mlir 生态应以 `llvm/torch-mlir` 和 `iree-org/iree-turbine` 为准。

---

## 概述

`torch.compile` 的默认后端 Inductor 将 FX Graph 编译为 Triton kernel（GPU）或 C++ kernel（CPU）。社区中还有另一条路线：**torch-mlir**（LLVM 孵化器项目），将 PyTorch 模型经 MLIR 多层 IR 递降编译到 GPU/NPU 等硬件。

本文从六阶段编译流水线的每一层出发，做 **A→B 概念对等映射**：在 Triton 路径中某个结构/行为，在 MLIR 中等价于什么。

```
══════════════════════════════════════════════════════════════════
Triton 路径 (2 层 IR):
  FX Graph → Inductor IR → Triton Python → Triton Compiler → PTX

MLIR 路径 (5+ 层 IR):
  FX Graph → Torch Dialect → Linalg → GPU Dialect → NVVM → PTX
══════════════════════════════════════════════════════════════════
```

**核心哲学差异**:

| | Triton | MLIR |
|---|--------|------|
| 编译模型 | JIT（首次执行时编译） | AOT（compile 调用时完成所有 pass） |
| IR 设计 | 单层 DSL（Inductor IR → Triton 语言） | 多层 Dialect（Torch→Linalg→GPU→NVVM→LLVM→PTX） |
| 融合策略 | 集中式 Scheduler（全局依赖图 + 10 轮贪心） | 分散式 Pass Pipeline（每个 dialect 独立 pass） |
| Lowering 性质 | 翻译（信息丢弃）：FX Graph 语义不再保留 | 递降（信息保留）：每层 dialect 完整描述程序语义 |
| Codegen 方式 | 生成 DSL 源码字符串 → 外部 JIT 编译器 | IR→IR→...→IR 递降至最终 target |

---

## 阶段 1: Dynamo（FX Graph 捕获）

**两路径完全一致。** Dynamo 是后端无关的 Python 级 JIT。

```
用户代码 → PEP 523 帧拦截
  → 符号执行 (InstructionTranslator)
  → VariableTracker 系统
  → FX Graph 构建 (OutputGraph) + Guard 生成
  → 输出: torch.fx.GraphModule
```

| Triton 概念 | MLIR 等价概念 | 对等说明 |
|------------|-------------|---------|
| `InstructionTranslator.step()` 逐条消费 Python 字节码 | MLIR 无此阶段 — torch-mlir 不接触 Python 字节码 | Dynamo 是 PyTorch 独有的前端。MLIR 路径从 FX Graph 开始 |
| `VariableTracker` — 符号跟踪（TensorVariable, ConstantVariable 等） | FX Graph 中的 `fx.Node` placeholder/get_attr 节点 | 两者都从 FX Graph 开始消费。MLIR 的 `FxImporter` 从此启动 |
| `OutputGraph.compile_subgraph()` 触发后端编译 | `FxImporter.import_stateless_graph()` | Dynamo 调用 Inductor；MLIR 路径调用 FxImporter |

> **分叉点**: Dynamo 输出的 FX Graph 是两条路径的共同起点。Triton 将其交给 `GraphLowering`，MLIR 将其交给 `FxImporter`。

---

## 阶段 2: AOT Eager（AotAutograd）

**两路径基本一致。** AotAutograd 将带 autograd 的 FX Graph 切分为前向/反向子图。

```
FX Graph → AotAutograd.apply()
  → 创建 JointGraph（前向+反向融合图）
  → partition_fn 切分 fw_graph / bw_graph
  → 分别调用 fw_compiler / bw_compiler
```

| Triton 概念 | MLIR 等价概念 | 对等说明 |
|------------|-------------|---------|
| `AotAutograd.__call__` → `compile_fx` | `AotAutograd.__call__` → `torch_mlir.compile` | 分叉点：将分区后的子图交给谁编译 |
| `partition_fn` 产出的 `fw_graph` / `bw_graph` | 同一个 GraphModule，作为 MLIR 导入输入 | 完全相同的中间产物 |
| `ShapeEnv` + `SymInt` 符号形状系统 | `FxImporter._symbolic_guards` + `sympy_expr_to_semi_affine_expr` | Triton 保留在 `SizeVarAllocator` 中；MLIR 转换为 `ValueRanges` 约束 |

---

## 阶段 3: Decomposition

**两路径执行同源分解逻辑，MLIR 路径有额外的 dialect 级分解。**

| Triton 概念 | MLIR 等价概念 | 对等说明 |
|------------|-------------|---------|
| `torch/_decomp/decompositions.py` — `aten.addmm` → 基础 op | `torch-decompose-complex-ops` pass — 同样目的，但在 Torch dialect 上执行 | Triton 在 lowering 前完成；MLIR 可在 dialect 层再次分解 |
| `make_fallback(op)` — 无法分解的 op → FallbackKernel | `torch.operator` — 保留未降级的外部调用 | 两者都需要兜底机制 |

---

## 阶段 4: Lowering — 第一条核心分叉线

### Triton: ATen → Inductor IR（破坏性翻译）

```python
aten.add.Tensor → lowerings[aten.add.Tensor](args)
  → make_pointwise()
  → Pointwise.create(
      device, dtype,
      inner_fn=lambda idx: x_loader(idx) + y_loader(idx),  # λ 闭包
      ranges=[s0, s1]
    )
```

**关键**: FX Graph 的拓扑信息被**丢弃**。`aten.add` 不存在于 IR 中。`Pointwise` 只知道"给一个 index 执行 inner_fn"，不知道"来自 `aten.add`"。

### MLIR: ATen → Torch Dialect（保留性翻译）

```mlir
%result = torch.aten.add.Tensor %x, %y, %alpha
  : !torch.vtensor<[?],f32> → !torch.vtensor<[?],f32>
```

**关键**: FX Graph 的操作语义被**精确保留**。`torch.aten.add.Tensor` op 完整记录了 ATen add 的参数结构。

### 概念等价映射

| Triton 概念 | MLIR 等价概念 | 对等原理 |
|------------|-------------|---------|
| `lowerings` 字典 (`Dict[Callable, Callable]`) | `TorchDialect` 的 `torch.aten.*` op 集合 | 都是"算子→IR 翻译"的注册表 |
| `Pointwise(inner_fn=λ ..., ranges=...)` | `linalg.generic { indexing_maps=[...], iterator_types=[...] }` 内的 region `{ ... }` | inner_fn λ ≈ linalg.generic 的计算体。两者描述"给定迭代空间一点，执行什么计算" |
| `Reduction(reduction_type="sum", ranges=...)` | `linalg.reduce { arith.addf }` | reduction_type + inner_fn ≈ linalg.reduce 的归约体 |
| `View.create(x, sizes)` — metadata-only `ReinterpretView` | `tensor.expand_shape` / `tensor.collapse_shape` | 零拷贝视图。只改 shape/stride 元数据 |
| `ExternKernel` (如 cuBLAS matmul) | `torch.aten.mm` 保留不降级，或降级到 `linalg.matmul` | 不能/不愿 codegen 的 op 的外部调用 |
| `FallbackKernel` — 回退 eager ATen | MLIR 的 `torch.operator` — "无法降级的 op 保留原样" | 兜底机制 |
| `TensorBox` (IRNode 的 MutableBox 包装) | MLIR `Value` (SSA 值) | IR 中值的句柄/引用 |

---

## 阶段 5: Scheduler — 第二条核心分叉线

### Triton: 集中式 Scheduler

```
Scheduler.__init__(ir.operations)
  → create_scheduler_node()    // 包装为 BaseSchedulerNode
  → compute_dependencies()     // 构建 ReadWrites 依赖图
  → topological_sort_schedule()
  → fuse_nodes() × 10 轮       // 贪心融合
     → get_possible_fusions()  // 按 used_buffer_names 分组
     → can_fuse(node1, node2)  // 合法性 (shared_data_score + device + 环检测)
        → can_fuse_vertical()  // node2 的 unmet_deps 被 node1 满足?
        → can_fuse_horizontal() // 共享输入?
     → score_fusion_key()      // 按内存节省排序
     → _try_fusion_pairs()     // 创建 FusedSchedulerNode
  → reorder_for_peak_memory()
  → compute_last_usage()
  → codegen()
```

### MLIR: 分散式 Pass Pipeline

融合分布在多个 dialect 的多个 pass 中：

- **Torch 层**: `torch-simplify-shape-inference`, `torch-fuse-quantized-ops`
- **Linalg 层** (最核心): `linalg-fuse-elementwise-ops` — 寻找 producer→consumer 的 `linalg.generic` 对，合并 indexing_maps + region
- **Affine 层**: `affine-loop-fusion`
- **GPU 层**: `gpu-kernel-fusion`

### 概念等价映射

| Triton 概念 | MLIR 等价概念 | 对等原理 |
|------------|-------------|---------|
| `BaseSchedulerNode` — 融合基本单元 | `linalg.generic` op — 可融合基本单元 | 都有明确的输入/输出/迭代空间 |
| `compute_dependencies()` — 通过 `ReadWrites` 分析 `MemoryDep`/`WeakDep`/`StarDep` | Use-def chain + liveness analysis | Triton 显式建依赖图；MLIR 的 SSA 天然携带 use-def |
| `FusedSchedulerNode` — 融合容器 | Fused `linalg.generic` — region 内含多个操作 | 都代表"独立的多次计算现在在一个 kernel 里" |
| `can_fuse_vertical(node1, node2)` | Producer-consumer fusion legality — 检查 producer result 的 users 是否只有 consumer；indexing 兼容性 | 相同的数学约束 |
| `can_fuse_horizontal(node1, node2)` | Sibling fusion — 检查两个 op 是否有重叠的输入 operand | 相同的优化目标：共享内存读取 |
| `score_fusion_memory()` — Σ size(shared_memory_dep) | Fusion cost model — 估算融合后的内存流量减少量 | 相同的评估指标 |
| `will_fusion_create_cycle()` — 融合前环检测 | SSA 形式下融合天然无环 | Triton 需显式检查；MLIR SSA 保证无环 |
| `reorder_for_peak_memory()` | `buffer-deallocation` + `buffer-hoisting` passes | 相同目标：最小化峰值内存 |
| `compute_last_usage()` | Liveness analysis — 标准编译器数据流分析 | 完全相同的编译器技术 |
| 10 轮 `fuse_nodes_once()` 贪心迭代 | Pass pipeline 固定顺序 + canonicalization 迭代至 fixpoint | Triton 贪心；MLIR 固定 pass 顺序 |

---

## 阶段 6: Codegen — 第三条核心分叉线

### Triton: IR → 单层 DSL → JIT

```
FusedSchedulerNode → TritonScheduling.codegen_node()
  → 确定 tiling (get_tiling_and_scores)
  → TritonKernel.codegen_kernel()
    → Grid 计算 (xnumel / ynumel / rnumel)
    → inner_fn λ → tl.load / tl.store / tl.sum / tl.dot
    → 输出完整 Triton Python 源码字符串
  → define_kernel(src_code) → PyCodeCache.load() → exec()
  → Triton JIT: Triton Python → Triton IR → Triton GPU IR → PTX → cubin
```

### MLIR: Dialect → 多层递降 → AOT

```
linalg.generic (已融合)
  → -convert-linalg-to-parallel-loops    // linalg → scf.parallel
  → -gpu-map-parallel-loops             // scf → gpu.block_id/thread_id
  → -convert-parallel-loops-to-gpu      // gpu.launch
  → -gpu-kernel-outlining              // extract gpu.func
  → -convert-gpu-to-nvvm               // gpu.func → nvvm.func
  → -convert-nvvm-to-llvm             // nvvm → llvm
  → -convert-llvm-to-ptx              // llvm → PTX 文本
  → mlir-translate + ptxas            // PTX → cubin
```

### 概念等价映射

| Triton 概念 | MLIR 等价概念 | 对等原理 |
|------------|-------------|---------|
| **Tiling** | | |
| `get_tiling_and_scores()` 搜索 BLOCK_M/BLOCK_N 等 | `linalg.tile` transform: `tile_sizes = [64, 64, 32]` | 同一数学操作。Triton autotune 搜索；MLIR 声明式指定 |
| `@triton.heuristics` (BLOCK_SIZE, num_warps, num_stages) | `gpu.launch` 的 `blockSize` + `gridSize` | GPU kernel launch configuration |
| **Kernel 主体** | | |
| `tl.program_id(0)` → `pid` | `gpu.block_id` (映射到 `blockIdx.x`) | 获取 block 的 grid 索引 |
| `tl.arange(0, BLOCK_SIZE)` → 线程内偏移 | `gpu.thread_id` (映射到 `threadIdx.x`) | block 内 thread 索引 |
| `tl.load(ptr + offset, mask=mask)` | `gpu.load` + `scf.if` (bounds check) | 带掩码的全局内存加载 |
| `tl.store(ptr + offset, val, mask=mask)` | `gpu.store` + bounds check | 带掩码的全局内存写入 |
| `tl.sum(val, axis=1)` 归约 | `linalg.reduce` → `gpu.shuffle` (warp reduce) 或 `gpu.all_reduce` (block reduce) | Triton 自动处理 warp/block reduce；MLIR 需显式 `gpu.shuffle` + barrier |
| `tl.zeros((BLOCK_M, BLOCK_N), dtype)` — 累加器初始化 | `memref.alloca` + `linalg.fill` — workgroup memory 中分配清零 | 分配并初始化寄存器/shared memory 空间 |
| **Shared Memory** | | |
| `tl.make_block_ptr` + `tl.advance` — 自动流水线管理 | `gpu.alloca workgroup` + 显式 `gpu.barrier` + `memref.copy` | Triton 抽象了 shared memory 流水线；MLIR 需显式管理 |
| **编译** | | |
| `PyCodeCache.load()` → `exec()` → `@triton.jit` → Triton JIT | `mlir-translate --mlir-to-nvvm` → `ptxas` | Triton JIT（首次执行编译）；MLIR AOT（编译时完成） |
| `async_compile.triton()` 异步编译 | Pass pipeline 串行执行；并行 pass 通过 pass manager 调度 | 多 kernel 编译的并行策略 |
| **Wrapper/内存** | | |
| `PythonWrapperCodegen.generate()` → `call(args)` | `gpu.launch_func` + buffer allocation/deallocation | 内存分配→参数打包→kernel launch→结果回收 |
| `AllocateLine` / `FreeIfNotReusedLine` / `ReuseLine` | `buffer-deallocation` + `buffer-hoisting` passes | 相同的内存复用技术 |
| `mutated_buffers` 处理 (in-place mutation) | `bufferization` pass — tensor→memref + inplace bufferization | SSA 不可变 tensor → 可变 buffer，允许原地修改 |

---

## 优劣势分析

### 1. 编译模型: JIT vs AOT

| | Triton (JIT) | MLIR (AOT) |
|---|---|---|
| **优势** | 零预编译时间。只有被调用的 kernel 才编译。Autotune 候选延迟到运行时编译 | 全图可见。整个模型的 MLIR 在编译时完整存在，可做跨 kernel 全局优化 |
| **劣势** | 跨 kernel 优化困难。每个 kernel 是独立 Python 字符串，公共子表达式消除只能依赖 buffer 复用 | 首次编译时间长。所有 pass 必须在运行时完成。大模型 pass pipeline 代价显著 |
| **典型场景** | 动态 shape 推理（增量重编译）；训练（大量重复调用摊销 JIT 成本） | 静态 shape 部署（export → 离线编译 → 二进制）；跨硬件 |

### 2. IR 设计: 单层 DSL vs 多层 Dialect

| | Triton (单层) | MLIR (多层) |
|---|---|---|
| **优势** | 理解成本低。Inductor IR → `tl.load/store` 映射透明。生成代码可读 | 每层可 inspect。可在任何 pass 后检查 IR。Linalg dialect 跨框架复用 |
| **劣势** | 无中间检查点。Inductor IR 不完整——无类型签名、mem layout、并行化信息 | 层数爆炸。5+ 层 IR 使 bug 追溯困难。学习曲线陡峭 |

### 3. 融合策略: 集中决策 vs 分散优化

| | Triton (集中) | MLIR (分散) |
|---|---|---|
| **优势** | 全局最优机会大。Scheduler 知道所有 buffer 生命周期。10 轮固定次数，结果确定 | 可组合。新增优化只需插入 pass。渐进式：先轻量 pass 后重量 pass |
| **劣势** | 硬编码 10 轮。深图可能不收敛。扩展规则需修改 Scheduler 核心 | Pass 顺序敏感。A→B→C vs B→A→C 结果不同。局部最优陷阱 |

### 4. 调试可观测性

| | Triton | MLIR |
|---|---|---|
| **优势** | 生成源码可读。`TORCH_LOGS="+fusion"` 显示每轮融合决策。`WhyNoFuse` 记录不融合原因 | 每层可 inspect。`mlir-opt --mlir-print-ir-after-all`。工具链成熟（mlir-opt/reduce/tblgen） |
| **劣势** | Triton JIT 内部不透明（shared memory swizzling、指令调度）。中间 IR 不可访问 | 信息过载。简单 `x+y` 经过 15 pass 产生数万行 MLIR。定位性能 bug 搜索空间大 |

### 5. 硬件适配

| | Triton | MLIR |
|---|---|---|
| **优势** | NVIDIA GPU 性能最优。针对 A100/H100 微架构深度 tuning（TMA、warp-group matmul、async copy） | 跨硬件容易。核心优化（Linalg fusion/tiling/vectorization）与 target 无关。换 target 只替换最后 1-2 层 dialect |
| **劣势** | 仅 NVIDIA + 少量 AMD。加新硬件（Apple GPU、TPU、Ascend）需写新 Triton 编译器后端 | 通用性有代价。Linalg 抽象不知 GPU warp 调度、bank conflict、register 大小等微架构细节 |

### 6. Autotune 与性能天花板

| | Triton | MLIR |
|---|---|---|
| **优势** | 搜索高效。直接 benchmark 实际 kernel 执行时间。Coordinate Descent 近似最优。Cache 持久化 | 搜索空间灵活。Transform dialect 表达任意 tiling pattern（如 hierarchical tiling） |
| **劣势** | Per-kernel 搜索不感知 kernel 间影响。动态 shape 变化 → cache miss → 重新 autotune | 搜索空间组合爆炸（tiling × fusion 变体 × pass 顺序）。需额外 cost model 剪枝 |

---

## 总结

```
┌──────────────────────────────────────────────────────────────┐
│                    Triton 适合的场景                           │
│  • NVIDIA GPU 训练/推理（95% 的实际场景）                       │
│  • 需要快速首次编译（JIT 仅编译用到的 kernel）                   │
│  • 动态 shape（缓存守卫 + 增量重编译）                           │
│  • 需要可读的生成代码来调试性能                                   │
│  • 小团队/快速迭代（Scheduler 行为可预测、可配置）                │
├──────────────────────────────────────────────────────────────┤
│                    MLIR 适合的场景                             │
│  • 非 NVIDIA 硬件（华为 Ascend、Apple GPU、自定义 AI 芯片）      │
│  • 需要跨模型全局优化（AOT 模式下可见全图）                       │
│  • 静态 shape 推理部署（export → 离线编译 → 二进制）             │
│  • 多框架统一编译栈（同一 Linalg 后端服务 PyTorch/TF/JAX）       │
│  • 编译器研究/自定义优化 pass                                    │
└──────────────────────────────────────────────────────────────┘
```

**一句话**: Triton 赢在"简单直接"——两层 IR、集中式 Scheduler、JIT 编译，对 NVIDIA GPU 是最优解。MLIR 赢在"架构灵活性"——多层 IR、可组合 pass、跨硬件，离开 NVIDIA 生态这是唯一可行的路。

---

## Related Pages

- [[02_engineering/01_ai_frameworks/index]]
- [[10_fx_lowering_to_inductor_ir_analysis]] — Inductor lowering 的完整技术分析
- [[13_scheduler_dependency_graph_fusion_and_ordering_analysis]] — Scheduler 融合算法详解
- [[20_inductor_codegen_analysis]] — Triton codegen 流程
- [[02_compile_stack/04_inductor/index]] — Inductor 完整编译流水线
- [[NPU_MLIR_Backend_Technical_Analysis]] — NPU 上的 MLIR 后端实现（TracedGraph 机制）
- [[20_npu_lowering_guide]] — NPU lowering 与 Triton lowering 的差异
- [[02_torch_compile_architecture]] — torch.compile 端到端架构
