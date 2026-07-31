# MLIR 核心概念: Dialect、Pass、IR 注册

> 从零理解 MLIR 编译器基础设施的三个核心机制：Dialect 词汇表、Pass 变换引擎、IR 注册链路
> 最后更新: 2026-05-11

---

## 1. 为什么需要 MLIR？

传统编译器是"一个前端 + 一个 IR + 一个后端"的单体。每新增一个硬件后端，需要重新实现整个编译栈。

MLIR 的设计目标：**让编译器变成可组合的乐高积木。** 不同抽象层级的 IR（dialect）可以混编在同一棵树中，Pass 逐层将"高层积木"替换为"低层积木"。

```
传统编译器:
  Python → [  一个大黑盒  ] → GPU 代码

MLIR 编译器:
  Python → [torch] → [linalg] → [gpu] → [nvvm] → GPU 代码
                ↘    ↙
              同一棵 IR 树，dialect 节点被逐批替换
```

---

## 2. Dialect: 可插拔的 IR 词汇表

### 核心定义

**Dialect = 一个命名空间下的 Operation + Type + Attribute 集合。** 它定义某个抽象层级"能说什么"。

MLIR 本身没有"一种统一的 IR"。它只定义了极薄的公共骨架——SSA 值、嵌套 Region/Block、Operation 基类。所有有意义的 IR 语法都由各自的 dialect 定义。

```
MLIR 公共骨架 (SSA, Region, Block, Operation 基类)
  ├── torch dialect   — 定义 torch.aten.add, !torch.vtensor
  ├── linalg dialect  — 定义 linalg.generic, linalg.matmul
  ├── gpu dialect     — 定义 gpu.launch, gpu.block_id
  ├── nvvm dialect    — 定义 nvvm.mma.sync
  ├── llvm dialect    — 定义 llvm.add, llvm.call
  ├── scf dialect     — 定义 scf.for, scf.if (结构化控制流)
  ├── func dialect    — 定义 func.func, func.return
  └── arith dialect   — 定义 arith.addf, arith.mulf
```

### 类比

如果把编译器比作翻译系统：

- **Triton**: 只有两本词典。Inductor IR (Pointwise/Reduction) → Triton (tl.load/tl.store/tl.sum)。
- **MLIR**: 有几十本词典。每一本对应一个抽象层级。"翻译"就是一本词典换到下一本，逐步接近硬件能理解的语言。

### MLIR 代码中的 Dialect 混编

```mlir
// 同一个函数体内，多种 dialect 的 Op 共存
func.func @main(%x: tensor<128x256xf32>, %y: tensor<256x512xf32>)
    -> tensor<128x512xf32> {     // func dialect
  %0 = linalg.matmul ins(%x, %y : ...) 
      outs(...) -> tensor<...>   // linalg dialect
  %1 = arith.addf %0, %0 : ...  // arith dialect
  return %1 : ...                // func dialect
}
```

---

## 3. IR 树形结构

MLIR 的 IR 是嵌套的树。这是理解 Pass 如何工作的前提。

### 层级关系

```
ModuleOp                          ← 顶层 module
  └── Region (body)
      └── Block
          ├── func.func @main    ← Function Op
          │   └── Region (body)
          │       └── Block
          │           ├── %0 = linalg.generic { ... }
          │           │   └── Region (计算体)
          │           │       └── Block
          │           │           └── arith.addf %arg0, %arg1
          │           ├── %1 = gpu.launch { ... }
          │           │   └── Region (kernel body)
          │           │       └── Block
          │           │           ├── %tid = gpu.thread_id x
          │           │           └── gpu.store %val, %ptr[%tid]
          │           └── func.return %1
          └── ...
```

**关键术语**:

| 概念 | 含义 | 类比 |
|------|------|------|
| **Operation (Op)** | IR 的最小语义单元。有名字、参数、结果、可选的嵌套 Region | 语法树中的一个节点 |
| **Value** | Operation 的结果，SSA 单一定义 | 节点之间的边 |
| **Region** | Operation 内的嵌套区域，包含 Block 序列 | 节点的"内部实现" |
| **Block** | 有序的 Operation 列表，最后一个 Op 必须是 terminator | 基本块 |
| **ModuleOp** | 顶层容器 Op | 编译单元 |

### 和 Triton / PyTorch 的概念对等

| PyTorch/Inductor | MLIR |
|------------------|------|
| `torch.fx.Node` (op + args + kwargs) | `Operation` (op name + operands + attributes + results) |
| `torch.fx.Graph` (有向无环图) | `Block` (Operation 的有序列表) |
| `torch.fx.GraphModule` | `ModuleOp` (顶层容器) |
| `TensorBox` (对 IRNode 的引用) | `Value` (SSA 值的引用) |
| `Pointwise.inner_fn` (λ 闭包, 描述计算) | `Region` (嵌套的 Block, 描述计算) |

---

## 4. Pass: 就地变换 IR 树

### 核心定义

**Pass = 在 IR 树上做就地替换的程序。** 每个 Pass 声明自己的"作用域"——它在树的哪一层操作。

关键理解：MLIR 只有一棵 IR 树。Pass 不创建新树，而是在这棵树上把子树 A 换成子树 B。这就是"递降 (progressive lowering)"。

### 四种 Pass 作用域

| Pass 类型 | 作用域 | 典型场景 |
|-----------|--------|---------|
| **Op-specific** | 遍历整棵树，找特定类型的 Op，只对该 Op 做变换 | `LinalgFuseElementwiseOps` — 找 `linalg.generic` 并融合 |
| **Dialect** | 找某个 dialect 的所有 Op，做跨 dialect 转换 | `ConvertTorchToLinalg` — 找所有 `torch.aten.*`，替换为 `linalg.*` |
| **Function** | 以一个 `func.func` 为单位，pass 看到整个函数体 | `CSE` — 公共子表达式消除，函数范围内找重复计算 |
| **Module** | 以顶层 `module` 为单位，pass 看到整个编译单元 | `Inline` — 跨函数边界内联 |

### 四种 Pass 作用域的设计哲学

**为什么需要四种作用域？** 本质原因：MLIR 的 IR 树是**多层异构嵌套**结构（同一棵树内混编 torch/linalg/gpu/scf 等不同 dialect），Pass 必须精确声明"我能看到什么、我能动什么"才能保证：

1. **安全性**：`ConvertTorchToLinalg` 不该触碰已经被降级为 `gpu.launch` 的节点。Pass 作用域是"访问控制"——它保证每层 pass 只看到自己应该看到的抽象层级。
2. **可组合性**：每个 Pass 做**一件事且只做一件事**。合并多个单功能 Pass 比写一个"超级 Pass"更灵活。新增优化只需插入一个新 Pass，不影响现有管线。
3. **并行调度**：Function-scoped 的 Pass（如 CSE、canonicalize）可以**并行执行**在每多个 `func.func` 上——Pass Manager 自动利用多核。这是编译器工程中"数据并行"的经典运用。
4. **测试与调试**：当 Pass 行为异常时，独立作用域让你可以单独运行一个 Pass（`mlir-opt --pass-pipeline="builtin.module(func.func(cse))"`），精确定位问题。

**为什么不像 Triton 做全局优化？**

这不是 MLIR 的缺陷，而是**IR 结构的根本差异**决定的：

```
Triton 的 IR 是"单层扁平":
  Pointwise(add) → Pointwise(mul) → Reduction(sum)
  所有节点都在同一个抽象层级 → Scheduler 全局可见 → 10 轮贪心融合

MLIR 的 IR 是"多层嵌套":
  torch.aten.add (高层框架语义)
    ↓ ConvertTorchToLinalg
  linalg.generic (中层线性代数语义)
    ↓ LinalgFuseElementwise
  linalg.generic (已融合)
    ↓ ConvertLinalgToGPU
  gpu.launch (低层并行语义)

  每个层级之间不可直接比较、不可直接融合
```

Triton 之所以可以全局优化，是因为 Inductor 的 lowering **丢弃了高层语义**——`aten.add` 变成 `Pointwise(inner_fn=λ...)`，所有节点坍缩为两种 IR 节点（Pointwise/Reduction）。这是"牺牲信息换简单性"。MLIR **保留了每层的完整语义**，代价是"牺牲简单性换可组合性"——每层需要独立的 Pass 来处理该层级的优化。

**与 Eager Mode 的概念对应**

| MLIR Pass 作用域 | 操作对象 | Eager Mode 概念对等 | 对应程度 |
|-----------------|---------|-------------------|---------|
| **Op-specific** | 单个 `linalg.generic`、单个 `scf.for` | **单算子 dispatch**：`aten.add(x,y)` → 后端 kernel。Eager 的每次调用只处理一个 op | ★★★★ 最接近 |
| **Dialect** | 某个命名空间下的所有 Op | **算子库切换**：把所有 `torch.ops.npu.*` 批量替换为 `torch.ops.ascend.*`。Eager 中通过 `__torch_dispatch__` 实现 | ★★★ 概念接近 |
| **Function** | 一个 `func.func` 的 body | **`torch.jit.script` 的函数**：对单个函数做优化。Eager 中没有此概念 | ★★☆ 需要图模式 |
| **Module** | 整个 `module`（跨函数） | **`torch.compile(Model())`**：整个模型的图级优化。Eager 本身没有，必须进入 compile 模式 | ★☆☆ 仅在编译模式 |

> **关键洞察**：Eager Mode 的本质是"每步执行后立即忘记上下文"——它天然运行在 Op-specific 粒度。MLIR 的四种 Pass 作用域本质上是在**回补 Eager Mode 缺失的信息层级**：Function Pass 回补"这些 op 在同一个函数内"的边界信息，Module Pass 回补"这些函数属于同一个模型"的全局信息。

### 实际运作

```
一棵 IR 树（嵌套结构）:

module {                          ← Module Pass 的视野
  func.func @main(...) {          ← Function Pass 的视野
    %0 = linalg.generic { ... }   ← Op-specific Pass 的目标
    %1 = linalg.generic { ... }   ← Op-specific Pass 的目标
    return %1
  }
}
```

`ConvertTorchToLinalg` pass 的伪代码逻辑：

```
walk(所有 Operation):
  if op 是 torch.aten.add.Tensor:
    // 创建 linalg.generic { arith.addf }，在 IR 树中替换
    replaceOpWithNewOp<linalg::GenericOp>(op, ...)
  elif op 是 torch.aten.mm:
    replaceOpWithNewOp<linalg::MatmulOp>(op, ...)
```

### Pass Manager

多个 Pass 通过 Pass Manager 编排执行顺序：

```cpp
// 一个典型的 GPU 编译 pass pipeline
mlir::PassManager pm(&context);
pm.addPass(mlir::torch::createConvertTorchToLinalgPass());  // Step 1
pm.addPass(mlir::createLinalgFuseElementwiseOpsPass());     // Step 2
pm.addPass(mlir::createConvertLinalgToGPU());               // Step 3
pm.addPass(mlir::createGpuKernelOutliningPass());           // Step 4
pm.addPass(mlir::createConvertGPUToNVVMPass());             // Step 5
pm.run(module);
```

**关键**: Pass 之间可以并行执行无依赖的 pass（如对每个 `func.func` 独立运行），Pass Manager 自动处理调度。

### 与 Triton 的概念对等

| Triton | MLIR |
|--------|------|
| `Scheduler` — 全局依赖图 + 10 轮融合 | `Pass Manager` — 编排 pass 执行顺序 |
| `fuse_nodes()` 单轮 → 创建 `FusedSchedulerNode` | `LinalgFuseElementwiseOps` pass → 合并相邻 `linalg.generic` |
| `can_fuse_vertical(node1, node2)` | Producer-consumer fusion legality check (在 pass 内) |
| `Scheduler.codegen()` → `TritonKernel.codegen_kernel()` | `ConvertLinalgToGPU` → `ConvertGPUToNVVM` → ... |
| `compute_last_usage()` — 确定 buffer 最后读取位置 | Liveness analysis pass (标准编译器数据流分析) |
| `reorder_for_peak_memory()` | `buffer-deallocation` + `buffer-hoisting` passes |

### 实例: 上游 MLIR ElementwiseOpFusion 源码解析

以 `mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp`（源自上游 LLVM/MLIR 与 torch-mlir 社区项目，本地代码库不含其源码，以上游为准）为例，这是一个 **Op-specific Pass**（只匹配 `linalg.generic`），功能等价于 Triton 的 `can_fuse_vertical` + `fuse_nodes()`。

**合法性检查 `areElementwiseOpsFusable`**：

```cpp
bool areElementwiseOpsFusable(OpOperand *fusedOperand) {
    auto producer = fusedOperand->get().getDefiningOp<GenericOp>();
    auto consumer = dyn_cast<GenericOp>(fusedOperand->getOwner());

    // ① 必须是两个 linalg.generic（对等 Triton 的 Pointwise+Pointwise）
    if (!producer || !consumer) return false;

    // ② Producer 必须全 parallel（对等 Triton 的 Pointwise 无 reduction）
    if (producer.getNumParallelLoops() != producer.getNumLoops())
        return false;

    // ③ fused operand 必须是 consumer 输入（对等 Triton 的 unmet_deps 检查）
    if (!consumer.isDpsInput(fusedOperand)) return false;

    // ④ 索引维度匹配（对等 Triton 的 iteration domain 兼容检查）
    AffineMap consumerIndexMap = consumer.getMatchingIndexingMap(fusedOperand);
    if (consumerIndexMap.getNumResults() != producer.getNumLoops())
        return false;

    // ⑤ producer 的结果索引 map 必须是可逆排列
    AffineMap producerResultIndexMap =
        producer.getIndexingMapMatchingResult(producerResult);
    if (!producerResultIndexMap.isPermutation()) return false;

    return true;
}
```

**融合执行 `fuseElementwiseOps`**：

```cpp
// 核心步骤：
// 1. 收集 Consumer 在 fusedOperand 之前的输入
// 2. 收集 Producer 全部输入（索引 map 重算到 fused 坐标系）
// 3. 收集 Consumer 在 fusedOperand 之后的剩余输入
// 4. 收集需要保留的 Producer 输出（被 consumer 外的用户使用）
// 5. 创建新 fused GenericOp，调用 generateFusedElementwiseOpRegion 合并计算体

auto fusedOp = GenericOp::create(
    rewriter, consumer.getLoc(), fusedResultTypes,
    fusedInputs, fusedOutputs,
    rewriter.getAffineMapArrayAttr(fusedIndexMaps),
    consumer.getIteratorTypes());

// 验证 fused op 合法（循环边界计算）
if (!fusedOp.getShapesToLoopsMap()) {
    rewriter.eraseOp(fusedOp);  // 融合非法，回滚
    return failure();
}

// 生成 fused region: 把 producer 和 consumer 的计算体拼接
generateFusedElementwiseOpRegion(
    rewriter, fusedOp, consumerToProducerLoopsMap,
    fusedOperand, consumer.getNumLoops(), preservedProducerResults);
```

**融合前后的 IR 变化**：

```mlir
// 融合前：两个独立 linalg.generic，%add 的结果需要写回内存再被 %mul 读出
// %0 = producer:
//   linalg.generic { ... } ins(%x, %y) outs(%init0) {
//     %add = arith.addf %a, %b  →  linalg.yield %add
//   }
// %1 = consumer:
//   linalg.generic { ... } ins(%0, %z) outs(%init1) {
//     %mul = arith.mulf %d, %e  →  linalg.yield %mul
//   }

// 融合后：单次 kernel launch，中间值 %add 走寄存器，不写回 HBM
// %fused = linalg.generic { ... } ins(%x, %y, %z) outs(%init1) {
//   %add = arith.addf %a, %b
//   %mul = arith.mulf %add, %e  // d 被替换为 producer 的 %add
//   linalg.yield %mul
// }
```

**与 Triton 融合作检查项的一一对应**：

| 检查项 | MLIR `areElementwiseOpsFusable` | Triton `can_fuse_vertical` |
|--------|-------------------------------|---------------------------|
| Op 类型 | producer/consumer 都是 `GenericOp` | node1/node2 都是 `BaseSchedulerNode` |
| 迭代类型 | producer 全 parallel | node1 无 reduction (Pointwise) |
| 数据依赖 | consumer 的 operand 来自 producer 的 result | node2 的 `unmet_deps` 由 node1 满足 |
| 索引兼容 | affine map 可逆排列 | `shared_data_score` 计算内存重叠 |
| 环检测 | SSA 天然无环 | `will_fusion_create_cycle()` 显式检查 |
| 驱动方式 | `OpRewritePattern` 反复匹配至 fixpoint | 10 轮 `fuse_nodes_once()` 贪心迭代 |

### torch-mlir 自有 Pass 实例: FuseQuantizedOps

`torch-mlir/lib/Dialect/Torch/Transforms/FuseQuantizedOps.cpp` 是一个 **Dialect 级 Pass**，展示 torch-mlir 特有的优化模式：

```cpp
// 核心 Pattern: 将 quantize → dequantize → compute(float) 融合为 compute(int)
class QuantizeOperandsPastCommutingOps : public OpRewritePattern<SrcOp> {
    LogicalResult matchAndRewrite(SrcOp op, PatternRewriter &rewriter) const {
        // 对每个需要量化的 operand:
        for (auto operand : QuantInfo<SrcOp>::operandsToQuantize(op)) {
            // 沿 def-use 链向上追溯:
            //   Case 1: 遇到 commuting op (transpose/reshape/view/slice)
            //     → 入栈，继续向上追溯
            //   Case 2: 遇到 dequantize op
            //     → 捕获上游 quantized tensor (int + scale + zero_point)
            //   Case 3: 都不是 → 放弃融合
        }
        // 重写: 用整数类型重建 commuting 链 + 计算 op
        // 在最外层重新附加 quantize/dequantize
    }
};
```

**关键设计差异**：

| 维度 | MLIR `ElementwiseOpFusion` | torch-mlir `FuseQuantizedOps` | Triton Scheduler |
|------|--------------------------|------------------------------|-----------------|
| **层次** | Linalg dialect (中层 IR) | Torch dialect (高层 IR) | Inductor IR (单层) |
| **Pass 类型** | Op-specific（只匹配 `linalg.generic`） | Op-specific（匹配量化链 ATen op） | 全局 Scheduler |
| **实现语言** | C++ (链接 libMLIR) | C++ (链接 libTorchMLIR) | Python |
| **需要编译** | 是 (mlir-opt 或动态库) | 是 (torch-mlir-opt 或动态库) | 否 |
| **融合范围** | 单 producer→consumer 对 | 量化链 (q→dq→compute→q) | 全局依赖图 10 轮贪心 |
| **驱动方式** | `OpRewritePattern` → fixpoint | `GreedyRewriteConfig` + depth 控制 | `fuse_nodes()` × 10 |

---

## 5. 自定义 Dialect 注册

### 完整注册链路

```
TableGen (.td 文件)              ← 声明式定义 Op/Type/Attribute
  ↓ mlir-tblgen 代码生成
生成的 .h.inc (C++ 类定义)
  ↓ C++ 编译
Dialect::initialize() 中注册
  ↓ 运行时
MLIRContext::getOrLoadDialect<>()
  ↓
Dialect registry 全局可用
  ↓
Parser 遇到 "myacc.matmul" → 查 registry → 调用 parse()
```

### Step 1: TableGen 定义 Op

```tablegen
// MyAccelerator.td
include "mlir/IR/OpBase.td"

// 定义 dialect 命名空间
def MyAccelerator_Dialect : Dialect {
  let name = "myacc";                     // 在 IR 文本中的前缀
  let cppNamespace = "::mlir::myacc";     // C++ 命名空间
}

// 定义 dialect 的 Op 基类
class MyAccelerator_Op<string mnemonic, list<Trait> traits = []>
    : Op<MyAccelerator_Dialect, mnemonic, traits>;

// 定义具体 Op
def MyAccelerator_MatMulOp : MyAccelerator_Op<"matmul"> {
  let summary = "matrix multiplication on custom accelerator";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  // 定义文本格式: myacc.matmul %a, %b : type(a) * type(b) -> type(result)
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `*` type($rhs) `->` type($result)";
}
```

**TableGen 自动生成的内容**: C++ 类 (`MatMulOp`)、parser (解析 `myacc.matmul %a, %b`)、printer (序列化为文本)、verifier (校验参数类型合法性)、builder (构造函数)。

### Step 2: C++ 胶水代码注册

```cpp
// MyAcceleratorDialect.cpp
#include "MyAcceleratorDialect.h.inc"  // 自动生成的 dialect 定义

void MyAcceleratorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MyAccelerator.h.inc"  // 自动生成的 Op 列表
#undef GET_OP_LIST
  >();
}

// 全局类型 ID (MLIR 的 RTTI 替代)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::myacc::MyAcceleratorDialect)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::myacc::MyAcceleratorDialect)
```

### Step 3: 运行时加载

```cpp
mlir::MLIRContext context;
// 加载需要的 dialect 到 context 的全局 registry
context.getOrLoadDialect<mlir::myacc::MyAcceleratorDialect>();

// 之后 parser 就能识别 myacc.matmul 语法
auto module = mlir::parseSourceString<mlir::ModuleOp>(
    R"(
      func.func @test(%a: tensor<128x256xf32>, %b: tensor<256x512xf32>)
          -> tensor<128x512xf32> {
        %0 = myacc.matmul %a, %b : tensor<128x256xf32> * tensor<256x512xf32>
            -> tensor<128x512xf32>
        return %0 : tensor<128x512xf32>
      }
    )",
    &context
);
```

### Step 4: 写降级 Pass

```cpp
class ConvertLinalgToMyAccelerator
    : public mlir::PassWrapper<ConvertLinalgToMyAccelerator,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  void runOnOperation() override {
    getOperation().walk([&](mlir::linalg::MatmulOp matmul) {
      if (isSizeCompatible(matmul)) {
        mlir::OpBuilder builder(matmul);
        // 创建 myacc.matmul 并替换 linalg.matmul
        auto myOp = builder.create<mlir::myacc::MatMulOp>(
            matmul.getLoc(), matmul.getResult().getType(),
            matmul.getInputs()[0], matmul.getInputs()[1]);
        matmul.replaceAllUsesWith(myOp);
        matmul.erase();
      }
    });
  }
};
```

### Parser 如何找到 Op?

MLIR 的 `MLIRContext` 维护一个全局 dialect registry。Parser 遇到 `myacc.matmul` 时：

```
"myacc.matmul"
  ↓ 按 "." 分割 → dialect name = "myacc", op name = "matmul"
  ↓ 查 MLIRContext 的 dialect registry
  ↓ 找到 MyAcceleratorDialect
  ↓ 调用 MyAcceleratorDialect::lookupOperation("matmul")
  ↓ 返回 MatMulOp 的 parse 函数指针
  ↓ parse 函数从文本流解析操作数、类型，构造 Operation
```

---

## 6. 递降实例: 一个 ATen add 的完整旅程

以 `aten.add.Tensor(x, y)` 在 GPU 上编译为例：

```
【FX Graph】
  %add = aten.add.Tensor(%x, %y)

【Step 1: FxImporter → Torch dialect】
  %0 = torch.aten.add.Tensor %x, %y, %alpha
    : !torch.vtensor<[128,256],f32> → !torch.vtensor<[128,256],f32>

【Step 2: ConvertTorchToLinalg pass】
  %0 = linalg.generic {
    indexing_maps = [affine_map<(i,j) -> (i,j)>,
                     affine_map<(i,j) -> (i,j)>,
                     affine_map<(i,j) -> (i,j)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%x, %y : tensor<128x256xf32>)
    outs(%init : tensor<128x256xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %add = arith.addf %a, %b : f32
      linalg.yield %add : f32
  } -> tensor<128x256xf32>

【Step 3: LinalgFuseElementwiseOps pass】
  // 如果邻接的 linalg.generic 可融合，合并为一个大 generic
  // (此例中可能不变，因为只有一个 op)

【Step 4: ConvertLinalgToGPU pass】
  gpu.launch blocks(%bx, %by) in (%gridX, %gridY)
             threads(%tx, %ty) in (%blockX, %blockY) {
    // 将 linalg.generic 的 body 映射到 GPU thread
    %val = memref.load %x[%bx * %blockX + %tx, %by * %blockY + %ty]
    %val2 = memref.load %y[...]
    %sum = arith.addf %val, %val2 : f32
    memref.store %sum, %out[...]
  }

【Step 5: ConvertGPUToNVVM pass】
  // gpu.launch → nvvm 内置函数
  %tid = nvvm.read.ptx.sreg.tid.x   // 读取 threadIdx.x
  %bid = nvvm.read.ptx.sreg.ctaid.x // 读取 blockIdx.x
  ...

【Step 6: ConvertNVVMToLLVM + ConvertLLVMToPTX】
  // 最终降为 PTX 汇编 → cubin → GPU 可执行
```

**关键观察**: 整个过程没有"新建 IR"。每一步都是**在原来的 IR 树上就地替换子树**。第 N 步的输出是第 N+1 步的输入，始终只有一棵树。

---

## 7. Dialect 的边界: 什么在 Dialect 内, 什么在 Dialect 外？

| 在 Dialect 内 (每个 dialect 自己定义) | 在 Dialect 外 (MLIR 公共骨架提供) |
|--------------------------------------|----------------------------------|
| Operation 名称和语义 (`torch.aten.add`, `linalg.matmul`) | SSA 值系统 (Value 有单一定义) |
| Type 定义 (`!torch.vtensor`, `tensor<...xf32>`) | Region/Block 嵌套结构 |
| Attribute 定义 (常量参数) | Pass Manager (pass 调度) |
| Verifier (Op 合法性校验) | Rewriter (替换 Op 的基础设施) |
| Parser/Printer (文本 ↔ Op 双向转换) | Location (源码位置追踪) |

---

## 总结

```
MLIRContext (全局上下文)
  │
  ├── 加载 Dialect A (torch)
  │   ├── 注册 Op: torch.aten.add, torch.aten.mm, ...
  │   ├── 注册 Type: !torch.vtensor<...>
  │   └── 注册 Attribute: ...
  │
  ├── 加载 Dialect B (linalg)
  │   ├── 注册 Op: linalg.generic, linalg.matmul, ...
  │   └── 注册 Type: ...
  │
  └── Pass Manager (遍历一棵 IR 树)
      ├── Pass 1: ConvertTorchToLinalg    → 替换 torch.aten.* → linalg.*
      ├── Pass 2: LinalgFuseElementwise   → 融合相邻 linalg.generic
      ├── Pass 3: ConvertLinalgToGPU      → 替换 linalg.* → gpu.*
      ├── Pass 4: GpuKernelOutlining      → 提取 gpu.func
      ├── Pass 5: ConvertGPUToNVVM        → 替换 gpu.* → nvvm.*
      ├── Pass 6: ConvertNVVMToLLVM       → 替换 nvvm.* → llvm.*
      └── Pass 7: ConvertLLVMToPTX        → llvm.* → PTX 文本
```

**三个核心概念的关系**：

- **Dialect** = 可插拔的"词汇表"（定义某个抽象层级能表达什么）
- **Pass** = 在一棵 IR 树上就地做"翻译+优化"的程序
- **注册** = TableGen 声明 → 自动生成 C++ → MLIRContext 加载 → Parser 可识别

**与 Triton 的根本差异**：Triton 是"两层词典，一个翻译步骤"（Inductor IR → Triton 源码）。MLIR 是"几十本词典，Pass Manager 编排的多步翻译流水线"。哪种更好取决于场景——GPU 上 Triton 的简单直接胜出；离开 NVIDIA 生态时 MLIR 的可组合性不可替代。

---

## 补充：MLIR 生态的四个关键扩展

> 最后更新: 2026-05-12

### A. MLIR Mesh Dialect：通信作为一等公民

**背景**：传统 ML 编译器（包括早期 MLIR）将集合通信（AllReduce、AllGather、All-to-All）视为"外部调用"——它们不出现在 IR 中，编译器 Pass 无法感知和优化通信。

**Mesh Dialect 解法**：

```mlir
// 1. 声明分布式 mesh（设备拓扑）
mesh.mesh @model_mesh<["tensor"=4, "data"=8]>  // 4-way TP × 8-way DP

// 2. 通信算子作为普通 IR Op（可被 Pass 分析和重排）
%ag_result, %token = mesh.all_gather %local_shard
    on @model_mesh[<"tensor">]
    async : tensor<256x512xf32> -> (tensor<1024x512xf32>, !mesh.token)

// 3. async token 表达依赖关系（细粒度同步）
%partial = mesh.wait_chunk %token, 0..256 : tensor<256x512xf32>
%result = linalg.matmul %partial, %weight ...
// → Pass 可以分析：matmul 只需要 all_gather 的前 256 行
//   → 可以生成 WaveEP 风格的流水线：后续 chunk 与 matmul 并行
```

**对 Pass 体系的影响**：
- 通信算子可以参与 `reorder_for_locality`（与计算节点重排）
- `bucket_*` Pass 的等价操作可以在 Mesh Dialect 层实现
- 未来：WaveEP 的 wave-level 流水线调度可以作为 Mesh Dialect Pass 自动生成

**当前状态**（截至 2026-05）：Mesh Dialect 已进入 MLIR 上游，但 Pass 体系（通信与计算的自动调度）仍在积极开发中。

---

### B. IREE：从 MLIR 到硬件的完整独立编译器

**IREE（Internet Relay Execution Engine）** 是基于 MLIR 的完整 ML 部署编译器，目标是"一套代码，所有硬件"：

```
IREE 编译流水线：

输入：StableHLO / MLIR Torch Dialect / TOSA
        ↓
┌─────────────────┐
│  Flow Dialect    │  ← 数据流分析、算子分发策略
│  Pass 重点:      │    workgroup 粒度决策
│  - 算子分组      │    I/O 分析
│  - 分发策略      │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Stream Dialect  │  ← 异步执行模型、内存管理
│  Pass 重点:      │    async execution + barrier 插入
│  - 异步调度      │    buffer 生命周期
│  - 内存规划      │
└────────┬────────┘
         ↓
┌─────────────────┐
│  HAL Dialect     │  ← 硬件抽象层（统一接口）
│  后端:           │
│  - CUDA (GPU)    │
│  - Vulkan (GPU)  │
│  - Metal (Apple) │
│  - ROCm (AMD)    │
│  - 昇腾 NPU      │
└─────────────────┘
```

**IREE 的核心差异化**：

| 特性 | IREE | torch.compile |
|------|------|--------------|
| 通信感知 | Stream Dialect 原生异步 | Post-Grad Pass 插入 |
| 硬件无关 | ✅（HAL 统一接口）| ❌（Triton=CUDA, MLIR=NPU）|
| AOT 部署 | ✅（生成独立二进制）| 部分（torch.export）|
| 社区状态 | `iree-org/iree-turbine`（活跃）| `pytorch/pytorch`（主干）|

**与 torch-mlir 的关系**：
```
torch-mlir（llvm/torch-mlir）:
  FX Importer → Torch Dialect → Linalg → ...（到 LLVM 为止）
  
IREE（iree-org）:
  接收 StableHLO/Linalg → Flow → Stream → HAL → 硬件二进制
  
组合使用：
  torch.compile（FX Graph捕获）
    → torch-mlir（Torch Dialect → Linalg）
    → IREE（Linalg → HAL → 硬件）
```

---

### C. StableHLO：跨框架的稳定 IR 锚点

**背景**：XLA HLO（High-Level Optimizer）是 Google 内部的 ML IR，但版本不稳定，社区难以依赖。

**StableHLO** 是从 HLO 提取的**稳定化公开版本**，作为跨框架的 IR 锚点：

```
框架层:
  PyTorch (torch.export) ─┐
  JAX                     ├──→ StableHLO IR（稳定、版本化）
  TensorFlow              ─┘
  
后端层:
  StableHLO ──→ XLA 执行（Google TPU/GPU）
  StableHLO ──→ IREE（多硬件部署）
  StableHLO ──→ 厂商自定义后端（接入 stablehlo-to-custom）
```

**StableHLO 提供的通信算子**：

```mlir
// 集合通信作为 StableHLO Op
%result = stablehlo.all_reduce %input
    replica_groups = [[0, 1, 2, 3]]
    channel_handle = {}
    : (tensor<8x8xf32>) -> tensor<8x8xf32>

// all_gather、reduce_scatter、collective_permute 同理
// → 后端可以将这些 Op 映射到 NCCL/DeepEP/自研通信库
```

**意义**：StableHLO 打通了"框架描述并行语义 → 编译器生成通信代码"的路径，是 GSPMD 等自动并行工具的 IR 基础。

---

### D. Triton 3.x：向 MLIR 后端迁移

**Triton 2.x 的 IR 架构**（两层）：

```
Triton DSL → TritonGPU IR → LLVM IR → PTX
```

**Triton 3.x 的变化**：Triton 的内部编译器正在迁移为基于 MLIR 的多 Dialect 架构：

```
Triton DSL
    ↓
Triton Dialect（MLIR）    ← 新增：Triton 算子的 MLIR 表示
    ↓
TritonGPU Dialect（MLIR） ← 新增：GPU 分块/调度相关语义
    ↓
LLVM Dialect（MLIR）
    ↓
PTX
```

**迁移的动机**：
1. **MLIR Pass 复用**：Triton 可以使用 MLIR 生态的标准变换（CSE、DCE、Canonicalization）
2. **Linalg 融合**：Triton kernel 的 tile 操作可以被 MLIR Linalg Pass 进一步优化
3. **TMA 支持**（H100 Tensor Memory Accelerator）：通过新的 Triton MLIR Dialect 描述异步 copy 语义
4. **生态对接**：torch-mlir 的 FxImporter 可以直接生成 Triton Dialect，无需经过 Inductor

**对 torch.compile 的影响**：

```
未来可能的路径：
  FX Graph
    ↓ FxImporter（今天 torch-mlir 做的事）
    ↓ Torch Dialect → Linalg Dialect
    ↓ Linalg Tiling + Fusion Pass
    ↓ Triton Dialect（直接生成，跳过 Inductor）
    ↓ TritonGPU Dialect（Triton 内部 MLIR）
    ↓ PTX

这条路径使得 Inductor Scheduler 的融合决策
可以在 Linalg Pass 层做，享受 MLIR 的形式化保证。
```

**当前状态**（截至 2026-05）：Triton 3.x MLIR 后端在 `triton-lang/triton` 的 `main` 分支上，H100 TMA 相关特性已可用，整体迁移仍在进行中。

---

## Related Pages

- [[11_torch_mlir_pass_pipeline_analysis]] — torch-mlir Pass 管线: 按执行顺序的 34 个 Pass 完整分析
- [[30_triton_vs_mlir_backend_analysis]] — Triton vs Torch-MLIR: 六阶段概念对等映射
- [[npu_mlir_backend_technical_analysis]] — NPU 上 MLIR 后端的完整实现 (TracedGraph、融合、编译)
- [[20_npu_lowering_guide]] — NPU lowering 与 Triton lowering 架构对比
- [[02_compile_stack/04_inductor/index]] — torch.compile 端到端流水线
- [[11_inductor_ir_values_loops_layouts_and_buffers_analysis]] — Inductor IR 与 Triton 代码生成
- [[31_comm_compute_fusion_guide]] — Mesh Dialect 在通算自动融合中的作用
- [[tilelang_analysis]] — TileLang 与 Triton 3.x MLIR 的 tile-level 概念关系
- [[10_mindspore_compiler_analysis]] — MindSpore ANF 图与 MLIR Dialect 的 IR 对比
