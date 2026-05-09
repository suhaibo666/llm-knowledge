# MLIR 核心概念: Dialect、Pass、IR 注册

> 从零理解 MLIR 编译器基础设施的三个核心机制：Dialect 词汇表、Pass 变换引擎、IR 注册链路
> 最后更新: 2026-05-09

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

## Related Pages

- [[triton_vs_mlir_backend_analysis]] — Triton vs Torch-MLIR: 六阶段概念对等映射
- [[NPU_MLIR_Backend_Technical_Analysis]] — NPU 上 MLIR 后端的完整实现 (TracedGraph、融合、编译)
- [[npu_lowering_guide]] — NPU lowering 与 Triton lowering 架构对比
- [[torch_compile_architecture]] — torch.compile 端到端流水线
- [[PyTorch_Inductor_Technical_Analysis]] — Inductor IR 与 Triton 代码生成
