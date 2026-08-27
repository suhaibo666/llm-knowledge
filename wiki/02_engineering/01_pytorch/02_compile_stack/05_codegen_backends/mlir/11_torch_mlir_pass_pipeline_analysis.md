---
title: "torch-mlir Pass 管线: 完整执行序分析"
---

# torch-mlir Pass 管线: 完整执行序分析

> 基于上游 `llvm/torch-mlir` 源码（源自 torch-mlir 上游社区项目，本地代码库不含其源码，以上游为准），追踪 `torch.compile` 自定义 backend → torch-mlir 导入/降级链 → MLIR Pass 管线的完整序列
> 最后更新: 2026-05-11

---

## 0. 定位说明: torch-mlir 在 PyTorch 生态中的角色

> 快速上手见 [[01_torch_mlir_quickstart]]（torch-mlir 在 PyTorch 生态的定位、与 Inductor 的关系、何时用、最小 backend 骨架、output_type 选择）。本节只保留**理解下文 Pass 管线**所需的最小上下文。

torch-mlir 的标准接法是写一个 `torch.compile(..., backend=...)` 的自定义 backend，把 Dynamo 捕获的 `gm` (GraphModule) 送进 torch-mlir 的导入/降级链。其关键入口函数 `stateless_fx_import(gm, output_type=...)` 接收 Dynamo 产出的 `torch.fx.GraphModule`，不需要经过 `torch.export`:

```python
# fx.py — stateless_fx_import 签名
def stateless_fx_import(
    gm: torch.fx.GraphModule,      # ← 直接接收 Dynamo 捕获的 FX Graph
    output_type: Union[str, OutputType] = OutputType.RAW,
    ...
):
    # 内部调用:
    #   FxImporter.import_stateless_graph(gm.graph) → FX → MLIR Torch Dialect
    #   _module_lowering(module) → 运行 MLIR Pass 管线
    #   lower_mlir_module(module) → 后端降级 (Linalg/TOSA/StableHLO)
```

**本文分析的就是这条线**: `torch.compile(backend=custom_mlir)` → Dynamo 捕获 FX Graph → `stateless_fx_import(gm)` → MLIR Pass 管线 → Linalg-on-Tensors → GPU。

三条路径（默认 Inductor / torch-mlir 自定义 backend / NPU monkey-patch）的对比汇总见本文 [§8](#8-三条路径对比总结)。

---

## 1. 总览: 三层递降管线

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer 0: Python 预处理 (torch-mlir fx.py)                         │
│   torch.export.export() → run_decompositions()                    │
│   → FxImporter.import_frozen_program()                            │
│   产出: Torch Dialect MLIR Module                                  │
├──────────────────────────────────────────────────────────────────┤
│ Layer 1: TorchFX IR → Torch Backend IR (4-7 Pass)                 │
│   pipeline: "torchdynamo-export-to-torch-backend-pipeline"        │
│   产出: 满足 Backend Contract 的 Torch IR                          │
├──────────────────────────────────────────────────────────────────┤
│ Layer 2: Torch Backend IR → Linalg-on-Tensors IR (18 Pass)        │
│   pipeline: "torch-backend-to-linalg-on-tensors-backend-pipeline"  │
│   产出: 纯 Linalg/Tensor/Arith/SCF IR (无 Torch 痕迹)              │
├──────────────────────────────────────────────────────────────────┤
│ Layer 3: Linalg → GPU 可执行代码 (上游 MLIR Pass)                   │
│   LinalgFuseElementwise → ConvertLinalgToGPU → GPUToNVVM          │
│   → NVVMToLLVM → LLVM → PTX                                       │
│   产出: PTX / cubin                                               │
└──────────────────────────────────────────────────────────────────┘
```

**源码位置**:
- Layer 0: `python/torch_mlir/fx.py` (`export_and_import` → `_module_lowering`)
- Layer 1: `lib/Dialect/Torch/Transforms/Passes.cpp` (`createTorchDynamoExportToTorchBackendPipeline`)
- Layer 2: `lib/Dialect/TorchConversion/Transforms/Passes.cpp` (`createTorchBackendToLinalgOnTensorsBackendPipeline`)
- Layer 3: 上游 `llvm-project/mlir/lib/Dialect/Linalg/`、`lib/Conversion/GPUToNVVM/` 等

---

## 2. Layer 0: FX Graph → MLIR Torch Dialect

`stateless_fx_import(gm, output_type=...)` 是 `torch.compile` 自定义 backend 中调用的 torch-mlir 入口。它内部做了三件事:

### 2.1 `FxImporter.import_stateless_graph(gm.graph)`

**做什么**: 将 Dynamo 捕获的 `torch.fx.GraphModule` 逐节点转换为 MLIR Torch Dialect IR。这是 1:1 的语法翻译，不涉及优化。

```
FX Graph Node                →  Torch Dialect Op
  aten.add.Tensor(x, y)      →  torch.aten.add.Tensor %x, %y
  aten.linear(x, w, b)       →  torch.aten.linear %x, %w, %b
```

**与 Dynamo 的关系**: Dynamo 的 frame capture 已经产出了 FX Graph，所有 tensor 的 shape/dtype 通过 fake tensor 在 trace 时确定。`import_stateless_graph` 只是把这个已经"准备好"的图翻译为 MLIR 格式——不需要重新推导任何信息。

### 2.2 `_module_lowering(module, output_type, ...)`

**做什么**: 根据 `output_type` 决定运行哪些 MLIR Pass:

| output_type | 行为 |
|-------------|------|
| `RAW` | 直接返回 MLIR Module，不运行任何 Pass |
| `TORCH` | 只运行 Layer 1 管线（TorchFX→Torch Backend IR） |
| `LINALG_ON_TENSORS` | 运行 Layer 1 + Layer 2 全部管线 |
| `TOSA` | 运行 Layer 1 + TOSA 后端管线 |
| `STABLEHLO` | 运行 Layer 1 + StableHLO 后端管线 |

**`output_type=LINALG_ON_TENSORS` 是最常见的 GPU 目标**——将 IR 降到 Linalg dialect 后，上游 MLIR 社区 Pass 接手后续到 GPU 的 lowering。

### 2.3 与 `torch.export` 路径的区别

torch-mlir 同时提供了 `export_and_import(f, args)` 入口（内部调用 `torch.export`）。两种方式最终都到达同样的 MLIR Pass 管线:

```
方式 1: torch.compile 自定义 backend
  Dynamo frame capture → FX Graph (gm) → stateless_fx_import(gm)

方式 2: torch.export 独立使用
  torch.export.export(model, args) → ExportedProgram
  → export_and_import(f) → 内部调用 import_frozen_program → _module_lowering

区别: 方式 1 的 gm 来自 Dynamo 的在线 frame capture
       方式 2 的 prog 来自 torch.export 的离线导出
共同点: 最终都执行相同的 MLIR Pass 管线 (Layer 1 + Layer 2)
```

---

## 3. Layer 1: TorchFX IR → Torch Backend IR

**Pipeline 字符串** (源码 `fx.py` `_module_lowering`):
```
builtin.module(
    func.func(torch-match-quantized-custom-ops),
    torchdynamo-export-to-torch-backend-pipeline{backend-legal-ops=..., extra-library=...}
)
```

**C++ 源码** (`Passes.cpp` `createTorchDynamoExportToTorchBackendPipeline`):
```cpp
pm.addPass(createInlinerPass());
pm.addNestedPass<func::FuncOp>(createReduceOpVariantsPass(options.extraLibrary));
pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
if (options.decompose) {
    pm.addNestedPass<func::FuncOp>(Torch::createDecomposeComplexOpsPass(options.backendLegalOps));
    pm.addNestedPass<func::FuncOp>(Torch::createRecomposeComplexOpsPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
}
```

**默认行为** (`decompose` 默认为 `false`): Python 端 `run_decompositions()` 已完成主力分解，C++ 端不重复执行。

---

### Pass 1: `torch-match-quantized-custom-ops`

| 属性 | 内容 |
|------|------|
| **做什么** | 将 `torch.ops.quantized.*` 等自定义量化 op 重写为标准 `aten.*` 量化操作 |
| **作用域** | `func::FuncOp` |
| **生效条件** | 仅量化模型。非量化模型空跑 |

**解决什么问题**: PyTorch 量化工具使用自定义 op 名称（如 `quantized::linear`），这些在标准 ATen dialect 中不存在。必须先标准化，后续 Pass 才能处理。

---

### Pass 2: `Inliner`

| 属性 | 内容 |
|------|------|
| **做什么** | 将 `func.call` 操作内联到调用点 |
| **作用域** | Module (全局) |

**解决什么问题**: `torch.export` 遇到 higher-order ops (`torch.while_loop`、`torch.cond`) 时，将循环体/条件体导出为独立 `func.func` 并通过 `func.call` 调用。内联消除函数边界，让后续 Pass 看到完整计算图。

注释: "Inline func.call operations created by higher-order ops like while_loop to conform to the linalg-on-tensors backend contract."

**为什么这样设计**:
- **最小化内联范围**: 只处理 higher-order ops 产生的间接调用。Dynamo 导出的主计算图已是平坦的，不需要大规模内联
- **MLIR pattern rewrite 约束**: 多数 MLIR pattern 工作在单函数范围内，`func.call` 阻断了分析

---

### Pass 3: `ReduceOpVariants`

| 属性 | 内容 |
|------|------|
| **做什么** | 将 PyTorch op 的多种"变体"统一为单一规范化形式 |
| **作用域** | `func::FuncOp` |

**四种规约**:

| 规约 | 示例 | 理由 |
|------|------|------|
| Non-value → value semantics | non-value `aten.add` → value `aten.add` + `CopyToNonValueTensorOp` | 消除 mutable tensor 类型 |
| In-place → out-of-place | `aten.add_` → `aten.add` + `OverwriteTensorContentsOp` | 计算与修改分离 |
| 专用后端 → 通用 ATen | `_scaled_dot_product_flash_attention_for_cpu` → 通用 attention | 硬件 dispatch 不是 IR 概念 |
| NonValueTensorLiteral → ValueTensorLiteral | 常量字面量形式统一 | 减少类型变体 |

**为什么这样设计**: "every operation becomes a pure value-semantic computation, and any necessary in-place update is made explicit via OverwriteTensorContentsOp"——这是编译器经典设计: 将副作用外化，使核心计算变成纯函数。

---

### Pass 4: `Canonicalizer` (第 1 次)

| 属性 | 内容 |
|------|------|
| **做什么** | MLIR 内置通用规范化: 常量折叠、代数简化 (`x+0→x`)、死代码消除 |
| **作用域** | `func::FuncOp` |

**解决什么问题**: ReduceOpVariants 产生的 `CopyToValueTensorOp → CopyToNonValueTensorOp` 对等碎屑。

---

### Pass 5-7 (可选): Decompose → Recompose → Canonicalizer

仅在用户显式指定 `decompose=true` 时执行。Python 端已完成主力分解，这三个 Pass 作为**兜底**:

| Pass | 做什么 |
|------|--------|
| `DecomposeComplexOps` | 100+ 种 ATen op → 基本 op。由 `backendLegalOps` 控制哪些不分解 |
| `RecomposeComplexOps` | 反向重组: `split+unpack → slice`、`copy_ on slice → index_put` |
| `Canonicalizer` | 清理分解碎屑 |

---

## 4. Layer 2: Torch Backend IR → Linalg-on-Tensors IR

**Pipeline 字符串** (源码 `compiler_utils.py` `lower_mlir_module`):
```
builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline{allow-non-finites=true})
```

**C++ 源码**: `lib/Dialect/TorchConversion/Transforms/Passes.cpp`

这是 **18 个 Pass** 的管线，分四个阶段。

---

### Phase A: 前置准备 (Pass 1-4)

---

#### Pass 1: `RestructureNonConstantAxes`

| 属性 | 内容 |
|------|------|
| **做什么** | 确保所有 reduction op 的 axis 参数是编译期常量。如果 axis 是动态值，通过 reshape 将动态维度转为静态 |
| **作用域** | `func::FuncOp` |

**解决什么问题**: Linalg dialect 要求 reduction dimension 是静态已知的。`sum(x, dim=k)` 中 `k` 若是变量，无法生成合法 `linalg.generic`。

**为什么这样设计**: 将"动态 axis"问题转换为"动态 shape"问题——通过 reshape 将动态 axis 维度变为已知。下游 shape 推断能处理动态 shape，但处理不了动态 axis。

---

#### Pass 2: `FuseQuantizedOps`

| 属性 | 内容 |
|------|------|
| **做什么** | 融合 `dequantize → compute(float) → quantize` 为纯整数计算 |
| **作用域** | `func::FuncOp` |

**为什么在 Torch→Linalg 之前**: 一旦转为 Linalg dialect，量化语义丢失，无法做量化融合。必须在此之前。

---

#### Pass 3: `ConvertTorchToTMTensor`

| 属性 | 内容 |
|------|------|
| **做什么** | 将部分 Torch op lowering 到 TMTensor (Tensor Metadata Tensor) dialect |
| **作用域** | `func::FuncOp` |

**解决什么问题**: 某些 Torch op 涉及 tensor 元数据操作，需要先转换为带守卫的 TMTensor 表示。注释: "pattern-matching against constants, e.g. dimensions which must be constant in a ranked programming model"——处理 ranked model 中必须为常量的维度。

---

#### Pass 4: `Canonicalizer` (第 2 次)

清理前置准备的碎屑。

---

### Phase B: 主力 Dialect 转换 — 五个 `ConvertTorchToXxx` (Pass 5-10)

这是管线的核心——将 Torch dialect **逐方言分解**为 MLIR 标准 dialect。

---

#### Pass 5: `ConvertTorchToLinalg` ★ 核心

| 属性 | 内容 |
|------|------|
| **做什么** | 将 Torch dialect 的线性代数/计算 op 转换为 Linalg dialect |
| **作用域** | `func::FuncOp` |
| **源码** | `lib/Conversion/TorchToLinalg/TorchToLinalg.cpp` |

**九组 Pattern (按注册顺序)**:

| 序号 | Pattern 组 | 典型转换 | 源文件 |
|------|-----------|---------|--------|
| 1 | TensorScalarInterop | `aten.add.Scalar` → `linalg.generic { arith.addf }` | `TensorScalarInterop.cpp` |
| 2 | Linear | `aten.linear`, `aten.matmul` → `linalg.matmul` / `linalg.batch_reduce_matmul` | `Linear.cpp` |
| 3 | Pooling | `aten.max_pool2d` → `linalg.pooling_*` | `Pooling.cpp` |
| 4 | Random | `aten.rand`, `aten.randn` → RNG op + tensor generate | `Random.cpp` |
| 5 | Uncategorized | 未归类的 op (activation, normalization 等) | `Uncategorized.cpp` |
| 6 | Reduction | `aten.sum`, `aten.mean` → `linalg.generic` with reduction iterator | `Reduction.cpp` |
| 7 | DataMovement | `aten.permute`, `aten.transpose` → `tensor.transpose` | `DataMovement.cpp` |
| 8 | IndirectDataMovement | `aten.index`, `aten.index_put` → scf loops + tensor insert/extract | `IndirectDataMovement.cpp` |
| 9 | TensorConstructors | `aten.zeros`, `aten.ones` → `tensor.generate` / `linalg.fill` | `TensorConstructors.cpp` |

**为什么这样设计**:

- **Partial Conversion**: 使用 `ConversionTarget` + `applyPartialConversion`——只转换"准备好"的 op，无法转换的保留给后续 Pass（如 ConvertTorchToSCF 处理控制流）
- **逐组分离**: 九组 pattern 在九个独立文件中，每组一个 `populate*PatternsAndLegality` 函数。新增 op 支持只需修改对应文件
- **`allowNonFinites` 选项**: 控制 Pooling/Reduction 组是否允许 Inf/NaN 优化

---

#### Pass 6: `Canonicalizer` (第 3 次)

清理 dialect 转换碎屑。Linalg op 创建后，相邻的 `tensor.reshape` + `linalg.generic` 可能被规范化融合。

---

#### Pass 7: `ConvertTorchToSCF`

| 属性 | 内容 |
|------|------|
| **做什么** | Torch 控制流 (`prim.IfOp`, `prim.LoopOp`) → SCF dialect (`scf.if`, `scf.for`) |
| **作用域** | `func::FuncOp` |

**为什么在 Linalg 之后**: 控制流内部的计算已被转为 Linalg。此时只转控制流壳层，壳内已是标准 dialect。

---

#### Pass 8: `ConvertTorchToArith`

| 属性 | 内容 |
|------|------|
| **做什么** | 残存的 Torch 标量运算 → Arith dialect |
| **作用域** | `func::FuncOp` |

**解决什么问题**: Linalg 转换后可能残存纯标量 Torch op (`aten.add.int`、`aten.mul.float`)。这些不需要 Linalg，直接转 `arith.addi`、`arith.mulf`。

---

#### Pass 9: `ConvertTorchToTensor`

| 属性 | 内容 |
|------|------|
| **做什么** | 最后的 Torch tensor 操作 → MLIR Tensor dialect |
| **作用域** | `func::FuncOp` |

**解决什么问题**: 前几个 Pass 可能遗漏 Torch 的 tensor 元操作（如 `aten.size` 的某些使用模式）。这是消除 Torch dialect 的最后一轮。

---

#### Pass 10: `ConvertTorchConversionToMLProgram`

| 属性 | 内容 |
|------|------|
| **做什么** | 清除 TorchConversion 辅助 dialect → MLProgram dialect |
| **作用域** | Module (全局) |

**解决什么问题**: TorchConversion 是 torch-mlir 内部辅助 dialect。转换完成后，辅助 op 本身也需要被消除。

---

### Phase C: MemRef 清理 (Pass 11-14)

---

#### Pass 11: `memref::ExpandOps`

| 属性 | 内容 |
|------|------|
| **做什么** | 展开 memref 复合操作 (`memref.subview`、`memref.realloc`) 为基本操作 |
| **作用域** | `func::FuncOp` |

---

#### Pass 12: `Canonicalizer` (第 4 次)

---

#### Pass 13: `memref::ResolveShapedTypeResultDims`

| 属性 | 内容 |
|------|------|
| **做什么** | 将 `tensor.dim` / `memref.dim` 的结果替换为静态已知的常量 |
| **作用域** | `func::FuncOp` |

**解决什么问题**: shape 查询 op (`tensor.dim %t, %c0`) 在编译时答案已知（%t 的 shape 已知），但 IR 中仍表示为动态查询。此 Pass 将其替换为常量值。

---

#### Pass 14: `CSE` (Common Subexpression Elimination)

| 属性 | 内容 |
|------|------|
| **做什么** | 消除重复的公共子表达式 |
| **作用域** | `func::FuncOp` |

注释: "tends to create identical ops. CSE them."

---

### Phase D: 类型转换 + 合约验证 (Pass 15-18)

---

#### Pass 15: `FuncBackendTypeConversion`

| 属性 | 内容 |
|------|------|
| **做什么** | 函数签名中的 Torch 类型 (`!torch.vtensor`) → 后端类型 (`tensor<...>`) |
| **作用域** | Module (全局) |

**为什么是 Module Pass**: 函数签名跨函数一致性问题——需要全局视角确保调用者和被调用者的签名同时更新。

注释: "Finish the type conversion from torch types to the types of the linalg-on-tensors backend contract."

---

#### Pass 16: `Canonicalizer` (第 5 次)

清理类型转换产生的 `UnrealizedConversionCastOp` 碎屑。

---

#### Pass 17: `FinalizingBackendTypeConversion`

| 属性 | 内容 |
|------|------|
| **做什么** | 最终化 Partial Conversion: 消除所有残留 `UnrealizedConversionCastOp` |
| **作用域** | `func::FuncOp` |

---

#### Pass 18: `VerifyLinalgOnTensorsBackendContract`

| 属性 | 内容 |
|------|------|
| **做什么** | 断言 IR 完全满足 Linalg-on-Tensors 后端合约 |
| **作用域** | Module (全局) |
| **失败行为** | `signalPassFailure()` → 编译失败，输出诊断 |

**检查项**: 无 Torch dialect 残留、无 Torch 类型残留、所有 op 属于 Linalg/Tensor/Arith/SCF/Math/Func/MemRef、所有 tensor 有静态 rank。

---

## 5. Layer 3: Linalg → GPU 代码生成 (上游 MLIR Pass)

Layer 2 产出的是纯 Linalg-on-Tensors IR。后续 GPU lowering 由上游 MLIR 社区 Pass 完成（不属于 torch-mlir 范围，但本文简要列出）:

| 序号 | Pass | 作用 |
|------|------|------|
| 1 | `LinalgFuseElementwiseOps` | 融合相邻 elementwise `linalg.generic` |
| 2 | `LinalgTilingPass` | Tile Linalg op 以适应 GPU thread block |
| 3 | `ConvertLinalgToParallelLoops` | Linalg → `scf.parallel` |
| 4 | `MapParallelLoopsToGPU` | `scf.parallel` → `gpu.launch` (block/thread 映射) |
| 5 | `GpuKernelOutliningPass` | 提取 `gpu.func` kernel 函数 |
| 6 | `ConvertLinalgToGPU` | 残存 Linalg op → GPU dialect |
| 7 | `ConvertGPUToNVVM` | GPU dialect → NVVM dialect |
| 8 | `ConvertNVVMToLLVM` | NVVM → LLVM dialect |
| 9 | LLVM → PTX | LLVM 标准后端生成 PTX |
| 10 | PTX → cubin | NVIDIA PTX assembler |

---

## 6. 完整 Pass 总结表

| Layer | # | Pass | 作用域 | 核心作用 | 可跳过? |
|-------|---|------|--------|---------|--------|
| **0** | P1 | `torch.export.export()` | Python | Dynamo tracing + fake tensor | ❌ |
| **0** | P2 | `run_decompositions()` | Python | 50+ ATen op 分解 | ❌ |
| **0** | P3 | `import_frozen_program()` | Python | FX Graph → Torch MLIR | ❌ |
| **1** | 1 | `torch-match-quantized-custom-ops` | FuncOp | 量化 op 名称标准化 | 仅量化模型生效 |
| **1** | 2 | `Inliner` | Module | 内联 higher-order op 的函数调用 | ❌ |
| **1** | 3 | `ReduceOpVariants` | FuncOp | 规约 op 变体 | ❌ |
| **1** | 4 | `Canonicalizer` | FuncOp | 清理碎屑 | ❌ |
| **1** | 5* | `DecomposeComplexOps` | FuncOp | 兜底分解 | ✅ (默认跳过) |
| **1** | 6* | `RecomposeComplexOps` | FuncOp | 重组结构性拆分 | ✅ (默认跳过) |
| **1** | 7* | `Canonicalizer` | FuncOp | 清理 | ✅ (默认跳过) |
| **2** | 8 | `RestructureNonConstantAxes` | FuncOp | 动态 axis → static | ❌ |
| **2** | 9 | `FuseQuantizedOps` | FuncOp | 融合量化计算链 | ❌ |
| **2** | 10 | `ConvertTorchToTMTensor` | FuncOp | Torch → TMTensor | ❌ |
| **2** | 11 | `Canonicalizer` | FuncOp | 清理 | ❌ |
| **2** | 12 | `ConvertTorchToLinalg` | FuncOp | **主力: Torch → Linalg (9 组)** | ❌ |
| **2** | 13 | `Canonicalizer` | FuncOp | 清理 dialect 转换碎屑 | ❌ |
| **2** | 14 | `ConvertTorchToSCF` | FuncOp | Torch 控制流 → SCF | ❌ |
| **2** | 15 | `ConvertTorchToArith` | FuncOp | Torch 标量 → Arith | ❌ |
| **2** | 16 | `ConvertTorchToTensor` | FuncOp | 残存 Torch tensor → Tensor | ❌ |
| **2** | 17 | `ConvertTorchConversionToMLProgram` | Module | 清除辅助 dialect | ❌ |
| **2** | 18 | `memref::ExpandOps` | FuncOp | 展开 memref 复合操作 | ❌ |
| **2** | 19 | `Canonicalizer` | FuncOp | 清理 | ❌ |
| **2** | 20 | `memref::ResolveShapedTypeResultDims` | FuncOp | shape 查询 → static 常量 | ❌ |
| **2** | 21 | `CSE` | FuncOp | 公共子表达式消除 | ❌ |
| **2** | 22 | `FuncBackendTypeConversion` | Module | 函数签名类型转换 | ❌ |
| **2** | 23 | `Canonicalizer` | FuncOp | 清理类型转换碎屑 | ❌ |
| **2** | 24 | `FinalizingBackendTypeConversion` | FuncOp | 最终化 Partial Conversion | ❌ |
| **2** | 25 | `VerifyLinalgOnTensorsBackendContract` | Module | 断言 IR 满足合约 | ❌ |

**总计**: Python 3 步 + Layer 1 中 4 核心 Pass + Layer 2 中 18 Pass = **22 个 Pass 实际执行**（Canonicalizer 共运行 5 次）。

---

## 7. 核心设计模式

### 7.1 渐进式 Dialect 消除

```
Layer 1 消除: non-value tensor 类型、op 变体、量化自定义 op
Layer 2 消除: torch 类型 → 后端类型
              torch op → Linalg / SCF / Arith / Tensor / TMTensor / MLProgram
              辅助 dialect → 清除
              memref 嵌套 → 展开 + 解析
```

最终 `VerifyLinalgOnTensorsBackendContract` 断言零残留。这是经典的"逐层消元"策略。

### 7.2 前后端分解分离

```
Python 端: run_decompositions()       ← 主力 (50+ op)
C++ 端:   DecomposeComplexOps         ← 兜底 (默认跳过)
```

"前端承担编译责任"的哲学——能做在 Python 端的就不做在 C++ 端。

### 7.3 Canonicalizer 五重散布

| 次数 | 位置 | 消除的碎屑类型 |
|------|------|-------------|
| 1 | Layer 1 Pass 4 | ReduceOpVariants 的 CopyToValue/CopyToNonValue 对 |
| 2 | Layer 2 Pass 4 | 前置准备的冗余 op |
| 3 | Layer 2 Pass 6 | Torch→Linalg 未优化的中间形式 |
| 4 | Layer 2 Pass 12 | memref 操作展开后的冗余 |
| 5 | Layer 2 Pass 16 | 类型转换的 UnrealizedConversionCastOp |

### 7.4 显式合约验证

`satisfiesBackendContract()` + `VerifyLinalgOnTensorsBackendContract` 构成双重断言:
- Layer 1 级别: 确保 Torch IR 满足后端合约（无 unranked tensor、无动态 dtype、无非值语义 tensor）
- Layer 2 级别: 确保 Linalg IR 完全消除 Torch 痕迹

显式合约优于隐式假设——任何未转换的残留都会在这里被捕获。

---

## 8. 三条路径对比总结

| 维度 | 路径 A: torch.compile GPU 默认 | 路径 B: torch.compile + torch-mlir backend (本文) | 路径 C: torch.compile NPU |
|------|--------------------------|--------------------------|--------------------------|
| **入口 API** | `torch.compile(model)` | `torch.compile(model, backend=mlir_backend)` | `torch.compile(model)` |
| **走 MLIR?** | ❌ 不走 | ✅ 走 (通过自定义 backend) | ✅ 走 (monkey-patch 方式) |
| **图捕获** | Dynamo frame capture | Dynamo frame capture | Dynamo frame capture |
| **torch-mlir 入口** | N/A | `stateless_fx_import(gm)` | FxImporter (内嵌在 codegen 中) |
| **IR 路径** | FX → Inductor IR → Triton IR | FX → MLIR Torch → Linalg → GPU | FX → Inductor IR(含 TracedGraph) → MLIR |
| **MLIR Pass 数** | 0 | 22 | 35+ (含 毕昇) |
| **GPU 代码生成** | Triton JIT → PTX | MLIR Linalg → GPU → NVVM → PTX | N/A (NPU 硬件) |
| **主要维护方** | PyTorch 官方 | torch-mlir 社区 (llvm-project) | 华为昇腾 |

---

## Related Pages

- [[10_mlir_core_concepts]] — MLIR 核心概念: Dialect、Pass、IR 注册、递降原理
- [[30_triton_vs_mlir_backend_analysis]] — Triton vs Torch-MLIR: 六阶段概念对等映射
- [[npu_mlir_backend_technical_analysis]] — NPU MLIR 管线分析 (毕昇编译器路线，非 GPU 路线)
- [[02_compile_stack/04_inductor/index]] — torch.compile 端到端流水线
- [[02_compile_stack/01_dynamo/index]] — Dynamo 帧评估 API、字节码符号执行
