# PyTorch Inductor Codegen 中 Dynamic Shape 的处理机制

## 1. 概述

在 PyTorch 2.0 的编译生态中，**Dynamic Shape**（动态形状）是指 tensor 的某些维度大小在编译时无法确定为具体整数值，而是以符号（`sympy.Symbol` / `SymInt`）表示。Inductor 的 codegen 环节需要将这些符号表达式**正确地传递到生成的可执行代码中**，并在运行时动态求值。

本报告分析 dynamic shape 在 Inductor codegen 中的核心处理机制，涵盖：
1. 符号如何从 IR 流入 kernel 和 wrapper
2. Triton kernel 对动态 shape 的支持方式
3. Grid 的动态计算
4. Unbacked symbols 的声明与解析
5. 内存规划中的动态 shape
6. C++ wrapper 中的特殊处理

---

## 2. 符号传递：从 IR 到 Codegen

### 2.1 SizeArg：符号的载体

在 `torch/_inductor/codegen/common.py:291` 中，`SizeArg` 是 codegen 中承载符号表达式的基础数据结构：

```python
@dataclasses.dataclass
class SizeArg:
    name: str
    expr: sympy.Expr
```

当一个 tensor 的 size/stride/numel 包含动态维度时，这些表达式会被收集为 `SizeArg`，并作为 kernel 的调用参数传递。

### 2.2 Triton Kernel 中的动态 numel

`TritonKernel.codegen_kernel()`（`triton.py:5258`）在构建 kernel 签名时，会将每个 range tree 的 `numel` 追加为 `SizeArg`：

```python
for tree in self.active_range_trees():
    sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
    signature.append(sizearg)
    argdefs.append(ArgName(sizearg.name))
```

这确保了 kernel 在运行时可以接收 `xnumel`、`ynumel`、`rnumel` 等动态值。

### 2.3 Wrapper 中的 numel 表达式计算

对于复杂的 `numel` 表达式（例如 `s0 * 64`），`TritonKernel.add_numel_to_call_args()`（`triton.py:5667`）会调用 wrapper 的 `generate_numel_expr`：

```python
def add_numel_to_call_args(self, name, call_args, arg_types):
    for tree in self.range_trees:
        if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)):
            expr = tree.numel
        else:
            expr = V.graph.wrapper_code.generate_numel_expr(name, tree)
        if not tree.is_reduction or self.inside_reduction:
            call_args.append(expr)
            arg_types.append(type(expr))
```

`generate_numel_expr`（`wrapper.py:2709`）会生成一个中间变量赋值语句，如：

```python
_triton_poi_0_xnumel = s0 * 64
```

然后通过 `SymbolicCallArg` 将其传递给 kernel 调用。这避免了在 kernel 调用处直接塞入复杂表达式，保持代码清晰。

### 2.4 符号实参的整型：`s0→ks0` 重命名与 `ks0*ks1` 防溢出（升 `tl.int64`）

kernel 内符号实参由 `rename_indexing`（`common.py`）翻译（`s0→ks0`，按发现顺序）；`signature_to_meta._decide_tl_dtype`（`triton_utils.py`）对**动态 size 实参刻意升 `tl.int64`**，避免多维动态下 `ks0*ks1` 乘积超 int32 上限而溢出：

```python
if (not config.triton.use_block_ptr and isinstance(arg, SizeArg) and arg.name.startswith("ks")):
    return "tl.int64"        # 动态 size 实参用 i64，防 ks0*ks1 溢出
```

> [!contradiction] 与 NPU Linearize 后端相反 — 见 [[npu_inductor_linearize_backend_analysis]]
> 昇腾 vector core 不支持 i64 计算，实验性 Linearize 后端反而把 `*i64→*i32`、`_triton_type_mapping["tl.int64"]="tl.int32"`，并在 launcher 运行期 downcast——**上游用 i64 规避的 `ks0*ks1` 溢出，NPU 用 i32 重新引入了**（大张量索引溢出风险）。这是 GPU↔NPU 在动态 shape 整型处理上的根本分歧，另见 [[inductor_autotuning_analysis]] §三。

---

## 3. Triton Grid 的动态计算

Triton kernel 的 launch grid 不能是编译时常量，必须与输入 shape 动态关联。Inductor 通过 **`GridExpr`** 体系实现这一点。

### 3.1 GridExpr 架构

`torch/_inductor/runtime/triton_heuristics.py:3807`

```python
class GridExpr:
    def generate(self, meta: dict[str, int]) -> None:
        raise NotImplementedError

    def ceildiv(self, numel: str | int, block: int | str | None) -> str | int:
        # 生成向上取整表达式，支持常数折叠
```

具体子类根据 kernel 的维度数生成 grid 表达式：

- `Grid1D`: `x_grid = ceil(xnumel / XBLOCK)`
- `Grid2D`: `x_grid = ceil(xnumel / XBLOCK)`, `y_grid = ceil(ynumel / YBLOCK)`
- `Grid2DWithYZOverflow`: 处理 ydim 超出 CUDA limit 的情况
- `Grid3D` / `BatchMatmulGrid3D`: 3D grid

### 3.2 Grid 代码的生成位置

`TritonKernel._get_grid_type()`（`triton.py:5647`）根据 `range_trees` 的数量和特性选择合适的 `GridExpr` 类，并将其名称写入 `inductor_meta["grid_type"]`。

真正的 grid 计算代码是在 **Triton heuristic launcher** 中生成的，而非直接写在 Inductor 的 wrapper 中。kernel 调用时通过 `meta` 字典（包含 `XBLOCK` 等配置参数）和 grid 函数动态求值：

```python
# 生成的 Triton heuristic 包装器中的伪代码
def grid(meta):
    return (
        -((xnumel) // -(meta['XBLOCK'])),
        -((ynumel) // -(meta['YBLOCK'])),
        1
    )

triton_kernel.run(..., grid=grid, stream=stream)
```

---

## 4. Unbacked Symbols：运行时才能确定的符号

### 4.1 什么是 Unbacked Symbols

某些符号（通常以 `u0`, `u1` 命名）**没有任何编译时约束**，完全依赖于运行时 tensor 的实际 shape（例如 `nonzero` 的输出大小）。这些符号在 codegen 中需要特殊处理。

### 4.2 Wrapper 中的声明与解析

`PythonWrapperCodegen` 维护 `unbacked_symbol_decls: OrderedSet[str]`（`wrapper.py:1117`），避免重复声明。

当 codegen 遇到需要从输出 tensor 的 shape/stride 中提取 unbacked symbol 时，调用：

```python
codegen_unbacked_symbol_defs_for_outputs(
    output_name, outputs, unbacked_bindings
)
```

该方法（`wrapper.py:3436`）通过 `pytree.KeyPath` 解析输出结构，生成类似如下的代码：

```python
u0 = outs[0].size(1)
u1 = outs[0].stride(0)
```

对于 C++ wrapper，同样有对应的 keypath 解析逻辑，生成 `outs[0].sizes()[1]` 等 C++ 表达式。

---

## 5. 运行时 Shape 断言：保证编译假设成立

### 5.1 assert_size_stride

在 `PythonWrapperCodegen.write_prefix()` 中，会调用 `codegen_input_size_asserts()`（`wrapper.py:1407`），为每个图输入生成运行时断言：

```python
assert_size_stride(arg0_1, (s0, 64), (64, 1))
```

这里的 `s0` 是输入 tensor 的实际大小。如果运行时 shape 与编译时的符号约束不一致，此断言会立即失败，防止隐藏 bug。

### 5.2 符号Guard与32-bit Indexing决策

`SIMDScheduling.can_use_32bit_indexing()`（`simd.py:1826`）检查 `numel` 和 buffer sizes 是否能确定在 `int32` 范围内：

```python
if not expr_fits_within_32bit(numel):
    return False
V.graph.sizevars.check_leq(numel, int_max)
```

如果不能静态保证，则 codegen 会回退到 64-bit indexing（`tl.int64`），避免溢出。这是一个典型的编译时/运行时协同决策。

---

## 6. 内存规划中的 Dynamic Shape

### 6.1 符号化的 buffer 复用

`memory_planning.py` 中的 `buffer_reuse_key`（`wrapper.py:100`）使用 `sympy_str` 而非具体数值来比较 buffer 大小：

```python
def buffer_reuse_key(node: BufferLike) -> ReuseKey:
    return (
        node.get_device_or_error(),
        node.get_dtype(),
        sympy_str(V.graph.sizevars.simplify(storage_size)),
        alignment,
    )
```

这确保了 **只有当两个 buffer 的符号大小完全相同时**（例如都是 `s0 * s1`），才允许复用。如果 size hint 相同但符号不同，不会错误复用。

### 6.2 动态 size 的分配代码

`AllocateLine`（`wrapper.py:748`）在 codegen 时生成实际的分配代码。对于动态大小，`make_allocation` 会输出包含符号表达式的 `empty_strided` 调用：

```python
buf0 = empty_strided_cuda((s0, 64), (64, 1), torch.float32)
```

Python 解释器在运行时求值 `s0`，因此动态分配是透明的。

### 6.3 Size Hint 与峰值估计

`EfficientPeakEstimate`（`memory_planning.py:707`）使用 `V.graph.sizevars.size_hint(..., fallback=0)` 来估计动态 buffer 的内存占用，用于峰值内存分析和分区决策。当符号无法推断时，fallback 为 0，可能低估峰值，但避免了过度保守的分配策略。

---

## 7. C++ Wrapper 中的 Dynamic Shape

当 `cpp_wrapper=True`（AOTInductor 模式）时，dynamic shape 的处理逻辑基本保持一致，但输出语言变为 C++：

- `SymbolicCallArgLine` 在 C++ 模式下生成 `uint32_t` 类型的中间变量。
- `GridExpr` 支持 `mode="cpp"`，生成 C++ 语法的 `ceildiv`（`(numel + XBLOCK - 1) / XBLOCK`）。
- `codegen_unbacked_symbol_defs_for_outputs` 针对 C++ 生成 `std::get<idx>(output)` 和 `.sizes()[dim]` 访问代码。
- 输入 shape 断言通过 C++ shim 调用 `aoti_torch_check_tensor` 等实现。

---

## 8. 关键源码索引

| 功能 | 文件 | 关键代码位置 |
|------|------|--------------|
| SizeArg 定义 | `codegen/common.py` | `class SizeArg` (L291) |
| Triton kernel 动态 numel | `codegen/triton.py` | `codegen_kernel()` (L5258), `add_numel_to_call_args()` (L5667) |
| Wrapper numel 表达式生成 | `codegen/wrapper.py` | `generate_numel_expr()` (L2709) |
| Symbolic 调用参数 | `codegen/wrapper.py` | `SymbolicCallArg` (L399), `SymbolicCallArgLine` (L1043) |
| Grid 动态计算 | `runtime/triton_heuristics.py` | `GridExpr` (L3807), `Grid1D/Grid2D/Grid3D` |
| Grid 类型选择 | `codegen/triton.py` | `_get_grid_type()` (L5647) |
| Unbacked symbols 声明 | `codegen/wrapper.py` | `codegen_unbacked_symbol_defs_for_outputs()` (L3436) |
| 输入 shape 断言 | `codegen/wrapper.py` | `codegen_input_size_asserts()` (L1407) |
| 32-bit indexing 决策 | `codegen/simd.py` | `can_use_32bit_indexing()` (L1826) |
| Buffer 复用 key | `codegen/wrapper.py` | `buffer_reuse_key()` (L100) |
| 内存分配线 | `codegen/wrapper.py` | `AllocateLine` (L748) |
| 峰值内存估计 | `codegen/memory_planning.py` | `EfficientPeakEstimate` (L707) |

---

## 9. XBLOCK 选择机制与 Dynamic Shape 的性能代价

> 本节补充说明 Triton kernel 的 tiling 参数（XBLOCK）如何与 dynamic shape 交互，以及由此带来的性能代价。

### 9.1 XBLOCK 候选值范围

```python
TRITON_MAX_BLOCK = {
    "X": 4096,   # x 方向最大 block size
    "R": 4096,   # reduction 方向最大
    ...
}
```

XBLOCK 候选值是 **2 的幂次序列**，范围 `[32, TRITON_MAX_BLOCK['X']]`。`pointwise()` decorator 的起点由 `size_hints` 决定：

```python
max_xblock = min(next_power_of_2(size_hints[0]), TRITON_MAX_BLOCK['X'])
# 然后生成候选: [32, 64, ..., max_xblock]
```

不同 `size_hints` 对应的候选集：

| size_hints | max_xblock | 候选 XBLOCK |
|-----------|-----------|------------|
| `[384]`   | 512       | 32, 64, 128, 256, 512 |
| `[16384]` | 4096      | 256, 512, 1024, 2048, 4096 |
| `[100]`   | 128       | 16, 32, 64, 128 |

### 9.2 三种 XBLOCK 决策模式

**模式 1：`triton.heuristics`（默认 pointwise）**

```python
@triton.heuristics(
    values={'XBLOCK': lambda meta: min(next_power_of_2(meta['xnumel']),
                                       TRITON_MAX_BLOCK['X'])}
)
@triton.jit
def triton_poi_0(..., xnumel, XBLOCK: tl.constexpr):
    ...
```

XBLOCK 在 **kernel launch 时**由 Python lambda 根据 `xnumel` 的实际运行时值计算。Triton JIT 为每个不同的 XBLOCK 值编译并缓存一个 `.ptx`。动态 shape 下 XBLOCK 自适应实际 xnumel，但每种新 XBLOCK 值第一次遇到需 JIT compile（约 0.1–0.5s）。

**模式 2：`triton.autotune`（`max_autotune` 模式，需显式开启）**

```python
@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 512}, num_warps=4),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 2048}, num_warps=8),
    ],
    key=['xnumel'],   # xnumel 变化时重新 benchmark
)
```

`key=['xnumel']` 使得每个不同的运行时 xnumel 值都会触发一次 benchmark，选出对该 xnumel 最优的 XBLOCK。代价是首次遇到每个新 shape 时需完整跑 benchmark（数秒）。

**关键限制**：configs 列表在编译期基于 `size_hints` 生成。若 hint 为 384，configs 里最大 XBLOCK 为 512；若运行时 xnumel=100000，最优的 XBLOCK=4096 根本不在候选列表里——这是 autotune 模式在 dynamic shape 下的根本短板。

**模式 3：静态特化**

shape 固定时，Inductor 使用精确 numel 生成候选，autotune 选出最优 XBLOCK 并常量折叠进 PTX。无运行时决策开销。

### 9.3 Dynamic Shape 的 Tiling 代价对比

| Op 类型 | Dynamic shape 对 tiling 的影响 | 原因 |
|--------|-------------------------------|------|
| **Pointwise** | 轻微（heuristics 模式基本最优） | 带宽受限，XBLOCK 影响有限；mask 边界开销可接受 |
| **Reduction** | 中等 | RBLOCK 影响并行归约效率；split-K 策略无法针对 shape 优化 |
| **GEMM/Matmul** | **严重**（autotune 候选集被 hint 截断） | BLOCK_M/N/K 对 GEMM 性能影响可达 2–5×；无法 shape-specific autotuning |

### 9.4 为什么 XBLOCK 是 `tl.constexpr`

Triton JIT 为每个不同的 `constexpr` 参数组合生成专用 PTX：

```
XBLOCK=256  → kernel_v256.ptx（循环展开 256 次，针对 256 的寄存器分配）
XBLOCK=1024 → kernel_v1024.ptx（完全不同的指令序列）
```

`constexpr` 不只是优化手段，而是 Triton 代码生成模型的基础——每个不同的 XBLOCK 值确实是一个不同的 kernel binary，cached 后按需复用。

---

## 10. 总结

Inductor 的 codegen 对 dynamic shape 的处理是一个**贯穿 kernel 生成、wrapper 生成、内存规划、运行时求值**的系统工程：

1. **符号透明传递**：所有动态维度以 `sympy.Expr` 形式存在于 IR 中，通过 `SizeArg` 流入 kernel 签名和 wrapper 调用。
2. **运行时求值**：复杂的 shape 表达式在 wrapper 中预计算为中间变量，避免 kernel 调用处表达式过于复杂。
3. **动态 Grid**：通过 `GridExpr` 体系在运行时根据 `numel / BLOCK_SIZE` 动态计算 launch grid。
4. **Unbacked Symbols**：从输出 tensor 的 shape/stride 中反向解析，在 wrapper 中显式声明（详见 [[unbacked_symint_analysis]]）。
5. **安全断言**：通过 `assert_size_stride` 在运行时验证输入 shape，确保编译假设成立。
6. **内存复用保守化**：`buffer_reuse_key` 使用符号字符串而非 size hint 比较，防止不同符号但相同 hint 的 buffer 被错误复用。
7. **XBLOCK 自适应**：默认 heuristics 模式根据运行时 xnumel 动态选 XBLOCK；autotune 模式受 hint 截断影响，对 GEMM 类 op 性能代价最大。

## Related Pages

- [[dynamic_shapes_full_analysis]] — Dynamic Shape 全链路: 静态→符号化→Guard→渐进动态化 (ShapeEnv 源码分析)
- [[unbacked_symint_analysis]] — Unbacked SymInt 深度分析：数据相关 shape 的处理机制
- [[02_engineering/01_ai_frameworks/index]]
- [[inductor_codegen_analysis]]
- [[lowering_analysis]]
