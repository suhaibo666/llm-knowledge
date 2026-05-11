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

## 9. 总结

Inductor 的 codegen 对 dynamic shape 的处理是一个**贯穿 kernel 生成、wrapper 生成、内存规划、运行时求值**的系统工程：

1. **符号透明传递**：所有动态维度以 `sympy.Expr` 形式存在于 IR 中，通过 `SizeArg` 流入 kernel 签名和 wrapper 调用。
2. **运行时求值**：复杂的 shape 表达式在 wrapper 中预计算为中间变量，避免 kernel 调用处表达式过于复杂。
3. **动态 Grid**：通过 `GridExpr` 体系在运行时根据 `numel / BLOCK_SIZE` 动态计算 launch grid。
4. **Unbacked Symbols**：从输出 tensor 的 shape/stride 中反向解析，在 wrapper 中显式声明。
5. **安全断言**：通过 `assert_size_stride` 在运行时验证输入 shape，确保编译假设成立。
6. **内存复用保守化**：`buffer_reuse_key` 使用符号字符串而非 size hint 比较，防止不同符号但相同 hint 的 buffer 被错误复用。

这些机制共同确保了 Inductor 在**完全动态 shape 的场景下**（如 LLM 的变长序列、数据依赖的 tensor 大小）仍能生成正确且高效的可执行代码。

## Related Pages

- [[dynamic_shapes_full_analysis]] — Dynamic Shape 全链路: 静态→符号化→Guard→渐进动态化 (ShapeEnv 源码分析)
- [[02_engineering/01_ai_frameworks/index]]
- [[inductor_codegen_analysis]]
- [[lowering_analysis]]
