---
title: "NPU Inductor Linearize 后端：动态 Shape 方案（编译一次）"
---

# NPU Inductor Linearize 后端：动态 Shape 方案（编译一次）

> 分析对象：实验性 `npu_inductor_2.9.0` 的动态 shape 实现（**≠** torch_npu 内置 `_inductor` 的 gears 分桶）
> 核心代码位置：`npu_inductor/codegen/triton.py`、`npu_inductor/npu_patch.py`
> 版本基线：`npu_inductor_2.9.0` 包 + upstream PyTorch 2.9.0
> 最后更新：2026-06-17

> 本页是 [[23_npu_inductor_linearize_backend_analysis]] 的动态 shape 分册。哲学：**「编译一次，运行期自适应」**——Linearize 展平 + 固定 `grid[40,1,1]` + group 循环后，只把**真正动态**的 `length`/`divisor` 作为运行时标量实参进签名，静态值折叠 `tl.constexpr`，**同一份 kernel 源码、同一次编译**覆盖所有 runtime shape。Dynamo/sizevars 的符号化前半段（`s0→ks0`、`size_hints`、guard）则**完全继承上游**，见 [[20_symbolic_shapes_guards_and_graph_reuse_analysis]]。

---

## 一、两种哲学：编译一次 vs 穷举分桶

| 维度 | 本后端（Linearize） | 内置后端（Split-Tiling + gears） |
|---|---|---|
| 核心哲学 | **一个 kernel，运行期自适应** | **多个变体，编译期分桶** |
| 迭代建模 | 1D 扁平索引 + divisor 链 | 多维 split/tiling/no_loop 轴 |
| Grid 启动 | 固定 `grid[40,1,1]` + group 循环 | 动态 `grid[X,Y,Z]`（`GridNpu`，按 split_axis/split_blocks 生成） |
| 动态 numel 流 | 真正动态的 length/divisor 作运行时实参 | C++ `NPUShapeHandling` 把动态维 pad 到预设 **gears（档位/桶）** + 多变体选择 |
| 编译次数 | **1 次** | **N 次**（按桶数） |
| CU 利用 | 40 核恒被调度（小 kernel 部分空转 + grid clamp） | 小 kernel 可能 < 5 核参与 |

内置后端的动态 shape 难点与 4 个改进方向见 [[01_npu_compile_paths_overview]] §九、[[21_npu_inductor_optimization_analysis]] §十一；本后端用运行期标量 + kernel 内循环让一份编译产物适应全谱。

---

## 二、签名：只传「真正动态」的 numel / divisor

判别用 `isinstance(value, (int, sympy.Integer))`；只有真正动态的值才进签名（`codegen_kernel` 签名生成，`triton.py:1542-1554`）：

```python
for node in tree.nodes.values():
    if node.name in tree_node_mapping:   # 被折叠的派生节点不进签名
        continue
    if not isinstance(node.length, (int, sympy.Integer)):
        signature.append(SizeArg(f"{node.name}numel", node.length))     # 动态 length
    if not isinstance(node.divisor, (int, sympy.Integer)):
        signature.append(SizeArg(f"{node.name}divisor", node.divisor))  # 动态 divisor
```

调用侧（`add_numel_to_call_args`，`triton.py:2155-2222`）把这些 per-node 量算成 wrapper 变量传入；整型实参还会经 `i64→i32` 降型（`triton.py:1560-1571`）。

> **与上游/GPU 的关键差异**：本后端把 permute 维**保持为原生子轴**（divisor 动态），故比上游多传 `*divisor` 实参；上游靠 kernel 内 `y//s`/`y%s` 运行期还原（GPU mod/div 廉价）。三方产物对照见 [[31_npu_inductor_linearize_vs_builtin_comparison]] §1。

---

## 三、header 三件套 + 动态 shape 三情形

每个未折叠节点的迭代由 `_codegen_header_npu_for_tree`（`npu_patch.py:420-637`）用 `real_block`/`_blocks`/`offset` 三件套描述，loop-invariant 提到 `pre_loop_code`：

```python
# pre_loop（静态时 constexpr，动态时退化为运行期标量算式）
x0numel : tl.constexpr = 1024                       # 静态 length 折叠
real_block_x0 = x0numel if x0numel <= (XBLOCK // divisor) else (XBLOCK // divisor) if (XBLOCK > divisor) else 1
x0_blocks     = (x0numel + real_block_x0 - 1) // real_block_x0   # 运行期 ceil 分块
# in-loop
x0offset = (group_base + i) % x0_blocks * real_block_x0
x0index  = x0offset + tl.arange(0, <arange_upper>)[<slot>]
x0mask   = x0index < x0numel                        # 运行期尾部 mask
```

`arange_upper`（`tl.arange` 上界）与 `needs_inner_loop` 取决于 **divisor 静态性 + `divisor_hint = size_hint(divisor)`**（`npu_patch.py:492-544`）。**Triton 要求 `tl.arange` 上界是 constexpr**——这是「动态 divisor 比动态 numel 更棘手」的根因。三情形：

| 情形 | 进签名 | `real_block_x0` | `arange_upper` | inner loop | 依据 |
|---|---|---|---|---|---|
| **A：numel 单独动态** | `x0numel` | 运行期算式 | `XBLOCK`（constexpr） | 否 | divisor 静态、`hint==1` 连续轴 |
| **B：divisor 动态，hint==1** | `x0divisor` | 含 `x0divisor` | `real_block_x0`（length 静态→常量化） | 否 | `divisor_hint==1` |
| **B/C：divisor 动态，hint>1** | `x0divisor`(+`x0numel`) | 含 `x0divisor` | `(XBLOCK // divisor_hint)`（编译期估） | **是** | 动态 divisor + `hint>1` |
| **C：两者都动态，hint==1** | `x0numel,x0divisor` | 全运行期算式 | `XBLOCK` | 否 | `hint==1` |

**情形 A**（最常见，如随输入变化的 batch/seq）：divisor 编译期已知，只 `x0numel` 进签名；单 tile 直接路径，`arange` 上界仍是 `XBLOCK`，靠 `x0mask` 处理 ragged 尾部，开销最低：

```python
real_block_x0 = x0numel if x0numel <= (XBLOCK // 1) else (XBLOCK // 1) if (XBLOCK > 1) else 1
x0_blocks     = (x0numel + real_block_x0 - 1) // real_block_x0
x0offset      = (group_base + i) % x0_blocks * real_block_x0
x0index = x0offset + tl.arange(0, XBLOCK)[...]
x0mask  = x0index < x0numel
```

**情形 B/C（divisor 动态且 `hint>1`）**：动态 divisor 最关键的设计——`arange` 尺寸必须 constexpr，故用**编译期估值 `divisor_hint`** 当上界，再**包一层 inner loop** 兜住「运行期实际 divisor 比估值小（real_block 更大）」的情况：

```python
real_block_x0 = x0numel if x0numel <= (XBLOCK // x0divisor) else (XBLOCK // x0divisor) if (XBLOCK > x0divisor) else 1
# arange_upper = ((XBLOCK // divisor_hint) if (XBLOCK > divisor_hint) else 1)  ← 编译期常量
for x0inner in range(0, real_block_x0, ((XBLOCK // divisor_hint) if (XBLOCK > divisor_hint) else 1)):
    x0index = x0offset + x0inner + tl.arange(0, ((XBLOCK // divisor_hint) ...))[...]
    x0       = x0index
    x0mask   = x0index < x0numel
    ... body ...
```

即：**估值偏大不会越界（mask 兜底），估值偏小由 loop 补足**。

**贯穿三种情形的两条不变量**：
1. `tl.arange` 上界必须 constexpr → 动态 divisor 时只能用 `divisor_hint` 估，并以 inner loop（`hint>1`）或尾部 mask（`hint==1`）补足真实覆盖。
2. `real_block`/`_blocks`/`offset` 按 numel/divisor 静态性退化——能折叠成 constexpr 就折叠（零运行期开销），不能就降标量算式。

---

## 四、配套机制（动态 shape 专属）

- `_fold_trivial_modular_indexing`（`triton.py:880`）：符号 shape 下上游 `simplify_with_ranges` 公理证不出时，丢弃恒等的 `ModularIndexing(base,1,m)`、把恒小 `FloorDiv` 折 0——动态 shape 专属化简补强。
- 符号 split 因子：`_maybe_split_fused_axes`（`triton.py:948`）支持 `FloorDiv(y0, s98)`/`ModularIndexing(y0, 1, s98)` 这类**符号 divisor/modulus**，只要 `c` 整除轴长（运行期可证）即拆分，索引退回纯仿射。
- 静态 split block（默认关，`NPU_STATIC_SPLIT_BLOCK`）：`_enable_static_split_block`（`npu_patch.py:1104`）把 split-reduction 的 per-partial `block_size` 固定为编译期常量、把动态长度移到 split（新）轴，避免动态 `reduction_numel` 下 `block_size` 是符号 `FloorDiv` 而把 div/mod 带进每个 load/store。因曾负优化默认关、留 env A/B。
- `i64→i32` 降型对动态 size 实参同样生效（昇腾无 i64 计算）——大张量溢出风险见 [[23_npu_inductor_linearize_backend_analysis]] §六。

---

## 五、产物实例（permute + add，三维全动态）

`x.permute(0,2,1) + y` `dynamic=True` 的本后端产物（完整三方对照见 [[31_npu_inductor_linearize_vs_builtin_comparison]] §1.4）关键片段：permute 维 `y1` 保持原生子轴（divisor=`s27` 动态、`divisor_hint=128>1` → 走情形 B/C 的 inner loop），签名多出 `y1divisor`：

```python
def triton_unk_fused_add_permute_0(..., ynumel, y0numel, y1numel, y1divisor, xnumel, x2numel,
                                   YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    total_thread = 40; group_id = tl.program_id(0)
    real_block_y1 = y1numel if y1numel <= (YBLOCK // y1divisor) else (YBLOCK // y1divisor) if (YBLOCK > y1divisor) else 1
    ...
    for i in range(group_size):
        for y1inner in range(0, real_block_y1, ((YBLOCK // 128) if (YBLOCK > 128) else 1)):
            y1index = y1offset + y1inner + tl.arange(0, ((YBLOCK // 128) if (YBLOCK > 128) else 1))[:, None, None]
            tmp0 = tl.load(in_ptr0 + (x2 + ks1*y0 + ks0*ks1*y1), ...)   # 纯仿射，无 mod/div
            tmp1 = tl.load(in_ptr1 + (y0 + ks0*x2 + ks0*ks1*y1), ...)   # permute 折进索引
            tmp2 = tmp0 + tmp1
            tl.store(out_ptr0 + (x2 + ks1*y0 + ks0*ks1*y1), tmp2, ...)
```

`_maybe_split_fused_axes` 撤销了上游对 y 轴的合并（否则会 `y//s27`/`y%s27`），于是 load/store 既无 mod/div 也无 kernel 内 permute。

---

## Related Pages

- [[23_npu_inductor_linearize_backend_analysis]] — 本后端总览（架构 + Linearize + 融合 + rsplit + 优化点）
- [[31_npu_inductor_linearize_vs_builtin_comparison]] — permute+add 三方 output code 逐行对比（§1）
- [[20_symbolic_shapes_guards_and_graph_reuse_analysis]] — 上游动态 shape 全链路（ShapeEnv/Guard，本后端继承的前半段）
- [[24_inductor_codegen_dynamic_shape_analysis]] — 上游 codegen 动态 shape（§2.4 `ks*` 升 i64 vs 本后端 i32）
- [[01_npu_compile_paths_overview]] — 内置后端动态 shape 难点（§九 GPU vs NPU）
- [[02_compile_stack/04_inductor/npu/index]] — NPU Inductor 后端目录索引
