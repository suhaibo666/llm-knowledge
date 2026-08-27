---
title: "Inductor GPU Kernel 派发模型（program_id / IterationRanges / Grid）"
---

# Inductor GPU Kernel 派发模型（program_id / IterationRanges / Grid）

> 分析对象：upstream PyTorch Inductor 的 Triton 后端**如何把迭代空间映射到 GPU grid 并派发**（kernel 骨架、IterationRanges 树、tiling、grid 类型）。
> 核心代码位置：本地 upstream `E:\97-codes\pytorch\pytorch`：`torch/_inductor/codegen/simd.py`、`codegen/triton.py`、`runtime/triton_heuristics.py`
> 最后更新：2026-06-17

> 本页是「GPU dispatch 模型」的基线，补 [[20_inductor_codegen_analysis]]（codegen 概览）与 [[24_inductor_codegen_dynamic_shape_analysis]] §3（grid 动态计算）未展开的 **kernel 骨架 + IterationRanges + grid 类型**。它也是 [[23_npu_inductor_linearize_backend_analysis]] 中 NPU「40-CU group dispatch」所**替换**的对象——对照阅读最清晰。

---

## 一、生成的 kernel 骨架：`program_id → offset → index → mask`，无循环

GPU 路线的「默认形态」是把分块交给 **grid（硬件调度器）**，kernel 内**没有 for 循环**。pointwise kernel 的 header 由 `iteration_ranges_codegen_header`（`triton.py:6936`）生成，pid 由 `iteration_ranges_get_pid`（`triton.py:6754`）给出：

```python
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex  = xoffset + tl.arange(0, XBLOCK)[:]
    xmask   = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + xindex, xmask)
    ...
    tl.store(out_ptr0 + xindex, tmp_out, xmask)
```

每个维度（x/y/z）各取一个 `tl.program_id(grid_dim)` 映射到 grid 轴；`program_id * BLOCK` 得到块基址，`tl.arange(0, BLOCK)` 是块内偏移，`mask` 滤越界。这是 NPU 后端用固定 `grid[40,1,1]` + `for i in range(group_size)` 所替换的核心（见 [[23_npu_inductor_linearize_backend_analysis]] §二）。

---

## 二、IterationRanges 树：迭代空间的结构化表达

迭代空间用一棵 range-tree 描述（`simd.py:105-405`）：

- `IterationRangesRoot`：一个被 tile 的维度。`prefix` = `x`/`y`/`z`（pointwise）或 `r0_`/`r1_`（reduction）；`tensor_dim`（Triton 张量维）、`grid_dim`（grid 轴 0/1/2）、`is_loop`（非 persistent reduction 为 True）。
- `IterationRangesEntry`：叶子，把扁平索引按 `FloorDiv`/`ModularIndexing` 分解（`lookup`，`simd.py:258`）：

```python
if V.graph.sizevars.statically_known_equals(divisor * length, self.numel):
    expr = FloorDiv(self.index_sym(), divisor)               # 整除：精确
else:
    expr = ModularIndexing(self.index_sym(), divisor, length)  # 不整除：取模
```

> 这棵树是 upstream 与 NPU 后端**共享**的；区别在于 GPU 直接用 `program_id + arange` 喂它且 mod/div 廉价，NPU 则要把 mod/div 化简成纯仿射并用 group 循环替代 grid（[[23_npu_inductor_linearize_backend_analysis]] §二的索引线性化）。

---

## 三、tiling：按 stride-1 连续性打分

`select_tiling`（`simd.py:4324`）→ `candidate_tilings`（`simd.py:3744`）按访存连续性选 tile：

```python
strides = V.graph.sizevars.stride_hints(dep.index, rw.range_vars)
split = strides.index(1) + 1            # 最右 stride-1 维作切点
score = V.graph.sizevars.optimization_hint(...)
if dep.name in write_names: score *= 2  # 写连续更值钱
if CandidateTiling.is_good_size(...): score *= 2   # is_good_size: s>=32 且 s%32==0
```

在最右 stride-1 维切开以合并访存，偏好 32 对齐 tile，候选按分数排序取最优；1D / 2D(x,y) / reduction tiling 据此决定。

> NPU 后端复用 `select_tiling`，但把 `is_good_size` 放宽到 `s>=2`（NPU 不要求 32 对齐）；torch_npu 内置后端则自带多级 `create_tiling`（见 [[11_npu_inductor_splittiling_backend_analysis]]）。

---

## 四、grid 生成与 Y/Z 溢出

launch grid 在**调用时**由 numel 与 BLOCK 算出（`runtime/triton_heuristics.py`）：

| 类 | 位置 | 生成 |
|---|---|---|
| `GridExpr`（基类） | `:5111` | `ceildiv(numel, BLOCK)` |
| `Grid1D` | `:5240` | `x_grid = ceildiv(xnumel, XBLOCK)` |
| `Grid2D` | `:5245` | `x_grid, y_grid` |
| `Grid2DWithYZOverflow` | `:5265` | y 超硬件上限（65535）时拆到 z；kernel 内 `(program_id(1) + program_id(2)*num_programs(1))*YBLOCK` 还原线性 y |
| `CooperativeReductionGrid` | `:5297` | `program_id(0)` 让给 reduction-split（见 [[22_inductor_reduction_codegen_deep_analysis]]） |

GPU grid 各维有硬件上限 `maxGridSize = [2^31-1, 65535, 65535]`，故有 Y/Z 溢出处理。

> **对照 NPU**：本后端固定 `grid[40,1,1]`（launcher `grid_0 = CU_COUNT`，简单 1D 才 clamp），不存在 Y/Z 溢出——因为分块在 kernel 内由 group 循环完成，不依赖 grid 维度。

---

## 五、小结

GPU dispatch 模型把「分块、派发、Y/Z 溢出」交给 grid + 硬件调度器，kernel 内无循环；reduction 的跨核同步见 [[22_inductor_reduction_codegen_deep_analysis]]；autotune 如何选 BLOCK/grid 见 [[21_inductor_autotuning_analysis]]。NPU 后端因「固定 40 核单维 dispatch」把这些搬进 kernel body（[[23_npu_inductor_linearize_backend_analysis]]）。

---

## Related Pages

- [[20_inductor_codegen_analysis]] — Inductor codegen 概览（本页是其 GPU 派发细节）
- [[14_codegen_kernel_mapping_autotuning_and_provenance_analysis]] — codegen/kernel 映射/autotuning/provenance 总线页（本页是其 §6"Loop codegen"的 GPU 派发纵深）
- [[24_inductor_codegen_dynamic_shape_analysis]] — §3 grid 动态计算（本页补 kernel 骨架/IterationRanges）
- [[22_inductor_reduction_codegen_deep_analysis]] — reduction codegen（persistent/looped/cooperative + CooperativeReductionGrid）
- [[21_inductor_autotuning_analysis]] — autotune 选 BLOCK/grid 与 Triton 编译
- [[13_scheduler_dependency_graph_fusion_and_ordering_analysis]] — 调度与融合（tiling 在融合中的作用）
- [[23_npu_inductor_linearize_backend_analysis]] — NPU 用 group dispatch 替换本模型
- [[02_compile_stack/04_inductor/index]] — 本目录索引
