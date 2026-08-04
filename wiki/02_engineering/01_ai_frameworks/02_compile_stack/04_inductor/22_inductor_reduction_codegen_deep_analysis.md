# Inductor Reduction CodeGen 深度分析（persistent / looped / split / cooperative）

> 分析对象：upstream PyTorch Inductor 的 Triton 后端**如何生成 reduction kernel**——persistent vs looped、split reduction、cooperative reduction、block pointer/TMA。
> 核心代码位置：本地 upstream `E:\97-codes\pytorch\pytorch`：`torch/_inductor/codegen/triton.py`、`codegen/simd.py`、`ir.py`
> 最后更新：2026-06-17

> 这是 [[20_inductor_codegen_analysis]] / [[10_fx_lowering_to_inductor_ir_analysis]] 未展开的 reduction codegen 细节，也是 [[23_npu_inductor_linearize_backend_analysis|NPU 实验性 Linearize 后端]] 中 NPU「persistent 恒关 + r 轴 rsplit」的**上游基线**。

---

## 一、决策：persistent vs looped

`should_use_persistent_reduction`（`triton.py:3302`）委托给 `V.choices`：

```python
def should_use_persistent_reduction(self) -> bool:
    return self.inside_reduction and V.choices.should_use_persistent_reduction(
        self.features, self.cooperative_reduction)
```

- **persistent reduction**（`triton.py:4769`）：整条 reduction 轴一次载入，单次 `tl.sum`/`tl.reduce`，**无累加循环**。reduction numel 较小时用，省循环开销（典型阈值 INNER 4096）。
- **looped reduction**（`triton.py:4890`）：累加器 + 循环 + 循环后归约：

```python
_tmp0 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)        # 累加器初始化
for r0_offset in range(0, r0numel, R0_BLOCK):
    r0_index = r0_offset + tl.arange(0, R0_BLOCK)
    r0_mask  = r0_index < r0numel
    tmp = tl.load(in_ptr0 + (xindex + xnumel*r0_index), r0_mask & xmask)
    _tmp0 = _tmp0 + tmp                                    # combine_fn（post_loop_combine 收尾）
tmp1 = tl.sum(_tmp0, 1)[:, None]                          # reduction_resize（triton.py:4499）
```

> **对照 NPU**：`NPUTritonKernel.should_use_persistent_reduction` **恒 False**（昇腾不支持 persistent reduction），一律走 looped 路径，并重写 `reduction_resize` 的多维槽位；torch_npu 内置后端则**支持** persistent（UB 塞满的单核形态，见 [[21_npu_inductor_optimization_analysis]] §四）。

---

## 二、split reduction（多层归约）

长 reduction 轴可拆成 `split` 段、加一条新轴递归归约：`Reduction.create_multilayer`（`ir.py:2085`），`block_size = ceildiv(reduction_numel, split)`，由 `config.split_reductions` 控制。它把 split 段变成一条**新的 x 轴**。

> **对照 NPU**：`split_reductions` 在 NPU 默认**关闭**——动态 numel 下 `s % split == 0` 不可证 → `need_mask=True` → load 被 `tl.broadcast_to` 包成非线性形态（NPU 慢）。

---

## 三、cooperative reduction（跨 block semaphore 同步）

更长的 reduction 用 cooperative reduction：`program_id(0)` 让给 split-id，各 block 算自己那段，再经 **semaphore + barrier 跨 block 同步**做最终合并（`init_cooperative_reduction`，`triton.py:3210`；配 `CooperativeReductionGrid`，见 [[23_inductor_gpu_kernel_dispatch_model]] §四）：

```python
rsplit_id = tl.program_id(0)
rsplit_chunk = (num_rblocks + RSPLIT - 1) // RSPLIT * RBLOCK
rsplit_start = rsplit_chunk * rsplit_id
# 各 block 局部归约后，HAS_RSPLIT 时经 semaphore 跨 block combine
```

> **对照 NPU（关键）**：cooperative reduction 依赖**单 kernel 内 cross-block barrier**,**昇腾无此同步原语**。所以 [[23_npu_inductor_linearize_backend_analysis]] §五的 **r 轴 rsplit** 用「partial 写 workspace + combine 求和」**两个独立 kernel** 实现同样的「沿 reduction 轴跨核」思路,绕开 barrier。

---

## 四、block pointer / TMA

`config.triton.use_block_ptr` 开启时,用 `tl.make_block_ptr` + `tl.load`/`tl.advance` 替代 arange 索引（`BlockPtrOptions`,`triton.py:754`）,在 A100+ 上走 **TMA**（Tensor Memory Accelerator）硬件搬运:连续/异步/低寄存器压力。仅在访问矩形、无间接索引、mask 为 dense 时启用。

> **对照 NPU**:本后端走 arange + 索引线性化路线,有 `codegen_broadcast_and_reshape` 补丁处理退化广播,无 TMA（昇腾搬运由 MTE/bishengir 处理）。

---

## 五、小结

| 形态 | 触发 | kernel 结构 | NPU 对应 |
|---|---|---|---|
| persistent | rnumel 小 | 一次载入 + 单次 reduce | **恒关** |
| looped | rnumel 大 | 累加器 + `for roffset` 循环 | 默认（恒走此路） |
| split | 超长 | 拆 split 段 + 新轴 | 默认关（动态 numel 非线性） |
| cooperative | 极长 | program_id 让给 split + semaphore barrier | 不可用(无 barrier)→ **r 轴 rsplit 两-kernel** |
| block ptr/TMA | 矩形访问 + A100+ | `make_block_ptr` | 不用(MTE/bishengir) |

---

## Related Pages

- [[23_inductor_gpu_kernel_dispatch_model]] — kernel 骨架与 grid（CooperativeReductionGrid 在此）
- [[20_inductor_codegen_analysis]] — codegen 概览
- [[14_codegen_kernel_mapping_autotuning_and_provenance_analysis]] — codegen/kernel 映射/autotuning/provenance 总线页（本页是其 §6"Loop codegen"的 reduction 纵深）
- [[10_fx_lowering_to_inductor_ir_analysis]] — Reduction IR（make_reduction / Welford）
- [[13_scheduler_dependency_graph_fusion_and_ordering_analysis]] — 融合（含 MixOrderReductions）
- [[21_inductor_autotuning_analysis]] — reduction config（R0_BLOCK / num_warps）
- [[23_npu_inductor_linearize_backend_analysis]] — NPU persistent 恒关 + r 轴 rsplit
- [[02_compile_stack/04_inductor/index]] — 本目录索引
