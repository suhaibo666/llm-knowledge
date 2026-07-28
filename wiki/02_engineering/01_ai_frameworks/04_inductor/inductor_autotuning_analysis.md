# Inductor Autotuning 与 Triton 编译流程

> [!correction] 页面角色、审计状态与集中纠错（见 [[correction_report]]）
> **页面角色**：Triton autotune生命周期专题。
> **原始基线**：见下方2026-06-17快照；**当前审计基线**：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`。
> **课程分工**：本页保留autotune纵深；当前codegen/autotune边界与本机未测GPU限制见 [[19_torch_compile_end_to_end/21_codegen_kernel_mapping_autotuning_and_provenance]]。

> 分析对象：upstream PyTorch Inductor 的 Triton kernel **autotune 生命周期 + 如何驱动 Triton 编译器**（`CachingAutotuner`、config 启发式、`config_of`/AttrsDescriptor、`make_launcher`、`triton.compile`→PTX/cubin、`DeviceProperties`）。
> 核心代码位置：本地 upstream `E:\97-codes\pytorch\pytorch`：`torch/_inductor/runtime/triton_heuristics.py`、`codegen/triton_utils.py`、`runtime/hints.py`
> 最后更新：2026-06-17

> 填补 [[01_ai_frameworks/index]] 知识空白「Inductor autotuning」。融合成本模型与 `CoordescTuner`（坐标下降）已在 [[PyTorch_Inductor_Technical_Analysis]] §4 详述,本页**不重复**,聚焦 autotune 运行时生命周期、metadata、launcher 与 Triton JIT 编译。

---

## 一、`CachingAutotuner` 生命周期
> [!correction] I-025、I-026：本区段按固定基线纠错；现行结论见 [[19_torch_compile_end_to_end/21_codegen_kernel_mapping_autotuning_and_provenance#8. 两层autotuning]]，逐项说明见 [[correction_report]]。
每个 kernel 被 `@triton_heuristics.pointwise/reduction(...)` 装饰,运行时由 `CachingAutotuner`（`triton_heuristics.py:421`）驱动,两级派发（`run`,`:2061`）：

```python
def run(self, *args, stream, **kwargs):
    fast = self._cached_launcher                 # ① 快路径：稳态后缓存最终 launcher（~2µs）
    if fast is not None and not kwargs and ...:  return fast(*args, stream=stream)
    if len(self.launchers) != 1:                 # ② 慢路径
        if len(self.launchers) == 0: self.precompile()
        if len(self.launchers) > 1:  self.autotune_to_one_config(*args, **kwargs)
    (launcher,) = self.launchers
    return launcher(*args, **kwargs, stream=stream)
```

- `precompile`（`:643`）：对所有候选 config 调 `_precompile_config`（→ `triton.compile`）→ `_make_launchers`。
- `autotune_to_one_config`（`:1542`）→ `benchmark_all_configs`（`:1479`）→ `bench`（`:1213`）：用 **CUDA event** 计时（`benchmarker.benchmark(rep=40)`）,**跳过寄存器 spill > 16 的 config**,取最快者。
- `save_cache_hook` 把 winner 落盘（`AutotuneCache`）,冷启动直接读最优 config 跳过 autotune。

> **对照 NPU**：`NPUCachingAutotuner` 继承本类,但用 **mspti / torch_npu profiler** 计时（非 CUDA event）、加 **UB 192KB 预算过滤**、`coordinate_descent` 强制关、launcher 包 i64/fp64→i32/fp32 降型（见 [[npu_inductor_linearize_backend_analysis]] §五）。

---

## 二、config 生成启发式

`triton_config`（`:3317`）：从 (x,y,z) block 起步,向 size_hints 收缩再向 grid/target 放大,`num_warps = numel // num_elements_per_warp`（默认 256,SM80+ 最少 4 warp）。

- `pointwise`（`:3877`）：1D 出 2 个候选（bs / bs//2,不同 elem/warp）；2D 出 6 个不同纵横比 `(32,32)(64,64)(256,16)(16,256)(bs,1)(1,bs)`。
- `reduction`（`:4547`）/`persistent_reduction`（`:4809`）：扫 X/R block,按 `ReductionHint.INNER/OUTER` 调 num_warps；`MAX_R0_BLOCK`（Blackwell `cc>=10` 取 1024,旧 2048）；persistent 的 XBLOCK 候选 `[1,8,32,128]`、`MAX_PERSISTENT_BLOCK_NUMEL=4096`。

候选数受 `disable_pointwise_autotuning` / `max_autotune` / `max_autotune_pointwise` 控制；`coordinate_descent_tuning` 在 autotune 出一个 config 后再逐旋钮微调（`CoordescTuner`,详见 [[PyTorch_Inductor_Technical_Analysis]] §4）。

---

## 三、`triton_meta` / `inductor_meta` 与 `config_of`（可整除性提示）

codegen 为每个 kernel 生成两份 metadata：

```python
triton_meta = {
    "signature": {arg: "*fp32"/"i32"/...},      # signature_to_meta（triton_utils.py:165）
    "device": DeviceProperties.create(dev),     # SM 数 / cc / 寄存器 / warp_size（hints.py:157）
    "constants": {...},                          # 含 equal_to_1 的常量（0/1 特化）
    "configs": [config_of(signature)],           # AttrsDescriptor
}
inductor_meta = {"grid_type": "Grid1D"/..., "mutated_arg_names": [...], "size_hints": ...}
```

`config_of`（`triton_utils.py:289`）产 **AttrsDescriptor**,给 Triton 三类提示:**divisible_by_16**（可证 16 整除的实参 → 对齐向量化 load）、**equal_to_1**（可证 ==1 的 size 实参 → 0/1 特化）、**pointer_range_32**（仅 AMD）。

```python
# signature_to_meta._decide_tl_dtype 的关键决策
if (not config.triton.use_block_ptr and isinstance(arg, SizeArg) and arg.name.startswith("ks")):
    return "tl.int64"        # 动态 size 实参升 i64，防 ks0*ks1 溢出
```

> **对照 NPU（尖锐）**：上游对动态 `ks*` 实参**特意升 `tl.int64`** 防乘积溢出；而 NPU 后端因昇腾不支持 i64 计算,签名层把 `*i64→*i32`、`_triton_type_mapping["tl.int64"]="tl.int32"`、launcher 再 downcast——**上游用 i64 规避的问题,NPU 用 i32 重新引入了**（大张量索引溢出风险,见 [[npu_inductor_linearize_backend_analysis]] §七、[[inductor_codegen_dynamic_shape_analysis]]）。

---

## 四、`make_launcher` 与 Triton 编译

`TritonCompileResult.make_launcher`（`triton_heuristics.py:2738`）动态 `exec` 出 launcher：`GridExpr.from_meta` 按 `grid_type` 算 `grid_0/1/2 = ceildiv(numel, BLOCK)`（见 [[inductor_gpu_kernel_dispatch_model]] §四）,再调 `runner(grid_0, grid_1, grid_2, stream, function, metadata, *call_args)`。

Triton 编译路径（`_precompile_config`,`:1114`）：

```python
binary = triton.compile(
    ASTSource(self.fn, compile_meta["signature"], compile_meta["constants"], compile_meta["configs"][0]),
    target=GPUTarget(compile_meta["device_type"], compile_meta["cc"], cc_warp_size(compile_meta["cc"])),
    options={num_warps, num_stages, enable_fp_fusion, ...})
```

`triton.compile` 内部:`@triton.jit` 源 → **Triton IR → TritonGPU IR → LLVM IR → PTX/cubin**,产出 `CompiledKernel`（带 `num_warps`/`shared`/`n_regs`/`n_spills`/`hash`）。Inductor 在这里只是 Triton 的**前端代码生成器 + autotune 驱动**。

`DeviceProperties`（`hints.py:157`,`@functools.cache`）携带 `type/index/multi_processor_count/cc/major/regs_per_multiprocessor/warp_size`,驱动 `num_warps` 上限、`MAX_R0_BLOCK`、occupancy 驱动的 RBLOCK 减半。

---

## 五、小结

| 阶段 | 上游 GPU | NPU 后端 |
|---|---|---|
| autotuner | `CachingAutotuner` | `NPUCachingAutotuner`（继承,coordinate_descent 关） |
| 计时 | CUDA event,rep=40 | mspti / profiler |
| config 过滤 | 寄存器 spill / 占用 | UB 192KB 预算 |
| 动态 size 类型 | `ks*` 升 i64 防溢出 | 降 i32（无 i64 计算,有溢出风险） |
| launcher grid | `ceildiv(numel, BLOCK)` 多维 | 固定 `grid_0 = CU_COUNT` |
| Triton 目标 | NVIDIA Triton → PTX/cubin | Triton-Ascend → bishengir/MLIR |

---

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]] — 当前固定基线的图编译系统化课程入口
- [[inductor_gpu_kernel_dispatch_model]] — grid 生成（launcher 调用的 grid 来源）
- [[inductor_reduction_codegen_deep_analysis]] — reduction config（R0_BLOCK / persistent）
- [[PyTorch_Inductor_Technical_Analysis]] — §4 融合成本模型 + CoordescTuner 坐标下降（本页不重复）
- [[inductor_codegen_dynamic_shape_analysis]] — 动态 shape 与 XBLOCK 选择、ks0*ks1 升 i64
- [[inductor_compiler_pipeline_analysis]] — §7.5 autotuning 基础设施（TuningProcessPool）
- [[npu_inductor_linearize_backend_analysis]] — NPU autotune（UB 过滤 + mspti + 降型）
- [[04_inductor/index]] — 本目录索引
