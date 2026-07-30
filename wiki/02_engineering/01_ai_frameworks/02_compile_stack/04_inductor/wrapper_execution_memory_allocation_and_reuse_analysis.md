# D05 · Wrapper 执行、内存分配与 Buffer Reuse

> 卷别：D · 编译产物、缓存与运行时  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[02_compile_stack/06_compile_cache/index]]  
> 后续：[[cudagraph_trees_warmup_record_and_replay_analysis]]  
> 最后更新：2026-07-30(kb-reorg P4 Task 6 迁入本目录,去 d05_ 前缀)
>
> **内存规划相关页归一进行中**:本页讲 wrapper 层 `MemoryPlanningLine`/reuse pool 的源码级机制,与
> [[inductor_memory_management_analysis]](编译期规划→`CUDACachingAllocator`→CUDA Graph 池三层全景)、
> [[inductor_memory_allocation_guide]](分配器实战与越界排查)存在视角重叠,尚未按 Task 8 计划与 C19 一并归一,暂各自独立成页,先在此互指。

## 1. 为什么kernel之外还需要host wrapper

一张Inductor graph通常lower为多个kernel、extern call和内存操作。运行时还需：

- 接收flat boxed inputs；
- 检查或对齐size/stride/alignment；
- 建立symbolic scalar；
- 分配中间buffer/workspace；
- 依次launch kernel/extern op；
- 处理stream/device context；
- 释放或复用dead buffer；
- 组装views、aliases和用户outputs；
- 可选profiler、debug sync、CUDAGraph。

**核心结论**：steady-state compiled call不是单kernel调用；generated wrapper是runtime
program，负责把Scheduler结果和内存生命周期变成可执行顺序。

## 2. Boxed calling convention

`CompiledFxGraph`的runtime入口使用单个input sequence，源码为这个协议定义
`_BoxedCallable`（`torch/_inductor/output_code.py:80-95`）。

`CompiledFxGraph.__call__`最终：

```python
return self.current_callable(inputs)
```

并在外层处理profiler、runtime metrics和first-call autotune cache bundler
（`torch/_inductor/output_code.py:785-813` 与
`torch/_inductor/output_code.py:817-840`）。

boxed list还允许runtime在确认安全时steal/clear input refs以缩短lifetime。

## 3. Wrapper code怎样组装

Python wrapper codegen按section构建：

- imports/header；
- kernel declarations；
- call body；
- subgraph/partition definitions；
- suffix/benchmark harness。

生成时先跑wrapper IR passes，再逐行codegen allocation/free/kernel calls，最后生成return
（`torch/_inductor/codegen/wrapper.py:2387-2416`、
`torch/_inductor/codegen/wrapper.py:2417-2418` 与
`torch/_inductor/codegen/wrapper.py:2420-2437`）。

最终把各section拼成源码和linemap
（`torch/_inductor/codegen/wrapper.py:2439-2468` 与
`torch/_inductor/codegen/wrapper.py:2469-2476`）。

## 4. MemoryPlanningLine为何先作为IR保存

如果一发现buffer就立即输出 `empty_strided`，之后看到free时已经来不及把两者改成reuse。
所以wrapper先记录：

- `AllocateLine`；
- `FreeIfNotReusedLine`；
- `ReuseLine`；
- `ReinterpretLine`；
- subgraph scope lines。

随后memory planning pass全局查看line序列，把匹配的allocate/free配对，再codegen。

## 5. Reuse key包含哪些事实

普通buffer reuse key至少考虑：

- device；
- dtype；
- allocation storage size；
- stream；
- layout/其他可复用约束。

通信buffer使用独立pool，并要求comm type和group等匹配，避免普通temporary错误复用collective
专用内存。两个pool定义见 `torch/_inductor/codegen/wrapper.py:468-497` 与
`torch/_inductor/codegen/wrapper.py:498-513`。

这说明“size一样”远远不足以证明可复用。

## 6. Allocate怎样寻找dead buffer

`AllocateLine.plan`：

1. removed buffer变为NullLine；
2. 通信buffer只查通信pool；
3. 普通buffer按reuse key查pool；
4. 取出最近free line；
5. 检查reuse是否会提高估算peak；
6. 成功则把free标为reused并替换成ReuseLine；
7. 不成功则放回pool并正常allocate。

见 `torch/_inductor/codegen/wrapper.py:986-1015`。

stream属于key，因此普通策略自然阻止跨stream不安全reuse。

## 7. Free为什么不是立即释放

`FreeIfNotReusedLine.plan`先把候选放入reuse pool；到codegen时：

- 已被后续allocation复用：不生成真正free；
- 未被复用：生成free/deletion；
- alias output或MultiOutputLayout等情况不进入普通reuse。

见 `torch/_inductor/codegen/wrapper.py:1075-1104`、
`torch/_inductor/codegen/wrapper.py:1105-1105` 与
`torch/_inductor/codegen/wrapper.py:1107-1120`。

这是“最后一次逻辑使用”到“wrapper变量释放/allocator可复用”的转换。

## 8. 两种memory planning模式

`run_wrapper_ir_passes`当前：

- inference且开启完整memory planning：调用独立 `MemoryPlanner`；
- training默认不用完整planner，因为当前可能增加peak；
- 其他路径仍可做line-based reuse。

见 `torch/_inductor/codegen/wrapper.py:2575-2582`。

`memory_plan_reuse`按line序列维护嵌套planning states，处理subgraph scope并估算分配总量
（`torch/_inductor/codegen/wrapper.py:2531-2553` 与
`torch/_inductor/codegen/wrapper.py:2554-2573`）。

## 9. Logical liveness、wrapper reuse和allocator reuse

三个概念不同：

| 层 | “释放”含义 |
|---|---|
| Scheduler liveness | 后续kernel不再读取该logical buffer |
| Wrapper code | Python/C++变量删除，或storage改名复用 |
| Caching allocator | block回到allocator pool，可供之后allocation |

CUDA Graph private pool、views/aliases、external outputs、retain_graph和用户引用都会让物理
内存行为偏离静态liveness估算。

## 10. View与alias为何限制reuse

如果一个dead-named buffer仍被live view/alias引用，storage并未dead。安全reuse必须考虑：

- graph output aliases；
- input mutation；
- MultiOutputLayout；
- reinterpret view范围；
- external kernel持有；
- async stream完成；
- saved tensor/backward lifetime。

因此DCE删除一个logical op和复用其storage是不同证明。

## 11. Output为什么不能作为普通reuse target

用户output在wrapper返回后继续存活。其storage不能被同一调用后续temporary覆盖；跨调用是否
可复用又取决于CUDAGraph/ownership契约。`memory_plan_reuse`会移除末尾无意义的planning
lines，但以graph output names为边界
（`torch/_inductor/codegen/wrapper.py:2531-2542`）。

## 12. 执行顺序与保序

wrapper lines来自Scheduler codegen顺序。安全重排需满足：

- reads在producer writes之后；
- mutation/alias dependencies；
- collective/stream/event顺序；
- allocation早于使用；
- free晚于最后使用；
- reuse后旧name不再可观察；
- extern calls的隐藏effect。

没有一个通用topo sort能单独证明这些；Scheduler dependency graph和wrapper memory plan共同
决定最终顺序。

## 13. 复杂度

设wrapper有 \(L\) lines、buffer数 \(B\)，reuse pool用hash key：

- 单遍line planning期望 \(O(L)\)；
- pool push/pop期望 \(O(1)\)；
- peak-between查询取决于估算数据结构；
- 源码生成 \(O(L)\)；
- runtime allocation次数在最佳复用下显著少于 \(B\)，但受key/alias/stream限制；
- 实际物理peak还受allocator fragmentation、reserved blocks和异步lifetime影响。

## 14. 常见误解

- **“compiled graph执行就是直接进kernel。”** 先经过boxed/runtime wrapper。
- **“node dead就能立刻释放storage。”** aliases、streams、outputs和backward可能仍live。
- **“相同numel即可复用。”** device/dtype/storage size/stream/layout/effects都重要。
- **“删除Python变量就把CUDA内存还给OS。”** 通常只是回到caching allocator。
- **“静态peak等于监控到的reserved peak。”** 二者处于不同层。

## 配套 Demo

本页对应卷级入口 `tools/labs_torch_compile/demo_d_artifact_runtime.py` 的 `wrapper_memory_reuse` 用例。默认以 CUDA 为验收设备：

```powershell
python -B tools\labs_torch_compile\demo_d_artifact_runtime.py `
  --case wrapper_memory_reuse --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\d05
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `wrapper_memory_reuse/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[19_buffer_liveness_memory_planning_and_reuse]]
- [[20_scheduler_dependency_graph_fusion_and_ordering]]
- [[02_compile_stack/06_compile_cache/index]]
- [[cudagraph_trees_warmup_record_and_replay_analysis]]
- [[kernel_fusion_memory_and_hardware_performance_analysis]]
