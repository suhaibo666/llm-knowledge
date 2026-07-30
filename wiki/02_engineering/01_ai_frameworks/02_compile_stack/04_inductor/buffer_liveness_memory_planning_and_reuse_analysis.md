# 19 · Buffer Liveness、内存规划与复用

> 前置：[[inductor_ir_values_loops_layouts_and_buffers_analysis]]、[[saved_tensors_recompute_and_runtime_abi_analysis]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. 四层“值”与内存

```text
FX logical value
→ lazy IR expression
→ realized named Buffer
→ runtime physical storage/allocation
```

不是每个FX value都占独立buffer；也不是每个buffer都对应独立allocator allocation：

- fusion可让intermediate不落地；
- view可alias；
- reuse可让两个不重叠lifetime buffers共享storage；
- external/output constraints可阻止reuse。

## 2. 从users到liveness

### FX users

说明哪些consumer Node引用producer value，不含physical storage/layout。

### IR reads/writes

说明operation访问哪些buffer/index/alias，是Scheduler dependency入口。

### Scheduler users与last use

在最终schedule上决定某buffer最后被谁使用；之后可free/reuse。

### runtime allocator

接收wrapper allocation/free请求，可在更底层缓存device blocks。编译期buffer reuse与CUDA
caching allocator是不同层。

## 3. 何时需要realize

realization必须拆成“谁决定需要实体化”和“实体化动作如何执行”两部分。
`StorageBox.realize()`只负责后者：把 lazy loop 换成 `ComputedBuffer`并注册；
它本身不能证明所有触发条件
（`torch/_inductor/ir.py:10578-10607`）。

当前源码中可直接定位的典型决策点包括：

| 触发场景 | 决策位置 | 设计原因 |
|---|---|---|
| graph output | `GraphLowering.output()`对结果调用 `ExternKernel.realize_input()`（`torch/_inductor/graph.py:1690-1719`） | wrapper ABI 需要可引用的 storage/layout |
| external input | `ExternKernel.realize_input()`递归处理 TensorBox/view/StorageBox（`torch/_inductor/ir.py:7469-7499`） | 外部调用不能接收任意 lazy Python loop closure |
| stream/mempool 边界 | `_realize_inputs_at_context_boundaries()`（`torch/_inductor/graph.py:1904-1923`） | 若继续内联，Scheduler 无法拆 kernel 或放入正确 pool |
| mutation 边界 | `mark_buffer_mutated()`先 realize 旧值的 users（`torch/_inductor/graph.py:1181-1194`） | mutation 前必须固定所有旧版本读取 |
| 预计重复读取昂贵 | `StorageBox.realize_hint()`检查 nontrivial read count（`torch/_inductor/ir.py:10609-10617`） | 避免未来消费者反复重算同一表达式 |

layout/stride、特定 reduction/template lowering 也有各自的 `require_stride*()`或
`realize()/realize_hint()`调用点，但不能用一处 `StorageBox.realize()`替代这些决策证据。
这也说明 realization 不是统一的“每个 Node 之后做一次”通用 pass，而是消费者契约驱动的
局部边界选择。

## 4. Scheduler last-use

Scheduler在最终reorders后reverse遍历nodes：

```python
future_used_buffers = graph_outputs
for node in reversed(nodes):
    node.set_last_usage(...)
    future_used_buffers.update(node.last_usage)
```

源码：`torch/_inductor/scheduler.py:8959-8968`。

codegen时 `free_buffers()`对可free buffers发wrapper free
（`torch/_inductor/scheduler.py:8970-8995`）。

### 4.1 last use 不是只看直接 read

单个 SchedulerNode 的 `set_last_usage()`先调用
`used_or_aliased_buffer_names()`，把 reads/writes 对应的名字沿
`get_inputs_that_alias_output()`递归展开，再通过 `mutation_real_name`映射回原 physical
name，最后减去未来仍会使用的集合
（`torch/_inductor/scheduler.py:1351-1393`）。

因此逆序扫描维护的是“此位置之后仍活跃的 physical/alias closure”，不是 FX `users`
的简单计数。ordering-only 且 `is_fake=True`的 WeakDep 被刻意排除，避免仅用于保序的边
错误延长 buffer lifetime。

## 5. 不能free/reuse的典型情况

- graph output；
- input/parameter ownership；
- alias/view仍存活；
- mutation target/version；
- MultiOutput layout；
- external library持有；
- stream/mempool不兼容；
- alignment/dtype/device/size不兼容；
- communication buffer专用pool。

## 6. 三套不能混写的“memory planning”

### A. Scheduler peak-memory reorder

改变合法topological schedule，比较baseline、LPMF、BFS、DFS等顺序的估计peak并选最小
（`torch/_inductor/memory.py:1016-1105`）。

它不直接发allocation或决定两个buffer共享地址。

### B. 默认wrapper reuse planning

wrapper IR有 `AllocateLine`、`FreeIfNotReusedLine`、`ReuseLine`
（`torch/_inductor/codegen/wrapper.py:963-984`;
`torch/_inductor/codegen/wrapper.py:986-1009`;
`torch/_inductor/codegen/wrapper.py:1010-1034`;
`torch/_inductor/codegen/wrapper.py:1076-1105`;
`torch/_inductor/codegen/wrapper.py:1131-1151`;
`torch/_inductor/codegen/wrapper.py:1155-1172`;
`torch/_inductor/codegen/wrapper.py:1174-1188`）。

reuse key包括：

- device；
- dtype；
- symbolic storage size；
- alignment；
- stream；
- mempool
  （`torch/_inductor/codegen/wrapper.py:108-138`）。

普通 `memory_plan_reuse()`两遍规划，利用last-use/free lines匹配后续allocation
（`torch/_inductor/codegen/wrapper.py:2531-2582`）。

### C. 可选pooled static planner

仅 `is_inference and config.memory_planning`选择 `MemoryPlanner`；training当前走普通reuse，
因为pooled planning可能增加training peak
（`torch/_inductor/codegen/wrapper.py:2526-2582`）。

五阶段：

1. drop removed；
2. convert pool lines；
3. live ranges；
4. allocate groups；
5. mark first/last use
   （`torch/_inductor/codegen/memory_planning.py:665-683`）。

## 7. Pooled planner不是interval graph最优着色

实现使用LiveRange与TemporalSplit/SpatialSplit allocation tree，按size/lifetime greedy放置
（`torch/_inductor/codegen/memory_planning.py:35-56`;
`torch/_inductor/codegen/memory_planning.py:59-76`;
`torch/_inductor/codegen/memory_planning.py:78-101`;
`torch/_inductor/codegen/memory_planning.py:104-133`;
`torch/_inductor/codegen/memory_planning.py:137-166`;
`torch/_inductor/codegen/memory_planning.py:168-194`;
`torch/_inductor/codegen/memory_planning.py:260-288`;
`torch/_inductor/codegen/memory_planning.py:289-316`;
`torch/_inductor/codegen/memory_planning.py:319-338`;
`torch/_inductor/codegen/memory_planning.py:342-370`;
`torch/_inductor/codegen/memory_planning.py:373-388`;
`torch/_inductor/codegen/memory_planning.py:777-817`）。

它不承诺global optimum。

## 8. memory_pool modes

当前语义：

| mode | 含义 |
|---|---|
| `none` | 不扩展pooled storage，但仍可reuse |
| `intermediates` | non-output共享；每个output独立 |
| `outputs` | intermediate与output分开pool |
| `combined` | 两者可在同一pool |

来源：`torch/_inductor/config.py:248-268`;
`torch/_inductor/codegen/memory_planning.py:497-549`。

## 9. Reuse legality

普通wrapper reuse还用peak estimate拒绝某些远距离reuse：若占用旧storage会让区间peak超过
原overall peak，则保留独立allocation
（`torch/_inductor/codegen/wrapper.py:976-1015`）。

stream在key中，因此不会无声跨stream复用；communication buffers另有严格key/pool。

## 10. Alias与mutation

view本身可能无owning allocation，但延长base lifetime。mutation rename使“旧logical name”
与“当前physical value”关联；Scheduler依赖与wrapper free必须使用real mapping。

MultiOutput/alias output通常不能作为普通reuse source压入reuse state；这不等于它们永远
不能free，wrapper仍可保留并生成`FreeIfNotReusedLine`
（`torch/_inductor/codegen/wrapper.py:1088-1105`）。

源码中，这两种关系分别通过不同机制生效：

1. alias：依赖构造把 alias names 指向共享的 user list，使对任一名字的使用都能约束同一
   storage（`torch/_inductor/scheduler.py:4731-4750`）；
2. mutation：构造 read-after-write/write-after-read 顺序边，并逐次更新
   `mutation_renames`与 `mutation_real_name`
   （`torch/_inductor/scheduler.py:4850-4879`；
   `torch/_inductor/scheduler.py:4880-4917`）；
3. lifetime：真实 view alias 使用 `is_fake=False`的 WeakDep 延长存活，而 clone 只需要
   `is_fake=True`的 ordering dependency，不延长其独立 storage lifetime
   （`torch/_inductor/scheduler.py:4874-4892`）。

这修正了一个常见误解：view lifetime 不是靠“在 view Node 和 base Node 之间保存一条特殊
FX 边”；它依赖 IR alias metadata、Scheduler user 合并与 last-use alias closure 共同实现。

## 11. Fusion与峰值

fusion可：

- 删除intermediate write/read与allocation；
- 延长输入liveness；
- 增大kernel register/shared memory；
- 改变schedule；
- 让template选择变化。

所以kernel数减少不保证device peak一定下降；需测wrapper/allocator与AOT saved activations。

## 12. AOT save/recompute与本层关系

AOT决定跨fw/bw boundary；Inductor在每张图内部规划buffers。

```text
save减少bw FLOPs但跨阶段持有activation
recompute减少saved bytes但bw新增operations/buffers
```

端到端peak是两层共同结果。

## 13. 复杂度

- 若每个 node 的 alias closure 已经展开，逆序 last-use 集合传播骨架近似
  `O(V + D + A)`，其中 `D`为 read/write dependency 总数，`A`为本次遍历实际访问的
  alias 关系；集合哈希操作按均摊常数计；
- 普通reuse字典匹配近线性，但当前line构造用 `scheduler.nodes.index(current)`，可使构造最坏
  达 `O(BV)`；
- peak reorder多heuristics，LPMF path可二次；
- pooled planner排序+allocation tree search最坏超线性；
- symbolic size comparison依赖expression simplification。

普通 wrapper reuse 的数据路径是：

```text
FreeIfNotReusedLine.plan()
  → 按 reuse key 压入可复用状态
AllocateLine.plan()
  → 同 key 弹出候选
  → peak-in-range 检查
  → ReuseLine 或保留 AllocateLine
```

reuse key 的实际字段是 device、dtype、符号 storage size、alignment、stream 与 mempool
（`torch/_inductor/codegen/wrapper.py:123-138`）；分配端的 lookup 与 peak 拒绝逻辑见
`torch/_inductor/codegen/wrapper.py:976-1012`。所以字典匹配近线性不代表整个阶段严格
线性：line 建立时的 `nodes.index()`、区间 peak 查询和符号 size 化简都需要单列。

## 14. 已验证 Lab

### 14.1 命令

```powershell
python tools/labs_torch_compile/part4_ir_scheduler_analysis.py `
  --output-dir tools/labs_torch_compile/artifacts/part4_ir

python tools/labs_torch_compile/part2_aot_recompute_analysis.py `
  --output-dir tools/labs_torch_compile/artifacts/part2_recompute
```

### 14.2 三组对照

| 设计问题 | 运行时对照 | 实际观察 |
|---|---|---|
| view还是copy | transpose vs transpose+contiguous clone | view输出为`ReinterpretView`且无新operation；copy产生`ComputedBuffer` |
| save还是recompute | activation budget `1.0` vs `0.0` | saved slots `3 → 2`，逻辑saved bytes `768 → 512`，低budget的bw出现`cos/t`重算 |
| buffer何时最后使用 | reduction→pointwise Scheduler链 | JSON记录每个node的reads、writes、`last_usage`、前驱/后继与每个buffer的alloc/free bytes |

对Scheduler链调用当前实现的
`torch._inductor.memory.prepare_planning_info()`与`estimate_peak_memory()`，产出：

```text
static_peak_estimate_recorded=True
estimated_peak_bytes=512
```

这里的`512`是该固定CPU case的**静态Scheduler估计**，而且本例主要由仍存活的512-byte
输入主导；它不是“输入总字节”字段，也不是runtime allocator实测峰值。
`768/512`则是AOT boundary上按tensor numel×element_size计算的逻辑saved bytes。两组数字
来自不同层、回答不同问题，不能相加或直接比较。

### 14.3 产物与未覆盖边界

- `tools/labs_torch_compile/artifacts/part4_ir/liveness_peak.json`：静态timeline、buffer与node planning信息；
- `tools/labs_torch_compile/artifacts/part4_ir/fusion_comparison.json`：fusion group对照；
- `tools/labs_torch_compile/artifacts/part4_ir/reorder_comparison.json`：peak-memory reorder开/关的顺序与静态peak；
- `tools/labs_torch_compile/artifacts/part4_ir/ir_matrix.json`：view/copy的ownership差异；
- `tools/labs_torch_compile/artifacts/part2_recompute/partition_comparison.json`：save/recompute签名、重算节点与逻辑bytes；
- 两个目录的`environment.json`：runtime/source边界。

该Lab没有声称测得physical allocator peak、wrapper实际地址复用、native kernel性能或CUDA
caching allocator行为。要回答这些问题，仍需在目标device上增加allocator snapshot/
profiler与真实generated-kernel执行。现有fusion case也只证明Scheduler group数变化，不能
单独证明物理峰值一定下降。

## 15. 排查清单

1. 这是FX value、IR Buffer还是allocator block？
2. buffer何时realize？
3. last use在最终哪个schedule上算？
4. alias/view是否延长base？
5. reuse key的device/dtype/size/alignment/stream/mempool都匹配吗？
6. 讨论的是peak reorder、wrapper reuse还是pooled planner？
7. AOT saved activation是否计入？

## 学习顺序

- 上一篇：[[inductor_ir_values_loops_layouts_and_buffers_analysis]]
- 下一篇：[[scheduler_dependency_graph_fusion_and_ordering_analysis]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[saved_tensors_recompute_and_runtime_abi_analysis]]
- [[inductor_ir_values_loops_layouts_and_buffers_analysis]]
- [[scheduler_dependency_graph_fusion_and_ordering_analysis]]
- [[inductor_memory_management_analysis]]
