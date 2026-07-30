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

## 16. 本页边界之外的两层：运行期物理池与 CUDA Graphs 私有池

前 15 节讲的都是**编译期**——Inductor 怎样规划 `alloc/reuse/free` 的"逻辑"。这只是完整内存
行为的第一层；本节把另外两层接上（2026-07-30 从已删除的 `inductor_memory_management_analysis.md`
判重并入，该页原本就自称是"三层叠加"模型，本页是其中"层 1"的权威页，§16-18 吸收其独有
的层 2/层 3 内容后使本页成为三层的统一入口）：

```text
层1 编译期 Inductor 规划（本页 §1-15）
  → 层2 运行期 CUDACachingAllocator（本节 §16.1）
  → 层3 CUDA Graphs cudagraph_trees 私有池（本节 §16.2，仅 reduce-overhead 模式）
```

**一句话澄清（贯穿三层）**：Inductor 的 `del`/`reuse` 是**逻辑复用**——只决定生成代码里
`empty_strided`/`del` 的顺序，减少分配次数和峰值；**物理显存最终都由 `CUDACachingAllocator`
池管理**。两层叠加才是最终行为。

### 16.1 层 2：`CUDACachingAllocator`（物理池）

§2/§6-9 规划好的 `empty_strided_cuda(...)` 调用，运行时最终都落到 PyTorch 的缓存设备
分配器——真正的物理显存池：一次 `cudaMalloc` 拿一大块 segment，切成 `Block`（`prev`/`next`
双向链表，`CUDACachingAllocator.cpp:201`）；释放只是标空闲 + 与相邻块 coalesce，**不还给
驱动**；按 stream 池化；支持 expandable segments。原因：`cudaMalloc`/`cudaFree` 同步且昂贵，
训练每步上万次张量分配不能每次打扰驱动。

**物理段大小怎么定**：请求先 `round_size`（`CUDACachingAllocator.cpp:3063`：`<512B → 512`；
否则向上取 `kMinBlockSize=512` 的倍数，或按 `roundup_power2_divisions` 在 2 的幂区间内细分），
再由 `get_allocation_size`（`:3697`）按档位决定 segment 大小：

| 请求大小 | segment 大小 | 常量（`c10/core/AllocatorConfig.h:16-24`） |
|---------|-------------|------------------------------------------|
| ≤ 1 MiB | 固定 2 MiB | `kSmallSize=1048576` → `kSmallBuffer=2097152` |
| 1–10 MiB | `large_segment_size()`（默认 20 MiB） | `kMinLargeAlloc=10485760` |
| ≥ 10 MiB | 向上取 2 MiB 的倍数 | `kRoundLarge=2097152` |

即"初始 `cudaMalloc` ≠ 请求字节"，而是按这三档向上取整成段；同一段内的小请求再
`should_split`（`:3677`）切 `Block` 复用。所以一个几 KB 的中间张量首次也会触发一次 2 MiB
段分配——这正是 `max_memory_reserved()`（段总量）常远大于 `max_memory_allocated()`（请求
总量）的原因。配置 `PYTORCH_CUDA_ALLOC_CONF`（`max_split_size_mb`/`roundup_power2_divisions`/
`expandable_segments`）可调这套档位。**两层关系**：Inductor 的 `del`/`reuse` 减少的是逻辑
分配次数；即便仍要 `empty_strided` 分配，物理块也由缓存池复用。`Block`/segment 切分链、
stream 借还、expandable segments、`requested_bytes` vs `allocated_bytes` 碎片度量的源码级
深页见 [[caching_allocator_autocast_profiler_analysis]] §一「缓存设备分配器」。

### 16.2 层 3：`cudagraph_trees` 跨图共享私有池

启用 `mode="reduce-overhead"` 后内存管理多出一层，核心实现 `torch/_inductor/cudagraph_trees.py`。

**为什么是"树"而非单池**：普通 `torch.cuda.graph` 单池捕获有两个硬限制——① 录了 A、B 后
只能按固定顺序重放 A→B；② A→非图代码→B 之后不安全保留 A 产出的中间张量（可能被 B 覆写）。
`cudagraph_trees` 用一棵图树共享同一个私有 mempool 解决：支持任意 A→B 或 A→B′ 的分支（不止
线性序列），图间张量不拷贝、地址稳定、死内存可被后续图复用；主要用于 Dynamo 跨 graph
break/fwd-bwd/多个编译图。

- `CUDAGraphTreeManager`（`:2243`）：每设备一个，把所有录制/执行组织成树，并强制后续执行
  遵循同样的顺序与输出生命周期。`CUDAGraphNode`（`:902`）：一次函数→CUDA Graph 的录制；
  `parent` 用 weakref 避免环，`children: dict[FunctionID, list]`；path = 从根到当前节点的
  有序链。
- **共享私有池**：池在 manager 里建一次（`torch.cuda.graph_pool_handle()`，`:2302`），用一个
  空 capture 的 dummy `CUDAGraph` 保活；单 stream（因为缓存分配器把 segment 绑到分配它的
  stream，多 stream 会妨碍跨录制复用）。每个 node 拿同一个 pool，capture 时
  `torch.cuda.graph(..., pool=self.cuda_graphs_pool, ...)`——一张图产出、下一张图消费的张量
  因此停在稳定地址、无需拷贝。
- **地址稳定/静态输入**：`static_input_idxs`（`:1019`）= 调用方声明的 static ∪
  `cudagraph_managed_idxs`（上一张图的输出）∪ opaque；其补集才需每次拷贝，用一次
  `torch._foreach_copy_` 批量拷进固定的录制 buffer。地址变了则 `check_invariants`
  （`:1932`）比对 `data_ptr`，不一致触发重录子节点，或按 `rerecord_if_static_inputs_change`
  （默认 `True`）硬报错。
- **checkpoint/replay 内存安全**：replay 只重放 GPU 算子、不重建分配器的 CPU 侧簿记，因此
  replay 之后的 eager 分配或录新子图会无法得知图占用的存活块。解法：录制结束时
  `torch._C._cuda_getCheckpointState`（`:1568`）存下分配器状态；需要时用
  `_cuda_setCheckpointPoolState` 重建 CPU 簿记，再对"自上次调用以来已死"的块逐个
  `raw_delete` 物理释放。
- **与 graph partition 的集成**：CUDA Graphs 不能录所有算子，`Scheduler.should_partition`
  （`scheduler.py:8856`）把这些切成独立的非 cudagraph 分区单独跑——CPU 算子、
  `DeviceCopy`、`Conditional`（控制流 cond）、unbacked/动态 shape、
  `is_cudagraph_unsafe_op`。
- **不变量**：无 CPU 算子/单 CUDA 设备；无动态 shape；mutation 只允许改参数/静态输入/图
  录制张量；录制/重放生命周期必须一致；输出不能跨代保留、需 clone（或
  `torch.compiler.cudagraph_mark_step_begin()`）。

### 16.3 三层串起来

| 开关/写法 | 层 | 作用 | 默认 |
|------------|----|------|------|
| `allow_buffer_reuse` | 1 | 逐 buffer 复用（`Allocate`→`Reuse`） | True |
| `reorder_for_peak_memory` | 1 | 算子重排压峰值 | True |
| `memory_planning` + `memory_pool` | 1 | 池化静态规划（仅 inference） | False / `intermediates` |
| （底层，无开关） | 2 | `CUDACachingAllocator` 物理块缓存 | 始终 |
| `mode="reduce-overhead"` | 3 | `cudagraph_trees` 跨图共享私有池 | 需显式开 |

## 17. Pooled planner 的池初始化大小怎么确定（带实例）

补 §6-7 pooled planner 的一个常见疑问：`AllocationPool` 一开始要分多大？答案是**编译期算
出来、不是预设常量**——池大小 = 把所有入池 buffer 的生命周期区间打进 `TemporalSplit`/
`SpatialSplit` 树后的总字节。

- **大小来源**：`AllocationPool.codegen_create`（`memory_planning.py:458`）取
  `nbytes = self.root.get_symbolic_size()`作为池的 `empty` 尺寸；`get_symbolic_size` 递归
  求值：`TemporalSplit`（时分复用）取各块最大值，`SpatialSplit`（空分并排）=
  `align(left) + right`。
- **怎么增长**：新块放不进现有树时，`allocate_at_end`（`:445`）把 `root` 包成
  `SpatialSplit(old_root, new_block)`——即在池末尾追加一段，池随之变大（`can_expand` 由
  `config.memory_pool != "none"` 控）。
- **codegen 形态**：若某单块恰等于整池大小，就按该 buffer 原形状分配；否则发一个扁平 1-D
  `uint8` 缓冲，长度 = `nbytes`。池内每个张量再用 `alloc_from_pool` 按 offset 取视图——
  `alloc_from_pool = torch.ops.inductor._alloc_from_pool`（`wrapper.py:1520`），对应 C++
  算子 `_alloc_from_pool(Tensor self, int offset_bytes, ScalarType dtype, int[] size, int[]
  stride) -> Tensor`（`torch/csrc/inductor/inductor_ops.cpp:36/129`）——**零分配**，只在已
  分配的池存储上按字节偏移建一个张量视图（类 `as_strided`）。

**真实实例**（`test/inductor/test_memory_planning.py:108-142`，`@config.patch(memory_planning=True)`）：

```python
class Foo(torch.nn.Module):           # 两个同时存活的中间张量
    def forward(self, x, y, z):
        t0 = x.matmul(y); t1 = x.matmul(z)
        t0 = x.transpose(0, 1).matmul(t1); t1 = x.matmul(t0)
        return t0.sum() + t1.sum()
# torch.compile(f, dynamic=True) 生成(GPU):
pool1 = empty_strided_cuda((4*s27*s77 + align(4*s77*s77), ), (1, ), ...)   # 扁平字节池
buf0  = alloc_from_pool(pool1, 0,                 torch.float32, (s77, s77), (s77, 1))
buf1  = alloc_from_pool(pool1, align(4*s77*s77),  ...)
```

池 `pool1` 是一条 1-D 缓冲，大小 = `4*s27*s77 + align(4*s77*s77)`（正是
`SpatialSplit = align(left) + right`），两个张量按 offset 各取一段；`alloc_from_pool` 只算
偏移、不分配，真正的 `cudaMalloc` 在运行期由 §16.1 的层 2 触发。动态形状下池大小是**符号
表达式**（`s77`/`s27`），运行期代入具体值后才知确切字节——先算尺寸再 `empty_strided` 的
两阶段流程。

## 18. Wrapper 层补充：boxed calling convention 与通信 buffer 独立池

以下内容 2026-07-30 从已删除的 `wrapper_execution_memory_allocation_and_reuse_analysis.md`
（D05）判重并入其独有部分（§1-3/§9 的宏观框架与本页 §1-2 重叠，未重复搬运）。

### 18.1 Boxed calling convention

一张 Inductor graph 通常 lower 为多个 kernel、extern call 和内存操作；运行时还需接收 flat
boxed inputs、检查/对齐 size/stride/alignment、建立 symbolic scalar、分配中间
buffer/workspace、依次 launch kernel/extern op、处理 stream/device context、释放或复用
dead buffer、组装 views/aliases 和用户 outputs。核心结论：**steady-state compiled call
不是单 kernel 调用；generated wrapper 是一段 runtime program**，负责把 Scheduler 结果和
内存生命周期变成可执行顺序。

`CompiledFxGraph` 的 runtime 入口使用单个 input sequence，源码为这个协议定义
`_BoxedCallable`（`torch/_inductor/output_code.py:80-95`）；`CompiledFxGraph.__call__`
最终 `return self.current_callable(inputs)`，并在外层处理 profiler、runtime metrics 和
first-call autotune cache bundler。boxed list 还允许 runtime 在确认安全时 steal/clear
input refs 以缩短 lifetime。

### 18.2 通信 buffer 独立池

普通 buffer reuse key（§9 已述：device/dtype/symbolic storage size/alignment/stream/
mempool）之外，**通信 buffer 使用独立 pool**，并要求 comm type 和 group 等匹配，避免普通
temporary 错误复用 collective 专用内存（两个 pool 定义见 `torch/_inductor/codegen/
wrapper.py:468-497` 与 `:498-513`）。这说明"size 一样"远不足以证明可复用——设备、dtype、
stream、layout、通信语义都是独立的复用维度。

### 18.3 `AllocateLine.plan` 完整决策序列

补 §6 wrapper reuse 的完整决策顺序（`AllocateLine.plan`，`wrapper.py:986-1015`）：
① removed buffer 变为 `NullLine`；② 通信 buffer 只查通信 pool；③ 普通 buffer 按 reuse key
查 pool；④ 取出最近 free line；⑤ 检查 reuse 是否会提高估算 peak；⑥ 成功则把 free 标为
reused 并替换成 `ReuseLine`；⑦ 不成功则放回 pool 并正常 allocate。stream 属于 key，因此
普通策略自然阻止跨 stream 不安全 reuse。

原 D05 页对应的卷级 Demo 用例仍可运行，验证 §18 的 wrapper reuse 机制（默认以 CUDA 为
验收设备；已从 `demo_manifest.json` 移除 d05 条目，因页面已并入本页——C 卷页不强制
"## 配套 Demo" 小节，此处仅作指路）：

```powershell
python -B tools\labs_torch_compile\demo_d_artifact_runtime.py `
  --case wrapper_memory_reuse --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\d05
```

## 学习顺序

- 上一篇：[[inductor_ir_values_loops_layouts_and_buffers_analysis]]
- 下一篇：[[scheduler_dependency_graph_fusion_and_ordering_analysis]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[saved_tensors_recompute_and_runtime_abi_analysis]]
- [[inductor_ir_values_loops_layouts_and_buffers_analysis]]
- [[scheduler_dependency_graph_fusion_and_ordering_analysis]]
- [[inductor_memory_allocation_guide]] — 实战版：分配器选型对照、`memory_stats`/snapshot 实测、越界/踩踏排查
- [[caching_allocator_autocast_profiler_analysis]] — 层 2 `CUDACachingAllocator` 的 Block/segment/stream/expandable 源码级深页
- [[cudagraph_trees_warmup_record_and_replay_analysis]] — 层 3 cudagraph_trees 的完整 warmup/record/replay 课程页
