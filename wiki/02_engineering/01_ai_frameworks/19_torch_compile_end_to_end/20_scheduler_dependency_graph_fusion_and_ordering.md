# 20 · Scheduler 依赖图、Fusion 与顺序

> 前置：[[18_inductor_ir_values_loops_layouts_and_buffers]]、[[19_buffer_liveness_memory_planning_and_reuse]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. Scheduler graph为什么不同于FX graph

Scheduler从realized operations构造：

- no-op → `NopKernelSchedulerNode`；
- Computed/TemplateBuffer → `SchedulerNode`；
- ExternKernel → `ExternKernelSchedulerNode`
  （`torch/_inductor/scheduler.py:4641-4651`）。

依赖来自buffer read/write、alias、mutation与额外order，不是FX args/users复制。

源码中的阶段转换是：

```text
GraphLowering.operations
  → create_scheduler_node(operation)
  → SchedulerNode / ExternKernelSchedulerNode / NopKernelSchedulerNode
  → 从 operation 的 read_writes 构造 name_to_users
  → unmet_dependencies / users / mutation renames
  → topological schedule
```

`create_scheduler_node()`只按 operation 类型选择包装类
（`torch/_inductor/scheduler.py:4641-4651`）；它没有复制 FX Node，也没有把 FX `args`
当作 Scheduler edge。因此 SchedulerNode 的 identity、边与生命周期都属于新的图。

## 2. dependency construction

`compute_dependencies()`：

- 创建buffer name→users；
- alias names共享/合并user list；
- mutation rename到最新version；
- 加read/write users；
-处理WeakDep/StarDep；
-处理unbacked symbol origin；
- output/input mutation roots
  （`torch/_inductor/scheduler.py:4689-4717`；
  `torch/_inductor/scheduler.py:4731-4761`；
  `torch/_inductor/scheduler.py:4904-4932`）。

反向依赖存于Scheduler对象，不会加回FX Graph。

### 2.1 源码状态机：一条 read 如何成为依赖边

对普通 read，`node.read_writes.reads`中的 buffer name 被交给 `add_user()`；producer
buffer 的 user list 随后反过来决定 consumer 的 unmet dependency。mutation 则在同一
扫描中先对旧名字加顺序依赖，再更新 `mutation_renames`，使后续读写指向最新版本
（`torch/_inductor/scheduler.py:4850-4879`；
`torch/_inductor/scheduler.py:4904-4917`）。

因此图中同时存在两种方向的索引：

| 结构 | 方向 | 用途 |
|---|---|---|
| buffer `users` | producer buffer → consumers | DCE、fusion、用户查询 |
| node `unmet_dependencies` / read deps | consumer → producer buffer name | topo、合法性、保序 |

它们是对同一依赖事实的不同访问索引，不是“正向 Scheduler 图”和“反向 Scheduler 图”
之间再连一条边。

## 3. Dependency类型

### MemoryDep

包含buffer name与indexing relation，可判断exact shared access/index equivalence。

### WeakDep

表达ordering但对lifetime/DCE较弱，可在目的消失后prune。不是“完全不影响fusion/排序”。

### StarDep

coarse non-index-specific dependency，用于部分mutation/user-defined kernel路径；不是所有
mutation的统一global edge。

### Alias/mutation

通过aliases、mutation renames、users与special deps共同表达，不是单一edge class。

## 4. 当前Scheduler pipeline

主要顺序：

1. global comm ordering；
2. dependencies；
3. topo sort；
4. Scheduler DCE；
5. ancestors/input distances；
6. foreach；
7. pre-fusion custom；
8. optional distributed autotune；
9. stream/mempool assignment；
10. fusion；
11. post-fusion custom；
12. optional second DCE；
13. merge loops、finalize multi-template；
14. combo kernels；
15. switch ordering；
16. peak-memory/comm overlap reorder；
17. grouped/partition processing；
18. last use
   （`torch/_inductor/scheduler.py:4235-4410`）。

## 5. Topological schedule

Scheduler按 `unmet_dependencies`的buffer names DFS，输出producer先于consumer
（`torch/_inductor/scheduler.py:5104-5129`）。

fusion后按min_order排序并再次topological sort
（`torch/_inductor/scheduler.py:6460-6462`）。

该 DFS 对每个节点只访问一次，但源码会对每个节点的
`unmet_dependencies`按 dependency name 排序。因此更精确的骨架复杂度是：

```text
O(V + E + Σ_v d_v log d_v)
```

其中 `d_v`是该节点未满足依赖数。只有依赖集合已预排序或每个 `d_v`有常数上界时，才可
简写为 `O(V+E)`。

## 6. Scheduler DCE

reverse topo；buffer users全weak/removed则inactive；operation无side effects且无active outputs才
删除（`torch/_inductor/scheduler.py:5055-5098`）。

它不使用FX Node users，也不基于ancestors字段定义dead。

## 7. Fusion与相关分组机制

- vertical：producer→consumer；
- horizontal：无直接producer relation但共享iteration/input；
- foreach：把operation list作为专门的组合调度单元；
- template epilogue/prologue；
- reduction/mix-order/nested；
- user-defined Triton等特定extern路径。

普通`ExternKernelSchedulerNode`通常是fusion barrier；只有特定user-defined
Triton/template路径开放受约束的融合（`torch/_inductor/scheduler.py:7914-8006`）。
Reduction也不是universal barrier，但需要domain与backend兼容。

`combo kernel`是fusion之后的另一层并行分组/launch摊销机制，不应和operator fusion混成
同一类。它在`create_combo_kernel_nodes()`中建立，并有独立codegen分支
（`torch/_inductor/scheduler.py:6493-6570`、
`torch/_inductor/scheduler.py:10059-10074`）。

## 8. Candidate generation

每轮按used buffer name grouping；group内每个node只与其后受
`max_fusion_buffer_group_pairwise_attempts`限制的window比较。aggressive fusion另按group
（`torch/_inductor/scheduler.py:6830-6884`）。

这不是总pair数上限；一个node属于多个buffer groups会产生重复候选，再由seen去重。

设第 `g`个 grouping 大小为 `n_g`，window 为 `W`，则一次
`check_all_pairs()`尝试数严格受：

```text
O(Σ_g n_g · min(n_g, W))
```

约束。`aggressive_fusion=True`只是额外加入按 Scheduler `group`划分的 groupings，仍调用
同一个有 window 的 `check_all_pairs()`；在当前默认 `W=64`时，不能直接把它写成
`Θ(V²)`。只有把 `W`视为可随 `V`增长或取消上界，且存在大 grouping 时，候选生成才可能
退化到二次。

## 9. Iterative rounds

最多10个normal rounds；no progress或仅剩1 node停止；配置可追加一次reorder round
（`torch/_inductor/scheduler.py:5268-5301`）。

每次fusion改变groups/dependencies，下一轮可发现新机会。

## 10. Legality

`can_fuse()`在失败时rollback speculative loop mutations。检查包括：

- self/cycle；
- dependencies与reorder legality；
- device/group/size；
- loop/reduction compatibility；
- template/foreach规则；
- no-fuse buffers；
- stream/mempool boundary
  （`torch/_inductor/scheduler.py:7818-8526`）。

cycle check遍历fused ancestors
（`torch/_inductor/scheduler.py:6886-6925`）。

### 10.1 常见融合边界不是同一种“硬墙”

- device不同通常不能组成同一backend codegen group；
- reduction需要iteration/reduction domain兼容，某些producer/consumer仍可融合；
- template可支持受约束的prologue/epilogue fusion，也可能延迟choice；
- extern通常保留library call，但其相邻pointwise是否吸收取决于具体backend/template规则；
- multi-output需要同时维护共享operation和各output users，不能只检查一个输出；
- mutation、collective、stream/mempool会增加ordering与资源约束。

所以排查`can_fuse=False`时必须查看具体dependency、group、backend与原因，不能把所有失败
概括为“遇到reduction/extern就停止”。

## 11. Profitability与priority

`score_fusion_memory()`考虑：

- exact shared MemoryDeps；
- same-buffer不同index overlap；
- common read sets/cache locality；
- mix-order reduction score
  （`torch/_inductor/scheduler.py:8547-8908`）。

`V.choices.score_fusion`提供策略层；template fusion还可异步compile/benchmark
（`torch/_inductor/scheduler.py:6181-6377`）。

## 12. Fusion不保证硬件细节

“fusion成功”表示backend将group作为combined codegen unit处理。它不形式化保证：

- intermediate一定在register；
- shared input只load一次；
- 无spill；
- 一定更快；
- peak memory一定更低。

最终由indexing、backend compiler、register pressure、cache和runtime决定。

## 13. Ordering passes

peak-memory reorder必须靠后，避免后续passes撤销；comm overlap也可重排。
`compute_last_usage()`在这些reorders之后
（`torch/_inductor/scheduler.py:4318-4405`）。

这与FX Pattern replacement后的stable sort是不同层机制。

## 14. 当前配置锚点

pinned OSS：

- memory score threshold 10；
- max fusion size 64；
- pairwise attempts 64
  （`torch/_inductor/config.py:988-1015`）。

配置是heuristic，不是语义；版本变化需重查source。

## 15. 复杂度

设Scheduler `V/E`、round候选 `C_r`：

- dependency常见近 `O(V+E+A)`，alias merge最坏可超线性；
- topo 为 `O(V+E+Σ_v d_v log d_v)`；DCE 的逆序骨架近
  `O(V+E)`，另计 output/user 集合操作；
- ancestors可 `Θ(V²)`；
- candidate attempts 为 `O(Σ_g n_g·min(n_g,W))`；当前 `W=64`时对固定 grouping
  multiplicity 近线性，只有 `W`无界或 grouping membership 本身超线性时才接近 `V²`；
- rounds约 `Σ_r C_r × legality/score`；
- cycle/ancestor traversal使保守上界可写
  `O(Σ_r [C_r × cost(can_fuse + score) + C_r log C_r])`，不含benchmark。

topological schedule本身还会对每个节点的dependency排序，合法 candidates 最后也按
score 排序为 `O(C_r log C_r)`；所以这里给的是参数化成本模型，不把所有sort隐藏成
无条件`O(V+E)`
（`torch/_inductor/scheduler.py:5104-5129`、
`torch/_inductor/scheduler.py:6830-6884`）。

## 16. 已验证 Lab

### 16.1 命令

```powershell
python wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/labs/part4_ir_scheduler_analysis.py `
  --output-dir wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/labs/artifacts/part4_ir

python wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/labs/part4_artifact_bundle.py `
  --output-dir wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/labs/artifacts/part4
```

### 16.2 dependency与fusion group

第一个脚本直接构造Scheduler，不需要native compiler。对
`sin(sum(x, dim=1))`的`max_fusion_size=1`case，它保存内部buffer producer→consumer edge，
而不是从FX `args/users`猜边；对两个独立pointwise输出比较：

```text
dependency_edges_recorded=True
fusion_toggle_observed=True

max_fusion_size=64: scheduler node_count=1, has_fused_scheduler_node=True
max_fusion_size=1:  scheduler node_count=2, has_fused_scheduler_node=False
```

每个node的`reads`、`writes`、dependency index、operation names与`last_usage`写入
`labs/artifacts/part4_ir/scheduler_dependencies.json`和
`fusion_comparison.json`。

该JSON是从当前Scheduler `read_writes`重建的便于阅读的producer→consumer视图；它保留
MemoryDep index等记录，但不是Scheduler内部mutation rename、WeakDep、StarDep与特殊
ordering语义的完整序列化。

同一脚本还在`max_fusion_size=1`下比较
`reorder_for_peak_memory=True/False`：

```text
reorder_comparison_recorded=True
reorder_effect_observed=True
off: [op0, op1, op2], estimated_peak_bytes=264192
on:  [op1, op0, op2], estimated_peak_bytes=263172
```

case让小matmul分支与大reduction分支在最终add前独立可调序。脚本同时断言开/关两种顺序
都满足producer先于consumer；当前heuristic把大输入可尽早释放的分支提前，静态估计降低
1020 bytes。完整记录位于 `labs/artifacts/part4_ir/reorder_comparison.json`。

该结果只覆盖这个固定case的Scheduler静态模型；它不能保证所有图都改变顺序，也不等于
物理allocator峰值或性能一定改善。

### 16.3 codegen对照与证据边界

第二个脚本在相同环境下截获fusion enabled/limited的Inductor trace与C++ source：

```text
fusion_enabled_has_fused_scheduler=True
fusion_limited_has_fused_scheduler=False
fusion_codegen_structure_changed=True
codegen_only_status=generated_not_executed
real_pointwise_compile_status=blocked_missing_msvc_cl
```

codegen-only路径patch了compiler存在性检查并以no-op callable截获source，故它能证明
Scheduler/codegen分组差异进入了wrapper/source。具体地，fusion enabled/limited的
Scheduler group数为`1/2`，captured C++中的`for(`计数为`2/3`，但两边都只有一个
`cpp_pybinding` entry point；所以这里不能把group数直接写成native kernel数。

该codegen-only路径不能证明数值或性能；当前Lab也未把CPU结论外推到GPU
reduction/template fusion，native fusion/reorder on/off的性能对照仍明确标为未执行。

当前native执行层仍因缺`cl`而未完成。

前面的dependency、fusion与静态memory schedule观察已经由不依赖native compiler的路径
独立运行成功。

## 17. 回答用户原问题

> 逆图序逐Node匹配是不是整图？

那是PatternMatcher候选snapshot逆FX顺序，不是Scheduler也不是AOT backward；只遍历注册root
候选union。

> 正向node与反向node有边吗？

AOT fw/bw无跨图edge。Scheduler又是各图lowering后的独立buffer dependency graph。

## 学习顺序

- 上一篇：[[19_buffer_liveness_memory_planning_and_reuse]]
- 下一篇：[[21_codegen_kernel_mapping_autotuning_and_provenance]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[13_pattern_expression_and_matcher_engine]]
- [[19_buffer_liveness_memory_planning_and_reuse]]
- [[21_codegen_kernel_mapping_autotuning_and_provenance]]
- [[scheduler_analysis]]
