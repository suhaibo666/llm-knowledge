# 10 · Saved Values、Recompute 与正反向 Runtime ABI

> 前置：[[aotautograd_joint_forward_backward_graphs_analysis]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-30(补与 [[activation_checkpoint_recompute_and_compile_analysis]] 的互指划界)

> [!note] 与 [[activation_checkpoint_recompute_and_compile_analysis]] 的分工
> 本页站在 partitioner 源码与 runtime ABI 层(min-cut flow network、`default_partition`、saved/recompute 的 fw↔bw 真实拼接);用户层 `torch.utils.checkpoint` 的 reentrant/non-reentrant 语义、Selective AC policy 与用户意图如何影响 partitioner,见 [[activation_checkpoint_recompute_and_compile_analysis]]。

## 1. 核心模型

对每个backward需要的forward value，partitioner必须二选一：

```text
save:
  fw计算 value → fw额外输出 → runtime保存 → bw placeholder

recompute:
  fw计算 value用于forward
  同一producer子图复制到bw并再次执行
```

实际还有“强制留在某侧”“无法重算”“symbolic/opaque保存”等约束，但没有第三种跨Graph
Node edge。

## 2. saved tensor 不是唯一 saved value

当前ABI至少区分：

- tensors requiring version-counter checks；
- tensors without version-counter checks；
- symbolic scalar values；
- opaque objects；
- tangents；
- RNG/effect state；
- optional BackwardState。

在runtime：

- 第一类tensor走 `ctx.save_for_backward`；
- no-version-check tensors、SymInt与opaque objects分开存储
  （`torch/_functorch/_aot_autograd/runtime_wrappers.py:2615-2683`）。

把所有边界值都叫saved tensors会漏掉关键类型和顺序。

## 3. 为什么要version check

forward后、backward前，用户可能原地修改saved Tensor。eager autograd会用version counter
检测可能破坏梯度正确性的mutation。AOT generated autograd.Function需保留同类语义。

不是所有内部tensor都需要同样检查，因此ABI分组能避免不必要约束。

## 4. `default_partition`

当前 `default_partition`主要保持forward原位置并保存backward所需forward values；也会尊重
显式recompute tags，并可在activation checkpointing/non-functional图路径转用min-cut
（`torch/_functorch/partitioners.py:1595-1671`;
`torch/_functorch/partitioners.py:1708-1805`）。

它的主要目标不是全局最小activation memory。

## 5. Min-cut rematerialization

min-cut把save/recompute选择构造成flow network：

```text
每个候选 X:
  X_in → X_out    容量代表保存 X 的代价

依赖:
  producer_out → consumer_in    高容量约束

source/sink:
  编码必须在forward、必须在backward、
  禁止/强制重算等条件
```

cut穿过的内部边对应saved values；未被cut但bw所需的forward computation被复制到bw
（`torch/_functorch/partitioners.py:2641-2659`;
`torch/_functorch/partitioners.py:2661-2674`;
`torch/_functorch/partitioners.py:2680-2709`;
`torch/_functorch/partitioners.py:2711-2721`;
`torch/_functorch/partitioners.py:2728-2757`;
`torch/_functorch/partitioners.py:2759-2765`;
`torch/_functorch/partitioners.py:2888-2890`;
`torch/_functorch/partitioners.py:3052-3072`）。

## 6. “最小”优化的是什么

不是简单最少Node数。capacity/cost model可考虑：

- tensor大小；
- recompute runtime；
- view/cheap op；
- fusibility；
- random/mutation；
- banned ops；
- user checkpoint tags；
- memory budget。

min-cut是在给定cost和constraints下求边界，不保证真实hardware wall time或峰值全局最优。

## 7. activation memory budget

budget控制partition cost trade-off，但例如 `0.4`不表示“保留模型全部activations的40%”。
基准/归一化对象、不可重算集合、输入输出、symbolic size hint和cost model都会影响结果。

应把它理解为partition算法的相对资源约束，不是用户层物理字节百分比承诺。

两个边界值有专门快速路径，不经过 knapsack 求解：`activation_memory_budget=0`直接返回
`node_info.inputs`（只保留原始输入，倾向最大重算）；`activation_memory_budget=1`直接返回
`solve_min_cut`的结果（不再叠加 knapsack 折中）；只有 `0`到 `1`之间的值才会用两者的
activation size 做归一化后进 knapsack 求解
（`torch/_functorch/partitioners.py:3471-3480`）。这一步的"边"是 partition 算法临时
构造的 flow-network 边，不是最终 fw/bw GraphModule 的跨图边。

## 8. Recompute如何进入bw

partition extraction为bw创建fresh graph/env。需要重算的joint forward nodes像普通节点一样：

1. inputs映射到bw placeholders或已复制producer；
2. `node_copy`到bw；
3. consumers使用bw内新 Node；
4. runtime执行时再次计算。

`node.meta["recompute"]`是选择/provenance metadata，不是runtime opcode
（`torch/_functorch/partitioners.py:1690-1770`;
`torch/_functorch/partitioners.py:1728-1750`;
`torch/_functorch/partitioners.py:1751-1770`;
`torch/_functorch/partitioners.py:1373-1394`;
`torch/_functorch/partitioners.py:1510-1539`;
`torch/_functorch/partitioners.py:1540-1546`;
`torch/_functorch/partitioners.py:654-661`）。

## 9. 带recompute的bw长什么样

示意：

```text
joint:
  a = sin(x)          forward
  y = cos(a)          forward output
  ga = backward_cos(a, tangent)
  gx = backward_sin(x, ga)

save a:
  fw(x) -> (y, a)
  bw(a, tangent) -> ga -> gx

recompute a:
  fw(x) -> (y, x)
  bw(x, tangent) -> a2=sin(x) -> ga -> gx
```

recompute 版本没有把 boundary value 省掉：它把保存的中间量 `a` 换成了重算 `a` 所需的
primal `x`。因此 fw 必须把 `x` 放进 saved-value ABI，bw 才能用对应 placeholder 重算
`sin(x)`；“不保存 `a`”不等于“bw 可以凭空重新访问 fw 的输入”。

## 10. 为什么还要重排bw

初始bw按joint图顺序复制，可能把所有recompute forward nodes放在gradient work之前，造成
中间值过早产生并长期存活。

`reordering_to_mimic_autograd_engine`创建另一张graph，在某backward node真正需要时才
materialize prerequisite subgraph
（`torch/_functorch/partitioners.py:1920-1995`）。

这是memory-aware schedule rewrite；它不是把bw“逆序”或添加forward↔backward edge。

## 11. Runtime autograd.Function

AOT post-compile创建generated `torch.autograd.Function`：

- `forward`调用compiled fw；
- 分离user outputs与saved groups；
- tensors存ctx，其他objects/scalars存专门fields；
- `backward`过滤/整理grad outputs；
- 按ABI拼bw arguments；
- 调用compiled bw；
- 返回与原primals对齐的gradients。

核心backward prologue组装symbolic values、tensors、opaque objects、filtered gradients与
optional effect/RNG state
（`torch/_functorch/_aot_autograd/runtime_wrappers.py:2982-3010`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:3031-3058`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:3060-3089`）。

最终外层还有RuntimeWrapper恢复原pytree、mutation、alias、subclass calling convention
（`torch/_functorch/_aot_autograd/runtime_wrappers.py:3806-3816`）。

### 11.1 post-compile wrapper 链与输出别名的 handler 派发

上面说的"RuntimeWrapper"不是唯一一层。post-compile 阶段按顺序叠加多个 `CompilerWrapper`
子类，各自负责不同的 ABI 归一化职责：`RuntimeWrapper`（恢复用户可见 calling convention，
`runtime_wrappers.py:189`）→ `AOTDispatchSubclassWrapper`（张量子类的扁平化/反扁平化，
`runtime_wrappers.py:1406`）→ `FunctionalizedRngRuntimeWrapper`（functionalized RNG 状态
的输入/输出接入，`runtime_wrappers.py:1212`）→ `AOTDispatchAutograd`（生成本节描述的
`torch.autograd.Function`，`runtime_wrappers.py:3624`）。dedupe/synthetic-base 两层
（`AOTDedupeWrapper`/`AOTSyntheticBaseWrapper`）在 capture/compile **之前**处理、
post-compile 逆序恢复，属于更早阶段，见 [[graph_effects_alias_mutation_and_order_analysis]]。

输出别名的重建按 `OutputType`（见 [[aotautograd_joint_forward_backward_graphs_analysis]]
§2）分派到专用 handler，而不是一段大 if/else：`_HANDLER_MAP`把每种 `OutputType`映射到
`NoopAliasHandler`/`AliasOfInputHandler`/`IsInputHandler`/`AliasOfIntermediateHandler`
之一，`make_output_handler`按此表构造具体 handler 实例
（`torch/_functorch/_aot_autograd/runtime_wrappers.py:320-345`）。`non_alias`/
`unsafe_view_alias`/`custom_function_view`共用 `NoopAliasHandler`（直接返回，不需要
view 重建）；`alias_of_intermediate`及其两个变体共用 `AliasOfIntermediateHandler`
（需要从 saved base 重放 view）。这解释了为什么 §9.2 的 `OutputType`枚举值比表面看起来的
"输出类型"更精细：它同时是 partition-time 分类，也是 runtime 该走哪条重建路径的 key。

> [!todo] 此处的 view 重放机制由 `gen_alias_from_base`（functional_utils.py:315，
> runtime_wrappers.py 六处调用）实现，原 aotautograd_analysis §10.2 有展开，归一时未落地；
> 待补为本节小节。

编译后的 forward/backward glue 本身也不是手写 Python：它由 `PySourceBuilder`按
`ctx`/`args`/回调签名逐行生成源码再执行（例如 `_codegen_compiled_forward`拼出
`fw_outs = _compiled_fw_(list(args)); _save_(ctx, fw_outs); return _finalize_(ctx, fw_outs)`
这类调用序列，`torch/_functorch/_aot_autograd/runtime_wrappers.py:3215-3256`），backward
prologue 同理生成。理解这一点后，`CompiledFunction.forward/backward`的"实现"应到生成的
source 里找，而不是在 `runtime_wrappers.py`里找一个固定的函数体。

## 12. Saved tensor hooks

用户saved-tensor hooks与compiled autograd交互需显式处理；不能笼统说“AOT不支持”或“照常
在trace时执行”。当前实现有defer/compiled mechanisms，具体path需按runtime wrapper配置
核验。新pass不应绕开ctx保存协议自行藏Tensor。

## 13. save/recompute如何影响后端内存

两层内存不能混：

1. AOT partition决定跨fw/bw生命周期的activation boundary；
2. Inductor在每张图内部决定fusion、materialization、buffer lifetime、reuse与peak reorder。

减少saved activations可能增加bw计算与内部buffer；加强fusion又可能消除某些materialized
intermediate。端到端峰值必须结合两层测量。

## 源码跟读：partitioner 怎样决定保存/重算，runtime 又怎样喂给 backward

### 1. `default_partition` 先恢复 forward 边界，再识别跨边界使用

`default_partition` 顺序扫描 joint graph，找到最后一个 forward/primal/forward-RNG
节点，并把此前非-tangent 节点视为原 forward 区域
（`torch/_functorch/partitioners.py:1595-1641`）。它不是仅凭
`node.meta["is_forward"]` 集合随意分组，而是保留原 forward placement。

在处理 activation-checkpoint/recompute 标记前，代码还会：

- 对非 functional graph 回退 min-cut；
- 清理 recompute tags；
- 强制保存不允许优化的 collective；
- 强制保存 effectful op 与 backward mutation source；
- 调用 `classify_nodes`
  （`torch/_functorch/partitioners.py:1642-1671`）。

这说明 save/recompute 的候选空间已经受语义约束。partition cost model 不能覆盖“不能重复
执行的 effect/mutation/collective”规则。

随后对 forward nodes 查看其 backward users：SymInt 单独进入 `saved_sym_nodes`；
`MUST_SAVE`、impure、opaque 值进入相应 saved 集合；只有未被 `must_recompute` 的普通
Tensor 才加入 `saved_values`
（`torch/_functorch/partitioners.py:1708-1756`;
`torch/_functorch/partitioners.py:1757-1770`）。

因此默认 partition 的核心判断可以写成：

```text
forward value 被 backward 使用
├─ 只需 symbolic size/stride → 保存 SymInt，而非整 Tensor
├─ MUST_SAVE / impure / opaque → 保存
├─ MUST_RECOMPUTE → 不把该中间值放进 saved_values
└─ 其他 → 保存
```

### 2. Min-cut 用拆点图把“保存一个值”表示为可切割容量边

min-cut builder 创建 `networkx.DiGraph`。对禁止重算的 Node，它添加
`source → node_in` 的无限容量边；required backward Node 则按语义连接
`node_out → sink` 或 `node_in → sink`
（`torch/_functorch/partitioners.py:2641-2674`;
`torch/_functorch/partitioners.py:2676-2695`）。

显式 `MUST_RECOMPUTE` 会添加 `node_in → sink` 无限容量边，强制该 operation 位于 cut 后/
backward 一侧；primal、RNG seed 和 cost/purity policy 又可禁止重算
（`torch/_functorch/partitioners.py:2698-2721`）。

最终调用 `nx.minimum_cut(nx_graph, "source", "sink")`
（`torch/_functorch/partitioners.py:2888-2890`）。源码的错误解释也明确列出拆点边语义：

```text
source → X_in       X 不能重算
X_in → X_out        X 的输出是否保存/其容量
X_out → Y_in        Y 依赖 X
X_in → sink         X 必须在 backward 计算
X_out → sink        X 的输出必须供 backward 使用
```

这些含义见 `torch/_functorch/partitioners.py:2933-2944`。所以 cut 优化的是 capacity model
定义的边界，不是抽象的“最少 Node”。

### 3. Recompute Node 进入 backward 的动作就是普通 `node_copy`

cut/默认规则只决定两个列表：

- 哪些 forward values 成为 fw extra outputs / bw placeholders；
- 哪些 forward-origin computations 必须留在 bw 所需 closure 内。

真正构造 bw 时仍调用 `_extract_graph_with_inputs_outputs`。它先为 saved/primal boundary
建立 bw placeholders，然后对 closure 内普通 call Node 执行：

```python
env[node] = new_graph.node_copy(node, lambda x: env[x])
```

对应实现见 `torch/_functorch/partitioners.py:609-659`。因此带 recompute 的 bw 图没有
`recompute` opcode；它只是多出诸如 `aten.sin`、`aten.mm` 的普通 call Nodes，而且这些
Node 的参数全部来自 bw graph 内的 placeholders 或已复制 producers。

以正文示意为例，若不保存 `a = sin(x)`，partitioner 仍必须保存/传入能重新得到 `a` 的
boundary 值 `x`。提取器不能越过 runtime ABI 去访问上一次 fw 执行的局部值。

### 4. 为什么复制后还要 `reordering_to_mimic_autograd_engine`

joint 原顺序是 forward 部分在前、backward 部分在后。直接从 joint 复制得到的 bw 往往是
“所有重算 forward 子图 → 原 backward”。源码 docstring 明确指出这会让重算中间量存活过久
（`torch/_functorch/partitioners.py:1920-1940`）。

reorder 创建另一张 fresh Graph 和 old→new env。对于即将插入的 backward Node，它向上收集
尚未复制的 prerequisite closure，按原图 order 排序后才 `node_copy`
（`torch/_functorch/partitioners.py:1943-1968`）。它从 tangent 的最早 user 找到 backward
起点，再逐 backward Node 触发所需 closure materialization
（`torch/_functorch/partitioners.py:1970-1995`）。

所以“带 recompute 的 backward 图”在重排后更接近：

```text
需要 grad step A
  → 临时重算 A 所需 forward 子图
  → 立即消费并继续 grad
需要 grad step B
  → 再物化 B 所需的剩余子图
```

它不是简单 reverse joint order；算法仍保持每个 Node 的 producer-before-consumer。

### 5. Forward runtime 如何按 slice 保存不同类型的边界值

`_AutogradSavedState.save_from_forward` 按 metadata slices 从 compiled fw outputs 中切出：

- 需要 version check 的 tensors；
- 不需要 version check 的 tensors；
- SymInt/SymFloat values；
- opaque objects。

第一组经必要的 view detach 后传给 `ctx.save_for_backward`，第二组放
`ctx._tensors_no_vc_check`，符号值和 opaque objects 分别放 `ctx.symints` 与
`ctx.opaque_objects`
（`torch/_functorch/_aot_autograd/runtime_wrappers.py:2615-2659`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:2661-2683`）。

这正是“saved tensors”一词不足的源码依据：同一 fw output suffix 通过 metadata 被切成多
种保存协议。

### 6. Backward runtime 按与 partition 相同的顺序重建实参

生成的 backward prologue 先过滤真正存活的 grad outputs，再构造：

```python
all_args = [
    *ctx_symints,
    *ctx_saved_tensors,
    *ctx_opaque_objects,
    *_bw_grads,
    *optional_tokens,
    *optional_rng_state,
]
```

源码生成逻辑见
`torch/_functorch/_aot_autograd/runtime_wrappers.py:3058-3089`。这与 partitioner 创建
bw placeholders 的顺序相互约束；任一侧改变顺序都必须同步 ABI metadata/codegen。

最外层 generated `CompiledFunction.forward` 调用 `_fwd_fn` 并传入
`saved_state.save_from_forward`，`backward` 调 `_bwd_fn` 和生成的 prologue/epilogue
（`torch/_functorch/_aot_autograd/runtime_wrappers.py:3499-3539`）。

完整运行关系因而是：

```text
compiled_fw(real primals)
→ tuple(user outputs, saved suffix)
→ save_from_forward 按 slice 存入 ctx
→ autograd engine 之后调用 CompiledFunction.backward
→ prologue 拼接 saved values + grad outputs + token/RNG
→ compiled_bw(*all_args)
```

两次 compiled call 之间传递的是真实 runtime objects。fw/bw GraphModules 在编译时就已
分开，runtime 不需要也不存在跨 Graph Node 引用。

### 源码边界

这些实现能证明 recompute 的复制方式、autograd-like reorder 和 runtime ABI；仅看
`node.meta["recompute"]` 不能推出最终一定重算。tag cleanup、强制保存规则、min-cut cost、
memory budget 与后续 DCE 都可能改变结果。反过来，bw 中出现 forward-origin target 也需
结合 provenance/partition 对照，不能仅凭算子名字断定它一定是 activation recompute。

## 14. 复杂度

设joint graph `V/E`，flow network `Vf/Ef`：

- boundary closure/extraction近 `O(V+E)`；
- max-flow/min-cut复杂度取决于实现算法与 `Vf/Ef`，不能写成固定线性；
- node copying与selected subgraph大小线性；
- autograd-like reorder含dependency traversal与新Graph复制；
- 真实cost通常由额外bw FLOPs、编译与saved bytes主导。

## 15. 已验证 Lab

从知识库根目录运行：

```powershell
python -B tools\labs_torch_compile\part2_aot_recompute_analysis.py `
  --output-dir tools\labs_torch_compile\artifacts\part2_recompute
```

模型是可微的 `sum(mm(cos(x), weight))`。正例在
`activation_memory_budget=1.0`下保留更多边界值；边界例把 budget 降到 `0.0`，要求 bw
出现额外 forward-origin targets、saved logical bytes下降，同时输出与两组梯度仍等于
eager。Lab记录：

- fw output slots；
- bw placeholders；
- 哪些forward target被复制到bw；
- eager与compiled gradient；
- 两个Graph无跨owner引用。

Lab只把低预算 bw target multiset 相比高预算多出的项列为“观察到的 recompute targets”；
不根据 name 猜测，也不把 forward output 的 user prefix 误计为 saved value。

2026-07-27、PyTorch `2.9.1+cpu`实测：

```text
budget_high_saved_slots=3
budget_low_saved_slots=2
budget_high_saved_bytes=768
budget_low_saved_bytes=512
budget_low_recompute_targets=aten.cos.default,aten.t.default,aten.t.default
budget_high_saved_bytes_gt_low=True
budget_low_recompute_targets_observed=True
gradient_matches=True
cross_graph_node_refs=0
physical_allocator_peak_measured=False
```

`saved_bytes`是 fw saved output tensor metadata 推得的**逻辑字节和**，不是 CPU/CUDA allocator
peak；alias/view slot、allocator rounding 和 backend buffer reuse 都可能使物理峰值不同。
持久 artifact 位于 `tools/labs_torch_compile/artifacts/part2_recompute/`，其中
`partition_comparison.json`定义每个 slot 与计算口径，四个 graph 文件可直接比较 high/low
fw/bw。自动合同 `PartTwoAotRecomputeContractTest`做 assertions；环境与命令见
[`tools/labs_torch_compile/README.md`](tools/labs_torch_compile/README.md)。

同一个模型还有一组运行时 saved-tensor hook 测量：

```powershell
python -B tools\labs_torch_compile\part2_activation_peak.py `
  --output-dir tools\labs_torch_compile\artifacts\part2_activation_peak
```

它在 `pack`/`unpack` 事件上维护两条独立曲线：

- **logical tensor bytes**：所有 active saved Tensor 的 `numel * element_size` 之和；同一
  storage 的两个逻辑 view 分别计数；
- **unique backing-storage bytes**：按 active `untyped_storage()` 去重后，对每个 storage
  只计一次 `nbytes()`。

`pack`返回 `tensor.detach()`而不保存原输入 Tensor 引用；这遵守 saved-tensor hook
“pack 输出不能持有输入 Tensor 引用”的约束
（`torch/autograd/graph.py:327-330`）。

针对同一 32-byte backing storage 上两个互不重叠的 16-byte view，回归测试分别得到
`logical=32`、`unique backing=32`，避免旧的 `max(view_bytes)`算法低估 storage。

2026-07-27 本机重新实测：

```text
budget_high_peak_logical_tensor_bytes=768
budget_low_peak_logical_tensor_bytes=512
budget_high_peak_unique_backing_storage_bytes=768
budget_low_peak_unique_backing_storage_bytes=512
budget_high_pack_unpack_balanced=True
budget_low_pack_unpack_balanced=True
budget_high_logical_tensor_live_bytes_returned_to_zero=True
budget_low_logical_tensor_live_bytes_returned_to_zero=True
budget_high_unique_backing_storage_live_bytes_returned_to_zero=True
budget_low_unique_backing_storage_live_bytes_returned_to_zero=True
gradient_matches=True
physical_allocator_peak_measured=False
physical_allocator_peak_status=blocked_no_cuda
```

这组 `[R]` 证据比“从 fw output metadata 求和”多证明了运行时 pack/unpack 生命周期，但
logical tensor、storage capacity、allocator block rounding、Inductor buffer reuse 与
reserved bytes 仍属于不同层。

CUDA caching allocator 物理峰值在当前无 CUDA 环境中为 `[B]`；它只能由兼容 CUDA
环境中的 allocator API 单独测量，不能从上述两个数反推。

## 16. 设计与调试问题

1. 该bw input是saved tensor、SymInt、opaque object还是tangent？
2. 它在fw output哪个slot，runtime保存在哪个ctx field？
3. 该forward op是saved producer还是copied recompute？
4. recompute metadata来自checkpoint、partition heuristic还是forced rule？
5. bw是否经过autograd-like reorder？
6. memory budget控制的cost定义是什么？
7. 峰值变化来自AOT boundary还是Inductor内部buffer？

## 学习顺序

- 上一篇：[[aotautograd_joint_forward_backward_graphs_analysis]]
- 下一篇：[[graph_stage_boundaries_identity_and_provenance_analysis]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[aotautograd_joint_forward_backward_graphs_analysis]]
- [[buffer_liveness_memory_planning_and_reuse_analysis]]
- [[graph_stage_boundaries_identity_and_provenance_analysis]]
- [[aot_autograd_quickstart]]
- [[activation_checkpoint_recompute_and_compile_analysis]] — 用户 API/策略层(`torch.utils.checkpoint`、Selective AC)的对应物,见页头分工声明
