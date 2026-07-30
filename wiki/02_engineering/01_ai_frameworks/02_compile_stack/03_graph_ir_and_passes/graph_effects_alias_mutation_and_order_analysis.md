# 05 · Effect、Alias、Mutation 与顺序

> 前置：[[fx_graph_core_data_model_analysis]]、[[graph_values_metadata_and_signatures_analysis]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. 数据依赖不是全部依赖

若 `b.args`引用 `a`，则 `a → b`是显式数据依赖。反过来，没有 Node 引用不代表两个操作
可交换：

```python
x.copy_(y)       # 写 storage
z = x + 1        # 读同一 storage
```

`copy_`的返回值即使没有作为 `add`参数出现，二者仍受 storage effect 约束。类似情况还包括：

- RNG state；
- print/log/I/O；
- collective 通信；
- device stream/event；
- opaque custom operator side effect；
- object/module state mutation；
- effect token。

安全改图必须同时满足 data order 和 effect order。

## 2. FX 的显式结构能表达什么

普通 FX 提供：

- `args/kwargs`数据引用；
- 链表 program order；
- `Node.is_impure()`的 DCE purity heuristic；
- target/schema/meta 供 pass 额外判断。

它没有一个覆盖所有 Python/设备副作用的通用 control-edge 类型。这是 FX 保持轻量通用
program IR 的设计边界，不意味着副作用不存在。

`Graph.eliminate_dead_code()`源码也明确警告：默认 side-effect detection 覆盖并不完备，
只有当图由 functional operations 构成或调用方提供自定义 impurity predicate 时才应假设
sound（`torch/fx/graph.py:2690-2732`）。

## 3. `Node.is_impure()`的当前定义

当前默认行为：

- `placeholder`与 `output`为 DCE 目的始终 impure；
- `call_module`读取 submodule `_is_impure`；
- `call_function`委托统一 library `is_impure`，并可将 random op 视为 impure；
- 其他默认 false
  （`torch/fx/node.py:759-808`）。

这是一套工程 heuristic，不是对任意 Python target 的形式化 effect system。尤其：

- 自定义 callable 的隐藏 I/O 可能无法自动识别；
- GraphModule submodule 内部可能含 effect，源码保留了兼容性限制；
- pass 若知道更强 stage invariant，可提供自己的 predicate。

## 4. Tensor、Storage、View 与 Alias

### 4.1 logical Tensor 与 physical storage

Tensor 包含 size、stride、offset、dtype、device 等 view metadata，并引用 Storage。两个 Tensor
对象可：

- 完全同一对象；
- 不同对象但共享 storage；
- 只共享部分 storage；
- 相同数值但 storage 独立。

图 pass 若只比较 Node identity，就看不到全部 alias 关系。

### 4.2 view

`view`、`transpose`、`slice`等通常产生新的 Tensor metadata 并共享底层 storage。对 base 或
view 的 in-place 写，可能改变另一者可观察值。

因此：

```text
view 无新数据 buffer
≠ view 没有语义
≠ view 永远不需要 copy
```

后端 fixed-stride/layout、alias/mutation、输出逃逸等条件可迫使 materialization/copy。

### 4.3 in-place mutation

`x.add_(1)`至少包含：

- 读取 x 原值；
- 写 x storage；
- 版本计数与 autograd 正确性；
- 对所有 aliases 的可观察影响；
- 返回与 x alias 的结果。

把它机械改成 `y = x + 1`而不把 y 写回/返回到正确位置，会改变用户语义。

## 5. Mutation 的几种类型

AOT metadata 不只记录一个 `mutated=True`。当前 `InputAliasInfo`区分：

- data mutation；
- metadata mutation；
- storage metadata mutation；
- mutation 是否发生在 no-grad/inference；
- mutation 是否 hidden from autograd；
- shallow-copy data；
- backward 是否需要 requires-grad 信息
  （`torch/_functorch/_aot_autograd/schemas.py:137-180`）。

`OutputAliasInfo`还要描述输出是 input alias、intermediate alias、custom function view、unsafe
view 等；枚举与字段定义需一起读
（`torch/_functorch/_aot_autograd/schemas.py:51-128`）。

这些分类决定 functionalization、synthetic base、runtime wrapper、mutation output 和 version
check，不是只为 debug。

## 6. Functionalization 真正做什么

functionalization 的目标是把 alias/mutation 语义转换成后端更容易处理的 functional
program，并在边界恢复用户可见 mutation。

典型思想：

```text
原程序:
  x.add_(1)
  return view_of_x

functional graph:
  x_new = add(x, 1)
  view_new = view(x_new)
  return mutation_result x_new, user_result view_new

runtime wrapper:
  把 mutation_result 写回原输入并重建 alias 语义
```

但“functionalization 删除所有副作用”仍过强。当前 AOT 支持：

- 配置关闭 functionalization；
- controlled input mutation 以 graph tail `copy_`表示；
- effect token；
- RNG functionalization；
- tensor subclass/opaque object 的专门 wrapper。

AOT 在 enabled path 断言 functional graph，同时允许受控 `copy_`尾部
（`torch/_functorch/_aot_autograd/graph_capture.py:340-403`;
`torch/_functorch/_aot_autograd/graph_capture_wrappers.py:1030-1056`;
`torch/_functorch/_aot_autograd/graph_capture_wrappers.py:1058-1070`;
`torch/_functorch/_aot_autograd/graph_capture_wrappers.py:1088-1104`;
`torch/_functorch/_aot_autograd/graph_capture_wrappers.py:1130-1144`）。

### 6.1 为什么后端又会 reinplace

functionalization 的目标是先得到容易分析的纯值语义，不代表后端永远不能恢复 in-place。
post-grad 在主要 functional patterns 与 collective/order 处理之后调用
`reinplace_inplaceable_ops`，随后才做 alias 修正和最终 lint/recompile
（`torch/_inductor/fx_passes/post_grad.py:385-474`）。

reinplace 的合法性不是“发现 out-of-place 对应的下划线算子”这么简单。当前
`can_inplace`会检查 storage alias、graph input view、后续 users、被 mutation 的多参数
是否互相 alias 等条件；同一 storage 已被选择为另一个 reinplace 目标时还可能克隆
（`torch/_inductor/fx_passes/reinplace.py:484-500`;
`torch/_inductor/fx_passes/reinplace.py:555-582`;
`torch/_inductor/fx_passes/reinplace.py:583-610`;
`torch/_inductor/fx_passes/reinplace.py:613-638`;
`torch/_inductor/fx_passes/reinplace.py:643-671`;
`torch/_inductor/fx_passes/reinplace.py:739-834`）。因此设计成
“先 functionalize、完成大多数分析，再在已知 liveness/alias 约束下受控 reinplace”，
是为了把语义证明和最终内存优化分层。

## 7. Synthetic base 为什么存在

若两个输入是同一 storage 的 views：

```python
f(a_view, b_view):
    a_view.add_(1)
    return b_view
```

把二者独立 functionalize 会丢失“写 a 影响 b”的关系。AOT 可将 alias-mutated views 合并为
一个 synthetic base 输入，在图内重建 views，再把结果恢复到原 calling convention。

当前 wrapper 顺序中，dedupe 与 synthetic-base 在 capture/compile 前处理，post-compile
逆序恢复（`torch/_functorch/_aot_autograd/graph_compile.py:185-189`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1586-1608`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1612-1639`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1660-1689`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1696-1716`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1725-1747`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1749-1766`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1844-1863`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1864-1880`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1909-1938`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1945-1960`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1963-1978`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1979-1999`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:2001-2018`）。

## 8. Effect token：把隐藏顺序显式化

effect token 将“操作必须发生且要按序”编码为普通 value threading：

```text
token0
  → with_effects opA → token1
  → with_effects opB → token2
```

当前 `with_effects(token, op, args, kwargs)`返回 new token 与 op results；`has_effects`判断
已注册 effect 类型
（`torch/_higher_order_ops/effects.py:75-138`）。

token 把 effect order 转成图内 use-def，但并非所有副作用都自动 tokenized。它是特定
capture/runtime 路径的机制。

## 9. RNG、collective、stream 与 mempool

### RNG

random op 既产生 Tensor，也推进 RNG state。只保留数值 dataflow 而删除“未使用”的 rand，
会改变后续随机数。FX purity detection 可把 random 视为 impure；AOT/Inductor 还可采用
functional RNG state。

### Collective

collective 的顺序必须在 ranks 间一致；只按 local Tensor dependency 重排可能 hang。
Scheduler 在普通 dependency 前先决定 global communication ordering
（`torch/_inductor/scheduler.py:4235-4244`），对 `cond/switch`也会强制额外顺序
（`torch/_inductor/scheduler.py:4314-4316`）。

### Stream 与 mempool

用户 stream/mempool context 可来自 FX metadata。GraphLowering 将其传播到 IR；
Scheduler 在 fusion 前建立 assignment，并禁止跨不同 stream/mempool fusion
（`torch/_inductor/scheduler.py:4275-4286`;
`torch/_inductor/scheduler.py:7874-7884`）。

这些都说明 Scheduler dependency graph 比 FX users 更丰富。

## 10. 为什么“拓扑正确”仍可能语义错误

对普通 data edges，任何 topological order 都保证 producer-before-consumer。但若隐藏 effect
没有变成 edge：

```text
原顺序: write A → read A
重排后: read A → write A
```

两种顺序都可能满足 FX `args/kwargs`拓扑，却返回不同值。

所以 stable topological sort 只能修复显式 data dependency order。它不会自动发现：

- storage alias；
- Python I/O；
- RNG state；
- collective order；
- user stream relation。

详见 [[dead_code_topology_and_effect_order_analysis]]。

## 11. Pass 合法性边界

结构匹配后至少问：

1. 输入/输出是否 alias？
2. 替换是否改变 mutation 的目标、次数或时机？
3. view/base 的 metadata mutation 是否保留？
4. op 是否 random、collective、I/O 或 opaque？
5. 是否跨 mutation region、stream、mempool？
6. 是否改变 autograd version counter/leaf mutation 规则？
7. functionalization 前后哪个阶段更适合这条规则？

PatternMatcher 当前在应用 match 前会拒绝跨 mutation region 或不同 stream/mempool context
的结果，并执行 `extra_check`
（`torch/_inductor/pattern_matcher.py:2622-2710`）。这只是 stage safety 的一部分，不是完整
alias/effect proof。

## 源码跟读：隐藏 effect 怎样被保留、显式化，再在边界恢复

effect 处理不是一个全局算法，而是三层互补机制：

```text
普通 FX                       AOT functionalization                 显式 effect
program order + purity        mutation → functional outputs        token0 → opA → token1
heuristic                     runtime wrapper 写回输入             token1 → opB → token2
```

### 1. 普通 FX DCE 只看“零 user + purity predicate”

`Node.is_impure` 把 placeholder/output 视为 impure；`call_module` 读取目标 module 的
`_is_impure`；`call_function` 委托 library `is_impure`；其余默认 false
（`torch/fx/node.py:760-808`）。源码还特别指出 GraphModule submodule 可能被当成 pure，
即使其内部含 impure op（`torch/fx/node.py:786-795`）。

`Graph.eliminate_dead_code` 在执行前先 `lint`，要求图已拓扑有序；它的 API 允许传入自定义
`is_impure_node`（`torch/fx/graph.py:2690-2704`;
`torch/fx/graph.py:2734-2740`）。同一 docstring 对默认覆盖给出明确警告：除非图由
functional operations 构成或调用者提供可靠 predicate，否则不能假设该 DCE sound
（`torch/fx/graph.py:2725-2732`）。

因此普通 FX 的设计不是“自动发现所有 effect”，而是：

- Graph 顺序保留原程序的线性次序；
- Node target/schema 提供识别 effect 的材料；
- 通用 DCE 使用保守但不完备的 impurity hook；
- 知道更强 stage invariant 的 pipeline 自己提供额外机制。

引入覆盖任意 Python、设备和自定义对象的统一 effect system，会让基础 FX 捕获几乎整个
Python 语义；当前实现选择把精确性下放到更受约束的 capture/pass stage。

### 2. AOT 在捕获前把原函数包装成 functionalized function

`_prepare_graph_capture_tracing` 检查 `disable_functionalization`：关闭时保留原函数与输入，
开启时调用 `create_functionalized_fn`，同时传入 `ViewAndMutationMeta`、AOT config 与
joint/non-joint 模式
（`torch/_functorch/_aot_autograd/graph_capture.py:214-238`）。

这一步不是在捕获完的 FX 图上简单把 `add_` 字符串换成 `add`。functionalization 运行于
Tensor dispatch/alias 语义层，它要跟踪 view、data mutation、metadata mutation 与 storage
关系，再让捕获看到 functional program。

捕获后，AOT 对 enabled path 调用 `assert_functional_graph`，给 epilogue copy 分配 stream，
包装需要保序的 sync nodes，做 DCE/recompile，并再次确认 functional graph 条件
（`torch/_functorch/_aot_autograd/graph_capture.py:380-395`）。这说明“functional”也是
该阶段主动校验的契约，而不是仅凭某个算子名字推断。

### 3. 用户输入 mutation 被变成输出，再由 runtime wrapper 应用回原对象

runtime wrapper 先按照 `mutated_inp_runtime_indices` 从 compiled outputs 分离 updated
inputs 与普通 outputs（`torch/_functorch/_aot_autograd/runtime_wrappers.py:699-705`）。
`_apply_input_mutations` 再根据每个 input 的 metadata 分流：

- storage metadata mutation 走 `set_` 或 shallow-copy-data；
- 仅 metadata mutation 走 `as_strided_`；
- data mutation 走 `copy_`；
- requires-grad leaf 的隐藏 mutation 使用 detach 后 copy
  （`torch/_functorch/_aot_autograd/runtime_wrappers.py:707-753`;
  `torch/_functorch/_aot_autograd/runtime_wrappers.py:755-790`）。

因此 functionalization 的完整语义是：

```text
编译图内部：原输入不可变，updated input 是一个返回值
runtime ABI：知道哪些返回位置对应哪些原输入
wrapper 边界：按 mutation 分类把 updated value 写回原对象
```

只看 functional FX Graph 会误以为用户输入不再 mutation；只看 wrapper copy 又会误以为
编译图内部仍按 eager in-place 执行。两者合起来才是用户可见语义。

### 4. effect token 把“必须发生且按序”翻译成普通 use-def

`WithEffects` 的接口返回 `(new_token, op_results)`，源码直接说明 token threading 用于阻止
AOT 后续优化重排 side-effectful ops；token 按 effect type 区分
（`torch/_higher_order_ops/effects.py:73-109`）。它目前拒绝 schema 带 alias 的 op，
`has_effects` 也要求已注册 effect 且无 alias
（`torch/_higher_order_ops/effects.py:97-130`）。

在 Proxy 模式下，`with_effects_proxy`：

1. 执行对应 fake/proxy 路径得到抽象结果；
2. unwrap token 与参数；
3. 标记原 op 有 side effect，避免 DCE；
4. 创建 `call_function(with_effects, ...)` Node；
5. 把结果重新关联到 proxy tree
   （`torch/_higher_order_ops/effects.py:170-197`）。

`handle_effects` 则维护 `effect type → current token` 映射；同一 effect type 的操作串接同一
token 链，缺 token 时只有 discovery 模式允许创建
（`torch/_higher_order_ops/effects.py:236-265`）。

token 方案的设计价值在于复用现有 use-def、拓扑、DCE 与 matcher 机制：effect order 一旦
成为数据依赖，普通图算法就能看见它。限制同样清楚：只有进入注册和包装路径的 effect
才被 tokenized；任意 Python I/O、未知 custom callable 或 aliasing mutation 不会因此自动
获得完整证明。

### 5. 为什么不能只依赖 program order

program order 能保存捕获时次序，却不能阻止后续 pass 在“显式数据拓扑仍合法”的前提下
交换两个 effect 节点。purity predicate 只能决定删不删，不能表达多个 effect 之间的先后。
functionalization 只适合可函数化的 mutation，也不能覆盖 print/collective/custom object。

所以三种机制各自解决不同问题：

| 机制 | 解决 | 不能单独解决 |
|---|---|---|
| program order + impurity | 保留顺序表示、避免明显误删 | 证明重排安全 |
| functionalization + wrapper | alias/mutation 的纯值化与边界恢复 | 任意外部副作用 |
| effect token | 把注册 effect 顺序变成显式 use-def | 未注册 effect、aliasing op 的全部语义 |

pass 合法性检查必须先确认当前 stage 已经把哪些 effect 显式化，再决定可否依赖普通拓扑
算法；不能把某一机制的适用边界推广到整条编译链。

## 12. 复杂度边界

令 `V/E`为 FX 节点/显式 uses，`A`为需追踪的 alias/view 关系数量：

- 普通 DCE 的结构扫描接近 `O(V+E)`，但 purity/effect callback 成本另计；
- functionalization 至少遍历被执行的算子与 alias/mutation 更新，成本不能只由 FX `V`
  给出，因为 Python dispatch、FakeTensor/meta kernel 与 view replay 都是外生成本；
- 精确 storage-alias 分析若对大量候选做 pairwise 比较，参数化上界会包含 `O(A²)`；
  当前实现通过 storage/metadata 集合缩小候选，不等于所有场景都线性；
- reinplace 对节点 users、storage 与候选参数进行局部检查；常见 bounded arity 下接近图扫描，
  但 user fan-out、嵌套参数和自定义 extra check 必须保留在上界中。

在没有 alias/fan-out 分布时，期望复杂度未定义。性能结论必须绑定具体 stage 和实现，
不能把上述机制统一压成一个 `O(V+E)`。

## 13. 已验证 Lab：DCE、alias、effect order 与 functionalization

从知识库根目录运行：

```powershell
python -B tools\labs_torch_compile\part1_effects_alias.py
python -B tools\labs_torch_compile\part2_normalization.py
```

第一个脚本构造 unused pure add 与 unused `copy_`，并把两个都写同一 storage、但彼此没有
data edge 的 `copy_`交换链表位置。两种顺序都通过 `Graph.lint()`，结果却从 `2`变成 `1`：
这正是“拓扑合法不等于 effect 保序”的错误/边界例。第二个脚本对 mutating view 做
functionalization，比较输出与用户可见 input mutation。PyTorch `2.9.1+cpu`实测：

```text
before_dce_call_functions=2
dce_changed=True
after_dce_targets=copy_.default
mutation_result=7.0
alias_observes_mutation=7.0
pure_dead_removed=True
impure_copy_retained=True
both_effect_orders_lint=True
effect_reorder_changes_result=True
original_has_inplace=True
functional_has_outplace_add=True
functional_output_matches=True
functional_input_semantics_match=True
```

结果说明：

- 无 users 的 pure add 被删除；
- `copy_`被 impurity heuristic 保留；
- view alias 能观察到底层 storage mutation。
- stable data topology 本身没有恢复两个 effect 的语义顺序；
- functionalization 改写图形态，但 wrapper-level 输入 mutation 语义与原函数一致。

这不证明默认 DCE 对所有自定义副作用 sound；源码警告仍成立。

stdout 可持久化到 `tools/labs_torch_compile/artifacts/logs/part1_effects_alias.txt`与
`part2_normalization.txt`；统一模型的 functional ATen 图在
`tools/labs_torch_compile/artifacts/end_to_end/functional_aten.py`。自动合同
`EffectAndFunctionalizationContractTest`对上述关键字段做 assertion。环境与命令见
[`tools/labs_torch_compile/README.md`](tools/labs_torch_compile/README.md)。

## 14. 本篇心智模型

```text
Node dependency
  = explicit value use

Program legality
  = value dependency
  + alias/storage relation
  + mutation semantics
  + effect ordering
  + runtime/compiler stage contract
```

## 学习顺序

- 上一篇：[[symbolic_shapes_guards_and_graph_reuse_analysis]]
- 下一篇：[[structured_outputs_higher_order_and_nested_graphs_analysis]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[fx_graph_core_data_model_analysis]]
- [[graph_normalization_decomposition_and_functionalization_analysis]]
- [[dead_code_topology_and_effect_order_analysis]]
- [[graph_rewrite_legality_validation_and_complexity_analysis]]
- [[19_buffer_liveness_memory_planning_and_reuse]]
- [[01_eager_runtime/05_autograd_engine/index]]
