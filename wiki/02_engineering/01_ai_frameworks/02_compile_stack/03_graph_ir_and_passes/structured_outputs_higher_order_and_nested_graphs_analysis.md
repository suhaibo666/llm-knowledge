# 06 · 结构化输出、Higher-Order Operator 与嵌套图

> 前置：[[graph_values_metadata_and_signatures_analysis]]、[[graph_effects_alias_mutation_and_order_analysis]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. 普通“一节点一张量 DAG”模型不够

真实程序会返回 tuple/dict，调用单个多结果算子，包含条件分支、循环、map/checkpoint，
还会在 Node 参数里携带子程序。FX 仍可用外层无环 program graph 表达这些结构，但需要
区分：

- 一个 Node 的 runtime value 是多值容器；
- Graph output 是 structured pytree；
- 一个 pattern 有多个外露 root；
- 一个 HigherOrderOperator Node 持有/引用子图。

四者不能混为“multi-output node”。

## 2. Graph output 的 pytree

`output.args[0]`可以是：

```python
(tensor_a, {"loss": tensor_b, "stats": [tensor_c, 3]})
```

Node 引用仍是 data leaves；tuple/list/dict 结构是函数返回 ABI。Graph codegen/pytree wrapper
决定 flatten/unflatten 边界。

### 为什么结构本身是语义

即使 leaf Tensor 相同，以下返回值也不等价：

```python
(x, y)
{"x": x, "y": y}
(y, x)
```

pass 替换 output 时必须保留 TreeSpec、顺序、key 和常量叶子。

## 3. 单 Node 返回多值与 `getitem`

某些 operator 的 runtime return 是 tuple-like，例如 `native_layer_norm`或 `max.dim`。
FX 常表示为：

```text
multi = call_function[target=...](...)
out0 = call_function[target=operator.getitem](multi, 0)
out1 = call_function[target=operator.getitem](multi, 1)
```

这里：

- `multi`仍是一个 FX Node；
- users 是各个 getitem consumers；
- 只有被读取的 slots 形成后续 use；
- 后端可用 MultiOutput IR 或 extern-kernel output layout 表达共享调用。

## 4. MultiOutputPattern 不是 tuple-return pattern 的同义词

Inductor `MultiOutputPattern`描述一个 connected match 有多个外露 root。第一 output 是用于
候选索引的 `_TargetExpr` root；后续 outputs 可从已绑定 child pattern 的 FX users 中寻找，
也可为 `None`
（`torch/_inductor/pattern_matcher.py:1021-1045`;
`torch/_inductor/pattern_matcher.py:1162-1233`）。

因此它可匹配：

- 一个 tuple-return op 的多个 getitem；
- 多个共享内部子图的独立 output Nodes；
- 一个主输出加一个可选 output。

它不要求某个单 Node 的 runtime value 必须是 tuple。详见
[[pattern_expression_and_matcher_engine_analysis]]。

## 5. HigherOrderOperator 为什么存在

普通 `call_function` target 接收 Tensor/scalar；HigherOrderOperator 接收函数/子程序作为
参数，表达：

- `cond`分支；
- `while_loop` condition/body；
- map；
- checkpoint/recompute region；
- invoke_subgraph；
- effectful region。

当前 `HigherOrderOperator`必须 subclass 创建，并注册在
`torch.ops.higher_order`命名空间；它有独立 dispatch/autograd/functionalization behavior
（`torch/_ops.py:282-345`）。

设计原因是“子程序值”不能当普通 Tensor 参数处理：

- FakeTensor 需要进入子程序推导输出；
- autograd 需要对子图构造梯度；
- functionalization 要递归处理 mutation；
- pass/DCE/lint 可能需要进入 nested graph；
- backend 要么递归编译，要么保留 opaque region。

### 5.1 `map`如何扩展这个模型

`map(f, xs, *args)`把 `xs` pytree 的每个 Tensor 第一维视为 mapped dimension，对每个 slice
运行 body，再 stack 输出。当前 wrapper 会 flatten `xs/args`，拒绝非 Tensor mapped leaves、
零 leading dimension 和不一致的 leading dimension，然后把 body 交给 `map_impl`
（`torch/_higher_order_ops/map.py:111-190`）。

Proxy tracing 不会在 Python 里把 dynamic batch 展开；它从第一片输入构造 body
GraphModule，把 child 注册到 outer tracer，再创建一个 `map_impl` Node
（`torch/_higher_order_ops/map.py:308-340`）。这与 `cond`一样说明：

- outer Node 表示 region invocation；
- body 有独立 placeholder/output/ownership；
- mapped inputs 与 invariant `*args`属于不同签名角色；
- pass 若要改 body，必须显式递归，不能只扫描 outer `Graph.nodes`。

`map`当前仍是 prototype，源码警告其 autograd 支持/误编译风险；本篇不把源码机制说明
写成已通过本机 end-to-end Lab 的结论。

## 6. `torch.cond`

概念模型：

```python
torch.cond(pred, true_fn, false_fn, operands)
```

外层 graph 有一个 cond HOP；两个 branch 是独立可捕获子程序，接收相同 operands。

当前 FakeTensor implementation：

1. flatten true output 与 false output；
2. 要求 TreeSpec 相同；
3. 逐 slot merge output metadata/value；
4. unflatten 为共同结构
   （`torch/_higher_order_ops/cond.py:393-415`）。

Tensor metadata 也必须兼容；当前 merge path使用 `torch._check`检查指定属性一致
（`torch/_higher_order_ops/cond.py:418-449`）。

所以 branch legality 不只是“返回 Tensor 个数一样”，而是 output pytree 与所需 metadata
contract 均兼容。

## 7. `while_loop`

外层 HOP 保持：

- cond function；
- body function；
- loop-carried inputs；
- optional additional inputs。

普通外层 FX use-def 仍无环；循环回边封装在 HOP 的 region semantics 中。当前
`WhileLoopOp`定义与 public wrapper path 在
`torch/_higher_order_ops/while_loop.py:35-133`;
`torch/_higher_order_ops/while_loop.py:227-260`。

这说明“FX Graph 必须有有环 edge 才能表达循环”是错误要求。

## 8. checkpoint/recompute region

checkpoint 可用 HigherOrderOperator 或 metadata 向 partitioner表达“这段 forward computation
允许/要求重算”。但 partition 后真正进入 bw 的 recompute 仍是普通 FX nodes 被复制到
backward graph，而不是 runtime 解释一个抽象 `recompute` Node。

region 是捕获/策略表达；copied nodes 是 partitioned backward implementation。详见
[[10_saved_tensors_recompute_and_runtime_abi]]。

## 9. Node 参数如何引用子图

常见表示有两类：

1. outer GraphModule 持有 child GraphModule，outer graph 用 `get_attr`取到它并传给 HOP；
2. target/argument 中保存可识别的 subgraph module/function object，由 capture/backend
   协议解释。

child graph 的 placeholders只描述 child signature；它不会自动捕获 outer graph 的 Node
作为“自由变量 edge”。outer values必须：

- 显式作为 operands；
- lifted 到 child inputs；
- 或由明确定义的 closure/state protocol 提供。

这保证 ownership：child Node 只引用 child Graph 的 Nodes。

## 10. 分支 signature 与自由变量

安全 branch capture 需要：

- operand 数、顺序和 pytree 一致；
- branch outputs共同 TreeSpec；
- dtype/device/shape/layout 满足 HOP contract；
- mutation/effect 能被 functionalization/token 处理；
- outer captured state被显式 lift 或受控引用；
- symbolic relationship有 guard/assert。

如果 true branch闭包读 parameter，false branch不读，capture仍需把 branch module state 和
signature 规范化，不能直接假设普通 FX users 足以表达。

## 11. 递归 pass 与 ownership 边界

对 outer graph 调用一个 pass，不一定自动处理 child graphs。必须查 stage driver 是否：

- 遍历 `get_attr`引用的 child GraphModule；
- 识别 HOP 中的 subgraph argument；
- 对每个 child 单独 lint/recompile；
- 保留 outer/child signature；
- 防止同一 child 被重复处理。

当前 FX `Graph.eliminate_dead_code()`只递归到：

- owning module 的 named child；
- child 是 `torch.fx.GraphModule`；
- outer graph 有相应 `get_attr` target。

它会对子 graph DCE 并 recompile
（`torch/fx/graph.py:2762-2774`）。未被 outer `get_attr`引用的 module child 会被跳过。

这不是所有 pass 的通用递归规则。

## 源码跟读：`cond` 怎样把两个 Python callable 变成 outer Node 与 child GraphModule

`cond` 是理解 nested graph ownership 的最好入口，因为源码同时展示 child capture、
module registration、outer Node 创建和 branch output 校验。

### 1. 两个 branch 先被各自捕获成独立 GraphModule

Proxy 路径进入 `trace_cond` 后，先检查 operands 类型，再分别调用
`reenter_make_fx(true_fn)` 与 `reenter_make_fx(false_fn)`
（`torch/_higher_order_ops/cond.py:248-255`）。此时已经得到两张带各自 placeholder/output
和 Node ownership 的 child graph。

源码随后读取两个 child 的 output Node，flatten leaves，并先检查 leaf 数量是否一致
（`torch/_higher_order_ops/cond.py:257-274`）。这项检查发生在 outer cond Node 创建之前，
避免先把不满足基本 branch ABI 的 region 写入外图。

### 2. child 不是跨图 Node，而是注册到 outer tracer root 的 GraphModule state

`trace_cond` 为两个 child 分配唯一属性名，并通过 `root.register_module` 注册
（`torch/_higher_order_ops/cond.py:276-285`）。接着它构造
`(pred, true_graph, false_graph, operands)`，递归 unwrap Proxy，最后只在 outer Graph 中创建
一个 `call_function(cond_op, ...)` Node
（`torch/_higher_order_ops/cond.py:287-297`）。

结构关系因而是：

```text
outer GraphModule
├─ attribute true_graph_0  ──► child GraphModule / child Graph
├─ attribute false_graph_0 ──► child GraphModule / child Graph
└─ outer Graph
   └─ call_function(cond_op,
        pred,
        get_attr-like child module references,
        explicit operands)
```

outer operands 通过 cond Node 参数进入 region；child 内部使用自己的 placeholders。不存在
“child Node.args 直接引用 outer Node”的跨 Graph edge。这样每张图都能独立满足 owner 与
producer-before-consumer 不变量，也使后端可以递归编译或将 region 保持 opaque。

### 3. FakeTensor 路径为什么还要再执行两个 branch

Proxy capture 解决“程序结构如何记录”；FakeTensor implementation 解决“cond 这个 Node
产生什么抽象值”。它在 mode 下分别执行两个 branch，flatten 结果并要求 TreeSpec 相同，
再逐 slot merge，最后按共同 TreeSpec unflatten
（`torch/_higher_order_ops/cond.py:393-415`）。

Tensor slot 的 metadata compatibility 通过 `torch._check` 校验指定属性
（`torch/_higher_order_ops/cond.py:418-433`）。因此：

- outer graph 只需要一个 cond Node；
- 两个 child 可以拥有完全不同的内部 Node；
- branch 的可观察输出结构和所需 tensor metadata 必须能合并；
- 仅比较 branch Node 数量或 target 序列没有意义。

这是 HOP 需要独立 dispatch 实现的原因。普通 `call_function` 不会自动进入一个 callable
参数构图，也不知道多个子程序的 output contract。

### 4. `map` 复用同一所有权模型，但 body capture 的样例来自 mapped slice

`trace_map` 禁用外层 proxy tracing，取 `xs` 的第一片作为 example input，再用
`reenter_make_fx(f)` 捕获 body；注释说明使用 first-slice copy 是为了避免遍历 batch 维并
对 symbolic size 产生 guard（`torch/_higher_order_ops/map.py:308-322`）。

body GraphModule 被注册到 outer root 后，代码先取得 fake outputs，再创建一个 outer
`map_impl` Node，最后用 `track_tensor_tree` 绑定 outer abstract result
（`torch/_higher_order_ops/map.py:323-340`）。

这说明“nested graph”不是一种固定控制流形态，而是一个通用所有权协议：

```text
捕获 child signature
→ child 注册为 outer module state
→ outer HOP Node 显式携带 child 与 operands
→ HOP-specific fake/autograd/functionalization/backend 解释 region
```

不同 HOP 的差异在 body/branch signature 与 merge/loop semantics，不在于创建跨图 Node edge。

### 5. outer pass 为什么通常不会自动递归

`Graph.nodes` 只遍历当前 Graph 的侵入式链表。child GraphModule 是 owning module 的属性，
不是 outer 链表中的 Node 序列。因此一个普通 `for node in graph.nodes` pass 天然只处理
outer graph。

FX DCE 是一个明确写出的例外：完成本图逆序删除后，它收集 outer `get_attr` targets，只对
被引用且确为 `GraphModule` 的 named children 递归 DCE，并对 child recompile
（`torch/fx/graph.py:2749-2774`）。

这段实现同时限定了递归边界：

- 它不是扫描 owning module 下所有 child；
- 它依赖 child 通过特定 `get_attr` target 被 outer 引用；
- 它只传播 DCE，不会让其他任意 pass 自动递归；
- child 仍需自己的 lint/签名/HOP-specific verifier。

如果同一 child 被多个 outer site 共享，修改 child 会同时影响所有引用；pass driver 还需
按 GraphModule identity 去重并判断共享修改是否合法。

### 6. structured output、tuple-return op 与 MultiOutputPattern 的源码边界

沿上述 cond 路径可看见三种独立结构：

1. child `output.args[0]` 的 pytree：函数 ABI；
2. 某个 call Node 的 tuple-like runtime result及其 getitem users：单次调用的多值结果；
3. matcher 的多个外露 roots：一个 rewrite match 对外连接的多个 Node。

它们可能在同一图中重合，但没有继承或等价关系。对其中一种结构的合法性检查不能替代另
两种：TreeSpec equality 不证明 MultiOutputPattern users 约束，多个 match roots 也不证明
runtime operator schema 返回 tuple。

## 12. nested graph 的 DCE

要分三层 dead：

1. outer HOP Node 无用户且 pure → whole region可能 dead；
2. branch 内某 Node 无用户且 pure → child-local DCE；
3. 某 branch 整体不可达 → 需要控制流/constant-predicate 特定 pass。

普通 outer `users`无法判断 branch 内部 Node dead；child DCE也不能在不知道 predicate 的
情况下删除另一 branch。

## 13. nested graph 的拓扑与 effect

outer graph拓扑正确不证明：

- branch graph各自拓扑正确；
- branch signatures一致；
- loop-carried values closure正确；
- effect token跨 region正确传递；
- collective 在不同分支的 global order安全。

验证器必须递归到相应 region，并使用 HOP-specific contract。

## 14. 复杂度

设 outer graph `V0/E0`，共有 `G`个 child graphs，第 i 个为 `Vi/Ei`：

- 完整递归线性 pass：`O(V0+E0+Σ(Vi+Ei))`；
- 若 child GraphModule可被多处引用，必须 deduplicate，否则按引用次数重复；
- HOP output merge 与 pytree leaf 数相关；
- nested DCE 还包括每个被引用 child 的 lint/recompile；
- 控制流 reachability/loop analysis可能需要专门 fixed point，不等于普通 DAG traversal。

复杂度不能只报 outer `len(graph.nodes)`。

## 15. 已验证 Lab：tuple/dict 与 cond nested graph

从知识库根目录运行：

```powershell
python -B tools\labs_torch_compile\part1_structured_hop.py
python -B tools\labs_torch_compile\series_artifact_bundle.py `
  --output-dir tools\labs_torch_compile\artifacts\end_to_end
```

正例捕获 tuple→dict 输出与合法 `torch.cond`，并对两个 child GraphModule 分别 lint。错误/
边界例让 true branch 返回 rank-2 clone、false branch 返回 rank-0 reduction；FakeTensor
merge 必须因 branch output metadata 不同而拒绝 export。PyTorch `2.9.1+cpu`实测：

```text
tuple_graph_has_getitem=True
output_is_nested_dict=True
cond_outer_target=cond
cond_subgraphs=true_graph_0,false_graph_0
cond_positive=6.0
cond_negative=-5.0
branch_output_specs_equal=True
```

Lab 分两部分：

- `make_fx`捕获一个 tuple-return op，再以 dict 返回选定 slots；
- `torch.export`捕获 `torch.cond`，列出 outer GraphModule 的两个 child GraphModule 并分别
  lint。
- 自动合同还断言 `hop_invalid_branch_rejected=True`，避免只验证两个合法 predicate 值却
  没有验证 branch contract。

持久 artifact 位于 `tools/labs_torch_compile/artifacts/end_to_end/hop_exported_program.py`、
`model_source.py`与 `model_contract.json`。命令、环境和 HOP 独立变体的原因见
[`tools/labs_torch_compile/README.md`](tools/labs_torch_compile/README.md)。

## 16. 排查清单

1. “多输出”指 runtime tuple、Graph pytree、多个 match roots，还是多个 physical buffers？
2. getitem slot 是否与 operator schema 对应？
3. branch/loop body 是 GraphModule、callable 还是 opaque target？
4. child graph自由变量如何显式进入 signature？
5. pass 是否真的递归，还是只处理 outer graph？
6. child 修改后是否 lint/recompile？
7. DCE 是 outer-local、child-local 还是 control-flow reachability？
8. branch effect/mutation/token contract是否一致？

## 学习顺序

- 上一篇：[[graph_effects_alias_mutation_and_order_analysis]]
- 下一篇：[[graph_capture_frontends_and_tracing_analysis]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[graph_values_metadata_and_signatures_analysis]]
- [[graph_effects_alias_mutation_and_order_analysis]]
- [[pattern_expression_and_matcher_engine_analysis]]
- [[09_aotautograd_joint_forward_backward_graphs]]
- [[10_saved_tensors_recompute_and_runtime_abi]]
- [[dead_code_topology_and_effect_order_analysis]]
