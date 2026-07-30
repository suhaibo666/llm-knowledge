# 12 · FX 改图原语与不变量

> 前置：[[02_fx_graph_core_data_model]]、[[05_graph_effects_alias_mutation_and_order]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. 一次“替换 Node”实际改了什么

最小改图可能涉及：

- intrusive list顺序；
- consumer `args/kwargs`；
- producer `users`与consumer `_input_nodes`；
- lookup side table；
- namespace/name；
- Node owner；
- output/placeholder signature；
- metadata/provenance；
- GraphModule generated forward。

因此只修改pretty-printed graph的一行不是完整操作。

## 2. Mutation-safe traversal

在遍历中插入/删除时先snapshot：

```python
for node in list(graph.nodes):
    ...
```

这段写法把“本轮遍历哪些 Node”固定在进入循环时。若直接遍历仍在变化的 live 链表：

- 新插入Node可能被意外再次处理；
- 删除会改变后续可见的链表状态；
- pass行为可能依赖插入位置，难以推理幂等性。

PatternMatcher同样先snapshot candidates；新root不会在同一次apply被重新发现
（`torch/_inductor/pattern_matcher.py:2627-2645`）。

## 3. Insertion point

使用：

```python
with graph.inserting_before(node):
    new = graph.call_function(...)
```

新producer必须出现在所有consumer之前。`create_node`在当前insert point插入，并更新
namespace、side table与长度（`torch/fx/graph.py:1645-1661`）。

## 4. args/users同步

使用公开setter与replace API；`_update_args_kwargs`负责删除旧reverse uses并添加新reverse uses
（`torch/csrc/fx/node.cpp:307-359`）。

不要直接写 `_args`、`_input_nodes`或 `users`。

## 5. replace 与 erase 的正确顺序

```python
old.replace_all_uses_with(new)
graph.erase_node(old)
```

`replace_all_uses_with`会snapshot users再逐consumer替换
（`torch/fx/node.py:717-744`、`torch/fx/node.py:745-757`）。`erase_node`在仍有users或
跨owner时拒绝（`torch/fx/graph.py:1675-1693`），随后同步list/table/input uses
（`torch/fx/graph.py:1702-1713`）。

若replacement自身引用old，必须用filter callback避免把new的input也替换成new形成self-cycle。

## 6. `node_copy` 与 `graph_copy`

跨图复制必须提供value remap：

```python
env = {}
new = dst.node_copy(old, lambda n: env[n])
env[old] = new
```

`node_copy`通过 `arg_transform` remap输入并浅复制 meta
（`torch/fx/graph.py:2386-2409`、`torch/fx/graph.py:2410-2420`）；`graph_copy`维护
val map、逐 Node 复制并把原 output 映射值返回给 caller
（`torch/fx/graph.py:1525-1551`）。

AOT partition的子图抽取会创建fresh graph、old→new env，并把指定输入变成fresh
placeholder（`torch/_functorch/partitioners.py:514-540`、`torch/_functorch/partitioners.py:541-545`）；
随后按env复制可达Node（`torch/_functorch/partitioners.py:640-659`）。这与 `graph_copy`
是同一类value-remap设计，但不是直接调用同一个helper。

## 7. placeholder 与 output 是ABI

增删/重排placeholder可能改变：

- generated forward签名；
- pytree flatten顺序；
- default args；
- Export/AOT signature位置；
- backend runtime wrapper。

修改output aggregate则要保留对应调用约定中的TreeSpec、mutation/user-output/saved-value/
tangent slots。AOT metadata显式记录forward返回边界
（`torch/_functorch/_aot_autograd/schemas.py:709-730`），Export graph signature也单独记录
`in_spec/out_spec`与backward signature（`torch/_functorch/_aot_autograd/schemas.py:980-999`）。
普通FX Graph API不知道这些上层ABI语义，调用方必须同步companion metadata。

## 8. ownership 与 state target

`get_attr/call_module` target必须存在于owning GraphModule；跨Graph Node引用非法。
`Graph.lint()`检查ownership、拓扑、side table、unique name、opcode和module target
（`torch/fx/graph.py:2610-2637`、`torch/fx/graph.py:2638-2650`、
`torch/fx/graph.py:2652-2678`、`torch/fx/graph.py:2679-2687`）。

构造含state引用的新 `GraphModule` 时，还需把parameter/buffer/submodule安装到新root，不能
只复制Node字符串；构造器会遍历 `get_attr/call_module` target并调用 `_copy_attr`
（`torch/fx/graph_module.py:579-600`），后者按对象类型安装buffer或attribute
（`torch/fx/graph_module.py:283-310`）。

## 9. metadata传播

常见策略：

- 语义等价一对一替换：复制必要meta；
- decomposition一对多：每个new Node记录共同origin与各自val；
- fusion多对一：聚合origin，重新推导output val；
- 新guard/copy/helper：标记生成原因。

`replace_all_uses_with(..., propagate_meta=True)`要求replacement原meta为空，否则拒绝，避免
无声覆盖（`torch/fx/node.py:717-744`）。

meta传播不能替代FakeTensor/shape propagation重新验证。

## 10. lint、DCE、sort、recompile

它们职责不同：

| 操作 | 解决 |
|---|---|
| local erase | 删除确定无users的old Node |
| DCE | 级联删除全图pure/no-user节点 |
| stable topo sort | 修复显式data dependency顺序 |
| lint | 检查结构不变量 |
| recompile | 更新GraphModule forward及PyTree codegen保存的in/out spec |

Pattern replacement不自动运行统一bundle；stage driver决定收尾
（`torch/_inductor/pattern_matcher.py:2609-2637`、
`torch/_inductor/pattern_matcher.py:2640-2669`、
`torch/_inductor/pattern_matcher.py:2670-2699`、
`torch/_inductor/pattern_matcher.py:2700-2726`）。`GraphModule.recompile()`才把编辑后的
graph重新生成Python code，并同步PyTree codegen的in/out spec
（`torch/fx/graph_module.py:924-940`）。

## 11. 推荐事务边界

复杂rewrite应：

1. 先用纯检查收集match与所有legality evidence；
2. 只有全部通过才开始写图；
3. 写图时先创建replacement；
4. reconnect outputs；
5. erase old chain；
6. stage-specific cleanup；
7. lint/recompile；
8. differential tests。

失败后“留半张replacement”比一次不匹配更危险。

## 源码跟读：一次可提交的 rewrite 如何跨越分析、编辑与生效边界

以下以“在 `old` 前建立 replacement，重连所有 users，删除 old”为主线。源码层有四个
独立提交点，不能省略成一个 `replace` 动词。

### 1. insertion context 只改变 `Graph._insert`，退出后恢复

`graph.inserting_before(n)` 先验证 `n.graph is graph`，再返回以 `n.prepend` 为插入函数的
`_InsertPoint`；`inserting_after` 对应 `n.append`
（`torch/fx/graph.py:1716-1738`; `torch/fx/graph.py:1741-1763`）。

这解释了两个常见问题：

- `with` 作用域只控制后续 create APIs 放到哪，不会移动已有 Nodes；
- 在错误 Graph 的 Node 前插入会立即失败，防止 insertion cursor 跨 owner。

若 replacement 需要多个 producer，应在同一 insertion context 内按其内部拓扑顺序创建。
`Graph.create_node` 每次都立即插入链表和 side table；它没有“批量节点尚未提交”的中间态。

### 2. reconnect 是 consumer 参数事务，不是删除事务

`old.replace_all_uses_with(new)` 复制 old users 后逐 consumer 调
`_replace_input_with`（`torch/fx/node.py:737-756`）。底层替换会遍历 consumer 的完整
args/kwargs aggregate，再调用 `_update_args_kwargs` 统一重建 input_nodes/users
（`torch/csrc/fx/node.cpp:368-399`）。

完成时：

```text
consumer.args/kwargs: old → new
old.users: 删除 consumer
new.users: 增加 consumer
Graph node list: old 与 new 都还存在
```

如果 `new.args` 本身引用 `old`，无过滤的 replace 会连 `new` 这个 user 也改掉，形成
`new` 引用自身。`delete_user_cb` 的职责正是把 replacement 自身或其他不该重连的 consumer
排除；这属于 rewrite 规则的语义，不由基础 API 猜测。

### 3. erase 是第二次显式提交

只有 reconnect 后 `old.users` 为空，`erase_node` 才允许执行。它从 lookup table 和双向
链表删除 old、标 erased、减长度，并清空 old 的输入以更新上游 producers
（`torch/fx/graph.py:1675-1693`; `torch/fx/graph.py:1702-1713`）。

因此一个 rewrite 在 replace 与 erase 之间短暂处于“new 已接管输出、old 仍在图中但无
users”的合法结构状态。若 handler 在这里抛异常，Graph 虽可能仍能 lint，却留下 dead old
Node；这就是为什么复杂 pass 要先做纯分析，或在 shadow graph 上编辑后整体 commit。

### 4. 跨图 shadow copy 的提交单位是 Graph，不是逐 Node 回滚

`graph_copy` 顺序遍历源图，用 caller 提供的 `val_map` 跳过已映射 inputs，对其余 Node 调
`node_copy`；遇到 output 时只返回映射后的 output value
（`torch/fx/graph.py:1525-1551`）。`node_copy` 再借 `arg_transform` 建立目标图内 use-def，
创建 fresh Node 并浅复制 meta
（`torch/fx/graph.py:2386-2420`）。

典型 shadow transaction：

```text
source gm.graph（保持不动）
  │ graph_copy / GraphModule state copy
  ▼
shadow GraphModule
  │ 完成全部 rewrite + stage verifier + lint
  ├─ 失败：丢弃 shadow
  └─ 成功：把完整新 Graph/Module 作为提交结果
```

这比试图撤销多次 create/replace/erase 更可推理，但代价是复制 Graph、state 引用和必要
metadata。若 companion signature 或 owning-module state 也改变，shadow transaction 必须
一起复制/验证它们，不能只换 `_graph`。

### 5. `.graph = new_graph` 与原地修改的生效方式不同

`GraphModule.graph` setter 会设置 `_graph`、反向设置 `g.owning_module = self`，并自动
`recompile`（`torch/fx/graph_module.py:669-686`）。而：

```python
gm.graph.call_function(...)
gm.graph.erase_node(...)
```

是在同一个 Graph 对象上原地修改，不经过 property setter，因此不会自动 recompile。

这一区分经常造成“graph print 已变，但调用 gm 仍执行旧 forward”：

| 操作 | Graph 数据结构 | owning_module | generated `forward` |
|---|---|---|---|
| `gm.graph = new_graph` | 替换 | setter 设置 | 自动刷新 |
| 原地编辑 `gm.graph` | 修改已有对象 | 通常不变 | 必须显式 `gm.recompile()` |

### 6. `lint`、DCE、sort、recompile 为什么不能做成每次 API 的自动尾声

局部 create/replace/erase 发生频率很高，而且中间步骤可能暂时存在无 users Node 或尚未完成
的整条 replacement。若每个 API 都自动：

- DCE，可能在 handler 完成连接前删除临时节点；
- topo sort，会改变 pass 正在依赖的 insertion/program order；
- recompile，会反复生成/编译 Python source；
- 做高层 signature/alias verifier，基础 Graph 又不知道所在 stage contract。

所以基础 API 只维护局部结构一致性；pass manager 在规则完成后选择 stage-specific cleanup。
这不是遗漏，而是把“局部数据结构不变量”和“整图/语义有效性”分层。

### 7. 推荐的真实状态机

```text
ANALYZE
  收集 candidates、inputs、outputs、alias/effect/shape 证据
  不写图
    │ all checks pass
    ▼
BUILD
  在合法 insertion point 创建完整 replacement
    ▼
RECONNECT
  只重连计划中的外露 uses / outputs
    ▼
ERASE
  逆依赖或已确认 zero-user 顺序删除 old nodes
    ▼
CLEANUP
  按 stage 选择 DCE / stable topo sort
    ▼
VERIFY
  lint + signature/metadata/alias/effect/autograd checks
    ▼
MATERIALIZE
  recompile 或提交 shadow GraphModule
```

`Graph.lint()`只覆盖 VERIFY 中的结构子集。一个 pass 的原子性要求是：任一语义检查失败时，
对外可见 Graph/companion state 保持原样；基础 FX API 本身不提供数据库式 rollback。

## 12. 复杂度

令待改图有 `V/E`、matched 子图 `K`个节点、被重连 uses 为 `U`、replacement 为 `Q`
个节点：

```text
结构分析/候选收集：依 matcher，至少保留候选与 pattern 参数
创建 replacement：O(Q + replacement 参数引用)
replace uses：O(U + 被修改 consumer 参数树大小)
local erase：O(K + 被清理 input references)
whole-graph DCE/lint/recompile：各自再扫描相应图与生成源码
```

`node_copy/graph_copy`的结构成本与 selected nodes/uses 线性；copy 的 state、metadata payload
或自定义 arg transform 成本另计。事务式 shadow copy 会增加 `O(V+E)`时间/空间，但能把
拒绝/异常前的写操作隔离。没有 candidate、fan-out 与 replacement 分布时，期望成本未定义。

## 13. 已验证 Lab

从知识库根目录运行：

```powershell
python -B tools\labs_torch_compile\part1_fx_core.py
python -B tools\labs_torch_compile\part3_passes.py
python -B tools\labs_torch_compile\part3_end_to_end_pass.py `
  --output-dir tools\labs_torch_compile\artifacts\part3
```

覆盖关系：

- `part1_fx_core.py`：create/replace/erase/recompile，边界为 live erase 与 cross-owner lint；
- `part3_passes.py`：逐 Node `node_copy`、`graph_copy`、错误拓扑与 stable repair；
- 贯穿 pass：在 shadow GraphModule 上分析/编辑，合法时一次 commit；broadcast reject 时
  code 与 node meta 都必须不变。

实测关键字段：

```text
node_copy_value_matches=True
graph_copy_value_matches=True
lint_failed_before_sort=True
topology_repaired_value=8.0
failure_atomicity_matches=True
second_run_modified=False
```

Part III 持久 artifact 位于 `tools/labs_torch_compile/artifacts/part3/`：`legal_before.py`、
`legal_after.py`、两份 illegal graph、`results.json`与 `manifest.json`。自动合同
`EditingAndPassManagerContractTest`和 `EndToEndPassContractTest`做 assertions。完整命令、
环境与正/边界案例见 [`tools/labs_torch_compile/README.md`](tools/labs_torch_compile/README.md)。

## 14. 检查清单

1. traversal是否snapshot？
2. insertion point是否保证producer先于consumer？
3. args/users是否经API同步？
4. replacement是否错误替换自己的input？
5. old Node是否仍有users？
6. owner与state target是否有效？
7. placeholder/output ABI是否同步？
8. meta是传播还是重新推导？
9. stage是否要求DCE/sort/lint/recompile？
10. alias/mutation/shape/autograd是否验证？

## 学习顺序

- 上一篇：[[11_graph_stage_boundaries_identity_and_provenance]]
- 下一篇：[[13_pattern_expression_and_matcher_engine]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[02_fx_graph_core_data_model]]
- [[13_pattern_expression_and_matcher_engine]]
- [[14_dead_code_topology_and_effect_order]]
- [[16_graph_rewrite_legality_validation_and_complexity]]
