# 02 · FX Graph 的核心数据模型

> 前置：[[01_graph_ir_motivation_and_taxonomy]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. 核心结论

FX 没有单独的 `Edge` 对象。它用两套互补结构保存一段程序：

```text
程序顺序：root sentinel <-> Node <-> Node <-> ... <-> root
数据依赖：consumer.args/kwargs 中直接引用 producer Node
反向索引：producer.users 记录使用它的不同 consumer Node
```

这使得“遍历程序”“查 producer”“查 users”“局部替换”都足够直接，同时避免每次修改参数
时再同步一组独立 Edge 实体。

## 2. 五个核心对象

### 2.1 Graph

`Graph` 是 FX IR 的容器；其 Node 列表共同构成一个有效 Python 函数
（`torch/fx/graph.py:1397-1443`）。当前构造状态包含：

- `_root`：侵入式链表 sentinel；
- `_insert`：当前插入函数；
- `_len`：节点数；
- namespace：生成唯一 name；
- owning `GraphModule`；
- `_FindNodesLookupTable`：按 opcode/target 查候选的 side table
  （`torch/fx/graph.py:1445-1465`）。

### 2.2 Node

Node 既表示一个操作 callsite，也代表该操作产生的值。六种普通 opcode 的语义来自
`torch/fx/node.py:258-284`：

| op | 含义 | target |
|---|---|---|
| `placeholder` | 图输入 | 参数名 |
| `get_attr` | 从 module 层级读取 state/attribute | qualified name |
| `call_function` | 调用自由函数/operator | callable |
| `call_module` | 调用子 module.forward | module qualified name |
| `call_method` | 调用对象方法 | method name |
| `output` | 函数 return | `"output"` |

Node 的 `name` 是图内 SSA-like value 名；`target` 是被调用实体，两者不能混淆。

### 2.3 GraphModule

`GraphModule` 把 `Graph` 与 parameter/buffer/submodule/attribute root 组合成可执行
`nn.Module`。它根据 graph 生成 Python `forward`
（`torch/fx/graph_module.py:517-528`;
`torch/fx/graph_module.py:924-1008`）。

赋一个新 `.graph` 会自动 recompile；原地修改已有 `gm.graph` 后必须显式
`gm.recompile()`。

### 2.4 Proxy

`Proxy` 是 tracing 时对 Node/value 的 Python 代理。对 Proxy 做运算，不是立即计算真实
Tensor，而是请求 Tracer 创建新 proxy/node。当前 `Proxy` 的 Python operator、
attribute/method 和 `__torch_function__` 路径最终调用 `create_proxy`
（`torch/fx/proxy.py:600-635`;
`torch/fx/proxy.py:680-705`;
`torch/fx/proxy.py:710-731`;
`torch/fx/proxy.py:732-743`;
`torch/fx/proxy.py:757-780`;
`torch/fx/proxy.py:781-806`）。

Proxy 属于“构图期间的用户接口”，Node 属于“图内存储结构”。

### 2.5 Tracer

Tracer 决定：

- 哪个 module/function 是 leaf；
- 参数如何变成 placeholder；
- Python 调用如何映射成 opcode/target；
- concrete argument 与控制流如何处理；
- root module 的 state 如何用 `get_attr` 表达。

`symbolic_trace` 调用 `Tracer.trace`，然后把结果包装成 `GraphModule`
（`torch/fx/_symbolic_trace.py:1413-1421`）。

## 源码跟读：一张 FX 图怎样从 Python 执行中长出来

这一节不把 `Graph`、`Node`、`Proxy`、`Tracer` 当作五个并列名词，而是沿一次真实调用把它们串起来。
先记住各层的职责边界：

| 层 | 主要对象 | 负责什么 | 不负责什么 |
|---|---|---|---|
| 用户入口 | `symbolic_trace` | 选择默认 `Tracer`，把 `Graph` 包装成 `GraphModule` | 不逐个解释 Python 运算 |
| 追踪执行 | `Tracer.trace` | 准备 placeholder/patch 环境，用 Proxy 执行目标函数并记录 output | 不直接维护链表和 users |
| 记录接口 | `create_proxy` / `create_node` | 把 Python 参数降成 FX `Argument`，创建 Node，补 scope/meta | 不生成最终 `forward` |
| IR 容器 | `Graph.create_node` | 分配名字、构造 Node、插入顺序、维护查询表和长度 | 不执行 Tensor 数值计算 |
| 可执行封装 | `GraphModule` / `recompile` | 从 Graph 生成并安装 Python `forward` | 不自动证明改图语义等价 |

### 第一步：入口只做“选择 Tracer、追踪、封装”

`symbolic_trace` 的主体非常短：构造默认 `Tracer`，调用 `trace` 得到 `Graph`，最后用
`_make_graph_module` 把 tracer root 与 graph 组合成 `GraphModule`
（`torch/fx/_symbolic_trace.py:1413-1421`）。

这解释了为什么“FX Graph”和“可运行的 FX Module”是两个层次：

```text
Python function / nn.Module
        │
        ▼
Tracer.trace(...) ─────────► Graph
                                │
root module/state ──────────────┤
                                ▼
                           GraphModule
```

`Graph` 只保存 IR；parameter、buffer、submodule 等可寻址状态仍由 root/`GraphModule` 持有。
因此 `get_attr`、`call_module` 的 `target` 是 qualified name，而不是把真实 module 或
parameter 复制进 Node。

### 第二步：`Tracer.trace` 先创建空图，再让目标函数在 Proxy 上“跑一遍”

`Tracer.trace` 在 module 与普通 callable 两种入口间确定 `root` 和待追踪函数，随后创建
`Graph(tracer_cls=...)`（`torch/fx/_symbolic_trace.py:801-828`）。它再由
`create_args_for_root` 按函数签名枚举参数；未具体化的输入最终通过 `create_proxy` 成为
placeholder Proxy（`torch/fx/_symbolic_trace.py:717-728`;
`torch/fx/_symbolic_trace.py:1006-1019`）。

真正容易误解的一行在 trace 尾部：

```python
self.create_node(
    "output",
    "output",
    (self.create_arg(fn(*args)),),
    {},
    type_expr=ann.get("return", None),
)
```

这里确实调用了 `fn(*args)`，但 `args` 中承载的是 Proxy。Proxy 的普通 Python
magic method 会被安装成 `create_proxy("call_function", ...)` 调用
（`torch/fx/proxy.py:921-933`）；module call 与 wrapped function 则由 trace 建立的 patch
环境接管。函数返回值再由 `create_arg` 转成 Node/aggregate，最终成为
`output.args[0]`（`torch/fx/_symbolic_trace.py:890-917`）。所以 symbolic trace 的本质
不是“解析 Python 源码生成 AST”，而是“用代理值执行可追踪路径，并记录代理值参与的操作”。

这个机制也给出其边界：如果 Python 控制流必须读取 Proxy 对应的真实数据，单靠默认
symbolic tracing 无法决定分支；`concrete_args` 只能把指定输入专门化，不能让任意
data-dependent Python 控制流自动变成图。

### 第三步：`create_proxy` 先把 Python 层参数降为 FX 参数，再创建 Node

`TracerBase.create_proxy` 先对 `args`、`kwargs` 调用 `create_arg`，确认结果满足 FX 的
tuple/dict 参数约定，然后调用 `create_node`，最后把 Node 包回 Proxy
（`torch/fx/proxy.py:340-374`）。这一步承担 Python 世界到 FX `Argument` 世界的边界转换：

```text
Python object / Proxy / nested container
              │ create_arg
              ▼
常量或嵌套 FX Argument（Proxy 被换成其 Node）
              │ create_node
              ▼
Node
              │ proxy(node)
              ▼
新的 Proxy，继续参与 Python 执行
```

`TracerBase.create_node` 随后转发到 `self.graph.create_node`，并在新 Node 上补充 module
scope、stack trace、autograd 标记等来源信息
（`torch/fx/proxy.py:215-242`; `torch/fx/proxy.py:244-283`）。因此：

- Node 的核心 use-def 结构由 `Graph`/`Node` 维护；
- tracing provenance 主要由 Tracer 写入 Node `meta`；
- 自定义 Tracer 可以覆写 `create_node` 增加策略，但不能因此绕过 Graph 的结构不变量。

### 第四步：`Graph.create_node` 同时提交四类结构状态

空 `Graph` 初始化 `_root` sentinel、当前插入函数、长度、namespace、owner、codegen 与
find-nodes side table（`torch/fx/graph.py:1445-1465`）。一次
`Graph.create_node` 不是单纯 `Node(...)`，而是一笔结构提交：

| 次序 | 源码动作 | 被更新的状态 |
|---:|---|---|
| 1 | 规范化并检查 `args/kwargs` | Node 参数表示 |
| 2 | namespace 生成唯一名字 | 名字空间 |
| 3 | `Node(self, ..., args, kwargs, ...)` | Node 本体、input_nodes、producer.users |
| 4 | `_insert(n)` | 程序顺序双向链表 |
| 5 | lookup table `insert(n)` | 按 op/target 的候选索引 |
| 6 | `_len += 1` | Graph 节点计数 |

对应实现分别位于 `torch/fx/graph.py:1619-1647` 与
`torch/fx/graph.py:1649-1661`。其中第 3 步会经 Node 参数更新逻辑建立 use-def；后面三步
才把 Node 放入 Graph 的顺序和辅助索引。

这也是 Graph 不设计独立 `Edge` 对象的关键原因：数据边已经由 consumer 的嵌套
`args/kwargs` 表达，反向查询由 producer 的 `users` 缓存。如果再创建一套 Edge 实体，
每次参数替换都要同步“参数树、Edge 集合、users”三份关系，增加新的失配状态；当前设计
只需让 Node 参数 setter 成为 use-def 的一致性边界。

### 第五步：Node 为何同时需要 `_input_nodes` 与 `users`

底层 `NodeBase` 同时保存链表指针、owner graph、op/target、参数、`_input_nodes`、`users`
和 meta（`torch/csrc/fx/node.cpp:154-171`）。它们表达两个方向：

```text
consumer._input_nodes  ──►  distinct producers
producer.users         ──►  distinct consumers
consumer.args/kwargs   ──►  带位置、嵌套结构和重复次数的真实 uses
```

`_update_args_kwargs` 先从旧 producer 的 `users` 删除当前 consumer，再清空
`_input_nodes`；随后单次遍历新参数树，同时完成容器不可变化、重建 `_input_nodes`、
回填各 producer 的 `users`（`torch/csrc/fx/node.cpp:318-354`）。

这里故意把 `users` 做成 distinct consumer 集合，而不是 use 次数表。`x + x` 的两个位置
信息仍完整保存在 `add.args`；`x.users` 只回答“谁消费 x”。若算法需要 use multiplicity，
它必须查看 consumer 参数，而不能把 `len(users)` 当作边数。这一取舍让替换、DCE 和
“是否只有一个 consumer”查询直接，但不免费提供精确 use 次数。

## 源码跟读：一次局部改图究竟改变了什么

设原图中 `old` 被 `u1`、`u2` 使用，要用 `new` 替换它。正确流程不是“改一个指针后旧节点
自动消失”，而是三个显式阶段：

```text
1. replace uses       2. erase old          3. validate / materialize
old.users = ∅         old 离开链表/索引      lint 检查结构
new.users += u1,u2    old inputs 解除 use     recompile 更新 forward
```

### `replace_all_uses_with`：改 use-def，不改程序顺序

实现先复制 `self.users`，再逐个 user 调用 `_replace_input_with`
（`torch/fx/node.py:737-756`）。复制是必要的，因为替换每个 user 输入时，`self.users`
本身会同步缩小。完成后：

- consumer 的 `args/kwargs` 中 `old` 被换成 `new`；
- `old.users` 移除这些 consumer；
- `new.users` 增加这些 consumer；
- `old` 和 `new` 在 Graph 链表中的位置都没有因此改变；
- `old` 仍是图中 Node，除非随后显式 erase。

`propagate_meta=True` 只有在 replacement 的 meta 为空时才允许执行
（`torch/fx/node.py:737-744`）。这不是繁琐限制：盲目覆盖已有 meta 会把 replacement
自己的来源或分析结果伪装成 old 的信息。

### `erase_node`：只有零 user Node 才能从 Graph 容器退出

`Graph.erase_node` 先拒绝仍有 users、owner 不匹配或已经被 erase 的情形
（`torch/fx/graph.py:1675-1693`）。成功路径再：

1. 从 find-nodes side table 删除；
2. 从双向链表摘除并标记 `_erased`；
3. `_len -= 1`；
4. 把该 Node 的输入清空，使其 producers 的 `users` 也解除对它的记录
   （`torch/fx/graph.py:1695-1713`）。

最后一步解释了“零 user”为什么仍不足以把删除视为纯链表 `O(1)`：待删 Node 自己可能有
很大的嵌套参数树，清空输入仍要维护它对上游 producer 的反向索引。

### `lint` 与 `recompile`：一个验证结构，一个刷新执行体

`Graph.lint` 顺序扫描 Node，用 `seen_values` 检查 owner 与定义先于使用，同时核对 opcode、
side table 和唯一 name（`torch/fx/graph.py:2620-2650`）；若有 owning module，还解析
`get_attr`/`call_module` target 并检查对象类型
（`torch/fx/graph.py:2652-2687`）。

`GraphModule.recompile` 则从 graph 生成 Python code、保存源码和行号映射，并把新函数安装
为该 GraphModule class 的 `forward`（`torch/fx/graph_module.py:932-948`;
`torch/fx/graph_module.py:990-1003`）。它不修改 Graph 拓扑，也不会自动做 DCE、拓扑排序
或语义验证。

所以常见的改图收尾：

```python
gm.graph.lint()
gm.recompile()
```

只表示“结构成立且执行体已刷新”。它不等价于“数值、shape、alias、effect、RNG 和梯度都
已经证明等价”；这些必须由 pass 所在阶段的额外分析与测试承担。

### 状态变化总表

| API | 顺序链表 | args/kwargs | producer.users | side table / len | 生成的 forward |
|---|---|---|---|---|---|
| `create_node` | 插入 | 初始化 | 增加 consumer | 插入 / `+1` | 不变 |
| `replace_all_uses_with` | 不变 | 修改 users 的输入 | old 减、new 增 | 不变 | 不变 |
| `erase_node` | 摘除 | 清空被删 Node 输入 | 上游 producers 移除它 | 删除 / `-1` | 不变 |
| `lint` | 只读 | 只读 | 间接依赖其一致性 | 核对 | 不变 |
| `recompile` | 只读 | 只读 | 只读 | 只读 | 重新生成并安装 |

这一表也回答“替换后是否自动通用重排序”：不会。局部 API 只维护自己负责的结构；
是否 DCE、是否 stable topological sort、何时 recompile，是 pass pipeline 的显式策略，
分别在第 14、15 篇展开。

## 3. 程序顺序：侵入式双向链表

Node 自己保存前驱/后继指针；Graph 不维护 Python list 作为权威存储。root sentinel
把首尾连接起来，因此插入和删除已知位置的节点可以是常数级链表操作。

Graph 创建 root、插入 cursor、长度和 side table 的当前实现见
`torch/fx/graph.py:1445-1465`；底层 Node 链表初始化和 prepend/append 在
`torch/csrc/fx/node.cpp:154-205`。

### 为什么不是普通 list

pass 经常在某 Node 前后插入多个节点。若权威结构是数组，中间插入会移动后缀；
侵入式链表提供：

- 稳定 Node 对象身份；
- `inserting_before/after` 的局部插入；
- O(1) 链接/摘除；
- 直接定义“当前图顺序”。

代价是随机下标访问不自然；FX pass 通常更需要顺序扫描和局部编辑。

## 4. 数据依赖：嵌套 args/kwargs 中的 Node 引用

`args` 与 `kwargs` 可以是嵌套 tuple/list/dict/slice，叶子可以包含 Node 或常量。consumer
直接引用 producer Node；这就是 FX 的数据边。

Node 还缓存：

- `_input_nodes`：所有不同的 Node-valued input；
- `users`：使用当前 Node 的不同 consumer，使用 dict 模拟 ordered set
  （`torch/fx/node.py:286-304`）。

### user 数不等于 use 数

对于 `y = x + x`：

- `add.args` 中有两次 `x`；
- `x.users` 里 `add` 只出现一次。

因此：

- `len(x.users)`回答“多少个不同 consumer”；
- 要统计边/use 次数，必须遍历 consumer 的嵌套参数。

Pattern 的 `_users=1`约束通常也是 distinct user 数，不是参数出现次数。

### 为什么需要反向 users

若只有 consumer→producer 参数引用，找 producer 很快，但找所有 consumer 要扫描整图。
`users`让 DCE、replace、liveness 和 pattern user-count 可快速访问反向关系。

## 5. 更新参数为何必须走 Node API

当 `args/kwargs` 改变时，必须同时：

1. 删除旧 producer 的 `users`；
2. 扫描新嵌套参数；
3. 构造新的 `_input_nodes`；
4. 将当前 Node 加入新 producer 的 `users`。

当前 C++ backed `_update_args_kwargs`正是做这件事
（`torch/csrc/fx/node.cpp:307-359`）。直接篡改私有字段会让正向引用和反向索引分裂。

推荐使用：

- `node.args = ...` / `node.kwargs = ...`；
- `node.replace_input_with(old, new)`；
- `old.replace_all_uses_with(new)`。

## 6. create、replace 与 erase

### 6.1 create_node

`Graph.create_node()`：

1. 校验/归一化 args/kwargs；
2. 生成唯一 name；
3. 构造 Node；
4. 在当前 insertion point 插入；
5. 更新 namespace、side table 与长度
   （`torch/fx/graph.py:1585-1661`）。

`graph.call_function`、`call_module` 等是面向相应 opcode 的便利包装。

### 6.2 replace

`replace_all_uses_with`先 snapshot users，再逐个 consumer 更新输入，所以不会在遍历同一
动态 users 集时漏项；也可选择传播 meta
（`torch/fx/node.py:717-757`）。

替换 use 后，旧 Node 仍存在。只有其 users 为空且删除合法时才能 erase。

### 6.3 erase_node

`erase_node`拒绝：

- 仍有 users 的 Node；
- 属于另一张 Graph 的 Node。

成功时从 side table 和链表移除、标记 erased、减长度，并清空输入引用以更新 producers'
users（`torch/fx/graph.py:1675-1713`）。

## 7. ownership：Node 只能引用本图 Node

Node 持有 `graph` owner。把另一张 Graph 的 Node 直接塞入 args 会破坏：

- ownership；
- 拓扑顺序；
- users 与 side table；
- GraphModule state target 的解析。

跨图复制必须建立 old→new 环境，并通过 `node_copy`/`graph_copy` 对输入 Node 做 remap。
AOT partitioner 构造 fw/bw 时正是这一模式。

## 8. `_FindNodesLookupTable`

Graph 的 side table支持按 root opcode/target 快速取候选：

- `call_function`按 `(op, target)`索引；
- 其他 ordinary op 多按 op 索引，再按 target 过滤
  （`torch/fx/graph.py:1360-1393`;
  `torch/fx/graph.py:1497-1522`）。

这也是 PatternMatcher 不必对“每个 graph node × 每条 pattern”做全笛卡尔积的基础。
任何 create/erase 都必须同步 side table；`lint()`会检查 Node 是否在其中。

## 9. lint 检查什么，不检查什么

`Graph.lint()`当前检查：

- Node opcode 合法；
- Node ownership；
- Node 在 lookup side table；
- producer 出现在 consumer 前；
- name 唯一；
- owning GraphModule 下 `get_attr/call_module` target 存在且类型合理
  （`torch/fx/graph.py:2610-2687`）。

它不证明：

- 数值等价；
- shape/dtype/layout 合法；
- effect/RNG/collective 顺序正确；
- alias/mutation 语义不变；
- 动态 shape guard 足够；
- autograd 梯度正确。

`lint()`是结构不变量检查，不是完整语义 verifier。

## 10. recompile：图数据结构到可执行 Python

原地改 `gm.graph`后，已有 `gm.forward`仍是旧生成代码。`gm.recompile()`重新生成并安装
forward；生成源码还通过 loader 注册进 `linecache`，便于 traceback 和 inspect
（`torch/fx/graph_module.py:80-115`;
`torch/fx/graph_module.py:136-165`;
`torch/fx/graph_module.py:924-1008`）。

因此典型收尾顺序是：

```python
gm.graph.lint()
gm.recompile()
```

是否先做 DCE 或 stable sort 由所在 pass stage 决定，不是每次局部 edit 的通用自动行为。

## 11. 复杂度

设 `V` 为 Node 数，`E_use` 为嵌套参数中的 Node use 次数：

| 操作 | 典型复杂度 | 说明 |
|---|---:|---|
| 顺序遍历 | `O(V)` | 链表扫描 |
| 已知位置插入/摘除 | `O(1)` 链表部分 | 还需更新参数/users/side table |
| 更新 args/kwargs | `O(old_args + new_args)` | 遍历嵌套 aggregate |
| 查 distinct users | `O(1)`取集合，`O(U)`遍历 | 不含重复 use |
| replace all uses | `O(U + 被改参数大小)` | 每个 consumer 更新输入 |
| find_nodes | 接近候选桶大小 | side table 降低整图扫描 |
| lint | `O(V + E_distinct + target path)` | ownership/topology/target 检查 |
| recompile | `O(V + 生成源码大小)` | 还包含 Python compile |

链表 `O(1)`删除不代表整个 `erase_node`与语义清理都是 O(1)；输入引用清空仍要更新 users。

## 12. 已验证 Lab：构造、替换、破坏与修复

从知识库根目录运行：

```powershell
python -B tools\labs_torch_compile\part1_fx_core.py
```

最小正例手工构造 `x+x`再乘常数，完成 replace、erase、lint 与 recompile；两个故意失败的
边界分别是擦除仍有 user 的 producer，以及把另一张 Graph 的 Node 放入当前 Node 参数。
两者都必须抛 `RuntimeError`，而不是继续生成损坏代码。实测要点：

```text
x_distinct_users=1
x_uses_in_add=2
before_replace=8.0
erase_live_node=RuntimeError
after_replace=6.0
lint_cross_graph=RuntimeError
```

模型：

```python
def f(x):
    y = x + x
    return y * 2
```

Lab 证明：

- `x + x`只有一个 distinct consumer，却有两个 use；
- live producer 不能直接 erase；
- replace 后必须 erase old node、lint、recompile；
- 跨图 Node 引用会被 lint 拒绝。

该小脚本的 artifact 是 stdout；可用 `Tee-Object`保存为
`tools/labs_torch_compile/artifacts/logs/part1_fx_core.txt`。同一核心数据结构在贯穿模型中的持久图位于
`tools/labs_torch_compile/artifacts/end_to_end/symbolic_fx.py`。环境、完整命令与证据等级见
[`tools/labs_torch_compile/README.md`](tools/labs_torch_compile/README.md)。本 Lab 验证结构不变量，不声称验证 alias/effect
语义；这些属于第 05、16 篇的独立 verifier。

## 13. 实战检查清单

每次改图至少检查：

1. 是否在 snapshot 上遍历，避免边遍历边插入导致遗漏？
2. 新 Node 是否位于所有 consumer 之前？
3. 是否用公开 args/replace API 同步 users？
4. 是否传播必要 meta，但不盲目覆盖已有 meta？
5. 旧 Node 是否已经无 users？
6. 所有 Node 是否属于同一 Graph？
7. state target 是否存在于 owning GraphModule？
8. 是否需要 DCE、stable sort、lint、recompile？
9. 是否另有 alias/mutation/effect/shape/autograd verifier？

## 学习顺序

- 上一篇：[[01_graph_ir_motivation_and_taxonomy]]
- 下一篇：[[graph_values_metadata_and_signatures_analysis]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[01_graph_ir_motivation_and_taxonomy]]
- [[graph_values_metadata_and_signatures_analysis]]
- [[fx_graph_editing_primitives_and_invariants_analysis]]
- [[dead_code_topology_and_effect_order_analysis]]
- [[pattern_expression_and_matcher_engine_analysis]]
- [[04_export_and_distributed/01_fx_export_extensibility/index]]
