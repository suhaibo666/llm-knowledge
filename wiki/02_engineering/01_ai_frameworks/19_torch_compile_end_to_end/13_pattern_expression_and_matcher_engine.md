# 13 · PatternExpr 与 PatternMatcher 引擎

> 前置：[[12_fx_graph_editing_primitives_and_invariants]]、[[06_structured_outputs_higher_order_and_nested_graphs]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. Pattern 定义了什么

PatternExpr是对FX候选子图的**递归predicate与capture AST**。它不是FX Node、不是可执行
Graph，也不是另一个backend IR。

它描述：

- root `op/target`；
- nested args/kwargs结构；
- child value继续匹配何种pattern；
- constants；
- user-count与sharing；
- 哪些values传给handler。

抽象base `PatternExpr.match()`建立context并调用polymorphic `_match`
（`torch/_inductor/pattern_matcher.py:526-558`）。

## 2. 为什么要有子类

图中不同维度需要不同匹配语义：

| 需求 | 子类族 |
|---|---|
| 任意输入并capture | `Arg` |
| 命名handler kw capture | `KeywordArg` |
| wildcard且不传handler | `Ignored` |
| exact value/target | constant/target expr |
| op+target+args+kwargs | `CallFunction/Method/Module` |
| 重复参数集合 | List/repeated expr |
| 多个外露roots | `MultiOutputPattern` |

当前主要声明位于 `torch/_inductor/pattern_matcher.py:526-583`,
`torch/_inductor/pattern_matcher.py:745-788`,
`torch/_inductor/pattern_matcher.py:791-1019`,
`torch/_inductor/pattern_matcher.py:1058-1233`。

## 3. Arg、KeywordArg、Ignored

- `Arg` capture任意 NodeOrConstant，按depth-first pattern traversal顺序加入handler positional
  args；不是“target函数第n个形参”。
- `KeywordArg("q")`把匹配值放入handler keyword `q`；它本身可位于pattern positional或
  keyword结构。
- `Ignored`匹配任意值但不进入handler ABI。

源码：`torch/_inductor/pattern_matcher.py:561-583`;
`torch/_inductor/pattern_matcher.py:745-762`。

## 4. CallFunction

`CallFunction(target, *arg_patterns, **kwarg_patterns)`先检查candidate opcode/target，再对
normalized argument structure递归匹配。`_TargetArgsExpr`可利用operator schema补default
kwargs，flatten aggregate，递归pattern leaves并精确比较constants
（`torch/_inductor/pattern_matcher.py:876-1019`;
`torch/_inductor/pattern_matcher.py:1058-1064`）。

## 5. AST如何表达DAG sharing

pattern语法看似tree，但 `MatchContext.pattern_to_node`按PatternExpr对象identity绑定。
同一个pattern实例复用两次，要求两处匹配到同一FX value。

```python
x = KeywordArg("x")
pattern = CallFunction(torch.ops.aten.add.Tensor, x, x)
```

这表达shared input。创建两个结构相同的KeywordArg对象不要求同一Node。
binding与user constraint见 `torch/_inductor/pattern_matcher.py:483-523`;
`torch/_inductor/pattern_matcher.py:841-861`。

## 6. users constraint

TargetExpr可要求expected distinct user count，避免把有外部consumer的intermediate错误删掉/
融合。output pattern和 `_users=MULTIPLE`可放宽普通exact-user约束。

user数是distinct consumer数，不是args出现次数；这继承FX `Node.users`语义。

## 7. MatchContext 与失败

普通top-level `match()`创建fresh context，失败尝试随context丢弃。
MultiOutput在alternative anchor之间snapshot/restore `pattern_to_node`
（`torch/_inductor/pattern_matcher.py:534-538`;
`torch/_inductor/pattern_matcher.py:1206-1223`）。

不要把这个局部rollback推广成“任意handler图修改都transactional”。handler开始mutation后
仍应自行保证失败原子性。

## 8. MultiOutputPattern

第一output必须是TargetExpr并作为candidate index root。匹配root后，后续outputs沿已绑定
child patterns的FX users寻找；slot也可为None
（`torch/_inductor/pattern_matcher.py:1021-1045`;
`torch/_inductor/pattern_matcher.py:1162-1233`）。

它表示connected subgraph的多个外露roots，不等于tuple-return op。

## 9. Candidate index 与逆序

registration按root `(pattern.op, target)`分桶。apply向Graph lookup table请求有注册root的
candidates，合并后按Node顺序逆序处理
（`torch/_inductor/pattern_matcher.py:2583-2656`）。

因此不是每个Node尝试每条pattern。逆序只表示later roots先处理：

- 不创建reverse graph；
- 不指AOT backward；
- 新replacement root不在当前snapshot，不会同轮再匹配。

## 10. 单次apply不是fixed point

每个candidate按registration order尝试bucket entries；root被erase后跳过剩余entry。apply
不restart，也不发现new roots
（`torch/_inductor/pattern_matcher.py:2641-2726`）。

重复由上层driver显式控制，例如pre-grad counter、ordered matcher lists或PassManager steps。

## 11. 三类Entry

- `LoweringPatternEntry`不会在 `apply()` 中直接调用 lowering handler。它先把绑定了
  `Match` 的 handler 作为新 `call_function` 插入，转移 root meta、替换 uses，再清除可删
  match；`register_lowering_pattern()`给 handler 标记
  `_inductor_lowering_function`，后续 `GraphLowering` 才以 Inductor IR values 调用它
  （`torch/_inductor/pattern_matcher.py:1373-1385`;
  `torch/_inductor/pattern_matcher.py:2296-2328`）。
- `GraphPatternEntry`只在 `graph.inserting_before(root)`上下文调用 handler；不会自动替换
  root、重连 uses 或清理旧节点，custom graph surgery及失败原子性都归 handler
  （`torch/_inductor/pattern_matcher.py:1388-1399`）。
- `ReplacementPatternEntry`要求 `Match.replacement_graph`已经由 traced-replacement 的
  `check_fn`建立；`apply()`按 `normalize_args`解释并复制 replacement GraphModule，
  reconnect outputs，最后做matched-node与dead replacement-node的local cleanup
  （`torch/_inductor/pattern_matcher.py:1401-1663`;
  `torch/_inductor/pattern_matcher.py:1878-2059`）。

`Match.erase_nodes()`仅逆序删除matched且users已空的Nodes
（`torch/_inductor/pattern_matcher.py:295-300`）。

## 12. traced replacement为何二次trace

generic trace构造广匹配pattern；命中后用matched fake values重建arguments，必要时trace
shape-specific search，再验证extra_check，最后trace replacement
（`torch/_inductor/pattern_matcher.py:1828-2092`）。

这是为了防止generic pattern忽略的shape/scalar细节授权非法replacement。

## 13. Serialized/precompiled pattern 为什么存在

对大型 traced replacement，每次进程启动重新 trace/search pattern 会增加 import/compile
延迟。当前实现可以把 PatternExpr 序列化成 Python：设置 `PYTORCH_GEN_PATTERNS` 时生成
serialized module，普通路径则按 search function 与 unique name 导入 precompiled pattern
（`torch/_inductor/pattern_matcher.py:2095-2224`）。这一入口在缺少目标属性时只记录 warning，
随后仍执行 `getattr`；它**没有**在该处自动回退到 tracing path。

它缓存的是“如何构造/注册 pattern”的编译产物，不改变运行时匹配语义：

- root bucket、candidate snapshot、sharing identity 与 handler ABI 仍由当前 matcher执行；
- cache key/版本必须覆盖会改变 graph/pattern 的 config、device/dtype/shape specialization；
- serialized AST 只保留可编码结构；任意 Python `extra_check`、handler side effect 与
  replacement trace 仍可能成为外生成本；
- 不能因为 pattern 来自 cache 就跳过 mutation/stream/mempool 与 legality checks。

## 14. 结构命中后的安全层

engine在apply前检查mutation region、stream、mempool，并在 `guard_or_false`下执行
`extra_check`
（`torch/_inductor/pattern_matcher.py:2622-2710`）。

Pattern AST本身不证明shape、dtype、layout、alias、effect、autograd或numerical equivalence。

## 源码跟读：一棵 PatternExpr AST 怎样在 FX DAG 上递归命中

先用一个具体 AST 固定语义：

```python
x = KeywordArg("x")
p = CallFunction(
    torch.ops.aten.mul.Tensor,
    CallFunction(torch.ops.aten.add.Tensor, x, x),
    Ignored(),
)
```

它不是在定义一段要执行的 `mul(add(x, x), *)` 程序，而是在定义从候选 `mul` root 向
producer 方向递归检查的 predicate，同时规定 handler ABI：

```text
candidate mul
├─ arg0 必须是 add
│  ├─ arg0 绑定 KeywordArg("x")
│  └─ arg1 必须再次绑定同一个 PatternExpr 对象 x
└─ arg1 任意匹配，但不传给 handler

成功 handler 入参：x=<匹配值>
```

### 1. 基类存在的原因：递归调度与“每类节点自己的匹配规则”分离

当前基线中 `PatternExpr._match` 是抽象方法；公开 `match(node)` 创建 fresh
`MatchContext`，再委托 context 调用多态 `_match`
（`torch/_inductor/pattern_matcher.py:526-558`）。

这里基类统一的是：

- top-level context 生命周期；
- failure 捕获；
- pattern identity/debug comparison；
- multi-output anchor 的公共接口。

子类决定的是“当前 AST 节点如何消费一个 FX Node/constant”：

- wildcard capture 不检查 target；
- target expression 检查 op/target/users；
- argument expression 还要递归 children；
- multi-output expression 要从已命中节点的 users 寻找其他 roots。

若只用一个带大量 optional 字段的 `Pattern` 类，就会出现无效组合，例如 wildcard 同时带
target、constant 同时带 users、multi-output 又伪装成单 root args。子类把每种图场景允许
的状态和算法绑定在一起，也让 `_match` 的返回 capture ABI 明确。

> 早期/其他版本中这些类可能位于用户提到的 482、511、521、536、845、948 一带；本系列
> 固定 SHA 下相应入口已经移动到本节标注的位置。语义分析绑定当前基线，不用旧行号反推
> 当前实现。

### 2. `MatchContext.match` 是 DAG sharing 的关键，不只是递归包装

context 保存 `pattern_to_node`。每次匹配某 PatternExpr 对象前：

1. 若该对象已经绑定到同一 Node/value，直接成功；
2. 若已经绑定到不同值，返回 `repeated pattern differs`；
3. 否则调用 `pattern._match`，再记录成功 Node 或失败的 `None`
   （`torch/_inductor/pattern_matcher.py:483-516`）。

所以 pattern AST 的 Python **对象 identity** 可以把树语法提升成 DAG constraint。上述
`x, x` 使用同一 `KeywordArg` 实例，要求两处 input 完全相同；写成
`KeywordArg("x"), KeywordArg("x")` 则是两个 pattern objects，随后 `Match.extend` 只会在
合并同名 kwargs 时检查两个值一致
（`torch/_inductor/pattern_matcher.py:277-285`）。两条路径最终都可能约束 equality，但
对象复用还能服务 anchor/sharing context，不应把“类名/字段相同”当成对象绑定相同。

### 3. 三种叶子为什么必须分开

`Arg._match` 返回 positional `args=[node]`；`Ignored._match` 返回空 capture，且 repr 为 `*`
（`torch/_inductor/pattern_matcher.py:561-580`）。`KeywordArg(name)` 则返回
`kwargs={name: node}`
（`torch/_inductor/pattern_matcher.py:745-758`）。

它们都匹配任意值，区别只在 handler ABI：

| 叶子 | 匹配约束 | 写入 Match | 适用场景 |
|---|---|---|---|
| `Arg()` | 任意 | positional | handler 按 DFS 顺序消费 |
| `KeywordArg("q")` | 任意 | named `q` | 规则希望稳定命名输入 |
| `Ignored()` | 任意 | 无 | 结构必须存在但 handler 不关心 |

如果只保留一种 wildcard，再用外部列表说明 capture，AST 结构与 handler 参数很容易在嵌套/
optional/multi-output pattern 变化后错位。capture 作为叶子类型的一部分，使注册时定义与
递归匹配共用一份结构。

### 4. `CallFunction` 的 `_match` 实际做了五道门

`CallFunction` 本身只把 `op` 固定为 `"call_function"`；算法来自
`_TargetArgsExpr._match`（`torch/_inductor/pattern_matcher.py:1058-1064`）：

1. `_match_fns` 检查 candidate 是 FX Node、`node.op` 相同且 normalized target 在 fns set；
2. 检查 args 数和 users constraint；
3. 若 candidate kwargs 少于 pattern 所需，调用 `normalize_function` 用 schema 补齐；
4. flatten candidate 与 pattern，要求 TreeSpec/简单 spec 完全相同；
5. 对每对 leaves：PatternExpr 递归 `ctx.match`，常量则精确比较
   （`torch/_inductor/pattern_matcher.py:963-1019`）。

构造函数会根据是否出现 tuple/list/dict 选择快速 simple flatten 或 pytree flatten，并把
mutable/immutable 容器规范到可比结构
（`torch/_inductor/pattern_matcher.py:876-935`）。

因此 pattern 定义的不只是“目标算子序列”，而是：

```text
op + target set
+ distinct-user constraint
+ normalized call signature
+ nested aggregate structure
+ recursive child patterns
+ exact constant leaves
```

它不做代数等价：`x + 0`、`0 + x`、省略/显式 default 只有 schema normalization 覆盖的
spelling 才可能等价；交换律、broadcast、dtype promotion 需要另写 pattern/extra_check。

### 5. users constraint 为什么在 AST 内，而不是 handler 后检查

`_TargetExpr._match_users` 在以下任一条件成立时通过：

- pattern 是一个对外 output root；
- `_users=MULTIPLE`；
- `len(node.users) == expected`
  （`torch/_inductor/pattern_matcher.py:849-860`）。

中间 Node 的外部 user 会决定 handler 能否安全 erase/fuse，因此它属于“这个子图的边界
形状”，应在结构 match 阶段拒绝。注意 `Node.users` 是 distinct consumers：`add(x, x)`
对 x 是一个 user、两个 uses。需要 use multiplicity 的规则必须检查 args 结构，不能只写
`_users=2`。

### 6. `MultiOutputPattern` 怎样从第一个 root 找到其他外露 roots

构造器要求 `outputs[0]` 是 `_TargetExpr`，并把它的 op/fns 暴露给 root bucket
（`torch/_inductor/pattern_matcher.py:1162-1178`）。匹配时先以传入 candidate 命中第一个
output，再对后续非 `None` pattern 调 `_match_from_anchors`
（`torch/_inductor/pattern_matcher.py:1190-1204`）。

anchor 查找会沿 pattern 中已经绑定的 child，查看该 FX Node 的 `users`，筛出 target 可能
匹配的候选（`torch/_inductor/pattern_matcher.py:1021-1044`）。每次 alternative 失败，
`_match_from_anchors` 恢复此前 `pattern_to_node` snapshot，避免失败分支污染下一候选
（`torch/_inductor/pattern_matcher.py:1206-1217`）。

这正对应“connected subgraph 有多个外露 roots”的图场景。若把它建模为一个普通
CallFunction 的 tuple output，就无法表达两个独立 FX Nodes 共享内部 producer 的情况。

### 7. 一次 `PatternMatcherPass.apply` 如何从全图缩到候选 bucket

pass registry 的核心是：

```python
patterns[(root_op, root_target)] -> [PatternEntry, ...]
```

（`torch/_inductor/pattern_matcher.py:2583-2607`）。apply 的真实顺序：

1. 若需要，先为整图计算 mutation region IDs；
2. 对每个已注册 root key 调 `graph.find_nodes(..., sort=False)`；
3. 合并为本轮 candidates snapshot；
4. 按 FX Node sort key 逆序排序；
5. 对 candidate bucket 中 entries 按注册顺序尝试；
6. match 后检查 mutation region、stream/mempool 与 `extra_check`；
7. 调对应 entry.apply；root 若已 erased，停止尝试其余 entries
   （`torch/_inductor/pattern_matcher.py:2609-2656`;
   `torch/_inductor/pattern_matcher.py:2657-2712`）。

所以“逆图序逐个 Node 匹配”应精确理解为：

- 遍历范围是当前整张 FX Graph 的**已注册 root 候选子集**；
- 不为 graph 创建 reverse edges；
- 不是每个 Node × 每条规则；
- new replacement Nodes 不在最初候选 snapshot 中；
- later root 先改写能减少包含它的 earlier candidate 在随后尝试时误用旧结构。

### 8. 结构 match 与改图 entry 是两个接口层

`PatternExpr.match` 只返回 `Match`，其中累积 positional/keyword captures、matched Nodes、
target map 和 context。它没有写 Graph。

成功后 Entry 才决定改图：

- Lowering entry 插入后续 GraphLowering 才会调用的 handler Node；
- Graph entry 把完全自定义 surgery 交给 handler；
- Replacement entry 复制预先 trace 的 replacement graph并 reconnect/cleanup。

`Match.erase_nodes()` 也只是按 matched-node 逆序删除当前已无 users 的 Nodes
（`torch/_inductor/pattern_matcher.py:295-299`）。它不会删除仍对外暴露的 Node，不会做
whole-graph DCE，也不会 recompile。

这个分层使同一 Pattern AST 可承载不同后端动作；代价是 handler/entry 必须各自满足改图
原子性、metadata 与语义验证，不能把 context 的匹配回滚能力误认为 Graph transaction。

### 9. 当前复杂度从源码各阶段分别得到

对整图 `V`、注册 root keys `R`、物化候选 `C`、候选 bucket entries `B(v)`：

```text
mutation regions（若尚未建立）       Σv M(v)
root lookup                           H + C
candidate reverse sort               C log C · Lcmp
recursive match                      Σv Σp∈B(v) [K(p) + A(p,v)]
successful rewrite                   Q + U + local erase
```

其中：

- `call_function` 可直接按 `(op,target)` 命中 FX lookup bucket；
- 其他 op 的 target 过滤可能扫描该 op bucket；
- `MultiOutputPattern` 的 anchor alternatives 和高 fan-out users 进入 `A(p,v)`；
- repeated PatternExpr 通常由 context map O(1) 检查，但 hash/自定义对象成本仍在 Python；
- `extra_check`、handler 与 replacement tracing 是任意外部计算，不能从 `V/E` 推出上界。

逆序不是额外 DFS；主要额外成本是候选排序。只有 root 数、pattern size、arity、fan-out、
sort-key compare length 和 callback 成本都有界时，结构部分才接近图/候选线性。

### 源码边界

这条调用链证明的是 matcher 的结构语义与调度方式。它不证明某条 rewrite 对所有输入合法。
shape/dtype/layout/alias/mutation/effect/autograd/numerical 条件必须由 stage invariant、
`extra_check`、replacement trace 和独立验证共同建立。

## 15. 复杂度

定义：

- `C`：本轮物化的 root candidates，`C <= V`；
- `M(v)`：首次计算mutation region时，判定Node `v`是否mutation的成本；
- `H`：非 `call_function` root lookup的扫描成本；对每个已注册
  `call_method/get_attr` `(op,target)` key扫描该 `op` 的Node桶，若存在
  `call_module` root则统一扫描一次 `call_module` 桶；
- `B(v)`：candidate `v`所在 root bucket 的 rules；
- `K(p)`：pattern AST 大小；
- `A(p,v)`：multi-output anchor、repeated pattern 与 user alternatives 的探索量；
- `Q/U`：replacement nodes 与重连 uses。
- `Lcmp`：比较两个 FX Node 程序序 sort key 时，字典序比较实际扫描的 key 元素数上界。

```text
O(Σv M(v))                 # 仅mutation_region_id尚未建立时
+ O(H + C)                 # root lookup与candidate物化
+ O(C log C · Lcmp)        # 逆程序序排序
+ Σv Σp∈B(v) O(K(p) + A(p,v))
+ Σsuccessful O(Q + U + local erase)
```

`apply()`会在缺少 `mutation_region_id` 时先扫描全图
（`torch/_inductor/pattern_matcher.py:2416-2425`、
`torch/_inductor/pattern_matcher.py:2619-2626`）。FX lookup table只为
`call_function`维护 `(op,target)`桶；`call_method/get_attr/call_module`按 `op` 存桶，
带target查询需要过滤整个op桶（`torch/fx/graph.py:1360-1393`）。matcher对
`call_module`做了一次统一扫描，对其他root key逐项查询
（`torch/_inductor/pattern_matcher.py:2627-2645`）。

`PatternMatcherPass.apply()` 对 candidates 调用 Python `sorted`；Node 比较委托给 C++
`std::lexicographical_compare`，而插入点之间反复插入会增长层级式 sort key
（`torch/csrc/fx/node.cpp:429-449`、`torch/csrc/fx/node.cpp:454-463`）。因此不能在严格界里把一次比较无条件当作 `O(1)`；
只有 sort-key 长度或比较前缀有界时，排序项才简化为 `O(C log C)`。

`extra_check`、Graph handler 与 traced replacement 是任意 Python/trace/tensor computation，
不能从 `V/E`给出统一严格上界。只有mutation region已经建立，且root均为
`call_function`（或非call-function op桶/注册key数量有界）、sort-key比较长度、pattern、
arity与fan-out均有界时，结构部分才可近似看作候选线性；没有这些分布时，期望复杂度未定义。

## 16. 已验证 Lab

从知识库根目录运行：

```powershell
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part3_pattern.py `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\part3_pattern
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part3_end_to_end_pass.py `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\part3
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part4_artifact_bundle.py `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\part4
```

### 16.1 最小输入与覆盖矩阵

第一个脚本不是只调用 `PatternExpr.match()`，而是按设计逐级覆盖 expression 与三类 entry：

| 阶段 | 最小输入 | 运行时断言 |
|---|---|---|
| unary | `-x` | `CallFunction(operator.neg, Arg())`命中且产生一个 positional capture |
| shared input | `(x + x) * y` | 复用同一个 `KeywordArg("x")`强制两条边绑定同一 Node |
| kwargs | `torch.clamp(x, min=-0.5, max=0.5)` | positional input与命名 kwarg/constant 同时命中，`max_value == 0.5` |
| ignored | `(x + x) * y` | `Ignored()`匹配 `y`但不进入 handler positional ABI |
| multi-output | `(a * 2, a - 3), a=x+x` | `MultiOutputPattern`沿共享 `a`的 users 找到第二个 exposed root |
| graph entry | `(x + x) * y` | 实际注册 `GraphPatternEntry`，handler按相同op顺序重建子图、标记新root并erase旧 Node，样例数值相同 |
| replacement entry | `aten.add(x, x)` | `register_replacement()`实际生成 `ReplacementPatternEntry`，复制出 `aten.mul(x, 2)`并执行数值检查 |
| lowering entry | `aten.add(x, y)` | 实际生成 `LoweringPatternEntry`；`apply()`只插 handler，`GraphLowering.run()`随后调用它并产生 `ComputedBuffer(Pointwise)` |

Graph handler后的关键结构是：

```text
x -----> add(x, x) -----> mul(add, y) -----> output
y ----------------------/
```

traced replacement后的关键结构是：

```text
x -----> aten.mul.Tensor(x, 2) -----> output
```

lowering pattern经过 `PatternMatcherPass.apply()`后的 FX target 是带 `Match` 的
`lower_add_direct` handler；经过 `GraphLowering.run()`后，artifact中的稳定摘要为：

```text
operation=ComputedBuffer data=Pointwise layout=FixedLayout
```

这三个正例都实例化并检查了对应 Entry 类型，不把decorator注册成功误写成replacement成功。

### 16.2 反例、重复运行与证据边界

脚本还固定了三类边界：

- `(x + y) * z`不能命中复用同一 `KeywordArg("x")`的 sharing pattern；
- `ReplacementPatternEntry.extra_check`返回false时，apply count为0且图文本不变；
- Graph规则用新root meta与 `extra_check`拒绝第二次改写；Replacement的新root不在原
  `aten.add` bucket，两者第二次apply均为0；
- lowering实际执行至 `GraphLowering`和Inductor loop IR。
- 本路径没有编译、加载或运行native kernel，因此明确输出
  `lowering_native_kernel_executed=False`。

Graph正例保持相同op顺序并执行tensor数值检查；traced replacement正例检查
`x+x → x*2`的样例数值。

本脚本不把有限样例提升为对所有dtype/shape的等价性证明，也没有覆盖梯度、alias、
mutation或dynamic shape。`part3_end_to_end_pass.py`补充了一个静态、窄合法域中的
forward、gradient、gradcheck、shape、alias、mutation与失败原子性合同，但它同样没有
完成dynamic guard/randomized matrix或目标设备验证。

结构matcher使用公开FX graph；Entry API属于 `torch._inductor`内部接口。

### 16.3 本机实测结果

固定源码定位基线为 `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`；实测wheel为
PyTorch `2.9.1+cpu`、git `5811a8d7da873dd699ff6687092c225caffcf1bb`，两者不相同。
因此“类职责与源码行”以固定源码为准，“以下布尔结果”只声称该wheel上的运行时观察：

```text
unary_pattern_matched=True
unary_arg_captured=True
shared_pattern_matched=True
kwargs_pattern_matched=True
kwargs_constant_captured=True
failed_sharing_pattern=True
multi_output_pattern_matched=True
pattern_matcher_pass_apply_count=1
pattern_matcher_handler_calls=1
graph_pattern_entry_type=GraphPatternEntry
graph_pattern_value_matches=True
graph_pattern_old_nodes_erased=True
graph_pattern_second_apply_count=0
replacement_entry_type=ReplacementPatternEntry
replacement_apply_count=1
replacement_value_matches=True
replacement_has_mul=True
replacement_has_no_add=True
replacement_second_apply_count=0
replacement_extra_check_rejected=True
lowering_entry_type=LoweringPatternEntry
lowering_apply_count=1
lowering_handler_deferred_until_graph_lowering=True
lowering_pattern_reached_inductor_ir=True
lowering_original_add_erased=True
lowering_native_kernel_executed=False
```

本Lab artifact位于 `labs/artifacts/part3_pattern/`：

- `summary.json`：环境、所有assertion与证据边界；
- `graph_pattern_after.txt`：handler图手术后的FX graph；
- `replacement_after.txt`：traced replacement复制后的FX graph；
- `lowering_pattern_after.txt`：插入lowering handler后的FX graph；
- `lowering_ir.txt`：`GraphLowering`产出的稳定IR类型摘要。

第二个脚本把结构命中接到受限 `add(matmul)→addmm`真实 replacement，并在它声明的静态
窄合法域内覆盖L0–L4；不能据此声称完整dynamic/backend/performance合同。

第三个脚本的custom op lowering继续验证codegen-only边界；其生成源码不能作为native
kernel已编译或执行的证据。

自动合同 `PatternMatcherContractTest`同时检查本Lab既有关键输出；脚本自身对新增路径逐项
assert，不是只打印结果。完整命令与全系列证据等级见
[`labs/README.md`](labs/README.md)。

## 学习顺序

- 上一篇：[[12_fx_graph_editing_primitives_and_invariants]]
- 下一篇：[[14_dead_code_topology_and_effect_order]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[12_fx_graph_editing_primitives_and_invariants]]
- [[06_structured_outputs_higher_order_and_nested_graphs]]
- [[14_dead_code_topology_and_effect_order]]
- [[15_graph_pass_pipeline_ordering_and_fixpoint]]
- [[torch_upstream_pass_deepdive]]
