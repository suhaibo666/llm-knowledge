# 04 · 符号形状、Guards 与图复用

> 前置：[[03_graph_values_metadata_and_signatures]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. 核心问题：一张图描述的是函数，还是一次输入？

捕获 `f(x)`时，样例 `x.shape == (3, 8)`只证明程序在这个输入上走过某条路径。要复用：

- Python 控制流必须仍走同一路径；
- dtype/device/layout 等假设必须成立；
- shape 关系与 range 必须满足；
- module/global/object 状态假设不能变化；
-生成代码采用的 indexing、allocation、kernel 选择仍合法。

PyTorch 用“图 + companion validity state”回答：

```text
FX program graph
  + Dynamo guards 或 Export range constraints
  + FakeTensor/SymInt 抽象值
  + 后端自己的 guards 与 runtime asserts
```

Guard 不是普通 FX data edge；它是 compiled entry 的适用条件。

## 2. 静态特化

最保守策略是把捕获样例的尺寸全部特化成常量。优点：

- shape 推导简单；
- 常量折叠和 layout/tiling 选择更确定；
- kernel 可针对固定 numel 优化。

缺点：

- 每个新 shape 可能 guard miss 并触发重编译；
- cache entry 数与编译开销增长；
- serving/sequence length 场景复用差。

“shape 在 cache key 里”是过度简化。当前 Dynamo 模型是：一个 code object 可关联多个
guarded compiled entries，运行时执行 check function 选择适用 entry；guard fail 后可能
重编译（`torch/__init__.py:3157-3166`;
`torch/_dynamo/convert_frame.py:1002-1021`;
`torch/_dynamo/convert_frame.py:1903-1927`）。

## 3. SymInt、SymBool 与 symbolic expression

动态维度不只是“未知整数”，而是带关系的表达式：

```text
s0
s0 + 1
s0 * s1
Eq(Mod(s0, 8), 0)
s0 <= 1024
```

`SymNode`保存 expression、ShapeEnv、Python scalar type、hint 与 optional constant
（`torch/fx/experimental/sym_node.py:89-136`）。

### hint 与 invariant

- hint：本次 tracing run 的 concrete value，用于需要选择一条执行路径的地方；
- invariant：未来输入必须满足的 guard/constraint；
- expression：图内/编译期可代数化的 symbolic 关系。

把 hint 当 invariant 会错误地把 dynamic graph 解释成 static graph。

## 4. ShapeEnv 的职责

ShapeEnv 负责在一次 tracing/compilation 中：

- 创建 backed/unbacked symbols；
- 维护 source、expression、hint 与 range；
- 简化和证明 symbolic predicates；
- 记录必须在未来输入上验证的 guards；
- 生成 deferred runtime assertions；
- 处理 duck sizing/equality replacement。

当前权威 backed mapping 是 `backed_var_to_val`；`var_to_val`是 deprecated property
（`torch/fx/experimental/symbolic_shapes.py:5960-5980`）。

### range 是保守近似

`var_to_range`保证真实值在 range 内，但 range 可能包含不可能值
（`torch/fx/experimental/symbolic_shapes.py:4024-4028`）。因此：

- range 能证明某些谓词；
- 不能把 range 当所有可能值的精确集合；
- 证明不了不等于谓词运行时为 false。

### size-like 不是无条件全局下界

`size_like`表示 size-oblivious reasoning 可按特定规则处理的 symbols，并不等于所有上下文
都已硬性记录 `symbol >= 2`
（`torch/fx/experimental/symbolic_shapes.py:4064-4068`）。

## 5. compile-time refinement 与跨调用重编译

一次编译内部，ShapeEnv 可根据关系表达式 refine active ranges
（`torch/fx/experimental/symbolic_shapes.py:8073-8134`;
`torch/fx/experimental/symbolic_shapes.py:9047-9105`）。

后续运行时输入不会反向修改旧 compiled graph 的 ShapeEnv。它们只会：

1. 满足旧 guards → 复用；
2. 不满足 → 尝试其他 cache entry；
3. 仍无匹配 → 触发新 compilation 或回退。

因此“系统从每次 runtime shape 学习并不断收窄同一图的 range”是错误模型。

## 6. equality replacement 为何仍需要 guard

假设本次 tracing 观察到两个 source size 相等，ShapeEnv 可能把表达式统一到同一 symbol，
从而简化图。但该统一依赖未来输入仍相等。

内部 replacement 是**在假设成立条件下的编译期简化**；runtime validity condition 仍要以
source equality/duck sizing/guard 的某种形式成立。它不一定以文字 `s0 == s1`出现，但不能
凭空消失。当前 guard 生成明确处理 non-trivial equality
（`torch/fx/experimental/symbolic_shapes.py:6031-6046`）。

## 7. backed 与 unbacked symbols

### backed

有 tracing hint，通常来自 input size/stride/scalar，可在捕获时选择路径，并由输入 source
重建 runtime value。

### unbacked

捕获时没有 hint，常来自数据相关标量，例如 Tensor `.item()`产生后又参与 shape。它不是
“完全无约束”：

- 可有 range；
- 可参与 equality/inequality；
- 可由 deferred runtime assertion 限制；
- codegen 需要在运行时定义/bind。

是否有 hint 与是否有 constraint 是两个维度。

## 8. `torch.compile` dynamic 策略

当前公开语义必须区分：

- `dynamic=True`：尝试预先生成尽可能 dynamic 的 kernel；
- `dynamic=False`：不生成 dynamic kernel，始终特化；
- `dynamic=None`：自动策略，先捕获，再在相关 guard failure 后尝试更 dynamic 的 compilation；
- `dynamic_shapes=`：独立的 shape specification，不能与 `dynamic=`混用
  （`torch/_dynamo/eval_frame.py:838-902`;
  `torch/__init__.py:3134-3148`;
  `torch/__init__.py:3175-3181`）。

自动动态化也不意味着每次都生成“排除原 static value”的 guard。PGO merge 在从 concrete
value 转成 automatic dynamic 时记录 `excluded_sizes/excluded_scalar`，并在状态继续变化时
清理旧值（`torch/_dynamo/pgo.py:399-423`）。`automatic_dynamic_exclusion_guard`当前默认
false；它至少显式控制 scalar exclusion 是否传给 ShapeEnv
（`torch/_dynamo/config.py:195-210`;
`torch/_dynamo/variables/builder.py:3388-3399`）。Tensor exclusions 还会随 symbolic
context 进入 shape creation，并由 ShapeEnv 记录 constraint
（`torch/_dynamo/variables/builder.py:4835-4846`;
`torch/fx/experimental/symbolic_shapes.py:4810-4832`）。所以“PGO 是否记录状态”和“是否把
exclusion 转成当前 graph 的 constraint/guard”必须分开检查，不能概括成“开关关闭就不维护”。

## 9. Export shape contract

`torch.export.export`当前默认 `strict=False`
（`torch/export/__init__.py:59-69`）：

- non-strict 使用 Python runtime path，同时验证关键 shape safety；
- strict 使用 Dynamo，提供更强 soundness capture path
  （`torch/export/__init__.py:179-187`）。

ExportedProgram 保存：

- GraphModule；
- graph signature；
- state dict/constants；
- range constraints；
- module call graph；
- verifiers
  （`torch/export/exported_program.py:1069-1152`）。

`.module(check_guards=True)`会 unlift state 并安装 guard-checking submodule；直接调用
ExportedProgram 会报错
（`torch/export/exported_program.py:1457-1501`）。

## 10. Guard、assert 与后端约束的分工

| 机制 | 典型职责 |
|---|---|
| Dynamo guard | compiled cache entry 是否适用于本次 Python/runtime state |
| Export range constraint | exported input domain |
| ShapeEnv guard | symbolic reasoning 依赖的 source relation |
| deferred runtime assert | tracing 时无 hint 的数据相关关系 |
| Inductor size/layout guard | 后端 indexing/layout/algorithm 假设 |
| `assert_size_stride` | wrapper 输入 size/stride tuple 检查的一部分 |

`assert_size_stride`不会自动验证所有 range、divisibility 或跨输入 equality。它记录/发射
输入 size/stride expectations
（`torch/_inductor/codegen/wrapper.py:1719-1741`;
`torch/_inductor/codegen/wrapper.py:1841-1904`），其他条件必须由引入该事实的层负责。

## 源码跟读：一个输入尺寸怎样变成 symbol、guard 与复用条件

符号形状链路可以压缩成四个动作，但每个动作保存的事实不同：

```text
真实样例 size/stride/offset
        │ create_symbolic_*
        ▼
SymInt(SymNode(expr, hint, ShapeEnv, source/range))
        │ 某处必须取得 Python bool/int
        ▼
ShapeEnv.evaluate_expr：证明、记录 guard，或拒绝
        │ produce_guards
        ▼
绑定 runtime source 的 Python 条件 ──► cache entry 能否复用
```

### 1. symbol 不是从整数值单独创建，而是从“值 + source + dynamic policy”创建

`ShapeEnv.create_symbolic_sizes_strides_storage_offset` 接收真实样例 Tensor 与 `Source`；
若 Tensor 已属于另一个 ShapeEnv，它先走 foreign-symbol transfer，否则把 size、stride、
storage offset 与 source 交给内部创建逻辑
（`torch/fx/experimental/symbolic_shapes.py:4834-4866`）。

内部逻辑依据 `SymbolicContext` 或默认策略，把每个维度标成 STATIC、DYNAMIC、DUCK 等模式；
在没有 context 的 legacy 路径里，`assume_static_by_default` 直接影响选择
（`torch/fx/experimental/symbolic_shapes.py:5122-5165`）。

这解释了为什么“样例 size 都是整数”不意味着图一定 static：整数是 hint 来源；是否创建
symbol 由 dynamic policy/context 决定。反过来，创建了 symbol 也不意味着所有整数都能
复用，source range、equality、stride 和后续分支仍会形成约束。

### 2. `SymNode` 需要 concrete bool 时，会把一次 Python 分支变成未来输入条件

当 Python 代码要求 `bool(sym_expr)`，`SymNode.guard_bool` 调用 `evaluate()`，再把结果转成
bool（`torch/fx/experimental/sym_node.py:576-584`）。ShapeEnv 的
`evaluate_expr` 契约是“计算表达式，必要时增加 guard”
（`torch/fx/experimental/symbolic_shapes.py:8538-8563`）。

内部完成本次 concrete 取值和谓词方向选择后，会构造 `ShapeGuard` 并加入
`self.guards`；export 且偏好 deferred assert 的路径则可能改走
`guard_or_defer_runtime_assert`
（`torch/fx/experimental/symbolic_shapes.py:8855-8873`）。

因此捕获时的分支不是凭空被“符号执行掉”：

```text
if x.shape[0] > 4:
    branch_A
else:
    branch_B

捕获样例走 A
=> 图中只记录 A 路径
=> companion guard 必须保证未来输入仍满足选择 A 的条件
```

这也是 guard 不放进普通 FX data edge 的原因：它约束“是否可选这份 compiled program”，
而不是某个 call Node 的数据参数。

### 3. backed 与 unbacked 的分叉发生在“能否用 hint 立即决定”处

`SymNode.expect_true` 在表达式有 hint、没有 free unbacked symbol 且策略允许时走
`guard_bool`；否则交给 ShapeEnv 的 `guard_or_defer_runtime_assert`
（`torch/fx/experimental/sym_node.py:586-604`）。

所以 backed/unbacked 不是“有约束/没约束”的区分：

- backed symbol 可用 hint 在捕获时取值，并把依赖转成 entry guard；
- unbacked symbol 没有这种捕获时取值来源，但仍可通过运行时产生的值、range 与 deferred
  assertion 被约束；
- 某些关系若被静态证明，既不需要新 guard，也不需要 runtime assert。

### 4. `produce_guards` 把内部 expression 重新绑定到输入 source

ShapeEnv 内部 guard 使用 SymPy expression；运行时却需要访问具体输入的 size/stride 等
source。`produce_guards` 调用 verbose producer，后者接收 placeholder FakeTensors、
对应 Sources、可选 equality/context，并生成 Python guard expressions
（`torch/fx/experimental/symbolic_shapes.py:6008-6046`）。

这一步是图复用的 ABI：symbol `s0` 本身不能在下一次调用里被直接读取，必须知道它来自
诸如 `L["x"].size()[0]` 的 source。也正因为如此，`node.meta["val"]` 单独不能定义图的
合法输入域；它保存抽象值，而 guards/range constraints 保存从未来输入重建并校验这些
假设的方法。

### 5. guard failure 为什么不是“在原 ShapeEnv 上继续学习”

一份 compiled entry 的 Graph、ShapeEnv 推导结果和 generated code 共同依赖已记录 guards。
运行时 guard failure 表示这份既有证明的前提不成立；安全动作是选择另一 entry、重新捕获/
编译，或回退，而不是用新输入修改旧证明。

若原地扩大旧 ShapeEnv 的 range，图中已经完成的分支选择、常量折叠、layout/indexing 与
kernel 策略不会自动回滚。把 compiled entry 视为“program + proof obligations”的不可变
组合，正是多 entry cache 比“一个会持续变形的图”更容易保持正确性的原因。

### 源码边界

上述源码能证明 symbol 创建、表达式求值、guard/deferred-assert 分流和 guard 生成机制；
它不能仅凭这一层预测某个用户程序一定生成几份 cache entry。Dynamo 的 graph break、
PGO、配置、后端 guard 和 cache policy 都会改变最终重编译次数，因此正文中的次数观察
只属于固定 Lab，不是 ShapeEnv API 的普遍保证。

## 11. symbolic shape 对后续图机制的影响

### Pattern

结构相同不代表 symbolic legality 相同。handler/extra_check 可能需要证明 broadcast、
divisibility、rank、stride 或 equality；无法证明时要保留原图或引入 guard。

### DCE

普通纯 value 是否 dead 主要由 users/effect 决定，但 shape node/runtime assertion 可承担
validity effect，不能仅因“输出 Tensor 未使用”就删除。

### Fusion/Scheduler

dynamic ranges影响：

- loop group compatibility；
- indexing equality；
- reduction split；
- buffer size/lifetime estimate；
- template choice；
- peak-memory hint。

### Codegen

symbolic numel 可能成为 kernel `SizeArg`，grid 运行时表达式可依赖它；block sizes仍可来自
compile-time heuristic/config。当前 pointwise 配置经 device heuristic registry 生成，不应
用一个历史固定 XBLOCK 列表概括
（`torch/_inductor/runtime/triton_heuristics.py:4365-4435`;
`torch/_inductor/runtime/triton_heuristics.py:5005-5152`）。

## 12. 复杂度与 cache 代价

symbolic reasoning 不是免费抽象：

- expression simplification/guard generation取决于表达式 DAG 与关系数；
- cache entry lookup 取决于 entry/guard 数及 guard cost；
- guard fail 可能触发完整编译；
- 更 dynamic 的 kernel 可能失去特化优化；
- 更 static 的策略可能造成 compilation explosion。

不能用单一 `O(V+E)`覆盖整个动态形状路径；symbolic algebra 的复杂度与 FX Node 数并非
线性绑定。

## 13. 已验证 Lab：static recompile 与 automatic dynamic

从知识库根目录运行：

```powershell
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part1_symbolic_shapes.py
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\series_artifact_bundle.py `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\end_to_end
```

第一个脚本以 batch 3、4、5 为正例，用自定义 backend 只记录 GraphModule，避免混入
Inductor 性能。第二个贯穿 Lab export 明确的 `1 <= batch <= 8`范围，并把 batch 9 作为
错误/边界例；`exported.module()`必须以 `Guard failed: x.size()[0] <= 8`拒绝。PyTorch
`2.9.1+cpu`实测：

```text
dynamic_false_compiles=3
dynamic_none_compiles=2
dynamic_none_second_graph_has_symint=True
export_range_constraints=1
```

输入 batch sizes 为 3、4、5：

- `dynamic=False`为每个新 size 特化；
- `dynamic=None`先特化，再生成可覆盖后续 size 的 symbolic graph；
- ExportedProgram 明确保存 range constraint。

这是 Lab 环境的观察，不承诺所有版本/程序恰好产生相同 compilation count；控制流、
配置、PGO 和 guard policy 会改变结果。

持久 artifact 位于 `labs/artifacts/end_to_end/dynamo_fx.py`、
`dynamo_guards.txt`和 `export_graph_signature.json`；自动合同还断言
`export_out_of_range_rejected=True`。环境、命令和 runtime/source 版本差异见
[`labs/README.md`](labs/README.md)。

## 14. 排查清单

1. 这个 size 是 constant、backed SymInt 还是 unbacked SymInt？
2. 当前看到的是 hint、expression、range 还是 runtime guard？
3. 关系是在编译期可证明，还是要 runtime assert？
4. guard failure 会选择旧 entry、新 entry、recompile 还是 graph break？
5. Export range 与 Dynamo guard 是否被错误地当成一个对象？
6. 后端是否又加入 layout/indexing guard？
7. 固定 candidate/performance 结论是否绑定具体 backend 与 source？

## 学习顺序

- 上一篇：[[03_graph_values_metadata_and_signatures]]
- 下一篇：[[05_graph_effects_alias_mutation_and_order]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[03_graph_values_metadata_and_signatures]]
- [[07_graph_capture_frontends_and_tracing]]
- [[16_graph_rewrite_legality_validation_and_complexity]]
- [[17_fx_lowering_to_inductor_ir]]
- [[21_codegen_kernel_mapping_autotuning_and_provenance]]
- [[dynamic_shapes_full_analysis]]
