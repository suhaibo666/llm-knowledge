# B05 · VariableTracker、Source 与 Python 对象模型

> 卷别：B · TorchDynamo 捕获  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[b04_instruction_translator_and_bytecode_state_machine_analysis]]  
> 后续：[[b06_output_graph_side_effects_and_graph_emission_analysis]]  
> 最后更新：2026-07-28

## 1. 为什么只用 `fx.Proxy`不够

Python bytecode栈里不只有 Tensor：

- int、float、bool、string；
- list、tuple、dict、iterator；
- function、method、module、class；
- context manager、exception；
- symbolic scalar；
- Tensor及其 subclass；
- 图内新建、无法从 frame重取的临时对象。

`fx.Proxy`只适合表达“某个 graph value”。Dynamo还需要知道值的 Python类别、可否当常量、
可否内联、怎样读取属性、需要什么 guard、graph break时怎样重建。

**核心结论**：`VariableTracker`回答“符号执行时这个 Python值能做什么”，`Source`回答
“运行时到哪里重新找到它”；FX Proxy只回答“它在 graph数据流中的位置”。

## 2. 三个彼此独立的维度

| 维度 | 典型对象 | 问题 |
|---|---|---|
| Python语义 | `VariableTracker`子类 | 这个值支持哪些 Python操作 |
| Runtime provenance | `Source`链 | guards和bytecode怎样重新取得它 |
| Graph dataflow | `fx.Proxy`/`fx.Node` | 这个值由哪个 graph node产生 |

一个 `TensorVariable`通常同时持 Source和Proxy；一个局部常量可能只有 Source；一个图内
临时对象可能有Proxy但没有可重建 Source。

## 3. VariableTracker的基类契约

源码将其定义为 tracked locals和stack values的基类，并规定实例应视为 immutable，需要
改变时复制（`torch/_dynamo/variables/base.py:326-350`、
`torch/_dynamo/variables/base.py:351-360`）。

为什么强调不可变/clone：

- speculation checkpoint要能复用旧状态；
- 同一个值可能出现在多个栈/容器位置；
- 原地修改tracker会让 rollback和alias推理失真；
- side effects必须单独记录，不能混成“tracker自身偷偷变了”。

基类还提供显式 worklist遍历，避免深层结构递归溢出，并保持既定访问顺序
（`torch/_dynamo/variables/base.py:362-383` 与
`torch/_dynamo/variables/base.py:385-407`）。

## 4. 为什么需要大量子类

不同 Python对象的可安全特化方式不同：

- Tensor要处理 fake value、proxy、metadata和alias；
- 常量可直接折叠，但要限制哪些值可视为常量；
- list/dict要跟踪元素和mutation；
- nn.Module要决定 specialized/unspecialized、parameters如何提升；
- function/method可能内联、作为常量或 graph break；
- SymInt要携带 ShapeEnv表达式和约束；
- user-defined object只能在有限的属性/调用规则下安全追踪。

一个“万能 SymbolicValue”会把所有分支塞入动态类型检查，难以定义每类对象的guard、
mutation和重建不变量。子类层次让 opcode/call handler按语义分派。

## 5. VariableBuilder：从真实值进入抽象域

`VariableBuilder(tx, source)`把真实 Python value包装为 VariableTracker。构造时要求：

- 必须有 Source；无来源的临时值应走 `SourcelessBuilder`；
- 必须处于 active `TracingContext`；
- 保存 translator、source和source name。

见 `torch/_dynamo/variables/builder.py:801-823`。

调用时先：

1. 记录 traced source；
2. 检查 side-effect identity table；
3. 生成 duplicate identity guard；
4. 查询按 Source索引的 tracker cache；
5. 再按真实类型选择具体 tracker。

前四步见 `torch/_dynamo/variables/builder.py:825-853` 和
`torch/_dynamo/variables/builder.py:855-869`。

## 6. Source不是值，而是“取值路径语法树”

`Source`的抽象方法包括：

- `reconstruct(codegen)`：生成取回值的 bytecode；
- `reconstruct_pycode`：生成人类可读/可执行表达式；
- `guard_source`：locals/globals/constant等根域；
- `_name_template`：稳定命名；
- `make_guard(fn)`：把取值路径和检查组合成 Guard。

见 `torch/_guards.py:1385-1405`、`torch/_guards.py:1407-1419` 与
`torch/_guards.py:1421-1447`。

`ChainedSource`持有 `base`并继承根 guard source
（`torch/_guards.py:1457-1475`），形成类似：

```text
LocalSource("self")
└── AttrSource("layer")
    └── AttrSource("weight")
        └── TensorPropertySource(SIZE, 0)
```

这棵树可以同时生成：

- guard显示名：`self.layer.weight.size()[0]`；
- 从 frame locals出发的运行时访问；
- transformed bytecode中的重建动作。

## 7. 典型 Source子类由哪些场景决定

### LocalSource

描述根 frame的 local/cell输入，并附带：

- 是否为root input；
-已知 dynamism；
-是否是 cell内容；
-是否为 `*args`或`**kwargs`。

见 `torch/_dynamo/source.py:146-174`。

### AttrSource

描述 `base.member`，可生成 `LOAD_ATTR`链或 `getattr`表达式
（`torch/_dynamo/source.py:303-327`、`torch/_dynamo/source.py:329-333`）。

### TensorPropertySource

描述 size、stride、storage_offset等不是独立 Python输入、但需要 guard/reconstruct的
Tensor属性（`torch/_dynamo/source.py:538-556`、
`torch/_dynamo/source.py:557-568` 与
`torch/_dynamo/source.py:571-584`）。

还有 global、getitem、dict key、module、random、optimizer等 Source，都是由“如何从
runtime roots稳定重取这个值”的场景差异决定。

## 8. Source与Guard的关系

Guard不是“对 node做检查”。它更接近：

```text
Source path + guard kind + expected compile-time property
```

例如：

- `LocalSource("x") + TENSOR_MATCH`；
- `AttrSource(LocalSource("self"), "training") + CONSTANT_MATCH`；
- `TensorPropertySource(...SIZE, 0) + symbolic shape expression`；
- module/object source + `ID_MATCH`。

所以 guard失败报告可以指出哪个 Python访问路径变化，而不是只报告某个 FX node变了。

## 9. Source与FX placeholder的关系

当某个 runtime值成为 graph input时，OutputGraph按 Source去重 input，避免同一来源创建多个
placeholder（`torch/_dynamo/output_graph.py:786-793`）。

但不是每个 Source都成为 placeholder：

- 特化常量可能只产生 guard；
- parameter/buffer可能注册为 `get_attr`；
- 图内临时值没有外部 Source；
- Python residual要重建的值可能不进入 FX数据流。

反过来，placeholder通常带回 Dynamo Source元数据，以便 backend/debugging保留 provenance。

## 10. Identity、alias与side effects为什么不能只看Source字符串

两个不同 Source可能指向同一个 Python对象，例如两个 locals alias同一 list或module。
VariableBuilder先查询 side-effect identity table，并为重复引用安装 dupe guard。这样：

- mutation只记录一份对象身份；
- 两条 Source路径的 alias关系受 guard保护；
- graph break后回放不会把一个对象错误地拆成两个；
- module hierarchy metadata仍可更新到当前 source。

Source是访问路径；对象 identity是另一个维度。

## 11. 无 Source值的边界

局部新建对象可能没有从 frame roots重取的路径。此时：

- 若完全留在图内，可由 proxy/graph表达；
- 若是可常量化的临时值，可由相应tracker持有；
- 若跨 graph break存活，就必须能输出、重建或显式拒绝；
- 若mutation/identity无法正确回放，就应 graph break或报错。

“没有 Source”不等于“没有值”，而是不能靠 runtime provenance guard它。

## 12. 复杂度

设嵌套 Python结构包含 \(V\) 个 tracker，Source链平均深度 \(D\)：

- VariableBuilder按 source cache命中通常接近 \(O(1)\)；
- 首次分类取决于对象类型和容器大小；
- tracker结构遍历为 \(O(V)\)；
- Source取值/guard访问通常为 \(O(D)\)；
- 深层 container/source tree会增加 guard构建和运行成本；
- alias/side-effect表使用 identity映射，期望查询 \(O(1)\)，但回放与剪枝随记录规模增长。

## 13. 常见误解

- **“VariableTracker就是GraphNode。”** 它是 Python抽象值；一个tracker可能没有node，
  也可能包装proxy指向node。
- **“Source保存真实Tensor。”** Source保存如何从运行时环境找到值。
- **“每个Source对应一个placeholder。”** 常量guard、get_attr和Python residual都是反例。
- **“两个相同Source name一定是同一对象。”** identity与访问路径必须分别处理。
- **“Ignored值不需要Source。”** 是否传给pattern handler与Dynamo provenance无关，属于
  完全不同的 pattern matcher层。

## 配套 Demo

本页对应卷级入口 `labs/demo_b_dynamo_capture.py` 的 `variable_source_guards` 用例。默认以 CUDA 为验收设备：

```powershell
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\demo_b_dynamo_capture.py `
  --case variable_source_guards --device cuda `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\volume_demos\b05
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `variable_source_guards/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[b04_instruction_translator_and_bytecode_state_machine_analysis]]
- [[b06_output_graph_side_effects_and_graph_emission_analysis]]
- [[b07_guards_cache_lookup_and_recompilation_analysis]]
- [[02_fx_graph_core_data_model]]
- [[13_pattern_expression_and_matcher_engine]]
