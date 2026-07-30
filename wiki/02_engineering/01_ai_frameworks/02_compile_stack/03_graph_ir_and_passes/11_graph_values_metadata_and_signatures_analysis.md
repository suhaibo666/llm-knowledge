# 03 · 图中的值、元数据与签名

> 前置：[[10_fx_graph_core_data_model_analysis]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. 一个 Node 引用同时承载三层含义

在 FX 的 `consumer.args` 中看到 producer Node 时，要分开三件事：

1. **程序值身份**：producer 的输出会作为 consumer 输入；
2. **编译期抽象值**：`node.meta["val"]`可描述 dtype/shape/stride/device；
3. **运行时真实值**：GraphModule 执行时，Node 对应某个 Tensor/标量/容器。

Node 本身不是 Tensor；FakeTensor 也不是 Node；`tensor_meta`更不是运行时值。把这几层
混在一起会产生“边里到底存了什么”的困惑。

## 2. 捕获前、捕获中、捕获后的值

| 时刻 | 典型对象 | 作用 |
|---|---|---|
| eager 运行 | real `torch.Tensor`、Python scalar/container | 真实计算和存储 |
| symbolic_trace | `Proxy` 包装 `Node` | 拦截 Python 层运算并构图 |
| make_fx/Dynamo/export | FakeTensor、SymInt 与 ProxyTensor/dispatch mode | 无真实数据或受控数据下传播算子和形状 |
| FX graph 存储 | `Node` 引用、Python 常量、嵌套 aggregate | 表达程序参数与 use-def |
| node meta | FakeTensor、TensorMetadata、stack/source info | 编译期分析与 provenance |
| GraphModule 运行 | env 中 real Tensor/scalar/container | 执行生成的 forward |

### 2.1 Proxy

Proxy 代表“如果程序继续对这个值做操作，就在图里创建对应 Node”。它属于 tracing
控制面；trace 结束后 Graph 内保存的是 Node，不保存 Proxy。

### 2.2 FakeTensor

FakeTensor 是 meta-backed Tensor subclass，还记录逻辑 device
（`torch/_subclasses/fake_tensor.py:834-845`）。它能让 shape/dtype/stride propagation
在不分配真实 payload 的情况下运行。

`FakeTensorMode`可选择拥有 `ShapeEnv`；没有 ShapeEnv 时通常是 static fake shape
（`torch/_subclasses/fake_tensor.py:1526-1536`;
`torch/_subclasses/fake_tensor.py:1550-1587`）。

### 2.3 SymInt/SymBool 与 SymNode

Python 可见的 `SymInt`包装内部 `SymNode`。`SymNode`持有：

- SymPy expression；
- optional ShapeEnv；
- Python scalar type；
- tracing hint；
- optional constant
  （`torch/fx/experimental/sym_node.py:89-136`）。

hint 是捕获样例的值，不是图对所有运行时输入的保证。unbacked symbol 没有这种 hint，
但仍可能具有 range/relationship constraints。

## 3. args/kwargs 中允许什么

FX `Argument`通常包括：

- Node；
- `None`、bool、int、float、str 等常量；
- dtype/device/layout/memory format；
- slice/range 等可序列化常量；
- 上述对象组成的 tuple/list/dict。

`args/kwargs`保留调用的结构，而 `_input_nodes`只抽取其中不同的 Node leaves。这解释了：

- 常量不是边；
- 容器结构属于 call ABI；
- 同一 producer 多次出现仍只有一个 distinct user；
- pattern 必须同时匹配结构、常量和 Node child。

当前 `Graph.create_node`还会对普通 call Node 参数中直接出现的 raw
`SymInt/SymFloat/SymBool`发出警告，建议用专门 helper 或物化 symbolic expression
（`torch/fx/graph.py:1631-1643`）。这是因为符号标量既可能是 compile-time expression，
也可能需要显式成为图值；含义必须明确。

## 4. placeholder、get_attr 与 lifted state

### 4.1 基础 FX Graph

普通 symbolic FX：

- 用户函数参数通常是 `placeholder`；
- module parameter/buffer/submodule attribute 常用 `get_attr`；
- GraphModule root 保存这些实际对象。

### 4.2 Export Graph

Export 的 graph signature 建立更强语义。当前 `InputKind`区分：

- `USER_INPUT`；
- `PARAMETER`；
- `BUFFER`；
- `CONSTANT_TENSOR`；
- `CUSTOM_OBJ`；
- `TOKEN`
  （`torch/export/graph_signature.py:81-113`）。

Export 保证 parameter/buffer/constant tensor 被 lifted 为显式 graph inputs；buffer mutation
不留在内部 state 写，而作为额外 graph outputs 表达
（`torch/export/graph_signature.py:166-181`）。

**lifted**只表示 state 边界显式化；**functional**表示 mutation 被转换为返回值/纯函数式
关系。两者相关但不是同义词。

### 4.3 AOT Graph

AOT fw/bw 的 placeholder/output 顺序由 `ViewAndMutationMeta`和 partition ABI 决定，除
用户输入外还可能包含：

- saved tensors；
- symbolic scalar values；
- opaque objects；
- tangents；
- effect/RNG token；
- BackwardState。

因此“第 n 个 placeholder 是第 n 个用户参数”只对最简单基础 FX 图成立。

## 5. 结构化输入输出、pytree 与 getitem

Python 函数可接收/返回 tuple、list、dict 和 dataclass-like pytree。捕获系统通常在某个
边界 flatten，再在 signature/codegen 层保留 TreeSpec 以重建。

图内常见两种多值表示：

1. `output.args[0]`本身是嵌套 aggregate；
2. 某个 call Node 的运行时值是 tuple-like，后续用 `operator.getitem`取元素。

这两者与 PatternMatcher 的 `MultiOutputPattern`也不同：后者描述一个 match 有多个外露
root，不要求某个单 Node 返回 tuple。详见
[[13_structured_outputs_higher_order_and_nested_graphs_analysis]]。

## 6. `node.meta["val"]`

ProxyTensor 当前将抽取出的 fake/symbolic value 放在 `meta["val"]`，并尽力生成
`meta["tensor_meta"]`
（`torch/fx/experimental/proxy_tensor.py:817-835`;
`torch/fx/experimental/proxy_tensor.py:934-954`;
`torch/fx/experimental/proxy_tensor.py:956-985`;
`torch/fx/experimental/proxy_tensor.py:986-1005`;
`torch/fx/experimental/proxy_tensor.py:1007-1028`）。

`val`可能是：

- FakeTensor；
- SymInt/SymFloat/SymBool；
- tuple/list/dict of fake/symbolic values；
- `None`或其他捕获值。

它用于：

- shape/dtype/device/layout legality；
- decomposition/lowering 选择；
- replacement retrace 的 fake examples；
- output signature 与 symbolic binding；
- debug。

源码明确警告：`meta["val"]`可支持 dtype/shape/stride/storage 查询，但不要把
`requires_grad`、`grad_fn`、`_base`当成完整 autograd 真值
（`torch/fx/experimental/proxy_tensor.py:817-823`）。

## 7. `tensor_meta` 与 `val` 的区别

`tensor_meta`是提取后的结构化摘要，典型包含 shape、dtype、stride、memory format、
quantization 参数等。它便于打印和静态 pass，但通常比 FakeTensor 能表达的信息少。

| 问题 | 优先来源 |
|---|---|
| 这个 arg 是哪一个 producer | `args/kwargs` Node 引用 |
| 编译期张量 shape/dtype/device | `meta["val"]` FakeTensor |
| 轻量打印/传播摘要 | `meta["tensor_meta"]` |
| 用户参数、parameter、buffer 语义 | graph signature |
| 图适用输入域 | guards/range constraints |
| 运行时真实数据 | GraphModule env 中的 real value |

## 8. provenance 元数据

常见 provenance key 包括：

- `stack_trace`；
- `source_fn_stack`；
- `nn_module_stack`；
- sequence/debug handle；
- AOT 的 `is_forward`/`is_backward`、recompute 等标记；
- Inductor origin sets。

这些 metadata 不构成 FX 数据边。它们回答“这个节点从哪里来”“属于哪个阶段”“应如何
debug”，而不是“执行时 consumer 读 producer 的值”。

进入 Inductor 后，`GraphLowering.run_node()`把当前 FX node 与输入 origins 合并，作为
新 IRNode 的 construction context
（`torch/_inductor/graph.py:1960-1992`）。这使 provenance 可从一对一逐渐变成
decomposition 一对多、fusion 多对一。

## 9. 三种 signature 不要混用

### 9.1 Python/GraphModule signature

由 placeholders、默认值、pytree codegen 与 generated forward 共同形成可调用接口。

### 9.2 ExportGraphSignature

把每个 graph input/output 位置映射到 user input、parameter、buffer、constant、mutation、
gradient、token 等语义。当前 output kinds 还包含 parameter/buffer/user-input mutation、
parameter/user-input gradient 和 loss
（`torch/export/graph_signature.py:121-162`）。

### 9.3 AOT runtime ABI

描述 fw 输出的用户可见 prefix、saved tensors/SymInt/opaque objects，以及 bw 输入的
saved values/tangents/token/RNG 顺序。它服务于生成的 autograd.Function，而不是
ExportedProgram 的 state mapping。

三个 signature 都可“解释 placeholder/output”，但服务对象不同：

```text
Python signature        用户如何调用 GraphModule
ExportGraphSignature    lifted program state 如何对应用户 module
AOT runtime ABI         fw 与 bw 在 autograd runtime 如何交换值
```

## 源码跟读：同一个 Node 如何在不同阶段对应不同“值”

“Node 是不是一个 Tensor”最好不要靠类比回答，而要沿构图与执行的两个环境看：

```text
捕获/分析环境                         运行环境
Proxy ──► Node                    Node ──► env[Node] = real value
          │                                  │
          ├─ args/kwargs: 程序引用            └─ consumer 取值后真正调用 target
          ├─ meta["val"]: 抽象值
          └─ signature: 边界位置语义
```

### 1. 捕获结束后，Proxy 被剥掉，Graph 只保留 Node

`create_proxy` 对参数调用 `create_arg`，把 Proxy 递归转换为其底层 Node，然后创建 Node 并
返回新的 Proxy（`torch/fx/proxy.py:340-374`）。因此 Proxy 是继续触发 tracing 的控制对象，
而 `args/kwargs` 中真正持久化的是 Node 或合法常量/aggregate。

这个分层避免让 pass 依赖 tracing 期间的动态代理状态。trace 结束后，即使所有 Proxy 都已
释放，Node 的 use-def、target 与 metadata 仍足以描述 program IR。

### 2. 真正执行 Graph 时，值保存在解释器的临时 `env`，不写回 Node

`torch.fx.Interpreter.run` 初始化 `self.env`，按 `Graph.nodes` 顺序执行，并把每个结果写入
`self.env[node]`（`torch/fx/interpreter.py:169-197`）。`run_node` 再从 env 取出当前 Node
参数对应的真实值，按 `node.op` 分派到 placeholder/get_attr/call/output 实现
（`torch/fx/interpreter.py:274-294`）。

所以：

- `consumer.args` 保存 producer **身份**；
- `env[producer]` 保存这一次执行产生的**真实值**；
- Node 本身不会因为运行一次就永久持有输出 Tensor；
- 同一 Graph 可用不同输入多次执行，每次都有新的 runtime env。

这也是 Graph 可以被编译、缓存和重复调用的基础。如果把真实 Tensor 直接塞进 Node 作为
“节点值”，图对象会同时承担程序定义和单次执行状态，输入变化、并发执行与生命周期都会
纠缠。

### 3. `meta["val"]` 是捕获时写入的抽象样本，不是 runtime env 的缓存

ProxyTensor 的 `set_meta` 把 `extract_val(val)` 写入 `proxy.node.meta["val"]`，并仅以
best-effort 方式提取 `tensor_meta`
（`torch/fx/experimental/proxy_tensor.py:817-835`）。源码在同一处明确限定：
dtype、shape、stride、storage 等低层 metadata 可依赖，但 `requires_grad`、`grad_fn`、
`_base` 不应被当作完整真值。

因此对一个 Node 可以同时成立：

```text
node.meta["val"] = FakeTensor(shape=(s0, 128), ...)
env[node]         = 某次运行得到的真实 CUDA/CPU Tensor
```

两者的 dtype/shape 等在 guards 成立时应兼容，但它们不是同一个对象，也没有相同生命周期。
pass 若把 hint 或 FakeTensor identity 当成运行时恒等关系，就跨越了源码承诺的边界。

### 4. signature 给 placeholder/output 的“位置”补语义

基础 FX placeholder 只从 opcode、target/name 和位置表达参数；Export 另建
`InputSpec(kind, arg, target, persistent)`，其 `InputKind` 区分 user input、parameter、
buffer、constant、custom object 与 token
（`torch/export/graph_signature.py:81-112`）。输出侧的 `OutputKind` 则区分 user output、
mutation、gradient、loss 与 token（`torch/export/graph_signature.py:121-155`）。

`ExportGraphSignature` 的类约定进一步说明：parameter/buffer/constant 被 lifted 为 graph
inputs，buffer mutation 被改为额外 outputs，并规定 state inputs 在 flattened user inputs
之前（`torch/export/graph_signature.py:166-181`）。

这里存在三套互补事实：

| 信息 | 存放位置 | 回答的问题 |
|---|---|---|
| `placeholder` Node | FX Graph | 程序从哪里读取一个输入值 |
| `InputSpec` / `OutputSpec` | graph signature | 这个位置在用户/module 语义上是什么 |
| `meta["val"]` | Node metadata | 捕获与分析时这个值具有什么抽象性质 |

把 signature 直接编码进所有 Node 会让基础 FX 依赖 Export/AOT 特有语义；只看 placeholder
又无法区分 lifted parameter 与 user input。独立 companion signature 因此是分层设计，
不是重复存储。

### 5. 修改图时三层信息不会自动一起正确

当 pass 替换一个 Node：

- `replace_all_uses_with` 只维护程序引用与 `users`；
- `meta` 是否传播由调用者选择，且 replacement 已有 meta 时禁止盲目覆盖；
- Export/AOT signature 若输入输出位置改变，必须由对应变换显式更新；
- runtime env 尚未存在，无需也不能由改图 API 更新。

这解释了为何“Graph.lint 通过”不能推出 signature 与 metadata 仍正确。`lint`验证
ownership、拓扑、opcode、name、target 等结构，不理解某个 placeholder 是 parameter 还是
tangent，也不执行 abstract value propagation。改图合法性必须分别检查 program IR、
metadata 与 boundary ABI。

## 10. guards 与 range constraints 不是 meta

- `meta["val"]`描述这次捕获中的抽象值；
- Dynamo guards规定何时某个 compiled entry 可复用；
- Export range constraints描述 exported program 的合法 symbolic input domain；
- deferred runtime asserts可在图/运行时验证数据相关关系。

这些 companion state 通常不作为普通 FX data edge 存在
（`torch/_guards.py:246-320`;
`torch/_guards.py:642-680`;
`torch/export/graph_signature.py:81-176`）。

## 11. 复杂度与存储成本

令 `V`为 Node 数、`E_use`为嵌套参数中的 Node use 次数、`S`为 signature 项数：

- 扫描 `args/kwargs`与建立 use-def 的结构成本是 `O(V+E_use)`；
- 按位置解释 `ExportGraphSignature`是 `O(S)`，通常 `S`由 lifted state、用户输入及
  mutation/user outputs 数量决定；
- `node.meta`的空间是所有 metadata payload 大小之和，不能简化成只与 `V`有关；
- ShapeProp/FakeTensor propagation 至少访问图结构，但抽象执行算子本身的成本由算子
  meta kernel 决定，不能无条件写成 `O(V+E)`；
- pytree flatten/unflatten 与 leaf 数线性，但用户注册的自定义节点转换成本需另计。

没有输入分布和 metadata payload 分布时，“期望成本”未定义；上面分别给出结构成本与
外部 callback/抽象算子成本，避免把二者混为一项。

## 12. 已验证 Lab：基础 FX 与 Export signature

从知识库根目录运行：

```powershell
python -B tools\labs_torch_compile\part1_values_signatures.py
python -B tools\labs_torch_compile\series_artifact_bundle.py `
  --output-dir tools\labs_torch_compile\artifacts\end_to_end
```

最小正例含一个 parameter、一个只读 buffer 和一个用户输入。错误/边界例是直接调用
`ExportedProgram`：当前 API 必须拒绝，并提示使用 `.module()`。2026-07-26 复跑的实际
stdout 是：

```text
symbolic_fx_ops=placeholder,get_attr,call_function,get_attr,call_function,output
symbolic_parameter_access=get_attr
exported_program_callable=False
export_input_kinds=PARAMETER,BUFFER,USER_INPUT
export_output_kinds=USER_OUTPUT
export_non_output_has_val_meta=True
```

Lab 使用一个 parameter、一个只读 buffer 和一个用户输入，展示：

- symbolic FX 用 `get_attr`读取 state；
- Export 将 state lifted 为输入；
- parameter/buffer 与用户输入在 signature 中具有不同 input kind；
- `ExportedProgram`不可直接调用，应使用 `.module()`。当前源码也在 `__call__`中明确抛错
  （`torch/export/exported_program.py:1457-1461`）。

这里的两个 `get_attr`分别读取 parameter 与 buffer；原先把序列压缩成一组
`get_attr,call_function`是不正确的实验记录，现已按脚本真实输出纠正。

当前 main 的 `ExportGraphSignature`还定义了 buffer/parameter/user-input mutation output
（`torch/export/graph_signature.py:121-181`）。

本 Lab 的 `2.9.1`只验证 state lifting，不用旧运行环境替代当前 main 的
mutation-contract 源码结论。

持久 artifact 是 `tools/labs_torch_compile/artifacts/end_to_end/export_graph_signature.json`、
`exported_program.py`与 `model_contract.json`；环境和命令见
[`tools/labs_torch_compile/README.md`](tools/labs_torch_compile/README.md)。

## 13. 阅读 dump 的顺序

面对一张陌生 FX 图：

1. 先数 placeholder/output，确认基础函数边界；
2. 若有 signature，用 signature 给位置贴语义标签；
3. 展开 `args/kwargs`看真实 use，而非只看 pretty print；
4. 查看 `meta["val"]`判断抽象 value；
5. 查看 stack/source metadata 追 provenance；
6. 另找 guards/range constraints，不在 Node edge 中臆测；
7. 明确这张图来自 symbolic_trace、Dynamo、export、joint、fw、bw 还是 post-grad。

## 学习顺序

- 上一篇：[[10_fx_graph_core_data_model_analysis]]
- 下一篇：[[20_symbolic_shapes_guards_and_graph_reuse_analysis]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[10_fx_graph_core_data_model_analysis]]
- [[20_symbolic_shapes_guards_and_graph_reuse_analysis]]
- [[13_structured_outputs_higher_order_and_nested_graphs_analysis]]
- [[11_aotautograd_joint_forward_backward_graphs_analysis]]
- [[20_graph_stage_boundaries_identity_and_provenance_analysis]]
- [[04_export_and_distributed/01_fx_export_extensibility/index]]
