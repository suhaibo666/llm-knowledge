# 07 · 图捕获前端与 Tracing

> 前置：[[02_fx_graph_core_data_model]]、[[04_symbolic_shapes_guards_and_graph_reuse]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. 为什么需要四种捕获路径

“把函数变成 FX”不是一个唯一算法。不同前端选择不同拦截层、抽象值和 soundness contract：

| 路径 | 主要拦截层 | 典型产物 | 擅长 |
|---|---|---|---|
| `symbolic_trace` | Python Proxy / `__torch_function__` | GraphModule | module 结构与 Python-visible ops |
| `make_fx` | ProxyTensor / dispatcher | GraphModule | 实际执行到的 ATen 级 ops |
| Dynamo | Python bytecode/frame evaluation | backend GraphModule + guards | 动态 Python、graph breaks、torch.compile |
| `torch.export` | non-strict runtime 或 strict Dynamo | ExportedProgram | 可部署图、signature、range constraints |

它们得到的 Node 数、target 层级、控制流行为和 companion state 不同，不能仅用“都是 FX”
判断等价。

## 2. `symbolic_trace`

`symbolic_trace`创建/使用 `Tracer`，运行 `Tracer.trace`，再返回 `GraphModule`
（`torch/fx/_symbolic_trace.py:1361-1421`）。

### 构图机制

- 函数/module 参数变成 Proxy-backed placeholders；
- 对 Proxy 的 Python operator/method/module call 创建 Node；
- parameter/buffer 读取常变成 `get_attr`；
- leaf module 保留为 `call_module`，非 leaf 可进入其 forward；
- trace 结束以 Proxy/aggregate 形成 output。

### 能力边界

Proxy 不是 real Tensor。若 Python `if`要求将 Proxy 转 bool，Tracer 通常无法决定分支；
数据相关循环也类似。可以用 concrete args 特化，但那改变图的输入域。

`symbolic_trace`主要通过 Proxy 的 Python operator 与 `__torch_function__`路径，不应和
ProxyTensor/FakeTensor 的 `__torch_dispatch__`机制叫成同一拦截点。

## 3. `make_fx`

`make_fx(f, ...)`先返回 callable，随后用 example inputs 调用才得到 GraphModule
（`torch/fx/experimental/proxy_tensor.py:3312-3385`）。

它在 dispatcher/ProxyTensor 层观察实际执行的 operators，因此常得到 functional/ATen-like
targets，而不是 module hierarchy：

```text
symbolic_trace: call_module Linear
make_fx:        aten.t, aten.addmm ...
```

当前 tracing modes：

- `real`：用真实 inputs；
- `fake`：FakeTensor propagation；
- `symbolic`：兼容保留但源码明确不推荐的新使用路径。

`make_fx`仍是 execution trace：未执行的 Python branch不会自动出现在普通图里。

## 4. Dynamo

Dynamo 在 frame/bytecode 层模拟 Python 执行，能：

- 跟踪 locals/stack/object state；
- 捕获 Tensor ops；
- 为 Python/shape/module/global 假设生成 guards；
- 在不支持区域 graph break；
- 为同一 code object维护多个 guarded compiled entries。

backend-facing context包含 GraphModule、example inputs、FakeTensorMode 与 symbolic contexts
（`torch/_dynamo/convert_frame.py:1052-1067`）；guards另行累积并编译成 check function
（`torch/_dynamo/convert_frame.py:986-1021`;
`torch/_dynamo/convert_frame.py:1903-1927`）。

因此 backend 只看到 GraphModule，不代表 graph 本身含有所有 guards。

## 5. `torch.export`

ExportedProgram拥有 GraphModule、signature、state、constants、range constraints、module call
graph 和 verifiers（`torch/export/exported_program.py:1069-1152`）。

当前 `export(..., strict=False)`为默认
（`torch/export/__init__.py:59-69`）：

- non-strict 路径借助 Python runtime，验证关键 shape safety；
- strict 路径使用 Dynamo，提供更强 soundness capture contract
  （`torch/export/__init__.py:179-187`）。

Export 的目标不是“把任意 Python 保存下来”，而是产生满足更强 invariant 的 Tensor
program，并把 state/input/output 语义放进 graph signature。

`ExportedProgram`不可直接调用；应使用 `.module()`，后者 unlift state 并按配置安装 guard
检查（`torch/export/exported_program.py:1457-1501`）。

## 6. real、fake 与 symbolic value

| value mode | 是否有真实 payload | shape | 风险/用途 |
|---|---|---|---|
| real | 有 | concrete | 会真实执行、可能昂贵/有副作用 |
| fake static | 无 | concrete integers | 便宜传播，但特化 |
| fake symbolic | 无 | SymInt expressions | 支持 dynamic shape 与 guards |

FakeTensor还建模 logical device，不只是 meta Tensor
（`torch/_subclasses/fake_tensor.py:834-845`）。

## 7. Python 控制流的五种结果

对 `if x.sum() > 0`：

1. eager：读取真实数据并选择 branch；
2. symbolic_trace：Proxy 无法转 bool，通常失败；
3. make_fx real：执行某一 branch，只捕获该路径；
4. Dynamo：可 graph break/特化，或要求改写为 structured control flow；
5. export：要求可证明/可捕获，推荐 `torch.cond`表达动态分支。

Python `if x.shape[0] > 3`不同：shape可成为 backed symbolic predicate并生成 guard/branch
specialization。

## 8. leaf function/module 是前端策略

“一个 Node 是 Linear”还是“Linear 展开成 ATen”由前端和 tracer leaf policy决定，而非
Linear 本身具有固定图粒度。

这直接影响 pass：

- 想匹配 module architecture，选择高层 graph；
- 想匹配 ATen pattern，需 decomposition/低层捕获；
- 同一 pattern不能无条件跨粒度复用。

## 9. Graph break 不等于编译失败

Dynamo 遇到不支持 Python 时可结束当前 graph segment，回到 Python 执行，再捕获后续
segment。`fullgraph=True`才要求一个完整 graph，否则 graph break 是分段执行协议。

代价是：

- segment 边界产生 Python/dispatcher 往返；
- 跨 segment fusion/优化不可见；
- state/effect 边界更复杂；
- cache/guard 数增加。

## 10. 捕获产物对照

| 维度 | symbolic_trace | make_fx | Dynamo | export |
|---|---|---|---|---|
| 输出类型 | GraphModule | GraphModule | backend 输入 | ExportedProgram |
| guards | 通常无 companion guard set | mode/shape env 内部 | 独立 guard manager | range constraints/verifiers |
| state | get_attr/module hierarchy | 常 functional op/input | capture-dependent | lifted + signature |
| op 粒度 | Python/module | dispatcher/ATen | 后续阶段决定 | functional ATen contract |
| graph break | 失败/特化 | trace 已执行路径 | 支持分段 | 导出失败或结构化表达 |
| 适合直接改图 | 是 | 是 | backend/pass 内 | 需维护 signature/invariants |

## 源码跟读：四个前端究竟在哪一层截获程序

四条路径最终都可能出现 FX `GraphModule`，但“进入 `Graph.create_node` 之前发生了什么”
不同，决定了图里有哪些 target、哪些 Python 事实被保留，以及正确性条件放在哪里。

### 1. `symbolic_trace`：让 Python 函数直接在 Proxy 参数上执行

入口创建默认 `Tracer`，调用 `trace`，再把结果包装成 `GraphModule`
（`torch/fx/_symbolic_trace.py:1413-1421`）。`Tracer.trace` 创建空 Graph 和 Proxy
placeholders，在 patch 环境中执行 `fn(*args)`，最后把返回值转为 output Node
（`torch/fx/_symbolic_trace.py:801-828`;
`torch/fx/_symbolic_trace.py:890-932`）。

因此它看到的是 Python 调用表面：Proxy operator、method、module leaf。某段 Python 若不对
Proxy 做可记录操作，或要求 Proxy 提供真实 bool/data，就没有 dispatcher 层的后备机制。

### 2. `make_fx`：外层仍借 FX 建图，算子截获点移到 dispatcher/ProxyTensor

`make_fx` 本身不立即追踪；它先解析 tracing mode/dynamic spec，构造 `_MakefxTracer`，返回的
`wrapped(*args)` 才调用 tracer.trace
（`torch/fx/experimental/proxy_tensor.py:3312-3350`;
`torch/fx/experimental/proxy_tensor.py:3353-3385`）。

真正 trace 时，源码先把输入按模式转为 fake/real，必要时为 varargs 生成假签名；随后同时
进入 FakeTensorMode、Python dispatcher、ProxyFunction、ProxyTorchDispatch 等 mode，
再调用 `dispatch_trace(wrap_key(...))`
（`torch/fx/experimental/proxy_tensor.py:3153-3189`;
`torch/fx/experimental/proxy_tensor.py:3206-3213`）。

调用链可以概括为：

```text
make_fx(f) → wrapped(example_inputs)
→ _MakefxTracer.trace / _trace_inner
→ 输入 fakify（按 real/fake/dynamic spec）
→ ProxyTorchDispatchMode 截获 dispatcher operator
→ FX tracer.create_proxy/create_node
→ GraphModule
```

所以 `make_fx` 不是抛弃 FX Tracer，而是用 dispatcher mode 决定“什么操作应请求 tracer
创建 Node”。这解释了为何 module hierarchy 常被实际执行的 ATen ops 展开。

### 3. Dynamo：backend GraphModule 与 guards 在前端输出里是并列产物

Dynamo 的 `DynamoOutput` 同时保存 tracer output、生成 bytecode 和编译信息；其
`build_guards` 从 `output_graph.guards` 构造 `CheckFunctionManager`
（`torch/_dynamo/convert_frame.py:986-1021`）。传给 backend 的 `BackendInput` 则保存
GraphModule、example inputs、FakeTensorMode 和 tensor symbolic contexts
（`torch/_dynamo/convert_frame.py:1052-1067`）。

这两个数据结构给出清晰边界：

```text
Dynamo frame/bytecode analysis
├─ backend input: FX GraphModule + fake examples/context
└─ cache validity: guards → CheckFunctionManager
```

backend GraphModule 不包含全部 Python/global/module/cache guards，不是信息丢失；guard
manager 属于“是否进入这份 compiled callable”的外层协议。若 pass 只序列化 GraphModule
却声称保留完整 Dynamo cache contract，就越过了这个边界。

### 4. Export：先选择捕获 contract，再构造更强的程序 artifact

公开 `export` 默认 `strict=False`，入口文档明确要求产物是 normalized functional ATen
program，并记录控制流消除所依赖的 shape constraints
（`torch/export/__init__.py:59-85`）。参数说明区分：

- non-strict：通过 Python runtime trace，并验证关键 shape safety；
- strict：通过 Dynamo，提供更强 soundness capture，但 Python 覆盖更有限
  （`torch/export/__init__.py:179-187`）。

入口验证 module 类型、解析 dynamic spec，最后把 `strict`、pre-dispatch 与 guard/assert
策略传给内部 `_export`
（`torch/export/__init__.py:205-235`）。

因此 strict/non-strict 改变的是“如何证明捕获成立”，不是让最终 Export IR 变成两个不同
schema。两条路径都必须交付 `ExportedProgram` 的 graph signature、state/constants、range
constraints 和 verifier contract。

### 5. 同一 Python 程序为何会得到不同 Node 粒度

以前端截获层来解释，比记忆输出示例更稳：

| 前端 | 截获事件 | 典型保留 | 典型被展开/移到 companion state |
|---|---|---|---|
| symbolic FX | Proxy 的 Python-visible operation | leaf module/method/function | 数据相关 Python 路径无法决定 |
| make_fx | dispatcher operator | 实际执行 ATen 调用 | module hierarchy、未执行分支 |
| Dynamo | bytecode 解释中的可编译 segment | Python/Tensor segment | guards、graph breaks、cache 放在外层 |
| export | strict/non-strict capture 后的规范程序 | functional ATen + signature | Python state 被 lift/消除并加约束 |

不存在“最完整、总应优先”的单一图。前端越靠近 Python，越能保留结构但越难获得低层算子
规范；越靠近 dispatcher/Export contract，越适合后端分析，但越需要把 Python 与 state
语义变成 guards、signature 或 graph breaks。

### 源码边界

这些调用链证明前端的拦截层和产物边界；它们不保证具体模型一定产生某个固定 target 数、
graph-break 数或 guard 数。leaf policy、decomposition table、dynamic config、custom op 和
版本实现都会改变图形，文档后续所有 pattern/pass 结论都必须先声明其预期输入阶段。

## 11. 复杂度与失败边界

捕获成本不能只用最终图的 `V/E`表示：

- symbolic tracing 至少执行一次 Python trace path；module/function leaf 策略决定展开量；
- make_fx/ProxyTensor 的 structural bookkeeping 近似与 dispatch 次数及参数树大小相关，真实
  meta/算子执行成本另计；
- Dynamo 还要解释 bytecode、构造 guards、管理 graph breaks 与 cache entries；一次 guard
  miss 可能触发完整重新捕获/编译；
- export 在捕获之外还要构建 signature、range constraints 与 verifier；
- 分成 `k`个 graph segments 会引入 `k`次 backend/caching 边界，不能只看每段 Node 数。

给定 bounded-arity、无 graph break 的普通张量代码时，结构成本常随执行事件数近线性；
Python callback、symbolic algebra、guard evaluation 和重新编译是外生成本。没有输入/guard
分布时，期望 compilation count 未定义。

## 12. 已验证 Lab

从知识库根目录运行：

```powershell
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part2_capture_frontends.py
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\series_artifact_bundle.py `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\end_to_end
```

第一个脚本对同一 Linear Module 比较四种前端，检查 Dynamo guards 和
`node.meta["example_value"]`，再用显式 `torch._dynamo.graph_break()`构造边界例；同一函数
必须被 backend 收到两段图且数值仍等于 eager。贯穿 bundle 保存完整 node/target/meta、
guards 与 signature；具体 ATen target 数可随版本 decomposition 变化。

实测摘要：

```text
symbolic_has_call_module=True
make_fx_has_call_module=False
dynamo_backend_graphs=1
dynamo_guards_recorded=True
dynamo_example_value_meta_recorded=True
explicit_graph_break_backend_graphs=2
export_input_kinds=PARAMETER,PARAMETER,USER_INPUT
export_range_constraints=0
```

此模型的 Linear 有 weight/bias 两个 parameters且未声明dynamic shape，所以最后两行正好反映
该输入/约束设计，不是Export对所有模型的固定数量。

持久 artifact 位于 `labs/artifacts/end_to_end/symbolic_fx.py`、`dynamo_fx.py`、
`dynamo_guards.txt`、`exported_program.py`与 `export_graph_signature.json`。自动合同
`CaptureFrontendContractTest`对正例、guard/meta 与 graph-break 边界做 assertion。环境和
命令见 [`labs/README.md`](labs/README.md)。

## 13. 选择指南

- 教学/模块结构改写：`symbolic_trace`；
- 精确 ATen execution trace：`make_fx`；
- `torch.compile`前端与 guards：Dynamo；
- 稳定部署/序列化 contract：`torch.export`；
- 需要 gradient graph：在相应捕获结果之上进入 AOTAutograd，而不是把 eager `grad_fn`
  当 FX Graph。

## 学习顺序

- 上一篇：[[06_structured_outputs_higher_order_and_nested_graphs]]
- 下一篇：[[08_graph_normalization_decomposition_and_functionalization]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[02_fx_graph_core_data_model]]
- [[04_symbolic_shapes_guards_and_graph_reuse]]
- [[08_graph_normalization_decomposition_and_functionalization]]
- [[09_aotautograd_joint_forward_backward_graphs]]
- [[14_fx_export_and_extensibility/index]]
- [[02_dynamo/index]]
