# B06 · OutputGraph、SideEffects 与“FX 图 + 残余字节码”双输出

> 卷别：B · TorchDynamo 捕获  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[b05_variable_tracker_source_and_python_object_model_analysis]]  
> 后续：[[b07_guards_cache_lookup_and_recompilation_analysis]]  
> 最后更新：2026-07-28

## 1. 为什么 Dynamo不能只输出一张 FX graph

Python程序包含FX图难以完整表达的语义：

- 任意Python对象和容器；
- 局部/全局变量；
- 对象属性、cell和closure；
- graph break；
- 异常与block stack；
- 副作用的发生顺序；
- 调用 compiled callable前后的状态恢复。

因此一次 frame转换必须同时产生：

1. 可交给 backend的 `fx.GraphModule`；
2. 调用它并保留剩余 Python语义的 transformed bytecode；
3. 保护这份转换的 guards。

**核心结论**：OutputGraph虽以“Graph”命名，却是 Dynamo frontend的聚合输出对象；它不仅
持有 FX graph，还持有 guards、side effects、source映射、example inputs和输出字节码。

## 2. OutputGraph的所有权边界

源码规定一个 OutputGraph对应一个正在处理的根 frame；内联函数继续写入根 OutputGraph
（`torch/_dynamo/output_graph.py:741-750`）。

初始化时它建立：

- guards/global state；
- root `SubgraphTracer`；
- source到input tracker的去重表；
- frame state；
- compile id；
- 当前 scopes和mode stack。

见 `torch/_dynamo/output_graph.py:769-785` 与
`torch/_dynamo/output_graph.py:786-799`。

高阶算子可以创建嵌套 SubgraphTracer，但 lineage必须保存；OutputGraph的
`create_proxy/create_node`只是转发到 current tracer
（`torch/_dynamo/output_graph.py:1452-1479`）。

## 3. 四类数据同时流入 OutputGraph

| 数据 | 生产者 | 最终消费者 |
|---|---|---|
| FX nodes/proxies | VariableTracker opcode/call handlers | backend |
| example/fake values | FakeTensor/ProxyTensor包装 | backend、shape推理、校验 |
| Sources/guards | VariableBuilder/operation specialization | code cache lookup |
| side effects/output bytecode | Python符号执行 | transformed CPython code |

这四类数据有关联，但不能互相替代。比如FX graph里的parameter `get_attr`仍需要Source和
guard；一个Python list mutation可能没有对应FX node，却必须在compiled region后回放。

## 4. 参数、buffer和外部Tensor如何进入图

`register_attr_or_module`根据对象和tracer层级选择：

- 动态module按普通 VariableTracker构建；
- parameter/buffer可在根图注册为 `get_attr`；
- 高阶子图中把值提升为输入，而在根图保留 `get_attr`；
- 对对象安装 `ID_MATCH`或 `TENSOR_MATCH` guard；
- 用 identity table避免重复注册。

关键决策见 `torch/_dynamo/output_graph.py:1671-1690`、
`torch/_dynamo/output_graph.py:1691-1705` 与
`torch/_dynamo/output_graph.py:1707-1735`。

这解释了为什么 FX graph的存储形式由可序列化性、module ownership、高阶图边界和
specialization共同决定，而不仅是“所有Tensor都做placeholder”。

## 5. SideEffects为什么是独立表

`SideEffects`记录并在codegen时应用：

- Python object/list/dict mutation；
- attribute set/delete；
- cell/global changes；
- tensor hooks和backward相关状态；
- 对象identity与keepalive。

职责说明见 `torch/_dynamo/side_effects.py:106-123`，核心存储字段见
`torch/_dynamo/side_effects.py:126-145` 与
`torch/_dynamo/side_effects.py:146-167`。

独立记录的原因：

- FX graph主要描述值依赖，不足以描述所有 Python heap effect；
- tracing时不能真的把用户对象改到最终状态后再假装可回滚；
- graph break需要checkpoint/rollback；
- compiled graph执行后必须按Python顺序回放；
- dead temporary object的effects可以在确认不可观察后剪枝。

## 6. `compile_subgraph`是双输出汇合点

当返回、graph break或其他原因结束当前区域时，`compile_subgraph`负责：

- 编译当前 subgraph；
- 生成调用 compiled subgraph的字节码；
- 应用 side effects；
- codegen stack/locals；
- 保存/恢复locals。

源码docstring见 `torch/_dynamo/output_graph.py:2053-2071`。

它首先停止 bytecode tracing、记录 compile reason、标记 `should_exit`
（`torch/_dynamo/output_graph.py:2092-2098`），然后整理不同frame层级的 stack/locals、
context和NULL槽位。

## 7. FX graph如何交给 backend

在 backend调用前，OutputGraph：

1. 清理 graph；
2. 构造 root module/GraphModule；
3. 整理 outputs和example inputs；
4. 运行 `gm.graph.lint()`；
5. 恢复必要的global state；
6. 调用 `call_user_compiler(gm, example_inputs)`。

lint和调用位置见 `torch/_dynamo/output_graph.py:3037-3052`。

`_call_user_compiler`统计op/placeholder，回填 Source元数据，处理调试backend override，
最终执行：

```python
compiled_fn = compiler_fn(gm, example_inputs)
assert callable(compiled_fn)
```

见 `torch/_dynamo/output_graph.py:3217-3238`、
`torch/_dynamo/output_graph.py:3239-3248` 与
`torch/_dynamo/output_graph.py:3286-3293`。

## 8. Compiled callable怎样嵌回 bytecode

`compile_and_call_fx_graph`生成“调用已编译函数”的 instructions，并返回可选Python形式的
codegen文本（`torch/_dynamo/output_graph.py:2852-2867`）。

`compile_subgraph`再把：

1. compiled call instructions；
2. graph outputs暂存；
3. side-effect replay；
4. stack/locals reconstruction；
5. resume或剩余原指令；

按顺序加入 `output_instructions`。调用图与后续codegen的拼接点见
`torch/_dynamo/output_graph.py:2376-2396`。

`add_output_instructions`同时维护bytecode和可选pycode，并设置 `should_exit`
（`torch/_dynamo/output_graph.py:3550-3564`）。

## 9. Graph cleanup和DCE发生在哪些层

这里至少有三种“死内容清理”：

| 层 | 对象 | 语义 |
|---|---|---|
| SideEffects pruning | 图内新建Python对象 | 不再可观察的临时对象effects |
| FX cleanup/DCE | `fx.Node` | 无用户且无副作用语义的图节点 |
| bytecode dead-code removal | `Instruction` | 改写后不可达/无用指令 |

`compile_subgraph`先剪临时对象effects再清理 graph
（`torch/_dynamo/output_graph.py:2144-2164`）。translator完成后又删除 dead bytecode
（`torch/_dynamo/convert_frame.py:967-980`）。

它们的数据结构、liveness根和副作用定义不同，不能统称为一次DCE。

## 10. Guards如何和输出绑定

bytecode transform完成后：

- `DynamoOutput.build_guards`用 OutputGraph guards构造 `CheckFunctionManager`
  （`torch/_dynamo/convert_frame.py:1002-1021`）；
- 生成的 transformed code与 `check_fn.guard_manager`组合为 `GuardedCode`
  （`torch/_dynamo/convert_frame.py:1931-1948`）；
- C++把它保存进 code-object CacheEntry。

所以 FX graph本身不携带完整的“何时可运行”契约；Dynamo code cache entry才把
guard manager、transformed code和backend绑定起来。

## 11. 保序机制

Dynamo不靠一次全局topological sort恢复Python语义。主要依赖：

- translator按bytecode顺序处理；
- FX graph通过数据依赖和effect tokens/显式副作用规则保持图内顺序；
- SideEffects按记录与codegen规则回放；
- `output_instructions`按append顺序拼接；
- stack/locals reconstruction位于unsupported instruction之前；
- transformed bytecode的control flow和exception table接受校验。

后端可以对纯计算图做拓扑合法的重排，但不得越过alias、mutation、random、collective或
其他可观察effect边界。

## 12. 复杂度

设：

- \(G\)：FX nodes；
- \(S\)：记录的side effects；
- \(L\)：live stack/locals规模；
- \(I\)：生成bytecode数。

典型 frontend emission为 \(O(G+S+L+I)\)。FX lint/cleanup通常至少线性扫描图；
side-effect pruning随tracked object/effect图增长；stack/locals codegen在graph break多且
live state大时会显著膨胀。backend编译成本另计，通常高于这部分。

## 13. 常见误解

- **“OutputGraph就是fx.Graph。”** 它是包含graph、guards、side effects和bytecode的容器。
- **“FX output node之后Python函数就结束。”** transformed bytecode仍可能继续运行。
- **“所有副作用都在FX边中表示。”** 许多Python heap effects由SideEffects单独回放。
- **“每次pass替换后Dynamo统一topo sort。”** Dynamo emission与后端FX passes是不同阶段；
  FX节点插入位置、依赖和lint/DCE共同维护合法性。
- **“DCE只看node.users为空。”** 可观察副作用、output、mutation和跨层runtime语义都会
  影响能否删除。

## 配套 Demo

本页对应卷级入口 `tools/labs_torch_compile/demo_b_dynamo_capture.py` 的 `output_graph_side_effects` 用例。默认以 CUDA 为验收设备：

```powershell
python -B tools\labs_torch_compile\demo_b_dynamo_capture.py `
  --case output_graph_side_effects --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\b06
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `output_graph_side_effects/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[b05_variable_tracker_source_and_python_object_model_analysis]]
- [[b07_guards_cache_lookup_and_recompilation_analysis]]
- [[b08_graph_break_resume_functions_and_partial_graphs_analysis]]
- [[02_fx_graph_core_data_model]]
- [[14_dead_code_topology_and_effect_order]]
