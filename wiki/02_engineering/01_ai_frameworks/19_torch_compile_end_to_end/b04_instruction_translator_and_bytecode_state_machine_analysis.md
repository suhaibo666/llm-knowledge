# B04 · InstructionTranslator 与字节码符号执行状态机

> 卷别：B · TorchDynamo 捕获  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[b03_eval_frame_callback_and_code_cache_analysis]]  
> 后续：[[b05_variable_tracker_source_and_python_object_model_analysis]]  
> 最后更新：2026-07-28

## 1. 为什么 Dynamo要解释 Python bytecode

直接运行原 Python函数并记录 Tensor op，只能看见“这一次实际执行了哪些算子”，却很难：

- 在执行前建立 Python变量与输入来源之间的 guards；
- 安全地处理 graph break和恢复栈/locals；
- 改写控制流以调用 compiled subgraph；
- 对 Python对象操作进行特化、内联或回退；
- 在不真实执行 Tensor计算的情况下传播 fake/proxy值。

所以 Dynamo建立了一个抽象解释器：按 CPython bytecode推进，但 operand stack和locals中保存
`VariableTracker`，Tensor计算则写入 FX graph。

**核心结论**：InstructionTranslator不是 FX interpreter；它解释的是 Python bytecode，
FX graph只是符号执行过程中累积的一个输出。

## 2. 状态机保存了什么

`InstructionTranslatorBase`的关键状态包括：

- `symbolic_locals`、`symbolic_globals`；
- `stack: list[VariableTracker]`；
- `instruction_pointer`与当前 instruction；
- `block_stack`；
- exception stack；
- inline parent/child关系；
- speculation checkpoint；
- 唯一共享的 `OutputGraph`。

字段定义见 `torch/_dynamo/symbolic_convert.py:1458-1483`。

构造时还保存输入 instructions、code options、真实 locals/globals/builtins、原 code object和
closure（`torch/_dynamo/symbolic_convert.py:5410-5432`）。

## 3. 根 translator和内联 translator的所有权

一个根 Python frame对应一个 root `InstructionTranslator`和一个 `OutputGraph`。被允许内联的
Python调用可创建 child translator，但继续写入根 OutputGraph。OutputGraph的类注释明确：
它与正在处理的 frame 1:1，内联 translator共享根输出
（`torch/_dynamo/output_graph.py:741-750`）。

这带来三层身份：

| 身份 | 负责 |
|---|---|
| root translator | 最终 transformed bytecode、根 locals/stack |
| inline translator | 被内联 Python调用的字节码状态 |
| OutputGraph/SubgraphTracer | FX nodes、inputs、guards、side effects、backend调用 |

“内联了 Python函数”不等于 FX graph里存在 `call_module`；其 body可能直接展开为算子 nodes。

## 4. 初始化如何连接 OutputGraph

根 `InstructionTranslator.__init__`创建 `OutputGraph`，把 compiler callback、frame state、
scope、原 code和 mode stack传入；然后把 OutputGraph交给基类
（`torch/_dynamo/symbolic_convert.py:5510-5539` 和
`torch/_dynamo/symbolic_convert.py:5540-5567`）。

因此 translator从一开始就同时拥有两套输出通道：

1. graph通道：Tensor/可图化操作写入 FX；
2. bytecode通道：调用 compiled region、恢复 Python状态、继续 eager。

## 5. `run → step → opcode handler`

`run()`压入当前 translator上下文，然后循环 `while self.step(): pass`
（`torch/_dynamo/symbolic_convert.py:2060-2069`）。

每个 `step()`：

1. 读取当前 instruction并推进 IP；
2. 更新源码行和 tracing位置；
3. 必要时检查一个 partial graph的 speculation结果；
4. 记录 bytecode日志；
5. 分派到对应 opcode handler。

前半段见 `torch/_dynamo/symbolic_convert.py:1673-1698`。

BytecodeDispatchTable metaclass让不同 Python版本的 opcodes映射到 handler方法。operand
stack的 push/pop效果必须与 CPython语义一致，否则 graph break后无法正确恢复。

## 6. 一个算子表达式如何经过状态机

以近似的 `y = torch.sin(x)`为例：

```text
LOAD_GLOBAL torch
  → Global/Module VariableTracker
LOAD_ATTR sin
  → callable VariableTracker
LOAD_FAST x
  → TensorVariable(Source=LocalSource("x"))
CALL
  → VariableTracker.call_function
  → OutputGraph.create_proxy("call_function", aten.sin, ...)
  → TensorVariable(proxy=<fx.Proxy>, example_value=<FakeTensor>)
STORE_FAST y
  → symbolic_locals["y"] = result tracker
```

状态机同时维护：

- Python值类别；
- 这个值从哪里来；
- 对应 FX proxy；
- fake example value；
- 为这次特化产生的 guards；
- 若无法图化，怎样还原成 Python bytecode。

## 7. Speculation为什么需要“失败后重启”

遇到可能不支持的 bytecode时，Dynamo不能在已经修改图、stack和side effects之后简单继续。
`break_graph_if_unsupported`先创建 speculation checkpoint，handler失败时：

1. 记录 graph-break reason；
2. 标记 speculation失败；
3. 抛出 restart-analysis；
4. 从 checkpoint按已知 break路径重新跑。

装饰器的 checkpoint/异常处理见 `torch/_dynamo/symbolic_convert.py:1081-1103`；
失败转为 restart见 `torch/_dynamo/symbolic_convert.py:1133-1148`。

外层 `transform_code_object`处于 attempt循环，捕获 `RestartAnalysis`后清理失败 graph并重试
（`torch/_dynamo/convert_frame.py:1592-1612`、
`torch/_dynamo/convert_frame.py:1613-1623`）。

设计原因是正确性：第一次尝试中的局部修改不能泄漏到重新选择的捕获路径。

## 8. Graph break不是“暂停解释器”

允许 partial graph时，translator会：

- 结束当前 FX subgraph；
- 生成调用 compiled subgraph的 bytecode；
- 重建当前 stack、locals、block/context状态；
- 生成/查找 resume code object；
- 让不支持的操作在 Python中执行；
- 在后续 resume frame上再次进入 Dynamo。

所以 graph break是一次程序变换，而不是在原 frame内临时暂停 FX recording。

## 9. Transformed code怎样形成

translator运行结束后，`OutputGraph.output_instructions`替换原 instruction列表，随后：

- 更新 code options；
- 传播 exception-table信息；
- 校验 exception-table；
- 删除死 bytecode和无意义跳转。

见 `torch/_dynamo/convert_frame.py:967-980`。

这一步的“dead code”是字节码控制流层的不可达/无用指令，不是 FX graph DCE，也不是
AOTAutograd保存激活的生命周期。

## 10. 不变量与失败边界

- symbolic operand stack必须模拟 CPython stack effect；
- 每个 VariableTracker必须能参与图化、守卫、重建或明确失败；
- speculation rollback必须覆盖 graph、guards、side effects和translator状态；
- inline translator不能独占根输出；
- 输出 bytecode必须满足新旧 code的参数/cell/freevar布局约束；
- fullgraph模式下不允许通过 partial graph掩盖 unsupported；
- resume prologue tracing中的失败不能再次被普通 graph break吞掉。

## 11. 复杂度

令：

- \(B\)：被处理字节码数，包括内联 body；
- \(R\)：restart次数；
- \(V\)：跟踪值/容器结构规模；
- \(G\)：生成 FX nodes。

理想单次符号执行约为 \(O(B+V+G)\)。有 speculation restart时上界近似：

\[
O\left(\sum_{r=1}^{R+1}(B_r+V_r+G_r)\right)
\text{backend compile}
\text{guard build}
\text{bytecode transform}
\]

若同一长前缀多次重跑，捕获成本会高于线性单遍。VariableTracker缓存、source去重和
speculation log用于降低重复工作与保证决策一致。

## 12. 常见误解

- **“Dynamo逐个遍历 FX node来构图。”** Dynamo先逐个解释 Python bytecode；FX nodes是
  某些 handler的副产物。
- **“InstructionTranslator就是 `torch.fx.Interpreter`。”** 前者解释 CPython bytecode，
  后者解释已有 FX graph。
- **“graph break后原 frame接着跑就行。”** 必须显式保存/恢复 stack、locals和block状态。
- **“restart说明系统失败了。”** restart是 speculative tracing的正常控制机制之一。
- **“输出只有 GraphModule。”** 输出还包含 guards、transformed code和相关元数据。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[a03_python_frames_code_objects_and_bytecode_analysis]]
- [[b03_eval_frame_callback_and_code_cache_analysis]]
- [[b05_variable_tracker_source_and_python_object_model_analysis]]
- [[b06_output_graph_side_effects_and_graph_emission_analysis]]
- [[b08_graph_break_resume_functions_and_partial_graphs_analysis]]
