# B08 · Graph Break、Resume Function 与 Partial Graph

> 卷别：B · TorchDynamo 捕获  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[15_guards_cache_lookup_and_recompilation_analysis]]  
> 后续：[[17_dynamic_shapes_generalization_and_fallback_analysis]]  
> 最后更新：2026-07-28

## 1. 为什么 graph break不是简单的“停止追踪”

在 break点，Python frame已经拥有：

- operand stack；
- locals、cells、free variables；
- block stack和活动context；
- 已累积的FX graph；
- 已记录但尚未回放的side effects；
- 当前instruction之后的剩余程序。

如果只停止记录并直接让原frame继续，已经被图化的前缀可能重复执行，栈状态也无法和原
instruction边界对齐。

**核心结论**：graph break是一项控制流重写：提交前缀图、生成调用字节码、还原Python
状态、执行不支持部分，再通过新code object从后续offset恢复捕获。

## 2. 哪些场景会触发 break

- 明确调用 `torch._dynamo.graph_break()`；
- 不支持的Python bytecode或对象操作；
- 数据依赖且无法图表达的控制流；
- 无法安全处理的mutation/alias/context；
- 特定higher-order op未完整捕获；
- 用户策略要求切图；
- 某些错误在partial模式被降级为break。

break原因由 `GraphCompileReason`保存，包括reason、user stack和是否真为graph break
（`torch/_dynamo/output_graph.py:333-344`）。

## 3. Speculation先失败，再按break路径重跑

带 `break_graph_if_unsupported`的opcode handler先在speculation checkpoint上尝试。
若抛出 `Unsupported`/`UserError`且允许partial：

1. 记录日志与reason；
2. 标记speculation失败；
3. restart analysis；
4. 再次到达该点时走已知break路径。

异常转reason与restart逻辑见 `torch/_dynamo/symbolic_convert.py:1124-1148`。

这避免第一次尝试留下半构造graph或半应用side effects。

## 4. Break点怎样结束当前图

已知break路径调用 `OutputGraph.compile_subgraph`：

- 传入当前translator；
- 记录break reason；
- 根据opcode stack effect决定哪些值要弹出/重建；
- 得到每层frame的stack/locals metadata。

调用点见 `torch/_dynamo/symbolic_convert.py:1150-1176`。

`compile_subgraph`随后生成：

```text
compiled prefix call
→ graph outputs
→ side-effect replay
→ stack/locals/cells reconstruction
→ unsupported instruction / resume call
```

## 5. 为什么需要resume function

CPython不能从任意bytecode offset创建一个普通函数并自动获得旧frame的全部中间状态。
Dynamo生成一个新code object，其prologue把：

- live locals；
- operand stack；
- cells/freevars；
- active block/context state；
- NULL slots；

还原成“从break之后继续执行”所需布局。

`ContinueExecutionCache.lookup`按原code和恢复状态key查找/生成resume code
（`torch/_dynamo/resume_execution.py:328-341`、
`torch/_dynamo/resume_execution.py:343-366`）。

## 6. Resume cache的key为何很大

同一源码offset不一定对应同一恢复需求。key还包含：

- resume offset；
- block target offsets；
- stack深度；
- locals参数名；
- NULL位置；
- context-manager reentry信息；
- nested code objects；
- 当前instruction是否向stack push。

translator调用lookup时传入这些字段，见
`torch/_dynamo/symbolic_convert.py:3544-3565`。

因此resume cache不能只用 `(code, offset)`；否则不同栈/上下文形态会错误共享code。

## 7. Partial graph的真实执行形态

假设源码为：

```python
a = tensor_region_1(x)
python_only(a)
b = tensor_region_2(a)
return b
```

实际可近似变换为：

```text
transformed original code:
  a = compiled_region_1(x)
  call python_only(a)
  return resume_fn(a, live_locals...)

resume_fn:
  b = compiled_region_2(a)
  return b
```

两张FX region图没有直接的普通FX边跨过Python段；值通过transformed bytecode的参数/locals
传递。这里的“跨图接口”与AOTAutograd fw输出saved values、bw placeholders的机制相似于
显式接口化，但数据结构完全不同。

## 8. 内联frame中的break更复杂

若break发生在被内联函数：

- 最终输出code仍属于root frame；
- 必须逐层恢复child到root的stack/locals；
- source position要映射回root输出code；
- 可能为嵌套resume函数安装正确globals/closure；
- 某些active context或loop backedge无法安全恢复，只能跳过整个frame。

源码在处理break时特意用root frame当前位置给输出code映射源码
（`torch/_dynamo/symbolic_convert.py:1183-1191`）。

## 9. 循环与backedge为什么限制partial graph

在循环内部break后，如果错误恢复：

- 可能每次迭代生成新图；
- resume offset与loop state组合爆炸；
- side effects重复；
- control-flow语义变化。

translator用启发式检测backedge；注释明确说明false positive偏向跳过frame，false negative
可能在循环break时生成多个图（`torch/_dynamo/symbolic_convert.py:1503-1521`）。

这是保守正确性与捕获覆盖率的权衡。

## 10. `fullgraph=True`改变什么

fullgraph/one_graph模式下，unsupported不能转成partial graph；原原因向上抛出。它用于：

- 建立“不能有break”的API契约；
- 把静默切图转为可定位错误；
- 避免用户误以为整段都优化；
- export等要求单图的流程。

它不使unsupported自动可表达，也不消除backend内部partition。

## 11. Break后的排序与DCE

不会存在一次“把所有partial graphs与Python代码合起来topo sort”的步骤：

- 每个FX region内部保持自身graph拓扑；
- region之间由transformed bytecode顺序连接；
- live值通过Python call参数、locals和resume prologue传递；
- 每个region可独立被backend优化/DCE；
- bytecode层另做dead-code和jump清理。

所以讨论“break后重排序”必须指出是在FX region、backend IR还是Python bytecode层。

## 12. 复杂度

设有 \(P\) 个compiled regions，break \(P-1\) 次，第 \(j\) 个break的live state为 \(L_j\)：

- graph compile成本为 \(\sum_j K(G_j)\)；
- resume codegen/参数传递约为 \(\sum_j O(L_j)\)；
- runtime多出region call、Python transition和guard开销；
- resume cache可复用相同恢复形态，miss时才生成；
- 频繁break可能降低图融合范围并显著增加launch/wrapper overhead。

graph break数量本身不是完整性能指标；位置、live state、被切断的融合机会更重要。

## 13. 常见误解

- **“graph break后Dynamo彻底退出。”** 后续resume frame可再次被捕获。
- **“break只把FX graph切成两段。”** 还需要Python stack/locals/context的显式重建。
- **“两个partial graphs通过FX edge连接。”** 中间值经transformed bytecode接口传递。
- **“同offset总能复用同一个resume。”** 恢复key还依赖栈、block、NULL和context形态。
- **“没有报错就等于没有break。”** partial模式可合法切图并继续执行。

## 配套 Demo

本页对应卷级入口 `tools/labs_torch_compile/demo_b_dynamo_capture.py` 的 `graph_break_resume` 用例。默认以 CUDA 为验收设备：

```powershell
python -B tools\labs_torch_compile\demo_b_dynamo_capture.py `
  --case graph_break_resume --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\b08
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `graph_break_resume/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[12_instruction_translator_and_bytecode_state_machine_analysis]]
- [[14_output_graph_side_effects_and_graph_emission_analysis]]
- [[17_dynamic_shapes_generalization_and_fallback_analysis]]
- [[dynamo_explain_and_graph_break_diagnosis_analysis]]
- [[20_graph_stage_boundaries_identity_and_provenance_analysis]]
