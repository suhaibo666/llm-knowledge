# A03 · Python Frame、Code Object 与 Bytecode：Dynamo 为什么从这里捕获

> 卷别：A · 执行模型前置基础  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[a02_operator_schema_dispatch_and_autograd_analysis]]  
> 后续：[[a04_dispatch_modes_proxy_tensor_and_fake_tensor_analysis]]  
> 最后更新：2026-07-28

## 1. 为什么不只在 Tensor operator 层追踪

模型代码里决定计算的不只有 Tensor operator：

```python
if x.shape[0] > limit:
    y = self.large_path(x)
else:
    y = self.small_path(x)
return {"value": y, "tag": self.mode}
```

operator dispatch 能看到实际执行的 Tensor op，却无法独立还原：

- 哪条 Python `if`选择了路径；
- `self.mode`来自哪个对象属性；
- dict/tuple 如何构造；
- 函数调用是内联、跳过还是 graph break；
- 执行结束后从哪条指令继续。

**核心结论**：Dynamo 选择 Python frame/bytecode，是为了在执行 Tensor op 的同时保留
Python 控制流、对象来源和恢复点；代价是它必须实现一套 Python 字节码的符号解释器。

## 2. Code object、frame 与 instruction

| 对象 | 生命周期 | 关键状态 |
|---|---|---|
| code object | 函数实现可复用 | bytecode、consts、names、varnames、flags、line table |
| frame | 一次具体调用 | code、locals/globals/builtins、value stack、instruction offset |
| instruction | code 中的一个操作 | opcode、arg、offset、jump target、source position |
| transformed code | 某组 guards 下的可执行替代 | compiled subgraph calls、resume calls、原 Python 残余 |

同一 code object 可以产生许多 frame；每个 frame 有不同 inputs/locals，但 bytecode
结构相同。这正好对应“同一程序结构，按输入 guard 选择多个 specialization”。

## 3. eval-frame hook 是什么

PyTorch C 扩展保存原有 eval-frame function，并在启用 Dynamo 时把 interpreter 的 frame
evaluator 换成 `dynamo_custom_eval_frame_shim`
（`torch/csrc/dynamo/eval_frame.c:226-253`）。

未处理或禁用时，shim 可以转回之前 evaluator 或 CPython default evaluator
（同文件 `:233-244`）。因此 Dynamo 不是替换整个 Python interpreter，而是在 frame
进入执行时获得一次“使用原代码还是转换代码”的选择权。

### callback 的三种状态

`set_eval_frame`源码给出：

- `None`：关闭 Dynamo；
- `False`：run-only，只复用已有编译；
- Python callable：启用转换。

见 `torch/csrc/dynamo/eval_frame.c:616-638`。参数必须是这三类之一
（同文件 `:652-663`）。

这比一个简单的 on/off flag 更强：系统可以停止产生新 specialization，但继续使用
已存在的 compiled entries。

## 4. 为什么 cache 属于 code object

Dynamo cache 是 linked list；每个 entry 包含 guard manager、输出 code 和 next pointer，
并存储在 `f_code`的 `co_extra`空间。frame 调用时遍历 entries，逐个运行 guards；
没有命中才重新编译并增加 entry
（`torch/_dynamo/cache_size.py:13-21`）。

所以要区分：

- **code identity**：哪个 Python 程序共享 cache；
- **frame state**：本次 locals/globals/inputs 是什么；
- **cache entry**：哪组 guards 与 transformed code 配对；
- **compiled callable/cache**：transformed code 内部调用的图产物。

公开 API 说明也明确指出 cache 是 per-code-object，而不是 per-frame
（`torch/__init__.py:3157-3166`）。

### 为什么不是给每个 frame 单独缓存

frame 是一次调用的临时执行状态，调用结束即失去复用价值。code object 才能让后续调用
用 guards 判断是否可重用。若 cache 挂在 frame 上，每次调用都会冷启动；若全局只按函数
名字缓存，又会混淆闭包、代码重定义和不同 code identity。

## 5. Dynamo 的 mutable Instruction

PyTorch 定义的 `Instruction`是 `dis.Instruction`的可变版本，除 opcode/opname/arg/offset
外，还显式保存 jump target 和 exception table entry
（`torch/_dynamo/bytecode_transformation.py:71-95`）。

使用对象 target 而不是只保存原始字节 offset，可以在插入/删除指令后重新计算跳转。
Instruction equality 采用 identity，也符合“这是可重写程序位置，不是按字段合并的值”。

## 6. 从 code object 到可重写 instructions

`cleaned_instructions()`先从缓存取标准化 instructions，再 clone 一份，因为后续转换会
原地修改 instruction array
（`torch/_dynamo/bytecode_transformation.py:1899-1909`）。

缓存构建路径：

```text
code object
  → dis.get_instructions
  → convert_instruction
  → line number propagation
  → exception table / jump virtualization
  → strip extended args
```

见 `torch/_dynamo/bytecode_transformation.py:1944-1959`。

“virtualize jump”的意义是把原始数字 offset 变成 Instruction target；真正重新组装前再
根据新位置 devirtualize。

## 7. 代码重写状态机

`transform_code_object()`执行：

1. 从原 code object复制 code options；
2. 生成 cleaned instructions；
3. 运行 transformations；
4. 清理并重新 assemble；
5. 返回新 code object和 tracer output。

对应入口见 `torch/_dynamo/bytecode_transformation.py:1824-1845`。

assemble 前要：

- 检查 exception table；
- 修复 locals；
- 重复更新 offsets、devirtualize jumps、修复 extended args，直到 offset 稳定；
- 重建 bytecode、line table、stack size 和 exception table。

实现见 `torch/_dynamo/bytecode_transformation.py:1848-1875`。

**设计原因**：插入一个 compiled-graph call 可能改变后续 instruction offsets；offset
变化又可能使 jump 参数需要更多字节。一次线性写出不能保证稳定，因此要迭代修复布局。

## 8. 符号解释器必须保存哪些 frame 状态

InstructionTranslator初始化时保存：

- `symbolic_locals` / `symbolic_globals`；
- value `stack`；
- instruction pointer/current instruction；
- block stack、active context managers、exception VT stack；
- original `f_locals/f_globals/f_builtins`；
- code options 与 `f_code`。

源码位置：

- mutable execution state：`torch/_dynamo/symbolic_convert.py:5377-5401`；
- instructions 与 frame namespaces：同文件 `:5410-5422`。

这不是把 Python VM 全部复制一遍，而是保存“继续解释、回滚 speculation、生成 residual
bytecode 和 resume function”所需的最小状态集合。

## 9. 一次 frame 调用的端到端形态

```mermaid
flowchart TD
    C["code object"] --> F["新 frame 与 locals"]
    F --> E["eval frame shim"]
    E --> L{"cache entry guard 命中"}
    L -->|是| O["执行 transformed code"]
    L -->|否| T["ConvertFrame 与 symbolic execution"]
    T --> G["FX region + guards"]
    G --> N["新 cache entry"]
    N --> O
    O --> R["原 Python 或 resume code"]
```

注意：transformed code 不是“整函数机器码”。它仍是 Python code object，只是把可编译
region 替换成对 compiled callable 的调用，并保留/生成必要的 Python residual path。

## 10. 为什么 graph break 需要 bytecode 层

graph break 的目标不是简单停止 tracing，而是：

1. 编译 break 之前已积累的 FX region；
2. 在原 Python 语义下执行不支持的部分；
3. 恢复 value stack、locals、context managers；
4. 从准确的 bytecode offset 继续捕获。

只有 operator trace 没有 instruction pointer、stack 和 control context，无法生成正确
resume program。卷 B08 会继续追踪 `ContinueExecutionCache`。

## 11. 不变量与失败边界

- transformed code 的 locals 数必须与 `co_varnames`一致；
- jump target 必须引用当前 instruction array；
- exception table entries 必须有效；
- stack-size analysis 必须覆盖重写后控制流；
- generator/coroutine 的 resume 语义有额外限制；
- code object/cache identity 不能用函数名代替；
- Python 版本改变 opcode/exception-table 格式，源码结论必须绑定版本与 commit。

## 12. 复杂度

设 code object 有 \(I\) 条 instructions、cache entries 数为 \(C\)：

- 初次 disassemble/standardize 为 \(O(I)\)，可由 cleaned-instruction cache 复用；
- 每次 transformation 至少读取/修改 \(O(I)\)；
- offset/extended-arg fixpoint 若迭代 \(q\) 次，为 \(O(qI)\)；
- cache lookup 最坏运行 \(C\) 组 guard managers；guard 内部成本另按 guard 数和表达式计算；
- frame symbolic execution 的主干与实际解释的 instructions/inline calls 成正比，Tensor
  kernel 成本不在这部分。

## 13. 常见误解

| 误解 | 修正 |
|---|---|
| `torch.compile(fn)`立即编译函数 | wrapper 创建与首次 frame 执行是两个时刻 |
| cache 挂在每次 frame 上 | compiled entries 挂在 code object 的 `co_extra` |
| Dynamo 只记录 Tensor op | 它符号执行 Python instructions 和对象状态 |
| graph break 后从函数头重来 | resume code 从保存的 program point 恢复 |
| transformed code 就是 native kernel | 它是调用 compiled regions 的新 Python code object |

## 14. 源码跟读：首次调用如何把 Python frame 变成 GuardedCode

这里沿 public API 走到 bytecode translator，再回到 code cache。最容易混淆的两个时刻是：
`torch.compile(fn)` 创建 wrapper，而 **wrapper 第一次执行** 才有真实 frame、locals 和输入，
因而才可能捕获与编译。

```mermaid
flowchart LR
    A["torch.compile"] --> W["optimized wrapper"]
    W --> H["temporary eval frame callback"]
    H --> C["C frame hook and code cache"]
    C -->|cache miss| F["ConvertFrame"]
    F --> T["InstructionTranslator"]
    T --> B["bytecode transform and FX region"]
    B --> G["GuardedCode"]
    G --> C
    C -->|cache hit| X["transformed code"]
```

### 14.1 public API 只先组装 wrapper

`torch.compile` 的文档明确说明 compiled results 按 code object 缓存，guard failure 会产生
重编译并受 cache-size limit 约束（`torch/__init__.py:3134-3166`）。函数末尾根据 backend
配置包装 callable，再调用 `torch._dynamo.optimize(...)(model)` 返回优化后的对象
（`torch/__init__.py:3361-3378`）。这里尚没有用户调用 frame，所以不能完成针对 locals、
shape 和对象状态的 guard 构建。

优化 wrapper 的 `_fn` 在真正调用原函数前设置 eval-frame callback，调用结束后恢复之前的
callback（`torch/_dynamo/eval_frame.py:1480-1505`）。`optimize` 与 `_optimize` 则负责把
backend、dynamic、guard hooks 等选项组装成这一入口
（`torch/_dynamo/eval_frame.py:1726-1761`）。callback 是动态作用域，而不是永久替换
整个解释器的执行函数。

### 14.2 C frame hook 是捕获与 cache 的边界

C 扩展把 callback 存在线程局部状态；`None`、`False` 和 callable 分别表示不同 eval-frame
策略（`torch/csrc/dynamo/eval_frame.c:616-638`）。当 frame 到达时，Python 侧
`CatchErrorsWrapper` 先检查 frame 是否应跳过、当前 compile mode 是否允许处理等边界
（`torch/_dynamo/convert_frame.py:2517-2561`），再进入真正的 ConvertFrame。

`ConvertFrameAssert.__call__` 接收 code object、cache entries 与 cache size 等信息
（`torch/_dynamo/convert_frame.py:632-656`）。这说明 cache identity 的锚点是 code
object 及其 entry 集合，不是一次性的 frame 实例；frame 提供本次 locals/stack，
code object 提供可复用程序身份。

### 14.3 `_compile` 为什么同时需要 instructions 和运行时状态

`_compile` 的参数契约同时包含 code、globals、locals、builtins、compiler function、
one-graph/export 选项和 frame state
（`torch/_dynamo/convert_frame.py:1647-1680`）。只读取 bytecode 不能决定 Tensor
shape、Python object identity 或分支结果；只读取运行时对象又没有 instruction pointer
和控制流结构。Dynamo 必须把两者放进同一次 symbolic execution。

内部 `compile_inner` 返回 transformed code、OutputGraph 等结果，随后更新 code state
（`torch/_dynamo/convert_frame.py:2115-2138`）。ConvertFrame 外层再把这些行为包入统一
异常与 replay/diagnostic 处理，而不是让每个 opcode handler自行决定 fallback
（`torch/_dynamo/convert_frame.py:2295-2337`）。

### 14.4 InstructionTranslator 是 bytecode 状态机，不是 operator recorder

translator 的 `step` 每次读取一条 instruction，维护当前 instruction、speculation
状态，并在需要时处理 graph break
（`torch/_dynamo/symbolic_convert.py:1673-1705`）。`run` 驱动整个 instruction loop，
还负责到达 fallback boundary 后的处理
（`torch/_dynamo/symbolic_convert.py:2060-2085`）。translator 初始化时还建立当前
translator 的线程局部上下文，供嵌套 tracing/inline 等路径使用
（`torch/_dynamo/symbolic_convert.py:5493-5515`）。

所以它的输出不只是 ATen Node 列表：locals、value stack、side effects、sources、
guards 和 resume point 都参与 transformed program 的生成。operator graph 只是这次
Python 符号执行中可编译区域的一个产物。

### 14.5 transformed code 与 guards 如何形成可缓存结果

bytecode transformer 先取得并清理 instructions，调用变换函数，再修复并组装新的
code object（`torch/_dynamo/bytecode_transformation.py:1824-1845`）。清理后的
instructions 可以缓存，但每次使用必须 clone，因为 transformation 会原地修改它们
（`torch/_dynamo/bytecode_transformation.py:1899-1915`）。

编译完成后，ConvertFrame 构建 guard manager，并把 transformed code 与 guard check
封装成 `GuardedCode`（`torch/_dynamo/convert_frame.py:1903-1938`）。cache hit 的含义
由此变得精确：不是“函数编过就无条件复用”，而是当前 frame 状态通过某个 entry 的
guards 后，执行该 entry 的 transformed code。

### 14.6 设计结论

1. wrapper 把捕获延迟到有真实 frame 的调用时刻。
2. C frame hook 负责解释器边界和 code-object cache，Python ConvertFrame 负责编译策略。
3. InstructionTranslator 模拟 Python 机器状态，因此能在 graph break 后生成正确 resume。
4. GuardedCode 把“优化后的程序”和“它成立的前提”作为一个 cache entry 保存。
5. 因而 Dynamo graph 不能脱离 transformed bytecode、guards 和 resume code 单独解释。

## 配套 Demo

本页对应卷级入口 `labs/demo_a_execution_model.py` 的 `python_frame_bytecode` 用例。默认以 CUDA 为验收设备：

```powershell
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\demo_a_execution_model.py `
  --case python_frame_bytecode --device cuda `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\volume_demos\a03
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `python_frame_bytecode/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[a02_operator_schema_dispatch_and_autograd_analysis]] — operator 执行层
- [[a04_dispatch_modes_proxy_tensor_and_fake_tensor_analysis]] — operator-level transform
- [[b03_eval_frame_callback_and_code_cache_analysis]] — eval-frame 与 cache 深入
- [[b04_instruction_translator_and_bytecode_state_machine_analysis]] — 符号解释器
- [[b08_graph_break_resume_functions_and_partial_graphs_analysis]] — resume code
- [[02_dynamo/index]] — Dynamo 领域资料
