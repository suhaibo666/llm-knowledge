# Dynamo 图改写与 Backend 开发方法论

> **Created**: 2026-07-22

> **Source baseline**: PyTorch `9922478dffa`，核验入口 `torch/_dynamo/backends/registry.py:81-157`、`torch/_dynamo/backends/inductor.py:19-30`、`torch/_dynamo/eval_frame.py:1527`。
>
> **结论先行**：Dynamo 没有一个与 Inductor `pre_grad_custom_pass` 对称的“通用内部 Pass 队列”。它的稳定扩展形态是 `torch.compile(backend=callable)`：Dynamo 捕获一张带 guards 的 FX 图，把 `(GraphModule, example_inputs)` 交给 backend，backend 返回可调用的编译结果。

---

## 1. 是什么

Dynamo 在 Python frame 层做符号执行，把一段可捕获的 Python 程序变成 FX `GraphModule`，同时生成使本次特化成立的 guards。之后 `lookup_backend()` 把字符串 backend 展开成 callable，并调用它。

```text
Python frame
  └─ Dynamo symbolic execution
       ├─ FX GraphModule
       ├─ guards / graph breaks
       └─ backend(gm, example_inputs) -> compiled callable
```

所以这里的“Pass”有两种含义：

1. **Dynamo 自身的捕获与规范化逻辑**：属于前端内部实现；
2. **backend 内对整张捕获图的检查/改写**：是第三方可以接入的扩展边界。

不要把 `allow_in_graph`、`disable` 当成图优化 Pass。它们改变的是捕获边界或 Dynamo 对 callable 的处理方式，不负责将一段 FX 子图融合成 kernel。

---

## 2. 为什么在 Dynamo 做

只有 Dynamo 阶段还能回答下面的问题：

- 哪段 Python 因 graph break 没进入图？
- 哪些 Python 值被特化并进入 guard？
- 一个捕获区域是否应该直接拒绝、回退或交给另一编译器？
- 模块/调用边界在进入 AOTAutograd 前是什么样？

如果优化依赖这些信息，放到 Pre-Grad 已经太晚，因为 Pre-Grad 只能处理**成功捕获并送入 Inductor 的图**。

反过来，如果规则依赖 functionalized ATen overload、前反向联合图、layout 或 buffer dependency，就不该放 Dynamo；这些信息尚未产生。

---

## 3. 适合做什么

| 场景 | 是否适合 | 原因 |
|---|---|---|
| 图捕获审计、打印/统计目标节点 | 适合 | backend 直接收到整张 Dynamo FX 图 |
| 选择另一个编译器或按图特征路由 | 适合 | backend 本来就是编译器分发边界 |
| 少量不依赖 AOT 语义的整图 rewrite | 可做但谨慎 | 必须维持 FX 合法性、meta 和 backend 契约 |
| 修复 graph break | 通常应改捕获支持 | 图外代码根本不会出现在 backend 输入中 |
| ATen 精确 pattern/fusion | 不适合 | 应放 Joint/Post-Grad |
| kernel 融合、layout、stream | 不适合 | 应放 Lowering/Scheduler/Codegen |

---

## 4. 关键 API

| API | 作用 | 注意事项 |
|---|---|---|
| `torch.compile(..., backend=callable)` | 直接传 backend，最少注册状态 | callable 接收 `gm, example_inputs`，必须返回 callable |
| `torch._dynamo.register_backend` | 给 backend 注册字符串短名 | 模块必须先 import；重复名字会报错 |
| `torch._dynamo.lookup_backend` | 把字符串或 callable 解析为 callable | Dynamo 内部在进入编译前调用 |
| `torch._dynamo.list_backends` | 列出可作为字符串使用的 backend | 默认排除 debug/experimental tag |
| `torch._dynamo.explain` | 解释图、graph break、guard 等捕获结果 | 用于诊断，不是注册 Pass |
| `torch._inductor.compile_fx` | 在自定义 backend 做完审计/改写后继续走 Inductor | 会接管并可能修改输入 `GraphModule` |

`register_backend()` 的类型契约在固定基线中是 `Callable[[fx.GraphModule, list[Tensor]], CompiledFn]`。真实 `example_inputs` 可能包含 FakeTensor；不要在 backend 中假设可以执行真实设备计算。

---

## 5. 如何注册并加入 `torch.compile`

### 5.1 直接传 callable（首选）

```python
import torch
from torch._inductor.compile_fx import compile_fx

def audit_then_inductor(gm: torch.fx.GraphModule, example_inputs):
    gm.graph.lint()
    call_targets = [
        str(node.target) for node in gm.graph.nodes if node.op == "call_function"
    ]
    print("captured targets:", call_targets)
    return compile_fx(gm, example_inputs)

compiled = torch.compile(model, backend=audit_then_inductor)
actual = compiled(*inputs)
```

### 5.2 注册字符串名字

```python
import torch
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx

@register_backend(name="audit_inductor")
def audit_inductor(gm: torch.fx.GraphModule, example_inputs):
    gm.graph.lint()
    return compile_fx(gm, example_inputs)

# 包含装饰器的模块必须已经 import。
compiled = torch.compile(model, backend="audit_inductor")
```

如果只是实验，不需要注册名字；直接传 callable 可避免 import 顺序和全局注册表冲突。

---

## 6. 真正改图时的最小规则

1. 先确认目标节点位于捕获图内；graph break 外的代码无法被 backend rewrite。
2. 使用 FX API 插入/替换节点，保持拓扑合法；改图后至少 `gm.graph.lint()`、`gm.recompile()`。
3. 保留/重算下游依赖的 `node.meta`，不要把真实 Tensor 塞进 FakeTensor 图。
4. 不要改变 placeholder/output 的 pytree 契约，否则 AOTAutograd 接口会错位。
5. 不要吞掉 Python 副作用、alias/mutation 或异常语义。
6. 若 rewrite 其实依赖 ATen/functionality，迁移到 Joint/Post-Grad custom pass。

---

## 7. 验证与排错

- 用 `torch._dynamo.explain` 确认图数量、break reason 和 guards。
- 用 `fullgraph=True` 把意外 graph break 变成明确错误，但不要把它当生产环境的自动修复。
- 对 backend 开/关做 eager/compile 对比；训练场景同时验证梯度。
- 改变 guard 或图结构时覆盖多组 shape/dtype/Python 参数，观察是否错误复用缓存或过度重编译。
- backend 内抛出的异常会表现为 backend compiler failure；先用只审计、不改图的版本隔离捕获问题和改写问题。

---

## 8. 何时离开 Dynamo

| 你的条件首次出现于 | 去哪里 |
|---|---|
| 高层 `F.linear`/module 结构 | [[pre_grad_passes_guide]] |
| functionalized 前反向联合 ATen 图 | [[joint_graph_passes_guide]] |
| 切分后的精确 ATen 图 | [[post_grad_passes_guide]] |
| ATen → IR、layout/realization | [[lowering_analysis]] |
| buffer 依赖、融合组、stream | [[scheduler_analysis]] |
| target kernel/wrapper ABI | [[codegen_extension_guide]] |

## Related Pages

- [[instruction_translator_and_bytecode_state_machine_analysis]] — 字节码符号执行深挖
- [[guards_cache_lookup_and_recompilation_analysis]] — guards 与重编译深挖
- [[output_graph_side_effects_and_graph_emission_analysis]] — OutputGraph 深挖
- [[dynamo_quickstart]] — `explain`、graph break 和重编译快速排查
- [[fx_pass_optimization_methodology]] — 八阶段 Pass 放置总方法论
- [[inductor_compiler_pipeline_analysis]] — Dynamo 到 Codegen 的端到端调用链
