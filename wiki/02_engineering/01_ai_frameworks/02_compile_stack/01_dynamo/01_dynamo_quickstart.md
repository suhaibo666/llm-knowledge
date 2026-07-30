# Dynamo Quick Start — 实用上手

> 层次:quick start(浅、实用)
> 核验基准:PyTorch 上游(`E:\97-codes\pytorch\pytorch`,所有 API/参数/日志键已对照源码)
> 最后更新:2026-06-13

---

## 1. Dynamo 是什么

Dynamo 是 `torch.compile` 的**图捕获前端**:它钩入 CPython 的 PEP 523 帧评估 API,
逐条符号执行 Python 字节码,把 PyTorch 算子序列抽取成 FX 图,再交给后端(默认 Inductor)编译。
凡是它无法捕获的 Python 逻辑,会被切成多张图(graph break),break 之间回落到普通 Python 执行。

---

## 2. 看捕获结果:`explain`

```python
import torch
import torch._dynamo

def fn(x):
    x = x + 1
    if x.sum() > 0:          # 数据依赖控制流 → 会产生 graph break
        return x * 2
    return x - 2

out = torch._dynamo.explain(fn)(torch.randn(4))
print(out)
```

真实接口(`torch/_dynamo/eval_frame.py:1812`):**推荐写法是 `explain(f)(*args, **kwargs)`**。
旧式 `explain(f, *args, **kwargs)` 仍可用但会抛 `FutureWarning`(同文件 1873-1881)。

返回值是 `ExplainOutput`(`torch/_dynamo/backends/debugging.py:598`),关键字段:

| 字段 | 含义 |
|------|------|
| `graph_count` | 捕获到的 FX 图数量 |
| `graph_break_count` | graph break 次数(= `graph_count - 1`) |
| `op_count` | 所有图里 `call_function` 节点总数 |
| `break_reasons` | 每个 break 的原因与用户栈 |
| `graphs` | 捕获到的 `torch.fx.GraphModule` 列表 |
| `ops_per_graph` / `out_guards` / `compile_times` | 每图算子、导出的 guards、编译耗时 |

`print(out)` 的 `__str__` 会打印 `Graph Count / Graph Break Count / Op Count / Break Reasons / ...`
(同文件 613-641)。**理想是 `Graph Break Count: 0`** —— 一张图跑完。

---

## 3. Graph break:定位与消除

### 常见原因
- **数据依赖的控制流**:`if x.sum() > 0`、依赖张量具体值的 `while`。
- **不支持的调用**:Dynamo 不认识的第三方/C 扩展函数、某些内建操作。
- **打印/IO/全局副作用**:在 break 处切图后回放。

### 如何定位
```bash
TORCH_LOGS="graph_breaks" python script.py
```
`graph_breaks` 是已注册的 visible artifact(`torch/_logging/_registrations.py:158-162`):
"Prints whenever Dynamo decides that it needs to graph break"。

### 如何强制无 break
```python
torch.compile(fn, fullgraph=True)(x)   # 出现 graph break 直接报错
```
`fullgraph` 是 `torch.compile` 的真实参数,默认 `False`(`torch/__init__.py:2577`、文档 2638-2642)。
`True` 时要求整段函数能进单图,否则抛错;它内部对应 Dynamo 的 `nopython=True`(`torch/__init__.py:2832`)。

---

## 4. Guards 与重编译

每张编译图都带一组 **guards**(运行期必须成立的前置条件,如张量 dtype/shape、类型、id)。
guard 失败 → 缓存不命中 → **重编译**。

```bash
TORCH_LOGS="guards,recompiles" python script.py
```
- `guards`:visible artifact,"prints the guards for every compiled Dynamo frame"(`_registrations.py:64-68`)。
- `recompiles`:visible artifact,"Prints the reason why we recompiled a graph. Very, very useful."(`_registrations.py:145-149`)。
- 想看**每条失败的 guard 检查**:`recompiles_verbose`(同文件 150-157,默认关)。

### recompile_limit(重编译上限)
- `torch._dynamo.config.recompile_limit = 8`(`torch/_dynamo/config.py:121`):同一函数的重编译上限,超出后回落到 eager。
- `torch._dynamo.config.accumulated_recompile_limit = 256`(同文件 124):全局累计上限。
- `torch._dynamo.config.fail_on_recompile_limit_hit = False`(同文件 135):置 `True` 则超限直接报错而非回落。
- 旧名 `cache_size_limit` / `accumulated_cache_size_limit` 现为别名(同文件 137-140)。
- `torch.compile(..., recompile_limit=N)` 可针对单个 compile 覆盖全局值(`torch/__init__.py:2708-2711`)。

**为什么重编译**:输入 shape/dtype 变化、Python 标量变化、被 ID_MATCH 守卫的对象身份变化等,
都会让对应 guard 失败而触发新一轮捕获。

---

## 5. 常用逃生阀

| 操作 | API | 源码 |
|------|-----|------|
| 让某函数完全不被编译(可递归) | `torch._dynamo.disable(fn, recursive=True, *, reason=None)` | `decorators.py:83` |
| 同上(公开别名) | `torch.compiler.disable(fn, recursive=True, *, reason=None)` | `compiler/__init__.py:250` |
| 把函数当不透明算子直接写进图(前端不 trace,后端仍 trace) | `torch.compiler.allow_in_graph(fn)` | `compiler/__init__.py:73` |
| 同上(内部实现) | `torch._dynamo.allow_in_graph(fn)` | `decorators.py:192` |
| 主动制造一次 graph break | `torch._dynamo.graph_break()` | 导出于 `_dynamo/__init__.py:96` |
| 清空所有编译缓存、恢复初始状态 | `torch._dynamo.reset()` | `_dynamo/__init__.py:138` |
| 同上(公开别名) | `torch.compiler.reset()` | `compiler/__init__.py:60` |
| 判断当前是否在编译/trace 中 | `torch.compiler.is_compiling() -> bool` | `compiler/__init__.py:477` |

```python
@torch._dynamo.disable                 # 整个函数跳过 Dynamo
def uses_unsupported_stuff(x): ...

torch.compiler.allow_in_graph(my_c_func)   # 已知后端能跑、只是前端 trace 不了

torch._dynamo.reset()                  # 测试/基准前清状态,等同重开进程(不删磁盘缓存)
```

> 注意 `allow_in_graph` 是 footgun:它跳过前端安全检查,误用会导致难查的静默错误
> (官方告警见 `compiler/__init__.py:84-91`)。能用 custom op 就别用它。

### 调试全开
```bash
TORCH_LOGS="+dynamo" python script.py   # dynamo 全量调试日志
TORCH_LOGS="bytecode" python script.py  # 看字节码改写(默认关,排查 codegen 用)
```
`dynamo` 是注册的 log,映射到 `torch._dynamo`(`_registrations.py:25`);`bytecode` 是 off-by-default artifact(同文件 70-74)。

---

## 6. 深入导航

- [[02_compile_stack/01_dynamo/index]] — deep dive 十篇:API 生命周期、backend 五控制面、帧评估/code cache、字节码符号执行、VariableTracker、OutputGraph、Guard/重编译、graph break、动态形状、backend 契约。
- [[02_compile_stack/04_inductor/index]] — 端到端流水线 Dynamo → AOTAutograd → Inductor。
- [[02_compile_stack/02_aot_autograd/index]] — 下一阶段:前/反向分解。

---

## Related Pages

- [[02_compile_stack/01_dynamo/index]]
- [[02_compile_stack/04_inductor/index]]
- [[02_compile_stack/02_aot_autograd/index]]
- [[01_ai_frameworks/index]]
