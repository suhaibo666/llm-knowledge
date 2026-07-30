> 层次:quick start(用)
> 核验基准:PyTorch upstream `E:\97-codes\pytorch\pytorch`(v2.13.0a0, commit 9922478)
> 最后更新:2026-06-15

# Eager 反向自动微分引擎 — 实战上手

本页面向已经会用 `loss.backward()`、但想搞清「叶子/非叶子、`grad_fn`、`retain_graph`/`create_graph`、自定义 `Function`、grad 模式、NaN/inplace 排查」的工程师。每个结论都对照上游源码标注 `相对路径:行号`(相对 `E:\97-codes\pytorch\pytorch` 根)。

> 概念全景(Node/Edge/AutogradMeta、与编译期 AOTAutograd 的区别)见 [[index]];源码级深析见 [[10_autograd_engine_analysis]]。

## 0. 30 秒最小可用路径

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)  # 叶子,要梯度
y = (x ** 2).sum()                                 # y 是非叶子,带 grad_fn
y.backward()                                       # 反向:把梯度灌进叶子的 .grad
print(x.grad)        # tensor([4., 6.])  == d(sum x^2)/dx = 2x
print(x.is_leaf, y.is_leaf)        # True False
print(y.grad_fn)                   # <SumBackward0 object ...>
```

要点:**只有叶子张量(`is_leaf=True`)且 `requires_grad=True`** 才会在 `.backward()` 后拿到 `.grad`;中间张量默认不保存 `.grad`(见 §1)。

---

## 1. requires_grad / 叶子 vs 非叶子 / grad_fn

### 谁是叶子?谁带 grad_fn?

| 张量来源 | `is_leaf` | `requires_grad` | `grad_fn` | 反向后有 `.grad` 吗 |
|---|---|---|---|---|
| 用户直接创建 + `requires_grad=True` | True | True | None | 是 |
| 用户创建(默认) | True | False | None | 不参与反向 |
| 任何 op 的输出(在 grad 开启下,输入需梯度) | False | True | `XxxBackward` | 否(除非 `retain_grad()`) |
| `.detach()` 的结果 | True | False | None | 切断历史 |

源码语义(C++ `AutogradMeta`,`torch/csrc/autograd/variable.h:225`):
- `requires_grad_`(`variable.h:263`)——「仅对叶子有意义」;
- `retains_grad_`(`variable.h:266`)——「仅对非叶子有意义」,即下文 `retain_grad()`;
- `grad_fn_`(`variable.h:229`,强引用)对内部张量给出产生它的反向节点;`grad_accumulator_`(`variable.h:230`,弱引用)对叶子给出 `AccumulateGrad` 汇点。
- `requires_grad()` 的判定就是 `requires_grad_ || grad_fn_`(`variable.h:301`):**叶子看自己的标志位,非叶子看是否挂着 grad_fn**。

### 把这些规则用起来

```python
import torch

w = torch.randn(3, requires_grad=True)   # 叶子
b = torch.randn(3)                        # 叶子, requires_grad=False
z = w + b                                 # 非叶子: requires_grad=True, 有 grad_fn

print(w.is_leaf, w.grad_fn)   # True  None
print(z.is_leaf, z.grad_fn)   # False <AddBackward0 ...>

# 想拿中间张量的梯度: 必须显式 retain_grad()
z.retain_grad()
z.sum().backward()
print(z.grad)                 # tensor([1., 1., 1.])  (否则为 None 并告警)

# 原地把叶子设为需要梯度
p = torch.randn(2, 2)
p.requires_grad_(True)        # 等价于创建时传 requires_grad=True
```

> 常见坑:对一个**已经 requires_grad 的非叶子**做原地 `requires_grad_(False)` 会报错;切断历史请用 `t.detach()`(返回新叶子,共享存储)或在 `no_grad` 下计算(§5)。

---

## 2. .backward() 与 .grad;.grad() 的区别

两条反向入口,底层都落到 C++ `Engine::execute(root_edges, inputs, keep_graph, create_graph, accumulate_grad, outputs)`(`torch/csrc/autograd/engine.cpp:1294`):

| API | 梯度去向 | `accumulate_grad` | 典型用途 |
|---|---|---|---|
| `tensor.backward()` / `torch.autograd.backward(...)` | 累加进各叶子的 `.grad` | True | 训练:配 optimizer |
| `torch.autograd.grad(outputs, inputs, ...)` | 直接 **return** 给调用者,不写 `.grad` | False | 取某几个输入的梯度、二阶导、可视化 |

```python
import torch
x = torch.tensor(3.0, requires_grad=True)
y = x ** 3

# 写法 A: 灌进 .grad(会累加!)
y.backward()
print(x.grad)        # 27.  (= 3x^2)

# 注意累加语义: 不清零会叠加
y2 = x ** 3
y2.backward()
print(x.grad)        # 54.  (27 + 27)  → 训练循环里每步前要 optimizer.zero_grad()

# 写法 B: 直接返回,不碰 .grad
x2 = torch.tensor(3.0, requires_grad=True)
(gx,) = torch.autograd.grad(x2 ** 3, x2)
print(gx, x2.grad)   # 27.  None
```

- **非标量必须给 `grad_outputs`**:`v.backward(gradient=...)` 或 `torch.autograd.grad(y, x, grad_outputs=...)`,语义是向量-雅可比积 `vᵀ·J`。标量(如 `loss`)默认 `gradient=1`。
- 累加发生在叶子的 `AccumulateGrad` 节点(deepdive 的 Layout 契约);这也是「训练循环每步 `zero_grad`」的根因。

---

## 3. 查看 grad_fn / next_functions:手走反向图

`grad_fn.next_functions` 是 Python 侧暴露的反向 DAG 边,直接对应 C++ `Node::next_edges()`(`torch/csrc/autograd/node.h:317`)。每个元素是 `(下游 Node, input_nr)`,`input_nr` 即 C++ `Edge::input_nr`(`torch/csrc/autograd/edge.h:14`)。

```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x * y          # MulBackward0

print(z.grad_fn)                       # <MulBackward0 ...>
for fn, input_nr in z.grad_fn.next_functions:
    print(fn, "| input_nr =", input_nr)
# <AccumulateGrad ...> | input_nr = 0     ← 通向叶子 x
# <AccumulateGrad ...> | input_nr = 0     ← 通向叶子 y
```

- 叶子那一侧的 `next_functions` 指向 `AccumulateGrad`(把梯度写进 `x.grad`);
- 内部张量那一侧会继续指向上一层的 `XxxBackward`,顺着 `next_functions` 递归即可打印整张反向图;
- `(None, 0)` 表示该输入不需要梯度(无效边占位)。

> `MulBackward0`、`AddBackward0` 这些类名由元类在定义自定义 `Function` 时自动生成同款 `XxxBackward`(见 §4 与 `FunctionMeta`,`torch/autograd/function.py:346`)。

---

## 4. retain_graph 与 create_graph(二阶导)

### 默认:反向一次就释放图

`.backward()` / `.grad()` 跑完会释放 SavedVariable 缓冲(对应 `Engine::execute` 的 `keep_graph` 参数,`engine.cpp:1294`)。**再 backward 第二次**会触发 `ERR_BACKWARD_TWICE`(检查点 `torch/csrc/autograd/custom_function.cpp:555`;文案定义在 `torch/csrc/autograd/saved_variable.cpp:291`):

> Trying to backward through the graph a second time ... Specify `retain_graph=True` if you need to backward through the graph a second time ...

```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)   # 保留图,可再来一次
y.backward()                    # 第二次(默认会释放);若两次都不留则第二次报错
print(x.grad)                   # 8.  (4 + 4 累加)
```

### create_graph:让反向过程本身可微 → 二阶导

`create_graph=True` 让反向算子也建反向边,于是可对梯度再求导。它**隐含** `retain_graph=True`。

```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3

# 一阶导,且保留计算图供二阶导用
(g1,) = torch.autograd.grad(y, x, create_graph=True)
print(g1)            # 12.  (= 3x^2)

# 对一阶导再求导 → 二阶导
(g2,) = torch.autograd.grad(g1, x)
print(g2)            # 12.  (= 6x)
```

> 提醒:`tensor.backward(create_graph=True)` 会在「参数 ↔ 其 .grad」之间形成引用环导致泄漏,上游为此打了 `TORCH_WARN_ONCE`(`engine.cpp:1306-1312`)。**算高阶导优先用 `torch.autograd.grad`**;若非用 `backward` 不可,用后把 `.grad` 置 `None` 断环。

---

## 5. 写自定义 torch.autograd.Function

当某算子没有 autograd 支持、或想手写更省/更稳的反向公式时,继承 `torch.autograd.Function`(`torch/autograd/function.py:514`),实现两个 `@staticmethod`:`forward`(`function.py:369`)与 `backward`(`function.py:433`)。**永远用 `MyFn.apply(...)` 调用,不要直接调 `forward`**。

```python
import torch
from torch.autograd import Function

class Square(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)      # 存反向要用的张量
        return x * x

    @staticmethod
    def backward(ctx, grad_out):      # 收到 d(loss)/d(out)
        (x,) = ctx.saved_tensors
        return grad_out * 2 * x       # 返回 d(loss)/d(x),个数 == forward 输入个数

x = torch.tensor([2.0, 3.0], requires_grad=True, dtype=torch.double)
y = Square.apply(x).sum()
y.backward()
print(x.grad)                          # tensor([4., 6.])

# 强烈建议用 gradcheck 数值校验反向公式(double 精度)
from torch.autograd import gradcheck
print(gradcheck(Square.apply, (x,)))   # True
```

### ctx 的关键契约(全部在 `FunctionCtx`)

| 方法 | 在哪调 | 作用 | 锚点 |
|---|---|---|---|
| `ctx.save_for_backward(*tensors)` | forward/setup_context | 安全保存反向所需张量(走版本检测,见 §6) | `function.py:40` |
| `ctx.mark_dirty(*tensors)` | forward | 声明被**原地改写**的输入,保证 autograd 检查正确 | `function.py:157` |
| `ctx.mark_non_differentiable(*tensors)` | forward | 声明某些输出不可导(如排序索引),反向跳过 | `function.py:203` |
| `ctx.set_materialize_grads(False)` | forward | 关闭「把 None 梯度物化成零张量」,自己处理 None | `function.py:235` |
| `ctx.needs_input_grad` | backward 读 | 布尔元组,跳过不需要梯度的输入以省算力 | `function.py:449-453` |
| `ctx.saved_tensors` | backward 读 | 取回 `save_for_backward` 的张量 | — |

- 非张量数据(如标量超参)直接挂在 `ctx` 上:`ctx.alpha = alpha`,反向 `alpha = ctx.alpha`。
- `backward` 必须**为 forward 的每个输入返回一项**;不可导/非张量输入返回 `None`。
- `backward` 与 `vjp` 是同一方法的别名(`function.py:462`),二选一实现;前向模式 AD 实现 `jvp`(`function.py:491`)。
- 反向时引擎实际调用的是元类自动生成的 `SquareBackward`(`BackwardCFunction`,`function.py:297`;其 `apply` 在 `function.py:313`,由 `FunctionMeta.__init__` 生成,`function.py:355`)。
- **不支持二阶导的反向**应加 `@torch.autograd.function.once_differentiable` 装饰 `backward`,这样误用 `create_graph=True` 时会明确报错。

> C++ 侧等价物:`torch::autograd::Function<T>`(`torch/csrc/autograd/custom_function.h:98`),用 `AutogradContext`(`custom_function.h:122`)的 `save_for_backward`(`custom_function.h:137`),契约与 Python 一一对应。

---

## 6. grad 模式:no_grad / enable_grad / inference_mode

都是 thread-local 开关,控制前向是否建反向边。

| API | 锚点 | 行为 | 何时用 |
|---|---|---|---|
| `torch.no_grad()` | `grad_mode.py:22` | 关闭建图:输出 `requires_grad=False`,省显存 | 推理、手改参数 |
| `torch.enable_grad()` | `grad_mode.py:89` | 在 `no_grad` 区域里**局部重新开启** | 嵌套场景 |
| `torch.set_grad_enabled(flag)` | `grad_mode.py:144` | 按布尔条件开关,也可当函数/上下文 | `set_grad_enabled(is_train)` |
| `torch.inference_mode()` | `grad_mode.py:213` | 比 no_grad 更激进:**额外关掉 view 跟踪与 version bump** | 纯推理热路径 |

```python
import torch
x = torch.tensor([1.0], requires_grad=True)

with torch.no_grad():
    y = x * 2
print(y.requires_grad)        # False

with torch.no_grad():
    with torch.enable_grad():
        z = x * 2
print(z.requires_grad)        # True(局部重新开启)

# 按条件开关(训练/评估常用)
is_train = False
with torch.set_grad_enabled(is_train):
    out = x * 3
print(out.requires_grad)      # False

# 推理最省: 但产物不能再进 autograd
with torch.inference_mode():
    r = x * 2
# r._version 会报 "Inference tensors do not track version counter"
```

> `no_grad` vs `inference_mode` 的区别(源自 `grad_mode.py:213` docstring):后者关掉 view 追踪与版本计数,更快更省,但**在该模式下产生的张量不能再用于 autograd 记录的计算**——只适合确定不再反传的纯推理。两者均**不影响前向模式 AD**。三者都是 thread-local,不会影响别的线程。

---

## 7. 排查:detect_anomaly 抓 NaN / inplace 改写

`torch.autograd.detect_anomaly`(`torch/autograd/anomaly_mode.py:12`)做两件事:
1. 反向报错时,打印**创建该反向函数的前向调用栈**(否则只看到反向栈,定位困难);
2. `check_nan=True`(默认)时,任何产生 `nan` 的反向计算立即抛错,定位「NaN 是从哪个反向算子开始的」。

```python
import torch

with torch.autograd.detect_anomaly():        # 会先 warn: 仅调试用,变慢
    x = torch.tensor([1.0], requires_grad=True)
    y = x.log()                              # x→0 时反向出 inf/nan
    (x * 0).sum().backward()
# 触发时打印 "Traceback of forward call that caused the error: ..."
```

也可用函数式开关 `torch.autograd.set_detect_anomaly(True, check_nan=True)`(`anomaly_mode.py:97`),记得用完关掉(开着显著变慢)。

### 最常见的反向报错:原地操作改写了反向需要的张量

`save_for_backward` 存的张量带版本号;反向 `unpack` 时比对当前版本(`torch/csrc/autograd/saved_variable.cpp:167-186`),不一致即抛:

> one of the variables needed for gradient computation has been modified by an inplace operation ... is at version N; expected version M instead.

```python
import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x.exp()           # 反向需要 y 本身(exp 的导数是 exp)
y += 1                # 原地改写 y → 篡改了反向缓存
y.sum().backward()    # RuntimeError: ... modified by an inplace operation ...
```

排查动作:
1. 打开 `detect_anomaly()`,错误信息会附「Hint: enable anomaly detection ...」并指出是哪个前向 op 的输出被改;
2. 把出问题的原地操作改成 out-of-place(`y = y + 1` 而非 `y += 1`);
3. 自定义 `Function` 里若确实原地改了输入,必须 `ctx.mark_dirty(...)`(§5)。

---

## 8. 速查表

| 想做 | 怎么做 |
|---|---|
| 让叶子参与求导 | `t.requires_grad_(True)` 或创建时 `requires_grad=True` |
| 拿中间张量梯度 | `t.retain_grad()` 后再 `backward` |
| 只取部分输入梯度、不写 `.grad` | `torch.autograd.grad(out, inputs)` |
| 反向两次 | 第一次 `backward(retain_graph=True)` |
| 二阶导 | `grad(..., create_graph=True)` 再 `grad(...)` |
| 看反向图 | 递归遍历 `t.grad_fn.next_functions` |
| 推理省显存 | `with torch.no_grad():` / 热路径用 `inference_mode()` |
| 校验自定义反向 | `torch.autograd.gradcheck(fn, inputs)`(double) |
| 抓 NaN / inplace | `with torch.autograd.detect_anomaly():` |
| 训练每步 | `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`(梯度是累加的) |

---

## Related Pages

- [[index]] — 本模块概述:Node/Edge/AutogradMeta 全景与 DAG 三要素
- [[10_autograd_engine_analysis]] — 源码级深析:建图、SavedVariable、GraphTask、多线程 Engine、AccumulateGrad、前向模式 AD
- [[02_compile_stack/02_aot_autograd/index]] · [[aotautograd_joint_forward_backward_graphs_analysis]] — 编译期 AOT 捕获前/反向联合图(对比:eager 逐 op 动态长磁带,无预捕获)
- [[01_eager_runtime/02_dispatcher_and_device/index]] — 前向算子经 dispatcher 的 `VariableType` 层建反向边
- [[01_eager_runtime/01_tensor_and_storage/index]] — Tensor/AutogradMeta/版本计数的存储底座
