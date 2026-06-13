# Unbacked SymInt 深度分析：数据相关 Shape 的处理机制

> 基于 PyTorch 主分支源码与官方文档分析
> 最后更新: 2026-05-22

---

## 1. Backed vs Unbacked：根本区别

PyTorch dynamic shape 系统中存在两类符号整数：

```
Backed symbol (s0):
  来源:  函数入口处读取输入 tensor 的 shape
  时机:  函数调用前就已知
  hint:  有（编译时见过的具体值，如 hint=3）
  约束:  → 翻译为 Guard，在 kernel 执行前检查

Unbacked symbol (u0):
  来源:  图内部某个 op 的输出 shape，由张量数据值决定
  时机:  必须执行该 op 之后才能知道
  hint:  无（编译期完全不知道）
  约束:  → 翻译为 deferred runtime assert，在 op 执行后检查
```

### 1.1 产生 Unbacked Symbol 的 Op 类型

| Op | 原因 |
|----|------|
| `torch.nonzero(x)` | 非零元素个数 = f(x 的数据值) |
| `tensor.item()` | 把标量 tensor 转换为 Python int/float |
| `torch.where(cond)` | 满足条件的元素个数 |
| `torch.unique(x)` | 唯一值个数未知 |
| `masked_select(x, mask)` | 被选元素数量 = f(mask 的数据) |
| `torch.bincount(x)` | 输出 size 取决于 x 的最大值 |
| `torch.nonzero_static(x, size=n)` | 截断为固定大小（特殊处理） |

### 1.2 命名惯例

- Backed symbols：`s0, s1, s2, ...`（ShapeEnv 内部）
- Unbacked symbols：`u0, u1, u2, ...`（ShapeEnv 内部）

用户可通过 `torch._dynamo.decorators.mark_unbacked(tensor, dim)` 手动标记某维度为 unbacked。

---

## 2. 为什么 Guard 机制对 Unbacked 无效

Guard 的执行时机是**在 kernel 调用之前**，根据输入 tensor 的 shape 做检查：

```python
# Guard 检查（执行前）：
def check_guards(args):
    assert 2 <= args[0].size(0)     # ✓ 输入 tensor 在调用前就存在

# Unbacked symbol 无法这样做：
def check_guards(args):
    # u0 = nonzero(args[0]) 的输出行数
    # 在执行 nonzero 之前，u0 根本不存在！
    assert u0 > 0                   # ✗ u0 从哪来？
```

Guard 负责决定**缓存是否命中**（Dynamo 层），unbacked symbol 的正确性验证必须在 op 执行后（Inductor codegen 层）插入断言。这是两个完全独立的机制。

---

## 3. ShapeEnv 内部：两类约束的存储方式

```python
class ShapeEnv:
    # Backed symbol 的约束 → 翻译为 Guard
    var_to_range: dict[Symbol, ValueRanges]   # s0 ∈ [2, ∞)
    guards: list[ShapeGuard]                   # → "2 <= arg0.size(0)"

    # Unbacked symbol 的约束 → 推迟到运行时
    deferred_runtime_asserts: dict[Symbol, list[RuntimeAssert]]
    # 例如: {u0: [RuntimeAssert(u0 > 0), RuntimeAssert(u0 < 1000)]}
```

核心方法 `guard_or_defer_runtime_assert(expr)` 根据 expr 涉及的符号类型做分流：

```
guard_or_defer_runtime_assert(expr):
  if 所有涉及的 symbol 都是 backed:
      → 直接加 guard（编译期检查）
  elif 涉及任意 unbacked symbol:
      → 加入 deferred_runtime_asserts
      → codegen 时在 wrapper 中生成运行时 assert 语句
      → 编译期假设该条件为 True（不触发报错，允许后续推导）
```

**"编译期假设为 True"** 的含义：ShapeEnv 在做符号推导时会把 `torch._check(u0 > 0)` 里的条件当作已知事实，用于化简后续的符号表达式（如推导出 `u0 + 1 > 1`）。

---

## 4. Inductor Codegen：Unbacked Symbol 在 Wrapper 中的体现

以 `nonzero` 为例，生成的 wrapper 代码结构：

```python
def call(args):
    arg0_1 = args[0]   # 输入 tensor

    # ① 执行产生 u0 的 op
    buf0 = torch.ops.aten.nonzero.default(arg0_1)

    # ② 从输出 tensor 反向解析 unbacked symbol
    #    (compute_unbacked_bindings 的结果, 由 pytree.KeyPath 确定位置)
    u0 = buf0.size(0)   # 现在才知道 u0 的值

    # ③ 插入 deferred runtime assertion（来自 torch._check(u0 > 0)）
    torch._check(u0 > 0)
    # 编译期: ShapeEnv 知道 u0 > 0 为 True
    # 运行时: 实际执行验证，违反则 RuntimeError

    # ④ 后续使用 u0 的计算（u0 在运行时是具体整数）
    buf1 = empty_strided_cuda((u0, 64), (64, 1), torch.float32)
```

对应源码：`codegen_unbacked_symbol_defs_for_outputs()` (`wrapper.py:3436`)，通过 `pytree.KeyPath` 知道"从哪个输出 tensor 的哪个 dim 读取 u0"。

---

## 5. torch._check() 的三种效果

`torch._check(expr)` 根据 expr 的形式产生不同效果：

### 效果 1：值域细化

```python
u0 = indices.size(0)            # u0 ∈ [0, ∞)（默认 size-like 范围）
torch._check(u0 > 0)            # → 细化: u0 ∈ [1, ∞)
torch._check(u0 < 1024)         # → 细化: u0 ∈ [1, 1023]

# 编译器现在知道 u0 有界，可安全做 32-bit indexing 等决策
```

### 效果 2：符号替换（消灭 unbacked）

```python
u0 = indices.size(0)            # unbacked
torch._check(u0 == s0)          # → ShapeEnv: replacements[u0] = s0

# 之后所有 u0 被替换为 backed symbol s0
# u0 从 deferred_runtime_asserts 移除，退化为普通 guard 处理
```

这是消灭 unbacked symbol 的最强手段。若能证明 u0 等于某个 backed symbol，整个问题退化到 backed 路径。

### 效果 3：条件记忆（解决控制流问题）

```python
u0 = indices.size(0)
torch._check(u0 > 4)            # 告诉编译器"此条件恒为真"

if u0 > 4:                      # 编译器知道走 True 分支
    return x * 2                # 唯一路径
# else 分支变成死代码，从编译产物中消失
```

`torch._check` 是对 ShapeEnv 的"事实注入"，不是普通 Python 断言。编译器无条件相信它，运行时插入验证断言——违反时 RuntimeError，而非静默错误。

---

## 6. GuardOnDataDependentSymNode：控制流与 Unbacked 的冲突

### 触发条件

当 unbacked symbol 参与 Python `if` 判断时发生：

```python
@torch.compile
def f(x):
    indices = torch.nonzero(x)
    n = indices.size(0)          # u0

    if n > 4:           # ← 致命！Python 需要 bool(u0 > 4)
        return x * 2
    else:
        return x + 3
```

调用链：

```
Python: bool(n > 4)
  → SymInt.__bool__
    → ShapeEnv.evaluate_expr(u0 > 4)
      → u0 没有 hint → 无法判断 True/False
        → raise GuardOnDataDependentSymNode
```

### 为什么 Backed Symbol 可以做 if 判断

Backed symbol 有 hint（编译时见过的具体值）。ShapeEnv 用 hint 做决策（如 hint=7，判断 `7 > 4 = True`），同时生成 guard `u0 > 4`。下次运行时 guard 检查实际 u0 是否满足，不满足则重编译走另一分支。

Unbacked symbol 没有 hint → 无法判断哪个分支 → 报错。

### 修法

```python
# Fix: torch._check() 在 if 之前注入约束
torch._check(n > 4)   # 告诉编译器"这个条件恒为真"
if n > 4:             # 现在可以：编译器取 True 分支
    return x * 2
# else 变成死代码
```

---

## 7. 相关 API

| API | 用途 |
|-----|------|
| `torch._check(cond)` | 注入约束，编译期信任，运行时验证 |
| `torch._check_is_size(x)` | 声明 x 将用作 tensor size（≥ 0，启用 size-like 特殊处理） |
| `torch.fx.experimental.symbolic_shapes.constrain_range(x, min, max)` | 显式设置值域 `[min, max]` |
| `torch._dynamo.decorators.mark_unbacked(tensor, dim)` | 手动将某维度标记为 unbacked |
| `torch.fx.experimental.symbolic_shapes.statically_known_true(expr)` | 不加 guard 地测试一个条件是否静态已知为真 |
| `torch.fx.experimental.symbolic_shapes.guard_or_false(expr)` | 若可 guard 则加 guard，否则返回 False（不崩溃） |

---

## 8. 常见误用与修法

### 误用 1：Python 切片使用 unbacked symbol

```python
# 失败：Python 的 [:n] 触发 __index__ → bool 判断
n = x.nonzero().size(0)
return y[:n]

# 修法：用 narrow()，它接受 SymInt
return y.narrow(0, 0, n)
```

### 误用 2：两个独立 unbacked 可以统一

```python
u0 = a.nonzero().size(0)
u1 = b.nonzero().size(0)
# 如果 a 和 b 的非零数量总是相同
torch._check(u0 == u1)   # 符号替换 → u1 消灭，变成 u0
result = torch.randn(u0, u0)
```

### 误用 3：item() 结果直接参与控制流

```python
# 失败版本
size = x.item()
if size > 0:
    ...

# 正确版本
size = x.item()
torch._check_is_size(size)  # 声明 size 是合法 tensor size (≥ 0)
torch._check(size > 0)      # 注入具体约束
if size > 0:                # 现在可以：取 True 分支
    ...
```

### 误用 4：empty_strided 与 unbacked stride

```python
# 失败：stride 由 unbacked symbol 计算，触发 output stride 推导问题
u0 = x.sum().item()
return torch.empty_strided((u0, 64), (64, 1))

# 修法：用 empty()，让 Inductor 推导 stride
return torch.empty((u0, 64))
```

---

## 9. Backed vs Unbacked 全链路对比

```
               Backed (s0)                     Unbacked (u0)
               ──────────────────              ──────────────────
何时产生      │ 函数入口读取 input shape       │ 图内部 op 执行后
何时已知      │ kernel 调用前                  │ 产生该 u0 的 op 执行后
约束存储      │ ShapeEnv.var_to_range          │ deferred_runtime_asserts
约束翻译      │ Guard（compile-time）           │ 运行时 assert（runtime）
控制流        │ 有 hint → 可做 if 判断          │ 无 hint → GuardOnDataDependentSymNode
消灭方式      │ 静态化（具体值替换）             │ torch._check(u0 == backed_expr)
Wrapper 体现 │ assert_size_stride(arg, ...)   │ u0 = output.size(dim) [先读取]
             │ 在 wrapper 开头执行             │ torch._check(u0>0)    [后断言]
```

---

## Related Pages

- [[dynamic_shapes_full_analysis]] — Dynamic Shape 全链路，ShapeEnv 与 Backed Symbol 体系
- [[inductor_codegen_dynamic_shape_analysis]] — Inductor codegen 中的符号传递机制
- [[torch_compile_architecture]] — torch.compile 端到端流水线
- [[PyTorch_Dynamo_Technical_Analysis]] — Dynamo 帧捕获与 Guard 系统
