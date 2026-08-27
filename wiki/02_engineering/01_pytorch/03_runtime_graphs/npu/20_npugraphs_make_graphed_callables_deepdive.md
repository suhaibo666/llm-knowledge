---
title: "make_graphed_callables 深度解析"
---

# make_graphed_callables 深度解析

> 本文基于 `torch_npu` 的 `make_graphed_callables` 实现（`torch_npu/npu/graphs.py`），深入分析其完整实现逻辑、内存分配与复用机制，以及内存峰值的 debug 分析方法。

---

## 一、函数概览

`make_graphed_callables` 接受一个或多个 callable（`nn.Module` 或普通函数）以及对应的样本参数，返回**图化版本**的 callable。图化后的 callable 接口与原始 callable 完全一致，但底层通过 NPU Graph 的 `replay()` 执行，省去了逐个算子下发的 CPU 开销。

```python
def make_graphed_callables(
    callables,            # 单个或 tuple of callable
    sample_args,          # 对应的样本参数
    num_warmup_iters=3,   # 预热次数
    allow_unused_input=False,
    pool=None             # 共享内存池 token
)
```

### 核心思路

1. 用 `sample_args` 做**预热**，触发所有延迟初始化
2. **捕获前向图**：录制 `func(*args)` 的所有算子
3. **捕获反向图**：录制 `torch.autograd.grad(...)` 的所有算子
4. 将捕获好的图包装成 `torch.autograd.Function`，替换原始 callable

---

## 二、完整实现流程（六个阶段）

### 阶段 1：输入校验与展平（490–516 行）

首先统一输入格式：如果用户传入的是单个 callable，包装成 tuple。

```python
if not isinstance(callables, tuple):
    just_one_callable = True
    callables = (callables,)
    sample_args = (sample_args,)
```

然后对每个 callable 和对应的 `sample_args` 做校验和展平：

#### 1.1 Module 约束检查

- **禁止带 hook**：如果 `nn.Module` 已注册了 `_backward_hooks`、`_forward_hooks` 或 `_forward_pre_hooks`，直接报错。图捕获要求计算图固定，hook 会改变执行逻辑。（图化完成后再注册 hook 是允许的）
- **Buffer 不能参与梯度**：所有 `buffers()` 必须 `requires_grad=False`，只有 `parameters()` 可以可训练。

#### 1.2 参数展平

```python
flatten_arg = torch.utils._pytree.arg_tree_leaves(*args)
flatten_sample_args.append(tuple(flatten_arg))
```

`arg_tree_leaves` 将任意嵌套结构（tuple/list/dict）的参数展平为一维的叶子张量序列：

| 传入的 `args` | 展平后 |
|---------------|--------|
| `(x,)` | `(x,)` |
| `(x, y)` | `(x, y)` |
| `((x, y), z)` | `(x, y, z)` |
| `(x, {"a": a, "b": b})` | `(x, a, b)` |

展平的目的：图只认识「第 0 个输入、第 1 个输入、…」的一维张量列表，不关心嵌套结构。捕获时和回放时必须用同一套扁平、有序的张量列表。

#### 1.3 类型检查

展平后的参数必须全部是 `torch.Tensor`，不允许 int、None 等其他类型。

---

### 阶段 2：构建静态输入表面（518–528 行）

图捕获时，前向传播不仅依赖用户传入的参数，还依赖 Module 内部的 `weight`、`bias` 等参数。需要把它们拼接在一起，构成**完整的静态输入表面（static input surface）**。

```python
per_callable_len_user_args = [len(args) for args in flatten_sample_args]

per_callable_module_params = [
    tuple(c.parameters()) if isinstance(c, torch.nn.Module) else ()
    for c in callables
]

per_callable_static_input_surfaces = [
    flatten_sample_args[i] + per_callable_module_params[i]
    for i in range(len(callables))
]
```

**示例**：假设线性层 `Linear(3, 5)`，用户传入 `(x,)`：

```
static_input_surface = (x, weight, bias)
len_user_args = 1      ← 前 1 个是用户参数（每次调用会变）
                         后面的是 Module 参数（被优化器原地更新，地址不变）
```

`len_user_args` 在后续回放时用来区分：哪些输入需要 `copy_`（用户参数），哪些不需要（Module 参数）。

---

### 阶段 3：预热（535–562 行）

在**独立的临时 stream** 上，对每个 callable 执行 `num_warmup_iters`（默认 3）次前向 + 反向：

```python
torch_npu.npu.synchronize()
with torch_npu.npu.stream(torch_npu.npu.Stream()):
    for func, args, static_input_surface in zip(...):
        for _ in range(num_warmup_iters):
            # 前向
            outputs = torch.utils._pytree.tree_leaves(func(*args))
            # 反向
            outputs_grad = tuple(o for o in outputs if o.requires_grad)
            if len(outputs_grad) > 0:
                grad_inputs = torch.autograd.grad(
                    outputs=outputs_grad,
                    inputs=tuple(i for i in static_input_surface if i.requires_grad),
                    grad_outputs=tuple(torch.empty_like(o) for o in outputs if o.requires_grad),
                    only_inputs=True,
                    allow_unused=allow_unused_input,
                )
        # 释放中间变量
        for v in [outputs, outputs_grad, grad_inputs]:
            del v
torch_npu.npu.synchronize()
```

#### 为什么需要预热

如果不预热，图捕获时可能会录进这些一次性的初始化操作：

- 算子库的 benchmark（如 cuDNN/CNNL 选最优卷积算法）
- 内存分配器首次分配时的扩容
- JIT 编译内核
- 其他 lazy initialization

这些操作只需执行一次，录进图后每次 replay 都会白白执行，浪费时间甚至导致错误。预热把这些「脏活」提前做完，保证正式捕获录到的是**纯粹的计算逻辑**。

#### 关键设计

- **独立 stream**：预热产生的所有内核调用和内存分配不会污染默认 stream，也不会被后面的图捕获录进去
- **前向 + 反向都要预热**：确保反向传播路径上的算子也完成初始化
- **多次预热**：某些初始化（如 cuDNN benchmark）可能需要多轮才能稳定
- **`del` 中间变量**：预热产生的输出和梯度无实际用途，及时释放节省内存

---

### 阶段 4：前向图捕获（568–577 行）

按**正序** 0→N 遍历每个 callable：

```python
per_callable_static_outputs = []
per_callable_output_unflatten_spec = []
for func, args, fwd_graph in zip(callables, sample_args, fwd_graphs):
    with torch_npu.npu.graph(fwd_graph, pool=mempool):
        outputs = func(*args)

    flatten_outputs, spec = torch.utils._pytree.tree_flatten(outputs)
    per_callable_static_outputs.append(tuple(flatten_outputs))
    per_callable_output_unflatten_spec.append(spec)
```

`with torch_npu.npu.graph(fwd_graph, pool=mempool)` 内部：
1. `synchronize()` + `empty_cache()` → 清理环境
2. `capture_begin(pool=mempool)` → 分配器进入捕获模式
3. 执行 `func(*args)` → 所有算子被录进 `fwd_graph`
4. `capture_end()` → 录制结束

#### 捕获后的处理

`with` 退出后，`tree_flatten(outputs)` 将输出展平并同时提取结构信息：

| 变量 | 含义 | 后续用途 |
|------|------|---------|
| `flatten_outputs` | 展平后的叶子张量（**地址固定**） | 反向图捕获时作为 `autograd.grad` 的 outputs |
| `spec` | 输出的结构骨架（TreeSpec） | 回放时用 `tree_unflatten` 还原输出的嵌套格式 |

---

### 阶段 5：反向图捕获（579–623 行）

按**逆序** N→0 遍历每个 callable：

```python
for static_input_surface, static_outputs, bwd_graph, module_params in zip(
    reversed(per_callable_static_input_surfaces),
    reversed(per_callable_static_outputs),
    reversed(bwd_graphs),
    reversed(per_callable_module_params),
):
```

#### 为什么要逆序

所有图共享同一个 `mempool`，为避免回放时内存互相踩踏，捕获顺序必须和实际运行顺序一致：

```
捕获/运行顺序: fwd_0 → fwd_1 → ... → fwd_N → bwd_N → bwd_{N-1} → ... → bwd_0
```

前向已按 0→N 正序捕获，所以反向要 N→0 逆序。

#### 5.1 构造伪梯度输出

```python
static_grad_outputs = tuple(
    torch.empty_like(o) if o.requires_grad else None for o in static_outputs
)
```

为每个前向输出创建同形状的空张量，作为梯度的"固定内存槽位"。回放时真正的上游梯度会被 `copy_` 进这些张量。

#### 5.2 录制反向图

```python
outputs_grad = tuple(o for o in static_outputs if o.requires_grad)
if len(outputs_grad) > 0:
    with torch_npu.npu.graph(bwd_graph, pool=mempool):
        grad_inputs = torch.autograd.grad(
            outputs=outputs_grad,
            inputs=tuple(i for i in static_input_surface if i.requires_grad),
            grad_outputs=tuple(o for o in static_grad_outputs if o is not None),
            only_inputs=True,
            allow_unused=allow_unused_input,
        )
```

关键：`outputs` 参数是前向图捕获时产生的静态输出张量。PyTorch autograd 在前向执行时已记住了计算图，`torch.autograd.grad` 沿着该计算图做反向传播，同时 `with` 上下文将这些反向算子录进 `bwd_graph`。

#### 5.3 梯度补 None 对齐

```python
static_grad_inputs = []
grad_idx = 0
for arg in static_input_surface:
    if arg.requires_grad and grad_inputs is not None:
        static_grad_inputs.append(grad_inputs[grad_idx])
        grad_idx += 1
    else:
        static_grad_inputs.append(None)
```

`torch.autograd.grad` 只返回对 `requires_grad=True` 的输入的梯度。但后面 `Graphed.backward` 需要对每一个输入都返回一个值。所以按 `static_input_surface` 的完整顺序重新对齐，给不需要梯度的位置补 `None`：

```
static_input_surface:  (x,        weight,      bias)
requires_grad:         (False,    True,        True)
autograd.grad 返回:              (grad_w,     grad_b)

对齐后:                (None,     grad_w,      grad_b)
```

#### 5.4 翻转回正序

```python
per_callable_static_grad_outputs.reverse()
per_callable_static_grad_inputs.reverse()
```

因为循环是逆序 append 的，最后 `reverse()` 保证 `per_callable_*[i]` 都一致地指向第 i 个 callable 的数据。

---

### 阶段 6：组装返回（626–716 行）

#### 6.1 创建 Graphed autograd Function

```python
def make_graphed_autograd_function(...):
    class Graphed(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs):
            # 把新数据 copy_ 到静态输入的固定内存
            for i in range(len_user_args):
                if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                    static_input_surface[i].copy_(inputs[i])
            fwd_graph.replay()
            return tuple(o.detach() for o in static_outputs)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, *grads):
            # 把上游梯度 copy_ 到静态梯度输出的固定内存
            for g, grad in zip(static_grad_outputs, grads):
                if g is not None:
                    if g.data_ptr() != grad.data_ptr():
                        g.copy_(grad)
            bwd_graph.replay()
            return tuple(b.detach() if b is not None else b for b in static_grad_inputs)

    def functionalized(*user_args):
        flatten_user_args = torch.utils._pytree.arg_tree_leaves(*user_args)
        out = Graphed.apply(*(tuple(flatten_user_args) + module_params))
        return torch.utils._pytree.tree_unflatten(out, output_unflatten_spec)

    return functionalized
```

**forward 回放流程**：
1. 只对用户参数（前 `len_user_args` 个）执行 `copy_`，Module 参数由优化器原地更新，地址不变
2. `data_ptr()` 检查：如果数据恰好已在正确位置，跳过 copy 节省开销
3. `fwd_graph.replay()`：一次性提交所有录制好的算子
4. 从 `static_outputs` 读取结果并 `detach()`

**backward 回放流程** 同理：`copy_` 上游梯度 → `bwd_graph.replay()` → 返回 `static_grad_inputs`

**functionalized 包装**：展平用户参数 → 拼接 Module 参数 → `Graphed.apply(...)` → `tree_unflatten` 还原输出结构

#### 6.2 根据 callable 类型做不同包装

```python
for i, func in enumerate(callables):
    graphed = make_graphed_autograd_function(...)

    if isinstance(func, torch.nn.Module):
        def make_graphed_forward(func, graph_training_state, graphed, orig_fwd):
            def new_fwd(*user_args):
                if func.training == graph_training_state:
                    return graphed(*user_args)
                else:
                    return orig_fwd(*user_args)
            return new_fwd

        func.forward = make_graphed_forward(func, func.training, graphed, func.forward)
        ret.append(func)
    else:
        ret.append(graphed)
```

- **Module** → 替换 `forward` 方法。如果当前 `training` 状态与捕获时一致，走图回放；否则降级为原始执行（因为 BatchNorm、Dropout 等在 train/eval 下行为不同，图录制的内容可能不适用）。返回的是原始 Module 对象本身。
- **普通函数** → 直接返回 `functionalized` 函数。

---

## 三、内存分配与复用机制

### 3.1 NPU Caching Allocator 两层结构

```
┌──────────────────────────────────────────────┐
│           应用层 (PyTorch 张量)                │
│  tensor = torch.randn(1024, device="npu")    │
└──────────────────┬───────────────────────────┘
                   │ malloc / free
┌──────────────────▼───────────────────────────┐
│         NPU Caching Allocator                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Segment 0│  │ Segment 1│  │ Segment 2│   │  ← 大块 (npuMalloc)
│  │┌────────┐│  │┌────────┐│  │┌────────┐│   │
│  ││Block A ││  ││Block D ││  ││Block G ││   │  ← 小块切分
│  │├────────┤│  │├────────┤│  │├────────┤│   │
│  ││Block B ││  ││Block E ││  ││Block H ││   │
│  │├────────┤│  │├────────┤│  │└────────┘│   │
│  ││Block C ││  ││Block F ││  │          │   │
│  │└────────┘│  │└────────┘│  │          │   │
│  └──────────┘  └──────────┘  └──────────┘   │
└──────────────────┬───────────────────────────┘
                   │ npuMalloc / npuFree
┌──────────────────▼───────────────────────────┐
│           设备物理内存 (HBM)                    │
└──────────────────────────────────────────────┘
```

- **Segment**：Caching Allocator 通过 `npuMalloc` 向设备申请的大块内存
- **Block**：Segment 内部切分出的小块，分配给单个张量
- Block 状态：`active_allocated`（使用中）、`active_pending_free`（等待释放）、`inactive`（空闲可复用）

### 3.2 图专用内存池（Private Pool）

```python
mempool = graph_pool_handle()  # 创建私有内存池
```

所有图（所有 callable 的 fwd_graph 和 bwd_graph）共享同一个 `mempool`。图内的所有分配都在这个**私有池**中进行，与普通 PyTorch 张量分配器**隔离**。

| | 普通内存池 | 图专用内存池 |
|---|---|---|
| 谁管理 | Caching Allocator 默认池 | `pool=` 参数指定的私有池 |
| 分配触发 | 每次 `torch.empty/randn` 等 | 图捕获期间的所有中间分配 |
| 释放行为 | tensor 引用计数归零时释放回池 | 图存在期间不会真正释放 |
| 跨图共享 | 所有非图操作共享 | 同一 pool token 的多个图共享 |

### 3.3 图捕获时的分配器行为

执行 `capture_begin` 后，分配器进入捕获模式：

1. **分配被录制**：每次 `malloc` 记录到图中（地址、大小、时序）
2. **释放被延迟**：`free` 不真正归还，只记录。同一 mempool 的后续图捕获或同图的后续 replay 中，这些内存才能被复用
3. **地址被固化**：捕获结束后，所有分配到的地址在后续 `replay()` 中保持不变

### 3.4 内存复用详解

#### 同一图内的复用

```
t1 = op1(x)       # 分配 t1 @addr_100
t2 = op2(t1)      # 分配 t2 @addr_200
del t1             # free t1 @addr_100 (被录制)
t3 = op3(t2)      # 分配 t3 → 复用 addr_100
```

分配器知道 `addr_100` 在 `op3` 执行时已被 free，所以 `t3` 可以复用它。

#### 跨图的复用（同一 mempool）

```
fwd_graph 捕获时:
  分配 activation @addr_A → 前向结束后标记为 free

bwd_graph 捕获时:
  需要 grad_buffer → 发现 addr_A 已 free → 复用 addr_A
```

这就是为什么捕获顺序必须和运行顺序一致：分配器按时序记录了哪些内存在什么时候可以被谁复用。

#### 静态张量不参与复用

`static_input_surface`、`static_outputs`、`static_grad_outputs`、`static_grad_inputs` 在整个训练过程中始终占用内存，是固定的"内存槽位"。

### 3.5 回放时的内存流转

```
┌──────────────────────────────────────────────────────┐
│                   共享 mempool                        │
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │static_input │  │static_output│  │static_grad   │ │
│  │  surface    │  │             │  │  _outputs     │ │
│  │ [x][w][b]   │  │ [out1][out2]│  │ [g1][g2]    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘ │
│         │                │                │          │
│     fwd_graph           │         bwd_graph         │
│    (读这里)        (写到这里)     (读这里)              │
│                                                      │
│  ┌──────────────┐                                    │
│  │static_grad   │                                    │
│  │  _inputs     │                                    │
│  │ [gx][gw][gb] │  ← bwd_graph 写到这里              │
│  └──────────────┘                                    │
└──────────────────────────────────────────────────────┘
```

**前向回放**：

```python
# Graphed.forward
for i in range(len_user_args):
    if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
        static_input_surface[i].copy_(inputs[i])  # 写入固定槽位
fwd_graph.replay()                                 # 从固定槽位读，写到固定槽位
return tuple(o.detach() for o in static_outputs)   # 从固定槽位读结果
```

- **必须用 `copy_` 而非赋值**：赋值会改变指针，和图录制时的地址不一致
- **`data_ptr()` 检查**：如果数据恰好已在同一内存，跳过 copy
- **Module 参数不需要 copy**：循环只遍历 `range(len_user_args)`，Module 参数由优化器原地更新，地址天然不变

**反向回放** 同理：`copy_` 上游梯度 → `replay()` → 从固定地址读梯度。

### 3.6 图模式 vs 普通执行的内存对比

| 对比项 | 普通 eager 执行 | 图模式 |
|--------|----------------|--------|
| 每次前向的内存分配 | 每次都要 `malloc` 中间张量 | 零次 `malloc`，全部复用固定地址 |
| 输入传递 | 直接使用 | `copy_` 到固定槽位（一次 memcpy） |
| 中间张量 | 每次分配/释放 | 地址固定，跨图复用 |
| 内存碎片 | 可能产生碎片 | mempool 内布局固定，无碎片 |
| 额外内存开销 | 无 | 需要"冻结"一份静态张量的固定内存 |

**一句话总结**：图模式用「捕获时固定内存布局 + 回放时 copy_ 进固定槽位」的方式，把每次迭代的大量 malloc/free/调度开销**完全消除**，代价是多占一份静态张量的固定内存。

---

## 四、内存峰值分析与 Debug 方法

### 4.1 方法一：`memory_stats` 宏观监控

在关键位置插桩，对比捕获前后、回放前后的内存变化：

```python
import torch_npu

def print_mem(tag):
    allocated = torch_npu.npu.memory_allocated() / 1024**2
    reserved = torch_npu.npu.memory_reserved() / 1024**2
    max_alloc = torch_npu.npu.max_memory_allocated() / 1024**2
    print(f"[{tag}] allocated={allocated:.1f}MB, reserved={reserved:.1f}MB, peak={max_alloc:.1f}MB")

torch_npu.npu.reset_peak_memory_stats()
print_mem("before capture")
graphed_model = torch_npu.npu.make_graphed_callables(model, sample_args)
print_mem("after capture")

for step in range(10):
    torch_npu.npu.reset_peak_memory_stats()
    output = graphed_model(real_input)
    loss = criterion(output, target)
    loss.backward()
    print_mem(f"step {step}")
```

核心指标：

| 指标 | API | 含义 |
|------|-----|------|
| 当前分配 | `memory_allocated()` | 活跃张量占用的内存 |
| 峰值分配 | `max_memory_allocated()` | 本轮最高点 |
| 当前预留 | `memory_reserved()` | Caching Allocator 向设备申请的总内存 |
| 峰值预留 | `max_memory_reserved()` | 预留的最高点 |

**判断标准**：`reserved - allocated` 越大，说明碎片或未复用越多。图模式下该差值应趋于稳定。

### 4.2 方法二：`memory_summary` 全景报告

```python
print(torch_npu.npu.memory_summary())
```

输出包含 Cur Usage / Peak Usage / Tot Alloc / Tot Freed 四列，以及 Allocated memory、Active memory、NPU reserved memory、Non-releasable memory 等指标。

- **Non-releasable memory 高** → 大量被 split 的 block 夹在中间无法释放，碎片严重
- **Peak - Current 差值大** → 大量短生命周期的临时分配（图模式应能减小该差值）

### 4.3 方法三：`memory_snapshot` 细粒度分析

查看每个 segment、每个 block 的详细状态：

```python
snapshot = torch_npu.npu.memory_snapshot()
for seg in snapshot:
    pool_id = seg["segment_pool_id"]
    total_size = seg["total_size"] / 1024**2
    print(f"Segment: pool={pool_id}, size={total_size:.1f}MB")
    for block in seg["blocks"]:
        print(f"  Block: state={block['state']}, size={block['size']/1024**2:.2f}MB")
```

**图内存专项分析**（使用 `_graph_tree.py` 的工具函数）：

```python
from torch_npu.npu._graph_tree import get_npugraph_segments, get_block_addrs

pool_id = fwd_graph.pool()

# 查看图内存池的所有 segment
segments = get_npugraph_segments(pool_id)
for seg in segments:
    print(f"Segment addr=0x{seg['address']:x}, size={seg['total_size']/1024**2:.1f}MB")
    for blk in seg["blocks"]:
        print(f"  {blk['state']}: {blk['size']/1024**2:.2f}MB")

# 查看所有活跃的 block 地址
live_addrs = get_block_addrs(pool_id, live_only=True)
print(f"Live blocks in graph pool: {len(live_addrs)}")
```

### 4.4 方法四：`_record_memory_history` + 调用栈追踪

最详细的 debug 手段，记录每一次分配/释放的 Python 调用栈：

```python
# 开启记录
torch_npu.npu.memory._record_memory_history(enabled="all", context="all", stacks="all")

# 执行要分析的代码
graphed_model = torch_npu.npu.make_graphed_callables(model, sample_args)
output = graphed_model(real_input)

# 导出 snapshot（包含 device_traces）
import pickle
snapshot = torch_npu._C._npu_memorySnapshot()
with open("mem_snapshot.pickle", "wb") as f:
    pickle.dump(snapshot, f)

# 关闭记录
torch_npu.npu.memory._record_memory_history(None)
```

导出的 snapshot 中 `device_traces` 每条 trace 包含：
- 事件类型：`ALLOC` / `FREE_REQUESTED` / `FREE_COMPLETED` / `SEGMENT_ALLOC` / `OOM`
- 地址、大小
- 完整的 Python 调用栈（可定位到具体哪行代码触发了分配）

### 4.5 方法五：对比分析法（判断峰值是否合理）

图模式理论内存构成：

```
图模式理论内存 =
    模型参数 (weights)
  + 优化器状态 (Adam: momentum + variance = 2× 参数大小)
  + 静态输入表面 (static_input_surface)
  + 静态输出 (static_outputs)
  + 静态梯度输出 (static_grad_outputs)
  + 静态梯度输入 (static_grad_inputs)
  + 图内中间激活 (被冻结在 mempool 中)
  + 少量对齐/碎片开销
```

手动估算：

```python
def estimate_graph_memory(model, sample_args):
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
    input_mem = sum(a.numel() * a.element_size() for a in sample_args)

    print(f"Parameters: {param_mem/1024**2:.1f}MB")
    print(f"Inputs: {input_mem/1024**2:.1f}MB")
    print(f"Estimated grad: {param_mem/1024**2:.1f}MB")
    print(f"Optimizer (Adam 2x): {2*param_mem/1024**2:.1f}MB")
```

对比实际 `max_memory_allocated()`，如果实际值远超估算，说明有额外的内存泄漏或碎片问题。

### 4.6 方法六：`check_memory_pool` 一致性检查

```python
from torch_npu.npu._graph_tree import check_memory_pool

check_memory_pool(
    device=0,
    pool_id=fwd_graph.pool(),
    live_storages_ptrs=known_live_storages
)
```

如果有不在预期列表中的 `active_allocated` block，会抛出异常并打印分配时的调用栈，帮助定位内存泄漏。

### 4.7 常见问题诊断清单

| 现象 | 可能原因 | 排查方法 |
|------|---------|---------|
| 图捕获后内存暴涨 | 预热产生的中间张量未释放 | `memory_snapshot` 看非图池的大量 block |
| 回放时内存持续增长 | 图外有张量引用了图内输出（阻止释放） | 检查是否对 `static_outputs` 做了持久引用 |
| `reserved` 很大但 `allocated` 小 | 碎片严重 | `memory_summary` 看 Non-releasable memory |
| 多个图的总内存超预期 | 图之间没有共享 mempool | 确认所有图都用同一个 `pool` token |
| OOM 发生在图捕获期间 | 预热 + 捕获同时存在导致双倍内存 | 捕获前 `empty_cache()`，减小 batch size |

---

## 五、使用约束与注意事项

1. `sample_args` 必须全部是 `torch.Tensor`，不支持其他类型
2. `sample_args` 中每个 Tensor 的 `requires_grad` 状态必须与真实训练时一致
3. 不支持高阶微分（如 double backward）
4. Module 传入时不能有已注册的 hook（传入后注册可以）
5. Module 的 buffer 必须 `requires_grad=False`
6. 图化后不能添加或删除 Module 的参数或 buffer
7. 调用时的参数顺序和格式必须与 `sample_args` 一致
8. 使用 AMP 时必须设置 `cache_enabled=False`
9. 如果传入多个 callable，它们在 tuple 中的顺序必须与实际运行顺序一致

---

## 六、总结

`make_graphed_callables` 通过「预热 → 捕获前向图 → 捕获反向图 → 包装为 autograd Function」的流程，将 callable 的前向和反向计算**一次性录制**成 NPU Graph。回放时通过 `copy_` + `replay()` 模式，以**固定内存布局、零 malloc、一次性提交**的方式执行，显著降低了 CPU 侧的算子调度开销和内存分配开销。

其内存管理的核心是**私有内存池 + 地址固化 + 生命周期复用**：所有图共享一个 mempool，捕获时记录的内存分配/释放时序在回放时被严格重放，使得不同图、不同生命周期的张量可以安全地复用同一块设备内存。

## Related Pages

- [[02_engineering/01_pytorch/index]]
- [[11_torch_compile_npugraphs_deepdive]]
