---
title: "AsyncCollectiveTensor：为可追踪而生的惰性 wait，以及它掩盖不了的东西"
---

# AsyncCollectiveTensor：为可追踪而生的惰性 wait，以及它掩盖不了的东西

*一条主线：ACT 不是为通信掩盖设计的调度器，而是 functional collectives 为了在 eager 下复刻「编译器会替你插 wait」这件事所做的替身；掩盖只是它的副产品，因此掩盖窗口有多大完全取决于调用方怎么排列代码，而 ACT 自己什么都排不了。*

> **源码基线**：`pytorch/torchtitan@a3168782c9a3a2e40afbd0de114818b96e2bda6e`（`main`，2026-08-27）；ACT 本体为 `pytorch/pytorch@ea5655fcebf726ec4cf1a859de75d2d0e6425805`（`main`，2026-07-21，工作区仅 submodule 指针有改动，`torch/distributed/_functional_collectives.py` 与 `torch/csrc/distributed/c10d/` 均为干净树）。§4.6 的 Megatron 对照取 `NVIDIA/Megatron-LM@71092579522a12522d9f323ae180c9825d01928a`（`dev`，2026-08-27），与本库 Megatron 系列同基线。
> **最后更新**：2026-08-28 · **叙事顺序**：本页按五拍组织——背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势。
>
> **本页回答**：`AsyncCollectiveTensor`（下称 ACT）到底是什么机制、它在 PyTorch 里为什么长成 tensor subclass 这个样子、一次 wait 在 CUDA 上究竟同步了什么，以及在**当前基线**下 torchtitan 还有没有真正靠 ACT 吃到的通信掩盖窗口。
>
> **重写说明**：本页 2026-08-28 之前的版本，以 torchtitan `AllToAllTokenDispatcher.combine()` 里「shared experts 与 combine all-to-all 并行」作为贯穿全文的例子。该例子在当前基线下**已不成立**——`torchtitan/models/common/moe.py:440-449` 把 shared experts 严格排在 routed path 之后。旧论证连同其代码块与两张配图**整体保留**在**附录 A**，并标注了它成立的那个 commit 区间。同一判定另见 [[02_engineering/02_train_frameworks/torchtitan/15_torchtitan_ep_analysis|专家并行 EP]] §5。
>
> **边界**：本页只追 ACT 这一个机制。torchtitan 的 EP dispatcher 家族（standard / DeepEP / HybridEP / MinimalAsyncEP）归 [[02_engineering/02_train_frameworks/torchtitan/15_torchtitan_ep_analysis|专家并行 EP]]；框架级的重叠手段清单归 [[02_engineering/02_train_frameworks/30_comm_compute_overlap_analysis|通信与计算重叠]]；图级调度归 [[02_engineering/02_train_frameworks/torchtitan/27_torchtitan_graph_trainer_compiler_runtime_analysis|graph trainer 编译与运行时]]。

---

## 1. 背景：把 Work handle 从 collective 的返回值里拿掉之后，wait 该由谁来发

朴素的异步集合通信长这样：`dist.all_reduce(t, async_op=True)` 返回一个 `Work` 句柄，用户自己找地方 `handle.wait()`。这套 API 有两个问题：**它不可追踪**（`Work` 是一个 Python 侧的可变对象，dynamo/FX 捕获不到它，也没法把 wait 当成图里的一个节点搬动），并且**它把同步点的位置写死在用户代码里**（用户在哪调 `wait()`，重叠窗口就到哪为止，编译器无从优化）。

PyTorch 因此新开了一套 functional collectives。引入它的提交把目的写得很直白：`e22d791287d1c00cd11b9229ad15d030224ab4b1`（2023-02-16，#93990，commit message：「This experimental API enables collectives to be fully traced by dynamo and FX.」）。**ACT 与 functional collectives 是同一个提交引进的**——也就是说，ACT 从第一天起就不是一个"通信掩盖工具"，而是"让 collective 变成纯函数、让 wait 变成一个可搬动的 op"这件事在 eager 侧留下的补丁。

问题于是变成：函数式的 collective 返回一个 tensor 而不是 handle，那**谁来插 wait**？源码在模块顶部的设计说明里给出了两条不同的答案（`torch/distributed/_functional_collectives.py:109-135`）：

> These apis are called by user code and expected to work both in eager execution and compilation, but there are significant differences to how the two modes are implemented underneath.（`:113-114`）

- **编译路径**：由编译器负责——「Compiled tracing currently relies on the compiler to perform this optimization」（`:117`）。
- **eager 路径**：由一个 tensor subclass 负责——「Eager execution is 'optimized' using a tensor subclass that schedules the synchronization (via `wait_tensor()` op) just before the tensor is first used.」（`:116-117`）

这个 subclass 就是 ACT。它要解决的问题精确地是：**在没有编译器、也不许用户拿 handle 的前提下，把 wait 推迟到"数据第一次被真正用到"的那一刻。**

---

## 2. 为什么这么设计：让 eager 顶替编译器，而不是让用户拿着 handle

### 2.1 源码直接陈述的判据：两条路径的对称性

模块 docstring 用两段并排的伪调用链把取舍摆出来（`torch/distributed/_functional_collectives.py:123-134`）：编译下 `_maybe_wrap_tensor(...)` 的注释是「wait_tensor() op is immediately called, **no AsyncTensor subclass needed**」（`:127`），eager 下则是「AsyncTensor wrapper applied to returned tensor, which issues wait_tensor() **at the time of first use**」（`:133-134`）。

这段代码在 `_maybe_wrap_tensor` 里一比一落实（`:1404-1407`）：

```python
def _maybe_wrap_tensor(self) -> torch.Tensor:
    if _are_we_tracing():
        return wait_tensor(self)          # 编译/追踪：立刻插 wait，交给编译器去搬
    return _wrap_tensor_autograd(self)    # eager：包成 ACT，推迟到首次使用
```

`_are_we_tracing()`（`:1390-1401`）用四个条件判定：dynamo 正在编译、FakeTensorMode 打开、PythonDispatcher key 打开、或存在 proxy mode。**任何一个成立就不产生 ACT。**

被否掉的替代因此不是"某种更好的 ACT"，而是"在 eager 下也照编译路径那样立刻 wait"。选 ACT 的判据就写在 docstring 里：eager 没有一个能看到整段程序的 pass，能替用户搬 wait 的只有 dispatcher；把判断塞进 `__torch_dispatch__`，用户代码一行不改就能拿到"通信不阻塞、直到你真的读它"的语义。

### 2.2 第二条被否掉的替代：为 autograd 单开一套 C++ 算子

早期为了让 collective 可反传，PyTorch 在 `_c10d_functional_autograd.*` 下另建了一套 C++ 实现，`*_autograd` 系列 Python wrapper 走那一套。这条路已经被整段删除：`7f1ec59073d082d0ba9cbaacb18c234ee709dd96`（2026-06-05，#172792）的 commit message 写明「[a] removing C++ implementations in `_c10d_functional_autograd.*`；[b] re-routing python `.*_autograd` counterparts to use standard functional collectives instead」，并明确「Note: [b] is for backward compatibility and can be removed at sometime in the future」。

结果就是当前的 `all_to_all_single_autograd`（`torch/distributed/_functional_collectives.py:594-605`）只剩一层转发：

```python
def all_to_all_single_autograd(self, output_split_sizes, input_split_sizes, group, tag=""):
    """
    Same as all_to_all_single but supports autograd.
    """
    # The base all_to_all_single now has autograd support, so we can just call it
    return all_to_all_single(self, output_split_sizes, input_split_sizes, group, tag)
```

> [!contradiction] 与本页旧版的差异
> 旧版在这里贴的是 `torch.ops._c10d_functional_autograd.all_to_all_single(...)` 加 `_FromTorchTensor.apply(tensor)`。这两个符号在当前基线下**都已不存在于该路径**：C++ 侧被 `7f1ec59073` 删除，Python 侧改为转发到 `all_to_all_single`（`:541-591`），由 `_maybe_wrap_tensor(tensor)`（`:591`）统一决定包不包 ACT。autograd 现在由 `torch.library.register_autograd` 挂在 `_c10d_functional::wait_tensor`（反向恒等 `:613-625`，注册 `:639-643`）与 `_c10d_functional::_wrap_tensor_autograd`（反向恒等 `:1356-1381`，注册 `:1384-1387`）上。

### 2.3 源码沉默处：为什么是"首次使用"而不是"作用域结束"

> [!note] 推断
> 源码只说 wait 发生在「first use」，**没有**解释为什么不选别的触发点（例如作用域退出、或显式的 region 边界）。本页的重建是：`__torch_dispatch__` 是 eager 下唯一一个"既能看见张量被消费、又不需要用户配合"的钩子；作用域式的触发需要用户标注 region，那就退回到了 handle 方案。要引用这条判断，请回到 `torch/distributed/_functional_collectives.py:116-119`（设计陈述）与 `:1130-1167`（唯一的触发点实现），不要引用本段推断。

---

## 3. 实现思路与细节：三层机制 + 一条可追的调用链

一句话：**ACT = 一个只有 metadata 是真的、数据还在路上的替身张量（Layer 1）+ 一个按"是不是 view op"决定要不要同步的 dispatch 拦截器（Layer 2）+ 一次按 storage 反查 NCCL Work 的 stream 级 block（Layer 3）。**

### 3.1 Layer 1：wrapper subclass —— 只借 metadata，不借数据

`class AsyncCollectiveTensor(torch.Tensor)`（`torch/distributed/_functional_collectives.py:1057`），类 docstring 自陈用途是「trigger a call to wait prior to first use of the underlying tensor」（`:1059-1060`）。状态只有两项（`:1068-1071`）：

```python
elem: torch.Tensor       # 底层 raw tensor（通信的输出缓冲，数据可能还没到）
completed: bool          # 是否已经 wait 过
__slots__ = ["elem", "completed"]
```

`__new__`（`:1073-1087`）用 `torch.Tensor._make_wrapper_subclass` 造出一个 size / stride / storage_offset / dtype / layout / device / requires_grad 与 `elem` 完全一致的空壳，再把 `r.elem = elem`、`r.completed = False` 挂上去。

这一层的意义是：**读 `.shape` / `.dtype` / `.device` 不经过 dispatcher，因此不触发同步**。调用方可以先按形状做各种准备工作，通信仍在飞。

### 3.2 Layer 2：`__torch_dispatch__` —— view 放行，非 view 同步

`__torch_dispatch__`（`:1130-1167`）分三段：

1. **`aten.view.default` 快路径**（`:1132-1138`）：直接对 `args[0].elem` 施加，再包一层 ACT 返回，连 `_is_view_op` 与 pytree 都不走。注释说明动机是「a lot of view related op goes to aten.view eventually, this avoids pytree slowdown」（`:1133-1134`）。
2. **unwrap 决策**（`:1140-1146`）：

```python
is_view_op = _is_view_op(func)

def unwrap(e: AsyncCollectiveTensor):
    if not is_view_op:
        return e.trigger_wait()   # 非 view op → 同步
    return e.elem                 # view op → 直接借 raw tensor，继续推迟
```

3. **输出重包**（`:1160-1167`）：只有 view op 的输出会被重新包成 ACT（`:1163-1165`），非 view op 的输出是已同步的普通张量，「we don't wrap the result as it doesn't need to be waited on」（`:1160`）。

`_is_view_op`（`:525-540`）的判据要点有二，都与常见的口口相传不同：
- **先排除 `CompositeImplicitAutograd`**（`:531-535`，注释指向 issue #133421）——这类 op 一律**不**当作 view，也就是一律触发同步；
- 再看 schema 第一个参数的 `alias_info`：「非空且不是写入」才算 view（`:536-540`）。函数上方的注释自陈「This is a bit unsafe … Today, this maps 1:1 with "aten ops that are views"」（`:523-524`），即这是一条**经验等价**而非语义保证。

`trigger_wait`（`:1115-1121`）保证幂等：

```python
def trigger_wait(self):
    if not self.completed:
        out = wait_tensor(self.elem)
        self.completed = True
        return out
    else:
        return self.elem
```

另有一个不置 `completed` 的 `wait()`（`:1123-1124`），DTensor 的 redistribute 走的是它（见 §4.4）。

### 3.3 Layer 3：`wait_tensor` —— 按 storage 反查 Work，然后阻塞 stream 而不是 CPU

Python 侧 `wait_tensor`（`:138-144`）只是转发到 `torch.ops._c10d_functional.wait_tensor`，docstring 一句话点题：「Waiting follows device semantics, which means blocking on CPU and synchronizing streams on CUDA.」（`:140-142`）

C++ 侧真正干活的是 `c10d::wait_tensor`（`torch/csrc/distributed/c10d/ProcessGroup.cpp:427-453`）：

1. 用张量的 **weak StorageImpl** 作为 key，去线程本地的 `WorkRegistry` 里 `pop_works(tensor)`（`:427-429`；registry 的键类型见 `:407-410`，登记入口见 `:286-329`）；
2. 本线程找不到时，跨所有 rank-local registry 再找一遍——注释说明这是为了覆盖「wait() is called from a different thread than where the collective was initiated」（`:431-447`）；
3. 对找到的每个 Work 调 `work->wait()`（`:449-451`），最后原样返回 tensor（`:452`）。

注意两个容易被略过的边界：**没有 storage 的张量登记不上**，`register_work` 会 `TORCH_WARN_ONCE` 并直接返回，警告文本明说「Calling c10d_functional.wait_tensor() on this tensor will not wait for the collective to complete」（`torch/csrc/distributed/c10d/ProcessGroup.cpp:293-299`）；进程结束时若还有未 wait 的 Work，析构函数只发警告并**故意泄漏**这些对象（`:381-403`）。

`work->wait()` 落到 NCCL 后端是 `ProcessGroupNCCL::WorkNCCL::wait`（`torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:830-870`）。它先 `synchronize()`（`:846`，注释「synchronize() will block the current stream on the NCCL stream」），核心是 `synchronizeStream()`（`:820-827`）：

```cpp
auto currentStream = at::cuda::getCurrentCUDAStream(device_.index());
// Block the current stream on the NCCL stream
ncclEndEvent_->block(currentStream);
stashed_for_allocator_safety_->unstash();
```

即**在 compute stream 上等一个 event，CPU 不阻塞**。只有当 `blockingWait_`（由 `TORCH_NCCL_BLOCKING_WAIT` 决定，`:955`）为真或调用方给了 timeout 时，才额外把 CPU 线程轮询到完成（`:850-869`）。这是"通信掩盖"成立的物理基础：wait 只是往 compute stream 里插了一个依赖，launch 队列继续往前跑。

### 3.4 一条可追的调用链（torchtitan MoE dispatch）

以 torchtitan 的 `AllToAllTokenDispatcher.dispatch()` 为例，**只在会产生 ACT 的分支上**（分支条件见 §4.3）：

| # | 位置 | 发生了什么 |
|---|---|---|
| ① | `torchtitan/models/common/token_dispatcher.py:459-464` | `_dispatch_token_exchange(...)` 发起数据 all-to-all，返回值赋回 `routed_input_RD` |
| ② | `torchtitan/models/common/token_dispatcher.py:328-333` | 该分支调 `torch.distributed._functional_collectives.all_to_all_single`（import 见 `:13`） |
| ③ | `torch/distributed/_functional_collectives.py:585-591` | 下发 `_c10d_functional.all_to_all_single`，再 `_maybe_wrap_tensor(tensor)` |
| ④ | `torch/distributed/_functional_collectives.py:1404-1407` | eager → `_wrap_tensor_autograd` → `AsyncCollectiveTensor(input)`（`:1330-1343`） |
| ⑤ | `torchtitan/models/common/token_dispatcher.py:512`、`:543` | 只读 `routed_input_RD.shape` —— 走 metadata，**不进 dispatcher，不同步** |
| ⑥ | `torchtitan/models/common/token_dispatcher.py:509-541` | `_permute` 在 compute stream 上构造置换索引（cumsum / `repeat_interleave` / arange），**全程不碰 ACT 的数据** |
| ⑦ | `torchtitan/models/common/token_dispatcher.py:544` | `routed_input_RD[permuted_indices, :]` —— 非 view op，`unwrap` 调 `trigger_wait()`，同步在此发生 |

⑤⑥ 之间那段就是当前基线下**真实存在的 ACT 掩盖窗口**，⑦ 是窗口的右端。它比一个 shared-expert MLP 小得多，但它是这条路径上确实存在、可以逐行走出来的那个窗口。

---

## 4. 约束：ACT 掩盖的三个前提，以及当前 torchtitan 满足了几个

### 4.1 三个前提

ACT 产生掩盖，必须同时满足：

> ① **发起通信**与**掩盖计算**位于**同一个调用栈**、且掩盖计算排在通信之后；
> ② 掩盖计算与通信结果**没有数据依赖**；
> ③ 掩盖计算与通信结果之间**没有其它非 view op 提前碰到那个 ACT**。

这三条不是设计目标，而是 §3.2 那套 dispatch 规则的直接推论——ACT 不选择顺序，它只是在你写下的顺序里找第一个非 view op。

### 4.2 前提②在当前 torchtitan 的 MoE 上已不成立

`MoE.forward`（`torchtitan/models/common/moe.py:396`）的排列是：

```python
out_TD = self.routed_experts(                      # moe.py:440
    x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E,
)

shared_out_TD = (                                  # moe.py:447
    self.shared_experts(x_TD) if self.shared_experts is not None else None
)

if shared_out_TD is not None:                      # moe.py:451
    out_TD = out_TD + shared_out_TD                # moe.py:452
return out_TD                                      # moe.py:453
```

`self.routed_experts(...)`（`:440`）内部完整走完 dispatch → grouped GEMM → combine（`torchtitan/models/common/moe.py:150-169`），**返回时 combine 的 all-to-all 早已被 scatter-add 同步掉**。shared experts 在 `:447` 才开始，此时没有任何在飞的 collective 可供掩盖。**窗口不存在，不是变小了。**

把 shared experts 搬出 dispatcher 的是 `963c20cba37f392fcfdff3b1c7519fde8ad4c0a7`（2026-05-20，#3386「Refactor MoE to clean DTensor boundaries for shared/routed experts」）。它在 Key changes 里第一条就写着「Move `shared_experts` computation from inside `TokenDispatcher.combine()` to `MoE.forward()`」，理由通篇是 DTensor 边界（router / shared experts / routed experts 三段各自的 `to_local` 与 placement 语义）。

> [!note] 推断
> 该提交**没有**提到它顺带删掉了一个通信掩盖窗口，也没有给出"这个窗口不值钱"的论证。「掩盖窗口是这次 DTensor 边界重构的附带损失」是本页的推断，不是提交作者的自陈。要引用事实部分，请用 `963c20cba37` 的 commit message（移动 shared_experts）与 `torchtitan/models/common/moe.py:440-453`（当前顺序），不要把这条推断当成上游意图。

### 4.3 当前 torchtitan 里还活着的 ACT 窗口

先要说清楚一件常被跳过的事：**`AllToAllTokenDispatcher` 在默认配置下根本不产生 ACT。**三处收发都长成同一个形状（`token_dispatcher.py:258-273`、`:325-341`、`:353-370`）：

```python
if (torch.compiler.is_compiling() or torch.compiler._is_non_strict_tracing()) \
        or get_spmd_backend() != "spmd_types":
    return all_to_all_single(...)          # torch.distributed._functional_collectives
return spmd.all_to_all(...)                # 外部包 spmd_types
```

把三种运行态代进去：

| 运行态 | 走哪条分支 | 是否产生 ACT |
|---|---|---|
| eager + `spmd_backend="spmd_types"`（**默认**，`torchtitan/distributed/utils.py:36`） | `spmd.all_to_all` | 本页**无法核实**（见下） |
| eager + `spmd_backend="partial_dtensor"` | `funcol.all_to_all_single` | **是** |
| `torch.compile` / 非严格追踪 | `funcol.all_to_all_single` | **否**——`_are_we_tracing()` 为真，`_maybe_wrap_tensor` 直接插 `wait_tensor`（`_functional_collectives.py:1404-1407`） |

`spmd` 来自 `import spmd_types as spmd`（`token_dispatcher.py:11`），是一个 pip 依赖 `spmd_types==0.2.5`（`pyproject.toml:30`、`.ci/docker/requirements.txt:10`），**不在本检出内**，因此本页不对该分支是否返回 ACT 作任何断言。

于是当前基线下可核实的 ACT 窗口只有两类：

**① dispatch 里的索引构造窗口（`partial_dtensor` + eager）。** 即 §3.4 的 ⑤⑥，右端在 `token_dispatcher.py:544`。

**② combine 里几乎为零的窗口。** `combine()` 的源码注释仍然自陈 ACT 语义——「All-to-all combine: returns AsyncCollectiveTensor — the a2a runs on the NCCL stream and won't block until the tensor is accessed.」（`token_dispatcher.py:589-590`）——但发起（`:591-596`）之后只隔着一句与 ACT 无关的 `out_TD = torch.zeros_like(x_TD)`（`:604`），紧接着就是 `routed_output_RD.to(torch.float32) * ...`（`:606-609`），非 view op，同步在此。

> [!contradiction] 源码内部的不一致
> `token_dispatcher.py:589-590` 的注释是 `09ea7d8e7391b9cd0edbe4658e0415a51adf8bbd`（2026-04-17，#2842）留下的，写于 shared experts 还在 `combine()` 内部、且 `spmd_types` 分支尚未加入（`b052f36fe50d`，2026-06-25，#3654）之时。它今天既不描述默认分支（走 `spmd.all_to_all`），也不再对应任何有意义的窗口。**以代码顺序为准，不要以该注释为准。**

**③ 真正被"主动选用"的 ACT：TP 边界的 redistribute。** DTensor 的 `redistribute` 默认 `async_op=False`（`torch/distributed/tensor/_api.py:715`、`torch/distributed/tensor/_redistribute.py:1538`），此时 `redistribute_local_tensor` 会在返回前把 ACT `wait()` 掉（`:1767-1769`）。torchtitan 在四处**显式反选**这个默认值：

- `torchtitan/distributed/tensor_parallel.py:63-65`（`NoParallel` 的输入 redistribute）与 `:77`（输出 redistribute，随后 `:78` `to_local()`）；
- `torchtitan/protocols/module.py:653`（按 `sharding_config` 归一输入）与 `:728`（归一输出）。

`DTensor.to_local()` 的 docstring 明说这种情况下拿到的就是 ACT：「When an `AsyncCollectiveTensor` object is returned, it means the local tensor is not ready yet (i.e. communication is not finished).」（`torch/distributed/tensor/_api.py:664-668`）。

> [!note] 推断
> 这四处 `async_op=True` **源码没有写理由**，提交历史里它们也是随 `NoParallel` / sharding-config 机制一起进来的，没有独立的性能说明。「torchtitan 在这里是有意把 wait 推到模块体内部的第一个消费点，以换取 module 入口开销与 redistribute 的重叠」属本页推断。要引用，请回到 `torchtitan/distributed/tensor_parallel.py:63-65`、`:77-78` 与 `torchtitan/protocols/module.py:653`、`:728` 这四个 locator 加上 `_redistribute.py:1767-1769` 的默认行为，不要引用本段动机。

### 4.4 ACT 自身的开销与两个逃逸口

**ACT 不是免费的抽象**，上游自己就在热路径上主动绕开它。`all_gather_single` 在 `gather_dim != 0` 时会判断能否用 view 优化，不能就**提前 wait**，注释写明动机（`torch/distributed/_functional_collectives.py:215-224`）：

> Check if `_maybe_view_chunk_cat` can use the view optimization. If not, it will use `torch.cat` which needs the data anyway, so **wait early to avoid AsyncCollectiveTensor dispatch overhead**.

这条注释来自 `b6de337d169795e105de1f7c40fe1e60bc65eb8d`（2023-12-05，#113324），同一提交也加了 §3.1 的 `aten.view` 快路径，commit message 把两件事并列为「a few optimizations to funcol」。**即：`__torch_dispatch__` 的拦截成本大到需要在已知无收益时提前放弃惰性。**

两个逃逸口：

- **绕过 dispatcher 的 kernel 看不见 ACT。** torchtitan 在 `_sync_token_count_exchange` 里手工插了一次 wait，注释写明原因（`torchtitan/models/common/token_dispatcher.py:286-292`）：「Need to wait explicitly because it is used by a **triton kernel** later which doesn't realize that AsyncCollectiveTensor needs unwrapping」。自定义/外部 kernel 直接吃指针，不经过 `__torch_dispatch__`，ACT 的保护在这里失效。
- **没有 storage 的张量登记不上 Work**（`ProcessGroup.cpp:293-299`），wait 变成静默的 no-op。

此外还有一处 CPU 侧的硬同步与 ACT 无关但常被算在它头上：同一函数把 output splits 以 `non_blocking=False` 搬回主机（`token_dispatcher.py:302-306`，注释「this would incur a device-to-host sync」），这是动态 dropless 的固有代价。

### 4.5 为什么 ACT 无法跨 micro-batch

![图 3：ACT 的能力边界](assets/async_collective_tensor_deep_dive_fig3.png)

*图 3：ACT 的能力边界。* **注**：图中上半区「ACT 能做的」一行里的 `shared_expert(并行)` 描述的是 `963c20cba37`（#3386）之前的形态，当前基线下该并行段已不存在（§4.2）；下半区「Megatron combined_1f1b 需要做的」与「核心差距」两段仍然成立。

ACT 是一种**惰性同步**，不是**调度**：它能让"发起"与"等待"之间插进别的计算，但这些计算必须由**同一个 Python 调用栈按顺序写出来**。跨 micro-batch 的交错需要有人在两个 micro-batch 的半途之间来回切换，而 PyTorch pipelining 提供的最小单元不在那个粒度上：

- `_PipelineStageBase`（`torch/distributed/pipelining/stage.py:186`）暴露的是 `forward_one_chunk`（`:944`）/ `backward_one_chunk`（`:1024`）；`forward_one_chunk` 内部就是一次 `self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)`（`:979`）——**整个 stage 的 forward 是一次原子调用**，没有"做完 dispatch 就交出控制权"的接口。
- 调度动作的枚举 `_ComputationType`（`torch/distributed/pipelining/schedules.py:54-66`）只有 `F` / `I` / `W` / `B` 以及 `UNSHARD` / `RESHARD` / 通信动作这一级，最细也是 stage × micro-batch，没有 sub-layer 节点。

一个值得单独指出的**新增项**（旧版页面写于其存在之前）：枚举里现在有 `OVERLAP_F_B`（`schedules.py:66`），由 `ScheduleDualPipeV`（`:3418-3424`，docstring 指向 DeepSeek DualPipe）在 `add_overlap_f_b` 中生成，把一个 `_Action(forward_stage, FORWARD, ...)` 与一个 `_Action(backward_stage, FULL_BACKWARD, ...)` 打包成一条动作（`:3506-3523`）。但运行时对它的处理是（`:2657-2661`）：

```python
elif action.computation_type == OVERLAP_F_B:
    if action.sub_actions is None:
        raise AssertionError("sub_actions must be set")
    for sub_a in action.sub_actions:
        _perform_action(sub_a)
```

**顺序执行两个子动作**。也就是说 `OVERLAP_F_B` 是一条"这两件事打算重叠"的标注，重叠本身仍要靠模型内部自己发出的异步通信去实现；子动作的粒度依旧是整段 stage F / B。结论不变：**ACT 解决不了调度粒度问题，而 pipelining 的调度粒度本身也还没下沉到 sub-layer。**

### 4.6 对照：Megatron 用显式 stream + event 换到了什么

![图 4：Megatron 的跨 mb + sub-layer 双 stream 调度](assets/async_collective_tensor_deep_dive_fig4.png)

*图 4：Megatron 的跨 mb + sub-layer 双 stream 调度。*

Megatron 不用 ACT。它的 `ScheduleNode`（`megatron/core/pipeline_parallel/utils.py:144-193`）在构造时就同时持有 **stream 与 event**，docstring 明确区分两类 stream 的用途：「'compute' stream: Used for computational nodes like attention and experts；'communicate' stream: Used for nodes that handle token communication, such as token dispatch and combine operations in MoE layers」（`:166-170`），并说明 event 的作用是「Each microbatch within a model chunk shares the same event, which is used to manage dependencies between nodes operating on different streams」（`:171-173`）。

同步由 `stream_acquire_context` 主动编排（`:293-314`）：进入时 `self.event.wait(self.stream)`（`:305`），退出时 `self.event.record(self.stream)`（`:314`）。模型要用这套调度，必须自己实现 `build_schedule_plan`（`AbstractSchedulePlan`，`:324-325`：「To use combined 1f1b, model must implement build_schedule_plan」）——也就是**把 layer 内部结构显式拆成节点交给调度器**。

| | torchtitan（ACT） | Megatron（手动 stream） |
|---|---|---|
| 最小调度单元 | `forward_one_chunk(stage, mb)`（`stage.py:944`） | `ScheduleNode(layer, sub_op, mb)`（`utils.py:144`） |
| stream 管理 | 系统默认 compute + NCCL stream，无显式管理 | 显式 comp / comm stream + 共享 event（`utils.py:166-173`） |
| 同步机制 | 惰性——首个非 view op 触发 `trigger_wait`（`_functional_collectives.py:1142-1146`） | 主动——`event.wait()` / `event.record()`（`utils.py:305`、`:314`） |
| 掩盖范围 | 同一调用栈内、发起点与首个消费点之间 | 跨 mb、跨 fwd/bwd、跨并行维度 |
| 对模型的侵入 | 零——stage 就是 `nn.Module` | 模型必须实现 `build_schedule_plan`（`utils.py:324-325`） |

差异的本质是**调度粒度**，不是"谁的通信更快"：ACT 用零侵入换掉了排布能力，Megatron 用模型侵入换回了排布能力。

---

## 5. 发展趋势：惰性 wait 正在让位给图级调度

以下三条各自锚在源码自陈的在途改动上，**结论部分为本页推断**。

**① 上游打算把两条路径合并，ACT 是待淘汰的那半。** 模块 docstring 自己留了话：「In the future, these paths may be unified if sufficient subclass support is added in dynamo.」（`torch/distributed/_functional_collectives.py:118-119`）。合并方向由同一段的措辞给定——编译路径「relies on the compiler to perform this optimization」（`:117`），ACT 只是 dynamo 还追踪不了 subclass 时的替代品。`7f1ec59073`（#172792）已经把 `*_autograd` 系列降级成「for backward compatibility and can be removed at sometime in the future」的转发层，是同一收敛方向上的一步。

**② torchtitan 对"跨 chunk 掩盖"的答案已经落在图上，不在 eager。** `graph_trainer` 里有一对配套的 pass：`ep_chunk_pass` 负责切（「split selected live-ins → run selected module on each chunk → materialize full live-outs when a non-chunked consumer needs them」，`torchtitan/experiments/graph_trainer/ep_chunk_pass.py:9-12`；「V1 creates two equal chunks because the EP schedule is pairwise」，`:42-44`），`ep_overlap_pass` 负责排（「This pass is intentionally a scheduler only」，`ep_overlap_pass.py:11-12`；它显式识别 `_c10d_functional.all_to_all_single` 的发起点与被规范化的 `_EP_TOKEN_EXCHANGE_WAIT` 等待点，`:20-25`，并「emit wait-gated phases: marker pair in chunk order, ready filler work…」，`:49-50`）。

配套的 eager 侧只负责产元数据、不负责重叠：`_EagerChunkedForward.__call__` 就是 `for chunk_id in (0, 1)` 顺序跑两遍（`ep_eager_chunk.py:318-332`），且整条路径以 `compile_config.enable` 为前提（`:368`）。这套能力当前仍默认关闭（`EpOverlapConfig.enabled = False`，`configs.py:30`），默认 `strategy = "graph"`（`:40`）。

> [!note] 推断
> "ACT 在 torchtitan 的角色正在从掩盖手段退化为纯粹的语义包装"是本页对上述事实的解读。事实部分是：`_maybe_wrap_tensor` 在追踪下不产生 ACT（`_functional_collectives.py:1404-1407`），而 torchtitan 新的重叠能力全部建在追踪之后的 FX 图上（`ep_chunk_pass.py`、`ep_overlap_pass.py`、`ep_eager_chunk.py:368`）。上游没有任何一处声明"我们放弃用 ACT 做重叠"。

**③ 压力来自 §4.4 的开销与 §4.5 的粒度。** ACT 的 dispatch 成本已经逼得 funcol 自己在无收益时提前 wait（`_functional_collectives.py:215-224`）；调度粒度上，`OVERLAP_F_B` 只做到"标注意图、顺序执行"（`schedules.py:2657-2661`）。两条约束指向同一个出口：**把 wait 变成图里一个可以被 pass 搬动的节点**——这正是 `ep_overlap_pass` 在做的事。

---

## 附录 A：已被上游推翻的旧论证——shared experts 掩盖窗口（保留）

> [!contradiction] 本节记录一次**在当时成立**的分析，不是当前行为
> 下面的代码、行号与两张配图对应 `963c20cba37f392fcfdff3b1c7519fde8ad4c0a7`（#3386，2026-05-20）**之前**的 torchtitan，其父提交为 `83e490429cc5bb5637647c89f886fece0ccbb690`（2026-05-20）。在该基线下，本页旧版引用的 `token_dispatcher.py:399-478`、`L432-438`、`L443`、`L448-449`、`L473-477` **逐条属实**。
> 当前基线（`a3168782c9`）下它们已全部失效：`combine()` 移到了 `torchtitan/models/common/token_dispatcher.py:556`，且 `torchtitan/models/common/moe.py:440`（routed）与 `:447`（shared）的先后顺序使这个窗口**不再存在**。同一判定另见 [[02_engineering/02_train_frameworks/torchtitan/15_torchtitan_ep_analysis|专家并行 EP]] §5。

### A.1 旧形态：shared experts 由调用方传进 `combine()`

在 `83e490429cc5` 下，`GroupedExperts.forward` 把 shared experts 模块**当参数传下去**（`torchtitan/models/common/moe.py:78-94`），docstring 写明意图：「shared_experts is passed to combine() where it overlaps with the async …」（`:87`）；`MoE.forward` 在 `:364-369` 传入 `shared_experts=self.shared_experts`。`MoE` 的类 docstring 也把这一步写进流程：「With EP, starts async communication (NCCL all-to-all or DeepEP combine), **runs shared_experts in parallel**, then forces sync (scatter_add for NCCL AllToAll, sync_combine for DeepEP)」（`:268-271`）。

`AllToAllTokenDispatcher.combine()`（`torchtitan/models/common/token_dispatcher.py:399-478` @ `83e490429cc5`）的关键三段：

```python
# token_dispatcher.py:428-439 @ 83e490429cc5
routed_output = self._unpermute(
    routed_output, metadata.input_shape, metadata.permuted_indices
)
# All-to-all combine: returns AsyncCollectiveTensor — the a2a runs
# on the NCCL stream and won't block until the tensor is accessed.
routed_output = all_to_all_single_autograd(
    routed_output,
    metadata.input_splits,
    metadata.output_splits,
    self.ep_mesh,
)

# token_dispatcher.py:441-443 @ 83e490429cc5
# shared_experts overlaps with the async a2a (NCCL stream).
# Score application + scatter_add forces the a2a to sync.
out = shared_experts(x) if shared_experts is not None else torch.zeros_like(x)

# token_dispatcher.py:445-449 @ 83e490429cc5 —— 非 view op，触发 trigger_wait()
if not self.score_before_experts:
    routed_output = (
        routed_output.to(torch.float32)
        * metadata.top_scores_experts_sorted.reshape(-1, 1)
    ).to(routed_output.dtype)

# token_dispatcher.py:473-477 @ 83e490429cc5 —— 结果在此被消费
out = deterministic_scatter_add(
    out,
    token_indices_experts_sorted.reshape(-1, 1).expand(-1, x.shape[-1]),
    routed_output,
)
```

`:408-409` 的 docstring 把机制说得比注释还清楚：「shared_experts overlaps with the async all-to-all combine — it runs while the a2a is in flight on the NCCL stream. scatter_add forces sync.」这是一个**教科书式的 §4.1 三前提全满足**的例子：同一调用栈、无数据依赖、发起与消费之间。

### A.2 旧配图（保留，附更正）

![图 1：combine() 的双 Stream 时间线——ACT 实现的同一 mb 内掩盖](assets/async_collective_tensor_deep_dive_fig1.png)

*图 1：combine() 的双 Stream 时间线——ACT 实现的同一 mb 内掩盖。* **注**：本图刻画的是 A.1 的旧形态。当前基线下 `combine()` 内不再有 `shared_experts(x)` 这一格（`torchtitan/models/common/token_dispatcher.py:556-618`），图中"掩盖效果"一栏不再成立；其余部分（launch A2A → ACT 返回 → 首个非 view op 触发 `trigger_wait` → `wait_tensor`）与当前机制一致。

![图 2：单个 mb 内 MoE forward 的完整双 Stream 执行过程](assets/async_collective_tensor_deep_dive_fig2.png)

*图 2：单个 mb 内 MoE forward 的完整双 Stream 执行过程。* **注**：本图有两处需要更正。其一，右半部分的 `shared_experts(x) + score apply（与 A2A combine 并行）`同属 A.1 的旧形态，当前已不成立。其二，左半部分标注的「routed experts 的 GroupedGEMM 在 compute stream 上与 dispatch A2A 并行」**在旧基线下也不成立**——grouped GEMM 消费的正是 dispatch 的输出，二者有数据依赖；该处真实可掩盖的只有 `_permute` 的索引构造（当前基线对应 `torchtitan/models/common/token_dispatcher.py:509-541`，见 §3.4）。

### A.3 这条旧论证今天还剩什么

机制部分（§2、§3）全部保留且经当前基线逐行重核；能力部分（"ACT 可以在同一 forward 内实现掩盖"）依然成立——`_functional_collectives.py:1142-1146` 的 unwrap 规则与 `ProcessGroupNCCL.cpp:820-827` 的 stream 级 block 都没变。**被推翻的只是"torchtitan 的 MoE 用它掩盖 shared experts"这一个实例**，以及建立在该实例上的"掩盖时长 ≈ 一个 shared-expert MLP"这一量级判断。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/15_torchtitan_ep_analysis|torchtitan 专家并行 EP]] —— dispatcher 四种后端的完整协议与容量语义；其 §5 给出了本页附录 A 同一条旧述的判定。
- [[02_engineering/02_train_frameworks/torchtitan/24_torchtitan_comm_optimizations_overlap_analysis|torchtitan 通信优化与重叠]] —— 框架里真正被用来做重叠的手段清单，ACT 只是其中最弱的一种。
- [[02_engineering/02_train_frameworks/torchtitan/27_torchtitan_graph_trainer_compiler_runtime_analysis|torchtitan graph trainer 编译与运行时]] —— §5 提到的 `ep_chunk_pass` / `ep_overlap_pass` 所在的图级调度栈。
- [[02_engineering/02_train_frameworks/torchtitan/12_torchtitan_tp_analysis|torchtitan 张量并行 TP]] —— §4.3 里那四处 `async_op=True` redistribute 的上下文。
- [[02_engineering/02_train_frameworks/torchtitan/14_torchtitan_pp_analysis|torchtitan 流水并行 PP]] —— `_PipelineStageBase` / 调度动作枚举在 torchtitan 侧的接入方式。
- [[02_engineering/02_train_frameworks/megatron-lm/15_megatron_pp_schedulers_analysis|Megatron-LM 流水调度器]] —— §4.6 对照里 `ScheduleNode` 与 combined 1F1B 的完整分析。
- [[02_engineering/02_train_frameworks/30_comm_compute_overlap_analysis|通信与计算重叠]] —— 跨框架的重叠手段与正确性前提总览。
