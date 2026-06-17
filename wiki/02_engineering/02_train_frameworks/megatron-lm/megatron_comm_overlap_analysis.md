# Megatron-Core 通信掩盖（Communication Overlap）技术详解

> 基于 Megatron-LM `dev` 分支代码分析，所有引用均来自实际代码路径与行号。

---

## 一、概述

Megatron-Core 在训练大规模模型时，通信（AllGather/ReduceScatter/All-to-All/P2P）往往占据 30%~50% 的 step 时间。框架通过在不同并行维度（TP/DP/PP/EP/CP）上引入**通信与计算重叠**，将通信时间从关键路径上隐藏。

### 维度对照表

| 维度 | 通信类型 | 掩盖方式 | 关键参数 | 默认状态 |
|---|---|---|---|---|
| **TP** | AllGather / ReduceScatter | Bulk 异步 + Pipelined Ring 替换 | `--tp-comm-overlap` | 默认 `False` |
| **DP（梯度）** | ReduceScatter / AllReduce | Bucket 分块异步，随反向传播触发 | `--overlap-grad-reduce` | 默认 `False` |
| **DP（参数）** | AllGather | 异步 prefetch，随前向计算触发 | `--overlap-param-gather` | 默认 `False` |
| **PP** | P2P send/recv | 1F1B 稳态异步重叠 | `--overlap-p2p-comm` | 默认 `False`（⚠️ 注意） |
| **EP** | A2A | 1F1B 跨 microbatch 重叠 + delay-wgrad | `--overlap-moe-expert-parallel-comm` | 默认 `False` |
| **CP** | AllGather / RS | Ring P2P / async AllGather 重叠 Attention | 随 `--context-parallel-size` 启用 | 自动开启 |

---

## 二、TP 通信掩盖（`--tp-comm-overlap`）

### 2.1 原理与参数入口

TP 通信掩盖仅在与 Sequence Parallel 结合时生效。配置定义在：

- **文件**：[`megatron/core/model_parallel_config.py`](Megatron-LM/megatron/core/model_parallel_config.py)
- **行号**：`196~276`

```python
# model_parallel_config.py:196-225
tp_comm_overlap: bool = False
"""If true, allows overlapping of Linear layer execution with tensor parallel communication"""

tp_comm_bulk_wgrad: bool = True
"""Controls All-Gather overlap with Bprop activation gradient GEMM."""

tp_comm_bulk_dgrad: bool = True
"""Controls Reduce-Scatter overlap with Bprop weight gradient GEMM."""

tp_comm_overlap_ag: bool = True
"""Controls All-Gather overlap with GEMM by pipelining."""

tp_comm_overlap_rs: bool = True
"""Controls Reduce-Scatter overlap with GEMM by pipelining."""
```

![alt text](image.png)

### 2.2 User Buffer 初始化

TP 掩盖依赖 Transformer Engine (TE) 预分配的静态 user buffer，在训练初始化时注册：

- **文件**：[`megatron/training/initialize.py`](Megatron-LM/megatron/training/initialize.py)
- **行号**：`206~276`

```python
# initialize.py:206-276
def _initialize_tp_communicators():
    """initializing the communicators with user buffers for high-performance
    tensor-model-parallel communication overlap"""
    te_module.base.initialize_ub(
        shape=input_shape,
        tp_size=args.tensor_model_parallel_size,
        use_fp8=(args.fp8 is not None),
        ub_cfgs=ub_cfgs,
        bootstrap_backend=args.tp_comm_bootstrap_backend,
    )
```

### 2.3 TE 层参数桥接

Megatron-Core 通过 `TELinear` 将上述配置传递给 TE：

- **文件**：[`megatron/core/extensions/transformer_engine.py`](Megatron-LM/megatron/core/extensions/transformer_engine.py)
- **行号**：`740~775`

```python
# transformer_engine.py:740-775
if self.config.tp_comm_overlap and parallel_mode != "duplicated":
    if is_te_min_version("1.5.0"):
        extra_kwargs["ub_overlap_ag"] = self.config.tp_comm_overlap_ag
        extra_kwargs["ub_overlap_rs"] = self.config.tp_comm_overlap_rs
    extra_kwargs["ub_bulk_wgrad"] = self.config.tp_comm_bulk_wgrad
    extra_kwargs["ub_bulk_dgrad"] = self.config.tp_comm_bulk_dgrad
    extra_kwargs["ub_name"] = tp_comm_buffer_name
```

### 2.4 Pipelined Overlap（有依赖）

**场景**：Linear GEMM 与通信存在**数据依赖**（如前向必须先 AllGather 完整输入，才能做 GEMM）。

**实现**：TE 将 AllGather / ReduceScatter 拆分为多步 ring-exchange，每步只传一个 chunk。GEMM 可以"流式"消费已到达的数据，无需等待整个张量通信完成。

```text
串行 (total=8):
      0 1 2 3 4 5 6 7 8
 计算 □ □ □ □ ■ ■ ■ ■ □  GEMM(4→8)
 通信 ■ ■ ■ ■ □ □ □ □ □  AG(0→4)
     无重叠

重叠 (total=5):
      0 1 2 3 4 5
 计算 □ ■ ■ ■ ■ □  GEMM chunk流式(1→5)
 通信 ■ ■ ■ ■ □ □  AG chunk流式(0→4)
      ├──3u──┤
```
> 通信chunk0完成(t=1)后GEMM即可启动消费,后续chunks流式到达,重叠3个时间单位。

**关键代码**：`ub_overlap_ag` / `ub_overlap_rs` 由 TE 底层实现，Megatron 仅传递开关。

### 2.5 Bulk Overlap（无依赖）

Bulk 掩盖处理的是**反向传播中通信与计算无数据依赖**的场景。

#### 2.5.1 `ub_bulk_wgrad`：AllGather 与 DGRAD GEMM 重叠

**物理场景**：RowParallelLinear（如 `proj`、`fc2`）反向时的数据依赖：

```
RowParallelLinear 反向:
  dgrad = grad_output @ W_shard       ← 纯本地! grad_output(上游) + W_shard(本地分片)
  wgrad = X_full.T @ grad_output      ← 需要完整X! 各rank只有X分片
```

关键洞察：**dgrad 不需要 X**，它只消费 `grad_output`（从上游反向传来）和本地 `W_shard`。但 **wgrad 需要完整 X**（`X_full`），而各 rank 只有 X 的 TP 分片，必须通过 AllGather 拼出完整 X。因此可以趁 AG(X) 异步传输的同时，顺手把 dgrad 算了。

**代码验证**：Legacy 层实现（`async_op=True` 的 AG 与 dgrad 并行）

- **文件**：[`megatron/core/tensor_parallel/layers.py`](Megatron-LM/megatron/core/tensor_parallel/layers.py)
- **行号**：`520~529`

```python
# layers.py:520-529
handle = dist_all_gather_func(
    all_gather_buffer, input, group=tp_group, async_op=True
)
# 立刻计算 dgrad —— 不依赖 all_gather_buffer！
grad_input = grad_output.matmul(weight)
# ... dgrad 算完后 ...
handle.wait()                            # 等 AG 完成,all_gather_buffer 此时有完整 X
# wgrad 现在拿着完整 X 计算 X_full^T @ grad_output
```

```text
数据依赖关系:
  dgrad ← grad_output + W_shard (纯本地,无依赖)
  wgrad ← X_full = AG(X_shard) (依赖通信结果)

串行 (total=11):
      0 1 2 3 4 5 6 7 8 9 10 11
 计算 □ □ □ □ ■ ■ ■ □ ■ ■ ■  ■ □   dgrad(4→7),wgrad(7→11)
 通信 ■ ■ ■ ■ □ □ □ □ □ □ □  □   AG(X)(0→4)
     无重叠

重叠 (total=8):
      0 1 2 3 4 5 6 7 8
 计算 ■ ■ ■ □ ■ ■ ■ ■ □   dgrad(0→3),wgrad(4→8)
 通信 ■ ■ ■ ■ □ □ □ □ □   AG(X)(0→4)
      ├──3u──┤    ↑ wgrad等AG结束(t=4)才启动
```
> dgrad 与 AG 并行(0→3),两者无依赖;AG 在 t=4 完成后 wgrad 才启动(4→8)。受益 = AG 时间被 dgrad 吸收 3u,total 从 11 降至 8。

#### 2.5.2 `ub_bulk_dgrad`：ReduceScatter 与 WGRAD GEMM 重叠

**物理场景**：ColumnParallelLinear（如 QKV、FC1）反向时的数据依赖：

```
ColumnParallelLinear 反向:
  dgrad = grad_output @ W_full.T     ← 需完整W,各rank做local GEMM后ReduceScatter汇总dX
  wgrad = X_shard.T @ grad_output    ← 纯本地! X_shard(本地) + grad_output(下游分片)
```

关键洞察：**wgrad 与 RS(dX) 操作不同内存对象**。dgrad 算完后得到局部 dX（各 rank 各自的片段），需要 RS 拼成完整 dX 分发回去。但 wgrad = `X_shard^T @ grad_output` 用的都是本地已存在的 tensor——与 RS 的输出/输入毫无关系，可以直接并行。

**代码验证**：

- **文件**：[`megatron/core/tensor_parallel/layers.py`](Megatron-LM/megatron/core/tensor_parallel/layers.py)
- **行号**：`552~558`

```python
# layers.py:552-558
# dgrad GEMM 完成后,异步启动 ReduceScatter(dX)
handle = dist_reduce_scatter_func(
    sub_grad_input, grad_input, group=tp_group, async_op=True
)
# 立刻计算 wgrad —— 只依赖 X_shard 和 grad_output,不碰 RS 的 buffer！
```

```text
数据依赖关系:
  RS(dX) ← dgrad结果 (需通信汇总)
  wgrad  ← X_shard + grad_output (纯本地,与RS无任何数据依赖)

串行 (total=11):
      0 1 2 3 4 5 6 7 8 9 10 11
 计算 ■ ■ ■ □ □ □ □ □ ■ ■ ■  ■ □   dgrad(0→3),wgrad(7→11)
 通信 □ □ □ ■ ■ ■ ■ □ □ □ □  □   RS(dX)(3→7)
     无重叠

重叠 (total=7):
      0 1 2 3 4 5 6 7
 计算 ■ ■ ■ □ ■ ■ ■ ■   dgrad(0→3),wgrad(3→7)
 通信 □ □ □ ■ ■ ■ ■ □   RS(dX)(3→7)
          ├──4u──┤
```
> RS(dX) 与 wgrad 操作不同内存对象,无数据依赖,可完全并行;4单位通信全部隐藏在 wgrad 计算中。

**收益来源**：

| 优化 | 依赖关系 | 重叠模式 | 收益 |
|------|---------|---------|------|
| `ub_bulk_wgrad` | dgrad ⇌ AG (无依赖) / wgrad → AG (有依赖) | dgrad∥AG, wgrad 等 AG 完 | AG 前段被 dgrad 吸收 |
| `ub_bulk_dgrad` | wgrad ⇌ RS (无依赖) | wgrad∥RS,完全并行 | RS 全部被 wgrad 隐藏 |

核心思想：利用反向传播中**部分 GEMM 不依赖通信结果**这一事实,将通信从关键路径上剥离。

---

## 三、DP 通信掩盖

### 3.1 梯度掩盖（`--overlap-grad-reduce`）

**背景**：Distributed Optimizer 使用 `reduce-scatter`（或普通 DDP 使用 `all-reduce`）同步梯度。若不重叠，所有层反向完成后批量通信，阻塞严重。

**实现**：DDP 包装器将梯度存储在连续 buffer 中，按反向传播顺序切分为 bucket。每个参数的 backward hook 调用 `register_grad_ready()`，当**一个 bucket 内所有梯度就绪**，立即在该 bucket 上启动**异步**通信。

- **文件**：[`megatron/core/distributed/param_and_grad_buffer.py`](Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py)
- **行号**：`709~731`

```python
# param_and_grad_buffer.py:709-731
def register_grad_ready(self, param, force_all_reduce=False):
    """Registers grads for the passed-in param to be 'ready' for grad sync."""
    if self.is_last_microbatch:
        if param not in self.per_param_grad_ready_counts:
            self.per_param_grad_ready_counts[param] = 0
        self.per_param_grad_ready_counts[param] += 1
        # If all params in bucket group have grads available, issue communication call.
        if not self.is_first_batch:
            if self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts:
                self.start_grad_sync(force_all_reduce=force_all_reduce)
```

异步通信启动：

- **文件**：[`megatron/core/distributed/param_and_grad_buffer.py`](Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py)
- **行号**：`587~604`

```python
# param_and_grad_buffer.py:587-604
with _coalescing_manager(communication_group, async_ops=async_op) as cm:
    for idx, bucket in enumerate(self.buckets):
        if self.ddp_config.use_distributed_optimizer and not force_all_reduce:
            grad_reduce_handle = dist_reduce_scatter_func(
                local_data_view, bucket.grad_data, op=reduce_op,
                group=communication_group, async_op=async_op
            )
```

```text
无重叠 (total=12):
      0 1 2 3 4 5 6 7 8 9 10 11 12
 计算 ■ ■ ■ ■ ■ ■ ■ ■ □ □ □  □  □   L4→1反向传播(0→8)
 通信 □ □ □ □ □ □ □ □ ■ ■ ■  ■  □   RS全部梯度(8→12)
     无重叠

重叠 (total=9):
      0 1 2 3 4 5 6 7 8 9
 计算 ■ ■ ■ ■ ■ ■ ■ ■ □ □   L4→1反向传播(0→8)
 通信 □ □ ■ ■ ■ ■ ■ ■ ■ □   RS bucket(2→9)
        ├─────6u─────┤
```
> bucket梯度就绪即异步通信,与后续层反向计算重叠;收益 ≈ sum(backward) + last_RS_tail。

**关键收益**：通信被隐藏在所有后续层的反向计算时间内，总时间 ≈ `sum(backward) + last_RS_tail`。

> [!update] 2026-06-16 · dev@232c478d4
> **dispatch 时顺手 drain 前驱 bucket 的 reduce-scatter**（#4940，`distributed_data_parallel.py:323`、`param_and_grad_buffer.py:206/567/749`）。仅在 `reduce_scatter_with_fp32_accumulation`（fp32 累加的梯度 RS）且单优化器实例时启用。该路径下，RS 的中间 all-to-all 输出张量会被 **pin 住直到 `.wait()`**；若不主动 drain，所有 bucket 的这些中间张量会一直存活到 step 末，显存峰值偏高。
> **机制**：`DistributedDataParallel.__init__` 把每个 bucket group 的 `previous_grad_reduce_bucket_group` 指向"反向中比它早一步 dispatch 的前驱"（反向按输出→输入顺序，故 `bucket_groups[i]` 的前驱是 `[i-1]`）。`start_grad_sync` 在为自己分配新 RS 缓冲**之前**，先 `finish_grad_sync()` 把前驱的 RS 等掉、释放其中间 buffer。新增 per-iteration 幂等标志 `grad_reduce_finished`（`param_and_grad_buffer.py:256`）：`finish_grad_sync` 第二次调用变 no-op，使"后继提前 drain 前驱"与"step 末 finalize 循环逐 bucket 收尾"不会重复 wait。**只省显存、不改通信量/重叠结构**；专家并行的 `expert_parallel_bucket_groups` 同样链接。

### 3.2 参数掩盖（`--overlap-param-gather`）

**原理**：前向计算某一层时，**下一层（或下一个 bucket）的参数 AllGather 已在后台完成**。

**实现**：通过 forward pre-hook 机制。当 Module 的前向开始时，hook 调用 `finish_param_sync()`：
1. `wait()` 当前 bucket 的异步 all-gather
2. 立即派发**下一个 bucket** 的异步 `start_param_sync()`

- **文件**：[`megatron/core/distributed/distributed_data_parallel.py`](Megatron-LM/megatron/core/distributed/distributed_data_parallel.py)
- **行号**：`353~357`, `388~422`

```python
# distributed_data_parallel.py:353-357
self.use_forward_hook = self.ddp_config.overlap_param_gather
if self.use_forward_hook:
    self.enable_forward_pre_hook()

# distributed_data_parallel.py:388-422
def _make_forward_pre_hook(self):
    def hook(module, *unused):
        for param in module.parameters(recurse=False):
            if param not in self.param_to_bucket_group:
                continue
            self.param_to_bucket_group[param].finish_param_sync(
                skip_next_bucket_dispatch=...
            )
    return hook
```

异步参数同步启动：

- **文件**：[`megatron/core/distributed/param_and_grad_buffer.py`](Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py)
- **行号**：`295~429`

```python
# param_and_grad_buffer.py:295-429
def start_param_sync(self, force_sync=False):
    async_op = self.ddp_config.overlap_param_gather and not force_sync
    with _coalescing_manager(..., async_ops=async_op) as cm:
        dist_all_gather_func(bucket.param_data, local_data_view, ... async_op=async_op)
```

```text
      0 1 2 3 4 5 6 7 8 9
 计算 ■ ■ ■ ■ ■ ■ ■ ■ ■ □   L1→3前向(0→9)
 通信 ■ ■ ■ ■ ■ ■ ■ ■ ■ □   Prefetch bucket(0→9)
      ├──── 全重叠 ────┤
```
> forward pre-hook: wait当前bucket → 立即派发下一bucket异步AG → 通信完全隐藏在前向计算中。

**关键收益**：AllGather 通信与前向计算流水化，消除参数同步的显式等待。

---

## 四、PP 通信掩盖（`--overlap-p2p-comm`）

### 4.1 重要修正：非默认开启

配置定义在：

- **文件**：[`megatron/core/model_parallel_config.py`](Megatron-LM/megatron/core/model_parallel_config.py)
- **行号**：`321~328`

```python
# model_parallel_config.py:321-328
overlap_p2p_comm: bool = False
"""When True some of the peer to peer communication for pipeline parallelism
   will overlap with computation. Must be False if batch_p2p_comm is true."""

batch_p2p_comm: bool = True
"""Use batch_isend_irecv instead of individual isend/irecv calls."""
```

**默认开启的是 `batch_p2p_comm=True`**（同步的 `batch_isend_irecv`）。**必须显式设置 `overlap_p2p_comm=True` 才能启用异步重叠**。

### 4.2 实现机制

当 `overlap_p2p_comm=True` 时，`P2PCommunicator._communicate()` 设置 `wait_on_reqs=False`，返回异步 request handles。

- **文件**：[`megatron/core/pipeline_parallel/p2p_communication.py`](Megatron-LM/megatron/core/pipeline_parallel/p2p_communication.py)
- **行号**：`275~340`

```python
# p2p_communication.py:275-340
def _communicate(self, ..., wait_on_reqs: bool = True):
    # wait_on_reqs=False 时返回异步 handles，由调用者自行 wait
```

更关键的是 `_p2p_ops()` 利用 even/odd rank 分组，将独立 send/recv 映射到不同 process group，实现真并发：

- **文件**：[`megatron/core/pipeline_parallel/p2p_communication.py`](Megatron-LM/megatron/core/pipeline_parallel/p2p_communication.py)
- **行号**：`55~128`

```python
# p2p_communication.py:55-128
if group.size() == 2 and torch.distributed.get_backend(group) != 'ucc':
    even_recv_odd_send_group = torch.distributed.group.WORLD
else:
    even_recv_odd_send_group = group

# even rank: send_next (pp_group) -> recv_prev (WORLD) -> send_prev (pp_group) -> recv_next (WORLD)
# odd rank:  recv_prev (WORLD) -> send_next (pp_group) -> recv_next (WORLD) -> send_prev (pp_group)
```

### 4.3 1F1B 稳态中的调度

在 `schedules.py` 的 1F1B 稳态中，通过 `pp_pre_forward()` / `pp_post_forward()` / `pp_pre_backward()` / `pp_post_backward()` 管理异步 P2P：

- **文件**：[`megatron/core/pipeline_parallel/schedules.py`](Megatron-LM/megatron/core/pipeline_parallel/schedules.py)

```text
      0 1 2 3 4 5 6
 计算 ■ ■ ■ □ ■ ■ ■   FWD(0→3),BWD(3→6)
 通信 ■ □ □ □ ■ □ □   recv/send(0→1),(3→4)
      ├1┤   ├1┤
```
> P2P通信在FWD/BWD启动时异步发起,随即与计算重叠;even/odd rank分组实现真并发。

---

## 五、EP 通信掩盖（MoE A2A）

### 5.1 1F1B A2A Overlap（`--overlap-moe-expert-parallel-comm`）

**核心机制**：利用 1F1B 调度，让**当前 microbatch 的 MoE A2A 通信**与**另一个 microbatch 的 Attention/MLP 计算**同时执行。

参数定义：

- **文件**：[`megatron/core/model_parallel_config.py`](Megatron-LM/megatron/core/model_parallel_config.py)
- **行号**：`279~284`

```python
# model_parallel_config.py:279-284
overlap_moe_expert_parallel_comm: bool = False
delay_wgrad_compute: bool = False
"""Delay the weight gradient computation to improve batch-level communication overlapping"""
```

### 5.2 无 PP 时的 1F1B 调度

- **文件**：[`megatron/core/pipeline_parallel/combined_1f1b.py`](Megatron-LM/megatron/core/pipeline_parallel/combined_1f1b.py)
- **行号**：`23~113`

```python
# combined_1f1b.py:23-113
def combined_1f1b_schedule_for_no_pipelining(...):
    # Phase 0: 1st microbatch forward (alone)
    # Phase 1~N-1: backward(N) + forward(N+1) overlapped
    # Phase N: last microbatch backward (alone)
    for i in range(num_microbatches - 1):
        output_tensor, num_tokens, _ = combined_forward_backward_step(
            ..., b_model=model, ...  # backward of N + forward of N+1
        )
```

### 5.3 层级别的精细调度

核心的双 Stream 调度定义在：

- **文件**：[`megatron/core/models/common/model_chunk_schedule_plan.py`](Megatron-LM/megatron/core/models/common/model_chunk_schedule_plan.py)
- **行号**：`195~259`

```python
# model_chunk_schedule_plan.py:195-259 (注释精确描述了重叠模式)
# comm_stream: combine_bwd | dispatch_fwd -> dispatch_bwd | combine_fwd
# comp_stream: attn_fwd    | mlp_bwd -> mlp_bwd_dw -> mlp_fwd | attn_bwd
```

```text
      0 1 2 3 4 5 6 7 8
 计算 ■ ■ ■ ■ ■ □ ■ ■ □
 通信 ■ ■ ■ □ □ ■ □ □ □
      ├─3u─┤     (t=5仅通信)

计算任务:
  attn_fwd(N+1)=[0→2)  mlp_bwd(N)=[2→4)  mlp_fwd(N+1)=[3→5)
  mlp_bwd_dw(N)=[4→5)  attn_bwd(N)=[6→8)

通信任务:
  combine_bwd(N)=[0→2)  dispatch_fwd(N+1)=[2→3)
  dispatch_bwd(N)+combine_fwd(N+1)=[5→6)

重叠区:
  t=0→2: attn_fwd(N+1) ∥ combine_bwd(N)     (2u)
  t=2→3: mlp_bwd(N)    ∥ dispatch_fwd(N+1)   (1u)
```
> MB N+1 前向的 attn_fwd 与 MB N 反向的 combine_bwd 完全重叠(2u),mlp_bwd 启动后与 dispatch_fwd 重叠(1u),t=5 处仅通信无计算可被 delay-wgrad 填补。

### 5.4 delay-wgrad-compute 的详细流水线

#### 5.4.1 标准反向 vs delay-wgrad（单层内部）

`TransformerLayerNode` 的 `backward_impl` 与 `backward_dw` 分离：

- **文件**：[`megatron/core/models/gpt/fine_grained_callables.py`](Megatron-LM/megatron/core/models/gpt/fine_grained_callables.py)
- **行号**：`314~348`

```python
# fine_grained_callables.py:314-348
class TransformerLayerNode(ScheduleNode):
    def backward_impl(self, outputs, output_grad):
        # 标准 backward：dX 和 dW 同时算完
        self.default_backward_func(outputs + self.before_detached, grads)
        if self.delay_wgrad_compute:
            self.output_grads = grads          # 保存梯度，不释放！
            self.delay_grads_release = True

    def backward_dw(self):
        # delay-wgrad：单独执行 dW GEMM
        for module in self.bwd_dw_callables:
            module.backward_dw()
        # 完成后才释放梯度内存
        if self.manual_release_grads:
            for tensor in self.output_grads:
                tensor.untyped_storage().resize_(0)
```

#### 5.4.2 三层对比图

```text
A: 无 delay-wgrad (total=7):
      0 1 2 3 4 5 6 7
 计算 ■ ■ ■ ■ □ □ □ □   mlp_bwd(dX)(0→2)+mlp_bwd_dw(dW)(2→4)
 通信 □ □ □ □ ■ ■ ■ □   A2A_wait(4→7)
     无重叠

B: 无 delay-wgrad + EP Overlap (total=7):
      0 1 2 3 4 5 6 7
 计算 ■ ■ ■ ■ □ ■ ■ ■   mlp_bwd(dX+dW)(0→4) attn_bwd(4→7)
 通信 □ □ □ □ ■ ■ ■ □   dispatch_bwd(4→7)
              ├3u┤

C: 有 delay-wgrad + EP Overlap (total=7):
      0 1 2 3 4 5 6 7
 计算 ■ ■ ■ ■ ■ ■ ■ □   mlp_bwd(dX)(0→2) attn_bwd(2→5) mlp_bwd_dw(5→7)
 通信 □ □ ■ ■ ■ □ □ □   dispatch_bwd(2→5)
          ├──3u──┤
```
> A: dX结束后dW仍阻塞,通信完全暴露(3u); B: dX+dW整体计算,通信与attn_bwd重叠(3u); C: dX分离后A2A提前到t=2,与attn_bwd大面积重叠(3u),dW推迟到t=5填充通信尾部空隙。

**图 C 的收益**：
- `mlp_bwd(dX)` 缩短到 2 单位，A2A (`dispatch_bwd`) 可以提前到 Time=2 开始
- `attn_bwd (2~5)` 与 `dispatch_bwd (2~5)` 大面积重叠
- `mlp_bwd_dw(dW)` 被推迟到 Time=5，填充了通信尾部的 SM 空隙

#### 5.4.3 多层全局流水线

```text
无 delay-wgrad (total=14):
      0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
 计算 ■ ■ □ □ ■ ■ □ □ ■ ■ □  □  ■  ■  □   L4(0→2) L3(4→6) L2(8→10) L1(12→14)
 通信 □ □ ■ ■ □ □ ■ ■ □ □ ■  ■  □  □  □   A2A(2→12,各层间歇)
     完全串行,无重叠

有 delay-wgrad (total=14):
      0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
 计算 ■ □ □ ■ ■ □ ■ ■ □ ■ ■  □  ■  ■  □
 通信 □ ■ ■ □ ■ ■ □ ■ ■ □ □  ■  □  □  □
             ├1┤   ├1┤   ├1┤
 计算: L4dX(0→1) L3dX+dW4(3→5) L2dX+dW3(6→8) L1dX+dW2(9→11) dW1(12→14)
 通信: A2A4(1→3) A2A3(4→6) A2A2(7→9) A2A1(10→12)
```
> 无delay时计算与通信完全串行交替;有delay后dX缩短使A2A提前,dW延后填充通信气泡,在t=4,7,10各产生1u重叠窗口。

### 5.5 dw 与 dx 的通信依赖关系（关键澄清）

**结论：dw 是纯本地计算，dx 才需要 A2A。**

Forward dispatch 已经把 token 路由到专家 rank 并缓存在本地。Backward 时：

- **dw**（专家权重梯度）：`dispatched_input.T @ grad_output`，全部 tensor 已在本地，**无需通信**
- **dx**（数据梯度）：专家本地算出的 `dx_expert` 是按专家排序的，必须通过 reverse A2A（`combine` kernel）恢复原始 token 顺序

代码验证：

- **文件**：[`megatron/core/transformer/moe/fused_a2a.py`](Megatron-LM/megatron/core/transformer/moe/fused_a2a.py)
- **行号**：`139~160`, `191~200`

```python
# fused_a2a.py:139-160
# FusedDispatch.backward = combine：把专家梯度送回原始位置
class FusedDispatch(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output, ...):
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(), handle, ...
        )

# fused_a2a.py:191-200
# FusedCombine.backward = dispatch：把上游梯度送到专家
class FusedCombine(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output, ...):
        grad_x, ... = buffer.dispatch(
            grad_output.contiguous(), handle=ctx.handle, ...
        )
```

### 5.6 DeepEP / HybridEP 后端

**收益点**：降低 A2A 的**绝对耗时**和**SM 占用**，属于"加速通信"而非"掩盖通信"。

- **文件**：[`megatron/core/transformer/moe/fused_a2a.py`](Megatron-LM/megatron/core/transformer/moe/fused_a2a.py)

```python
# fused_a2a.py:69-138
class FusedDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, token_indices, token_probs, num_experts, group,
                async_finish=False, allocate_on_comm_stream=False):
        # async_finish=True: 使用 EventOverlap 实现 stream 间异步
        buffer = get_buffer(group, get_hidden_bytes(x))
        buffer.dispatch(x, topk_idx=token_indices, topk_weights=token_probs,
                        async_finish=async_finish,
                        allocate_on_comm_stream=allocate_on_comm_stream)
```

- **文件**：[`megatron/core/transformer/moe/token_dispatcher.py`](Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py)
- **类**：`_DeepepManager`（line 1113）、`_HybridEPManager`（line 964）

#### 5.6.1 两级通信：为什么能降 A2A 绝对耗时

DeepEP/HybridEP 之所以能"加速通信"，核心是把单级的 GPU↔GPU A2A 拆成**两级**，并在节点间做**去冗余**——把流量从稀缺的 IB 转到富裕的 NVLink。`fused_dispatch` 的 `get_dispatch_layout` 给出**两套**变长计数，对应两级（`fused_a2a.py:135`）：

| 计数 | 粒度 | 阶段 | 链路 |
|------|------|------|------|
| `num_tokens_per_rdma_rank` | 每 **node** | `inter_dispatch` | RDMA / IB（瓶颈） |
| `num_tokens_per_rank` | 每 **GPU** | `intra_dispatch` | NVLink（富裕） |

buffer 也分两块：`num_rdma_bytes` + `num_nvl_bytes`（`get_buffer:62`）。**核心规则**：一个 token 不论在目标 node 上命中几个专家/几张卡，**跨 node 只发一次**（RDMA），落地后由该 node 内 NVLink 复制给目标卡。于是**跨节点流量 ∝ token 的"目标节点数" `|R(t)|`（≤ k），而非"目标专家数 k"**：

$$
\text{RDMA(跨节点)}(t)=|R(t)|\cdot M,\qquad
\boxed{\ \text{IB 加速比}=\dfrac{k/P}{\,1-(1-1/P)^k\,}\ }\quad(P=\text{node 数},\ M=H\times\text{bytes/elt})
$$

例：2 node、topk=4 → 跨节点 IB 流量约降到 1/2.13；topk=8 → ≈1/4。专家越多、topk 越大、目标越聚集在少数远端 node，省得越多（DeepSeek-V3 256 专家/topk8 即此场景）。**完整逐字节走查 + 两级公式推导见 [[megatron_ep_analysis]] §③.3**。

> **与"掩盖"的关系**：两级拆分让 A2A 的长极（跨节点 RDMA 段）与廉价的 NVLink 段在不同引擎上推进；配合 §5.7 的 `high_priority_a2a_comm_stream`（让 A2A kernel 优先抢 SM）与 `moe_hybridep_num_sms_preprocessing`（调元数据扫描 SM 数），可在 §5.1 的 1F1B overlap 之上进一步压短 A2A 暴露在关键路径上的尾延迟。即：**§5.6 降 A2A 绝对耗时（去冗余 + 两级）+ §5.1 把剩余 A2A 掩盖到计算后面**，二者叠加。

### 5.7 增量更新（ee3f1ff → dev@232c478d4）

> [!update] 2026-06-16 · dev@232c478d4 — 高优先级 A2A 通信流
> 新增 `high_priority_a2a_comm_stream`（默认 `False`，`transformer_config.py:686`，#4694）。combined-1F1B 的 `set_streams` 现接受 `high_priority` 形参（`pipeline_parallel/utils.py:336`）：打开后，§5.3 那条专跑 dispatch/combine A2A 的 `comm_stream` 以 **CUDA 高优先级**创建（`torch.cuda.Stream.priority_range()` 的 high 端）。目的：让 A2A 通信 kernel 优先抢 SM，减少被同设备的计算 kernel 卡住的尾延迟。`combined_1f1b.py` 的两个 schedule（no-pipelining / interleaved）都按 `config.high_priority_a2a_comm_stream` 透传。配套地，HybridEP 还新增 `moe_hybridep_num_sms_preprocessing`（默认 108，metadata scan kernel 的 SM 数，见 [[megatron_ep_analysis]] §③ 增量更新）。

> [!update] 2026-06-16 · dev@232c478d4 — A2A Overlap 支持 Megatron-FSDP
> EP A2A 重叠（combined-1F1B）现可与 **Megatron-FSDP** 共用（#3797，`combined_1f1b.py`、`model_chunk_schedule_plan.py:176`）。难点：细粒度 overlap schedule **绕过** `TransformerLayer.forward`（直接调子模块），从而绕过 FSDP 注册在 unit module 上的 forward/backward hook —— all-gather 出来的分片参数不会被正常释放。解法：`TransformerLayerSchedulePlan.set_fsdp_reshard_hooks` 给单个 schedule node 显式挂"释放参数"回调（最后一个前向 node 后 `post_forward_release_module`，最后一个反向 node（attn）后 `post_backward_release_module`）；`combined_forward_backward_step` 在反向前后调 `fsdp_wrapper.pre/post_backward()`，并在调度前 `_replace_param_with_raw_if_needed()`。仅 `optim_grads_params`（参数分片）策略需逐层 reshard hook。**限制**：交错式 PP（VPP>1）+ FSDP 暂不支持（显式 assert）。FSDP 内部分片/reshard 细节见 [[megatron_ddp_optimizer_analysis]]。

> [!update] 2026-06-16 · dev@232c478d4 — FSDP 双缓冲 wgrad 竞态修复
> 修复 FSDP 双缓冲（`fsdp_double_buffer`）下的 wgrad 竞态（#5222，`megatron_fsdp/param_and_grad_buffer.py:2727`）。原先在反向 hook 里**预先**调 `_enforce_double_buffer_limit` 腾 bucket；改为把该调用下沉进懒加载的 `main_grad_getter` —— 即**真正 fetch 到 incoming bucket 的那一刻**才腾退旧 bucket，使双缓冲 reduce-scatter 流水线的 bucket 生命周期与梯度写入精确对齐，消除"buffer 还在被 wgrad 写就被双缓冲分配器回收"的竞态。属 FSDP 内部正确性修复，与本页"通信重叠不改数值"前提一致；FSDP 缓冲机制详见 [[megatron_ddp_optimizer_analysis]]。

> [!update] 2026-06-16 · dev@232c478d4 — flex 后端配置补充
> §八配置速查表的 `moe_flex_dispatcher_backend` 现多一个取值 `"deepepv2"`（DeepEP v2 ElasticBuffer 后端，#4793）；`moe_deepep_num_sms` 默认从 `20` 改为 `None`（自适应）。详见 [[megatron_ep_analysis]] §③ 增量更新。

---

## 六、CP 通信掩盖（Context Parallel）

### 6.1 原理

CP 将序列切分到多 rank，Self-Attention 需要获取完整 KV。Megatron-Core 通过**异步 AllGather KV 与 Attention 计算重叠**来掩盖通信。

### 6.2 代码实现

- **文件**：[`megatron/core/transformer/dot_product_attention_context_parallel.py`](Megatron-LM/megatron/core/transformer/dot_product_attention_context_parallel.py)
- **行号**：`108~133`, `181~227`

```python
# dot_product_attention_context_parallel.py:108-133
class AllGatherComm:
    def all_gather(self, output_tensor, input_tensor):
        handle = torch.distributed.all_gather_into_tensor(
            output_tensor, input_tensor, group=self.group, async_op=True
        )
        self.handles.append(handle)

    def wait(self):
        for handle in self.handles:
            handle.wait()

# dot_product_attention_context_parallel.py:181-227 (Forward)
comm.all_gather(kv_buffer_copy[0], k_0)  # 异步启动第一轮 AG
for i in range(0, nheads_k, heads_k_stride):
    comm.wait()                           # 等上一轮 AG 完成
    kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer
    if i < nheads_k - heads_k_stride:
        comm.all_gather(kv_buffer_copy[0], send_k)  # 启动下一轮 AG
    # 同步做 Attention 计算（与下一轮 AG 重叠）
    out_i, probs_i = eager_attn_fwd(q_i, k_i, v_i, ...)
```

### 6.3 图示

```text
      0 1 2 3 4 5 6 7 8
 计算 □ □ ■ ■ ■ ■ ■ ■ □   Attn(head₀→₂)(2→8)
 通信 ■ ■ ■ ■ ■ ■ □ □ □   AG KV(head₀→₂)(0→6)
        ├─4u overlap─┤
```
> AG KV₁ 启动(2→4)与 Attn₀(2→4)重叠,AG KV₂(4→6)与 Attn₁(4→6)重叠;head级流水线使通信隐藏在注意力计算中。

---

## 七、综合收益量化

| 优化 | 隐藏/降低了哪部分时间 | 典型收益场景 |
|---|---|---|
| **TP Bulk** | AllGather / ReduceScatter 与独立 GEMM 并发 | 所有 SP + TP 训练 |
| **TP Pipelined** | AG/RS 与有依赖 GEMM 的串行等待 | 大 batch、长序列 |
| **DP Grad** | ReduceScatter 与后续反向计算 | 大 DP size |
| **DP Param** | AllGather 与后续前向计算 | Distributed Optimizer + 大模型 |
| **PP P2P** | Send/Recv 与 stage 内计算 | PP > 2，microbatch 多 |
| **EP 1F1B** | A2A 与跨 microbatch 的 Attn/MLP | MoE + EP > 1（A2A 占 30-40% 时收益最大） |
| **EP delay-wgrad** | dW GEMM 与 A2A 通信气泡 | 任何开启 EP overlap 的场景 |
| **CP** | KV AllGather 与 Self-Attention | CP > 1，长上下文 |

---

## 八、配置速查表

```python
# TP
args.tp_comm_overlap = True          # 总开关
args.tp_comm_bulk_wgrad = True       # AG 与 dgrad GEMM 的 Bulk 重叠（默认 True）
args.tp_comm_bulk_dgrad = True       # RS 与 wgrad GEMM 的 Bulk 重叠（默认 True）
args.tp_comm_overlap_ag = True       # Pipelined AG（默认 True）
args.tp_comm_overlap_rs = True       # Pipelined RS（默认 True）
args.tp_comm_overlap_cfg = None      # userbuffer YAML 配置

# DP
args.overlap_grad_reduce = True      # 梯度 bucket 异步 reduce-scatter
args.overlap_param_gather = True     # 参数异步 prefetch all-gather

# PP
args.overlap_p2p_comm = True         # ⚠️ 默认 False，需显式开启
args.batch_p2p_comm = False          # 必须与 overlap_p2p_comm 互斥

# EP
args.overlap_moe_expert_parallel_comm = True  # 1F1B A2A 重叠
args.delay_wgrad_compute = True               # 延迟 wgrad（需与上者同开）
args.moe_token_dispatcher_type = "flex"       # flex dispatcher
args.moe_flex_dispatcher_backend = "deepep"   # 或 "deepepv2" / "hybridep"（deepepv2 见 §5.7，#4793）

# CP
args.context_parallel_size = 2       # CP > 1 时自动启用 attention overlap
```

---

> **文档说明**：所有代码片段与文件路径均来自 Megatron-LM `dev` 分支（commit `3beeaa65b` 附近；§3.1 末、§5.7 的增量更新基准为 `dev@232c478d4`）的实际源码。若后续版本代码结构发生变化，请以实际代码为准。

## Related Pages

- [[megatron_tp_analysis]] · [[megatron_cp_analysis]] · [[megatron_ep_analysis]] · [[megatron_pp_schedulers_analysis]] · [[megatron_ddp_optimizer_analysis]]
- [[megatron_distributed_optimizer_analysis]] · [[megatron_parallelism_orchestration_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
