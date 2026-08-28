---
title: "MoE 训练优化技术全景报告 — 基于 Megatron-LM 源码分析"
---

# MoE 训练优化技术全景报告 — 基于 Megatron-LM 源码分析

**Date**: 2026-05-12
**Status**: Complete
**Scope**: Megatron-LM mcore 源码级分析，覆盖 7 维优化技术 + 选型决策

> **源码基线**：`NVIDIA/Megatron-LM@71092579522a12522d9f323ae180c9825d01928a`（`dev`，2026-08-27）
> **重定基线**：2026-08-28 由 `232c478d43ce2f8b4c8db3507d3623fa82f55823`（2026-06-16）推进，跨 280 个提交；本页全部 `path:line` 形式的引用已在新基线下逐条重核;**代码块内被点名的符号与不带行号的裸路径不在该次扫描口径内**,已知漏网处已于 2026-08-28 单独更正。
> **叙事顺序**：本页按五拍组织——背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势。
> **最近更新**：2026-08-28。按五拍重排章节顺序；机制正文与既有引用未改。

---

## 1. 背景：MoE 训练的三座大山

### 1.1 MoE 架构核心

MoE（Mixture of Experts）用稀疏激活的专家网络替代标准 Transformer 的稠密 FFN：

```
Hidden State [S, B, H]
    │
    ▼
Router（TopK selection）
    │
    ├──→ Expert 0 ──→ weighted output
    ├──→ Expert 2 ──→ weighted output  (sparse: only K of N experts activated)
    └──→ Expert 7 ──→ weighted output
    │
    ▼
Combined Output [S, B, H]
```

### 1.2 训练的三座大山

| 山 | 根因 | 量级 |
|----|------|------|
| **显存墙** | N 个 expert 各自完整复制 FFN 参数 → 总参数量膨胀 Nx；每个 expert 需独立保存激活用于 backward | 671B MoE 参数 + 优化器状态 + 激活 → 单卡 >1TB |
| **通信墙** | Token 需从 router 所在 rank 发送到 expert 所在 rank（All-to-All）；TP/PP/DP 各有独立通信组 | EP All-to-All 单步通信量 = S×B×H×2 bytes |
| **计算效率墙** | 稀疏路由导致 token 分布不均 → expert 负载不均衡；MoE 层大量小 GEMM（per-expert）→ GPU 利用率低 | Load imbalance >20% 常见 |

### 1.3 Megatron-LM 的应对体系

Megatron-LM 通过 7 个维度的协同优化应对这些挑战。每个维度解决一个特定瓶颈，但彼此正交可叠加：

```
                  ┌──────────────────┐
                  │  多维并行策略     │ ← 通信墙（空间维度）
                  │  TP/PP/EP/CP     │
                  └────────┬─────────┘
                           │
  ┌────────────────────────┼────────────────────────┐
  │                        │                        │
  ▼                        ▼                        ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ 分布式优化器  │  │  重计算技术   │  │ 低精度训练    │
│ (状态分片)   │  │ (激活换计算) │  │ (FP8/FP4)    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────┬───────┴─────────────────┘
                 │
                 ▼
       ┌──────────────────┐     ┌──────────────────┐
       │  通信-计算 Overlap│────→│   内存优化        │
       │  (隐藏延迟)       │     │   (Paged/Offload) │
       └──────────────────┘     └──────────────────┘
                 │
                 ▼
       ┌──────────────────┐
       │  融合算子优化      │
       │  (kernel级加速)   │
       └──────────────────┘
```

### 1.4 报告约定

- 源码引用格式：`megatron/core/<path>:<line>`
- 每个技术按"优化点 → 为什么有效 → 何时适用 → 关键实现"组织
- 括号中的 `[[wiki page]]` 链接到详细分析页面

---

## 2. 为什么这么设计：优化粒度定在"缓冲区 / 通信组"，而不是"逐参数 / 逐并行轴"

§1 列出了三座大山，但真正决定 Megatron 形态的不是"用不用某个优化"，而是**把每个优化挂在多粗的粒度上**。源码在四处给出了理由，其中一条是被上游判定为不正确、随后改掉的做法。

**① 参数与梯度先拼成一整块连续缓冲、再切桶——offload、量化、NCCL 注册全部以 buffer 为单位。**
`_ParamAndGradBuffer` 的 docstring 一句话定型：「Groups parameters and gradients into a **contiguous buffer**, and then breaks the buffer into buckets with roughly `bucket_size` parameters each」（`megatron/core/distributed/param_and_grad_buffer.py:1066-1071`）。三项下游能力全部建立在这块整体分配上，而不是单个 `nn.Parameter`：
- **CPU 备份 / 卸载**：`disable_grad_buffers_cpu_backup` / `disable_param_buffers_cpu_backup` 的语义是「allocate DDP gradient/parameter **buffers** in a torch_memory_saver region without CPU backup」——粒度写在字段名里（`megatron/core/distributed/distributed_data_parallel_config.py:33-38`）。
- **NCCL user buffer 注册**：`nccl_ub` 的 docstring 是「allocate and register NCCL userbuffer for **param and grad buffer**」，并附一张按通信域列出的 SM 占用表（`megatron/core/distributed/distributed_data_parallel_config.py:130-148`）。注册对象是一整块显存，逐参数根本无从注册。
- **通信效率**：`bucket_size` 默认 `max(40000000, 1000000 * dp_size)`，理由明写「larger DP sizes need larger buckets to ensure collectives do not become latency-bound」（`:60-63`）；`pad_buckets_for_high_nccl_busbw` 再把桶补到 `2^16` 的倍数，理由是 NCCL「message size … apparently needs to be divisible by a power of 2 for high busbw」（`:70-76`）。
→ 判据是**通信与分配的最小有效单位**：集合通信在小消息上被延迟支配，而 NCCL 注册与 memory-saver 区域只能对整块显存生效。逐参数做 offload / 量化在这两条上都拿不到收益，还会把通信 kernel 数量放大（同一条约束在 [[20_megatron_comm_overlap_analysis]] §2② 有独立记录）。

**② Token dispatcher 抽象成 manager，是为了让"分发逻辑"与"并行策略"解耦。**
`MoEFlexTokenDispatcher` 的类 docstring 直接写目的：「A flexible token dispatcher that **abstracts the underlying tensor and expert parallelism**. It uses **a single communication group over all TP and EP ranks**, making the dispatch logic **independent of the specific parallelism strategy**」（`megatron/core/transformer/moe/token_dispatcher.py:1858-1862`）。契约由抽象基类 `_DispatchManager` 固定成一份统一的 routing_map：「handles token dispatching according to the routing_map of format `[num_local_tokens, world_size, num_instances]`」，且说明 `num_instances`「can be the number of local experts, or the size of sub_group」（`:989-999`）。四个后端——`_HybridEPManager`（`:1060`）、`_DeepepManager`（`:1283`）、`_DeepepV2Manager`（`:1529`）、`_NCCLEPManager`（`:1637`）——都只实现这一份契约，选择发生在 `MoEFlexTokenDispatcher.__init__` 的一串 `elif` 里（`:1884` 起）。
→ 判据：EP 的通信后端换代极快（§12 记录的一个月里就多出两个，重核后又多出第四个），而路由逻辑本身不变。把"TP×EP 合成单一通信组 + 一份 routing_map 契约"钉死，换后端就只是换一个 manager，不必动路由、不必动 MoE 层。

**③ §1.3 说 7 个维度"正交可叠加"——机制上成立，配置上则被写成硬 assert，而不是留给用户试。**
最典型的是 Paged Stash：`moe_paged_stash` 一旦打开，`cpu_offloading` 必须关（「moe_paged_stash cannot be enabled with cpu_offloading.」）、`moe_expert_rank_capacity_factor` 必须设（「there is no need to use paged stashing without it.」）、`offload_modules` 里不得再出现 `expert_fc1` / `moe_act` / `fused_group_mlp`（「paged stash covers those activations」）——三条连着写在 `megatron/core/transformer/transformer_config.py:2547-2561`。
→ 判据：显存维度的两套机制会**争抢同一份激活的所有权**。与其让两者同时接管、再在运行期表现为难查的双份拷贝，不如在构造期把冲突挑明。§11 汇总了这一类交叉约束。

**④ 一处被上游判定为"不正确"并改掉的做法：aux/z-loss 的 `× tp_cp_group.size()` 预乘。**
§12 的 `[!contradiction]` 已经记下这次反转；它的判据写在新代码的注释里：「Use the reduced count directly: with THD padding or dynamic CP, valid token counts can differ by rank/group, so **`local_num_tokens * group_size` is not generally correct**」（`megatron/core/transformer/moe/router.py:598-604`）。
→ 判据：一旦引入 THD packing 与动态 CP，"每个 rank 的有效 token 数相同"这个隐含前提就不成立；凡是靠 `group_size` 反推全局量的写法都要改成真正的组内 `all_reduce`。同一条判据也解释了 HybridEP 变长补齐为什么被收紧（见 §12 末的补充勘误）。

> [!note] 推断
> 上面四条的**理由**都能落到 docstring、注释或 assert 文案；**"框架把优化粒度统一定在缓冲区与通信组这一层"这条归纳由本页承担**，源码没有一处这样陈述。依据是三组可核验事实：DDP 侧的分配 / 注册 / 通信全部以 buffer 与 bucket 为单位（`megatron/core/distributed/param_and_grad_buffer.py:1066-1071`、`megatron/core/distributed/distributed_data_parallel_config.py:33-38`、`:60-76`、`:130-148`）；MoE 侧的分发以"TP×EP 单一通信组 + 统一 routing_map"为单位（`megatron/core/transformer/moe/token_dispatcher.py:989-999`、`:1858-1862`）；维度之间的冲突以构造期 assert 表达（`megatron/core/transformer/transformer_config.py:2547-2561`）。要引用这条判断，请回到这些 locator，不要引用本段推断。

---

## 3. 多维并行策略

### 3.1 并行维度全景

| 维度 | 切分对象 | 通信组 | 通信原语 | 单层通信量 | 正交性 |
|------|---------|--------|---------|-----------|--------|
| **DP** | Batch | `get_data_parallel_group()` | AllReduce / ReduceScatter+AllGather | 2P | 与所有并行正交 |
| **TP** | 权重矩阵（列/行切分） | `get_tensor_model_parallel_group()` | AllReduce / ReduceScatter | 2×B×S×H per layer | 与 PP/EP 正交 |
| **PP** | 层（按 depth 切分） | `get_pipeline_model_parallel_group()` | P2P Send/Recv | 2×B×S×H per boundary | 与 TP/DP/EP 正交 |
| **EP** | Experts（按 expert 切分） | `get_expert_model_parallel_group()` | All-to-All | S×B×H×2K/ep_size | 可与 TP 组合（ETP） |
| **CP** | 序列长度 | `get_context_parallel_group()` | AllGather（KV） | ~H×S/cp_size per head | 与 TP/PP/EP 正交 |

### 3.1.1 通信组层级关系

```
                    Global World (所有 GPU)
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     DP Group          TP Group          PP Group
  (相同TP/PP位置)    (同PP stage)     (跨层流水线)
          │                │
          ├── CP Group     ├── ETP Group (expert内TP)
          │   (相同DP组内) │
          │                │
          ├── EP DP Group  ├── TP_EP Group
          │   (expert DP)  │   (TP×EP混合)
          │
          └── Intra/Inter Instance
              (HSDP分层)
```

**通信组获取**（`megatron/core/parallel_state.py`）：
```python
# TP: 同一个 PP stage 内的 rank
get_tensor_model_parallel_group()          # → tp_group

# PP: 同一个 TP rank 跨 PP stage
get_pipeline_model_parallel_group()        # → pp_group

# DP: 所有相同 TP/PP/EP 位置的 rank
get_data_parallel_group(with_context_parallel=True)  # → dp_group

# CP: DP 组内切分序列维度
get_context_parallel_group()               # → cp_group

# EP: 所有 expert 分布到的 rank
get_expert_model_parallel_group()          # → ep_group

# ETP: expert 内的 TP rank
get_expert_tensor_parallel_group()         # → etp_group

# EP DP: expert 的 data-parallel rank
get_expert_data_parallel_group()           # → ep_dp_group
```

### 3.2 Tensor Parallelism（TP）

**优化点**：将大矩阵乘法按列/行切分到多个 GPU，减少单卡显存和计算量。

**通信组**：`get_tensor_model_parallel_group()` — 同一 PP stage 内的 `tp_size` 个 GPU。
**通信原语**（`megatron/core/tensor_parallel/mappings.py:22-198`）：
- `_reduce`（`:22`）：等价于 `all_reduce(input, group=tp_group)`
- `_gather_along_first_dim`（`:118`）：等价于 `all_gather_into_tensor(output, input, group=tp_group)`
- `_reduce_scatter_along_first_dim`（`:159`）：等价于 `reduce_scatter_tensor(output, input, group=tp_group)`

**Megatron 实现**（`megatron/core/tensor_parallel/layers.py`）：
- `ColumnParallelLinear`：权重 `[H, H/tp]` 按列切分（输入全量，输出部分和）
- `RowParallelLinear`：权重 `[H/tp, H]` 按行切分（输入部分，输出全量）

### 3.2.1 TP 正反向通信详细分析

设 hidden_size = H, tp_size = T, batch × seq = N：

```
═══════════════════════════════════════════════════════════════════
                ColumnParallelLinear (如 Attention QKV, FFN FC1)
═══════════════════════════════════════════════════════════════════
权重: W [H, H/T]  ← 每个 rank 持有 1/T 列
输入: X [N, H]    ← 所有 rank 相同（replicated）

Forward:
  Y_partial = X @ W    ← 本地计算 [N, H/T], 无通信
  Y = AllGather(Y_partial, group=tp_group)  ← [N, H/T] × T = [N, H]
  通信量: N × H × (T-1)/T bytes (output gather)

Backward:
  ∂L/∂Y_partial = ∂L/∂Y.chunk(T)[rank]  ← [N, H/T], 无通信
  ∂L/∂W = X.T @ ∂L/∂Y_partial          ← [H, H/T], 本地
  ∂L/∂X_partial = ∂L/∂Y_partial @ W.T   ← [N, H], shape [N, H] 但内容不完整
  ∂L/∂X = AllReduce(∂L/∂X_partial, group=tp_group) ← [N, H]
  通信量: N × H × 2(T-1)/T bytes (grad input allreduce)

═══════════════════════════════════════════════════════════════════
                RowParallelLinear (如 Attention Output, FFN FC2)
═══════════════════════════════════════════════════════════════════
权重: W [H/T, H]  ← 每个 rank 持有 1/T 行
输入: X [N, H/T]  ← 按列分片（从 ColParallel 或 SP 继承）

Forward:
  Y_partial = X @ W        ← 本地计算 [N, H], 不完整
  Y = ReduceScatter(Y_partial, group=tp_group)  ← [N, H/T]
  通信量: N × H × (T-1)/T bytes (partial sum reduction)

Backward:
  ∂L/∂X_partial = ∂L/∂Y @ W.T  ← [N, H/T], 本地
  ∂L/∂W_partial = ∂L/∂Y.T @ X   ← [H/T, H], 本地
  ∂L/∂W = AllReduce(∂L/∂W_partial, group=tp_group) ← [H/T, H]
  通信量: H² × 2(T-1)/T bytes (grad weight allreduce) ← 注意: 与 N 无关

═══════════════════════════════════════════════════════════════════
                TP Layer 总通信量（一层 Attention + FFN）
═══════════════════════════════════════════════════════════════════
Forward:  2 × N × H × (T-1)/T  bytes  (2 次 AllGather/ReduceScatter)
Backward: 2 × N × H × (T-1)/T + 4 × H² × (T-1)/T  bytes
          ↑ input grad        ↑ weight grad × 4 matrices
```

**为什么有效**：通信量中 `N×H` 项随 batch/seq 增长，但 `H²` 项是固定的。大 batch 下，通信/计算比趋近于 `(T-1)/T`，近线性加速。

**何时适用**：
- ✓ Hidden size ≥ 4096（通信/计算比合理）
- ✓ 单机多卡 NVLink 连接（TP 通信对带宽敏感）
- ✗ 跨节点（IB/RoCE）带宽不足时通信成为瓶颈
- ✗ TP > 8（通信开销随 T 增长）

### 3.2.2 TP 的 autograd Function 设计

ColumnParallelLinear 和 RowParallelLinear 通过自定义 `torch.autograd.Function` 将通信注入到计算图中，而非在 forward 中直接调用 AllReduce：

```python
# ColumnParallelLinear: f = identity(fwd), AllReduce(bwd)
class F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x  # 前向无需通信
    @staticmethod
    def backward(ctx, grad_output):
        torch.distributed.all_reduce(grad_output, group=tp_group)
        return grad_output

# RowParallelLinear: g = AllReduce(fwd), identity(bwd)
class G(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        torch.distributed.all_reduce(x, group=tp_group)
        return x
    @staticmethod
    def backward(ctx, grad_output): return grad_output  # 反向无需通信
```

**为什么必须用 autograd.Function**：直接调用 `torch.distributed.all_reduce` 不会被 autograd 追踪，导致反向传播图中缺少通信操作，各 TP rank 梯度不一致，模型发散。

### 3.2.3 SP 与 TP 的边界

| 维度 | SP（Sequence Parallelism） | TP（Tensor Parallelism） |
|------|--------------------------|-------------------------|
| **切分模块** | 仅 LayerNorm、Dropout 等 seq-dim element-wise 算子 | 线性层的权重/激活（hidden dim） |
| **通信原语** | `ReduceScatter` + `AllGather` | `AllReduce` / `ReduceScatter` + `AllGather` |
| **通信位置** | LN 前后的 RS/AG | 线性层前后的 f/g autograd Function |

**为什么 TP 必开 SP**：LayerNorm 和 Dropout 是 element-wise 算子，与 `hidden` 维度无关。TP 切分的是 `hidden` 维度，因此这些算子的输入/输出激活在每个 TP rank 上仍然是完整序列长度的副本，显存无法下降。SP 在 LN 前后插入 `ReduceScatter`/`AllGather`，将序列维度切分，使每个 rank 只持有 `seq/tp` 的激活。**通信量不变，但显存峰值下降**。

**无 SP 时**：`ColParallel(out) → AllReduce → LayerNorm(full seq) → Dropout(full seq) → RowParallel(in)`
**有 SP 时**：`ColParallel(out) → ReduceScatter → LayerNorm(seq/tp) → Dropout(seq/tp) → AllGather → RowParallel(in)`

### 3.3 Pipeline Parallelism（PP）

**优化点**：将模型按层切分到不同 GPU，每卡只存一部分层。

**通信组**：`get_pipeline_model_parallel_group()` — 同一 TP rank 跨 PP stage 的 `pp_size` 个 GPU。
**通信原语**：`torch.distributed.send` / `torch.distributed.recv`（P2P 点对点）。
**通信实现**（`megatron/core/pipeline_parallel/p2p_communication.py`）：支持 blocking / non-blocking send/recv，future tensor 模式用于 overlap。

**Megatron 实现**（`megatron/core/pipeline_parallel/schedules.py`）：
- **1F1B**（one-forward-one-backward）：标准流水线调度，warm-up forward → steady 1F1B → cool-down backward
- **Interleaved 1F1B**：每个 rank 有多个 virtual stage，进一步减少 bubble 时间

**PP 正反向通信**：
```
Forward:  Stage i → send(hidden_i) → Stage i+1 → recv(hidden_i+1)
Backward: Stage i+1 → send(grad_i+1) → Stage i → recv(grad_i)
```
每 micro-batch 在每个 PP boundary 传递 `B×S×H` 个元素（≈ B×S×H×2 bytes for BF16）。

**Bubble 时间**：
```
1F1B bubble = (pp_size - 1) × t_fwd / num_microbatches
Interleaved bubble ≈ (pp_size - 1) × t_fwd / (num_microbatches × vpp)
```

**为什么有效**：PP 通信量极小（只在 stage 边界传递 hidden states），跨节点友好。与 TP 配合 = "TP 处理单层内的超大规模矩阵，PP 处理层间的显存分摊"。

**何时适用**：
- ✓ 模型深度 ≥ 40 层（自然的 PP 切分粒度）
- ✓ 跨节点场景（通信量 ~ 2×B×S×H / boundary，远小于 TP）
- ✗ Microbatch 太少（bubble 占比过高）
- ✗ 首尾层计算量差异大（负载不均衡）

### 3.4 Expert Parallelism（EP）

**优化点**：将 MoE 的不同 Expert 分布到不同 GPU，每个 rank 只负责部分 expert 的计算。

**通信组**：`get_expert_model_parallel_group()` — 所有 expert 分布到的 `ep_size` 个 GPU。
**通信组辅助**：`get_expert_tensor_parallel_group()`（ETP Group），`get_expert_data_parallel_group()`（EP DP Group）。

**通信原语**（`megatron/core/transformer/moe/token_dispatcher.py`）：
- `all_to_all`（`:16`）：基于 NCCL 的 All-to-All 集群通信
- `gather_from_sequence_parallel_region`（`:17`）：AllGather（TP 维度）
- `reduce_scatter_to_sequence_parallel_region`（`:18`）：ReduceScatter（TP 维度）

**Megatron 实现**（`megatron/core/transformer/moe/token_dispatcher.py`）：
- **AlltoAll Dispatcher**：最常用的实现，token 通过 All-to-All 发送到目标 expert
- **AllGather Dispatcher**：用于 TP/EP 混合场景
- **Flex Dispatcher**：使用 DeepEP/HybridEP 融合通信

### 3.4.1 EP 正反向通信详细分析

设 S = seq_len, B = batch, H = hidden_size, K = topk, E = ep_size：

```
═══════════════════════════════════════════════════════════════════
                AlltoAll Dispatcher 正反向通信
═══════════════════════════════════════════════════════════════════

Forward (Dispatch Phase):
  dispatch_preprocess():
    routing_map [S*B/TP, num_experts]  ← 只计算元数据, 无通信
    tokens_per_expert, input/output splits ← CPU metadata

  All-to-All Dispatch:
    每个 rank 发送 {S×B×H×K/E} 元素 (BF16)
    每个 rank 接收 {S×B×H×K/E} 元素 (BF16)
    通信量: S×B×H×K × 2(E-1)/E² bytes per rank
    ← 注意: All-to-All 的每 rank 通信量为总数据量 × (E-1)/E²

  dispatch_postprocess():
    AllGather(TP) if tp > 1: {S×B×H×K/E} × (T-1)/T bytes
    Permute tokens by local expert id ← 本地操作, 无通信

Forward (Expert Compute):
  Expert MLP forward ← 纯计算, 无通信

Forward (Combine Phase):
  combine_preprocess():
    Permute output back ← 本地操作

  All-to-All Combine:
    每个 rank 发送 {S×B×H×K/E} 元素
    每个 rank 接收 {S×B×H×K/E} 元素
    通信量: S×B×H×K × 2(E-1)/E² bytes per rank

  combine_postprocess():
    ReduceScatter(TP) if tp > 1: {S×B×H×K/E} × (T-1)/T bytes

═══════════════════════════════════════════════════════════════════
                EP Layer 总通信量（1 个 MoE Layer）
═══════════════════════════════════════════════════════════════════
EP All-to-All:  2 × S×B×H×K × 2(E-1)/E² bytes  (dispatch + combine)
TP within EP:   + S×B×H×K/E × (T-1)/T × 2 bytes (AllGather + ReduceScatter, 仅 TP>1)

总 ≈ 4×S×B×H×K × (E-1)/E² bytes (主导项)
```

**为什么有效**：EP 是解决 MoE 参数膨胀的最直接手段——N 个 expert 分布在 `ep_size` 个 GPU 上，每个 GPU 只存 `N/ep_size` 个 expert。显存随 EP 规模线性递减。

**何时适用**：
- ✓ Expert 数量 ≥ 8（有足够的划分粒度）
- ✓ 单机多卡（NVLink 带宽支撑 All-to-All）
- ✗ Expert 数量少且没有足够的 token 填充（All-to-All 延迟 > 收益）
- ✗ 可与 TP 组合为 ETP（Expert Tensor Parallelism）

### 3.4.2 MoE Router 与负载均衡

**Top-K Router + Aux Loss**：
- Aux loss 惩罚路由不均衡，鼓励 router 将 token 均匀分配
- 梯度只更新 **router 的权重矩阵**（gate 网络），不影响 expert 参数

**Dynamic Expert Bias（aux-loss-free）**：
- 为每个 expert 维护可学习的 bias 项
- 根据 expert 负载动态更新：过载的 expert bias↓，欠载的 expert bias↑
- 推理时可直接丢弃 bias（或固定为最终值），避免 aux loss 对训练目标的干扰

**Group-Limited Routing**（`moe_router_num_groups` + `moe_router_group_topk`）：
- 将 expert 分组，每个 token 先选 top-k 组，再在组内选 expert
- 对应 DeepSeek-V2/V3 的 Device-Limited Routing — 限制 token 只访问部分设备上的 expert，减少 All-to-All 通信跳数

### 3.4.3 MoE Parallel Folding

传统实现中 attention 和 MoE 共享并行拓扑，因此 EP 受限于 DP（$EP \leq DP$）。**MoE Parallel Folding** 将 CP 和 EP 维度"折叠"组合，使 MoE 层拥有独立于 attention 层的并行配置：

```
Attention: TP=4,  CP=1,  DP=8  → 小规模并行
MoE:       ETP=1, EP=32, EDP=1  → EP 可远大于 attention DP
```

**效果**：EP 可以远大于 attention 的 DP，支持更多 expert 和更大 EP 规模，打破 $EP \leq DP$ 的限制。

### 3.5 Context Parallelism（CP）

**优化点**：将超长序列按长度切分到多个 GPU，每 rank 只处理一部分 token 的 Attention。

**通信组**：`get_context_parallel_group()` — DP 组内切分序列维度的 `cp_size` 个 GPU。
**通信原语**：`all_gather_into_tensor` — 异步 AllGather KV cache。
**通信实现**（`megatron/core/transformer/dot_product_attention_context_parallel.py:108-132`）：
- `AllGatherComm`（`:108`）：封装 async AllGather + wait 的双缓冲调度器

**Megatron 实现**（`megatron/core/transformer/dot_product_attention_context_parallel.py:150`）：
- **Ring Attention** 变体：通过 AllGather 流水线轮转 KV cache
- **双缓冲**：KV buffer 在 `kv_buffer` 和 `kv_buffer_copy` 间交替，AllGather 与计算重叠
- **Zigzag Mask**（`:135-147`）：将 attention mask 按 zigzag 模式重排

### 3.5.1 CP 正反向通信详细分析

设 S = seq_len, H = hidden_size, C = cp_size, nheads = h, heads_per_iter = h_k：

```
═══════════════════════════════════════════════════════════════════
                CP Ring Attention 正反向通信
═══════════════════════════════════════════════════════════════════

Forward:
  for i in range(0, num_heads_kv, heads_per_iter):
    Wait(prev AllGather)                          ← 等待上一轮 KV allgather 完成
    Swap KV buffers (double buffering)            ← kv_buffer ↔ kv_buffer_copy
    AllGather(next KV chunk, async=True)           ← 异步启动下一轮, 与 compute 重叠
      通信量: S/C × H × 2(C-1)/C bytes per allgather
    Compute Attention(Q_i, K_gathered, V_gathered) ← 本地计算

  Total AllGather calls: n_heads / heads_per_iter
  Total CP 通信量: (n_heads / h_k) × S/C × H × 2(C-1)/C bytes
  ← 注意: 通信量与注意力头数线性相关

Backward:
  对称: 同样需要 AllGather KV for grad computation
  总通信量 ≈ 与 forward 相同量级 (每个 head 需要完整 KV)

═══════════════════════════════════════════════════════════════════
                CP vs. 无 CP 显存对比
═══════════════════════════════════════════════════════════════════
无 CP: Attention 激活 [B, h, S, S]  ← 每个 rank 存完整 S²
有 CP: Attention 激活 [B, h, S/C, S] ← 每 rank 只存 1/C 的 query 维度
显存节省: ~1/C (但 KV allgather 需要 buffer 空间)
```

**为什么有效**：长序列下 Attention 的显存为 O(S²)，CP 将序列切分为 S/cp_size，每个 rank 的注意力显存为原始的 1/cp_size 级别。

**何时适用**：
- ✓ 序列长度 ≥ 32K（Attention 成为显存瓶颈）
- ✓ 与 TP 正交（TP 切 hidden/heads，CP 切 sequence）
- ✗ 短序列（CP 通信 overhead > 收益）
- ✗ CP size 过大（AllGather 延迟掩盖计算）

### 3.5.2 Dynamic Context Parallelism

Megatron-LM 引入了 Dynamic CP 优化 packed sequence 训练：

**问题**：标准 CP 将序列均匀切分。但在 packed sequence（SFT 数据）中，每个 sample 长度差异大，固定切分导致某些 CP rank 的有效 token 远少于其他 rank —— straggler effect。

**Dynamic CP 原理**：
- 训练开始前，从 DP×CP 的 rank pool 中动态构建大小不等的 CP groups
- 长序列 → 大 CP group（更多 rank 分担）；短序列 → 小 CP group（减少通信开销）
- 模型并行拓扑（TP/PP）保持不变，仅 CP group 划分动态调整
- 效果：相比固定 CP，packed sequence SFT 场景下可提升高达 **1.48x** 速度

**支持的 Scheduler**：
- `dp_balanced`：按原始顺序打包到 max_seqlen，保证 DP rank 间负载均衡
- `default_dynamic_cp`：打包同时考虑 CP 组内 token 分布，使每个动态 CP group 内计算量更均衡

### 3.6 并行策略组合

| 并行策略 | 671B MoE @ 128 GPU 推荐配置 | 128 GPU 利用率分析 |
|---------|--------------------------|-------------------|
| DP | dp_size = 16（含分布式优化器分片） | 每 copy 独立处理不同 batch |
| TP | tp_size = 4 | 每层参数切分为 4 份 |
| PP | pp_size = 8（48 层 ÷ 8 = 6 层/rank） | 减少单卡显存 |
| EP | ep_size = 8（64 experts ÷ 8） | 每 rank 8 个 expert |
| CP | cp_size = 4（128K seq ÷ 4 = 32K/rank） | 减少 Attention 激活 |

验证：16×4×8×8×4 = 2048 > 128。实际按 `DP × TP × PP × EP` 计算，DP/EP 可部分重叠，最终 `16×4×8×8 = 4096` 但 CP 融入 TP 维度，经配置优化可匹配 128 GPU。

### 3.7 通信量总览

| 并行维度 | Forward 通信 | Backward 通信 | 每 Step 总通信 | 通信组 |
|---------|-------------|-------------|---------------|--------|
| **TP** (ColParallel) | AllGather: N×H×(T-1)/T | AllReduce: N×H×2(T-1)/T + W²×2(T-1)/T | ~4×N×H×(T-1)/T + const | `tp_group` |
| **TP** (RowParallel) | ReduceScatter: N×H×(T-1)/T | AllReduce: N×H×2(T-1)/T + W²×2(T-1)/T | ~4×N×H×(T-1)/T + const | `tp_group` |
| **PP** | P2P Send/Recv: 1×N×H | P2P Send/Recv: 1×N×H | 2×N×H per boundary | `pp_group` |
| **EP** (AlltoAll) | 2×S×B×H×K×(E-1)/E² | 2×S×B×H×K×(E-1)/E² | 4×S×B×H×K×(E-1)/E² | `ep_group` |
| **CP** | AllGather(KV): S×H/C×(C-1)/C × #heads | 同 Forward | 2×S×H×(C-1)/C² × #heads | `cp_group` |
| **DP** (DistOpt) | AllGather(P): P | ReduceScatter(P): P | 2P (same as AllReduce) | `dp_group` |

> **图例**: T=tp_size, E=ep_size, C=cp_size, P=参数总量, N=seq×batch, H=hidden_size, K=topk, S=seq_len

**通信量排序**（典型 671B MoE, S=128K, H=8192, T=4, E=8, C=4, K=8）:
```
EP All-to-All (S×H×K ≈ 8GB) > TP (N×H ≈ 2GB) ≈ DP (P/D ≈ 1.5GB) > CP (S×H/C ≈ 1GB) > PP (P2P ≈ 0.5GB)
```

---

## 4. 分布式优化器

### 4.1 核心机制

Megatron 分布式优化器（`megatron/core/optimizer/distrib_optimizer.py:113`）实现 ZeRO-1（优化器状态分片）+ ZeRO-2（梯度分片）：

```
                标准 All-Reduce                    分布式优化器

梯度同步：    AllReduce(∂L/∂W)          ReduceScatter(∂L/∂W)  [只保留自己的分片]
状态维护：    每 rank 存全部 Adam       每 rank 只存自己的 1/dp_size
参数更新：    每 rank 更新全部参数       每 rank 只更新自己的分片
参数同步：    无需                       AllGather(W_updated)   [广播更新结果]
```

**通信总量相同（2×P），但显存节省 ~3×/dp_size。**

### 4.2 关键实现细节

**Bucket 分片**（`megatron/core/distributed/param_and_grad_buffer.py:1066` `_ParamAndGradBuffer`；对齐/补齐规则在 `megatron/core/distributed/nonuniform_tp.py:49-63` 的 `pad_param_start` / `pad_bucket_end`）：
所有参数平铺为连续 Buffer，Padding 到 `lcm(dp_size, 128, 65536)` 的倍数后均匀切分。分片边界可能穿过参数中间——一个参数的不同片段由不同 DP rank 维护状态。

**HSDP 多实例**（`megatron/core/distributed/param_and_grad_buffer.py:786-812`）：
`num_distributed_optimizer_instances > 1` 时，DP 域分为组内（ReduceScatter 分片）和组间（AllReduce 聚合），实现分层分片。

**精度感知优化器**：
标准路径下主权重为 FP32。精度感知模式下（`use_precision_aware_optimizer`），主权重、exp_avg、exp_avg_sq 可按不同精度存储，使用 `.decoupled_grad` 解耦模型 dtype 和优化器 dtype。

### 4.3 CPU Offloading 两种模式

| 模式 | 类 | 卸载内容 | 时机 |
|------|-----|---------|------|
| HybridDeviceOptimizer | `megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py:14` | 部分参数的优化器状态永久放 CPU | `offload_fraction` 控制 |
| ChunkedOptimizerStateOffloader | `megatron/core/optimizer/cpu_offloading/chunked_optimizer_state_offload.py:57`（原 `optimizer_state_offloader.py` 的 `OptimizerStateOffloader`，#6244 重写为分块版并改名） | 全部状态在 step 间暂存 CPU | step 后 offload，下次 step 前 reload |

**D2H/H2D 重叠**（`megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py:162-179`）：
```
_d2h_stream: GPU grad → CPU (async)
  → GPU optimizer.step()
    → cpu_optimizer.step()  (after d2h sync)
      → _h2d_stream: CPU param → GPU (async, via post-hook)
```

### 4.4 配置建议

- <10B 模型：`use_distributed_optimizer=True`（开箱即用）
- 100B+ 模型：+ `overlap_param_gather=True` + `overlap_grad_reduce=True`
- 极端显存紧张：+ `optimizer_cpu_offload` 或 `offload_optimizer_states`

详细分析见 [[16_megatron_distributed_optimizer_analysis]]。

### 4.5 三种梯度/参数分片策略对比

Megatron-LM 实际有 **三套** 并行分片方案，`DistributedOptimizer` 只是其中之一：

| 方案 | 分片粒度 | ZeRO 等效 | 核心文件 |
|------|---------|----------|---------|
| **DistributedOptimizer** | 参数级（连续 Buffer 切分） | ZeRO-1/2 | `megatron/core/optimizer/distrib_optimizer.py:113` |
| **TorchFullyShardedDataParallel** | Module 级（FSDP Unit） | ZeRO-3 | `megatron/core/distributed/torch_fully_sharded_data_parallel.py:28` |
| **MegatronFSDP** | Module 级（FSDP Unit） | ZeRO-1/2/3 可配置 | `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:94` |

#### 3.5.1 MegatronFSDP — 自研 FSDP 实现

**分片策略谱系**（`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:100-108`）：
```python
'no_shard'              # 传统 DP
'optim'                 # ZeRO-1: 仅优化器状态分片
'optim_grads'           # ZeRO-2: 梯度 + 状态分片
'optim_grads_params'    # ZeRO-3: 参数 + 梯度 + 状态全分片
```

**FSDP Unit**（`:130`）：最细粒度的可释放模型单元。参数按 Unit 分组——进入 Unit 时 AllGather 参数离开时释放，Backward 时重新 AllGather。默认 Unit = TransformerLayer。

**四种训练状态**（`:51-63`）：
```
FORWARD → PRE_BACKWARD → POST_BACKWARD → IDLE
  参数 unshard    参数 unshard    梯度 re-shard   空闲
```

**关键特性**：
- **与 Activation Checkpointing 协同**（`:115-118`）：重算整个 Layer 时只 Gather 一次参数，供重算和 Backward 共同使用
- **Delayed Wgrad Overlap**（`:66-91`）：expert 参数的梯度 reduce-scatter 延迟到 MoE dispatch backward 完成，最大化 EP+DP 通信重叠
- **NCCL UserBuffer**（`:152-156`）：`nccl_ub=True` 使用 NCCL UB 减少 SM 占用，自动开启双缓冲
- **HSDP 分层**：组内 ZeRO-3 + 组间复制（`outer_dp_sharding_strategy`）

#### 3.5.2 TorchFullyShardedDataParallel — PyTorch 原生 FSDP2

对接 PyTorch >= 2.4 的 `torch.distributed.fsdp.fully_shard` API：

```python
sub_modules_to_wrap = {TransformerLayer, LanguageModelEmbedding,
                        RotaryEmbedding, ColumnParallelLinear}
fully_shard(sub_module, mesh=device_mesh, reshard_after_forward=True)
```

- **Per-module 包装**：每个 sub-module 独立 `fully_shard` → 逐层按需 AllGather/释放
- **FP8 转置缓存处理**（`:93-98`）：PyTorch FSDP2 无法感知 micro-batch → 自定义属性保存/恢复
- **Backward Prefetch 协调**（`:136-141`）：显式设置 prefetch schedule 防止 Activation Checkpointing 破坏 FSDP2 的自动 prefetch

#### 3.5.3 选型决策

| 场景 | 推荐 |
|------|------|
| 标准训练 | `DistributedOptimizer` — 最成熟 |
| PyTorch >= 2.4 新项目 | `TorchFullyShardedDataParallel` — 原生 API |
| CUDA Graph + FSDP | `MegatronFSDP` — TorchFSDP2 不兼容 |
| ZeRO-3 参数全分片 | `MegatronFSDP` — `optim_grads_params` |
| MoE（EP 耦合） | `MegatronFSDP` — EP 参数自动检测 + Delayed Wgrad |

---

## 5. 重计算技术（Activation Checkpointing）

### 5.1 核心思想

以额外的计算换显存——forward 时不保存某些中间激活，backward 时重新计算。Transformer 中 Attention 和 FFN 的激活占显存的 60-80%，是首选的 checkpoint 目标。

### 5.2 Megatron 的实现层次

| 层次 | 控制参数 | 机制 |
|------|---------|------|
| **Full Checkpointing** | `recompute_method='block'` | 每个 Transformer Block 的输入保存，内部激活全部重算 |
| **Selective Checkpointing** | `recompute_modules` | 只重算指定的 modules（如只重算 Attention，不重算 FFN） |
| **MoE-Specific** | `moe_paged_stash=True` | 对 MoE expert 的激活做页面化暂存（见 §8） |

### 5.3 与 PP 调度的协同

在 1F1B 调度中，activation checkpointing 与 pipeline bubble 有协同效应：
- forward warm-up 阶段保存 checkpoint
- backward 阶段重算激活（与 bubble 空闲时间部分重叠）

### 5.4 什么时候用

- ✓ 模型 ≥ 1B 参数（激活值超过参数显存）
- ✓ MoE（每个 expert 独立激活，显存压力更大）
- ✓ 长序列训练（Attention 激活 O(S²)）
- ✗ 推理阶段（不需要 backward 激活）
- ✗ 极致低延迟需求（~33% 额外计算）

详细分析见 [[12_activation_checkpointing_analysis]]。

---

## 6. 低精度训练

### 6.1 精度体系

| 精度 | 参数显存 | 适用硬件 | Megatron 控制 |
|------|---------|---------|--------------|
| BF16 | 2 bytes/param | 所有 | 默认 |
| FP8 E4M3 | 1 byte/param | H100+ | `fp8='e4m3'`, `fp8_param=True` |
| MXFP8 | 1 byte/param + block scale | GH200, Blackwell | `fp8_recipe='mxfp8'` |
| NVFP4 | 0.5 byte/param | Blackwell | `fp4=True`, `fp4_recipe` |

### 6.2 Megatron 的 FP8 实现

**Transformer Engine 集成**（`megatron/core/fp8_utils.py`）：
- 所有 GEMM 在 FP8 精度下执行
- 梯度采用 FP8 计算（`fp8_wgrad=True`）
- 参数可选择以 FP8 存储（`fp8_param=True`）：all-gather/ReduceScatter 通信量减半
- Amax 全规约：延迟缩放（delayed scaling）或张量级缩放（tensorwise scaling）

**MoE 特殊考量**：
- Router 通常保持高精度（路由决策对精度敏感）
- Expert 内部 FP8 可安全使用（每个 expert 处理 token 子集）
- 首尾层保护（`first_last_layers_bf16`）：保持首尾 N 层在 BF16

### 6.3 为什么低精度在 MoE 中更重要

MoE 的参数量是稠密模型的 N×（N=expert 数量），在 671B MoE 中 FP8 节省的参数显存是 `671B × 1 byte = 671GB`，而不是稠密模型的 `67B × 1 byte = 67GB`。效果随着 expert 数量线性放大。

### 6.4 FP8 Input Store

激活融合算子（SwiGLU, GEGLU）支持 `fp8_input_store`——forward 中将 backward 需要的输入转为 FP8 存储（节省 50% 激活显存），backward 时恢复。对 MoE 每层的 [10K tokens, 2×8192 hidden] 节省 ~164MB。

详细分析见 [[13_low_precision_training_analysis]] 和 [[14_transformer_engine_analysis]]。

---

## 7. 通信-计算 Overlap

### 7.1 六维 Overlap 全景

Megatron-LM 在所有并行维度实现了通信-计算的流水线化重叠：

| 维度 | 重叠方式 | 关键文件 |
|------|---------|---------|
| **DP Grad** | Backward 部分层时启动 bucket 的 ReduceScatter | `megatron/core/distributed/distributed_data_parallel.py:500` |
| **DP Param** | Forward 部分层时异步 AllGather 下一层参数 | `megatron/core/distributed/param_and_grad_buffer.py:448` |
| **TP** | Attention/FFN 内部 TP 通信与下一操作重叠 | `megatron/core/tensor_parallel/layers.py` |
| **PP** | P2P send/recv 与 1F1B 计算交错 | `megatron/core/pipeline_parallel/schedules.py` |
| **EP** | All-to-All dispatch/combine 与 expert compute 在不同 CUDA stream | `megatron/core/transformer/moe/fused_a2a.py` |
| **CP** | KV AllGather（双缓冲）与 attention 计算交错 | `megatron/core/transformer/dot_product_attention_context_parallel.py:198` |

### 7.2 DP 维度：Grad Reduce 与 Param Gather 的 Overlap

**Grad Overlap**（`megatron/core/distributed/distributed_data_parallel.py:500-530` `_make_backward_post_hook`）：
```
Backward Layer 48 →
  Backward Layer 47 → Bucket 1 ReduceScatter (async)
  Backward Layer 46 → Bucket 2 ReduceScatter (async)
  ...
```
每层 backward 完成立即触发对应 bucket 的异步 ReduceScatter，main stream 继续下一层。

**Param Overlap**（`megatron/core/distributed/param_and_grad_buffer.py:448` `_ParamAndGradBucketGroup.start_param_sync`）：
```
Forward Layer 1: wait for bucket 1 → trigger bucket 2 AllGather (async)
Forward Layer 2: wait for bucket 2 → trigger bucket 3 AllGather (async)
...
```
通过 `_make_forward_pre_hook` 实现：每层 forward 前检查其参数的 AllGather 是否完成，触发下一 bucket 的 AllGather。

### 7.3 EP 维度：DeepEP/HybridEP

`megatron/core/transformer/moe/fused_a2a.py` 提供两种后端：

**DeepEP**（`:116-312`）：
- `FusedDispatch`：Layout 计算 + Permute + All-to-All + Unpermute 融合
- `FusedCombine`：Permute + All-to-All + Unpermute 融合
- `async_finish` + CUDA event：dispatch 可与 expert compute overlap

**HybridEP**（`:506-807`）：
- 进一步支持 `fuse_permute_dispatch` 和 `fuse_unpermute_combine`
- Fine-grained SM control：可调分配给 dispatch/combine/permute 的 SM 数量

### 7.4 为什么 Overlap 在 MoE 中至关重要

MoE 的通信密度远高于稠密模型——每个 MoE layer 额外需要 2 次 All-to-All（dispatch + combine）。在 8 个 expert 的场景，如果 EP=8，每个 rank 的 All-to-All 通信量约为 `S×B×H×2 bytes`。Overlap 将这部分通信隐藏在 expert compute 或相邻层的计算中，实际上是"免费"的。

详细分析见 [[20_megatron_comm_overlap_analysis]]。

---

## 8. 显存优化

### 8.1 MoE 的显存挑战

MoE 的显存问题比稠密模型严重得多：
- N 个 expert × FFN 参数 = 参数量 N 倍
- 每个 expert 独立激活（如果 topk=2，激活为稠密的 ~2 倍）
- All-to-All 需要较大的临时通信 buffer

### 8.2 Paged Stash — MoE 激活的虚拟内存

`megatron/core/transformer/moe/paged_stash.py` 是 Megatron-LM 针对 MoE 最独特的显存优化：

```
激活值管理 = 操作系统虚拟内存的 GPU 实现
  - Page（页面）: 连续的激活 tensor 片段
  - Page Table: 记录每个 tensor 占用了哪些 page
  - Stash（暂存区）: GPU 上的固定大小 page pool
  - Spill（溢出）: 当 GPU page pool 满时，溢出到 CPU pinned memory
  - Rerun（重运行）: 当 CPU pool 也满时，丢弃 stash，backward 时重算
```

**三级处理**：
- Tier 1（快速）：CUDA page pool → 无额外延迟
- Tier 2（中等）：CPU pinned buffer → D2H/H2D 延迟，但与计算重叠
- Tier 3（回退）：Rerun → 额外计算开销，但极其罕见（仅当 buffer 设置过小时触发）

**CUDA Graph 集成**（`:1247-1267`）：capture 阶段记录 page 分配模式，replay 时零开销执行。

> [!deprecated] 该机制在基线 `71092579` 下已不存在（`megatron/core/transformer/moe/paged_stash.py` 全文只有 1240 行，`:1247-1267` 越界；全文件仅 `:10`、`:1134`、`:1135`、`:1143` 四处涉及 CUDA Graph，且都是"fallback 时先 `FullCudaGraphWrapper.reset_cuda_graph()` 再释放 stash 页缓冲"(`:1133`–`:1137`)，没有"capture 阶段记录 page 分配模式"这样的代码）。以上描述无法在新基线上定位，亦无法在旧基线 `232c478d4` 上定位（该 commit 下此文件同为 1240 行），保留原文仅供追溯。

**配置**：
```python
moe_paged_stash_page_size = 64         # page 大小
moe_paged_stash_buffer_size_factor_cuda = 1.10  # 1.10× peak 通常足够
moe_paged_stash_buffer_size_factor_cpu = 0.0    # 可选 CPU buffer
```

### 8.3 Fine-Grained Activation Offloading

`megatron/core/pipeline_parallel/fine_grained_activation_offload.py` 提供 sub-layer 粒度的激活卸载：

- 支持卸载模块（`megatron/core/transformer/transformer_config.py:1453-1461` `offload_modules`）：`attn_norm`, `qkv_linear`, `core_attn`, `attn_proj`, `mlp_norm`, `expert_fc1`, `moe_act`
- 选择性卸载：跳过大 tensor（`min_offloaded_tensor_size`）、跳过后几层（`offload_margin`）
- 与 CUDA Graph 兼容：延迟卸载在 graph replay 后批量执行

### 8.4 Buffer 复用策略

| 复用模式 | 文件 | 节省 |
|---------|------|------|
| MXFP8 共享 Buffer | `megatron/core/distributed/param_and_grad_buffer.py:1231` | 参数 all-gather 和梯度 reduce-scatter 共享存储 → 1x 参数量节省 |
| Grad→Param 回收 | `megatron/core/distributed/param_and_grad_buffer.py:536-544` | 梯度 reduce-scatter 后 buffer 清零重用于参数 all-gather |
| NVFP4 打包布局 | `megatron/core/distributed/param_and_grad_buffer.py:1159-1184` | 参数 buffer 为 N/2 bytes，梯度 buffer 为 N×2 bytes |
| NCCL Memory Pool | `megatron/core/nccl_allocator.py:53` | `ncclMemAlloc` 避免 PyTorch allocator 碎片化 |

### 8.5 显存优化组合策略

```
Layer 1（零开销）: FP8 参数 + 分布式优化器 + Sequence Parallelism
  → 参数: 1x → 0.5x, 状态: 3x → 3x/dp

Layer 2（极小开销）: Grad Buffer 复用 + NCCL Pool + FP8 Input Store
  → 临时 buffer -50%, 激活 -40%

Layer 3（可控计算开销）: Activation Checkpointing + Paged Stash
  → 激活: 5-10x → 1-2x, MoE 激活 -90%

Layer 4（~10-30% 速度损失）: CPU Offloading (Optimizer States + Activations)
  → 进一步押榨 GPU 显存
```

详细分析见 [[22_megatron_memory_optimization_analysis]]。

---

## 9. 融合算子优化

### 9.1 融合原理

GPU 计算分为两类：
- **计算受限（Compute-bound）**：矩阵乘法（GEMM）、大卷积 → GPU FLOPS 是瓶颈
- **内存受限（Memory-bound）**：激活函数、归一化、Dropout → HBM 带宽是瓶颈

融合算子的核心价值是将多个内存受限操作合并为单个 kernel，消除中间张量的 HBM 往返。

### 9.2 Megatron 的融合层次

| 实现层次 | 代表性融合 | 技术 |
|---------|-----------|------|
| `@jit_fuser` | GEGLU, SwiGLU, GELU, SquaredReLU | TorchScript / torch.compile |
| Apex CUDA | LayerNorm | NVIDIA 手写 CUDA persistent kernel |
| Custom CUDA | Scaled Masked Softmax | 手写 CUDA（消除 mask tensor 分配） |
| Triton | Pad Routing Map, Indices Converter, MLA RoPE | Triton autotune + debug_barrier |
| CUTLASS/cuTile | Linear+CrossEntropy, MHC | Blackwell 专用 tile 级编程 |

### 9.3 核心融合实例

**Bias + GEGLU/SwiGLU + Weighting**（`megatron/core/fusions/fused_bias_geglu.py`）：
```
标准: y = y + bias → chunk(y, 2) → GELU(y1) → y1*y2 → *weights  (5 kernels)
融合: y, bias, weights → bias_geglu_weighted_kernel → output      (1 kernel)
```

**Fused Cross-Entropy**（`megatron/core/fusions/fused_cross_entropy.py`）：
- 将 logits_max 和 sum_exp_logits 拼接 → 只需 1 次 AllReduce
- TP 场景下通信量减半

**Fused Softmax**（`megatron/core/fusions/fused_softmax.py:179`）：
- Causal mask 在 CUDA kernel 的寄存器中即时生成
- 消除 `[b, np, sq, sk]` 的 mask tensor（对 128K 序列节省 ~64GB/层）

### 9.4 MoE 专用融合

| 融合 | 作用 | 为什么在 MoE 中重要 |
|------|------|-------------------|
| Pad Routing Map（`megatron/core/fusions/fused_pad_routing_map.py`） | 将 routing map padding 到 GEMM 对齐的倍数 | MoE 每层都需要，涉及整个 routing map 的扫描 |
| Indices Converter（`megatron/core/fusions/fused_indices_converter.py`） | `[N, K]` sparse ↔ `[N, E]` multi-hot 双向转换 | Dispatch 前需要 multihot，Combine 后需要 indices |
| Weighted Squared ReLU（`megatron/core/fusions/fused_weighted_squared_relu.py`） | ReLU² × routing weights | MoE routing probability 直接融入 activation |
| Fused A2A（`megatron/core/transformer/moe/fused_a2a.py`） | Permute + All-to-All + Unpermute 合并 | 消除通信前的排列开销 |

### 9.5 FP8 Input Store

SwiGLU/GEGLU 的 forward 中将输入转为 FP8 保存（1 byte vs 2 bytes），backward 时恢复。对 MoE 场景（10K tokens, 2×8192 hidden），每层每 expert 节省 ~164MB。

### 9.6 适用准则

- ✓ 所有 Transformer 训练（激活融合无条件适用）
- ✓ MoE（MoE 专用融合）
- ✗ 极小的 hidden size（LayerNorm 融合的 persistent kernel 不支持则回退）
- ✗ 非 Blackwell GPU（Linear+CrossEntropy CUTLASS 不可用）

详细分析见 [[21_megatron_fusion_operators_analysis]]。

---

## 10. 总结 — 技术矩阵与选型决策树

### 10.1 7 维技术关联矩阵

```
                 TP  PP  EP  CP  DistOpt  Recomp  LowPrec  Overlap  Memory  Fusion
TP（显存+计算）    -  ++  ○   ○    ○       ○       ○        ++       ○       ○
PP（显存）        ++  -   ++  ○    ○       ++      ○        ++       ○       ○
EP（参数爆炸）    ○   ++  -   ○    ○       ○       ○        ++       ++      ○
CP（长序列）      ○   ○   ○   -    ○       ○       ○        ++       ○       ○
DistOpt（状态）   ○   ○   ○   ○    -       ○       ++       ++       ++      ○
Recomp（激活）    ○   ++  ○   ○    ○       -       ○        ○        ++      ○
LowPrec（精度）   ○   ○   ○   ○    ++      ○       -        ++       ++      ○
Overlap（隐藏）   ++  ++  ++  ++   ++      ○       ++       -        ○       +
Memory（压榨）    ○   ++  ++  ○    ++      ++      ++       ++       -       ○
Fusion（kernel）  +   ○   ○   ○    ○       ○       ○        +        ○       -

符号: ++ 强协同, + 正向关联, ○ 正交或弱关联, - 不适用
```

### 10.2 各模型规模的推荐配置

#### 中等规模 MoE (~50B 总参数量, ~5B 激活)

```
硬件: 8-16 GPUs (A100/H100)
TP=2, PP=4, EP=4, DP=1-2
优化: 分布式优化器 + FP8 + Activation Checkpointing (selective)
通信: DP Overlap（grad reduce + param gather）
关键瓶颈: 可能 EP 收益不大（expert 数量通常在 8 左右）
```

#### 大规模 MoE (~200-300B 总参数量, ~30-40B 激活)

```
硬件: 32-64 GPUs (H100)
TP=4, PP=8, EP=8, DP=1-2, CP=1
优化: 全量 FP8 + Distributed Optimizer + Selective AC + Paged Stash
通信: 全维度 Overlap (DP/TP/PP/EP)，DeepEP/HybridEP
关键瓶颈: EP All-to-All 通信 + Expert 负载不均衡
```

#### 超大规模 MoE (671B 总参数量, 37B 激活 — DeepSeek-V3 级别)

```
硬件: 128-256 GPUs (H100/B200)
TP=4, PP=16, EP=16, DP=2, CP=2
优化: FP8 全量 + NVFP4 Param + Distributed Optimizer + Full AC + Paged Stash
      + FP8 Input Store + Fine-Grained Activation Offloading
通信: 全维度 Overlap + NCCL Pool + MXFP8 Shared Buffer
显存: 多层组合（Paged Stash + Activation Offloading + Optimizer State Offloading）
关键瓶颈: 所有维度同时成为瓶颈，需要极致的协同设计
```

#### 极端规模 MoE (1.x TB 总参数量, 50B+ 激活 — DeepSeek-V4 级别)

```
硬件: 512+ GPUs (Blackwell B200+)
TP=8, PP=32, EP=32, DP=8, CP=4
优化: NVFP4 全量 + Distributed Optimizer + Full AC + Paged Stash (3-tier)
      + Fine-Grained Offloading + CUDA Graph 全覆盖
      + Resharding（动态并行策略切换）
通信: 全维度 Overlap + NCCL NVLS + DeepEP async + CUTLASS/cuTile fusion
      + Linear+CrossEntropy Blackwell fusion
显存: 全部组合（Paged Stash + Offload × 2 + Buffer 复用 × 3 + FP4 params）
关键瓶颈: 通信带宽成为最终限制，需要 NVLink + InfiniBand 全配合
```

### 10.3 选型决策树

```
开始
 │
 ├─→ 模型 < 10B？
 │    └─ YES → 仅需 TP + DP + 基础 AC。Done.
 │
 ├─→ Expert 数量 > 8？
 │    ├─ NO  → EP 收益有限，使用 TP + PP 为主
 │    └─ YES → 必须使用 EP，继续 ▼
 │
 ├─→ 序列长度 > 32K？
 │    ├─ NO  → 跳过 CP
 │    └─ YES → 添加 CP（cp_size 按 S/cp_size ≤ 32K 选择）
 │
 ├─→ GPU 总数 > 16？
 │    ├─ NO  → EP + TP 为主，PP 少量
 │    └─ YES → 全维度展开，继续 ▼
 │
 ├─→ GPU 单卡显存 < 80GB？
 │    ├─ NO  → Paged Stash + Selective AC 即可
 │    └─ YES → 需要 + CPU Offloading，继续 ▼
 │
 ├─→ 是 Blackwell GPU？
 │    ├─ NO  → FP8 + MXFP8 + Triton fusions
 │    └─ YES → NVFP4 + CUTLASS/cuTile fusions + Linear+CE fusion
 │
 └─→ 最终策略 = 根据上述决策组合选定的技术
```

### 10.4 核心洞察

1. **EP 是 MoE 训练的第一性原理**——没有 EP，大规模 MoE 根本无法装入 GPU。EP 与其他所有维度正交，而且通信可以与计算重叠。

2. **通信 Overlap 是免费的午餐**——在 MoE 密集的通信模式下（DP + TP + EP + CP），Overlap 能将相当一部分通信隐藏在计算中。Megatron-LM 的 6 维 Overlap 是所有优化中 ROI 最高的。

3. **显存优化的层次性**——从零开销的 FP8/分布式优化器/SQ，到~33% 计算开销的 AC，到~10-30% 速度损失的 CPU Offloading，选择应遵循"先零开销、后可控开销"的原则。

4. **融合算子是积少成多**——单个融合可能只节省 5-10μs，但在 80 层 × 多个 element-wise 操作的累积下，每天训练可节省数小时。

5. **MoE 梯度 = 专家独立性**——每个 expert 的梯度和激活彼此独立，因此可以采用 Paged Stash 和 per-expert 显存管理；这也是 MoE 相比稠密模型最主要的优化空间。

---

## 11. 约束

§1.3 说 7 个维度"正交可叠加"，这在**机制**上成立、在**配置**上不成立：源码把大量交叉点写成了构造期 assert。本节汇总**框架层面**的前提、互斥与失效条件，每条带 locator；单个维度内部的细约束见各 sibling 页。

### 11.1 维度之间的硬互斥

| 组合 | 结果 | locator |
|---|---|---|
| `moe_paged_stash` + `cpu_offloading` | assert 失败：「moe_paged_stash cannot be enabled with cpu_offloading.」 | `megatron/core/transformer/transformer_config.py:2547-2548` |
| `moe_paged_stash` 未设 `moe_expert_rank_capacity_factor` | assert 失败：「there is no need to use paged stashing without it.」 | `megatron/core/transformer/transformer_config.py:2549-2552` |
| `moe_paged_stash` + `offload_modules` 含 `expert_fc1` / `moe_act` / `fused_group_mlp` | assert 失败（paged stash 已接管这些激活） | `megatron/core/transformer/transformer_config.py:2553-2561` |
| `num_buckets` 与 `bucket_size` 同时指定 | assert 失败：「Cannot specify both num_buckets and bucket_size」 | `megatron/core/distributed/distributed_data_parallel_config.py:313-315` |
| `nccl_ub` + torch < 2.11.0a0 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | `raise ValueError` | `megatron/core/distributed/distributed_data_parallel_config.py:300-305` |

### 11.2 EP 后端的前提

- **三个 flex 后端都只吃 fp32 probs**：HybridEP（`megatron/core/transformer/moe/token_dispatcher.py:1194-1198`）、DeepEP（`:1370-1374`，另有 `:1497` 的 assert）、DeepEP v2（`:1588-1592`），错误文案一律以「please set --moe-router-dtype=fp32」结尾。
- **flex dispatcher 要求 TP×EP > 1**：`deepep`（`:1886`）、`deepepv2`（`:1896`）、`ncclep`（`:1914`）三处同文 assert。
- **HybridEP 的变长 token 补齐现在必须显式打开**：字段 docstring 写明「When disabled, the caller must guarantee equal token counts across the HybridEP communication group … CUDA Graph inputs should be statically padded upstream and leave this option disabled.」（`megatron/core/transformer/transformer_config.py:981-988`），实现只剩一条 `if self.config.moe_hybridep_pad_variable_tokens:` 分支（`megatron/core/transformer/moe/token_dispatcher.py:1132-1142`）。详见 §12 末的补充勘误。

### 11.3 不变量

- **DDP 的一切以 buffer / bucket 为单位**：参数与梯度先拼成连续缓冲再切桶（`megatron/core/distributed/param_and_grad_buffer.py:1066-1071`）；桶大小、桶数、NCCL 注册、CPU 备份全是 buffer 级旋钮（`megatron/core/distributed/distributed_data_parallel_config.py:33-38`、`:60-76`、`:130-148`）。改动任何一维优化都不能打破这个粒度。
- **flex dispatcher 的契约是一份 routing_map**：形状固定为 `[num_local_tokens, world_size, num_instances]`（`megatron/core/transformer/moe/token_dispatcher.py:989-999`），四个后端不得改变它。
- **aux/z-loss 的缩放不能用 `group_size` 反推**：`megatron/core/transformer/moe/router.py:598-604` 的注释就是这条不变量的直接陈述。

### 11.4 失效条件

- **变长 / 动态形状**：THD packing 与动态 CP 会让"各 rank token 数相同"这个隐含前提失效。它同时打掉了旧的 aux-loss 缩放（§2④）、HybridEP 的自动补齐（§12 末），也是 MoE 层难以整层图化的根因（见 [[23_megatron_precision_cudagraph_fusion_analysis]] §4.3）。
- **TP×EP == 1**：flex dispatcher 直接 assert 失败（§11.2），只能退回 `alltoall` dispatcher。
- **旧 MoE 日志 API**：`megatron/core/transformer/moe/moe_utils.py` 里成组的 `@deprecated(version="0.16", removal_version="0.18", …)`（`:991`、`:1027`、`:1109`、`:1128`、`:1142`）——§12 记的日志重构会在 0.18 移除这些旧入口。

### 11.5 本页自身的边界

本页是**全景 + 选型**页：每个维度的机制深潜在 sibling 页，本页只承担"维度之间怎么组合"。所有 `path:line` 以基线 `71092579` 为准；§12 记录的增量本身钉在 `232c478d4`，其中已被推翻的部分都带 `[!contradiction]`。

---

## 12. 增量更新（ee3f1ff → 232c478d4）：五个维度的实际变化，及其中已被推翻的结论

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> 本章是全景报告的"自上次快照（ee3f1ff，2026-05-19）以来的 MoE 相关增量"。这里只做**全景级一段话总结 + 指向详细 sibling 页的交叉链接**，不重复深潜（深潜见各 `[[link]]`）。正文 §3–§10 描述的 7 维体系与机制在当前 HEAD 上**依旧成立**，下面仅补充这一个月内新增/变化的 MoE 维度。

### 通信 / EP 维度 — Token Dispatcher 后端扩张

`moe_flex_dispatcher_backend` 由原来的 `{deepep, hybridep}` 扩为 **`{deepep, deepepv2, hybridep}`** 三选一（`megatron/core/transformer/transformer_config.py:972`，#4793）。新增的 `deepepv2` 后端走 DeepEP v2 的 **ElasticBuffer** API（`_DeepepV2Manager` 见 `megatron/core/transformer/moe/token_dispatcher.py:1529`，`get_elastic_buffer` 见 `megatron/core/transformer/moe/fused_a2a.py:90`），与 v1 的 `Buffer` 完全隔离，仅支持 `float32` probs（须配 `--moe-router-dtype=fp32`）。配套地 `moe_deepep_num_sms` 默认值从固定 `20` 改为 `None`（`megatron/core/transformer/transformer_config.py:1041`）——v1 回退到 20，v2 使用其理论默认。此外 deepep/hybridep 现已支持 **thd（sequence-packing / varlen）格式训练**（#4816），以及 **high-priority A2A stream**（`high_priority_a2a_comm_stream`，`megatron/core/transformer/transformer_config.py:764`；`set_streams(high_priority=…)` 见 `megatron/core/pipeline_parallel/utils.py:350`）与 **HybridEP 预处理 SM 数**可调（`moe_hybridep_num_sms_preprocessing=108`，`megatron/core/transformer/transformer_config.py:1059`，#4694）；dispatcher 内 `all_to_all` 新增 `use_nccl_stream` 形参（shared-experts 场景置 True）。详见 [[14_megatron_ep_analysis]]。

### 显存维度 — Paged Stash 落地 + EP 显存估算修正

§8.2 预告的 **Paged Stash** 已作为完整特性落地（新模块 `megatron/core/transformer/moe/paged_stash.py`，1240 行，#4247）：开关 `moe_paged_stash`（`megatron/core/transformer/transformer_config.py:1495`），`moe_paged_stash_page_size=64`、`*_buffer_size_factor_cuda=1.10`、`*_cpu=0.0`，并**要求同时设置 `moe_expert_rank_capacity_factor`**；与 `cpu_offloading` 互斥，且 `offload_modules` 不得再包含 `expert_fc1`/`moe_act`（已由 paged stash 接管）。理论显存估算 `megatron/training/theoretical_memory_usage.py` 现**正确区分 EP-sharded 路由专家参数与复制参数**，按 `ETP×EP` 切分路由专家（#4687）；Megatron-FSDP 的 grouped expert 权重 padding 也被收紧（#5013）。详见 [[22_megatron_memory_optimization_analysis]]。

### 精度 / 融合维度 — ClampedSwiGLU 与 Dense+Grouped GEMM 融合

DeepSeek-V4 的 **ClampedSwiGLU** 接入 MoE mlp_op_fuser：当 `activation_func_clamp_value` 非空时，SwiGLU 走 TE 的 `ScaledClampedQGeGLU(alpha=1.0, limit=clamp)`（cuDNN geglu kernel 是 swiglu 的超集，`megatron/core/transformer/moe/experts.py:491`，需 TE≥2.17.0.dev0，#5130）。新增 **TEFusedDenseMLP**：`use_grouped_gemm_for_dense_mlp`（`megatron/core/transformer/transformer_config.py:920`）令 dense MLP 走 `GroupedLinear(num_groups=1)`，在 SM100+ / MXFP8 下触发 `ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8` 融合（须 `use_te_op_fuser=True`，#4318）。详见 [[23_megatron_precision_cudagraph_fusion_analysis]] 与 [[21_megatron_fusion_operators_analysis]]。

### 训练稳定性 / 可观测性维度 — 梯度缩放修正 + 日志重构

修复 **TP>1 时 MoE aux_loss / z_loss 梯度缩放错误**（`megatron/core/transformer/moe/router.py:598-628`，#5047）：在 `calculate_per_token_loss` 路径下，aux/z loss 现按 `num_local_tokens × tp_cp_group.size()` 预乘，抵消 `finalize_model_grads` 中 `total_global_tokens` 除子里隐含的 `tp_cp` 因子，使有效缩放回到目标的 `1/(num_micro_batches × dp_size)`。MoE 日志被抽出为独立模块 **`megatron/core/transformer/moe/moe_logging.py`**（`MoEMetricsTracker`，逐层 metric 采集 + 跨 rank 规约 + TensorBoard/W&B，#3431），`megatron/core/transformer/moe/moe_utils.py` 相应瘦身。此外 hash routing 新增 **force-balance / force-biased**（`moe_router_force_load_balancing` / `moe_router_force_biased`，用随机 topk 覆盖路由，`megatron/core/transformer/moe/router.py:919`，#5130）。详见 [[28_megatron_training_stability_observability_analysis]]。

### 模型结构 / 配方维度 — MoE Recipes 与性能基线

新增 `examples/moe_recipes/`（#4890 / #5012 / #5289），覆盖 **DeepSeek-V3 / DeepSeek-V4-Flash / Qwen3-235B-A22B / Qwen3-30B-A3B** 的可复现 YAML 配方，并给出 **Median TFLOP/s/GPU** 基线：如 DeepSeek-V3 GB300 MXFP8（256GPU, TP1/PP4/EP64, HybridEP+partial CG）≈ **1357.7**，Qwen3-235B GB300 full CG ≈ **1323.1**，DeepSeek-V4-Flash GB200（128GPU, EP64, paged stash + full CG + HybridEP）≈ **646.4**。这些配方把上文各维度（HybridEP/DeepEP、paged stash、partial/full CUDA Graph、EP overlap、MXFP8）组合成端到端可跑的样例，是 §10.2"各模型规模的推荐配置"的真实对照。详见 [[10_megatron_model_structure_analysis]]。

> [!contradiction] 本章描述的是基线 `232c478d4`(2026-06-16)的状态；重定基线到 `71092579`(2026-08-27) 后，其中三处结论**已被后续提交改写**(行号均按新基线重核)：
> - **后端枚举不止三个。** `moe_flex_dispatcher_backend` 现为 `Literal['deepep','deepepv2','hybridep','ncclep']`(`megatron/core/transformer/transformer_config.py:972`)，新增第四个后端 **`ncclep`**(`_NCCLEPManager`,`megatron/core/transformer/moe/token_dispatcher.py:1637`；封装在 `megatron/core/transformer/moe/fused_a2a.py:839` `ensure_nccl_ep_bootstrapped` / `:917` `nccl_ep_dispatch` / `:945` `nccl_ep_combine`)，走 TransformerEngine 的 `transformer_engine.pytorch.ep` API，并带新旋钮 `moe_ncclep_static_shape`(`megatron/core/transformer/transformer_config.py:1062`)让 MoE A2A 可被 CUDA Graph 捕获。
> - **`moe_deepep_num_sms` 已废弃。** 它与 `moe_hybridep_num_sms` 统一进新字段 `moe_flex_dispatcher_num_sms`(`megatron/core/transformer/transformer_config.py:1036`)，旧字段仅在 `__post_init__` 里路由并告警(`:2163`–`:2183`)；"None 时 deepep 回退 20、v2 用其默认"的语义仍成立，但读的是新字段(`megatron/core/transformer/moe/token_dispatcher.py:1345`–`:1349` 与 `:1555`–`:1560`)。
> - **aux/z loss 的 `× tp_cp_group.size()` 预乘已被删除。** 基线 `232c478d4` 下确实是 `aux_loss * num_local_tokens * self.tp_cp_group.size()`；新基线改为对 `aux_loss_scale_num_tokens` 在 `aux_loss_scale_reduce_groups` 上做 `all_reduce` 后直接相乘(`megatron/core/transformer/moe/router.py:598`–`:628`，#4359)，源码注释明说 "local_num_tokens * group_size is not generally correct"——因为 THD padding / 动态 CP 下各 rank 的有效 token 数不同。z-loss 同样改为在 per-token-loss 路径上直接挂 `z_loss_sum`(`:666`–`:675`)。本节"按 `num_local_tokens × tp_cp_group.size()` 预乘"的描述**只对旧基线成立**。

> [!contradiction] 2026-08-28 补（基线 `71092579`）：**HybridEP 的 THD 自动补齐已被取消。**上文"deepep/hybridep 现已支持 thd（sequence-packing / varlen）格式训练（#4816）"这一半仍然成立，但"自动补齐"那一半在 `904ef6d86`（2026-07-09，commit message 即「[dev] Fix HybridEP token equalization under torch.compile without CUDA graphs (#5668)」）里被删掉：原先只要 `sequence_packing_scheduler is not None` **或** `moe_hybridep_pad_variable_tokens` 为真，`_HybridEPManager.setup_metadata` 就会做一次 `all_reduce(MAX) + .item()` 把本地 token 数补到组内最大（并为 CUDA graph 捕获留了一条特判分支）；该提交把 `sequence_packing_scheduler` 这条自动路径连同那个特判分支整段删除，现在**只有显式打开 `moe_hybridep_pad_variable_tokens` 才补齐**（`megatron/core/transformer/moe/token_dispatcher.py:1132-1142`）。字段 docstring 同步改写为「Dynamically pad uneven local token counts … Enable this when local token counts can differ across ranks. When disabled, the caller must guarantee equal token counts across the HybridEP communication group, for example by padding THD inputs to a fixed maximum before dispatch. CUDA Graph inputs should be statically padded upstream and leave this option disabled.」（`megatron/core/transformer/transformer_config.py:981-988`）。理由写在 commit 标题里：那次运行期的 `all_reduce + .item()` 在 torch.compile 且无 CUDA Graph 时会出问题。

---

## 13. 发展趋势

> [!note] 推断
> 本节由 **§12 里带 PR 号的 `[!update]` / `[!contradiction]` 记录**与**当前基线里的 `TODO` / `@deprecated` 注释**共同锚定；方向判断属本页推断，不是源码自陈的路线图。

- **EP 后端还在扩张，并开始向"可图化"收敛**。§12 的 `[!contradiction]` 记下第四个后端 `ncclep`（`megatron/core/transformer/moe/token_dispatcher.py:1637`）与旋钮 `moe_ncclep_static_shape`（`megatron/core/transformer/transformer_config.py:1062`）——后者的目的正是让 MoE A2A 能被 CUDA Graph 捕获；同一方向上，HybridEP 运行期的 `all_reduce + .item()` 自动补齐被 #5668 收回成显式开关（§12 末）。→ 趋势是把 MoE 通信路径上的动态量逐个赶走，好让整层进图。
- **一个后端一套旋钮的写法在退场**：`moe_deepep_num_sms` / `moe_hybridep_num_sms` 已统一进 `moe_flex_dispatcher_num_sms`（`megatron/core/transformer/transformer_config.py:1036`，旧字段在 `:2163-2183` 路由并告警，见 §12）。→ 后续新增后端大概率直接复用统一旋钮。
- **变长 / 动态并行正在成为一等公民**：aux/z-loss 改用组内 `all_reduce`（#4359，`megatron/core/transformer/moe/router.py:598-628`）与 HybridEP 的收紧同源，判据都是"各 rank 有效 token 数可能不同"。→ 凡是靠 `group_size` 反推全局量的代码都会被同样改写。
- **全局 `parallel_state` 正在被 `pg_collection` 取代**：`# TODO(Hepteract): delete the usage of the global parallel_state.` 出现在 `megatron/core/transformer/moe/moe_layer.py:239`、`megatron/core/transformer/moe/moe_utils.py:1199`、`:1479`、`megatron/core/transformer/moe/shared_experts.py:128`；并带一个命名里程碑：`# TODO(M4): breaking api, switched from pass in tp_group to pass in pg_collection.`（`megatron/core/transformer/moe/experts.py:179`、`:1260`）。→ 这是一次会 break API 的重构，§3.1.1 的通信组层级描述届时需按 `pg_collection` 重写。
- **MoE 日志有明确的移除窗口**：`@deprecated(version="0.16", removal_version="0.18", alternative="get_moe_metrics_tracker()…")` 成组挂在 `megatron/core/transformer/moe/moe_utils.py:991`、`:1027`、`:1109`、`:1128`、`:1142`。→ §12 记的日志重构（#3431）会在 0.18 收口。
- **已知未修的观测缺口**：`# TODO (zijiey): fix the per_layer_logging for MTP, currently it will incorrectly …`（`megatron/core/transformer/moe/router.py:577`）——MTP 下的逐层日志目前不正确。

---

## Related Pages

- [[16_megatron_distributed_optimizer_analysis]]
- [[22_megatron_memory_optimization_analysis]]
- [[21_megatron_fusion_operators_analysis]]
- [[20_megatron_comm_overlap_analysis]]
- [[17_megatron_parallelism_orchestration_analysis]]
- [[14_megatron_ep_analysis]]
- [[15_megatron_pp_schedulers_analysis]]
- [[12_activation_checkpointing_analysis]]
- [[13_low_precision_training_analysis]]
- [[14_transformer_engine_analysis]]
- [[14_megatron_ep_analysis]]
- [[23_megatron_precision_cudagraph_fusion_analysis]]
- [[28_megatron_training_stability_observability_analysis]]
- [[10_megatron_model_structure_analysis]]
