# Megatron-LM MoE 零冗余通信实现分析

## 1. 核心概念与原理

### 1.1 什么是零冗余通信（Zero-Redundancy Communication）？

在MoE（Mixture of Experts）模型中，零冗余通信指的是：
- **每个GPU只存储部分专家**：通过Expert Parallelism (EP)，将N个专家分布到多个GPU上
- **按需传输token**：每个token只被发送到包含其选中专家的GPU上
- **避免全量复制**：不同于传统方法将所有专家复制到每个GPU，零冗余方法只传输必要的数据

### 1.2 零冗余通信的核心优势

```
传统方法（全量复制）：
  - 内存占用：O(num_experts × expert_size)
  - 通信量：O(0) - 但需要巨大显存

零冗余方法（Expert Parallelism）：
  - 内存占用：O(num_experts/EP × expert_size)
  - 通信量：O(tokens × hidden_size) - 但显存大幅降低
```

### 1.3 Megatron-LM的零冗余实现策略

Megatron-LM提供三种Token Dispatcher实现零冗余通信：

#### 1) **MoEAlltoAllTokenDispatcher** (推荐用于EP>1)
- 使用All-to-All集合通信原语
- 适合Expert Parallelism场景
- 通信复杂度：O(S×H/TP) 其中S是token数，H是hidden size

#### 2) **MoEAllGatherTokenDispatcher**
- 使用AllGather + ReduceScatter
- 适合TP>1但EP较小的场景
- 通信模式：先聚合所有token，后分散结果

#### 3) **MoEFlexTokenDispatcher** (最新，使用DeepEP)
- 融合permutation和communication操作
- 适合大规模训练和细粒度MoE架构
- 性能优化更激进

> [!update] 2026-06-16 · dev@232c478d4
> **机制复核（ee3f1ff→232c478d4）**：本页第 2 节描述的 `MoEAlltoAllTokenDispatcher` 七阶段流程（dispatch_preprocess → token_dispatch(All-to-All) → dispatch_postprocess → expert compute → combine_preprocess → token_combine(reverse All-to-All) → combine_postprocess）、`input_splits`/`output_splits` 在 combine 阶段互换、以及 `permute`/`unpermute` + variable-size All-to-All 的零冗余机制，在当前 HEAD 上**依旧成立**（`token_dispatcher.py:623-902`，三个 dispatcher 类与方法名均未变）。
>
> **新增点**：`MoEFlexTokenDispatcher` 的后端从 `{deepep, hybridep}` 扩为 **`{deepep, deepepv2, hybridep}`**（`moe_flex_dispatcher_backend`，`transformer_config.py:881`，#4793）。`deepepv2` 走 DeepEP v2 的 **ElasticBuffer** API（`_DeepepV2Manager`，`token_dispatcher.py:1470`），与 v1 的 `Buffer` 隔离、仅支持 `float32` probs；deepep/hybridep 现支持 **thd（sequence-packing）格式**（#4816）。全景级总结见 [[moe_training_optimization_report]] 的「ee3f1ff→232c478d4 增量」与 [[ep_analysis]]。

---

## 2. 核心代码实现分析

### 2.1 MoEAlltoAllTokenDispatcher 工作流程

#### **阶段1: Dispatch Preprocess** (`dispatch_preprocess`)

```python
def dispatch_preprocess(self, hidden_states, routing_map, probs):
    """
    关键操作：
    1. 计算每个expert需要接收多少token (tokens_per_expert)
    2. 计算EP通信的input_splits和output_splits
    3. 对token进行第一次permutation，按expert分组
    """
    # routing_map shape: [num_local_tokens, num_experts]
    # 这是一个bool矩阵，表示每个token是否被路由到某个expert

    # 统计每个expert被分配的token数
    num_local_tokens_per_expert = routing_map.sum(dim=0)  # [num_experts]

    # 计算EP维度的split：每个EP rank需要发送/接收多少token
    self.input_splits = num_local_tokens_per_expert.reshape(
        self.ep_size, self.num_local_experts
    ).sum(axis=1)  # [ep_size]

    # 通过AllGather获取全局token分布信息
    num_global_tokens_per_expert = gather_from_sequence_parallel_region(
        num_local_tokens_per_expert, group=self.tp_ep_group
    )  # [tp_size * ep_size * num_experts]

    # 计算output_splits：当前rank会从其他rank接收多少token
    self.output_splits = num_global_tokens_per_rank[self.tp_rank]  # [ep_size]

    # Permutation 1: 按expert对token重新排列
    permutated_tokens, permuted_probs, self.reversed_mapping = permute(
        hidden_states, routing_map, probs
    )

    return permutated_tokens, permuted_probs
```

**关键数据结构：**
- `input_splits`: 当前rank发送给每个EP rank的token数量
- `output_splits`: 当前rank从每个EP rank接收的token数量
- `routing_map`: [num_tokens, num_experts] 的bool矩阵

#### **阶段2: Token Dispatch** (`token_dispatch`)

```python
def token_dispatch(self, permutated_local_input_tokens, permuted_probs):
    """
    核心All-to-All通信：
    - 每个rank将token发送到包含对应expert的rank
    - 使用variable-size all-to-all通信
    """
    # All-to-All communication across EP ranks
    global_input_tokens = all_to_all(
        self.ep_group,                    # Expert Parallel group
        permutated_local_input_tokens,     # 已经按expert排列的local tokens
        self.output_splits,                # 接收split sizes
        self.input_splits                  # 发送split sizes
    )

    global_probs = all_to_all(
        self.ep_group,
        permuted_probs,
        self.output_splits,
        self.input_splits
    )

    return global_input_tokens, global_probs
```

**All-to-All通信模式：**
```
Rank 0: [tokens for E0, E1, E2, E3] → All-to-All → [tokens from all ranks for E0]
Rank 1: [tokens for E0, E1, E2, E3] → All-to-All → [tokens from all ranks for E1]
Rank 2: [tokens for E0, E1, E2, E3] → All-to-All → [tokens from all ranks for E2]
Rank 3: [tokens for E0, E1, E2, E3] → All-to-All → [tokens from all ranks for E3]
```

#### **阶段3: Dispatch Postprocess** (`dispatch_postprocess`)

```python
def dispatch_postprocess(self, global_input_tokens, global_probs):
    """
    1. TP维度的AllGather（如果TP>1）
    2. 按local expert排序（如果num_local_experts > 1）
    """
    # TP AllGather: 聚合TP维度的tokens
    if self.tp_size > 1:
        global_input_tokens = gather_from_sequence_parallel_region(
            global_input_tokens,
            group=self.tp_group,
            output_split_sizes=self.output_splits_tp
        )

    # Permutation 2: 按local expert排序
    if self.num_local_experts > 1:
        global_input_tokens, global_probs = sort_chunks_by_idxs(
            global_input_tokens,
            self.num_global_tokens_per_local_expert.ravel(),
            self.sort_input_by_local_experts,
            probs=global_probs
        )

    return global_input_tokens, tokens_per_expert, global_probs
```

#### **阶段4: Expert Computation**

```python
# 在experts.py中的GroupedMLP
def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
    """使用GroupedGEMM并行计算多个专家"""
    # 重塑权重为grouped格式
    w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
    w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

    # Grouped GEMM: 一次性计算所有local experts
    fc1_output = grouped_gemm(permuted_local_hidden_states, w1, tokens_per_expert)
    intermediate = activation_func(fc1_output, permuted_probs)
    fc2_output = grouped_gemm(intermediate, w2, tokens_per_expert)

    return fc2_output
```

#### **阶段5: Combine Preprocess** (`combine_preprocess`)

```python
def combine_preprocess(self, hidden_states):
    """
    准备combine操作：
    1. 反向排序（unsort）
    2. TP维度的ReduceScatter
    """
    # Unpermutation 2: 恢复expert的排序
    if self.num_local_experts > 1:
        hidden_states = sort_chunks_by_idxs(
            hidden_states,
            self.num_global_tokens_per_local_expert.T.ravel(),
            self.restore_output_by_local_experts
        )

    # TP ReduceScatter: 聚合TP维度的结果
    if self.tp_size > 1:
        hidden_states = reduce_scatter_to_sequence_parallel_region(
            hidden_states,
            group=self.tp_group,
            input_split_sizes=self.output_splits_tp
        )

    return hidden_states
```

#### **阶段6: Token Combine** (`token_combine`)

```python
def token_combine(self, hidden_states):
    """
    反向All-to-All通信：
    将expert输出发送回原始token所在的rank
    """
    # Reverse All-to-All: 将结果发送回原始位置
    permutated_local_input_tokens = all_to_all(
        self.ep_group,
        hidden_states,
        self.input_splits,   # 注意：这里input_splits和output_splits互换
        self.output_splits
    )

    return permutated_local_input_tokens
```

#### **阶段7: Combine Postprocess** (`combine_postprocess`)

```python
def combine_postprocess(self, permutated_local_input_tokens):
    """
    恢复原始token顺序
    """
    # Unpermutation 1: 恢复原始token顺序
    output = unpermute(
        permutated_local_input_tokens,
        self.reversed_local_input_permutation_mapping,
        restore_shape=self.hidden_shape_before_permute,
        routing_map=self.routing_map
    )

    # 恢复原始shape
    output = output.view(self.hidden_shape)

    return output
```

---

## 3. 代码执行调用流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     MoELayer.forward()                          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Router.forward(hidden_states)                          │
│  ┌───────────────────────────────────────────────────┐          │
│  │ • gating(): Linear(hidden_states) → logits        │          │
│  │ • routing(): TopK selection                       │          │
│  │ • Output: probs, routing_map                      │          │
│  └───────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: router_and_preprocess()                                │
│  ┌───────────────────────────────────────────────────┐          │
│  │ TokenDispatcher.dispatch_preprocess()             │          │
│  │ • Compute tokens_per_expert                       │          │
│  │ • Calculate input_splits, output_splits           │          │
│  │ • Permutation 1: permute(tokens, routing_map)     │          │
│  │ • Save reversed_mapping for unpermute             │          │
│  └───────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: dispatch() - COMMUNICATION                             │
│  ┌───────────────────────────────────────────────────┐          │
│  │ TokenDispatcher.token_dispatch()                  │          │
│  │ • All-to-All across EP group                      │          │
│  │ • Send tokens to expert-owning ranks              │          │
│  │ • Variable-size communication                     │          │
│  └───────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: routed_experts_compute()                               │
│  ┌───────────────────────────────────────────────────┐          │
│  │ TokenDispatcher.dispatch_postprocess()            │          │
│  │ • AllGather across TP group (if TP>1)             │          │
│  │ • Permutation 2: sort_chunks_by_experts           │          │
│  │                                                    │          │
│  │ Experts.forward()                                 │          │
│  │ • GroupedGEMM: parallel expert computation        │          │
│  │ • fc1 → activation → fc2                          │          │
│  │                                                    │          │
│  │ TokenDispatcher.combine_preprocess()              │          │
│  │ • Unpermutation 2: restore expert order           │          │
│  │ • ReduceScatter across TP group (if TP>1)         │          │
│  └───────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: combine() - COMMUNICATION                              │
│  ┌───────────────────────────────────────────────────┐          │
│  │ TokenDispatcher.token_combine()                   │          │
│  │ • Reverse All-to-All across EP group              │          │
│  │ • Send expert outputs back to original ranks      │          │
│  └───────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: Final Output                                           │
│  ┌───────────────────────────────────────────────────┐          │
│  │ TokenDispatcher.combine_postprocess()             │          │
│  │ • Unpermutation 1: unpermute to original order    │          │
│  │ • Reshape to original shape                       │          │
│  │ • Add shared expert output (if any)               │          │
│  └───────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Tensor流转示意图

### 4.1 单机4卡示例（EP=4, num_experts=4, topk=2）

假设：
- 4个GPU (Rank 0-3)
- 4个专家 (E0-E3)，每个GPU持有1个专家
- 8个tokens (T0-T7)
- TopK=2

#### **初始状态：Router输出**

```
Rank 0, 1, 2, 3 都有相同的tokens:
┌────────────────────────────────────────┐
│ Tokens: [T0, T1, T2, T3, T4, T5, T6, T7]│
└────────────────────────────────────────┘

Router输出的routing_map (shape: [8, 4]):
       E0  E1  E2  E3
    ┌──────────────────┐
T0  │ 1   0   1   0   │  → T0选择E0和E2
T1  │ 0   1   0   1   │  → T1选择E1和E3
T2  │ 1   0   0   1   │  → T2选择E0和E3
T3  │ 0   1   1   0   │  → T3选择E1和E2
T4  │ 0   0   1   1   │  → T4选择E2和E3
T5  │ 1   1   0   0   │  → T5选择E0和E1
T6  │ 0   1   1   0   │  → T6选择E1和E2
T7  │ 1   0   0   1   │  → T7选择E0和E3
    └──────────────────┘
```

#### **Step 1: Dispatch Preprocess - Permutation 1**

每个rank对本地tokens按expert分组排列：

```
Rank 0 (before All-to-All):
┌─────────────────────────────────────────────────────┐
│ For E0: [T0, T2, T5, T7]  (4 tokens)                │
│ For E1: [T1, T3, T5, T6]  (4 tokens)                │
│ For E2: [T0, T3, T4, T6]  (4 tokens)                │
│ For E3: [T1, T2, T4, T7]  (4 tokens)                │
└─────────────────────────────────────────────────────┘

计算input_splits和output_splits:
input_splits = [4, 4, 4, 4]   # 发送给每个rank的token数
output_splits = [4, 4, 4, 4]  # 从每个rank接收的token数
```

#### **Step 2: Token Dispatch - All-to-All Communication**

```
All-to-All 通信矩阵 (每个格子表示一个token传输):

        To Rank:
         0    1    2    3
From  ┌─────────────────────┐
  0   │ T0  T1  T0  T1     │
      │ T2  T3  T3  T2     │
      │ T5  T5  T4  T4     │
      │ T7  T6  T6  T7     │
      └─────────────────────┘

零冗余体现：
- 每个token的副本只在需要的expert所在rank之间传输
- Rank 0只接收需要E0处理的tokens
- Rank 1只接收需要E1处理的tokens
- 没有全量广播，避免了冗余数据传输
```

#### **Step 3: After All-to-All (每个rank收到的数据)**

```
Rank 0 (持有E0):
┌────────────────────────────────────┐
│ 接收到需要E0处理的所有tokens:       │
│ [T0, T2, T5, T7] × hidden_size     │
└────────────────────────────────────┘

Rank 1 (持有E1):
┌────────────────────────────────────┐
│ 接收到需要E1处理的所有tokens:       │
│ [T1, T3, T5, T6] × hidden_size     │
└────────────────────────────────────┘

Rank 2 (持有E2):
┌────────────────────────────────────┐
│ 接收到需要E2处理的所有tokens:       │
│ [T0, T3, T4, T6] × hidden_size     │
└────────────────────────────────────┘

Rank 3 (持有E3):
┌────────────────────────────────────┐
│ 接收到需要E3处理的所有tokens:       │
│ [T1, T2, T4, T7] × hidden_size     │
└────────────────────────────────────┘
```

#### **Step 4: Expert Computation (本地计算)**

```
每个rank独立计算其专家：

Rank 0: Expert0(T0, T2, T5, T7) → output0
Rank 1: Expert1(T1, T3, T5, T6) → output1
Rank 2: Expert2(T0, T3, T4, T6) → output2
Rank 3: Expert3(T1, T2, T4, T7) → output3

零冗余体现：
- 每个expert的参数只存储在一个GPU上
- 没有expert参数的复制
- 显存消耗 = 总expert参数 / EP_size
```

#### **Step 5: Token Combine - Reverse All-to-All**

将expert输出发送回原始token所在的rank：

```
Reverse All-to-All 通信矩阵:

        To Rank:
         0    1    2    3
From  ┌─────────────────────────┐
  0   │ out0 out0 out0 out0    │  (E0的输出返回到所有需要它的rank)
  1   │ out1 out1 out1 out1    │  (E1的输出返回)
  2   │ out2 out2 out2 out2    │  (E2的输出返回)
  3   │ out3 out3 out3 out3    │  (E3的输出返回)
      └─────────────────────────┘
```

#### **Step 6: Combine Postprocess - Unpermute**

每个rank恢复原始token顺序并加权组合：

```
Rank 0 最终输出:
T0 = prob(E0)*out0[T0] + prob(E2)*out2[T0]
T1 = prob(E1)*out1[T1] + prob(E3)*out3[T1]
T2 = prob(E0)*out0[T2] + prob(E3)*out3[T2]
...
T7 = prob(E0)*out0[T7] + prob(E3)*out3[T7]
```

---

## 5. 零冗余通信的关键优化技术

### 5.1 Variable-Size All-to-All

```python
# 不同于固定大小的All-to-All，使用变长通信
def all_to_all(group, input_tensor, output_split_sizes, input_split_sizes):
    """
    output_split_sizes: 从每个rank接收的数据大小 (可变)
    input_split_sizes: 发送到每个rank的数据大小 (可变)
    """
    # 根据实际token分布动态调整通信量
    # 避免padding带来的冗余传输
```

### 5.2 Fused Permutation

```python
# 融合permutation操作，减少kernel launch开销
permuted_input, permuted_probs, sorted_indices = permute(
    hidden_states,
    routing_map,
    probs=probs,
    fused=True  # 使用fused kernel
)
```

### 5.3 DeepEP优化（最新）

```python
# DeepEP将permutation和communication融合
hidden_states, handle = fused_dispatch(
    hidden_states,
    token_indices,
    token_probs,
    num_experts,
    group,
    async_finish=True  # 异步通信
)

# 一个kernel完成：permute + all-to-all
```

> [!update] 2026-06-16 · dev@232c478d4
> 上方为 **DeepEP v1** 路径（`fused_dispatch`/`fused_combine`，`fused_a2a.py`）。新增 **DeepEP v2** 后端基于 **ElasticBuffer**：先 `get_elastic_buffer(group, num_max_tokens_per_rank, hidden, num_topk)`（`fused_a2a.py:90`）申请/复用弹性通信缓冲，再调 `deepepv2_dispatch(buffer, …)` / `deepepv2_combine(buffer, …)`（均为带 autograd 的 `torch.autograd.Function`），仅支持 `float32` probs（#4793）。
>
> 此外，原生 `MoEAlltoAllTokenDispatcher` 的 `all_to_all` 现新增 `use_nccl_stream` 形参（`token_dispatcher.py:703/716/862`）——当 MoE 带 shared-experts 时置 True，把 A2A 放到独立 NCCL 流上与 shared-expert GEMM 重叠；并可配 `high_priority_a2a_comm_stream`（`transformer_config.py:686`，#4694）将该通信流提到 CUDA 高优先级。详见 [[ep_analysis]] 与 [[megatron_comm_overlap_analysis]]。

---

## 6. 通信量与显存分析

### 6.1 通信量对比

假设：
- S: sequence length
- B: batch size
- H: hidden size
- E: number of experts
- EP: expert parallel size
- K: topk

**传统方法（全量复制）：**
```
显存: E × expert_params_per_expert
通信: 0 (无需通信，但显存占用巨大)
```

**零冗余方法（All-to-All）：**
```
显存: (E / EP) × expert_params_per_expert
通信量 (Forward):
  - Dispatch All-to-All: S×B×H / TP
  - TP AllGather (if TP>1): S×B×H / TP × (TP-1)/TP
  - Combine All-to-All: S×B×H / TP
总通信: ≈ 2×S×B×H (假设TP=1)

显存节省: EP倍
通信增加: 2×S×B×H (相比无通信方案)
```

### 6.2 实际性能权衡

| Scenario | Expert Memory | Communication | Recommendation |
|----------|---------------|---------------|----------------|
| EP=1 (单机) | 100% | 0 | 小模型可用 |
| EP=8 (多机) | 12.5% | Medium | 大模型推荐 |
| EP=8 + TP=4 | 3.125% | High | 超大模型 |

---

## 7. 关键代码位置索引

```
megatron/core/transformer/moe/
├── moe_layer.py              # MoE层主入口
│   └── MoELayer.forward()    # 主要前向传播逻辑
│
├── token_dispatcher.py        # Token分发器实现
│   ├── MoEAlltoAllTokenDispatcher    # All-to-All方案
│   ├── MoEAllGatherTokenDispatcher   # AllGather方案
│   └── MoEFlexTokenDispatcher        # DeepEP方案
│
├── router.py                  # 路由器实现
│   └── TopKRouter.routing()  # TopK路由逻辑
│
├── experts.py                 # 专家网络实现
│   ├── GroupedMLP            # 分组GEMM优化
│   ├── TEGroupedMLP          # TE加速版本
│   └── SequentialMLP         # 顺序执行版本
│
└── moe_utils.py              # 工具函数
    ├── permute()             # Token排列
    ├── unpermute()           # Token恢复
    └── switch_load_balancing_loss_func()  # 负载均衡
```

> [!update] 2026-06-16 · dev@232c478d4
> 该目录在 ee3f1ff 之后新增/变化的关键文件：
> - `fused_a2a.py` — DeepEP/HybridEP 融合通信后端，含 v1（`fused_dispatch`/`fused_combine`）、**DeepEP v2 ElasticBuffer**（`get_elastic_buffer`/`deepepv2_dispatch`/`deepepv2_combine`，#4793）、HybridEP（`hybrid_ep_dispatch`/`hybrid_ep_combine`）。
> - `paged_stash.py` — MoE 路由专家激活的页式暂存（虚拟内存式管理，#4247），开关 `moe_paged_stash`。
> - `moe_logging.py` — 独立的 MoE 指标采集/规约/可视化模块（`MoEMetricsTracker`，#3431），原先散落在 `moe_utils.py` 的日志逻辑迁出于此。

---

## 8. 使用示例

```bash
# 启用零冗余MoE训练
python pretrain_gpt.py \
    --num-experts 8 \
    --expert-model-parallel-size 8 \
    --moe-router-topk 2 \
    --moe-token-dispatcher-type alltoall \
    --moe-grouped-gemm \
    --moe-permute-fusion \
    --use-distributed-optimizer \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4
```

关键参数说明：
- `--expert-model-parallel-size 8`: 将8个expert分布到8个GPU（零冗余）
- `--moe-token-dispatcher-type alltoall`: 使用All-to-All通信
- `--moe-permute-fusion`: 融合permutation操作
- `--moe-grouped-gemm`: 使用GroupedGEMM加速

---

## 9. 总结

Megatron-LM的MoE零冗余通信实现通过以下技术实现了显存和通信的最优平衡：

### 核心技术
1. **Expert Parallelism**: 将专家参数分布到多个GPU，每个GPU只存储部分专家
2. **Variable-Size All-to-All**: 根据路由结果动态调整通信量，只传输必要的tokens
3. **Token Permutation**: 通过高效的排列/恢复操作减少通信复杂度
4. **Grouped GEMM**: 并行计算多个local experts，提升计算效率
5. **Communication Fusion**: 融合permutation和communication操作，减少开销

### 性能优势
- **显存节省**: 相比全量复制，节省 EP 倍显存
- **扩展性强**: 可扩展到数百个专家和数千个GPU
- **通信高效**: 通过All-to-All实现点对点通信，避免冗余广播
- **计算优化**: GroupedGEMM和异步通信overlap进一步提升性能

### 适用场景
- 大规模MoE模型训练（如Mixtral 8x7B, DeepSeek-V3）
- 多节点分布式训练
- 显存受限但需要大容量模型的场景


## Related Pages

- [[02_engineering/02_train_frameworks/megatron-lm/index]]
- [[Megatron-LM_Distributed_Parallel_Exam]]
- [[llm_initiliaze_analysis]]
- [[mHC]]
- [[moe_training_optimization_report]]
- [[ep_analysis]]
- [[megatron_comm_overlap_analysis]]
