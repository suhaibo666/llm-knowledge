---
title: "DeepSeek-V4 Tensor Parallel 边界案例"
---

# DeepSeek-V4 Tensor Parallel 边界案例

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）。
> **本页定位**：本页不是第二份 TP 教程，只回答 DeepSeek-V4 Hybrid Attention 接入 Megatron 后，哪些模块实际被 TP 切分、哪些只是保留了 TP 接口，以及为什么当前实现强制 `TP=1`。
> **先修**：先读 [[12_megatron_tp_analysis]] 理解 Column/Row Parallel，再读 [[10_megatron_model_structure_analysis]] 理解 spec 装配；MoE 的 EP/ETP 语义见 [[14_megatron_ep_analysis]]。
> **最近更新**：2026-09-03。删除通用 TP/overlap 重复教材和过期迁移叙事，保留 DSv4 专有实现边界。

---

## 1. 先给结论：接口长得像 TP，不等于当前执行了 TP

`DSv4HybridAttention.__init__` 对 `pg_collection.tp` 的大小做硬断言：只能等于 1（`megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:90-99`）。因此要分开看两件事：

| 层次 | 冻结基线中的事实 | 不应误读成 |
|---|---|---|
| Attention 的 TP 进程组 | 大小必须为 1 | “只是不推荐 TP>1” |
| `q_down_proj`、Compressor、Indexer | 显式 `duplicated`，或直接不传 TP group | 它们已按 hidden/head 维分片 |
| `q_up_proj`、`kv_proj`、输出投影 | 使用 Column/Row Parallel 形状接口，但所在 TP group 只有一个 rank | 当前存在跨 rank 的 TP collective |
| MoE | 另有 EP/ETP 与 dispatcher 通信 | Attention `TP=1` 等于“整模型只能单卡” |

这里的 `TP=1` 只约束 DSv4 Hybrid Attention 收到的 tensor-parallel group；DP、PP、CP、EP 仍是不同并行轴。进程组怎样组合归 [[17_megatron_parallelism_orchestration_analysis]]，不要从本页的 attention 断言外推整个训练拓扑。

## 2. 从 spec 到构造器：硬边界在哪里生效

GPT 的实验性 attention spec 把 Compressor、CSA Indexer、Compressed Sparse Attention 和 `DSv4HybridSelfAttention` 组装成一棵模块树，并为不同线性层保留 Column/Row Parallel 后端接口（`megatron/core/models/gpt/experimental_attention_variant_module_specs.py:190-246`）。但真正决定“能否多 rank TP”的不是接口名字，而是实例化后的第一道运行时断言：

1. `DSv4HybridAttention` 取得 `pg_collection.tp`；
2. 构造器立即要求 group size 等于 1；
3. 随后才创建 Q/KV/output 等投影。

> [!note] 分析重建，不是源码自陈动机
> 源码明确给出 TP=1 guard 和 guard 之后的参数布局，却没有注释“为什么选择 TP=1”。本页对扩展 TP 的证明义务（包括 §4 的四项列表）只是从这些可观察约束重建，不把它冒充为作者原话或冻结源码的未来承诺。

所以这不是“尚未给某个投影挑好切分维度”的软缺口，而是冻结基线公开执行契约的一部分。若移除断言，还必须重新证明本页后续列出的 duplicated 参数、grouped output projection、mHC 梯度同步和 CSA 数据布局在多 rank 下都正确；仅把 `tp_group` 改成多卡并不能闭合这条证明链。

## 3. Attention 参数所有权账本

### 3.1 Q/KV 投影

`DSv4HybridSelfAttention` 的四个关键接口不是同一种所有权：

| 模块 | 构造事实 | 当前 TP 含义 |
|---|---|---|
| `linear_q_down_proj` | 只接受 `TELinear`；设置 `parallel_mode='duplicated'` 且 `tp_group=None` | 每个参与实例持有完整投影，不走 TP group |
| `linear_q_up_proj` | `gather_output=False`，传入 `pg_collection.tp` | 保留 Column Parallel 接口；因 group size=1，没有跨 rank gather |
| `linear_kv_proj` | `gather_output=False`，传入 `pg_collection.tp` | 同上 |
| `linear_proj` | `input_is_parallel=True`，传入 `pg_collection.tp` | 保留 Row Parallel 输出接口；当前仍为单 rank |

前三项构造位于 `megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:520-570`，输出接口位于 `megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:188-202`。这张表解释了一个常见误判：源码里出现 `gather_output`、`input_is_parallel` 或 `tp_comm_buffer_name`，只能证明模块遵循并行线性层 API，不能绕过页首的 `TP=1` 断言。

### 3.2 grouped output projection

`linear_o_group_proj` 不是 parallel-linear 模块，而是直接创建的 `torch.nn.Parameter`；其输入宽度先按 `o_groups` 分组，随后再送入 `linear_proj`。构造器要求 `num_attention_heads * v_head_dim` 能被 `o_groups` 整除（`megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:171-202`）。

该参数使用 `params_dtype` 和 `init_method` 初始化，但这两个字段不是 DSv4 的私有配置：dtype 契约归 [[23_megatron_precision_cudagraph_fusion_analysis]]，通用初始化契约归 [[10_megatron_model_structure_analysis]]。

## 4. Compressor 与 Indexer：源码明确选择 duplicated

CSA Compressor 的 `linear_wkv` 与 `linear_wgate` 都设置 `parallel_mode='duplicated'`（`megatron/core/transformer/experimental_attention_variant/csa.py:1064-1088`）。CSA Indexer 的 Q 投影 `linear_wq_b` 与权重投影 `linear_weights_proj` 也采用同一模式；后者还刻意在 FP8 初始化上下文之外构建，以保持 BF16（`megatron/core/transformer/experimental_attention_variant/csa.py:1477-1506`）。

这四个 locator 证明的是“当前参数复制语义”，不是“理论上永远不能切”。若未来引入 TP>1，必须同时定义：

- 投影权重沿哪一维分片；
- top-k 索引需要局部 head 还是全局 head；
- compressed K/KV 的 collective 在 TP 与 CP 两个 group 上按什么顺序发生；
- 反向梯度在哪个 group 归并。

冻结实现没有给出这些多 rank TP 契约，因此本页不把“存在 parallel-linear 接口”包装成已经支持。

## 5. mHC：用同步标记补非 TP-aware 参数，而不是偷偷切权重

`HyperConnectionModule` 的动态映射使用普通 `nn.Linear`，另外持有 `alpha_pre`、`alpha_post`、`alpha_res` 和 bias；这些并非 Column/Row Parallel 参数（`megatron/core/transformer/hyper_connection.py:235-266`）。当 `sequence_parallel` 开启时，初始化逻辑给这些参数加上 `sequence_parallel=True`，让框架在梯度收口阶段识别它们需要同步（`megatron/core/transformer/hyper_connection.py:306-319`）。

因此这里的设计是“完整参数 + 显式梯度同步标记”，不是新的 TP 切分规则。一般性的 sequence-parallel 梯度收口归 [[12_megatron_tp_analysis]]；本节只记录 DSv4/mHC 模块为何不能仅凭 `nn.Linear` 判断它绕开了同步。

## 6. MoE：Attention TP 与 expert TP 必须分账

DSv4 模型还可以包含 shared/routed experts，但它们不改变 attention 的 `TP=1` 断言：

- `SharedExpertMLP` 把 `moe_shared_expert_intermediate_size` 写入复制后的 config，再以 `pg_collection.tp` 构造标准 MLP（`megatron/core/transformer/moe/shared_experts.py:123-140`）。共享专家 overlap 会关闭线性层自带的 TP AG/RS，并把执行拆成 dispatcher 必须按序调用的一组阶段（`megatron/core/transformer/moe/shared_experts.py:158-185`）；完整机制归 [[14_megatron_ep_analysis]]。
- 当 `use_transformer_engine_op_fuser=True` 时，GroupedMLP 构造器要求 `_is_fused_impl_supported()` 为真；expert TP group 大小超过 1 是明确的失败原因，因此会触发 assert，并不会自动换回 non-fused 实现（`megatron/core/transformer/moe/experts.py:276-281`、`megatron/core/transformer/moe/experts.py:372-418`）。要走 non-fused 路径必须由配置方关闭 op-fuser；这不等于 routed expert 整体不能使用 ETP。
- 只有在既无显式 cached expert-TP world-size override、也没有已初始化的 expert-TP group 时，兼容函数才回退报告普通 TP world size（`megatron/core/parallel_state.py:1929-1938`）。这是一条兼容语义，不是证明两个轴永远相同。

配置 `moe_shared_expert_intermediate_size` 的字段契约已归 [[14_megatron_ep_analysis]]；本页只保留它与 DSv4 attention TP 边界相遇时的解释。

## 7. 通信账本：当前 attention 没有 TP collective，其他轴仍可能通信

| 路径 | 当前跨 rank 通信 | owner |
|---|---|---|
| DSv4 attention 的 Q/KV/output TP | 无；TP group size 为 1 | 本页 |
| Compressor / Indexer TP | 无；duplicated | 本页 |
| mHC 参数梯度 | 由 sequence-parallel 标记进入通用梯度同步逻辑；在单 rank TP group 上不会产生跨 rank TP 流量 | [[12_megatron_tp_analysis]] |
| CP 的 boundary P2P 与 compressed-state gather | 有，取决于 CP size | [[35_deepseek_v4_context_parallel_analysis]] |
| MoE token dispatch、EP/ETP | 有，取决于 MoE 拓扑与 dispatcher | [[14_megatron_ep_analysis]] |
| 多轴同时 ready 时的 stream/SM/显存竞争 | 可能有 | [[20_megatron_comm_overlap_analysis]] |

这也是为什么旧版页面的通用 “bulk vs pipelined TP overlap” 教材被删除：它属于 [[12_megatron_tp_analysis]]，而在本页的 attention 主路径上没有多 rank TP 通信可供 overlap。

## 8. 硬边界与失败方式

| 前提 | 源码落点 | 破坏后的表现 |
|---|---|---|
| attention TP group size 必须为 1 | `megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:90-92` | 构造时 assert |
| 不支持 core-attention checkpoint 与 QKV linear offload | `megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:94-99` | 构造时 assert |
| heads×value-head-dim 必须被 `o_groups` 整除 | `megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:171-186` | 构造时 assert |
| forward 不接收外部 RoPE、attention bias、flash-decoding 参数 | `megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:240-251` | forward 时 assert |
| 当前不支持 inference context/params | `megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:252-254`、`megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:603-609` | forward 时 assert |
| CP>1 必须走 THD contiguous partition | `megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:259-281` | ValueError；细节归 35 |
| 启用 TE op-fuser 时，fused GroupedMLP 的 expert TP 必须为 1 | `megatron/core/transformer/moe/experts.py:276-281`、`megatron/core/transformer/moe/experts.py:399-418` | 构造时 assert；源码只记录不支持原因，不自动回退。关闭 op-fuser 后是否采用 non-fused 路径由配置方决定 |

这些约束分属不同子系统。诊断时先定位是哪一个 process group、哪一个模块抛错，再决定去 TP、CP、MoE 或推理 owner 页，避免把所有失败都归因于“DSv4 只能 TP=1”。

## 9. 本轮收缩与配置所有权

旧页各部分按以下规则处理：

- **保留**：TP=1、duplicated projections、grouped output、mHC、MoE 交界和硬约束，都是 DSv4-specific。
- **迁出**：通用 TP 分片与 bulk/pipelined overlap 归 12；跨轴资源竞争归 20。
- **删除**：没有冻结源码支撑的未来 TP 方案、固定通信收益数字和长篇旧基线迁移叙事。
- **配置归位**：`init_method`、`output_layer_init_method`、`attention_dropout` 归 10；`params_dtype` 归 23；`moe_shared_expert_intermediate_size` 归 14。案例页只在机制需要时引用，不再充当这些通用字段的 coverage owner。

历史上“案例页兼做通用教程”和旧基线审计的结果保留在 [[changelog]]，不再占用首次阅读主线。

## Related Pages

- [[12_megatron_tp_analysis]] —— Column/Row Parallel、sequence parallel 与 TP overlap 的唯一机制 owner。
- [[10_megatron_model_structure_analysis]] —— DSv4 spec 装配，以及初始化与 attention dropout 配置契约。
- [[14_megatron_ep_analysis]] —— shared/routed expert、EP/ETP、dispatcher 与 shared-expert overlap。
- [[35_deepseek_v4_context_parallel_analysis]] —— 同一 attention 在 CP 轴上的两阶段通信与硬边界。
- [[20_megatron_comm_overlap_analysis]] —— TP/CP/EP/DP/PP 同时运行时的资源竞争与诊断。
- [[13_deepseek_v4_analysis]] —— DeepSeek-V4 模型结构背景；本页只解释 Megatron 接入边界。
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]] —— 返回本域索引。
