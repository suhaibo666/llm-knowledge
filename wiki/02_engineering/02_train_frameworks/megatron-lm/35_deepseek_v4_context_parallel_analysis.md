---
title: "DeepSeek-V4 Context Parallel 两阶段实现案例"
---

# DeepSeek-V4 Context Parallel 两阶段实现案例

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）。
> **本页定位**：本页不是第二份 CP 教程，只解释 DSv4 Hybrid Attention 在 Megatron 中怎样用“左边界 P2P + 压缩状态 AllGather”补齐本地 CSA/HCA 计算，以及它对 THD、contiguous partition 和 Dynamic CP 的特殊要求。
> **先修**：CP group、`cp_comm_type` 与 hierarchical CP 见 [[13_megatron_cp_analysis]]；packed sequence 与 Dynamic CP 调度见 [[29_megatron_packed_dataset_dynamic_cp_analysis]]。
> **最近更新**：2026-09-03。以当前已落地的 `_forward_thd_cp` 为主线，删除旧基线“尚未实现/不支持 Dynamic CP”的失效长篇审计。

---

## 1. 当前主线：两类依赖、两种通信

DSv4 的压缩稀疏注意力不能只拿本 rank 的 token 直接算完。冻结基线把跨 rank 依赖拆成两段：

| 阶段 | 搬什么 | 通信方式 | 为什么不能合成一句“做一次 CP gather” |
|---|---|---|---|
| Stage 1 | 相邻 rank 的固定左边界 hidden rows；反向把对应梯度送回 owner | 邻接 P2P，当前实现发起后立即 wait | 压缩窗口跨过 contiguous rank 边界，需要的是邻居尾部，不是完整全局序列 |
| Stage 2 | 各 rank 产生的 compressed Indexer-K 与 compressed attention-KV | CP group AllGather；有 Indexer 时异步发起，并与本地 Q/weight projection 重叠 | 稀疏选择和 attention 要看到跨 rank 的压缩块；搬的是压缩状态，不是原始完整 KV |

> [!note] 分析重建，不是源码自陈动机
> 两阶段的执行事实、payload、`global_start` 计算与 contiguous guards 来自下列 locators；“为什么不能合成一次 gather”以及 §5 对 contiguous 取舍的解释，是本页根据数据依赖与 guard 重建的 rationale，不是源码作者写下的完整设计动机。

入口 `CompressedSparseAttention.forward` 在 CP>1 且 THD 时分派到 `_forward_thd_cp`（`megatron/core/transformer/experimental_attention_variant/csa.py:2094-2120`）。这条调用链已经闭合，因此当前页面不再把两阶段 CP 描述成论文与代码之间的 gap。

## 2. Stage 1：左边界交换及其反向所有权

`exchange_cp_boundary_hidden` 先把 hidden state 展平，再计算：

- 当 `compress_ratio == 4` 时，压缩依赖宽度 `d_comp=8`；
- 其他 `compress_ratio>1` 时，`d_comp=compress_ratio`；
- 最终 `d_window=max(csa_window_size, d_comp)`。

该规则与 reshape 位于 `megatron/core/transformer/experimental_attention_variant/csa_utils/cp_utils.py:191-202`。自定义 autograd function 的数据所有权很明确（`megatron/core/transformer/experimental_attention_variant/csa_utils/cp_utils.py:124-188`）：

1. 非首 rank 从左邻居接收 `d_window` 行；
2. 非末 rank 把自己的尾部 `d_window` 行发给右邻居；
3. `batch_isend_irecv` 返回的 request 逐个 `wait` 后才返回 boundary；
4. backward 反向发送 `grad_boundary`，右邻居贡献的梯度累加回本 rank 尾部 owner。

因此 Stage 1 是一个带 autograd 所有权的同步边界修补，不是可与整段 attention 自由重排的全局 Ring。当前实现还要求 `local_rows >= d_window`；否则在通信之前直接抛 `RuntimeError`（同文件 `:135-139`）。

## 3. Stage 2：先发 compressed gather，再做本地投影

`_forward_thd_cp` 先构造本地与 boundary 共同参与的 compressor 输入，然后为每个完整 `compress_ratio` token group 产生一条 compressed row；每条序列不足一个完整 group 的尾部用 floor 规则丢弃（`megatron/core/transformer/experimental_attention_variant/csa.py:2537-2605`）。

有 Indexer 时，执行顺序是：

1. 产生本地 compressed Indexer-K，并先异步发起它的 CP AllGather（`megatron/core/transformer/experimental_attention_variant/csa.py:2639-2653`）；
2. 产生本地 compressed attention-KV，再异步发起第二个 AllGather（`megatron/core/transformer/experimental_attention_variant/csa.py:2655-2680`）；
3. 两个 collective 在所有 rank 上保持相同入队次序；其飞行期间执行本地 Indexer Q 与 weight projection，并应用 CP-aware RoPE（`megatron/core/transformer/experimental_attention_variant/csa.py:2682-2720`）；
4. top-k 前等待 Indexer-K gather；attention 取 compressed KV 前再等待第二个 gather（`megatron/core/transformer/experimental_attention_variant/csa.py:2722-2751`）。

如果没有 Indexer，compressed KV 改走同步 `gather_from_sequence_parallel_region`（`megatron/core/transformer/experimental_attention_variant/csa.py:2752-2755`）。所以“Stage 2 一定异步”也不准确：异步 overlap 是有 Indexer 的分支行为。

## 4. Dynamic CP：每个 microbatch 选组，返回前恢复

当前 DSv4 Hybrid forward 会先保存静态 CP group。若 `packed_seq_params.local_cp_size` 不为空，它要求同时提供 `packed_seq_params.cp_group`，把本次 microbatch 的 group 写入 `pg_collection.cp`；完成输出投影后恢复原 group（`megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:256-281`、`megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:470-480`）。

CSA 内层也保存并恢复收到的 group，并按这个动态 group 的 size 决定是否进入 `_forward_thd_cp`（`megatron/core/transformer/experimental_attention_variant/csa.py:2094-2120`）。因此冻结基线的准确结论是：

- DSv4 **支持**由 `PackedSeqParams` 携带的 Dynamic CP group；
- 它不负责如何为样本挑选 group，调度与 packing owner 是 [[29_megatron_packed_dataset_dynamic_cp_analysis]]；
- group 选择必须在进入 attention 前完成，且所有 participating ranks 必须得到一致 collective 顺序。

## 5. contiguous partition 与 CP-aware RoPE

DSv4 CP 假定每个 rank 持有全局序列的一段连续区间。`_forward_thd_cp` 用 `global_start = cp_rank * l_local` 建立本地行到全局位置的映射（`megatron/core/transformer/experimental_attention_variant/csa.py:2552-2569`），本地 Indexer Q 的 fused/unfused RoPE 都显式接收这个 `global_start`（`megatron/core/transformer/experimental_attention_variant/csa.py:2690-2716`）。

这解释了为什么入口不是只检查 `qkv_format='thd'`，还要求 `cp_partition_mode='contiguous'`：边界交换、compressed block 映射和 RoPE 位置三者共用连续 rank-major 布局。一般 CP 的 zigzag/contiguous 选择及转换职责归 [[13_megatron_cp_analysis]]；本页只记录 DSv4 为什么锁定后者。

## 6. 通信与 overlap 账本

| 阶段 | payload 随什么增长 | 当前重叠窗口 | 明确不能推出的结论 |
|---|---|---|---|
| 左边界 P2P | `d_window × hidden`；`d_window` 由窗口与压缩比共同决定，不随本 rank 整段长度线性增长 | forward/backward 都在返回前 wait | 不能写成完全被计算隐藏 |
| Indexer-K AllGather | 各 rank 的 compressed K rows | 与 attention compressor、本地 Indexer Q/weight projection 重叠 | 不能从压缩比直接推出固定端到端加速 |
| attention-KV AllGather | 各 rank 的 compressed KV rows | 与本地 Indexer Q/weight projection 重叠 | 不能写成“搬完整 KV”，也不能承诺固定 128× 通信缩减 |
| 无 Indexer 分支 | compressed KV rows | 同步 gather | 不能套用有 Indexer 分支的异步时间线 |

这张表是 DSv4 路径的局部数据依赖。多轴同时抢 stream、SM 和显存时如何判断实际 overlap，归 [[20_megatron_comm_overlap_analysis]]；通用 CP Ring/AllGather/A2A 的通信代数归 [[../../../01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis|20_ring_attention_and_context_parallel_analysis]]。

## 7. 硬边界与回退

先区分 live path：`_forward_thd_cp` 会为 Indexer 与 attention compressor 显式传入 `compressed_group_ids`（`megatron/core/transformer/experimental_attention_variant/csa.py:2611-2620`、`:2631-2637`、`:2655-2662`）。因此下游 `_forward_thd` 进入 `pre_grouped=True` 分支并直接走 eager compressor，根本不会尝试 fused helper（`megatron/core/transformer/experimental_attention_variant/csa.py:1237`、`:1270-1325`）。下面的 fused envelope 只约束**非 pre-grouped THD** 路径，不能解读成“CP 路径满足 SM100 等条件即可 fused”。

| 前提 | 源码落点 | 失败或回退 |
|---|---|---|
| CP>1 必须为 THD | `megatron/core/transformer/experimental_attention_variant/csa.py:2101-2104` | `ValueError` |
| CP>1 必须 contiguous | `megatron/core/transformer/experimental_attention_variant/csa.py:2105-2109` | `ValueError`；partition 转换明确由 CSA 外部负责 |
| contiguous CP 必须来自 sequence-packing scheduler | `megatron/core/transformer/transformer_config.py:1654-1665` | 配置校验期 `ValueError` |
| DSv4+CP 拒绝 zigzag | `megatron/core/transformer/transformer_config.py:1678-1708` | 配置校验期 `ValueError` |
| 本地 token 行数必须不少于 `d_window` | `megatron/core/transformer/experimental_attention_variant/csa_utils/cp_utils.py:128-139` | 边界通信前 `RuntimeError` |
| THD CP 当前只支持 self-attention，且 boundary hidden/KV 必须齐全 | `megatron/core/transformer/experimental_attention_variant/csa.py:2557-2574` | `RuntimeError` |
| compressed row 只来自完整 ratio group | `megatron/core/transformer/experimental_attention_variant/csa.py:2594-2605` | 每序列尾部不足 ratio 的部分不生成 compressed row |
| 非 pre-grouped THD 的 fused compressor fast path 必须同时通过 enable/CUDA、SM100 + frontend、ratio/coff（ratio 128 另限 head dim 128/512）、BF16 KV/score + FP32 APE、非 deterministic/compile、shape/capacity/32-bit offset 等 gate | `megatron/core/transformer/experimental_attention_variant/csa_utils/fused_compressor.py:140-162`、`megatron/core/transformer/experimental_attention_variant/csa_utils/fused_compressor.py:221-284` | 任一 gate 未通过时 helper 返回 None、由调用者保留 eager；支持设备缺 frontend 时会 warning 一次，并非所有回退都静默 |
| DSv4 Hybrid inference 当前不支持 | `megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:240-254` | forward assert |

另有一条组合约束：contiguous CP 与 `MTP + TP>1 + sequence_parallel` 的 token-side padding mask 存在已知 bug，配置层暂时拒绝该组合（`megatron/core/transformer/transformer_config.py:1666-1676`）。它不是 CSA 数学限制，排障时应区分。

## 8. 已纠正的历史结论

旧页以更早源码审计得出“两阶段 CP 尚未实现”和“Dynamic CP 不支持 MLA/DSv4”。这两个结论在当前冻结基线都已失效：

- `CompressedSparseAttention.forward → _forward_thd_cp` 已有明确分派；
- Stage 1 boundary autograd 与 Stage 2 compressed gather 都有可达实现；
- DSv4 outer attention 与 CSA inner attention 都读取、使用并恢复 dynamic CP group。

本轮不再把被推翻的代码摘录保留成 100 多行附录，以免读者先记住错误再读勘误。历史审计“当时的结论及后来为何失效”只在 [[changelog]] 留档；当前机制页只陈述冻结基线事实。

## 9. 通用机制与配置所有权

旧页各部分已经按内容类型归位：

- **DSv4-specific，留在本页**：两阶段 CSA/HCA CP、boundary autograd、compressed gather、CP-aware RoPE、THD contiguous 约束、Dynamic CP 接入。
- **generic unique delta，迁到 owner**：`hierarchical_context_parallel_sizes` 的 N 级进程组构造迁入 [[13_megatron_cp_analysis]]。
- **generic duplicate，删除并指向 owner**：四种 `cp_comm_type`、Native/TE CP 基础归 13；packing/Dynamic CP 调度归 29；跨轴资源竞争归 20。
- **obsolete，删除正文**：旧基线 gap 审计、Dynamic CP 不支持结论、固定 “128×/5%/CP=8 最优” 数字和无当前证据的趋势段。

字段 `hierarchical_context_parallel_sizes` 与 `cp_partition_mode` 现由 13 的配置契约拥有；`attention_dropout` 归 [[10_megatron_model_structure_analysis]]。本页只保留这些字段改变 DSv4 路径时的必要交界。

## Related Pages

- [[13_megatron_cp_analysis]] —— Megatron CP group、四种通信类型、hierarchical group 与 `cp_partition_mode` 配置 owner。
- [[29_megatron_packed_dataset_dynamic_cp_analysis]] —— THD packing、Dynamic CP group 选择和 scheduler owner。
- [[34_deepseek_v4_tensor_parallel_analysis]] —— 同一 DSv4 Hybrid Attention 的 TP=1、duplicated 参数与 MoE 交界。
- [[20_megatron_comm_overlap_analysis]] —— 多并行轴的 stream/SM/显存竞争与 profiler 诊断。
- [[../../../01_theory/01_models/deepseek/23_deepseek_v4_cp_analysis|23_deepseek_v4_cp_analysis]] —— DSv4 两阶段 CP 的模型/算法视角。
- [[../../../01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis|20_ring_attention_and_context_parallel_analysis]] —— 通用 CP 与 Ring Attention 理论。
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]] —— 返回本域索引。
