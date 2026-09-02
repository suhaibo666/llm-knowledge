---
title: "Megatron-LM 上下文并行(Context Parallelism)深度解析"
---

# Megatron-LM 上下文并行(Context Parallelism)深度解析

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **重定基线**：2026-09-01 由 `71092579`（2026-08-27）推进，跨 7 个提交；落在本轮改动文件（`transformer_config.py` / `transformer_engine.py` / `transformer_layer.py` / `cuda_graphs.py`）上的引用已逐条重核，全部断言仍成立，仅行号漂移。
> **重定基线**：2026-08-28 由 `ee3f1ffa…`（2026-05-19）推进，跨 578 个提交；本页全部 `path:line` 形式的引用已在新基线下逐条重核;**代码块内被点名的符号与不带行号的裸路径不在该次扫描口径内**,已知漏网处已于 2026-08-28 单独更正。
> 核心:`megatron/core/transformer/dot_product_attention_context_parallel.py`(原生 all-gather 实现)、
> `megatron/core/transformer/transformer_config.py:1101`(`cp_comm_type`)、`megatron/core/transformer/attention.py`(CP 接入点)
> 配套阅读:`15_megatron_pp_schedulers_analysis.md`、`14_megatron_ep_analysis.md`、`12_megatron_tp_analysis.md`
> 适用读者:已了解 transformer 训练与 TP/PP/DP,想吃透 Megatron 上下文并行实现的工程师。
> **叙事顺序**：本页按五拍组织——背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势。
> **最近更新**：2026-08-28。按五拍重排章节顺序；机制正文与既有引用未改。
>
> **划界声明**:CP 通用机制(序列切分、因果负载均衡、因果块裁剪、Ring/All-gather/Ulysses/分层混合四种通信调度、通信量代数、与 TP/PP/DP/EP 的组合关系)已归一到 [[../../../01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis|20_ring_attention_and_context_parallel_analysis]]。**本页只保留 Megatron-LM 的框架实现差异**:`cp_comm_type` 配置接口与按层配置、TE 透传架构、原生 all_gather 回退实现的配置约束、选型决策树,以及 Dynamic CP 的 Megatron 特有源码细节。

---

## 1. 背景：序列切开之后,K/V 还得搬 —— 而搬法不止一种,且随拓扑变

### 1.1 要解决的问题

序列切开只是第一步。切完之后每张卡只持有 `s/cp` 段 Q/K/V,而 attention 要求每个 query 看到它之前的**全部** K/V —— 所以 K/V 必须在 CP 组内搬运,**搬法**才是 Megatron 侧真正的实现分歧点(序列切分本身与因果负载均衡是通用机制,见理论页)。

朴素做法是写死一种搬法。但源码把四种搬法并列进同一个字段的 docstring,并逐条标出它们**互斥的性质**:`p2p` 是「P2P is async and can be overlapped with attention compute」、`all_gather` 是「The all-gather is not async, and cannot be overlapped」、`a2a` 换的是 head 轴而非序列轴、`a2a+p2p` 则「uses A2A communications in low-level CP groups (e.g., via NVLink), and P2P communications in high-level CP groups (e.g., via IBLink)」(`megatron/core/transformer/transformer_config.py:1108-1116`)。**能不能与计算重叠、走哪条物理链路、head 够不够分**这三件事随集群拓扑与模型形状变化 —— 写死任何一种,换个拓扑就会变成瓶颈。这就是 `cp_comm_type` 这个字段存在的理由。

### 1.2 CP 是什么

**上下文并行(Context Parallelism,CP)** 把序列维 `s` 切成 `cp` 段分到 `cp` 张卡,专治 attention 的 `O(s²)` 显存/算力墙——通用动机、收益与限制见理论页 §1;通用机制见理论页全文。

### 1.3 CP 在并行体系中的位置

与 TP/PP/DP/EP 的组合关系(显存账本对照表)见理论页 §2.1;CP 折叠进 EP(MoE Parallel Folding)见理论页 §2.2 + [[14_megatron_ep_analysis]] §9。

### 1.4 记号约定

| 符号 | 含义 |
|------|------|
| `cp` | CP 度(`--context-parallel-size`) |
| `s` | 全局序列长度;每卡持有 `s/cp` |
| `b` / `h` / `d` | micro-batch / hidden / head dim |
| `a` / `a_kv` | attention 头数 / KV 头数(GQA 时 `a_kv < a`) |
| CP 进程组 | `parallel_state.get_context_parallel_group()` |

---

## 2. 为什么这么设计：把搬 K/V 的方式做成一个可按层配置的字符串,而四种里有三种交给 TransformerEngine

Megatron 在 CP 上做了两个不那么显然的选择:**通信调度是一个可以逐层不同的配置字符串**,而**这四种调度里有三种它自己不写**。源码陈述了其中两条理由,并在历史里留下了一处被否掉的替代;"ring / a2a 为什么不自己实现"这条源码全程沉默,由本页重建并标为推断。

**① 这个字段引入时就是"按层配置",而在此之前它根本不存在。**
`cp_comm_type` 由提交 `91a8a4c8d`(2024-10-21)引入,commit message 即「ADLR/megatron-lm!2215 - Configure per-layer communication type for context parallelism」——**"per-layer" 直接写在标题里**。落到类型上是 `Optional[Union[str, List[str]]]`(`megatron/core/transformer/transformer_config.py:1101`),docstring 头两行把两种形态并列:「str: all layers share same communication type. / List[str]: each layer has its separate communication type.」(`:1103-1104`);消费端建层时按层号取下标,并专门为 1-indexed 留了注释「layer_number is 1-indexed, so we need to subtract 1 to get the correct index」(`megatron/core/transformer/transformer_layer.py:374-381`)。
**被否掉的替代就是该提交之前的状态**:同一个 commit 里,`megatron/core/extensions/transformer_engine.py` 的 `TEDotProductAttention.__init__` 才**第一次**出现 `cp_comm_type` 形参与 `extra_kwargs["cp_comm_type"]` 的写入 —— 在此之前 Megatron 根本不向 TE 传这个参数,全模型只有一种调度(该提交给 `None` 留的兜底值正是 `"p2p"`,基线下仍在 `megatron/core/extensions/transformer_engine.py:1689-1690`),无从逐层区分。

**② 三种调度透传给 TE,Megatron 侧只做参数装配与版本门控。**
基线下 Megatron 对 `p2p` / `a2a` / `a2a+p2p` 的全部动作,就是把 CP 组、CP 全局 rank 列表和一条专用 `cp_stream` 塞进 `extra_kwargs` 再交给 `te.pytorch.DotProductAttention`(`megatron/core/extensions/transformer_engine.py:1683-1687`),然后按 TE 版本分档:CP 本身要求 TE ≥ 1.0.0(`:1678-1680`)、`cp_comm_type` 这个参数要求 TE ≥ 1.10.0(`:1688`)、`a2a+p2p` 额外要求 TE ≥ 1.12.0 并把 `cp_group` 整个换成分层 CP 组列表(`:1691-1699`)。**源码在这里陈述的是"哪个 TE 版本起可用"这一事实,没有陈述"为什么不自己实现"。**

**③ 原生实现只留 `all_gather` 一种,而且是被校验硬钉死的。**
不依赖 TE 的那条路来自提交 `157bec937`(2025-11-18,#1859「[Community][Dev] feat(moe): Adding context parallel support to eager attention implementation」),它同时引入开关 `fallback_to_eager_attn`,docstring 写明用途是「Suggested for when desired features are not available in TE implementation」(`megatron/core/transformer/transformer_config.py:1453-1455`)。同一提交加的校验把这条路的调度限死一种:`fallback_to_eager_attn` 或 `transformer_impl == "local"` 时 `cp_comm_type` 必须全部是 `all_gather`,否则 `ValueError`(`:3810-3821`)。
→ 判据可以从原生实现自身的形状读出:`AttentionFuncionWithContextParallel.forward`(`megatron/core/transformer/dot_product_attention_context_parallel.py:150`)把 K/V 收齐后调 `eager_attn_fwd`(`:220`)一次算出完整 softmax,并把整个 `probs` 连同输出一起 `save_for_backward`(`:231`),**没有** online-softmax 的分块合并逻辑。eager 路径本来就要 materialize 完整概率矩阵,ring 的"分块算、增量合并"在这里省不出东西。

> [!note] 推断
> ①③ 的引用都落在源码与提交信息上;**但"ring / a2a / a2a+p2p 的实现为什么留在 TransformerEngine、而不是写进 Megatron"这一条,源码全程沉默** —— 既没有设计注释,也没有 commit message 陈述该取舍,能读到的只有一组 `is_te_min_version` 版本门控(`megatron/core/extensions/transformer_engine.py:1678`、`:1688`、`:1692`)与"装好参数交给 TE"这个事实(`:1683-1701`)。本页据此重建的解释是:这三种调度的收益全部来自 attention 内核**内部**的通信-计算重叠(把 KV 传输塞进分块 softmax 的空档 —— `megatron/core/transformer/transformer_config.py:1108-1109` 的「P2P is async and can be overlapped with attention compute」即指此),必须与 flash-attention 内核写在一起才成立,Megatron 自己写就等于复刻一份 flash-attention。**这是本页的推断,不是作者的自陈。**要引用这条判断,请回到 `megatron/core/extensions/transformer_engine.py:1678`、`:1688`、`:1692-1699` 与 `megatron/core/transformer/transformer_config.py:1108-1109` 这几个 locator,不要引用本段。

---

## 3. `cp_comm_type`:四种通信调度的配置接口

Megatron 把序列切分(通用机制,见理论页 §3)与因果掩码处理(理论页 §4)交给上层统一逻辑,自身的差异化实现集中在**怎么搬 K/V**——由 `cp_comm_type` 选择,取值 `p2p` / `all_gather` / `a2a` / `a2a+p2p`(`megatron/core/transformer/transformer_config.py:1105`)。四种调度的通用机制(Ring 主循环、online-softmax、All-gather 双缓冲、Ulysses 换轴、分层混合分组)见理论页 §5-§8,下面只记 Megatron 特有的接线方式与配置约束。

### 3.1 TE 透传架构(Megatron 特有事实)

**实际的 ring/a2a/a2a+p2p attention 内核在 TransformerEngine 里**——Megatron 只是把 `cp_comm_type` 透传给 TE 的 `DotProductAttention`,自己不实现这三种调度的通信代码。`megatron/core/transformer/dot_product_attention_context_parallel.py` 是**不依赖 TE 时的原生 all-gather 回退实现**(`AttentionFuncionWithContextParallel`,机制见理论页 §6.1,本页不重复代码)。

### 3.2 `p2p`(Ring Attention)—— Megatron 侧配置

机制见理论页 §5。Megatron 侧只是把 `cp_comm_type="p2p"` 传给 TE;README 给出的经验阈值:**超长序列(`s ≥ 8K`)、CP 跨节点时是长上下文训练的默认选择**,`a2a+p2p` 的高层组件复用同一套 p2p 内核。

### 3.3 `all_gather` —— Megatron 侧配置与约束

机制见理论页 §6(原生实现代码即以本页 `megatron/core/transformer/dot_product_attention_context_parallel.py` 为骨架抽取)。Megatron 特有的配置约束:

- `megatron/core/transformer/transformer_config.py:3810` 显示某些场景(如 `fallback_to_eager_attn` 或 `transformer_impl="local"`)**强制要求** `all_gather`——Native CP(`DotProductAttention` 的 eager 路径)只支持这一种通信类型,若要用 `p2p`/`a2a`/`a2a+p2p` 必须走 TE 的 fused flash attention 路径。
- 不推荐用于大 CP / 跨节点超长序列——同步 all-gather 的暴露会拖垮吞吐。

### 3.4 `a2a`(DeepSpeed Ulysses)—— Megatron 侧配置

机制见理论页 §7。Megatron/TE 侧的约束:`cp_comm_type=a2a` 要求 `a_kv ≥ cp`(头要够分,不整除时退回 Ring),文档原话"scatter attention heads across the CP group, and gather to get full sequence of QKV"(`megatron/core/transformer/transformer_config.py:1112`)。适合 head 数足够多、NVLink 域内的 CP。

### 3.5 `a2a+p2p`(分层混合)—— Megatron 侧配置

机制(N 级分层分组构造)见理论页 §8.2,该构造代码实际收录在 [[35_deepseek_v4_context_parallel_analysis]] §1.2(源码级最完整版本)。Megatron 侧配置:`megatron/core/transformer/transformer_config.py:1114` 描述"低层 CP 组用 A2A(如经 NVLink)、高层 CP 组用 P2P(如经 IBLink)"。**推荐**:跨多节点的超长上下文(128K、1M)训练首选;单节点则 `a2a` 或 `p2p` 足够。

---

## 4. 动态上下文并行(Dynamic CP)—— Megatron 特有源码细节

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `85902ef59`。
> ee3f1ff 之后引入并持续完善 **动态上下文并行(Dynamic Context Parallelism, DCP)**——在 THD(packed varlen)训练中**逐 microbatch / 逐样本动态选择 CP 度**,而非全程固定 `cp`(#4226 / #5215 / #5123)。通用机制(动机、`packed_seq_params.cp_group` 切换/恢复原理)见理论页 §10;本节记 Megatron 侧的源码级细节(理论页 §10.2 只narrative 复述了字段与解析逻辑的存在,未展开到函数名/行号级别,以下为完整源码定位)。

### 4.1 机制(源码)

- **`PackedSeqParams` 新增两字段**(`megatron/core/packed_seq_params.py:32-33`):`local_cp_size`(本 microbatch 实际 CP 度)与 `cp_group`(对应的 CP 进程子组),由调度器 `DefaultDynamicCPScheduler` 按样本长度算出。
- **`resolve_cp_group(static_cp_group, packed_seq_params)`**(`megatron/core/packed_seq_params.py:82`,#4226):统一"**优先用 `packed_seq_params.cp_group`,否则回退建图期静态 CP 组**"的解析逻辑,供 `GPTModel`、`GatedDeltaNet`、MTP 层共用(此前各处分散硬编码 `self.pg_collection.cp`)。
- **TE attention 接入**(`megatron/core/extensions/transformer_engine.py:1808`):`TEDotProductAttention.forward` 按 `packed_seq_params.local_cp_size` **切换 TE 内部的 CP 组** —— `local_cp_size==1` → `set_context_parallel_group(None,...)`(该样本关 CP);否则换成 `packed_seq_params.cp_group`。
  - **#5215 修复**(`megatron/core/extensions/transformer_engine.py:1919`):forward **开头先保存原始 CP 组**(`_te_orig_cp_group`),**结尾再恢复**。否则被换掉的动态 CP 组会**泄漏**到后续不带 dynamic CP 的 microbatch,导致 attention 用错组、结果错误。
- **dispatcher 兼容**:sequence packing(THD)原仅支持 `alltoall` dispatcher,现已放宽到 `flex`(#4816,见 [[14_megatron_ep_analysis]] §③ 增量更新);THD 下 HybridEP 会把各 rank 不齐的 token 数补齐到组内最大值。
- **CUDA Graph 守卫**(#4226):`cuda_graph_impl=full_iteration` 与 `cu_seqlens`(THD 变长)互斥,守卫在 `megatron/training/arguments.py:1334-1337`、`:2169-2170`;`_broadcast_cu_seqlens` 在 `cu_seqlens` 为 `None` 时只广播计数 `n=0`、不再广播张量本体(`megatron/core/utils.py:2121-2127`)。

> [!contradiction] 上一条「`cuda_graph_impl=full_iteration` 与 `cu_seqlens` 互斥」在基线 `71092579` 下已不成立。
> ① 两处被引的守卫本身**与 `cu_seqlens` 无关**:`megatron/training/arguments.py:1334-1337` 是 `full_iteration` 要求 `--no-check-for-nan-in-loss-and-grad`,`:2169-2170` 是 `full_iteration` 要求 `--cuda-graph-modules` 为空;在旧基线 `232c478d4` 的 `:1329-1332`/`:2050` 亦是同样两条,并非 THD 互斥判定。
> ② 更关键的是 **#4359(`7f9175207`,[Dev] add cuda graph support for thd format training)已为 THD 变长训练加上 CUDA Graph 支持**:`megatron/core/transformer/cuda_graphs.py` 现有 `thd_sequence_length_upper_bound`(`:1774`/`:1790`)、`_get_thd_varlen_max_num_microbatches`(`:2357`)、`_is_thd_cuda_graph`(`:2268`)等 THD 专用捕获路径,`megatron/core/transformer/transformer_config.py:1231` 的 `thd_max_packed_sequences` 与 `megatron/core/model_parallel_config.py:121` 的 `thd_tail_padding_policy` 为固定尺寸捕获铺垫。
> 因此这里应读作:**只保留 `_broadcast_cu_seqlens` 的空张量协议这一条事实**,「THD 与 full-iteration CUDA Graph 互斥」不再有效。

### 4.2 入口与示例

- 开关:`--dynamic-context-parallel --sequence-packing-scheduler default_dynamic_cp --max-seqlen-per-dp-cp-rank N`。
- 基准示例(#5123):`examples/dynamic_context_parallel/benchmark_dcp.sh`,对比 `dp_balanced` 定长 packed 与 DCP 两条 run,复用 `pretrain_gpt.py` + `MockVarlenDataset`,不引入新模型/数据集类。
- **数据集/调度器侧的完整机制**(packing、`max-seqlen-per-dp-cp-rank` 分配)见 [[29_megatron_packed_dataset_dynamic_cp_analysis]];本节只覆盖 CP/attention 侧的接入。

---

## 5. 约束

CP 在 Megatron 侧不是"设一个 `--context-parallel-size` 就成立"的自由项。下列前提每条都能落到一个 `file:line`,越出前提要么直接抛错,要么**静默降级**。

| # | 前提 / 不变量 | 源码落点 | 破坏后的表现 |
|---|---|---|---|
| 1 | `cp_comm_type` 传 list 时长度必须等于 `num_layers` | `assert`(`megatron/core/transformer/transformer_config.py:3742-3751`) | 直接 assert 失败;没有"补齐到层数"或"广播最后一项"的兜底 |
| 2 | eager / local 路径只能用 `all_gather` | `ValueError`(`megatron/core/transformer/transformer_config.py:3810-3821`);`fallback_to_eager_attn` 本身还要求 `transformer_impl == "transformer_engine"`(`:3804-3808`) | 想在原生路径上用 `p2p`/`a2a` 会在配置校验期就被拒 |
| 3 | TE 版本门控 | CP 要 TE ≥ 1.0.0(`megatron/core/extensions/transformer_engine.py:1678-1680`);`a2a+p2p` 要 TE ≥ 1.12.0(`:1692-1695`) | 不满足直接 assert 失败 |
| 4 | **TE < 1.10.0 时 `cp_comm_type` 被静默丢弃** | `if is_te_min_version("1.10.0"):` 是整段写入的外层守卫(`megatron/core/extensions/transformer_engine.py:1688`) | 不报错也不告警:配置里写的 `a2a` 不会到达 TE,实际跑的是 TE 自己的默认调度 —— 本页唯一一条静默失效 |
| 5 | `cp_comm_type` 只管标准 attention | docstring「This option controls standard attention layers. Linear-attention layers use `linear_cp_mode` instead.」(`megatron/core/transformer/transformer_config.py:1106-1107`) | 给 GDN 一类线性 attention 层配 `cp_comm_type` 无效,要改 `linear_cp_mode`(`:1119`) |
| 6 | DSAttention + CP 只支持 `all_gather` | `assert`(`megatron/core/transformer/transformer_config.py:1862-1875`,原话「DSAttention context parallelism currently supports cp_comm_type=allgather only.」) | 配其它三种直接 assert 失败 |
| 7 | 原生实现依赖 `einops`,且 `attention_mask` 不能为 `None` | `ImportError`(`megatron/core/transformer/dot_product_attention_context_parallel.py:158-159`)、`assert`(`:188-190`) | 缺包 / 无掩码时原生 CP 直接不可用 |
| 8 | 原生实现的 `heads_k_stride` 写死为 1 | `heads_k_stride = 1`(`megatron/core/transformer/dot_product_attention_context_parallel.py:168`),配套 `assert nheads % nheads_k == 0 and nheads_k % heads_k_stride == 0`(`:169`) | KV 双缓冲一次只预取一个 KV head,重叠粒度不可调(源码自带 `# TODO make it configurable`,`:234`) |
| 9 | 原生路径开 CP 时 attention dropout 必须为 0 | `assert`(`megatron/core/transformer/dot_product_attention.py:60-65`) | dropout 非 0 直接 assert 失败 |
| 10 | Dynamic CP 要求 `dp-cp` 组内 rank 数为偶数 | `assert`(`megatron/core/parallel_state.py:925-927`,「Dynamic context parallel requires an even number of ranks」) | 奇数 rank 数下动态 CP 建不出组 |

**代价**:§3.1 那条"透传"是拿**可观测性**换来的 —— `p2p` / `a2a` / `a2a+p2p` 的真实通信行为、重叠效果与失败模式全部发生在 TE 内部,Megatron 侧能看到的只有一个字符串和一组版本断言(`megatron/core/extensions/transformer_engine.py:1678-1701`)。调不动的时候,能改的旋钮只剩"换调度类型"或"换 TE 版本"。

**故意不做的事**:原生 all-gather 路径**不参与** THD / Dynamic CP。`DotProductAttention.forward` 开头就 `assert packed_seq_params is None`(`megatron/core/transformer/dot_product_attention.py:159-162`),`attention_bias` 同样不支持(`:163`);走到 CP 分支时它取的是全局的 `parallel_state.get_context_parallel_group()`(`:191`),而**不是** §4.1 里那个 `packed_seq_params.cp_group`。也就是说 §4 的 Dynamic CP 只存在于 TE 路径上。

---

## 6. 适用场景及选型(Megatron 特有操作指南)

### 6.1 何时用 CP

| 场景 | 是否用 CP | 原因 |
|------|----------|------|
| 序列 < 8K | ❌ 一般不用 | attention `O(s²)` 还没成瓶颈,CP 的通信纯亏 |
| 序列 ≥ 8K(README 阈值) | ✅ 用 CP | 把 attention 激活/算力 `÷cp` |
| 超长上下文 128K / 1M | ✅ 必须用 CP | 否则 attention 单卡绝无可能 |
| 模型权重放不下 | ❌ CP 救不了 | CP 不切权重;用 TP/PP/EP |

### 6.2 `cp_comm_type` 选型决策树

```
要训长序列(s ≥ 8K)?
└─ 是 ──► 开 CP,选 cp_comm_type:
          │
          ├─ 跨多节点超长序列(128K/1M)?
          │   └─ 是 ──► a2a+p2p(节点内 A2A + 节点间 P2P,各用所长)
          │
          ├─ 单节点 / NVLink 域内,且 head 数 ≥ cp?
          │   └─ 是 ──► a2a(Ulysses,换 head 轴,attention 见全序列)
          │
          ├─ 长序列、要异步重叠通信?
          │   └─ 是 ──► p2p(ring attention,P2P 异步可重叠,通用默认)
          │
          └─ 要实现简单 / 小 CP / 特性强制?
              └─ 是 ──► all_gather(全收 KV,逻辑最简,通信不可重叠)

并行组合(README Guideline 5):
  - CP 与 TP/PP/DP/EP 正交,可任意叠加
  - MoE:attention 用 TP×CP×DP,CP 折叠进 EP(MoE Parallel Folding,见 14_megatron_ep_analysis.md §9)
  - CP 与 TP 同属高带宽通信,优先压在 NVLink 域内
```

### 6.3 一句话总结

- **CP 的本质**:把序列维 `s` 切成 `cp` 段,专治 attention 的 `O(s²)` 显存/算力墙;切激活、不切权重(通用机制,详见理论页)。
- **Megatron 的接线方式**:`cp_comm_type` 四选一,`p2p`/`a2a`/`a2a+p2p` 透传给 TransformerEngine 内核,`all_gather` 走原生回退实现;按层配置见 §3、动态选择见 §4。

---

## 7. 发展趋势

> [!note] 推断:锚点是基线 `71092579` 下的源码事实(新增字段、TODO、被限死的新特性)与 §4 已核过的 PR 编号,方向判断由本页承担,不是源码的自陈计划。

**一、CP 的配置轴正在按 attention 家族分裂。** 基线下新增字段 `linear_cp_mode`(`megatron/core/transformer/transformer_config.py:1119`),docstring 明写它「Independent of `cp_comm_type`, which only controls standard attention」(`:1121`),并给出 `chunkwise`(默认;保持序列分片,用 CP-aware 线性核)与 `headwise`(Ulysses 式换 head 轴)两种模式,还自评后者「Correct but memory-heavy」(`:1128-1129`);对称地,`cp_comm_type` 的 docstring 也被补上「This option controls standard attention layers. Linear-attention layers use `linear_cp_mode` instead.」(`:1106-1107`)。**由此可推断**:`cp_comm_type` 四选一已经不是 CP 配置面的全部,混合线性/标准 attention 的模型要同时调两个轴;再读到"CP 怎么配"的代码时,应先确认它管的是哪一类 attention 层。

**二、新 attention 变体接入 CP 的第一步都是 `all_gather`。** DSAttention 在基线下的 CP 支持被硬限制成「DSAttention context parallelism currently supports cp_comm_type=allgather only.」(`megatron/core/transformer/transformer_config.py:1862-1875`),该约束由 `da482cf5c`(「[split 4/4] Enable DSA CP and THD hooks」,#5246)带入;eager attention 的 CP 支持(#1859)落地时是同一形状(见 §2③)。**由此可推断**:`all_gather` 在 Megatron 里的定位不只是"小 CP 的简单实现",更是**新特性接入 CP 的最低成本入口** —— 它不要求内核内部做分块合并。看到某个新 attention 变体"支持 CP"时,应先查它到底支持哪几种 `cp_comm_type`。

**三、Dynamic CP 仍在补边角,且只覆盖 TE 路径。** 除 §4 已核过的 PR 链(#4226 建 `resolve_cp_group`、#5215 修 CP 组泄漏、#5123 加基准脚本、#4359 补 THD 的 CUDA Graph 捕获)之外,源码里还挂着两条未决项:`megatron/core/parallel_state.py:922` 的「TODO: Are gloo groups needed for Dynamic CP?」,以及混合 CP 调度器里的「TODO[pmannan]: PP not yet supported. Add PP scheduling.」(`megatron/core/pipeline_parallel/hybrid_cp_schedule.py:190`);再叠加 §5"故意不做"那条(原生路径完全不参与 Dynamic CP)。**由此可推断**:Dynamic CP 目前是"TE 路径 + 无 PP"这个组合下才成立的特性,要把它与 PP 或原生 attention 一起打开,须先回源码确认这两条 TODO 是否已经关闭。

---

*生成依据:`Megatron-LM` `dev` 分支 `85902ef59`(2026-09-01;由 `71092579` 重定基线而来,§4 的特性增量基准仍为 `dev@232c478d4`)。源码行号以 `85902ef59` 为准。`p2p`/`a2a`/`a2a+p2p` 的实际 attention 内核位于 TransformerEngine,Megatron 透传 `cp_comm_type`;原生 `all_gather` 实现见 `megatron/core/transformer/dot_product_attention_context_parallel.py`,通用机制骨架已归一至理论页。配套文档:`15_megatron_pp_schedulers_analysis.md`、`14_megatron_ep_analysis.md`、`12_megatron_tp_analysis.md`。*

---

## 配置契约：CP 的两个补充字段

本页正文覆盖了 `cp_comm_type` 等主干开关。本节补 `# Model parallelism` 段里两个此前零提及的字段。**下表直接取自 `megatron/core/model_parallel_config.py` 的类体**。



### `ModelParallelConfig`（`megatron/core/model_parallel_config.py`，2 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `min_dynamic_context_parallel_size` | `int` | `1` | Minimum CP group size for dynamic context parallel. Default 1 (no CP). The maximum is dp_size * context_parallel_size (the full DPxCP group). | `:91` |
| `hybrid_context_parallel` | `bool` | `False` | Deprecated. Use `dynamic_context_parallel` instead. | `:95` |

> 该类共 74 个字段，本表收 2 项；其余 72 项已在别处归属：主要归 [[15_megatron_pp_schedulers_analysis]] 16 项、[[12_megatron_tp_analysis]] 10 项、[[20_megatron_comm_overlap_analysis]] 10 项、[[22_megatron_memory_optimization_analysis]] 6 项，另散见 14 页（完整归属见 `docs/coverage/megatron-lm.yaml`）。

## Related Pages

- [[../../../01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis|20_ring_attention_and_context_parallel_analysis]] —— CP/Ring Attention 通用机制(序列切分、因果裁剪、四种通信调度、通信量代数、并行组合关系)
- [[15_megatron_pp_schedulers_analysis]] · [[14_megatron_ep_analysis]] · [[12_megatron_tp_analysis]] · [[29_megatron_packed_dataset_dynamic_cp_analysis]]
- [[35_deepseek_v4_context_parallel_analysis]] —— DeepSeek-V4 在同一套 Megatron CP 基础设施上的模型特有适配(MLA/CSA/HCA)
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]


