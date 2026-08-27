---
title: "Megatron-LM 上下文并行(Context Parallelism)深度解析"
---

# Megatron-LM 上下文并行(Context Parallelism)深度解析

> 代码基准:`Megatron-LM/` 子仓库 `dev` 分支,commit `ee3f1ff`
> 核心:`megatron/core/transformer/dot_product_attention_context_parallel.py`(原生 all-gather 实现)、
> `transformer_config.py:927`(`cp_comm_type`)、`attention.py`(CP 接入点)
> 配套阅读:`15_megatron_pp_schedulers_analysis.md`、`14_megatron_ep_analysis.md`、`12_megatron_tp_analysis.md`
> 适用读者:已了解 transformer 训练与 TP/PP/DP,想吃透 Megatron 上下文并行实现的工程师。
>
> **划界声明**:CP 通用机制(序列切分、因果负载均衡、因果块裁剪、Ring/All-gather/Ulysses/分层混合四种通信调度、通信量代数、与 TP/PP/DP/EP 的组合关系)已归一到 [[../../../01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis|20_ring_attention_and_context_parallel_analysis]]。**本页只保留 Megatron-LM 的框架实现差异**:`cp_comm_type` 配置接口与按层配置、TE 透传架构、原生 all_gather 回退实现的配置约束、选型决策树,以及 Dynamic CP 的 Megatron 特有源码细节。

---

## 0. 总览

### 0.1 CP 是什么

**上下文并行(Context Parallelism,CP)** 把序列维 `s` 切成 `cp` 段分到 `cp` 张卡,专治 attention 的 `O(s²)` 显存/算力墙——通用动机、收益与限制见理论页 §1;通用机制见理论页全文。

### 0.2 CP 在并行体系中的位置

与 TP/PP/DP/EP 的组合关系(显存账本对照表)见理论页 §2.1;CP 折叠进 EP(MoE Parallel Folding)见理论页 §2.2 + [[14_megatron_ep_analysis]] §6。

### 0.3 记号约定

| 符号 | 含义 |
|------|------|
| `cp` | CP 度(`--context-parallel-size`) |
| `s` | 全局序列长度;每卡持有 `s/cp` |
| `b` / `h` / `d` | micro-batch / hidden / head dim |
| `a` / `a_kv` | attention 头数 / KV 头数(GQA 时 `a_kv < a`) |
| CP 进程组 | `parallel_state.get_context_parallel_group()` |

---

## 1. `cp_comm_type`:四种通信调度的配置接口

Megatron 把序列切分(通用机制,见理论页 §3)与因果掩码处理(理论页 §4)交给上层统一逻辑,自身的差异化实现集中在**怎么搬 K/V**——由 `cp_comm_type` 选择,取值 `p2p` / `all_gather` / `a2a` / `a2a+p2p`(`transformer_config.py:931`)。四种调度的通用机制(Ring 主循环、online-softmax、All-gather 双缓冲、Ulysses 换轴、分层混合分组)见理论页 §5-§8,下面只记 Megatron 特有的接线方式与配置约束。

### 1.1 TE 透传架构(Megatron 特有事实)

**实际的 ring/a2a/a2a+p2p attention 内核在 TransformerEngine 里**——Megatron 只是把 `cp_comm_type` 透传给 TE 的 `DotProductAttention`,自己不实现这三种调度的通信代码。`dot_product_attention_context_parallel.py` 是**不依赖 TE 时的原生 all-gather 回退实现**(`AttentionFuncionWithContextParallel`,机制见理论页 §6.1,本页不重复代码)。

### 1.2 `p2p`(Ring Attention)—— Megatron 侧配置

机制见理论页 §5。Megatron 侧只是把 `cp_comm_type="p2p"` 传给 TE;README 给出的经验阈值:**超长序列(`s ≥ 8K`)、CP 跨节点时是长上下文训练的默认选择**,`a2a+p2p` 的高层组件复用同一套 p2p 内核。

### 1.3 `all_gather` —— Megatron 侧配置与约束

机制见理论页 §6(原生实现代码即以本页 `dot_product_attention_context_parallel.py` 为骨架抽取)。Megatron 特有的配置约束:

- `transformer_config.py:2797` 显示某些场景(如 `fallback_to_eager_attn` 或 `transformer_impl="local"`)**强制要求** `all_gather`——Native CP(`DotProductAttention` 的 eager 路径)只支持这一种通信类型,若要用 `p2p`/`a2a`/`a2a+p2p` 必须走 TE 的 fused flash attention 路径。
- 不推荐用于大 CP / 跨节点超长序列——同步 all-gather 的暴露会拖垮吞吐。

### 1.4 `a2a`(DeepSpeed Ulysses)—— Megatron 侧配置

机制见理论页 §7。Megatron/TE 侧的约束:`cp_comm_type=a2a` 要求 `a_kv ≥ cp`(头要够分,不整除时退回 Ring),文档原话"scatter attention heads across the CP group, and gather to get full sequence of QKV"(`transformer_config.py:936`)。适合 head 数足够多、NVLink 域内的 CP。

### 1.5 `a2a+p2p`(分层混合)—— Megatron 侧配置

机制(N 级分层分组构造)见理论页 §8.2,该构造代码实际收录在 [[35_deepseek_v4_context_parallel_analysis]] §1.2(源码级最完整版本)。Megatron 侧配置:`transformer_config.py:938` 描述"低层 CP 组用 A2A(如经 NVLink)、高层 CP 组用 P2P(如经 IBLink)"。**推荐**:跨多节点的超长上下文(128K、1M)训练首选;单节点则 `a2a` 或 `p2p` 足够。

---

## 2. 适用场景及选型(Megatron 特有操作指南)

### 2.1 何时用 CP

| 场景 | 是否用 CP | 原因 |
|------|----------|------|
| 序列 < 8K | ❌ 一般不用 | attention `O(s²)` 还没成瓶颈,CP 的通信纯亏 |
| 序列 ≥ 8K(README 阈值) | ✅ 用 CP | 把 attention 激活/算力 `÷cp` |
| 超长上下文 128K / 1M | ✅ 必须用 CP | 否则 attention 单卡绝无可能 |
| 模型权重放不下 | ❌ CP 救不了 | CP 不切权重;用 TP/PP/EP |

### 2.2 `cp_comm_type` 选型决策树

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
  - MoE:attention 用 TP×CP×DP,CP 折叠进 EP(MoE Parallel Folding,见 14_megatron_ep_analysis.md §6)
  - CP 与 TP 同属高带宽通信,优先压在 NVLink 域内
```

### 2.3 一句话总结

- **CP 的本质**:把序列维 `s` 切成 `cp` 段,专治 attention 的 `O(s²)` 显存/算力墙;切激活、不切权重(通用机制,详见理论页)。
- **Megatron 的接线方式**:`cp_comm_type` 四选一,`p2p`/`a2a`/`a2a+p2p` 透传给 TransformerEngine 内核,`all_gather` 走原生回退实现;按层配置见 §1、动态选择见 §3。

---

## 3. 动态上下文并行(Dynamic CP)—— Megatron 特有源码细节

> [!update] 2026-06-16 · dev@232c478d4
> ee3f1ff 之后引入并持续完善 **动态上下文并行(Dynamic Context Parallelism, DCP)**——在 THD(packed varlen)训练中**逐 microbatch / 逐样本动态选择 CP 度**,而非全程固定 `cp`(#4226 / #5215 / #5123)。通用机制(动机、`packed_seq_params.cp_group` 切换/恢复原理)见理论页 §10;本节记 Megatron 侧的源码级细节(理论页 §10.2 只narrative 复述了字段与解析逻辑的存在,未展开到函数名/行号级别,以下为完整源码定位)。

### 3.1 机制(源码)

- **`PackedSeqParams` 新增两字段**(`packed_seq_params.py:23-24`):`local_cp_size`(本 microbatch 实际 CP 度)与 `cp_group`(对应的 CP 进程子组),由调度器 `DefaultDynamicCPScheduler` 按样本长度算出。
- **`resolve_cp_group(static_cp_group, packed_seq_params)`**(`packed_seq_params.py:69`,#4226):统一"**优先用 `packed_seq_params.cp_group`,否则回退建图期静态 CP 组**"的解析逻辑,供 `GPTModel`、`GatedDeltaNet`、MTP 层共用(此前各处分散硬编码 `self.pg_collection.cp`)。
- **TE attention 接入**(`extensions/transformer_engine.py:1798`):`TEDotProductAttention.forward` 按 `packed_seq_params.local_cp_size` **切换 TE 内部的 CP 组** —— `local_cp_size==1` → `set_context_parallel_group(None,...)`(该样本关 CP);否则换成 `packed_seq_params.cp_group`。
  - **#5215 修复**(`transformer_engine.py:1886`):forward **开头先保存原始 CP 组**(`_te_orig_cp_group`),**结尾再恢复**。否则被换掉的动态 CP 组会**泄漏**到后续不带 dynamic CP 的 microbatch,导致 attention 用错组、结果错误。
- **dispatcher 兼容**:sequence packing(THD)原仅支持 `alltoall` dispatcher,现已放宽到 `flex`(#4816,见 [[14_megatron_ep_analysis]] §③ 增量更新);THD 下 HybridEP 会把各 rank 不齐的 token 数补齐到组内最大值。
- **CUDA Graph 守卫**(#4226,`training/utils.py`):`cuda_graph_impl=full_iteration` 与 `cu_seqlens`(THD 变长)互斥,`_broadcast_cu_seqlens` 直接短路返回 `None`。

### 3.2 入口与示例

- 开关:`--dynamic-context-parallel --sequence-packing-scheduler default_dynamic_cp --max-seqlen-per-dp-cp-rank N`。
- 基准示例(#5123):`examples/dynamic_context_parallel/`(`benchmark_dcp.sh`),对比 `dp_balanced` 定长 packed 与 DCP 两条 run,复用 `pretrain_gpt.py` + `MockVarlenDataset`,不引入新模型/数据集类。
- **数据集/调度器侧的完整机制**(packing、`max-seqlen-per-dp-cp-rank` 分配)见 [[29_megatron_packed_dataset_dynamic_cp_analysis]];本节只覆盖 CP/attention 侧的接入。

---

*生成依据:`Megatron-LM` `dev` 分支 `ee3f1ff`(§3 增量基准 `dev@232c478d4`)。源码行号以对应 commit 为准。`p2p`/`a2a`/`a2a+p2p` 的实际 attention 内核位于 TransformerEngine,Megatron 透传 `cp_comm_type`;原生 `all_gather` 实现见 `dot_product_attention_context_parallel.py`,通用机制骨架已归一至理论页。配套文档:`15_megatron_pp_schedulers_analysis.md`、`14_megatron_ep_analysis.md`、`12_megatron_tp_analysis.md`。*

## Related Pages

- [[../../../01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis|20_ring_attention_and_context_parallel_analysis]] —— CP/Ring Attention 通用机制(序列切分、因果裁剪、四种通信调度、通信量代数、并行组合关系)
- [[15_megatron_pp_schedulers_analysis]] · [[14_megatron_ep_analysis]] · [[12_megatron_tp_analysis]] · [[29_megatron_packed_dataset_dynamic_cp_analysis]]
- [[35_deepseek_v4_context_parallel_analysis]] —— DeepSeek-V4 在同一套 Megatron CP 基础设施上的模型特有适配(MLA/CSA/HCA)
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
