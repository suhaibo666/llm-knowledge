# Megatron-LM 上下文并行(Context Parallelism)深度解析

> 代码基准:`Megatron-LM/` 子仓库 `dev` 分支,commit `ee3f1ff`
> 核心:`megatron/core/transformer/dot_product_attention_context_parallel.py`(原生 all-gather 实现)、
> `transformer_config.py:927`(`cp_comm_type`)、`attention.py`(CP 接入点)
> 配套阅读:`megatron_pp_schedulers_analysis.md`、`megatron_ep_analysis.md`、`megatron_tp_analysis.md`
> 适用读者:已了解 transformer 训练与 TP/PP/DP,想吃透 Megatron 上下文并行实现的工程师。

---

## 0. 总览

### 0.1 CP 是什么

**上下文并行(Context Parallelism,CP)**:把**序列维 `s`** 切成 `cp` 段,分到 `cp` 张卡上,每卡只持有 `s/cp` 个 token 的 Q/K/V 与激活。它专门为**超长序列训练**(8K、128K、1M token)而生。

难点在 attention:self-attention 里**每个 query 要看到全部 key/value**,但 K/V 被切散在各卡上了。所以 CP 的核心是:**如何在序列被切开的前提下,让每个 query 仍能算到对全序列的 attention** —— 答案是在各卡间搬运 K/V(或搬运 attention 的部分结果)。这套搬运策略有 4 种,由 `cp_comm_type` 选择,是本文逐个解读的"调度器"。

> 除 attention 外的算子(LayerNorm、MLP、逐元素算子)都是 **token-wise** 的,序列切开后各卡独立算、互不通信 —— CP 的全部通信都集中在 attention。

### 0.2 CP 在并行体系中的位置

| 并行轴 | 切什么 | 峰值激活 | 权重 | 优化器 | 通信特征 |
|--------|--------|---------|------|--------|---------|
| TP | 单层权重矩阵 | 1/tp(SP) | 1/tp | 1/tp | 高频、关键路径 |
| EP | 专家(按个数) | ~1 | MoE 1/N | 1/N | 中 |
| PP | 层(按深度) | 1(VPP>1) | 1/N | 1/N | 中,点对点 |
| **CP** | **序列** | **1/cp** | **1(不切权重)** | **1/cp(分布式优化器)** | **中,仅 attention,可重叠** |
| DP | 批次 | 1 | 1 | 1/N(分布式优化器) | 低 |

关键:**CP 切激活与序列相关显存(`÷cp`),但不切权重** —— 与 TP 相反。CP 的通信只发生在 attention,且(ring/a2a 模式下)能与计算重叠。

### 0.3 记号约定

| 符号 | 含义 |
|------|------|
| `cp` | CP 度(`--context-parallel-size`) |
| `s` | 全局序列长度;每卡持有 `s/cp` |
| `b` / `h` / `d` | micro-batch / hidden / head dim |
| `a` / `a_kv` | attention 头数 / KV 头数(GQA 时 `a_kv < a`) |
| CP 进程组 | `parallel_state.get_context_parallel_group()` |

---

## 1. CP 的目的与动机

### 1.1 要解决的问题:attention 的 `O(s²)` 墙

self-attention 的注意力分数矩阵是 `[s, s]`,激活显存与计算量都是 **`O(s²)`**。序列从 4K 拉到 128K,attention 激活暴涨 1024 倍 —— 这是长序列训练的头号显存/算力墙。

其他并行轴都救不了它:
- **TP** 切 hidden/head 维,不切序列 → attention 分数矩阵仍是 `[s, s]`。
- **PP** 切层,不切序列。
- **DP** 切 batch,不切序列。

只有**把序列本身切开**,才能把单卡的 attention 负担从 `O(s²)` 降到 `O(s²/cp)`。这就是 CP。

### 1.2 CP 的收益

- **激活显存 `÷cp`**:每卡只存 `s/cp` token 的激活;attention 分数矩阵每卡 `O((s/cp)·s)`(ring/all-gather)或更低。
- **能训练否则放不下的超长序列**:CP 是 128K/1M 上下文训练的关键使能技术。
- **与 TP/PP/DP/EP 正交**:可任意组合。

### 1.3 CP 不解决什么

- **不切权重**:每卡仍持完整模型权重(CP 不是模型并行)。权重显存要靠 TP/PP/EP。
- **引入 attention 通信**:K/V 或部分结果要在 CP 组内搬运(§3 四种策略)。

### 1.4 与 MoE Parallel Folding 的关系

对 MoE 模型,CP 对**专家层无意义**(token 独立处理,无需跨序列)。Megatron 的 MoE Parallel Folding 正是利用这一点:attention 用 `TP×CP×DP`,MoE 把 `CP` **折叠进 `EP`**(`ETP×EP×EDP`)。详见 `megatron_ep_analysis.md` §6。

---

## 2. CP 核心机制

### 2.1 序列切分 + attention 的通信需求

```
全局序列 s = [tok 0 … tok s-1]   按 CP 切成 cp 段:
  CP rank 0: tok[0 : s/cp]          ← 各卡只持有本段的 Q/K/V
  CP rank 1: tok[s/cp : 2S/cp]
  …
  CP rank cp-1: tok[(cp-1)s/cp : s]

非 attention 算子(LN / MLP / 逐元素):token-wise,各卡独立算,零通信 ✅
attention:每个 query 要看全序列 K/V → 必须在 CP 组内搬运 K/V 或部分结果 ⚠️
```

`cp_comm_type` 决定"怎么搬",取值 `p2p` / `all_gather` / `a2a` / `a2a+p2p`(`transformer_config.py:931`)。**实际的 ring/a2a attention 内核在 TransformerEngine 里**,Megatron 把 `cp_comm_type` 透传给 TE 的 `DotProductAttention`;`dot_product_attention_context_parallel.py` 是不依赖 TE 时的**原生 all-gather 回退实现**。

### 2.2 因果掩码下的负载均衡(zigzag 切分)

**因果(causal)attention** 里,query 位置 `i` 只 attend 到 key `0..i`。若按序列朴素均分:持有**靠后**段的卡要 attend 几乎全序列(计算多),持有**靠前**段的卡几乎不算 —— 严重负载不均。

Megatron 的解法:把序列切成 **`2·cp`** 个小块,给 CP rank `r` 分配第 `r` 块和第 `2cp-1-r` 块(一前一后)。于是每张卡都拿到"一个早块 + 一个晚块",计算量被拉平。源码 `to_zz_mask_attn_bias`(`dot_product_attention_context_parallel.py:135`,`zz` = **z**ig**z**ag)就是按这个重排做掩码:

```python
chunked = attention_mask.chunk(dim=3, chunks=cp_size * 2)               # 切成 2·cp 块
zz_mask = [x for p in zip(chunked[:cp_size], reversed(chunked[cp_size:])) for x in p]
#         配对:(块0,块2cp-1)、(块1,块2cp-2)、… → 每个 CP rank 拿一对(早+晚)
```

```
朴素切分(因果):  rank0 [▁▁▁] 算得少    rank3 [███] 算得多   ← 不均衡
zigzag 切分:      每个 rank = 一个早块 + 一个晚块            ← 计算量拉平 ✅
  rank0 = 块0 + 块7   rank1 = 块1 + 块6   rank2 = 块2 + 块5   rank3 = 块3 + 块4
```

这是 CP 的"等效负载气泡"消除手段 —— 类比 EP 的 `aux_loss`、PP 的 VPP。

---

## 调度器① — `cp_comm_type = "p2p"`(Ring Attention)

### ①.1 动机与解决的问题

**思路**:不把 K/V 全收上来,而是让 K/V 分块**在 CP 组内沿环形(ring)逐站传递**。每张卡保留自己的 Q,轮流拿到别人的 K/V 块,算"本地 Q 对该块 K/V"的部分 attention,用 **online-softmax** 累加。`cp` 张卡转一圈(`cp-1` 步 P2P)后,每个 query 就算全了对全序列的 attention。

**解决的核心问题**:P2P 通信**异步**,可与 attention 计算**重叠** —— 当前块在算时,下一块的 K/V 已在后台传输。于是通信延迟被算力掩盖,这是长序列 CP 的**推荐默认**。

### ①.2 机制与流程

```
初始:每 rank 持有自己的 Q,K,V 块(s/cp 长度)
step 0: 算 本地Q × 本地KV 的部分 attention;同时 P2P 把本地 KV 发往环的下一站
step 1: 收到上游 KV 块 → 算 本地Q × 该KV;同时把它继续 P2P 传下去
  …
step cp-1: 转完一圈,每个 query 累积了对全序列的 attention(online-softmax 合并)
```

```
       KV 环形传递(P2P,async,与计算重叠)
   ┌──────────────────────────────────────┐
   ▼                                      │
 rank0 ──KV──► rank1 ──KV──► rank2 ──KV──► rank3
   ▲                                      │
   └──────────────────────────────────────┘
   每 rank:本地 Q 固定,轮流接收各站 KV,partial-attention + online-softmax 累加
```

### ①.3 开销分析

| 维度 | Ring(p2p) |
|------|-----------|
| 通信量/rank | `cp-1` 次 P2P,每次传 `s/cp` 的 KV ≈ 总量 `∝ s·b·a_kv·d` |
| 通信暴露 | **低** —— P2P 异步,与 attention 计算重叠 |
| 峰值激活 | `O((s/cp)·s)` 的 attention 分数 + 双 KV 缓冲(收/发) |
| 因果负载 | 需 zigzag 切分拉平 |

### ①.4 适用场景

- **超长序列**(README:`s ≥ 8K`)、CP 跨节点:P2P 异步可掩盖通信,这是长上下文训练的默认选择。
- `a2a+p2p` 的高层组件(见④)。

---

## 调度器② — `cp_comm_type = "all_gather"`(全收 KV)

### ②.1 动机与解决的问题

**最朴素**:attention 前直接 **all-gather** 把全序列 K/V 收到每张卡,然后每卡用本地的 `s/cp` 个 query 对**完整 `s`** 的 K/V 算 attention。逻辑最简单,不需要 online-softmax,不需要环形调度。

**代价**:all-gather **不是异步的**(`transformer_config.py:935` 明确写 "not async, cannot be overlapped"),通信暴露在关键路径上;且每卡要物化完整 K/V。

### ②.2 源码(原生实现,`dot_product_attention_context_parallel.py`)

这是不依赖 TE 时的回退实现 `AttentionFuncionWithContextParallel`(`:150`):

```python
def forward(ctx, q, k, v, attention_mask, ..., pg):              # :154
    comm = AllGatherComm(group=pg)
    # 把本地 K/V 块 all-gather 成全序列(按 head 分批,边收边算以省显存)
    comm.all_gather(kv_buffer_copy[0], k_0)
    comm.all_gather(kv_buffer_copy[1], v_0)
    for i in range(0, nheads_k, heads_k_stride):
        comm.wait()                                              # 等本批 KV 收齐
        if i < nheads_k - heads_k_stride:
            comm.all_gather(kv_buffer_copy[0], send_k)           # 预取下一批 head 的 KV
            comm.all_gather(kv_buffer_copy[1], send_v)
        out_i, probs_i = eager_attn_fwd(q_i, k_i, v_i, attn_bias, ...)  # 本地 Q × 全序列 KV
    out = torch.cat(outs, dim=2)
```

反向 `backward`(`:240`)对称:all-gather 全 KV,算梯度,`reduce_scatter_tensor` 把 `dk/dv` 散回各卡(`:333`)。注意它仍**按 head 分批 all-gather** 并预取下一批 —— 在"逐 head"粒度上做了一点重叠,但整体 all-gather 仍属同步。

### ②.3 开销分析

| 维度 | all_gather |
|------|-----------|
| 通信量/rank | all-gather 全 KV(`∝ s·b·a_kv·d`)+ 反向 reduce-scatter |
| 通信暴露 | **高** —— all-gather 同步,难重叠 |
| 峰值激活 | 每卡物化完整 `s` 的 K/V;attention 分数 `O((s/cp)·s)` |
| 实现复杂度 | 低(无 online-softmax、无环形调度) |

### ②.4 适用场景

- 实现简单、调试友好;CP 较小、序列不极端时可用。
- `transformer_config.py:2797` 显示某些场景(如配合特定特性)**强制要求** `all_gather`。
- 不推荐用于大 CP / 跨节点超长序列 —— 同步 all-gather 的暴露会拖垮吞吐。

---

## 调度器③ — `cp_comm_type = "a2a"`(DeepSpeed Ulysses)

### ③.1 动机与解决的问题

ring 和 all_gather 都在"序列切分"这个轴上想办法。**Ulysses 换轴**:attention 计算本身在 **head 维**是天然并行的(各 head 独立)。于是:

> attention **之前**:用 all-to-all 把张量从"**序列切分**"重排成"**head 切分**" —— 每卡换成持有**完整序列**但只有 `a/cp` 个 head;
> attention **之后**:再用 all-to-all 换回"序列切分"。

每卡在 attention 时看到完整序列(无需 ring,无需 online-softmax),只是 head 少了。`cp_comm_type=a2a`(`transformer_config.py:936`):"scatter attention heads across the CP group, and gather to get full sequence of QKV"。

### ③.2 机制与流程

```
attention 前 A2A:  [s/cp 序列,  a 头]  ──all-to-all──►  [s 完整序列,  a/cp 头]
                         │                                      │
                         ▼                                      ▼
                  (序列切分,头完整)                      (序列完整,头切分)
                                                                 │
                                              每卡对 a/cp 个 head 算完整序列 attention
                                                                 │
attention 后 A2A:  [s 完整序列,  a/cp 头]  ──all-to-all──►  [s/cp 序列,  a 头]
```

每个 attention 含 **2 次 all-to-all**(QKV 进、输出出)。

### ③.3 开销分析

| 维度 | a2a(Ulysses) |
|------|---------------|
| 通信量/rank | 2 次 A2A,量 `∝ s·b·h`,与 `cp` 关系是 A2A 的固有规律 |
| 通信暴露 | 中 —— A2A 比同步 all-gather 好,但不如 ring 易重叠 |
| 峰值激活 | attention 时每卡 `a/cp` 头、完整 `s` 序列 |
| 约束 | **需 `a_kv ≥ cp`**(head 要够分);`cp` 整除头数 |

### ③.4 适用场景

- head 数足够多(`a ≥ cp`)、序列长但没到极端跨多节点。
- A2A 在 NVLink 域内效率高 → Ulysses 适合**单节点或 NVLink 域内**的 CP。
- head 不够分(GQA、`a_kv < cp`)时不适用 → 退回 ring。

---

## 调度器④ — `cp_comm_type = "a2a+p2p"`(分层混合)

### ④.1 动机与解决的问题

Ulysses(a2a)适合 NVLink 域内,ring(p2p)适合跨节点异步重叠。大规模训练 CP 往往**跨多节点**:节点内带宽高、节点间带宽低。

**`a2a+p2p` 的动机**:把 CP 组拆成两层,**各用所长**(`transformer_config.py:938`):
- **低层 CP 组(节点内,NVLink)**:用 **A2A**(Ulysses),吃满 NVLink 带宽。
- **高层 CP 组(节点间,IB)**:用 **P2P 环形**(ring),异步可重叠,扛跨节点低带宽。

### ④.2 机制

```
CP 组 = 低层(节点内)× 高层(节点间)

  节点 0                          节点 1
 ┌───────────────┐              ┌───────────────┐
 │ rank0  rank1  │  ◄─P2P ring─► │ rank2  rank3  │   高层:跨节点 P2P(异步重叠)
 │   └─A2A─┘     │              │   └─A2A─┘     │   低层:节点内 A2A(NVLink)
 └───────────────┘              └───────────────┘
```

节点内用 Ulysses 换 head 轴,节点间用 ring 传 KV —— 两级嵌套。

### ④.3 开销与适用场景

| 维度 | a2a+p2p |
|------|---------|
| 通信 | 节点内 A2A(高带宽)+ 节点间 P2P(异步重叠) |
| 通信暴露 | 低 —— 跨节点那段用可重叠的 P2P |
| 适用 | **多节点超长序列**:既要 NVLink 效率,又要跨节点可重叠 |
| 复杂度 | 最高(两级 CP 组嵌套) |

**推荐**:跨多节点的超长上下文(128K、1M)训练的首选;单节点则 `a2a` 或 `p2p` 足够。

---

## 5. 开销分析(汇总)

### 5.1 显存

| 项 | CP 的影响 |
|----|----------|
| 序列相关激活 | **`÷cp`**(每卡只存 `s/cp` token) |
| attention 分数矩阵 | `O(s²)` → 每卡 `O(s²/cp)`(ring/all-gather);Ulysses 下每卡 `a/cp` 头 |
| 权重 | **`×1`,不切**(CP 不是模型并行) |
| 优化器状态 | `÷cp`(配合分布式优化器,梯度/状态可按 CP 再分片) |

### 5.2 通信量与暴露

| `cp_comm_type` | 通信原语 | 是否可重叠 | 适合 |
|----------------|---------|-----------|------|
| `p2p`(ring) | `cp-1` 步环形 P2P | ✅ 异步,与计算重叠 | 长序列、跨节点默认 |
| `all_gather` | 同步 all-gather KV | ❌ 难重叠 | 简单、小 CP |
| `a2a`(Ulysses) | 每 attention 2 次 A2A | ⚠️ 部分 | NVLink 域内、head 够分 |
| `a2a+p2p` | 节点内 A2A + 节点间 P2P | ✅ 跨节点段可重叠 | 多节点超长序列 |

### 5.3 CP 的"等效气泡"

CP 无流水线气泡。低效来源:
1. **因果负载不均衡** —— 由 zigzag 切分(§2.2)拉平。
2. **attention 通信暴露** —— ring / a2a+p2p 用异步 P2P 把它藏到计算后;all_gather 藏不住。

CP 效率高度依赖"通信能否与 attention 计算重叠"(README Guideline 5 原话:"CP efficiency depends on overlapping communication with computation")。

---

## 6. 适用场景及选型

### 6.1 何时用 CP

| 场景 | 是否用 CP | 原因 |
|------|----------|------|
| 序列 < 8K | ❌ 一般不用 | attention `O(s²)` 还没成瓶颈,CP 的通信纯亏 |
| 序列 ≥ 8K(README 阈值) | ✅ 用 CP | 把 attention 激活/算力 `÷cp` |
| 超长上下文 128K / 1M | ✅ 必须用 CP | 否则 attention 单卡绝无可能 |
| 模型权重放不下 | ❌ CP 救不了 | CP 不切权重;用 TP/PP/EP |

### 6.2 cp_comm_type 选型决策树

```
要训长序列(s ≥ 8K)?
└─ 是 ──► 开 CP,选 cp_comm_type:
          │
          ├─ 跨多节点超长序列(128K/1M)?
          │   └─ 是 ──► ④ a2a+p2p(节点内 A2A + 节点间 P2P,各用所长)
          │
          ├─ 单节点 / NVLink 域内,且 head 数 ≥ cp?
          │   └─ 是 ──► ③ a2a(Ulysses,换 head 轴,attention 见全序列)
          │
          ├─ 长序列、要异步重叠通信?
          │   └─ 是 ──► ① p2p(ring attention,P2P 异步可重叠,通用默认)
          │
          └─ 要实现简单 / 小 CP / 特性强制?
              └─ 是 ──► ② all_gather(全收 KV,逻辑最简,通信不可重叠)

并行组合(README Guideline 5):
  - CP 与 TP/PP/DP/EP 正交,可任意叠加
  - MoE:attention 用 TP×CP×DP,CP 折叠进 EP(MoE Parallel Folding,见 megatron_ep_analysis.md §6)
  - CP 与 TP 同属高带宽通信,优先压在 NVLink 域内
```

### 6.3 一句话总结

- **CP 的本质**:把序列维 `s` 切成 `cp` 段,专治 attention 的 `O(s²)` 显存/算力墙;切激活、不切权重。
- **唯一的通信热点是 attention**(其余算子 token-wise 零通信);`cp_comm_type` 决定 K/V 怎么搬。
- **四种策略**:`p2p` 环形异步(通用默认)、`all_gather` 全收(最简但不可重叠)、`a2a` Ulysses 换 head 轴(NVLink 域内)、`a2a+p2p` 分层(多节点超长序列)。
- **因果负载均衡**靠 zigzag(`2cp` 块,早+晚配对)拉平;**通信暴露**靠异步 P2P 重叠掩盖。

---

## 7. 动态上下文并行(Dynamic CP)

> [!update] 2026-06-16 · dev@232c478d4
> ee3f1ff 之后引入并持续完善 **动态上下文并行(Dynamic Context Parallelism, DCP)** —— 在 THD(packed varlen)训练中**逐 microbatch / 逐样本动态选择 CP 度**,而非全程固定 `cp`(#4226 / #5215 / #5123)。

### 7.1 动机

§6 之前的 CP 把序列**固定**切成 `cp` 段。但 packed varlen 训练里样本长度差异极大(示例数据集取 lognormal,`128~8192` token)。固定 CP 会把**每个样本都摊到全部 `cp` 卡**:短样本被强行切碎,attention 通信(K/V 搬运)纯亏,还把短序列补到全 CP 尺寸浪费算力。

**DCP 的思路**:按样本长度给每个 microbatch 选一个**恰好够用的 local CP 度** —— 短样本用 1 卡(`local_cp_size=1`,等于不切 CP),中等样本用 2 卡,长样本才用满 `cp`。`examples/dynamic_context_parallel/README.md` 实例:`--max-seqlen-per-dp-cp-rank 2048` 下,≤2048 用 1 rank、≤4096 用 2 rank、≤8192 用 4 rank。

### 7.2 机制(源码)

- **`PackedSeqParams` 新增两字段**(`packed_seq_params.py:23-24`):`local_cp_size`(本 microbatch 实际 CP 度)与 `cp_group`(对应的 CP 进程子组),由调度器 `DefaultDynamicCPScheduler` 按样本长度算出。
- **`resolve_cp_group(static_cp_group, packed_seq_params)`**(`packed_seq_params.py:69`,#4226):统一"**优先用 `packed_seq_params.cp_group`,否则回退建图期静态 CP 组**"的解析逻辑,供 `GPTModel`、`GatedDeltaNet`、MTP 层共用(此前各处分散硬编码 `self.pg_collection.cp`)。
- **TE attention 接入**(`extensions/transformer_engine.py:1798`):`TEDotProductAttention.forward` 按 `packed_seq_params.local_cp_size` **切换 TE 内部的 CP 组** —— `local_cp_size==1` → `set_context_parallel_group(None,...)`(该样本关 CP);否则换成 `packed_seq_params.cp_group`。
  - **#5215 修复**(`transformer_engine.py:1886`):forward **开头先保存原始 CP 组**(`_te_orig_cp_group`),**结尾再恢复**。否则被换掉的动态 CP 组会**泄漏**到后续不带 dynamic CP 的 microbatch,导致 attention 用错组、结果错误。
- **dispatcher 兼容**:sequence packing(THD)原仅支持 `alltoall` dispatcher,现已放宽到 `flex`(#4816,见 [[megatron_ep_analysis]] §③ 增量更新);THD 下 HybridEP 会把各 rank 不齐的 token 数补齐到组内最大值。
- **CUDA Graph 守卫**(#4226,`training/utils.py`):`cuda_graph_impl=full_iteration` 与 `cu_seqlens`(THD 变长)互斥,`_broadcast_cu_seqlens` 直接短路返回 `None`。

### 7.3 入口与示例

- 开关:`--dynamic-context-parallel --sequence-packing-scheduler default_dynamic_cp --max-seqlen-per-dp-cp-rank N`。
- 基准示例(#5123):`examples/dynamic_context_parallel/`(`benchmark_dcp.sh`),对比 `dp_balanced` 定长 packed 与 DCP 两条 run,复用 `pretrain_gpt.py` + `MockVarlenDataset`,不引入新模型/数据集类。
- **数据集/调度器侧的完整机制**(packing、`max-seqlen-per-dp-cp-rank` 分配)见 [[megatron_packed_dataset_dynamic_cp_analysis]];本节只覆盖 CP/attention 侧的接入。

---

*生成依据:`Megatron-LM` `dev` 分支 `ee3f1ff`(§7 增量基准 `dev@232c478d4`)。源码行号以对应 commit 为准。`p2p`/`a2a`/`a2a+p2p` 的实际 attention 内核位于 TransformerEngine,Megatron 透传 `cp_comm_type`;原生 `all_gather` 实现见 `dot_product_attention_context_parallel.py`。配套文档:`megatron_pp_schedulers_analysis.md`、`megatron_ep_analysis.md`、`megatron_tp_analysis.md`。*

## Related Pages

- [[megatron_pp_schedulers_analysis]] · [[megatron_ep_analysis]] · [[megatron_tp_analysis]] · [[megatron_packed_dataset_dynamic_cp_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
