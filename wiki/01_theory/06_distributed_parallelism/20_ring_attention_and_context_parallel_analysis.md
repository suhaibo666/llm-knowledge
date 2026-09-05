---
title: "Ring Attention 与上下文并行(Context Parallelism)—— 通用机制"
---

# Ring Attention 与上下文并行(Context Parallelism)—— 通用机制

> 层次:原理(principle)· 引擎无关
> 前置:[[10_collectives_analysis]](all-gather / reduce-scatter / all-to-all 代价)、[[13_tensor_sequence_parallel_analysis]](CP 在 TP/SP 之后的位置)
> **本页定位**:抽出 CP/Ring-Attention 在 Megatron-LM、torchtitan、MindSpeed、DeepSeek-V4(Megatron 实现)四份工程分析页里**重复讲的通用机制**——序列怎么切、因果负载怎么拉平、Ring/All-gather/Ulysses/分层混合四种通信调度怎么工作、通信量的代数关系。**本页是纯搬运合成**:每节骨架逐字取自四页中讲得最深/最完整的那一页,differences 与补充逐段注明来源;不新造任何机制。四框架各自的源码走读、性能实测、配置项与限制留在各自页面,只在本页留指针。
> 四份实现页(各自只保留框架特有内容 + 指回本页):[[../../02_engineering/02_train_frameworks/megatron-lm/13_megatron_cp_analysis|Megatron-LM CP]] · [[../../02_engineering/02_train_frameworks/torchtitan/13_torchtitan_cp_analysis|torchtitan CP]] · [[../../02_engineering/02_train_frameworks/mindspeed/20_mindspeed_context_parallel_analysis|MindSpeed CP]] · [[../../02_engineering/02_train_frameworks/megatron-lm/35_deepseek_v4_context_parallel_analysis|DeepSeek-V4 CP(Megatron 实现)]]
> 最后更新:2026-07-31

---

## 罗盘:一句话定位

**上下文并行(Context Parallelism,CP)** 把序列维 $S$ 切到 $cp$ 张卡,每卡只持 $S/cp$ 个 token 的 Q/K/V 与激活,专治 attention 的 $O(S^2)$ 显存/算力墙。难点恒在 attention:每个 query 要看到全序列 K/V,而 K/V 被切散在各卡——CP 的全部复杂度都在**如何在序列被切开的前提下,让每个 query 仍能算到对全序列的 attention**。这件事有四条通用解法(通信调度):Ring(P2P 环形轮转 + online-softmax)、All-gather(先收齐全 KV 再算)、Ulysses(A2A 换轴,attention 时看全序列但只算部分 head)、分层混合(节点内 A2A + 节点间 P2P)。四条解法之外还有两个横切问题:**因果 mask 下的负载不均衡**(靠折叠/头尾配对切分解决)和**因果 mask 下的算力浪费**(靠三分支裁剪跳过全空块解决)。

---

## 0. 记号约定

| 符号 | 含义 |
|------|------|
| $cp$ | CP 度 |
| $S$ | 全局序列长度;每卡持有 $S/cp$ |
| $b$ / $h$ / $d$ | micro-batch / hidden / head dim |
| $a$ / $a_{kv}$ | 注意力头数 / KV 头数(GQA 时 $a_{kv}<a$,$h=a\cdot d$) |
| $TP$ | 张量并行度(若头先被 TP 切,CP 在 $a/TP$ 头上再操作) |
| $u$ / $r$ | 分层混合下 Ulysses 子度 / Ring 子度($cp=u\cdot r$) |

*(合并自 `13_megatron_cp_analysis.md` §1.4 与 `20_mindspeed_context_parallel_analysis.md` §0.2;`u`/`r` 取自 MindSpeed,因分层混合一节要用)*

---

## 1. CP 的动机:attention 的 $O(S^2)$ 墙

> [!note] 关于「骨架取自 …」这类溯源注记为什么不再带节号
> 本页多处溯源注记只保留页名、不带节号,原因是**那些内容当年正是归一到本页的部分,源页随后把它们移出了**——归属仍然成立,但源页已无对应小节可指。
> - `13_torchtitan_cp_analysis.md`：该页在 2026-08 的重构（`4b789b5`）里改成了「路径演进 → 配置 → 输入 → attention 边界 → FSDP mesh → 组合矩阵」七节 + 五拍子节的结构，并主动移出了 ring 机制内容——`zigzag` / `rotater` / 稀疏掩码负载均衡 / 反向第二根环 / 量化算例在该页现已**全部零命中**，它自己也在 §2 写明「不应把旧 dispatcher 中的 `AsyncCollectiveTensor` 或 Ring 时序写成当前 TorchTitan 机制」。> - `20_mindspeed_context_parallel_analysis.md`：该页做过「归一瘦身」,「代数推导」「rank 分组」「量化论证」等串现已零命中(通信量代数移到了 §4.3,已按新号重指);仍存在的 §3.1-3.2、§5.1、§7.2 保留原节号。

> 骨架取自 `13_megatron_cp_analysis.md` §1(全节逐字为骨架,仅去掉 Megatron 专属的 `cp_comm_type` 提前指路)。

### 1.1 要解决的问题

self-attention 的注意力分数矩阵是 $[S,S]$,激活显存与计算量都是 $O(S^2)$。序列从 4K 拉到 128K,attention 激活暴涨 1024 倍——这是长序列训练的头号显存/算力墙。

其他并行轴都救不了它:

- **TP** 切 hidden/head 维,不切序列 → attention 分数矩阵仍是 $[S,S]$。
- **PP** 切层,不切序列。
- **DP** 切 batch,不切序列。

只有**把序列本身切开**,才能把单卡的 attention 负担从 $O(S^2)$ 降到 $O(S^2/cp)$。这就是 CP。

### 1.2 CP 的收益

- **激活显存 $\div cp$**:每卡只存 $S/cp$ token 的激活;attention 分数矩阵每卡 $O((S/cp)\cdot S)$(ring/all-gather)或更低。
- **能训练否则放不下的超长序列**:CP 是 128K/1M 上下文训练的关键使能技术。
- **与 TP/PP/DP/EP 正交**:可任意组合(§2)。

### 1.3 CP 不解决什么

- **不切权重**:每卡仍持完整模型权重(CP 不是模型并行)。权重显存要靠 TP/PP/EP。
- **引入 attention 通信**:K/V 或部分结果要在 CP 组内搬运(§5-§8 四种调度)。
- **非 attention 算子(LayerNorm、MLP、逐元素算子)都是 token-wise 的**:序列切开后各卡独立算、互不通信——CP 的全部通信都集中在 attention。这一点四份实现页(Megatron/torchtitan/MindSpeed/DeepSeek-V4)口径完全一致。

---

## 2. CP 与其它并行维度的组合关系

> 骨架取自 `13_megatron_cp_analysis.md` §1.3 + §1.4 + §6.2；进程组与其它并行轴的编排边界见 `17_megatron_parallelism_orchestration_analysis.md`。

### 2.1 CP 在并行体系中的位置

| 并行轴 | 切什么 | 峰值激活 | 权重 | 优化器 | 通信特征 |
|--------|--------|---------|------|--------|---------|
| TP | 单层权重矩阵 | 1/tp(SP) | 1/tp | 1/tp | 高频、关键路径 |
| EP | 专家(按个数) | ~1 | MoE 1/N | 1/N | 中 |
| PP | 层(按深度) | 1(VPP>1) | 1/N | 1/N | 中,点对点 |
| **CP** | **序列** | **1/cp** | **1(不切权重)** | **1/cp(分布式优化器)** | **中,仅 attention,可重叠** |
| DP | 批次 | 1 | 1 | 1/N(分布式优化器) | 低 |

关键:**CP 切激活与序列相关显存(÷cp),但不切权重**——与 TP 相反。

Megatron 中这些轴如何组合成正交进程组、又如何注入模型与训练循环，由 `17_megatron_parallelism_orchestration_analysis.md` 统一说明；本页只保留算法层面的显存与通信关系。

### 2.2 与 MoE 的组合:CP 折叠进 EP

对 MoE 模型,CP 对**专家层无意义**(token 独立处理,无需跨序列)。Megatron 的 MoE Parallel Folding 正是利用这一点:attention 用 $TP\times CP\times DP$,MoE 把 $CP$ **折叠进 $EP$**($ETP\times EP\times EDP$)。双 `RankGenerator` 的构造见 [[02_engineering/02_train_frameworks/megatron-lm/17_megatron_parallelism_orchestration_analysis|Megatron 并行组编排]],EP/ETP/EDP 的运行时所有权见 [[02_engineering/02_train_frameworks/megatron-lm/14_megatron_ep_analysis|Megatron EP 闭环]]。

### 2.3 正交叠加原则

- CP 与 TP/PP/DP/EP 正交,可任意叠加。
- CP 与 TP 同属高带宽通信,优先压在 NVLink/HCCS 域内。
- MindSpeed 在 TP-2D 场景下还有一处特殊耦合——CP 与 TP 的 y 方向合并成 `TensorParallelYUnionCP` 联合通信域,这是 MindSpeed 特有的实现细节,见 [[../../02_engineering/02_train_frameworks/mindspeed/20_mindspeed_context_parallel_analysis|20_mindspeed_context_parallel_analysis]] §1 附注。

---

## 3. 序列切分方案

### 3.1 朴素连续切分

> 骨架取自 `13_megatron_cp_analysis.md`(切分图示)+ `13_torchtitan_cp_analysis.md`(DTensor 级实现)。

```
全局序列 S = [tok 0 … tok S-1]   按 CP 切成 cp 段:
  CP rank 0: tok[0 : S/cp]          ← 各卡只持有本段的 Q/K/V
  CP rank 1: tok[S/cp : 2S/cp]
  …
  CP rank cp-1: tok[(cp-1)S/cp : S]
```

torchtitan/PyTorch 侧,这一步由 `distribute_tensor(buffer, mesh, [Shard(seq_dim)], src_data_rank=None).to_local()` 完成——把张量沿 `seq_dim` 均匀切成 `cp_size` 段,rank `r` 拿第 `r` 段。**这一步本身不产生通信**(`src_data_rank=None` 时 `distribute_tensor` 等价于本地 slice);真正的跨 rank 通信发生在 attention 计算阶段(§5-§8)。这一点对 Megatron/MindSpeed 同样成立——序列切分是纯本地 view/slice,通信只在 attention 内核里发生。

### 3.2 因果负载不均衡问题(三角形问题)

> 骨架取自 `13_torchtitan_cp_analysis.md`(最完整的量化算例)。

因果掩码下 $\text{mask}[i,j]=1 \iff \texttt{q\_idx} \ge \texttt{kv\_idx}$,计算量正比于矩阵里 1 的个数。以 `seq_len=8, cp=2` 朴素均分为例:

```
            KV_index
   [1,0,0,0,0,0,0,0]
   [1,1,0,0,0,0,0,0]
   [1,1,1,0,0,0,0,0]   rank0 → 1+2+3+4 = 10 次计算
   [1,1,1,1,0,0,0,0]
 ──────────────────────
   [1,1,1,1,1,0,0,0]
   [1,1,1,1,1,1,0,0]   rank1 → 5+6+7+8 = 26 次计算
   [1,1,1,1,1,1,1,0]
   [1,1,1,1,1,1,1,1]
```

**后段 rank 的 Q 行更靠下,要 attend 的 KV 更多**——rank1 工作量是 rank0 的 2.6 倍。在 ring attention 里所有 rank 每步同步,最慢的 rank 成为 straggler 拖垮整组。

`20_mindspeed_context_parallel_analysis.md` §2 给出了这一问题更严格的**渐近量化**:因果三角下,第 $j$ 块(共 $2cp$ 块)的 attention 算量 $\propto j+1$;朴素连续切让 rank $i$ 持 $\{2i,2i+1\}$ 两块、算量 $\propto 4i+3$,最重/最轻块之比 $\approx (4cp-1)/3$——即 straggler 把整组拖慢到约 $\tfrac{4}{3}cp\times$ 最轻卡,且这个比值随 $cp$ 线性恶化。

### 3.3 折叠/头尾配对负载均衡(Zigzag / Head-Tail)—— 核心机制

> 骨架取自 `20_mindspeed_context_parallel_analysis.md` §2(定量证明最严格)+ `13_torchtitan_cp_analysis.md`(最完整的量化算例)。两种代码实现(Megatron 的 `to_zz_mask_attn_bias` 与 torchtitan 的 `_HeadTailLoadBalancer`)并列给出——这是**同一个算法的两种独立代码实现**,Megatron/DeepSeek-V4/MindSpeed 叫它 **Zigzag**,torchtitan/PyTorch 叫它 **Head-Tail**。

**算法**:把序列切成 $2\cdot cp$ 个等长小块,给 CP rank $r$ 分配**第 $r$ 块(早/头,轻)和第 $2cp{-}1{-}r$ 块(晚/尾,重)**。于是每张卡都拿到"一个早块 + 一个晚块",计算量被拉平。

```
朴素切分(因果):  rank0 [▁▁▁] 算得少    rank3 [███] 算得多   ← 不均衡
zigzag/头尾切分:  每个 rank = 一个早块 + 一个晚块            ← 计算量拉平 ✅
  rank0 = 块0 + 块7   rank1 = 块1 + 块6   rank2 = 块2 + 块5   rank3 = 块3 + 块4
```

**为什么均衡**(MindSpeed 的定量证明):早块 $r$ 的算量 $\propto (r{+}1)$,晚块 $2cp{-}1{-}r$ 的算量 $\propto (2cp{-}r)$,配对后每 rank 总算量:

$$
(r+1) + (2cp - r) = 2cp + 1
$$

**与 $r$ 无关,完全均衡**——这是消除 straggler 的充分条件,也是后续因果块裁剪(§4)能砍掉近半算量的**前提**。

**具体算例**(torchtitan,`seq_len=8, cp=2`):4 个 chunk(每个长 2),头尾配对索引 `[0,7,1,6,2,5,3,4]`:

```
   rank0:chunk0(行0,1)+chunk3(行6,7) → 1+8+2+7 = 18
   rank1:chunk1(行2,3)+chunk2(行4,5) → 3+6+4+5 = 18    ← 完全均衡
```

**代码实现一:Megatron/DeepSeek-V4 的 `to_zz_mask_attn_bias`**(按 chunk 重排注意力偏置,`dot_product_attention_context_parallel.py:135`,`zz` = zigzag):

```python
chunked = attention_mask.chunk(dim=3, chunks=cp_size * 2)               # 切成 2·cp 块
zz_mask = [x for p in zip(chunked[:cp_size], reversed(chunked[cp_size:])) for x in p]
#         配对:(块0,块2cp-1)、(块1,块2cp-2)、… → 每个 CP rank 拿一对(早+晚)
```

**代码实现二:torchtitan/PyTorch 的 `_HeadTailLoadBalancer`**(生成重排索引,先 `torch.gather` 打乱数据再均匀切;`seq_len=8, cp=2` 时 `head_idx=[0,1]`、`tail_idx=[3,2]`,重排索引 `[0,7,1,6,2,5,3,4]`——与 Megatron 的分块配对是同一个数学结构,只是从"分块打包"变成"重排索引"两种代码路径)。直觉:第 $r$ 个头 chunk(轻)配第 $r$ 个尾 chunk(重),头尾之和对每个 rank 恒定。

> **`seq_len` 须被 $2\cdot cp$ 整除**:头尾配对要把序列切成 $2\cdot cp$ 个等长 chunk,这是该算法对序列长度的硬约束(torchtitan `seq_len_divisor`、MindSpeed `seq%(2cp)==0`,两页独立核实到同一约束)。

### 3.4 任意稀疏掩码下的处理时间均衡:PTRR

> 唯一来源 `13_torchtitan_cp_analysis.md`——四页中仅此一页覆盖非因果三角形的任意稀疏掩码负载均衡,作为 §3.3 zigzag/头尾算法的补充延伸,而非同一点的另一版本。

Head-Tail 假定掩码是标准因果三角(纯几何规则);面对**任意稀疏 `BlockMask`**(滑动窗口、文档掩码等)时无法用固定几何,必须**真去数每个 Q-block 的实际计算量**。**PTRR(Processing-Time based Round-Robin)**:

1. 从 `BlockMask` 取每个 Q-block 实际要算的 KV-block 数,作为"处理时间"。
2. 按处理时间降序排,每 `cp_size` 个一组做"蛇形(serpentine)正逆交替"分配——把最大的和次小的配在一起摊平,是经典 LPT(Longest-Processing-Time)多机调度的近似算法。

PTRR 返回逐样本的索引(每个样本的 `BlockMask` 不同,重排逐样本不同);Head-Tail 返回全 batch 共用的单一索引(纯因果与样本内容无关)。**负载均衡只改"数据怎么切"**,切分前一次 `gather` 重排,完全不碰 ring 通信本身——它与通信调度(§5-§8)正交。

### 3.5 位置编码切分的正确性不变量

> 唯一来源 `20_mindspeed_context_parallel_analysis.md` §2——四页中仅此一页显式指出这条隐形不变量,作为横切正确性约束补充。

RoPE 位置编码必须按**和 token 完全相同**的方式切到 CP rank,否则每卡拿到的 token 和它的旋转相位对不上、注意力全错。这是 CP 家族里最容易被忽略、却必须与 §3.3 切分严格对齐的一环——不论用 zigzag/头尾哪种代码实现,位置编码的切分逻辑都必须镜像 token 切分逻辑(逐算法各自实现,见 MindSpeed 页 `get_pos_emb_on_this_cp_rank`)。

---

## 4. 因果 mask 裁剪:把计算量本身砍掉

负载均衡(§3)解决的是"每 rank 算多少";这一节解决的是"能不能干脆不算"——利用 §3.3 折叠切分后每个 KV 块相对当前 Q 块**要么全可见、要么全不可见、要么只半可见**的性质,跳过全不可见块、只算半块的可见部分。

### 4.1 三分支裁剪算法

> 唯一深入来源 `20_mindspeed_context_parallel_analysis.md` §4.2/§4.3——四页中量化最完整、且给出显式三分支代码逻辑的版本;三分支裁剪机制本体在 §4.2,但"从朴素 $cp\cdot$(全块)降到约一半"这句精确量化陈述的原始措辞出自 §4.3 通信量代数节的 `[!tip]` 优化点 callout(该 callout 的③即引用§4.2 三分支),下方沿用其原话。

2·cp 配对切分(§3.3)让每个 KV 块相对当前 Q 块**要么全可见、要么全不可见、要么只半可见**,据此三分支裁剪:

- **对角块**($\texttt{q\_block\_id} = \texttt{kv\_block\_id}$):带因果掩码全算。
- **KV 在 Q 之前**:全可见,只取本地 KV **前半**、无需掩码。
- **KV 在 Q 之后**:只有 Q **后半**能 attend 该 KV 块。

**收益**:再叠加折叠切分的负载均衡,整环 attention 计算量从朴素 $cp\cdot$(全块)降到**约一半**——这是"折叠切分"与"块裁剪"两个机制叠加的结果:折叠切分保证均衡,块裁剪把均衡后仍然全空/半空的部分跳过不算。

### 4.2 等价视角:每步选择掩码 / SKIP

> 唯一来源 `13_torchtitan_cp_analysis.md`——与 §4.1 描述的是同一件事的另一种代码路径(逐步判定 vs 静态三分支),作为等价补充。

在 ring 主循环(§5)的每一步里,`_is_causal_behavior` 判定当前步该用什么掩码:第 0 步用标准因果(本地 Q×本地 KV);后续步若传来的 KV 全在 Q 之前 → 不掩码算满;若不开负载均衡且 KV 全在 Q 之后 → **直接 `SKIP`**(因果下全被 mask,跳过整步计算)。**开了折叠/头尾负载均衡后永远不会 SKIP**——因为每个 rank 的本地块含头+尾,任何一步都有非空计算;这正是 §4.1 三分支裁剪"没有全空步"这一性质在 ring 循环里的运行期体现。两种代码路径(MindSpeed 的静态三分支 vs torchtitan 的逐步判定)描述的是同一个因果裁剪原理。

### 4.3 变长/EoD 打包下的裁剪

> 唯一来源 `20_mindspeed_context_parallel_analysis.md` §4.2 后半——四页中仅此一页覆盖变长打包场景下的裁剪逻辑。

多条样本拼成一条 packed 序列(THD)训练时,2·cp 配对需要**逐子序列**做,而非对整条 packed 序列做一次。做法:对每段 `[prev_eod, eod]` 取中点 `mid`,把前半划给 KV、后半划给 Q;前向据此预生成 `q_index/kv_index/softmax_indices` 供反向复用。数据侧 packing 与动态 CP 的完整机制见 [[../../02_engineering/02_train_frameworks/megatron-lm/29_megatron_packed_dataset_dynamic_cp_analysis|29_megatron_packed_dataset_dynamic_cp_analysis]](框架实现细节,本页不展开)。

---

## 5. Ring 通信调度(P2P Ring Attention)

四种通信调度里最常用、异步开销最低的一种:Q 不动,K/V 在 CP 组内沿环形逐站传递,每步算一个局部 attention 用 online-softmax 累加。

### 5.1 机制与主循环

> 骨架取自 `13_torchtitan_cp_analysis.md`——四页中唯一给出完整可读伪代码 + 逐步演算图的版本。

核心循环(`_templated_ring_attention` 风格伪代码):

```python
rank = dist.get_rank(group);  size = dist.get_world_size(group)
sdpa_merger = _SDPAMerger(...)
rotater = _create_rotater(group, 2)

for i in range(size):                                     # size = cp_world_size 步
    if i > 0:
        next_kv = rotater.next_buffer()                   # (A) 取上一步发起的传输结果
        key, value = 从 next_kv 切出
    if i < size - 1:
        next_kv = rotater.exchange_buffers(next_kv)        # (B) 发起下一步要用的 K/V 传输(异步)
    is_causal_behavior = _is_causal_behavior(rank, size, i, is_causal)
    if is_causal_behavior == SKIP:
        continue
    out, lse, *rest = op(q, k, v, is_causal=..., **kwargs)  # (C) 当前步局部 attention
    sdpa_merger.step(out, lse, partial)                     # (D) 在线 softmax 合并
```

**核心思想**:Q 不动(每个 rank 永远只算自己那段 Q),K/V 在 CP 组内**环形传递**。第 $i$ 步 rank $r$ 手里的 K/V 来自 rank $(r-i) \bmod size$。跑满 $size$ 步后,每个 rank 的 Q 都和**全部** K/V 算过一遍,在线 softmax 把 $size$ 次局部结果合并成最终输出。

**四步演算图**($cp=4$):

```
初始(step 0):每个 rank 用自己的 K/V
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│rank0 │  │rank1 │  │rank2 │  │rank3 │
│Q0 KV0│  │Q1 KV1│  │Q2 KV2│  │Q3 KV3│
└──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘
   │ KV0     │ KV1     │ KV2     │ KV3      每步把当前 KV 发给右邻
   └───►─────┴───►─────┴───►─────┴──►─(绕回 rank0)

step 1:KV 环转一格,rank_r 收到 KV[(r-1) mod 4]
   rank0: Q0×KV3   rank1: Q1×KV0   rank2: Q2×KV1   rank3: Q3×KV2
step 2:rank0: Q0×KV2  rank1: Q1×KV3  rank2: Q2×KV0  rank3: Q3×KV1
step 3:rank0: Q0×KV1  rank1: Q1×KV2  rank2: Q2×KV3  rank3: Q3×KV0

N=4 步后每个 Q_r 都与 KV0..KV3 全部相乘过。
```

Megatron 侧的等价配置与拓扑见 `13_megatron_cp_analysis.md` §3.2：KV 沿 CP ring 传递、Q 留在本 rank；本节用 torchtitan 版本展开逐步演算。

### 5.2 在线 softmax 合并

> 骨架取自 `20_mindspeed_context_parallel_analysis.md` §4.4(公式最严格,显式给出数值稳定的合并式);torchtitan 的 Python 实现是同一公式的等价代码形态,并列给出。

每步产出局部 $(\text{out}_{\mathrm{cur}}, m_{\mathrm{cur}}, \ell_{\mathrm{cur}})$($m$=running max,$\ell$=log-sum-exp 的和项),按下式无误差并入累积量:

$$
\begin{aligned}
m
&\leftarrow \max(m_{\mathrm{prev}},m_{\mathrm{cur}}),\quad \ell \leftarrow e^{m_{\mathrm{prev}}-m}\ell_{\mathrm{prev}} \\
&\quad +e^{m_{\mathrm{cur}}-m}\ell_{\mathrm{cur}},\quad \text{out} \leftarrow \frac{e^{m_{\mathrm{prev}}-m}\ell_{\mathrm{prev}}}{\ell}\text{out}_{\mathrm{prev}} \\
&\quad +\frac{e^{m_{\mathrm{cur}}-m}\ell_{\mathrm{cur}}}{\ell}\text{out}_{\mathrm{cur}}
\end{aligned}
$$

torchtitan 侧(`_SDPAMerger.step`)是这一数学式的等价 PyTorch 实现,用 `sigmoid`/`logsigmoid` 改写同一组增量合并公式,并强制 `convert_to_f32=True` 全程 fp32 累加避免误差:

```python
out = out - sigmoid(block_lse - lse) * (out - block_out)
lse = lse - logsigmoid(lse - block_lse)
```

这就是 ring attention **不需要一次性持有完整 K/V** 的数学关键——用在线 softmax 把"分 $size$ 步、每步一段 K/V"的局部结果正确合并成"全量 K/V"的结果,不需要对 $S\times S$ 做全局 reduce。

### 5.3 通信掩盖:下一步传输与当前步计算重叠

> 骨架取自 `13_torchtitan_cp_analysis.md`——四页中对"为什么能重叠"给出最清晰的指令顺序说明。

这是 ring attention 性能的命门。看 §5.1 循环体的**指令顺序**:

```
for i in range(size):
    (A) next_buffer()        ← 收割上一步发起的传输(这里才 wait)
    (B) exchange_buffers()   ← 发起"下一步要用的 K/V"传输 —— 异步,不阻塞
    (C) op(q,k,v)            ← 当前步局部 attention 计算
    (D) sdpa_merger.step()   ← 合并
```

关键:**(B) 在 (C) 之前发起,且是异步的**——(B) 发起的集合通信与 (C) 的计算**在硬件上并行**,到下一轮的 (A) 才真正 `wait()`。

```
            step i              step i+1            step i+2
计算流:    [attn(i)        ]   [attn(i+1)      ]   [attn(i+2)      ]
通信流:    [exchange KV→i+1]   [exchange KV→i+2]   [exchange KV→i+3]
                ↑ 第 i 步发起的传输,与第 i 步计算并行,第 i+1 步开头收割
```

即:**第 $i$ 步发起第 $i+1$ 步要用的 K/V 传输,然后立刻算第 $i$ 步的 attention,传输与计算重叠;第 $i+1$ 步开头才收割传输结果。** 这一步计算的耗时把上一步发起的 K/V 传输延迟"藏"了进去——前提是单步计算时间 ≥ 单步传输时间(长序列、$S/cp$ 足够大时成立)。

**四份实现页在"用什么原语发起这次异步传输"上各不相同**,这是框架实现差异,不属于通用机制:
- torchtitan/PyTorch:靠 `torch.distributed._functional_collectives` 返回的 `AsyncCollectiveTensor` 延迟 wait,不手写 CUDA stream(见 `13_torchtitan_cp_analysis.md`)。
- Megatron(TE 路径):Megatron 负责把 CP group 与 `cp_comm_type` 交给 Transformer Engine；具体 stream/event 调度属于 TE 内部实现，不在本知识库冻结的 Megatron 源码树内(见 `13_megatron_cp_analysis.md` §3.1)。
- MindSpeed:`RingP2P.async_send_recv` 用 `isend`/`irecv` 按 `ring_rank % 2` 决定收发顺序避免死锁,`use_cp_send_recv_overlap` 时收发各走独立组(见 `20_mindspeed_context_parallel_analysis.md` §4.4)。

三种实现遵循的是同一条通用原则(异步发起下一步传输、wait 延迟到真正需要时),只是各自选的异步原语不同。

### 5.4 两种"收集"策略的分野:逐步 P2P vs 一次性 All-gather

> 唯一来源 `13_torchtitan_cp_analysis.md`——四页中仅此一页把 ring 循环本身的"取数"策略抽象成可插拔的 rotater,值得作为通用机制单独记录(注意与 §6 独立的 All-gather CP 模式是两个不同机制,见下方辨析)。

Ring 循环第 (B) 步"发起下一步要用的 K/V 传输"可以有两种底层实现(`rotate_method` 选择):

| | 逐步 P2P(`_AllToAllRotater`) | 一次性 All-gather(`_AllGatherRotater`,默认) |
|---|---|---|
| 原语 | 每步一次置换传输(P2P 式,送右邻) | 第一步一次 all-gather 收齐全部 K/V |
| 每步开销 | 传 1 份 K/V(共 $size-1$ 次) | 一次性传 $size$ 份,后续只是本地 `chunk()` |
| 重叠 | 真正逐步 P2P 重叠 | 仅第一次 all-gather 与第 0 步计算重叠 |

**辨析(不要与 §6 混淆)**:即使选了"一次性 All-gather"这个 rotater,ring 循环仍然跑满 $cp$ 步、每步做局部 attention + online-softmax 合并(只是"取数"从网络传输变成本地 `chunk()`)——这与 §6 描述的"All-gather CP"模式(先一次性收齐全 KV,再对本地 Q 做**单次**完整 attention、不逐步、不需要 online-softmax)是**两种不同的机制**,只是都用了 all-gather 这个通信原语,不可望文生义地等同。

### 5.5 反向传播:为什么要多一根梯度环

> 骨架取自 `13_torchtitan_cp_analysis.md`——四页中对反向"为什么需要第二根环"给出最清晰的原理说明。

Ring backward **同时维护两个环**:

```python
kv_rotater  = _create_rotater(group, 2)                        # K/V 前向轮转(同 forward)
dkv_rotater = _create_rotater(group, 2, method=ALL_TO_ALL)     # K/V 梯度轮转,强制 all-to-all
```

每步做两件事:

1. **K/V 本身环形轮转**(同 forward):重演 forward 时那对 $(Q_{\mathrm{local}}, KV_{\text{from rank }(r-i)})$ 才能算梯度。
2. **K/V 梯度的环形规约**:某段 K/V 在 forward 里被**每个** rank 的 Q attend 过,其梯度是所有 rank 局部贡献之和——一个 `Partial` 规约。ring backward 把 dK/dV 跟着 K/V 一起绕环,每经过一个 rank 就把该 rank 的局部贡献累加进去,绕满 $size$ 步后得到完整的 `grad_key/grad_value`。

`grad_query` 不需要环规约——Q 是每个 rank 独占的分片,直接累加即可。

> **为什么 `dkv_rotater` 强制 all-to-all 而不能用 all-gather**:梯度必须**逐 rank 顺序累加**(每步 = 上一 rank 的部分和 + 本 rank 贡献),all-gather 一次性收齐就没法做这个增量累加。

Megatron 的 TE 路径把反向实现委托给 TE，因此冻结的 Megatron 源码只能确认进程组与通信类型的交接，不能据此断言 TE 内部使用哪一种梯度环。MindSpeed 在通用双环基础上还实现了 outer/inner window 的反向 dKV 环，属其自身的分层扩展，留在 MindSpeed 页(§4.4)。

---

## 6. All-Gather CP:最简回退

不逐步环传,而是 attention 前直接把全序列 K/V 收到每张卡,再用本地 Q 对**完整** $S$ 的 KV 做**一次**完整 attention——逻辑最简单,不需要 online-softmax,不需要环形调度。

### 6.1 机制:head-stride 双缓冲 All-gather

> Megatron 原生(非 TE)all-gather 回退路径由 `13_megatron_cp_analysis.md` §3.3 统一维护；这里抽取其算法骨架。

不依赖 TransformerEngine 时,CP 通过 `AttentionFuncionWithContextParallel`(`torch.autograd.Function`)实现:forward 中 all-gather KV,backward 中 reduce-scatter 梯度。

**Forward 采用 double-buffering + head-stride 迭代**策略,按 head 分批 all-gather、边收边算以省显存:

```python
kv_buffer = torch.empty(
    (2, k.shape[0] * cp_size, k.shape[1], heads_k_stride, k.shape[3]),
    dtype=k.dtype, device=k.device,
)
kv_buffer_copy = torch.empty_like(kv_buffer)

# All-gather 第一个 head stride
k_0 = k[:, :, :heads_k_stride].contiguous()
v_0 = v[:, :, :heads_k_stride].contiguous()
comm.all_gather(kv_buffer_copy[0], k_0)
comm.all_gather(kv_buffer_copy[1], v_0)

for i in range(0, nheads_k, heads_k_stride):
    comm.wait()                                          # 等本批 KV 收齐
    kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

    if i < nheads_k - heads_k_stride:                     # 预取下一批 head 的 KV(异步)
        send_k = k[:, :, kvsl:kvsr].contiguous()
        send_v = v[:, :, kvsl:kvsr].contiguous()
        comm.all_gather(kv_buffer_copy[0], send_k)
        comm.all_gather(kv_buffer_copy[1], send_v)

    out_i, probs_i = eager_attn_fwd(q_i, k_i, v_i, attn_bias, ...)  # 本地 Q × 全序列 KV
```

四步概括:① 初始化 double-buffer KV 缓冲;② 发起首个 head stride 的 all-gather;③ 按 head stride 迭代——wait 当前批、swap buffer、异步发起下一批、用当前完整 KV 算 attention。这里确实存在 **head-stride 粒度的通信/计算 overlap**；它的粒度与 TE 内部的 CP 调度不同。

CP 下的 attention mask 同样需要 §3.3 的 zigzag 重排(`to_zz_mask_attn_bias`,与 §3.3 引用的是同一份源码)才能匹配 all-gather 后的 KV 顺序。

### 6.2 通信量与 head-stride overlap 边界

> 公式、KV buffer 代价与适用范围以 `13_megatron_cp_analysis.md` §3.3 的当前 Megatron 回退路径为准。

**通信量(每 head stride)**:

$$
\begin{aligned}
\text{Forward}
&= 2\times b\times S\times h_{\mathrm{k\_stride}}\times d\times \frac{cp-1}{cp} \quad(\text{AllGather}(K) \\
&\quad +\text{AllGather}(V))
\end{aligned}
$$

$$
\begin{aligned}
\text{Total}
&\approx 4\times \frac{n_{\mathrm{heads\_k}}}{\mathrm{head\_stride}}\times b\times S\times h_{\mathrm{k\_stride}}\times d\times \frac{cp-1}{cp}
\end{aligned}
$$

(这里的 $S$ 是每个 CP rank 持有的序列长度;总量随 head 数线性增长,效率低于 §5/§7/§8 的方案。)

**显存代价(不只是通信量的问题)**:All-gather 完成后,**每个 rank 都要物化完整的全局 KV**——不再是分片的 $S/cp$,而是完整 $S$。KV buffer 的显存占用量级为

$$
\text{KV buffer} \approx 2 \times S \times b \times h_{kv} \times d \quad (\text{K + V,按全局序列长度 } S)
$$

这正是 All-gather CP 相对 Ring/Ulysses"省显存"初衷的抵消项:Ring 任意时刻每 rank 只持 1~2 份 $S/cp$ 分片,All-gather 却让每 rank 都退化回持有 $O(S)$ 的 KV——序列越长、CP 度越大,这份 buffer 越不划算。

**重叠边界**:`AllGatherComm.all_gather` 以 `async_op=True` 发起；当前 head slice 在消费前必须 `wait()`，但下一 head slice 的 all-gather 会在当前 slice 的 attention 计算之前发起，因此二者能够局部重叠（`megatron/core/transformer/dot_product_attention_context_parallel.py:108-132`、`:173-224`）。这不是 TE 的内核级 CP 调度。`TransformerConfig` 当前 docstring 仍称 all-gather “not async, cannot be overlapped”（`megatron/core/transformer/transformer_config.py:1108-1111`），与执行代码存在口径冲突；行为判断以调用链为准，冲突账本见 `13_megatron_cp_analysis.md` §3.3。

**适用场景**:实现简单、调试友好;CP 较小、序列不极端时可用;部分特性(如某些 fallback 路径)甚至强制要求 all_gather。不推荐用于大 CP / 跨节点超长序列——同步 all-gather 的暴露会拖垮吞吐。

### 6.3 反向:ReduceScatter

> 反向路径同样归 `13_megatron_cp_analysis.md` §3.3 所有。

Backward 与 forward 结构对称,额外引入 **ReduceScatter** 对 dK/dV 梯度分片:

```python
dk_i = torch.zeros((k_i.shape[1] // cp_size, k_i.shape[0], k_i.shape[2], k_i.shape[3]), ...)
dv_i = torch.zeros((v_i.shape[1] // cp_size, v_i.shape[0], v_i.shape[2], v_i.shape[3]), ...)
torch.distributed.reduce_scatter_tensor(dk_i, _dk_i, group=pg)
torch.distributed.reduce_scatter_tensor(dv_i, _dv_i, group=pg)
```

即:反向也按 head stride 重复"AllGather(完整 KV 梯度计算所需)+ ReduceScatter(把 dK/dV 分片回各 rank)"。这与 §5.5 Ring backward 的"环形增量累加"是完全不同的机制——All-gather CP 的反向不需要环,直接一次 reduce-scatter 就把所有 rank 的局部贡献求和并分片。

---

## 7. Ulysses:头维 All-to-All 换轴

Ring 和 All-gather 都在"序列切分"这个轴上想办法;**Ulysses 换轴**:attention 计算本身在 **head 维**是天然并行的(各 head 独立)。于是 attention **前**用 all-to-all 把张量从"序列切分"重排成"head 切分"——每卡换成持有**完整序列**但只有 $a/cp$ 个 head;attention **后**再用 all-to-all 换回"序列切分"。每卡在 attention 时看到完整序列,无需环形、无需 online-softmax,只是 head 少了。

### 7.1 机制

> 骨架取自 `20_mindspeed_context_parallel_analysis.md` §3.1-3.2——四页中唯一给出源码级形状流转图 + 实际调用代码的版本(Megatron/DeepSeek-V4 侧的 Ulysses 实际内核在 TransformerEngine 里,只有概念/开销描述,无代码;torchtitan 不支持 Ulysses 路径)。

核心是一次 `all_to_all_single` 加两次 reshape/transpose:scatter 头维、gather 序列维,转置把头维换到散射轴、`all_to_all_single`、再转回。

```
输入  [S/cp, b, a,   d]        每卡:切序列、全部头
拆头  [S/cp, b, cp, a/cp, d]    把头维拆出 cp 份(待散)
a2a   跨 cp 卡交换(行↔列转置)
还原  [S,    b, a/cp, d]        每卡:全序列、1/cp 头  ← 在此做本地 attention
              │  本地 attention([S, b, a/cp, d])
输出  [S,    b, a/cp, d]  ──a2a(gather↔scatter)──►  [S/cp, b, a, d]
```

每层 **4 次大块 all-to-all**(Q、K、V 各一次,输出 O 一次),对称的反向 autograd 把 `scatter_idx/gather_idx` 对调再做一次 a2a。

> **计次口径差异,不作合并**:Megatron/DeepSeek-V4 页把 Q/K/V 三者的进场 a2a 算作合起来的"1 次批量 A2A"、加输出 1 次,合称"每个 attention 含 **2 次** all-to-all"(`13_megatron_cp_analysis.md` 调度器③.2);MindSpeed 侧 Q/K/V 各自单独发起 `_SeqAllToAll` 调用,合计"**4 次**"(源码级独立调用,`ulysses_context_parallel.py:197-199,214`)。两种计数针对同一机制,只是"是否把 Q/K/V 的传输算作 3 次独立调用还是 1 次批量调用"的实现粒度不同,不构成矛盾——本页按 MindSpeed 的源码级计数(4 次)为准,因其可直接对应到具体调用点。

### 7.2 通信量代数

> 唯一给出显式代数推导的来源:`20_mindspeed_context_parallel_analysis.md` §4.3(通信量代数;该页重构后此内容由原 §3.3 移至此)。deepseek_v4 页给出的是不区分 forward/backward、且未显式扣除 $/TP$ 因子的合并总量(§9 统一小节里对照说明差异来源)。

a2a 的本地张量是切序列布局 $[S/cp,b,a/TP,d]$,元素数 $\dfrac{b\,S\,h}{cp\cdot TP}$;每次 all-to-all 每卡发出其中 $\dfrac{cp-1}{cp}$。四次(Q/K/V/O):

$$
V_{\text{ulysses}} \;\approx\; 4\cdot\frac{cp-1}{cp}\cdot\frac{b\,S\,h}{cp\cdot TP}
$$

对照 Ring(§5,通信只随 KV 头 $h_{kv}$ 走):

$$
V_{\text{ring}} \;\approx\; 2\cdot\frac{cp-1}{cp}\cdot\frac{b\,S\,h_{kv}}{TP}
$$

比值:

$$
\frac{V_{\text{ulysses}}}{V_{\text{ring}}}=\frac{2}{cp}\cdot\frac{a}{a_{kv}}
$$

**选型结论**:**MHA**($a{=}a_{kv}$)下 $cp{>}2$ 时 Ulysses 通信量更小;**GQA**($a/a_{kv}$ 大,如 8)下要 $cp{>}16$ Ulysses 才划算——因为 Ulysses 要把 KV 头 `repeat_interleave` 补到 $a$ 头,通信跟 $h$ 走而非 $h_{kv}$(见 §7.3)。

### 7.3 GQA 头补齐与变长处理

> 唯一来源 `20_mindspeed_context_parallel_analysis.md`——四页中仅此一页覆盖这两个源码级细节。

**GQA 下 KV 头不足以被 $cp$ 整除时**,先 `repeat_interleave` 把 KV 头补齐到 $a$ 头:

```python
if seq_world_size > key.shape[scatter_idx] and query.shape[scatter_idx] % key.shape[scatter_idx] == 0:
    key   = key.repeat_interleave(query.shape[scatter_idx] // key.shape[scatter_idx], dim=scatter_idx)
    value = value.repeat_interleave(query.shape[scatter_idx] // value.shape[scatter_idx], dim=scatter_idx)
```

这正是 §7.2 中"Ulysses 通信量跟 $h$ 走而非 $h_{kv}$"的源码级原因。

**不等长 a2a**:序列被 mask 成不能被 $cp$ 整除时,等分 reshape 会崩。解法是持有一个可注入的尺寸计算器,从 attention mask 的真实序列长度算出 a2a 输出的实际尺寸,forward 据此走支持 `split_sizes` 不等切的 a2a,算完再按各 rank 的实际尺寸 reshape 回去——这让 Ulysses 能处理变长/稀疏序列而不强制 padding。

### 7.4 约束与适用场景

- **约束**:`a % (cp·TP) == 0`(头数要够分,`transformer_config.py:936` 文档同样要求 $a_{kv}\ge cp$);头不够分时退回 Ring 或 All-gather。
- **适用**:head 数足够多、序列长但没到极端跨多节点;A2A 在 NVLink/HCCS 域内效率高 → 适合**单节点或高带宽域内**的 CP;head 不够分(GQA、$a_{kv}<cp$)时不适用。

---

## 8. 分层混合:节点内 A2A + 节点间 P2P

Ulysses(A2A)适合高带宽域内,Ring(P2P)适合跨节点异步重叠。大规模训练 CP 往往**跨多节点**:节点内带宽高、节点间带宽低。分层混合的思路是把 CP 组拆成两层,**各用所长**:低层(节点内)用 A2A 吃满高带宽,高层(节点间)用 P2P 环形异步重叠、扛低带宽。

### 8.1 动机

> 骨架取自 `13_megatron_cp_analysis.md` §3.5(概念/拓扑动机)+ `20_mindspeed_context_parallel_analysis.md`(为什么比纯任一种好的量化论证)。

```
CP 组 = 低层(节点内)× 高层(节点间)

  节点 0                          节点 1
 ┌───────────────┐              ┌───────────────┐
 │ rank0  rank1  │  ◄─P2P ring─► │ rank2  rank3  │   高层:跨节点 P2P(异步重叠)
 │   └─A2A─┘     │              │   └─A2A─┘     │   低层:节点内 A2A(高带宽)
 └───────────────┘              └───────────────┘
```

**为什么比纯任一种好**(MindSpeed 的量化论证):纯 Ulysses 在 $cp$ 卡上跨节点做,4 次 a2a 是带宽受限的,而跨节点带宽常比节点内低约一个量级,a2a 整段暴露在关键路径上;纯 Ring 在 $cp$ 卡上做,$cp{-}1$ 步 P2P 的延迟逐步叠加。分层混合让 **a2a 永不跨节点**——只在节点内 $u$ 卡上做,吃满高带宽域;**跨节点流量收敛为外层 Ring 的 $r{-}1$ 步可掩盖 P2P**,体量 $\propto h_{kv}$ 且能被计算掩盖,而不是把整个 a2a 压到慢链路上。两根轴各用所长:高带宽段交给 a2a,高延迟段交给可重叠 P2P。

### 8.2 分组构造机制(N 级分层)

> Megatron 的分层组构造由 `13_megatron_cp_analysis.md` §3.5 维护；它描述 rank 分组与向 TE 的 group 交接。TE 内部如何逐级执行通信不在冻结的 Megatron 源码树内。

通过 `hierarchical_context_parallel_sizes` 参数创建**多层级**的 CP 子组,以匹配集群的物理拓扑(不限于两级):

> **示例**:CP size = 16,`hierarchical_context_parallel_sizes = [2, 2, 4]`:
> - Level-1(NVLink):8 个子组,每组 2 GPU — [g0,g1], [g2,g3], ...
> - Level-2(NVLink):8 个子组,每组 2 GPU — [g0,g2], [g1,g3], ...
> - Level-3(IBLink):4 个子组,每组 4 GPU — [g0,g4,g8,g12], ...
>
> 低层通信走 NVLink(~600GB/s),高层走 IB(~50GB/s),与分层通信配合实现物理拓扑感知的调度。

**分层子组的创建算法**:用 einops 的 `rearrange` 对 rank 列表做张量维度分解:

```python
rearranged_ranks = einops.rearrange(
    np.array(ranks),
    "(l s u) -> (l u) s",
    u=int(np.prod(hierarchical_group_sizes[:level])),
    s=hierarchical_group_sizes[level],
    l=int(np.prod(hierarchical_group_sizes[level + 1:])),
).tolist()
```

这种基于张量维度分解的方法避免了手动枚举各级子组的复杂性,天然支持任意层级深度(不限于二级)。

> [!warning] 源码可见性边界
> Megatron 会创建分层 CP groups，并在 `a2a+p2p` 配置下把低层 A2A group 与高层 P2P group 交给 TE；冻结仓库并不包含 TE kernel 的逐层 stream、event 与传输顺序。因此不能从 Megatron 源码推出“先逐级 A2A、再跨节点 P2P”之类的具体运行时序列，也不能据此计算每一级实际承担的字节数。这里的分层图只表达拓扑意图，执行细节需以对应 TE 版本为准。

### 8.3 二级具体实例:Ulysses × Ring 的 rank 布局

> 唯一来源 `20_mindspeed_context_parallel_analysis.md` §5.1——四页中仅此一页给出二级混合(内层 Ulysses、外层 Ring)的具体 rank 分组代码,作为 §8.2 通用 N 级机制在二级场景下的一个具体实例。

组构造把 CP 组分解为 $cp = u\times r$($u$=Ulysses 子度,$r$=Ring 子度),关键是 rank 的摆放("把 Ulysses ranks 放在同一节点内"):

```python
for m in range(ring_degree):                    # ulysses 子组:连续 u 个 rank(尽量同节点)
    ulysses_ranks = [ranks[idx] for idx in range(m*ulysses_degree, (m+1)*ulysses_degree)]
for m in range(ulysses_degree):                 # ring 子组:跨步 stride=u(跨节点)
    ring_ranks = [ranks[idx] for idx in range(m, len(ranks), ulysses_degree)]
```

```
节点0 [r0 r1 | r2 r3]   节点1 [r4 r5 | r6 r7]      u=2, r=4(cp=8)
  └a2a┘ └a2a┘              └a2a┘ └a2a┘             内层 Ulysses:同节点连续 rank
   r0 ──P2P──── r2 ──P2P──── r4 ──P2P──── r6        外层 Ring:跨节点 stride=u
   r1 ──P2P──── r3 ──P2P──── r5 ──P2P──── r7
```

约束:$r=cp/u>1$ 且整除、$a \% (u\cdot TP)==0$。

---

## 9. 通信量比较的口径边界

通信量只有在统计口径固定后才能横向比较：至少要同时声明统计的是逻辑 payload 还是总线收发字节、只计 forward 还是 forward+backward，以及 head 数是否已经经过 TP 切分。本页分别在 §5.2、§6.2、§7.2 与 §8 保留各机制原始来源下的公式或定性边界，不再把不同口径的系数拼成一张“统一”表。

DeepSeek-V4 当前 Megatron 路径也不能用一个固定倍数概括。源码可确认的是两段通信链：左边界 hidden rows 的 autograd P2P（接收后才在本地投影 boundary KV），以及压缩后的 Indexer-K/KV gather；具体张量、形状与约束见 `35_deepseek_v4_context_parallel_analysis.md` §1-§7。它们既不是标准 MHA 的单一 KV ring，也不能从 head 数直接推导“约 128 倍”端到端通信或性能收益。

---

## 10. 动态上下文并行(Dynamic CP)—— 通用机制部分

> 通用机制以 `13_megatron_cp_analysis.md` §4 为准；DeepSeek-V4 的具体 group 切换/恢复点及额外 shape、kernel 约束见 `35_deepseek_v4_context_parallel_analysis.md` §4 与 §7。

### 10.1 动机

Megatron 的静态 CP 把序列**固定**切成 $cp$ 段。但 packed varlen 训练里样本长度差异极大,固定 CP 会把**每个样本都摊到全部 $cp$ 卡**:短样本被强行切碎,attention 通信纯亏,还把短序列补到全 CP 尺寸浪费算力。Dynamic CP(DCP)按样本长度给每个 microbatch 选一个**恰好够用的 local CP 度**——短样本用 1 卡(等于不切 CP),中等样本用 2 卡,长样本才用满 $cp$。

### 10.2 机制

- `PackedSeqParams` 新增两字段:`local_cp_size`(本 microbatch 实际 CP 度)与 `cp_group`(对应的 CP 进程子组),由调度器按样本长度算出。
- 统一的 CP 组解析逻辑:"优先用 `packed_seq_params.cp_group`,否则回退建图期静态 CP 组"。
- **TE attention 接入**:forward 按 `packed_seq_params.local_cp_size` **切换** TE 内部的 CP 组——`local_cp_size==1` 时该样本关闭 CP;否则换成 `packed_seq_params.cp_group`。DeepSeek-V4 页给出的是这一切换逻辑的具体代码形态:

```python
_orig_cp_group = self.pg_collection.cp
if packed_seq_params is not None and packed_seq_params.local_cp_size is not None:
    assert packed_seq_params.cp_group is not None, "cp_group must be set in dynamic-cp mode"
    self.pg_collection.cp = packed_seq_params.cp_group

# ... attention compute ...

self.pg_collection.cp = _orig_cp_group   # 恢复原始 CP 组
```

  > **正确性坑**:forward **开头先保存原始 CP 组**、**结尾再恢复**,否则被换掉的动态 CP 组会**泄漏**到后续不带 dynamic CP 的 microbatch,导致 attention 用错组、结果错误(Megatron 侧 #5215 修复记录了这个坑)。
- 开关:`--dynamic-context-parallel --sequence-packing-scheduler default_dynamic_cp --max-seqlen-per-dp-cp-rank N`。

### 10.3 与本页其它机制的关系

Dynamic CP 是"要不要切、切多少度"这一决策层的机制,与 §3-§9 描述的"切开之后怎么通信"是两个不同层次的问题——一旦某 microbatch 决定用 $\texttt{local\_cp\_size}$,后续走的仍然是 §5-§8 的某一种通信调度。

---

## 11. 各框架实现差异速览

以下是各框架在通用机制之上的**框架特有**内容,详见各自页面(不在本页复述):

| 框架 | 页面 | 框架特有内容 |
|------|------|------|
| Megatron-LM | [[../../02_engineering/02_train_frameworks/megatron-lm/13_megatron_cp_analysis\|13_megatron_cp_analysis]] | `cp_comm_type` 四选一配置接口、TE 透传机制、动态 CP 完整源码级细节(`resolve_cp_group`、CUDA Graph 守卫)、选型决策树 |
| torchtitan | [[../../02_engineering/02_train_frameworks/torchtitan/13_torchtitan_cp_analysis\|13_torchtitan_cp_analysis]] | `_context_parallel_shard` 的 DTensor 调用链、SDPA-ring 与 FlexAttention-allgather 两条路径的取舍、functional collectives 异步实现细节 |
| MindSpeed | [[../../02_engineering/02_train_frameworks/mindspeed/20_mindspeed_context_parallel_analysis\|20_mindspeed_context_parallel_analysis]] | 五算法运行期分派脊柱(`CPDotProductAttentionImpl.forward`)、双环(outer/inner window)结构、Adaptive CP 调度驱动、KV-cache CP 显存换通信、CP×TP-2D 合并域 |
| DeepSeek-V4(Megatron 实现) | [[../../02_engineering/02_train_frameworks/megatron-lm/35_deepseek_v4_context_parallel_analysis\|35_deepseek_v4_context_parallel_analysis]] | 左边界 autograd P2P、压缩 Indexer-K/KV gather、CP-aware RoPE、THD contiguous 布局、Dynamic CP group 切换/恢复，以及 fused/eager 选择门槛 |

---

## Related Pages

- [[10_collectives_analysis]] —— 集合通信原语与 α-β 代价模型(本页所有通信量公式的前置)
- [[13_tensor_sequence_parallel_analysis]] —— TP/SP/CP 原理总览,CP 在其中的位置
- [[14_expert_parallel_analysis]] —— EP:CP 与 MoE 的组合(Parallel Folding)
- [[../../02_engineering/02_train_frameworks/megatron-lm/13_megatron_cp_analysis|13_megatron_cp_analysis]] —— Megatron-LM 实现差异
- [[../../02_engineering/02_train_frameworks/torchtitan/13_torchtitan_cp_analysis|13_torchtitan_cp_analysis]] —— torchtitan 实现差异
- [[../../02_engineering/02_train_frameworks/mindspeed/20_mindspeed_context_parallel_analysis|20_mindspeed_context_parallel_analysis]] —— MindSpeed 实现差异
- [[../../02_engineering/02_train_frameworks/megatron-lm/35_deepseek_v4_context_parallel_analysis|35_deepseek_v4_context_parallel_analysis]] —— DeepSeek-V4 实现差异
- [[01_theory/06_distributed_parallelism/index|分布式并行原理]] —— 分布式并行原理目录索引
