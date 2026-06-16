# Megatron-LM 流水线并行补遗(PP Supplements)

> 代码基准:`Megatron-LM/` 子仓库 `dev` 分支,commit `ee3f1ff`
> 核心文件:`megatron/core/pipeline_parallel/` 下的 `p2p_communication.py`、`fine_grained_activation_offload.py`、`hybrid_cp_schedule.py`、`bridge_communicator.py`、`multimodule_communicator.py`
> 配套阅读:`pp_schedulers_analysis.md`(主文档)、`parallelism_orchestration_analysis.md`
> 定位:`pp_schedulers_analysis.md` 只讲了 `schedules.py` 的 5 个调度器,把 `pipeline_parallel/` 目录其余 4 块跳过了。本文补齐。

---

## 0. 总览

`pipeline_parallel/` 目录里,主文档覆盖的是 `schedules.py` + `combined_1f1b.py`。剩下 4 块各管一摊:

| # | 主题 | 文件 | 一句话 |
|---|------|------|--------|
| 1 | P2P 通信内部 | `p2p_communication.py` | 调度器用的 `send/recv_*` 接口的底层实现 |
| 2 | 细粒度激活换出 | `fine_grained_activation_offload.py` | 把激活异步搬到 CPU 内存,用 PCIe 带宽换显存 |
| 3 | 混合 CP 动态调度 | `hybrid_cp_schedule.py` | 变长序列下按长度动态分配 CP 数、均衡负载 |
| 4 | 多模块/多模态流水线 | `bridge_communicator.py` + `multimodule_communicator.py` | 连接并行度不同的子模型(VLM 等) |

---

## 1. P2P 通信内部(`p2p_communication.py`)

### 1.1 动机

`pp_schedulers_analysis.md` 里调度器②③④反复用 `recv_forward` / `send_forward_recv_backward` / `send_forward_backward_recv_forward_backward` 等,它们都封装在 `P2PCommunicator`(`p2p_communication.py:140`)里。但底层的"一次 P2P 到底怎么发"有两条路径,且**直接决定调度器④ `overlap_p2p_comm` 能不能开** —— 这是主文档没讲的。

### 1.2 两种 P2P 实现

**① `_batched_p2p_ops`(`p2p_communication.py:17`)—— `batch_p2p_comm=True`(默认)**

把本次要做的所有方向(send_prev / recv_prev / send_next / recv_next)打包成一组 `P2POp`,一次 `batch_isend_irecv` 发出:

```python
ops = []
if tensor_send_prev is not None: ops.append(P2POp(isend, tensor_send_prev, prev_rank, group))
if tensor_recv_prev is not None: ops.append(P2POp(irecv, tensor_recv_prev, prev_rank, group))
... # send_next / recv_next 同理
reqs = torch.distributed.batch_isend_irecv(ops)
```

简单、规整;但**不能与 `overlap_p2p_comm` 共存**(`transformer_config.py` 有断言:二者互斥)。

**② `_p2p_ops`(`p2p_communication.py:55`)—— `batch_p2p_comm=False`**

逐个 `isend`/`irecv`,关键是**按 even/odd rank 错开收发顺序以避免死锁**:

```python
if group.rank() % 2 == 0:        # 偶数 rank:先 send_next,再 recv_prev,再 send_prev,再 recv_next
    isend(tensor_send_next, next_rank); irecv(tensor_recv_prev, prev_rank); ...
else:                            # 奇数 rank:先 recv_prev,再 send_next,…(与偶数互补)
    irecv(tensor_recv_prev, prev_rank); isend(tensor_send_next, next_rank); ...
```

若所有 rank 同时先 `isend`,在同步语义下会**互相等待对方 recv 而死锁**;even/odd 错开保证每个 send 都有对侧的 recv 已就位。**这条路径是调度器④ `overlap_p2p_comm` 的必需实现** —— 它返回带名字的 `reqs` 字典(`send_next`/`recv_prev`/…),调度器把这些 handle 留到后面再 `wait`,从而把通信移出关键路径。

还有一个细节(`:67`):当 PP 组只有 2 个 rank 时,用全局 `WORLD` group 跑其中一个方向 —— 因为单个 NCCL 通信器内的两个 P2P 会串行化,借 WORLD group 让两个独立方向真正并行。

### 1.3 变长序列:先换形状再换张量

`_communicate_shapes`(`:186`):当 `config.variable_seq_lengths=True`(每个 microbatch 序列长度不同),收发双方无法预知张量形状,于是**先用一轮小 P2P 交换形状元数据,再按收到的形状分配缓冲、做真正的张量 P2P**。固定序列长度时这步跳过(形状由 `get_tensor_shapes` 静态算出)。

### 1.4 衔接

```
batch_p2p_comm=True  → _batched_p2p_ops  → 调度器②③ 默认路径
batch_p2p_comm=False → _p2p_ops(even/odd)→ 调度器④ overlap_p2p_comm 必需
variable_seq_lengths → 每次 P2P 前多一轮 _communicate_shapes
```

---

## 2. 细粒度激活换出(`fine_grained_activation_offload.py`)

### 2.1 动机与解决的问题

省激活显存有两条路:
- **重计算(recompute)**:前向丢弃激活,反向重算 —— **用算力换显存**。
- **换出(offload)**:前向把激活异步拷到 CPU 内存,反向需要时再拷回 —— **用 PCIe 带宽换显存**。

二者正交,可叠加。重计算的代价是多一遍前向计算;offload 的代价是 D2H/H2D 传输 —— 只要**用独立 CUDA 流让传输与计算重叠**,这个代价就能基本藏住。`schedules.py` 在每步收尾调 `off_interface.reset()`,与 PP/VPP 调度天然兼容。

### 2.2 机制

**`OffloadTensorPool`(`:99`)—— pinned-memory 缓冲池**:CPU 侧用 pinned memory(锁页内存,D2H/H2D 必需的高速通道)。池化复用:相同 `(shape, dtype)` 的 CPU 缓冲反复借还,避免每步重新分配 pinned memory(分配很慢)。

**`OffloadTensorGroup`(`:333`)—— 一组激活 + 同步 event**:把一个模块的激活打包成 group,持有 `offload_event` / `reload_event` 两个 CUDA event。前向 D2H 拷完 `record_offload_event`;反向用之前 `wait_reload_event` 确保 H2D 拷回完成才用。

**模块级粒度**:`--offload-modules` 选择换哪些模块的激活(`attn_norm` / `core_attn` / `attn_proj` / `mlp_norm` / `expert_fc1` / `moe_act`)。不是整层一刀切,而是挑激活大、重计算又贵的模块换出。

```
前向:  模块计算 ──► 激活 ──D2H 异步(side stream)──► CPU pinned 缓冲
                          └─ record offload_event,GPU 激活随后释放
        (D2H 与后续模块的前向计算重叠)

反向:  需要该激活前 ──H2D 异步(side stream)──► GPU
                          └─ wait reload_event 后才参与反向
        (H2D 与前面模块的反向计算重叠 —— 需提前预取)
```

### 2.3 开销与适用场景

| 维度 | 说明 |
|------|------|
| 省显存 | 换出模块的激活不占 GPU 显存 |
| 代价 | D2H/H2D PCIe 带宽;重叠不好则拖慢;占用 CPU pinned 内存 |
| vs 重计算 | 重计算换算力、offload 换带宽;算力紧张时选 offload,PCIe 紧张时选重计算,可叠加 |
| 适用 | 显存墙严重、又不想全量重计算;`docs/.../fine_grained_activation_offloading.md` 有详述 |

---

## 3. 混合上下文并行动态调度(`hybrid_cp_schedule.py`)

### 3.1 动机与解决的问题

`cp_analysis.md` 讲的是**固定 CP 度**:所有样本一律切 `cp` 份。但**变长序列训练**(SFT、文档级语料、多分辨率多模态)里样本长度差异巨大 —— 对所有样本用同一个 CP 不合理:短样本用大 CP 是浪费,长样本用小 CP 又放不下、且 attention 的 `O(S²)` 负载在 DP×CP 组间严重不均。

`hybrid_cp_schedule.py` 的动机:**按样本长度动态决定它的 CP 度,并把样本打包成各 DP×CP 组工作量均衡的批**。

### 3.2 机制(`BalancedCPScheduler`,`:14`)

```python
def gpus_needed(self, seq_len):              # 该样本需要几张卡做 CP
    return max(1, 2 ** ceil(log2(seq_len / self.max_seq_len_per_rank)))
    # 按"每卡最大序列长度"算,向上取 2 的幂(匹配可用的 CP 组大小)

def get_total_workload(self, seq_length, cp_size):
    return (seq_length ** 2) / cp_size       # attention O(S²) 负载估计,除以 CP 摊分
```

- `gpus_needed`:长样本分到更多 CP 卡,短样本少分 —— CP 度**随样本长度自适应**。
- `make_buckets_equal`(`:55`)/ `next_hdp_group`(`:104`):用 `seq²/cp` 作工作量估计,把样本贪心打包成若干桶,使每个 DP×CP 组(代码里叫 hdp,hybrid data parallel 组)拿到的总工作量大致相等(`delta` 控制 5% 松弛)。`strategy="dp"/"pp"` 决定沿 DP 还是 PP 轴铺开。
- `hybrid_context_parallel_forward_backward`(`:477`):配套的前向反向调度,按动态分好的组跑。

### 3.3 要点与适用场景

这是 CP 在**变长数据**下的负载均衡机制,与其他轴的均衡手段并列:

| 并行轴 | 负载不均衡的来源 | 均衡手段 |
|--------|----------------|---------|
| PP | microbatch 填充/排空 | VPP(`pp_schedulers_analysis.md` 调度器③) |
| CP | 因果掩码 | zigzag 切分(`cp_analysis.md` §2.2) |
| **CP(变长)** | **样本长度差异** | **本文:动态 CP 度 + 工作量均衡分桶** |
| EP | 路由不均 | aux_loss / 容量因子(`ep_analysis.md` §4) |

适用:变长序列训练 —— SFT、长文档预训练、多分辨率多模态。固定长度的标准预训练用不到。

> 本节的 `BalancedCPScheduler` 是动态 CP 均衡逻辑的**类形态**;它与序列打包在 `data_schedule.py` 里被统一成一条流水线(`DefaultDynamicCPScheduler` 是打包调度器的子类),完整分析见 `packed_dataset_dynamic_cp_analysis.md`。

---

## 4. 多模块/多模态流水线(`bridge_communicator.py` + `multimodule_communicator.py`)

### 4.1 动机与解决的问题

标准 `P2PCommunicator`(§1)假设:上下游 PP stage 在**同一个并行网格**里、TP/DP/CP 都一样,激活张量形状对得上。

但**多模态模型**不是这样:视觉编码器、LLM 主干、生成头是**不同的子模型**,各自可能用**完全不同的并行配置**(编码器 TP=2/PP=1,LLM TP=8/PP=4……),甚至输出张量维数都不同(视觉编码器常出 2D `[b*s,h]`,LLM 要 3D `[s,b,h]`)。标准 P2P 接不上。`pp_schedulers_analysis.md` 里 `schedules.py` 的 `is_multimodule` 分支、`backward_step_multimodule` 就是为它们留的钩子。

### 4.2 `BridgeCommunicator` —— 连接两个并行网格

`bridge_communicator.py:39`。连接一对 `HyperCommGrid`(源网格 → 目标网格,见 `parallelism_orchestration_analysis.md` §4③):

- 接收 `src_grid` / `dest_grid` 两个网格,二者并行度可不同。
- `build_comm_map`(`:268`):算出"源网格哪些 rank 该把张量发给目标网格哪些 rank"。当两侧 batch/并行度不同,需要 **fan-in / fan-out**(按 batch 维 split 或 broadcast)—— 用缓存的 broadcast 进程组(`_broadcast_pg_cache`)实现。
- `dim_mapping` 标明张量的 `s`(序列)/`b`(批)/`h`(隐藏)维位置;`tensor_ndim` 处理 2D/3D 差异。
- 提供与 `P2PCommunicator` 同名的接口:`send_forward` / `recv_forward` / `send_forward_recv_backward` / `send_backward_recv_forward`,所以上层调度无需改。
- 当前限制:CP 暂不支持(`:95`,断言两侧 CP=1)。

> [!update] 2026-06-16 · dev@232c478d4 — 跨网格 P2P 改走专用进程组(#5234)
>
> 上文「与 `P2PCommunicator` 同名的接口」在 `ee3f1ff` 时,跨网格的 `dist.send/recv`、`P2POp(isend/irecv)` 都**不带 `group=` 参数**,即隐式走默认的全局 `WORLD` 组。这会让两个子模型之间的桥接 P2P 与其它集合通信共用同一个 NCCL 通信器,存在串行化与标签冲突风险。
>
> #5234 为每条桥接边新建一个**专用进程组** `bridge_pg`(`bridge_communicator.py:167-168`):取「源网格 TP leader ∪ 目标网格 TP leader」并排序,用 `dist.new_group(ranks, backend='nccl')` 建组,并以类级缓存 `_bridge_pg_cache`(键 = 排序后的 rank 列表)避免重复创建相同通信器(`:204-210`);`destroy_bridge_pgs`(`:63-68`)统一销毁。此后所有桥接 `send_forward` / `recv_forward` / `send_backward` / `recv_backward` 及其融合 `P2POp` 都显式传 `group=self.bridge_pg`(`:381`、`:424`、`:514`、`:553`、`:668` 等共 12 处 send/recv/P2POp 站点)。
>
> 意义:把跨网格 leader↔leader 的点对点通信从全局组**隔离**到独立通信器,避免与各子网格内部的 TP/PP/DP 集合通信争用同一 NCCL 资源。`_broadcast_pg_cache`(fan-in/fan-out 广播组)与新增的 `_bridge_pg_cache`(leader 间 P2P 组)现在是 `BridgeCommunicator` 的两类缓存进程组。
>
> 行号漂移:上文 §4.2 提到的 `build_comm_map` 因本 PR 在其上方插入了 `_get_or_create_bridge_pg` 等代码,已从 `:268` 下移到 `bridge_communicator.py:290`。
>
> 配套:`BridgeCommunicator` 连接的 `HyperCommGrid` 本身也在 #5148 获得了 named views(同一段 rank 挂多套命名分解),为异构子模型提供几何基础 —— 详见 [[parallelism_orchestration_analysis]] §4③ 的 2026-06-16 更新。

### 4.3 `MultiModulePipelineCommunicator` —— 编排模块 DAG

`multimodule_communicator.py:110`。把多个子模块组织成一张有向图:

```python
module_to_grid_map = {'image_encoder': enc_grid, 'audio_encoder': aud_grid,
                      'llm': llm_grid, 'generator': gen_grid}
topology = {'image_encoder': ['llm'], 'audio_encoder': ['llm'],   # 模块 DAG
            'llm': ['generator'], 'generator': []}
```

`_build_bridge_comms`(`:168`)为 DAG 的每条边建一个 `BridgeCommunicator`;对外同样暴露 `recv_forward` / `send_forward` / `send_forward_recv_backward` 等(张量以 `Dict[str, Tensor]` 形式按模块名组织)。于是 `schedules.py` 的非交错 1F1B 调度可以无感地驱动一条**跨异构子模型**的流水线。

### 4.4 流程示意

```
image_encoder grid (TP2,PP1) ─┐
                              ├─BridgeCommunicator(fan-in,broadcast PG)─► llm grid (TP8,PP4) ─► generator grid
audio_encoder grid (TP4,PP1) ─┘                                              │
       多个子模型,各自并行配置不同 ── MultiModulePipelineCommunicator 按 topology 编排 ──┘
```

### 4.5 适用场景

VLM、音频-语言、encoder-decoder 等多模态/异构模型 —— 各子模型独立选并行度,再用 bridge 串成一条流水线。纯单模型(GPT)用不到,走标准 `P2PCommunicator`。

---

## 5. 小结

| 主题 | 何时关心 |
|------|---------|
| **P2P 内部** | 想开 `overlap_p2p_comm`(调度器④)→ 必须 `batch_p2p_comm=False` 走 `_p2p_ops`;变长序列 → 多一轮形状交换 |
| **激活换出** | 显存墙严重、不想全量重计算 → `--fine-grained-activation-offloading`,用 PCIe 带宽换显存,可与重计算叠加 |
| **混合 CP 调度** | 变长序列训练(SFT/长文档/多分辨率)→ 按样本长度动态分 CP、均衡 DP×CP 负载 |
| **多模块 PP** | 多模态/异构模型,子模型并行度各异 → `BridgeCommunicator` 连网格,`MultiModulePipelineCommunicator` 编排 DAG |

这 4 块都是 `pp_schedulers_analysis.md` 五调度器之外的"周边设施":通信底座、省显存手段、变长数据支持、多模型支持。

---

*生成依据:`Megatron-LM` `dev` 分支 `ee3f1ff`。源码行号以该 commit 为准。本文是"第一层补遗"3 份文档之②(PP 补遗),已完成:① 并行编排 capstone;后续:③ TP·FSDP·resharding 补遗。*

## Related Pages

- [[pp_schedulers_analysis]] · [[parallelism_orchestration_analysis]] · [[packed_dataset_dynamic_cp_analysis]] · [[recompute_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
