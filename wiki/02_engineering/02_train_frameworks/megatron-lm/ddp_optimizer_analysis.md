# Megatron-LM 数据并行与分布式优化器(DDP & Distributed Optimizer / ZeRO)深度解析

> 代码基准:`Megatron-LM/` 子仓库 `dev` 分支,commit `ee3f1ff`
> 核心文件:`megatron/core/distributed/distributed_data_parallel.py`(635 行)、`param_and_grad_buffer.py`(1589 行)、
> `distributed_data_parallel_config.py`、`megatron/core/optimizer/distrib_optimizer.py`(3051 行)
> 配套阅读:`pp_schedulers_analysis.md`、`ep_analysis.md`、`tp_analysis.md`、`cp_analysis.md`、`optimizer_internals_analysis.md`
>
> **更正记录(2026-05-22)**:原稿把模型态字节数写作 `16Ψ`(套用 ZeRO 论文假设 fp16 梯度)。经源码二次核查更正为 **`18Ψ`** —— `arguments.py:1296-1310` 中 bf16 训练强制 `accumulate_allreduce_grads_in_fp32`,梯度 buffer 为 fp32(4 字节)。详见 §0.4、§2.5。

---

## 0. 总览

### 0.1 DP / DDP / 分布式优化器是什么

**数据并行(Data Parallelism,DP)**:把 global batch 切成 `dp` 份,`dp` 张卡各持**一份完整模型副本**,各算各的 microbatch,反向后把梯度**跨卡求平均**(all-reduce),保证 `dp` 个副本始终一致。DP 不省模型显存,只分摊 batch、提高吞吐 —— 它是其他所有并行轴(TP/PP/CP/EP)之上**最外层**的并行。

**DDP(`DistributedDataParallel`)**:Megatron 的 DP 实现。它做两件超出"朴素 all-reduce"的事:① 把梯度塞进**连续扁平缓冲区(flat buffer)**并分**桶(bucket)**,让梯度通信能**异步、与反向计算重叠**;② 支持梯度以高于参数的精度(fp32)累加。

**分布式优化器(Distributed Optimizer)**:朴素 DP 的痛点是**每卡都存一份完整优化器状态**(fp32 master 权重 + Adam 动量 + 方差,共 12 字节/参数),纯冗余。分布式优化器把这些状态**沿 DP 组切成 `1/dp`**,每卡只更新自己那 `1/dp` 的参数 —— 这就是 **ZeRO**(Zero Redundancy Optimizer)的思想。Megatron 的 `--use-distributed-optimizer` ≈ **ZeRO-1**;配合 Megatron-FSDP 还能做到 ZeRO-2/3。

### 0.2 DP 在并行体系中的位置

| 并行轴 | 切什么 | 峰值激活 | 权重 | 梯度 | 优化器状态 | 通信特征 |
|--------|--------|---------|------|------|-----------|---------|
| TP | 单层权重矩阵 | 1/tp | 1/tp | 1/tp | 1/tp | 高频、关键路径 |
| EP | 专家 | ~1 | MoE 层 1/ep | 1/ep | 1/ep | 中 |
| PP | 层 | 1(VPP>1) | 1/pp | 1/pp | 1/pp | 中,点对点 |
| CP | 序列 | 1/cp | 1 | 1 | 1/cp | 中,仅 attention |
| **DP(DDP)** | **批次** | **1** | **1** | **1** | **1** | **低,每步一次,可重叠** |
| **DP(ZeRO-1)** | 批次 | 1 | 1 | 1 | **1/dp** | 低,= DDP |
| **DP(ZeRO-3)** | 批次 | 1 | **1/dp** | **1/dp** | **1/dp** | 低,但 1.5× DDP |

关键:**朴素 DP 完全不省模型显存**;分布式优化器(ZeRO)在 DP 之上把模型态(参数/梯度/优化器)逐级切分,是"用 DP 通信换显存"的关键基建。

### 0.3 四个 ZeRO 阶段一览

`data_parallel_sharding_strategy`(`distributed_data_parallel_config.py:93`)取 4 值,对应 ZeRO 4 个阶段:

| # | 策略 | ZeRO 阶段 | 切分对象 | 触发 |
|---|------|-----------|---------|------|
| ① | `no_shard` | ZeRO-0 | 都不切(朴素 DDP) | 默认 |
| ② | `optim` | ZeRO-1 | 优化器状态 | `--use-distributed-optimizer` |
| ③ | `optim_grads` | ZeRO-2 | 优化器状态 + 梯度 | Megatron-FSDP |
| ④ | `optim_grads_params` | ZeRO-3 | 优化器状态 + 梯度 + 参数 | Megatron-FSDP / FSDP2 |

本文把这 4 个阶段当作 PP 文档"5 调度器"的对应物逐个解读。

### 0.4 记号约定

| 符号 | 含义 |
|------|------|
| `dp` | DP 度(DP 组大小,= world_size / (tp·pp·cp·ep)) |
| `Ψ` | 模型参数量(个数) |
| 混合精度 Adam(标准 bf16 训练) | bf16 权重 `2Ψ` + **fp32 梯度 `4Ψ`** + fp32 master `4Ψ` + Adam m `4Ψ` + Adam v `4Ψ` = **`18Ψ` 字节** |
| "优化器状态" | fp32 master + m + v = `12Ψ` 字节(ZeRO 术语) |
| bucket | 梯度桶,DDP 通信的最小单位 |

> **为什么梯度是 fp32(4 字节)而非 bf16(2 字节)**:`arguments.py:1296-1310`,`if args.bf16:` 分支注释明写 *"bfloat16 requires gradient accumulation and all-reduce to be done in fp32"* —— bf16 尾数仅 7 位,跨 microbatch 累加会灾难性丢精度。除非显式 `--grad-reduce-in-bf16`,Megatron 对 bf16 训练强制 `accumulate_allreduce_grads_in_fp32 = True`,梯度 buffer dtype 为 `torch.float`(`param_and_grad_buffer.py:812`)。仅当 `--grad-reduce-in-bf16` 时梯度才是 `2Ψ`、合计 `16Ψ`(即 ZeRO 论文的教科书值)。本文按 Megatron **默认**口径取 `18Ψ`。

---

## 1. DP 的目的与动机

### 1.1 朴素数据并行与它的浪费

最朴素的 DP:`dp` 卡各持完整模型,反向后对梯度做一次 all-reduce 求平均,各卡用相同梯度跑相同的 optimizer step → `dp` 个副本永远一致。问题有二:

1. **模型态完全冗余**:每卡都存一份 `18Ψ` 的模型态(参数+梯度+优化器)。`dp=64` 就是 64 份完全相同的优化器状态 —— 纯浪费。
2. **梯度通信暴露**:反向算完后,一次性 all-reduce 整个梯度 buffer(`4Ψ` 字节),这段通信卡在关键路径上。

### 1.2 Megatron DDP 的两项改进(动机)

- **针对暴露**:把梯度分**桶**,反向每算完一个桶的梯度就**异步**发起该桶的 all-reduce,与后续层的反向计算**重叠** → 通信延迟被算力掩盖。
- **针对冗余**:分布式优化器把 `12Ψ` 优化器状态沿 DP 组切成 `1/dp`(ZeRO-1),再往上还能切梯度(ZeRO-2)、切参数(ZeRO-3)。

### 1.3 DP 的收益与定位

- **收益**:线性提升吞吐(更多 batch);配合 ZeRO 后,模型态显存按阶段逐级 `÷dp`。
- **定位**:DP 是**最外层**并行。通信量最低、每步一次、可重叠 —— 所以总是优先把并行度给 DP(README Guideline 1:"Minimize model parallelism, maximize data parallelism")。

---

## 2. DDP 核心机制

`DistributedDataParallel`,`distributed_data_parallel.py:23`。

> [!update] 2026-06-16 · dev@232c478d4 — 行号基线刷新(机制未变)
> 自 `ee3f1ff` 起 `distributed_data_parallel.py` 因 layer-wise 整合等改动增长,本节锚点上移:`_make_backward_post_hook` `:431 → :449`、`no_sync` `:461 → :480`、`start_param_sync` `:474 → :510`、`start_grad_sync` `:532`、`finish_grad_sync` `:544`(未变);新增 `_start_bucket_group_param_sync`(`:492`,见 §阶段②)。其它文件:`distrib_optimizer.py` 的 `DistributedOptimizer` `:103 → :108`;`distributed_data_parallel_config.py` 的 `data_parallel_sharding_strategy` `:93 → :100`、`outer_dp_sharding_strategy` `:153 → :162`;bf16 强制 fp32 累加的 `arguments.py` 分支 `:1296-1310 → :1317-1328`(`param_and_grad_buffer.py` 梯度 dtype `grad_dtype = torch.float ...` 现 `:861`)。

### 2.1 连续扁平缓冲区 + 分桶

`_ParamAndGradBuffer`(`param_and_grad_buffer.py`)把一组参数的梯度打包进**一块连续显存**,每个参数的 `.main_grad` 是这块大 buffer 的一个视图。好处:① 梯度通信可以整桶发,不必逐参数发(减少 kernel/通信启动开销);② 便于与分布式优化器的分片对齐。

buffer 再切成若干 **bucket**。bucket 是 DDP 通信的最小单位 —— 一个 bucket 内所有参数的梯度都就绪后,这个 bucket 立刻发起 all-reduce / reduce-scatter。

### 2.2 overlap_grad_reduce —— 反向 post-hook 驱动的重叠

每个参数注册一个 backward post-hook(`_make_backward_post_hook`,`distributed_data_parallel.py:431`):

```python
def hook(*unused):
    if param in self.param_to_bucket_group:
        if param.grad is not None and not param.grad_added_to_main_grad:
            param.main_grad.add_(param.grad.data)        # 梯度累加进扁平 buffer
        param.grad = None
        if self.ddp_config.overlap_grad_reduce:
            self.param_to_bucket_group[param].register_grad_ready(param, self.force_all_reduce)
            # ↑ 该桶所有参数就绪后,立即异步发起 all-reduce / reduce-scatter
```

效果:**反向算到哪,梯度通信就发到哪**。深层的梯度先算完先发,与浅层的反向计算并行。

```
反向计算流:  [layer L 反向]─[layer L-1 反向]─[layer L-2 反向]─ … ─[layer 0 反向]
通信流:              └ bucket3 异步 RS ┘  └ bucket2 异步 RS ┘  …   └ bucket0 RS ┘
                      ↑ 桶满即发,与后续反向计算重叠;只有最后一个桶的通信尾巴露在关键路径
```

`finish_grad_sync`(`:544`)在反向结束后 `wait` 所有桶的通信句柄。

### 2.3 overlap_param_gather —— 前向的参数 all-gather 重叠

用分布式优化器时,优化器只更新了 `1/dp` 参数,需要 all-gather 才能拿到全量参数做前向。`start_param_sync`(`:474`)按桶异步发起参数 all-gather,与前向计算重叠(`--overlap-param-gather`)。

### 2.4 no_sync —— 梯度累加

`no_sync()` 上下文(`:461`)把 `is_last_microbatch` 置 False,让前 `m-1` 个 microbatch(`m` = microbatch 数)**只把梯度累加进 buffer、不触发通信**,最后一个 microbatch 才同步。这与 PP 文档里 `forward_backward_no_pipelining` 的 `no_sync` 是同一机制。

### 2.5 梯度精度:bf16 训练默认 fp32 累加

`grad_reduce_in_fp32`(DDP config)/ `--accumulate-allreduce-grads-in-fp32`(训练参数):梯度用 fp32 缓冲区累加与规约。

**关键:对 bf16 训练这是默认行为,不是可选项**。`arguments.py:1296-1310` 的 `if args.bf16:` 分支:除非用户显式 `--grad-reduce-in-bf16`,否则 `accumulate_allreduce_grads_in_fp32` 被置 True,并打印 *"accumulate and all-reduce gradients in fp32 for bfloat16 data type"*。`param_and_grad_buffer.py:812` 据此把梯度 buffer dtype 设为 `torch.float`。

- 默认(bf16 训练):梯度 buffer = fp32 → **`4Ψ` 字节**,模型态合计 `18Ψ`。
- `--grad-reduce-in-bf16`:梯度 buffer = bf16 → `2Ψ` 字节,模型态合计 `16Ψ`(省 2Ψ,但大 DP 下累加精度下降)。

### 2.6 finalize_model_grads

`finalize_model_grads.py` 在反向后做跨并行轴的梯度收尾:DP 梯度规约、PP 首尾 stage 共享 embedding 的梯度 all-reduce、SP 下 LayerNorm 梯度的 all-reduce。是 DDP 与 TP/PP 协同的拼接点。

---

## 阶段① — `no_shard`(ZeRO-0,朴素 DDP)

### ①.1 机制

不切任何模型态。每卡完整持有 `18Ψ`。反向后梯度做 **all-reduce** 求平均,每卡各自跑完整 optimizer step。

```
各 rank:全量梯度 ──all-reduce(求平均)──→ 各 rank 得全量平均梯度
        ──→ 各 rank 用全量优化器状态(每卡一份 12Ψ,完全冗余)各自更新全量参数
```

### ①.2 开销

| 维度 | no_shard |
|------|----------|
| 模型态显存/卡 | **`18Ψ` 字节**(完全不省) |
| 梯度通信/步 | all-reduce 整个梯度 buffer(基准 1×) |
| 通信暴露 | 低 —— `overlap_grad_reduce` 把桶通信藏进反向 |

### ①.3 适用场景

- 模型态单卡放得下、且不缺显存时:最简单,无 all-gather 开销。
- 小模型、调试。
- 一旦显存吃紧 → 上 ZeRO-1。

> [!update] 2026-06-16 · dev@232c478d4 — Megatron-FSDP `no_shard` 收敛性修复(#3835/#3754,`megatron_fsdp.py:1234`、`optimizer/__init__.py:1060`)
> `no_shard`(ZeRO-0)下,梯度经 all-reduce 后在各 DP rank 上是**复制**的。原实现仍把梯度统计/范数在 dist-opt(DP)组上规约一遍 → grad norm **虚高**、梯度裁剪过度 → **不收敛**。修复:`no_shard` 时把 grad-stats 规约组从 DP 组改为只用 `model_parallel_group`(TP/PP)(`effective_intra_dist_opt_group = mp_group if no_shard else intra_dist_opt_group`);同时 `start_param_sync` 对 `no_shard` 直接 return(参数已复制,无需 all-gather),并禁止 `no_shard` 配 meta-device 初始化。此修复仅针对 Megatron-FSDP 的 `no_shard` 路径;与 [[optimizer_internals_analysis]] §5 梯度范数"需跨 TP×PP all-reduce"的论述一致 —— 关键是 **DP 维度此时不能再 reduce**。

---

## 阶段② — `optim`(ZeRO-1,DistributedOptimizer)

`DistributedOptimizer`,`distrib_optimizer.py:103`。类注释开门见山:"Optimizer that shards state across data-parallel ranks ... by distributing optimizer states (like momentum and variance buffers) across GPUs in the data-parallel group."

### ②.1 动机与解决的问题

朴素 DDP 里 `dp` 卡存 `dp` 份**完全相同**的 `12Ψ` 优化器状态 —— 这是最大块的纯冗余。ZeRO-1 的洞察:**优化器状态只在 optimizer step 那一刻用到**,可以让每卡只持有、只更新 `1/dp` 的参数对应的优化器状态。

### ②.2 机制:all-reduce → reduce-scatter + all-gather

```
ZeRO-1 一步:
  ① 反向:各 rank 得全量梯度
  ② reduce-scatter:每 rank 只收到自己负责的 1/dp 梯度分片(求和已在 RS 中完成)
  ③ optimizer step:每 rank 只更新自己那 1/dp 参数(只需存 1/dp 的 fp32 master + m + v)
  ④ all-gather:各 rank 把更新后的参数分片 all-gather,拿回全量参数供下一步前向
```

关键恒等式:**`all-reduce = reduce-scatter + all-gather`**。所以 ZeRO-1 把 DDP 的一次 all-reduce 拆成"反向后 RS + 更新后 AG",**通信总量不变**,却把 `12Ψ` 优化器状态切成了 `1/dp`。这是"几乎免费"的显存节省 —— 也是 README 把 `--use-distributed-optimizer` 列为通用性能项的原因。

### ②.3 开销

| 维度 | optim(ZeRO-1) |
|------|----------------|
| 模型态显存/卡 | 参数 `2Ψ` + 梯度 `4Ψ` + 优化器 **`12Ψ/dp`** = `6Ψ + 12Ψ/dp` |
| 通信/步 | reduce-scatter + all-gather(**与 DDP 相同**) |
| 通信暴露 | 低 —— RS 藏进反向(`overlap_grad_reduce`)、AG 藏进前向(`overlap_param_gather`) |

### ②.4 适用场景

- **几乎所有多卡训练的默认项**:通信量与 DDP 相同,白拿 `12Ψ→12Ψ/dp` 的优化器显存节省。
- README General Tips 直接把 `--use-distributed-optimizer` 列为通用开关。

> [!update] 2026-06-16 · dev@232c478d4 — LayerWise(Muon)现在复用本节的 DDP buffer + DistOpt 分片(#4509/#4771)
> `LayerWiseDistributedOptimizer` 已整合进本文的 DDP grad/param buffer 基建:它预计算 shard-aligned 的参数 layout,让每个矩阵整体落在某个 shard 内,从而复用本节的 reduce-scatter/all-gather 与 `overlap_grad_reduce`/`overlap_param_gather`。配 `--optimizer muon --use-distributed-optimizer` 时,Muon 管的 2D 矩阵权重走 LayerWise(等效 ZeRO-1/2 沿 DP 分片),其余 embedding/bias/LayerNorm 等**非-Muon 参数则路由到一个独立的标准 `DistributedOptimizer`**(本节的字节级 ZeRO),二者由 `ChainedOptimizer` 串起。新增 `_start_bucket_group_param_sync`(`distributed_data_parallel.py:492`)让两个 sibling 优化器各自只同步自己那批 bucket group,不重复 all-gather。详见 [[megatron_distributed_optimizer_analysis]] §A.7 与 [[optimizer_internals_analysis]] §7 的 2026-06-16 更新。

---

## 阶段③ — `optim_grads`(ZeRO-2)

### ③.1 动机

ZeRO-1 之后,每卡仍存全量梯度 `4Ψ`(反向累加需要)。ZeRO-2 进一步:既然 reduce-scatter 后每卡只用 `1/dp` 梯度做 step,那**梯度也只需保留 `1/dp`** —— RS 之后即可丢弃非本分片的梯度。

### ③.2 机制

与 ZeRO-1 相同的 RS+AG 通信结构,但梯度 buffer 在 reduce-scatter 后只保留本 rank 的 `1/dp` 分片。由 Megatron-FSDP 的 `data_parallel_sharding_strategy = 'optim_grads'` 实现。

### ③.3 开销

| 维度 | optim_grads(ZeRO-2) |
|------|---------------------|
| 模型态显存/卡 | 参数 `2Ψ` + 梯度 **`4Ψ/dp`** + 优化器 `12Ψ/dp` = `2Ψ + 16Ψ/dp` |
| 通信/步 | `≈` 与 DDP 相同 |

### ③.4 适用场景

- ZeRO-1 仍不够、但还不想付 ZeRO-3 的额外通信时的中间档。

---

## 阶段④ — `optim_grads_params`(ZeRO-3 / 完全分片 FSDP)

### ④.1 动机

ZeRO-2 之后只剩参数 `2Ψ` 没切。ZeRO-3 把**参数本身**也沿 DP 组切成 `1/dp`:每卡只常驻 `1/dp` 参数,**用到某层时才 all-gather 出该层的完整参数,用完立刻释放**。模型态显存彻底降到 `18Ψ/dp`。

这就是 **FSDP(Fully Sharded Data Parallel)**。Megatron 有两套实现:Megatron-FSDP(`--use-megatron-fsdp`)与 FSDP2(`--use-torch-fsdp2`)。

### ④.2 机制

```
ZeRO-3:参数也只存 1/dp
  前向:逐层 all-gather 出本层完整参数 → 算 → 立刻释放(只留 1/dp)
  反向:再次 all-gather 本层参数 → 算梯度 → reduce-scatter 梯度 → 释放参数
  ⇒ 参数在前向、反向各被 all-gather 一次
```

### ④.3 开销

| 维度 | optim_grads_params(ZeRO-3) |
|------|----------------------------|
| 模型态显存/卡 | **`18Ψ/dp`**(参数/梯度/优化器全切) |
| 通信/步 | 前向 AG + 反向 AG + 梯度 RS = **3 趟** ≈ **1.5× DDP** |
| 通信暴露 | 较高 —— 参数 AG 必须靠 prefetch / 计算重叠掩盖,否则拖慢 |

**关键代价**:ZeRO-3 的通信是 DDP 的 **1.5 倍**(参数要在前向、反向各 gather 一次),且 AG 与逐层计算强耦合,重叠不好就掉吞吐。

### ④.4 适用场景

- 模型态实在塞不下、又不想/不能再加 TP/PP 时的终极手段。
- 与 EP 组合(README v0.15:"Support FSDP with EP for MoE models")。
- 不推荐当 ZeRO-1 + TP/PP 已能装下时使用 —— 多付 50% 通信不划算。

> [!update] 2026-06-16 · dev@232c478d4 — Megatron-FSDP 的 A2A Overlap(ZeRO-3 + MoE 的通信重叠)(#3797,`megatron_fsdp.py`、`mcore_fsdp_adapter.py`、`combined_1f1b.py`)
> ZeRO-3 与 MoE 叠加时,**两层通信**同时存在:FSDP 的逐层参数 all-gather / 梯度 reduce-scatter(DP 轴)与 MoE 的 dispatch/combine **All-to-All**(EP 轴)。#3797 让二者**重叠**:把 FSDP 的 `post_forward_release_module` / `post_backward_release_module` 等 hook 暴露出来交给 1F1B overlap pipeline 手动调度,并新增 `enable_fine_grained_param_gather_backward_hook` 支持 backward 侧细粒度参数 gather;配合 `delayed_wgrad`(expert 权重梯度延迟到 dispatch-backward 后再 reduce-scatter,见 [[megatron_distributed_optimizer_analysis]] §A.2)最大化 EP-A2A 与 DP-梯度同步的重叠。
> 这是对 §④.3"参数 AG 必须靠 prefetch/计算重叠掩盖"的 FSDP-内部强化。通信调度/1F1B 重叠角度详见 [[megatron_comm_overlap_analysis]]。

---

## 3. HSDP —— 混合分片数据并行

`outer_dp_sharding_strategy`(`distributed_data_parallel_config.py:153`)+ `num_distributed_optimizer_instances`。

**动机**:ZeRO 的分片域越大,all-gather/reduce-scatter 的参与方越多、跨节点流量越重。HSDP 把 DP 组拆成**两层**:
- **内层(节点内,NVLink)**:做 ZeRO 分片(`optim` / `optim_grads_params`)。
- **外层(节点间,IB)**:只做**副本**(`no_shard`)—— 即普通 DP all-reduce。

```
HSDP:DP 组 = 外层(跨节点,replicate)× 内层(节点内,shard)

  节点0 内层分片组              节点1 内层分片组
  [r0 r1 r2 r3]  ◄─外层 all-reduce─►  [r4 r5 r6 r7]
   └ZeRO 分片┘                          └ZeRO 分片┘
```

效果:分片只在节点内(高带宽 NVLink)发生,跨节点只走轻量的副本同步 —— 兼顾 ZeRO 的省显存与跨节点通信效率。类比 CP 的 `a2a+p2p` 分层思想。

---

## 4. 开销分析汇总

### 4.1 模型态显存阶梯(标准 bf16 + Adam,`dp` = DP 度,单位:字节×Ψ)

| 阶段 | 参数 | 梯度 | 优化器 | 合计/卡 | `dp=64` 时 |
|------|------|------|--------|---------|-----------|
| ① no_shard(ZeRO-0) | `2Ψ` | `4Ψ` | `12Ψ` | **`18Ψ`** | `18Ψ` |
| ② optim(ZeRO-1) | `2Ψ` | `4Ψ` | `12Ψ/dp` | `6Ψ + 12Ψ/dp` | `≈6.19Ψ` |
| ③ optim_grads(ZeRO-2) | `2Ψ` | `4Ψ/dp` | `12Ψ/dp` | `2Ψ + 16Ψ/dp` | `≈2.25Ψ` |
| ④ optim_grads_params(ZeRO-3) | `2Ψ/dp` | `4Ψ/dp` | `12Ψ/dp` | **`18Ψ/dp`** | `≈0.28Ψ` |

```
模型态/卡:  18Ψ ──ZeRO-1──→ 6Ψ+12Ψ/dp ──ZeRO-2──→ 2Ψ+16Ψ/dp ──ZeRO-3──→ 18Ψ/dp
            └ 优化器冗余最大(12Ψ),先切它最划算(ZeRO-1 几乎免费)      └ 参数也切,但通信 ×1.5
```

> 若用 `--grad-reduce-in-bf16`(梯度 bf16):上表梯度列 `4Ψ→2Ψ`,ZeRO-0 合计 `16Ψ`,ZeRO-1 `4Ψ+12Ψ/dp`,ZeRO-2 `2Ψ+14Ψ/dp`,ZeRO-3 `16Ψ/dp`。

### 4.2 通信量

| 阶段 | 梯度通信 | 参数通信 | 趟数合计 | 相对 DDP |
|------|---------|---------|---------|----------|
| ① no_shard | all-reduce(1 趟) | —— | 1 趟 all-reduce | 1× |
| ② optim | reduce-scatter | all-gather | RS + AG | **1×** |
| ③ optim_grads | reduce-scatter | all-gather | RS + AG | **1×** |
| ④ optim_grads_params | reduce-scatter | 前向 AG + 反向 AG | RS + 2×AG | **1.5×** |

要点:**ZeRO-1/2 与 DDP 通信量完全相同**(`all-reduce = RS + AG`)—— 这是 ZeRO-1 被视为"免费"的根本原因;**ZeRO-3 才多付 50%**(参数前向、反向各 gather 一次)。

### 4.3 DP 的"等效气泡"

DP 无流水线气泡。低效来源是**通信暴露**:
- 梯度 RS/all-reduce → `overlap_grad_reduce` 分桶藏进反向,只剩最后一个桶的尾巴。
- 参数 AG → `overlap_param_gather` 藏进前向 / optimizer step。
- ZeRO-3 的逐层参数 AG → 靠 prefetch(`suggested_communication_unit_size`)与逐层计算重叠,重叠不好就是吞吐损失。

---

## 5. 适用场景及选型

### 5.1 选型决策树

```
多卡训练?
└─ 是 ──► DP 是最外层并行,选 ZeRO 阶段:
          │
          ├─ 模型态单卡放得下,且不缺显存?
          │   └─ 是 ──► ① no_shard(最简单,无 AG 开销)
          │
          ├─ 想省显存但不想多付通信?(绝大多数情况)
          │   └─ 是 ──► ② optim / --use-distributed-optimizer(ZeRO-1)
          │             通信与 DDP 相同,白拿 12Ψ→12Ψ/dp,几乎必开
          │
          ├─ ZeRO-1 还不够,梯度也想切?
          │   └─ 是 ──► ③ optim_grads(ZeRO-2,Megatron-FSDP)
          │
          └─ 模型态实在塞不下,且不想再加 TP/PP?
              └─ 是 ──► ④ optim_grads_params(ZeRO-3 / FSDP,通信 ×1.5)
                        跨节点训练 → 配 HSDP,把分片压在节点内 NVLink

通用建议(README Guidelines):
  - 优先把并行度给 DP(通信最低、可重叠);TP/PP/EP 能小则小
  - --use-distributed-optimizer 几乎必开(ZeRO-1 免费省显存)
  - 配 --overlap-grad-reduce + --overlap-param-gather 把 DP 通信藏进计算
  - 显存够时优先 ZeRO-1 + 适度 TP/PP,而非直接上 ZeRO-3
```

### 5.2 一句话总结

- **DP 的本质**:最外层并行,切 batch 不切模型;朴素 DP 每卡冗余存 `18Ψ` 模型态。
- **DDP 的工程**:扁平梯度 buffer + 分桶 + 反向 post-hook,让梯度通信异步重叠进反向。
- **模型态 `18Ψ`**:标准 bf16 训练 = bf16 权重 `2Ψ` + **fp32 梯度 `4Ψ`** + fp32 master `4Ψ` + Adam m `4Ψ` + v `4Ψ`。梯度是 fp32 因为 bf16 跨 microbatch 累加必丢精度(Megatron 强制)。
- **ZeRO 四阶段**:逐级切分模型态 —— ZeRO-1 切优化器(`18Ψ→6Ψ+12Ψ/dp`,**通信不变,几乎免费**)、ZeRO-2 再切梯度、ZeRO-3 连参数也切(`18Ψ/dp`,**通信 ×1.5**)。
- **关键恒等式**:`all-reduce = reduce-scatter + all-gather` —— ZeRO-1/2 只是把 DDP 的一次 all-reduce 拆两半,所以通信零增量。
- **HSDP**:分片压在节点内 NVLink、跨节点只做副本,兼顾省显存与跨节点效率。

---

*生成依据:`Megatron-LM` `dev` 分支 `ee3f1ff`。源码行号以该 commit 为准。`--use-distributed-optimizer` 对应 `DistributedOptimizer`(ZeRO-1);`optim_grads` / `optim_grads_params` 由 Megatron-FSDP 实现。配套文档:`pp_schedulers_analysis.md`、`ep_analysis.md`、`tp_analysis.md`、`cp_analysis.md`、`optimizer_internals_analysis.md`。*

## Related Pages

- [[tp_analysis]] · [[optimizer_internals_analysis]] · [[tp_fsdp_resharding_supplements_analysis]] · [[parallelism_orchestration_analysis]]
- [[megatron_distributed_optimizer_analysis]] · [[megatron_comm_overlap_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
