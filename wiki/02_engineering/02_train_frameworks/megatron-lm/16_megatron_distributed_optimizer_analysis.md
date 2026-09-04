---
title: "Megatron-LM DDP、ZeRO 与分片实现深度解析"
---

# Megatron-LM DDP、ZeRO 与分片实现深度解析

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **学习前置**：[[03_megatron_parallelism_geometry_quickstart]]；若只想跑通训练，先读 [[02_megatron_training_quickstart]]。
> **回答的问题**：Megatron 怎样沿 DP 轴组织 DDP buffer，并在 ZeRO-0/1/2/3、HSDP、Torch FSDP2 与 Megatron-FSDP 之间分配参数、梯度和 optimizer state 的所有权？
> **不覆盖**：单次 optimizer step、loss scaling、LR/WD、CPU optimizer offload、Muon 与 μP 归 [[26_megatron_optimizer_step_internals_deepdive]]；低精度 recipe 归 [[23_megatron_precision_cudagraph_fusion_analysis]]。
> **叙事顺序**：背景 → 为什么按扁平 buffer 分片 → DDP/ZeRO 实现 → 选型 → 约束 → 趋势。
> **最后复核**：2026-09-03。

---

## 1. 背景：DP 只切 batch,模型态在每张卡上原样复制一份 `18Ψ`

### 1.1 DP / DDP / 分布式优化器是什么

**数据并行(Data Parallelism,DP)**:把 global batch 切成 `dp` 份,`dp` 张卡各持**一份完整模型副本**,各算各的 microbatch,反向后把梯度**跨卡求平均**(all-reduce),保证 `dp` 个副本始终一致。DP 不省模型显存,只分摊 batch、提高吞吐 —— 它是其他所有并行轴(TP/PP/CP/EP)之上**最外层**的并行。

**DDP(`DistributedDataParallel`)**:Megatron 的 DP 实现。它做两件超出"朴素 all-reduce"的事:① 把梯度塞进**连续扁平缓冲区(flat buffer)**并分**桶(bucket)**,让梯度通信能**异步、与反向计算重叠**;② 支持梯度以高于参数的精度(fp32)累加。

**分布式优化器(Distributed Optimizer)**:朴素 DP 的痛点是**每卡都存一份完整优化器状态**(fp32 master 权重 + Adam 动量 + 方差,共 12 字节/参数),纯冗余。分布式优化器把这些状态**沿 DP 组切成 `1/dp`**,每卡只更新自己那 `1/dp` 的参数 —— 这就是 **ZeRO**(Zero Redundancy Optimizer)的思想。Megatron 的 `--use-distributed-optimizer` ≈ **ZeRO-1**;配合 Megatron-FSDP 还能做到 ZeRO-2/3。这与 ZeRO-1(优化器状态分片)等价;配合 Grad Buffer 的 Reduce-Scatter 替代 All-Reduce,额外实现了 ZeRO-2(梯度分片)的效果。

### 1.2 DP 在并行体系中的位置

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

### 1.3 四个 ZeRO 阶段一览

`data_parallel_sharding_strategy`(`megatron/core/distributed/distributed_data_parallel_config.py:112`)取 4 值,对应 ZeRO 4 个阶段:

| # | 策略 | ZeRO 阶段 | 切分对象 | 触发 |
|---|------|-----------|---------|------|
| ① | `no_shard` | ZeRO-0 | 都不切(朴素 DDP) | 默认 |
| ② | `optim` | ZeRO-1 | 优化器状态 | `--use-distributed-optimizer` |
| ③ | `optim_grads` | ZeRO-2 | 优化器状态 + 梯度 | Megatron-FSDP |
| ④ | `optim_grads_params` | ZeRO-3 | 优化器状态 + 梯度 + 参数 | Megatron-FSDP / FSDP2 |

本文把这 4 个阶段当作 PP 文档"5 调度器"的对应物逐个解读(§4-§7,阶段①-④)。

### 1.4 记号约定

| 符号 | 含义 |
|------|------|
| `dp` | DP 度(DP 组大小,= world_size / (tp·pp·cp·ep)) |
| `Ψ` | 模型参数量(个数) |
| 混合精度 Adam(标准 bf16 训练) | bf16 权重 `2Ψ` + **fp32 梯度 `4Ψ`** + fp32 master `4Ψ` + Adam m `4Ψ` + Adam v `4Ψ` = **`18Ψ` 字节** |
| "优化器状态" | fp32 master + m + v = `12Ψ` 字节(ZeRO 术语) |
| bucket | 梯度桶,DDP 通信的最小单位 |
| `P` | 模型参数总量(元素个数,§3.9/§3.10/§8 通信组/通信量描述中沿用原文记号,与 `Ψ` 等价) |

> **为什么梯度是 fp32(4 字节)而非 bf16(2 字节)**:`megatron/training/arguments.py:1319-1333`,`if args.bf16:` 分支注释明写 *"bfloat16 requires gradient accumulation and all-reduce to be done in fp32"* —— bf16 尾数仅 7 位,跨 microbatch 累加会灾难性丢精度。除非显式 `--grad-reduce-in-bf16`,Megatron 对 bf16 训练强制 `accumulate_allreduce_grads_in_fp32 = True`,梯度 buffer dtype 为 `torch.float`(`megatron/core/distributed/param_and_grad_buffer.py:976`)。仅当 `--grad-reduce-in-bf16` 时梯度才是 `2Ψ`、合计 `16Ψ`(即 ZeRO 论文的教科书值)。本文按 Megatron **默认**口径取 `18Ψ`。

### 1.5 朴素数据并行与它的浪费

最朴素的 DP:`dp` 卡各持完整模型,反向后对梯度做一次 all-reduce 求平均,各卡用相同梯度跑相同的 optimizer step → `dp` 个副本永远一致。问题有二:

1. **模型态完全冗余**:每卡都存一份 `18Ψ` 的模型态(参数+梯度+优化器)。`dp=64` 就是 64 份完全相同的优化器状态 —— 纯浪费。
2. **梯度通信暴露**:反向算完后,一次性 all-reduce 整个梯度 buffer(`4Ψ` 字节),这段通信卡在关键路径上。

### 1.6 Megatron DDP 的两项改进(动机)

- **针对暴露**:把梯度分**桶**,反向每算完一个桶的梯度就**异步**发起该桶的 all-reduce,与后续层的反向计算**重叠** → 通信延迟被算力掩盖。
- **针对冗余**:分布式优化器把 `12Ψ` 优化器状态沿 DP 组切成 `1/dp`(ZeRO-1),再往上还能切梯度(ZeRO-2)、切参数(ZeRO-3)。

### 1.7 DP 的收益与定位

- **收益**:线性提升吞吐(更多 batch);配合 ZeRO 后,模型态显存按阶段逐级 `÷dp`。
- **定位**:DP 是**最外层**并行。通信量最低、每步一次、可重叠 —— 所以总是优先把并行度给 DP(README Guideline 1:"Minimize model parallelism, maximize data parallelism")。

---

## 2. 为什么这么设计：分片切在"扁平梯度缓冲区的字节"上,而不是切参数、也不复用 PyTorch 原生 DDP

朴素做法有两条现成的路:① 直接用 PyTorch 自带的 `torch.nn.parallel.DistributedDataParallel` —— 它本来就会分桶、会把 all-reduce 与反向重叠;② 让分布式优化器**按参数**分片 —— 每个参数整体归某一个 DP rank 管,边界干净。Megatron 两条都写过,两条都删掉了。源码陈述了其中三条理由;第四条源码沉默,由本页重建并标为推断。

**① 分片对象是"梯度缓冲区的字节区间",不是参数 —— 这条写在 docstring 里。**
`_build_model_gbuf_param_range_map`(`megatron/core/optimizer/distrib_optimizer.py:134`)的 docstring 说:每块梯度 buffer(「padded to be an even multiple of DP-world-size」)被「conceptually divided into DP-world-size contiguous regions, where each DP rank 'owns' a contiguous region」(`:144-146`),而且「This conceptual partitioning of the grad buffer does **NOT** respect parameter boundaries ... it is easiest to think of each DP rank as operating (i.e., reducing, gathering) purely on views into the grad buffer, for all model-to-main & main-to-model operations」(`:151-156`)。
→ 决定取舍的判据是**让 reduce-scatter / all-gather 退化成"把一整块连续显存等分"**:通信侧不必知道参数边界,也不会因为参数大小参差而出现分片负载不均。代价是一个参数可能被两个 rank 各持一半(§3.8),以及必须为此额外生成四组 range 映射(`:158-162`)。

**被否掉的替代①:按 param group 拼接、优化器自己另开 shard。**
提交 `cb6f96b68`(2022-02-15,commit message 即「wip; switching to grad-buffer-centric design」)之前,`Float16DistributedOptimizer.__init__` 走的正是这条路:先给每个 `param_group` 建一张 `{param: {start, end}}` 偏移表(`cb6f96b68^:megatron/optimizer/optimizer.py:734-752`),按该 group 的拼接长度算 `max_world_shard_size = ceil(model_param_size / dp_world_size)` 并逐 rank 记进 `world_shard_infos`(`:775-781`),再用 `allocate_shard = lambda shard_size, dtype: torch.empty(...)`(`:763-767`)**另开**一块张量装 main param 及其 grad(`:882-883`)。也就是说:分片曾经建在"参数组的虚拟拼接 + 优化器自己的独立分配"上,与 DDP 那块扁平 grad buffer 无关。紧随其后的 `a3f3c3ad7`(commit message「todo; align shards with model's contiguous buffer」)把方向钉死;今天 `docs/user-guide/features/dist_optimizer.md:22` 的一句「This distributed optimizer uses contiguous buffers for parameters and main gradients」就是这次转向的结论。

**② 梯度先落进连续扁平 buffer 再分桶异步发,而不是逐参数各发一次通信。**
`DistributedDataParallel` 的类 docstring 把三件事并列写清楚:「stores grads in contiguous buffers」、「has option of overlapping communication with backprop computation by breaking up full model's gradients into smaller buckets and running all-reduce / reduce-scatter on each bucket asynchronously」、「provides the option to do the gradient accumulation in a type other than the param type (e.g., fp32 for a bf16 model)」(`megatron/core/distributed/distributed_data_parallel.py:58-62`)。桶该开多大也有陈述:「larger DP sizes need larger buckets to ensure collectives do not become latency-bound」(`megatron/core/distributed/distributed_data_parallel_config.py:60-63`),构造期注释补上机理 ——「chunks used in NCCL ring-reduce implementations are large enough to remain bandwidth-bound rather than latency-bound」(`megatron/core/distributed/distributed_data_parallel.py:92-93`)。

**被否掉的替代②:PyTorch 原生 DDP(`--DDP-impl torch`)。**
Megatron 曾经让用户在两套 DDP 之间二选一 —— `group.add_argument('--DDP-impl', default='local', choices=['local', 'torch'], ...)`(`3fb3e95ec^:megatron/arguments.py:1018-1021`);提交 `3fb3e95ec`(2023-08-31,commit message 即「Deprecate torchDDP and get rid of args.DDP_impl」)把 `torch` 这一支连同 `schedules.py` 里为 torchDDP 特设的 `no_sync` 分支一起删除。**判据就写在被删掉的那几行断言里**:「If we do accumulation and all-reduces in fp32, we need to have local DDP.」→ `assert args.DDP_impl == 'local'`(`3fb3e95ec^:megatron/arguments.py:174-176`);「If we use the distributed optimizer, we need to use local DDP.」→ 同样的断言(`:182-184`);更早的版本里还有一行 `if args.DDP_impl == 'torch': args.use_contiguous_buffers_in_local_ddp = False`(`b0df10cf0^:megatron/arguments.py:190-191`)。即:**fp32 梯度累加(§3.5)与分布式优化器(§5)都建在"自己那块连续 buffer"上,而 torchDDP 拿不到它** —— 当这两项成为常规配置后,原生 DDP 那一支已无处可用。

**被否掉的替代③:把"连续 buffer"与"重叠"留成开关。**
`b0df10cf0`(2023-08-18,commit message「Remove old LocalDDP wrapper and replace with new OverlappingLocalDDP」)删掉了旧的 `DistributedDataParallel`(`b0df10cf0^:megatron/model/distributed.py:378`;它的 docstring 只声称「has the potential to reduce memory fragmentation」`:381`,并把连续 buffer 做成构造参数 `use_contiguous_buffers`,`:390-396`)与命令行开关 `--no-contiguous-buffers-in-local-ddp`(`b0df10cf0^:megatron/arguments.py:1031`),把 `OverlappingDistributedDataParallel`(`b0df10cf0^:megatron/model/distributed.py:225`)直接改名顶上。结论:**"扁平 buffer + 分桶"从一项可关闭的优化变成唯一实现** —— §3.1 那块连续显存不再是可选特性,而是 ZeRO 分片、`overlap_grad_reduce`、`overlap_param_gather` 共同的地基。

**③ 优化器状态沿 DP 组切、不再每卡复制一份 —— 出处直接写着 ZeRO。**
`docs/user-guide/features/dist_optimizer.md:12` 一句话给全:「The distributed optimizer saves memory by sharding optimizer state across data parallel ranks **instead of replicating it on every rank**, as described in the [ZeRO paper](https://arxiv.org/abs/1910.02054)」;同一文件的字节表把「`bf16` parameters, `fp32` gradients」这一行写成 `18` → `6 + 12/d`(`:16-19`),与本页 §1.4 / §9.1 的口径一致。

**④ 为什么默认只切到 ZeRO-1,而不是一上来就切参数。**

> [!note] 推断
> 源码陈述的只有三件事:`data_parallel_sharding_strategy` 的四个取值、语义与**默认值 `'no_shard'`**(`megatron/core/distributed/distributed_data_parallel_config.py:112-114`);开不开 `use_distributed_optimizer` 只改变 buffer 的规约算子 —— `reduction_collective = "reduce-scatter" if self.ddp_config.use_distributed_optimizer else "all-reduce"`,并把这句判断直接打成 INFO 日志(`megatron/core/distributed/param_and_grad_buffer.py:278-286`);以及 ZeRO-2/3 由 Megatron-FSDP 承担(§11)。**"因为 `all-reduce = reduce-scatter + all-gather`,所以 ZeRO-1 通信零增量、应当先切优化器状态"这层判断由本页承担,源码没有这样表态**,也没有任何注释比较过四个阶段的通信代价。要引用这条判断,请回到 `megatron/core/distributed/param_and_grad_buffer.py:278-286`、`megatron/core/distributed/distributed_data_parallel_config.py:112-114` 与 `docs/user-guide/features/dist_optimizer.md:12-19` 这三个 locator,不要引用本段推断。

---

## 3. DDP 核心机制

`DistributedDataParallel`,`megatron/core/distributed/distributed_data_parallel.py:56`(新基线上它已继承自 `_BaseDataParallel`)。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — 行号基线刷新(机制未变)
> `megatron/core/distributed/distributed_data_parallel.py` 自 `ee3f1ff` 起因 layer-wise 整合等改动大幅增长,本节锚点在新基线 `71092579` 下为:`_make_backward_post_hook` `:500`(旧 `:431`)、`no_sync` `:532`(旧 `:461`)、`start_param_sync` `:562`(旧 `:474`)、`start_grad_sync` `:597`(旧 `:532`)、`finish_grad_sync` `:609`(旧 `:544`);`_start_bucket_group_param_sync` 在 `:544`(见 §5 阶段②)。其它文件:`megatron/core/optimizer/distrib_optimizer.py` 的 `DistributedOptimizer` `:113`(旧 `:103`);`megatron/core/distributed/distributed_data_parallel_config.py` 的 `data_parallel_sharding_strategy` `:112`(旧 `:93`)、`outer_dp_sharding_strategy` `:182`(旧 `:153`);bf16 强制 fp32 累加的 `megatron/training/arguments.py` 分支 `:1319-1333`(旧 `:1296-1310`),梯度 dtype `grad_dtype = torch.float if grad_reduce_in_fp32 else param.dtype` 现在 `megatron/core/distributed/param_and_grad_buffer.py:976`(旧 `:812`)。

### 3.1 连续扁平缓冲区 + 分桶

`_ParamAndGradBuffer`(`megatron/core/distributed/param_and_grad_buffer.py`)把一组参数的梯度打包进**一块连续显存**,每个参数的 `.main_grad` 是这块大 buffer 的一个视图。好处:① 梯度通信可以整桶发,不必逐参数发(减少 kernel/通信启动开销);② 便于与分布式优化器的分片对齐。

buffer 再切成若干 **bucket**。bucket 是 DDP 通信的最小单位 —— 一个 bucket 内所有参数的梯度都就绪后,这个 bucket 立刻发起 all-reduce / reduce-scatter。

### 3.2 overlap_grad_reduce —— 反向 post-hook 驱动的重叠

每个参数注册一个 backward post-hook(`_make_backward_post_hook`,`megatron/core/distributed/distributed_data_parallel.py:500`):

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

`finish_grad_sync`(`megatron/core/distributed/distributed_data_parallel.py:609`)在反向结束后 `wait` 所有桶的通信句柄。

### 3.3 overlap_param_gather —— 前向的参数 all-gather 重叠

用分布式优化器时,优化器只更新了 `1/dp` 参数,需要 all-gather 才能拿到全量参数做前向。`start_param_sync`(`megatron/core/distributed/distributed_data_parallel.py:562`)按桶异步发起参数 all-gather,与前向计算重叠(`--overlap-param-gather`)。

### 3.4 no_sync —— 梯度累加

`no_sync()` 上下文(`megatron/core/distributed/distributed_data_parallel.py:532`)把 `is_last_microbatch` 置 False,让前 `m-1` 个 microbatch(`m` = microbatch 数)**只把梯度累加进 buffer、不触发通信**,最后一个 microbatch 才同步。这与 PP 文档里 `forward_backward_no_pipelining` 的 `no_sync` 是同一机制。

### 3.5 梯度精度:bf16 训练默认 fp32 累加

`grad_reduce_in_fp32`(DDP config)/ `--accumulate-allreduce-grads-in-fp32`(训练参数):梯度用 fp32 缓冲区累加与规约。

**关键:对 bf16 训练这是默认行为,不是可选项**。`megatron/training/arguments.py:1319-1333` 的 `if args.bf16:` 分支:除非用户显式 `--grad-reduce-in-bf16`,否则 `accumulate_allreduce_grads_in_fp32` 被置 True,并打印 *"accumulate and all-reduce gradients in fp32 for bfloat16 data type"*。`megatron/core/distributed/param_and_grad_buffer.py:976` 据此把梯度 buffer dtype 设为 `torch.float`。

- 默认(bf16 训练):梯度 buffer = fp32 → **`4Ψ` 字节**,模型态合计 `18Ψ`。
- `--grad-reduce-in-bf16`:梯度 buffer = bf16 → `2Ψ` 字节,模型态合计 `16Ψ`(省 2Ψ,但大 DP 下累加精度下降)。

### 3.6 finalize_model_grads

`megatron/core/distributed/finalize_model_grads.py` 在反向后做跨并行轴的梯度收尾:DP 梯度规约、PP 首尾 stage 共享 embedding 的梯度 all-reduce、SP 下 LayerNorm 梯度的 all-reduce。是 DDP 与 TP/PP 协同的拼接点。

### 3.7 bucketing 算法与 overlap 调度(机制细节)

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — §3.1/3.2/3.3 的机制级补全:分桶**怎么切**、双向 overlap **怎么被 hook 驱动**、bucket_size **怎么调**

§3.1–§3.3 给了"桶是通信单位、post-hook 发 RS、`start_param_sync` 做 AG"的轮廓。本节补到源码层。文件简称:PGB = `megatron/core/distributed/param_and_grad_buffer.py`、DDP = `megatron/core/distributed/distributed_data_parallel.py`、cfg = `megatron/core/distributed/distributed_data_parallel_config.py`。

#### 三级结构

| 层级 | 类 / 位置 | 角色 |
|---|---|---|
| Buffer | `_ParamAndGradBuffer`(PGB`:1066`) | 每 (dtype, DP 组) 一块连续扁平缓冲(param 一块、grad 一块) |
| Bucket | `_ParamAndGradBucket`(PGB`:103`) | buffer 的连续切片,通信寻址单位 |
| BucketGroup | `_ParamAndGradBucketGroup`(PGB`:220`;`partition_buckets` 定义于 PGB`:1803`、在 DDP`:310` 被调用切分) | **一次 NCCL collective 的粒度**,组内多桶由 `_coalescing_manager` 合并成一次调用 |

#### 分桶算法:逆序贪心(`_compute_default_per_buffer_param_layout`,PGB`:1015-1063`)

参数按 **`params[::-1]` 逆序**(PGB`:1041`,即 backprop 顺序)遍历,贪心累加 numel,当 `当前桶累计 ≥ bucket_size`(PGB`:1047`)即封桶、`bucket_id += 1`(PGB`:1048-1052`);不足一桶的尾巴归最后一桶(PGB`:1055-1057`)。每参数落 `(start, end, bucket_id)` 入 `param_index_map`(PGB`:1044`);`bucket_size=None` 则单桶。

**逆序是 overlap 的前置设计**:反向时末层梯度最先就绪,逆序使末层落 bucket 0 → bucket 0 最先填满最先发 RS。

#### bucket_size 默认与约束

- 默认 `max(40_000_000, 1_000_000 × dp_size)`(DDP`:100-103`);`overlap_grad_reduce=False` 时置 `None`(单桶,DDP`:104-106`)。
- 理由(cfg`:60-63`):*"larger DP sizes need larger buckets to ensure collectives do not become latency-bound"* —— ring 算法每 rank 实际报文 = `bucket_size / dp_size`(cfg`:73`),DP 越大报文越小,桶须放大才吃满带宽。可改用 `num_buckets`(cfg`:65-68`,与 bucket_size 二选一)。
- distopt 约束:整块 buffer 须可分片,`assert self.numel % self.data_parallel_world_size == 0`(PGB`:1220-1221`);非 distopt 则不允许 padding,`assert self.numel == self.numel_unpadded`(PGB`:1224-1225`)。`pad_buckets_for_high_nccl_busbw` 把桶凑到 `2^16` 倍数拉高 NCCL busbw(cfg`:70-74`)。

> [!note] 补充(2026-07-31 · 由原 `16_megatron_distributed_optimizer_analysis.md` §2.3.5 并入)该更正自 `dev@232c478d4`（2026-06-16）起适用,行号已重核至基线 `71092579` — bucket 对齐/尺寸的两处更正
> **① 65536 对齐是条件性的,不是恒定的**:bucket 末端对齐 divisor 现集中在 `megatron/core/optimizer/param_layout.py:29` `bucket_end_divisor()`,只有 `pad_buckets_for_high_nccl_busbw=True` 时才是 `lcm(dp, 128, 2**16)`;否则只对齐到 `lcm(dp, 128)`。即上文的 `65536` 项是"高 NCCL busbw"开关下的产物,默认未必启用。
> **② 默认 `bucket_size` 公式改由 pg_collection 计算**(#5006,`megatron/training/models/dist_utils.py:329`):`max(40000000, 1000000 * pg_collection.dp_cp.size())` —— 数值口径与原 `1000000 * dp_size` 一致,但来源从全局 `mpu.get_data_parallel_world_size()` 换成显式 `pg_collection.dp_cp.size()`(同时 `pp_rank`、`expert_data_parallel_world_size` 等也都改走 pg_collection)。`pg_collection` 现在对 Megatron-FSDP 与 DistributedOptimizer **两条路径都会传入**(原仅 FSDP 传)。

#### 反向 overlap:grad RS 的就绪触发(承接 §3.2)

backward post-hook(`_make_backward_post_hook`,DDP`:500-529`)里其实是**两步,别混为一谈**:① **填数据** `param.main_grad.add_(param.grad.data)`(DDP`:518-521`)—— `main_grad` 是扁平 grad buffer 的**视图**,梯度**原地**累加进该参数在桶里的固定区段,没有"搬进桶"这个动作(用融合 wgrad 时 `grad_added_to_main_grad=True`,连这次 `add_` 都省);② **记账** `register_grad_ready(param)`(DDP`:524-527` → PGB`:913`)**不碰数据**,只把计数器 +1(`per_param_grad_ready_counts[param] += 1`,PGB`:930`)。

所以**"桶满"不是攒够字节**(桶的大小与成员在初始化时已由逆序贪心定死),而是**该 bucket group 所有成员的梯度都算齐**:当 `per_param_grad_ready_counts == golden_per_param_grad_ready_counts`(PGB`:933`)才 `start_grad_sync()` 发一次 coalesced 异步 RS(PGB`:651` → `dist_reduce_scatter_func` 实际调用在 PGB`:769`)。两个细节:

- **golden count 不一定是 1**:参数若在前向被用多次(tied embedding / 多算子消费),其梯度会多次就绪;第一个 batch 记录每个参数"应有的就绪次数"为 golden(PGB`:300-307` 注释与初始化、PGB`:336-340` 在 `reset()` 里落 golden),之后必须集齐该次数才算 ready。
- 仅 `is_last_microbatch` 计数(PGB`:926`),故梯度累积下前 m−1 个 microbatch 不发通信(与 §3.4 `no_sync` 一致)。逆序分桶使末层最先集齐 → bucket 0 最先发 RS,与前面层反向计算重叠。

#### 前向 overlap:param AG 的预取流水(承接 §3.3)

需求驱动 + 预取下一桶,是"计算 overlap 掉参数通信"的核心:

1. `start_param_sync`(PGB`:448`)对一个 bucket group 发异步 all-gather:`_coalescing_manager` 合并组内各桶的 `all_gather_into_tensor`,每 rank 贡献自己的 shard(PGB`:583-599`),句柄存 `param_gather_handle`(PGB`:601`)。
2. 每个 module 注册 **forward pre-hook**(`_make_forward_pre_hook`,DDP`:468`;挂载 `:443-446`)。module 前向前,对它用到的每个参数调其 bucket group 的 `finish_param_sync` —— 新基线上这一步经由 `_finish_param_sync_for_bucket_group`(DDP`:489` → DDP`:493-498`)转发。
3. `finish_param_sync`(PGB`:611`):先 **wait** 本组 AG 完成(PGB`:633-635`,保证这层参数齐),再**立刻派发下一组** `next_param_gather_bucket_group.start_param_sync()`(PGB`:646`)—— 这就是预取:本 module 用刚 gather 好的参数算时,后台已在 gather 下一组。
4. `next_param_gather_bucket_group` 链按前向序在 DDP`:337-348` 串好(注释说明按桶逆序串,因 all-gather 按桶逆序发生);链首(第一组)AG 由 PP schedule 经 `config.param_sync_func` 先发(`megatron/core/pipeline_parallel/schedules.py:1321-1322`、`:1443-1455`;该回调在 `megatron/training/training.py:4384` 绑成 `model_chunk.start_param_sync`),或被 `finish_param_sync` 懒发(PGB`:630-631`)。
5. 若下一组 AG 已被提前派发 → PGB`:638-644` 警告 *"mismatch between the order of parameter registration and forward pass execution, which will hurt the communication-computation overlap performance"* —— **预取假设 module 前向序 == 参数注册序**。

#### fp32 累加 RS：派发新桶前排空前驱桶

`reduce_scatter_with_fp32_accumulation` 的中间 all-to-all 输出会一直被句柄持有到 `.wait()`。如果所有桶都只在 step 尾统一收尾，这些中间张量会同时存活，放大显存峰值。因此 DDP 只在 `overlap_grad_reduce=True` + fp32-accumulation RS + 单 distributed-optimizer instance 时，把 `bucket_groups[i-1]` 记为 `bucket_groups[i]` 的 `previous_grad_reduce_bucket_group`（`megatron/core/distributed/distributed_data_parallel.py:350-365`）。

每个 bucket group 进入 `start_grad_sync` 时，先检查前驱是否已派发；若是，在为当前桶分配新的 RS 中间状态之前调 `finish_grad_sync`（`megatron/core/distributed/param_and_grad_buffer.py:651-680`）。这是**有界的在飞缓冲策略**：它不改变桶序、collective 数量或反向触发条件，只避免上一个 fp32-accumulation RS 的中间输出跨越多个后续桶。普通 DP 与 expert-DP 的 bucket-group 列表都建立这条链（DDP `:363-365`）。与其他轴同时在飞时的全局显存/吞吐取舍见 [[20_megatron_comm_overlap_analysis]]。

时间线(理想):

```
forward :  [算 g0]      [算 g1]      [算 g2]   ...
AG 流  : AG_g0(头)│ AG_g1 ──┘ AG_g2 ──┘ AG_g3 ──┘     每组在上一组被消费时后台 gather
           ↑首组暴露(可塞进 optimizer step)         ↑尾组无计算可盖则暴露
backward:  ...[算 g2][算 g1][算 g0]
RS 流  :          └RS_g0┘ └RS_g1┘ … └RS_last┘        逆序桶,首组(末层)最先发
                                       ↑末桶 RS 尾巴,finish_grad_sync 收尾
```

#### 调节点

| 旋钮 | 太小 / 错 | 太大 / 错 | 甜点 |
|---|---|---|---|
| **bucket_size**(主) | 桶多→collective 太小→latency-bound、ring 报文 `bucket_size/dp` 吃不满带宽 | 桶大→粒度粗:反向迟发首桶、前向首桶 gather 阻塞起步;单桶 = 零重叠 | 大到 BW-bound 又有多桶 → 默认 `max(40M,1M·dp)` + `pad_…_busbw` |
| **桶序 = 执行序** | 注册序 ≠ 前向序 → PGB`:638-644` 警告、overlap 退化 | — | 反向逆序入桶、前向顺序预取 |
| **align_param_gather**(cfg`:24`) | 不开时各 PP stage 自行发 AG 可能互抢 | — | 开启由 `megatron/core/pipeline_parallel/schedules.py` 统一调度跨 stage 对齐(pre-hook 路径上的 `skip_next_bucket_dispatch`,DDP`:493-498`) |
| **暴露的头尾** | — | — | 头(首桶 gather)可经 `overlap_param_gather_with_optimizer_step` 塞进 step;尾(末桶 RS)由 `finish_grad_sync` 收 |

**一句话**:bucket 把整块 DP 通信切成 N 段并按执行序排好,hook 让"算第 i 段"自动等齐第 i 段并预取第 i+1 段;bucket_size 定 N —— 太小没东西可重叠、太大每段通信低效,默认 `max(40M, 1M·dp_size)` 在"够大到带宽受限"与"够多到首段早发、仅末段尾巴暴露"间取平衡。通信调度与 1F1B 重叠的全局视角见 [[20_megatron_comm_overlap_analysis]]。

### 3.8 分片边界与参数视图:不按参数对齐(补充,2026-07-31 · 由原 `16_megatron_distributed_optimizer_analysis.md` §2.2 并入)

`megatron/core/optimizer/distrib_optimizer.py:134`(`param_local_*` 的计算在 `:176-181`)核心映射 `_build_model_gbuf_param_range_map`:

```python
# 一个参数可能被多个 DP rank 分片持有
param_local_start = max(0, param_world_start - gbuf_world_range.start)
param_local_end = min(gbuf_world_range.size, param_world_end - gbuf_world_range.start)
```

分片边界可能"切"在参数的中间。一个参数的不同部分由不同 DP rank 维护优化器状态。这意味着每个 rank 上的参数只是一个 view/shard,不是完整的参数——这是 §3.1 连续扁平缓冲区在 DistributedOptimizer(ZeRO-1)层面的具体切法:buffer 按 `dp_world_size` 等大块切分(与桶边界无关),每个 rank "拥有"对应块上的参数子集,负责① reduce-scatter 归约到自己分片、② 仅为自己分片存 Adam state、③ all-gather 把更新后参数广播回全体。

### 3.9 通信组定义(补充,2026-07-31 · 由原 `16_megatron_distributed_optimizer_analysis.md` §2.3.1 并入)

| 通信组 | 获取函数 | 组成员 | 通信操作 |
|--------|---------|--------|---------|
| **DP Group** | `get_data_parallel_group()` | 所有数据并行的 rank | AllReduce(标准DP)/ ReduceScatter + AllGather(DistOpt) |
| **Intra-Instance DP** | DP Group 的子组 | `num_distributed_optimizer_instances` 划分的子组 | ReduceScatter(梯度分片) |
| **Inter-Instance DP** | DP Group / Intra-Instance | 跨 instance 的 rank | AllReduce(梯度去重, HSDP only) |
| **EP DP Group** | `get_expert_data_parallel_group()` | Expert 数据并行的 rank | AllReduce(expert 梯度) |

**通信组定义**(`megatron/core/parallel_state.py:1482, 2000`):
```python
# DP Group: 跨所有 TP/PP/EP 的相同位置 rank 组成
get_data_parallel_group(with_context_parallel=True)

# EP DP Group: Expert 的 DP 组(EP 隔离专家参数后形成)
get_expert_data_parallel_group()
```

Intra-Instance / Inter-Instance 两组是 §8 HSDP 的通信基础;EP DP Group 是 EP 场景下专家参数独立于稠密参数的分片域(见 §11.5)。

### 3.10 FP8/FP4 参数对通信量与实现的影响(补充,2026-07-31 · 由原 `16_megatron_distributed_optimizer_analysis.md` §2.3.4+§3.4 并入)

| 参数精度 | AllGather 通信量 | ReduceScatter 通信量 | 总通信节省 |
|---------|-----------------|---------------------|-----------|
| BF16 (基准) | P × 2 bytes | P × 2 bytes | 0% |
| FP8 E4M3 (`fp8_param_gather=True`) | P × 1 byte | P × 2 bytes (grad 仍为 BF16) | 25% |
| MXFP8 (`fp8_param_gather` + `reuse_grad_buf`) | 共享 buffer → 0 | P × 2 bytes (仅 grad) | ~33% |
| NVFP4 (`fp4_param_gather=True`) | P × 0.5 byte | P × 2 bytes (grad 仍为 BF16) | 37.5% |

> 注:梯度 ReduceScatter 始终在 BF16/FP32 精度下执行,因为梯度累积需要全精度。

**实现细节**(`megatron/core/optimizer/distrib_optimizer.py`/`megatron/core/distributed/param_and_grad_buffer.py`):

- **FP8 参数 All-Gather**(收集器 `_get_fp8_params_and_shard_fp32_from_fp8` 在 `megatron/core/optimizer/distrib_optimizer.py:2687`、NVFP4 版在 `:2739`;量化调用在 `:2886-2909`):参数在 FP8 格式下做 All-Gather,传输量减半;FP32 主权重通过 `quantize_param_shard()` 量化回 FP8。
- **NVFP4 双 Buffer 布局**(`megatron/core/distributed/param_and_grad_buffer.py:1159-1186`;打包布局的计算在 `_compute_nvfp4_packed_layout`,`:1582`):参数 Buffer 每字节 2 个 FP4 值(numel/2);梯度 Buffer 全精度 BF16(numel);需要两套索引映射。
- **MXFP8 共享 Buffer**(`megatron/core/distributed/param_and_grad_buffer.py:1302-1321`):参数 All-Gather 和梯度 Reduce-Scatter 共享同一块显存,开关字段 `reuse_grad_buf_for_mxfp8_param_ag`(`megatron/core/distributed/distributed_data_parallel_config.py:98`)。注:新基线上共享 buffer 的分配条件已抽成局部变量 `shared_param_grad_buffer`(`megatron/core/distributed/param_and_grad_buffer.py:1231-1233`,判据是 `use_distributed_optimizer` 且 buffer 内含 MXFP8 张量),`reuse_grad_buf_for_mxfp8_param_ag` 则在 all-gather 侧决定是否复用(`:374`、`:1407`)。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — 精度/MXFP8 相关行号与门控
> - **行号漂移**:解耦梯度赋值 `shard_main_param.decoupled_grad = shard_model_grad`(见 [[26_megatron_optimizer_step_internals_deepdive]] §4.3)在新基线为 `megatron/core/optimizer/distrib_optimizer.py:2842`(`ee3f1ff` 时 `:2568`、`232c478d4` 时 `:2728`);`shard_main_param.grad = shard_model_grad.float()` 现 `:2844`(旧 `:2570` / `:2730`)。语义未变。
> - **MXFP8 共享 buffer 的 all-gather 后处理被抽函数**(#4771,`megatron/core/distributed/distributed_data_parallel.py:544` `_start_bucket_group_param_sync`):原内联在 `start_param_sync` 里的"把 all-gather 出的 MXFP8 参数从共享 buffer 拷回 `param.data` 并清零 buffer 供梯度累加"逻辑被抽成单 bucket-group 方法,便于 LayerWise+DistOpt 链式各自只同步自己的 bucket。
> - **ChainedOptimizer 的 MXFP8 defer-sync 改为探测 DDP config**(#4982,`megatron/core/optimizer/optimizer.py:1806`):`_should_defer_mxfp8_param_sync` 不再信 `OptimizerConfig.overlap_param_gather`,而是逐个探测子 `DistributedOptimizer.ddp_config.overlap_param_gather`(详见 [[26_megatron_optimizer_step_internals_deepdive]] §2.1 的 2026-06-16 更新)。

### 3.11 精度感知优化器:decoupled_grad(补充,2026-07-31 · 由原 `16_megatron_distributed_optimizer_analysis.md` §3.3 并入)

**标准混合精度**:模型 BF16 → 主权重 FP32 → 梯度从 BF16 转 FP32
```python
# megatron/core/optimizer/distrib_optimizer.py:2844
shard_main_param.grad = shard_model_grad.float()
```

**精度感知优化器** (`use_precision_aware_optimizer: True`):主权重、exp_avg、exp_avg_sq 可采用不同的低精度格式;使用 `.decoupled_grad` 解耦模型参数 dtype 和优化器 state dtype:
```python
# megatron/core/optimizer/distrib_optimizer.py:2842
shard_main_param.decoupled_grad = shard_model_grad
```

这与 [[26_megatron_optimizer_step_internals_deepdive]] §4(混合精度优化器:fp32 master 副本)是同一套"18 bytes/param"体系在精度可配置场景下的变体——`decoupled_grad` 让梯度/master/m/v 各自可选不同精度,而非固定 fp32,详见 [[26_megatron_optimizer_step_internals_deepdive]] §4.3。

---

## 4. 阶段① — `no_shard`(ZeRO-0,朴素 DDP)

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

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — Megatron-FSDP `no_shard` 收敛性修复(#3835/#3754,`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:1289-1290`、`megatron/core/optimizer/__init__.py:1071`)
> `no_shard`(ZeRO-0)下,梯度经 all-reduce 后在各 DP rank 上是**复制**的。原实现仍把梯度统计/范数在 dist-opt(DP)组上规约一遍 → grad norm **虚高**、梯度裁剪过度 → **不收敛**。修复:`no_shard` 时把 grad-stats 规约组从 DP 组改为只用 `model_parallel_group`(TP/PP)(`effective_intra_dist_opt_group = mp_group if no_shard else intra_dist_opt_group`);同时 `start_param_sync` 对 `no_shard` 直接 return(参数已复制,无需 all-gather),并禁止 `no_shard` 配 meta-device 初始化。此修复仅针对 Megatron-FSDP 的 `no_shard` 路径;与 [[26_megatron_optimizer_step_internals_deepdive]] §7 梯度范数"需跨 TP×PP all-reduce"的论述一致 —— 关键是 **DP 维度此时不能再 reduce**。

---

## 5. 阶段② — `optim`(ZeRO-1,DistributedOptimizer)

`DistributedOptimizer`,`megatron/core/optimizer/distrib_optimizer.py:113`。类注释开门见山:"Optimizer that shards state across data-parallel ranks ... by distributing optimizer states (like momentum and variance buffers) across GPUs in the data-parallel group."

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

关键恒等式:**`all-reduce = reduce-scatter + all-gather`**。所以 ZeRO-1 把 DDP 的一次 all-reduce 拆成"反向后 RS + 更新后 AG",**通信总量不变**,却把 `12Ψ` 优化器状态切成了 `1/dp`。这是"几乎免费"的显存节省 —— 也是 README 把 `--use-distributed-optimizer` 列为通用性能项的原因(跨框架视角下"用 AG(param) 替换 AG(grad)"这一表述见 [[32_distributed_optimizer_deepdive]] §一)。

### ②.3 开销

| 维度 | optim(ZeRO-1) |
|------|----------------|
| 模型态显存/卡 | 参数 `2Ψ` + 梯度 `4Ψ` + 优化器 **`12Ψ/dp`** = `6Ψ + 12Ψ/dp` |
| 通信/步 | reduce-scatter + all-gather(**与 DDP 相同**) |
| 通信暴露 | 低 —— RS 藏进反向(`overlap_grad_reduce`)、AG 藏进前向(`overlap_param_gather`) |

### ②.4 适用场景

- **几乎所有多卡训练的默认项**:通信量与 DDP 相同,白拿 `12Ψ→12Ψ/dp` 的优化器显存节省。
- README General Tips 直接把 `--use-distributed-optimizer` 列为通用开关。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — LayerWise(Muon)现在复用本节的 DDP buffer + DistOpt 分片(#4509/#4771)
> `LayerWiseDistributedOptimizer` 已整合进本文的 DDP grad/param buffer 基建:它预计算 shard-aligned 的参数 layout,让每个矩阵整体落在某个 shard 内,从而复用本节的 reduce-scatter/all-gather 与 `overlap_grad_reduce`/`overlap_param_gather`。配 `--optimizer muon --use-distributed-optimizer` 时,Muon 管的 2D 矩阵权重走 LayerWise(等效 ZeRO-1/2 沿 DP 分片),其余 embedding/bias/LayerNorm 等**非-Muon 参数则路由到一个独立的标准 `DistributedOptimizer`**(本节的字节级 ZeRO),二者由 `ChainedOptimizer` 串起。新增 `_start_bucket_group_param_sync`(`megatron/core/distributed/distributed_data_parallel.py:544`)让两个 sibling 优化器各自只同步自己那批 bucket group,不重复 all-gather。详见 [[26_megatron_optimizer_step_internals_deepdive]] §10 的 2026-06-16 更新。

---

## 6. 阶段③ — `optim_grads`(ZeRO-2)

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

## 7. 阶段④ — `optim_grads_params`(ZeRO-3 / 完全分片 FSDP)

### ④.1 动机

ZeRO-2 之后只剩参数 `2Ψ` 没切。ZeRO-3 把**参数本身**也沿 DP 组切成 `1/dp`:每卡只常驻 `1/dp` 参数,**用到某层时才 all-gather 出该层的完整参数,用完立刻释放**。模型态显存彻底降到 `18Ψ/dp`。

这就是 **FSDP(Fully Sharded Data Parallel)**。Megatron 有两套实现:Megatron-FSDP(`--use-megatron-fsdp`)与 FSDP2(`--use-torch-fsdp2`),完整对比见 §11。

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

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — Megatron-FSDP 的 A2A Overlap(ZeRO-3 + MoE 的通信重叠)(#3797,`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py`、`megatron/core/distributed/fsdp/mcore_fsdp_adapter.py`、`megatron/core/pipeline_parallel/combined_1f1b.py`)
> ZeRO-3 与 MoE 叠加时,**两层通信**同时存在:FSDP 的逐层参数 all-gather / 梯度 reduce-scatter(DP 轴)与 MoE 的 dispatch/combine **All-to-All**(EP 轴)。#3797 让二者**重叠**:把 FSDP 的 `post_forward_release_module` / `post_backward_release_module` 等 hook 暴露出来交给 1F1B overlap pipeline 手动调度,并新增 `enable_fine_grained_param_gather_backward_hook` 支持 backward 侧细粒度参数 gather;配合 `delayed_wgrad`(expert 权重梯度延迟到 dispatch-backward 后再 reduce-scatter,见 §11.2)最大化 EP-A2A 与 DP-梯度同步的重叠。
> 这是对 §7 ④.3"参数 AG 必须靠 prefetch/计算重叠掩盖"的 FSDP-内部强化。通信调度/1F1B 重叠角度详见 [[20_megatron_comm_overlap_analysis]]。

---

## 8. HSDP —— 混合分片数据并行

`outer_dp_sharding_strategy`(`megatron/core/distributed/distributed_data_parallel_config.py:182`)+ `num_distributed_optimizer_instances`。

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

### 8.1 通信量视角(补充,2026-07-31 · 由原 `16_megatron_distributed_optimizer_analysis.md` §2.3.2/§2.3.3/§3.2 并入)

用 §3.9 的通信组记号,设 `num_distributed_optimizer_instances = K`,`D` 为 DP 世界大小、`P` 为参数总量(元素个数,= `Ψ`):

```
Intra-instance(组内 D/K 个 rank,对应 §3.9 的 Intra-Instance DP):
  Forward:  1× AllGather(param) = P/K bytes(组内共享完整参数)
  Backward: 1× ReduceScatter(grad) = P/K bytes(组内梯度分片)
Inter-instance(跨 K 个 instance,对应 §3.9 的 Inter-Instance DP):
  Backward: 1× AllReduce = 2P/D × (K-1) bytes(组间去重)
  总: 2P/K + 2P/D×(K-1) bytes
```

`num_distributed_optimizer_instances > 1` 时,DP 域划分为组内(intra-instance)和组间(inter-instance)——组内做 Reduce-Scatter 分片 + 组间做 All-Reduce 去重,参见 `megatron/core/distributed/param_and_grad_buffer.py:785-810`(`start_grad_sync` 内的 inter-instance all-reduce 分支)。这与上面 ASCII 图的"内层分片、外层副本"是同一机制的公式化表达。

---

## 9. 开销分析汇总

### 9.1 模型态显存阶梯(标准 bf16 + Adam,`dp` = DP 度,单位:字节×Ψ)

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

### 9.2 通信量

| 阶段 | 梯度通信 | 参数通信 | 趟数合计 | 相对 DDP |
|------|---------|---------|---------|----------|
| ① no_shard | all-reduce(1 趟) | —— | 1 趟 all-reduce | 1× |
| ② optim | reduce-scatter | all-gather | RS + AG | **1×** |
| ③ optim_grads | reduce-scatter | all-gather | RS + AG | **1×** |
| ④ optim_grads_params | reduce-scatter | 前向 AG + 反向 AG | RS + 2×AG | **1.5×** |

要点:**ZeRO-1/2 与 DDP 通信量完全相同**(`all-reduce = RS + AG`)—— 这是 ZeRO-1 被视为"免费"的根本原因;**ZeRO-3 才多付 50%**(参数前向、反向各 gather 一次)。FP8/FP4 参数精度对这张表的影响(通信量再降 25%-37.5%)见 §3.10。

### 9.3 DP 的"等效气泡"

DP 无流水线气泡。低效来源是**通信暴露**:
- 梯度 RS/all-reduce → `overlap_grad_reduce` 分桶藏进反向,只剩最后一个桶的尾巴。
- 参数 AG → `overlap_param_gather` 藏进前向 / optimizer step。
- ZeRO-3 的逐层参数 AG → 靠 prefetch(`suggested_communication_unit_size`)与逐层计算重叠,重叠不好就是吞吐损失。

---

## 10. 适用场景及选型

### 10.1 选型决策树(按 ZeRO 阶段)

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

### 10.2 一句话总结

- **DP 的本质**:最外层并行,切 batch 不切模型;朴素 DP 每卡冗余存 `18Ψ` 模型态。
- **DDP 的工程**:扁平梯度 buffer + 分桶 + 反向 post-hook,让梯度通信异步重叠进反向。
- **模型态 `18Ψ`**:标准 bf16 训练 = bf16 权重 `2Ψ` + **fp32 梯度 `4Ψ`** + fp32 master `4Ψ` + Adam m `4Ψ` + v `4Ψ`。梯度是 fp32 因为 bf16 跨 microbatch 累加必丢精度(Megatron 强制)。
- **ZeRO 四阶段**:逐级切分模型态 —— ZeRO-1 切优化器(`18Ψ→6Ψ+12Ψ/dp`,**通信不变,几乎免费**)、ZeRO-2 再切梯度、ZeRO-3 连参数也切(`18Ψ/dp`,**通信 ×1.5**)。
- **关键恒等式**:`all-reduce = reduce-scatter + all-gather` —— ZeRO-1/2 只是把 DDP 的一次 all-reduce 拆两半,所以通信零增量。
- **HSDP**:分片压在节点内 NVLink、跨节点只做副本,兼顾省显存与跨节点效率。

### 10.3 按规模的具体配置(补充,2026-07-31 · 由原 `16_megatron_distributed_optimizer_analysis.md` §5 并入)

| 场景 | 推荐配置 |
|------|---------|
| 单机 8 GPU,<10B 模型 | `use_distributed_optimizer=True`, 无需特殊设置 |
| 多机 32 GPU,70B 模型 | `use_distributed_optimizer=True`, `overlap_param_gather=True` |
| 多机 128 GPU,200B+ MoE | `use_distributed_optimizer=True`, `overlap_param_gather=True`, `overlap_grad_reduce=True`, 考虑 `fp8_param=True` |
| 极端规模(H100/Blackwell) | 全开 overlap, `fp4`/`fp8_param`, 复用 `reuse_grad_buf_for_mxfp8_param_ag` |
| 显存仍然不足 | 启用 `optimizer_cpu_offload` 或 `offload_optimizer_states`([[26_megatron_optimizer_step_internals_deepdive]] §9) |

### 10.4 何时适用 checklist(补充,2026-07-31 · 由原 `16_megatron_distributed_optimizer_analysis.md` §6 并入)

- ✓ 任何使用 Adam/AdamW 的场景
- ✓ 模型越大收益越高(优化器状态占比随参数量线性增长)
- ✓ DP 世界大小 ≥ 2
- ✓ 可与 TP、PP、EP 叠加(正交优化)
- ✗ DP=1 时无收益(无分片对象)
- ✗ SGD 无状态优化器收益有限(SGD 只有 momentum buffer)

---

## 11. 三种梯度/参数分片实现方案对比:DistributedOptimizer / TorchFullyShardedDataParallel / MegatronFSDP

(补充,2026-07-31 · 由原 `16_megatron_distributed_optimizer_analysis.md` 附录 A 并入,提升为正式章节)Megatron-LM 现有 **三套** 并行梯度/参数分片方案,适用于不同场景:

### 11.1 三套方案概览

| 维度 | DistributedOptimizer | TorchFullyShardedDataParallel | MegatronFSDP |
|------|---------------------|------------------------------|-------------|
| **文件** | `megatron/core/optimizer/distrib_optimizer.py:113` | `megatron/core/distributed/torch_fully_sharded_data_parallel.py:28` | `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:94` |
| **分片粒度** | 参数级别(连续 buffer 切分) | Module 级别(FSDP Unit) | Module 级别(FSDP Unit) |
| **分片策略** | ZeRO-1/2(状态+梯度分片) | PyTorch FSDP2(参数+梯度+状态) | ZeRO-1/2/3 可配置 |
| **依赖** | 无外部依赖 | PyTorch >= 2.4, DTensor | 自研(不依赖 PyTorch FSDP) |
| **与 EP 协同** | 通过 `expert_parallel_buffers` 隔离 | **无 EP 专门处理**(FSDP2 路径不识别 expert 参数) | 通过 `has_expert_parameters` 自动检测 |
| **通信 Overlap** | `overlap_param_gather` + `overlap_grad_reduce` | PyTorch FSDP2 自动 | `overlap_param_gather` + `overlap_grad_reduce` 默认开启 |
| **CUDA Graph** | 兼容 | 不兼容(PyTorch FSDP2 限制) | 兼容 |
| **NCCL UB** | 不支持 | 不支持 | 支持(`nccl_ub` 减少 SM 占用) |

> [!contradiction] 上表「与 EP 协同」一行对 **TorchFullyShardedDataParallel** 的归因有误,且在新基线 `71092579` 下核实为**不成立**:`_check_module_parameter_types` 并不在 `megatron/core/distributed/torch_fully_sharded_data_parallel.py` 里,而是 MegatronFSDP 的方法,`has_expert_parameters = self._check_module_parameter_types()` 在 `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:296`。也就是说该列本该写的两格其实是同一个机制,且都属于 MegatronFSDP。新基线上 `torch_fully_sharded_data_parallel.py` 全文 `grep -n 'expert|Expert'` **零命中**,即这条 FSDP2 路径没有任何专门的 EP 处理。补记:此错并非本轮基线漂移造成——在旧基线 `ee3f1ffa…` 下 `_check_module_parameter_types` 同样不在该文件中,是原稿的归因错误。

`data_parallel_sharding_strategy = 'optim_grads'`(§6 阶段③,ZeRO-2)与 `'optim_grads_params'`(§7 阶段④,ZeRO-3)均由 Megatron-FSDP 实现;DistributedOptimizer(§5 阶段②)是 ZeRO-1 的原生实现。

### 11.2 MegatronFSDP 详细分析

> MegatronFSDP 的内部实现(FSDP unit 分组、四类 buffer、hook 状态机与双流水线、桶分配器、与 EP/TP/HSDP 的叠加、接入层)已归一到 [[36_megatron_fsdp_analysis]]。
> 本页只保留**三方对比**:概览见 §11.1,选型矩阵见 §11.4,MoE 场景定位见 §11.5。

### 11.3 TorchFullyShardedDataParallel 详细分析 (`megatron/core/distributed/torch_fully_sharded_data_parallel.py`)

对接 PyTorch FSDP2 `torch.distributed.fsdp.fully_shard` API:

**Sub-module 级别的 `fully_shard` 包装**(`megatron/core/distributed/torch_fully_sharded_data_parallel.py:60-64, 126-134`):
```python
sub_modules_to_wrap = {
    TransformerLayer,           # 所有 Transformer 层
    LanguageModelEmbedding,     # 初始嵌入层
    RotaryEmbedding,            # 初始 RoPE
    ColumnParallelLinear,       # 最终输出层
}
# 每个 sub-module 独立 fully_shard → 逐层 AllGather/释放参数
fully_shard(sub_module, mesh=device_mesh, reshard_after_forward=True)
```

**FP8 权重转置缓存处理**(`megatron/core/distributed/torch_fully_sharded_data_parallel.py:93-98`):
PyTorch FSDP2 无法感知 micro-batch 边界,会缓存 FP8 权重的转置版本。Megatron 通过 `save_custom_attrs` / `restore_custom_attrs` 机制在每次 `fully_shard` 前后保存/恢复参数的 FP8 属性,避免不必要的显存占用。

**与 Activation Checkpointing 的 Backward Prefetch 协调**(`megatron/core/distributed/torch_fully_sharded_data_parallel.py:136-141`):
```python
if config.recompute_granularity is not None:
    sub_module.set_modules_to_backward_prefetch(
        [prev_module] if prev_module else []
    )
```
显式设置 backward prefetch schedule,防止 Activation Checkpointing 的重复计算破坏 FSDP2 自动生成的 prefetch 顺序。

### 11.4 三套方案选型矩阵

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 标准训练(无特殊需求) | `DistributedOptimizer` | 最成熟、与所有 MCore 特性深度集成 |
| PyTorch >= 2.4 + 新项目 | `TorchFullyShardedDataParallel` | 对接 PyTorch 原生 API,社区支持好 |
| 需要 CUDA Graph + FSDP | `MegatronFSDP` | TorchFSDP2 不支持 CUDA Graph |
| 需要 NCCL UB 优化 | `MegatronFSDP` | 独有的 NCCL UserBuffer 支持 |
| 需要 ZeRO-3 全分片 | `MegatronFSDP` | `optim_grads_params` 策略 |
| MoE(与 EP 深度耦合) | `MegatronFSDP` | 自动检测 EP 参数 + Delayed Wgrad Overlap |
| HSDP 分层分片 | `MegatronFSDP` | 组内 ZeRO-3 + 组间复制的分层策略 |

### 11.5 全分片(ZeRO-3)在 MoE 场景的选型判据

> [!update] 2026-09-02 · 本节已改写
> 原标题为「为什么 FSDP2 在 MoE 训练中重要」，但四条里前三条讲的其实是 **MegatronFSDP** 的能力，
> 只有第四条是 TorchFSDP2。这与本页 §11.1 的 `[!contradiction]` 与 §12.3 的结论直接冲突——
> **FSDP2 路径没有任何 EP 相关处理**：`megatron/core/distributed/torch_fully_sharded_data_parallel.py`
> 全文 165 行，`expert` 零命中。原写法会让读者把「与 EP 的层级协同」读成 FSDP2 的能力。

**为什么 MoE 场景值得考虑全分片**：expert 数量增长（64 → 256+）后，EP 不能无限扩展——EP 度越大
All-to-All 的跨节点通信量越重（见 [[14_megatron_ep_analysis]] §11.1）。当 EP 已经顶到通信预算上限、
参数仍放不下时，剩下的办法是在 **DP 维度**继续切 expert 参数。这就是 ZeRO-3 档相对
DistributedOptimizer（ZeRO-1，只切优化器状态）的价值所在：它把参数本身也切了。

**这条路由哪个实现承担，本页的三方对比给了明确答案**：

| | DistributedOptimizer | **MegatronFSDP** | TorchFSDP2 |
|---|---|---|---|
| 分片档位 | ZeRO-1 | ZeRO-2/3 | ZeRO-3 |
| **EP 相关处理** | — | **有** | **无** |

MegatronFSDP 侧的证据：`megatron/core/distributed/fsdp/mcore_fsdp_adapter.py:63` 的
`_get_default_fsdp_unit_modules(overlap_moe_expert_parallel_comm)` 按该开关决定默认 FSDP unit 组成，
`:122-127` 在开启 MoE EP 通信重叠时把 `TEGroupedMLP` 与 `SharedExpertMLP` 纳入 unit——
**即"把 expert 当作独立 FSDP unit、用时 AllGather、用完释放"这件事是 MegatronFSDP 做的**，
机制详见 [[36_megatron_fsdp_analysis]] §7。

**TorchFSDP2 在这里的定位是另一回事**：它的价值不在 MoE，而在**跟随上游**——
`TorchFullyShardedDataParallel` 让 Megatron 直接复用 PyTorch 的 `fully_shard`
（per-param FSDP、DTensor 集成），降低自维护成本。代价就是上面那一格：**它不认识 expert**。
所以选型判据很直接——**MoE + 需要 ZeRO-3 → MegatronFSDP；稠密模型 + 想吃上游红利 → TorchFSDP2**。

### 11.6 FSDP 与并行拓扑的关系

> 已归一到 [[36_megatron_fsdp_analysis]] §7「与 EP / TP / HSDP·HFSDP 的叠加」：FSDP 最后作用于已按 TP/EP 切过的 shard，正交性由独立 DeviceMesh 维表达，**不是**包含当前 rank 的进程组彼此空交；该节同时解释 EP 双 mesh 与 HSDP/HFSDP 两档外层策略。

## 12. 约束

本页只保留 DDP、ZeRO/HSDP 与三种分片实现的约束；optimizer step、offload 和 emerging optimizer 的边界归 [[26_megatron_optimizer_step_internals_deepdive]] §11。文件简称沿用 §3.7：PGB = `megatron/core/distributed/param_and_grad_buffer.py`。

### 12.1 前提与不变量

| # | 前提 / 不变量 | 源码落点 | 破坏后的表现 |
|---|---|---|---|
| 1 | 开 distributed optimizer 时,整块 buffer 的 `numel` 必须被 DP 世界大小整除 | PGB`:1220-1223`(`assert self.numel % self.data_parallel_world_size == 0`;有 NVFP4 参数时打包 numel 另有同款断言) | 直接 assert 失败 —— 这正是 buffer 必须 padding 的原因,也是 §3.7 逆序贪心分桶之后还要补一次对齐的原因 |
| 2 | **不**开 distributed optimizer 时反过来**不允许** padding | PGB`:1224-1225`(`assert self.numel == self.numel_unpadded`) | 两条路径对 buffer 长度的要求相反,同一份 layout 不能混用 |
| 3 | 参数 all-gather 的预取假设"module 前向序 == 参数注册序" | PGB`:638-644`,只 `warnings.warn` | **不报错**,只是下一组 AG 已被提前派发、overlap 退化;§3.7 那张时间线图随之失效 |
| 4 | `reduce_scatter_with_fp32_accumulation` 与 `num_distributed_optimizer_instances > 1`(HSDP,§8)互斥 | PGB`:271-276` | assert 失败 —— "线上传低精度、本地 FP32 累加"的那条 RS 实现进不了 HSDP |
| 5 | `nccl_ub=True` 时不能再开 grad/param buffer 的 CPU backup | PGB`:1245-1252` | assert 失败;NCCL UserBuffer(§11.1)与"buffer 可换出到 CPU"二选一 |

### 12.2 代价

- **"分片不认参数边界"的账单在 checkpoint 侧。** 因为一个参数可能被两个 rank 各持一段(§3.8),优化器状态无法按参数直接存取,`_build_model_gbuf_param_range_map` 必须一次生成 `gbuf_world` / `gbuf_world_in_bucket` / `gbuf_local` / `param` 四组 range(`megatron/core/optimizer/distrib_optimizer.py:158-162`),重分片时还要跨这四层换算。§5 说 ZeRO-1"几乎免费",指的是显存与通信量,不包含这层复杂度。
- **重叠是拿"通信提前派发"换来的。** `finish_param_sync` 在 wait 完本组之后立刻派发下一组(§3.7 第 3 步),所以任何打乱前向序的改动(重排 module、动态跳层、非均匀 stage 切分)都会踩到上表第 3 条那个 warning —— 而它**只警告不报错**,退化是静默的。
- **ZeRO-3 多付 50% 通信**(§9.2),且参数 AG 与逐层计算强耦合。这条代价是 §11.4 选型矩阵把 ZeRO-3 放在最后一档、以及 §8 HSDP 存在的直接原因。
- **`no_shard` 下梯度已在 DP 维复制,再规约一次就错。** §4 的 2026-06-16 更新记录的正是这笔账:grad-stats 只能在 `model_parallel_group` 上规约,否则 grad norm 虚高、裁剪过度、不收敛。

### 12.3 故意不做的事

- **FSDP2 路径不处理 EP。** 新基线下 `megatron/core/distributed/torch_fully_sharded_data_parallel.py` 全文对 `expert|Expert` 零命中;expert 参数的自动识别只存在于 MegatronFSDP —— `has_expert_parameters = self._check_module_parameter_types()`(`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:296`,方法体 `:351`)。这不是行号漂移,而是这条路径**没有**这项功能(见 §11.1 表格与其后的 `[!contradiction]`)。
- **Megatron-FSDP 下不走 MXFP8 的 param-buffer 直拷,也只认一种 checkpoint 分片格式。** `_copy_main_params_to_param_buffer` 一发现 `ddp_config.use_megatron_fsdp` 就 `raise NotImplementedError`(`megatron/core/optimizer/distrib_optimizer.py:2957-2960`);`sharded_state_dict` 在 Megatron-FSDP 下只接受 `sharding_type == "fsdp_dtensor"`,其余一律 `NotImplementedError`(`:1576-1579`)。

---

## 13. 发展趋势

> [!note] 推断：以下方向判断锚定冻结基线中的弃用标记与当前实现，不是源码给出的交付承诺。optimizer/offload 侧趋势见 [[26_megatron_optimizer_step_internals_deepdive]] §12。

**四、DistributedOptimizer 的 checkpoint 格式在向 model-space 收敛,两条旧格式已挂弃用告警。**
`sharded_state_dict` 的 `sharding_type` 形参本身被标注「is deprecated and will be removed. Use `metadata["distrib_optim_sharding_type"]` instead」(`megatron/core/optimizer/distrib_optimizer.py:1562-1569`),默认值现在是 `'fully_sharded_model_space'`(`:1570-1573`);`fully_sharded_bucket_space` 另有一条「deprecated and will be removed in the future. Please switch to `full_sharded_model_space`」(`:1585-1592`);类内 `checkpoint_fully_reshardable_formats` 只列 `fully_reshardable` / `fully_sharded_model_space` / `fsdp_dtensor` 三种(`:127-131`)。**由此可推断**:与 buffer 布局绑定的 bucket-space 格式正在退场 —— 这正是 §12.2 第一条那笔"分片不认参数边界的 checkpoint 账单"被推着还的方向(检查点侧的全貌见 [[19_megatron_dist_checkpointing_analysis]])。

**五、`use_custom_fsdp` 已弃用,§11 的三套实现正在收敛成两极。**
`megatron/core/distributed/distributed_data_parallel_config.py:105-110` 明写「The flag `use_custom_fsdp` is deprecated and will be removed in future versions. Please use `use_megatron_fsdp` instead, as all functionality will be migrated there」。同时 FSDP2 那条路径在基线下对 `expert|Expert` 零命中(§12.3),而 MoE 相关的新工作(delayed wgrad、A2A overlap、grouped-expert 分桶)全部落在 Megatron-FSDP 侧(§11.2)。**由此可推断**:§11.4 的选型矩阵会继续朝"标准训练用 DistributedOptimizer、需要全分片就用 Megatron-FSDP"两极收敛,`TorchFullyShardedDataParallel` 更像跟随 PyTorch 上游 API 的对接层,而不是 MoE 的主路径 —— 对这一点源码本身没有表态。

---

## 14. 小结

- DDP 用连续扁平 buffer、反向 post-hook 和分桶，把梯度规约重叠进反向；`param_sync_func` 则把参数同步回调交给调度层。
- ZeRO-0/1/2/3 逐级切梯度、优化器状态与参数；`all-reduce = reduce-scatter + all-gather` 是通信分解的核心。
- HSDP 把分片限制在高速域、在域间保留副本；三种实现的选择取决于全分片、EP、CUDA Graph 与上游 PyTorch 接口。
- 一次更新内部如何 unscale、检查溢出、裁剪、更新 master 参数，以及 LR/WD 与 μP 如何落到 param group，见 [[26_megatron_optimizer_step_internals_deepdive]]。

---

## 配置契约：参数同步回调

| 字段 | 来源 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|---|
| `param_sync_func` | `ModelParallelConfig` | `Optional[Callable]` | `None` | 发起异步参数同步（例如 distributed optimizer parameter all-gather）的回调；接收待同步参数迭代器。 | `megatron/core/model_parallel_config.py:213-216` |

其余数值精度字段归 [[23_megatron_precision_cudagraph_fusion_analysis]]，FSDP 实现开关归 [[36_megatron_fsdp_analysis]]，scheduler 与 μP 字段归 [[26_megatron_optimizer_step_internals_deepdive]]。

## Related Pages

- [[26_megatron_optimizer_step_internals_deepdive]] —— 承接 optimizer factory、mixed-precision step、scheduler、CPU offload 与 Muon/μP。
- [[17_megatron_parallelism_orchestration_analysis]] —— 提供本页 DDP/optimizer 消费的 DP、DP-CP 与 expert-DP 进程组。
- [[20_megatron_comm_overlap_analysis]] —— 组合 DP 参数/梯度同步与其他并行轴的通信窗口。
- [[36_megatron_fsdp_analysis]] —— 深入 Megatron-FSDP 的 unit、buffer、hook 与全分片状态机。
- [[23_megatron_precision_cudagraph_fusion_analysis]] —— 解释 bf16/fp16/FP8 参数精度和 CUDA Graph 边界。
- [[19_megatron_dist_checkpointing_analysis]] —— 解释 distributed optimizer/FSDP 状态如何持久化与重分片。
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]] —— 返回本域索引。
