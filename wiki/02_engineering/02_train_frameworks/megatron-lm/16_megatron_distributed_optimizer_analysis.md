---
title: "Megatron-LM 数据并行、分布式优化器与优化器内部机制 深度解析"
---

# Megatron-LM 数据并行、分布式优化器与优化器内部机制 深度解析

> **源码基线**：`NVIDIA/Megatron-LM@71092579522a12522d9f323ae180c9825d01928a`（`dev`，2026-08-27）
> **重定基线**：2026-08-28 由 `ee3f1ffa…`（2026-05-19）推进，跨 578 个提交；本页全部 `path:line` 已在新基线下逐条重核。原先并存的两套基线（`ee3f1ffa…` 正文 + `232c478d4` 增量块）已统一到 `71092579`
> 核心文件:`megatron/core/distributed/distributed_data_parallel.py`、`megatron/core/distributed/param_and_grad_buffer.py`、`megatron/core/distributed/distributed_data_parallel_config.py`、`megatron/core/optimizer/distrib_optimizer.py`、`megatron/core/optimizer/optimizer.py`、`megatron/core/optimizer/grad_scaler.py`、`megatron/core/optimizer/clip_grads.py`、`megatron/core/optimizer_param_scheduler.py`、`megatron/core/optimizer/cpu_offloading/`
> 配套阅读:`15_megatron_pp_schedulers_analysis.md`、`14_megatron_ep_analysis.md`、`12_megatron_tp_analysis.md`、`13_megatron_cp_analysis.md`
>
> **2026-07-31 合并说明**:本页由原三篇高度交叠的独立页合并而成——`megatron_ddp_optimizer_analysis.md`(DP/DDP/ZeRO 分片"怎么切")、`megatron_optimizer_internals_analysis.md`(单个优化器实例内部"算什么")、以及本页原有的 `16_megatron_distributed_optimizer_analysis.md`(通信组/FP8-FP4/CPU offload/三种 FSDP 实现对比/Layer-Wise+Muon 集成)。三页原本已通过大量"配套阅读"互相指路,现按逐节台账去重合并为一页,保留篇幅更全面者(原 `megatron_ddp_optimizer_analysis.md` 的 ZeRO 阶梯框架)为骨架,其余两页独有段落逐字并入(见各节"补充"标注),文件名保留最通用的 `16_megatron_distributed_optimizer_analysis`。原两篇已删除,全库入链已改写指向本页。
>
> **更正记录(2026-05-22)**:原稿把模型态字节数写作 `16Ψ`(套用 ZeRO 论文假设 fp16 梯度)。经源码二次核查更正为 **`18Ψ`** —— `megatron/training/arguments.py:1319-1333` 中 bf16 训练强制 `accumulate_allreduce_grads_in_fp32`,梯度 buffer 为 fp32(4 字节)。详见 §1.4、§3.5。
>
> **叙事顺序**：本页按五拍组织——背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势。
> **最近更新**：2026-08-28。按五拍重排章节顺序；机制正文与既有引用未改。

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
> 源码陈述的只有三件事:`data_parallel_sharding_strategy` 的四个取值、语义与**默认值 `'no_shard'`**(`megatron/core/distributed/distributed_data_parallel_config.py:112-114`);开不开 `use_distributed_optimizer` 只改变 buffer 的规约算子 —— `reduction_collective = "reduce-scatter" if self.ddp_config.use_distributed_optimizer else "all-reduce"`,并把这句判断直接打成 INFO 日志(`megatron/core/distributed/param_and_grad_buffer.py:278-286`);以及 ZeRO-2/3 由 Megatron-FSDP 承担(§18)。**"因为 `all-reduce = reduce-scatter + all-gather`,所以 ZeRO-1 通信零增量、应当先切优化器状态"这层判断由本页承担,源码没有这样表态**,也没有任何注释比较过四个阶段的通信代价。要引用这条判断,请回到 `megatron/core/distributed/param_and_grad_buffer.py:278-286`、`megatron/core/distributed/distributed_data_parallel_config.py:112-114` 与 `docs/user-guide/features/dist_optimizer.md:12-19` 这三个 locator,不要引用本段推断。

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
4. `next_param_gather_bucket_group` 链按前向序在 DDP`:337-348` 串好(注释说明按桶逆序串,因 all-gather 按桶逆序发生);链首(第一组)AG 由 PP schedule 经 `config.param_sync_func` 先发(`megatron/core/pipeline_parallel/schedules.py:1321-1322`、`:1443-1455`;该回调在 `megatron/training/training.py:4202` 绑成 `model_chunk.start_param_sync`),或被 `finish_param_sync` 懒发(PGB`:630-631`)。
5. 若下一组 AG 已被提前派发 → PGB`:638-644` 警告 *"mismatch between the order of parameter registration and forward pass execution, which will hurt the communication-computation overlap performance"* —— **预取假设 module 前向序 == 参数注册序**。

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

Intra-Instance / Inter-Instance 两组是 §8 HSDP 的通信基础;EP DP Group 是 EP 场景下专家参数独立于稠密参数的分片域(见 §18.5)。

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
> - **行号漂移**:解耦梯度赋值 `shard_main_param.decoupled_grad = shard_model_grad`(见 §12.3)在新基线为 `megatron/core/optimizer/distrib_optimizer.py:2842`(`ee3f1ff` 时 `:2568`、`232c478d4` 时 `:2728`);`shard_main_param.grad = shard_model_grad.float()` 现 `:2844`(旧 `:2570` / `:2730`)。语义未变。
> - **MXFP8 共享 buffer 的 all-gather 后处理被抽函数**(#4771,`megatron/core/distributed/distributed_data_parallel.py:544` `_start_bucket_group_param_sync`):原内联在 `start_param_sync` 里的"把 all-gather 出的 MXFP8 参数从共享 buffer 拷回 `param.data` 并清零 buffer 供梯度累加"逻辑被抽成单 bucket-group 方法,便于 LayerWise+DistOpt 链式各自只同步自己的 bucket。
> - **ChainedOptimizer 的 MXFP8 defer-sync 改为探测 DDP config**(#4982,`megatron/core/optimizer/optimizer.py:1806`):`_should_defer_mxfp8_param_sync` 不再信 `OptimizerConfig.overlap_param_gather`,而是逐个探测子 `DistributedOptimizer.ddp_config.overlap_param_gather`(详见 §11.1 的 2026-06-16 更新)。

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

这与 §12(混合精度优化器:fp32 master 副本)是同一套"18 bytes/param"体系在精度可配置场景下的变体——`decoupled_grad` 让梯度/master/m/v 各自可选不同精度,而非固定 fp32,详见 §12.3。

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
> `no_shard`(ZeRO-0)下,梯度经 all-reduce 后在各 DP rank 上是**复制**的。原实现仍把梯度统计/范数在 dist-opt(DP)组上规约一遍 → grad norm **虚高**、梯度裁剪过度 → **不收敛**。修复:`no_shard` 时把 grad-stats 规约组从 DP 组改为只用 `model_parallel_group`(TP/PP)(`effective_intra_dist_opt_group = mp_group if no_shard else intra_dist_opt_group`);同时 `start_param_sync` 对 `no_shard` 直接 return(参数已复制,无需 all-gather),并禁止 `no_shard` 配 meta-device 初始化。此修复仅针对 Megatron-FSDP 的 `no_shard` 路径;与本页 §15 梯度范数"需跨 TP×PP all-reduce"的论述一致 —— 关键是 **DP 维度此时不能再 reduce**。

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
> `LayerWiseDistributedOptimizer` 已整合进本文的 DDP grad/param buffer 基建:它预计算 shard-aligned 的参数 layout,让每个矩阵整体落在某个 shard 内,从而复用本节的 reduce-scatter/all-gather 与 `overlap_grad_reduce`/`overlap_param_gather`。配 `--optimizer muon --use-distributed-optimizer` 时,Muon 管的 2D 矩阵权重走 LayerWise(等效 ZeRO-1/2 沿 DP 分片),其余 embedding/bias/LayerNorm 等**非-Muon 参数则路由到一个独立的标准 `DistributedOptimizer`**(本节的字节级 ZeRO),二者由 `ChainedOptimizer` 串起。新增 `_start_bucket_group_param_sync`(`megatron/core/distributed/distributed_data_parallel.py:544`)让两个 sibling 优化器各自只同步自己那批 bucket group,不重复 all-gather。详见本页 §19 的 2026-06-16 更新。

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

这就是 **FSDP(Fully Sharded Data Parallel)**。Megatron 有两套实现:Megatron-FSDP(`--use-megatron-fsdp`)与 FSDP2(`--use-torch-fsdp2`),完整对比见 §18。

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
> ZeRO-3 与 MoE 叠加时,**两层通信**同时存在:FSDP 的逐层参数 all-gather / 梯度 reduce-scatter(DP 轴)与 MoE 的 dispatch/combine **All-to-All**(EP 轴)。#3797 让二者**重叠**:把 FSDP 的 `post_forward_release_module` / `post_backward_release_module` 等 hook 暴露出来交给 1F1B overlap pipeline 手动调度,并新增 `enable_fine_grained_param_gather_backward_hook` 支持 backward 侧细粒度参数 gather;配合 `delayed_wgrad`(expert 权重梯度延迟到 dispatch-backward 后再 reduce-scatter,见 §18.2)最大化 EP-A2A 与 DP-梯度同步的重叠。
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
| 显存仍然不足 | 启用 `optimizer_cpu_offload` 或 `offload_optimizer_states`(§17) |

### 10.4 何时适用 checklist(补充,2026-07-31 · 由原 `16_megatron_distributed_optimizer_analysis.md` §6 并入)

- ✓ 任何使用 Adam/AdamW 的场景
- ✓ 模型越大收益越高(优化器状态占比随参数量线性增长)
- ✓ DP 世界大小 ≥ 2
- ✓ 可与 TP、PP、EP 叠加(正交优化)
- ✗ DP=1 时无收益(无分片对象)
- ✗ SGD 无状态优化器收益有限(SGD 只有 momentum buffer)

---

## 11. 优化器类层次与 ChainedOptimizer

> 本节起(§11-§16)讲**单个优化器实例内部**怎么运作——混合精度、loss scaling、梯度裁剪、LR 调度;§1-§10 讲的是"优化器状态怎么沿 DP 切分(ZeRO)"。一个训练步的尾部:反向算完梯度 → `finalize_model_grads`(DP/PP/SP 规约,见 §3.6)→ **`optimizer.step()`**(§13 拆开这个 `step()`)。bf16/fp16 训练有**数值精度**问题(梯度下溢、累加误差),需要 fp32 master 副本 + loss scaling;还要做**梯度裁剪**稳定训练、**LR/WD 调度**。这些都封装在 Megatron 的优化器类里。本页 §3/§5 阶段②的 `DistributedOptimizer`(ZeRO-1)是本节 `MixedPrecisionOptimizer` 的**子类** —— 分布式分片是"在哪算",本节是"算什么"。

`megatron/core/optimizer/optimizer.py` 的继承链:

```
MegatronOptimizer (ABC, :134)              抽象基类:clip_grad_norm / get_loss_scale / scale_loss / step
   │
   ├── MixedPrecisionOptimizer (:717)      混合精度:fp32 master 副本 + grad scaler
   │      ├── Float16OptimizerWithFloat16Params (:929)   fp16/bf16 模型参数的具体实现
   │      └── DistributedOptimizer          ← ZeRO-1,见 §5 阶段②
   │
   ├── FP32Optimizer (:1286)               纯 fp32,无 scaling、无 master 副本
   │
   └── ChainedOptimizer (:1478)            把多个优化器串成一个(见 §11.1)
```

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> **行号基线刷新**:`megatron/core/optimizer/optimizer.py` 自 `ee3f1ff` 起明显增长(emerging-optimizer / MXFP8 / layer-wise 相关代码),上面继承链的行号整体上移。新基线 `71092579` 锚点(`megatron/core/optimizer/optimizer.py`):
> - `MegatronOptimizer` `:134`(`ee3f1ff` `:100` → `232c478d4` `:133`)、`MixedPrecisionOptimizer` `:717`(`:465` → `:589`)、`Float16OptimizerWithFloat16Params` `:929`(`:654` → `:779`)、`FP32Optimizer` `:1286`(`:918` → `:1042`)、`ChainedOptimizer` `:1478`(`:1104` → `:1229`)。
> - `MixedPrecisionOptimizer.step()` `:895`(`:621` → `:745`);`prepare_grads` `:807`、`step_with_ready_grads` `:852`(`232c478d4` 时为 `:676`/`:712`)。
> 类层次与五步流程本身**未变**,仅行号漂移。

`get_megatron_optimizer`(`megatron/core/optimizer/__init__.py`)是工厂:按配置(`bf16`/`fp16`/`fp32`、是否分布式、是否 MoE)挑类、切 param group、装配。

### 11.1 `ChainedOptimizer` 为什么需要

一个模型常需要**多个优化器实例**:
- **稠密参数 vs 专家参数**:MoE 模型里专家参数走 EP 组、稠密参数走普通 DP 组,分片域不同 → 各用一个 `DistributedOptimizer`。
- `num_distributed_optimizer_instances > 1`(HSDP)。

`ChainedOptimizer` 把它们包成一个对外统一的优化器:`step()` 时依次驱动每个子优化器,`get_loss_scale` / `clip_grad` 跨子优化器协调。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — ChainedOptimizer 的 MXFP8 defer-sync 门控修正(#4982,`megatron/core/optimizer/optimizer.py:1806` `_should_defer_mxfp8_param_sync`)
> 当 `reuse_grad_buf_for_mxfp8_param_ag=True`(MXFP8 参数 all-gather 复用梯度 buffer)且 DDP 层 **未** 开 `overlap_param_gather` 时,链式 step 间会有参数 buffer 复用竞态,需把 MXFP8 参数同步**延迟**到所有子优化器 step 完成后再做。原实现用 `self.config.overlap_param_gather` 作为判据,但 `OptimizerConfig` 与 DDP config 的该字段**可能不一致**;修复后改为**直接探测每个子 `DistributedOptimizer.ddp_config.overlap_param_gather`**,任一为 False 即触发延迟同步。这是 ChainedOptimizer 与 DDP 层耦合的一个隐蔽点。

### 11.2 param group:weight decay 的区分

`get_megatron_optimizer` 建 param group 时把参数分两组:**该用 weight decay 的**(线性层权重)和**不该用的**(bias、LayerNorm 的 `weight`/`bias`)。后者 `weight_decay=0`。这是标准做法,避免对归一化/偏置施加权重衰减。

---

## 12. 混合精度优化器:fp32 master 副本

### 12.1 动机

模型用 bf16/fp16 做前向反向(省显存、快)。但**优化器更新若也用 bf16**:`param += lr · update`,当 `update` 比 `param` 小几个数量级时,bf16 的尾数位不够,加法**直接丢失** → 训练停滞。

**解法:fp32 master 副本**。优化器维护一份 fp32 的参数主拷贝,所有 Adam 更新在 fp32 上做;每步结束把 fp32 master **拷回** bf16 模型参数供下一步前向用。

### 12.2 这就是"18 bytes/param"的来源

`MixedPrecisionOptimizer` 持有的东西,正是 §1.4/§9.1 ZeRO 显存表里的 `18Ψ`:

| 张量 | 精度 | bytes/param | 谁持有 |
|------|------|-------------|--------|
| 模型权重 | bf16 | 2 | 模型 |
| 模型梯度 | **fp32** | **4** | DDP grad buffer(bf16 训练强制 fp32 累加) |
| **master 权重** | fp32 | 4 | **优化器** |
| **Adam 动量 m** | fp32 | 4 | **优化器** |
| **Adam 方差 v** | fp32 | 4 | **优化器** |
| | | **合计 18** | |

> 梯度为 **fp32(4 字节)** 而非 bf16(2 字节):bf16 尾数仅 7 位,跨 microbatch 累加会丢精度,Megatron 对 bf16 训练强制 fp32 梯度累积(`megatron/training/arguments.py:1319-1333`,见 §3.5)。仅 `--grad-reduce-in-bf16` 时梯度为 2 字节、合计 16。

§1.4/§5 阶段②说 ZeRO-1 把"优化器状态 `12Ψ`"切成 `1/dp` —— 切的就是这里的 master + m + v。

`FP32Optimizer` 则相反:模型本身就是 fp32,无需 master 副本、无需 scaler。

### 12.3 精度感知优化器:decoupled_grad

见 §3.11——`use_precision_aware_optimizer: True` 时,master 权重、exp_avg、exp_avg_sq 可采用不同的低精度格式,用 `.decoupled_grad` 解耦模型参数 dtype 和优化器 state dtype,而非本节默认的固定 fp32 master。

---

## 13. `optimizer.step()` 流程

在展开 `optimizer.step()` 内部五步之前,先看它在**整个训练迭代**里的位置(补充,2026-07-31 · 由原 `16_megatron_distributed_optimizer_analysis.md` §3.5 并入)——这条流程串起了 §3(DP 通信)与本节(优化器内部):

```
Forward(参数 All-Gather 可与计算 overlap,见 §3.3)
  → Backward(梯度 Reduce-Scatter 可与计算 overlap,见 §3.2)
    → _copy_model_grads_to_main_grads()  [model grad → FP32 main grad]
      → optimizer.step()(仅更新本 rank 的分片,下方五步)
        → _copy_main_params_to_model_params() [FP32 → BF16/FP8]
          → start_param_sync() [All-Gather 参数]
            → 下一轮迭代
```

`MixedPrecisionOptimizer.step()`(`megatron/core/optimizer/optimizer.py:895`;`prepare_grads` 在 `:807`、`step_with_ready_grads` 在 `:852`):

```python
def step(self):
    found_inf_flag = self.prepare_grads()          # ① 收梯度 + unscale + 查 inf/nan
    if found_inf_flag:
        return False, None, None                   #    有 inf/nan → 跳过本步
    grad_norm = 0.0
    if self.config.clip_grad > 0.0:
        grad_norm = self.clip_grad_norm(self.config.clip_grad)   # ② 全局梯度裁剪
    num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else 0  # ③ 可选统计
    success = self.step_with_ready_grads()          # ④ Adam 更新 + master→bf16 回拷
    return success, grad_norm, num_zeros_in_grad
```

五步:

```
① prepare_grads     bf16 模型梯度 ──拷贝/累加──► fp32 main grad
                     除以 loss scale(unscale)
                     扫描 inf/nan → found_inf_flag

   found_inf_flag?  ──是──► return False(跳过本步,dynamic scaler 随后降 scale)
        │否
② clip_grad_norm    跨 TP/PP/DP 算全局梯度范数,超阈值则等比缩放(§15)
        │
③ count_zeros       (可选)统计零梯度数,日志用
        │
④ step_with_ready_grads
        │           base optimizer(FusedAdam)在 fp32 master 上做 Adam 更新
        │           fp32 master ──拷回──► bf16 模型参数
        ▼
   返回 (success, grad_norm, num_zeros)
```

关键:**inf/nan 检查在最前面**。一旦发现非有限梯度,整步丢弃(参数不动),交给 dynamic scaler 调整(§14)。这是 fp16 训练能稳住的安全阀。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — `count_zeros` 兼容解耦梯度 / Megatron-FSDP(#4802,`megatron/core/optimizer/clip_grads.py:199` `count_zeros_fp32`,新基线下行号未变)
> 第 ③ 步 `count_zeros`(统计零梯度)原来固定读 `param.grad`。但两种新路径下梯度不在 `.grad`:① **precision-aware / 解耦优化器**(`use_decoupled_grad=True`)梯度在 `param.decoupled_grad`(见 §12.2 注、§3.11);② **Megatron-FSDP** 管理的参数梯度是 FSDP 分片后的 DTensor,需取 `._local_tensor`。修复后 `count_zeros_fp32` 先按 `use_decoupled_grad` 选 `decoupled_grad`/`grad` 属性,再对 `__fsdp_param__` 参数取 local shard,避免漏统计或读到 `None`。

---

## 14. Loss Scaling 与 GradScaler

### 14.1 动机:fp16 的下溢

fp16 动态范围窄(最小正规数 ~6e-5)。反向里很多梯度比这还小 → **下溢成 0** → 参数收不到更新。

**Loss scaling**:前向后把 loss 乘一个大数 `S`(`scale_loss`),反向链式法则使所有梯度同样 ×`S`,把小梯度抬进 fp16 可表示区间;`prepare_grads` 里再 ÷`S` 还原(unscale)。

> bf16 动态范围与 fp32 几乎一样宽,通常**不需要** scaler(或用 `ConstantGradScaler(1.0)`)。loss scaling 主要为 fp16。

### 14.2 两种 scaler(`megatron/core/optimizer/grad_scaler.py`)

**`ConstantGradScaler`**:固定 `S`。简单,适合 bf16 或已知稳定的场景。

**`DynamicGradScaler`(`megatron/core/optimizer/grad_scaler.py:64`)**:自适应。
- 连续 `growth_interval` 步无 inf/nan → `S ×= growth_factor`(往上试探,尽量大)。
- 连续 `hysteresis` 步检测到 inf/nan → `S ×= backoff_factor`(`<1`,减半之类)、且这些步**跳过更新**。
- `min_scale` 兜底。

直觉:`S` 越大越不下溢,但太大会上溢成 inf。dynamic scaler 在"尽量大"和"不溢出"之间自动平衡 —— 不断试着调大,溢出了就回退。

---

## 15. 梯度裁剪(`megatron/core/optimizer/clip_grads.py`)

`clip_grad_norm(clip_grad)`:计算**全局梯度范数** `‖g‖`(所有参数拼起来的 L2 范数),若 `‖g‖ > clip_grad`(默认 1.0),把所有梯度等比缩放到 `clip_grad`:`g ← g · clip_grad / ‖g‖`。

并行要点:参数被 TP/PP 切散在多卡,所以"全局范数"要**跨 TP×PP 组 all-reduce** 各分片的范数平方和再开根。`MegatronOptimizer.clip_grad_norm`(`megatron/core/optimizer/optimizer.py:444`;范数计算在 `megatron/core/optimizer/clip_grads.py:55` `get_grad_norm_fp32`、缩放在 `:147` `clip_grad_by_total_norm_fp32`)负责协调这次跨并行的范数规约。作用:挡住偶发的梯度尖峰,稳定训练。

---

## 16. LR / WD 调度(`OptimizerParamScheduler`)

`megatron/core/optimizer_param_scheduler.py:102`(`class OptimizerParamScheduler`)。每步更新 base optimizer 各 param group 的 **learning rate** 和 **weight decay**。

典型曲线 = **warmup + decay**:
- **warmup**:前 `lr_warmup_steps` 步,LR 从 0(或 `lr_warmup_init`)线性升到峰值 —— 避免初期大步长震荡。
- **decay**:之后按 `lr_decay_style` 衰减到 `min_lr`。可选:
  - `cosine` —— 余弦衰减,最常用。
  - `linear` —— 线性。
  - `constant` —— 不衰减。
  - `WSD`(Warmup-Stable-Decay)—— 先 warmup、再长时间**恒定**、最后 `wsd_decay_steps` 步快速衰减;便于在"stable"段任意点取 checkpoint 续训。
- weight decay 也可独立调度。

它不在 `optimizer.step()` 内部,而是训练循环每步单独调一次。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — per-param-group 调度覆盖值的 resume 修复(#5213,`megatron/core/optimizer_param_scheduler.py:102/151/351`,新基线下三处行号均未变)
> `OptimizerParamScheduler` 支持 **per-param-group 覆盖**:某个 param group 可以带自己的 `max_lr`/`min_lr`/`start_wd`/`end_wd`(`_OPT_PARAM_SCHEDULER_OVERRIDE_KEYS`),它们在 `get_lr()`/`get_wd()` 中**优先于** scheduler 的类级值。两个 bug 被修:
> 1. **`override_opt_param_scheduler` 模式下 resume 丢失覆盖值**:checkpoint 里 param group 携带的 max_lr/min_lr 会覆盖当前 run 的命令行参数。修复:`__init__` 时用当前 run 的参数快照各 group 的覆盖值(`self._param_group_scheduler_overrides`),`load_state_dict` 里新增 `_restore_param_group_scheduler_overrides()` 在重放 schedule 前还原。
> 2. **`step(increment=num_steps)` 时机错误**:原来在还原 `start_wd`/`wd_incr_style` 等 WD 字段**之前**就调了 `self.step()`,导致 resume 后第一步用了旧 WD 状态。修复:把 `step(increment=num_steps)` 移到所有字段还原(含覆盖值还原)**之后**。

---

## 17. CPU Offloading 机制(补充,2026-07-31 · 由原 `16_megatron_distributed_optimizer_analysis.md` §4 并入)

### 17.1 HybridDeviceOptimizer

`megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py:14` — 将参数按比例拆分到 GPU 和 CPU:

- `offload_fraction`(默认 0.5):控制多少参数放在 CPU
- 双流 Overlap:`_d2h_stream` 传梯度到 CPU,`_h2d_stream` 传参数回 GPU
- 支持 `param_update_in_fp32`:CPU 上做 FP32 更新
- 通过 step hooks 自动化参数回拷

名字里的 "Hybrid":一部分参数的优化器状态/更新在 GPU、一部分在 CPU,按显存压力混合 —— 用 PCIe 带宽 + CPU 算力换 GPU 显存(类比激活 offload 的思路,见 `18_megatron_recompute_analysis.md` §0.2)。

### 17.2 OptimizerStateOffloader

`megatron/core/optimizer/cpu_offloading/optimizer_state_offloader.py` — 在 optimizer.step() 完成后将状态暂存 CPU:
- offload:D2H 异步拷贝 exp_avg, exp_avg_sq, master weights
- release:GPU 显存 resize_(0) 释放
- reload:两阶段——先分配 GPU 显存,再 H2D 异步拷回
- 下次 step 前调用 `sync_before_step()` 等待 H2D 完成

---

## 18. 三种梯度/参数分片实现方案对比:DistributedOptimizer / TorchFullyShardedDataParallel / MegatronFSDP

(补充,2026-07-31 · 由原 `16_megatron_distributed_optimizer_analysis.md` 附录 A 并入,提升为正式章节)Megatron-LM 现有 **三套** 并行梯度/参数分片方案,适用于不同场景:

### 18.1 三套方案概览

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

### 18.2 MegatronFSDP 详细分析 (`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py`)

MegatronFSDP 是 NVIDIA 自研的 FSDP 实现,提供从 ZeRO-1 到 ZeRO-3 的完整分片谱系:

**分片策略**(`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:101-109`):
```python
# data_parallel_sharding_strategy 控制:
'no_shard'             # 传统 DP(无分片)
'optim'                # ZeRO-1: 仅优化器状态分片(+ FP32 主权重)
'optim_grads'          # ZeRO-2: 梯度 + 优化器状态分片
'optim_grads_params'   # ZeRO-3: 参数 + 梯度 + 优化器状态全分片
```

**四种训练状态**(`class TrainingState`,`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:51-63`):
```python
class TrainingState(Enum):
    FORWARD = auto()       # Forward: 参数需 unshard
    PRE_BACKWARD = auto()  # Pre-backward: 参数需 unshard
    POST_BACKWARD = auto() # Post-backward: 梯度需 re-shard
    IDLE = auto()          # 空闲:无 un/sharding 活动
```

**FSDP Unit 概念**(`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:130-133`):
FSDP Unit 是最小可释放模型单元。参数按 Unit 分组——在 Forward 进入 Unit 时 AllGather 参数,离开时释放;Backward 进入时重新 AllGather。默认 Unit = `TransformerLayer`。

**与 Activation Checkpointing 的协同**(`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:116-119`):
> 重算整个 Transformer Layer 时，参数只需 Gather 一次，随后可供重算和 Backward 计算共同使用。

**Delayed Wgrad Overlap**(`setup_delayed_wgrad_acc_hook`,`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:66-91`):
当启用 `overlap_dispatch_backward_with_experts_wgrad` 时,expert 参数的梯度 reduce-scatter 延迟到 MoE dispatch backward 完成后再执行,最大化 EP 通信与 DP 梯度同步的重叠。

**NCCL UserBuffer**(`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:153-160`):
```python
nccl_ub=True → 使用 NCCL UserBuffer 进行 FSDP 通信
  - 减少 SM 占用(通信操作占用更少计算资源)
  - 自动设置 fsdp_double_buffer=True(使用额外 GPU 显存换性能)
```

**HSDP 分层分片**(缺省 `ddp_config` 即此组合,`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:250-255`):
```python
data_parallel_sharding_strategy="optim_grads_params"  # 组内全分片
outer_dp_sharding_strategy="no_shard"                 # 组间无分片(复制)
```
组内做 ZeRO-3 全分片,组间做 AllReduce 去重——与 §8 的 HSDP 是同一思想在 ZeRO-3 场景下的具体配置。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — Megatron-FSDP 内部一组修复(FSDP-internal)
> **① `no_shard`(ZeRO-0)收敛性修复**(#3835/#3754,`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:1289-1290`、`megatron/core/optimizer/__init__.py:1071`):`no_shard` 下参数本就在各 DP rank 复制,故 ① `start_param_sync` 对 `no_shard` 直接 return(无需 all-gather);② 梯度统计/范数只能在 **TP/PP(model_parallel_group)** 上规约,**不能**再在 DP 维度规约(梯度已是 all-reduce 后的复制值,再 reduce 会**虚高 grad norm 致不收敛**)—— 通过 `effective_intra_dist_opt_group = mp_group if no_shard else intra_dist_opt_group` 实现。另禁止 `no_shard` 配 meta-device 初始化。详见 §4「阶段①」小节的 2026-06-16 更新。
> **② grouped expert 权重减少 padding**(#5013,`_should_split_from_grouped_expert_bucket`,`megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py:1889-1897`,调用点 `:2012`):当 ≥3D 的 grouped-expert 张量与异构 chunk-size-factor 混在同一 bucket 时,LCM 对齐会**放大 padding**;修复把这类 grouped-expert 张量拆到独立 bucket,避免 LCM 膨胀。利好大规模 MoE(见 §18.5)。
> **③ 跨 AllGatherPipeline reset 保留非-FSDP-unit bucket**(#4717,`megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py`):pipeline reset 时不再误清非 FSDP-unit 的 bucket。
> **④ A2A Overlap**(#3797):把 MoE 的 All-to-All dispatch/combine 与 FSDP 的参数 all-gather / 梯度 reduce-scatter 重叠,详见 §7「阶段④」小节与 [[20_megatron_comm_overlap_analysis]]。

### 18.3 TorchFullyShardedDataParallel 详细分析 (`megatron/core/distributed/torch_fully_sharded_data_parallel.py`)

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

### 18.4 三套方案选型矩阵

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 标准训练(无特殊需求) | `DistributedOptimizer` | 最成熟、与所有 MCore 特性深度集成 |
| PyTorch >= 2.4 + 新项目 | `TorchFullyShardedDataParallel` | 对接 PyTorch 原生 API,社区支持好 |
| 需要 CUDA Graph + FSDP | `MegatronFSDP` | TorchFSDP2 不支持 CUDA Graph |
| 需要 NCCL UB 优化 | `MegatronFSDP` | 独有的 NCCL UserBuffer 支持 |
| 需要 ZeRO-3 全分片 | `MegatronFSDP` | `optim_grads_params` 策略 |
| MoE(与 EP 深度耦合) | `MegatronFSDP` | 自动检测 EP 参数 + Delayed Wgrad Overlap |
| HSDP 分层分片 | `MegatronFSDP` | 组内 ZeRO-3 + 组间复制的分层策略 |

### 18.5 为什么 FSDP2 在 MoE 训练中重要

1. **ZeRO-3 对 Expert 参数的全分片**:MoE 的 expert 数量增长(64→256+),在 EP 不能无限扩展时(通信开销),FSDP 可在 DP 维度进一步分片 expert 参数。`MegatronFSDP` 的 `optim_grads_params` 支持参数级的分片释放,对 1.xT MoE 至关重要。

2. **与 EP 的层级协同**:EP 处理跨 expert 的分布(通信模式:All-to-All),FSDP 处理跨 DP 的参数分片(通信模式:AllGather/ReduceScatter),两者正交叠加。

3. **FSDP Unit 粒度控制**:MoE 中的 Expert 可以作为独立的 FSDP Unit,在不需要时释放参数,需要的 token 到达时再 AllGather。

4. **PyTorch FSDP2 (`fully_shard`) 的原生支持**:`TorchFullyShardedDataParallel` 让 Megatron 可以跟随 PyTorch 上游的 FSDP2 优化(如 per-param FSDP、DTensor 集成),降低维护成本。

### 18.6 FSDP 与并行拓扑的关系

**FSDP 的 shard 在哪个进程组维度上执行?**

FSDP 的 shard 在 **DP 维度**上执行(即 `world_size / (TP × CP × PP × EP)`)。FSDP 的 shard group 不应与 TP/EP/CP/PP group 重叠,因为那些 group 已经做了参数分片或 activation 分片,FSDP 只负责 DP 维度的冗余消除。

```
在 TP=4, EP=8 配置下的 FSDP 分组:
  总 rank = 256
  TP group:    4 个 rank(同 PP stage, 同 EP rank)
  EP group:    8 个 rank(跨 expert)
  PP group:    P 个 rank(跨层)
  剩余维度 = 256 / (4 × 8 × P) = DP size
  FSDP shard:  在 DP group 上执行(只在此 group 内 reduce-scatter/all-gather)

  关键:FSDP group ∩ TP group = ∅
        FSDP group ∩ EP group = ∅
        FSDP group ∩ PP group = ∅
```

---

## 19. Layer-Wise 分布式优化器与 Muon 集成

(补充,2026-07-31 · 由原 `16_megatron_distributed_optimizer_analysis.md` 附录 A.7 与原 `megatron_optimizer_internals_analysis.md` §7 合并——前者讲"分片布局怎么整合进 DDP buffer",后者讲"优化器数学实现本身与版本更新",互补)

### 19.1 ChainedOptimizer 分片布局整合(原 §A.7)

`Layer-Wise Distributed Optimizer`(`--layer-wise-distributed-optimizer`)将参数按**层**分配到 DP rank,而非按扁平的参数列表:

**解决的问题**:
- 支持**多个优化器组合**(如 Muon 处理 ≥2D 矩阵参数,AdamW 处理 vector/bias 参数),普通 distributed optimizer 难以优雅支持 per-parameter optimizer 切换
- 更细粒度的 all-gather overlap:可在计算第 L 层 forward 的同时,异步 all-gather 第 L+1 层的参数

**ChainedOptimizer 分配规则**:
- 通过 `param_group` 的 `optimizer_name` 或 `foreach` 映射规则路由不同参数到不同底层优化器
- 例如:所有 `weight` 矩阵参数(≥2D)→ `MuonOptimizer`,所有 `bias`、`norm`、`embedding` 参数 → `AdamWOptimizer`

**选择场景**:使用混合优化器(如 Muon + AdamW)或超大模型需要极致 per-layer overlap 时。

> [!deprecated] 2026-06-16:**触发方式更正**。不存在 `--layer-wise-distributed-optimizer` 这个 flag。Layer-wise 分布式优化器通过 **`--optimizer muon`(或其它 emerging 优化器)+ `--use-distributed-optimizer`** 触发:`megatron/training/arguments.py:1853-1866` 在 optimizer 非 `sgd`/`adam` 且开了 distributed optimizer 时,把 `use_layer_wise_distributed_optimizer` 置 True、并关掉普通 `use_distributed_optimizer`。`--optimizer dist_muon` 是旧写法,已弃用。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — LayerWise 与 DDP buffer 基建整合 + 非-Muon 参数改走真正的 DistributedOptimizer(#4509 / #4771,`megatron/core/optimizer/layer_wise_optimizer.py`、`megatron/core/optimizer/__init__.py:725-972` 的 `_get_megatron_emerging_optimizer`、`megatron/core/optimizer/distrib_optimizer.py:3223`)
>
> 这组 PR 实质性改写了 layer-wise 的实现,并**修正了上文"普通 distributed optimizer 难以优雅支持 per-parameter optimizer 切换"的暗示** —— 现在两者是**链式协作**,而非二选一:
>
> **① LayerWise 不再用独立 ping-pong 路径,而是建在 DDP 的 grad/param buffer 之上**(#4509)。它预计算一个 shard-aligned 的 `FullParamLayout`/`PerBufferParamLayout`(`megatron/core/optimizer/param_layout.py`),把参数按 backprop 顺序装进**对齐到 shard 边界**的 bucket,使任何参数都不跨 shard 边界,从而能直接复用 DDP 的 reduce-scatter(`use_distributed_optimizer=True` 时,见 §3)/ all-gather 通信与 `overlap_grad_reduce`/`overlap_param_gather` 重叠语义(见 §3.2/§3.3)。装箱算法在 #4771 中从"同尺寸配对(size-matching)"换成 **LPT 贪心装箱**(按 numel 降序塞进当前负载最小的 shard),在保证 bucket 连续 backprop 区间的同时让各 shard 尽量均衡。
>
> **② 非-Muon 参数改由独立的 `DistributedOptimizer` 按字节级分片管理**(#4771)。新增 `is_managed_by_layer_wise_optimizer(param)`(`megatron/core/optimizer/layer_wise_optimizer.py:43`):2D 矩阵权重且非 embedding/output → Muon/LayerWise 接管;embedding、bias、LayerNorm 等 → **路由到一个独立的 `DistributedOptimizer`**(真正的 ZeRO 字节级分片,见 §5 阶段②)。`BufferKey` 增加 `is_managed_by_layer_wise_optimizer` 维度(新基线上 `BufferKey` 已移出 `param_and_grad_buffer.py`,现为 `megatron/core/optimizer/param_layout.py:46`,该字段在 `:66`;`param_and_grad_buffer.py` 侧改为在 `group_params_for_buffers` 里 `from ..optimizer.param_layout import BufferKey` 导入,`megatron/core/distributed/param_and_grad_buffer.py:938-964`),让两类参数落进不同 buffer;`DistributedOptimizer.start_param_sync_for_bucket_group_subset()`(`megatron/core/optimizer/distrib_optimizer.py:3223`)只同步自己那批 bucket group,避免与 sibling LayerWise 重复 all-gather。最终 `LayerWiseDistributedOptimizer`(Muon)+ `DistributedOptimizer`(Adam)由 `ChainedOptimizer`(§11.1)串成一个。
>
> **结论(对上文 Muon/ZeRO 框架的修正)**:Muon 现在**可以与 ZeRO 分片共存**。Muon 管的矩阵权重经 LayerWise 走 shard-aligned 的 reduce-scatter/all-gather(等效 ZeRO-1/2 沿 DP 分片优化器状态与梯度),非-Muon 参数走标准 `DistributedOptimizer`。早期"Muon 对 ZeRO 切分的根本性挑战"指的是 Newton-Schulz 正交化需要**整块矩阵**、无法像 Adam 那样按字节随意切;LayerWise 的解法正是 **shard-aligned bucket + 按层/按整参数分配**,让每个矩阵整体落在某个 shard 内,从而既正交化又分片(跨框架的 Muon/ZeRO 张力综述见 [[32_distributed_optimizer_deepdive]] §六)。
>
> **限制**:此 split 路径要求 `use_layer_wise_param_layout=True`(默认开;`--no-use-layer-wise-param-layout` 回退到 legacy ping-pong)、`num_distributed_optimizer_instances == 1`、且不支持 expert-parallel 的非-Muon 参数组与 `overlap_param_gather_with_optimizer_step`(`megatron/core/optimizer/__init__.py:761` 断言)。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — MTP-stage word_embeddings 必须打 `is_embedding_or_output_parameter` 标签(#5034,`megatron/core/models/common/language_module/language_module.py:205-213`,新基线下行号未变)
> `is_embedding_or_output_parameter` 标签决定参数被 Muon/LayerWise 接管还是路由给 Adam/DistOpt(见上)。MTP(Multi-Token Prediction)阶段的 `word_embeddings.weight` 是 pre_process embedding 的**副本**(靠跨 stage all-reduce 同步),原来漏打此标签 → 被 LayerWise 当作 2D 矩阵接管、且因 `shared_embedding=True` 在 `_emit_bucket` 里把整个 `(vocab × hidden)` 张量**复制到全部 `dp_size` 个 shard**,使该 chunk 的 buffer 膨胀约 8×。修复:`pre_process` 或 `mtp_process` 任一为真就打标签,让 MTP embedding 正确归 Adam/DistOpt 管理。

### 19.2 优化器实现与版本更新(原 `megatron_optimizer_internals_analysis.md` §7)

| 优化器 | 文件 | 一句话 |
|--------|------|--------|
| **Muon** | `megatron/core/optimizer/muon.py` | 新型优化器,对矩阵参数用 Newton-Schulz 正交化更新方向;v0.16 引入,配 layer-wise 分布式优化器(§19.1) |
| **layer-wise 分布式优化器** | `megatron/core/optimizer/layer_wise_optimizer.py` | 按层组织优化器状态/通信,降低峰值显存(§19.1) |
| **CPU offload** | `megatron/core/optimizer/cpu_offloading/`(`HybridDeviceOptimizer`) | 把优化器状态与 step 计算放 CPU,`--optimizer-cpu-offload`,GPU 显存极紧时用,详见 §17 |
| emerging optimizers | `megatron/core/optimizer/emerging_optimizers.py` | 其他较新优化器 |

> [!deprecated] 2026-06-16:**Muon 的真正实现不在 `megatron/core/optimizer/muon.py`**。`megatron/core/optimizer/muon.py` 已是一个 28 行的 *backward-compatible shim*(`get_megatron_muon_optimizer` 在 `megatron/core/optimizer/muon.py:8`,仅转调 `get_megatron_optimizer`,且 `dist_muon` 已弃用;新基线 `71092579` 下该文件仍为 28 行)。Muon / AdaptiveMuon 的实际实现是 `megatron/core/optimizer/emerging_optimizers.py` 里的 `TensorParallelMuon`(`:160`)/ `TensorParallelAdaptiveMuon`(`:294`),经 `_EMERGING_OPTIMIZERS` 注册表(声明 `megatron/core/optimizer/emerging_optimizers.py:152`、填充 `:429`)接入,并依赖外部包 `emerging-optimizers`。注:此 shim 在 `ee3f1ff` 已存在,原表项的文件归属一直是错的。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — emerging optimizers / Muon 一组更新
> **① 升级到 v0.3.0**(#5320,`pyproject.toml`、`megatron/core/optimizer/emerging_optimizers.py`):外部 `emerging-optimizers` 包由 v0.2.0 → **v0.3.0**;`TensorParallelAdaptiveMuon` 新增暴露 `scale_mode` / `extra_scale_factor`;`OptimizerConfig` 删除 `soap_precondition_frequency` 字段。注册表当前内建 `muon`、`adaptive_muon`(本地 TP 版),并自动收编上游包注册的其它优化器(如 SOAP)。
> **② 触发方式**:emerging 优化器通过 `--optimizer muon`(或 `adaptive_muon`/`soap` 等,即非 `sgd`/`adam`)选择;若同时 `--use-distributed-optimizer`,会自动转成 **layer-wise distributed optimizer**(`megatron/training/arguments.py:1853-1866`,`use_layer_wise_distributed_optimizer=True`)。`--optimizer dist_muon` 已弃用。emerging 优化器目前**不支持** Torch-FSDP2 / Megatron-FSDP(`megatron/training/arguments.py:1879-1882` 断言)。
> **③ Muon 参数路由(关键)**:默认 override 规则把 **非线性/embedding/output 参数路由给 Adam**(`_is_nonlinear_or_embedding`,定义在 `megatron/core/optimizer/emerging_optimizers.py:128-130`、作为默认 override 注册在 `:81-87`),Muon 只接管 2D 矩阵权重。配合 #4509/#4771,Muon 矩阵权重走 `LayerWiseDistributedOptimizer`、其余 Adam 参数走独立 `DistributedOptimizer`,二者由 `ChainedOptimizer` 串起(详见 §19.1 的 2026-06-16 更新)。
> **④ Muon QKV split 支持 gated attention**(#4728,`megatron/core/optimizer/emerging_optimizers.py:133` `_get_qkv_split_shapes`、`megatron/core/optimizer/__init__.py:779-797`):Muon 对 fused `linear_qkv.weight` 需按 Q/K/V 分块各自做 Newton-Schulz 正交化。当 `attention_output_gate=True`(门控注意力)时,QKV 切分形状由 3 段变为 **4 段** `[q, q_gate, k, v]`;并改为**逐参数**携带 `param.qkv_split_shapes`,且对 `shape[0] % sum(splits) != 0` 的参数跳过 QKV 标记(避免误切)。
> **⑤ QK-Clip**:`megatron/core/optimizer/qk_clip.py:31`(`clip_qk`)对注意力 QK logits 做裁剪以稳住数值,是 Muon 训练注意力稳定性的配套件(该文件在 `ee3f1ff` 已存在,此前表中漏列)。

---

## 20. 约束

前面几节把 DDP / ZeRO / FSDP / offload 的收益都算清楚了,但这些收益只在下列前提成立时才是那个形状。每条都能落到一个 `file:line`,越出前提就不再适用。文件简称沿用 §3.7:PGB = `megatron/core/distributed/param_and_grad_buffer.py`。

### 20.1 前提与不变量

| # | 前提 / 不变量 | 源码落点 | 破坏后的表现 |
|---|---|---|---|
| 1 | 开 distributed optimizer 时,整块 buffer 的 `numel` 必须被 DP 世界大小整除 | PGB`:1220-1223`(`assert self.numel % self.data_parallel_world_size == 0`;有 NVFP4 参数时打包 numel 另有同款断言) | 直接 assert 失败 —— 这正是 buffer 必须 padding 的原因,也是 §3.7 逆序贪心分桶之后还要补一次对齐的原因 |
| 2 | **不**开 distributed optimizer 时反过来**不允许** padding | PGB`:1224-1225`(`assert self.numel == self.numel_unpadded`) | 两条路径对 buffer 长度的要求相反,同一份 layout 不能混用 |
| 3 | 参数 all-gather 的预取假设"module 前向序 == 参数注册序" | PGB`:638-644`,只 `warnings.warn` | **不报错**,只是下一组 AG 已被提前派发、overlap 退化;§3.7 那张时间线图随之失效 |
| 4 | `reduce_scatter_with_fp32_accumulation` 与 `num_distributed_optimizer_instances > 1`(HSDP,§8)互斥 | PGB`:271-276` | assert 失败 —— "线上传低精度、本地 FP32 累加"的那条 RS 实现进不了 HSDP |
| 5 | `nccl_ub=True` 时不能再开 grad/param buffer 的 CPU backup | PGB`:1245-1252` | assert 失败;NCCL UserBuffer(§18.1)与"buffer 可换出到 CPU"二选一 |
| 6 | `overlap_param_gather_with_optimizer_step` 与 `reuse_grad_buf_for_mxfp8_param_ag`(§3.10)互斥 | `megatron/core/optimizer/optimizer_config.py:513-517`,抛 `ValueError` | 构造期直接失败 —— 共享 buffer 一旦复用,参数 AG 就不能再提前塞进 step |
| 7 | 精度感知优化器(§3.11/§12.3)只支持 `adam`,且必须同时开 distributed optimizer | 同文件 `:519-525` | assert 失败 |
| 8 | chunked optimizer-state offload(§17)限 Adam/Muon、与 `optimizer_cpu_offload` 互斥、且不支持 optimizer CUDA graph | 同文件 `:445-448`、`:450-452`、`:453-455` | assert 失败 —— §17.1 的 `HybridDeviceOptimizer` 与 chunked offload 是两条不能同开的路 |
| 9 | emerging 优化器(Muon 等,§19)不能配 `overlap_param_gather_with_optimizer_step`,也不支持 fp16 | `megatron/core/optimizer/__init__.py:761-766`、`:769-770` | 前者 assert 失败,断言文本自己给了理由 ——「the emerging-optimizer path does not split model_chunks into (first, rest) groups, so the per-chunk param-gather dispatch never fires」;后者 `ValueError` |

### 20.2 代价

- **"分片不认参数边界"的账单在 checkpoint 侧。** 因为一个参数可能被两个 rank 各持一段(§3.8),优化器状态无法按参数直接存取,`_build_model_gbuf_param_range_map` 必须一次生成 `gbuf_world` / `gbuf_world_in_bucket` / `gbuf_local` / `param` 四组 range(`megatron/core/optimizer/distrib_optimizer.py:158-162`),重分片时还要跨这四层换算。§5 说 ZeRO-1"几乎免费",指的是显存与通信量,不包含这层复杂度。
- **重叠是拿"通信提前派发"换来的。** `finish_param_sync` 在 wait 完本组之后立刻派发下一组(§3.7 第 3 步),所以任何打乱前向序的改动(重排 module、动态跳层、非均匀 stage 切分)都会踩到上表第 3 条那个 warning —— 而它**只警告不报错**,退化是静默的。
- **ZeRO-3 多付 50% 通信**(§9.2),且参数 AG 与逐层计算强耦合。这条代价是 §18.4 选型矩阵把 ZeRO-3 放在最后一档、以及 §8 HSDP 存在的直接原因。
- **`no_shard` 下梯度已在 DP 维复制,再规约一次就错。** §4 的 2026-06-16 更新记录的正是这笔账:grad-stats 只能在 `model_parallel_group` 上规约,否则 grad norm 虚高、裁剪过度、不收敛。

### 20.3 故意不做的事

- **FSDP2 路径不处理 EP。** 新基线下 `megatron/core/distributed/torch_fully_sharded_data_parallel.py` 全文对 `expert|Expert` 零命中;expert 参数的自动识别只存在于 MegatronFSDP —— `has_expert_parameters = self._check_module_parameter_types()`(`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:296`,方法体 `:351`)。这不是行号漂移,而是这条路径**没有**这项功能(见 §18.1 表格与其后的 `[!contradiction]`)。
- **Megatron-FSDP 下不走 MXFP8 的 param-buffer 直拷,也只认一种 checkpoint 分片格式。** `_copy_main_params_to_param_buffer` 一发现 `ddp_config.use_megatron_fsdp` 就 `raise NotImplementedError`(`megatron/core/optimizer/distrib_optimizer.py:2957-2960`);`sharded_state_dict` 在 Megatron-FSDP 下只接受 `sharding_type == "fsdp_dtensor"`,其余一律 `NotImplementedError`(`:1576-1579`)。
- **优化器不负责构造进程组。** `get_megatron_optimizer` 只在拿不到 `pg_collection.tp` 时回落到全局单例,并用 `setattr` 把 tp_group 挂上去,旁边留着 `# TODO(M4): plumb tp_group through optimizer constructors so this setattr disappears`(`megatron/core/optimizer/__init__.py:688-692`)—— 进程组的构造是 [[17_megatron_parallelism_orchestration_analysis]] 的职责,本页只消费。

---

## 21. 发展趋势

> [!note] 推断:锚点是基线 `71092579` 下的源码事实(弃用标记、TODO、被替换掉的文件),方向判断由本页承担,不是源码的自陈计划。

**一、CPU offload 正在从"整块状态换出"改成"分块换出",旧接口只剩兼容别名。**
§17.2 描述的那条路径已经不在原文件里:`megatron/core/optimizer/cpu_offloading/` 在基线下只剩 `__init__.py`、`README.md`、`hybrid_optimizer.py` 与**新增的** `chunked_optimizer_state_offload.py`(`class ChunkedOptimizerStateOffloader` 在 `:57`),由提交 `9050d4c5f`(commit message「[dev] Add chunked optimizer-state and master-weight offload (#6244)」)引入,并被 `MegatronOptimizer` 直接持有(`megatron/core/optimizer/optimizer.py:52` 导入、`:149` 字段、`:172` `enable_chunked_optimizer_state_offload`)。配置侧 `offload_optimizer_states` 已降级为「Deprecated alias for :attr:`chunked_optimizer_state_offload`」(`megatron/core/optimizer/optimizer_config.py:385-386`),`__post_init__` 对它发 `FutureWarning` 并自动改写成新开关(`:418-429`),`megatron/core/optimizer/cpu_offloading/README.md:98` 也只把它称作「parser compatibility spelling」。**由此可推断**:§17.2 那套"step 后整块 D2H → `resize_(0)` → 两阶段 reload"的粒度正在被 chunk 化(chunk 大小与 offload 比例都成了可调项),再调 offload 时应按新开关读代码,不要沿用旧类名。

**二、参数 layout 正在从 DDP buffer 里独立出去,成为多个优化器共享的第三方描述。**
§19.1 已经记录:`BufferKey` 不再定义在 `param_and_grad_buffer.py`,而是搬去 `megatron/core/optimizer/param_layout.py:46`,再由 `group_params_for_buffers` 反向导入(`megatron/core/distributed/param_and_grad_buffer.py:938-964`);§3.7 那条更正里的 bucket 末端对齐 divisor 同样集中到了 `megatron/core/optimizer/param_layout.py:29`。**由此可推断**:"谁决定参数怎么装桶"正在从 DDP 侧移到优化器侧 —— 因为 LayerWise(Muon)与 `DistributedOptimizer` 必须对同一份 layout 达成一致(§19.1);后续读分桶代码,应先看 `param_layout.py` 再看 `param_and_grad_buffer.py`。

**三、进程组正在从全局单例改成显式传入,而优化器这一层还没走完。**
`get_megatron_optimizer` 仍在 `pg_collection` 缺 `tp` 时回落到 `parallel_state.get_tensor_model_parallel_group()`,并用 `setattr` 把 tp_group 挂到优化器实例上,旁边写着「TODO(M4): plumb tp_group through optimizer constructors so this setattr disappears」(`megatron/core/optimizer/__init__.py:688-692`);工厂入口另有一条「TODO: the standard and emerging optimizer paths handle pg_collection differently; unify them so both use a single pg_collection-based flow」(`:1023-1024`)。**由此可推断**:§11 的类层次短期内不会变,但"优化器从哪里拿 DP/TP 组"会变;跨版本对照优化器代码时,通信组的来源是最容易漂移的一处(编排侧的同向变化见 [[17_megatron_parallelism_orchestration_analysis]])。

**四、DistributedOptimizer 的 checkpoint 格式在向 model-space 收敛,两条旧格式已挂弃用告警。**
`sharded_state_dict` 的 `sharding_type` 形参本身被标注「is deprecated and will be removed. Use `metadata["distrib_optim_sharding_type"]` instead」(`megatron/core/optimizer/distrib_optimizer.py:1562-1569`),默认值现在是 `'fully_sharded_model_space'`(`:1570-1573`);`fully_sharded_bucket_space` 另有一条「deprecated and will be removed in the future. Please switch to `full_sharded_model_space`」(`:1585-1592`);类内 `checkpoint_fully_reshardable_formats` 只列 `fully_reshardable` / `fully_sharded_model_space` / `fsdp_dtensor` 三种(`:127-131`)。**由此可推断**:与 buffer 布局绑定的 bucket-space 格式正在退场 —— 这正是 §20.2 第一条那笔"分片不认参数边界的 checkpoint 账单"被推着还的方向(检查点侧的全貌见 [[19_megatron_dist_checkpointing_analysis]])。

**五、`use_custom_fsdp` 已弃用,§18 的三套实现正在收敛成两极。**
`megatron/core/distributed/distributed_data_parallel_config.py:105-110` 明写「The flag `use_custom_fsdp` is deprecated and will be removed in future versions. Please use `use_megatron_fsdp` instead, as all functionality will be migrated there」。同时 FSDP2 那条路径在基线下对 `expert|Expert` 零命中(§20.3),而 MoE 相关的新工作(delayed wgrad、A2A overlap、grouped-expert 分桶)全部落在 Megatron-FSDP 侧(§18.2)。**由此可推断**:§18.4 的选型矩阵会继续朝"标准训练用 DistributedOptimizer、需要全分片就用 Megatron-FSDP"两极收敛,`TorchFullyShardedDataParallel` 更像跟随 PyTorch 上游 API 的对接层,而不是 MoE 的主路径 —— 对这一点源码本身没有表态。

---

## 22. 全文小结

**ZeRO 分片(§1-§10)**:
- DP 的本质是最外层并行,切 batch 不切模型;朴素 DP 每卡冗余存 `18Ψ` 模型态。
- DDP 的工程:扁平梯度 buffer + 分桶 + 反向 post-hook,让梯度通信异步重叠进反向。
- ZeRO 四阶段逐级切分模型态——ZeRO-1 切优化器(几乎免费)、ZeRO-2 再切梯度、ZeRO-3 连参数也切(通信 ×1.5)。关键恒等式 `all-reduce = reduce-scatter + all-gather`。
- HSDP 把分片压在节点内 NVLink、跨节点只做副本。

**优化器内部机制(§11-§16)**:
- 优化器类层次:`MegatronOptimizer` → `MixedPrecisionOptimizer`(fp32 master)→ `Float16OptimizerWithFloat16Params` / `FP32Optimizer` / `DistributedOptimizer`;多实例用 `ChainedOptimizer` 串。
- 混合精度核心:模型 bf16、优化器持 fp32 master + m + v —— 这就是 ZeRO 显存表 `18 bytes/param` 的来源。
- `step()` 五步:prepare_grads(unscale + 查 inf/nan)→ 裁剪 → count_zeros → Adam 更新 + master 回拷;**inf/nan 即跳步**是 fp16 训练的安全阀。
- Loss scaling 把小梯度抬出 fp16 下溢区;`DynamicGradScaler` 自适应"尽量大又不溢出"。
- 梯度裁剪是全局范数裁剪,范数需跨 TP×PP all-reduce。
- LR/WD 调度:warmup + decay(cosine / WSD …),`OptimizerParamScheduler` 每步更新。

**扩展机制(§17-§19)**:CPU offload(HybridDeviceOptimizer/StateOffloader)用 PCIe+CPU 换 GPU 显存;三套分片实现(DistributedOptimizer/TorchFSDP2/MegatronFSDP)按依赖/CUDA-Graph/NCCL-UB/EP 协同需求择一;Layer-Wise+ChainedOptimizer 让 Muon(矩阵正交化,需整块权重)与 ZeRO 分片共存——shard-aligned bucket 是关键解法。

---

*生成依据:`Megatron-LM` `dev` 分支 `71092579`(2026-08-28 由 `ee3f1ff` 重定基线,途经 `dev@232c478d4` 的增量对照已一并推进)。源码行号以该 commit 为准。2026-07-31 由三篇原始页(`megatron_ddp_optimizer_analysis.md`、`megatron_optimizer_internals_analysis.md`、本页原稿)合并而成,配套文档:`15_megatron_pp_schedulers_analysis.md`、`14_megatron_ep_analysis.md`、`12_megatron_tp_analysis.md`、`13_megatron_cp_analysis.md`。*

## Related Pages

- [[17_megatron_parallelism_orchestration_analysis]] · [[22_megatron_memory_optimization_analysis]] · [[13_low_precision_training_analysis]] · [[14_megatron_ep_analysis]]
- [[12_megatron_tp_analysis]] · [[27_megatron_tp_fsdp_resharding_supplements_analysis]] · [[15_megatron_pp_schedulers_analysis]] · [[13_megatron_cp_analysis]]
- [[23_megatron_precision_cudagraph_fusion_analysis]] · [[19_megatron_dist_checkpointing_analysis]] · [[20_megatron_comm_overlap_analysis]]
- [[../32_distributed_optimizer_deepdive|32_distributed_optimizer_deepdive]] — FSDP2/ZeRO/MindSpeed 三方对比, 梯度累积通信量分析, Adam vs Muon
- [[11_muon_analysis]] — Muon 优化器本身的数学原理(Newton-Schulz 正交化),与本页 §19 的分布式集成互补
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
