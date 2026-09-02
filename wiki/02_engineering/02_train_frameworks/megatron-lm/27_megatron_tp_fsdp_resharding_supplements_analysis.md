---
title: "Megatron-LM TP·FSDP·Resharding 补遗"
---

# Megatron-LM TP·FSDP·Resharding 补遗

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **重定基线**：2026-09-01 由 `71092579`（2026-08-27）推进，跨 7 个提交；该增量只触及 20 个 `megatron/` 文件，本页 `path:line` 引用所涉源文件均不在其中，故无行号漂移，无需逐条重核。
> **重定基线**：2026-08-28 由 `ee3f1ffa…`（2026-05-19）推进，跨 578 个提交；本页全部 `path:line` 形式的引用已在新基线下逐条重核;**代码块内被点名的符号与不带行号的裸路径不在该次扫描口径内**,已知漏网处已于 2026-08-28 单独更正。
> 核心文件:`megatron/core/distributed/fsdp/`(Megatron-FSDP，~12000 行)、`megatron/core/distributed/nonuniform_tp.py`(1461 行)、`megatron/core/resharding/`（~5000 行；旧基线页面记为 ~2000 行，实测即已是 ~5000 行，已更正）
> 配套阅读:`12_megatron_tp_analysis.md`、`16_megatron_distributed_optimizer_analysis.md`、`17_megatron_parallelism_orchestration_analysis.md`
> **叙事顺序**：本页按五拍组织——背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势。
> **最近更新**：2026-08-28。按五拍重排章节顺序；机制正文与既有引用未改。
> 定位:"第一层补遗"第③份。补齐 3 块前面只点了名的内容。先做一处**勘误**:tier-1 清单里把它们叫"非均匀 TP / 弹性 resharding",读源码后更准确的命名是 ——

| tier-1 旧称 | 准确含义 |
|------------|---------|
| Megatron-FSDP 内部 | ZeRO-2/3 的**具体实现机器**(`16_megatron_distributed_optimizer_analysis.md` 只给了概念) |
| "非均匀 TP" | **Nonuniform TP(NTP)= TP 级容错**:备用 rank 顶替故障 rank |
| "弹性 resharding" | **Resharding / Refit = 跨并行配置搬运权重**,主用于 RL 的训练→推理 |

---

## 1. 背景:三条与"怎么并行训练"正交的支线

这三块内容的共同点是:它们都**不在"怎么把一个模型并行训练起来"这条主线上**。前面几份文档回答的是"模型怎么切、通信怎么排";本页三块分别回答"ZeRO-2/3 那台机器长什么样""TP 组里死了一张卡怎么办""训练完的权重怎么换到另一套并行布局里去推理"。三者各自的问题陈述分别在 §3.1 / §4.1 / §5.1。

| # | 主题 | 代码 | 一句话 |
|---|------|------|--------|
| 1 | Megatron-FSDP 内部 | `megatron/core/distributed/fsdp/src/megatron_fsdp/` | ZeRO-2/3 的流水线 AG/RS、预取、临时桶分配器、DTensor 分片 |
| 2 | Nonuniform TP(NTP) | `megatron/core/distributed/nonuniform_tp.py` | TP 组里留备用 rank,核心 rank 故障时重分片续训 |
| 3 | Resharding / Refit | `megatron/core/resharding/` | 在两套并行布局间搬运权重(RL:训练模型→推理模型) |

---

## 2. 为什么这么设计:三块支线各自的取舍

三块内容彼此独立,设计动机也各不相同。本节把三者"为什么走这条路、否掉了什么"分别落到源码或官方文档的原话上;源码沉默处标为推断。

### 2.1 Megatron-FSDP:临时缓冲复用一组池,而不是每次 unshard 现分配

**源码陈述的定位**:`megatron/core/distributed/fsdp/src/README.md:15` 把 Megatron-FSDP 定义成「an NVIDIA-developed distributed parallelism library **written in native PyTorch** that provides a high-performance implementation of Fully Sharded Data Parallelism (FSDP)」,并强调「seamless cross-compatibility with various deep learning frameworks and parallelism libraries such as Megatron-Core」;兼容面逐条列在 `:21-24`(PyTorch DeviceMesh / DTensor / DCP、Megatron Core、TransformerEngine、NeMo 容器)。

**被否掉的替代①:每次 unshard 都新分配一块临时缓冲。**
`RotaryBucketAllocator` 的 docstring 把理由写死 —— 它「implements a circular buffer recycling strategy **to minimize memory fragmentation** in FSDP operations」,并解释「This approach helps prevent memory fragmentation that typically occurs with **frequent allocation and deallocation** of temporary buffers during FSDP operations」(`megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py:567-574`);`FixedPoolAllocator` 同口径 ——「maintains a fixed pool of pre-allocated buffers, reusing them to **reduce the overhead and fragmentation caused by frequent allocation and deallocation** of temporary buffers」(`:652-661`),池大小默认 2,注释直说是双缓冲(`:672`)。基类 `TemporaryBucketAllocator` 仍保留"用完即释放"的语义 ——「It helps optimize memory usage by allowing temporary buckets to be released when no longer needed」(`:462-470`)。
→ 判据是**拿显存换分配器抖动**:§3.4 的三种分配器就是这条权衡的三个档位;`fsdp_double_buffer` 多占的那份显存,换来的是可注册 NCCL user buffer 的稳定地址(`megatron/core/distributed/fsdp/src/README.md:141`)。

**被否掉的替代②:逐 Module 聚合参数。**
`enable_fine_grained_param_gather` 的说明是「modifies FSDP to all-gather parameters with per-Module granularity **instead of collectively unsharding all sub-modules of a unit module** in Megatron-FSDP」(`megatron/core/distributed/fsdp/src/README.md:134`)—— 默认是"整个 FSDP unit 一次 all-gather",细粒度是 opt-in。同一节还写明 `sync_model_each_microbatch` 虽然让训练循环"干净",却「damages performance in situations where optimization is delayed (e.g. gradient accumulation) when the communications of the previous training iteration can be overlapped with the compute of the next training iteration」(`:130`)—— 这解释了为什么默认不在每个 microbatch 收口。

### 2.2 Nonuniform TP:非侵入子类,且只弥合梯度

NTP 的四条设计取舍(为什么做成子类、为什么只动梯度不动参数、为什么 post-sync reshard 推迟到最后一个 bucket、为什么 spare rank 直接退出进程)已在 [[25_megatron_nonuniform_tp_analysis]] §2 逐条落到源码原话上,本页不重复。这里只保留一条锚点:模块 docstring 自称「Non-intrusive implementation」,并写明「All NTP logic is contained in this module as **subclasses of core components**, making it non-intrusive to the main codebase」(`megatron/core/distributed/nonuniform_tp.py:2-15`)—— §4.3 的"非侵入式设计"即由此而来。

### 2.3 Resharding:计划集中构建并缓存,同 rank 传输短路

**被否掉的替代①:每次 refit 都重建计划。**
`megatron/core/resharding/README.md:87-97` 把流程写成五步 —— 各 rank 抽取参数元数据 → `dist.gather_object()` 汇到 rank 0 → rank 0 建完整传输表 → scatter 回去(每 rank 只拿自己的 send/recv op)→ 「The plan is **cached so repeated refits skip steps 1-4**」(`:97`)。缓存表甚至专门带了一列 "Why":`_plan_cache` 的理由是「Avoid collective plan rebuild on repeated refits」,`_service_cache` 的理由是「Avoid re-creating CUDA streams / NVSHMEM buffers」(`:121-124`)。
→ 判据很直白:RL 循环里 refit **每个迭代都要做一次**,而建计划本身是一轮 gather + scatter 的集合通信,不能每次都付。

**被否掉的替代②:同 rank 的数据也走网络栈。**
「All backends detect same-rank (local) transfers via `task_id` and short-circuit them into direct `tensor.copy_()` **instead of going through the network stack**」(`megatron/core/resharding/README.md:83-85`)—— collocated 部署(同一批 rank 同时持训练与推理模型)是主用场景,这条短路直接免掉其中大部分传输。

**被否掉的替代③:MXFP8 在发送端量化。**
`megatron/core/resharding/README.md:110-113` 给出条件:1D swizzled scale「encodes scales across the full weight tensor. **Partial updates would corrupt the layout**, so all BF16 slices are accumulated into a single buffer and quantized once all slices arrive」;代码侧把它做成硬拒绝 —— `convert_on_send=True` 一旦遇到 `scale.ndim == 1` 直接 `NotImplementedError`,报文写着「Use convert_on_send=False instead, which receives BF16 and quantizes the full weight on the receiver」(`megatron/core/resharding/transforms.py:198-205`)。2D scale 则相反 ——「Slices are independent, so received BF16 data is converted to MXFP8 per-slice immediately」(`README.md:107-109`)。此外量化必须写进**持久** MXFP8 buffer,理由同样写明:「so that CUDA-graph device-pointer captures remain valid across refits」(`:115-117`)。

**为什么 plan 由 rank 0 独算。**

> [!note] 推断
> README 与源码陈述的是**事实**:planner 是「Centralized plan builder (rank 0 builds, scatters to all)」(`megatron/core/resharding/README.md:13`),元数据经 `dist.gather_object()` 汇总到 rank 0(`:90`),计划再 scatter 回去、每 rank 只收自己的 op(`:96`)。**§5.4 表里"避免每个 rank 各算一遍"这条理由由本页承担 —— README 与源码都没有这样陈述**,也没有任何地方比较过"每 rank 各自建计划"的方案。要引用这条判断,请回到 `megatron/core/resharding/README.md:13`、`:87-97` 与 `:121-124`,不要引用本段推断。

---

## 3. Megatron-FSDP 内部实现(ZeRO-2/3 的机器)

> 本节已整体归一到 [[36_megatron_fsdp_analysis]]——那里按五拍完整覆盖了 FSDP unit 分组、四类 buffer 与 ZeRO 阶梯、hook 状态机与两条流水线、四档桶分配器、与 EP/TP/HSDP 的叠加,以及接入层 `mcore_fsdp_adapter`。
> 本页保留另外两条支线:**NTP**(§4)与 **Resharding / Refit**(§5)。

## 4. Nonuniform TP(NTP)—— 已归一到 25 号页

> [!update] 2026-09-02 · 本节由机制重述收缩为指针
> 原 §4.1–§4.4 重述了 NTP 的机制，而**重述版本与源码相反**，故按 merge-over-coexist 收缩：
> 权威页是 [[25_megatron_nonuniform_tp_analysis]]，本页只留定位与一条锚点。
>
> 被推翻的三条（保留记录以免同样的误解再次写入）：
> 1. 原写「健康时 `tp_base − tp_spares` 个 core rank 干活，**spare 待命**」——
>    源码里 spare rank 在 `initialize_nonuniform_tp_process_groups` 中直接 `sys.exit(0)`
>    （`megatron/core/distributed/nonuniform_tp.py:556-558`，日志原文 "Rank %s is a spare rank, exiting"），
>    **不待命**；健康副本是全部 `tp_base` 个 rank 都在算。
> 2. 原写「故障 rank 的分片**重分片到存活 core rank**，通过 `_ntp_all_to_all`(`:100`) 做分片再分配」——
>    `:100` 只是保留非连续输出视图的通用 all_to_all 辅助，不是分片再分配；
>    **NTP 全程不搬参数数据**，两次 A2A 搬的是**梯度**：
>    "NTP gathers those extra gradients onto healthy core ranks, runs the data-parallel gradient
>    reduction on peerable ranks, then post-sync reshards **gradients** back to the extra ranks
>    before optimizer step."（`megatron/core/distributed/README_NONUNIFORM_TP.md:205-208`）
> 3. 原写「扛住组内 rank 故障**而不重启**」「spare rank 平时空转」——
>    [[25_megatron_nonuniform_tp_analysis]] 把 NTP 定位为**冷重启**容错，并明写它不做在线故障检测。

**一句话定位**：NTP 让一个 TP 组在**部分 rank 缺席**的情况下仍能训练——健康副本是完整 `tp_base`，
缩减副本少若干 rank，两者的梯度靠**两次 all-to-all**（同步前 gather 多出来的梯度、同步后 reshard 回去）
弥合，参数本身不搬。它以 Megatron 核心类的**子类**形式收在单文件
`megatron/core/distributed/nonuniform_tp.py`（1461 行），对主干零侵入。

机制、取舍、容错边界与代价见 [[25_megatron_nonuniform_tp_analysis]]；本页 §2.2 已给出「只弥合梯度」这条取舍的锚点。

---

## 5. Resharding / Refit —— 跨并行配置的权重搬运

`megatron/core/resharding/`。`megatron/core/resharding/README.md` 首句:"Transfer model weights between different parallelism configurations (TP, PP, EP, DP) ... Used primarily in RL loops to move weights from a training model to an inference model that may use a different parallelism layout."

### 5.1 动机与解决的问题

**RLHF / RL 训练**里有两个模型实例:**训练模型**(为训练吞吐选一套并行布局,如 TP8×PP4)和**推理模型**(为生成延迟选另一套,如 TP2×PP1,可能还是 MXFP8)。每个 RL 迭代:训练模型更新权重 → 必须把新权重**搬进**推理模型,而两者并行布局不同 —— 参数的切分方式完全不一样。Resharding 解决的就是"**把权重从布局 A 搬到布局 B**"。

### 5.2 架构(四层)

以下四个文件 / 目录均在 `megatron/core/resharding/` 下（仓库根相对路径）：

```
refit.py        高层 API:swap_model_weights / prepare_swap_model_weights;计划缓存;MXFP8 自动识别
   │
planner.py      rank 0 集中构建 ReshardPlan(谁把哪个参数切片发给谁),scatter 给所有 rank
   │
execution.py    把 plan 里的 send/recv op 提交给 CopyService,处理写回
   │
copy_services/  可插拔传输后端:
   ├── nccl     GPU↔GPU,torch.distributed P2P
   ├── gloo     经 CPU 中转,Gloo 进程组
   └── nvshmem  NVSHMEM 流水线式 GPU↔GPU
```

外加 `megatron/core/resharding/transforms.py` 的 `MXFP8ReshardTransform` —— 搬运途中顺带做**格式转换**(BF16 → MXFP8),给推理模型用。

### 5.3 流程

```
prepare_swap_model_weights()  ── 一次性:构建并缓存 ReshardPlan(此时不传数据)
        │                         若目标是 MXFP8,先把目标权重量化成持久 MXFP8 buffer
        ▼
每个 RL 迭代:
swap_model_weights(src_model, target_model, refit_method="nccl")
        │  按缓存的 plan,src 各 rank 把参数切片 P2P 发给 target 各 rank
        ▼
推理模型拿到训练模型的最新权重(布局已转换),开始 rollout
```

### 5.4 要点与适用场景

| 维度 | 说明 |
|------|------|
| 核心 | 不是"训练并行",是**两套并行布局之间的权重迁移工具** |
| 计划集中化 | rank 0 算 plan 再 scatter,避免每个 rank 各算一遍 |
| 后端可插拔 | collocated(同卡同时持两模型)用 nccl;分离部署可用 gloo/nvshmem |
| 适用 | RLHF / RL 训练循环;训练与推理用不同并行配置的任何场景 |

它与前面所有文档不同 —— 那些讲"怎么并行训练",这一节讲"训练好的权重怎么换到另一套并行布局里去推理"。

---

## 6. 约束

三块内容各有各的边界。下面按主题分列,每条都能落到一个 `file:line`,越出前提就不再适用。

### 6.1 Megatron-FSDP

| # | 前提 / 代价 | 源码落点 | 破坏后的表现 |
|---|---|---|---|
| 1 | HSDP 外层分片(`outer_dp_sharding_strategy == "optim"`)要求内层必须是 `optim_grads_params` | `megatron/core/distributed/fsdp/src/megatron_fsdp/fully_shard.py:350-359`,抛 `ValueError`;旁边挂着 `# TODO(@shjwudp, @cspades): Requires various modifications to support`(`:354`) | 直接抛错,报文写明「outer sharding is dependent on inner sharding」——§3.5 提到的 HSDP 组合不是任意的 |
| 2 | TP sub-mesh 即使不用 TP 也必须给 | 同文件 `:411-413` 注释:「TP sub-mesh should be optional if not using TP, but is required for Megatron, TransformerEngine (default TP=1), and strided sharding when using DTensor-based TP」 | 这是已知的接口债,不是可配置项 |
| 3 | FP8 参数只在"全分片 compute 参数 + 高精度 main weight"这一种形态下支持 | `megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py:3658-3662` 的 TODO 自陈:目前只覆盖 FSDP,`no_shard` / `optim` / `optim_grads` 的量化路径仍是开放问题 | ZeRO-1/2 下不要指望同等的 FP8 覆盖 |
| 4 | `nccl_ub` 会强制连带开启 `fsdp_double_buffer`,多占显存 | `megatron/core/distributed/fsdp/src/README.md:141`:「Enabling this option will cause additional memory overhead due to the requirement to enable the `fsdp_double_buffer` option」 | 显存换 SM 效率,对应 §3.4 的 `FixedPoolAllocator` |
| 5 | `keep_fp8_transpose_cache` 在 Blackwell 上没有收益 | 同文件 `:136`:「This option will cause (number of parameter × 1 Byte) of memory overhead, but can skip the weight transpose operation in the backward propagation. This feature will not give any benefit from the Blackwell architecture」 | 新硬件上纯亏显存 |
| 6 | `sync_model_each_microbatch` 配 `no_shard` / `optim` 时,用户必须自己调 `zero_grad_buffer()` | 同文件 `:133` 的 WARNING | 否则未分片梯度会被重复 all-reduce 进梯度累加 buffer(README 类比 PyTorch DDP 的 `no_sync()`) |
| 7 | mcore 侧的 device mesh 轴序写死 | `megatron/core/distributed/fsdp/mcore_fsdp_adapter.py:587-590`:`# TODO: Supports configurable (dp, cp, ep, tp) order.`,mesh 由 `einops.rearrange(..., "(dp_cp ep tp) -> ep dp_cp tp")` 生成 | 与 [[17_megatron_parallelism_orchestration_analysis]] §1.5 的 `order` 不是同一套可配置机制,想换轴序改不了 |

### 6.2 Nonuniform TP

NTP 的前提、代价与失效条件已在 [[25_megatron_nonuniform_tp_analysis]] §5(尤其 §5.6 的九条前提表)逐条列出,本页不重复。三条最容易踩到的:spare rank 默认会 `sys.exit(0)`、`non_active_ranks_per_dp` 填的是 local TP slot 而非 global rank、`tp_spares=0` 时整套 helper 静默退化为 no-op。本页 §4.4 那张开销表要与之合读。

### 6.3 Resharding / Refit

| # | 前提 / 代价 | 源码落点 | 破坏后的表现 |
|---|---|---|---|
| 1 | 源模型与目标模型都必须带 `pg_collection`,且字段齐全 | `megatron/core/resharding/README.md:131-140`:`tp` 必需;`dp` 必需(source 侧缺失时自动从 `parallel_state` 补);PP>1 需 `pp`;MoE 需 `ep`;expert TP 需 `expt_tp` | 缺字段就建不出计划——这也是 §5.2 那四层架构的输入契约 |
| 2 | 非 collocated 部署时,**闲置 rank 也必须参与集合通信** | 同文件 `:70-72`:「Idle ranks (must still participate in collectives)」,调用形如 `swap_model_weights(None, None, "nccl", ...)` | 漏掉即死锁——建计划那一步是集合操作 |
| 3 | `prepare_swap_model_weights()` 必须在目标模型参数**还是 BF16** 时调用 | 同文件 `:42-46` | 错过时机就量化不出持久 MXFP8 buffer,§5.3 那条"一次性准备"的前提不成立 |
| 4 | plan 缓存的 key 是 `(rank, src_config, dst_config, num_experts)` | 同文件 `:124` | 并行布局一变缓存即失效,要重付一次集合式建计划(§2.3) |
| 5 | 销毁进程组前必须先 `clear_all_caches()` | 同文件 `:126-127`:「Call `clear_all_caches()` before destroying distributed process groups to avoid stale references. This also finalizes NVSHMEM resources.」 | 悬垂引用 / NVSHMEM 资源不回收 |
| 6 | `_plan_tp` 只接受"单一 TP descriptor" | `megatron/core/resharding/planner.py:209-210`,`NotImplementedError` | 多维切分的参数走的是另一条 block-interleaved 路径(README `:93-94`) |
| 7 | `nvshmem` 后端需要 NVSHMEM 库 | `megatron/core/resharding/README.md:20`、`:81` | 缺库只能退回 `nccl` / `gloo`,后者是 CPU 中转、延迟更高 |

### 6.4 三块共同的代价

- **FSDP 的重叠是默认开着的。** `optim_grads` / `optim_grads_params` 下梯度 reduce-scatter 与参数 all-gather 的重叠默认打开(§3.2),关掉就严重掉速。这意味着**这条调优空间已经被用掉了** —— 再想省显存只能动 bucket 与分配器(§3.4),不能靠关重叠。
- **NTP 的 bucket 调参方向与普通 DDP 相反。** 见 [[25_megatron_nonuniform_tp_analysis]] §5.6:NTP 要"少而大"的桶,而普通 DDP 要"多而小"才有得重叠([[16_megatron_distributed_optimizer_analysis]] §3.7)。
- **Resharding 的固定成本是一次集合式建计划。** 五步流程里前四步都要通信(§2.3),只能靠 `_plan_cache` 摊薄;布局一变就要重付。

### 6.5 故意不做的事

- **Resharding 不做训练本身。** README 首句就把范围钉死:「Transfer model weights between different parallelism configurations (TP, PP, EP, DP) with optional format conversion (e.g. BF16 to MXFP8)」(`megatron/core/resharding/README.md:3-6`)——它是搬运工具,不是并行策略。
- **Megatron-FSDP 不承担 mesh 轴序的配置**(见 §6.1 第 7 条),轴序仍由适配层写死。
- **NTP 不碰 checkpoint 与优化器状态**,见 [[25_megatron_nonuniform_tp_analysis]] §5.2 / §5.3 与该页 §4 的零引用表。

---

## 7. 发展趋势

> [!note] 推断:锚点是基线 `71092579` 下的源码事实(experimental 子包、TODO/FIXME、弃用函数、发布方式),方向判断由本页承担,不是源码的自陈计划。

**一、Megatron-FSDP 正在被重写成一条"最小实现"路径。**
基线下多出一个 `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/` 子包(`dbuffer.py`、`fully_shard.py`、`layout.py`、`module.py`、`parameter_group.py`、`placement.py`、`__init__.py`)。三处 docstring 把意图写得很直白:`experimental/__init__.py:15`「Experimental Megatron-FSDP implementation.」、`experimental/fully_shard.py:15`「**Minimal** Megatron-FSDP fully_shard entrypoint.」、`experimental/module.py:15`「Module mixin for the **minimal** Megatron-FSDP path.」。对照现役实现:`megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py` 5332 行,文件第 3 行就挂着 `# TODO: Split this file into smaller files.`。**由此可推断**:§3.3 那套 `ParamAndGradBuffer` + 两条流水线的组织方式,正在被一套以 `DBuffer` / `Placement` / `ParameterGroup` 为骨架的实现替代;这条 experimental 路径里 `experimental/module.py:223` 还挂着一条与主仓 PR 绑定的待办 ——「After NVIDIA/Megatron-LM#5411 lands, move this sync to the optimizer post-step hook instead of running it every microbatch」。

**二、Megatron-FSDP 正在从"Megatron 的一个目录"变成"独立发布的库"。**
`megatron/core/distributed/fsdp/src/README.md:26-33` 已经给出 `pip install megatron-fsdp` 与 PyPI 链接,并把源码位置写成 GitHub 上的一个子路径;兼容面(`:21-24`)也是按"给外部框架用"的口径列的。同方向的直接证据是 `megatron/core/distributed/fsdp/src/megatron_fsdp/fully_shard.py:55` 那条 `TODO(@cspades): Copied from megatron.core.utils to avoid depending on MCore`。**由此可推断**:§3 讲的这台机器会继续减少对 mcore 内部约定的依赖,而 §3.5 提到的 `mcore_fsdp_adapter.py` 作为"缝合层"的角色会更吃重 —— 读 FSDP 代码时应当分清"库本体"与"接入层"两侧。

**三、DTensor 侧的接口仍在收敛,旧入口已挂弃用说明。**
`megatron/core/distributed/fsdp/src/megatron_fsdp/uneven_dtensor.py:390-392` 的 `gather_uneven_dtensor_to_full_tensor` 只剩一句 docstring「Deprecated: use `redistribute_uneven_dtensor_to_replicated` instead.」,函数体就是转调新名;同文件 `:127` 另挂一条同步优化待办。`param_and_grad_buffer.py:5248`、`:5283` 还留着两条校验缺口(「Add validation checks for the legality of DTensor」「Implement consistency check for duplicated TP parameters」)。**由此可推断**:§3.5 里"DTensor 分片 / 不整除分片"这一块尚未定型,跨版本使用 `--ckpt-format fsdp_dtensor` 时要留意接口改名与校验缺失。

**四、Resharding 会继续沿 transform 接口加档,而不是改 planner。**
现有三个后端定位明确(`megatron/core/resharding/README.md:77-81`:`nccl` 延迟最低且是默认、`gloo` 用于 NCCL 跨集群走不通的场合、`nvshmem` 走双缓冲流水线内核),格式转换则被抽成可插拔的 `ReshardTransform` 基类,目前只有 `MXFP8ReshardTransform` 一个实现(文件表见 `:142-154`)。基类留了三个空方法:`prepare_send` / `prepare_recv` / `finalize_recv` 都是 `raise NotImplementedError`(`megatron/core/resharding/transforms.py:47`、`:51`、`:62`),其中 `finalize_recv` 的 docstring 明说「This is where receiver-side format conversion can happen」(`:56-61`)。**由此可推断**:RL 侧对低精度推理格式的需求(见 [[30_megatron_rl_posttraining_consistency_analysis]])会沿这三个挂点继续加实现,而 §5.2 的四层架构与 planner 本身大概率不动 —— 源码对此没有表态。

---

## 8. 小结

| 主题 | 何时关心 | 与已有文档的关系 |
|------|---------|-----------------|
| **Megatron-FSDP 内部** | 用 ZeRO-2/3(`--use-megatron-fsdp`) | 是 `16_megatron_distributed_optimizer_analysis.md` 阶段③④的实现机器:预取流水线 AG/RS + 临时桶分配器 |
| **Nonuniform TP** | 千卡级长训练要扛 TP 组单点故障 | `12_megatron_tp_analysis.md` 的容错扩展;非侵入式子类 |
| **Resharding / Refit** | RL 训练,训练模型↔推理模型布局不同 | 独立工具,跨 `17_megatron_parallelism_orchestration_analysis.md` 描述的两套布局搬权重 |

至此"第一层补遗"3 份文档全部完成:① 并行编排 capstone、② PP 补遗(2026-08-01 起并入 `15_megatron_pp_schedulers_analysis.md` §1.5/§8,原页已删除)、③ TP·FSDP·resharding 补遗(本页)。

---

*生成依据:`Megatron-LM` `dev` 分支 `85902ef599ea4eb06ada7567a479c524b605767a`(2026-09-01;由 `71092579` 重定基线而来,更早一次为 2026-08-28 由 `ee3f1ff` 推进)。源码行号以该 commit 为准；2026-08-28 由 `ee3f1ff` 重定基线。配套文档:`12_megatron_tp_analysis.md`、`16_megatron_distributed_optimizer_analysis.md`、`17_megatron_parallelism_orchestration_analysis.md`、`15_megatron_pp_schedulers_analysis.md`。*

## Related Pages

- [[12_megatron_tp_analysis]] · [[16_megatron_distributed_optimizer_analysis]] · [[17_megatron_parallelism_orchestration_analysis]] · [[30_megatron_rl_posttraining_consistency_analysis]] · [[19_megatron_dist_checkpointing_analysis]] · [[15_megatron_pp_schedulers_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
