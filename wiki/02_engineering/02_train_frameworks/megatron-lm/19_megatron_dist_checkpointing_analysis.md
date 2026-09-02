---
title: "Megatron-LM 分布式 Checkpoint 深度解析(Distributed Checkpointing)"
---

# Megatron-LM 分布式 Checkpoint 深度解析(Distributed Checkpointing)

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **重定基线**：2026-09-01 由 `71092579`（2026-08-27）推进，跨 7 个提交；该增量只触及 20 个 `megatron/` 文件，本页 `path:line` 引用所涉源文件均不在其中，故无行号漂移，无需逐条重核。
> **重定基线**：2026-08-28 由 `ee3f1ffa…`（2026-05-19）推进，跨 578 个提交；本页全部 `path:line` 形式的引用已在新基线下逐条重核;**代码块内被点名的符号与不带行号的裸路径不在该次扫描口径内**,已知漏网处已于 2026-08-28 单独更正。
> 核心文件:`megatron/core/dist_checkpointing/` 下 `megatron/core/dist_checkpointing/mapping.py`(`ShardedTensor`)、`megatron/core/dist_checkpointing/serialization.py`(`save`/`load`)、`megatron/core/dist_checkpointing/strategies/`、`megatron/core/dist_checkpointing/validation.py`
> 配套阅读:`17_megatron_parallelism_orchestration_analysis.md`、`16_megatron_distributed_optimizer_analysis.md`、`27_megatron_tp_fsdp_resharding_supplements_analysis.md` §5(resharding)
> **叙事顺序**：本页按五拍组织——背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势。
> **最近更新**：2026-08-28。按五拍重排章节顺序；机制正文与既有引用未改。
> 定位:模型/优化器状态如何**存盘与读取**。

---

## 1. 背景：模型被 TP×PP×EP×DP 切散,朴素存档与当时的并行配置死绑

### 1.1 它和 resharding 不是一回事

两者都处理"并行布局不同",但:

| | **dist_checkpointing**(本文) | **resharding / refit**(`27_megatron_tp_fsdp_resharding_supplements_analysis.md` §5) |
|--|------------------------------|--------------------------------|
| 干什么 | 模型/优化器状态**存盘 / 从盘读取** | 两个**运行中**的模型之间**实时**搬权重 |
| 介质 | 磁盘 | GPU↔GPU(NCCL/NVSHMEM/Gloo) |
| 场景 | 训练存档、断点续训、换集群续训 | RL:训练模型 → 推理模型 |
| 共同点 | 都要跨**不同并行布局** | |

本文讲 dist_checkpointing。

### 1.2 动机:并行无关的存档

一个模型被 TP×PP×EP×DP 切散在成百上千张卡上 —— **每张卡只持有每个张量的一小片**。

朴素存档(每张卡存自己那片)的致命问题:**checkpoint 和当时的并行配置死死绑定**。用 `TP8×PP4` 训练存的档,**只能用 `TP8×PP4` 加载**。想换并行度续训、换集群规模、或训练用一套布局而推理用另一套 —— 全做不到。

`dist_checkpointing` 的目标:**存成"并行无关"的格式**。checkpoint 里存的是逻辑上的**全局张量**;加载时每张卡声明"我要全局张量 X 的这一片",框架按需读对应切片。于是 **`TP8×PP4` 存、`TP2×PP1` 载** 也能跑。

---

## 2. 为什么这么设计：让每个 rank 只声明自己那一片,存与载全程不搬数据

朴素做法有两条:每张卡各存各的分片(§1.2 已否),或者先把所有分片 all-gather 成完整张量、由 rank 0 单点写盘。Megatron 两条都没走 —— 它让每个 rank 用一个**纯元数据的描述子**说清"我这一片在全局张量的哪个位置",落盘那一刻才把描述子翻译成后端格式。下面四条源码陈述了理由;第五条源码只给结论,由本页标为推断。

**① 全局布局由元数据推出,存与载都不需要数据通信。**
`megatron/core/dist_checkpointing/serialization.py` 的模块 docstring 把这条写进接口约定:`load`「expects the sharded state dict argument as a guidance for loading the sharded tensors」(`:5-8`)—— 加载端先用**当前布局**造一份只有元数据的 sharded state dict,再据它去读。写入端同理:把 MCore `ShardedTensor` 翻成 PyT 类型时源码明写「Create a ShardedTensor without invoking communication. Determine global shards」(`megatron/core/dist_checkpointing/strategies/torch.py:199`),全局分片表直接由 `axis_fragmentations` 的笛卡尔积枚举出来(`:203-210`)。
并行存档把这条推到极致:`FullyParallelSaveStrategyWrapper` 的 docstring 写「The save distribution happens without any *data* communication. Only the *metadata* is exchanged and based on data replication on different ranks, we try to distribute the save as uniformly as possible.」(`megatron/core/dist_checkpointing/strategies/fully_parallel.py:47-51`)。
→ 判据:**只要元数据自足,分片就能被任意重新划分** —— 这正是"`TP8×PP4` 存、`TP2×PP1` 载"成立的前提。

**② 为什么自留一套 `ShardedTensor`,而不是直接用 PyTorch DCP 的类型。**
转换函数的 docstring 把差异逐条列出:「Convert MCore ShardedTensor to PyT ShardedTensor. **PyT requires information about all chunks.**」,以及「Additionally, it saves `prepend_axis_num` and `has_flattened_range` (**specific to MCore**) as attributes for further restoration in `_unwrap_pyt_sharded_tensor`」(`megatron/core/dist_checkpointing/strategies/torch.py:140-145`)。
→ 判据由源码给出:PyT 的类型要求"所有 chunk 的信息",而 MCore 有两个 PyT 表达不了的概念(专家轴前置、扁平切片),只能挂成属性再还原。于是 MCore 的 `ShardedTensor` 是**本地自足的描述层**,PyT 类型只在落盘那一刻构造。

**③ 被否掉的后端写在历史里:zarr。**
`torch_dist` 不是唯一存在过的后端。提交 `04cb1b0db`(2025-10-29,#2004「zarr soft deprecation」)先给 zarr 策略挂上 warning:「`zarr` distributed checkpoint backend is deprecated. Please switch to PyTorch Distributed format (`torch_dist`).」;提交 `faa6037b8`(2026-01-16,commit message 即「fully remove zarr support (#2944)」)把整条路径删掉 —— 基线 `71092579` 下 `megatron/` 全树 `zarr` 零命中。
**源码只留下"改用 torch_dist"这一句,没有陈述取舍理由**,见本节末推断块。

**④ 两处"故意不抛错 / 故意延后写"的取舍,理由源码自陈。**
- checkpoint 目录非空时**只打 WARNING、不抛异常**,注释给的理由是「Don't throw exception here since this could cause a cascade of failures without human intervention in cases where multiple jobs are queued up.」(`megatron/core/dist_checkpointing/serialization.py:407-411`)—— 判据是**排队作业场景下抛异常会级联失败**,宁可覆盖一个残档。
- 异步存档把 `metadata.json` 挂成完成回调:「step (7) is added as one of the finalization functions, so that metadata.json is written only if the checkpoint is complete」(`:361-365`)—— 判据是**用一个后写的小文件换整档的原子性**。

**⑤ 并行加载明说自己是近似解 —— 被否掉的替代是"最优交换"。**
`FullyParallelLoadStrategyWrapper.load` 的 docstring 交代四步(交换元数据 → 各 rank 确定性排布 → 各读各的一份 → 全体交换分片),并直接承认:「Currently, the shards are all gathered between all ranks in the parallelization group. **This might not be optimal (some ranks do not need all tensors), but it's a reasonable approximation for an optimal exchange in most scenarios.**」(`megatron/core/dist_checkpointing/strategies/fully_parallel.py:196-211`);排布算法也自陈是贪心:「realized with a greedy algorithm described in `distribute_shards_to_ranks`」(`:206-207`)。
→ 判据:**用近似换实现复杂度**。真正的"最优交换"被留成 TODO(见 §10 ③)。

> [!note] 推断
> ③ 里"zarr 为什么输给 `torch_dist`"这层判断,源码**只给了结论、没有给理由** —— 两处提交信息与那行 warning 都只说"改用 torch_dist"。本页不为它编一个原因;要引用这条,请回到 `04cb1b0db`、`faa6037b8` 两个提交与「Please switch to PyTorch Distributed format (`torch_dist`)」这句话本身。
> 同样,① 末尾"元数据自足 ⇒ 分片可任意重划分"是本页把三处 docstring(`megatron/core/dist_checkpointing/serialization.py:5-8`、`strategies/torch.py:199`、`strategies/fully_parallel.py:47-51`)串起来的读法,源码没有把这条因果写成一句话。

---

## 3. 核心抽象:`ShardedTensor`

`megatron/core/dist_checkpointing/mapping.py:52`。整个子系统的核心 —— 描述**本地张量(本 rank 这一片)↔ 全局张量(逻辑全量)**的映射:

```python
@dataclass
class ShardedTensor(ShardedBase):
    key: str                       # 全局张量的唯一标识
    data: torch.Tensor             # 本 rank 持有的局部数据(校验时可为 None)
    dtype, local_shape             # 局部片的形状
    global_shape: Tuple            # 全局张量的形状
    global_offset: Tuple           # 本局部片在全局张量里的偏移(元素数)
    axis_fragmentations: Tuple     # 每个轴被切成几片
    replica_id: ReplicaId = 0      # 本片是第几个副本(DP 副本数据相同)
    prepend_axis_num: int = 0      # 全局张量比局部多出的前置轴数(如专家轴)
    allow_shape_mismatch: bool     # 允许全局形状不严格匹配(padded 张量)
    flattened_range: slice         # 局部片是某个扁平 buffer 的一段切片
```

读法:`(global_shape, global_offset, local_shape)` 三者一起,精确说明"本 rank 的 `data` 是全局张量 `key` 的哪一块"。关键字段:
- **`replica_id`**:DP 副本持有**完全相同**的数据。存档时只让 `replica_id` 为某个值的副本真正写盘,其余跳过 —— **避免 DP 冗余写**。
- **`prepend_axis_num`**:MoE 专家权重等,局部张量是单个专家、全局多了一个"专家"轴 —— 用前置轴表达。
- **`flattened_range`**:DDP 的扁平梯度 buffer、分布式优化器的扁平状态(见 `16_megatron_distributed_optimizer_analysis.md`)—— 局部片是大扁平 buffer 的一段,用这个字段定位。**注意**:基线 `71092579` 下裸 `ShardedTensor` 已**不再接受**该参数(直接 `raise CheckpointingException`,#2126 `5ab481cb4` 整体删除该路径),它现在只在 `ShardedTensorFactory` 上有效——详见约束节。
- `from_rank_offsets`(`:190`):便捷构造器,从"本 rank 在各并行轴的 rank 号"直接算出 offset。
- `narrow`(`:262`):把一个 ShardedTensor 再切窄(load 时按需取子片)。

**正是 `ShardedTensor` 把"并行布局"这个信息编码进了 checkpoint** —— 存的时候各片拼成全局,读的时候按新布局重新切。

---

## 4. 其他 sharded 类型

`megatron/core/dist_checkpointing/mapping.py` 还有三类(都继承 `ShardedBase`):

| 类型 | 用途 |
|------|------|
| **`ShardedObject`**(`:360`) | 非张量对象的分片(如各 rank 各有一份的元数据);`unique_key` 标识 |
| **`ShardedTensorFactory`**(`:438`) | **延迟构造**:存/载时需要变换的张量(如分布式优化器的扁平 buffer 要先重组成逻辑张量)。`build()` 生成 ShardedTensor,`apply()` 做逆变换 |
| **`LocalNonpersistentObject`**(`:342`) | **不该持久化**的对象 —— rank 本地、加载时重算的东西。`save` 时直接丢弃 |

---

## 5. `sharded_state_dict` —— 模型如何声明自己的分片

普通 PyTorch 模型用 `state_dict()` 返回 `{名字: 张量}`。Megatron 的每个模块额外实现 **`sharded_state_dict()`** —— 返回的 dict 里每个张量被包成 `ShardedTensor`,带上正确的"局部↔全局"映射。

前面文档见过的例子:`ColumnParallelLinear.sharded_state_dict`(`12_megatron_tp_analysis.md`,权重按输出维切→沿 axis 0 分片)、`RowParallelLinear.sharded_state_dict`(沿 axis 1)、`DistributedOptimizer.sharded_state_dict`(`16_megatron_distributed_optimizer_analysis.md`,优化器状态按 DP 切→用 `flattened_range`)、`TEGroupedMLP` 的专家用 `prepend_axis_num` 表达专家轴。

所以 `sharded_state_dict` 是**模型对"我的每一片在全局的哪个位置"的自我声明** —— dist_checkpointing 的 `save`/`load` 完全靠它驱动。

---

## 6. `save` / `load` 流程

### 6.1 `save`(`megatron/core/dist_checkpointing/serialization.py:332`)

```
save(sharded_state_dict, checkpoint_dir, ...):
  ① 应用 ShardedTensorFactory(延迟构造展开)
  ② 抽出并丢弃 LocalNonpersistentObject
  ③ 抽出所有 ShardedBase 对象
  ④ rank 0 把"非分片"的普通对象存进 common.pt
  ⑤ (可选)抽出并存 ShardedObject
  ⑥ 存所有 ShardedTensor —— 按 strategy(§7),各片写进各自位置
  ⑦ 写 metadata.json(后端、版本)
```

- 步骤⑥可**异步**(`async_sharded_save=True`)—— 返回一个 `AsyncRequest`,真正写盘在后台跑,训练继续;此时⑦作为完成回调,**整档写完才落 metadata.json**(保证 checkpoint 原子性)。
- `validate_access_integrity`（`megatron/core/dist_checkpointing/serialization.py:336` 的开关参数）触发 §8 的校验；真正执行校验的是 `validate_sharding_integrity`（`megatron/core/dist_checkpointing/validation.py:369`）。

### 6.2 `load`(`megatron/core/dist_checkpointing/serialization.py:62`)

逆过程:每张卡用**当前(可能是新的)并行布局**生成自己的 `sharded_state_dict`(只填 metadata、`data=None`),`load` 据此从 checkpoint 里读出对应的全局张量切片,填回 `data`。**这一步就是"换并行布局加载"发生的地方** —— 新布局的 `global_offset`/`local_shape` 决定读哪一段。

辅助 API:`load_tensors_metadata` / `load_sharded_metadata`(不载数据、只看 checkpoint 里有什么)、`load_plain_tensors`、`load_common_state_dict`、`remove_sharded_tensors`。

---

## 7. 策略(`strategies/`)

`save`/`load` 的"怎么写盘/读盘"由可插拔 strategy 决定:

| strategy | 文件 | 作用 |
|----------|------|------|
| **TorchDist** | `megatron/core/dist_checkpointing/strategies/torch.py` | 基于 PyTorch 分布式 checkpoint(DCP)格式,`--ckpt-format torch_dist`,默认 |
| **fully-parallel** | `megatron/core/dist_checkpointing/strategies/fully_parallel.py` | **全并行存/载**:把写盘工作**摊给所有 rank**(而非只 rank 0),每 rank 写大致等量 → checkpoint 大幅提速;天然跳过 DP 副本冗余 |
| **async** | `megatron/core/dist_checkpointing/strategies/filesystem_async.py`、`megatron/core/dist_checkpointing/strategies/async_utils.py` | **异步存档**:`save` 把数据交给后台线程写盘,主训练流不阻塞 |
| **nvrx** | `megatron/core/dist_checkpointing/strategies/nvrx.py` | NVIDIA Resiliency 扩展集成 —— 本地/内存级 checkpoint,从瞬时故障快速重启 |

`async` 与 `fully-parallel` 是两个关键性能特性:前者把存档延迟**藏进计算**,后者把存档 I/O **摊到所有卡**。大模型 checkpoint 动辄 TB 级,这两者让"每隔 N 步存一次"不至于拖垮吞吐。

> Megatron-FSDP 另用 `fsdp_dtensor` 格式(基于 DTensor,见 [[36_megatron_fsdp_analysis]]——原 `27_megatron_tp_fsdp_resharding_supplements_analysis.md` §3 已于 2026-08-28 归一到该页)。

---

## 8. 校验(`megatron/core/dist_checkpointing/validation.py`)

`validate_access_integrity` 开关（实际执行者是 `validate_sharding_integrity`，`megatron/core/dist_checkpointing/validation.py:369`；每个 key 的具体校验在 `:412` 的 `_validate_sharding_for_key` 与 `:454` 的 `_compute_shards_access`）检查 `sharded_state_dict` 的**完整性与一致性**:把所有 rank 的 `ShardedTensor` 元数据汇总,验证每个全局张量 ——
- **无缺口**:全局张量的每个元素都被某个非副本片覆盖。
- **无重叠**:不同 rank 的非副本片不重叠。
- 形状、`replica_id` 自洽(`allow_shape_mismatch` 的张量放宽)。

这一步挡住"并行配置算错 offset 导致存了个残缺/重叠的 checkpoint"。

---

## 9. 约束

**9.1 `ShardedTensor` 的元数据硬约束。**
`validate_metadata_integrity` 在 `__post_init__` 里逐条校验,违反即 `CheckpointingException`(`megatron/core/dist_checkpointing/mapping.py:93-134`):`data.dtype` 必须等于 `dtype`(`:107-110`);非扁平时 `data.shape` 必须等于 `local_shape`(`:111-114`);`global_offset` 与 `global_shape` 维数必须相等(`:116-119`);`len(local_shape) + prepend_axis_num` 必须等于 `len(global_shape)`(`:120-124`);每个轴上 `global_offset` 必须能被 `local_shape` 整除(`:126-131`)—— 最后一条正是"规则网格"这个前提的落点。
`allow_shape_mismatch=True` 是唯一的放宽口子,docstring 把用途限定为「representing tensors with flexible shape, e.g. padded」(`:72-75`)。

**9.2 `flattened_range` 已在裸 `ShardedTensor` 上被禁用。**
基线 `71092579` 下,一个带 `flattened_range` 的 `ShardedTensor` **直接抛错**:`raise CheckpointingException("ShardedTensor.flattened_range is not supported.")`(`megatron/core/dist_checkpointing/mapping.py:133-134`);`from_rank_offsets` 也拒绝该参数,并把调用者指向一个**在新基线下已不存在**的 `from_rank_offsets_flat`(`:214-217`)。这条路径由 #2126(`5ab481cb4`,2025-12-11,commit message 即「Remove flattened_range code paths for distributed optimizer checkpointing」)整体删除。
该字段只在 `ShardedTensorFactory` 上继续有效(`:469`),分布式优化器经 `replace(sharded_metadata, flattened_range=item_slice, ...)` 落在 factory 上,并当场 `validate_metadata_integrity()`(`megatron/core/optimizer/distrib_optimizer.py:2095-2105`)。
→ **§3 里"`flattened_range` 用于 DDP 扁平梯度 buffer / 分布式优化器扁平状态"的描述对应 #2126 之前的形态**;在新基线读这条时请按上面两个 locator 理解:走 factory,不走裸 `ShardedTensor`。

**9.3 规则网格假设。**
把 MCore `ShardedTensor` 翻成 PyT 类型的函数明写前提:「this function assumes regular (grid) sharding of the MCore ShardedTensor. The only local irregularities could be introduced with a `flattened_range` attribute」(`megatron/core/dist_checkpointing/strategies/torch.py:147-148`),枚举全局分片时再次标注「NOTE: here we assume a regular grid of shards」(`:202`)。N-D 扁平张量还要**改写全局形状**才能高效存,docstring 自陈这会带来后果:「This will need special handling while resharding.」(`:155-159`)。

**9.4 校验的代价与盲区。**
`validate_sharding_integrity` 用 `torch.distributed.all_gather_object` 收齐**所有 rank 的分片元数据**,然后**只由 global rank 0** 做检查(`megatron/core/dist_checkpointing/validation.py:374-375`、`:393-394`);缺口/重叠汇总成一条 `CheckpointingException` 抛出(`:407-409`)。也就是说:校验的元数据量随 world size 线性增长,检查本身单点串行。
可选的 `verify_integrity` 更贵 —— docstring 自陈「Adds I/O overhead proportional to the total checkpoint size (one extra read pass over all files on rank 0)」(`megatron/core/dist_checkpointing/serialization.py:387-391`)。

**9.5 fully-parallel 的前提与代价。**
存:wrapper 假设「setting `replica_id` to 0 will make the underlying strategy do the saving on current rank」(`megatron/core/dist_checkpointing/strategies/fully_parallel.py:53-55`);`do_cache_distribution` 只在**每次调用的 state dict 结构完全相同**时才能开(`:66-68`)。
载:第(1)步(元数据)与第(4)步(实际数据)都要**跨节点通信**(`:203-204`);parallelization group 大小 ≤ 1 时整个 wrapper 退化为基础策略、直接透传(`:226-227`)。

**9.6 故意不做的事。**
- **目录非空不报错**:`save` 只打一条 WARNING 就继续覆盖(`megatron/core/dist_checkpointing/serialization.py:407-411`,理由见 §2 ④)—— 框架**不保证**不会覆盖一个残档。
- **`LocalNonpersistentObject` 一律丢弃**:`save` 第②步直接抽出并丢掉(§6.1),这类对象加载时必须由 rank 本地重算。
- **不做 GPU↔GPU 实时搬权重**:那是 resharding 的职责,与本页是两件事(§1.1)。

---

## 10. 发展趋势

四条都锚在基线 `71092579` 的实读位置,并**标为推断** —— 它们是源码里在途的清理与迁移,不是路线声明。

**① MCore 自研的异步存档正在让位给 NVRx。**
`save` 的 `async_strategy` 形参默认值已经是 `"nvrx"`(`megatron/core/dist_checkpointing/serialization.py:342`);选 `"mcore"` 时打一次性警告:「MCore's async save is deprecated and will be removed in the future releases. Please, use NVRx async solution by setting `async_strategy` to `nvrx`.」(`megatron/core/dist_checkpointing/strategies/torch.py:672-679`)。
→ **推断**:§7 表里的 `async` 一行会收敛到 `strategies/nvrx.py`,自研的 `filesystem_async.py` / `async_utils.py` 退居兼容。源码只说 deprecated,没有给移除版本。

**② `common.pt` 这一步正在被 DCP 吸收。**
`save_common` / `load_common` 都在函数体第一行打弃用警告,并给出理由:「`torch_dist` now handles all non-tensor data as part of default PyTorch DCP behavior.」(`megatron/core/dist_checkpointing/strategies/common.py:23-26`、`:46-49`)。
→ **推断**:§6.1 的第④步(rank 0 单独写 `common.pt`)会消失,非张量数据整体交给 DCP。

**③ 并行加载的分片交换留着明确的优化位。**
`exchange_utils.py` 在两处相同的分支上写着「TODO: we can employ some optimizations even for `len(shard_to_ranks) > 1` case, e.g. P2P exchange. Currently handling this case saves most of the work though.」(`megatron/core/dist_checkpointing/exchange_utils.py:314-316`、`:500`),与 §2 ⑤ 里 fully-parallel load 自陈的"近似"正好对上。
→ **推断**:all-gather 会逐步被点对点交换替换 —— 这是 §9.5 那条跨节点通信成本压出来的方向。

**④ 三处为绕开上游 bug / 未完成能力而留的临时代码等着删。**
`exchange_utils.py` 顶部写「TODO: remove TE references once the TE bug is fixed」(`:19`);FP8 交换处进一步说明「Because of a TE bug, we have to exchange a nominal dtype instead of FP8 … TODO: remove it once the bug is fixed」(`:345-348`,另一处同样在 `:515`)。异步写盘侧留着「TODO: For persistent worker, this work should be changed to move the cpu tensor to shared_memory.」(`megatron/core/dist_checkpointing/strategies/filesystem_async.py:152-153`)。
→ **推断**:这三处是明确在途的清理项,不是设计意图 —— 引用本页时不要把它们当成稳定行为。

---

## 11. 小结

- **要解决的问题**:模型被 TP×PP×EP×DP 切散,朴素存档与并行配置死绑,不能换布局/换集群续训。
- **核心抽象 `ShardedTensor`**:用 `(global_shape, global_offset, local_shape)` 描述"本 rank 这一片在全局张量的哪个位置",把并行布局信息编码进 checkpoint;`replica_id` 去 DP 冗余、`prepend_axis_num` 表达专家轴、`flattened_range` 对接扁平 buffer。
- **`sharded_state_dict()`**:每个 Megatron 模块自我声明分片映射,驱动 save/load。
- **save/load**:存成并行无关的全局张量;加载时按**当前(可不同的)布局**重新切片 —— 这就是"`TP8×PP4` 存、`TP2×PP1` 载"成立的原理。
- **策略**:`torch_dist` 默认;`fully-parallel`(I/O 摊到所有卡)+ `async`(后台写、不阻塞训练)是两个关键提速特性;`nvrx` 做快速故障重启。
- **校验**:`megatron/core/dist_checkpointing/validation.py` 保证全局张量被分片无缺口、无重叠地覆盖。
- 与 resharding 的区别:dist_checkpointing 是**磁盘**存档(续训),resharding 是 GPU↔GPU **实时**搬权重(RL 训推)。

---

*生成依据:`Megatron-LM` `dev` 分支 `85902ef599ea4eb06ada7567a479c524b605767a`(2026-09-01;由 `71092579` 重定基线而来,更早一次为 2026-08-28 由 `ee3f1ff` 推进)。源码行号以该 commit 为准；2026-08-28 由 `ee3f1ff` 重定基线。配套文档:`17_megatron_parallelism_orchestration_analysis.md`、`16_megatron_distributed_optimizer_analysis.md`、`27_megatron_tp_fsdp_resharding_supplements_analysis.md`。*

---

## 配置契约：`CheckpointConfig`

本页正文讲的是 **core 侧的存档机制**——`ShardedTensor`、并行无关格式、`fully-parallel`/`async` 策略。但用户实际操作存档的面不在 `megatron/core/dist_checkpointing`，而在 `CheckpointConfig`：它是 `megatron/training/config/` 下**最大的一个 config 类**，经 [[41_megatron_config_surface_analysis]] §2 的 `ArgumentGroupFactory` 自动转成 CLI（`megatron/training/arguments.py:3943`）。

**下表的类型、默认值与说明直接取自 `CheckpointConfig` 类体**（行号为 `training_config.py` 内行号），与生成 CLI 的是同一份声明。读法：先按**存哪儿**（`save` / `load` / `pretrained_checkpoint`）、**多久存一次**（`save_interval` 与 non-persistent 那一组）、**存什么**（`no_save_optim` / `no_save_rng` 及其 load 对偶）、**怎么存**（`ckpt_format`、`async_save`、fully-parallel 一组）四类分组读，比按字母序逐条看要快得多。


### `CheckpointConfig`（`megatron/training/config/training_config.py`，45 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `save_params_interval` | `int \| None` | `None` | Number of iterations between param.name->param.data mapping saves. | `:427` |
| `save_wgrads_interval` | `int \| None` | `None` | Number of iterations between wgrad (main_grad) saves. | `:436` |
| `save_retain_interval` | `int \| None` | `None` | Number of iterations between retained checkpoints (other checkpoints except the last checkpoint are automatically deleted). | `:442` |
| `most_recent_k` | `int \| None` | `-1` | Number of latest checkpoint to be saved. | `:447` |
| `save_optim` | `bool` | `True` | Do not save current optimizer. | `:450` |
| `save_rng` | `bool` | `True` | Do not save current rng state. | `:453` |
| `load_optim` | `bool` | `True` | Do not load optimizer when loading checkpoint. | `:459` |
| `load_main_params_from_ckpt` | `bool` | `False` | Load main parameters from checkpoint. When loading a model from a checkpoint without loading the optimizer, the model parameters are updated but for fp16 opt… | `:462` |
| `load_rng` | `bool` | `True` | Do not load rng state when loading checkpoint. | `:468` |
| `non_persistent_save_interval` | `int \| None` | `None` | Number of iterations between non-persistent saves. | `:471` |
| `non_persistent_ckpt_type` | `Literal['global', 'local', 'in_memory'] \| None` | `None` | Type of non-persistent model checkpoints. "global" - Saved as a standard checkpoint (e.g., on Lustre) with old checkpoints being removed. "local" - [TBD] Eac… | `:474` |
| `non_persistent_global_ckpt_dir` | `str \| None` | `None` | Directory containing global non-persistent model checkpoints. | `:481` |
| `non_persistent_local_ckpt_dir` | `str \| None` | `None` | Directory containing local non-persistent model checkpoints. | `:484` |
| `non_persistent_local_ckpt_algo` | `Literal['fully_parallel', 'atomic']` | `'fully_parallel'` | Algorithm for local non-persistent checkpointing. | `:487` |
| `finetune` | `bool` | `False` | Load model for finetuning. Do not load optimizer or rng state from checkpoint and set iteration to 0. Assumed when loading a release checkpoint. | `:490` |
| `pretrained_checkpoint` | `str \| None` | `None` | Directory containing a pretrained model checkpoint for finetuning. | `:494` |
| `ckpt_step` | `int \| None` | `None` | Checkpoint step to load model from. | `:497` |
| `use_checkpoint_args` | `bool` | `False` | Override model-related command-line arguments with arguments from checkpoint | `:500` |
| `use_mp_args_from_checkpoint_args` | `bool` | `False` | Copy model parallelism command-line arguments from checkpoint | `:503` |
| `use_tokenizer_model_from_checkpoint_args` | `bool` | `True` | If set, do not use tokenizer model path from checkpoint | `:506` |
| `exit_on_missing_checkpoint` | `bool` | `False` | If 'load' is set, but checkpoint is not found (e.g., path typo), then exit instead of random initialization. | `:509` |
| `auto_detect_ckpt_format` | `bool` | `False` | Determine if the checkpoint format is in legacy or distributed format. If False, expects distributed checkpoint iff args.ckpt_format != "torch". Might slow d… | `:519` |
| `ckpt_convert_format` | `Literal['torch', 'torch_dist'] \| None` | `None` | Checkpoint format for conversion. | `:525` |
| `ckpt_convert_save` | `str \| None` | `None` | Save directory for converted checkpoint. | `:528` |
| `ckpt_convert_update_legacy_dist_opt_format` | `bool` | `False` | When loading a checkpoint, update the legacy format for the distributed optimizer, which previously used a merged param/grad buffer and a different bucket ma… | `:531` |
| `fully_parallel_save` | `bool` | `field(default=True, metadata={'argpar…` | Disable applying full save parallelization across DP for distributed checkpoints. Depending on ckpt format might decrease the number of files in the checkpoi… | `:537` |
| `async_save` | `bool` | `False` | Apply async checkpointing save. Currently works only with `torch_dist` distributed checkpoint format. | `:550` |
| `use_persistent_ckpt_worker` | `bool` | `False` | Use a persistent background worker for async checkpoint saves. When enabled, creates a dedicated worker thread/process for handling async saves. When disable… | `:556` |
| `async_ckpt_cpu_priority` | `int` | `10` | CPU nice value target (0-19, higher = lower priority) for the async checkpoint writer process. If it exceeds 19, it will be set to 19. If the current nice va… | `:561` |
| `async_ckpt_io_priority` | `Optional[int]` | `3` | I/O scheduling class (0-3, 3=idle) for the async checkpoint writer process. | `:566` |
| `async_ckpt_use_cpu_shm` | `bool` | `False` | Copy GPU tensors to CPU shared-memory in the training process before handing off to the async checkpoint worker. Avoids CUDA IPC / NVLink fabric handles in t… | `:569` |
| `fully_parallel_load` | `bool` | `field(default=False, metadata={'argpa…` | Apply full load parallelization across DP for distributed checkpoints. | `:575` |
| `ckpt_fully_parallel_load_exchange_algo` | `Literal['broadcast', 'gather_rounds', 'gather_object']` | `'broadcast'` | Algorithm for fully parallel load of distributed checkpoints. "broadcast"(default): Broadcast the checkpoint from rank 0 to all other ranks. "gather_rounds":… | `:586` |
| `ckpt_fully_parallel_save_process_group` | `Literal['dp', 'ep_dp']` | `'dp'` | Process group for fully parallel save of distributed checkpoints. "dp"(default): Data parallel process group. "ep_dp": Expert data parallel process group. | `:595` |
| `ckpt_fully_parallel_load_process_group` | `Literal['dp', 'ep_dp']` | `'dp'` | Process group for fully parallel load of distributed checkpoints. "dp"(default): Data parallel process group. "ep_dp": Expert data parallel process group. | `:601` |
| `ckpt_assume_constant_structure` | `bool` | `False` | Assume the checkpoint structure is constant across saves to enable optimizations. | `:607` |
| `ckpt_load_validate_sharding_integrity` | `bool` | `True` | Whether to validate sharding access integrity when loading a distributed checkpoint. When True (default), each tensor shard is checked to be accessed exactly… | `:610` |
| `strict_fsdp_dtensor_load` | `bool` | `True` | Whether to enforce strict loading for FSDP DTensor checkpoints. When False, allows partial loading. | `:615` |
| `dist_ckpt_strictness` | `Literal['assume_ok_unexpected', 'log_unexpected', 'log_all', 'raise_unexpected', 'raise_all', 'return_unexpected', 'return_all', 'ignore_all']` | `'assume_ok_unexpected'` | Determine handling of key mismatch during checkpoint load. Check StrictHandling docs for flags meaning. NOTE: This flag controls only distributed checkpoint … | `:618` |
| `dist_ckpt_save_pre_mcore_014` | `bool` | `False` | Revert checkpointing simplifications introduced in Megatron-Core v0.14. This option affects only checkpoint saving format and will be removed soon (checkpoin… | `:631` |
| `dist_ckpt_optim_fully_reshardable` | `bool` | `False` | Make optimizer distributed checkpoint fully reshardable (TP/PP/EP/DP) as opposed to plain DP reshardability. | `:636` |
| `distrib_optim_fully_reshardable_mem_efficient` | `bool` | `False` | During distributed optimizer checkpoint save and load tries to use as little memory as possible by using Gloo (instead of NCCL) and only one rank for saving.… | `:639` |
| `save_tokenizer_assets` | `bool` | `True` | Save tokenizer files to checkpoint directory. When enabled, saves all tokenizer artifacts (vocab files, special tokens, tokenizer config) to make checkpoints… | `:644` |
| `replication_jump` | `int \| None` | `None` | Specifies `J`, the spacing between ranks storing replicas of a given rank's data. Replicas for rank `n` may be on ranks `n+J`, `n+2J`, ..., or `n-J`, `n-2J`,… | `:652` |
| `replication_factor` | `int` | `2` | Number of machines storing the replica of a given rank's data. | `:657` |

> 该类共 55 个字段，本表收 45 项；其余 10 项已在别处归属：主要归 本页他处 7 项、[[43_megatron_job_resilience_analysis]] 3 项（完整归属见 `docs/coverage/megatron-lm.yaml`）。

> **与正文的接缝**：`ckpt_fully_parallel_save` / `ckpt_fully_parallel_load` 对应本页讲的 `FullyParallelSave/LoadStrategyWrapper`；`async_save` 对应异步策略与 finalize 流程；non-persistent 那一组则是本页 core 机制之外、由训练侧编排的一层（详见 [[40_megatron_feature_tree_analysis]] §4 标记的「训练侧存档编排」待补项）。

---

## 配置契约：异构存档补充

本页前一节给了 `CheckpointConfig`。本节补 `TransformerConfig` 里一个与异构模型存档相关、此前零提及的字段。





### `TransformerConfig`（`megatron/core/transformer/transformer_config.py`，1 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `hetereogenous_dist_checkpoint` | `bool` | `False` | Whether to use heterogenous layers in distributed checkpoint. | `:1438` |

> 该类共 266 个字段，本表收 1 项；其余 265 项已在别处归属：主要归 [[10_megatron_model_structure_analysis]] 92 项、[[14_megatron_ep_analysis]] 38 项、[[23_megatron_precision_cudagraph_fusion_analysis]] 38 项、[[21_megatron_fusion_operators_analysis]] 26 项，另散见 19 页（完整归属见 `docs/coverage/megatron-lm.yaml`）。

## Related Pages

- [[17_megatron_parallelism_orchestration_analysis]] · [[16_megatron_distributed_optimizer_analysis]] · [[27_megatron_tp_fsdp_resharding_supplements_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]


