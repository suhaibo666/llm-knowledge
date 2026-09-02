---
title: "Megatron-LM 序列打包与动态 CP 的统一流水线深度解析"
---

# Megatron-LM 序列打包与动态 CP 的统一流水线深度解析

> **源码基线**:`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`(`dev`,2026-09-01)
> **重定基线**：2026-09-01 由 `71092579`（2026-08-27）推进，跨 7 个提交；该增量只触及 20 个 `megatron/` 文件，本页 `path:line` 引用所涉源文件均不在其中，故无行号漂移，无需逐条重核。
> **重定基线**:2026-08-28 由 `ee3f1ffa…`(2026-05-19)推进,跨 578 个提交;本页全部 `path:line` 形式的引用已在新基线下逐条重核;**代码块内被点名的符号与不带行号的裸路径不在该次扫描口径内**,已知漏网处已于 2026-08-28 单独更正。
> 核心文件:`megatron/core/datasets/data_schedule.py`(1166 行)、`megatron/core/datasets/data_schedule_utils.py`(936 行);`megatron/core/packed_seq_params.py`;`megatron/core/pipeline_parallel/hybrid_cp_schedule.py`
> 配套阅读:`11_megatron_dataset_analysis.md` §5、`15_megatron_pp_schedulers_analysis.md` §8.1、`13_megatron_cp_analysis.md`
> **叙事顺序**：本页按五拍组织——背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势。
> **最近更新**：2026-08-28。按五拍重排章节顺序；机制正文与既有引用未改。
> 定位:**勘误 + 补全**。`11_megatron_dataset_analysis.md` 把序列打包当主角、`15_megatron_pp_schedulers_analysis.md` §8.1(原 `26_megatron_pp_supplements_analysis.md` §3,2026-08-01 合并入 15_)把动态 CP 当独立特性,分两处讲。本文说清楚:**在代码里它俩是同一条流水线、同一个类继承链** —— 动态 CP 不是和打包并列协作的特性,而是打包调度器的一个子类。

---

## 1. 背景：打包与动态 CP 被当成两个特性讲,可代码里它们是一个继承链

`megatron/core/datasets/data_schedule.py` 里序列打包的类层次:

```
BasePackingScheduler                    抽象基类:get_groups_and_subsamples() + run()
   └── DpBalancedScheduler              打包 + DP 均衡;is_dynamic_cp = False
                                        max_seq_len_all_ranks = max_seqlen_per_dp_cp_rank × cp_size
          └── DefaultDynamicCPScheduler  is_dynamic_cp = True
                                        只重写 get_groups_and_subsamples(),其余全继承
```

`config.sequence_packing_scheduler` 选哪个(`scheduler_map`:`"dp_balanced"` / `"default_dynamic_cp"`)。关键事实:

> **`DefaultDynamicCPScheduler` 是 `DpBalancedScheduler` 的子类,而 `DpBalancedScheduler` 是打包调度器。** 也就是说"动态 CP"本身就是一个序列打包调度器 —— 它把 `DpBalancedScheduler` 的整条 `run()` 流水线**原样继承**,只改了其中一步(怎么给样本分组、每个样本分几张 CP 卡)。

所以正确的心智模型不是"打包 + 动态 CP 两个东西配合",而是"**序列打包是框架,动态 CP 是这个框架的 `is_dynamic_cp=True` 档**"。

---

## 2. 为什么这么设计：把"谁来算这条样本"做成一次全局分组 + 一次数据重排,其余全部共用

打包与动态 CP 解的是同一件事:**样本是变长的,而每个 DP×CP rank 的算力是等量的**。源码给出的答案是"一条流水线 + 一个可换的分组步",并在最近半年里两次把其中的通信形态推翻重做。下面四条源码或提交历史陈述了理由;第五、六条源码沉默,由本页重建并整段标为推断。

**① 调度必须先跨 DP all-gather 全局序列长度 —— 理由源码自陈。**
`get_batch_and_global_seqlens` 的 docstring 写:「Each DP rank loads the same number of sequences, so we need to gather the sequence lengths from all ranks then we can schedule the sequences into groups.」(`megatron/core/datasets/data_schedule_utils.py:543-546`)。实现是先 all-gather 各 rank 的子样本个数、按最大值 padding 后再 gather 长度(`:180-196`)。
→ 判据:**分组是一次全局决策**,每个 rank 都必须看到整个 global batch 的长度分布,否则算不出均衡分桶。

**② reroute 从 all-to-all 换成 DP 组 all-gather —— 被否掉的替代写在历史里。**
新基线的 docstring 逐条给出理由:「Each CP lane gathers the samples from its DP group, then keeps only the samples assigned to its DPxCP rank. **Gathering within `dp_group` avoids collecting the identical input held by every CP sibling and avoids the fully connected P2P transport created by NCCL all-to-all.**」(`megatron/core/datasets/data_schedule_utils.py:364-367`)。
**被否掉的替代就是它自己的上一版**:提交 `d48bd6be0`(2026-08-21,commit message 即「[dev] Replace DP balance all-to-all rerouting with all-gather (#6378)」)之前,同一函数的 docstring 写的是「For each key in the batch dict, we perform an all-to-all communication to transfer the data to the correct ranks.」(`d48bd6be0^:megatron/core/datasets/data_schedule_utils.py:361-362`),实现落在 `torch.distributed.all_to_all_single`(`d48bd6be0^:megatron/core/datasets/data_schedule_utils.py:453`)。
→ 决定取舍的两条判据都写在 docstring 里:**CP 兄弟 rank 持有完全相同的输入**(所以在 DP 组内 gather 一次就够,不必让每个 CP rank 各收一份),以及 **NCCL all-to-all 会建出全连接的 P2P 传输**。
同一段还交代第三条取舍 —— gather **按 key 逐个发起**:「This pays the fixed collective latency once per key, but bounds temporary memory to one global field at a time. Selected slices are cloned before advancing to the next key so the full gather buffer can be freed.」(`:374-377`)—— 拿"每 key 一次固定集合延迟"换"峰值临时显存只有一个全局字段"。

**③ 不再做 PP 组广播 —— 被否掉的替代是 `broadcast_to_pp_group`。**
`DpBalancedScheduler.run` 的 docstring 明写「Note: There is no PP-group broadcast. In packed-sequence mode is_dataset_built_on_rank returns True for every PP stage on TP rank 0」(`megatron/core/datasets/data_schedule.py:245-246`),函数体里再解释一遍「every stage independently fetches data and computes the global seqlen stats」(`:300-305`)。
**理由写在入口判据处**:`pretrain_gpt.py:410` 的 `is_dataset_built_on_rank` 对打包 / SBHD 校验路径直接返回 True,注释给出原因 ——「Packed THD and SBHD validation both need padding metadata on every pipeline stage so each MoE layer excludes physical padding」(`:419-421`)。
**被否掉的替代**:提交 `959a542a1`(#4226「Minor improvements for Dynamic-cp」)把 `broadcast_to_pp_group` 整体删除(详见 §3.1 的 `[!contradiction]`)。
配套补偿是新增的第 3 步 —— 按本 PP stage 裁掉用不到的数据字段,注释自陈目的是「to avoid unnecessary rerouting communication」(`megatron/core/datasets/data_schedule.py:320-324`)。

**④ 动态 CP 的分桶换成 packing-aware 版本 —— 被否掉的替代是 legacy DCP 调度器。**
`next_hdp_group_packing_aware` 的 docstring 以"与旧版的差异"开篇:「This differs from **the legacy DCP scheduler** in two ways: 1. Short sequences may use a larger CP group than their minimum required CP size when that lowers the critical-path rank workload. 2. Candidate placements are bounded by `tall * max_seq_len_per_rank`, the per-rank workload upper bound for packing sequences no longer than the local tallest sequence in the microbatch.」(`megatron/core/datasets/data_schedule_utils.py:598-605`),由提交 `d2e7ec5b8`(#5154「Improve default dynamic CP packing scheduler」)引入。
两条公式源码给得很直白:`workload(seq_len, cp_size) = (seq_len * seq_len) / cp_size`(`:623-624`);`dcp_gpus_needed` 自陈是「Number of GPUs needed, rounded up to the next power of 2, lower-bounded by min_cp_size」(`:933-936`)。
→ 判据由 docstring 点名:**压低关键路径 rank 的工作量**("lowers the critical-path rank workload")。

> [!note] 推断
> 两处判断源码沉默,由本页承担 ——
> **⑤ "动态 CP 为什么做成打包调度器的子类,而不是并列特性"**:源码只用类继承(`megatron/core/datasets/data_schedule.py:407` 的 `DefaultDynamicCPScheduler(DpBalancedScheduler)`,只重写 `__init__` 与 `get_groups_and_subsamples`,见 `:412`、`:419`)和一张名字表(`scheduler_map`,`:450-453`)表达这件事,**从未说明为什么选继承而不是组合**。本页 §1 / §11 的"一个继承链,不是两个特性"是对这个代码形状的读法。
> **⑥ "`seq²/cp` 为什么是合理的工作量代理"**:源码只写了这个表达式(`megatron/core/datasets/data_schedule_utils.py:623-624`),没有解释它对应 attention 的 `O(S²)` 复杂度被 CP 摊分。§4.2 里"用 `seq²/cp` 估 attention 负载"那句是本页的解释,不是源码陈述。
> 要引用这两条,请回到上面括注的 locator,不要引用本段推断。

---

## 3. 统一的 `run()` 9 步流水线

**先说新基线下的形态。** `DpBalancedScheduler.run` 的 docstring 在 `71092579` 下列出的九步是(`megatron/core/datasets/data_schedule.py:234-243`):

```
① 取 microbatch + 跨 DP all-gather 全局 seqlen         # Step 1  data_schedule.py:309
② 校验 required sample keys                           # Step 2  data_schedule.py:314
③ 按本 PP stage 裁掉用不到的数据字段  ← 新增           # Step 3  data_schedule.py:320
④ get_groups_and_subsamples  ★唯一分叉点★              # Step 4  data_schedule.py:335
⑤ reroute_samples_to_dcp_ranks(DP 组 all-gather)      # Step 5  data_schedule.py:348
⑥ build_packed_microbatches                           # Step 6  data_schedule.py:362
⑦ 算 FLOPs 信息(Σseqlen、Σseqlen²)                    # Step 7  data_schedule.py:367
⑧ 跨 TP 组广播标量 broadcast_scalars                             data_schedule.py:380-392
⑨ create_data_iterator(VPP 时按 vpp_needs_data 产出列表)         data_schedule.py:394-397
```

代码里的 `# Step N` 注释只写到 Step 8,且那条注释(`:394`)把"跨 TP 组广播"和"建 data_iterator"合并成一句;上表按 docstring 的九步拆开,VPP 的 `vpp_needs_data` 在 `:277-298` 先算好再传进 `create_data_iterator`。
**没有 PP 组广播**:docstring 紧接着写 *There is no PP-group broadcast. In packed-sequence mode is_dataset_built_on_rank returns True for every PP stage on TP rank 0*(`:245-246`)。
本页下文的 §4 / §5 / §6 仍沿用**旧基线的步号**("第③步分组、第④步 reroute、第⑤步打包"),对应新步号是**④ / ⑤ / ⑥** —— 差一位,来自新增的第③步。**"唯一分叉 = `get_groups_and_subsamples`"这条核心结论在两套编号下都成立。**

### 3.1 旧基线的九步表述与第⑦步的删除

`DpBalancedScheduler.run()`(`megatron/core/datasets/data_schedule.py:220`;抽象签名在 `BasePackingScheduler.run` `:113`)定义了完整流水线,`DefaultDynamicCPScheduler` **不重写它**,直接继承:

```
入口 wrap_data_iterator(megatron/core/datasets/data_schedule.py:486)
  按 config.sequence_packing_scheduler 实例化调度器,调 .run()
        │
        ▼
run() 九步:
  ① get_batch_and_global_seqlens   取 microbatch,跨 DP all-gather 每个样本的全局序列长度
  ② 校验 required sample keys      (tokens/labels/loss_mask/position_ids/original_seq_len/...)
  ③ get_groups_and_subsamples      ★唯一分叉点★ 把样本编成组(决定打包方式 & CP 数)
  ④ reroute_samples_to_dcp_ranks   all-to-all,把每个子样本搬到"将要计算它"的 rank
  ⑤ build_packed_microbatches      在本 rank 拼出 THD 打包 buffer(传入 is_dynamic_cp)
  ⑥ 算 FLOPs 信息                  Σseqlen、Σseqlen²(给吞吐统计)
  ⑦ broadcast_to_pp_group          PP 中间 stage 拿到元数据
  ⑧ broadcast_scalars              非 TP-0 rank 拿到标量
  ⑨ create_data_iterator           产出新 data_iterator(VPP 时为 list)
```

**九步里有八步(①②④⑤⑥⑦⑧⑨)对两种调度器完全相同**。唯一不同的是 **第③步 `get_groups_and_subsamples`** —— 这就是"打包"和"动态 CP"真正分叉的地方,也是它们唯一的区别。

> [!contradiction] 上面九步中的**第⑦步 `broadcast_to_pp_group` 在基线 `71092579` 下已不存在**。
> `broadcast_to_pp_group` 由 #4226(`959a542a1`,Minor improvements for Dynamic-cp)整体删除,全仓 `git grep broadcast_to_pp_group` 在 `71092579` 为 0 命中(在旧基线 `ee3f1ff` 尚有 `megatron/core/datasets/data_schedule_utils.py:194` 的定义与 `data_schedule.py:298` 的调用)。
> `DpBalancedScheduler.run` 的 docstring 现明说 *There is no PP-group broadcast. In packed-sequence mode is_dataset_built_on_rank returns True for every PP stage on TP rank 0*(`megatron/core/datasets/data_schedule.py:245-246`)—— 打包模式下每个 PP stage 的 TP-0 rank 都自建数据集,不必再从首/末 stage 广播。
> 同时该 docstring 列出的九步已改为:①取 batch + 全局 seqlen ②校验 required keys ③**按本 PP stage 裁掉不需要的数据字段**(新增)④分组(仍是唯一分叉点)⑤reroute ⑥打包 microbatch ⑦算 FLOPs ⑧跨 TP 组广播标量 ⑨VPP(`:234-243`)。
> 本页"唯一分叉 = `get_groups_and_subsamples`"的核心结论**仍然成立**(它现在是第④步),但步序编号与第⑦步的 PP 广播需按新基线读。

---

## 4. 唯一的分叉:第③步 `get_groups_and_subsamples`

### 4.1 `DpBalancedScheduler` —— 定长 CP + 贪心打包

`megatron/core/datasets/data_schedule.py:165`(类定义 `:146`)。所有样本用**同一个固定 `cp_size`**,一个打包 microbatch(横跨该 DP 组的所有 CP rank)的容量是 `max_seq_len_all_ranks = max_seqlen_per_dp_cp_rank × cp_size`。

```python
for i in range(len(sample_id_seqlens)):
    if sum_seqlen + seqlen[i] <= max_seq_len_all_ranks:   # 还塞得下
        single_microbatch.append(i); sum_seqlen += seqlen[i]
    else:                                                  # 塞不下 → 封一个 microbatch
        packed_id_groups.append(single_microbatch)
        single_microbatch = [i]; sum_seqlen = seqlen[i]
```

**贪心 first-fit 打包**:按原顺序往一个 microbatch 里塞样本,塞满就封、开下一个。再把 microbatch 数量补齐成 `dp_size × microbatch_group_size_per_vp_stage` 的整数倍(从后面的 microbatch 挪样本)。CP 度对所有样本恒定。

### 4.2 `DefaultDynamicCPScheduler` —— 按长度分 CP + 工作量均衡

`megatron/core/datasets/data_schedule.py:419`(类定义 `:407`)。**每个样本分到的 CP 卡数随它的长度变化**:

```python
gpus_fn     = lambda seq_len: dcp_gpus_needed(seq_len, mslpr, min_cp)   # 该样本要几张 CP 卡
workload_fn = lambda seq_len, cp=None: ...                             # 工作量 ≈ seq²/cp
buckets_fn  = next_hdp_group_packing_aware(...)                        # 打包感知的均衡分桶

sample_id_seqlens = sorted(..., key=seqlen, reverse=True)              # 长样本优先
while sample_id_seqlens:
    mb, sample_id_seqlens, ... = next_hdp_group(                       # 每次形成一个均衡的 hdp 组
        sample_id_seqlens, workload_fn, total_hdp_gpus, gpus_fn, buckets_fn, ...)
    groups.append(mb)
```

- `dcp_gpus_needed`(`megatron/core/datasets/data_schedule_utils.py:933`):长样本分更多 CP 卡(向上取 2 的幂),短样本少分 —— **CP 度自适应序列长度**。
- 工作量估计:用 `seq²/cp` 估 attention 负载(`O(S²)` 除以 CP 摊分)。
- `next_hdp_group_packing_aware`(`megatron/core/datasets/data_schedule_utils.py:592`,调用点 `data_schedule.py:431`):把样本贪心打包成一个个 **hdp 组**(hybrid DP 组),使每个 DP×CP rank 的总工作量大致相等。
  > [!note] 上面这段伪代码里的 `dcp_get_total_workload` / `dcp_make_buckets_equal` / 裸 `next_hdp_group` 在基线 `71092579` 下**全域零命中**(2026-08-28 核),已按真实符号更正;它们属于更早版本的形态。
- `align_sample_id_groups`:VPP 时对齐组数。

> `megatron/core/pipeline_parallel/hybrid_cp_schedule.py` 的 `BalancedCPScheduler`(`15_megatron_pp_schedulers_analysis.md` §8.1 分析过)是同一套均衡逻辑的**类形态兄弟**;`megatron/core/datasets/data_schedule.py` 的 `DefaultDynamicCPScheduler` 实际用的是 `megatron/core/datasets/data_schedule_utils.py` 里的**函数形态** `dcp_*` + `next_hdp_group`。二者算法一致,是并行实现。**真正把打包与动态 CP 缝在一起的集成点,是 `DefaultDynamicCPScheduler`。**

### 4.3 两者对照

| | `DpBalancedScheduler` | `DefaultDynamicCPScheduler` |
|--|----------------------|------------------------------|
| `is_dynamic_cp` | False | True |
| CP 度 | 所有样本固定 `cp_size` | **每样本按长度自适应**(`dcp_gpus_needed`) |
| 第③步算法 | 贪心 first-fit 打包 | `next_hdp_group` 工作量均衡分桶 |
| 容量基准 | `max_seqlen_per_dp_cp_rank × cp_size` | `max_seqlen_per_dp_cp_rank`(每 rank) |
| ①②④⑤⑥⑦⑧⑨ | ←———— 完全相同 ————→ | |

---

## 5. 第④步 `reroute`:把样本搬到"该算它的 rank"

`reroute_samples_to_dcp_ranks`(`megatron/core/datasets/data_schedule_utils.py:358`)。这一步是打包/动态 CP 流水线**必须有**的:

- 数据集 loader 是按 **DP** 把样本分给各 rank 的(谁 load 了哪条)。
- 但第③步的调度结果是"样本 X 应该由 DCP rank `d` 计算"—— load 它的 rank 和算它的 rank **通常不是同一个**。
- 于是对 batch 里每个 key 做 **DP 组内的 all-gather**,再只保留分给本 DP×CP rank 的那些样本,从而把每个子样本的数据从"load 它的 rank"搬到"算它的 rank"(`megatron/core/datasets/data_schedule_utils.py:364-367`)。

`is_dynamic_cp` 在这里也透传 —— 动态 CP 下一个样本可能要发给多个 CP rank。

> [!contradiction] 上面这一行**此前写作**"对 batch 里每个 key 做 all-to-all",对应旧基线 `ee3f1ffa…`;在基线 `71092579` 下 **reroute 已不再用 all-to-all**,正文已按新实现更正(2026-08-28)。
> 提交 `d48bd6be0`(2026-08-21,commit message 即「[dev] Replace DP balance all-to-all rerouting with all-gather (#6378)」)把它换成 **DP 组内的 all-gather**:`d48bd6be0^` 的 docstring 还写着「For each key in the batch dict, we perform an all-to-all communication to transfer the data to the correct ranks.」(`d48bd6be0^:megatron/core/datasets/data_schedule_utils.py:361-362`,`torch.distributed.all_to_all_single` 在 `:453`);新基线改成「Each CP lane gathers the samples from its DP group, then keeps only the samples assigned to its DPxCP rank.」(`megatron/core/datasets/data_schedule_utils.py:364-365`)。
> **"把样本搬到该算它的 rank"这个语义没变**,变的是用哪种集合通信实现;换路的理由见 §2 ②。本页 §11 小结里"用 all-to-all"的同一说法,同样按此更正。

---

## 6. 第⑤步:`build_packed_microbatches` + `PackedSeqParams` + CP 切片

`build_packed_microbatches`(`megatron/core/datasets/data_schedule_utils.py:475`)在本 rank 把分到的子样本拼成 **THD 打包 buffer**,产出 `PackedSeqParams`(`megatron/core/packed_seq_params.py`,见 `11_megatron_dataset_analysis.md` §5.2):`cu_seqlens` 标出每条子序列边界,`qkv_format='thd'`。

动态 CP 的衔接点:
- `PackedSeqParams` 的 **`local_cp_size` / `cp_group`** —— 每个打包 buffer **可以有自己的 CP 度**(因为不同 microbatch 的样本长度不同 → 动态 CP 给的 cp_size 不同)。
- `get_cp_slice_for_thd`(`megatron/core/datasets/data_schedule_utils.py:26`):一个打包 THD buffer 若 `cp_size > 1`,要再沿序列切给 CP rank —— 且按 zigzag 均衡(因果掩码,见 `13_megatron_cp_analysis.md`)。
- `get_batch_on_this_rank_for_sequence_packing(..., dynamic_cp=True)`(`megatron/core/datasets/data_schedule.py:564`)取数时读 `batch['local_cp_size']`,据此决定本 buffer 的 CP 切分。

所以:**打包决定"buffer 里装哪几条序列",动态 CP 决定"这个 buffer 用几张卡的 CP、怎么切" —— 二者在同一个 `PackedSeqParams` 对象上汇合**。

---

## 6.5 NEW:varlen 数据源 + get_batch 统一(dev@232c478d4)

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> 这条统一 `run()` 流水线的**上游入口**和**下游取数**都有更新,但九步骨架与"唯一分叉=第③步"的结论不变:
>
> - **第①步 `_unpack_batch` 支持双输入**(`megatron/core/datasets/data_schedule_utils.py:105`,#4832):除了 `SFTDataset` 那种"一条样本里 `cu_seqlens` 拼了多条子序列、需切开"的**预打包**形态,新增 `VarlenDataset`(`--use-varlen-dataset`)这种"每 index 已是单条子样本、自带 `padded_seq_len`"的**已拆开**形态 —— 后者只需丢掉 collate_fn 多加的 batch 维、缺 `original_seq_len` 时从 `padded_seq_len` 补,再走同一条 ②→⑨。`VarlenDataset` 细节见 `11_megatron_dataset_analysis.md` §5.4。
> - **下游 get_batch 统一 + SFT THD 支持 PP**(#4103):`get_batch_on_this_*` 取数函数收敛进 `megatron/core/utils.py` —— `get_batch_on_this_tp_rank`(`:2052`,**长度前缀协议**广播变长的 `cu_seqlens`,并在动态 CP 下广播 `local_cp_size` / `hybrid_cp_seq_length`)、`get_thd_batch_on_this_cp_rank`(`:2608`,对应第⑤步在本 rank 的 THD 切片)。SFT 的 THD 打包现可与 PP 共用(原理由第⑦步 `broadcast_to_pp_group` 承担;该函数已在 `71092579` 删除,改由"每个 PP stage 的 TP-0 rank 各自建数据集"实现,见 §3.1 的 `[!contradiction]`)。
> - **第⑥步 seqlen 统计修正**:`train_step` 现保留 seqlen 统计(commit 95654c956);`sequence_packing_scheduler` 非空时的 TFLOPs 计算修正(#5342)—— 对应本页第⑥步 `Σseqlen / Σseqlen²` 的吞吐统计。

---

## 7. 一张图看清统一关系

```
                      config.sequence_packing_scheduler
                                  │
                  ┌───────────────┴────────────────┐
            "dp_balanced"                  "default_dynamic_cp"
            DpBalancedScheduler            DefaultDynamicCPScheduler
                  │                                │
                  └────────► 共享同一个 run() ◄─────┘
                                  │
   ①取数+全局seqlen ②校验 ┌─③ 分组(唯一分叉)─┐ ④reroute ⑤打包buffer ⑥FLOPs ⑦⑧广播 ⑨迭代器(基线 `71092579` 下第⑦步的 PP 广播已删除,见 §3.1 的 [!contradiction])
                          │ 定长CP贪心打包      │
                          │ vs                 │
                          │ 按长度分CP+均衡分桶 │
                          └────────────────────┘
                                  │
                          PackedSeqParams(cu_seqlens + 可选 local_cp_size/cp_group)
                                  │
                          get_cp_slice_for_thd → 打包 buffer 再按 CP 切(zigzag)
```

序列打包是骨架;动态 CP 是骨架上第③步的一个变体 + `PackedSeqParams` 里多带 `local_cp_size`。

---

## 8. 对前两份文档的勘误

| 文档 | 原表述 | 修正 |
|------|--------|------|
| `11_megatron_dataset_analysis.md` §5.3 | "packed dataset 配 `BalancedCPScheduler`" | 不是"配",而是 `DefaultDynamicCPScheduler` **本身就是** packing 调度器的子类;动态 CP = `is_dynamic_cp=True` 档 |
| `15_megatron_pp_schedulers_analysis.md` §8.1 | 把 `megatron/core/pipeline_parallel/hybrid_cp_schedule.py` `BalancedCPScheduler` 当作"动态 CP"主体 | 那只是均衡逻辑的**类形态兄弟**;集成入口在 `megatron/core/datasets/data_schedule.py` `DefaultDynamicCPScheduler`,用 `megatron/core/datasets/data_schedule_utils.py` 的 `dcp_*` 函数 |

建议在那两份文档对应位置各加一行指针:"打包与动态 CP 的统一流水线见 `29_megatron_packed_dataset_dynamic_cp_analysis.md`"。

---

## 9. 约束

**9.1 只有 TP-0 参与调度。**
`run()` 一进来就断言 `tp_group.rank() == 0, "Only TP rank 0 should have data_iterator"`(`megatron/core/datasets/data_schedule.py:307`);下游取数侧对称地断言 `data_iterator is not None`(`:619`)与 `data_iterator is None, "Non TP 0 rank should not have data_iterator"`(`:624`)。required sample keys 缺一个即 assert(`:314-318`;`DpBalancedScheduler` 要求的六个 key 列在 `:154-163`)。

**9.2 reroute 的前提,源码明写。**
「All ranks in `dp_group` must provide the same set of data keys. **CP siblings that share a non-CP DP rank must additionally provide byte-identical sample contents.** This holds for the in-tree samplers, which use the non-CP DP rank to select dataset indices.」(`megatron/core/datasets/data_schedule_utils.py:369-372`)—— 换句话说,**自定义 sampler 若让 CP 兄弟 rank 拿到不同数据,这条流水线就不再正确**。
key 集合被白名单锁死:未知 key 直接 assert 失败,并提示必须同时扩 `_REROUTE_KEY_ORDER` 且给出元素布局分类(`:386-391`);各样本的 key 集合必须完全一致(`:392-397`)。`run()` 第 3 步的 `keys_to_keep` 是同一条约束的另一半,注释写「DpBalancedScheduler supports the six fields below. Extend keys_to_keep and the reroute schema together when adding custom dataset metadata.」(`megatron/core/datasets/data_schedule.py:322-325`)。

**9.3 分组算法的失效条件。**
`DpBalancedScheduler` 把 microbatch 数补齐到 `dp_size × microbatch_group_size_per_vp_stage` 的倍数时,若没有可挪的样本就直接 `assert i >= 0, "Not enough samples to move"`(`megatron/core/datasets/data_schedule.py:204`)。
`align_sample_id_groups` 在 VPP 对齐时若尾部没有可拆的 microbatch,直接 `assert False, 'align_sample_id_groups: no tail microbatch has enough ids to split'`(`megatron/core/datasets/data_schedule_utils.py:912`)。
`next_hdp_group_packing_aware` 自陈一条不变量与一条退化:「keeps the legacy invariant that each returned microbatch has no empty DPxCP rank after the fill step. **For non-power-of-two DPxCP layouts, it falls back to the full DPxCP group if power-of-two expansion cannot fill every rank.**」(`:607-610`)—— **DP×CP 不是 2 的幂时,动态 CP 会退化成整组 CP**。

**9.4 容量与 padding 的硬约束。**
`thd_max_packed_sequences` 必须 `>= 1`(`megatron/core/datasets/data_schedule.py:469-470`);当 THD padding 会追加 dummy 序列时必须 `>= 2`,因为这个上限**把 dummy 序列也算在内**(`:476-481`)。
打包实长必须不超过全局 padding 目标(`megatron/core/packed_seq_params.py:282-285`);zigzag 切分要求该目标能被 `2 * cp_size` 整除(`:287-290`)。
CP 切片前,被切的每个张量必须是 1-D 且与 `padding_mask` 等长(`megatron/core/datasets/data_schedule.py:69-71`);`cu_seqlens` / `cu_seqlens_padded` 必须是等长的 1-D `int32`(`:39-41`、`:778-781`)。

**9.5 逐 key gather 的代价,源码自陈。**
「The gather is **intentionally** issued one data key at a time. This pays the fixed collective latency once per key, but bounds temporary memory to one global field at a time.」(`megatron/core/datasets/data_schedule_utils.py:374-377`)—— 这是一条明确的"延迟换显存"取舍,不是实现疏漏。

**9.6 故意不做的事。**
- **不做 PP 组广播**(§2 ③):代价是每个 PP stage 的 TP-0 rank 都要各自跑一遍取数与全局 seqlen 统计 —— 注释原文「every stage independently fetches data and computes the global seqlen stats」(`megatron/core/datasets/data_schedule.py:300-305`)。
- **VPP 中间 stage 不给全量数据**:`vpp_needs_data` 只对首 PP 的第一个 VPP、末 PP 的最后一个 VPP 以及 MTP stage 置 True,注释写「Middle VPP stages only need metadata (cu_seqlens, max_seqlen, etc.).」(`:287-298`)。
- **分组步不切分单条样本**:两个调度器都只把样本 id 分进桶,以"整条样本"为最小单位(`:174-185`、`:428-437`),要再切只能交给 CP(§6 的 `get_cp_slice_for_thd`)。

---

## 10. 发展趋势

**先说锚点的边界**:基线 `71092579` 下,`git grep -n -E "TODO|FIXME|deprecat|WIP"` 在 `megatron/core/datasets/data_schedule.py`、`data_schedule_utils.py`、`megatron/core/packed_seq_params.py` 三个文件上**零命中**。所以本节不锚 TODO,只锚**提交历史**与 **docstring 自陈的版本差异**;每条都**标为推断**。

**① 分桶算法刚换过一代,两套实现正在分叉。**
`data_schedule_utils.py` 侧已是 packing-aware 版(`next_hdp_group_packing_aware`,`:592`,#5154 `d2e7ec5b8`),docstring 里两次以 "the legacy DCP scheduler" / "the legacy invariant" 指称旧版(`:598-610`);而 §4.2 提到的"类形态兄弟" `megatron/core/pipeline_parallel/hybrid_cp_schedule.py` 仍停在 `next_hdp_group`(`:104`,由 `:466` 调用),**没有跟进 packing-aware 的两条改进**。
→ **推断**:两套并行实现已经不再"算法一致";后续要么 `hybrid_cp_schedule.py` 跟进,要么它被 `data_schedule_utils.py` 的函数形态取代。源码没有声明哪一种。

**② reroute 的通信形态仍在调整期。**
它在 2026-06(#4226 删 `broadcast_to_pp_group`)与 2026-08(#6378 all-to-all → all-gather)半年内被改了两次,新版 docstring 还专门解释"为什么逐 key 发起"(`megatron/core/datasets/data_schedule_utils.py:374-377`)。
→ **推断**:这一步的通信形态不宜当成稳定接口引用;引本页 §5 时请连同基线一起标注。

**③ 打包路径的入口判据尚未在所有 pretrain 脚本上统一。**
`pretrain_gpt.py:410` 与 `pretrain_hybrid.py:352` 的 `is_dataset_built_on_rank` 都带 `is_packed_sequence` 形参并对打包路径返回 True;而 `megatron/elastification/pretrain_hybrid_flex.py:500-507` 的同名函数**没有这个形参**,仍是"首/末 PP stage 且 TP-0"的旧判据。
→ **推断**:elastification 路径还没接上"每个 PP stage 自建数据集"的新形态。源码没有说明这是有意为之还是待补 —— 引用前请回到这三处 locator 自行核对。

---

## 11. 小结

- **打包与动态 CP 不是两个协作的特性,而是一个类继承链**:`BasePackingScheduler → DpBalancedScheduler → DefaultDynamicCPScheduler`。动态 CP 调度器**就是**打包调度器的子类。
- **统一的 `run()` 九步流水线**两种调度器共享八步;**唯一分叉是第③步 `get_groups_and_subsamples`**:
  - `DpBalancedScheduler` —— 固定 CP 度 + 贪心 first-fit 打包。
  - `DefaultDynamicCPScheduler` —— 每样本按长度自适应 CP 度(`dcp_gpus_needed`)+ `next_hdp_group` 工作量(`seq²/cp`)均衡分桶。
- **`reroute`(④)** 用 **DP 组 all-gather**(#6378 之前是 all-to-all)把样本从"load 它的 rank"搬到"算它的 rank" —— 打包/动态 CP 必备。
- **`PackedSeqParams` 是汇合点**:打包给出 `cu_seqlens`,动态 CP 给出 `local_cp_size`/`cp_group`;`get_cp_slice_for_thd` 把打包 buffer 再按 CP zigzag 切片。
- 一句话:**序列打包是框架,动态 CP 是它的 CP 感知档** —— 之前两份文档把这条统一流水线拆成了两半,本文合回。

---

*生成依据:`Megatron-LM` `dev` 分支 `85902ef599ea4eb06ada7567a479c524b605767a`(2026-09-01;由 `71092579` 重定基线而来,更早一次为 2026-08-28 由 `ee3f1ff` 推进)。源码行号以该 commit 为准。配套文档:`11_megatron_dataset_analysis.md`、`15_megatron_pp_schedulers_analysis.md` §8.1、`13_megatron_cp_analysis.md`、`packed_seq_params` 见 `11_megatron_dataset_analysis.md` §5.2。*

## Related Pages

- [[11_megatron_dataset_analysis]] · [[15_megatron_pp_schedulers_analysis]] · [[13_megatron_cp_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
