# Megatron-LM 序列打包与动态 CP 的统一流水线深度解析

> 代码基准:`Megatron-LM/` 子仓库 `dev` 分支,commit `ee3f1ff`
> 核心文件:`megatron/core/datasets/data_schedule.py`(954 行)、`data_schedule_utils.py`(1020 行);`packed_seq_params.py`;`pipeline_parallel/hybrid_cp_schedule.py`
> 配套阅读:`11_megatron_dataset_analysis.md` §3、`26_megatron_pp_supplements_analysis.md` §3、`13_megatron_cp_analysis.md`
> 定位:**勘误 + 补全**。`11_megatron_dataset_analysis.md` 把序列打包当主角、`26_megatron_pp_supplements_analysis.md` §3 把动态 CP 当独立特性,分两处讲。本文说清楚:**在代码里它俩是同一条流水线、同一个类继承链** —— 动态 CP 不是和打包并列协作的特性,而是打包调度器的一个子类。

---

## 0. 总览:一个继承链,不是两个特性

`data_schedule.py` 里序列打包的类层次:

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

## 1. 统一的 `run()` 9 步流水线

`DpBalancedScheduler.run()`(`data_schedule.py:164`)定义了完整流水线,`DefaultDynamicCPScheduler` **不重写它**,直接继承:

```
入口 wrap_data_iterator(data_schedule.py:395)
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

---

## 2. 唯一的分叉:第③步 `get_groups_and_subsamples`

### 2.1 `DpBalancedScheduler` —— 定长 CP + 贪心打包

`data_schedule.py:111`。所有样本用**同一个固定 `cp_size`**,一个打包 microbatch(横跨该 DP 组的所有 CP rank)的容量是 `max_seq_len_all_ranks = max_seqlen_per_dp_cp_rank × cp_size`。

```python
for i in range(len(sample_id_seqlens)):
    if sum_seqlen + seqlen[i] <= max_seq_len_all_ranks:   # 还塞得下
        single_microbatch.append(i); sum_seqlen += seqlen[i]
    else:                                                  # 塞不下 → 封一个 microbatch
        packed_id_groups.append(single_microbatch)
        single_microbatch = [i]; sum_seqlen = seqlen[i]
```

**贪心 first-fit 打包**:按原顺序往一个 microbatch 里塞样本,塞满就封、开下一个。再把 microbatch 数量补齐成 `dp_size × microbatch_group_size_per_vp_stage` 的整数倍(从后面的 microbatch 挪样本)。CP 度对所有样本恒定。

### 2.2 `DefaultDynamicCPScheduler` —— 按长度分 CP + 工作量均衡

`data_schedule.py:347`。**每个样本分到的 CP 卡数随它的长度变化**:

```python
gpus_fn     = lambda seq_len: dcp_gpus_needed(seq_len, mslpr, min_cp)   # 该样本要几张 CP 卡
workload_fn = lambda seq_len, cp=None: dcp_get_total_workload(...)      # 工作量 ≈ seq²/cp
buckets_fn  = dcp_make_buckets_equal(...)                              # 均衡分桶

sample_id_seqlens = sorted(..., key=seqlen, reverse=True)              # 长样本优先
while sample_id_seqlens:
    mb, sample_id_seqlens, ... = next_hdp_group(                       # 每次形成一个均衡的 hdp 组
        sample_id_seqlens, workload_fn, total_hdp_gpus, gpus_fn, buckets_fn, ...)
    groups.append(mb)
```

- `dcp_gpus_needed`(`data_schedule_utils.py:971`):长样本分更多 CP 卡(向上取 2 的幂),短样本少分 —— **CP 度自适应序列长度**。
- `dcp_get_total_workload`:用 `seq²/cp` 估 attention 负载(`O(S²)` 除以 CP 摊分)。
- `next_hdp_group` + `dcp_make_buckets_equal`:把样本贪心打包成一个个 **hdp 组**(hybrid DP 组),使每个 DP×CP rank 的总工作量大致相等。
- `align_sample_id_groups`:VPP 时对齐组数。

> `hybrid_cp_schedule.py` 的 `BalancedCPScheduler`(`26_megatron_pp_supplements_analysis.md` §3 分析过)是同一套均衡逻辑的**类形态兄弟**;`data_schedule.py` 的 `DefaultDynamicCPScheduler` 实际用的是 `data_schedule_utils.py` 里的**函数形态** `dcp_*` + `next_hdp_group`。二者算法一致,是并行实现。**真正把打包与动态 CP 缝在一起的集成点,是 `DefaultDynamicCPScheduler`。**

### 2.3 两者对照

| | `DpBalancedScheduler` | `DefaultDynamicCPScheduler` |
|--|----------------------|------------------------------|
| `is_dynamic_cp` | False | True |
| CP 度 | 所有样本固定 `cp_size` | **每样本按长度自适应**(`dcp_gpus_needed`) |
| 第③步算法 | 贪心 first-fit 打包 | `next_hdp_group` 工作量均衡分桶 |
| 容量基准 | `max_seqlen_per_dp_cp_rank × cp_size` | `max_seqlen_per_dp_cp_rank`(每 rank) |
| ①②④⑤⑥⑦⑧⑨ | ←———— 完全相同 ————→ | |

---

## 3. 第④步 `reroute`:把样本搬到"该算它的 rank"

`reroute_samples_to_dcp_ranks`(`data_schedule_utils.py:366`)。这一步是打包/动态 CP 流水线**必须有**的:

- 数据集 loader 是按 **DP** 把样本分给各 rank 的(谁 load 了哪条)。
- 但第③步的调度结果是"样本 X 应该由 DCP rank `d` 计算"—— load 它的 rank 和算它的 rank **通常不是同一个**。
- 于是对 batch 里每个 key 做 **all-to-all**,把每个子样本的数据从"load 它的 rank"搬到"算它的 rank"。

`is_dynamic_cp` 在这里也透传 —— 动态 CP 下一个样本可能要发给多个 CP rank。

---

## 4. 第⑤步:`build_packed_microbatches` + `PackedSeqParams` + CP 切片

`build_packed_microbatches`(`data_schedule_utils.py:485`)在本 rank 把分到的子样本拼成 **THD 打包 buffer**,产出 `PackedSeqParams`(`packed_seq_params.py`,见 `11_megatron_dataset_analysis.md` §3.2):`cu_seqlens` 标出每条子序列边界,`qkv_format='thd'`。

动态 CP 的衔接点:
- `PackedSeqParams` 的 **`local_cp_size` / `cp_group`** —— 每个打包 buffer **可以有自己的 CP 度**(因为不同 microbatch 的样本长度不同 → 动态 CP 给的 cp_size 不同)。
- `get_cp_slice_for_thd`(`data_schedule_utils.py:15`):一个打包 THD buffer 若 `cp_size > 1`,要再沿序列切给 CP rank —— 且按 zigzag 均衡(因果掩码,见 `13_megatron_cp_analysis.md` §2.2)。
- `get_batch_on_this_rank_for_sequence_packing(..., dynamic_cp=True)`(`data_schedule.py:466`)取数时读 `batch['local_cp_size']`,据此决定本 buffer 的 CP 切分。

所以:**打包决定"buffer 里装哪几条序列",动态 CP 决定"这个 buffer 用几张卡的 CP、怎么切" —— 二者在同一个 `PackedSeqParams` 对象上汇合**。

---

## 4.5 NEW:varlen 数据源 + get_batch 统一(dev@232c478d4)

> [!update] 2026-06-16 · dev@232c478d4
> 这条统一 `run()` 流水线的**上游入口**和**下游取数**都有更新,但九步骨架与"唯一分叉=第③步"的结论不变:
>
> - **第①步 `_unpack_batch` 支持双输入**(`data_schedule_utils.py:48`,#4832):除了 `SFTDataset` 那种"一条样本里 `cu_seqlens` 拼了多条子序列、需切开"的**预打包**形态,新增 `VarlenDataset`(`--use-varlen-dataset`)这种"每 index 已是单条子样本、自带 `padded_seq_len`"的**已拆开**形态 —— 后者只需丢掉 collate_fn 多加的 batch 维、缺 `original_seq_len` 时从 `padded_seq_len` 补,再走同一条 ②→⑨。`VarlenDataset` 细节见 `11_megatron_dataset_analysis.md` §3.4。
> - **下游 get_batch 统一 + SFT THD 支持 PP**(#4103):`get_batch_on_this_*` 取数函数收敛进 `megatron/core/utils.py` —— `get_batch_on_this_tp_rank`(`:1992`,**长度前缀协议**广播变长的 `cu_seqlens`,并在动态 CP 下广播 `local_cp_size` / `hybrid_cp_seq_length`)、`get_thd_batch_on_this_cp_rank`(`:2439`,对应第⑤步在本 rank 的 THD 切片)。SFT 的 THD 打包现可与 PP 共用(呼应第⑦步 `broadcast_to_pp_group`)。
> - **第⑥步 seqlen 统计修正**:`train_step` 现保留 seqlen 统计(commit 95654c956);`sequence_packing_scheduler` 非空时的 TFLOPs 计算修正(#5342)—— 对应本页第⑥步 `Σseqlen / Σseqlen²` 的吞吐统计。

---

## 5. 一张图看清统一关系

```
                      config.sequence_packing_scheduler
                                  │
                  ┌───────────────┴────────────────┐
            "dp_balanced"                  "default_dynamic_cp"
            DpBalancedScheduler            DefaultDynamicCPScheduler
                  │                                │
                  └────────► 共享同一个 run() ◄─────┘
                                  │
   ①取数+全局seqlen ②校验 ┌─③ 分组(唯一分叉)─┐ ④reroute ⑤打包buffer ⑥FLOPs ⑦⑧广播 ⑨迭代器
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

## 6. 对前两份文档的勘误

| 文档 | 原表述 | 修正 |
|------|--------|------|
| `11_megatron_dataset_analysis.md` §3.3 | "packed dataset 配 `BalancedCPScheduler`" | 不是"配",而是 `DefaultDynamicCPScheduler` **本身就是** packing 调度器的子类;动态 CP = `is_dynamic_cp=True` 档 |
| `26_megatron_pp_supplements_analysis.md` §3 | 把 `hybrid_cp_schedule.py` `BalancedCPScheduler` 当作"动态 CP"主体 | 那只是均衡逻辑的**类形态兄弟**;集成入口在 `data_schedule.py` `DefaultDynamicCPScheduler`,用 `data_schedule_utils.py` 的 `dcp_*` 函数 |

建议在那两份文档对应位置各加一行指针:"打包与动态 CP 的统一流水线见 `29_megatron_packed_dataset_dynamic_cp_analysis.md`"。

---

## 7. 小结

- **打包与动态 CP 不是两个协作的特性,而是一个类继承链**:`BasePackingScheduler → DpBalancedScheduler → DefaultDynamicCPScheduler`。动态 CP 调度器**就是**打包调度器的子类。
- **统一的 `run()` 九步流水线**两种调度器共享八步;**唯一分叉是第③步 `get_groups_and_subsamples`**:
  - `DpBalancedScheduler` —— 固定 CP 度 + 贪心 first-fit 打包。
  - `DefaultDynamicCPScheduler` —— 每样本按长度自适应 CP 度(`dcp_gpus_needed`)+ `next_hdp_group` 工作量(`seq²/cp`)均衡分桶。
- **`reroute`(④)** 用 all-to-all 把样本从"load 它的 rank"搬到"算它的 rank" —— 打包/动态 CP 必备。
- **`PackedSeqParams` 是汇合点**:打包给出 `cu_seqlens`,动态 CP 给出 `local_cp_size`/`cp_group`;`get_cp_slice_for_thd` 把打包 buffer 再按 CP zigzag 切片。
- 一句话:**序列打包是框架,动态 CP 是它的 CP 感知档** —— 之前两份文档把这条统一流水线拆成了两半,本文合回。

---

*生成依据:`Megatron-LM` `dev` 分支 `ee3f1ff`。源码行号以该 commit 为准。配套文档:`11_megatron_dataset_analysis.md`、`26_megatron_pp_supplements_analysis.md` §3、`13_megatron_cp_analysis.md`、`packed_seq_params` 见 `11_megatron_dataset_analysis.md` §3.2。*

## Related Pages

- [[11_megatron_dataset_analysis]] · [[26_megatron_pp_supplements_analysis]] · [[13_megatron_cp_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
