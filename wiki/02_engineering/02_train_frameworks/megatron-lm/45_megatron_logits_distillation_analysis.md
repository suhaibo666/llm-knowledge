---
title: "Megatron-LM 离线 logits 蒸馏：把教师的一次前向存下来反复用"
---

# Megatron-LM 离线 logits 蒸馏：把教师的一次前向存下来反复用

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **维度**：功能树模块 Q。`megatron/training/distillation/`（3 个文件、1906 行），此前**全域零覆盖**。
> **核心文件**：`logits_saver.py`（588 行）· `cached_logits_loss.py`（867 行）· `utils_logits.py`（451 行）
> **最近更新**：2026-09-02 首建。

---

## 1. 背景：蒸馏的成本不在算法，在于教师要陪跑

知识蒸馏的目标函数很简单——让学生的输出分布逼近教师的。难的是工程账：**教师模型必须在学生训练的每一步都做一次前向**。教师通常比学生大，于是一次蒸馏训练里，算力大头花在一个参数根本不更新的模型上；而且教师要和学生一起占显存、一起走并行拓扑，学生能用的并行配置被教师绑住。

**离线蒸馏把这笔账拆开**：教师先单独跑一遍数据集，把每个 token 的输出分布**落盘**；学生训练时只需读文件。教师的前向从"每步一次"变成"整个数据集一次"，并且两边**完全解耦**——教师用什么并行度、什么时候跑，学生一概不关心。

代价随之而来，也正是这个模块的复杂度所在：

1. **落盘量爆炸**。`[序列长度 × 词表大小]` 的完整 logits 是不可能存的——词表十万量级，一个 token 的分布就是几百 KB。
2. **两次运行必须看到同一份数据**。学生第 N 步读到的教师输出，必须对应它自己第 N 步在算的那批 token。数据管线只要有一点不一致，训出来的东西是错的**而且不会报错**。
3. **两次运行的并行度可以不同**。教师用 DP=8 存的，学生可能跑 DP=32。

本页按这三条展开。

**本页不覆盖**：蒸馏算法本身的理论（温度、软标签的作用）；`megatron/core/post_training/modelopt` 的**量化**蒸馏（那是另一条路，见 [[40_megatron_feature_tree_analysis]] §2 模块 I）；训练主循环怎么调 loss → [[01_megatron_architecture_analysis]]。

---

## 2. 存下来的不是分布，是 top-K 的一截

**被否掉的替代很清楚：存完整 logits。** 那是 `[tokens × vocab]` 的稠密张量，一个 10 万词表、8K 序列的 microbatch 就是 GB 级；乘上整个数据集完全不可行。

现选方案是**只存 top-K**（`--logits-save-top-k`），可选再叠一层 top-P 截断（`--logits-save-top-p` / `--logits-save-top-p-min-k`）。判据是分布的长尾对 KL 的贡献极小——保留概率最高的 K 个，其余合并或丢弃。

教师侧的采集挂在前向 hook 上：`LogitsSaverHooks`（`megatron/training/distillation/logits_saver.py:81`）由 `--logits-save-dir` 触发，`attach_hooks`（`:234`）把 `_forward_hook`（`:204`）挂到模型上，逐 microbatch 走 `_process_single_microbatch`（`:294`）→ `_compute_global_topk`（`:369`）→ 可选的 `_apply_topp_truncation`（`:432`），最后由 `_save_accumulated_log_probs`（`:263`）落盘。

它还会**接管损失函数**：`_override_language_model_loss`（`:244`）——教师这一趟只是为了产出 logits，真去算语言模型损失、反向传播是纯浪费。

### 2.1 索引的 17 位打包

存 top-K 要同时存**值**（logprob）与**位置**（在词表里的索引）。索引的取值范围就是词表大小；十万量级需要 17 位——比 `uint16` 多一位，而 `int32` 又浪费近一倍。

`pack_indices`（`megatron/training/distillation/utils_logits.py:233`）的做法是**把 17 位拆成两半**：低 16 位存成 `uint16`，第 17 位单独存成 `bool`——docstring 即 "Split 17-bit global indices into uint16 lower bits + bool 17th bit."（`:234`），`unpack_indices`（`:240`）用 `(bit_17.long() << 16) | low_bits.long()` 还原（`:242`）。

**这比直接用 `int32` 省下约 45% 的索引存储**（16+1 位 vs 32 位）。对一个要存整个数据集的模块，这个常数值得单独写一段代码——这也解释了为什么它没有简单地 `.to(torch.int32)` 了事。

落盘容器是 tar，可选 zstd 压缩，配套 `open_logit_file` / `batched_tar_filename` / `sorted_batched_tars` 与 `decode_logprobs_payload`。路径层还抽象了远端存储：`is_msc_path`（`:65`）/ `is_remote_storage_path`（`:70`）与一组 `storage_*` 包装（`:88-139`），让 `--logits-save-dir` 可以直接指向 `msc://` 或对象存储——**教师和学生很可能不在同一台机器上跑，落盘位置必须能是共享存储**。

---

## 3. 一致性：两次运行必须看到同一个样本流

这是本模块最容易被低估的部分。学生第 N 个 microbatch 读到的教师输出，必须**恰好**对应它自己正在算的那批 token。教师和学生是两次独立的进程，中间只有文件——没有任何运行时握手能保证这件事。

`compute_dataset_hash`（`megatron/training/distillation/utils_logits.py:187`）的解法是**给"样本流身份"算一个哈希**并写进元数据，加载时比对。它的 docstring 明确了纳入哈希的字段范围（`:190-193`）："The fields included are exactly those that determine the global sample stream itself: `seed`, `sequence_length`, `train_samples`（回退到 `train_iters * global_batch_size`），and the data `blend`."

**"exactly those that determine the global sample stream" 这句是这个设计的关键**：哈希刻意**不**包含模型结构、并行度、batch 切分方式——那些变了不影响"第 N 个样本是哪个样本"。只有种子、序列长度、总样本数与数据混合比例会改变样本流本身。

判据由此可推（**本页重建**，源码未直接陈述）：哈希收得太宽会让本可复用的教师产出被误判为不兼容（比如学生换了并行度就得重跑教师，那离线蒸馏的价值就没了）；收得太窄则会放过真正的错配，而错配**不报错、只是训错**。`_blend_identifiers`（`:164`）单独处理数据混合的身份，`_verify_logprobs_metadata` 在加载时执行比对。

---

## 4. DP 重映射：教师和学生的数据并行度可以不同

教师用 DP=8 存的数据，学生跑 DP=32 也要能读。`_compute_dp_remapping`（`megatron/training/distillation/cached_logits_loss.py:139`）负责这层换算。

它的前提是**落盘顺序是确定的轮转**（docstring `:149-152`）："Data is distributed across DP ranks in a round-robin (wrapping) fashion: with `dp_size_saved` ranks and `G` global microbatches, saved rank `d` holds microbatch indices `d, d + dp_size_saved, d + 2·dp_size_saved, …`"

于是两个方向各有算法（`:154-161`）：

| 方向 | 条件 | 做法 |
|---|---|---|
| **升配** | `dp_size_saved < dp_size` | 每份存档被 `dp_size // dp_size_saved` 个当前 rank 共享，各自按步长跨取其中的 microbatch |
| **降配** | `dp_size_saved > dp_size` | 每个当前 rank 读 `dp_size_saved // dp_size` 份存档，**交错**它们的 microbatch 以重建全局顺序 |

"interleaves their microbatches to reconstruct its share of the global ordering"（`:160-161`）这句点出了要害：**重映射的正确性判据不是"读到了数据"，而是"重建出的全局顺序与教师当时一致"**——顺序错了，token 与它的教师分布就对不上，而 §3 的哈希校验查不出这一层（它只管样本流身份，不管本次读取的排列）。

保存的 DP 规模由 `detect_saved_dp_size` 从目录里探测；找不到分批 tar 时退化为恒等映射 `([dp_rank], 0, 1, dp_size)`（`:146-147`）。

**约束**：两个方向都要求整除。非整倍数的 DP 变更不在支持范围内。

---

## 5. 学生侧：流式加载与 TP 感知的稀疏 KL

### 5.1 加载

`TeacherTarDataset`（`megatron/training/distillation/cached_logits_loss.py:211`）是个 `IterableDataset`：`_discover_shards`（`:288`）发现属于本 rank 的分片，`_slice_microbatches`（`:308`）切出本次要的部分。配套 `TarShardPrefetcher` 做预取，由 `--logits-load-decode-threads` / `--logits-load-prefetch-factor` / `--logits-load-msc-prefetch-depth` 调参。

**为什么必须流式**：教师产出是整个数据集规模的，不可能全量载入内存。`IterableDataset` 而非 `Dataset` 的选择就是这个约束的直接结果——后者要求可随机索引，意味着要么全载入、要么每次寻道。

### 5.2 损失

学生侧的 logits 由 `StudentLogitsCapture`（`:99`）经 hook 抓取（`attach_hooks` `:106`、`_capture_logits` `:114`、`pop` `:120`）；`CachedLogitsKDLoss` 与 `LossFuncCallable` 组装成可被训练主循环调用的损失函数，由 `--logits-load-dir` + `--logits-load-kd-loss-alpha` 触发。

核心是 `topk_kl_div`（`:523`）——**在 top-K 的稀疏支撑集上算 KL**，而不是在完整词表上。它的签名（`:523-531`）带 `tp_size` / `tp_rank` / `tp_group`：**学生的 logits 是沿词表维被 TP 切开的**（见 [[12_megatron_tp_analysis]] 的 `VocabParallelEmbedding`），而教师存的 top-K 索引是**全局词表**下的编号。所以这个核必须自己完成"全局索引 → 本 TP rank 是否持有 → 跨 TP 归约"这条链，不能简单地对本地 logits 做 softmax。

入口处 `student_logits.float()` 与 `teacher_topk_logprobs.float()`（`:533-534`）把两侧都提到 fp32——KL 涉及 log 与差值，低精度下数值不稳。

`add_ghost_token` 参数（`:530`）用于处理被 top-K 截断掉的那部分概率质量。

> [!note] 待展开
> `topk_kl_div` 的**具体归约路径**（本 rank 持有哪些全局索引、跨 TP 怎么合并部分和）本页只给了它必须解决的问题与签名依据，没有逐行走查函数体（`:523-590` 一带）。同样未展开的还有 `add_ghost_token` 的确切语义、`_compute_global_topk` 在 TP 下如何取全局 top-K。

---

## 6. 配置契约

蒸馏的开关**不在任何被 [[41_megatron_config_surface_analysis]] §2 的 `ArgumentGroupFactory` 追踪的 config dataclass 里**——它们是 `megatron/training/arguments.py` 的 `Logits Distillation` 组里**手写的 argparse**（`_add_logits_distillation_args`）。

**这正是本页此前零覆盖的直接原因**：[[40_megatron_feature_tree_analysis]] §3.2 的配置面对账只枚举 14 个 config 类，看不见手写 argparse 组；而文件面对账虽然把这 3 个文件标成了 🟡，却因为没有 flag 牵引而一直没人认领。**没有 flag 盯着的地方，正是页面容易漏的地方**——这条在 40 号页写过，蒸馏模块就是它的实例。

| 开关 | 侧 | 作用 |
|---|---|---|
| `--logits-save-dir` | 教师 | 触发采集并指定落盘位置（可为 `msc://` / 对象存储） |
| `--logits-save-top-k` | 教师 | 每 token 保留的候选数 |
| `--logits-save-top-p` · `--logits-save-top-p-min-k` | 教师 | 可选的 top-P 截断与其最小 K 下限 |
| `--logits-save-dtype` | 教师 | 落盘数值精度 |
| `--logits-load-dir` | 学生 | 教师产出目录 |
| `--logits-load-kd-loss-alpha` | 学生 | KD 损失在总损失里的权重 |
| `--logits-load-ignore-errors` | 学生 | 遇到损坏分片时继续 |
| `--logits-load-decode-threads` · `--logits-load-prefetch-factor` · `--logits-load-msc-prefetch-depth` | 学生 | 流式加载的并发与预取深度 |

---

## 7. 约束与边界

| 边界 | 表现 | 证据 |
|---|---|---|
| 只存 top-K，长尾不可恢复 | 学生看不到教师在低概率区的形状 | `--logits-save-top-k` 为必需项 |
| 索引按 17 位打包 | 词表超过 $2^{17}$（131072）时该编码不成立 | `utils_logits.py:233-242` |
| 数据管线必须一致 | 种子/序列长度/总样本数/blend 任一不同即拒绝加载 | `utils_logits.py:187-193` |
| DP 变更须整除 | 非整倍数的升/降配不支持 | `cached_logits_loss.py:154-161` |
| 教师产出与数据集同规模 | 存储成本随数据集线性增长，须走共享/对象存储 | `utils_logits.py:65-139` 的 `storage_*` 抽象 |
| 学生 logits 是 TP 切分的 | KL 核必须自己处理全局索引到本地分片的映射 | `cached_logits_loss.py:523-531` 的签名 |
| 教师侧禁用 LM 损失 | 采集趟不做反向 | `logits_saver.py:244` |

---

## Related Pages

- [[40_megatron_feature_tree_analysis]] — 功能树总览；本页填的是它 §4 仪表盘里模块 Q 那一行（此前唯一无页可落的模块）
- [[41_megatron_config_surface_analysis]] — §6 说明本模块的开关为何**不在**那里讲的 dataclass 驱动体系内，以及这为什么导致它长期不可见
- [[12_megatron_tp_analysis]] — §5.2 的 KL 核要处理的词表并行切分在那里
- [[11_megatron_dataset_analysis]] — §3 的样本流一致性依赖的数据管线在那里
- [[24_megatron_linear_cross_entropy_analysis]] — 另一处"不物化完整 logits"的做法，与本页 §2 的 top-K 截断是同一压力下的两条路
- [[courses/megatron_lm|Megatron-LM 阅读路径]] — 本页在阅读路径中的位置（站 6 之后的专题）
