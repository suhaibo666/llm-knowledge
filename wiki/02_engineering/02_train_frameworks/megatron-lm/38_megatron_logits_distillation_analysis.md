---
title: "Megatron-LM 离线 logits 蒸馏：top-K 缓存协议与稀疏 KL"
---

# Megatron-LM 离线 logits 蒸馏：top-K 缓存协议与稀疏 KL

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **维度**：功能树模块 Q。`megatron/training/distillation/`（3 个文件、1906 行），此前**全域零覆盖**。
> **核心文件**：`logits_saver.py`（588 行）· `cached_logits_loss.py`（867 行）· `utils_logits.py`（451 行）
> **阅读位置**：运行时与交付专题中的离线教师产物路径；先读 [[11_megatron_dataset_analysis]] 的样本流，再读本页的保存、重映射与稀疏 KL。
> **最近更新**：2026-09-03。由 45 重排到 38；按冻结源码补出 CP/microbatch 复用边界、student loss 全链，并标明 producer 的 tar flush 接线尚未闭合。

---

## 1. 背景：蒸馏的成本不在算法，在于教师要陪跑

知识蒸馏的目标函数很简单——让学生的输出分布逼近教师的。难的是工程账：**教师模型必须在学生训练的每一步都做一次前向**。教师通常比学生大，于是一次蒸馏训练里，算力大头花在一个参数根本不更新的模型上；而且教师要和学生一起占显存、一起走并行拓扑，学生能用的并行配置被教师绑住。

**离线蒸馏的目标架构**是把这笔账拆开：教师先跑数据集并发布每个 token 的稀疏输出，学生训练时再流式读取。时间上两次运行可以分离，TP 词表切分也由全局 top-K 协议隔开；但它们并非“完全解耦”：学生必须复现同一数据顺序、同一 CP layout 和同一 micro-batch size，DP 也只支持整除比重映射（`megatron/training/distillation/cached_logits_loss.py:30-39`）。

代价随之而来，也正是这个模块的复杂度所在：

1. **落盘量爆炸**。`[序列长度 × 词表大小]` 的完整 logits 是不可能存的——词表十万量级，一个 token 的分布就是几百 KB。
2. **两次运行必须看到同一份数据**。学生第 N 步读到的教师输出，必须对应它自己第 N 步在算的那批 token。数据管线只要有一点不一致，训出来的东西是错的**而且不会报错**。
3. **只有部分并行维度可变化**。TP 由全局索引协议处理；DP 可在整除前提下升降配；CP rank 布局与 micro-batch size 必须保持一致。

本页按这三条展开。

**本页不覆盖**：蒸馏算法本身的理论（温度、软标签的作用）；`megatron/core/post_training/modelopt` 的**量化**蒸馏（那是另一条路，见 [[40_megatron_feature_tree_analysis]] §2 模块 I）；训练主循环怎么调 loss → [[01_megatron_architecture_analysis]]。

---

## 2. Producer：top-K 缓存到 pending buffer，写盘接线尚未闭合

**被否掉的替代很清楚：存完整 logits。** 那是 `[tokens × vocab]` 的稠密张量，一个 10 万词表、8K 序列的 microbatch 就是 GB 级；乘上整个数据集完全不可行。

当前协议只保留 top-K（`--logits-save-top-k`），可选再叠 top-P 截断（`--logits-save-top-p` / `--logits-save-top-p-min-k`）。源码没有保证“长尾对 KL 的贡献很小”；K/P 是存储量与蒸馏保真度之间的显式旋钮。学生侧固定以 `add_ghost_token=True` 调用 KL，把 top-K 之外的**总概率质量**聚成一个 ghost token，但尾部分布内部形状仍不可恢复（`megatron/training/distillation/cached_logits_loss.py:560-577`、`:769-777`）。

教师侧的采集挂在前向 hook 上：`LogitsSaverHooks`（`megatron/training/distillation/logits_saver.py:81`）由 `--logits-save-dir` 触发，`attach_hooks`（`:234`）把 `_forward_hook`（`:204`）挂到模型上，逐 microbatch 走 `_process_single_microbatch`（`:294`）→ `_compute_global_topk`（`:369`）→ 可选的 `_apply_topp_truncation`（`:432`）。`_save_accumulated_log_probs` 这个名字容易误导：它只把张量搬到 CPU 并经 `_buffer_iteration` 写入内存中的 `_pending_writes`（`:263-285`、`:490-510`），此时磁盘上还没有 tar。

真正的 tar writer 是 `take_pending_data` → `_write_batched_tar`：前者转移 pending bytes，后者在 checkpoint 后台进程中做 zstd 压缩、写 metadata/payload 并原子移动临时文件（`:516-588`）。但冻结树中 `get_logits_saver`、`take_pending_data` 与 `_write_batched_tar` 除定义/导出外没有调用点；因此本仓基线只闭合到**待异步 flush 的内存 buffer**，没有闭合 producer→disk 的集成 hop。读者不能仅凭 `--logits-save-dir` 推断 tar 一定被发布；可能存在的外部接线不在本仓证据范围内。

它还会替换 LM loss，但不是跳过 backward：`_override_language_model_loss` 返回 `(logits * 0).sum(...)`，docstring 明写“preserves gradient edges”（`:244-252`）。这消除了原 LM loss 的数值贡献，同时刻意保留计算图边；正常训练编排仍可能执行反向。

### 2.1 索引的 17 位打包

存 top-K 要同时存**值**（logprob）与**位置**（在词表里的索引）。索引的取值范围就是词表大小；十万量级需要 17 位——比 `uint16` 多一位，而 `int32` 又浪费近一倍。

`pack_indices`（`megatron/training/distillation/utils_logits.py:233`）的做法是**把 17 位拆成两半**：低 16 位存成 `uint16`，第 17 位单独存成 `bool`——docstring 即 "Split 17-bit global indices into uint16 lower bits + bool 17th bit."（`:234`），`unpack_indices`（`:240`）用 `(bit_17.long() << 16) | low_bits.long()` 还原（`:242`）。

这不是物理上的 17-bit packed scalar：PyTorch `uint16` 通常占 2 byte，`bool` 占 1 byte，所以在**未计容器、对齐与压缩**的原始 tensor payload 口径下是约 3 byte/index，对 `int32` 的 4 byte/index 约省 **25%**。源码只定义两张 tensor 的 dtype（`:233-242`），不保证序列化后的固定压缩率。

预期落盘容器是 tar，payload 使用 zstd；writer 在缺少 `zstandard` 时直接 `ImportError`，所以压缩不是该实现里的可选档（`megatron/training/distillation/logits_saver.py:540-586`）。配套的 `open_logit_file` / `batched_tar_filename` / `sorted_batched_tars` 与 `decode_logprobs_payload` 支持读取协议。路径层还抽象了远端存储：`is_msc_path`（`megatron/training/distillation/utils_logits.py:65`）/ `is_remote_storage_path`（`:70`）与一组 `storage_*` 包装（`:88-139`），使协议可面向共享/对象存储；是否真正发布仍受上一段未闭合 flush hop 限制。

---

## 3. 一致性：两次运行必须看到同一个样本流

这是本模块最容易被低估的部分。学生第 N 个 microbatch 读到的教师输出，必须**恰好**对应它自己正在算的那批 token。教师和学生是两次独立的进程，中间只有文件——没有任何运行时握手能保证这件事。

`compute_dataset_hash`（`megatron/training/distillation/utils_logits.py:187-210`）写入的不是数据文件内容哈希，而是**样本流声明哈希**：它序列化 `seed`、`sequence_length`、`train_samples`（缺失时回退到 `train_iters * global_batch_size`）与 `blend`，再计算 MD5。模型结构、TP/DP 切分和 batch 切分方式都不在这个声明里。

`blend` 的口径还要再收紧一层：`_blend_identifiers` 只保留每个数据前缀的**去扩展名 basename 与权重**，目录路径和 `.bin` / `.idx` 文件内容都不参与哈希（`:164-184`）。因此它能发现“种子、长度、样本数、blend 名称或权重改变”，却不能证明底层数据字节相同；若同名数据文件被替换而这些声明不变，校验仍可能通过。`TeacherTarDataset` 计算本次期望值并在读取 tar 元数据时传给校验路径（`megatron/training/distillation/cached_logits_loss.py:253-256`）。

所以正确判据是：该哈希是**必要的声明一致性检查，不是内容完整性证明**。它收得更宽会降低教师产物的复用性，收得更窄则会放过错配；现实现选择了路径无关的配方身份，调用方仍需自行保证同名 IndexedDataset 内容没有漂移。

---

## 4. DP 重映射：教师和学生的数据并行度可以不同

教师用 DP=8 存的数据，学生跑 DP=32 也要能读。`_compute_dp_remapping`（`megatron/training/distillation/cached_logits_loss.py:139`）负责这层换算。

它的前提是**落盘顺序是确定的轮转**（docstring `:149-152`）："Data is distributed across DP ranks in a round-robin (wrapping) fashion: with `dp_size_saved` ranks and `G` global microbatches, saved rank `d` holds microbatch indices `d, d + dp_size_saved, d + 2·dp_size_saved, …`"

于是两个方向各有算法（`:154-161`）：

| 方向 | 条件 | 做法 |
|---|---|---|
| **升配** | `dp_size_saved < dp_size` | 每份存档被 `dp_size // dp_size_saved` 个当前 rank 共享，各自按步长跨取其中的 microbatch；每个存档 iteration 的 microbatch 数还必须整除这个比值 |
| **降配** | `dp_size_saved > dp_size` | 每个当前 rank 读 `dp_size_saved // dp_size` 份存档，**交错**它们的 microbatch 以重建全局顺序 |

"interleaves their microbatches to reconstruct its share of the global ordering"（`:160-161`）这句点出了要害：**重映射的正确性判据不是"读到了数据"，而是"重建出的全局顺序与教师当时一致"**——顺序错了，token 与它的教师分布就对不上，而 §3 的哈希校验查不出这一层（它只管样本流身份，不管本次读取的排列）。

保存的 DP 规模由 `detect_saved_dp_size` 从目录里探测；找不到分批 tar 时退化为恒等映射 `([dp_rank], 0, 1, dp_size)`（`:146-147`）。

**约束**：两个方向都要求新旧 DP world size 整除；升配还要求存档 iteration 内的 `num_mb` 能被 `dp_ratio` 整除，否则 `_slice_microbatches` 直接抛 `ValueError`（`megatron/training/distillation/cached_logits_loss.py:308-321`）。

---

## 5. 学生侧：流式加载与 TP 感知的稀疏 KL

### 5.1 加载

`TeacherTarDataset`（`megatron/training/distillation/cached_logits_loss.py:211`）是个 `IterableDataset`：`_discover_shards`（`:288`）发现属于本 rank 的分片，`_slice_microbatches`（`:308`）切出本次要的部分。配套 `TarShardPrefetcher` 做预取，由 `--logits-load-decode-threads` / `--logits-load-prefetch-factor` / `--logits-load-msc-prefetch-depth` 调参。

**为什么这个实现选择流式 IterableDataset**（分析重建）：当前 tar 协议没有“逻辑 microbatch → 随机字节偏移”的索引；reader 按 tar 顺序解码，线程池结果以 FIFO 顺序交付，并在已知 shard 耗尽后刷新目录以发现并发发布的新 shard（`:288-306`、`:406-515`）。因此顺序迭代与这个协议天然一致，也避免全量载入教师产出。这里不能反推 map-style `Dataset` 天生不能 lazy read；若另建随机访问索引，也可以实现惰性读取，只是不是本仓选择的协议。

### 5.2 损失

这里不是“类存在，所以主循环大概会用”的静态拼图，而是有一条可闭合的 live path：

1. 模型构建后，`training.py` 在 `--logits-load-dir` 非空时创建 `StudentLogitsCapture`，把 hook 挂到最后一个模型分片的 output layer（`megatron/training/training.py:2787-2791`）。
2. `pretrain_gpt.loss_func` 检查同一开关，构造并缓存 `LossFuncCallable`，随后以本 microbatch 的 `loss_mask`、LM loss tensor 与 model 调用它（`pretrain_gpt.py:265-283`、`:302-314`）。
3. `LossFuncCallable.__call__` 从全局 capture `pop()` 出带梯度的学生 logits，调用 `CachedLogitsKDLoss` 取教师 microbatch 并进入 `topk_kl_div`（`megatron/training/distillation/cached_logits_loss.py:837-848`）。
4. `topk_kl_div` 返回各 TP rank 的**局部稀疏 KL 贡献**；wrapper 先按 token mask 求和，再显式在 TP group 上 `all_reduce`，最后按 `alpha` 与 LM loss 加权（`:849-864`）。

`topk_kl_div`（`:523-583`）内部的 TP 分工要拆成“全局归一化”和“最终 loss 汇总”两件事：

- 两侧先转 fp32（`:533-534`）；学生 logits 的局部最大值经 TP `MAX all_reduce`，局部 `sum(exp)` 再经可微的 TP `SUM all_reduce`，得到跨完整词表一致的 log-softmax（`:536-545`）。
- 按 `offset = local_vocab_size * tp_rank` 把教师的**全局词表索引**映射到本地 shard；mask 只保留本 rank 真正持有且非 sentinel 的 top-K 项（`:547-558`）。
- ghost token 把教师和学生各自 top-K 之外的**总概率质量**各压成一个桶；学生残差先跨 TP 求和，ghost 项只由 TP rank 0 计入，避免重复（`:560-577`）。这保住了尾部总质量，但不能恢复尾部内部形状。
- 函数本身只对本 rank mask 命中的项求和并返回 `[B, S]`（`:579-583`）；跨 TP 的最终 KL 求和不在此函数内，而在上面的 `LossFuncCallable`（`:847-850`）。

---

## 6. 配置契约

蒸馏的开关**不在任何被 [[41_megatron_config_surface_analysis]] §2 的 `ArgumentGroupFactory` 追踪的 config dataclass 里**——它们是 `megatron/training/arguments.py` 的 `Logits Distillation` 组里**手写的 argparse**（`_add_logits_distillation_args`）。

**这正是本页此前零覆盖的直接原因**：[[40_megatron_feature_tree_analysis]] §3.2 的配置面对账只枚举 14 个 config 类，看不见手写 argparse 组；而文件面对账虽然把这 3 个文件标成了 🟡，却因为没有 flag 牵引而一直没人认领。**没有 flag 盯着的地方，正是页面容易漏的地方**——这条在 40 号页写过，蒸馏模块就是它的实例。

| 开关 | 侧 | 作用 |
|---|---|---|
| `--logits-save-dir` | 教师 | 触发 producer hook 并指定目标目录（可为 `msc://` / 对象存储）；冻结树只闭合到 pending buffer，不能据此断言已写盘 |
| `--logits-save-top-k` | 教师 | 每 token 保留的候选数 |
| `--logits-save-top-p` · `--logits-save-top-p-min-k` | 教师 | 可选的 top-P 截断与其最小 K 下限 |
| `--logits-save-dtype` | 教师 | pending payload / 预期存档的数值精度 |
| `--logits-load-dir` | 学生 | 教师产出目录 |
| `--logits-load-kd-loss-alpha` | 学生 | KD 损失在总损失里的权重 |
| `--logits-load-ignore-errors` | 学生 | 遇到损坏分片时继续 |
| `--logits-load-decode-threads` · `--logits-load-prefetch-factor` · `--logits-load-msc-prefetch-depth` | 学生 | 流式加载的并发与预取深度 |

---

## 7. 约束与边界

| 边界 | 表现 | 证据 |
|---|---|---|
| 只存 top-K，尾部形状不可恢复 | ghost token 只保留 top-K 外的总概率质量 | `cached_logits_loss.py:560-577`、`:769-777` |
| 索引按 17 位打包 | 词表超过 $2^{17}$（131072）时该编码不成立 | `utils_logits.py:233-242` |
| 声明哈希不是内容哈希 | 同 basename、同权重的数据内容被替换时可能漏检 | `utils_logits.py:164-210` |
| CP 与 micro-batch 必须相同 | 当前 reader 按相同 `cp_rank` 和 microbatch 边界消费教师数据 | `cached_logits_loss.py:30-39` |
| DP 变更有两层整除条件 | 新旧 DP world size 须整除；升配时每个 iteration 的 `num_mb` 还须整除 `dp_ratio` | `cached_logits_loss.py:154-161`、`:308-321` |
| producer 写盘链未闭合 | hook 只写 `_pending_writes`；writer 存在，但冻结树没有调用接线 | `logits_saver.py:263-285`、`:490-588` |
| 学生 logits 是 TP 切分的 | 核内负责全局归一化和本地贡献，wrapper 再做最终 TP loss 归约 | `cached_logits_loss.py:523-583`、`:837-850` |
| 教师 LM loss 数值归零但保留图边 | `(logits * 0).sum(...)` 不等于跳过 backward | `logits_saver.py:244-252` |

### 7.1 当前基线能依赖什么

| 能依赖 | 不能仅凭本仓断言 |
|---|---|
| producer hook 会计算全局 top-K 并形成待写 buffer | `--logits-save-dir` 已让 tar 自动落盘 |
| tar writer/reader 的格式实现存在，consumer 读取与稀疏 KL 调用链闭合 | writer 已被训练或 checkpoint 编排实际调用 |
| TP 可由全局索引协议适配；DP 可做整除重映射 | 教师与学生可任意改变 CP、micro-batch 或数据内容 |

---

## Related Pages

- [[40_megatron_feature_tree_analysis]] — 功能树总览；本页填的是它 §4 仪表盘里模块 Q 那一行（此前唯一无页可落的模块）
- [[41_megatron_config_surface_analysis]] — §6 说明本模块的开关为何**不在**那里讲的 dataclass 驱动体系内，以及这为什么导致它长期不可见
- [[12_megatron_tp_analysis]] — §2.2 说明 KL 核所消费的词表并行 logits 布局
- [[11_megatron_dataset_analysis]] — §3 的样本流一致性依赖的数据管线在那里
- [[24_megatron_linear_cross_entropy_analysis]] — 另一处"不物化完整 logits"的做法，与本页 §2 的 top-K 截断是同一压力下的两条路
- [[courses/megatron_lm|Megatron-LM 阅读路径]] — 本页位于“分支七：特殊训练、案例与参考”中的入口与前置关系
