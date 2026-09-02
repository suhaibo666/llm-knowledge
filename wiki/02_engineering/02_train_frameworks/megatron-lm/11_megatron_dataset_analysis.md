---
title: "Megatron-LM 数据集深度解析:原始 GPT 数据集与序列打包"
---

# Megatron-LM 数据集深度解析:原始 GPT 数据集与序列打包

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **重定基线**：2026-09-01 由 `71092579`（2026-08-27）推进，跨 7 个提交；该增量只触及 20 个 `megatron/` 文件，本页 `path:line` 引用所涉源文件均不在其中，故无行号漂移，无需逐条重核。
> **重定基线**：2026-08-28 由 `ee3f1ffa…`（2026-05-19）推进，跨 578 个提交；本页全部 `path:line` 形式的引用已在新基线下逐条重核;**代码块内被点名的符号与不带行号的裸路径不在该次扫描口径内**,已知漏网处已于 2026-08-28 单独更正。
> 核心文件:`megatron/core/datasets/` 下 `megatron/core/datasets/gpt_dataset.py`(984 行)、`megatron/core/datasets/indexed_dataset.py`、`megatron/core/datasets/blended_dataset.py`、`megatron/core/datasets/blended_megatron_dataset_builder.py`、`megatron/core/datasets/data_schedule.py`(1166 行);`megatron/core/packed_seq_params.py`
> 配套阅读:`15_megatron_pp_schedulers_analysis.md` §8.1(混合 CP 动态调度)、`13_megatron_cp_analysis.md`
> **叙事顺序**：本页按五拍组织——背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势。
> **最近更新**：2026-08-28。按五拍重排章节顺序；机制正文与既有引用未改。
> 范围:**只讲 LLM(GPT)路径**。`megatron/core/datasets/bert_dataset.py` / `megatron/core/datasets/t5_dataset.py` / `megatron/core/datasets/masked_dataset.py` / `megatron/core/datasets/multimodal_dataset.py` 不展开。

---

## 1. 背景：语料是变长的文档,模型要的是定长 token 序列 —— 于是有了两种"打包"

LLM 训练的数据问题本质是:语料是**变长的文档**,而模型要的是**定长的 token 序列**。怎么把变长 doc 装进定长样本,有两套做法 —— 这正是用户问的"原本的 dataset"和"packed dataset":

| 方式 | 代码 | 思路 | 用于 |
|------|------|------|------|
| **隐式打包**(原始 GPT 数据集) | `megatron/core/datasets/gpt_dataset.py` `GPTDataset` | 所有 doc 首尾相接成一条长流,按 `sequence_length` **切定长** | 预训练 |
| **显式打包**(packed dataset) | `megatron/core/datasets/data_schedule.py` + `PackedSeqParams` | 把若干**完整**变长序列 bin-pack 进定长 buffer,**不切断** | SFT / 变长数据 |

二者都叫"packing",但隐式打包**切断 doc**、靠 EOD+reset 隔离;显式打包**不切断**、靠 `cu_seqlens` 隔离。下面分别拆。

---

## 2. 为什么这么设计：把"变长 → 定长"做成可换的一层,而不是焊死在数据集里

`IndexedDataset` 之上的每一层都在解同一个问题:**语料是变长的,喂给模型的必须是定长的**。Megatron 没有把这一步写成一个固定实现,而是拆成"底座 + 可换的装填策略"。下面四条源码或提交历史给出了理由;第五条源码沉默,由本页重建并整段标为推断。

**① 底座切成 `.bin` + `.idx`,读 `.bin` 的方式是可换的,默认 mmap。**
`IndexedDataset` 的 docstring 把 mmap 写成默认档:「mmap (bool): Whether to mmap the .bin files. Defaults to True.」(`megatron/core/datasets/indexed_dataset.py:619`),默认实现是 `_MMapBinReader`(「A _BinReader that memory maps the data (.bin) file」,`:389-394`)。同一层并排放着三个替代 reader:`_FileBinReader`(「reads from the data (.bin) file using a file pointer」,`:431-436`)、`_S3BinReader`(`:500`)、`_MultiStorageClientBinReader`(`:586`)。元数据(每条 sequence 的元素数与字节偏移指针、每个 document 的 sequence 区间)单独放进 `.idx`,由 `megatron/core/datasets/readme.md` 的 "The index file stores document-level and sequence-level metadata second" 一节定义。
**源码只给出默认值与四个并列实现,没有一句话陈述"为什么选 mmap"** —— 取舍判据由本页读出,见本节末的推断块。

**② sample index 的构建放进 C++ 扩展 —— 被否掉的替代是 Python 版。**
`build_sample_idx` 现在只是 `megatron/core/datasets/helpers.py:12` 的薄包装,实现来自 `helpers_cpp`(`megatron/core/datasets/helpers.py:8-9`),docstring 自陈「Build the 2-D sample index using the properly typed templated C++ function from helpers.cpp」(`megatron/core/datasets/helpers.py:21`)。
**被否掉的替代写在历史里**:提交 `f66c58a9b`(2020-04-07,commit message 即「added build sample index to c++」)把原先的 Python `_build_sample_idx` 换成 `helpers.build_sample_idx`,旧调用被注释掉留在同一处 diff 里。
现存源码还留了一条侧写 —— `megatron/core/datasets/gpt_dataset.py:599-600` 在解释另一件事时点明这一步跑在 C++ 里:「The GIL is held when entering the c++ program, improving the speed of which improves parallelism」。

**③ 最后一个 epoch 不进全局 shuffle —— 被否掉的替代是"所有 epoch 一起 shuffle"。**
`_build_document_index` 的 `separate_final_epoch` 参数,源码自陈语义为「Whether to exclude the last epoch from the global shuffle」(`megatron/core/datasets/gpt_dataset.py:735`);实现是把前 `num_epochs-1` 个 epoch 与最后 1 个 epoch **各自 shuffle 后拼接**,而不是拼好再整体 shuffle(`:749-751`)。开关由一条阈值决定:「Separate the final epoch if it falls below the threshold」+ `threshold = 0.80`(`:556-558`),两侧各有一条不变量兜底(`:551`、`:554`)。
**被否掉的替代写在历史里**:提交 `39181113e`(2020-12-11,commit message 即「Last epoch should not be globally shuffled」)之前,document index 是所有 epoch 一起 shuffle 的。同一提交引入的注释交代了 0.80 的来历:「If we have less than 80% of the samples for the last epoch, seperate out the epoch and treat it differently. Note: the 80% number is just based on common sense and can be adjusted if needed.」—— 即**这个阈值源码自陈只是"常识",从未给出量化依据**;该注释在后续重构中被删,现基线只剩 `threshold = 0.80` 与一行说明。

**④ 建索引缓存时按"访问密度"临时放弃 mmap —— 判据由源码逐条列出。**
`megatron/core/datasets/gpt_dataset.py:593` 用 `len(document_index) * 2 > len(sequence_lengths)` 判断访问密度,高则把 mmap 的 `sequence_lengths` 整份 `copy()` 进内存(`:601`)。理由源码写了两条:「1. We sequentially pre-load the whole file, most of which we expect to read」与「2. The GIL is held when entering the c++ program, improving the speed of which improves parallelism」(`:594-600`)。这条启发式由提交 `9dca04b2c`(2024-05-21,commit message 即「Add a heuristic for data-cache building to improve speed and stability」)引入 —— 在那之前一路走 mmap 惰性读。

**⑤ 显式打包为什么另起 `cu_seqlens` 一套,而不是把 `reset_attention_mask` 扩成打包。**
源码只分别给出两套机制的语义:`reset_attention_mask` 是「Option to reset the attention mask from the dataset」(`megatron/core/datasets/gpt_dataset.py:32-33`),配套的 `create_attention_mask` 说明「Can be disabled if attention kernel generates masks by itself」(`:38-41`);打包侧则由 `PackedSeqParams` 用 `cu_seqlens` + `qkv_format='thd'` 描述边界(`megatron/core/packed_seq_params.py`,见 §5.2)。**两者在源码里从未被并排比较,也没有任何一处说明"为什么不复用 reset 路径"。**

> [!note] 推断
> ① 里"选 mmap 是为了随机读且不把语料驻留内存",以及 ⑤ 里"reset 路径要显式物化 `[s,s]` mask、`cu_seqlens` 路径把边界下推给 kernel,所以变长场景选后者" —— 这两层判断**都是本页从默认值与并列实现反推的,源码没有陈述**。要引用它们,请回到 `megatron/core/datasets/indexed_dataset.py:619`、`:389-394`、`:431-436` 与 `megatron/core/datasets/gpt_dataset.py:32-33`、`:38-41` 这几个 locator,不要引用本段推断。
> 新基线里有一条**间接旁证**:`inter_document_masking` 让 `GPTDataset` 也吐 `cu_seqlens`(`megatron/core/datasets/gpt_dataset.py:95-97`),开启后 `create_attention_mask_in_dataloader` 会被强制关掉,注释写「The dataset omits attention_mask when inter-document masking is enabled」(`megatron/training/arguments.py:1436-1440`)。这说明两条路线正在收敛(见 §8),但它仍不是"当初为什么分开"的源码陈述。

---

## 3. 数据底座:`IndexedDataset`(`.bin` / `.idx`)

最底层接口(`megatron/core/datasets/indexed_dataset.py`,`megatron/core/datasets/readme.md` 有描述)。预处理阶段把语料 tokenize 后写成两个文件:

```
xxx.bin   所有文档的 token id,扁平、连续存(按 dtype 编码)
xxx.idx   元数据:每条 sequence 的元素数、字节偏移指针;
                  每个 document 对应的 sequence 索引范围 [...);
                  (多模态时)每条 sequence 的 mode
```

`IndexedDatasetBuilder` 负责构建与合并;`IndexedDataset` 提供 `.get(doc_id, offset, length)` 随机读。`.bin` 用 mmap,不全载入内存。这是所有上层数据集的共同底座。

---

## 4. 原始 GPT 数据集(`GPTDataset`)

`megatron/core/datasets/gpt_dataset.py:127`。经典 LLM 预训练数据集。

### 4.1 三个索引(`_build_document_sample_shuffle_indices`,`:461`)

`GPTDataset` 的核心是预先构建三层索引,把"定长样本 → 原始 doc 里的 token"映射建好:

```
① document index   1-D。语料的 doc id,重复 num_epochs 遍、并打乱 doc 顺序后的有序数组。
                    (跑几个 epoch、doc 怎么洗,都体现在这里)

② sample index     2-D。把 document index 指向的所有 doc 的 token【首尾相接成一条长流】,
                    每隔 sequence_length 切一刀;每个样本记 (起始 doc 序号, doc 内偏移)。
                    ⇒ 一个样本可以【跨多个 doc】,一个 doc 也可能【被切到两个样本】。

③ shuffle index    1-D。sample index 下标的随机排列(打乱样本喂入顺序)。
```

三个索引算一次后存盘成 `.npy`(`path_to_cache`),后续 **mmap 懒加载**(`:386`),重启训练不重算。`unique_description_hash` 作 key —— 配置变了缓存自动失效。

### 4.2 `__getitem__` 取样(`:251`)

```
idx ──shuffle index──► 打乱后的样本号
    ──sample index──► (起始doc, 偏移) ~ (结束doc, 偏移)
    ──document index + IndexedDataset.get()──► 拼出 sequence_length+1 个 token 的 text

tokens = text[:-1]          ┐  标准 next-token prediction:
labels = text[1:]           ┘  labels 是 tokens 左移一位
_get_ltor_masks_and_position_ids() ──► 因果 mask、loss_mask、position_ids
```

若拼出的 token 不足一个样本(语料尾巴),用 `_pad_token_id` 补齐,并把 padding 处的 `loss_mask` 置 0(不计损失)。

### 4.3 EOD 分隔与 reset —— 隐式打包的"样本内隔离"

隐式打包把不同 doc 接进同一个样本,带来一个问题:样本里 doc B 的 token 会不会 attend 到前面 doc A?默认会 —— 这通常无害(预训练),但可配置隔离:

- **EOD token**:doc 之间插入 end-of-document token,模型能学到"文档边界"。
- **`reset_position_ids`**:每遇 EOD,position id 从 0 重新计 —— doc B 不继承 doc A 的位置。
- **`reset_attention_mask`**:每遇 EOD,attention mask 重置 —— doc B **不 attend** doc A,样本内各 doc 真正独立。
- **`eod_mask_loss`**:EOD 位置的 loss 屏蔽。

开了 reset,一个定长样本里的多个 doc 在数值上等价于独立序列 —— 这其实已经是"打包"的雏形,只是 doc 仍可能被切断。

> `_get_ltor_masks_and_position_ids`(`:786`):`create_attention_mask` 为 False 时**不显式建 `[s,s]` mask**(交给 FlashAttention 等内核隐式处理因果性),省一大块激活。

### 4.4 Blending —— 多源混合(`BlendedDataset`)

`megatron/core/datasets/blended_dataset.py`。实际训练语料来自多个数据源(网页、代码、书……),要按**权重**混合。`BlendedDataset` 按权重从各 `GPTDataset` 采样;`BlendedMegatronDatasetBuilder`(`megatron/core/datasets/blended_megatron_dataset_builder.py`)是统一构造入口,`BlendedMegatronDatasetConfig` / `GPTDatasetConfig` 是配置。

> `megatron/core/datasets/readme.md` 强调:**所有 rank 都要走 builder 构建数据集,否则程序挂起**;真正落盘索引的 rank 由 config 控制(通常 rank 0 建、其余等)。

---

## 5. 序列打包(packed dataset,`megatron/core/datasets/data_schedule.py`)

### 5.1 动机:隐式打包对 SFT / 变长数据不够

`GPTDataset` 的"首尾相接 + 切定长"适合预训练,但对 **SFT / 指令微调 / 变长语料**有两个硬伤:

1. **不能切断样本**:SFT 的一条样本是完整的"指令 + 回答",从中间切断就破坏了语义/标签结构。
2. **逐样本补 padding 浪费巨大**:变长样本若各自 padding 到 `sequence_length`,短样本的 buffer 大半是 padding —— 算力全浪费在 padding 上。

### 5.2 解法:显式打包 + `PackedSeqParams`(THD 格式)

**显式打包**:把若干条**完整、不切断**的变长序列,用 bin-packing 塞进一个定长 buffer。关键是让 attention 内核知道"这个 buffer 里其实是 N 条独立序列" —— 靠 `PackedSeqParams`(`megatron/core/packed_seq_params.py`)描述的 **THD / `cu_seqlens` 格式**:

```
打包 buffer:  [─ 序列A(5) ─][─ 序列B(2) ─][─ 序列C(4) ─][padding(5)]   总长 16

PackedSeqParams:
  cu_seqlens_q = [0, 5, 7, 11]      累积长度:序列 A=[0,5) B=[5,7) C=[7,11)
  max_seqlen_q = 5
  total_tokens = 16
  seq_idx      = [0,0,0,0,0, 1,1, 2,2,2,2, 3,3,3,3,3]   每个 token 属于哪条子序列
  qkv_format   = 'thd'              T(总token)·H(头)·D(头维),无独立 batch 维
```

`seq_idx` 由 `__post_init__` 从 `cu_seqlens` 自动算出(`megatron/core/packed_seq_params.py:41`,用 `repeat_interleave`),供 Mamba mixer 和 CUDA Graph 用。attention 内核(TE)凭 `cu_seqlens` 让**每条子序列只 attend 自己** —— 等价于 N 条独立序列,但只跑一个 kernel、零 padding 浪费(除尾部)。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。**GDN 现也支持序列打包(THD)**(#2645，现在 `megatron/core/ssm/gated_delta_net/gdn.py:189+`；原单文件 `megatron/core/ssm/gated_delta_net.py` 已被 #6088（cherry-pick #5843）拆成 `megatron/core/ssm/gated_delta_net/` 包，THD 分支现落在 `megatron/core/ssm/gated_delta_net/gdn.py`）。此前 GDN 遇到 `packed_seq_params` 直接 `NotImplementedError`;现在 `qkv_format=='thd'` 时按 `cu_seqlens` 把打包 buffer 拆成各子序列、**逐条**做 CP↔HP all-to-all 再 chunk 扫描(要求 `batch==1`、非 deterministic)。即:`cu_seqlens` 这套打包元数据不止给 attention/Mamba,GDN 线性注意力也消费它(详见 `10_megatron_model_structure_analysis.md` §9)。

与 §4.3 GPTDataset 的对比:GPTDataset **切断 doc**、靠 EOD + `reset_attention_mask` 隔离;packed dataset **不切断**、靠 `cu_seqlens` + THD 格式隔离。后者保住了样本完整性。

### 5.3 `megatron/core/datasets/data_schedule.py` 的打包调度

**先说新基线下的形态**:打包模式下**没有 PP 组广播** —— `is_dataset_built_on_rank` 对**每个 PP stage 的 TP-0 rank** 都返回 True,各 stage 自行取数(`megatron/core/datasets/data_schedule.py:245-246` 的 docstring 明写 *There is no PP-group broadcast*;入口判据在 `pretrain_gpt.py:410`、`:415-421`),广播只发生在 TP 组内(`:382` 的 `broadcast_scalars`、`:808-816` 的 `broadcast_tensor`)。下面四条为原表述,原文保留;其中最后一条(`broadcast_to_pp_group`)在新基线下已作废,勘误见其后的 `[!deprecated]`。

`megatron/core/datasets/data_schedule.py` 的 `BasePackingScheduler` / `build_packed_microbatches`:
- 把一批变长样本**贪心 bin-pack** 成定长 microbatch(尽量塞满,减少 padding)。
- 配 `BalancedCPScheduler`(`15_megatron_pp_schedulers_analysis.md` §8.1 的混合 CP 动态调度):变长数据下,长序列分更多 CP 卡,并让各 DP×CP 组的 `seq²/cp` 工作量均衡。
- `PackedSeqParams` 里的 `cp_group` / `local_cp_size` —— 打包**与 CP 协同**:一条打包子序列还能再被 CP 切到多卡(`get_cp_slice_for_thd`)。
- 产出对齐到 PP 组的数据迭代器(`broadcast_to_pp_group`、`create_data_iterator`)。

> [!deprecated] 上面最后一条里的 `broadcast_to_pp_group` 在基线 `71092579` 下已不存在（全仓库符号零命中；由 #4226 “Minor improvements for Dynamic-cp” 从 `megatron/core/datasets/data_schedule_utils.py` 删除）。新基线改成**不再做 PP 组广播**：打包模式下 `is_dataset_built_on_rank` 对每个 PP stage 的 TP-0 rank 都返回 True，各 stage 自行取数，只在 TP 组内广播（`megatron/core/datasets/data_schedule.py:245-246` 的源码注释、`:382` 的 `broadcast_scalars`、`:808-816` 的 `broadcast_tensor`）。该描述对应旧基线 `ee3f1ffa…`；同句里的 `create_data_iterator` 仍存在。

所以，packed dataset 不只是“打包数据”，而是一套由**打包、变长 CP 负载均衡和 microbatch 调度**组成的完整方案，用于支撑 SFT、长文档、RL 等变长场景。

> **打包与动态 CP 的统一流水线**(`DefaultDynamicCPScheduler` 如何作为打包调度器的子类把二者缝在一起)见专文 `29_megatron_packed_dataset_dynamic_cp_analysis.md`。

### 5.4 NEW:`VarlenDataset` 独立入口 + get_batch 统一

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> ee3f1ff 之后,显式打包(SFT/变长)多了一条**独立数据集入口**,并统一了下游取数路径:
>
> - **`VarlenDataset`**(`--use-varlen-dataset`,`megatron/training/datasets/varlen_dataset.py`,#4832):面向 SFT 风格**变长指令数据**的 THD 打包数据集,继承 `SFTDataset` 家族但与 `--sft` 标志**解耦**(不再隐式耦合)。支持多源加载(HuggingFace Hub repo / 本地 `.parquet` / `.jsonl`),并按列名**自动识别四种 schema**:`openai-messages`(`messages`)、`sharegpt`(`conversations`)、`alpaca/dolly`(`instruction|prompt...`+`output|response...`)、`pretrain-text`(`text`,返回裸串无角色掩码);配套 `MockVarlenDataset`(`varlen_mock_dataset_config_json`,与 SFT mock 同 schema 但独立)。`varlen_sbhd_validation` 提供一个 SBHD 参考路径,跳过打包、用于 THD 数值正确性校验。
> - **每 index 直接吐一条子样本**:不同于 `SFTDataset`(一条样本里 `cu_seqlens` 拼了多条子序列),`VarlenDataset` 每个样本已是单条、自带 `padded_seq_len`。于是打包流水线第④步前的 `_unpack_batch`(`megatron/core/datasets/data_schedule_utils.py:105`)现支持**两种输入**:预打包(切开 `cu_seqlens`)与已拆开(只归一 batch 维、缺 `original_seq_len` 时从 `padded_seq_len` 补)。
> - **get_batch 统一 + SFT THD 支持 PP**(#4103):原先散落 `megatron/training/utils.py` 的取数逻辑收敛进 `megatron/core/utils.py`,按场景拆成 `get_batch_on_this_tp_rank`(`:2052`,长度前缀协议广播 `cu_seqlens`、动态 CP 的 `local_cp_size`)、`get_sft_batch_on_this_cp_rank`（#5403 后更名为 `_get_batch_on_this_cp_rank_per_document_balancing`，`:2339`）、`get_pretrain_batch_on_this_cp_rank`（#5403 后更名为 `_get_batch_on_this_cp_rank_per_sequence_balancing`，`:2389`；两者现由新增的分发器 `get_batch_on_this_cp_rank`（`:2545`）统一调用）、`get_thd_batch_on_this_cp_rank`(`:2608`)。**关键能力:SFT 的 THD 打包现可与 PP 共用**(PP 中间 stage **不靠广播**:打包模式下 `is_dataset_built_on_rank` 对每个 PP stage 的 TP-0 rank 都返回 True,各 stage 自行建数据集、自己拿到打包元数据;源码给出的理由是「Packed THD and SBHD validation both need padding metadata on every pipeline stage so each MoE layer excludes physical padding」—— `pretrain_gpt.py:410`、`:415-421`,另见 `megatron/core/datasets/data_schedule.py:245-246`)。

---

## 6. 两者对比

| 维度 | 原始 GPTDataset(隐式打包) | packed dataset(显式打包) |
|------|---------------------------|---------------------------|
| 装填方式 | doc 首尾相接 → 切定长 | 完整变长序列 → bin-pack 进定长 buffer |
| 是否切断样本 | **会**(doc 可能跨样本) | **不会** |
| 样本内隔离 | EOD token + `reset_attention_mask`/`reset_position_ids` | `cu_seqlens` + THD 格式(`PackedSeqParams`) |
| padding 浪费 | 几乎无(只有语料尾巴) | 几乎无(bin-pack 尽量塞满) |
| 索引 | 三级索引(document/sample/shuffle),落盘 mmap | 打包调度器在线 bin-pack |
| 典型用途 | **预训练** | **SFT / 变长 / 长文档 / RL** |
| 与 CP 关系 | 定长,标准 CP | 变长,配 `BalancedCPScheduler` 动态 CP |

二者底座相同(`IndexedDataset` 的 `.bin`/`.idx`),区别在"变长 → 定长"那一步的策略。

---

## 7. 约束

每一层都有前提、代价与失效条件,以下逐条带 locator。

**7.1 配置层的必填与互斥。**
`GPTDatasetConfig.__post_init__` 把四项设成硬前提:`tokenizer`、`reset_position_ids`、`reset_attention_mask`、`eod_mask_loss` 都不能为 `None`(`megatron/core/datasets/gpt_dataset.py:103-107`)。`--varlen-sbhd-validation` 与 `--dynamic-context-parallel` **互斥**,报错语句自陈原因「SBHD mode is not packed」(`:109-113`)。`--per-dataset-sequences-path` 要求 tokenizer 报得出 `vocab_size`,否则构造期 assert 失败(`:120-124`)。
动态 CP 还对**样本长度本身**提整除要求 —— config docstring 写 `each sample should be divisible by the data parallel size * context parallel size * 2`,开 SP 时再乘一个 `sequence_parallel_size`(`:62-67`)。

**7.2 三级索引的前提与代价。**
索引构建强制 `int32`:`assert document_index.dtype == numpy.int32`、`assert self.dataset.sequence_lengths.dtype == numpy.int32`(`:591-592`)。缓存 key 由 `unique_description_hash` 拼出(`:498-499`),**任何配置改动都让三份 `.npy` 整体失效、全部重算**。若 `path_to_cache` 为 `None`,索引**不落盘**,只打一条 WARNING(`:632-637`)—— 每次重启都要重建。
末 epoch 的两条不变量越界即 assert 失败:`num_samples_from_final_epoch >= 0`(`:551`)与 `<= num_samples_per_epoch + 1`(`:554`)。

**7.3 底座的一致性校验与对象存储限制。**
`IndexedDataset.__init__` 建完索引后校验三条计数一致(`megatron/core/datasets/indexed_dataset.py:673-676`),但这三条 assert 在 `fast_cache_load=True` 时**被整体跳过**(`:673`)—— 快缓存模式拿一致性换启动速度。S3 / MSC 数据源下 **`mmap` 必须关掉**,docstring 明写「Note that `mmap` must be disabled for S3 data loading」(`:621-625`)。

**7.4 Blending 的硬上限。**
`BlendedDataset.__init__` 要求子数据集数 **`< 32767`**(`megatron/core/datasets/blended_dataset.py:50`)、所有子数据集**同类型、同 split**(`:51-52`)、权重同类型且全为正(`:53-54`);`size is None` 且权重为浮点时,权重必须是整值(`:55-56`)。只有一个数据集时给 WARNING 提示不必 blend(`:59-62`)。
`megatron/core/datasets/readme.md` 的 "NB" 是全局前提:**所有 rank 都必须走 builder,否则程序挂起**。

**7.5 显式打包路径的边界。**
打包调度只在 **TP-0** 上持有 data_iterator,其余 rank 断言 `data_iterator is None`(`megatron/core/datasets/data_schedule.py:307`、`:624`);required keys 缺一个即 assert(`:622`)。`thd_max_packed_sequences` 必须 `>= 1`(`:469-470`);当 THD padding 会追加 dummy 序列时必须 `>= 2`,报错语句自陈原因是该上限**把那条 dummy 序列也算在内**(`:476-481`)。zigzag CP 切分要求打包总长能被 `2 * cp_size` 整除(`megatron/core/packed_seq_params.py:287-290`)。

**7.6 故意不做的事。**
`GPTDataset` **不保证样本内 doc 完整** —— 一个 doc 可以被切到两个样本(§4.1)。这是隐式打包的定义性代价:`GPTDatasetConfig` 的全部字段里(`megatron/core/datasets/gpt_dataset.py:26-97`)没有任何"不切断 doc"的开关,要不切断只能换到 §5 的显式打包路径。
反向的代价同样存在:显式打包的贪心装填**按整条样本入桶、从不切分单条样本**(`megatron/core/datasets/data_schedule.py:165-185`),因此 `GPTDataset` 那种"一个样本可跨多个 doc、token 流连续"的装填在打包路径上不复存在。

---

## 8. 发展趋势

以下四条都锚在基线 `71092579` 的实读位置或提交历史,并**标为推断** —— 它们是源码里在途的改动,不是路线声明。

**① 隐式打包正在改用显式打包的隔离机制。**
`GPTDatasetConfig.inter_document_masking` 让 `GPTDataset` 也返回 `cu_seqlens`:「When True, return cu_seqlens marking document boundaries within each sample so that attention is restricted to individual documents」(`megatron/core/datasets/gpt_dataset.py:95-97`);`__getitem__` 里据 `document_lengths` 算出 `cu_seqlens` 与 `max_seqlen`,并**逐 document 重置 position_ids**(`:308-335`)。开启后 `create_attention_mask_in_dataloader` 被强制关掉,注释自陈「The dataset omits attention_mask when inter-document masking is enabled; disable the flag to avoid a TP broadcast mismatch」(`megatron/training/arguments.py:1436-1440`)。该特性由 #5298(`f88b85f8c`)引入,#5635(`bf32f4415`,commit message 即「Fix inter-document masking crash and NaNs with TP > 1 and micro_batch_size > 1」)修 bug。
→ **推断**:§6 对照表里"样本内隔离"一行的两栏正在合并 —— 隐式打包侧的 `reset_attention_mask` 有被 `cu_seqlens` 取代的趋势。这是本页从上述三处读出的方向,源码没有声明要弃用 `reset_attention_mask`。

**② 对象存储配置在做一次改名收敛。**
`S3Config` 已被别名成 `ObjectStorageConfig`,注释写「S3Config is deprecated, use ObjectStorageConfig instead」(`megatron/core/datasets/object_storage_utils.py:42-43`);`IndexedDataset.__init__` 保留 `s3_config` 形参只为兼容,同处注释重复了这条弃用(`megatron/core/datasets/indexed_dataset.py:654-655`)。
→ **推断**:`s3_config` 这个入参会消失;源码只写了 "deprecated",没有给移除时间点。

**③ `IndexedDataset` 的 document-indices 读写接口已标待废。**
`get_document_indices` 与 `set_document_indices` 的 docstring 都写着「This method is slated for deprecation.」(`megatron/core/datasets/indexed_dataset.py:892`、`:902`),而同一份数据已由 `document_indices` property 暴露(`:880-887`)。
→ **推断**:这两个方法会被 property 完全取代。

**④ THD 路径仍在用一条 SBHD 参考路径做数值校验。**
`varlen_sbhd_validation` 的 docstring 自陈用途:「Used to obtain a SBHD reference run that mirrors the THD path's tokenization but skips all packing — useful for THD numerical-correctness validation」(`megatron/core/datasets/gpt_dataset.py:88-93`)。
→ **推断**:显式打包(THD)的数值正确性仍在被主动验证,这条脚手架的存在本身说明该路径尚未被当作完全稳定 —— 这是本页的读法,源码只陈述了该开关的用途。

---

## 9. 小结

- **数据底座**:`IndexedDataset` 的 `.bin`(扁平 token)+ `.idx`(doc/sequence 元数据),mmap 随机读。
- **原始 GPT 数据集**(隐式打包,预训练):三级索引(document→sample→shuffle)把所有 doc 首尾相接、按 `sequence_length` 切定长;一个样本可跨 doc、一个 doc 可被切断;EOD + `reset_*` 做样本内隔离;`tokens/labels` 错位即 next-token 标签;`BlendedDataset` 按权重混多源;索引落盘缓存。
- **packed dataset**(显式打包,SFT/变长):把完整变长序列 bin-pack 进定长 buffer、**不切断**,用 `PackedSeqParams` 的 `cu_seqlens`/THD 格式让 attention 内核把一个 buffer 当 N 条独立序列;`megatron/core/datasets/data_schedule.py` 还把打包与混合 CP 负载均衡、microbatch 调度合在一起。
- **核心区别**:隐式打包切断 doc、靠 EOD+reset;显式打包不切断、靠 `cu_seqlens`。预训练用前者,SFT/变长用后者。

---

*生成依据:`Megatron-LM` `dev` 分支 `71092579`（2026-08-27）。源码行号以该 commit 为准；2026-08-28 由 `ee3f1ff` 重定基线。本文只覆盖 GPT/LLM 路径;BERT/T5/多模态数据集见 `megatron/core/datasets/bert_dataset.py` / `megatron/core/datasets/t5_dataset.py` / `megatron/core/datasets/multimodal_dataset.py`。配套文档:`15_megatron_pp_schedulers_analysis.md` §8.1、`13_megatron_cp_analysis.md`。*

## Related Pages

- [[29_megatron_packed_dataset_dynamic_cp_analysis]] · [[15_megatron_pp_schedulers_analysis]] · [[13_megatron_cp_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
