---
title: "Megatron-LM 数据集深度解析:原始 GPT 数据集与序列打包"
---

# Megatron-LM 数据集深度解析:原始 GPT 数据集与序列打包

> 代码基准:`Megatron-LM/` 子仓库 `dev` 分支,commit `ee3f1ff`
> 核心文件:`megatron/core/datasets/` 下 `gpt_dataset.py`(907 行)、`indexed_dataset.py`、`blended_dataset.py`、`blended_megatron_dataset_builder.py`、`data_schedule.py`(954 行);`megatron/core/packed_seq_params.py`
> 配套阅读:`15_megatron_pp_schedulers_analysis.md` §6.1(混合 CP 动态调度)、`13_megatron_cp_analysis.md`
> 范围:**只讲 LLM(GPT)路径**。`bert_dataset.py` / `t5_dataset.py` / `masked_dataset.py` / `multimodal_dataset.py` 不展开。

---

## 0. 总览:两种"打包"

LLM 训练的数据问题本质是:语料是**变长的文档**,而模型要的是**定长的 token 序列**。怎么把变长 doc 装进定长样本,有两套做法 —— 这正是用户问的"原本的 dataset"和"packed dataset":

| 方式 | 代码 | 思路 | 用于 |
|------|------|------|------|
| **隐式打包**(原始 GPT 数据集) | `gpt_dataset.py` `GPTDataset` | 所有 doc 首尾相接成一条长流,按 `sequence_length` **切定长** | 预训练 |
| **显式打包**(packed dataset) | `data_schedule.py` + `PackedSeqParams` | 把若干**完整**变长序列 bin-pack 进定长 buffer,**不切断** | SFT / 变长数据 |

二者都叫"packing",但隐式打包**切断 doc**、靠 EOD+reset 隔离;显式打包**不切断**、靠 `cu_seqlens` 隔离。下面分别拆。

---

## 1. 数据底座:`IndexedDataset`(`.bin` / `.idx`)

最底层接口(`indexed_dataset.py`,`readme.md` 有描述)。预处理阶段把语料 tokenize 后写成两个文件:

```
xxx.bin   所有文档的 token id,扁平、连续存(按 dtype 编码)
xxx.idx   元数据:每条 sequence 的元素数、字节偏移指针;
                  每个 document 对应的 sequence 索引范围 [...);
                  (多模态时)每条 sequence 的 mode
```

`IndexedDatasetBuilder` 负责构建与合并;`IndexedDataset` 提供 `.get(doc_id, offset, length)` 随机读。`.bin` 用 mmap,不全载入内存。这是所有上层数据集的共同底座。

---

## 2. 原始 GPT 数据集(`GPTDataset`)

`gpt_dataset.py:104`。经典 LLM 预训练数据集。

### 2.1 三个索引(`_build_document_sample_shuffle_indices`,`:384`)

`GPTDataset` 的核心是预先构建三层索引,把"定长样本 → 原始 doc 里的 token"映射建好:

```
① document index   1-D。语料的 doc id,重复 num_epochs 遍、并打乱 doc 顺序后的有序数组。
                    (跑几个 epoch、doc 怎么洗,都体现在这里)

② sample index     2-D。把 document index 指向的所有 doc 的 token【首尾相接成一条长流】,
                    每隔 sequence_length 切一刀;每个样本记 (起始 doc 序号, doc 内偏移)。
                    ⇒ 一个样本可以【跨多个 doc】,一个 doc 也可能【被切到两个样本】。

③ shuffle index    1-D。sample index 下标的随机排列(打乱样本喂入顺序)。
```

三个索引算一次后存盘成 `.npy`(`path_to_cache`),后续 **mmap 懒加载**(`:312`),重启训练不重算。`unique_description_hash` 作 key —— 配置变了缓存自动失效。

### 2.2 `__getitem__` 取样(`:228`)

```
idx ──shuffle index──► 打乱后的样本号
    ──sample index──► (起始doc, 偏移) ~ (结束doc, 偏移)
    ──document index + IndexedDataset.get()──► 拼出 sequence_length+1 个 token 的 text

tokens = text[:-1]          ┐  标准 next-token prediction:
labels = text[1:]           ┘  labels 是 tokens 左移一位
_get_ltor_masks_and_position_ids() ──► 因果 mask、loss_mask、position_ids
```

若拼出的 token 不足一个样本(语料尾巴),用 `_pad_token_id` 补齐,并把 padding 处的 `loss_mask` 置 0(不计损失)。

### 2.3 EOD 分隔与 reset —— 隐式打包的"样本内隔离"

隐式打包把不同 doc 接进同一个样本,带来一个问题:样本里 doc B 的 token 会不会 attend 到前面 doc A?默认会 —— 这通常无害(预训练),但可配置隔离:

- **EOD token**:doc 之间插入 end-of-document token,模型能学到"文档边界"。
- **`reset_position_ids`**:每遇 EOD,position id 从 0 重新计 —— doc B 不继承 doc A 的位置。
- **`reset_attention_mask`**:每遇 EOD,attention mask 重置 —— doc B **不 attend** doc A,样本内各 doc 真正独立。
- **`eod_mask_loss`**:EOD 位置的 loss 屏蔽。

开了 reset,一个定长样本里的多个 doc 在数值上等价于独立序列 —— 这其实已经是"打包"的雏形,只是 doc 仍可能被切断。

> `_get_ltor_masks_and_position_ids`(`:709`):`create_attention_mask` 为 False 时**不显式建 `[s,s]` mask**(交给 FlashAttention 等内核隐式处理因果性),省一大块激活。

### 2.4 Blending —— 多源混合(`BlendedDataset`)

`blended_dataset.py`。实际训练语料来自多个数据源(网页、代码、书……),要按**权重**混合。`BlendedDataset` 按权重从各 `GPTDataset` 采样;`BlendedMegatronDatasetBuilder`(`blended_megatron_dataset_builder.py`)是统一构造入口,`BlendedMegatronDatasetConfig` / `GPTDatasetConfig` 是配置。

> `readme.md` 强调:**所有 rank 都要走 builder 构建数据集,否则程序挂起**;真正落盘索引的 rank 由 config 控制(通常 rank 0 建、其余等)。

---

## 3. 序列打包(packed dataset,`data_schedule.py`)

### 3.1 动机:隐式打包对 SFT / 变长数据不够

`GPTDataset` 的"首尾相接 + 切定长"适合预训练,但对 **SFT / 指令微调 / 变长语料**有两个硬伤:

1. **不能切断样本**:SFT 的一条样本是完整的"指令 + 回答",从中间切断就破坏了语义/标签结构。
2. **逐样本补 padding 浪费巨大**:变长样本若各自 padding 到 `sequence_length`,短样本的 buffer 大半是 padding —— 算力全浪费在 padding 上。

### 3.2 解法:显式打包 + `PackedSeqParams`(THD 格式)

**显式打包**:把若干条**完整、不切断**的变长序列,用 bin-packing 塞进一个定长 buffer。关键是让 attention 内核知道"这个 buffer 里其实是 N 条独立序列" —— 靠 `PackedSeqParams`(`packed_seq_params.py`)描述的 **THD / `cu_seqlens` 格式**:

```
打包 buffer:  [─ 序列A(5) ─][─ 序列B(2) ─][─ 序列C(4) ─][padding(5)]   总长 16

PackedSeqParams:
  cu_seqlens_q = [0, 5, 7, 11]      累积长度:序列 A=[0,5) B=[5,7) C=[7,11)
  max_seqlen_q = 5
  total_tokens = 16
  seq_idx      = [0,0,0,0,0, 1,1, 2,2,2,2, 3,3,3,3,3]   每个 token 属于哪条子序列
  qkv_format   = 'thd'              T(总token)·H(头)·D(头维),无独立 batch 维
```

`seq_idx` 由 `__post_init__` 从 `cu_seqlens` 自动算出(`packed_seq_params.py:28`,用 `repeat_interleave`),供 Mamba mixer 和 CUDA Graph 用。attention 内核(TE)凭 `cu_seqlens` 让**每条子序列只 attend 自己** —— 等价于 N 条独立序列,但只跑一个 kernel、零 padding 浪费(除尾部)。

> [!update] 2026-06-16 · dev@232c478d4:**GDN 现也支持序列打包(THD)**(#2645,`ssm/gated_delta_net.py:340+`)。此前 GDN 遇到 `packed_seq_params` 直接 `NotImplementedError`;现在 `qkv_format=='thd'` 时按 `cu_seqlens` 把打包 buffer 拆成各子序列、**逐条**做 CP↔HP all-to-all 再 chunk 扫描(要求 `batch==1`、非 deterministic)。即:`cu_seqlens` 这套打包元数据不止给 attention/Mamba,GDN 线性注意力也消费它(详见 `10_megatron_model_structure_analysis.md` §7)。

与 §2.3 GPTDataset 的对比:GPTDataset **切断 doc**、靠 EOD + `reset_attention_mask` 隔离;packed dataset **不切断**、靠 `cu_seqlens` + THD 格式隔离。后者保住了样本完整性。

### 3.3 `data_schedule.py` 的打包调度

`data_schedule.py` 的 `BasePackingScheduler` / `build_packed_microbatches`:
- 把一批变长样本**贪心 bin-pack** 成定长 microbatch(尽量塞满,减少 padding)。
- 配 `BalancedCPScheduler`(`15_megatron_pp_schedulers_analysis.md` §6.1 的混合 CP 动态调度):变长数据下,长序列分更多 CP 卡,并让各 DP×CP 组的 `seq²/cp` 工作量均衡。
- `PackedSeqParams` 里的 `cp_group` / `local_cp_size` —— 打包**与 CP 协同**:一条打包子序列还能再被 CP 切到多卡(`get_cp_slice_for_thd`)。
- 产出对齐到 PP 组的数据迭代器(`broadcast_to_pp_group`、`create_data_iterator`)。

所以，packed dataset 不只是“打包数据”，而是一套由**打包、变长 CP 负载均衡和 microbatch 调度**组成的完整方案，用于支撑 SFT、长文档、RL 等变长场景。

> **打包与动态 CP 的统一流水线**(`DefaultDynamicCPScheduler` 如何作为打包调度器的子类把二者缝在一起)见专文 `29_megatron_packed_dataset_dynamic_cp_analysis.md`。

### 3.4 NEW:`VarlenDataset` 独立入口 + get_batch 统一

> [!update] 2026-06-16 · dev@232c478d4
> ee3f1ff 之后,显式打包(SFT/变长)多了一条**独立数据集入口**,并统一了下游取数路径:
>
> - **`VarlenDataset`**(`--use-varlen-dataset`,`megatron/training/datasets/varlen_dataset.py`,#4832):面向 SFT 风格**变长指令数据**的 THD 打包数据集,继承 `SFTDataset` 家族但与 `--sft` 标志**解耦**(不再隐式耦合)。支持多源加载(HuggingFace Hub repo / 本地 `.parquet` / `.jsonl`),并按列名**自动识别四种 schema**:`openai-messages`(`messages`)、`sharegpt`(`conversations`)、`alpaca/dolly`(`instruction|prompt...`+`output|response...`)、`pretrain-text`(`text`,返回裸串无角色掩码);配套 `MockVarlenDataset`(`varlen_mock_dataset_config_json`,与 SFT mock 同 schema 但独立)。`varlen_sbhd_validation` 提供一个 SBHD 参考路径,跳过打包、用于 THD 数值正确性校验。
> - **每 index 直接吐一条子样本**:不同于 `SFTDataset`(一条样本里 `cu_seqlens` 拼了多条子序列),`VarlenDataset` 每个样本已是单条、自带 `padded_seq_len`。于是打包流水线第④步前的 `_unpack_batch`(`data_schedule_utils.py:48`)现支持**两种输入**:预打包(切开 `cu_seqlens`)与已拆开(只归一 batch 维、缺 `original_seq_len` 时从 `padded_seq_len` 补)。
> - **get_batch 统一 + SFT THD 支持 PP**(#4103):原先散落 `training/utils.py` 的取数逻辑收敛进 `megatron/core/utils.py`,按场景拆成 `get_batch_on_this_tp_rank`(`:1992`,长度前缀协议广播 `cu_seqlens`、动态 CP 的 `local_cp_size`)、`get_sft_batch_on_this_cp_rank`(`:2269`)、`get_pretrain_batch_on_this_cp_rank`(`:2321`)、`get_thd_batch_on_this_cp_rank`(`:2439`)。**关键能力:SFT 的 THD 打包现可与 PP 共用**(PP 中间 stage 经 broadcast 拿到打包元数据)。

---

## 4. 两者对比

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

## 5. 小结

- **数据底座**:`IndexedDataset` 的 `.bin`(扁平 token)+ `.idx`(doc/sequence 元数据),mmap 随机读。
- **原始 GPT 数据集**(隐式打包,预训练):三级索引(document→sample→shuffle)把所有 doc 首尾相接、按 `sequence_length` 切定长;一个样本可跨 doc、一个 doc 可被切断;EOD + `reset_*` 做样本内隔离;`tokens/labels` 错位即 next-token 标签;`BlendedDataset` 按权重混多源;索引落盘缓存。
- **packed dataset**(显式打包,SFT/变长):把完整变长序列 bin-pack 进定长 buffer、**不切断**,用 `PackedSeqParams` 的 `cu_seqlens`/THD 格式让 attention 内核把一个 buffer 当 N 条独立序列;`data_schedule.py` 还把打包与混合 CP 负载均衡、microbatch 调度合在一起。
- **核心区别**:隐式打包切断 doc、靠 EOD+reset;显式打包不切断、靠 `cu_seqlens`。预训练用前者,SFT/变长用后者。

---

*生成依据:`Megatron-LM` `dev` 分支 `ee3f1ff`。源码行号以该 commit 为准。本文只覆盖 GPT/LLM 路径;BERT/T5/多模态数据集见 `bert_dataset.py` / `t5_dataset.py` / `multimodal_dataset.py`。配套文档:`15_megatron_pp_schedulers_analysis.md` §6.1、`13_megatron_cp_analysis.md`。*

## Related Pages

- [[29_megatron_packed_dataset_dynamic_cp_analysis]] · [[15_megatron_pp_schedulers_analysis]] · [[13_megatron_cp_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
