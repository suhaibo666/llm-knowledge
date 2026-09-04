---
title: "Megatron-LM 数据入口深度解析：分词、GPT 数据集与序列打包"
---

# Megatron-LM 数据入口深度解析：分词、GPT 数据集与序列打包

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **重定基线**：2026-09-01 由 `71092579`（2026-08-27）推进，跨 7 个提交；该增量只触及 20 个 `megatron/` 文件，本页 `path:line` 引用所涉源文件均不在其中，故无行号漂移，无需逐条重核。
> **重定基线**：2026-08-28 由 `ee3f1ffa…`（2026-05-19）推进，跨 578 个提交；本页全部 `path:line` 形式的引用已在新基线下逐条重核;**代码块内被点名的符号与不带行号的裸路径不在该次扫描口径内**,已知漏网处已于 2026-08-28 单独更正。
> 核心文件：`megatron/core/tokenizers/**`；`megatron/training/config/training_config.py` 的 `TokenizerConfig`；`megatron/core/datasets/` 下 `gpt_dataset.py`、`indexed_dataset.py`、`blended_dataset.py`、`blended_megatron_dataset_builder.py`、`data_schedule.py`；`megatron/core/packed_seq_params.py`。
> 配套阅读:`15_megatron_pp_schedulers_analysis.md` §8.1(混合 CP 动态调度)、`13_megatron_cp_analysis.md`
> **叙事顺序**：先建立“变长语料 → 定长训练样本”的问题与设计判据，再回到文本 → token 的输入边界，随后沿索引底座走到定长预训练与变长打包。
> **最近更新**：2026-09-03。吸收原 44 页的 tokenizer 机制与 20 字段配置契约，修正 `build_tokenizer` 的选路描述，并把它放回数据入口。
> 范围：数据集部分只讲 LLM（GPT）路径；tokenizer 部分覆盖统一工厂、文本/视觉库族、chat template 与 parser，但不逐个比较第三方分词算法。BERT/T5/多模态数据集本体不展开。

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
源码只分别给出两套机制的语义:`reset_attention_mask` 是「Option to reset the attention mask from the dataset」(`megatron/core/datasets/gpt_dataset.py:32-33`),配套的 `create_attention_mask` 说明「Can be disabled if attention kernel generates masks by itself」(`:38-41`);打包侧则由 `PackedSeqParams` 用 `cu_seqlens` + `qkv_format='thd'` 描述边界(`megatron/core/packed_seq_params.py`,见 §6.2)。**两者在源码里从未被并排比较,也没有任何一处说明"为什么不复用 reset 路径"。**

> [!note] 推断
> ① 里"选 mmap 是为了随机读且不把语料驻留内存",以及 ⑤ 里"reset 路径要显式物化 `[s,s]` mask、`cu_seqlens` 路径把边界下推给 kernel,所以变长场景选后者" —— 这两层判断**都是本页从默认值与并列实现反推的,源码没有陈述**。要引用它们,请回到 `megatron/core/datasets/indexed_dataset.py:619`、`:389-394`、`:431-436` 与 `megatron/core/datasets/gpt_dataset.py:32-33`、`:38-41` 这几个 locator,不要引用本段推断。
> 新基线里有一条**间接旁证**:`inter_document_masking` 让 `GPTDataset` 也吐 `cu_seqlens`(`megatron/core/datasets/gpt_dataset.py:95-97`),开启后 `create_attention_mask_in_dataloader` 会被强制关掉,注释写「The dataset omits attention_mask when inter-document masking is enabled」(`megatron/training/arguments.py:1436-1440`)。这说明两条路线正在收敛(见 §9),但它仍不是"当初为什么分开"的源码陈述。

---

## 3. 文本到 token：统一工厂、实现库与模型约定

Tokenizer 是数据链的第一道可执行边界：原始文本先在这里变成 token id，之后才进入 `IndexedDataset`。把它单独留在“训练结束后的导出”旁边会割裂读者的因果链；本节因此只讲它怎样把外部文本协议归一成数据集可消费的整数序列。

### 3.1 目录结构：实现库与模型约定是两个维度

`megatron/core/tokenizers/` 的结构把“怎么编码”与“哪个模型家族”拆开：

```text
tokenizers/
├── megatron_tokenizer.py      # metadata 驱动的统一恢复入口
├── text/
│   ├── libraries/             # SentencePiece、HF、TikToken、Null、SFT 等实现
│   ├── models/                # GPT、BERT、T5、Mamba 与默认模型约定
│   └── parsers/               # 生成文本的 reasoning / tool-call 解析
├── vision/                    # 多模态与 Null 多模态实现
└── utils/build_tokenizer.py   # 训练参数到统一入口的适配层
```

源码没有把这套布局命名为“正交分类”；但从 `TEXT_LIBRARIES` / `VISION_LIBRARIES` 与 `TOKENIZER_MAPPING_NAMES` 两张独立映射可见，library 决定底层实现，`model_type` 决定模型包装类（`megatron/core/tokenizers/megatron_tokenizer.py:12-32`、`:188-213`）。**分析结论**：新增一种库或一个模型约定不必复制另一维的全部组合，这正是拆成两层的工程收益。

### 3.2 两段分派链：显式类型选参数分支，非 Null 分支可由 metadata 覆盖 library

训练侧实际入口是 `build_tokenizer(args)`。它先要求 `args.tokenizer_type` 属于 `SUPPORTED_TOKENIZERS`，否则直接 `ValueError`（`megatron/core/tokenizers/utils/build_tokenizer.py:26-32`）；随后按这个**显式类型**选择参数分支、构造对应 `kwargs`，并算出默认 library（`:34-86`）。普通文本、多模态与 SFT 分支没有 `metadata_path` 时，把默认值包装成 `{'library': tokenizer_library}`；若用户给了外部 metadata 文件，则把该路径传下去（`:109-121`）。

两个 Null 分支是明确例外：`NullTokenizer` / `NullMultimodalTokenizer` 在分支内直接构造 `{'library': 'null-text'}` 或 `{'library': 'null-multimodal'}`，调用 `from_pretrained()` 后提前返回（`:87-107`）。它们不会走到 `args.metadata_path` 的判断，因此忽略外部 metadata。

`MegatronTokenizer` 本身不能直接实例化：构造函数明确抛异常，要求走 `from_pretrained()`（`megatron/core/tokenizers/megatron_tokenizer.py:37-46`）。后者读取**实际 metadata 内容**的 `library`，检查 tokenizer 路径与多模态必填项，再由 `_get_tokenizer_model_class` 选具体模型包装类（`:62-100`、`:188-213`）。因此对**非 Null 分支**，提供外部 `metadata_path` 时，文件里的 library 才是最终 class 分派依据；`build_tokenizer` 没有检查它是否与前面按 `tokenizer_type` 选出的 kwargs 分支一致。`write_metadata()` 则执行反向操作：校验 library、据其归类为 text/vision，并持久化 library、class、model type 与 chat template（`:104-165`）。

> [!correction] 选路契约
> 原合并页把这条链写成“从零散路径参数反推 library”，这不成立。当前冻结基线对**非 Null 分支**采用两级判据：`tokenizer_type` 先选择 kwargs 分支与**默认** library；没有外部 metadata 时，它也决定最终 library；一旦提供 `metadata_path`，文件内的 `metadata['library']` 可以覆盖默认 library，而 kwargs 仍来自 `tokenizer_type` 分支。两者不一致没有专门校验，可能在后续路径/必填参数检查或具体构造器中失败。`NullTokenizer` / `NullMultimodalTokenizer` 则在读取 `args.metadata_path` 之前固定 null library 并返回；模型路径缺失也不会自动切换到另一 library。

### 3.3 词表 padding：让词表维能被 TP 均分

构建完成后，只要 `pad_vocab_size=True` 且尚未设置 `padded_vocab_size`，`_set_padded_vocab_size` 就调用 `vocab_size_with_padding`（`megatron/core/tokenizers/utils/build_tokenizer.py:103-105`、`:117-143`）。计算式是向上取整到

$$
\texttt{make\_vocab\_size\_divisible\_by}\times\texttt{tensor\_model\_parallel\_size}
$$

的整数倍（`:124-136`）。这保证词表维可以在 TP ranks 间等分；具体的 `VocabParallelEmbedding` 切分见 [[12_megatron_tp_analysis]]。

`megatron/training/vocab_utils.py` 还有同一计算的查询版本：`calculate_padded_vocab_size` 把纯计算放进 `@lru_cache(maxsize=128)` 的 helper，却把日志留在缓存外（`megatron/training/vocab_utils.py:9-45`）。因此 core 入口负责构建后写回 `args`，training helper 适合模型构建时重复查询；这是两个调用语境，不应把“存在两份实现”误读成两套不同公式。

### 3.4 chat template、输出 parser 与视觉分支

`text/libraries/chat_template.py` 负责把会话消息格式化成模型约定文本；`text/parsers/` 则处理反方向输出，当前具体 parser 包括 DeepSeek-R1 reasoning 与 Qwen3-Coder tool call。推理服务的 `ServeConfig.parsers` 暴露 parser 列表（`megatron/core/inference/apis/serve_config.py:29`），所以这棵子树的职责不止 encode/decode，也覆盖生成结果的结构化边界。

视觉侧与文本侧并列：`vision/libraries/` 提供 multimodal/null-multimodal，实现经同一个 `MegatronTokenizer.from_pretrained` 入口分派；多模态分支还强制要求 `prompt_format`、`special_tokens` 与 `image_tag_type`（`megatron/core/tokenizers/megatron_tokenizer.py:83-95`）。

### 3.5 从 text 到 `.bin/.idx`：离线预处理的实际接缝

`tools/preprocess_data.py` 给出了 tokenizer 与 §4 `IndexedDataset` 之间的真实 caller hop，而不只是概念相邻：

```text
JSON line
  → Encoder.initializer: build_tokenizer(args)
  → Encoder.encode: tokenizer.tokenize(sentence) [+ EOD]
  → (doc_ids, sentence_lens)
  → IndexedDatasetBuilder.add_document(...)
  → builder.finalize(idx_path)
  → output.bin + output.idx
```

worker 初始化时构建 tokenizer（`tools/preprocess_data.py:51-80`），`Encoder.encode` 对每个句子调用 `tokenize`、可选追加 EOD，并返回 token ids 与句长（`:95-117`）。主进程用 tokenizer 的 `vocab_size` 选择 `.bin` dtype、创建 builder（`:162-187`），再逐文档 `add_document`，最后 `finalize` 对应 `.idx`（`:193-204`）。这条路径闭合了“文本怎样成为 §4 随机读底座”的执行链。

它是 `IndexedDataset` 的离线 JSON 预处理入口，不代表 §6 的 Varlen/HuggingFace 数据路径也必须先落成同一组 `.bin/.idx`；后者有自己的运行时数据入口。

### 3.6 `TokenizerConfig`：20 个字段的完整契约

这些字段由 `megatron/training/config/training_config.py:690-772` 的同一个 dataclass 声明；配置生成器再把它们接到 CLI。与原合并页只列 16 项不同，本表把当前 20 个 coverage owner 全部落在本页，避免“另见本页他处”却没有逐字段契约。

| 字段 | 声明类型 | 默认值 | 契约 |
|---|---|---:|---|
| `vocab_size` | `int` | `None` | EOD 或 padding 前的词表大小。 |
| `padded_vocab_size` | `int` | `None` | 模型实际词表大小；未给时可由原词表大小计算。 |
| `pad_vocab_size` | `bool` | `True` | 是否自动计算缺失的 `padded_vocab_size`。 |
| `vocab_file` | `str` | `None` | 词表文件路径。 |
| `merge_file` | `str` | `None` | BPE merges 文件路径。 |
| `vocab_extra_ids` | `int` | `0` | T5 span masking 使用的额外 token 数。 |
| `tokenizer_type` | `Literal[...]` | `None` | 选择 kwargs 分支与默认 library；非 Null 分支可由外部 metadata 覆盖最终 library，两个 Null 分支固定自身 library 并提前返回。12 个取值见同一类体 `:713-727`。 |
| `tokenizer_model` | `str` | `None` | tokenizer 模型或仓库路径。 |
| `metadata_path` | `str \| None` | `None` | tokenizer metadata JSON；CLI 名为 `--tokenizer-metadata`；Null 两分支忽略该值。 |
| `special_tokens` | `Optional[list[str]]` | `None` | 额外特殊 token；TikToken 有一组约定必需项。 |
| `tiktoken_pattern` | `Literal['v1', 'v2']` | `None` | TikToken pattern 版本。 |
| `tiktoken_num_special_tokens` | `int` | `1000` | TikToken 特殊 token 数。 |
| `tokenizer_sentencepiece_legacy` | `bool` | `False` | SentencePiece wrapper 的 legacy 行为。 |
| `tokenizer_sentencepiece_ignore_extra_whitespaces` | `bool` | `True` | SentencePiece 是否忽略额外空白。 |
| `tokenizer_hf_no_use_fast` | `bool` | `False` | 关闭 HuggingFace fast tokenizer。 |
| `tokenizer_hf_no_include_special_tokens` | `bool` | `False` | HF text-to-id 时不加入特殊 token。 |
| `trust_remote_code` | `bool` | `False` | 是否允许 HF tokenizer 执行仓库自定义代码；默认关闭。 |
| `null_tokenizer_eod_id` | `int` | `None` | NullTokenizer 的 EOD id；默认回退到 `vocab_size - 1`。 |
| `null_tokenizer_pad_id` | `int` | `-1` | NullTokenizer 的 pad id；应避开真实数据 token。 |
| `chat_template` | `Optional[str]` | `None` | 会话格式化用的自定义 Jinja 模板。 |

字段的类型、默认值与 docstring 均来自同一个类体（`megatron/training/config/training_config.py:690-772`）。其中 library 专属字段只有在 `tokenizer_type` 命中对应分支时才被读取；`trust_remote_code=True` 会扩大到外部仓库代码执行边界，启用前应单独审查来源。

### 3.7 本层边界

| 边界 | 失败方式 | 证据 |
|---|---|---|
| 不支持直接构造统一工厂 | `MegatronTokenizer()` 立即抛异常 | `megatron/core/tokenizers/megatron_tokenizer.py:40-43` |
| 类型不在支持列表 | `build_tokenizer` 抛 `ValueError` | `megatron/core/tokenizers/utils/build_tokenizer.py:26-32` |
| 普通 library 缺 tokenizer path | `from_pretrained` 断言失败；仅 byte/null 例外 | `megatron/core/tokenizers/megatron_tokenizer.py:83-85` |
| 多模态缺协议参数 | 缺 prompt format、special tokens 或 image tag 即断言 | `megatron/core/tokenizers/megatron_tokenizer.py:87-95` |
| 自动 padding 被关闭 | 调用方必须自行保证 `padded_vocab_size` 与 TP 几何兼容 | `megatron/core/tokenizers/utils/build_tokenizer.py:117-119` |

---

## 4. 数据底座:`IndexedDataset`(`.bin` / `.idx`)

最底层接口(`megatron/core/datasets/indexed_dataset.py`,`megatron/core/datasets/readme.md` 有描述)。预处理阶段把语料 tokenize 后写成两个文件:

```
xxx.bin   所有文档的 token id,扁平、连续存(按 dtype 编码)
xxx.idx   元数据:每条 sequence 的元素数、字节偏移指针;
                  每个 document 对应的 sequence 索引范围 [...);
                  (多模态时)每条 sequence 的 mode
```

`IndexedDatasetBuilder` 负责构建与合并;`IndexedDataset` 提供 `.get(doc_id, offset, length)` 随机读。`.bin` 用 mmap,不全载入内存。这是所有上层数据集的共同底座。

---

## 5. 原始 GPT 数据集(`GPTDataset`)

`megatron/core/datasets/gpt_dataset.py:127`。经典 LLM 预训练数据集。

### 5.1 三个索引(`_build_document_sample_shuffle_indices`,`:461`)

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

### 5.2 `__getitem__` 取样(`:251`)

```
idx ──shuffle index──► 打乱后的样本号
    ──sample index──► (起始doc, 偏移) ~ (结束doc, 偏移)
    ──document index + IndexedDataset.get()──► 拼出 sequence_length+1 个 token 的 text

tokens = text[:-1]          ┐  标准 next-token prediction:
labels = text[1:]           ┘  labels 是 tokens 左移一位
_get_ltor_masks_and_position_ids() ──► 因果 mask、loss_mask、position_ids
```

若拼出的 token 不足一个样本(语料尾巴),用 `_pad_token_id` 补齐,并把 padding 处的 `loss_mask` 置 0(不计损失)。

### 5.3 EOD 分隔与 reset —— 隐式打包的"样本内隔离"

隐式打包把不同 doc 接进同一个样本,带来一个问题:样本里 doc B 的 token 会不会 attend 到前面 doc A?默认会 —— 这通常无害(预训练),但可配置隔离:

- **EOD token**:doc 之间插入 end-of-document token,模型能学到"文档边界"。
- **`reset_position_ids`**:每遇 EOD,position id 从 0 重新计 —— doc B 不继承 doc A 的位置。
- **`reset_attention_mask`**:每遇 EOD,attention mask 重置 —— doc B **不 attend** doc A,样本内各 doc 真正独立。
- **`eod_mask_loss`**:EOD 位置的 loss 屏蔽。

开了 reset,一个定长样本里的多个 doc 在数值上等价于独立序列 —— 这其实已经是"打包"的雏形,只是 doc 仍可能被切断。

> `_get_ltor_masks_and_position_ids`(`:786`):`create_attention_mask` 为 False 时**不显式建 `[s,s]` mask**(交给 FlashAttention 等内核隐式处理因果性),省一大块激活。

### 5.4 Blending —— 多源混合(`BlendedDataset`)

`megatron/core/datasets/blended_dataset.py`。实际训练语料来自多个数据源(网页、代码、书……),要按**权重**混合。`BlendedDataset` 按权重从各 `GPTDataset` 采样;`BlendedMegatronDatasetBuilder`(`megatron/core/datasets/blended_megatron_dataset_builder.py`)是统一构造入口,`BlendedMegatronDatasetConfig` / `GPTDatasetConfig` 是配置。

> `megatron/core/datasets/readme.md` 强调:**所有 rank 都要走 builder 构建数据集,否则程序挂起**;真正落盘索引的 rank 由 config 控制(通常 rank 0 建、其余等)。

---

## 6. 序列打包(packed dataset,`megatron/core/datasets/data_schedule.py`)

### 6.1 动机:隐式打包对 SFT / 变长数据不够

`GPTDataset` 的"首尾相接 + 切定长"适合预训练,但对 **SFT / 指令微调 / 变长语料**有两个硬伤:

1. **不能切断样本**:SFT 的一条样本是完整的"指令 + 回答",从中间切断就破坏了语义/标签结构。
2. **逐样本补 padding 浪费巨大**:变长样本若各自 padding 到 `sequence_length`,短样本的 buffer 大半是 padding —— 算力全浪费在 padding 上。

### 6.2 解法:显式打包 + `PackedSeqParams`(THD 格式)

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

与 §5.3 GPTDataset 的对比:GPTDataset **切断 doc**、靠 EOD + `reset_attention_mask` 隔离;packed dataset **不切断**、靠 `cu_seqlens` + THD 格式隔离。后者保住了样本完整性。

### 6.3 `megatron/core/datasets/data_schedule.py` 的打包调度

**先说新基线下的形态**:打包模式下**没有 PP 组广播** —— `is_dataset_built_on_rank` 对**每个 PP stage 的 TP-0 rank** 都返回 True,各 stage 自行取数(`megatron/core/datasets/data_schedule.py:245-246` 的 docstring 明写 *There is no PP-group broadcast*;入口判据在 `pretrain_gpt.py:410`、`:415-421`),广播只发生在 TP 组内(`:382` 的 `broadcast_scalars`、`:808-816` 的 `broadcast_tensor`)。下面四条为原表述,原文保留;其中最后一条(`broadcast_to_pp_group`)在新基线下已作废,勘误见其后的 `[!deprecated]`。

`megatron/core/datasets/data_schedule.py` 的 `BasePackingScheduler` / `build_packed_microbatches`:
- 把一批变长样本**贪心 bin-pack** 成定长 microbatch(尽量塞满,减少 padding)。
- 配 `BalancedCPScheduler`(`15_megatron_pp_schedulers_analysis.md` §8.1 的混合 CP 动态调度):变长数据下,长序列分更多 CP 卡,并让各 DP×CP 组的 `seq²/cp` 工作量均衡。
- `PackedSeqParams` 里的 `cp_group` / `local_cp_size` —— 打包**与 CP 协同**:一条打包子序列还能再被 CP 切到多卡(`get_cp_slice_for_thd`)。
- 产出对齐到 PP 组的数据迭代器(`broadcast_to_pp_group`、`create_data_iterator`)。

> [!deprecated] 上面最后一条里的 `broadcast_to_pp_group` 在基线 `71092579` 下已不存在（全仓库符号零命中；由 #4226 “Minor improvements for Dynamic-cp” 从 `megatron/core/datasets/data_schedule_utils.py` 删除）。新基线改成**不再做 PP 组广播**：打包模式下 `is_dataset_built_on_rank` 对每个 PP stage 的 TP-0 rank 都返回 True，各 stage 自行取数，只在 TP 组内广播（`megatron/core/datasets/data_schedule.py:245-246` 的源码注释、`:382` 的 `broadcast_scalars`、`:808-816` 的 `broadcast_tensor`）。该描述对应旧基线 `ee3f1ffa…`；同句里的 `create_data_iterator` 仍存在。

所以，packed dataset 不只是“打包数据”，而是一套由**打包、变长 CP 负载均衡和 microbatch 调度**组成的完整方案，用于支撑 SFT、长文档、RL 等变长场景。

> **打包与动态 CP 的统一流水线**(`DefaultDynamicCPScheduler` 如何作为打包调度器的子类把二者缝在一起)见专文 `29_megatron_packed_dataset_dynamic_cp_analysis.md`。

### 6.4 NEW:`VarlenDataset` 独立入口 + get_batch 统一

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> ee3f1ff 之后,显式打包(SFT/变长)多了一条**独立数据集入口**,并统一了下游取数路径:
>
> - **`VarlenDataset`**(`--use-varlen-dataset`,`megatron/training/datasets/varlen_dataset.py`,#4832):面向 SFT 风格**变长指令数据**的 THD 打包数据集,继承 `SFTDataset` 家族但与 `--sft` 标志**解耦**(不再隐式耦合)。支持多源加载(HuggingFace Hub repo / 本地 `.parquet` / `.jsonl`),并按列名**自动识别四种 schema**:`openai-messages`(`messages`)、`sharegpt`(`conversations`)、`alpaca/dolly`(`instruction|prompt...`+`output|response...`)、`pretrain-text`(`text`,返回裸串无角色掩码);配套 `MockVarlenDataset`(`varlen_mock_dataset_config_json`,与 SFT mock 同 schema 但独立)。`varlen_sbhd_validation` 提供一个 SBHD 参考路径,跳过打包、用于 THD 数值正确性校验。
> - **每 index 直接吐一条子样本**:不同于 `SFTDataset`(一条样本里 `cu_seqlens` 拼了多条子序列),`VarlenDataset` 每个样本已是单条、自带 `padded_seq_len`。于是打包流水线第④步前的 `_unpack_batch`(`megatron/core/datasets/data_schedule_utils.py:105`)现支持**两种输入**:预打包(切开 `cu_seqlens`)与已拆开(只归一 batch 维、缺 `original_seq_len` 时从 `padded_seq_len` 补)。
> - **get_batch 统一 + SFT THD 支持 PP**(#4103):原先散落 `megatron/training/utils.py` 的取数逻辑收敛进 `megatron/core/utils.py`,按场景拆成 `get_batch_on_this_tp_rank`(`:2052`,长度前缀协议广播 `cu_seqlens`、动态 CP 的 `local_cp_size`)、`get_sft_batch_on_this_cp_rank`（#5403 后更名为 `_get_batch_on_this_cp_rank_per_document_balancing`，`:2339`）、`get_pretrain_batch_on_this_cp_rank`（#5403 后更名为 `_get_batch_on_this_cp_rank_per_sequence_balancing`，`:2389`；两者现由新增的分发器 `get_batch_on_this_cp_rank`（`:2545`）统一调用）、`get_thd_batch_on_this_cp_rank`(`:2608`)。**关键能力:SFT 的 THD 打包现可与 PP 共用**(PP 中间 stage **不靠广播**:打包模式下 `is_dataset_built_on_rank` 对每个 PP stage 的 TP-0 rank 都返回 True,各 stage 自行建数据集、自己拿到打包元数据;源码给出的理由是「Packed THD and SBHD validation both need padding metadata on every pipeline stage so each MoE layer excludes physical padding」—— `pretrain_gpt.py:410`、`:415-421`,另见 `megatron/core/datasets/data_schedule.py:245-246`)。

---

## 7. 两者对比

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

## 8. 约束

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
`GPTDataset` **不保证样本内 doc 完整** —— 一个 doc 可以被切到两个样本(§5.1)。这是隐式打包的定义性代价:`GPTDatasetConfig` 的全部字段里(`megatron/core/datasets/gpt_dataset.py:26-97`)没有任何"不切断 doc"的开关,要不切断只能换到 §6 的显式打包路径。
反向的代价同样存在:显式打包的贪心装填**按整条样本入桶、从不切分单条样本**(`megatron/core/datasets/data_schedule.py:165-185`),因此 `GPTDataset` 那种"一个样本可跨多个 doc、token 流连续"的装填在打包路径上不复存在。

---

## 9. 发展趋势

以下四条都锚在基线 `71092579` 的实读位置或提交历史,并**标为推断** —— 它们是源码里在途的改动,不是路线声明。

**① 隐式打包正在改用显式打包的隔离机制。**
`GPTDatasetConfig.inter_document_masking` 让 `GPTDataset` 也返回 `cu_seqlens`:「When True, return cu_seqlens marking document boundaries within each sample so that attention is restricted to individual documents」(`megatron/core/datasets/gpt_dataset.py:95-97`);`__getitem__` 里据 `document_lengths` 算出 `cu_seqlens` 与 `max_seqlen`,并**逐 document 重置 position_ids**(`:308-335`)。开启后 `create_attention_mask_in_dataloader` 被强制关掉,注释自陈「The dataset omits attention_mask when inter-document masking is enabled; disable the flag to avoid a TP broadcast mismatch」(`megatron/training/arguments.py:1436-1440`)。该特性由 #5298(`f88b85f8c`)引入,#5635(`bf32f4415`,commit message 即「Fix inter-document masking crash and NaNs with TP > 1 and micro_batch_size > 1」)修 bug。
→ **推断**:§7 对照表里"样本内隔离"一行的两栏正在合并 —— 隐式打包侧的 `reset_attention_mask` 有被 `cu_seqlens` 取代的趋势。这是本页从上述三处读出的方向,源码没有声明要弃用 `reset_attention_mask`。

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

## 10. 小结

- **分词供给**：`build_tokenizer` 先按显式 `tokenizer_type` 选择参数分支与默认 library；非 Null 分支提供外部 metadata 时，其 `library` 决定最终包装类，Null 两分支则忽略外部 metadata 并提前返回。词表随后按 TP 几何补齐，本页接管 `TokenizerConfig` 全部 20 个字段。
- **数据底座**:`IndexedDataset` 的 `.bin`(扁平 token)+ `.idx`(doc/sequence 元数据),mmap 随机读。
- **原始 GPT 数据集**(隐式打包,预训练):三级索引(document→sample→shuffle)把所有 doc 首尾相接、按 `sequence_length` 切定长;一个样本可跨 doc、一个 doc 可被切断;EOD + `reset_*` 做样本内隔离;`tokens/labels` 错位即 next-token 标签;`BlendedDataset` 按权重混多源;索引落盘缓存。
- **packed dataset**(显式打包,SFT/变长):把完整变长序列 bin-pack 进定长 buffer、**不切断**,用 `PackedSeqParams` 的 `cu_seqlens`/THD 格式让 attention 内核把一个 buffer 当 N 条独立序列;`megatron/core/datasets/data_schedule.py` 还把打包与混合 CP 负载均衡、microbatch 调度合在一起。
- **核心区别**:隐式打包切断 doc、靠 EOD+reset;显式打包不切断、靠 `cu_seqlens`。预训练用前者,SFT/变长用后者。

---

*生成依据：`Megatron-LM` `dev` 分支 `85902ef599ea4eb06ada7567a479c524b605767a`（2026-09-01）。源码行号以该 commit 为准。Tokenizer 覆盖统一工厂与边界；dataset 机制只覆盖 GPT/LLM 路径，BERT/T5/多模态数据集本体不展开。*

## Related Pages

- [[12_megatron_tp_analysis]] — §2.2 说明词表 padding 所服务的 embedding 与 LM head 并行切分。
- [[29_megatron_packed_dataset_dynamic_cp_analysis]] — 把 §6 的打包数据继续推进到动态 CP 调度全链。
- [[15_megatron_pp_schedulers_analysis]] — pipeline 调度与打包批次如何进入各 stage。
- [[13_megatron_cp_analysis]] — packed THD 序列在 context parallel 下如何切分。
- [[37_megatron_trtllm_export_analysis]] — 与本页输入端相对的训练产物离线导出边界。
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]] — 返回本域导航。
