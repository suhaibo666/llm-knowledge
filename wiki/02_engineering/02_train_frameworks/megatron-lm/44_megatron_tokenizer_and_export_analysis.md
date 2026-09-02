---
title: "Megatron-LM 训练管线的两个端点：分词供给与权重导出"
---

# Megatron-LM 训练管线的两个端点：分词供给与权重导出

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **维度**：功能树模块 K 的分词侧（`megatron/core/tokenizers/**`，33 个 `.py`）与模块 M 的导出侧（`megatron/core/export/**`，17 个 `.py`）。
> **最近更新**：2026-09-02 首建。

---

## 1. 为什么这两棵子树放在同一页

它们是训练管线的**两个端点**：分词器在最前——文本进来变成 token id；权重导出在最后——训练出来的权重出去变成外部执行器能吃的格式。

放一起不是凑数，而是因为它们共享一组性质，这组性质决定了它们为什么会被漏掉：

1. **都不参与训练主循环**。分词在数据准备阶段完成，导出在训练结束之后发生。任何一条"跟着一次迭代走"的因果链都不会穿过它们。
2. **都是边界适配**。它们的复杂度不来自算法，来自要兼容的外部世界有多杂——分词器要接 HuggingFace、SentencePiece、TikToken 等多个生态，导出要对上 TRT-LLM 的层名与布局约定。
3. **都在配置面上几乎不可见**。合计 50 个源文件，本域 28 篇机制页此前**一字未提**，而且它们没有对应的 `TransformerConfig` 字段——[[40_megatron_feature_tree_analysis]] 的配置面对账看不见它们，**只有文件面对账能把它们暴露出来**。这是功能树相对 config-coverage 枚举轴的独立价值的最好例证。

**本页不覆盖**：数据集本体（IndexedDataset、GPT 数据集、打包） → [[11_megatron_dataset_analysis]]、[[29_megatron_packed_dataset_dynamic_cp_analysis]]；推理引擎与服务 → [[31_megatron_inference_engine_analysis]]；RL 侧的权重同步与 refit → [[30_megatron_rl_posttraining_consistency_analysis]]、[[27_megatron_tp_fsdp_resharding_supplements_analysis]]。

> [!note] 一处极易误判
> [[31_megatron_inference_engine_analysis]] 里出现过 `trt_llm_engine_wrapper.py`，那是**推理引擎里的一个桩**（该页自陈"从头到尾是个桩"）。它与本页 §5 讲的 `megatron/core/export/trtllm/` 权重导出子树**是两回事**：前者是把 TRT-LLM 当推理后端用的接口占位，后者是把 Megatron 权重转成 TRT-LLM 格式的转换器。搜到前者不等于后者有覆盖。

---

## 2. 分词供给：一个工厂、两级分类

### 2.1 目录结构就是分类法

`megatron/core/tokenizers/` 的组织是**两级正交分类**：

```
tokenizers/
├── megatron_tokenizer.py      # 统一入口（from_pretrained 分派）
├── base_tokenizer.py
├── text/
│   ├── libraries/             # 第一级：用哪个分词库实现
│   ├── models/                # 第二级：哪个模型家族的预设
│   └── parsers/               # 输出侧：把生成文本解析成结构
└── vision/                    # 与 text/ 平行的多模态分支
```

**`libraries` 与 `models` 的分工是这套设计的核心**：`libraries` 回答"用什么实现来分词"（HuggingFace、SentencePiece、TikToken、ByteLevel、Null、SFT、MegatronHF——`text/libraries/` 下 7 个具体实现加一个 `abstract_tokenizer.py`）；`models` 回答"这个模型家族的特殊约定是什么"（`gpt_tokenizer.py` / `bert_tokenizer.py` / `t5_tokenizer.py` / `mamba_tokenizer.py` / `default_tokenizer.py`）——特殊 token、BOS/EOS 约定、掩码规则这类东西。

**被否掉的替代是按模型家族一维展开**（`GPT2BPETokenizer`、`BertWordPieceTokenizer`……，这是 Megatron 更早的写法）。一维展开的代价是组合爆炸：同一个模型家族可以用不同库实现，同一个库可以服务不同家族，`M × N` 个类里绝大多数是重复代码。拆成两级后新增一个库只写一个文件，新增一个家族也只写一个文件。

### 2.2 入口：metadata 驱动的分派

`MegatronTokenizer`（`megatron/core/tokenizers/megatron_tokenizer.py:37`）的构造函数直接抛异常（`:40-43`），要求走 `from_pretrained()`（`:46`）——**这是刻意堵死直接实例化**，因为分词器需要先决定用哪个实现类，而这个决定依赖元数据。

分派逻辑读 `metadata['library']`（`:83`），再由 `_get_tokenizer_model_class(library, metadata)`（`:188`）取到具体类。有三个库走特殊路径（`:84`）：`byte-level`、`null-text`、`null-multimodal`——它们不需要加载任何模型文件。

`write_metadata()`（`:104`）是反向操作，把库类型与配置写成元数据；它断言库名必须在 `TEXT_LIBRARIES` 或 `VISION_LIBRARIES` 里（`:139-141`），并据此推出 `tokenizer_type` 是 `'text'` 还是 `'vision'`（`:143`）——**文本/视觉的分支由库名反推，不用用户再声明一次**。

### 2.3 从 CLI 到实现：`build_tokenizer` 的推断链

`megatron/core/tokenizers/utils/build_tokenizer.py:26` 的 `build_tokenizer(args, **kwargs)` 是训练侧的实际入口。它做的是**从用户给的零散参数反推 library**（`:35-91`）：给了 SentencePiece 模型文件就推 `sentencepiece`（`:51`）、给了 TikToken 相关参数推 `tiktoken`（`:58`）、给了 HF 路径推 `huggingface`（`:68`）、多模态推 `multimodal`（`:78`）、SFT 场景推 `sft`（`:84`）。推断结果组装成 `metadata = {'library': tokenizer_library}`（`:91`、`:112`），交给 `MegatronTokenizer.from_pretrained`（`:101`、`:113`）。

这层推断的存在说明一个务实的取舍：**用户不必知道 library 这个概念**，他只知道自己有一个 `.model` 文件或一个 HF 仓库名。代价是推断规则本身成了隐式契约——参数组合变了，推出来的 library 可能变，而用户不会察觉。

### 2.4 词表 padding：两处实现与它们的分工

`vocab_size_with_padding(orig_vocab_size, args)`（`build_tokenizer.py:124`）把词表大小向上取整到 `make_vocab_size_divisible_by × tensor_model_parallel_size` 的倍数（`:129-130`），并在 rank 0 打印补了多少个 dummy token（`:131-136`）。`_set_padded_vocab_size`（`:141`）在 `args.padded_vocab_size` 未设时调用它。

**为什么必须 padding**：词表维度要在 TP 组间均分（`VocabParallelEmbedding` 的切分，见 [[12_megatron_tp_analysis]]），除不尽就没法切；乘上 `make_vocab_size_divisible_by` 则是为了让每个分片的大小对 GPU 友好（对齐到 128 之类）。补出来的 token 永远不会被采样到，只是占位。

`megatron/training/vocab_utils.py:9` 的 `calculate_padded_vocab_size` 是**同一计算的另一处实现**，差别在于它带 `lru_cache`（`:4` 导入，`:29` 调用 `_calculate_padded_vocab_size_cached`），并且**把日志与缓存分开**（`:33` 注释明写 "Handle logging separately to avoid affecting cache behavior"）——如果日志写在被缓存的函数里，第二次调用命中缓存就不打印了，看日志的人会以为没发生。

两处并存本身是技术债，但分工是清楚的：core 侧那个在构建分词器时顺带算并写回 `args`；training 侧那个供模型构建器（`megatron/training/models/gpt.py`、`hybrid.py`）反复查询，故需要缓存。

### 2.5 chat template 与输出解析器

`text/libraries/chat_template.py` 处理对话模板——把多轮消息按模型约定拼成一个 token 序列。

`text/parsers/` 是**反方向**的：把模型生成的文本解析回结构。当前有两个具体实现，`deepseek_r1_reasoning_parser.py`（分离 `<think>` 推理段与最终回答）与 `qwen3_coder_tool_parser.py`（解析工具调用），加一个 `base_parser.py`。

它们的消费方在推理服务：`megatron/core/inference/apis/serve_config.py` 的 `parsers` 字段。**这条链值得记住**——分词器子树里的 parser 会一路走到推理服务的配置面，[[42_megatron_rl_runtime_analysis]] §7 提到的 `--rl-inference-parsers` 也接在这里。所以"分词器"这个名字其实低估了这棵子树的职责范围：它同时管**入口的 tokenize 与出口的 detokenize/parse**。

### 2.6 视觉分支

`vision/` 与 `text/` 平行：`libraries/multimodal_tokenizer.py` 与 `null_multimodal_tokenizer.py`，加 `models/default_tokenizer.py` 与 `vision_tokenizer.py`。结构上复用同一套两级分类，说明这套分类法在设计时就考虑了非文本模态。

> [!note] 待展开
> 本节覆盖了这棵子树的**结构、分派链与词表 padding**，`TokenizerConfig` 的逐字段契约见本页 §4。仍未展开的是**各个具体 library 实现的内部差异**——例如 TikToken 与 SentencePiece 在特殊 token 处理上的不同、`chat_template.py` 的模板语法、两个 parser 的具体解析规则。那需要逐文件走查，本轮未做。

---

## 3. 权重导出：把训练态权重翻译给 TRT-LLM

`megatron/core/export/` 17 个文件，目标单一：把 Megatron 的模型 state dict 转成 TensorRT-LLM 能加载的权重与配置。

### 3.1 `ExportConfig`：只有四个字段

`megatron/core/export/export_config.py:9` 的 `ExportConfig` 出奇地小：

| 字段 | 默认 | 含义 |
|---|---|---|
| `inference_tp_size` | 1 | 目标推理的 TP 度 |
| `inference_pp_size` | 1 | 目标推理的 PP 度 |
| `use_parallel_embedding` | False | embedding 是否并行 |
| `use_embedding_sharing` | None | **已废弃** |

`use_embedding_sharing` 的 `__post_init__` 会发 `DeprecationWarning`（`:23-30`），指明改用 `TRTLLMHelper` 的 `share_embeddings_and_output_weights`。

**这个字段少得有信息量**：导出的目标并行度与训练的并行度**解耦**——训练用 TP8/PP4，导出可以指定 TP2/PP1。推理与训练的最优并行度本来就不同（推理无梯度与优化器状态，更小的 TP 换更低通信延迟），所以导出这一步必须能重切分。

### 3.2 两条转换路径与它们的判据

`TRTLLMHelper.get_trtllm_pretrained_config_and_model_weights`（`trtllm_helper.py:264`）有一个 `on_device_distributed_conversion: bool = False` 开关（`:269`），分出两条实现：

| | 分布式路径（`on_device_distributed_conversion=True`） | 单卡路径（默认） |
|---|---|---|
| 实现 | `_get_trtllm_pretrained_config_and_model_weights_in_distributed_setting`（`:377`）→ `DistributedTRTLLMModelWeightsConverter`（`:405`） | `_get_..._list_on_single_device`（`:451`）→ `SingleDeviceTRTLLMModelWeightsConverter`（`:482`） |
| 前提 | state dict **已按目标推理并行度分好片**，每卡拿自己那份（`:288` docstring） | 单机持有完整 state dict |
| 模型限制 | 仅 `gpt` / `gptnext` / `llama` / `nemotron_nas`（`:303-311`） | 无此限制 |
| `export_config` | **必须为 None**（`:312-315`） | 由用户给出 |
| 额外必需参数 | `vocab_size`（`:302`）、`gpus_per_node`（`:317-319`） | — |

**选路判据源码写得很直白**。分布式路径的 `export_config must be None` 那条断言，错误信息是："Export config is inferred based on the parallel state. If you want to set inference tp 2, then load the model with this TP2 setting and just pass in the model state dict."（`:313-315`）

翻译：**分布式路径不做重切分，它假设你已经按目标并行度加载好了**。目标并行度从进程组状态推断，不从配置读——两个来源会冲突，所以直接禁止传配置。

于是判据清楚了：**要重切分就走单卡路径**（一台机器上有完整权重，可以任意重排）；**权重大到单机放不下、且你已经能按目标并行度加载**，才走分布式路径，代价是只支持四种模型且不能改并行度。

### 3.3 层名映射

`trtllm_layers.py` 与 `model_to_trllm_mapping/default_conversion_dict.py` 负责 Megatron 层名到 TRT-LLM 层名的映射。这是纯粹的**词汇表翻译**，没有算法，但它是导出正确性的全部——名字对错不上，权重就装到错的位置，而模型仍能跑出（错误的）结果。

### 3.4 量化 scale 的迁移

FP8 模型导出时多一步：`_load_scaling_factors(model_state_dict)`（`:210`）在入口处按 `fp8_quantized` 提取 scale（`:298`），随后 `_add_scales_to_converter`（`:352`）把它们注入转换器（该方法同时接受两种 converter，`:355`）。

**为什么 scale 要单独走一条路**：FP8 权重的数值意义由 `(量化值, scale)` 共同决定，scale 不是普通权重、不在层名映射表里，但少了它权重就是一堆无意义的整数。

入口处还有一行值得注意的清洗（`:299`）：`model_state_dict = {k: v for k, v in model_state_dict.items() if "extra_state" not in k}`——**丢掉所有 `extra_state` 键**。那是 TransformerEngine 用来存 FP8 amax 历史等训练期状态的，推理不需要，且它的格式 TRT-LLM 不认识。

### 3.5 引擎构建

`build_and_save_engine`（`:532`）与 `engine_builder/trtllm_engine_builder.py` 把转换好的权重与配置交给 TRT-LLM 构建引擎。这一步已经跨出 Megatron 边界，依赖外部的 TRT-LLM 包。

> [!note] 待展开
> `default_conversion_dict.py` 的**具体映射条目**、`trt_model_config.py` / `trt_model_type.py` / `data_type.py` 的类型对应关系，本页未逐条走查。

---

## 4. 约束与边界

| 边界 | 表现 | 证据 |
|---|---|---|
| 分词器不能直接构造 | 构造函数抛异常，必须走 `from_pretrained` | `megatron/core/tokenizers/megatron_tokenizer.py:40-43` |
| library 由参数组合隐式推断 | 参数变则推断结果可能变，用户不会察觉 | `megatron/core/tokenizers/utils/build_tokenizer.py:35-91` |
| 词表必须能被 `divisible_by × TP` 整除 | 否则 `VocabParallelEmbedding` 无法均分 | `build_tokenizer.py:129-130` |
| 词表 padding 有两处实现 | core 侧写回 args，training 侧带缓存供反复查询 | `megatron/training/vocab_utils.py:29-33` |
| 导出的分布式路径只支持四种模型 | `gpt` / `gptnext` / `llama` / `nemotron_nas` | `megatron/core/export/trtllm/trtllm_helper.py:303-311` |
| 分布式路径不做重切分 | `export_config` 必须为 None，并行度从进程组推断 | `:312-315` |
| `use_embedding_sharing` 已废弃 | 构造时发 `DeprecationWarning` | `megatron/core/export/export_config.py:23-30` |
| `extra_state` 键被丢弃 | TE 的训练期 FP8 状态，推理不需要且 TRT-LLM 不认 | `trtllm_helper.py:299` |

---

---

## 配置契约：`TokenizerConfig`

§2 讲的是分词子树的**结构与分派链**；本节给它的**配置面**。这些字段经 [[41_megatron_config_surface_analysis]] §2 的 `ArgumentGroupFactory` 自动转成 CLI（`megatron/training/arguments.py:4368` 那一处调用），字段名到 flag 名的规则是下划线转连字符、bool 按默认值决定 `--x` 还是 `--no-x`/`--disable-x`。**下表的类型、默认值与说明直接取自 `megatron/training/config/training_config.py` 的 `TokenizerConfig` 类体**（行号为该文件内行号），与生成 CLI 所用的是同一份声明，因此不会与实际 flag 漂移。

读法上有三组值得先看：**`vocab_file` / `merge_file` / `tokenizer_model` / `metadata_path` 决定走哪条 library 推断分支**（见 §2.3）；**`tiktoken_*` 与 `tokenizer_sentencepiece_*` / `tokenizer_hf_*` 是库特化参数**——只在推断出对应 library 时生效，配错了不会报错、只是被忽略；**`pad_vocab_size` 与 §2.4 的词表 padding 直接相关**，关掉它就必须自己给 `padded_vocab_size`。


### `TokenizerConfig`（`megatron/training/config/training_config.py`，16 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `pad_vocab_size` | `bool` | `True` | Whether to pad vocab size of the model automatically if padded_vocab_size is not provided. | `:701` |
| `vocab_file` | `str` | `None` | Path to the vocab file. | `:704` |
| `merge_file` | `str` | `None` | Path to the BPE merge file. | `:707` |
| `vocab_extra_ids` | `int` | `0` | Number of additional vocabulary tokens. They are used for span masking in the T5 model. | `:710` |
| `tokenizer_model` | `str` | `None` | Path to the tokenizer model. | `:729` |
| `metadata_path` | `str \| None` | `field(default=None, metadata={'argpar…` | Path to the tokenizer metadata file in json format. | `:732` |
| `special_tokens` | `Optional[list[str]]` | `field(default=None, metadata={'argpar…` | List of special tokens. For TikTokenizer needs to have ["<unk>", "<s>", "</s>", "<mask>", "<pad>", "<cls>", "<sep>"] | `:737` |
| `tiktoken_pattern` | `Literal['v1', 'v2']` | `None` | Which tiktoken pattern to use. Options: [v1, v2] | `:743` |
| `tiktoken_num_special_tokens` | `int` | `1000` | Number of special tokens in tiktoken tokenizer. | `:746` |
| `tokenizer_sentencepiece_legacy` | `bool` | `False` | SentencePiece tokenizer wrapper legacy behavior. Allows special tokens usage. | `:749` |
| `tokenizer_sentencepiece_ignore_extra_whitespaces` | `bool` | `True` | Whether to ignore extra whitespaces in the input text while encoding. | `:752` |
| `tokenizer_hf_no_use_fast` | `bool` | `False` | Whether to use fast HuggingFace tokenizer. | `:755` |
| `tokenizer_hf_no_include_special_tokens` | `bool` | `False` | Converting text to ids will not include special for HuggingFace tokenizer. | `:758` |
| `trust_remote_code` | `bool` | `False` | Whether or not to allow PreTrainedTokenizer to execute remote code. | `:761` |
| `null_tokenizer_eod_id` | `int` | `None` | EOD token id for NullTokenizer. Defaults to `vocab_size - 1`. | `:764` |
| `null_tokenizer_pad_id` | `int` | `-1` | Pad token id for NullTokenizer. Defaults to -1 (no pad token). Set to a value outside the dataset to avoid masking real tokens. | `:767` |

> 该类共 20 个字段，本表收 16 项；其余 4 项已在别处归属：`vocab_size`、`padded_vocab_size`、`tokenizer_type`、`chat_template` → 本页他处。

> **一处安全相关的默认值**：`trust_remote_code` 默认 `False`。开启它等于允许 HuggingFace 分词器执行仓库里携带的任意代码——与 [[41_megatron_config_surface_analysis]] §3.2 的 `TargetAllowlist`、[[42_megatron_rl_runtime_analysis]] §3.2 的 agent 白名单同属一类边界：**凡是让外部内容决定「执行什么」的入口，默认都关着**。

## Related Pages

- [[40_megatron_feature_tree_analysis]] — 功能树总览；本页覆盖的两棵子树是它 §4 仪表盘里唯一"配置面完全看不见、只能靠文件对账暴露"的空白
- [[11_megatron_dataset_analysis]] — 数据集与 IndexedDataset；分词器的下游，两者共同构成模块 K
- [[12_megatron_tp_analysis]] — §2.4 词表 padding 的动因在那里：`VocabParallelEmbedding` 的 TP 切分要求整除
- [[31_megatron_inference_engine_analysis]] — §2.5 的 parser 一路走到那里的 `serve_config`；另注意该页的 `trt_llm_engine_wrapper.py` 与本页 §3 的导出子树是两回事
- [[30_megatron_rl_posttraining_consistency_analysis]] — 另一条权重交付路径：RL rollout 的 refit 与本页的 TRT-LLM 导出同属模块 M，但目标执行器不同
- [[42_megatron_rl_runtime_analysis]] — `--rl-inference-parsers` 接的正是 §2.5 的解析器族


