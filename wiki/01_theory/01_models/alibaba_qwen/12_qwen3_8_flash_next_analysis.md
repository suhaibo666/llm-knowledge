# Qwen3.8-Flash-Next：Qwen4 架构预览，把"参数扩展"从 MoE 挪到嵌入表

> **来源基线**：
> - 模型卡快照 `raw/01_theory/01_models/alibaba_qwen/Qwen3_8_Flash_Next_model_card_f5d08274.md`，对应 [Qwen/Qwen3.8-Flash-Next `f5d08274`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/f5d08274)（HF 仓库创建于 2026-08-24）。
> - 结构数值取自同 revision 的 `config.json` 快照 `Qwen3_8_Flash_Next_config_f5d08274.json`（同目录）。
> - 参数量取自 HuggingFace safetensors 索引（2026-08-27 读取）。
>
> **维度**：开放权重发布分析（模型卡 + 权重配置交叉核对）。
> **更新**：2026-08-27。

> [!warning] 存在技术报告，但**本页尚未摄入**
> 与 Qwen3.8 其余档位不同，Flash-Next **有一份技术报告** PDF：`github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf`（`模型卡 :37`），另有博客 `qwen.ai/blog?id=qwen3.8-flash-next`。**本页的基线只到模型卡 + `config.json`**，尚未读该报告。因此本页给出的是"发布了什么结构"，四项创新的**动机论证、消融与理论依据仍在那份未摄入的报告里**——这是本页最大的缺口，已登记在 §6。

---

## 一、中央论点：这不是 Qwen3.8 的一档，是 Qwen4 的预览

卡片把话说得很重：这是"**将支撑 Qwen4 的架构**的实验性预览"，围绕"现代 LLM 各核心组件在规模下如何交互"做了一次根本性重思（`:24`）。

`config.json` 印证了这个定位——**它连模型类型都不叫 qwen3**：

| 字段 | 值 |
|---|---|
| `architectures` | `Qwen4ExpForConditionalGeneration` |
| `model_type` | `qwen4_exp` |
| `text_config.model_type` | `qwen4_exp_text` |

（`exp` = experimental。作为对照，[[11_qwen3_8_27b_analysis|Qwen3.8-27B]] 是 `qwen3_5` / `Qwen3_5ForConditionalGeneration`，[[10_qwen3_8_analysis|2.4T 旗舰]] 同属 Qwen3.5 谱系。）

卡片提出的问题也很清楚（`:22`）：前沿模型一路推高参数量与上下文长度，**问题已经不是"能扩多大"，而是"扩得多有效率"**。四项创新分别对应四个不同的效率轴：

| 创新 | 攻击的成本 | 配置证据 |
|---|---|---|
| **QSA** 混合稀疏注意力 | 长上下文**延迟** | `indexer_budget` / `indexer_compress_ratio` |
| **Gated Residual** | 深度**可训练性** | `hc_count` / `hc_lowrank` |
| **N-gram Embedding** | 参数扩展的**算力与显存** | `ngram_vocab_size_base` / `ple_layer_ids` |
| **定制训练配方** | 优化器**步数** | 卡片声称，配置无痕迹 |

---

## 二、参数账：125B ≠ 全部，三项相加才是 180B

卡片的参数口径很不寻常，值得原样抄下（`:47`）：

> Number of Parameters: **125B with 6B activated**, plus **51B n-gram embedding** and **4B MTP**

而 safetensors 索引实测 **180.0B**（BF16，335.3 GiB）。

**125 + 51 + 4 = 180 ——三项相加与实测精确吻合。** 这既验证了卡片的诚实，也说明一件重要的事：

> [!contradiction]
> **"125B" 不是这个 checkpoint 的大小。** 下载下来的权重是 **180B**，其中 **51B（28%）是 n-gram 嵌入表**。若按常规习惯只读"# Params: 125B"来估算显存或对比模型规模，会低估 44%。卡片在基准表里专门单列了一行 `# N-gram embedding params`（`:80`）来避免这个误读——**引用时必须把这三项分开说**。

这个拆分本身就是 N-gram Embedding 那条创新的论点：**嵌入表是一条"参数多、算力少、且易于 offload"的扩展轴**，所以它值得从主参数量里单列出来（`:34`）。

---

## 三、四项创新逐条对账

### 3.1 QSA：把稀疏的粒度从 token 挪到微块

**卡片声称**（`:32`）：Gated DeltaNet + Gated Attention 的配对被改造成 Gated DeltaNet + **Qwen Sparse Attention (QSA)**；**QSA 不是逐个挑选 token，而是在"微块（micro-block）"粒度上操作**，从而显著削减长上下文延迟。

**配置证据**：

| 卡片声称 | 配置字段 | 值 | 对账 |
|---|---|---:|:---:|
| Indexer 结构：MQA，4 个 query 头 + 1 个共享 key 头 | `indexer_n_heads` / `indexer_kv_heads` | 4 / 1 | ✅ |
| Indexer 头维 128 | `indexer_head_dim` | 128 | ✅ |
| 预算：**512 块或 2048 token** | `indexer_budget` ÷ `indexer_compress_ratio` | 2048 ÷ 4 = **512** | ✅ |

**微块大小就是 `indexer_compress_ratio = 4`**：2048 token 的预算 ÷ 每块 4 个 token = 512 个微块。卡片那句"512 blocks or 2048 tokens"（`:57`）由此得到完全的配置解释。

**层布局**：`layer_types` 统计为 36 个 `linear_attention` + 12 个 `full_attention`，后者落在层号 3, 7, 11, …, 47，即每 4 层一次（`full_attention_interval: 4`），48 = 12 × 4，与卡片的 `12 × (3 × (Gated DeltaNet → MoE) → 1 × (QSA → MoE))`（`:52`）完全一致。

> [!note]
> **`layer_types` 里的 `full_attention` 是命名遗留，那 12 层实际跑的是 QSA（稀疏）**。字段名沿用了 [[11_qwen3_8_27b_analysis|Qwen3.8-27B]] 的 Gated Attention 时代；判断是否稀疏要看 `indexer_*` 字段是否存在，而不是 `layer_types` 的字面值。

**与其他稀疏注意力的粒度谱系**：

| 机制 | 选择粒度 | 出处 |
|---|---|---|
| [[10_moba_analysis\|MoBA]] | 块 | Moonshot |
| DSA（[[26_deepseek_v4_technical_deepdive\|V4 用作 CSA 内部步骤]]） | token（V3.2）／压缩块（V4） | DeepSeek |
| **QSA** | **微块（4 token）** | Qwen |

- **事实**：卡片明确说 QSA"不逐个选 token，而在微块粒度操作"（`:32`）。
- **【推断】**：微块（4 token）介于 MoBA 的块与 DSA 的 token 之间，是在"选择精度"与"indexer 打分成本 + 访存连续性"之间取的中间点。
- **边界**：**卡片没有给任何粒度消融**，没有说明为什么是 4 而不是 1 或 16，也没有给延迟对照数据（只说 "cuts long-context latency significantly"）。上表的横向对比是本页所作，不是任何一家厂商的表述。

### 3.2 Gated Residual：Hyper-Connections 的另一条分支

**卡片声称**（`:33`）：带归一化的残差流是深层 LLM 可训练的关键。Gated Residual 用**逐元素、数据相关的 read gate** 与**每分支一个标量的 write gate**，调制流经**加宽残差流**的信息，在保持训练稳定与低推理开销的同时，带来跨层的更细粒度表达力。

**配置证据**：

| 卡片声称 | 配置字段 | 值 |
|---|---|---:|
| 分支数 4 | `hc_count` | 4 |
| 瓶颈秩 320 | `hc_lowrank` | 320 |

**注意字段前缀是 `hc_`** —— hyper-connection。这把它放进了一条本库已有的技术线：

| | Qwen3.8-Flash-Next | [[25_mhc_analysis\|mHC]]（DeepSeek） |
|---|---|---|
| 配置字段 | `hc_count: 4`、`hc_lowrank: 320` | `hc_mult: 4`、`hc_sinkhorn_iters: 20` |
| 扩展率 | **4** | **4** |
| 约束方式 | **门控**（read gate + write gate） | **流形投影**（双随机矩阵 / Birkhoff polytope） |
| 已见采用方 | Qwen | DeepSeek-V4、[[12_glm_5_3_flash_analysis\|GLM-5.3-Flash]] |

- **事实**：两者都源自 Hyper-Connections 这条线（字段前缀相同、扩展率同为 4），但**解决 HC 不稳定性的手段不同**——mHC 靠把残差映射投影到双随机矩阵流形，Qwen 靠门控。
- **【推断】**：扩展率 4 可能已成为该系列的事实默认值。
- **边界**：卡片没有提 Hyper-Connections，也没有引用 mHC；这条谱系归属是本页依据字段命名与机制描述所作的**推断**，两家均未确认。**不要写成"Qwen 采用了 DeepSeek 的 mHC"——它没有。**

### 3.3 N-gram Embedding：与 DeepSeek Engram 高度同构

**卡片声称**（`:34`）：嵌入提供了一条独特的参数扩展轴——**所需算力更少，且比 MoE 更适合 offload**。通过短 n-gram 索引，这种方式让参数扩展在显存受限的加速器上非常高效，且不牺牲质量。

**配置证据**：

| 卡片声称 | 配置字段 | 值 |
|---|---|---:|
| N-gram 词表 20,000,000 | `ngram_vocab_size_base` | 20000000 |
| bigram / trigram | `ngram_size` | 3 |
| **在第 2 层** | `ple_layer_ids` | `[2]` |
| — | `heads_per_ngram` | **8** |
| — | `ple_embed_dim` | 2560 |
| — | `split_ngram_parts` | 128 |

**这与本库 [[29_engram_analysis|DeepSeek Engram]] 的机制高度同构**：

| 维度 | Qwen N-gram Embedding | [[29_engram_analysis\|DeepSeek Engram]] |
|---|---|---|
| 核心思想 | 嵌入表作为独立参数扩展轴 | 在计算稀疏（MoE）之外引入**记忆稀疏** |
| 索引方式 | 短 n-gram（`ngram_size: 3`） | N-gram 哈希查表 |
| 多头哈希 | `heads_per_ngram: 8` | **确定性多头哈希**，多 head 抗冲突 |
| Offload 论据 | "more amenable to offloading than MoE" | 表可存 CPU 内存，**确定性查找**故可异步预取 |
| 算力 | "requires less computation" | **零 FLOPs**（纯查表） |

- **事实**：两家在同一时期、各自独立地把"n-gram 查表嵌入"作为 MoE 之外的第二条参数扩展轴，且都强调**低算力**与**可 offload**这两条同样的理由；Qwen 的 `heads_per_ngram: 8` 与 Engram 的多头哈希设计对应。
- **边界**：卡片**没有引用 Engram**，也没给 U 型 scaling law 那类"算力 vs 记忆"的配比分析（那是 Engram 论文的内容）。两者是否有承袭关系**无法判定**，本页只记录同构。

### 3.4 训练配方：唯一在配置里查不到的一项

**卡片声称**（`:35`）：Muon 与 AdamW 分别施加于**特定的权重类别**以最大化效率；在**重新拟合的 scaling law** 指导下，**取消传统的 batch-size warmup，直接从目标 batch size 开始**，大幅减少优化器总步数，同时安全地支持更大的学习率。

- **事实**：这是四项创新里唯一属于**训练**而非结构的一项，因此 `config.json` 里没有任何对应字段——**完全无法从权重侧核验**。
- **价值**：其中"取消 batch-size warmup"是一条相当具体、可被他人验证的工程主张（多数大模型训练配方都保留 warmup）。
- **边界**：没有给出哪些权重走 Muon、哪些走 AdamW，没有给 scaling law 的重拟合形式，没有 with/without warmup 的对照曲线。**这些大概率在未摄入的技术报告里。**

---

## 四、其余结构参数（配置对账）

| 组件 | 卡片声称 | 配置字段 | 配置值 | 对账 |
|---|---|---|---:|:---:|
| 层数 | 48 | `num_hidden_layers` | 48 | ✅ |
| hidden | 2560 | `hidden_size` | 2560 | ✅ |
| 词表 | 248,320 | `vocab_size` | 248320 | ✅ |
| 专家数 | 512 | `num_experts` | 512 | ✅ |
| 激活专家 | 10 routed + 1 shared | `num_experts_per_tok` / `shared_expert_intermediate_size` | 10 / 640 | ✅ |
| 专家中间维 | 640 | `moe_intermediate_size` | 640 | ✅ |
| GDN V 头 / QK 头 | 48 / 16 | `linear_num_value_heads` / `linear_num_key_heads` | 48 / 16 | ✅ |
| GDN 头维 | 128 | `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 | ✅ |
| QSA Q 头 / KV 头 | 24 / 2 | `num_attention_heads` / `num_key_value_heads` | 24 / 2 | ✅ |
| QSA 头维 | 256 | `head_dim` | 256 | ✅ |
| RoPE 维 64 | — | `partial_rotary_factor` × `head_dim` | 0.25 × 256 = 64 | ✅ |
| MTP | 1 层，多步训练 | `mtp_num_hidden_layers` | 1 | ✅ |
| 上下文 | 262,144 原生，可扩到 1,000,000 | `max_position_embeddings` | 262144 | ✅ |

**专家中间维只有 640** —— 对照 [[10_qwen3_8_analysis|2.4T 旗舰的 2048]]，这是极细粒度的专家切分，与"6B 激活"的目标一致。

---

## 五、能力位置：6B 激活参数打赢 13B 激活

同表对比（`:88-222`）。卡片把参数账直接放进了表头，这张表因此格外有说服力：

| | **Qwen3.8-Flash-Next** | [[11_qwen3_8_27b_analysis\|Qwen3.8-27B]] | Qwen3.7-Plus | [[31_deepseek_v4_released_checkpoints_analysis\|DeepSeek-V4-Flash-0731]] | Opus 4.6 (Max) |
|---|---:|---:|---:|---:|---:|
| 总参数 | 125B (+51B n-gram) | 27B | 397B | 284B | — |
| **激活参数** | **6B** | 27B | 17B | 13B | — |
| DeepSWE 1.1 | **58.7** | 42.2 | 16.5 | 54.4 | — |
| SWE-bench Pro ‡ | **62.5** | 61.7 | 55.8 | 56.0 | 53.4 |
| SWE-bench Multilingual | **81.0** | 73.8 | 75.8 | — | 77.5 |
| NL2Repo-Bench | 48.1 | 42.3 | 41.1 | **54.2** | 47.6 |
| CoWorkBench † | **73.9** | 70.7 | 65.1 | 45.1 | 68.2 |
| JobBench | **55.7** | 33.4 | 27.6 | 41.3 | 36.6 |
| Toolathlon Verified | **73.5** | 67.1 | 50.6 | 70.3 | — |
| IFBench | **81.3** | 79.5 | 79.1 | 79.2 | 62.5 |
| GPQA Diamond | **91.7** | 89.2 | 90.3 | 90.8 | 91.3 |
| HLE | 35.9 | 30.8 | 34.7 | 33.8 | **40.0** |
| LiveCodeBench v6 | **91.9** | 90.3 | 89.6 | 90.6 | 88.8 |

† 阿里内部基准　‡ 见下方口径说明

**核心读法**：**6B 激活参数在 12 项里拿下 9 项最优**，包括压过 13B 激活的 DeepSeek-V4-Flash-0731 与 17B 激活的 Qwen3.7-Plus。这是对"效率而非规模"论点最直接的支撑。落后的两项是 NL2Repo-Bench（输给 V4-Flash）与 HLE（输给 Opus 4.6）——**HLE 这类广知识推理正是最吃参数容量的方向**，与 51B n-gram 嵌入所补的"事实记忆"是不同的东西。

> [!contradiction]
> **这张表的口径需要三重折扣**：① SWE-bench Pro 一行与 [[11_qwen3_8_27b_analysis|27B 页]] 同样存在"**修正了有问题的题目并重测所有基线**"的说明（`:222`），与公开榜单分数不可直接比；② CoWorkBench 是内部基准；③ DeepSWE 1.1 一行"**在 Claude Code 与 mini-SWE-agent 两个 harness 中取最高分**"，且卡片自己注明 Flash-Next 在 mini-SWE-agent 上表现最好（`:222`）——**取二者最大值对本模型有利**，其他列是否也享受同等待遇未说明。
>
> 另外，表中 `DeepSeek-V4-Flash-0731` 记作 **284B / 13B 激活**——那是**论文口径**。该 checkpoint 的实际发布权重经 safetensors 索引实测为 **304.2B**（见 [[31_deepseek_v4_released_checkpoints_analysis]] §5）。

---

## 六、未披露边界

**首要缺口**：**技术报告与官方博客均未摄入**（见页首 warning）。四项创新的动机推导、消融、scaling law 重拟合形式，很可能都在其中。这应是本目录的下一个摄入目标。

模型卡本身没有给出的：

- QSA 微块大小为何取 4，无粒度消融；无长上下文延迟/吞吐实测（只说 "significantly"）。
- Gated Residual 与 Hyper-Connections / mHC 的关系未说明（§3.2 的谱系归属是本页推断）。
- N-gram Embedding 的算力/记忆配比、命中率、offload 实测收益均无；与 [[29_engram_analysis|Engram]] 的关系未提。
- Muon/AdamW 的权重类别划分、取消 batch-size warmup 的对照曲线均无。
- 预训练 token 数、数据构成未披露。
- 视觉侧基准表（`:223-307`）本页未逐行摄入。
- 1M 上下文同样是 262K 的外推（参见 [[11_qwen3_8_27b_analysis]] §3 的 YaRN 配方）。

---

## 关联页面

- [[10_qwen3_8_analysis]] — Qwen3.8 旗舰档（2.4T-A95B + Max），同期同系列。
- [[11_qwen3_8_27b_analysis]] — Qwen3.8-27B：Qwen3.5 谱系的紧凑稠密档，与本页的 Qwen4 谱系形成代际对照。
- [[29_engram_analysis]] — DeepSeek Engram：与 N-gram Embedding 高度同构的"记忆稀疏"路线。
- [[25_mhc_analysis]] — mHC：Hyper-Connections 的另一条约束分支。
- [[10_moba_analysis]] — MoBA：块级稀疏注意力，QSA 微块粒度的对照点。
- [[26_deepseek_v4_technical_deepdive]] — CSA/HCA/DSA/MLA：稀疏注意力粒度谱系的另一端。
- [[31_deepseek_v4_released_checkpoints_analysis]] — 本页基准表中 DeepSeek-V4-Flash-0731 的参数口径修正来源。
- [[12_kimi_linear_analysis]] · [[20_gdn_kda_linear_attention_analysis]] — Gated DeltaNet 机制深挖。
- [[01_theory/01_models/index|模型架构与模型家族总索引]]
