# LongCat-2.0：在国产 AI ASIC 上把 1.6T MoE 推到近前沿 Agentic Coding

> **来源**: 美团 LongCat 团队官方 Tech Blog — Introducing LongCat-2.0
> **URL**: https://longcat.chat/blog/longcat-2.0 · 模型卡 https://huggingface.co/meituan-longcat/LongCat-2.0 · 仓库 https://github.com/meituan-longcat/LongCat-2.0
> **License**: MIT（权重「coming soon」，截至抓取日尚未放出）
> **Baseline / 访问日期**: 2026-07-02
> **维度**: Overview + 机制级深挖（架构 / 预训练 / 后训练 / AI Infra / 低精度与数值可靠性 / 稳定性 / 效果）

> [!note] 来源与保真度说明（务必先读）
> 官方博客是 **JS 渲染的 SPA**，直取只得到标题；本页内容经**渲染代理提取 + 三源交叉核对**（官方博客渲染文本、HF/GitHub `README.md`、DeepWiki 镜像），并与二手报道（VentureBeat / DeepWiki 摘要）对比去伪。因此：
> - 明确**多源一致**的事实（参数量、tokens、LSA/N-gram/ScMoE/MOPD/6D 并行、评测表）按源陈述；
> - 单源摘要或有分歧处**已标注**（如「加速器·小时 vs 天」）；
> - 博客**未披露**的量（层数、隐藏维、每层专家数、top-k、学习率/batch、是否 FP8）一律标 **未披露**，绝不臆造；
> - 与二手「常识」冲突处，**以官方源为准并显式指出**（见 §9）。
> 待官方权重/config.json/技术报告放出后，应回头用 `file:line`/表号补精确基线。

---

## 一、概览：一条主线

**一句话主线**：LongCat-2.0 的核心赌注不是「换更便宜的注意力去堆更大模型」这一条常规路线，而是**证明「前沿规模训练 + 近前沿 Agentic Coding 能力」可以完全建立在国产 AI ASIC superpod 上**——为此它在架构（LSA 稀疏注意力 + N-gram 稀疏扩参 + ScMoE）、训练（Muon 大规模 + 原生 1M 长上下文）、Infra（6D 并行 + PD 分离 + 确定性算子）三层同时做了「为异构硬件量身」的工程，最终在 >35T tokens 预训练中**零回滚、无不可恢复 loss spike**。

### 1.1 核心参数

| 参数 | 值 | 来源/备注 |
|------|-----|----------|
| 总参数量 | **1.6 T** | 博客/README 多源一致 |
| 每 token 激活参数 | **~48 B** | 博客/README 一致（**未**披露动态区间，见 §9） |
| 预训练 tokens | **> 35 T** | 博客/README 一致 |
| 上下文长度 | **1 M tokens**（原生训练） | 数百亿 token 的 1M 上下文数据 |
| N-gram Embedding | **135 B 参数, n=5** | 与 MoE 正交的稀疏扩参层 |
| MTP | **3-step 模块** | speculative decoding |
| 训练硬件 | **50K+ 国产 AI ASIC** | 全栈非 NVIDIA |
| 训练算力 | **数百万 加速器·小时级** | HF README「millions of accelerator-hours」；博客渲染文本被摘为「accelerator-days」，存 24× 分歧（见 §9） |
| 层数 / 隐藏维 / 每层专家数 / top-k | **未披露** | 博客未给；待权重/config |
| 对标模型 | Claude Opus 4.6/4.7/4.8 · GPT-5.5 · Gemini 3.1 Pro | 评测表列（§8） |

### 1.2 架构总览（ASCII，避免 mermaid 定界符风险）

```
┌──────────────────────────────────────────────────────────────┐
│                 LongCat-2.0  (1.6T total / ~48B act)          │
├──────────────────────────────────────────────────────────────┤
│  Input tokens                                                 │
│    │                                                          │
│    ├── Token Embedding                                        │
│    └── N-gram Embedding (135B, n=5)  ← 与 MoE 正交的稀疏维扩参 │
│    │        · embedding 空间约扩 100×                          │
│    │        · 占总参数预算 < 10%                               │
│    ▼                                                          │
│  Transformer Block × N   (N 未披露)                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LongCat Sparse Attention (LSA)                        │   │
│  │   ├─ SI  Streaming-aware Indexing  (连续访问+随机选择)  │   │
│  │   ├─ CLI Cross-Layer Indexing      (跨相邻层复用索引)   │   │
│  │   └─ HI  Hierarchical Indexing     (块级粗筛→token细选) │   │
│  └──────────────────────────────────────────────────────┘   │
│    │                                                          │
│    ▼                                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  ScMoE  (per-core 显式控制 → dense/MoE 分支全并行)      │   │
│  │   · 从 LongCat-Flash 的「计算-通信重叠」再进一步        │   │
│  └──────────────────────────────────────────────────────┘   │
│    │                                                          │
│    ▼                                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  MTP (3-step)  → speculative decoding                  │   │
│  │   · CLI 延伸：第 2/3 步复用第 1 步的注意力索引          │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘

    训练/部署底座：国产 AI ASIC superpod（单体 ≤48 机 all-to-all）
                   6D 并行 TP/CP/EP/DP/PP + EMBP
```

---

## 二、模型架构

架构自陈**承袭 LongCat-Flash 并「进一步推参数效率」**（措辞为架构继承，而非从 checkpoint 续训；隐含从头预训练）。三个 headline 创新如下。

### 2.1 LongCat Sparse Attention (LSA)：三种正交索引压 1M 上下文

![LSA 总览：左 Owner Layer 跑完整索引——Streaming Tokens 绿→Contiguous KV 约 50% 预算；Non-Streaming Tokens 黄→Block Indexer→Token Indexer 两级 top-k→Non-Contiguous KV 约 50% 预算；右 Reuse Layer 无索引器、直接复用 Owner 层的索引](assets/lsa_overview.png)

> **图源**：官方博客 LSA 总览图（`lsaimage-CCkXmBaN.svg`，原图注「Overview of the LongCat Sparse Attention design. Sink tokens omitted for clarity.」）。原始 SVG 存于 `assets/lsa_overview.svg`，本页 PNG 为 2× 渲染。

**读图（这张图把三种索引一次画全）**：Full KV Tokens 先分成两股——

- **Streaming Tokens（绿）**：**不进索引器**，直接作 **Contiguous KV（约 50% 预算）**。这就是「连续保留」的那一半——StreamingLLM 式的 **sink token + 近窗局部连续段**（图注注明 sink 已略去）；它天然顺序访存、coalesced。
- **Non-Streaming Tokens（黄）**：走 **Block Indexer →(top-k 选块)→ Token Indexer →(块内 top-k 选 token)** 两级筛，得 **Non-Contiguous KV（约 50% 预算）**；两个索引器 **Sharing Parameters**（共享参数）。

两股拼成 Indexed KV，与 Query 做 Attn。**右侧 "LSA from the Reuse Layer" 里没有任何索引器**，顶部标注 **"Directly Reusing the Indices from the Owner Layer"**——直接拿 Owner Layer 算好的索引，只保留 Top-k Selector + Attn。三种正交索引在图上的落点：**SI** = 「Streaming(连续) + Non-Streaming(动态)」这个约 50/50 拆分；**HI** = Non-Streaming 内的 Block→Token 两级；**CLI** = Owner Layer 与 Reuse Layer 之间的索引复用。

**动机（源忠实：LSA 是冲着 DSA 的短板设计的）**：博客把参照系明确指向 **DeepSeek Sparse Attention (DSA) 的 Lightning Indexer**，并点名其两处瓶颈——**输出不连续（output discontinuity）**（选出的 token 在显存里碎片化、随机访存）与**打分的二次方成本（quadratic scoring cost）**。LSA 用三种**相互正交**的索引策略，分别正面修复这两处、再叠一层跨层摊薄：

| 索引 | 机制 | 修复的 DSA 短板 | 训练方式 |
|------|------|----------|----------|
| **SI**（Streaming-aware Indexing） | 把 token 选择预算重塑为「硬件对齐的**连续访问** + 动态随机选择」结合——碎片化随机访存转成可预测**顺序读**，达成 **coalesced HBM 访问** | **输出不连续** | 训练中启用 |
| **CLI**（Cross-Layer Indexing） | 利用「**相邻层注意力显著性经验稳定**」——一次索引 pass 服务多个连续层，摊薄索引成本 | 索引的**逐层重复计算** | 需**跨层蒸馏**训练 |
| **HI**（Hierarchical Indexing） | 两段式 coarse-to-fine：先**块级近似打分粗召回**，再在候选内**细粒度 token 选择**——缩小 indexer 每 query 要处理的候选空间 | **二次方打分成本** | **training-free**，仅对选定的超长上下文任务启用 |

> **为什么不直接用 DSA 的 Lightning Indexer？（源明确点名）** DSA（DeepSeek / GLM-5 的可学习稀疏注意力，见 [[glm_5_analysis]]）用 Lightning Indexer 选 token，但它**输出不连续**会把访存打成随机碎片、**二次方打分**在 1M 下昂贵。LSA 不另起炉灶，而是**针对这两处逐一修复**：SI 管访存形态（→coalesced 访问）、HI 管打分成本（→分层粗筛降二次方）、CLI 再叠一层跨层摊薄。本质是把「稀疏注意力」从「省 FLOPs」重定义为「**省访存 + 省索引**」——这正是带宽受限的国产 ASIC 最吃紧的两处。

#### 读图问答（对着上图逐条澄清 LSA 的常见疑问）

**Q1. SI 的 "streaming token" 是不是就是固定窗口的 token？**
方向对，但要精确说是「**连续保留的那约一半 KV**」。图里 **Streaming Tokens（绿）不进索引器、直接作 Contiguous KV（~50% 预算）**——对应 **sink token + 近窗局部连续段**（StreamingLLM 式；图注专门说 sink 略去）。所以它确实是「固定/连续、总是保留」的部分，但两点补充：① 除局部窗口外还含 **sink**；② 它只占**约一半预算**，另一半（Non-Streaming）是**动态选**出来的。SI 的本质 = 把 KV 预算拆成「一半连续 + 一半动态」，让动态那半也尽量块对齐、可 coalesced。

**Q2. 层次化 indexer 是不是「连续做两次选择」来稀疏化？**
**完全正确**。图中 Non-Streaming 路径就是 **Block Indexer →(top-k 选块)→ Token Indexer →(块内 top-k 选 token)** 两级 top-k：先块级粗召回、再块内 token 级细选。这把索引打分从「对全序列每 token 打分（二次方）」降成「先对块打分、只在选中块里对 token 打分」。两级索引器还 **共享参数**。

**Q3. CLI 的 reuse 是不是「一个 indexer 对应多个 transformer layer、算一次后续复用」？**
**正确**。图右 "LSA from the **Reuse Layer**" **没有 Block/Token Indexer**，只剩一个 Top-k Selector，顶部箭头写明 **"Directly Reusing the Indices from the Owner Layer"**：**Owner Layer 跑一次完整索引得到 indices，后面连续若干 Reuse Layer 直接拿这套 indices，不再自己算**。一次算、多层复用（相邻层注意力显著性稳定是其经验前提）。

**Q4. 这个复用，实现上是缓存还是重算？**
**是缓存，不是重算**——而且从图上看是**结构性必然**：Reuse Layer **根本没有索引器**，它没有可「重算」的东西，只能接收 Owner Layer 传来的 index。所以实现上就是：**Owner Layer 算出的 top-k 索引集合（一个整型 index 张量：块索引 + token 索引）被缓存下来，喂给后续 Reuse Layer 的 Top-k Selector**。
- **关键区分**：被缓存/复用的是**「选哪些 KV」的索引**，**不是注意力结果**——每个 Reuse Layer 仍用**自己这一层的 K/V**、在这套共享索引上算**自己的 Attn**（图里每层都有独立 Attn 框）。省掉的是 indexer 的打分开销（最贵、二次方那块），不是省 attention 本身。
- **为什么不可能是重算**：若每层都重算 indexer，CLI 就没有意义（博客原话「amortize indexing cost」）；MTP 那段也明确用 **"reusing the index set generated in step 1"**——复用的就是**索引集合**。
- **代价/前提**：跨层直接复用索引，要求「相邻层注意力显著性稳定」成立，故训练时用 **cross-layer distillation** 把 reuse layer 对齐到「用 owner 的索引也不掉点」。这是它的代价——多一条训练约束，换推理时把索引成本摊到多层。

### 2.2 N-gram Embedding：135B 参数、与 MoE 正交的「稀疏维」扩参

![N-gram Embedding 总览：当前 token 处取 5/4/3/2-gram（各自 Hash+Embedding+Projection、多张哈希表），与 Base Embedding 相加得最终 Embedding Vector](assets/ngram_embedding_overview.png)

> **图源**：官方博客 N-gram Embedding 总览图（`ngram-emb-new.drawio-DtU8Umnl.svg`）。原始 SVG 存于 `assets/ngram_embedding_overview.svg`。

**读图**：对当前位置（图中 "improvements"），分别取以它结尾的 **2/3/4/5-gram**（如 5-gram = "introduces three orthogonal efficiency improvements"）；每个 n-gram 各过一组 **Hash + Embedding + Projection**（图中叠放的多张卡 = 多张哈希表/桶），再把 5/4/3/2-gram 的向量与 **Base Embedding**（普通 token embedding）**逐一相加**，得最终 **Embedding Vector**。即：**用「哈希查表」而非计算，把多 token 组合（n-gram）的信息直接注入到输入 embedding**——这正是「几乎不增 FLOPs、参数长在稀疏查表维」的由来；135B 参数就活在这些哈希 embedding 表里。

**机制**：在 Token Embedding 之外并联一个 **n=5 的 N-gram Embedding 层**，参数量达 **135B**，把 embedding 表征空间约**扩大 100×**，同时**控制在总参数预算的 <10%**。

**为什么这么设计**（四拍）：
- **动机（源忠实：MoE 稀疏度已「过了甜点」）**：博客明确 MoE 的稀疏度**已越过甜点区（约 97% 稀疏）**——再堆专家边际收益递减、且继续推高 all-to-all 与路由负担；此时把 135B 参数**挪到 N-gram Embedding，其收益「远超标准专家」**。于是转向**另一个正交维度**廉价扩参。
- **机制**：N-gram embedding 在「**与 MoE 正交的稀疏维度**」扩参——n=5 的 n-gram 组合查表而非计算，几乎不增加 FLOPs，把 embedding 表征空间约扩 100×。
- **证据/收益**：博客称其**提升参数效率、并降低大 batch 解码时的 I/O**——把参数从专家挪到 N-gram Embedding，大 batch 解码的显存 I/O 下降、生成加速（查表命中率高、访存规整）。
- **代价/为什么不选替代**：不是简单加大 vocab embedding（会线性放大主 embedding 访存），而是 n-gram 组合稀疏查表；代价是需要额外的 N-gram 索引结构与**专门的并行维（EMBP，见 §5）** 来分片这 135B。

### 2.3 ScMoE：从「重叠」到「全并行」的计算-通信

**机制**：ScMoE 承袭 LongCat-Flash 的计算-通信重叠思路，但通过 **per-core（每计算核）显式控制**，把 **dense 分支与 MoE 分支从「重叠执行」推进到「完全并行执行」**。

> **为什么关键**：MoE 的 all-to-all（分发/回收 token）是长延迟通信；若 dense 与 MoE 只是「部分重叠」，通信仍会拖尾。ScMoE 用显式 per-core 调度让两分支真正并行跑满，是 §5 中「+35% 训练吞吐」的架构侧来源之一。（EP 的 all-to-all 原理见 [[expert_parallel_analysis]] / [[megatron_ep_analysis]]。）

### 2.4 MTP（Multi-Token Prediction）：3-step + 复用 LSA 索引

- **深度**：3-step 模块，用于**投机解码（speculative decoding）** 加速。
- **与 LSA 的协同**：MTP 的第 2、3 步**复用第 1 步的注意力索引**（CLI 的延伸）——多预测的 token 不再重复做索引，进一步压低投机解码的开销。
- MTP 概念与 DeepSeek-V3 一脉相承（见 [[deepseek_v3_analysis]]）。

---

## 三、预训练

| 项 | 内容 |
|----|------|
| 规模 | **> 35T tokens**；数百万加速器·小时级；50K+ 国产 ASIC |
| 长上下文 | **数百亿 token 的 1M 上下文数据**；**all-gather 式 CP 可扩到 512+**，实现**原生 1M 长度训练**（而非纯外推） |
| 数据流水 | 训练数据在 **get-batch 阶段 reshuffle**，用**均衡 CP 策略**分片（配合 CP-512 的负载均衡） |
| 优化器 | **Muon**，大规模部署（详见 §3.1） |
| 基座关系 | 架构承袭 **LongCat-Flash**、推参数效率；措辞为架构继承（隐含从头训练，非续训） |
| 学习率/batch/warmup/课程阶段 | **未披露** |

### 3.1 Muon 优化器的大规模工程

博客明确「**在加速器上大规模部署 Muon**」，并做了三项针对性优化：

1. **TP 并行下的处理**——Muon 的正交化/Newton-Schulz 迭代涉及矩阵运算，需在张量并行切分下正确高效执行；
2. **DP 状态冗余消除**（DP state redundancy removal）——去掉数据并行副本间的优化器态冗余（类 ZeRO 思路，见 [[zero_fsdp_analysis]]）；
3. **高效对称矩阵乘 kernel**（symmetric matmul kernel）——为 Muon 的 $G G^\top$ 类运算定制。

> Muon 已成为 2026 年前沿国产/开源大模型的共同选择：Kimi K2 的 **MuonClip**（见 [[kimi_k2_analysis]]）、GLM-5 的 **Muon Split**（见 [[glm_5_analysis]]）、DeepSeek-V4（见 [[deepseek_v4_analysis]]）。LongCat-2.0 的贡献点在**异构 ASIC 上把 Muon 跑到 1.6T 规模并做 TP/DP/kernel 三处适配**。Muon 原理见 [[muon_analysis]]。

---

## 四、后训练：MOPD 多教师在线策略蒸馏

**主线**：不追求单一「全能」策略，而是**先训三组各有所长的 teacher 专家群，再用 MOPD 把三者的最强能力融进一个学生模型**。

![MOPD 多专家后训练架构总览：Agent / Reasoning / Interaction 三组 teacher expert 群，经 MOPD 融合蒸馏进统一学生模型](assets/mopd_overview.png)

> **图源**：官方博客 MOPD 架构图（`mopd-CIX9ZFo9.svg`）。原始 SVG 存于 `assets/mopd_overview.svg`。

**读图**：从 **LongCat SFT 检查点**出发，分头训练三组专精 teacher，再经 **MOPD** 融合蒸馏成 **LongCat 2.0（Unified, Advanced）**。图中 MOPD 副标题写作 **"Multi-Teacher On-Policy Distill"**，两条职责标注为 **Real-World Scenarios**（整合 agentic 执行 / 推理 / 交互能力）与 **Domain Expert Integration**（融合各专精专家能力）。

> [!contradiction] MOPD 的展开以官方图为准：Multi-Teacher On-Policy Distillation
> 二手摘要（DeepWiki 等）曾把 MOPD 解作「**Multi-Objective Policy Distribution**（多目标策略分布）」；但**官方博客架构图**副标题白纸黑字写的是 **"Multi-Teacher On-Policy Distill(ation)"（多教师在线策略蒸馏）**。以官方图为准——本页早期版本与首次 changelog 的「多目标策略分布」为二手误读，特此订正。

- **MOPD = Multi-Teacher On-Policy Distillation（多教师在线策略蒸馏）**：以三组 teacher 为多教师、对**学生自己生成的轨迹（on-policy）** 做蒸馏，把三者能力融进统一学生。**on-policy 是关键**——在学生自身分布上蒸馏（而非离线照抄 teacher 输出），能针对学生**真实会犯的错**纠偏、规避训练-推理分布错配（与 [[RL_Training_Inference_Precision_Analysis]] 同源问题）。
- **三组 teacher expert groups**（原子能力据官方图逐一列全）：

| 专家群 | 目标域 | 优化的「原子能力」（图中列举） |
|--------|--------|--------------------|
| **Agent Experts** | 细粒度垂域：**代码 / 工作 / 搜索** | Tool Use（工具调用）· API Parsing（参数解析）· Self-Correction（自纠正） |
| **Reasoning Experts** | **数学 / STEM / 多跳推理** | Multi-Hop Reasoning · STEM Reasoning · **Adaptive Computation（自适应算力）** |
| **Interaction Experts** | 交互 / 通用对话 | Instruction Following（指令遵循）· Human Alignment（人类对齐）· Hallucination Suppression（幻觉抑制） |

> **为什么不用单教师/单目标 RL？** Agentic coding、深推理、通用交互三者的**奖励信号与数据形态差异极大**，混在一个策略里互相拉扯（reward 冲突）。先分群把各自「原子能力」练到位、再蒸馏融合，是把「多目标对齐」从「一个策略硬扛」改成「分而治之 + 融合」。这与 GLM-5 的「分阶段 RL + 跨阶段蒸馏防遗忘」（见 [[glm_5_analysis]] §三）思路相通。
>
> **仍未披露**：on-policy 蒸馏的**具体损失形式**（KL / reverse-KL / 排序？）、是否含显式 RL（奖励模型 / GRPO 等）、各 teacher 群的训练细节与数据量——博客与图均未给。

---

## 五、AI Infra：为国产 ASIC 量身的 6D 并行 + PD 分离

**主线**：整套训练与大规模部署**完全建立在国产 AI ASIC superpod 上**（非 NVIDIA），Infra 的每一处都在补「异构硬件 + 超大稀疏模型 + 1M 上下文」的短板。

### 5.1 训练：6D 并行 + superpod 拓扑

```
物理底座：国产 AI ASIC superpod
  · 单个 superpod ≤ 48 机，机内 all-to-all 高带宽
  · superpod 间 RoCE 互联  →  额外约 +30% 预训练吞吐
  · 相对 naive 实现总体  →  +35% 训练吞吐

6D 并行 = 标准 5D (TP / CP / EP / DP / PP)  +  EMBP
  · TP/CP/EP：把高带宽通信域「加宽到数百设备」
  · CP（上下文并行）：扩到 512+  →  原生 1M 训练
  · EMBP（Embedding Parallelism）：专门并行 135B N-gram Embedding
```

- **EMBP 是本模型独有的第 6 维**：135B 的 N-gram Embedding 若挤在 TP/DP 里会破坏负载均衡，故单列一维专门分片。这是「架构创新（N-gram 扩参）倒逼 Infra 创新（新并行维）」的典型。
- 相关原理：TP/CP/SP 见 [[tensor_sequence_parallel_analysis]]，EP 见 [[expert_parallel_analysis]]，PP 见 [[pipeline_parallel_analysis]]，通信重叠工程参照 [[megatron_comm_overlap_analysis]]。

### 5.2 推理·模型专属优化（Model-Specific）

针对「1.6T 参数 × 1M 上下文 × HBM 受限」的解码，博客列了几招（多为吸收/流水线类的硬件友好改写）：

- **注意力吸收计算（absorb computation）**：把注意力里可合并的投影/缩放**吸收进相邻矩阵**，减少解码时的显存读写与算子数（*本页推断：类 MLA 家族的 absorb 技巧*）。
- **索引器流水线化（pipelining the indexer）**：把 LSA 的 indexer 与后续注意力**流水线重叠**，让「选哪些 KV」的开销藏进计算。
- **KV-cache 并行（KVP）**：把超长上下文 KV cache 切到多设备，缓解单卡 HBM 压力（与下方部署段 decode 侧呼应）。
- **ScMoE 调度前推**：把架构侧的 ScMoE（§2.3）在推理调度上进一步优化，维持 dense/MoE 分支并行。

### 5.3 推理·部署：Prefill–Decode 分离

| 阶段 | 并行/技术 | 目标 |
|------|-----------|------|
| **Prefill 节点** | 多节点 **CPP**（chunked pipeline parallelism）+ **Attention SP**（sequence parallelism） | 压低 **TTFT** |
| **Decode 节点** | **KVP**（KV-cache parallelism）+ **EP128**（128 路专家并行） | 提高 **TPOT/吞吐** |
| P↔D 之间 | KV-cache 传输走**内置 200 Gbps 网卡** | 平衡 TTFT 与 TPOT |

> PD 分离是 2026 年大模型推理的主流范式（Kimi Mooncake 见 [[moonshot_kimi/index]]；vLLM 见 [[02_engineering/03_infer_frameworks/vllm/index]]）。LongCat-2.0 的特色是 **decode 侧 EP128 + KVP** 与**国产网卡 200Gbps 的 KV 搬运**。

### 5.4 推理·加速器导向优化（Kernel / 访存）

- **Super Kernels**：把多个小算子融进一个大 kernel，**降 kernel launch 开销**（国产 ASIC 上 launch 开销尤其敏感）。
- **权重预取（weight prefetch）/ L2 cache 预取**：把某算子的 **I/O（权重加载）延迟藏进前一个算子的计算**里，隐藏访存延迟。
- **EPLB**（Expert-Parallel Load Balancing）：部署期专家负载均衡。
- （P↔D 间 KV-cache 走内置 **200 Gbps 网卡**——见 §5.3 表。）

---

## 六、低精度与数值可靠性（重要的源忠实澄清）

> [!contradiction] 「低精度」在 LongCat-2.0 语境下 ≠ FP8/FP4 量化训练
> 用户常把「低精度」等同于 DeepSeek-V3 的 **FP8 训练**（见 [[low_precision_training_analysis]] / [[deepseek_v3_analysis]]）或 GLM-5 的 **INT4 QAT**（见 [[glm_5_analysis]]）。但**官方博客通篇未提 FP8/FP4/BF16 的训练或推理量化**。LongCat-2.0 的「精度」叙事完全落在**另一侧面：国产 ASIC 上的数值可靠性 / 可复现性**。这是一个诚实且重要的区分——若权重/技术报告后续披露 FP8，再补正。

在国产 ASIC 上「让数值可信」是本模型稳定训练的前提，具体手段（均归在 Determinism & Reliability）：

- **确定性算子套件**：自研确定性算子/模块，**覆盖 Embedding、FlashAttention、LSA、MoE** 层，保证**通信与计算双路径**的确定性（bitwise 可复现）。
- **二叉树分段累加**（binary-tree segmented accumulation）：所有 **reduction 类算子**用此策略**降低浮点误差累积**——大规模求和顺序敏感，分段树累加把误差控制住。
- **对齐高精度基线验证**：在**真实 LLM 负载**下，把 ASIC 的计算精度**对齐一个严格的高精度基线**做验证（确认国产芯片算得「对」）。

> 与 RL 训练里「训练-推理精度不一致」问题（见 [[RL_Training_Inference_Precision_Analysis]]）同源：都是「同一模型在不同执行路径上数值必须一致」。LongCat-2.0 把这条做到训练算子级的 bitwise 确定性。

---

## 七、训练稳定性：>35T tokens 零回滚

**成果陈述**：>35T tokens 预训练**无回滚、无不可恢复 loss spike**——在**非 NVIDIA 的异构硬件**上做到这点，是博客反复强调的核心可信度证据。

支撑手段：

1. **数值确定性**（§6）——bitwise 可复现是排障与稳定的基础（能复现才能定位 spike）。
2. **二叉树分段累加**——抑制大规模 reduction 的浮点误差累积（误差累积是慢性发散源）。
3. **Bit-flip 检测**：在**选定的计算密集算子**里引入 bit-flip 检测，**及时捕获硬件位翻转异常**（国产 ASIC 大集群下硬件比特翻转是真实风险）。
4. **端到端监控与自动恢复**：端到端监控驱动**故障识别 → 流量切换 → 恢复**，**无需人工介入**。

> **为什么在国产 ASIC 上稳定性是「硬骨头」**：成熟 NVIDIA 栈的确定性/容错工具链多年沉淀；换到国产 ASIC 需**自建**确定性算子、精度验证、bit-flip 检测与自动容错整链。LongCat-2.0 把这条整链做通，本身就是「前沿规模训练可迁移到替代硬件」的最强论据。

---

## 八、评测结果

**诚实框架**：LongCat-2.0 定位**开源、近前沿的 Agentic Coding 模型**。在多数 **code/agent** 项上打平或**超过 GPT-5.5、Gemini 3.1 Pro**，但**整体仍落后于最强的 Claude Opus 4.8**；其真正卖点是**在受限的国产算力上**做到了这一梯队。全部分数归一化到 0–100；带 `*` 为外部/报告口径，`†` FORTE 有 45 分钟超时限制。

| 类别 | Benchmark | **LongCat-2.0** | Gemini 3.1 Pro | GPT-5.5 | Claude Opus 4.6 | Claude Opus 4.7 | Claude Opus 4.8 |
|------|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Code Agent** | Terminal-Bench 2.1 | 70.8 | 70.7* | 73.8* | — | 71.7* | 78.9* |
| | **SWE-bench Pro** | **59.5** | 54.2* | 58.6* | 57.3* | 64.3* | 69.2* |
| | SWE-bench Multilingual | 77.3 | 76.9* | — | 77.8* | 80.5* | 84.8* |
| **General Agent** | FORTE † | 73.2 | 70.3 | 77.8 | 73.2 | 77.6 | 77.2 |
| | BrowseComp | 79.9 | 85.9* | 84.4* | 84.0* | 79.3* | 84.3* |
| | RWSearch | 78.8 | 76.3 | 85.3 | 81.3 | 79.3 | 77.3 |
| **Foundational** | IFEval | 90.0 | 96.1 | 95.0 | 92.2 | 88.7 | 86.0 |
| | Writing Bench | 83.8 | 83.7 | 84.7 | — | 85.3 | 85.2 |
| | IMO-AnswerBench | 81.8 | 90.0 | 79.5 | 75.3* | 81.8 | 75.3 |
| | GPQA-diamond | 88.9 | 94.3* | 93.6* | 91.3* | 94.2* | 92.4 |

**读表要点**：
- **代码/Agent 是长板**：SWE-bench Pro **59.5 > GPT-5.5 58.6**、> Gemini 3.1 Pro 54.2；Terminal-Bench 2.1 70.8 与 Gemini 持平、略低于 GPT-5.5；SWE-bench Multilingual 77.3 与 Gemini 持平。
- **对最强 Claude Opus 4.8 仍有系统性差距**：SWE-bench Pro 69.2、Multilingual 84.8、Terminal 78.9 全面领先。
- **基础项互有胜负**：IFEval（指令遵循）90.0 落后于 Gemini/GPT/Opus4.6/4.7；但 **IMO-AnswerBench 81.8 反超 GPT-5.5(79.5) 与 Opus4.8(75.3)**；GPQA-diamond 88.9 落后前沿闭源。
- 结论：**在开源阵营与「受限国产算力」这两个约束下，LongCat-2.0 达到近前沿**，尤其 agentic coding 最能打。

### 附：官方能力演示 showcase（3 个场景，定性非基准）

博客在评测前用三个场景做定性演示：

- **Codebase Migration（代码库迁移）**：读入**完整代码库 + 迁移文档**，映射整体架构，把插件**改写迁移到新 SDK**——一次演示「1M 长上下文 + agentic coding」的端到端闭环。
- **Agentic & Research（智能体与研究）**：多步工具调用 / 搜索的自主任务执行。
- **Content Generation（内容生成）**：通用写作类生成。

---

## 九、源忠实修正与未披露项

### 9.1 与二手「常识」冲突处（以官方源为准）

> [!contradiction] 「动态激活 33–56B / zero-compute experts」——博客未如此陈述
> 部分二手报道（如媒体解读）称 LongCat-2.0「**动态激活 33B–56B 参数 / 用 zero-compute experts 跳过计算**」。核对官方博客：**激活参数就是 ~48B**，博客**未披露任何推理期动态激活区间**；提到 zero-expert 之处仅是**训练期把 padding token 路由到 zero-expert 以省显存**，**不是** LongCat-Flash 式的推理期动态算力分配机制。判断：二手报道疑似**把 LongCat-Flash 的 zero-compute experts（其激活确有动态区间）张冠李戴到 2.0**。本页只采信博客的「~48B 激活」。

> [!contradiction] 训练算力单位：accelerator-hours vs accelerator-days
> HF/GitHub `README.md` 写「millions of **accelerator-hours**」；官方博客经渲染代理提取后被摘为「millions of **accelerator-days**」，二者差 24×。用 `6·N_act·D ≈ 6×48e9×35e12 ≈ 1e25` FLOPs、除以「50K ASIC × 合理有效算力」粗算，**「数百万加速器·小时」量级自洽**，「天」高约一个数量级。本页取 README 口径（**加速器·小时**），待技术报告确认。

### 9.2 博客未披露、需后续补的量

> **已做完整大纲审计（2026-07-03）**：逐节比对博客全部章节（Introduction / Architecture / Scalable Infrastructure / Learning from Multiple Teachers / Capability Demonstration / Evaluations）后确认，下列项目**在正文任何位置都未出现**，非本页漏读。

- 模型细节：**层数、隐藏维、注意力头数/头维、每层 routed/shared 专家数、top-k、专家中间维**；
- 结构选型（大纲审计确认未提）：**激活函数、归一化类型、位置编码（RoPE/NoPE）、词表大小、tokenizer**；
- 训练细节：**学习率/batch/warmup、上下文扩展的分阶段课程、预训练是否分 stage、数据配比（代码/数学/多语/网页占比）**；
- 精度：**是否使用 FP8/FP4/BF16 混合精度训练**（博客只讲数值可靠性，未讲低比特量化）；
- 后训练：MOPD 已知为**多教师 on-policy 蒸馏**（§4），但**蒸馏损失形式、是否含显式 RL/奖励设计、各专家群训练数据量**未披露；
- 推理：**吞吐（tokens/s）/ MFU / 成本**等数字未披露；
- 权重与 config.json（「coming soon」）、正式技术报告/arXiv（截至 2026-07-03 未见）。**已核实**：HF/GitHub 仓库 `main` 分支当前仅 `README.md`(2.86 kB) + `figures/` + `LICENSE`，**无 `config.json` / 无建模代码**——故上述模型细节的硬数字目前从任何官方渠道都不可得，非本页遗漏；待权重放出即可从 config 一次性补全。

> 按本库 Query Workflow：待 raw 源（权重/config/技术报告）到位后，回到源用 `file:line`/表号把上述项补成精确基线，并更新本页头 Baseline。

---

## Related Pages

**同域模型（对比阅读）**：
- [[meituan_longcat/index]] — 美团 LongCat 家族总览（本页所属家族入口）
- [[glm_5_analysis]] — GLM-5：MoE + Muon Split + DSA 稀疏注意力 + INT4 QAT（最相近的对照）
- [[kimi_k2_analysis]] — Kimi K2：1T MoE + MuonClip + Agentic RL
- [[deepseek_v3_analysis]] — FP8 训练 · MTP · 671B MoE（低精度/MTP 对照）
- [[deepseek_v4_analysis]] — CSA/HCA 稀疏注意力 · Muon · 1.6T MoE
- [[deepseek_moe_analysis]] — MoE 路由与负载均衡原理

**技术原理（机制交叉链接）**：
- [[muon_analysis]] — Muon 优化器原理
- [[low_precision_training_analysis]] — FP8 低精度训练（与本模型「数值可靠性」路线对照）
- [[RL_Training_Inference_Precision_Analysis]] — 训练-推理精度一致性
- [[expert_parallel_analysis]] · [[tensor_sequence_parallel_analysis]] · [[pipeline_parallel_analysis]] — 6D 并行的原理层
- [[megatron_ep_analysis]] · [[megatron_comm_overlap_analysis]] — EP 与通信重叠工程
- [[02_engineering/03_infer_frameworks/vllm/index]] — PD 分离推理（工程对照）

**上级索引**：
- [[01_theory/01_models/index]] — 模型架构与家族总索引
- [[01_theory/index]] — 理论研究总览
