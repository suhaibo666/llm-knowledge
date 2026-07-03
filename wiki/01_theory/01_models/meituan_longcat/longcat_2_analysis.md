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

**动机**：1M 上下文下，注意力的**显存带宽/索引开销**（选哪些 KV 参与注意力）成为主瓶颈；直接 top-k 选 token 会产生**碎片化随机访存**，对硬件极不友好。LSA 用三种**相互正交**的索引策略分别攻不同代价：

| 索引 | 机制 | 攻的代价 | 训练方式 |
|------|------|----------|----------|
| **SI**（Streaming-aware Indexing） | 把 token 选择重塑为「硬件对齐的**连续访问** + 动态随机选择」结合——把碎片化随机访存转成**顺序读** | 访存带宽/局部性 | 训练中启用 |
| **CLI**（Cross-Layer Indexing） | 利用「**相邻层注意力显著性经验稳定**」——一次索引服务多个连续层，摊薄索引成本 | 索引重复计算 | 需**跨层蒸馏**训练 |
| **HI**（Hierarchical Indexing） | 两段式「**块级近似打分 → 细粒度 token 选择**」的 coarse-to-fine | 打分粒度/长尾超长任务 | **training-free**，仅对选定的超长上下文任务启用 |

> **为什么不选朴素稀疏注意力？** 朴素 top-k 稀疏注意力省了 FLOPs 却把访存打成随机碎片，在带宽受限的加速器上得不偿失；LSA 的三招都在「**让稀疏对硬件友好**」这条线上——SI 管访存形态、CLI 管跨层复用、HI 管超长任务的分层——是把「稀疏注意力」从「省算」重定义为「省访存 + 省索引」。这与 DeepSeek DSA / GLM-5 DSA（见 [[glm_5_analysis]]）同属「可学习稀疏注意力」大类，但 LSA 的三正交轴与「跨层复用 + 硬件对齐」是其区分点。

### 2.2 N-gram Embedding：135B 参数、与 MoE 正交的「稀疏维」扩参

**机制**：在 Token Embedding 之外并联一个 **n=5 的 N-gram Embedding 层**，参数量达 **135B**，把 embedding 表征空间约**扩大 100×**，同时**控制在总参数预算的 <10%**。

**为什么这么设计**（四拍）：
- **动机**：MoE 已在「专家维」扩参，但继续堆专家会推高 all-to-all 通信与路由负担；团队想在**另一个维度**廉价扩参。
- **机制**：N-gram embedding 在「**与 MoE 正交的稀疏维度**」扩参——查表而非计算，几乎不增加 FLOPs。
- **证据/收益**：博客称其**提升参数效率、并降低大 batch 解码时的 I/O**（查表命中率高、访存规整）。
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

## 四、后训练：MOPD 多教师蒸馏

**主线**：不追求单一「全能」策略，而是**先训三组各有所长的 teacher 专家群，再用 MOPD 把三者的最强能力融进一个学生模型**。

- **MOPD** = Multi-Objective Policy Distribution（多目标策略分布），负责整合三组专家群的最强能力。
- **三组 teacher expert groups**：

| 专家群 | 目标域 | 优化的「原子能力」 |
|--------|--------|--------------------|
| **Agent Experts** | 细粒度垂域：**代码 / 工作 / 搜索** | 精确工具调用、可靠参数解析、自纠正机制 |
| **Reasoning Experts** | **数学 / STEM 解题 / 多跳推理** | 长链推理 |
| **Interaction Experts** | 交互 / 通用对话 | （博客侧重前两者，交互群细节较略） |

> **为什么不用单教师/单目标 RL？** Agentic coding、深推理、通用交互三者的**奖励信号与数据形态差异极大**，混在一个策略里互相拉扯（reward 冲突）。先分群把各自「原子能力」练到位、再蒸馏融合，是把「多目标对齐」从「一个策略硬扛」改成「分而治之 + 融合」。这与 GLM-5 的「分阶段 RL + 跨阶段蒸馏防遗忘」（见 [[glm_5_analysis]] §三）思路相通。
>
> **未披露**：具体 RL 算法（GRPO/PPO 变体？）、奖励模型设计、蒸馏损失形式、各群数据量——博客未给。

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

### 5.2 推理：Prefill–Decode 分离

| 阶段 | 并行/技术 | 目标 |
|------|-----------|------|
| **Prefill 节点** | 多节点 **CPP**（chunked pipeline parallelism）+ **Attention SP**（sequence parallelism） | 压低 **TTFT** |
| **Decode 节点** | **KVP**（KV-cache parallelism）+ **EP128**（128 路专家并行） | 提高 **TPOT/吞吐** |
| P↔D 之间 | KV-cache 传输走**内置 200 Gbps 网卡** | 平衡 TTFT 与 TPOT |

> PD 分离是 2026 年大模型推理的主流范式（Kimi Mooncake 见 [[moonshot_kimi/index]]；vLLM 见 [[02_engineering/03_infer_frameworks/vllm/index]]）。LongCat-2.0 的特色是 **decode 侧 EP128 + KVP** 与**国产网卡 200Gbps 的 KV 搬运**。

### 5.3 Kernel/访存优化

- **Super Kernels**：把多个小算子融进一个大 kernel，**降 kernel launch 开销**（国产 ASIC 上 launch 开销尤其敏感）。
- **L2 cache 预取**：把某算子的 **I/O 延迟藏进前一个算子的计算**里，隐藏访存延迟。
- **EPLB**（Expert-Parallel Load Balancing）：部署期专家负载均衡。

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

---

## 九、源忠实修正与未披露项

### 9.1 与二手「常识」冲突处（以官方源为准）

> [!contradiction] 「动态激活 33–56B / zero-compute experts」——博客未如此陈述
> 部分二手报道（如媒体解读）称 LongCat-2.0「**动态激活 33B–56B 参数 / 用 zero-compute experts 跳过计算**」。核对官方博客：**激活参数就是 ~48B**，博客**未披露任何推理期动态激活区间**；提到 zero-expert 之处仅是**训练期把 padding token 路由到 zero-expert 以省显存**，**不是** LongCat-Flash 式的推理期动态算力分配机制。判断：二手报道疑似**把 LongCat-Flash 的 zero-compute experts（其激活确有动态区间）张冠李戴到 2.0**。本页只采信博客的「~48B 激活」。

> [!contradiction] 训练算力单位：accelerator-hours vs accelerator-days
> HF/GitHub `README.md` 写「millions of **accelerator-hours**」；官方博客经渲染代理提取后被摘为「millions of **accelerator-days**」，二者差 24×。用 `6·N_act·D ≈ 6×48e9×35e12 ≈ 1e25` FLOPs、除以「50K ASIC × 合理有效算力」粗算，**「数百万加速器·小时」量级自洽**，「天」高约一个数量级。本页取 README 口径（**加速器·小时**），待技术报告确认。

### 9.2 博客未披露、需后续补的量

- 模型细节：**层数、隐藏维、每层 routed/shared 专家数、top-k、专家中间维**；
- 训练细节：**学习率/batch/warmup、上下文扩展的分阶段课程、预训练是否分 stage**；
- 精度：**是否使用 FP8/FP4/BF16 混合精度训练**（博客只讲数值可靠性，未讲低比特量化）；
- 后训练：**MOPD 的具体 RL 算法、奖励设计、蒸馏损失、各专家群数据量**；
- 权重与 config.json（「coming soon」）、正式技术报告/arXiv（截至 2026-07-02 未见）。

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
