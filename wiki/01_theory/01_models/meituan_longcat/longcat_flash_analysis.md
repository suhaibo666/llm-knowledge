# LongCat-Flash：用「零计算专家 + ScMoE 短路」把 560B MoE 的激活压到 ~27B

> **来源**: LongCat-Flash Technical Report（美团 LongCat 团队）
> **arXiv**: [2509.01322](https://arxiv.org/abs/2509.01322) v1（2025-09-01）
> **开源**: HF `meituan-longcat/LongCat-Flash-Chat`（含 `config.json` + `modeling_longcat_flash.py`，trust_remote_code）· MIT
> **Baseline**: arXiv 2509.01322v1 + `LongCat-Flash-Chat/config.json`（访问 2026-07-06）
> **维度**: Overview + 机制级深挖（架构 / 缩放与稳定性 / 预训练 / Infra&推理 / 后训练&Agentic / 效果 / 与 2.0 的演进）
> **定位**: **LongCat-2.0 的架构前身**——ScMoE 短路与零计算专家在此首创，2.0 在其上加 LSA/N-gram 并迁到国产 ASIC（对照见 [[longcat_2_analysis]]）

---

## 一、概览：一条主线

**一句话主线**：LongCat-Flash 的核心赌注是**「按 token 难度动态分配算力」**——用 **560B 总参**换表达力、却靠**零计算专家（zero-computation experts）**让每 token 平均只激活 **~27B**（18.6–31.3B 动态区间），再用 **ScMoE 短路**把 MoE 的通信藏进计算、把推理 TPOT 砍到约 DeepSeek-V3 的一半。省下的算力预算全部投向**Agentic 工具使用**——这是它区别于「堆参数」路线的本质。

### 1.1 核心参数（据 released `config.json`）

| 参数 | 值 | 定位 |
|------|-----|------|
| 架构类名 | `LongcatFlashForCausalLM` | config.json |
| 总参数 / 每 token 激活 | **560 B / 18.6–31.3 B（均值 ~27 B，动态）** | 报告 摘要 / §2.1 |
| 层数 num_layers | **28** | config.json |
| 隐藏维 hidden_size | **6,144** | config.json |
| 注意力 | **MLA**（q_lora 1,536 / kv_lora 512 / qk_nope 128+rope 64 / v 128, 64 heads） | config.json · 报告 §2.4 |
| 稠密 FFN 中间维 / 专家 FFN 中间维 | **12,288 / 2,048** | config.json |
| 路由专家 / 零计算专家 | **512 routed + 256 零计算(identity)** | config.json · 报告 §2.1 |
| 每 token top-k / routed_scaling | **12** / **6.0** | config.json |
| 词表 / 归一化 / 激活 | **131,072** / RMSNorm(1e-5) / SiLU | config.json |
| 位置编码 / 上下文 | RoPE(θ=1e7，无 YaRN) / **128 K** | config.json |
| MTP | 单个**稠密**头（非 MoE），投机解码接受率 **>90%** | 报告 §2.4 |
| 预训练 | **20 T tokens / 30 天 / 98.48% 可用率** | 报告 §3 |
| 训练/部署硬件 | **NVIDIA H800**（>100 TPS，$0.70/M output） | 报告 §5 |

### 1.2 与 LongCat-2.0 的一句话关系

**Flash = MLA + ScMoE 短路 + 零计算专家**（本页）；**2.0 = Flash + LSA 稀疏注意力 + N-gram Embedding + 更大（38 层 / 768+128 专家）+ 迁到国产 ASIC**。共享的 ScMoE/零计算专家机制在 SGLang `longcat_flash.py` 里两代同一份代码（2.0 复用 Flash 模型类）——详见 §八 与 [[longcat_2_analysis]]。

---

## 二、模型架构

每个解码层 = **2 个 MLA 注意力块 + 异构 FFN（稠密 FFN 与 MoE 专家）**，接成 ScMoE 短路（结构与 [[longcat_2_analysis]] 图 1 右同构，仅专家数/规模不同）。两个 headline 创新如下。

### 2.1 零计算专家（Zero-computation Experts）：按 token 动态分配算力

**机制（报告 §2.1, Eq.1）**：每层专家池 = **512 个真实 FFN 专家 + 256 个「零计算专家」**，router 对全体 768 打分、每 token 选 **top-12**。零计算专家的输出**就是输入本身**（恒等直通 $\text{FFN}_{\text{zero}}(x_t)=x_t$），**不产生任何计算**。于是：

> 一个 token 若被路由到 $j$ 个零计算专家，它这一层实际只做 $12-j$ 个真实专家的 FFN——**激活的算力随 token 难度自适应**，简单 token 少算、难 token 多算。这就是 18.6–31.3B 动态区间、~27B 均值的来源。

**怎么把均值稳定控在目标（报告 §2.1, Eq.2-5）**：
- **PID 控制器**逐步更新每个专家的偏置 $b_i$（Eq.2），把「平均激活的真实 FFN 专家数」拉到目标 $K_e$，让零计算专家吸收多余预算——**免辅助损失、又稳住算力**。
- **设备级负载均衡损失**（Eq.3-5）把零计算专家当作一个额外分组，约束真实专家组与零专家组各自的目标占比，防止路由塌缩。

> **为什么不用「共享专家 + 固定 top-k」（DeepSeek 式）？** 固定 top-k 给每个 token **等量**算力，与「token 难度差异极大」的事实不符。零计算专家把「激活多少」变成**可学习、可控**的量：既保留 MoE 的稀疏性，又拿回「按需分配」的自由度。代价是要额外用 PID + 均衡损失把预算钉住——否则路由会漂。

### 2.2 Shortcut-connected MoE（ScMoE）：把 MoE 通信藏进计算

**机制（报告 §2.2）**：MoE 层的 **dispatch/combine 全对全通信**是长延迟瓶颈。ScMoE 引入一条**跨层短路**——把**前一块的稠密 FFN 计算**与**当前 MoE 层的分发/回收通信**重排为**并行执行**，从而**扩大计算-通信重叠窗口**（比「仅靠共享专家做重叠」的窗口大得多），并支持 token 维度的细粒度切分并发。

**收益（报告 §2.2, Fig.4）**：推理**理论 TPOT 较 DeepSeek-V3 降低近 50%**；且 Fig.4 显示训练 loss 曲线与非短路版**完全重合**——**质量中性**（纯系统收益、不损精度）。

> ScMoE 是 §五「>100 TPS / $0.70 每百万 token」的架构底座。它也是 2.0「per-core 显式控制 → dense/MoE 全并行」的前身。EP 全对全原理见 [[expert_parallel_analysis]] / [[megatron_ep_analysis]]。

### 2.3 MLA + 方差对齐（让大规模训练稳的关键小细节）

- **注意力是 MLA**（报告 §2.4 · config.json）：64 heads、低秩压缩 q_lora 1,536 / kv_lora 512(+rope 64)、v 128——与 DeepSeek MLA 同构，KV cache 省显存。**LongCat-Flash 已用 MLA**（2.0 只是在其上再加 LSA 稀疏索引）。
- **MLA 尺度修正（Eq.6-7）**：低秩分量与全维分量的**方差不匹配**会拖累训练；用 $\alpha_q=\sqrt{d_{\text{model}}/d_q}$、$\alpha_{kv}=\sqrt{d_{\text{model}}/d_{kv}}$ 两个缩放因子中和之。
- **专家初始化补偿（Eq.8）**：细粒度专家切分带来门控稀释/降维，用 $\gamma=m$ 缩放聚合后的专家输出补偿。

> 这些「方差对齐」看似小，却是 560B 规模能**一次训成、无 spike** 的必要条件（配合 §三 的稳定性套件）。

### 2.4 MTP（Multi-Token Prediction）

- **单个稠密层**的 MTP 头（**非 MoE**，报告 §2.4），mid-training 引入；投机解码**接受率 >90%**——是 §五 高 TPS 的另一半来源。
- 概念与 DeepSeek-V3 一脉（见 [[12_deepseek_v3_analysis]]）；2.0 把它扩到 3-step 并让草稿步共用 LSA 索引。

---

## 三、缩放与训练稳定性（560B 一次训成的「工程学」）

**主线**：先在**小代理模型**上把超参调好，再**无痛迁移**到 560B，并用一套稳定性套件压住大规模训练的所有已知发散源。

| 手段 | 内容 | 定位 |
|------|------|------|
| **超参迁移（μP 式）** | 宽度缩放 $s=8$（代理宽度 768）；embedding 方差/LR 不变，hidden/unembedding 随 $s$ 反比缩放——小模型调好的超参直接迁到大模型 | Table 1, §3.1.1 |
| **模型生长初始化** | 先训**半规模 14 层**模型，**堆叠 $r=2$** → 初始化 28 层；保留训练态（样本计数、LR 调度、优化器态） | §3.1.2 |
| **Router 稳定** | 监控梯度范数比 $R_g$、目标 **$R_g<0.1$**；专家偏置走 PID（§2.1） | §3.2 |
| **激活控制** | **Hidden z-loss**（Eq.10）以极小系数抑制「巨活化」（massive activations） | §3.2 |
| **优化器** | **Adam $\epsilon=10^{-16}$**（而非默认 1e-5）——避免大规模下的阈值效应 | §3.2 |
| **确定性计算** | 确定性算子 → 可复现 + **静默数据损坏（SDC）检测** | §3 / §4 |

> 这套「小代理调参 → 生长初始化 → 稳定性套件」是 §四「20T tokens/30 天/98.48% 可用率、无人工干预」的底座。**确定性 + SDC 检测**在 2.0 演进成「确定性算子 + 二叉树累加 + bit-flip 检测」（见 [[longcat_2_analysis]] §6-7）。

---

## 四、预训练

| 项 | 内容 |
|----|------|
| 规模/速度 | **20 T tokens / 30 天 / 98.48% 时间可用率**，故障无需人工介入 |
| 通用阶段 | 序列长 **8,192**；SampleMix 实例级混合（质量/多样性打分） |
| 强化阶段 | STEM 与代码占比升到 **70%**，按困惑度监控渐进调配 |
| 长上下文 | **80B tokens（8k→32k）**，再 **20B tokens（32k→128k）**；自然长文 + 代码仓库 |
| 去污染 | Web/code 用 **13-gram 重叠**；合成数据用 **BGE-m3 语义相似 >0.9** 阈值剔除 |

---

## 五、Infra 与推理

- **SBO（Single Batch Overlap）流水线**：把**机内 TP（NVLink）** 与**跨机 EP（RDMA）** 通信重叠——配合 ScMoE，推理 **TPOT 较 DeepSeek-V3 降低约 50%**。SBO 的 decode 侧四阶段调度——把窗口内的 MLA **拆成 QKV 投影段 / 核心注意力+输出投影段两个 phase**，分别掩盖 all-to-all **dispatch / combine**，而 **MoE 专家 GEMM 裸露、靠 wide EP 压薄**；训练侧则改用 **token 维双 chunk 互掩**——阶段级细节与对照见 [[longcat_2_analysis]] §5.5。
- **实测**：H800 上 **>100 TPS**、**$0.70 / 百万 output tokens**；万卡级专家并行部署。
- **投机解码**：用 MTP 头，接受率 >90%。

> 与 2.0 的差异：Flash 全栈在 **NVIDIA H800**；2.0 把整套训练/部署迁到**国产 AI ASIC**（见 [[longcat_2_analysis]] §5）。

---

## 六、后训练与 Agentic 能力

**主线**：把省下的算力预算投向**智能体工具使用**——这是 Flash 的最强项。

- **Agentic 工具使用**：在 **τ²-Bench 67.7** 与自研 **VitaBench 24.30**（真实业务场景、**30+ 工具、60+ 交互轮**）上领先——面向多工具、长交互的真实 agent 任务。
- **安全**：在 Harmful / Criminal / Misinformation / Privacy 各类相对同侪表现突出。
- 具体 RL 算法/奖励设计报告着墨于能力构建（Agentic 数据合成 + 工具环境）；后续 **LongCat-Flash-Thinking**（arXiv 2509.18883）专攻推理、**Flash-Omni**（2511.00279）扩多模态——见 §八家族线。

---

## 七、评测结果

**Base 模型（Table 2，节选）**：

| Benchmark | DeepSeek-V3.1 | Llama-4 | Kimi-K2 | **LongCat-Flash** |
|-----------|:---:|:---:|:---:|:---:|
| MMLU | 87.46 | 84.41 | 87.47 | **87.05** |
| MMLU-Pro | 59.29 | 63.90 | 68.36 | **70.32** |
| GPQA | 47.16 | 48.08 | 45.89 | **51.09** |
| GSM8K | 92.22 | 84.61 | 92.27 | **92.19** |
| MBPP+ | 59.26 | 70.11 | 80.49 | **77.25** |

**Chat / 指令模型（Table 3，节选）**：

| Benchmark | DeepSeek-V3.1 | Qwen3-MoE | Kimi-K2 | Claude-4-Sonnet | **LongCat-Flash** |
|-----------|:---:|:---:|:---:|:---:|:---:|
| ArenaHard-V2 | 84.10 | 88.20 | 85.70 | 62.10 | **86.50** |
| IFEval | 86.69 | 88.54 | 88.91 | 88.35 | **89.65** |
| MATH500 | 96.08 | 98.80 | 97.60 | 93.80 | **96.40** |
| AIME25 | 49.27 | 68.33 | 50.66 | 37.00 | **61.25** |
| ZebraLogic | 85.30 | 94.22 | 89.11 | 75.85 | **89.30** |
| TerminalBench | — | — | — | — | **39.51**（开源第 2） |
| τ²-Bench（Agentic） | — | — | — | — | **67.7** |

**读表**：Flash 在**通用/推理**上与 DeepSeek-V3.1、Kimi-K2 同档（MMLU-Pro/GPQA/AIME25 反超部分对手），**Agentic 工具使用**是最亮的长板。以 **~27B 激活**做到这一梯队，是「动态算力分配」的最好证据。

---

## 八、家族线：从 Flash 到 2.0 的演进

### 8.1 LongCat 家族时间线

```
2025-09  LongCat-Flash (560B/27B)   ← 本页：ScMoE 短路 + 零计算专家首创（H800）
    ├── 2025-09  Flash-Thinking (2509.18883)  推理专精
    ├── 2025-1x  Flash-Omni (2511.00279)      多模态
    ├──          Flash-Lite                    N-gram Embedding 在此引入
    └── 2026-06  LongCat-2.0 (1.6T/48B)  ← 承袭 + LSA/N-gram + 国产 ASIC（见 [[longcat_2_analysis]]）
```

### 8.2 Flash vs 2.0 逐项对照

| 维度 | **LongCat-Flash**（2025-09） | **LongCat-2.0**（2026-06） |
|------|:---:|:---:|
| 总 / 激活 | 560B / ~27B（18.6–31.3B） | 1.6T / ~48B（动态） |
| 层数 / hidden | 28 / 6,144 | 38 / 8,192 |
| 注意力 | **MLA**（dense 全注意力） | **MLA + LSA 稀疏索引**（SI/CLI/HI） |
| 专家 | 512 routed + **256 零计算** · top-12 | 768 routed + **128 零计算** · top-12 |
| ScMoE 短路 / 零计算专家 | ✅ **首创** | ✅ 承袭 |
| N-gram Embedding | ✗ | ✅ 135B（承袭自 Flash-Lite） |
| 上下文 | 128 K | 1 M（RoPE-YaRN factor 120） |
| 词表 | 131,072 | 163,840 |
| 训练 tokens | 20 T | >35 T |
| **硬件** | **NVIDIA H800** | **国产 AI ASIC** |
| 推理 | SBO，>100 TPS，$0.7/M | PD 分离 + EP128 |

> **一句话**：2.0 保留了 Flash 的「省算力」骨架（ScMoE + 零计算专家 + MLA），把注意力从**全注意力**升级为**稀疏（LSA）** 以吃下 1M 上下文、加了 **N-gram** 稀疏扩参、并把整套栈从 **H800 迁到国产 ASIC**。

---

## Related Pages

- [[meituan_longcat/index]] — 美团 LongCat 家族总览（本页所属家族入口）
- [[longcat_2_analysis]] — **LongCat-2.0**（本模型的后继，架构承袭 + LSA/N-gram/国产 ASIC）
- [[12_deepseek_v3_analysis]] — MLA · MTP · FP8 MoE（ScMoE 对标的 DeepSeek-V3；TPOT 基准）
- [[20_deepseek_moe_analysis]] — MoE 路由与负载均衡（零计算专家/PID 均衡的对照）
- [[11_kimi_k2_analysis]] — 同期 1T MoE Agent 模型（评测对手）
- [[expert_parallel_analysis]] · [[megatron_ep_analysis]] — EP 全对全（ScMoE 掩盖的通信）
- [[muon_analysis]] — 优化器（Flash 用 Adam+μP；2.0 用 Muon）
- [[01_theory/01_models/index]] — 模型架构与家族总索引
