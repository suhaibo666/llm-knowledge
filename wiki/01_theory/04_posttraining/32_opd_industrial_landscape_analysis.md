# OPD 工业实践全景：厂商谱系、四类用途、教师经济学与两条实现路线之争

> **本页覆盖**：在线策略蒸馏（On-Policy Distillation, OPD）在 2024–2026 年工业界的落地全景——哪些厂商在用、用来做什么、教师从哪里来、教师值多少钱、两条互不相容的实现路线各有什么一手证据、以及哪些厂商没在用或不肯说。本页是《大语言模型在线策略蒸馏综述》（`OPD-Survey-2026-08.md`）§5 全章（§5.1 总览表至 §5.12 教师来源）的展开分析页，并吸收其调研底稿 `research-notes/opd_research_cn_industry.md`、`opd_research_west_industry.md`、`addendum_cn_reverify.md`、`addendum_west_reverify.md` 中正文省略的逐字引文与复核记录。方法论、散度演进、失败模式等不在本页范围，见 Related Pages。
> **调研基线**：2026-08-10（源稿基线）。
> **最后更新: 2026-08-11**

> **保真度说明（沿用源稿三级标注约定）**
> - 正文默认为**一手核实**内容（已打开 arXiv 原文指定章节/表号、官方技术报告 PDF 或官方博客逐字核对）；
> - 标 **⚠️** 者为**仅有二手来源**（第三方清单、媒体报道、他人转述）或未能打开原文的条目，引用前需自行回源；
> - 标 **【推断】** 者为分析推断，**非来源声称**。
> - 本页额外引入一个页面级标记 **【独立核验更正】**：源稿在该处与一手原文不符，本页已按回源结果重述。五处更正的完整清单见 §9。凡带此标记处，**以本页表述为准，不要沿用源稿旧表述**。
> - 所有断言均带定位符（arXiv 编号 + 版本 + §/Table/Eq 号，或官方 URL）。源稿已有定位符照搬；本页新增的定位符来自底稿逐字记录。

---

## 1. 一句话总览：一年之内的收敛，与仍未收敛的部分

2025-05 到 2026-08 的十五个月里，OPD 从「Qwen 一家的小模型压缩技巧」变成了头部开源旗舰的**标准后训练原语**：

- **收敛的部分**：中国五家头部开源厂商（阿里 Qwen、DeepSeek、智谱 GLM、小米、月之暗面 Kimi）**全部**公开采用 OPD；NVIDIA 两条产品线独立采用；用途从"压缩"扩展到"合并 / 防遗忘 / 跨模态对齐"四类；开源 RL 框架（veRL、slime、TRL、NeMo-RL、Tinker）全面基建化。
- **未收敛的部分**：**实现路线**（全词表 logit 反传 vs token 级 KL-as-reward）上，DeepSeek-V4 与 Kimi K3 给出**互相矛盾的一手经验报告**，无定论；**采用面**上，MiniMax 与字节 Seed 明确不用；**披露面**上，OpenAI 与 Anthropic 零披露，Google 的蒸馏表述在 Gemini 3 代**中断**。

一句话概括工业界的分工共识（源稿 P4）：**RL 创造能力，OPD 搬运能力**——搬运到小模型（压缩）、搬运进通才（合并）、搬运回自己（防遗忘）、搬运跨模态（对齐）。而"创造仍然贵、搬运变得便宜"正是本页 §6 教师经济学要给出的成本刻度。

---

## 2. 厂商谱系总表

### 2.1 中国厂商线（全部一手全文核实，除标 ⚠️ 者）

| 厂商 | 代表模型（时间） | 蒸馏类型 | 用途 | 实现路线 | 一手出处 |
|---|---|---|---|---|---|
| 阿里 Qwen | Qwen3（2025-05） | **OPD**（off-policy 打底 + on-policy 精修两段式） | 压缩：大→小（5 dense + 1 MoE） | A（logit 对齐） | arXiv:2505.09388v1 §4.5 / §4.7 / Table 21 |
| 阿里 Qwen | **Qwen3.5-Omni**（2026-04） | **OPD**（Stage 1 specialist 蒸馏 + Stage 2 跨模态 OPD） | **跨模态对齐**：text 条件能力→audio 条件 | 未披露损失细节 | arXiv:2604.15804v2，后训练两阶段 |
| 阿里 Qwen | Qwen3-Coder-Next（2026-02） | Expert Distillation（机制未言明，**全文 0 次 "on-policy"**，不得断言为 OPD） | 合并 | — | arXiv:2603.00729 §4.2.5 |
| DeepSeek | R1（2025-01） | 纯 off-policy SFT（800K 样本） | 压缩：对外赋能小模型 | — | arXiv:2501.12948 §2.4 / §4.1 / Table 6 |
| DeepSeek | V3 / V3.2（2024-12 / 2025-12） | off-policy 数据级（专家产数据） | 能力回灌 / 合并 | — | arXiv:2412.19437v2 §5.1 / §5.4.1；arXiv:2512.02556v1 §3 |
| DeepSeek | **V4**（2026） | **多教师全词表 OPD，整段替换 mixed RL** | 合并：专家→通才 | **A**（明确批评 B） | arXiv:2606.19348v1 §5.1 / §5.1.2 / §5.2.2 |
| 智谱 GLM | GLM-4.5（2025-08） | off-policy 自蒸馏（专家输出 SFT），全文 "on-policy" 0 次 | 合并 | — | arXiv:2508.06471v1 §3 / §3.1 / §3.3.2 |
| 智谱 GLM | **GLM-5**（2026-02） | **On-Policy Cross-Stage Distillation**（收官阶段） | **防遗忘**：前序 checkpoint 当教师 | **B**（KL-as-advantage，GRPO group size = 1） | arXiv:2602.15763v2 §3.5 / §3.6.1 |
| 小米 | **MiMo-V2-Flash**（2026-01） | **MOPD**（多教师 OPD + ORM 奖励混合） | 合并：专家→通才（消 see-saw） | **B**（Eq. 9 与结果奖励相加） | arXiv:2601.02780v2 §4.1 / §4.4 / Table 7 |
| 月之暗面 Kimi | k1.5 / K2 / K2 Thinking / K2.5 | **不用教师-学生蒸馏**（k1.5 的 long2short 为长→短迁移） | — | — | arXiv:2501.12599v4 §2.4；arXiv:2507.20534v2 全文 "teacher" 0 次；arXiv:2602.02276v2 全文 "distill" 0 次 |
| 月之暗面 Kimi | **K3**（2026-07） | **MOPD**（九专家多教师 OPD，后训练第三阶段） | 合并：专家→通才（域 × effort） | **B**（Eq. 15 clip 截断；实测 top-k 无明显优势） | arXiv:2607.24653v2 §4.1 / §4.1.2 / §4.1.3 |
| 腾讯混元 | **HY-MT1.5**（2025-12，翻译线） | **Strong-to-Weak OPD**（7B 教师→1.8B，per-token reverse KL） | 压缩（垂直域） | RKL（词表粒度未披露） | arXiv:2512.24092 §2.3；通用旗舰线 TurboS（arXiv:2505.15431）全文 "distill"/"on-policy" 均 0 次 |
| 腾讯混元 | HY-Embodied-0.5（2026-04）⚠️ | FKL on-policy 蒸馏（32B→MoT-2B 端侧） | 压缩 | — | arXiv:2604.07430 ⚠️（取自 AwesomeOPD 条目，未自查全文） |
| 百川 | **Baichuan-M3**（2026-02，医疗 235B） | TaskRL → 离线 FKL 模仿 → **MOPD**（reverse KL + 任务奖励并用） | 合并 + RL 混合 | B 系 | arXiv:2602.06570 摘要 / 三阶段框架 |
| 上海 AI Lab | **Agents-A1**（2026-06，35B MoE） | **多教师域路由 OPD** + Salient Vocabulary Alignment（教师 top-k 截断 RKL + 域归一化聚合） | 合并：六域→单模型 | top-k 截断（A/B 之间） | arXiv:2606.30616v2 摘要 / §2.3 / §4.3 |
| 快手 ⚠️ | Keye-VL-2.0（30B-A3B）；KAT-Coder-V2 / V2.5 | 跨模态 MOPD（13 个 RL 域专家教师池）；step 级 OPD；5 专家 MOPD + 漂移感知截断 | 合并 | — | arXiv:2606.10651（abs 已核实标题）；arXiv:2603.27703 / 2607.05471 ⚠️ |
| 理想 ⚠️ | Mach-Mind-4-Flash（2026-07，35B-A3B） | MOPD（"RL 与 OPD 单一加权损失切换"设计） | 合并 | — | arXiv:2607.09375 ⚠️ |
| MiniMax | M1 / M2 系 / M3（2026） | **无 OPD**：仅 off-policy 数据级（教师轨迹产 agent 数据）+ prompt distillation | 数据管线 | — | arXiv:2605.26494v2 全文 "on-policy" 0 次；arXiv:2506.13585 全文 "distill" 0 次；arXiv:2606.13392（纯架构论文） |
| 字节 Seed | Seed-Thinking-v1.5；Seed 2.0 / 2.1 | **旗舰无蒸馏**；但 Seed 合著 OPD 方法论文 | — | — | arXiv:2504.13914 全文 "distill" 0 次；arXiv:2607.05394（Direct-OPD）、arXiv:2607.13124（ShortOPD） |
| 百度 | ERNIE 5.0（2026-02） | **无 OPD**（distill 仅见于音频 tokenizer 表征蒸馏与压缩术语对比） | — | — | arXiv:2602.04705 全文检索 |
| 阶跃 Step | Step3-VL-10B（2026-01） | 数据级蒸馏（"distilled high-quality responses from internal frontier model"），self-distillation 仅列为 future work | — | — | arXiv:2601.09668 |

### 2.2 海外厂商线

| 厂商 | 代表模型 | 蒸馏类型 | 用途 | 实现路线 | 一手出处 |
|---|---|---|---|---|---|
| Thinking Machines Lab | Tinker（2025-10 起） | **OPD 配方产品化**（含多教师、多轮 agent、SDFT） | 全部四类 + 持续学习 | **B**（advantage $=-$ RKL） | TML 博客 https://thinkingmachines.ai/blog/on-policy-distillation/ （DOI 10.64434/tml.20251026）+ tinker-cookbook |
| Google / DeepMind | Gemini 1.5 Flash（2024-03） | **官方声明 "online distilled"（并列引用 GKD）** | 压缩 | ⚠️ 具体配方未公开 | arXiv:2403.05530v5 §3.2 / Table 45 |
| Google / DeepMind | Gemini 2.5 Flash 及以下 | 蒸馏（教师 next-token 分布用 **k-sparse 近似存储**；不再引 GKD） | 压缩 | off-policy 为主【推断】 | arXiv:2507.06261 p.3 |
| Google / DeepMind | Gemma 2 / 3 | 预训练蒸馏（2B/9B 以 KD **替代** next-token prediction；Gemma 3 全系蒸馏，教师未具名） | 压缩 | off-policy【推断】 | arXiv:2408.00118；arXiv:2503.19786 |
| Google / DeepMind | Gemini 3 系（5 份模型卡）、Gemma 4 | **披露中断**：distill/teacher/GKD 全部 0 命中 | — | — | 5 份 DeepMind 模型卡 PDF；arXiv:2607.02770 全文检索 |
| Meta | Llama 3.2 1B/3B | 单次结构化剪枝 + 以 3.1 8B/70B 的 **logits 做 token 级目标**的预训练蒸馏 | 压缩 | off-policy | 官方博客 2024-09-25 https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/ |
| Meta | Llama 4 Maverick | 预训练 **codistillation**（Behemoth 为教师，"dynamically weights the soft and hard targets"） | 压缩 | 【推断】在线计算但数据是固定预训练流，**非 OPD** | 官方博客 2025-04-05 https://ai.meta.com/blog/llama-4-multimodal-intelligence/ |
| NVIDIA | **Nemotron 3 Ultra**（2026-06，550B-A55B） | **MOPD = Multi-teacher OPD**（十余领域教师，两轮迭代） | 合并 | **B**（负 reverse-KL 目标 Eq. 1，异步 behavior/proximal 解耦） | arXiv:2606.15007 §3 / §3.3 / §3.3.1–3.3.4 |
| NVIDIA | **Nemotron-Cascade 2**（2026-03，30B-A3B） | **MOPD = Multi-domain OPD**（教师取自内部 Cascade RL checkpoint）【独立核验更正】 | 合并 + 防回退 | **B**（仅学生采样 token，截断 IS $\varepsilon\in[0.5,2.0]$） | arXiv:2603.19220 §4 / §4.4 / Table 3 |
| Mistral | Magistral Medium（2025-06） | **明确声明纯 RL、不从既有推理模型蒸馏** | — | — | arXiv:2506.10910v1 摘要 / 贡献列表 / §5.2 / §6.2 |
| Mistral | Magistral Small；Ministral 3（2026-01） | Small 用 Medium 轨迹 SFT+RL；Ministral 3 = **Cascade Distillation**（迭代剪枝 + 蒸馏，parent = Mistral Small 3.1） | 压缩 | off-policy | arXiv:2506.10910v1 Figure 4；arXiv:2601.08584 摘要 / §1 |
| OpenAI | Model Distillation 产品（2024-10） | **纯 hard-label off-policy**（`store=True` → metadata 过滤 → SFT，无 logprobs/KL/软标签） | 产品化 API | — | OpenAI 社区官方帖 964021；Cookbook distillation notebook；现并入 SFT 指南 |
| OpenAI | GPT-4o mini / o-mini / GPT-5-mini·nano / GPT-5.4 mini·nano | **无公开技术细节**（"是蒸馏产物"仅为第三方推测） | — | — | GPT-4o mini 发布文全篇无 "distillation"；GPT-5 系统卡（60 页）、GPT-5.5 系统卡（45 页）distill/teacher 均 0 命中 |
| Anthropic | Haiku 系（3 / 3.5 / 4.5） | **无公开技术细节** | — | — | Claude 3 模型卡 42 页 + 3.5 Haiku 附录卡 14 页 + Haiku 4.5 系统卡 39 页，**合计 95 页 distill/teacher 全部 0 命中** |

### 2.3 采用面全景（按核实级别分层，截至 2026-08-10）

- **一手全文核实的 OPD 采用者（13 例）**：Qwen3（压缩）、Qwen3.5-Omni（跨模态）、DeepSeek-V4（合并）、GLM-5（防遗忘）、Kimi K3（合并）、MiMo-V2-Flash（合并 + ORM）、腾讯 HY-MT1.5（垂直压缩）、Baichuan-M3（医疗 MOPD）、Agents-A1（六域合并）、Gemini 1.5 Flash（官方 "online distilled"）、NVIDIA Nemotron 3 Ultra、NVIDIA Nemotron-Cascade 2、TML Tinker（产品化）。
- **⚠️ 清单级（未自查全文，取自第三方全读清单 thinkwee/AwesomeOPD）**：快手 Keye-VL-2.0 / KAT-Coder-V2·V2.5、理想 Mach-Mind-4-Flash、腾讯 HY-Embodied-0.5。
- **明确未采用**：MiniMax（M1/M2 系/M3 三重验证）、字节 Seed 旗舰、Mistral Magistral Medium（仅限旗舰推理线）。
- **无公开细节**：OpenAI 全系小模型、Anthropic Haiku 全系、Google Gemini 3 代及 Gemma 4。

---

## 3. 四类工业用途

源稿 §5.11 归纳出三类主用途，另加一类 2026 年新出现的形态。四类的关键区分不在算法（都是"学生采样 + 教师逐 token 监督"），而在**教师是谁、师生规模关系如何、以及它在流水线中替代了什么**。

| 用途 | 师生规模关系 | 教师身份 | 它替代了什么 | 代表案例 |
|---|---|---|---|---|
| ① **压缩（大→小）** | 教师 ≫ 学生 | 更大的旗舰 | 小模型的完整后训练管线 | Qwen3、Gemini 1.5 Flash、HY-MT1.5、（off-policy 版）R1→Qwen/Llama、Llama 3.2、Gemma 2/3 |
| ② **合并（专家→通才）** | 教师 ≈ 学生（**同规模**） | 同基座领域专家 | **混合 RL**（reward hacking + 能力跷跷板） | DeepSeek-V4、Kimi K3、MiMo-V2-Flash、Nemotron 3 Ultra、Agents-A1、Baichuan-M3 |
| ③ **防遗忘 / 持续学习（自己→自己）** | 教师 = 学生的另一个版本 | 前序阶段 checkpoint / 早期版本 | 多阶段 RL 的能力回归修复 | GLM-5、Nemotron-Cascade 2、TML 个性化实验、SDFT、⚠️ Cursor Composer 2.5（HF 博客归类） |
| ④ **跨模态对齐（同一模型的跨条件）** | 教师 = 学生（**同一模型不同输入条件**） | 文本条件下的自己 | 模态间的能力落差 | Qwen3.5-Omni |

### 3.1 用途 ①：压缩——最初动机，仍是最广的存量

Qwen3 的 Strong-to-Weak Distillation 是这一类的标准配方（arXiv:2505.09388v1 §4.5）：五个 dense（0.6B / 1.7B / 4B / 8B / 14B）+ 一个 MoE（30B-A3B）**不走旗舰的四阶段后训练**，改走两阶段蒸馏。报告 §4 引言的理由是纯经济学的：

> "This approach eliminates the necessity of performing an exhaustive four-stage training process individually for every small-scale model. It leads to better immediate performance, as indicated by higher Pass@1 scores, and also improves the model's ability of exploration, as reflected in improved Pass@64 results. In addition, it achieves these gains with much greater training efficiency, requiring only 1/10 of the GPU hours compared to the four-stage training method."（arXiv:2505.09388v1 §4 引言，PDF p.10）

垂直域的对应案例是腾讯 HY-MT1.5（arXiv:2512.24092 §2.3）：教师 = 训练完成的 HY-MT1.5-7B，学生 = 1.8B，约 100 万条单语样本覆盖 33 种语言，损失是 **per-token reverse KL**——"Loss Function. We employ per-token reverse KL divergence to align the student's output dis[tribution]"。这是"通用旗舰线无 OPD、垂直线有 OPD"的典型分裂（混元 TurboS 报告 arXiv:2505.15431 全文 "distill" 与 "on-policy" 均 0 次）。

### 3.2 用途 ②：合并——2026 年增长最快，且是范式转折点

这一类是 OPD 从"技巧"升格为"主干阶段"的根本原因。其触发条件是旗舰后训练收敛到了一个共同形态：**先按领域分头做大规模 RL 训出专家，再需要把多个专家合并进单一模型**。而合并的既有方案（混合 RL）有两个已被多家一手报告点名的缺陷：

- **能力跷跷板（see-saw effect）**：小米 MiMo-V2-Flash 把它写进了 MOPD 的动机（arXiv:2601.02780v2 §4.1，动机 = capability imbalance + learning inefficiency）；
- **每域信号被稀释**：Nemotron 3 Ultra 的表述最直接——混合环境 RLVR 中 "each domain contributes only a relatively small number of samples to any given training batch, diluting the per-domain learning signal"（arXiv:2606.15007 §3.3）；
- **RLVR 与 RLHF 互相拆台**：Nemotron-Cascade 2 的表述——"certain RLVR training often reduces model entropy and shortens reasoning traces, thus can negatively impact mathematical reasoning performance, while RLHF-oriented optimization can partially trade off against instruction-following behavior"（arXiv:2603.19220 §4.4）。

于是 OPD 被扶正为**能力整合算子**。最强的单条证据是 DeepSeek-V4 §5.1 的逐字声明（arXiv:2606.19348v1）：

> "Although the training pipeline largely mirrored that of DeepSeek-V3.2, a critical methodological substitution was made: **the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD)**."

注意这一类的定义性特征：**教师与学生同规模**。HuggingFace 官方 2026-07 的年度盘点（Paniego, 2026-07-08，https://huggingface.co/blog/sergiopaniego/distillation-2026 ）把"教师是同等规模的领域专家而非更大模型"列为年度范式——这与经典 KD 的"大→小"直觉正好相反，也是 §6 教师经济学的出发点。

### 3.3 用途 ③：防遗忘——OPD 作为"阶段胶水"

GLM-5 定义了这一类（arXiv:2602.15763v2 §3.5 逐字）：

> "In our multi-stage RL pipeline, sequentially optimizing for distinct objectives can lead to the cumulative degradation of previously acquired capabilities. To mitigate this issue, we perform on-policy cross-stage distillation as the final stage, adopting an on-policy distillation algorithm [14; 52; 51; 28] to swiftly recover the skills acquired in earlier SFT and RL stages (Reasoning RL and General RL). Specifically, the final checkpoints from the preceding training stages serve as teacher models, where the training prompts are sampled from the corresponding teachers' RL training sets and mixed in appropriate proportions."

流水线位置是 SFT（interleaved thinking）→ Reasoning RL（§3.2）→ Agentic RL（§3.3）→ General RL（§3.4）→ **§3.5 On-Policy Cross-Stage Distillation（最终阶段）**。教师**全部是自己的前序 checkpoint**，因此额外训练成本近零。

TML 博客给出了同一用途的另一个形态——**持续学习闭环**（https://thinkingmachines.ai/blog/on-policy-distillation/ "Distillation for personalization" 节，四组数字已逐字目验）：Qwen3-8B 基线 Internal QA 18% / IF-eval 85% → midtrain（100% 内部文档）43% / 45%（知识↑，指令遵循崩） → midtrain（70%）36% / 79% → **+ OPD（教师 = 早期版本的自己）41% / 83%**。原文结论："on-policy distillation recovers nearly full performance on IF-eval without losing any knowledge."

Nemotron-Cascade 2 则把"合并"与"防遗忘"合成一件事：其 MOPD 教师直接取自 Cascade RL 流程内各阶段的最强验证 checkpoint，"which makes it easy to assemble a capability-diverse teacher pool **without introducing external models**"（arXiv:2603.19220 §4.4），摘要中的定位是 "efficiently recover benchmark regressions and sustain strong performance gains along the way"。

### 3.4 用途 ④：跨模态对齐——三类之外的新形态

Qwen3.5-Omni（arXiv:2604.15804v2）的后训练分两阶段：Stage 1 是 specialist 蒸馏（"All teacher models are fine-tuned from the pre-trained Qwen-3.5 base checkpoint. … These teacher models are used to generate domain-specific data, enabling the specialized capabilities learned in each domain to be distilled into a single unified model."）；**Stage 2 是跨模态 OPD**：

> "we introduce a second-stage training procedure based on on-policy distillation (OPD), with the goal of distilling the model's stronger response capabilities under text inputs into the audio-input setting. … We then use this response as the distillation target for the corresponding audio-conditioned query. By training on such on-policy targets, the model gradually aligns its audio-…"

评测部分归因："We believe that OPD and interaction-aligned RL have a positive effect on improving the instruction-following capabilities of an omni-model LLM."（Table 4 讨论）

其意义在于把 OPD 的"师生"关系从**跨模型**推广到**同一模型的跨输入条件之间**：教师与学生是同一组权重，差别只在输入模态。这与 §6 的教师来源类④（推断时增强的自己）、以及自蒸馏中的"特权信息"（privileged information）分支同构——**教师的优势不来自参数，而来自它拿到了更好的输入**。

### 3.5 一个跨用途的观察：损失细节的披露率随用途下降

用途 ① 的报告普遍披露损失形式（Qwen3 的 logit 对齐、HY-MT1.5 的 per-token reverse KL），用途 ② 的报告普遍给出公式（V4 Eq. 29、K3 Eq. 15、MiMo Eq. 9、Nemotron Eq. 1-2），而用途 ④ 的 Qwen3.5-Omni **未披露损失细节**（只说 "based on on-policy distillation"）。这提示新用途的公开度滞后于成熟用途约一个身位。【推断】

---

## 4. 逐厂商深读

### 4.1 阿里 Qwen：工业界首个公开的 OPD 生产配方

**两阶段配方**（arXiv:2505.09388v1 §4.5 逐字）：

> "(1) **Off-policy Distillation**: At this initial phase, we combine the outputs of teacher models generated with both /think and /no think modes for response distillation. This helps lightweight student models develop basic reasoning skills and the ability to switch between different modes of thinking, laying a solid foundation for the next on-policy training phase.
> (2) **On-policy Distillation**: In this phase, the student model generates on-policy sequences for fine-tuning. Specifically, prompts are sampled, and the student model produces responses in either /think or /no think mode. The student model is then fine-tuned by aligning its logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence."

**Table 21 消融**（§4.7，Qwen3-8B，三行同起自一个 off-policy 蒸馏 checkpoint；括号内为 pass@64）：

| 方法 | AIME'24 | AIME'25 | MATH500 | LCB v5 | MMLU-Redux | GPQA-Diamond | GPU 时数 |
|---|---|---|---|---|---|---|---|
| Off-policy 蒸馏（起点） | 55.0 (90.0) | 42.8 (83.3) | 92.4 | 42.0 | 86.4 | 55.6 | **–** |
| + Reinforcement Learning | 67.6 (90.0) | 55.5 (83.3) | 94.8 | 52.9 | 86.9 | 61.3 | **17,920** |
| + On-policy 蒸馏 | **74.4 (93.3)** | **65.5 (86.7)** | **97.0** | **60.3** | **88.3** | **63.3** | **1,800** |

报告原文对该表的结论（§4.7 逐字）："The results, summarized in Table 21, show that distillation achieves significantly better performance than reinforcement learning while requiring approximately only 1/10 of the GPU hours. Furthermore, distillation from teacher logits enables the student model to expand its exploration space and enhance its reasoning potential, as evidenced by the improved pass@64 scores on the AIME'24 and AIME'25 benchmarks after distillation, compared to the initial checkpoint. In contrast, reinforcement learning does not lead to any improvement in pass@64 scores."

**已独立复核为准确**：主稿转录的四基准 × 三行共 12 格数字与原表逐格一致；"aligning its logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence" 逐字无误；"pass@64 只在蒸馏下提升、RL 不提升"有明确原文支撑（上引段落）。MMLU-Redux 与 GPQA-Diamond 两列取自底稿的 PDF 转录。

**但引用 1,800 vs 17,920 时必须补两条限定（本页新增）**：

1. **off-policy 起点行的 GPU 时是 "–"（未报告）**。因此 1,800 vs 17,920 严格来说只是**两个增量阶段**之比，而非端到端总成本之比——两条支线共享的 off-policy 打底成本未计入任一侧。若打底成本可观，真实的端到端倍数会小于 10×。
2. **报告全文未说明这两个数字是否计入教师推理成本**。这与 TML 博客刻意区分三档成本口径（9× / 18× / 30×，见 §4.7）形成对照：Qwen3 只给一个数字，口径不明。

这两条与"1/10 GPU 时"是全领域被引用最多的数字这一事实叠加，构成引用风险：该数字在 TML 博客、DeepSeek-V4、GLM-5、MiMo、K3 报告中被反复转引，但**口径从未被追问过**。【推断】

**Qwen 2026 年动向**：Qwen3.5 主线 LLM（397B-A17B 及后续尺寸）截至 2026-08-10 仍无技术报告（HF 卡 `Qwen/Qwen3.5-397B-A17B` 全文 0 次 distill），小尺寸蒸馏策略无公开细节；Qwen3-Next 模型卡只提 GSPO RL、无蒸馏声明；Qwen3-Coder-Next（arXiv:2603.00729 §4.2.5）有 Expert Distillation 但**全文 0 次 "on-policy"**，机制未言明，**不得断言为 OPD**。真正的 2026 年证据是 Qwen3.5-Omni（§3.4）。

### 4.2 DeepSeek：从纯 off-policy 到"OPD 替换 RL"的四代演进

DeepSeek 是观察范式迁移的最佳单一样本，四代报告构成一条完整演化线：

1. **R1**（arXiv:2501.12948 §2.4）：R1 生成 800K 样本对 Qwen/Llama 小模型**纯 SFT**——"We apply only SFT and do not include an RL stage, even though incorporating RL could substantially boost model performance."§4.1 / Table 6 给出著名对比：R1-Zero-Qwen-32B（32B 直接大规模 RL）AIME'24 47.0 vs R1-Distill-Qwen-32B **72.6**；结论逐字："distilling more powerful models into smaller ones yields excellent results, whereas smaller models relying on the large-scale RL … require enormous computational power and may not even achieve the performance of distillation."
2. **V3**（arXiv:2412.19437v2 §5.1 / §5.4.1）：内部 R1 → 领域专家（SFT+RL）→ 拒绝采样产 SFT 数据回灌 V3；代价是响应长度显著膨胀（Table 9：MATH-500 74.6→83.2，但平均长度 769→1510；原文 "the distillation leads to better performance but also substantially increases the average response length"）。
3. **V3.2**（arXiv:2512.02556v1 §3）：Specialist Distillation 定型——六个专业领域（数学 / 编程 / 通用逻辑推理 / 通用 agent / agentic coding / agentic search）+ 写作与通用问答，**全部从同一 pre-trained base checkpoint 微调**，"Each specialist is trained with large-scale Reinforcement Learning (RL) computing"，专家产数据给最终 checkpoint，再统一 Mixed RL（GRPO）。仍是数据级，全文 0 次 "on-policy"。
4. **V4**（arXiv:2606.19348v1）：mixed RL 被 OPD 整段替换（§5.1 引文见 §3.2）。

**V4 §5.1.2 的核心机制（已逐字复核，可放心作一手结论）**：

> "we employ multi-teacher On-Policy Distillation (OPD …) as the primary technique for merging expert capabilities into the final model. … This is achieved by having the student learn from the output distributions of teacher models on its own generated trajectories."

目标函数（Eq. 29），reverse KL、按教师加权求和：

$$
L_{\mathrm{OPD}}(\theta)=\sum_i w_i\, D_{KL}\big(\pi_\theta \,\big\|\, \pi_{E_i}\big)
$$

原文强调 "requires sampling training trajectories from the student $\pi_\theta$ to maintain on-policy learning"，且 **"more than ten teacher models covering various domains are employed to distill a single student model"**。

**对路线 B 的批评（逐字，路线之争的核心文本）**：

> "prior works usually simplify the full-vocabulary KL loss into a token-level KL estimate at each token position, and reuse RL framework by replacing $\mathrm{sg}[\log \pi_E/\pi_\theta]$ as the per-token advantage estimate … **it leads to high variance in gradient estimation and often causes training instability. Therefore, we adopt full-vocabulary logit distillation in our OPD.**"

**配套基建**（§5.2.1–5.2.3）：§5.2.2 "Efficient Teacher Scheduling for Full-Vocabulary OPD" 支持 "effectively unbounded number of teachers, each potentially comprising trillions of parameters" 的教师权重 offload 调度；§5.2.1 FP4（MXFP4）QAT 下的师生数值一致性；§5.2.3 可抢占、容错的 rollout 服务同时服务 RL 与 OPD。⚠️ 另有二手分析（Maxime Labonne, "DeepSeek V4: ten teachers, one student"）称其缓存教师**隐藏状态**而非完整 logits、按教师索引排序样本、用自定义 TileLang 内核算精确 KL——与报告 §5.2.2 的教师调度描述相容，但未见于报告原文。

**引用系谱**：OPD 引用 = Thinking Machines 博客（`lu2025onpolicydistillation`）+ MiniLLM。

**日期疑点已解释**：abs 页 Submission history 为 "[v1] Sun, 26 Apr 2026 14:49:33 UTC"，2026-04-26 确为星期日，与 dateline 自洽；报告 2026-04 已以 PDF 形式发布于 HuggingFace。【推断】经 moderation hold 后按公告月分配编号，属 arXiv 已知现象。引用写 "arXiv:2606.19348v1" 即可。

### 4.3 智谱 GLM：从数据级自蒸馏到跨阶段 OPD

GLM-4.5（arXiv:2508.06471v1 §3）还是 off-policy：Stage 1 训三域专家（Reasoning / Agent / General chat），Stage 2 "employ self-distillation techniques to integrate multiple experts"；§3.3.2 的 Iterative Distillation 是 RL → 自蒸馏（自产响应替换冷启动数据）→ 再 RL 的循环。全文 "on-policy" 0 次。

GLM-5（arXiv:2602.15763v2 §3.5）转向 OPD，损失实现是把 GRPO（Eq. 1）的 advantage 替换为 token 级 KL 估计（Eq. 2）：

$$
\begin{aligned}
\hat{A}_{i,t}
&=\mathrm{sg}\!\left[\log \frac{\pi_{\text{teacher}}^{\text{infer}}(y_t\mid \cdot)}{\pi_\theta^{\text{train}}(y_t\mid \cdot)}\right]
\end{aligned}
$$

一个常被忽略的工程含义：既然 advantage 不再需要组内比较来估计，**GRPO 的 group size 可以降到 1**（batch = 1024）。原文："it is no longer necessary to maintain a large group of samples per prompt to estimate advantages; the advantage is computed directly from the gap with the teacher models instead."——这是路线 B 的一项隐性红利：**省掉了 GRPO 每 prompt 多次采样的 rollout 开销**。教师 logits 目前经推理引擎获取。

**基建证据**：§3.6.1 逐字确认智谱开源 RL 框架 slime 在统一栈内支持 OPD——"GLM-5 leverages this capability to support a broad range of domains and training paradigms, including but not limited to reasoning RL, general RL, agentic RL, and **on-policy distillation**, all within a unified training stack."（注意：slime 的 README 本身未把蒸馏列为核心特性，**slime 支持 OPD 的权威出处应引 GLM-5 §3.6.1 而非 README**。）

GLM-4.6 无独立技术报告（HF 卡仅链接 GLM-4.5 报告 arXiv:2508.06471），未核实。

### 4.4 小米 MiMo-V2-Flash：MOPD 与"蒸馏-RL 奖励统一"

MiMo-V2-Flash（arXiv:2601.02780v2，309B-A15B）的三阶段：SFT → 各域独立 RL 训出教师（agentic search / coding / general tool use + math / general reasoning / safety）→ MOPD。摘要逐字：

> "To efficiently scale post-training compute, MiMo-V2-Flash introduces a novel **Multi-Teacher On-Policy Distillation (MOPD)** paradigm. In this framework, domain-specialized teachers (e.g., trained via large-scale reinforcement learning) provide dense and token-level reward, enabling the student model to perfectly master teacher expertise."

§4.1 的机制表述："Rather than merging model parameters or generating static offline datasets from experts, we formulate multi-teacher knowledge integration as an on-policy reinforcement learning process. The student model samples from its own evolving distribution and receives token-level supervision from domain-specific teachers through KL divergence rewards."

**独特贡献（Eq. 9）**——唯一公开把 OPD 信号与结果奖励显式**相加**的工业报告：

$$
\begin{aligned}
\hat{A}_{\mathrm{MOPD}}
&=\mathrm{sg}\!\left[\log\frac{\pi_{\text{domain}}}{\pi_\theta}\right]+\alpha\cdot\hat{A}_{\mathrm{ORM}}
\end{aligned}
$$

教师可以是 RL 模型、SFT 模型、甚至学生自身（Table 7 中标注 **Self**）。

#### 4.4.1 【独立核验更正】Table 7 的 Δ 列是"学生 − 最强教师"，不是"前后变化"

源稿 §5.5 写"也如实报告了 BrowseComp −6.3、创意写作 −3.9 的失分域"，这**混淆了两个不同的量**。回一手原文核实后的正确读法：**Table 7 的 Δ 列 = 学生（MOPD 后）− 该域最强教师**，与"学生自身的前后变化"是两回事。

| 基准 | 学生 MOPD 前 → 后 | 学生自身变化 | 该域最强教师 | Δ（学生 − 教师） | 正确定性 |
|---|---|---|---|---|---|
| AIME 2025 | 89.3 → 94.1 | **+4.8** | 93.9 | +0.2 | 超越教师 |
| HMMT Feb 2025 | 76.9 → 84.4 | **+7.5** | 82.6 | +1.8 | 超越教师 |
| LiveCodeBench | 77.5 → 83.2 | **+5.7** | 82.6 | +0.6 | 超越教师 |
| Tau2-Bench | 75.9 → 80.3 | **+4.4** | 79.6 | +0.7 | 超越教师 |
| SWE-Bench Verified | 67.8 → 73.4 | **+5.6** | 74.2 | **−0.8** | **学生提升，但仍略低于教师**（差距，非回退） |
| BrowseComp | 42.5 → **45.4** | **+2.9** | 51.7（SFT 教师） | **−6.3** | **学生提升 +2.9，−6.3 是与教师的差距，不是失分** |
| Arena-Hard（Creative Writing） | 90.1 → 86.2 | **−3.9** | Self（即 90.1） | −3.9 | **真回退**（教师即 Self，故两个口径重合） |
| GPQA-Diamond | 84.9 → 84.3 | **−0.6** | — | — | **真回退** |

**结论重述**：MiMo-V2-Flash 的真正性能回退是 **Arena-Hard 创意写作 90.1→86.2（教师即 Self）** 与 **GPQA-Diamond 84.9→84.3**；BrowseComp **不是失分域**（学生 42.5→45.4，上升 2.9），它只是**未追平 SFT 教师 51.7** 的那一格；SWE-Bench 同理（+5.6 但 Δ=−0.8）。

这一更正有两层意义：(a) 它把"MOPD 在多数域追平或超过各自最强教师"的证据强度**提高**了（失分域实际只有两个，而非四个）；(b) 它把"创意写作在教师即自己的情况下仍然退化"这一更有意思的现象凸显出来——**当教师就是学生的旧版本时，OPD 仍可能造成回退**，这与源稿 §7.3 的多样性坍缩失败模式（RKL mode-seeking → Pass@k 退化，arXiv:2603.07079）以及"OPSD 是压缩而非纠错"的定位（Kim & Lee，经 arXiv:2604.00626 §7.1 转述）互相印证：创意写作恰是最不适合 mode-seeking 压缩的任务几何。【推断】

#### 4.4.2 【独立核验更正】Eq. 7–8 的截断是**训推比**，不是师生比，也未称 "IcePop"

源稿 §9 开放问题 4 写"MiMo 已用 IcePop 式截断（2601.02780 Eq. 7-8）"，两处需更正：

1. **正文未出现 "IcePop" 一词**——该词仅见于参考文献标题。原文的表述是 **"Following Zhao et al. (2025)"**。
2. **截断作用于训推比 $\pi_\theta/\mu_\theta$，不是师生比**。即训练策略 $\pi_\theta$ 与采样（推理引擎）策略 $\mu_\theta$ 之比

   $$
\begin{aligned}
w_t
&=\frac{\pi_\theta(y_t\mid x, y_{<t})}{\mu_\theta(y_t\mid x, y_{<t})},\qquad w_t \leftarrow 0 \ \text{ if } w_t\notin[\varepsilon_{\mathrm{low}},\varepsilon_{\mathrm{high}}]
\end{aligned}
   $$

   落在区间外即置零。这是**训推不一致（train–inference mismatch）修正**，源于训练框架与推理引擎的数值/实现差异，**与 OPD 的师生 KL 是完全不同的两件事**。

**为什么必须区分**：路线 B 的报告里同时存在两类"截断/clip"，极易混淆——

| 报告 | 截断对象 | 截断的是什么比值 | 目的 |
|---|---|---|---|
| MiMo-V2-Flash（Eq. 7–8） | $\pi_\theta/\mu_\theta$ | **训练策略 / 采样策略** | 修正训推不一致 |
| Kimi K3（Eq. 15） | $\log(\pi_{\text{teacher}}/\pi_\theta)$ | **教师 / 学生** | 抑制极端 advantage |
| Nemotron-Cascade 2（§4.4） | 重要性权重，$\varepsilon\in[0.5,2.0]$ | **训练 / 推理**（train–inference 失配） | 同 MiMo |
| Agents-A1（§2.3） | 教师分布 top-k 支持集 | **词表维度截断** | 限制 RKL 的支持集 |

即：**MiMo 与 Nemotron-Cascade 2 截的是训推比，K3 截的是师生比，Agents-A1 截的是词表**。三种截断解决三个不同问题，不可互相替代，也不应统称为"OPD 的截断技巧"。【推断，由上述四份一手报告归纳】

### 4.5 月之暗面 Kimi：从最大反例到九专家 MOPD

**K2.5 之前，Kimi 是头部厂商中唯一公开路线完全不用教师-学生蒸馏的**：k1.5 的 long2short（arXiv:2501.12599v4 §2.4）是长→短 CoT 迁移，四种方法为模型合并 / 最短拒绝采样 / DPO / long2short RL，论文虽在 §3.2 自称 "long-to-short distillation" 但**无学生采样 + 教师逐 token 监督**；K2 全文 "teacher" 0 次（arXiv:2507.20534v2，唯一一次 "distill" 是 §3.2.2 描述 critic 信号迁移的修辞用法）；K2 Thinking 模型卡无蒸馏表述（仅 INT4 QAT）；K2.5 全文 "distill" 0 次（arXiv:2602.02276v2）。

**转折在 K3**（arXiv:2607.24653，v1 2026-07-27 / **v2 2026-08-07**，2.8T MoE，47 页）。§4.1 三阶段（PDF p.13 逐字）：

> "initializing baseline agent capabilities via supervised fine-tuning (SFT), developing specialized domain experts at varying reasoning effort via Reinforcement Learning (RL), and consolidating these domain-specific policies into a single model using **Multi-Teacher On-Policy Distillation (MOPD)**."

**九专家 = 三域 × 三档 reasoning effort**（已逐字复核，可放心作一手结论）：三大域为 (i) general tasks（含 vision / reasoning / search / knowledge work）、(ii) general agents（长程助手 / deep research / 写作）、(iii) coding；effort 档位 $e\in\{\text{low},\text{high},\text{max}\}$。原文："Crossing these three domain experts with three reasoning effort levels in {low,high,max} yields a total of nine expert models."（PDF p.12）；教师矩阵的 effort 维度由长度预算 $\tau$ 退火得到（§4.1.2："anneal $\tau$ to smaller values to obtain the high- and low-effort expert models"）。

**逐 token OPD 奖励（Eq. 15，已逐字复核）**：

$$
\begin{aligned}
r^{d}_{\mathrm{opd}}(y_t \mid e,x,y_{<t})
&=\mathrm{clip}\!\left(\mathrm{sg}\!\left[\log\frac{\pi^{(d,e)}_{\mathrm{teacher}}(y_t \mid x,y_{<t})}{\pi_\theta(y_t \mid e,x,y_{<t})}\right],\,-R_{\max},\,R_{\max}\right)
\end{aligned}
$$

即路线 B + clip 截断。选择路线 B 的理由是**基建复用**：报告强调这一稠密奖励 "seamlessly integrates into our RL framework, naturally enabling infrastructure-level optimizations such as **partial rollout training for long-horizon tasks**"。

**对路线之争的直接反证（逐字，已复核）**：

> "While we also experimented with more fine-grained top-k distillation objectives, we observed no clear advantage in either convergence speed or final performance in our setting."

**引用系谱**：标准三件套——[76] TML 博客、[136] MiMo-V2-Flash（2601.02780）、[29] DeepSeek-V4。§4.1.4 采用 MXFP4 QAT（承 K2 Thinking 的 INT4 QAT 传统）。

⚠️ **必须分层的舆情**：媒体对 K3"蒸馏美国模型"的质疑与官方否认是**舆情事实而非训练事实**。K3 报告描述的 MOPD 教师是**自家训练的九个域专家**，与该指控是两回事，报告本身既不证实也不证伪该指控。引用时须明确标注为 allegation/denial。

至此，**Qwen / DeepSeek / GLM / 小米 / Kimi 五家头部开源厂商全部公开采用 OPD**。

### 4.6 NVIDIA：MOPD 跨出中国阵营的一手证据（且此处有两处更正）

#### 4.6.1 Nemotron 3 Ultra（arXiv:2606.15007，2026-06-09，550B 总参 / 55B 激活，MoE Hybrid Mamba-Attention，65 页）

摘要逐字："We pre-trained Nemotron 3 Ultra on 20 trillion text tokens, then extended the context length to 1M tokens, and post-trained using Supervised Fine Tuning (SFT), Reinforcement Learning (RL), and **Multi-teacher On-Policy Distillation (MOPD)**."

章节定位：§3 Post-Training（p.15）→ §3.2 RL（p.20）→ **§3.3 MOPD（p.20）** → §3.3.1 Algorithm → §3.3.2 Specialized Teachers（p.22）→ §3.3.3 MOPD Warmup（p.27）→ §3.3.4 Results and Discussions（p.27）→ §3.3.5 Limitations and Open Problems（p.28）→ §3.4 MTP Boosting。全文 `MOPD` 命中 70 次。

**算法（§3.3.1, Eq. 1–2）**——完全 on-policy 情形即最大化负 reverse-KL 目标：

$$
\begin{aligned}
J_{\mathrm{MOPD}}(\theta)
&=\sum_i \lambda_i\, \mathbb{E}_{q\sim D_i,\; y\sim\pi_\theta}\left[\sum_t \log \pi_{T_i}(y_t\mid s_t)-\log \pi_\theta(y_t\mid s_t)\right]
\end{aligned}
$$

原文："Equivalently, at each prefix $s_t$, the student minimizes $D_{KL}(\pi_\theta(\cdot\mid s_t)\|\pi_{T_i}(\cdot\mid s_t))$ … MOPD provides a **dense token-level learning signal** from the relevant teacher distribution."异步稳定化采用**行为策略与 proximal 策略解耦**，蒸馏优势为 $\hat{A}_t=\mathrm{sg}[\ell^{T_i}_t-\ell^{\mathrm{prox}}_t]$；全程三类 worker（rollout / teacher-scoring / learner）异步流水线。

**动机**：混合环境 RLVR 中 "each domain contributes only a relatively small number of samples to any given training batch, diluting the per-domain learning signal"，故训练 **"more than ten specialized teacher models"**。

**教师清单（Figure 10）**：第一轮 = STEM / Chat / Instruction-following / Terminal-use / Conversational tool-use / SWE / Search / Office work / Usability / Agentic Safety Teacher；第二轮新增或刷新 = Coding / Chat 2 / Conversational tool-use 2 / SWE 2 / Office work 2。

**TML 影响力硬证据**：Nemotron 3 Ultra 以 DOI 正式引用 TML 博客（参考文献 p.58 逐字："Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026"）。另一个跨厂商注脚：其通用推理教师的增益来自 "additional large-scale SFT and RL on a separate reasoning mixture **generated by DeepSeek-V4-Pro**"——NVIDIA 的教师链上游用了 DeepSeek 生成的数据。

##### 【独立核验更正】之一：§3.3.5 的逐字引语无法独立核实，降级为待核

源稿 §5.9 与 §9 开放问题 5 把 §3.3.5 的 "a limitation of the on-policy distillation setting" 作为**一手逐字引文**呈现，并以 HLE 增益小作为"教师天花板"的核心证据（P6 的关键支撑）。回源核实时，**arXiv HTML 与 ar5iv 两条路径均在 §3.3.1 之后截断**，无法读到 §3.3.5 正文。

因此本页的处理：

- ⚠️ **不以逐字引号呈现**该句；
- ⚠️ "Nemotron 3 Ultra 在 HLE 等前沿基准上增益小、并自承这是 on-policy 蒸馏设定的局限"这一论点**降级为待核**（底稿记录来自另一取数路径的 PDF 提取，但本次未能独立复现）；
- **影响范围**：源稿 P6（"OPD 是能力放大器而非创造器"）在"一手承认教师天花板"这一条支撑上应视为**证据强度下调**。该论点仍有其他独立支撑（arXiv:2604.13016 的成功两条件；G-OPD/ExOPD arXiv:2602.12125 的 $\lambda$ 外推工具的存在本身即预设了教师天花板问题），但**不应再把 Nemotron 的自承作为最硬的一手证据引用**，除非重新取得 §3.3.5 原文（建议路径：NVIDIA 官方 PDF https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf p.28）。

##### 【独立核验更正】之二："两轮师生共进化"的机制被简化失真

源稿 §5.9 与 §5.12 写"**RLVR 后的学生反过来当下一轮教师**"。回一手 Figure 10 图注核实后，正确机制是**两条并行但不同的事**：

1. **第二轮教师由 Ultra MOPD1 初始化，并复用第一轮教师**。Ultra MOPD1 = 第一轮 MOPD 的**学生**产物。即第二轮的新/刷新教师是**从第一轮学生 checkpoint 出发再做专业化**得到的，同时第一轮的教师仍在教师池中继续使用——这是"教师池的滚动更新"，不是"学生原地变教师"。
2. **RLVR Student 是"自教师"（self-teacher）**，其作用范围被明确限定为 **"areas not covered by specialized teachers"**（专用教师未覆盖的领域）——是一个**补洞机制**，不是主教师。

两者都不等同于源稿的"RLVR 后的学生当下一轮教师"。正确的概括应是：**教师池随迭代滚动更新（新教师从上一轮学生初始化），并对未覆盖领域启用学生自教师兜底**。

这一更正削弱但不推翻源稿 §5.12 的"师生边界正在消融"判断：边界确实在消融（学生 checkpoint 成为下一轮教师的**初始化**、学生对未覆盖域**自教**），但消融的方式是"角色在训练拓扑中滚动"而非"学生直接晋升教师"。同一趋势的另外两条独立证据不受影响——MiMo 的教师池里明确包含学生自身（arXiv:2601.02780v2 Table 7 标注 Self）、GLM-4.5 的 iterative self-distillation（arXiv:2508.06471v1 §3.3.2）。

#### 4.6.2 Nemotron-Cascade 2（arXiv:2603.19220，2026-03-16，30B MoE / 3B 激活，63 页）

##### 【独立核验更正】之三：MOPD 在这份报告里是 Multi-**domain**，不是 Multi-teacher

源稿 §5.1 表与 §5.9 把 Nemotron-Cascade 2 的 MOPD 记为"多教师 OPD"。回一手核实：**§4.4 节标题为 "Multi-domain On-Policy Distillation (MOPD)"**（§4 章标题为 "Cascade RL and Multi-Domain On-Policy Distillation"），摘要亦写 "we introduce **multi-domain on-policy distillation** from the strongest intermediate teacher models for each domain throughout the Cascade RL process"。

这不是文字游戏，而是一个**术语陷阱**：缩写 MOPD 在 2026 年的工业报告里有**两种不同展开**——

| 展开 | 强调的轴 | 采用者 | 出处 |
|---|---|---|---|
| **Multi-teacher** On-Policy Distillation | 教师的**多个身份** | 小米 MiMo-V2-Flash、Kimi K3、NVIDIA Nemotron 3 Ultra、Baichuan-M3、方法论文 MOPD | 2601.02780v2 摘要；2607.24653v2 §4.1；2606.15007 摘要；2602.06570 摘要；2606.30406 |
| **Multi-domain** On-Policy Distillation | 覆盖的**多个领域** | NVIDIA Nemotron-Cascade 2 | 2603.19220 §4.4 节标题 |

同一家 NVIDIA 的两份报告用同一个缩写指两个不同的词组，检索与转述时极易串味。引用时请一律带展开全称。

##### 【独立核验更正】之四：71.5→85.5 是 ArenaHard V2.0 的 **Hard Prompt 子项**，且 "52 vs 160" 是论文自选对比点

源稿 §2.5 与 §5.9 写"MOPD 52 步胜过 RLHF 160 步（ArenaHard v2 71.5→85.5）"。回一手 Table 3 核实后需三处修正：

1. **71.5→85.5 是 Hard Prompt 子项，不是 ArenaHard V2.0 总分**；
2. **Creative Writing 另计**：40.6→71.0；
3. **Table 3 显示 RLHF 在 100 步已达 Hard Prompt 81.7**，160 步为 80.7 / 71.2。因此"52 步 vs 160 步"是论文**自选的对比点**——若以"到达 81.7 附近"为口径，RLHF 只需 100 步。

修正后的正确表述：**MOPD 在 52 步达到 Hard Prompt 85.5 / Creative Writing 71.0，高于 RLHF 在 100 步（Hard Prompt 81.7）与 160 步（80.7 / 71.2）的水平**。结论方向（MOPD 更 step-efficient）不变，但**"3 倍步数效率"的量化倍数被削弱**——按"达到 RLHF 100 步水平"口径，效率优势约为 2× 而非 3×。引用效率倍数时须注明口径。

其余一手内容（未受更正影响）：

- **算法（p.13, Eq. 2–4）**：token 级 reverse-KL 优势 $a^{\mathrm{MOPD}}_t=\log \pi_{\text{domain}_i}(y_t\mid s_t)-\log \pi_{\text{train}}(y_t\mid s_t)$，并明确 **"The log-probability difference is computed only on the student-sampled token rather than over the full vocabulary."**——这是**路线 B 的最强工业背书**；train–inference 失配用截断重要性加权 $\varepsilon_{\mathrm{low}}=0.5,\ \varepsilon_{\mathrm{high}}=2.0$。
- **教师三枚**：math = 初始 SFT ckpt、RLHF ckpt、multi-domain ckpt，全部来自 Cascade RL 流程内部，"without introducing external models"；学习率 2e-6，40–50 步收敛。
- **效率主张原文**："MOPD provides a dense token-level distillation advantage, whereas GRPO relies on a sparse sequence-level outcome reward that is shared across all generated tokens. This makes MOPD **substantially more sample- and step-efficient** in practice."
- **另一组数据（Fig. 3(c)）**：AIME25 上 GRPO 25 步 89.9→91.0，MOPD 30 步内达 92.0，"recovers teacher-level performance"。
- **引用集**（§4.4 逐字）："(Agarwal et al., 2024; Gu et al., 2024; Lu and Lab, 2025; Xiao et al., 2026; Yang et al., 2025; Zeng et al., 2026)"——GKD、MiniLLM、TML 博客均在列。

### 4.7 Thinking Machines Lab：命名者、产品化者与叙事供应方

TML 博客（Kevin Lu, 2025-10-27, DOI 10.64434/tml.20251026）的贡献**不在方法新颖性**——其自认承自 GKD/MiniLLM/Qwen3（原文："We extend prior on-policy distillation work by Agarwal et al., Gu et al., and the Qwen3 team."）——而在**叙事与可复现性**：

- **定义**："sample trajectories from the *student* model and use a high-performing teacher to grade *each token* of each trajectory."
- **2×2 框架**（采样分布 × 奖励信号密度）：SFT/离线蒸馏 = off-policy + dense；RL = on-policy + sparse；**OPD = on-policy + dense**。
- **实现**：per-token advantage 设为**负 reverse KL**，直接调用 RL 的 importance-sampling 损失——"在 RL 脚本上改一行"。折扣因子取零（"at any given timestep, the student only optimizes the immediate next token"）。
- **三档成本口径（逐字目验，引用时勿混用）**：数据集现成或已跨多次训练摊销 → **9×**；换算成 GPU 时（教师打分可廉价并行）→ **18×**；把 off-policy 基线的教师生成数据成本全部计入 → **30×**。基线均为"外推到 SFT-2M 达到约 70% AIME'24 的成本"。
- **实验**：Qwen3-8B-Base 学生从 SFT-400K（60% AIME'24）起，约 150 步 OPD 达 **70%**；同设置 RL 达 68%。
- ⚠️ **必须一并引用的脚注**：正文叙事用 Qwen3-32B 教师，但原文另一句明说 **"We actually use Qwen3-8B as a teacher, as it performs slightly better."** ——即真实实验的教师与学生**同尺寸**。这条脚注是 §6.3"教师不必更大"论断的原始证据。
- **引用时的版本注意**：页面已插入 2026-06 更新注记——"The teacher (Qwen3-32B) and student (Qwen3-8B-Base) used here have since been retired from the Tinker model lineup. The distillation recipes in the Tinker cookbook have been updated to use **Qwen3.5-9B-Base as the student and Qwen3.5-9B as the teacher**, so the experiments remain reproducible."
- **知识流动闭环**：博客用 Qwen3 做实验并逐字致谢——"The Qwen team also reports reaching a higher score of 74.4 on AIME'24 at one-tenth the cost of RL with on-policy distillation, **which served as inspiration for our work**"；随后 DeepSeek-V4、GLM-5、MiMo-V2-Flash、Kimi K3、NVIDIA 两份报告**反过来引用该博客**。这是一个罕见的"博客 ↔ 工业报告"互引闭环。

**Tinker cookbook 现状**：`on_policy_distillation.py`、`on_policy_multi_teacher.py`（多教师）、`harbor_multiturn.py` 与 `on_policy_distillation_harbor_multi_turn.py`（多轮工具使用，**环境 token 在训练中被掩蔽**——"only the student's generated tokens contribute to the loss"）、以及 **SDFT 配方**（Self-Distillation Fine-Tuning，实现自 arXiv:2601.19897；教师 = 拿到"问题 + golden answer 作为 in-context 示范"的同一模型，学生只拿问题，损失为学生补全每 token 位置上的 **forward KL**）。TML 官方 tinker-project-ideas 另提出 **on-policy context distillation** 课题。

### 4.8 Google / DeepMind：最早把 OPD 写进旗舰产品描述，也最早停止披露

这是全调研中证据链最硬的单条发现——Gemini 1.5 技术报告（arXiv:2403.05530v5 §3.2 逐字）：

> "Gemini 1.5 Flash is a transformer decoder model … and is also **online distilled** (Agarwal et al., 2024b; Anil et al., 2018; Beyer et al., 2021; Bucila et al., 2006; Hinton et al., 2015) from the much larger Gemini 1.5 Pro model."

模型卡 Table 45（p.105）重复同一表述。**关键交叉引用已核实**：参考文献页（p.75）确认 "Agarwal et al., 2024b" 就是 GKD/OPD 论文（"On-policy distillation of language models: Learning from self-generated mistakes. In ICLR 2024"）。这是**学术 OPD → 旗舰产品的最短公开证据链**。

⚠️ **但不能过度解读**：官方并列引用了 5 篇蒸馏文献（含 Anil et al. 2018 codistillation、Hinton et al. 2015），**不能断言 Flash 只用了 GKD 式 OPD**。【推断】官方措辞 "online distilled" + 该引用集合表明是在线/共训练蒸馏族方法，具体配方未公开。

**披露口径的三段式收缩**（这是本页认为比"用没用"更值得注意的现象）：

| 代际 | 蒸馏表述 | 是否引 GKD | 定位符 |
|---|---|---|---|
| Gemini 1.5（2024-03） | "**online distilled** from the much larger Gemini 1.5 Pro" | **是** | arXiv:2403.05530v5 §3.2 / Table 45 |
| Gemini 2.5（2025-07） | "use **distillation** … approximate it using a **k-sparse distribution** over the vocabulary" | 否 | arXiv:2507.06261 p.3 |
| Gemini 3 系（2025-11 起） | **无任何蒸馏表述**；无技术报告，仅 5 份模型卡，distill/teacher/GKD **全部 0 命中** | — | Gemini 3 Pro（10 页）/ 3 Flash（6 页）/ 3.1 Flash-Lite（7 页）/ 3.5 Flash（7 页）/ 3.6 Flash（7 页）模型卡 PDF |
| Gemma 2（2024-07） | 2B/9B 用 KD **替代** next-token prediction，训练量超 compute-optimal **50×** | — | arXiv:2408.00118 |
| Gemma 3（2025-03） | "The Gemma 3 models are **trained with distillation**"（教师未具名） | — | arXiv:2503.19786 |
| Gemma 4（2026-06） | **"distill" 0 命中**（已做验伪：同文档 "Gemini" 命中 5 次，提取正常） | — | arXiv:2607.02770，17 页 |

**必须写清的边界**：Gemini 3 系模型卡的训练披露只到 "Gemini 3 Flash is based on Gemini 3 Pro"（3 Flash 卡 p.3）与 "Gemini 3 Pro is trained using reinforcement learning techniques"（3 Pro 卡 p.3）这种粒度。**这只能说明披露口径收缩，不能推断 Google 弃用了蒸馏**。同理，⚠️ 二手博客称"KD 是 Gemma 4 的核心特征/defining characteristic"在 Gemma 4 技术报告中**找不到任何文字依据**，勿引。

**对齐侧的交叉工作**：BOND（arXiv:2407.14622）把 RLHF 化为 Best-of-N 分布蒸馏（用 **Jeffreys 散度**权衡 mode-covering/mode-seeking，迭代蒸馏 + moving anchor），用于 Gemma——这是"教师 = 推断时增强的自己"这一教师来源类④的代表作（见 §6.1）。

**2026 年动向**：定向检索未发现 DeepMind 在 2025-11 之后有以 OPD 为题的新论文（检索式：`Google DeepMind 2026 on-policy distillation paper GKD speculative knowledge distillation`）。

### 4.9 Meta 与 Mistral

**Meta**：Llama 3.2 1B/3B（官方博客 2024-09-25）= 从 Llama 3.1 8B **单次结构化剪枝** + 以 3.1 8B/70B 的 logits 做 token 级目标的预训练蒸馏（"Knowledge distillation was used after pruning to recover performance"）。Llama 4（官方博客 2025-04-05）= Maverick 从 Behemoth **codistilled**，"novel distillation loss that dynamically weights the soft and hard targets"，且 "Codistillation from Llama 4 Behemoth **during pre-training** amortizes the computational cost of resource-intensive forward passes"。

【推断】codistillation 的教师目标虽是在线计算的，但**数据是固定预训练流而非学生 rollout**——属"在线计算的 off-policy 蒸馏"，**不是 TML 定义的 OPD**。这一区分很重要：不能因为"在线"二字就把 Llama 4 划进 OPD 阵营。复核确认截至 2026-08 无 Llama 4.5/5 发布，Llama 4 codistillation 仍是 Meta 最新公开做法。

**Mistral 的"双面立场"**（源稿曾有失真风险，此处按复核结论精确表述）：

- **拒绝蒸馏的部分，仅限 Magistral Medium**（arXiv:2506.10910v1）：贡献列表逐字 "We present in detail how we trained **Magistral Medium with RL alone, with no distillation from pre-existing reasoning models**, yielding a nearly 50% boost in AIME-24 (pass@1)"；§5.2 标题即 "Magistral Medium – reasoning RL from scratch"；摘要 "Instead of relying on existing implementations and RL traces distilled from prior models, we follow a ground up approach"。§6.2 并明确反驳 DeepSeek-R1 的结论——"**our findings contradict this observation**: we achieved strong results even with pure RL"，同时承认 "RL on top of the distilled checkpoint can yield even better performance, leading to over 5 points gain across various benchmarks"。
- **拥抱蒸馏的部分**：**Magistral Small 恰恰是用 Medium 的推理轨迹做 SFT + RL 得到的**（Figure 4 caption 逐字："We use these generated traces to finetune Mistral Small 3 and then perform RL to get Magistral Small."）；**Ministral 3**（arXiv:2601.08584）以 **Cascade Distillation**（"an iterative pruning and continued training with distillation technique"）为核心配方，全家族 14B→8B→3B 逐级剪枝 + 蒸馏，parent = Mistral Small 3.1（24B）。

**正确的一句话概括**：Mistral 是唯一公开声明**旗舰推理模型不用任何教师蒸馏冷启动、纯 RL 从零起步**并给出反 R1 结论的厂商；但**向下压缩仍全面依赖剪枝 + logit 蒸馏**。只引 Magistral 会造成"Mistral 反蒸馏"的失真印象。

### 4.10 垂直与二线厂商：OPD 的长尾

- **Agents-A1**（上海 AI Lab / InternScience，arXiv:2606.30616v2，35B MoE）：摘要逐字 "we propose a **multi-teacher domain-routed on-policy distillation with salient vocabulary alignment** to improve knowledge transfer efficiency across different domains, unifying six heterogeneous domains into one deployable student model"；机制（§2.3 / §4.3）= 教师侧 **top-k 支持集上的截断反向 KL** + 域归一化聚合。这是 §5 路线之争中的**中间路线**（既非全词表也非单采样 token）。
- **Baichuan-M3**（arXiv:2602.06570，医疗 235B）：三阶段框架逐字——"Task-Specific Reinforcement Learning (TaskRL), Offline Policy Distillation, and **Multi-Teacher Online Policy Distillation (MOPD)**"；Stage 2 用**前向 KL** 模仿（mode-covering），**Stage 3 MOPD** 逐字 "the student model re-enters the online interaction environment, performing rollouts across mixed domain distributions … this stage employs **reverse KL regularization**"，学生**同时**受任务奖励与多教师先验约束（"从模仿者变为决策者"），并支持循环迭代。这是"FKL 打底 → RKL 精修"散度调度的少见工业实例。
- **快手 / 理想 ⚠️**（均取自第三方全读清单，未自查全文）：Keye-VL-2.0（arXiv:2606.10651，30B-A3B，跨模态 MOPD，**13 个 RL 域专家教师池**）；KAT-Coder-V2（arXiv:2603.27703，**step 级 OPD**）/ V2.5（arXiv:2607.05471，5 专家 MOPD + 漂移感知截断）；理想 Mach-Mind-4-Flash（arXiv:2607.09375，35B-A3B，"RL 与 OPD 单一加权损失切换"设计）。

---

## 5. 两条实现路线：一手证据的正面对立

### 5.1 路线定义

同为"学生采样 + 教师逐 token 监督"，工业实现分裂为两支，这是 2026 年最重要的 taxonomy 轴之一。

| 路线 | 机制 | 教师需返回什么 | 采用者 | 出处 |
|---|---|---|---|---|
| **A. 全词表 logit 反传** | 教师在学生轨迹每个位置给出完整（或 top-k）词表分布，KL **直接反传** | 每位置完整或 top-k 分布 | Qwen3、DeepSeek-V4 | arXiv:2505.09388v1 §4.5；arXiv:2606.19348v1 §5.1.2 |
| **B. token 级 KL-as-reward（复用 RL 框架）** | 把 $\hat{A}_t=\mathrm{sg}\big[\log\frac{\pi_T(y_t\mid\cdot)}{\pi_\theta(y_t\mid\cdot)}\big]$ 当作 advantage 塞进 GRPO/PPO | **仅采样 token 的 logprob**（标量/位置） | GLM-5、MiMo-V2-Flash、Kimi K3、Nemotron 3 Ultra、Nemotron-Cascade 2、TML/Tinker | arXiv:2602.15763v2 §3.5；arXiv:2601.02780v2 §4.4；arXiv:2607.24653v2 §4.1.3；arXiv:2606.15007 §3.3.1；arXiv:2603.19220 §4.4；TML 博客 |
| **中间态：top-k 截断** | 教师 top-k 支持集上的截断 RKL | top-k 分布 | Agents-A1、veRL 的 `forward_kl_topk` 模式、Gemini 2.5 的 k-sparse 存储 | arXiv:2606.30616v2 §2.3；veRL OPD 文档；arXiv:2507.06261 p.3 |

【推断】这两条路线是 2023 年学术分岔的直接投影：**路线 A 是 GKD 式监督反传的后代**（轨迹采出后视为常量、逐 token 散度可微分直接反传，绕开策略梯度）；**路线 B 是 MiniLLM 式策略梯度的后代**（似然比当稠密奖励）。

### 5.2 正面对立的三段一手文本

**DeepSeek-V4 批评 B（arXiv:2606.19348v1 §5.1.2）**：把全词表 KL 简化为逐 token 采样估计并复用 RL 框架 "**leads to high variance in gradient estimation and often causes training instability. Therefore, we adopt full-vocabulary logit distillation in our OPD.**"

**Kimi K3 反证（arXiv:2607.24653v2 §4.1.3）**："**While we also experimented with more fine-grained top-k distillation objectives, we observed no clear advantage in either convergence speed or final performance in our setting.**"其坚持 B 的理由是稠密奖励 "seamlessly integrates into our RL framework, naturally enabling infrastructure-level optimizations such as partial rollout training for long-horizon tasks"，并用 **clip 截断**替代 V4 所担心的极端 advantage。

**Nemotron-Cascade 2 站队 B（arXiv:2603.19220 §4.4）**："The log-probability difference is computed only on the student-sampled token rather than over the full vocabulary."配截断重要性采样权重 $\varepsilon\in[0.5,2.0]$。

三段文本的对立是**真实的、无法用"各说各话"化解的**：V4 说 B 高方差不稳定，K3 说细粒度目标（更接近 A）没有优势，NVIDIA 用 B 拿到了 step 效率结果。

### 5.3 学术侧的仲裁尝试与目前的结论

学术侧的对应理论：token 级 OPD 相对序列级 reverse-KL **有偏但方差界更紧**（arXiv:2603.25562v2）——**两条路线本质是偏差-方差权衡的两端**。因此 V4 与 K3 互相矛盾的经验报告并不构成"一方说错"，而是说明**最优点依赖具体设置**（学生规模、教师数量、序列长度、异步程度、是否混入任务奖励）。

本页的结论：**路线之争在 2026-08 无定论，且短期内不会有**。理由是两条路线的选择往往不由算法效果决定，而由**既有基建决定**——

- 已有大规模 RL 栈（GRPO/PPO + partial rollout + 异步 rollout 服务）的团队，路线 B 的边际成本近零（TML 的"一行改动"、K3 的"seamlessly integrates"、GLM-5 直接换 advantage 且 group size 降到 1）；
- 愿意为蒸馏专门建基建的团队才走路线 A（DeepSeek-V4 为此建了万亿参数教师权重 offload 调度、按教师索引排序样本、FP4 QAT 师生一致性三套东西）。

**信号密度-成本谱系**（由上表归纳，与 Infra 篇的带宽账互补）：全词表（DeepSeek-V4，单条 16K 序列 × 128K 词表 bf16 约 4.2 GB/轨迹/教师）→ top-k（veRL GKD 模式、Agents-A1、Gemini 2.5 k-sparse）→ 单采样 token 的 logprob（slime、GLM-5、MiMo、K3、Nemotron，带宽极低）。**路线选择本质上是在为"每条轨迹每个教师传多少字节"定价**。【推断】

### 5.4 框架侧的中立：veRL 同时实现两者

veRL 官方 OPD 算法页（https://verl.readthedocs.io/en/latest/algo/opd.html ）实现了两种损失：**GKD OPD**（`forward_kl_topk`，教师 top-k 反传）与 **PG OPD**（`k1/k3` reverse-KL 当奖励走策略梯度），并提供独立教师资源池、多教师按样本 metadata 路由、`use_task_rewards` 可混 PPO 奖励（对齐 MiMo Eq. 9 的用法）。这意味着**路线之争在框架层已被降级为一个配置项**——对使用方而言，两条路线是可以 A/B 的旋钮而非架构承诺。

---

## 6. 教师从哪里来：五类来源与教师生产经济学

OPD 文献大多把教师当作给定输入，很少正面回答"教师本身如何得到"——但这恰是理解 OPD 经济学的关键：**教师的生产方式决定了蒸馏的真实成本口径**。

### 6.1 五类教师来源

| 来源 | 生产方式 | 代表案例 | 出处 |
|---|---|---|---|
| **① 更大的旗舰（既有资产摊销）** | 旗舰走完整后训练管线（如 Qwen3 四阶段：冷启动→推理 RL→模式融合→通用 RL），产出即教师 | Qwen3（教师 = Qwen3-32B / 235B-A22B）；Gemini 1.5 Pro→Flash；HY-MT1.5（7B→1.8B）；Llama 4 Behemoth→Maverick；Ministral 3（parent = Mistral Small 3.1） | 2505.09388v1 §4.5；2403.05530v5 §3.2；2512.24092 §2.3；Meta 博客 2025-04-05；2601.08584 §1 |
| **② 同基座领域专家（为蒸馏专门生产）——2026 主流** | 同一 pre-trained base checkpoint 分头做"领域 SFT + 大规模 RL"；K3 再乘一维：对长度预算 $\tau$ 退火得到 low/high/max 三档 effort 教师 | DeepSeek-V4（10+ 专家）；Kimi K3（3 域 × 3 effort = 九专家）；MiMo；Agents-A1（六域）；Baichuan-M3；Nemotron 3 Ultra（十余领域）；⚠️ Keye-VL-2.0（13 专家） | 2606.19348v1 §5.1.1；2607.24653v2 §4.1.2；2601.02780v2 §4.1；2606.30616v2 §4.3；2606.15007 §3.3.2 |
| **③ 自己的其他版本（近零额外训练）** | 前序训练阶段的 final checkpoint；midtrain 之前的早期版本；拿到特权上下文（golden answer in-context）的同一模型 | GLM-5（前序各阶段 checkpoint）；Nemotron-Cascade 2（内部 Cascade RL checkpoint，三枚）；Nemotron 3 Ultra（第二轮教师由 Ultra MOPD1 初始化 + RLVR 自教师补洞）；TML 个性化实验（早期版 Qwen3-8B）；SDFT | 2602.15763v2 §3.5；2603.19220 §4.4；2606.15007 Fig. 10；TML 博客；2601.19897 |
| **④ 推断时增强的自己** | 教师分布 = 自身 Best-of-N 分布；或教师提供的已验证解（仅对学生解不出的难题）；或**同一模型在更强输入条件下的响应** | BOND（Best-of-N 分布，Jeffreys 散度）；TREK（forward KL 注入已验证解扩展探索支持集）；**Qwen3.5-Omni（文本条件 → 音频条件）** | 2407.14622；2607.05339；2604.15804v2 |
| **⑤ 外部模型** | 开放权重直接用（白盒，需 logits + 共享词表或跨词表方法）；API 黑盒走判别器路线（GAD）或数据级；未经授权的采集即"蒸馏攻击" | Nemotron 3 Ultra 的教师数据上游用 DeepSeek-V4-Pro 生成；GAD（对 GPT-5-Chat）；R1→Qwen/Llama（数据级） | 2606.15007 §3.3.5 语境；2511.10643；2501.12948 §2.4 |

**教师矩阵的维度**：2026 年的教师池不再是一维列表，而是至少三维——**领域**（V4 10+ / K3 3 / Nemotron Ultra 10+ / Agents-A1 6 / ⚠️ Keye 13）× **reasoning effort 档位**（K3 的 low/high/max；V4 的 Non-think / Think High / Think Max）× **训练阶段**（GLM-5 的前序 checkpoint、Cascade 2 的流程内 checkpoint、Nemotron Ultra 的两轮迭代）。

### 6.2 教师生产的经济学

- **类①的教师成本是摊销出来的**：Qwen3 的"1/10 GPU 时"**不含教师训练**——旗舰四阶段是无论如何都要做的既有投资。TML 的 9× / 18× / 30× 三档口径，区分的正是教师成本"数据集现成 / 打分可并行 / 全部计入"三种算法。**这也是为什么引用效率数字必须带口径**（回看 §4.1 的两条限定：Qwen3 的 1,800 vs 17,920 口径不明，且起点行 GPU 时未报告）。
- **类②的专家训练是当前旗舰后训练算力的主要去向**：DeepSeek-V3.2 原文 "Each specialist is trained with **large-scale Reinforcement Learning (RL) computing**"（arXiv:2512.02556v1 §3）。贵的是用 RL 造专家，便宜的是用 OPD 整合（Nemotron-Cascade 2：MOPD 52 步 vs RLHF 100 步的对比，见 §4.6.2 更正后口径）。这给"RL 创造能力、OPD 搬运能力"补上成本刻度：**创造仍然贵，搬运变得便宜**。
- **算力模型**（arXiv:2604.00626 §7.4）——off-policy 与 on-policy 的成本差：

  $$
  C_{\mathrm{off}}\approx N\,(F_{T}+F_{S}+B_{S}),\qquad C_{on}\approx N\,(G_{S}+\rho F_{T}+F_{S}+B_{S})
  $$

  其中 $F/B$ 为前向/反向 FLOPs，$G_S$ 为学生自回归生成成本（$G_S\gg F_S$，这是 on-policy 溢价的主体），$\rho\in(0,1]$ 为教师监督刷新率。**注意这个公式里没有教师的训练成本**——它只算蒸馏阶段，这正是"1/10 GPU 时"类数字的隐含边界。
- **判据层面**，Distillation Scaling Laws（arXiv:2502.08606，ICML'25，⚠️ Apple 归属为通行说法）的结论可直接翻译为教师来源决策：**教师已存在（类①③④）或一位教师蒸多个学生（类②的复用）时蒸馏才划算；为单个学生现训教师通常不如直接训学生**。这解释了为什么类②在 2026 年成为主流——旗舰厂商的专家教师**天然被复用**（一批专家蒸一个通才，且往往跨多个尺寸的学生复用）。

### 6.3 三个非显然的结论

**结论 1：教师不必更大。**

三条独立证据：(a) TML 博客明说数学实验实际用 **Qwen3-8B 当教师**（与学生同尺寸，"performs slightly better" 于 32B）；(b) HuggingFace 2026 年度盘点把"教师是同等规模的领域专家而非更大模型"列为年度范式；(c) 类②的专家教师普遍与学生同规模（V4、K3、MiMo、Nemotron 全部如此）。

**教师的优势来自"专"（RL 深耕单域）而非"大"**——这也解释了为什么教师矩阵会长出"effort 档位"这种与规模完全无关的维度（K3 的 $\tau$ 退火产出 low/high/max 三档教师）。

**结论 2：同源性比规模更重要——且同时是效果条件与风险条件。**

MOPD 方法论文（arXiv:2606.30406，小米 LLM-Core / PKU / HKU / RUC，已部署于 MiMo-V2-Flash）的核心结论是 **"same-origin teachers are essential"**；Rethinking OPD（arXiv:2604.13016v2）的成功条件之一即师生思维模式兼容，机制上 **97–99% 的概率质量对齐发生在共享 token 集上**。工业侧的对应事实是：V3.2/V4/K3/Qwen3.5-Omni 的专家教师**全部从同一 pre-trained base checkpoint 微调**（V3.2 §3 逐字："with all specialist models being fine-tuned from the same pre-trained DeepSeek-V3.2 base checkpoint"；Qwen3.5-Omni 逐字："All teacher models are fine-tuned from the pre-trained Qwen-3.5 base checkpoint"）。

**但同源性同时是风险条件**：subliminal learning（arXiv:2507.14805；Anthropic Fellows 计划，官方博客 2025-07-22）证明教师的隐藏特质（含 misalignment）可经语义无关数据传递给学生，**且关键条件正是师生共享同一 base model——跨基座不发生**。【推断】选择同源教师时，效果与安全风险是同一枚硬币的两面；且 OPD 的逐 token 软监督信息带宽远大于该论文实验所用的 hard-label 数据蒸馏，风险敞口值得专门研究（原论文实验为数据蒸馏，此外推需谨慎）。

**结论 3：师生边界正在消融，但消融的方式需要精确描述。**【本页按更正后重述】

三条证据：

- **Nemotron 3 Ultra 的两轮迭代**：第二轮教师**由 Ultra MOPD1（第一轮 MOPD 的学生）初始化**并复用第一轮教师；**RLVR Student 作为"自教师"覆盖专用教师未涉及的领域**（arXiv:2606.15007 Fig. 10）。【独立核验更正：这不等同于"RLVR 后的学生直接当下一轮教师"】
- **MiMo 的教师池里明确包含学生自身**（arXiv:2601.02780v2 Table 7 标注 Self）——且 §4.4.1 显示恰恰是这一格（创意写作）出现了真回退。
- **GLM-4.5 的 iterative self-distillation** 是其数据级前身（arXiv:2508.06471v1 §3.3.2）；GLM-5 则把它升级为 on-policy 版。

教师正在从"静态资产"变成"训练流程中滚动更新的角色"。【推断】照此趋势，"教师/学生"最终可能退化为同一模型在训练拓扑中的两个端口，OPD 与 self-play 的边界将变得模糊。

---

## 7. 未采用阵营与不披露阵营

一份只写"谁在用"的全景图会系统性高估收敛程度。本节按证据强度分三档呈现反面。

### 7.1 明确未采用（一手全文核实）

**MiniMax（反例经三重验证加固）**：

- M2 系报告 **v2**（arXiv:2605.26494v2，2026-07-30 修订，覆盖 M2→M2.5→M2.7）**全文 "on-policy" 0 命中**。蒸馏相关仅三处且均非 OPD：agent 轨迹数据级（"trajectories are distilled from a rotating set of strong teacher models under deliberately perturbed scaffolds"）、**prompt distillation**（采样时给全量增强 system prompt、训练时选择性丢弃，把最佳实践内化为默认行为——**不是模型间蒸馏**）、以及 §6.2.6 的 MTP 草稿头 top-K KL 损失（投机解码，非蒸馏）。
- M1（arXiv:2506.13585）全文 "distill" **0 命中**；唯一 "on-policy" 在 CISPO/RL 裁剪讨论中。
- M3 对应的是纯架构论文 MiniMax Sparse Attention（arXiv:2606.13392），唯一一次 "distillation" 是描述 KL 梯度路由的失效模式（"a self-distillation effect: the backbone can lower the KL loss by simplifying the Main Branch attention distribution"），与训练法蒸馏无关；M3 的完整训练报告未见。
- 第三方全读式清单（thinkwee/AwesomeOPD，自称对 2026-04-25→06-20 区间 152 篇全文通读）工业表**零 MiniMax 条目**。

⚠️ 残余缺口：minimax.io 的 M2.1 后训练官方博客未抓取成功，未逐字验证（HF M2.1 卡 0 次 distill 为替代证据）。

**字节 Seed**（源稿曾写"无公开资料"，此处按复核结论修正措辞）：

- 旗舰层面无蒸馏：Seed-Thinking-v1.5（arXiv:2504.13914）**全文 "distill" 0 命中**（唯一 "on-policy" 在 §5.2 Streaming Rollout System 讨论 RL 采样新旧比例）；Seed 2.0（2026-02-14 发布）/ 2.1（2026-06-23 发布）**均无公开技术报告**。
- ⚠️ 公司立场（媒体报道级）：2026-08 多家媒体报道创始人明确要求 Seed 团队"不用蒸馏构建模型，即使短期落后"。这本身是可引用的公开表态，但属舆情级证据。
- **但"不参与研究"是另一回事**：ByteDance Seed **合著了 OPD 方法论文**——Direct-OPD（arXiv:2607.05394，Weak-to-Strong Generalization via Direct On-Policy Distillation，署名含 ByteDance Seed）与 ShortOPD（arXiv:2607.13124，剪枝恢复自蒸馏）。且字节生态的 **veRL 框架官方支持 OPD**（见 §5.4）——**"不用于旗舰"与"不参与研究/不提供基建"是三件不同的事**。

### 7.2 双面立场：Mistral

见 §4.9。要点：**"拒绝蒸馏"仅适用于 Magistral Medium 且仅指"从既有推理模型蒸馏冷启动"**；Magistral Small 与整个 Ministral 3 家族都是蒸馏产物。

### 7.3 不披露阵营（零命中级证据）

**OpenAI**：

- 产品侧机制已核实：Model Distillation（2024-10 DevDay）= `store=True` 存储大模型补全 → metadata 过滤 → 对小模型 SFT，**无 logprobs / KL / 软标签**，是教科书式 off-policy 序列级蒸馏产品化；现已 301 重定向并入 SFT 指南的一小节。
- 模型侧零披露：GPT-4o mini 官方发布文**全篇不含 "distillation"**；**GPT-5 系统卡（60 页，覆盖 mini/nano，全文 "mini" 88 次）与 GPT-5.5 系统卡（2026-04-23，45 页）distill/teacher 均 0 命中**；GPT-5.4 mini/nano 公告（2026-03-17）全文 "distill" 0 次。
- **"mini 是蒸馏产物"的流行说法至今无任何官方依据**——这是源稿复核中被验伪的一条流行叙事。
- 相关但不同的机制：Deliberative Alignment（arXiv:2412.16339v2 §2.1）用的是 **context distillation**（安全 spec 放进系统提示 → 生成带 CoT 补全 → judge 过滤 → **去掉 spec 上下文**做 SFT），与 OPD 机制不同。

**Anthropic**：

- Claude 3 模型卡 42 页 + Claude 3.5 Haiku 附录卡 14 页 + Haiku 4.5 系统卡 39 页，**合计 95 页，distill/teacher 全部 0 命中**；截至 2026-08 无更新的 Haiku（Haiku 档仍为 4.5）。Haiku 与 Opus/Sonnet 的训练关系**确认无任何公开信息**。
- Anthropic 对本领域的公开贡献在**安全侧**：subliminal learning 研究（§6.3 结论 2）与 2026-02-23 的《Detecting and preventing distillation attacks》官方新闻稿（https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks ）——披露约 24,000 个假账户、1600 万+ 次交互，归因 DeepSeek（15 万次）、Moonshot（340 万次）、MiniMax（1300 万次）。⚠️ **须严格区分**："蒸馏（技术，本页主题）"与"违反 ToS 的输出采集行为（指控内容）"是两件事；且以上均为公司单方披露，无司法认定。

**Google**：见 §4.8 的三段式披露收缩表。**披露中断 ≠ 弃用**。

### 7.4 方法论注记：否定性结论的半衰期以周计

源稿把这一点提升为核心论点 P7，本页认为它对工业全景页尤其重要，因为**本页 §7 的每一条否定性结论都可能在数周内失效**。已发生的三次推翻：

1. **"Kimi 全系不用蒸馏"** → 被 K3（arXiv:2607.24653，2026-07-27）推翻，发布时间早于调研基线仅两周。
2. **"腾讯混元无 OPD"** → 被 HY-MT1.5（arXiv:2512.24092 §2.3）推翻；更讽刺的是，**首个 OPD 专题综述的作者团队（Mao Zheng）正是腾讯混元 MT 团队**——即综述作者自己团队的模型就是 OPD 生产实践。
3. **"Qwen 2026 年动向无公开细节"** → 被 Qwen3.5-Omni（arXiv:2604.15804v2）推翻。

**可操作的规范**：本页所有否定性结论都附了检索式或"全文 N 次命中"的可复现口径（如"arXiv:2605.26494v2 全文 on-policy 0 命中"）；引用任何"某厂商不用 OPD"时，**必须一并注明核实日期与核实的具体文档版本**，否则该结论无法被证伪也无法被更新。

---

## 8. 本页的四个交叉观察

1. **"OPD 替换了什么"比"OPD 是什么"更能解释其扩散速度。** 四类用途分别替换了：小模型的完整后训练管线（①）、混合 RL（②）、多阶段 RL 的能力回归修复（③）、模态间能力落差的专门训练（④）。每一类被替换的东西都**已经很贵且已经在做**——OPD 的扩散不需要新预算，只需要重新分配既有预算。这解释了为什么它能在一年内跨越五家头部厂商。【推断】

2. **路线之争的胜负由基建而非算法决定。** §5.3 已述。一个可检验的推论：**随着 veRL/slime/NeMo-RL 把两条路线都做成配置项，路线之争会从"厂商立场"退化为"任务级超参"**，V4 与 K3 的对立文本将成为历史文献而非活跃争议。【推断】

3. **教师矩阵的维度膨胀是新的复杂度来源。** 从"一个教师"到"域 × effort × 阶段"的三维教师池，带来了一批只在多教师场景出现的问题：多教师破坏性干扰（能力区不重叠的教师逐 token 信号相消而非组合，Counteraction-Aware MOPD，经 arXiv:2604.00626 §7.2 转述）、教师权重 offload 调度（V4 §5.2.2）、按样本 metadata 路由（veRL）。**"多教师"不是"单教师 × N"，而是一个独立的问题域。**

4. **披露度与采用度正在反向运动。** 中国厂商的报告越来越详细（V4 给公式与批评、K3 给消融结论、MiMo 给完整 Table 7），而 Google/OpenAI/Anthropic 的披露越来越少（Gemini 三段式收缩、GPT-5.5 系统卡 0 命中、Haiku 95 页 0 命中）。**结果是：2026 年关于 OPD 的一手工业知识主要来自中国厂商与 NVIDIA 的技术报告**——这本身是理解本页证据分布不均的关键背景。【推断】

---

## 9. 五处独立核验更正汇总

本页在写作过程中回一手原文核实，发现源稿五处与原文不符。**凡引用这五处，请以本页表述为准。**

| # | 位置（源稿） | 源稿表述 | 一手核实结论 | 本页展开 |
|---|---|---|---|---|
| 1 | §5.5 / §5.1 表（MiMo Table 7） | "如实报告了 BrowseComp −6.3、创意写作 −3.9 的失分域" | **Table 7 的 Δ 列 = 学生 − 最强教师**。BrowseComp 学生 42.5→45.4（**上升 +2.9**），−6.3 是与 SFT 教师 51.7 的**差距**而非失分。真回退是 Arena-Hard 创意写作 90.1→86.2（教师即 Self）与 GPQA-Diamond 84.9→84.3；SWE-Bench 67.8→73.4 的 Δ=−0.8 也是差距而非回退 | §4.4.1 |
| 2 | §2.5 / §5.9（Nemotron-Cascade 2） | "MOPD（多教师）… 52 步胜过 RLHF 160 步（ArenaHard v2 71.5→85.5）" | (a) §4.4 节标题为 **"Multi-domain On-Policy Distillation"**，是 multi-**domain** 不是 multi-teacher；(b) 71.5→85.5 是 **ArenaHard V2.0 的 Hard Prompt 子项**，非总分，Creative Writing 另计 40.6→71.0；(c) Table 3 显示 **RLHF 在 100 步已达 Hard Prompt 81.7**，"52 vs 160" 是论文自选对比点 | §4.6.2 |
| 3 | §5.9 / §7.1 / §9-5（Nemotron 3 Ultra §3.3.5） | 以逐字引号呈现 "a limitation of the on-policy distillation setting"，作为教师天花板的一手证据 | **无法独立核实**：arXiv HTML 与 ar5iv 两条路径均在 §3.3.1 后截断。该引语及 HLE 语境**降级为待核（⚠️）**，不得以逐字引号呈现；P6 在此条上的证据强度下调 | §4.6.1 |
| 4 | §5.9 / §5.12 结论 3（Nemotron 3 Ultra Fig. 10） | "RLVR 后的学生反过来当下一轮教师" | Figure 10 图注：**第二轮教师由 Ultra MOPD1（第一轮 MOPD 的学生）初始化，并复用第一轮教师**；**RLVR Student 是"自教师"（self-teacher）**，仅用于专用教师未覆盖的领域。二者均不等同于源稿表述 | §4.6.1、§6.3 结论 3 |
| 5 | §9 开放问题 4（MiMo Eq. 7–8） | "MiMo 已用 **IcePop 式截断**" | (a) 正文**未出现 "IcePop" 一词**（仅见于参考文献标题），原文为 **"Following Zhao et al. (2025)"**；(b) Eq. 8 的截断作用于**训推比 $\pi_\theta/\mu_\theta$**（训练策略 vs 采样策略），**不是师生比值**——这是训推不一致修正，与 OPD 的师生 KL 是两件事 | §4.4.2 |

**同时确认为准确、可放心作一手结论引用的**（本次已回一手原文逐字核对）：

- **DeepSeek-V4 §5.1.2 全部内容**：`the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD)`；Eq. 29 为 $L_{\mathrm{OPD}}(\theta)=\sum_i w_i D_{KL}(\pi_\theta\|\pi_{E_i})$；`more than ten teacher models covering various domains`；对路线 B 的批评逐字为 `it leads to high variance in gradient estimation and often causes training instability. Therefore, we adopt full-vocabulary logit distillation in our OPD.`
- **Kimi K3 §4.1.3**：Eq. 15 的 clip 形式；`we also experimented with more fine-grained top-k distillation objectives, we observed no clear advantage in either convergence speed or final performance in our setting`；九专家 = 三域 × 三档 reasoning effort $e\in\{\text{low},\text{high},\text{max}\}$。
- **Qwen3**：Table 21 主稿转录的 12 格数字全部准确；`aligning its logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence` 逐字无误；pass@64 只在蒸馏下提升有明确原文支撑。**但须补两条限定**（见 §4.1）：off-policy 起点行的 GPU 时是 "–"（未报告），故 1,800 vs 17,920 只是两个增量阶段之比；报告全文未说明这两个数字是否计入教师推理成本。

---

## 10. 本页级待核与存疑清单

1. **Nemotron 3 Ultra §3.3.5 原文**（教师天花板论述）：arXiv HTML / ar5iv 均截断，建议改从 NVIDIA 官方 PDF（p.28）取。⚠️ 在取得前，"OPD 有教师天花板"的一手工业证据处于空缺状态。
2. **Qwen3 的 1,800 / 17,920 GPU 时是否含教师推理成本**：报告未说明；off-policy 起点行 GPU 时为 "–"。
3. **Qwen3 未披露** on-policy 阶段的 KL 方向与是否全词表（§4.5 只说 "aligning its logits … minimize the KL divergence"），路线 A 的归类基于"logit 对齐"措辞的合理解读。
4. **Qwen3.5-Omni 未披露** OPD 的损失细节（散度方向、词表粒度）。
5. **⚠️ 清单级、未自查全文的四条**：HY-Embodied-0.5（arXiv:2604.07430）、Keye-VL-2.0 正文（arXiv:2606.10651，仅核实 abs 标题）、KAT-Coder-V2 / V2.5、Mach-Mind-4-Flash。
6. **⚠️ Hunyuan-A13B**："无 OPD"依据第三方清单的排除名单（二手），其报告 PDF 在 GitHub 未做逐字提取。
7. **⚠️ minimax.io 的 M2.1 后训练官方博客**未抓取成功；MiniMax M3 完整训练报告若后续发布需重查。
8. **⚠️ DeepSeek-V4 的教师隐藏状态缓存 / TileLang 内核**：来自 Labonne 二手分析，与报告 §5.2.2 相容但未见于原文。
9. **⚠️ 字节 Seed 创始人表态**：媒体报道级（2026-08）。
10. **⚠️ Anthropic 致美参议院信函**（涉 Alibaba/Qwen 的 2880 万次交互指控）：原件未取得，仅二手转述。
11. **机构归属类推断**：Distillation Scaling Laws = Apple、GAD = 微软、SDFT = MIT/ETH、MOPD 方法论文归属等，abs 页未列，属推断或二手。

---

## Related Pages

- [[14_on_policy_distillation_analysis]] — OPD 的定义、2×2 框架、学术谱系与总论（本页的上位页）
- [[15_opd_divergence_and_objective_evolution_analysis]] — 散度选择与目标函数演进（FKL→RKL→JSD/f-散度→自适应混合），本页 §5 路线之争的算法前史
- [[33_opd_effectiveness_and_failure_modes_analysis]] — 有效性条件、失败模式清单与决策框架；本页 §4.4.1 的创意写作回退、§6.3 的同源性风险在此有机制级展开
- [[13_opd_infra_mechanism_analysis]] — 训练回路系统分解、信号格式带宽账、框架选型；本页 §5.3 的信号密度-成本谱系在此展开
- [[25_on_policy_off_policy_staleness_analysis]] — 异步训练的 off-policyness 与陈旧度修正，与本页 §4.4.2 的训推比截断直接相关
- [[13_reasoning_rl_algorithm_evolution_analysis]] — 推理 RL 算法演进；本页"OPD 替换混合 RL"的对照面
- [[20_grpo_analysis]] — GRPO；路线 B 的宿主算法（GLM-5 直接替换其 advantage 并把 group size 降到 1）
- [[22_gspo_analysis]] — GSPO；Qwen3-Next 模型卡唯一声明的后训练方法
- [[26_tim_causal_chain_analysis]] — 因果链视角的训练机制分析
- [[13_deepseek_v4_analysis]] — DeepSeek-V4 专页（本页 §4.2 的展开）
- [[01_glm_5_analysis]] — GLM-5 专页（本页 §4.3 的展开）
- [[14_kimi_k3_analysis]] — Kimi K3 专页（本页 §4.5 的展开）
- [[01_theory/04_posttraining/index]] — 后训练理论索引
