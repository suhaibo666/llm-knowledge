# OPD 框架支持逐项对照与选型：veRL / slime / TRL / NeMo-RL / Tinker / KDFlow

> **定位**：本页是"选哪个框架做 on-policy distillation（OPD）"的对照表与决策依据。系统机制（训练回路、带宽账、八项 Infra 工作 W1–W8）见 [[13_opd_infra_mechanism_analysis]]；通用 RL 框架横评见 [[30_rl_framework_comparison]]；算法总览见 [[14_on_policy_distillation_analysis]]。
> **最后更新: 2026-08-11**
> **保真度约定**：正文默认为一手核实内容（框架官方文档 URL、技术报告原文指定章节，均在调研中逐条打开）；**⚠️** = 仅有二手来源或本轮未能独立核实；**【推断】** = 本页作者的分析推断，非来源声称；**【本文推算】** = 本页作者的算术。§10 单列本页相对上游综述稿的独立核验更正。

---

## 1. 对照之前必须先定的两个坐标

框架对照表本身没有意义，除非先回答两个问题——**它们决定了矩阵里哪几列对你重要**。

### 1.1 坐标一：走路线 A 还是路线 B

工业实现分裂为两条路线（主稿 §2.5）：

- **路线 A：全词表 / top-k logit 反传**——教师在学生轨迹每个位置给出完整（或 top-k）词表分布，KL 直接反传。代表：Qwen3（arXiv:2505.09388 §4.5）、DeepSeek-V4（arXiv:2606.19348 §5.1.2）。
- **路线 B：token 级 KL-as-reward**——把 $\hat{A}_t=\mathrm{sg}\big[\log\frac{\pi_T(y_t\mid\cdot)}{\pi_\theta(y_t\mid\cdot)}\big]$ 当作 advantage 塞进 GRPO/PPO。代表：GLM-5（arXiv:2602.15763 §3.5，Eq. 2，group size $=1$）、MiMo-V2-Flash（arXiv:2601.02780 §4.4 Eq. 9，再与 ORM 奖励线性混合）、Kimi K3（arXiv:2607.24653 §4.1.3 Eq. 15，clip 截断）、Nemotron-Cascade 2（arXiv:2603.19220 §4.4）、TML/Tinker。

两条路线的**教师服务形态、传输格式、损失算子完全不同**，一套实现通吃是不现实的期望【推断】。两家头部厂商的一手经验报告互相矛盾（DeepSeek-V4 §5.1.2 逐字批评路线 B "leads to high variance in gradient estimation and often causes training instability"；Kimi K3 §4.1.3 逐字反证 "we observed no clear advantage in either convergence speed or final performance in our setting"），说明这不是有标准答案的选择题，见 [[15_opd_divergence_and_objective_evolution_analysis]]。

### 1.2 坐标二：教师从哪里取数

生态已收敛出三条路径（主稿 §6.2【推断，由框架实现归纳】）：

| 路径 | 机制 | 适用条件 | 框架实例 |
|---|---|---|---|
| **(a) 教师推理服务化** | 教师作为独立推理服务（vLLM/SGLang）返回 top-k 或全量 logprob | 教师过大或架构异构（slime 官方文档原话） | veRL（独立资源池 + ZMQ）、slime（`--opd-type sglang`）、Tinker（`compute_logprobs` API） |
| **(b) 教师共置训练框架** | 教师在训练框架内做前向 | 师生同构、规模可控 | TRL、slime（`--opd-type megatron`）、NeMo-RL（DTensor） |
| **(c) 传隐藏状态代替 logits** | 只传最后一层 hidden states，学生端经 prediction head 重算 logits | 带宽敏感、走路线 A | KDFlow；DeepSeek-V4 自研（⚠️ 细节部分来自二手分析） |

信号密度-成本谱系随之分层，量级差约 4.8 个数量级，见 [[13_opd_infra_mechanism_analysis]] §4。

---

## 2. 六框架支持矩阵（截至 2026-08，逐项一手核实）

| 框架 | 归属 / 生态 | OPD 路线 | 教师取数方式 | 异步 | 多教师 | 跨词表 | 一手出处 |
|---|---|---|---|---|---|---|---|
| **veRL** | 字节生态开源（verl-project） | **A + B 双路线**（`forward_kl_topk` / `k1`,`k3` + 策略梯度） | 独立教师资源池（`distillation.nnodes`）+ ZMQ top-k logprob 服务 | ✅ 官方 one/two-step-off recipe | ✅ 按样本 metadata（如 `data_source`）路由 | ✗ | https://verl.readthedocs.io/en/latest/algo/opd.html ；异步 recipe https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html |
| **slime** | 智谱 / THUDM（GLM-5 生产栈） | B（advantage 上叠加加权 reverse-KL，与任意 estimator 正交） | SGLang 外部服务 **或** Megatron 直载，二选一 | 框架本身支持异步 RL | 文档未见 | ✗ | https://thudm.github.io/slime/zh/advanced/on-policy-distillation.html ＋ GLM-5 arXiv:2602.15763 §3.6.1 |
| **TRL** | HuggingFace | A（GKDTrainer：`lmbda` / `beta` / `seq_kd`） | 共置 HF 前向；GOLD 另加 vLLM 做学生生成 | ✗ | ✗ | ✅ **GOLDTrainer**（ULD + hybrid loss，唯一开源跨词表 OPD） | https://huggingface.co/docs/trl/gkd_trainer ；https://huggingface.co/docs/trl/main/en/gold_trainer |
| **NeMo-RL** | NVIDIA | A（学生生成 → 教师 logits → KL） | 共置（**仅 DTensor + vLLM，Megatron 路径未支持**——文档原话） | ✗ | ✗ | ✗ | https://docs.nvidia.com/nemo/rl/latest/about/algorithms/on-policy-distillation.html |
| **Tinker** | Thinking Machines（托管服务） | B（advantage $=-\mathrm{RKL}$ + importance sampling 损失） | `compute_logprobs` API（服务端执行） | 托管，用户不可见 | ✅ `on_policy_multi_teacher.py` | ✗ | https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation |
| **KDFlow** | 学术开源（【推断】OPD 综述作者团队，README 署名 Zhang et al.） | A 变体（传隐藏状态、学生端重算 logits） | SGLang 教师 + 共享内存零拷贝 | 训推解耦（教师 SGLang / 学生 FSDP2） | ✅ 多教师路由 | ✅ 跨 tokenizer | https://github.com/songmzhang/KDFlow ；论文 arXiv:2603.01875v3 |
| OpenRLHF | 开源 | **无原生 OPD 特性**（定向检索无果） | — | — | — | — | 见 §4 辨析；arXiv:2604.00626 §7.4/§8.3 |

**读表要点**【推断】：
- **只有 veRL 一家同时提供两条路线**——如果算法路线尚未定死，或者预计要做 A/B 对比实验，这是决定性差异。
- **只有 TRL 一家提供开箱可用的跨词表 OPD**，且只在单机级别。
- **只有 veRL 一家有官方异步 recipe**（Tinker 的异步在托管侧，用户不可控）。
- **多教师在开源侧只做到"路由"，没有一家做到"权重调度"**（§8 Gap 3）。

---

## 3. 逐框架选型要点

### 3.1 veRL —— 当前功能面最全的开源选择

**机制**：两种模式并存——**GKD OPD**（`loss_mode=forward_kl_topk`，教师 top-k logits 直接反传，即路线 A）与 **PG OPD**（`loss_mode=k1`/`k3`，reverse-KL 估计当 reward 走策略梯度，`use_policy_gradient=true`，即路线 B）。配置命名空间 `distillation.*` 齐全：`distillation.enabled`、独立教师资源池 `distillation.nnodes`；`use_task_rewards` 支持把 KL 信号与任务奖励混合——这正好对齐 MiMo 的 Eq. 9 用法（arXiv:2601.02780 §4.4）。rollout 用 vLLM/SGLang，训练用 FSDP/Megatron；教师 logprob 在 rollout 期间逐样本异步计算。

**异步 recipe**（唯一一家有官方实现）：one/two-step-off 调度重叠 rollout / 训练 / 教师取数，文档明言以**牺牲严格 on-policy** 为代价；ZMQ 教师服务提供 top-k logprob；脚本 `recipe/gkd/teacher/start_server.sh`、`recipe/gkd/run_moonlight_dsv3_training.sh`，配置 `recipe/gkd/config/on_policy_distill_trainer.yaml`；报告权重同步优化约 $12\times$ 加速。

**在研方向**（均为已核实的仓库状态，非已合并特性）：
- TCOD RFC——issue #6552，2026-06-01 开，**Open**，把多轮 agent 的时间课程落进 veRL agent 栈；动机是轨迹级 KL 不稳定（arXiv:2604.24005），声称 $+15\%$ 成功率、$-32\%$ 训练时间。
- SDPO——PR #5499，作者 jonhue（即 SDPO 论文 arXiv:2601.20802 一作），2026-03-05 创建，**open 未合并**。
- TCAD recipe——**verl-project/verl-recipe** PR #51（注意：`volcengine/verl-recipe` 不存在，会 404；这是流传的错误组织名），2026-02-26 创建，open，约 4165 行 / 33 文件；TCAD $=$ Confidence-Aligned OPD（教师序列级置信度重加权 + Support-Symmetric Truncation + Coverage-Aware Regularization）。

**适合**：自建大规模多教师 OPD；想保留全词表/top-k 选项；需要异步流水线的团队。**要自己补的**：教师权重调度（W2）、可观测性（W7）、跨词表。

### 3.2 slime —— GLM-5 的生产栈，设计最简

**机制**：OPD 是 advantage 上的一个**正交惩罚项**——advantage 减去加权 reverse-KL，与任意 estimator（GRPO/PPO…）正交叠加。参数极少：`--use-opd`、`--opd-type sglang|megatron`、`--opd-kl-coef`（默认 $1.0$）、`--opd-teacher-load`。教师取数两模式正好覆盖两种现实：**SGLang 外部服务**（教师过大或架构异构）与 **Megatron 直载**（师生同构、追求效率）。

**权威背书**：GLM-5 报告 §3.6.1 确认 slime 在统一栈内支持 OPD——"reasoning RL, general RL, agentic RL, and on-policy distillation, all within a unified training stack"（arXiv:2602.15763）。⚠️ 注意：**slime 的 README 本身并未把蒸馏列为核心特性**，引用"slime 支持 OPD"时应引 GLM-5 §3.6.1 或官方文档页，而不是 README。

**社区讨论中暴露的真实问题**（可作为落地时的踩坑预告）：issue #1068（open，2025-12-09）质疑为何用 reward-based loss 而非直接 KL、为何不经 rollout 传教师 logits；issue #1449（closed，2026-01-18）讨论 advantage 是否应 detach，讨论中直接引 MiMo-V2-Flash 技报。

**适合**：已在用 slime/Megatron 栈、路线 B 明确的团队。**要自己查的**：异步窗口与截断 IS 钩子（[[13_opd_infra_mechanism_analysis]] W3）；多教师文档未见。架构细节见 [[slime/index]]。

### 3.3 TRL —— 单机事实标准 + 唯一开源跨词表实现

**GKDTrainer**（同词表 OPD）：`lmbda` 控制 on-policy 数据比例（$0=$ 全离线、$1=$ 全 on-policy）、`beta` 控制 JSD 插值（$0\approx$ forward KL、$1\approx$ reverse KL）、`seq_kd`（序列级 KD $=$ 教师生成 + 监督 FT）、`teacher_model_name_or_path`。这套参数化直接来自 GKD 论文（arXiv:2306.13649）。

**GOLDTrainer**（跨词表 OPD，**目前唯一开源实现**）：ULD 的扩展——增量解码对齐师生 token 边界、按可见文本分组合并概率（首位置边际分布 $\times$ 后续实际 token 的标量条件概率，链式法则，故意不归一化以配合 ULD 的排序 $+$ L1）；`uld_use_hybrid_loss` 对词表交集 token 用精确 JSD、未匹配 token 回退 ULD；继承 GKD 的 on/off-policy 调度；支持 vLLM colocate/server 学生生成与 sleep mode；可做 VLM-to-VLM 跨架构蒸馏。

**两个必须知道的注意事项**：
1. **GKD/GOLD 均已移入 `trl.experimental` 命名空间**（v1.5.1 起源码在 `trl/experimental/gkd/gkd_trainer.py`；GOLD 在 `trl/experimental/gold/gold_trainer.py`、`gold_config.py`，示例 `examples/scripts/gold.py`）——API 稳定性没有保证。
2. **跨词表的正确性陷阱**：TRL issue #4562（2025-11-22 开，**closed as not planned**）报告 GKD/MiniLLM trainer 把学生 tokenizer 的 token id 直接喂给教师，教师按自己的词表误读，**训练信号静默损坏**。正确做法是保留文本、用教师 tokenizer 重切再算 logprob（GOLD 正是这么做的）。**跨词表时这不是优化项而是正确性前提**。

**适合**：单机/中小规模；师生词表不同的场景。**不适合**：大规模分布式。

### 3.4 NeMo-RL —— NVIDIA 官方路径，限制明确

端到端 OPD：学生生成 $\to$ 教师给 logits $\to$ 最小化 KL（路线 A）。单机示例 `uv run python examples/run_distillation.py`（Qwen3-1.7B-Base 学生 / Qwen3-4B 教师）。可接 **NeMo Gym** 做多步 rollout——学生经 OpenAI 兼容 HTTP 服务暴露给环境，从而支持 agent 场景蒸馏。官方公告见 GitHub Discussion #1445（2025-10-28）。

**当前限制是文档原话**：只支持 **DTensor 训练 + vLLM 生成，Megatron 路径尚不支持**。对已经把大模型训练压在 Megatron 上的团队，这是硬门槛。

**适合**：NeMo 生态既有用户；想用 NeMo Gym 做 agent 蒸馏的团队。

### 3.5 Tinker —— 唯一把 OPD 做成托管服务原语的产品

教师打分、采样、损失全部 API 化：`compute_logprobs` 取教师 logprob，训练侧令 per-token advantage $=-\mathrm{RKL}$ 并调用 importance sampling 损失——即 TML 博客所说的"在 RL 脚本上改一行"。cookbook 覆盖面在六家里最广：`on_policy_distillation.py`、`off_policy_reasoning.py`、`on_policy_multi_teacher.py`（多教师 + 多数据集批采样拼接）、`harbor_multiturn.py` / `on_policy_distillation_harbor_multi_turn.py`（多轮 agent 环境、零任务奖励纯教师 KL 信号、**环境 token 掩蔽**——"only the student's generated tokens contribute to the loss"），另有 SDFT 配方。推荐值如 LoRA lr $1\mathrm{e}{-4}$、max tokens 16384 直接写在 cookbook 里。

**代价是模型清单受平台约束**：2026-06-12 已发生过师生模型对的下架替换（`Qwen3-8B-Base` $\to$ `Qwen3.5-9B-Base`、`Qwen3-32B` $\to$ `Qwen3.6-27B`，见 tinker-docs model-deprecations 页；Cookbook 0.4.2 changelog 侧证"Migrated all tutorial series and the cookbook recipes off the models retired on June 12, 2026"）。**做长期可复现实验时这是实质风险**。

**适合**：不想自建 infra、要快速验证 OPD 是否对自己的任务有效的团队。

### 3.6 KDFlow —— 把带宽问题做成核心卖点的专用蒸馏框架

架构：**教师推理交给 SGLang、学生训练交给 FSDP2**，避免同构后端的结构性失配。关键优化：不传全量 logits（其配置下大模型约 $160\ \mathrm{GB}$ 级），改传教师紧凑 hidden states、学生侧重算完整 logit 分布，共享内存零拷贝；**数学上与标准 KD 等价**。覆盖 off-policy / on-policy / cross-tokenizer 蒸馏，内置 FKL、RKL、JSD、TVD；多教师路由（`--multi_teacher_config`）、EMA 自蒸馏（`--use_ema_teacher`）、`--chunked_loss_size` 等。论文实测对 TRL / MS-SWIFT / ROLL 提速 $1.44\times$–$6.36\times$，对 MoE 教师尤其有效（arXiv:2603.01875v3，v1 2026-03-02 / v3 2026-07-17）。

**适合**：蒸馏本身就是主业务、带宽敏感、愿意接受较新代码库的团队；也是"自研隐藏状态传输方案"最直接的参考实现。

---

## 4. OpenRLHF：为什么"可用"与"原生支持"必须分开说

这是本主题里最容易产生分歧的一条，两种说法都能找到依据：

- **本调研的定向检索结论**：找不到 OpenRLHF 原生 OPD 特性的证据——没有官方 recipe、文档页或 issue。⚠️ 注意这**不等于证明不存在**，只是检索无果。
- **上游综述的说法**：Song & Zheng（arXiv:2604.00626 §7.4/§8.3）把 OpenRLHF 列为 OPD 常用框架，逐字为 "much of the existing RLHF infrastructure applies directly…why frameworks like OpenRLHF serve both communities"，并列出 "the frameworks most widely adopted for OPD (OpenRLHF, veRL, SLIME, verl-pipeline)"。

**两说不矛盾**，正确的表述是"**可用而非原生支持**"：路线 B 与 RLHF 共享同一条 rollout–score–update 回路，把 RM 打分换成教师 logprob 即可复用，无需框架提供任何 OPD 专门特性。这条辨析本身有超出 OpenRLHF 的意义——它解释了为什么 OPD 能在一年内被"全部主流 RL 框架内建"：**大部分框架的"内建"其实只是把一条已有通道换了个信号源**，真正需要新代码的是路线 A（新增蒸馏损失分支）与教师服务化。选型时应据此追问：某框架宣称的"支持 OPD"到底是提供了教师取数与调度基建，还是只提供了一个 advantage 替换点【推断】。

---

## 5. 周边与基线框架

- **EasyOPD**（arXiv:2607.11012v1，2026-07-13，Jie Sun / Mao Zheng / Mingyang Song 等 10 人，与 arXiv:2604.00626 综述同组）：**基于 veRL** 构建的统一 OPD 框架，实例化**三个代表性 OPD 设置**——cross-tokenizer OPD、on-policy self-distillation、step-wise OPD。⚠️ 注意：某些搜索摘要称其覆盖 "10+ methods"，与 v1 abstract 的 "three OPD settings" 不符，**以 abs 页为准**。其存在本身是"OPD 已进入主流 RL 框架"的旁证——学术侧的新框架直接构建在 veRL 之上而不是从头写。
- **MS-SWIFT 与 ROLL**：在 KDFlow 论文中作为对比基线出现（arXiv:2603.01875），说明二者亦有蒸馏训练路径；⚠️ 其 OPD 细节本调研未核实，不做评价。
- **NeMo-Aligner（旧线）**：其蒸馏能力是离线 KD（SFT 数据高效 KD），**不是 OPD**；OPD 能力在新的 NeMo-RL 中。⚠️ 该判断依据为 NVIDIA 开发者博客（搜索级）。

---

## 6. 生产系统的自研层（开源框架之外）

技术报告披露的自研 infra 代表当前上限，也是开源生态的 roadmap 参照：

| 系统 | 自研点 | 出处 |
|---|---|---|
| **DeepSeek-V4** | 全词表 OPD 的教师调度（教师数量"实际无上界"、万亿参数权重 offload、中心化缓冲只存最后层 hidden states + 训练时经 prediction head 即时重建 logits、minibatch 内按教师身份排序样本）；可抢占容错的 rollout 服务**同时服务 RL 与 OPD**；FP4（MXFP4）QAT 下的师生数值一致性 | arXiv:2606.19348 §5.1.2 / §5.2.1–5.2.3（调度实现细节部分经 Labonne 二手分析与 arXiv:2604.00626 §8.3 转述交叉印证 ⚠️；TileLang 精确 KL 内核为**单一二手来源** ⚠️） |
| **Kimi K3** | partial rollout 跨迭代续跑（"an individual long-horizon trajectory naturally spans multiple iterations"）+ per-token 正则消化陈旧性；外置 KV-cache 保留；可续跑 microVM 沙箱；MXFP4/MXFP8 QAT 贯穿后训练 | arXiv:2607.24653 §4.1.1–4.1.4 |
| **Nemotron 3 Ultra** | MOPD $=$ **Multi-teacher** On-Policy Distillation（摘要逐字）；十余个领域专家教师、两轮师生共进化（RLVR 后的学生反过来当下一轮教师，Fig. 10）；rollout / teacher-scoring / learner 三类 worker 的异步流水线 | arXiv:2606.15007 §3.3–3.3.2。⚠️ 其中"异步 behavior/proximal 策略解耦"本轮**未能独立核实**（arXiv HTML 与 ar5iv 均在 §3.3.1 之后截断），标**待核** |
| **Nemotron-Cascade 2** | MOPD $=$ **Multi-domain** On-Policy Distillation（§4.4 节标题逐字，见 §10 更正 2）；教师取自内部 Cascade RL 各阶段的最强 checkpoint，"without introducing external models"；路线 B（"computed only on the student-sampled token rather than over the full vocabulary"）；train–inference 失配用**截断重要性加权** $\varepsilon_{low}=0.5$、$\varepsilon_{high}=2.0$（已核实为真） | arXiv:2603.19220 §4.4 |
| **GLM-5** | slime 统一栈内做 OPD；教师 logits 经推理引擎获取；group size 降为 1 直接省 rollout 采样量 | arXiv:2602.15763 §3.5 / §3.6.1 |

**一条容易踩的术语坑**：**MOPD 这个缩写在 NVIDIA 两份报告里指两件不同的事**——Nemotron 3 Ultra 是 multi-**teacher**，Nemotron-Cascade 2 是 multi-**domain**。引用"NVIDIA 用 MOPD"时必须指明是哪一份报告、哪一节，否则会把"多个教师模型"和"多个领域的信号"混为一谈（详见 §10 更正 2）。

**对开源生态的启示**【推断】：这张表里的每一行都对应 [[13_opd_infra_mechanism_analysis]] 的一项 W 工作，而其中最难的两项——**教师权重调度（W2）与数值一致性/QAT（W5）——在开源侧完全没有对应物**。这就是下一节 Gap 分析的来源。

---

## 7. 按场景的选型建议

【推断，基于 §2–§3 的已核实功能面】

| 场景 | 建议 | 需要自己补的部分 |
|---|---|---|
| 大规模多教师、自建集群、可能走全词表 | **veRL**（唯一双路线 + 官方异步 recipe），参照 DeepSeek-V4 §5.2.2 的调度思路补教师权重管理 | W2 教师调度、W7 可观测性、跨词表 |
| 已有 Megatron/slime 栈、明确走 KL-as-reward | **slime**（GLM-5 同款生产栈） | W3 异步窗口与截断 IS 钩子自查、多教师路由 |
| 师生词表不同 | **TRL GOLDTrainer**（唯一现成实现）；大规模则需自行把 GOLD 思路移植进 RL 框架 | 分布式扩展；注意 issue #4562 的正确性陷阱 |
| 多轮 agent 蒸馏 | **Tinker harbor**（托管）或 **NeMo-RL + NeMo Gym**；自建走 veRL + TCOD RFC 方向 | 自建路线下 W4 全部要自己做（环境 token 掩蔽、KV-cache 保留、可续跑沙箱） |
| 蒸馏是主业务、带宽敏感 | **KDFlow** 的隐藏状态传输方案，或自研等价物 | 与自有训练栈的集成 |
| 不想自建 infra、先验证有效性 | **Tinker** | 接受模型清单受平台约束（2026-06 已发生过下架替换） |
| 单机小规模、快速起步 | **TRL GKDTrainer** | 无法扩展到分布式 |

---

## 8. 六条生态 Gap（Infra 团队的机会清单）

1. **跨词表 OPD 在 RL 框架中缺位**。GOLD 只在 TRL（单机级）；veRL / slime / NeMo-RL 均不支持——**"任意教师"的 OPD 在大规模框架里还做不到**。而这恰恰是"用外部开放权重模型当教师"这一整类需求的前提（教师来源第⑤类，主稿 §5.12）。
2. **agent OPD 的系统支持不成熟**。轨迹级 KL 不稳定的治理（时间课程、KL 感知终止）都停留在论文/RFC 阶段：TCOD 是论文（arXiv:2604.24005）、veRL #6552 是 open 的 RFC、KAT 的 KL 感知提前终止（arXiv:2606.09471）没有任何框架内建。
3. **教师调度无开源实现**。DeepSeek-V4 式的"unbounded teachers + offload"没有开源对应物；**多教师在开源侧只到"路由"层面，未到"权重调度"层面**——veRL/KDFlow/Tinker 的多教师都假设教师能同时驻留。
4. **可观测性空白**。[[13_opd_infra_mechanism_analysis]] W7 列出的失败模式监控（KL 分布而非均值、教师熵分层、rock token 驻留集合、pass@k 曲线）没有任何框架内建，全靠各家自研 callback。这是六条里**收益/成本比最高**的一条。
5. **异构硬件**。现有实现深度绑定 CUDA 生态（vLLM / SGLang / FSDP2 / Megatron / TileLang）。向其他加速器移植时，**教师 logprob 服务（W1）与训推数值一致性（W5）是两个改造重心**【推断】。
6. **KL 计算算子**。全词表 KL 在大词表（$128\mathrm{K}+$）下值得专用 kernel——DeepSeek-V4 的 TileLang 内核（⚠️ 单一二手来源）是唯一已知先例，开源侧无对应物。

---

## 9. 预算分配的三段模式（选框架之前先定预算结构）

多个工业管线（Qwen3 / DeepSeek-V4 / R1 系）呈现同一模式（arXiv:2604.00626 §8.3 的归纳，与 §6 的一手证据相容）：

1. **off-policy 预热拿走预算大头**——教师产数据做 SFT，廉价、高带宽，把学生分布拉到教师附近；这同时降低后续 on-policy 阶段的无效 rollout 比例。
2. **on-policy logit 蒸馏拿走 on-policy 份额的大头**——在学生自身轨迹上关闭分布差距。
3. **reward-guided 精修只留最后一小份**——稀疏结果奖励推学生越过教师天花板。

原则一句话（逐字）："front-load cheap, high-bandwidth learning and reserve expensive on-policy compute for the final quality push"。

**对框架选型与容量规划的直接含义**【推断】：三段的负载画像完全不同——**阶段①没有 rollout 引擎**（纯 SFT，可以在便宜的机型上跑）、**阶段②是教师带宽峰值**（决定 §1.2 教师取数路径与集群拓扑）、**阶段③ verifier/RM 才进场**（需要沙箱与判分服务，见 [[11_rl_sandbox_design_analysis]]）。按峰值给三段统一配一套资源会严重浪费；反过来，如果框架只覆盖其中一段（例如 TRL 只覆盖①②的单机版本），就必须准备阶段切换时的迁移成本。

---

## 10. 本页相对上游综述稿的独立核验更正

| # | 上游稿表述 | 本轮核验结论 | 本页处理 |
|---|---|---|---|
| 1 | 把 MiMo 的截断（"IcePop 式"）与 K3 的师生 log-ratio clip、Nemotron 的截断 IS 并列为同一类"off-policyness 修正" | MiMo 正文**未出现 "IcePop"**（该词仅见于参考文献标题），原文为 "Following Zhao et al. (2025)"；且 **Eq. 8 的截断作用于训推比 $\pi_\theta/\mu_\theta$**（训练策略 vs 采样策略），属**训推不一致（TIM）修正**，与 OPD 的师生 KL 是两件不同的事 | 本页不复述该并列；两类截断的完整辨析见 [[13_opd_infra_mechanism_analysis]] W3-a / W3-b 与 [[26_tim_causal_chain_analysis]] |
| 2 | Nemotron-Cascade 2 的 MOPD 被当作"多教师 OPD"引用 | §4.4 节标题与摘要均为 "**Multi-domain** On-Policy Distillation"——是**多领域**，不是多教师；其截断 IS 区间 $\varepsilon_{low}=0.5$、$\varepsilon_{high}=2.0$ **已核实为真**（arXiv:2603.19220 §4.4）。同一缩写在 Nemotron 3 Ultra 摘要中则确为 "Multi-**teacher** On-Policy Distillation"（arXiv:2606.15007） | §6 表格按各自报告的原义分别标注，并单列术语坑提示 |
| 3 | Nemotron 3 Ultra 的"异步 behavior/proximal 策略解耦"作为一手事实列入自研层 | **本轮未能独立核实**——arXiv HTML 与 ar5iv 均在 §3.3.1 之后截断 | §6 表格中标 ⚠️ **待核** |

另有两条一手引文本轮已从 PDF 原文复核，可放心作一手引用：

- **DeepSeek-V4 §5.1.2（逐字）**："it leads to high variance in gradient estimation and often causes training instability. Therefore, we adopt full-vocabulary logit distillation in our OPD."
- **Kimi K3 §4.1.3（逐字，两句）**："This dense reward signal seamlessly integrates into our RL framework, naturally enabling infrastructure-level optimizations such as partial rollout training for long-horizon tasks." / "we also experimented with more fine-grained top-k distillation objectives, we observed no clear advantage in either convergence speed or final performance in our setting."

后一句的重要性在选型语境下值得单独点出：**K3 选路线 B 的公开理由不是"省带宽"，而是"能继承 RL 框架里已有的一切长程优化"**——这把框架选型和算法路线选择绑成了同一个决策【推断】。

---

## 参考定位符汇总

**框架文档**：veRL OPD https://verl.readthedocs.io/en/latest/algo/opd.html ；veRL 异步 recipe https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html ；slime OPD https://thudm.github.io/slime/zh/advanced/on-policy-distillation.html ；TRL GKDTrainer https://huggingface.co/docs/trl/gkd_trainer ；TRL GOLDTrainer https://huggingface.co/docs/trl/main/en/gold_trainer ；NeMo-RL OPD https://docs.nvidia.com/nemo/rl/latest/about/algorithms/on-policy-distillation.html ；Tinker cookbook https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation ；KDFlow https://github.com/songmzhang/KDFlow

**仓库级条目（已核实存在性与状态）**：veRL issue #6552（TCOD RFC，open）；veRL PR #5499（SDPO，open）；verl-project/verl-recipe PR #51（TCAD，open）；slime issue #1068（open）/ #1449（closed）；TRL issue #4562（closed as not planned）；NeMo-RL Discussion #1445（2025-10-28 官方公告）。

**论文与技术报告**：KDFlow arXiv:2603.01875v3；EasyOPD arXiv:2607.11012v1；GKD arXiv:2306.13649；ULD arXiv:2402.12030；DeepSeek-V4 arXiv:2606.19348v1 §5.1.2/§5.2.1–5.2.3；Kimi K3 arXiv:2607.24653v2 §4.1.1–4.1.4；Nemotron 3 Ultra arXiv:2606.15007 §3.3；Nemotron-Cascade 2 arXiv:2603.19220 §4.4；GLM-5 arXiv:2602.15763v2 §3.5/§3.6.1；MiMo-V2-Flash arXiv:2601.02780v2 §4.4；Qwen3 arXiv:2505.09388v1 §4.5；Song & Zheng 综述 arXiv:2604.00626v4 §7.4/§8.3；TCOD arXiv:2604.24005；KAT arXiv:2606.09471；token 级偏差-方差 arXiv:2603.25562。

---

## Related Pages

- [[13_opd_infra_mechanism_analysis]] —— OPD 基础设施机制与八项工作清单 W1–W8（本页的机制前置）
- [[30_rl_framework_comparison]] —— 通用 RL 框架横评（本页只覆盖 OPD 维度）
- [[02_engineering/04_posttrain_frameworks/index]] —— 后训练框架索引
- [[verl/index]] —— veRL 框架索引（§3.1 的展开）
- [[slime/index]] —— slime 软件架构与实现分析（§3.2 的展开）
- [[14_on_policy_distillation_analysis]] —— OPD 算法总览
- [[15_opd_divergence_and_objective_evolution_analysis]] —— 散度与目标函数演化（§1.1 路线之争的算法侧）
- [[32_opd_industrial_landscape_analysis]] —— 厂商采用格局与教师来源（§6 自研层的产业背景）
- [[33_opd_effectiveness_and_failure_modes_analysis]] —— 失败模式（Gap 4 可观测性的需求来源）
- [[26_tim_causal_chain_analysis]] —— 训推不一致因果链（§10 更正 1 的机制页）
- [[12_rl_infra_efficiency_analysis]] —— RL 训练回路效率工程
- [[11_rl_sandbox_design_analysis]] —— 沙箱设计（§9 阶段③与 agent 场景）
