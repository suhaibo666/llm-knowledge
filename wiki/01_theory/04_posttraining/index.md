# LLM 后训练算法理论 — 目录索引

> 覆盖 RLHF、DPO、GRPO/PPO 家族、Reasoning RL 算法演进、Agentic RL、on/off-policy 与 staleness、
> 训推一致性(TIM)与对齐安全。
> 最后更新: 2026-07-31(kb-reorg P5 收尾:三域整合完成,`03_posttraining/` 纵向学习域已解散)

---

阅读路线入口(跨域纯导读页,正文全部归属本表 + `04_posttrain_frameworks`/`verl/`/`moonshot_kimi`
三处功能树,不计入下表):[[courses/posttraining_frontier|LLM 后训练前沿阅读课程]] —— D01→D12 顺序
索引 + 六级能力门槛,原 `03_posttraining/` 域(D00–D12)已随 kb-reorg P5 逐任务解散归位于此。

## 段位与阅读顺序(kb-reorg P5 Task 8,2026-07-31)

文件名两位数字前缀 = 段位:段 0(01-09)入门导览;段 1(10-19)核心机制主线,按学习序排列——InstructGPT
三步 RLHF 基础 → PPO-Clip → DPO 直接偏好优化 → Reasoning RL 算法演进权威页(GRPO/DAPO/GSPO/SAO 谱系
统一收口);段 2(20-29)深潜/专题——GRPO/DAPO/GSPO/RLOO 四篇论文档案(公式与动机已收缩指向段 1 权威页,
仅保留元数据/原始实验数字/消融)、Agentic RL、on/off-policy staleness、TIM 因果链、VAPO、RLHF 多论文
综合档案、Kimi K1.5 案例档案;段 3(30-39)方法论/对照——DPO 家族横向对比、Reward Hacking 防御方法论。
下表按段位排列,与下方分主题小节互为索引:

| 段 | 页面 | 一句话 |
|---|------|------|
| 0 | [[01_posttraining_frontier_map_analysis]] | 后训练前沿全景地图:四组张力组织全局坐标 |
| 1 | [[10_instructgpt_rlhf_analysis]] | InstructGPT 三步 RLHF 基础 |
| 1 | [[11_ppo_analysis]] | PPO-Clip、surrogate loss、GAE |
| 1 | [[12_dpo_analysis]] | DPO 直接偏好优化 |
| 1 | [[13_reasoning_rl_algorithm_evolution_analysis]] | Reasoning RL 算法演进权威页(GRPO/DAPO/GSPO/SAO 谱系) |
| 2 | [[20_grpo_analysis]] | GRPO 论文档案(元数据/实验数字) |
| 2 | [[21_dapo_analysis]] | DAPO 论文档案 |
| 2 | [[22_gspo_analysis]] | GSPO 论文档案 |
| 2 | [[23_rloo_analysis]] | RLOO 论文档案 |
| 2 | [[24_agentic_rl_algorithm_analysis]] | Agentic RL 算法与环境专题 |
| 2 | [[25_on_policy_off_policy_staleness_analysis]] | On/off-policy 与 staleness 专题 |
| 2 | [[26_tim_causal_chain_analysis]] | TIM 因果链深潜 |
| 2 | [[27_vapo_analysis]] | VAPO 论文档案 |
| 2 | [[28_rlhf_foundations_analysis]] | RLHF 基础方法多论文综合档案 |
| 2 | [[29_kimi_k1_5_analysis]] | Kimi K1.5 案例档案 |
| 3 | [[30_preference_optimization_analysis]] | DPO 家族对照(IPO/SimPO/ORPO/KTO/MODPO) |
| 3 | [[31_reward_hacking_defense_analysis]] | Reward Hacking 防御方法论 |

## 页面列表

### 核心方法

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[10_instructgpt_rlhf_analysis]] | InstructGPT (2203.02155) | 三步 RLHF (SFT→RM→PPO), KL 惩罚, 1.3B > 175B |
| [[11_ppo_analysis]] | PPO (1707.06347) | PPO-Clip, surrogate loss, GAE 优势估计 |
| [[12_dpo_analysis]] | DPO (2305.18290) | 直接偏好优化, 闭式策略-奖励关系, 无需采样 |

### DPO 系列变体

| 页面 | 核心主题 |
|------|---------|
| [[30_preference_optimization_analysis]] | DPO 家族对比: IPO, SimPO, ORPO, KTO, MODPO |

### GRPO 系列

> **定位**（2026-07-31 kb-reorg P5）：[[13_reasoning_rl_algorithm_evolution_analysis|D02 Reasoning RL 算法演进]] 是 GRPO/DAPO/Dr.GRPO/GSPO/SAO 公式演进、系统约束与跨算法对照的统一权威页；下表各篇瘦身为对应论文的元数据、原始实验数字与消融档案，重叠的公式推导/动机叙述已收缩为指向 D02 的一句话。

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[20_grpo_analysis]] | DeepSeek-R1 (2501.12948) | 组相对优势, 无价值函数, 纯 RL 推理 |
| [[21_dapo_analysis]] | DAPO (2503.14476) | 解耦裁剪, 动态采样, AIME 50 |
| [[22_gspo_analysis]] | GSPO (2507.18071) | 序列级重要性比, 修复 GRPO token 级不稳定 |
| [[23_rloo_analysis]] | RLOO (2402.14740) | REINFORCE + leave-one-out baseline |

### 后训练前沿整合（kb-reorg P5 迁入）

> 2026-07-31 从 `wiki/03_posttraining/`（原 D01/D02/D03/D04）迁入,是后训练三域整合的一部分。

| 页面 | 核心主题 |
|------|---------|
| [[01_posttraining_frontier_map_analysis]] | 后训练前沿全景地图:优化粒度、on-policy/freshness、训练—推理一致性、Agentic 环境四组张力 |
| [[13_reasoning_rl_algorithm_evolution_analysis]] | Reasoning RL 算法演进(**GRPO 系列权威页**):统计单位、有效样本分布、行为策略比率、训练—推理一致性四类修正谱系 |
| [[24_agentic_rl_algorithm_analysis]] | Agentic RL 算法与环境:多轮 trajectory、工具调用、reward 与 credit assignment |
| [[25_on_policy_off_policy_staleness_analysis]] | On-policy、Off-policy 与 Staleness:policy lag、correction 方案与 TIM 的严格区分 |

### 训推一致性（TIM）与 RL 稳定性

> 2026-07 新建簇。覆盖「kernel 非确定性 → logprob 偏差 → 重要性比方差放大 → 训练崩溃」这条因果链，及其算法侧与系统侧修法。与 [[07_training_reliability/index]] 问题 2 直接接壤（后者讲系统侧上游，本簇讲中间两环与算法侧修法）。

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[26_tim_causal_chain_analysis]] | Diagnosing TIM (2605.14220)、FP16 (2510.26788)、Beyond Precision (2602.01826)、TBIK (2511.17826)、M2PO (2510.01161)、VCPO (2602.17616)、TIS (OpenReview 8MHqvb4lK9)、TRM (2512.23075)、ALP (2603.19470)、MIPU (2606.29526)、Qwen (2512.01374) 等 | 四环因果链；VeXact 零 mismatch 基线；recomputation 与 bypass 两种崩溃形态；前兆指标的领先性与盲区；系统侧/算法侧修法全谱与确定性税；三处根本分歧 |

**待建页面**

- `moe_routing_replay_analysis` — R2 / R3 / PR² / RSPO 谱系，MoE 路由漂移的机理与系统侧利用（ReLibra / ForeMoE）。相关 raw 已列入 `docs/research/INGEST_MANIFEST_block1_tim.md`。

### 高级方法

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[27_vapo_analysis]] | VAPO (2504.05118) | 基于价值模型的 RL, AIME 60.4 |
| [[28_rlhf_foundations_analysis]] | 多篇综合 | ReMax, Weak-to-Strong, RM Overoptimization, RigorLLM |
| [[29_kimi_k1_5_analysis]] | Kimi K1.5 | 长上下文 RL 推理训练 |

### 对齐安全

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[31_reward_hacking_defense_analysis]] | Anthropic 2025-11 + Claude 4.5 Model Card | Reward Hacking 四层防御（环境/penalty/inoculation prompting/agentic safety），misalignment 泛化机制 |

---

## Knowledge Gaps

> 本节记录**已确认在公开文献中无一手来源**的问题。按 CLAUDE.md 的 Query Workflow，这些缺口不应用推测填补；若日后出现相关工作，应先入 `raw/` 再补页。
> 2026-07-25 首次建立，来源为 [[26_tim_causal_chain_analysis]] 的系统性检索。

| # | 缺口 | 现状与最接近的替代证据 |
|---|------|----------------------|
| 1 | **TIM 致崩的校准阈值** | 不存在通用阈值。最接近的三条均为特定配置下的经验观测：FP8-RL 的 KL>5（Qwen3-30B-A3B，step≈700）；Miles 博客的 TIS clip 上界 2.0 崩 / 1.5 不崩；M2PO 的 $\tau_{M_2}=0.04$（超参选择，非实测崩溃点） |
| 2 | **重要性比的尾部刻画** | 无任何工作给出尾指数 α、Hill 估计、Pareto $\hat k$、分位数表或峰度。全部用二阶矩类聚合量（$M_2$ / ESS / CV）替代。唯一真实引擎失配下的 ESS 绝对值来自 AIS：0.95 → 0.70 |
| 3 | **逐位置 TIM 增长曲线** | 无 16k/32k 长 CoT 上按 token 位置的 $\lvert\delta_t\rvert$ 实证。只能用 FP16 论文的序列级斜率 + Beyond Precision 的 $O(T^2)$ 界拼接 |
| 4 | **极端 token 的出现频率** | Diagnosing TIM 证明了 $\lvert\delta_t\rvert$ 近 1.0 的 token 存在并给了一个 argmax flip 实例，但**没有 rate** |
| 5 | **重尾 → 熵坍塌的因果** | OPEFO 未提 TIM；TIM 一系未连熵动力学。唯一桥梁是 M2PO Fig.4(b) 的横截面相关性（9000 万 token） |
| 6 | **RL 闭环的确定性税** | 三篇系统侧论文（TBIK / LLM-42 / Bit-Exact）均只测推理侧，无人把 rollout 侧 22–63% 开销折算进 RL 总训练时间 |
| 7 | **VeXact 自身的吞吐开销** | Diagnosing TIM 全篇无 vs vLLM 的 benchmark，只有定性的 "retains reasonable throughput" |
| 8 | **RL 阶段的 MoE 路由坍塌** | 常规意义的 routing collapse（token 挤向少数专家）在 RL 阶段尚未形成独立研究主题。最接近的 RSPO (2510.23027) 讲的是 router shift → reward collapse，语义不同 |
| 9 | **vLLM logprobs/logits 语义一致性 RFC** | 2026 Q2 RL roadmap（issue #41733）链到的 #37737 实为流式 tool_call logprobs 缺失的 bug 报告，已 closed as not planned。**无专门 RFC** |

---

## 关联域

- [[courses/posttraining_frontier]] — LLM 后训练前沿阅读课程(D01→D12 顺序索引 + 六级能力门槛)
- [[07_training_reliability/index]] — 万卡训练确定性与可靠性（问题 2「训推数值不一致 / batch 不变性」与本域 TIM 簇直接接壤）
- [[04_posttrain_frameworks/index]] — 后训练框架与 RL Infra（verl / sandbox / 效率）
- [[02_pretraining/index]] — 预训练技术（优化器、低精度训练；[[low_precision_training_analysis]] 与 [[RL_Training_Inference_Precision_Analysis]] 现居此处）
- [[02_train_frameworks/megatron-lm/index]] — 分布式训练基础设施
- [[01_ai_frameworks/index]] — torch.compile 与图编译优化

> [!deprecated] 以下三条为旧目录结构下的链接，当前已无对应页面，保留以备追溯，请使用上方新链接：`[[02_training/index]]`、`[[06_infra/megatron-lm/index]]`、`[[torch_compile/index]]`
