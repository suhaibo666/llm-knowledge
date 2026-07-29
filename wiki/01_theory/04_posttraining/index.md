# LLM 对齐与偏好优化 — 目录索引

> 覆盖 RLHF、DPO、GRPO、PPO 等对齐方法及相关前置研究
> 最后更新: 2026-07-25

---

## 页面列表

### 核心方法

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[instructgpt_rlhf_analysis]] | InstructGPT (2203.02155) | 三步 RLHF (SFT→RM→PPO), KL 惩罚, 1.3B > 175B |
| [[ppo_analysis]] | PPO (1707.06347) | PPO-Clip, surrogate loss, GAE 优势估计 |
| [[dpo_analysis]] | DPO (2305.18290) | 直接偏好优化, 闭式策略-奖励关系, 无需采样 |

### DPO 系列变体

| 页面 | 核心主题 |
|------|---------|
| [[preference_optimization_analysis]] | DPO 家族对比: IPO, SimPO, ORPO, KTO, MODPO |

### GRPO 系列

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[grpo_analysis]] | DeepSeek-R1 (2501.12948) | 组相对优势, 无价值函数, 纯 RL 推理 |
| [[dapo_analysis]] | DAPO (2503.14476) | 解耦裁剪, 动态采样, AIME 50 |
| [[gspo_analysis]] | GSPO (2507.18071) | 序列级重要性比, 修复 GRPO token 级不稳定 |
| [[rloo_analysis]] | RLOO (2402.14740) | REINFORCE + leave-one-out baseline |

### 训推一致性（TIM）与 RL 稳定性

> 2026-07 新建簇。覆盖「kernel 非确定性 → logprob 偏差 → 重要性比方差放大 → 训练崩溃」这条因果链，及其算法侧与系统侧修法。与 [[07_training_reliability/index]] 问题 2 直接接壤（后者讲系统侧上游，本簇讲中间两环与算法侧修法）。

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[tim_causal_chain_analysis]] | Diagnosing TIM (2605.14220)、FP16 (2510.26788)、Beyond Precision (2602.01826)、TBIK (2511.17826)、M2PO (2510.01161)、VCPO (2602.17616)、TIS (OpenReview 8MHqvb4lK9)、TRM (2512.23075)、ALP (2603.19470)、MIPU (2606.29526)、Qwen (2512.01374) 等 | 四环因果链；VeXact 零 mismatch 基线；recomputation 与 bypass 两种崩溃形态；前兆指标的领先性与盲区；系统侧/算法侧修法全谱与确定性税；三处根本分歧 |

**待建页面**

- `moe_routing_replay_analysis` — R2 / R3 / PR² / RSPO 谱系，MoE 路由漂移的机理与系统侧利用（ReLibra / ForeMoE）。相关 raw 已列入 `raw/_ingest/INGEST_MANIFEST_block1_tim.md`。

### 高级方法

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[vapo_analysis]] | VAPO (2504.05118) | 基于价值模型的 RL, AIME 60.4 |
| [[rlhf_foundations_analysis]] | 多篇综合 | ReMax, Weak-to-Strong, RM Overoptimization, RigorLLM |
| [[RL_PPO_Loss_and_GRPO_Analysis]] | 源码分析 | PPO Loss 与 GRPO 的代码级对比 |
| [[kimi_k1.5_analysis]] | Kimi K1.5 | 长上下文 RL 推理训练 |

### 对齐安全

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[reward_hacking_defense_analysis]] | Anthropic 2025-11 + Claude 4.5 Model Card | Reward Hacking 四层防御（环境/penalty/inoculation prompting/agentic safety），misalignment 泛化机制 |

---

## Knowledge Gaps

> 本节记录**已确认在公开文献中无一手来源**的问题。按 CLAUDE.md 的 Query Workflow，这些缺口不应用推测填补；若日后出现相关工作，应先入 `raw/` 再补页。
> 2026-07-25 首次建立，来源为 [[tim_causal_chain_analysis]] 的系统性检索。

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

- [[07_training_reliability/index]] — 万卡训练确定性与可靠性（问题 2「训推数值不一致 / batch 不变性」与本域 TIM 簇直接接壤）
- [[04_posttrain_frameworks/index]] — 后训练框架与 RL Infra（verl / sandbox / 效率）
- [[02_pretraining/index]] — 预训练技术（优化器、低精度训练；[[low_precision_training_analysis]] 与 [[RL_Training_Inference_Precision_Analysis]] 现居此处）
- [[02_train_frameworks/megatron-lm/index]] — 分布式训练基础设施
- [[01_ai_frameworks/index]] — torch.compile 与图编译优化

> [!deprecated] 以下三条为旧目录结构下的链接，当前已无对应页面，保留以备追溯，请使用上方新链接：`[[02_training/index]]`、`[[06_infra/megatron-lm/index]]`、`[[torch_compile/index]]`
