# LLM 对齐与偏好优化 — 目录索引

> 覆盖 RLHF、DPO、GRPO、PPO 等对齐方法及相关前置研究
> 最后更新: 2026-05-07

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

### 高级方法

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[vapo_analysis]] | VAPO (2504.05118) | 基于价值模型的 RL, AIME 60.4 |
| [[rlhf_foundations_analysis]] | 多篇综合 | ReMax, Weak-to-Strong, RM Overoptimization, RigorLLM |
| [[RL_PPO_Loss_and_GRPO_Analysis]] | 源码分析 | PPO Loss 与 GRPO 的代码级对比 |
| [[kimi_k1.5_analysis]] | Kimi K1.5 | 长上下文 RL 推理训练 |

---

## 关联域

- [[../02_training/index]] — 训练技术（优化器, 精度）
- [[../06_infra/megatron-lm/index]] — 分布式训练基础设施
- [[../../torch_compile/index]] — torch.compile 编译优化
