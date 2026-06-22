# 后训练框架 — 目录索引

> 覆盖 RLHF/对齐训练基础设施、奖励模型训练框架、Coding RL Sandbox 与 Infra、RL 训练框架源码分析
> 最后更新: 2026-06-22

---

## 子目录

| 目录 | 核心主题 |
|------|---------|
| [[verl/index]] | 字节 **HybridFlow** 开源版 RL 后训练(RLHF)编排框架;单控制器编排 + 多控制器 SPMD 执行、DataProto、`RayPPOTrainer.fit` 数据流、统一 Worker+Engine 抽象、vLLM/SGLang rollout + 3D-HybridEngine 重分片、PPO/GRPO/GSPO 等算法;源码级分析 9 篇(`main` 8a694930) |

---

## 页面列表

### 数值与确定性

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[batch_invariance_guide]] | 综合分析 | 训练批次不变性: 数学定义、数值稳定性、loss 聚合顺序依赖 |

### Coding RL Sandbox 与 Infra

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[rl_sandbox_design_analysis]] | RollArt / ProRL Agent / Anthropic 公开资料 | 10 万级并发 sandbox, Firecracker microVM, Disaggregated 架构, Rollout 三阶段 |
| [[rl_infra_efficiency_analysis]] | RollArt / RollPacker / ProRL Agent / Claude 4 访谈 | 异步训练、长尾治理（redundant rollouts）、hardware-aware 调度、in-flight reward、environment 池十万级 |

---

## 关联域

- [[../../01_theory/04_posttraining/index]] — 后训练算法理论
- [[../02_train_frameworks/index]] — 训练框架
- [[../03_infer_frameworks/index]] — 推理框架
