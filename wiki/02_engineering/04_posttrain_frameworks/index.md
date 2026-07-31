# 后训练框架 — 目录索引

> 覆盖 RLHF/对齐训练基础设施、奖励模型训练框架、Coding RL Sandbox 与 Infra、RL 训练框架源码分析
> 最后更新: 2026-07-31

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
| [[rl_sandbox_design_analysis]] | RollArt / ProRL Agent / Anthropic 公开资料 / Kimi K3 | 10 万级并发 sandbox, Firecracker microVM, Disaggregated 架构, Rollout 三阶段, K3 harness 版本化与 Fork/Pause/Snapshot |
| [[rl_infra_efficiency_analysis]] | RollArt / RollPacker / ProRL Agent / Claude 4 访谈 / AReaL / Kimi K3 | 异步训练、长尾治理（redundant rollouts）、hardware-aware 调度、in-flight reward、environment 池十万级、admission-aware backpressure |

### 后训练框架源码对照（kb-reorg P5 迁入）

> 2026-07-31 从 `wiki/03_posttraining/`（原 D05–D11）迁入,是后训练三域整合的一部分。

| 页面 | 核心主题 |
|------|---------|
| [[posttraining_infra_mechanism_analysis]] | 后训练 Infra 核心机制(原 D05):control/data/weight 三平面模型、五种执行结构、backpressure 接口定义、weight publish 协议、checkpoint 与故障域 |
| [[rl_framework_comparison]] | 工业后训练框架对比(原 D06):verl/slime/AReaL/ROLL 统一机制矩阵、四级支持证据、控制面可修改性、async 语义对照 |
| [[slime_architecture_analysis]] | slime 高性能与异步架构:Megatron + SGLang、DataSource/buffer、weight transport、async producer |
| [[areal_async_architecture_analysis]] | AReaL Fully Async 与 Agentic 架构:微服务、Hermes、policy lag、agent trajectory |
| [[roll_strategy_and_ascend_analysis]] | ROLL Strategy、异构与 Ascend:多后端 Strategy、AutoDeviceMapping、RLVR 与 Agentic async 差异 |
| [[cuda_ascend_posttraining_stack_comparison]] | CUDA–Ascend 后训练栈对照:通信、推理、并行、权重同步、kernel 与诊断的能力与差距矩阵 |

---

## 关联域

- [[../../01_theory/04_posttraining/index]] — 后训练算法理论
- [[../02_train_frameworks/index]] — 训练框架
- [[../03_infer_frameworks/index]] — 推理框架
