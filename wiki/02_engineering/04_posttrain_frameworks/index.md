# 后训练框架 — 目录索引

> 覆盖 RLHF/对齐训练基础设施、Coding RL Sandbox 与 Infra、PPO/GRPO 源码级实现、工业 RL 训练框架
> (verl/slime/AReaL/ROLL)源码分析与 CUDA–Ascend 映射。
> 最后更新: 2026-07-31(kb-reorg P5 收尾:三域整合完成,`03_posttraining/` 纵向学习域已解散)

---

阅读路线入口(跨域纯导读页,正文全部归属本表 + `verl/`/`01_theory/04_posttraining`/`moonshot_kimi`
三处功能树,不计入下表):[[courses/posttraining_frontier|LLM 后训练前沿阅读课程]] —— D01→D12 顺序
索引 + 六级能力门槛,原 `03_posttraining/` 域(D00–D12)已随 kb-reorg P5 逐任务解散归位于此。

## 子目录

| 目录 | 核心主题 |
|------|---------|
| [[verl/index]] | 字节 **HybridFlow** 开源版 RL 后训练(RLHF)编排框架;单控制器编排 + 多控制器 SPMD 执行、DataProto、`RayPPOTrainer.fit` 数据流、统一 Worker+Engine 抽象、vLLM/SGLang rollout + 3D-HybridEngine 重分片、PPO/GRPO/GSPO 等算法;源码级分析 10 篇(9 篇 `main` 8a694930 深潜 + 端到端主链页基线 983cb0f) |

---

## 页面列表

### RL 算法源码实现（kb-reorg P5 归位）

> `batch_invariance_guide` 已迁至 [[07_training_reliability/index]]（训练批次不变性属确定性/可靠性问题域，非框架源码分析）。

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[rl_ppo_loss_and_grpo_analysis]] | TorchTitan + vLLM 源码 | PPO Loss 计算与 GRPO 训练流程的代码级走查(与 verl 的 [[verl_rl_algorithms_analysis]] 同类、框架不同) |

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

- [[courses/posttraining_frontier]] — LLM 后训练前沿阅读课程(D01→D12 顺序索引 + 六级能力门槛)
- [[../../01_theory/04_posttraining/index]] — 后训练算法理论
- [[07_training_reliability/index]] — 训练可靠性(`batch_invariance_guide` 现居此处)
- [[../02_train_frameworks/index]] — 训练框架
- [[../03_infer_frameworks/index]] — 推理框架
