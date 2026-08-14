# 后训练框架 — 目录索引

> 覆盖 RLHF/对齐训练基础设施、Coding RL Sandbox 与 Infra、PPO/GRPO 源码级实现、工业 RL 训练框架
> (verl/slime/AReaL/ROLL)源码分析与 CUDA–Ascend 映射。
> 最后更新: 2026-08-14(slime `main@681b3adc` 独立知识域补齐官方特性源码解读，并加入 `vllm-project/vime@8144096e` 的 vLLM backend 衍生实现与支持度审计)

---

阅读路线入口(跨域纯导读页,正文全部归属本表 + `verl/`/`01_theory/04_posttraining`/`moonshot_kimi`
三处功能树,不计入下表):[[courses/posttraining_frontier|LLM 后训练前沿阅读课程]] —— D01→D12 顺序
索引 + 六级能力门槛,原 `03_posttraining/` 域(D00–D12)已随 kb-reorg P5 逐任务解散归位于此。

## 子目录

| 目录 | 核心主题 |
|------|---------|
| [[verl/index]] | 字节 **HybridFlow** 开源版 RL 后训练(RLHF)编排框架;单控制器编排 + 多控制器 SPMD 执行、DataProto、`RayPPOTrainer.fit` 数据流、统一 Worker+Engine 抽象、vLLM/SGLang rollout + 3D-HybridEngine 重分片、PPO/GRPO/GSPO 等算法;源码级分析 10 篇(9 篇 `main` 8a694930 深潜 + 端到端主链页基线 983cb0f)+ **文档级 1 篇**(v1/TransferQueue 路径,2026-08-11 补) |
| [[slime/index]] | THUDM **slime** 独立源码知识域；SGLang-native + Megatron-native，覆盖配置、端到端迭代、Ray/数据/rollout/训练/loss/权重、训推一致性、容错观测、backend 扩展、vime/vLLM 衍生实现、OPD、在线 MTP、FP8/INT4、新架构、Agent、优化与稳定性，共 21 篇 |

---

## 段位与阅读顺序(kb-reorg P5 Task 8,2026-07-31)

文件名两位数字前缀 = 段位;下表覆盖本目录根并只列子域入口(`verl/`、`slime/` 内部段位见各自 index)。段 0(01-09)入门
导览——三平面机制总览;段 1(10-19)核心机制主线,按 PPO/GRPO 算法实现 → sandbox 执行环境 → infra
效率调度排列;段 2(20-29)深潜/专题——slime/AReaL/ROLL 三个框架各自的架构专题;段 3(30-39)方法论/
对照——工业框架统一机制矩阵、CUDA–Ascend 后训练栈对照。下表按段位排列,与下方分主题小节互为索引:

| 段 | 页面 | 一句话 |
|---|------|------|
| 0 | [[01_posttraining_infra_mechanism_analysis]] | control/data/weight 三平面机制总览 |
| 1 | [[10_rl_ppo_loss_and_grpo_analysis]] | TorchTitan + vLLM PPO Loss/GRPO 源码实现 |
| 1 | [[11_rl_sandbox_design_analysis]] | Coding RL Sandbox 架构设计 |
| 1 | [[12_rl_infra_efficiency_analysis]] | RL Infra 效率优化机制 |
| 1 | [[13_opd_infra_mechanism_analysis]] | OPD 基础设施机制:带宽账与八项工程工作 W1-W8 |
| 2 | [[slime/index]] | slime 独立源码知识域入口 |
| 2 | [[21_areal_async_architecture_analysis]] | AReaL 框架架构专题 |
| 2 | [[22_roll_strategy_and_ascend_analysis]] | ROLL 框架架构专题 |
| 3 | [[30_rl_framework_comparison]] | 工业框架统一机制矩阵对比 |
| 3 | [[31_cuda_ascend_posttraining_stack_comparison]] | CUDA–Ascend 后训练栈对照 |
| 3 | [[32_opd_framework_support_comparison]] | 六框架 OPD 支持对照与选型(veRL/slime/TRL/NeMo-RL/Tinker/KDFlow) |

## 页面列表

### RL 算法源码实现（kb-reorg P5 归位）

> `20_batch_invariance_guide` 已迁至 [[07_training_reliability/index]]（训练批次不变性属确定性/可靠性问题域，非框架源码分析）。

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[10_rl_ppo_loss_and_grpo_analysis]] | TorchTitan + vLLM 源码 | PPO Loss 计算与 GRPO 训练流程的代码级走查(与 verl 的 [[15_verl_rl_algorithms_analysis]] 同类、框架不同) |

### Coding RL Sandbox 与 Infra

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[11_rl_sandbox_design_analysis]] | RollArt / ProRL Agent / Anthropic 公开资料 / Kimi K3 | 10 万级并发 sandbox, Firecracker microVM, Disaggregated 架构, Rollout 三阶段, K3 harness 版本化与 Fork/Pause/Snapshot |
| [[12_rl_infra_efficiency_analysis]] | RollArt / RollPacker / ProRL Agent / Claude 4 访谈 / AReaL / Kimi K3 | 异步训练、长尾治理（redundant rollouts）、hardware-aware 调度、in-flight reward、environment 池十万级、admission-aware backpressure |

### 后训练框架源码对照（kb-reorg P5 迁入）

> 2026-07-31 从 `wiki/03_posttraining/`（原 D05–D11）迁入,是后训练三域整合的一部分。

| 页面 | 核心主题 |
|------|---------|
| [[01_posttraining_infra_mechanism_analysis]] | 后训练 Infra 核心机制(原 D05):control/data/weight 三平面模型、五种执行结构、backpressure 接口定义、weight publish 协议、checkpoint 与故障域 |
| [[30_rl_framework_comparison]] | 工业后训练框架对比(原 D06):verl/slime/AReaL/ROLL 统一机制矩阵、四级支持证据、控制面可修改性、async 语义对照 |
| [[slime/index]] | slime 独立源码知识域：架构总览、配置与端到端主链、Ray/DataSource/SGLang/Megatron 实现、loss/并行、权重提交、容错、rollout backend 扩展、vime/vLLM 衍生实现支持度、OPD、在线 MTP、低精度、新模型架构、Agent workflow、rollout 优化、训推一致性与稳定性；slime 基线 `main@681b3adc`，vime 基线 `main@8144096e` |
| [[21_areal_async_architecture_analysis]] | AReaL Fully Async 与 Agentic 架构:微服务、Hermes、policy lag、agent trajectory |
| [[22_roll_strategy_and_ascend_analysis]] | ROLL Strategy、异构与 Ascend:多后端 Strategy、AutoDeviceMapping、RLVR 与 Agentic async 差异 |
| [[31_cuda_ascend_posttraining_stack_comparison]] | CUDA–Ascend 后训练栈对照:通信、推理、并行、权重同步、kernel 与诊断的能力与差距矩阵 |

### 在线策略蒸馏（OPD）的系统侧（2026-08 新建）

> OPD 对基建的要求可概括为「**RL 的回路，加一个新角色**」——学生 rollout、权重同步与 PPO/GRPO 完全同构，唯一新增的是**教师**：一个只做 prefill 打分、不做 decode 的推理服务。压力集中在带宽/存储、调度、一致性三处。算法侧见 [[14_on_policy_distillation_analysis]]。

| 页面 | 核心主题 |
|------|---------|
| [[13_opd_infra_mechanism_analysis]] | 与 SFT/RL 的系统需求对照（critic 消失是省出的预算）、**信号格式四档带宽账**（全词表 vs 采样 token 相差约 5 个数量级）、成本模型与教师刷新率 $\rho$、**八项工作清单 W1–W8**、为何 OPD 的 staleness 容忍窗口比 RLHF 更窄 |
| [[32_opd_framework_support_comparison]] | veRL/slime/TRL/NeMo-RL/Tinker/KDFlow 逐项支持矩阵与选型、OpenRLHF「可用而非原生支持」辨析、生产系统自研层、六条生态 Gap、预算分配三段模式 |

**原始来源**：仓库外的 OPD 调研稿目录 `opd-survey/OPD-Infra-Survey-2026-08.md` 及其 `research-notes/` 底稿（**按用户决定未纳入 `raw/`**）。

---

## 关联域

- [[courses/posttraining_frontier]] — LLM 后训练前沿阅读课程(D01→D12 顺序索引 + 六级能力门槛)
- [[../../01_theory/04_posttraining/index]] — 后训练算法理论
- [[07_training_reliability/index]] — 训练可靠性(`20_batch_invariance_guide` 现居此处)
- [[../02_train_frameworks/index]] — 训练框架
- [[../03_infer_frameworks/index]] — 推理框架
