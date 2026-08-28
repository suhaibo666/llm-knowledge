---
title: "后训练框架 — 目录索引"
---

# 后训练框架 — 目录索引

> 覆盖 RLHF/对齐训练基础设施、Coding RL Sandbox 与 Infra、TitanRL 异步 GRPO/DAPO 源码级实现、工业 RL 训练框架
> (verl/slime/AReaL/ROLL)源码分析与 CUDA–Ascend 映射。
> 最后更新: 2026-08-28（verl `main@254a23ed`：默认 V1、TransferQueue、stable/experimental async、CheckpointEngine delta 与当前 Engine 矩阵）

---

## 模块定位：做什么 · 提供什么能力 · 边界在哪

**一句话**：后训练框架解决的是**同一批权重必须同时以两种形态存在**——训练态（按 Megatron/FSDP 的方式分片、要反向、有优化器状态）与推理态（按 vLLM/SGLang 的方式分片、只前向、要 KV cache）——并且这两态要在**每一轮迭代里互相喂数据、互相同步权重**。

**为什么必须独立成一层**：SFT 只是训练，可以完全住在 [[02_engineering/02_train_frameworks/index|训练框架]] 里。但 RL 后训练把**推理引擎拉进了训练循环**，于是出现训练框架和推理框架各自都没有的三个问题：

1. **权重要在两套分片布局之间来回搬**——训练侧的 TP/PP/EP 切法和推理侧不一样，每次策略更新后都要重分片并发布；
2. **两个引擎在时间上争抢同一批卡**——rollout 的长尾（少数长轨迹拖住整批）与训练的气泡必须互相填充，否则利用率对半砍；
3. **训练与推理的数值必须一致**——两侧 log-prob 若不逐位对齐，importance ratio 就不是 1，梯度带上系统性偏差。

这三件事决定了本域的核心抽象不是"模型"或"算法"，而是 **control plane / data plane / weight plane 三个平面**：谁下命令、数据怎么流、权重怎么发布。

### 本域覆盖的系统与各自定位

本域涉及多个框架，覆盖度高度不均衡，如实标注：

| 系统 | 在本域中的定位 | 本库覆盖 | 基线 |
|---|---|---|---|
| **slime** | THUDM，SGLang-native + Megatron-native；本域**覆盖最全**的框架，从配置、端到端主链到容错、低精度、Agent workflow 逐项展开 | 20 篇 + index（**系统性源码覆盖**） | `THUDM/slime@681b3adc` |
| **verl** | 字节 **HybridFlow** 开源版；当前默认 V1 以 TransferQueue 解耦控制/数据，Worker/Engine 与 CheckpointEngine 分别承担计算和权重发布 | 14 篇 + index（**系统性源码覆盖**） | `volcengine/verl@254a23ed`；3 篇 V0 档案冻结于 `8a694930` |
| **AReaL** | Fully Async 与 Agentic 架构的代表：微服务化、Hermes、policy lag 的显式管理 | 1 篇（**架构专题，非全景**） | 见页头 |
| **ROLL** | 多后端 Strategy 抽象 + AutoDeviceMapping；异构与 **Ascend** 侧的代表 | 1 篇（**架构专题，非全景**） | 见页头 |
| **vime** | vLLM backend 的衍生实现；用于审计 slime 的 rollout backend 扩展点 | 1 篇（在 `slime/` 下） | `vllm-project/vime@8144096e` |
| **TRL / NeMo-RL / Tinker / KDFlow / OpenRLHF** | 仅在 OPD 支持度对照表里做**选型定位**，本库**没有**源码级深挖 | 0 篇（仅出现在对照表） | — |
| **Sandbox / RL Infra 效率 / OPD 系统侧** | **技术专题而非产品**：跨框架的公开资料综述（RollArt、ProRL Agent、Kimi K3、Claude 4 访谈等） | 5 篇 | 公开资料，见各页头 |

> 引用注意：能力清单里的源码锚点主要落在 **verl 与 slime**。AReaL 与 ROLL 只有架构专题一篇，不足以支撑"这类框架都如何如何"的断言；跨框架的横向结论请以 [[30_rl_framework_comparison]] 的证据分级为准，那页对每条能力标了四级支持证据。

### 本域提供的能力

下表按**三平面**组织；"样本与锚点"列说明本库是拿谁的实现讲的。源码锚点按侧车 checkout `verl@254a23ed`、`slime@681b3adc` 核对路径存在：

| 平面 | 能力 | 具体提供什么 | 样本与源码锚点 | 详见 |
|---|---|---|---|---|
| control | **单控制器编排 + 多控制器执行** | 一个 driver 下命令、各 rank SPMD 执行，避免把编排逻辑写进每个 worker | verl `verl/single_controller/` | [[verl/index\|verl]] |
| control | **训练主循环** | 一轮迭代的完整拍子：生成 → 打分 → 优势 → 更新 → 发布权重 | verl `verl/trainer/ppo/v1/` · slime `slime/ray/` | [[10_verl_end_to_end_iteration_analysis]] · [[01_posttraining_infra_mechanism_analysis]] |
| control | **异步控制器** | policy version、completion queue、staleness 与 rollout/train 解耦 | verl stable V1 + experimental · AReaL（架构专题） | [[17_verl_v1_async_trainer_analysis]] · [[22_verl_fully_async_dynamic_schedule_deepdive]] · [[21_areal_async_architecture_analysis]] |
| data | **统一数据协议** | controller 传引用、worker 延迟物化变长轨迹；V0 保留 DataProto 档案 | verl TransferQueue/`KVBatchMeta`/`tqbridge` | [[16_verl_v1_transfer_queue_analysis]] |
| data | **rollout 后端抽象** | 把 vLLM/SGLang 包成可被训练循环驱动的生成服务，而不是各写一份胶水 | verl `verl/workers/rollout/` · slime `slime/backends/sglang_utils/` | [[13_slime_sglang_rollout_engine_analysis]] · [[19_slime_rollout_backend_extension_analysis]] |
| weight | **训推权重重分片与发布** | 训练态分片 → 推理态 full/delta payload 的转换与原地更新，避免落盘往返 | verl CheckpointEngine/`delta_sharded` · slime `slime/backends/megatron_utils/` | [[14_verl_rollout_resharding_analysis]] · [[21_verl_delta_weight_sync_deepdive]] · [[16_slime_weight_sync_analysis]] |
| 算法 | **RL 目标函数实现** | PPO/GRPO/DAPO 等的 loss、优势估计、clip 与 KL 项 | verl `verl/trainer/ppo/core_algos.py` | [[15_verl_rl_algorithms_analysis]] |
| 一致性 | **训推数值一致** | 两侧 log-prob 对齐的工程手段与验收 | slime 专页 | [[17_slime_train_inference_consistency_analysis]] |
| 环境 | **Sandbox 与 Agent 执行环境** | 十万级并发的代码/工具执行环境，Fork/Pause/Snapshot 与 harness 版本化 | 公开资料（**非本地源码**） | [[11_rl_sandbox_design_analysis]] |
| 效率 | **异步与长尾治理** | 冗余 rollout、in-flight reward、hardware-aware 调度、admission-aware backpressure | 公开资料 + AReaL | [[12_rl_infra_efficiency_analysis]] |
| 异构 | **多后端与 Ascend** | Strategy 抽象、AutoDeviceMapping；CUDA↔Ascend 能力与差距矩阵 | ROLL（架构专题） | [[22_roll_strategy_and_ascend_analysis]] · [[31_cuda_ascend_posttraining_stack_comparison]] |
| 蒸馏 | **OPD 的系统侧** | 在 RL 回路上加一个只做 prefill 打分的教师角色，及其带宽/调度/一致性代价 | 调研稿（**非源码**） | [[13_opd_infra_mechanism_analysis]] · [[32_opd_framework_support_comparison]] |

### 不属于本模块的

- RL/对齐**算法本身**为什么这么设计（GRPO 为何去掉 critic、DPO 与 PPO 的关系）→ [[01_theory/04_posttraining/index|后训练理论]]；本域只讲这些目标函数如何落地与编排；
- rollout 引擎内部的调度与 KV 管理 → [[02_engineering/03_infer_frameworks/index|推理框架]]；本域是它的**消费方**，只关心它暴露的生成/权重更新接口；
- 训练侧的并行策略与通信掩盖 → [[02_engineering/02_train_frameworks/index|训练框架]]；
- 训推数值一致性的**成因与判据**（batch 不变性、确定性算子）→ [[02_engineering/07_training_reliability/index|训练可靠性]]；本域只承担"两侧必须对齐"这个需求。

### 与兄弟域的关系

本域是全栈唯一**同时消费训练域与推理域**的层——它把 `02` 的训练引擎与 `03` 的推理引擎缝进一个循环，因此也是唯一会被两边实现细节同时反噬的地方（重分片、数值一致、资源争抢三个问题全部诞生于这道缝）。

---

阅读路线入口(跨域纯导读页,正文全部归属本表 + `verl/`/`01_theory/04_posttraining`/`moonshot_kimi`
三处功能树,不计入下表):[[courses/posttraining_frontier|LLM 后训练前沿阅读课程]] —— D01→D12 顺序
索引 + 六级能力门槛,原 `03_posttraining/` 域(D00–D12)已随 kb-reorg P5 逐任务解散归位于此。

## 子目录

| 目录 | 核心主题 |
|------|---------|
| [[verl/index]] | 字节 **HybridFlow** 开源版 RL 后训练编排框架；14 篇内容页统一覆盖默认 V1 sync、TransferQueue、stable V1 async、experimental fully async、Worker/Engine、rollout/PD、CheckpointEngine full/`delta_sharded`、算法与优化；当前基线 `main@254a23ed`，另保留 3 篇 `8a694930` V0 机制档案 |
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
| 1 | [[10_rl_ppo_loss_and_grpo_analysis]] | TitanRL 异步 controller、版本窗口、零有效 token 不变量、rollout/weight-sync、checkpoint 边界与 GRPO/DAPO |
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

### TitanRL 异步运行时与 RL 算法实现（kb-reorg P5 归位）

> `20_batch_invariance_guide` 已迁至 [[07_training_reliability/index]]（训练批次不变性属确定性/可靠性问题域，非框架源码分析）。

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[10_rl_ppo_loss_and_grpo_analysis]] | TorchTitan TitanRL `main@a3168782c` | 有界异步 controller、windowed FIFO、全局 response-token 分母、版本化权重同步、restart 非 exactly-once 边界与逐 token GRPO/DAPO（与 verl 的 [[15_verl_rl_algorithms_analysis]] 同类、框架不同） |

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
