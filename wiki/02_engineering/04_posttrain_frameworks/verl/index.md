# verl(HybridFlow)RLHF 框架 — 知识地图

> **代码基准**:verl `main` @ `8a694930`(9 篇实现深潜的基线)+ `983cb0f`(端到端迭代主链,见 [[10_verl_end_to_end_iteration_analysis]]) · 配套 Ray / vLLM / SGLang / FSDP2 / Megatron-LM 后端
> **最后更新**:2026-07-31(kb-reorg P5:并入 verl 端到端迭代主链页——原 `wiki/03_posttraining/07_verl_end_to_end_iteration_analysis.md`,基线 `983cb0f`;9 篇 `8a694930` 深潜页头加基线横幅互指,`use_v1` 默认值两基线间反转以 `[!contradiction]` 双记)· 原建 2026-06-22(9 篇 verl 源码级分析,从架构→实现→优化、overview→quickstart→deep dive)
> 一套 **10 篇** verl(开源版 **HybridFlow**,EuroSys'25,arXiv:2409.19256)源码级分析:9 篇基线 `8a694930` 的实现深潜 + 1 篇基线 `983cb0f` 的端到端迭代主链([[10_verl_end_to_end_iteration_analysis]])。verl 是 RL **后训练(RLHF)编排框架**,与预训练并行框架 [[torchtitan/index]] / [[megatron-lm/index]] 互补——后两者是 verl 的**训练后端**,verl 在其上编排「生成(rollout)→ 打分(reward)→ 优势(advantage)→ 更新(actor/critic)」的强化学习数据流。两基线机制若有出入,以新基线页 [[10_verl_end_to_end_iteration_analysis]] 为先(该页 `[!contradiction]` 记录了已知差异)。

---

## 设计哲学:单控制器编排,多控制器执行

verl 的核心是 **HybridFlow 混合控制器(hybrid-controller)编程模型**:

> **在 driver 上用「单控制器(single-controller)」的视角写一份中心化的 RL 数据流;每个算子下发到一组「多控制器(multi-controller)」的 SPMD worker 上并行执行。** 控制流集中、计算流分布。

这把 RLHF 这种「多模型、多阶段、强数据依赖」的复杂 dataflow,从手写分布式胶水里解放出来:driver 一行 `actor.update_actor(batch)` 背后,是把 `DataProto` 按 DP 切分(chunk)→ 散播到 N 个 worker → 各自 SPMD 计算 → 收集(concat)回 driver 的全套 dispatch/collect 机制(详见 [[11_verl_single_controller_analysis]] 与 [[12_verl_dataproto_analysis]])。

### 五条平面(plane)

| 平面 | 位置 | 职责 | 主分析页 |
|------|------|------|---------|
| **入口/驱动** | `trainer/main_ppo.py` · `trainer/ppo/ray_trainer.py` | Hydra 启动、Ray 初始化、`RayPPOTrainer.fit()` 主循环(`983cb0f` 起 `use_v1` 默认反转为 `true`,默认改道 `TaskRunnerV1`) | [[10_verl_end_to_end_iteration_analysis]] · [[20_verl_ray_trainer_analysis]] |
| **控制面** | `single_controller/` | `Worker`/`WorkerGroup`/`RayWorkerGroup` + `@register` dispatch/collect | [[11_verl_single_controller_analysis]] |
| **数据面** | `protocol.py` | `DataProto` / `DataProtoFuture`:driver↔worker 的统一数据契约 | [[12_verl_dataproto_analysis]] |
| **计算面** | `workers/engine_workers.py` · `workers/engine/*` | 统一 Worker(actor+rollout+ref 混合)+ FSDP/Megatron/... 引擎抽象 | [[13_verl_workers_engine_analysis]] |
| **生成面** | `workers/rollout/*` | vLLM/SGLang 异步推理服务 + 3D-HybridEngine 权重重分片 | [[14_verl_rollout_resharding_analysis]] |
| **算法面** | `trainer/ppo/core_algos.py` | 14 种优势估计 + 11 种 policy loss + KL 惩罚 | [[15_verl_rl_algorithms_analysis]] |

## 段位与阅读顺序(kb-reorg P5 Task 8,2026-07-31)

文件名两位数字前缀 = 段位:段 0(01-09)入门导览——架构总览(阅读起点)与快速上手;段 1(10-19)核心机制
主线,按「五条平面」表的实际管线顺序排列(入口/驱动的当前基线主链 → 控制面 → 数据面 → 计算面 → 生成面
→ 算法面);段 2(20-29)深潜/专题——`verl_ray_trainer_analysis` 是入口/驱动平面上与端到端主链并列的
`8a694930` **legacy 深潜**页(见上文「HEAD 架构演进提示」),不计入主线;段 3(30-39)方法论/工程实践——
性能/显存优化横切指南。下表按段位排列,与下方分主题小节互为索引:

| 段 | 页面 | 一句话 |
|---|------|------|
| 0 | [[01_verl_architecture_overview_analysis]] | HybridFlow 架构总览(阅读起点) |
| 0 | [[02_verl_quickstart_guide]] | 安装与快速上手 |
| 1 | [[10_verl_end_to_end_iteration_analysis]] | 端到端主链(当前基线 `983cb0f`) |
| 1 | [[11_verl_single_controller_analysis]] | 控制面机制 |
| 1 | [[12_verl_dataproto_analysis]] | 数据面机制 |
| 1 | [[13_verl_workers_engine_analysis]] | 计算面机制 |
| 1 | [[14_verl_rollout_resharding_analysis]] | 生成面 + 3D-HybridEngine 权重重分片机制 |
| 1 | [[15_verl_rl_algorithms_analysis]] | 算法面(优势估计/policy loss 注册表) |
| 2 | [[20_verl_ray_trainer_analysis]] | `RayPPOTrainer.fit` legacy 深潜(`8a694930`) |
| 3 | [[30_verl_optimization_analysis]] | 性能/显存优化方法论 |

## 文档系列(10 篇)

### 端到端主链(当前基线)

| 页面 | 层次 | 核心机制 |
|------|------|---------|
| [[10_verl_end_to_end_iteration_analysis]] | **End-to-End · 当前基线 `983cb0f`** | 一轮 PPO/GRPO 迭代的端到端调用链、角色/资源池创建机制、DataProto 契约表、advantage/policy-loss 注册表、权重刷新时序、on-policy/TIM 诊断、`use_v1` 默认反转的 `[!contradiction]` 记录 |

### 由浅入深三层(overview → quick start → deep dive,基线 `8a694930`)

| 页面 | 层次 | 核心机制 |
|------|------|---------|
| [[01_verl_architecture_overview_analysis]] | **Overview · 架构** | HybridFlow 混合控制器、五平面映射、五角色与混合 worker、v0/v1 入口分裂、master 架构图(**阅读起点**) |
| [[02_verl_quickstart_guide]] | **Quick Start** | 安装、`python -m verl.trainer.main_ppo` Hydra 启动、config 体系、一次 GRPO 端到端走查、后端/算法切换旋钮 |

### 深挖实现(deep dive · implementation,基线 `8a694930`)

| 页面 | 子系统 | 核心机制 |
|------|--------|---------|
| [[11_verl_single_controller_analysis]] | **控制面** | `Worker`/`WorkerGroup`、`_bind_worker_method`、`@register`+8 种 `Dispatch` 模式、`DP_COMPUTE_PROTO` chunk/concat、`RayWorkerGroup`+placement group、colocate(`create_colocated_worker_cls`/`FusedWorker`) |
| [[12_verl_dataproto_analysis]] | **数据面** | `DataProto`(batch/non_tensor_batch/meta_info)、chunk/concat/union/repeat/pad、`BatchData` 类型分发、`DataProtoFuture` 异步句柄 |
| [[20_verl_ray_trainer_analysis]] | **编排(legacy 深潜)** | `Role`/`ResourcePoolManager`/`init_workers`、`fit()` 逐步追踪(gen→reward→old/ref logprob→values→KL→advantage→update_critic→update_actor→update_weights)、PPO/GRPO 数据流时序图;`983cb0f` 起 `use_v1` 默认反转,本页记录的是需显式关闭 v1 才会跑的路径,当前默认主链见 [[10_verl_end_to_end_iteration_analysis]] |
| [[13_verl_workers_engine_analysis]] | **计算面** | `TrainingWorker`/`ActorRolloutRefWorker`、`BaseEngine` 模板方法(train_batch/infer_batch)、FSDP/Megatron 引擎、`update_actor` 策略梯度路径、worker 级 offload |
| [[14_verl_rollout_resharding_analysis]] | **生成面 + 优化** | `BaseRollout`/异步 server(`ServerAdapter`)、vLLM/SGLang 集成、sleep/wake KV 释放、**3D-HybridEngine 重分片**(`get_per_tensor_param`+`CheckpointEngine`+CUDA-IPC bucketed transfer) |

### 算法与优化(algorithms & optimization,基线 `8a694930`)

| 页面 | 主题 | 核心机制 |
|------|------|---------|
| [[15_verl_rl_algorithms_analysis]] | **RL 算法** | `register_adv_est` 注册表(GAE/GRPO/RLOO/REINFORCE++/ReMax/OPO/GPG…14 种)、`register_policy_loss`(vanilla/GSPO/CISPO/clip_cov/kl_cov/geo_mean…11 种)、KL k1/k2/k3 估计与 in-reward vs in-loss 两处施加、算法→config 映射 |
| [[30_verl_optimization_analysis]] | **性能/显存** | 重分片经济性、colocated vs disaggregated placement、param/grad/optimizer offload、序列打包/去填充/`balance_batch`/动态批、Ulysses 序列并行、异步 RL(one-step-off / fully-async / v1 TransferQueue) |

## 五个逻辑角色

verl 把 RLHF 的模型职责拆成五个**逻辑角色**,但物理上高度 colocate:

| 角色 | 职责 | 物理承载(HEAD `8a694930`) |
|------|------|------|
| **Actor** | 被训练的策略;`update_actor` 跑策略梯度 | `ActorRolloutRefWorker` 内的训练 `TrainingWorker` |
| **Rollout** | 用 vLLM/SGLang 高速生成 response | 同一 worker 内的异步推理 server,与 actor **时分复用同一组 GPU** |
| **Reference** | 冻结参考策略,算 ref logprob 供 KL | 同一 worker 内第二个 `TrainingWorker`(无 LoRA 时可退化) |
| **Critic** | 价值函数(仅 PPO 等需要) | 带 value head 的 `TrainingWorker`(`model_type="value_model"`) |
| **Reward** | 函数式/模型式奖励 + KL 惩罚 | `workers/reward_manager/*` + `experimental/reward_loop` |

> [!note] HEAD 架构演进提示
> 本系列基准 `8a694930` 的代码已显著重构,与多数博客描述的「经典 HybridFlow」有出入,各页已逐一标注:
> - `RayPPOTrainer`(`ray_trainer.py`)被标 `@deprecated`;在 `8a694930` 上默认 `trainer.use_v1=false` 仍走它,**但到 [[10_verl_end_to_end_iteration_analysis]] 的基线 `983cb0f`,该默认值已反转为 `use_v1=true`**——默认执行路径变成 `TaskRunnerV1`(TransferQueue + `AgentLoopManager` 驱动),legacy `RayPPOTrainer.fit` 降级为需要显式 `trainer.use_v1=false` 才会跑的路径(详见该页的 `[!contradiction]` 记录)。本系列 9 篇深潜文档仍以 legacy 路径为教学主线(结构最完整、逐行有源码),v1/TransferQueue 路径本系列尚无专页覆盖。
> - **不存在独立的 `CriticWorker`/`RewardModelWorker` 类**——critic 是带 value head 的 `TrainingWorker`,reward 走 reward_manager/reward_loop。
> - rollout 已退役 SPMD 同步模式,改为**异步 server**;生成由 `LLMServerManager`/`AgentLoopManager` 驱动,而非 worker RPC。

## 经典 RL 数据流(一个 PPO/GRPO step)

```mermaid
flowchart LR
    D[Driver: RayPPOTrainer.fit] -->|gen_batch| R[Rollout vLLM/SGLang]
    R -->|responses| LP[Actor: old_log_prob]
    LP --> REF[Reference: ref_log_prob]
    REF --> V[Critic: values（PPO）]
    V --> RW[Reward: 函数/模型打分 + apply_kl_penalty]
    RW --> ADV[compute_advantage（GAE/GRPO…）]
    ADV --> UC[update_critic]
    UC --> UA[update_actor]
    UA -->|update_weights 重分片| R
```

每一步对应「driver 调 worker 方法 → dispatch 切分 DataProto → SPMD 执行 → collect 回填字段」;当前基线(`983cb0f`)的端到端主链见 [[10_verl_end_to_end_iteration_analysis]],`8a694930` 逐行追踪见 [[20_verl_ray_trainer_analysis]],dispatch 机制见 [[11_verl_single_controller_analysis]],优势/损失数学见 [[15_verl_rl_algorithms_analysis]]。

## 与训练后端的关系(Cross-Domain Links)

verl 不自己实现并行,而是把 FSDP2 / Megatron-LM 当**训练后端**、vLLM / SGLang 当**生成后端**。要理解 verl 计算面的真实机制,需下钻到这些后端的源码分析:

- [[torchtitan/index]] / [[torchtitan_fsdp_analysis]] —— FSDP2 `fully_shard` 逐参数分片、`DTensor.full_tensor()`(verl 训练侧重分片的底座)
- [[megatron-lm/index]] —— Megatron TP/PP/EP 分片(verl Megatron 后端;`bridge.export_hf_weights` 反分片回 HF 权重)
- [[distributed_optimizer_deep_dive]] —— FSDP2/ZeRO/MindSpeed 优化器分片与 offload,对应 [[30_verl_optimization_analysis]] 的显存手段
- [[torchtitan_cp_analysis]] —— 序列/上下文并行,对照 [[30_verl_optimization_analysis]] 的 Ulysses 序列并行
- [[comm_compute_overlap_analysis]] —— 通信掩盖,对照 verl 异步 RL 的生成-训练重叠

## Related Pages

- [[10_verl_end_to_end_iteration_analysis]] — verl `983cb0f` 当前基线的端到端主链、角色/资源池机制、算法与 weight refresh(**本域当前基线权威页**;`RayPPOTrainer.fit` 本身在此基线已非默认路径,见该页 `[!contradiction]`)
- [[30_rl_framework_comparison]] — verl、slime、AReaL、ROLL 统一机制矩阵
- [[courses/posttraining_frontier]] — 后训练前沿阅读课程(原 D00–D12 学习域已解散,内容归位至功能树)
- [[01_verl_architecture_overview_analysis]] · [[02_verl_quickstart_guide]] —— 入门两篇(架构总览 + 快速上手,基线 `8a694930`)
- [[11_verl_single_controller_analysis]] · [[12_verl_dataproto_analysis]] · [[20_verl_ray_trainer_analysis]] · [[13_verl_workers_engine_analysis]] · [[14_verl_rollout_resharding_analysis]] —— 实现五篇(控制/数据/编排/计算/生成,基线 `8a694930`)
- [[15_verl_rl_algorithms_analysis]] · [[30_verl_optimization_analysis]] —— 算法与优化两篇(基线 `8a694930`)
- [[02_engineering/04_posttrain_frameworks/index]] —— 后训练框架目录索引(本系列所在)
- [[02_engineering/02_train_frameworks/index]] · [[torchtitan/index]] · [[megatron-lm/index]] —— 训练框架(verl 的训练后端)
