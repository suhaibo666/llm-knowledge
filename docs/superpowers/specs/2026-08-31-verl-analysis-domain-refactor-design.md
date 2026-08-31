# Verl 分析域能力优先重构设计

> 批准日期：2026-08-31
> 源码基线：`verl` `main` @ `254a23edc62f25ebfae626e3932ae285d6f86009`
> 源码工作区：`E:/97-codes/torch_parallel/verl`
> 文档域：`wiki/02_engineering/04_posttrain_frameworks/verl/`

## 1. 中心命题

verl 不是一个单体 PPO trainer，而是一组围绕轨迹、批数据、模型状态和资源状态协作的运行时边界：
TaskRunner 与 Trainer 拥有控制依赖，AgentLoop/RewardLoop 拥有轨迹生成与打分，TransferQueue
拥有跨边界数据状态，Worker/Engine 拥有模型计算，LLM server 拥有请求与 KV 状态，
CheckpointEngine 拥有 actor 到 rollout 的权重发布会话，训练 checkpoint 则拥有跨重启恢复所需的
持久状态。文档重构必须让每个能力、状态和生命周期只有一个权威页面。

本次重构只修改 llm-knowledge 中的 Verl 分析域，不修改上游 Verl 源码。

## 2. 冻结条件

- 仓库与提交：`E:/97-codes/torch_parallel/verl` @
  `254a23edc62f25ebfae626e3932ae285d6f86009`。
- 分支状态：`main` 与 `origin/main` 同提交；源码工作区仅有用户的未跟踪文件
  `GRPO_Analysis.md`，不得修改。
- 读者：需要理解、扩展或诊断 Verl 的 RL 系统工程师；默认理解 PPO/GRPO 和分布式训练基础。
- Wiki 位置：`wiki/02_engineering/04_posttrain_frameworks/verl/`。
- live/legacy 边界：V1 sync、V1 stable async 和公共运行时属于当前主线；
  `RayPPOTrainer` V0 是仍可显式选择的 legacy；`experimental/fully_async_policy` 是独立实验路径。
- 证据缺口：外部 TransferQueue 包内部一致性、部分可选 transport、真实多节点性能与 crash 测试
  只能标为外部事实或未验证，不能由调用点外推。

## 3. 能力地图与唯一 owner

| 能力 | 权威页面 | 不属于该页面的内容 |
|---|---|---|
| 系统组成、模式路由、live/legacy 边界 | 01 架构总览 | 子系统状态机细节 |
| 启动、样例与配置选择 | 02 快速上手 | 源码机制深潜 |
| Ray 资源、WorkerGroup、RPC dispatch/collect | 11 single-controller | Trainer 生命周期、批数据语义 |
| DataProto/TensorDict 本地批契约 | 12 DataProto | TQ 存储、Trainer 调度 |
| Worker/Engine、模型与并行后端 | 13 Worker/Engine | Ray dispatch、rollout 服务、跨组权重传输 |
| 请求、KV、sleep/abort、PD 与服务版本 | 14 Rollout runtime | Agent 语义、权重 wire 协议 |
| reward shaping、advantage、loss 与 mask | 15 RL 算法 | 调度、存储、后端执行 |
| TQ key/tag、存储、延迟物化与 snapshot API | 16 TransferQueue | AgentLoop 行为、组合恢复顺序 |
| AgentLoop、多轮工具调用、RewardLoop | 18 Agent/Reward runtime | LLM server 内部、TQ 存储、advantage |
| actor 到 rollout 的 full/delta 权重发布 | 21 Weight publication | 训练 checkpoint、rollout 服务策略 |
| 训练 checkpoint 与恢复一致性 | 23 Training recovery | actor 到 rollout 在线权重发布 |
| 指标口径、实验控制与优化决策顺序 | 30 优化指南 | 各机制的重复实现说明 |

## 4. 静态状态所有权

| 状态 | 运行时 owner | 文档 owner |
|---|---|---|
| TaskRunner 选择、trainer mode 与系统组成 | `TaskRunnerV1`、Hydra config | 01；操作入口由 02 承接 |
| Ray actor、placement group、dispatch/collect | `RayResourcePool`、`RayWorkerGroup`、`register` | 11 |
| 本地 tensor/non-tensor/meta batch | `DataProto`、`TensorDict` | 12 |
| prompt、turn、tool result、response mask、score | `AgentLoop*`、`RewardLoop*` | 18 |
| prompt/trajectory key、tag、field 与存储 | TransferQueue、`KVBatchMeta`、`tqbridge` | 16 |
| 参数、优化器、并行布局与 backend context | Worker、Engine | 13 |
| request、KV cache、服务睡眠态、model-visible version | `LLMServer*`、rollout adapter | 14 |
| publication topology、wire、delta baseline | `CheckpointEngine*`、rollout loader | 21 |
| 持久化 global step、模型、dataloader、TQ 与在途恢复语义 | Trainer checkpoint | 23 |
| 算法张量及 reward/advantage/loss 语义 | PPO algorithm functions and registries | 15 |

## 5. 代表性动态生命周期

| 生命周期 | 唯一 owner | 完成定义 |
|---|---|---|
| 默认 V1 sync global step | 10 | 从 TaskRunner 初始化到下一版本 rollout 可见，字段和状态转移完整 |
| 稳定 V1 colocate/separate async | 17 | 覆盖 ReplayBuffer、staleness/refill、old policy 与 GPU lending |
| Experimental fully async | 22 | 独立 TaskRunner、MessageQueue、版本窗口、partial rollout 和动态调度完整 |
| V0 legacy Ray trainer | 20 | 只描述当前提交仍存在的 V0 路径，并明确其 legacy 边界 |
| 单条 agent trajectory 与 reward | 18 | prompt 到多轮/工具/打分/训练字段的状态转移完整 |
| 单次 actor 到 rollout 权重发布 | 21 | pause、export、transport、apply、verify、resume 与失败边界完整 |
| 一次保存、崩溃与恢复 | 23 | 按模式列明保存与未保存状态、恢复顺序和不可证明的原子性 |

## 6. 页面蓝图

### 6.1 导览与主线

#### `01_verl_architecture_overview_analysis.md` — Overview，结构性重写

- thesis：verl 是多种状态 owner 通过明确边界协作的 RL runtime，而不是一个 driver 函数。
- reader question：系统由哪些组件构成，状态在哪里，V1/V0/experimental 如何分流？
- owns：能力地图、静态 ownership、入口路由、live/legacy 边界、顶层不变量。
- excludes：TQ、AgentLoop、算法、权重、恢复等子系统状态机。
- evidence：`verl/trainer/main_ppo.py`、`verl/trainer/ppo/v1/trainer_base.py`、
  各 owner 的抽象入口。
- dependencies：所有静态 owner 页；实现时最后反向收敛。
- visuals：静态 ownership flowchart；动态模式路由只保留第二张简图。
- completion test：静态结构先于动态运动；每个状态只有一个深潜链接。

#### `02_verl_quickstart_guide.md` — Guide，局部更新

- thesis：用最小 GRPO 样例建立可复核基线，再按资源和目标选择 mode/backend。
- reader question：如何运行、观察并安全修改第一组配置？
- owns：命令、样例、Hydra override、首次运行检查与选择入口。
- excludes：内部状态机和算法推导。
- evidence：`examples/grpo_trainer/run_qwen3_4b_fsdp.sh`、
  `verl/trainer/config/ppo_trainer.yaml`、`docs/start/quickstart.rst`。
- dependencies：01、10、30。
- visuals：配置决策表；不新增没有信息增益的架构图。
- completion test：命令和默认值在冻结提交可定位；机制均链接 owner。

#### `10_verl_end_to_end_iteration_analysis.md` — Deep Dive，边界重构

- thesis：默认 V1 sync 由固定 PPO skeleton 与少量 mode hooks 组成。
- reader question：一个 global step 以什么顺序改变哪些 live state？
- owns：sync 启动、采样、计算、更新、发布的动态顺序。
- excludes：AgentLoop、TQ、算法、权重、checkpoint 的内部机制。
- evidence：`verl/trainer/main_ppo.py`、`trainer_base.py`、`trainer_sync.py`。
- dependencies：11–18、21、23。
- visuals：单条 sync sequence diagram。
- completion test：每个阶段列出输入、owner、输出和下一状态；没有重复深潜。

### 6.2 共享静态机制

#### `11_verl_single_controller_analysis.md` — Deep Dive，当前基线重写

- thesis：single-controller 是 V0 和 V1 共用的 Ray/SPMD 控制基础设施，不等于 V0 trainer。
- reader question：一次 group method call 如何绑定、dispatch、remote execute 和 collect？
- owns：Worker、WorkerGroup、ResourcePool、decorator、Ray backend。
- excludes：Trainer 生命周期、DataProto 语义、Engine 计算。
- evidence：`verl/single_controller/` 及 V1、rollout、CheckpointEngine 的当前 imports。
- dependencies：12 仅用于数据 dispatch 边界。
- visuals：call → bind → dispatch → execute → collect flowchart。
- completion test：全部定位属于当前提交；明确哪些能力仍 live。

#### `12_verl_dataproto_analysis.md` — Deep Dive，当前基线重写

- thesis：DataProto 是当前共享的本地批容器，但不是 V1 controller-worker 的唯一传输协议。
- reader question：tensor、non-tensor 和 meta 如何保持 batch 对齐并跨本地边界转换？
- owns：结构、不变量、转换、索引、concat/chunk、padding、Future 与序列化成本。
- excludes：TQ key/storage、Trainer 生命周期。
- evidence：`verl/protocol.py`、V1 局部物化点、single-controller dispatch。
- dependencies：11、16。
- visuals：三层数据结构表；仅在状态关系需要时保留 flowchart。
- completion test：删除历史 caller 假设与非仓根定位；当前调用点可验证。

#### `13_verl_workers_engine_analysis.md` — Deep Dive，边界收紧

- thesis：Worker 定义 RPC 粒度，Engine 定义模型、优化器和并行后端语义。
- reader question：控制请求如何落到具体训练 backend，哪些状态不能跨层外推？
- owns：Worker/Engine 分层、context、后端矩阵、offload 与 export 语义。
- excludes：Ray dispatch、rollout 服务、CheckpointEngine wire。
- evidence：`verl/workers/engine/base.py`、`engine_workers.py`、backend implementations。
- dependencies：11、12。
- visuals：静态分层图与支持矩阵。
- completion test：后端矩阵与当前 registry 一致；publication 只保留边界。

#### `14_verl_rollout_runtime_analysis.md` — Deep Dive，由旧 14 改名并重写

- thesis：rollout runtime 拥有请求与 KV 生命周期；权重发布只是改变其服务状态的外部会话。
- reader question：请求如何路由、生成、abort、sleep/resume，PD 和 model version 如何约束服务？
- owns：LLM server/client/manager、adapter、request/KV/prefix cache、PD、partial resume 边界。
- excludes：AgentLoop 语义、full/delta transport 与 training checkpoint。
- evidence：`verl/workers/rollout/llm_server.py`、vLLM/SGLang adapters、PD replica。
- dependencies：11、13、18、21。
- visuals：请求/KV 服务状态机。
- completion test：旧页 full/delta 章节迁出；仅保留 publication 的服务侧前后置条件。

#### `15_verl_rl_algorithms_analysis.md` — Deep Dive，边界收紧

- thesis：算法注册表改变 reward/advantage/loss 语义，不复制运行时架构。
- reader question：给定 estimator 和 loss，字段、mask、基线与聚合如何改变？
- owns：reward shaping、KL、advantage、policy/value loss 与不变量。
- excludes：调度、数据存储和 backend execution。
- evidence：PPO core algorithms、algorithm registry 与 config。
- dependencies：12。
- visuals：保留有信息量的算法映射和公式；不新增架构图。
- completion test：公式、字段和注册名可定位；不复述 trainer 流程。

#### `16_verl_v1_transfer_queue_analysis.md` — Deep Dive，边界收紧

- thesis：TQ 解耦控制引用和数据存储，但不消除 DataProto，也不自动提供 freshness/durability。
- reader question：key、tag、field 和 materialization 怎样连接 controller 与 worker？
- owns：KVBatchMeta、prompt/trajectory schema、bridge、backend 与 snapshot API 表面。
- excludes：AgentLoop 行为、组合恢复、ReplayBuffer 策略。
- evidence：`agent_loop_tq.py`、`transferqueue_utils.py`、TQ config。
- dependencies：12、18、23。
- visuals：key/tag lifecycle flowchart。
- completion test：TQ 与 DataProto、AgentLoop、recovery 的边界各有一句明确 exclusion。

#### `18_verl_agent_loop_reward_runtime_analysis.md` — Deep Dive，新增

- thesis：AgentLoop 是可替换的 per-trajectory coroutine runtime，RewardLoop 可在 trajectory 完成前后异步接入打分。
- reader question：prompt 如何经过单轮或工具多轮、postprocess 和 reward 路由形成训练字段？
- owns：AgentLoop registry/base/worker/manager、ToolAgentLoop、response mask、postprocess、RewardLoop manager/model。
- excludes：LLM server 内部、TQ storage、advantage/reward shaping。
- evidence：`verl/experimental/agent_loop/`、`experimental/reward_loop/`、V1 wiring。
- dependencies：12、14、16。
- visuals：trajectory sequence diagram，展示 generate/tool/postprocess/score 边界。
- completion test：单轮、多轮、工具、已有 reward、异步 reward、错误边界和扩展点均覆盖。

### 6.3 模式、发布与恢复

#### `17_verl_v1_async_trainer_analysis.md` — Deep Dive，边界重构

- thesis：stable async 通过 ReplayBuffer、版本/旧策略和资源状态机控制重叠代价。
- reader question：colocate_async 与 separate_async 如何保持可训练 batch 和可解释版本？
- owns：mode hooks、ReplayBufferAsync、staleness/refill、stable old policy、GPU lending。
- excludes：TQ internals、training recovery protocol、weight publication internals。
- evidence：`trainer_colocate_async.py`、`trainer_separate_async.py`、`replay_buffer.py`。
- dependencies：14、16、18、21、23。
- visuals：双时间线状态图；不重复 checkpoint sequence。
- completion test：两种 stable mode 的库存、版本和资源状态完整，checkpoint 只留调用边界。

#### `20_verl_ray_trainer_analysis.md` — Legacy Deep Dive，当前基线复核

- thesis：V0 是仍可显式选择的 driver-centric legacy lifecycle，而不是当前默认架构。
- reader question：当前提交中的 V0 路径仍怎样串联 rollout、reward、update 和保存？
- owns：当前 V0 lifecycle 与 legacy 边界。
- excludes：共享 Ray substrate、DataProto 方法、算法和 publication 深潜。
- evidence：`verl/trainer/main_ppo.py` V0 route、`verl/trainer/ppo/ray_trainer.py`。
- dependencies：11、12、15、21、23。
- visuals：一张 legacy sequence diagram。
- completion test：历史提交逐行叙述被当前代码重验；无法证明的历史细节删除。

#### `21_verl_weight_publication_analysis.md` — Deep Dive，由旧 21 改名

- thesis：权重发布由 Engine semantic export、CheckpointEngine transport 和 rollout loader apply 三段组成。
- reader question：full 和 delta_sharded 如何让新 actor 参数成为 rollout 可见版本？
- owns：publication topology/session、wire、dense seed、shard-local delta、loader、校验与支持矩阵。
- excludes：training checkpoint、rollout routing/KV policy、常规 Engine compute。
- evidence：`verl/checkpoint_engine/`、Engine exporters、rollout loaders。
- dependencies：13、14。
- visuals：三段 owner flowchart 与 delta state machine。
- completion test：full/delta 所有机制只在本页；明确 CheckpointEngine 不是恢复 checkpoint。

#### `22_verl_fully_async_dynamic_schedule_deepdive.md` — Experimental Deep Dive，边界重构

- thesis：experimental fully async 是独立双循环系统，以 completion order 和版本窗口换取持续重叠。
- reader question：Rollouter、Trainer、MessageQueue 和 dynamic scheduler 如何只在版本边界相遇？
- owns：独立入口、MQ、admission、version、partial rollout、dynamic resources。
- excludes：stable V1 async、通用 AgentLoop、跨模式 recovery owner。
- evidence：`verl/experimental/fully_async_policy/`。
- dependencies：14、18、21、23。
- visuals：双循环与版本边界 flowchart。
- completion test：stable/experimental 边界清楚；checkpoint 仅保留本路径特有缺口并链接 23。

#### `23_verl_training_checkpoint_recovery_analysis.md` — Deep Dive，新增

- thesis：训练恢复是模型、优化器、dataloader、TQ/MQ 和在途工作的组合状态协议，不能由单个 checkpoint 文件证明一致。
- reader question：每种模式保存了什么、没保存什么，崩溃后按什么顺序恢复，哪些语义不可证明？
- owns：V1 save/load、callback durability、TQ composition、V0/experimental 对照和 crash consistency。
- excludes：CheckpointEngine 在线权重发布、Engine backend 具体文件布局。
- evidence：V1 `_save_checkpoint`/`_load_checkpoint`、checkpoint callback、V0/fully async save/load。
- dependencies：12、13、16、17、20、22。
- visuals：saved-state inventory table 与 recovery sequence diagram。
- completion test：按 mode 列出 saved/unsaved/reissued/dropped state；CheckpointEngine 歧义消除。

#### `30_verl_optimization_analysis.md` — Decision Guide，条件保留并大幅瘦身

- thesis：性能决策必须共同核算吞吐、显存、sample freshness、publication pause 和 recovery tax。
- reader question：给定瓶颈，应按什么顺序测量、选择机制并验证 correctness？
- owns：指标口径、控制变量、选择矩阵、实验和回归检查顺序。
- excludes：各机制实现说明。
- evidence：`verl/utils/tracking.py` 和各 owner 页的可观测输出。
- dependencies：13、14、17、21、22、23。
- visuals：决策矩阵优先于流程图。
- completion test：每项建议链接唯一 owner；若删去重复机制后没有独立决策价值，则整页删除并修复入链。

## 7. 现有内容处置矩阵

| 页面 | 保留 | 重写/迁移 | 删除 |
|---|---|---|---|
| 01 | 顶层命题、不变量 | 静态 ownership 先于动态路由 | 子系统机制复述 |
| 10 | sync 主顺序 | owner/输入/输出边界 | AgentLoop/TQ/算法/weight/checkpoint 深入说明 |
| 11 | Worker/Group/dispatch 核心 | 当前 V1/CE/rollout 使用关系与当前行号 | “纯 V0 控制面”和 full DataProto 必经 driver 假设 |
| 12 | 数据结构和核心不变量 | 当前 V1 局部角色与仓根定位 | 旧 caller 全集、冗长方法速查、历史唯一契约结论 |
| 14 | request/KV/PD/sleep/version | rollout runtime thesis | full/delta transport、loader 与 CE 状态机 |
| 17 | stable async 独有机制 | checkpoint 改成模式边界链接 | 跨模式恢复深潜 |
| 20 | V0 reader question | 当前提交重新验证 | 仅历史提交成立的逐行叙述 |
| 21 | 现有三段 owner、full/delta、支持矩阵 | 文件名和 training checkpoint exclusion | 其它页重复内容吸收后不再并存 |
| 22 | experimental 独有机制 | recovery 改为本路径缺口并链接 23 | 通用 AgentLoop 与跨模式 checkpoint 复述 |
| 30 | 指标口径、决策顺序 | 机制段落改为选择矩阵 | 13/14/17/21/22/23 的实现复述 |
| index | 页面入口表 | 新文件名和一句话职责 | 架构图、阅读路线、更新记录、证据规则复述 |

## 8. 文件操作

- rename：`14_verl_rollout_resharding_analysis.md` → `14_verl_rollout_runtime_analysis.md`。
- rename：`21_verl_delta_weight_sync_deepdive.md` → `21_verl_weight_publication_analysis.md`。
- add：`18_verl_agent_loop_reward_runtime_analysis.md`。
- add：`23_verl_training_checkpoint_recovery_analysis.md`。
- update：其余 Verl 页面、Verl `index.md`、父级 `index.md`、`wiki/changelog.md`。
- radar：源码基线没有前移，无需修改 `docs/radar/watchlist.yaml`；实现时仍需核对现有值。

## 9. 实施顺序

1. 静态基础：11、12、13、16、18、14、21、23、15。
2. 动态路径：10、17、22、20。
3. 反向收敛：01、02、30。
4. 文件改名、全库入链、Verl index、父级 index、changelog。
5. 运行全库质量门并只检查/暂存本任务路径。

## 10. 质量门

- `python tools/check_links.py --strict`：broken、ambiguous、bare_index 均为 0。
- `python tools/check_math.py --changed --strict`：本次变更公式错误和警告均为 0。
- `git diff --check`：无 whitespace error。
- 全部内容页具有 3–7 条精选 `## Related Pages`；index 豁免。
- 所有源码引用使用 repo-root-relative path 和冻结提交行号。
- 所有 Mermaid 块逐块通过 parser-trap 人工检查；能使用现有渲染工具时实渲。
- `rg` 确认旧文件名入链归零。
- `rg` 确认 `CheckpointEngine` 与 training checkpoint 的 owner 不再混淆。
- 覆盖矩阵人工复核：能力、live state、代表性 lifecycle 各有且仅有一个权威页。
- `git diff --name-only` 不得包含现有 Megatron 用户改动或 Verl 源码侧文件。
