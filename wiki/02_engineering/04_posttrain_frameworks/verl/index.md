---
title: "verl 源码级分析"
---

# verl 源码级分析

> **当前代码基准**：volcengine/verl `main` @ `254a23edc62f25ebfae626e3932ae285d6f86009`（2026-08-28 10:08 +08）
> **最后复核**：2026-08-31 · **规模**：16 篇内容页 + 本索引
> **入口**：[[02_engineering/04_posttrain_frameworks/index|后训练框架]]

当前分析域按两层组织：共享能力解释稳定机制，生命周期页只解释各 trainer 模式如何组合这些能力。默认主线是 V1 sync；stable V1 async、experimental fully async 和 V0 legacy 各自拥有独立生命周期页。

## 入口与决策

| 页面 | 唯一职责 |
|---|---|
| [[01_verl_architecture_overview_analysis]] | 共享能力、生命周期关系与系统不变量 |
| [[02_verl_quickstart_guide]] | 从仓内 Qwen3-4B FSDP 样例完成第一次 V1 sync 运行 |
| [[30_verl_optimization_analysis]] | 性能诊断、机制选型顺序与验收标准 |

## 动态生命周期

| 页面 | 唯一职责 |
|---|---|
| [[10_verl_end_to_end_iteration_analysis]] | 默认 V1 sync 一步训练的端到端顺序 |
| [[17_verl_v1_async_trainer_analysis]] | stable V1 async、ReplayBuffer、旧度与 GPU lending |
| [[22_verl_fully_async_dynamic_schedule_deepdive]] | experimental fully async、MQ、staleness 与动态资源 |
| [[20_verl_ray_trainer_analysis]] | 当前 V0 legacy `RayPPOTrainer` 生命周期 |

## 共享静态机制

| 页面 | 唯一职责 |
|---|---|
| [[11_verl_single_controller_analysis]] | `Worker`、`WorkerGroup`、资源池与 dispatch/collect |
| [[12_verl_dataproto_analysis]] | `DataProto`、`TensorDict` 与本地批不变量 |
| [[16_verl_v1_transfer_queue_analysis]] | TransferQueue key、tag、field、failure state 与 bridge |
| [[18_verl_agent_loop_reward_runtime_analysis]] | AgentLoop、tool loop、trajectory 与 RewardLoop |
| [[13_verl_workers_engine_analysis]] | Worker/Engine 边界、训练 backend、并行与 offload |
| [[14_verl_rollout_runtime_analysis]] | rollout request、KV、sleep、abort、PD 与 partial request |
| [[15_verl_rl_algorithms_analysis]] | estimator、policy loss、mask 与全局归一化 |
| [[21_verl_weight_publication_analysis]] | actor 到 rollout 的 full 与 `delta_sharded` 在线发布 |
| [[23_verl_training_checkpoint_recovery_analysis]] | 跨重启训练 checkpoint、恢复组合与失败窗口 |

## 推荐阅读路线

| 目标 | 顺序 |
|---|---|
| 第一次运行 | [[01_verl_architecture_overview_analysis]] → [[02_verl_quickstart_guide]] → [[10_verl_end_to_end_iteration_analysis]] |
| 异步与旧样本 | [[16_verl_v1_transfer_queue_analysis]] → [[17_verl_v1_async_trainer_analysis]] → [[22_verl_fully_async_dynamic_schedule_deepdive]] |
| Agent 与奖励 | [[18_verl_agent_loop_reward_runtime_analysis]] → [[15_verl_rl_algorithms_analysis]] → [[10_verl_end_to_end_iteration_analysis]] |
| 后端与显存 | [[13_verl_workers_engine_analysis]] → [[14_verl_rollout_runtime_analysis]] → [[30_verl_optimization_analysis]] |
| 权重与恢复 | [[21_verl_weight_publication_analysis]] → [[23_verl_training_checkpoint_recovery_analysis]] |
| 维护 legacy V0 | [[11_verl_single_controller_analysis]] → [[12_verl_dataproto_analysis]] → [[20_verl_ray_trainer_analysis]] |

## 基线与更新规则

全部内容页统一以提交 `254a23edc62f25ebfae626e3932ae285d6f86009` 为冻结基线，源码定位使用 Verl 仓库根目录相对的 `file:line`。上游基线变化时，先复核入口路由、V1 trainers、TransferQueue、Agent/Reward runtime、Engine、rollout、CheckpointEngine 和 fully async，再更新本索引与变更日志。
