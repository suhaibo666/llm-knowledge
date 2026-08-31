---
title: "verl 性能决策指南：先定位预算，再选择机制"
---

# verl 性能决策指南：先定位预算，再选择机制

> **代码基准**：verl `main` @ `254a23edc62f25ebfae626e3932ae285d6f86009`
> **最后复核**：2026-08-31
> **概念所有权**：本页唯一负责跨机制的性能诊断、选型顺序和验收标准；机制实现以各 owner 页为准。

## 核心判断

Verl 的调优目标不是单独提高 MFU，而是在五个预算之间取得可验证的平衡：**生成等待、训练计算、设备内存、样本新鲜度和恢复成本**。吞吐提高若来自更多旧样本、隐式丢样、目标函数变化或不可恢复的在途状态，就不是等价优化。

所以正确顺序是：先固定正确性基线，按阶段测量，再只改变一个预算；最后用 loss、policy version 和恢复实验证明优化没有改变训练语义。

## 1. 先建立可比较的基线

默认 V1 sync 的公共 pipeline 已把 sample、reward、old/ref log-prob、value、advantage 和 update 分开计时（`verl/trainer/ppo/v1/trainer_base.py:540-590`）。第一轮基线至少固定：

- 相同数据切片、随机种子、模型和算法配置；
- 相同有效 prompt 数、trajectory 数和 response token 数；
- 相同 loss aggregation、KL 与 rollout correction；
- 相同 checkpoint 起点和验证频率；
- `trainer.use_v1=true`、`trainer.v1.trainer_mode=sync` 作为对照生命周期。

Tracking 能收集 scalar、state、span 和 Prometheus 指标（`verl/utils/tracking.py:224-320`、`verl/utils/tracking.py:357-458`），但指标存在不代表分母正确。吞吐应以有效 token 或有效 trajectory 计数，不能把 padding、被丢弃或失败重试的样本算作收益。

## 2. 用症状定位预算

| 观测到的症状 | 首先补充的证据 | 优先检查的 owner | 不应先做的事 |
|---|---|---|---|
| generation wait 占 step 大头 | prefill、decode、排队、长尾、cache hit | [[14_verl_rollout_runtime_analysis]]、[[18_verl_agent_loop_reward_runtime_analysis]] | 直接切 fully async |
| actor 或 critic 计算占主导 | 有效 token、padding、MFU、collective、micro-batch | [[13_verl_workers_engine_analysis]] | 先增加 rollout GPU |
| OOM 或频繁 offload | train peak、KV 占用、H2D/D2H 时间 | [[13_verl_workers_engine_analysis]]、[[14_verl_rollout_runtime_analysis]] | 同时改五个 batch 旋钮 |
| async 吞吐高但 reward 恶化 | policy age、drop/refill、IS/rejection、版本分布 | [[17_verl_v1_async_trainer_analysis]] | 只比较 wall time |
| step 尾部权重发布很长 | abort、prepare、payload、apply、resume 分段时间 | [[21_verl_weight_publication_analysis]] | 假设 delta 必然更快 |
| 重启后吞吐或 reward 突变 | dataloader、TQ/MQ、pending/running、global step | [[23_verl_training_checkpoint_recovery_analysis]] | 只验证模型权重能加载 |

只有证据能把瓶颈归入某个 owner，才进入下一节的机制选择。

## 3. 调优决策阶梯

### 3.1 先消除无效计算

优先减少 padding 和不必要的重算：remove padding、sequence packing、dynamic batch，以及与真实 token 长度匹配的 micro-batch。然后才比较 gradient checkpoint、并行布局、offload 与 backend-specific fusion。

Engine registry 按 model type、backend 和 device/vendor 选择实现（`verl/workers/engine/base.py:351-399`）。FSDP Turbo 会接管部分 offload 与 CP 语义，并拒绝同时启用 Verl Ulysses（`verl/workers/engine/fsdp/fsdp_turbo_impl.py:25-70`）；TorchTitan 当前支持 FSDP2、TP、CP、EP 等组合，但 PP 仍不支持（`docs/workers/torchtitan_workers.rst:6-71`）。因此 backend 不是一个脱离模型、设备和并行布局的通用加速开关。

### 3.2 显存不足时按代价递增处理

建议按下列顺序实验，每次只改变一层：

1. 降低 rollout KV 预算或序列长度；
2. 缩小 actor、ref、critic 的 micro-batch，保持全局 batch 与 loss aggregation 不变；
3. 打开 activation recompute；
4. 选择 param 或 optimizer offload，并显式记录搬运时间；
5. 重新设计 TP、CP、EP、FSDP 或 Megatron layout；
6. 最后再换 backend-specific Turbo 或融合实现。

当前通用 `EngineConfig` 暴露的是 param 与 optimizer offload，不存在可独立配置的 `grad_offload`（`verl/workers/config/engine.py:89-153`）。使用旧调优清单时必须先核对当前配置面。

### 3.3 生成等待显著时再引入 overlap

| 选择 | 何时考虑 | 必须一起计量的代价 |
|---|---|---|
| V1 sync | correctness 基线或生成等待不高 | barrier 与长尾 |
| `colocate_async` | 同池 GPU，希望用预生成隐藏部分等待 | abort、partial trajectory、staleness、resume |
| `separate_async` | rollout 可独立常驻，生成是稳定瓶颈 | 固定资源、replay 库存、权重发布、off-policy |
| experimental fully async | 需要长期 producer-consumer 解耦并接受实验性边界 | MQ 丢弃、policy version、动态资源、非无损恢复 |

stable V1 三模式共享 pipeline，差异集中在 lifecycle hook（`verl/trainer/ppo/v1/trainer_sync.py:25-42`、`verl/trainer/ppo/v1/trainer_colocate_async.py:25-59`、`verl/trainer/ppo/v1/trainer_separate_async.py:43-398`）。experimental fully async 则是独立 TaskRunner，不应作为 `trainer_mode` 的直接横向替换。

### 3.4 固定资源长期失配时再动态化

`separate_async` 可以把空闲的 hybrid trainer GPU 临时借给 generation；决策需要比较预计补样收益与最近双向切换成本，且默认关闭（`verl/trainer/config/ppo_trainer.yaml:254-280`、`verl/trainer/ppo/v1/trainer_separate_async.py:261-383`）。

experimental fully async 的 dynamic controller 则在 `STANDALONE_ONLY` 与 `HYBRID_ACTIVE` 状态之间改变资源比例（`verl/experimental/fully_async_policy/dynamic_schedule/dynamic_resource_controller.py:15-159`）。两者解决相似的闲置问题，却属于不同生命周期，启用、恢复和故障边界不能混用。

### 3.5 权重发布占主导时再比较 full 与 delta

colocated `naive`、disaggregated full 和 `delta_sharded` 共享暂停请求、准备、传输、apply 与恢复服务的生命周期，但 payload 和资源代价不同（`verl/checkpoint_engine/base.py:381-506`）。`delta_sharded` 第一次仍需 dense seed，steady state 才发送稀疏变化，并额外消耗 host snapshot、diff、gather 与校验资源（`verl/checkpoint_engine/delta_checkpoint_engine.py:271-359`、`verl/checkpoint_engine/delta_checkpoint_engine.py:578-637`）。

因此只有当 full payload 确实是瓶颈、模型更新足够稀疏、SGLang 与训练 Engine 均落入支持矩阵时，delta 才值得实验。协议、支持矩阵和失败后不可直接重试的窗口见 [[21_verl_weight_publication_analysis]]。

## 4. 三个不能被性能优化破坏的不变量

### 4.1 目标函数不随并行切分变化

`agg_loss` 使用 global token 或 batch 信息和 `dp_size` 校正梯度尺度（`verl/trainer/ppo/core_algos.py:1140-1206`）。改变 micro-batch、DP 或变长策略后，必须验证 loss 与梯度仍对应同一聚合语义；否则吞吐曲线比较的是两个训练目标。

### 4.2 rollout 只看见完整 policy version

发布前后的新请求不能混用半更新参数。每次优化 weight sync、abort/resume 或 replica 扩缩后，应记录请求使用的 policy version，并验证一次发布完成前后存在明确切点。详细版本边界见 [[14_verl_rollout_runtime_analysis]] 与 [[21_verl_weight_publication_analysis]]。

### 4.3 恢复后的样本集合变化必须显式

stable V1 async 只有在 TransferQueue 具备相应 save/load 能力时才保存 TQ snapshot；恢复时 finished 轨迹可能复用，而 pending/running prompt 会清理局部输出并重新派发（`verl/trainer/ppo/v1/trainer_base.py:843-950`）。experimental fully async 不能无损保存所有 pending、active、result 与 MQ 内容（`verl/experimental/fully_async_policy/fully_async_rollouter.py:655-660`、`verl/experimental/fully_async_policy/fully_async_trainer.py:917-978`）。

延长 checkpoint 周期换取吞吐时，必须把故障后的重生成、丢失在途工作和重复消费计入成本；完整协议见 [[23_verl_training_checkpoint_recovery_analysis]]。

## 5. 每个实验都用同一张验收表

| 类别 | 必须通过的检查 |
|---|---|
| 正确性 | loss、梯度尺度、mask 和 advantage 分组与基线一致 |
| 数据 | 有效 prompt、trajectory、token、drop、refill 和 retry 均可核对 |
| 版本 | old policy 语义稳定；rollout version 有完整切点 |
| 性能 | 分阶段时间、设备峰值、offload bytes、网络 payload 有证据 |
| 恢复 | 在目标故障点实际中断并恢复，核对 step、dataloader 和在途集合 |
| 可回退 | 配置变更可单独撤销，未把多种机制绑成一个不可解释实验 |

最终报告至少同时给出吞吐收益、显存变化、新鲜度分布和恢复代价。只有四者都可解释，才算完成一次可迁移的 Verl 优化。

## Related Pages

- [[10_verl_end_to_end_iteration_analysis]] —— V1 sync 正确性与计时基线
- [[13_verl_workers_engine_analysis]] —— 训练 backend、并行布局与 offload
- [[14_verl_rollout_runtime_analysis]] —— 生成、KV、sleep、abort 与 PD
- [[17_verl_v1_async_trainer_analysis]] —— stable async、旧度与 GPU lending
- [[21_verl_weight_publication_analysis]] —— full 与 delta 发布成本
- [[22_verl_fully_async_dynamic_schedule_deepdive]] —— experimental 队列与动态资源
- [[23_verl_training_checkpoint_recovery_analysis]] —— 恢复成本与丢样边界
