---
title: "verl 优化地图：吞吐、显存、新鲜度与恢复性的联合预算"
---

# verl 优化地图：吞吐、显存、新鲜度与恢复性的联合预算

> **代码基准**：verl `main` @ `254a23edc62f25ebfae626e3932ae285d6f86009`（2026-08-28）
> **最后更新**：2026-08-28 · **系列**：verl RLHF 框架源码级分析（见 [[verl/index]]）
>
> **核心结论**：verl 的性能不是单一 MFU 问题，而是四个预算的乘积：生成/训练重叠提高吞吐，offload 与 parallel backend 控制显存，staleness/rollout correction 保持样本可用，checkpoint 与队列状态决定优化是否可恢复。任何“加速”如果依赖更多旧样本、静默丢样或不可恢复在途状态，都必须把代价一起记账。

---

## 1. 统一观测量

先把一次 step 分解为：

```text
data fetch and submit
generation wait
reward and policy state
actor critic compute
weight publish
checkpoint validation logging
```

V1 基类为 sample/reward/old-logp/ref/value/advantage/update 分别计时（`verl/trainer/ppo/v1/trainer_base.py:540-590`）。优化实验至少同时记录：

| 维度 | 观测量 | 防止的误判 |
|---|---|---|
| 吞吐 | 有效 response token/s、generation wait、actor MFU | 把 padding/丢样当加速 |
| 显存 | train peak、rollout KV、offload bytes/time | 只看 OOM 是否消失 |
| 新鲜度 | prompt age、min/max policy version、IS/rejection | 用更旧数据换吞吐 |
| 发布 | abort/drain、send/apply、cache resume | 忽略权重同步停顿 |
| 恢复 | dataloader、TQ/MQ、in-flight prompt | checkpoint 后改变样本集合 |

RL-Insight/Tracking 已能汇聚 scalar、state/span、agent session 与 Prometheus endpoint（`verl/utils/tracking.py:224-320,357-458`；`verl/workers/rollout/llm_server.py:594-609`）。指标存在不代表语义正确，仍需按上述阶段定义分母与版本。

---

## 2. 第一选择：匹配生成与训练节拍

稳定 V1 提供三种模式：

| 模式 | 重叠方式 | 主要成本 |
|---|---|---|
| `sync` | 不跨步重叠 | 长尾 generation 直接形成 barrier |
| `colocate_async` | warmup/积压隐藏生成，但训练时收回同池 GPU | abort/resume、partial trajectory、staleness |
| `separate_async` | standalone rollout 常驻 | 更多固定资源、权重发布与 off-policy 控制 |

三种模式共享 `PPOTrainer` pipeline，差异集中在 hook（`verl/trainer/ppo/v1/trainer_sync.py:25-42`；`trainer_colocate_async.py:25-59`；`trainer_separate_async.py:43-398`）。选择顺序应是：先用 sync 建 correctness/吞吐基线，再看 generation wait 占比；只有等待显著时，才用 async 换 overlap。

async ReplayBuffer 的 `drop`/`wait`、DAPO/failure refill、streaming fetch 与 checkpoint recovery 见 [[17_verl_v1_async_trainer_analysis]]。吞吐结果必须同时报告 dropped/refilled group 与 sample age。

---

## 3. 资源动态化：稳定 lending 与 experimental scheduler

separate async 可把空闲 hybrid trainer GPU 临时借给 generation。决策比较预计补样收益与最近双向切换成本，并按库存阈值收回；默认关闭（`verl/trainer/config/ppo_trainer.yaml:254-280`；`verl/trainer/ppo/v1/trainer_separate_async.py:261-383`）。

experimental fully async 则让长期运行的 Rollouter/Trainer 通过 MessageQueue 解耦，并用动态 controller 在 `STANDALONE_ONLY` 与 `HYBRID_ACTIVE` 间切换（`verl/experimental/fully_async_policy/dynamic_schedule/dynamic_resource_controller.py:15-159`）。默认 policy 根据等待样本、积压、收益与切换成本调整 deactivate ratio（`verl/experimental/fully_async_policy/dynamic_schedule/default_policy.py:38-235`）。

两者不能混写：稳定 lending 发生在 V1 `separate_async` trainer 内，experimental scheduler 属于独立 TaskRunner。后者的队列会在满时丢最老样本，checkpoint 也不保存全部在途工作；完整边界见 [[22_verl_fully_async_dynamic_schedule_deepdive]]。

---

## 4. 训练 Engine：先消除无效计算，再考虑 offload

当前 Engine 选择维度是 model type、backend、device/vendor（`verl/workers/engine/base.py:351-399`）。常见优化层次：

1. remove padding、sequence packing/dynamic batch，减少 padding FLOPs；
2. gradient checkpoint/recompute，以计算换 activation memory；
3. TP/CP/EP/FSDP/Megatron layout，匹配模型结构和互联；
4. param/optimizer offload，以 host/device 搬运换显存；
5. backend-specific fusion/Turbo。

FSDP Turbo 在 CUDA/NPU language model 上接管模型包装、offload 与 CP，且禁止同时启用 verl Ulysses（`verl/workers/engine/fsdp/fsdp_turbo_impl.py:25-70`）。TorchTitan 后端委托 FSDP2/TP/CP/EP、optimizer/checkpoint，但不支持 PP（`docs/workers/torchtitan_workers.rst:6-71`）。后端矩阵与 offload 语义见 [[13_verl_workers_engine_analysis]]。

当前用户配置面没有独立 `grad_offload`；保留的是 param/optimizer offload（`verl/workers/config/engine.py:89-153`）。所以旧的“三个独立开关”调优清单已经过期。

---

## 5. Loss 聚合也会影响性能实验可信度

micro-batch、DP size 或变长 sequence 的调整不能改变目标函数。`agg_loss` 使用 global token/batch 信息与 `dp_size` 保证 FSDP/Megatron 下的梯度尺度（`verl/trainer/ppo/core_algos.py:1140-1206`）；新增 `token-sum` 也显式补偿 DP 平均。

因此性能实验必须同时核对 loss 数值/梯度不变量。若缩小 micro-batch 后 loss 变化，看到的吞吐曲线不是同一训练目标。actor/critic 的全局归一化与测试见 [[15_verl_rl_algorithms_analysis]]。

---

## 6. 权重发布：流量、内存与暂停时间的交换

| backend | steady-state payload | 额外资源 | 适用边界 |
|---|---|---|---|
| colocated `naive` | full tensors | 同池 IPC/adapter | 简单 sync、同池 |
| disaggregated full | full named tensors | 临时通信拓扑 | vLLM/SGLang |
| `delta_sharded` | sparse positions + values | pinned host snapshot、diff/gather | 仅 SGLang、受 Engine 支持矩阵约束 |

full 与 delta 的共同 lifecycle 仍要 abort、释放 KV、send/receive、apply、finalize、resume（`verl/checkpoint_engine/base.py:381-506`）。`delta_sharded` 第一次仍发送 dense seed，之后才用本地 shard snapshot 做 sparse steady state（`verl/checkpoint_engine/delta_checkpoint_engine.py:271-359,578`）。

delta 不是免费压缩：它用 host snapshot、diff kernel、位置编码和 checksum 换网络流量。模型更新稠密时，payload 优势会下降。支持矩阵与失败边界见 [[21_verl_delta_weight_sync_deepdive]]；rollout full/PD 生命周期见 [[14_verl_rollout_resharding_analysis]]。

---

## 7. Rollout：KV、PD 与 prefix-cache 观测

vLLM PD 支持一 prefill + 多 decode 与非对称 TP，但当前只允许特定 transport、单节点和 DP=PP=1（`verl/workers/rollout/vllm_rollout/vllm_pd_replica.py:36-100`）。PD 优化的是 prefill/decode 与 KV 传输，不会减少 actor 权重发布量；SGLang delta 又显式拒绝 PD 组合（`verl/workers/rollout/sglang_rollout/sglang_rollout.py:384-388`）。

最终基线新增 `num_cached_tokens` 的可见性：vLLM server 把 prefix-cache hit 写入 TokenOutput，FullyAsync client 在 partial resume 中保留首次 prefill 的 hit count（`verl/workers/rollout/vllm_rollout/vllm_async_server.py:647-690`；`verl/workers/rollout/llm_server.py:406-486`）。这让 cache 命中可观测，但不能用单一命中率推导端到端加速，仍要结合 prefill/decode 时间与 abort/retry。

---

## 8. checkpoint：吞吐优化的恢复税

稳定 V1 async 只有在 TransferQueue `>=0.1.9` 且具备 save/load API 时保存 TQ snapshot（`verl/trainer/ppo/v1/trainer_base.py:106-116,942-950`）。恢复时 finished 轨迹可复用，pending/running prompt 的旧局部输出会被清除并重新派发（`verl/trainer/ppo/v1/trainer_base.py:843-887`）。

checkpoint callback 可以在 driver 生命周期中调度保存，但 callback 返回不等价于远端产物已经 durable（`verl/trainer/ppo/checkpoint_callback.py:14-82`）。experimental fully async 则明确无法无损恢复 pending/active/result/MQ 中的全部样本（`verl/experimental/fully_async_policy/fully_async_rollouter.py:655-660`；`fully_async_trainer.py:917-978`）。

优化方案若延长 checkpoint 周期，应把重算生成、丢失 in-flight work 与恢复后样本集合变化计入成本。

---

## 9. 一套可复用的决策顺序

1. 用 V1 sync 固定算法、数据、有效 token 与 checkpoint，建立 correctness baseline。
2. 分解 generation、train compute、weight publish、checkpoint 时间。
3. padding 高先做 dynamic batch/remove padding；显存高再选 recompute/offload/backend。
4. generation wait 高再选 colocate/separate async，并同时记录 staleness/refill。
5. weight publish 高再比较 full 与 `delta_sharded`，先检查 rollout/Engine 支持矩阵。
6. 固定资源比例长期失配，才启用 lending 或 experimental dynamic scheduler。
7. 每次变更复跑 loss/gradient、版本可见性与恢复实验，而不只比较 reward 和 wall time。

---

## Related Pages

- [[17_verl_v1_async_trainer_analysis]] —— 稳定 V1 overlap、staleness 与 lending
- [[22_verl_fully_async_dynamic_schedule_deepdive]] —— experimental 队列与动态调度
- [[13_verl_workers_engine_analysis]] —— Engine、offload 与后端矩阵
- [[14_verl_rollout_resharding_analysis]] —— rollout、full 更新与 PD
- [[21_verl_delta_weight_sync_deepdive]] —— delta 发布机制与成本
- [[15_verl_rl_algorithms_analysis]] —— loss/gradient 不变量
- [[verl/index]] —— 系列导航
