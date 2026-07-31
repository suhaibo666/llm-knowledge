# D07 verl 端到端训练迭代

> **阶段**：S02
> **文档编号**：D07
> **源码基线**：verl `983cb0f24443f87b3d161fad318445130a620b07`
> **核验日期**：2026-07-27
> **结论先行**：学习 verl 应先贯通 stable `RayPPOTrainer.fit` 的同步主链，再单独研究 experimental fully async；不要把两条路径拼成一个不存在的调用链。
> **阅读导航**：[[rl_framework_comparison|上一篇 D06]] · [[slime_architecture_analysis|下一篇 D08]]

---

## 1. 稳定主链地图

```mermaid
flowchart LR
    M["main_ppo 入口"] --> T["RayPPOTrainer"]
    T --> G["Rollout generate"]
    G --> D["DataProto"]
    D --> R["Reward"]
    R --> A["Advantage"]
    A --> U["Actor update"]
    U --> W["Weight refresh"]
    W --> G
```

固定源码入口：

- [`verl/trainer/main_ppo.py:34-168`](https://github.com/verl-project/verl/blob/983cb0f24443f87b3d161fad318445130a620b07/verl/trainer/main_ppo.py#L34-L168)：Hydra main、Ray task 和 `run_ppo`；
- [`verl/trainer/ppo/ray_trainer.py:286`](https://github.com/verl-project/verl/blob/983cb0f24443f87b3d161fad318445130a620b07/verl/trainer/ppo/ray_trainer.py#L286)：`RayPPOTrainer`；
- `verl/trainer/ppo/ray_trainer.py:772`：初始化 workers；
- `verl/trainer/ppo/ray_trainer.py:1380`：`fit` 主循环。

> [!contradiction] `trainer.use_v1` 默认值在两基线间反转
> [[verl_ray_trainer_analysis]]（基线 `8a694930`）记录的状态是：`RayPPOTrainer`（`ray_trainer.py:285` 带 `@deprecated`）"目前仍是默认跑的 PPO 编排器"，因为当时 `trainer.use_v1` 默认 `false`（`trainer/config/ppo_trainer.yaml:201`），`main_ppo.py` 因此默认落到 `main_ppo_v0.TaskRunner` → 本页的 `RayPPOTrainer.fit`。
> 到本页基线 `983cb0f`，该默认值已反转为 `use_v1: true`（`trainer/config/ppo_trainer.yaml:219`），`main_ppo.py:184-193` 因此**默认改道 `TaskRunnerV1`**（`verl/trainer/ppo/v1/*`，TransferQueue 驱动，`main_ppo.py:103-164`）；本页与 [[verl_ray_trainer_analysis]] 走读的 `RayPPOTrainer.fit` 降级为需要显式设置 `trainer.use_v1=false` 才会执行的 legacy 路径，且仍带 `main_ppo_v0.py` 的移除预告（v0.9.0）。`@deprecated` 装饰器本身在两基线间未变，变的只是路由默认值。
> 本页仍以 `RayPPOTrainer.fit` 为教学主链（源码结构最完整、有 [[verl_ray_trainer_analysis]] 的逐行深潜），但**不代表它是本基线下的默认执行路径**；`TaskRunnerV1`/TransferQueue 路径本系列尚无专页覆盖。

## 2. 启动与角色创建

`verl/trainer/main_ppo.py` 的职责是：

1. 读取 Hydra 配置；
2. 初始化 Ray；
3. 在 remote task 内构造 tokenizer、reward manager、role worker mapping 与 resource pools；
4. 创建 trainer，调用 `init_workers()` 后进入 `fit()`。

角色抽象让 actor/rollout、critic、reference、reward model 可以映射到不同 worker 类与资源池。真正的 placement、并行与后端细节在 worker，而 trainer 主要编排数据依赖。

阅读时先回答：

```text
role name
worker implementation
resource pool
colocate or disaggregate
training backend
rollout backend
```

角色枚举与资源池的实际映射机制（`verl/trainer/ppo/utils.py`、`verl/single_controller/ray/base.py`，本页基线下行号）：

- **Role 枚举**（`trainer/ppo/utils.py:27-56`）：`Actor`/`Rollout`/`ActorRollout`/`Critic`/`RefPolicy`/`RewardModel`/`ActorRolloutRef`/`TeacherModel`，`__str__` 给出配置里用的短名（`actor`/`rollout`/`critic`/`ref`/`rm`/`actor_rollout_ref`/`teacher`）。
- **要不要这个角色**由四个 `need_*` 纯函数从 config 推出（`utils.py:75-107`）：`need_reference_policy`（`use_kl_in_reward` 或 `actor.use_kl_loss`）、`need_reward_model`（`reward.reward_model.enable`）、`need_critic`（显式 `critic.enable`，否则仅当 `adv_estimator==GAE` 才需要——GRPO 等 critic-free 算法据此自动关掉 critic）。`RayPPOTrainer.__init__` 把结果缓存成 `use_reference_policy`/`use_rm`/`use_critic`（`ray_trainer.py:343-348`）。
- **colocate 还是 disaggregate**：资源池由 `ResourcePoolManager`（`single_controller/ray/base.py:185`）管理，`create_resource_pool()`（`:195`）按 spec 建池，FSDP 后端 `max_colocate_count=3`（actor_critic_ref/rollout/reward 三类 WorkerGroup 共享一组 GPU），Megatron 后端用 >1 分池。真正实现"同 GPU 多角色"的是 `init_workers()`（`ray_trainer.py:772`）里的 `create_colocated_worker_cls(class_dict)`（`:873`）——把落在同一资源池下的各角色类合并成一个 `WorkerDict`，在**一个 Ray actor 内实例化多个子 worker**，子 worker 方法以 `{prefix}_{method}` 形式挂到外层（dispatch 机制见 [[verl_single_controller_analysis]]）；随后 `wg_dict.spawn(...)`（`:879`）把合并 group 拆回按角色寻址的句柄字典。LoRA 打开时 `ref_in_actor=True`（`ray_trainer.py:356-360`），参考策略直接复用不挂 adapter 的 actor，无需单独起 ref worker group。

`init_workers()` 完整初始化顺序（建池 → 注册 actor/critic/ref 类 → 合并建 WorkerGroup → 分发句柄 → 各角色 `init_model()`）的逐行追踪见 [[verl_ray_trainer_analysis]] §2（旧基线 `8a694930`，机制未变、行号有漂移）。

## 3. DataProto 是跨 role 契约

[`verl/protocol.py:318`](https://github.com/verl-project/verl/blob/983cb0f24443f87b3d161fad318445130a620b07/verl/protocol.py#L318) 定义的 `DataProto` 是 driver 与所有 worker 之间的唯一数据契约；构造、chunk/concat、padding、`DataProtoFuture`、序列化成本的方法级剖析见 [[verl_dataproto_analysis]]。

| 容器 | 装什么 | 本页涉及的关键操作 |
|---|---|---|
| `batch`（TensorDict） | 等长张量：`input_ids`/`log_probs`/`advantages`… | `union`（`:781`）逐跳并入新字段、`reorder`（`:963`）负载均衡后复原顺序 |
| `non_tensor_batch`（numpy object） | 逐样本非张量数据：`uid`、原始 prompt、reward 附加信息 | `repeat`（`:971`）按 `rollout.n` 复制、`pop`（`:721`）抽出送 rollout 的子集 |
| `meta_info` | 与样本无关的全局元信息：`temperature`、policy version | 切分时整体复制给每个分片，不参与拼接维度 |

工业修改必须维护的四条不变量：batch 第一维一致；prompt 重复后 group uid 不丢；response mask 与 log-prob shape 对齐；reorder 后 reward/advantage 同步重排；policy version 和 sampling meta 不被 `pop`/`union` 丢失。

## 4. 一轮 `fit` 的真实顺序

固定快照中的承重点：

| 阶段 | locator | 读写对象 |
|---|---|---|
| rollout | `verl/trainer/ppo/ray_trainer.py:1488` | prompt batch → response batch |
| reward | rollout 后的 reward manager/model 路径 | token/sequence score |
| advantage | `verl/trainer/ppo/ray_trainer.py:1642` 调 `compute_advantage` | reward、mask、old values |
| actor update | `verl/trainer/ppo/ray_trainer.py:1665`，内部 `_update_actor` 在 `1302` | policy loss、optimizer |
| worker update | `verl/trainer/ppo/ray_trainer.py:1344` 调 `actor_rollout_wg.update_actor` | sharded batch |
| rollout refresh | `verl/trainer/ppo/ray_trainer.py:1690-1691` | 新 actor 参数 → rollout |

抽象时序：

```text
prompt batch
  generate sequences
  attach uid and masks
  compute reward
  compute or attach old and reference log probabilities
  compute advantages
  rebalance and split minibatches
  update actor
  publish new weights to rollout
```

有 critic、reference、reward model、validation 或 async reward 时会插入额外 future，但算法不变量不变。

## 5. Advantage 与 loss

`verl/trainer/ppo/ray_trainer.py:187` 的 `compute_advantage` 根据 estimator 分派。新增 estimator 时应同时确认：

1. group uid/schema；
2. response mask；
3. reward/return 粒度；
4. normalization divisor；
5. distributed aggregate。

policy loss registry 在 `verl/trainer/ppo/core_algos.py`：

| 算法/路径 | locator | 关注点 |
|---|---|---|
| vanilla PPO 类 | `1279-1358` | token ratio、clip、rollout IS |
| GSPO | `1538-1594` | sequence ratio 与 clip |
| SAPO | `1615` 起 | soft gate 语义 |
| CISPO | `2007` 起 | clip importance sampling |
| REINFORCE | `2292-2348` | baseline 与 rollout IS |
| bypass | `2373-2456` | rollout log-prob 与 correction |

新增算法的最小改动通常是 estimator + loss function + config；若改变 group/single-rollout、trajectory schema 或 policy version，就不再是纯 loss 修改，必须改 data/rollout。

## 6. 权重刷新

`RayPPOTrainer` 在 actor update 后触发 rollout weight refresh；完整搬运机制（3D-HybridEngine、CUDA IPC bucket、`CheckpointEngine` 两条路径、sleep/wake 显存接力、经济性测算）见 [[verl_rollout_resharding_analysis]]。本基线的时序承重点：

```text
actor update 完成
  → engine_workers.py:705-725  sleep/wake 边界：rollout resume(weights)
  → engine_workers.py:783-787  调用 rollout update_weights
  → vllm_rollout.py:271-320    异步权重接收、bucket、cache reset、开放 generation
```

`vllm_rollout.py:278` 说明 CUDA IPC 不可用时可 fallback 到 shared memory——这是传输实现细节，不等于跨节点任意 layout 自动兼容。

评审 weight refresh 时要找：

```text
trainer update completion
rollout sleep or request drain
weight buckets and transport
all-rank install completion
KV and prefix cache reset
new version visibility
```

## 7. On-policy 与 TIM 接口

verl 的同步 batch barrier 只控制版本时序，不消除 TIM。固定快照提供：

- policy loss 中可选 rollout IS；
- `verl/trainer/ppo/rollout_corr_helper.py:554-601` 的 raw correction weight、sequence expansion、clip、token/sequence rejection；
- bypass 路径直接使用 rollout log-prob。

推荐诊断顺序：

1. 同 checkpoint 比较 rollout/train token log-prob；
2. 区分 recompute 与 bypass；
3. 记录 token 和 sequence mismatch；
4. 再启用 IS/rejection；
5. 和 exact/small baseline 对比，不只看最终 reward。

## 8. Stable 与 experimental fully async

`verl/experimental/fully_async_policy/README.md:64` 把实验架构拆成 rollouter、trainer、message queue 等组件；`verl/experimental/fully_async_policy/fully_async_main.py:25-29,222` 使用独立入口。

边界是：

| stable PPO path | experimental fully async |
|---|---|
| `verl/trainer/main_ppo.py` + `RayPPOTrainer.fit` | `verl/experimental/fully_async_policy/fully_async_main.py` |
| phase-level rollout/update | producer/consumer 长期并发 |
| `DataProto` batch 主导 | message queue/version 主导 |
| 容易建立 correctness baseline | 适合研究 bounded staleness |

不能用 experimental README 解释 stable trainer 的实时行为，也不能以 stable tests 推断 fully async 已达到相同成熟度。

## 9. 修改指南

### 9.1 加新 loss

```text
register loss in core_algos
declare required tensors and masks
add config and validation
unit-test positive and negative advantage branches
test all-reduce divisor and variable response length
```

### 9.2 加新 rollout/agent loop

```text
define trajectory to DataProto conversion
preserve token loss mask and per-call version
separate tool observation from policy tokens
handle timeout and retry without reward pollution
add weight-refresh request-drain semantics
```

### 9.3 加新硬件/后端

```text
worker device and collective
train parallel backend
rollout engine
weight layout conversion
TIM measurement
checkpoint and profiler
```

## 10. 最小验证实验

| 层次 | 实验 | 通过标准 |
|---|---|---|
| 数据 | 2 prompt × 2 response 手工 batch | uid、mask、reward、advantage 对齐 |
| 算法 | tiny model 一步 update | loss/ratio 与独立实现一致 |
| 权重 | update 前后固定 prompt logit | rollout 看到完整新版本 |
| TIM | 同参数跨 train/rollout | mismatch 有基线与阈值 |
| 恢复 | weight refresh 中断 | 不暴露混合版本 |
| 性能 | 固定有效 token 与 freshness | 吞吐提升不靠旧样本/丢样 |

## Related Pages

- [[verl/index]] —— verl 系列总入口与知识地图
- [[verl_ray_trainer_analysis]] —— `RayPPOTrainer.fit` 逐方法源码走读（旧基线 `8a694930`），角色/资源池/`init_workers` 与本页 §2 互补
- [[verl_dataproto_analysis]] —— `DataProto` 完整方法级剖析，本页 §3 契约表的展开
- [[verl_rollout_resharding_analysis]] —— 3D-HybridEngine 权重重分片完整机制，本页 §6 时序的展开
- [[verl_single_controller_analysis]] —— colocate WorkerDict、dispatch/collect 装饰器机制
- [[verl_rl_algorithms_analysis]] —— 优势估计与策略损失注册表，本页 §5 的数学细节
- [[rl_framework_comparison|D06 工业后训练框架对比]]
- [[slime_architecture_analysis|D08 slime 高性能与异步架构]]
- [[on_policy_off_policy_staleness_analysis|D04 On-policy、Off-policy 与 Staleness]]
