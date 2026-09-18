---
title: "slime Megatron 训练后端分析"
---

# slime Megatron 训练后端分析

> **源码基线**：`THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e`（`main`，2026-09-03）
> **源码基线**：`NVIDIA/Megatron-LM@1dcf0dafa884ad52ffb243625717a3471643e087`（`dev`，2026-02-14）
> **主题**：slime 怎样在 Megatron 外围把 rollout 训练字典落成 optimizer step：先用一个四 rollout、五样本的实例回放 DP 调度、CP 打包、角色 tag 切换、PPO critic 数据面与训练步，再逐层展开初始化与 checkpoint、CPU tag 角色、`build_dp_schedule`、`get_batch`、前向复用、advantage 位置、训练步和 sleep/wake。其后是变体与成本、调用树与源码阅读路线、配套机制（Role YAML、钩子、调试转储、路由重放、优化器状态重置、参数冻结）、约束与配置契约，核心代码在 `slime/backends/megatron_utils/` 与 `slime/utils/dp_schedule.py`。
> **适用范围**：Megatron 训练执行与数据落地；Sample 与训练字典语义归 [[12_slime_sample_datasource_analysis|Sample 与 DataSource]]，loss、KL 入口与归约数值归 [[15_slime_loss_parallelism_analysis|Loss 与并行归一化]]，权重同步归 [[16_slime_weight_sync_analysis|权重同步]]，logprob 一致性归 [[17_slime_train_inference_consistency_analysis|训推一致性]]，MTP 训练归 [[21_slime_speculative_decoding_mtp_analysis|投机解码与 MTP]]。
> **最近更新**：2026-09-17。按 `4c193f1f` 基线覆盖训练后端全链路，含 PPO critic 数据面回放、`--allgather-cp` 的架构限制、参数冻结、优化器状态重置与 slime 强制的 Megatron 默认值。

---

## 1. 特性概览

### 1.1 问题背景

rollout 交付的是长度不一、带样本标识、mask 与可选行为策略字段的训练字典（见 [[12_slime_sample_datasource_analysis|Sample 与 DataSource]]）；Megatron 需要的却是每个 VPP stage 按相同 schedule 推进的 micro-batch 序列、每个 micro-batch 先变成 THD packed token 流与 `PackedSeqParams`，再交给流水线 forward/backward，并且 optimizer 只在整步反向完成后推进一次。两边之间必须有人守住四条训练侧不变量：rollout 身份先于打包固定，否则 compact 扇出的片段会改变 step 大小与训练进度；各 DP rank 与 VPP stage 执行相同数量的 micro-batch，否则流水线失步；token、序列边界与 next-token mask 一起变换，否则 packed attention 或 loss 位置会串样本；optimizer 只在完整流水线反向后推进，否则 micro-batch 会被误当成独立更新。此外 RL 还要在一轮里给同一批 token 算 ref、teacher、old-policy 的对照 logprob，而这些角色都不需要各自的 optimizer。

### 1.2 解决方法

slime 没有另写通用 RL trainer，而是在 Megatron 外围加一层 actor 适配。进入内核前的工作全部在还能看到完整 step 的地方完成：`RolloutManager` 调 `build_dp_schedule` 按 `rollout_id` 首次出现顺序把样本聚成逻辑 rollout，每 `global_batch_size` 个 rollout 组成一个训练步，每步先 first-fit（或静态切块）打成 micro-batch，再把 micro-batch 数对齐到 `dp_size × mb_group` 的倍数，最后轮询或按估算 FLOPs 分给 DP rank；每个 rank 收到 `partition`、`micro_batch_indices`、`num_microbatches`、`global_batch_sizes`。trainer 侧 `DataIterator` 只按 `micro_batch_indices` 取子集，`get_batch` 再做 CP 切片、THD 拼接、补齐与 mask 对齐。角色方面，`TensorBackuper` 给同一份 GPU 模型维护 `actor`、`ref`、`teacher`、`old_actor` 等 CPU pinned 副本，`_switch_model` 整份换入后走同一个 `get_forward_backward_func(forward_only=True)`；只有 critic 是独立的 `RayTrainGroup`，`--advantage-estimator ppo` 时它先训练，再把训练前的 values 交给同 rank 的 actor。进入内核后继续使用 Megatron 原生 `get_model`、DDP、流水线 schedule、`get_megatron_optimizer` 与 `OptimizerParamScheduler`，slime 只提供 data closure 与 loss 回调，并把该步的逻辑 rollout 数 `step_global_batch_size` 同时交给 loss 缩放与 scheduler 的 `increment`。

### 1.3 收益、开销和约束

| 维度 | 直接收益 | 必付成本或边界 |
|---|---|---|
| 训练内核 | 复用 Megatron 的 TP/PP/CP/EP、distributed optimizer、overlap hooks 与模型专属能力 | 与镜像钉住的 Megatron `1dcf0daf` 及 `megatron.patch` 耦合；官方 quick start 自承镜像可能含临时 patch |
| 调度 | step、micro-batch、DP partition 在 `RolloutManager` 算一次，所有 rank 回放同一计划 | 计划经 Ray object store 的 CPU 张量送达，actor 注释自承"不确定是否成为瓶颈" |
| 动态打包 | first-fit 逼近 `max_tokens_per_gpu × cp_size`，对齐靠拆 bin 而非补样本 | 拆出的 bin 不均匀；`balance_by_flops` 不保证 token cap；静态路径不对齐直接断言 |
| CP 切片 | zigzag 两段让每个 CP rank 的因果注意力负载接近；mask 与 token 同一规则切分 | 每条样本各自补齐到 `2·cp·chunk`，短样本可能在某个 rank 上没有 response 位置；`--allgather-cp` 在 CP>1 时只允许 DSA 架构 |
| 角色共享 | ref/teacher/old_actor 不占第二份 GPU 显存与 process group | 每次切换是一次 CPU↔GPU 整份拷贝加设备同步；每个 tag 一份 pinned host memory |
| 前向复用 | 单步、`kl_coef = 0` 等条件下跳过独立 old-policy 前向 | 多步、critic、GSPO、mismatch 指标等任一条件都要多一次全批前向 |
| 卸载 | `sleep`/`wake_up` 让 colocate 或 PPO 下训练与 rollout 分时用卡 | 销毁并重建 process group；PP>2 需额外 barrier 预热 WORLD；PPO 下训练阶段每 GPU 两次，`update_weights` 再一次 |
| 进度计数 | `step_global_batch_size` 同时驱动 loss 缩放、指标分母与 LR increment | `train_iters` 按名义样本数估算；动态采样、过滤或自定义 step splitter 改变实际步数时，schedule 可能提前或推迟到达平台 |
| 优化器状态 | `--use-stateless-adam` 不分配 Adam 矩；`--reset-optimizer-states` 每轮清零 | 两者只在每轮单步时更新相同；stateless 必须 `--no-save-optim`（§4.5） |

### 1.4 术语约定

| 术语 | 含义 |
|---|---|
| 训练步 / `step_global_batch_size` | `global_batch_sizes` 的一个元素，即一次 optimizer step 覆盖的逻辑 rollout 数 G；不是样本数也不是 token 数 |
| micro-batch / K / M | 一步内打包出的 bin；K 是全步总数，M = K / dp_size 是每 rank 的 `num_microbatches` |
| `align_to` | `dp_size × (mb_group if vpp_size > 1 else 1)`，K 必须是它的倍数 |
| `partition` | 某 DP rank 拥有的样本在展平批次中的全局位置；`micro_batch_indices` 是 partition 内的局部下标 |
| tag | `TensorBackuper` 里一份 CPU pinned 参数副本的名字：`actor`、`ref`、`teacher`、`old_actor`、`rollout_actor` |
| zigzag CP | 每条样本补齐后切成 `2·cp` 段，rank r 取第 r 段与第 `2cp−1−r` 段 |
| THD | Megatron packed sequence 布局，一条 `[1, T]` token 流加 `cu_seqlens` 描述各样本边界 |
| `forward_only` | 用同一流水线函数只做前向并在 PP last stage 收集非 loss 输出 |
| `valid_step` | `check_for_nan_in_loss_and_grad` 为假时由 slime 预检梯度，inf/nan 则跳过本步；为真时恒为真，`optimizer.step` 报告失败即断言 |
| V_old | PPO 下 critic 在本轮训练前 `forward_only(get_values)` 得到的 value；既是 value loss 的 clip 基准，也交给 actor 算 GAE |

---

## 2. 训练后端详细方案

### 2.1 最小实例：四个逻辑 rollout、五条样本进入一次 optimizer step

取 `dp_size=2`、`cp_size=2`、`tp=1`、无 VPP，`--use-dynamic-batch-size` 且 `max_tokens_per_gpu=9`（cap = 9 × 2 = 18），`data_pad_size_multiplier=8`，`rollout_batch_size=2`、`n_samples_per_prompt=2`、`num_steps_per_rollout=1`（故 `global_batch_size=4` 个逻辑 rollout），估计器取默认的 `grpo`。rollout 交来五条样本，prompt 一律 4 个 token：s0（r0，response 8，其中下标 3 是工具 token、mask 0）、s1（r1，response 2）、s2a 与 s2b（同属 r2 的 compact 扇出，response 4 与 6）、s3（r3，response 2）。converter 已按 [[12_slime_sample_datasource_analysis|Sample 与 DataSource]] 的规则给出 `rollout_ids=[0,1,2,2,3]` 与 `rollout_mask_sums=[7,2,10,10,2]`。本例开 `--use-kl-loss`、`kl_coef=0`：ref tag 因 `use_kl_loss` 而装载（`with_ref = kl_coef ≠ 0 or use_kl_loss`），reward 不做 KL 整形。即便把 `--kl-coef` 设成非零，默认 grpo 也不整形 reward：`slime/utils/ppo_utils.py::get_grpo_returns` 返回 `ones_like(kl) × reward`，只有 ppo、reinforce_plus_plus、reinforce_plus_plus_baseline 三个估计器读 `kl_coef`；grpo 下它只会强制装载 ref、关掉 §2.1.3 的 logprob 复用、与 `kl_loss_coef` 互斥，并让 `rollout/kl` 日志出现非零值，完整语义见 [[15_slime_loss_parallelism_analysis#2.2 从最小实例到整套 loss 层|KL 的两个入口]]。上图回放调度与打包两次决定性转换，§2.1.3 的下图回放一轮角色切换、PPO 数据面与训练步。

![上图：build_dp_schedule 组步、打包、拆 bin 与分发；get_batch 的 zigzag 与 allgather 两种 CP 切片及 mask 对齐](assets/slime_megatron_train_step.svg)

| 步骤 | 输入 | 决定性转换 | 输出 |
|---|---|---|---|
| 组步 | `rollout_ids=[0,1,2,2,3]`，`global_batch_size=4` | 按首次出现聚成 r0 r1 r2 r3，`4 // 4 = 1` 步；r2 的两个片段同步 | 一步五条样本，长度 `[12,6,8,10,6]` |
| 打包 | 长度与 cap 18 | first-fit：`[s0 s1]=18 [s2a s2b]=18 [s3]=6 → K=3` | 三个 bin |
| 对齐 | K=3，`align_to=2` | `target_K = ceil(3/2)×2 = 4`；拆和最大的多样本 bin，和相同取下标大者 → bin1 `[s2a s2b]` 按长度降序分成 `[s2b]` 与 `[s2a]` | bins `[s0 s1] [s2b] [s3] [s2a]`，K=4 |
| 分发 | `balance_data=False` | rank r 取 bin r、r+2 | rank 0 `partition=[0,1,4]`、`micro_batch_indices=[[0,1],[2]]`；rank 1 `partition=[3,2]`、`[[0],[1]]`；`num_microbatches=[2]`、`global_batch_sizes=[4]` |
| 打包 mb0 | rank 0 的 `[s0 s1]`，cp0 | s0 chunk 3 无补齐，取段 0 与 3；s1 chunk 2 补 2 个 pad；拼成 10 个 token 再补到 16 | cp0 tokens `s0:0 s0:1 s0:2 s0:9 s0:10 s0:11 s1:0 s1:1` + 8 pad，`cu_seqlens=[0,12,20,32]` |
| 对齐 mask | s0 mask `1 1 1 0 1 1 1 1`、s1 mask `1 1` | 左补 `prompt_len−1=3`、右补 1，再同规则切片 | cp0 有效 2（s0 的 response 下标 `{6,7}`，s1 无），cp1 有效 7（s0 的 `{0,1,2,3,4,5}` 含一个 0，s1 的 `{0,1}`）；2 + 7 = 9 |
| 角色 | ref tag 存在，单步，`kl_coef=0` | ref 前向存 `ref_log_probs`；`can_reuse_log_probs_in_loss=True` → 不做 old-policy 前向 | 全批前向 2 次（ref 1 + 训练 1）；若 `num_steps_per_rollout=2`（此时 G 变为 2、共 2 步），可复用条件失效，多一次 old-policy 全批前向共 3 次，按 `forward_backward_func` 调用计则从 2 次变为 6 次 |
| 训练步 | K=2 个 mb/rank，G=4 | 清梯度 → `forward_backward_func` 累积 2 个 mb → `optimizer.step()` → `scheduler.step(increment=4)` | `train_iters = 100×2×2 // 4 = 100`，`lr_decay_steps = 400`（`num_rollout=100`） |
| PPO | `--advantage-estimator ppo`，同一 schedule | critic 先取 V_old、算 returns、训练 value head，再把 V_old 交给 actor | 每 GPU 一轮全批前向 critic 2 + actor 3 = 5，optimizer.step 2 次（§2.1.5） |

同一份数据若走静态路径（`micro_batch_size=2`），切块得 `[s0 s1] [s2a s2b] [s3]`，K=3 不是 2 的倍数，`build_dp_schedule` 直接抛 `AssertionError` 并要求调整 step size、micro batch 或 DP/VPP，而不是拆块，因为拆块会破坏"静态 micro-batch 定长"的不变量。

#### 2.1.1 调度层：先固定统计单位，再考虑物理装箱

组步发生在打包之前，所以无论 r2 拆成几个片段，它只占一个 rollout 名额，训练步数与 `step_global_batch_size` 都不受扇出影响；尾部凑不满一整步的 rollout 连同片段被丢出 schedule，不足一步立即断言，每步样本数少于 `dp_size` 也断言。拆 bin 是唯一的对齐手段：`expand_bins_by_splitting` 每次取和最大的多样本 bin（Python 元组比较使和相同时取下标大者），`_split_bin_by_tokens` 按长度降序把样本逐个放进当前较轻的一半（两半相等时放左半），两半都是原 bin 的真子集，因此新 bin 不会大于原 bin；所有 bin 都成单样本仍凑不够时断言。s2a 与 s2b 在本例落到同一 rank 的两个 micro-batch，这正是 [[12_slime_sample_datasource_analysis|Sample 与 DataSource]] 预先算 `rollout_mask_sums` 的原因：切分后局部批次看不见完整分母。

#### 2.1.2 打包层：token、边界与 mask 同一规则变换

`get_batch` 先保留原始 token 列表为 `unconcat_tokens`（CP 下算 logprob 要用完整 token），再对每条样本单独 `slice_with_cp`：chunk 大小为 `ceil(total / (2·cp))`，补齐到 `2·cp·chunk` 后取两段；拼接后把总长补到 `tp × data_pad_size_multiplier` 的倍数，局部 `cu_seqlens` 乘 `cp_size`。源码注释说 THD 要求 `cu_seqlens` 是原始长度，实际乘回的是每条样本补齐后的全局长度：本例 s0 为 12，s1 为 8 而非原长 6。mask 不能直接切：response 第 r 个 token 由位置 `prompt_len + r − 1` 的 logit 预测，所以先左补 `prompt_len − 1`、右补 1 把 mask 对齐到 token 流位置，再做相同切片，最后断言 mask 与 tokens 同形。本例 s1 在 cp0 上没有任何 response 位置，因为 cp0 分到的两段是 prompt 头与补齐后的尾部。这是 CP 上"空 rank"的来源，loss 侧对它的处理见 [[15_slime_loss_parallelism_analysis#2.2 从最小实例到整套 loss 层|CP 上的分子可加]]。上图下半的 allgather 对照是另一种布局，适用边界见 §2.2.4。

#### 2.1.3 角色层：一份 GPU 模型、多份 CPU tag

![下图：默认 grpo 一轮的 tag 切换与前向计数；同一 schedule 走 PPO 时 critic 与 actor 的分时数据面；train_one_step 闭环](assets/slime_megatron_train_round.svg)

actor 初始化时 `backup("actor")`，随后按 `with_ref`、`with_opd_teacher`、`keep_old_actor` 依次用 `load_other_checkpoint` 把别的 checkpoint 装进同一份模型并各存一个 tag。一轮里 `train_actor` 先切 `ref` 做 `forward_only` 得 `ref_log_probs`，再切 `teacher`，再切 `old_actor`（或 `actor`），只有 `can_reuse_log_probs_in_loss` 不成立时才再做一次 old-policy 前向；接收 critic 的 `values`；切回 `actor` 算 advantage；训练；`backup("actor")`。本例的可复用条件全部满足，少做一次全批前向。

#### 2.1.4 步层：一次 optimizer step 的三个边界

`train` 对 `num_microbatches` 的每个元素调一次 `train_one_step`；每步清 grad buffer 与 optimizer grad，`forward_backward_func` 跑完本 rank 全部 M 个 micro-batch（不是每个 micro-batch 单独 step），`valid_step` 为真才 `optimizer.step()` 并 `assert update_successful`（NaN 检查关闭时由 slime 预检梯度，开启时恒为真），然后 `opt_param_scheduler.step(increment=step_global_batch_size)`。因此 rollout round 是数据版本边界，`global_batch_sizes` 的一个元素是 optimizer 边界，micro-batch 只是流水线与梯度累积的执行单元。

#### 2.1.5 同一实例走 PPO：critic 数据面

`--advantage-estimator ppo` 让参数校验派生 `use_critic=True`、把 critic 的节点数与每节点 GPU 数设成 actor 的，并强制 `offload_train`。critic 是第二个 `RayTrainGroup`，与 actor 共用同一个 placement group（每个 worker 声明 `num_gpus=0.4`），所以两组靠 sleep/wake 分时占卡。本例若保持 `cp_size=2`，ppo 分支会在 cp_rank 0 对本地 response 片执行 `token_level_rewards[-1] += reward`，而 s1、s2b、s3 在 cp0 上本地片为空，会抛 `IndexError`（源码推导，未运行；reward 落点的完整分析在 [[15_slime_loss_parallelism_analysis|Loss 与并行归一化]]）。因此这里取 `cp_size=1`、`max_tokens_per_gpu=18`：cap 仍为 18，`build_dp_schedule` 给出与上图逐项相同的 partition 与 micro-batch 计划，下面只看两组之间多出来的数据面。

1. driver `train.py::train` 先发 `critic_model.async_train(rollout_id, rollout_data_ref)`，再发 actor 的 `async_train(..., external_data=value_refs)`；两组拿到同一份 per-rank ref。`RayTrainGroup.create` 对两组都调 `set_rollout_manager`，`TrainRayActor.set_rollout_manager` 在 `args.rank == 0` 时上报 `train_parallel_config`，critic 又在 actor 之后创建，所以两组 rank 0 都上报，后创建的 critic 覆盖 actor 的；拓扑一致时两者相同。官方 `megatron-config.md` 声明两组拓扑必须一致，代码没有守卫。
2. critic worker：`wake_up` 恢复 memory saver 并重建 process group；`MegatronTrainRayActor.train_critic` 先 `forward_only(get_values)` 取每条样本 response 段的 V_old，`get_values` 调 `get_responses(..., apply_temperature=False)`，value head 输出不除 rollout 温度（`tests/test_value_temperature.py::test_get_values_does_not_apply_rollout_temperature`）。随后 `compute_advantages_and_returns` 按 critic 自己的参数运行：critic 从不做 logprob 前向，未开 `--use-rollout-logprobs` 时 `log_probs` 为空，KL 按 values 的形状取零，critic 的回归目标 returns 不含 KL 整形。开了 `--use-rollout-logprobs` 时 `log_probs` 取 rollout logprob、不为空；不给 `--megatron-config-path` 的 critic 保留 CLI 的 `kl_coef`，非零时会以 `ref_log_probs=None` 调 `compute_approx_kl` 而抛 `TypeError`（源码推断，未运行；§5.1）。接着原地把自己的 `loss_type` 改成 `value_loss` 训练 value head（1 次 optimizer step，scheduler `increment=4`），PP last stage 把 V_old 搬回 CPU 作为 `{"values": ...}` 返回，`train` 末尾 sleep。
3. actor worker：`RayTrainGroup.async_train` 把 critic worker i 的返回 ref 作为 `external_data` 交给 actor worker i。按 Ray 对顶层 ObjectRef 参数先解析再执行的契约，actor 的 `train` 在同一 GPU 上的 critic 训练并 sleep 之后才开始。actor `wake_up` → ref 前向 → 因 `use_critic` 使复用条件失效而多做一次 old-policy 前向 → PP last stage 用 `tensors_to_gpu` 放入 values（与 critic 训练前取的同一份 V_old）→ 以 actor 自己的 `kl_coef` 做 GAE → policy 训练 → `backup("actor")` → sleep。

训练阶段每 GPU：全批前向 critic 2 + actor 3 = 5（默认 grpo 为 2），optimizer.step 2 次，wake_up/sleep 与 process group 重建各 2 次。训练之后 driver 每轮还调 `actor_model.update_weights()`：有可更新引擎时，actor worker 在 `offload_train ∧ use_critic ∧ ¬colocate` 下为重连 rollout 引擎再做一次完整的 wake_up/sleep，colocate 下只重建再销毁 process group，所以一轮合计重建 3 次（发布过程归 [[16_slime_weight_sync_analysis|权重同步]]）。与默认 grpo 相比，PPO 多出三处语义差别：多一份模型与 optimizer 状态，靠强制卸载分时共卡；actor 用的是 critic 更新之前的 value；critic 的回归目标不含 KL 整形，而 actor 的 advantage 在 `kl_coef ≠ 0` 时含。`--num-critic-only-steps` 之前的轮次 driver 只等待 critic 的结果，actor 既不训练也不保存；`num_rollout == 0` 时 critic 根本不创建。

### 2.2 从最小实例到整个训练后端

下面各组件的"为何"是本页依据源码形态与失败路径重建的设计理由（标"本页推断"），"怎样"与"代价"以冻结基线的实现为准；Megatron 内部的 schedule、DDP 平均与 optimizer 语义按其公开契约叙述，slime 源码只证明自己传了什么、断言了什么。页头第二条基线 `NVIDIA/Megatron-LM@1dcf0daf` 是 slime `docker/Dockerfile` 的 `MEGATRON_COMMIT`，镜像在它之上再打 `docker/patch/latest/megatron.patch`；这条基线只用于核对本页点名的依赖侧契约，核对过的只有这几处，例如 `OptimizerParamScheduler.step` 把 `num_steps` 加上 `increment`。

#### 2.2.1 初始化：Megatron 模型、优化器、scheduler 与 checkpoint 分派

**职责。** `MegatronTrainRayActor.init` 在 `--debug-rollout-only` 下直接返回 0：trainer actor 照常创建，只是不建模型，其后的 `train`、`save_model`、`update_weights` 也立即返回；`--debug-train-only` 不启动 SGLang 服务，trainer 完整初始化，只有 `update_weights`（以及 `RolloutManager.eval`）立即返回；`--load-debug-rollout-data` 会打开 `debug_train_only`，两个 debug 开关互斥（debug 模式的部署形态归 [[11_slime_ray_control_plane_analysis|Ray 控制面]]）。否则先 `monkey_patch_torch_dist`；`TrainRayActor.init` 经 `slime/utils/accelerator` 选设备与通信后端（CUDA、ROCm 与 MUSA 共用这层抽象）并建默认组与 gloo 组；按环境变量注册可销毁的默认 process group；`init(args)`（`mpu.initialize_model_parallel`、随机种子、tokenizer、`init_num_microbatches_calculator`）；逐 GPU 串行读 HF config 与 tokenizer 以避免并发写缓存；再调 `initialize_model_and_optimizer`。后者由 `setup_model_and_optimizer` 用 Megatron `get_model` 构建 model chunks，模型 provider 外包一层参数冻结（§4.6）。`num_rollout == 0`（只评估）时到此返回，optimizer 与 scheduler 都是 `None`，并置 `no_load_optim`。否则把 `OptimizerConfig` 的同名字段从 args 复制过去，`get_megatron_optimizer` 建优化器，`get_optimizer_param_scheduler` 建 scheduler；然后 `load_checkpoint` 分派：目录存在且非空是断言前提，含 `latest_checkpointed_iteration.txt` 或目录名匹配 `iter_XXXXXXX` 走 Megatron 原生加载，否则走 HF→Megatron 权重加载并在混合精度下 `optimizer.reload_model_params()`，iteration 记 0。ROCm 的 HIP 运行时下还会先把 Megatron 的 `FileSystemWriterAsync` 换成 `ROCmFileSystemWriterAsync`。

**为何。** 复制一套 Megatron 模型与优化器构造会让 slime 追着 Megatron 的每个并行特性改；只做参数映射与分派，Megatron 的优化继续留在其所有者手中（本页推断；被否方案是"统一 trainer"，判据是要不要重新实现 PP/VPP、CP、distributed optimizer 与 overlap hooks）。`init_num_microbatches_calculator` 的源码注释写明"我们不用它，只为通过 Megatron 校验"，真实的每步 `num_microbatches` 来自上游 schedule。

**怎样。** scheduler 以样本数计：`train_iters = num_rollout × rollout_batch_size × n_samples_per_prompt // global_batch_size`，`lr_decay_iters` 缺省等于它，`lr_decay_steps = lr_decay_iters × global_batch_size`，warmup 按 `lr_warmup_fraction × lr_decay_steps` 或 `lr_warmup_iters × global_batch_size`。`get_optimizer_param_scheduler` 的注释列出的漂移来源是动态采样、过滤与自定义 step splitter，它们改变每轮实际的样本或 rollout 数；compact 扇出不在其列，因为调度按 rollout 分步，`RolloutManager._split_train_data_by_dp` 的 docstring 写明每轮步数固定为 `rollout_batch_size × n_samples_per_prompt // global_batch_size`，与每个 rollout 产出几条样本无关（§2.1.1）。schedule 靠 `opt_param_scheduler.num_steps` 记录真实进度（续训时随 checkpoint 保存），最坏是提前或推迟到达平台；需要精确控制时显式给 `--lr-decay-iters`。`--use-stateless-adam` 在构造期间临时把 Megatron 的 `Adam`（及 `CPUAdam`、distrib optimizer 的 `Adam`）替换成 `StatelessAdam`，并把 distributed optimizer 的 `init_state_fn` 换成空操作，更新语义见 §4.5。critic 从 actor 的 Megatron checkpoint 加载时，`_critic_output_layer_needs_reinit` 读 dist-checkpoint 元数据：`output_layer.weight`/`bias` 缺失、或形状与单输出 value head（`LinearForLastLayer`，`output_size=1`）不符，就记一条告警并返回真；加载完成后 `_reinitialize_critic_output_layer` 以 `init_method_std`（缺省 0.02）正态初始化权重、偏置置零，混合精度下再 `optimizer.reload_model_params()` 同步主参数。形状一致（例如从已有 critic checkpoint 续训）则保留已训练的 head。trainer 最后把 `train_parallel_config`（`dp_size` 取 `with_context_parallel=False`、`cp_size`、`vpp_size`、`microbatch_group_size_per_vp_stage`）经 rank 0 送给 `RolloutManager`。

**代价与边界。** `setup_model_and_optimizer` 断言 `not moe_use_upcycling` 且 `load` 或 `pretrained_checkpoint` 之一存在；stateless Adam 断言 `optimizer == adam` 与 `no_save_optim`；`init` 断言 numpy 1.x。checkpoint 模块启动时猴补 `ShardedTensor` 的元数据校验以加速大模型加载，注释自承 hacky。参数层面 slime 还强制改写一批 Megatron 默认值（distributed optimizer 恒开、变长序列、MoE 分发器等），清单见 §6。

#### 2.2.2 角色与 CPU tag：ref、teacher、old_actor 共享一份 GPU 模型

**职责。** 非 critic worker 建一个 `TensorBackuper(source_getter)`：每个 tag 一份按参数名索引的 CPU pinned 张量；`backup` 首次分配 `empty_like(pin_memory=True)` 再 `copy_(non_blocking)`，结束时 `accelerator.synchronize()`；`restore` 断言每个参数名都有备份后整份拷回；`copy` 在两个 tag 间搬；`get` 直接返回某 tag 的字典。`_switch_model` 对不在 `backup_tags` 里的 tag 抛 `ValueError`。`load_other_checkpoint` 暂存 `load/no_load_optim/no_load_rng/finetune/ckpt_step`，用目标路径替换 `load`，关闭 optimizer/RNG 加载、开 finetune，ref 与 teacher 可各自指定 `ckpt_step`（`--ref-ckpt-step`、`--opd-teacher-ckpt-step`），加载成功后恢复原 args、`backup(tag)` 并把 active tag 设为该 tag；恢复语句不在 `finally` 中，加载异常会中止此路径。因此未开 offload 时，初始化结束后 GPU 上是最后装载的那份权重，要到 `train_actor` 的 advantage 分支才切回 actor（关闭该分支时的后果见 §5.1 最后一行）；`create_weight_updater` 构造 weight updater 时拿到的 `weights_getter` 指向 CPU `actor` tag。

**为何。** 为 ref、teacher、old_actor 各建常驻 trainer 会复制 process group、模型显存与调度对象，却没有对应的 optimizer 工作；它们只需为同一 token batch 提供对照 logprob，所以用传输时间换显存（本页推断；判据是该角色是否需要独立的可训练状态）。critic 有独立目标与 optimizer state，因此是唯一独立的 `RayTrainGroup`，与 actor 共享同一 placement group 区域（`pgs["critic"] = pgs["actor"]`），参数经 `parse_megatron_role_args` 或 deepcopy 得到。

**怎样。** 一轮顺序见 §2.1.3。`keep_old_actor` 且 `update_weights_interval == 1` 时初始化额外 `backup("rollout_actor")`；`update_weights` 之后按队列轮转 `copy(rollout_actor → old_actor)` 再 `backup("rollout_actor")`，interval 非 1 时直接 `backup("old_actor")`。它维护的是训练侧对照快照，`old_actor` 不等于当前 GPU 模型。`(rollout_id + 1) % ref_update_interval == 0` 且有 ref tag 时刷新 ref。PPO 的 critic 数据面（V_old 的取得、critic 另算 returns、worker i 到 worker i 的 ref 交接）见 §2.1.5；`num_critic_only_steps` 之前的轮只训练 critic。

**代价与边界。** 每个 tag 占一份 pinned host memory；每次切换是一次全模型 H2D 拷贝加同步；参数校验强制 PPO 开 `offload_train`；`offload_train` 会同时置 `disable_grad_buffers_cpu_backup` 与 `disable_param_buffers_cpu_backup`，但 critic 把后者改回假（`create_training_models` 与 `_apply_megatron_role_overrides`），critic 仍保留 param buffer 的 CPU 备份。官方文档把 actor/critic 不同拓扑标为当前不支持。

#### 2.2.3 调度层：build_dp_schedule 先按 rollout 组步，再打包、对齐、分发

**职责。** 纯 Python 函数，输入 args、`train_parallel_config`、`total_lengths`、`global_batch_size` 与 `rollout_indices`，输出 `partitions`、`micro_batch_indices`、`num_microbatches`、`global_batch_sizes`。模块 docstring 把策略写成"pack first, distribute second"，并列出四条被 `tests/test_dp_schedule.py` 断言的不变量：各 rank 每步 `num_microbatches` 相同；动态路径（不开 `balance_by_flops`）每个 mb ≤ `max_tokens_per_gpu × cp_size`，唯一例外是单条超 cap 样本独占一个 mb；各 rank 样本并集等于裁剪后保留的样本且每条只放一次；每 rank 的 `micro_batch_indices` 展平恰为 `range(num_samples_rank)`。

**为何。** 若让每个 Megatron rank 独立读 rollout 数据，就要在所有 rank 上重复解释 `rollout_id`、变长打包、compact 扇出与动态过滤，还要证明各 PP rank 的 micro-batch 次序一致；在仍拥有全局视图的 `RolloutManager` 算一次再按 DP 身份取数，牺牲 CPU/Ray 传输换来语义与执行边界分离（本页推断；判据是 rollout 语义解释是否需要全局视图）。同一 DP 副本内的 TP、PP、CP、EP rank 取同一个 `rollout_data_ref[dp_rank]`，CP 之后才在 `get_batch` 内切同一序列，EP 在前向时按路由分发 token，都不是各自领新样本。

**怎样。** 打包变体的枚举依据是 `_pack_step_into_mbs` 的分支：`use_dynamic_batch_size` 且 `balance_by_flops` 时先用总 token / cap 上取整定 bin 数，再按 `calculate_fwd_flops` 估算的工作量做 Karmarkar-Karp（`equal_size=False`），注释写明不保证 token cap；仅动态时 `first_fit_pack`；否则按 `micro_batch_size` 固定步长切块。分发变体的依据是 `args.balance_data`：开时对每个 mb 的估算 FLOPs 做 `get_seqlen_balanced_partitions(..., equal_size=True)`，该函数用最大差分法：按负载排序建状态，每次弹出差值最大的两个状态并把重端与轻端反向配对合并，直到剩一个，等大小约束保证每 rank 相同数量的 mb；关时 rank r 取 bin r、r+dp、r+2dp。本例 4 个 bin 的工作量是 `f(12)+f(6)`、`f(10)`、`f(6)`、`f(8)`；对任意 `f(L)=aL+bL²`（a、b 非负且不全为 0），排序与差值关系不变，KK 都给出 rank 0 `{bin0,bin2}`、rank 1 `{bin1,bin3}`，与轮询相同。`balance_by_flops` 会同时置 `balance_data=True`，并要求 `use_dynamic_batch_size`。参数 `--num-steps-per-rollout` 在校验时反算 `global_batch_size` 并断言一致，不改变"逻辑 rollout"这一计数单位。

**代价与边界。** 断言：`num_steps ≥ 1`、每步样本数 ≥ `dp_size`、动态路径拆到全单样本仍不足、静态路径 K 不对齐。KK 均衡的是估算计算量而非 wall time。`--balance-data` 的 help 仍写着"可能把同一 prompt 的不同 response 放进不同训练步"，与冻结基线不符：分步在打包之前按 rollout 完成，KK 只在每步内部把 micro-batch 分给 rank。单条超 cap 样本不截断，独占一个超 cap mb，官方 quick start 同样如此说明。

#### 2.2.4 打包层：DataIterator 与 get_batch 的 THD 流与 CP 切片

**职责。** `process_rollout_data(rollout_data_ref, dp_rank, dp_size)` 断言 ref 数等于 `dp_size`，`ray.get` 本 rank 那份，把全局 `total_lengths` 存进 `Timer().seq_lens` 供性能日志后按 `partition` 投影，并为本地样本派生 `local_raw_reward`。`_get_rollout_data` 再把 `tokens`（long）、`loss_masks`（int）、`rollout_mask_sums`（float32）与多模态张量预搬到当前加速器设备，`rollout_log_probs` 与 `teacher_log_probs` 先按 CP 规则 `slice_log_prob_with_cp` 再搬。`get_data_iterator` 为每个 VPP stage 建一个共享同一 `micro_batch_indices` 但各自计 offset 的 `DataIterator`；`get_next(keys)` 对每个键取子集，缺失键给 `None`。`get_batch` 的两个分支由 `allgather_cp` 选择：默认 zigzag 分支逐样本切片（§2.1.2）；`allgather_cp` 分支先把整个 micro-batch 拼成一条全局流、补到 `cp_size × pad_size` 的倍数后按 `cp_rank` 连续等分，mask 同样先拼再切。

allgather 布局只服务 DSA 稀疏注意力。`megatron_parse_args` 在解析期调用 `_validate_allgather_cp_supported`：`--allgather-cp` 与 `--context-parallel-size > 1` 同开、而 HF config 的 `architectures` 不含 `DeepseekV32ForCausalLM` 或 `GlmMoeDsaForCausalLM` 时抛 `ValueError`，报错原文说明非 DSA 模型仍按 zigzag CP 布局计算，开它会静默打乱 token 顺序（`tests/test_megatron_argument_validation.py::test_allgather_cp_rejects_non_dsa_cp_models`；CP=1 时不检查，`--debug-rollout-only` 跳过 HF 校验也就跳过这一检查）。这个 flag 在 argparse 里没有 help 文本，这条限制只有校验函数与测试可查。用 §2.1 的 mb0 对照两种切法（真实运行需换成上述 DSA 架构）：两条样本拼成 18 个 token，补到 `2 × 8 = 16` 的倍数 32，`cu_seqlens=[0,12,18,32]` 不乘 cp；cp0 拿 `s0:0–11` 与 `s1:0–3`（有效 mask 8），cp1 拿 `s1:4–5` 与 14 个 pad（有效 mask 1）。有效 mask 总数与 zigzag 相同，都是 9，但两个 rank 的真实 token 数是 16 对 2。源码注释称之为 DSA 模式，logprob 与 value 侧还需 `_allgather_cp_redistribute` 把连续片重排回 zigzag 布局；两分支的 loss 归约差异见 [[15_slime_loss_parallelism_analysis#2.2 从最小实例到整套 loss 层|CP 上的分子可加]]。CP 布局之外还有一处模型侧边界：替换进来的 HF 线性注意力模块（Qwen3-Next/Qwen3.5）在 CP>1 时反向会丢跨 rank 梯度项（`slime_plugins/models/hf_attention.py::_AllGatherForDuplicatedComputation.backward` 只返回本 rank 的梯度切片），见 [[23_slime_model_architecture_extension_analysis|模型架构扩展]] §2.6。多模态输入按键在样本维拼接，路由重放的 `rollout_routed_experts` 由 `prepare_routed_experts_for_routing_replay` 按同一 pad 与切片规则对齐（归 [[17_slime_train_inference_consistency_analysis|训推一致性]]）。

**为何。** 在线路径不走 Megatron 的 GPT Dataset / DataLoader：训练侧只需按预计算下标回放，不需要再解释 RL 语义；THD 打包让变长样本共享一次前向而不必 padding 到最长（本页推断；判据是变长 RL 样本在 batch 维 padding 的浪费）。zigzag 两段是 Megatron THD CP 的既定布局，slime 只是按它切片与对齐 mask，这一点属于依赖契约而非 slime 源码证明；Megatron 侧 CP 怎样接入 attention 与为什么首尾配对均衡，见 [[13_megatron_cp_analysis|Megatron 上下文并行]]（该页按更新的 `85902ef5` 分析，它是镜像钉住的 `1dcf0daf` 的后代）。

**代价与边界。** 补齐到 `tp × data_pad_size_multiplier`（默认 128）的倍数在小 micro-batch 上浪费明显；`pad_token_id` 固定为 0 并带注释"应该没问题？"；`get_batch` 断言 `tokens` 在 keys 中且 mask 与 tokens 同形。`_get_rollout_data` 在设备搬运处挂着 `# TODO: this is ugly, move to somewhere else?`（本页推断：指取数与搬运混在一处）。

#### 2.2.5 前向层：forward_only、logprob、value 与 can_reuse_log_probs_in_loss

**职责。** `forward_only(f, ...)` 重置迭代器，设 eval 模式，可选执行 `custom_megatron_before_log_prob_hook_path`，对每个训练步调 `get_forward_backward_func(forward_only=True)`，回调 `f` 是 `get_log_probs_and_entropy` 或 `get_values`；logprob 路径（`compute_log_prob` 传 `use_rollout_top_p_replay=True`）在 `rollout_top_p != 1.0` 时把 top-p 记录键并入 batch keys 并传给回调作 keep-mask。PP last stage 把各 mb 输出按键拼接，动态批次下再按 `micro_batch_indices` 还原原始顺序；`build_dp_schedule` 让每 rank 的 `micro_batch_indices` 展平恰为 `range(n)`，所以这一还原在冻结基线是恒等映射。`get_log_probs_and_entropy` 在 `rollout_temperature ≠ 1` 时先把 logits 除以温度，再对整条 `[T, V]` logits 一次计算：按 CP 布局构造 shifted target，需要时构造 `[T, vocab_local]` 的 top-p keep-mask（只作用于 logprob，entropy 用未掩码 logits），`calculate_log_probs_and_entropy` 支持按 `log_probs_chunk_size` 分块并在 TP 组上做 vocab-parallel softmax，`entropy_coef == 0` 时不保存 entropy 的反向激活；最后按 CP 布局抽出每条样本的 response 片。`get_values` 走 `get_responses(..., apply_temperature=False)`，value head 输出不除温度。

`forward_only` 传给回调的 `with_entropy` 取自 `--use-rollout-entropy`：开启时 ref、teacher、old-policy 前向顺带返回 `ref_entropy`、`teacher_entropy`、`entropy`，help 给的用途是"做特殊 loss mask"，消费方是 `--rollout-data-postprocess-path` 钩子，另外进入 rollout 日志与训练转储。复用 logprob 时没有 old-policy 前向，所以只会有 `ref_entropy` 而没有无前缀的 `entropy`（源码路径推断）。训练前向的 `policy_loss_function` 总是 `with_entropy=True`，与此开关无关。

**为何。** 对照 logprob 与训练前向共用同一模型与并行拓扑，数值路径一致，这是 [[17_slime_train_inference_consistency_analysis|训推一致性]] 讨论的前提；用推理引擎另算会引入第二套数值实现（本页推断）。`can_reuse_log_probs_in_loss` 的判据是"该步尚未更新参数"：单步、policy loss、`kl_coef == 0`、不用 rollout logprob、不算 mismatch 指标、无 critic、无 old_actor、无 OPD、非 GSPO、路由重放关闭或用 R3 时，训练前向的 detached logprob 就是 old logprob；多步后参数已变，就不能沿用。`get_mismatch_metrics` 即便开 `use_rollout_logprobs` 也会多一次前向（参数校验有日志提示）。value 不除温度是因为温度只属于采样分布，value head 不是 softmax 输出（本页推断）。

**代价与边界。** 每个额外角色一次全批前向；`get_log_probs_and_entropy` 断言 logits 为 float32 且 batch 维为 1；`rollout_top_p != 1.0` 却缺 top-p 记录抛 `ValueError`；`--rollout-temperature ≤ 0` 在参数解析期就被拒绝；动态批次的结果重排挂着两条 TODO（§5.4）。

#### 2.2.6 advantage 位于训练边界内

**职责。** `compute_advantages_and_returns` 只判断 `is_pipeline_last_stage()`；docstring 说"log_probs 与 values 都为空时提前返回"，代码并不这样检查。`kl_coef == 0` 或没有 logprob 时 KL 按 `log_probs or rollout_log_probs or values` 的形状取零，三者都缺时迭代 `None` 抛 `TypeError`；否则对 `log_probs` 与 `ref_log_probs` 做 `compute_approx_kl`（`kl_loss_type` 选 k1/k2/k3/low_var_kl）。自定义 `custom_advantage_function_path` 接管并原地写 `advantages`/`returns`；否则按 `advantage_estimator` 分派：grpo/gspo/cispo 调 `get_grpo_returns` 把标量 reward 广播到每个 token，不读 KL 数值；ppo 把 KL 乘 `-kl_coef`，cp_rank 0 把 reward 加到本地末位后做 GAE；reinforce_plus_plus 做折扣回报，reinforce_plus_plus_baseline 减 `kl_coef × KL`，这三个才读 `kl_coef`。`use_opd` 再减 `opd_kl_coef × (student − teacher)`；`normalize_advantages` 在 DP-with-CP 组上做带 mask 的白化，CP 下先按 token 归属切 mask，且注释强调即便某 CP rank 没有 response token 也必须参加集合通信。随后 `rollout_data_postprocess_path` 钩子可改写整份 rollout_data（含 loss mask），`slime/observability/train_metric_utils.py::log_rollout_data` 用同一 `rollout_mask_sums` 归约后经 DP-with-CP 的 gloo 组 gather 报告。

**为何。** 放进 rollout 服务会迫使推理侧拥有 ref/teacher/critic/current 的 Megatron 数值路径与训练侧并行归一化域，也容易在 DP/CP 切分前后形成两套统计实现；训练边界内所有信号齐备且参数尚未更新（本页推断，与旧版设计分析一致）。`--disable-compute-advantages-and-returns` 可整体关闭，供 SFT 或自定义 loss 使用，所以这是默认所有权而非硬编码。估计器的数值语义与白化的分母见 [[15_slime_loss_parallelism_analysis|Loss 与并行归一化]] 的估计器一节，KL 两个入口见同页 [[15_slime_loss_parallelism_analysis#2.2 从最小实例到整套 loss 层|KL 的两个入口]]。

**代价与边界。** `--advantage-estimator` 的 argparse `choices` 先拒绝未知值，绕过解析改写 args 时 `compute_advantages_and_returns` 才抛 `NotImplementedError`；OPD 缺 `teacher_log_probs` 抛 `ValueError`；REINFORCE++ 系列参数校验强制 `normalize_advantages`；白化全局 mask 和为 0 抛 `ValueError`；ppo 分支遇到 cp0 本地片为空的样本抛 `IndexError`（§2.1.5）。

#### 2.2.7 训练步：train、train_one_step 与 optimizer 边界

**职责。** `train` 断言 `num_microbatches` 与 `global_batch_sizes` 等长，重置迭代器，设 train 模式，把 `optimizer.scale_loss` 装为 `grad_scale_func`，`overlap_grad_reduce` 时断言 `no_sync_func` 为空后装 DDP 的 `no_sync`（`align_grad_reduce` 再装 `start_grad_sync`），`overlap_param_gather` 且 `align_param_gather` 时装 `start_param_sync`，`finalize_model_grads_func` 用 Megatron 的 `finalize_model_grads`。`reset_optimizer_states` 在此处、即每轮第一个训练步之前把每个 chained optimizer 的 `step`、`exp_avg`、`exp_avg_sq` 清零（与 stateless Adam 的区别见 §4.5）；`manual_gc` 调 `gc.disable(); gc.collect()` 关闭自动 GC 以对齐各 rank 的回收时机；slime 此后不再打开它，回收只发生在 `clear_memory` 等显式 `gc.collect()` 处，`manual_gc_interval` 只被断言非负。使用 distributed optimizer 且 `overlap_param_gather` 时，第一步前禁用 forward pre-hook 并暂时摘掉 `param_sync_func`，第 0 步成功后再启用，训练结束再关闭。每步调用 `train_one_step`（§2.1.4）。MTP 训练时前向额外传 `mtp_kwargs={"mtp_labels": tokens}`，这个参数由镜像的 `megatron.patch` 加入，上游 `1dcf0daf` 的 `GPTModel` 没有它；slime 侧的 `1 / num_microbatches` 只缩放记入日志的 `train/mtp_*_loss`；反向的缩放由 Megatron 的 `MTPLossAutoScaler` 完成，`megatron/core/pipeline_parallel/schedules.py::forward_step_calc_loss` 在非 per-token 模式下把它设为 `loss_scale / num_microbatches`（`loss_scale` 取 `grad_scale_func(1)`，per-token 时不除）（已在 `1dcf0daf` 核对，镜像补丁未改这两处）。MTP 梯度只流进 MTP 层的隔离来自 `megatron.patch`，机制见 [[21_slime_speculative_decoding_mtp_analysis#2.2 从最小实例到整个闭环|Megatron 补丁里的 MTP 机制]]。主 rank 记录 `train/<key>`、`grad_norm`、每个 param group 的 lr、该步 `global_batch_size` 与 `train/step = rollout_id × num_steps_per_rollout + step_id`，CI 模式下断言 `ppo_kl`、`kl_loss` 与 logprob 差异阈值。`train_one_step` 在 `check_for_nan_in_loss_and_grad` 为假时自行 `prepare_grads` 并检查 `grad_norm` 的 inf/nan 决定 `valid_step`；无效步跳过更新但仍清梯度。

**为何。** 让 Megatron 的流水线函数拥有 micro-batch 循环，slime 只在 closure 里取数、打包、返回 loss 回调，PP/VPP schedule、梯度通信与 DDP hook 都不必重写（本页推断，判据同 §2.2.1）。`opt_param_scheduler.step(increment=step_global_batch_size)` 的源码注释说明用每步真实 rollout 数让 scheduler 的 samples-seen 计数贴近现实；它与 loss 缩放共用同一个 G，二者的耦合归 [[15_slime_loss_parallelism_analysis|Loss 与并行归一化]]。梯度规约留给 Megatron：`finalize_model_grads` 是原样的上游函数，`c403335d` 删除了 slime 曾对它的 TP 侧梯度打的分块 all-reduce 猴补（`slime/backends/megatron_utils/megatron_patch/`，由 `SLIME_GRAD_COALESCE_CHUNK_BYTES` 控制块大小），`slime/backends/megatron_utils/__init__.py` 现在只剩一处 DeepEP 补丁：构造 `deep_ep.Buffer` 时暂时关掉 torch_memory_saver 的追踪区域，让这些常驻 buffer 的生命周期不受卸载影响（依赖侧语义按注释理解）。DDP 梯度规约、distributed optimizer 的 reduce-scatter → step → all-gather 与 overlap hook 的内部见 [[16_megatron_distributed_optimizer_analysis|Megatron 分布式优化器]]（按更新的 `85902ef5` 分析）。

**代价与边界。** `optimizer.step` 返回失败直接断言；`overlap_grad_reduce` 与自定义 `no_sync_func` 互斥；`return_schedule_plan`（combined 1F1B）不能与 MTP 训练同开。梯度累积的数值缩放链（loss 预缩放、Megatron 除 M、DDP 平均）在本页只到"交给 Megatron"为止，账本见 [[15_slime_loss_parallelism_analysis#2.2 从最小实例到整套 loss 层|loss_function 的缩放链]]；该页以 `1dcf0daf` 核对缩放，Megatron 在 `85902ef5` 前后又改过缩放钩子，所以链接 Megatron 域的 loss 缩放页时要以版本差为准。

#### 2.2.8 sleep、wake_up 与两种 checkpoint

**职责。** `sleep` 断言 `offload_train`，清理显存与主机内存，actor 在有 critic 且非 colocate 时先让支持该接口的 weight updater `disconnect_rollout_engines`，再 `destroy_process_groups` 并 `torch_memory_saver.pause()`；`wake_up` 反向恢复，PP > 2 时在恢复后先做一次 `dist.barrier` 预热 WORLD（注释解释 Megatron 打补丁的批量 P2P 用默认组，PP=4 时前两个 stage 先进入 `batch_isend_irecv` 会触发 NCCL 首次初始化要求所有 rank 到场），actor 再 `_switch_model("actor")`。`train` 只在 `offload_train` 开启时包一层 wake → 训练 → `del rollout_data` → sleep；critic 初始化后若开 offload 立即 sleep。`save_model` 在 offload 下先 wake，`async_save` 时先阻塞 finalize 上一次异步保存，`save` 在关闭 forward pre-hook 的窗口内调 Megatron `save_checkpoint`，以 `rollout_id` 作 iteration，`force_sync` 才再次阻塞等待本次；续训时 `start_rollout_id` 取加载的 iteration 加一，有 critic 时采用 critic 的，与 DataSource 游标的切点归 [[18_slime_fault_tolerance_observability_analysis|容错与可观测性]]；分片落盘与异步完成阶梯归 Megatron，见 [[19_megatron_dist_checkpointing_analysis|Megatron 分布式 checkpoint]]（按更新的 `85902ef5` 分析）。`save_hf` 仅 actor 执行，`save_hf_model_to_path` 断言输出目录不等于 `hf_checkpoint` 且后者是本地目录，rank 0 清除已有 HF 权重、复制资产并广播模型名与量化配置，再用 `HfWeightIteratorDirect` 转换权重、各节点 writer 按 chunk 写 safetensors 并汇总索引；它不保存 optimizer。

**为何。** 角色 backup 只保存可按 tag 恢复的参数与缓冲区，`sleep`/`wake_up` 则暂停整个训练 GPU 状态并销毁通信组，两者解决的是不同问题：前者是角色切换，后者是让 rollout 引擎分时用卡（[[16_slime_weight_sync_analysis|权重同步]] 讨论 colocate 下的完整生命周期）。销毁方式上被否的方案是逐个销毁子组：`_destroy_default_accelerator_process_group` 的注释记录，纯 PP=4 时各 pipeline rank 持有的 singleton、embedding、PP 子组集合并不相同，逐个销毁本地 wrapper 会让 rank 0 已进入子组重建、另一个 rank 仍在拆除，rank 0 于是永久阻塞在 `new_group()`。现实现先做一次 gloo barrier，再一次性销毁 WORLD：PyTorch 按全局顺序关闭所有已注册的 NCCL 与 Gloo 后端并释放全部通信器显存，随后建一个临时 Gloo WORLD，`wake_up` 时再换回加速器后端（判据是多 rank 拆除顺序能否一致）。另一个被否方案是 sleep 时保留 process group：wake 更快，但分时共卡时 rollout 引擎需要那块通信器显存（本页推断）。销毁也有代价：`wake_up` 的注释说明 PyTorch 要求组上首个 NCCL 操作所有 rank 都到场，而 PP=4 重建后只有前两个 stage 先进入批量 P2P，所以 PP > 2 时先做一次 barrier。`release_train` 是另一条兄弟轴：不卸载而是释放并重建 Megatron actor，参数校验禁止它与 critic、`keep_old_actor` 同用，归 [[11_slime_ray_control_plane_analysis|Ray 控制面]]。

**代价与边界。** 每轮销毁与重建 process group：PPO 下训练阶段每 GPU 两次，`update_weights` 再一次（§2.1.5，[[16_slime_weight_sync_analysis|权重同步]]）。默认（环境变量 `SLIME_DESTROY_WORLD_PROCESS_GROUP` 未设，或取 `0`、`false`、`no` 以外的值，大小写不敏感）连 WORLD 一起销毁并在 wake 时重建，外部代码缓存的原始 `dist.group.WORLD` 引用因此失效；源码注释要求这类引用会跨过 sleep/wake 时把它设为 0（`false`、`no` 同效）。这时 `destroy_process_groups` 退回 `ReloadableProcessGroup.destroy_process_groups` 逐个销毁子组，也就是上面注释记录过拆除死锁的路径，WORLD 的通信器也不在 sleep 时释放（后者为本页推断）。`--save-hf` 以 `save_hf.format(rollout_id=rollout_id)` 展开：模板必须用具名占位 `{rollout_id}`；help 写的是位置参数形式 `save_hf.format(rollout_id)`，照它写 `{}` 会抛 `IndexError`（文档与实现冲突，以实现为准）；不带占位时每次保存都清空并重写同一目录，没有守卫。HF 保存对已有目录先清除后重写，没有目录级事务回滚。

### 2.3 变体：同一实例在五条选择轴上

| 选择轴 | 枚举依据 | 变体 | 本例的表现 | 压力与上限 |
|---|---|---|---|---|
| 打包 | `_pack_step_into_mbs` 三个分支 | first-fit（默认动态）/ 静态切块 / `balance_by_flops` KK | first-fit 得 K=3 再拆成 4；静态 K=3 直接断言；`balance_by_flops` 先定 `ceil(42/18)=3` 个 bin，KK（`equal_size=False`）对任意正系数的 `aL+bL²` 都分出 `{s0}`、`{s1,s2b}`、`{s2a,s3}`，再把 `{s1,s2b}` 拆成 `[s2b]` 与 `[s1]` 凑到 4；随后被强制打开的 `balance_data` 在 a > 22b 时给 rank 0 `[s2b] [s0]`、rank 1 `[s2a s3] [s1]`，a ≤ 22b 时给 rank 0 `[s2b] [s2a s3]`、rank 1 `[s0] [s1]`。真实模型按 `calculate_fwd_flops` 逐项归并，非 MLA 稠密层的 a/b ≈ 2h + 2·n_kv·d_head + 3·ffn + V/层数，Qwen2.5-0.5B 的形状约 2.3×10⁴，远落在前一侧（本页推导）；本例最大 bin 14 个 token 未超 cap，但此路径不保证 cap | 变长样本的 GPU 利用率；上限是 `max_tokens_per_gpu × cp_size` 与 OOM |
| 分发 | `args.balance_data` | 轮询 / Karmarkar-Karp（`equal_size=True`） | 轮询把 r2 的两个片段都给 rank 1；KK 对任意 `aL+bL²` 工作量也配出同样的 `{bin0,bin2}` / `{bin1,bin3}` | rank 间 wall time 差；均衡对象是估算计算量，只在一步之内调配 |
| CP 布局 | `args.allgather_cp`；CP>1 时 `_validate_allgather_cp_supported` 只放行 DSA 架构 | zigzag 逐样本两段 / 全局拼接后连续等分 | zigzag 下两个 rank 各 16 位、有效 mask 2 与 7；allgather 下 cp0 为 s0 全部加 s1 前 4 位（有效 8），cp1 只有 2 个真实 token（有效 1），logprob 经 `_allgather_cp_redistribute` 一次可微 all-reduce 切回 zigzag，loss 侧在 `allgather_cp` 且 CP>1 时加 `0 * logits.sum()` 防止反向死锁 | zigzag 让因果注意力负载接近均衡；allgather 服务需要连续全局序列的 DSA 稀疏注意力，非 DSA 模型在解析期即被拒绝；代价是本例这种 rank 间真实 token 失衡与额外 all-reduce（未测量） |
| old logprob 来源 | `train_actor` 的条件链 | 复用训练前向 / 独立 old-policy 前向 / `use_rollout_logprobs` | 本例复用，一轮 2 次全批前向（2 次 `forward_backward_func` 调用）；两步时 3 次全批前向、6 次调用 | 多一次全批前向的时间；`use_rollout_logprobs` 与 `use_tis` 互斥 |
| 角色集合 | `init` 的 `with_ref`、`with_opd_teacher`、`keep_old_actor`、`use_critic` | actor 单独 / +ref / +teacher / +old_actor(+rollout_actor) / PPO 独立 critic | 本例 actor+ref；PPO 时（§2.1.5）critic 先训练并回传 V_old，每 GPU 一轮 5 次全批前向、2 次 optimizer step | 每个 tag 一份 pinned 内存与一次切换；critic 与 actor 共用同一批 GPU（同一 placement group，每个 worker 声明 `num_gpus=0.4`），多出一份模型与 optimizer 状态，靠强制 `offload_train` 分时占用 |

训练后端本身的兄弟轴是 `--train-backend`：`_pre_parse_mode` 把它限定为 `choices=["megatron"]`，`slime/backends/` 下也只有 `megatron_utils` 与 `sglang_utils`，冻结基线没有第二个训练后端；真正的替换点是 `RayTrainGroup` 的 `actor_cls`（归 [[11_slime_ray_control_plane_analysis|Ray 控制面]]）。Megatron teacher 以 tag 装入同一模型槽，必须与 actor 同架构；异构 teacher 走 `--opd-type sglang`（归 [[20_slime_on_policy_distillation_analysis|在线策略蒸馏]]）。角色 YAML 覆盖见 §4.1，哪些参数参与训练由 §4.6 的冻结决定。

### 2.4 整体开销

| 维度 | 来源 | 评估状态 |
|---|---|---|
| 计算 | 每个对照角色一次全批前向；多步时额外 old-policy 前向；PPO 每轮 critic 2 + actor 3 次全批前向；`get_log_probs_and_entropy` 对整条 `[T, V]` 做一次 softmax | 源码可见，未测量 |
| 内存 | 每个 tag 一份 pinned host 副本；THD 补齐到 `tp × 128` 倍数；entropy 反向激活仅在 `entropy_coef ≠ 0` 时保存；PPO 多一份模型与 optimizer 状态；stateless Adam 不分配两份 Adam 矩 | 源码可见 |
| 传输 | 每轮全模型 CPU→GPU 拷贝若干次；schedule 与样本经 Ray object store 的 CPU 张量到达；PPO 的 V_old 经 CPU 从 critic 交给 actor | 源码注释自承潜在瓶颈，未测量 |
| 同步 | 每步 `optimizer.step` 内的 DP 通信（Megatron）；`sleep`/`wake_up` 销毁重建 process group（PPO 下训练阶段每 GPU 两次，`update_weights` 再一次）；PP>2 额外 barrier；每次 backup/restore 一次设备同步 | 源码可见 |
| 兼容性 | 与 Megatron `1dcf0daf` 及 `megatron.patch` 耦合；`ShardedTensor` 校验猴补；actor/critic 需同拓扑；`--allgather-cp` 仅 DSA 架构 | 源码与官方文档 |
| 实现复杂度 | 取数与搬运混在一处、动态批次结果重排、`is_megatron_main_rank` 判定方式均带 TODO | 源码可见 |

**总体代价与运行包络。** 训练后端把 RL 特有的角色顺序、数据 ABI 与 loss 接点留在适配层，换来 Megatron 原生并行与优化器不被改写；代价集中在角色切换的拷贝、对照前向与卸载时的通信组重建，PPO 把后两项翻倍。失败边界是一组早断言与解析期 `ValueError`：schedule 的四条（步数、每步样本数、对齐、拆分）、`get_batch` 的形状断言、`optimizer.step` 成功断言、allgather CP 的架构校验、stateless Adam 与 HF 保存的路径断言（§5.1）；也有几处没有守卫、只能靠配置避开的边界（PPO 的 cp0 空片、`--load` 指向 `iter_` 目录、`--save-hf` 缺占位）。本页未运行 slime 训练，所有耗时判断均为源码推断。

---

## 3. 代码实现分析

### 3.1 对象与所有权视图

<!-- Figure spec: ownership graph; driver owns RayTrainGroup(s); each MegatronTrainRayActor owns Megatron model chunks, optimizer, scheduler, TensorBackuper tags, weight updater; RolloutManager computes schedule and hands per-rank Box refs; DataIterator per VPP stage reads the local dict; Megatron pipeline executes closures. -->
```mermaid
flowchart TB
    DR["train.py driver<br/>rollout_id 循环"]
    AG["RayTrainGroup actor<br/>（critic 另一组，共享 pg）"]
    TA["MegatronTrainRayActor<br/>model chunks、optimizer、scheduler、weight_updater"]
    TB["TensorBackuper<br/>CPU tags: actor ref teacher old_actor rollout_actor"]
    RM["RolloutManager<br/>build_dp_schedule → Box × dp_size"]
    DI["DataIterator × vpp_size<br/>micro_batch_indices 回放"]
    GB["get_batch<br/>CP 切片、THD、mask 对齐"]
    MG["Megatron 流水线<br/>forward_backward_func、DDP、optimizer.step"]
    DR --> AG --> TA
    TA --> TB
    RM -->|rollout_data_ref| TA
    TA -->|process_rollout_data → CUDA| DI --> GB --> MG
    TB -->|_switch_model restore| TA
    MG -->|backup actor| TB
```

| 对象 | 所在进程 | 拥有的状态 | 生命周期 |
|---|---|---|---|
| `RayTrainGroup` | driver | actor handles、`master_addr/port`、release/disk 权重版本；PPO 时另有一组 critic | 训练全程 |
| `MegatronTrainRayActor` | 每 GPU 一个 Ray actor（PPO 时每 GPU 再有一个 critic 实例） | `args`、`model`、`optimizer`、`opt_param_scheduler`、`weights_backuper`、`_active_model_tag`、`weight_updater`（`create_weight_updater` 构造）、`train_parallel_config`、`prof`；critic 实例没有 backuper 与 weight updater | 训练全程；`release_train` 下按轮重建 |
| `TensorBackuper` | 同上 | 每 tag 一份 pinned CPU dict | 训练全程 |
| Megatron 全局 | 同上 | `mpu` 并行组、`get_args()`、microbatch calculator、RNG tracker | `sleep` 销毁通信组，`wake_up` 重建 |
| per-rank rollout dict | 从 Ray object store 取到 CPU 再搬加速器 | `partition`、逐样本字段、`micro_batch_indices`、`num_microbatches`、`global_batch_sizes`、`total_lengths`（全量→本地）、`raw_reward`（全量）+`local_raw_reward` | 一轮内 |
| `DataIterator` | 同上 | `rollout_data` 引用与 `offset` | 一轮内，多次 `reset` |
| 训练步 closure | 流水线回调 | `batch`（含 `unconcat_tokens`、`packed_seq_params`、`full_loss_masks`）与 loss 回调的 partial | 一个 micro-batch 内 |

### 3.2 调用流程

#### 3.2.1 一轮训练：从 driver 到 optimizer step

```text
train.py::train
|-- actor_trains = (not use_critic) or rollout_id >= num_critic_only_steps
|-- [use_critic] critic_model.async_train(rollout_id, ref) → value_refs；[¬actor_trains] ray.get(value_refs) 后本轮不训练 actor
`-- [actor_trains] ray.get(actor_model.async_train(rollout_id, rollout_data_ref[, external_data=value_refs]))
    `-- RayTrainGroup.async_train → worker i: actor.train.remote(..., external_data=value_refs[i])
        `-- MegatronTrainRayActor.train
            |-- [offload_train] wake_up → memory_saver.resume、reload_process_groups、[PP>2] barrier、_switch_model("actor")
            |-- _get_rollout_data → slime/utils/data.py::process_rollout_data（ray.get 本 dp_rank）→ 张量搬设备、logprob 按 CP 切片
            |-- [critic] train_critic：forward_only(get_values) → compute_advantages_and_returns → loss_type=value_loss → train → values 回 CPU
            `-- train_actor
                |-- get_data_iterator（每 VPP stage 一个）
                |-- [use_rollout_routing_replay] fill_routing_replay                                   [归训推一致性页]
                |-- [compute_advantages_and_returns]
                |   |-- ["ref" in tags] _switch_model("ref") → compute_log_prob(store_prefix="ref_")
                |   |-- ["teacher" in tags] _switch_model("teacher") → compute_log_prob("teacher_")
                |   |-- _switch_model("old_actor" | "actor") → [¬can_reuse ∧ (¬use_rollout_logprobs ∨ mismatch)] compute_log_prob("")
                |   |-- [use_critic ∧ PP last] rollout_data["values"] ← tensors_to_gpu(external_data["values"])
                |   |-- [_active_model_tag ≠ actor] _switch_model("actor")
                |   `-- compute_advantages_and_returns（只在 PP last stage 计算）
                |-- [rollout_data_postprocess_path] hook(args, rollout_id, rollout_data)
                |-- train_metric_utils.log_rollout_data → gather_log_data（DP-with-CP gloo）
                |-- [save_debug_train_data ∧ 无 log_probs] enable_log_prob_capture
                |-- slime/backends/megatron_utils/model.py::train
                |   |-- config.grad_scale_func / no_sync_func / grad_sync_func / param_sync_func / finalize_model_grads_func
                |   |-- [reset_optimizer_states] 清 step、exp_avg、exp_avg_sq
                |   `-- for step_id: train_one_step
                |       |-- zero_grad_buffer、optimizer.zero_grad、[before_train_step hook]
                |       |-- get_forward_backward_func()(forward_step, K=num_microbatches[step_id], forward_only=False)
                |       |   `-- forward_step → get_batch → model(**kwargs) → partial(loss_function, ...)   [归 Loss 页]
                |       |-- [valid_step] optimizer.step() → assert → opt_param_scheduler.step(increment=global_batch_sizes[step_id])
                |       `-- [PP last] train_metric_utils.reduce_train_step_metrics
                |-- [capture] drain_captured_log_probs → 按 partition 回填 rollout_data["log_probs"]
                |-- train_data_utils.save_debug_train_data
                |-- weights_backuper.backup("actor")；[ref_update_interval 到期] backup("ref")
                `-- train_metric_utils.log_perf_data(extra_metrics=weight_updater.pop_metrics())
            `-- [offload_train] del rollout_data → sleep
```

完成边界是 `optimizer.step` 成功并 `backup("actor")`：新参数已在 GPU 与 CPU `actor` tag 上可见，但尚未发布给 rollout 引擎；发布归 [[16_slime_weight_sync_analysis|权重同步]]。

#### 3.2.2 计划与批次：从 build_dp_schedule 到 forward_step 的 batch

```text
slime/ray/rollout.py::RolloutManager._split_train_data_by_dp                        [RolloutManager 进程；接口归 Sample 页]
`-- slime/utils/dp_schedule.py::build_dp_schedule(args, train_parallel_config, total_lengths, global_batch_size, rollout_indices)
    |-- 按 rollout id 分组 → num_steps = len(rollout_ids) // global_batch_size（assert ≥ 1）
    `-- 每步：assert len(samples) ≥ dp_size
        |-- _pack_step_into_mbs → first_fit_pack | 静态切块 | get_seqlen_balanced_partitions(KK, equal_size=False)
        |-- target_K 对齐 → [动态] expand_bins_by_splitting → _split_bin_by_tokens；[静态] raise AssertionError
        |-- [balance_data] get_seqlen_balanced_partitions(mbs FLOPs, dp_size, equal_size=True) → karmarkar_karp | 轮询 range(r, K, dp)
        `-- partitions[r] / micro_batch_indices[r] 累加
trainer 进程
`-- train_one_step.forward_step | forward_only.forward_step
    `-- slime/backends/megatron_utils/data.py::get_batch(data_iterator, keys, data_pad_size_multiplier, allgather_cp)
        |-- DataIterator.get_next(keys) → 按 micro_batch_indices[offset] 取子集，offset += 1
        |-- unconcat_tokens ← tokens
        |-- [allgather_cp] 全局拼接 → 补到 cp·pad_size → chunk(cp)[cp_rank]；cu_seqlens 全局
        |-- [zigzag] 每样本 slice_with_cp → 拼接 → 补到 pad_size → cu_seqlens × cp_size
        |-- PackedSeqParams(thd)，tokens.unsqueeze(0)
        |-- mask：F.pad(prompt_len−1, 1) → 同规则切片 → 拼接补齐 → assert 同形 → full_loss_masks
        `-- 多模态张量按键拼接
```

#### 3.2.3 初始化、角色装载与保存

```text
slime/utils/arguments.py::parse_args
`-- megatron_parse_args → _hf_validate_args、_validate_allgather_cp_supported、_set_default_megatron_args
RayTrainGroup.create → actor.init(args, role, with_ref, with_opd_teacher)
|-- [debug_rollout_only] return 0
|-- monkey_patch_torch_dist → TrainRayActor.init（accelerator.set_device、dist.init_process_group、init_gloo_group、NUMA 亲和）
|-- [SLIME_DESTROY_WORLD_PROCESS_GROUP 不是 0/false/no] register_default_process_group
|-- slime/backends/megatron_utils/initialize.py::init(args)（末尾调 custom_megatron_init_path）
|-- 逐 GPU 串行读 AutoConfig / AutoTokenizer（gloo barrier）
|-- initialize_model_and_optimizer
|   |-- setup_model_and_optimizer：get_model(get_model_provider_func → freeze_model_params)
|   |   |-- [num_rollout == 0] return model, None, None
|   |   `-- OptimizerConfig ← args → [stateless] _patch_megatron_adam → get_megatron_optimizer → get_optimizer_param_scheduler
|   |-- _critic_output_layer_needs_reinit（读 dist-ckpt 元数据）
|   |-- load_checkpoint → [tracker 文件 | iter_XXXXXXX] Megatron load | _load_checkpoint_hf（iteration 0）
|   `-- [需要] _reinitialize_critic_output_layer → reload_model_params
|-- train_parallel_config（dp 不含 cp、cp、vpp、mb_group）
|-- [critic] [offload_train] sleep → return start_rollout_id
|-- TensorBackuper(source_getter) → backup("actor")
|-- [with_ref] load_other_checkpoint("ref", ref_load)；[with_opd_teacher] ("teacher", opd_teacher_load)
|-- [keep_old_actor] load_other_checkpoint("old_actor", load)；[interval==1] backup("rollout_actor")
|-- create_weight_updater（按 mode/transport/colocate 四路选择）                                   [归权重同步页]
`-- [offload_train] _switch_model("actor") → sleep
save_model(rollout_id, force_sync)
|-- [offload_train] wake_up
|-- [async_save] maybe_finalize_async_save(blocking=True)
|-- slime/backends/megatron_utils/model.py::save → [should_disable_forward_pre_hook] disable_forward_pre_hook → megatron save_checkpoint → [同条件] enable_forward_pre_hook
|-- [force_sync ∧ async_save] maybe_finalize_async_save(blocking=True)
|-- [save_hf ∧ actor] slime/backends/megatron_utils/hf_checkpoint_saver.py::save_hf_model_to_path(save_hf.format(rollout_id=rollout_id))
`-- [offload_train] sleep
```

### 3.3 源码阅读路线

1. actor 生命周期：`slime/ray/train_actor.py::TrainRayActor.init` / `set_rollout_manager` → `slime/backends/megatron_utils/actor.py::MegatronTrainRayActor.init` / `sleep` / `wake_up` / `_get_rollout_data` / `_switch_model` / `compute_log_prob` / `train` / `train_critic` / `train_actor` / `save_model` / `load_other_checkpoint` → `slime/utils/reloadable_process_group.py::register_default_process_group` / `_destroy_default_accelerator_process_group` / `destroy_process_groups` / `reload_process_groups` / `ReloadableProcessGroup.destroy_process_groups` → `slime/utils/tensor_backper.py::TensorBackuper.backup` / `restore` / `copy` / `get` → `slime/backends/megatron_utils/update_weight/__init__.py::create_weight_updater`。
2. 模型与优化器：`slime/backends/megatron_utils/model.py::setup_model_and_optimizer` / `get_optimizer_param_scheduler` / `_patch_megatron_adam` / `_disable_distributed_optimizer_state_initialization` / `_critic_output_layer_needs_reinit` / `_reinitialize_critic_output_layer` / `initialize_model_and_optimizer` → `slime/backends/megatron_utils/stateless_adam.py::StatelessAdam.step` → `slime/backends/megatron_utils/model_provider.py::get_model_provider_func` / `wrap_model_provider_with_freeze` / `freeze_model_params` / `_is_indexer_parameter` / `LinearForLastLayer` → `tests/test_stateless_adam.py`、`tests/test_model_provider_freeze.py`。
3. checkpoint：`slime/backends/megatron_utils/checkpoint.py::load_checkpoint` / `_is_megatron_checkpoint` / `_load_checkpoint_hf` → `slime/backends/megatron_utils/model.py::save` → `slime/backends/megatron_utils/hf_checkpoint_saver.py::save_hf_model_to_path` / `_finalize_distributed_shards` / `_clear_existing_hf_weights`。
4. 参数与并行组：`slime/backends/megatron_utils/arguments.py::megatron_parse_args` / `_validate_allgather_cp_supported` / `_set_default_megatron_args` / `validate_args` → `tests/test_megatron_argument_validation.py::test_allgather_cp_rejects_non_dsa_cp_models` → `slime/backends/megatron_utils/initialize.py::init` / `_initialize_distributed` / `_set_random_seed` / `is_megatron_main_rank` → `slime/utils/distributed_utils.py::init_gloo_group` / `get_gloo_group` / `distributed_masked_whiten`。
5. 调度：`slime/ray/rollout.py::RolloutManager._split_train_data_by_dp` → `slime/utils/dp_schedule.py::build_dp_schedule` / `_pack_step_into_mbs` → `slime/utils/seqlen_balancing.py::first_fit_pack` / `_split_bin_by_tokens` / `expand_bins_by_splitting` / `karmarkar_karp` / `get_seqlen_balanced_partitions` → `slime/utils/flops_utils.py::calculate_fwd_flops` → `tests/test_dp_schedule.py`。
6. 取数与打包：`slime/utils/data.py::process_rollout_data` → `slime/backends/megatron_utils/data.py::DataIterator` / `get_data_iterator` / `get_batch` / `tensors_to_cpu` / `tensors_to_gpu` → `slime/backends/megatron_utils/cp_utils.py::slice_with_cp` / `get_logits_and_tokens_offset_with_cp` / `slice_log_prob_with_cp` / `prepare_routed_experts_for_routing_replay` → `tests/test_process_rollout_data.py`。
7. 前向、advantage 与训练步：`slime/backends/megatron_utils/model.py::forward_only` / `train_one_step` / `train` / `should_disable_forward_pre_hook` → `slime/backends/megatron_utils/loss.py::get_log_probs_and_entropy` / `_build_shifted_tokens` / `_build_topp_keep_mask` / `_extract_per_sample` / `_allgather_cp_redistribute` / `get_values` / `compute_advantages_and_returns` / `loss_function` / `enable_log_prob_capture` / `drain_captured_log_probs` → `slime/utils/ppo_utils.py::get_grpo_returns` / `calculate_log_probs_and_entropy` / `_VocabParallelLogProbEntropy` → `tests/test_value_temperature.py` → `slime/observability/train_metric_utils.py::log_rollout_data` / `gather_log_data` / `reduce_train_step_metrics` / `log_perf_data` → `slime/observability/train_data_utils.py::save_debug_train_data`。
8. 角色、PPO 与参数：`slime/ray/placement_group.py::create_placement_groups` / `create_actor_model` / `create_training_models` → `slime/ray/actor_group.py::RayTrainGroup.async_train` → `slime/utils/arguments.py::_apply_megatron_role_overrides` / `parse_megatron_role_args` / `slime_validate_args`（`use_critic`、`offload_train`、`num_steps_per_rollout`、`balance_by_flops`、`rollout_temperature`、冻结名单互斥的归一化与校验）→ `tests/utils/test_megatron_role_config.py` → `train.py::train`。

---

## 4. 配套机制

### 4.1 Role YAML 只改角色参数，不接管资源

`parse_megatron_role_args` 要求 YAML 顶层有 `megatron` 列表，每个 role 至多一条，缺失角色继承 CLI；`_apply_megatron_role_overrides` 深拷贝共享 args 后应用 `overrides`（兼容旧键 `args`），忽略 `num_nodes` 与 `num_gpus_per_node`，对科学计数法字符串按原属性类型强转，未知键仍设置但告警；critic 强制 `kl_coef=0`、`use_opd=False`、`custom_advantage_function_path=None`、`untie_embeddings_and_output_weights=True`，且未显式覆盖时 `disable_param_buffers_cpu_backup=False`。这些强制项只在给了 `--megatron-config-path` 时生效；不给时 `create_training_models` 直接 deepcopy CLI 参数，只把 `disable_param_buffers_cpu_backup` 改回假（未开 `--use-rollout-logprobs` 时 critic 的 KL 仍为零，因为它从不算 logprob；开了且 `kl_coef ≠ 0` 会抛 `TypeError`，§2.1.5）。单测覆盖 actor/critic 独立覆盖、critic 强制项、缺失角色继承、缺顶层键报错与无 critic 时 actor 覆盖生效。官方文档把边界说得更窄：主要服务 PPO actor/critic，资源仍由 CLI 控制，actor 与 critic 当前必须相同 Megatron 并行拓扑并共享 train placement group；`--advantage-estimator ppo` 会自动启用 critic，不需要额外 flag。

### 4.2 钩子与自定义入口

训练后端在固定基线暴露六个函数路径：`custom_megatron_init_path`（`slime/backends/megatron_utils/initialize.py::init` 末尾，模型构造之前，签名 `(args)`）、`custom_megatron_before_log_prob_hook_path`（每次 `forward_only` 前，带 `store_prefix`）、`custom_megatron_before_train_step_hook_path`（每步清梯度后）、`rollout_data_postprocess_path`（advantage 之后、日志之前，可改 loss mask）、`custom_advantage_function_path`（替换内置估计器）、`custom_model_provider_path`（替换模型构造，critic 仍换 value head，冻结包装照常生效）。`loss_type=custom_loss` 与 `custom_pg_loss_reducer_function_path` 属于 loss 回调内部的扩展点，签名与归约契约见 [[15_slime_loss_parallelism_analysis#4.1 自定义 loss 与 pg reducer|自定义 loss 与 pg reducer]]。官方 customization 文档把 custom loss 定位为新 RL 目标、多目标或正则项，而不是替换 trainer。

### 4.3 调试转储与 logprob 捕获

`save_debug_train_data` 设定时，若本轮没有独立算出 `log_probs`（复用或 `use_rollout_logprobs`），`train_actor` 在训练前 `enable_log_prob_capture`，`policy_loss_function` 内 `_maybe_capture_log_probs` 按 batch 里的 `partition`（仅此时加入训练 keys）记录每条样本首次出现的 CP-local logprob，训练后 `drain_captured_log_probs` 并按本 rank `partition` 回填；`slime/observability/train_data_utils.py::save_debug_train_data` 再把各 DP shard 的 CP 切片字段还原成完整 response 后写一个文件。参数校验拒绝 `--save-debug-train-data` 与 `--save-debug-rollout-data` 取同一路径（`ValueError`）。转储格式与恢复边界归 [[18_slime_fault_tolerance_observability_analysis|容错与可观测性]]。

### 4.4 路由重放接点

`use_rollout_routing_replay` 时 `fill_routing_replay` 要求 rollout dict 带 `rollout_routed_experts`，逐 micro-batch 用 `prepare_routed_experts_for_routing_replay` 对齐到训练 token 布局，再按每个 VPP stage 的层偏移把 MoE 层的路由记录进 `RoutingReplay`，断言记录数等于全部 replay 对象；`ROUTING_REPLAY_STAGE` 环境变量在 ref/teacher 前向设 `fallthrough`、old-policy 前向设 `replay_forward` 或 `record`、训练设 `replay_backward`。critic 组不做路由重放。机制与一致性归 [[17_slime_train_inference_consistency_analysis|训推一致性]]。

### 4.5 优化器状态重置：`--reset-optimizer-states` 与 `--use-stateless-adam`

两个开关都让 Adam 不带着上一轮的矩进入新数据，区别在清零的时机与是否保留张量。`--reset-optimizer-states` 在 `slime/backends/megatron_utils/model.py::train` 开头、即每个 rollout 轮第一个 `train_one_step` 之前，遍历 `optimizer.chained_optimizers`，把 param group 与 state 里的 `step` 置 0、`exp_avg` 与 `exp_avg_sq` 清零；张量仍然存在，轮内后续步照常累积，未设 `--no-save-optim` 时 checkpoint 也照常保存这些矩。它的 help 写的是"每个 rollout 结束时清空"，实现是在下一次 `train` 开始时清：更新序列相同，差别只在两轮之间保存的 checkpoint 仍带着矩（文档与实现不一致，以实现为准）。

`--use-stateless-adam` 让每个 optimizer step 都从零矩起算。`StatelessAdam.step` 把 `step` 固定为 1，偏差修正后的一阶、二阶矩恰为 g 与 g²，默认 `bias_correction` 下更新退化为 `p ← p − lr · g / (|g| + eps)`（AdamW 模式先乘 `1 − lr·wd`），近似逐元素的符号更新；它不分配 `exp_avg`/`exp_avg_sq`，distributed optimizer 的状态初始化被换成空操作，`load_state_dict` 丢弃 state，因此要求 `--optimizer adam` 与 `--no-save-optim`，且不支持 amsgrad。`tests/test_stateless_adam.py::test_stateless_adam_matches_reinitialized_adam_each_step` 锁定它与"每步重建一个 Adam"逐步一致，`test_stateless_adam_does_not_persist_moment_tensors` 锁定不留矩张量。

由此可推出（本页推导）：`num_steps_per_rollout=1` 时两个开关的参数更新相同，stateless 版本还省下每个 DP 分片两份与主参数同形的 FP32 矩；多步时 reset 只清每轮第一步，stateless 每步都清。两者都按训练轮计，而不是按权重推送计：`train_async.py` 在 `--update-weights-interval > 1` 时多轮才推送一次（归 [[16_slime_weight_sync_analysis|权重同步]]），这时"每次推送后重置"与"每轮重置"不再相同。critic 与 actor 共用 `train`，同样受 `--reset-optimizer-states` 影响（YAML 可按角色覆盖）。"每次把权重推送到推理端后重置优化器状态"这一做法的背景见 [[12_training_dynamics_stability_analysis|训练动力学稳定性]]；该页转述的是 GLM-5 的配方，它与本页开源基线的对应关系是上面的推论，而非源码证据。

### 4.6 参数冻结：两份互斥名单与 `--freeze-indexer`

`get_model_provider_func` 把 `_get_model_provider_func` 包进 `wrap_model_provider_with_freeze`：每个 model chunk（每个 PP/VPP 段）构造完立刻调 `freeze_model_params`，所以冻结发生在 DDP 与 optimizer 构造之前（Megatron 怎样跳过 `requires_grad=False` 的参数属依赖侧契约）。三种输入：`--only-train-params-name-list` 先把全部参数置为不可训练，再把名字被任一正则 `re.search` 命中的置回可训练；`--freeze-params-name-list` 只把命中的置为不可训练；二者同时给出时 `slime_validate_args` 抛 `ValueError`。`--freeze-indexer` 按结构识别 DSA indexer：`self_attention` 下一级名字属于 `wq_b`、`wk`、`k_norm`、`weights_proj`、`index_kpool_compress_ape`、`index_kpool_compress_gate`（GLM 插件的布局），或 `self_attention` 之后的路径中间含 `indexer`（Megatron 上游 DSA 的 `self_attention.core_attention.indexer.*`）。某个 chunk 含 `self_attention` 参数却一个 indexer 参数都没识别到时抛 `RuntimeError`；不含 attention 参数的 PP 段允许为空，识别结果记在 `model._slime_frozen_indexer_param_names`。它还把 `config.freeze_indexer` 写进 `TransformerConfig`，兼容旧 GLM Megatron 分支读这个字段的约定。`tests/test_model_provider_freeze.py` 覆盖两种 indexer 命名只冻结 indexer、以及未识别时报错。

按名冻结与结构识别的差别在匹配粒度：help 给的正则例子（如 `self_attention.wq_b`）不带锚点，`re.search` 会命中任何含该子串的参数名，而 `--freeze-indexer` 不会误冻同名的非 indexer 投影。独立 Megatron teacher server 把 `only_train_params_name_list` 设成 `["nothing_to_train"]` 冻结全部参数，并在校验时要求保持这个值（归 [[20_slime_on_policy_distillation_analysis|在线策略蒸馏]]）。GLM-5 DSA 训练本身目前没有专页，`--freeze-indexer` 与 `--allgather-cp` 是本页记录的两处接点。

---

## 5. 约束、适用场景与趋势

### 5.1 硬约束与失败边界

| 前提 | 源码边界 | 破坏后的行为 |
|---|---|---|
| 逻辑 rollout 数 ≥ `global_batch_size` | `slime/utils/dp_schedule.py::build_dp_schedule` | `AssertionError`（"need at least one rollout per step"） |
| 每步样本数 ≥ `dp_size` | 同上 | `AssertionError` |
| 动态路径拆到全单样本后仍能凑够 `target_K` | 同上（`expand_bins_by_splitting` 之后） | `AssertionError` |
| 静态路径 K 是 `dp_size × mb_group` 的倍数 | 同上 | `AssertionError`，提示调 step size / micro batch / DP·VPP |
| 动态路径需 `max_tokens_per_gpu` | `build_dp_schedule` 与 `slime/utils/arguments.py::slime_validate_args` | `AssertionError` |
| `balance_by_flops` 需 `use_dynamic_batch_size` | `slime/utils/arguments.py::slime_validate_args` | `AssertionError` |
| `--allgather-cp` 与 CP>1 同开时模型是 DSA 架构 | `slime/backends/megatron_utils/arguments.py::_validate_allgather_cp_supported` | 解析期 `ValueError` |
| `--rollout-temperature > 0` | `slime/utils/arguments.py::slime_validate_args` | 解析期 `ValueError` |
| `--save-debug-train-data` ≠ `--save-debug-rollout-data` | 同上 | 解析期 `ValueError` |
| 两份冻结名单至多给一份 | 同上 | `ValueError` |
| `--freeze-indexer` 时含 attention 的 chunk 能识别出 indexer 参数 | `slime/backends/megatron_utils/model_provider.py::freeze_model_params` | `RuntimeError` |
| `num_microbatches` 与 `global_batch_sizes` 等长 | `slime/backends/megatron_utils/model.py::train` | `AssertionError` |
| `tokens` 在 batch keys 中；mask 与 tokens 同形 | `slime/backends/megatron_utils/data.py::get_batch` | `AssertionError` |
| `optimizer.step` 成功 | `slime/backends/megatron_utils/model.py::train_one_step` | `AssertionError` |
| `overlap_grad_reduce` 时 `no_sync_func` 为空 | `slime/backends/megatron_utils/model.py::train` | `AssertionError` |
| combined 1F1B（`return_schedule_plan`）不与 MTP 训练同开 | `slime/backends/megatron_utils/model.py::train_one_step` 的 `forward_step` | `AssertionError` |
| `load` 目录存在且非空 | `slime/backends/megatron_utils/checkpoint.py::load_checkpoint` | `AssertionError` |
| `load` 或 `pretrained_checkpoint` 之一存在；不用 `moe_use_upcycling` | `slime/backends/megatron_utils/model.py::setup_model_and_optimizer` | `AssertionError` |
| stateless Adam 需 `optimizer == adam` 且 `no_save_optim` | 同上 | `AssertionError` |
| 切换的 tag 已备份 | `slime/backends/megatron_utils/actor.py::MegatronTrainRayActor._switch_model`；`slime/utils/tensor_backper.py::TensorBackuper.restore` | `ValueError`；参数名缺失时 `AssertionError` |
| `sleep`/`wake_up` 需 `offload_train` | `slime/backends/megatron_utils/actor.py::MegatronTrainRayActor.sleep` / `wake_up` | `AssertionError` |
| HF 保存目录 ≠ `hf_checkpoint`，且后者为本地目录 | `slime/backends/megatron_utils/hf_checkpoint_saver.py::save_hf_model_to_path` | `ValueError` |
| `rollout_top_p != 1.0` 时 batch 带 top-p 记录 | `slime/backends/megatron_utils/loss.py::get_rollout_top_p_logprob_kwargs` | `ValueError` |
| R3 时 rollout dict 带 `rollout_routed_experts`，记录数等于 replay 对象数 | `slime/backends/megatron_utils/actor.py::MegatronTrainRayActor.fill_routing_replay` | `ValueError`；`AssertionError` |
| 支持的 `advantage_estimator` | CLI `choices` 先拒绝；绕过解析改写 args 时才到 `slime/backends/megatron_utils/loss.py::compute_advantages_and_returns` | argparse 报错；`NotImplementedError` |
| PPO critic 不在 `kl_coef ≠ 0` 时同开 `--use-rollout-logprobs`（仅无 `--megatron-config-path` 时可能出现） | `slime/backends/megatron_utils/loss.py::compute_advantages_and_returns` 在 critic 进程内 | 无守卫，`compute_approx_kl(..., ref_log_probs=None)` 下标 `None` 抛 `TypeError`（源码推断，未运行） |
| PP last stage 至少有 `log_probs`、`rollout_log_probs`、`values` 之一 | `slime/backends/megatron_utils/loss.py::compute_advantages_and_returns` | 迭代 `None` 抛 `TypeError` |
| ppo 且 CP>1 时每条样本在 cp0 上有 response 位置 | 同上的 ppo 分支 `token_level_rewards[-1] += reward` | 无守卫，`IndexError`；非空但不含末 token 时 reward 静默错位（源码推导，归 [[15_slime_loss_parallelism_analysis|Loss 与并行归一化]]） |
| numpy 1.x | `slime/backends/megatron_utils/initialize.py::init` | `AssertionError` |
| PPO 下 `offload_train`、critic GPU 数等于 actor | `slime/utils/arguments.py::slime_validate_args` | 无守卫，强制改写 |
| actor 与 critic 相同 Megatron 拓扑 | 官方 `megatron-config.md` 声明 | 无守卫，文档标"当前不支持，可能在初始化或训练时失败" |
| `load_other_checkpoint` 加载失败 | `slime/backends/megatron_utils/actor.py::MegatronTrainRayActor.load_other_checkpoint`（恢复不在 `finally`） | 无守卫，args 停留在临时值，此路径中止 |
| `--load` 指向没有 tracker 文件、也没有 `config.json` 的 `iter_XXXXXXX` 目录 | `slime_validate_args` 只认 tracker 文件，而 `slime/backends/megatron_utils/checkpoint.py::_is_megatron_checkpoint` 也认 `iter_` 目录名 | 无守卫，`load` 被静默替换为 `ref_load` 并置 finetune、不载 optimizer（两处判定不一致，源码路径推断） |
| `--save-hf` 模板含具名占位 `{rollout_id}` | `slime/backends/megatron_utils/actor.py::MegatronTrainRayActor.save_model` | `{}` 抛 `IndexError`；不带占位时每次保存覆盖同一目录，无守卫 |
| 动态批次的 forward_only 结果顺序 | `build_dp_schedule` 让每 rank 的 `micro_batch_indices` 展平恰为 `range(n)`，`tests/test_dp_schedule.py::assert_invariants` 断言 | 冻结基线下 `forward_only` 的还原是恒等映射；换成非连续下标的调度时，这段还原才真正起作用 |
| 关闭 advantage 计算时 GPU 上是 actor 权重 | `MegatronTrainRayActor.init` 只在 `offload_train` 时于末尾 `_switch_model("actor")`；`MegatronTrainRayActor.train_actor` 只在 `compute_advantages_and_returns` 分支内切回 actor | 无守卫：`--disable-compute-advantages-and-returns`、未开 `offload_train`、且初始化最后装载的是 ref 或 teacher（未开 `keep_old_actor`；开了则最后装载的 `old_actor` 与 actor 同为 `args.load`）时，仅首个成功 optimizer step 受影响：它的前向与反向在那份权重上进行。bf16 参数的 FP32 主副本仍是 actor 的，Megatron 在 step 后把主副本拷回模型参数（`MixedPrecisionOptimizer.step_with_ready_grads` 与 `DistributedOptimizer._copy_main_params_to_model_params`，已在 `1dcf0daf` 核对），此后各步与各轮都在"actor 主参数 + 首步在 ref/teacher 权重上算出的更新"上继续，`backup("actor")` 与 weight updater 发布的也是它；以 fp32 存放的参数在 distributed optimizer 里主副本就是模型参数的视图（`shard_fp32_groups`），装载 ref/teacher 时已被一并覆盖，不会被拉回 actor。只有 actor 的 `load` 与 ref/teacher 的 checkpoint 不同时才有实际影响（slime 侧为源码路径推断，整条路径未运行验证） |

### 5.2 常见误读

| 误读 | 固定基线的实际行为 |
|---|---|
| ref/teacher 是独立 GPU trainer | 它们是 actor 模型的 CPU tag，切换即整份拷回；只有 critic 是独立可训练 group |
| 在线 rollout 进入 Megatron Dataset / DataLoader | `RolloutManager` 算好 step、DP、micro-batch 计划，trainer 用 `DataIterator` 回放；Megatron 从流水线调度接管 |
| 角色切换只是改一个字符串 | `_switch_model` 真实 restore 参数并同步；每轮训练后再 backup actor，有带宽与 host memory 代价 |
| 动态 batching 的 token cap 永远是硬上限 | `balance_by_flops` 分支明确不保证 cap；单条超 cap 样本独占超 cap mb |
| 静态路径会自动拆块凑对齐 | 静态路径不拆，K 不对齐直接断言；最后一块可以短于 `micro_batch_size` |
| 每张训练卡各取一份不同样本 | 只有 DP rank 分不同 partition；同一 DP 副本内的 TP/PP/CP/EP rank 共享样本身份，CP 在 `get_batch` 内切同一序列 |
| 每轮都多做一次 old-policy 前向 | 单步、`kl_coef = 0`（`use_kl_loss` 不影响）、无 critic/old_actor/OPD/GSPO/mismatch 且不用 rollout logprob 时复用训练前向的 detached logprob |
| 默认 grpo 下 `--kl-coef` 是对 ref 的 KL 正则 | grpo/gspo/cispo 不读 KL 数值；`kl_coef` 只对 ppo 与两种 REINFORCE++ 整形 reward，grpo 下需要 KL 约束应走 `--use-kl-loss` |
| `--allgather-cp` 是通用的 CP 布局选项 | CP>1 时只允许 `DeepseekV32ForCausalLM`、`GlmMoeDsaForCausalLM`，其他架构解析期报错 |
| actor/critic YAML 可自由选不同拓扑 | 官方文档标为当前不支持；资源键在 YAML 里被忽略 |
| PPO 的 actor 用的是 critic 更新后的 value | actor 拿到的是 critic 训练前 `forward_only` 得到的 V_old；critic 的 returns 也不含 KL 整形 |
| advantage 放哪里只是代码风格 | 默认在信号齐备、参数未更新、PP last stage 计算；改位置须重证统计域与版本边界 |
| `old_actor` 就是当前 GPU 模型 | 它是 `keep_old_actor` 下按 `update_weights_interval` 轮转的训练侧对照快照 |
| `sleep` 只是清显存 | 它销毁 process group 并暂停 memory saver 管理的全部训练状态；与角色 backup 是两件事 |
| `train_iters` 精确等于总步数 | 它是按 `num_rollout × rollout_batch_size × n // global_batch_size` 的估算，只用于 LR decay 长度；compact 扇出不改变它，动态采样与过滤才会让实际步数偏离 |
| `--reset-optimizer-states` 与 stateless Adam 是同一件事 | 只在每轮单步时更新相同；前者保留矩张量并按轮清零，后者每步从零矩起算且不分配矩（§4.5） |

### 5.3 何时使用与检查清单

| 场景 | 建议 | 原因 |
|---|---|---|
| 变长 response 的 GRPO | `--use-dynamic-batch-size --max-tokens-per-gpu`，CP 时把 cap 设为 `max_response_len // cp_size` 量级 | first-fit 逼近 cap；对齐靠拆 bin，不需要 step size 整除 |
| 长度差异极大、注意力主导 | 加 `--balance-by-flops` | 用 `calculate_fwd_flops` 估算均衡 mb，代价是可能超 cap |
| rank 间明显不均衡 | `--balance-data` | KK 按估算 FLOPs 配对 mb，只改一步之内的 rank 分配；help 里"拆到不同训练步"的提醒与冻结基线不符 |
| 需要 KL 正则 | 默认 grpo 下用 `--use-kl-loss` / `--kl-loss-coef`；`--kl-coef` 只对 ppo、reinforce_plus_plus、reinforce_plus_plus_baseline 整形 reward | ref 以 tag 形式存在，只多前向不多显存；grpo 下设 `--kl-coef` 不改梯度，却会强制装载 ref（没开 `use_kl_loss` 时也要 `--ref-load`）、多一次 ref 前向、关掉复用而多一次 old-policy 前向、与 `kl_loss_coef` 互斥，并让 `rollout/kl` 出现非零值（§2.1、[[15_slime_loss_parallelism_analysis#2.2 从最小实例到整套 loss 层|KL 的两个入口]]） |
| 需要 OPD | `--use-opd --opd-type megatron` / `sglang` | Megatron teacher 须与 actor 同架构，异构 teacher 走 `--opd-type sglang` |
| PPO | `--advantage-estimator ppo`，接受强制 `offload_train`；CP>1 时确认短样本不会在 cp0 上没有 response 位置 | critic 独立 group 且与 actor 分时用卡；cp0 空片会 `IndexError` |
| DSA 模型的长序列 CP | `--allgather-cp`，必要时 `--freeze-indexer` | allgather 布局只对 DSA 架构开放；冻结 indexer 按结构识别 |
| 每轮不带旧动量 | 单步时 `--use-stateless-adam --no-save-optim`，多步或需要保留矩时 `--reset-optimizer-states` | 前者省两份矩但不能从 checkpoint 恢复动量；后者轮内仍累积 |

改动训练后端前逐项核对：`global_batch_size` 是否整除逻辑 rollout 数且每步能给每个 DP rank 至少一条样本；静态或动态 micro-batch 计划是否满足 `dp_size × mb_group` 对齐；同 rollout 的片段是否仍在同一 step；`max_tokens_per_gpu × cp_size` 是否考虑单条超长样本独占 mb；新增角色是否真的需要 optimizer（否则应是 tag）；新钩子改 loss mask 后 `rollout_mask_sums` 是否仍与统计口径一致（归 [[15_slime_loss_parallelism_analysis|Loss 与并行归一化]]）；改变 `step_global_batch_size` 语义时 loss 缩放、指标分母与 LR increment 是否同步；`offload_train` 下新增的通信组是否能被销毁重建；HF 保存目录是否与 `hf_checkpoint` 不同且模板带 `{rollout_id}`。

### 5.4 当前演进方向

| 位置 | 注释原文 | 指向什么 |
|---|---|---|
| `slime/backends/megatron_utils/actor.py::MegatronTrainRayActor._get_rollout_data` | `# Fetch data through ray on CPU, not sure if this will be performance bottleneck.`；`# TODO: this is ugly, move to somewhere else?` | 取数（Ray/CPU）与设备搬运写在同一函数；NIXL 传输是替换点（归 [[12_slime_sample_datasource_analysis|Sample 与 DataSource]]） |
| `slime/backends/megatron_utils/model.py::forward_only` 动态批次分支 | `# TODO: This is ugly... Find a better way to make the data have the same order.`；`# TODO: move this out of the loop.` | 冻结基线下这段还原是恒等映射；TODO 指向的是把重排移出循环或删掉（本页推断），不是现存的顺序错误 |
| `slime/backends/megatron_utils/checkpoint.py` 模块顶部 | `# TODO: may need to copy those 2 functions and do refactoring.`；`# TODO: find a less hacky way to do this.` | 对 Megatron 加载/保存与 `ShardedTensor` 校验的猴补被标为临时 |
| `slime/backends/megatron_utils/initialize.py::is_megatron_main_rank` | `# TODO shall we use a simpler method to determine which rank to init wandb?` | 主 rank 判定（DP-with-CP 0、TP 0、PP last）可能改变日志归属 |
| `slime/backends/megatron_utils/data.py::get_batch` | `# use 0 as the pad token id should be fine?` | pad token 固定为 0 尚未被确认 |
| `slime/ray/placement_group.py::create_training_models` | `# TODO how to decide rollout start id when critic is involved?` | 有 critic 时续训起点取 critic 的 iteration，尚无定论 |

> [!note] 推断
> 六处标记方向一致：**边界不动，落点收窄**。待搬的是搬运落点、重排位置、猴补与续训起点，而不是"RolloutManager 算计划、trainer 回放、Megatron 执行"这条分工。若 forward_only 的重排被移出循环，§3.2.1 调用树里 `DataIterator` 与结果收集的接口会变，四条不变量不变。源码只写了"ugly""hacky""not sure"，没有给出替代方案或时间；这层归纳由本页承担，不代表项目路线图。

---

## 6. 配置契约

slime 域尚无配置 coverage ledger（是否建立待规划）；下表只列本页训练路径直接读取的 slime 参数，按用途分组，默认值取自 `slime/utils/arguments.py`；Megatron 原生参数（TP/PP/CP/EP、优化器、lr schedule 等）沿用 Megatron 定义，只列被 slime 重置默认值或强制改写的项。其余参数与脚本的对应归 [[02_slime_quickstart_and_configuration_guide|配置指南]]。Megatron 自身怎样从 dataclass 生成 CLI、校验与实例化配置，见 [[41_megatron_config_surface_analysis|Megatron 配置面]]（该页按 `85902ef5` 分析；slime 镜像钉的是更早的 `1dcf0daf`，且 slime 同时兼容两种 Megatron 布局，例如 `_vocab_size_with_padding` 的两个导入路径与缺 `enable_gloo_process_groups` 时的默认值）。

### slime 强制的 Megatron 默认值

`megatron_parse_args` 在 Megatron 解析之后先比对 HF config（`_hf_validate_args`：hidden size、头数、层数、FFN、MoE FFN、共享专家、tie embedding、norm eps、rope theta 不一致即 `AssertionError`）、做 allgather CP 架构校验，再由 `_set_default_megatron_args` 改写：

| 字段 | slime 设定 | 说明 |
|---|---|---|
| `use_distributed_optimizer` | 恒为 True | Megatron `1dcf0daf` 中 `--use-distributed-optimizer` 是 store_true 开关，slime 无条件打开 |
| `enable_gloo_process_groups` | 缺失时设 True | 兼容没有该参数的 Megatron 布局 |
| `bf16` | `not fp16` | 不开 fp16 即 bf16 |
| `use_persistent_ckpt_worker` / `ckpt_assume_constant_structure` / `ckpt_fully_parallel_load` | True | 注释称不改 checkpoint 内容，只减少重复校验并并行加载 |
| `dist_ckpt_save_pre_mcore_014` | True | 带 TODO：等上游修好 optimizer-cpu-offload 的保存 bug 再改 |
| `seq_length` / `max_position_embeddings` | 4096 / 取 `seq_length`（仅当未设） | 占位；显式给出的 `max_position_embeddings` 保留，Megatron 也用它作 YaRN 原始长度 |
| `rope_type` | 未设时 MLA 取 `yarn`、否则 `rope` | 兼容项 |
| `padded_vocab_size` / `tokenizer_model` / `tokenizer_type` | 由 `vocab_size` 推出 / 缺省取 `hf_checkpoint` / `HuggingFaceTokenizer` | tokenizer 默认跟 HF checkpoint |
| `variable_seq_lengths` | 恒为 True（`validate_args`） | 总是变长 |
| `moe_token_dispatcher_type` | `allgather` 改写为 `alltoall`（`validate_args`） | allgather 分发器不支持变长序列 |
| `decoder_first/last_pipeline_num_layers` | PP=1 时必须为 None（`validate_args`） | 否则 `AssertionError` |

本组 11 行覆盖 `_set_default_megatron_args` 与 `validate_args` 改写的全部 15 个字段；`megatron_validate_args` 在 `--debug-rollout-only` 下不执行。

### 批次与调度

| 参数 | 默认 | 契约 |
|---|---|---|
| `--global-batch-size` / `--num-steps-per-rollout` | None / None | 单位是逻辑 rollout；后者给出时前者换算为 `rollout_batch_size × n // steps` 并断言一致 |
| `--micro-batch-size` | 1 | 仅静态路径使用；开动态时忽略 |
| `--use-dynamic-batch-size` / `--max-tokens-per-gpu` | False / None | first-fit 的 cap 为 `max_tokens_per_gpu × cp_size`；开动态必须给 cap |
| `--log-probs-max-tokens-per-gpu` | None → 同 `max_tokens_per_gpu` | 冻结基线只在参数校验里补默认值，Megatron 路径没有读者；ref/teacher/old 前向复用训练的 `micro_batch_indices`，打包仍按 `max_tokens_per_gpu` |
| `--balance-data` | False | KK 按估算 FLOPs 把 mb 分给 DP rank；`balance_by_flops` 会强制打开；help 称可能跨训练步，与实现不符 |
| `--balance-by-flops` | False | 用 FLOPs 估算做 mb 打包，不保证 cap；要求动态批次 |
| `--data-pad-size-multiplier` | 128 | THD 流补到 `tp × 此值` 的倍数 |
| `--allgather-cp` | False | CP 布局改为全局拼接后连续等分（DSA 模式）；CP>1 时只允许 `DeepseekV32ForCausalLM`、`GlmMoeDsaForCausalLM`，否则解析期 `ValueError`；argparse 无 help 文本 |

本组 8 行覆盖 10 个 flag。

### 角色与 checkpoint

| 参数 | 默认 | 契约 |
|---|---|---|
| `--ref-load` / `--ref-ckpt-step` | None / None | `kl_coef ≠ 0` 或 `use_kl_loss` 时必须存在；装入 `ref` tag |
| `--ref-update-interval` | None | 每隔多少轮把 actor 备份为 ref；None 不更新 |
| `--keep-old-actor` / `--update-weights-interval` | False / 1 | 装入 `old_actor`；interval 为 1 时额外维护 `rollout_actor` 队列 |
| `--use-opd --opd-type megatron --opd-teacher-load` / `--opd-teacher-ckpt-step` | False / None | teacher 装入同一模型的 `teacher` tag，可单独指定 ckpt step（归 [[20_slime_on_policy_distillation_analysis|在线策略蒸馏]]） |
| `--advantage-estimator ppo` | `grpo` | 派生 `use_critic=True`，critic 节点与 GPU 数等于 actor，强制 `offload_train`；`num_rollout == 0` 时不建 critic |
| `--num-critic-only-steps` | 0 | 前若干轮只训练 critic |
| `--megatron-config-path` | None | 角色 YAML 覆盖；资源键被忽略；critic 强制项只在此路径生效 |
| `--load` / `--save` / `--save-interval` / `--async-save` / `--no-save-optim` | None / None / None / False / False（slime 以 `reset_arg` 重置或新增） | `load` 没有 `latest_checkpointed_iteration.txt` 时置 `no_load_optim`、`no_load_rng`、`finetune`，给了 `ref_ckpt_step` 就写入 `ckpt_step`，`start_rollout_id` 缺省为 0；也不是可加载的 HF 目录时改用 `ref_load`（只有 `iter_` 目录名、没有 tracker 的 Megatron checkpoint 也会被替换，§5.1）；`save_interval` 要求 `save` |
| `--save-hf` | None | 以 `save_hf.format(rollout_id=rollout_id)` 展开，模板须含 `{rollout_id}`；仅 actor，不含 optimizer |
| `--use-stateless-adam` / `--reset-optimizer-states` | False / False | 前者替换 Adam 构造、每步从零矩起算并要求 `no_save_optim`；后者每轮训练开头清零 `step` 与两份矩（§4.5） |
| `--offload-train` / `--colocate` / `--offload` | None / False / False | colocate 缺省开 `offload_train`（`release_train` 除外）；开后置 `disable_grad_buffers_cpu_backup` 与 `disable_param_buffers_cpu_backup`，critic 的后者被改回假 |
| `--distributed-timeout-minutes` | 10 | process group 超时；重建默认组时沿用 |

本组 12 行覆盖 24 个 flag。

### 前向、advantage 与训练步

| 参数 | 默认 | 契约 |
|---|---|---|
| `--disable-compute-advantages-and-returns` | 计算 | 关闭后跳过 ref/teacher/old 前向与 advantage，供 SFT 或自定义 loss |
| `--use-rollout-logprobs` | False | 用 rollout logprob 作 old logprob；与 `use_tis` 互斥 |
| `--get-mismatch-metrics` | False | 需 `custom_tis_function_path`；即便用 rollout logprob 也多一次前向 |
| `--use-rollout-entropy` | False | ref/teacher/old 前向顺带返回 `ref_entropy`/`teacher_entropy`/`entropy`，供 postprocess 钩子做 loss mask；不影响训练前向的 entropy |
| `--log-probs-chunk-size` | -1 | logprob 计算分块大小 |
| `--rollout-temperature` / `--rollout-top-p` | 1.0 / 1.0 | 前者须 > 0（解析期校验），在 logprob 前缩放 logits，value 不缩放；后者非 1 时 top-p 记录成为必备字段 |
| `--use-routing-replay` / `--use-rollout-routing-replay` | False / False | 控制 `ROUTING_REPLAY_STAGE` 与 `fill_routing_replay`；后者会打开前者 |
| `--calculate-per-token-loss` | False | 改变 loss 缩放与指标分母（归 [[15_slime_loss_parallelism_analysis|Loss 与并行归一化]]） |
| `--lr` / `--clip-grad` / `--seed` | 1e-6 / 1.0 / 1234 | slime 重置的 Megatron 默认值 |
| `--enable-mtp-training` | False | 要求 `mtp_num_layers`；训练前向传 `mtp_kwargs`（`megatron.patch` 引入）；`1/num_microbatches` 只缩放记录的 `train/mtp_*_loss`，反向缩放由 Megatron `MTPLossAutoScaler` 完成，非 per-token 时为 `loss_scale / num_microbatches`（机制见 [[21_slime_speculative_decoding_mtp_analysis#2.2 从最小实例到整个闭环|Megatron 补丁里的 MTP 机制]]） |

本组 10 行覆盖 14 个 flag；KL 相关的 `--kl-coef`、`--use-kl-loss`、`--kl-loss-coef` 归 [[15_slime_loss_parallelism_analysis|Loss 与并行归一化]]。

### 参数冻结

| 参数 | 默认 | 契约 |
|---|---|---|
| `--only-train-params-name-list` | None | 正则列表；未命中者全部冻结 |
| `--freeze-params-name-list` | None | 正则列表；命中者冻结；与上一项互斥（`ValueError`） |
| `--freeze-indexer` | False | 按结构冻结 DSA indexer；含 attention 却识别不到 indexer 时 `RuntimeError`（§4.6） |

本组 3 行覆盖 3 个 flag。

### 钩子与调试

| 参数 | 默认 | 契约 |
|---|---|---|
| `--custom-megatron-init-path` | None | `slime/backends/megatron_utils/initialize.py::init` 末尾、模型构造之前调用，签名 `(args)` |
| `--custom-megatron-before-log-prob-hook-path` | None | 每次 `forward_only` 前调用，签名 `(args, model, store_prefix)` |
| `--custom-megatron-before-train-step-hook-path` | None | 每步清梯度后调用，签名 `(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)` |
| `--rollout-data-postprocess-path` | None | advantage 之后调用，可改 loss mask |
| `--custom-advantage-function-path` | None | 替换内置估计器，原地写 `advantages` / `returns` |
| `--custom-model-provider-path` | None | 替换模型构造；critic 仍换单输出 head，冻结包装照常生效 |
| `--save-debug-train-data` | None | 触发 logprob 捕获与训练侧转储；不得与 `--save-debug-rollout-data` 相同 |
| `--manual-gc` / `--manual-gc-interval` | Megatron 默认 | `train` 首次调用时 `gc.disable()` 且此后不再打开，回收只在显式 `gc.collect()` 处；`manual_gc_interval` 只被断言非负 |

本组 8 行覆盖 9 个 flag。

## Related Pages

- [[12_slime_sample_datasource_analysis]] — 本页输入的训练字典如何从 Sample 压缩而来，以及 `rollout_mask_sums` 为何在切分前算好。
- [[15_slime_loss_parallelism_analysis]] — loss 回调、KL 两个入口、归约器与 `step_global_batch_size` 如何在 DP/CP/micro-batch 切分下保持目标函数不变。
- [[16_slime_weight_sync_analysis]] — optimizer step 之后新参数如何发布给 rollout 引擎，以及 colocate 下 sleep/wake 与权重传输的关系。
- [[17_slime_train_inference_consistency_analysis]] — ref/old/current logprob、top-p keep-mask 与路由重放为何影响训练侧前向。
- [[20_slime_on_policy_distillation_analysis]] — teacher 以 tag 形式进入同一模型时的信号流与版本边界。
- [[23_slime_model_architecture_extension_analysis]] — custom model provider 与 HF↔Megatron 权重映射的扩展边界。
- [[16_megatron_distributed_optimizer_analysis]] — slime 恒开的 distributed optimizer 在 Megatron 侧怎样分片状态、规约梯度并完成 `optimizer.step`（按更新的 `85902ef5` 分析）。
