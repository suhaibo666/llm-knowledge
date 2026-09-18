# C 组：14 Megatron 训练后端、15 Loss 与并行归约

评审者：独立评审（未参与写作），只读。基线 `THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`；Megatron 侧按 slime `docker/Dockerfile` 钉的 `MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087` 用 `git show` 核对。

## A. 判定表

| page | profile(fit?) | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check (n/3 + anchors) | feature-check | host | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 14 | feature（fit） | pass（§2.2.8 引注误读，见 B4） | pass（§3.2.1 走到 optimizer.step→scheduler→backup） | pass | transform, layout, timing | pass（复跑冻结 build_dp_schedule/KK 共 16 组系数、zigzag/allgather、前向计数、scheduler；生成器逐行对得上；PPO critic 面未回放，见 B9） | 2/3：`slime/utils/dp_schedule.py::build_dp_schedule` ✓、`slime/backends/megatron_utils/actor.py::MegatronTrainRayActor.train_actor`（can_reuse）✓、`slime/backends/megatron_utils/model.py::get_optimizer_param_scheduler` 注释 ✗ | FAIL §2.3/§5.1（allgather 边界） | minor | REJECT |
| 15 | feature＋数值/估计量（fit） | pass | pass（loss 回调、advantage、报告三棵树） | pass | transform, layout | pass（三种均值、reward 回落、[1,9,0.2,1.8]→3、13.833→3.458、per-token 42→2.381、rejection 3、GSPO 0.2 对 0.35/0.14、PPO 落点逐项复算；与 Megatron@1dcf0daf 一致） | 3/3：`cp_utils.py::get_sum_of_sample_mean` ✓、`loss.py::loss_function` ✓、`tests/test_loss_cp_invariance.py::test_backward_grad_is_cp_invariant` ✓；加抽 `slime/utils/ppo_utils.py::get_grpo_returns` ✗ | FAIL §2.3（KL 轴） | minor | REJECT |

事实抽查：14 页约 20 项，错 4 处；15 页 16 项，错 2 处。其余与冻结源码一致，包括 argparse 默认值、断言、测试常数 1249.875、8.0、78/12。

## B. 问题（按严重度）

1. [major] 15 §2.3「KL 位置」行、§5.2、§6 `--kl-coef`、§2.1.2 括注、图面板 2；14 §2.1「reward 不做 KL 整形」同属此问题。页面把 `kl_coef` 写成对所有估计器都生效的 reward 整形；实际 grpo、gspo、cispo 的 returns 与 KL 无关。默认 grpo 下 `kl_coef≠0` 只会装载 ref 并多做一次前向、关掉 logprob 复用、与 `kl_loss_coef` 互斥、记录 `rollout/kl`，不改变梯度。证据：`slime/utils/ppo_utils.py::get_grpo_returns`（`ones_like(kl)*reward`），只有 `loss.py::compute_advantages_and_returns` 的 ppo、reinforce_plus_plus、reinforce_plus_plus_baseline 三个分支用 `kl_coef`；4c193f1f 仍相同。修正：KL 轴按估计器拆开，同步改各处。
2. [major] 14 §2.3「CP 布局」行、§2.2.4、§5.1、§6 `--allgather-cp`：allgather 变体缺硬边界。CP>1 时只允许 `DeepseekV32ForCausalLM` 与 `GlmMoeDsaForCausalLM`，其他架构参数解析时抛 ValueError（报错原文称会静默打乱 token 顺序）。页面把它当通用兄弟布局。证据：`slime/backends/megatron_utils/arguments.py::_validate_allgather_cp_supported`；`tests/test_megatron_argument_validation.py::test_allgather_cp_rejects_non_dsa_cp_models`。修正：§5.1 加一行，§6 与 §2.3 注明适用架构。
3. [minor] 14 §1.3「进度计数」、§2.2.1：页面说"扇出下 train_iters 只是估算"，并把"扇出"算作源码注释原话。注释列的是 dynamic sampling、filtering、custom step splitter；docstring 写明每轮步数与每个 rollout 产出几条样本无关，也与本页 §2.1.1 矛盾。证据：`model.py::get_optimizer_param_scheduler`、`slime/ray/rollout.py::RolloutManager._split_train_data_by_dp`。
4. [minor] 14 §2.2.8：注释被误读。销毁 WORLD 的理由是纯 PP=4 下逐个销毁子组会出现拆除顺序死锁；"释放通信器显存"只是补充。被否定的方案是逐子组销毁，而 env=0 仍走这条路径，所以 env=0 的代价漏了死锁风险。证据：`slime/utils/reloadable_process_group.py::_destroy_default_nccl_process_group` 与 `destroy_process_groups`。
5. [minor] 14 §2.2.6："只在 PP last stage 有 log_probs 或 values 时工作"取自 docstring。代码只判断 `is_pipeline_last_stage()`；零 KL 的形状取自 `log_probs or rollout_log_probs or values`，三者都缺时抛 TypeError。证据：`loss.py::compute_advantages_and_returns`。
6. [minor] 15 §2.2.5、§2.4：页面称 Megatron"未钉版本"。slime `docker/Dockerfile` 钉了 `MEGATRON_COMMIT=1dcf0daf…`；在该提交核对 `schedules.py::forward_step_calc_loss`、`distributed_data_parallel.py`（`gradient_scaling_factor`）、`finalize_model_grads.py::finalize_model_grads` 都与页面一致；`docker/patch/latest/megatron.patch` 没改这三处。修正：补 Megatron 基线行与补丁说明。
7. [minor] 15 §1.1、§2.3：兄弟轴未点名。MTP loss（`MTPLossAutoScaler` ÷M）与 MoE aux loss（×cp/M）由 Megatron 自己缩放，不经 slime 归约器，"拓扑不变量"覆盖不到；`scripts/run-glm4.7-30B-A3B.sh` 开了 `--enable-mtp-training`。修正：§2.3 点名并链到 21。
8. [minor] 15 §2.2.2、§2.2.7：OPSM 的 mask 按 token `advantage<0` 判定，GAE 下并非整条序列置零；`opsm_clipfrac` 分子没乘 loss_mask，含工具 token 时单条序列就会超过 1，不限于扇出。证据：`ppo_utils.py::compute_opsm_mask`。
9. [minor] 14 §2.3「角色集合」：PPO critic 数据面只有散文，没有回放；未说明回传给 actor 的是训练前 `forward_only` 算出的 values（V_old），也未说明 critic 会按自身 args 另算一次 advantage。证据：`actor.py::MegatronTrainRayActor.train_critic`。
10. [minor] 两页 header 与 §6："主题"写成机制摘要并预告结论（如 15 页"逐项抵消""只改分子"）；§6 各分组缺"字段总数、覆盖数、余项归属"行。
11. [nit] 14：stateless Adam 的"零阶矩"用词错，未写矩清零后更新退化为 −lr·g/(|g|+eps)（`stateless_adam.py::StatelessAdam.step`）；`--save-hf` 必须带 `{rollout_id}` 占位（`actor.py::save_model`）；"a/b≈隐藏维度量级"不准（实际约 3·ffn＋V/层数，Qwen2.5-0.5B 形配置约 2.3×10⁴，a>22b 结论不变）；`slime_validate_args` 会把没有 tracker 也没有 config.json 的 `--load iter_…` 改成 `ref_load`。
12. [nit] 15：`--kl-loss-type` 也作用于 loss 侧 KL；§4.3 漏 `ci_disable_kl_checker` 门控，阈值比较是 `<=`；normalizer 实为 `torch.tensor(1)`；两页跨页链接标签写 §2.2.x，实际锚到 §2.2。

## C. 缺失内容

- TP 侧梯度分块 all-reduce 补丁（`SLIME_GRAD_COALESCE_CHUNK_BYTES`）与 DeepEP Buffer 补丁 — `megatron_utils/megatron_patch/megatron_chunked_grad_coalesce_patch.py`、`megatron_utils/__init__.py` — 归 14 — 中（全域未提；基线后 c403335d 已删除该补丁）。
- slime 强制改写的 Megatron 默认值：distributed optimizer 恒开、`variable_seq_lengths`、allgather 分发器改 alltoall、ckpt I/O 默认 — `megatron_utils/arguments.py::_set_default_megatron_args`、`validate_args` — 归 14 §6 — 中。
- 参数冻结 `--only-train-params-name-list` 与 `--freeze-params-name-list`（互斥）、`--freeze-indexer` — `model_provider.py`、`tests/test_model_provider_freeze.py` — 归 14 或 23 — 中。
- 断点续训：checkpoint iteration 即 rollout_id、起点为 loaded+1、由 critic 决定起点（TODO） — `actor.py::save_model`/`init`、`placement_group.py::create_training_models` — 归 14 或 18 — 中。
- `--use-rollout-entropy` — `model.py::forward_only` — 归 14 — 低。
- value head 不应用 rollout 温度 — `loss.py::get_values`、`tests/test_value_temperature.py` — 归 14/15 — 低。
- rollout 侧非白名单张量键（如 `kl`、`entropy`）按 mean·cp·count 报告，不在 rollout 均值空间 — `data.py::log_rollout_data` — 归 15 — 低。
- MFU/FLOPs 性能日志 — `data.py::log_perf_data` — 归 18 — 低。
- ROCm HIP 异步写补丁 — `model.py::initialize_model_and_optimizer` — 归 14 — 低。

## D. 环境缺口与无法验证项

- 无 SGLang 检出，rollout logprob 与 top-p 核集只核对 slime 侧调用面。
- 未运行 GPU 或分布式任务：PPO cp0 IndexError、allgather 零项防死锁、WORLD 销毁与重建、per-token 梯度、KK 实际耗时均为源码推导。
- Megatron 只经 `git show 1dcf0daf` 与静态读补丁核对，未构建镜像。
- `/Users/suhaibo/97-llm/slime@4c193f1f` 只用于确认 `get_grpo_returns` 在基线后未变。

## 协调者复核

- B1 属实：`slime/backends/megatron_utils/loss.py::compute_advantages_and_returns` 中 `advantage_estimator in ["grpo","gspo","cispo"]` 分支调用 `get_grpo_returns(rewards, kl)`，该函数返回 `torch.ones_like(kl[i]) * rewards[i]`；`--advantage-estimator` 默认 `grpo`。15 页第 201、380、431 行与 31 页第 98 行均把 `kl_coef` 无条件写成 reward 整形。
- B2 属实：`slime/backends/megatron_utils/arguments.py::_validate_allgather_cp_supported` 对非 DSA 模型且 CP>1 抛 ValueError，并在同文件第 197 行调用；14 页只在 §6 表写"（DSA 模式）"，未写拒绝边界。
