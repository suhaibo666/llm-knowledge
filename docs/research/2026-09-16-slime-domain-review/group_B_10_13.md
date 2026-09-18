# B 组：10 端到端迭代、11 Ray 控制面、12 Sample/DataSource、13 SGLang rollout engine

评审者：独立评审（未参与写作），只读。基线 `THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`。

## A. 判定表

| page | profile（是否合适） | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check（n/3 与锚点） | profile-check | host | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | mechanism（合适） | pass | pass | pass | timing | **FAIL §5.1–5.3** | 3/3：`train_async.py::train` 发布前等 future；`RayTrainGroup._reload_rollout_weights_from_disk` CI 比对版本抛 RuntimeError；`AsyncRolloutWorker._make_done_cb` ABORTED 回队 | completion gate **FAIL §2.4/§6**（异步 checkpoint 切口） | minor | **REJECT** |
| 11 | feature（合适） | pass | pass（§3.2.3 省略几跳） | pass | layout | pass | 3/3：`_allocate_rollout_engine_addr_and_ports_normal` 的 `get_port(30+dp)`；`test_placement_group_layout` 十组用例；`_reload_rollout_weights_from_disk` 顺序 pull→pause→flush→update→删目录→continue | feature: pass（有 minor） | minor | PASS-with-minors |
| 12 | feature（合适） | pass | pass | pass | transform, layout | pass（归一化分支未回放，minor） | 3/3：`RolloutDataSourceWithBuffer.add_samples` 组长断言；`_convert_samples_to_train_data` 补 id 并算分母；`generate_and_rm` 先清 mask 再早退 | feature: pass | minor | PASS-with-minors |
| 13 | feature（合适） | pass | pass | pass | timing, coupled-planes | **FAIL §2.3.1** | 3/3：`abort_server_until_idle`；`should_drop_dynamic_filter_output`；`add_sglang_arguments` 的 `skipped_args` 十五项 | feature **FAIL §2.3.2**（依赖侧行为当作事实写） | minor | **REJECT** |

## B. 问题清单（由重到轻）

### 页 10

- [major] §5.1–5.3 缺原理图回放。三种调度是 timing 触发，必须有原理图。现有三张 Mermaid 只画调用顺序，没有给每个 batch 标 serving 版本和 policy age，也没有 fully-async 泳道；§5.1.1 的数字例子只在正文里。按 `train_async.py::train` 推导：interval=1 时 B1 由 θ0 生成、在 θ1 训练（age 1）；interval=2 时 B2 由 θ0 生成、在 θ2 训练（age 2）。修正：用同一个 3 轮例子画 sync / async(1) / async(2) / fully-async 四条时间线，标出版本、发布栅栏和积压队列。
- [major] §2.4/§6 称"保存和恢复都以 rollout id 对齐"，对异步入口不成立。`train_async.py::train` 先 `generate.remote(rollout_id + 1)`，后 `ray.get(rollout_manager.save.remote(rollout_id))`。RolloutManager 是同步 actor（`placement_group.py::create_rollout_manager` 未设 max_concurrency），save(i) 实际在 generate(i+1) 取完数之后执行，保存的游标已跨过 i+1 那批 prompt。从 i 恢复会跳过这一批；fully-async 的队列和在途组也不进 checkpoint。§7 也没有写这条边界（分析推导）。修正：补失败边界，并链接 18。
- [minor] §2.2/§5.3 说 pause 窗口会让组"暴露为 ABORTED"。这取决于 SGLang `/pause_generation` 语义，属依赖侧，页面未标注；与 13 §2.3.2 互相矛盾，需统一。
- [minor] §8 推断"整条轨迹重来"与源码不符：ABORTED 的 Sample 重新取回后，`generate` 允许 ABORTED 状态并复用 `sample.tokens`，会续生成；也与本页 §7 和 13 不一致。
- [minor] 证据写法：正文 40 多个行号永久链接，没有集中的源码阅读路线；§2.1 引的 `rollout.py:749-814` 不包含 `_post_process_rewards`（在 722-747）。
- [minor] §2.4 称"同一个条件控制 actor/critic checkpoint"，但 actor 保存还要满足 `if actor_trains`：critic-only 预热期只存 critic 和 DataSource。
- [nit] 同时训练 critic 和 actor 时，`offload_train` 闭包只调用 actor 的 `clear_memory`；full+disk 下 `UpdateWeightFromDisk.weight_version` 也会递增；旧版 §3.6 的保存频率与恢复规则已移到 18，本页没有链接。

### 页 11

- [minor] §2.1.1 表把 debug 两行写成"不建 trainer / 无 trainer actor"，错。`create_training_models` 仍创建 trainer actor，只是 `MegatronTrainRayActor.init` 在 `if args.debug_rollout_only:` 分支里 `return 0`；参数归一化有 `actor_num_gpus_per_node = min(8, rollout_num_gpus)`。与 14 矛盾。
- [minor] §3.2.3 调用树和 §2.3 成本账本：full+disk 省略了 trainer 侧几跳（恢复、每个 rank 查一次六元组、无 engine 时早退），也省略了 `UpdateWeightFromDisk` 的三次 gloo barrier 和 post-write hook；账本里"六元组 1 次"少算。
- [minor] §2.2.2 "避免 critic 的备份设置污染 actor" 理由反了。critic 用 deepcopy；实际是 PPO 强制 `offload_train` 后把 `disable_param_buffers_cpu_backup` 置 True，critic 再把自己那份改回 False；源码没有说明原因。
- [minor] §5.1/§6 release-train 约束不全：非 megatron 后端、critic、keep_old_actor、缺 `--save` 都会 ValueError；`save_interval` 被默认成 1；disk 传输必须给 `--update-weight-disk-dir`。
- [minor] 页眉：主题提前写出机制结论；"最近更新"写的是改写过程。
- [nit] 重连分支是 if/elif，树里像两步都执行；`--num-gpus-per-node` 还决定 trainer 的 NUMA 亲和；external 布局只在表里，没进图。

### 页 12

- [minor] §3.2.1 调用树父子关系错：`_validate_rollout_id_annotated` 和展平画在 `call_rollout_fn` 下，实际是 `_get_rollout_data` 里并列调用。
- [minor] §2.2.3 "否则…save/load 直接返回" 不完全对。`RolloutDataSource.save` 只判断 `rollout_global_dataset`；global dataset 开着但没有 prompt_data 时，游标照样存盘和读回。
- [minor] reward 归一化开启分支没有用运行 ② 回放：5 个 reward 不等于 2×2，会被当成一个大组；页面断言了退化，图里只画"归一化关闭"。
- [minor] 页眉问题同 11。
- [nit] "三处注释"实为四处（`_split_train_data_by_dp` docstring 也写 "falling back to samples[i].index"）；R3 全零检查只在 topk>1 时做；`tests/test_read_file_slicing.py` 不在阅读路线里。

### 页 13

- [major] §2.3.2 称 ABORTED 回队分支"用默认 generate 时实际到不了"，只看了 `GenerateState.aborted`。默认 `generate` 也经过 `Sample._apply_meta_info`，其中 `case "abort": ... ABORTED`；`fully_async_rollout` docstring 写明在途生成会 "surfaces Sample.Status.ABORTED"。与 10 矛盾，且 pause 是否中止在途请求属依赖侧。修正：改成"服务端以 abort 结束请求时触发（依赖侧未核）"，两页统一。
- [major] §2.3.1 streaming 是独立在线数据面（CI 测试 `test_qwen3_4B_streaming_partial_rollout`），原理图里没有它的泳道，只有表格一句。修正：加 D 组逐 chunk 泳道（快照→重置→追加→abort 截断），并与非流式最终 JSON 对照。
- [minor] streaming 直接用 `client.stream("POST", ...)`，绕过 `_post` 的 60 次重试，一次 HTTP 错误就让整轮失败；账本把 60 次重试写成通用行为。
- [minor] §2.2.5 漏了 slime 改写的 router 默认值：`router_balance_abs_threshold=10, router_balance_rel_threshold=1.2`，log level=warn。
- [minor] fully-async 下 hook 收到的 `rollout_id` 取自模块全局 `_current_rollout_id`，是最近一次 generate 的值，不是这组被取出时的轮次。
- [minor] 页眉问题同 11。
- [nit] "默认过滤器"不存在（路径默认 None）；博客引用缺路径 `docs/en/blogs/introducing_slime.md`；§5.1 漏了 `validate_args` 的 dp-attention 与 pp 整除断言；§3.2.3 回队后缺 `return`。

## C. 负责范围内缺失的内容

- **PD 分离的请求与服务路径**（几页只点名，没有负责页）：`sglang_engine.py::_compute_server_args` 里 prefill 的 `follow_bootstrap_room`、decode 的 `prefill_round_robin_balance` 与跳过 hierarchical cache；`_start_router(has_pd_disaggregation)`；`docker/patch/latest/sglang.patch` 的 PD abort/retract 改动；`docs/en/advanced/pd-disaggregation.md`；`tests/test_*pd_mooncake.py`、`tests/test_qwen3_4B_external_pd.py`。建议 13 负责（端口与注册归 11），或交回 planning。高。
- **异步与 fully-async 的 checkpoint 切口**（缺）：`train_async.py::train`、`RolloutDataSource.save`、`AsyncRolloutWorker`。建议 10（或 18）。高。
- **rm_hub 各奖励函数语义**（13 只列名字）：deepscaler 在响应里没有 `</think>` 或 `###Response` 时返回 0；dapo 返回 dict `{score:±1, acc, pred}`，需配 `--reward-key score`；ifbench 运行时 git clone 并 pip install；另有 gpqa、f1、math。测试 `tests/test_rm_*.py`。建议 13。中高。
- engine 环境变量默认值（`ServerGroup.start_engines`，如 `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false`）→ 11，低。
- `rollout_num_gpus==0` 时信号量容量为 0、HTTP 客户端未初始化 → 13，低。
- `_init_ray_distributed_post` 创建 detached actor（13 只点名）→ 13，低。

## D. 环境缺口 / 无法核实

- SGLang v0.5.15.post1 无本机源码：`/pause_generation` 默认模式（决定 10/13 矛盾的实际结论）、`abort_all` 是否返回部分输出、`/v1/loads` 字段、SSE 是否累计输出、禁用生成健康检查后 `/health_generate` 行为，均未核。
- Ray 同步 actor 串行、placement group、NIXL 只按文档契约理解；异步 checkpoint 那条依赖"默认串行执行"，已确认 RolloutManager 创建时未设 max_concurrency。
- 全部静态审查，未运行 GPU 或 Ray；SVG 只抽文字核对，未渲染目检（协调者另做了渲染检查）。

## 协调者复核

- 页 10 异步 checkpoint 切口：属实（分析推导）。`train_async.py::train` 先提交 `generate.remote(rollout_id + 1)`，后 `ray.get(rollout_manager.save.remote(rollout_id))`；`slime/ray/placement_group.py` 构造 RolloutManager 的 options 只有 `num_cpus/num_gpus/runtime_env`（NIXL 时加 `enable_tensor_transport`），`slime/ray/rollout.py` 中 `generate`、`save` 均为同步方法，无 async 方法，按 Ray 同步 actor 串行契约 save(i) 排在 generate(i+1) 之后。
- 页 13 ABORTED 可达性：`slime/utils/types.py::Sample._apply_meta_info` 含 `case "abort":` 分支，页面"实际到不了"的判断缺依赖侧前提，属实。
