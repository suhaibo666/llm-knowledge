# A 组：01 架构总览、02 快速开始与配置指南

评审者：独立评审（未参与写作），只读。基线 `THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`，本机 `/Users/suhaibo/97-llm/slime-681b3adc`。

## A. 判定表

| page | profile(fit?) | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | profile-check | host | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 | software-architecture（契合） | pass | pass（缺口见 B8） | pass | layout §3.3、timing §3.2、transform §3.5、coupled-planes §2.2 | pass（§3.4 minor） | 3/3：`rollout.py::RolloutManager._get_updatable_server`；`update_weight_from_distributed.py::UpdateWeightFromDistributed.update_weights`；`megatron_utils/model.py::train_one_step` | architecture：FAIL 5（§5.1）、FAIL 6（§5.3）；第 7 项有 minor | pass（nit） | REJECT |
| 02 | guide（契合） | pass | FAIL：§2 图中 GV 节点，§7.1 `--sglang-*` 行 | pass | none | n/a | 3/3：`slime/utils/arguments.py:2032-2067`；`slime/ray/rollout.py:1274-1298`；`sglang_utils/arguments.py:189-212` | n/a（命令和 flag 均能解析） | minor | REJECT |

## B. 发现（按严重度排序）

1. **[major] 01 §5.1/§5.3 SFT**
   - 问题：入口、命令和调用树都写 `train.py`，还称选项"来自冻结的 ReTool SFT recipe"。但基线的 4 个 SFT recipe 全部用 `python3 train_async.py`：`scripts/run-qwen3-4B-base-sft.sh`、`run-qwen3-235B-A22B-sft.sh`、`run-qwen3.5-35B-A3B-sft.sh`、`examples/retool/retool_qwen3_4b_sft.sh`。28 页已写明这一点，两页矛盾，冲突未显性。源码上 `train.py` 加 `--debug-train-only` 也能跑。
   - 修复：按 recipe 入口改，或显式说明两个入口的差异。
2. **[major] 01 §5.1 场景清单未对账**
   - 缺的顶层入口：`megatron_utils/server/megatron_server.py::main`（归 20）、`slime_plugins/rollout_buffer/` 服务（归 19）、AMD `scripts/run-qwen3-4B-amd.sh`、reproducibility `scripts/run-qwen2.5-0.5B-reproducibility.sh`、`examples/delta_weight_sync` 与 release-train 发布、fault tolerance。
   - 未归类的 examples：search-r1、tau-bench、strands_sglang、eval_multi_task、train_infer_mismatch_helper。
   - 没有 legacy/stale 类：`SglangConfig.from_prefill_num_servers` 自己就标了 legacy。
   - 修复：补行、分类，并链接归属页。
3. **[major] 02 §2 图、§7.1**
   - 问题：页面称 namespace 合并后做"SGLang 原生校验"。实际调用的是 slime 包装 `sglang_utils/arguments.py::validate_args`。`ServerArgs(**…)` 只在 engine actor 的 `SGLangEngine._init_normal` 里构造，时间在 placement 和 router 之后。external 路径从不构造，只在 `_init_external` 比对 `/server_info`。
   - 修复：把原生校验移到 engine 初始化阶段，并与 argparse 的 type/choices 失败区分开。
4. **[minor] 02 §4.1** "只有 colocate 才派生 `rollout_num_gpus`"不对：external 模式在解析期由 `external.py::apply_external_engine_info_to_args` 无条件覆盖为探测到的 GPU 总数（在 `slime_validate_args` 内发 HTTP 请求）。与 19 页不一致。
5. **[minor] 02 §3 与 §4.1 自相矛盾**：§3 说 colocate"强制"卸载，§4.1 说"默认"。实现上 `slime_validate_args` 只在值为 None 时置 True，所以 `--no-offload-*` 能生效（release_train、use_critic 例外）。parser help 却写"always true"，这个冲突页面没有点出。
6. **[minor] 02 §4.3 缺违反"产出=消耗"时的行为**：`dp_schedule.py::build_dp_schedule` 按整除取步数，尾部 rollout 静默丢弃，只有 `num_steps>=1` 一个断言。§3 表写"global batch=训练样本数"，与该函数 docstring 按 rollout 计数不符。`docs/zh/get_started/quick_start.md:163` 称 num-steps-per-rollout 默认 1，parser 默认是 None。
7. **[minor] 02 §5.1/§7.1 两处文档冲突未显性**：help 称仍接受 legacy `critic` 配置，但 `parse_megatron_role_args` 断言必须有顶层 `megatron`，`tests/utils/test_megatron_role_config.py::test_requires_top_level_megatron_key` 也明确拒绝。`docs/{zh,en}/advanced/megatron-config.md:77` 用了不存在的 `--use-critic`，会被静默吞掉。
8. **[minor] 01 §2.2/§5.2 ASCII 调用树**：同步树缺 serving 启动与健康等待链 `RolloutManager.__init__→start_rollout_servers→_resolve_sglang_config/_start_router→ServerGroup.start_engines→ray.get(init handles)`；首次发布缺 `connect_rollout_engines`；`train_actor` 下缺 `compute_log_prob`、`compute_advantages_and_returns`；异步树循环首的 `ray.get` 没标"future 非空"条件；保存/评测/dispose 子树和 `[初始发布]` 注记没有映射到模块；§5.2、§5.4 卡片缺失败语义。
9. **[minor] 01 §3.7** worker type 列举漏了 `encoder`（`ServerGroupConfig.__post_init__`，EPD 两阶段启动），与 19 页矛盾。
10. **[minor] 01 路径与位置错误**：§4 的 `megatron_to_hf.py` 在基线不存在，实为目录；§5.1 fully-async 的 rollout function 在核心 `slime/rollout/fully_async_rollout.py`，不在 `examples/fully_async/`；§3.4 图里 P/Q 没有带过过滤和补采分支。
11. **[nit]** 01 §3.1：只有 `debug_train_only`/`load_debug_rollout_data` 决定是否跳过 SGLang parser；01 §3.3：`ServerGroup` 不保存 init futures，而是返回它们；02 §4.2：distributed optimizer 是强制开启，不是默认；02 §4.4：delta 只检查 flag 非空，不检查目录存在；02 §7.1：`--sglang-config` 由 SGLang 包装 parser 注册；host：02 的"主题"顺序与正文相反，"最近更新"写了核验过程。

**合同漂移（非 rubric 失败，回 planning）**：01 在各模块下重述了 10–18 的大量内容；01 说"完整命令以 02 为准"，但 02 没有 async、SFT、eval-only 的命令；01 §3.1 与 02 §2 两张 parser 组装图语义不一致；02 §5.1 与 14 页的 role YAML 内容重复。

## C. 缺失内容

| 项 | 源码位置 | 建议归属 | 重要度 |
|---|---|---|---|
| 结构性校验：`--rollout-batch-size` 必填；`--num-rollout`/`--num-epoch` 必选其一；`--save-interval` 需 `--save`；kl_coef 与 kl_loss_coef 互斥；`over_sampling>=rollout_batch`；reinforce_plus_plus* 需 `--normalize-advantages` | `slime_validate_args` | 02 | high |
| `--use-kl-loss` 要求 `--ref-load` 存在并加载 ref 快照；GLM4-9B 脚本 coef=0 时仍会触发 | `slime_validate_args`、`create_actor_model` | 02 | high |
| wandb/tensorboard 参数，无页拥有；缺名称且无 `TENSORBOARD_DIR` 时抛 ValueError | `add_wandb_arguments`、`tensorboard_utils.py` | 02 | med |
| `--freeze-params-name-list` 与 `--only-train-params-name-list` 互斥，无页拥有 | `freeze_model_params` | 14 | med |
| slime 改写的 router 默认值（10/1.2/warn） | `add_sglang_arguments` | 02 | low |
| Megatron 包装的附加强制，以及 `max_position_embeddings` 默认值 | `megatron_utils/arguments.py` | 02 | low |
| `--padded-vocab-size`、`--qwen-gdn-backend` | 同上；`qwen3_5.py`、`qwen3_next.py` | 23 | low |
| `--min-batch-collection-ratio`、`--http-proxy`：已注册但无消费者（死参数） | `slime/utils/arguments.py` | 19/02 | low |

已有归属、不算缺失：eval-* 归 27，external 探测归 19，`--num-gpus-per-node` 归 11，`--loss-mask-type` 归 28。

## D. 环境缺口 / 无法核验

- SGLang 没有 checkout：`ServerArgs.add_cli_args` 的实际 flag 名、构造期校验内容、router fork（0.3.2-9daabcd）的路由语义都无法核实，按依赖侧记录。
- Megatron 1dcf0daf 的 flag 大多由 dataclass 工厂生成，只抽查了 `--spec`、`--padded-vocab-size`、`global_batch_size`；`megatron*.patch` 对 flag 的增改未展开。
- 没有渲染 SVG/Mermaid，没有做 GPU/Ray 运行期验证。两页都没有基线之后的声明，所以没用到 4c193f1f。

## 协调者复核

- B1 属实：01 §5.1 表（第 422 行）与 §5.3 命令写 `python train.py`，而 4 个 SFT recipe 均为 `python3 train_async.py`；28 页第 120 行写"脚本实际使用 `train_async.py`"。
- B3 属实：02 §2 图 `GV["SGLang 原生校验"]`；`slime/backends/sglang_utils/arguments.py::validate_args` 只做别名归一化等包装处理，`ServerArgs(**server_args_dict)` 在 `slime/backends/sglang_utils/sglang_engine.py` 引擎启动时构造。
