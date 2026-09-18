# 协调者机械核验与渲染检查

本文件记录协调者本人完成的检查。评审者报告在同目录 `group_*.md`。所有检查只读 wiki 与源码，没有修改任何 wiki 页面。

## 1. T0 门禁

| 检查 | 命令 | 结果 |
|---|---|---|
| 链接 | `python tools/check_links.py --strict` | 453 页；broken、ambiguous、bare_index、stale_section、orphans 均为 0 |
| 公式 | `python tools/check_math.py --strict wiki/02_engineering/04_posttrain_frameworks/slime` | 24 文件，0 错误 0 警告 |
| Markdown / Mermaid | `python tools/check_markdown.py --strict <slime>` | 24 文件，0 错误 0 警告 |
| 资源 | `python tools/check_assets.py --strict <slime>` | 24 文件，0 错误 0 警告 |
| 旧式 locator | `python tools/check_locators.py --dir <slime>` | citations audited 15，pass 0；errors 0，warnings 2（25 页 `vllm-project/vime`、`vime` 不在 watchlist），env 13（`THUDM/slime@681b3adca 无本地 checkout`）。原因是 `docs/radar/watchlist.yaml` 的 slime `checkout: ../slime` 指向不存在的 `/Users/suhaibo/96-knowledge/slime`，门禁实际没有核验任何 slime 行号引用 |
| 图测试 | `node --test tools/figs/svg/lib/slime_*.test.mjs` | 8 个文件 41 项全部通过；8 个测试文件都读取页面正文；生成器输出与已跟踪 SVG 一致 |

## 2. 锚点机械核验（替代失效的 locator 门禁）

脚本直接读 `/Users/suhaibo/97-llm/slime-681b3adc`（`git rev-parse HEAD` = `681b3adca54105d5ecd3fb822fa0dc58a427e0f9`）。

| 类别 | 数量 | 结果 |
|---|---|---|
| `THUDM/slime` GitHub 永久链接（带 `#Lx-Ly`） | 557 | 全部钉在基线 commit；文件都在 `git ls-files` 中；行区间都在文件长度内；没有超过 100 行的区间 |
| 分布 | 02:66、10:49、19:52、20:50、21:40、22:74、23:54、24:75、25:6、30:54、31:37 | 01、11–18、26–28 没有行号链接 |
| 其他仓永久链接 | sgl-project/sglang 24（16:8、21:12、23:4）；vllm-project/vime 93（25） | 本机无对应 checkout，只核格式 |
| `path::symbol` 锚点 | 381 可解析 | 路径存在且符号出现在文件中 |
| 裸文件名锚点 | 46 | 如 `rollout.py::…`（2 个候选）、`arguments.py::slime_validate_args`（4 个）、`common.py::all_gather_param`（4 个）、`__init__.py::_LOADERS`（29 个）。分布：11:7、12:6、13:1、14:3、15:3、16:6、17:6、18:3、23:9、25:2。可定位但不符合"仓库相对路径 + 限定符号"的默认锚点形式 |
| 符号未找到 | 2 | 16 的 `update_weight/__init__.py::create_weight_updater`（页面已标为基线后 7e4ac3be 的符号）；25 的 `hf_checkpoint_saver.py::save_hf_model_bridge_to_path`（vime 侧符号）。两处都不是错误 |

## 3. SVG 渲染检查

用 `tools/mkdocs-site` 的 puppeteer-core 与本机 Chrome 按 viewBox 尺寸渲染 9 张 SVG，逐张目视，并用 `getBBox` 检查文字越界与文字互相重叠。

| 图 | 画布 | 文字元素 | 越界 | 文字重叠 | 目视问题 |
|---|---|---|---|---|---|
| `slime_architecture.svg` | 1120×810 | 40 | 0 | 0 | 无；外部运行时与依赖侧已标注 |
| `slime_ray_control_plane_layout.svg` | 1180×840 | 105 | 0 | 0 | 无 |
| `slime_sample_data_contract.svg` | 1180×1250 | 188 | **1** | 0 | 面板 3 右侧注释"默认 generate 让每个 sample task 都返回 plain Sample…"越出注释框与画布右缘；左侧加粗行标签"rollout_mask_sums（offpolicy 开）"越过面板左边框；底部留白偏大 |
| `slime_rollout_admission_timeline.svg` | 1180×900 | 139 | 0 | 0 | 左下面板图例行"蓝：入选与决定性判定…"横穿到右侧面板；页脚"源码基线"行压在左下面板底边上；无 streaming 泳道（见 B 组） |
| `slime_megatron_train_step.svg` | 1180×1420 | 250 | 0 | 0 | 面板 A 橙色框最后一行压在框底边；面板 B、C、D 含 5–6 行段落 |
| `slime_loss_reducer_ledger.svg` | 1180×1280 | 221 | 0 | 0 | 信息密集，可读 |
| `slime_weight_sync_planes.svg` | 1180×1484 | 221 | 0 | 0 | 五个面板塞进一张图，面板内多段多行说明；按 `drawing-wiki-figures`"注释要短""超过一页可读范围就拆图"应考虑拆分；图③"拷贝可能仍在排队"未标补丁闸门（见 D 组） |
| `slime_train_infer_replay.svg` | 1180×1052 | 118 | 0 | 0 | 推断与依赖侧已在图中标注；图① top-k 图注有误（见 D 组） |
| `slime_fault_recovery_timeline.svg` | 1180×900 | 75 | 0 | 0 | 无；文档与实现默认值冲突已在图中显性 |

说明：`getBBox` 只检查文字与画布、文字与文字，检查不到文字压框线或跨面板，后两类来自目视。

## 4. 内容保留对账（ea68986 → HEAD）

对 2026-09-10 提交 `ea68986` 与当前 HEAD 之间改动的 14 个文件（01、10–19、30、31、index），按页抽取反引号内的标识符、flag 与 wikilink，检查在整个 slime 域中消失的项。

| 页 | 旧版标识符 | 从全域消失 | 结论 |
|---|---|---|---|
| 01 | 55 | `RolloutEngine`、`TrainerEngine`、`set_weights` | 基线源码中不存在，属旧示意名，删除合理 |
| 13 | 95 | `target_data_size` | 内部局部变量，概念仍在 |
| 14 | 114 | `max_per_bin`、`mpu.get_data_parallel_rank` | 内部变量/调用，概念仍在 |
| 15 | 51 | `new_log_prob`、`world_size_DP` | 公式记号 |
| 16 | 81 | `expert_id`、`experts_per_ep_rank`、`loaded_weight`、`start_idx`、`update_weight_buffer_size` | 前四项为局部名；`--update-weight-buffer-size` 仍在 01、16、30 |
| 17 | 117 | `--tis-lower-bound`、`--tis-upper-bound`、`--tis-batch-normalize`、`output_index` | 三个 `--tis-*` 在基线只出现于 `examples/train_infer_mismatch_helper/README.md`，实现读取 `mis.yaml` 属性 `tis_lower_bound` 等，页面保留了属性名，属于已记录的更正；README 的 CLI 写法与实现不符这一冲突未在页面显性（P2） |
| 18 | 89 | `add_fault_tolerance_arguments` | 函数名锚点；默认值结论仍在正文与图中 |
| 10、11、12、19、30、31、index | — | 0 | — |

没有 wikilink 丢失，没有概念级静默丢失。

## 5. CLI flag 独立枚举

用 `ast` 解析 `slime/utils/arguments.py`、`slime/backends/sglang_utils/arguments.py`、`slime/backends/megatron_utils/server/arguments.py` 中所有 `add_argument` 调用，按首个长选项去重。SGLang `ServerArgs`、router 与 Megatron 的透传参数不计入。

- 共 218 个 slime 自有 flag；182 个在 slime 页中以连字符或下划线形式出现；**36 个在全库 wiki 中都未出现**。
- 36 个未出现的 flag 按类别：
  - 指标与日志落点（17）：`--use-wandb`、`--wandb-mode`、`--wandb-dir`、`--wandb-key`、`--wandb-host`、`--wandb-team`、`--wandb-group`、`--wandb-run-id`、`--disable-wandb-random-suffix`、`--wandb-always-use-train-step`、`--use-tensorboard`、`--tb-project-name`、`--tb-experiment-name`、`--log-multi-turn`、`--custom-rollout-log-function-path`、`--custom-eval-rollout-log-function-path`、`--memory-snapshot-dir`
  - 评估采样覆盖（9）：`--eval-input-key`、`--eval-label-key`、`--eval-tool-key`、`--eval-temperature`、`--eval-top-p`、`--eval-top-k`、`--eval-max-response-len`、`--eval-min-new-tokens`、`--eval-max-context-len`（27 讲了语义但未点名）
  - rollout buffer 插件（4）：`--rollout-buffer-url`、`--fetch-trajectory-retry-times`、`--min-batch-collection-ratio`（基线无读取方）、`--rollout-task-type`
  - 训练与模型（4）：`--qwen-gdn-backend`、`--freeze-params-name-list`、`--use-rollout-entropy`、`--opd-teacher-ckpt-step`
  - 死参数或透传别名（2）：`--http-proxy`（无读取方）、`--sglang-pipeline-parallel-size`
- 早先正则扫描得到 241 个，多出的是重复与注释伪影（如 `--foo-bar`、`--sglang-foo-bar`），以 AST 结果为准。

## 6. 跨域链接

- 23 篇 slime 正文指向 slime 目录以外 wiki 页面的链接只有 1 条：17 → `07_training_reliability/10_determinism_and_numerical_reliability_analysis`。
- 指向 slime 的入链来自：`courses/posttraining_frontier`、`wiki/index.md`、`01_theory/04_posttraining/01_posttraining_frontier_map_analysis`、`30_rl_framework_comparison`、`21_areal_async_architecture_analysis`、后训练框架 index、`verl/10_verl_end_to_end_iteration_analysis`。
- 已存在但未被链接的相关 owner：
  - `02_engineering/02_train_frameworks/megatron-lm/`（41 篇，如 13 CP、16 分布式优化器、19 分布式 checkpoint、27 作业韧性、30 RL 训推一致性、33 RL runtime、38 logits 蒸馏、41 配置面）。注意其基线为 `NVIDIA/Megatron-LM@85902ef5`（dev，2026-09-01），slime 镜像钉的是 `1dcf0daf`（是其祖先），链接时需注明版本差。
  - `02_engineering/03_infer_frameworks/`：`sglang/`、`speculative_decoding/`、`mooncake_analysis`、`vllm/17_vllm_quantization_analysis`。
  - `01_theory/04_posttraining/`：`11_ppo_analysis`、`20_grpo_analysis`、`22_gspo_analysis`、`14_on_policy_distillation_analysis`、`25_on_policy_off_policy_staleness_analysis`、`31_reward_hacking_defense_analysis`。
  - `02_engineering/04_posttrain_frameworks/`：`13_opd_infra_mechanism_analysis`、`32_opd_framework_support_comparison`、`12_rl_infra_efficiency_analysis`、`23_dora_multi_version_rollout_analysis`，以及 verl 的迭代、权重发布、agent loop、checkpoint 恢复页。
  - `02_engineering/07_training_reliability/`：`11_fault_tolerance_and_recovery_analysis`、`12_training_dynamics_stability_analysis`、`20_batch_invariance_guide`。

## 7. 索引与计数

- slime：`find` 得 24（23 正文 + index），与 slime/index、后训练框架 index、`wiki/index.md` 一致。
- 后训练框架目录：`find` 得 53，`wiki/index.md` 第 52 行写 52（DORA 页 `2f6a04d` 未回写主索引）。属 slime 域外的顺带发现。

## 8. 源码与环境

- slime 基线 worktree：`/Users/suhaibo/97-llm/slime-681b3adc`，HEAD 与页头一致，工作区干净；全程未移动。
- 用户原 checkout：`/Users/suhaibo/97-llm/slime` 在 `4c193f1f`（2026-09-03），比基线新 26 个提交（含 v0.3.2），只用 `git log`/`git show` 读取。
- Megatron：slime `docker/Dockerfile` 钉 `MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087`；该对象在 `/Users/suhaibo/97-llm/Megatron-LM`（HEAD `1ff25ca7`）中存在，评审用 `git show 1dcf0daf:<path>` 读取。
- SGLang：Dockerfile 底座 `slimerl/sglang:v0.5.15.post1-cu129`，本机无源码。slime 自带补丁 `docker/patch/latest/*.patch` 在基线可读，评审据此核验补丁侧行为。
- vime：`/tmp/slime-audit-vime` 只剩空目录骨架（不是 git 仓库），25 页 vime 侧结论本轮无法复核；未克隆。
