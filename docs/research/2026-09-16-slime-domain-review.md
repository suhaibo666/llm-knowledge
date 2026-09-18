# slime 域全量独立审查（2026-09-16）

本轮对象是 `wiki/02_engineering/04_posttrain_frameworks/slime/` 下的 23 篇正文和 index，另含 9 张 SVG 与配套生成器、测试。按文档类别审查：

- **导航**：index。检查宪法对 index 的限制和读者入口。
- **使用说明**：02（`_guide`）。检查命令、flag、校验时机与失败定位是否与源码一致，以及使用完备度。
- **架构分析**：01。检查 software-architecture profile 和 8 项架构检查。
- **核心机制（段 1）**：10–19。11–18 于 2026-09-10/11 按 feature 画像重写；10、19 仍是旧版因果五拍的 mechanism 页。
- **能力专题（段 2）**：20–24、26–28，按 feature 画像审查。页面结构与画像标题不同本身不算缺陷，缺的是实质内容才算。
- **衍生实现审计**：25（vime，另一仓库）。
- **诊断方法（段 3）**：30、31，按 mechanism 画像加经验/诊断视角审查。

所有页面都过基础五项 rubric；另有两项全域要求：**遗漏内容**（覆盖度）和**跨页、跨域一致性**。

- **源码**：`THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`（main，2026-08-12），本机 detached worktree `/Users/suhaibo/97-llm/slime-681b3adc`，全程只读、未移动。Megatron 按 slime `docker/Dockerfile` 钉的 `1dcf0daf` 用 `git show` 核对。SGLang 无本机源码，但 slime 自带的 `docker/patch/latest/*.patch` 在基线可读。vime `8144096e` 的临时 checkout 已被清空，25 页 vime 侧本轮无法复核。
- **组织方式**：8 个独立评审者均未参与写作：7 组按页分工，1 组做覆盖度与跨域审计。协调者负责机械门禁、锚点机械核验、SVG 渲染检查、内容保留对账、flag 枚举和 P0 源码复核，没有修改任何 wiki 页面。
- **明细**：各组报告在 [`2026-09-16-slime-domain-review/`](2026-09-16-slime-domain-review/)，协调者的机械核验在其中的 `coordinator_mechanical_checks.md`。

## 0. 结论

| 维度 | 结果 |
|---|---|
| 页面判定 | **PASS（带 P2）4 篇**：11、12、18、19；index 通过（有 P2）。**REJECT 19 篇** |
| 严重度 | **P0 5 条**，涉及 7 页（14、15、17、21、24、30、31）。**P1 约 30 条**。**P2 约 95 条** |
| 机械门禁 | T0 全绿：links 453 页五项为 0；slime 目录 math、markdown、assets 均为 0 错误 0 警告；图测试 41/41 通过。但 `check_locators` 对 slime 实际失效（watchlist 路径错，见 §12），由协调者脚本代核：557 条 slime 行号链接全部命中基线文件和合法区间 |
| 内容保留 | 相对 `ea68986`，01、11–18 的重写没有概念级静默丢失 |
| 遗漏内容 | 218 个 slime 自有 CLI flag 中 **36 个全库未出现**。**PD 分离、依赖补丁与平台、GLM-5 DSA 训练**没有 owner 页。23 篇正文对 slime 以外的 wiki 只有 **1 条**出链 |

REJECT 的 19 篇大致分三类，不必整页重写：

1. **有 P0 事实错误**：14、15、17、21、24、30、31。
2. **旧模板页缺原理图回放、变体枚举或成本账**：10、20、22、23、26、28。
3. **依赖补丁边界或页面归属问题**：13、16、25，以及 01、02 的场景与校验时机。

### 需要优先修复的 P0（均已由协调者在源码中复核属实）

| # | 页面与位置 | 页面说法 | 源码事实 |
|---|---|---|---|
| 1 | 15 §2.3「KL 位置」行（第 201 行）、§5.2 误读表（第 380 行）、§6 `--kl-coef`（第 431 行）、图面板 2；14 §2.1；31 §4.2 配置表（第 98 行） | `--kl-coef` 在 advantage 之前做 reward 整形，对所有估计器成立 | `--advantage-estimator` 默认 `grpo`。`loss.py::compute_advantages_and_returns` 的 grpo、gspo、cispo 分支调用 `ppo_utils.py::get_grpo_returns`，该函数返回 `torch.ones_like(kl[i]) * rewards[i]`，与 KL 数值无关。只有 ppo、reinforce_plus_plus、reinforce_plus_plus_baseline 用到 `kl_coef`。默认估计器下设 `kl_coef≠0` 只会加载 ref、多做一次前向、关掉 logprob 复用、与 `kl_loss_coef` 互斥，并记录 `rollout/kl` |
| 2 | 17 §2.2.3、§5.1 表（第 256 行）、§6、图①图注 | 支持集重放"top-k 不在范围内" | `docker/patch/latest/sglang-top_p.patch::_top_p_keep_mask_sorted` 的说明是 "Reproduces SGLang's sampler truncation (rank < top_k, cumulative prob within top_p…)"，首行为 `keep = ranks < top_ks.view(-1, 1)`。`rollout_top_p≠1` 时返回的候选集已包含 top-k 截断。`top_p=1` 且 `top_k≠-1` 时不请求 ids，两侧都在全词表域，但此时用作 old policy/TIS 的不是行为概率 |
| 3 | 21 §5.2/§5.3（第 129 行） | CI 在全截断场景检查非 MTP 参数梯度为 0，"证明…主 policy loss 被 mask 后不会污染主干梯度" | `tests/test_mimo_7B_mtp_only_grad.py` 验证的是 MTP loss 的梯度**只流向 MTP 层**。这种隔离来自 slime 自带的 `docker/patch/latest/megatron.patch`：`decoder_input.detach()`、`keep_graph=True→False`、`mtp_output_weight.detach()`，并新增 `mtp_kwargs`。上游 `Megatron-LM@1dcf0daf` 的 `multi_token_prediction.py` 为 `keep_graph=True`，没有隔离。页面结论读反了测试含义，也漏写了对补丁的依赖 |
| 4 | 24 §4.3（第 105 行） | adapter 先把响应 flush 给客户端，flush 成功后才调用 `record_turn` | 只对流式成立。`slime/agent/adapters/common.py` 先拿到 `response`，再 `self.manager.record_turn(...)`，最后 `return response`。非流式 `openai.py::_respond` 返回 `web.json_response(...)`，由 aiohttp 在 handler 返回后才发送。只有 `_render_stream` 先 `await out.prepare(request)`。因此非流式时，客户端断开也会记下它从未收到的 assistant turn |
| 5 | 30 §7 决策矩阵（第 235、237 行） | "trainer ready→data ready 需自建计时"；"publish/offload 分段需 timer/trace" | `actor.py` 第 433 行 `with inverse_timer("train_wait"), timer("train")`，`update_weights` 带 `@timer`。`train_metric_utils.py` 第 27 行统一映射为 `perf/{key}_time`，并派生 `perf/train_wait_time`、`perf/wait_time_ratio`、`perf/step_time`、`perf/update_weights_time`。这些默认键已经存在 |

## 1. 导航：index

判定：**通过（P2）**。

- 23 行条目与目录文件一一对应。"23 篇内容页 + index 共 24 个"与后训练框架 index、`wiki/index.md` 一致。
- P2：「阅读术语」一节承载术语定义和归属说明，超出宪法"index 只维护本目录入口表"的限制。建议移入 01 或对应 owner 页，index 只留链接。
- P2：阅读顺序句和术语段用纯数字引用（"先读 01 架构"、"见 11"、"见 12、15"），不符合"读者路径用语义标签"。26–28 行带显示别名，其余 20 行不带，呈现不一致。页头「系列规模」一行夹带了新增历史。
- P2：没有 Knowledge Gaps 小节；§8 列出的无 owner 能力宜在此登记。
- 25 作为段 2 条目的归属问题见 §6。

## 2. 使用说明：02

判定：**REJECT**。hop-walk 在校验时机一环失败。

- **P1 §2 图、§7**：图中合并后的节点写"SGLang 原生校验"，实际调用的是 slime 包装函数 `sglang_utils/arguments.py::validate_args`，只做别名归一和整除、互斥断言。`ServerArgs(**server_args_dict)` 到 `sglang_engine.py` 启动引擎时才构造，已经在 placement 和 router 之后；external 路径则从不构造。§7"失败发生在哪一层"缺少"引擎启动时 ServerArgs 报错"这一行。
- **P2**：
  - §4.1 称只有 colocate 才派生 `rollout_num_gpus`；external 模式在解析期由 `external.py::apply_external_engine_info_to_args` 覆盖，与 19 不一致。
  - §3 说 colocate"强制"卸载，§4.1 说"默认"，自相矛盾；实现只在值为 None 时置 True，parser help 却写 "always true"，冲突未标。
  - §4.3 未写 `build_dp_schedule` 按整除取步数，尾部 rollout 会被静默丢弃。
  - §5.1、§7.1 两处文档与实现的冲突未显性：legacy `critic` 配置会被 `parse_megatron_role_args` 断言拒绝；`megatron-config.md` 使用了不存在的 `--use-critic`。
  - 页头"主题"顺序与正文相反，"最近更新"写的是核验过程。
- **完备度缺口**：
  - 01 说"完整命令以 02 为准"，但 02 没有 async、SFT、eval-only 的命令。
  - 结构性校验没有集中写出：`--rollout-batch-size` 必填；`--num-rollout`/`--num-epoch` 必须给一个；`--save-interval` 需要 `--save`；两个 KL 系数互斥；`over_sampling ≥ rollout_batch`；REINFORCE++ 需要 `--normalize-advantages`；`--use-kl-loss` 需要 `--ref-load`。重要度高。
  - W&B/TensorBoard 配置（中）。
  - checkpoint 导入/导出路线（中）。
  - `qa.md` 列出的 FAQ 失败模式（中）。
  - 多节点启动与网络环境变量（中）。

## 3. 架构分析：01

判定：**REJECT**。base 五项通过；8 项架构检查中**第 5、6 项 FAIL**，第 7 项有 P2。

- **P1 第 5 项，场景清单未与实际入口对账**：
  - 缺 `megatron_utils/server/megatron_server.py::main`（归 20）、rollout buffer 服务（归 19）、AMD 脚本、reproducibility 脚本、`examples/delta_weight_sync` 与 release-train、容错场景。
  - search-r1、tau-bench、strands_sglang、eval_multi_task、train_infer_mismatch_helper 没有归类。
  - 没有 legacy/stale 类，但 `SglangConfig.from_prefill_num_servers` 自己就标了 legacy。
- **P1 第 6 项**：§5.1 表（第 422 行）和 §5.3 命令写 `python train.py`，并称选项"来自冻结的 ReTool SFT recipe"。4 个 SFT recipe 实际都是 `python3 train_async.py`，28 页第 120 行也这样写，两页矛盾。`train.py` 加 `--debug-train-only` 在源码上也能跑，但出处写错。
- **P2 第 7 项及其他**：
  - 同步调用树缺 serving 启动与健康等待链（`start_rollout_servers → _start_router → ServerGroup.start_engines`）；首次发布缺 `connect_rollout_engines`；`train_actor` 下缺 `compute_log_prob`、`compute_advantages_and_returns`；异步树循环首的 `ray.get` 没标"future 非空"条件；部分子树没有映射到模块。
  - §3.7 worker type 漏了 `encoder`，与 19 矛盾。
  - §4 的 `megatron_to_hf.py` 实为目录；fully-async 函数在 `slime/rollout/fully_async_rollout.py`，不在 examples。
- **合同漂移**（交 planning，不算 rubric 失败）：
  - 01 共 549 行，在模块设计下重述了 10–18 的大量内容。
  - 01 §3.1 与 02 §2 的两张参数组装图语义不一致。
  - 02 §5.1 与 14 的 role YAML 内容重复。
- 静态视图单一分类轴成立，依赖侧已标注，渲染无越界。

## 4. 核心机制（段 1）：10–19

| 页 | 画像 | 判定 | 主要问题 |
|---|---|---|---|
| 10 端到端迭代 | mechanism（旧模板） | REJECT | **P1** timing 触发，但三张 Mermaid 只画调用顺序；缺 sync / async(interval=1) / async(interval=2) / fully-async 同一例子的时间线（serving 版本、policy age、发布栅栏、积压队列）。**P1** §2.4/§6"保存和恢复都以 rollout id 对齐"对 `train_async.py` 不成立：`train_async.py::train` 先提交 `generate.remote(rollout_id+1)` 再 `save.remote(rollout_id)`；RolloutManager 是同步 actor（`placement_group.py` 未设 max_concurrency，`rollout.py` 无 async 方法），save(i) 排在 generate(i+1) 之后，保存的游标已跨过 i+1 批，从 i 恢复会跳过一批（按 Ray 串行契约推导，调用顺序与 actor 选项已复核）；fully-async 队列与在途组也不进 checkpoint。**P2** ABORTED 可达性与 13 矛盾；§8"整条轨迹重来"与续生成实现不符；40 多条行号链接，没有集中的阅读路线 |
| 11 Ray 控制面 | feature | PASS（P2） | debug 两行"不建 trainer"错（actor 仍创建，`init` 早退），与 14 矛盾；full+disk 调用树与账本省略 gloo barrier 和 post-write hook；release-train 约束不全；critic 备份理由写反；主题预告结论 |
| 12 Sample/DataSource | feature | PASS（P2） | 调用树一处父子关系错；save/load 早退条件不准；reward 归一化开启分支未回放 |
| 13 SGLang rollout | feature | REJECT | **P1** streaming 是独立在线数据面（有 CI 测试 `test_qwen3_4B_streaming_partial_rollout`），原理图没有它的泳道。**P1** §2.3.2 称 ABORTED 回队分支"默认 generate 到不了"，但 `Sample._apply_meta_info` 有 `case "abort"` 分支，是否可达取决于 SGLang pause 语义（依赖侧，未核），且与 10 矛盾。**P2** streaming 绕过 `_post` 的 60 次重试；漏 slime 改写的 router 均衡默认值（abs 10、rel 1.2、log warn）；fully-async 下 hook 拿到的 rollout_id 取自模块全局变量 |
| 14 Megatron 训练 | feature | REJECT | **P0 #1**。**P1** `--allgather-cp` 在 CP>1 时只允许 DSA 模型，否则解析期抛 ValueError（`megatron_utils/arguments.py::_validate_allgather_cp_supported`），页面把它当通用兄弟布局，只在 §6 标"（DSA 模式）"。**P2** train_iters 漂移原因误引；销毁 WORLD 的理由误读（实为逐子组销毁会死锁）；PPO critic 数据面未回放 |
| 15 Loss 归约 | feature（数值） | REJECT | **P0 #1**。**P2** 称 Megatron"未钉版本"（Dockerfile 钉 1dcf0daf，评审已核三处缩放与页面一致）；MTP 与 MoE aux loss 不经 slime 归约器，未点名；OPSM mask 粒度与 clipfrac 分子写错。数值回放（三种均值、DP×CP 账本、per-token、GSPO、PPO 落点）逐项复算通过 |
| 16 权重同步 | feature | REJECT | **P1** 依赖补丁边界：`sglang-deterministic.patch` 已在 flattened-bucket 加载后 `torch.cuda.synchronize()`，页面仍写"slime 无法证明 SGLang 何时完成 GPU 拷贝"，图③写"拷贝可能仍在排队"；`/post_process_weights`、`/weight_version` 的改写来自 `sglang.patch`；Dockerfile 的 `PATCH_VERSION=latest` 与 `ENABLE_SGLANG_PATCH` 闸门未写。四条数据面的数值全部独立复算通过 |
| 17 训推一致性 | feature | REJECT | **P0 #2**。**P1** GLM-5 对齐栈缺补丁交接：`alignment/env.py` 的变量由 `sglang-deterministic.patch` 消费，`MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS` 由 `megatron-sglang-aligned.patch` 消费；`test_glm52_6layer_deterministic_e2e.py::_skip_reason` 缺补丁时 skip 而非 fail，门禁表未写。**P2** 图① −1.313 未标依赖侧；L3"取 2 次"的前提条件未写 |
| 18 容错与可观测性 | feature | PASS（P2） | W&B/TensorBoard 开关与 `--custom-rollout-log-function-path` 没有归属；主题含论断；tracker 写入时机可在 Megatron@1dcf0daf 核实后升为已核实 |
| 19 backend 扩展 | mechanism（旧模板） | PASS（P2） | external 协议边界漏了 `/pull_weights` 补丁依赖；buffer 插件缺失败边界（只留最新组、无 rollout logprob）；契约测试的适用范围未写（仓内 fully-async 与 buffer 函数自己过不了） |

## 5. 能力专题（段 2）：20–24、26–28

| 页 | 判定 | 主要问题 |
|---|---|---|
| 20 OPD | REJECT | **P1** 学生项来源写成无条件的"当前学生重算"，实际受 `--use-rollout-logprobs`（`loss.py` 第 731 行直接取 rollout logprob）和 `--keep-old-actor`（切到 old_actor 前向）影响，参数校验不禁止这两种组合。**P1** logprob 对齐（丢首项、尾部裁剪）与逐 token advantage 注入没有最小实例和原理图。**P1** 缺 reverse-KL 的 score-function 推导、"独立 KL loss"备选的取舍判据，也没说明经过 clip 和 normalize 后 λ 不再是绝对系数。`--opd-teacher-ckpt-step` 全库未出现 |
| 21 投机解码与 MTP | REJECT | **P0 #3**。**P1** MTP 层号从 0 开始、只有 expert 下标加 EP offset 的改名规则，以及 `spec_accept_rate` 按样本等权平均，都没有复现；没有成本账本、紧凑阅读路线或调用树。**P2** PP 放置在 Megatron@1dcf0daf 可核验，却写"不能断言"；写"未额外打补丁"，但官方镜像会应用 SGLang 补丁（其中新增 draft 的 `post_process_weights` 转发）；页头"适用范围"写了本轮核验边界 |
| 22 低精度 | REJECT | **P1** 原理图示例退化：单个恒值 128×128 block 无法区分 blockwise 与 per-tensor，二维块网格也不宜用 Mermaid；可用 `tests/test_block_fp8_zero_block.py` 的 256×256 例子生成 SVG。**P1** INT4 fake-QAT 缺数值语义：在 `megatron.patch` 的 `_FakeInt4QuantizationSTE`，只作用于 MoE experts，q_max=7，scale 下限 1e-5，STE 直通。**P1** compressed-tensors 在线量化只支持 int4，与转换工具能产出的 FP8 channel、W8A16 冲突，未写。**P2** 基线后 2fa9a442（`--save-hf` 关闭 UE8M0 转换）与 876cd89b（ROCm INT4）漏记；`NVTE_FP8_BLOCK_SCALING_FP32_SCALES` 已由 `actor_group.py` 默认注入 |
| 23 模型架构扩展 | REJECT | **P1** 构造侧 spec 变体没有枚举依据：脚本绑定了 glm4、glm5（DSA）、minimax_m2、qwen3_5、qwen3_next，`--qwen-gdn-backend`（fla/flashqla）未点名。**P1** 只用 Mermaid 文本重放 GLU TP2；qwen3_next 门控 ×2 交错、EP 专家编号偏移、CP 两段重组都没重放。**P2** 默认 Direct 路径只断言 `partition_dim is not None`，缺 `partition_stride` 断言，非 GLU 的 fc1 会被静默错拆；46 个裸文件名锚点中 9 个在本页 |
| 24 Agent 工作流 | REJECT | **P0 #4**。**P1** 把消息层 rewrite-merge（`_try_merge_assistant_rewrite`）与 token drift 混成一层，realign 前后状态没有重放。**P2** 只有外层 `rollout_guard_sec` 返回 ABORTED；内层 `agent_time_budget_sec` 超时后仍评分，并返回半棵树的 Samples；`finish_session` 截断后状态仍为 COMPLETED，未写 |
| 26 多模态 VLM | REJECT | **P1** mRoPE 推进规则 `current_position += max(h,w)//merge`（所以文本从 3 而不是 5 继续）没有推导；`_inject_vision_embeddings` 从完整位置到 feature 行再到 CP 局部索引的过程没有例子；mRoPE 网格用的是文本。**P1** 无成本账：每个 TP rank 复制一份 ViT，每个 CP rank 都对整批图像跑 ViT，processor 本地与 SGLang 各算一次。**P2** "累计 budget 4096"不准 |
| 27 评估路径 | REJECT（小修） | **P2** 同级 eval 函数未点名（forge 回放 `eval_` dump；buffer 示例抛 NotImplementedError）；采样数不齐时 `compute_pass_rate` 断言直接崩溃，不只是口径问题；9 个 `--eval-*` flag 讲了语义但没写 flag 名；`tests/test_eval_config.py` 不在阅读路线里 |
| 28 SFT 与 loss mask | REJECT | **P1** mask 图无法重放：流程图加位串；`get_batch` 的 `F.pad(mask,(prompt_length-1,1))` 平移和 qwen3_5 的字符→token 投影不在图里；示例不是真实模板，且漏了会被训练的 `<\|im_end\|>\n`。**P1** 缺长度包络：list 形式 prompt 只告警不过滤，超长样本单独占一个 micro-batch 且可超 token 上限，全程不截断。**P2** SFT 脚本带 `--debug-train-only`，`RolloutManager.eval` 直接返回，评估是静默空操作，页面却让读者"另用 27 的路径" |

## 6. 衍生实现审计：25（vime）

判定：**REJECT**，归属问题交 `planning-codebase-analysis`。

- **P1 把承袭内容记成 vime 新增**：slime 基线的 `sglang_config.py` 已有 `valid_types = {…, "encoder"}`，启动流程也已有 "EPD phase 1"；"只选第一个可更新 server"、external 文档的 `delta + nccl` 行在 slime@681b3adc 同样存在。页面把这些算作 vime 新增（第 132 行"又加入了 encoder disaggregation"），并据此推断"vime 仍快速演进"。应加一列"承袭 / 新增 / 改写"，并补 vime 从 slime 分叉的基线。
- **P1 页头**：用了 7 个非标准字段（定位、上游对照基线、基线提交时间、核验日期、系列入口），缺"主题""适用范围"；基线行缺分支与日期；"最近更新"写的是核验过程。
- **P1 放置与登记**：
  - 另一仓库的独立分析放在 slime/ 下，与 AReaL(21)、ROLL(22)、DORA(23) 放在父域第 2 段的先例不一致。
  - `vllm-project/vime` 不在 `docs/radar/watchlist.yaml`，radar 看不到漂移；vime 基线提交日期为 2026-08-03，上次核验是 2026-08-18。
  - 后训练框架 index 的系统表里，slime 的"23 篇"已含 25，又单列 vime"1 篇"。
- **P2**：
  - 适配面清单与 19 §6 重复，条数不一致，应以 19 为唯一清单。
  - "强 / 强但慢"没有标为分析判断。
  - 99 条逐句行号链接，显示文本分不清指向哪个仓。
- **环境**：vime 无可用 checkout，93 条 vime 链接只核了格式（都钉在 `8144096e`、行区间 ≤100 行），vime 侧结论本轮均未复核。

## 7. 诊断方法（段 3）：30、31

- **30 rollout 性能优化**：判定 **REJECT**。
  - **P0 #5**。
  - **P1** §3.3 overlap 周期"下界" `max(…)+T_fence+T_publish` 把 fence 等待算了两次：`train_async.py::train` 本来就在 max 里等 future。应为 `≈max(T_rollout+T_data, T_train+T_save)+T_publish`，并补一条数值时间线。
  - **P1** §6.2 和 §7 把 `--update-weights-interval` 列为降低发布成本的旋钮，没说它对同步训练无效：`train.py` 第 85 行每轮都调用 `update_weights()`，只有 `train_async.py` 第 66 行读取该参数。这与 16 的误读表冲突。
  - **P1** 容量模型把 RM 当独立服务站。实际上规则 RM 在同一个 `AsyncLoopThread` 事件循环里同步执行，会阻塞整个循环；远程 RM 的连接上限写死为 64。
  - **P2** §11.3"rollout 步数由显式参数给出"已过时（基线已有 `--num-epoch`）；§11.2 单引擎恢复影响半径说过头；健康检查参数的前提 `--use-fault-tolerance` 未写。缺 `rollout/prefix_cache_hit_rate`、router 均衡默认值、`tools/analyze_profile.py` 与 profiling 文档的链接。
- **31 后训练稳定性**：判定 **REJECT**。
  - **P0 #1**（第 98 行）。
  - **P1** 判别规则所依赖的指标可能结构性为零：
    - 单步且可复用 logprob 时，`train/ppo_kl`、`train/pg_clipfrac` 恒为 0；
    - `metric_utils.py::has_repetition` 只在 `len(text) > 10000` 时生效，短响应的 `rollout/repetition_frac` 恒为 0；
    - 开零方差过滤后，`zero_std/count_*` 只统计入选组。
    - §4.1 和 §9 的联读示例因此失效。
  - **P1** 缺训练侧 `rollout/*` 汇总键（raw_reward、advantages、kl、entropy）与 `passrate/*`。`--no-check-for-nan-in-loss-and-grad` 会静默跳过训练步且没有计数，官方 `qa.md` 却推荐它，页面没写。
  - **P2** 页头"最近更新"写核验过程；`_0.0` 键名示例只对 float reward 成立。

## 8. 遗漏内容（覆盖度）

以下"无 owner"指没有任何页面讲清机制；"只提及"指出现了名字但缺入口、机制、变体或边界。

| 能力 / 场景 | 基线证据 | 现状 | 建议 | 重要度 |
|---|---|---|---|---|
| PD 分离（Mooncake、router PD 模式、bootstrap、外部 PD） | `docs/en/advanced/pd-disaggregation.md`；`sglang_engine.py::_compute_server_args`；`_start_router(has_pd_disaggregation)`；`sglang.patch` 的 PD abort/retract；`tests/test_{glm4.7_30B_A3B,qwen3.6_35B_A3B}_pd_mooncake.py`、`test_qwen3_4B_external_pd.py` | 只提及（11 端口、13 卡数、19 发现、30 调参） | 13 扩成 owner，11/19/30 改为链接；或交 planning 定归属 | 高 |
| 依赖补丁合同与镜像闸门 | Dockerfile 在 SGLang `v0.5.15.post1` 上依次应用 5 个 SGLang 补丁（sglang、top_p、release_hicache、pull_weights、deterministic）和 2 个 Megatron 补丁（megatron、megatron-sglang-aligned）；`ENABLE_SGLANG_PATCH` | 零散（16 有 pull_weights，17 有 top_p，22 有 QAT）；hicache、deterministic、aligned、`/post_process_weights` 全域缺失 | 新增"部署栈、平台与补丁"页，给出"补丁 → 依赖它的特性"矩阵，16/17/19/21/22 链入 | 高 |
| GLM-5 DSA 训练（sparse MLA、indexer、跨层 index sharing、tilelang 前后向、PP 切分约束） | `slime_plugins/models/glm5/glm5.py` 与 `ops/*`；`tests/test_glm5_indexer_*`；`--freeze-indexer` | 只提及（14 allgather、17 门禁、22 量化名） | 新专页，或并入 23 并链接 14/17 | 高 |
| Megatron 补丁中的 MTP 梯度隔离、`mtp_kwargs`、loss_mask 随 roll 移位 | `docker/patch/latest/megatron.patch` | 全库缺失 | 21（链接 14） | 高 |
| 平台：ROCm/AMD、Ascend NPU、GB10（基线后还有 MUSA） | `Dockerfile.rocm*`、`amd_patch`、`amd_tutorial.md`；`npu_patch`（钉 slime v0.2.2）；`Dockerfile.gb10`、`NOTES_GB10.md` | 基本缺失（11 提到 ROCm 跳过 NUMA 绑定，22 提到 NPU patch） | 并入上面的部署栈页；NPU 标为旧版、非主线 | 中高 |
| 奖励打分器语义 | `rollout/rm_hub/*`；`tests/test_rm_*` | 只提及（13 讲分派链） | 13 加打分器表：deepscaler 没有 `</think>` 时返回 0；dapo 返回 dict，须配 `--reward-key score`；ifbench 在 import 时 git clone 并 pip install | 中高 |
| 指标落点与指标目录 | 17 个日志/指标 flag（W&B、TensorBoard、自定义日志钩子、`--log-multi-turn`、`--memory-snapshot-dir`）；`--wandb-always-use-train-step` 会改 x 轴；rollout/、train/、perf/、eval/ 四类前缀 | 缺失 / 分散 | 18 加"指标落点与 x 轴"小节，并按前缀建索引（只链接 owner） | 中 |
| 参数冻结 | `freeze_model_params`（两个名单互斥）、`--freeze-indexer`；`tests/test_model_provider_freeze.py` | 缺失 | 14 或 23 | 中 |
| checkpoint 导入 / 导出工具 | `tools/convert_hf_to_fp8.py`（`--strategy channel`、`--scale-fmt ue8m0`）、`convert_hf_to_int4.py`（GPTQ W4A16/W8A16）、`fp8_cast_bf16.py`、`convert_k2_thinking_int4_to_bf16.py`、`convert_torch_dist_to_hf(_parallel).py`；`--save-hf` 必须带 `{rollout_id}` | 导出流程缺失 | 02 加导入/导出路线表；量化转换细节归 22 | 中 |
| FAQ 失败模式 | `docs/*/get_started/qa.md` 共 13 条 | 部分在 02 §7 | 补进 02 §7；IMA 与 Ray 调试加到 31 | 中 |
| 多节点大 MoE 配方 | `scripts/run-glm5.2-744B-A40B.sh` 的 HOSTFILE 与 IB/NVSHMEM 变量；DeepEP buffer 约束 | 缺失（02 只有单机 glm4-9B） | 部署栈页或 02 | 中 |
| 模型支持矩阵 | 39 个 `scripts/models` preset、23 个 `run-*.sh`、r3 门禁测试 | 23 只有转换注册表 | 23 §4.4 加 preset、plugin、CI 门禁三列 | 中 |
| 异步与 fully-async 的 checkpoint 切口 | 见 §4 页 10 | 缺失 | 10 或 18 | 高 |
| 外部 PD 与 `/pull_weights` 的依赖 | `test_qwen3_4B_external_pd.py`（docstring 称 delta+disk 是"唯一真正可用的同步路径"） | 缺失 | 19（链接 16） | 中高 |
| rollout buffer 配置 | `--rollout-buffer-url`、`--fetch-trajectory-retry-times`（默认 -1，无限重试）、`--rollout-task-type`、`base_generator.py` | 19 只讲机制 | 19 §3.4 加配置行 | 低 |
| 评估 flag 与同级 eval 函数 | 9 个 `--eval-*`；forge 与 buffer 的 eval 行为 | 27 讲了语义但未点名 | 27 §5 | 低 |
| 小项 | `--use-rollout-entropy`、`--opd-teacher-ckpt-step`、`--qwen-gdn-backend`；critic 输出层 reinit、value 不乘温度 | 缺失 | 分别归 14、20、23、14 | 低 |

**应明确排除并登记的**：

- 无读取方的 `--http-proxy`、`--min-batch-collection-ratio`。
- 注释伪影 `--foo-bar` 类。
- 仓内无引用的 `learnable_softmax_attention.py`、`flash_dot_product_attention.py`。
- 已损坏的 `tools/convert_to_hf.py`：导入不存在的 `megatron_utils.update_weight_utils`。
- 测试 harness `utils/external_utils/*`。
- 基线后 `c403335d` 已删除的 grad-coalesce 补丁：只需在 14 的漂移注记中提一句。

**独立枚举轴**：36 个未出现 flag 的完整分类见 `coordinator_mechanical_checks.md` §5。本域没有 `docs/coverage/slime.yaml`，建议建立 ledger，把 218 个 flag 逐一映射到 owner 或排除类（dead、artifact、pass-through），接入 `check_coverage.py`，否则这次对账无法复跑。

## 9. 全域系统性问题

### 9.1 两代页面模板并存

- **11–18**：2026-09-10/11 按 feature 画像重写。有数据驱动 SVG、页面数值测试和 `path::symbol` 阅读路线，零行号链接。
- **10、19–24、30、31**：保留旧版因果五拍结构。只有 Mermaid，每页 37–75 条逐句行号永久链接，没有集中的阅读路线。
- **26–28**：9 月 10 日新增的读者优先结构。

algorithm-replay 失败的 10 页（10、13、20、21、22、23、24、26、28、30）中，9 页属于旧模板或新增结构。主要原因是 timing、layout、mask 等触发条件出现了，却没有用同一个最小例子做原理图。另有 4 页（22 FP8 分块、23 GLU/QKV 布局、26 mRoPE 网格、28 token mask）用 Mermaid 或文本表示二维网格，按 `drawing-wiki-figures` 应改用生成 SVG。

### 9.2 依赖补丁边界系统性缺失

slime 镜像的实际行为是"上游 SGLang/Megatron + 仓内补丁"。补丁在基线可读，却常被当作"上游行为"或"本机无法核实"。P0 #2（top-k）和 #3（MTP 隔离）都出自这里。同类问题还有：

- 16：IPC 同步、`/weight_version` 改写与镜像闸门；
- 17：GLM-5 对齐栈与 skip 语义；
- 19：`/pull_weights`；
- 22：INT4 fake-QAT；
- 13：release_hicache 与 PD abort。

建议补一张统一的"补丁 → 特性"交接表（见 §8 第 2 行），各页在依赖边界处链入。

### 9.3 跨页矛盾

| 主题 | 冲突页 | 以源码为准 |
|---|---|---|
| `kl_coef` 作用范围 | 15、14、31 | 只对 ppo 与两种 REINFORCE++ 生效（P0 #1） |
| ABORTED 回队是否可达 | 10 与 13 | 取决于 SGLang pause 是否中止在途请求（依赖侧），两页应统一为条件表述 |
| SFT 入口 | 01 与 28 | recipe 用 `train_async.py` |
| `--update-weights-interval` | 30 与 01、10、16 | 只对 `train_async.py` 生效 |
| external 派生 `rollout_num_gpus` | 02 与 19 | external 也会派生 |
| worker type 是否含 `encoder` | 01 与 19 | 基线已含 `encoder` |
| EPD 是否 vime 新增 | 25 与 slime 基线 | slime 已有 |
| debug 模式是否创建 trainer | 11 与 14 | 创建，`init` 早退 |
| 一致性模型层数 | 课程页、25 的 Related Pages 写"四层"，17 为 L0–L5 | 六层 |

### 9.4 与知识库其余部分几乎不相连

23 篇正文指向 slime 以外 wiki 的链接只有 1 条（17 → `07_training_reliability/10_determinism_and_numerical_reliability_analysis`）。相关 owner 页已经存在，却没有被链接：

- **Megatron-LM 域 41 页**：CP、分布式优化器、分布式 checkpoint、作业韧性、RL 训推一致性、RL runtime、logits 蒸馏、配置面。注意其基线为 `85902ef5`，slime 镜像钉 `1dcf0daf`，是祖先版本，链接时需注明版本差。
- **推理框架域**：`sglang/`、`speculative_decoding/`、`mooncake_analysis`、`vllm/17_vllm_quantization_analysis`。
- **后训练理论域**：PPO、GRPO、GSPO、OPD、staleness、reward hacking。
- **后训练框架根目录**：OPD infra、OPD 对照、RL infra 效率、DORA，以及 verl 的迭代、权重发布、agent loop、checkpoint 恢复页。
- **训练可靠性域**：容错恢复、训练动力学、batch invariance。

这违反"链接到读者需要的前置或深入页"的要求，也使 slime 页的依赖边界处无处可去。反方向同样缺反链，见 §10。

### 9.5 页头、锚点与引用写法

- 25 页头不合规（§6）。
- 11–18 的"主题"写成机制摘要并预告结论，例如 15"逐项抵消""只改分子"，16"每条都在 pause/flush 与 continue 之间提交"。
- 02、10、19、20、24、30、31 的"最近更新"和 21 的"适用范围"写的是核验过程，应移到 changelog。
- 46 个 `x.py::symbol` 锚点只写裸文件名，有 2–29 个同名候选；页内多处用"见 11""第 12/15 页"这类纯数字引用。

### 9.6 图形渲染检查（协调者）

9 张 SVG 经无头 Chrome 渲染、目视和 `getBBox` 检查，没有文字互相重叠，有 1 处越出画布：`slime_sample_data_contract.svg` 面板 3 的右侧注释。

目视还发现以下问题：

- `slime_rollout_admission_timeline.svg`：图例行横穿到相邻面板，页脚压在面板底边上。
- `slime_megatron_train_step.svg`：面板 A 橙框末行压线。
- `slime_weight_sync_planes.svg`（1484 px，221 个文字元素）以及 14、15 的图：面板内有多段多行说明，按规范应缩短注释或拆图。

13 缺 streaming 泳道，17 图① top-k 图注错误（P0 #2），16 图③"拷贝可能仍在排队"未标补丁闸门。各图的数值回放均与生成器模型一致。

## 10. 跨域一致性与索引

- **`30_rl_framework_comparison` §7 两处说过头**：
  - "checkpoint 队列状态由 DataSource save/load 保存"：`RolloutDataSource.save` 只写游标与 metadata，partial 和 fully-async 队列都不进 checkpoint。
  - "engine 端比对版本号"：只在 full+disk 且开 `--ci-test` 时成立。
- **`13_opd_infra_mechanism_analysis`** 把 slime OPD 标成 `k1`/`k3`，不对。`apply_opd_kl_to_advantages` 只计算 `student_logp − teacher_logp`；k1–k3 是 `--kl-loss-type`，属于对 ref 的 KL。
- **`32_opd_framework_support_comparison`** 的 slime 条目：
  - 漏了独立 Megatron teacher server；
  - "支持异步"没有证据；
  - §3.2 与 20 §6 重复，应缩成一句加链接。
- **基线分裂**：
  - 课程页称 D01 固定在 `681b3adc`，但 `01_posttraining_frontier_map_analysis` §3 和 `01_posttraining_infra_mechanism_analysis` 实际钉的是更早的 `aaf5c20`。
  - `24_agentic_rl_algorithm_analysis` 的行号属于 aaf5c20。
- **GLM-5 理论页**（`24_glm5_agentic_rl`、`25_glm5_training_stability`）：
  - 只转述论文，对 13、18、21、22 零链接；
  - "服务器周期发心跳、自动重路由"与开源基线不同：基线是 driver 侧轮询，失败即整组标死，下次 `update_weights` 才重建。
  - 应加"论文 vs 开源基线"注记。
- **缺反链**：
  - `07_training_reliability/12` 讲优化器重置，未链 14 的 `--reset-optimizer-states`、`--use-stateless-adam`；
  - `01_posttraining_infra` §6 列了 slime 四种权重传输，未链 16；
  - `31_cuda_ascend_posttraining_stack_comparison` 完全没提 slime 的 `npu_patch`。
- **计数**：slime 为 24，三级索引一致。后训练框架目录 `find` 为 53，`wiki/index.md` 写 52（DORA 页未回写），属域外顺带发现。

## 11. 基线漂移

`681b3adc → 4c193f1f` 共 26 个提交，含 v0.3.2。基线固定本身不是缺陷；下表列出升级基线时需要改的页。详见 `group_H_coverage.md` §4。

| 提交 | 影响 |
|---|---|
| `d8ff51c4`（拆分 rollout.py）、`daebd20b`（删除 rollout_validation.py） | 11、13、18、02 中 `rollout.py::ServerGroup/RolloutServer/_start_router/…` 的锚点需要迁移 |
| `624b824a`（新建 `slime/observability/`） | 12、14、15、17、18、30、31 的指标与 trace 锚点 |
| `7fc5715c`（删死代码） | 18 的 `--profile-target`；16、21 的 `_named_params_and_buffers_*` |
| `7e4ac3be` | 11 §2.2"由 `MegatronTrainRayActor.init` 选 updater"未加注（16 已加注） |
| `a37dd90b`、`8f20503f`、`d8ad1b57`（测试删除或移动） | 15、17、28 引用的测试 |
| `4c1ab402`（外部流式 rollout） | 13 §5"只支持累计 SSE"失效；17 重放边界、19 external 路径受影响 |
| `4c193f1f` | 24 §6.1 abort 语义 |
| `a0d6d26a` | 01 §5.4 eval-only 成本 |
| `16c15fc2`、`41014d1f` | 13、14、17、18 的配置与约束行 |
| `2fa9a442`、`876cd89b`、`e593fa0a` | 22 的"INT4 kernel 仅 CUDA"与 UE8M0；11 的设备绑定 |

## 12. 机械门禁、工具与配置

| 检查 | 结果 |
|---|---|
| `python tools/check_links.py --strict` | 453 页：broken、ambiguous、bare_index、stale_section、orphans 均为 0 |
| `check_math.py --strict <slime>` | 24 文件，0 错误 0 警告 |
| `check_markdown.py --strict <slime>` | 24 文件，0 错误 0 警告 |
| `check_assets.py --strict <slime>` | 24 文件，0 错误 0 警告 |
| `check_locators.py --dir <slime>` | 只 audit 15 条，errors 0、warnings 2（vime 不在 watchlist）、env 13（slime 无本地 checkout）。**门禁对 slime 实际失效** |
| 协调者锚点脚本 | 557 条 slime 行号链接全部命中基线文件、区间合法且 ≤100 行；381 个 `path::symbol` 可解析；46 个裸文件名锚点有歧义；2 个未找到的符号分属基线后提交与 vime，不是错误 |
| `node --test tools/figs/svg/lib/slime_*.test.mjs` | 41/41 通过；测试读取页面正文；生成器与已跟踪 SVG 一致 |
| 内容保留（`ea68986` → HEAD） | 无概念级丢失 |

工具与配置缺口：

1. `docs/radar/watchlist.yaml` 中 slime 的 `checkout: ../slime` 解析为不存在的 `/Users/suhaibo/96-knowledge/slime`。应按同文件 vllm 条目的写法改为 `../../97-llm/slime`（该 checkout 含 `681b3adc`）。否则 locator 与 radar 都算不出结果。同文件的 `../sglang`、`../verl` 也解析到不存在的目录（本机 `/Users/suhaibo/97-llm/` 下也没有这两个 checkout），属 slime 域外的顺带发现。
2. `vllm-project/vime` 没有 watchlist 条目，本机也没有 checkout。
3. 没有 `docs/coverage/slime.yaml`，§8 的 flag 对账无法复跑。
4. 本机没有 SGLang 源码，16、21、23 的 24 条 SGLang 行号链接长期无法复核。

说明：机械门禁全绿，而 P0、P1 仍然存在。门禁只保证链接、公式、资源和图测试可用，保证不了事实正确、依赖边界和覆盖完整。11–18 在 9 月 10–11 日经独立复审 PASS 后，本轮仍发现 14、15（KL 轴）和 17（top-k）的 P0。

## 13. 修复路由建议

1. **第一波：P0 页内修正**。不改页面边界，可并发：15/14/31 的 KL 轴按估计器拆开；17 的 top-k 两种情形与图①图注；21 的 MTP 测试结论与补丁依赖；24 限定为流式；30 决策矩阵改用已有 perf 键。按规则由原写作者修改，另派未参与写作者复审。
2. **第二波：页内 P1**。
   - 02 校验时机与失败定位；01 场景清单对账与 SFT 入口。
   - 10 异步 checkpoint 切口；13 streaming 泳道与 ABORTED 条件表述（与 10 统一）。
   - 14 allgather 边界；16 补丁闸门；17 GLM-5 交接表。
   - 20 学生项来源轴与推导；25 承袭/新增列与页头。
   - 30 周期公式、RM 事件循环、interval 前提；31 指标出现条件。
3. **第三波：旧模板页补原理图、成本账和变体枚举**：10、20、21、22、23、24、26、28、30。二维网格类（22、23、26、28）改用生成 SVG 并配数值测试。顺带把逐句行号链接收敛为集中的阅读路线。
4. **交 `planning-codebase-analysis`**（涉及页面合同或归属）：
   - PD 分离的 owner；
   - 是否新增"部署栈、平台与补丁"页；
   - GLM-5 DSA 训练是否独立成页；
   - 25 迁到父域、并入对照页或保留；
   - 01 与 10–18、02 的重复和命令归属；
   - 建立 `docs/coverage/slime.yaml`。
5. **收尾**：
   - 修 watchlist 路径并补 vime 条目；
   - 统一页头（主题不预告论证，核验过程移到 changelog）；
   - 裸文件名锚点补全路径；
   - 补 §9.4 的跨域出链和 §10 的反链；修正 `30_rl_framework_comparison`、`13_opd_infra`、`32_opd` 的 slime 描述与课程页的"四层"；
   - index 的 P2；修 3 张 SVG 的越界与压线；
   - 决定是否整体升级到 v0.3.2 之后的基线（见 §11）。
   - 每波修改后跑 T0 四项门禁；触及仍带行号链接的页，先修好 watchlist 再跑 `check_locators`。

本轮只做静态读码：没有启动 slime、SGLang、Megatron、Ray 或 GPU 任务，没有运行 pytest（避免在冻结 checkout 中写入缓存），没有修改 wiki 页面，也没有提交或推送。
