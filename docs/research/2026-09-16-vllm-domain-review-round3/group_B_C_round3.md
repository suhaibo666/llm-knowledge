# 第三轮独立复核（Group B/C）：02 架构 / 06 Engine / 07 Scheduler / 08 KV Cache

- 评审人：独立 reviewer（非作者）。**只读**：未修改 `wiki/`、`docs/`、`skills/` 或源码 checkout。
- 冻结基线：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`，checkout `/Users/suhaibo/97-llm/vllm`，全程未移动 HEAD（`git rev-parse HEAD` 已核对）。
- 被评对象：工作区未提交改动。`git diff --numstat`：02 `+100/−46`、06 `+66/−41`、07 `+102/−29`、08 `+187/−8`。
- 方法：先读 round-2 判定（`2026-09-16-vllm-domain-review-recheck.md` 与 `group_B_recheck.md` / `group_C_recheck.md`），再对每条 open item 回**源码**核对（不以新 prose 为证），并把四页 diff 中的新增文字当新稿逐条核验；所有新数值例重放。
- 机械门禁（本机 `.venv` py3.13，本轮实跑）：`check_links --strict` 453 页 **broken/ambiguous/bare_index/stale_section/orphans 全 0**；`check_math --changed --strict`、`check_markdown --changed --strict`、`check_assets --changed --strict` 各 59 文件 **0 error 0 warning**。scoped mkdocs build 由协调者复核通过（30 页，四类 0），本报告不重复。

---

## 0. 结论行（verdict rows）

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 02_vllm_architecture_overview_analysis | pass | pass | pass | timing, layout | pass（本页无新数值例） | 26/30 新增锚点一致 | **REJECT** | `architecture: FAIL item 5`、`FAIL item 7`（两者均为**新**原因，round-2 的旧原因已修）；item 1/2/3/4/6/8 pass。新增 P1 ×3、P2 ×4 |
| 06_vllm_engine_architecture_analysis | pass | pass | pass | timing | pass（默认 R 重放、fence 例成立） | 8/8 一致 | **PASS** | `feature: pass`（async 解析 → Scheduler 类 → 容量 → step 入口 选择轴）。无 P0/P1；新增 P2 ×1，遗留 P2 ×3 |
| 07_vllm_scheduler_analysis | pass | pass | pass | timing, transform | pass（watermark 200/60000 换算与 2+2>3 拒绝可replay） | 11/12 一致 | **条件通过（1 P1）** | `feature: pass §8.1.1–§8.1.4`；新增 P1 ×1（§8.4 delay 归属），P2 ×1 |
| 08_vllm_kv_cache_management_analysis | pass | pass | pass | layout, transform, timing | pass（21/20/64 与 cap 公式逐符号重放成立） | 9/9 一致 | **PASS** | round-2 三条跨页 P1 全部 FIXED；无 P0/P1；新增 P2 ×2 |

- **02**：`architecture: FAIL item 5`（场景清单与真实入口对账）、`FAIL item 7`（调用树父子/兄弟顺序）。
- **06**：`feature: pass`（选择轴单元）。**07**：`feature: pass`（§8.1.1/§8.1.2/§8.1.3/§8.1.4 四个单元）。**08**：`feature: pass`（§2.3、§3.1.3、§3.3.1、§3.3.3、§4.5.1、§5.1.3 六个单元，本轮改写未破坏其中任何一个）。
- 新增 **P0 = 0**，新增 **P1 = 4**（02 三条、07 一条），新增 **P2 = 9**。

---

## 1. Round-2 open items 逐条状态

### 1.1 B3 = 02 的 architecture item 5 / item 8（round-2 REJECT 的直接原因）

| round-2 finding | 状态 | 证据（源码优先） |
|---|---|---|
| **F-2** `test_abort_defers_free` 写成普遍延迟释放 | **FIXED** | 02:~330 现写「`Scheduler.__init__` 只在「配了 KV connector 且它是 consumer」**并且** `max_concurrent_batches > 1` 时才把 `defer_block_free` 置 True，两条同时成立才延迟」。源码 `vllm/v1/core/sched/scheduler.py::Scheduler.__init__`：`multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1`，`if multiple_inflight_batches and kv_transfer_config.is_kv_consumer: self.defer_block_free = True`，且页面给出的理由（重叠批次可能仍在写已释放请求的 KV，而 consumer connector 的 load 与该写之间无顺序保证）与源码注释一字对应。与 06 §6.2 口径一致，矛盾解除。**残留 P2**：交回 06 的写法是 `[[06…|Engine 运行]] §6.2` 裸页链接 + 手写节号（见 P2-5）。 |
| **F-4** §5.3 把 gRPC/Rust/Omni 交给不拥有该内容的 13 | **FIXED** | 02:~296 改为「**这三项在本域没有深度 owner**：`vllm/entrypoints/grpc_server.py::serve_grpc` 的协议合同、Rust frontend 的进程模型与 Omni 的独立入口都尚未有页面展开（13 只在 §4.1 写了「Rust frontend 默认按一个多线程进程处理、显式 API count 大于一会被改为一」这一条计数规则）。覆盖缺口已提交 `planning-codebase-analysis` 裁决 owner，本页不把它们指给不拥有该内容的页面」。跨页核对：全域 `grpc` 命中 = 02(4)/05(2)/13(1)/23(2)/24(1)/index(1)，13 的唯一命中在 13:535（§4.1，`### 4.1 部署拓扑与 API 数量` 起于 523、`### 4.2` 起于 539），且 13 自己也声明「本域也尚无页面展开其协议合同」；`omni` 命中只在 02（7 次，均为本页自述）与 15/16 的无关语境。02 引用 13 的那句计数规则与 13:535 原文一致。`index.md:11` 的「已知覆盖边界」章节存在。 |
| **F-5** 14/15/16 在架构里无落点或链接 | **FIXED** | 对 14 的 wikilink 由 **0 → 3**（§2.3.3「token 选择」交接、§3.7 横切能力表新行、§6 读者表新行）；15 = 3；16 = 2；13/18/22/25/26 各 ≥2。§3.7 新增三行（采样与结构化输出、多模态执行、投机解码），§6 表新增 7 行（P2-9 同时修掉）。抽查新 §3.7 采样行的断言「grammar 校验只返回可接受前缀，不代表请求已推进」：`vllm/v1/structured_output/backend_types.py::StructuredOutputGrammar.validate_tokens` docstring 明写返回值「will be a prefix of the input tokens」且「Will not advance the FSM」，成立。 |
| **F-6** 缺 scale-out 场景（render → token-in 往返、门控） | **PARTIAL** | §5.5 新增三段，写出了 `VLLM_ENABLE_SCALE_OUT_ENDPOINTS`（三态）、专用模式优先级（render 优先于 `--tokens-only`）、显式 `0` + 专用模式 → `ValueError`、非专用模式且未开 → 打 info 后整组不注册、`force_no_detokenize=args.tokens_only`、`--tokens-only` 额外挂 `/abort_requests`（注释 "Disaggregated Everything"）——以上逐条与 `vllm/entrypoints/scale_out/factories.py::register_scale_out_api_routers` 及 `token_in_token_out/api_router.py::attach_router` 一致。**但路由挂载条件与结论写错（新 P1-2）**，另有 derender 路由集不全（P2-3）。 |
| **F-7** §5.6 缺 DP=2 的真实入口与 13 链接 | **FIXED** | §5.6 新增「DP=2 默认起几个 API server」整段。源码 `vllm/entrypoints/cli/serve.py::ServeSubcommand.cmd`：`args.api_server_count is None` 时 `is_multi_port or is_external_lb or rust_frontend_path → 1`；`is_hybrid_lb → data_parallel_size_local or 1`；`else → data_parallel_size`；随后 elastic EP 把 >1 压回 1；`args.api_server_count > 1 or rust_frontend_path → run_multi_api_server(args)`。页面四种模式的取值、"DP=2 默认 2 个 API server 走 `run_multi_api_server`"、"多个 API children 共用同一个监听 socket"（`setup_server(args, reuse_port=num_api_servers > 1)` 的 `sock` 传给 `APIServerProcessManager(sock=sock, …)`）均成立，并显式交回 13 §4.1。 |

**item 8（场景完成条件与读者交接）= pass**：14 已有落点与链接，15/16 在 §3.7 有架构分量，§6 表补齐 13/14/15/18/22/25/26。
**item 5（场景清单与真实入口对账）仍 FAIL，但原因全部是新的**：scale-out 的路由门控写反（P1-2）、Rust frontend 路径解析的报错条件写反（P1-3）。round-2 的四个旧原因（缺 scale-out、§5.3 错配、§5.6 缺入口、P2-8 清单缺口）都已消除。
**item 7（调用树）由 pass 转 FAIL**：新增的 `AsyncLLM._run_output_handler` 节点挂错父节点与顺序（P1-1）。

### 1.2 02 的 round-2 P2 状态

| # | 状态 | 证据 |
|---|---|---|
| P2-1 「无 KV 关闭 chunked prefill」归属 | **FIXED（正文）/ PARTIAL（树）** | 正文改为「**不在这个函数里**，它在 `EngineCore.__init__` 中、`get_scheduler_cls()` 之后、构造 `Scheduler(...)` 之前，按 `kv_cache_config.kv_cache_groups` 是否为空判定并打 warning」。源码 `core.py:147-155`：`Scheduler = vllm_config.scheduler_config.get_scheduler_cls()` → `if len(kv_cache_config.kv_cache_groups) == 0:` → `logger.warning("Disabling chunked prefill for model without KVCache")` → `Scheduler(...)`，位置、条件、warning 全对。同时核对留在 `_initialize_kv_caches` 的那半句（non-causal → 关 chunked prefill 与 prefix caching）：`core.py:269-283` 成立。启动树仍未加该节点（P2-9 新编号见 §3）。 |
| P2-2 `LLMEngine` 别名路径 | **FIXED** | 改为「`vllm/engine/llm_engine.py` 的 `LLMEngine` 与 `vllm/engine/async_llm_engine.py` 的 `AsyncLLMEngine` 都只是一行赋值别名」。两文件确实各只有 `LLMEngine = V1LLMEngine` / `AsyncLLMEngine = AsyncLLM`。 |
| P2-3 `EngineCoreRequest` 行「资源调度接收」 | **FIXED** | 改为经 `EngineCore.preprocess_add_request` → `Request.from_engine_core_request`（`core.py:999-1013`）；并补「Scheduler 看到的从来不是这个对象本身」。 |
| P2-4 树缺 `_run_output_handler` | **FIXED 但写错** | 见 P1-1。 |
| P2-5 §3.5 MRV2 `execute_model` 顺序 | **FIXED** | 改为 `apply_staged_writes()` → `dispatch_cg_and_sync_dp`（选图/DP）→ `prepare_inputs` / `prepare_attn`。源码 `vllm/v1/worker/gpu/model_runner.py`：1575-1580 `update_pp_decode_requests/finish_requests/free_states/add_requests/update_requests/block_tables.apply_staged_writes()`，1617 `dispatch_cg_and_sync_dp`，1646 `prepare_inputs`，1649 `prepare_attn`。顺序完全对上。 |
| P2-6 `VLLM_RUST_FRONTEND_PATH` 默认 `auto` | **FIXED 但引入新错** | 默认值确为 `"auto"`（`envs.py:166`）；但新写的报错条件写反，见 P1-3。 |
| P2-7 §5.6 树中 manager 缩进层级 | **FIXED** | 改为 `[with] launch_core_engines` 下挂 `APIServerProcessManager / RustFrontendProcessManager 在 with 体内构造`，`wait_for_completion_or_failure  已退出 with`。源码 `serve.py:329-385`：两个 manager 在 `with launch_core_engines(...)` 体内构造，`wait_for_completion_or_failure` 在 with 之后的 `try` 中调用。 |
| P2-8 §5.1/§5.6 场景清单不全 | **FIXED** | 新增三个兄弟入口段并核对文件：`examples/deployment/llm_engine_example.py`（`while test_prompts or engine.has_unfinished_requests(): … add_request … engine.step()`，与「调用方自己判断何时无未完成请求」一致）、`examples/features/torchrun/torchrun_example_offline.py`（`LLM(… distributed_executor_backend="external_launcher")`）、`examples/features/data_parallel/data_parallel_offline.py`（`from multiprocessing import Process`、逐进程写 `VLLM_DP_RANK`/`VLLM_DP_RANK_LOCAL`，四个跨机参数 `--dp-num-nodes`/`--dp-node-rank`/`--dp-master-addr`/`--dp-master-port` 全部存在）。§5.3 变体表新增「非 OpenAI 协议前端」行：`/v1/messages` 与 `/v1/messages/count_tokens`（`entrypoints/anthropic/api_router.py:51,89`）、`/v1/responses` + GET `{response_id}` + POST `{response_id}/cancel`（`openai/responses/api_router.py:48,80,110`）、`/v2/rerank`（`pooling/scoring/api_router.py:106`）全部命中。 |
| P2-9 §6「下一页」表缺 7 页 | **FIXED** | 新增 13/14/15/18/22/25/26 七行。 |
| P2-10 `run_batch` 弃用警告（可选） | **NOT_FIXED** | 本就是可选记录，不计失败。 |
| N-1 `model_executed` 语义 | **NOT_FIXED** | 见 P2-4。 |
| N-2 图 3 下图 `par` 会被读成进程级并发 | **PARTIAL** | 图后新增「并列区域强调 Engine 和设备侧可以重叠，不表示 GPU 时间比例」，§2.4 源码路线也写明「`non_block=True` 并不把整个 worker 调用搬到后台线程」。仍未点名 TP=1 单进程内联 vs TP>1 真并发的差别，但正文与图不再冲突。 |
| N-3 §2.3.3 owner 链接缺 `#4.1` 锚点 | **NOT_FIXED** | 见 P2-5。 |

**02 计数**：FIXED 12 / PARTIAL 2 / NOT_FIXED 3（其中 1 条可选）。

### 1.3 06 的 round-2 P2 状态

| # | 状态 | 证据 |
|---|---|---|
| P2-11 fence 取 `sched_step_seq` | **FIXED** | §6.2 改为「用 `last_sched_seq <= processed_step_seq` 判断…**要注意入队时写的 fence 值是 `self.sched_step_seq`**…CoW 释放走 `_free_cow_retained_blocks(..., fence_seq)` 这条另行传入 fence 的路径时就会分开」。源码 `scheduler.py:2557-2581`：gate 正是 `request.last_sched_seq <= self.processed_step_seq`，入队 `self.deferred_frees.append((self.sched_step_seq, blocks))`，CoW 另有 `_free_cow_retained_blocks(blocks, fence_seq)`（调用点 :1351 传 `self.sched_step_seq + 1`）。逐点成立。 |
| P2-12 `wave_complete` 发送者条件 | **FIXED** | §7 改为「发送条件是 `dp_rank == 0 or not has_coordinator`，且 `client_index` 取 `-1 if has_coordinator else 0`」，并把 wave 递增/step 归零移出该条件。源码 `core.py:2249-2266` 逐字一致（含「offline spmd case」注释语义）。 |
| P2-13 R 未走到第二个 token 与完成 | **PARTIAL（同 round-2）** | §4.2 仍只写「第二个输出要等 S2 自己的 future 被消费才归并；R 的用户可见完成仍按 §2.3 处理」，未用 R 收尾 `finish_reason=length` / `_free_request`。 |
| P2-14 图 2 把 F1 与 exec future 混在一起 | **FIXED 但引入新 P2** | 图 2 已把 `X1 execute future（不取值）` 与 `F1 sampling future` 分开画，并新增一段解释两者角色。但图/spec 把队列项写成二元组，与源码三元组冲突，见 P2-1。 |
| P2-15 术语（归并/对账） | **FIXED** | §2.2 新增「（术语对齐：本页说的"归并结果"与 Scheduler 页说的"结果对账"是同一个操作——`update_from_output()` 按原计划把返回结果落回请求状态。）」 |
| N-5 §2 scoping 句覆盖过宽 | **NOT_FIXED** | §2 开头仍以「本节先固定 `async_scheduling=False`、PP=1…」笼罩整节，§2.1/§2.3 未加「与是否异步调度无关」的限定。 |

**06 计数**：FIXED 4 / PARTIAL 1 / NOT_FIXED 2（含 N-5）。

### 1.4 07 / 08 的 round-2 P2 与跨页 P1 状态

| # | 状态 | 证据 |
|---|---|---|
| 07 P2-1 `next_decode_eligible_step` 缺 `use_v2_model_runner` | **FIXED** | §5.1 改为「**且仅当 `use_v2_model_runner` 为真时**…MRV1 下这个字段不被写入，因此这道节拍门在 MRV1 路径上不生效」。源码 `async_scheduler.py::AsyncScheduler._update_after_schedule` 的 `if self.use_v2_model_runner:` 分支内才写该字段。 |
| 07 P2-2 DP 节流锚点不精确 | **FIXED** | 路线 3 现同时给出基类恒 False 与 `DPEngineCoreProc._should_throttle_prefills`（`prefill_schedule_interval > 1 且 step_counter % interval != 0`）及 `SchedulerConfig.prefill_schedule_interval`。源码 `core.py:603-606`（基类 `return False`）、`core.py:2190-2197`（`DPEngineCoreProc`，class 起于 2015）、`config/scheduler.py:174`（`prefill_schedule_interval: int = Field(default=1, ge=1)`）逐条一致。 |
| 07 P2-3 watermark 是比例、缺换算 | **FIXED** | §7 新增「**它是比例不是块数**：`KVCacheManager.__init__` 以 `watermark_blocks = int(watermark * kv_cache_config.num_blocks)` 换算…同一个 `watermark=0.01` 在 200 块的池上是 2 块、在 60,000 块的池上是 600 块」。源码 `kv_cache_manager.py:174` 一致；算例 0.01×200=2、0.01×60000=600 成立；后续「需 2 块、余量 2、free=3 → 拒绝」与 `required_blocks = num_blocks_to_allocate + watermark_blocks`（:485）一致（4 > 3）。 |
| 07 P2-4 `prefill_scheduled` 语义 | **FIXED** | §5.2 新增括号：「`prefill_scheduled |= request.is_prefill_chunk` 沿 running 遍历累积，因此它是"本轮已扫过的部分"的状态，而不是对整批的预判」。源码 `scheduler.py:569` 初始化、:757 在 running 循环内累积、:1002 在 waiting 分支读 `not prefill_scheduled`，一致。 |
| 07 P2-6 abort-in-remote-wait 延迟释放是 Scheduler 规则 | **FIXED 但过度修正** | 已写出 `finish_requests()` 的 `status == WAITING_FOR_REMOTE_KVS` + `request_id not in finished_recving_kv_req_ids` 与两处 `discard`，与 `scheduler.py:2500-2508` 一致；但新加的绝对化断言写错归属，见 P1-4。 |
| 07 P2-7 encoder budget「只有 4」 | **FIXED** | 改为「本步 encoder budget **剩余** 4（是本步已被前面请求消耗后的余量，不是每步上限本身）」。 |
| 07 P2-8 终态表 resumable 分支 | **FIXED** | 改为「随后转入 `WAITING_FOR_STREAMING_REQ` 挂起等待下一段输入，收到 `_update_request_as_session` 的更新后才复位到 `WAITING`」。源码 `scheduler.py:2297-2299`（先 `_update_request_as_session(request, update)`，否则置 `WAITING_FOR_STREAMING_REQ`）、:2428-2434、:1533 一致。 |
| 07 NEW-3 `has_sync_kv_loads` 消费者写窄 | **FIXED** | §8.1.2 该行改为「**两代 runner 同一规则**：MRV2 在 `ActiveKVConnector.pre_forward`、MRV1 在 `kv_connector_model_runner_mixin.py`（`start_after_forward = not scheduler_output.has_sync_kv_loads`）各用它决定 forward 前还是 forward 后发起 load」；路线 18 已补该文件。 |
| 07 NEW-6 §8.1 原尾段被孤立 | **FIXED** | 新开 `#### 8.1.4 三张表之外：placeholders 与四元组账本的逐步算例`，把 async placeholders 记账与 21/20/3 spec 回退算例移入，并声明「16 §9 指向的正是这个算例」。 |
| 07 NEW-8 单向链接（15 §4.2 / 22） | **FIXED（07 侧）** | §8.4 encoder 释放句已补「完整规则与 `7 <= computed - 4` 的算例归 15 §4.2」；delay 段落也补了指向 22 的链接。 |
| 07 NEW-9 路线 18 两个测试只描述不点名 | **FIXED** | 路线 18 现点名 `test_scheduled_encoder_input_stats_disabled_without_iteration_logging`、`test_scheduled_encoder_input_stats_disabled_without_log_stats`。 |
| 07 P2-14 前半（16:432 入站引用漂移） | 不属本组页面 | 缺陷在 16，本轮未核（本组只负责 02/06/07/08）。 |
| **08 NEW-1**（round-2 跨页 P1 之一：`apply_admission_cap` 只影响整段准入门） | **FIXED** | §3.3.1 现写「`apply_admission_cap=True` 在基线里有**两个**调用点，不是一个：…二是 `Scheduler._request_remaining_blocks`，其结果经 `_inflight_prefill_reserved_blocks()` 汇总成异步 load 分配时使用的 `reserved_blocks`…因此**在途 prefill 的预留估算也是按截断后的 cap 算的**」。源码 `scheduler.py:2849-2861`（`_request_remaining_blocks` 传 `apply_admission_cap=True`）、:2863-2868（求和）、:1110（`reserved_blocks = self._inflight_prefill_reserved_blocks()`）逐点成立。 |
| **08 NEW-2**（显式 False + 不支持 HMA 的 connector 的真实结局） | **FIXED** | §4.5.1 现写「None 分支下是**静默自动关闭**，显式开启下是**建 connector 时启动硬失败**」，并给出 `hma_enabled = not disable_hybrid_kv_cache_manager` 与 `raise ValueError("Connector ... does not support HMA but HMA is enabled")`。源码 `vllm/distributed/kv_transfer/kv_connector/factory.py::KVConnectorFactory.create_connector:54-60` 逐字一致；§6.3 路线 14 已补 `create_connector`；§6.2 新增一行把两种结局分开。 |
| **08 X-3**（`is_mamba_prefill_checkpoint_valid()` 两份完整解释） | **FIXED（去重达成，方向与建议相反）** | 07 §5.4 删掉五条条件与 `start=0/end=100/hash=8/block=64/alignment=16` 算例，改为「归 08 §5.1.3，本页不重列，只消费它的判定结果决定要不要切分」；08 §5.1.3 保留条件清单并新增该算例（96 有效、88 因「相对起点 16 对齐」不满足而无效、未声明 alignment 一律无效）与 Kimi K3 KDA 逐层 `block_size` 陷阱。全域现只有一份。**残留 P2-8**：08 自己两处仍把「调度对齐」指回 07。 |
| 08 P2-9 §2.3「[[24…|LoRA]]」标签误导 | **FIXED** | 标签改为「运行时 LoRA resolver 与插件」。 |
| 08 P2-10 未点名 `prefix_match_unit` / `prefix_cache_retention_interval` | **FIXED 但一处措辞不准** | §5.3 新增两键：`prefix_match_unit` 的 docstring 引用（「equals to the `hash_block_size`」、默认 None、每组 `block_size` 须能整除、「只控制匹配粒度，不控制多久存一次状态」）与 `config/cache.py:100-111` 逐字一致；`prefix_cache_retention_interval`（默认 0、来自已弃用 env）与 `cache.py:61-67,157-162` 一致，但「保留时长」的说法不准，见 P2-2。 |
| 08 P2-11 §6.3 路线缺正文引用的测试 | **FIXED** | 新增第 17 条，点名 `test_internal_checkpoint_uses_partial_hash_lifecycle` 与 `test_mamba_speculative_block_relocation_requires_exclusive_ownership`。 |
| 08 P2-12 §4.6 树缺两个 drain | **FIXED** | 树已补 `kv_cache_manager.take_boundary_state_offloads` / `take_kv_cache_block_copies`，与 07 §10 树对称。 |
| 08 P2-13 「至多复用 `num_tokens−1`」重复 | **FIXED** | §5.1 改为「这条上限规则已在 §3.2 展开（含 B 的 9→8 算例），此处不重述」。 |
| 08 NEW-7 §6.2 可观测边界表未随新机制扩展 | **FIXED** | 新增 4 行：HMA 自动关闭、full 组被拒而 SWA/chunked 看似有余量、startup plan 未应用、`num_gpu_blocks_override` 生效。 |

---

## 2. §2 重复解释（round-2 唯一变差项）：现在的归属

| 概念 | 现在的唯一 owner | 推出方是否已压成链接 | 证据 |
|---|---|---|---|
| stale / drop 计数与三类迟到结果 | **07 §8.3**（五行表 + 图 7 + 计数规则） | **是**。06 §5.1 改题为「为什么结果必须按产生它的那份计划对账」，只留一段不变量 + 一句「这三类的判定表、计数规则与图归 07 §8.3，本页不再重放一遍」，**整张图 3 已删除** | 06 的图从 5 幅减为 4 幅，spec 已重编号为图1–图4，全页无残留「图5」引用（`grep 图\s*[0-9]` 仅命中四个 spec 注释） |
| 四种 finished 信号「给谁看」对照表 | **06 §6.1**（四行，含 `SchedulerOutput.finished_req_ids` 行） | **是**。07 §8.5 删掉三行表，改为「对照表归 06 §6.1，本页不再重列」，只保留 `EngineCoreOutputs.finished_requests` 的产出侧一句 | 07 §8.5 现文已核 |
| `finished_req_ids` 生产条件 + 「不能 `clear()`」 | **07 §8.1.2** | **是**。06 §6.1 删掉原地 `clear()` 那句，改为「生产侧…归 07 §8.1.2」 | 三处副本收敛为一处；`scheduler.py:1528-1531` 的 `self.finished_req_ids = set()` 与 NOTE 仍支持该表述 |
| async 解析条件轴（`async_scheduling` → 容量 → `step_fn`） | **06 §4.1** | **是**。07 §2.1 删掉整段复述，改为「上表第一行读的是**已解析后**的 `async_scheduling`…owner 都是 06 §4.1；本页只消费它的结果来选 Scheduler 类」，并保留「有 batch queue ≠ 选了 AsyncScheduler」这一句本页需要的边界 | 07 现已有指向 06 §4.1 的指针（round-2 的「全页没有指向 §4.1 的指针」已解决） |
| `is_mamba_prefill_checkpoint_valid()` 条件与算例 | **08 §5.1.3** | **是**（07 §5.4 已删并交回） | 见 §1.4 X-3 |

**保守性（conservation）核对**：06 §5.1 删除的每条有据主张都能在 07 §8.3 找到——
「preempt 把 stale 赋值为当前 in-flight、不累加」→ 07「普通 preempt 把 `num_stale_output_tokens` **赋值为**当前 in-flight，不累加」；「排空前暂缓恢复」→「waiting 看见可交付 stale 尚未排空就跳过该请求」；「AsyncScheduler 只对非 stale 扣 placeholders、只有仍 RUNNING 才提交 cache block」→「AsyncScheduler 仅在更新前状态仍为 RUNNING 时 cache 新确认的 blocks，PREEMPTED stale 不会提交到已释放的旧 KV」；「drop 份额保持 drop」→「多次抢占时，尚未排空的 drop 份额仍保持 drop」；drop 触发条件（`reset_prefix_cache(reset_running_requests=True)` 同轮恢复、`requires_kv_delivery` connector）→ 07 同段。**唯一丢失的是教学性数值重放**（两批在途 2→1→0 的逐份排空）：`num_stale_output_tokens` 现在全域只出现在 07，而 07 §8.3 只有规则、无该算例 → 记为 P2-7。

---

## 3. 新增 / 回归 findings

### P1-1 [P1｜02 §2.4 调用树] `AsyncLLM._run_output_handler` 挂错父节点与顺序

- 页面原文（树内）：
  ```
  │  └─ AsyncLLM._add_request
  │     ├─ AsyncLLM.check_admission             前端阈值检查
  │     ├─ AsyncLLM._run_output_handler         首次调用时懒启动输出处理任务
  ```
- 源码：`vllm/v1/engine/async_llm.py::AsyncLLM._add_request`（:492-512）只做 `check_admission` → `output_processor.add_request` → `await engine_core.add_request_async`，**不调用** `_run_output_handler`。调用点在 `AsyncLLM.add_request`（:465，注释「We start the output_handler on the first call to add_request()」），位于 `await self._add_request(...)`（:474/:486）之前；另一处在 `AsyncLLM.__init__`（:186-190）：`asyncio.get_running_loop()` 成功即 `self._run_output_handler()`。
- 本树代表的在线路径正好命中 `__init__` 的**急启**分支：`vllm/entrypoints/launchers/api_server/entry.py::build_async_engine_client_from_engine_args` 是 async 生成器，`AsyncLLM.from_vllm_config(...)` 在事件循环内执行，因此 output handler 在构造期就已创建，不是「首次调用时懒启动」。
- 后果：三重错误（父节点、兄弟顺序、注记的适用条件），正是 architecture item 7 要守的那条。
- 建议：把该节点提到 `AsyncLLM.add_request` 之下、`_add_request` **之前**，注记改为「已在事件循环内构造时 `__init__` 即启动；否则首次 `add_request` 懒启动（源码注释解释这是为了让 `__init__` 能在事件循环外优雅失败）」。

### P1-2 [P1｜02 §5.5 scale-out] 路由挂载条件写反，「必须两个进程」的结论因此不成立

- 页面原文：「由 `register_scale_out_api_routers` 按任务集合分别挂载：`render` 任务挂 `/v1/chat/completions/render`、`/v1/messages/render`、`/v1/completions/render` 与对应的 `…/derender`；`generate` 任务挂 `/inference/v1/generate`。」以及「因此一个完整的 render → token-in → derender 往返需要**两个进程按不同门启动**，而不是一个服务的两种用法。」
- 源码三处：
  1. `vllm/entrypoints/launchers/api_server/routers.py:52-55`：`if "generate" in supported_tasks or "render" in supported_tasks: register_scale_out_api_routers(app, supported_tasks)`。
  2. `vllm/entrypoints/scale_out/factories.py::register_scale_out_api_routers`：过了 env/专用模式门之后**无条件** `app.include_router(render_render)` 与 `app.include_router(derender_render)`；只有 `/inference/v1/generate` 被 `if "generate" in supported_tasks:` 包住。`"render" in supported_tasks` 仅用于推导 `dedicated_mode`，不挂载任何路由。
  3. `vllm/entrypoints/launchers/api_server/app_state.py:123-132`：`if "generate" in supported_tasks:` 内调用 `init_scale_out_state`，它经 `init_render_state` 建好 `serving_render`、`serving_derender`，再建 `serving_tokens` —— 所以普通 generate 服务上这三组 handler 都是活的。
- 后果：普通 `vllm serve <model>` 只要 `VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1`，**一个进程**就同时暴露 render、derender 与 `/inference/v1/generate`；页面的部署约束与门控归属都与源码相反。
- 建议：改为「门过了之后 render 与 derender 路由无条件挂载，`/inference/v1/generate` 只在 `generate` 任务存在时挂；`render` 任务的作用是使 `dedicated_mode` 成立（从而允许不设环境变量）。因此 `vllm launch render` 是**只有**渲染端点的 GPU-less 进程，而普通 generate 服务开了环境变量后可一并提供三段——分两个进程是部署选择，不是代码约束。」

### P1-3 [P1｜02 §5.3 变体表] `VLLM_RUST_FRONTEND_PATH` 的报错条件写反

- 页面原文：「`VLLM_RUST_FRONTEND_PATH` **默认即 `"auto"`**（不是未设置），由解析逻辑去找二进制，显式设了值却解析不出才报错。」
- 源码 `vllm/envs.py::_resolve_rust_cli_path`（:580-609）：`raw = os.environ.get("VLLM_RUST_FRONTEND_PATH", "auto")`；`if raw.lower() in ("auto","1","true"):` 才去找 `<pkg_dir>/vllm-rs`，**找不到就 `raise FileNotFoundError("VLLM_RUST_FRONTEND_PATH=auto but the vllm-rs binary was not found at …")`**；否则 `return raw` —— 显式路径**不做任何校验**（`vllm/v1/utils.py::RustFrontendProcessManager` 也只把 `binary_path` 原样传给子进程）。
- 后果：恰好相反——报错发生在 `auto`（或 `1`/`true`）而二进制缺失时；显式值是被无条件采信的。
- 建议：「默认 `auto`：解析时在包目录找 `vllm-rs`，找不到直接 `FileNotFoundError`；显式给路径则原样采用、解析期不校验，错路径要到拉起子进程时才暴露。未开 `VLLM_USE_RUST_FRONTEND`/`VLLM_USE_RUST_BENCH` 时返回 None，并对已设路径打 warning。」

### P1-4 [P1｜07 §8.4] 「delay 不是 connector 直接要求的」是过度修正

- 页面原文：「**delay 与否由 Scheduler 判定，不是 connector 直接要求的**：`finish_requests()` 只对 `status == WAITING_FOR_REMOTE_KVS` 的请求算 `delay_free_blocks = request_id not in self.finished_recving_kv_req_ids`……」
- 源码 `vllm/v1/core/sched/scheduler.py::_free_request`（:2516-2542）：`connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)`；EC connector 的 `request_finished(request)` 结果再 `|=`；最后 `delay_free_blocks |= connector_delay_free_blocks`，`if not delay_free_blocks: self._free_blocks(request)`。而 `_connector_finished`（:2793 起）返回的正是 `delay_free or partial_tail_delay`，其中 `delay_free` 来自 KV connector 的 `request_finished()`。
- 后果：connector 确实能直接要求延迟释放；`finish_requests()` 的 WAITING_FOR_REMOTE_KVS 判定是**另一个**叠加来源。原稿「connector 要求 delay 时」是对的，新稿把一个补充条件写成了唯一条件。
- 建议：「`_free_request` 把两处 delay 求或：connector（及 EC connector）`request_finished()` 返回的 delay 与 partial-tail 保存，以及 `finish_requests()` 对 `WAITING_FOR_REMOTE_KVS` 且尚未登记完成接收的请求额外置的 delay（同时把 id 从 `finished_recving_kv_req_ids`/`failed_recving_kv_req_ids` 中 discard）。」

### P2-1 [P2｜06 §4.2 图 2 与 spec] 队列项写成二元组，与源码三元组及本页 §4.1 冲突

- spec 原文：「入队的是(sampling future, S)二元组」；图内 `C->>Q: 入队 (F1, S1)`；图后段落「Core 拿到它只为确认提交成功」。
- 源码 `core.py:706`：`batch_queue.appendleft((future, scheduler_output, exec_future))`；队列类型标注为 `deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]]`（:210-213）。本页 §4.1 自己也写「每个队列项保存三者：结果 future、对应 SchedulerOutput、原 execute future」。
- 另：exec future 不止「确认提交成功」——采样结果为 `None` 时靠它重抛原异常（`core.py:725-730`），pooling/本轮未执行模型时它本身就是入队的 future（:685-687）；这两点本页别处已写对。
- 建议：spec 与图改为「入队 (F1, S1, X1)」或至少不断言二元组；图后句改为「X1 在正常路径不被取值，只在采样结果为 None 时用于重抛原执行异常」。

### P2-2 [P2｜08 §5.3] `prefix_cache_retention_interval` 写成「保留时长」

- 页面原文：「它管的是缓存块保留时长而不是边界粒度」。
- 源码 `vllm/config/cache.py:157-162` docstring：「Token interval between retained sliding-window and Mamba prefix-cache checkpoints. `0` retains only semantic checkpoints…」——它是**保留 checkpoint 的 token 间隔**（多久存一次），不是时间维度的保留时长；而 `prefix_match_unit` 的 docstring 恰好用「not how often states are stored」把两者对开。
- 建议：改为「它管的是**每隔多少 token 额外保留一个 SWA/Mamba checkpoint**，`0` 只保留语义 checkpoint；与匹配粒度无关」。

### P2-3 [P2｜02 §5.5] derender 路由集不全

- 页面原文：「…`/v1/completions/render` 与对应的 `…/derender`」，读作三个 render 各有一个 derender。
- 源码 `vllm/entrypoints/scale_out/derender/api_router.py` 只有两个：`/v1/chat/completions/derender`（:31）与 `/v1/completions/derender`（:81）；**没有** `/v1/messages/derender`。
- 建议：显式列出两条 derender 路由。

### P2-4 [P2｜02 §2.3.3、§2.4；06 §4.1] `model_executed` 仍被读成「本轮是否执行了模型」（round-2 N-1 未修）

- 02：「pooling 或本轮未执行模型时，队列直接保存执行 Future」；06 §4.1 同义。
- 源码 `core.py:674-687`：`model_executed = False`，**仅当** `if self.is_ec_consumer:` 才改写为 `total_num_scheduled_tokens > 0`；`is_ec_consumer` = `ec_transfer_config is None or ec_transfer_config.is_ec_consumer`（:218-221）。因此 EC producer 引擎上即使排了 token，`model_executed` 恒 False，永不采样。
- 建议：任一页补一行 EC producer 例外即可（普通部署结论不变）。

### P2-5 [P2｜02 §2.3.3/§2.4；07 §2.1/§8.5] owner 交接仍用「裸页链接 + 手写节号」

- 02 §2.3.3「继续见 [[06…|Engine 运行循环]]」（无 `#4.1`）、02 §2.4「归 [[06…|Engine 运行]] §6.2」、07 §2.1「owner 都是 [[06…|Engine 架构]] §4.1」、07 §8.5「对照表归 [[06…|Engine 架构]] §6.1」。
- 对照：11:46、12:49、19:360 都写成 `[[06_vllm_engine_architecture_analysis#4.1 …|…]]`。手写节号不被 `check_links` 的 `stale_section` 覆盖（它只看正文里紧邻的纯文本 `§N`），正是 round-2 B1 与 P2-14 那类漂移的温床。
- 建议：统一改成锚点链接（`#4.1`/`#6.1`/`#6.2` 三个标题都存在）。

### P2-6 [P2｜06 §2] scoping 句仍覆盖整节（round-2 N-5 未修）

§2.1（前端登记、MP ADD）与 §2.3（输出线程、前端可见完成）在两条分支下相同，却被「本节先固定 `async_scheduling=False`、PP=1…」笼罩。建议把适用范围限到 §2.2 及其图，或在 §2.1/§2.3 各加半句。

### P2-7 [P2｜跨 06/07] stale 排空的数值重放在去重中丢失

06 §5.1 删除时带走了唯一一处「两批在途、stale 2→1→0 逐份排空」的算例；`num_stale_output_tokens` 现在全域只出现在 07，而 07 §8.3 只给规则与流程图。按 feature 体例这条规则应当可重放。建议在 07 §8.3 末补两行算例（in-flight=2 → 首份返回 2→1 并追加 t1 → 次份 1→0 并追加 t2，期间不再扣已重置的 computed/placeholders）。

### P2-8 [P2｜08 §5.3 与 §5.1.3] Mamba checkpoint 对齐的归属句仍指回 07

08:690「Mamba checkpoint 的调度对齐和 padding 演算见 07」，08 §5.1.3 结尾「07 负责挑选可执行的 checkpoint 对齐终点」；而 07 §5.4 现在把条件与算例交给 08 §5.1.3。窄读可成立（08 管**有效性判定**、07 管**用它选切分点与 padding**），但读者在两页之间会被弹一次。建议 08 把这两句改成「候选是否有效由本页判定；用它挑切分点、算 padding 归 07 §5.4」。

### P2-9 [P2｜02 §2.2 启动树] 仍缺「无 KV → 关 chunked prefill」节点

正文已把时点钉在 `get_scheduler_cls()` 与 `Scheduler(...)` 之间，树里却只有 `SchedulerConfig.get_scheduler_cls` 与 `所选 Scheduler 类(...)` 两个相邻节点。补一行 `[kv_cache_groups 为空] SchedulerConfig.enable_chunked_prefill = False` 即可让树与正文一致。

---

## 4. 本轮新增文字的其余核验（全部成立，不计 finding）

| 新主张（页面） | 源码核对 |
|---|---|
| 02 §3.8 平台选择轴：`current_platform` 一次探测并缓存 | `vllm/platforms/__init__.py:289-315`（`_current_platform` 惰性解析并缓存） |
| 同上：`XPUWorker`/`XPUModelRunnerV2`、`CPUWorker`/`CPUModelRunner` | `v1/worker/xpu_worker.py:24`、`xpu_model_runner.py:30`、`cpu_worker.py:33`、`cpu/model_runner.py:10` |
| 同上：`get_default_ir_op_priority` 的 provider 默认序按平台 | `platforms/interface.py:1321`、`cuda.py:726`、`rocm.py:1095`、`xpu.py:481` 各自覆写 |
| 同上：`CustomOp` 落到 `forward_cuda/forward_hip/forward_xpu/forward_cpu` | `model_executor/custom_op.py:138-158` |
| 02 §5.9 LoRA 三入口：`enable_lora` 默认 False | `engine/arg_utils.py:618` `enable_lora: bool = False` |
| 同上：`VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` 才注册两个端点并 warning | `entrypoints/serve/lora/api_router.py:24-31`（`if not envs.…: return`；warning 原文「This should ONLY be used for local development!」）；`/v1/load_lora_adapter` 于 :42 |
| 02 §2.4 批队列树 6 个新节点与例外段 | `core.py:649-762` 逐行（appendleft → 有空位 `return None` → `pop` → `future.result()` → `_process_aborts_queue` → `update_from_output`；deferred 分支先对账旧结果再算 grammar 并补发 sampling；`model_output is None` → `exec_model_fut.result()` 重抛） |
| 02 §5.3 gRPC 入口 | `entrypoints/cli/serve.py:55-59`（`if getattr(args,"grpc",False): uvloop.run(serve_grpc(args))`） |
| 02 §5.5 `vllm launch render` 使 `supported_tasks` 含 `render` | `entrypoints/launchers/render/entry.py:39` `build_app(args, ("render",))`；CLI 名称见 `cli/launch.py:59` |
| 07 §8.1.3 `pending_structured_output_tokens`「本步增加前就已有 placeholders」 | `async_scheduler.py::_update_after_schedule`：`pending… |= use_structured_output and num_output_placeholders > 0` 出现在 `num_output_placeholders += …` **之前** |
| 08 §3.3.1 两条 cap 公式与 21/20/64 算例 | `kv_cache_interface.py:670-694`（chunk：`cdiv(min(chunk+I, M), B)`）、:716-748（SWA：`cdiv(min(W−1+E+I, M), B)+1`，`+1` 注释与页面「窗口起点可能落在块中间」一致）。重放 B=16、W=Q=64、E=0、M=1024、I=2×128=256：SWA `cdiv(319,16)+1=21`、chunk `cdiv(320,16)=20`、full `1024/16=64` ✓ |
| 08 §5.1 新增支持范围段（partial hit 能力门） | `kv_cache_coordinator.py:625-668`：`assert pcp_world_size == 1`；DCP>1 时 `assert isinstance(spec,(FullAttentionSpec,MambaSpec))`；`has_partial_mamba_group` 要求 `mamba_cache_mode=="align"` 且（DCP=1 时 `block_size > hash_block_size`、DCP>1 时 `>=`）；随后对 `prefix_cacheable` 且不支持 fine-grained 且 `block_size != hash_block_size` 的 manager 关闭并 warning；`UnitaryKVCacheCoordinator` 的 `assert not enable_caching or hash_block_size == self.block_size`（:523） |
| 08 §5.3 `prefix_match_unit` 的三条约束 | `config/cache.py:100-111` docstring 逐字 |
| 06 §6.2 fence 三条新断言 | 见 §1.3 P2-11 |
| 06 §7 wave_complete 两个条件 | 见 §1.3 P2-12 |

## 5. 未复核范围

- 未运行 vLLM、GPU、多进程服务、EC/PD 分离部署或故障注入；P2-4 的 EC producer 分支、P1-2 的单进程三段路由共存均为读码推断（含 `app_state`/`routers` 两处调用门）。
- 只打开了 03/11/12/13/15/16/22/index 的相关小节做跨页对账，未复审这些页面自身的 finding（属其他组）。
- 未重跑 mkdocs build（协调者已给出通过证据），未跑 `check_locators`（四页新增文字不含 `path:line` 引用）。
