# 独立复审（第二轮）Group B：02 软件架构 / 06 Engine 架构

- 评审者：同一独立 reviewer（非作者）。本轮只读，未修改 `wiki/`、`skills/`、`docs/` 或源码 checkout。
- 冻结基线：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`，commit time `2026-09-06T17:54:32-07:00`（2026-09-07 00:54 UTC）。本机 `/Users/suhaibo/97-llm/vllm` 的 `HEAD` 已核对为该提交，全程未移动 HEAD。
- 审阅对象：工作区未提交修改（`git diff`）。02 +98/−? 行、06 +52 行、index.md +16 行；另打开了 07、11、12、13、19、24、03 的相关小节做跨页对账。
- 复核方式：每条新主张都回到源码符号核对（`vllm/config/vllm.py`、`vllm/config/scheduler.py`、`vllm/v1/engine/core.py`、`vllm/v1/core/sched/{scheduler,async_scheduler}.py`、`vllm/v1/executor/{abstract,uniproc_executor,multiproc_executor}.py`、`tests/test_config.py`、`tests/v1/engine/test_engine_core.py`），未运行 GPU / 多进程 / 服务。
- 机械门禁（本轮实跑，`.venv` py3.13）：`check_links --strict` 452 页 broken/ambiguous/bare_index/stale_section/orphans 全 0；`check_markdown --changed --strict`、`check_math --changed --strict`、`check_assets --changed --strict` 各 39 文件 0 错 0 警。未跑 mkdocs build（本轮无需建站证据）。

---

## 0. 结论速览（verdict rows）

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 02_vllm_architecture_overview_analysis | pass | **pass**（原 FAIL 已修） | pass | timing, layout | pass | 12/12 新增或改写锚点全部一致 | **REJECT** | P0 已修；`architecture: FAIL item 5 / item 8`，5 条 P1（F-2、F-4、F-5、F-6、F-7）与 10 条 P2 一条未动 |
| 06_vllm_engine_architecture_analysis | pass | pass | pass | timing | pass | 8/8 新增锚点与两个新测试名全部一致 | **PASS** | `feature: pass`（选择轴单元）。剩余仅 P2：P2-11、P2-12、P2-14 未改，P2-13 部分改善 |

- **02 architecture: FAIL item 5、FAIL item 8**（item 1/2/3/4/6/7 pass）
- **06 feature: pass（`async_scheduling` → `max_concurrent_batches` → `step_fn` / Scheduler 类选择轴）**
- 新增 P0 = 0，新增 P1 = 0，新增 P2 = 5。**本轮所有新增文字均可在基线上核实**，未发现事实回归。

---

## 1. 上一轮findings的逐条状态

### 1.1 P0

| # | finding | 状态 | 证据 |
|---|---|---|---|
| F-1 | 02 §2.3.3/§2.3.4/§2.4：默认“普通生成”路径写成 `EngineCore.step` | **FIXED** | 四条建议全部落地：①§2.3.3 新段明确写出 `async_scheduling=None` → `VllmConfig.__post_init__` 解析为 True、`max_concurrent_batches=2`、选 `step_with_batch_queue`（核对 `vllm/config/vllm.py:1270-1319`、`:567-577`）；②§2.4 调用树改为 `step_with_batch_queue` 分支，节点顺序 schedule → execute_model(non_block) → get_grammar_bitmask → sample_tokens(non_block) → appendleft → 有空位 return None → pop → result → `_process_aborts_queue` → `update_from_output`，与 `core.py:649-745` 逐行一致；③§2.3 第二张时序图的“opt 需要独立采样”已删除，改成 par 提交 + opt 收取；§2.4 表 `ModelRunnerOutput / Future` 行改为“采样 Future、原计划、执行 Future 配对入队”，与 `core.py:705` 三元组一致；④§2.3 末段改为“本节条件下默认就有容量为 2 的批队列…只有 `step` 分支才先等待执行结果，并在该结果为 `None` 时同步调用采样”，与 `core.py:607-637` 一致。另新增例外段（pooling / 未执行模型 / `pending_structured_output_tokens` / 采样结果为 None 时用 exec future 重抛）全部与 `core.py:685-702、725-730` 相符 |

### 1.2 P1

| # | finding | 状态 | 证据 |
|---|---|---|---|
| F-2 | 02 §2.4 测试条目把 `test_abort_defers_free` 写成普遍延迟释放 | **NOT_FIXED** | 02:318 原文未动：“`tests/v1/core/test_deferred_block_free.py::test_abort_defers_free`：取消后，在途步骤尚未收齐时，block 不立即释放。”仍缺 gate 条件。源码 `vllm/v1/core/sched/scheduler.py::Scheduler.__init__` 仍要求 KV consumer connector 且 `max_concurrent_batches > 1`；06 §6.2 本轮仍明确写着“不能把‘凡是async都延迟回收’写成通用规则”，两页仍互相矛盾 |
| F-3 | 06 §2/§2.2/§4.1 未建立默认路径与同级选择轴 | **FIXED** | §2 标题改为“同步 core step”，新增 scoping 段落声明 `async_scheduling=False`、PP=1、未自定义 `scheduler_cls`，并明说“不是当前普通生成模型的缺省执行分支”；§4.1 新增两张表（解析条件表 + 容量/Scheduler/step_fn 表）并自我声明 owner；§4.2 新增“默认 R”重放表；§2.3 图后新增一句区分同步图与默认分支；§8 源码路线拆成“async解析/Scheduler类/容量/step入口”与“队列配对/默认R/延期sampling”两行。逐项核对见下 §2 |
| F-4 | 02 §5.3 把 gRPC/Rust/Omni 交给不拥有该内容的 13 | **NOT_FIXED** | 02 §5.3 原文未动：“客户端操作与完整部署参数由 [[13_vllm_serving_control_plane_analysis|Serving 控制面]] 和对应外部项目负责”。13 本轮 +283 行后，`grep -i "grpc\|omni"` 在 13 中命中 0 次，`rust` 仅命中 13:535（“Rust frontend 默认按一个多线程进程处理，显式 API count 大于一会被改为一”）与 13:719 的验证范围声明。`vllm/entrypoints/grpc_server.py::serve_grpc` 在全域仍无 owner 页 |
| F-5 | 02 E2E：14/15/16 在架构中没有落点或链接 | **NOT_FIXED** | 重新统计 02 的 wikilink：指向 **14 = 0 次**（与上轮相同）；15 只在 §5.3 变体表 1 次；16 只在 §6 读者表 1 次。§2.3.3 出现“token 选择”但未链接 14；§3.3/§3.5/§3.7 未新增横切行；§6 表也未补 14 |
| F-6 | 02 §5.5 / 场景清单缺 render → token-in 往返与 scale-out 门控 | **NOT_FIXED** | 02 全文 `grep -i "scale.out\|SCALE_OUT\|tokens-only"` 命中 0。§5.5 末段仍只写“下一步模型计算仍需独立执行服务”，未写是哪个服务、`VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1` 或 `--tokens-only` 门控。全域中 scale_out 仅 03:308 的源码路线锚点（`ServingRender`/`ServingTokens`/`ServingDerender`），仍无正文 owner |
| F-7 | 02 §5.6 DP=2 模板的真实入口与 13 链接缺失 | **NOT_FIXED** | 02 §5.6 未改：模板后无“DP=2 默认 2 个 API server、走 `run_multi_api_server`”的说明，本节链接仍只有 22 与 18。源码 `ServeSubcommand.cmd` 的 internal LB 分支仍把 `api_server_count` 取为 `data_parallel_size`；13 §4.1 写对了但 02 未接 |

### 1.3 P2

| # | finding | 状态 | 证据 |
|---|---|---|---|
| P2-1 | 02 §2.2 把“无 KV 关闭 chunked prefill”归给 `_initialize_kv_caches` | **NOT_FIXED** | 02:100 原句未动。源码在 `EngineCore.__init__`（`core.py:151-156`，位于 `get_scheduler_cls` 之后、`Scheduler(...)` 之前）。启动树也未加该节点 |
| P2-2 | 02 §3.8 `vllm.engine.LLMEngine` 别名路径 | **NOT_FIXED** | 02:526 未动 |
| P2-3 | 02 §2.4 表 `EngineCoreRequest` 行“资源调度接收” | **NOT_FIXED** | 该行未动（`preprocess_add_request` → `Request.from_engine_core_request` 才是实际交接） |
| P2-4 | 02 §2.4 树缺 `AsyncLLM.add_request` 下的 `_run_output_handler` | **NOT_FIXED** | 树中仍只有 `process_inputs` / `_add_request` 两个子节点 |
| P2-5 | 02 §3.5 MRV2 `execute_model` 顺序措辞 | **NOT_FIXED** | 02:435 未动（源码顺序为 `apply_staged_writes` → `dispatch_cg_and_sync_dp` → `prepare_inputs`/`prepare_attn`） |
| P2-6 | 02 §5.3 `VLLM_RUST_FRONTEND_PATH` 默认 `auto` | **NOT_FIXED** | §5.3 变体表未动 |
| P2-7 | 02 §5.6 树中 manager 缩进层级 | **NOT_FIXED** | 树未动 |
| P2-8 | 02 §5.1/§5.6 场景清单不全（`llm_engine_example.py`、`data_parallel_offline.py`、`torchrun_example_offline.py`、Anthropic/Cohere/Responses 路由） | **NOT_FIXED** | §5 整节在本轮 diff 中零改动 |
| P2-9 | 02 §6 “下一页”表缺 13/14/15/18/22/25/26 | **NOT_FIXED** | §6 表未动 |
| P2-10 | 02 §3.8 可选：`run_batch.py` 弃用警告写 `vllm run_batch` | **NOT_FIXED** | 未加（本就是可选记录） |
| P2-11 | 06 §6.2 与图 4：fence 取 `sched_step_seq` 而非 `last_sched_seq` | **NOT_FIXED** | 06 §6.2 仍写“设 S1、S2 都曾安排 R，`last_sched_seq=2`…保留到fence 2”；源码 `_free_request_blocks` 追加的是 `(self.sched_step_seq, blocks)` |
| P2-12 | 06 §7 `wave_complete` 发送者条件 | **NOT_FIXED** | 06:286 仍写“全局空闲时发 `wave_complete`”，未补 `dp_rank == 0 or not has_coordinator` |
| P2-13 | 06 R 没有走到第二个 token 与完成 | **PARTIAL** | §4.2 新增“第二个输出要等 S2 自己的 future 被消费才归并；R 的用户可见完成仍按 §2.3 处理”，把第二个 token 的归并位置交代清楚；但仍未用 R 收尾 `finish_reason=length` / `_free_request` / 前端 finished |
| P2-14 | 06 图 2 把 F1 与 exec future 混在一起 | **NOT_FIXED** | 图 2 未动，仍是 `E-->>C: F1` 后 `C->>Q: 保存F1 + S1 + execute future`，未把 F1 标成 sampling future |

**计数**：02 = FIXED 1 / PARTIAL 0 / NOT_FIXED 15（5 条 P1 + 10 条 P2）。06 = FIXED 1 / PARTIAL 1 / NOT_FIXED 3。

---

## 2. 新增文字的逐项核对（重点：有无回归）

以下每条都在基线上验证过，**全部成立**，故不计入 finding：

| 新主张（页面） | 源码核对 |
|---|---|
| `SchedulerConfig.async_scheduling` 原始缺省 `None`（06 §4.1、02 §2.3.3） | `vllm/config/scheduler.py:179` `async_scheduling: bool \| None = None` |
| `None` + 普通生成 + 无不兼容项 → True（两页） | `vllm/config/vllm.py:1270-1319`，`else: async_scheduling = True` |
| `None` + pooling → False，且理由是“当前实现性能负收益”，显式 True 不被 pooling 拒绝（06 §4.1） | `vllm.py:1272-1281` 用 `logger.debug`，pooling 分支只在 `elif ... is None` 支路；`vllm.py:1239-1269` 的显式分支不含 pooling |
| `None` + 不兼容 spec / `disable_padded_drafter_batch` / executor 不支持 / ROCm+DeepEP HT+DBO → False 并 warning（06 §4.1） | `vllm.py:1282-1317`，四支均 `logger.warning_once` |
| 允许集合 `get_args(EagleModelTypes)`、`get_args(NgramGPUTypes)`、`draft_model`、`dspark`（06 §4.1） | `vllm.py:1249-1260`、`1282-1288`；`vllm/config/speculative.py:66,69` |
| 显式 True 遇不兼容抛 `ValueError`，非静默回落（06 §4.1） | `vllm.py:1243-1269` |
| Executor 支持性来自选中类 `supports_async_scheduling()`，基类 False、UniProc/Multiproc True（06 §4.1） | `abstract.py:383-387` False；`uniproc_executor.py:157-158`、`multiproc_executor.py:558-559` True；`vllm.py:1231-1232` 先 `Executor.get_class(self)` |
| 容量表 5 行（06 §4.1）：F/p=1→1；F/p>1→p；T/MRV2→p+1；T/MRV1,p=1→2；T/MRV1,p>1→p | `vllm.py:567-577`，逐支一致，含“MRV1 对 async+PP 支持不完整”的源码注释 |
| 未指定 `scheduler_cls` 时 async 为真选 `AsyncScheduler`，否则 `Scheduler`；自定义类覆盖并告警（06 §4.1） | `scheduler.py:201-222` |
| 仅容量 > 1 才 `deque(maxlen=capacity)`；`batch_queue is None` 才选 `step`（06 §4.1） | `core.py:210-216`、`235-237` |
| 选择链顺序：`__post_init__` → `get_scheduler_cls` → `max_concurrent_batches` → `EngineCore.__init__`（06 §4.1；02 启动树同）| `core.py:146`（SOM）→`149`（get_scheduler_cls）→`162`（Scheduler(...)）→`210-216`（queue）→`235`（step_fn） |
| 队列分支中先 `execute_model(non_block=True)`，再 grammar，再 `sample_tokens(non_block=True)`，不等 exec future（两页） | `core.py:677-697` |
| pooling / 本轮未执行模型 → 直接排 exec future；`pending_structured_output_tokens` → 延期 sampling（两页） | `core.py:685-702`、`747-762` |
| 采样结果为 `None` 时读回 exec future 重抛原异常（02 §2.4、06 §7） | `core.py:725-730` |
| 有空位且（本轮执行了模型或仍有请求）时 `return None`；否则 `pop` 最旧（两页） | `core.py:705-711`、`720` |
| `[有内部输出] output_queue.put_nowait`（02 §2.4 树） | `core.py:1483-1486` `for output in outputs.items() if outputs else ()` |
| `AsyncOutputFuture.result` 延后 `get_output`；`non_block=True` 不把 worker 调用搬到后台线程（02 §2.4 末段） | `uniproc_executor.py:31-47`、`88-114`（`run_method` 内联执行） |
| `schedule` / `update_from_output` 由 `AsyncScheduler` 继承、经覆写钩子维护异步进度（02 §2.4 前言） | `async_scheduler.py` 只覆写 `_update_after_schedule` 与 `_update_request_with_output` |
| 时序图 “grammar 或 None”（02 §2.3.4 下图） | `scheduler.py:1813-1835`，两处 `return None` |
| 默认 R 重放第 1 次入队即返回、第 2 次达容量取 S1（06 §4.2） | `core.py:705-711`；`async_scheduler.py:19-50` 记 1 个 placeholder（`is_prefill_chunk` 见 `scheduler.py:1501-1503`，整段 prefill 不算 chunk）；第 2 次 `current_step=2` 满足 `next_decode_eligible_step=2`（`scheduler.py:536,607`），且 max_tokens=2 不触发 `scheduler.py:591-605` 的提前跳过；`num_new_tokens = 13 − 12 = 1` |
| `test_engine_core_concurrent_batches` “显式关闭 async 并强制容量 2”（02 §2.4、06 §4.2/§8） | `tests/v1/engine/test_engine_core.py:302-329`：`async_scheduling=False` + `patch.object(VllmConfig, "max_concurrent_batches", return_value=2)` |
| 新引的两个配置测试名（06 §8） | `tests/test_config.py:886 test_async_scheduling_with_pipeline_parallelism_is_allowed`、`:946 test_draft_model_enables_async_scheduling_by_default`，主题相符 |

---

## 3. 新增/回归 findings

### N-1 [P2] 02 §2.3.3、§2.4 与 06 §4.1：把 `model_executed` 读成“本轮是否执行了模型”

- 页面原文（02 §2.4）：“pooling 或本轮未执行模型时，队列直接保存执行 Future，无需采样”；06 §4.1 同义：“pooling 或本轮没有模型执行时直接排入 execute future”。
- 源码证据：`vllm/v1/engine/core.py:682-687`
  ```
  if self.is_ec_consumer:
      model_executed = scheduler_output.total_num_scheduled_tokens > 0
  if self.is_pooling_model or not model_executed:
      future = cast(Future[ModelRunnerOutput], exec_future)
  ```
  `is_ec_consumer` 定义在 `core.py:218-221`：`ec_transfer_config is None or ec_transfer_config.is_ec_consumer`。因此 `model_executed` 的语义是“EC consumer（含无 EC 配置的普通部署）且本轮有 scheduled token”。在 **EC producer**（`ec_transfer_config` 非 None 且 `is_ec_producer`）的引擎上，即使本轮排了 token，`model_executed` 仍为 False，于是照样只排 exec future、永不采样。
- 影响：普通部署下结论不变（`ec_transfer_config is None` ⇒ True），但“本轮未执行模型”会让读者把这个标志当成模型是否真的跑过，EC 分离部署的分支因此缺席；同一个 `model_executed` 还被 `_process_engine_step` 用来决定 `post_step` 与 sleep(0.001)。
- 建议修复：把条件写成“pooling 模型，或本轮没有 EC-consumer 侧的 scheduled token（`model_executed` 为假，EC producer 恒为假）”，或在 06 §4.1 的例外列表里补一行 EC producer。

### N-2 [P2] 02 §2.3.4 图 3 下图：`par` 并列会被读成进程级并发，而代表路径（TP=1）是同线程内联

- 页面原文（图内）：`par Engine 提交与排队 … and 执行侧按提交顺序推进`，并列支里 `X->>R: 分发模型执行`。
- 源码证据：默认 `vllm serve <model>`（TP=1，world_size=1）选 `UniProcExecutor`；`uniproc_executor.py:104-114` 在 `non_block=True` 下仍 `run_method(self.driver_worker, ...)` **内联**执行，只把结果包成 `AsyncOutputFuture`。真正的重叠来自 CUDA stream 与延后的 `get_output()`，不是两个并行的参与者。多进程（TP>1）才有图上那种真并发。
- 影响：该页自己在正文和源码路线里已经否定了“搬到后台线程”，所以不是事实错误；但图与正文的强弱不一致，读者会先信图。
- 建议修复：把并列支改成顺序提交 + 设备侧异步推进的表达（例如 `Note over X,R: 单进程内联提交，设备工作异步推进；多进程时为真并发`），或在图说明里点名 TP=1/TP>1 的差别。

### N-3 [P2] 02 §2.3.3 的 owner 链接缺 `#4.1` 锚点，与 11/12/19 的写法不一致

- 页面原文：“完整选择轴与兼容条件继续见 [[06_vllm_engine_architecture_analysis|Engine 运行循环]]”。
- 对照：11:46、12:49、19:360 都写成 `[[06_vllm_engine_architecture_analysis#4.1 队列里必须同时保留 future 和原计划|…]]`。02 作为架构页反而落到页级链接。
- 建议修复：改成同一个 `#4.1` 锚点（该标题存在，`check_links` 已验证 0 stale_section）。

### N-4 [P2] 07 §2.1 与 06 §4.1 各写一遍解析条件，新指定的单一 owner 立刻出现第二份副本

- 页面原文（07:113）：“`async_scheduling=None` 还不是最终选择：`VllmConfig.__post_init__()` 会结合 executor 支持、spec 方法与兼容条件解析。pooling 默认关闭 async；不兼容 spec 方法、禁用 padded drafter batch 或 ROCm DeepEP high-throughput DBO 等组合会使自动选择关闭，显式强开不兼容组合则报错。”
- 判断：与 06 §4.1 目前**内容一致、无矛盾**，且 07 有自己的 `get_scheduler_cls` 选择轴表需要这段背景；但这是全域报告 §4.1 指定 06 独占的那条轴的第二份完整表述，且 07 这段没有把读者交回 06（07:721 只在 fence 处提到 06）。这正是上一轮“启动顺序画 6 遍”的同类漂移风险。
- 建议修复：07 保留“Scheduler 实现由解析后的 async 决定”一行，条件清单压成一句并链接 06 §4.1；或在这段末尾补一句 owner 指向。

### N-5 [P2] 06 §2 的 scoping 句覆盖了整节，但 §2.1/§2.3 与同步/异步分支无关

- 页面原文：“本节先固定 `async_scheduling=False`、PP=1、未自定义 `scheduler_cls`…”，而 §2.1（前端登记、MP ADD 传输）和 §2.3（输出线程、前端可见完成）在两条分支下完全相同。
- 影响：读者可能以为输出交付路径也随 async 改变；§2.3 图后那句“图中仍采用本节显式关闭 async…的同步分支”会强化这一误读。
- 建议修复：把 scoping 句的适用范围限定到 §2.2（及其图），或在 §2.1/§2.3 各加半句“这两步与是否异步调度无关”。

---

## 4. 跨页一致性复核

### 4.1 默认 step 路径与 Scheduler/AsyncScheduler 轴（02/06 vs 07/11/12/19）

| 页面 | 现在的口径 | 与 06 §4.1 是否一致 |
|---|---|---|
| 02 §2.3.3、§2.3 末段、§2.4 | 默认 AsyncScheduler + 容量 2 `step_with_batch_queue`；关闭 async 且 PP=1 才 `step`；PP>1 仍用队列 | ✓ |
| 06 §2、§4.1、§4.2 | owner 自述；§2 标为同步教学分支 | — |
| 07 §2.1（109、113 行） | Scheduler 实现由解析后的 async 决定；batch queue 由 `max_concurrent_batches > 1` 创建，“有 batch queue ≠ 选了 AsyncScheduler” | ✓（内容一致，但重复，见 N-4） |
| 11 §1.4（46 行）、§1.4 对照段（80 行）、§8 树（403 行） | 选中 MRV1 不表示关闭 async；PP=1 默认容量 2；同步分支单独标注；显式交回 06 §4.1 | ✓ |
| 12 §1.5（49 行）、§2.3（162 行）、§8 树（451、457 行） | MRV2 容量 PP+1，PP=1 缺省即容量 2；显式交回 06 §4.1 | ✓ |
| 19 §7.3 前言（360 行）、§8 树（447 行）、Related（650 行） | “06 是唯一拥有该选择轴的页” | ✓ |

结论：上一轮最严重的“默认路径口径分裂”在 02/06/07/11/12/19 之间已经收口，术语（`AsyncScheduler`、`max_concurrent_batches`、`step_fn`、批队列容量）一致，测试注记（`test_engine_core_concurrent_batches` 非默认配置测试）三页写法一致。唯一残留是 N-3（锚点粒度）与 N-4（重复表述）。

### 4.2 02 对 14 / 15 / 16 的落点

- 14：**0 链接**（与上轮相同）。§2.3.3“token 选择”、§2.4 树中的 `Scheduler.get_grammar_bitmask` 与 `GPUModelRunner.sample_tokens` 仍无交接对象。item 8 因此仍 FAIL。
- 15：仅 §5.3 变体表 1 次，仍未说明它横跨 Renderer / encoder 预算 / Runner 的哪些模块。
- 16：仅 §6 读者表 1 次；§3 无落点。
- 注：14/15/16 三页本轮都有改动（分别 +92/+42/+89 行），但 02 侧的落点没有补，属单向缺口。

### 4.3 gRPC / Rust frontend / run-batch / scale-out 的 owner 现状

| 功能点 | 02 现在指向 | 目标页是否真有内容 | 结论 |
|---|---|---|---|
| gRPC（`vllm/entrypoints/grpc_server.py::serve_grpc`） | 13 | 13 中 `grpc` 命中 0 次 | 仍错配（F-4） |
| Rust frontend 进程管理 | 13 | 仅 13:535 一句“显式 API count 大于一会被改为一” | 仍近似无主 |
| `run-batch` 文件作业 | 02 §5.4 自持 | 24:74 只把 `run_batch.py` 当作 endpoint 插件 no-op 的例子 | 无深度 owner（与上轮同） |
| scale-out token-in 端点 | 02 未提 | 03 仅 §源码路线锚点（03:308） | 仍无正文 owner（F-6） |

index.md 本轮只调整阅读依赖（11/12 默认代际、26 上移、20/21 次序、23 箭头方向），未为上述四项指定 owner；全域报告已把它们路由到 `planning-codebase-analysis`，所以这些覆盖缺口属于“待 planning 决策”，但 **02 指向 13 的那句错配是页内可修的，不应保留**。

---

## 5. 契约重跑

### 5.1 02：base rubric + 8 项架构检查

| # | 检查 | 结果 | 说明 |
|---|---|---|---|
| 1 | 背景、目标、能力边界，单一分类轴 | pass | 未改动，仍成立 |
| 2 | 静态 / 动态 / 代码图同名 | pass | 新增图沿用同一组五/六个参与者名 |
| 3 | 每模块职责、理由、被否方案、限制 | pass | §3.2 图注与节点标签改写后仍自洽 |
| 4 | 多对多代码映射 | pass | §4 表 Engine 行已补 `step_with_batch_queue` |
| 5 | 场景清单与真实入口对账 | **FAIL** | scale-out 场景缺失（F-6）；§5.3 交接错配（F-4）；§5.6 DP=2 入口未写（F-7）；P2-8 清单缺口未补 |
| 6 | 命令与 flag 对 parser | pass | §5 未改动，上轮已逐条对过 parser |
| 7 | 调用树父子关系、顺序兄弟、返回到调用者 | pass | 新 EngineCore 子树父子关系与源码一致；残留 P2-4、P2-7 |
| 8 | 场景完成条件与读者交接 | **FAIL** | 14 零链接（F-5）；15/16 无架构落点；§6 表仍缺 7 页（P2-9） |

base rubric：beat2 pass；**hop-walk 由 FAIL 转 pass**；delete-code pass（新增内容无源码搬运）；figure-trigger 满足（timing）；algorithm-replay pass；spot-check 12/12 一致。综合仍 **REJECT**（架构 item 5/8 + 5 条 P1）。

### 5.2 06：base rubric + feature review

- 引言（问题 → 解法形状 → 代价）：pass。
- 最小例子可重放：pass，且新增的“默认 R”表可独立重放（见 §2 表末三行证据）。
- **变体集合与枚举依据：pass（原 FAIL）**。选择轴单元现在具备：
  - 目的：确定 `step_fn` 与 Scheduler 类的入口选择，并明说 owner 是本页；
  - 输入：`async_scheduling`（含 None）、runner 代际、PP 大小、`scheduler_cls`、Executor 类的 `supports_async_scheduling()`；
  - 处理逻辑：`__post_init__` 解析 → `get_scheduler_cls` → `max_concurrent_batches` → `EngineCore.__init__` 建队列与 `step_fn`，两张表把 5 类输入与 5 种容量组合枚举完；
  - 边界约束：显式 True 抛 `ValueError`、pooling 只在自动模式关闭、自定义 `scheduler_cls` 覆盖且接口不保证、async=False + PP>1 仍有队列；
  - 支持范围：只证两个内置 Scheduler 与两个内置 Executor，spec 算法交 16，设备后果交 11/12，调度算法交 07。
- 是否重复 07：§4.1 未复述预算/抢占/字段合同，只写入口选择；§4.3 的 placeholder 叙述与 07 §8 仍是上一轮已记录的既存重复，本轮未加重。
- 第三方依赖边界、失败边界与成本、页面替换与保存：pass。
- 综合：**PASS**，`feature: pass`；剩余 P2-11、P2-12、P2-14（未改）与 P2-13（部分）以及 N-1、N-5。

---

## 6. 未复核范围

- 未运行 GPU、多进程、真实服务、EC/PD 分离部署，因此 N-1 的 EC producer 分支是读码推断。
- 未跑 mkdocs build / mathjax-corpus（本轮无建站证据需求，T0 四项已全绿）。
- 只打开了 07/11/12/13/19/24/03 的相关小节，未复审这些页面的其余 finding（属其他组）。
