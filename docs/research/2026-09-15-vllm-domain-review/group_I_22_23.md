# Group I 独立审读：vLLM 22（分离式 KV Serving）与 23（可观测性与可靠性）

- 审读人：独立 reviewer（未参与两页写作）；只报告发现，未改动 wiki/、skills/、docs/ 或源码 checkout。
- 冻结源码：`/Users/suhaibo/97-llm/vllm`，`git rev-parse HEAD` = `199cb9b964822e59ab9b58d88e7be31eb419a2ae`（审读开始与结束各核对一次；仅既有未跟踪 `artifacts/`）。未 fetch/checkout。
- 方法：`source-fidelity.md`、`codebase.md`、`analysis-focus.md`、`page-review-rubric.md`、`document-types/feature-analysis.md`、`reviews/feature-analysis.md` 全文；CLAUDE.md 所有权/出处规则。
- 运行边界：全部为源码静态阅读与手工算例重放；未运行 vLLM、GPU、多机、NIXL/MoRI/Mooncake/OTLP 服务或 pytest。Mermaid 只审源文本，未重新渲染。
- 并发说明：审读期间工作区里 `13_vllm_serving_control_plane_analysis.md` 与 `wiki/changelog.md` 有他人未提交改动；本报告对 13 的判断同时核对了 HEAD 与工作区版本（两者都不含 P/D 路由内容）。

---

## 1. 判定行

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 22_vllm_disaggregated_kv_serving_analysis | pass | pass | pass | transform, layout, timing, coupled-planes | pass | 14/14 | REJECT（小范围） | feature: FAIL §9/§5.3 失败边界（HMA 下 NIXL READ 失败以"成功接收"形态回到 Scheduler，页面只写成"恢复未完整实现"）；另有 E2E 路由段无 owner 的**合同漂移**（§2/§14.4 指向 13、03，二者均未覆盖），应回 planning 而非算 rubric 失败 |
| 23_vllm_observability_reliability_analysis | pass | pass | pass | transform, timing | pass | 16/16 | REJECT（小范围） | feature: FAIL §5.2 传输观测变体集无枚举依据（只讲 NIXL，未按 `build_kv_connector_stats/build_prom_metrics` 覆写点列出，MoRIIO 无遥测未提）；§6.1 诊断把"数据装载"归到 prefill，但异步远端 KV 等待实际落在 queue 区间；§3.1/§6 失败终结请求进入完成 histogram 的样本语义缺失 |

算例重放结果（均与页面一致）：

- 22：NIXL pull `(12,True)`，成功后 `12→11`；仅 41 invalid → `idx=1` → computed 4；NIXL 非 HMA 整组 invalid → 0；反向复用阈值例 remote 8 / local 4 / 阈值 5 → `(0,False)`，阈值 4 → `(4,True)`；MoRIIO READ `(11,False)`、WRITE `(12,True)`；Mooncake 直连 D TP1/P TP1 `pull_tasks_count=1`，P TP2 时 D 侧 `tp_ratio=-2`→`[0,1]`、P 各 rank `tp_ratio=2`→`[0]`；store 4 heads、store TP2：rank-local 3 键、sharded 6 键，LBHNC 每 tensor/chunk/shard 1 段、LBNHC 4 段；Multi A=8/B=12 选 A，A=12/B=None 整体 `(None,False)`；PUSH_REG msgpack 11 键。
- 23：TTFT 0.090、queue 0.020、prefill 0.050、ITL 0.025/0.030、decode 0.055、inference 0.105、TPOT 0.055/2=0.0275、E2E 0.145，queue+prefill+decode=0.125≠0.145；抢占例 queue 2、prefill 6、decode 3、inference 9、preemptions 1；MM 分摊 0.012/2=0.006、0.016/2=0.008，按 ID 合并取 max=0.008。

---

## 2. 抽查锚点与结果

### 22（14 组，均与页面陈述一致；第 7 组暴露遗漏后果，见 P1-1）

| # | 锚点 | 页面陈述 | 源码结果 |
|---|---|---|---|
| 1 | `vllm/distributed/kv_transfer/kv_connector/factory.py::KVConnectorFactory.register_connector`（文件末尾调用） | 16 个注册名 / 15 类，`NixlConnector` 为别名 | 16 次调用；`nixl/connector.py` 末尾 `NixlConnector = NixlPullConnector` ✓；`get_connector_class` 外部 module path 优先、空串 raise、`supports_kw(...,"kv_cache_config")` ✓ |
| 2 | `vllm/config/kv_transfer.py::KVTransferConfig` | 13 字段、默认值、5 个死字段 | 13 个字段；`kv_buffer_size=1e9`、`kv_port=14579`、`kv_load_failure_policy="fail"` ✓；`grep -rnw` 仅 `lmcache_mp_connector.py` 与 `k2_horizon.py` 同名局部变量 ✓ |
| 3 | `nixl/pull_scheduler.py::NixlPullConnectorScheduler.get_num_new_matched_tokens / request_finished` | `(12,True)`；阈值默认 64；13 个返回键；abort 空 recv | ✓（`kv_recompute_threshold` 默认 64 在 `base_scheduler.__init__`） |
| 4 | `vllm/v1/core/sched/scheduler.py::Scheduler.schedule` | `delay_cache_blocks`、`reserved_blocks`、`WAITING_FOR_REMOTE_KVS`、先记 computed=12 | ✓，并注意 async-load 分支在 `record_event(SCHEDULED)` 之前 `continue`（用于 23 的 P1-3） |
| 5 | `Scheduler._update_waiting_for_remote_kv / _try_promote_blocked_waiting_request` | `12→11`；`PREEMPTED if num_preemptions else WAITING` | ✓ |
| 6 | `Scheduler._update_requests_with_invalid_blocks / _handle_invalid_blocks / update_from_output` | 首个 invalid 截断；默认 fail 先 `finish_requests(FINISHED_ERROR)` 再 `_update_from_kv_xfer_finished` | ✓ |
| 7 | `nixl/base_worker.py::NixlBaseConnectorWorker._handle_failed_transfer / get_finished` | 仅非 HMA 填 invalid | ✓；但 `_failed_recv_reqs` 在 HMA 下也入队并被 `done_recving.update(failed_recv_reqs)` 报为接收完成 |
| 8 | `nixl/push_scheduler.py::NixlPushConnectorScheduler.build_connector_meta / request_finished`；`push_worker.py::_do_send_reg_notif/_push_writer_loop` | watchdog 只 pop+warning；P lease 用 `_kv_lease_duration`；PUSH_REG 11 键；3b 精确 pop | ✓（contradiction 块成立） |
| 9 | `moriio/moriio_connector.py::MoRIIOConnectorScheduler.get_num_new_matched_tokens / update_connector_output`；`MoRIIOConnectorWorker.wait_for_layer_load / _pop_done_transfers / get_finished` | READ `(11,False)`；barrier 超时 warning；READ 不报 `finished_recving`；ACK 停放按 ReqId；`defer_timeout` 60 | ✓ |
| 10 | `moriio/moriio_engine.py::MoRIIOWriter.seal_pending_transfers / _finalize_if_complete / _mark_request_done` | seal 冻结实际入队数；`completion_notified` 先置位；异常走 `_mark_request_done` | ✓（另见 P2-2：seal 路径的异常未被捕获） |
| 11 | `mooncake/mooncake_connector.py::MooncakeConnectorWorker.process_pulling_result`；`kv_connector/utils.py::TransferTopology.handshake_target_ranks` | 只对 `ok_reqs` 减计数；TP 例 | ✓ |
| 12 | `mooncake/store/worker.py::MooncakeStoreWorker.start_load_kv / _select_store_layout`；`store/scheduler.py::get_num_new_matched_tokens` | 入队后 `assert self.load_async`；store TP ≥ 且整除本地 TP；`(None,False)` | ✓ |
| 13 | `multi_connector.py::MultiConnector.get_num_new_matched_tokens / update_state_after_alloc` | 任一 `None` 立即 `(None,False)`；非选中子收到真实 blocks、external=0 | ✓（§8 与 §14.7 表述一致） |
| 14 | 计数：extra 键 NIXL 9 / MoRIIO 13 / Mooncake 直连 3 / store 12 / Multi 1；环境变量 10+6=16；`MooncakeStoreConfig` 9 字段；`NIXL_CONNECTOR_VERSION=10` 与 hash factors | ✓（按页面 §11.3/§14.8 的 grep 重跑） |

### 23（16 组，均与页面陈述一致）

| # | 锚点 | 结果 |
|---|---|---|
| 1 | `vllm/v1/metrics/stats.py::IterationStats.update_from_output / update_from_events / update_from_finished_request` | TTFT 用 `iteration_timestamp - arrival_time`；`scheduled_ts==0` 才写；TPOT 分母 `n-1` ✓ |
| 2 | `vllm/v1/metrics/loggers.py::LoggingStatLogger._update_stats / _reset / log` | `_update_stats` 内 `_reset` 清零 `num_preemptions/num_corrupted_reqs` 后 `log` 才读取 ✓（页面指出的实现限制成立） |
| 3 | `loggers.py::StatLoggerManager.__init__`；`vllm/v1/engine/async_llm.py::AsyncLLM.__init__` | `log_stats or has_custom_loggers`；`enable_default_loggers=log_stats`、INFO、`client_count==1`；Prometheus 替换判定 ✓ |
| 4 | `loggers.py::PrometheusStatLogger.__init__ / record` | `vllm:num_preemptions`、`vllm:request_num_preemptions`、`num_requests_waiting_by_reason{capacity,deferred}`、`request_success{finished_reason}`、labels `model_name/engine`、`mostrecent` ✓ |
| 5 | `nixl/stats.py::NixlKVConnectorStats / NixlPromMetrics` | 7 组数组、`/1e6`、`is_empty` 查失败数组、`reduce` 吞吐 = 总 MiB / duration 和、7 个指标名 ✓ |
| 6 | `vllm/distributed/kv_transfer/kv_connector/v1/metrics.py::KVConnectorLogging / KVConnectorProm` | `build_kv_connector_stats` 为 None 时 warning_once；prom 为 None 时 observe 直接返回 ✓ |
| 7 | `vllm/v1/engine/output_processor.py::OutputProcessor.do_tracing / propagate_error / process_outputs`；`RequestOutputCollector.get` | do_tracing assert stats；propagate_error 只 `put(e)`；collector 遇 Exception 重抛 ✓ |
| 8 | `vllm/v1/fault_tolerance/engine_core_sentinel.py::EngineCoreSentinel.on_fault / retry / handle_command`；`fault_tolerant_wrapper` | 清 resumed → `finish_requests(None, FINISHED_ABORTED)` → 清 batch_queue → DEAD/UNHEALTHY；只 UNHEALTHY 可执行指令；超时 `raise` ✓ |
| 9 | `vllm/v1/worker/sentinel/gpu_worker_sentinel.py::WorkerSentinel.__init__ / retry / _clean_worker_state` | `FT_BACKEND_SET={deepep_low_latency,nixl_ep}`；DP>1 清 a2a buffer、重建 CPU group；MRV1/MRV2 清理分支 ✓ |
| 10 | `vllm/v1/engine/core.py::EngineCoreProc._send_engine_dead`、`run_engine_core` except 分支、`_handle_client_request` 的 `EXECUTOR_FAILED` | join 5 秒；`RuntimeError("Executor failed.")` ✓ |
| 11 | `vllm/v1/executor/multiproc_executor.py::MultiprocExecutor.start_worker_monitor / _ensure_worker_termination` | `is_failed=True` → shutdown → callback；宽限 → SIGTERM → 4 秒 → kill ✓ |
| 12 | `vllm/v1/engine/core_client.py::BackgroundResources.validate_alive`、`process_outputs_socket`、`MPClient.start_engine_core_monitor` | sentinel 帧抛 EngineDeadError；异常/取消均投递队列；monitor 置 `engine_dead` ✓ |
| 13 | `vllm/entrypoints/serve/instrumentator/health.py::health`；`vllm/v1/metrics/prometheus.py::get_prometheus_registry / setup_multiprocess_prometheus` | render-only 200；`PROMETHEUS_MULTIPROC_DIR` 选 MultiProcessCollector ✓ |
| 14 | `vllm/config/observability.py::ObservabilityConfig`（13 字段）、`vllm/config/parallel.py::ParallelConfig`（61 个顶层注解字段）、`vllm/config/fault_tolerance.py::FaultToleranceConfig.engine_recovery_timeout_sec=120`；`collect_model_forward_time/execute_time` 无消费者 | ✓ |
| 15 | `vllm/v1/core/kv_cache_metrics.py::BlockMetricsState`（`deque(maxlen=4)`）、`stats.py::CachingMetrics(max_recent_requests=1000)` | ✓ |
| 16 | `vllm/benchmarks/mm_processor.py::get_timing_stats_from_engine`（max 合并）、`GPUModelRunner.timed_encoder_operation / get_encoder_timing_stats`；`vllm/tracing/otel.py::get_span_exporter`；`tests/v1/tracing/test_tracing.py::test_traces`（15 秒）；`docs/design/metrics.md`（"most recent SCHEDULED"） | ✓ |

---

## 3. E2E / 连贯性

### 3.1 22：P/D 在端到端链路中的位置

页面把"同一 R 的 KV 交接"讲得很完整（§5 调度→分配→worker 挂点→聚合→晋升/释放，§6–8 六条数据面同例重放），与 07（remote-KV 等待/晋升，07 第 500 行只做指向）、08（本地分配/offload）、06（step fence 与 `delay_free_blocks` 两层，06 §220 指回 22 §5.4）、18（KV 平面 rank 约定，18 两处明确交给 22 §11.1）术语一致，未发现重复拥有或互相矛盾。

断点在链路两端：

1. **路由/代理 → P → 响应携带 `kv_transfer_params` → 代理转发给 D → D 流式输出** 这一段没有任何页面拥有（P1-2）。22 §2 写"实例发现与将请求送到匹配 P/D 的控制面由 13 解释"，Related Pages 写 13 讲"P/D 实例路由"，§14.4 又把"请求语义层怎样携带与回传 `kv_transfer_params`"交给 03；但 13（HEAD 与工作区）和 03 中 `kv_transfer_params`/P/D/proxy 均 0 命中。§14.4 最后一行又自己承认"基线下全库 wiki 无人展开"，同页前后矛盾。读者无法从 wiki 得知：P 腿请求为何带 `do_remote_decode=True`、`max_tokens=1`、`stream=False`（否则 `request_finished` 的 `FINISHED_LENGTH_CAPPED/STOPPED` 门不成立、不会延迟释放），P 的 `kv_transfer_params` 如何进入 HTTP 响应再被代理塞回 D 请求，D 被服务层拒绝时 `_with_kv_transfer_rejection_cleanup → notify_kv_transfer_request_rejected` 如何触发 §5.4 的"空 recv 通知"分支。
2. **§1 的时延模型 `T_route+T_prefill+T_KV handoff+T_decode queue+T_decode` 没有映射到任何 vLLM 指标**，而 23 也没有接住（见下 3.2 第 1 点）。

### 3.2 23：事件 → 指标/trace → 05 排障 → 13/26 故障传播

- 06/07 → 23：事件来源（`add_request` QUEUED、`schedule` SCHEDULED、`_preempt_request` PREEMPTED）、`update_from_output` 附事件与 `make_stats`、batch-queue 交给 06，调用树与源码一致。
- 23 → 13/26：worker 死亡与 EngineCore 死亡两路、FT 状态机与 13 §4.4 的简述一致（13 明确把完整状态机交回 23），26 的 `is_failed`/SIGTERM/4 秒与 23 一致，无矛盾。
- 主要断点：
  1. **P/D consumer 的异步远端 KV 等待被计入 `request_queue_time_seconds` 与 D 侧 TTFT**（`Scheduler.schedule` 的 async-load 分支在记录 SCHEDULED 前 `continue`，晋升回 WAITING 后下一次调度才记 SCHEDULED）；同步 load（MoRIIO READ）则进 prefill。23 §6.1 却写"queue 稳定而 prefill 上升，才继续检查已调度阶段的执行或数据装载"，对 P/D 部署会把传输问题误导到 prefill（P1-3）。05 第 136 行反而正确提示 `deferred` 可能含 KV transfer——23 是 05 的机制 owner，却没有给出这一依据。
  2. **失败/中止终结的请求会进入完成请求 histogram 且区间以 0 为基点**（P1-5），23 只写了"慢请求不提前进入 histogram"和"fatal 不保证产生 FinishedRequestStats"，没有覆盖 ERROR/FT ABORT 这两条最常见的故障样本来源。
  3. **05 → 23 的回指有两处错号**（P1-6，问题在 05）：05 §4 表"运行中批量报 EngineDeadError"一行写"按 `27` 的状态边界处置"（域内只有 01–26，FT 状态边界在 23 §6.3）；"报错落在编译或 graph replay"一行写"下一 owner 是 `23`"（应为 19）。
  4. **22 → 23 的传输观测交接只接住了 NIXL**（P1-4）：22 §14 intro 与 §14.4 把"transfer latency、lease expiry 与 invalid blocks 的信号"交给 23，但 23 §5.2 只展开 NIXL，未说明 MoRIIO 没有 connector stats、Mooncake 直连只有 text stats 且成功样本只在 P 侧、store/Multi/LMCacheMP/Offloading/HF3FS 各自有无 Prometheus collector。

### 3.3 附录式清单是否挤占 E2E

22 第 702–1080 行（§11–§14，约 35%）为配置、环境变量、二十条流程、所有权、注册名、几何守卫、Multi 折叠与 grep 口径；§1 末段已明确"查配置直接读 §11、完整流程留在 §14"，主叙事 §1–§10 可独立读完，未发现清单插在机制解释前面。§14.3 的触发/完成点表与 §5 调用树有较多重述（例如 ⑯ 行复写 `_connector_finished` 顺序），不构成矛盾，属可压缩项，不计为缺陷。23 的 §7 配置表只在机制讲完后出现，比例合适。

---

## 4. 发现（按严重度）

未发现 P0（与冻结源码直接相反、足以误导的事实错误）。

### P1

**P1-1｜22 §9「lease 续期、失效上报与超时边界」第 4 段 + §5.3 第 4 段｜HMA 下 NIXL READ 失败以"成功接收"回到 Scheduler，页面没有写出该后果**

- 页面原文：「…只有非 HMA 分支向 invalid block 队列填入目标 IDs…不能写成"所有 hybrid group 的自动恢复已经完整实现"」；§5.3：「`finished_recving` 证明的是接收可收尾，成功还需要错误集为空及设备后处理完成」。
- 源码：`nixl/base_worker.py::NixlBaseConnectorWorker._handle_failed_transfer` 在 `_is_hma_required` 为真时不写 `_invalid_block_ids`，但无条件 `self._failed_recv_reqs.put(req_id)`；`NixlBaseConnectorWorker.get_finished` 执行 `done_recving.update(failed_recv_reqs)` 并跳过后处理。于是 Scheduler 收到 `finished_recving` 且 `invalid_block_ids` 为空；`Scheduler._update_waiting_for_remote_kv` 走成功分支 `cache_blocks(...)` 并 `12→11`。另外 `Scheduler._update_requests_with_invalid_blocks` 本身也是 `(req_block_ids,) = get_block_ids(...)` 单组实现（TODO HMA）。
- 后果：HMA/混合模型上，传输失败的目标块被当作有效 KV 缓存并参与计算，既不 fail 也不 recompute。按页面自己的判据"错误集为空"会误判为成功。
- 建议：在 §9 该段与 §5.3 明写"HMA 路径下失败请求被报为 `finished_recving` 且无 invalid block，Scheduler 按成功路径缓存未写入的块"，给出上述两个锚点；§5 第一张图的失败分支注释补一句 HMA 例外；§14.2 ⑭ 的启用条件加"非 HMA"。

**P1-2｜22 §2 第 3 段、§14「它不是什么」、§14.4 表 `P/D 实例拓扑` 与 `kv_transfer_params` 两行、Related Pages｜P/D 端到端路由段无 owner，页面指向的 13/03 均未覆盖，且与本页 §14.4 末行自相矛盾（合同漂移）**

- 页面原文：「实例发现与将请求送到匹配 P/D 的控制面由 [[13…|Serving 控制面]]解释」；§14.4「请求语义层怎样携带与回传它 → **03**」；末行「`examples/disaggregated/` 的 proxy/router 参考实现 | 无（只登记）| 建议归 13；基线下全库 wiki 无人展开」。
- 证据：13（HEAD 与工作区）和 03 中 `kv_transfer_params`、P/D、分离式、proxy 均无内容。源码链路：`tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`（P 腿设置 `do_remote_decode=True`、`max_tokens=1`、`stream=False`，再把响应 JSON 的 `kv_transfer_params` 写回 D 请求）；`nixl/pull_scheduler.py::NixlPullConnectorScheduler.request_finished`（仅 `FINISHED_LENGTH_CAPPED/STOPPED` 才延迟释放并返回参数）；`vllm/entrypoints/generate/base/serving.py::…_with_kv_transfer_rejection_cleanup`（D 服务层拒绝时 `notify_kv_transfer_request_rejected`，对应 §5.4 的 abort 清理分支）。
- 建议：按 rubric 这是合同漂移，交 `planning-codebase-analysis` 决定 owner（13 或 22）。在决定前，22 至少应：（a）删除"由 13 解释"的断言，改为"当前无 owner，见 §14.4"；（b）在 §1 或 §5.4 增加一段最小 E2E：client→proxy→P（上述三个字段）→P 响应带参数→proxy→D→D 输出，并挂上三个锚点；（c）Related Pages 中 13 的描述去掉"P/D 实例路由"。

**P1-3｜23 §6.1「先区分故障域」第 1 段、§5.2 第 2 段、§3｜异步远端 KV 等待落在 queue 区间与 D 侧 TTFT，页面的诊断顺序把"数据装载"归到 prefill**

- 页面原文：「queue 上升说明首次调度前等待增长…queue 稳定而 prefill 上升，才继续检查已调度阶段的执行或数据装载」；§5.2：「时间起止点来自 NIXL 库 telemetry，不是请求从 waiting 到恢复执行的整个区间」（未说明该区间落在哪里）。
- 源码：`vllm/v1/core/sched/scheduler.py::Scheduler.add_request` 记 QUEUED；`Scheduler.schedule` 的 `if load_kv_async:` 分支设置 `WAITING_FOR_REMOTE_KVS` 后 `continue`，在 `request.record_event(EngineCoreEventType.SCHEDULED, …)` 之前；晋升回 WAITING 后，下一次调度进入 running 分支才记 SCHEDULED。`stats.py::IterationStats.update_from_finished_request` 的 `queued_time = scheduled_ts - queued_ts`。因此 D 侧握手、传输与晋升等待全部计入 `vllm:request_queue_time_seconds`，D 的 TTFT 起点是 D 前端到达（不含 P 腿）；同步 load（MoRIIO READ）则在 SCHEDULED 之后，计入 prefill。P 腿（`max_tokens=1`）只贡献 TTFT，不贡献 ITL。
- 建议：在 §6.1 加一个 P/D 分支：consumer 上 queue 升高先对照 `deferred` gauge 与 connector 传输统计，不要只看 capacity；说明同步/异步 load 分别落在 prefill/queue；在 §3.2 或 §5.2 用 22 的 R 给出一行区间归属。22 §1 的时延模型可以链接到这里。

**P1-4｜23 §5.2「KV 传输：通用搬运 stats，connector 决定实际指标」｜传输观测变体集没有从源码选择点给出枚举依据，只讲 NIXL**

- 页面原文：「以 [[22…]] 使用的 NIXL 为例…」，全节没有说明其他 connector 是否有遥测。
- 源码枚举依据：覆写 `build_kv_connector_stats` 的有 `nixl/connector.py`、`mooncake/mooncake_connector.py`、`mooncake/store/connector.py`、`multi_connector.py`、`lmcache_mp_connector.py`、`offloading_connector.py`、`hf3fs/hf3fs_connector.py`；同时覆写 `build_prom_metrics` 的是其中除 Mooncake 直连以外的 6 个。`moriio/` 两者都没有。`MooncakeConnector.get_kv_connector_stats` 的 docstring 写明 P 侧记录成功时延/字节，D 侧只记录失败。
- 后果：22 把"transfer latency…的信号"交给 23，但 22 展开的六条数据面里，MoRIIO READ/WRITE 没有 connector 遥测，Mooncake 直连没有 Prometheus 指标，且成功样本只出现在 P 侧。读者无法区分"没有指标"和"指标为 0"。
- 建议：§5.2 开头给出枚举依据（上述覆写点）和一张 connector × {text stats, Prometheus} 小表，并点明 MoRIIO 无遥测、Mooncake 直连的 P/D 不对称；NIXL 继续作为展开例。

**P1-5｜23 §3.1「完成时才结算整条请求」末句、§6.2 倒数第 2 段、§7.4｜ERROR/FT ABORT 终结的请求会产生以 0 为基点的完成样本和无 token 的 TTFT 样本，页面未说明**

- 页面原文：「一个始终未完成的慢请求不会提前进入完成请求 histogram」；「fatal 并不保证为每个未完成请求产生正常 FinishedRequestStats…完成请求 histogram 会遗漏这些…样本」。
- 源码：（a）`Scheduler.update_from_output` 对 KV load 默认 fail 或 grammar 错误调用 `finish_requests(…FINISHED_ERROR)`，并为其生成带 `events=request.take_events()` 的 `EngineCoreOutput`。从未被调度的请求只带 QUEUED。（b）FT 下 `EngineCoreSentinel.on_fault → EngineCoreProc._send_abort_outputs → _send_finish_outputs_to_client` 生成 `EngineCoreOutput(req_id, [], finish_reason=ABORT)`，不带事件。（c）前端 `OutputProcessor.process_outputs` 对每个输出调用 `IterationStats.update_from_output`：只要 `is_prefilling` 就追加 TTFT，并把 `first_token_ts` 设为输出时间戳，没有 token 数判断。随后 `update_from_finished_request` 对 `scheduled_ts==0` 也不做判断：`queued_time = 0 - queued_ts`（负值），`prefill_time/inference_time = monotonic 时间戳 - 0`，数值接近 EngineCore 进程的 monotonic 读数。（d）`PrometheusStatLogger.record` 对所有 `finished_requests` 无条件 observe。
- 后果：P/D 传输失败或 FT 中止时，TTFT/queue/prefill/inference histogram 会混入异常样本，正好是 05 排障依赖的那几条序列。prometheus_client 对负值 observe 的内部处理属第三方行为，本审读未核验。
- 建议：在 §3.1 结算段或 §6.2 增加"失败终结样本"一段，给出上述锚点，说明样本来源（ERROR 输出带部分事件、ABORT 输出不带事件）和查询时应按 `finished_reason` 过滤或排除；§7.4 的 stale/边界清单补一行。

**P1-6｜05 §4 排障表（页外，影响 23↔05 交接）｜两处错误的页号指向**

- 05 原文：「可恢复 fault-tolerance 状态与不可恢复死亡有不同合同，按 `27` 的状态边界处置」；「报错落在编译或 graph replay…下一 owner 是 `23`」。
- 证据：vLLM 域只有 01–26，FT 状态机在 23 §6.3；编译/CUDA Graph 的 owner 是 19（`19_vllm_compilation_cudagraph_analysis.md`），23 不拥有编译机制。
- 建议：交给 05 的 owner 修正为 `[[23…|可观测性与可靠性]] §6.3` 和 `[[19…|编译与 CUDA Graph]]`（使用 wikilink，避免裸数字再次漂移）。不在本组两页的编辑范围内。

### P2（共 11 项）

1. **22 §5.2 调用树 `ActiveKVConnector.no_forward`**：写"pre_forward + post_forward"，漏了 `post_forward(finished_req_ids, wait_for_save=False)`，零 token 步不会调用 `wait_for_save`（`vllm/v1/worker/gpu/kv_connector.py::ActiveKVConnector.no_forward`）。建议补上参数。
2. **22 §6.3 MoRIIO WRITE 失败段**：只写 `_write_worker_loop/_process_deferred_tasks` 捕获异常。另一条路径是 `MoRIIOConnectorWorker.wait_for_save → MoRIIOWriter.seal_pending_transfers → _finalize_if_complete → MoRIIOWrapper.waiting_for_transfer_complete`，它在前向线程上同步等待（最长 `transfer_timeout`=30 秒），抛出的 `TransferError` 在 seal 路径未被捕获，会沿 `wait_for_save` 传出。该路径是否触发取决于 seal 时 `writes_done>=expected`，频率未验证。建议在失败段与 §12 成本表各补一句。
3. **22 §6.3/§5.4**：MoRIIO P 侧 `MoRIIOConnectorScheduler.request_finished` 只接受 `FINISHED_LENGTH_CAPPED`（NIXL 同时接受 STOPPED），页面没有指出这一差异。*未验证推测*：READ 模式下 D 排队超过 `defer_timeout`（60 秒）时，P 会被强制回收源块，可能早于 D 读取。
4. **22 §6.4 标题「请求面走 ZMQ」**：D 通过 `MooncakeConnectorWorker._connect_to_prefiller_bootstrap` 用 `httpx` GET `/query` 发现 P，之后的元数据才走 ZMQ。建议写成"bootstrap 走 HTTP，元数据走 ZMQ"。
5. **22 §14.4 `requires_piecewise_for_cudagraph` 行**：后果交给 19，但 19 没有提到该 connector 钩子，只有通用 PIECEWISE 内容。交接偏弱，建议在 19 加一句回指，或在 22 里说明降级入口是 `VllmConfig.__post_init__`。
6. **22 §14.3**：⑯/⑤/⑥ 等行基本复写 §5 调用树，维护时容易两处漂移。可以只保留触发和完成点列。
7. **23 §6.3 NaN 段**：只核验了 MRV1 的 `AsyncGPUModelRunnerOutput.get_output`，但在本基线 MRV2 是默认 runner（`VllmConfig.use_v2_model_runner`），它的对应实现在 `vllm/v1/worker/gpu/async_utils.py`（同样读取 `VLLM_RAISE_ON_LOGIT_NANS`）。建议补上 MRV2 锚点，避免读者以为默认 runner 未覆盖。
8. **23 §4.3 末句**：「使用本页链路必须保持有效 stats 采集」没有写出后果。开 OTLP 且 `disable_log_stats=True`、又没有 custom logger 时，`RequestState.stats` 为 None，`OutputProcessor.do_tracing` 的 `assert` 在 output handler 内触发，经 `propagate_error` 让所有在途请求失败，handler 退出后 `AsyncLLM.errored` 为真、`/health` 返回 503。源码中未找到把 tracing 与 log_stats 绑定的配置守卫（仅静态阅读）。
9. **23 §3.1「首输出建立时间基点」**：页面叙述顺序是先 TTFT、再事件、最后 first/last 和 token 计数；实际 `num_generation_tokens` 在处理事件前就已累加。不影响数值，属叙述顺序小误差。
10. **23 §9 源码路线**：`FaultToleranceConfig` 实际在 `vllm/config/fault_tolerance.py`，路线只列了 `vllm/config/parallel.py::ParallelConfig`。
11. **22 §5 第一张 Mermaid 图**：节点 H「整组 40/41/42 无效 12→0」直接连到 R「recompute 策略」，图上看不出默认 `fail` 会先终结请求，只能靠正文补足。建议给 H→R 边加"仅 recompute"标签。

---

## 5. 未验证推测（不计入发现）

- FT `WorkerSentinel._clean_worker_state` 不清理 KV connector 的在途状态（如 NIXL `_recving_metadata`、MoRIIO 写队列）。P/D 与 FT 同时开启时，retry 后是否会残留跨 Engine 传输债务，需要故障注入验证。
- NIXL push 的 P 侧 WRITE 失败（`push_worker.py` 提交失败只 release handle）与 PUSH_REG 丢失一样，会让 D 心跳持续为 P 续租。22 §6.2 的 contradiction 块覆盖了注册丢失，没有覆盖 WRITE 失败。结构上同类，未逐行追完。
