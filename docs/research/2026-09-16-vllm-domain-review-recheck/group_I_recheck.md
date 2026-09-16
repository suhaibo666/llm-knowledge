# Group I 复审（第 2 轮）：vLLM 22（分离式 KV Serving）与 23（可观测性与可靠性）

- 审读人：同一独立 reviewer（未参与两页写作）；本轮只报告发现，**未改动任何文件**（wiki/、docs/、skills/、源码 checkout 均未写入）。
- 冻结源码：`/Users/suhaibo/97-llm/vllm`，`git rev-parse HEAD` = `199cb9b964822e59ab9b58d88e7be31eb419a2ae`（审读开始核对；`git status` 仅既有未跟踪 `artifacts/`）。未 fetch/checkout，HEAD 未移动。
- 被审对象：工作区未提交改动（`git diff`），22 +37 行 / 23 +68 行；同时交叉核对了同批改动的 05、07、08、13、26 与 `wiki/changelog.md`。
- 方法：先读自己上一轮报告 `docs/research/2026-09-15-vllm-domain-review/group_I_22_23.md`，对每条旧发现回到源码复核（不看新叙述下结论），再逐条核验新增文字，最后重放算例与跑 T0 门禁。
- 运行边界：全部为源码静态阅读 + 手工算例重放 + T0 checker；未运行 vLLM、GPU、多机、NIXL/MoRI/Mooncake/ZMQ/OTLP 服务，未做故障注入，Mermaid 未重新渲染（只审源文本 + checker）。

---

## 1. 判定行

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 22_vllm_disaggregated_kv_serving_analysis | pass | pass | pass | transform, layout, timing, coupled-planes | pass | 9/9（本轮新锚点）+ 14/14（上轮沿用） | ACCEPT（保留 planning 待决项） | feature: **pass** §9/§5.3 失败边界（HMA 缺口已按源码写出并贯通图/§14.1/§14.2/§14.3）；P/D E2E 路由段仍无 owner，changelog 已声明留待规划批准，按 rubric 不计页面失败，但 §2／§14／§14.4／Related Pages 的**页内自相矛盾未修**（P1-2 NOT_FIXED）；另 4 项新 P2 |
| 23_vllm_observability_reliability_analysis | pass | pass | pass | transform, timing, coupled-planes | pass | 18/18（本轮新锚点）+ 16/16（上轮沿用） | ACCEPT | feature: **pass** §5.2 connector 遥测矩阵、**pass** §6.1 区间归属、**pass** §3.1/§6.2/§7.4 异常终态样本、**pass** 新增 §5.3 KV cache 事件链；4 项新 P2（矩阵口径、deferred 标签、KVEventsConfig 未登记、两处措辞外推） |

- **无 P0，无 P1 新发现，未发现回归。** 上一轮已修正的表述没有被新文字重新破坏（例如 22 §5.3 原先"成功还需要错误集为空"的错误判据已被删除，未在别处复活）。
- T0 门禁（`.venv` python 3.13）：`check_links --strict` 452 页 broken/ambiguous/bare_index/stale_section/orphans **全 0**；`check_math --changed --strict`、`check_markdown --changed --strict`、`check_assets --changed --strict` 各 39 文件 **0 error / 0 warning**。22 新增的两个带锚点 wikilink（23 的 §5.2、§5.3）与 05 新增的 `#6.3 …` 锚点均解析通过。

算例重放（本轮自行重算，均与页面一致）：

- 22：`NixlPullConnectorScheduler.get_num_new_matched_tokens` 对 `do_remote_prefill` **无条件** `return count, True` → 12-token 例 `(12,True)`；`_update_waiting_for_remote_kv` 正常分支 `num_computed_tokens == num_tokens` 才 `= num_tokens - 1` → 12→11；仅 41 invalid → `idx=1`、`num_computed_tokens = idx * block_size = 4`；非 HMA 整组 invalid → 0。
- 23：TTFT 0.090、queue 0.020、prefill 0.050、ITL 0.025/0.030、decode 0.055、inference 0.105（=0.050+0.055）、TPOT 0.055/2=0.0275、E2E 0.145，queue+prefill+decode=0.125≠0.145；抢占例 queue 2 / prefill 6 / decode 3 / inference 9（=6+3）/ preemptions 1；ZMQ shutdown 预算 1 秒 + `thread.join(timeout=1)` 与 `SHUTDOWN_TIMEOUT=1.0` 一致。

---

## 2. 旧发现逐条状态（对源码复核，不据新叙述）

| # | 位置 | 状态 | 复核依据（冻结源码） |
|---|---|---|---|
| P1-1 | 22 §9 第 4 段 + §5.3（HMA 下 NIXL READ 失败以"成功接收"回到 Scheduler） | **FIXED** | `nixl/base_worker.py::NixlBaseConnectorWorker._handle_failed_transfer`（L2534-2552）：`if not self._is_hma_required: self._invalid_block_ids.put(...)`，`self._failed_recv_reqs.put(req_id)` 在两种分支都执行；`get_finished`（L2316+）`done_recving.update(failed_recv_reqs)` 后对失败 req `continue` 跳过 `sync_recved_kv_to_device` 与全部后处理；`Scheduler._update_from_kv_xfer_finished` → `finished_recving_kv_req_ids` → `_try_promote_blocked_waiting_request` → `_update_waiting_for_remote_kv`（L2870-2911）因 req 不在 `failed_recving_kv_req_ids`（`_handle_invalid_blocks` 未被调用，invalid 集为空）走 else 分支 `cache_blocks(...)`。新文字在 §5.3、§9、§14.1 ⑭、§14.2 ⑭、§14.3 ⑭ 与第一张图（X/Y 节点）五处一致写出该后果，并把"证明的至多是 connector 报告接收可收尾"替换了旧的错误判据。另核实该配置真实可达：`NixlBaseConnector(KVConnectorBase_V1, SupportsHMA)`，`_has_mamba` 分支 `assert self._is_hma_required`，仅 PP>1 / permute / CPU_ATTN 三处对 HMA 报错，async 标志与 HMA 无关。遗留精度问题见新 N1/N2/N3。 |
| P1-2 | 22 §2 末句、§14「它不是什么」、§14.4 两行、Related Pages（P/D E2E 路由段无 owner，页内自相矛盾） | **NOT_FIXED**（按 changelog 留待规划审批） | 工作区 22 §2 仍写"实例发现与将请求送到匹配 P/D 的控制面由 13 解释"；§14「它不是什么」仍写"不是 P/D 实例路由…归 13"；§14.4 同时保留"→ **13**"行与"建议归 **13**；基线下全库 wiki 无人展开"行；Related Pages 仍把 13 描述为"P/D 实例路由…"。13（工作区，+283 行）对 `kv_transfer_params`/`do_remote_decode`/proxy 仍 0 命中（唯一 "proxy" 命中是 Envoy 负载均衡外链）；03 只在取消/拒绝语境提到 `_with_kv_transfer_rejection_cleanup` 并回指 22，不覆盖 proxy→P→参数回传→D 这一段。`wiki/changelog.md` 新条目明确"P/D proxy…留待规划批准"，故 owner 判定属 planning；但**同页三处断言与末行"无人展开"互相矛盾**这一点是 22 自身可修的缺陷，仍未修。 |
| P1-3 | 23 §6.1（异步远端 KV 等待落在 queue 区间，页面把"数据装载"归 prefill） | **FIXED** | `Scheduler.schedule` L1161-1192：`if load_kv_async:` 置 `WAITING_FOR_REMOTE_KVS`、`step_skipped_waiting.prepend_request`、`continue`，**位于** `request.record_event(EngineCoreEventType.SCHEDULED, …)`（L1200）之前；`IterationStats.update_from_finished_request` `queued_time = scheduled_ts - queued_ts`。新 §6.1 写"直到 load 完成才发出首个 SCHEDULED event，所以这段数据等待计入 queue"，并把同步 MoRIIO READ 正确归到 SCHEDULED 之后的 prefill；§9 新增一行源码路线（`Scheduler.schedule / _update_waiting_for_remote_kv` + `MoRIIOConnector(.Worker).start_load_kv / wait_for_layer_load`，四个符号均存在）。遗留精度问题见新 N5。 |
| P1-4 | 23 §5.2（传输观测变体集无枚举依据，只讲 NIXL） | **FIXED** | 新增 8 行矩阵，逐行回源核对全部正确：`build_kv_connector_stats` 全库仅 8 处定义（base + nixl/connector、mooncake/mooncake_connector、mooncake/store/connector、multi_connector、lmcache_mp_connector、offloading_connector、hf3fs/hf3fs_connector）；`build_prom_metrics` 仅 7 处（base + 上述除 mooncake 直连）。NIXL→`NixlKVConnectorStats`/`NixlPromMetrics`；Mooncake 直连→`MooncakeKVConnectorStats`，无 prom 覆写（base 返回 None）；store→`MooncakeStoreConnectorStats`/`MooncakeStorePromMetrics`；Multi→`MultiKVConnectorStats`，`build_prom_metrics` 以 `seen_classes` 去重委托子类并 `get_kv_connector_stats` 按 `c.__class__.__name__` 分组；Offloading→`OffloadingConnectorStats`/`OffloadPromMetrics`；HF3FS→`HF3FSKVConnectorStats`/`HF3FSPromMetrics`；LMCache MP→两个 hook 都覆写且 `return None`；MoRIIO→两者都不覆写。Mooncake 直连行的 P/D 不对称与 `MooncakeConnector.get_kv_connector_stats` docstring 原文一致（"P records successful transfer latency, bytes, descriptor counts, while D only records failures"）。§9 新增 hook 路线行的 8 个类名全部存在（含 `LMCacheMPConnectorUpstream`、`HF3FSKVConnector`）。22 交给 23 的另一半"lease expiry 信号"由 §5.2 既有段落（`record_kv_expired_req` → `vllm:nixl_num_kv_expired_reqs`，7 项指标名与 `nixl/stats.py` 完全对应）承接，交接已闭合。口径问题见新 N4。 |
| P1-5 | 23 §3.1/§6.2/§7.4（ERROR/FT ABORT 终结请求进入完成 histogram，区间以 0 为基点） | **FIXED** | `Scheduler.update_from_output` L2166-2179：对 `grammar_compile_error_reqs ∪ failed_kv_load_req_ids`（`fail` 策略）`finish_requests(FINISHED_ERROR)` 并发 `EngineCoreOutput(new_token_ids=[], finish_reason=…, events=request.take_events())`；`EngineCoreProc._send_finish_outputs_to_client` 发 `EngineCoreOutput(req_id, [], finish_reason=ABORT)`（无 events），`_send_abort_outputs` 经其 ABORT 包装调用；`OutputProcessor._update_stats_from_output` 对每个输出无条件调用，`IterationStats.update_from_output` 仅按 `is_prefilling` 追加 TTFT、无 token 数判断并把 `first_token_ts` 设为 `engine_core_timestamp`；`update_from_finished_request` 对 `scheduled_ts==0` 不设防（queue = 0 − queued_ts、prefill/inference = monotonic − 0）；`PrometheusStatLogger.record` 对 `iteration_stats.finished_requests` 逐条 observe 全部 7 个延迟/计数 histogram。标签断言也成立：`vllm:request_success` 的 labelnames 含 `finished_reason`，而 `vllm:e2e_request_latency_seconds`/`vllm:request_queue_time_seconds`/`vllm:request_prefill_time_seconds` 等只用基础 labelnames，**无法按完成原因剔除**。措辞外推见新 N8。 |
| P1-6 | 05 §4 两处错误页号（`27`、编译归 `23`） | **FIXED** | 05 工作区已改为 `[[19_vllm_compilation_cudagraph_analysis|编译与 CUDA Graph]]` 与 `[[23_vllm_observability_reliability_analysis#6.3 受控恢复：只恢复可恢复的执行环境|可观测性与可靠性 §6.3]]`；23 §6.3 标题逐字匹配；`check_links --strict` stale_section=0。05 header 的"最近更新"也同步说明了本次路由修正。 |
| P2-1 | 22 §5.2 调用树 `no_forward` 漏 `wait_for_save=False` | **NOT_FIXED** | 页面 L267 仍为"`pre_forward + post_forward`"；源码 `vllm/v1/worker/gpu/kv_connector.py::ActiveKVConnector.no_forward` L110-117 明确 `self.post_forward(finished_req_ids, wait_for_save=False)`。 |
| P2-2 | 22 §6.3 MoRIIO seal 路径同步等待与未捕获 `TransferError` | **NOT_FIXED** | §6.3 失败段（L473 引用块）仍只写 `_write_worker_loop/_process_deferred_tasks`；§11 配置表已列 `transfer_timeout=30.0` 并注"`waiting_for_transfer_complete` 抛 `TransferError` 前的等待上限"，但未把该异常沿 `wait_for_save` 传出的后果写进失败段或 §12 成本表。 |
| P2-3 | 22 §6.3/§5.4 MoRIIO P 侧只接受 `FINISHED_LENGTH_CAPPED`（NIXL 兼收 STOPPED） | **NOT_FIXED** | 页面仅在 NIXL pull 段（L321）列出 `FINISHED_LENGTH_CAPPED / FINISHED_STOPPED`，MoRIIO 差异仍未指出。 |
| P2-4 | 22 §6.4 标题「请求面走 ZMQ」（bootstrap 实为 HTTP） | **NOT_FIXED** | 标题 L477 原样。 |
| P2-5 | 22 §14.4 `requires_piecewise_for_cudagraph` → 19 交接偏弱 | **NOT_FIXED** | 该行未改，19 侧亦无回指。 |
| P2-6 | 22 §14.3 与 §5 调用树重述 | **NOT_FIXED**（上轮即判为可压缩项、不计缺陷） | ⑭ 行按 HMA 更新，其余行未动。 |
| P2-7 | 23 §6.3 NaN 只核 MRV1，MRV2 为本基线默认 runner | **NOT_FIXED** | §6.3（L416）仍只写 MRV1 同步/异步两处并以"不能外推为所有 Runner"收尾；§9 路线行仍只列 `gpu_model_runner.py::GPUModelRunner._get_nans_in_logits / AsyncGPUModelRunnerOutput.get_output`，未补 `vllm/v1/worker/gpu/async_utils.py`。 |
| P2-8 | 23 §4.3 末句"必须保持有效 stats 采集"缺后果 | **NOT_FIXED** | L231 原样，未写出 `do_tracing` assert → `propagate_error` → 在途请求全失败 → `/health` 503 的链条。 |
| P2-9 | 23 §3.1 叙述顺序小误差（token 计数先于事件累加） | **NOT_FIXED** | 新段落插在表格之后，未调整原顺序。 |
| P2-10 | 23 §9 `FaultToleranceConfig` 路径缺失 | **NOT_FIXED** | "可恢复路径何时生效？"行仍只列 `vllm/config/parallel.py::ParallelConfig`；§7.1 有 `FaultToleranceConfig` 小标题但 §9 路线未补 `vllm/config/fault_tolerance.py`。 |
| P2-11 | 22 §5 首图 H→R 缺"仅 recompute"标签 | **NOT_FIXED** | 图中 `H --> R` 仍无标签（新增边 `A -->|HMA 失败例外| X`、`X -->|错误集为空| Y` 都带标签，风格已可参照）。 |

小计：P1 六条 → **FIXED 5 / NOT_FIXED 1**（NOT_FIXED 的那条 owner 判定已移交 planning）。P2 十一条 → **FIXED 0 / NOT_FIXED 11**（本轮显然只处理了 P1 清单）。

---

## 3. 新发现与回归（本轮新增文字）

**回归：0。** 新增文字未与页内既有结论、22↔23／23↔05／23↔13/26／22↔07/08 的任何一处交接冲突；被替换掉的旧错误判据没有在别处复活。**新 P0/P1：0。** 以下均为 P2。

**N1｜P2｜22 §5.3 第 5 段、§9 第 2 段、§5 首图节点 X 与 Figure spec｜HMA 触发条件写成单条件，漏 `disable_hybrid_kv_cache_manager`**

- 页面原文：「`NixlBaseConnectorWorker.__init__` 只有在 cache groups 中存在非 `FullAttentionSpec` 时才令 `_is_hma_required=True`」；图节点 X「独立 HMA 变体：含非 full-attention spec」。
- 源码：`nixl/base_worker.py` L387-393：`self._is_hma_required = (not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager and any(not isinstance(spec, FullAttentionSpec) for spec in self._layer_specs.values()))`；`_layer_specs` 来自 `kv_cache_config.transfer_groups`（`UniformTypeKVCacheSpecs` 展开）。
- 后果：读者据此判断"混合模型 ⇒ 必然命中该缺口"，但显式 `--disable-hybrid-kv-cache-manager` 的部署会回到非 HMA 的 invalid-block 路径（反而有恢复合同）。这正是运维可用的规避手段，漏掉它削弱了这段的可操作性。
- 建议：改为"cache groups 中存在非 `FullAttentionSpec` **且未设** `disable_hybrid_kv_cache_manager`"，并在 §9 一句点明关闭 hybrid manager 即退回 invalid-block 合同。

**N2｜P2｜22 §5.3 第 5 段末、§9 第 2 段、§5 首图节点 Y｜"把已承诺进度 N 改成 N−1"写成无条件**

- 页面原文：「…并把已承诺进度 N 改成 N−1 后继续计算」／「再把已承诺进度 N 改成 N−1 继续」；图 Y「继续 N → N-1」。
- 源码：`Scheduler._update_waiting_for_remote_kv` else 分支的递减包在 `if request.num_computed_tokens == request.num_tokens:` 之内（full-prompt hit 才 `num_tokens - 1`）。HMA 失败若发生在部分命中（远端只覆盖一部分 prompt）上，不会递减，而是直接以未确认有效的 computed 计数进入后续 chunked prefill。
- 后果：本身不影响"失败被当成功"的结论，但把一个条件分支写成必然，且同页上一段刚刚正确给出条件，读者会误以为 HMA 分支另有规则。
- 建议：改为"按同一条件（整 prompt 命中时）把 N 改成 N−1，部分命中则直接沿未确认的 computed 继续"。

**N3｜P2｜22 §5 首图｜HMA 变体与 full-attention 算例共用同一分配节点与同一重算位置**

- 图上 `A["分配 40 / 41 / 42｜computed = 12 是承诺"] -->|HMA 失败例外| X`，`Y --> C["重算位置 11…"]`；而 X 自称"独立 HMA 变体"，Figure spec 亦写"右侧另切换为含非 FullAttentionSpec 的 HMA 变体"。
- 问题：两个互斥配置（单组 full-attention 12-token 算例 vs 含非 full-attention spec 的 hybrid 模型）在图上共享同一分配前提与同一"位置 11"重算终点。正文虽已声明"不是同一配置"，图仍可被读成同一次运行的第三条分支。
- 建议：把 X/Y 放进独立 subgraph（如 `D'：另一配置的同类请求`），或让 Y 指向一个中性节点（"按正常路径继续计算"）而不是复用 C；同时给 H→R 补 P2-11 要求的"仅 recompute"标签，两处一并处理。

**N4｜P2｜23 §5.2 矩阵前后文｜矩阵只列 8 个实现，收尾句读作穷尽，实际另有 6 个注册名两个 hook 都不覆写；且其中几个仍会发 KV events**

- 页面原文：「固定基线中的 hook 矩阵把这个差异具体化」…「LMCache MP 与 MoRIIO 则连这两个 hook 都不产出对象，必须回到请求区间、实现日志或 connector 自身出口取证」。
- 源码：`KVConnectorFactory` 注册 16 个名字；除矩阵覆盖的 8 个实现外，`LMCacheConnectorV1`、`FlexKVConnectorV1`、`SimpleCPUOffloadConnector`、`DecodeBenchConnector`、`ExampleConnector`、`ExampleHiddenStatesConnector` 同样两个 hook 都不覆写（`grep -rn "def build_kv_connector_stats\|def build_prom_metrics"` 只命中 base + 8 个实现文件）。反向地，`take_events` 覆写点包括 `lmcache_connector.py`、`flexkv_connector.py`、`simple_cpu_offload_connector.py`、`offloading_connector.py`、`lmcache_mp_connector.py`、`multi_connector.py`、`mooncake/store/connector.py` —— 即"无 transfer stats"并不等于"无任何可观测出口"（KV events 仍可能有）。
- 后果：读者可能把矩阵当成注册名全集，进而对 LMCache v1 / FlexKV / SimpleCPUOffload 得出错误预期（既可能误以为它们有 NIXL 式遥测，也可能误以为它们完全不可观测）。这正是 P1-4 要求的"枚举依据"应当封住的口子。
- 建议：矩阵前加一句范围说明（"覆盖 22 展开的六条数据面 + Multi/LMCacheMP；其余注册名两个 hook 都不覆写"），并在收尾句区分"无通用 connector 遥测"与"无 KV cache 事件"两件事；§5.3 已经拥有后者，正好互指。

**N5｜P2｜23 §6.1 第 1 段｜把 capacity/deferred 与"异步远端 KV load"并列为不同来源，实际异步等待者正是 `deferred` 这一标签**

- 页面原文：「queue 上升通常说明首次调度前等待增长，既可能来自 capacity/deferred waiting，也可能来自 P/D consumer 的**异步**远端 KV load…应同时对照 waiting reason、KV usage 与 connector transfer stats」。
- 源码：`Scheduler.schedule` 的 async 分支把请求放进 `step_skipped_waiting`（L1166）→ 合并进 `self.skipped_waiting`（L1245）；`make_stats` 输出 `num_skipped_waiting_reqs=len(self.skipped_waiting)`；`PrometheusStatLogger.record` 把它 set 到 `vllm:num_requests_waiting_by_reason{reason="deferred"}`，`num_waiting_reqs` 才是 `capacity`。该 gauge 的 documentation 原文即把 "KV transfer" 列为 deferred 的成因之一。
- 后果：这是 05 §"`deferred` 可能含 KV transfer"提示的机制依据，也是本页作为 05 机制 owner 必须给出的那一条。按现在的并列写法，运维会把 `deferred` 理解为 LoRA/blocked，而去别处找 KV transfer，等于丢掉唯一能直接定位的 gauge。
- 建议：改为"这类等待**就计入** `num_requests_waiting_by_reason{deferred}`（与 LoRA 预算、blocked status 同标签），不会出现在 `capacity`；先用 deferred 与 connector transfer stats 交叉，再看 KV usage"。

**N6｜P2｜23 §7.1｜新拥有的 `KVEventsConfig` 未进配置/成本表，replay buffer 的内存代价无法据页面估算**

- 现状：§5.3 按名提到 `buffer_steps`、`max_queue_size`、`hwm`、`replay_endpoint`、`topic`，但 §7.1 逐类枚举字段的表格只覆盖 `ObservabilityConfig`（13/13）、`ParallelConfig`（2/61）、`FaultToleranceConfig`（1/1），没有 `KVEventsConfig`。
- 源码：`vllm/config/kv_events.py::KVEventsConfig` 共 8 字段，默认 `enable_kv_cache_events=False`、`publisher=None→zmq/null`、`endpoint="tcp://*:5557"`、`replay_endpoint=None`、`buffer_steps=10_000`、`hwm=100_000`、`max_queue_size=100_000`、`topic=""`。`ZmqEventPublisher` 的 `deque(maxlen=buffer_steps)` 持有 msgpack 后的 payload，是常驻内存成本。
- 后果：本页的分工是"§5 讲机制、§7 讲配置与成本"，KV events 成为本页所有物后，成本侧出现空洞：读者无法知道默认就有 1 万批次的 replay buffer 与 10 万项队列。
- 建议：§7.1 补一张 `KVEventsConfig` 8 字段小表（含上列默认值与"buffer_steps × 批 payload 为常驻内存"一句），与 §7.3 sampling/§7.4 stale 的既有口径衔接。

**N7｜P2｜跨页（owner 在 07，22↔07 交接）｜07 仍把异步 load 失败合同写成完整，未带 HMA 例外或回指**

- 07 状态表 `WAITING_FOR_REMOTE_KVS` 行：「等待 worker connector 的 finished/failed 信号。ready 后缓存有效块，失败时截到有效前缀或释放」；07 §500 另有正确的"这个 computed 值在 transfer ready 前不能当作已加载成功的 KV"。
- 问题：22 现在明确写出 HMA 下失败以 `finished_recving` + 空 invalid 集返回、Scheduler 走正常缓存分支；07 的"失败时截到有效前缀或释放"在该配置下不成立。07 本轮也被改动（+95 行），是同批修复的一部分。
- 建议：07 该行加半句"（NIXL HMA 例外：失败不产出 invalid block，见 22 §5.3）"，或在 07 §500 末尾回指 22 §5.3。机制仍归 22，07 只需一处限定，避免下次基线变更时两页各自漂移。

**N8｜P2｜23 §6.2 末段、§7.4 新增段｜"负 queue"被同时挂到 ERROR 与 ABORT 两条路径，ABORT 无 events 时应为 0 而非负**

- 页面原文（§6.2）：「即便 token 为空、events 不完整，也可能进入完成 histogram，并出现 §3.1 所述的负 queue 或超大 prefill/inference」；（§7.4）「显式 ERROR/ABORT 输出…产生负 queue、接近 monotonic 纪元的 prefill/inference」。
- 源码：负 queue 需要 `queued_ts != 0`，即该请求此前收到过带 QUEUED event 的输出；Scheduler 的 `FINISHED_ERROR` 输出带 `events=request.take_events()`（含 QUEUED）故为负，而 `_send_finish_outputs_to_client` 的 ABORT 输出**不带 events**，从未调度过的请求 `queued_ts` 仍为 0 → `queued_time = 0`（失真但非负）。
- 说明：§3.1 的原始表述是准确的（把负值归给 ERROR 路径，对 ABORT 只说"可能产生失真的完成区间"），后两处复述把两条路径合并后略微外推。
- 建议：§7.4 与 §6.2 复述时把"负 queue"限定为"带 QUEUED event 的 ERROR 终态"，ABORT 写成"queue 为 0 的零基点样本"。

---

## 4. 本轮抽查锚点（新增文字对应，全部回冻结源码）

### 22（9 组）

| # | 锚点 | 页面陈述 | 源码结果 |
|---|---|---|---|
| 1 | `nixl/base_worker.py::NixlBaseConnectorWorker.__init__`（L387） | HMA 由非 full-attention spec 决定 | 条件成立但还 AND `not disable_hybrid_kv_cache_manager`（N1） |
| 2 | 同文件 `_handle_failed_transfer`（L2534） | HMA 不写 invalid、两分支都入 `_failed_recv_reqs` | ✓（且仅在 `_recving_metadata` 仍有 meta 时入队，页面已用 multi-read 句覆盖） |
| 3 | 同文件 `get_finished`（L2316） | 无条件合入 `done_recving`、跳过后处理 | ✓（`continue` 前 warning "Skipping KV post-processing for failed request"） |
| 4 | `Scheduler._update_waiting_for_remote_kv`（L2870） | 走正常缓存分支、N→N−1 | ✓，递减有 full-hit 条件（N2） |
| 5 | `Scheduler._handle_invalid_blocks`（L3081） | 无 invalid 即不进入该流程 | ✓（由 `kv_connector_output.invalid_block_ids` 为真才调用） |
| 6 | `Scheduler._update_requests_with_invalid_blocks` | `num_computed_tokens = idx * block_size` | ✓（算例 41→4 成立） |
| 7 | `nixl/pull_scheduler.py::get_num_new_matched_tokens` | 远端 prefill 异步 | ✓ `return count, True`，与 HMA 无关 → 缺口确在异步晋升路径上 |
| 8 | `nixl/connector.py::NixlBaseConnector` + `_has_mamba` | HMA 为真实支持组合 | ✓ `SupportsHMA`；`assert self._is_hma_required` |
| 9 | `vllm/v1/worker/gpu/kv_connector.py::ActiveKVConnector.no_forward` | 调用树"pre+post" | 源码为 `post_forward(finished_req_ids, wait_for_save=False)`（P2-1 仍在） |

### 23（18 组）

| # | 锚点 | 结果 |
|---|---|---|
| 1-8 | 8 个 connector 的 `build_kv_connector_stats` / `build_prom_metrics` 覆写点与返回值 | 矩阵 8 行全部正确（细节见 P1-4 行） |
| 9 | `base.py::KVConnectorBase_V1.build_kv_connector_stats / build_prom_metrics` | 两者 `return None` ✓（"继承基类 None"成立） |
| 10 | `multi_connector.py::MultiConnector.build_prom_metrics / get_kv_connector_stats` | `seen_classes` 去重 + `c.__class__.__name__` 分组 ✓ |
| 11 | `mooncake_connector.py::MooncakeConnector.get_kv_connector_stats` docstring | P 记成功、D 只记失败 ✓ |
| 12 | `Scheduler.schedule` async 分支与 `record_event(SCHEDULED)` 相对位置 | `continue` 在前 ✓ |
| 13 | `IterationStats.update_from_output / update_from_finished_request` | 无 token 判定即登记 TTFT；`scheduled_ts==0` 无防护 ✓ |
| 14 | `Scheduler.update_from_output` 的 `FINISHED_ERROR` 输出 | `new_token_ids=[]` + `events=take_events()` ✓ |
| 15 | `EngineCoreProc._send_finish_outputs_to_client / _send_abort_outputs` | `EngineCoreOutput(req_id, [], finish_reason=ABORT)`、无 events ✓ |
| 16 | `PrometheusStatLogger.record` + 各 histogram/counter labelnames | 逐条 observe；只有 `vllm:request_success` 带 `finished_reason` ✓ |
| 17 | `vllm/distributed/kv_events.py`（事件类、factory、`ZmqEventPublisher`） | `BlockStored` 13 字段含 session_id 且 docstring 明言非独占所有权；`AllBlocksCleared` 仅 `reset_prefix_cache` 成功后 append；factory 对 None/禁用/`"null"` 返回 `NullEventPublisher`；端口按 DP rank 偏移、`publish` 写 `data_parallel_rank`、线程发 `(topic, 8B seq, msgpack)`、ROUTER replay 走 `deque(maxlen=buffer_steps)`、`Queue(maxsize=max_queue_size)` 阻塞 put、`SHUTDOWN_TIMEOUT=1.0` + `join(1.0)`、docstring 有 at-least-once 措辞 ✓ |
| 18 | `KVEventsConfig.__post_init__`；`BlockPool.take_events / reset_prefix_cache`；`Scheduler.update_from_output` 合并发布；`maybe_convert_block_hash` + `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES`（默认 1，`& ((1<<64)-1)`）；`examples/features/kv_events/kv_events_subscriber.py`；`docs/deployment/integrations/llm-d.md` L11（"reads vLLM's KV-cache events and routes each request to the replica that already holds its prefix"） | 全部与 §5.3 陈述一致 ✓（配置默认值未进 §7 → N6） |

---

## 5. 跨页复核

- **22 ↔ 23（两页均改）**：22 §14「它不是什么」与 Related Pages now 指向 23 §5.2（connector 遥测枚举）与 §5.3（KV 事件发布与外部前缀路由），23 §5.2 矩阵 + §5.3 确实拥有这两块，`VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES` 在 22 §11 环境变量表标"→ **23**"、在 23 §5.3 落地，双向一致；22 交出的"lease expiry 信号"由 23 §5.2 既有段落承接（`vllm:nixl_num_kv_expired_reqs`）。无重复拥有、无矛盾。唯一口径缺口是 N4。
- **23 ↔ 05**：05 §4 两处指向已修为带锚点 wikilink，锚点解析通过；05 的 `deferred` 提示与 23 §6.1 新段方向一致，但 23 未点明 deferred 即该 gauge（N5），交接仍差一句。
- **23 ↔ 13 / 26**：13（+283 行）新增内容集中在 DP 消息通道与 Coordinator，26（+48 行）集中在 RPC 配对与首错/超时，两页仍把健康信号与 FT 状态机交回 23；23 §6.2/§6.3 未与之冲突。
- **22 ↔ 07 / 08**：07 新增内容是 SchedulerOutput 22 字段，明确把 `kv_connector_metadata` 的内容与传输交给 22，无重复；08 全页 0 处 KV events 内容，23 接手 KV 事件发布不产生双主。唯一遗留是 07 状态表的失败合同缺 HMA 限定（N7）。

## 6. 未验证推测（不计入发现）

- 上轮两条（FT `_clean_worker_state` 不清 connector 在途状态；NIXL push 的 P 侧 WRITE 失败与 PUSH_REG 丢失同类）本轮未新增证据，仍需故障注入。
- HMA 缺口的实际触发频率（需要 hybrid/Mamba 模型 + NIXL + 真实传输失败）未做注入验证；本轮只确认路径可达且无保护。
- prometheus_client 对负值 `observe` 的内部行为仍属第三方，未核验。
