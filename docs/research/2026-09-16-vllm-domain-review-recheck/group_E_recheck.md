# Group E 复核（第 2 轮）：11 MRV1 / 12 MRV2 / 19 编译与 CUDA Graph

- 评审人：独立 reviewer（非作者）。只报告，不修改任何页面。
- 冻结基线：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`，checkout `/Users/suhaibo/97-llm/vllm` HEAD 一致（`2026-09-06 17:54:32 -0700` = 2026-09-07 UTC），未移动 HEAD。
- 本轮方式：先读 `git diff`（三页均未提交），再对**新增文字的每条断言**回源码核对；旧发现逐条按源码重判，不按新措辞判。
- 附带跑了 T0 四项（只读）：`check_links --strict` 452 页 broken/ambiguous/bare_index/stale_section/orphans 全 0；`check_math/check_markdown/check_assets --changed --strict` 39 文件 0 错 0 警。新增的三个深链锚点（06 §4.1、17 §6.2、08 §5.1）都解析成功。
- 本轮仍未运行 vLLM 的 GPU/分布式测试。

---

## 1. 判定行

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 11_vllm_model_runner_v1_analysis | pass | pass | pass | transform, timing | pass（新 Core 树与时序图重放一致） | 12/12 | **ACCEPT WITH FIXES** | feature: pass §1.4（F4/F18 已闭合：async⇒batch queue、立即提交采样、MRV1 无正常 None 返回）；余 P2：F15 段号、N1「未自定义 Scheduler」条件 |
| 12_vllm_model_runner_v2_analysis | pass | pass | pass | transform, layout, timing | pass | 12/12 | **ACCEPT WITH FIXES** | feature: pass §1.5（F4 已闭合，含 AsyncOutput 与线程组织解耦）；余 P2：F8、F16、F17 |
| 19_vllm_compilation_cudagraph_analysis | pass | **pass（F1 已修，hop 真实存在）** | pass | **FAIL §1.2 图 1、§6.5 图 2** | pass §7.3 新重放；**FAIL §1.2/§6.5** | 14/14（新增锚点全对） | **REJECT** | feature: pass §7.3（MRV1 派发变体依据已补，唯缺两代对照表）；两条 P1 图错未动（F2/F3），并与本轮刚加强的 12 §2.6/§2.9 直接冲突；另 8 条 P2 未动 |

Feature 复判：

- 11：`feature: pass §1.4`。默认异步⇒`step_with_batch_queue`、sample 紧随 execute 非阻塞提交、grammar 延期/pooling/零 token/EC producer 三条边界、MRV1 `sample_tokens` 无正常 `None` 返回——四点全部与源码一致，时序图已改为「固定 async + mp + PP=1」并自证范围。
- 12：`feature: pass §1.5`。另加两条本轮新写且正确的边界：`AsyncOutput` 与 `async_scheduling` 解耦、mp/UniProc 两种物化通路。
- 19：`feature: pass §7.3`。MRV1 派发的选择点、`invalid_modes`、`force_eager`、DP 重派发、五条短路与两处断言全部核对无误；仍缺 F6 要求的两代对照（LoRA case 集合等）。但页面另有两张图携带 P1 事实错误，故整页 REJECT。

---

## 2. 旧发现逐条状态

| 编号 | 严重度 | 位置 | 状态 | 依据 |
|---|---|---|---|---|
| F1 | P0 | 19 §7.3/§7.5 MRV1 派发入口 | **FIXED** | 入口改为 `gpu_model_runner.py::GPUModelRunner._determine_batch_execution_and_padding` → 内部闭包 `dispatch_cudagraph` → `CudagraphDispatcher.dispatch`（源码 4040/4085），SP padding、`disable_full=use_cascade_attn or has_encoder_output`、`force_eager⇒valid_modes={NONE}`、DP `coordinate_batch_across_dp` 后按 `valid_modes={CUDAGraphMode(synced)}` 重派发并断言 token 数——五项逐条核对全部一致；调用树与 §12 读码路线同步更新，`dispatch_cudagraph` 明确标注为闭包 |
| F2 | P1 | 19 §1.2 图 1 | **NOT_FIXED** | 图与正文未改：仍写「采样出的新 token 写回同一 `InputBuffers` storage」、边 `SMP -->\|sampled token ids 同址写回\| BUF`、`BUF -->\|下一步的 num_tokens 与 num_reqs\| RUN`。源码 `gpu/input_batch.py::post_update` 写的是 `last_sampled_tokens / all_token_ids / total_len / num_computed_tokens`（均属 `RequestState`）；`InputBuffers.__init__` 只有 `input_ids/positions/is_padding/query_start_loc/seq_lens/dcp_local_seq_lens`，无 sampled 字段 |
| F3 | P1 | 19 §6.5 图 2 | **NOT_FIXED** | 图未改：`B(resolve) -->\|resolved CUDAGraphMode\| C(首次 dummy forward)`。源码 `engine/core.py::_initialize_kv_caches` 先 `determine_available_memory`（→`profile_run`→首次 trace/compile），`resolve_cudagraph_mode_and_sizes` 只在 `initialize_kv_cache` 路径调用（MRV1 `gpu_model_runner.py:7279`、MRV2 `gpu/model_runner.py:664`），即 C 在 B 之前 |
| F4 | P1 | 11 §1.4、12 §1.5 | **FIXED** | 两页都加了「async（默认）⇒ 容量≥2 ⇒ `step_with_batch_queue`」、「sample 不等 execute 返回 None」；容量规则与 `VllmConfig.max_concurrent_batches`（MRV2 `pp+1`；MRV1 `pp<=1→2` 否则 `pp`；非 async→`pp`）一致；`step_with_batch_queue` 的立即 `get_grammar_bitmask`+`sample_tokens(non_block=True)`、延期分支、pooling/零 token/EC producer 用 exec_future、None 兜底全部核对一致 |
| F5 | P1 | 19 §4.3、§9.1 | **NOT_FIXED** | §4.3 仍写「**硬约束**：启用 graph 时 `CudagraphDispatcher._compute_bs_to_padded_graph_size` 会……直接抛 `ValueError`……单点特化和 capture 阶梯必须对齐」，§9.1 `compile_sizes` 行仍写「不得被 capture padding 改写，否则 `ValueError`」，均未限定 MRV1。源码 `CudagraphDispatcher(` 仅在 `gpu_model_runner.py`（MRV1）、`spec_decode/llm_base_proposer.py`、`spec_decode/extract_hidden_states.py` 实例化；MRV2 的 `gpu/cudagraph_utils.py` 无等价校验。默认 MRV2 下这条约束不被强制 |
| F6 | P1 | 19 §7.3 枚举依据 | **PARTIAL** | 已补：五条短路（`keys_initialized`/`cudagraph_mode==NONE`/`max_size is None`/`num_tokens>max_size`/`allowed_modes<={NONE}`）、`assert NONE in allowed_modes`、`assert len(allowed_modes)>=1`、caller 注入的 valid/invalid、DP 重派发——全部与 `cudagraph_dispatcher.py::dispatch` 一致。仍缺：两代对照表。MRV1 `_get_lora_cases()` 未特化时为 `[max_loras+1]`（**无 0 case**），MRV2 `gpu/lora_utils.py::get_lora_capture_cases` 为 `[0, max_loras+1]`；§6.2 只写了 MRV2 一侧，§9.2 ⑰ 的键集合公式仍是旧表述 |
| F7 | P2 | 19 页头日期 | **FIXED** | 改为「`main` 快照，2026-09-07 UTC」，与 index、11、12 一致 |
| F8 | P2 | 12 §4 CoW helper | **NOT_FIXED** | 仍写「旧稿写的"独立 head groups"在源码中找不到，按上述规则更正」。`tests/v1/worker/test_attn_utils.py:246::test_copy_kv_cache_blocks_separate_head_groups` 存在（注释：LHBNC 下一个 block 的字节散落在 L*H 个区域），且被 11 §3.3 引用 |
| F9 | P2 | 19 §7.3 MRV2 blocker 复述 | **NOT_FIXED** | 仍写「部分 DBO 组合」「ngram 类投机」。`_get_dbo_unsupported_features`：`VLLM_USE_V2_MODEL_RUNNER is None` 时直接 `return ["dual batch overlap"]`——自动选择场景下 DBO 一律 blocker；`_get_v2_model_runner_unsupported_features` 还有 `method not in (eagle,eagle3,mtp,dflash,dspark,extract_hidden_states)` 的兜底（如 `draft_model`）与 `parallel_drafting` 项 |
| F10 | P2 | 19 §6.2 子句计数 | **NOT_FIXED** | 仍写「三条 `>=` ……两条 `==`」，而同一张表列出三行「必须相等」（`uniform_token_count`、`num_active_loras`、`num_ubatches`）。页面自相矛盾 |
| F11 | P2 | 19 §9.2 ⑭ | **NOT_FIXED** | ⑭ 仍写「draft mode 恒被收窄为 `FULL_DECODE_ONLY` 或 `NONE`」；同页 §6.4 已正确写明 autoregressive 的 prefill manager 原样使用目标 mode，只窄化 decode manager |
| F12 | P2 | 19 §9.1、§9.2 ① | **NOT_FIXED** | 仍写 `max_cudagraph_capture_size = min(max_num_seqs * decode_query_len * 2, 512)`、interactivity「1..32 全覆盖」。`_set_cudagraph_sizes` 还有 `max_cudagraph_capture_size = min(max_num_batched_tokens, ...)`；interactivity 是 `range(1, min(max,32)+1)` **再叠加** 8/16 步长网格；另追加 `max_num_tokens`（在上限内）与 `uniform_decode_sizes`（`decode_query_len>1`），SP+TP>1 还会 `update_sizes_for_sequence_parallelism` 截断 |
| F13 | P2 | 19 §1.4 ⑪ | **NOT_FIXED** | 条件仍只写 `is_cuda_alike()` 且 `cudagraph_mode != NONE`。`Worker.determine_available_memory` 在显式 `cache_config.kv_cache_memory_bytes` 时只跑 `profile_run` 并直接返回（不做 profiling）；`EngineCore._initialize_kv_caches` 在无 KV cache 或 `VLLM_ELASTIC_EP_SCALE_UP_LAUNCH` 时整段跳过。12 §2.9 已写对 |
| F14 | P2 | 19 §7.2 | **NOT_FIXED** | 仍写「有**三**种走法」随后列四种（mode 不匹配 / hit / miss / 无 forward context，末者被称作「第三种」）；并仍写 guard「只由 MRV1 的 `gpu_model_runner.py` 在 capture 结束后调用」。实际 `set_cudagraph_capturing_enabled(False)` 在 `gpu_model_runner.py:6870`（`profile_cudagraph_memory`）与 `:6991`（`capture_model`）两处 |
| F15 | P2 | 11 §3.4 | **NOT_FIXED** | 第 563 行仍写「15 §7.4：`update_from_output()` 交 `ECConnectorBase.update_connector_output()`」。15 §7.4 是 `prompt_embeds`，EC transfer 在 15 §7.7 |
| F16 | P2 | 12 §2.3、§6.1 | **NOT_FIXED** | 仍写「异步 MRV2 为 PP size + 1，否则为 PP size」。`max_concurrent_batches` 在 async+MRV1+PP≤1 时为 2。本轮 06 §4.1 已给出完整五行表，页面 §6.1 的派生属性契约与之不一致 |
| F17 | P2 | 12 §3.2 | **NOT_FIXED** | 调用树仍缺 `_set_active_loras`、`pp_handler.broadcast`、`prompt_logprobs_worker.compute_prompt_logprobs`（后两者在 `gpu/model_runner.py::sample_tokens` 中确实位于 sample 之后、`AsyncOutput` 之前），也没有 elided 标记 |
| F18 | P2 | 11 §1.4 第 4 点 | **FIXED** | 改为「Core 保留该通用兜底，**但 MRV1 的 `sample_tokens()` 没有正常返回 None 的分支**」。源码：MRV1 `sample_tokens` 空状态返回 `ModelRunnerOutput.with_kv_conn_output_only(...)`（→`EMPTY_MODEL_RUNNER_OUTPUT`），全函数无 `return None`；MRV2 才有 `return None`。§3.3 新增段落对 docstring 与实现的不一致、以及两个 PP 测试「`assert output in (EMPTY_MODEL_RUNNER_OUTPUT, None)` 断言宽松」的说明也逐字核对无误 |

统计：FIXED 4（F1、F4、F7、F18）｜PARTIAL 1（F6）｜NOT_FIXED 13｜SUPERSEDED 0。
按页：11 → FIXED 2（F4、F18）、NOT_FIXED 1（F15）；12 → FIXED 1（F4 本页份）、NOT_FIXED 3（F8、F16、F17）；19 → FIXED 2（F1、F7）、PARTIAL 1（F6）、NOT_FIXED 9（F2、F3、F5、F9～F14）。

---

## 3. 新增文字的核对结果（regression hunt）

**全部新增断言逐条回源核对，未发现新的 P0/P1。** 已核对且正确的关键点：

- MRV1 派发（19 §7.3 表 + §7.5 树）：`_is_uniform_decode` 用未 padding 的真实值 → `_pad_for_sequence_parallelism`（`enable_sp and tp_size>1` 时 `round_up`）→ 首次 `dispatch_cudagraph(num_tokens_padded, disable_full=...)` → SP 断言 → DP → 重派发 → 断言；`has_encoder_output = model_config.is_encoder_decoder and num_encoder_reqs > 0`，`num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs)`；`force_eager=is_profile or (cudagraph_runtime_mode == NONE)`（`_dummy_run` 调用点）；`skip_compiled=has_encoder_input` 确实是 MRV1 `execute_model` 在 `set_forward_context` 上另设（`gpu_model_runner.py:4542`）。
- DP 语义：`dp_utils.py::_run_ar` 四行 int32、`_post_process_cudagraph_mode` 取 min（0=NONE/1=PIECEWISE/2=FULL）、`should_dp_pad = synced != 0 or should_ubatch` 时补到 max——与新表第 4 行逐字一致。
- 新数值重放（我自己重放）：SP+TP=2 时 3→`round_up(3,2)`=4；`force_eager` 使 `allowed_modes={NONE}`，命中 `allowed_modes <= {NONE}` 短路，返回 `BatchDescriptor(4)`——页面「NONE descriptor 仍携带 4」正确。DP 例子：本地 `FULL/4` 与 `PIECEWISE/2` → `min(2,1)=1`(PIECEWISE)、`should_dp_pad=True` → 两 rank 都补到 4 → 以 `valid_modes={PIECEWISE}` 重派发得容量 4 的 relaxed 键——正确；若该模式无键则 `assert NONE in allowed_modes` 报错，页面表述正确。
- 测试断言：`TestCudagraphDispatcher.test_dispatcher` 第 4、5 段确实检查 `invalid_modes={FULL}` 的回退与 `valid_modes={NONE}` 的强制关闭（`tests/v1/cudagraph/test_cudagraph_dispatch.py:182/198`）。11 引用的两个 PP 测试、06 引用的两个 config 测试、`test_engine_core_concurrent_batches` 全部存在。
- 默认执行轴（11/12/19 与 06/07）：`__post_init__` 的 async 解析、`get_scheduler_cls`、`max_concurrent_batches`、`EngineCore.__init__` 的 `batch_queue_size>1 ⇒ step_fn=step_with_batch_queue`、`step()` 的「先等 execute future，None 才同步 sample」——四页表述互相一致且与源码一致。06 §4.1 已成为唯一 owner 且五行容量表正确；07 §2 已改标「同步教学分支……不是当前缺省」。
- 输出物化通路：`WorkerProc.handle_output`（async→`async_output_queue`，否则直接 `enqueue_output`）、`enqueue_output` 内 `get_output()`、`UniProcExecutor.collective_rpc`（同步路径直接 `get_output`，`non_block` 返回 `AsyncOutputFuture`）——11/12 新段落全部正确。MRV2 `AsyncOutput` 无条件构造（`gpu/model_runner.py:1974`），页面「即使 `async_scheduling=False` 也如此」正确。
- 19 §6.1 新增 Marlin 交界段：`WorkspaceManager` 按 `(ubatch, lane)` 管理（`v1/worker/workspace.py:47-68`）；Marlin kernel 自持 `self.workspace = marlin_make_workspace_new(device, existing=...)`；`g_idx_sort_indices` 由 `replace_parameter(..., prefer_copy=True)` 注册为 layer Parameter。与 17 §6.2、20 的新段落三方一致，无 owner 冲突。

### 新发现

**N1（P2，11 §1.4）——把「未自定义 Scheduler」写成了 async 解析与 `step()` 的必要条件**
- 页面原文：「普通生成、兼容配置、PP=1 且未自定义 Scheduler 时，`async_scheduling=None` 仍解析成 True」；「显式关闭 async 或自动回落到 False，且 PP=1、未自定义 Scheduler 时，Core 才走 `EngineCore.step()`」。
- 源码证据：`vllm/config/vllm.py::__post_init__` 的 async 解析分支完全不读 `scheduler_cls`；`SchedulerConfig.get_scheduler_cls()` 只决定类（自定义时仅 `warning_once`）；`EngineCore.__init__` 的 `step_fn` 只看 `batch_queue is None`，即只看 `max_concurrent_batches`。自定义 Scheduler + async=False + PP=1 仍走 `step()`。同轮 06 §4.1（owner）措辞正确，11 与它不一致。
- 修改建议：把「未自定义 Scheduler」只挂在「因此使用 `AsyncScheduler`」这半句上，两处涉及 `step()`/容量的条件里删掉它。

**N2（P2，19 §7.5 Core 调用树）——把队列消费写成每次调用的无条件末步**
- 页面原文：树末行「`` `-- [消费最旧项] future.result；处理 abort；scheduler.update_from_output ``」，正文未给条件。
- 源码证据：`EngineCore.step_with_batch_queue` 在 `len(batch_queue) < batch_queue_size and (model_executed or self.scheduler.has_requests())` 时 `return None, model_executed`，本次调用不消费任何结果。11 §1.4 与 12 §1.5 都写了这个条件，19 缺。
- 修改建议：给该行加条件标注（如 `[队列已满或无更多可调度工作]`），或一句「未满且仍有工作时本次调用直接返回」。

---

## 4. E2E / 跨页复核

| 轴 | 结论 |
|---|---|
| 默认 step 路径（11/12/19 vs 06/07） | **一致**。06 §4.1 唯一拥有 async 解析 / Scheduler 类 / 容量 / `step_fn`，表内五行与源码逐行一致；07 §2 已标注为同步教学分支；11/12/19 只接续其后果并各自回链。唯一瑕疵是 N1 的条件措辞。 |
| sampled token 落点（19 vs 12） | **仍矛盾（F2）**。12 §2.6 与源码一致（`postprocess_sampled → post_update` 写 `RequestState`，下一步 `combine_sampled_and_draft_tokens` 才拷进 `input_buffers.input_ids`），19 §1.2 图 1 仍写回 `InputBuffers`。本轮 12 的措辞更明确，冲突因此更刺眼。 |
| 启动顺序（19 vs 12） | **仍矛盾（F3）**。12 §2.9 不仅正确，本轮还补齐了 `kv_cache_memory_bytes`/无 KV/elastic EP 三处跳过；19 §6.5 图 2 仍把 resolve 放在首次编译之前并标成其输入。 |
| pass/kernel owner（19 vs 20/21） | **一致**。20 明确「Marlin workspace 不是 19 的通用 `WorkspaceManager` 槽」、reload 地址规则归 17 §6.2、融合开关取值归 21 §9.1、range 端点消费归 19 §4.2；19 §6.1 新段落与之对称。无第二份矩阵。 |
| 11 → 15 段号 | **仍错（F15）**：应为 15 §7.7（EC transfer），页面写 §7.4（`prompt_embeds`）。 |
| 19 §7.3 vs 12 §5.1 的 blocker 清单 | **仍是弱化的第二份矩阵（F9）**，建议按上轮建议直接改成链接。 |

---

## 5. 结论与剩余阻塞

- 11、12：本轮的 P1（F4/F18）已按源码闭合，新增文字未引入新错误，剩余全是 P2。可在修掉 F15/F16/F17/F8/N1 后收。
- 19：P0（F1）已修得很干净（含 §1.3、§7.3、§7.5、§9.2、§12 五处同步），但**上轮两条 P1 图错（F2 sampled token 落点、F3 启动顺序）一个字都没动**，且都与同轮加强过的 12 直接矛盾；F5（`compile_sizes` 硬约束未限定 MRV1）也未动。这三条是剩余阻塞，须先改这三条再复审。
- 建议的最小收口顺序：19 图 1 回边 → 19 图 2 时间轴 → 19 §4.3/§9.1 的 MRV1 限定 → F6 的两代对照表 → 其余 P2（F9～F14、F15～F17、F8、N1、N2）。
