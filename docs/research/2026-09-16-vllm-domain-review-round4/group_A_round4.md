# Group A 第四轮独立验收：02 架构 / 06 Engine / 07 Scheduler / 08 KV / 11 MRV1 / 12 MRV2 / 19 编译与 CUDA Graph

- 评审人：独立 reviewer（非作者）。**只读**：未修改任何 `wiki/`、`docs/`、`skills/` 或源码 checkout 文件。
- 冻结基线：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`，checkout `/Users/suhaibo/97-llm/vllm`，HEAD 全程未移动（`git log -1` 复核，工作区仅有未跟踪 `artifacts/`）。
- **本轮方法学的关键改进**：把上一轮四个 reviewer 保存的 diff（`scratchpad/diff02|06|07|08.txt`、`scratchpad/review3/diff11|12|19.txt`）用 `git apply` 打回 `HEAD` 版本，重建出 **round-3 当时的页面状态**，再与当前工作区逐页 `diff -u`，从而把「最终修复轮」的改动从 round-2+3 的累计 diff 中**精确切出来**。重建产物在 `scratchpad/review4/r3tree/`，逐页最终 diff 在 `scratchpad/review4/final_*.diff`。
- 最终修复轮实际规模（仅本轮新增/重写）：02 `+8/−5`、06 `+8/−7`、07 `+12/−5`、08 `+2/−2`、11 `+4/−4`、12 `+2/−2`、19 `+8/−6`。**这 44 行新增文字逐句回源码核验，无一遗漏**。
- 未运行 vLLM、GPU、多进程服务或故障注入。

---

## 0. 判定行

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 02_vllm_architecture_overview_analysis | pass | pass | pass（本轮无删除） | layout, timing | pass（本轮无新数值例） | 14/14 新符号一致 | **PASS** | `architecture: pass item 5`、`pass item 7`（round-3 两条 FAIL 的原因全部消除，且重写句无新错）；item 1/2/3/4/6/8 沿用 round-3 pass。新增 P0/P1 = 0 |
| 06_vllm_engine_architecture_analysis | **条件 pass** | pass | pass | timing | pass | 6/6 | **ACCEPT** | `feature: pass`。新增 P2 ×1（§4.1 新段落把「因此…缺省」结论吞进不相关段落）、P3 ×1 |
| 07_vllm_scheduler_analysis | pass | pass | pass | timing | **pass（新表 2→1→0 逐格重放成立）** | 8/8 | **ACCEPT** | `feature: pass §8.1.1–§8.1.4、§8.3`。round-3 唯一 P1（§8.4）已修且源码正确；新增 P2 ×2 |
| 08_vllm_kv_cache_management_analysis | pass | pass | pass | — | pass | 4/4 | **PASS** | `feature: pass`。两条 P2 全修；新增 P3 ×2 |
| 11_vllm_model_runner_v1_analysis | pass | pass | pass | — | pass | 5/5 | **ACCEPT WITH FIX** | `feature: pass §1.4`。N1 已修，但改写留下一句**未删干净的旧句**并因此丢掉 PP=1 前提（P2） |
| 12_vllm_model_runner_v2_analysis | pass | pass | pass | — | pass | 3/3 | **PASS** | `feature: pass §1.5/§3.2`。NEW-1（LoRA 位置）与 NEW-2 全修，位置与源码逐行一致 |
| 19_vllm_compilation_cudagraph_analysis | pass | pass | pass | layout | pass | 12/12 | **PASS** | `feature: pass §7.3`。F6、N2、NEW-3/4/5/7 全修；NEW-6 半修（P3） |

**新增 P0 = 0，新增 P1 = 0，新增 P2 = 4，新增 P3 = 5。**

---

## 1. Round-3 未决项逐条状态

### 1.1 协调者已复核的五条 P1：邻句连带核验

| # | 项 | 状态 | 邻句核验结论 |
|---|---|---|---|
| 1 | 02 §2.4 output-handler 父节点 | **FIXED** | 节点已移到 `AsyncLLM.add_request` 下、`InputProcessor.process_inputs` 之后、`_add_request` 之前。回源：`async_llm.py::add_request` 顺序为 `process_inputs(_async)` → `assign_request_id` → `_run_output_handler()`(:465) → `_add_request`(:474/:486)，**兄弟顺序完全一致**；注记「懒启动兜底：`__init__` 已在有事件循环时先启动过」与 `__init__`(:186-190) 的 `asyncio.get_running_loop()` + `_run_output_handler()` 一致。邻句 `_add_request` 三个子节点（`check_admission` → `OutputProcessor.add_request` → `AsyncMPClient.add_request_async`）与 `_add_request`(:492-512) 逐行一致，**未被这次移动破坏**；`process_inputs` 标注「已渲染输入路径」与 `if isinstance(prompt, dict) and "type" in prompt:` 分支注释 “Rendered EngineInput; no blocking preprocessing needed” 一致 |
| 2 | 02 §5.5 render 门控 | **FIXED** | 新写法四条断言全部回源成立：`routers.py:52` 门是 `"generate" in supported_tasks or "render" in supported_tasks`；`factories.py` 在 env/专用模式门后**无条件** `include_router(render_render)` 与 `include_router(derender_render)`，只有 `/inference/v1/generate` 被 `if "generate" in supported_tasks:` 包住；`app_state.py:123` 的 `if "generate" in supported_tasks:` 内调 `init_scale_out_state` → `init_render_state` 建 `serving_render`/`serving_derender` 再建 `ServingTokens`。邻句保留的三态门、`ValueError`、info 后 `return`、`force_no_detokenize`/`/abort_requests` 与 `factories.py:66-100` 一致，**未被结论改写带偏** |
| 3 | 02 §5.3 Rust 前端报错条件 | **FIXED** | `envs.py::_resolve_rust_cli_path` 逐字核对：`raw = os.environ.get(..., "auto")`；未启用时（且已设该变量）打 warning 并 `return None`；`raw.lower() in ("auto","1","true")` 才找 `<pkg_dir>/vllm-rs`，找不到 `raise FileNotFoundError`；否则 `return raw` 不校验。页面新增的 `VLLM_USE_RUST_BENCH` 也确在 `use_rust` 的 `or` 侧。**四句全对** |
| 4 | 07 §8.4 `delay_free_blocks \|= connector_delay_free_blocks` | **FIXED**（留 1 条 P2） | `scheduler.py:2538` 求或、:2521 `_connector_finished`、:2530 EC `\|=`、:2502-2511 `finish_requests` 算出后作为 `delay_free_blocks=` 入参传入——页面四点全对。P2 见 N4-4 |
| 5 | 12 §3.2 LoRA 位置 | **FIXED** | 行已移到 `model_state.preprocess_state` 之后、`[num_ubatches>1] UBatchRunner.prepare` 之前，行首标记改回 `+--`。回源 `gpu/model_runner.py:1640-1701`：非 dummy 分支内 `prepare_inputs`(1646) → `prepare_attn`(1650) → `preprocess_state`(1654) → `make_lora_inputs`(1662) → `_set_active_loras`(1668)，随后(1699+)才 ubatch/attn metadata 与四条执行分支。注记「唯一调用点」核实：`grep _set_active_loras` 该文件**仅 1668 一处** |

### 1.2 任务点名的 P2 逐条

| 项 | 状态 | 源码证据 |
|---|---|---|
| 06 §4.2 图/spec 队列项写成二元组 | **FIXED** | spec 与图均改为三元组 `(F1, S1, X1)` / `(F2, S2, X2)`。`core.py:706` `batch_queue.appendleft((future, scheduler_output, exec_future))`；类型标注 `deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]]`(:210-213)。exec future 用途也改对：图后新句「`future.result()` 返回 `None` 说明原来的 `execute_model()` 失败了，此时 Core 正是靠 `exec_model_fut.result()` 把原异常重抛出来」与 `core.py:723-730` 的注释 “None from sample_tokens() implies that the original execute_model() call failed - raise that exception.” 逐字对应；「pooling 或本轮没有模型执行时，入队的 future 本身就是 execute future」与 :684-687 一致 |
| 08 §5.3 `prefix_cache_retention_interval` 语义 | **FIXED** | 新写法「每隔多少 token 额外保留一个 SWA/Mamba checkpoint；`0` 只保留语义 checkpoint（最新重放边界、共享前缀交汇点）；正值才额外留周期性 checkpoint」与 `config/cache.py:159-165` docstring 逐句对应；「默认 0，来自已弃用 env」与 `_get_prefix_cache_retention_interval()`(:61-67) 的 `return 0 if env_value is None else int(env_value)` 一致；与 `prefix_match_unit` 的对开句（“not how often states are stored”）也回源成立 |
| 19 图 1 `num_tokens/num_reqs` 边 | **FIXED** | 边标签改为「下一步 prepare 复用同址 storage 的切片」。`input_batch.py:24-25` 的 `InputBuffers` 只有 `max_num_reqs/max_num_tokens` 容量常量与六个 tensor，无本步 `num_tokens/num_reqs`，新标签不再声称传递不存在的对象 |
| 19 `post_update` 是否另写 `output_bin_counts` | **FIXED** | 正文改为「落进 `RequestState` 的是…；`post_update` 的 kernel 另写一项**不属于** `RequestState` 的 `sampler.penalties_state.output_bin_counts`（仅末 PP rank 非 None）」。回源 `gpu/model_runner.py::postprocess_sampled`(1522-1547)：`if self.is_last_pp_rank: output_bin_counts = self.sampler.penalties_state.output_bin_counts else: None`，随后作为 `post_update` 第 4 个实参；`input_batch.py::post_update`(600-638) 把它与 `stride(0)` 一起传给 `_post_update_kernel`。**「仅末 PP rank 非 None」与「不属于 RequestState」两点都对** |
| 19 §9.2 ⑰ 键集合公式 | **FIXED** | 新公式「mixed 一套 `product(cudagraph_capture_sizes, lora_cases)`（PIECEWISE 再 relax `num_reqs`/`uniform`）＋ `decode_mode()==FULL and separate_routine()` 时另一套 decode 键，size 受 `decode_query_len <= x <= max_num_seqs*decode_query_len`，上限约 2×」与 `cudagraph_dispatcher.py::initialize_cudagraph_keys`(167-233) 逐条一致（含 `replace(batch_desc, num_reqs=None, uniform=False)` 只对 PIECEWISE、decode 过滤 `x <= max_num_tokens and x >= uniform_decode_query_len`） |
| 19 §6.5 MRV1 侧 resolve 的调用者 | **PARTIAL** | §6.5 已点名 `GPUModelRunner._check_and_update_cudagraph_mode`，链路核实成立：`initialize_kv_cache`(7491) → `initialize_attn_backend`(7509/def 7111) → `_check_and_update_cudagraph_mode`(7206/def 7252) → `resolve_cudagraph_mode_and_sizes`(7279)。**但 §12 阅读路线仍未登记该符号**（全页仅 §6.5 一处命中），且句内「由 `initialize_attn_backend` 经 `GPUModelRunner.initialize_kv_cache`」把链路写成由内向外，读者需要回读一次。见 N4-7（P3） |
| 11 §1.4 / 12 §1.5「未自定义 Scheduler」 | **FIXED（11 留 1 条 P2）** | 两页都改成「解析本身不读 `scheduler_cls`；解析为 True 之后未自定义 Scheduler 才用 `AsyncScheduler`」。回源 `config/vllm.py::__post_init__`(1270-1319)：`elif async_scheduling is None:` 分支只判 pooling / spec method 白名单 / `disable_padded_drafter_batch` / `executor_supports_async_sched` / `uses_rocm_deepep_ht_dbo`，**既不读 `scheduler_cls` 也不读 PP**；`config/scheduler.py::get_scheduler_cls`(201-222) 才因自定义类放弃 `AsyncScheduler`（且仅 `warning_once`）。11 新增的五项清单与源码五条分支**一一对应**。11 的残留缺陷见 N4-2 |
| 12 的 MRV1/MRV2 LoRA capture case 对照（19 F6） | **FIXED** | 19 §7.3 新增三行对照，逐条回源成立：无 LoRA 两代都 `[0]`（`cudagraph_dispatcher.py:118-120` / `gpu/lora_utils.py:30-31`）；开 `cudagraph_specialize_lora` 两代都是 `[0] + get_captured_lora_counts(...)`（:122-127 / :32-35，且 `lora/utils.py:63-65` 的 range 从 1 起，**永不返回 0**，故 MRV2 的 `if c > 0` 过滤不产生差异，页面「两代都是」成立）；关 specialize 时 MRV1 `[max_loras + 1]`（:129-130，**无 0 case**）而 MRV2 `[0, max_loras + 1]`（:36）。页面推论「MRV1 不为『本步无活跃 adapter』单独建图」正确 |
| 07 stale 排空数值重放是否补回 | **FIXED 且正确** | 见 §2 |

### 1.3 其余 round-3 项

| 项 | 状态 | 证据 |
|---|---|---|
| 02 P2-9 §2.2 启动树缺「无 KV → 关 chunked prefill」 | **FIXED** | 新节点插在 `SchedulerConfig.get_scheduler_cls` 与 `所选 Scheduler 类(...)` 之间，与 `core.py:148-156`（`get_scheduler_cls()` → `if len(kv_cache_groups)==0:` → warning + `enable_chunked_prefill = False` → `Scheduler(...)`）位置一致 |
| 02 P2-4 / 06 `model_executed` 语义 | **FIXED** | 02 §2.4 改为「pooling 或 `model_executed` 为假时…（该标志在 EC producer 引擎上恒假，见 06 §4.1）」；06 §4.1 新增整段。回源 `core.py:674-687`：`model_executed = False`，仅 `if self.is_ec_consumer:` 才改写为 `total_num_scheduled_tokens > 0`；`is_ec_consumer = ec_transfer_config is None or ec_transfer_config.is_ec_consumer`(:218-221)，而 `ec_transfer_config` 默认 `None`(`config/vllm.py:406`)，故「普通部署恒为 consumer」成立，纯 `ec_producer` 实例（`ec_transfer.py::is_encode_only`）上恒 False 也成立 |
| 02 P2-3 derender 路由集不全 | **FIXED** | 现写「derender 侧**只有两条**…没有 `/v1/messages/derender`」。回源 `scale_out/derender/api_router.py` 只有 `/v1/chat/completions/derender`(:31) 与 `/v1/completions/derender`(:81)；render 侧三条(:27/:53/:77) |
| 06 P2-6 §2 scoping 覆盖过宽 | **FIXED** | 改为「**这条 scoping 只约束 §2.2 那一格**…§2.1 与 §2.3 在两条分支下是同一套流程」。与 06 的实际小节结构（2.1 前端登记 / 2.2 同步 schedule-执行-归并 / 2.3 前端可见完成）一致 |
| 08 P2-8 Mamba 对齐归属互指 | **FIXED** | §5.3 改为「候选 checkpoint 是否有效由本页判定；**用它挑切分点并算 padding** 归 07 §5.4」。07 §5.4 标题「Mamba split 保证缓存的是哪个位置的状态」确实是该内容的 owner；08:614 的「07 负责挑选可执行的 checkpoint 对齐终点，11/12 负责实际状态导出，本页拥有容量/位置表/身份/归还」与新句同一口径，不再互推 |
| 19 NEW-7「全库只有三处构造 `CudagraphDispatcher`」 | **FIXED** | 改为「产品代码只有三处（`tests/` 下另有若干）」 |
| 19 N2 §7.5 Core 树末行 | **FIXED** | 树末改为两条并列分支：`[队列未满且仍有工作] 直接 return None，本次调用不消费任何结果` / `[否则消费最旧项] …`。与 `core.py:707-711` 的 `if len(batch_queue) < batch_queue_size and (model_executed or scheduler.has_requests()): return None, model_executed` 一致 |
| round-3 P2-5 owner 交接「裸页链接 + 手写节号」 | **PARTIAL** | 已改成锚点链接的：02 §2.4 → `06#6.2`、07 §2.1 → `06#4.1`、07 §8.5 → `06#6.1`、11 §1.4 → `06#4.1`/`06#4.2`、12 §1.5 → `06#4.1`、07 §8.7 新增 `23#5.3`。**仍是裸链接/裸节号的**：02:173「继续见 `[[06…\|Engine 运行循环]]`」、02:374 同款、08:690 新写的「归 07 §5.4」（连链接都没有）、11:76「`[[06…\|Engine 架构]]` 第 4 节」。见 N4-8（P3） |
| 06 round-2 P2-13（R 未走到第二个 token 与完成） | **NOT_FIXED** | §4.2 文本未动，仍只写「第二个输出要等 S2 自己的 future 被消费才归并」。round-3 即记为 PARTIAL，本轮无变化；非阻塞项 |

**统计（我负责的 7 页）：FIXED 20 ｜ PARTIAL 2（P2-5 锚点化、19 NEW-6）｜ NOT_FIXED 1（06 P2-13，round-2 起的非阻塞遗留）。**

---

## 2. 07 §8.3 新增数值重放：逐格源码重放

新表主张与 `vllm/v1/core/sched/scheduler.py` 的逐条对应（全部成立）：

| 页面主张 | 源码 |
|---|---|
| 抢占时 stale **赋值**为当前 in-flight（2），不累加 | `_preempt_request`:1473 `request.num_stale_output_tokens = request.num_in_flight_tokens`，注释明写 “num_in_flight_tokens already includes any undrained stale share, so assign rather than accumulate” |
| 抢占时 computed 与 output placeholders 归零、回 waiting | :1460 `num_computed_tokens = 0`；:1474 `num_output_placeholders = 0`；:1480 `self.waiting.prepend_request(request)` |
| 每份返回按**该计划**的 scheduled token 数扣减，2→1→0 | `update_from_output`:1906-1911 `request.num_in_flight_tokens -= num_tokens_scheduled`；`if num_stale_output_tokens > 0: output_is_stale = True; -= num_tokens_scheduled; assert >= 0` |
| 仍追加有效输出 token t1/t2 | :1991-1994 `_update_request_with_output(request, new_token_ids, is_stale=output_is_stale)` → 基类 :2314 `append_output_token_ids`（`is_stale` 只被 AsyncScheduler 覆写使用） |
| 不再扣已归零的 computed/placeholders，旧 spec rejection 不二次回滚 | :2029-2034 `if not output_is_stale:` 才 `num_computed_tokens -= num_rejected` / `num_output_placeholders -= num_rejected`；`async_scheduler.py:61-63` `if not is_stale: num_output_placeholders -= len(new_token_ids)`，注释 “Placeholders were zeroed at preemption; a stale delivery must not decrement them (it would underflow)” |
| 排空前 waiting 跳过 R，排空后才可重新调度 | :840-849 `if request.num_stale_output_tokens > 0 and not request.drop_stale_output:` → `pop_request()` + `step_skipped_waiting.prepend_request()` |
| drop 路径走同样 2→1→0 排空、只是不追加 | 扣减(:1907-1911) 发生在 `if output_is_stale and request.drop_stale_output: continue`(:1924-1926) **之前**，故排空相同、不追加 token |

**结论：算例可重放，`algorithm-replay` pass。** 唯一缺口见 N4-3。

---

## 3. 本轮新发现（无 P0/P1）

### N4-1（**P2**｜06 §4.1）新插入的 `model_executed` 段落吞掉了本节结论句

- 现文：`…所以这个名字读作"本轮是否执行了模型"会在 EPD 分离部署下读错。因此**普通生成、兼容配置、MRV2、PP=1 的缺省是 AsyncScheduler + 容量 2 的 batch queue**，pooling 或自动回落到 False 也只有在 PP=1 时才回到同步 step。`
- 问题：`因此` 原本承接的是上一段的容量表与 `batch_queue is None → step` 规则；新段落插在中间后，`因此` 在字面上从「EC producer 上 `model_executed` 恒 False」推出「缺省是 AsyncScheduler + 容量 2」，两者没有因果关系。本节是 06 的 feature 单元（选择轴）的结论句，被降级成一个例外段的尾巴。
- 事实本身无误（`model_executed` 段与容量结论各自都对，已回源）。
- 修法：把 `因此**普通生成…** … **batch queue 和 async scheduling 不是同义词**：…` 切回 `model_executed` 段之前，让新段落独立成段落尾。

### N4-2（**P2**｜11 §1.4）改写留下未删干净的旧句，并因此丢掉 PP=1 前提

- 现文（同一句内）：`…解析为 True 之后，未自定义 Scheduler 才用 AsyncScheduler，PP 则另外影响容量。``async_scheduling=None` 仍解析成 True，因此使用 `AsyncScheduler`、容量 2 的 `EngineCore.step_with_batch_queue()`。`
- 问题有两层：(1) 后一句是重写前旧句的残片，与前半句重复；(2) 旧句原带 `PP=1` 前提（round-3 原文是「普通生成、兼容配置、**PP=1** 且未自定义 Scheduler 时」），新写法把 PP 移出条件后，这句残片就变成**无条件断言「容量 2」**。源码 `config/vllm.py::max_concurrent_batches`(567-577)：async + MRV1 只有 `pp_size <= 1` 才返回 2，`p>1` 返回 `p`。也就是说残片对 MRV1 + async + PP>1 是错的，而本页正是 MRV1 页。
- 修法：删掉该残片整句即可（前半句已完整表达，容量规则已交给 06 §4.1）。

### N4-3（**P2**｜07 §8.3）drop 模式与 waiting 跳过的对照被合并成一句，易读反

- 现文：`排空前 waiting 会跳过 R，避免新执行重采同一位置、旧输出随后又交付一次；``drop_stale_output` 路径走同样的 2→1→0 排空，只是不追加 t1/t2。`
- 源码：waiting 的跳过条件是 `num_stale_output_tokens > 0 **and not drop_stale_output**`（:841-843）。drop 模式**不被跳过**——这正是 drop 存在的理由（`reset_prefix_cache` 同步恢复、connector hand-off 需要同一步就恢复）。分号连接「会跳过」与「drop 走同样的排空」，读者容易把跳过也带到 drop 分支上。
- 邻句「waiting 看见**可交付** stale 尚未排空就跳过该请求」写法正确，两句相邻反而更易混。
- 修法：改成「排空前 waiting 只跳过**可交付** stale 的 R；`drop_stale_output` 份额走同样的 2→1→0 扣减，但既不追加 t1/t2，也不阻塞重新调度」。

### N4-4（**P2**｜07 §8.4）「delay 有两个独立来源」漏了 partial-tail 这一路

- 现文：`**delay 有两个独立来源，`_free_request` 把它们求或**…其一是 connector 自己要求的——`_connector_finished(request)` 转发 KV connector 的 `request_finished()` 返回值…`
- 源码 `_connector_finished`(:2793-2847) 的返回值是 `return delay_free or partial_tail_delay, kv_xfer_params`，其中 `partial_tail_delay` 来自 `self.connector.register_finished_partial_tail(request, block_ids, finished_partial_tails)`（producer 侧 `finalize_partial_tail_offloads` 非空时）。所以「其一」实际是两路的或，全链共三个来源。
- 页面末句确实提到「producer 的 partial Mamba tail 还可能在 finalize/store 完成前继续保留」，但没有把它挂到 `delay_free_blocks` 上，读者无法把两处连起来。
- 修法：在「其一」里补半句「（同一处还或进 `register_finished_partial_tail()` 的 partial-tail delay）」。

### N4-5（P3｜08 §5.3）同一节里「默认 0」与「默认 `None`」并存

- 本轮新写的第一段：`prefix_cache_retention_interval`（**默认 0**，从已弃用 env 读取）；同节第四段仍写「缓存保留策略…**默认 `None`** 保留密集可达 checkpoint」。
- 两句各有所本：配置键默认确为 0（`cache.py:157-158` 的 `default_factory`），而 `None` 是 `single_type_kv_cache_manager.py::cache_blocks` 形参 `retention_interval: int | None = None` 的默认（docstring “``None`` keeps dense checkpointing”）。但同节相隔 27 行出现两个不同的「默认」，读者会当成矛盾。
- 修法：第四段改为「取 `None` 时保留密集可达 checkpoint（这是 manager 形参默认，不是配置键默认）」。

### N4-6（P3｜06 §4.2 图 2）`X2` 出现在队列元组里，但图中从未被返回

- 图内有 `C->>Q: 入队 (F2, S2, X2)`，而 S2 那一步只有 `E-->>C: F2 sampling future` 一条返回消息，X2 没有对应的 `E-->>C`。S1 那一步则完整画了 X1。
- 图前 spec 写「每步Core先非阻塞发execute再非阻塞发sample」，加上 `C->>E: S2：…同样两次非阻塞提交` 可以推出 X2，但这违反了本页图的自述规则（跨边界对象都实名画出）。
- 修法：给 S2 补一条 `E-->>C: X2 execute future（不取值）`，或把 S2 的入队标签写成 `(F2, S2, X2 同上)`。

### N4-7（P3｜19 §6.5、§12）符号已点名，但链路写成由内向外，且阅读路线未登记

- 现文：`MRV1 的直接调用者是 …::GPUModelRunner._check_and_update_cudagraph_mode，由 `initialize_attn_backend` 经 `GPUModelRunner.initialize_kv_cache``——实际链路是 `initialize_kv_cache` → `initialize_attn_backend` → `_check_and_update_cudagraph_mode`，页面按内→外排列，「由 A 经 B」易被读成 A 在 B 之外。
- `_check_and_update_cudagraph_mode` 全页只在 §6.5 出现一次，**§12 源码阅读路线仍未登记它**（round-3 NEW-6 的后半条）。
- 修法：改成「经 `initialize_kv_cache` → `initialize_attn_backend` → `_check_and_update_cudagraph_mode`」，并在 §12 补一行。

### N4-8（P3｜02 §2.3.3/§3.x、08 §5.3、11 §1.4）仍有四处裸页链接 + 手写节号

- 02:173「完整选择轴与兼容条件继续见 `[[06…|Engine 运行循环]]`」（无 `#4.1`）、02:374 同款、08:690 新写的「归 07 §5.4」（**纯文本，无链接**）、11:76「`[[06…|Engine 架构]]` 第 4 节」。
- `check_links` 的 `stale_section` 不覆盖这种写法（本轮实跑 0 命中），正是 round-3 B1 那类漂移的温床；且 08 新加的那处是本轮引入的。
- 修法：统一改成锚点链接（`06#4.1`、`07#5.4 Mamba split 保证缓存的是哪个位置的状态` 两个标题都存在）。

### N4-9（P3｜11 §1.4）`第 4 节 解释` 之间多一个空格

- 现文：`…由 [[06…|Engine 架构]] 第 4 节 解释。` 排版噪点，随 N4-2 一并改即可。

---

## 4. 02 的架构八项复判

| item | round-3 | 本轮 | 依据 |
|---|---|---|---|
| 1 目标与约束 | pass | pass | 未改动 |
| 2 模块与边界 | pass | pass | 未改动 |
| 3 数据/控制流 | pass | pass | 未改动 |
| 4 关键机制 | pass | pass | §2.4 例外段重写后与 `core.py:674-730` 一致 |
| **5 场景清单与真实入口对账** | **FAIL** | **PASS** | 两条 FAIL 原因（§5.5 路由门控写反、§5.3 Rust 报错条件写反）均已按源码改正且新句无误；§5.5 新增的 derender 路由集、`init_scale_out_state` 归属、`/inference/v1/generate` 的额外条件均回源成立；§5.3 Rust 格四句全对。round-3 已 pass 的 §5.1/§5.6/非 OpenAI 前端未被本轮改动触及 |
| 6 失败与降级 | pass | pass | 未改动 |
| **7 调用树父子/兄弟顺序** | **FAIL** | **PASS** | `_run_output_handler` 已移到正确父节点与正确兄弟位；`_add_request` 三子节点顺序与 `async_llm.py:492-512` 一致；§2.2 启动树新增的 chunked-prefill 节点位置与 `core.py:148-156` 一致；EngineCore 树十个节点顺序与 `core.py:675-737` 逐行一致 |
| 8 场景完成条件与读者交接 | pass | pass | §6 新增「插件扩展」行指向 24，页面存在、链接解析 |

---

## 5. 机械门禁（本机 `.venv` py3.13，本轮实跑）

| 检查 | 结果 |
|---|---|
| `check_links.py --strict` | pages=453，broken / ambiguous / bare_index / stale_section / orphans 全 **0** |
| `check_math.py --changed --strict` | 64 文件，**0 错 0 警** |
| `check_markdown.py --changed --strict` | 64 文件，**0 错 0 警** |
| `check_assets.py --changed --strict` | 64 文件，**0 错 0 警** |

未跑 `check_locators`（七页本轮新增文字不含 `path:line` 引用）；未重跑 mkdocs scoped build（本轮未新增/改名任何标题，仅 02→24 一条新链接与 4 条新锚点链接，`check_links` 已全部解析）。

---

## 6. 未复核范围

- 未运行 vLLM、GPU、多进程服务、EC/PD 或 EPD 分离部署、故障注入；EC producer 上 `model_executed` 恒 False、单进程同时暴露 render/derender/generate 三段均为读码推断（含 `routers.py`/`app_state.py`/`factories.py` 三处门）。
- 只按需打开了 03/13/14/15/16/22/23/24 的相关小节做跨页对账，未复审这些页面自身的 finding（属其他组）。
- round-2 起的非阻塞遗留（06 P2-13）只确认「未变化」，未重新论证其严重度。
