# 独立评审报告 Group B：02 软件架构 / 06 Engine 架构

- 评审者：独立 reviewer（非作者），只报告问题，未修改任何 wiki/skills/docs/源码文件。
- 冻结基线：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`。已核对 checkout `HEAD` 就是这个提交，commit time `2026-09-06T17:54:32-07:00`（即 2026-09-07 00:54 UTC）。工作区只有未跟踪的 `artifacts/`，没有移动过 HEAD。
- 依据：
  - `skills/source-faithful-analysis/references/` 下的 source-fidelity、codebase、analysis-focus、page-review-rubric；
  - document-types 与 reviews 下的 software-architecture 和 feature-analysis 两个 profile；
  - CLAUDE.md；
  - 协调者追加的 E2E / 连贯性要求。
- 验证方式：只读源码和测试，没有运行 GPU、多进程或服务。下文结论都已在 checkout 或其他 wiki 页面中核实；未能核实的会标 “unverified suspicion”。

---

## 0. 结论速览（verdict rows）

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 02_vllm_architecture_overview_analysis | pass | **FAIL §2.3.3/§2.4** | pass | timing, layout | pass | 3/3 取样；共 21 处，2 处有误 | **REJECT** | 代表路径把 `EngineCore.step` 当“普通/基本”分支。基线默认的生成请求实际走 AsyncScheduler + `step_with_batch_queue` |
| 06_vllm_engine_architecture_analysis | pass | pass | pass | timing | pass | 3/3 取样；共 18 处，未发现锚点错误 | **REJECT** | 缺默认路径与 Scheduler/AsyncScheduler 这条同级选择轴（feature 检查），修复量约一段话 |

- **02 architecture: FAIL**
  - item 5（§5 场景清单）：缺 scale-out 场景，即 render → token-in 往返与 `VLLM_ENABLE_SCALE_OUT_ENDPOINTS` 门控；§5.3 的 gRPC/Rust/Omni 交给 13，但 13 实际不拥有这部分。
  - item 8（读者交接）：采样/结构化输出（14）、投机（16）、多模态（15）在架构中没有落点或链接。
  - item 1/2/3/4/6 pass。
  - item 7 父子关系基本 pass，只有 P2 小问题；代表路径选错已记为 hop-walk FAIL。
- **06 feature: FAIL §2.2 / §4.1**
  - 页面把 §2 称作“普通 core step”，但基线默认配置下 R 不走 `EngineCore.step`。
  - `SchedulerConfig.get_scheduler_cls`（Scheduler ↔ AsyncScheduler）这一同级选择轴没有在页内说明。

---

## 1. 02 软件架构页

### 1.1 八项架构检查

| # | 检查 | 结果 | 说明 |
|---|---|---|---|
| 1 | 背景、目标、能力边界，单一分类轴 | pass | §1.1–1.3 完整；六模块按“谁拥有请求语义/生命周期/资源政策/设备协作/执行状态/模型合同”划分，轴一致 |
| 2 | 静态 / 动态 / 代码图同名 | pass | SVG 标签、§2.1 表、§3 小节、§4 表都用同一组六个模块名 |
| 3 | 每模块有职责、理由及被否方案、实现、限制 | pass | §3.1–3.6 都有“直观方案→问题→当前设计”，并标注为分析推断 |
| 4 | 多对多代码映射 | pass | §4 表加目录树，双向说明 |
| 5 | 场景清单与真实入口对账 | **FAIL** | 见 F-6、F-4 和 P2-8 |
| 6 | 命令与 flag 对 parser | pass | 以下都已对到 parser：`serve --host/--port/--grpc/--tensor-parallel-size/--data-parallel-size`、`run-batch -i -o --model`、`launch render <model> --host --port`、`chat/complete --url --model-name --quick`、`bench throughput --dataset-name random --num-prompts`、`bench serve --model`、`collect-env`，以及 examples 的 `--model/--max-tokens` |
| 7 | ASCII 调用树父子关系、顺序兄弟、返回到调用者 | pass（有 P2） | 小问题见 P2-4、P2-7；代表路径选错记在 hop-walk（F-1） |
| 8 | 场景完成条件与读者交接 | **FAIL** | §5.3 交接对象错误（F-4）；14/15/16 没有架构落点（F-5） |

### 1.2 锚点抽查（02）

| # | 锚点 | 页面主张 | 源码实际 | 结果 |
|---|---|---|---|---|
| 1 | `vllm/v1/engine/core.py::EngineCore.__init__` / `_initialize_kv_caches` | 启动顺序：executor → KV specs/layout → memory → configs → initialize_from_config → warmup → StructuredOutputManager → Scheduler → step_fn；遇 non-causal 关闭 chunked prefill 和 prefix caching；无 KV 模型关闭 chunked prefill | 顺序一致；non-causal 分支在 `_initialize_kv_caches` 内。“无 KV 关闭 chunked prefill”写在 `EngineCore.__init__` 里，位于 `_initialize_kv_caches` 返回之后（`len(kv_cache_groups)==0`） | ✓（归属有误，P2-1） |
| 2 | `EngineCore.step` | 顺序：schedule → execute_model → get_grammar_bitmask → future.result → 结果为 None 时 sample_tokens → `_process_aborts_queue` → update_from_output | `step` 本身完全一致；但它不是默认 step_fn | ✓ / 路径选择 ✗（F-1） |
| 3 | `vllm/v1/engine/async_llm.py::AsyncLLM.check_admission` | 检查未完成请求数，以及仍在 prefill 的 prompt token 总量 | `max_num_queued_reqs` 与 `max_num_queued_tokens`，一致 | ✓ |
| 4 | `AsyncLLM._add_request` | check_admission → `OutputProcessor.add_request` → `add_request_async` | 一致（n>1 时 admission 在 `add_request` 中做） | ✓ |
| 5 | `vllm/v1/engine/core_client.py::EngineCoreClient.make_client` | asyncio 且非 multiprocess 时抛 `NotImplementedError` | 一致 | ✓ |
| 6 | `vllm/v1/executor/abstract.py::Executor.get_class` | 可选 uni/mp/Ray/external launcher/自定义；自定义类做类型检查 | 一致，并含 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND` 分支 | ✓ |
| 7 | `vllm/config/vllm.py::VllmConfig.use_v2_model_runner` | 显式环境变量优先；ROCm 特定架构、无 Triton、有不支持特性时回退 MRV1 | 一致 | ✓ |
| 8 | `vllm/v1/core/kv_cache_manager.py::KVCacheManager.allocate_slots` | `full_sequence_must_fit` 先做准入预检并可能返回 None；之后 `remove_skipped_blocks`，再计算需求 | 一致 | ✓ |
| 9 | `vllm/v1/executor/multiproc_executor.py::MultiprocExecutor.start_worker_monitor`；`vllm/v1/worker/gpu_worker.py::Worker.execute_model` | monitor 发现 worker 意外退出后置 failed、shutdown、回调通知；Worker 在复用 buffer 前等待上一轮 PP send | 一致（`_pp_send_work` handle.wait） | ✓ |
| 10 | `vllm/entrypoints/cli/serve.py::ServeSubcommand.cmd` / `run_headless` / `run_multi_api_server` | 分支 grpc / headless / multi-port / multi-API 或 Rust / single；LB 模式互斥；headless 拒绝正的 API 数 | 一致 | ✓ |
| 11 | `vllm/entrypoints/launchers/api_server/entry.py::run_server/run_server_worker/build_async_engine_client_from_engine_args` | setup_server → context 内建 AsyncLLM → reset_mm_cache → build_and_serve → 退出 context 后 shutdown → 等 shutdown_task → sock.close | 一致（`serve_http` 会 await server_task） | ✓ |
| 12 | `vllm/entrypoints/launchers/run_batch.py::BatchRequestInput`、`build_endpoint_registry`、`run_batch`、`main` | 类注释称只支持 chat，registry 实际注册 6 类；JSON 坏行在 gather 之前抛出 | 一致 | ✓ |
| 13 | `vllm/entrypoints/launchers/render/entry.py::run_launch_fastapi` | 清除 quantization，抑制 CPU KV 警告，不建 Scheduler | `model_config.quantization=None`、`VLLM_CPU_KVCACHE_SPACE=0`，一致 | ✓ |
| 14 | `vllm/entrypoints/cli/openai.py::ChatCommand/CompleteCommand` | `--url --model-name --quick`；交互模式读到 EOF 结束 | 一致（`-q/--quick`，默认 url `http://localhost:8000/v1`） | ✓ |
| 15 | `vllm/entrypoints/cli/benchmark/main.py::maybe_exec_rust_bench`、`vllm/benchmarks/throughput.py::main`、`vllm/benchmarks/serve.py` parser | Rust bench 委托；缺路径报错；throughput 走 validate_args/get_requests/run_vllm(_async)；bench serve 有 `--model` | 一致 | ✓ |
| 16 | `examples/basic/offline_inference/generate.py`、`embed.py`、`examples/deployment/async_llm_streaming.py` | 4 条提示词、`--max-tokens`；embed 用 pooling runner；streaming 示例写死模型并依次处理 3 条 | 一致 | ✓ |
| 17 | `vllm/entrypoints/openai/api_server.py` | 带弃用警告的转发模块，警告里写的是 `vllm server` | 一致 | ✓ |
| 18 | `vllm/entrypoints/llm.py::LLM.__init__/generate/enqueue/wait_for_completion`、`vllm/entrypoints/offline_utils.py::_run_engine` | DP>1 guard、renderer_num_workers 警告、非 generate runner 抛 ValueError、按 request id 排序 | 一致 | ✓ |
| 19 | `vllm/model_executor/model_loader/utils.py::initialize_model`、`vllm/v1/attention/selector.py::get_attn_backend` | 旧签名先警告再猜参数；按 KV 类型的覆盖优先 | 一致 | ✓ |
| 20 | `tests/v1/core/test_deferred_block_free.py::test_abort_defers_free` | “取消后在途步骤未收齐时 block 不立即释放” | 测试使用专门打开 gate 的 `_create_deferring_scheduler`；生产 gate 要求 KV consumer connector | ✗（泛化过度，F-2） |
| 21 | `vllm/v1/engine/async_llm.py::AsyncLLM.finish_weight_update`；`vllm/plugins/__init__.py::load_general_plugins` | 先等 worker 再写版本；进程内只加载一次 | 一致 | ✓ |

另核实：02 §2.4 列出的 3 个测试和 06 §8 列出的 12 个测试，函数名在基线上都存在。

---

## 2. 06 Engine 架构页（feature analysis）

### 2.1 feature 检查

- 引言（问题 → 解法形状 → 代价）：pass（§1）。
- 最小例子可重放：pass。§4.2 表格与 `test_engine_core_concurrent_batches` 逐项一致，而且页面对第 3 次调用“等 S2”的解释比测试自带注释更准确（测试注释差了一拍）。
- 变体集合与枚举依据：**FAIL**
  - Client 变体取自 `make_client`，Executor 变体取自 `get_class`，stale 模式取自 `drop_stale_output`，这三组都有来源；
  - 但 step_fn 默认值与 Scheduler 类选择这一同级轴没有建立，见 F-3。
- 第三方依赖边界：pass。§6.3 明确说没有读取 pyzmq 内部实现。
- 失败边界、成本：pass（§7、§3.2 收益与代价）。
- 页面替换与保存：§4.3 说明了对旧 02 async 主题的接续，没有发现丢失内容。

### 2.2 锚点抽查（06）

| # | 锚点 | 页面主张 | 源码实际 | 结果 |
|---|---|---|---|---|
| 1 | `vllm/v1/engine/llm_engine.py::LLMEngine.add_request` | InputProcessor → assign id → `OutputProcessor.add_request` → `engine_core.add_request` | 一致 | ✓ |
| 2 | `core_client.py::InprocClient.get_output` | 直接调 `step_fn` 与 `post_step` | 一致 | ✓ |
| 3 | `EngineCoreClient.make_async_mp_client` | external LB 用 `DPAsyncMPClient`，internal LB 用 `DPLBAsyncMPClient` | 一致 | ✓ |
| 4 | `core.py::EngineCoreProc.process_input_sockets` | ABORT 同时进入 input queue 和 abort queue，靠幂等保证安全 | 一致（源码注释原意） | ✓ |
| 5 | `scheduler.py::Scheduler._update_after_schedule` | computed 与 in-flight 同时增加；finished/preempted 集合换成新 set | 一致 | ✓ |
| 6 | `VllmConfig.max_concurrent_batches` | PP=pp；async 时 V2 为 pp+1，V1 在 pp≤1 时为 2 | 一致 | ✓ |
| 7 | `EngineCore.step_with_batch_queue` | appendleft/pop FIFO；入口 assert 队列未满；有空位返回 None；第三元素 exec_future 用于重抛异常 | 一致 | ✓ |
| 8 | `tests/v1/engine/test_engine_core.py::test_engine_core_concurrent_batches` | 10 token 预算、12 token prompt、容量 2、async 关闭；三次调用的表格 | 一致 | ✓ |
| 9 | `VllmConfig.__post_init__` async 兼容性 | 显式开启遇不兼容时报错；自动模式关闭；pooling 默认关；spec 允许 EAGLE/MTP/DraftModel/NGram GPU/DSpark | 一致 | ✓ |
| 10 | `Scheduler._preempt_request` / `update_from_output`；`async_scheduler.py::AsyncScheduler._update_request_with_output` | stale 取当前 in-flight（赋值不累加）；drop 模式粘滞；stale 结果不扣 placeholders；只有 RUNNING 请求 cache blocks；可交付 stale 未排空前 waiting 跳过 | 一致 | ✓ |
| 11 | `Scheduler.__init__` 的 defer gate、`_free_request_blocks`、`_drain_deferred_frees` | gate 为 KV consumer 且 `max_concurrent_batches>1`；deferred list 只排空头部 | 一致（fence 取值见 P2-11） | ✓ |
| 12 | `EngineCoreProc._send_msg_tracking_payload` / `process_output_sockets` | 第一帧 track=True；tracker done 之前留在 pending | 一致 | ✓ |
| 13 | `vllm/v1/engine/tensor_ipc.py::TensorIpcSender.set_target_engine` | 只支持 engine 0 | 一致 | ✓ |
| 14 | `DPEngineCoreProc._has_global_unfinished_reqs`；`vllm/config/parallel.py::ParallelConfig.dp_sync_interval` | 在 step 1 和 interval 倍数同步，默认 16 | 一致 | ✓ |
| 15 | `uniproc_executor.py::AsyncOutputFuture.result`；`multiproc_executor.py::FutureWrapper.result` | 延迟 `get_output`；按顺序 drain 前面的 future | 一致 | ✓ |
| 16 | `MultiprocExecutor.collective_rpc` | permanent failed 时拒绝新 RPC | `if self.is_failed: raise RuntimeError` | ✓ |
| 17 | `Scheduler.has_requests`；`EngineCoreProc._process_engine_step` | 有 connector pending push 时仍返回 true；未执行模型但仍有工作时 sleep 0.001 | 一致 | ✓ |
| 18 | `docs/design/arch_overview.md` “AsyncLLMEngine” 一节 | 仍写 async 类是同步类的 wrapper | 一致 | ✓ |

---

## 3. 编号问题清单

### P0（与源码不符且会误导）

**F-1 [P0] 02 §2.3.3 / §2.3.4 末段 / §2.4 调用树前言及树中 EngineCore 部分：代表性“普通生成请求”的默认执行路径写错**

- 页面原文：
  - “普通生成路径中，EngineCore 先取得模型执行结果；若执行接口返回 `None`，再调用采样接口”
  - “启用异步调度或流水线并行时，多步可以处于在途状态”
  - “EngineCore 树展示基本 `step` 分支，异步队列分支另读 `step_with_batch_queue`”
- 源码证据：
  - `vllm/config/scheduler.py::SchedulerConfig.async_scheduling` 默认 `None`，`vllm/engine/arg_utils.py::EngineArgs.async_scheduling` 同样默认 None。
  - `vllm/config/vllm.py::VllmConfig.__post_init__` 在 `async_scheduling is None` 时，除非遇到 pooling、不兼容的 spec 方法、`disable_padded_drafter_batch`、executor 不支持或 ROCm DeepEP HT DBO，否则设为 **True**。
  - `UniProcExecutor.supports_async_scheduling` 和 `MultiprocExecutor.supports_async_scheduling` 都返回 True。
  - `VllmConfig.max_concurrent_batches` 在 async 下为 2（V1、PP≤1）或 pp+1（V2），因此 `EngineCore.__init__` 会建 batch_queue，`self.step_fn = self.step_with_batch_queue`。
  - `SchedulerConfig.get_scheduler_cls` 在 async 时返回 `AsyncScheduler`。
  - `step_with_batch_queue` 在 `execute_model(non_block=True)` 之后立即 `sample_tokens(grammar_output, non_block=True)`，不等 execute 结果，也不以 None 作为条件（`pending_structured_output_tokens` 除外）；第一次调用不等待就返回 `None`。
- 结论：用默认参数执行 `vllm serve <生成模型>` 时，本节选的在线聊天请求走的是 AsyncScheduler 加 batch queue。页面标为“普通/基本”的 `step` 分支只在 `--no-async-scheduling`、pooling 模型或不兼容组合下生效。
- 跨页矛盾：12 §5.3 写明“异步调度默认开启（pooling 除外）”；07 §3 也描述了 None 的解析过程。
- 建议修复：
  1. 在 2.3.3 或 2.4 前言明确写出默认解析结果；
  2. 代表树改用 `EngineCore.step_with_batch_queue` 分支，或保留 `step` 但标为“同步调度 / pooling / 显式关闭 async 时”，并补一棵默认分支的简树（schedule → execute_model(non_block) → [无 pending grammar] sample_tokens(non_block) → 入队 → [队列满] pop 最旧 → result → `_process_aborts_queue` → update_from_output）；
  3. 同步修正 §2.3 第二张时序图中 “opt 需要独立采样” 的条件，以及 §2.4 表中 `ModelRunnerOutput / Future` 一行“普通路径可能先返回 None，再独立采样”的说法；
  4. 2.3.4 末段改为“默认即有两批在途，关闭 async 时才退化为单批 step”。

### P1（实质缺口或跨页矛盾）

**F-2 [P1] 02 §2.4 末尾测试条目：把延迟释放写成普遍行为**

- 页面原文：“`test_abort_defers_free`：取消后，在途步骤尚未收齐时，block 不立即释放。”
- 源码证据：
  - `vllm/v1/core/sched/scheduler.py::Scheduler.__init__` 只在 `kv_transfer_config.is_kv_consumer` 且 `max_concurrent_batches > 1` 时设置 `defer_block_free=True`。
  - `tests/v1/core/test_deferred_block_free.py::_create_deferring_scheduler` 的 docstring 说明生产 gate 还需要 PD KV-consumer connector。
  - `test_gate_disabled_without_connector` 验证：只有 async、没有 connector 时，释放是立即的。
- 与 06 §6.2 的明文冲突：“不能把‘凡是async都延迟回收’写成通用规则”。
- 建议：改为“在启用 KV consumer connector 且多批在途（`defer_block_free` gate 打开）时，取消后……”，并补上 `test_gate_disabled_without_connector` 作为反例。

**F-3 [P1] 06 §2（标题“先跟随 R 走完一次普通 core step”）、§2.2、§4.1：没有建立默认路径和同级选择轴（feature 检查失败）**

- 页面原文：§2.2 “`EngineCore.step` 先检查…”；§4.1 “EngineCore 在 `max_concurrent_batches > 1` 时创建 batch queue”。
- 源码证据：与 F-1 相同（`VllmConfig.__post_init__`、`SchedulerConfig.get_scheduler_cls`、`VllmConfig.max_concurrent_batches`、`EngineCore.__init__` 中的 `step_fn` 选择）。
- 06 在 §4.3 提到“自动配置会对不兼容组合关闭 async”，§5.1 也出现了 AsyncScheduler，但始终没有明说默认生成请求走 batch queue 和 AsyncScheduler。读者会把 §2 当作 R 的默认生命周期。
- 建议：
  1. 在 §2.2 开头或 §4.1 加一段选择表：step_fn 由 `max_concurrent_batches` 决定，Scheduler 类由 `get_scheduler_cls` 决定，`async_scheduling=None` 按上述条件解析，默认生成模型的结果是 AsyncScheduler + 容量 2 的 batch queue；
  2. 注明 §2.2 描述的是同步（async 关闭或 pooling）路径，是理解 §4 的前置；
  3. 用一句话说明 R 在默认配置下对应 §4 表格中的第几次调用。

**F-4 [P1] 02 §5.3 “gRPC、Rust 与 Omni 的调用边界”一段：交接给了不拥有该内容的页面**

- 页面原文：“客户端操作与完整部署参数由 [[13 Serving 控制面]] 和对应外部项目负责”。
- 证据：
  - 13 全文没有 gRPC / Omni 内容，Rust 只在 §4.1 “API count 改为一” 出现一次；
  - 本域 26 页里只有 02、05、23 提到 gRPC；
  - `vllm/entrypoints/grpc_server.py::serve_grpc` 在域内没有 owner 页。
- 建议：改为“本域暂无 owner 页（覆盖缺口）”，或交给 planning-codebase-analysis 决定 owner。不要把读者引向 13。

**F-5 [P1] 02 E2E 交接：采样/结构化输出、投机解码、多模态在架构中没有落点**

- 页面原文：2.3.3 “模型前向计算与 token 选择在接口上可以分开…”，调用树中有 `Scheduler.get_grammar_bitmask`、`GPUModelRunner.sample_tokens`，但全页链接 14 的次数为 **0**。
- 证据：
  - 用 grep 统计 02 指向各页的 wikilink，14=0；16 只出现在 §6 表；15 只在 §5.3 作为服务变体出现。
  - 源码中这些能力横跨多个模块：`Scheduler.get_grammar_bitmask` / `StructuredOutputManager`（资源调度）、`GPUModelRunner.sample_tokens`（设备运行）、`AsyncScheduler._update_after_schedule` 的 spec placeholders、`Scheduler.update_from_output` 的 spec rejection。
- 建议：
  1. 2.3.3 在“token 选择”处链接 14；
  2. 在 §3.3 / §3.5 的“源码与边界”或 §3.7 表中各加一行，说明结构化输出、投机、多模态分别横跨哪些模块，并链接 14 / 16 / 15；
  3. §6 表补“采样与约束怎样落到 token？”→ 14。

**F-6 [P1] 02 §5.5 “独立渲染”与场景清单：缺 render → token-in 往返的执行面**

- 页面原文：“render 成功证明的是预处理/后处理完成，下一步模型计算仍需独立执行服务”。页面没有说明是哪个服务、如何开启。
- 源码证据：
  - `vllm/entrypoints/launchers/api_server/routers.py` 在 supported tasks 含 generate 或 render 时调用 `vllm/entrypoints/scale_out/factories.py::register_scale_out_api_routers`；
  - 非 render、非 `--tokens-only` 模式要求 `VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1` 才注册；
  - `examples/scale_out/example_mm_serve.py` 演示 `VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1 vllm serve …` 下 `/v1/chat/completions/render` → `/inference/v1/generate` 的往返；
  - 03 §源码路线已有 `ServingRender` / `ServingTokens` / `ServingDerender` 锚点。
- 建议：在 §5.5 完成与限制，或 §5.3 变体表加一行 “scale-out token-in 生成服务”，写出 env / `--tokens-only` 门控，链接 03。

**F-7 [P1] 02 §5.6 多副本场景：没有链接 DP 路由 owner，DP=2 模板的实际入口也没说**

- 页面原文：`vllm serve "<兼容模型>" --data-parallel-size 2`；表格写 “serve 分支、Executor、DP coordinator”；Mermaid 画单个“API 前端”加“Engine 运行 副本路由与协调”；本节只链接 22、18。
- 源码证据：
  - `ServeSubcommand.cmd` 在未设 `api_server_count` 且为 internal LB 时取 `args.api_server_count = args.data_parallel_size`（=2），因此走 `run_multi_api_server`，启动 2 个 API 进程，不走 5.3 的单 API `run_server`；
  - 13 §4.1 表对此写得正确；
  - DP 路由、wave、READY/shutdown 的 owner 是 13，但 5.6 没有链接它。
- 建议：在模板后注明“DP=2 默认 2 个 API server，走 `run_multi_api_server`”，并在完成与限制中链接 13 §2 / §4。

### P2（次要）

- **P2-1** 02 §2.2 第二段：“无 KV cache 的模型也会关闭 chunked prefill”被归到 `EngineCore._initialize_kv_caches`。实际在 `EngineCore.__init__`（`len(kv_cache_config.kv_cache_groups)==0` 分支）。建议改写归属，或在启动树 `Scheduler(...)` 前加 `[无 KV group] 关闭 chunked prefill`。
- **P2-2** 02 §3.8：“`vllm.engine.LLMEngine` 和 `AsyncLLMEngine` 是 V1 实现的别名”。`vllm/engine/__init__.py` 只有 license 头，别名实际在 `vllm/engine/llm_engine.py::LLMEngine`、`vllm/engine/async_llm_engine.py::AsyncLLMEngine`。06 §8 写对了。建议改成模块全路径。
- **P2-3** 02 §2.4 表 `EngineCoreRequest` 行写“资源调度接收”。Scheduler 实际接收的是 `EngineCore.preprocess_add_request` 经 `Request.from_engine_core_request` 转换出的 `Request`。建议写成“Engine 运行转换为 `Request` 后交给资源调度”。
- **P2-4** 02 §2.4 调用树：
  - `AsyncLLM.add_request` 在 `_add_request` 之前调用 `self._run_output_handler()`，这是后台 output_handler 的创建点，树里没有画出，最后一段只写“由 `_run_output_handler` 创建”。建议在 `add_request` 下补一条 `[首次] _run_output_handler`。
  - `AsyncLLM.generate` 的子树实际在流式消费者迭代时才执行，页面已注明“由响应路径迭代”，可接受。
- **P2-5** 02 §3.5：“MRV2 的 `execute_model` 先处理完成…再应用暂存的 block 写入，准备设备输入及 attention metadata，选择图执行路径”。源码 `vllm/v1/worker/gpu/model_runner.py::GPUModelRunner.execute_model` 在 `apply_staged_writes` 之后，先 `dispatch_cg_and_sync_dp` 选图和 DP 同步，再 `prepare_inputs` / `prepare_attn`。建议调整顺序措辞。
- **P2-6** 02 §5.3 Rust frontend 行：“配置 `VLLM_USE_RUST_FRONTEND` 与可解析的 `VLLM_RUST_FRONTEND_PATH`”。`vllm/envs.py` 中 `VLLM_RUST_FRONTEND_PATH` 默认 `"auto"`，由 `_resolve_rust_cli_path` 自动发现随包二进制，通常只需设 `VLLM_USE_RUST_FRONTEND=1`。建议注明默认 auto。
- **P2-7** 02 §5.6 调用树：
  - `APIServerProcessManager/RustFrontendProcessManager` 是在 `launch_core_engines` 的 with 体内构造的，并用 `gather_actual_addresses` 回填握手地址，树里画成了与 context 平级；
  - `setup_server` 在 `launch_core_engines` 之前，树中省略。
  - 建议把 manager 缩进到 `[context body]` 下。
- **P2-8** 02 §5.1 与 §5.6 场景清单不全，以下都是仓库中可执行的顶层用法：
  - `examples/deployment/llm_engine_example.py`（直接 `LLMEngine.add_request/step`）；
  - `examples/features/data_parallel/data_parallel_offline.py`（`LLM.__init__` 的 DP guard 报错信息直接指向它）；
  - `examples/features/torchrun/torchrun_example_offline.py`（external launcher）；
  - Anthropic / Cohere / Responses 等协议路由在 §5.3 变体表缺席（03 表已覆盖）。
  - 建议至少各给一行分类加 owner 链接。
- **P2-9** 02 §6 “下一页”表缺 13、14、15、18、22、25、26（有些只在正文或 Related Pages 出现）。E2E 读者交接不完整，建议补齐。
- **P2-10** 02 §3.8 可顺带指出：`vllm/entrypoints/openai/run_batch.py` 的弃用警告同样写着不存在的 `vllm run_batch`（CLI 注册名是 `run-batch`），与 `vllm server` 同类，属于可选的矛盾记录。
- **P2-11** 06 §6.2 与图 4：“R最后使用序号2…保留到fence 2”。源码 `Scheduler._free_request_blocks` 追加的是 `(self.sched_step_seq, blocks)`，即释放时的**全局**已调度序号，不是 `request.last_sched_seq`；后者只决定是否需要延迟。例子中两者恰好相等，但在 PP 队列 >2 时 fence 可能大于 R 的最后序号。建议写成“fence 取当时的 `sched_step_seq`（≥ R 的 last_sched_seq）”。
- **P2-12** 06 §7：“全局空闲时发 `wave_complete`、递增wave并将step counter归零”。`DPEngineCoreProc.run_busy_loop` 只在 `dp_rank == 0 or not has_coordinator` 时发 `wave_complete`（有 coordinator 时经 client_index -1），各 rank 都会递增 wave 并清零计数。建议补上发送者条件。
- **P2-13** 06 §1/§2：R 设定为“生成两个输出 token”，但页面只走到第一个 token。第二个 token、`finish_reason=length`、`_free_request` 以及前端 finished 没有用 R 收尾（§2.3、§6 只有泛化描述）。建议用一两句把 R 带到完成。
- **P2-14** 06 图 2：`E-->>C: F1` 与 `C->>Q: 保存F1 + S1 + execute future`。源码中非 pooling 且已执行的批次，队列第一元素是 `sample_tokens(non_block=True)` 返回的 future，execute future 是第三元素；图里 F1 来自 execute_model，两者混在一起。建议把 F1 标成 sample future，并单独标出 exec future。

P2 共 14 条。

---

## 4. E2E / 连贯性

### 4.1 02 的启动与单请求闭环是否覆盖每个 E2E 阶段，是否链接 owner

| E2E 阶段 | 02 位置 | 链接 owner | 结果 |
|---|---|---|---|
| 入口 / API（CLI、HTTP 路由） | §2.4 末段、§5.3 | 03（2.3.1）；13 仅链接给 gRPC/Rust/Omni，属于错配（F-4） | 部分；DP 路由 owner 13 未从 5.6 链接（F-7） |
| 请求渲染 / InputProcessor | §2.3.1、§3.1 | 03 | ✓ |
| EngineClient / EngineCore | §2.3.2、§3.2 | 06 | ✓，但默认 step 路径与 06/07/12 口径冲突（F-1、F-3） |
| Scheduler | §2.3.2、§3.3 | 07 | ✓ |
| KV | §3.3 | 08（22 在 3.7） | ✓ |
| Executor / Worker | §3.4 | 26、18 | ✓ |
| ModelRunner | §3.5 | 11、12、19 | ✓ |
| Attention / 模型前向 | §3.6 | 09、10、17、20、21 | ✓ |
| 采样 / 结构化输出 | §2.3.3 与调用树只有函数名 | **无（14 零链接）** | ✗（F-5） |
| 输出处理 / detokenize | §2.3.4、§3.1 | 03 只在 2.3.1 链接一次，2.3.4 无链接 | 弱 |
| 响应 / HTTP 流 | §2.3.4 | 03（间接） | 弱 |

术语：六模块名在 02 内一致。13 链接 06 时用“Engine 运行”，06 自称“Engine 架构”，index 用“Engine架构”，属于可接受的别名。真正的术语冲突是“普通/基本 step”：02、06 这样说，12 §5.3、07 §3 则写明默认 async。

### 4.2 静态视图 / 场景中没有 owner 的部分，以及 02 未放入架构的深度页

- **没有 owner 深度页的场景或组件：**
  - gRPC server（`vllm/entrypoints/grpc_server.py`）；
  - Rust frontend 进程管理（13 仅一句）；
  - `run-batch` 文件作业（只有 02，24 只提到 run_batch 插件）；
  - `chat/complete` 终端客户端（只有 02，影响小）；
  - scale-out token-in 端点（03 只有锚点，02 未列）。
- **02 从未放入架构的深度页：**
  - 14 采样与结构化输出（零链接）；
  - 16 投机解码（只在 §6 读者表）；
  - 15 多模态（只作为 serve 协议变体链接，没有说明它横跨 Renderer、encoder 预算、Runner 的哪些模块）；
  - 13 的实际职责（DP LB / wave / READY / shutdown）没有从 §3.2 或 §5.6 链接，反而被错配给 gRPC/Rust/Omni；
  - 其余 23、24、25、22、26、18、19、20、21、17、09、10、11、12 都有落点。

### 4.3 06 对 02 / 07 / 11 / 12 / 13 / 26 的交接

- 06 → 02：Related Pages ✓。
- 06 → 07：§2.2 预算与分配交给 07/08 ✓；07 §7 反向把 `dp_sync_interval` 交给 06，06 §7 已覆盖，双向一致 ✓。
- 06 → 11 / 12：§4.3 链接 ✓；11 §1.4、12 §2.x 均引用“06 §4”，该节存在，内容一致 ✓。
- 06 → 13：§7 ✓；13 §2.2、§3.3 把 `_process_engine_step` 闭环和终态交给 06 ✓，DPLB finished_requests 语义一致 ✓。
- 06 → 26：§3.2 与 Related Pages ✓；`FutureWrapper` drain 与 output_rank / aggregator 描述和 26 一致 ✓。
- 06 → 22：引用 “§5.4”，目标小节存在 ✓。
- **缺口 / 矛盾：** 06 §2 把同步 `step` 当作 R 的普通路径，与 12 §5.3 “异步调度默认开启” 以及 07 的 None 解析说明口径不一致（F-3）。06 的 Related Pages 没有 11（正文有链接，影响小）。

### 4.4 最重要的连贯性问题（按优先级）

1. 默认执行路径口径分裂：02、06 说 `step` 是普通路径，07、12 说默认 async 加 batch queue（F-1、F-3）。
2. 02 的 E2E 链路缺“采样/结构化输出”这一跳（F-5），15、16 也没有放进架构。
3. 02 §5.3 把 gRPC/Rust/Omni 交给不拥有它们的 13；13 的真实职责又没从 DP 场景链接（F-4、F-7）。
4. §5.5 render 场景缺 token-in 执行面（F-6），render → 生成的 E2E 没有闭合。
5. 延迟释放 gate 在 02 被泛化，与 06 §6.2 冲突（F-2）。
