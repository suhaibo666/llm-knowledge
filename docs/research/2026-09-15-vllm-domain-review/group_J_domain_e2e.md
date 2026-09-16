# Group J：vLLM 域级 E2E 连贯性与完备性审阅

- 范围：`wiki/02_engineering/03_infer_frameworks/vllm/index.md` + 01–26；上级 `03_infer_frameworks/index.md`、`wiki/index.md` vLLM 行、`wiki/courses/**`。
- 源码核对：`/Users/suhaibo/97-llm/vllm @ 199cb9b964822e59ab9b58d88e7be31eb419a2ae`（只读；commit 时间 2026-09-06T17:54-07:00 = **2026-09-07 00:54 UTC**）。
- 方法：抽取每页页头、H2/H3 目录、闭环/位置图、所有权与“归/见 NN”陈述、Related Pages；构建 26×26 wikilink 矩阵；自动解析跨页 `NN §x.y` 引用（78 处，76 处命中目标标题，2 处为正则误报），对可疑者人工打开目标章节；对疑似 orphan 能力逐一在 checkout 中确认文件/符号存在。未逐页核验事实准确性（由其他组负责）。
- 注意：审阅期间 `13_vllm_serving_control_plane_analysis.md` 出现**未提交的工作区修改**（`git status` 显示 M，页头“最近更新 2026-09-15”，行数 470→750）。本报告按工作区最新版本审阅 13。

**总体结论**：普通文本请求的在线闭环（HTTP → render → EngineCoreRequest → Scheduler → Runner → Attention → 采样 → update_from_output → OutputProcessor → SSE → 释放 → 指标）在域内**可以端到端读通**，相邻页的对象名（`EngineCoreRequest`/`SchedulerOutput`/`ModelRunnerOutput`/`EngineCoreOutputs`/`GrammarOutput`/`AsyncOutput`）拼写一致，未发现 P0。主要缺口集中在**启动链前后两端**（配置解析、显存 profiling 与 KV 预算）、**Scheduler→Runner 合同的生产端**、以及若干**无 owner 的用户可见能力**（tool/reasoning parser、beam search、n>1、pooling 执行、非 CUDA 平台、KV events）。约定层面有页头日期/格式漂移、两代页面模板并存、索引阅读箭头方向不一致等 P2 问题。

---

## 1. E2E 覆盖图

状态：**完整** = 机制在 owner 页有实质解释；**仅片段** = 只有一段/一行或散落多页无主解释；**仅链接** = 页面指向 owner，但 owner 未写；**缺失** = 无 owner；**重叠一致 / 重叠不一致** = 多页解释同一机制。

### (a) 启动链

| # | 阶段 | owner 页 § | 状态 | 交接/备注 |
|---|---|---|---|---|
| a1 | CLI / `LLM()` / `AsyncLLM` 入口 | 01 §3–4（使用）；02 §5.1、§5.3（调用树） | 完整 | 02 §5.3 `ServeSubcommand.cmd → run_server → build_async_engine_client_from_engine_args → AsyncLLM.from_vllm_config` |
| a2 | 配置解析 `EngineArgs → VllmConfig.__post_init__ → platform.check_and_update_config` | **无 owner**；片段：01 §6.3–6.4（batch 默认、`-O2`）、19 §3.2（graph mode 九处改写）、12 §2.1 / 11 §2.10（runner 选择）、07 §2.1（async_scheduling 解析）、18 §9（ParallelConfig） | **仅片段** | 见 F1 |
| a3 | 平台探测 | 24 §3.3（`resolve_current_platform_cls_qualname`、OOT 优先规则） | 完整（仅探测） | 平台对象提供什么能力（worker_cls、attention 选择、`check_and_update_config`）无 owner，见 F13 |
| a4 | 进程拓扑：API server ×N / DP Coordinator / EngineCore proc / workers | 13 §3.1、§4.1、§4.2；06 §3.1（Client 变体）；26 §2–3（WorkerProc 与 READY）；02 §3.8 | 完整（分散但一致） | 13 管 Engine 级 READY 与 Coordinator 订阅，26 管 Worker 级 READY；18 页头明确“进程启动/广播 RPC/响应 FIFO 归 26”，交接一致 |
| a5 | 模型类选择与构造 | 09 §1.5、§2.1–2.3 | 完整 | |
| a6 | 权重加载 / 量化后处理 | 09 §2.4–2.11；17 §4–§6 | 完整 | 17 页头“checkpoint 字节写入本身归 09”，与 09 页头一致 |
| a7 | KV spec 收集、layout 协商、non-causal 关闭 chunked/prefix | 10 §2.5–2.6；02 §2.2；09 §1.5 图 | 完整 | |
| a8 | 显存 profiling → 可用 KV 字节 → `num_gpu_blocks` | 名义 owner 08（09 图 1 标注“determine_available_memory … 08”）；实际片段：08 §3.1.2 一段、11 §2.9、12 §2.9、19 §6.3（graph 显存估算）、05 §2.3（容量校验案例） | **仅片段** | 预算公式、`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 是否计入、startup plan 均无 owner，见 F2 |
| a9 | KV backing 分配与层绑定 | 08 §3.1.2；10 §2.5 | 完整 | |
| a10 | compile / warmup / CUDA Graph capture | 19 §3.4、§5、§6；11 §2.9；12 §2.9 | 完整 | |
| a11 | Scheduler 构造 → READY → health | 02 §2.2；13 §4.2、§4.4；26 §3；23 §6.2 | 完整 | 13 §4.4 与 23 §6.2 对 `/health`→`check_health`→503 描述一致 |
| — | 启动顺序本身 | 02 §2.2、09 §1.5、10 §1.5/§2.5、11 §2.9、12 §2.9、19 §3.4 | **重叠一致** | 六处各画一遍，当前一致，但有漂移风险，见 F20 |

### (b) 一次请求

| # | 阶段 | owner 页 § | 状态 | 交接/备注 |
|---|---|---|---|---|
| b1 | HTTP / `LLM.generate` 调用 | 01 §3–5；02 §2.3.1、§2.4 | 完整 | |
| b2 | 模板渲染、tokenize、输出上限 | 03 §2.1–2.2 | 完整 | 01 §6.1–6.2 只给用法并交 03 |
| b3 | InputProcessor → `EngineCoreRequest`，前端 admission | 03 §2.3；02 §2.3.1 | 完整 | 03 → 06 对象名一致 |
| b4 | EngineCoreClient 传输（ADD、ZMQ、编码缓冲） | 06 §2.1、§3.1、§6.3 | 完整 | |
| b5 | `EngineCore.add_request → Scheduler.add_request` | 06 §2.1；07 §3.1 | 完整 | |
| b6 | Scheduler step：预算、running/waiting、抢占 | 07 §4–§7 | 完整 | |
| b7 | KV 分配、prefix cache 命中 | 08 §3.2–3.4 | 完整 | |
| b8 | **`SchedulerOutput` 的生产端字段合同**（`NewRequestData`/`CachedRequestData`、`kv_cache_block_copies` 等） | 应归 07；实际只在消费端 11 §3.4、12 §2.10 逐字段列出 | **仅链接（断链）** | 12 §2.10 称“字段怎样由 schedule() 生成归 07 §4.1、§8.1”，07 未写，见 F3 |
| b9 | Executor → Worker RPC、PP 收发 | 06 §3.2；26 §4；18 §5；02 §3.4 | 完整 | |
| b10 | Runner 输入准备（行、token 行、slot） | 11 §2.2–2.6；12 §2.2–2.4 | 完整 | |
| b11 | attention metadata 与 backend 调用 | 10 §2.8–2.11；11 §2.5；12 §2.10 | 完整 | 10 §2.11 块号交接链与 08/11/12 的指针一致 |
| b12 | 模型 forward、kernel、编译图派发 | 10 §2.10；19 §7；20；21 §3 | 完整 | |
| b13 | logits（`compute_logits`、`logits_indices`） | 14 §1.2；12 §2.6 | 完整 | |
| b14 | grammar bitmask、采样 | 14 §2–§6 | 完整 | 07 `get_grammar_bitmask` ↔ 14 §6.2 `GrammarOutput` 一致 |
| b15 | 投机解码 propose/verify/rollback | 16 §2–§8 | 完整 | |
| b16 | `ModelRunnerOutput`（异步 D2H、copy_event） | 11 §2.7–2.8、§3.4；12 §2.6、§2.10 | 完整 | 字段去向指向 07 §8.x，76/78 处 § 引用命中；1 处错指见 F7 |
| b17 | `update_from_output` 对账 | 07 §8.1–8.4 | 完整 | |
| b18 | `EngineCoreOutputs` 分客户端交付 | 07 §8.5；06 §2.3 | 完整 | |
| b19 | OutputProcessor：detokenize、stop string | 03 §3.1–3.2 | 完整 | owner 标注不一致见 F8 |
| b20 | 流式协议恢复（SSE、finish_reason、usage） | 03 §3.3；01 §5 | 完整 | tool/reasoning 解析只有一句，见 F4 |
| b21 | 完成、延迟释放、block fence | 07 §8.4；08 §3.5、§5.1.1；06 §6.1–6.3 | **重叠一致** | 06 §6.2 的 gate“KV consumer connector 且 `max_concurrent_batches > 1`”与源码 `Scheduler.__init__` 一致 |
| b22 | 指标、trace | 23 §3–§4；07 §8.6；04 §4（客户端口径） | 完整 | 04 与 23 分别为客户端/服务端口径，互不矛盾 |
| b+ | prompt logprobs | 14 称“实现归 11/12”；11/12 只列填写点 | **仅链接** | 见 F10 |
| b+ | `n>1` 父子请求 / beam search | 无 | **缺失** | 见 F11 |

### (c) 横切能力

| 能力 | owner 页 § | 状态 | 备注 |
|---|---|---|---|
| DP 路由、wave、LB 模式 | 13 §2–§4；18 §5.3；06 §7；19 §7.4 | 完整 | 13 → 18 §5.3 的交接存在 |
| TP/PP/EP/PCP/DCP、EPLB、DBO、Elastic EP | 18；13 §4.6 | 完整 | |
| 分离式 KV / connector / offload | 22；08 §5.4；07（blocked waiting） | 完整 | EPD 部署拓扑与 P/D proxy 自述无 owner，见 F15 |
| 多模态 | 03 §5.1；15；07 §5.3 | 重叠一致 | 07 §5.3 与 15 §4.1 都详写 `_try_schedule_encoder_inputs` |
| LoRA | 09 §4.2（manager 与 add_lora 链）；24 §4（resolver）；07（max_loras 准入）；11 §2.4 / 12 §2.4（行）；20 §9（kernel）；19（capture 特化） | 机制完整，**使用入口缺失** | 见 F16 |
| 插件 | 24 | 完整 | 页头字段是占位，见 F18 |
| 在线权重更新 | 25；17 §6.2；22 §10；09 §2.11 | 完整 | |
| sleep / wake | 25 §2.4（level 语义）；02 §5.9 | **仅片段** | 见 F14 |
| 可观测性 | 23；07 §8.6；05 §3；04 §4 | 完整 | KV events 断链，见 F9 |
| 故障传播 / FT | 23 §6；13 §4.4；26 §8.1 | 完整、一致 | |
| 关闭 | 13 §4.5；26 §8.2；23 §6.2；02 §5.3 | 完整、一致 | 26 与 23 对“等待 `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` → SIGTERM 4 s → kill”一致 |
| 结构化输出 / 投机 / 量化 / 编译 / Kernel / IR | 14 / 16 / 17 / 19 / 20 / 21 | 完整 | |
| pooling / embedding 执行 | 03 §5.2（API）；11 `_pool`、12 `pool()`；07 §8.4 | **仅片段** | 12 自述“本域暂无专页”，见 F12 |
| tool / reasoning parser、Responses / Anthropic / harmony | 03 §3.3 一段 + 03 表格各一行 | **缺失** | 见 F4 |
| run-batch、render、bench、collect-env | 02 §5.4、§5.5、§5.8；03 §5.3 | 完整（场景级） | |
| 非 CUDA 平台（CPU/ROCm/XPU/TPU worker 与 runner） | 24 §3.3（仅探测）；11 §2.10、12 §2.1 各一句 | **缺失** | 见 F13 |
| KV events 发布 | 07 §8.6 一句；22 §11.3 指向 23 | **仅链接（断链）** | 见 F9 |

---

## 2. 编号发现

### P0

无。

### P1

**F1 [P1] 启动链的“配置解析”阶段无 owner**
- 位置：全域；02 §2.2 的启动树从 `EngineCore.__init__` 开始，§5.3 从 `AsyncLLM.from_vllm_config` 开始；其余页各引用 `VllmConfig.__post_init__` 中与自己相关的一段。
- 证据：全域唯一提到 `create_engine_config` 的是 18 §9 的一行：“ParallelConfig 构造（pydantic，发生在 EngineArgs.create_engine_config 里，早于 VllmConfig.__post_init__）”。`current_platform.check_and_update_config` 只在 21 出现一次。源码：`vllm/engine/arg_utils.py::EngineArgs.create_engine_config`；`vllm/config/vllm.py::VllmConfig.__post_init__` 内调用 `current_platform.check_and_update_config(self)`。19 §3.2 自己就记录了“resolve 之前已经有九处”graph mode 改写，说明这一阶段本身有独立的顺序语义。
- 影响：读者无法回答“我给的 flag 在哪一步、按什么顺序被改写成最终配置”。01 §6.4 只能提示“运行后的最终配置还受硬件、模型能力与上述解析过程影响”。
- 修复：由 02 新增 §2.2.0（或独立小节）“配置解析”，给出 `EngineArgs.from_cli_args → create_engine_config → 各子 Config → VllmConfig.__post_init__`（优化级别默认、async_scheduling、runner 选择、cudagraph mode/size、compile ranges、`platform.check_and_update_config`）的顺序树，每项链接到 19 §3.2、12 §2.1、07 §2.1、18 §9、01 §6.3–6.4。另一种做法是指定 01 §6.4 只讲用户视角，02 负责顺序。

**F2 [P1] 显存 profiling 与可用 KV 预算没有 owner 级解释**
- 位置：08 §3.1.2（名义 owner）；09 §1.5 图 1 把 `determine_available_memory` 标为“：08”；11 §2.9、12 §2.9、19 §6.3、05 §2.3。
- 证据：08 只有一段：“`Worker.determine_available_memory()` 的 profile 路径从执行器预算扣除非 KV 开销，CUDA Graph 估算还受相应开关控制”。源码 `vllm/v1/worker/gpu_worker.py::Worker.determine_available_memory` 包含：`maybe_apply_startup_plan`、`memory_profiling(init_snapshot, weights_memory)`、`profile_cudagraph_memory`、`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 决定是否计入、`requested_memory` 与 `available_kv_cache_memory_bytes` 的推导，以及建议 util 的日志。`vllm/v1/worker/startup_plan.py`（`VLLM_ENABLE_STARTUP_PLAN`，跳过 profiling）在全域 0 次提及。
- 影响：01 §6.3、05 §2.3、04 §5 都把用户引向 08 去理解“Available KV cache memory / 块数从哪来”，但 08 没有公式，也没有算例；启动失败排障的机制依据断在这里。
- 修复：08 §3.1.2 扩写一个“预算推导”小节：`requested = total × gpu_memory_utilization`，减去权重、非 torch 与瞬时峰值、（按开关）graph 估算，得到 KV 字节，再经 `get_kv_cache_configs` 得到块数并做 null block 与最大长度校验。给一个数值例，并链接 19 §6.3（graph 估算）和 11/12 §2.9（profile_run 形态）。startup plan 在同节作为变体登记。

**F3 [P1] `SchedulerOutput` 的生产端合同无人负责，消费端指回 07 却落空**
- 位置：12 §2.10；11 §3.4；07 §4.1、§8.1。
- 证据：12 §2.10 写道“字段怎样由 `schedule()` 生成归 [[07…|Scheduler]] §4.1、§8.1”。但 07 全页 `scheduled_new_reqs`、`scheduled_cached_reqs`、`NewRequestData`、`CachedRequestData`、`_make_cached_request_data`、`kv_cache_block_copies`、`new_block_ids_to_zero`、`free_encoder_mm_hashes` 均为 0 次；07 §4.1 只有阶段表，“构造 `SchedulerOutput`”一行带过。源码：`vllm/v1/core/sched/output.py::SchedulerOutput / NewRequestData / CachedRequestData`。
- 影响：Scheduler→Runner 是 e2e 中最核心的合同。现在只能从两个 Runner 页的消费表反推生产规则。例如“`new_token_ids` 仅 PP 且非 async 时填写”“`all_token_ids` 只给 MRV1”只在 11/12 中出现，07 读者看不到这些发送条件。
- 修复：07 §4.1 末尾加“计划封装：`SchedulerOutput` 字段由谁、何时写入”表（22 个字段 × 生产函数 × 条件），列出 `_make_cached_request_data` 的 PP/async/MRV1 分支；11 §3.4 和 12 §2.10 的备注列改为链接该表，不再复述发送条件。

**F4 [P1] Tool-call / reasoning parser 与 Responses/Anthropic 协议无 owner**
- 位置：03 §3.3（一段）与 §4 表格一行；01 未提及；14 §6.4 只讲 grammar 跨 reasoning 边界。
- 证据：03 §3.3 写到“parser 将文本与 token 信息解释为 reasoning、正文或 tool calls”，03 表格中“OpenAI Responses”“Anthropic Messages”各只有一行。源码：`vllm/tool_parsers/`（52 个文件）、`vllm/reasoning/`（33 个文件）、`vllm/parser/parser_manager.py`、`vllm/parser/harmony.py`、`vllm/entrypoints/openai/responses/`、`vllm/entrypoints/anthropic/`、`vllm/entrypoints/mcp/`。`tool_parser`、`ToolParser`、`reasoning_parser`、`ReasoningParser`、`Harmony` 在全域均为 0 次。
- 影响：聊天服务的主要用户能力（`--enable-auto-tool-choice`、`--tool-call-parser`、`--reasoning-parser`、流式增量解析、finish_reason=`tool_calls`）既没有使用说明，也没有机制解释，e2e 的 b20 阶段只覆盖纯文本。
- 修复：短期在 03 新增 §3.4“parser 管线：选择、流式增量解析、与 stop/grammar 的先后”，并在 01 §4/§6 增加 tool/reasoning 启动参数用法。长期若篇幅不够，按 planning 规则新增一篇“工具调用与推理解析”页，由 index 的“请求与资源”表登记；Responses/Anthropic/harmony 放在同一 owner。

**F5 [P1] 排障指南引用不存在的“27”号页（断链交接）**
- 位置：05 §4 分层定位表“运行中批量报 EngineDeadError 或 health 503”行。
- 证据：“可恢复 fault-tolerance 状态与不可恢复死亡有不同合同，按 `27` 的状态边界处置”。域内只有 01–26；FT 状态机在 23 §6.3。
- 修复：改为 `[[23_vllm_observability_reliability_analysis|可观测性与可靠性]] §6.3`；13 §4.4 的部署侧结论可作补充链接。

### P2

**F6 [P2] 索引“阅读依赖”箭头方向与编号顺序不一致**
- 23 行写“调试与排障 → Scheduler、KV、Serving与分布式”，箭头后列的是**前置**页（23 §3–§4 依赖 07 的事件与 `update_from_output`，§6.2 依赖 26 的 worker monitor）。其他行都按“前置 → 后续”写。
- 26 行“Engine与分布式 → Serving、Runner与可观测性”，后续列的 13、11/12、23 都是更早编号的页；同时 02 §3.4、06、18 页头都把 26 当作 worker 启动/RPC 的 owner。26 放在“系统专题”末尾，读者在 06/18 就被引向尚未出现的页。
- 20 行前置含“IR”（21），21 行后续为“融合算子”（20）：编号与依赖倒置。
- 12 行依赖“Runner V1及其前置”，但 MRV2 才是默认 runner（12 §2.1 表第 5 行“默认 MRV2”），读者会先学回退路径。
- 修复：23 行改为“Scheduler、KV、Serving、MultiprocExecutor → 调试与排障（反查）”。26 移到“请求与资源”表中 06 之后，或在 06 行后续中登记。20/21 行加注“建议先读 21 §1–§3 再读 20”，或在索引中调换两行顺序（不改文件编号）。12 行改为“Scheduler、KV与Attention → 生成特性；与 11 可独立阅读”。

**F7 [P2] 11 §3.4 跨页 § 引用错位**
- 位置：11 §3.4 输出表 `ec_connector_output` 行。
- 证据：“15 §7.4：`update_from_output()` 交 `ECConnectorBase.update_connector_output()`”。15 §7.4 是“`prompt_embeds`：两条独立通路”，EC transfer 在 15 §7.7；且 `update_connector_output` 在 15 全页 0 次。
- 修复：改为“15 §7.7（EC 协议）；调度侧消费见 07 §8.3”，或在 15 §7.7 补一句 Scheduler 侧 `update_connector_output` 的调用点。

**F8 [P2] “前端 OutputProcessor”的归属标注不一致**
- 12 §1.5 图：`F["前端 OutputProcessor<br/>归 06"]`；14 §1.2 图：`FE["前端 OutputProcessor<br/>归 03 与 06"]`。detokenize、stop string 与 collector 的实质解释在 03 §3.1–3.2；06 §2.3 只描述输出队列交接。
- 修复：统一为“归 03（传输交接见 06 §2.3）”。

**F9 [P2] KV events 发布：22 指向 23，23 没有这部分内容**
- 22 §11.3：“`VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES` | `True` | KV event 发布 | → **23**”。23 中 `KV event`/`KVEvent` 为 0 次；07 §8.6 只有“合成 `KVEventBatch` 后发布”一句；08 §4.4 表格一行。源码：`vllm/distributed/kv_events.py`（574 行，`BlockStored`、`KVEventBatch`、publisher）。
- 修复：在 23 §5 增加“KV cache 事件”小节（事件类型、发布通道、`kv_events_config`、与 prefix cache 路由的消费关系），或把 22 的指针改到 08 并由 08 §4.4 补全。

**F10 [P2] prompt logprobs 算法只有链接**
- 14 §1 写“`PromptLogprobsWorker` 在采样之后由 runner 调用，实现归 11/12”，但 11 只有 `_bookkeeping_sync()` 调 `_get_prompt_logprobs_dict()` 一行，12 只有“`PromptLogprobsWorker` 在 `sample_tokens()` 中计算”一行。源码 `vllm/v1/worker/gpu/sample/prompt_logprob.py::PromptLogprobsWorker`。跨 prefill chunk 的累积与切片无人解释。
- 修复：由 14 §3.5 接管 prompt logprobs 的计算形状与分块规则（与 sample logprobs 并列），11/12 保留填写点。

**F11 [P2] `n>1` 并行采样与 beam search 无 owner**
- `n>1`：06 §2.1“`n > 1` 会有多个 child；这里先固定单个 R”；03 §3.1 只提 parent/child 聚合。源码 `vllm/v1/engine/parallel_sampling.py::ParentRequest`。
- beam search：03 中两次出现，都是“被拒绝”的组合。源码 `vllm/entrypoints/generate/beam_search/offline.py`（457 行）、`online.py`（222 行）。
- 修复：03 §5 增加“一个请求变多个核心请求：`n>1` fan-out 与 beam search 前端循环”，说明 child id、输出聚合、与 prefix cache 的关系。

**F12 [P2] pooling / embedding 的执行侧无专页**
- 12 §2.6：“pooler 算法与 `PoolingRunner` 的内部状态本域暂无专页”。源码 `vllm/model_executor/layers/pooler/`、`vllm/v1/worker/gpu/pool/pooling_runner.py`、`vllm/v1/pool/late_interaction_runner.py`。02 §5.2 与 03 §5.2 只覆盖场景和 API。
- 修复：在 09 §4 增加“pooler 模型接口”接缝小节，或在 index 显式登记为未覆盖。

**F13 [P2] 非 CUDA 平台与平台接口无 owner，索引未声明范围**
- 11 §2.10：“本域暂无页面专门展开平台 runner”；24 §3.3 只讲探测。源码 `vllm/platforms/{cpu,rocm,xpu,tpu,zen_cpu}.py`、`vllm/v1/worker/{cpu_worker,xpu_worker}.py`。01/04/05 页头限定 NVIDIA，但 index 与 02 §1.3 能力表都没有声明平台覆盖边界。
- 修复：index 页头“目录范围”加一句“机制页以 CUDA/NVIDIA 为主线；平台接口与 CPU/ROCm/XPU runner 暂无 owner”；02 §3.8 补 `Platform` 接口承担的选择点清单。

**F14 [P2] sleep 模式只有语义，没有机制**
- 25 §2.4 给出 level 0/1/2 语义；`CuMemAllocator`、`sleep_mode_backend` 在全域 0 次。源码 `vllm/device_allocator/cumem.py`（426 行）、`sleep_mode_backend.py`、`Worker.sleep`。
- 修复：25 §2.4 补“tag 化内存池如何 offload/丢弃与恢复”，或在 index 登记未覆盖。

**F15 [P2] EPD/encoder-only 部署拓扑与 P/D proxy 自述无 owner**
- 22 §14.4：“**本域仍无 owner**，与 `15:§9.4` 的空白登记互指”；“`examples/disaggregated/` 的 proxy/router 参考实现 … 建议归 **13**；基线下全库 wiki 无人展开”。13（含 09-15 工作区更新）未接收。
- 修复：13 §4.1 增补“P/D 与 EPD 部署拓扑”小节，或在 index 显式登记为缺口，避免两页互指成环。

**F16 [P2] LoRA 机制分散，缺使用入口与串联**
- 机制链可以拼出来（24 §4 resolver → `engine_client.add_lora` → 09 §4.2 `Executor.add_lora → … → LRUCacheWorkerLoRAManager` → 11/12 行 → 20 §9 kernel），但没有页面给出 `--enable-lora`、`--lora-modules`、`/v1/load_lora_adapter`（`vllm/entrypoints/serve/lora/api_router.py`）和“请求 `model` 名 → LoRARequest”的用法。`LoRARequest` 全域 0 次，`load_lora_adapter` 仅 08 §2.3 提到一次 unload。
- 修复：01 §6 增加 LoRA 服务用法并链接 09 §4.2；02 §3.6 的 LoRA 段补一行链路图作为 hub。

**F17 [P2] 页头基线行的日期与格式漂移**
- commit 为 2026-09-07 UTC，但 19、20 写“（`main`，2026-09-08）”，21、22、23 写“（`main`，2026-09-06）”（太平洋本地日期）。
- 格式有四种：01–06、13、18 为“（`main` 快照，2026-09-07 UTC）”；07、08 的 main 无反引号，08 句末多“。”；09–12、14–16、24–26 省略“快照/UTC”；17 为“只读 main 快照”。
- 修复：统一为 index 的“`vllm-project/vllm@199cb9b…`（`main` 快照，2026-09-07 UTC）”。

**F18 [P2] 24–26 页头字段是占位文本**
- 24：“**主题**：vLLM 扩展插件系统（机制分析）”，“**最近更新**：2026-09-14”后无变更说明；25、26 相同（“功能分析”“并发与分布式机制分析”）；适用范围只有对象列表，没有“不拥有/归 NN”陈述。三页页头后另有孤立的 `---`。
- 修复：按 01–23 的格式补主题句、所有权边界（例如 26“TP/PP 张量合同归 18，Future 与计划配对归 06”）和更新说明。

**F19 [P2] “最近更新”与索引/工作区状态不一致**
- index 写“最后更新：2026-09-14”，13 工作区已为 2026-09-15（未提交）。02 页头写 2026-09-10，但 09-14 的 `f268b46` 对 02 有 792 行改动；06 页头写 2026-09-08，同一 commit 也改了 06（4 行）。git 提交是批量的，这项只作低置信提示。
- 修复：13 提交时同步 index 与 `wiki/changelog.md`；02 页头按实际最后一次实质修改更新。

**F20 [P2] 重复解释（当前一致，但有漂移风险）**
- 启动顺序：02 §2.2、09 §1.5 图 1、10 §1.5/§2.5、11 §2.9、12 §2.9、19 §3.4 各写一遍。
- 请求单步闭环：02 §2.3–2.4 与 06 §2.2 几乎逐步重复；另有 11 §1.4、12 §1.5、14 §1.2、15 §1.2、16 §1.1 的“位置图”。
- encoder 准入：07 §5.3 与 15 §4.1 都详写 `_try_schedule_encoder_inputs`，15 页头却写“一般调度归 Scheduler 页”。
- 延迟释放：06 §6.2、08 §3.5/§5.1.1、07 §8.4。
- 修复：指定 02 §2.2 与 06 §2.2 为规范版本，其余页的位置图只保留本页负责的节点，并链接规范版本；15 页头改为“encoder 准入的调度侧细节与 07 §5.3 共享，本页负责 embedding 行口径”。

**F21 [P2] 两代页面模板并存**
- 09–12、14–20 有“位置/闭环图 + 核心流程清单 + 调用树 + 所有权 + 配置契约 + 成本账”；06、07、08、13、21–26 大多缺“核心流程清单/配置契约/成本账”（例如 07 的核心流程清单为 0，而 11 页头写“按 07 的流程覆盖标准”）。
- 修复：在 `maintaining-llm-knowledge` 或 index 中明确哪些分节是必需的；缺的页按需补“核心流程清单”，至少保证所有权陈述。

**F22 [P2] 术语：裸 “V1/V2” 指代 runner 代际**
- 14 有约 60 处、16 约 46 处裸 “V1/V2”，例如 14 §3.2 标题“V1 native：用指数噪声竞赛实现 categorical sampling”，这里指 MRV1 的 sampler。02 §3.8 明确警告“Engine V1 与 Model Runner V1/V2 是两个版本维度”。
- 修复：14、16 统一改为 MRV1/MRV2（首次出现注明 Model Runner V1/V2）。

**F23 [P2] 03 的定位与用户期望不一致，指南线缺“API 调用参考”**
- 用户把 03 视为使用说明；但 03 页头为“拥有接口与任务语义…”，含源码阅读路线，index 也把它放在“请求与资源”机制表。指南线实际是 01 → 04 → 05，01 §4–6 只覆盖 chat/completion 基本字段。
- 修复：二选一。其一，保持 03 为分析页，在 01 增补“常用 API 与参数速查”（tools、logprobs、n、response_format、多模态消息）并链接 03/14/15。其二，把 03 前半改写为指南，机制内容下沉。无论哪种，index “读者入口”表都应说明 03 的类型。

**F24 [P2] 架构页的出口、外部学习路径与课程**
- 02 §6“小结与专题阅读入口”漏列 13、14、15、18、22–26。
- `01_llm_inference_technology_stack_analysis.md` 的学习路径为“index → 06（称‘引擎架构与请求生命周期’）→ 01”，跳过 02 软件架构。
- `wiki/courses/` 没有任何 vLLM 课程；`torch_compile_end_to_end.md` 没有链接 19/21（反向链接只在 PyTorch CUDA Graph / Pass 页）。宪法不要求一定建课程，这里只作建议。
- 修复：02 §6 表补全或改为“完整列表见 index”；技术栈页学习路径第 2 步改为 02；可在 torch_compile 课程末尾加 19/21 作为“生产框架应用”。

**F25 [P2] 01 的文件名与 H1 不符**
- `01_vllm_feature_optimizations_guide.md` 的 H1 为“vLLM 使用指南：从安装到离线推理与流式服务”，“feature optimizations”是旧版遗留 slug。改名涉及全库链接，只建议在下次重命名批次中处理，并保留别名。

---

## 3. 阅读顺序与依赖（问题 2 小结）

- 五张表（读者入口 / 请求与资源 / 模型与设备执行 / 生成与模型特性 / 系统专题）作为学习路径基本合理：先 02 建立六模块，再按“请求→资源→执行→生成特性→系统专题”下沉。
- 问题集中在 F6：23 箭头方向反了、26 位置太晚、20/21 依赖倒置、11 先于默认 12。
- 未发现“依赖只在后编号页定义、且未链接”的硬断点：前向依赖都有 wikilink（例如 18→26 有 4 处链接，20→21 有 6 处，07→11/12/14/16 均有链接）。唯一实质性的“指回却落空”是 F3（12→07 字段生产）和 F10（14→11/12 prompt logprobs）。
- 链接矩阵显示 01 的入链只有 4 页，21 只有 4 页，24 只有 5 页；06 与 10 之间没有互链（06 不需要 10，可接受）。

## 4. 约定一致性（问题 3 小结）

- 基线提交哈希 26 页全部一致；日期与格式漂移见 F17，24–26 页头占位见 F18，“最近更新”与状态不一致见 F19。
- 分节命名：两代模板并存（F21）；Related Pages 27/27 都有。
- 术语：交接对象名一致（没有 `SchedulerOutputs`/`ModelRunnerOutputs` 等变体）；MRV1/MRV2 与裸 V1/V2 混用见 F22；“行”在 11（row/请求行）、12（稳定行/state row）、14（logits 行）中含义不同，14 §4.1 已用 `idx_mapping` 做了区分，暂不列为问题。
- 跨页 § 引用：78 处中 76 处命中目标标题且语义对得上；1 处错位（F7）；无指向不存在章节的引用；另有 1 处指向不存在的页号（F5）。
- 所有权冲突：只有 OutputProcessor 标注不一致（F8）。其余“归 NN”陈述两两核对后一致，例如 17↔09（checkpoint 写入）、18↔26（进程/RPC）、19↔21（PassConfig 16 字段归 21、其余 CompilationConfig 归 19）、20↔17（量化 ABI）、22↔06（`delay_free_blocks` 协议 vs step fence）、11/12↔08（Scheduler 侧 CoW 引用）。

## 5. 指南与分析的衔接（问题 4 小结）

- 01 → 04 → 05 构成连贯的用户旅程：01 §7 明确分流到 04、05、02；04 §5 表每行给出“机制入口”；05 §1 声明“本页只消费这些模块暴露的信号”，§2.3 只给容量校验边界并交 08。三页没有明显复述内部机制。
- 断点：05 → 23 的交接写成“27”（F5）；05/01/04 指向 08 理解 KV 预算，而 08 缺公式（F2）；01 缺 tool/reasoning、LoRA、多卡部署的用法（F4、F16），这些用户需求目前只能直接进入分析页或无处可去。
- 01 §5 把“取消、输出收集和故障传播的内部合同”交给 23。输出收集与故障传播在 23 §6.2 有覆盖；取消的内部合同实际在 06 §5.2 与 07 §3.1。建议补上 06 链接。

## 6. 索引与总索引（问题 5 小结）

- 页数：vLLM 目录 26 篇 + index = 27，与 `wiki/index.md` 行“vLLM知识地图 | 27 | 活跃”及上级 index“26 篇正文 + index”一致。
- index 各行标题、问题句与各页 H1/范围一致；抽查的数字性陈述与源码一致（22“16 个注册 connector”对应 `factory.py` 中 16 次 `register_connector`）。
- 上级 `03_infer_frameworks/index.md` 写“使用、调优、排障、架构与机制五类入口”，与 vLLM index 的五张表一致。
- 课程层没有 vLLM 链接（F24）。

## 7. 已核验、无问题的点（供其他组参考）

- `defer_block_free` 的开启条件：06 §6.2 与 `vllm/v1/core/sched/scheduler.py::Scheduler.__init__` 一致（`max_concurrent_batches > 1 and kv_transfer_config.is_kv_consumer`）。
- `/health` 语义：13 §4.4 与 23 §6.2 一致；worker 关闭升级顺序：23 §6.2 与 26 §8.2 一致。
- 平台 runner 继承关系：11 §2.10（`cpu_model_runner.py`/`xpu_model_runner.py` 继承 MRV1）与 12 §2.1（`cpu/model_runner.py`、`XPUModelRunnerV2` 继承 MRV2）并不矛盾，源码两套文件都存在。
- 07 §8.5（`EngineCoreOutputs` 分客户端）→ 03 §3.1（RequestState/OutputProcessor）字段名称与语义对齐。
