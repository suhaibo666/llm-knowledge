# vLLM 01–26 全域独立评审（2026-09-15）

本轮对象是重构完成后的 vLLM 域：`wiki/02_engineering/03_infer_frameworks/vllm/` 下的 26 篇正文和 index。评审标准按用户指定分三类：

- **使用说明**：01、03，另含 04、05 两篇指南。检查引导逻辑和内容完备度。
- **架构分析**：02。检查 software-architecture profile 和 8 项架构检查。
- **特性分析 / 功能点介绍**：06–26。检查基础五项 rubric 和 feature review。另查每个功能点是否写明目的、输入输出、处理逻辑、边界约束和支持范围。

另有一项全域要求：内容连贯，能端到端讲完整。

- **源码**：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`，本机 `/Users/suhaibo/97-llm/vllm`。commit 时间为 2026-09-07 00:54 UTC。全程只读，前后 HEAD 一致，只有既存未跟踪的 `artifacts/`。
- **组织方式**：10 个独立评审者均未参与写作。9 组按页分工，1 组做域级 E2E 连贯性评审。协调者负责机械门禁、内容保留对账、P0 抽样复核和汇总，没有修改任何 wiki 页面。
- **明细**：各组报告在 [`2026-09-15-vllm-domain-review/`](2026-09-15-vllm-domain-review/)，含每页锚点抽查表、页面原文摘录、源码证据和修复建议。

## 0. 结论

| 维度 | 结果 |
|---|---|
| 页面判定 | **PASS 5 篇**：04、07、10、13\*、21。**REJECT 21 篇** |
| 严重度 | **P0 7 条**：6 页各有事实错误，其中 17 有 2 条。**P1 约 60 条**。**P2 约 150 条** |
| 机械门禁 | 全绿：links 452 页五项为 0；vLLM 目录 math、markdown、assets 均为 0 错误 0 警告；无 `path:line` 引用，不触发 locator 门禁 |
| 内容保留 | 相对重构前 `dfbd2fa` 未发现概念级静默丢失。24、25、26 的源码路线把一批 `path::symbol` 锚点和测试名压成了点分 qualname 或裸符号，属 P2 |
| E2E | 普通在线文本请求可以沿 26 页端到端读通，交接对象名一致（`EngineCoreRequest`、`SchedulerOutput`、`ModelRunnerOutput`、`EngineCoreOutputs`）。主要缺口有三类：默认执行路径口径分裂、约 20 处“互相推给对方”的无主功能点、启动链两端和若干用户能力无 owner |

\* 13 在评审期间有他人未提交的工作区修改：页头改为 2026-09-15，内容 +281 行。G 组判 PASS，J 组按工作区版本审阅。修改提交后应复审 13。

21 篇 REJECT 中，6 篇含 P0。其余 15 篇的主因多是功能点缺口或跨页交接断链，不必整页重写；例外是 01 和 03，涉及使用说明完备度与页面分类，需要较大调整。

### 需要优先修复的 P0（均已由协调者在源码中抽查属实）

| # | 页面 | 错误 | 源码事实 |
|---|---|---|---|
| 1 | 02 §2.3.3、§2.3.4、§2.4 调用树 | 把 `EngineCore.step` 写成普通路径：结果为 `None` 才调 `sample_tokens` | `SchedulerConfig.async_scheduling=None` 在 `VllmConfig.__post_init__` 中默认解析为 True，`max_concurrent_batches=2`。因此 `EngineCore.__init__` 选 `step_with_batch_queue` 和 `AsyncScheduler`，`execute_model` 之后立即以非阻塞方式调 `sample_tokens` |
| 2 | 14 §4.5（§1.3、§2.1、§4.4、§7.1 同一说法） | 有 draft 时 `Sampler.apply_sampling_params`“一行都不执行” | V2 的 `spec_decode/rejection_sampler.py::RejectionSampler._verify` 先调 `self.sampler.apply_sampling_params` 再 `rejection_sample`。V1 的 `RejectionSampler.forward` 执行 `apply_logits_processors` 和 `apply_sampling_constraints`。min_p、logit_bias 不可用来自 `SamplingParams._validate_spec_decode`。此处与 16 §7、§10.1 矛盾 |
| 3 | 17 §1.1 #1–#2、§8.3 调用树 | 两个调用者写错 | `_verify_quantization` 由 `ModelConfig.__post_init__` 调用。`resolve_quantization_config` 由 `EngineArgs.__post_init__` 调用 |
| 4 | 17 §4.3 | 缺 q_scale 时回退为 k_scale | `BaseKVCacheMethod.process_weights_after_loading` 先 `_q_scale.copy_(k_scale)`。但尾部 `layer.q_scale` 仍 < 0，于是 `q_scale = 1.0` 并再次 `copy_`，最终值为 1.0。源码的 warning 文案本身有误导，页面应补矛盾说明 |
| 5 | 18 §5.1 | 称 §7.1 证明“MRV2 拒绝 PP” | `gpu/model_runner.py` 在 `use_pp` 时构造 `PPHandler`，MRV2 支持 PP。拒绝 PP 的是 MRV2 的 DBO（`_get_dbo_unsupported_features`）和 PCP；§7.1 那条 assert 针对的是 SP |
| 6 | 19 §7.3、§7.5 | MRV1 派发入口 `GPUModelRunner._dispatch_cudagraph` | 源码中不存在该符号。实际入口是 `GPUModelRunner._determine_batch_execution_and_padding` 内的 `dispatch_cudagraph` 闭包，页面未写：SP padding、cascade/encoder 输出排除 FULL、`force_eager`、DP 协调后重新派发 |
| 7 | 20 §1.6 ④⑥⑧（图 2、§4.3、§10） | 说普通默认路径运行 vLLM C 的 add+RMSNorm、rope、activation kernel | 默认 inductor 且 mode≠NONE 时 `custom_ops` 追加 `"none"`，且 `CudaPlatformBase.get_default_ir_op_priority` 返回 `["native"]`，这三个 C kernel 都不走。此处与本页 §3.2 自相矛盾 |

## 1. 使用说明：01、03（附 04、05）

| 页 | 引导逻辑 | 完备度 | 判定 | 关键问题 |
|---|---|---|---|---|
| 01 | 基本合格。读者与范围 → 前置条件 → 离线/在线最小示例 → 流式 → 参数 → 输出字段 → 下一步，顺序正确，示例可照写 | **不合格**。只覆盖文本 chat、generate 和 streaming | REJECT | 见下方 01 问题清单 |
| 03 | **不是使用说明**。按内部流水线 Renderer → InputProcessor → OutputProcessor 组织，要先认识 `EngineInput`、`RequestState` 才能读下去 | 全页没有 curl/SDK 请求示例；除输出上限外，请求字段没有默认值或约束表，也没有响应样例 | 按使用说明 REJECT；按分析 rubric 五项通过 | 入站合同断两处：14 §0 称 `response_format` → `StructuredOutputsParams` 映射归 03，但 03 未写；01 §4 把 `return_token_ids` 推给 03，03 也未写 |
| 04 | 合格。§7 案例数字自洽 | 以文本生成为主，符合页头声明 | PASS | P2：`--ready-check-timeout` 默认 0 时跳过 ready 检查 |
| 05 | 合格 | 插件和平台加载失败只有一句，未链接 24 | REJECT | §4 表两处 owner 引用错误：“按 `27` 的状态边界处置”指向不存在的页，应为 23 §6.3；编译/graph 行的下一 owner 写 23，应为 19。另：§2.5 用 `request_success_total` 证明恢复，但该计数也含 abort 和 error |

**01 问题清单**

- **§6.1 事实错误**：称 `generation_config=vllm` 不读模型的 generation_config.json。实际 `ModelConfig.try_get_generation_config` 对 `{"auto","vllm"}` 都会读取，EOS id 仍通过 `update_from_generation_config` 进入停止条件。04 §2.1 同错，且与 03 §2.2 矛盾。
- **§2.2 未标注冲突**：写 CUDA 12.9 wheel，但 `envs.VLLM_MAIN_CUDA_VERSION` 默认 `"13.0"`，`requirements/cuda.txt` 为 cu13。安装前提应补 R580+ 驱动。
- **§7 下一步断链**：没有指向 03、14、15、16、17、18、24 等能力入口。

**全域使用能力覆盖（A 组从基线枚举）。** 枚举来源为 `LLM` 及其 mixin 的公开方法、`vllm/entrypoints/**` 的全部路由和 `vllm/entrypoints/cli/**` 子命令。

| 类别 | 能力 |
|---|---|
| 有使用说明 | `LLM.generate/chat`、`/v1/chat/completions`、`/v1/models`、`/health`、`/metrics`、profile、`vllm serve`、bench serve/throughput/latency、collect-env、prefix caching |
| 只有语义或机制，缺可操作说明 | embeddings/pooling/classify/score/rerank、transcription、realtime、多模态输入、结构化输出、reasoning parser、LoRA（`--enable-lora`、`/v1/load_lora_adapter`）、量化 checkpoint、投机解码配置、TP/PP/DP 部署。部署命令散落在架构页 02 §5 |
| 全域缺失 | beam search、tool calling（`--enable-auto-tool-choice`、`--tool-call-parser`）、`/v1/responses`、Anthropic `/v1/messages`、`/tokenize` 与 `/detokenize`、`/generative_scoring`、`/v1/chat/completions/batch`、`/v1/audio/translations`、`/invocations`、bench startup 与 mm-processor、Docker、非 CUDA 平台、插件打包 how-to |

**结论。**“安装 → 跑通 → 调用各类 API/特性 → 读输出 → 调优 → 排障 → 扩展”这条用户旅程中，“调用各类 API/特性”一步没有使用说明 owner。这是**页面合同与分类变更**，应回到 `planning-codebase-analysis` 决定，候选方案有两个：

- **方案 A（评审者推荐）**：03 保留为请求语义机制页，另立“API/特性使用手册”页，按 endpoint 分节，吸收 02 §5 的可运行命令。
- **方案 B**：原地把 03 改成按 API 分节的手册，现有机制内容压成附录，并按保留规则逐项迁移。

用户把 03 归为使用说明，与 index 和文件名 `_analysis` 的现有归类冲突。无论选哪个方案，都要同步 index 分组。

## 2. 架构分析：02

判定：base rubric 的 hop-walk 为 FAIL（§2.3.3、§2.4 默认路径选错，即 P0 #1）。8 项架构检查中，item 1、2、3、4、6、7 通过；**item 5 场景清单和 item 8 读者交接未通过**。

- **item 5**：缺 scale-out 场景，即 render → token-in 往返。`register_scale_out_api_routers` 只在 render 服务或 `--tokens-only` 下自动启用，其余要 `VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1`。§5.6 的 DP=2 模板默认启动 2 个 API server，走 `run_multi_api_server`，不是 `run_server`，且未链接 DP 路由 owner 13。
- **item 8**：全页链接 14（采样/结构化输出）0 次；15、16 只出现在读者表或协议变体中，没有架构落点。§5.3 把 gRPC、Rust、Omni 交给 13，但 13 基本没有这些内容，`grpc_server.py::serve_grpc` 在域内无 owner。
- **§2.4 测试条目**：把 `test_abort_defers_free` 描述为普遍延迟释放。实际只在 KV consumer connector 且多批在途时成立，与 06 §6.2 冲突。
- **P2 共 14 条**，例：`vllm.engine.LLMEngine` 别名不存在（`vllm/engine/__init__.py` 为空）。

## 3. 特性分析：06–26

| 页 | 判定 | 最小失败单元（P0/P1） |
|---|---|---|
| 06 Engine | REJECT | §2.2、§4.1：未说明默认 `async_scheduling` 解析，也未交代 Scheduler/AsyncScheduler 选择轴，与 07、12 口径冲突 |
| 07 Scheduler | **PASS** | 16/16 锚点一致。P1 为跨页问题：滑窗/chunked-local 的 admission cap 与 08 互相推给对方；stale output 与 fence 在 06 和 07 各完整写一遍 |
| 08 KV Cache | REJECT | §4.5：缺 hybrid KV manager 开关（`disable_hybrid_kv_cache_manager` → `unify_hybrid_kv_cache_specs`，及 `VllmConfig.__post_init__` 的自动关闭条件）。§2.3：prefix hash 算法（默认 sha256）、非加密 hash 的随机种子、只在启用 prefix cache 或 connector 时计算，均未写。§5.1：partial hit/CoW 只存在于含 Mamba align group 的 hybrid 模型，页面写成通用行为。Mamba align 块生命周期与“spec 下哪些尾 token 可缓存”分别被 07/12 和 16 委托过来，本页未接住 |
| 09 模型库 | REJECT | §2.4、§2.6：MoE 专家权重写入（`RoutedExperts.weight_loader`、`make_expert_params_mapping`）无 owner；packed 量化参数切片（`adjust_shard_indexes_for_packing` 等）09 与 17 互相推给对方 |
| 10 Attention | **PASS** | P1 为跨页问题：KV scale 在 attention kernel 中的消费方式与 17 互相推给对方 |
| 11 MRV1 | REJECT | §1.4：未说明“默认 async ⇒ batch queue”。时序图出现 async 位于 `step()` 内部，这种组合不存在。MRV1 的 `sample_tokens` 不会返回 `None` |
| 12 MRV2 | REJECT | §1.5：同上 |
| 13 Serving | **PASS**\* | 需在未提交修改落地后复审。跨页问题：Elastic EP 编排与 Ray DP 监督无 owner |
| 14 采样 | REJECT | P0 #2。另有两处 E2E 问题：投机下的 p 由哪些约束构成无 owner；spec 下 `logprob_token_ids` 边界无 owner |
| 15 多模态 | REJECT | §4.2：缺每步释放 `Scheduler._free_encoder_inputs`，条件为 `offset+len+lookahead <= computed − placeholders`。index 把“E 何时释放”归给本页 |
| 16 投机解码 | REJECT | §5.1：Step3.5 在 `use_multi_module_mtp()` 下是 `MultiModuleMTPSpeculator`；V1 变体表漏 `EagleProposer`、`DFlashProposer`、`Gemma4Proposer`、`ExtractHiddenStatesProposer`。§8.3：称 async 下 draft 从不回 CPU，实际结构化输出走 deferred 分支，`DraftTokensHandler` 会拷回 CPU |
| 17 量化 | REJECT | P0 #3、#4。§4.2–§4.3：`CompressedTensorsKVCacheMethod` 完全覆写 post-load，未列。Marlin workspace 与 `g_idx_sort_indices` 在 reload 下的地址稳定性，17、25、19、20 四页口径不一 |
| 18 分布式 | REJECT | P0 #5。§1.2、§5.2：PP 反向采样同步只在 MRV2（`PPHandler`）存在，页面未把 runner 代际列为选择轴。`reinitialize_distributed` 实际在 `DPEngineCoreProc` |
| 19 编译/Graph | REJECT | P0 #6。图 1 把 sampled token 写回 `InputBuffers`，与 12 §2.6 矛盾。图 2 启动顺序错：首次编译发生在 `profile_run`，`resolve_cudagraph_mode_and_sizes` 位于 `initialize_kv_cache`。“compile_sizes 必须匹配 capture ladder”只对 MRV1 成立。MRV1/MRV2 的 LoRA capture 集合与 key 过滤差异未写 |
| 20 融合算子 | REJECT | P0 #7。其余 P1：PassConfig 开关归属写成 19，19、21 和本页 §12 均写归 21。`swiglu_limit` 矛盾框的更正本身是错的。norm+quant 漏两个静态 FP8 kernel。MoE backend 改名不止 TRITON→BATCHED_TRITON。QK-Norm+RoPE+KV 融合的落点写错，21 §6.3 是对的。oracle 9 个只覆盖 2 个。LoRA 分组与 rope 配对只有表格，没有原理图 |
| 21 IR/Pass | **PASS** | 交接问题：把 OOT 实现注册、tolerance、`register_impl` 交给 20，但 20 没有这些内容 |
| 22 分离式 KV | REJECT | §9、§5.3：HMA 下 NIXL READ 失败不写 invalid 块，却仍报 `finished_recving`，Scheduler 会缓存未写入的块，页面未写此后果。proxy → P → `kv_transfer_params` → D 这一段指向 13 和 03，两页都没有，属合同漂移 |
| 23 可观测性 | REJECT | §6.1：D 侧异步远端 KV 等待被计入 queue 区间和 D 的 TTFT，页面归到 prefill，排障会走错方向。§5.2：只讲 NIXL，未给出哪些 connector 覆写 stats 的枚举依据。§3.1、§6.2：以 ERROR/ABORT 结束的请求会进入完成 histogram，产生负 queue 等样本，未说明 |
| 24 插件 | REJECT | §3.1：“五类 ABI”漏 `vllm.logits_processors` 组，与 14 矛盾。§6.2：`VLLM_PLUGINS` 也过滤 platform、IO、stat-logger 插件，页面只写 General 和 Endpoint。§2 最小示例调用的 `collective_rpc("get_scheduler_config")` 在 worker 上不存在，且未写 endpoint 须配对 general 插件的合同 |
| 25 在线更新 | REJECT | §4.2 调用树跳过进程跳（`AsyncMPClient.call_utility_async`），也跳过 DP fan-out（`DPLBAsyncMPClient` gather 后只返回第一个结果），DP 下“finish 返回”的语义未闭合。§6.1 称“Ray 避开 MP 首错问题”只对 `RayDistributedExecutor` 成立，`RayExecutorV2` 继承 MP |
| 26 MultiprocExecutor | REJECT | §1、§8.4：无 `Executor.get_class` 变体枚举依据，漏多节点 follower 与 `RayExecutorV2`。§4.2、§6：`get_response` 遇首个 FAILURE 或超时即抛出，剩余回复留在队列，下一次 RPC 会与陈旧回复配对，破坏本页强调的 FIFO 不变量 |

## 4. 全域 E2E 与连贯性

### 4.1 系统性问题一：默认执行路径口径分裂

02、06、11、12 §1.5 按 `step()` 讲普通路径；07、12 §5.3、16 按默认 async + batch queue 讲。基线默认值是后者（P0 #1）。建议由 06 独占“`async_scheduling` 解析 → `max_concurrent_batches` → `step_fn` / Scheduler 类选择”这条选择轴，并写入默认值、关闭条件和两条路径差异；02、11、12、19 只链接过去，同时修正各自的调用树和时序图。

### 4.2 系统性问题二：互相推给对方的无主功能点

以下每项都是“A 页说归 B，B 页说归 A 或 C，而目标页没有内容”。这类问题机械门禁查不出来，是本轮 E2E 断点的主要来源。

| 功能点 | 推诿链 | 建议 owner |
|---|---|---|
| 滑窗/chunked-local admission cap | 07 ↔ 08 | 08（资源侧）；07 链接 |
| hybrid KV manager 开关、block hash 生成、Mamba align 块生命周期 | 07、12、15 → 08（未写） | 08 |
| spec 下可缓存的尾 token | 08 ↔ 16 | 08 写规则，16 链接 |
| `SchedulerOutput` 生产端字段合同 | 11、12 → 07 §4.1、§8.1（未写） | 07 补“字段 × 生产者 × 条件”表 |
| MoE 专家权重写入 | 09、17、18 均未写 | 09 |
| packed 量化参数加载切片、v1/v2 loader 选择 | 09 ↔ 17 | 17（量化 ABI） |
| attention kernel 消费 KV scale | 10 ↔ 17 | 10 |
| Marlin workspace 与 `g_idx_sort_indices` 地址稳定 | 17 → 自身；25 → 19（未写）；20 另写一遍 | 17，其余链接 |
| MRV1 cudagraph 派发 | 11 → 19（符号不存在） | 19 |
| 投机下的 p 约束 | 16 → 14（14 写反） | 14 修正后自然闭合 |
| grammar + draft | 14 §6.4 与 16 各写且不一致 | 语义归 14，发布路径归 16 |
| IR provider 注册、tolerance | 21 → 20（未写） | 20 |
| prompt logprobs 实现 | 14 → 11/12（只列填写点） | 14 |
| `response_format` 映射、`return_token_ids` | 14、01 → 03（未写） | 03 |
| P/D proxy 路由段 | 22 → 13、03（均未写） | 交 planning 定 owner |
| KV events 发布 | 22 → 23（未写） | 23 |
| 传输观测的 connector 枚举 | 22 → 23（只讲 NIXL） | 23 |
| Elastic EP 编排（`reinitialize_distributed` → `eep_ready` → `commit_prepared_elastic_ep` …） | 18 → 13 → 18；23 → 18 | 交 planning 定 owner（候选 13） |
| Ray、external_launcher 执行与监督 | 23 → 18、13；18 声明不管 Ray；26 只写本地 | 交 planning |
| 配置解析总链（`EngineArgs` → `VllmConfig.__post_init__` → `check_and_update_config`） | 片段分散在 01、07、12、18、19 | 02 增加配置解析树，或交 planning |
| 显存 profiling → KV 预算 → `num_gpu_blocks` | 09 图标注 08，但 08 只有一段 | 08 补公式、算例和 `startup_plan` 变体 |

### 4.3 系统性问题三：无 owner 的阶段和能力

- **启动链两端**：配置解析，以及显存 profiling 到 KV 预算（见上表）；Platform 对象提供哪些能力。
- **请求链**：`n>1` 父子请求与 beam search；tool/reasoning parser（`vllm/tool_parsers` 52 个文件，`vllm/reasoning` 33 个）；Responses、Anthropic、harmony。
- **横切能力**：pooling/embedding 执行（12 自述“本域暂无专页”）、非 CUDA 平台的 worker 与 runner、sleep mode 机制（25 只有 level 语义）、gRPC server 与 Rust frontend、EPD 部署拓扑。
- **用户面**：见 §1“全域缺失”一行。

以上多数需要新增或调整页面合同，属于 `planning-codebase-analysis` 的决定，不宜在页内顺手补。

### 4.4 重复解释（目前一致，但有漂移风险）

- **启动顺序画了 6 遍**：02 §2.2、09 §1.5、10、11 §2.9、12 §2.9、19 §3.4。19 那一版已与 12 冲突。
- **stale output 与 deferred-free fence**：06 §5.1、§6 和 07 §8.3–§8.5 各完整写一遍，各配一图。
- **DP pause 共识讲了 3 遍**：13 §2.4、18 §5.3、25 §5.3。
- **encoder 调度两页详写**：07 §5.3 与 15 §4.1 都详写 `_try_schedule_encoder_inputs`。
- **layerwise reload 讲两遍**：17 与 25。
- **双重认领**：`bind_kv_cache` 由 09 和 10 同时认领。
- **owner 标注不一致**：OutputProcessor 在 12 中写“归 06”，在 14 中写“归 03 与 06”。PassConfig 在 20 中写归 19，其余页写归 21。QK-Norm+RoPE+KV 融合在 20 与 21 中描述不一致。

### 4.5 引用、索引与约定

- **错误跨页引用**：05 → “27”（不存在）；05 → 23，应为 19；11 → 15 §7.4，应为 §7.7；16、12 → 08 的入向 § 引用有漂移。其余 76/78 处跨页 § 引用命中目标章节。
- **index 阅读依赖箭头**：23 方向反了；26 位置偏晚；20 与 21 的依赖倒置；默认 runner 是 MRV2，但 11 排在 12 前面，需要注明。
- **页头基线日期不一**：19、20 写 2026-09-08，21、22、23 写 2026-09-06，其余写 2026-09-07 UTC。commit 为 2026-09-07 00:54 UTC，应统一。24–26 页头字段模板与其余页不同。
- **锚点风格**：01–23 用 `path::symbol`；24–26 用点分 qualname，部分为裸符号，测试锚点较重构前减少。二者都能定位源码，但全域应统一。
- **其他 P2**：14、16 使用裸“V1/V2”，与 MRV1/MRV2 混用；两代页面模板并存；`wiki/courses/**` 没有任何 vLLM 链接；01 文件名 `feature_optimizations_guide` 与标题“使用指南”不符。
- **已核验一致**：页数 26+index=27，与 `wiki/index.md` 和上级 index 一致；22 的 16 个注册 connector、`KVTransferConfig` 13 字段、`ObservabilityConfig` 13 字段与源码一致；`/health` 与 shutdown 升级顺序在 13、23、26 间一致；`defer_block_free` 条件在 06 中正确。

## 5. 机械门禁与内容保留

| 检查 | 结果 |
|---|---|
| `python tools/check_links.py --strict` | 452 页：broken、ambiguous、bare_index、stale_section、orphans 均为 0 |
| `check_math.py --strict <vllm>` | 27 文件，0 错误 0 警告 |
| `check_markdown.py <vllm>` | 27 文件，0 错误 0 警告 |
| `check_assets.py --strict <vllm>` | 27 文件，0 错误 0 警告 |
| `check_locators` | 不触发：vLLM 页无 `path:line` 引用 |
| 保留对账（`dfbd2fa` → HEAD） | 按 slug 对比每页的反引号符号和 wikilink。去掉锚点改写后，最终符号名在全域正文中消失的只有 24（io_processor/LoRA resolver 注册函数、若干测试名）、25（`is_sleeping`、深睡测试）、26（两个 PP 测试名）和 21（`output_attn`）。逐项核对后，概念都仍在正文中，属于源码路线锚点精度下降（P2），不是静默丢失 |

说明：机械门禁全绿，而 P0 与 P1 大量存在。门禁只能保证链接、公式和资源可用，保证不了事实正确和交接完整。2026-09-14 的 21–26 评审给出 PASS 的页面中，本轮仍发现 24 的“五类 ABI”遗漏、25 的 DP fan-out 未闭合和 26 的 FIFO 缺口。

## 6. 修复路由建议

1. **第一波：页内事实修正**，不改页面边界，可并发。
   - 7 条 P0：02、14、17、18、19、20。
   - 纯页内 P1：01 §6.1 与 §2.2、04 §2.1、05 §4 两处引用、15 §4.2、16 §5.1 与 §8.3、17 §4.2、22 §9、23 §6.1 与 §3.1、24 §3.1、§6.2 与 §2、25 §4.2 与 §6.1、26 §4.2、§6 与 §8.4、20 其余 P1。
   - 各页原写作者修改，另派未参与写作者复审。
2. **第二波：默认执行路径统一**。按 §4.1 由 06 独占选择轴，02、11、12、19 同步调用树和时序图。
3. **第三波：推诿链收口**。§4.2 表中已有明确 owner 的项由 owner 页补功能点（目的、I/O、处理逻辑、边界、范围），推出方只改链接。
4. **交 `planning-codebase-analysis`**，涉及页面合同或分类变更：
   - 03 是否改为使用说明，或新增 API/特性使用手册页（§1）；
   - P/D proxy 路由、Elastic EP 编排、Ray/external_launcher、配置解析总链、tool/reasoning parser、pooling 执行、非 CUDA 平台、gRPC/Rust frontend 的 owner；
   - 是否新增课程页串联 vLLM 阅读路径。
5. **收尾**。统一页头日期与锚点风格，调整 index 阅读依赖箭头，13 的并发修改落地后复审。每波修改后跑 T0 四项门禁。

本轮只做静态读码：没有启动 vLLM、GPU、跨机传输或故障注入，也没有修改 wiki 页面、提交或推送。
