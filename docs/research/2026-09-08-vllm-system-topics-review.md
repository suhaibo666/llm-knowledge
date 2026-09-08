# vLLM 第三批系统专题：非作者独立复核

日期：2026-09-08。复核者不编辑被审正文。本文件只记录证据审计与逐页结论；功能树与页面内容仍以 wiki 为权威。

## 范围与证据合同

复核范围为第三批 G/H 的 17、22、23、24、25、26、28、29。采用 `source-faithful-analysis` 的共享 evidence kernel、codebase source pack、mechanism-analysis profile、analysis-focus、page-review-rubric，并依 `drawing-wiki-figures` 检查实际渲染图。

源码只读路径为 `/Users/suhaibo/97-llm/vllm`，已实际核对 `HEAD = 199cb9b964822e59ab9b58d88e7be31eb419a2ae`，`main`，提交时间 `2026-09-06T17:54:32-07:00`。附件旧基线由此前已确立的新基线替代；本轮未移动 checkout。已读第一批报告、03 样稿、旧 02；旧 02 的保留与最终删除仍归协调者。

## 准备审计：组合场景风险与跨页核对项

以下是已打开源码形成的审计输入，**不代表任何未交付页面已通过**。

| 跨页边界 | 源码事实与风险 | 本轮核对项 |
|---|---|---|
| 23 ↔ 14/15/16/20 | `VllmConfig` 的 `enforce_eager` 分支同时将 compile mode 与 cudagraph mode 设为 NONE；adaptive verification 明确拒绝无图、LoRA、PP 组合 | 23 必须接续第一批留下的双重禁用语义，分别解释编译选择、图捕获、运行时 dispatch；不能把排障开关的效果归因于单一优化 |
| 22 ↔ 24 | `BaseRouter._select_experts` 先计算逻辑 expert ID，再记录逻辑 ID，最后应用 EPLB 物理映射；`EplbState.rearrange` 先搬权重后提交 maps | 24 拥有局部路由/分组/专家计算/加权还原；22 拥有 EP 通信与放置、负载统计、搬迁和发布；须用同一逻辑/物理 ID 词义贯通，不能把 replica 当成新的模型专家 |
| 22 ↔ 24/29 | `EplbState.drain_async` 会消费但不应用在途结果；rank readiness 由 all-reduce 协调，worker stop 使用独立 EPLB group | 若正文涉及 async EPLB 与更新/重排，必须区分结果 ready、消费确认、权重应用、map 发布，不能把后台复制完成视为前向已使用新布局 |
| 28 ↔ 13/15/16/23/29 | `OpenAIServingModels.resolve_lora` 在每名称锁内依序 resolver → `add_lora` → 发布前端名字；404 与找到但加载失败的400分开；显式 load/unload 同样只持当前前端锁 | resolver 发现不等于 worker 装载完成；多 API 进程不因此获得分布式原子性。模型/Runner 内部仍归 E，28 给到实际接续入口，不新增 LoRA 重复专题 |
| 28 ↔ 12/29 | `unload_lora_adapter` 只删除前端 `lora_requests`；`_gen_lora_extra_hash_keys` 使用 LoRA 名称；主模型更新 finish 后调用 `reset_lora_state` | 不能称 HTTP unload 已释放所有 worker adapter；同名覆盖不自动等价于 cache 内容版本隔离；主权重更新后的 LoRA 缓存处理须与29一致 |
| 26 ↔ 29 | `AsyncLLM.finish_weight_update` 先 worker collective 完成，再可选设置版本；`EngineCore.set_weight_version` 仅赋值字符串；block hash extra keys 与 NIXL compatibility factors 没有该权重版本字段 | 版本字符串不能写成 KV namespace、内容校验或跨实例同步屏障。生产者和消费者权重一致性及旧 KV 清除由外部更新流程保证，须明确本源码证明范围 |
| 26 ↔ 17/29 | `EngineCore._finish_pause` 同步设备并可调用 `_reset_caches`；后者要求本地与 connector reset；基类 `KVConnectorBase_V1.reset_cache` 默认返回 None | pause 的本地完成不能夸大为任意 connector/外部存储清空或所有实例一致；外部 reset 支持及返回值要核对具体 connector |
| 22 ↔ 17 | `ParallelConfig` 对 EPLB 要求 CUDA/ROCm、开启 EP、TP×PCP×DP > 1 | EP group 不能仅写 TP×DP；前端副本、DP engine、TP/PP/PCP rank 拓扑不得混为一个 world size |

准备审计实际打开的稳定锚点：

- `vllm/config/vllm.py::VllmConfig.__post_init__` 的 eager 分支、`_validate_adaptive_verification`、`_get_dbo_unsupported_features`。
- `vllm/config/parallel.py::ParallelConfig` 的 EPLB 校验；`vllm/model_executor/layers/fused_moe/router/base_router.py::BaseRouter._select_experts`。
- `vllm/distributed/eplb/eplb_state.py::EplbState.rearrange`、`drain_async`、`_all_ranks_result_ready`。
- `vllm/entrypoints/openai/models/serving.py::OpenAIServingModels.load_lora_adapter`、`unload_lora_adapter`、`resolve_lora`。
- `vllm/v1/core/kv_cache_utils.py::_gen_lora_extra_hash_keys`、`generate_block_hash_extra_keys`。
- `vllm/v1/engine/async_llm.py::AsyncLLM.finish_weight_update`、`update_weight_version`；`vllm/v1/worker/gpu_worker.py::Worker._start_weight_update`、`update_weights`、`finish_weight_update`。
- `vllm/v1/engine/core.py::EngineCore._reset_caches`、`_finish_pause`、`pause_scheduler`、`set_weight_version`。
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py::compute_nixl_compatibility_hash`；`vllm/distributed/kv_transfer/kv_connector/v1/base.py::KVConnectorBase_V1.reset_cache`。

## 逐页复核

八篇均在作者交付后单独打开全文、主路径和 3 个关键锚点，并检查实际渲染图；需要修正的页面已复读修正并重新检查受影响图。最终结论为8/8 PASS；本表未替代协调者机械门禁。

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 17_vllm_serving_control_plane_analysis | pass | pass | pass | transform | pass | 3/3 | PASS | 30/20与15/20评分、启动就绪层次和完成计数闭合 |
| 22_vllm_distributed_inference_analysis | pass | pass | pass | transform, coupled-planes | pass | 3/3 | PASS | TP/PCP/DCP数值、PP lifetime及EPLB/微批完成点闭合 |
| 23_vllm_compilation_cudagraph_analysis | pass | pass | pass | transform | pass | 3/3 | PASS | 已补实际ModelCudaGraphManager输出切片边界并复读 |
| 24_vllm_fused_ops_and_kernels_analysis | pass | pass | pass | transform | pass | 3/3 | PASS | norm/quant与MoE数值闭合，workspace和外部provider边界明确 |
| 25_vllm_ir_and_fusion_passes_analysis | pass | pass | pass | transform, coupled-planes | pass | 3/3 | PASS | donation/clone、双输出变换和SP分片合同可重放 |
| 26_vllm_disaggregated_kv_serving_analysis | pass | pass | pass | transform, coupled-planes | pass | 3/3 | PASS | 已修NIXL整组失败与Scheduler单块反例，WRITE本地ACK及store引用闭合 |
| 28_vllm_extension_plugin_system_analysis | pass | pass | pass | transform | pass | 3/3 | PASS | client 建立与插件注入时序已修正并复读 |
| 29_vllm_weight_transfer_online_update_analysis | pass | pass | pass | transform, coupled-planes | pass | 3/3 | PASS | pause/session/version、三数据通路及partial failure闭合 |

### 17：独立复核 PASS

- Beat-2：拒绝只按 round-robin/进程存活分发，解释 stale snapshot 下本地 inflight 下限、KV 排队惩罚及分阶段就绪各自针对的问题；简单轮询不是历史实现断言。
- Hop-walk：已打开 client factory → `DPAsyncMPClient.add_request_async` → `DPLBAsyncMPClient.get_core_engine_for_request` → `_send_input / _send_input_message` → output task 的 subclass hook → `process_engine_outputs`。R 的 engine 映射在发送前建立，finished set 消费才扣 inflight。启动路径另核对 Core handshake yield → input thread response → coordinator READY → ready_event → launcher READY；没有把数据通道与全局屏障画成假串行。
- 正式抽查 3 锚点：`vllm/v1/engine/core_client.py::DPLBAsyncMPClient.get_core_engine_for_request`；`tests/v1/engine/test_engine_core_client.py::test_dplb_kv_pressure_amplifies_waiting_penalty`；`vllm/v1/engine/utils.py::get_engine_process_shutdown_timeout`。评分及 ROCm 双零宽限吻合。另读 `VllmConfig.needs_dp_coordinator`、`MPClient._apply_ready_response`、`wait_for_engine_startup`、multi-API finally 与 supervisor probe/monitor。
- Delete-code：删除 Mermaid 后场景、计算、状态和失败界限仍完整。
- 图触发 `transform`。实际查看 `/tmp/vllm-wave-figs/17-0.png`：R 的同一 waiting/running 输入只改 KV 使用率，能还原 30/20 选择 E1、15/20 选择 E0，再分别更新映射、inflight 和本地 waiting；末节点明确 ADD 非 Scheduler 准入、finished set 才释放。文字无遮挡裁切；这是两次对照而非复制请求，正文已明示。

### 23：首轮独立复核

最终结论：**PASS**。作者已补 §7 的实际 `ModelCudaGraphManager.run_fullgraph`：基类提交 replay，子类返回 capture 持久输出的容量切片，真实有效行由 Runner 后续选择；§11 增补稳定符号，独立复读确认没有把取得引用写成 CPU 数值完成。

- Beat-2：三 token 同步 decode 与 mixed 对照能解释 range、单点、capture 容量的不同职责；独立 compile/graph 轴和 generic/manager 两种 miss 语义明确，成本与事实/推断有分界。
- Hop-walk：已打开 `GPUModelRunner.execute_model` 的 dispatch、FULL/PW/NONE 分支；`CudaGraphManager._init_candidates / dispatch / capture / run_fullgraph / run_pw_graph`；`PiecewiseBackend.compile_all_ranges / _find_range_for_shape / __call__`；`CUDAGraphWrapper.__call__` 与 capture guard。发现返回边界缺少 `ModelCudaGraphManager.run_fullgraph`：基类只提交 replay，子类才返回容量切片的 hidden states 或 PP intermediate tensors。最小修正为 §7 FULL 一句及读码路线补该符号，保持设备提交不等于 CPU 数值可见的限制。
- 正式抽查 3 锚点：`vllm/compilation/piecewise_backend.py::PiecewiseBackend._find_range_for_shape`（single优先、range覆盖）；`vllm/compilation/cuda_graph.py::CUDAGraphWrapper.__call__`（capture guard/地址检查）；`tests/v1/cudagraph/test_cudagraph_manager.py::test_uniform_decode_beyond_capture_ladder_falls_back`（NONE）。另读 mixed decode-only token 回归测试。
- Delete-code 通过。图触发 `transform`，实际查看 `/tmp/vllm23.png`：3×H经范围代码、容量4 FULL和mixed容量4 PIECEWISE三路，能还原padding/边界提交成本，外部PyTorch/CUDA用虚线标出，无遮挡裁切。

### 29：独立复核 PASS

- Beat-2：以单个reload RPC及整模型shadow swap为对照，解释分阶段可见性、按层暂存与原位storage之间的取舍；W=(1,2)→(1,9)具体说明 sparse 网络节省与完整shape暂存成本。
- Hop-walk：已打开 `AsyncLLM.pause_generation` → client utility → `EngineCoreProc.pause_scheduler` 即时/idle callback → `_finish_pause / _reset_caches`；`AsyncLLM.finish_weight_update` → multiprocessing fanout/回复读取 → `Worker.finish_weight_update` → backend finish → 可选 `set_weight_version`。另走 sparse receive → checkpoint patches → NaN final copy，dense/IPC start→receive→layerwise restore，以及RDT update→queues/streams/free-group drain→finalize；DP pause consensus与resume barrier也已核对。
- 正式抽查 3 锚点：`tests/v1/engine/test_engine_core.py::test_pause_synchronizes_device_before_cache_reset`；`vllm/model_executor/model_loader/checkpoint_weight_patch.py::_load_nan_masked_weights`；`vllm/distributed/weight_transfer/sharded_rdt_engine.py::ShardedRDTWeightTransferEngine.drain_pending`。分别支持idle-before-reset、NaN保留旧值且finally只恢复copy方法、队列/stream/producer free信号完成。
- Delete-code通过；`transform, coupled-planes` 触发。实际查看 `/tmp/vllm-wave-figs/29-0.png`、`29-1.png`：R保留历史但重算KV、两个worker扇出、成功标签后置与部分失败停住清楚；三数据通路重放同一目标值，图2明确sparse完整shape暂存和RDT finish fence。无文字裁切遮挡，序列图较长但没有比例时间误导。
- 组合场景：外部reset None/False、Core忽略布尔返回、版本不是KV namespace、RDT拒绝EPLB、主/draft LoRA尾处理差异均明确，未用通用finish夸大为全局原子换版。

### 28：首轮独立复核

最终结论：**PASS**。作者已将 §2、§3.4 更正为 Phase A 尚未向插件注入 EngineClient，并补入 `build_and_serve` 源码路线；独立复读确认两处及图文一致。以下保留首轮问题与实际审计记录。

- Beat-2：以 import-time 自动注册和单阶段 `register()` 为具体替代，说明选择必须先于副作用、注入必须满足 app 状态依赖；LoRA 有序候选以 engine 接受为提交条件。
- Hop-walk：实际打开 `build_and_serve → build_app → attach_endpoint_plugins → load_endpoint_plugins → load_plugins_by_group → DummyAdminEndpointPlugin.attach_router`，以及 `init_app_state / init_render_app_state → init_endpoint_plugins_state → init_state → scheduler_config handler → collective_rpc`。发现 §2、§3.4 把“尚未向插件注入 client”写成“client 尚未建立/不可用”：`build_and_serve` 在 `build_app` 前已经调用 `engine_client.get_supported_tasks()`。最小修正是更正这两处时序并补入口锚点，无需重做图。
- LoRA 路径另读 `BaseServing._check_model → OpenAIServingModels.resolve_lora → add_lora → lora_requests`，支持按需开关、名称锁、400/404、resolver 自身异常不在 add_lora try 内的区分。协调者复抽发现原路线使用旧类名，已更正为该文件实际 `BaseServing`，独立复读类定义确认。unload、同名热换版属于组合场景后续登记，不因本页范围未展开而拒绝。
- Delete-code：正文没有靠代码块承担解释，去图后仍可理解发现/选择/初始化及失败域。
- 正式抽查 3 个锚点：`vllm/plugins/__init__.py::load_general_plugins`（guard 在导入/回调前）；`vllm/entrypoints/openai/models/serving.py::OpenAIServingModels.resolve_lora`（等待后发布）；`tests/plugins_tests/test_endpoint_plugins.py::test_endpoint_plugin_end_to_end`（cfg-a/cfg-b 与 200）。三者均吻合。
- 图触发为有序选择/状态变换 `transform`。已实际查看 `/tmp/vllm-wave-figs/28-0.png`、`28-1.png`：从同一配置查询还原 API 200 与 render 503；从 test-lora 还原查表、候选尝试、等待接受、发布及400/404。两图文字无裁切遮挡，陌生读者可区分 route 存在与 client 注入，以及发现 adapter 与接受 adapter。图 2 的异常捕获范围由紧邻注释明确约束。

### 24：独立复核 PASS

- Beat-2：以未融合 residual/norm/quant 三阶段及逐expert小GEMM为具体替代，说明融合究竟消除哪个中间张量、保留哪些读写与调度成本；没有把fusion当成无条件更快或不落显存。
- Hop-walk：实际打开 `RMSNorm.forward_cuda / forward_native` → IR `maybe_inplace` → `IrOpInplaceOverload._inner_call / IrOp.dispatch` → `vllm_c.fused_add_rms_norm` → C kernel 的residual写回、归约、重读与输出；另读normquant的RMS、动态scale和最终quant三个阶段。MoE路径打开modular `apply / _prepare` → NoDPEP prepare → Triton assignment、两次GEMM、activation/quant → `moe_sum` → NoDPEP finalize与`TopKWeightAndReduceNoOP`，确认不重复乘路由权重。
- 正式抽查3锚点：`csrc/libtorch_stable/layernorm_kernels.cu::fused_add_rms_norm_kernel`（通用路径仍写读residual）；`vllm/model_executor/layers/fused_moe/experts/triton_moe.py::TritonExperts.apply`（cache1/3共享workspace2，cache2在workspace13，GEMM2后sum）；`tests/kernels/moe/test_moe_align_block_size.py::_verify_expert_level_sorting`（同expert内部slot顺序可变）。另读 `TestFusedAddRMSNorm.test_native_semantics`、`test_moe_sum`、`float_to_int8_rn / ScaledQuant`、`FusedMoEExperts.is_supported_config`和`swiglu_limit`字段注释。
- Delete-code通过。图触发`transform`；实际查看 `/tmp/vllm-wave-figs/24-0.png`、`24-1.png`。第一图由x/r重放u=(2,2,2,2)、y=(1,2,1,2)、scale=2/127和q=(64,127,64,127)，将少掉的y写读限定为逻辑流量。第二图由A/B四个有效slot及sentinel4重放两次专家GEMM、SiLU门控和1.25cA/1.75cB，monolithic外部依赖虚线边界清晰；tile4是教学布局而非硬件支持认证。文字无裁切遮挡。
- `swiglu_limit`配置注释声称通用filter会拒不支持clamp的backend，而当前filter未读字段，正文明确保留此冲突，未替源码保证任意provider数值等价。

### 22：独立复核 PASS

- Beat-2：同样增加两卡分别解决容量、吞吐与长上下文，TP求和/拼接、PP顺序、DP副本、PCP query与DCP KV切分均给出具体恢复合同。DCP平均6的反例和只换map误执行E2的反例说明不变量为何必要。
- Hop-walk：实际打开 `EngineCore.step` → `MultiprocExecutor.execute_model / collective_rpc` 广播与响应future → `Worker.execute_model` 的上轮send等待、irecv包装、Runner调用和下游isend → `AsyncIntermediateTensors.wait_for_comm` 首次tensors访问等待；回到Engine的future、可选sampling、abort消费和scheduler update。输出rank在TP2/PP2/PCP1确为2，aggregator改变收集范围，文中未把future包装成全局barrier。
- 正式抽查3锚点：`vllm/model_executor/layers/linear.py::RowParallelLinear.forward`（仅rank0 bias、allreduce）；`vllm/v1/worker/gpu/pcp_manager.py::PCPManager._build_batch_layout`（restore索引/最大rank padding/write mask）；`vllm/distributed/eplb/eplb_state.py::_move_to_workspace`（move_from_buffer先于per-layer map commit及消费事件）。另读 `_iter_rank_chunks / _reorder_segments / restore_hidden_states`、DCP AG/RS与A2A函数及数学测试、`initialize_model_parallel`各轴reshape、async worker同步与pending发布、MRV1/2微批DP同步、`UBatchRunner.run`及DeepEP dispatch前yield。
- Delete-code通过。`transform, coupled-planes`触发；实际查看 `/tmp/vllm-wave-figs/22-0.png` 至 `22-4.png` 五图，分别重放55+250=305、ABG_padding_CDEF按0/1/4/5/6/7/2恢复、两个head各得8、E1迁移到槽2仍保持逻辑身份、A/B事件交接后顺序归并。五图文字无遮挡裁切，DBO图明确为依赖次序而非测量时间比例。
- 组合核验：PCP例子明确8个已算历史，避免源注释七token示例忽略fresh segment排序；MRV2全padding末微批仍参加collective且graph NONE，MRV1会否决；async EPLB逐层发布且不保证全模型瞬时原子切图；RDT拒EPLB与29一致。

### 25：独立复核 PASS

- Beat-2：以直接绑定opaque kernel和从低层ATen猜回语义为具体替代，解释延迟provider的作用；从相同返回值、不同caller输入状态说明donation为什么独立于数学等价。分析推断与源码事实分开。
- Hop-walk：实际打开 `RMSNorm.forward_native` / `IrOpInplaceOverload`、backend `configure_post_pass` → pre-grad `VllmIRInplaceFunctionalizationPass` → manager已配置pass循环与cleanup → lowering fake-arg dispatch → `IrOpImpl.func_impl_fn`保护clone → `UnsafeCloneEliminationPass` → cleanup → `FixFunctionalizationPass`；终点是可交回Inductor的lowered图，外部编译运行由23接续，没有假称本轮GPU执行。
- 正式抽查3锚点：`vllm/compilation/passes/ir/inplace_functionalization.py::VllmIRInplaceFunctionalizationPass.__call__`（later-user硬拒绝与placeholder donation）；`tests/compile/passes/ir/test_clone_cleanup.py::TestCloneCleanupWithDonatedInputs.test_donated_input_clone_removed`（clone删除及返回数值）；`vllm/compilation/passes/fusion/sequence_parallelism.py::FirstAllReduceRMSNormPattern.register`（norm全量恢复、residual留分片）。另读clone layout helper/unknown HOP、AddRMSNorm及reshape pattern、SP range guard、lowering UUID、finalfix allowlist与no-DCE注释。
- Delete-code通过。`transform, coupled-planes`触发；实际查看 `/tmp/vllm-wave-figs/25-0.png`、`25-1.png`、`25-2.png`：A/B storage与值状态、(y,u)两输出的(2,4)/(1,2,4)形状、TP四行norm恢复而residual每rank两行都可由图重放。编译替换粗箭头与运行数据流可区分；图1包含布局反例，文字无遮挡裁切。
- 已核对三项未解决边界：SplitCoalescing只比input/sizes而不比dim、测试只用dim=-1；Oink支持谓词拒weight=None，若进入supported providers测试参数集则与统一provider断言冲突；BF16+DeepGEMM UE8M0 packed scale的测试skip是未覆盖风险，不是runtime fallback。正文均未把这些边界包装成一般alias或数值证明。

### 26：独立复核 PASS

初审发现两处必须澄清的接缝，作者已修并重新渲染，独立复读通过：图1的invalid41→computed4是通用Scheduler仅收到单个invalid ID的恢复例子；本例非HMA且无本地命中时，NIXL整笔handle失败会把40/41/42全部标失效，computed应到0。MoRIIO WRITE的完成ACK由P在等待statuses后本地生成，不需要D额外返回ACK；旧图“ACK到P”已改成明确的本地生成及发write_done给D。

- Beat-2：以十二token全命中仍需重算末token、按FIFO配对、固定timeout与按request/future-consumer持有源的具体反例，说明身份、可计算性、lease与per-job引用的不同职责；成本模型明确未测量。
- Hop-walk：实际打开Scheduler命中、partial-tail选择、async allocation/remote waiting → `ActiveKVConnector.pre_forward / post_forward / no_forward` → attention layer wait/save hooks → NIXL READ提交/handle轮询/get_finished设备后处理 → `KVOutputAggregator.aggregate` → Scheduler invalid处理/finished集合/下步晋升。另开push writer两到达顺序会合与D通知收齐、MoRIIO实际入队seal/statuses/本地ACK/deferred-free过滤、Mooncake pin/save finally/completed-worker-count释放；均闭合到consumer可计算或producer本地引用归还。
- 正式抽查3锚点：`tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py::test_full_block_prompt`（全命中后只重算末token）；`vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py::_finalize_if_complete`（先等statuses再write_done及本地ACK）；`tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py::test_store_job_blocks_are_released_once_every_rank_reports`（跨worker完成前保持ref）。另读NIXL `_handle_failed_transfer`、`test_async_load_failure`、MoRIIO seal计数测试、Mooncake reset注释与worker finally。
- Delete-code通过。`transform, coupled-planes`触发；实际查看 `/tmp/vllm-wave-figs/26-0.png` 至 `26-3.png`，修正后再次打开0和2。四图可分别重放12→11/4/0的不同证据输入、NIXL push双前提、registered3/queued2的seal与生命周期双门、Mooncake ref1→2→1→0与remaining2→1→0；文字无遮挡裁切。
- 组合边界已核验：普通remote_prefill不受反向复用threshold限制；零字节仍需源释放通知；failed finished_recving不是成功，HMA invalid-block填充仍TODO；协议hash无runtime weight_version；reset None/False与Core不消费bool不能证明任意外部store已清空。正文没有把这些边界扩大成全局一致性保证。

## 执行限制

准备审计为冻结源码静态阅读；未运行 GPU、NCCL、NIXL、LoRA 装载或在线权重更新，不将源码/测试合同称为已测性能或生产能力认证。
