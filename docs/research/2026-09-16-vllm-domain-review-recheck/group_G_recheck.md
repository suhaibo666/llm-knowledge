# Group G 复审（13 / 18 / 25 / 26）——针对写作者已应用的修复

- 审阅者：独立 reviewer（未参与写作），只报告，不改页面。
- 冻结基线：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`（复审前后 `git rev-parse HEAD` 一致，checkout 未移动；工作区仅有未跟踪的 `artifacts/`）。
- 首轮报告：`docs/research/2026-09-15-vllm-domain-review/group_G_13_18_25_26.md`。
- 本轮方法：对每条旧发现回到源码复核（不看新 prose 的自述）；对新增文本逐句核验；重放新增数值例子；重跑 T0 门禁。
- T0 门禁（`.venv` python 3.13）：`check_links --strict` pages=452，broken/ambiguous/bare_index/stale_section/orphans 全 0；`check_math --changed --strict`、`check_markdown --changed --strict`、`check_assets --changed --strict` 各 39 文件，0 error 0 warning。新增的 `\operatorname*{arg\,min}` 公式块与四张新 Mermaid 图均通过。

---

## 1. 判定行

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 13_vllm_serving_control_plane_analysis | pass | pass | pass | timing, transform, topology（§1.4/§2.2/§3.2/§4.2 四张新图各有触发理由与 spec） | pass（评分公式、30 vs 20、C=2 burst 表、KV 惩罚等价列、wave 7→8、退出预算全部复算通过） | 12/12 | PASS | feature: pass §1.4/§2.2/§3.2/§3.3/§4.2。新增算法模型、逐条协议表、Coordinator 合成表全部与源码一致；仅剩 N5/N7 两处 P2 与旧 P2-1/2/3 未动 |
| 18_vllm_distributed_inference_analysis | pass | pass（§5.1–§5.2 已修复） | pass | layout, transform, timing, coupled-planes | pass | 10/10 | PASS | feature: pass §5.1/§5.2/§7.1/§11。P0-1、P1-1 已按源码改正；但 7 条 P2（符号归属/类归属/默认值）全部未动 |
| 25_vllm_weight_transfer_online_update_analysis | pass | pass（§4.2 已补进程跳与扇出） | pass | transform, timing | pass | 9/10 | PASS | feature: pass §4.2/§6.1。P1-2、P1-3 已修；**新增一条 P2 锚点错误 N1**（`DPEngineCoreProc.pause_scheduler` 不存在），旧 6 条 P2 未动 |
| 26_vllm_multiproc_executor_rpc_deepdive（机制 profile） | pass | pass | pass | timing, state | pass（1120 MiB、rank=2、FIFO drain 复算仍成立） | 10/10 | PASS | variant/feature: pass §1.1/§4.2/§6/§8.4。变体枚举依据与 FIFO 破坏边界均已补齐；N2/N3/N4 为新增文本的精度缺口 |

四页的 P0/P1 已全部关闭；剩余为 P2 与**域级 E2E-1/E2E-2 归属缺口（仍是 P1 级 contract drift，未修）**。

---

## 2. 旧发现逐条状态（均以冻结源码复核，非以新 prose 自述）

### P0

| 编号 | 状态 | 源码复核 |
|---|---|---|
| P0-1｜18 §5.1“MRV2 拒绝 PP” | **FIXED** | §5.1 现写“排除 MRV2 的原因是 MRV2 在 TP>1 时不支持 sequence parallelism，不是 MRV2 不支持 PP”。核对 `vllm/config/vllm.py::VllmConfig._get_v2_model_runner_unsupported_features`：`enable_sp and TP>1` → "sequence parallelism"；V2 只另拒 "pipeline parallelism with external_launcher"。`vllm/v1/worker/gpu/model_runner.py::GPUModelRunner.__init__` 的 `self.use_pp = pipeline_parallel_size > 1` 与 `self.pp_handler = PPHandler(...)` 确认 MRV2 支持 PP。§7.1 contradiction 框也改为“`_get_dbo_unsupported_features` 拒绝的是 DBO 与 … 的组合”，与源码逐条对应（LoRA/spec/PP/CP/多模态或 encoder-decoder/hybrid/CUDA graph/mm-encoder-only，且 `VLLM_USE_V2_MODEL_RUNNER is None` 时整项拒绝）。 |

### P1

| 编号 | 状态 | 源码复核 |
|---|---|---|
| P1-1｜18 PP 反向通道的 runner 轴 | **FIXED** | §1.2⑥/§1.3⑥/§5.2/§10⑥ 均已按代际分开。核对：MRV1 `gpu_model_runner.py::sample_tokens` 在 `use_async_scheduling` 下、`not broadcast_pp_output and pp.world_size>1 and pp.is_last_rank` 调 `_pp_broadcast_prev_sampled_token_ids`（`group=pp.device_group`）；非末 rank 在 `execute_model_state is None` 分支调 `_pp_receive_prev_sampled_token_ids_to_input_batch`，保存 `_pp_recv_work` 并 `output_token_ids.append(-1)`；`_prepare_input_ids` 开头 `self._pp_recv_work.wait()`；两端都在 `_is_all_reqs_chunked_prefill()` 时跳过。`broadcast_pp_output = (backend=="external_launcher" and len(pp.ranks)>1)`，其分支用 `broadcast_tensor_dict(model_output_broadcast_data, src=last)` 广播 logits ✓。§11 调用树也已把 PPHandler 挂到 `MultiprocExecutor.sample_tokens → Worker.sample_tokens → gpu.model_runner…`（`Worker.sample_tokens` 与 `MultiprocExecutor.sample_tokens(unique_reply_rank=output_rank)` 均存在）。新引测试 `tests/v1/worker/test_gpu_model_runner.py::test_sample_tokens_receives_pp_sampled_ids_only_on_non_last_rank` 存在 ✓。 |
| P1-2｜25 §6.1 Ray 结论漏 RayExecutorV2 | **FIXED** | §6.1 现按 `Executor.get_class` 分两代。核对 `vllm/v1/executor/ray_executor_v2.py`：`class RayExecutorV2(MultiprocExecutor)`、`RayWorkerProc(WorkerProc)`，文件内**没有** `def collective_rpc / execute_model / sample_tokens` 覆写，复用 `rpc_broadcast_mq` / `response_mqs` / `FutureWrapper` ✓。恢复建议也已补“除非能证明所有 response queues 已 drain，否则重建 Engine/executor”，并链到 26 §6。 |
| P1-3｜25 §4.2 调用树跳过进程边界与 DP 扇出 | **FIXED** | 四棵树都加了 `AsyncMPClient.call_utility_async` / `[ZMQ 进程边界] EngineCoreProc._handle_client_request(UTILITY)` / `_invoke_utility_method`，并标注 `[DPLB internal/hybrid] gather 所有 EngineCore`。核对 `core_client.py`：`collective_rpc_async → call_utility_async("collective_rpc", method, timeout, args, kwargs)`；`pause_scheduler_async → call_utility_async("pause_scheduler", mode, clear_cache)`；`set_weight_version_async → call_utility_async("set_weight_version", …)`；`DPLBAsyncMPClient.call_utility_async` 对 `core_engines` 全量 `asyncio.gather(...)[0]`；`DPAsyncMPClient` 走继承版本只发 `self.core_engine` ✓。新增段落“任一 EngineCore 抛错都会让公开调用失败……只是 API 丢弃其余返回值”与代码一致。 |
| P1-4｜26 §1/§8.4 变体基础 | **FIXED** | 新增 §1.1 表由 `Executor.get_class` 六分支 + `ParallelConfig.__post_init__` 默认解析给出依据，逐行核对通过：CUDA `nnodes>1` → mp；单机 `device_count() < world_size` → 直接 `ValueError`（提示 ray 或 `--nnodes`），不自动切 Ray；`data_parallel_backend=="ray"` 或 placement group 才默认 ray；`world_size_across_dp<=1 且 world_size==1` → uni。leader/follower 也核对无误：`node_rank_within_dp==0` 才建 `rpc_broadcast_mq` 与 `response_mqs`（远端项取 `workers[0].peer_worker_response_mqs[rank]`），`collective_rpc` 首行 `assert self.rpc_broadcast_mq is not None, "collective_rpc should not be called on follower node"`；`create_mq_broadcaster` / `create_single_reader_mq_broadcasters` 都是 `1<<22`、6 槽 ✓。§8.4 收束句已限定到 leader + RayExecutorV2。 |
| P1-5｜26 §4.2/§6 首错破坏 FIFO 配对 | **FIXED** | §4.2 加了因果链，§6 加了带“分析推断”标签的破坏场景与处置建议。核对 `MultiprocExecutor.collective_rpc::get_response`：逐 mq `dequeue(timeout=剩余)`，`TimeoutError` 直接 raise，`status != SUCCESS` 直接 `RuntimeError`，其后队列不 drain；响应帧是 `(status, result)`，**无 call ID**；异常路径没有任何 resync/drain 协议 ✓。 |
| E2E-6｜25 DP 权重更新扇出 | **FIXED** | 由 P1-3 覆盖，并链接 13 §3.1–3.2 的 client 变体。 |

### P2（13）

| 编号 | 状态 | 说明 |
|---|---|---|
| P2-1 §4.1 漏 `--grpc` / `serve_grpc` sibling 入口 | **NOT_FIXED** | 全页 `grep -n grpc` 无命中。 |
| P2-2 §4.5 → 26 §8.2 退出链正向链接 | **NOT_FIXED** | 13 全页（含 Related Pages）无 `26_vllm_multiproc` 链接。 |
| P2-3 Ray DP backend（`CoreEngineActorManager` / `DPMoEEngineCoreActor`）故障与退出无主 | **NOT_FIXED** | 13 仅在 §4.2 地址分支提到 Ray DP；两个符号在全域 26 页中均无命中。 |

### P2（18）

| 编号 | 状态 | 说明 |
|---|---|---|
| P2-4 `ensure_model_parallel_initialized` 四条 assert 只校验尺寸 | **NOT_FIXED** | §1.2②/§1.3②/§10 仍写“重复初始化被四条 assert 拦截”。 |
| P2-5 `EngineCore.reinitialize_distributed / commit_prepared_elastic_ep` 应为 `DPEngineCoreProc.*` | **NOT_FIXED** | 18 第 90、302、304、586 行仍写 `EngineCore.*`；23 页写 `DPEngineCoreProc.*`（正确）。源码：两方法只定义在 `DPEngineCoreProc`（`core.py:2302`）。同时构成 E2E-5。 |
| P2-6 §4.3 四张状态机只讲 existing scale-up 一条 | **NOT_FIXED** | 源码复核仍成立：`reinitialize_distributed` 只产生 `worker_type="removing" if is_shutdown else "existing"`；`worker_type="new"` 来自 `_eep_scale_up_before_kv_init`。 |
| P2-7 §5.1“唯一设置 `all_gather_tensors` 的调用点” | **FIXED** | 现同时点出 `Worker.execute_model`（`assert not self.use_v2_model_runner`）与 MRV1 `GPUModelRunner.execute_model` 的 `broadcast_pp_output` rare-case 分支；后者确实构造同名 `{"residual": not is_residual_scattered_for_sp(...)}` ✓。 |
| P2-8 §9 表 `data_parallel_index` 默认值 0 | **NOT_FIXED** | 第 473 行仍写默认 0；实际 `Field(init=False)`，由 `__post_init__` 置为 `data_parallel_rank` 并在 `run_engine_core` 覆写。 |
| P2-9 EPLB 模块级函数被写成 `EplbState` 方法 | **NOT_FIXED** | 第 592 行路由仍把 `compute_logical_maps / _commit_eplb_maps / _commit_eplb_maps_for_layer / _move_to_workspace` 挂在 `EplbState` 下。 |
| P2-10 §1.3③ `get_response_mqs` 的 assert 以 output rank 为界 | **NOT_FIXED** | 第 118 行原样保留；该 assert 以 `world_size` 为界，且 `collective_rpc` 不经过 `get_response_mqs`。 |
| P2-11 `dist.barrier(dp_group)` 是 test-only utility | **NOT_FIXED** | 第 84、122 行仍把它写成可观察完成点，未注明 docstring 的 test-only 限定。 |

### P2（25）

| 编号 | 状态 | 说明 |
|---|---|---|
| P2-12 `finish_weight_update(version)` 参数名 | **NOT_FIXED** | 第 423 行仍写 `version`；`LLM` / `AsyncLLM` 均为 `weight_version`。 |
| P2-13 §4.2“绑定 DP、local world size、Worker rank” | **NOT_FIXED** | 紧随新树的这句未改；源码 `gpu_worker.py:1412` 用 `parallel_config.data_parallel_rank * parallel_config.world_size + rank`（world_size = PP×TP×PCP，不是 local world size）。dense DP 索引重合的“未核实怀疑”仍未处理。 |
| P2-14 入口 sibling 轴（RLHF dev `api_router`、`clients.py` 的 HTTP/Ray 同步客户端） | **NOT_FIXED** | 页内 `api_router` / `HTTPVLLMWeightSyncClient` / `RayVLLMWeightSyncClient` 均无命中。 |
| P2-15 §3.3 图只有 sparse lane 带例子身份 | **NOT_FIXED** | 该图未改。 |
| P2-16 §4.4 路由漏 `sharded_rdt_trainer.py` | **NOT_FIXED** | 页内 `sharded_rdt_trainer` 无命中（只在树里出现类名）。 |
| P2-17 §5.3 DP pause 第三次讲解 | **NOT_FIXED** | §5.3 未改；仍与 13 §2.4、18 §5.3 三处并列（见 E2E-4）。 |

### P2（26）

| 编号 | 状态 | 说明 |
|---|---|---|
| P2-18 §5.1 `/dev/shm` 预留漏 TP/DCP `GroupCoordinator.mq_broadcaster` 与多节点 4 MiB×6 | **PARTIAL** | §1.1 已点明远端 group 的 `1<<22`×6 合同，但 §5.1 表与 `160 + 4×240 = 1120 MiB` 仍无任何限定，TP/DCP 组的 broadcaster（world>1 时每组 ~24 MiB）仍未计。另见新发现 N2。 |
| P2-19 §8.3“源码注释限定 TP>1”与代码冲突 | **NOT_FIXED** | §8.3 原样保留；`MultiprocExecutor.execute_model / sample_tokens` 无条件传 `timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`。 |
| P2-20 §3.1“父进程先创建广播 MQ”只在 leader 成立；“响应线程”只在 async scheduling 存在 | **PARTIAL** | §1.1 已补 leader/follower，但第 82 行仍无限定；第 178 行“响应线程”仍未说明只在 `scheduler_config.async_scheduling` 时存在（同步路径在 busy loop 线程内 `get_output`）。 |
| P2-21 术语“Executor 父进程”未点明即 EngineCore 进程 | **NOT_FIXED** | 全页仍未把父进程与 `EngineCoreProc` 对齐。 |

### E2E / 跨页

| 编号 | 状态 | 说明 |
|---|---|---|
| E2E-1 Elastic EP 编排无主（P1） | **NOT_FIXED** | `_eep_wait_for_setup_switch_complete`、`eep_ready` 轮询、`commit_prepared_elastic_ep` 的客户端序列、HTTP `vllm/entrypoints/serve/elastic_ep/api_router.py` 在全域 26 页仍无任何命中（`eep_ready` 只出现在 18 的 engine 侧状态机）。13 §4.6 仍只讲成员表接缝并把“实际重配”推给 18。 |
| E2E-2 Ray / external_launcher 监督拓扑无主（P1） | **PARTIAL** | 26 §1.1/§8.4 已承接 `RayExecutorV2`（同一控制面）并显式声明 Ray V1 / UniProc / external launcher 不属本页；25 §6.1 补齐两代 Ray 的收集语义。但 23 第 347 行仍写“Ray 和外部 launcher 的监督拓扑见 18 与 13”，而 18 §5 仍不展开 Ray 图，13 只有地址分支；Ray DP engine actor 生命周期仍无主。悬挂指针未消。 |
| E2E-3 13 §4.5 → 26 §8.2 正向链接 | **NOT_FIXED** | 同 P2-2。 |
| E2E-4 DP pause 共识讲三遍 | **NOT_FIXED** | 13 §2.4（owner）、18 §5.3、25 §5.3 仍各讲一遍，内容彼此一致、无矛盾，但仍是三处。 |
| E2E-5 符号归属跨页不一致 | **NOT_FIXED** | 同 P2-5。 |
| E2E-6 DP 权重更新扇出 | **FIXED** | 见 P1-3。 |

---

## 3. 新发现（回归与新增文本中的错误）

新增文本整体质量高：13 的评分公式、协议表、Coordinator 合成逻辑，18 的两代 PP 通道，25 的 utility 跳与 Ray 两代，26 的执行器选择表，逐句核对全部与冻结源码一致；所有新引测试名（11 个）在指定文件中都存在。以下是新增文本引入的缺陷，均为 P2，无新 P0/P1。

**N1（P2，回归）｜25 §4.4 路由表新写 `vllm/v1/engine/core.py::DPEngineCoreProc.pause_scheduler`**
- 页面原文：`| DP pause 共识 | vllm/v1/engine/core.py::DPEngineCoreProc.pause_scheduler |`（旧行只写 `DPEngineCoreProc`，本次编辑把它“具体化”成了不存在的符号）。
- 源码证据：`core.py` 中 `def pause_scheduler` 只定义在 `EngineCore`(860) 与 `EngineCoreProc`(1937)；`DPEngineCoreProc`(2015) 只定义 `_pause_complete`(2088) 与 `resume_scheduler`(2122)，DP 共识实际在 `_pause_complete` / `_has_global_unfinished_reqs` + `ParallelConfig.sync_dp_state`。
- 影响：与旧 P2-5（18 页把 `DPEngineCoreProc.*` 写成 `EngineCore.*`）同类的类归属错误，方向相反；`check_locators` 的 `unresolved` 类缺陷。
- 修正：写成 `DPEngineCoreProc._pause_complete / _has_global_unfinished_reqs / resume_scheduler`（pause 入口则标 `EngineCoreProc.pause_scheduler`）。

**N2（P2）｜26 §1.1 把多节点 MQ 容量差异归因于“远端 reader”，实际按部署分支**
- 页面原文：“本页 §5.1 的 16/24 MiB SHM ring 容量和‘本机最慢 reader’背压仅适用于 local reader……远端 group 的 broadcaster 参数是单条 `1 << 22` bytes、6 个槽”。
- 源码证据：`WorkerProc._init_message_queues` 的分支键是 `vllm_config.parallel_config.nnodes_within_dp == 1`，不是 reader 是否远端。多节点 DP 下**所有** Worker（含 leader 同机 Worker）都走 `create_mq_broadcaster(...)` 与 `create_single_reader_mq_broadcasters(reader_rank_in_group=0)`，两者固定 `1<<22`、6 槽；`MessageQueue.create_from_process_group_single_reader` 对 `same_node` 只是把 `n_local_reader` 设为 1（仍是 SHM），槽宽仍为 4 MiB×6。
- 影响：读者会以为多节点部署里同机 Worker 仍是 24 MiB×10，于是 §5.1 的 `160 + 4×240 = 1120 MiB` 会被错误套用到多节点拓扑。
- 修正：把限定词从“local reader”改成“`nnodes_within_dp == 1` 的单节点 DP 组”；在 §5.1 的容量表加一行多节点分支（输入与响应各 4 MiB×6），并声明 1120 MiB 只对单节点四 Worker 成立。

**N3（P2）｜26 §1.1 uni 行漏掉 TPU/XLA SPMD 默认分支**
- 源码证据：`ParallelConfig.__post_init__` 的默认解析第一条是 `if current_platform.is_tpu() and envs.VLLM_XLA_USE_SPMD: backend = "uni"`，先于 CUDA 各分支。
- 修正：uni 行补“TPU 且 `VLLM_XLA_USE_SPMD` 时的默认”，使变体依据覆盖非 CUDA 平台。

**N4（P2）｜26 §1.1 与 25 §6.1 未说明 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND` 在本基线默认开启**
- 源码证据：`vllm/envs.py` 的 getter 是 `bool(int(os.getenv("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "1")))`——默认 **1**（第 65 行的 `bool = False` 只是类型标注，非运行时默认）。
- 影响：两页都把 `RayDistributedExecutor` 排在前、`RayExecutorV2` 写成“该开关开启时”，读者会把 MQ 控制面的首错/超时未 drain 边界当成 opt-in；实际上 `--distributed-executor-backend ray` 默认就落在 `RayExecutorV2`。
- 修正：在 26 §1.1 的两行 Ray 条目与 25 §6.1 明确“本基线默认 = V2”，并提示注解与 getter 默认值不一致。

**N5（P2）｜13 §3.2“表中的 `target_identity` 是 API 的 `EngineIdentity`”**
- 源码证据：载荷来自 `add_request_async` 的 `msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))`，`chosen_engine` 是**目标 Engine** 的 ZMQ identity（`rank.to_bytes(2,"little")`）；`EngineIdentity = bytes` 是 Engine 身份类型，不标识 API。
- 影响：与紧接其后的 §2.3 contradiction（bytes 与 `self.engine_index` 比较）联读时容易误解成“API 身份被拿去和 engine_index 比”。
- 修正：改为“API 侧持有的**目标 Engine** `EngineIdentity`（bytes）”。

**N6（P2）｜25 §4.2 pause 树缺 `_invoke_utility_method` 一跳，与其余三棵树不一致**
- 源码证据：`EngineCoreProc._handle_client_request` 的 UTILITY 分支统一走 `_invoke_utility_method(method_name, get_result, output, enqueue_output)`，pause 不例外（Future 结果还会挂 done callback 后再入队）。
- 修正：pause 树补同一跳，或四棵树统一省略并在正文说明。

**N7（P2，provenance）｜13 §2.2 引用站外 Envoy 1.25.7 文档作为 P2C 定义来源**
- 该链接不在 `raw/`，本轮离线无法核验版本与该页内容；页面其余推断都自带“分析推断/测试合同”标签，这一条是唯一依赖未核验外部来源的定义性引用。
- 修正：或把 P2C 定义降为通用术语描述（不挂具体版本 URL），或把该文档落到 `raw/` 作为可核验来源。

**另一处措辞（P2-lite）｜18 §5.1“普通 MP 路径由 `Worker.execute_model` …”**：`Worker.execute_model` 同样服务 ray/uni 后端，真正的对照轴是“非 external_launcher”。建议把“普通 MP 路径”改为“非 external_launcher 的 Worker 路径”。

---

## 4. 本轮重放记录（新增例子）

- DPLB 评分：`max(client_count*inflight, waiting+running) + waiting*6.0*max(0, kv-0.5)`（`if waiting` 守卫）、严格 `<`、`eng_start_index = len*client_index//client_count`、选中后 `current_counts[eng_index][0] += client_count`、起点 `(eng_start_index+1) % num_engines`——页面“旧起点加一，不是选中位置加一”与源码一致 ✓。
- 30 vs 20 例子：E0 = max(0,15)+5·6·0.5 = 30，E1 = max(0,20)+0 = 20 → 选 E1；把 E0 KV 改 0.2 → 15 < 20 改选 E0 ✓，与 `test_dplb_kv_pressure_amplifies_waiting_penalty` 的两组输入一致 ✓。
- KV 惩罚等价列：75% → 6·0.25 = 1.5/waiting → running + 2.5·waiting ✓；100% → 3/waiting → running + 4·waiting ✓；斜率 3/(1−0.5)=6 ✓（源码注释 “Ramps from 0 at <=50% usage to 3x waiting at 100%”）。
- C=2 burst 表：R1 同分从 E0（1/0）→ R2 基础 2/0 选 E1（1/1）→ R3 同分从 E0（2/1）→ R4 基础 4/2 选 E1（2/2）✓，差值 ≤ 1 ✓。引用的 `test_dplb_burst_round_robins_despite_snapshot_rebinds` 确为 4 Engine、`client_count=1`、8 请求 → 每个 2 ✓（页面已把 C=2 标为自建例子）。
- Coordinator 合成：三个独立 `if` 顺序 publish_back → publish_front → output_back；前端订阅/SCALE 分支 `continue` 会跳过本轮 `output_back` ✓；`wait_for = 100 if stats_changed else 5000`，再减 elapsed；`enable_wave_coordination and last_step_counts is None` → `min_timeout = 50` ✓；`wave_state_changed` 的 `(None, wave, running)` 发布不更新 `last_publish_time` ✓；`last_step_counts` 只在 `stats_changed` 时做 `copy.copy` 快照且发布后不清 dirty ✓。
- 协议字节：ADD `b"\x00"`、ABORT `b"\x01"`、START_DP_WAVE `b"\x02"`（`send_multipart((type, encode((wave, exclude))))`）、UTILITY `b"\x03"` ✓；`client_index == -1` 分流到 coordinator PUSH、输出线程补 `outputs.engine_index` ✓；`make_zmq_socket` 的 `bind = socket_type not in (PUSH, SUB, XSUB)` 证实“B/C 由 API 绑定、D/E/F 由 Coordinator 绑定”✓；XPUB 一律设 `XPUB_VERBOSE` ✓；`first_req_sock_addr = get_open_zmq_inproc_path()` 证实“inproc PAIR”✓。
- Coordinator 启动：Pipe 回传顺序 `(publish_front, output_back, publish_back)` 的 `LAST_ENDPOINT`，父进程存为 `stats_publish_address` / `coord_out_address` / `coord_in_address` ✓；`local_only = not local_engines_only`、`local_only_eng = dp_size == dp_size_local`、elastic EP 置 False ✓；`_wait_for_zmq_addrs` timeout=120 ✓；订阅循环 `for _ in self.engines: recv() != b"\x01" → 记录错误并 return`，无自身超时 ✓。
- DP 同步：`dp_sync_interval` 默认 16（`Field(default=16, ge=1)`）、首步 + 整数倍步同步 ✓；`wave_complete` 用 `current_wave <= wave` ✓。
- 26 数值未变：`_get_output_rank` 4−2×1=2 ✓；广播 16 MiB×10 + 4×24 MiB×10 = 1120 MiB（单节点前提，见 N2）✓。

---

## 5. 结论

- 8 条 P0/P1（P0-1、P1-1…P1-5、E2E-6，含 18 的 hop-walk FAIL 与 26 的 variant FAIL）已全部按源码修复，四页 verdict 由 REJECT 转 PASS。
- 新增约 500 行文本未引入 P0/P1；新增 7 条 P2（其中 N1 是本次编辑造成的锚点回归）。
- 仍是阻塞项的是域级归属：**E2E-1（Elastic EP 编排仍无主）**、**E2E-2（23 → 18/13 的 Ray/external launcher 监督指针仍悬挂）**。两者都需要回到 planning 指定 owner，不是单页编辑能关闭的。
- 未动的 19 条旧 P2 中，18 的 P2-5/P2-9/P2-10（符号与类归属）与 25 的 P2-12/P2-13（参数名、索引语义）属于 constitution 已要求修的 provenance 缺陷，建议与 N1 一并处理。
