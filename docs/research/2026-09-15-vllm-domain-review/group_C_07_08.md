# Group C 独立评审：07 Scheduler / 08 KV Cache 管理

- 评审人：独立 reviewer（非作者），只读评审，未改动 wiki/skills/docs/源码 checkout。
- 冻结基线：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`（checkout `/Users/suhaibo/97-llm/vllm`，`git rev-parse HEAD` 核对一致，仅有未跟踪 `artifacts/`）。
- 方法：`source-fidelity.md`、`codebase.md`、`analysis-focus.md`、`page-review-rubric.md`、`document-types/feature-analysis.md`、`reviews/feature-analysis.md`；附加用户标准：功能点完备性（目的 / 输入输出契约 / 处理逻辑 / 边界约束 / 支持范围）与全域 E2E 连贯性。
- 交叉阅读：06（全文）、11/12（CoW、zero、SchedulerOutput 契约相关段）、16（lookahead、draft KV、07/08 引用段）、22（§5.1–§5.4 与 07/08 引用段）、10（kernel block / Mamba 归属段）、02 §3.3、index。
- 所有数值例均已按源码重放（见各锚点“重放”）。未运行任何测试或 GPU。

## 1. 结论行

```
| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 07_vllm_scheduler_analysis | pass | pass | pass | timing, transform | pass | 16/16 | PASS | feature: pass；无 P0；P1 均为跨页所有权/E2E 缺口（见 E2E 节），§10 路线 #3 锚点不精确（P2） |
| 08_vllm_kv_cache_management_analysis | pass | pass | pass | layout, transform, timing | pass | 16/16 | REJECT | feature: FAIL §4.5（缺 HMA 开关 `disable_hybrid_kv_cache_manager` 这一兄弟选择轴，且无任何页面拥有）；另有 §5.1 支持范围后置、prefix hash 功能点不完备、Mamba align 块生命周期被 07/12 委托却缺席 |
```

- 07 feature: **pass**。变体枚举依据来自 `SchedulerConfig.get_scheduler_cls` / `SchedulingPolicy` / `VllmConfig.__post_init__`，并点名 batch queue、KV/EC connector 等兄弟轴；每个 load-bearing 功能点（接入、就绪、预算、联合保留、抢占、waiting 准入、计划发布、结果对账、stop/free、交付、暂停/维护、观测）都有目的、输入输出、处理逻辑与边界；成本账（§9）与局部成本一致；闭环终点到 `EngineCoreOutputs` 与资源各自完成条件。
- 08 feature: **FAIL §4.5**。§4.5 声称“实现集合以源码选择点为准”，但 `get_kv_cache_groups()` 的第一个选择点 `scheduler_config.disable_hybrid_kv_cache_manager → unify_hybrid_kv_cache_specs()` 未出现；它决定 §5.2/§5.3 整个 hybrid 平面是否存在，且会被非 HMA connector、非 GPU 平台、chunked-local attention 自动打开（见 P1-2）。按 `reviews/feature-analysis.md`“sibling selection axis … neither explained nor named with its owner”即判失败。其余单元（分页、refcount/free queue、allocate_slots、fixed-point、packed layout、CPU tier）均通过重放。

## 2. 锚点抽查记录

### 07（16 处，全部与页面一致）

| # | 锚点 | 页面主张 | 源码实际 | 结果 |
|---|---|---|---|---|
| 1 | `vllm/config/scheduler.py::SchedulerConfig.get_scheduler_cls` | 未指定 `scheduler_cls` 时 async→`AsyncScheduler`，否则 `Scheduler`；自定义类有非稳定接口警告 | 完全一致 | ✓ |
| 2 | `vllm/config/vllm.py::VllmConfig.__post_init__`（async 解析段） | pooling 默认关；非 EAGLE/NGramGPU/draft_model/dspark、`disable_padded_drafter_batch`、executor 不支持、ROCm DeepEP HT DBO 自动关；显式开启报错 | 一致；`max_concurrent_batches`：async+V2=`pp+1`，async+V1+PP≤1=2，否则 pp | ✓ |
| 3 | `Scheduler.schedule` running 段 | 候选量 = `num_tokens_with_spec + placeholders − computed`；顺序 long-prefill→token/input→max_model_len（减采样位）→Mamba→encoder→lookahead；零则 `continue` | 一致 | ✓ |
| 4 | `Scheduler.schedule` 抢占循环 | priority 取 `max(running,key=(priority,arrival_time))`，前方 victim 调小游标，已登记则撤回 token/`restored+draft_slots`/blocks/spec/encoder；FCFS `running.pop()`；victim==当前请求 → `new_blocks is None` → break 整个 running 扫描 | 一致 | ✓ |
| 5 | `Scheduler.schedule` waiting 门 | `if not preempted_reqs and _pause_state == UNPAUSED` | 一致 | ✓ |
| 6 | `Scheduler.schedule` connector 前缀 | 38 本地命中 → 对齐 32、tail 6；ext=4/6 保留本地、external=0；ext=8 截回 32、external=8 → 40 | `partial_tail and ext_tokens > partial_tail` 分支，重放一致 | ✓ |
| 7 | `KVCacheManager.allocate_slots` / `watermark_blocks` | full-seq 门由 `scheduler_reserve_full_isl` 传入；watermark 仅对 WAITING/PREEMPTED 且 `has_scheduled_reqs=bool(self.running)`；running 扩展不受限；48 输入/16/剩 2 块被拒 | 一致（`watermark_blocks=int(watermark*num_blocks)`）；48→3 块>2 重放一致 | ✓ |
| 8 | `Scheduler._preempt_request` | computed=0、清 draft、`num_stale_output_tokens = num_in_flight_tokens`（赋值）、placeholders=0、按策略 prepend、reset id | 一致 | ✓ |
| 9 | `Scheduler.update_from_output` 循环顺序 | 先减 in-flight/stale → KV load 失败跳过 → 终态跳过 → drop stale 跳过 → rejection 回退（非 stale 才改计数） | 一致；spec 例 21/20/3 排 4、接受 1+1：computed 24→22、placeholders 4→2→0 重放一致 | ✓ |
| 10 | `AsyncScheduler._update_after_schedule` / `_update_request_with_output` | placeholders += `num_sampled_tokens_per_step + len(spec)`；仅更新前 RUNNING 才 `cache_blocks`；非 stale 才扣 placeholders | 一致；§8.2 四元组两路 21/20/0/0→23/22/0/0 重放一致。注：`next_decode_eligible_step` 只在 `use_v2_model_runner` 时设置（P2-1） | ✓ |
| 11 | `Scheduler.__init__` defer gate、`_free_request_blocks`、`_drain_deferred_frees` | gate = KV consumer 且 `max_concurrent_batches > 1` | 一致 | ✓ |
| 12 | `Scheduler.get_num_unfinished_requests` / `has_finished_requests` / `has_requests` | PAUSED_ALL→0、PAUSED_NEW→len(running)、否则扣 streaming 会话；has_requests 含 connector pending push | 一致 | ✓ |
| 13 | `Scheduler._mamba_block_aligned_split` + `vllm/v1/kv_cache_interface.py::is_mamba_prefill_checkpoint_valid` | 1984/3602/1600：initial 列 1、checkpoint 列 1 → 无效、先算 1216；0/100/96、hash8/block64/align16 有效，88 无效 | `initial=(start-1)//mb`、`ckpt=cdiv(end,mb)-2`，重放一致 | ✓ |
| 14 | `Scheduler._reserve_prefill_lookahead` | prompt10、lookahead3、候选 8 → 7 | remaining=2 → 8−1=7，一致 | ✓ |
| 15 | `Scheduler._update_requests_with_invalid_blocks` / `_handle_invalid_blocks` | 截到首个坏块 `idx*block_size`；sync recompute 保留 blocks；fail 驱逐坏块及后缀；单组 TODO | 一致（`(req_block_ids,) = ...` 解包单组） | ✓ |
| 16 | tests：`test_priority_scheduling_preemption`、`test_schedule_partial_requests`、`test_no_spec_tokens_scheduled_for_prefill_chunks`、`test_schedule_order`、`test_delayed_kv_connector_free_keeps_scheduler_active` | L/H 容量例（6 块含 null）；800/100/100→1/700/0；50→30→4；800/800/10/10 与 1024；空计划+`finished_sending` 删除对象 | 全部一致；L/H priority 与 FCFS 两路按源码重放一致 | ✓ |

### 08（16 处，全部与页面一致）

| # | 锚点 | 页面主张 | 源码实际 | 结果 |
|---|---|---|---|---|
| 1 | `BlockPool.free_blocks` + `FreeKVCacheBlockQueue.prepend_n/append_n` | 无 hash 零引用块放队首（LIFO），有 hash 放队尾 | `prepend_n` 保序插头、`append_n` 插尾，一致 | ✓ |
| 2 | `tests/v1/core/test_prefix_caching.py::test_prefill` | 16-token 块，公共 [1,2,3] ref 2，私有尾 4/5，全释放后 `[5,4,6,7,8,9,10,3,2,1]` | 一致 | ✓ |
| 3 | `BlockPool.touch` / `is_block_writable` / `reset_prefix_cache` | ref0 命中块摘出队列；可写=非 null、ref1、无 hash；reset 遇已用块返回 False | 一致（null block ref 不维护，源码注释同） | ✓ |
| 4 | `KVCacheManager.get_computed_blocks` | 上限 `num_tokens−1`；跳过读取返回空块/0；返回 `shared_prefix_boundary` | 一致 | ✓ |
| 5 | `KVCacheManager.allocate_slots` | full-seq 门（不减 reserved）→`remove_skipped_blocks(total−in_flight)`→计数→`free−reserved` 比较→touch/external→new→`cache_blocks(min(total+new,num_tokens))`；失败不回滚已回收窗口 | 一致 | ✓ |
| 6 | `KVCacheCoordinator.allocate_new_computed_blocks` | 先 touch 全部组本地命中，再为各组分配 external（#33775） | 一致 | ✓ |
| 7 | `SlidingWindowManager.get_num_skipped_tokens` | 窗口4、块2、processed 7 → 丢 0～3、前两槽 null；若按 9 会误删 4、5 | `computed − window + 1 − extra_retained`：7→4、9→6，一致 | ✓ |
| 8 | `SingleTypeKVCacheManager.add_local_computed_blocks/allocate_new_blocks/_apply_cow` + `Scheduler._free_cow_retained_blocks` | S:0→1（touch）；D:1→2（copy ref）；即时分支 S 1→0、D 2→1；defer 分支按 `sched_step_seq+1` 入队 | 一致 | ✓ |
| 9 | `MambaManager.allocate_new_blocks`（running CoW）/`get_num_blocks_to_allocate` | running 请求保留 S，`move_block_hashes(S→D)`、S ref+1、D 进 `cached_blocks_this_step`；命中本步集合的尾块返回 `num_gpu_blocks+1` | 一致 | ✓ |
| 10 | `MambaManager.finalize_partial_tail_offload`、`MooncakeStoreScheduler.register_finished_partial_tail/has_pending_push_work` | 仅 computed==boundary 且 in-flight==0 交出；Mooncake touch 源块、返回 False、job 完成才释放；pending push 保活 | 一致 | ✓ |
| 11 | `HybridKVCacheCoordinator.find_longest_cache_hit` / `__init__` | full 优先、候选只降、full 截短复用、simple hybrid 少一轮；PCP 拒绝 hybrid、DCP 只收 full/Mamba；12→8→4 稳定 | 一致，重放一致 | ✓ |
| 12 | `_get_packed_kv_cache_groups` + `_approximate_gcd` + `_get_kv_cache_bytes_per_block` + `get_kv_cache_config_from_groups` | A3/B3 与 C5：下界 3，pad(3)=1、pad(4)=4、pad(5)=2 → 3；C 分 [C0,C2,C4]/[C1,C3]；stride=max 组=12 KiB；60 KiB→5 块 | 一致；SVG 中 A0‥A2、B0‥B2 连续排布与按 spec 分段的 offset 循环一致 | ✓ |
| 13 | GLM5-Next 测试（`tests/v1/core/test_kv_cache_utils.py`） | 9/9/8/8、PP 投影 5/5/4/4、全局 7/7/7/7/6 | 断言一致 | ✓ |
| 14 | `CPUOffloadingManager.prepare_store/complete_store/prepare_load/complete_load`、`BlockStatus.is_ready`、`VllmConfig` offload 选择 | pending ref=-1；成功→0/可驱逐；失败删除未完成项；双 load 0→1→2→1→0；native 默认 `OffloadingConnector`、`VLLM_USE_SIMPLE_KV_OFFLOAD` 选 Simple | 一致 | ✓ |
| 15 | `_gen_lora_extra_hash_keys`、`OpenAIServingModels.unload_lora_adapter` | hash 用 LoRA 名称；卸载只删前端映射 | 一致 | ✓ |
| 16 | 容量算例 + `vllm/v1/worker/gpu/model_runner.py::GPUModelRunner.update_requests` | 512 B/token/层、40 KiB/token/rank、640 KiB/pool id；zero 后 CoW copy 再 forward | 算术与源码顺序一致 | ✓ |

## 3. 编号发现

> 未发现 P0（误导性事实错误）。以下均已对 checkout 或他页核实，除非标注“unverified suspicion”。

### P1

**P1-1（E2E / 07 §7 ↔ 08 §3.3）回收型规格的逐请求准入上限无人拥有**
- 页面主张：07 §7“KV manager 先估计整个已知输入序列的块需求（考虑 prefix 与窗口等）…allocator 如何计算共享、窗口和异构组的块需求仍由 08 页负责”；08 §3.3.1“full-sequence admission…其 watermark 条件、reserved blocks 和抢占选择归 07”。
- 源码：`KVCacheManager.allocate_slots` 的 full-seq 门调用 `coordinator.get_num_blocks_to_allocate(..., apply_admission_cap=True)`；`SingleTypeKVCacheManager.get_num_blocks_to_allocate` 用 `_max_admission_blocks_per_request` 截断；`get_manager_for_kv_cache_spec` 仅对 `SlidingWindowSpec`/`ChunkedLocalAttentionSpec` 注入 `max_admission_blocks_per_request(max_in_flight_tokens, max_model_len)`（`min(window或chunk + max_in_flight_tokens, max_model_len)` 块），注释说明它是启动池规模与运行准入的“single source of truth”，漂移会导致 #39734 死锁或 prefill 中途 OOM；`VllmConfig.max_in_flight_tokens`。07 引用的 `test_can_fit_full_sequence_full_attention_still_gates_oversized` 的主题正是“cap 只放宽 SWA 组”。全域 grep `max_in_flight|admission_cap|max_admission` 无命中。
- 建议：在 08 §3.3 增加一个功能点单元（目的：避免 SWA/chunked-local 被按全长预留而过度拒绝；输入：`max_in_flight_tokens`、窗口/chunk；输出：块数上限；边界：R-SWA 不设 cap、仅 full-seq 门使用、逐步分配必须不带 cap），07 §7 保留一句并链接。

**P1-2（08 §4.5，feature FAIL）缺 HMA 开关这一兄弟选择轴**
- 页面主张：“实现集合以源码选择点为准”，表内列 `get_kv_cache_coordinator`、registry、`get_kv_cache_groups`、offload。
- 源码：`vllm/v1/core/kv_cache_utils.py::get_kv_cache_groups` 首先 `if scheduler_config.disable_hybrid_kv_cache_manager: unify_hybrid_kv_cache_specs(...)`；`VllmConfig.__post_init__` 在平台不支持 hybrid、chunked-local attention（EAGLE 或未设 `VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE`）、或 connector 不支持 HMA（`KVConnectorFactory.supports_hma_config`）时自动置 True，显式开启冲突则报错；`Scheduler._connector_finished` 对非 `SupportsHMA` connector 断言单组。全 vLLM 域 grep `disable_hybrid_kv_cache_manager` 无命中。
- 建议：§4.5 表新增一行（入口、自动关闭条件、效果：所有层统一为可合并规格、失去 §5.2/§5.3 hybrid 行为、SWA 退为全量保留而性能下降、hybrid SSM 启动失败），并在 22 的 connector 能力处反链。

**P1-3（08 §2.3，功能点“prefix hash”不完备；E2E：15 委托而 08 未落地）**
- 页面主张：“prefix hash 链接父 hash、当前完整 hash 单元的 tokens，以及必要的额外语义键…再加 group id”。
- 源码：hash 在 `EngineCore.__init__` 仅当 `enable_prefix_caching or kv_connector` 时由 `get_request_block_hasher(hash_block_size, get_hash_fn_by_name(cache_config.prefix_caching_hash_algo))` 构造，按 `hash_block_size` 逐块 `hash_block_tokens(fn, parent or NONE_HASH, tokens, extra_keys)`；`CacheConfig.prefix_caching_hash_algo` 默认 `sha256`（另有 `sha256_cbor`、`xxhash`）；`resolve_none_hash_seed/init_none_hash`：非密码学 hash 且未设 `PYTHONHASHSEED` 时每进程随机种子，hash 跨进程不可复现。15 页 §1 明言“extra key 的组合与 `hash_block_tokens` 归 08”，但 08 未出现 `hash_block_tokens`/算法/种子；22 的 store/offload 键跨实例复用依赖此契约。
- 建议：08 §2.3 补“输入（token 块、父 hash、extra keys、hash_block_size）→ 输出（BlockHash 链）→ 计算时机（请求创建/追加 token）→ 边界（算法选择、NONE_HASH 种子与跨实例可复现性、碰撞取舍）”。

**P1-4（08 §5.1，支持范围在首次出现处缺失）**
- 页面主张：§5.1 以“hash 粒度 2、group 块长 4 的 6-token 命中”引入 partial/CoW，§6.1 成本行“Partial CoW”无适用条件，范围直到 §5.3 才一句带过。
- 源码：细粒度命中仅在 `HybridKVCacheCoordinator.enable_partial_hash_hits` 为真时启用（存在 Mamba `align` 组且其 block_size > hash_block_size，且所有可缓存 manager `supports_fine_grained_hash_lookup` 或块长等于 hash 粒度）；`UnitaryKVCacheCoordinator` 在开启缓存时断言 `hash_block_size == block_size`；`FullAttentionManager.cache_blocks` 在 `block_size == hash_block_size` 时不登记 partial tail。测试文件 `tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py` docstring 明写“for hybrid (full attention + mamba "align") models”，且未列入 §6.3 路线。
- 建议：§5.1 开头给出启用条件与“纯 full-attention 模型不会出现 partial CoW”；§6.1 成本行加适用条件；路线补该测试文件。

**P1-5（E2E / 07 §5.4、12 ↔ 08）Mamba align 状态块生命周期被委托却缺席**
- 页面主张：07 §5.4“具体 state block 的引用、轮换与回池归 08 页”“块分配与 checkpoint 的物理保存、partial-tail hash/CoW 仍由 08 页展开”；12 页 `MambaHybridModelState` 行“块与 checkpoint 语义见 08 §5.1.2”。
- 源码：`MambaManager.allocate_new_blocks`（null 填充跳过状态、`num_speculative_blocks`、`last_state_block_idx`、`_relocate_speculative_block`）、`MambaManager.remove_skipped_blocks`（释放两步前状态块）、`get_num_skipped_tokens = computed−1`、内部 prefill checkpoint 槽（`_needs_internal_checkpoint`、`checkpoint_idx = cdiv(end, block)−2`、`_checkpoint_positions`、`_cache_partial_tail_block(..., replace_existing_hashes=True)`）。08 仅在 §4.3 一句“Mamba 保存恢复所需的状态边界”与 §5.1.2 的 producer CoW 特例；轮换与 checkpoint 物理槽无解释。
- 建议：08 §5 增一小节以同一 1600-block 例重放“每步分配/轮换/释放 + 内部 checkpoint 槽与 hash 重键”，或修改 07/12 的委托文字并登记缺口。

**P1-6（E2E / 08 §3.3.2 ↔ 16）可重填尾部的缓存登记规则循环委托**
- 页面主张：08“多模块投机路径还会扣除可能再次 prefill 的尾部，见 16”，Related Pages 称 16“展开…哪些 token 内容可以登记为缓存”。
- 源码与他页：`KVCacheCoordinator.num_reprefillable_tokens = max(0, num_prefill_lookahead − 1)`、`KVCacheCoordinator.cache_blocks` 扣除；`HybridKVCacheCoordinator.cache_blocks` 对 EAGLE 组额外允许登记一个 lookahead 块；`SlidingWindowSpec.extra_retained_tokens`。16 页 grep 无 `reprefill`/`cache_blocks`，且 16 明言“hash/refcount/prefix 命中归 08”。
- 建议：规则归 08（它拥有 `cache_blocks`），补“登记长度 = min(total+new, num_tokens) − (lookahead−1)，EAGLE 组 +1 块，SWA 保留 extra_retained”并删掉指向 16 的承诺。

**P1-7（E2E / 07 §8.3–§8.5 ↔ 06 §5.1、§6.1–§6.2）同一机制两份完整正文与图**
- 页面主张：07 §8.3 表与图 7（普通/drop stale、失效 KV、终态）；§8.5 三种“完成信息”表；§8.4 defer fence。06 §5.1 图 3（同一 normal/drop 分支）、§6.1 四种 finished 表、§6.2 fence 图。
- 核对：两页内容当前与源码一致、互不矛盾（stale 赋值、drop 保持、gate 条件、`finished_requests` 语义均同），但构成两个真相源；index 将“两批在途工作怎样配对并释放资源”分给 06，07 §8.4 自己也写“fence 时序已在 06 页重放”。
- 建议：stale/drop 计数规则留 07（它拥有 `update_from_output`），06 只保留“为何需要原计划配对”并链接 07 §8.3；finished 信号表只留一处（建议 06 §6.1），另一处改为链接。22 §5.1 与 07 §7 对 partial-tail/ext_tokens 的规则也重复（一致），同样收敛到 07。

### P2

1. **07 §5.1**：“`AsyncScheduler._update_after_schedule()` 在非 partial-prefill 请求提交后，把 `next_decode_eligible_step` 设为 `current_step + pp_size`”——源码仅在 `self.use_v2_model_runner` 时设置；补条件。
2. **07 §7 / §10 路线 #3**：DP prefill 节流未点名配置 `SchedulerConfig.prefill_schedule_interval`（默认 1 即关闭），真实实现是 `DPEngineCoreProc._should_throttle_prefills`（`interval>1 and step_counter % interval != 0`）；路线列出的 `EngineCore._should_throttle_prefills` 基类恒返回 False。
3. **07 §7**：watermark 是比例（`int(watermark * kv_cache_config.num_blocks)`），例子“余量要求 2 块”未给换算。
4. **07 §5.2**：“batch 尚无已排 prefill”——`prefill_scheduled` 只在 running 段更新，waiting 段新准入的 prefill 不阻止后续 padding；“容量不足…本轮先不准入”实为 `break` 终止整个 waiting 扫描。
5. **07 §4.4 / E2E**：“计算共同前缀”（`KVCacheManager.get_num_common_prefix_blocks`：以 running[0] 为基准、`ref_cnt == len(req_to_blocks)`、SWA/Mamba 恒 0）在 07/08 均无解释，而 11 §2.6.1 消费它；功能点无主。
6. **07 §8.4**：WAITING_FOR_REMOTE_KVS 请求被 abort 时的延迟释放是 Scheduler 规则（`finish_requests` 中 `delay_free_blocks = request_id not in finished_recving_kv_req_ids`），页面归为“connector 要求 delay”；22 §6.2 有该路径的泄漏分析，可加链接。
7. **07 §5.3**：encoder 例“本步 encoder budget 只有 4、需要 6”未说明是剩余预算还是每步上限；若为上限则该项永不可调度，例子易误读。
8. **07 §3 终态表**：FINISHED_STOPPED 的 resumable 分支也可能进入 `WAITING_FOR_STREAMING_REQ`，不只“复位到 WAITING”（§3.1 已正确描述，表格简化）。
9. **08 §2.3**：“[[24…|LoRA]]”链接目标 24 页只讲 LoRA Resolver 插件，不涉及 adapter 权重/KV 隔离；链接标签误导。
10. **08 §5.3**：未点名 `CacheConfig.prefix_match_unit`（显式 hash 粒度，`resolve_kv_cache_block_sizes` 校验）与 `CacheConfig.prefix_cache_retention_interval`（None/0/正值策略的配置入口）。
11. **08 §6.3 路线**：正文引用但路线未列的测试：`test_prefix_caching.py::test_cache_hit_local_and_external`（两阶段 touch）、`prefix_cache/test_partial_prefix_cache_hits.py`（partial/CoW）、`test_deferred_block_free.py::test_cow_retentions_deferred_until_copy_step_processed`。
12. **08 §4.6 调用树**：schedule 树缺 `take_kv_cache_block_copies → Scheduler._free_cow_retained_blocks` 与 `take_boundary_state_offloads`，而 §5.1 以它们为 load-bearing。
13. **08 §3.2 与 §5.1**：“至多复用 `num_tokens−1`、缓存不保存 logits”段落重复。
14. **入站引用漂移**：16 页 §8 称“完整的四元组账本与 21/20/3 的逐步示例见 07 §8.1”，四元组表实在 07 §8.2；12 页指向“08 §5.1.2”取 checkpoint 语义，实际多在 07 §5.4（且见 P1-5）。
15. **术语**：`update_from_output` 在 06 称“归并/结算”，07/11/12 称“对账”，22 称“接收收尾”（不同事件，可接受）；建议 06 与 07 统一为“结果对账”。

## 4. E2E / 连贯性

### 4.1 两页在端到端链中的位置与交接

| E2E 步骤 | 所有者 | 07/08 的交接情况 |
|---|---|---|
| 前端登记 → Core `add_request` | 03/06 | 07 §3.1 从 `Scheduler.add_request` 接手，06 §2.1 负责线传与输入队列；衔接清楚 |
| waiting/running 选择、预算、就绪 | 07 §2–§7 | 完整 |
| 前缀查找 / 分配 / 窗口回收 | 08 §3.2–§3.3（被 07 §4.3/§7 调用） | 调用点、返回语义一致；**准入 cap 无主（P1-1）** |
| 远端命中 / async load | 07 §7（调度侧）、22 §5（协议侧） | 规则一致但重复（P1-7 附注） |
| `SchedulerOutput` 构造与字段 | 07 §8.1；12 §3 字段消费 | 12 引 07 §4.1/§8.1 有效 |
| runner 块表、zero、CoW copy、slot mapping | 08 §3.4（V2）→ 11/12、10 | 与 11/12 的 zero→copy→forward 顺序一致；10 负责 manager/kernel 虚拟拆分，08 已链接 |
| draft KV / lookahead 预留 | 16 §6.3–§6.4 ↔ 07/08 | `num_lookahead_tokens` 边界一致；**可重填尾部缓存登记循环委托（P1-6）** |
| 结果配对（future FIFO） | 06 §4 | 07 §8.2 仅引用，一致 |
| 对账、stale、失效 KV | 07 §8.3（06 §5 重复，P1-7） | 一致 |
| stop / 终态 / finished 信号 | 07 §8.4–§8.5（06 §6.1 重复） | 一致 |
| `_free_request` → connector delay | 07 §8.4、22 §5.4 | 一致；abort-in-remote-wait 细节见 P2-6 |
| deferred fence / CoW retention | 06 §6.2、07 §8.4、08 §3.5/§5.1.1 | 三页一致（gate、`sched_step_seq+1`、零 token 步不推进） |
| `BlockPool.free_blocks` → 可驱逐/保留 hash | 08 §2.2/§3.5 | 完整 |
| Mamba align 状态块轮换与 checkpoint 槽 | 07 → 08 委托 | **缺失（P1-5）** |
| hybrid 开/关（HMA） | 无 | **缺失（P1-2）** |
| block hash 生成算法/种子 | 15 → 08 委托 | **缺失（P1-3）** |
| cascade 共同前缀块数 | 11 消费 | 生产侧无主（P2-5） |

### 4.2 首要连贯性问题（按影响排序）
1. 回收型 KV 规格的准入上限（`max_admission_blocks_per_request` / `max_in_flight_tokens`）在 07 与 08 之间互相推诿，全域无解释（P1-1）。
2. HMA 开关及其自动关闭条件无页面拥有，08 的变体枚举因此不完整（P1-2）。
3. prefix hash 的生成契约（算法、NONE_HASH 种子、跨实例可复现性）被 15 委托给 08 而 08 未写，22 的跨实例键依赖它（P1-3）。
4. Mamba align 状态块生命周期被 07/12 委托给 08 而 08 缺席（P1-5）；可重填尾部缓存规则在 08↔16 循环委托（P1-6）。
5. 06 与 07 对 stale 输出、finished 信号、fence 各有一套完整正文与图；当前一致，但违反单一真相源（P1-7）。

### 4.3 未发现的问题（明确说明）
- 未发现 07 与 08、06、11、12、16、22 之间的事实矛盾：抢占 victim、running-first、waiting 门、async placeholder、stale 赋值规则、defer gate、free queue 次序、两阶段 touch、fixed-point 命中、CoW 引用账、CPU tier pending/ready 在各页表述一致。
- 两页的数值例（R/P/Q、L/H、四元组、spec 21/20/3、Mamba 1984/3602、lookahead 10/3、A/B 分页、test_prefill、窗口 9−2、12→8→4、packed 9/12/8、60 KiB、容量 512 B）全部可按源码重放。
