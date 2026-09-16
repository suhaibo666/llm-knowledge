# Group C 复评（recheck）：07 Scheduler / 08 KV Cache

- 评审人：独立 reviewer（非作者）。只读评审，未改动 wiki / skills / docs / 源码 checkout。
- 冻结基线：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`（`git log -1` 核对：`199cb9b9…`，commit time 2026-09-06 17:54 -0700 = 2026-09-07 00:54 UTC）。checkout `/Users/suhaibo/97-llm/vllm`，HEAD 未移动。
- 被评对象：工作区未提交改动（`git diff`）。07 +95 行、08 +179 行；26 个 vLLM 页 + index + changelog 同批改动。
- 方法：先读上轮报告 `docs/research/2026-09-15-vllm-domain-review/group_C_07_08.md`，逐条按**源码**（不是新 prose）核验；新增段落全部重放数值例；跨页交接另行核对邻页 diff。
- T0 门（本机 `.venv` py3.13）：`check_links --strict` 452 页 **0/0/0/0/0**；`check_math --changed --strict`、`check_markdown --changed --strict`、`check_assets --changed --strict` 各 39 文件 **0 error 0 warning**。无新增门失败。

## 1. 结论行

```
| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 07_vllm_scheduler_analysis | pass | pass | pass | timing, transform | pass | 14/14 新增点 | PASS | feature: pass §8.1.1–§8.1.3（22/22 字段齐、无重复无遗漏）；页内无 P0/P1；新增 P2 五项（NEW-3/5/6/8/9）+ 上轮 P2 六项未修；P1-7 仍未收敛且被扩大 |
| 08_vllm_kv_cache_management_analysis | pass | pass | pass | layout, transform, timing | pass | 22/22 新增点 | PASS（上轮 REJECT 已解除） | feature: pass §4.5（HMA 兄弟轴已补，§4.5.1 拥有）；P1-1…P1-6 全部 FIXED；页内无 P0/P1；新增 P2 四项（NEW-1/2/7/X-3）+ 上轮 P2 五项未修 |
| （跨页，缺陷在 12） | — | — | — | — | — | 2/2 | 待 12 修 | X-1 `align 只缓存块边界` 与 08 §5.1.3 矛盾（P1）；X-2 `max_concurrent_batches` 与 06 §4.1 及源码矛盾（P1） |
```

- 07/08 页内**无 P0、无新增 P1**。跨页新发现两个 P1，缺陷都落在 **12** 上（X-1 align 缓存规则与 08 §5.1.3 矛盾；X-2 `max_concurrent_batches` 与 06 §4.1 及源码矛盾），交 12 的作者处理。
- 上轮六个 P1 中 **P1-1…P1-6 全部 FIXED**（均对冻结源码逐条核实，非仅凭新 prose）；**P1-7 NOT_FIXED**，且 07 新增 §8.1.2 把它指出的重复扩大成三处（见 §4.2 / NEW-5）。
- **Merge blockers（3）**：P1-7（06↔07 双真相源未收敛，且 07 §8.1.2 新增第三份副本）、X-1、X-2（两处 12 与 08 / 06 / 源码矛盾的事实）。
- 其余（NEW-1/2/3/5/6/7/8/9 与 11 项未修 P2）均可与后续编辑合并处理，不阻塞。
- 状态计数 —— **07**：FIXED 1（P2-5）、PARTIAL 1（P2-1）、NOT_FIXED 7（P1-7、P2-2/3/4/6/7/8）、新增 P2 5；**08**：FIXED 6（P1-1…P1-6）、PARTIAL 2（P2-11、P2-14）、NOT_FIXED 5（P2-9/10/12/13、P2-15）、新增 P2 4。

- 上轮 08 的 feature FAIL 原因（`disable_hybrid_kv_cache_manager` 这一兄弟选择轴无人拥有）已解除：§4.5 表首行 + §4.5.1 完整功能点（目的 / I-O / 三态解析表 / 资源变换 / 失败边界 / 源码路线）。
- 07 被要求补的 `SchedulerOutput` 生产端字段合同已交付并且**字段数可核**：dataclass 22 个顶层字段，页面 9+10+3=22，逐一对应，无遗漏无重复；`NewRequestData` 11/11、`CachedRequestData` 7/7 亦全覆盖；`GrammarOutput` 被正确排除。

## 2. 上轮发现逐条状态

### P1

| 编号 | 状态 | 源码级证据（全部对 checkout 核实） |
|---|---|---|
| **P1-1** 回收型规格逐请求准入上限无人拥有 | **FIXED**（残留一处 P2，见 NEW-1） | 08 §3.3.1 新增完整功能点。公式核对 `vllm/v1/kv_cache_interface.py::SlidingWindowSpec.max_admission_blocks_per_request` = `cdiv(min(sliding_window-1+extra_retained_tokens+max_in_flight_tokens, max_model_len), block_size) + 1`、`ChunkedLocalAttentionSpec.max_admission_blocks_per_request` = `cdiv(min(attention_chunk_size+max_in_flight_tokens, max_model_len), block_size)`：页面两式**逐符号一致**，"滑窗多出的 1 块" 与源码 `+1` 注释一致，chunk 无 +1 一致。`VllmConfig.max_in_flight_tokens`（`config/vllm.py:580`）= `max_concurrent_batches * max_num_batched_tokens` 一致。`max_memory_usage_bytes` 复用同一方法乘 `page_size_bytes` 一致；SWA 版断言 `decode_context_parallel_size == 1` 一致。`get_manager_for_kv_cache_spec` 仅对 `(SlidingWindowSpec, ChunkedLocalAttentionSpec)` 注入 cap、注释明写 R-SWA 继承 full-attention 界，页面"R-SWA 不设独立 cap"一致。`KVCacheManager.__init__`（`kv_cache_manager.py:140`）`max_in_flight_tokens is None → max_model_len` 回退一致。**重放教学例**：B=16、W=Q=64、E=0、M=1024、I=2×128=256 → SWA `ceil(319/16)+1 = 20+1 = 21`、chunk `ceil(320/16)=20`、full 组 `ceil(1024/16)=64`，与页面 21/20/64 完全一致。07 §7 已收敛为一句 + 链接。 |
| **P1-2** 缺 HMA 开关兄弟选择轴（08 feature FAIL） | **FIXED**（残留一处 P2，见 NEW-2） | 08 §4.5 新增表行 + §4.5.1。六行解析表逐项对 `vllm/config/vllm.py::VllmConfig.__post_init__`（1786–1857）核对：`not current_platform.support_hybrid_kv_cache()` → 要求关闭 ✓；`attention_chunk_size is not None` 且 `speculative_config.use_eagle()` → 要求关闭 ✓；否则 `not envs.VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE` → 要求关闭 + warning ✓；`is None` 分支才查 `KVConnectorFactory.supports_hma_config(kv_transfer_config)` ✓；显式 `False` + need_disable → `ValueError` ✓；显式 `True` 尊重 ✓；末尾 `is None → False`（默认开启）✓。双重否定提示正确。`MultiConnector` 断言"all sub-connectors support HMA"✓。资源变换核对 `get_kv_cache_groups`（`kv_cache_utils.py:2193`）先 `unify_hybrid_kv_cache_specs`；后者对已 uniform / uniform-type 原样返回 ✓，否则 `_promote_local_kv_cache_specs`：`SlidingWindowMLASpec→MLAAttentionSpec`、`SlidingWindowSpec→FullAttentionSpec`、`ChunkedLocalAttentionSpec→FullAttentionSpec`，`drop=("extra_retained_tokens",)`、统一 `block_size` 与 `page_size_padded` ✓ —— 页面"SWA MLA 对应 MLA""移除已无意义的 extra_retained_tokens"一致。"full+Mamba 的 hybrid SSM 抛 `ValueError`" **核实为真**：该组合无可提升项 → `promoted_specs` 仍非 uniform → `raise ValueError("Failed to promote local KV cache specs to one unified type.")`（`kv_cache_utils.py:1837`）。"模型仍按原窗口计算 attention"与 `_promote_local_kv_cache_specs` docstring 一致。 |
| **P1-3** prefix hash 功能点不完备（15 委托未落地） | **FIXED** | 08 §2.3 新增"生成合同与时机"+ 算法表 + 种子段 + 源码路线。`EngineCore.__init__`（`v1/engine/core.py:224-231`）`if cache_config.enable_prefix_caching or kv_connector is not None` 才建 `request_block_hasher` ✓。`Request.__init__` 调 `update_block_hashes()`（`request.py:224`）、`append_output_token_ids` 再调（:276）✓。`get_request_block_hasher` 从 `len(request.block_hashes) * hash_block_size` 续算、只产完整单元、`hash_block_tokens(fn, parent or NONE_HASH, tuple(tokens), extra_keys)` ✓。算法表对 `vllm/utils/hashing.py`：`sha256`=pickle+SHA-256 ✓、`sha256_cbor`=canonical CBOR+SHA-256 ✓、`xxhash`/`xxhash_cbor`=pickle/canonical CBOR + `xxh3_128_digest` ✓、可选依赖 ✓、默认 `sha256`（`config/cache.py:141`）✓。种子段对 `resolve_none_hash_seed`：`PYTHONHASHSEED` 优先 → 非密码学 `os.urandom(32).hex()` → 否则 `DEFAULT_NONE_HASH_SEED` ✓；`init_none_hash` 再散列得 `NONE_HASH` 并对非密码学+无 PYTHONHASHSEED 告警"not reproducible across processes" ✓；`get_none_hash_seed()` 供 P2P 握手 ✓。"canonical CBOR 只固定序列化，不会自动消除 xxhash 的随机首节点差异"**正确**。**重放数值例**：hash 粒度 4、10 token → `start=0`，产生 end=4、end=8 两个链节点（end=12>10 break）；追加到 12 → `start=2*4=8`，产生第三个，含第二个 hash → 识别整段 12-token 前缀 ✓。 |
| **P1-4** partial/CoW 支持范围在首次出现处缺失 | **FIXED** | 08 §5.1 开头新增"先限定支持范围"。对 `HybridKVCacheCoordinator.__init__`（`kv_cache_coordinator.py:640-668`）逐条核对：`has_partial_mamba_group` = 存在 `MambaSpec` 且 `mamba_cache_mode=="align"` 且（`dcp==1 and block_size > hash_block_size`）或（`dcp>1 and block_size >= hash_block_size`）—— 页面"非 DCP 时大于；DCP>1 时可相等（有效 attention 块另被放大）"与源码注释一致 ✓；`unsupported_partial_hit_managers` 只看 `group.kv_cache_spec.prefix_cacheable and not supports_fine_grained_hash_lookup and block_size != hash_block_size`，命中则关闭 + `warning_once` ✓，"`prefix_cacheable=False` 的 scratch 不参与"✓；`assert pcp_world_size == 1` ✓；`dcp>1` 时断言每组为 `(FullAttentionSpec, MambaSpec)` ✓；`UnitaryKVCacheCoordinator` `assert not enable_caching or hash_block_size == block_size` ✓。§6.1 成本行已改为"Partial CoW（受 §5.1 能力门约束）"并补适用条件 ✓。§5.1.3 源码路线已登记 `tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py`（存在）✓。 |
| **P1-5** Mamba align 状态块生命周期被 07/12 委托却缺席 | **FIXED**（但新造一处重复，见 X-3；并暴露 12 的 X-1） | 08 新增 §5.1.3（目的 / I-O / 三种预算 / 四时点 / 两幅例 / 内部 checkpoint / 源码路线 + 两个回归入口）。核对：`MambaSpec.max_memory_usage_bytes` 三分支 = `all`:`cdiv(max_model_len,block)+num_speculative_blocks` page、`align`:`2+num_speculative_blocks+num_prefill_checkpoint_blocks` page、`none`:`1+num_speculative_blocks` page —— **逐项一致**；`max_num_blocks_per_req` align 仍按 `cdiv(max_len,block)+num_speculative_blocks` 且源码注释明写"row length must cover max_len"，页面"位置表长度仍按…因为旧位置变 null 而不是删列"一致 ✓。`allocate_new_blocks` align 分支：`num_tokens = num_tokens_main_model`（不含 lookahead）✓、`last_state_block_idx = prev_len-1-num_speculative_blocks`（已分配）/`prev_len-1`（新请求命中）✓、跳过列补 `_null_block` 且 `null_end = num_skipped_blocks - checkpoint_block` ✓、`max_new_blocks = 1 + partial + checkpoint (+spec)` ✓、`_allocated_block_reqs.add` ✓。`get_num_skipped_tokens = num_computed_tokens - 1` ✓。`remove_skipped_blocks` align 追加按 `last_state_block_idx < cdiv(processed,block)-1` 释放并置 null ✓。`cache_blocks` 把新 hash 与 partial hash 放入 `cached_blocks_this_step`、`get_num_blocks_to_allocate` 命中本步集合返回 `num_gpu_blocks+1`、`new_step_starts()` 下一步清集合 ✓。`pop_blocks_for_free` 清 `_allocated_block_reqs`/`last_state_block_idx`/`_checkpoint_positions`/`_producer_partial_tail_reqs` 并丢弃未交出的 `_pending_boundary_state_offloads` ✓。`_relocate_speculative_block` `assert is_block_writable`（非 null、ref1、无 hash）、append 到表尾、原列置 null、**不领块不复制** ✓；重定位只在 `blocks_allocated and not checkpoint_block` 分支，页面"checkpoint 分支可能重新分配 scratch，不能把重定位套到所有分支"✓。准入探测不写 `_checkpoint_positions`（`if not apply_admission_cap:` 才写）✓。**重放 1600 例**：步1 `num_required=1`→`[S0]`；步2 `last_state_block_idx=0`、`num_skipped=1`、新增 1 → `[S0,S1]`；`remove_skipped_blocks(3200)`：`0 < cdiv(3200,1600)-1 = 1` → 释放并置 null → `[null,S1]`；步3 → `[null,S1,S2]` ✓ 与页面三步快照一致；"初次 prefill 直接跳到 4800"：`num_skipped=2`→补两 null + 1 新块 ✓。**内部 checkpoint 例逐字命中回归测试**：`tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py::test_internal_checkpoint_uses_partial_hash_lifecycle`（hash16 / mamba block32 / 120 token / `checkpoint_idx=2` / `running_block=3` / `block_hash_num_tokens==112` / 96 键不再命中 / free 后重放命中 112），与页面"列2 checkpoint、列3 running、导出 state@112、替换临时 state@96、重放命中 112"完全一致；`_cache_partial_tail_block` 的 `checkpoint_idx = cdiv(num_tokens, block_size) - 2`、`replace_existing_hashes=True`、multi-module MTP 的 `TODO` 均在源码 ✓。 |
| **P1-6** 可重填尾部缓存登记规则循环委托（08↔16） | **FIXED** | 08 新增 §3.3.3 并把 §3.3.2 末句由"见 16"改为"见本页 §3.3.3"；Related Pages 对 16 的描述改为"draft 生成与 rejection；由本页 §3.3.3 闭合对应的 cache tail 登记边界"。三分支表逐条核对：基类 `KVCacheCoordinator.cache_blocks` → `max(0, num_computed_tokens - num_reprefillable_tokens)`，`num_reprefillable_tokens = max(0, num_prefill_lookahead - 1)` ✓；`HybridKVCacheCoordinator.cache_blocks` 普通组 = `_align_cacheable(C)`（partial hits 开启原样返回，否则 `round_down(·, scheduler_block_size)`）且**确实没有统一扣 R** ✓；EAGLE 组（`manager.use_eagle and cached_num_computed_tokens > 0`）= `min(F, _align_cacheable(F) + manager.block_size)`，`F = max(0, C-R)` ✓。**重放**：C=70、R=2、sched block 32、group block 8、partial 关 → 基类 68、普通组 `round_down(70,32)=64`、EAGLE `min(68, 64+8)=68` ✓；C=72、R=0 → EAGLE `min(72, 64+8)=72`、普通组 64 ✓。`Scheduler.__init__`：EAGLE 且 `use_multi_module_mtp()` → `num_prefill_lookahead=num_spec_tokens`，其他 EAGLE → 1，非 EAGLE → 0 ✓。`get_kv_cache_configs` 对 `use_multi_module_mtp()` 把每个 `SlidingWindowSpec.extra_retained_tokens` 设为 `num_speculative_tokens - 1` ✓；`SlidingWindowManager.get_num_skipped_tokens = max(0, computed - sliding_window + 1 - extra_retained_tokens)` ✓。`allocate_slots` 的 `C = min(total_computed_tokens + num_new_tokens, request.num_tokens)`、`not enable_caching or delay_cache_blocks` 跳过登记 ✓。`test_eagle_swa_alignment_caches_extra_block` 存在于 `tests/v1/core/test_prefix_caching.py` ✓。 |
| **P1-7** 06 与 07 同一机制两份完整正文与图 | **NOT_FIXED（且被扩大）** | 06 的 diff 为 +44/−8，全在 batch queue / 默认 async 轴，未触及 §5.1 / §6.1 / §6.2。(i) stale/drop：06 §5.1 与 07 §8.3 仍各有完整正文 + 图；(ii) finished 信号表：06 §6.1 四行表与 07 §8.5 三行表并存，域 index 仍把该题分给 06；(iii) defer fence：**这一项可接受**（06 §6.2 拥有时序与图，07 §8.4 只留 gate 条件 + "已在 06 页重放"）。回归：07 新增 §8.1.2 又写了一遍 `finished_req_ids` 语义与"不能原地 `clear()`"，于是同一规则出现在 06 §6.1、07 §8.1.2、07 §8.5 三处。详见 §4.2 与 NEW-5。 |

### P2

| 编号 | 主题 | 状态 | 证据 |
|---|---|---|---|
| P2-1 | 07 §5.1 `next_decode_eligible_step` 未标 `use_v2_model_runner` 条件 | **PARTIAL** | 该段（07:385）未改动，仍写"`AsyncScheduler._update_after_schedule()` 在非 partial-prefill 请求提交后，把 `next_decode_eligible_step` 设为 `current_step + pp_size`"；源码 `async_scheduler.py` 只在 `if self.use_v2_model_runner:` 内设置。上一句（07:383）"V2 + PP + async 还检查 `next_decode_eligible_step`"提供了语境，但"设置"这一句本身仍无条件。 |
| P2-2 | 07 §7 / §10 路线 #3 的 DP 节流锚点不精确 | **NOT_FIXED** | 07:860 路线 3 仍只列 `vllm/v1/engine/core.py::EngineCore._should_throttle_prefills`（基类恒 False）；全页无 `SchedulerConfig.prefill_schedule_interval`、无 `DPEngineCoreProc._should_throttle_prefills`。 |
| P2-3 | 07 §7 watermark 是比例、例子未给换算 | **NOT_FIXED** | 07:492 仍只写"余量要求 2 块"，无 `int(watermark * kv_cache_config.num_blocks)`。 |
| P2-4 | 07 §5.2 `prefill_scheduled` 语义与 `break` | **NOT_FIXED** | 07:393 仍为"batch 尚无已排 prefill""容量不足以保留整个 `1+K` 时，本轮先不准入"。 |
| P2-5 | cascade 共同前缀块数生产侧无主 | **FIXED** | 07 §8.1.1 新增 `num_common_prefix_blocks` 行，并且**精度高于上轮建议**：写明"检查的是所有仍有已分配 KV 的请求…因而 scheduled 请求共享前缀时也可能返回 0"，与 `KVCacheManager.get_num_common_prefix_blocks` docstring（"ALL requests with allocated KV cache share it"，ref_cnt == len(req_to_blocks)，并明写 0 的 edge case）逐点一致；`FullAttentionManager` 实现 `ref_cnt == len(self.req_to_blocks)` ✓；07 §10 路线 19 已登记该 helper。 |
| P2-6 | 07 §8.4 abort-in-remote-wait 延迟释放是 Scheduler 规则 | **NOT_FIXED** | 07:～720 仍写"connector 要求 delay 时"，未出现 `finish_requests` 的 `delay_free_blocks = request_id not in finished_recving_kv_req_ids`，也未链接 22 §6.2。 |
| P2-7 | 07 §5.3 encoder budget 例"只有 4"未说明是剩余预算还是每步上限 | **NOT_FIXED** | 07:399 未改动。 |
| P2-8 | 07 §3 终态表 FINISHED_STOPPED resumable 分支 | **NOT_FIXED** | 终态表仍只写"随后立即复位到 `WAITING`"，未含 `WAITING_FOR_STREAMING_REQ`（§3.1 与状态表其他行已正确描述该状态）。 |
| P2-9 | 08 §2.3 "[[24…|LoRA]]" 标签误导 | **NOT_FIXED** | 08:95 链接与标签未变。 |
| P2-10 | 08 §5.3 未点名 `CacheConfig.prefix_match_unit` / `prefix_cache_retention_interval` | **NOT_FIXED** | 两个配置键在 08 全页 grep 无命中（仅出现泛称"retention 规则"）。 |
| P2-11 | 08 §6.3 路线缺正文引用的测试 | **PARTIAL** | `test_partial_prefix_cache_hits.py::test_internal_checkpoint_uses_partial_hash_lifecycle` 与 `test_single_type_kv_cache_manager.py::test_mamba_speculative_block_relocation_requires_exclusive_ownership` 现已在 §5.1.3 行内路线登记；`test_prefix_caching.py::test_cache_hit_local_and_external`、`test_deferred_block_free.py::test_cow_retentions_deferred_until_copy_step_processed` 在 08 仍无登记（后者已被 07 路线 18 登记）。 |
| P2-12 | 08 §4.6 调用树缺 `take_kv_cache_block_copies`/`take_boundary_state_offloads` | **NOT_FIXED** | 08 §4.6 的 `Scheduler.schedule` 树仍以"组装并返回 SchedulerOutput"结尾。对比：07 §10 树本轮**已补** `kv_cache_manager.take_boundary_state_offloads / take_kv_cache_block_copies`，两页树因此不对称。 |
| P2-13 | 08 §3.2 与 §5.1 "至多复用 `num_tokens−1`"重复 | **NOT_FIXED** | 08:239 与 08:511 两段仍各自完整陈述该规则。 |
| P2-14 | 入站引用漂移（16→07 §8.1、12→08 §5.1.2） | **PARTIAL** | 12→08 已改为指向新增的 §5.1.3 生命周期节（实质正确，锚点精度见 NEW-4）；16:432 仍写"四元组账本与 21/20/3 示例见 07 §8.1"，而四元组表在 §8.2（07:639），21/20/3 在 §8.1.3 尾部（07:634）。详见 §4.3。 |
| P2-15 | `update_from_output` 术语（06 "归并/结算" vs 07 "对账"） | **NOT_FIXED** | 06 §2.2 标题仍为"schedule 给出计划，然后才执行并归并结果"，07 统一用"结果对账"。仅措辞，不影响事实。 |

## 3. 新增 / 回归发现

> 新增段落里**未发现 P0 或 P1**。所有新增数值例都已按源码重放且成立（见 §2）。以下五项均为 P2，全部已对 checkout 核实。

**NEW-1（P2｜08 §3.3.1"回收型 spec 的 admission cap"）—— "只影响前置整段准入门"不成立，cap 有第二个消费者且无人点名**
- 引文：「它只影响**前置整段准入门**；窗口回收后的本步实际分配不传 True」。
- 源码：`apply_admission_cap=True` 在基线里有**两个**调用点。除 `vllm/v1/core/kv_cache_manager.py::KVCacheManager.allocate_slots` 的 `full_sequence_must_fit` 门（:483）之外，`vllm/v1/core/sched/scheduler.py::Scheduler._request_remaining_blocks`（:2860）也传 True，其结果经 `_inflight_prefill_reserved_blocks()` 汇总为异步 load 分配时使用的 `reserved_blocks`。页面自己在同节把"reserved blocks"划给 07，07 §7 只说"分配时考虑其它 in-flight prefill 尚需的容量"，两页都没有说明这项估算同样套 cap。
- 后果：读者会推出"reserved 估算按整段未截断长度"，与源码相反；"只"字把一个双消费者契约写成单消费者。
- 建议：把该句改为"影响两处：`allocate_slots` 的整段准入门，以及 `Scheduler._request_remaining_blocks` → `_inflight_prefill_reserved_blocks` 的在途 prefill 预留估算；本步实际分配不传 True"，并在源码行补 `Scheduler._request_remaining_blocks`。

**NEW-2（P2｜08 §4.5.1 HMA）—— 显式 `False` + 不支持 HMA 的 connector 的真实结局是启动硬失败，页面把它推给 22**
- 引文：「connector 的 `supports_hma_config` 检查位于 **None 分支**，不能声称显式 False 总在这里被 connector gate 拒绝；实际 connector 与 group 的后续兼容约束仍由 22 的协议合同负责。」
- 源码：前半句正确（`VllmConfig.__post_init__` 的 connector 检查确实只在 `is None` 分支）。但后半句把结局定位错了：`vllm/distributed/kv_transfer/kv_connector/factory.py::KVConnectorFactory.create_connector`（:54-60）计算 `hma_enabled = not disable_hybrid_kv_cache_manager`，并在 `hma_enabled and not cls.supports_hma_config(...)` 时 `raise ValueError("Connector … does not support HMA but HMA is enabled. Please set --disable-hybrid-kv-cache-manager.")` —— 这是启动期硬失败，属于本页拥有的"选择轴 + 失败边界"，不是 22 的传输协议问题。08 §6.3 路线 14 也只登记了 `supports_hma_config`，没有 `create_connector`。
- 建议：补一句"显式开启（False）时若 connector 不支持，`KVConnectorFactory.create_connector` 在建 connector 时 `ValueError` 直接失败"，并把 `create_connector` 加入路线 14；再把对 22 的委托限定为"跨 Engine 的 group/协议兼容"。

**NEW-3（P2｜07 §8.1.2 `has_sync_kv_loads` 行）—— 消费者范围写窄成 MRV2 专属**
- 引文：「MRV2 `ActiveKVConnector.pre_forward` 用它决定在 forward 前发起同步 load；否则 async load 可延至 post-forward」。
- 源码：MRV1 路径用同一规则。`vllm/v1/worker/kv_connector_model_runner_mixin.py:86`：`start_after_forward = not scheduler_output.has_sync_kv_loads`，`if not start_after_forward: kv_connector.start_load_kv(...)`，否则在 `finally` 里 post-forward 启动 —— 与 `vllm/v1/worker/gpu/kv_connector.py::ActiveKVConnector.pre_forward`（:69）语义相同。07 §10 路线 18 也只列了 MRV2 那个文件。
- 建议：消费者列写"11/12 两条 runner 路径"，路线 18 补 `kv_connector_model_runner_mixin`。

**NEW-4 —— 撤回（不是发现）：标签写子节、锚点落父节是本域既定体例**
- 现象：新增链接形如 `[[08_…#5.1 更细粒度复用：命中 6 个 token，为什么还要复制半个块|KV Cache §5.1.3]]`、`[[08_…#3.3 分配：把候选命中变成受保护的请求映射|KV Cache §3.3.1]]`（07 ×2、12 ×1、16 ×1），锚点指 H3、标签指 H4。
- 复核后撤回：全 vLLM 目录 `grep '\[\[[^]|]*#x.y.z'` **零命中**，`#x.y` 锚点 **26 处**。即本域**没有任何**链接锚到三级标题，父节锚 + 子节标签是既定体例。四条新链接与体例一致，`check_links --strict` 亦通过。除非整域统一改体例，否则不应作为本轮缺陷。

**NEW-5（P2｜07 §8.1.2 ↔ 07 §8.5 ↔ 06 §6.1）—— 回归：`finished_req_ids` 语义与"不能 `clear()`"规则新增第三份副本**
- 详见 §4.2。新写的 07 §8.1.2 把上轮 P1-7 已经指出的重复扩大成三处（其中两处同页）。三处内容互不矛盾且都与 `scheduler.py:1528-1531` 一致，属单一真相源违反，不是事实错误。
- 建议：生产条件留 07 §8.1.2，07 §8.5 该行改链接，四种 finished 对照表只留 06 §6.1。

**NEW-6（P2｜07 §8.1 结构）—— 插入的三个子节把 §8.1 原尾段孤立到 §8.1.3 之下**
- 详见 §4.4。§8.1.3 以"三张顶层表覆盖 9+10+3=22 个字段…本合同只承诺这些生产点和交接语义"收尾后，又接 `AsyncScheduler` placeholders 与 21/20/3 spec 回退两段原文，与该子节标题"异步补写与计划完成点"不符；并使 16:432 对"07 §8.1"的引用更难判断。

**NEW-7（P2｜08 §6.2 可观测边界表未随新机制扩展）**
- 08 本轮 +179 行新增五个机制单元，但 §6.2「从哪些边界判断正确性与运行状态」七行未变，没有登记任何新单元的可观测边界。按 feature-analysis 的体例，这张表应当能解释新机制在运行中被观察到时怎样读。
- 缺口举例：HMA 被自动关闭后"SWA 层不再按窗口释放、容量与性能都变化"；"SWA/chunked 组被 cap 截到 21 块而 full 组仍要 64 块，于是同一请求可能因 full 组超池被拒"；"startup plan 未应用（free 低于记录基线）→ 回落完整 profiling"；"`num_gpu_blocks_override` 生效时 `available_memory` 被改写，容量门与最终配置一致但不代表物理够用"。
- 建议：补 3–4 行。

**NEW-8（P2｜单向链接两处，低优先）**
- 07 §8.4（07:717）保留了 encoder input 释放规则的压缩版（「结果确实执行后才 `_free_encoder_inputs()`；确认进度是 computed 减 placeholders，还须越过媒体末端与 drafter lookahead」），而 15 §4.2 本轮成为该规则的长文 owner（含 `7 <= computed - 4` 的算例），07 侧**没有指向 15 §4.2 的链接**。两处不矛盾，只是缺回链。
- 22 全文用 "HMA" 20 余次（均为 connector 能力语境，不争 08 的配置开关归属），但**没有链接到 08 §4.5.1**；08 §4.5.1 已单向指向 22。建议补一条反链（与 P1-2 的原建议一致）。

**NEW-9（P2｜07 §10 路线 18）—— 两个测试只用描述指代，未给名字**
- 引文：「`test_make_scheduled_encoder_input_stats_output_embeddings` 及两项 disabled 测试验证统计口径/门控」。
- 事实：两项测试确实存在且对应（`tests/v1/core/test_scheduler.py::test_scheduled_encoder_input_stats_disabled_without_iteration_logging`、`::test_scheduled_encoder_input_stats_disabled_without_log_stats`），但路线的体例是点名锚点；此处退化为描述，读者无法按名定位。
- 建议：直接写出两个测试名。

### 观察（非发现）：08 §3.1.3 是本轮未被要求的范围扩张，但内容正确且边界已声明

08 新增 `#### 3.1.3 GPU profiling → KV 字节预算 → 全 worker 共同块数`（约 50 行 + 一幅图），不在上轮任一发现的修复范围内。逐点核验**全部成立**：

- 公式：`D_requested = ceil(T·u)` = `request_memory` 的 `math.ceil(total_memory * gpu_memory_utilization)` ✓；`D_consumed = F₀−F₁` = `memory_profiling` 的 `total_consumed = before_create.free_memory - after_profile.free_memory` ✓；`D_transient = P−A` = `transient_peak_headroom = after_profile.torch_peak - after_profile.torch_allocated` ✓；`D_nonKV = D_consumed + D_transient` = `non_kv_cache_memory = total_consumed + transient_peak_headroom`（`mem_utils.py:326`）✓；`D_KV = D_requested − D_nonKV − G − M` = `available_kv_cache_memory_bytes = requested_memory - non_kv_cache_memory - cudagraph_memory_estimate_applied`，随后 `reserve_mm_ipc_gpu_memory` 扣 M ✓。
- `request_memory` 的 free<requested → `ValueError` ✓；非 ROCm fallback 时 `assert init_free_memory >= free_gpu_memory` ✓；`maybe_rocm_profiling_fallback` 走替代计数并改写 `total_consumed`/`non_kv_cache_memory` ✓；init snapshot 在 `init_worker_distributed_environment` 之后取（源码注释"Now take memory snapshot after NCCL is initialized"）✓。
- `current_platform.is_cuda_alike() and cudagraph_mode != NONE` 才 `profile_cudagraph_memory()`，`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 默认 **1**（`envs.py:2139`）控制是否计入，关闭则 G=0，注释明写 XPU 排除 ✓。
- `reserve_mm_ipc_gpu_memory`：`mm_config is None` 原样返回 ✓；帧池预算"divided among API processes, so it is not multiplied"、GPU video backend 的 decoder/context 按 `api_process_count` 放大 ✓；不剩容量 `ValueError` ✓。
- 旁路：显式 `kv_cache_memory_bytes` 仍跑 `profile_run()`、跳过 `memory_profiling` 与 graph 估计、不按 utilization 重算 ✓；`VLLM_ENABLE_STARTUP_PLAN` 默认 **0**（`envs.py:1917`）✓；fingerprint = schema + vllm 版本 + `vllm_config.compute_hash()` + device name/total memory/capability + torch + cuda + rank + world_size ✓，docstring 明写 driver-only 变化不在 key ✓；apply 门 = schema/fingerprint 匹配 + `kv_bytes > 0` + `current_free_memory >= free_memory_baseline` ✓；`compile_or_warm_up_model`（:757）以 `redundancy_buffer_memory = 150 * (1<<20)` 算建议值并 `maybe_save_startup_plan(self, kv_cache_memory_bytes_to_requested_limit)`，`os.replace` 原子写 ✓。
- 块数：`get_kv_cache_configs` 的 `check_memory` 按 `avail_mem - _pool_bytes_per_block(groups)` 预留 null（注释"Allocation below still uses the full memory"）✓；`num_gpu_blocks_override` 改写 `available_memory = override * bytes_per_block` ✓；跨 worker `min_num_blocks` 后对较大者以 `min_num_blocks * _pool_bytes_per_block(groups)` **重新生成布局**（注释"strides and offsets stay consistent"）✓；auto-fit 缩短 `max_model_len` 后 Core 广播（`core.py:325-329`）✓；attention-free → 空 group + `num_blocks=1`（注释"BlockPool always needs a null_block"）✓；`VLLM_ELASTIC_EP_SCALE_UP_LAUNCH` 用已取得的 `available_gpu_memory_for_kv_cache` 不重新 profile（`core.py:298-304`）✓。
- **手算重放**：T=80 GiB、u=0.9 → 72；F₀−F₁=76−50=26；P−A=8−3=5；nonKV=31；KV=72−31−2−1=**38 GiB** ✓。38 GiB = 39,845,888 KiB，÷640 KiB = 62,259.2 → **62,259**，扣 null → **62,258** ✓。图内"另一 worker 60,000 → 共同 60,000、非 null 59,999" ✓。
- 图两幅（§3.1.3 预算转换、§5.1.3 状态表快照）都带 `<!-- Figure spec -->`、走本页既有 normal/blue/orange classDef、边有标注、无引号管道标签；`check_markdown --changed` 无新增 MD003。

范围上它声明了"profile 中的编译、graph capture 生命周期继续由 19 展开""未外推 CPU/TPU 等 worker"，边界清楚，不构成与 19 的第二真相源。保留即可；若要收紧，可把 startup-plan 那段压成一段并把 fingerprint 细节交给 19/26。

## 4. 跨页交接与重复（06 / 11 / 12 / 15 / 16 / 22）

### 4.0 跨页新发现（两项落在 12，一项落在 08 自身）

**X-1（P1｜12 §? `MambaHybridModelState` 行，12:229）—— 与 08 §5.1.3 对"align 缓存什么"的规则矛盾**
- 12:229 引文（**该括号是本轮之前就有的旧文**，本轮只改了行尾的委托链接）：「`mamba_cache_mode='align'`（开 prefix caching 时的默认，**只缓存落在块边界上的步末 Mamba 状态**）」。
- 08 §5.1.3（新 owner，12 现在正指向它）导出的是 **state@112**，而 Mamba block=32 的块边界是 96 / 128 —— 112 是 **hash 边界**，不是块边界。
- 源码站在 08 一侧：`MambaManager._cache_partial_tail_block`（`single_type_kv_cache_manager.py:1959`）的两条分支都产生非块边界 hash —— checkpoint 分支用 `checkpoint_position = (num_tokens-1)//hash_block_size*hash_block_size` 并 `replace_existing_hashes=True`；partial-tail 分支的前置条件之一正是 `if num_tokens % self.block_size == 0: return None`（**必须不在块边界**）加 `num_tokens % hash_block_size == 0`。回归测试 `test_internal_checkpoint_uses_partial_hash_lifecycle` 断言 `checkpoint_block.block_hash_num_tokens == 112` 且 96 的 full hash 已不可命中。
- 因此 12 的"只缓存落在块边界上的"仅在"无内部 checkpoint 且块长等于 hash 粒度"的配置下成立，而它委托过去的 §5.1/§5.1.3 讲的恰恰是另一种配置。
- 建议（归 12 的作者）：改为"缓存步末状态；块长大于 hash 粒度或启用内部 checkpoint 时还可在 hash 边界登记 checkpoint/partial tail，细则见 08 §5.1.3"，或删掉括号只留链接。

**X-2（P1｜12:608 表行）—— `max_concurrent_batches` 与 06 §4.1（本轮新声明的 owner）及源码不一致**
- 12:608 引文：「`VllmConfig.max_concurrent_batches` | 派生 property | **异步 MRV2 为 PP size + 1，否则为 PP size** |」（该行本轮未改）。
- 源码 `vllm/config/vllm.py::VllmConfig.max_concurrent_batches`（:570-577）：async + V2 → `pp_size + 1`；async + **非** V2 且 `pp_size <= 1` → **2**；其余 → `pp_size`。
- 06 §4.1 本轮新增的容量表正确覆盖了这一档：「| async=True，MRV1，p=1 | **2** |」，11:46 也写"容量 2"。12 的"否则为 PP size"在该档给出 1。
- 12 在别处（12:127 新增行）已写"完整选择轴归 06 §4.1"，所以该行应改正或删除。
- 建议（归 12 的作者）：改为"async+MRV2 为 p+1；async+MRV1 且 p≤1 为 2；其余为 p（完整表见 06 §4.1）"。

**X-3（P2｜08 §5.1.3 ↔ 07 §5.4）—— P1-5 的修复又造出一份 `is_mamba_prefill_checkpoint_valid()` 的完整解释**
- 07 §5.4（07:436，**旧文**）：「这必须通过共用 `is_mamba_prefill_checkpoint_valid()`：起点 hash 对齐、checkpoint 严格在 query 内、离起点至少一个 hash block、相对起点满足 backend alignment，而且 checkpoint 列必须在 initial-state 列之后。」并自带算例（start=0、end=100、hash=8、block=64、alignment=16 → 96 有效、88 无效）。
- 08 §5.1.3（08:606，**本轮新写**）：「backend 必须提供 `prefill_checkpoint_alignment`，query 起点须 hash 对齐，候选须严格位于 query 内、至少离起点一个 hash 单元，满足相对 backend 对齐，并且 checkpoint 列晚于 initial-state 列。」并自带另一个算例（hash16、block32、120 → 112）。
- 两份都与源码 `is_mamba_prefill_checkpoint_valid`（`kv_cache_interface.py:972-996`）一致，**不是事实错误**；问题是 08 自己两处声明了相反的归属：§5.3（08:686）「Mamba checkpoint 的**调度对齐和 padding 演算见 07**」，§5.1.3 结尾「**07 负责挑选可执行的 checkpoint 对齐终点**，11/12 负责实际状态导出，本页拥有它们对应的容量、位置表、身份和归还规则」。按 08 自己的划分，谓词该留在 07。
- 建议：08 §5.1.3 把条件清单压成"能否导出由 07 §5.4 的 `is_mamba_prefill_checkpoint_valid()` 判定"，只保留本页真正拥有的部分（预留列是否为空/可替换 scratch、准入探测不写 `_checkpoint_positions`、`replace_existing_hashes` 的重键与 owner 保持）。两个算例保留一处即可。

### 4.1 交接现在一致的部分

| 交接 | 结论 | 证据 |
|---|---|---|
| (a) admission cap / (b) HMA / (c) hash 合同 / (e) 可缓存尾部是否被二次解释 | **没有重复**。对 06 / 07 / 16 三页 grep `max_admission_blocks`、`max_in_flight_tokens`、`disable_hybrid_kv_cache_manager`、`HMA`、`hash_block_tokens`、`NONE_HASH`、`prefix_caching_hash_algo`、`num_reprefillable`、`extra_retained_tokens`、`unify_hybrid`：**三页零命中**；全域这些标识符只出现在 08（13 行）与 15 的一句边界声明、22 的 connector 能力语境。四项均由 08 独家展开，他页只留一行链接（07:494、16:394）。 | grep 全域 |
| (d) Mamba align | **部分重复**：state block 容量/位置表/身份/归还由 08 §5.1.3 独家拥有，但 `is_mamba_prefill_checkpoint_valid()` 的条件清单与算例在 07 §5.4 与 08 §5.1.3 各有一份 —— 见 X-3。 | 07:436、08:606、08:686 |
| 11 ↔ 07 字段数 | 一致。11 §3.4（11:525）「**输入：`SchedulerOutput` 的 22 个字段。** 按 dataclass 声明逐项核对…"MRV1 不读"指这些文件里没有读取点」与 07 §8.1 的 22 / 9+10+3 完全对齐；抽查的 `kv_connector_block_state`（不到 worker）、`scheduled_encoder_input_stats`（MRV1 不读）两项与 07 §8.1.1/§8.1.2 不矛盾。这是本轮最干净的新交接。 | 11:525、07:569-633 |
| 16 ↔ 08 可缓存尾部（原 P1-6 的另一半） | **已解环**。16:394 改为"已确定的尾 token 何时有 slot、何时能登记 block hash 是另一条资源合同，归 …KV Cache §3.3.3"；08 §3.3.3 拥有规则，08 Related Pages 对 16 的描述同步改为"draft 生成与 rejection；由本页 §3.3.3 闭合对应的 cache tail 登记边界"。两侧不再互指。 | 16:394、08:130、08:800 |
| 12 ↔ 08 Mamba state block（原 P2-14 后半） | **已修正到正确 owner**。12:229 由旧的"08 §5.1.2"改为 `…#5.1 更细粒度复用…|KV Cache §5.1.3`，指向新增的 §5.1.3 生命周期节；实质正确（锚点精度见 NEW-4）。 | 12:229 |
| 16 §6.3 EAGLE group 标注 / `use_eagle_block_drop` ↔ 08 | 一致。16:361 明确把命中层面算法归 08、自己只讲 group 身份判定；与 08 §3.3.3 的 EAGLE 分支不冲突（一个讲"哪些 group 被标注"，一个讲"被标注的 group 能登记到哪"）。 | 16:361、08:304-320 |
| 15 ↔ 08 hash extra keys | 一致且已闭合：15 原本把"extra key 组合与 `hash_block_tokens`"委托给 08，08 §2.3 本轮补齐生成合同与 `generate_block_hash_extra_keys` 路线。 | 08:77-95 |
| 22 ↔ 08 partial tail / boundary offload | 一致。08 §5.1.2 仍只讲 pool 侧身份与 pin，网络协议归 22；新增 §5.1.3 第 4 点"请求清理"与 `pop_blocks_for_free` 丢弃未交出 offer 的描述与 22 的 producer/consumer 语义不矛盾。 | 08:548-575 |

### 4.2 P1-7 复评：**NOT_FIXED，且本轮新增了第三份副本**

06 未收敛。`git diff` 对 06 是 +44/−8，全部集中在 batch queue / 默认 async 轴（§4.2 的两批在途重放），**没有触及 §5.1 / §6.1 / §6.2**：

- (i) stale / drop 计数：06 `### 5.1 普通抢占后的旧输出仍可交付，但不能重复结算重置计数`（06:193）仍是完整正文 + 图3；07 `### 8.3 三种迟到/失败结果，不能都叫"丢弃 stale"`（07:673）仍是完整五行表 + 图7。两页各有完整解释与图，内容一致但构成两个真相源。
- (ii) finished 信号表：06 `### 6.1 四种"finished"分别通知谁`（06:226）仍是四行完整表；07 §8.5 仍是三行完整表（`finish_reason` / `SchedulerOutput.finished_req_ids` / `EngineCoreOutputs.finished_requests`）。域 index 第 25 行仍把"两批在途工作怎样配对并释放资源"分给 06。
- (iii) defer fence：这一项**可接受**。06 `### 6.2 两批在途时的 block fence`（06:237）拥有时序与图4；07 §8.4 只保留 gate 条件一句并明写"具体 Core 队列与 fence 时序已在 06 页重放"。owner 清楚。

**新增的第三份副本（回归）**：07 新写的 §8.1.2 又解释了一遍 `finished_req_ids` 的语义和"换新 set"规则——
- 07 §8.1.2 表行：「`finished_req_ids` | `_free_request` 等生命周期路径积累的 finished 集合直接放进本步 output；`_update_after_schedule` 为下一步换新 set | worker 删除旧镜像；集合非空不证明物理 KV 已回池，也不等于 `EngineCoreOutputs.finished_requests`」
- 07 §8.1.2 收尾段：「清理集合必须"发布旧对象、内部换新对象"，不能调用原 set 的 `clear()`，否则 Core 留存的旧计划会丢失清理通知。」
- 06 §6.1 收尾段（**先于本轮存在**）：「schedule 将旧 finished/preempted set 放进计划后换成一个**新 set**，不能对原 set 原地 `clear()`，否则已经交给 Executor 的计划也会被清空。」
- 07 §8.5 表行（**先于本轮存在**）：「`SchedulerOutput.finished_req_ids` | worker 清除持久 batch 中的旧请求镜像，下一次计划带出该差量 | 否」

于是 `finished_req_ids` 的消费者与"不能 `clear()`"规则现在有 **三处**（06 §6.1、07 §8.1.2、07 §8.5），其中两处在同一页。三处互不矛盾（都与 `scheduler.py:1530-1531` 的 `self.finished_req_ids = set()` 及其 NOTE 一致），但这是 P1-7 要解决的问题被扩大而不是收敛。

- 建议：`finished_req_ids` 的生产条件留在 07 §8.1.2（07 拥有 `schedule`/`_update_after_schedule`），07 §8.5 的该行改为一句链接；四种 finished 信号的"给谁看"对照表只留 06 §6.1；stale/drop 计数规则留 07 §8.3，06 §5.1 收成"为何必须用原计划配对"一段 + 链接（图保留一处即可）。

### 4.3 仍未修的入站引用漂移（P2-14 前半）：**NOT_FIXED**

16:432：「完整的四元组账本与 21/20/3 的逐步示例见 [[…07_vllm_scheduler_analysis|Scheduler]] **§8.1**，本页不再另算一遍」。
- 21/20/3 的逐步示例本轮**仍在 §8.1 之下**（07:634，§8.1.3 尾部）—— 这一半对。
- 但"四元组账本"那张表在 **§8.2**（07:639：「用四元组 **已知 token 数 / computed / placeholders / in-flight** 记 Scheduler 状态」）。16 用一个 §8.1 指两样东西，其中一样在别的小节。
- 该引用是裸页链接 + 手写节号，没有 anchor，`check_links` 无法发现。
- 建议：改为"21/20/3 示例见 07 §8.1.3，四元组账本见 07 §8.2"。

另：16 的表格在同一行里混用本页节号与 07 节号（16:68/16:70 的"§8.1"/"§8.2"列指 16 自己，同行正文里的"07 §8"指 07），属既有体例问题，本轮未变。

### 4.4 新增结构问题（P2）：§8.1 的原尾段被插入的三个子节"孤立"到 §8.1.3 之下

07 把 §8.1.1–§8.1.3 插在 §8.1 开头正文与原有尾段之间。结果 §8.1.3（标题："异步补写与计划完成点"）先以一段总结收尾（「三张顶层表覆盖 9+10+3=22 个字段。发布完成点是：… 本合同只承诺这些生产点和交接语义。」），随后又接了两段与字段合同无关的原文：`AsyncScheduler` 的 placeholders 记账，以及 21/20/3 的 spec 回退重放（末句"见下一小节"）。读者会在一个讲"字段补写"的小节里读到 async placeholder 与 rejection 算术。
- 建议：把这两段移到 §8.1 开头正文之后（子节之前），或另起 `#### 8.1.4 async placeholders 与一次 spec 回退`；同时 16:432 的引用可随之精确化。

### 4.5 未发现的问题（明确说明）

- 未发现 07/08 与 06、11、12、15、16、22 之间新的**事实矛盾**。抢占 victim、running-first、waiting 门、async placeholder、stale 赋值、defer gate、free queue 次序、两阶段 touch、fixed-point 命中、CoW 两端 retention、CPU tier pending/ready、`num_invalid_spec_tokens` 的统计口径在各页仍一致。
- 本轮新增的五个 08 单元没有在他页产生第二真相源（见 §4.1 第一行）。
- 08/07 两页的全部数值例（含新增的 21/20 块 cap、C=70/R=2 的 68/64/68、Mamba 1600 三步表、hash16/block32/120 的 112 替换 96、38 GiB → 62,259 → 62,258）均已按源码或回归测试重放成立。

## 5. Feature / rubric 复评

```
feature: pass 07 §8.1.1 执行集合与输入差量
feature: pass 07 §8.1.2 生命周期与资源工作
feature: pass 07 §8.1.3 异步补写与计划完成点
feature: pass 08 §3.1.3 GPU profiling → KV 字节预算 → 全 worker 共同块数
feature: pass 08 §3.3.1 回收型 spec 的 admission cap
feature: pass 08 §3.3.3 投机尾部的三条边界
feature: pass 08 §4.5.1 HMA 开关
feature: pass 08 §5.1.3 Mamba align 正常生命周期
feature: pass 08 §2.3 prefix hash 生成合同
```

判定依据：每个新单元都给出目的（为什么存在 / 避免什么错误）、输入输出契约、处理逻辑（含分支与非直觉分支）、边界约束（失败、断言、不适用范围）、支持范围（哪些 spec/平台/runner 覆盖，哪些不承诺），并附源码路线与回归入口；兄弟选择轴均已点名或指向 owner。§8.1.x 的字段枚举可按 dataclass 逐字段对账且 22/22 齐备，是本轮最强的一项。
