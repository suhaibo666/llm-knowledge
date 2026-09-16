# Round-4 independent verification — vLLM 17/18/20/21/22/23/24/25/26 + index + domain view

- **Role**: independent reviewer (fourth round). **Findings only — no file edited** (`wiki/`, `docs/`, `skills/`, source checkout all untouched).
- **Frozen source**: `/Users/suhaibo/97-llm/vllm`, `git rev-parse HEAD` = `199cb9b964822e59ab9b58d88e7be31eb419a2ae` at start and end; HEAD not moved, no fetch/checkout (only untracked `artifacts/`).
- **Object**: uncommitted working tree (`git diff -- <page>`). vLLM dir diffstat: 27 files, **+2108 / −735** (round-3 measured +2104/−729 over 30 files), i.e. the final pass is small and surgical.
- **Method**: each round-3 item re-judged **against the source, not against the new prose**; every sentence the diff added or rewrote on my pages re-verified; mechanical extraction of all anchored wikilinks and cross-page `§` refs; index coverage-boundary table audited row by row at both ends; AST-level class-membership checks for every qualified symbol in the new text.
- **Run boundary**: static source reading + T0 checkers. No vLLM, GPU, NIXL/Mooncake/MoRIIO/ZMQ/OTLP service, no fault injection, no site build (coordinator owns that).

## 0. Mechanical gates (re-run read-only, `.venv` py3.13)

| Gate | Result |
|---|---|
| `check_links.py --strict` | 453 pages; broken / ambiguous / bare_index / stale_section / **orphans all 0** |
| `check_math.py --changed --strict` | 64 files, **0 error / 0 warning** |
| `check_markdown.py --changed --strict` | 64 files, **0 error / 0 warning** |
| `check_assets.py --changed --strict` | 64 files, **0 error / 0 warning** |

---

## 1. Verdict rows

```
17_vllm_quantization_analysis      : round3 items 1 FIXED / 1 NOT_FIXED | new text pass | VERDICT PASS (1 P2)
18_vllm_distributed_inference      : round3 P1 FIXED (source-verified) / 1 P2 NOT_FIXED | new text pass | VERDICT PASS (1 P2, 1 P3)
20_vllm_fused_ops_and_kernels      : 2 FIXED / 4 NOT_FIXED (all P2) | new text pass | VERDICT PASS (4 P2)
21_vllm_ir_and_fusion_passes       : no round3 items | 7 new claims re-sourced, all correct | VERDICT PASS
22_vllm_disaggregated_kv_serving   : 3 FIXED (incl. round3 P1-A) / 1 NOT_FIXED (P2) | new text pass | VERDICT PASS (1 P2)
23_vllm_observability_reliability  : 4 FIXED, all source-exact | new text pass | VERDICT PASS
24_vllm_extension_plugin_system    : 3 FIXED, all source-exact | new text pass | VERDICT PASS
25_vllm_weight_transfer_online_upd : 2 FIXED | **1 NEW P1** (§13 route class attribution) | VERDICT CONDITIONAL PASS — blocker R4-1
26_vllm_multiproc_executor_rpc     : 2 FIXED / 1 NOT_FIXED (P2) | new text pass | VERDICT PASS (1 P2)
index.md                           : round3 P1-B + 2 P2 rows FIXED | **1 NEW P1** (dangling `03 §9`) | VERDICT CONDITIONAL PASS — blocker R4-2
domain view                        : 34/34 anchored links resolve; 82/82 cross-page §refs resolve; index table honest at both ends | VERDICT CONDITIONAL PASS
```

**Counts**: FIXED **17**, PARTIAL **0**, NOT_FIXED **7** (all P2/P3). New **P0 = 0**, new **P1 = 2**, new **P2 = 1**.

---

## 2. Round-3 items, status per item

### 24 — extension / plugin system

| Item | Status | Source evidence |
|---|---|---|
| N-C §2 exception type | **FIXED** | L67 now: "`AttributeError` 会被就地吞掉并改抛 `NotImplementedError(f"Method {method!r} is not implemented.")`——排错时按后者的消息找". `vllm/v1/serial_utils.py::run_method` @486: `try: func = getattr(obj, method) except AttributeError: raise NotImplementedError(f"Method {method!r} is not implemented.") from None` — verbatim match |
| N-D pooling/spec + explicit `--logits-processors` `ValueError` | **FIXED** | §3.1 gating column L127 now "**无显式 `--logits-processors` 时**：pooling 返回空集合、speculative 只保留 `MinTokensLogitsProcessor`；配了显式项则这两支各自 `raise ValueError`". New §6.1 row L331. `logits_processor/__init__.py::build_logitsprocs` @185-210: pooling branch `if custom_logitsprocs: raise ValueError(STR_POOLING_REJECTS_LOGITSPROCS)`; spec branch `raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)` — two distinct message constants, exactly as the page says |
| N-A MRV2 example (coordinator-verified) — **surrounding sentences re-checked** | **FIXED, surroundings sound** | L129's replacement example (MRV1 + speculative) holds at both ends: frontend `SamplingParams.verify` @793 calls `_validate_logits_processors` **unconditionally** → `validate_logits_processors_parameters` @224 → `cached_load_custom_logitsprocs(logits_processors)` even for `None` → `_load_custom_logitsprocs` @159 whose only early return is TPU → `_load_logitsprocs_plugins()` enumerates entry points. Runner side: spec branch returns before that call. The added caution "不能拿 MRV2 举例" is correct: `_get_v2_model_runner_unsupported_features` @2547 appends `"custom logits processors"` (`vllm/config/vllm.py:2599-2608`), so auto-selection falls back to MRV1 and forced `VLLM_USE_V2_MODEL_RUNNER=1` hits `_validate_v2_model_runner` @2779 `raise ValueError("Model Runner V2 does not yet support: …")`. §6.1 row L329's parenthetical "此时进程必定是 MRV1" also holds in both sub-cases (installed entry point **or** explicit `--logits-processors` each block MRV2) |

### 17 — quantization

| Item | Status | Evidence |
|---|---|---|
| N-E §6.1 second full explanation of 09 §2.8's mechanism | **FIXED** | L408 compressed to one conclusion + pointer: "…它对**凡是挂着 `quant_method` 的模块无条件豁免**——普通 linear 权重并不是"可能豁免"而是必然豁免；这条判定的机制与范围归 [[09]] §2.8，本页不重述。" The `QuantizeMethodBase` / `UnquantizedLinearMethod` chain no longer appears here. Symbol check: `DefaultModelLoader.track_weights_loading` exists (`default_loader.py:445`), gate `enable_weights_track` @436-444 ✓ |
| NEW-P2-b §3.3 anchor/label mismatch | **NOT_FIXED** | L193 still `[[…09_vllm_model_library_analysis#2.6 融合前分别分片，融合后仍能拆回各投影\|模型库 §2.6.1]]`. 09's real headings: `### 2.6 …` @182 and `#### 2.6.1 MoE 专家写入…` @217. Anchor resolves (so `check_links` passes) but lands one subsection above the label's promise |
| Line-anchor drift check | — | 17's round-3 line anchors 191/193/247/249/530/563/566/608 all still hold, so the final pass touched only L408 on this page |

### 18 — distributed inference

| Item | Status | Evidence |
|---|---|---|
| Round-3 #6 / N-B: `worker_type="new"` attribution (coordinator-verified) — **surroundings re-checked** | **FIXED, source-exact** | L302 now: "`worker_type="new"` 由新加入 engine 自己的 `DPEngineCoreProc._eep_scale_up_before_kv_init` 在 KV 初始化之前产生——基类 `EngineCore` 上的同名方法只是 `raise NotImplementedError`，实体同样只在 `DPEngineCoreProc`". AST: `EngineCore._eep_scale_up_before_kv_init` @1023 body is `raise NotImplementedError`; `DPEngineCoreProc._eep_scale_up_before_kv_init` @2397 builds `ElasticEPScalingState(..., worker_type="new", scale_type="scale_up", reconfig_request=None)` then `run_pre_kv_init_states()`; call site `EngineCore.__init__` @141-142 under `VLLM_ELASTIC_EP_SCALE_UP_LAUNCH`, before `_initialize_kv_caches` @145. §13 route L586 also fixed: `DPEngineCoreProc._eep_scale_up_before_kv_init`（基类…只是 `raise NotImplementedError`）. The added consequence clause ("下文的 existing scale-up 走法不能套到新 engine 那一支上——新 engine 根本没有旧 dp_group 要退役") is a correct reading of @2397 |
| NEW-P2-b spread to 18 | **NOT_FIXED** | L352 uses the same `#2.6 …\|模型库 §2.6.1` pairing |
| Line-anchor drift check | — | 18's round-3 anchors 84/90/118/122/302/304/352/586 all still hold → the final pass on 18 was confined to L302 |
| Header date | **P3** | `最近更新：2026-09-15` while the pass edited on 09-16. 20 was bumped to 09-16; 18/21/22/23/24/25/26 were not. Either bump all or state the domain-wide date convention |

### 20 — fused ops and kernels

| Item | Status | Evidence |
|---|---|---|
| N-F Helion "no reference point" | **FIXED** | L271 replaced the "全库…没有任何引用点" claim with the criterion round 3 asked for: "**判据是产品侧没接线**——`vllm/kernels/__init__.py` 只 `from . import aiter_ops, oink_ops, vllm_c`，不导入 `helion`，也没有任何平台把它们写进默认 priority"，and it now acknowledges `tests/kernels/helion/` and the in-package `register.py` / `config_manager.py`. `vllm/kernels/__init__.py` is verbatim `from . import aiter_ops, oink_ops, vllm_c` with `__all__ = ["vllm_c", "aiter_ops", "oink_ops"]` ✓ |
| N-G header date | **FIXED** | `最近更新：2026-09-16` |
| N5 §5.2 HIP in-place wording | **NOT_FIXED** | L417 still: "**完成点**：CUDA 的设备路径与**直接复用它的 HIP 分支**、CPU 路径原地旋转并返回同一对张量". Source `rotary_embedding/base.py::RotaryEmbedding.forward_hip` @254-271: `if self.use_aiter: self.rocm_aiter_triton_rotary_embedding(...); return query, key` — a *distinct* implementation that is also in-place; only the `use_aiter == False` path does `return self.forward_cuda(...)`. The conclusion (in-place) survives; the attribution ("directly reuses it") does not. **Fix**: "HIP 有两支：`use_aiter` 走 `rocm_aiter_triton_rotary_embedding` 后返回同一对张量，否则回落 `forward_cuda`；两支都是原地" |
| N6 §5.3 figure `pairOffset` edge label | **NOT_FIXED** | L473 still `NX -->\|pairOffset = 4 / 2\| SH`, readable as "4 或 2"; the figure spec itself (L460) says `pairOffset=2`. **Fix**: `pairOffset = (8/2)/2 = 2` or just `pairOffset = 2` |
| N7 both new figures' palette | **NOT_FIXED** | §5.3 (L478-481) and §9.1 both define only `classDef compute fill:#e8f1fb,stroke:#416b91,…` / `cost fill:#fff1dd,stroke:#ab702c,…` and class 2–3 nodes; all remaining nodes carry no class and render in Mermaid's default lavender. The same page's other figures use `classDef neutral fill:#ffffff,stroke:#64748b,color:#0f172a` + `acc1 fill:#dbeafe,stroke:#2563eb`. `skills/drawing-wiki-figures/SKILL.md:66` — "Use `.neutral` (white fill, gray border) for most nodes" — and its checklist line 181 both ask for this. The diff shows these classDefs were never touched by the final pass |
| P2-3 residual: masked path's `silu_with_clamp` name branch | **NOT_FIXED** | 20 has 0 hits for `silu_with_clamp`; `fused_moe/activation.py:261` still unmentioned. Very minor |

### 21 — IR and fusion passes (no round-3 items; new text re-sourced)

All seven new/rewritten claims verified against the frozen source:

- "该比较本身在 `UnsafeCloneEliminationPass.__call__` 内（用 `node_to_idx`…），`user_writes_to_node` 只是谓词" ✓ — `clone_elimination.py`: `user_writes_to_node` module-level @39; `UnsafeCloneEliminationPass` @72, `__call__` @88 builds `node_to_idx` @90 and compares @117-124.
- "`dump_prefix` 只在 pass 实际执行后 `+= 1`；被 range 跳过的 pass 不占序号" ✓ — `passes/pass_manager.py:113-119`: `if pass_.is_applicable_for_range(compile_range): pass_(graph); VllmInductorPass.dump_prefix += 1 else: logger.debug("Skipping …")`.
- `RocmAiterRMSNormQuantFusionPass` / `RocmAiterSiluMulFp8GroupQuantFusionPass` gated on `rocm_aiter_ops.is_enabled() or rocm_aiter_ops.is_rdna_aiter_enabled()`, "不是平台为 ROCm 即加" ✓ — `passes/pass_manager.py:21` (import guard) and @189-191.
- "`QKNormRoPEFusionPass` 并没有 `is_applicable_for_range` 覆写" ✓ — 0 hits in `passes/fusion/qk_norm_rope_fusion.py`.
- "`rms_quant_fusion.py` 中没有任何 NVFP4 replacement…只有 FP8 侧的四个 `_C` 入口" ✓ — `FUSED_OPS` holds exactly four distinct ops (`rms_norm_static_fp8_quant`, `fused_add_rms_norm_static_fp8_quant`, `rms_norm_dynamic_per_token_quant`, `rms_norm_per_block_quant`), no NVFP4 key; `kNvfp4Dynamic` appears only in `QUANT_OPS`, so no NVFP4 pattern can register.
- `PassConfig` **16/16** fields, `CompilationConfig` **36** fields ✓ (AST count).

### 22 — disaggregated KV serving

| Item | Status | Evidence |
|---|---|---|
| Round-3 P1-A §14.4 L1015 "P/D 实例拓扑 → 13" (coordinator-verified) — **surroundings re-checked** | **FIXED and consistent at every occurrence** | L1015 now "…→ **本域暂无 owner，已提交规划**（与本表末行、§2、§14 及 Related Pages 口径一致；13 拥有的是 DP 副本内的路由，不是 P/D 之间）"; L1025, §2 ("本域目前没有 owner…已提交 `planning-codebase-analysis` 裁决（§14.4 末行同样登记）"), §14「它不是什么」and Related Pages L204 all agree. Page 13 still has **0** hits for `kv_transfer_params` / `do_remote_decode` / P/D, so the ownerless claim is true |
| Round-2 N1 HMA condition | **FIXED** | L134: "`_is_hma_required=True` 需要**两个条件同时成立**：`not scheduler_config.disable_hybrid_kv_cache_manager`，**且** cache groups 中存在非 `FullAttentionSpec`。也就是说显式关掉 hybrid manager 就退回下面那条普通的 invalid-block 合同". `nixl/base_worker.py:387-393` is exactly that conjunction ✓ |
| Round-2 N2 N→N−1 scope | **FIXED at all three occurrences** | L134 "**仅当 `num_computed_tokens == num_tokens` 即整段 prompt 全命中时**…非全命中时进度不动"; L299 "再在整段 prompt 全命中时把已承诺进度 N 改成 N−1 继续（非全命中不减）"; figure node Y "全命中时才 N → N-1". `Scheduler._update_waiting_for_remote_kv` @2870, else-branch @2903-2909: `if request.num_computed_tokens == request.num_tokens: request.num_computed_tokens = request.num_tokens - 1`, with the source comment "on a full prompt hit, we need to re-compute the last token" ✓ |
| Round-2 N10 page-level "→ **23**" | **NOT_FIXED** | moved from L854 to **L1019**: `… lease expiry、invalid blocks 的信号及观测边界与故障注入 → **23**`, while L1012's env row (L135 of the diff) and Related Pages use anchored `23 §5.2` / `23 §5.3`. One-cell precision fix |

### 23 — observability and reliability

| Item | Status | Evidence |
|---|---|---|
| Round-2 N5 `deferred` / `skipped_waiting` attribution (round-3's highest-value unfixed item) | **FIXED, and stronger than asked** | §6.1 L341 now: "**P/D consumer 的异步远端 KV load 不是 `deferred` 之外的另一种原因，它本身就被计入 `deferred`**…`num_requests_waiting_by_reason` 的 `deferred` 标签正是由 `len(self.skipped_waiting)` 供数…因此该指标的两个标签是**互斥且穷尽**的（两者之和等于 `vllm:num_requests_waiting`）". `metrics/loggers.py:1121-1132`: `total_waiting = num_waiting_reqs + num_skipped_waiting_reqs` → `gauge_scheduler_waiting`; capacity ← `num_waiting_reqs`; deferred ← `num_skipped_waiting_reqs`. The metric's own documentation @515-520 says "'deferred' = deferred by transient constraints (LoRA budget, KV transfer, blocked status)" and "Sum of all reasons equals vllm:num_requests_waiting" — the page's exhaustiveness claim is the source's own ✓ |
| Round-2 N6 `KVEventsConfig` cost row | **FIXED, defaults source-exact** | §7.1 L458-472 adds the 8-field table. Verified field-by-field against `vllm/config/kv_events.py`: `enable_kv_cache_events=False`, `publisher=None` (+ `__post_init__` → `"zmq" if enable else "null"`), `endpoint="tcp://*:5557"`, `replay_endpoint=None`, `buffer_steps=10_000`, `hwm=100_000`, `max_queue_size=100_000`, `topic=""` — all eight exact, and the `hwm` / `max_queue_size` contract text matches the source docstrings ✓ |
| Round-2 N4 telemetry-matrix exhaustiveness | **FIXED, and the arithmetic checks out** | §5.2 adds "**这张表不是全集。** `KVConnectorFactory` 共注册 **16** 个 connector，上表逐行核过其中 8 个；未列入的还有 `ExampleConnector`、`ExampleHiddenStatesConnector`、`LMCacheConnectorV1`、`NixlPullConnector`、`NixlPushConnector`、`DecodeBenchConnector`、`FlexKVConnectorV1`、`SimpleCPUOffloadConnector`…**"表里没有"只等于"本页未核"**", plus the separation of connector telemetry from KV cache events (§5.3). `factory.py` registers exactly **16** names and the 8 listed are exactly the complement of the 8 tabulated ✓. Per-row hooks verified mechanically: NIXL both (`nixl/connector.py:263,273`), Mooncake direct stats-only (`mooncake/mooncake_connector.py:590`, no `build_prom_metrics`), Mooncake Store both (390,400), MultiConnector both (601,653), Offloading both (201,211), HF3FS both (886,901), LMCache MP overrides both and returns `None` (1000,1011), MoRIIO overrides neither ✓ |
| Round-2 N8 ABORT vs negative queue | **FIXED at both residual sites** | §6.2 L379: "**两种终态的失真形态不同**：Scheduler 的 `FINISHED_ERROR` 通常已带 QUEUED event，缺的是 SCHEDULED，于是出现…负 queue…；FT sentinel 的 `FINISHED_ABORTED` 则可能一个 EngineCore event 都没有，此时各区间以基点 0 计算". §7.4 L505 same split. The lumped wording round 3 quoted is gone from both |
| New §5.3 figure | — | uses the house palette (`neutral fill:#ffffff,stroke:#64748b` + `acc1 fill:#dbeafe`) — contrast with 20's two figures |

### 25 — weight transfer / online update

| Item | Status | Evidence |
|---|---|---|
| `pause_scheduler` attribution | **FIXED** | Call tree L268 and §13 route L365 both write `EngineCoreProc.pause_scheduler`, with the explicit note "（pause 入口，`DPEngineCoreProc` 未覆写它）". AST of `vllm/v1/engine/core.py`: `EngineCore.pause_scheduler` @860, `EngineCoreProc.pause_scheduler` @1937 (override), `DPEngineCoreProc` (@2015) does **not** override it ✓ |
| `_invoke_utility_method` hop | **FIXED** | Call trees L279/291/303/313 route `[ZMQ 进程边界] EngineCoreProc._handle_client_request(UTILITY) → EngineCoreProc._invoke_utility_method → EngineCore.collective_rpc`. AST: `EngineCoreProc._handle_client_request` @1551, `EngineCoreProc._invoke_utility_method` @1614, `EngineCore.collective_rpc` @983 ✓. `DPEngineCoreProc._handle_client_request` @2153 exists but delegates everything except `START_DP_WAVE` to `super()`, so the tree is accurate for DP too ✓ |
| Other new symbols in the same rewrite | **pass** | `AsyncMPClient.call_utility_async` @1199, `_call_utility_async` @1202, `set_weight_version_async` @1269, `collective_rpc_async` @1292, `DPLBAsyncMPClient.call_utility_async` @1599 ✓; `ParallelConfig.sync_dp_state` @772 ✓; `EngineCore.set_weight_version` @992 ✓; the four HTTP routes `/init_weight_transfer_engine` @157, `/start_weight_update` @175, `/update_weights` @187, `/finish_weight_update` @205 in `vllm/entrypoints/serve/dev/rlhf/api_router.py` ✓; `finish_weight_update(weight_version=…)` on both `LLM` @895 and `AsyncLLM` @1241, `Worker.finish_weight_update()` @1423 no-arg ✓; "`world_size` 是模型并行 world（PP×TP×PCP，`external_launcher` 时再 ×DP）" ✓ (`config/parallel.py` `__post_init__` @867-870 and the `external_launcher` `*= data_parallel_size` @875) — and the page is right to override the stale `world_size` docstring ("world_size is TPxPP") |
| — | **NEW P1** | see R4-1 below |

### 26 — MultiprocExecutor RPC

| Item | Status | Evidence |
|---|---|---|
| multi-node MQ capacity (`nnodes_within_dp == 1` branch) | **FIXED** | L194: "…约为 `160 + 4 * 240 = 1120 MiB`——**这个数只对 `nnodes_within_dp == 1` 的单节点四 Worker 成立**；多节点 DP 下按上表最后一行的 4 MiB×6 重算，两套数不能混用（§1.1）"，plus L196 separating `GroupCoordinator`'s own `MessageQueue.create_from_process_group(cpu_group, 1 << 22, 6)` from the executor layer |
| TPU / SPMD `uni` row | **FIXED** | §1.1 L47: "backend=`uni`；TPU 且 `VLLM_XLA_USE_SPMD` 时这是默认解析的**第一条**分支（先于所有 CUDA 分支）" |
| `VLLM_USE_RAY_V2_EXECUTOR_BACKEND` defaults to on | **NOT_FIXED** | §1.1 L45/L46 enumerate the `=0` and `=1` branches but never say which is the default, and §8.3「配置合同」(L361-368, which *does* carry a 默认值 column for five other knobs) omits the variable entirely. Source `vllm/envs.py:928-929`: `lambda: bool(int(os.getenv("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "1")))` → **on by default**, so `backend=ray` selects `RayExecutorV2` (and therefore *does* reuse this page's control plane) unless the operator opts out. The trap is real: the type stub at `envs.py:65` reads `VLLM_USE_RAY_V2_EXECUTOR_BACKEND: bool = False`, so a reader checking the annotation gets the opposite answer. **Fix**: add "默认 **1**（`envs.py` 的类型标注 `= False` 只是 stub，运行期 `getenv(…, "1")`）" to the L46 row, or add the row to §8.3 |

---

## 3. Anchored-link audit (domain)

**34 anchored `[[page#heading]]` links** across the 26 pages + index. **34/34 resolve to a heading that exists verbatim in the target page — 0 broken.** `check_links --strict` agrees (453 pages, all five counters 0).

Six carry a **label/anchor section-number mismatch** — the anchor lands on the parent section while the display label names a sub-subsection:

| src | line | target | anchor § | label § |
|---|---|---|---|---|
| 07 | 411 | 08 | 5.1 | §5.1.3 |
| 07 | 492 | 08 | 3.3 | §3.3.1 |
| 12 | 229 | 08 | 5.1 | §5.1.3 |
| 16 | 394 | 08 | 3.3 | §3.3.3 |
| **17** | **193** | **09** | **2.6** | **§2.6.1** |
| **18** | **352** | **09** | **2.6** | **§2.6.1** |

The two bolded rows are round-3's NEW-P2-b, still open and still spread across both pages. The other four are the same pattern on pages owned by other reviewers — worth flagging domain-wide, since the mechanical checker cannot see it (the anchors resolve).

## 4. Cross-page `NN §x.y` audit

Mechanical extraction against each target page's real heading numbers:

- **82 cross-page `NN §x.y` references across the 26 pages → 0 dangling.**
- **`index.md` → 1 dangling**: see R4-2.

## 5. index.md coverage-boundary table, row by row

| Row | Registrar column | Verified at the named end? |
|---|---|---|
| “调用 API / 使用特性”的使用说明归属 | **无页面登记**（03 §9 …） | Row substance ✓ (01 still carries zero gap text), **but the pointer is dangling — R4-2** |
| P/D proxy/router | 22 §2、§14.4 | ✓ both; §2 registers "本域目前没有 owner…已提交 `planning-codebase-analysis` 裁决", §14.4 L1015 **and** L1025 now agree; 13 has 0 P/D hits |
| 配置解析总链 | **无页面登记**；各页只登记自己消费的字段 | ✓ — `create_engine_config` appears once, incidentally, at 18 L521 |
| Ray / external launcher 监督拓扑 | 23 §6.2、13 §4.2 | ✓ — 23 §6.2 L347: "**Ray 与 external launcher 的监督拓扑（含 Ray DP actor 的生命周期）在基线下全域没有 owner**…本页不把读者交给不拥有该内容的页面" |
| tool / reasoning parser | **无页面登记**（03 只在行文中提到 parser） | ✓ **FIXED** (round-3 P2-1) — `tool_parser` / `ToolParser` / `reasoning_parser` / `ReasoningParser` are **0 hits domain-wide** |
| pooling 执行（`PoolingRunner` 内部） | 12 §2.6、§2.10 | ✓ for §2.6 ("pooler 算法与 `PoolingRunner` 的内部状态本域暂无专页", L239) and for 12's §1.x handoff table (L98); **§2.10 mentions the pooling runner but carries no gap sentence** — P3, drop "§2.10" or add one line there |
| 非 CUDA 平台 | 11 §2.10、02 §3.8 | ✓ both — 11 §2.10 "本域暂无页面专门展开平台 runner"; 02 §3.8 "本域 26 页的核验范围是 CUDA…该覆盖缺口见…「已知覆盖边界」" (and it back-links to this very table) |
| gRPC / Rust frontend | 02 §5.3 | ✓ |
| scale-out render → token-in 往返 | 02 §5.5 | ✓ |
| `n>1` / beam search | **无页面登记**（14 只把 `n>1` 作为 trace-replay 的排除项提及） | ✓ **FIXED** (round-3 P2-1) — `ParentRequest` 0 hits; `beam search` appears only in 03's API-compat tables as a *rejected* combination, which is not a gap registration |
| 投机解码下的 `logprob_token_ids` | — | **removed** ✓ (round-3 P1-B closed; 14 §3.5 remains its declared owner) |

**Reverse check — does any page claim ownership of something this table calls ownerless?** No. Sweeps for `tool_parser`/`ReasoningParser`, `ParentRequest`/beam search, P/D proxy symbols and `create_engine_config` all come back consistent with the table.

## 6. New findings

### P1

**R4-1 — 25 §13 source-reading route attributes `_has_global_unfinished_reqs` to the base class `EngineCore`; it exists only on `DPEngineCoreProc`**

- Location: `/Users/suhaibo/96-knowledge/llm-knowledge/wiki/02_engineering/03_infer_frameworks/vllm/25_vllm_weight_transfer_online_update_analysis.md:365`
- Quote: `| DP pause 共识 | vllm/v1/engine/core.py::EngineCoreProc.pause_scheduler（pause 入口，DPEngineCoreProc 未覆写它）、DPEngineCoreProc._pause_complete / resume_scheduler、`**`EngineCore._has_global_unfinished_reqs`**`、vllm/config/parallel.py::ParallelConfig.sync_dp_state |`
- Source: AST of `vllm/v1/engine/core.py` gives exactly one definition — `DPEngineCoreProc._has_global_unfinished_reqs` @2283 (class `DPEngineCoreProc` @2015). There is **no** `EngineCore._has_global_unfinished_reqs`. `vllm/v1/engine/coordinator.py:41` names it in prose as "DPEngineCoreProc._has_global_unfinished_reqs method". Repo-wide the only other hit is the call site @2244, also inside `DPEngineCoreProc`.
- Why it matters: this is the third occurrence in three rounds of the same failure mode — a DP-only entry point attributed to the base class (round-3 #5 `reinitialize_distributed`, round-3 #6 `_eep_scale_up_before_kv_init`, now this one) — and it sits in the **§13 source-reading route**, which is exactly where CLAUDE.md's provenance rule requires a stable `path + qualified symbol` anchor. The cell also reads oddly because the *same cell* correctly qualifies the two neighbours as `DPEngineCoreProc.*`. The text was added by this uncommitted pass (the pre-image row was `vllm.v1.engine.core.DPEngineCoreProc`).
- Fix: one word — `DPEngineCoreProc._has_global_unfinished_reqs`.

**R4-2 — index's coverage-boundary table points at `03 §9`, which does not exist**

- Location: `/Users/suhaibo/96-knowledge/llm-knowledge/wiki/02_engineering/03_infer_frameworks/vllm/index.md:17`
- Quote: `| “调用 API / 使用特性”这一步的使用说明归属 | **无页面登记**（03 §9 登记的是输入路径缺口，不是这一条） |`
- Source of truth: page 03 has exactly six H2 sections — `## 1.` @12, `## 2.` @22, `## 3.` @119, `## 4.` @196, `## 5.` @214, `## 6. 成本、兼容入口与源码阅读路线` @287. The input-path registration round 3 cited (L291, "当前推荐输入路径是 Renderer → EngineInput → InputProcessor…") sits inside **§6**, not §9.
- Why it matters: this cell was written *by this pass* to fix round-3 P2-2 (the false "01 的覆盖缺口" registrar). The substance is now honest, but the replacement pointer is dangling, and `check_links` cannot see a bare `NN §x.y` reference — it would ship. It is also the **only** dangling `§` reference anywhere in the domain (82/82 resolve on the pages themselves), and it is on the index, the most-read navigation surface.
- Fix: `03 §6` (or drop the parenthetical).

### P2

**R4-3 — 26 §1.1 now owns the `Executor.get_class` selection table, but 18 still enumerates "六分支" in four places without naming 26 as its owner**

- 26 §1.1 (new this pass) is the domain's first full mapping table for `Executor.get_class`. 18 states "`Executor.get_class` 的六个分支" at L80 (⑬ flow table), L118 (③ startup table), L478 (config contract) and in the §11 call tree L531 (`|-- Executor.get_class  [自定义类 / ray(+VLLM_USE_RAY_V2_EXECUTOR_BACKEND) / mp / uni / external_launcher / 限定名]`).
- The two pages do **not contradict** each other, and 18 §1「它不是什么」already hands process management / broadcast / FIFO / shutdown to 26 with an anchored Related Pages entry. But "executor 选择" itself has no declared owner, which is the same drift pattern the domain review spent two rounds closing (`non_block` order, startup-order figures).
- Fix: one clause in 18's §1「它不是什么」 or on L80 — "executor 类的选择矩阵归 26 §1.1，本页只用它的结果".

### P3 (noted, not blocking)

- **Header-date inconsistency**: 20 → `2026-09-16`; 17/18/21/22/23/24/25/26 → `2026-09-15` although the pass edited them. Pick one convention.
- **index row "pooling 执行 → 12 §2.6、§2.10"**: §2.10 mentions the pooling runner in the phase table but carries no gap sentence.
- **20 L513**: the masked MoE activation path's `silu_with_clamp` name branch (`fused_moe/activation.py:261`) is still unmentioned (round-2 P2-3 residual).
- **21 §9.4 其四**: "没有任何 NVFP4 replacement" is precise about *replacements*, but `QUANT_OPS[kNvfp4Dynamic] = torch.ops._C.scaled_fp4_quant.out` does exist in the same file; a half-sentence would pre-empt a reader's objection.

## 7. Duplication sweep on the mechanisms this pass added

- **KV cache events**: 23 §5.3 sole owner; 07 L754 is a one-line boundary sentence that now carries the anchored `23#5.3 …` link ✓ (round-3 P2-10's 07 half is closed).
- **`deferred` waiting reason**: 23 §6.1 sole owner; 05's hint points there ✓.
- **`KVEventsConfig`**: 23 §7.1 only; 22's env table row links 23 §5.3 with an anchor ✓ (`VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES` default `True` confirmed at `envs.py:279/1955`).
- **utility RPC hop / response FIFO**: 25 §13 + L407 defer the FIFO consequence to 26 and link it at L445/L455 ✓.
- **Elastic EP class attribution**: 18 and 23 §6.3 agree on `DPEngineCoreProc.*` ✓.
- **One new overlap**: R4-3 above.

## 8. Remaining blockers

1. **R4-1** — one word on `25:365`.
2. **R4-2** — one cell on `index.md:17`.

Everything else on my pages is a P2/P3 one-liner: 20's N5 / N6 / N7 (+ `silu_with_clamp`), 22's L1019 page-level `→ 23`, 26's missing `VLLM_USE_RAY_V2_EXECUTOR_BACKEND` default, 17/18's `#2.6 … | §2.6.1` label pairing (plus the four same-pattern rows on 07/12/16), and R4-3. No new P0. T0 four gates all green.
