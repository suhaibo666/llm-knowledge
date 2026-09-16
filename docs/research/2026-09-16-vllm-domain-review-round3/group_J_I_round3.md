# Round-3 domain-level verification: vLLM 01–26 + index + stack/course pages

- **Role**: independent reviewer (third round). Findings only — **no file was edited** (wiki/, docs/, skills/, source checkout all untouched).
- **Object**: uncommitted working tree, 30 files (`git diff --stat`: +2104/−729). All 30 files carry mtimes 09:23–09:58 on 2026-09-16, i.e. the whole domain was touched in this pass.
- **Frozen source**: `/Users/suhaibo/97-llm/vllm`, `git rev-parse HEAD` = `199cb9b964822e59ab9b58d88e7be31eb419a2ae` at both start and end. Read-only; HEAD not moved, no fetch/checkout.
- **Method**: mechanical extraction + verbatim heading match for every `[[page#heading]]`; mechanical extraction of every cross-page `NN §x.y` and `本页 §N`; heading-id/alias-anchor audit against the built `site/`; per-item re-check of round-2 open items against the frozen source (not against the new prose); single-owner exclusivity sweeps for the mechanisms added this round.
- **Run boundary**: static source reading + T0 checkers + one scoped mkdocs build. No vLLM, GPU, multi-host, NIXL/Mooncake/MoRIIO/ZMQ/OTLP service, no fault injection, no live browser.

## 0. Mechanical gates

| Gate | Result |
|---|---|
| `check_links --strict` | 452 pages; broken / ambiguous / bare_index / stale_section / orphans **all 0** |
| `check_math --changed --strict` | 59 files, **0 error / 0 warning** |
| `check_markdown --changed --strict` | 59 files, **0 error / 0 warning** |
| `check_assets --changed --strict` | 59 files, **0 error / 0 warning** |
| `python -m tools.mkdocs_site.cli build --changed` | **PASS**, 35 pages: broken_links 0, missing_anchors 0, missing_assets 0, missing_legacy_routes 0, orphans skipped (scoped) |

---

## 1. Anchored-link audit

**27 anchored wikilinks** across the 26 pages plus the two changed pages (`01_llm_inference_technology_stack_analysis.md`, `courses/torch_compile_end_to_end.md`). Every one resolves to a heading that exists **verbatim** in the target page. **0 defects.** The two changed pages contain no anchored links of their own (the course page's three new links are page-level, not anchored).

| src | line | target | anchor | status |
|---|---|---|---|---|
| 01 | 257 | 03 | `2.2 输出上限：先选择缺省值，再应用硬上限` | OK |
| 05 | 204 | 23 | `6.3 受控恢复：只恢复可恢复的执行环境` | OK |
| 07 | 411 | 08 | `5.1 更细粒度复用：命中 6 个 token，为什么还要复制半个块` | OK |
| 07 | 492 | 08 | `3.3 分配：把候选命中变成受保护的请求映射` | OK |
| 10 | 340 | 17 | `4.3 KV scale 参数从哨兵到消费` | OK |
| 11 | 46 | 06 | `4.1 队列里必须同时保留 future 和原计划` | OK |
| 12 | 49 | 06 | `4.1 队列里必须同时保留 future 和原计划` | OK |
| 12 | 229 | 08 | `5.1 更细粒度复用：命中 6 个 token，为什么还要复制半个块` | OK |
| 12 | 559 | 06 | `4.1 队列里必须同时保留 future 和原计划` | OK |
| 14 | 460 | 16 | `8.3 草稿发布：新候选怎样回到 Scheduler` | OK |
| 15 | 331 | 16 | `8.4 MRV1、抢占和在途结果的边界` | OK |
| 16 | 394 | 14 | `6.4 接受新 token、draft 预演、跨 reasoning 边界、完成与失败` | OK |
| 16 | 394 | 08 | `3.3 分配：把候选命中变成受保护的请求映射` | OK |
| 16 | 442 | 14 | `6.4 接受新 token、draft 预演、跨 reasoning 边界、完成与失败` | OK |
| 17 | 193 | 09 | `2.6 融合前分别分片，融合后仍能拆回各投影` | OK |
| 18 | 352 | 09 | `2.6 融合前分别分片，融合后仍能拆回各投影` | OK |
| 19 | 270 | 17 | `6.2 reload 与 CUDA Graph：本页负责哪一半` | OK |
| 19 | 362 | 06 | `4.1 队列里必须同时保留 future 和原计划` | OK |
| 19 | 460 | 12 | `1.5 位置：MRV2 在 EngineCore 一步里接什么、交什么` | OK |
| 20 | 735 | 17 | `6.2 reload 与 CUDA Graph：本页负责哪一半` | OK |
| 22 | 924 | 23 | `5.2 KV 传输：通用搬运 stats，connector 决定实际指标` | OK |
| 22 | 924 | 23 | `5.3 KV cache 事件：从 block 生命周期发布到外部前缀路由` | OK |
| 22 | 1103 | 23 | `5.2 KV 传输：通用搬运 stats，connector 决定实际指标` | OK |
| **24** | **129** | **14** | **`4.3 logits processor 的变体集合从哪里枚举出来`** | **OK (was B1)** |
| **25** | **317** | **13** | **`3.1 进程、对象与状态归属`** | **OK (was B1)** |
| **25** | **407** | **26** | **`6. Future：响应没有请求 ID 时怎样保持配对`** | **OK (was B1)** |
| 25 | 226 | 17 | `6.2 reload 与 CUDA Graph：本页负责哪一半` | OK |

Also checked and clean: no markdown-style `](page.md#anchor)` links and no same-page `[[#anchor]]` links exist in these pages (so the audit above is complete for the anchored class).

**Render-level confirmation.** Extracted all 24 unique anchored `href="…html#…"` from the built `site/` and matched them against the union of `<hN id>` and the rewriter's alias `<a name>` anchors: **24/24 resolve**. MkDocs' own validator logs two INFO lines (14→16 §8.3, 15→16 §8.4) because it inspects heading ids only and does not know about the alias anchors; the repo checker counts aliases and correctly reports `missing_anchors: 0`. **Not a defect — no action.**

*One accuracy note, not a link defect*: the new `wiki/changelog.md` entry concludes "站点实际生成 `#83-scheduler` 一类锚点，**因此指向中文小节的链接落在页首**". The first half is true (heading ids strip CJK), the conclusion is not — the alias anchors make the links land on the right section. Worth correcting so a future maintainer does not "fix" working links.

## 2. Cross-page `§N` references

Mechanically extracted and checked against the target page's real heading numbers:

- **124 cross-page `NN §x.y` references → 0 dangling.**
- **57 `本页 §N` references → 0 dangling.**

## 3. Round-2 open items, item by item

| Item | Status | Evidence |
|---|---|---|
| **§1 B1** — 3 broken wikilink anchors | **CLOSED** | 27/27 verbatim (table above); scoped build PASSES with all five counters 0 |
| **§3 (a)** — 22 §2 / §14 / §14.4 asserting 13 owns P/D routing | **MOSTLY FIXED, 1 residual (P1-A)** | §2 L49 ✓ ("本域目前没有 owner…已提交 `planning-codebase-analysis` 裁决"; 13 owns **DP-replica-internal** routing only). §14「它不是什么」L924 ✓ ("在基线下全域无 owner"). §14.4 L1025 ✓. Related Pages L1101 ✓. **But §14.4 L1015 still reads `… proxy/router 参考实现、实例入口与请求生命周期 → **13**`** — ten lines above L1025, in the same table, which says the opposite |
| **§3 (b)** — 23 L347 pointing Ray/external-launcher supervision at 18 and 13 | **FIXED** | 23 §6.2 L347 now: "**Ray 与 external launcher 的监督拓扑（含 Ray DP actor 的生命周期）在基线下全域没有 owner**…18 拥有的是 executor 类的选择与 `external_launcher` 对 world size 的影响，不是它们的故障监督，本页不把读者交给不拥有该内容的页面". 13 §4.2 L578 registers the Ray DP actor half consistently (`CoreEngineActorManager`, `DPMoEEngineCoreActor`, "故障传播与退出语义在基线下全域无 owner") and links the index's coverage-boundary section |
| **index coverage-boundary note** | **ADDED but 1 row contradicts the pages (P1-B) + 3 rows have false registrars (P2)** | See §4 |
| **domain P1 N1** (3 anchors) | **FIXED** | as above |
| **domain P1 N2** (`reinitialize_distributed` class) | **FIXED** | 18 L90 / L302 / L586 all now `DPEngineCoreProc.…`, with an explicit in-line note "这两个入口都定义在 `DPEngineCoreProc` 上，不在基类 `EngineCore`…同一入口在 23 §6.3 的写法一致". 23 L414 agrees. Source: `vllm/v1/engine/core.py`, `reinitialize_distributed` @2302 and `commit_prepared_elastic_ep` @2356, both inside `class DPEngineCoreProc` (@2015). Contradiction gone |

### Round-2 new P2s on 22/23 that this pass touched

| # | Item | Status | Evidence (frozen source) |
|---|---|---|---|
| 22 N1 | HMA trigger written as a single condition | **NOT FIXED** | 22 L134 still "只有在 cache groups 中存在非 `FullAttentionSpec` 时才令 `_is_hma_required=True`". Source `nixl/base_worker.py:387-393`: `self._is_hma_required = (not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager and any(not isinstance(spec, FullAttentionSpec) …))`. The missing AND term is the operator's workaround, so omitting it costs the section its actionability |
| 22 N2 | "N → N−1" written as unconditional | **NOT FIXED** | L134 ("并把已承诺进度 N 改成 N−1 后继续计算"), L299 ("再把已承诺进度 N 改成 N−1 继续"), and figure node Y ("继续 N → N-1") are all unconditional. Source `Scheduler._update_waiting_for_remote_kv` (scheduler.py @2870) guards the decrement: `if request.num_computed_tokens == request.num_tokens: request.num_computed_tokens = request.num_tokens - 1` — full-prompt hit only |
| 23 N4 | telemetry matrix reads as exhaustive | **NOT FIXED** | The new lead-in only defines what "有" means ("classmethod 返回实际 stats/collector，不是仅覆写后返回 `None`"); no scope sentence was added, and the closing sentence still enumerates as if closed. The factory registers 16 names; 6 more (`LMCacheConnectorV1`, `FlexKVConnectorV1`, `SimpleCPUOffloadConnector`, `DecodeBenchConnector`, `ExampleConnector`, `ExampleHiddenStatesConnector`) override neither hook |
| 23 N6 | `KVEventsConfig` missing from the cost table | **NOT FIXED** | §5.3 L308 names `buffer_steps` / `max_queue_size` / `hwm` / `replay_endpoint` / `topic`; §7.1's per-class field tables cover only `ObservabilityConfig`, `ParallelConfig` + `FaultToleranceConfig`, and 构造参数与环境开关. No `KVEventsConfig` table, so the default 10 000-batch replay buffer and 100 000-item queue are not estimable from the page that now owns the mechanism |
| 23 N5 | deferred / `skipped_waiting` attribution | **NOT FIXED — highest-value unfixed item** | §6.1 still writes the async wait as an *alternative* to deferred: "既可能来自 capacity/deferred waiting，也可能来自 P/D consumer 的**异步**远端 KV load". Source: `Scheduler.schedule` @1162-1166 puts the `load_kv_async` request into `step_skipped_waiting` → `self.skipped_waiting` @1245 → `make_stats` `num_skipped_waiting_reqs` → `loggers.py:1130` sets `gauge_waiting_by_reason[WAITING_REASON_DEFERRED]`, and the gauge's own documentation reads "'deferred' = deferred by transient constraints (LoRA budget, **KV transfer**, blocked status)". The async wait **is** the `deferred` label. 23 is the declared owner of 05's `deferred` hint, so this wording sends an operator away from the one gauge that localizes the symptom |
| 23 N8 | "negative queue" attributed to both ERROR and ABORT | **PARTIALLY FIXED** | §3.1's new paragraph is now precise — ABORT is described as "同样可能产生失真的完成区间", with no negative-queue claim. §6.2 ("即便 token 为空、events 不完整…出现 §3.1 所述的负 queue 或超大 prefill/inference") and §7.4 ("显式 ERROR/ABORT 输出…产生负 queue") still lump the two. Negative queue requires `queued_ts != 0`, which `_send_finish_outputs_to_client`'s event-less ABORT output cannot establish |

## 4. index.md

- **Page count consistent.** `ls *.md` = 27 = 26 content pages + index; matches "26篇内容页 + 本索引".
- **Titles and one-line questions consistent.** All 26 index labels map to an existing page whose H1 matches the label's subject; no stale label.
- **Reading arrows.** The three round-2 corrections hold: 26 moved ahead of 18 and 18's prerequisites now include MultiprocExecutor; 21 precedes 20; 23's row reads "Scheduler、KV、Serving、MultiprocExecutor与分布式 → 调试与排障", which agrees with 23's own Related Pages (05 listed first, as the place to *apply* 23) and with 05 §4 → 23 §6.3. The new MRV2 note above the runner table ("默认先进入 MRV2（12）；11 是 MRV1 的对照与回落路径") agrees with 12 §2.1 and 19 §7.3.
- **New「已知覆盖边界」section**: 11 rows, each marked as submitted to `planning-codebase-analysis`. This is the right fix for round-2 N9 and is the single biggest reader-visible improvement this round. Three problems in the registrar column:

| index row | registrar column | actual |
|---|---|---|
| 投机解码下的 `logprob_token_ids` | 14、16 | **Contradiction (P1-B)**: 14 §3.5 says verbatim "**它有一条投机例外，这里是它的 owner**" and documents both branches (`num_draft_tokens == 0 or rejection_sampler is None` if/else; `include_token_ids=` on the batch-sharded gather; the "Rejection sampler does not return logprob token ids" comment); 16 §9.1 closes with "该字段的正常语义与本例外的 owner 是 14 §3.5". The topic **has** an owner, so listing it under "本域没有 owner 页面" is wrong in both directions |
| tool / reasoning parser | 03 | 03 L170/L176 mention parsers only in passing and register **no** gap; `tool_parser` / `ToolParser` / `reasoning_parser` / `ReasoningParser` are 0 hits domain-wide. The registrar points a reader at a page that says nothing about the gap — the exact defect this pass fixed in 22/23/02 |
| `n>1` / beam search | 14 | 14 mentions `n>1` only as a trace-replay exclusion (L256, L391); `ParentRequest` is 0 hits. No gap registration |
| "调用 API / 使用特性"的使用说明归属 | 01 的覆盖缺口、03 的分类 | 01 has **zero** gap-registration text (no 无 owner / 暂无 / 缺口 / 规划 anywhere). 03 L291 registers input-path gaps (streaming input, multi-prompt TODO, sparse media mask, streaming Derender), not the usage-doc ownership question |

Correctly registered rows (verified at both ends): P/D proxy (22 §2 + §14.4 — modulo P1-A), Ray/external launcher (23 §6.2 + 13 §4.2), pooling execution (12 §2.6 and §2.10 "pooler 与 `PoolingRunner` 内部本域暂无专页"), non-CUDA platform (11 §2.10), gRPC/Rust frontend (02 §5.3 "这三项在本域没有深度 owner…已提交 `planning-codebase-analysis`"), scale-out round trip (02 §5.5 "本域尚无页面展开 scale-out 的部署编排与失败语义"), config-resolution chain (registered only as "多页各自登记自己消费的字段" — vague but not a false pointer).

## 5. Duplication and new domain-level problems

**Genuinely improved (verified at both ends, not just "arrow swapped"):**

- `finished_req_ids` production rule: 06 L224 now says the production side "归 07 §8.1.2"; 07 owns it at §8.1.2 L603 and keeps only a one-line consumer summary at L227. The round-2 "three copies" is down to one owner + one pointer + one summary line.
- stale/drop decision: 06 §5.1 now keeps only the invariant ("`update_from_output()` 是按传入的那份 `SchedulerOutput` 的 `num_scheduled_tokens` 遍历的") and states "这三类的判定表、计数规则与图归 07 §8.3，本页不再重放一遍".
- async resolution axis: 07 §2.1 L113 now names 06 §4.1 as owner and keeps only the one boundary it needs ("有 batch queue" ≠ "必定选了 AsyncScheduler").
- `SchedulerOutput` contract survived the §8.1.4 split: source has **exactly 22** top-level annotated fields; 07 states 9+10+3=22 at both L627 and L631, and 11 L525 / 12 L332 both say 22. No count drift.
- B2 #3 (24 `vllm.logits_processors`): 24 now writes "两个进入点：前端 `SamplingParams.verify` 的每请求校验，与 MRV1 `build_logitsprocs`" consistently in the table (L118), the body (L120), the figure spec (L80), the flowchart node (L97), the call tree (L282-283) and the source route (L308).
- B3 item 5 (02 `test_abort_defers_free`): now carries the double gate ("配了 KV connector 且它是 consumer" **并且** `max_concurrent_batches > 1`) and defers the full wording to 06 §6.2 — matches 07 L723.
- B3 item 8 (02 outlets): 02 now links 14 (3×), 15 (3×), 16 (2×), and §5.3 / §5.5 / §5.6 cover gRPC/Rust/Omni, the scale-out round trip and DP=2 → 2 API servers → `run_multi_api_server` (delegating the derivation table to 13 §4.1).
- F7 (round-2 NOT_FIXED): 11 L563 now reads "15 §7.7", and 15 §7.7 is indeed "EC transfer"; §7.4 is `prompt_embeds`. Fixed.
- §4.2 #13 (round-2 PARTIAL): 11 L560 and 12 L357 both now back-link to 14 §3.8. Closed.
- §4.2 #19 (round-2 PARTIAL): closed as an **honest registration**, not as coverage — the right outcome given the planning route.

**Single-owner exclusivity of the mechanisms added this round** — no new double ownership:

- KV cache events: 23 only (5 hits) plus one boundary sentence in 07 L745. `KVEventBatch` / `BlockStored` / `AllBlocksCleared` appear in no other page; 08 has none.
- scale-out (`VLLM_ENABLE_SCALE_OUT_ENDPOINTS`, `--tokens-only`, `register_scale_out_api_routers`): 02 only.
- Handoff object names remain spelling-consistent domain-wide; no `SchedulerOutputs` / `ModelRunnerOutputs` variants appeared.

**Remaining / new P2s:**

- `non_block` submission order is still stated in **7 pages** (02=6, 06=5, 07=3, 11=6, 12=6, 19=2, 23=1). Unchanged from round 2; 06 deepened as owner (2→5) but no push-side prose was deleted. Drift risk only — wording is currently consistent.
- Startup-order figures still appear across ~10 pages (08's new KV-budget figure included). Registered risk, content consistent.
- 07 L745 names `KVEventBatch` without pointing at 23 §5.3, while the rest of the domain links 23 §5.2/§5.3 with anchors. One-line inconsistency in pointer precision.
- 22 L854's environment-variable table row still points at page-level "→ **23**" while the same page's §14.4 and Related Pages use anchored `23 §5.3`. Round-2 N10, unchanged.
- **02 §6's outlet table omits 24.** It now covers 01, 03–23, 25, 26 and the index — 24 is the only content page with no landing point in the architecture page's exit table.
- 22 §14.4's HMA row (L1012) and `requires_piecewise_for_cudagraph` row are fine; round-2 P2-1 through P2-11 on 22/23 (call-tree `wait_for_save=False`, MoRIIO `TransferError`, MoRIIO P-side finish reasons, §6.4 title, figure H→R label, NaN MRV2 route line, §9 `FaultToleranceConfig`) — of these, §6.3's MRV2 NaN branch and §9's `FaultToleranceConfig` path **were** fixed this round (23 §6.3 now documents `Sampler.compute_nans` and the three MRV2 consumption points; §9 now lists `vllm/config/fault_tolerance.py::FaultToleranceConfig`). The rest remain.

## 6. Changed course and tech-stack pages vs the CLAUDE.md courses rule

**`wiki/courses/torch_compile_end_to_end.md` (+4 lines) — COMPLIES.** The addition is a single reading-path bullet: a placement instruction ("读完 §4 与 §7 后转"), three wikilinks into the functional tree in order (19 → 21 → 20), one parenthetical orientation each ("compile range、capture 阶梯与运行期 dispatch" / "自定义 post-grad pass 与 IR lowering" / "pass 改写后的实现端点"), and an explicit non-duplication clause ("这三页是 PyTorch 侧机制的下游消费者，不重复本课已讲的编译栈内部"). Reading order + links + one-line orientation only; no body content, no mechanism restated, no second source of truth. This also closes the round-2 F24 sub-item "`wiki/courses/**` 仍无 vLLM 链接".

**`wiki/02_engineering/03_infer_frameworks/01_llm_inference_technology_stack_analysis.md` (+1/−1) — fine.** Step 2 of the learning path now inserts 02 ("先建立六个职责模块与它们的协作边界") before 06. Both link labels match the target H1s. This is a stack overview page, not a `courses/` page, so the courses rule does not bind it; the change adds no body content either. Closes the other half of round-2 F24.

---

## 7. New findings this round

### P1

**P1-A — 22 §14.4 L1015 still routes P/D instance topology to 13, contradicting L1025 of the same table**
- Location: `wiki/02_engineering/03_infer_frameworks/vllm/22_vllm_disaggregated_kv_serving_analysis.md:1015`
- Evidence: `| P/D 实例拓扑 | 一句约束：路由方必须把请求送到 transfer_mode 匹配的实例 | examples/disaggregated/ 的 proxy/router 参考实现、实例入口与请求生命周期 → **13** |`, while L1025 of the same table says "**本域暂无 owner**，已提交 `planning-codebase-analysis`；候选归属 13，但基线下 13 未展开，§2 与 §14 的措辞与此一致". 13 (+283 lines this round) still has 0 hits for `kv_transfer_params` / `do_remote_decode` / P/D proxy.
- Why it matters: this is the last surviving instance of the exact defect the pass set out to fix; a reader following the ownership table lands on 13 and finds nothing.
- Fix: change L1015's third column to match L1025 — e.g. `… 实例入口与请求生命周期 → **本域暂无 owner**，已提交规划（见本表末行）`.

**P1-B — index's coverage-boundary table lists `logprob_token_ids` under spec as owner-less, but 14 §3.5 declares itself its owner**
- Location: `wiki/02_engineering/03_infer_frameworks/vllm/index.md`, row "投机解码下的 `logprob_token_ids` | 14、16"
- Evidence: 14 §3.5 — "**它有一条投机例外，这里是它的 owner**" — then documents the two branches and the source comment; 16 §9.1 (L477) closes "该字段的正常语义与本例外的 owner 是 14 §3.5". Both ends are written and mutually linked.
- Why it matters: the new section's whole purpose is to tell readers truthfully where the domain stops. A false gap is as misleading as a hidden one, and it contradicts two pages that were upgraded to closure in round 2.
- Fix: delete the row (the topic is covered), or restate it as the genuinely deferred remainder if one exists.

### P2

1. **index rows "tool / reasoning parser → 03" and "`n>1` / beam search → 14" name registrars that register nothing.** Neither page contains a gap statement. Either add a one-line registration to 03 and 14, or change the column to "无页面登记".
2. **index row "01 的覆盖缺口" is false** — 01 contains no coverage-gap text at all.
3. **22 N1 HMA condition** (L134) — add the `disable_hybrid_kv_cache_manager` AND term and note in §9 that closing the hybrid manager returns to the invalid-block contract.
4. **22 N2 N→N−1** (L134, L299, figure node Y) — qualify with the full-prompt-hit condition.
5. **23 N5 deferred attribution** (§6.1) — state that the async wait **is** `num_requests_waiting_by_reason{deferred}`, not an alternative to it. Highest-value P2; borderline P1 because 23 is the declared owner of 05's `deferred` hint.
6. **23 N4 matrix scope** (§5.2) — add a scope sentence naming the 6 registered connectors the matrix does not cover, and separate "no generic connector telemetry" from "no KV cache events" (§5.3 owns the latter; the two can cross-link).
7. **23 N6 `KVEventsConfig`** (§7.1) — add the 8-field table with defaults so the replay-buffer memory cost is estimable.
8. **23 N8 residue** (§6.2, §7.4) — restrict "negative queue" to ERROR terminals carrying a QUEUED event; describe the event-less ABORT sample as zero-based.
9. **02 §6 outlet table omits 24.**
10. **07 L745** names `KVEventBatch` without pointing at 23 §5.3; **22 L854** still uses a page-level "→ 23" where the rest of the page uses anchored §5.3.
11. **`wiki/changelog.md`** overstates the anchor tool gap: the alias anchors do resolve, so "指向中文小节的链接落在页首" is not the case.

### P0

**None.** No new factual error against the frozen baseline was found in the text this pass introduced, and the round-2 P0 set remains fixed.

---

## 8. Domain verdict

**(a) Normal online text request — PASS.** The Scheduler→Runner contract is now three-way consistent (producer 07 §8.1.1–8.1.4, consumers 11 §3.4 and 12 §2.10, all agreeing on 22 fields, which matches the source exactly). The default async axis is single-owner (06 §4.1) with five pages pointing at it. Round-2's one remaining misroute (F7, 11 → 15 §7.4) is fixed to §7.7. prompt logprobs now close at both ends. No break remains on this path.

**(b) Startup — PASS with one missing link at the head.** Memory profiling → KV budget → `num_gpu_blocks` is owned by 08 §3.1.3 with a worked example. The one gap is still the **config-resolution chain** (`EngineArgs → VllmConfig.__post_init__ → check_and_update_config`): `create_engine_config` appears once, incidentally, in 18 L521. A reader still cannot answer "in what order is my flag rewritten into the final config". **Planning-deferred** and now visible to readers via the index's coverage-boundary section, which is the correct handling under CLAUDE.md.

**(c) User journey — install → run → read output → tune → debug is coherent (01 → 04 → 05, with 05's owner pointers fixed and anchored). `call API` remains the real break**: tool calling, `/v1/responses`, Anthropic `/v1/messages`, `/tokenize`, beam search, `n>1`, the LoRA serving entry and embedding/rerank operation still have no owner, and 01 still has no API quick-reference section. `extend` is half-open: 24 owns the plugin ABI, but packaging how-to and non-CUDA onboarding have nowhere to go. All of these are **planning-deferred** and (except the LoRA entry and the API quick-reference) now registered in the index.

**Fixable now, not planning-deferred:** P1-A (one table cell), P1-B (one index row), and P2 items 1–10 — every one is a single sentence or cell.

**Overall: CONDITIONAL PASS.** This pass did what the recheck asked: B1 is closed with build evidence, the `reinitialize_distributed` attribution contradiction is gone, the "point readers at a page that doesn't own it" defect was fixed in 22 §2/§14, 23 §6.2 and 02 §5.3, the index finally exposes the domain's known boundary, the duplication the last round flagged was resolved by *deleting* push-side prose rather than adding links, and the course page now connects to the domain without absorbing body content. What stands in the way of merge is two one-line self-contradictions (P1-A, P1-B), both introduced by this pass's own honesty edits not being carried to every occurrence — the same failure mode as round 2's B1. Recommend fixing those two plus P2 items 3–5 (the 22/23 source-fidelity qualifiers) before commit, and batching the rest.
