# Independent review — vLLM Serving control plane refactor

Reviewer: `/root/review_serving`; never the page writer; no wiki source edited by reviewer.

Reviewed source baseline: `/Users/suhaibo/97-llm/vllm` HEAD verified as `199cb9b964822e59ab9b58d88e7be31eb419a2ae`. Checkout was not moved.

Contract: existing vLLM 13 page, feature-analysis profile, Megatron-LM 12 used as organizational reference; DP load/wave protocol and Python serving topology/readiness/shutdown owned here; device collective, full FT and request protocol retain their neighboring owners.

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 13_vllm_serving_control_plane_analysis | pass | pass | pass | transform, timing, coupled-planes | pass | 3/3+ | PASS | feature: pass; initial findings corrected and rechecked |

## Initial findings and disposition

1. **Source fidelity, §2.3 normal-wave exclusion — corrected.** API `EngineIdentity` is bytes, derived with `rank.to_bytes(2, "little")`; FIRST_REQ passes that identity without conversion; Coordinator preserves it; the Engine compares it with integer `engine_index`. The original prose/figure incorrectly promised that E1 was excluded. The final contradiction block distinguishes implementation from intended exclusion, and the re-rendered sequence shows the same broadcast reaching E0 and E1. Engine-originated reports carry integer `outputs.engine_index` and retain the working exclusion case. This is static source-path evidence, not device reproduction.
2. **Hop-walk, §3.3 preprocess-to-queue edge — corrected.** `preprocess_add_request` and `input_queue.put_nowait` are sequential sibling operations inside `process_input_sockets`; the final caller tree now shows two siblings.
3. **Conservation, §4.1 shared HTTP entry — corrected.** The prior two-API example explicitly shared one HTTP entry. The final page states that Python multi-API children receive the same listening socket, contrasted with multi-port per-rank endpoints. This agrees with `run_multi_api_server` construction.
4. **Principle figure guard, §2.4 — corrected.** Stale frontend-wave notification broadcasts only while Coordinator records idle. The final branch box includes that state guard and was re-rendered and inspected.

No unresolved blocking findings remain.

## Source spot-check and hop-walk evidence

Three primary load-bearing anchor groups, all actually opened:

- `vllm/v1/engine/coordinator.py::DPCoordinatorProc.process_input_socket / _send_start_wave`: subscriber READY, frontend wake, stale-wave guard, Engine-origin start/complete acceptance, stats polling/publication. Confirmed START transmission itself does not set Coordinator running, completion requires a non-stale wave, and stats are asynchronous rather than an atomic full-rank snapshot.
- `vllm/v1/engine/core.py::DPEngineCoreProc.add_request / _handle_client_request / run_busy_loop / _has_global_unfinished_reqs / _pause_complete / resume_scheduler`, plus `vllm/config/parallel.py::ParallelConfig.sync_dp_state / has_unfinished_dp`: matching/stale wave paths, actual rank-0 start edge, short-wave completion case, first-step and interval synchronization, pause consensus and resume barrier/guard.
- `vllm/v1/engine/core_client.py::DPAsyncMPClient.add_request_async / _ensure_stats_update_task / DPLBAsyncMPClient.get_core_engine_for_request / process_engine_outputs`, plus `tests/v1/engine/test_engine_core_client.py::test_dplb_kv_pressure_amplifies_waiting_penalty / test_dplb_finished_requests_release_inflight`: replayed 30/20 and 15/20 scoring and confirmed mapping/inflight release only on consumed finished IDs.

Additional anchors actually walked: `EngineCoreClient.make_async_mp_client`; `MPClient` identity construction; `EngineCoreProc.process_input_sockets / _process_input_queue / _handle_client_request / _process_engine_step / process_output_sockets`; `EngineCore.add_request`; `AsyncMPClient._send_input / _send_input_message / _ensure_output_queue_task`; `ServeSubcommand.cmd / run_multi_api_server`; `launch_core_engines / wait_for_engine_startup / get_engine_process_shutdown_timeout`; `VllmConfig.needs_dp_coordinator`; `ParallelConfig.data_parallel_external_lb`; `vllm/v1/utils.py::shutdown`; and both cited late-request pause/deadlock tests.

The main request path was followed from target selection and ADD submission through Engine preprocessing, queue and Scheduler handoff, then the output thread and API finished-request consumer. The control path was followed separately across message boundaries rather than treating cross-process notifications as direct calls. Deployment selection, launch-context entry/yield/exit, startup handshake, and shared shutdown deadline were also checked.

## Conservation and feature contract

Compared with `/tmp/vllm-serving-refactor/before.md`: topology/API counts and headless split, state ownership, dynamic endpoints and pipe failure cases, two distinct Core READY orderings, ready metadata aggregation, global startup hash guard, scoring inputs and exact examples, inflight/abort completion, admission correction, default failure propagation, multi-port health aggregation, FT restrictions, shared shutdown budget/ROCm caveat, documentation conflicts, test routes and related-page owners remain present. Dense-external coordinator wording was corrected against the configuration's explicit supported-scope documentation. Existing substantive material was neither silently discarded nor moved to a nonexistent owner.

The final introduction establishes problem and solution before implementation. The request and wave examples develop the primitive into the real API/Coordinator/Engine system; queue, execution, result visibility and global idle are distinct boundaries. Variants are enumerated from construction/configuration sites; external LB and device execution dependencies are scoped rather than narrated as externally verified internals. Component costs reconcile with the overall cost ledger, and improvements inferred by the analyst are labeled.

## Rendered figures

Actually viewed `/tmp/vllm-serving-refactor/figure-0.png` through `figure-3.png`; re-viewed final regenerated figures 2 and 3 after revisions.

- Figure 0 replays the same R and two queue inputs through KV-dependent scores, selected owner, optimistic waiting and completion boundary.
- Figure 1 distinguishes direct ADD and independent control notification, actual start confirmation, request completion and wave completion; identity-type contradiction is reflected.
- Figure 2 shows stale Coordinator/API state branches and explicit-pause protection through resume, with idle guard visible.
- Figure 3 visibly separates process management from direct API↔Engine traffic and Coordinator status/control traffic; per-direction contracts remain in the adjacent table.

Labels were readable; no overflow, overlap or leaked wiki markup was observed. Stranger-reader checks are supported by the nearby prose and explicit examples. The ownership view supplements, rather than replaces, the principle figures.

This review is static source and rendered-document verification. No GPU model, online serving, device fault injection or third-party runtime execution was performed. Repository mechanical quality gates remain the coordinator's integration responsibility.


## Integration verification by page writer

- Scope: one existing analysis page, its domain-index row, changelog, and this review record. The source repository HEAD remains at the pinned commit; this task did not modify it. Final status showed an untracked `artifacts/` directory absent at the initial check; its origin was not investigated; it was not inspected or changed by this task.
- Conservation map: old §1–2 problem/topology/owners → new §1, §3.1, §4.1; old §3 readiness/coordinator → §2.1, §2.3–2.4, §4.2; old §4 scoring/inflight/admission → §2.2, §3.3, §4.3; old §5–6 failure/shutdown → §4.4–4.5; old §7 conflicts/tests/source route → §5–6. All six Related Pages retained.
- Full wiki link check: 452 pages, broken/ambiguous/bare_index/stale_section/orphans all zero. Changed-file math/Markdown/assets checks reported zero errors and warnings. The changed scope includes unrelated work and is not this task's owned-file count.
- Full managed MkDocs staging failed on an existing unrelated link from slime/01 to slime/11's removed `3.2.3 full+disk 的版本属于 RayTrainGroup` heading. The same issue was already recorded by other work in the changelog before this refactor. It was not repaired in this task.
- Isolated rendering build uses the real full inventory for wiki-link resolution, the normal `_render_page` conversion, shipped theme/assets/extensions, and only this page plus the vLLM and master indices as rendered sources. External destination HTML is intentionally omitted and its not-found validation suppressed only in that temporary build. `mkdocs build --strict` passed there; this is page-rendering evidence, not a full-site link/orphan pass.
- Four Mermaid blocks were rendered with the shipped Mermaid runtime and inspected; final wave identity, idle guard and ownership-view revisions were re-rendered and independently re-viewed. Diagrams are stored as source blocks in the page; temporary PNGs were review evidence only.
- No GPU tests, online inference, multi-host communications or fault-injection experiments were run. State/score examples and the identity-type finding are static analysis.

- Scoped MathJax check: PASS, the one selected page contains no MathJax formulas.
- Browser check of isolated built HTML: four Mermaid SVGs rendered, seven H2 headings present, no leaked wiki-link markup, and no page JavaScript errors. Top-page layout visually inspected.
