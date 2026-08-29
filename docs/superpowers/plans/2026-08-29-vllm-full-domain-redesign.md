# vLLM Full Knowledge-Domain Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Waves 2–6 of the approved vLLM redesign so every page has a unique mechanism owner, explains design rationale and state transitions rather than copying source, and is verified against the frozen latest source baseline.

**Architecture:** Rewrite one approved page contract at a time from entry/control boundaries through resource/device hot paths, model specialization, scale/production mechanisms, and finally the usage guide. Each page owns one reader question and one mechanism boundary; whole-domain navigation, backlinks, coverage, counts, radar, and changelog are integrated only after all owner pages are stable.

**Tech Stack:** Markdown, Obsidian wikilinks, Mermaid, PowerShell, Git, vLLM Python/CUDA source plus same-commit tests and design documents.

**Spec:** `docs/superpowers/specs/2026-08-29-vllm-latest-codebase-analysis-design.md`

## Global Constraints

- Knowledge-base worktree: `E:/97-codes/torch_parallel/llm-knowledge/.worktrees/push-skills-20260829`; branch: `codex/vllm-full-redesign`.
- Frozen source worktree: `E:/97-codes/torch_parallel/.worktrees/vllm-6b110bad`; required `HEAD`: `6b110badbb22d3f66c7218b71138f13b7a6b3419`; it is read-only for this plan.
- Read `CLAUDE.md`, `skills/planning-codebase-analysis/SKILL.md`, `skills/source-faithful-analysis/SKILL.md`, `skills/source-faithful-analysis/references/codebase.md`, `skills/maintaining-llm-knowledge/SKILL.md`, and the figure/Mermaid skills before authoring. The approved spec overrides any old mechanical 500-line proposal rule.
- Execute page tasks sequentially. A later page may consume an earlier page's ownership boundary, but it must not duplicate the earlier mechanism proof.
- Every page begins with its reader question, central thesis, owned concepts, excluded concepts, and exact source baseline. Its body follows background → why this design → causal mechanism/state transition → invariants, costs, fallback/failure boundary → source-backed outlook only when evidence exists.
- Source code is evidence, not the prose skeleton. Keep only the smallest excerpt that proves a guard, ordering, ABI, or state transition. The deletion test must leave enough prose to explain the mechanism and why the design beats the obvious alternative.
- Every non-trivial source-derived claim carries an exact backticked locator such as `vllm/v1/core/sched/scheduler.py:123-160`. Use current code/tests for behavior, same-commit design docs for declared intent, and label reconstructed rationale as analysis/inference.
- Every page ends with 3–7 annotated `Related Pages` links. Summaries link to the unique owner rather than re-explaining the owned mechanism.
- Use `apply_patch` for authored changes. Stage only the task's explicit files. Never touch the user's dirty root worktree or the mutable vLLM checkout.
- Diagram tasks must write a one-paragraph figure specification before the Mermaid block. Diagram labels explain responsibility/state, not source directories. Re-read changed Mermaid against both project figure skills.
- The controller provisions `.superpowers/sdd/2026-08-29-vllm-full-domain-redesign/validate-vllm-page.ps1`. Every page task invokes it with the exact page path and minimum diagram count. The validator must verify the frozen source `HEAD`, inline locator existence/bounds, exact baseline presence, absence of old baseline hashes, required Related Pages count, and Mermaid minimum. It does not perform a repository-wide scan.
- Each page report records at least three load-bearing locators manually opened and compared with the prose, plus the deletion-test result and any code/doc conflict.
- Validation scope is limited to `wiki/02_engineering/03_infer_frameworks/vllm/`, directly changed parent/master indexes, direct backlinks/assets, the vLLM rows in the knowledge radar, `wiki/changelog.md`, this plan/spec, and frozen-source locators. Do not run full `python -m pytest tools/`, full strict Wiki link checking, full docs tests, or unrelated diagnostics.
- Page tasks do not update coverage/counts/changelog. Task 23 owns navigation and migration metadata after all page contracts are stable; Task 24 owns the final scoped gate and coverage reconciliation.

## Page Task Completion Protocol

Every Task 1–22 performs these exact steps with the task-specific values below:

- [ ] Verify the source worktree `HEAD` is the exact baseline and its status is empty.
- [ ] Open the existing page completely when it exists; inspect the specified live source, relevant tests, guards/assertions/fallbacks, and same-commit design docs before drafting.
- [ ] Use `apply_patch` to rewrite/create only the named page. Preserve valuable verified insights, but replace source-order narration, oversized excerpts, mechanical function lists, unsupported trends, and stale baseline claims.
- [ ] Write any required figure specification and diagram, then run the page validator, `git diff --check -- <page>`, a heading/locator scan, and the prose deletion test.
- [ ] Commit only the named page with the exact commit message. The task reviewer must approve both contract compliance and explanation quality before the next page starts.

---

### Task 1: Rebuild the System Design Principles Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/02_vllm_system_design_principles_analysis.md`

**Contract:** Explain why dynamic request arrivals force continuous scheduling, paged state, asynchronous execution, and capability contracts. Own global constraints/design pivots; exclude the full layer map and subsystem call chains.

**Evidence entry points:** `vllm/config/`, `vllm/v1/core/`, `vllm/v1/metrics/`, `docs/design/`, and tests that expose capacity, backpressure, or compatibility boundaries.

- [ ] Build a bottleneck → naïve alternative → design pivot → paid cost causal chain for each principle. Include one compact causal diagram; do not repeat the six-layer architecture from `03`.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/02_vllm_system_design_principles_analysis.md' -MinDiagramCount 1`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild system design principles" -- wiki/02_engineering/03_infer_frameworks/vllm/02_vllm_system_design_principles_analysis.md`.

### Task 2: Add the Request Semantics Owner Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/04_vllm_request_semantics_analysis.md`

**Contract:** Explain how Generate, Pooling, Render, Transcription, Realtime, and protocol variants become Engine inputs and are reconstructed as user outputs. Own task/protocol/render/input/output semantics; exclude admission and GPU sampling.

**Evidence entry points:** `vllm/tasks.py`, `vllm/entrypoints/`, `vllm/renderers/`, `vllm/v1/engine/input_processor.py`, `vllm/v1/engine/output_processor.py`, task-specific tests.

- [ ] Create a mechanism-first page with a request-semantics conversion diagram, explicit shared-versus-task-specific contracts, live/legacy boundaries, and failure behavior for incompatible inputs/capabilities.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/04_vllm_request_semantics_analysis.md' -MinDiagramCount 1`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): add request semantics analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/04_vllm_request_semantics_analysis.md`.

### Task 3: Rebuild the Engine Architecture Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/10_vllm_engine_architecture_analysis.md`

**Contract:** Explain why Client, EngineCore, and Executor are separate and where resource promises become committed state. Own internal object/process seams; exclude codebase-wide layering and HTTP semantics.

**Evidence entry points:** `vllm/v1/engine/llm_engine.py`, `async_llm.py`, `core_client.py`, `core.py`, `vllm/v1/executor/`, and lifecycle/failure tests.

- [ ] Replace old request-walkthrough duplication with an object/process ownership diagram and a precise create → submit → core step → result commit path. Distinguish frontend request state, core request state, and executor work.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/10_vllm_engine_architecture_analysis.md' -MinDiagramCount 1`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild engine ownership analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/10_vllm_engine_architecture_analysis.md`.

### Task 4: Rebuild the Serving Control Plane Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/16_vllm_serving_control_plane_analysis.md`

**Contract:** Explain how launcher, API server, DP coordinator, and Core clients manage startup, routing, backpressure, readiness, and fault domains. Own serving topology/lifecycle; exclude request rendering details.

**Evidence entry points:** launchers and serve CLI under `vllm/entrypoints/`, DP coordinator, `core_client.py`, fault-tolerance paths and tests.

- [ ] Organize the page around startup/ready/failure transitions and process ownership. Add the required launcher → API → Core → Worker readiness/failure-propagation diagram.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/16_vllm_serving_control_plane_analysis.md' -MinDiagramCount 1`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild serving control plane analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/16_vllm_serving_control_plane_analysis.md`.

### Task 5: Rebuild the Scheduler Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/11_vllm_scheduler_analysis.md`

**Contract:** Explain how one schedule step performs a multi-resource admission transaction and commits progress after output. Own waiting/running, token/encoder/spec budgets, preemption, and finish; exclude the physical runner.

**Evidence entry points:** `vllm/v1/core/sched/`, request state, scheduler output/update methods, and scheduler tests.

- [ ] Explain state and invariants before loops. Add a request-state diagram and one admission transaction showing budget reservation, KV allocation, scheduled work, output update, preempt/finish rollback or release.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/11_vllm_scheduler_analysis.md' -MinDiagramCount 2`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild scheduler transaction analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/11_vllm_scheduler_analysis.md`.

### Task 6: Rebuild the KV Cache Management Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/12_vllm_kv_cache_management_analysis.md`

**Contract:** Explain how logical/physical blocks, prefix cache, hybrid layouts, and local offload preserve ownership inside one Engine. Exclude cross-Engine transfer.

**Evidence entry points:** KV cache manager, block pool, KV cache utilities, local offload, and their tests under `vllm/v1/core/` and `tests/v1/`.

- [ ] Lead with block identity/refcount/free/evict invariants and allocation transaction boundaries. Add the required logical ↔ physical block and hash/refcount/free/evict state diagram.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/12_vllm_kv_cache_management_analysis.md' -MinDiagramCount 1`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild kv cache ownership analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/12_vllm_kv_cache_management_analysis.md`.

### Task 7: Rebuild the Model Runner V2 Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/15_vllm_model_runner_v2_analysis.md`

**Contract:** Explain how default MRV2 uses persistent rows, staged writes, and async-first execution to rebuild the device hot path, and exactly when V1 remains or is selected. Own device request state and buffer/graph lifecycle; exclude global admission.

**Evidence entry points:** `vllm/v1/worker/gpu/`, `gpu_worker.py`, the legacy runner, MRV2 design docs, and MRV2 tests.

- [ ] Explain persistent-row invariants and visibility points before buffer manipulation. Add the required CPU step N+1 versus GPU step N overlap timeline and a sourced MRV2/V1 capability boundary.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/15_vllm_model_runner_v2_analysis.md' -MinDiagramCount 1`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild model runner v2 analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/15_vllm_model_runner_v2_analysis.md`.

### Task 8: Add the Sampling and Structured Output Owner Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/17_vllm_sampling_structured_output_analysis.md`

**Contract:** Explain how logits pass through processors, penalties, top-k/top-p, grammar masks, and sampling while maintaining a valid distribution. Own token selection and structured constraints; exclude detokenization/protocol output.

**Evidence entry points:** `vllm/v1/sample/`, `vllm/v1/structured_output/`, sampling config, scheduler/runner seams, and sampling/grammar tests.

- [ ] Create a page that distinguishes per-request mutable grammar state from tensor transforms and shows invalid-token masking/order constraints. Add the required logits → constraints → sample pipeline/state diagram.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/17_vllm_sampling_structured_output_analysis.md' -MinDiagramCount 1`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): add sampling and structured output analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/17_vllm_sampling_structured_output_analysis.md`.

### Task 9: Add the Multimodal Execution Owner Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/18_vllm_multimodal_execution_analysis.md`

**Contract:** Explain how media parsing, processor cache, encoder budgets/cache, and device runner transform media into model input. Own preprocessing/feature/encoder state; exclude concrete VLM network architecture.

**Evidence entry points:** `vllm/multimodal/`, `vllm/v1/worker/gpu/mm/`, model multimodal interfaces, scheduler seams, and multimodal tests.

- [ ] Create a page that names cache keys/owners, scheduling budgets, and feature-to-token alignment invariants. Add the required media → processor/cache → encoder → model-input diagram.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/18_vllm_multimodal_execution_analysis.md' -MinDiagramCount 1`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): add multimodal execution analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/18_vllm_multimodal_execution_analysis.md`.

### Task 10: Rebuild the Model Library Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/13_vllm_model_library_analysis.md`

**Contract:** Explain how Registry, unified construction ABI, weight mapping, parallel layers, and LoRA attachment form an executable model. Own the model/weight ABI; exclude per-step state and quantization kernels.

**Evidence entry points:** model registry, model loader, linear/parallel layers, LoRA integration, representative models, and loader tests.

- [ ] Reframe model examples as proofs of the common ABI. Explain registration → class resolution → construction → weight mapping/commit → optional LoRA, including partial/stacked parameter failure boundaries. A compact construction diagram is optional; if added, require one Mermaid block.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/13_vllm_model_library_analysis.md' -MinDiagramCount 0`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild model library abi analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/13_vllm_model_library_analysis.md`.

### Task 11: Rebuild the Attention Backends Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/14_vllm_attention_backends_analysis.md`

**Contract:** Explain how metadata, KV layout, and capability negotiation let dynamic scheduling select specialized attention implementations. Own the attention contract/backend selection; exclude global scheduling and kernel internals.

**Evidence entry points:** `vllm/v1/attention/`, attention layers/selectors, backend implementations, and selector/capability tests.

- [ ] Explain the stable contract and capability predicates before cataloguing backends. Identify fallback/rejection behavior and why metadata is the seam between scheduling and kernels. A compact contract diagram is optional.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/14_vllm_attention_backends_analysis.md' -MinDiagramCount 0`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild attention backend contract analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/14_vllm_attention_backends_analysis.md`.

### Task 12: Rebuild the Speculative Decoding Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/20_vllm_speculative_decoding_analysis.md`

**Contract:** Explain when draft cost beats target serial steps and how propose/verify/accept preserve sampling distribution and KV correctness. Own speculative contracts; exclude the general sampler.

**Evidence entry points:** spec-decode modules, scheduler seams, MRV2 sampling/graph paths, acceptance methods, and correctness tests.

- [ ] Center the page on propose → target score → accept/reject → rollback/commit and the economic break-even conditions. Add one draft/verify state-flow diagram and link general sampling details to `17`.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/20_vllm_speculative_decoding_analysis.md' -MinDiagramCount 1`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild speculative decoding analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/20_vllm_speculative_decoding_analysis.md`.

### Task 13: Rebuild the Quantization Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/21_vllm_quantization_analysis.md`

**Contract:** Explain why format, scale, load transform, hardware kernel, and fallback are one joint decision. Own the quantization ABI/dispatch; exclude generic weight loading and kernel programming detail.

**Evidence entry points:** quantization configs/methods/layers/kernels, model-loader integration, platform capability checks, and quantization tests.

- [ ] Organize by configure → create/load transformed parameters → post-load process → runtime dispatch/fallback. Explain compatibility predicates and correctness/performance tradeoffs. A three-stage load diagram is optional.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/21_vllm_quantization_analysis.md' -MinDiagramCount 0`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild quantization abi analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/21_vllm_quantization_analysis.md`.

### Task 14: Rebuild the Compilation and CUDA Graph Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/23_vllm_compilation_cudagraph_analysis.md`

**Contract:** Explain how a dynamic-shape system obtains compilable, capturable, address-stable execution regions. Own compile/graph lifecycle; exclude IR operation semantics and specific kernels.

**Evidence entry points:** `vllm/compilation/`, MRV2 CUDA-graph utilities, compile configuration, design docs, and compilation/graph tests.

- [ ] Explain shape partitioning, address stability, capture pools, dispatch keys, invalidation/fallback, and eager/compile/piecewise/full modes. Add one execution-mode dispatch/lifecycle diagram.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/23_vllm_compilation_cudagraph_analysis.md' -MinDiagramCount 1`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild compilation and cudagraph analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/23_vllm_compilation_cudagraph_analysis.md`.

### Task 15: Rebuild the Fused Operators and Kernels Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/24_vllm_fused_ops_and_kernels_analysis.md`

**Contract:** Explain when fusion/specialized kernels reduce launches or memory traffic and when fallback is required. Own provider/kernel families and the benefit model; exclude IR pass ordering.

**Evidence entry points:** kernels, fused MoE, custom-op/provider selection, platform dispatch, and kernel tests/benchmarks.

- [ ] Replace kernel cataloguing with a cost model: eliminated intermediates/launches versus shape, dtype, hardware, workspace, and maintainability constraints. Explain provider selection/fallback with representative families only; an optional selection diagram may be added.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/24_vllm_fused_ops_and_kernels_analysis.md' -MinDiagramCount 0`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild fused kernel mechanism analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/24_vllm_fused_ops_and_kernels_analysis.md`.

### Task 16: Rebuild the IR and Fusion Passes Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/25_vllm_ir_and_fusion_passes_analysis.md`

**Contract:** Explain how IR preserves stable semantics, expresses donation/alias, and orders rewrites/lowering safely. Own IR/pass/functionalization; exclude whole-model compilation strategy.

**Evidence entry points:** `vllm/ir/`, compilation passes, custom ops, functionalization/alias metadata, design docs, and pass tests.

- [ ] Explain invariants that each pass consumes/produces, why ordering matters, and how unsafe alias/donation transforms are rejected or bounded. Add one pass-pipeline diagram.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/25_vllm_ir_and_fusion_passes_analysis.md' -MinDiagramCount 1`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild ir and fusion pass analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/25_vllm_ir_and_fusion_passes_analysis.md`.

### Task 17: Rebuild the Distributed Inference Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/22_vllm_distributed_inference_analysis.md`

**Contract:** Explain how TP/PP/DP/EP/CP, DBO, and executors map to rank state and collective order. Own parallel/communication semantics; exclude online weight-update transactions.

**Evidence entry points:** `vllm/distributed/`, executors/coordinators, parallel-state initialization, DBO paths, and distributed tests.

- [ ] Separate logical dimensions from process/rank ownership and communication sequencing. Add both the required rank/group mapping diagram and the DBO overlap timeline, including ordering/deadlock invariants.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/22_vllm_distributed_inference_analysis.md' -MinDiagramCount 2`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild distributed inference analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/22_vllm_distributed_inference_analysis.md`.

### Task 18: Rebuild the Disaggregated KV Serving Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/26_vllm_disaggregated_kv_serving_analysis.md`

**Contract:** Explain how transferable groups, connector, lease, and store move KV state across Engines and reclaim it on failure. Own cross-instance KV protocol; exclude single-Engine block lifecycle.

**Evidence entry points:** KV transfer connectors, transferable groups, offload/store implementations, Mooncake/NIXL/MoRIIO paths, and connector tests.

- [ ] Center the page on producer/consumer identity, metadata/control versus data transfer, readiness/completion, lease ownership, timeout/failure cleanup, and fallback. Add the required producer → store/connector → consumer lease/state diagram.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/26_vllm_disaggregated_kv_serving_analysis.md' -MinDiagramCount 1`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild disaggregated kv serving analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/26_vllm_disaggregated_kv_serving_analysis.md`.

### Task 19: Rebuild the Observability and Reliability Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/27_vllm_observability_reliability_analysis.md`

**Contract:** Explain how metrics, events, traces, and sentinels connect SLO symptoms back to resource commitments and fault domains. Own observation/failure feedback; exclude service routing.

**Evidence entry points:** V1 metrics, tracing/instrumentation, fault-tolerance sentinels, error propagation, design docs, and reliability tests.

- [ ] Reframe metric lists into a feedback loop: commitment → measurement → correlation → fault classification → mitigation/visibility. Explain cardinality, sampling, stale-signal, and process-boundary costs. An optional feedback-loop diagram may be added.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/27_vllm_observability_reliability_analysis.md' -MinDiagramCount 0`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild observability reliability analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/27_vllm_observability_reliability_analysis.md`.

### Task 20: Rebuild the Extension and Plugin System Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/28_vllm_extension_plugin_system_analysis.md`

**Contract:** Explain how discovery, process scoping, and staged initialization extend the core without hidden global-state pollution. Own plugin ABI/lifecycle; exclude built-in model registry and protocol implementation.

**Evidence entry points:** plugin discovery/loader, endpoint/platform/io/LoRA-resolver plugins, initialization sites, docs, and plugin tests.

- [ ] Explain discover → select/configure → import/register → per-process initialize → teardown/failure boundaries, and why eager global mutation is unsafe. Add one plugin-lifecycle diagram if needed; otherwise prose plus a contract table must fully carry the mechanism.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/28_vllm_extension_plugin_system_analysis.md' -MinDiagramCount 0`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild extension plugin lifecycle analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/28_vllm_extension_plugin_system_analysis.md`.

### Task 21: Add the Online Weight Update Owner Page

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/29_vllm_weight_transfer_online_update_analysis.md`

**Contract:** Explain how pause/sleep, start/update/finish, weight versioning, and post-commit cache/runner work form an online-update transaction. Own vLLM-side transfer and visibility; exclude trainer algorithms and rollout orchestration.

**Evidence entry points:** distributed weight-transfer code, `LLM`/`AsyncLLM`/Core utility methods, worker/model-runner update paths, and online-update tests.

- [ ] Create a page that separates preparation, staging, validation, version commit, post-commit invalidation, resume, and failure rollback. Add the required online weight-version transaction diagram and explicitly audit in-flight request/cache/spec-draft consequences where source proves them.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/29_vllm_weight_transfer_online_update_analysis.md' -MinDiagramCount 1`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): add online weight update analysis" -- wiki/02_engineering/03_infer_frameworks/vllm/29_vllm_weight_transfer_online_update_analysis.md`.

### Task 22: Recalibrate the Feature and Optimization Guide

**File:** `wiki/02_engineering/03_infer_frameworks/vllm/01_vllm_feature_optimizations_guide.md`

**Contract:** Teach readers to run, measure, identify the limiting resource, choose a configuration family, and validate the result instead of presenting a flag catalogue. Own usage/benchmark/tuning loop; exclude internal state machines.

**Evidence entry points:** `vllm/entrypoints/llm.py`, current CLI/config help, `vllm/benchmarks/`, user-facing docs, and benchmark tests/examples at the frozen commit.

- [ ] Rebuild the guide as workload hypothesis → baseline measurement → bottleneck diagnosis → one change → correctness/performance validation → rollback. Use decision tables and links to mechanism owners; do not copy internal call flows or claim universal tuning values.
- [ ] Run `validate-vllm-page.ps1 -Page 'wiki/02_engineering/03_infer_frameworks/vllm/01_vllm_feature_optimizations_guide.md' -MinDiagramCount 0`, `git diff --check --` for that page, and record three verified locators.
- [ ] Commit with `git commit -m "docs(vllm): rebuild optimization workflow guide" -- wiki/02_engineering/03_infer_frameworks/vllm/01_vllm_feature_optimizations_guide.md`.

### Task 23: Integrate the Final Domain Navigation and Migration Metadata

**Files:**
- `wiki/02_engineering/03_infer_frameworks/vllm/index.md`
- Direct vLLM page backlinks that still point at retired ownership or omit a new owner
- `wiki/02_engineering/03_infer_frameworks/index.md`
- `wiki/index.md` only if its vLLM/domain count is materially affected
- The vLLM rows in the repository knowledge radar/source inventory
- `wiki/changelog.md`

**Contract:** Make the final 23-page vLLM domain navigable by reader question and dependency, with one owner per concept and no duplicated mechanism summaries. Record the completed source baseline and added pages without rewriting historical changelog entries.

- [ ] Inventory all 23 content pages (`01`, `02`, `03`, `04`, `10`–`18`, `20`–`29`) and classify each by reader question, owner concepts, exclusions, baseline, prerequisites, and follow-up links.
- [ ] Rewrite the domain index as a knowledge map, not a filename list. Include suggested paths for architecture, request/resource hot path, model/device optimization, and production/scale; keep mechanism prose on owner pages.
- [ ] Repair direct backlinks for new `04`, `17`, `18`, `29` owners and remove ownership-conflicting summaries. Update the parent count to `23 篇 + index`, master count only if it actually stores this count, and only vLLM-specific radar/source rows.
- [ ] Prepend one changelog entry covering Waves 2–6, the four new owner pages, mechanism-first rewrite, frozen baseline, scoped validation, and final count. Do not edit earlier entries except broken live links.
- [ ] Run `rg -n "vLLM|vllm"` only over the directly affected indexes/radar/changelog, `git diff --check --` over explicit integration paths, and commit with `git commit -m "docs(vllm): integrate full domain redesign"` using only those paths.

### Task 24: Run the Final Scoped Gate and Reconcile Coverage

**Files:**
- Verify `wiki/02_engineering/03_infer_frameworks/vllm/**`
- Verify only directly edited parent/master indexes, backlinks/assets, vLLM radar rows, and changelog entry
- Modify `docs/superpowers/specs/2026-08-29-vllm-latest-codebase-analysis-design.md`

**Contract:** Prove the approved page contracts, ownership map, source locators, diagrams, links, and final counts are closed without broad repository validation.

- [ ] Run the page validator across all 23 content pages, using required Mermaid minima from Tasks 1–22. Fail on wrong source `HEAD`, stale baselines, missing/out-of-bounds locators, missing ownership headers, insufficient Related Pages, or a missing required diagram.
- [ ] Mechanically scan every vLLM page for exact baseline, locator syntax, stale old commit hashes, duplicate H1/title problems, empty/oversized code dumps, bare `[[index]]`, and retired slugs/assets. Resolve only findings inside this domain.
- [ ] Run a scoped wikilink resolver for every Markdown page under the vLLM directory plus only directly edited parent/master files. Resolve wiki-root, current-page-relative, and unique-suffix targets; do not use the full strict checker.
- [ ] Render or parser-check every changed Mermaid block and visually inspect each required figure against the figure specifications. Run page-local math checking only for pages with added formulas.
- [ ] Reconcile the spec coverage matrix so every contract completed by this plan is `covered (Wave 2)` through `covered (Wave 6)` as appropriate; do not alter unresolved evidence questions unless this implementation actually closed them with cited proof.
- [ ] Run `git diff --check` only for the vLLM directory and directly changed integration/spec files; confirm the dirty root worktree is untouched and the frozen source worktree remains clean.
- [ ] Commit only the spec coverage reconciliation with `git commit -m "docs(vllm): record full redesign coverage" -- docs/superpowers/specs/2026-08-29-vllm-latest-codebase-analysis-design.md`.
