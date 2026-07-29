# `torch.compile` End-to-End Learning Series Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement
> this plan task-by-task in the current workspace. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a source-faithful A→F curriculum that takes a beginner from PyTorch/Python execution
prerequisites through Dynamo, the existing graph compiler course, runtime artifacts, debugging,
performance, training, distributed execution, extension and deployment.

**Architecture:** Add one parent domain under `19_torch_compile_end_to_end/`. New A/B/D/E/F pages
own the missing concepts; the existing `16_graph_compiler_foundations/00–21` remains the canonical
volume C and is linked rather than copied. Every source claim is pinned to the clean PyTorch checkout,
and final course evidence is generated independently of the rendered Markdown reports.

**Tech Stack:** Markdown/Obsidian wikilinks, Mermaid, pinned PyTorch source
`e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`, Python 3.13 audit tools, JSON/JSONL, PowerShell.

## Global Constraints

- Follow
  `docs/superpowers/specs/2026-07-28-torch-compile-end-to-end-learning-series-design.md`.
- Preserve all existing page bodies; only add pages, links, correction callouts or status notes.
- Do not modify `raw/`.
- Do not create, stage or commit Git commits.
- Use `E:/97-codes/torch_parallel/p` at
  `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52` as implementation truth.
- Separate pinned source claims from PyTorch `2.9.1+cpu` runtime observations.
- Do not add demo scripts in this phase.
- Do not claim native C++/CUDA/Triton execution without a compatible receipt.
- Every new page ends with a non-empty final `## Related Pages`.
- Every source locator is opened before use; formal source evidence spans at most 30 lines.
- Each page answers motivation, mechanism, state, real call chain, invariants, why-not,
  boundary, complexity/cost and upstream/downstream relation.

---

### Task 1: Parent Domain, Manifest and Course Contract

**Files:**
- Create:
  `wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/index.md`
- Create:
  `wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/00_torch_compile_end_to_end_index.md`
- Create:
  `docs/audits/torch_compile_end_to_end/2026-07-28/course_manifest.json`
- Create:
  `docs/audits/torch_compile_end_to_end/2026-07-28/source_map.md`
- Modify:
  `wiki/02_engineering/01_ai_frameworks/index.md`

**Interfaces:**
- Consumes: approved design, current graph-series index and sibling domain indexes.
- Produces: stable A/B/C/D/E/F display IDs, file inventory, prerequisite graph and source subsystem map.

- [ ] **Step 1: Create the domain shell**

  Add the parent index and `00` page with the six-volume map, beginner path, compiler-developer path,
  production-debugger path, evidence legend and current native backend boundary.

- [ ] **Step 2: Link volume C without copying it**

  Map `C01–C21` to the existing
  `16_graph_compiler_foundations/01–21` pages. State that display IDs do not rename files or change
  their evidence ledger.

- [ ] **Step 3: Create the exact manifest**

  List `00`, A01–A05, B01–B10, D01–D07, E01–E09 and F01–F08 in order. Include the fixed source
  baseline and the external volume-C index.

- [ ] **Step 4: Build the source subsystem map**

  Record exact entry files for API, Dynamo eval-frame, symbolic conversion, guards, resume execution,
  backend registry, AOT runtime wrappers, Inductor compile/cache/async/wrapper, CUDAGraph Trees,
  logging/repro/minifier, Compiled Autograd, distributed and AOTInductor.

- [ ] **Step 5: Validate the shell**

  Confirm all declared paths are unique, numbering is gap-free, `Related Pages` is last, local links
  resolve, source checkout HEAD is exact and `git diff --check` succeeds.

---

### Task 2: Volume A — Execution Prerequisites

**Files:**
- Create:
  `a01_tensor_storage_layout_and_views_analysis.md`
- Create:
  `a02_operator_schema_dispatch_and_autograd_analysis.md`
- Create:
  `a03_python_frames_code_objects_and_bytecode_analysis.md`
- Create:
  `a04_dispatch_modes_proxy_tensor_and_fake_tensor_analysis.md`
- Create:
  `a05_eager_capture_compile_and_replay_cost_model_analysis.md`
- Modify: parent `index.md` and `00_torch_compile_end_to_end_index.md`

All paths are under
`wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/`.

**Interfaces:**
- Consumes: TensorImpl/Storage, dispatcher, autograd, CPython-facing Dynamo C extension,
  ProxyTensor/FakeTensor and public compile entry.
- Produces: the value, operator, Python execution and cost vocabulary assumed by volume B.

- [ ] **Step 1: Map A01 source**

  Verify TensorImpl sizes/strides/storage offset, StorageImpl ownership, view/alias construction,
  version counter and mutation sites. Write the computation-value versus storage-identity model.

- [ ] **Step 2: Write A01**

  Explain why shape/layout/alias cannot be collapsed into one Tensor value, how views share storage,
  and which facts become guards, metadata or mutation constraints.

- [ ] **Step 3: Map and write A02**

  Trace Python operator → dispatcher schema/key selection → kernel → autograd wrapper/Node. Separate
  dispatcher execution edges from autograd dependency edges and explain why compiler capture can
  intercept at several layers.

- [ ] **Step 4: Map and write A03**

  Trace code object → frame → instruction/value stack/locals → eval-frame callback. Explain why the
  code object, not a transient frame, owns the reusable compilation cache.

- [ ] **Step 5: Map and write A04**

  Compare `__torch_function__`, `__torch_dispatch__`, TorchDispatchMode, ProxyTensor and FakeTensor.
  Trace one operator through proxy creation and fake metadata propagation without presenting them as
  interchangeable mechanisms.

- [ ] **Step 6: Map and write A05**

  Define eager execution, capture, graph compilation, native compilation/module load, warmup and
  replay. Give a parameterized cost model separating one-time, per-specialization and per-call cost.

- [ ] **Step 7: Validate volume A**

  Check source paths/line bounds, Mermaid syntax, A01→A05 navigation, volume-B forward links,
  `Related Pages`, no demo files and `git diff --check`.

---

### Task 3: Volume B1 — API, Context, Frame and Bytecode

**Files:**
- Create:
  `b01_torch_compile_api_and_first_call_lifecycle_analysis.md`
- Create:
  `b02_backend_modes_options_stances_and_fullgraph_analysis.md`
- Create:
  `b03_eval_frame_callback_and_code_cache_analysis.md`
- Create:
  `b04_instruction_translator_and_bytecode_state_machine_analysis.md`
- Create:
  `b05_variable_tracker_source_and_python_object_model_analysis.md`

**Interfaces:**
- Consumes: volume A vocabulary and fixed Dynamo source.
- Produces: the capture-side state model used by B06–B10.

- [ ] **Step 1: Write B01 from the public entry**

  Trace decorator/direct invocation, backend wrapper construction,
  `torch._dynamo.optimize()` context creation, first frame execution, backend callback and returned
  callable. Distinguish wrapper creation time from first-call compile time.

- [ ] **Step 2: Write B02 from option normalization**

  Explain `backend`, `mode`, `options`, `fullgraph`, `dynamic`, `dynamic_shapes`,
  `recompile_limit`, `isolate_recompiles`, `disable` and stance. Trace each option to the subsystem
  that consumes it and document mutual exclusions/fallback.

- [ ] **Step 3: Write B03 from the frame callback**

  Trace Python/C eval-frame registration, callback invocation, code/cache entry lookup,
  ConvertFrame and transformed code. Model per-code-object, per-region and per-frame state separately.

- [ ] **Step 4: Write B04 from InstructionTranslator**

  Explain instruction pointer, stack, locals/globals, block stack, speculation, inline translator and
  exception/side-effect state. Follow bytecode dispatch to VariableTracker operations and graph output.

- [ ] **Step 5: Write B05 from VariableTracker/Source**

  Build the Python-object model: constant, tensor, module, function, container, user object, lazy
  variable and source lineage. Explain identity/equality guards and why a Proxy alone cannot model
  Python state.

- [ ] **Step 6: Validate B01–B05**

  Verify the public API→eval-frame→translator call chain, link all source ranges, scan Mermaid, ensure
  no runtime claim exceeds available evidence and run Markdown diff checks.

---

### Task 4: Volume B2 — Graph Emission, Guards, Breaks, Dynamics and Backend

**Files:**
- Create:
  `b06_output_graph_side_effects_and_graph_emission_analysis.md`
- Create:
  `b07_guards_cache_lookup_and_recompilation_analysis.md`
- Create:
  `b08_graph_break_resume_functions_and_partial_graphs_analysis.md`
- Create:
  `b09_dynamic_shapes_generalization_and_fallback_analysis.md`
- Create:
  `b10_backend_contract_and_custom_backend_analysis.md`
- Modify:
  `16_graph_compiler_foundations/00_pytorch_graph_series_index.md`

**Interfaces:**
- Consumes: B01–B05 state model.
- Produces: the exact boundary into volume C and the runtime decisions needed by D/E.

- [ ] **Step 1: Write B06**

  Trace root/inline translator ownership of OutputGraph, SubgraphTracer, graph inputs by Source,
  guards, side effects, graph compile reason, GraphModule creation and backend invocation.

- [ ] **Step 2: Write B07**

  Trace GuardBuilder/CheckFunctionManager to cache-entry lookup. Separate guard creation cost,
  per-call evaluation cost, cache-list traversal and recompilation limits.

- [ ] **Step 3: Write B08**

  Trace unsupported/user graph break to partial graph compile, resume code generation,
  ContinueExecutionCache lookup and restoration of stack/locals/context state. Document fullgraph
  rejection and generator/coroutine boundaries.

- [ ] **Step 4: Write B09**

  Explain static specialization, automatic dynamic generalization, ShapeEnv guards, unbacked symbols,
  per-region and accumulated recompile limits, eager fallback and failure modes.

- [ ] **Step 5: Write B10**

  Define backend callable input GraphModule/example inputs, fake inputs, returned callable, ownership,
  mutation and exception contract. Explain registry string lookup versus direct callable.

- [ ] **Step 6: Bridge volume C**

  Add a short “端到端课程中的位置” section to the volume-C index only. Link B06/B10 before C07 and
  D01 after C21 without changing C-page numbering or claim semantics.

- [ ] **Step 7: Validate volume B and C bridge**

  Check B01–B10 navigation, backend boundary links, existing volume-C gate, source locator validity,
  Mermaid syntax and `git diff --check`.

---

### Task 5: Volume D — Compiled Artifacts, Caches and Runtime

**Files:**
- Create D01–D07 exactly as named in the design.
- Modify parent indexes and cache/runtime sibling indexes for backlinks.

**Interfaces:**
- Consumes: B10 backend contract and C09–C21 graph pipeline.
- Produces: compiled artifact/cache/runtime state used by volume E and F.

- [ ] **Step 1: Write D01**

  Trace Inductor backend wrapper → `compile_fx` → AOTAutograd → fw/bw compiler callback →
  `compile_fx_inner`. Document input GraphModule ownership and recursive config patching.

- [ ] **Step 2: Write D02**

  Trace AOT runtime wrappers, saved slots, forward callable, lazy backward compilation and backward
  callable caching. Separate AOT graph creation from runtime autograd invocation.

- [ ] **Step 3: Write D03**

  Trace async compile submission, worker/process ownership, future completion, code cache write,
  dynamic import/module load and exception propagation.

- [ ] **Step 4: Write D04**

  Build the cache hierarchy matrix: Dynamo code cache, AOTAutograd cache, FXGraph cache, Python/C++
  code cache, algorithm cache and Triton config cache. For each, record key, value, storage, lookup,
  invalidation and evidence boundary.

- [ ] **Step 5: Write D05**

  Trace generated wrapper call, input unpacking, allocation/reuse/free, extern/kernel launch, output
  assembly and mutation/alias ABI.

- [ ] **Step 6: Write D06**

  Trace CUDAGraph Tree manager roots/nodes, warmup, recording, generation, allocator checkpoint,
  replay and output-liveness invariants. Keep GPU runtime measurements blocked.

- [ ] **Step 7: Write D07**

  Unify artifact states: not compiled, compiling, generated, cached, loaded, callable, invalidated,
  failed and fallback. Map exceptions to their owning layer.

- [ ] **Step 8: Validate volume D**

  Check cache terminology against `17_compile_cache`, runtime links, no generated-only→executed
  evidence upgrade, source ranges, Mermaid and Markdown gates.

---

### Task 6: Volume E — Observability, Correctness and Performance

**Files:**
- Create E01–E09 exactly as named in the design.
- Modify Dynamo/Inductor/runtime index backlinks.

**Interfaces:**
- Consumes: B capture state, C graph state and D artifact/runtime state.
- Produces: diagnosis and acceptance procedures used by volume F and production readers.

- [ ] **Step 1: Write E01**

  Map logging registrations, counters, explain output, compile debug traces, graph dumps, generated
  source and profiler artifacts. State the strongest claim each artifact can support.

- [ ] **Step 2: Write E02**

  Build an explain/graph-break decision flow from break reason to bytecode instruction, user source,
  partial graphs and resume function. Separate intentional break, unsupported operation and error.

- [ ] **Step 3: Write E03**

  Trace guard failure to cache miss/recompile/fallback. Provide a state-based diagnosis sequence for
  shape, type, identity, alias, global state and dynamic generalization.

- [ ] **Step 4: Write E04**

  Build the AOT/Inductor failure-localization ladder: metadata, functionalization, partition,
  pre/joint/post-grad pass, lowering, scheduler, codegen, compile/load and runtime.

- [ ] **Step 5: Write E05**

  Trace repro generation, after-Dynamo/after-AOT minifier loops and CompilerBisector subsystem
  selection. Explain preservation requirements for guards, inputs and failure signatures.

- [ ] **Step 6: Write E06**

  Define correctness gates for forward values, dtype/shape, gradients, alias, mutation, RNG/effect,
  dynamic shapes, exception behavior and idempotence.

- [ ] **Step 7: Write E07**

  Define cold process, cold cache, warm cache, warmup and steady-state measurements. Parameterize
  total cost and break-even call count without inventing hardware numbers.

- [ ] **Step 8: Write E08**

  Connect graph count, fusion groups, kernel launches, layout, bandwidth, occupancy, allocator peak
  and synchronization. Separate source inspection from native profiling.

- [ ] **Step 9: Write E09**

  Define staged rollout, eager shadow comparison, cache/version isolation, fallback, monitoring,
  artifact retention and rollback gates.

- [ ] **Step 10: Validate volume E**

  Ensure commands/config names are fixed-source valid, no obsolete environment variables are copied
  from old pages, every decision flow reaches an evidence boundary, links and Mermaid pass.

---

### Task 7: Volume F — Training, Distributed, Extensions and Deployment

**Files:**
- Create F01–F08 exactly as named in the design.
- Modify AOTAutograd, distributed, FX/export/custom-op and backend indexes for backlinks.

**Interfaces:**
- Consumes: all earlier volumes.
- Produces: advanced end-to-end application and extension model.

- [ ] **Step 1: Write F01**

  Contrast AOTAutograd with Compiled Autograd. Trace autograd compiler registration, capture,
  dynamic policy, generated backward graph, compiler callback and disable/reset boundaries.

- [ ] **Step 2: Write F02**

  Separate user activation checkpoint, saved tensor hooks, AOT min-cut rematerialization and
  CUDAGraph/runtime memory. Explain double-recompute and side-effect/RNG constraints.

- [ ] **Step 3: Write F03**

  Trace DDP optimization/backend wrapping, reducer/buckets, backward overlap, optimizer regions and
  compiled-autograd compatibility.

- [ ] **Step 4: Write F04**

  Explain FSDP parameter materialization, DTensor placements, collective effects, process-group state,
  rank-dependent guards and graph/runtime boundaries.

- [ ] **Step 5: Write F05**

  Trace custom-op schema/registration, fake/meta implementation, autograd formula, functionalization,
  decomposition and Inductor lowering/fallback. Provide a completeness matrix without adding a demo.

- [ ] **Step 6: Write F06**

  Trace backend registry, callable contract, DeviceInterface, backend scheduling registration,
  lowering/codegen and out-of-tree version-coupling boundaries.

- [ ] **Step 7: Write F07**

  Compare JIT `torch.compile` and AOTInductor capture/package/load paths, graph signature, constants,
  ABI, target environment, cache and deployment lifecycle.

- [ ] **Step 8: Write F08**

  Compare training/inference state, freezing, mutation, parameter treatment, CUDAGraph eligibility and
  dynamic-shape tradeoffs. End with a Transformer-oriented source-level decision tree.

- [ ] **Step 9: Validate volume F**

  Check distributed/custom-op/AOTI source entrypoints, private-versus-public API labels, native/GPU
  blocks, backlinks, Mermaid and Markdown gates.

---

### Task 8: Evidence Ledger, Index Integration and Delivery Report

**Files:**
- Create:
  `docs/audits/torch_compile_end_to_end/2026-07-28/course_claim_decisions/`
- Create:
  `docs/audits/torch_compile_end_to_end/2026-07-28/course_claim_ledger.jsonl`
- Create:
  `docs/audits/torch_compile_end_to_end/2026-07-28/course_claim_summary.md`
- Create:
  `docs/audits/torch_compile_end_to_end/2026-07-28/course_claim_errors.json`
- Create:
  `docs/audits/torch_compile_end_to_end/2026-07-28/final_delivery_report.md`
- Modify: framework index, sibling indexes, `wiki/changelog.md`

**Interfaces:**
- Consumes: all final pages, fixed source and existing formal runtime receipts.
- Produces: machine-readable evidence closure and human-facing delivery report.

- [ ] **Step 1: Generate atomic claim candidates**

  Reuse the graph-series Markdown parser for paragraphs, list items, table rows and code fences.
  Preserve page path, line span, text hash and stable claim ID.

- [ ] **Step 2: Decide every claim**

  Assign `[S]/[R]/[I]/[M]/[B]` or nonassertive. Verify `[S]` locators in the clean pinned checkout;
  bind `[I]` to existing parent claims with rationale; use runtime evidence only when an existing
  receipt exactly supports the claim.

- [ ] **Step 3: Build and validate the ledger**

  Require one decision per claim, no stale hashes, no missing source path/range, no inference cycles
  and zero validation errors.

- [ ] **Step 4: Integrate navigation**

  Add the new domain to `01_ai_frameworks/index.md`; add backlinks from Dynamo, AOT, Inductor,
  runtime, distributed, custom-op and compile-cache indexes; verify zero dangling wikilinks.

- [ ] **Step 5: Update changelog**

  Record the exact volumes/pages completed, source SHA, evidence counts, demo freeze and native
  backend limitations.

- [ ] **Step 6: Write the final delivery report**

  Summarize changed files, knowledge added, misconceptions corrected, evidence counts, validation
  results, blocked capabilities and recommended reading paths.

---

### Task 9: Full Verification

**Files:**
- Verify all files from Tasks 1–8; do not add implementation content.

**Interfaces:**
- Consumes: final workspace state.
- Produces: fresh evidence for completion claims.

- [ ] **Step 1: Verify structure**

  Check exact numbered inventory, H1/display IDs, final `Related Pages`, reciprocal index links,
  balanced fences and no empty Mermaid blocks.

- [ ] **Step 2: Verify locators**

  Resolve every repository-relative source path in the pinned checkout, validate line bounds and
  formal evidence spans of at most 30 lines.

- [ ] **Step 3: Verify evidence**

  Rebuild the claim ledger into official outputs and require zero errors.

- [ ] **Step 4: Verify audit tools**

  Run:

  ```powershell
  python -m unittest discover -s docs/audits/pytorch_graph_series/tools -p 'test_*.py' -v
  ```

  Expected: exit 0 and no failed tests.

- [ ] **Step 5: Verify repository safety**

  Run `git diff --check`, require an empty Git index, confirm `raw/` was not modified, and confirm
  the pinned source checkout is clean at the exact SHA.

- [ ] **Step 6: Reconcile reports**

  Compare every number in the final report and changelog with generated JSON/JSONL artifacts. Replace
  stale intermediate counts before claiming completion.

## Plan Self-Review

- The plan covers every file and acceptance criterion in the approved design.
- No page is named ambiguously and no volume has a numbering gap.
- B10/F06 and C/D responsibilities are explicitly separated.
- No task adds demos or requires unavailable native hardware.
- Evidence, integration and verification are first-class tasks rather than final prose-only checks.
- Git commit steps are intentionally absent because the user requires an uncommitted delivery.
- The user selected inline direct execution in advance, so no additional execution-choice pause is
  required.
