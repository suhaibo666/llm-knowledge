# torch.compile Volume Demos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build six CUDA-first, volume-level teaching Demo entry points with multiple independently runnable cases, structured evidence, full A–F page mapping, and regression tests.

**Architecture:** A shared standard-library harness owns CLI parsing, capability detection, status semantics, environment snapshots, and JSON output. Six volume modules register focused cases; C delegates to the existing audited scripts, while A/B/D/E/F add new mechanisms. A JSON manifest is the single mapping from all 60 teaching pages to executable cases and is enforced by tests.

**Tech Stack:** Python 3.13, PyTorch 2.9.1+cpu for local preflight, CUDA-first runtime contract, `unittest`, JSON artifacts, Markdown/Obsidian links.

## Global Constraints

- PyTorch source claims remain pinned to `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`.
- Default Demo device is `cuda`; unavailable required capabilities produce `BLOCKED`, never `PASS`.
- Exit codes are 0 PASS, 2 FAIL, 3 BLOCKED without failure, and 4 contract/CLI error.
- No new third-party dependencies.
- Existing C scripts and artifacts are reused, not rewritten.
- Every A/B/D/E/F page gets a `配套 Demo` section before final `Related Pages`.
- Every one of the 60 body pages appears exactly once as a primary mapping in `demo_manifest.json`.
- Current local execution must not claim CUDA, multi-GPU, Triton, native compiler, or AOTI success.

---

### Task 1: Shared Harness and Manifest Contract

**Files:**
- Create: `tools/labs_torch_compile/test_volume_demo_contract.py`
- Create: `tools/labs_torch_compile/demo_harness.py`
- Create: `tools/labs_torch_compile/demo_manifest.json`

**Interfaces:**
- Produces: `CaseSpec`, `DemoContext`, `CaseResult`, `CapabilitySnapshot`, `run_volume_cli(volume, cases, argv=None) -> int`.
- Produces: result schema `torch-compile-volume-demo/v1`.
- Consumes: only Python standard library plus `torch`.

- [ ] **Step 1: Write the failing harness tests**

```python
class HarnessContractTest(unittest.TestCase):
    def test_missing_cuda_is_blocked_before_case_body_runs(self):
        called = False
        def body(context):
            nonlocal called
            called = True
            return {"unexpected": True}
        result = execute_case(
            CaseSpec("cuda_case", ("a01",), ("cuda",), body),
            fake_context(cuda_available=False),
        )
        self.assertEqual(result.status, "BLOCKED")
        self.assertFalse(called)

    def test_exception_is_fail_and_is_not_swallowed(self):
        result = execute_case(
            CaseSpec("broken", ("a01",), (), lambda _: 1 / 0),
            fake_context(cuda_available=False),
        )
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(result.error["type"], "ZeroDivisionError")
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest wiki.02_engineering.01_ai_frameworks.19_torch_compile_end_to_end.labs.test_volume_demo_contract -v
```

Expected: import failure because `demo_harness.py` does not exist.

- [ ] **Step 3: Implement the minimal harness**

Implement frozen case metadata, environment/capability detection, pre-execution requirement checks, exception capture, deterministic JSON writing, `--list`, repeated `--case`, device selection, seed setup, and the four exit codes.

- [ ] **Step 4: Run harness tests and verify GREEN**

Expected: harness-only tests pass; volume tests remain absent.

### Task 2: Volume A and B Cases

**Files:**
- Modify: `.../labs/test_volume_demo_contract.py`
- Create: `.../labs/demo_a_execution_model.py`
- Create: `.../labs/demo_b_dynamo_capture.py`

**Interfaces:**
- Consumes: `demo_harness.CaseSpec`, `DemoContext`, `run_volume_cli`.
- Produces: `CASES: tuple[CaseSpec, ...]` in each module.

- [ ] **Step 1: Add failing registry and behavior tests**

Assert exact A/B case IDs from the design, unique IDs, correct page IDs, CPU execution of `tensor_storage_layout`, `dispatcher_autograd`, `compile_lifecycle`, `guards_recompile`, and CUDA blocking of `eager_compile_cost`.

- [ ] **Step 2: Run the focused tests and verify RED**

Expected: imports fail because volume modules do not exist.

- [ ] **Step 3: Implement A cases**

Use actual tensors to assert storage pointers/strides/aliasing, `TorchDispatchMode` to observe dispatch, autograd gradients, `dis`/frame inspection, FakeTensor/make_fx graphs, and CUDA events for first-call versus steady-state timing.

- [ ] **Step 4: Implement B cases**

Use backend callables and real `torch.compile` calls to record graph counts, `torch._dynamo.explain` for guards/breaks, shape changes for recompilation, explicit graph breaks, dynamic export constraints, and a backend contract that checks GraphModule/example inputs without mutating them.

- [ ] **Step 5: Run A/B tests and verify GREEN**

Expected: device-independent cases pass on CPU; CUDA performance case returns `BLOCKED`.

### Task 3: Volume C Orchestrator

**Files:**
- Modify: `.../labs/test_volume_demo_contract.py`
- Create: `.../labs/demo_c_graph_compiler.py`

**Interfaces:**
- Produces: six C cases.
- Subprocess receipt: command, exit code, stdout, stderr, artifact paths.

- [ ] **Step 1: Add failing C delegation tests**

Create a temporary child script that prints a key/value and exits 0; assert the orchestrator preserves command/stdout/exit code. Add a failing child and assert the case becomes `FAIL`.

- [ ] **Step 2: Verify RED**

Expected: C module import failure.

- [ ] **Step 3: Implement C groups**

Map each group to the exact existing scripts from the design. Run children from the knowledge-root working directory with `-B`, capture streams, and preserve exit code. Do not translate a child failure into a blocked result.

- [ ] **Step 4: Verify GREEN**

Expected: delegation contract passes without running all expensive children.

### Task 4: Volume D Runtime and Artifact Cases

**Files:**
- Modify: `.../labs/test_volume_demo_contract.py`
- Create: `.../labs/demo_d_artifact_runtime.py`

**Interfaces:**
- Produces: seven D cases and JSON observations for compile counts, cache directories, wrapper outputs, CUDA allocator values, CUDAGraph replay, and lifecycle states.

- [ ] **Step 1: Add failing D tests**

Assert registry completeness, CPU-safe lifecycle state transitions, and `BLOCKED` for CUDA allocator/CUDAGraph cases without CUDA.

- [ ] **Step 2: Verify RED**

Expected: D module import failure.

- [ ] **Step 3: Implement D cases**

Use a scoped temporary cache, custom backend counters, real forward/backward calls, cache file snapshots, CUDA memory statistics, `torch.compile(mode="reduce-overhead")`, `torch.compiler.cudagraph_mark_step_begin`, and explicit lifecycle transition validation.

- [ ] **Step 4: Verify GREEN**

Expected: state and contract cases pass locally; GPU-only cases block.

### Task 5: Volume E Diagnostic Cases

**Files:**
- Modify: `.../labs/test_volume_demo_contract.py`
- Create: `.../labs/demo_e_diagnostics.py`

**Interfaces:**
- Produces: nine E cases with diagnostic observations and bounded failure injection.

- [ ] **Step 1: Add failing E tests**

Assert registry, real eager/compiled correctness on CPU, controlled backend failure localization, fallback state transitions, and CUDA blocking for timing/profiler cases.

- [ ] **Step 2: Verify RED**

Expected: E module import failure.

- [ ] **Step 3: Implement E cases**

Capture `torch._dynamo.explain`, backend graph counts, controlled failures at backend/runtime boundaries, a minimized deterministic reproducer artifact, output/gradient/alias comparison, CUDA event timing, profiler summaries, and fallback/error-budget decisions.

- [ ] **Step 4: Verify GREEN**

Expected: diagnostic contracts pass locally; GPU-only metrics block.

### Task 6: Volume F Advanced Cases

**Files:**
- Modify: `.../labs/test_volume_demo_contract.py`
- Create: `.../labs/demo_f_advanced_topics.py`

**Interfaces:**
- Produces: eight F cases; requirements distinguish CUDA, multi-GPU, distributed, Linux, Triton, and native compiler.

- [ ] **Step 1: Add failing F tests**

Assert exact registry/page mapping, CPU-safe custom-op/custom-backend behavior, and capability blocking for DDP/FSDP/AOTI/CUDAGraph cases.

- [ ] **Step 2: Verify RED**

Expected: F module import failure.

- [ ] **Step 3: Implement F cases**

Use Compiled Autograd configuration in a scoped context, activation checkpointing, DDP/FSDP/DTensor preconditions, `torch.library.custom_op` with fake/autograd registration, backend callables, AOTInductor package APIs, inference/freezing, and CUDAGraph mode. Restore all global configuration in `finally`.

- [ ] **Step 4: Verify GREEN**

Expected: custom contracts pass locally; unavailable GPU/multi-GPU/native cases block with exact reasons.

### Task 7: Manifest and Markdown Integration

**Files:**
- Modify: `.../labs/test_volume_demo_contract.py`
- Modify: `.../labs/README.md`
- Modify: `.../00_torch_compile_end_to_end_index.md`
- Modify: `.../index.md`
- Modify: all 39 `a??_`, `b??_`, `d??_`, `e??_`, `f??_` course pages
- Modify: `wiki/changelog.md`

**Interfaces:**
- Consumes: six `CASES` registries and `demo_manifest.json`.
- Produces: one primary case mapping for every A–F body page.

- [ ] **Step 1: Add failing mapping tests**

```python
def test_manifest_covers_every_body_page_and_markdown_links_case(self):
    body_pages = expected_body_pages_from_course_manifest()
    mappings = load_demo_manifest()
    self.assertEqual(set(mappings), body_pages)
    for page_id, mapping in mappings.items():
        text = page_path(page_id).read_text(encoding="utf-8")
        self.assertIn("## 配套 Demo", text) if page_id[0] in "abdef" else None
        self.assertIn(mapping["case"], text)
```

- [ ] **Step 2: Verify RED**

Expected: manifest/page link coverage failures.

- [ ] **Step 3: Populate manifest and docs**

Add one concise section per A/B/D/E/F page before `Related Pages`: entry, case, command, expected observations, and capability boundary. Update Labs README with the six entry points and all cases. Update both course indexes and append a changelog entry.

- [ ] **Step 4: Verify GREEN**

Expected: all 60 body pages map to existing cases and every new page section points to its primary case.

### Task 8: Evidence Ledgers and Full Verification

**Files:**
- Modify: `docs/audits/torch_compile_end_to_end/2026-07-28/course_claim_decisions/*`
- Modify: `docs/audits/torch_compile_end_to_end/2026-07-28/course_claim_ledger.jsonl`
- Modify: `docs/audits/torch_compile_end_to_end/2026-07-28/course_claim_summary.md`
- Modify: `docs/audits/torch_compile_end_to_end/2026-07-28/course_claim_errors.json`
- Modify: `docs/audits/torch_compile_end_to_end/2026-07-28/final_delivery_report.md`

**Interfaces:**
- Consumes: final Markdown and pinned source checkout.
- Produces: exact claim-to-decision closure with zero validation errors.

- [ ] **Step 1: Rebuild A–F decisions**

Run `build_course_decisions.py` using the unchanged C ledger as the graph-ledger input.

- [ ] **Step 2: Validate the unified claim ledger**

Expected: claim count equals decision count; validation errors 0.

- [ ] **Step 3: Run Demo tests**

```powershell
python -m unittest discover `
  -s tools\labs_torch_compile `
  -p "test_volume_demo_contract.py" -v
```

- [ ] **Step 4: Run existing regression suites**

```powershell
python -m unittest discover -s docs\audits\pytorch_graph_series\tools -p "test_*.py" -v
python -m unittest discover -s tools\labs_torch_compile -p "test_*.py" -v
```

- [ ] **Step 5: Run structure/link validation**

Validate numbering, final `Related Pages`, fences, Mermaid, Markdown links, wikilinks, manifest uniqueness, and old C-directory absence.

- [ ] **Step 6: Record the delivery report**

State exact PASS/BLOCKED counts. Explicitly say CUDA acceptance remains pending until an actual GPU run.

