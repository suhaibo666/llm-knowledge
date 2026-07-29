# PyTorch Graph Learning Series Batch 0 Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagent execution is disabled for this workspace; execute inline with review checkpoints.

**Goal:** Build a source-pinned, mechanically complete audit of all existing graph-related wiki material so the 21-page refactor starts from verified facts rather than inherited assumptions.

**Architecture:** Use a detached PyTorch source worktree at the current local `origin/main` commit, a manifest of legacy wiki pages, and a repeatable inventory tool. Mechanical extraction proves content coverage; manual source reading assigns `verified-current`, `verified-historical`, `corrected`, or `unresolved` and produces a correction report that gates Part I writing.

**Tech Stack:** Markdown, JSON, Python 3.12 standard library, Git, ripgrep, local PyTorch source.

## Global Constraints

- Accuracy and completeness take precedence over document length and schedule.
- Current source baseline is local `pytorch/origin/main` commit `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`, dated 2026-07-23.
- The existing dirty PyTorch checkout at `ea5655fcebf726ec4cf1a859de75d2d0e6425805` is read-only and must not be reset or cleaned.
- Audit current behavior in a detached source worktree; record historical claims separately when an older baseline can be established.
- Existing wiki text is an audit lead, not an authority.
- Every non-trivial implementation claim must be read at its current source locator before it receives `verified-current`.
- Do not delete or rewrite legacy wiki content during Batch 0.
- Preserve all pre-existing uncommitted knowledge-repo changes.
- The design spec and implementation plan remain local and uncommitted.
- Do not create Git commits unless the user explicitly requests them.

---

### Task 1: Establish the isolated source baseline and audit workspace

**Files:**
- Create: `docs/audits/pytorch_graph_series/2026-07-23/source_baseline.md`
- Create: `docs/audits/pytorch_graph_series/2026-07-23/audit_manifest.json`
- Create outside knowledge repo: `E:/97-codes/torch_parallel/p/`

**Interfaces:**
- Consumes: local PyTorch repository at `E:/97-codes/torch_parallel/pytorch`
- Produces: immutable source root recorded in `source_baseline.md`; manifest consumed by the inventory tool

- [x] **Step 1: Verify the source commit exists and record current dirty state**

Run:

```powershell
git -C E:\97-codes\torch_parallel\pytorch cat-file -e e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52^{commit}
git -C E:\97-codes\torch_parallel\pytorch status --short
```

Expected: the first command exits 0; the second lists existing submodule/untracked changes that must remain untouched.

- [x] **Step 2: Create the detached source worktree**

Run:

```powershell
git -C E:\97-codes\torch_parallel\pytorch -c core.longpaths=true worktree add --detach E:\97-codes\torch_parallel\p e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52
```

Expected: a detached worktree is created at the exact commit without changing the original checkout.

- [x] **Step 3: Verify the detached worktree**

Run:

```powershell
git -C E:\97-codes\torch_parallel\p rev-parse HEAD
git -C E:\97-codes\torch_parallel\p status --short
```

Expected: the first command prints `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`; the second prints nothing.

- [x] **Step 4: Write the source baseline record**

Record:

```markdown
# PyTorch Graph Series Audit Baseline

- Current audit source: `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
- Branch reference: local `origin/main`
- Commit date: 2026-07-23
- Source worktree: `E:/97-codes/torch_parallel/p`
- Previous report baseline: `ea5655fcebf726ec4cf1a859de75d2d0e6425805`
- Historical wiki claims without a declared commit start as `unresolved`.
- The original PyTorch checkout was dirty and was not modified.
```

- [x] **Step 5: Write the explicit legacy-page manifest**

The JSON array must contain these repository-relative pages:

```json
[
  "wiki/02_engineering/01_ai_frameworks/03_aot_autograd/aotautograd_analysis.md",
  "wiki/02_engineering/01_ai_frameworks/03_aot_autograd/aot_autograd_quickstart.md",
  "wiki/02_engineering/01_ai_frameworks/03_aot_autograd/fx_graph_construction_and_transformation_analysis.md",
  "wiki/02_engineering/01_ai_frameworks/14_fx_export_and_extensibility/fx_graph_export_and_custom_ops_analysis.md",
  "wiki/02_engineering/01_ai_frameworks/14_fx_export_and_extensibility/fx_export_custom_op_quickstart.md",
  "wiki/02_engineering/01_ai_frameworks/04_inductor/dynamic_shapes_full_analysis.md",
  "wiki/02_engineering/01_ai_frameworks/04_inductor/inductor_codegen_dynamic_shape_analysis.md",
  "wiki/02_engineering/01_ai_frameworks/04_inductor/decomposition_passes_guide.md",
  "wiki/02_engineering/01_ai_frameworks/04_inductor/pre_grad_passes_guide.md",
  "wiki/02_engineering/01_ai_frameworks/04_inductor/joint_graph_passes_guide.md",
  "wiki/02_engineering/01_ai_frameworks/04_inductor/post_grad_passes_guide.md",
  "wiki/02_engineering/01_ai_frameworks/04_inductor/torch_upstream_pass_deepdive.md",
  "wiki/02_engineering/01_ai_frameworks/04_inductor/fx_pass_optimization_methodology.md",
  "wiki/02_engineering/01_ai_frameworks/04_inductor/lowering_analysis.md",
  "wiki/02_engineering/01_ai_frameworks/04_inductor/scheduler_analysis.md",
  "wiki/02_engineering/01_ai_frameworks/04_inductor/inductor_codegen_analysis.md",
  "wiki/02_engineering/01_ai_frameworks/04_inductor/inductor_autotuning_analysis.md",
  "wiki/02_engineering/01_ai_frameworks/04_inductor/PyTorch_Inductor_Technical_Analysis.md",
  "wiki/02_engineering/01_ai_frameworks/04_inductor/torch_compile_architecture.md",
  "wiki/02_engineering/01_ai_frameworks/13_runtime_memory_amp_profiler/caching_allocator_autocast_profiler_analysis.md",
  "wiki/02_engineering/01_ai_frameworks/17_compile_cache/aotautograd_cache_analysis.md",
  "wiki/02_engineering/01_ai_frameworks/17_compile_cache/fx_graph_cache_analysis.md"
]
```

- [x] **Step 6: Verify the manifest**

Run the bundled Python interpreter to load the JSON and assert every file exists.

Expected: exit 0 and `22 manifest pages exist`.

### Task 2: Build the repeatable Markdown inventory tool

**Files:**
- Create: `docs/audits/pytorch_graph_series/tools/audit_graph_docs.py`
- Create: `docs/audits/pytorch_graph_series/tools/test_audit_graph_docs.py`

**Interfaces:**
- Consumes: `audit_manifest.json`, knowledge-repo root, detached PyTorch source root
- Produces: JSON records with page, heading path, source line, content kind, locator text, source-resolution status, wikilinks and code-fence metadata

- [x] **Step 1: Write parser tests**

Tests must cover:

- nested H2/H3 heading paths;
- fenced Python and Mermaid blocks;
- wikilinks with aliases and anchors;
- full locators such as `torch/fx/graph.py:2688-2774`;
- shorthand locators such as `runtime_wrappers.py:3215`;
- duplicate locator occurrences;
- unbalanced code fences.

- [x] **Step 2: Run tests and verify they fail**

Run:

```powershell
& C:\Users\suhaibo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest docs.audits.pytorch_graph_series.tools.test_audit_graph_docs -v
```

Expected: failure because `audit_graph_docs` is not implemented.

- [x] **Step 3: Implement the inventory tool**

The tool must expose:

```python
def parse_markdown(path: pathlib.Path) -> list[dict[str, object]]
def extract_locators(text: str) -> list[dict[str, object]]
def resolve_locator(
    locator: collections.abc.Mapping[str, object],
    source_root: pathlib.Path,
) -> dict[str, object]
def audit_manifest(
    repo_root: pathlib.Path,
    source_root: pathlib.Path,
    manifest_path: pathlib.Path,
) -> list[dict[str, object]]
```

Command-line arguments:

```text
--repo-root
--source-root
--manifest
--jsonl-output
--summary-output
--ledger-output
--decisions DECISION_FILE [DECISION_FILE ...]
```

Full-path locators resolve against the source root and verify line bounds. Shorthand paths remain `needs_manual_resolution`; they must never be guessed.

- [x] **Step 4: Run unit tests**

Expected: all parser and locator tests pass.

- [x] **Step 5: Compile-check the tool**

Run:

```powershell
& C:\Users\suhaibo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m py_compile docs\audits\pytorch_graph_series\tools\audit_graph_docs.py
```

Expected: exit 0.

### Task 3: Generate the complete mechanical inventory and ledger skeleton

**Files:**
- Create: `docs/audits/pytorch_graph_series/2026-07-23/inventory.jsonl`
- Create: `docs/audits/pytorch_graph_series/2026-07-23/inventory_summary.md`
- Create: `docs/audits/pytorch_graph_series/2026-07-23/coverage_ledger.md`

**Interfaces:**
- Consumes: Task 1 manifest and source root; Task 2 tool
- Produces: the authoritative Batch 0 work queue for manual source audit

- [x] **Step 1: Run the inventory tool**

Run with the exact repo, source, manifest and output paths.

Expected: exit 0; JSONL and summary files are non-empty.

- [x] **Step 2: Verify mechanical completeness**

Assert for each manifest page:

- one page record exists;
- every H2/H3 appears;
- every code fence appears with a balanced status;
- every Mermaid fence appears;
- every wikilink appears;
- every source-like locator occurrence appears.

Expected: all 22 pages have records; unbalanced fences or unreadable pages fail the task.

- [x] **Step 3: Generate the coverage ledger**

Each row must include:

```text
legacy_page | section | source_line | content_kind | locator |
claimed_baseline | current_result | status | destination | action | notes
```

Initial `status` is `unresolved`; mechanical path/line success is recorded separately and does not count as semantic verification.

- [x] **Step 4: Verify that no legacy unit is silently omitted**

Compare H2/H3 and code-fence counts between source pages and ledger rows.

Expected: exact count equality for all pages.

### Task 4: Audit foundational FX, export and dynamic-shape claims

**Files:**
- Modify: `docs/audits/pytorch_graph_series/2026-07-23/coverage_ledger.md`
- Create: `docs/audits/pytorch_graph_series/2026-07-23/semantic_decisions_foundations.json`
- Create: `docs/audits/pytorch_graph_series/2026-07-23/corrections_foundations.md`

**Interfaces:**
- Consumes: ledger rows for FX/export/dynamic-shape pages
- Produces: verified source map for future documents 01–07 and 12–16

- [x] **Step 1: Audit FX Node/Graph/GraphModule storage claims**

Read the current definitions and complete call paths for Node construction, args/users synchronization, linked-list ordering, lookup tables, lint, DCE and recompile.

- [x] **Step 2: Audit capture-path claims**

Verify symbolic tracing, ProxyTensor/make_fx, Dynamo and export entry points and their actual output contracts.

- [x] **Step 3: Audit FakeTensor, metadata, SymInt, ShapeEnv, guards and constraints**

Separate graph-resident values from external guard/signature state; flag statements that collapse these layers.

- [x] **Step 4: Classify every audited row**

Use only:

```text
verified-current
verified-historical
corrected
unresolved
```

Each non-unresolved row requires a source locator and one-sentence evidence.

- [x] **Step 5: Write the correction report**

For each correction, include the old claim, current evidence, whether the old baseline is known, severity, and destination document.

### Task 5: Audit AOTAutograd, joint/fw/bw and recompute claims

**Files:**
- Modify: `docs/audits/pytorch_graph_series/2026-07-23/coverage_ledger.md`
- Create: `docs/audits/pytorch_graph_series/2026-07-23/semantic_decisions_aotautograd.json`
- Create: `docs/audits/pytorch_graph_series/2026-07-23/corrections_aotautograd.md`

**Interfaces:**
- Consumes: AOTAutograd legacy rows
- Produces: verified source map for future documents 05 and 08–11

- [x] **Step 1: Audit metadata collection and functionalization**

Verify mutation/view/subclass metadata, synthetic-base behavior and functional graph contracts.

- [x] **Step 2: Audit joint graph capture**

Verify primals/tangents, create_joint, autograd.grad, node tagging and joint outputs.

- [x] **Step 3: Audit partition and graph extraction**

Verify classify closures, fresh placeholders, old-to-new env mapping, fw/bw signatures and absence of cross-Graph Node edges.

- [x] **Step 4: Audit saved tensors, min-cut and recompute**

Verify flow-network construction, forced-save/recompute constraints, cut interpretation, node copying and backward reorder.

- [x] **Step 5: Audit runtime ABI**

Verify saved tensors/SymInts/opaque objects/tangents ordering and forward/backward wrapper assembly.

- [x] **Step 6: Classify rows and write corrections**

Do not mark examples verified unless their commands were run or their output is directly supported by tests.

### Task 6: Audit PatternMatcher, DCE, ordering and pass-stage claims

**Files:**
- Modify: `docs/audits/pytorch_graph_series/2026-07-23/coverage_ledger.md`
- Create: `docs/audits/pytorch_graph_series/2026-07-23/semantic_decisions_passes.json`
- Create: `docs/audits/pytorch_graph_series/2026-07-23/corrections_passes.md`

**Interfaces:**
- Consumes: pass-related legacy rows
- Produces: verified source map for future documents 12–16

- [x] **Step 1: Audit PatternExpr and MatchContext**

Verify current subclass locations, capture behavior, shared bindings, MultiOutput anchors and failure rollback.

- [x] **Step 2: Audit candidate indexing and reverse traversal**

Verify registry keys, Graph.find_nodes, candidate sorting, bucket iteration and mutation/stream boundaries.

- [x] **Step 3: Audit replacement entries**

Verify lowering, graph and replacement entry application, insertion, use replacement, node erasure and metadata preservation.

- [x] **Step 4: Audit DCE and stable topological sort**

Verify impurity, reverse cascade, nested graphs, sort data structures and stage-driver cleanup checkpoints.

- [x] **Step 5: Audit pre/decomposition/joint/post ordering**

Verify current drivers, custom hook contracts, normalization dependencies, mutation-tail rules and recompile/lint placement.

- [x] **Step 6: Re-derive complexity claims**

Derive candidate, match, replacement, DCE, sort, lint and pipeline complexity from current loops and data structures.

- [x] **Step 7: Classify rows and write corrections**

Old line numbers that merely drifted are `verified-current` with new locators, not semantic corrections.

### Task 7: Audit lowering, Inductor IR, memory, Scheduler and codegen claims

**Files:**
- Modify: `docs/audits/pytorch_graph_series/2026-07-23/coverage_ledger.md`
- Create: `docs/audits/pytorch_graph_series/2026-07-23/corrections_inductor.md`

**Interfaces:**
- Consumes: Inductor-related legacy rows
- Produces: verified source map for future documents 17–21

- [x] **Step 1: Audit GraphLowering and lowering registration**

Verify FX-to-IR environment mapping, regular lowerings, fallbacks, extern kernels and lowering-pattern passthrough.

- [x] **Step 2: Audit IR value, loop, layout and buffer objects**

Verify current live classes and identify legacy classes or renamed paths.

- [x] **Step 3: Audit liveness and memory planning**

Verify reads/writes, last-use, allocation/free/reuse and alias/mutation constraints.

- [x] **Step 4: Audit Scheduler dependency and fusion**

Verify dependency types, node construction, topological scheduling, fusion legality/profitability and reorder behavior.

- [x] **Step 5: Audit codegen, algorithm selection and provenance**

Verify backend dispatch, scheduler-group-to-kernel mapping, wrapper launch, autotuning placement, cache and debug artifacts.

- [x] **Step 6: Classify rows and write corrections**

Backend-specific claims must name the backend and must not be generalized to all Inductor codegen.

### Task 8: Consolidate the Batch 0 audit and gate Part I

**Files:**
- Create: `docs/audits/pytorch_graph_series/2026-07-23/correction_report.md`
- Create: `docs/audits/pytorch_graph_series/2026-07-23/batch_0_summary.md`
- Modify: `docs/audits/pytorch_graph_series/2026-07-23/coverage_ledger.md`

**Interfaces:**
- Consumes: all correction files and classified ledger rows
- Produces: explicit proceed/block decision for Part I implementation

- [x] **Step 1: Merge correction reports without losing provenance**

Group by severity:

```text
P0 semantic/correctness error
P1 stale architecture or removed path
P2 line/path drift with semantics intact
P3 clarity, terminology or missing boundary
```

- [x] **Step 2: Compute coverage and status totals**

Report totals by page and status. A mechanical row may not disappear during consolidation.

- [x] **Step 3: Verify Batch 0 acceptance criteria**

Required:

- all 22 manifest pages inventoried;
- 100% H2/H3 and code fences represented;
- every current-authoritative claim used by Part I has been semantically verified;
- all P0/P1 findings have planned destinations;
- no falsely verified Lab output;
- no unresolved claim is allowed into a new authoritative page.

- [x] **Step 4: Run repository checks**

Run:

```powershell
git diff --check
```

Run the inventory tool tests and regenerate the inventory once more from a clean process.

Expected: zero test failures and no diff-check errors.

- [x] **Step 5: Present the Batch 0 findings before writing Part I**

Report:

- current source baseline;
- number of audited units;
- verified/corrected/unresolved totals;
- highest-severity historical errors;
- what is safe to reuse;
- what Part I must rewrite from source.
