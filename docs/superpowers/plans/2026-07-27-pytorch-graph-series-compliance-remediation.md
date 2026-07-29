# PyTorch Graph Series Compliance Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> where tasks can be isolated by file ownership; use inline execution for shared generated
> artifacts. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every repository-internal gap against
`docs/superpowers/specs/2026-07-23-pytorch-graph-learning-series-design.md`, produce honest
environment-gated evidence for native CPU/GPU-only requirements, and publish one final delivery
report whose conclusions are derived from machine-readable audit data.

**Architecture:** Replace the rendered Markdown ledger as a data source with a canonical JSONL
ledger built from stable structural/claim records. Validate every destination as a real file plus
real H2/H3 heading, audit every legacy and new-course claim against pinned source/runtime evidence,
then extend the Lab suite for real stage hooks, AOT-to-Inductor continuity and measured memory.
Markdown reports, indexes and changelog are derived human-facing views; they never become the input
for status arithmetic.

**Tech Stack:** Python 3.13 `unittest`, PyTorch `2.9.1+cpu`, pinned PyTorch source checkout
`e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`, Markdown/Obsidian wikilinks, JSON/JSONL, PowerShell.

## Global Constraints

- Preserve every historical page body; annotations and correction callouts may be added, but old
  explanatory content must not be deleted.
- Do not modify `raw/`.
- Do not create, stage or commit Git commits; the user explicitly requested an uncommitted delivery.
- Treat `E:/97-codes/torch_parallel/p` at
  `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52` as source truth.
- Keep source-baseline facts separate from runtime observations on PyTorch
  `2.9.1+cpu` / `5811a8d7da873dd699ff6687092c225caffcf1bb`.
- Never convert a missing compiler/GPU into a pass. Native CPU and Triton evidence must record
  actual compile/execute/measurement results from a compatible environment.
- Every destination is a repository-relative Markdown path plus an exact H2/H3 heading text and
  optional occurrence index; bare filenames and unvalidated slugs are forbidden in canonical data.
- Every production-code change follows red-green-refactor; human prose is reviewed and mechanically
  validated but is not tested by brittle exact-text assertions.
- Every page retains a non-empty final `## Related Pages` section and the six domain indexes plus
  changelog remain synchronized.

---

### Task 1: Canonical Audit Records and Markdown Parser

**Files:**
- Modify: `docs/audits/pytorch_graph_series/tools/audit_graph_docs.py`
- Modify: `docs/audits/pytorch_graph_series/tools/test_audit_graph_docs.py`

**Interfaces:**
- Consumes: Markdown pages and the existing manifest/semantic-decision JSON files.
- Produces:
  - `parse_markdown(path) -> list[dict[str, object]]`
  - `build_ledger_rows(records, decisions, claim_decisions) -> list[dict[str, object]]`
  - `write_ledger_jsonl(path, rows) -> None`
  - `read_ledger_jsonl(path) -> list[dict[str, object]]`
  - `summarize_ledger(rows) -> dict[str, object]`

- [ ] **Step 1: Add failing parser tests**

  Add tests with hand-written fixtures for:

  ```python
  def test_table_with_escaped_pipe_is_one_table_and_preserves_cells(): ...
  def test_pipe_prose_and_pipe_inside_fence_are_not_markdown_tables(): ...
  def test_inline_and_reference_images_are_inventory_units(): ...
  def test_lab_heading_creates_one_experiment_span(): ...
  def test_prose_paragraphs_become_claim_candidates(): ...
  ```

  The table fixture must include ``| op | `\|=` |`` and assert logical column count without
  computing the expected result through production helpers.

- [ ] **Step 2: Verify the new tests fail for missing record kinds**

  Run:

  ```powershell
  python -m unittest `
    docs.audits.pytorch_graph_series.tools.test_audit_graph_docs.ParseMarkdownTests -v
  ```

  Expected: failures because `markdown_table`, `image`, `experiment` and `claim_candidate` records
  do not exist.

- [ ] **Step 3: Implement structural parsing**

  Implement a single-pass state machine that:

  - keeps current H2/H3 ownership;
  - treats a pipe block as a table only when a header is followed by a valid delimiter row;
  - recognizes only unescaped pipe separators;
  - records one table span with header/data row counts;
  - records Markdown images outside fences;
  - records explicit `Lab`/`Experiment`/`实验` heading regions as experiment spans;
  - records non-empty prose paragraphs outside fences/tables as claim candidates;
  - preserves Mermaid as both a code-fence record and a figure classification without duplicate
    ledger identity.

- [ ] **Step 4: Verify parser tests pass and legacy tests remain green**

  Run:

  ```powershell
  python -m unittest discover `
    -s docs/audits/pytorch_graph_series/tools `
    -p test_audit_graph_docs.py -v
  ```

- [ ] **Step 5: Add failing canonical-ledger tests**

  Add tests proving:

  - JSONL rows round-trip text containing `|`, backticks and newlines;
  - Markdown rendering is never parsed for status totals;
  - duplicate stable IDs and overlapping claim decisions fail;
  - an unreviewed claim candidate cannot acquire a verified status from a broad range decision;
  - JSONL kind/status totals equal the CLI summary.

- [ ] **Step 6: Implement canonical rows and JSONL authority**

  Stable IDs include page path, page hash, kind and source span. Range decisions remain a migration
  compatibility layer for structural units; claim decisions must identify an explicit claim ID or
  exact source span. Render `coverage_ledger.md` from canonical rows only.

- [ ] **Step 7: Run the full audit-tool suite**

  Run the command from Step 4 and require zero failures.

---

### Task 2: Validated Destination Schema and One-Time Migration

**Files:**
- Create: `docs/audits/pytorch_graph_series/2026-07-27/destination_aliases.json`
- Create: `docs/audits/pytorch_graph_series/2026-07-27/legacy_unit_decisions.jsonl`
- Modify: `docs/audits/pytorch_graph_series/tools/audit_graph_docs.py`
- Modify: `docs/audits/pytorch_graph_series/tools/test_audit_graph_docs.py`
- Modify: `docs/audits/pytorch_graph_series/2026-07-23/semantic_decisions_aotautograd.json`
- Modify: `docs/audits/pytorch_graph_series/2026-07-23/semantic_decisions_foundations.json`
- Modify: `docs/audits/pytorch_graph_series/2026-07-23/semantic_decisions_inductor.json`
- Modify: `docs/audits/pytorch_graph_series/2026-07-23/semantic_decisions_passes.json`

**Interfaces:**
- Consumes: legacy string destinations and the actual 21 course pages/index.
- Produces:
  - `Destination(path, anchor_text, anchor_occurrence)`
  - `validate_destinations(rows, repo_root) -> list[dict[str, object]]`
  - a migration command that is idempotent and refuses ambiguous mappings.

- [ ] **Step 1: Add failing destination validation tests**

  Test valid exact heading, missing path, missing anchor, duplicate heading without occurrence,
  old numbered alias normalization and the removed
  `20_debugging_observability_and_verification_labs.md` case.

- [ ] **Step 2: Verify failures occur for the missing schema/validator**

  Run the destination test class and require failures caused by absent destination objects.

- [ ] **Step 3: Implement heading-index and destination validation**

  Index exact H2/H3 text from target Markdown. A destination passes only when its path exists and
  its heading text resolves to exactly one occurrence, or the supplied occurrence selects one.

- [ ] **Step 4: Build the alias file**

  Map `00_...` through `21_...` aliases to current repository-relative files. Map the two renamed
  pages explicitly:

  ```json
  {
    "12_fx_graph_editing_primitives.md": "12_fx_graph_editing_primitives_and_invariants.md",
    "13_patternexpr_and_patternmatcher.md": "13_pattern_expression_and_matcher_engine.md"
  }
  ```

  Do not mechanically map the removed debugging page; split its source decisions across current
  debugging/Lab headings after semantic review.

- [ ] **Step 5: Migrate every existing decision to destination objects**

  Replace comma strings with arrays of exact path+heading objects. For topics explicitly retained
  outside the 21-page mainline, point to the retained page's own exact H2/H3 plus its role note.

- [ ] **Step 6: Verify destination closure**

  Regenerate canonical rows and assert:

  ```text
  bare_destination_strings = 0
  missing_destination_paths = 0
  missing_destination_anchors = 0
  ambiguous_destination_anchors = 0
  unassigned_structural_units = 0
  ```

---

### Task 3: Complete the 28-Page Historical Semantic Audit

**Files:**
- Create: `docs/audits/pytorch_graph_series/2026-07-27/legacy_claim_decisions/*.jsonl`
- Modify: the 28 pages in
  `docs/audits/pytorch_graph_series/2026-07-23/audit_manifest.json`
- Modify: correction reports under
  `docs/audits/pytorch_graph_series/2026-07-23/`

**Interfaces:**
- Consumes: claim candidates, pinned PyTorch source, current Labs and existing correction IDs.
- Produces: one explicit decision for every non-trivial legacy claim and every table/image/Lab
  structural unit.

- [ ] **Step 1: Freeze audit batches**

  Split by non-overlapping page ownership:

  1. AOTAutograd and FX/export;
  2. decomposition/pre-grad/joint/post-grad/pass methodology;
  3. lowering/Scheduler/codegen/autotune/Inductor overviews;
  4. runtime memory, activation checkpointing and compile cache.

- [ ] **Step 2: Audit each claim with source-faithful evidence**

  For every claim candidate, assign exactly one status:

  - `verified-current`: pinned current source/test supports the qualified statement;
  - `verified-historical`: evidence identifies the historical baseline;
  - `corrected`: the correction report and local callout state the replacement;
  - `unresolved`: evidence is insufficient and the claim remains visibly quarantined.

  An unresolved legacy claim may be retained only when its destination action is `retain-quarantined`
  and no new authoritative claim depends on it.

- [ ] **Step 3: Audit every table, image and experiment**

  Record its role, evidence status, destination and action. Experiments require command, environment,
  observed result and artifact; otherwise mark them historical/unverified and keep them out of
  authoritative course evidence.

- [ ] **Step 4: Add local correction callouts**

  Add `[!correction]` or `[!contradiction]` at the affected historical section, preserving the
  original text. Every `corrected` decision references a local callout and a consolidated correction
  ID.

- [ ] **Step 5: Verify the historical gate**

  Require:

  ```text
  manifest_pages = 28
  unaudited_claim_candidates = 0
  unassigned_structural_units = 0
  unresolved_claims_imported_by_new_series = 0
  corrected_without_local_callout = 0
  destination_validation_errors = 0
  ```

---

### Task 4: Prove Every New-Course Implementation Claim

**Files:**
- Create: `docs/audits/pytorch_graph_series/2026-07-27/course_claim_decisions/*.jsonl`
- Create: `docs/audits/pytorch_graph_series/2026-07-27/course_claim_ledger.jsonl`
- Create: `docs/audits/pytorch_graph_series/2026-07-27/course_claim_summary.md`
- Modify: `wiki/02_engineering/01_ai_frameworks/16_graph_compiler_foundations/*.md`

**Interfaces:**
- Consumes: claim candidates from 21 pages, pinned source and runtime artifacts.
- Produces: explicit `[S]`, `[R]`, `[I]`, `[M]` or `[B]` evidence classification per claim.

- [ ] **Step 1: Generate claim candidates without assigning truth**

  Produce stable claim IDs and source spans. Do not infer verification from the mere presence of a
  locator somewhere else in the section.

- [ ] **Step 2: Review pages in four non-overlapping batches**

  - Part I: 01–06;
  - Part II: 07–11;
  - Part III: 12–16;
  - Part IV: 17–21.

  Open every cited source range before approving an `[S]` claim. `[R]` claims name the exact script,
  artifact and runtime baseline. `[I]` claims list their supporting claim IDs. `[M]` and `[B]`
  remain explicit limitations.

- [ ] **Step 3: Repair unsupported prose**

  Add a verified locator, narrow the statement, label it as inference, or remove the unsupported
  assertion while preserving the learning objective. Do not attach a nearby locator that does not
  support the sentence.

- [ ] **Step 4: Normalize the evidence legend**

  Fix the index inconsistency by defining all five semantic classes used by the claim ledger and
  explaining how the runtime-only `[M]`/`[B]` boundary relates to `[R]`.

- [ ] **Step 5: Verify the course-claim gate**

  Require:

  ```text
  claim_candidates_without_decision = 0
  source_claims_with_invalid_locator = 0
  inference_claims_without_parent_evidence = 0
  runtime_claims_without_artifact = 0
  blocked_claims_presented_as_executed = 0
  ```

---

### Task 5: Real Stage Placement, Continuous AOT-to-Inductor and Memory Evidence

**Files:**
- Create: `wiki/02_engineering/01_ai_frameworks/16_graph_compiler_foundations/labs/part2_continuous_aot_inductor.py`
- Create: `wiki/02_engineering/01_ai_frameworks/16_graph_compiler_foundations/labs/part3_real_stage_hooks.py`
- Create: `wiki/02_engineering/01_ai_frameworks/16_graph_compiler_foundations/labs/part2_activation_peak.py`
- Modify: `wiki/02_engineering/01_ai_frameworks/16_graph_compiler_foundations/labs/test_series_contract.py`
- Modify: `wiki/02_engineering/01_ai_frameworks/16_graph_compiler_foundations/labs/README.md`
- Modify: relevant Part II/III pages and artifact manifest.

**Interfaces:**
- Produces:
  - real `torch.compile` pre-grad/post-grad hook observations;
  - a single invocation where the partition-produced fw/bw GraphModules are the direct inputs to
    the recorded Inductor GraphLowering segment;
  - measured saved-tensor live-byte peak and separately labelled Scheduler static estimate.

- [ ] **Step 1: Write failing contract tests**

  Tests require:

  - hook callbacks actually invoked by Inductor driver code;
  - the same lab origin tokens observed at AOT partition output, compiler callback and
    GraphLowering input;
  - explicit object/owner identity transitions;
  - saved-tensor live-byte peak from runtime pack/unpack events;
  - no claim that this equals a CUDA caching-allocator peak.

- [ ] **Step 2: Verify failures before adding scripts**

  Run the three new test methods and confirm failures are missing-script/missing-evidence failures.

- [ ] **Step 3: Implement real stage-hook Lab**

  Install supported `torch._inductor.config` custom graph passes under a scoped patch, execute
  `torch.compile`, capture op/target spelling and match counts, then restore configuration. Use the
  same semantic rewrite with correct and incorrect stage contracts and assert second-run
  idempotence.

- [ ] **Step 4: Implement continuous AOT-to-Inductor Lab**

  Wrap the AOT partition and fw/bw compiler callbacks, pass those exact GraphModules into
  GraphLowering/Scheduler capture, and record token continuity. Stop before native codegen if the
  local compiler is unavailable; continuity through Scheduler must remain real.

- [ ] **Step 5: Implement measured activation peak**

  Use runtime saved-tensor pack/unpack hooks to record live logical bytes and peak live bytes for
  high/low recompute budgets. Keep physical allocator metrics in a separate nullable field.

- [ ] **Step 6: Run the expanded contract suite**

  Require all old and new tests to pass with fresh artifact output.

---

### Task 6: Native CPU and GPU/Triton Environment Gate

**Files:**
- Create: `wiki/02_engineering/01_ai_frameworks/16_graph_compiler_foundations/labs/native_backend_contract.py`
- Create: `wiki/02_engineering/01_ai_frameworks/16_graph_compiler_foundations/labs/native_backend_environment.schema.json`
- Create: `wiki/02_engineering/01_ai_frameworks/16_graph_compiler_foundations/labs/NATIVE_BACKEND_RUNBOOK.md`
- Modify: `wiki/02_engineering/01_ai_frameworks/16_graph_compiler_foundations/labs/test_series_contract.py`

**Interfaces:**
- Consumes: a machine with a supported native C++ compiler and, for Triton, a CUDA-capable device.
- Produces: signed-by-hash environment/result JSON with compile, execution, numerical, performance,
  allocator and autotune evidence.

- [ ] **Step 1: Add failing schema/contract tests**

  Reject a result that lacks compiler path/version, CUDA device identity, generated-source hash,
  numerical comparison, timing protocol, allocator trace or autotune candidate/result/cache data.
  A blocked local probe is a valid diagnostic artifact but not a passing native result.

- [ ] **Step 2: Implement the environment probe and exact runbook**

  The CPU path compiles and executes pointwise/reduction kernels and compares eager outputs. The GPU
  path runs Triton autotuning, records candidates, winner, timings and cache artifact. Both record
  warmup/iteration counts and software/hardware versions.

- [ ] **Step 3: Run every capability available on the current host**

  Record the current Windows host as missing MSVC/CUDA without promoting it to pass. If a compatible
  WSL/remote environment is already available, run the contract there without changing shared
  system packages. Otherwise retain this task as an explicit external evidence gate in the final
  report.

---

### Task 6A: Number the Final Learning Path from 00 through 21

**Files:**
- Rename: `wiki/02_engineering/01_ai_frameworks/16_graph_compiler_foundations/index.md`
  to `00_pytorch_graph_series_index.md`
- Rename: the 21 course pages to `01_...md` through `21_...md`
- Modify: every affected wikilink, Related Pages section, domain index, audit destination and Lab
  page-to-script mapping.

**Interfaces:**
- Produces: a lexicographically sortable `00` index plus `01`–`21` learning path.
- Preserves: exact course semantics, backlinks and validated destination anchors.

- [ ] **Step 1: Freeze the old-to-new filename map**

  Require one unique destination for each of the current 22 files. Keep explicit aliases for the
  two historically renamed topics and refuse the removed debugging-page alias.

- [ ] **Step 2: Rename files and synchronize visible numbering**

  Put the same number in filename, H1 title, index table and previous/next navigation. Do not rely
  on an index-only number because alphabetical knowledge-base views must retain learning order.

- [ ] **Step 3: Rewrite and validate every inbound reference**

  Update wikilinks, Markdown links, Related Pages, historical migration destinations, Lab README
  mappings, six domain indexes and changelog.

- [ ] **Step 4: Verify numbering closure**

  Require:

  ```text
  numbered_course_files = 22
  missing_or_duplicate_numbers = 0
  stale_unnumbered_course_links = 0
  dangling_links_after_rename = 0
  destination_validation_errors = 0
  ```

---

### Task 7: Regenerate Reports, Indexes and Final Delivery Report

**Files:**
- Regenerate: `docs/audits/pytorch_graph_series/2026-07-23/inventory.jsonl`
- Regenerate: `docs/audits/pytorch_graph_series/2026-07-23/inventory_summary.md`
- Regenerate: `docs/audits/pytorch_graph_series/2026-07-23/coverage_ledger.md`
- Create: `docs/audits/pytorch_graph_series/2026-07-27/final_delivery_report.md`
- Modify: `docs/audits/pytorch_graph_series/2026-07-26/design_conformance_review.md`
- Modify: `docs/audits/pytorch_graph_series/2026-07-23/batch_0_summary.md`
- Modify: `wiki/changelog.md`
- Modify: six domain indexes as required by changed status.

**Interfaces:**
- Consumes: canonical ledger summaries, claim summaries and Lab manifests.
- Produces: one clause-by-clause final report with `PASS`, `PARTIAL`, `FAIL` and `BLOCKED` derived
  from named evidence fields.

- [ ] **Step 1: Regenerate, never hand-edit, inventory and ledger views**

  The report reads totals from JSONL, not Markdown splitting.

- [ ] **Step 2: Correct historical status arithmetic**

  Remove the invalid `1,030 + 1 + 10` decomposition and report status/current-result dimensions
  separately.

- [ ] **Step 3: Recalculate every design clause and §12 acceptance item**

  Each status links to its proof artifact. Hardware-only gaps remain `BLOCKED` unless a compatible
  result JSON passes the native contract.

- [ ] **Step 4: Write the final delivery report**

  Include:

  - exact changed-file/content inventory;
  - corrected and newly learned mechanisms;
  - historical migration and claim-audit closure metrics;
  - full test/static/native verification commands and outputs;
  - remaining blockers, if any, with no completion claim for blocked clauses;
  - confirmation that no Git commit/staging was performed.

- [ ] **Step 5: Synchronize indexes, backlinks and changelog**

  Preserve each retained legacy page's role and point it to validated course headings.

---

### Task 8: Final Verification and Independent Review

**Files:**
- Verify all files changed by Tasks 1–7.

**Interfaces:**
- Produces: fresh evidence supporting the final delivery report.

- [ ] **Step 1: Run audit-tool tests**

  ```powershell
  python -m unittest discover `
    -s docs/audits/pytorch_graph_series/tools `
    -p test_audit_graph_docs.py -v
  ```

- [ ] **Step 2: Run all Lab contracts and every mechanism script**

  ```powershell
  python -m unittest discover `
    -s wiki/02_engineering/01_ai_frameworks/16_graph_compiler_foundations/labs `
    -p test_series_contract.py -v
  ```

  Invoke all top-level Lab scripts not directly launched by the contract and require exit code zero.

- [ ] **Step 3: Run canonical audit gates**

  Require zero invalid source locators, destinations, claim decisions, unassigned structural units,
  dangling wikilinks, unbalanced fences and Related Pages failures.

- [ ] **Step 4: Validate Mermaid and Python**

  Run Python compilation on Lab/audit source files. Run static Mermaid safety checks for all changed
  diagrams; render with an available renderer when possible and record the exact unavailable
  capability otherwise.

- [ ] **Step 5: Run repository diff checks**

  ```powershell
  git diff --check
  git diff --cached --quiet
  ```

  Require no whitespace errors and no staged changes.

- [ ] **Step 6: Independent final review**

  Compare the final report line-by-line against the original design, canonical ledgers and fresh
  test output. Any load-bearing discrepancy returns to the owning task; no report status may be
  upgraded by prose alone.
