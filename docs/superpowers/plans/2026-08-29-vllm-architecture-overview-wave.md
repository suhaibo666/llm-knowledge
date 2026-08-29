# vLLM Architecture Overview Wave Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved Wave 0/1 exemplar by replacing the DeepSeek-centric request walkthrough with a source-verified vLLM architecture overview that explains static responsibility layers before the online request lifecycle.

**Architecture:** Work from a detached, read-only vLLM worktree pinned to `6b110badbb22d3f66c7218b71138f13b7a6b3419`. Rewrite only the `03` owner page first, then repair its direct backlinks and migration metadata; later page waves remain gated on user acceptance of this exemplar. The overview owns whole-system boundaries and one representative lifecycle, while subsystem internals remain summaries linked to their existing owner pages.

**Tech Stack:** Markdown, Obsidian wikilinks, Mermaid flowchart/sequence diagrams, PowerShell, Git, vLLM Python/CUDA source and same-commit design docs/tests.

**Spec:** `docs/superpowers/specs/2026-08-29-vllm-latest-codebase-analysis-design.md`

## Global Constraints

- Approved source baseline: `vllm-project/vllm@6b110badbb22d3f66c7218b71138f13b7a6b3419`, `main`, 2026-08-29.
- Do not move, clean, reset, stash, or modify the user's current vLLM checkout at `E:/97-codes/torch_parallel/vllm`.
- Use a detached source worktree at `E:/97-codes/torch_parallel/.worktrees/vllm-6b110bad`; verify its `HEAD` before every locator audit.
- The page must present static structure first and dynamic motion second; directory layout and call order are evidence, not the architecture taxonomy.
- The six approved responsibility layers are interface/semantics, engine lifecycle, resource control, distributed execution, device runtime, and model/operators; cross-cutting data planes are shown separately.
- Do not add a standalone “where to add features” section. Capabilities, input→output contracts, owned state, non-owned responsibilities, and boundaries must make placement inferable.
- Source excerpts are evidence, not the narrative skeleton. The deletion test must leave a coherent explanation of why each boundary exists, how state moves, and what breaks if ownership is blurred.
- Do not use page length, a 500-line threshold, layer count, code-block count, or code/explanation ratio as a completion gate; split decisions follow concept ownership and reader load.
- Current behavior comes from code/tests at the frozen commit; same-commit design docs explain intent. Label reconstructed rationale as inference and surface code/doc conflicts explicitly.
- Keep one authoritative page per mechanism. `03` may summarize Scheduler, KV, MRV2, model, serving, and parallelism only to explain their layer contracts, then link to their owner pages.
- Each changed content page ends with 3–7 annotated Related Pages links and at least one existing-page link.
- Use `apply_patch` for authored changes. A rename may use `git mv`; delete tracked obsolete assets only after confirming no live page references them.
- Validation is limited to `wiki/02_engineering/03_infer_frameworks/vllm/`, the directly edited parent index/backlink/changelog files, the two changed Mermaid blocks, and source locators in the new `03` page.
- Do not run full `python -m pytest tools/`, full `python tools/check_links.py --strict`, full `npm run docs:test`, or any unrelated repository-wide diagnostic.
- Preserve unrelated user changes by working only in `E:/97-codes/torch_parallel/llm-knowledge/.worktrees/push-skills-20260829` and staging explicit task paths.

---

### Task 1: Replace the DeepSeek Walkthrough with the Architecture Exemplar

**Files:**
- Read: `E:/97-codes/torch_parallel/vllm/**` at commit `6b110badbb22d3f66c7218b71138f13b7a6b3419`
- Rename: `wiki/02_engineering/03_infer_frameworks/vllm/03_vllm_request_flow_walkthrough_analysis.md` → `wiki/02_engineering/03_infer_frameworks/vllm/03_vllm_architecture_overview_analysis.md`
- Delete: `wiki/02_engineering/03_infer_frameworks/vllm/assets/deepseek_v3_inference_flow_interactive.html`
- Delete: `wiki/02_engineering/03_infer_frameworks/vllm/assets/deepseek_v3_inference_flow_interactive.js`

**Interfaces:**
- Consumes: the page contract in spec §9.1, the six-layer ownership map in spec §6, and the online lifecycle in spec §7.2.
- Produces: a page titled `vLLM 架构概览：责任分层、状态所有权与请求生命周期`, with the new basename `03_vllm_architecture_overview_analysis` for Task 2 backlinks.

- [ ] **Step 1: Create or verify the frozen source worktree without touching the dirty checkout**

Run from `E:/97-codes/torch_parallel/vllm`:

```powershell
$sourceWorktree = 'E:/97-codes/torch_parallel/.worktrees/vllm-6b110bad'
$baseline = '6b110badbb22d3f66c7218b71138f13b7a6b3419'
git rev-parse --verify "$baseline`^{commit}"
if (-not (Test-Path -LiteralPath $sourceWorktree)) {
  git worktree add --detach $sourceWorktree $baseline
}
git -C $sourceWorktree rev-parse HEAD
git -C $sourceWorktree status --short
```

Expected: `HEAD` prints the exact baseline and status is empty. If the path already exists with another `HEAD`, stop this task; do not move or delete it.

- [ ] **Step 2: Read the load-bearing source ranges before drafting**

Locate and open the current symbols in these exact source areas:

```powershell
$sourceWorktree = 'E:/97-codes/torch_parallel/.worktrees/vllm-6b110bad'
rg -n "class LLM|def generate|class AsyncLLM|add_request|EngineCoreRequest|class EngineCore|def step|class Scheduler|def schedule|class GPUWorker|class GPUModelRunner|class ModelRegistry" `
  "$sourceWorktree/vllm/entrypoints/llm.py" `
  "$sourceWorktree/vllm/v1/engine/async_llm.py" `
  "$sourceWorktree/vllm/v1/engine/llm_engine.py" `
  "$sourceWorktree/vllm/v1/engine/core_client.py" `
  "$sourceWorktree/vllm/v1/engine/core.py" `
  "$sourceWorktree/vllm/v1/core/sched/scheduler.py" `
  "$sourceWorktree/vllm/v1/worker/gpu_worker.py" `
  "$sourceWorktree/vllm/v1/worker/gpu" `
  "$sourceWorktree/vllm/model_executor/models/registry.py"
```

Also inspect `vllm/tasks.py`, `vllm/v1/engine/input_processor.py`, `vllm/v1/engine/output_processor.py`, `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/executor/`, `vllm/distributed/`, `vllm/v1/attention/`, and `docs/design/arch_overview.md`. Record exact line ranges only after opening them at this worktree.

- [ ] **Step 3: Write the two figure specifications before drawing**

Use these approved figure contracts verbatim as the drafting checklist:

```text
Figure A — Static responsibility map
Top-to-bottom layers: interface/semantics → engine lifecycle → resource control → distributed execution → device runtime → model/operators.
Each layer label names its capability and owned state in one short phrase.
Solid arrows carry request/plan/result contracts; a side rail names KV transfer, online weight update, plugins, metrics, and fault tolerance as cross-cutting planes.
The diagram must not show source directories as layers and must not include a feature-modification catalogue.

Figure B — Online request lifecycle
Protocol request → renderer/input processor → frontend RequestState plus EngineCoreRequest → EngineCoreClient → Scheduler/KV admission → executor/worker/device runner → model/attention/sampler → Scheduler commit → OutputProcessor → user stream.
Mark the three ownership transitions: frontend state becomes valid before core submission; Scheduler owns admission/KV progress; output becomes externally visible only after core result commit and frontend processing.
Keep subsystem internals collapsed and link them from prose.
```

- [ ] **Step 4: Rename and rewrite the page around responsibility rather than source order**

Run `git mv` for the Markdown filename, then use `apply_patch` to replace the body. The finished page must contain this structure:

```text
YAML title
H1 title
Baseline / dimension / central thesis / ownership boundary
§1 Background and central thesis
§2 Static architecture: Figure A plus layer contract table
§3 Layer-by-layer design logic: six subsections, each covering why it exists, capability, input→output, owned state/invariant, non-owned responsibility, smallest evidence
§4 Dynamic lifecycle: Figure B plus the real online path and visibility/commit points
§5 Cross-cutting planes: why they cross layers and where their state commits
§6 Live/legacy and failure boundaries: V1 aliases, engine-versus-runner naming, backend-dependent topology, code/doc conflicts
§7 Reading handoff: concise links to owner pages by mechanism, not a modification catalogue
Related Pages: 3–7 annotated links
```

For every layer subsection, write the problem and design choice before implementation details. Retain source code only when an exact guard, ordering, or alias is itself part of the argument. DeepSeek MLA/MoE, Python syntax notes, full function indexes, parallelism tables, and detailed startup barriers do not belong in this page.

- [ ] **Step 5: Remove the retired interactive assets after proving they have no live consumer**

Run:

```powershell
rg -n "deepseek_v3_inference_flow_interactive" wiki/02_engineering/03_infer_frameworks/vllm
```

Expected before deletion: only the renamed page and the two assets refer to the asset names. After the rewrite no Markdown page may refer to them; then remove the two tracked asset files. Historical prose in `wiki/changelog.md` is handled by Task 2 and is not a live asset consumer.

- [ ] **Step 6: Run page-local author checks**

Run:

```powershell
$page = 'wiki/02_engineering/03_infer_frameworks/vllm/03_vllm_architecture_overview_analysis.md'
rg -n "^#|^##|^###|```mermaid|6b110bad|Related Pages" $page
rg -n "DeepSeek|Python 语法|函数索引|去哪.*加|where to add|500" $page
git diff --check -- $page wiki/02_engineering/03_infer_frameworks/vllm/assets
```

Expected: the required structure and exactly two Mermaid blocks are visible; excluded DeepSeek/mechanical/feature-catalogue/length-rule material has no hit; diff check is clean. Re-read each Mermaid block against `skills/writing-mermaid-diagrams/SKILL.md` and both figures against `skills/drawing-wiki-figures/SKILL.md`.

- [ ] **Step 7: Commit only the exemplar page and retired assets**

```powershell
git add -A -- `
  wiki/02_engineering/03_infer_frameworks/vllm/03_vllm_request_flow_walkthrough_analysis.md `
  wiki/02_engineering/03_infer_frameworks/vllm/03_vllm_architecture_overview_analysis.md `
  wiki/02_engineering/03_infer_frameworks/vllm/assets/deepseek_v3_inference_flow_interactive.html `
  wiki/02_engineering/03_infer_frameworks/vllm/assets/deepseek_v3_inference_flow_interactive.js
git commit -m "docs(vllm): rewrite architecture overview exemplar"
```

### Task 2: Repair Direct Navigation, Backlinks, and Migration Metadata

**Files:**
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/index.md`
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/10_vllm_engine_architecture_analysis.md`
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/16_vllm_serving_control_plane_analysis.md`
- Modify: `wiki/02_engineering/03_infer_frameworks/index.md`
- Modify: `wiki/changelog.md`

**Interfaces:**
- Consumes: Task 1 basename `03_vllm_architecture_overview_analysis` and its six-layer/online-lifecycle contract.
- Produces: all live inbound links point to the architecture overview; current indexes describe Wave 1 as a mixed-baseline migration without claiming the rest of the domain is rebaselined.

- [ ] **Step 1: Update the vLLM domain index minimally for Wave 1**

Using `apply_patch`:

- Replace the old `03` link/title/question with the new architecture overview.
- Replace the old “request-walkthrough exception” language with a migration note: `03` is verified at `6b110bad`; remaining pages retain their existing baselines until their approved waves complete.
- Remove the old exception that the page is organized only by time sequence.
- Describe `03` as the owner of static layers, state boundaries, and one representative request lifecycle; do not copy its six layer explanations into the index.
- Keep the current page count unchanged because Task 1 is a rename, not a new page.

- [ ] **Step 2: Repair the two known live backlinks**

In `10_vllm_engine_architecture_analysis.md` and `16_vllm_serving_control_plane_analysis.md`, replace the old slug and display text. Adjust the surrounding sentence so `03` is cited for the whole-system boundary and representative lifecycle, while detailed startup/process ownership remains with `10`/`16`.

- [ ] **Step 3: Update only the directly affected parent-index row**

In `wiki/02_engineering/03_infer_frameworks/index.md`, keep the count at `19 篇 + index` and replace the baseline cell with an explicit migration state: current domain baseline remains mixed; the architecture overview is at `6b110bad`, and later waves will converge the rest. Do not touch unrelated framework rows or `wiki/index.md`.

- [ ] **Step 4: Append a changelog entry without rewriting history**

Prepend a dated entry that records:

- old `03` renamed/rebuilt as `03_vllm_architecture_overview_analysis`;
- the new page explains static layers first, lifecycle second, and code as evidence rather than copied prose;
- DeepSeek-specific MLA/MoE, syntax table, mechanical function index, and oversized interactive assets were retired or returned to owner pages;
- validation is intentionally scoped to the vLLM directory and direct backlinks/indexes;
- later waves remain pending user acceptance of the exemplar.

Do not edit the 2026-08-26 historical changelog entry; it remains a record of what existed then.

- [ ] **Step 5: Check direct-link closure and commit**

Run:

```powershell
rg -n "03_vllm_request_flow_walkthrough_analysis" `
  wiki/02_engineering/03_infer_frameworks/vllm `
  wiki/02_engineering/03_infer_frameworks/index.md
rg -n "03_vllm_architecture_overview_analysis" `
  wiki/02_engineering/03_infer_frameworks/vllm `
  wiki/02_engineering/03_infer_frameworks/index.md `
  wiki/changelog.md
git diff --check -- `
  wiki/02_engineering/03_infer_frameworks/vllm `
  wiki/02_engineering/03_infer_frameworks/index.md `
  wiki/changelog.md
```

Expected: no old slug in live vLLM pages or the parent index; the new slug appears in the domain index and both backlinks; diff check is clean.

```powershell
git add -- `
  wiki/02_engineering/03_infer_frameworks/vllm/index.md `
  wiki/02_engineering/03_infer_frameworks/vllm/10_vllm_engine_architecture_analysis.md `
  wiki/02_engineering/03_infer_frameworks/vllm/16_vllm_serving_control_plane_analysis.md `
  wiki/02_engineering/03_infer_frameworks/index.md `
  wiki/changelog.md
git commit -m "docs(vllm): integrate architecture overview"
```

### Task 3: Run the Scoped Wave Gate and Record Coverage

**Files:**
- Verify: `wiki/02_engineering/03_infer_frameworks/vllm/**`
- Verify: `wiki/02_engineering/03_infer_frameworks/index.md`
- Verify: direct vLLM entries added to `wiki/changelog.md`
- Modify: `docs/superpowers/specs/2026-08-29-vllm-latest-codebase-analysis-design.md`

**Interfaces:**
- Consumes: Task 1 source locators and Task 2 live-link closure.
- Produces: spec coverage row `全系统静态分层 + 代表请求生命周期` marked `covered (Wave 1)` only after all scoped gates pass; this is the user-review checkpoint before Wave 2.

- [ ] **Step 1: Verify every frozen-baseline inline `file:line` locator**

Run this PowerShell check from the knowledge-base worktree:

```powershell
$source = 'E:/97-codes/torch_parallel/.worktrees/vllm-6b110bad'
$baseline = '6b110badbb22d3f66c7218b71138f13b7a6b3419'
$page = 'wiki/02_engineering/03_infer_frameworks/vllm/03_vllm_architecture_overview_analysis.md'
$text = Get-Content -Raw -LiteralPath $page
$matches = [regex]::Matches($text, '`(?<path>(?:vllm|docs|tests)/[^`:]+):(?<start>\d+)(?:-(?<end>\d+))?`')
$unique = @{}
foreach ($m in $matches) {
  $path = $m.Groups['path'].Value
  $start = [int]$m.Groups['start'].Value
  $end = if ($m.Groups['end'].Success) { [int]$m.Groups['end'].Value } else { $start }
  $unique["$path`:$start-$end"] = @($path, $start, $end)
}
$errors = [System.Collections.Generic.List[string]]::new()
foreach ($item in $unique.Values) {
  $full = Join-Path $source $item[0]
  if (-not (Test-Path -LiteralPath $full)) { $errors.Add("missing path: $($item[0])"); continue }
  $lineCount = (Get-Content -LiteralPath $full).Count
  if ($item[1] -lt 1 -or $item[2] -lt $item[1] -or $item[2] -gt $lineCount) {
    $errors.Add("bad range: $($item[0]) L$($item[1])-L$($item[2]) of $lineCount")
  }
}
$head = (git -C $source rev-parse HEAD).Trim()
if ($head -ne $baseline) { $errors.Add("wrong source HEAD: $head") }
if ($unique.Count -eq 0) { $errors.Add('no inline file:line locators found') }
if ($errors.Count -gt 0) { $errors; "LOCATOR_BOUNDS=FAIL unique=$($unique.Count) head=$head"; exit 1 }
"LOCATOR_BOUNDS=PASS unique=$($unique.Count) head=$head"
```

Expected: `LOCATOR_BOUNDS=PASS unique=64 head=6b110badbb22d3f66c7218b71138f13b7a6b3419`.

- [ ] **Step 2: Manually spot-check three load-bearing claims at their opened ranges**

Open and compare the cited source for exactly these claims:

1. frontend/EngineCore state separation and request submission;
2. `EngineCore.step()` scheduling→execution→output commit order;
3. device runner/model boundary or V1 alias/MRV2 fallback boundary.

Record the three verified `file:line` ranges in the SDD task report. If prose claims more than the opened code proves, narrow the prose or label the extra reasoning as inference.

- [ ] **Step 3: Run a scoped wikilink resolver over only changed live pages**

Build the target universe from filenames, but inspect links only in the vLLM directory and directly edited parent index/backlink files:

```powershell
$wiki = (Resolve-Path 'wiki').Path
$checked = @(
  Get-ChildItem 'wiki/02_engineering/03_infer_frameworks/vllm' -Recurse -Filter '*.md'
  Get-Item 'wiki/02_engineering/03_infer_frameworks/index.md'
)
$allPages = @(Get-ChildItem $wiki -Recurse -Filter '*.md')
$byStem = @{}
foreach ($p in $allPages) {
  if (-not $byStem.ContainsKey($p.BaseName)) { $byStem[$p.BaseName] = @() }
  $byStem[$p.BaseName] += $p
}
$errors = [System.Collections.Generic.List[string]]::new()
foreach ($page in $checked) {
  $text = Get-Content -Raw -LiteralPath $page.FullName
  foreach ($m in [regex]::Matches($text, '\[\[([^\[\]\r\n]+?)\]\]')) {
    $raw = $m.Groups[1].Value -replace '\\\|', '|'
    $target = (($raw -split '\|', 2)[0] -split '#', 2)[0].Trim() -replace '\\', '/'
    $target = $target -replace '\.md$', ''
    if (-not $target) { continue }
    if ($target -eq 'index') { $errors.Add("bare index: $($page.FullName)"); continue }
    if ($target.Contains('/')) {
      $candidate = Join-Path $wiki ($target + '.md')
      if (-not (Test-Path -LiteralPath $candidate)) { $errors.Add("broken: $($page.Name) -> $target") }
    } elseif (-not $byStem.ContainsKey($target)) {
      $errors.Add("broken: $($page.Name) -> $target")
    } elseif ($byStem[$target].Count -ne 1) {
      $errors.Add("ambiguous: $($page.Name) -> $target")
    }
  }
}
if ($errors.Count -gt 0) { $errors | Sort-Object -Unique; exit 1 }
'scoped wikilinks: PASS'
```

Expected: `scoped wikilinks: PASS`. This command must not be replaced with the full Wiki strict checker.

- [ ] **Step 4: Run only changed-figure, changed-file, and conditional math checks**

Run:

```powershell
$page = 'wiki/02_engineering/03_infer_frameworks/vllm/03_vllm_architecture_overview_analysis.md'
rg -n "```mermaid|flowchart|sequenceDiagram|subgraph|-->|-.->|end$" $page
git diff --check 6e803c57 -- `
  wiki/02_engineering/03_infer_frameworks/vllm `
  wiki/02_engineering/03_infer_frameworks/index.md `
  wiki/changelog.md `
  docs/superpowers/specs/2026-08-29-vllm-latest-codebase-analysis-design.md
git diff 6e803c57 -- $page | rg -n '^\+.*\$|^\+.*\\\(|^\+.*\\\['
```

Re-read both Mermaid blocks against the parser checklist. If the final command finds added formulas, run:

```powershell
python tools/check_math.py --strict wiki/02_engineering/03_infer_frameworks/vllm/03_vllm_architecture_overview_analysis.md
```

If it finds no formula, record “math check not applicable” and do not run a broader math scan.

- [ ] **Step 5: Apply the deletion test and ownership review**

Temporarily ignore fenced code/diagram blocks while reading the page and answer from prose alone:

- why each of the six boundaries exists;
- what crosses each boundary and who owns the state;
- where admission/KV progress commits;
- when output becomes externally visible;
- why cross-cutting planes cannot be assigned to one directory-derived layer.

Also confirm that the page contains no detailed Scheduler/KV/MRV2/model/parallelism proof that belongs to `11/12/15/13/14/22`, and no standalone feature-modification catalogue.

- [ ] **Step 6: Mark only the exemplar coverage row complete and commit**

After Steps 1–5 pass, use `apply_patch` in the spec coverage matrix to change only:

```text
全系统静态分层 + 代表请求生命周期 | 03 | 一段局部定位 + 链接 | planned
```

to:

```text
全系统静态分层 + 代表请求生命周期 | 03 | 一段局部定位 + 链接 | covered (Wave 1)
```

Leave every other row `planned` or `gap → planned`.

```powershell
git diff --check -- docs/superpowers/specs/2026-08-29-vllm-latest-codebase-analysis-design.md
git add -- docs/superpowers/specs/2026-08-29-vllm-latest-codebase-analysis-design.md
git commit -m "docs(vllm): record architecture exemplar coverage"
```

Expected final state: the new `03` page and direct navigation are committed, all scoped checks pass, unrelated work is absent from the branch, and Waves 2–6 remain unimplemented pending user review of the exemplar.
