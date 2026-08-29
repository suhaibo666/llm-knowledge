# Planning Codebase Analysis Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repository-level planner that discovers a new codebase's capabilities, architecture, lifecycles, and core mechanisms, proposes a document blueprint for user approval, then coordinates page-level source-faithful analysis.

**Architecture:** `planning-codebase-analysis` owns whole-repository reconnaissance, content coverage, page ownership, the approval gate, and multi-page orchestration. `source-faithful-analysis` remains the downstream executor for one analysis unit/page and may cross any number of source files; the existing wiki maintenance and figure/math skills keep their current responsibilities.

**Tech Stack:** Markdown skill definitions, YAML frontmatter, JSON evaluation cases, Python 3.13 + pytest integration tests, Git partial staging, repository documentation gates

**Spec:** `docs/superpowers/specs/2026-08-29-planning-codebase-analysis-skill-design.md`

## Global Constraints

- The new planner applies only to a new whole codebase, a large new codebase domain, or a knowledge-domain-level replanning request.
- A focused mechanism question or one approved page contract goes directly to `source-faithful-analysis`.
- “One analysis unit/page” is not “one source file”; page-level analysis may cross files to trace state, ownership, tests, and call chains.
- The planner must perform only read-only reconnaissance before the user explicitly approves the document blueprint.
- The blueprint owns capability coverage, static architecture, representative lifecycles, page boundaries, numbering, concept ownership, and implementation order.
- The planner must not prescribe a fixed number of layers, pages, lines, code blocks, or numbering bands.
- Wiki directories derive from the existing functional tree and capability ownership, never directly from the source directory tree.
- An approved blueprint is reopened only for the material drift conditions listed in the spec; ordinary locator or subsection changes do not trigger another approval.
- Preserve every existing uncommitted change. `AGENTS.md`, `CLAUDE.md`, `skills/README.md`, `skills/source-faithful-analysis/SKILL.md`, and `references/codebase.md` are already dirty; use incremental edits and partial staging.
- Do not stage the existing drawing-skill routing edits, prior source-analysis prose changes, `references/paper.md`, Wiki pages, figures, or any unrelated file.
- Keep one physical project skill copy under `skills/`; do not add `.agents/skills`, `.codex/skills`, or another mirrored tree.
- Keep the first version self-contained in one `SKILL.md`; the blueprint contract is short enough that a separate `references/analysis-blueprint.md` would add routing overhead without progressive-disclosure value.

## File Map

- `skills/planning-codebase-analysis/SKILL.md`: trigger boundary, read-only reconnaissance workflow, blueprint contract, approval/replanning gates, downstream skill dispatch, and coverage completion gate.
- `skills/planning-codebase-analysis/evals/evals.json`: six realistic routing and behavior scenarios for baseline and forward testing.
- `tools/test_planning_codebase_analysis_skill.py`: structural integration tests for the new entrypoint, evaluation matrix, route tables, and downstream boundary.
- `skills/source-faithful-analysis/SKILL.md`: explicitly route unplanned whole-codebase work to the planner while preserving paper/general and focused deep-dive behavior.
- `skills/source-faithful-analysis/references/codebase.md`: redefine its code scope as one planned analysis unit/page or focused mechanism, not one source file or repository-wide page planning.
- `AGENTS.md`: Codex project-level on-demand route for the planner and downstream analysis skill.
- `CLAUDE.md`: constitution-level on-demand route for the planner and downstream analysis skill.
- `skills/README.md`: human-readable routing examples and the planner → writer → maintenance composition.

---

### Task 1: Establish the Behavior Contract and Add the Planner Skill

**Files:**
- Create: `skills/planning-codebase-analysis/SKILL.md`
- Create: `skills/planning-codebase-analysis/evals/evals.json`
- Create: `tools/test_planning_codebase_analysis_skill.py`

**Interfaces:**
- Consumes: a repository path or checkout, target audience, target knowledge-base context, existing related pages, and the repository's current branch/commit.
- Produces before approval: a repository baseline, system thesis, capability map, static architecture map, representative lifecycle map, document plan, coverage matrix, implementation order, unresolved questions, and material-drift risks.
- Produces after approval: one page contract at a time for `source-faithful-analysis`, plus coverage-matrix state used by the coordinator.
- Must stop before approval without creating, renaming, or rewriting any Wiki body page.

- [ ] **Step 1: Run the no-planner behavioral control five times**

Use five fresh-context, read-only subagent runs with this exact prompt and the current project skills, before creating the new `SKILL.md`:

```text
工作区是 E:\97-codes\torch_parallel\llm-knowledge。把 tools/docs-site 当成一个第一次接触的新 codebase。用户要求：“分析这个仓库的架构、模块和核心机制，并整理成一组 wiki 文档。”只说明你下一步会做什么、会交付什么，以及何时开始修改 wiki；不要真的修改文件。
```

Record in the implementation commentary whether each run:

1. builds a capability/architecture/content blueprint before page writing;
2. proposes page ownership and a coverage check;
3. explicitly waits for user approval before Wiki writes.

Expected RED: at least one run omits the approval gate, page ownership, or coverage matrix, or proceeds directly toward writing. Capture the exact rationalization. If all five controls already satisfy all three criteria, stop and report that the proposed skill would duplicate current behavior; do not create it without revisiting the design with the user.

- [ ] **Step 2: Write the failing structural integration tests**

Create `tools/test_planning_codebase_analysis_skill.py` with this content:

```python
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER = REPO_ROOT / "skills/planning-codebase-analysis"


def test_planner_entrypoint_declares_scope_and_hard_gate():
    text = (PLANNER / "SKILL.md").read_text(encoding="utf-8")
    assert "name: planning-codebase-analysis" in text
    assert "Use when" in text
    for phrase in (
        "whole codebase",
        "capability map",
        "static architecture",
        "dynamic lifecycle",
        "coverage matrix",
        "user approval",
        "source-faithful-analysis",
    ):
        assert phrase in text, f"planner contract missing {phrase!r}"


def test_planner_has_six_routing_and_behavior_evals():
    payload = json.loads((PLANNER / "evals/evals.json").read_text(encoding="utf-8"))
    assert payload["skill_name"] == "planning-codebase-analysis"
    assert {item["id"] for item in payload["evals"]} == {1, 2, 3, 4, 5, 6}
    assert all(item["prompt"] and item["expected_output"] for item in payload["evals"])
```

- [ ] **Step 3: Run the tests and verify the planner is absent**

Run:

```powershell
python -m pytest tools/test_planning_codebase_analysis_skill.py -q
```

Expected: FAIL with `FileNotFoundError` for `skills/planning-codebase-analysis/SKILL.md`.

- [ ] **Step 4: Create the six evaluation cases**

Create `skills/planning-codebase-analysis/evals/evals.json` with this complete payload:

```json
{
  "skill_name": "planning-codebase-analysis",
  "evals": [
    {
      "id": 1,
      "prompt": "给你一个第一次接触的新框架仓库。请分析整体架构、模块和核心机制，并整理成一组 Wiki 文档。",
      "expected_output": "先冻结仓库基线并只读侦察，产出能力地图、静态架构、代表性生命周期、文档蓝图和覆盖矩阵；在用户确认目录、页面边界、编号与范围之前不创建正文页面。",
      "files": []
    },
    {
      "id": 2,
      "prompt": "目录和页面职责已经确认。现在只分析 scheduler 的请求状态机、抢占和提交边界，完成 scheduler_analysis.md。",
      "expected_output": "不重新规划整个代码库；将该页面视为一个分析单元，调用 source-faithful-analysis 跨文件核验状态、调用链、设计理由和边界。",
      "files": []
    },
    {
      "id": 3,
      "prompt": "按已经批准的蓝图实施 architecture_overview.md：该页拥有静态职责分层与端到端生命周期，不展开 scheduler 和 cache 的内部算法。",
      "expected_output": "遵守已确认的页面所有权和不包含内容，进入单篇 source-faithful-analysis；不修改目录，不把 scheduler/cache 深挖复制进概览。",
      "files": []
    },
    {
      "id": 4,
      "prompt": "这个仓库恰好有 api、engine、workers、models、kernels 五个一级目录，请直接按这五个目录规划五篇架构文档。",
      "expected_output": "拒绝把目录树直接改名为架构；先从用户能力、职责、依赖方向和状态所有权验证页面边界，允许合并、拆分或跨目录形成一个机制页。",
      "files": []
    },
    {
      "id": 5,
      "prompt": "已批准把 executor 和 scheduler 写成两页，但侦察发现两者共同维护一个不可拆分的请求状态机。请继续按原计划写，不必再问。",
      "expected_output": "识别概念所有权发生实质变化，停止受影响页面的实施，提交修订后的页面边界和覆盖矩阵供用户重新确认。",
      "files": []
    },
    {
      "id": 6,
      "prompt": "这是一个包含数千个文件和几十个后端的大仓库。请确保每个目录都有对应分析页，越完整越好。",
      "expected_output": "先按能力和高内聚子域缩小范围，选择代表性生命周期与核心机制；不按目录逐一建页，不使用页数或行数作为覆盖完成标准。",
      "files": []
    }
  ]
}
```

- [ ] **Step 5: Create the minimal planner skill**

Create `skills/planning-codebase-analysis/SKILL.md`. Use this frontmatter exactly:

```yaml
---
name: planning-codebase-analysis
description: >-
  Use when a user provides a new whole codebase or large codebase domain and wants its architecture,
  modules, core mechanisms, or a multi-page knowledge-base analysis planned before writing begins.
  Also use for knowledge-domain-level replanning when page ownership, coverage, directory placement,
  or implementation order is not yet agreed. Do not use for one focused mechanism, one approved page
  contract, ordinary Wiki maintenance, or operating and fixing code without an analysis deliverable.
---
```

The body must contain these sections and contracts, using concise imperative prose rather than repeating the design spec verbatim:

```markdown
# Planning Codebase Analysis

## Responsibility boundary
- Own whole-codebase reconnaissance, content selection, page ownership, user approval, and coverage.
- Treat one page as one analysis unit, not one source file.
- Route focused or approved units to `source-faithful-analysis`.

## Hard gate
**NO WIKI BODY WRITES BEFORE USER APPROVES THE BLUEPRINT.**
Read-only repository and existing-Wiki inspection is allowed before approval.

## Workflow
1. Anchor the repository, target audience, existing Wiki context, branch, commit, and date.
2. Build the discovery map from capabilities first, then static architecture, representative lifecycles, state owners, core mechanisms, extension boundaries, engineering constraints, and live/legacy status.
3. Convert the discovery map into the blueprint contract below.
4. Present the blueprint and stop for explicit user approval.
5. Persist an approved multi-page blueprint under `docs/superpowers/specs/` when it must survive multiple pages or sessions.
6. Dispatch one approved page contract at a time to `source-faithful-analysis`; invoke maintenance, math, figure, and Mermaid skills only when their concrete work appears.
7. Update the coverage matrix after each wave and reopen approval only for material drift.

## Blueprint contract
- Repository baseline, audience, Wiki placement, system thesis, live/legacy boundary, unresolved evidence gaps.
- Capability map, static responsibility/state-ownership map, and one or more representative dynamic lifecycles.
- Per-page table: path/title, page type, thesis, reader question, owned concepts, explicit exclusions, core mechanisms, evidence entry points, dependencies, visual candidates, and completion test.
- Coverage matrix: each capability/lifecycle/mechanism has one authoritative page, permitted summaries elsewhere, and planned/covered/gap state.
- Implementation order based on conceptual dependencies, not filename order.

## Planning rules
- Derive content from user-visible capabilities and mechanism ownership, never directly from directories.
- Use discovery dimensions as prompts, not a mandatory taxonomy.
- Do not set fixed layer, page, line, code-block, or numbering quotas.
- Fit the existing Wiki functional tree; propose a new directory only when no existing functional owner fits.
- Number by local convention and reading dependency; do not assign `01` merely because a feature is prominent.
- Existing related pages require a reuse/rewrite/merge decision before new pages are proposed.

## Replanning gate
Reconfirm only when page groups are added/removed, concept ownership moves across pages, a Wiki directory must be created/moved, the source baseline changes, source evidence overturns the system thesis/module split, or the audience/scope/deliverable expands.

## Completion gate
Completion means every planned core capability, representative lifecycle, state owner, and mechanism has one authoritative page; duplicates and gaps are resolved; source and Wiki gates pass. Page count and length are not completion evidence.

## Red flags
| Rationalization | Required response |
|---|---|
| “The directory layout is obvious; I can plan from it.” | Build the capability and ownership maps first. |
| “I can draft the overview while the user reviews the plan.” | Stop; drafting Wiki body content crosses the approval gate. |
| “More pages means better coverage.” | Use the coverage matrix, not page count. |
| “A writer found a better page split and can continue.” | Return material ownership changes to the coordinator and user. |
| “One source file should become one document.” | Define one causal analysis unit and follow it across files. |
```

- [ ] **Step 6: Run structural tests and the skill validator**

Run:

```powershell
python -m pytest tools/test_planning_codebase_analysis_skill.py -q
python -X utf8 'C:\Users\suhaibo\.codex\skills\.system\skill-creator\scripts\quick_validate.py' 'E:\97-codes\torch_parallel\llm-knowledge\skills\planning-codebase-analysis'
```

Expected: both planner tests PASS and the validator prints `Skill is valid!`.

- [ ] **Step 7: Run the planner-guided whole-repository scenario five times**

Use five fresh-context subagents. Give each the new skill path, the `tools/docs-site` scope, and evaluation case 1. Require a read-only response. Every run must:

- start with repository/Wiki reconnaissance rather than a page draft;
- include capability, static, dynamic, page-plan, and coverage outputs;
- stop at explicit user approval before any Wiki writes.

Expected GREEN: 5/5 satisfy all three criteria. Read every response manually; keyword counts alone do not pass the gate.

- [ ] **Step 8: Commit the new skill contract**

```powershell
git add skills/planning-codebase-analysis/SKILL.md skills/planning-codebase-analysis/evals/evals.json tools/test_planning_codebase_analysis_skill.py
git diff --cached --check
git diff --cached --name-status
git commit -m "feat(skills): add codebase analysis planner"
```

Expected staged paths: exactly the three files listed above.

---

### Task 2: Route Whole-Codebase Work and Narrow the Page-Level Executor

**Files:**
- Modify: `tools/test_planning_codebase_analysis_skill.py`
- Modify: `AGENTS.md:11-16`
- Modify: `CLAUDE.md:13-18`
- Modify: `skills/README.md:17-27`
- Modify: `skills/source-faithful-analysis/SKILL.md:1-11,90-139,181-216`
- Modify: `skills/source-faithful-analysis/references/codebase.md:1-23`

**Interfaces:**
- Consumes: an implicit user request plus either an unplanned whole-codebase scope or an approved/focused analysis unit.
- Produces: deterministic routing — planner first for the former, `source-faithful-analysis` directly for the latter.
- Preserves: paper/spec/dataset/general analysis behavior and the existing mechanism, citation, architecture, and deletion-test contracts.

- [ ] **Step 1: Extend the integration test with failing routing assertions**

Append this code to `tools/test_planning_codebase_analysis_skill.py`:

```python
ROUTE_DOCS = (
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "CLAUDE.md",
    REPO_ROOT / "skills/README.md",
)
SOURCE_SKILL = REPO_ROOT / "skills/source-faithful-analysis/SKILL.md"
CODEBASE_PACK = REPO_ROOT / "skills/source-faithful-analysis/references/codebase.md"


def test_new_codebase_planner_is_discoverable_from_every_route_table():
    for path in ROUTE_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "planning-codebase-analysis" in text, f"planner is not routed from {path}"
        assert "source-faithful-analysis" in text, f"page-level writer disappeared from {path}"


def test_source_analysis_delegates_unplanned_whole_codebases():
    core = SOURCE_SKILL.read_text(encoding="utf-8")
    pack = CODEBASE_PACK.read_text(encoding="utf-8")
    assert "planning-codebase-analysis" in core
    assert "approved blueprint" in core
    assert "one analysis unit/page" in pack
    assert "not one source file" in pack
```

- [ ] **Step 2: Run the routing tests and verify they fail for the missing route**

Run:

```powershell
python -m pytest tools/test_planning_codebase_analysis_skill.py -q
```

Expected: the entrypoint/eval tests PASS; route/delegation tests FAIL because the three route tables and source executor do not yet name the planner boundary.

- [ ] **Step 3: Add the planner and writer rows to all route tables**

In `AGENTS.md` and `CLAUDE.md`, preserve the existing figure and Mermaid rows and replace the single source-ingest row with two responsibilities:

```markdown
| Discover and plan a new codebase or large codebase domain, propose the document blueprint, wait for approval, then coordinate implementation | [`planning-codebase-analysis`](skills/planning-codebase-analysis/SKILL.md) |
| Write one approved analysis unit/page, or analyze one focused mechanism from a codebase, paper, spec, dataset, or other source | [`source-faithful-analysis`](skills/source-faithful-analysis/SKILL.md) |
```

In `skills/README.md`, use the same distinction in Chinese:

```markdown
| 侦察一个新代码库或大型新子域，规划能力/架构/核心机制与文档蓝图，确认后编排实施 | [`planning-codebase-analysis`](planning-codebase-analysis/SKILL.md) |
| 按已确认合同写一个分析单元/页面，或深入分析一个明确机制（可跨多个源码文件） | [`source-faithful-analysis`](source-faithful-analysis/SKILL.md) |
```

Replace the existing composition example with:

```markdown
例如，“把一个新代码库整理成知识域”先读 `planning-codebase-analysis`；蓝图确认后，每篇页面再读 `source-faithful-analysis`，实际落盘时叠加 `maintaining-llm-knowledge`，公式和图按页面需要加载。论文或一个明确机制的单篇分析不经过 planner。
```

- [ ] **Step 4: Add the routing gate to the source-faithful core**

Make these minimal changes without removing its paper/general workflows:

1. Add this final sentence to the YAML description:

```text
Do not use as the first step for an unplanned whole-codebase or multi-page codebase-domain analysis; use planning-codebase-analysis first.
```

2. Insert this section immediately before `## The workflow`:

```markdown
## Codebase-wide routing boundary

For a new whole codebase or large codebase domain with no user-approved document blueprint, stop and use `planning-codebase-analysis` first. Once that planner hands off one approved page/analysis-unit contract, use this skill to follow the required mechanism across as many source files as needed. Do not re-plan the directory, numbering, page ownership, or excluded concepts inside the page-writing task.

This boundary is codebase-specific. Paper, spec, dataset, incident, and focused one-off analyses continue to scale through the workflow below without requiring the codebase planner.
```

3. Under Phase 0's granularity bullet, add:

```markdown
- **For an approved codebase page:** inherit granularity, title, concept ownership, exclusions, and evidence entry points from the blueprint; do not reopen repository-wide page planning.
```

4. Under Phase 3 and Phase 4, state that codebase-wide wave assignment, coverage reconciliation, and user-visible replanning stay with `planning-codebase-analysis`; the page writer returns verified locators, thesis, boundary findings, and material drift to the coordinator.

- [ ] **Step 5: Narrow the codebase pack to one analysis unit/page**

Insert this block after the opening paragraph of `references/codebase.md`:

```markdown
## Routing gate

This pack executes **one analysis unit/page or one focused mechanism**, not one source file. A mechanism may cross entry points, state owners, helpers, backends, tests, and history; follow all load-bearing files needed to explain it. If the request instead covers a new whole codebase or a multi-page domain and there is no approved blueprint, use `planning-codebase-analysis` before this pack.

When an approved blueprint exists, inherit the page thesis, owned concepts, explicit exclusions, evidence entry points, and completion test. Do not redesign the repository's page tree from inside the page task.
```

Replace the current `Decompose by subsystem × depth` bullet with:

```markdown
- **Map the owned unit, not the whole documentation tree.** Identify the page's entry symbols, state owners, real call path, tests, and external contracts. Return a material-drift finding to the planner if the approved ownership cannot match the source.
```

Keep the current architecture-overview contract and “Explain the code; do not transport it” section unchanged.

- [ ] **Step 6: Run the routing tests and both skill validators**

Run:

```powershell
python -m pytest tools/test_planning_codebase_analysis_skill.py -q
python -X utf8 'C:\Users\suhaibo\.codex\skills\.system\skill-creator\scripts\quick_validate.py' 'E:\97-codes\torch_parallel\llm-knowledge\skills\planning-codebase-analysis'
python -X utf8 'C:\Users\suhaibo\.codex\skills\.system\skill-creator\scripts\quick_validate.py' 'E:\97-codes\torch_parallel\llm-knowledge\skills\source-faithful-analysis'
```

Expected: all tests PASS; both validators print `Skill is valid!`.

- [ ] **Step 7: Run five routing samples in each direction**

Use fresh-context, read-only agents and manually inspect all 15 responses:

- five runs of eval 1 without naming a skill: must select the planner and stop at blueprint approval;
- five runs of eval 2 without naming a skill: must select page-level source analysis, not rebuild the repository plan;
- five runs of eval 3 with the approved page contract: must preserve owned/excluded concepts and begin only the requested page analysis.

Expected: 5/5 correct for each scenario and low wording variance in the routing decision. A response that names both skills but begins Wiki writing before approval fails eval 1.

- [ ] **Step 8: Partially stage only this task's shared-file hunks and commit**

Because every shared route/source file already contains unrelated working-tree changes, use interactive partial staging and inspect the cached patch:

```powershell
git add tools/test_planning_codebase_analysis_skill.py
git add -p -- AGENTS.md CLAUDE.md skills/README.md skills/source-faithful-analysis/SKILL.md skills/source-faithful-analysis/references/codebase.md
git diff --cached --check
git diff --cached --name-status
git diff --cached
git commit -m "docs(skills): route codebase domains through planner"
```

The cached diff may include only planner route rows, the source-skill routing boundary, the codebase-pack routing gate, and the new integration-test additions. Reject any hunk containing the pre-existing drawing-figure routing work, architecture-overview prose, deletion-test prose, or other user changes. If adjacent old and new lines share one hunk, choose patch-edit mode (`e`) and remove the unrelated `+`/`-` lines before accepting it; never stage the whole file as a shortcut.

---

### Task 3: Pressure-Test the Approval and Coverage Discipline

**Files:**
- Modify only if a guided run exposes a loophole: `skills/planning-codebase-analysis/SKILL.md`
- Modify only if expected outputs need clarification after an observed failure: `skills/planning-codebase-analysis/evals/evals.json`

**Interfaces:**
- Consumes: the six committed eval cases and fresh-context agents.
- Produces: evidence that the planner resists directory mirroring, premature writing, page-count pressure, unauthorized plan drift, and source-file-per-page decomposition.

- [ ] **Step 1: Run evals 4–6 with five fresh samples each**

Run 15 read-only samples and manually score them against the exact `expected_output` fields:

- eval 4: rejects directory-to-page mirroring and rebuilds responsibility boundaries;
- eval 5: stops and requests reconfirmation because concept ownership moved;
- eval 6: narrows by capability/domain and rejects page-count completeness.

Expected: 5/5 correct for each scenario.

- [ ] **Step 2: Check for the five known rationalizations**

Search the 30 guided outputs from Tasks 1–3 for these failure patterns, reading every match in context:

```text
start drafting while approval is pending
one page per directory or source file
more pages/lines as completeness evidence
writer silently changes concept ownership
focused page request triggers repository-wide replanning
```

Expected: zero true violations. Quoted counter-examples in the skill or response do not count as failures.

- [ ] **Step 3: Apply only evidence-backed wording corrections**

If a true violation appears, add the corresponding exact counter to the existing red-flags table, rerun that scenario five times, and require 5/5 compliance. Do not add rules for hypothetical failures and do not change the capability/discovery model when the observed problem is only routing wording.

- [ ] **Step 4: Re-run the specific tests and validators**

```powershell
python -m pytest tools/test_planning_codebase_analysis_skill.py -q
python -X utf8 'C:\Users\suhaibo\.codex\skills\.system\skill-creator\scripts\quick_validate.py' 'E:\97-codes\torch_parallel\llm-knowledge\skills\planning-codebase-analysis'
python -X utf8 'C:\Users\suhaibo\.codex\skills\.system\skill-creator\scripts\quick_validate.py' 'E:\97-codes\torch_parallel\llm-knowledge\skills\source-faithful-analysis'
```

Expected: tests PASS and both validators print `Skill is valid!`.

- [ ] **Step 5: Commit only if Task 3 changed tracked content**

```powershell
git add -p -- skills/planning-codebase-analysis/SKILL.md skills/planning-codebase-analysis/evals/evals.json
git diff --cached --check
git diff --cached
git commit -m "docs(skills): harden codebase planning gates"
```

Skip this commit when all behavior checks pass without content changes.

---

### Task 4: Full Repository Verification and Final Scope Audit

**Files:**
- Verify: all files changed in Tasks 1–3
- Do not modify: Wiki content, figures, `references/paper.md`, or unrelated dirty files

**Interfaces:**
- Consumes: completed planner skill, downstream routing, eval matrix, and repository quality gates.
- Produces: fresh evidence that the implementation is structurally valid, behaviorally routed, and does not regress the Wiki tooling/site.

- [ ] **Step 1: Verify the targeted skill integration**

```powershell
python -m pytest tools/test_planning_codebase_analysis_skill.py -q
python -X utf8 'C:\Users\suhaibo\.codex\skills\.system\skill-creator\scripts\quick_validate.py' 'E:\97-codes\torch_parallel\llm-knowledge\skills\planning-codebase-analysis'
python -X utf8 'C:\Users\suhaibo\.codex\skills\.system\skill-creator\scripts\quick_validate.py' 'E:\97-codes\torch_parallel\llm-knowledge\skills\source-faithful-analysis'
```

Expected: targeted tests PASS and both skills are valid.

- [ ] **Step 2: Run every repository quality gate**

```powershell
python tools/check_links.py --strict
python tools/check_math.py --changed --strict
python -m pytest tools/
npm run docs:test
git diff --check
```

Expected:

- links: `broken=0`, `ambiguous=0`, `bare_index=0`, `stale_section=0`, `orphans=0`;
- math: `0 error(s), 0 warning(s)`;
- pytest: all collected tests PASS, including `test_planning_codebase_analysis_skill.py`;
- docs: 69 unit tests PASS and the browser smoke test reports PASS;
- diff check: exit code 0, allowing only line-ending warnings.

- [ ] **Step 3: Audit the final diff against the spec**

Confirm each requirement has one implementation location:

```text
whole-repository trigger              -> planner frontmatter + route tables
capability/static/dynamic discovery   -> planner workflow and blueprint
document ownership + coverage         -> blueprint contract + evals
user approval before Wiki writes      -> hard gate + behavioral samples
material-drift reconfirmation         -> replanning gate + eval 5
single analysis unit, cross-file      -> source core + codebase routing gate
no fixed counts or directory mirroring-> planning rules + evals 4 and 6
downstream skill dispatch             -> planner workflow + route docs
```

Run:

```powershell
git status --short
git log -3 --oneline
```

Expected: no task-created file remains untracked or staged; unrelated pre-existing working-tree changes remain present and uncommitted.

- [ ] **Step 4: Report the implemented contract**

Report:

- new planner path and trigger boundary;
- the page-level executor boundary;
- behavior-test sample counts and results;
- targeted validator/test results and full quality-gate totals;
- commits created by this plan;
- unrelated dirty files preserved.
