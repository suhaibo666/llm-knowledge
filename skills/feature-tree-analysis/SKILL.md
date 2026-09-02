---
name: feature-tree-analysis
description: >-
  Use when a repository or repo subdomain must be inventoried as a feature tree — major modules →
  function items → sub-functions → leaf function points — with a contract-style spec per function
  point (inputs/outputs, processing logic, boundary constraints, supported scope), or when
  re-verifying such a tree after a baseline bump. Triggers: 梳理功能树 / 功能点清单 / 功能点定义 /
  feature inventory / function-point spec / capability breakdown / "list everything this repo
  provides with per-feature I/O and constraints". For mechanism-level why-design analysis use
  planning-codebase-analysis + source-faithful-analysis instead; this skill owns the exhaustive
  what-and-contract view.
---

# Feature-Tree Analysis

Turn a repository (or a subdomain of one) into a **repo feature tree** — major modules → function
items → sub-functions → leaves (function points, 功能点) — plus a contract-style spec per function
point: inputs, outputs, processing logic, boundary constraints, supported scope. The deliverable
answers "what exactly does this repo provide, and what is each feature's contract"; the acceptance
bar is **coverage completeness (MECE + two-way reconciliation to zero) and verifiable contracts**,
not causal-analysis depth.

> Terminology: in this repo, "功能树" by default means the wiki's own directory authority (see
> `CLAUDE.md`). This skill's product is always called the **repo feature tree** — the functional
> decomposition of the analyzed repository. The two are unrelated.

## Responsibility boundary

- **This skill owns**: tree construction and leaf criteria, the function-point spec template and
  its field semantics, two-way tree↔code coverage reconciliation, the tree-proposal approval gate,
  wave-based spec writing with independent review, and targeted re-verification after baseline
  bumps.
- **Not owned — route away**:
  - Mechanism-level causal analysis (why a design exists, state models, execution traces) →
    `planning-codebase-analysis` + `source-faithful-analysis`. Spec pages and mechanism pages
    **cross-link, never substitute**: a spec page links to mechanism pages for the "why"; a
    mechanism page links to spec pages for the full contract.
  - Source-fidelity discipline (verified locators / frozen baseline / fact vs inference / visible
    conflicts) → execute the **Source fidelity** section of `source-faithful-analysis` directly;
    this skill does not copy its text.
  - Wiki page operations when the host is this wiki (page types, numbering, cross-links,
    changelog) → `maintaining-llm-knowledge`.

## Hard gate

**Before the user approves the feature-tree proposal, write no function-point spec content.**
The tree is the blueprint: mis-cut modules force rework of every leaf spec. Before approval only
read-only reconnaissance and the proposal document itself are allowed. Same semantics as the
`planning-codebase-analysis` hard gate.

## Workflow

| Phase | What | Reference |
|---|---|---|
| 0 Anchor | Pin repository + **frozen commit** ("current worktree state / with uncommitted changes" is not a baseline; report uncommitted changes and let the user pick the baseline); declare scope (whole repo or subdomain) and host (this wiki or standalone deliverable) | — |
| 1 Build tree | Entry-surface inventory → level-by-level functional decomposition → attach code ownership per node → two-way reconciliation → produce the **feature-tree proposal** (tree + leaf-row list + reconciliation diff report + exclusion table) | `references/tree-method.md` |
| ⛔ Approval | Present the proposal to the user, **stop**, wait for explicit approval; material tree changes (leaf splits/merges, ownership moves) return to this gate | — |
| 2 Write specs | Wave by wave per module, leaf status planned→spec'd; parallel dispatch allowed — the writer contract reuses `source-faithful-analysis` `references/parallel-agent-contract.md` with OWNED OUTPUT replaced by "tree slice + this skill's template" | `references/feature-point-template.md` |
| 3 Verify | Independent reviewer (≠ writer) samples each wave per the review rubric + host mechanical gates | `references/review-rubric.md` |
| 4 Maintain | On a baseline bump: changed files in the commit delta → map back through leaf code ownership to the affected function points → re-verify those, advance status to verified; when the host is this wiki also update `kb_baseline` in `docs/radar/watchlist.yaml` | — |

## Mechanical checks vs manual review

Per `skills/README.md`, every rule states how it is verified:

| Rule | Verification |
|---|---|
| Locators are real at the frozen baseline | Host = this wiki: `python tools/check_locators.py`; standalone delivery: reviewer samples per rubric R1 |
| Wiki output conventions (links/math) | `check_links.py --strict`, `check_math.py --changed --strict` |
| Tree↔source-file and tree↔flag reconciliation | For now: manual zeroing during tree building per tree-method §4 (the flag surface can borrow the enumeration output of `check_coverage.py --generate`); `check_feature_tree.py` is a future hardening item — the leaf-row format (tree-method §5) is its reserved parsing surface |
| Leaf granularity, main-path logic, MECE judgment | Manual only (rubric R2/R5 + user sampling) |

## Host-specific rules

- **Host = this wiki**: tree overview page `NN_<repo>_feature_tree_analysis.md` + one spec page per
  module (the `_analysis` suffix is on the whitelist); canonical baseline header
  `> **源码基线**：owner/repo@hex（branch，YYYY-MM-DD）`; Related Pages / changelog /
  `kb_baseline` / all five quality gates apply in full; split a spec page over 500 lines by
  sub-module.
- **Standalone deliverable**: output directory set by the task; minimum gates = frozen-baseline
  header + rubric R1 locator sampling + reconciliation diff zeroed.

## Red flags (rationalizations caught in live baseline testing)

| Rationalization | Correction |
|---|---|
| "Baseline: current worktree state (with uncommitted changes)" | Not a frozen baseline — locators pinned to a moving target. Pin a commit hash; report uncommitted changes and let the user decide. |
| "Cut modules by directory; one source file = one function item" | Directories and files are evidence, not the decomposition. A function item answers "what it does for the caller"; `models.py` and `utils/` are not features. |
| "Internal helper functions count as function points; finer is better" | The leaf test is "independently triggerable + closable contract". An internal filter/normalize helper is one step of some function point's processing logic. More leaves ≠ better coverage; a zeroed reconciliation is. |
| "Test directories and data artifacts get tree nodes too" | Tests and artifacts are evidence and outputs: they go into the "tests & links" and "outputs" fields, never become nodes. |
| "The tree is done — write all the specs, then show the user" | Violates the hard gate. Reworking a hundred leaves costs far more than one approval round. |
| "Supported scope can carry some policy/background too" | The four core fields have fixed semantics (see the template); supported scope takes only "supported / unsupported / defaults per dimension, with evidence". |
| "I skimmed it — close enough to write the spec" | A spot you have not opened yields no claim. If you cannot finish reading, mark the leaf planned for the next wave; never write a half-understood spec. |
