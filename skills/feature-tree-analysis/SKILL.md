---
name: feature-tree-analysis
description: >-
  Use when a repository or repo subdomain must be inventoried as a feature tree — major modules →
  function items → sub-functions → leaf function points — with a contract-style spec per function
  point (inputs/outputs, processing logic, boundary constraints, supported scope), or when
  re-verifying such a tree after a baseline bump. Triggers: 梳理功能树 / 功能点清单 / 功能点定义 /
  feature inventory / function-point spec / capability breakdown / "list everything this repo
  provides with per-feature I/O and constraints". Not for why-design analysis: a focused mechanism
  goes to source-faithful-analysis alone; a whole codebase goes to planning-codebase-analysis first,
  then approved architecture, feature, and mechanism pages use the matching document profile under
  source-faithful-analysis. This skill owns the exhaustive what-and-contract view.
---

# Feature-Tree Analysis

Turn a repository (or a subdomain of one) into a **repo feature tree** — major modules → function
items → sub-functions → leaves (function points, 功能点) — plus a contract-style spec per function
point: inputs, outputs, processing logic, boundary constraints, supported scope. The deliverable
answers "what exactly does this repo provide, and what is each feature's contract". The
acceptance bar is **coverage with an audit trail** — the tree and every leaf's ownership recorded
in one manifest, a two-way reconciliation that the checker reports as zero, and a delivery gate
that only opens when every leaf is `verified` — plus **verifiable contracts**. Causal-analysis
depth is not the bar. A zero reconciliation is a mechanical floor, not a proof of completeness:
reviewer judgment (rubric R2) and user sampling sit on top of it.

> Terminology: in this repo, "功能树" by default means the wiki's own directory authority (see
> `CLAUDE.md`). This skill's product is always called the **repo feature tree** — the functional
> decomposition of the analyzed repository. The two are unrelated.

## Responsibility boundary

- **This skill owns**: tree construction and leaf criteria, the ownership manifest (tree nodes,
  leaves, ownership, exclusions, review records) and its checker (`tools/check_feature_tree.py`),
  the function-point spec template and its field semantics, the tree-proposal approval gate,
  wave-based spec writing under a dedicated writer contract, independent review, the delivery
  gate, and re-reconciliation after baseline bumps.
- **Not owned — route away**:
  - Why-design analysis: a focused mechanism → `source-faithful-analysis` on its own; a whole
    codebase or unplanned multi-page domain → `planning-codebase-analysis` first. After approval,
    an architecture page → `source-faithful-analysis` / `software-architecture`; a concrete feature
    page → `source-faithful-analysis` / `feature-analysis`; a mechanism page →
    `source-faithful-analysis` / `mechanism-analysis`. Spec pages and analysis pages **cross-link,
    never substitute**: a spec page links to the owning analysis page for the "why"; an analysis
    page links to spec pages for the full contract.
  - Source-fidelity discipline (verified source anchors / frozen baseline / fact vs inference / visible
    conflicts) → read `../source-faithful-analysis/references/source-fidelity.md` directly; this
    skill imports that shared kernel but does not enter a prose document profile or copy its text.
  - Wiki page operations when the host is this wiki (page types, numbering, cross-links,
    changelog) → `maintaining-llm-knowledge`.

## Hard gate

**Before the user approves the feature-tree proposal, write no function-point spec content and
persist nothing into the host.** The tree is the blueprint: mis-cut modules force rework of every
leaf spec. Before approval only read-only reconnaissance is allowed; the proposal package itself
lives in the conversation, a scratch directory, or the git-ignored local `docs/superpowers/specs/`
— never in `wiki/`, `docs/feature-tree/`, or the delivery directory. Same semantics as the
`planning-codebase-analysis` hard gate.

## Workflow

| Phase | What | Reference |
|---|---|---|
| 0 Anchor | Pin repository + **frozen commit** (the full 40-hex hash; "current worktree state / with uncommitted changes" is not a baseline — report uncommitted changes and let the user pick); declare scope (whole repo or subdomain) and host (this wiki or standalone deliverable) | — |
| 1 Build tree | Entry-surface inventory at symbol level → level-by-level functional decomposition → record the **tree (`nodes`) and every leaf's ownership in the manifest** → run `check_feature_tree.py --phase proposal --strict` until it reports zero → produce the **feature-tree proposal package** (tree, ownership manifest, leaf-row list, reconciliation report, exclusion table, pending judgment calls) | `references/tree-method.md` |
| ⛔ Approval | Present the proposal, **stop**, wait for explicit approval. Material tree changes (leaf splits/merges, ownership moves, new leaves surfaced by re-reconciliation) return to this gate. On approval the manifest moves into its host location | — |
| 2 Write specs | Wave by wave per module, status planned→spec'd. Dispatch writers **only** with `references/spec-writer-contract.md` — never with the mechanism-analysis writer contract, whose beats (rejected alternative, causal mechanism, execution trace) contradict the spec template. After each wave: `check_feature_tree.py --phase spec` (pages and leaf-id headings must exist) | `references/feature-point-template.md`, `references/spec-writer-contract.md` |
| 3 Verify | Independent reviewer (≠ writer) samples each wave per the rubric; the coordinator records every verdict as a `reviews:` row in the manifest; a leaf becomes `verified` only with a PASS row (checker V1) | `references/review-rubric.md` |
| 4 Deliver | `check_feature_tree.py --phase delivery` (implies `--strict`): every leaf `verified` with a PASS review, every spec page and leaf-id heading present, reconciliation still zero. The tree is not delivered while any leaf is `planned` or `spec'd` | — |
| 5 Baseline bump | **Re-enumerate every surface at the new commit and re-run the reconciliation** (tree-method §7). The commit delta only orders the re-verification of existing leaves; unowned new files, flags, entries, registrations are candidate leaves → tree change → approval gate. The manifest commit moves forward only after that | `references/tree-method.md` §7 |

## Mechanical checks vs manual review

Per `skills/README.md`, every rule states how it is verified:

| Rule | Verification |
|---|---|
| Manifest is well-formed and cannot zero on a vacuum: strict key whitelist, 40-hex commit, non-empty `nodes`/`leaves`, every include glob hits ≥1 file, non-empty scope, every exclusion has a reason | `check_feature_tree.py` X1 (schema, short-circuits everything else) and V0 (scope) |
| The tree is complete data: every node has a name and responsibility, every node/leaf parent chain resolves, no empty node, ids unique | T1 / T2 / D1 |
| Ownership ↔ frozen commit: files, flags (`Class.field` identity, ambiguous bare names rejected), symbol-level entries all claimed or excluded; no phantom ownership; every leaf's `path::symbol` entry anchor resolves to a real file and a real symbol at the frozen commit (AST for Python, identifier match otherwise) | F1–F4 / G1–G3 / E1–E2 / F3 |
| Status is backed by artifacts: `spec'd`/`verified` leaves have a real page with a leaf-id heading; `verified` leaves have a PASS review row; delivery only when all leaves are verified | S1 / S3 / V1 / V2 (`--phase spec`, `--phase delivery`) |
| Overview leaf-row table agrees with the manifest | S2 |
| Spec evidence is real at the frozen baseline | Reviewer samples stable source anchors per rubric R1; run `check_locators.py` only for changed pages that retain legacy `path:line` citations |
| Wiki output conventions (links/math) | `check_links.py --strict`, `check_math.py --changed --strict` |
| Leaf granularity, main-path logic, MECE judgment, a second behavior hiding in an already-claimed file, surfaces the manifest does not enumerate | Manual only (rubric R2/R5 + user sampling) |
| This skill's own behavior after an edit to it | Re-run `evals/scenarios.md` (live agent scenarios with pass criteria) |

## Host-specific rules

- **Host = this wiki**: manifest at `docs/feature-tree/<domain>.yaml`; tree overview page
  `NN_<repo>_feature_tree_analysis.md` (rendered from `nodes` + `leaves`) + one spec page per
  module (the `_analysis` suffix is on the whitelist); canonical baseline header
  `> **源码基线**：owner/repo@hex（branch，YYYY-MM-DD）`; Related Pages / changelog / all five
  quality gates apply in full; split a spec page over 500 lines by sub-module.
  **`kb_baseline` in `docs/radar/watchlist.yaml` is repo-wide** — `tools/radar.py` reports
  upstream drift against it for every page pinned to that repo. Advance it only when the feature
  tree covers the repo's whole wiki domain and every page pinned to that repo has been moved to
  the same commit (the same precondition `maintaining-llm-knowledge` step 7 states); a subdomain
  tree records its baseline in its own pages and leaves `kb_baseline` alone.
- **Standalone deliverable**: output directory set by the task; manifest at
  `<output_dir>/feature-tree.yaml` with an explicit `checkout:`; minimum gates = frozen-baseline
  header + `check_feature_tree.py` at zero in the current phase + rubric R1 source-anchor sampling.

## Red flags (rationalizations caught in live baseline testing and review)

| Rationalization | Correction |
|---|---|
| "Baseline: current worktree state (with uncommitted changes)" / "commit: HEAD" | Not a frozen baseline. Pin the 40-hex commit (the checker rejects anything else); report uncommitted changes and let the user decide. |
| "Cut modules by directory; one source file = one function item" | Directories and files are evidence, not the decomposition. A function item answers "what it does for the caller"; `models.py` and `utils/` are not features. |
| "The tree is in my head / in the Markdown; the manifest only needs leaves" | The tree is data: `nodes` with names, responsibilities, and parent links are mandatory, and every leaf's parent chain must resolve (T1). A drawing is a view. |
| "Internal helper functions count as function points; finer is better" | The leaf test is "independently triggerable + closable contract". An internal filter/normalize helper is one step of some function point's processing logic. More leaves ≠ better coverage; a zeroed reconciliation is. |
| "Test directories and data artifacts get tree nodes too" | Tests and artifacts are evidence and outputs: they go into the "tests & links" and "outputs" fields, never become nodes. |
| "The tree is done — write all the specs, then show the user" | Violates the hard gate. Reworking a hundred leaves costs far more than one approval round. |
| "The checker says zero, so the tree is complete" | Zero is the mechanical floor. File-level claims cannot see a second behavior inside an already-claimed file, and surfaces the manifest does not enumerate are invisible to it. Completeness still needs symbol-level entries, the full surface list, and rubric R2. |
| "Zero with `leaves: []` and a `**` exclusion, or an include that matches nothing, still counts" | The checker rejects it (X1/V0). A reconciliation over nothing reconciles nothing. |
| "Claim the flag by name; it is the same option everywhere" | Identity is `Class.field`. The same name in two classes is two contracts until each is claimed (G3). |
| "Baseline bump: map the changed files onto existing leaves and re-verify those" | New files, flags, endpoints, and registrations have no owner yet and never appear in a delta-to-leaf mapping. Re-enumerate all surfaces at the new commit and re-run the checker; the delta only orders the re-verification. |
| "Reuse the mechanism-analysis writer contract for spec writers" | Its beats contradict the spec template and turn specs into mechanism pages. Use `references/spec-writer-contract.md`. |
| "All specs are written; ship it" | Delivery needs `--phase delivery` at zero: every leaf `verified` by an independent PASS review, not merely `spec'd`. |
| "Supported scope can carry some policy/background too" | The five core fields have fixed semantics (see the template); supported scope takes only "supported / unsupported / defaults per dimension, with evidence". |
| "I skimmed it — close enough to write the spec" | A spot you have not opened yields no claim. If you cannot finish reading, mark the leaf planned for the next wave; never write a half-understood spec. |
