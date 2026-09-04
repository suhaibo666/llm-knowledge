# LLM Knowledge Wiki — Constitution

This file defines only what this knowledge base **is**: its layers, where content authority
lives, what provenance it demands, and which gates every change must pass.

**How to write documents is deliberately not here.** Those operations live in
[`skills/`](skills/README.md) and are **loaded on demand, never by default**. Read this file
before doing anything; open a skill only when you are about to do the thing it covers.

## Skill index (load on demand)

| Task | Skill |
|---|---|
| Add / edit / rename / merge a page in `wiki/`, maintain an `index.md` or the changelog | [`maintaining-llm-knowledge`](skills/maintaining-llm-knowledge/SKILL.md) |
| Write or change a LaTeX formula | [`writing-obsidian-math`](skills/writing-obsidian-math/SKILL.md) |
| Add or change a figure on a page — medium choice, house visual spec | [`drawing-wiki-figures`](skills/drawing-wiki-figures/SKILL.md) |
| Draw or change a Mermaid diagram (parser traps) | [`writing-mermaid-diagrams`](skills/writing-mermaid-diagrams/SKILL.md) |
| Discover and plan a new whole codebase or unplanned multi-page codebase domain, propose the document blueprint, wait for approval, then coordinate implementation | [`planning-codebase-analysis`](skills/planning-codebase-analysis/SKILL.md) |
| Write or review an approved/boundary-defined software architecture or concrete feature, write an approved/focused codebase mechanism, or analyze a paper, spec, dataset, incident, report, or other non-code artifact at any scale; load its document profile on demand | [`source-faithful-analysis`](skills/source-faithful-analysis/SKILL.md) |
| Inventory a repository as a feature tree down to leaf function points, each with a contract-style spec (I/O, processing logic, boundary constraints, supported scope); reconcile and re-verify it after baseline bumps | [`feature-tree-analysis`](skills/feature-tree-analysis/SKILL.md) |

There is exactly one copy of these skills, in `skills/`. `.claude/skills` is a symlink to it, so
Claude Code discovers them natively; Codex reaches them through `AGENTS.md`, which points at the
same directory. Never add a second copy under an agent directory — that duplication existed
before and silently drifted.

## Architecture

Three layers:

1. **Raw sources** (`raw/`) — the **source index** for document-type material: per-paper pages
   carrying the arXiv/official link and metadata, plus articles and diagram sources. Read-only;
   agents cite from it and do not rewrite it.
2. **Wiki** (`wiki/`) — the generated Markdown. Agents own this layer completely: create pages,
   update them, maintain cross-references, keep it consistent.
3. **This constitution** — defines structure and authority. The *how* lives in `skills/`.

`raw/` and `wiki/` are **independent trees**. They are not required to mirror each other
directory-for-directory. Most `wiki/` domains are sourced from external code repositories
(referenced as side-by-side checkouts, never copied into `raw/`) rather than from a matching
`raw/` subtree. What belongs in `raw/` is settled by the Provenance Policy below.

## Directory layout

- The functional tree (`wiki/01_theory/`, `wiki/02_engineering/`) is the **sole content
  authority**. The filesystem plus each level's `index.md` is the truth. Do **not** maintain a
  parallel deep ASCII tree in `CLAUDE.md` or `README.md` — it drifts from reality immediately.
- Each directory's `index.md` maintains **only its own entry table** (page list, one line each,
  status/segment). It does not describe the internals of descendant directories and does not
  draw a multi-level tree.
- `wiki/index.md` (the master index) keeps **only the domain-level table** (entry point, one
  line, page count, status). It does not repeat what a lower `index.md` already says and does
  not draw a deep tree. Page counts are recomputed with `find`/`rglob` and corrected as content
  drifts — never carried forward from a one-off count.
- `README.md` addresses readers outside the repo: it describes directory responsibilities and
  hierarchy. Precise statistics belong to the domain table in `wiki/index.md`.

## Courses (reading-path layer)

`wiki/courses/` sits above the functional tree as a **pure reading-path layer**, answering
"in what order should I read this" across directories (e.g. torch.compile end to end,
post-training frontier).

- A learning domain or course series may only be created as **one guide page under
  `wiki/courses/`**. Do not create a separate vertical directory for it. (Historic
  `19_torch_compile_end_to_end/` and `03_posttraining/` were exactly this anti-pattern and were
  dissolved into the functional tree during the kb-reorg.)
- A course page contains **only**: reading order, `[[wiki links]]` into the functional tree, and
  a one-line orientation per entry. It **must not carry body content** — it may not restate
  mechanisms that a functional-tree page already covers, and must never become a second source
  of truth.
- To add substance, write it into the corresponding functional-tree page, then add a link plus a
  one-line orientation from the course page. If a course page contradicts or lags the functional
  tree, fix the functional-tree page; the course page only updates links, order, and orientation.
- The purpose of this rule is to close the recurring root cause of "a new learning domain
  re-explains old content".

## Provenance policy

- **`raw/` holds document-type sources only**: paper source pages, articles, diagram sources
  (`.eddx`/`.html`, …). Do **not** copy an external code repository into `raw/` — code sources
  are referenced as a side-by-side checkout with a pinned commit.
- **A code-analysis page must pin "repository + commit baseline" in its header.** Any page
  produced by analysing a code repository (PyTorch, Megatron-LM, vLLM, torch_npu, …) states the
  repository and the exact commit/tag in its header (e.g. `baseline main @ 8a694930`). Later
  analyses of the same repository either match that baseline or declare a new one explicitly —
  never a vague "latest code".
- There is **no** blanket requirement that every claim trace back to a file inside `raw/`. Most
  engineering pages are sourced from code repositories, not `raw/` documents. It counts as
  source-faithful when document-type sources are findable in `raw/` **and** code-type sources
  have their repository + commit baseline pinned in the page header.
- The governing principle remains **source fidelity**: every claim is grounded in evidence opened
  at the declared baseline, and on conflict the source wins. Code pages default to stable anchors
  (`repository-relative path + qualified symbol/config key/test name`) collected in a compact
  source-reading route; volatile line numbers are optional, not a per-claim requirement. Document
  sources keep their natural locators (`§`/Table/Eq, a `raw/` filename). Method:
  [`source-faithful-analysis`](skills/source-faithful-analysis/SKILL.md).

## Quality gates

Gates are **tiered by what the change touches**. Running the whole battery for a content edit
costs ~9 minutes and validates nothing that edit could have broken. Run the tier that matches.
Per-rule guidance lives in the matching skill.

### T0 — every change

```bash
python tools/check_links.py --strict              # wikilinks: broken/ambiguous/bare_index/orphans must be 0
python tools/check_math.py --changed --strict     # formulas in this change: 0 errors AND 0 warnings
python tools/check_markdown.py --changed --strict # list markers / mermaid labels that break rendering
python tools/check_assets.py --changed --strict   # images and local resources resolve
```

~6s. Content edits (`wiki/`, `docs/`, `raw/` — 78% of commits) stop here.

### T0 conditional — only when the change hits one of these

| Change hits | Add | Cost |
|---|---|---|
| a page still carrying explicit `path:line` citations | `python tools/check_locators.py --changed` | ~10s |
| `wiki/02_engineering/01_pytorch/**` or `wiki/courses/**` | `pytest tools/labs_torch_compile/test_volume_demo_contract.py -q` | 33s |
| you need built-page proof (assets, anchors, formula rendering) | `python -m tools.mkdocs_site.cli build --changed` then `node tools/mkdocs-site/mathjax-corpus.mjs --pages <changed>` | 73s + ~5s |

Both steps read the **built** `site/` and mean nothing without one. `--changed` narrows the
validation walk to the changed routes (75s → 11s over 50 pages) and `--pages` narrows the formula
render (118s over 162 pages → ~5s over three); the mkdocs build itself (~31s) is all-or-nothing.
Two limits are deliberate: a scoped run **skips `orphans`** and prints it as skipped rather than
zero, and it only sees references *out of* the changed pages — a heading renamed here that breaks
an inbound link from elsewhere is caught by `check_links`, which is cheap enough to stay
repo-wide. `check_math` is the syntax gate and needs no build at all.

No other test under `tools/` reads the real `wiki/` tree — the rest run against `tmp_path`
fixtures and cannot be broken by a content edit. Do not run them for one.

### By touched area

| Touched | Gate | Cost |
|---|---|---|
| `skills/` | `pytest tools/test_math_skill.py tools/test_planning_codebase_analysis_skill.py tools/test_source_faithful_analysis_skill.py tools/test_source_faithful_architecture_profile.py -q` | 19s |
| `tools/check_*.py`, `tools/radar.py` | `pytest tools/test_check_*.py tools/test_radar.py -q` | 25s |
| `tools/mkdocs_site/`, `tools/mkdocs-site/`, `mkdocs.yml` | `npm run docs:mkdocs:test` | 8m30s |
| `tools/labs_torch_compile/` | `pytest tools/labs_torch_compile -q` | 145s |
| `tools/docs-site/` | `npm run docs:test` | 60s |

### Before push / merge — once, not per edit

```bash
python -m pytest tools/ -q      # ~5min, 429 tests
npm run docs:mkdocs:test        # 8m30s — the stack CI actually deploys
```

`npm run docs:test` covers `tools/docs-site`, which `.github/workflows/pages.yml` does **not**
deploy; it is a gate for that stack only, never a proxy for the published site. The one check
that genuinely cannot be narrowed is **`orphans`** — the whole built tree minus everything a route
or a scanned reference owns — so only the unscoped `python -m tools.mkdocs_site.cli build` reports
it, which is why it belongs here. `broken_links`, `missing_anchors`, `missing_assets` and
`missing_legacy_routes` are per-page, and `--changed` covers them at T0 cost.

### Baselines (2026-09-04)

| Checker | Scope | Baseline |
|---|---|---|
| `check_links --strict` | 446 pages | **0** broken / ambiguous / bare_index / stale_section / orphans |
| `check_math --strict` | 446 pages | **0 errors, 0 warnings** |
| `check_markdown` | 446 pages | **0 errors**, 120 `MD003` warnings |
| `check_assets` | 446 pages | **0 errors, 0 warnings** |
| `check_locators` | 444 pages, 5197 citations | **errors=16, warnings=471, env=77** |

Any new finding against a zero baseline was introduced by the change in front of you — do not
wave it through as pre-existing debt.

`check_markdown`'s 120 warnings are all `MD003`, quoted mermaid pipe labels.
[`writing-mermaid-diagrams`](skills/writing-mermaid-diagrams/SKILL.md) classifies those as
renderer-dependent rather than broken, and says existing diagrams that render need not be
rewritten — so write new diagrams without the quotes and leave the old ones alone.
`--changed` reports only what this change introduced (findings are matched by rule plus
offending line, so a legacy warning that merely shifted does not resurface); pass a path
instead of `--changed` to see a file's full standing count.

`check_locators` remains a **conditional legacy gate**, never an always-on authoring
requirement: run it from the T0 table when a changed page still carries explicit
`path:line` citations. New code-analysis prose uses stable symbol anchors instead.
It grades its findings three ways: `missing_file` is an error; `out_of_range`,
`ambiguous`, `unanchored_page` and `unknown_repo` are page defects an author can fix;
`unresolved_repo`, `commit_unavailable` and `unverifiable` are **environment gaps** — this
machine lacks the checkout or the commit — reported separately and kept out of the exit code
unless `--include-env`. What the split exposes is worth acting on: only **4** of the 471 warnings
concern line numbers. The bulk is bare filenames that resolve ambiguously (250) and pages
carrying code citations with no pinned baseline header (183) — both are provenance defects this
constitution already requires fixing, not line-number noise.
