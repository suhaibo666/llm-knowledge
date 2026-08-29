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
| Write an approved/focused codebase unit, or analyze a paper, spec, dataset, incident, report, or other non-code artifact at any scale | [`source-faithful-analysis`](skills/source-faithful-analysis/SKILL.md) |

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
  re-explains old content" (see `docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md`
  §1).

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
- The governing principle remains **source fidelity**: every claim carries a verifiable locator
  (`file:line`, `§`/Table/Eq, a `raw/` filename), and on conflict the source wins. Method:
  [`source-faithful-analysis`](skills/source-faithful-analysis/SKILL.md).

## Quality gates

Every change must pass these before it lands. Per-rule guidance lives in the matching skill.

```bash
python tools/check_links.py --strict          # wikilinks: broken/ambiguous/bare_index/orphans must be 0
python tools/check_math.py --changed --strict # formulas in this change: 0 errors AND 0 warnings
python -m pytest tools/                       # the maintenance tooling's own tests
npm run docs:test                             # local docs site: unit tests + end-to-end check
```

Repo-wide baseline: as of 2026-08-26, `check_links` and `check_math --strict` report **0 errors
and 0 warnings across all 409 pages**. Any new finding was introduced by the change in front of
you — do not wave it through as pre-existing debt.
