# Knowledge Metadata Synchronization Design

> **Date:** 2026-09-01  
> **Repository baseline:** `main @ 97a23022d31a72b0b513497e2bfa577f10e6cf76`  
> **Status:** approved in chat; awaiting written-spec review

## 1. Problem

The knowledge base changes faster than its reader-facing summaries. Article additions, removals,
renames, and domain reorganizations currently require an agent to remember several independent
follow-up edits:

- update recursive page counts in `wiki/index.md`;
- update duplicated counts in `README.md`;
- repair Markdown links in `README.md` after page renames;
- reconsider claims such as “most-linked page” when inbound-link rankings change;
- keep the documented maintenance workflow aligned with `CLAUDE.md` and `skills/`.

This memory-based workflow has already drifted at baseline `97a2302`:

- `README.md` says 409 pages while `wiki/**/*.md` contains 437 files;
- four README domain counts differ from the filesystem;
- one README link points to the retired
  `03_vllm_request_flow_walkthrough_analysis.md`;
- the README calls DeepSeek-V3 the most-linked model page even though DeepSeek-V4 currently has
  more inbound links;
- the README says workflows live in `CLAUDE.md`, while the current constitution deliberately puts
  operational workflows in `skills/`.

The underlying failure is duplication: volatile facts are copied into prose that has no mechanical
owner. Updating today's numbers without changing that ownership model would only reset the drift
clock.

## 2. Goals

1. Make deterministic repository facts mechanically maintainable.
2. Keep human-authored descriptions, reading recommendations, and semantic choices under human or
   agent control.
3. Detect README link breakage and prohibited volatile claims before merge.
4. Give agents and maintainers one local command to refresh metadata and one read-only command for
   CI.
5. Integrate the commands into the repository's existing constitution, maintenance skill, tool
   documentation, and GitHub Actions.
6. Make generated updates narrow, deterministic, reviewable, and idempotent.

## 3. Non-goals

- Generate article summaries from filenames or page text.
- Select “best” or “core” articles automatically.
- Rewrite arbitrary README prose.
- Generate every directory `index.md`; local index descriptions and reading order remain semantic
  content.
- Update `wiki/changelog.md` or radar `kb_baseline` without an agent deciding what changed.
- Let CI commit generated changes back to a contributor branch.
- Introduce pre-commit, a new package manager, or a third-party Markdown parser.

## 4. Considered approaches

### A. Generate README and indexes completely

This maximizes automation, but it requires a second metadata manifest for descriptions, ordering,
status, and featured pages. The manifest becomes another source of truth, while generated output
obscures editorial changes. It also cannot infer whether a page deserves to be featured.

### B. Check only

A linter could report stale counts and broken README links without writing anything. This is safe,
but every article addition would still require repetitive manual edits and would not satisfy the
goal of automatic deterministic maintenance.

### C. Hybrid synchronization — selected

Remove volatile computed facts from README, annotate only the numeric cells in `wiki/index.md`,
write those cells deterministically, and validate everything else. This assigns one owner to each
kind of content:

| Content | Owner | Automation behavior |
|---|---|---|
| Recursive domain counts | Filesystem inventory | write and check |
| README local Markdown links | Filesystem | check only |
| README volatile ranking/count claims | Repository policy | check and reject |
| Domain descriptions/status | Human or agent | preserve exactly |
| Featured/core article selection | Human or agent | preserve; validate links only |
| Local directory index membership | Human or agent + `check_links.py` orphan gate | check only |
| Changelog and radar baseline | Human or agent semantic judgment | preserve |

This is the smallest design that removes recurring arithmetic work without pretending that prose is
machine-generable.

## 5. Reader-facing refresh

The first implementation applies a one-time cleanup before the new checker becomes authoritative.

### 5.1 `README.md`

- Replace “Claude Code Agent” with agent-neutral wording that names Claude Code and Codex as the
  supported maintainers.
- Remove the repository-wide page total.
- Remove the `篇数` column from both second-level overview tables. Keep links and one-line domain
  descriptions.
- Remove the fixed “409 pages” build-size statement.
- Replace the inbound-ranking promise with a stable editorial description such as “精选阅读入口”.
- Remove “most-linked” and exact inbound-count claims.
- Point the retired vLLM request-flow entry at
  `wiki/02_engineering/03_infer_frameworks/vllm/03_vllm_architecture_overview_analysis.md` and align
  its title/description with that page's responsibility and request-lifecycle scope.
- Align quality-gate examples with the constitution.
- Explain that `CLAUDE.md` is the constitution and `skills/` contains operational workflows.
- Scope the scheduled-radar sentence to the maintainer machine rather than presenting host-local
  state as a repository guarantee.

README remains a stable reader entry point. Precise counts live only in `wiki/index.md`.

### 5.2 `wiki/index.md`

- Recompute every existing domain-row count recursively from the directory linked by that row.
- Count tracked-shape Markdown pages by filesystem path, including nested `index.md` pages and
  excluding any file named `SUPERSEDED.md`.
- Update the page's `最后更新` date to 2026-09-01 during the one-time refresh.
- Replace prose that carries old per-domain recount dates with a stable statement that counts are
  generated from the filesystem.
- Keep entry names, links, descriptions, status, quick navigation, and all other prose unchanged
  unless a separately verified stale link requires repair.

Each generated number is wrapped in a narrow marker inside its existing table cell:

```markdown
<!-- knowledge-meta:count path=01_theory/01_models -->67<!-- /knowledge-meta:count -->
```

The path is relative to `wiki/`. The marker is intentionally local to one numeric cell: the tool
never owns or regenerates the surrounding row.

## 6. Synchronization tool

Create `tools/sync_knowledge_meta.py` with a small importable core and CLI wrapper.

### 6.1 Inputs and inventory

The tool resolves the repository root from its own file location, then reads:

- `wiki/**/*.md` for page inventory and wikilinks;
- `wiki/index.md` for count markers;
- `README.md` for local Markdown links and volatile-claim validation.

The inventory follows symlink-safe filesystem traversal already used by repository tools. A page is
countable when it ends in `.md` and its basename is not `SUPERSEDED.md`.

### 6.2 Count-marker contract

For every `knowledge-meta:count` marker, the tool must:

1. parse one relative POSIX path;
2. reject absolute paths, `..`, duplicate paths, malformed/nested markers, and targets outside
   `wiki/`;
3. require the target directory to exist;
4. compute its recursive Markdown count using the shared inventory rule;
5. compare the rendered integer with the computed value.

`--write` changes only the integer between a valid marker pair. It does not reorder rows, change
labels, or add markers heuristically. A missing marker is a reviewable source change and must be
added explicitly.

### 6.3 README contract

The checker scans ordinary inline Markdown links in `README.md`, excluding external URLs, pure
anchors, fenced code, and inline code. A local link must resolve from the repository root after its
optional fragment is removed. Missing targets are errors in both modes; `--write` does not guess a
replacement.

The checker also rejects the narrow volatile patterns that caused the current drift:

- repository/domain page-count prose such as `全库共 N 篇`, `全量 N 页`, or a `篇数` table column;
- exact inbound-link counts such as `N 次入链`;
- ranking claims such as `被引最多` or an assertion that featured entries are ordered by current
  inbound count.

Tool-version requirements such as Node 22 and time windows such as seven days are not page counts
and must remain valid README content.

### 6.4 CLI

Exactly one mode is required:

```bash
python tools/sync_knowledge_meta.py --write
python tools/sync_knowledge_meta.py --check
```

- `--write` updates valid stale count markers atomically, then runs all read-only validations.
- `--check` performs no writes and exits non-zero when a marker is stale or any contract error is
  present.
- Both modes print a compact summary: marker count, updated/stale count, README local-link count,
  and error count.
- Diagnostics name the file, line, rule, observed value, and expected value or missing target.
- File writes use UTF-8, preserve the rest of the file byte-for-byte, and replace via a temporary
  sibling file only after all markers parse successfully.

The tool has no network access and no third-party dependency.

## 7. Workflow integration

### 7.1 Agent workflow

Update `skills/maintaining-llm-knowledge/SKILL.md` so any add, delete, rename, merge, or relocation
under `wiki/` ends with:

```bash
python tools/sync_knowledge_meta.py --write
python tools/sync_knowledge_meta.py --check
```

Ordinary edits that do not change inventory still run `--check`, because they can rename or remove
a README-linked page indirectly during a broader change.

### 7.2 Constitution and tool documentation

- Add `python tools/sync_knowledge_meta.py --check` to the `CLAUDE.md` quality gates.
- Document both modes and the generated-marker boundary in `tools/README.md`.
- Keep `README.md` focused on reader usage; it lists the check command but does not explain internal
  marker mechanics.

### 7.3 Continuous integration

Add `.github/workflows/knowledge-quality.yml` for pull requests and pushes to `main`. The initial
workflow installs Python 3.13 and runs:

```bash
python tools/sync_knowledge_meta.py --check
python tools/check_links.py --strict
python tools/check_math.py wiki --strict
python -m pytest tools/test_sync_knowledge_meta.py tools/test_check_links.py tools/test_check_math.py
```

CI remains read-only. A stale generated value fails with an instruction to run `--write` locally.
The existing Pages workflow remains responsible for deployment and is not made responsible for
repairing repository sources.

## 8. Testing strategy

Create `tools/test_sync_knowledge_meta.py` using temporary repository fixtures. Tests cover:

1. recursive counts include nested indexes and exclude `SUPERSEDED.md`;
2. `--check` reports a stale count without changing the file;
3. `--write` updates only the number between markers;
4. a second `--write` is byte-identical and reports zero updates;
5. malformed, duplicate, absolute, escaping, and missing-directory markers fail before any write;
6. README local links accept anchors/external URLs and reject a missing local file;
7. fenced and inline-code examples are not treated as live links;
8. volatile page-count and inbound-ranking phrases fail while Node versions and radar time windows
   pass;
9. the real repository passes `--check` after the one-time refresh.

The full repository gates run after implementation. This change contains no formulas or diagrams,
but the standard gates still verify that the documentation edits did not disturb unrelated content.

## 9. Failure boundaries

- The tool cannot infer a description, status, course order, featured page, changelog entry, or
  radar baseline. It reports related breakage and stops.
- A new `wiki/index.md` row without a count marker is not silently adopted. Reviewers must decide
  whether the row is a count-bearing domain entry and add the marker explicitly.
- A page can be present in the filesystem yet missing from its local domain index. That remains the
  responsibility of `check_links.py` orphan/index coverage and the maintenance skill.
- Renaming a README-linked page causes CI failure until a human or agent selects the correct new
  owner. Filename similarity is not a safe semantic migration rule.
- CI detects drift but never pushes changes. This avoids surprising contributor-branch mutations
  and keeps generated diffs reviewable.

## 10. Acceptance criteria

1. README contains no computed page totals, per-domain page-count column, exact inbound count, or
   live inbound-ranking claim.
2. Every README local Markdown link resolves.
3. Every count-bearing row in `wiki/index.md` has one valid marker and displays the current recursive
   count.
4. `--write` is idempotent and modifies no human-authored text outside marker payloads.
5. `--check` fails on count drift, malformed markers, missing README links, and prohibited volatile
   claims.
6. The maintenance skill and constitution name the new commands.
7. Pull-request CI runs the read-only check.
8. The repository's link, math, Python-tool, and documentation-site gates pass.

