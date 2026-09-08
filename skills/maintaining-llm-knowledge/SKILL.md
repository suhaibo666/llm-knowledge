---
name: maintaining-llm-knowledge
description: Use when creating, updating, renaming, merging, indexing or auditing pages in the llm-knowledge wiki. Load it before writing into wiki/, not by default.
---

# Maintaining the LLM Knowledge Wiki

`CLAUDE.md` is the constitution: it fixes what the wiki *is* (three layers, the functional tree as the only content authority, provenance, quality gates). This skill carries the *operations*: how to add, name, link, merge and audit a page. Load it on demand when you are about to write into `wiki/`.

Formula and diagram conventions live in their own skills: [`writing-obsidian-math`](../writing-obsidian-math/SKILL.md) and [`writing-mermaid-diagrams`](../writing-mermaid-diagrams/SKILL.md). The decomposition method for turning a source into pages is [`source-faithful-analysis`](../source-faithful-analysis/SKILL.md).

### Page Types

| Type | Suffix | Purpose | Example |
|------|--------|---------|---------|
| Index | `index.md` | Domain entry point, directory contents & link map | `01_theory/index.md` |
| Entity | `*_analysis.md` | Deep analysis of a specific paper/technology | `11_muon_analysis.md` |
| Guide | `*_guide.md` | How-to or implementation walkthrough | `20_npu_lowering_guide.md` |
| Quickstart | `*_quickstart.md` | Minimal path to a working example before the deep dive | `01_autograd_engine_quickstart.md` |
| Deepdive | `*_deepdive.md` (no separator inside `deepdive`; not `_deep_dive`) | Focused deep dive on one subtopic within a module | `22_npu_fusion_passes_deepdive.md` |
| Comparison | `comparison.md` | Side-by-side comparison of approaches | `npu/30_comparison.md` |
| Changelog | `changelog.md` | Chronological log of all ingest operations | `wiki/changelog.md` (historical entries are archived by quarter under `wiki/changelog/`) |

**Only these six page suffixes are allowed**: `_analysis`, `_guide`, `_quickstart`, `_deepdive`, `comparison`, `index`. The changelog is the dedicated log file above. Do not introduce legacy suffixes such as `_deep_dive`, `_report`, `_methodology`, `_overview`, `_map`, `_model`, `_concepts`, `_details`, `_diagrams`, or `_v2`.

**Do not create `README.md` inside `wiki/`.** Every directory uses `index.md` as its entry point. `README.md` is a repository-level file for human visitors, not a wiki page type.

**Segment numbering:** Content filenames have a two-digit `NN_` prefix (`index.md` is unnumbered). The tens digit identifies the segment and establishes a reading order from introductory to advanced:

| Segment | Number range | Meaning |
|---|---|---|
| 0 | `01`–`09` | Introduction / orientation: quickstarts, knowledge maps, overviews |
| 1 | `10`–`19` | Core mechanisms: `_analysis` pages in pipeline or learning order |
| 2 | `20`–`29` | Deep dives / special topics: `_deepdive`, focused analyses, peripheral mechanisms |
| 3 | `30`–`39` | Methods / comparisons / engineering practice: development guides, `comparison`, troubleshooting |

If a segment exceeds capacity, use an adjacent empty segment and explain this in the directory's `index.md` segment table. Apply the same rule recursively in hardware subdirectories such as `npu/` or `cuda/`; their numbering is independent of the parent. Numbering is optional for small directories with fewer than four content pages.

### Naming Conventions

- Filenames use `snake_case`: no uppercase letters, camelCase, or dots inside the stem. Correct nonconforming names; for example, use `kimi_k2_5`, not `kimi_k2.5`.
- Index pages are always named `index.md`: one unnumbered entry point per directory.
- One concept per page; prefer splitting over merging distinct subtopics. For overlapping topics, rather than natural subdivisions of one concept, apply **Merge over coexist** below.

### Ingest Workflow

When a new source is added to `raw/`, follow this sequence:

1. **Read** the source document thoroughly.
2. **Discuss** key takeaways with the user before writing.
3. **Create** a new wiki page in the correct functional-tree module, or update an existing page if the topic is already covered. Follow the page types, naming conventions, and segment numbering above.
4. **Update** the domain `index.md` with the new page: entry table and segment, without a deep tree.
5. **Cross-reference**: Add `[[wiki links]]` to and from related existing pages, following the Cross-Reference Rules below.
6. **Append** an entry to `wiki/changelog.md` describing what was added or updated. Escape illustrative `[[...]]` syntax with backticks, as explained below.
7. **Update the radar baseline when the entire repository domain advances.** `docs/radar/watchlist.yaml` has one repository-wide `kb_baseline`; `tools/radar.py` uses it to assess drift across **all** baseline-pinned pages for that repository. Update it only after all those pages have advanced to the same commit in a whole-domain wave. If only one subdomain or feature tree advances, keep its new baseline in its own header and leave the repository-wide field unchanged, consistent with the host rules in `feature-tree-analysis`. Omitting the repository-wide update after a completed wave causes repeated reports of already-resolved drift.
8. **Flag contradictions**: If new information contradicts existing wiki content, preserve both claims and add a `> [!contradiction]` callout.

### Cross-Reference Rules

- Every page MUST contain a `## Related Pages` section at the bottom. `index.md` is exempt because it is already a link map. Select **3–7 links**, each followed by **one sentence** explaining the relationship. More than seven suggests that the page needs a narrower scope, rather than a longer link list.
- **Bare basenames are valid by default.** Content filenames are globally unique, so `[[page_name]]` needs no display alias unless the link acts as a sentence's subject/object and its filename does not make its meaning clear. Prefer an alias when in doubt.
- **Index links require a path and a display alias.** `index.md` is not globally unique; bare `[[index]]` is ambiguous. Use `[[<path relative to wiki root>/index|<semantic directory name>]]`, for example `[[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM]]`. Links to a directory also target its `index`, not a bare directory name.
- **Do not use `../` paths.** Write link paths relative to the `wiki/` root, such as `01_theory/06_distributed_parallelism/index`. Parent traversal paths become fragile after directory moves and are harder to check statically.
- **Escape example links.** When discussing `[[...]]` as syntax rather than an intended live link, wrap it in backticks, such as `` `[[index]]` ``. This includes changelog descriptions of historical renames, for example from `RL_PPO_Loss_and_GRPO_Analysis` to `rl_ppo_loss_and_grpo_analysis`.
- Every new page MUST link to at least one existing page.
- When updating a page, check whether other pages should gain a backlink.
- Acceptance: `python tools/check_links.py --strict` must report broken=0, ambiguous=0, and bare_index=0. See the checker for its full set of rules.

### Update Principles

- **Merge over coexist:** Merge overlapping coverage into an authoritative page instead of keeping two versions with reciprocal links. First identify the authoritative version by completeness, baseline freshness, and granularity; absorb the other page's unique contributions. **Repair every inbound link before deletion**, then record the merge and destination in `wiki/changelog.md`.
- Mark outdated claims with `> [!deprecated] Updated by [[page_name]]`.
- Mark contradictions with `> [!contradiction] See also [[page_name]]`.
- Record the date of each significant update in the page header.
- When a page grows too large (>500 lines), propose a split to the user.

### Quality Standards

- Skill instructions are written in English. **Wiki output language follows the user's request or the host's established page/domain convention**; if neither supplies a convention, use the source material's language. English skill prose does not require English wiki pages. Preserve exact source identifiers and required output-format literals.
- Use Mermaid for architecture, data-flow, and sequence visualizations, following [`writing-mermaid-diagrams`](../writing-mermaid-diagrams/SKILL.md). Load [`drawing-wiki-figures`](../drawing-wiki-figures/SKILL.md) to choose the appropriate medium for each figure.
- Use LaTeX for mathematical formulas, following [`writing-obsidian-math`](../writing-obsidian-math/SKILL.md).
- For code analysis, pin repository + commit and use a compact source-reading route of repository-relative paths plus qualified symbols/config keys/test names. Line numbers are optional for exact excerpts or ambiguous spots, not a per-claim default.

### Reader Navigation and Progressive Disclosure

- Within the constitution's entry-table limits, make index entries explain what a reader can learn or accomplish. Use semantic link labels for reading routes instead of number-only sequences that require a lookup.
- Keep cross-directory learning sequences in `wiki/courses/`: order, links, and one-line orientations only. Put explanations in their authoritative functional-tree pages and link to them.
- Introduce the reader's problem and a concrete ordinary case before detailed variations, configuration inventories, and failure cases. Choose the explanatory focus through `source-faithful-analysis`; state ownership is relevant when persistent state or coordination explains the behavior, not a universal outline for algorithms.
- Link to a prerequisite or deeper section where the reader needs it, with a short reason to follow the link. Keep the final Related Pages list curated. Navigation complements substantive coverage; it does not replace missing explanations.
- When rewriting, preserve unique concepts, examples, corrections, configuration coverage, and link destinations, following the conservation rules in `source-faithful-analysis`. Manual reading review should check that a reader can enter through the index, follow the ordinary example, and reach the relevant deeper explanation without guessing page numbers.

### Baseline Header Convention

Code-analysis pages must pin an implementation baseline. This also supplies parsing input for legacy line citations checked by `tools/check_locators.py`. **Use the canonical format for new pages**, one line per repository. The Chinese label and punctuation below are exact host output-format literals; the placeholders are English instructions:

```
> **源码基线**：`owner/repo@<full hex or at least 12 hex characters>`（`branch`，YYYY-MM-DD）
```

- When a page analyzes multiple repositories, give each its own baseline line, either in the header or the relevant section. Default to stable anchors such as `path::qualified.symbol`. If retaining a repository's legacy `path:line` citations, pin its commit so the checker can attribute them.
- The checker tolerates historical formats such as `verl main @ 254a23ed` and `name vX@hex`; do not introduce them in new content.
- The repository needs an entry and a local `checkout:` in `docs/radar/watchlist.yaml`; otherwise its references can only be reported as unresolved/unverifiable warnings, not verified.
- Conditional acceptance: only when an edited page still contains explicit `path:line` citations, run `python tools/check_locators.py --dir <affected-domain>`. This validates legacy references; it does not require new content to produce line numbers.

### House Page Shape for Analysis Pages

The analysis method belongs to [`source-faithful-analysis`](../source-faithful-analysis/SKILL.md). That portable contract defines **semantic order**, not a heading template. The conventions below define this wiki's page presentation without depending on an example page that may later change.

The header is a blockquote with the following order and no empty lines by default. The four Chinese labels and their punctuation are exact host output-format literals; write the placeholder content in the page's output language:

```
> **源码基线**：`owner/repo@<hex>`（`branch`，YYYY-MM-DD）
> **主题**：Two or three sentences describing this page's topics in body order; optionally name the core code directory.
> **适用范围**：One line stating the page's scope and where adjacent topics belong.
> **最近更新**：YYYY-MM-DD. One sentence.
```

The header answers what the page covers; it does not preview the argument. Put claims, exceptions, and reasoning in the body; expanded source paths in the source-reading route; and changed-section details in `wiki/changelog.md`. Older headers, before 2026-09-06, included separate core-source and central-conclusion fields. Migrate those headers while editing their pages; **do not run a separate bulk migration**.

Baseline history, old-baseline notes, and declarations of narrative order belong in `wiki/changelog.md`, not the header.

The following tables are **optional presentation examples**, not three mandatory sections or a completeness checklist. Use the shape that helps explain the page's actual subject; prose, a worked example, or a figure may be clearer.

| Purpose | Example columns |
|---|---|
| Benefits and costs, in a feature overview | Dimension \| Direct benefit \| Required cost or boundary |
| Hard constraints and failure boundaries, in a constraints section | Precondition \| Source boundary \| Behavior when violated |
| Configuration contract, near the end | Field \| Type \| Default \| Contract |

- **Ground each hard constraint in evidence appropriate to its behavior.** If the implementation has an explicit `assert`, `raise`, or warning, name that location. If there is no explicit guard, state that and anchor the enforcing computation, protocol invariant, or relevant test; explain the supported consequence, such as numerical error, an invalid result, or stalled progress. Do not invent an exception or treat the absence of a guard as evidence that no constraint exists. If the violation behavior has not been verified, label it as unverified. These semantic checks require source review; keyword counts cannot establish them.
- **Group configuration contract tables by configuration class.** End each subsection with one line stating the class's total field count, how many the table covers, and where the remaining fields' owners are recorded in a coverage ledger such as `docs/coverage/megatron-lm.yaml`. Put the ledger path in that line, not in the heading.
- During a rewrite, **retain every field assigned to this page by the coverage ledger**; see the conservation rule in `source-faithful-analysis`. `tools/check_coverage.py` C2 rechecks only **manual** owners, not mention coverage for `auto: true` rows. This requirement therefore needs manual/process review beyond the automated gate.

### MCP Tools

Two MCP servers are configured in `.mcp.json`:

1. **filesystem** (`@modelcontextprotocol/server-filesystem`) — Read, write, search, and manage files in `wiki/` and `raw/`. Use for all file operations when this connector is available; otherwise use the runtime's file tools.
2. **qmd** (`qmd mcp`) — Search engine over wiki pages. Currently operating in **BM25 keyword mode only** (no embedding model). Use `search` for keyword matching; `vsearch` and `query` require embeddings (not yet available on this system).

When to use which:
- **filesystem** for: reading/writing files, listing directories, exact text search.
- **qmd** for: keyword search across all wiki pages, finding pages that mention a specific term, cross-reference discovery. If unavailable, use local keyword search.

### Query Workflow

When the user asks a question:

1. **Search wiki first**: Use `qmd search` to find wiki pages matching relevant keywords, or local keyword search if the connector is unavailable.
2. **Navigate the graph**: Check the relevant `index.md` to understand the domain landscape and follow `[[wiki links]]`.
3. **Read the pages**: Use filesystem `read_file`, or the runtime's file tools, to read the full content of the most relevant pages.
4. **Synthesize**: If the answer requires synthesizing multiple pages, do so and note which pages contributed.
5. **Check raw sources on gap**: If the answer is NOT in the wiki, do NOT just say "not found". Instead:
   - Scan `raw/` for relevant source documents using topic keywords in filenames; for code-based topics, check whether a relevant sibling repository checkout is already referenced elsewhere in the wiki.
   - If a relevant source exists, **automatically ingest it** following the Ingest Workflow, then answer from the newly created wiki content.
   - If no relevant source exists, say so and offer to create a stub wiki page in the correct functional-tree module, not a new standalone learning-track directory; see the constitution's Courses rules.
6. **Grow the wiki**: Every query that reveals a gap should result in either a new wiki page or a note under Knowledge Gaps in the relevant `index.md`.

### Maintenance Workflow

Periodically, or when the user requests:

1. **Consistency check**: Run `python tools/check_links.py --strict`; broken, ambiguous, and bare_index must be zero.
2. **Math check**: Run `python tools/check_math.py --changed --strict`; formula errors and warnings introduced by the change must be zero.
3. **Orphan check**: `check_links.py` covers pages with no inbound links and no mention in an `index.md`. Integrate each orphan into its corresponding index.
4. **Contradiction review**: Scan for `> [!contradiction]` callouts and propose resolutions.
5. **Staleness review**: Flag pages that have not been updated in more than 30 days for review.
6. **Duplication review**: When multiple pages repeat the same topic, rather than explain natural implementation differences, apply **Merge over coexist**.
