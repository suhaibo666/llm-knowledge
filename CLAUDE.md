# LLM Knowledge Wiki — Schema & Maintenance Rules

This document governs how the LLM agent maintains this knowledge wiki. Read it before any wiki operation.

## Architecture

This wiki has three layers:

1. **Raw sources** (`raw/`) — Immutable source documents (PDFs, papers, articles). The LLM reads from them but NEVER modifies them.
2. **Wiki** (`wiki/`) — LLM-generated markdown files. The LLM owns this layer entirely. It creates pages, updates them, maintains cross-references, and keeps everything consistent.
3. **This schema** (`CLAUDE.md`) — Tells the LLM how the wiki is structured, what conventions to follow, and what workflows to execute.

## Directory Layout

完整的 raw/ 和 wiki/ 目录结构分别由各层级的 `index.md` 维护，详见 [[index]]（总索引）和各领域入口。

## Page Types

| Type | Suffix | Purpose | Example |
|------|--------|---------|---------|
| Index | `index.md` | Domain entry point, directory contents & link map | `llm/index.md` |
| Entity | `*_analysis.md` | Deep analysis of a specific paper/technology | `muon_analysis.md` |
| Guide | `*_guide.md` | How-to or implementation walkthrough | `npu_lowering_guide.md` |
| Comparison | `comparison.md` | Side-by-side comparison of approaches | `npugraphs/comparison.md` |
| Changelog | `changelog.md` | Chronological log of all ingest operations | `wiki/changelog.md` |

## Naming Conventions

- File names use `snake_case`
- Analysis pages end with `_analysis`
- Guide pages end with `_guide`
- Index pages are always named `index.md`（每目录一个，作为入口）
- One concept per page; prefer splitting over merging

## Ingest Workflow

When a new source is added to `raw/`, follow this sequence:

1. **Read** the source document thoroughly
2. **Discuss** key takeaways with the user before writing
3. **Create** a new wiki page (or update an existing one if the topic is already covered)
4. **Update** the domain `index.md` to include the new page
5. **Cross-reference**: Add `[[wiki links]]` to and from all related existing pages
6. **Append** an entry to `wiki/changelog.md` documenting what was added/updated
7. **Flag contradictions**: If new information contradicts existing wiki content, preserve both claims and add a `> [!contradiction]` callout

## Cross-Reference Rules

- Every page MUST contain a `## Related Pages` section at the bottom with `[[wiki links]]`
- Every new page MUST link to at least one existing page
- Use Obsidian `[[wiki link]]` syntax for internal links
- When updating a page, check if other pages should gain a backlink

## Update Principles

- **Never delete** existing content — only extend or annotate
- Mark outdated claims with `> [!deprecated] Updated by [[page_name]]`
- Mark contradictions with `> [!contradiction] See also [[page_name]]`
- Record the date of each significant update in the page header
- When a page grows too large (>500 lines), propose a split to the user

## Quality Standards

- Write in the same language as the source material (Chinese for Chinese sources, English for English sources)
- Use Mermaid diagrams for architecture, data flow, and sequence visualizations
- Use LaTeX for mathematical formulas
- Include code references with file paths and line numbers when analyzing source code
- Every claim should trace back to a source in `raw/`

## MCP Tools

Two MCP servers are configured in `.mcp.json`:

1. **filesystem** (`@modelcontextprotocol/server-filesystem`) — Read, write, search, and manage files in `wiki/` and `raw/`. Use for all file operations.
2. **qmd** (`qmd mcp`) — Search engine over wiki pages. Currently operating in **BM25 keyword mode only** (no embedding model). Use `search` for keyword matching; `vsearch` and `query` require embeddings (not yet available on this system).

When to use which:
- **filesystem** for: reading/writing files, listing directories, exact grep
- **qmd** for: keyword search across all wiki pages, finding pages that mention a specific term, cross-reference discovery

## Query Workflow

When the user asks a question:

1. **Search wiki first**: Use `qmd search` to find wiki pages matching relevant keywords
2. **Navigate the graph**: Check the relevant `index.md` to understand the domain landscape and follow `[[wiki links]]`
3. **Read the pages**: Use filesystem `read_file` to read the full content of the most relevant pages
4. **Synthesize**: If the answer requires synthesizing multiple pages, do so and note which pages contributed
5. **Check raw sources on gap**: If the answer is NOT in the wiki, do NOT just say "not found". Instead:
   - Scan `raw/` for relevant source documents (check filenames for topic keywords)
   - If a relevant source exists in `raw/`, **automatically ingest it** following the Ingest Workflow, then answer the question from the newly created wiki content
   - If no relevant source exists in `raw/`, say so and offer to create a stub wiki page to track this knowledge gap
6. **Grow the wiki**: Every query that reveals a gap should result in either a new wiki page or a note in the relevant index.md under Knowledge Gaps

## Maintenance Workflow

Periodically (or when the user requests):

1. **Consistency check**: Verify all `[[wiki links]]` point to existing pages
2. **Orphan check**: Find pages not linked from any `index.md` and integrate them
3. **Contradiction review**: Scan for `> [!contradiction]` callouts and propose resolutions
4. **Staleness review**: Flag pages that haven't been updated in >30 days for review
