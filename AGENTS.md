# AGENTS.md — LLM Knowledge Wiki

This is a **markdown knowledge wiki**, not a code project. There are no build, test, or lint commands.

## Architecture

Three layers:
- **`raw/`** — Immutable source PDFs/papers. **NEVER modify.**
- **`wiki/`** — Agent-owned markdown pages. Create, update, cross-reference here.
- **`CLAUDE.md`** — Schema & maintenance rules. **Read before any wiki operation.**

Domains: `llm/`, `megatron-lm/`, `torch_compile/` (each has an `overview.md`).

## Critical Conventions

- **`[[wiki link]]` syntax** — Obsidian-style, no `.md` extension, paths relative to `wiki/` (e.g. `[[llm/muon_analysis]]`)
- **Every page MUST end with `## Related Pages`** containing `[[wiki links]]` to related pages
- **Naming**: `snake_case`, analysis pages end with `_analysis`, guides with `_guide`, overviews are `overview.md`
- **Never delete** existing wiki content — extend or annotate only. Use `> [!deprecated]` and `> [!contradiction]` callouts
- **Language**: match the source document's language (Chinese sources → Chinese pages)
- **Split pages** at >500 lines

## MCP Tools

Configured in `opencode.json` (and `.mcp.json` for Claude):

| Tool | Use for |
|------|---------|
| `filesystem` | All file read/write/list/search in `wiki/` and `raw/` |
| `qmd` | Keyword search across wiki pages (BM25 only — `vsearch`/`query` require embeddings, not available) |

## Ingest Workflow (new source in `raw/`)

1. Read source → discuss key takeaways with user → create/update wiki page
2. Update the domain `overview.md` with the new page
3. Add `[[wiki links]]` bidirectionally to all related pages
4. Append entry to `wiki/changelog.md`
5. Flag contradictions with `> [!contradiction]` callouts

## Query Workflow (user asks a question)

1. Search wiki with `qmd search` → check relevant `overview.md` → read pages
2. **If not in wiki**: scan `raw/` for matching sources → auto-ingest if found → answer
3. **If no source exists**: offer to create a stub page to track the gap

## Environment

- **OS**: Windows (PowerShell). Use `workdir` param for directory changes, not `cd &&`.
- **Hook**: `.claude/hooks/check-raw.ps1` runs on session start.
- **`.obsidian/`**: Obsidian config — do not modify.
- **`tmp_deepseek/`**: Temporary scratch — safe to ignore.

## What to Read First

- `CLAUDE.md` — full schema, page types, quality standards
- `wiki/<domain>/overview.md` — knowledge map for the relevant domain
- `wiki/changelog.md` — recent activity and what's been covered
