# AGENTS.md — LLM Knowledge Wiki

Before doing ANY work in this directory, read [`CLAUDE.md`](CLAUDE.md). Despite the name it is
agent-neutral: it is the knowledge base's constitution — the three layers, the 功能树 as the only
content authority, the provenance policy, and the quality gates.

`CLAUDE.md` deliberately does **not** describe how to write documents. Those operations live in
[`skills/`](skills/README.md) as one shared copy that every agent reads, **loaded on demand**:

| Task | Skill |
|---|---|
| Add / edit / rename / merge a page in `wiki/`, maintain index and changelog | [`maintaining-llm-knowledge`](skills/maintaining-llm-knowledge/SKILL.md) |
| Write or change a LaTeX formula | [`writing-obsidian-math`](skills/writing-obsidian-math/SKILL.md) |
| Draw or change a Mermaid diagram | [`writing-mermaid-diagrams`](skills/writing-mermaid-diagrams/SKILL.md) |
| Turn a new source into wiki pages | [`source-faithful-analysis`](skills/source-faithful-analysis/SKILL.md) |

Only Claude and Codex are supported here. The duplicated `.agents/skills/` tree and the
`opencode.json` config were removed; `skills/` is now the single source of truth.
