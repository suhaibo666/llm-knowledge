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
| Add or change a figure on a page — medium choice, house visual spec | [`drawing-wiki-figures`](skills/drawing-wiki-figures/SKILL.md) |
| Draw or change a Mermaid diagram (parser traps) | [`writing-mermaid-diagrams`](skills/writing-mermaid-diagrams/SKILL.md) |
| Discover and plan a new whole codebase or unplanned multi-page codebase domain, propose the document blueprint, wait for approval, then coordinate implementation | [`planning-codebase-analysis`](skills/planning-codebase-analysis/SKILL.md) |
| Write an approved/focused codebase unit, or analyze a paper, spec, dataset, incident, report, or other non-code artifact at any scale | [`source-faithful-analysis`](skills/source-faithful-analysis/SKILL.md) |
| Inventory a repository as a feature tree down to leaf function points, each with a contract-style spec (I/O, processing logic, boundary constraints, supported scope); reconcile and re-verify it after baseline bumps | [`feature-tree-analysis`](skills/feature-tree-analysis/SKILL.md) |

`skills/` is the single physical copy. `.claude/skills` is a symlink to it (git stores it as a
symlink object), which is how Claude Code discovers the skills natively.

There is no equivalent symlink for Codex, deliberately: Codex has no project-level skill
discovery — its skills come from `~/.codex/skills/` and the plugin marketplaces (`codex plugin`).
For Codex the load path is this file: read the table above and open the skill you need.

Only Claude and Codex are supported here. The duplicated `.agents/skills/` tree and the
`opencode.json` config were removed.
