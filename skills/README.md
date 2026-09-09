# skills/ — Shared Agent Skills

This directory is the repository's **single physical copy** of skill definitions, shared by all supported agents. A second copy under an agent directory drifts, so `tools/test_math_skill.py` fails when one appears.

## Loading Conventions

`CLAUDE.md` is the knowledge base's **constitution**. It defines what the knowledge base *is*: three layers, the functional tree as the sole content authority, provenance policy, and quality gates. Document-writing operations belong in the skills here, **loaded on demand, never by default**.

Choose the skill for the operation:

| Task | Read |
|---|---|
| Add, rewrite, rename, or merge pages in `wiki/`; maintain indexes and changelog | [`maintaining-llm-knowledge`](maintaining-llm-knowledge/SKILL.md) |
| Write or change any LaTeX formula | [`writing-obsidian-math`](writing-obsidian-math/SKILL.md) |
| Add or change a page figure: whether a figure helps, which medium to use, and its visual form | [`drawing-wiki-figures`](drawing-wiki-figures/SKILL.md) |
| Draw or change a Mermaid diagram, including parser pitfalls | [`writing-mermaid-diagrams`](writing-mermaid-diagrams/SKILL.md) |
| Discover a new codebase or an unplanned multi-page codebase domain; plan capabilities, architecture, core mechanisms, and a document blueprint; await approval, then coordinate implementation | [`planning-codebase-analysis`](planning-codebase-analysis/SKILL.md) |
| Write or review an approved/boundary-defined software architecture or concrete feature, write an approved/focused codebase mechanism, or analyze a paper, spec, dataset, incident, report, or other non-code artifact at any scale; load the matching document profile on demand | [`source-faithful-analysis`](source-faithful-analysis/SKILL.md) |
| Inventory a repository as a feature tree with a contract-style specification for every leaf function point (inputs/outputs, processing logic, boundary constraints, supported scope), or reconcile and reverify after a baseline advance | [`feature-tree-analysis`](feature-tree-analysis/SKILL.md) |

For example, organizing a new codebase into a knowledge domain starts with `planning-codebase-analysis`. After blueprint approval, repository architecture pages, concrete feature pages, and focused mechanism pages all use `source-faithful-analysis`, selecting the `software-architecture`, `feature-analysis`, or `mechanism-analysis` profile respectively. Add `maintaining-llm-knowledge` when writing into this wiki, and load math or figure skills when the page needs them. Feature-tree inventories use `feature-tree-analysis` directly, without an analysis-article profile. Non-code artifact analysis at any scale, and one focused mechanism in an otherwise unplanned codebase, do not go through the planner.

## Agent Integration

This directory is the **only physical copy**. The repository's two supported agents reach it through different entry points:

- **Claude Code** uses `.claude/skills`, a symlink to this directory, stored by Git as a symlink object with mode `120000`. This is the repository's native discovery entry point:

  ```
  .claude/skills -> ../skills
  ```

- **Codex** reads the task-to-skill table in `AGENTS.md` and opens the required file in this directory. Keep this repository's explicit loading convention; do not create a mirrored `.codex/skills` directory or duplicate skill definitions.

Only these two integrations are maintained here. `tools/test_math_skill.py` guards two invariants: agent-side skill paths must resolve to this directory, and each skill has only one physical `SKILL.md` in the repository.

## Language Contract

Write skill instructions, reference guidance, evaluation prompts, and evaluation rubrics in **English** throughout `skills/`. This is the language of the reusable instructions, not a requirement for the artifacts they produce. Artifact language follows the user's request or the host's established convention; a Chinese wiki page can be produced from an English skill.

Preserve non-English text when its exact spelling is part of the evidence or interface: quoted source material, parser/regression fixtures, source identifiers, and required output-format literals such as wiki header labels. Explain each such exception in English near its use. These exceptions do not permit untranslated explanatory paragraphs or rubrics. Review language across referenced files and evaluations as well as top-level `SKILL.md` files.

## Editing Skills

Skills are **operational contracts**, not essays:

- Make rules checkable by the tools under `tools/`, or explicitly identify the parts that require manual review.
- When changing a rule, update its checker and tests (`tools/check_math.py`, `tools/check_links.py`, `tools/test_*.py`) where applicable so the instructions and acceptance gates agree.
- Keep rule strength explicit: distinguish mandatory requirements, conditions that activate a requirement, and optional presentation examples. A useful table shape is not automatically a required section for every subject.
- Select explanatory dimensions for the subject: algorithms may need mathematical reasoning, representations, and complexity; stateful or concurrent mechanisms may need ownership, transitions, and completion conditions. Keep the detailed selection method in `source-faithful-analysis` rather than copying it into every skill.
- Do not repeat the same rule in both `CLAUDE.md` and a skill. The constitution defines what the knowledge base is; skills define how to operate it.
