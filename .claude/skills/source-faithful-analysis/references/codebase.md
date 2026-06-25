# Source-type pack — Codebase / framework / library

The concrete edge of source-faithful analysis when the source is **code**. Read this alongside the
SKILL.md core. Output can be any language and any doc system.

## Locator & baseline
- **Locator = `path/to/file.ext:NN`**, path relative to the repo root. You **open the file and
  confirm the line before you cite it** — never invent, guess, or copy a line from memory/a blog. A
  fabricated `file:line` is worse than none: it looks authoritative and is wrong.
- **Baseline = HEAD commit + branch + date.** Code is only true *at a commit*; put it in every
  header. If the repo lives in a sibling checkout, fast-forward it first and note the delta.

## Ingest & map (Phase 0–1)
- **Directory skeleton first** (dirs only, ~2 levels) — a recursive file dump drowns you in
  generated files, configs, and tests. See the shape, then drill.
- **Live vs. legacy.** Most codebases carry a deprecated path beside the current one (an old engine,
  a v1/v2 split). Find the real target; mention legacy only for contrast. Reproducing a removed
  path is the classic stale-memory failure — *the code wins*, say so.
- **Load-bearing entry files per subsystem** — grep for the orchestrator / main loop / registry /
  scheduler, not every leaf. Find where each subsystem *starts*.
- **Decompose by subsystem × depth**: Overview → Quick Start → Deep Dive, one concept per doc.

## Essence checklist (Principle 2, for code)
Hunt, per subsystem, for:
- **The thesis** — the core design idea in 1–2 sentences ("a decoupled two-process pipeline with a
  busy loop", "OS-style paging for the KV cache").
- **Key data structures & state machines** — what state exists, who owns it, how it transitions.
  This is usually where the real design lives.
- **The actual call chain** — trace one real path end to end (entry point + the hops), with
  `file:line` at each hop — not a vague block diagram.
- **Tradeoffs / edge cases / invariants** — why this and not the obvious design, what breaks if you
  change it, what the code deliberately does *not* do.

## Fidelity loop (Phase 2)
`grep/search to locate → read the targeted range (not the whole file) → cite file:line + state the
mechanism in your own words`. Reading whole large files "to be safe" wastes context and tempts you to
summarize instead of analyze.

## Doc structure (the code variant of the template)
- **## 1. Overview** — positioning + an architecture/data-flow diagram (annotate with `file:line`
  where useful) + a key-concepts table.
- **## 2. Quick Start** — the minimal entry point (API/flag/config that triggers this path) +
  "where to start reading the source" (entry function, with `file:line`) + one minimal, traceable
  call chain.
- **## 3. Deep Dive** (the bulk) — mechanism-level per component/step, dense verified `file:line`;
  data structures / state machines / algorithms as tables/code/diagrams; tradeoffs, edge cases,
  invariants; corrections to common misconceptions.
- **## Related / Cross-references**.

## Type-specific red flags
| If you catch yourself… | Do this instead |
|---|---|
| Listing function signatures / "takes X returns Y" | Replace with the mechanism + the *why*. |
| Citing a line number from memory or a blog | Re-derive it from the current checkout. |
| Reproducing "how X works" from training memory | Open the source; it has probably changed — flag the contradiction. |
| Documenting a legacy/removed path as current | Check live-vs-legacy; focus the current path, note the old one as contrast. |
| Reading whole large files into context | Grep to locate, read targeted ranges. |
