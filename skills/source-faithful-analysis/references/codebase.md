# Source-type pack — Codebase / framework / library

The concrete edge of source-faithful analysis when the source is **code**. Read this alongside the
SKILL.md core. Output can be any language and any doc system.

## Routing gate

This pack executes **one analysis unit/page or one focused mechanism**, not one source file. A mechanism may cross entry points, state owners, helpers, backends, tests, and history; follow all load-bearing files needed to explain it. If the request instead covers a new whole codebase or a multi-page domain and there is no approved blueprint, use `planning-codebase-analysis` before this pack.

When an approved blueprint exists, inherit the page thesis, owned concepts, explicit exclusions, evidence entry points, and completion test. Do not redesign the repository's page tree from inside the page task.

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
- **Map the owned unit, not the whole documentation tree.** Identify the page's entry symbols, state owners, real call path, tests, and external contracts. Return a material-drift finding to the planner if the approved ownership cannot match the source.

## Essence checklist (Principle 2, for code)
Per subsystem, first the **thesis** — the core design idea in 1–2 sentences ("a decoupled two-process
pipeline with a busy loop", "OS-style paging for the KV cache"). Then write the section in the core's
mandatory **five-beat order**, instantiated for code. Beat 2 is the one agents skip on a codebase —
code states *what* it does and almost never *why it isn't the other thing*; that is exactly why you
have to go get it.

| Beat | For a codebase, that means | Where to dig for it |
|---|---|---|
| 1 **背景** | the workload / bottleneck / bug this path exists for, and what the previous or naive implementation did | the commit or PR that introduced it (`git log -S '<symbol>'`, `git log --follow`, `git blame` → read the message), the design comment atop the module, the issue it references, the path it replaced |
| 2 **为什么这么设计** | why *this* structure and not the obvious one — a queue vs a lock, a busy loop vs a callback, paging vs contiguous — and the criterion that decided it (latency? memory? a hardware constraint?) | design comments, the PR discussion, the alternative visible in history. When the code is silent, say so and mark your reconstruction as inference — never present it as the author's stated rationale |
| 3 **实现思路与细节** | the idea in one paragraph, then: **key data structures & state machines** (what state exists, who owns it, how it transitions — usually where the real design lives) and **the actual call chain** traced end to end with `file:line` at each hop, not a vague block diagram | grep the orchestrator / main loop / registry, then follow one real path |
| 4 **约束** | preconditions and invariants, the cost it pays, edge cases, what breaks if you change it, what the code deliberately does *not* do | asserts and guards, `raise NotImplementedError` / "unsupported" branches, fallback paths, the tests (they encode the boundary) |
| 5 **发展趋势** (optional) | the deprecation in flight, the `TODO`/RFC the code points at, the direction the beat-4 constraint pressures | `TODO`/`FIXME`/`DeprecationWarning`, an open RFC or design doc referenced in comments — anchor it and mark as inference, or omit the beat |

## Fidelity loop (Phase 2)
`grep/search to locate → read the targeted range (not the whole file) → cite file:line + state the
mechanism in your own words`. Reading whole large files "to be safe" wastes context and tempts you to
summarize instead of analyze.

## Doc structure (the code variant of the template)
- **## 1. Overview** — 背景/问题 first (what this subsystem exists to solve; what the naive approach
  costs) → positioning + the thesis + an architecture/data-flow diagram (annotate with `file:line`
  where useful) + a key-concepts table.
- **## 2. Quick Start** — the minimal entry point (API/flag/config that triggers this path) +
  "where to start reading the source" (entry function, with `file:line`) + one minimal, traceable
  call chain.
- **## 3. Deep Dive** (the bulk) — one section per component/step, **each in the five-beat order**
  (背景 → 为什么这么设计 → 实现思路与细节 → 约束 → 趋势可选), dense verified `file:line`; data
  structures / state machines / algorithms as tables/code/diagrams; corrections to common
  misconceptions.
- **## Related / Cross-references**.

## Type-specific red flags
| If you catch yourself… | Do this instead |
|---|---|
| Listing function signatures / "takes X returns Y" | Replace with the five beats: 背景 → 为什么 → 实现+细节 → 约束. |
| Opening with "this class does X" — no 背景 | Find the commit/PR that added the path (`git log -S`, `git blame`) and open with the problem it solved. |
| Reconstructing the 为什么 yourself, then stating it as the code's intent | Say the code is silent and mark it as your inference — or find the comment/PR that actually says it. |
| A subsystem written up as a free win | Read the guards, error branches and tests; state its preconditions, cost and failure modes under 约束. |
| Citing a line number from memory or a blog | Re-derive it from the current checkout. |
| Reproducing "how X works" from training memory | Open the source; it has probably changed — flag the contradiction. |
| Documenting a legacy/removed path as current | Check live-vs-legacy; focus the current path, note the old one as contrast. |
| Reading whole large files into context | Grep to locate, read targeted ranges. |
