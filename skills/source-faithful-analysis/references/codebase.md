# Source-type pack — Codebase / framework / library

The concrete edge of source-faithful analysis when the source is **code**. Read this alongside the
SKILL.md core. Output can be any language and any doc system.

## Routing gate

This pack executes **one analysis unit/page or one focused mechanism**, not one source file. A mechanism may cross entry points, state owners, helpers, backends, tests, and history; follow all load-bearing files needed to explain it. If the request instead covers a new whole codebase or a multi-page domain and there is no approved blueprint, use `planning-codebase-analysis` before this pack.

When an approved blueprint exists, inherit the page thesis, owned concepts, explicit exclusions,
evidence entry points, and completion test; also inherit the approved repository commit. Do not
redesign the repository's page tree from inside the page task.

## Locator & baseline
- **Locator = `path/to/file.ext:NN`**, path relative to the repo root. You **open the file and
  confirm the line before you cite it** — never invent, guess, or copy a line from memory/a blog. A
  fabricated `file:line` is worse than none: it looks authoritative and is wrong.
- **Baseline = HEAD commit + branch + date.** Code is only true *at a commit*; put it in every
  header.
- **Approved codebase page:** verify that the checkout is at the inherited exact commit and keep
  that commit frozen throughout page execution. Do not fetch, pull, fast-forward, switch, checkout,
  reset, or move it. If the approved commit is unavailable or a baseline change is required, stop
  and return to `planning-codebase-analysis` with the concrete revision and reason for approval.
- **Focused code analysis with no approved blueprint:** establish and record a safe baseline. If a
  sibling checkout is clean, on the expected branch, and updating it is in scope, fast-forward it
  first and note the delta. Never reset or overwrite unrelated or dirty state to establish it.

## Ingest & map (Phase 0–1)
- **Directory skeleton first** (dirs only, ~2 levels) — a recursive file dump drowns you in
  generated files, configs, and tests. See the shape, then drill.
- **Live vs. legacy.** Most codebases carry a deprecated path beside the current one (an old engine,
  a v1/v2 split). Find the real target; mention legacy only for contrast. Reproducing a removed
  path is the classic stale-memory failure — *the code wins*, say so.
- **Load-bearing entry files per subsystem** — grep for the orchestrator / main loop / registry /
  scheduler, not every leaf. Find where each subsystem *starts*.
- **Map the owned unit, not the whole documentation tree.** Identify the page's entry symbols, state owners, real call path, tests, and external contracts. Return a material-drift finding to the planner if the approved ownership cannot match the source.

## Architecture overview contract

Build the overview in two passes:

1. **Static responsibility map.** Identify the source's actual layers from stable responsibility,
   dependency direction, and state ownership. Useful discovery prompts are: public/task entry and
   configuration, lifecycle/orchestration, execution policy/runtime, domain/model composition, and
   primitive/backend. These are prompts, **not a mandatory five-layer taxonomy** — merge, split, or
   rename them to match the source.
2. **Dynamic lifecycle.** Trace one real unit of work end to end. A useful starting hypothesis is
   `config/request → topology/resources → local executable state → schedule/execute →
   commit/visibility`, but replace it with the system's real stages and name where state becomes
   valid or changes owner.

For every actual layer, record and explain:

| Question | Required answer |
|---|---|
| Why does it exist? | The problem or variability it absorbs, and the obvious alternative it avoids |
| What does it provide? | Core capabilities, not a list of classes or files |
| What crosses its boundary? | Inputs → outputs and the contract imposed on the next layer |
| What does it own? | State, invariants, policy decisions, and failure boundary |
| What does it not own? | Responsibilities deliberately delegated upward/downward |
| Where is the evidence? | Verified entry symbols and the few load-bearing `file:line` locators |

Static structure comes before dynamic motion. A directory tree relabelled as layers is not an
architecture, and a call chain alone is not an architecture overview. Clear layer ownership should
make likely extension points inferable; add a separate modification catalogue only when the user
explicitly asks for a developer extension guide.

**Overview test:** before entering the deep dive, the reader can answer what the system is made of,
why those boundaries exist, what each layer owns, and how one unit of work crosses them — without
reading pasted implementation code.

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

### Explain the code; do not transport it

Use this positive paragraph contract for every load-bearing mechanism:

1. **Problem and design choice:** what pressure created the mechanism, what route it chose, and why
   that route beats the obvious alternative. If the rationale is reconstructed, label it inference.
2. **Causal mechanism:** describe the participating components, control/data flow, and the sequence
   of state changes that makes the result happen.
3. **Invariant and boundary:** state what must remain true, what the mechanism deliberately does not
   handle, and where it fails or pays a cost.
4. **Evidence:** cite the minimal load-bearing `file:line` range; quote code only when exact syntax,
   ordering, or a guard is itself part of the argument.

Code excerpts never replace steps 1–3. Do not mirror a function body statement by statement or use
signatures as the section outline. For cross-file behavior, a state table, sequence diagram, or
short pseudocode often explains the mechanism better than several pasted functions. **Deletion
test:** remove every code block; the prose must still let the reader explain why the design exists,
how state moves, and what would break if it changed.

## Doc structure (the code variant of the template)
- **## 1. Overview** — 背景/问题 first (what this subsystem exists to solve; what the naive approach
  costs) → positioning + the thesis → a static responsibility/layer map → a separate end-to-end
  lifecycle when combining it would blur the layers → a table of `layer | capability | input→output
  | owned state/boundary | evidence`.
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
| Pasting a function and paraphrasing it line by line | State the design choice and causal state transitions first; retain only the lines that prove a load-bearing claim. Apply the deletion test. |
| Using copied code as the section outline | Organize by mechanism and responsibility; source order is evidence order, not explanation order. |
| Relabelling directories as architecture layers | Derive layers from responsibility, contracts, dependency direction, and state ownership. |
| Treating the call chain/lifecycle as the whole architecture | Add the static layer view first; use the lifecycle only to show how work crosses those boundaries. |
| Adding "where to modify" to compensate for vague layers | Clarify each layer's capability, owned state, and boundary; add a modification catalogue only for an explicitly requested extension guide. |
| Opening with "this class does X" — no 背景 | Find the commit/PR that added the path (`git log -S`, `git blame`) and open with the problem it solved. |
| Reconstructing the 为什么 yourself, then stating it as the code's intent | Say the code is silent and mark it as your inference — or find the comment/PR that actually says it. |
| A subsystem written up as a free win | Read the guards, error branches and tests; state its preconditions, cost and failure modes under 约束. |
| Citing a line number from memory or a blog | Re-derive it from the current checkout. |
| Reproducing "how X works" from training memory | Open the source; it has probably changed — flag the contradiction. |
| Documenting a legacy/removed path as current | Check live-vs-legacy; focus the current path, note the old one as contrast. |
| Reading whole large files into context | Grep to locate, read targeted ranges. |
