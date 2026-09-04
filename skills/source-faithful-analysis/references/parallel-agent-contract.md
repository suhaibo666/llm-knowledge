# Parallel writer-agent contract

Read this only when delegation is authorized and several approved analysis units can be written
independently. The coordinator owns page boundaries, shared baseline, profile selection, cross-page
coverage, and final integration; each writer owns one file. This handoff applies the routed method
without redefining it.

## Contract template

Fill the placeholders and send the whole block to each writer:

```text
You are writing ONE source-faithful analysis unit. Treat the source as read-only ground truth.

SOURCE
- Frozen baseline: <project @ commit | document id+version | snapshot/window>
- Source location: <repo root | page-markered text | dataset/log/API location>
- Required source pack: <codebase.md | paper.md | general.md | mixed list>
- Required document profile: <mechanism-analysis.md | feature-analysis.md | software-architecture.md>
- Evidence slice and entry points: <files/symbols | pages/tables | clauses/fields/windows>

OWNED OUTPUT
- Write only: <absolute output path>
- Thesis/question: <one main reader question and answer>
- Owned concepts: <list>
- Explicit exclusions and sibling owners: <list>
- Completion test: <what a reader must be able to explain or trace>

METHOD
1. Read the shared source-fidelity contract, required source pack(s), and selected document profile
   completely. The selected document profile determines explanatory order and required views.
2. Open the supporting source before using it; ground every non-trivial claim in evidence read at
   the frozen baseline. Mark analyst inference and visible conflicts.
3. For code, consolidate stable `path::qualified.symbol` anchors into a compact source-reading route;
   do not attach `file:line` to every claim or semantic hop.
4. Reproduce only load-bearing evidence. Quotes, code, tables, trees, and figures do not replace the
   explanation required by the profile.

SOURCE-SPECIFIC PROOF
- Code: trace live selection/entry, state owners, and the real path to completion or visibility;
  record crossing objects, state transitions, execution semantics, guards, tests, and partial
  failure where the codebase pack triggers them.
- Paper: pair claims with exact sections/tables/equations and baselines; state the evaluated regime,
  limitations, and implementation status.
- General: define the atomic locator and snapshot, then ground claims in the relevant clause,
  row/cell, endpoint/field, event, screenshot, or report table.

HOST INTEGRATION
- Follow the coordinator's page names and links; do not redesign ownership or directories.
- Use the active house figure skill and supplied visual specification. Do not invent a rendering
  pipeline or copy figure-format rules into the page.
- If evidence requires a material boundary or baseline change, stop that part and report it.

RETURN TO COORDINATOR
1. title and a 3–5 sentence thesis;
2. 6–10 deduplicated load-bearing source anchors personally opened;
3. source conflicts, inference, unsupported paths, and material drift;
4. suggested owner-aware cross-links;
5. uncovered findings another page or gap should own;
6. any figure requirement, described semantically rather than as rendering instructions.
```

## Coordinator checks

- Freeze names, ownership, exclusions, baseline, and selected profile before dispatch.
- Calibrate one real page first when style or figures are expensive to redo.
- Spot-check returned anchors and one negative/failure boundary per page.
- Reject work that follows source-file order, flattens inference into fact, or stops before the
  selected profile's completion or visibility boundary.
- Reconcile coverage and conflicts, apply the routed independent review, and run host gates centrally.
