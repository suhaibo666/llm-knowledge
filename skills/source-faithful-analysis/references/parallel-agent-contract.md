# Parallel writer-agent contract

Read this only when delegation is authorized and several analysis units can be written independently.
The coordinator owns page boundaries, the shared baseline, cross-page coverage, and final integration;
each writer owns one file. This contract is deliberately self-contained at the handoff boundary, but
it does not redefine the core method.

## Contract template

Fill the placeholders and send the whole block to each writer:

```text
You are writing ONE source-faithful analysis unit. Treat the source as read-only ground truth.

SOURCE
- Frozen baseline: <project @ commit | document id+version | snapshot/window>
- Source location: <repo root | page-markered text | dataset/log/API location>
- Required pack: <codebase | paper | general>
- Evidence slice and entry points: <files/symbols | pages/tables | clauses/fields/windows>

OWNED OUTPUT
- Write only: <absolute output path>
- Thesis/question: <one main pressure and design choice>
- Owned concepts: <list>
- Explicit exclusions and sibling owners: <list>
- Completion test: <what a reader must be able to explain or trace>

ANALYSIS CONTRACT
1. Open every cited spot before using it. Every non-trivial claim gets a verified exact locator.
2. Explain the problem and constraints, why this design beat the obvious alternative, the mechanism
   and evidence, and the costs/failure boundary. Mark analyst inference and source conflicts.
3. Organize by design question and state/model, not by source-file, API, or function order.
4. Reproduce only load-bearing evidence. Quotes, code, tables, and figures do not replace the causal
   explanation.

TYPE-SPECIFIC PROOF
- Code: trace the live selection/entry point, state owners and a minimal real execution path. For
  each semantic hop identify the crossing object, state transition, execution semantics, and
  completion or visibility boundary; include guards, tests and partial-failure behavior.
- Paper: pair each contribution with its problem passage, rejected alternative, mechanism, and
  result/ablation table including the baseline column; state the evaluated regime and limitations.
- General: define the source's atomic locator and snapshot, then ground each claim in the relevant
  clause, row/cell, endpoint/field, event, screenshot, or report table.

HOST INTEGRATION
- Follow the coordinator's page names and links; do not redesign ownership or directories.
- Use the active house figure skill and any supplied figure specification. Do not invent a rendering
  pipeline or copy figure-format rules into this page.
- If source evidence requires a material boundary or baseline change, stop that part and report it.

RETURN TO COORDINATOR
After writing the file, return:
1. title and 3–5 sentence thesis;
2. 6–10 load-bearing locators you personally opened;
3. source conflicts, inference, unsupported paths, and material drift;
4. suggested owner-aware cross-links;
5. uncovered findings another page or gap should own;
6. any figure requirement, described semantically rather than as rendering instructions.
```

## Coordinator checks

- Freeze names, ownership, exclusions, and baseline before dispatch.
- Calibrate one real page first when style or figures are expensive to redo.
- Spot-check 2–3 returned locators and one negative/failure boundary per page.
- Reject pages that open with class/function order, flatten inference into fact, or confuse submission
  with business completion.
- Reconcile coverage and conflicts, update the host spine, and run the host quality gates centrally.
