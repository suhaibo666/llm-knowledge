# Source-type pack — Codebase / framework / library

Read this with the core `SKILL.md` whenever current behavior, architecture, or design must be
derived from code. This pack executes **one analysis unit/page or one focused mechanism, not one source file**.
Follow the mechanism across entry points, state owners, helpers, backends, tests, and
history as far as the causal argument requires.

## Routing and baseline

When an approved blueprint exists, inherit the page thesis, owned concepts, exclusions, evidence
entry points, completion test, and **inherit the approved repository commit**. Do not redesign the
page tree from inside the page task.

- **Default anchor:** repository-relative **path + qualified symbol**, such as
  `src/scheduler.py::Scheduler.submit` or `tests/test_scheduler.py::test_cancel_after_submit`.
  A bare path is only for module-level behavior; open the symbol before using it.
- **Line numbers are optional.** Use a tight range for an exact excerpt/diff or unnamed/ambiguous
  code, never by default for ordinary control flow, state transitions, or each claim/hop.
- **Baseline:** commit + branch/tag + commit date. Every current-behavior page states it.
- **Approved page:** verify the inherited checkout and keep it frozen. Do not fetch, pull,
  fast-forward, switch, checkout, reset, or move it. If the commit is unavailable or must change,
  return to `planning-codebase-analysis` with the proposed revision and reason.
- **Focused code analysis with no approved blueprint:** establish a safe baseline. If a sibling
  checkout is clean, on the expected branch, and updating it is in scope, fast-forward it before
  evidence collection and record the old→new commit delta. Never overwrite dirty or diverged work.

Once evidence collection starts, freeze the checkout. A baseline migration requires revalidating
the affected claims and anchors; changing only the header commit is not an update.

## Map the owned mechanism

1. Inspect the directory shape to roughly two levels; exclude generated/vendor/build artifacts.
2. Find construction and selection sites before treating a class as live. Separate current,
   compatibility, fallback, experimental, and removed paths.
3. Locate the orchestrator, state-bearing structures, registries, admission/resource managers,
   protocol boundaries, negative paths, and tests.
4. Build a compact design map: pressure → state owner/invariants → choice vs alternative → actual
   call path → cost/failure boundary → evidence.

The source tree tells you where code lives; it does not determine the explanatory order.

## Architecture overview contract

Build the overview in two passes:

1. **Static responsibility map:** derive layers/components from responsibility, dependency
   direction, state ownership, and contracts—not from directories or a fixed taxonomy.
2. **Dynamic lifecycle:** trace one real request/job/batch through those boundaries and identify
   when state becomes valid, changes owner, completes, and becomes externally visible.

For each real layer answer: why it exists; what capability and input→output contract it provides;
what state, invariants, policy, and failure boundary it owns; what it delegates; and which verified
entry symbols prove the boundary. A lifecycle is motion, not a substitute for the static map.

## Code mechanism contract

Instantiate the core’s canonical beats without repeating them as a second template:

- **What:** locate the mechanism in the system; state its capability, input/output contract, owned
  state, invariants, and explicit non-responsibilities.
- **Why:** name the pressure and the obvious alternative, then support the selection criterion with
  design comments, history, tests, or clearly marked analyst inference.
- **How:** explain the state model first, then prove it with load-bearing structures and the
  smallest real execution trace that exposes state mutation, authority transfer, scheduling, or
  invariant enforcement.
- **Boundary:** cover guards, unsupported combinations, fallback, cost, failure, observability, and
  the tests that protect those conditions.

Code excerpts never replace the model. Do not mirror a function body statement by statement or use
signatures as headings. The prose must survive deleting every code block.

### Execution trace contract

For every load-bearing path, record enough of this ledger that a reader can continue in source:

| Field | Required explanation |
|---|---|
| Hop | Real **caller → callee**, including registry binding, wrapper, RPC, or scheduler hops that change semantics |
| Crossing object | The request, batch, future, queue record, handle, or state token crossing the boundary and its load-bearing fields |
| State transition | Relevant **pre-state → post-state**, mutation, identity, and old→new **state owner** |
| Execution semantics | **local/remote**, **sync/async**, **blocking/non-blocking**, and the first point that actually waits or materializes data |
| Completion | The **completion signal** for submission, acceptance, execution, publication, externally visible state, and downstream consumption—do not conflate them |
| Failure | Guards, propagation, cancellation/settling, retry/idempotency, **partial side effects**, cleanup, and whether **rollback** exists |
| Evidence | Named symbols actually opened; map unique `path::qualified.symbol` anchors to the semantic hops in one compact source-reading route |

Trace only hops that change state/owner, cross an execution boundary, enforce an invariant, or define
completion/visibility. **Collapse pure forwarding helpers**; exhaustive call graphs hide the design.

### Evidence presentation

Keep investigation precision separate from citation density. Open every implementation, guard,
test, and history spot needed; scratch notes may be detailed. In the page, name participating
symbols in prose and add one deduplicated **compact source-reading route** per mechanism: entry /
selection → state owner → state-changing worker → completion/visibility → negative test. Add an
anchor only for a distinct boundary or design decision; never append `file:line` to every sentence
or ledger row. The narrative still explains objects, transitions, synchronization, completion, and
failure; the route makes it findable without becoming a volatile line-number index.

Use the completion ladder as a diagnostic, not a mandatory set of headings:

```text
created → submitted → accepted → running → local-complete → published/visible → consumed/settled
```

Name only the stages the mechanism actually has, and identify which API event proves each stage.

### Conditional depth profiles

Apply a profile **only when the trigger is present**. These are investigation prompts, not sections
that every page must contain.

| Trigger | Required follow-up |
|---|---|
| **Multiple live implementations** or versions | Trace the factory/registry/configuration selection; record applicability, primary path, fallback, and unsupported combinations |
| A **data representation** crosses a boundary or changes form | Track identity, type/shape/schema, copy/share/mutation, padding/chunk/concat/reorder, storage location, and delayed writeback as applicable |
| **Asynchronous, concurrent**, distributed, stateful, or externally effectful work | Separate scheduling from completion; inspect partial success, sibling cancellation/settling, cleanup, retry, idempotency, and rollback |
| Executable or observable behavior | Provide a **runtime verification** route through tests, assertions, logs, metrics, traces, a minimal reproduction, or another observable contract |

If a trigger is absent, say so only when that absence closes a likely misunderstanding; do not pad
the page with empty matrices.

## Evidence roles

Keep evidence roles distinct when they disagree:

| Evidence | What it establishes |
|---|---|
| Current implementation | What the frozen commit actually does |
| Tests | Which behavior and negative boundaries are protected |
| Public/design docs | Intended model and supported contract |
| Comments, issues, commits, PRs | Local or historical rationale; useful but not authoritative alone |
| Analyst inference | A reconstruction that must be labelled and argued from evidence |

For current behavior, implementation plus executable tests normally wins. For public promises, the
published contract remains relevant even when implementation diverges. Describe the mismatch rather
than silently selecting one account.

## Completion check

- The reader can identify the mechanism’s position, capability, state owner, and non-responsibility.
- The design choice is compared with the obvious alternative under a concrete constraint.
- One real execution path is traceable from its live selection/entry point through its business
  completion or visibility boundary.
- Data and completion semantics are deepened only where their conditional triggers apply.
- Guards, costs, unsupported paths, fallback, partial failure, and verification evidence are not
  hidden behind a happy-path call graph.
- Every load-bearing claim and hop is grounded in code actually opened in the frozen checkout; the
  page exposes a deduplicated stable-symbol route for the load-bearing implementation and tests.
