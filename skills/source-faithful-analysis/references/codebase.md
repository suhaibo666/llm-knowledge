# Source-type pack — Codebase / framework / library

Read this with the shared source-fidelity contract whenever evidence comes from code. This pack
supports **one analysis unit/page or one focused mechanism, not one source file**. Follow evidence
across entries, selection, state owners, helpers, backends, tests, and history as far as the selected
document profile requires.

## Code locator and checkout evidence

The shared kernel owns baseline and boundary policy. This pack adds only the code/Git-specific
locator format and safe checkout handling.

- **Default anchor:** repository-relative **path + qualified symbol**, such as
  `src/scheduler.py::Scheduler.submit` or `tests/test_scheduler.py::test_cancel_after_submit`.
  A bare path is only for module-level behavior; open the symbol before using it.
- **Line numbers are optional.** Use a tight range for an exact excerpt/diff or unnamed/ambiguous
  code, never by default for ordinary control flow, state transitions, or each claim/hop.
- **Code snapshot identity:** commit + branch/tag + commit date.
- **Frozen-checkout safeguard:** after the owning workflow selects the commit, verify it and do not
  fetch, pull, fast-forward, switch, checkout, reset, or move the checkout during evidence
  collection. If unavailable, report the mismatch to the owning workflow; do not pick a substitute.
- **Focused code analysis with no approved blueprint:** establish a safe baseline. If a sibling
  checkout is clean, on the expected branch, and updating it is in scope, fast-forward it before
  evidence collection and record the old→new commit delta. Never overwrite dirty or diverged work.

Any approved baseline migration is implemented by revalidating affected claims and anchors;
changing only the header commit is not code evidence.

## Build the code evidence map

1. Inspect the directory shape to roughly two levels; exclude generated/vendor/build artifacts.
2. Find construction and selection sites before treating a class as live. Separate current,
   compatibility, fallback, experimental, and removed paths.
3. Locate the orchestrator, state-bearing structures, registries, admission/resource managers,
   protocol boundaries, negative paths, and tests.
4. Build a compact evidence map: live entry/selection → state owner/invariants → actual call path →
   completion/visibility → guards/tests/history.

The source tree tells you where evidence lives; it does not determine the explanatory order or
software-layer classification.

## Execution trace evidence contract

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
completion/visibility. **Collapse pure forwarding helpers** only in non-direct summaries; an exact
caller tree retains them when omission creates a false edge, or marks the edge `transitive/elided`.

### Evidence presentation

Keep investigation precision separate from citation density. Open every implementation, guard,
test, and history spot needed; scratch notes may be detailed. In the page, name participating
symbols in prose and add one deduplicated **compact source-reading route** per mechanism: entry /
selection → state owner → state-changing worker → completion/visibility → negative test. Add an
anchor only for a distinct boundary or design decision; never append `file:line` to every sentence
or ledger row.

Use the completion ladder as a diagnostic, not a mandatory set of headings:

```text
created → submitted → accepted → running → local-complete → published/visible → consumed/settled
```

Name only the stages the code actually has, and identify which API event proves each stage.

## Conditional evidence depth

Apply a profile **only when the trigger is present**. These are investigation prompts, not sections
that every page must contain.

| Trigger | Required follow-up |
|---|---|
| **Multiple live implementations** or versions | Trace the factory/registry/configuration selection; record applicability, primary path, fallback, and unsupported combinations |
| A **data representation** crosses a boundary or changes form | Track identity, type/shape/schema, copy/share/mutation, padding/chunk/concat/reorder, storage location, and delayed writeback as applicable |
| **Asynchronous, concurrent**, distributed, stateful, or externally effectful work | Separate scheduling from completion; inspect partial success, sibling cancellation/settling, cleanup, retry, idempotency, and rollback |
| Executable or observable behavior | Provide a **runtime verification** route through tests, assertions, logs, metrics, traces, a minimal reproduction, or another observable contract |

Deepen the evidence **only when the trigger is present**. If absent, omit the profile unless that
absence closes a likely misunderstanding; do not add empty matrices.

## Evidence completion check

- Live construction/selection and current versus legacy paths are distinguished.
- One real path is traceable through the selected document profile's completion boundary.
- Conditional data, concurrency, failure, and runtime evidence is collected only where triggered.
- Every load-bearing claim and hop is grounded in code actually opened at the frozen baseline; the
  page exposes a deduplicated stable-symbol route for the implementation and tests.
