# Function-point review rubric — independent sampling per spec wave

Isomorphic to `references/page-review-rubric.md` of `source-faithful-analysis`, with the target
switched from "causal-argument quality" to **contract completeness and verifiability**. The
mechanical tier (locator existence, links, math) runs as `tools/` gates and is not repeated here.

## Who runs it

- **The reviewer is never the writer.** After each spec wave the coordinator dispatches one
  independent reviewer; the user samples on top of that.
- Input: the wave's spec pages + the tree overview (leaf-row list) + read access to the frozen
  baseline checkout.
- Output: one verdict row per leaf. A rejected leaf returns to spec stage for the original
  writer; the reviewer never rewrites it.

## The six checks

| # | Check | Pass condition | Failure mode it catches |
|---|---|---|---|
| R1 | Locator sampling | Open 3 random locators (prefer behavior-assertion ones) at the frozen baseline; all three say what the spec claims | fabricated or drifted citations |
| R2 | Contract closure | Input→output statable on its own, with no un-split "depends on the sub-case" branches; all seven blocks present (N/A needs a reason) | mis-granular leaves, half-understood specs |
| R3 | Boundaries evidenced | Every boundary constraint points to validation code or a test; "no validation (silently accepts)" also carries the spot that was checked | assumed boundaries |
| R4 | Negatives evidenced | Every "unsupported" claim has a NotImplementedError / guard / skipif / doc anchor | guessed support scope |
| R5 | Logic walk | The processing-logic steps walk in locator order at the frozen baseline with zero reviewer-side grep | broken main paths |
| R6 | Tree consistency | The spec matches the tree-overview leaf row on ID/entry/status; status is truthful (`planned` leaves have no sneaked-in spec, `spec'd` leaves' specs actually exist) | tree↔spec drift |

## Verdict format

```text
| Leaf ID | R1 | R2 | R3 | R4 | R5 | R6 | Verdict | Note |
|---|---|---|---|---|---|---|---|---|
| gates/links/scan | 3/3 | pass | pass | pass | pass | pass | PASS | — |
| gates/math/incremental | 2/3 | pass | FAIL | pass | pass | pass | REJECT | "mutual exclusion" boundary lacks validation evidence |
```

- `REJECT` must name the failing check and the smallest failing unit, so the writer can fix it
  point-blank instead of relitigating the whole page.
- The reviewer records which 3 locators were sampled; later reviews rotate coverage.

## Boundaries

- Judge contract quality only; style, length, and leaf count are out of scope (length is never
  evidence in either direction).
- A **tree-cut problem** found during review (a leaf that should split/merge, a wrong module) is
  not a spec failure — it is tree drift: return it to tree stage with evidence, back through the
  approval gate (semantics aligned with the `planning-codebase-analysis` replanning gate).
- Review findings never authorize the reviewer to edit pages directly; fixes belong to the
  writer, so ownership stays single.
