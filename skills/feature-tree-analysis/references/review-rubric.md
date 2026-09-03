# Function-point review rubric — independent sampling per spec wave

Isomorphic to `references/page-review-rubric.md` of `source-faithful-analysis`, with the target
switched from "causal-argument quality" to **contract completeness and verifiability**. The
mechanical tier (`check_feature_tree.py`, links, math, and legacy line locators when present) runs
as `tools/` gates and is not repeated here.

## Who runs it

- **The reviewer is never the writer.** After each spec wave the coordinator dispatches one
  independent reviewer; the user samples on top of that.
- Input: the wave's spec pages + the ownership manifest + the tree overview (leaf-row list) +
  read access to the frozen baseline checkout.
- Output: one verdict row per leaf, **recorded in the manifest's `reviews:` list** by the
  coordinator (the reviewer never edits the manifest). A leaf becomes `verified` only when a
  PASS row exists for it (checker V1); `--phase delivery` requires this for every leaf. A
  rejected leaf returns to spec stage for the original writer; the reviewer never rewrites it.

## The six checks

| # | Check | Pass condition | Failure mode it catches |
|---|---|---|---|
| R1 | Source-anchor sampling | Open 3 random stable anchors (prefer behavior-assertion ones) at the frozen baseline; all three say what the spec claims | fabricated, vague, or drifted evidence references |
| R2 | Contract closure | Input→output statable on its own, with no un-split "depends on the sub-case" branches; all seven blocks present and each of the **five core fields** filled (N/A needs a reason); no causal or design-rationale prose | mis-granular leaves, half-understood specs, mechanism pages in disguise |
| R3 | Inputs & outputs evidenced | Every input's type, default, and valid domain traces to a signature / config definition / env read; every output (return, product, side effect, error or exit code) traces to a return, write, or raise site; defaults match code, not docs | invented defaults, missing side effects |
| R4 | Boundaries & scope evidenced | Every boundary constraint points to validation code or a test ("no validation" carries the spot checked); conditional boundary dimensions are covered wherever their trigger exists in the code; every **supported and unsupported** scope claim has a guard / skipif / NotImplementedError / build-dependency / doc anchor | assumed boundaries, guessed support scope |
| R5 | Logic walk | The processing-logic steps name their symbols and walk against the compact source-reading route at the frozen baseline without undocumented discovery | broken main paths |
| R6 | Tree & manifest consistency | The spec's ID, entry, and status match the manifest and the overview leaf-row table; the spec page has the leaf-id heading (checker S3); the files the writer reports opening fall inside the leaf's `owns.files` or the drift was reported; status is truthful (`planned` leaves have no sneaked-in spec, `spec'd` leaves' specs exist) | tree↔spec drift, silent ownership drift |

## Verdict format

The reviewer returns one row per leaf; the coordinator appends each as a `reviews:` entry:

```yaml
reviews:
  - {leaf: gates/links/scan, date: 2026-09-02, reviewer: reviewer-a,
     r1: 3/3, r2: pass, r3: pass, r4: pass, r5: pass, r6: pass, verdict: PASS}
  - {leaf: gates/math/incremental, date: 2026-09-02, reviewer: reviewer-a,
     r1: 2/3, r2: pass, r3: pass, r4: "FAIL: 'mutual exclusion' boundary lacks validation evidence",
     r5: pass, r6: pass, verdict: REJECT}
```

The same rows may be rendered as a Markdown table in the wave report; the manifest is the record.

- `REJECT` must name the failing check and the smallest failing unit (inside the `rN` value), so
  the writer can fix it point-blank instead of relitigating the whole page.
- The reviewer records which 3 source anchors were sampled; later reviews rotate coverage.

## Boundaries

- Judge contract quality only; style, length, and leaf count are out of scope (length is never
  evidence in either direction).
- A **tree-cut problem** found during review (a leaf that should split/merge, a wrong module, an
  unowned behavior) is not a spec failure — it is tree drift: return it to tree stage with
  evidence, back through the approval gate (semantics aligned with the
  `planning-codebase-analysis` replanning gate).
- Review findings never authorize the reviewer to edit pages or the manifest directly; fixes
  belong to the writer or the coordinator, so ownership stays single.
