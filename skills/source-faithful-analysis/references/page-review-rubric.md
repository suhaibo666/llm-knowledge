# Page review rubric — independent quality review for analysis pages

Five checks, each aimed at a failure mode this knowledge base has actually shipped. The rubric is
the **substance tier** of quality control; the mechanical tier (baseline/header, links, math, and
legacy line-number locators when a changed page still contains them) runs as `tools/` gates and is
not repeated here.

## Who runs it

- **The reviewer is never the writer.** In a planned wave, the coordinator dispatches one
  independent reviewer per wave; the user samples pages on top of that. Outside waves, the user or
  a separately-invoked reviewer applies it before a page is declared done.
- Input: the finished page, its approved page contract (thesis / owned concepts / exclusions /
  completion test), and read access to the frozen checkout at the page's pinned commit.
- Output: one verdict row per page (see format below). A rejected page returns to Phase 2 in
  `source-fidelity.md` with the failing check named; the reviewer does not rewrite the page.

## The five checks

| # | Check | Pass condition | Failure mode it catches |
|---|---|---|---|
| 1 | **Beat-2 substance** | Every major unit names a **concrete rejected alternative** and the **criterion** that decided against it. "For performance/efficiency/simplicity" with no named alternative fails. Reconstructed rationale is explicitly marked as analyst inference. | Thin pages that restate WHAT without WHY — the original "过于简单" failure |
| 2 | **Hop-walk** | Take the page's main execution trace and follow its named symbols plus compact source-reading route from the entry point to the stated completion/visibility boundary, opening each anchor in the frozen checkout. The page must identify every semantic hop without requiring undocumented discovery; line-number citations are not required. | Broken or hand-waved call chains; "跟着文档读不下去" |
| 3 | **Delete-the-code test** | Mentally (or actually) drop every code block and read only the prose. The mechanism, its state model, and the design argument must still be understandable. If the narrative collapses without the excerpts, the page is transcription, not analysis. | Code-paste padding masquerading as depth |
| 4 | **Anchor spot-check** | Pick 3 load-bearing source anchors (prefer guards, state changes, or tests; include any optional line-specific citation that supports a quote or number) and open them in the frozen checkout. All 3 must say what the page claims they say. | Fabricated, vague, or drifted evidence references |
| 5 | **Algorithm replay / principle figure** | Apply the algorithmic trigger and exemptions exactly as defined by the selected document profile. Record `figure-trigger: none` or the applicable subset of `transform, layout, timing, coupled-planes`. With **no trigger**, no figure is required. When triggered, inspect the **rendered figure** with `drawing-wiki-figures` and its **stranger-reader line**: it must replay the **smallest example** from input through the decisive transformation/layout/timing/state to output and expose the governing invariant, constraint, or cost. A **class diagram**, ownership inventory, **caller tree**, code excerpt, prose, or table alone does not pass. Every distinct live algorithm/data plane must reuse that example and remain traceable through local compute, data/state/ownership movement, communication or synchronization, reconstruction, applicable forward/backward differences, and incremental cost. | Algorithm pages whose prose names operations but never makes the derivation, ownership change, schedule, or variant difference reconstructable |

Checks 1–3 and 5 are judged per major unit/section; one failing unit fails the page for that check.

## Verdict format

One row per page, appended to the wave report:

```
| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 15_megatron_pp_schedulers_analysis | pass | pass | pass | timing | pass | 3/3 | PASS | — |
| 16_megatron_distributed_optimizer_analysis | FAIL §4 | pass | pass | layout | pass | 3/3 | REJECT | §4 无被否方案 |
```

- `REJECT` requires naming the failing check **and the smallest failing unit** so the writer can
  fix without re-litigating the whole page.
- The reviewer records which 3 anchors were spot-checked (so successive reviews rotate coverage).

## Boundaries

- The rubric judges **substance against the page's own contract**, not taste: heading style, length,
  and wording are out of scope (length is never evidence in either direction). The algorithm gate is
  also not a taste-based image quota: the reviewer records the observable trigger, or records
  `figure-trigger: none` and passes an otherwise complete page with no figure.
- A material page-boundary or ownership problem discovered during review is not a rubric failure —
  it is contract drift: send it back to `planning-codebase-analysis` instead of failing the page.
- Review findings never justify the reviewer editing source pages directly; the writer (or the
  coordinator) applies fixes so ownership stays single.
