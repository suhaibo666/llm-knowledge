# Source-type pack — Anything else

For sources that aren't code or a paper: a **spec/RFC/standard/contract**, a **dataset/schema**,
**API/SDK/protocol docs**, a **running system / incident / logs / metrics**, a **business / financial
/ market report**, a **product / competitor teardown** — or something not listed. Read alongside the
SKILL.md core.

## The one rule that generalizes everything
Whatever the artifact, before you analyze it answer three questions — they instantiate the two
principles for *this* source:
1. **What's the atomic locator** that pins a single claim here? (the analogue of `file:line`)
2. **What's the baseline/snapshot** that makes a claim time-true? (version / date / window / build)
3. **What's the essence** worth hunting — the mechanism / the why / the decision, not the surface?

Then run Phase 0–5 exactly as in the core: ingest into a citeable form, map the shape, locate → read
the spot → cite, (fan out if big), integrate + verify.

## Per-kind cheatsheet

| Source kind | Locator | Baseline | Essence to hunt | Ingest / pitfalls |
|---|---|---|---|---|
| **Spec / RFC / standard / contract** | clause / section № (+ doc id) | doc version + date | the *normative* requirements (MUST/SHOULD), the state machine it defines, ambiguities & under-specification, what it forbids | the official doc; cite clause №. Beware "errata"/later versions superseding text. |
| **Dataset / schema / table** | `table.column` / row id / cell | snapshot date / version / row count | the schema & invariants, the distribution, the few columns/rows that carry the argument, data-quality caveats (nulls, dupes, leakage) | load & **profile** before claiming; cite by column/row; a number with no row-count/filter is folklore. |
| **API / SDK / protocol docs** | endpoint + field / method signature | API version | the request/response contract, auth & error model, the real call sequence, rate-limits & gotchas | the reference / OpenAPI; **cross-check against the actual SDK/code** — docs drift, code wins. |
| **Running system / incident / logs / metrics** | host·service·timestamp·trace-id / dashboard panel + window | time window + build/version | the *observed* sequence of events, the causal chain, what the data does vs does **not** show | pull the logs/traces/metrics; separate **observed vs hypothesized** rigorously (this is where folklore breeds). |
| **Business / financial / market report** | page / figure / table № (+ title, date, publisher) | report date + publisher | the thesis/claim, the load-bearing numbers **with their basis/comparison**, the methodology & assumptions, spin vs substance | cite the page; cross-check against primary sources where possible; flag where the number's basis is undisclosed. |
| **Product / competitor teardown** | screenshot + timestamp / version / build | product version + date | how it *actually behaves* (observed), the design choices & tradeoffs, the gaps | drive it / observe directly; separate **observed vs assumed**; a claim from marketing copy ≠ observed behavior. |

## Applying the core contract

Organize by the artifact's natural aspects. For a spec or incident, the design-choice question is
usually “why this rule?” or “why did this fail?”, while the boundary is the under-specified or
unobserved part. Preserve those source-specific answers without redefining the core's section order.

## Type-specific red flags
| If you catch yourself… | Do this instead |
|---|---|
| Stating a log/metric *interpretation* as fact | Mark observed vs hypothesized; cite the exact timestamp/window. |
| Trusting API/spec prose over the shipping behavior | Cross-check against the code/SDK or the live system; the artifact wins. |
| Quoting a report's headline number with no basis | Cite the page and the comparison/assumption behind it; flag undisclosed basis. |
| A dataset claim with no row-count / filter | Profile it; state N and the filter. A number without its denominator argues nothing. |
| Treating marketing copy as observed product behavior | Drive the product; cite a screenshot/version. |
| An aspect written up with no 背景 — straight to "the spec says" | Open with the problem the clause/metric/report exists to settle; a rule with no problem behind it can't be judged. |
| A 发展趋势 paragraph with no anchor | Anchor it to a stated roadmap / errata / newer version or to the beat-4 constraint, and mark it as inference — or omit it. |
