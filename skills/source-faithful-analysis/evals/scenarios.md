# Live behavior scenarios for source-faithful analysis

These manual evaluations test judgment that static contract tests cannot prove. Evaluators work
read-only and make **no repository writes**. Judge the produced analysis plan and reasoning, not
literal headings or wording.

## Baseline without the feature profile

Give a fresh evaluator the current core `../SKILL.md` and `../references/codebase.md`, but not
`../references/document-types/feature-analysis.md`. Use each request below and record concrete
omissions or unwanted content before changing the skill.

The 2026-09-04 baseline already handled frozen evidence, causal reasoning, state/ownership, exact
execution hops, failure, and source routes well. Its repeatable gap was page shaping: for the
parallel case, the smallest primitive, system-wide module walk, full training closure, companion
mechanism depth, aggregate cost, and exact ASCII caller tree remained optional author judgment.

## S1 — Parallel training feature

Prompt: plan a source-grounded Tensor Parallel feature page that explains the primitive matrix
partition, its placement across the model, the complete training path, companion mechanisms,
implementation relationships, and total cost.

Green result: the plan moves from one real primitive to every load-bearing system module; closes
forward through objective/loss, backward, and gradient handoff; expands only evidenced companion
mechanisms at equal depth; reconciles component costs; and includes an ASCII caller tree consistent
with the stable-symbol route. Because tensor partition is an algorithmic implementation, the plan
also requires a principle figure that replays the smallest matrix example through partition, local
compute, communication, reconstruction, and the different backward transformation.

## S2 — Ordinary software feature

Prompt: plan a source analysis of a synchronous link check or cache-invalidation feature, including
why it exists, how it runs, its costs, and its boundaries.

Green result: the plan uses the same problem → smallest real case → system placement → component
closure → aggregate cost shape and an ASCII tree for a real multi-hop path. It does not invent
training, loss/backward, accelerator memory, collectives, ranks, or companion mechanisms. Cost
dimensions are selected from the implementation actually found.

## S3 — Asynchronous stateful feature

Prompt: plan a source analysis from an async publication API through background completion,
visibility, downstream consumption, and failure.

Green result: the feature profile shapes the page and ASCII caller view, while `codebase.md` remains
the sole owner of async completion, retry/settling/rollback, data-lifecycle, and per-hop evidence
semantics. The plan neither duplicates those contracts nor forces training-specific sections.

## S4 — Algorithmic implementation visual gate

Prompt: review a proposed EP feature page that has a token-to-expert table, prose for AllGather,
AllToAll, and Flex, an ASCII caller tree, a qualitative cost table, and forward/backward prose, but
no principle figure and no replay of the same token example through each distinct data plane. Then
state whether the same rule applies to PP scheduling, TP/CP partitioning, and packed-dataset layout,
and whether it forces a diagram onto a direct synchronous cache invalidation feature.

Green result: the algorithmic pages fail until at least one rendered principle figure replays a
concrete example from input through the decisive transformation to output. Distinct live data planes
reuse the same example and expose local compute, data/ownership movement, reconstruction, backward
differences where applicable, and incremental cost; one clearly separated comparison figure may
cover several variants. A caller tree, class diagram, prose, or table alone does not satisfy this
gate. The direct cache invalidation control has no algorithmic transformation trigger and does not
gain an empty figure requirement.

## S5 — Subject selection with local source fixtures

Use the current router, its selected references, and the two synthetic Python fixtures in
`fixtures/`. These are skill-test inputs, not upstream implementation evidence. Give a fresh
evaluator the following five tasks; request a compact source-grounded outline, one explanatory
paragraph, the source route, conditional-depth/figure selection, and actual completion boundary.
No wiki writes or figure rendering are required for this explanation-selection exercise.

1. Explain `algorithms.py::lower_bound` as a concrete feature, using `[1, 2, 2, 4]` and target `2`.
2. Explain `algorithms.py::quantize_row` and `reconstruct` as a numerical mechanism, using
   `[0, 0.5, 1]`; inspect numerical edge behavior, not just the input guard.
3. Explain `algorithms.py::parse_value` as a concrete feature, including both registered variants
   and unsupported/invalid input.
4. Explain `cache.py::Cache.refresh` as a concrete feature with concurrent refreshes, cancellation
   and readers. Respect the fixture's single-event-loop scope.
5. Consider a proposed composition that quantizes a row inside a cache loader. Separate what the
   two files prove from the hypothetical integration, and identify evidence still needed.

Green result: the first case centers on interval invariants, branch validity, termination and
cost; the second on arithmetic, error and actual numerical limits; the third on dispatch and error
contracts; the fourth on state, ordering, publication and cancellation; the fifth combines lenses
without inventing source integration. Pure function results close their own paths. Ownership and
visibility remain necessary for the actual cache collaboration. Algorithmic examples retain the
principle-figure requirement, while non-algorithmic forwarding does not acquire one by association.

The fixtures used on 2026-09-08 have SHA256 identities:

- `algorithms.py`: `daddc58f12c8b7c1bc8c78b5f29b42a6649f3a837dc4058addb73b9fd1dd5a21`
- `cache.py`: `3802c793374564c7378c8093d7e93f529693a682fbd906a1fb6b75ce02d7f0dd`

## Run log

| date | run | result | observation |
|---|---|---|---|
| 2026-09-04 | S1 baseline | expected gap | Source fidelity passed; six page-shaping decisions were left to evaluator judgment. |
| 2026-09-04 | S2 baseline | control PASS | It avoided parallel-only content, but the feature-page shape and ASCII choice were evaluator judgment rather than a contract. |
| 2026-09-04 | S3 baseline | control PASS | Async semantics were complete under `codebase.md`; page shape and visual separation were evaluator judgment. |
| 2026-09-04 | S1 green | historical PASS | Pre-visual-contract run: primitive→system, training closure, conditional companion depth, total cost, and ASCII tree became explicit; it did not yet test a required principle figure. |
| 2026-09-04 | S2 green | PASS | No parallel/training-only content or empty conditional sections appeared. |
| 2026-09-04 | S3 green | PASS | Page shape improved while async/data trace ownership stayed with `codebase.md`. |
| 2026-09-04 | S4 baseline | expected gap | Three read-only evaluators confirmed that current prose/table coverage can pass without a principle figure or same-example variant replay; the ordinary non-algorithmic control should remain exempt. |
| 2026-09-04 | S4 green | PASS | Fresh evaluators rejected the prose-only EP variants, required principle figures for PP/TP/CP/EP/packed algorithms, and preserved `figure-trigger: none` for the ordinary CRUD control. |
| 2026-09-04 | S1 visual rerun | PASS | A fresh evaluator required TP's matrix-layout figure, local GEMM, reconstruction communication, forward/backward differences, and cost; prose and caller tree alone failed. |
| 2026-09-08 | Subject-selection baseline | scoped control PASS | Before edits, an independent outline probe already avoided invented persistent owners for binary search and pure quantization. It identified the stronger per-major-unit rejected-alternative requirement. This was not a reproduced universal ownership-table failure; the motivating overgeneralization occurred in the preceding user conversation. |
| 2026-09-08 | S5 green | explanation-selection PASS | One fresh evaluator completed all five fixture tasks, opened the hash-matched source, and ran Python 3.13.15 probes. It explained interval correctness, numerical error, parser delegation and cache publication separately, and labelled the proposed composition as unimplemented. It found finite-input quantization underflow/overflow instead of inferring safety from the guard. This was an outline/reasoning exercise, not a rendered-page or repeated statistical evaluation. |
