# Live behavior scenarios for concrete feature analysis

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
