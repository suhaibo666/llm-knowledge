# PyTorch Graph Learning Series — Consolidated Correction Report

## Authority and interpretation

- Current implementation baseline:
  `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52` (2026-07-23).
- Source root used for verification: `E:/97-codes/torch_parallel/p`.
- Legacy wiki pages are discovery material, not authority.
- `corrected` means the old passage cannot be migrated verbatim. It may contain a useful
  motivation or partial truth, but the new page must use the correction report and pinned source.
- `unresolved` means quarantined: the claim or example is not allowed into an authoritative page
  until separately verified.

Detailed evidence remains in:

- `corrections_foundations.md` (`F-*`);
- `corrections_aotautograd.md` (`A-*`);
- `corrections_passes.md` (`P-*`);
- `corrections_inductor.md` (`I-*`).

No correction is replaced by this summary; the identifiers below are the provenance join keys.

## Severity totals

| Severity | Meaning | Findings |
|---|---|---:|
| P0 | semantic/correctness error that can teach a wrong mechanism or produce invalid code | 49 |
| P1 | stale architecture, current behavior, API or removed/renamed path | 18 |
| P2 | locator/path drift with the core mechanism otherwise retained | 6 |
| P3 | missing boundary, terminology or qualification that must be added | 21 |
| **Total** |  | **94** |

## P0 — Semantic or correctness errors

### Foundations

`F-001`, `F-004`, `F-005`, `F-006`, `F-007`, `F-013`, `F-014`, `F-016`,
`F-017`, `F-019`

Highest-impact corrections:

- a concrete observed shape is a specialization guarded by predicates, not literally a complete
  cache-key tuple;
- equality refinement does not erase runtime guard obligations;
- `assert_size_stride` is not the universal implementation of all symbolic constraints;
- unbacked symbols can still have constraints;
- `__torch_function__` and `__torch_dispatch__` are different mechanisms;
- `PassResult.modified` produces a fixed point only when the driver is configured to repeat.

Destinations: documents 02–04, 07, 15 and 16.

### AOTAutograd

`A-001`, `A-002`, `A-004`, `A-005`, `A-006`, `A-008`, `A-009`, `A-010`,
`A-011`, `A-012`, `A-014`, `A-017`

Highest-impact corrections:

- metadata collection, joint capture and compiler invocation are separate stages;
- fw and bw are separate graphs with no cross-graph FX edges;
- saved values are a runtime ABI, not a hidden Node connection;
- recompute enters bw by ordinary graph-node copying plus reorder;
- activation memory budget is a partition cost control, not “save this percentage of model
  activations”;
- decomposition, constant folding, DCE and fusion are not a universal four-step AOT pipeline;
- direct `aot_function` caching is not Dynamo's guarded compile-cache model.

Destinations: documents 08–11 and 16.

### Pattern matching, DCE and pass ordering

`P-003`, `P-005`, `P-006`, `P-010`, `P-011`, `P-012`, `P-015`, `P-019`,
`P-020`

Highest-impact corrections:

- `Arg`/`KeywordArg` define handler capture ABI, not the candidate call's Python spelling;
- reverse candidate order does not create a reverse graph and is not the reason new nodes are not
  revisited;
- one `PatternMatcherPass.apply()` is not a fixed-point engine;
- pattern replacement does not run one universal DCE/sort/lint/recompile bundle;
- topological correctness does not prove effect/mutation order correctness;
- decomposition is capture configuration, not merely a post-grad rewrite;
- not all Inductor rewrites use PatternMatcher.

Destinations: documents 13–16.

### Inductor

`I-001`, `I-002`, `I-006`, `I-009`, `I-010`, `I-011`, `I-012`, `I-015`,
`I-016`, `I-018`, `I-019`, `I-020`, `I-021`, `I-022`, `I-025`, `I-027`,
`I-028`, `I-029`

Highest-impact corrections:

- FX→IR is interpretation plus lazy realization, not one FX node → one IR node;
- unsupported lowering does not guarantee fallback success;
- views are conditionally zero-copy, not universally free;
- Scheduler DCE and dependencies operate on realized operations/buffers, not FX users;
- fusion does not guarantee a register/shared-memory placement and reductions/extern kernels are
  not categorical barriers;
- peak-memory reorder, wrapper reuse and pooled inference planning are three mechanisms;
- the default path still has a generated Python wrapper;
- algorithm choice and Triton launch-config autotuning are two layers;
- the historical generic fusion cost formula and custom-fusion tutorial are not current
  implementations;
- loop IR is central but not the only Inductor IR form.

Destinations: documents 17–21.

## P1 — Stale architecture, behavior or API

### Foundations

`F-002`, `F-003`, `F-010`, `F-012`

- Refresh current dynamic/export contracts, ShapeEnv state and XBLOCK selection.
- Include current nested-`GraphModule` DCE behavior.

### AOTAutograd

`A-003`, `A-018`, `A-020`

- Replace schema excerpts with the current runtime ABI.
- Re-derive saved-tensor-hook handling and debug-backend behavior.

### Pattern matching and passes

`P-013`, `P-017`

- Use current impurity detection and the materially different pre-grad/joint/post-grad drivers.

### Inductor

`I-004`, `I-007`, `I-008`, `I-013`, `I-014`, `I-017`, `I-023`, `I-024`,
`I-026`

- Include lowering-pattern passthrough, template/multi-output/choice IR and Scheduler foreach
  grouping.
- Refresh fusion rounds, candidate generation, legality/profitability and current config names.
- Use the scheduling-plus-wrapper backend contract and current combined CUDA/XPU schedulers.
- Do not publish fixed universal Triton candidate lists.

## P2 — Source locator drift

`F-011`, `F-015`, `F-020`, `A-019`, `P-001`, `I-031`

These findings retain useful mechanisms but all paths/line references must be replaced with
pinned-SHA locators. Old line numbers are historical evidence only.

## P3 — Missing boundaries or qualifications

### Foundations

`F-008`, `F-009`, `F-018`

Add Export runtime contracts, separate lifted state from functionalization, and state the fake
implementation contract for custom operators.

### AOTAutograd

`A-007`, `A-013`, `A-015`, `A-016`

Enumerate tensor/non-tensor saved-value classes, qualify view replay, scope subclass/alias support
per path and separate structural complexity from traced computation.

### Pattern matching and passes

`P-002`, `P-004`, `P-007`, `P-008`, `P-009`, `P-014`, `P-016`, `P-018`,
`P-021`, `P-022`

Explain why the pattern subclass family exists, how candidate indexing changes complexity, where
safety checks occur, who owns graph mutation, how retracing works, how recursion is scoped and why
legacy pseudocode is not executable specification.

### Inductor

`I-003`, `I-005`, `I-030`, `I-032`

Separate lowering normalization from global optimization, describe fallback as an external
scheduled operation and keep fusion central without reducing all backend optimization to fusion.

## Quarantined legacy material

The following categories stay `unresolved` and may not be copied:

- internal-API quickstarts or custom-pass examples not run on the pinned commit;
- pass catalogs whose individual registrations have not been re-derived;
- the legacy hardware-backend and custom Triton fusion tutorial;
- mixed-stage constant-folding explanations;
- runtime CUDA Graph claims not separately audited;
- cache/runtime allocator pages reserved for their later domain audits;
- NPU/out-of-tree behavior without its own pinned source authority;
- fixed performance numbers or generated-kernel splits not observed in a recorded Lab.

This is deliberate information preservation: the ledger retains every unit and destination, while
the new series refuses to convert an unverified legacy statement into an authoritative claim.
