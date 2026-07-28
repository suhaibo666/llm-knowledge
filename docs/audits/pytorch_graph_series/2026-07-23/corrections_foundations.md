# PyTorch Graph Series — Foundations Audit and Corrections

## Audit scope and authority

- Current source authority: PyTorch commit `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52` (2026-07-23).
- Detached source root: `E:/97-codes/torch_parallel/p`.
- Historical baselines named by the audited pages:
  - `9922478dffa606fd798cc2346a227d4867e8b6ee` (2026-06-09);
  - `ea5655fcebf726ec4cf1a859de75d2d0e6425805` (2026-07-21).
- A mechanically valid path and line range is not treated as semantic verification.
- This report covers the foundational portions of FX, capture frontends, Export, FakeTensor,
  symbolic shapes, guards and dynamic-shape code generation. AOTAutograd partitioning,
  PatternMatcher, pass-stage ordering and Inductor scheduling are audited in their own batches.

## Current-source facts that remain valid

### FX storage and execution model

1. `torch.fx.Node` is the public FX operation/value node. Its six ordinary opcodes and their
   `target`/`args` meanings are documented at `torch/fx/node.py:258-304`.
2. A node's structured `args` and `kwargs` hold producer `Node` references. `_input_nodes`
   deduplicates its input nodes, while each producer's `users` is an ordered-set-like dictionary;
   one consumer appears once even if it uses the producer multiple times
   (`torch/fx/node.py:286-304`).
3. `_update_args_kwargs` removes the old reverse-use relationships, maps the new aggregate, and
   updates both `_input_nodes` and producer `users`
   (`torch/csrc/fx/node.cpp:307-359`).
4. Program order is an intrusive doubly linked list. `Graph` owns a root sentinel, insertion
   cursor, length, namespace and `_FindNodesLookupTable`
   (`torch/csrc/fx/node.cpp:154-205`; `torch/fx/graph.py:1445-1465`).
5. `Graph.find_nodes` uses the side table. `call_function` is indexed by `(op, target)`; the
   other ordinary opcodes are indexed by `op` and optionally filtered by target
   (`torch/fx/graph.py:1360-1393,1497-1522`).
6. `Graph.create_node` constructs a `Node`, inserts it at the current insertion point, updates
   the lookup table and length (`torch/fx/graph.py:1585-1661`). `erase_node` rejects live users
   and then removes list, lookup and input-use state (`torch/fx/graph.py:1674-1713`).
7. `Graph.lint` verifies ownership, side-table membership, producer-before-consumer order,
   unique names, opcodes and module targets (`torch/fx/graph.py:2610-2687`).
8. `GraphModule` generates a Python `forward`. Reassigning `.graph` recompiles automatically;
   mutating the existing graph requires an explicit `recompile()`
   (`torch/fx/graph_module.py:517-528,924-1008`). Generated source is registered through a
   loader used by `linecache` (`torch/fx/graph_module.py:80-115,136-165`).

### Capture products are related but not interchangeable

1. `symbolic_trace` runs `Tracer.trace` and returns a `GraphModule`
   (`torch/fx/_symbolic_trace.py:1361-1421`). `Proxy` is a `Node` wrapper; overloaded calls and
   `__torch_function__` route to `TracerBase.create_proxy`
   (`torch/fx/proxy.py:600-635,680-805`).
2. `make_fx(f, ...)` returns a callable; invoking that callable with examples returns a
   `GraphModule` containing the operations executed under ProxyTensor tracing
   (`torch/fx/experimental/proxy_tensor.py:3312-3385`). Its current modes are:
   `real`, `fake`, and the backward-compatible but explicitly discouraged `symbolic` mode.
3. Dynamo's backend-facing product contains a `GraphModule`, example inputs, the tracing
   `FakeTensorMode`, and symbolic contexts (`torch/_dynamo/convert_frame.py:1052-1067`).
   Guards are separately accumulated and compiled into the cache-entry check function
   (`torch/_dynamo/convert_frame.py:986-1021,1903-1927`).
4. `torch.export.export` currently defaults to `strict=False`
   (`torch/export/__init__.py:59-69`). Non-strict tracing uses the Python runtime and validates
   critical shape-safety assumptions; strict tracing uses Dynamo and supplies the stronger
   soundness path (`torch/export/__init__.py:179-187`).
5. An `ExportedProgram` owns a graph module, graph signature, state dictionary, range
   constraints, module-call graph, constants and verifiers
   (`torch/export/exported_program.py:1069-1152`). It is not directly callable:
   `__call__` raises and instructs callers to use `.module()`
   (`torch/export/exported_program.py:1457-1461`).
6. `.module(check_guards=True)` un-lifts state and inserts a guard-checking submodule after the
   placeholders; `check_guards=False` uses a more limited pre-hook path
   (`torch/export/exported_program.py:1478-1501`).

### Fake values, symbolic values and graph metadata are different layers

1. `FakeTensor` is a meta-backed tensor subclass that also models the logical device
   (`torch/_subclasses/fake_tensor.py:834-845`). `FakeTensorMode` optionally owns a `ShapeEnv`;
   absence of a `ShapeEnv` normally implies static fake shapes
   (`torch/_subclasses/fake_tensor.py:1526-1536,1550-1587,1627`).
2. `SymNode` owns the SymPy expression, optional `ShapeEnv`, Python scalar type, hint and
   optional constant. A hint is a value from the tracing run and is not an invariant; an
   unbacked symbol has no such hint (`torch/fx/experimental/sym_node.py:89-136`).
3. ProxyTensor places extracted fake/symbolic values in `node.meta["val"]` and best-effort
   tensor metadata in `node.meta["tensor_meta"]`
   (`torch/fx/experimental/proxy_tensor.py:817-835,934-1035`).
4. `meta["val"]` is not an autograd-complete runtime tensor model. The source explicitly permits
   dtype, shape, stride and storage queries but says not to rely on `requires_grad`, `grad_fn`
   or `_base` (`torch/fx/experimental/proxy_tensor.py:817-823`).
5. Dynamo guards are not FX data edges. `Guard`, `GuardsSet`, `ShapeGuard`, the generated guard
   manager, Export graph signatures and range constraints are companion state that establishes
   where a captured graph is valid (`torch/_guards.py:246-320,642-680`;
   `torch/export/graph_signature.py:81-176`).

## Required corrections

Severity means:

- **high**: the old statement can produce a wrong execution, validity or mental model;
- **medium**: the main mechanism exists, but the stated boundary or control flow is wrong;
- **low**: the mechanism remains valid and the correction is primarily source/provenance drift.

| ID | Severity | Legacy baseline status | Primary destination |
|---|---|---|---|
| F-001 | high | 2026-05-11 historical main; commit not recorded | `04_symbolic_shapes_guards_and_graph_reuse.md` |
| F-002 | high | 2026-05-11 historical main; commit not recorded | `04_symbolic_shapes_guards_and_graph_reuse.md` |
| F-003 | medium | 2026-05-11 historical main; commit not recorded | `04_symbolic_shapes_guards_and_graph_reuse.md` |
| F-004 | high | 2026-05-11 historical main; commit not recorded | `04_symbolic_shapes_guards_and_graph_reuse.md` |
| F-005 | high | 2026-05-11 historical main; commit not recorded | `04_symbolic_shapes_guards_and_graph_reuse.md` |
| F-006 | high | historical main; exact codegen-page commit not recorded | `04_symbolic_shapes_guards_and_graph_reuse.md` |
| F-007 | medium | 2026-05-11 historical main; commit not recorded | `17_fx_lowering_to_inductor_ir.md`, `21_codegen_kernel_mapping_autotuning_and_provenance.md` |
| F-008 | high | `9922478dffa606fd798cc2346a227d4867e8b6ee` | `03_graph_values_metadata_and_signatures.md`, `07_graph_capture_frontends_and_tracing.md` |
| F-009 | high | `9922478dffa606fd798cc2346a227d4867e8b6ee` | `03_graph_values_metadata_and_signatures.md`, `08_graph_normalization_decomposition_and_functionalization.md` |
| F-010 | medium | `ea5655fcebf726ec4cf1a859de75d2d0e6425805` | `14_dead_code_topology_and_effect_order.md` |
| F-011 | low | `ea5655fcebf726ec4cf1a859de75d2d0e6425805` | `09_aotautograd_joint_forward_backward_graphs.md`, `10_saved_tensors_recompute_and_runtime_abi.md` |
| F-012 | high | historical main; exact codegen-page commit not recorded | `21_codegen_kernel_mapping_autotuning_and_provenance.md` |
| F-013 | medium | historical main; exact codegen-page commit not recorded | `03_graph_values_metadata_and_signatures.md`, `04_symbolic_shapes_guards_and_graph_reuse.md` |
| F-014 | medium | historical main; exact codegen-page commit not recorded | `21_codegen_kernel_mapping_autotuning_and_provenance.md` |
| F-015 | low | historical main; exact codegen-page commit not recorded | `17_fx_lowering_to_inductor_ir.md`, `19_buffer_liveness_memory_planning_and_reuse.md`, `21_codegen_kernel_mapping_autotuning_and_provenance.md` |
| F-016 | medium | `9922478dffa606fd798cc2346a227d4867e8b6ee` | `07_graph_capture_frontends_and_tracing.md` |
| F-017 | medium | `9922478dffa606fd798cc2346a227d4867e8b6ee` | `15_graph_pass_pipeline_ordering_and_fixpoint.md` |
| F-018 | high | `9922478dffa606fd798cc2346a227d4867e8b6ee` | supporting custom-op page; linked from `07_graph_capture_frontends_and_tracing.md` |
| F-019 | high | 2026-05-11 historical main; commit not recorded | `04_symbolic_shapes_guards_and_graph_reuse.md` |

### F-001 — A concrete shape is a guard specialization, not literally “the cache key contains the shape”

**Affected page:** `dynamic_shapes_full_analysis.md`, §1.1-1.2.

The old text collapses cache lookup and guard evaluation into one “cache key” model. Dynamo
associates multiple compiled results with a code object and evaluates the guards of cache
entries to decide applicability. A changed size normally fails a guard and can cause a new
entry; this should not be described as the raw size simply being a component of one universal
cache key.

Current anchors:

- `torch/__init__.py:3157-3166` — per-code-object compiled results and guard failure;
- `torch/_dynamo/config.py:115-149` — `recompile_limit=8` and accumulated limit;
- `torch/_dynamo/convert_frame.py:1002-1021,1903-1927` — building the executable check function.

**Replacement wording:** “Dynamo stores guarded compiled entries for a code object. Runtime
inputs select an entry by satisfying its guards; a size specialization is one such guard.”

### F-002 — `dynamic=True`, `False`, `None`, and `dynamic_shapes=` have distinct current contracts

**Affected page:** `dynamic_shapes_full_analysis.md`, §2.4 and §4.

The statement “`dynamic=False` still allows dimensions explicitly marked by `mark_dynamic` to
be symbolic” is not the current public contract. The `torch.compile` docstring says:

- `dynamic=True`: attempt an up-front, maximally dynamic kernel;
- `dynamic=False`: never generate dynamic kernels and always specialize;
- `dynamic=None`: start with automatic detection and make a later compilation more dynamic
  after relevant guard failures.

`make_set_enable_dynamic(False)` disables automatic dynamic shapes and enables the
assume-static policy (`torch/_dynamo/eval_frame.py:744-753`), but this internal patcher is not
evidence for the stronger old claim.

The current API also has `dynamic_shapes=`, which is mutually exclusive with `dynamic=` and is
normalized to the newer `ShapesSpec` representation
(`torch/_dynamo/eval_frame.py:838-902`; `torch/__init__.py:3134-3148,3175-3181`).

### F-003 — `ShapeEnv` state names and meanings were oversimplified

**Affected page:** `dynamic_shapes_full_analysis.md`, §2.2-2.3.

- The authoritative mapping is `backed_var_to_val`; `var_to_val` is now a deprecated property
  (`torch/fx/experimental/symbolic_shapes.py:5960-5980`).
- `size_like` does not mean that the actual symbol is globally constrained to be `>= 2` in
  every reasoning context. The source says it is the set for which *size-oblivious tests* may
  make that assumption (`torch/fx/experimental/symbolic_shapes.py:4064-4068`).
- `var_to_range` is conservative: the actual value must be within the recorded range, but the
  range may include impossible values
  (`torch/fx/experimental/symbolic_shapes.py:4024-4028`).
- `DimDynamic.DUCK` unifies equal hints; `DYNAMIC` does not promise an independent symbol in
  every later simplification, because constraints can still relate or replace symbols
  (`torch/fx/experimental/symbolic_shapes.py:1988-2019`).

### F-004 — Range refinement is compile-time reasoning, not learning from later runtime calls

**Affected page:** `dynamic_shapes_full_analysis.md`, §3.5.

The old three-step explanation says that runtime-observed shapes progressively narrow
`var_to_range` and may turn an existing symbol into a constant. That is incorrect.
`_refine_ranges` is invoked while the `ShapeEnv` reasons about a relational expression during
tracing/guard construction
(`torch/fx/experimental/symbolic_shapes.py:8073-8134,9047-9105`). Later runtime calls
evaluate the generated guards or select/recompile cache entries; they do not mutate the
already-compiled graph's `ShapeEnv` in place to narrow its ranges.

**Replacement wording:** “Within one tracing/compilation, learned relational facts can refine
the active ShapeEnv. Across calls, new observations affect cache-entry selection and possibly a
new compilation, not the old ShapeEnv.”

### F-005 — Internal equality replacement does not eliminate the need for runtime equality guards

**Affected page:** `dynamic_shapes_full_analysis.md`, §3.6.

The old text says no `s0 == s1` guard is needed because the replacement records the equality.
The replacement is an internal simplification justified by a fact that future inputs must
still satisfy. `produce_guards_verbose` explicitly warns that some equality guards are
non-trivial (`torch/fx/experimental/symbolic_shapes.py:6031-6046`). A relationship may be
expressed through duck
sizing/source equality rather than rendered in the exact textual form `s0 == s1`, but the
runtime validity condition cannot simply disappear.

### F-006 — `assert_size_stride` does not enforce every symbolic range or divisibility constraint

**Affected pages:** `dynamic_shapes_full_analysis.md`, §3.4;
`inductor_codegen_dynamic_shape_analysis.md`, §5.1.

`codegen_input_size_asserts` records expected size and stride tuples for graph inputs
(`torch/_inductor/codegen/wrapper.py:1719-1741`), which are later emitted as
`assert_size_stride` calls (`torch/_inductor/codegen/wrapper.py:1841-1904`). If `s0` was itself
loaded from the same
input's size, checking that dimension against `s0` does not establish an unrelated range such
as `s0 <= 1024` or an arbitrary divisibility property. Such conditions belong to Dynamo
guards, Export checks, Inductor guards, or deferred runtime assertions according to where the
fact was introduced.

### F-007 — The matmul example must not promise a Triton pointwise kernel

**Affected page:** `dynamic_shapes_full_analysis.md`, §5.5.

“Inductor lowers `matmul` to a Triton kernel” is too strong. Matmul may select an external
kernel, a template, CUTLASS/CK, native Triton code, or another backend-specific path depending
on device, layout, dtype, configuration and autotuning. The only safe invariant for this
example is that symbolic dimensions survive through lowering and participate in layout,
guard, allocation and launch decisions.

### F-008 — Current Export shape UX and runtime contract are missing

**Affected pages:** both FX/export pages, especially their Export sections.

The existing material correctly records the legacy `Dim` form and `strict=False`, but the
current source marks `ShapesSpec`/`ParamsSpec` as the recommended shape-specification path for
export and compile (`torch/export/__init__.py:133-176`). The refactored series must explain
both:

- `Dim`/container specifications as the established API;
- `ShapesSpec` as the current unbacked, cross-compile/export specification with assumptions
  and derived expressions.

It must also state that an `ExportedProgram` is not directly callable and that
`ep.module()` performs un-lifting plus input guard installation
(`torch/export/exported_program.py:1457-1501`).

### F-009 — “Lifted state” and “functional graph” are separate ideas

**Affected page:** `fx_graph_export_and_custom_ops_analysis.md`, §6.

Lifting parameters/buffers means state access is represented by explicit graph inputs and the
signature maps these inputs back to state. Functionalization separately removes mutation by
representing updated state as outputs. `ExportGraphSignature` documents both the lifted inputs
and mutation outputs (`torch/export/graph_signature.py:166-181`).

The refactored text must not use “lifted” as a synonym for “all operations are pure.” Public
`torch.export.export` targets a functional ATen graph, while training/pre-dispatch variants and
intermediate graphs may have different invariants; `run_decompositions()` additionally targets
the Core ATen operator set (`torch/export/exported_program.py:1513-1532`).

### F-010 — DCE now includes nested `GraphModule` cleanup

**Affected page:** `fx_graph_construction_and_transformation_analysis.md`, §7.

The root rule remains “no users and not impure,” and reverse traversal still cascades deletes.
Current `Graph.eliminate_dead_code` first calls `lint`, then recursively applies DCE and
`recompile()` to referenced child `GraphModule` values reached by `get_attr`
(`torch/fx/graph.py:2690-2776`). Complexity and explanations must therefore distinguish one
graph from the complete nested graph-module tree.

### F-011 — AOTAutograd partitioner moved; this is locator drift, not yet a mechanism change

**Affected page:** `fx_graph_construction_and_transformation_analysis.md`.

The old full paths under `torch/_functorch/_aot_autograd/partitioners.py` no longer exist at the
current baseline. The implementation is now `torch/_functorch/partitioners.py`; the named
functions still occur at:

- `_extract_graph_with_inputs_outputs`: line 514;
- `_extract_fwd_bwd_modules`: line 1343;
- `default_partition`: line 1595;
- `min_cut_rematerialization_partition`: line 4091.

Whether every surrounding AOT claim remains semantically current is intentionally deferred to
the AOT-specific audit; this correction only resolves the path move.

### F-012 — The XBLOCK “three modes” model is not the current implementation

**Affected page:** `inductor_codegen_dynamic_shape_analysis.md`, §9.1-9.3.

The old text describes default pointwise codegen as a runtime
`@triton.heuristics(values={"XBLOCK": lambda meta: ...xnumel...})` and claims that every new
runtime `xnumel` selects/compiles another XBLOCK. Current code instead:

1. computes compile-time optimization hints for each symbolic numel
   (`torch/_inductor/codegen/triton.py:7064-7073`);
2. asks a device-registered pointwise heuristic for one or more Triton configs
   (`torch/_inductor/runtime/triton_heuristics.py:4365-4435`;
   `torch/_inductor/heuristics/triton_codegen/pointwise.py:29-116`);
3. uses the selected config's constexpr block sizes while the launch grid remains a runtime
   expression over `xnumel`, `ynumel`, and `znumel`
   (`torch/_inductor/runtime/triton_heuristics.py:5005-5152`).

The published candidate sequences and the statement `key=['xnumel']` are therefore not a valid
general description of the current Inductor autotuner. The claimed 0.1-0.5 second and
“several-second” costs are also unsupported by the source and must be removed unless backed by
a pinned benchmark.

The table that attributes the largest pointwise-XBLOCK penalty to GEMM is structurally wrong:
GEMM/template/external-kernel selection has its own search spaces and should not be explained
as an XBLOCK extension of the ordinary pointwise heuristic.

### F-013 — Unbacked symbols are not “symbols with no compile-time constraints”

**Affected page:** `inductor_codegen_dynamic_shape_analysis.md`, §4.1.

An unbacked symbol has no input-derived concrete hint at trace time. It can still have a
`ValueRanges` bound and deferred runtime assertions. `ShapeEnv.deferred_runtime_asserts`
exists specifically for conditions that become checkable after unbacked values enter scope
(`torch/fx/experimental/symbolic_shapes.py:4074-4104`). The old definition should be narrowed
to “no backed hint,”
not “no constraints.”

### F-014 — C++ symbolic arguments are not generally declared as `uint32_t`

**Affected page:** `inductor_codegen_dynamic_shape_analysis.md`, §7.

`SymbolicCallArgLine` delegates to the wrapper's symbolic-call helper
(`torch/_inductor/codegen/wrapper.py:1290-1299`). The C++ wrapper's ordinary declaration prefix
is `auto`, while `uint32_t` is used for launch-grid ABI values
(`torch/_inductor/codegen/cpp_wrapper_cpu.py:227-238,276-295`;
`torch/_inductor/codegen/cpp_wrapper_gpu.py:492,510-512`). The old statement conflates
symbolic scalar temporaries with
grid dimensions.

### F-015 — `SizeArg`, dynamic numel, grid expressions and symbolic reuse keys remain valid but moved

**Affected page:** `inductor_codegen_dynamic_shape_analysis.md`, §§2-3, §6.1 and source index.

The mechanisms remain current:

- `SizeArg`: `torch/_inductor/codegen/common.py:285-292`;
- kernel range numels appended to signature: `torch/_inductor/codegen/triton.py:7137-7146`;
- complex numel call arguments: `torch/_inductor/codegen/triton.py:7436-7446`;
- `GridExpr`: `torch/_inductor/runtime/triton_heuristics.py:5005-5152`;
- symbolic buffer reuse key, now also including stream and mempool:
  `torch/_inductor/codegen/wrapper.py:108-138`;
- unbacked output bindings: `torch/_inductor/codegen/wrapper.py:4409-4465`;
- 32-bit indexing decision and installed size guards:
  `torch/_inductor/codegen/simd.py:3068-3103`.

The historical line numbers should be replaced rather than treating these sections as removed.

### F-016 — `__torch_function__` and `__torch_dispatch__` must not be called the same mechanism

**Affected page:** `fx_graph_export_and_custom_ops_analysis.md`, §1.

FX symbolic `Proxy` implements `__torch_function__` and also uses generated Python operator and
method overloads. ProxyTensor/FakeTensor paths use dispatch-mode and `__torch_dispatch__`
machinery at a lower operator-dispatch layer. They are related extensibility protocols, not one
interchangeable interception point. This distinction is essential to explaining why
`symbolic_trace` and `make_fx` observe different operator levels.

### F-017 — `PassResult.modified` only drives a fixed-point loop when `PassManager.steps` permits it

**Affected pages:** both FX/export pages, PassBase sections.

`PassBase.__call__` only runs `requires`, `call`, and `ensures`
(`torch/fx/passes/infra/pass_base.py:28-78`). `PassManager` has `steps=1` by default; it repeats
until no pass reports modification only within that configured maximum, and it recompiles a
`GraphModule` after each managed pass (`torch/fx/passes/infra/pass_manager.py:154-195,254-317`).
The old wording should not imply that any `PassResult(modified=True)` automatically causes an
unbounded or default fixed-point iteration.

### F-018 — The quickstart custom-op fake implementation has an unstated contract

**Affected page:** `fx_export_custom_op_quickstart.md`, §4.

The numerical implementation calls `torch.mul(x, y)`, which supports broadcasting and dtype
promotion, but the fake implementation returns `empty_like(x)`. That fake implementation is
correct only if the custom operator's contract requires `x` and `y` to have the same shape and
the output dtype/layout to match `x`. The page does not declare those restrictions.

The new example must either:

- explicitly validate and document the same-shape/same-dtype contract; or
- implement fake behavior with the same broadcast and promotion semantics as the numerical
  kernel.

The API locations remain current with line drift:
`torch/_library/custom_ops.py:67,272,392,523,639` and
`torch/library.py:1016,1158,1321,1786`.

### F-019 — Automatic dynamic shapes do not unconditionally add exclusion guards

**Affected page:** `dynamic_shapes_full_analysis.md`, §4.2-4.3.

The workflow diagram shows every static-to-dynamic recompilation producing an exclusion guard
such as `size[0] != 3`, then using that guard to route the original size back to an older static
entry. Current `automatic_dynamic_exclusion_guard` defaults to `False`
(`torch/_dynamo/config.py:209-214`). When enabled, the mechanism uses
`FrameStateSizeEntry.automatic_dynamic_exclusions`, records the concrete static values, and
propagates them into symbolic contexts and guard production
(`torch/_dynamo/pgo.py:416-465`;
`torch/_dynamo/variables/builder.py:3362-3396`;
`torch/_dynamo/guards.py:3326-3367`).

The refactored explanation must therefore separate:

- automatic dynamic generalization after a relevant recompile, which is the ordinary policy;
- the optional exclusion-guard optimization, which must not appear as an unconditional stage.

### F-020 — A historical source index is navigation, not current evidence

The source-index block in `dynamic_shapes_full_analysis.md` mixes old line numbers with mechanism
labels. A path or symbol that still exists does not make the old range an implementation proof.
The numbered course therefore reopens the current `ShapeEnv`, guard and runtime-assertion
implementations and binds every use to the pinned SHA; the old index remains only as historical
navigation. This is the same evidence rule enforced by the canonical locator validator, not a
claim that all listed mechanisms disappeared.

## Migration consequences for the new series

1. `02_fx_graph_core_data_model.md` may reuse the verified FX linked-list/use-def material, updated
   to include nested DCE and current source anchors.
2. `03_graph_values_metadata_and_signatures.md` must explicitly separate:
   runtime tensor, fake tensor, symbolic scalar, FX `Node` reference, `node.meta`, graph
   signature, range constraints, and guard state.
3. `04_symbolic_shapes_guards_and_graph_reuse.md` must be rewritten around compile-time ShapeEnv
   reasoning and runtime guarded cache-entry selection. Sections F-001 through F-006 and F-019 are
   blockers; the old narrative cannot be migrated verbatim.
4. `07_graph_capture_frontends_and_tracing.md` must compare concrete output contracts:
   `symbolic_trace -> GraphModule`, `make_fx(f)(args) -> GraphModule`,
   Dynamo backend input plus external guards, and `export -> ExportedProgram`.
5. `21_codegen_kernel_mapping_autotuning_and_provenance.md` must discard the old XBLOCK
   three-mode model and rebuild from the current heuristic registry, config generation,
   autotuner and grid-expression paths.
6. Custom-op and `torch.func` material remains useful domain documentation, but it should be
   linked from the graph series rather than consuming the graph-series mainline unless needed
   to explain opaque calls or higher-order transforms.

## Foundation audit gate

The following claims are safe foundations for writing:

- FX storage, use-def maintenance, linked-list order, lookup table, lint and recompile;
- the four capture products and their distinct output contracts;
- FakeTensor/SymNode/ShapeEnv/node-meta layer separation;
- Dynamo guards and Export signature/range constraints as companion state;
- current `SizeArg`, grid-expression, unbacked-binding and symbolic reuse-key mechanisms.

The following old sections are blocked from direct migration:

- runtime “learning” and range tightening;
- equality replacement without runtime validity checks;
- `dynamic=False` with partial dynamic dimensions;
- `assert_size_stride` as a universal range/divisibility checker;
- matmul always becoming a Triton kernel;
- unconditional automatic-dynamic exclusion guards;
- the XBLOCK candidate/mode/per-runtime-numel story;
- unbacked meaning unconstrained;
- C++ symbolic temporaries being `uint32_t`.
