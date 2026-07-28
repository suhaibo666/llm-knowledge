# PyTorch Graph Series — AOTAutograd Audit and Corrections

## Audit scope and authority

- Current source authority: PyTorch commit
  `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52` (2026-07-23).
- Detached source root: `E:/97-codes/torch_parallel/p`.
- Audited legacy pages:
  - `wiki/02_engineering/01_ai_frameworks/03_aot_autograd/aotautograd_analysis.md`;
  - `wiki/02_engineering/01_ai_frameworks/03_aot_autograd/aot_autograd_quickstart.md`;
  - AOTAutograd portions of
    `wiki/02_engineering/01_ai_frameworks/03_aot_autograd/fx_graph_construction_and_transformation_analysis.md`.
- Historical pages are audit leads, not source authority. Code blocks in those pages have not been
  executed on the pinned source and therefore are not classified as verified Labs.

This report covers metadata collection, functionalization, joint-graph capture, fw/bw extraction,
saved-value selection, recomputation, runtime calling conventions and their principal complexity.
PatternMatcher, general DCE/topology and Inductor backend passes are handled by later audit batches.

## The current mental model

### There are three graph states, not one graph that gradually changes identity

1. Metadata collection runs the forward under `FunctionalTensorMode` and returns
   `ViewAndMutationMeta`; it explicitly does **not** construct an FX graph
   (`torch/_functorch/_aot_autograd/collect_metadata_analysis.py:167-242`).
2. When autograd is needed, `aot_dispatch_autograd_graph` constructs joint inputs as
   `(primals, traced_tangents)`, prepares `create_joint`, and captures a joint FX `GraphModule`
   with `make_fx` (`torch/_functorch/_aot_autograd/graph_capture.py:472-536`;
   `torch/_functorch/_aot_autograd/graph_capture.py:92-183`).
3. A partitioner creates two new FX graphs from the joint graph. `_extract_graph_with_inputs_outputs`
   allocates a fresh `fx.Graph`, creates fresh placeholders, and maps old joint nodes to copied
   nodes in the new graph (`torch/_functorch/partitioners.py:514-615`). The result is two separate
   `GraphModule` objects (`torch/_functorch/partitioners.py:1573-1592`).

Consequently, a joint node, a forward node and a backward node may describe related computation,
but they are not the same `Node` object and do not belong to the same `Graph`.

### Joint graph semantics

`create_joint` first executes the prepared forward. Existing forward nodes are tagged
`is_forward`, while tangent-originated and backward-created nodes are tagged `is_backward`
(`torch/_functorch/_aot_autograd/graph_capture_wrappers.py:294-345`). It then selects the outputs
that participate in differentiation and calls `torch.autograd.grad`; the joint function returns
the forward-side outputs plus one gradient-or-`None` slot per primal
(`torch/_functorch/_aot_autograd/graph_capture_wrappers.py:348-477`).

The joint graph is therefore a convenient supergraph for partition decisions. It is not the
runtime object that directly connects two independently executing graphs.

### Forward/backward relationship is an ABI, not an FX edge

The forward and backward graphs have no cross-`Graph` `Node` edge. The relationship is encoded by:

1. a partition-time old-node-to-new-node mapping;
2. a forward output layout;
3. runtime state stored on the generated `torch.autograd.Function` context;
4. backward placeholders in a prescribed order.

The current core saved-value order is:

```text
forward graph:
  inputs  = primals (+ optional forward RNG inputs)
  outputs = forward-visible prefix
            + tensors requiring version-counter checks
            + tensors without version-counter checks
            + opaque objects
            + symbolic scalar values

backward graph:
  inputs  = symbolic scalar values
            + saved tensors
            + opaque objects
            + tangents
            + optional backward RNG inputs
            + optional BackwardState
  outputs = gradients corresponding to primal inputs
```

The forward-visible prefix itself includes mutated-input returns, user outputs and intermediate
bases, with effect-token/RNG adaptations applied by wrappers. The exact slices are carried in
`ViewAndMutationMeta` (`torch/_functorch/_aot_autograd/schemas.py:446-565`;
`torch/_functorch/_aot_autograd/schemas.py:781-863`). Partition extraction constructs the forward
and backward signatures at `torch/_functorch/partitioners.py:1473-1546`.

At runtime, version-checked tensors go through `ctx.save_for_backward`; the no-version-check group,
symbolic scalars and opaque objects are stored separately
(`torch/_functorch/_aot_autograd/runtime_wrappers.py:2615-2683`). The generated backward prologue
reconstructs its argument vector as symbolic values, tensors, opaque objects, filtered gradients,
then optional effect tokens and functionalized RNG state
(`torch/_functorch/_aot_autograd/runtime_wrappers.py:2982-3089`).

### Recompute is graph extraction policy, not a Node opcode

The partitioner selects a boundary:

- values on the saved side become additional forward outputs and backward placeholders;
- a required forward operation not cut by that boundary is copied into the backward graph and
  executes again there.

Recomputed operations are ordinary copied FX nodes. No `recompute` opcode is introduced.
`node.meta["recompute"]` is policy/provenance metadata used to constrain selection, not runtime
connectivity (`torch/_functorch/partitioners.py:1690-1770`;
`torch/_functorch/partitioners.py:2630-2765`).

The min-cut flow graph splits each candidate `X` into `X_in -> X_out`; the capacity of that internal
edge represents the cost of saving `X`, dependency edges have infinite capacity, and source/sink
constraints encode values that must be saved, recomputed, or available to backward. The cut's
internal edges become saved values (`torch/_functorch/partitioners.py:2641-2765`;
`torch/_functorch/partitioners.py:2888-3069`).

After extraction, the backward graph initially retains joint-graph order. That can place all copied
forward computation before the gradient work and keep its temporaries alive too long.
`reordering_to_mimic_autograd_engine` creates another graph and materializes each prerequisite
subgraph only when a backward node needs it (`torch/_functorch/partitioners.py:1920-1995`).

## Current-source facts that remain valid

### Metadata and functionalization

1. Metadata analysis detects data, metadata, storage-metadata and shallow-copy mutations, whether
   mutations are hidden from autograd or occur under no-grad/inference, input aliasing, output
   aliasing, intermediate bases, output differentiability and tangent candidates
   (`torch/_functorch/_aot_autograd/collect_metadata_analysis.py:252-289`;
   `torch/_functorch/_aot_autograd/collect_metadata_analysis.py:291-510`;
   `torch/_functorch/_aot_autograd/collect_metadata_analysis.py:760-805`).
2. `InputAliasInfo` and `OutputAliasInfo` are calling-convention data, not merely descriptive
   annotations. Current fields include `mutation_is_shallow_copy_data` and
   `requires_grad_for_backward`, which are absent from the legacy snapshots
   (`torch/_functorch/_aot_autograd/schemas.py:83-180`).
3. Duplicate input objects are handled before capture by `AOTDedupeWrapper`; aliased and mutated
   input views can be merged into synthetic bases by `AOTSyntheticBaseWrapper`
   (`torch/_functorch/_aot_autograd/graph_compile.py:185-189`;
   `torch/_functorch/_aot_autograd/runtime_wrappers.py:1586-1766`;
   `torch/_functorch/_aot_autograd/runtime_wrappers.py:1844-2018`).
4. Functionalization is conditional. When enabled, AOT asserts a functional graph, but supported
   input mutations may be represented by a controlled `copy_` tail; when
   `disable_functionalization=True`, that contract is intentionally not enforced
   (`torch/_functorch/_aot_autograd/graph_capture.py:340-403`;
   `torch/_functorch/_aot_autograd/graph_capture_wrappers.py:1030-1144`).
5. Tensor subclass flattening and effect-token handling are distinct transformations around
   functionalization, not parts of a single undifferentiated “functionalization” step
   (`torch/_functorch/_aot_autograd/graph_capture.py:214-263`).

### Partition and compilation

1. `default_partition` preserves original forward placement and generally saves forward values
   needed by backward. It also honors explicit recompute tags and can fall back to min-cut for
   activation checkpointing on a non-functional graph
   (`torch/_functorch/partitioners.py:1595-1671`;
   `torch/_functorch/partitioners.py:1708-1805`).
2. Inductor's normal partition function runs joint-graph passes and, absent a custom partitioner,
   calls `min_cut_rematerialization_partition`
   (`torch/_inductor/compile_fx.py:2454-2518`). The standalone `aot_function` API still defaults
   its `partition_fn` argument to `default_partition`
   (`torch/_functorch/aot_autograd.py:712-738`).
3. `aot_eager` explicitly uses min-cut; `aot_eager_default_partitioner` provides the contrasting
   debug backend (`torch/_dynamo/backends/debugging.py:417-440`).
4. Backward compilation may happen ahead of time so its guards are available at forward time, but
   the runtime also supports a lazy backward compiler path. Saved-activation strides can be
   adjusted to the layout selected by the forward compiler
   (`torch/_functorch/_aot_autograd/graph_compile.py:2304-2365`).
5. Compiler choices are explicit stage arguments. `aot_stage2_compile` does not mutate the
   `AOTConfig` compiler fields as the legacy pseudocode claims
   (`torch/_functorch/_aot_autograd/graph_compile.py:339-370`;
   `torch/_functorch/_aot_autograd/schemas.py:1271-1277`).

### Runtime wrappers

Compiler wrappers exist because a calling-convention transformation needs two inverse halves:
one before capture/compile and one after compilation. The base contract says exactly this
(`torch/_functorch/_aot_autograd/schemas.py:1302-1315`).

For the main dispatch path, dedupe then synthetic-base wrappers run in pre-compile order; their
post-compile halves run in reverse (`torch/_functorch/_aot_autograd/graph_compile.py:185-189`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:3862-3896`). Forward compiler adaptation also
has explicit effect-token, subclass, functionalized-RNG and fake-output wrappers
(`torch/_functorch/_aot_autograd/graph_compile.py:2740-2881`). Finally,
`AOTDispatchAutograd.post_compile` builds the custom `torch.autograd.Function` and places the
general `RuntimeWrapper` around its `.apply`
(`torch/_functorch/_aot_autograd/runtime_wrappers.py:3806-3816`).

## Required corrections

Severity means:

- **high**: the old statement creates an incorrect graph, ABI or compilation mental model;
- **medium**: the named mechanism exists, but its boundary, ordering or applicability is wrong;
- **low**: semantics mostly survive and the problem is stale source, incomplete fields or wording.

| ID | Severity | Legacy baseline status | Primary destination |
|---|---|---|---|
| A-001 | high | unknown | `09_aotautograd_joint_forward_backward_graphs.md` |
| A-002 | high | unknown | `09_aotautograd_joint_forward_backward_graphs.md` |
| A-003 | medium | unknown | `05_graph_effects_alias_mutation_and_order.md`, `09_aotautograd_joint_forward_backward_graphs.md` |
| A-004 | high | unknown | `08_graph_normalization_decomposition_and_functionalization.md` |
| A-005 | high | unknown | `11_graph_stage_boundaries_identity_and_provenance.md` |
| A-006 | high | `ea5655f` report is partially correct | `09_aotautograd_joint_forward_backward_graphs.md`, `10_saved_tensors_recompute_and_runtime_abi.md` |
| A-007 | high | unknown | `10_saved_tensors_recompute_and_runtime_abi.md` |
| A-008 | medium | `ea5655f` report is substantially correct | `10_saved_tensors_recompute_and_runtime_abi.md` |
| A-009 | high | unknown | `10_saved_tensors_recompute_and_runtime_abi.md` |
| A-010 | medium | unknown | `10_saved_tensors_recompute_and_runtime_abi.md` |
| A-011 | high | unknown | `08_graph_normalization_decomposition_and_functionalization.md`, `15_graph_pass_pipeline_ordering_and_fixpoint.md` |
| A-012 | high | unknown | `09_aotautograd_joint_forward_backward_graphs.md` |
| A-013 | medium | unknown | `05_graph_effects_alias_mutation_and_order.md` |
| A-014 | medium | unknown | `04_symbolic_shapes_guards_and_graph_reuse.md`, `10_saved_tensors_recompute_and_runtime_abi.md` |
| A-015 | medium | unknown | `06_structured_outputs_higher_order_and_nested_graphs.md`, `09_aotautograd_joint_forward_backward_graphs.md` |
| A-016 | medium | unknown | `16_graph_rewrite_legality_validation_and_complexity.md` |
| A-017 | medium | unknown | `11_graph_stage_boundaries_identity_and_provenance.md` |
| A-018 | medium | unknown | `10_saved_tensors_recompute_and_runtime_abi.md` |
| A-019 | low | unknown | `09_aotautograd_joint_forward_backward_graphs.md` |
| A-020 | medium | current quickstart line drift | `10_saved_tensors_recompute_and_runtime_abi.md` |

### A-001 — Metadata analysis is not joint-graph capture

**Old claim:** metadata collection belongs inside `aot_stage1_graph_capture`, and the metadata pass
is described as graph tracing.

**Current evidence:** `create_aot_state` runs
`run_functionalized_fw_and_collect_metadata(...)(*fake_args)` before
`aot_stage1_graph_capture` is called
(`torch/_functorch/aot_autograd.py:568-709`;
`torch/_functorch/aot_autograd.py:797-838`). The metadata implementation says “We didn't do any
tracing” after executing the forward under functional tensors
(`torch/_functorch/_aot_autograd/collect_metadata_analysis.py:190-242`).

**Required wording:** metadata analysis is a pre-capture execution/analysis pass whose result
controls later capture and runtime ABI decisions.

### A-002 — The joint graph is produced during stage-1 graph capture, not stage-2 compile

**Old claim:** diagrams place `aot_dispatch_autograd_graph` in the compilation/partition stage.

**Current evidence:** `aot_stage1_graph_capture` calls `aot_dispatch_autograd_graph` or the base
capture path and returns an `AOTGraphCapture`
(`torch/_functorch/_aot_autograd/graph_compile.py:192-284`). `aot_stage2_compile` consumes that
captured graph and selects autograd versus inference compilation
(`torch/_functorch/_aot_autograd/graph_compile.py:339-370`).

### A-003 — Legacy schema excerpts are incomplete and cannot define the ABI

**Old claim:** abbreviated `InputAliasInfo`, `OutputAliasInfo` and `ViewAndMutationMeta` snippets
are presented as their defining structures.

**Current evidence:** the current schemas include additional mutation, backward differentiability,
saved-value grouping, token, RNG, stream and dynamic-saved-tensor fields
(`torch/_functorch/_aot_autograd/schemas.py:83-180`;
`torch/_functorch/_aot_autograd/schemas.py:446-565`).

**Required wording:** use selected-field tables with a pinned source link; do not reproduce a
partial dataclass as if it were exhaustive.

### A-004 — Functionalization is a contract with explicit exceptions, not “remove all side effects”

**Old claim:** “all operations become pure” and every input mutation becomes only an extra output.

**Current evidence:** when functionalization is enabled, AOT checks the graph, but supported
in-graph input mutations can be materialized as a tagged `copy_` tail. Backward mutation support is
limited and separately checked; disabling functionalization bypasses the functional-graph
contract (`torch/_functorch/_aot_autograd/graph_capture_wrappers.py:907-1030`;
`torch/_functorch/_aot_autograd/graph_capture_wrappers.py:1030-1144`;
`torch/_functorch/_aot_autograd/graph_capture.py:340-403`).

**Required wording:** distinguish out-of-graph mutation returns, allowed in-graph copy tails,
effect tokens, unsupported mutations and the explicit disable path.

### A-005 — The wrapper diagram has the wrong composition order

**Old claim:** `create_joint -> create_functionalized_fn -> fn_input_mutations_to_outputs ->
handle_effect_tokens_fn -> aot_dispatch_subclass`.

**Current evidence:** the autograd path prepares the forward, builds `create_joint`, then
`_prepare_graph_capture_tracing` applies functionalization, subclass desugaring and effect-token
handling in that order (`torch/_functorch/_aot_autograd/graph_capture.py:472-521`;
`torch/_functorch/_aot_autograd/graph_capture.py:214-263`). Dedupe and synthetic bases are outer
compiler wrappers applied before this capture path
(`torch/_functorch/_aot_autograd/graph_compile.py:185-223`).

### A-006 — fw/bw independence is correct, but “saved tensors connect them” is incomplete

**Old claim:** the forward and backward graphs are separate and saved tensors are the connecting
mechanism.

**Current evidence:** fresh graphs and placeholders prove structural independence
(`torch/_functorch/partitioners.py:514-615`). The runtime bridge additionally carries symbolic
scalars, opaque objects and tangents, and has optional token/RNG/BackwardState channels
(`torch/_functorch/partitioners.py:1473-1546`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:2615-2683`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:3076-3089`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:3215-3256`).

**Required wording:** call this a versioned runtime ABI, not a hidden cross-graph edge.

### A-007 — Saved activations have two tensor classes and several non-tensor classes

**Old claim:** all saved activations go through `ctx.save_for_backward`.

**Current evidence:** values with eager-style version checking use `save_for_backward`; explicitly
marked no-version-check tensors are stored on `_tensors_no_vc_check`, and symbolic/opaque values
have separate context fields (`torch/_functorch/_aot_autograd/runtime_wrappers.py:2615-2683`).
The partitioner performs a stable grouping before rebuilding graph signatures
(`torch/_functorch/partitioners.py:1473-1546`).

### A-008 — Recompute enters backward by ordinary node copying, followed by a memory-aware reorder

**Old claim:** the broad idea is present, but descriptions may imply special recompute nodes or a
cross-graph link.

**Current evidence:** extraction uses `node_copy` with an old-to-new environment, and the reorder
again copies ordinary nodes into a new graph (`torch/_functorch/partitioners.py:514-615`;
`torch/_functorch/partitioners.py:1920-1995`).

**Required wording:** metadata selects policy; copied ordinary nodes implement runtime recompute.

### A-009 — `activation_memory_budget=0.4` is not an absolute “keep 40% of model activations”

**Old claim:** `0.4` means only 40% of activations are retained.

**Current evidence:** zero maps to the full-region checkpoint/input boundary, one maps to the
runtime-optimized min-cut set, and intermediate budgets are normalized between those two activation
sizes. The implementation may try multiple cut policies and a knapsack solver
(`torch/_functorch/partitioners.py:3446-3525`;
`torch/_functorch/partitioners.py:3527-3631`).

**Required wording:** it is a normalized trade-off within the compiled region, not a percentage of
all model activations or all forward tensors.

### A-010 — `default_partition` and min-cut have different primary jobs

**Old claim:** `default_partition` is described as the general activation-checkpoint decision
engine, or as never recomputing.

**Current evidence:** `default_partition` primarily preserves original forward placement and
saves boundary values. It can still honor explicit checkpoint tags and invokes reorder/RNG work
when recomputation is present (`torch/_functorch/partitioners.py:1595-1671`;
`torch/_functorch/partitioners.py:1769-1829`). The min-cut partitioner is the general
save-versus-recompute optimizer (`torch/_functorch/partitioners.py:4091-4125`).

### A-011 — Decomposition, constant folding, DCE and fusion are not four sequential AOT phases

**Old claim:** after fw/bw compilation, AOT runs a “decomposition and graph optimization phase”
containing decomposition, pattern fusion and constant folding.

**Current evidence:** decompositions are supplied directly to `make_fx` during capture
(`torch/_functorch/_aot_autograd/graph_capture.py:116-137`). AOT performs local DCE during joint
cleanup and partition extraction (`torch/_functorch/_aot_autograd/graph_capture.py:543-575`;
`torch/_functorch/partitioners.py:4091-4157`). Inductor runs its own joint and backend pass
pipelines before and after partition; generic PatternMatcher fusion is not a single fixed AOT stage
(`torch/_inductor/compile_fx.py:2454-2518`).

**Required wording:** document ownership and timing per transformation instead of a fabricated
linear phase.

### A-012 — A custom autograd.Function backward is not categorically untraceable

**Old claim:** the forward can enter the graph, but a custom `autograd.Function.backward` cannot be
traced.

**Current evidence:** joint construction invokes `torch.autograd.grad` while ProxyTensor tracing is
active (`torch/_functorch/_aot_autograd/graph_capture_wrappers.py:388-458`). Metadata explicitly
recognizes Python `BackwardCFunction` and C++ custom autograd results; custom-function views are
included among tangent-producing outputs
(`torch/_functorch/_aot_autograd/collect_metadata_analysis.py:470-497`;
`torch/_functorch/_aot_autograd/collect_metadata_analysis.py:775-805`).

**Required wording:** custom functions have capture and aliasing constraints, but “backward cannot
be traced” is false as a general statement.

### A-013 — View replay is a correctness/performance trade-off, not universally faster

**Old claim:** view replay is always more efficient than `as_strided`.

**Current evidence:** AOT tries a recorded view sequence when safe, then a view function, and
finally falls back to `as_strided`; symbolic inputs and cache serialization constrain replay
(`torch/_functorch/_aot_autograd/functional_utils.py:315-390`;
`torch/_functorch/config.py:135-147`). The source also documents cases where `as_strided` backward
is slow, but that does not establish universal dominance.

### A-014 — Static input indices do not themselves eliminate recompilation

**Old claim:** marking parameters static assumes their shapes never change and avoids recompilation.

**Current evidence:** current frontend documentation describes static indices primarily for
CUDAGraph/static-input handling, while partitioners use them to identify static-lifetime input
nodes and alter save-cost reasoning
(`torch/_functorch/_aot_autograd/frontend_utils.py:251-335`;
`torch/_functorch/partitioners.py:4005-4088`). Shape specialization and recompilation remain
governed by upstream fake/symbolic-shape and guard/cache state.

### A-015 — Subclass and alias support must be stated per path

**Old claim:** a broad assertion is shown as “subclass input does not support output aliasing.”

**Current evidence:** the current autograd path rejects intermediate-base aliasing when subclass
dispatch is required, and export has additional independent restrictions for subclass inputs,
metadata mutation, requires-grad input mutation and functionalized RNG
(`torch/_functorch/aot_autograd.py:637-698`). Dedupe and synthetic-base wrappers have their own
subclass/export restrictions (`torch/_functorch/_aot_autograd/runtime_wrappers.py:1641-1657`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1882-1900`).

**Required wording:** name the exact path and condition; do not generalize one branch's restriction
to all AOTAutograd subclass support.

### A-016 — AOT complexity must separate structural graph work from traced computation

**Old claim:** the whole pipeline can be summarized as linear graph construction plus a few
independent pass costs.

**Current evidence and derived bounds:**

| Operation | Structural bound | Important qualification |
|---|---:|---|
| Metadata analysis | one functionalized forward execution plus metadata scans | Dominated by dispatched fake/symbolic operations and alias/mutation analysis; not just `O(N + E)` |
| Joint capture | traced forward + `autograd.grad` + FX recording | Cost depends on operator decompositions, fake propagation, symbolic reasoning and autograd formulas |
| One graph extraction | `O(N + E)` time, `O(N + E)` new graph space | Assumes aggregate argument traversal accounts for each use edge |
| Node classification | `O(N + E)` plus extraction | Walks users/closures and builds a forward-only graph |
| One min-cut construction | `O(N + E)` graph size | Flow graph has `O(N)` vertices and `O(N + E)` edges |
| One min-cut solve | `T_flow(O(N), O(N+E))` | Do not hard-code a NetworkX algorithm/version-independent polynomial |
| Budget search | multiple cuts + selected knapsack solver | `0 < budget < 1` can invoke increasingly aggressive cuts and DP/greedy/ILP policy |
| Backward reorder | `O(N + E + N log N)` conservative bound | Each node is copied once; prerequisite batches are sorted by original order |
| Runtime ABI packing | `O(A + S + O)` container work | Excludes compiled tensor kernels; `A` args/tangents, `S` saved values, `O` outputs |

The extraction and reorder bounds follow directly from
`torch/_functorch/partitioners.py:514-615` and
`torch/_functorch/partitioners.py:1920-1995`. The flow graph and repeated budget decisions are in
`torch/_functorch/partitioners.py:2641-2765` and
`torch/_functorch/partitioners.py:3446-3631`.

### A-017 — Direct `aot_function` caching is not the same as guarded Dynamo/AOTAutograd caching

**Old claim:** “AOTAutograd cache” is presented as though every entry point performs a uniform
shape-keyed lookup.

**Current evidence:** direct `aot_function` stores its first `(compiled_fn, out_spec)` in a closure
and reuses it (`torch/_functorch/aot_autograd.py:797-842`). Dynamo-integrated compilation has
separate guarded cache and serializable AOTAutograd cache machinery. The new series must name which
entry point owns the cache being discussed.

### A-018 — Saved-tensor hooks are deferred or compiled through explicit mechanisms

**Old claim:** saved tensors are treated as plain direct forward returns without a hook boundary.

**Current evidence:** active saved-tensor hooks are disabled during tracing so ordinary hooks run
at runtime around the generated custom autograd function
(`torch/_functorch/aot_autograd.py:541-552`). A separate compile-time path may inline graph saved
tensor hooks after partition (`torch/_functorch/_aot_autograd/graph_compile.py:2203-2262`).

### A-019 — Current partitioner source path and signatures have drifted

**Old claim:** implementation is under `_aot_autograd/partitioners.py`; examples show obsolete
`num_fwd_outputs_saved_for_bw` and `num_bw_outputs` parameters.

**Current evidence:** current implementation is `torch/_functorch/partitioners.py`, and
`default_partition` accepts `num_fwd_outputs`, optional static-lifetime indices/nodes
(`torch/_functorch/partitioners.py:1595-1602`). Min-cut has its current signature at
`torch/_functorch/partitioners.py:4091-4098`.

### A-020 — Debug backends must not be conflated with standalone defaults

**Old claim:** “min-cut is the default” and “default partition is the default” appear without a
call-site qualifier.

**Current evidence:** standalone `aot_function` defaults to `default_partition`;
`aot_eager` explicitly selects min-cut; Inductor's normal path also selects min-cut after joint
passes (`torch/_functorch/aot_autograd.py:712-738`;
`torch/_dynamo/backends/debugging.py:417-440`;
`torch/_inductor/compile_fx.py:2454-2518`).

**Required wording:** always attach the default to an entry point: standalone AOT API, debug backend
or Inductor integration.

## Migration consequences

The future AOT documents must:

1. start with the reason for metadata analysis and calling-convention wrappers;
2. separate metadata execution, joint capture, partition, backend compilation and runtime assembly;
3. draw fw and bw as independent graphs connected by an ABI box, never by an FX edge;
4. give an exact core argument/output order and label optional channels;
5. explain recompute first as a cut decision, then as ordinary node copying, then as backward
   scheduling/reordering;
6. distinguish standalone `aot_function`, debug backends and Inductor integration;
7. keep the functional-graph contract conditional and expose mutation/effect exceptions;
8. report complexity as a sum of traced computation, structural graph work, flow/knapsack
   optimization and runtime packing.

## Task-5 acceptance result

- Metadata collection and functionalization: source-verified.
- Joint capture and `autograd.grad` construction: source-verified.
- Fresh fw/bw graph extraction and absence of cross-graph FX edges: source-verified.
- Saved-value classes, min-cut construction, recompute copying and backward reorder:
  source-verified.
- Core runtime ABI ordering and context storage: source-verified.
- Historical executable examples: unresolved until run on the pinned environment; none are
  promoted to verified Labs by this audit.

