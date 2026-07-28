# PyTorch Graph Series — Inductor Lowering, IR, Scheduler, Memory and Codegen Audit

## Audit scope and authority

- Current source authority: PyTorch commit
  `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52` (2026-07-23).
- Detached source root: `E:/97-codes/torch_parallel/p`.
- Audited legacy material:
  - `lowering_analysis.md`;
  - `scheduler_analysis.md`;
  - `inductor_codegen_analysis.md`;
  - `inductor_autotuning_analysis.md`;
  - the Inductor portions of `PyTorch_Inductor_Technical_Analysis.md`;
  - `torch_compile_architecture.md`.
- Dynamic-shape codegen was already classified in the foundations audit. This report adds the
  lowering/IR/Scheduler/codegen boundary needed by documents 17–21.
- Historical pages are audit leads, not source authority. Their internal-API examples were not
  executed on the pinned environment and therefore are not verified Labs.

## The current end-to-end mental model

### GraphLowering interprets FX; it does not clone FX into an isomorphic IR graph

`GraphLowering` subclasses `torch.fx.Interpreter`. For each FX node, the interpreter fetches
already-lowered arguments from its environment, runs the applicable lowering, and stores the
returned Python/Inductor object as that FX node's environment value
(`torch/_inductor/graph.py:386-386`; `torch/_inductor/graph.py:1925-2000`).

The result is deliberately not one IR operation per FX node:

- pointwise and reduction expressions may remain lazy inside `TensorBox(StorageBox(...))`;
- views may wrap or reinterpret storage without creating a schedulable operation;
- one lowering may create multiple buffers/operations;
- several FX operations may be composed into one lazy loop before realization;
- template, extern and multi-output paths have different IR shapes.

`StorageBox.realize()` is the important boundary for lazy loop IR: a lazy
`Pointwise`/`Reduction`/`Scan`/`Sort` becomes a `ComputedBuffer` with a `FlexibleLayout`, is
registered as a buffer and operation, and receives a name
(`torch/_inductor/ir.py:10547-10607`). The Scheduler is constructed from registered
`V.graph.operations`, so it schedules realized operations rather than a copy of all FX nodes
(`torch/_inductor/graph.py:1130-1166`; `torch/_inductor/scheduler.py:4191-4213`).

### Inductor IR separates values, storage, layouts, computations and operations

The live class families on the pinned source include:

| Dimension | Representative current classes |
|---|---|
| lazy tensor value/storage | `TensorBox`, `StorageBox` |
| loop computation | `Loops`, `Pointwise`, `Reduction` |
| view/addressing | `BaseView`, `ReinterpretView` |
| output/storage description | `OutputSpec`, `Layout`, `FixedLayout`, `FlexibleLayout`, `NoneLayout` |
| named storage/operation | `Buffer`, `Operation`, `OperationBuffer`, `ComputedBuffer` |
| template/extern implementation | `TemplateBuffer`, `InputsKernel`, `ExternKernel`, `FallbackKernel` |
| multiple external results | `MultiOutputLayout`, `MultiOutput` |
| algorithm choice | `ChoiceCaller`, `MultiTemplateBuffer` |

The relevant declarations are at `torch/_inductor/ir.py:589-655`,
`torch/_inductor/ir.py:956-1061`, `torch/_inductor/ir.py:1219-1420`,
`torch/_inductor/ir.py:3386-4090`, `torch/_inductor/ir.py:4399-4755`,
`torch/_inductor/ir.py:5040-5435`, `torch/_inductor/ir.py:5876-6315`,
`torch/_inductor/ir.py:6645-7035`, `torch/_inductor/ir.py:9314-9355` and
`torch/_inductor/ir.py:10144-10190`.

This class split exists because these are independent questions:

1. What value expression is being computed?
2. Does it own storage or alias another value?
3. What size/stride/offset/layout does the storage have?
4. Is the computation lazy, realized, template-backed or external?
5. Is it schedulable, and what buffers does it read/write?

Collapsing those questions into a single “IR node” class would make view semantics, lazy
realization, mutation, backend choices and scheduling dependencies inseparable.

### Scheduler builds a buffer dependency graph, not an FX-user graph

Scheduler nodes are created from realized operations:

- no-op operation → `NopKernelSchedulerNode`;
- `ComputedBuffer` or `TemplateBuffer` → `SchedulerNode`;
- `ExternKernel` → `ExternKernelSchedulerNode`
  (`torch/_inductor/scheduler.py:4641-4651`).

`compute_dependencies()` constructs name-based use/dependency state while accounting for aliases,
mutation renames, weak ordering, output/input mutation and unbacked-symbol origins. It is not a
copy of `fx.Node.users` (`torch/_inductor/scheduler.py:4689-5053`). The topological scheduler then
orders nodes by `unmet_dependencies`, which are buffer-name dependencies
(`torch/_inductor/scheduler.py:5104-5129`).

The current constructor pipeline includes communication ordering, dependency construction,
topological sorting, Scheduler DCE, ancestor computation, foreach grouping, custom passes,
stream/mempool assignment, iterative fusion, optional second DCE, loop merging, multi-template
finalization, combo kernels, switch ordering, optional peak-memory/communication reorders, grouped
node processing, optional graph-partition reorder and last-use computation
(`torch/_inductor/scheduler.py:4235-4410`).

### “Memory planning” names three different mechanisms

#### 1. Scheduler peak-memory reorder

`torch/_inductor/memory.py` estimates buffer lifetimes and peak live bytes, evaluates the original
topological order plus LPMF/BFS/DFS legal orders, and selects the estimated lowest-peak order
(`torch/_inductor/memory.py:486-567`; `torch/_inductor/memory.py:1016-1105`). This changes execution
order; it does not emit allocations or perform physical storage reuse.

#### 2. Default wrapper allocation/free/reuse planning

Scheduler last-use analysis marks buffers by reverse traversal, and codegen emits frees for
freeable buffers (`torch/_inductor/scheduler.py:8959-8995`). Wrapper IR contains
`AllocateLine`, `FreeIfNotReusedLine` and `ReuseLine`; its reuse key includes device, dtype,
symbolic allocation size, alignment, stream and mempool
(`torch/_inductor/codegen/wrapper.py:108-138`;
`torch/_inductor/codegen/wrapper.py:963-1024`;
`torch/_inductor/codegen/wrapper.py:1076-1188`).

The ordinary two-pass `memory_plan_reuse()` runs whenever the optional pooled planner is not
selected. `allow_buffer_reuse` controls actual unrelated-buffer reuse, but the wrapper planning
pass still performs its cleanup and planning traversal
(`torch/_inductor/codegen/wrapper.py:2531-2582`).

#### 3. Optional pooled static memory planner

When `is_inference and config.memory_planning`, wrapper code invokes
`codegen/memory_planning.py::MemoryPlanner`; training currently falls back to the ordinary wrapper
reuse path because pooled planning can increase training peak memory
(`torch/_inductor/codegen/wrapper.py:2526-2582`).

The pooled planner performs five passes: remove dead lines, convert pool lines, compute live
ranges, allocate groups and mark first/last use
(`torch/_inductor/codegen/memory_planning.py:665-683`). Its allocator is a greedy
temporal/spatial allocation tree, not an exact interval-graph coloring solver
(`torch/_inductor/codegen/memory_planning.py:35-397`;
`torch/_inductor/codegen/memory_planning.py:497-549`;
`torch/_inductor/codegen/memory_planning.py:777-817`).

The current `memory_pool` modes mean:

- `none`: no expandable pooled storage, but compatible live ranges can still be reused;
- `intermediates`: non-outputs share storage; outputs receive unique storage;
- `outputs`: separate pools for intermediates and outputs;
- `combined`: one pool may contain intermediates and outputs
  (`torch/_inductor/config.py:248-268`).

### Code generation has a kernel side and a wrapper side

`DeviceCodegen` registers a scheduling constructor plus Python/C++/FX wrapper constructors.
Kernel generation is owned by the device scheduling implementation; host-side allocation,
launch and returns are owned by the wrapper
(`torch/_inductor/codegen/common.py:309-318`;
`torch/_inductor/codegen/common.py:389-434`).

Current built-in dispatch is not simply “CUDA/XPU use `TritonScheduling`”:

- CPU selects C++/Halide/Triton/Pallas from config;
- CUDA's Triton choice uses `CUDACombinedScheduling`;
- XPU uses `XPUCombinedScheduling`;
- MPS uses `MetalScheduling`;
- MTIA uses `TritonScheduling`
  (`torch/_inductor/codegen/common.py:515-613`).

`GraphLowering.codegen()` initializes the wrapper, constructs/updates the Scheduler, asks the
Scheduler to generate backend and wrapper lines, then finalizes the wrapper. Module compilation is
a subsequent step (`torch/_inductor/graph.py:2991-3008`;
`torch/_inductor/graph.py:3050-3067`). Therefore “codegen” is not, by itself, identical to
“machine code is already compiled and executed.”

### Algorithm selection and Triton launch autotuning are two layers

`AlgorithmSelectorCache` selects among implementation choices for GEMM/convolution-like operations,
including extern and template callers. It owns persistent choice caching, precompilation and
benchmarking; selection can be deferred via `MultiTemplateBuffer` so Scheduler fusion can affect
the choice (`torch/_inductor/select_algorithm.py:3949-4035`;
`torch/_inductor/select_algorithm.py:4190-4255`).

`CachingAutotuner` is a different layer attached to one generated Triton kernel. It precompiles
launch configurations; on execution it benchmarks multiple launchers down to one, optionally
performs coordinate-descent tuning, then caches the steady-state launcher
(`torch/_inductor/runtime/triton_heuristics.py:531-803`;
`torch/_inductor/runtime/triton_heuristics.py:1789-1859`;
`torch/_inductor/runtime/triton_heuristics.py:2412-2525`).

An accurate document must say which level is being discussed. “Inductor autotuning” is not one
single candidate list or one single cache.

### Provenance is propagated, aggregated and re-entered

While lowering an FX node, `GraphLowering.run_node()` combines that node with origins gathered from
its lowered arguments and installs the origin set as an `IRNode` construction context
(`torch/_inductor/graph.py:1960-1992`). `IRNode` stores origins, origin node, traceback, stream and
mempool metadata (`torch/_inductor/ir.py:589-645`). A fused Scheduler node can therefore aggregate
origins from several IR operations; during wrapper generation Scheduler chooses the latest
origin by FX graph order and enters that source context
(`torch/_inductor/scheduler.py:9053-9069`).

This supports a many-to-many provenance model:

```text
Python source → FX node(s) → lazy/realized IR object(s)
              → Scheduler node/group → generated kernel/wrapper lines
```

It is not safe to assume stable node identity or a one-to-one mapping across stages.

## Corrections

### I-001 — Lowering is interpretation plus lazy realization, not mechanical node translation

Legacy diagrams that show every FX node producing one durable IR node are pedagogical only.
`GraphLowering` interprets nodes and environment values can be constants, lists/tuples, views,
lazy boxes, realized buffers or extern/template objects. Documents 17 and 18 must make the
realization boundary explicit (`torch/_inductor/graph.py:1925-2000`;
`torch/_inductor/ir.py:10547-10607`).

### I-002 — A missing lowering does not guarantee a successful eager fallback

The old statement that fallback “ensures compilation never fails” is false. A missing target may:

- become an allow-listed fallback;
- become an implicit fallback only when `config.implicit_fallbacks` permits it;
- raise `MissingOperatorWithDecomp`;
- raise `MissingOperatorWithoutDecomp`.

Layout constraints and backward-specific contiguous constraints can also be applied before fallback
(`torch/_inductor/graph.py:1413-1473`). Explicit per-node fallback metadata, user lowerings with a
recursion guard and normal lowerings form a separate priority path
(`torch/_inductor/graph.py:1517-1546`).

### I-003 — `register_lowering` normalizes arguments; it does not prove global optimization

The registration wrapper can transform arguments for broadcasting/type promotion, invoke a
decomposition function and validate returned IR
(`torch/_inductor/lowering.py:481-553`). Fusion opportunity comes from the lazy/loop representation;
global fusion choice belongs to Scheduler. “Every lowering has global graph visibility” is an
incorrect ownership model.

### I-004 — Lowering-pattern passthrough is a distinct call path

Targets carrying `_inductor_lowering_function` are directly invoked before the ordinary lowering
table path (`torch/_inductor/graph.py:1406-1411`). This is how a post-grad pattern can install a
lowering-time callable; it does not turn `PatternExpr` itself into Inductor IR.

### I-005 — Fallback is represented by an ExternKernel family

`fallback_handler()` produces `FallbackKernel.create(...)`; `FallbackKernel` is an
`ExternKernelAlloc`, under the `InputsKernel`/`OperationBuffer` family
(`torch/_inductor/lowering.py:2714-2745`;
`torch/_inductor/ir.py:6645-7035`;
`torch/_inductor/ir.py:9314-9355`). “Fallback” is therefore a scheduled external operation with
materialization/layout constraints, not a transparent escape from all compiler semantics.

### I-006 — View operations are not universally “free”

Views can often remain as index/layout transformations, but downstream fixed-stride requirements,
realization, alias/mutation boundaries or copy-producing view-like operations can require a
materialized buffer. The new series must say “may be zero-copy when layout/alias constraints permit,”
not “view never copies” (`torch/_inductor/ir.py:3386-4090`;
`torch/_inductor/graph.py:1478-1515`).

### I-007 — The old four-class lowering output taxonomy is incomplete

Pointwise, reduction, view and extern cover common paths but omit template buffers, multi-output
objects, scans/sorts, mutation/no-op layouts, choice callers and structured Python values. New
documents will organize classes by value/storage/layout/operation/backend responsibility rather
than assert an exhaustive four-item list.

### I-008 — Foreach grouping is a Scheduler transformation

Lowerings register operation lists, but `Scheduler.create_foreach_nodes()` replaces eligible
operation groups with `ForeachKernelSchedulerNode` after initial DCE/ancestor computation
(`torch/_inductor/graph.py:1165-1172`;
`torch/_inductor/scheduler.py:4255-4257`;
`torch/_inductor/scheduler.py:4653-4687`). Calling this a lowering-time horizontal fusion obscures
the stage boundary.

### I-009 — The Scheduler's dead-node definition is operation- and side-effect-aware

Scheduler DCE reverse-walks a topological schedule. A buffer is inactive when every user is weak or
already removed; an operation is removed only when it has no active output and
`has_side_effects()` is false. It then removes that operation from the read buffers' user lists and
prunes obsolete weak dependencies (`torch/_inductor/scheduler.py:5055-5098`).

This is not FX `Graph.eliminate_dead_code()`, and it is not based on the precomputed `ancestors`
field. “Dead FX node,” “dead realized operation,” “unsaved AOT value” and “unmaterialized lazy
expression” are different predicates.

### I-010 — Dependency order is reconstructed from buffers, aliases and mutations

Scheduler does not add backward edges to the FX graph. It maintains a separate set of Scheduler
objects and dependency/user structures keyed by buffer names. Alias lists can share/merge user
lists, mutation versions are renamed, and special dependencies can be introduced for ordering
(`torch/_inductor/scheduler.py:4689-5053`). This distinction belongs in documents 19 and 20.

### I-011 — `WeakDep` affects ordering semantics even though it is weak for lifetime/DCE

A weak dependency must not be described as “ignored by fusion” or “not a dependency.” It can
constrain scheduling while being prunable when its purpose disappears. Its precise behavior must
be explained at each consumer: dependency construction, DCE, fusion legality and lifetime
estimation.

### I-012 — `StarDep` is not a generic label for every mutation

`StarDep` is a coarse, non-index-specific dependency used by particular mutation/user-defined
kernel paths. Alias and mutation ordering also use rename maps, users and other dependency forms.
The legacy shorthand “StarDep = global mutation edge” is too broad.

### I-013 — Fusion is bounded iterative candidate selection, not one all-pairs pass

`fuse_nodes()` performs at most ten normal rounds and stops on no progress or one remaining node;
it may add one reorder round
(`torch/_inductor/scheduler.py:5268-5301`). Per round, candidates are grouped by used buffer names;
within a grouping each node is compared with a bounded forward window. Aggressive fusion adds
group-based candidates, then candidates are prioritized and scored
(`torch/_inductor/scheduler.py:6830-6884`).

The bound is per node/grouping window, not a global cap on candidate pairs.

### I-014 — Fusion has separate legality, priority and optional benchmark decisions

`can_fuse()` checks legality and rolls back speculative loop mutations on failure. Current legality
also prevents crossing stream and mempool boundaries
(`torch/_inductor/scheduler.py:7818-7884`). `score_fusion_memory()` estimates saved/shared memory
access and handles exact dependencies, same-buffer overlap and mix-order reductions; it is not
merely `len(reads₁ ∩ reads₂)`
(`torch/_inductor/scheduler.py:8631-8675`). Template paths can additionally benchmark profitability
before committing a fusion (`torch/_inductor/scheduler.py:6395-6462`).

### I-015 — Fusion does not guarantee a specific hardware residence

Vertical fusion removes a materialization boundary when the selected backend emits a combined
kernel, but “the intermediate is always kept in registers/shared memory” is an implementation
outcome, not a Scheduler invariant. Horizontal fusion similarly does not guarantee one physical
load of a shared input. Generated indexing, cache behavior, register pressure and backend codegen
determine the final machine behavior.

### I-016 — Reduction and extern nodes are not categorical no-fusion boundaries

The current Scheduler contains mix-order/nested-reduction paths, template prologue/epilogue
fusion, user-defined Triton handling and `FusedExternTritonKernelSchedulerNode`. Therefore old
statements such as “reduction cannot fuse with pointwise” or “ExternKernelSchedulerNode never
fuses” are false as universal rules (`torch/_inductor/scheduler.py:4286-4300`;
`torch/_inductor/scheduler.py:7886-7903`;
`torch/_inductor/scheduler.py:8631-8675`).

### I-017 — Current fusion configuration names/defaults must be source-derived

On the pinned OSS source:

- `score_fusion_memory_threshold = 10`;
- `max_fusion_size = 64`;
- `max_fusion_buffer_group_pairwise_attempts = 64`;
- prologue fusion is selected by `prologue_fusion_enabled()`, not a universal hard-coded false
  (`torch/_inductor/config.py:71-82`; `torch/_inductor/config.py:988-1015`).

The old `config.max_fused_size` name is invalid. Custom partition operation names must use the
expected qualified operator spelling; legacy illustrative snippets are not verified APIs.

### I-018 — Scheduler re-sorts after fusion; there is no universal post-pass reorder bundle

Each fusion round sorts by `min_order` and topologically sorts the result
(`torch/_inductor/scheduler.py:6460-6462`). Later optional peak-memory and communication passes may
choose another legal order, and `compute_last_usage()` runs after those constructor-stage reorders
(`torch/_inductor/scheduler.py:4318-4405`). This is Scheduler-local ordering, not an automatic
FX stable-topological-sort after every PatternMatcher replacement.

### I-019 — Peak-memory reorder is different from wrapper buffer reuse

The legacy Scheduler/codegen pages merge these. Scheduler reorder selects among legal execution
orders using estimated peaks (`torch/_inductor/memory.py:1016-1105`). Wrapper reuse maps compatible
free storage to a later allocation (`torch/_inductor/codegen/wrapper.py:963-1015`). Either can be
enabled or effective without implying the other.

### I-020 — `config.memory_planning` gates only the optional pooled inference planner

It does not turn all liveness/free/reuse logic on or off. Inference plus the flag selects
`MemoryPlanner`; otherwise wrapper `memory_plan_reuse()` is used, including training
(`torch/_inductor/codegen/wrapper.py:2526-2582`). The old claim that training uses the pooled
planner is false on this baseline.

### I-021 — The pooled planner is not exact interval graph coloring

It creates live ranges and greedily places allocation groups into temporal/spatial split trees.
Mode-specific pool expansion and output separation affect placement
(`torch/_inductor/codegen/memory_planning.py:35-397`;
`torch/_inductor/codegen/memory_planning.py:497-549`;
`torch/_inductor/codegen/memory_planning.py:777-817`). New material must not claim optimal coloring
or a globally minimal allocation.

### I-022 — Generated wrapper code is still an execution layer

Inductor reduces eager per-operator Python/dispatcher overhead by generating larger kernels and a
compiled call path, but the default path still generates a Python wrapper. “No Python intermediate
layer” is false as a general JIT statement
(`torch/_inductor/codegen/common.py:389-405`;
`torch/_inductor/graph.py:2991-3008`).

### I-023 — Backend support is a scheduling-plus-wrapper contract

Registering only a kernel printer is not enough. `register_backend_for_device()` records scheduling
and wrapper constructors, with optional custom graph pass and config
(`torch/_inductor/codegen/common.py:410-434`). The large legacy “new hardware backend” example uses
unverified/non-current method names and constructor signatures; it must be rewritten from a small
current testable backend skeleton rather than migrated.

### I-024 — The CUDA/XPU backend descriptions in the old overview are stale

CUDA's Triton choice is currently `CUDACombinedScheduling`, and XPU is
`XPUCombinedScheduling`; both delegate across more than one codegen path
(`torch/_inductor/codegen/common.py:525-585`). A diagram that labels both as plain
`TritonScheduling` loses an important dispatch layer.

### I-025 — “Autotuning” must name its selection level

The old autotuning page describes only `CachingAutotuner`, while the technical analysis mixes
algorithm choice, tile candidates and coordinate descent into one process. Document 21 must
separate:

1. `AlgorithmSelectorCache`: choose an extern/template algorithm for an operation, possibly defer
   through `MultiTemplateBuffer`;
2. `CachingAutotuner`: choose the launch configuration for one generated Triton kernel.

The two layers have different candidates, timing points and cache keys
(`torch/_inductor/select_algorithm.py:3949-4035`;
`torch/_inductor/select_algorithm.py:4190-4255`;
`torch/_inductor/runtime/triton_heuristics.py:1789-1859`;
`torch/_inductor/runtime/triton_heuristics.py:2473-2519`).

### I-026 — Fixed universal Triton candidate lists are stale

Current pointwise/reduction configuration generation delegates into backend heuristic registries,
and operation templates have their own configuration logic. A historical list of XBLOCK,
BLOCK_M/N/K, stages and warps may describe one path/version but is not the universal Inductor
search space. Such lists remain `unresolved` unless tied to a specific current generator and
backend.

### I-027 — The old fusion “cost formula” is invented

The technical page's pseudocode
`memory_saved > threshold && register_pressure < limit` is not the generic current Scheduler
decision. The current path uses candidate grouping, legality checks, `V.choices` scoring,
memory-overlap heuristics and selected benchmark paths
(`torch/_inductor/scheduler.py:6395-6462`;
`torch/_inductor/scheduler.py:6830-6884`;
`torch/_inductor/scheduler.py:7818-7905`;
`torch/_inductor/scheduler.py:8610-8675`).

Register pressure may matter inside generated-kernel compilation/autotuning, but it must not be
presented as the universal Scheduler fusion equation.

### I-028 — The legacy custom fusion tutorial is not a verified current implementation

The large example omits required registration context, constructs template choices incorrectly,
assumes incorrect multi-output/native-layer-norm pattern semantics and uses unverified debug/API
spelling. The current upstream `mm_plus_mm` path is a useful source-backed replacement:

- post-grad pattern and legality check:
  `torch/_inductor/fx_passes/post_grad.py:955-993`;
- tuned lowering:
  `torch/_inductor/kernel/mm_plus_mm.py:128-178`;
- current extern/template choice construction:
  `torch/_inductor/kernel/mm_plus_mm.py:29-177`.

The old example will be classified `unresolved`/rewrite and will not be copied into the new series.

### I-029 — Inductor's common case is loop-level IR, but not every operation becomes a loop

The overview's “Inductor IR no longer records calls, only index lambdas” is too absolute.
Pointwise/reduction paths are loop/index based, while extern kernels, templates, fallback kernels,
collectives and higher-order/control operations retain different operation representations.
Documents 17–18 should say that loop IR is the fusion-friendly core, not the only live IR form
(`torch/_inductor/ir.py:1057-1420`; `torch/_inductor/ir.py:5876-6315`;
`torch/_inductor/ir.py:6645-7035`).

### I-030 — Fusion is central but not Inductor's only optimization objective

The overview's bandwidth motivation is useful, but compute-bound templates, layout constraints,
algorithm selection, launch overhead, communication overlap, memory peak and backend-specific
codegen are independent concerns. The new overview will preserve fusion as a major mechanism
without reducing the whole backend to “put all intermediates in registers.”

### I-031 — Debug/source-navigation lists require current artifacts and pinned locators

The Scheduler page's observability checklist and file index are useful search prompts, but a flag,
log name or old line number is not proof that a current run emitted the claimed artifact.
`Scheduler` owns its current debug renderings and `GraphLowering.codegen()` invokes the current
debug/output-code hooks (`torch/_inductor/scheduler.py:1215-1281`;
`torch/_inductor/graph.py:2991-3135`). The numbered course keeps the debugging workflow, refreshes
locators to the pinned SHA and labels each artifact by whether the Lab actually produced it.

### I-032 — An illustrative compile journey cannot promise one kernel split or autotune path

The legacy codegen/source-map and architecture examples correctly motivate the FX → lowering →
Scheduler → wrapper path, but they over-specialize the eventual kernel split. Current codegen
constructs a Scheduler and wrapper, then dispatches according to operation/backend choices; cache,
fallback, template and extern paths can change the artifact shape
(`torch/_inductor/graph.py:2449-2606`; `torch/_inductor/graph.py:2991-3135`).
The replacement text therefore reports only the split observed by the pinned Lab and does not
generalize it to every `mm` or epilogue.

## Complexity model for documents 17–21

Let:

- `N_fx` be the number of FX nodes received by `GraphLowering`;
- `V` be realized Scheduler operations/nodes;
- `E` be explicit Scheduler dependency/user incidences after construction;
- `A` be alias/mutation bookkeeping incidences;
- `C_r` be fusion candidate pairs in round `r`;
- `R ≤ 10` be normal fusion rounds, plus at most one reorder round;
- `B` be planned buffers/wrapper memory lines;
- `K` be algorithm or launch configurations benchmarked.

The following are implementation-aware bounds, not performance promises:

| Stage | Typical structural cost | Important worst-case source |
|---|---|---|
| GraphLowering interpreter | `O(N_fx + nested argument traversal + lowering work)` | a lowering may create/realize multiple IR objects; no `N_fx = V` assumption |
| dependency construction | commonly near `O(V + E + A)` | alias-list merging scans the current name map, and mutation/user expansion can make the path superlinear; conservative worst case is at least `O(V² + E + A)` |
| Scheduler DCE | `O(V + E)` after user lists exist | reverse visit plus removal from producer user lists |
| topological sort | `O(V + E + Σ d log d)` | dependencies are sorted by name before DFS |
| ancestor materialization | can require `Θ(V²)` space/time on dense/transitive DAGs | every node stores a transitive ancestor set |
| fusion candidate generation | bounded windows are roughly `O(Σ group_size × window)` before dedup | repeated buffer group membership and aggressive grouping can approach `O(V²)` |
| fusion rounds | approximately `Σ_r(candidate build + candidate legality/score/benchmark)` | cycle/legality checks can traverse fused ancestry; conservative expression is `O(Σ_r C_r(V+E) + C_r log C_r)` excluding kernel benchmarks |
| peak-memory reorder | multiple topological heuristics plus repeated peak estimates | current LPMF selection can be quadratic in `V`; overall worst case is at least `O(V² + E)` |
| ordinary wrapper reuse | dictionary reuse is near-linear in emitted lines | current allocation/free construction obtains Scheduler positions with list lookup, so construction can become `O(BV)` |
| pooled planner | greedy sort and allocation-tree search | overlapping live ranges and tree placement are superlinear in the worst case; it is not an optimal-polynomial coloring claim |
| autotuning | structural selection is `O(K)` | wall time is dominated by candidate compilation plus repeated benchmarking; persistent caches change steady-state cost |

Source anchors for these bounds are `torch/_inductor/scheduler.py:4689-5053`,
`torch/_inductor/scheduler.py:5055-5129`, `torch/_inductor/scheduler.py:5268-5301`,
`torch/_inductor/scheduler.py:6395-6462`, `torch/_inductor/scheduler.py:6830-6925`,
`torch/_inductor/memory.py:1016-1105`, `torch/_inductor/codegen/wrapper.py:963-1086` and
`torch/_inductor/codegen/memory_planning.py:665-817`.

## Migration rules

- Documents 17–21 may reuse the high-level motivation and class names only after applying the
  corrections above.
- No internal-API code block in the audited legacy pages is a verified Lab.
- Backend-specific claims must name the backend and pinned commit.
- Performance explanations must distinguish a semantic invariant from a likely generated-machine
  outcome.
- Any unresolved CUDA Graph, AOTInductor packaging, NPU/out-of-tree backend or fixed heuristic-list
  claim requires a separate source audit before publication.
- New pages will cite the pinned commit with current locators; old line numbers will be preserved
  only as historical evidence.
