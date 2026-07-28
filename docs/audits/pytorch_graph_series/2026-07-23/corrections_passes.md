# PyTorch Graph Series — Pattern Matching, DCE, Ordering and Pass Audit

## Audit scope and authority

- Current source authority: PyTorch commit
  `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52` (2026-07-23).
- Detached source root: `E:/97-codes/torch_parallel/p`.
- Audited legacy material:
  - the PatternMatcher, DCE, ordering and complexity portions of
    `fx_graph_construction_and_transformation_analysis.md`;
  - `decomposition_passes_guide.md`;
  - `pre_grad_passes_guide.md`;
  - `joint_graph_passes_guide.md`;
  - `post_grad_passes_guide.md`;
  - `torch_upstream_pass_deepdive.md`;
  - the graph-pass portions of `fx_pass_optimization_methodology.md`.
- Historical pages are audit leads, not source authority. Illustrative code blocks are not
  classified as verified Labs unless they are executed later on the pinned environment.

This report fixes the mental model first. Individual optimization catalogs will be re-derived
from their registration sites when documents 08 and 15 are authored; they are not accepted merely
because an old page listed them.

## The current mental model

### A pattern is a predicate-and-capture AST over an FX graph

`PatternExpr` is not an FX node and a pattern is not a second executable graph. It is a recursive
description of:

1. which FX `op` and `target` a candidate must have;
2. how its nested `args` and `kwargs` must be shaped;
3. which child values must recursively satisfy another pattern;
4. which values are captured for a handler;
5. which user-count, sharing and multi-output relationships must hold.

The base class supplies the common entry point and polymorphic protocol; concrete subclasses exist
because constants, calls, captures, repeated lists and multi-root subgraphs require different
matching behavior (`torch/_inductor/pattern_matcher.py:483-558`;
`torch/_inductor/pattern_matcher.py:745-788`;
`torch/_inductor/pattern_matcher.py:791-1019`;
`torch/_inductor/pattern_matcher.py:1058-1233`).

`Arg` captures an arbitrary `NodeOrConstant` into positional handler arguments in depth-first
pattern traversal order. It does **not** mean “the nth formal parameter of the FX target.”
`KeywordArg("q")` captures an arbitrary value under the handler keyword `q`; the pattern object may
itself occur in either the positional or keyword argument structure of a `CallFunction`.
`Ignored` is an unconstrained wildcard that contributes no handler argument
(`torch/_inductor/pattern_matcher.py:561-583`;
`torch/_inductor/pattern_matcher.py:745-762`).

### A tree syntax can still express DAG sharing

`MatchContext.pattern_to_node` keys by `PatternExpr` object identity. Reusing the same pattern object
twice therefore requires both occurrences to bind to the same FX value. `_TargetExpr` separately
checks the expected number of users, except that an output pattern and `_users=MULTIPLE` relax the
ordinary exact-user constraint (`torch/_inductor/pattern_matcher.py:483-523`;
`torch/_inductor/pattern_matcher.py:841-861`).

This distinction must remain explicit:

- reusing the **same pattern instance** expresses equality/sharing;
- constructing two structurally equal pattern instances does not by itself require the same FX node;
- user-count constraints describe external uses, not just edges internal to the pattern.

Ordinary `PatternExpr.match()` allocates a fresh context, so a failed top-level attempt is discarded
with that context. `MultiOutputPattern` has a narrower rollback mechanism: it snapshots and restores
`pattern_to_node` between alternative anchor candidates. It is inaccurate to generalize this into a
global transactional matcher rollback guarantee (`torch/_inductor/pattern_matcher.py:534-538`;
`torch/_inductor/pattern_matcher.py:1206-1223`).

### MultiOutputPattern means multiple exposed roots, not necessarily a tuple-returning op

The first output must be a `_TargetExpr` and is the root used for candidate indexing. After matching
that root, later outputs are found by walking from already-bound child patterns through FX `users`;
an output slot may also be `None`. Thus the abstraction represents a connected match with multiple
externally visible roots. It is not synonymous with a single node whose runtime result is a tuple
(`torch/_inductor/pattern_matcher.py:1021-1045`;
`torch/_inductor/pattern_matcher.py:1162-1233`).

## Corrections

### P-001 — Source line numbers in the original question are historical

On the pinned source, the principal anchors are:

- `MatchContext`: `torch/_inductor/pattern_matcher.py:483-523`;
- `PatternExpr`: `torch/_inductor/pattern_matcher.py:526-558`;
- `Arg` and `Ignored`: `torch/_inductor/pattern_matcher.py:561-583`;
- `KeywordArg`: `torch/_inductor/pattern_matcher.py:745-762`;
- `CallFunction`: `torch/_inductor/pattern_matcher.py:1058-1064`;
- `MultiOutputPattern`: `torch/_inductor/pattern_matcher.py:1162-1233`;
- `PatternMatcherPass`: `torch/_inductor/pattern_matcher.py:2583-2726`.

The old numbers are useful only when attached to their old commit. The new series will always pair
line references with the pinned SHA.

### P-002 — The subclass design is determined by graph matching dimensions

The subclass family is not inheritance for its own sake. Each branch isolates an independent graph
dimension:

| Graph requirement | Implementation family |
|---|---|
| exact value or target | constant/target expressions |
| `op + target + args + kwargs` | `_TargetArgsExpr`, `CallFunction`, `CallMethod`, `CallModule` |
| unconstrained captured input | `Arg`, `KeywordArg`, `ExclusiveKeywordArg` |
| wildcard without handler ABI | `Ignored` |
| repeated argument collection | `ListOf`, repeated-expression variants |
| multiple exposed roots | `MultiOutputPattern` |

`_TargetArgsExpr` can normalize omitted default kwargs using the operator schema, flattens nested
argument structures, compares the resulting structure, then recursively matches pattern children
and compares constants exactly (`torch/_inductor/pattern_matcher.py:876-1019`).

### P-003 — `Arg` and `KeywordArg` describe handler capture, not FX calling convention

The legacy shorthand “`Arg` 按位置捕获，`KeywordArg("q")` 捕获 q 入参” is incomplete. The position
is the depth-first pattern traversal order, while `q` names the handler keyword. Neither class
asserts that the candidate FX call used a positional or keyword spelling at that location
(`torch/_inductor/pattern_matcher.py:561-568`;
`torch/_inductor/pattern_matcher.py:745-758`;
`torch/_inductor/pattern_matcher.py:996-1019`).

### P-004 — Candidate traversal is not “every graph node × every pattern”

Registration buckets entries by the root `(pattern.op, target)`. `PatternMatcherPass.apply()` asks
the graph lookup table for only the registered root keys, merges those lists, and then tries only
the bucket associated with each candidate. `call_module` is scanned once and filtered by extracted
target (`torch/_inductor/pattern_matcher.py:1342-1370`;
`torch/_inductor/pattern_matcher.py:2583-2656`).

FX maintains a side table for these queries. `call_function` is indexed by `(op, target)`; other
op kinds are indexed by `(op, None)` and optionally filtered by target
(`torch/fx/graph.py:1360-1393`; `torch/fx/graph.py:1497-1522`).

Therefore the pass visits the union of registered-root candidates. It approaches a whole-graph scan
only when registered root keys cover most graph nodes.

### P-005 — “Reverse graph order” is a candidate-processing order

The candidate lists are materialized before any replacement and sorted in reverse `Node` order.
`Node` comparison is backed by a lexicographic sort key that tracks the intrusive graph-list order
(`torch/_inductor/pattern_matcher.py:2627-2645`;
`torch/csrc/fx/node.cpp:405-466`).

This does not construct a reverse graph, add reverse edges, or refer to an AOT backward graph.
It means “process later root candidates before earlier root candidates.” Because the candidate list
is a snapshot, replacement nodes inserted during the pass are not candidates in that same
`apply()` call. That no-revisit property comes from snapshotting, not from reverse order itself.

### P-006 — One `PatternMatcherPass.apply()` call is not a fixed-point engine

The driver makes one candidate snapshot and performs one pass over it. Rules in a candidate's bucket
are tried in registration order; if a rule erases the root, later entries for that root are skipped.
Newly created roots are not discovered again and the pass does not restart after a match
(`torch/_inductor/pattern_matcher.py:2641-2726`).

Repeated application is explicit at a higher layer:

- pre-grad may run a configured matcher `counter` times
  (`torch/_inductor/fx_passes/pre_grad.py:384-402`);
- joint and post-grad have fixed ordered lists of matcher passes
  (`torch/_inductor/fx_passes/joint_graph.py:748-753`;
  `torch/_inductor/fx_passes/post_grad.py:227-267`);
- the generic FX `PassManager` repeats only up to its configured `steps`, whose default is one
  (`torch/fx/passes/infra/pass_manager.py:154-192`;
  `torch/fx/passes/infra/pass_manager.py:254-317`).

These are different mechanisms and must not be collapsed into an implicit global fix point.

### P-007 — Structural matching is followed by stage safety checks

Before applying a successful match, the engine rejects a result that spans mutation regions or
different stream/mempool contexts. It then evaluates `extra_check` under `guard_or_false`.
Graph-pattern handlers are additionally observed for newly created mutation ops so mutation-region
metadata can be recomputed when necessary (`torch/_inductor/pattern_matcher.py:2622-2626`;
`torch/_inductor/pattern_matcher.py:2657-2710`).

The PatternExpr AST therefore describes local structure and capture. It does not, by itself, prove
alias safety, effect ordering, stream legality, dtype/shape legality or numerical equivalence.

### P-008 — Entry types have different mutation ownership

- `LoweringPatternEntry` inserts a handler call before the root, transfers metadata, replaces uses
  and erases now-unused matched nodes (`torch/_inductor/pattern_matcher.py:1373-1385`).
- `GraphPatternEntry` invokes custom graph-surgery code at an insertion point; the handler owns the
  rewrite (`torch/_inductor/pattern_matcher.py:1388-1399`).
- `ReplacementPatternEntry` interprets a traced replacement `GraphModule`, copies its nodes into
  the destination graph, reconnects one or multiple outputs, transfers metadata, erases removable
  matched nodes and performs only a local cleanup of dead replacement nodes
  (`torch/_inductor/pattern_matcher.py:1401-1652`).

`Match.erase_nodes()` is also local: it walks the matched-node list in reverse and erases a node only
when its `users` set is empty (`torch/_inductor/pattern_matcher.py:295-300`).

### P-009 — `register_replacement` uses generic and match-specific traces

Unless a serialized/pre-traced pattern is supplied, registration first traces the search function
using example inputs and constructs the broad search pattern. At match time, `check_fn` rebuilds
arguments from matched fake values, retraces a shape-specific search pattern when necessary,
matches that specific pattern, applies `extra_check`, and only then traces the replacement function
and stores its graph on the match (`torch/_inductor/pattern_matcher.py:1828-1877`;
`torch/_inductor/pattern_matcher.py:1878-2033`;
`torch/_inductor/pattern_matcher.py:2052-2092`).

This two-level design prevents an initial pattern that intentionally ignored scalar types/shapes
from authorizing an invalid replacement. It is not “trace search and replacement once at import,
then blindly paste.”

### P-010 — Pattern replacement does not run a universal cleanup bundle

`PatternMatcherPass.apply()` returns a match count but does not globally invoke DCE, stable
topological sort, `lint()` or `recompile()` (`torch/_inductor/pattern_matcher.py:2609-2726`).
The replacement entry performs only the local cleanup described in P-008.

Cleanup is stage-specific:

- pre-grad always runs its custom hooks, stable sort, quant lifting, lint and recompile after the
  optional built-in matcher block (`torch/_inductor/fx_passes/pre_grad.py:336-433`);
- joint runs stable sort, lint and recompile only when its local `count` is nonzero
  (`torch/_inductor/fx_passes/joint_graph.py:699-772`);
- post-grad runs optional DCE at entry, stable sort after its main pattern/custom-post block,
  may sort again after collective bucketing, then runs late mutation-introducing passes and finally
  recompiles and lints (`torch/_inductor/fx_passes/post_grad.py:165-180`;
  `torch/_inductor/fx_passes/post_grad.py:227-303`;
  `torch/_inductor/fx_passes/post_grad.py:385-474`).

It is therefore wrong to teach “every successful replacement is followed by
DCE + sort + lint + recompile.”

### P-011 — Joint cleanup is conditional, but `count` is not a generic changed bit

The joint driver increments `count` for custom hooks, matcher hits and random replacements. Its
conditional tail is not a general comparison of the graph before and after every preceding pass:
`remove_noop_ops`, constant folding and AutoChunker are invoked without adding their change result
to this counter in the driver (`torch/_inductor/fx_passes/joint_graph.py:711-771`).

The new documents will say exactly which events contribute to the condition instead of calling it
“any graph change.” Individual passes may maintain/recompile their own returned graph, but that is a
separate contract.

### P-012 — “Dead” is a graph-local use/effect predicate

`Graph.eliminate_dead_code()` first calls `lint()` because the algorithm requires a topologically
ordered graph. It then visits nodes in reverse order and erases a node exactly when the selected
impurity predicate says it is pure and `len(node.users) == 0`. Erasing a consumer updates producer
user counts, enabling the same reverse sweep to cascade upstream
(`torch/fx/graph.py:2689-2760`).

This implementation does not first compute reachability from the output, and it knows nothing about
whether a node is “connected” to another FX graph. A backward placeholder is retained because
placeholders are treated as impure for DCE; its runtime value arrives through the fw/bw ABI, not a
cross-Graph `Node` edge.

### P-013 — Current impurity detection is stronger than the legacy `copy_` example

`Node.is_impure()` retains placeholders and outputs, reads `_is_impure` for `call_module`, and
delegates `call_function` to the unified library predicate
(`torch/fx/node.py:760-808`). That predicate treats an `OpOverload` with a mutable schema as impure,
along with registered effects and random-state behavior
(`torch/_library/utils.py:624-708`).

Consequently, the old concrete warning that a normal `aten.copy_` node may simply be classified
pure and deleted is not an accurate description of the pinned implementation: mutable-schema ops
are recognized. The broader warning remains essential because FX itself states that impurity
coverage is incomplete, and `call_module` may be considered pure even when an inner GraphModule
contains impure operations (`torch/fx/graph.py:2725-2732`;
`torch/fx/node.py:776-795`).

### P-014 — DCE recurses into only referenced child GraphModules

After local deletion, DCE collects `get_attr` targets, visits named child modules whose names are
present in that set, recursively runs child-graph DCE, and recompiles each visited child
(`torch/fx/graph.py:2762-2776`).

The complexity and semantics of “DCE on a GraphModule tree” must include those referenced nested
graphs; it is not only the outer graph sweep.

### P-015 — Stable topological sort repairs data-dependency order, not all effects

The Inductor helper scans a node's `args` and `kwargs` for FX-node dependencies. A node with
unresolved inputs waits on the last unresolved dependency; when all are ready it is moved after a
cursor in the intrusive list. A cycle, a foreign dependency or otherwise unresolved state triggers
the final assertion (`torch/_inductor/pattern_matcher.py:2940-2980`).

The source test shows that an already valid order is retained and that an intentionally misplaced
producer is repaired without reordering independent placeholders
(`test/inductor/test_pattern_matcher.py:2201-2225`).

The sorter cannot infer hidden mutation, RNG, collective or stream ordering. Such constraints must
already be represented by data/control dependencies, region checks or stage rules. For example,
when fallback random operators share a global RNG, post-grad explicitly chains them with
`control_deps` before sorting (`torch/_inductor/fx_passes/post_grad.py:115-141`;
`torch/_inductor/fx_passes/control_dependencies.py:1-40`;
`torch/_inductor/fx_passes/control_dependencies.py:134-217`).

### P-016 — “Stable” means relative to the existing sequence under dependency repair

The algorithm is seeded from the original graph sequence and its test contract preserves independent
nodes in that observed order. It should be described as deterministic, dependency-respecting repair
relative to the current list—not as a semantic scheduler or a proof that moving any two independent
FX nodes is effect-safe (`torch/_inductor/pattern_matcher.py:2946-2980`;
`test/inductor/test_pattern_matcher.py:2201-2225`).

### P-017 — The three stage drivers are materially different

The old “one identical driver shape repeated three times” thesis is too strong.

- Pre-grad operates on non-functional, non-normalized IR, has predispatch and non-predispatch
  branches, supports explicit matcher counters, and always performs its final sort/lint/recompile
  path (`torch/_inductor/fx_passes/pre_grad.py:200-299`;
  `torch/_inductor/fx_passes/pre_grad.py:336-433`).
- Joint currently has `early_patterns` plus **two** ordered main matcher passes. Its canonicalizer
  currently calls `canonicalize_quant_mapping`; AutoChunker can return a new GraphModule, and the
  final cleanup is conditional as described in P-011
  (`torch/_inductor/fx_passes/joint_graph.py:44-89`;
  `torch/_inductor/fx_passes/joint_graph.py:691-772`).
- Post-grad has **three** main matcher passes, but continues after its first stable sort through
  backend, collective, overlap and mutation-introducing transforms. It adds current steps absent
  from older catalogs, including current-device rejection, fallback-random ordering, SPMD checking,
  communication decomposition, reduce-scatter deduplication, low-contention collective replacement
  and dtype-view repair (`torch/_inductor/fx_passes/post_grad.py:84-89`;
  `torch/_inductor/fx_passes/post_grad.py:165-324`;
  `torch/_inductor/fx_passes/post_grad.py:326-474`).

The old `reinplace_fsdp_all_gather` entry is not in the pinned `post_grad_passes()` tail. The current
tail is `reinplace_inplaceable_ops`, dtype-view repair, Triton-wrapper decomposition,
auto-functionalized decomposition, scan/map lowering, then recompile and lint
(`torch/_inductor/fx_passes/post_grad.py:449-474`).

### P-018 — Stage pass recursion is part of the real pipeline

The compile driver processes nested graph modules before the enclosing graph. Joint handling also
detects subgraphs created by its own passes and processes those new modules. Post-grad recursively
processes nested graphs before the outer graph (`torch/_inductor/compile_fx.py:564-650`).

Complexity, source navigation and debugging advice must therefore distinguish “one Graph” from the
aggregate graph-module tree.

### P-019 — Decomposition is capture configuration, not a post-grad rewrite pass

Inductor builds its decomposition table from core ATen and Inductor-specific entries, then removes
operators it intentionally wants to preserve. `select_decomp_table()` switches random and selected
fallback behavior (`torch/_inductor/decomposition.py:114-160`;
`torch/_inductor/decomposition.py:1161-1177`).

`compile_fx` wraps an explicitly supplied table in a constant provider; otherwise it uses
`select_decomp_table`, obtains the table for AOTAutograd, and threads the provider through later
compilers (`torch/_inductor/compile_fx.py:2890-2915`;
`torch/_inductor/compile_fx.py:3109-3120`). AOT graph capture passes that table to `make_fx`
(`torch/_functorch/_aot_autograd/graph_capture.py:120-139`).

Thus the legacy decomposition guide's central placement is correct. Its runnable snippets remain
unverified until executed on the pinned environment.

### P-020 — The “all Inductor rewrites use PatternMatcher” thesis loses mechanisms

PatternMatcher is important, but current stage drivers also use direct graph traversals, module-level
transforms, DCE, locality reorder, group/batch fusion search, communication bucketing, control
dependencies, reinplacing and higher-order-op decomposition. `GraphTransformObserver` observes and
times a pass; it does not turn every pass into a PatternMatcher rule
(`torch/fx/passes/graph_transform_observer.py:22-118`;
`torch/_inductor/fx_passes/group_batch_fusion.py:1594-1646`;
`torch/_inductor/fx_passes/post_grad.py:165-474`).

The series will use “a common PatternMatcher used by many transformations inside a heterogeneous
pass pipeline,” not “almost all graph rewrites are one engine.”

### P-021 — Legacy pass pseudocode is not an executable specification

Several stage guides turn approximate intent into unconditional rewrite examples. A concrete
counterexample is the Joint guide's statement that an AMP chain
`float32 -> float16 -> float32` is replaced by the original float32 input. The current
`pointless_convert` explicitly preserves a narrower intermediate before an upcast, because the
narrowing can lose information (`torch/_inductor/fx_passes/joint_graph.py:829-864`).

Similarly, optimization-round labels such as “pass 0 is simple, pass 1 is online softmax, pass 2 is
complex GEMM” are informal catalog summaries, not contracts enforced by `PatternMatcherPass`.
The executable facts are the ordered pass registries and each rule's registration site
(`torch/_inductor/fx_passes/joint_graph.py:44-57`;
`torch/_inductor/fx_passes/post_grad.py:84-89`;
`torch/_inductor/fx_passes/post_grad.py:850-860`).

All such examples will be reclassified as one of:

- **Source pseudocode**: directly faithful to a pinned implementation;
- **Illustration**: explanatory but not a claim about an actual registered rule;
- **Lab**: executed with inputs, environment and observed output.

Until then they remain unresolved and cannot be migrated as verified behavior.

### P-022 — Serialized-pattern concepts survive, but inventories must be regenerated

The legacy `fwd_only`/`joint_fwd_bwd` distinction and precompiled-pattern motivation remain useful,
but a fixed generated-pattern inventory is not a stable API. On the pinned source,
`_serialize_pattern` writes generated modules, `_known_precompiled_patterns` tracks registered
artifacts, and the import path is selected by the generation environment
(`torch/_inductor/pattern_matcher.py:2095-2224`). The course retains the mechanism and generation
contract, while requiring the current generator and tests for any inventory or cleanup count.

## Complexity model

For one graph, define:

- `N`: number of FX nodes;
- `E`: total FX-node references in all `args` and `kwargs`;
- `C`: number of root candidates materialized for one matcher pass, with `C <= N`;
- `B(v)`: number of registered entries in candidate `v`'s root bucket;
- `K(p)`: size of pattern `p`;
- `A(p,v)`: extra anchor/user exploration for multi-output or repeated patterns;
- `R`: replacement nodes inserted;
- `U`: uses rewritten by a replacement;
- `d(v)`: number of input references of node `v`.

### PatternMatcherPass

```text
T_apply
  = O(C)                         candidate materialization
  + O(C log C)                   reverse order sort
  + Σv Σp∈B(v) O(K(p)+A(p,v))   structural matching
  + Σmatches rewrite_cost
```

For traced replacement, a successful match can additionally pay match-specific search tracing and
replacement tracing. A structural replacement is at least `O(R + U + E_match)` plus any custom
handler work.

The common near-linear case requires small root buckets, bounded pattern size/arity and bounded
anchor fan-out. Worst cases arise when many rules share a root, fail late, or repeatedly scan
high-fan-out anchors. An explicit `T`-round driver multiplies the applicable discovery, match and
rewrite terms by up to `T`.

### DCE

For a single graph, lint plus the reverse deletion cascade is `O(N + E)` under ordinary constant-time
operator/target checks. Across referenced nested GraphModules, use the sum of `N` and `E` over all
visited graphs. The source's impurity checks and target resolution are semantic costs, not extra
graph traversals (`torch/fx/graph.py:2609-2687`; `torch/fx/graph.py:2689-2776`).

### Stable topological sort

Each successful placement and dependency wake-up is linear in aggregate, but a waiting node rescans
all its arguments each time another selected dependency wakes it. Common bounded-arity graphs are
close to `O(N + E)`. A conservative implementation-level bound is:

```text
O(N + E + Σv d(v)^2)
```

with `O(N + E)` auxiliary storage. This bound is more accurate than promising unconditional linear
time for arbitrarily high-arity nodes (`torch/_inductor/pattern_matcher.py:2940-2980`).

### Full pass stage and nested graphs

For ordered stage passes `q=1..Q` over graph modules `g`, the graph-algorithm cost is additive:

```text
Σg Σq (
    candidate_sort_q,g
  + match_q,g
  + rewrite_q,g
  + optional_cleanup_q,g
)
```

It excludes real tensor execution during metadata/fake propagation, tracing, max-flow partitioning,
lowering, backend compilation and autotuning. Those costs belong to other documents in the series.

## Migration rules for the new series

1. Document 13 will define the PatternExpr AST, MatchContext sharing, candidate index, entry types,
   replacement tracing and the exact one-pass behavior.
2. Document 14 will define use-def liveness, impurity, nested DCE, stable topology and explicit
   effect-order mechanisms.
3. Document 15 will derive the current pre/joint/post pipelines from the pinned source and will
   distinguish ordered rounds from fixed points.
4. Document 16 will combine rewrite legality, alias/effect/shape constraints, verification and the
   complexity model.
5. Source snippets may migrate as “Source” evidence. Illustrative or runnable examples remain
   “Unverified” until a Lab is executed and its environment and output are recorded.
