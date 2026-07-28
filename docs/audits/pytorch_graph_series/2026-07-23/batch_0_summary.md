# PyTorch Graph Learning Series — Batch 0 Audit Summary

> [!note] 2026-07-28 最终闭合
> 本页下方主体仍是 2026-07-23 Batch 0 的历史快照，2026-07-26 注记也是当时尚未闭合的
> 中间状态。后续逐 claim 审计已经取代这些旧状态数字：冻结历史页共 28 篇，
> 2190/2190 个 claim candidates 已有决定，其中 91 个已纠正，2099 个证据不足的结论
> 全部以 `retain-quarantined` 保留并隔离；1602 个结构单元已有去向决定，94/94 个
> correction dispositions 已闭合，destination validation error 为 0，新课程导入
> unresolved history claim 的数量为 0。正式最终结论见
> `docs/audits/pytorch_graph_series/2026-07-26/design_conformance_review.md`。
>
> 因此，下方的 832 个 `TBD`、1,041 个 unresolved-like rows 等数字只表示旧审计生成器
> 当时的覆盖状态，不能再当作当前交付状态，也不能解释为 2099 条隔离结论已被证明正确。

> [!note] 2026-07-26 扩展复核
> 本页主体保留2026-07-23 Batch 0的历史快照数字。2026-07-26按设计补入activation
> checkpointing、runtime memory quickstart、两份cache专题和两份Inductor memory页面后，
> manifest为28页；重生成inventory得到2,514 records、2,022个heading/code/locator机械
> ledger rows、0个unbalanced fence。当前仍有832个`TBD` destination、1,041个
> unresolved-like rows（1,030 `unresolved`、1 `not_semantically_audited`、10
> `needs_manual_resolution`）、0个带真实章节anchor的destination。故“PROCEED”只授权课程重写，
> 不能解释成历史资料无损迁移已验收；当前符合性见
> `docs/audits/pytorch_graph_series/2026-07-26/design_conformance_review.md`。
>
> 当前ledger生成器只为heading、code fence和source locator建立机械row；Mermaid作为
> code fence被保留，但普通Markdown表格、图片、实验语义和逐条非平凡claim尚未拥有
> 独立row。因此2,022不是设计§7.2所需审计单元的完备总数，不能用它证明逐内容无损。

## Decision

**Gate result: PROCEED with the 21-page rewrite.**

The source foundation is sufficient to begin Part I and continue through Part IV. The proceed
decision does not certify every legacy sentence. It certifies that:

1. every legacy structural unit is retained in the mechanical inventory and coverage ledger;
2. all mechanisms needed for the new series have a pinned current-source path;
3. known semantic errors have explicit replacements and destinations;
4. unverified examples and adjacent cache/runtime topics are quarantined and cannot enter new
   authoritative pages.

## Baseline

- PyTorch commit: `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`.
- Commit date: 2026-07-23.
- Detached clean source: `E:/97-codes/torch_parallel/p`.
- Legacy material: 22 manifest pages.
- Inventory hashes preserve the legacy pages before the later migration banners were inserted.
- Knowledge-repo working tree and original dirty PyTorch checkout were preserved.
- No Git commit was created.

## Mechanical coverage

| Metric | Result |
|---|---:|
| Manifest pages | 22 |
| Inventory records | 1,850 |
| Auditable heading/code/locator rows | 1,532 |
| Unbalanced code fences | 0 |
| Pages missing an inventory record | 0 |
| H2/H3 or code fences intentionally dropped | 0 |

The audit tool preserves duplicate locators, heading context, all code fences, Mermaid fences and
wikilinks. A valid path/line is still treated only as mechanical evidence until a semantic decision
is attached.

## Semantic status totals

| Status | Rows | Meaning |
|---|---:|---|
| `verified-current` | 279 | mechanism can be migrated with the pinned source and recorded qualifications |
| `corrected` | 724 | useful topic may remain, but wording/mechanism/API must be rewritten |
| `verified-historical` | 3 | retain only as explicitly historical behavior |
| `unresolved` | 526 | quarantined; not allowed into an authoritative page |
| **Total** | **1,532** |  |

The 526 unresolved rows are not hidden debt:

- 259 rows are the three cache/runtime-domain pages reserved for their later dedicated audit;
- 59 rows are unverified portions of the legacy all-in-one Inductor technical page;
- 131 rows are old pre-grad/joint/post-grad catalogs or examples whose individual registrations
  must be re-derived only when their destination chapter needs them;
- the remainder are quickstart/internal-API examples and isolated claims explicitly marked with a
  destination and required evidence.

No unresolved row is approved for migration.

## Highest-severity findings

1. AOTAutograd fw and bw are two independent FX graphs; saved/recomputed values are represented by
   copied nodes/placeholders and runtime ABI positions, not cross-graph Node edges.
2. Recompute is ordinary graph computation copied into bw and then reordered; a memory-budget
   number is not a literal percentage of all model activations saved.
3. A PatternExpr is a predicate/capture AST, not an FX graph. Reverse candidate traversal is not a
   reverse graph, and one matcher apply call is not a fixed point.
4. Pattern replacement does not imply a universal cleanup/reorder pipeline. DCE, stable topology,
   lint and recompile are stage-owned.
5. FX DCE, AOT saved-value selection and Scheduler DCE use different “dead” predicates.
6. GraphLowering interprets FX into lazy and realized IR; there is no one-to-one FX-node/IR-node
   identity.
7. Scheduler builds buffer/alias/mutation dependencies over realized operations, not a reverse FX
   graph.
8. Peak-memory reorder, wrapper buffer reuse and pooled inference planning are separate memory
   mechanisms.
9. Fusion does not guarantee register/shared-memory placement, and reduction/extern nodes are not
   universal fusion barriers.
10. Algorithm selection and Triton launch autotuning are distinct layers with different candidates
    and caches.
11. The historical generic fusion cost equation, fixed universal tile lists and large custom
    backend/fusion snippets are not source-faithful current implementations.
12. Dynamic/export symbolic-shape guards were repeatedly oversimplified; compile-time refinement
    does not eliminate runtime validity obligations.

The consolidated report now contains 94 correction identifiers: 49 P0, 18 P1, 6 P2 and 21 P3.
Four identifiers added during final claim-level closure (`F-020`, `P-022`, `I-031`, `I-032`)
cover historical source-index authority, serialized-pattern inventory drift, Scheduler
observability/source navigation and illustrative kernel-split qualification.

## What is safe to reuse

- conceptual motivation when it is restated with the corrected boundaries;
- the FX intrusive order plus args/users use-def model;
- the distinction between joint, fw and bw after applying the current construction/runtime ABI;
- PatternExpr's recursive predicate/capture role and root-indexed candidate lookup;
- reverse-cascading DCE and stable-topological-sort concepts when scoped to the correct graph;
- GraphLowering as an FX Interpreter;
- lazy loop IR, realization, Scheduler dependency/fusion and wrapper codegen after adopting the
  current class/stage map;
- dynamic shape concepts already tied to current ShapeEnv/SizeVar/codegen sources.

Every reused non-trivial claim still receives a pinned locator in its destination page.

## What must be rewritten from source

- all historical line-number maps;
- all quickstart/internal-API examples before they become Labs;
- fw/bw saved/recompute diagrams;
- pass-stage ordering and cleanup diagrams;
- Matcher complexity and fixed-point explanations;
- Scheduler dependency, DCE, fusion and memory-planning diagrams;
- backend registration examples and CUDA/XPU dispatch;
- autotuning architecture;
- generated-kernel/provenance explanations;
- any fixed candidate list, performance promise or specific generated kernel split.

## Acceptance criteria

| Criterion | Result |
|---|---|
| all 22 manifest pages inventoried | pass |
| all H2/H3 and code fences represented | pass |
| every mechanism required to author Part I has current source authority | pass |
| every P0/P1 finding has a destination | pass |
| no unexecuted example is labeled as a verified Lab | pass |
| unresolved claims blocked from authoritative migration | pass |
| source locators in the four detailed correction reports are line-valid | pass |
| inventory parser tests and repository diff checks | pass |

Final verification regenerated all audit artifacts in a fresh process, reported 22 pages, 1,850
records and zero unbalanced fences, passed all eight inventory tests and the Python compile check,
validated all 356 locators across the four detailed correction reports, and completed
`git diff --check` without errors. Part I writing can begin without another design checkpoint.
