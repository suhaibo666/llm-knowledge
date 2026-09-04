# Concrete Software Feature Analysis Profile

Read this through the core router after `../source-fidelity.md` and `../codebase.md` when the
deliverable must explain one concrete software feature in enough depth for a reader to understand
the design and continue into source. This is a **page-shaping profile**. The shared kernel owns
source fidelity; `../codebase.md` owns source anchors, state/data/control tracking, and
execution-trace semantics. Do not repeat those contracts here or create a second top-level skill
for the same work.

This profile is not for a repository architecture chapter, an exhaustive feature tree, or an API
reference. Route those tasks as the core skill specifies.

## Universal feature-page contract

Treat this as a content contract, not a fixed heading or length template. Preserve the following
explanatory order while adapting titles to the feature and host.

### 1. Introduce the feature

- State the problem and pressure in **one compact paragraph**. Establish enough context to make the
  design necessary; do not split background into a chronology of mini-sections.
- Describe the solution shape at concept level: what state, data, control, or ownership changes and
  where the feature sits in the surrounding system.
- Summarize benefits, costs, and constraints before entering implementation details.
- Add a symbol/term table last and only when notation is reused later. Do not inventory names that
  the prose can define at first use.

### 2. Explain the mechanism from primitive to system

- Start with the **smallest meaningful example** that exposes the feature's decisive transformation
  or state transition. For an algorithm this may be one operation; for an ordinary service feature
  use the smallest real request or transaction. Do not invent a toy example that changes semantics.
- Expand from that example to the feature's **primitive → system** placement: selection/construction,
  participating modules, consumers, completion boundary, and adjacent owners.
- **Per load-bearing component**, answer all applicable questions:
  1. What responsibility and input→output or pre-state→post-state contract does it own?
  2. **Why this component** or boundary exists, and which obvious alternative loses under the stated
     constraint?
  3. How does state, data, or control move through it, including the output/completion semantics?
  4. What guard, direct cost, system cost, unsupported case, or failure boundary does it introduce?
- Explain live variants and selection conditions where they exist. Do not turn class and function
  catalogs into fake implementation depth.

### 3. Close the feature as a whole

- Reconcile the component explanations into one end-to-end lifecycle. The last step must reach the
  feature's real result, externally visible state, or downstream handoff—not merely a helper return.
- Give one **cost ledger** that connects benefit to payment. Select only applicable dimensions:
  latency, throughput, CPU/accelerator work, memory, storage, network/I/O, synchronization or
  contention, startup, compatibility, implementation complexity, and operational burden.
- State the **aggregate cost** and operating envelope together with the failure boundary. If a cost
  is plausible but unmeasured or a rationale is inferred, label that evidence status rather than
  presenting a precise claim.

## Implementation presentation

Use prose as the explanation and views as verification aids:

- Add a **class / ownership view** when multiple classes, processes, or state-bearing objects
  cooperate. It shows responsibility, dependency, and state ownership—not every inheritance edge.
- For a real multi-hop function path, add a fenced **ASCII caller tree** from live entry/selection
  to completion or visibility. Render the path selected under `../codebase.md`'s execution-trace
  contract rather than defining a second hop-selection, elision, or annotation policy here.
- Use an additional data-layout, state, or sequence figure only when it answers a different question.
  Delegate medium and house style to the active figure skill. A **call graph does not replace** the
  state model, per-hop execution semantics, design reasons, or failure analysis.
- Keep the ASCII caller tree consistent with the compact source-reading route owned by
  `../codebase.md`.

Example shape—not literal names:

```text
public_entry(request)
`-- select_live_implementation(config)
    `-- owner.prepare(request)
        `-- worker.execute(prepared)
            `-- publisher.commit(result)
```

## Conditional depth

**Apply only when the trigger is present.** These rows deepen the universal contract; they are not
mandatory headings. **Do not create empty sections** to prove that an absent trigger was considered.

| Trigger | Required closure |
|---|---|
| **Training** and the feature affects gradient semantics | Trace **forward → objective/loss → backward** → gradient-ready or optimizer handoff. Explain different forward/backward transformations and communication rather than mirroring one by analogy. |
| **Parallel or distributed** execution | Identify partition/replication axes, rank/topology ownership, collectives or messages, local/global identities, synchronization points, and compute/communication/memory costs. |
| A **companion mechanism** is required to satisfy the page's stated, evidenced correctness, capacity, or performance target | Explain why the combination is needed, its boundary and data/state handoff, and its incremental cost at the **same explanatory depth** as the primary mechanism. A merely compatible or generally beneficial option needs only a scoped note or owner-page link. |
| Several network/system modules participate | Walk the real feature through every load-bearing module using the per-component questions; do not assume one primitive example proves whole-system behavior. |

Async/concurrent execution and data-representation changes already have conditional depth profiles
in `../codebase.md`; apply those profiles directly instead of restating them here.

## Completion gate

Apply the shared base rubric and `../reviews/feature-analysis.md`. The feature is complete only when
the primitive-to-system explanation, end-to-end completion boundary, conditional depth, cost ledger,
operating envelope, and source-reading route agree.
