# Mechanism Analysis Profile

Use this profile for one focused mechanism/design analysis and, by default, for a paper or general
artifact whose primary reader question is causal: why does this design/rule/result exist, how does
it work, and where does it stop? Read it after the shared source-fidelity contract and the selected
source pack.

This profile owns the **causal page shape**. It does not own source locators, code-hop semantics,
Wiki operations, or repository-wide planning.

Use `../analysis-focus.md` to choose the subject's explanatory center. For an algorithm, develop
the representation, decisive rule, correctness/termination, and complexity; for a numerical method,
develop the transformation, error and execution cost. Ownership and publication are central when
the mechanism has shared state or effects, not simply because its source is software.

## Central thesis and causal order

Lead every analysis unit with its **central thesis**: the pressure that matters and the main design
choice made under it. Preserve this semantic order; headings may follow the host style.

| Beat | Required answer |
|---|---|
| **1. Background / problem** | What workload, bottleneck, failure, requirement, or decision forced this unit to exist? What did the previous or naive approach do? |
| **2. Why this design** | What makes the decisive rule valid or the design appropriate? For a design choice, which obvious alternative lost and what criterion decided? Mark reconstructed rationale as analyst inference. |
| **3. Mechanism and evidence** | What representation, algorithm, numerical transformation, or state model makes the result happen, how does it execute, and which verified evidence proves it? |
| **4. Constraints and failure boundary** | What must remain true, what does it cost, what is unsupported, and where or how does it fail? |
| **5. Outlook** *(optional)* | What anchored change, deprecation, future-work statement, or constraint-driven pressure points forward? No anchor means omit it. |

The order is causal, not cosmetic: readers need the problem and deciding criterion before details.
A section that starts with “class X does Y” and never establishes the problem or alternative is
reference documentation, not analysis.

## State model and execution trace

For a local algorithm, explain its representation and invariants, then replay the decisive steps to
the returned result and establish correctness and cost. Multi-step arithmetic alone does not call
for a resource-ownership inventory.

For a mechanism with persistent/shared state, cooperating components, or asynchronous effects,
explain its state/ownership model separately from the real execution trace. The model states owners
and invariants; the trace proves transitions, boundary crossings, and completion. A call graph alone
does not establish the mechanism's state model.

For code, use the Execution trace contract in `../codebase.md`; do not restate it. For a spec,
incident, protocol, or paper, identify the corresponding state transition, evidence locator, and
completion/observation boundary in its native terms.

## Algorithmic implementation and principle figure

An analysis unit is an **algorithmic implementation** when its result depends on a non-identity
transformation or ordered rule such as partitioning, placement, routing, grouping, packing, masking,
permutation, scheduling, reduction, optimization, or iterative state transition. Ordinary CRUD,
direct field assignment, parameter validation, and one-to-one forwarding do not trigger this rule
unless the unit actually explains such an algorithm.

When triggered, start with named input identities and shapes and replay the smallest example through
the decisive intermediate state/layout/owner to its output. Include **at least one principle figure**
that makes this derivation reconstructable and marks the invariant, constraint, or cost that makes
the algorithm work. A class diagram, caller tree, code excerpt, or prose/table alone does not count.
For distinct live algorithms or data planes, reuse the same concrete example and give each path a
separately traceable lane or figure. Show local compute and the result; include
data/state/ownership movement, communication or synchronization, reconstruction, applicable
forward/backward differences, and incremental cost wherever those operations actually occur.

Use `drawing-wiki-figures` for medium choice, the written figure spec, the rendered artifact, and
the stranger-reader check; its pruning rules govern optional figures, not this required one.

## Evidence-shaped output

- **Code mechanism:** thesis and baseline → problem and decisive rule/design choice → smallest
  example and representation → live implementation evidence → costs/failure boundary → compact
  source-reading route. Expand state/ownership where the mechanism triggers it.
- **Paper/general analysis:** motivation → mechanism/model → evidence with its baseline → rejected
  alternative or comparison → limits.

Code excerpts, quotes, tables, and figures are evidence. Prose must carry the causal argument without
code excerpts or quotes; when the algorithmic trigger fires, prose and the required principle figure
must agree and each must make its assigned part of the explanation intelligible. Signatures and
source order are not headings.

## Completion gate

- The unit answers what problem it solves, why this design won, how it works, and where it stops or
  fails.
- State, ownership, invariants, alternatives, costs, and failure boundaries are explicit where
  applicable.
- For a code mechanism, the main execution trace passes a **hop-walk** from its live entry/selection
  through the actual completion or visibility boundary using the compact source-reading route.
- Optional outlook claims have direct version, deprecation, future-work, or constraint evidence.
- The shared source-fidelity gate and base `../page-review-rubric.md` pass.

A short complete causal argument is better than a long catalog of APIs, functions, or quotations.
