# Mechanism Analysis Profile

Use this profile for one focused mechanism/design analysis and, by default, for a paper or general
artifact whose primary reader question is causal: why does this design/rule/result exist, how does
it work, and where does it stop? Read it after the shared source-fidelity contract and the selected
source pack.

This profile owns the **causal page shape**. It does not own source locators, code-hop semantics,
Wiki operations, or repository-wide planning.

## Central thesis and causal order

Lead every analysis unit with its **central thesis**: the pressure that matters and the main design
choice made under it. Preserve this semantic order; headings may follow the host style.

| Beat | Required answer |
|---|---|
| **1. Background / problem** | What workload, bottleneck, failure, requirement, or decision forced this unit to exist? What did the previous or naive approach do? |
| **2. Why this design** | Which route won, what obvious alternative lost, and what criterion decided? If rationale is reconstructed rather than stated, mark it as analyst inference. |
| **3. Mechanism and evidence** | What state/model makes the result happen, how does it execute, and which verified evidence proves it? |
| **4. Constraints and failure boundary** | What must remain true, what does it cost, what is unsupported, and where or how does it fail? |
| **5. Outlook** *(optional)* | What anchored change, deprecation, future-work statement, or constraint-driven pressure points forward? No anchor means omit it. |

The order is causal, not cosmetic: readers need the problem and deciding criterion before details.
A section that starts with “class X does Y” and never establishes the problem or alternative is
reference documentation, not analysis.

## State model and execution trace

For a stateful or multi-stage mechanism, explain its state/ownership model separately from the real
execution trace. The model states owners and invariants; the trace proves transitions, boundary
crossings, and completion. A call graph alone does not establish the mechanism's state model.

For code, use the Execution trace contract in `../codebase.md`; do not restate it. For a spec,
incident, protocol, or paper, identify the corresponding state transition, evidence locator, and
completion/observation boundary in its native terms.

## Evidence-shaped output

- **Code mechanism:** thesis and baseline → bottleneck/constraints → state/model → design versus
  alternative → live implementation evidence → costs/failure boundary → compact source-reading
  route.
- **Paper/general analysis:** motivation → mechanism/model → evidence with its baseline → rejected
  alternative or comparison → limits.

Code excerpts, quotes, tables, and figures are evidence. The prose must remain intelligible if they
are removed; signatures and source order are not headings.

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
