---
name: source-faithful-analysis
description: >-
  Use for source-grounded, mechanism-level analysis of a codebase, paper, spec/RFC, dataset, API,
  incident, report, or other artifact: architecture and subsystem deep dives, implementation
  walkthroughs, design-rationale audits, comparisons, reverse engineering, and knowledge-base
  ingestion. Trigger when the answer must explain why a design exists, how state/data/control move,
  what constraints or failure boundaries apply, or when current sources may contradict folklore.
  Every non-trivial claim is tied to a verified exact locator and frozen baseline. For a new whole
  codebase or unplanned multi-page codebase domain, use planning-codebase-analysis first; use this
  skill for an approved page/unit or a focused mechanism.
---

# Source-Faithful Analysis

Turn an actual source into an explanation the reader can verify and reason from. The skill owns two
things only: **source fidelity** and the **causal analysis contract**. Source-type packs instantiate
that contract; host-specific planning, Wiki maintenance, math, and figure production stay with their
own skills.

## Routing

Read one source-type pack completely before acting:

| Source | Pack | Locator | Baseline |
|---|---|---|---|
| Code/framework/library | `references/codebase.md` | `file:line` | commit + branch/tag + date |
| Research paper | `references/paper.md` | § / page / Table / Fig. / Eq. | document id + version + date |
| Spec, dataset, API, incident, report, product, other | `references/general.md` | clause / cell / endpoint / timestamp | version / snapshot / time window |

For mixed evidence, read every relevant pack. Let each source answer the question it can support:
for example, a paper can explain rationale while released code/config proves current behavior.

For a new whole codebase or unplanned multi-page codebase domain, stop and use
`planning-codebase-analysis`. Once an approved blueprint exists, execute exactly one approved
analysis unit/page at a time. Inherit its thesis, concept ownership, exclusions, evidence entry
points, completion test, and approved repository commit; do not reopen repository-wide planning.

## Source fidelity

- **Verified exact locator.** Open and read the cited spot before making the claim. Never guess a
  locator, copy one from memory, or use a secondary summary as if it were the source.
- **Frozen baseline.** Record the exact commit/version/snapshot and keep it fixed while collecting
  evidence. A locator without a baseline is not stable evidence.
- **Source fact vs analyst inference.** Label what the source states or measures separately from
  what you infer. If the source is silent about rationale, say so before reconstructing it.
- **Conflicts stay visible.** Distinguish current behavior, tested guarantees, documented intent,
  historical rationale, and inference. For “what happens now,” current implementation or observed
  data normally wins; for a public promise, the published contract still matters.
- **The source beats folklore.** Surface corrections when the frozen source contradicts memory,
  marketing, an older version, or an existing page.

## Canonical analysis contract

Lead every analysis unit with its **central thesis**: the pressure that matters and the main design
choice made under it. Then use this single canonical order; source packs add evidence details but do
not redefine the order.

| Beat | Required answer |
|---|---|
| **1. Background / problem** | What workload, bottleneck, failure, requirement, or decision forced this unit to exist? What did the previous or naive approach do? |
| **2. Why this design** | Which route won, what obvious alternative lost, and what criterion decided? If rationale is reconstructed rather than stated, mark it as analyst inference. |
| **3. Mechanism and evidence** | What state/model makes the result happen, how does it execute, and which verified evidence proves it? |
| **4. Constraints and failure boundary** | What must remain true, what does it cost, what is unsupported, and where or how does it fail? |
| **5. Outlook** *(optional)* | What anchored change, deprecation, future-work statement, or constraint-driven pressure points forward? No anchor means omit it. |

The order is causal, not cosmetic: readers need the problem and design criterion before details.
Do not force the beat labels into headings when fluent prose communicates the same structure, but do
not omit the questions. A section that starts with “class X does Y” and never establishes the
problem or alternative is reference documentation, not analysis.

### Static structure and dynamic motion

For a system or multi-stage artifact, answer two orthogonal questions:

1. **Static structure:** responsibilities, contracts, state ownership, invariants, and boundaries.
2. **Dynamic lifecycle:** how one real unit of work crosses those boundaries and when state becomes
   valid, changes owner, completes, or becomes externally visible.

Show structure before motion. A directory tree is not an architecture, and a call graph alone does
not explain responsibility. Use separate views when combining them would blur either question.

## Workflow

Scale the workflow to the request. A focused question may use only Phases 0 and 2; a substantial
analysis normally uses all phases.

### Phase 0 — Anchor

1. Obtain the exact source in a citeable form and record the frozen baseline.
2. Read one relevant house-style example if the output joins an existing document system.
3. Select and read the required source-type pack.
4. For an approved codebase page, inherit the approved repository commit and verify that the
   checkout is at that exact commit. Do not move the approved source checkout during page execution.

### Phase 1 — Map before writing

- Skim the source shape, then locate the few load-bearing entry points and evidence regions.
- Separate live behavior from compatibility, legacy, background, and speculation.
- Map the owned unit: its question, state/model, evidence, boundary, and adjacent owner pages.
- Decompose documents only for non-code sources or unplanned focused analysis. An approved codebase
  page may organize sections inside its assigned contract, but it must not rename, split, or
  reassign pages locally. A required page split is material drift: stop and return it to
  `planning-codebase-analysis` with the evidence and proposed boundary change.

### Phase 2 — Locate, model, cite

Repeat per non-trivial claim:

1. **Locate** the exact function, passage, table, field, row, or event.
2. **Read** enough surrounding context to catch conditions and negative cases.
3. **Model** the pressure, state/ownership, invariant, design choice, and consequence in your own
   words.
4. **Cite** the verified locator and reproduce only the evidence needed to prove that model.

Source excerpts are evidence, not the narrative skeleton. Remove every excerpt as a mental test:
the prose should still explain why the design exists, how it works, and what breaks.

### Phase 3 — Write at the owned boundary

- Keep one thesis and one concept owner per analysis unit.
- Choose headings for the reader’s question, not for source-file or API order.
- Use the source-type pack’s structures, tables, and evidence routes only where they carry the
  argument.
- If authorized delegation is useful, read `references/parallel-agent-contract.md`; otherwise run
  the same contract serially. Analysis quality must not depend on agent availability.
- When writing into a host knowledge base, invoke its maintenance, math, and figure skills rather
  than copying their rules here.

### Phase 4 — Integrate and verify

- Reconcile new evidence with existing claims; preserve explicit contradictions instead of silently
  choosing the convenient version.
- Link summaries to the authoritative owner instead of re-explaining the same mechanism.
- Verify cross-references mechanically and spot-check cited locators against the frozen source.
- When the unit belongs to a planned wave, an **independent reviewer (never the writer)** applies
  `references/page-review-rubric.md` and returns a per-page verdict; a rejected page goes back to
  Phase 2 with the failing check named. The user samples pages on top of the reviewer.
- Run the host repository’s required quality gates.

### Phase 5 — Grow on demand

When a follow-up exposes a gap, return to Phase 2 and fold the verified answer into the authoritative
unit. Do not answer a newly exposed source question from memory.

## Output shapes

Match the source, audience, and host conventions. Preserve the contract rather than a rigid heading
template:

- **Mechanism/design page:** thesis and baseline → bottleneck/constraints → state/model → design vs
  alternative → implementation or measured evidence → costs/failure boundary → source-reading route.
- **Overview/index:** domain thesis → static responsibility/owner map → representative lifecycle →
  reading routes to owner pages.
- **Paper/general analysis:** motivation → mechanism/model → evidence with its baseline → rejected
  alternative → limits.

A short page with a complete causal argument is better than a long page padded with signatures,
quotes, or repeated summaries.

## Completion gate

Before delivery, verify:

- Every non-trivial claim has a locator you actually opened at the declared baseline.
- Every owned unit answers what problem it solves, why this design won, how it works, and where it
  stops or fails.
- State, ownership, invariants, alternatives, costs, and failure boundaries are explicit where
  applicable.
- Architecture overviews contain both the static responsibility view and one real lifecycle.
- Inference, conflict, legacy behavior, and unsupported cases are labelled rather than flattened.
- The main execution trace passes a **hop-walk**: following its locators in order reaches the
  completion boundary without any search step the page does not provide.
- Cross-references point to concept owners and the host quality gates pass.

The throughline: **if it is not supported by the spot you just read, it does not belong in the
analysis; if it does not explain the causal choice, it is only a summary.**
