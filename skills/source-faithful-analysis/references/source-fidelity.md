# Shared source-fidelity contract

Read this for every `source-faithful-analysis` route and wherever another workflow, such as
`feature-tree-analysis`, imports source fidelity directly. It owns the evidence and boundary rules
shared by analysis pages and contract-style outputs. Source packs may specialize evidence
acquisition; the owning workflow defines output shape, approval, and review. Neither should copy
this contract.

## Source fidelity

- **Verified source evidence.** Open and read the supporting source before making the claim. Never
  guess an anchor, copy one from memory, or use a secondary summary as if it were the source.
- **Stable code anchor.** For code, preserve repository-relative paths plus qualified symbols,
  config keys, registry names, or test names in a compact route. Lines are optional, never a
  per-claim quota.
- **Frozen baseline.** Record the exact commit/version/snapshot and keep it fixed while collecting
  evidence. An anchor without a baseline is not stable evidence.
- **Source fact vs analyst inference.** Label what the source states or measures separately from
  what you infer. If the source is silent about rationale, say so before reconstructing it.
- **Conflicts stay visible.** Distinguish current behavior, tested guarantees, documented intent,
  historical rationale, and inference. Current implementation or observed data normally decides
  “what happens now”; a published contract still matters when describing a public promise.
- **The source beats folklore.** Surface corrections when the frozen source contradicts memory,
  marketing, an older version, or an existing page.

## Approved boundary and baseline

Every consumer inherits its **owning output contract**: scope, ownership, exclusions, evidence entry
points, completion test, and source baseline. Verify the selected source is at that exact baseline
and do not move it during evidence collection.

For an approved codebase page under `source-faithful-analysis`, inherit the blueprint's thesis and
repository commit. The page may organize sections inside its assigned contract, but it must not
rename, split, or reassign pages locally. Decompose documents only for non-code sources or unplanned
focused analysis.

When evidence requires a material ownership, scope, or baseline change, stop and return it to the
**owning approval workflow** with the evidence and proposed change. For a planned analysis page that
is `planning-codebase-analysis`; for a repo feature tree or function-point spec it is the
`feature-tree-analysis` approval gate. Do not silently apply one workflow's drift policy to another.

## Evidence workflow

Scale the workflow to the request. A focused question may need only anchor, locate, and verify; a
substantial page normally uses every phase.

### Phase 0 — Anchor

1. Obtain the exact source in a citeable form and record its baseline.
2. Read one relevant house-style example if the output joins an existing document system.
3. Load every evidence and output contract selected by the owning workflow completely.
4. For approved codebase work, verify the inherited commit before collecting claims.

### Phase 1 — Map before writing

- Skim the source shape, then locate the few load-bearing entries and evidence regions.
- Separate live behavior from compatibility, legacy, background, and speculation.
- Map the owned unit: output contract, evidence, boundary, and adjacent owners.
- Treat source layout as an evidence map, not the document outline.

### Phase 2 — Locate, model, anchor

Repeat for every non-trivial claim:

1. **Locate** the exact function, passage, table, field, row, or event.
2. **Read** enough surrounding context to catch conditions and negative cases.
3. **Model** the state, ownership, invariant, choice, or contract field required by the owning output
   contract; do not introduce causal prose into a contract-style spec that forbids it.
4. **Anchor** at the source pack's useful granularity. Code pages consolidate stable anchors into a
   compact source-reading route instead of repeating citations after every sentence or hop.

Source excerpts are evidence, not the output skeleton. Remove every excerpt as a mental test: the
remaining explanation or contract fields should still be intelligible and verifiable.

### Phase 3 — Write at the owned boundary

- Follow the owning output contract's structure and field semantics rather than source-file order.
- Keep one authoritative owner for each concept or function point and use owner-aware cross-links.
- Use tables, traces, and figures only where the owning contract calls for them.
- Do not broaden scope to make collected evidence convenient.

### Phase 4 — Integrate and verify

- Reconcile new evidence with existing claims; preserve contradictions explicitly.
- Verify cross-references mechanically and spot-check anchors against the frozen source.
- Run the owning review/approval contract and the host repository's quality gates.

### Phase 5 — Grow on demand

When a follow-up exposes a gap, return to Phase 2 and fold the verified answer into the
authoritative unit. Do not answer a newly exposed source question from memory.

## Shared completion gate

- Every non-trivial claim is grounded in evidence actually opened at the declared baseline.
- Inference, conflict, legacy behavior, and unsupported cases are labelled rather than flattened.
- Cross-references point to authoritative owners; duplicated explanations do not create a second
  source of truth.
- The owning evidence, output, review/approval, and host contracts all pass.

The governing rule: **if it is not supported by the spot you just read, it does not belong in the
analysis.**
