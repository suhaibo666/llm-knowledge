# Spec-writer contract — dispatching one function-point spec wave

Use this — and only this — to dispatch parallel writers for function-point specs. Do **not**
reuse `../../source-faithful-analysis/references/parallel-agent-contract.md`: it dispatches a prose
analysis under a selected document profile, while a spec must follow the fixed contract fields
below. The coordinator owns the tree, the manifest, the baseline, and integration; each writer
owns one output file and only the leaves assigned to it.

Every writer still follows the shared evidence rules in
`../../source-faithful-analysis/references/source-fidelity.md`. Import that evidence kernel only;
do not load a prose document profile for function-point specs.

## Contract template

Fill the placeholders and send the whole block to each writer:

```text
You are writing function-point SPECS for an approved repo feature tree. Treat the source as
read-only ground truth. Specs are contracts, not analyses.

SOURCE
- Frozen baseline: <owner/repo @ full commit> (<branch>, <date>). Do not fetch, pull, checkout,
  reset, or otherwise move the checkout.
- Checkout path: <absolute path>
- Ownership manifest: <path> — your leaves and their `owns` extents are listed there.

OWNED OUTPUT
- Write only: <absolute output path>
- Leaves assigned (IDs from the manifest): <id list>. Do not add, split, merge, or rename leaves.
- Template: <path to feature-point-template.md> — seven blocks; five core fields with fixed
  semantics (inputs, outputs, processing logic, boundary constraints, supported scope).

SPEC CONTRACT
1. Apply the shared source-fidelity contract. Name opened symbols in the fields and add one
   deduplicated `path::qualified.symbol` source-reading route per leaf; line numbers are optional
   for exact excerpts, never a per-entry requirement.
2. No causal argument, no design rationale, no rejected alternatives, no execution-trace prose —
   link to the owning mechanism page instead.
3. Field semantics are fixed: boundary = constraints that exist in code and their consequences
   ("no validation" is a finding; cover the conditional dimensions only where their trigger
   exists); supported scope = supported / unsupported / defaults per dimension, each with
   evidence; claims without evidence are not written.
4. Apply the shared kernel's source-fact, analyst-inference, and visible-conflict rules without
   restating or weakening them.
5. Cannot finish reading a leaf's implementation → leave it `planned` and say so; never fill
   fields from a skim.

HOST INTEGRATION
- Use the coordinator's page names, anchors, and links; do not touch the manifest or the tree
  overview — report changes instead.
- Cross-link the owning mechanism pages by their existing names; follow the host's page and link
  rules when the host is a knowledge base.

RETURN TO COORDINATOR
1. Per leaf: status written (spec'd / left planned) and the deduplicated source anchors you opened.
2. Per leaf: files you read outside the leaf's `owns.files` (ownership drift), and any flag,
   subcommand, endpoint, registration, trigger, or format you met that no leaf owns (candidate
   new leaves) — do not fold them into your leaves.
3. Tree-cut problems (a leaf that should split/merge or belongs elsewhere), with evidence — do
   not act on them.
4. Source conflicts, and the points you deliberately left as "unverified, no claim".
```

## Coordinator checks

- Freeze the manifest, baseline, output paths, and leaf assignment before dispatch; one leaf has
  exactly one writer.
- Calibrate with one real leaf before fanning out when the template is new to the writers.
- On return: update `owns` for reported extent drift, add candidate leaves and tree-cut findings
  to the pending list (approval gate when material), run `python tools/check_feature_tree.py`,
  then hand the wave to the reviewer (`review-rubric.md`).
- Reject on sight a return that opens with class/function order, argues why a design won, or
  narrates an execution trace — those are mechanism pages, not specs.
