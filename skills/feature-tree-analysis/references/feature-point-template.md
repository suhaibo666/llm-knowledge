# Function-point spec template and evidence recipes

One function point = seven blocks: a position block, the **five core fields** (inputs, outputs,
processing logic, boundary constraints, supported scope), and a tests-and-links block. The
semantics of the five core fields are **fixed** — do not repurpose them. Each field comes with an
evidence recipe: where in the code to mine it. Source fidelity follows the shared
`../../source-faithful-analysis/references/source-fidelity.md` contract: **open and read the source
before making the claim**. In the page, deduplicate evidence as stable
`path::qualified.symbol` anchors at the frozen baseline; line numbers are optional for exact
excerpts, not a per-field or per-step quota.

## Template

```text
### <ID> <function point name>
- Position: one-line definition (what it does for the caller); stable `path::qualified.symbol`
  entry; module path.
- Inputs: explicit parameters/config options (name, type, default, valid domain); implicit
  inputs (environment variables, global state, upstream data and its shape conventions).
- Outputs: return values/products (files, state changes, communication); observable side
  effects; error outputs (exception types, exit codes, logs).
- Processing logic: numbered main-path steps naming the participating symbols, walkable against
  the compact source-reading route; key branches and their trigger conditions listed separately.
- Boundary constraints: the actual consequence of illegal/out-of-range input (assert / raise /
  silent truncation / no validation — say which); dimensional limits (caps, alignment,
  divisibility); mutual exclusion and dependency with other function points; plus the
  conditional dimensions below when their trigger exists.
- Supported scope: a support matrix over dimensions (platform/backend/dtype/parallel
  mode/version … as fits the repo); every supported and unsupported claim is grounded in opened
  evidence; defaults.
- Tests & links: targeted test anchors (file or case name; write "no targeted tests" if none);
  links to related mechanism pages / adjacent function points.
```

## Field semantics (drift control)

- **Boundary constraints** ≠ usage advice, ≠ project policy. Only "constraints that actually
  exist in the implementation, and their consequences". "No validation (silently accepts illegal
  values)" is itself an important finding — state it, with the spot you checked.
- **Boundary constraints — conditional dimensions.** When the trigger exists in the code, the
  boundary block must cover it; when it does not, write nothing (no empty matrices):
  concurrency and ordering (locks, races, ordering assumptions) · resource capacity (memory,
  handles, queue bounds, backpressure) · completion and visibility (when an effect is durable or
  observable; submission vs completion) · retry and idempotency · partial side effects on
  failure · cleanup and rollback. These implementation conditions often require deeper evidence in
  an analysis page; here they yield contract constraint lines, not mechanism prose.
- **Supported scope** ≠ background, ≠ quality bars. Only "supported / unsupported / defaults per
  dimension", each with evidence — positive claims need anchors too, not just negatives.
- **Processing logic** walks the main path: pure forwarding helpers collapse into one step; an
  exhaustive call graph drowns the contract.
- If a field genuinely does not apply, write "N/A: <reason>" — never leave it blank.

## Evidence recipes

| Field | Mine from | Note |
|---|---|---|
| Inputs | signatures, argparse/click definitions, dataclass/schema fields, env reads | copy defaults from code, not docs |
| Outputs | return statements, file-write/message-send sites, raise sites, exit codes | keep return values and side effects apart |
| Processing logic | the main call chain downward from the entry | main path only; branches listed separately |
| Boundary constraints | validation code, assert/raise search, parameter cross-checks; for the conditional dimensions: locks/queues/pools, retry loops, cleanup/finally blocks, transaction or rollback code | "no validation" is a finding too |
| Supported scope | NotImplementedError, platform/version guards, `pytest.mark.skipif` (high-confidence), build dependency declarations, doc support matrices | never write a positive or negative claim without evidence |
| Test anchors | same-name/same-topic files under tests/ and their assertions | tests are evidence of protected behavior |

Apply the fact/inference and visible-conflict rules from the shared source-fidelity contract. This
template does not redefine evidence priority.

## Writing rules

- A spec is a **contract, not an essay**: no causal argument, no design trade-offs — that is the
  mechanism pages' job; link to them.
- One block per function point, all seven blocks present, followed by a deduplicated source-reading
  route for the symbols/tests that support its fields.
- **Can't finish reading → don't write**: a leaf whose key implementation you have not fully read
  goes back to `planned` for the next wave; filling fields from a skim is forbidden.
- Files you read outside the leaf's `owns.files`, and any flag / subcommand / endpoint /
  registration / trigger no leaf owns, go into your return to the coordinator — not into your
  leaf.
- No length quota: a spec is complete when all five core fields carry evidence; piling up code
  blocks adds nothing.

## Example (fictional repo)

The example deliberately points at **no real code** — a half-real example (real file names plus
invented behavior) gets copied into real specs. Copy the structure below; the content always
comes from code you personally read at the frozen baseline. Assume a fictional Markdown checker
repo `mdlint`:

```text
### lint/dead-links/incremental incremental dead-link check
- Position: reports only dead links introduced by this change, so legacy debt does not block new
  edits; entry `cli.py::main` (`--changed`).
- Inputs: --changed (bool, default off), --strict (bool), explicit path list; implicit input:
  git HEAD baseline content.
- Outputs: stdout diagnostic lines; exit code 0/1 (2 on environment error); no files written.
- Processing logic: ① `collect_changed_files` collects the changed-file set → ②
  `baseline_diagnostics` diffs tracked files against HEAD → ③ `check_file` fully checks new files
  → ④ `main` aggregates the exit code.
- Boundary constraints: run outside a git repo → the git invocation fails → exit code 2; untracked
  new files are treated as brand new and fully checked; the combination rule when
  explicit paths and --changed are both given is written per the actual code, case by case
  (multi-entry combination behavior is a boundary-constraint hotspot — never wave it away with
  "ignored/overridden"); no concurrency, retry, or rollback triggers exist in this code path, so
  the conditional dimensions are omitted.
- Supported scope: only git-tracked/added Markdown; when HEAD does not exist (empty repo)
  the code does not handle it explicitly — write "unverified, no claim", or test it and state
  the result.
- Tests & links: <tests/test_incremental.py::test_?>; link the relevant mechanism page for
  background (use real links when the host is the wiki).
- Source-reading route: `cli.py::main`, `incremental.py::collect_changed_files`,
  `incremental.py::baseline_diagnostics`, `checker.py::check_file`,
  `tests/test_incremental.py::test_?`.
```
