# Function-point spec template and evidence recipes

One function point = seven blocks. The semantics of the four core fields are **fixed** — do not
repurpose them. Each field comes with an evidence recipe: where in the code to mine it. Source
fidelity follows the Source fidelity section of `source-faithful-analysis`: **open and read the
spot before making the claim**; every locator must be real at the frozen baseline.

## Template

```text
### <ID> <function point name>
- Position: one-line definition (what it does for the caller); entry file:line; module path.
- Inputs: explicit parameters/config options (name, type, default, valid domain); implicit
  inputs (environment variables, global state, upstream data and its shape conventions).
- Outputs: return values/products (files, state changes, communication); observable side
  effects; error outputs (exception types, exit codes, logs).
- Processing logic: numbered main-path steps, each with a locator, walkable in order; key
  branches and their trigger conditions listed separately.
- Boundary constraints: the actual consequence of illegal/out-of-range input (assert / raise /
  silent truncation / no validation — say which); dimensional limits (caps, alignment,
  divisibility); mutual exclusion and dependency with other function points.
- Supported scope: a support matrix over dimensions (platform/backend/dtype/parallel
  mode/version … as fits the repo); explicit unsupported items must carry evidence locators;
  defaults.
- Tests & links: targeted test anchors (file or case name; write "no targeted tests" if none);
  links to related mechanism pages / adjacent function points.
```

## Field semantics (drift control)

- **Boundary constraints** ≠ usage advice, ≠ project policy. Only "constraints that actually
  exist in the implementation, and their consequences". "No validation (silently accepts illegal
  values)" is itself an important finding — state it, with the spot you checked.
- **Supported scope** ≠ background, ≠ quality bars. Only "supported / unsupported / defaults per
  dimension" with evidence.
- **Processing logic** walks the main path: pure forwarding helpers collapse into one step; an
  exhaustive call graph drowns the contract.
- If a field genuinely does not apply, write "N/A: <reason>" — never leave it blank.

## Evidence recipes

| Field | Mine from | Note |
|---|---|---|
| Inputs | signatures, argparse/click definitions, dataclass/schema fields, env reads | copy defaults from code, not docs |
| Outputs | return statements, file-write/message-send sites, raise sites, exit codes | keep return values and side effects apart |
| Processing logic | the main call chain downward from the entry | main path only; branches listed separately |
| Boundary constraints | validation code, assert/raise search, parameter cross-checks | "no validation" is a finding too |
| Supported scope | NotImplementedError, platform/version guards, `pytest.mark.skipif` (high-confidence), build dependency declarations, doc support matrices | never write a negative (unsupported) without evidence |
| Test anchors | same-name/same-topic files under tests/ and their assertions | tests are evidence of protected behavior |

**Evidence priority (for current behavior)**: code > tests > docs > comments. When a documented
promise conflicts with code behavior, present both with their own locators — never silently pick
one (the same visible-conflict rule as `source-faithful-analysis`).

## Writing rules

- A spec is a **contract, not an essay**: no causal argument, no design trade-offs — that is the
  mechanism pages' job; link to them.
- One anchored block per function point, all seven blocks present.
- **Can't finish reading → don't write**: a leaf whose key implementation you have not fully read
  goes back to `planned` for the next wave; filling fields from a skim is forbidden.
- No length quota: a spec is complete when all four core fields carry evidence; piling up code
  blocks adds nothing.

## Example (fictional repo)

The example deliberately points at **no real code** — a half-real example (real file names plus
invented behavior) gets copied into real specs. Copy the structure below; the content always
comes from code you personally read at the frozen baseline. Assume a fictional Markdown checker
repo `mdlint`:

```text
### lint/dead-links/incremental incremental dead-link check
- Position: reports only dead links introduced by this change, so legacy debt does not block new
  edits; entry <cli.py:L?> (--changed).
- Inputs: --changed (bool, default off), --strict (bool), explicit path list; implicit input:
  git HEAD baseline content.
- Outputs: stdout diagnostic lines; exit code 0/1 (2 on environment error); no files written.
- Processing logic: ① collect the changed-file set <L?> → ② for files with a HEAD baseline,
  diff against baseline diagnostics <L?> → ③ full-check new files <L?> → ④ aggregate the exit
  code <L?>.
- Boundary constraints: run outside a git repo → the git invocation fails → exit code 2 <L?>;
  untracked new files are treated as brand new and fully checked <L?>; the combination rule when
  explicit paths and --changed are both given is written per the actual code, case by case
  (multi-entry combination behavior is a boundary-constraint hotspot — never wave it away with
  "ignored/overridden").
- Supported scope: only git-tracked/added Markdown; when HEAD does not exist (empty repo) the
  code does not handle it explicitly — write "unverified, no claim", or test it and state the
  result.
- Tests & links: <tests/test_incremental.py::test_?>; link the relevant mechanism page for
  background (use real links when the host is the wiki).
```
