# Behavior evals for feature-tree-analysis

Live-agent scenarios with pass criteria, in the RED–GREEN–REFACTOR sense of the writing-skills
discipline. Re-run the affected scenarios after any edit to this skill and append to the run log.
Target: this repository's own `tools/` directory — small, real, no external checkout needed
(freeze the repo's HEAD commit as the baseline). Mechanical rules are covered by
`tools/test_check_feature_tree.py`; these scenarios cover the judgment the checker cannot.

## S0 — Baseline without the skill (expected to FAIL)

Prompt: "把 <repo>/tools 当作独立小代码仓，梳理它的功能树，并为每个功能点给出定义（输入输出 /
处理逻辑 / 边界约束 / 支持范围）。写成 Markdown 到 <scratch>/…"

Failure modes observed on 2026-09-02, now countered in the SKILL.md red-flags table: unfrozen
baseline ("工作区当前状态含未提交改动"); directory-shaped tree with one file = one function item;
internal helper functions as leaves (142 leaves); tests and data artifacts as nodes; no
reconciliation; policy text in the supported-scope field; skim-based specs for the labs part; no
approval stop (1194 lines written in one go).

## S1 — Tree phase with the skill (must PASS)

Prompt: same task, plus "read `skills/feature-tree-analysis/SKILL.md` and every file under its
`references/` first and follow them; write only to <scratch>/…".

Pass criteria:
- full frozen commit hash; uncommitted changes inside the scope reported, not silently used;
- surfaces enumerated at symbol level before decomposition; nodes named by caller-visible
  behavior; no `utils` / tests / artifacts nodes;
- ownership manifest written in the v3 schema: `nodes` with name / responsibility / parent for
  every non-leaf node, leaves with `Class.field` flag claims where names collide, exclusions each
  with a reason; `python tools/check_feature_tree.py <manifest> --phase proposal --strict` run and
  reported at zero;
- proposal package presented with pending judgment calls; **no spec content written**; the run
  ends waiting for approval; nothing persisted outside scratch.

## S2 — Spec phase with the skill (must PASS)

Prompt: the filled `references/spec-writer-contract.md` for two approved leaves (e.g.
`gates/math/full-scan` and `gates/math/incremental` of `tools/check_math.py`) at the frozen
commit, with a manifest slice listing their `owns`.

Pass criteria:
- output contains only spec blocks — no rationale, alternatives, or trace prose;
- every behavior claim is grounded in source the writer opened; all five core fields are present;
  "no validation" and "unverified, no claim" used where applicable;
- the RETURN block lists deduplicated source anchors, ownership drift, candidate leaves, and tree-cut
  findings separately, and the writer did not act on tree-cut findings;
- rubric R1–R6 pass on both leaves when applied by a separate reviewer.

## S3 — Baseline bump (must PASS)

Setup: a small v3 manifest for the root-level Python tools (`surfaces.files.include:
["tools/*.py"]`) frozen at commit `8408308` (before `check_coverage.py` / `check_locators.py`
existed), zero in phase `proposal`; then ask for a bump to `af1278db…` following tree-method §7.
The delta adds two tools and their tests, so unowned files must surface.

Pass criteria: surfaces re-enumerated at the new commit (entries re-listed, not just mapped);
the checker's F1/E1 gaps for the new files turned into candidate leaves (placed under a node) or
reasoned exclusions before any status advances; the delta ∩ `owns.files` affected-leaf list
produced; the manifest commit replaced last; `docs/radar/watchlist.yaml` `kb_baseline` untouched.

## Run log

| date | scenario | skill version | result | notes |
|---|---|---|---|---|
| 2026-09-02 | S0 | none | FAIL (expected) | 142 internal-function leaves, unfrozen baseline, no stop |
| 2026-09-02 | S1 | pre-manifest draft | PASS | 64 leaves, 3 surfaces diff = 0 by hand, stopped at gate, 4 judgment calls |
| 2026-09-02 | S2 | pre-contract draft | PASS | evidence fully verified; exposed a false claim in the template example (fixed) |
| 2026-09-02 | S2 | v2 (manifest + spec-writer contract + five fields) | PASS | 2 leaves, 431 lines, source evidence re-verified, behaviors reproduced by sandbox runs; RETURN separated drift / 4 unowned candidates / 1 tree-cut finding (shared rule engine) without acting; coordinator spot-check 2/2 anchors, R2 clean |
| 2026-09-02 | S1 | v2 (manifest + expanded surfaces + checker) | PASS | tools/** at af1278db: 5 surface reports + root gates enumerated at symbol level (303 entries, 73 flag fields); 7 modules / 57 leaves; checker 10/10 zero (278 files = 262 claimed + 16 excluded); zero stated as a floor; 11 judgment calls with leanings; stopped at gate, nothing persisted into the host; coordinator re-ran the checker: errors=0 warnings=0 |
