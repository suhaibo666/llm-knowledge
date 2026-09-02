# Tree method — from entry surfaces to the feature-tree proposal

This phase produces the **feature-tree proposal** package for user approval. No spec content
before approval.

## 1. Entry-surface inventory (before any decomposition)

Features grow out of entry points, not out of file listings. Enumerate every surface and record
where the evidence lives:

| Surface | How to enumerate |
|---|---|
| CLI | executables, subcommands, argparse/click definitions, `console_scripts`, `package.json` scripts, Makefile targets |
| Public API | `__init__` exports, `__all__`, headers, exported symbol tables, SDK manuals |
| Config surface | dataclass/argparse/schema field lists (AST enumeration; see `tools/check_coverage.py` for the approach) |
| Services/protocols | endpoint route tables, RPC service definitions, registered message types |

A repository usually has 1–3 dominant entry surfaces; record them all — they double as the
mechanical enumeration surfaces for the reconciliation in §4.

## 2. Decomposition rules

- **Top level = architectural modules**: use the repo's own attested module boundaries (README
  architecture sections, build targets, package boundaries). High overlap with top-level
  directories is normal — but the justification must be "it is a block of external
  responsibility", never "it is a directory".
- **Every level below = caller-visible behavior**: a node's name answers "what it does for the
  caller", not "how the code is organized". `utils/`, `common/`, `models.py`, test directories,
  and data artifacts **never become nodes** — their contents attach to the function points they
  serve (tests go to the "tests & links" field, artifacts to "outputs").
- **One file ≠ one function item**: a file can carry several function points; a function point
  can span several files. Cut by behavior, not by file.
- **Reading depth is phased**: at tree stage read just enough to define nodes, claim entries, and
  finish the reconciliation; full reading of implementations happens at spec stage (the
  template's "can't finish reading → don't write" constraint applies to specs, not to tree
  building).

## 3. Leaf (function point) criteria

All four must hold before sealing a leaf:

1. **Independently triggerable**: it has its own entry — an API, a CLI flag group, a config
   option group, a message type. "Triggerable" is judged from the **repository's caller**: a
   helper called only internally is not a leaf; it is one step of some leaf's processing logic.
2. **Closable contract**: input→output can be stated without "depends on which sub-case"; if it
   cannot, keep splitting.
3. **Single behavior**: if the processing logic contains "with config A it takes an entirely
   different path with its own input domain" → split.
4. **Testable in principle**: a targeted test group could exist for it (even if none does today).

**Reverse merge**: two candidate leaves that always co-occur, share one contract, and have no
independent switch → merge into one function point.

Typical depth is 3–5 levels; no quota on levels or leaf count. More leaves is not better
coverage — a hundred internal-function leaves are worth less than a few dozen contract leaves
plus a zeroed reconciliation.

## 4. Two-way reconciliation (MECE, mechanized)

- **Tree→code**: every leaf declares code ownership (entry locator + main implementation extent:
  files or directory ranges).
- **Code→tree**: take the set difference on every enumeration surface recorded in §1:
  - **Source file list**: `git ls-files` at the frozen commit (or an equivalent), minus
    tests/vendored/generated files; every remaining file maps to ≥1 node or goes into the
    exclusion table;
  - **Config flag surface**: every field maps to some leaf's "inputs" field or is excluded
    (e.g. debug-only switches);
  - **Entry surface**: every subcommand/endpoint/exported symbol is claimed by some leaf.
- **A non-empty diff = the tree missed a feature or the scope statement is unclear.** Only after
  the diff is zeroed (claimed or explicitly excluded) may the proposal be submitted.
- Every exclusion-table row carries a one-line reason (build script / CI / third-party / data /
  debug switch …); an unexplained exclusion is not an exclusion.

## 5. Leaf-row format (the authoritative table in the tree overview; parsing surface reserved for the future checker)

One row per leaf, six fixed columns (locators in the example are placeholders):

| ID | Function point | One-line definition | Entry | Spec anchor | Status |
|---|---|---|---|---|---|
| `gates/links/scan` | wiki link health check | broken/ambiguous/bare_index verdicts for all wikilinks | `<check_links.py:L?>` | `[[spec-page#anchor]]` | planned |

- **IDs are slug paths** (`module/sub/leaf`, lowercase, hyphens), not numbers — inserting a new
  leaf never renumbers the others;
- **Status** is three-valued: `planned` (in the tree, no spec yet) / `spec'd` (spec written) /
  `verified` (review or baseline re-verification passed);
- An indented tree drawing may exist, but **the leaf-row table is authoritative**; the drawing is
  a view of it, and on divergence the table wins.

## 6. The proposal package

1. **Tree**: modules → … → leaves, with a one-line responsibility per non-leaf node;
2. **Leaf-row list**: §5 format, all `planned`;
3. **Reconciliation diff report**: per surface — total / claimed / excluded / diff = 0;
4. **Exclusion table**: a one-line reason each;
5. **Pending judgment calls** (if any): granularity or ownership questions you cannot settle
   yourself, each with your leaning — do not silently decide them for the user, and do not block
   the proposal on them either.

Plus three declarations: the frozen baseline (repo + commit + branch + date), the scope
statement, and the proposed spec wave order. Submit, then **stop** and wait for user approval.
