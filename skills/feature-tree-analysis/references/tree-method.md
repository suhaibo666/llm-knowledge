# Tree method — from entry surfaces to the feature-tree proposal

This phase produces the **feature-tree proposal package** (§6) for user approval. No spec content
and no host persistence before approval.

## 1. Entry-surface inventory (before any decomposition)

Features grow out of entry points, not out of file listings. Enumerate every surface the
repository actually has, **at symbol level**, and record where the evidence lives:

| Surface | How to enumerate | Unit of enumeration |
|---|---|---|
| CLI | executables, subcommands, argparse/click definitions, `console_scripts`, `package.json` scripts, Makefile targets | subcommand / flag group |
| Public API | `__init__` exports, `__all__`, headers, exported symbol tables, SDK manuals | exported symbol |
| Config surface | dataclass/argparse/schema field lists — AST-enumerated; the checker does this for the dataclasses listed under `surfaces.flags`, identity `Class.field` | field |
| Services / protocols | endpoint route tables, RPC service definitions, registered message types | endpoint / message type |
| Registries, plugins, hooks, callbacks | `register*` calls and decorators, plugin entry points, hook/callback names the framework accepts, extension base classes | registered name / hook |
| Background & lifecycle behavior | scheduled or periodic tasks, daemon threads, startup/shutdown handlers, signal and `atexit` handlers, env-var-triggered behavior, watchdogs | trigger |
| Data formats & external contracts | on-disk formats (checkpoints, datasets, indexes), wire formats, schema versioning and migration paths | format / version |

Use the rows this repository has; do not pad with empty ones. A repository usually has 1–3
dominant surfaces, but the long tail (a registry, a signal handler, a checkpoint format) is
exactly where trees lose leaves. Everything enumerated here goes into the manifest's `surfaces`
and is what §4 reconciles against. Symbol level matters: a file claimed by one leaf may still
carry a second, unclaimed behavior, and only a per-symbol / per-field / per-registration list
makes that visible.

## 2. Decomposition rules

- **Top level = architectural modules**: use the repo's own attested module boundaries (README
  architecture sections, build targets, package boundaries). High overlap with top-level
  directories is normal — but the justification must be "it is a block of external
  responsibility", never "it is a directory".
- **Every level below = caller-visible behavior**: a node's name answers "what it does for the
  caller", not "how the code is organized". `utils/`, `common/`, `models.py`, test directories,
  and data artifacts **never become nodes** — their contents attach to the function points they
  serve (tests go to the "tests & links" field, artifacts to "outputs").
- **Every non-leaf node carries a one-line responsibility** and is recorded in the manifest
  (§5) — the tree is data, not a drawing. A node with no children is a modeling error (T2).
- **One file ≠ one function item**: a file can carry several function points; a function point
  can span several files. Cut by behavior, not by file.
- **Reading depth is phased**: at tree stage read just enough to define nodes, claim entries, and
  finish the reconciliation; full reading of implementations happens at spec stage (the
  template's "can't finish reading → don't write" constraint applies to specs, not to tree
  building).

## 3. Leaf (function point) criteria

All four must hold before sealing a leaf:

1. **Independently triggerable**: it has its own entry — an API, a CLI flag group, a config
   option group, a message type, a registered hook. "Triggerable" is judged from the
   **repository's caller**: a helper called only internally is not a leaf; it is one step of some
   leaf's processing logic.
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

- **Tree→code**: every leaf records its ownership in the manifest (§5): stable entry anchor, owned
  files (globs), owned flags, owned entries. This is the persisted input for review and for
  baseline bumps — ownership that lives only in the analyst's head is not ownership.
- **Code→tree**: `tools/check_feature_tree.py` takes the set difference on every surface:
  - **files** — `git ls-tree` at the frozen commit, within `surfaces.files.include`; every file
    is claimed by ≥1 leaf or matched by an exclusion glob (F1 gaps, F2 phantom claims, F4 stale
    exclusions); every include glob must hit at least one file and the scope must be non-empty
    (V0) — an empty scope is never a clean reconciliation;
  - **flags** — AST-enumerated fields of the listed classes, identity **`Class.field`**; every
    field is claimed by some leaf's inputs or excluded (G1 gaps, G2 unknown claims). A bare field
    name is accepted only when it is unique across all enumerated classes; a name that exists in
    several classes must be claimed per class (G3) — same-named fields in different classes are
    different contracts until each is claimed explicitly;
  - **entries** — the symbol-level list you wrote under `surfaces.entries` (subcommands,
    endpoints, exported symbols, registrations, triggers, formats); every item is claimed or
    excluded (E1 gaps, E2 unknown claims). The checker verifies claims; it cannot discover
    entries — that is why §1 is on you.
- **A non-empty diff = the tree missed a feature or the scope statement is unclear.** Only after
  the checker reports zero (every item claimed or explicitly excluded) may the proposal be
  submitted.
- **Zero is a floor, not a proof.** File-level claims cannot see a second behavior inside a file
  that is already claimed for its first one, and the checker cannot see a surface you did not
  enumerate. Multi-behavior files are caught only by symbol-level entries and by the reviewer's
  R2 judgment; say so in the proposal instead of presenting zero as completeness.
- Every exclusion carries a one-line reason (build script / CI / third-party / data / debug
  switch / tests referenced from test anchors …); the checker rejects a reason-less exclusion
  (X1) — an unexplained exclusion is not an exclusion.

## 5. Ownership manifest (authoritative) and the leaf-row view

The manifest is the single machine-readable source of truth for the **tree** (`nodes`), the
**leaves** with their ownership and status, the **exclusions**, and the **review records**.
Host = this wiki: `docs/feature-tree/<domain>.yaml`; standalone: `<output_dir>/feature-tree.yaml`.
Relative paths inside it resolve against the manifest's own directory. The checker validates the
schema strictly (unknown keys, types, enums → X1) before anything else, so a typo cannot produce a
vacuous zero.

```yaml
domain: <name>
repo: <docs/radar/watchlist.yaml name>      # host = this wiki: checkout resolved via the watchlist
checkout: ../path/to/checkout               # or an explicit checkout (relative to this file)
commit: <full 40-hex frozen commit>          # HEAD / branch / short hash are rejected
overview: ../../wiki/.../NN_x_feature_tree_analysis.md   # optional: leaf-row table cross-checked (S2)
spec_dir: ../../wiki/.../                    # required from phase spec on: spec pages live here (S1/S3)
surfaces:
  files:
    include: ["src/**", "tools/**"]          # required; every glob must match ≥1 file (V0)
  flags:
    - {file: src/config.py, class: TrainConfig}   # AST-enumerated dataclass fields → TrainConfig.<field>
  entries:                                    # symbol-level list you enumerated in §1 (unique)
    - "src/cli.py::train"
    - "registry:optimizers/adamw"
    - "signal:SIGTERM handler"
nodes:                                        # the non-leaf tree: modules → function items → sub-functions
  - {id: train, name: Training, responsibility: "run a training job end to end", parent: null}
  - {id: train/memory, name: Memory optimization, responsibility: "trade compute for activation memory", parent: train}
leaves:
  - id: train/memory/recompute               # slug path; parent = id minus last segment must be a node (T1)
    name: activation recompute
    definition: "drop selected activations in forward and recompute them in backward"   # optional one-liner (leaf-row view)
    entry: src/transformer/block.py::TransformerBlock.forward  # stable path::symbol; F3 verifies the file
    spec: 22_memory_analysis#train-memory-recompute   # page basename[#anchor]; required once spec'd
    status: planned                            # planned | spec'd | verified
    owns:
      files: ["src/transformer/block.py", "src/recompute/**"]
      flags: [TrainConfig.recompute_granularity, recompute_method]   # qualified, or bare when unique
      entries: ["src/cli.py::train"]
exclusions:                                   # every row needs a reason (X1)
  files:
    - {glob: "tests/**", reason: "tests: referenced from leaf test anchors"}
    - {glob: "third_party/**", reason: "vendored"}
  flags:
    - {name: TrainConfig.debug_dump_every, reason: "debug-only switch"}
  entries:
    - {name: "npm:docs:test", reason: "CI aggregate of leaves gates/*"}
reviews:                                      # one row per review; a verified leaf needs a PASS row (V1)
  - {leaf: train/memory/recompute, date: 2026-09-02, reviewer: reviewer-a,
     r1: 3/3, r2: pass, r3: pass, r4: pass, r5: pass, r6: pass, verdict: PASS}
```

Rules the checker enforces on this file:

- **Tree** — `nodes` is non-empty; every node's `parent` equals its id minus the last segment
  and exists; every leaf's parent is a declared node; a node without children is an error (T1,
  T2); node and leaf ids share one namespace (D1). Ids are lowercase slug paths.
- **Flags** — identity is `Class.field`; `owns.flags` and `exclusions.flags` accept a qualified
  name, or a bare name only when it is unique across all enumerated classes (G3 otherwise).
- **Spec anchor** — the spec page named by `spec` must contain a heading whose text contains the
  leaf id (S3); the template's `### <ID> <name>` heading satisfies this.
- **Phases** — `--phase proposal` (default) checks the tree and the reconciliation;
  `--phase spec` additionally requires `spec_dir` and real pages for every `spec'd`/`verified`
  leaf; `--phase delivery` requires every leaf `verified` with a PASS review row and implies
  `--strict`. A `verified` leaf without a PASS review is an error in every phase (V1).

Run `python tools/check_feature_tree.py <manifest> --phase proposal --strict` until it reports zero.

The tree overview page's **leaf-row table is a view of the manifest**, one row per leaf, six
fixed columns; the checker fails on any ID or status that disagrees with the manifest (S2). No
unescaped `|` inside a cell (anchors in the example are placeholders):

| ID | Function point | One-line definition | Entry | Spec anchor | Status |
|---|---|---|---|---|---|
| `gates/links/scan` | wiki link health check | broken/ambiguous/bare_index verdicts for all wikilinks | `tools/check_links.py::main` | `[[spec-page#anchor]]` | planned |

- **IDs are slug paths** (`module/sub/leaf`), not numbers — inserting a leaf never renumbers
  the others;
- **Status** is three-valued: `planned` (in the tree, no spec yet) / `spec'd` (spec written) /
  `verified` (an independent review row with verdict PASS exists, or a baseline re-verification
  passed and was recorded as one);
- The Markdown tree (nested list or indented drawing) in the overview page is **rendered from
  `nodes` + `leaves`**; on any divergence the manifest wins.

## 6. The proposal package

1. **Tree**: rendered from the manifest's `nodes` (each with its one-line responsibility) and
   `leaves`, modules → … → leaves;
2. **Ownership manifest** (§5), all leaves `planned`, checker at zero in phase `proposal`;
3. **Leaf-row list** in the §5 view format;
4. **Reconciliation report**: the checker's output at zero, per surface — total / claimed /
   excluded / gap = 0;
5. **Exclusion table**: the manifest's exclusions with their reasons;
6. **Pending judgment calls** (if any): granularity or ownership questions you cannot settle
   yourself, each with your leaning — do not silently decide them for the user, and do not block
   the proposal on them either.

Plus three declarations: the frozen baseline (repo + commit + branch + date), the scope
statement, and the proposed spec wave order. Keep the whole package in the conversation, a scratch
directory, or the git-ignored `docs/superpowers/specs/` until approval; then move the manifest to
its host location. Submit, then **stop** and wait for user approval.

## 7. Re-reconciliation on a baseline bump

A delta-to-leaf mapping only sees code that already has an owner. New files, flags, endpoints,
registrations, and triggers appear as unowned items only if you re-enumerate. Procedure:

1. Copy the manifest and set `commit` to the new hash; keep the old manifest until step 5.
2. Re-enumerate §1 at the new commit: files and flags are re-enumerated by the checker
   automatically; **re-list `surfaces.entries` yourself** — new subcommands, endpoints,
   registrations, and triggers do not enumerate themselves.
3. Run the checker in phase `proposal`. F1/G1/E1 gaps are candidate new leaves or scope changes →
   tree change → approval gate when material; F2/F3/G3 mean existing ownership drifted or became
   ambiguous → fix `owns` / `entry`.
4. `git diff --name-only <old>..<new>` ∩ each leaf's `owns.files` → affected leaves → re-verify
   their specs (source anchors, boundaries, scope) → a new review row → status `verified`, or back to
   `spec'd` when the contract changed.
5. Only when the checker is at zero and every affected leaf is re-verified, replace the manifest.
   Then apply the host `kb_baseline` rule from `SKILL.md` — a subdomain tree never advances it.
