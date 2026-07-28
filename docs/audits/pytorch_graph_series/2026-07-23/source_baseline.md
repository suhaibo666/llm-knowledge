# PyTorch Graph Series Audit Baseline

- Current audit source: `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
- Branch reference: local `origin/main`
- Commit date: 2026-07-23
- Source worktree: `E:/97-codes/torch_parallel/p`
- Checkout state at final verification: detached `HEAD`, clean, with no tracked or untracked
  changes.
- Previous report baseline: `ea5655fcebf726ec4cf1a859de75d2d0e6425805`
- Historical wiki claims without a declared commit start as `unresolved`.
- Other PyTorch checkouts in the workspace were not used as evidence and were not modified.
- `inventory.jsonl` is regenerated from the final 28-page legacy manifest after navigation and
  correction callouts are frozen. The exact frozen page hashes and per-claim decision-file hashes
  are recorded in
  `docs/audits/pytorch_graph_series/2026-07-27/legacy_claim_closure.json`.

## Baseline policy

The detached worktree is the source of truth for current behavior in this audit. The course-claim
gate rejects a source checkout whose `HEAD` differs from the pinned commit or whose worktree is
dirty. A historical wiki claim may be marked `verified-historical` only when its original commit or
another exact version can be established and inspected. Merely finding a similarly named symbol in
the current source does not verify the old claim.
