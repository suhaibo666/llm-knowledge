# Verl What–How–Why Quality Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen the seven audited Verl pages so every load-bearing module explains its positioning, design pressure, causal mechanism, and cost or failure boundary.

**Architecture:** Keep the approved 16-page ownership map unchanged. Rewrite the overview and algorithm owner at their existing boundaries, then add targeted rationale to five mechanism pages without duplicating material owned by neighboring pages.

**Tech Stack:** Markdown, Obsidian wikilinks, Mermaid, LaTeX, repository maintenance checks.

**Spec:** `docs/superpowers/specs/2026-08-31-verl-analysis-domain-refactor-design.md`, plus the user-approved quality audit in this task.

## Global Constraints

- Preserve the frozen Verl baseline `254a23edc62f25ebfae626e3932ae285d6f86009`.
- Do not rename, split, merge, or change concept ownership of any page.
- Every new non-trivial claim must carry a verified `file:line` locator or be explicitly marked as analysis inference.
- Each key mechanism must cover: problem and positioning, alternative and design reason, causal implementation, and constraint or cost.
- Keep each `## Related Pages` list within 3–7 annotated links.
- Update `wiki/changelog.md`; do not change the Verl index because the page set and ownership remain unchanged.

---

### Task 1: Rebuild the architecture overview contract

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/verl/01_verl_architecture_overview_analysis.md`

- [x] Replace the capability inventory with a static responsibility map that states each capability's problem, positioning, input/output contract, owned state, design reason, and boundary.
- [x] Rework the connection section around controller, data, trajectory, compute/service, algorithm, publication, and recovery state transitions.
- [x] Retain the four lifecycle modes and cross-mode invariants, but make their relationship to the static capabilities explicit.
- [x] Re-read the page without owner-page links and verify the reader can still explain what each capability is, how work crosses it, and why the boundary exists.

### Task 2: Rebuild the RL algorithm explanation by design families

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/verl/15_verl_rl_algorithms_analysis.md`

- [x] Group the 14 estimators by the baseline or credit-assignment problem they solve, rather than leaving them as a registry inventory.
- [x] Group the 12 policy losses by how they constrain or reweight policy movement, and explain the closest alternative and trade-off for each family.
- [x] Preserve the verified GAE, GRPO, DRO, mask, and global-normalization mechanisms; add missing selection criteria and input preconditions.
- [x] Add a decision table that prevents estimator, loss mode, and aggregation mode from being treated as one algorithm switch.

### Task 3: Explain the default step as a dependency graph

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/verl/10_verl_end_to_end_iteration_analysis.md`

- [x] For initialization, `fit`, `_step_once`, and publication, add the state each stage consumes, produces, and why its order cannot be swapped with the obvious alternative.
- [x] Explain balance, old/ref log-prob, value, critic-first, actor update, and publication as correctness dependencies rather than a function list.
- [x] Preserve internal mechanism ownership in pages 14–23 and keep this page focused on the sync lifecycle.

### Task 4: Strengthen infrastructure and compute boundaries

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/verl/11_verl_single_controller_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/verl/12_verl_dataproto_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/verl/13_verl_workers_engine_analysis.md`

- [x] Explain why Worker and WorkerGroup are separate and why RPC shape is method metadata instead of handwritten controller proxies.
- [x] Explain why DataProto separates tensor, non-tensor, and call metadata, and why Future delays materialization without promising a transaction.
- [x] Explain why model semantics live in Engine, why registry selection is multi-dimensional, and what abstraction cost a new backend must pay.
- [x] Preserve existing state machines, guards, compatibility corrections, and failure-boundary tables.

### Task 5: Strengthen AgentLoop and RewardLoop design rationale

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/verl/18_verl_agent_loop_reward_runtime_analysis.md`

- [x] Explain why a per-trajectory coroutine is separated from LLM serving and from batch-level Ray management.
- [x] Explain the Worker/Manager/Loop hierarchy through owned state and failure isolation.
- [x] Explain when reward belongs inside the trajectory coroutine versus a colocated trainer phase, including latency and resource trade-offs.
- [x] Preserve TQ storage ownership in page 16 and algorithm ownership in page 15.

### Task 6: Record and verify the quality pass

**Files:**
- Modify: `wiki/changelog.md`

- [x] Add one concise changelog entry covering the seven strengthened pages and the unchanged ownership map.
- [x] Run `python tools/check_links.py --strict` and require all counters to be zero.
- [x] Run `python tools/check_math.py --changed --strict` and require zero errors and warnings.
- [x] Run `python -m pytest tools/` and require all 323 tests to pass.
- [ ] Run `npm run docs:test`; if the 30-second cold-start gate fails, separate environment setup latency from rendered-content failures and rerun after the pinned runtime cache is ready.
- [x] Review `git diff --check`, the seven-page diff, page lengths, and changed-file scope before completion.

**Verification note (2026-08-31):** `docs:test:unit` passed 69/69, but the Quartz smoke gate timed out while the pinned runtime was still provisioning and `docs:repair` later hung after plugin installation. The independent MkDocs content build completed successfully for 437 pages with zero broken links, missing anchors/assets/routes, or orphans. The unchecked item records the unresolved Quartz environment gate rather than a content failure.
