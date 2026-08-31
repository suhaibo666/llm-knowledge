# Verl Call-Chain Deepening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild seven Verl analysis pages so readers can follow real source symbols, data mutations, remote/async boundaries, and completion states instead of stopping at component summaries.

**Architecture:** Keep the approved page tree and concept ownership unchanged. Establish page 10 as the trace exemplar, deepen owner pages along control/compute/data/trajectory seams, then regenerate page 01's top-level lifecycle from the verified owner traces.

**Tech Stack:** Markdown, Obsidian wikilinks, Mermaid, LaTeX, PowerShell/rg, repository validation tools.

**Spec:** `docs/superpowers/specs/2026-08-31-verl-call-chain-deepening-design.md`

## Global Constraints

- Freeze the Verl source checkout at `254a23edc62f25ebfae626e3932ae285d6f86009` throughout execution.
- Never modify, fetch, switch, reset, or clean `E:/97-codes/torch_parallel/verl`; preserve its untracked `GRPO_Analysis.md`.
- Modify only pages 01/10/11/12/13/15/18, `wiki/changelog.md`, and this implementation plan.
- Do not change page names, page ownership, indexes, or the repository radar baseline.
- Every cross-file call hop needs separately verified caller and callee locators.
- Every trace records input/pre-state, owner/action, output/post-state, and local/remote/async/blocking semantics.
- Keep each page below 500 lines and each `## Related Pages` list at 3–7 annotated links.
- Use `【分析推断】` for design rationale not stated by source or history.

---

### Task 1: Rebuild page 10 as the call-chain exemplar

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/verl/10_verl_end_to_end_iteration_analysis.md`
- Read: `verl/trainer/main_ppo.py`
- Read: `verl/trainer/ppo/v1/trainer_base.py`
- Read: `verl/trainer/ppo/v1/trainer_sync.py`
- Read: `verl/trainer/ppo/v1/agent_loop_tq.py`
- Read: `verl/single_controller/base/worker_group.py`
- Read: `verl/single_controller/base/decorator.py`
- Read: `verl/single_controller/ray/base.py`
- Read: `verl/utils/transferqueue_utils.py`
- Read: `verl/workers/engine_workers.py`
- Read: `verl/workers/engine/base.py`
- Read: `verl/checkpoint_engine/base.py`

**Interfaces:**
- Consumes: approved sync lifecycle ownership from the design spec.
- Produces: the prose, call-chain block, and state-ledger format reused by Tasks 2–6.

- [x] **Step 1: Verify the four source traces**

Open and record the exact caller/callee ranges for:

```text
TaskRunnerV1.run → trainer.init → AgentLoopManagerTQ.create → trainer.fit
step → prepare_step → _submit_batch_to_rollout → AgentLoopManagerTQ.generate_sequences
_compute_old_log_prob → WorkerGroup wrapper → tqbridge → ActorRolloutRefWorker → TrainingWorker → Engine
_update_actor → train_mini_batch/train_batch → Engine optimizer → on_step_end → CheckpointEngine publication
```

Expected: every arrow has a verified caller line and callee definition; no range is copied from the existing page without reopening it.

- [x] **Step 2: Replace stage summaries with trace sections**

For each trace add:

```markdown
#### 调用链

```text
ExactClass.method (file:line)
  → ExactClass.method (file:line)
```

| Hop | 输入/前态 | 动作与 owner | 输出/后态 | 执行语义 |
|---|---|---|---|---|
```

Keep the nine-stage table only as navigation and move the causal detail into the trace ledgers.

- [x] **Step 3: Add branch and completion semantics**

Explicitly cover bypass vs recomputed old log-prob, independent vs colocated reward, conditional critic/ref paths, `blocking=False`/future behavior, TQ cleanup, and the difference among step return, checkpoint save, actor update, publication completion, and next-request visibility.

- [x] **Step 4: Run the page-level review**

Run:

```powershell
rg -n "TaskRunnerV1\.run|AgentLoopManagerTQ|_compute_old_log_prob|_update_actor|CheckpointEngine|执行语义|输入/前态" wiki/02_engineering/04_posttrain_frameworks/verl/10_verl_end_to_end_iteration_analysis.md
python tools/check_links.py --strict
git diff --check -- wiki/02_engineering/04_posttrain_frameworks/verl/10_verl_end_to_end_iteration_analysis.md
```

Expected: all trace anchors are present; links and diff checks pass.

---

### Task 2: Deepen the single-controller RPC path

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/verl/11_verl_single_controller_analysis.md`
- Read: `verl/single_controller/base/decorator.py`
- Read: `verl/single_controller/base/worker_group.py`
- Read: `verl/single_controller/ray/base.py`
- Read: `verl/protocol.py`
- Read: `verl/workers/engine_workers.py`

**Interfaces:**
- Consumes: the concrete `actor_rollout_wg.compute_log_prob` boundary established in Task 1.
- Produces: the authoritative dynamic-binding, dispatch, remote execution, collect, and future trace.

- [x] **Step 1: Trace metadata write and consumption**

Verify how `@register` wraps the worker method with `tqbridge`, writes dispatch/execute/blocking metadata, and how `_bind_worker_method` consumes those attributes to call Ray `func_generator`.

- [x] **Step 2: Write one concrete group-call ledger**

Trace controller `actor_rollout_wg.compute_log_prob(batch)` through dynamic binding, DP mesh dispatch, per-handle remote calls, worker execution, and collect. State which object exists at controller, Ray boundary, worker entry, and return.

- [x] **Step 3: Write the non-blocking future trace**

Follow `blocking=False` into Ray refs/`DataProtoFuture`, identify the first actual `ray.get`, and explain partial side effects and lack of rollback.

- [x] **Step 4: Verify page 11**

Run:

```powershell
rg -n "compute_log_prob|_bind_worker_method|func_generator|tqbridge|DataProtoFuture|ray\.get" wiki/02_engineering/04_posttrain_frameworks/verl/11_verl_single_controller_analysis.md
git diff --check -- wiki/02_engineering/04_posttrain_frameworks/verl/11_verl_single_controller_analysis.md
```

Expected: one actual call can be followed from controller to materialized result.

---

### Task 3: Deepen Worker/Engine and algorithm execution

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/verl/13_verl_workers_engine_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/verl/15_verl_rl_algorithms_analysis.md`
- Read: `verl/workers/engine_workers.py`
- Read: `verl/workers/engine/base.py`
- Read: `verl/workers/utils/losses.py`
- Read: `verl/trainer/ppo/core_algos.py`
- Read: `verl/trainer/ppo/v1/utils.py`
- Read: `verl/trainer/ppo/v1/trainer_base.py`

**Interfaces:**
- Consumes: page 11's RPC boundary and page 10's old-log/update calls.
- Produces: the compute owner trace and the algorithm-to-gradient trace.

- [x] **Step 1: Trace Engine registry resolution**

Record the actual lookup order for `model_type`, `backend`, detected/overridden device, vendor-specific key, device-only key, CUDA/NVIDIA fallback, and failure.

- [x] **Step 2: Write page 13 infer/train/export traces**

Cover:

```text
ActorRolloutRefWorker.compute_log_prob → TrainingWorker.infer_batch → Engine.eval_mode/infer_batch
ActorRolloutRefWorker.update_actor → TrainingWorker.train_mini_batch → train_batch → Engine.train_batch
actor export branch → Engine full/shard/delta API → CheckpointEngine boundary
```

Record mini-batch/epoch iteration, zero-grad/forward-backward/optimizer-step, metric aggregation, and which rank returns output.

- [x] **Step 3: Write page 15 advantage dispatch trace**

Follow TQ field selection, padded DataProto construction, KL/rollout correction, V1 multi-trajectory handling, estimator registry lookup, estimator output, nested conversion, and TQ writeback.

- [x] **Step 4: Write page 15 policy-loss-to-backward trace**

Follow actor update into `actor_loss`, `get_policy_loss_fn`, concrete loss, entropy/reference KL, `agg_loss`, Engine backward, and optimizer step. Record tensor shape/mask transitions and global denominator inputs.

- [x] **Step 5: Verify pages 13 and 15**

Run:

```powershell
rg -n "TrainingWorker\.(infer_batch|train_batch)|EngineRegistry|get_policy_loss_fn|get_adv_estimator_fn|optimizer_step|写回" wiki/02_engineering/04_posttrain_frameworks/verl/13_verl_workers_engine_analysis.md wiki/02_engineering/04_posttrain_frameworks/verl/15_verl_rl_algorithms_analysis.md
python tools/check_math.py --changed --strict
git diff --check -- wiki/02_engineering/04_posttrain_frameworks/verl/13_verl_workers_engine_analysis.md wiki/02_engineering/04_posttrain_frameworks/verl/15_verl_rl_algorithms_analysis.md
```

Expected: formula pages remain clean and both execution traces reach Engine/backward boundaries.

---

### Task 4: Deepen DataProto state transformations

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/verl/12_verl_dataproto_analysis.md`
- Read: `verl/protocol.py`
- Read: `verl/trainer/ppo/ray_trainer.py`
- Read: `verl/single_controller/base/decorator.py`

**Interfaces:**
- Consumes: page 11 dispatch/collect semantics.
- Produces: the local batch identity and future materialization trace used by page 01.

- [x] **Step 1: Find one real DataProto lifecycle**

Locate a current V0/local caller that constructs or receives a DataProto, selects/reorders/repeats or chunks it, merges worker output, and eventually consumes the result.

- [x] **Step 2: Write the three-container mutation ledger**

For construct, select/reorder, chunk, concat, and union, record how `batch`, `non_tensor_batch`, and `meta_info` change and which invariant is checked.

- [x] **Step 3: Write the Future materialization ledger**

Trace Ray refs → collect function → concat → optional dispatch → materialized output, and explain why operations on the driver are restricted before `get()`.

- [x] **Step 4: Verify page 12**

Run:

```powershell
rg -n "DataProtoFuture|select_idxs|reorder|chunk|concat|union|meta_info|输入/前态" wiki/02_engineering/04_posttrain_frameworks/verl/12_verl_dataproto_analysis.md
git diff --check -- wiki/02_engineering/04_posttrain_frameworks/verl/12_verl_dataproto_analysis.md
```

Expected: the page reads as a data-state walkthrough rather than an API inventory.

---

### Task 5: Deepen AgentLoop and RewardLoop coroutine paths

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/verl/18_verl_agent_loop_reward_runtime_analysis.md`
- Read: `verl/experimental/agent_loop/agent_loop.py`
- Read: `verl/experimental/agent_loop/tool_agent_loop.py`
- Read: `verl/experimental/reward_loop/reward_loop.py`
- Read: `verl/trainer/ppo/v1/agent_loop_tq.py`

**Interfaces:**
- Consumes: page 10's prompt submission boundary.
- Produces: authoritative ordinary-gather, V1 fire-and-forget, session status, postprocess, and reward traces.

- [x] **Step 1: Read the Mermaid skill before changing the sequence diagram**

Read `skills/writing-mermaid-diagrams/SKILL.md` completely and follow its parser rules.

- [x] **Step 2: Replace abstract participants/messages with real symbols**

The diagram and adjacent chain must distinguish:

```text
AgentLoopManager.generate_sequences → Ray workers → asyncio.gather
AgentLoopManagerTQ.generate_sequences → AgentLoopWorkerTQ.generate_sequences → background _run_prompt
```

- [x] **Step 3: Trace per-prompt and per-session state**

Follow `running` tag, rollout.n task creation, `_run_agent_loop`, concrete single/tool loop `run`, postprocess, trajectory key writes, sibling settlement, and `finished/failure` publication.

- [x] **Step 4: Trace both reward branches**

Cover async reward handles inside the trajectory path and colocated Trainer batch reward, including final-output assignment, sibling broadcast, padding/unpadding, and extra-info writeback.

- [x] **Step 5: Verify page 18 and Mermaid syntax**

Run the Mermaid validation commands required by the project skill, then:

```powershell
rg -n "AgentLoopManagerTQ|AgentLoopWorkerTQ|_run_prompt|_run_agent_loop|running|finished|failure|compute_score" wiki/02_engineering/04_posttrain_frameworks/verl/18_verl_agent_loop_reward_runtime_analysis.md
git diff --check -- wiki/02_engineering/04_posttrain_frameworks/verl/18_verl_agent_loop_reward_runtime_analysis.md
```

Expected: a reader can follow one prompt and all rollout.n siblings to terminal group state.

---

### Task 6: Rebuild the overview lifecycle from verified traces

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/verl/01_verl_architecture_overview_analysis.md`

**Interfaces:**
- Consumes: verified traces from Tasks 1–5 and existing owner links.
- Produces: one real top-level lifecycle without duplicating owner internals.

- [x] **Step 1: Write one actual-symbol top-level chain**

Trace from `TaskRunnerV1.run` through prompt generation, TQ terminal state, ReplayBuffer admission, old/ref/value/advantage/update, and CheckpointEngine publication to next-request visibility.

- [x] **Step 2: Add an owner handoff ledger**

For every top-level hop state the object crossing the boundary, old owner, new owner, and completion signal. Link detailed hops to pages 10–18/21 rather than copying their internals.

- [x] **Step 3: Verify page 01**

Run:

```powershell
rg -n "TaskRunnerV1\.run|AgentLoopManagerTQ|ReplayBuffer|_compute_advantage|CheckpointEngine|完成信号" wiki/02_engineering/04_posttrain_frameworks/verl/01_verl_architecture_overview_analysis.md
git diff --check -- wiki/02_engineering/04_posttrain_frameworks/verl/01_verl_architecture_overview_analysis.md
```

Expected: the overview has a real lifecycle while keeping owner internals in their pages.

---

### Task 7: Integrate, record, and verify

**Files:**
- Modify: `wiki/changelog.md`
- Modify: `docs/superpowers/plans/2026-08-31-verl-call-chain-deepening.md`

**Interfaces:**
- Consumes: all seven completed pages.
- Produces: repository-ready change set and evidence report.

- [x] **Step 1: Add the changelog entry**

Record the seven-page call-chain deepening, representative traces, unchanged page tree/baseline, and any corrected factual sequence discovered during source tracing.

- [x] **Step 2: Run semantic scope checks**

Run a per-page trace audit and verify:

```powershell
rg -n "调用链|输入/前态|输出/后态|执行语义|完成" wiki/02_engineering/04_posttrain_frameworks/verl/{01,10,11,12,13,15,18}_*.md
```

On PowerShell, expand the exact seven paths rather than relying on Bash brace expansion.

- [x] **Step 3: Run full quality gates**

Run:

```powershell
python tools/check_links.py --strict
python tools/check_math.py --changed --strict
python -m pytest tools/
npm run docs:test
git diff --check
```

Expected: all repository gates pass. If Quartz cold-start provisioning fails again, record the exact failure and run the independent MkDocs content build; do not report the Quartz smoke gate as passed.

Result (2026-08-31): strict links, changed-file math, 323 pytest cases, the 336-block Mermaid corpus, and the independent MkDocs build all passed. `npm run docs:test` passed its 69 unit cases but Quartz did not become ready within 30 seconds, so that command exited 1; the fallback MkDocs build rendered 437 pages with zero broken links, missing anchors, missing assets, missing legacy routes, or orphans.

- [x] **Step 4: Review final scope and page sizes**

Verify only the approved nine paths changed, each page is below 500 lines, every Related Pages list has 3–7 entries, and all new locators exist within the frozen source checkout.

- [x] **Step 5: Stop before commit/push**

Report the completed diff and verification evidence. Commit or push only after an explicit user request.
