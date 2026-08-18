# slime Analysis Causal Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the complete slime analysis series so every page explains the problem background, constraints, design reasoning, implementation mechanism, rejected alternatives, and operational boundaries with verified source/community evidence.

**Architecture:** Preserve the current 21-page information architecture, calibrate the method on the Sample/DataSource page, then rewrite pages in dependency-ordered waves. Treat `11`–`17` as concept owners and use cross-links instead of duplicating their mechanisms; rebuild the index only after page theses stabilize.

**Tech Stack:** Obsidian Markdown, Mermaid, pinned GitHub source links, local slime/vime source checkouts, repository link/math validation scripts.

**Spec:** `docs/superpowers/specs/2026-08-18-slime-analysis-causal-rewrite-design.md`

## Global Constraints

- Source baseline for upstream slime pages is `THUDM/slime main@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`.
- Distinguish source facts, official/community documentation facts, and analysis inferences in the prose.
- Every non-trivial implementation claim must have a verified fixed-commit locator.
- Preserve user modifications already present in `16_slime_weight_sync_analysis.md`, `index.md`, and `wiki/changelog.md`.
- Every content page must end with 3–7 annotated links in `## Related Pages`.
- Modified formulas must use Obsidian dollar delimiters and pass strict math validation.
- New Mermaid labels must obey the repository portability rules in `CLAUDE.md`.

---

### Task 1: Calibrate the causal analysis format on Sample/DataSource

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/12_slime_sample_datasource_analysis.md`

**Interfaces:**
- Consumes: `slime/utils/types.py`, `slime/ray/rollout.py`, datasource implementations, rollout tests, official README/examples at the pinned baseline.
- Produces: the reference structure and terminology for Sample identity, DataSource lifecycle, partial continuation, compact fanout, and trainer batch ABI.

- [ ] **Step 1: Build the evidence map**

Open the definitions and all call sites for `Sample`, `DataSource`, `get_samples`, `add_samples`, partial continuation, nested flattening, rollout ids, metadata merge, and train-dict conversion. Record the exact fixed-commit lines used by each planned claim.

- [ ] **Step 2: Rewrite the opening causal chain**

Replace field-first exposition with the concrete conflict between heterogeneous rollout executions and a stable trainer tensor ABI. Add the constraints, the three-layer separation, and why raw dictionaries or one final tensor batch would break recovery, filtering, fanout, or extensibility.

- [ ] **Step 3: Re-anchor mechanisms to design decisions**

For Sample fields, DataSource lifecycle, partial response, tool/environment tokens, compact fanout, filtering, and train-dict conversion, explain which invariant each mechanism protects and what failure appears without it.

- [ ] **Step 4: Add alternatives, boundaries, and worked executions**

Include at least one normal rollout, one interrupted/continued rollout, and one nested compact fanout example. Explicitly answer whether old and new response tokens train, how metadata is merged, and how rollout id preserves execution-level grouping after flattening.

- [ ] **Step 5: Verify the calibration page**

Run `python tools/check_math.py --strict wiki/02_engineering/04_posttrain_frameworks/slime/12_slime_sample_datasource_analysis.md`, manually inspect Mermaid blocks, and spot-check at least three cited source ranges against the local checkout.

### Task 2: Rewrite the architecture, configuration, and iteration entry pages

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/01_slime_architecture_overview_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/02_slime_quickstart_and_configuration_guide.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/10_slime_end_to_end_iteration_analysis.md`

**Interfaces:**
- Consumes: the calibrated causal format from Task 1 and the top-level README/train/config entry paths.
- Produces: the series-level architecture thesis, assembly/configuration model, and versioned iteration lifecycle used as context by all later pages.

- [ ] **Step 1: Rewrite `01` around the orchestration problem**

Explain why keeping Megatron-native and SGLang-native capability forces slime to own cross-engine lifecycle, data, scheduling, and consistency contracts. Contrast the thin orchestration choice with a unified engine abstraction and state the resulting benefits and leaks.

- [ ] **Step 2: Rewrite `02` around system assembly**

Retain the runnable path, but group parameters by the subsystem contract they configure. Explain defaults, coupled flags, validation timing, resource/topology assumptions, and why a syntactically valid CLI can still describe an impossible deployment.

- [ ] **Step 3: Rewrite `10` around the iteration transaction**

Trace rollout → processing → training → weight publish as a versioned loop. Explain why phase boundaries exist, how async moves those boundaries, and which on-policy or resource invariants survive overlap.

- [ ] **Step 4: Cross-review the three entry pages**

Remove duplicate deep mechanisms owned by `11`–`17`; replace them with short design summaries and links. Confirm the three pages use the same meanings for engine, actor, sample, rollout, update, version, and colocate.

### Task 3: Rewrite the core control/data/engine pages

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/11_slime_ray_control_plane_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/13_slime_sglang_rollout_engine_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/14_slime_megatron_training_analysis.md`

**Interfaces:**
- Consumes: Sample/DataSource semantics from Task 1 and iteration boundaries from Task 2.
- Produces: authoritative definitions for Ray ownership, rollout service/request state, and Megatron actor/training execution.

- [ ] **Step 1: Deepen `11` from hierarchy to ownership design**

For group, manager, server, server group, engine, actor, and placement group, explain the resource/state each owns, when it participates, why the hierarchy is separated, and which calls are control-plane versus data-plane.

- [ ] **Step 2: Deepen `13` around concurrent request state**

Explain why SGLang is exposed as services, how manager/router/engine responsibilities prevent centralized token transport, and how partial, streaming, dynamic sampling, cancellation, and recovery preserve request identity and policy metadata.

- [ ] **Step 3: Deepen `14` around the RL-to-Megatron impedance mismatch**

Explain actor/ref/teacher/critic role reuse, batch conversion, logprob and advantage placement, pipeline execution, optimizer boundaries, and why slime wraps rather than forks Megatron training internals.

- [ ] **Step 4: Verify ownership boundaries**

Trace one concrete sample and one weight update across the three pages. Remove statements that assign the same state or responsibility to two different objects without explaining delegation.

### Task 4: Rewrite loss, synchronization, consistency, and recovery pages

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/15_slime_loss_parallelism_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/16_slime_weight_sync_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/17_slime_train_inference_consistency_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/18_slime_fault_tolerance_observability_analysis.md`

**Interfaces:**
- Consumes: training ownership from Task 3 and the pre-existing user expansion in `16`.
- Produces: authoritative statistics, weight commit, consistency taxonomy, and fault-domain models.

- [ ] **Step 1: Rewrite `15` from formulas to estimator semantics**

For each objective and reducer, state the statistical unit, mask/denominator, parallel reconstruction point, and invariance being protected. Contrast token-, sample-, and group-level normalization and connect numerical differences to concrete failure modes.

- [ ] **Step 2: Merge causal framing into the current `16` worktree content**

Preserve the existing topology, CUDA IPC, and MoE routing additions. Add the motivating train/inference topology mismatch, transaction invariants, transport alternatives, and failure consequences without undoing user edits.

- [ ] **Step 3: Rewrite `17` as a layered diagnostic model**

Separate weight, input/data, sampling distribution, routing, and kernel precision consistency. For each layer, explain why equality at the preceding layer is insufficient and identify the comparison hook or replay evidence.

- [ ] **Step 4: Rewrite `18` around fault-domain isolation**

Explain why a global transaction is impractical, how control-plane, rollout engine, data, and trainer failures receive different recovery scopes, and how debug replay, metrics, tracing, and checkpoints establish the evidence needed for safe recovery.

- [ ] **Step 5: Cross-check the four correctness pages**

Ensure “version,” “consistency,” “recovery,” “replay,” and “commit” are not used interchangeably. Add links where one page relies on another page’s invariant.

### Task 5: Rewrite backend and capability extension pages

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/19_slime_rollout_backend_extension_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/20_slime_on_policy_distillation_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/21_slime_speculative_decoding_mtp_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/22_slime_low_precision_training_rollout_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/23_slime_model_architecture_extension_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/24_slime_agent_workflow_examples_analysis.md`

**Interfaces:**
- Consumes: stable contracts defined by Tasks 1–4.
- Produces: an extension taxonomy showing which boundary each feature plugs into and which upstream invariant it must preserve.

- [ ] **Step 1: Rewrite `19` around extension boundary selection**

Differentiate external SGLang deployment, rollout request/data customization, and a genuinely new backend. Explain the minimum protocol, native capability leakage, weight-update obligations, and why a universal backend interface would either be leaky or restrictive.

- [ ] **Step 2: Rewrite `20` and `21` around role/version coupling**

For OPD, explain teacher placement and reuse of the Sample/training ABI. For MTP, explain draft/main version coupling, weight synchronization, acceptance accounting, and why a static independent draft service is unsafe for online updates.

- [ ] **Step 3: Rewrite `22` and `23` around multidimensional compatibility**

Separate training compute/storage/communication/rollout precision in `22`. In `23`, trace architecture registration, parameter-name conversion, tensor sharding, loader compatibility, and tests as one semantic mapping problem.

- [ ] **Step 4: Rewrite `24` around execution-to-training projection**

Explain why agent/tool runtimes remain in rollout, how nested logical executions become compact train fragments, how rollout ids preserve group statistics after flattening, and why tool/environment tokens are masked.

- [ ] **Step 5: Verify extension pages against their owning contracts**

Check every extension page names the upstream contract it consumes, the invariant it preserves, and the capability it cannot express. Replace repeated mechanism detail with cross-links.

### Task 6: Audit the vime/vLLM derivative as a separate baseline

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/25_vime_vllm_backend_support_analysis.md`

**Interfaces:**
- Consumes: upstream extension, rollout, synchronization, and recovery models from Tasks 3–5.
- Produces: a source-faithful fork comparison that never attributes vime-only behavior to upstream slime.

- [ ] **Step 1: Pin and verify the derivative baseline**

Locate the vime/vLLM checkout or retrieve the official repository at the commit already cited by the page. Verify README/docs/code locations before retaining any claim.

- [ ] **Step 2: Reframe the page around changed contracts**

Compare deployment topology, request protocol, PD/EPD routing, multi-model/external operation, weight synchronization, async execution, and recovery. For each area state the upstream problem, the derivative design choice, and the cost or lost compatibility.

- [ ] **Step 3: Separate source facts from project-document claims**

Mark any documented capability not found in the pinned implementation, and any implementation behavior absent from documentation. Remove zero-locator mechanism claims.

- [ ] **Step 4: Verify cross-project attribution**

Search for every use of “slime supports” and “vime supports”; confirm the subject and baseline are correct, then spot-check at least five source links.

### Task 7: Rewrite the performance and stability synthesis pages

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/30_slime_rollout_optimization_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/31_slime_posttraining_stability_analysis.md`

**Interfaces:**
- Consumes: all mechanism-owner pages.
- Produces: decision-oriented diagnosis guides grounded in the same subsystem contracts.

- [ ] **Step 1: Rewrite `30` around a capacity and critical-path model**

Explain why aggregate tok/s is insufficient, define the service/queue/long-tail/trainer waiting components, and map each optimization to the bottleneck it can actually remove. Include counterexamples where added concurrency, PD, or overlap reduces end-to-end throughput.

- [ ] **Step 2: Rewrite `31` around interacting control loops**

Separate data quality, policy version, estimator/numerical behavior, and infrastructure faults. For each symptom show the observable, likely invariant violation, intervention, and why common clipping/filtering fixes may only hide the cause.

- [ ] **Step 3: Remove duplicated owner-page explanations**

Keep these pages as synthesis/decision guides. Link to the mechanism owner for Sample, rollout, loss, synchronization, consistency, and recovery details.

### Task 8: Rebuild navigation, record provenance, and run repository validation

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/index.md`
- Modify: `wiki/changelog.md`
- Review: all 21 slime content pages.

**Interfaces:**
- Consumes: final page theses and cross-links from Tasks 1–7 plus the current user modifications in both target files.
- Produces: the final series navigation, audit trail, and repository-wide validation evidence.

- [ ] **Step 1: Merge a problem-oriented map into `index.md`**

Preserve current user edits. Add a compact table mapping system problem → responsible module/page → core design choice → principal tradeoff, and revise reading routes so readers can follow architecture, data, correctness, extension, or operations questions.

- [ ] **Step 2: Normalize cross-links and page endings**

Confirm every content page has 3–7 annotated Related Pages links, index links are path-qualified, and deep mechanisms have a single owner page.

- [ ] **Step 3: Append one consolidated changelog entry**

Preserve the current unpublished 2026-08-18 entry. Add a separate entry describing the 21-page causal rewrite, evidence baselines, major design clarifications, and validation results.

- [ ] **Step 4: Run mechanical validation**

Run `python tools/check_links.py --strict`, `python tools/check_math.py --changed --strict`, the repository Markdown/math tests discovered under `tests/`, and `git diff --check`. Fix all errors attributable to this work.

- [ ] **Step 5: Run the final source-faithfulness audit**

Spot-check at least three fixed-commit locators from every rewritten page, scan for claims using author-intent language without source support, inspect every new Mermaid block, and verify that current worktree changes in `16`, `index`, and `changelog` survived intact.

- [ ] **Step 6: Commit in reviewable batches**

Stage only the planned files. Use separate commits for calibration, core pages, extension pages, synthesis/index, and final validation fixes so each causal rewrite wave can be independently reviewed.

