# vLLM Knowledge Domain Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the vLLM knowledge domain around design constraints, invariants, implementation mechanisms, alternatives, and failure boundaries using the frozen upstream commit, then fold the proven method back into the personal `source-faithful-analysis` skill.

**Architecture:** Preserve stable existing filenames where possible, add six missing design pages, and make each mechanism page own one design question. Treat call chains as evidence for a design claim rather than the narrative spine. Use the vLLM checkout at `d66300a1baa7779c68c7dfa4e51eee2502b48017` as the only current-code baseline.

**Tech Stack:** Markdown, Obsidian wiki links, Mermaid, Python repository checks, Git, vLLM Python/CUDA source and same-commit design docs/tests.

**Spec:** `docs/superpowers/specs/2026-08-20-vllm-knowledge-domain-redesign.md`

## Global Constraints

- Do not move the frozen vLLM baseline if `origin/main` advances.
- Do not edit the vLLM source checkout or its untracked `deepseek_v3_inference_flow.md`.
- Preserve unrelated Slime and changelog work; stage shared files selectively.
- Cite current code with exact repository-relative `file:line` locators and label analytical inference.
- Give each mechanism page a thesis, constraints/invariants, design choice, implementation evidence, rejected alternative, and cost/failure boundary.
- Keep one authoritative page per concept and use annotated Related Pages links elsewhere.
- Use `apply_patch` for authored file changes.
- Do not mark any task complete without fresh verification evidence.

---

## Task 1: Freeze Evidence and Establish Skill RED Baseline

**Files:**
- Read: `../vllm/**`
- Read: `wiki/02_engineering/03_infer_frameworks/vllm/*.md`
- Read: `C:/Users/suhaibo/.agents/skills/source-faithful-analysis/**`
- Create: `C:/Users/suhaibo/.agents/skills/source-faithful-analysis-workspace/skill-snapshot/**`
- Create: `C:/Users/suhaibo/.agents/skills/source-faithful-analysis-workspace/evals/evals.json`
- Create: `C:/Users/suhaibo/.agents/skills/source-faithful-analysis-workspace/iteration-1/**`

- [ ] Record the exact vLLM commit, branch state, and source-tree cleanliness.
- [ ] Inventory current vLLM page sizes, headings, baselines, citations, and cross-links.
- [ ] Snapshot the old skill before changing it.
- [ ] Define at least three realistic evaluation prompts covering codebase, paper/spec, and multi-page knowledge-domain analysis.
- [ ] Run the old-skill baseline inline and record whether it overuses call chains, mirrors source directories, omits invariants/alternatives, or duplicates concept ownership.
- [ ] Convert the observed failures into explicit assertions and an evaluation report.

## Task 2: Build the Shared System Model and Navigation

**Files:**
- Create: `wiki/02_engineering/03_infer_frameworks/vllm/02_vllm_system_design_principles_analysis.md`
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/index.md`

- [ ] Map TTFT, TPOT, throughput, KV capacity, CPU scheduling, launch overhead, and distributed synchronization into one constraint model.
- [ ] Define the serving control plane, engine control plane, model execution plane, and memory/communication substrate.
- [ ] Explain the system-wide bets: continuous batching, paged KV, asynchronous overlap, backend contracts, and compile/capture specialization.
- [ ] Add a problem-to-design-to-page navigation table without duplicating mechanism internals.
- [ ] Verify all initial links and leave unfinished pages visibly marked as pending until authored.

## Task 3: Rewrite KV Cache as the Exemplar Mechanism Page

**Files:**
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/12_vllm_kv_cache_management_analysis.md`

- [ ] Re-read BlockPool, KVCacheManager, coordinator/group logic, prefix caching, scheduler integration, and attention consumption at the frozen commit.
- [ ] Derive the allocation/ownership/refcount invariants and last-token recompute rule from code and tests.
- [ ] Explain why contiguous per-request tensors and naive LRU allocation fail under online dynamic batching.
- [ ] Cover hybrid KV, offload/tiering boundaries, costs, and failure behavior without absorbing disaggregated serving.
- [ ] Use this page to calibrate the common causal page contract before rewriting the rest.

## Task 4: Rewrite the Core Engine Domain

**Files:**
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/10_vllm_engine_architecture_analysis.md`
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/11_vllm_scheduler_analysis.md`
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/13_vllm_model_library_analysis.md`
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/14_vllm_attention_backends_analysis.md`
- Create: `wiki/02_engineering/03_infer_frameworks/vllm/15_vllm_model_runner_v2_analysis.md`
- Create: `wiki/02_engineering/03_infer_frameworks/vllm/16_vllm_serving_control_plane_analysis.md`

- [ ] Explain EngineCore boundaries through ownership, isolation, backpressure, and error propagation rather than entry-point chronology.
- [ ] Rebuild scheduler analysis around admission budgets, request state, running-first policy, preemption, fairness, and asynchronous commit.
- [ ] Rebuild model-library analysis around the model/weight ABI and tensor-parallel-aware layers.
- [ ] Rebuild attention analysis around the metadata contract, layout/capture capabilities, backend selection, and fallback.
- [ ] Add MRV2 analysis centered on async-first persistent rows, StagedWriteTensor, GPU-native metadata/sampling, and explicit graph lifecycle.
- [ ] Add serving-control-plane analysis for launcher, API server, AsyncLLM, DP supervision, lifecycle, routing, and backpressure.

## Task 5: Rewrite the Optimization Mechanisms

**Files:**
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/20_vllm_speculative_decoding_analysis.md`
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/21_vllm_quantization_analysis.md`
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/22_vllm_distributed_inference_analysis.md`
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/23_vllm_compilation_cudagraph_analysis.md`
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/24_vllm_fused_ops_and_kernels_analysis.md`
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/25_vllm_ir_and_fusion_passes_analysis.md`

- [ ] Tie speculative decoding to acceptance economics and distribution-preserving verification.
- [ ] Tie quantization to format/loading/kernel/hardware joint dispatch and scale semantics.
- [ ] Tie TP/PP/DP/EP/CP and DBO to rank ownership, collective ordering, lockstep, and overlap.
- [ ] Tie compile/CUDA Graph to dynamic-shape guards, address stability, graph modes, and backend capability negotiation.
- [ ] Separate kernel fusion economics from vLLM IR semantic rewriting and explain their interaction through cross-links.
- [ ] Add explicit alternatives, fallback paths, compatibility boundaries, and observability hooks to every page.

## Task 6: Add Production-Serving and Extension Pages

**Files:**
- Create: `wiki/02_engineering/03_infer_frameworks/vllm/26_vllm_disaggregated_kv_serving_analysis.md`
- Create: `wiki/02_engineering/03_infer_frameworks/vllm/27_vllm_observability_reliability_analysis.md`
- Create: `wiki/02_engineering/03_infer_frameworks/vllm/28_vllm_plugin_extension_analysis.md`

- [ ] Explain producer/consumer KV transfer, connector ownership, NIXL lease/push lifecycle, completion, timeout, and cleanup.
- [ ] Explain metrics, events, tracing, replay, process fault domains, containment, and recovery boundaries as a feedback loop.
- [ ] Explain plugin discovery, registration, two-stage initialization, endpoint task gating, IO/LoRA/platform extension points, and ABI risk.
- [ ] Keep observability/reliability and plugin lifecycle separate because they have different state owners.

## Task 7: Finish Usage, Indexing, and Cross-Domain Integration

**Files:**
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/01_vllm_feature_optimizations_guide.md`
- Modify: `wiki/02_engineering/03_infer_frameworks/vllm/index.md`
- Modify: `wiki/02_engineering/03_infer_frameworks/index.md`
- Modify: `wiki/changelog.md`

- [ ] Reduce the guide to installation, minimal serving/offline use, benchmarking, and symptom-driven tuning.
- [ ] Replace repeated mechanism explanations with short conclusions and links to authoritative pages.
- [ ] Complete navigation by reader goal and problem type; update page counts and frozen baseline metadata.
- [ ] Add 3–7 annotated Related Pages links per content page and repair reverse links.
- [ ] Add a changelog entry without overwriting concurrent changelog edits.

## Task 8: Improve `source-faithful-analysis` from the Observed Failures

**Files:**
- Modify: `C:/Users/suhaibo/.agents/skills/source-faithful-analysis/SKILL.md`
- Modify: `C:/Users/suhaibo/.agents/skills/source-faithful-analysis/references/codebase.md`
- Modify if required: `C:/Users/suhaibo/.agents/skills/source-faithful-analysis/references/general.md`
- Create: `C:/Users/suhaibo/.agents/skills/source-faithful-analysis/evals/evals.json`
- Create: `C:/Users/suhaibo/.agents/skills/source-faithful-analysis/references/knowledge-domain.md`

- [ ] Add a source-type route for multi-page knowledge domains and architecture-level documentation redesign.
- [ ] Require design-question boundaries and a concept-ownership map before file-by-file writing.
- [ ] Add a positive mechanism-page contract: thesis, constraints, model/invariants, design choice, implementation evidence, alternative, and boundary.
- [ ] Add a call-chain diagnostic: a call chain is supporting evidence, not the default outline.
- [ ] Require baseline coherence and explicit conflict handling between code, design docs, tests, and analysis inference.
- [ ] Add size/decomposition checks so large outputs split by concepts instead of source directories.
- [ ] Keep the core skill concise through progressive disclosure and avoid vLLM-specific overfitting.
- [ ] Run the same evaluation prompts against the revised skill and record the delta.

## Task 9: End-to-End Verification

**Files:**
- Verify: all files above

- [ ] Mechanically validate every `file:line` locator against the frozen vLLM checkout.
- [ ] Manually inspect at least three load-bearing locators per page.
- [ ] Audit every page for thesis, constraints/invariants, alternative, and failure boundary.
- [ ] Audit concept ownership and duplicate explanations across pages.
- [ ] Run `python tools/check_links.py --strict`.
- [ ] Run `python tools/check_math.py --changed --strict` and any required target-page math checks.
- [ ] Inspect every new Mermaid block for safe labels and semantic correctness.
- [ ] Run the full knowledge-base test suite.
- [ ] Run `git diff --check` for the knowledge repository and skill files.
- [ ] Confirm unrelated Slime work and the vLLM untracked file remain unchanged and unstaged.
- [ ] Review the final diff against the approved specification before claiming completion.
