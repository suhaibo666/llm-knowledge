---
name: source-faithful-analysis
description: >-
  Use when writing or reviewing a source-grounded analysis of a codebase, paper, spec/RFC,
  dataset, API, incident, report, or other artifact. For code, this includes an approved or
  boundary-defined repository-level software architecture overview, one concrete software
  feature, or one focused mechanism. Trigger when the result must explain structure, design
  choices, state/data/control movement, constraints, costs, or failure boundaries from a frozen
  source baseline. For a new whole codebase or unplanned multi-page domain, use
  planning-codebase-analysis first. For an exhaustive repo feature tree and function-point
  contracts, use feature-tree-analysis instead.
---

# Source-Faithful Analysis

This is the lightweight entry point for analytical documents. It routes three independent
dimensions—**source pack**, **document profile**, and **review profile**—so evidence rules are
shared while each document type keeps a precise writing contract. Load only the files selected
below; do not preload every profile.

## 1. Load the shared evidence kernel

Always read `references/source-fidelity.md` completely. It owns provenance, baseline discipline,
fact/inference separation, conflict handling, approved-boundary control, and the evidence workflow.

## 2. Select source pack(s)

Read each pack needed by the evidence actually used:

| Source | Source pack |
|---|---|
| Code/framework/library | `references/codebase.md` |
| Research paper | `references/paper.md` |
| Spec, dataset, API, incident, report, product, other | `references/general.md` |

Mixed-source work may load more than one pack. Each pack defines how to locate and verify evidence;
it does not decide the page structure.

## 3. Select exactly one primary document profile

| Deliverable | Document profile |
|---|---|
| Focused mechanism/design analysis, or paper/general artifact analysis | `references/document-types/mechanism-analysis.md` |
| One concrete software feature from primitive through system integration | `references/document-types/feature-analysis.md` |
| Approved/boundary-defined repository software architecture overview or existing architecture chapter | `references/document-types/software-architecture.md` |

Use the profile as a semantic contract, not a fixed heading or length template. For a hybrid page,
choose the profile matching its primary reader question and use links for adjacent owners; do not
merge multiple full profiles into one oversized page.

## 4. Route workflow products away

- New whole codebase or unplanned multi-page codebase domain → `planning-codebase-analysis`.
- Exhaustive repo feature tree, ownership manifest, checker reconciliation, and per-leaf contract
  specs → `feature-tree-analysis`. That workflow imports the shared evidence kernel directly; a
  function-point spec is a contract, not a feature-analysis essay.
- Wiki maintenance, math, and figures remain with their host skills.

## 5. Execute and review

Follow the shared kernel, selected source pack, and selected document profile together. If
authorized delegation is useful, also read `references/parallel-agent-contract.md`; the handoff
must name the selected profile.

For a planned wave, an independent reviewer who was never the writer applies the base
`references/page-review-rubric.md`, then the routed review profile when present:

| Document | Review profile |
|---|---|
| Mechanism or non-code analysis | base rubric only |
| Concrete software feature | `references/reviews/feature-analysis.md` |
| Software architecture | `references/reviews/software-architecture.md` |

Completion requires the selected profile's contract, evidence at the declared baseline, the
applicable review verdict, and the host repository's quality gates. Material page-boundary or
ownership drift returns to `planning-codebase-analysis` rather than being repaired locally.
