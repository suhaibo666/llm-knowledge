---
name: planning-codebase-analysis
description: >-
  Use when a user provides a new whole codebase or large codebase domain and wants its architecture,
  modules, core mechanisms, or a multi-page knowledge-base analysis planned before writing begins.
  Also use for codebase-domain-level replanning when page ownership, coverage, directory placement,
  or implementation order is not yet agreed. Do not use for one focused mechanism, one approved page
  contract, ordinary Wiki maintenance, or operating and fixing code without an analysis deliverable.
---

# Planning Codebase Analysis

## Responsibility boundary
- Own whole-codebase reconnaissance, content selection, page ownership, user approval, and coverage.
- Treat one page as one analysis unit, not one source file.
- Route focused or approved units to `source-faithful-analysis`.

## Hard gate
**NO PERSISTED MUTATION BEFORE USER APPROVES THE BLUEPRINT.**
Before approval, do not write a page body, stub, or frontmatter; create, rename, or rewrite a page
or directory; update an index or changelog; or save a persisted plan/spec. Read-only repository and
existing-Wiki inspection is allowed before approval. Workflow persistence begins only after approval.

## Workflow
1. Anchor the repository, target audience, existing Wiki context, branch, commit, and date.
2. Build the discovery map from capabilities first, then static architecture, representative lifecycles, state owners, core mechanisms, extension boundaries, engineering constraints, and live/legacy status.
3. Convert the discovery map into the blueprint contract below.
4. Present the blueprint and stop for explicit user approval.
5. Persist an approved multi-page blueprint under `docs/superpowers/specs/` when it must survive multiple pages or sessions.
6. Dispatch one approved page contract at a time to `source-faithful-analysis`; invoke
   `maintaining-llm-knowledge`, `writing-obsidian-math`, `drawing-wiki-figures`, and
   `writing-mermaid-diagrams` only when their concrete work appears.
7. Update the coverage matrix after each wave and reopen approval only for material drift.

## Blueprint contract
- Record a capability map, static architecture, and representative dynamic lifecycle evidence before assigning pages.
- Repository baseline, audience, Wiki placement, system thesis, live/legacy boundary, unresolved evidence gaps.
- Capability map, static responsibility/state-ownership map, and one or more representative dynamic lifecycles.
- Per-page table: path/title, page type, thesis, reader question, owned concepts, explicit exclusions,
  core mechanisms, evidence entry points, approved repository commit, dependencies, visual candidates,
  and completion test.
- Coverage matrix: each capability/lifecycle/mechanism has one authoritative page, permitted summaries elsewhere, and planned/covered/gap state.
- Implementation order based on conceptual dependencies, not filename order.

## Planning rules
- Derive content from user-visible capabilities and mechanism ownership, never directly from directories.
- Use discovery dimensions as prompts, not a mandatory taxonomy.
- Do not set fixed layer, page, line, code-block, or numbering quotas.
- Do not use a fixed source-code or code-to-explanation ratio as a quality or completion constraint.
- Fit the existing Wiki functional tree; propose a new directory only when no existing functional owner fits.
- Number by local convention and reading dependency; do not assign `01` merely because a feature is prominent.
- Existing related pages require a reuse/rewrite/merge decision before new pages are proposed.

## Replanning gate
Reconfirm only for material contract drift: an authoritative/core concept ownership change that
materially alters page boundaries or coverage; page groups are added/removed; a Wiki directory must
be created/moved; the approved source baseline changes; source evidence overturns the system
thesis/module split; or the audience/scope/deliverable expands. Ordinary wording and link-only edits
remain local.

## Completion gate
Completion means every planned core capability, representative lifecycle, state owner, and mechanism has one authoritative page; duplicates and gaps are resolved; source and Wiki gates pass. Page count and length are not completion evidence.

## Red flags
| Rationalization | Required response |
|---|---|
| “The directory layout is obvious; I can plan from it.” | Build the capability and ownership maps first. |
| “I can draft the overview while the user reviews the plan.” | Stop; drafting Wiki body content crosses the approval gate. |
| “More pages means better coverage.” | Use the coverage matrix, not page count. |
| “A writer found a better page split and can continue.” | Return material ownership changes to the coordinator and user. |
| “The user said ‘continue’ or ‘don’t ask,’ so I can revise ownership myself.” | A generic continue/don’t-ask instruction does not approve a material ownership revision that has not been presented. Moving an indivisible mechanism from split or ambiguous ownership to one authoritative page plus linked summaries is material even when page count and cross-links stay unchanged; calling it a “coupling supplement” does not make it immaterial. Stop and present the concrete revised page boundary plus coverage matrix for explicit approval before writing; ordinary wording or link-only edits remain immaterial. |
| “One source file should become one document.” | Define one causal analysis unit and follow it across files. |
