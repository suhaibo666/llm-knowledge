---
name: source-faithful-analysis
description: >-
  Use when the user wants source-grounded, mechanism-level analysis of a codebase, framework,
  research paper, spec/RFC/standard, dataset, API/SDK, running system or incident, report, or other
  artifact: architecture or subsystem deep dives, “how it works internally,” design-rationale
  audits, implementation comparisons, reverse engineering, or ingestion into a wiki or knowledge
  base. Trigger when claims need exact verifiable source locators and a frozen baseline, especially
  for large or unfamiliar sources and multi-page knowledge domains. Prefer this over a surface
  summary when the request asks why a design exists, how state/data/control flows, what constraints
  it has, or how current source contradicts folklore. Do not use merely to operate a tool or fix one
  isolated bug without an analysis deliverable. Do not use as the first step for an unplanned whole-codebase or multi-page codebase-domain analysis; use planning-codebase-analysis first.
---

# Source-Faithful Analysis

A methodology for turning **any source** into accurate, mechanism-level technical analysis — a
codebase, a research paper, a spec, a dataset, an API, a running system, a report. It is portable in
both directions: the *input* can be any format, and the *output* can land in any doc system
(Markdown, Obsidian, a wiki, a design doc, an inline answer).

Two principles carry the whole method. Everything below serves them. The **concrete specifics** —
what a "locator" is, how to ingest the source, what "the essence" looks like — live in a per-type
pack you read once you know what you're analyzing (see *Pick your source-type pack*).

## Principle 1 — Be faithful to the source

Analysis that can't be traced to a spot in the source is folklore. Blogs go stale, press releases
overclaim, your training memory is a year+ old and conflates things, and sources get rewritten
between versions. So:

- **Every non-trivial claim carries an exact *locator*, and you open the source to that spot and
  read it *before* you cite it.** The locator is whatever pins a claim in *this* source: `file:line`
  (code), `§ / page / Table / Fig. / Eq.` (paper), clause № / URL+anchor (spec/web), `column·row` /
  cell (dataset), endpoint·field (API), host·timestamp·trace-id (logs). Never invent, guess, or copy
  a locator from memory or a blog. A citation you didn't verify is worse than none — it looks
  authoritative and is wrong. This single discipline is what separates this skill from "summarize
  what I remember."
- **Pin a baseline.** The exact commit/branch/date, arXiv id+**version**, doc version+date, or data
  snapshot. It goes in the header of whatever you produce — a claim is only true *at a baseline*.
- **Separate what the source *states/measures* from what you *infer*.** "The source says X (locator)"
  vs "this implies Y" (your reasoning). Don't launder inference as the source's finding.
- **When the source contradicts the folklore, the marketing, or your priors, the source wins — and
  say so out loud.** The most valuable findings are the corrections ("the V0 engine was removed",
  "DSA is continued-pretrained from dense, not trained sparse from scratch", "this flag no longer
  exists"). Surface them; don't quietly reproduce the outdated mental model.

## Principle 2 — Capture the essence (抓住重点)

A wall of "this function takes X and returns Y" — or a paraphrase of the abstract — is a summary, not
analysis. The reader wants the *mechanism* and the *why*. Two things carry that: a **thesis** for the
unit, and a fixed **narrative order** inside it.

- **The thesis — the one main bet (一条主线).** The one or two sentences that name the core design
  idea ("a decoupled two-process pipeline with a busy loop"; "trade a cheaper attention for a bigger
  model + longer context"; "OS-style paging for the KV cache"). Lead with it; make every section
  serve it.

### The five beats — the mandatory order inside every unit

**背景 → 为什么这么设计 → 实现思路与细节 → 约束 → 发展趋势（可选）.** Every unit — subsystem,
contribution, aspect — is written in this order. It is an *order*, not just a checklist: the reader
has to know **what problem existed** and **why this route won** before a single implementation detail
lands, otherwise they have no basis on which to judge the detail. A section that opens with the
mechanism has already lost that.

| # | Beat | 中文 | What must be in it | Skippable? |
|---|---|---|---|---|
| 1 | **Background** | 背景 / 问题 | The problem this unit exists to solve: the bottleneck, failure, workload or requirement that forced it, and what the previous or naive approach did. Cite where the source states the problem. | No — if you can't name the problem, you haven't understood the design |
| 2 | **Why this design** | 为什么这么设计 | The route chosen **and the obvious alternative it beat**, plus the criterion that decided it. The load-bearing beat — this is the 设计思想. If the source never says why, say *that* and mark your reconstruction as inference. | No |
| 3 | **Mechanism & detail** | 实现思路与细节 | The idea first, then the details, then the **evidence** that it does what it claims (the code path, the ablation table *with its baseline*, the measurement). | No |
| 4 | **Constraints** | 约束 / 边界 | Preconditions, invariants, what it costs, edge cases, what it deliberately does *not* do, when it breaks. The source is quietest here — dig. Never write a design up as a free win. | No |
| 5 | **Outlook** | 发展趋势 | Where this is heading: the source's own future-work line, what a newer version changed, a deprecation in flight, or what the beat-4 constraint pressures next. | **Optional** — under the fence below |

**Beat 3's load-bearing specifics differ per type:** the key data structures / state machines / call
chains (code); the ablation deltas with their baseline column (paper); the few rows/fields that carry
the argument (data). Pull what argues — don't transcribe every cell or signature.

**The outlook beat must not launder speculation as a finding.** It is the one beat that leaves the
source, so it is fenced: anchor it to something real (a future-work or limitation line *with its
locator*, a newer version's change, a TODO/deprecation, or the constraint you just wrote in beat 4)
**and** mark it explicitly as inference. No anchor → omit the beat; a unit with four beats is
complete. An unanchored trend paragraph is exactly the folklore Principle 1 exists to kill.

If you're transcribing signatures or rewording the abstract, stop — that's reference docs, not
analysis. If you're deep in the mechanism and never wrote 背景 and 为什么, stop too — that's a manual.

---

## Pick your source-type pack

Once you know what you're analyzing, read the matching pack for the concrete locator format, the
ingestion recipe, the essence checklist, and the type-specific red flags. Then run the workflow
below. The packs are the sharp edge of this skill — don't skip yours.

| Source | Read | Locator | Baseline |
|---|---|---|---|
| Code / framework / library | `references/codebase.md` | `file:line` | commit / branch / date |
| Research paper (arXiv/PDF) | `references/paper.md` | `§ / Table / Fig. / Eq.` | arXiv id + **version** |
| Anything else — spec/RFC, dataset, API docs, running system/incident, report, product | `references/general.md` | clause / column·row / endpoint / timestamp | version / snapshot / time window |

Mixed sources are common and welcome — a model paper analyzed *alongside* its released `config.json`
and repo is the norm, not the exception (the paper gives the *why*, the artifact gives the *what*).
Read both relevant packs.

---

## Codebase-wide routing boundary

For a new whole codebase or large codebase domain with no approved blueprint, stop and use `planning-codebase-analysis` first. Once that planner hands off one approved page/analysis-unit contract, use this skill to follow the required mechanism across as many source files as needed. Do not re-plan the directory, numbering, page ownership, or excluded concepts inside the page-writing task.

This boundary is codebase-specific. Paper, spec, dataset, incident, and focused one-off analyses continue to scale through the workflow below without requiring the codebase planner.

---

## The workflow

Scale it to the task. A one-spot question needs Phase 2 only; turning a whole framework or a 60-page
paper into a deep-dive series needs all of it. Don't skip Phase 0 — the baseline and a target
structure are cheap and prevent rework.

### Phase 0 — Anchor: baseline + reference style + ingest

- **Get the exact source and a way to cite it.** Record the baseline (it goes in every header). Get
  the source into a *citeable* form — see your pack: clone+grep for code; download the PDF and
  extract a page-markered text dump for a paper; load+profile a dataset; pull the log/metric window.
- **Find or define the format.** If a sibling analysis exists in the same house style, read one in
  full to match its depth, structure, citation density, and language. If none, fix the structure up
  front (see the doc template) so parallel work stays consistent.
- **Decide the granularity:** a single overview, or an overview **plus** a set of deep-dive pages.
  Big sources usually want both.
- **For an approved codebase page:** inherit granularity, title, concept ownership, exclusions, and
  evidence entry points from the blueprint. Also inherit the approved repository commit, verify
  that the checkout is at that exact commit, and keep it frozen for the page execution; do not
  reopen repository-wide page planning. Do not fetch, pull, fast-forward, switch, checkout, reset,
  or move the approved repository. If the commit is unavailable or must change, stop and return a
  concrete proposed revision to `planning-codebase-analysis` for approval. Focused code analysis
  with no approved blueprint may establish a safe baseline under the codebase pack instead.

### Phase 1 — Map before you write

Resist reading the source top-to-bottom into context. First build a map:

- **Skim the shape** — the directory skeleton (code) / section+figure+table list (paper) / TOC +
  schema (general). This reveals where the real content and the load-bearing evidence live.
- **Separate live from legacy / novel from background.** Most sources carry a deprecated path or a
  pile of preliminaries; find the 2–5 things that actually matter and focus there. Mention the rest
  only as scaffolding/contrast.
- **Locate the load-bearing entry points** — where each subsystem *starts*, the contributions
  paragraph, the key tables. Grep for the orchestrator/loop/registry, not every leaf.
- **Decompose into docs only for non-code sources or unplanned focused analysis.** Organize by the
  source's natural seams — subsystem, theme, or aspect — and within each by depth. One concept per
  doc; prefer splitting over a sprawling page, and fix doc names up front for cross-references. An
  approved codebase page may organize sections inside its assigned contract only; it must not rename,
  split, or reassign pages locally. A required page split is material drift: stop the affected page
  and return it to `planning-codebase-analysis` with the evidence and proposed boundary change.

### Architecture overviews — structure first, then motion

When the artifact is a system, framework, or multi-stage design, an overview must answer two
orthogonal questions:

1. **Static structure:** what responsibility layers/components exist; why each exists; what
   capability, input→output contract, state/invariants, and responsibility boundary each owns.
2. **Dynamic lifecycle:** how one real request/job/batch/artifact crosses those layers; when state
   becomes valid, changes owner, or becomes externally visible.

Present the static view first and the dynamic view second. A call graph or lifecycle diagram alone
shows motion, not architecture. Derive layers from stable responsibility and state ownership — not
from the directory tree, and not from an arbitrary layer count. If one diagram cannot keep both
views legible, use separate structure and lifecycle diagrams.

The layer explanation is the extension map: when responsibility and contracts are clear, a reader
can infer where a change belongs. Do not add a standalone "where to add features" catalogue unless
the requested deliverable is explicitly a developer extension guide.

**Source excerpts are evidence, not the narrative skeleton.** State the problem, design logic,
mechanism, state transition/invariant, and boundary in your own words; then attach the smallest
source excerpt or locator that proves the claim. Removing code blocks, equations, table fragments,
or quoted clauses must still leave a coherent causal explanation of why the mechanism works. A
fixed page length, layer count, or code-quotation ratio is not a quality target; split by concept
ownership, thesis, and reader load.

### Phase 2 — Locate, read, cite (the fidelity loop)

The core motion, repeated per claim:

1. **Locate** the spot — grep/search the function, the section, the table, the field.
2. **Read the targeted range** — enough to state it correctly and catch the conditions ("only with
   Muon", "at 128K", "deprecated since v2", "on the private set"). Not the whole file; not the
   abstract.
3. **Cite** the verified locator and state the mechanism *and the why* in your own words; reproduce
   load-bearing evidence (a table *with its baseline*, a state machine, a call chain).

Keep your own context lean: locate → targeted read → cite. Use the Principle-2 essence checklist (and
your pack's) as the reading agenda for each unit.

### Phase 3 — Fan out (parallel agents), when the work is big and divisible

If the analysis spans several independent units/pages, dispatch one focused subagent per page — they
run concurrently, each owns one file, no shared-state conflict. Faster, and each scope stays tight.

The catch: independent agents drift in style and **fabricate citations** if under-instructed
(inventing a "Table 4" or a line number). So give every writer agent the **strict shared contract**
in `references/parallel-agent-contract.md`: the baseline, the source slice they own, the exact doc
structure, the *read-the-spot-before-you-cite* rule, the predetermined sibling doc-names, and a
**structured return** (verified locators + thesis) so you can write the index without re-reading
everything.

Calibrate with a *real page*: the coordinator writes **one full page itself as the exemplar** (and,
if the analysis has diagrams, builds the figure toolchain and renders that page's figures), gets it
right, then points every agent at it as the template. Diagrams especially earn this — costly to redo
N times. On each finished page, spot-check 2–3 cited locators against the actual source.

For codebase work, codebase-wide wave assignment and coverage reconciliation stay with
`planning-codebase-analysis`. The page writer returns verified locators, thesis, boundary findings,
and material drift to the coordinator.

(For small scope, skip this — just do Phase 2 yourself.)

### Phase 4 — Integrate and verify

A pile of pages isn't a knowledge base. Tie it together:

- **Write the overview/map:** the thesis/design philosophy, a contribution/results/concepts table,
  the static responsibility map plus the dynamic lifecycle (separate diagrams when needed), and the
  cross-link web.
- **Update the host system's spine** — the parent index / TOC / changelog — matching the conventions
  already in use (e.g. this wiki's constitution plus its maintenance skill: add the page to the
  domain `index.md`, append a changelog entry, add `[[wiki links]]` both ways).
- **Reconcile with content already on the page.** If you're ingesting into an existing overview that
  carried pre-publication estimates or stale numbers, replace them with the real values and annotate
  the change (`> [!contradiction]`, "superseded by Table 7") — don't silently leave both.
- **Render figures the way the house does.** Author each diagram as HTML/SVG and rasterize to PNG if
  that's the house style (don't ship raw mermaid where the host can't render it); **calibrate one
  first**. Eyeball every rendered figure for overflow and stray `[[…]]` that leaked into a caption.
- **Verify zero dangling references** mechanically (grep/script): every cross-link target exists, and
  spot-check that cited locators are real. Broken links and phantom citations erode trust in the
  whole set. Forward-references to *planned* sibling pages are fine — mark them as planned.
- **Keep codebase-wide replanning with `planning-codebase-analysis`.** The page writer returns
  verified locators, thesis, boundary findings, and material drift to the coordinator.

### Phase 5 — Grow on demand

When a follow-up question exposes a gap, don't answer from memory — **go back to the source** (same
Phase-2 discipline; re-read the actual spot), then fold the answer back as a new page or section.
Each question is a chance to extend the knowledge base, and the answer is only trustworthy if it's
source-grounded like everything else.

---

## Output: the doc template

Adapt to your doc system, and **match the language** of the source/audience (analyze a Chinese-doc'd
project in Chinese). The middle section depends on the source type — see your pack.

```
# <Title> — <thesis-style subtitle, the one main bet>

> **Source baseline**: <project @ commit / arXiv id+version / doc version+date>
> **Dimension**: Overview | Deep Dive (mechanism-level)
> <2–3 sentences: what this page answers; how it relates to the overview + sibling deep-dives>

## 1. Overview
- 背景/问题 FIRST — 2–4 sentences: what problem this unit exists to solve, what the previous or
  naive approach did, why it stopped working (beat 1, at page scale)
- Then the thesis (一条主线) in 1–2 sentences — the one main bet that answers it
- For architectures: a static layer map + a dynamic lifecycle map, and a table of
  `layer | capability | input→output | owned state/boundary | evidence`; merge visuals only when
  both questions remain legible
- Otherwise: a diagram (mermaid/ASCII/SVG) + a key-concepts or contribution/results table
- (code: + a Quick Start — minimal entry point/flags + where to start reading, with file:line)
- (model paper: + a complete structure figure + an exact-hyperparameter table from the released config)

## 2..N — the bulk: one section per unit, EACH in the five-beat order
背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节（+证据）→ 约束/边界 → 发展趋势（可选，须锚定并标为推断）
- code: the beats carry the data structures/state machines + the real call chain, dense locators
- paper: the beats carry the math/diagram + the ablation table WITH its baseline
- general: the beats carry each claim + its grounding locator
- if the page covers a single topic, the five beats ARE its top-level sections

## Related / Cross-references
- overview + sibling deep-dives + adjacent topics, as [[links]]
```

Use diagrams for architecture/flows, tables for structures/ablations (always with the baseline
column), and math notation for formulas. A short, sharp page with real citations beats a long one
padded with signatures or abstract-paraphrase.

---

## Red flags — you're drifting from the method

| If you catch yourself… | Do this instead |
|---|---|
| Writing a claim with no verified locator, or one you didn't actually open | Stop, locate it, read it, then cite. If you can't find it, don't claim it. |
| Citing a locator from memory or a blog | Re-derive it from the current source; the baseline may have moved. |
| Transcribing signatures / rewording the abstract / listing contributions flatly | Rewrite as the five beats: 背景 → 为什么（含被否掉的替代）→ 实现+证据 → 约束. |
| Letting copied source carry the explanation | Write the causal mechanism in prose first; keep only the excerpt that proves a specific claim. Remove the excerpt as a test — the explanation must still stand. |
| Calling a call graph or lifecycle diagram the architecture overview | Add the static responsibility/state-ownership view first, then use the lifecycle to show how work crosses it. |
| Opening a section with the mechanism — "this class does X", "the method is defined as" | Restructure: 背景 first, then 为什么. If you can't state the problem it solves, you don't yet understand it — go back to the source (the commit/PR, the § intro). |
| A unit that reads as a free win — no 约束 | Every design pays something. Hunt the guards, error branches, Limitations §, the conditions in captions; state the cost and when it breaks. |
| A 发展趋势 paragraph spun from your priors | Anchor it (future-work line + locator / newer version / TODO / the beat-4 constraint) and mark it as inference — or delete the beat. |
| Stating your inference as the source's finding | Mark it: "the source says X" vs "this implies Y". |
| Reproducing folklore / the press-release framing | Open the source; if it contradicts, the source wins — flag it. |
| Reading the whole large source "to be safe" | Map first (Phase 1), then read targeted spots per claim. |
| An evidence claim with no baseline | Reproduce the table/measurement *with its baseline*; numbers without a baseline argue nothing. |
| Dispatching writer agents with a loose prompt | Give them the strict contract; calibrate + spot-check locators on wave 1. |
| A page with no cross-references / not in the index | Integrate it (Phase 4). |

(Your source-type pack adds its own red flags — read it.)

The throughline: **if it isn't in the spot you just read, it doesn't go in the analysis.**
