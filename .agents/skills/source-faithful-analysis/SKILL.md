---
name: source-faithful-analysis
description: >-
  Deep, source-faithful analysis of ANY artifact — a codebase, a research paper, a spec/RFC/standard,
  a dataset, API/SDK docs, a running system or incident, a business/market report — turning it into a
  mechanism-level technical write-up, wiki page, design doc, or knowledge-base entry where every
  non-trivial claim is traced to its exact source locator (file:line / §·Table·Fig·Eq / clause /
  column·row / endpoint·field / log timestamp), leads with the central thesis, and explains WHY each
  design choice beats the obvious alternative — not just what it is. Use this whenever the goal is to
  understand or explain something at the mechanism/why level from its ACTUAL source rather than a
  one-line summary: "analyze X", "how does Y work internally", "write a deep dive / source-level
  walkthrough of Z", reverse-engineering an architecture, mapping a subsystem, dissecting a paper's
  design rationale, auditing a dataset/spec/report, comparing two implementations or approaches, or
  ingesting any of these into a knowledge base. After you start, read the matching
  references/<type>.md pack (codebase / paper / general) for the concrete locators, ingestion recipe,
  and essence checklist. Trigger even when the user doesn't say "skill", "analysis", or
  "documentation" — any time they want a faithful, essence-first reading of something's internals as
  opposed to merely using it or fixing one isolated bug. Prefer this over a quick surface summary
  whenever the user wants depth backed by exact, verifiable citations; especially apt for large or
  unfamiliar sources and for multi-page / multi-subsystem efforts.
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
analysis. The reader wants the *mechanism* and the *why*. For each unit (subsystem / contribution /
aspect), hunt for:

- **The thesis — the one main bet (一条主线).** The one or two sentences that name the core design
  idea ("a decoupled two-process pipeline with a busy loop"; "trade a cheaper attention for a bigger
  model + longer context"; "OS-style paging for the KV cache"). Lead with it; make every section
  serve it.
- **The *why*, not just the *what*** — ideally four beats: **motivation** (the bottleneck/failure
  that forced it) → **mechanism** (how it works, with a diagram or math) → **evidence** (the
  code/table/ablation/measurement that justifies it) → **why-not-the-obvious** (the alternative it
  was weighed against and why it lost).
- **The load-bearing specifics.** The key data structures / state machines / call chains (code); the
  ablation deltas reproduced *with their baseline column* (paper); the few rows/fields that carry the
  argument (data). Pull what argues; don't transcribe every cell or signature.
- **The non-obvious: tradeoffs, costs, edge cases, invariants, limits.** Why this design and not the
  obvious one, what it *costs*, what it deliberately does *not* do, when it breaks. This is where
  real understanding lives — and where the source is usually quietest.

If you're transcribing signatures or rewording the abstract, stop — that's reference docs, not
analysis.

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

### Phase 1 — Map before you write

Resist reading the source top-to-bottom into context. First build a map:

- **Skim the shape** — the directory skeleton (code) / section+figure+table list (paper) / TOC +
  schema (general). This reveals where the real content and the load-bearing evidence live.
- **Separate live from legacy / novel from background.** Most sources carry a deprecated path or a
  pile of preliminaries; find the 2–5 things that actually matter and focus there. Mention the rest
  only as scaffolding/contrast.
- **Locate the load-bearing entry points** — where each subsystem *starts*, the contributions
  paragraph, the key tables. Grep for the orchestrator/loop/registry, not every leaf.
- **Decompose into docs.** Organize by the source's natural seams — **subsystem** (code), **theme**
  (paper: architecture / data / training infra / post-training / eval …), or **aspect** (general) —
  and within each by depth. One concept per doc; prefer splitting over a sprawling page. **Fix the
  doc names up front** so cross-references can be written before the docs exist.

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

(For small scope, skip this — just do Phase 2 yourself.)

### Phase 4 — Integrate and verify

A pile of pages isn't a knowledge base. Tie it together:

- **Write the overview/map:** the thesis/design philosophy, a contribution/results/concepts table, a
  diagram spanning the pieces, and the cross-link web.
- **Update the host system's spine** — the parent index / TOC / changelog — matching the conventions
  already in use (e.g. a wiki's `AGENTS.md`: add the page to the domain `index.md`, append a
  changelog entry, add `[[wiki links]]` both ways).
- **Reconcile with content already on the page.** If you're ingesting into an existing overview that
  carried pre-publication estimates or stale numbers, replace them with the real values and annotate
  the change (`> [!contradiction]`, "superseded by Table 7") — don't silently leave both.
- **Render figures the way the house does.** Author each diagram as HTML/SVG and rasterize to PNG if
  that's the house style (don't ship raw mermaid where the host can't render it); **calibrate one
  first**. Eyeball every rendered figure for overflow and stray `[[…]]` that leaked into a caption.
- **Verify zero dangling references** mechanically (grep/script): every cross-link target exists, and
  spot-check that cited locators are real. Broken links and phantom citations erode trust in the
  whole set. Forward-references to *planned* sibling pages are fine — mark them as planned.

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
- Lead with the thesis (一条主线) in 1–2 sentences
- A diagram (mermaid/ASCII/SVG) + a key-concepts or contribution/results table
- (code: + a Quick Start — minimal entry point/flags + where to start reading, with file:line)
- (model paper: + a complete structure figure + an exact-hyperparameter table from the released config)

## 2..N — the bulk (per your pack)
- code: per subsystem/step — mechanism, data structures/state machines, the real call chain, dense locators
- paper: per contribution — motivation → mechanism (math/diagram) → evidence (table WITH baseline) → why-not-the-obvious
- general: per aspect — the claim, its grounding locator, the why, the tradeoff/limit

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
| Transcribing signatures / rewording the abstract / listing contributions flatly | Replace with mechanism + the *why* + the evidence + the rejected alternative. |
| Stating your inference as the source's finding | Mark it: "the source says X" vs "this implies Y". |
| Reproducing folklore / the press-release framing | Open the source; if it contradicts, the source wins — flag it. |
| Reading the whole large source "to be safe" | Map first (Phase 1), then read targeted spots per claim. |
| An evidence claim with no baseline | Reproduce the table/measurement *with its baseline*; numbers without a baseline argue nothing. |
| Dispatching writer agents with a loose prompt | Give them the strict contract; calibrate + spot-check locators on wave 1. |
| A page with no cross-references / not in the index | Integrate it (Phase 4). |

(Your source-type pack adds its own red flags — read it.)

The throughline: **if it isn't in the spot you just read, it doesn't go in the analysis.**
