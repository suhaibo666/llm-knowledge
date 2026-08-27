# Parallel writer-agent contract (template)

Read this when you're about to fan out (Phase 3): dispatch one subagent per independent page so they
run concurrently. Each agent reads the same read-only source and drafts exactly one page. The risks
with independent agents are **style drift** and **fabricated citations** (inventing a "Table 4", a
line number, a clause that isn't there) — this contract neutralizes both.

How to use: fill the `<…>` placeholders and paste the whole thing as the agent's prompt, once per
page. Parts in **【bold brackets】** are constant across a batch; **Scope** is per-agent. Bracketed
`{code | paper | general}` notes pick the wording for your source type.

---

## The contract (paste per agent)

```
You are a senior analyst writing ONE source-faithful analysis page. The source is read-only ground
truth — never modify it, never invent results.

【Source baseline】<project @ commit (branch, date) | arXiv id+version (title, authors, date) | doc
version+date>. Source at: <repo root | page-markered .txt dump | url/dataset path>. Read your slice
by <grep the entry file | Grep "===== PAGE <X> ====="> ; your slice is <subsystem | PAGE X–Y | clause
range>. Focus on <this unit>; treat the rest only as context (a sibling page owns it).

【Your page】Write only this one file (do not touch any other file): <abs path to output page>.

【Scope — what to cover】
<the specific unit; the exact locators to read (files/dirs | §/Tables/Figures | clauses/columns); the
items that MUST be explained. Be concrete — and per type:
  code → the entry file, the key classes, the call chain to trace, AND where to find why this path
         exists (the introducing commit/PR, the design comment) — beat 1–2 material;
  paper → the contributions to cover, and for each: the § that states the problem, the alternative it
         was weighed against, the ablation table that justifies it;
  general → the aspects to cover, and for each the clause / column / window that grounds it.
Whatever the type, every item is written up in the FIVE BEATS below — the scope fixes WHAT, the
beats fix HOW.>

【Page structure — exactly these sections】
- Header block: title (+ a thesis-style subtitle) + a blockquote with the source baseline (verbatim
  above) + 2–3 sentences on what this page answers and how it relates to the sibling pages <list>.
- "## 1. Overview" — open with 背景/问题 (2–4 sentences: what problem this unit exists to solve, what
  the previous or naive approach did) → then the thesis → a key-concepts/contribution/results table +
  the key diagram. {code: + a Quick Start: minimal entry point + where to start reading, with file:line.}
- "## 2..N" — the bulk: one section per unit, EACH written in these FIVE BEATS, in this order:
  ① 背景/问题 — the bottleneck, failure or requirement that forced this, and what the previous or
     naive approach did. Cite the spot where the source states the problem. Never open with the
     mechanism.
  ② 为什么这么设计 — the route chosen AND the obvious alternative it beat, plus the criterion that
     decided it. Mandatory. If the source never says why, write that it is silent and mark your
     reconstruction as inference — do not present it as the source's stated rationale.
  ③ 实现思路与细节 — the idea first, then the details, then the evidence. {code: data structures /
     state machines + the real call chain, dense file:line. paper: math/diagram + the ablation table
     WITH its baseline column. general: every claim with its grounding locator.}
  ④ 约束/边界 — preconditions, invariants, the cost it pays, edge cases, what it deliberately does NOT
     do, when it breaks. Mandatory: never write a unit up as a free win. {code: guards, error
     branches, the tests. paper: Limitations § + the conditions in captions.}
  ⑤ 发展趋势 — OPTIONAL, and fenced: include it only if you can anchor it to something real (a
     future-work/limitation line with its locator, a newer version's change, a TODO/deprecation, or
     the constraint you just wrote in ④) AND you mark it explicitly as inference. No anchor → omit
     the beat; four beats is a complete section. Never invent a roadmap.
- "## Related / Cross-references" — link the sibling pages (names below) + adjacent topics.

【Figures】(only if this batch uses the house SVG→PNG workflow)
Author your flowchart(s) as one HTML file at <figs dir>/<page-base>.html, `<link>`-ing the shared
stylesheet and copying the DOM/CSS patterns from the calibration page's figure file. Each figure is a
`<div class="diagram" data-name="<page-base>_figN">…</div>`; keep flowcharts process-oriented (boxes +
arrows that show a flow), not decoration. Embed each in the page as
`![图 N：caption](assets/<page-base>_figN.png)`. Do NOT run any renderer, and do NOT put `[[wiki
links]]` inside figure text — the coordinator renders, and link syntax leaks into the PNG as literal
brackets.

【Hard rules】
1. Every non-trivial claim carries a real, verified locator (<file:line | §/Table/Fig./Eq. | clause |
   column·row | timestamp>). You MUST open the source to that spot and read it BEFORE citing — catch
   the conditions ("only with X", "at 128K", "deprecated since v2", "on the private set"). Never
   fabricate, guess, or lift a locator from the abstract/a blog/memory. Locate → read the targeted
   spot → cite. If it isn't in the source, don't claim it.
2. Capture the essence, not the surface: lead with the design thesis, and write every unit in the five
   beats above — 背景 and 为什么 come BEFORE the mechanism, always. A section that opens with "this
   class does X", a signature list, or an abstract-paraphrase needs a rewrite, not a tweak.
3. Reproduce load-bearing evidence WITH its baseline (a table's baseline column, a state machine, a
   call chain). Separate "the source states X" from "this implies Y" (your inference).
4. Language: <match the audience, e.g. Chinese>. Length: ~<300–500> lines; the per-unit sections are
   the bulk. Math notation (LaTeX) for formulas. Diagrams: if the house renders SVG→PNG see 【Figures】;
   otherwise mermaid/ASCII.
5. When the source contradicts common belief / the marketing, the source wins — flag it explicitly
   (e.g. "continued-pretrained from dense, NOT trained sparse from scratch"; "the V0 path was removed").

【Sibling page names】(link these even if not yet created): <the predetermined doc-name list>
【Cross-reference targets】(adjacent pages that already exist, link where relevant): <list>

【Return — your final message is a structured summary for the coordinator, not prose for a human】
Return: ① the page title; ② 6–10 key verified locators you actually opened; ③ a 3–5 sentence thesis
(for the overview/index) that names the problem (背景) and the alternative this design beat; ④ suggested cross-links; ⑤ anything you found but did NOT cover that another
page / a gap should handle; ⑥ your figure data-names + one-line captions. Finish writing the page file
first, then return the summary.
```

---

## Coordinator checklist around the fan-out

- **Fix the doc-name list before dispatching** so every agent can link siblings that don't exist yet.
  Consistent names = a clean cross-link web with zero rework. Forward-links to planned pages are fine.
- **Wave 1, then calibrate.** Launch a first batch (or write the exemplar page yourself), read one
  finished page end to end for style, and **spot-check 2–3 of its cited locators against the actual
  source** (does Table 4 exist? does `foo.py:88` say what the page claims?). Also check the **beat
  order actually held**: does each section open with 背景, name the rejected alternative, and carry a
  约束 section — and is any 发展趋势 anchored and marked as inference? Agents drift back to
  mechanism-first when the unit is technical. If discipline slipped, fix the contract before
  launching the rest. Fabricated citations are the #1 failure mode; mechanism-first is the #2.
- **Calibration page + figure toolchain first.** For a diagram-heavy series, write one full page
  yourself and build/render its figures before fanning out — diagrams are expensive to redo N times.
- **Inherit the strong model.** Verifying locators and explaining the *why* rewards capability; don't
  down-tier writer agents unless scope is trivial.
- **You write the overview/index from the returned summaries** (Phase 4), not by re-reading everything.
- **After all pages land, verify mechanically:** zero dangling cross-links (extract every reference,
  confirm each target exists), and spot-check that cited locators are real.
- **Render and eyeball the figures (if SVG→PNG):** run the house renderer, then actually *look* at
  each PNG for overflow/clipping and stray `[[…]]` that leaked into a caption; fix and re-render.
  Keep reader-facing figure numbers sequential across the page.
