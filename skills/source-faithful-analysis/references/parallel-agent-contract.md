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
  code → name the entry file, key classes, the call chain to trace;
  paper → the contributions, each with the four beats (motivation, mechanism, evidence, why-not-alt);
  general → the aspects, each: claim → grounding locator → why → tradeoff.>

【Page structure — exactly these sections】
- Header block: title (+ a thesis-style subtitle) + a blockquote with the source baseline (verbatim
  above) + 2–3 sentences on what this page answers and how it relates to the sibling pages <list>.
- "## 1. Overview" — lead with the thesis + a key-concepts/contribution/results table + the key
  diagram. {code: + a Quick Start: minimal entry point + where to start reading, with file:line.}
- "## 2..N" — the bulk, per your unit. {code: mechanism + data structures/state machines + the real
  call chain, dense file:line. paper: one section per contribution, the FOUR beats, tables WITH
  baseline. general: one section per aspect — claim → locator → why → tradeoff.}
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
2. Capture the essence, not the surface: lead with the design thesis; give mechanism + the *why* + the
   evidence + the rejected alternative — not signatures or an abstract-paraphrase.
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
(for the overview/index); ④ suggested cross-links; ⑤ anything you found but did NOT cover that another
page / a gap should handle; ⑥ your figure data-names + one-line captions. Finish writing the page file
first, then return the summary.
```

---

## Coordinator checklist around the fan-out

- **Fix the doc-name list before dispatching** so every agent can link siblings that don't exist yet.
  Consistent names = a clean cross-link web with zero rework. Forward-links to planned pages are fine.
- **Wave 1, then calibrate.** Launch a first batch (or write the exemplar page yourself), read one
  finished page end to end for style, and **spot-check 2–3 of its cited locators against the actual
  source** (does Table 4 exist? does `foo.py:88` say what the page claims?). If discipline slipped,
  fix the contract before launching the rest. Fabricated citations are the #1 failure mode.
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
