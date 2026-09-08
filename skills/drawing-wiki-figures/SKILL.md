---
name: drawing-wiki-figures
description: Use when adding or changing a figure on a wiki page, including mechanism diagrams, layouts, timelines, and quantitative comparisons. For Mermaid parser constraints, also load writing-mermaid-diagrams.
---

# Drawing Wiki Figures

This skill governs **whether a figure helps, which medium to use, and how to make it readable and verifiable**.
The instructions are in English; figure labels and wiki prose follow the page's language and the user's request.

## 1. Decide what the figure must explain

Draw figures for three purposes:

1. **A mechanism's flow**: how inputs pass through the important transformations to outputs.
2. **The internal structure of the core mechanism**: the part that most needs to be unpacked.
3. **The most informative result or cost**: the evidence that supports the page's explanation.

Skip decorative background, related-work or terminology diagrams, abstract design-philosophy sketches,
anything a single sentence explains, and repetitive charts for every ablation.

Choose the explanatory question before the shape. An algorithm figure should expose a concrete
input, intermediate transformations, and output; a layout figure should expose placement and movement;
a stateful mechanism may need state transitions and completion boundaries. **A call tree or module
graph supports source navigation; it does not by itself explain an algorithm's principle.**
For a local algorithm or numerical transform without shared state or ownership movement, explain
the computation without an ownership inventory. When a represented path does move ownership,
including a distributed numerical algorithm, show that movement and its relevant guarantees.

### Two acceptance criteria

> **A. New-reader criterion:** Can someone unfamiliar with the source code or paper understand
> the intended mechanism from the figure and its annotations? If not, split it or add the missing labels.
>
> **B. Reorganization criterion:** Does the figure merely reproduce an upstream figure or official
> architecture diagram? A new figure must reorganize information around this page's explanation,
> making relevant dimensions, costs, or rejected paths visible. It must serve the page's thesis.

## 2. Choose the medium

| Content | Medium | Reason |
|---|---|---|
| Call chains, pass pipelines, state machines, module dependencies, decision branches | **Mermaid** `flowchart` / `stateDiagram` | Topological relationships are native to Mermaid and easy to maintain |
| Multi-party interactions, communication handshakes, request lifecycles | **Mermaid** `sequenceDiagram` | Expresses ordering and participants directly |
| **Two-dimensional grids**: pipeline schedules (time × stage), tensor/expert sharding, memory layouts, attention masks, geometric communication topology | **Generator → external `.svg`** (§3.5) | Requires precise two-dimensional placement |
| Gantt-like computation/communication overlap, hiding, dual streams | **Generator → external `.svg`** (§3.5) | Requires proportional placement along a time axis |
| Directory trees and simple two- or three-level hierarchies | **ASCII** | Appropriate for compact tree structure |
| Quantitative comparisons, hyperparameters, ablations | **Table** | Supports direct comparison without decorative charting |

Reserve ASCII for tree structure, not grids or schedules. Do not use bitmap image generation for
technical figures: labels, dimensions, formulas, and numbers must be editable and traceable.
Rendered figures use generated `.svg` with real text (§3.5). Mermaid remains a valid medium when
it expresses the required information; the absence of an SVG is not a quality failure by itself.

## 3. Visual specification for rendered figures

Design tokens live in `tools/figs/figstyle.css`: the original flat palette plus a **paper-figure layer**
(informed by zsyggg/paper-craft-skills `styles/paper-figure.md`). HTML figures use these classes
directly; SVG generators reuse the same class names and color values in their own `<style>`
(see `tools/figs/svg/pp_schedule.mjs`). Do not create a separate vocabulary for each medium.

### Neutral by default; accents identify the important parts

- Use `.neutral` (white fill, gray border) for most nodes and `.ghost` for background, controls, or rejected paths.
- Use **at most two accent colors** per figure: `.acc1` (blue) for the key mechanism/path and
  `.acc2` (orange) for costs, bottlenecks, or conflicts.
- Express categories with grouping boxes and captions. Equal use of all legacy
  `.blue/.purple/.green/.amber/.slate/.rose` colors makes everything compete for attention.

Distinguish the main path, `.arrow.main` (thicker and colored), from auxiliary paths,
`.arrow.aux` (thinner and lighter), so readers can see the reading order.

### Reusable elements

| Element | Class | Uses |
|---|---|---|
| Matrix cells, including three intensity levels `.h1/.h2/.h3` and unused `.x` | `.cells` / `.cell` | Attention masks, tensor shards, KV occupancy, expert routing |
| Token sequence, including selected `.on` and rejected `.rej` | `.seq` / `.tok` | Sequence parallelism, speculative decoding, chunked prefill |
| Local enlargement | `.zoom` | Useful detail when a figure has excessive empty space |
| Dimension/dtype tag | `.dim` | Adjacent labels such as `[B,S,H] bf16` |
| Caption label | `.cap` | Short labels; put full explanatory sentences in prose |

### Density and typography

- Aim for the main explanation to occupy **75%–85%** of the canvas. Use useful enlarged details,
  dimensions, or comparison paths to fill excessive gaps rather than enlarging the font.
- Keep annotations short: module names, variables, key operations, at most one line of explanation.
- Split a figure when its content exceeds a readable page; do not squeeze it into one image.
- Avoid blueprint styling, 3D renders, text walls, copied upstream layouts, slide covers, and marketing posters.

## 3.5 Output pipeline: generator → external `.svg` → standard image syntax

Use the verified pipeline:

```text
Data/simulation → Node generator → wiki/<domain>/assets/<name>.svg → ![description](assets/<name>.svg)
```

Reference implementation: [`tools/figs/svg/pp_schedule.mjs`](../../tools/figs/svg/pp_schedule.mjs).

| Property | Inline `<svg>` in Markdown | **External `.svg` + `![]()`** | HTML → PNG (legacy) |
|---|---|---|---|
| Obsidian editing view | A large text block | Normal image | Normal image |
| docs-site | Broken by the Quartz pipeline below | Rendered as `<img>`; asset MD5 preserved | Supported |
| Searchable/diffable text | Yes | Yes | No; binary |
| Scaling | Vector | Vector | Fixed 2×; blurs when enlarged |
| Dependencies | None | None | Edge + puppeteer |

A browser parses an external SVG as an **independent XML document**: `<defs>`, `<pattern>`,
camelCase attributes, and `<style>` work, with styles isolated from the site's CSS.

The verified Quartz v5 `remark-rehype → hast-util-to-jsx-runtime` path breaks inline SVG:

1. It self-closes `<svg …>` immediately and places the remaining elements outside it.
2. It lowercases case-sensitive attributes (`patternTransform` → `patterntransform`).
3. It treats self-closing elements as nested (`<rect/><rect/>` → `<rect><rect></rect></rect>`).

Raw HTML such as `<div>` passing through is not evidence that inline SVG works.

### Computed figures must come from data

For a figure derived from a solver or simulation, **calculate every result from that source**;
do not hand-enter computed quantities into SVG. For example, schedule cells, `makespan=22`,
and peak live-activation counts must come from the same simulation. Changing the generator's
parameters must recompute the result. Explicit example inputs and source-derived labels still
need provenance; they are not themselves solver outputs.

**Solver-derived figures require regression tests.** Lock the solver output to the numerical
claims cited by the page and include the tests in the applicable `pytest` / `node --test` gate.
Reference: `tools/figs/svg/lib/megatron_pp_sim.test.mjs`, which compares simulation output with
page schedule cells. When the source baseline changes behavior, a failing test should require
updating the page, figure, and baseline declaration together.

**The test must actually read the `.md` file.** Hardcoding a number in the test and looking for
it in SVG checks only the image: the prose can drift while the test stays green. Read the page itself (for example, with `readFile`)
and assert that the computed quantities agree with its claims. See `megatron_cp_figures.test.mjs`
for the page/figure numerical-consistency test. Ask: **If only the prose changed, would this test fail?**

Conceptual diagrams and source-navigation graphs do not require a numerical solver or solver
regression tests. Their labels and relationships still must match the source at the declared baseline.

**Canvas bounds do not prove non-overlap.** All elements may lie inside the viewBox while a cell
covers adjacent text. Check element-to-element bounding-box overlaps or actually render and inspect
the figure (§5; `chrome-headless-shell` + `puppeteer-core` is available). Doing neither leaves
layout unchecked.

## 4. Write the specification before drawing

Before implementation, describe the figure in natural language down to arrow endpoints and box
labels. This also provides the contract if figure work is delegated.

```text
Figure 2: Why 1F1B steady-state peak activation memory scales with pp rather than m

Explain: At the same bubble rate in this example, GPipe retains m activations,
while 1F1B has approximately pp activations in flight.

Layout: two timelines stacked for comparison.
- Top: GPipe, four rows (stages 0–3), time on the horizontal axis; F cells followed by B cells.
  Right .zoom: "Peak: all m=8 activations retained".
- Bottom: 1F1B, the same four rows; warmup staircase → alternating F/B → cooldown.
  Right .zoom: "Peak: 4 activations in flight".
- Use equal total timeline lengths to show the equal bubble rate in this example.

Annotations:
- F uses a dark tone, B a lighter tone of the same family, bubbles neutral hatching: fill:url(#hatch).
- Label rows "stage 0" through "stage 3".
- GPipe cost box uses acc2; 1F1B benefit box uses acc1.
- Add a makespan ruler below each timeline; equal lengths convey the comparison.
- .cap: "P=4, m=8. Bubble rate: (P-1)/(m+P-1)=3/11; peak activations: GPipe ∝m, 1F1B ∝P".
```

For a set of figures, finish and inspect the first as a calibration example before generating the rest.

## 5. Pre-submission checks

- [ ] Meets the new-reader and reorganization criteria.
- [ ] Uses the medium appropriate to the information; no ASCII substitutes for grids or Gantt timelines.
- [ ] Uses at most two accents for the key mechanism and cost, with mostly `.neutral` / `.ghost` nodes.
- [ ] Distinguishes main and auxiliary paths when both are present.
- [ ] Uses the canvas well, targeting 75%–85% for the main explanation without large empty areas.
- [ ] Every number, dimension, and source symbol agrees with the prose or the source at the frozen baseline;
  figure and prose share evidence, without requiring line-number citations.
- [ ] Uses short caption labels, with full explanations in prose.
- [ ] Mermaid diagrams also pass [`writing-mermaid-diagrams`](../writing-mermaid-diagrams/SKILL.md).
- [ ] Rendered figures use **external `.svg`** + `![](assets/x.svg)`, not inline `<svg>`.
- [ ] Solver-derived figures compute their results and have tests that compare them with the actual page.
- [ ] The rendered result has been inspected for overflow, overlap, and leaked `[[…]]` markup.
