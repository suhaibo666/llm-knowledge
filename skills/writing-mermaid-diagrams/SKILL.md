---
name: writing-mermaid-diagrams
description: Use when adding or editing a Mermaid flowchart, sequence diagram, or state diagram in this wiki.
---

# Writing Mermaid Diagrams

This skill covers **Mermaid parser traps**. For whether to draw a figure and which medium to
use, read [`drawing-wiki-figures`](../drawing-wiki-figures/SKILL.md). Use generated SVG for
precise two-dimensional grids and proportional Gantt timelines.

These instructions are in English; diagram labels follow the page's language and the user's request.
Non-English labels are allowed. Use simple ASCII letters/digits for node IDs.

Mermaid uses `[]`, `()`, and `{}` to delimit node shapes, and `|` to delimit edge labels.
Embedding those delimiters or literal newlines in labels can make their boundaries ambiguous.
The repository's observed failures fall into two severity tiers.

## 1. Parser failures: zero tolerance

**Nested delimiters in special shapes** are unsafe: `X[(No [N,V] activations)]` nests brackets
inside a cylinder, and `{Check[i]}` nests brackets inside a diamond. Keep labels in special
shapes—cylinders `[(...)]`, subroutines `[[...]]`, diamonds `{...}`—to simple text without
nested `[]` or `()`. A plain quoted rectangle is a safe alternative: `X["No N×V activations"]`.

## 2. Renderer-dependent syntax: avoid in new diagrams

Quoted rectangular labels such as `A["logits[N,V]"]`, and inline quoted dotted-edge labels
such as `A -. "label" .-> B`, work in many Mermaid versions but fail in some renderers.
For portability:

- Write tensor shapes as `N×d` or `B·S·V`, not `[N,V]`.
- Use pipe labels for all labeled edges: `A -->|label| B` or `A -.->|label| B`.
- Keep quotes, parentheses, and `|` out of the edge-label text.
- Existing diagrams known to render need not be rewritten solely for these preferences;
  new diagrams must use the portable forms.

## Other common traps

For pipe labels, replace `A -->|"label(x)"| B` with `A -->|label x| B`.
For subgraphs, `subgraph id["Title"]` may contain `( )`, `:`, or `/`, but not `[ ]` or `|`.

| Trap | Correction |
|---|---|
| Literal newline within a label | Use `<br/>` |
| Non-English text or symbols used as node IDs | Use IDs such as `H` or `NA`; put display text in `["..."]` |
| Unclosed subgraph | Close each subgraph with `end` on its own line |
| Missing diagram declaration | The first line of the code block must declare the type, such as `flowchart TB`, `flowchart LR`, `sequenceDiagram`, or `graph TD` |

## Required review after generation

1. Immediately reread each Mermaid block: check bare `[] ()` in labels, nested shape delimiters,
   quotes/parentheses/`|` in edge text, and newlines other than `<br/>`.
2. Before committing, locate every diagram in each changed file with `rg -n mermaid <file>`
   and review each block against this checklist.
3. Render when a renderer is available (`mmdc` or a Mermaid live editor). If rendering is
   unavailable, perform the strict manual checklist and report that limitation accurately.
4. Fix failures in place. Never leave a block whose syntax is merely hoped to work for a commit.
