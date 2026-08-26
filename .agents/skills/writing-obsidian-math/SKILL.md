---
name: writing-obsidian-math
description: Use when writing, editing, reviewing, or generating Markdown notes that contain LaTeX, MathJax, inline math, display equations, probability notation, matrices, or formulas in the llm-knowledge Obsidian vault.
---

# Writing Obsidian Math

Keep mathematical meaning, Obsidian rendering, and source readability correct at the same time. A formula is not finished until the repository checker and the manual semantic review both pass.

## Canonical delimiters

Use dollar delimiters everywhere in knowledge-base Markdown.

Inline math:

```markdown
策略比率为 $r_t=\exp(\log p_t-\log p_{\mathrm{old},t})$。
```

Display math, with each delimiter on its own line:

```markdown
$$
L(\theta)=\mathbb{E}_{x\sim\mathcal D}[\ell_\theta(x)].
$$
```

Never introduce `\(...\)` or `\[...\]`. Do not put content on the same line as `$$`. These alternatives may work in some MathJax environments, but this vault standardizes on Obsidian's dollar syntax for stable Live Preview and source portability.

Code fences and inline code may show literal delimiters as examples; the checker intentionally ignores those regions.

## Give notation its intended semantics

### Semantic subscripts are upright

Ordinary letters are italic mathematical variables. Put words, roles, phases, and configuration labels in `\mathrm{...}` or `\text{...}`.

- Bad: `$C_{low}$`, `$p_{train-old}$`, `$T_{reward/verifier}$`
- Good: `$C_{\mathrm{low}}$`, `$p_{\mathrm{train,old}}$`, `$T_{\mathrm{reward/verifier}}$`
- Use `\text{...}` when the label contains natural-language spaces: `$C_{\text{single module}}$`

Two-letter mathematical index combinations such as `$m_{it}$` may remain italic when they really mean indices $i$ and $t$, not the word “it”.

### Separate mathematical symbols from API identifiers

Do not invent a mathematical variable by escaping underscores in an identifier.

- Bad: `$accept\_rate=accepted/drafted$`
- Preferred: define `$r_{\mathrm{accept}}$` in prose and use it in the equation.
- If the literal program name is essential, render it as `$\texttt{accept\_rate}$` and keep the mathematical definition separate.

Use `\operatorname{...}` for named operators such as `$\operatorname{clip}(x,a,b)$`. Do not use it merely to hide a long prose sentence inside math.

### Use semantic relation symbols

- Conditioning: `$p(y\mid x)$`, not `$p(y|x)$`.
- Absolute value: `$\lvert w-1\rvert$`, not `$|w-1|$`.
- Norm: `$\lVert x\rVert_2$`.
- Sets whose vertical bar means “such that” also use `\mid` unless set-builder spacing requires a carefully chosen `\middle|`.

This is especially important in Markdown tables, where a raw `|` can split a cell before MathJax sees the formula.

## Structure multiline equations explicitly

Use `aligned` when an equation has multiple logical rows or repeated relation points:

```markdown
$$
\begin{aligned}
\hat r_i &= r_i-\bar r_g, \\
\hat r_i^{\mathrm{std}}
&=\frac{r_i-\bar r_g}{s_g+10^{-6}}.
\end{aligned}
$$
```

Place `&` immediately before the relation to align, and end every non-final row with `\\`. Source-code line wrapping without `\\` does not create a rendered line break.

Use `\left` and `\right` in matched pairs. For very tall expressions, prefer readable nested fractions and `aligned` rows over one long line.

## Tables and callouts

- Keep table-cell math short and inline.
- Move display equations outside Markdown tables.
- Never use a raw conditional `|` in table math; use `\mid`.
- In an Obsidian callout, keep the entire display block consistently quoted with `>` or place the formula immediately after the callout. Do not quote only one delimiter.

## Required workflow

1. Preserve the original mathematical claim. Before changing notation, identify what every symbol, index, operator, and denominator means.
2. Write with the canonical delimiters and semantic notation above.
3. Reread every changed formula as rendered mathematics, not just as LaTeX source:
   - Are delimiters paired?
   - Are braces and `\left`/`\right` paired?
   - Are semantic words upright?
   - Does every `|` mean conditioning, absolute value, a norm, or a Markdown column?
   - Do multiline rows have explicit `\\` and valid `&` alignment points?
   - Did a notation cleanup accidentally change the formula's meaning?
4. Run strict checking on the files or directory you edited:

```powershell
python tools/check_math.py --strict wiki/path/to/page.md
```

5. Before commit, check all Git-modified Markdown files:

```powershell
python tools/check_math.py --changed --strict
```

6. Resolve every error. Under strict mode, resolve every warning by correcting the notation or manually proving it is a deliberate mathematical exception. Do not weaken, escape, or hide valid source text merely to silence a heuristic.

The checker validates high-confidence Markdown/MathJax structure and selected notation smells. It cannot prove algebraic correctness, equivalence to a paper, or whether a symbol definition matches the surrounding prose; the manual semantic pass remains mandatory.

## Quick diagnosis

- Formula disappears or shows raw source: first check delimiters and unmatched braces.
- Formula renders but identifiers look like multiplied italic letters: use semantic subscripts or a named operator.
- A table gains extra columns: find raw `|` inside math and replace it with `\mid`, or move the formula outside the table.
- A long derivation appears on one line: add `aligned`, `&`, and explicit `\\` row endings.
- A checker warning appears inside a literal code/API name: put the literal name in `\texttt{...}` and define a separate mathematical symbol if the equation uses it repeatedly.
