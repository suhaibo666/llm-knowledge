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

## Checker rules, and what each one actually wants

`tools/check_math.py` emits stable codes. Errors block; warnings block under `--strict`.
The corpus was taken to **0 errors and 0 warnings** on 2026-08-26, so any new finding is
something you just introduced.

| Code | Severity | Means | Fix |
|---|---|---|---|
| `MATH001` | error | legacy `\(` `\)` `\[` `\]` delimiter | use `$...$` / `$$` on their own lines |
| `MATH002` | error | display `$$` opened but never closed | close the block |
| `MATH003` | error | inline `$` not paired on the line | pair it, or escape a literal `\$` |
| `MATH004` | error | unbalanced braces inside math | balance `{}` |
| `MATH005` | error | raw `\|` inside table math, or display math in a table | `\mid`, or move the block out of the table |
| `MATH101` | warning | a `$$` shares its line with content | each `$$` gets its own line |
| `MATH102` | warning | `\_` outside a text-like command | `\texttt{...}` / `\mathrm{...}`, or define a symbol |
| `MATH103` | warning | word-like subscript left italic | `\mathrm{...}`, or `\text{...}` when it contains spaces |
| `MATH104` | warning | raw `\|` inside math | see the bar table below |
| `MATH105` | warning | display line >120 chars without `aligned` | restructure into `aligned` rows |

### Choosing the right vertical bar

The checker accepts any `|` preceded by a backslash, so all four of these pass — pick by meaning,
not by what silences the warning.

| Meaning | Write | Not |
|---|---|---|
| conditioning | `p(y \mid x)` | `p(y|x)` |
| absolute value / cardinality | `\lvert y_i \rvert` | `|y_i|` |
| norm | `\lVert x \rVert_2` | `||x||` |
| KL divergence | `\mathrm{KL}(p \,\|\, q)` | `\mathrm{KL}(p || q)` |
| sized delimiter, set-builder | `\big\vert`, `\Big\vert` | `\big|`, `\Big|` |

## Things that look like math problems but are not

Four traps cost real time during the 2026-08-26 cleanup. Recognise them before "fixing" anything.

1. **Escaped square brackets in prose are not display math.** `List\[str\]`, `A\[t\]`,
   `\[B,H,N,N\]`, and step labels like `\[B\]` are Markdown escapes for a literal `[`.
   Converting them to `$$` corrupts the page. Put them in inline code instead —
   `` `List[str]` `` — which is also what they mean. Do **not** simply unescape them:
   `[C][D]` is Markdown reference-link syntax.
2. **`\\[2pt]` is LaTeX row spacing, not a `\[` delimiter.** It appears inside `aligned`
   and `cases`. Leave it alone; the checker skips escaped backslashes.
3. **Inline math may legitimately start with a digit.** `$4.2 \times 10^{-4}$` is a formula,
   while `$5.576M` in a cost table is money. The checker distinguishes them by looking for a
   closing `$` and for mathematical content in between.
4. **`h_{t-1}` is an index, not a label.** Subscripts made of indices and operators stay
   italic. `\mathrm{}` is for words, roles, phases and configuration labels
   (`old`, `low`, `teacher`, `GRPO`) — three or more consecutive letters.

## Restructuring a long equation

`MATH105` wants real structure, not a token `aligned` wrapper. Break at a **top-level**
relation — one that is not nested inside `{}`, `\left...\right`, or another
`\begin{...}...\end{...}`:

```markdown
$$
\begin{aligned}
V_{\text{total}}
&\approx \underbrace{c \cdot n_{kv} \cdot d_h \cdot B}_{\text{Stage 1}} \\
&\quad + \underbrace{\frac{(P-1) \cdot S \cdot n_{kv} \cdot d_h \cdot B}{P^2 \cdot c}}_{\text{Stage 2}}
\end{aligned}
$$
```

Three constraints that are easy to get wrong:

- **Every row but the last needs `\\`.** Two `&` without a `\\` between them put two alignment
  points on one row, which is a LaTeX error.
- **`\left` and `\right` cannot cross a row break.** If a bracket must span rows, switch to a
  fixed size: `\Big( ... \Big)`.
- **A source line break is not a row break.** `LHS` on one line and `&= RHS` on the next is a
  single aligned row — which is exactly what you want for a two-part equation.

For an expression with no relation to align on (a bare objective such as
`\min_\theta \mathbb{E}[...]`), keep it as one row and soft-wrap it at top-level spaces; no `&`
or `\\` is needed. If the whole equation lives inside `\boxed{}`, move the `aligned` *inside*
the box.

## Callout blocks

Quote the whole display block consistently — never just one delimiter:

```markdown
> $$
> M_{\text{HBM}}^{\text{target}} = \text{device\_max} - \text{reduction\_memory}
> $$
```
