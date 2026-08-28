import assert from "node:assert/strict"
import test from "node:test"

import { markdownMathInputs } from "./mathjax-corpus.mjs"

test("source math discovery includes blockquotes without code or currency false positives", () => {
  const markdown = [
    "---",
    'title: "$frontmatter is not math$"',
    "---",
    "",
    "Inline $x + y$ is math, while a $5.5M budget is not.",
    "",
    "> [!note]",
    "> $$",
    "> \\operatorname{mean\\text{-}pool}(x)",
    "> $$",
    "",
    "Inline code `$ignored$` is not math.",
    "",
    "```markdown",
    "> $$",
    "> ignored",
    "> $$",
    "```",
  ].join("\n")

  assert.deepEqual(
    markdownMathInputs(markdown, "wiki/quoted.md").map(({ kind, line }) => ({
      kind,
      line,
    })),
    [
      { kind: "inline", line: 5 },
      { kind: "display", line: 8 },
    ],
  )
})
