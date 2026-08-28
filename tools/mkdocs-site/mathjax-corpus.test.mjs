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

test("a fenced example inside an HTML comment cannot hide later math", () => {
  const markdown = [
    "<!--",
    "```markdown",
    "$$",
    "commented example",
    "$$",
    "-->",
    "",
    "> $$",
    "> x + y",
    "> $$",
  ].join("\n")

  assert.deepEqual(markdownMathInputs(markdown, "wiki/comment.md"), [{
    sourcePath: "wiki/comment.md",
    kind: "display",
    line: 8,
    latex: "\nx + y\n",
  }])
})

test("an HTML comment token inside inline code cannot hide later math", () => {
  const markdown = [
    "Literal `<!--` remains code.",
    "Real math $x + y$ remains visible.",
  ].join("\n")

  assert.deepEqual(markdownMathInputs(markdown, "wiki/inline-code.md"), [{
    sourcePath: "wiki/inline-code.md",
    kind: "inline",
    line: 2,
    latex: "x + y",
  }])
})

test("four-space-indented code is not source math", () => {
  const markdown = [
    "    $ignored$",
    "",
    "Visible $x + y$ remains math.",
  ].join("\n")

  assert.deepEqual(markdownMathInputs(markdown, "wiki/indented.md"), [{
    sourcePath: "wiki/indented.md",
    kind: "inline",
    line: 3,
    latex: "x + y",
  }])
})

test("multiline comments and blockquoted fences keep independent parser state", () => {
  const markdown = [
    "> <!--",
    "> ```markdown",
    "> $$",
    "> commented example",
    "> $$",
    "> -->",
    "> ```markdown",
    "> <!-- $fenced$ -->",
    "> $$",
    "> fenced example",
    "> $$",
    "> ```",
    "> $$",
    "> x + y",
    "> $$",
  ].join("\n")

  assert.deepEqual(markdownMathInputs(markdown, "wiki/combined.md"), [{
    sourcePath: "wiki/combined.md",
    kind: "display",
    line: 13,
    latex: "\nx + y\n",
  }])
})
