import assert from "node:assert/strict"
import { after, before, test } from "node:test"
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises"
import { createRequire } from "node:module"
import os from "node:os"
import path from "node:path"
import { fileURLToPath } from "node:url"

import { findBrowserExecutable } from "../docs-site/listeners.mjs"
import { corpusPages, htmlMathOrigins } from "./mathjax-corpus.mjs"

const toolDir = path.dirname(fileURLToPath(import.meta.url))
const require = createRequire(import.meta.url)
const puppeteer = require(path.join(toolDir, "node_modules", "puppeteer-core"))

let browser
let analyzer

before(async () => {
  browser = await puppeteer.launch({
    executablePath: await findBrowserExecutable(),
    headless: true,
    args: ["--disable-background-networking", "--no-sandbox"],
  })
  analyzer = await browser.newPage()
})

after(async () => {
  await analyzer?.close()
  await browser?.close()
})

test("final HTML DOM excludes comments and rendered code from math origins", async () => {
  const html = String.raw`<!doctype html><html><body>
    <!-- $$ commented $$ -->
    <p>Multiline <code>$$
      code span
    $$</code> remains literal.</p>
    <pre><code>    $indented$
    $$fenced$$</code></pre>
    <script>const sample = "$script$"</script>
    <style>.sample { content: "$style$"; }</style>
  </body></html>`

  assert.deepEqual(await htmlMathOrigins(analyzer, html, "code.html"), {
    markerUnits: 0,
    rawDollarDelimiters: 0,
    rawDollarUnits: 0,
    rawTexDelimiters: 0,
    rawTexUnits: 0,
    units: 0,
  })
})

test("final HTML DOM counts list math but not adjacent currency", async () => {
  const html = String.raw`<!doctype html><html><body>
    <p>Budget $5.5M</p>
    <p>Raw $x + y$.</p>
    <ol><li>
      <span class="arithmatex">\(x + y\)</span>
    </li></ol>
  </body></html>`

  assert.deepEqual(await htmlMathOrigins(analyzer, html, "list.html"), {
    markerUnits: 1,
    rawDollarDelimiters: 2,
    rawDollarUnits: 1,
    rawTexDelimiters: 0,
    rawTexUnits: 0,
    units: 2,
  })
})

test("final HTML DOM includes blockquoted raw dollar display math", async () => {
  const html = String.raw`<!doctype html><html><body>
    <blockquote><p>$$
      \operatorname{mean\text{-}pool}(x)
    $$</p></blockquote>
  </body></html>`

  assert.deepEqual(await htmlMathOrigins(analyzer, html, "quoted.html"), {
    markerUnits: 0,
    rawDollarDelimiters: 2,
    rawDollarUnits: 1,
    rawTexDelimiters: 0,
    rawTexUnits: 0,
    units: 1,
  })
})

test("final HTML DOM correlates mixed generated and raw math units", async () => {
  const html = String.raw`<!doctype html><html><body>
    <p><span class="arithmatex">\(a\)</span> and raw $b$.</p>
    <div class="arithmatex">\[c\]</div>
    <span class="md-ellipsis">Navigation copy \(d\)</span>
    <div class="no-mathjax">$ignored$</div>
  </body></html>`

  assert.deepEqual(await htmlMathOrigins(analyzer, html, "mixed.html"), {
    markerUnits: 2,
    rawDollarDelimiters: 2,
    rawDollarUnits: 1,
    rawTexDelimiters: 2,
    rawTexUnits: 1,
    units: 4,
  })
})

test("final HTML DOM never pairs dollar delimiters across block nodes", async () => {
  const html = String.raw`<!doctype html><html><body>
    <p>$$ split opener</p>
    <p>split closer $$</p>
  </body></html>`

  assert.deepEqual(await htmlMathOrigins(analyzer, html, "split.html"), {
    markerUnits: 0,
    rawDollarDelimiters: 2,
    rawDollarUnits: 0,
    rawTexDelimiters: 0,
    rawTexUnits: 0,
    units: 0,
  })
})

test("paired numeric dollar delimiters take precedence over currency", async () => {
  const html = "<!doctype html><html><body><p>$5$</p></body></html>"

  assert.deepEqual(await htmlMathOrigins(analyzer, html, "numeric.html"), {
    markerUnits: 0,
    rawDollarDelimiters: 2,
    rawDollarUnits: 1,
    rawTexDelimiters: 0,
    rawTexUnits: 0,
    units: 1,
  })
})

test("unmatched currency is ignored before a generated math marker", async () => {
  const html = String.raw`<!doctype html><html><body>
    <p>Budget $5.5M</p>
    <p><span class="arithmatex">\(x\)</span></p>
  </body></html>`

  assert.deepEqual(await htmlMathOrigins(analyzer, html, "currency.html"), {
    markerUnits: 1,
    rawDollarDelimiters: 0,
    rawDollarUnits: 0,
    rawTexDelimiters: 0,
    rawTexUnits: 0,
    units: 1,
  })
})

test("dangling dollar and TeX openers remain visible as residuals", async () => {
  const html = String.raw`<!doctype html><html><body>
    <p>dangling $x</p>
    <p>dangling \(y</p>
  </body></html>`

  assert.deepEqual(await htmlMathOrigins(analyzer, html, "dangling.html"), {
    markerUnits: 0,
    rawDollarDelimiters: 1,
    rawDollarUnits: 0,
    rawTexDelimiters: 1,
    rawTexUnits: 0,
    units: 0,
  })
})

test("TeX delimiters never pair across separate DOM text nodes", async () => {
  const html = String.raw`<!doctype html><html><body>
    <p>\( split opener</p>
    <p>split closer \)</p>
  </body></html>`

  assert.deepEqual(await htmlMathOrigins(analyzer, html, "split-tex.html"), {
    markerUnits: 0,
    rawDollarDelimiters: 0,
    rawDollarUnits: 0,
    rawTexDelimiters: 2,
    rawTexUnits: 0,
    units: 0,
  })
})

test("corpus includes a residual-only generated route for failure reporting", async () => {
  const repoRoot = await mkdtemp(path.join(os.tmpdir(), "math-corpus-residual-"))
  try {
    const siteRoot = path.join(repoRoot, "site")
    const cacheRoot = path.join(repoRoot, ".mkdocs-cache")
    await mkdir(path.join(siteRoot, "domain"), { recursive: true })
    await mkdir(cacheRoot, { recursive: true })
    await writeFile(
      path.join(siteRoot, "domain", "dangling.html"),
      "<!doctype html><html><body><p>dangling $x</p></body></html>",
    )
    await writeFile(
      path.join(cacheRoot, "routes.json"),
      JSON.stringify([{ source: "domain/dangling.md", output: "domain/dangling.html" }]),
    )

    const result = await corpusPages({ analyzer, repoRoot })

    assert.equal(result.inspectedPages, 1)
    assert.deepEqual(result.pages.map((item) => ({
      sourcePath: item.sourcePath,
      output: item.output,
      rawDollarDelimiters: item.rawDollarDelimiters,
      units: item.units,
    })), [{
      sourcePath: "wiki/domain/dangling.md",
      output: "domain/dangling.html",
      rawDollarDelimiters: 1,
      units: 0,
    }])
  } finally {
    await rm(repoRoot, { recursive: true, force: true })
  }
})
