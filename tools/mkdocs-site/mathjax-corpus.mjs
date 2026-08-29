import assert from "node:assert/strict"
import { createServer } from "node:http"
import { access, readFile } from "node:fs/promises"
import { createRequire } from "node:module"
import path from "node:path"
import { fileURLToPath } from "node:url"

import { findBrowserExecutable } from "../docs-site/listeners.mjs"

const scriptFile = fileURLToPath(import.meta.url)
const toolDir = path.dirname(scriptFile)
const defaultRepoRoot = path.resolve(toolDir, "..", "..")
const nodeModules = path.join(toolDir, "node_modules")
const puppeteerPackage = path.join(nodeModules, "puppeteer-core", "package.json")
const projectBase = "/llm-knowledge/"
const searchIndexPath = `${projectBase}search/search_index.json`

const contentTypes = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".svg", "image/svg+xml"],
  [".woff2", "font/woff2"],
])

function safeSitePath(url, siteRoot, basePath) {
  const parsed = new URL(url, "http://127.0.0.1")
  if (!parsed.pathname.startsWith(basePath)) return null
  const relative = decodeURIComponent(parsed.pathname.slice(basePath.length))
  const candidate = path.resolve(siteRoot, relative || "index.html")
  const prefix = `${path.resolve(siteRoot)}${path.sep}`
  if (candidate !== path.resolve(siteRoot) && !candidate.startsWith(prefix)) return null
  return candidate
}

async function startStaticSite(siteRoot, basePath) {
  const server = createServer(async (request, response) => {
    try {
      const candidate = safeSitePath(request.url || "/", siteRoot, basePath)
      if (!candidate) {
        response.writeHead(404).end()
        return
      }
      const content = await readFile(candidate)
      response.writeHead(200, {
        "content-type": contentTypes.get(path.extname(candidate).toLowerCase())
          || "application/octet-stream",
      })
      response.end(content)
    } catch (error) {
      response.writeHead(error.code === "ENOENT" ? 404 : 500).end()
    }
  })
  await new Promise((resolve, reject) => {
    server.once("error", reject)
    server.listen(0, "127.0.0.1", resolve)
  })
  const address = server.address()
  assert.ok(address && typeof address !== "string")
  return { server, origin: `http://127.0.0.1:${address.port}` }
}

async function closeServer(server) {
  await new Promise((resolve) => {
    const timer = setTimeout(() => {
      server.closeAllConnections?.()
      resolve()
    }, 2_000)
    timer.unref?.()
    server.close(() => {
      clearTimeout(timer)
      resolve()
    })
    server.closeIdleConnections?.()
  })
}

/* This function is serialized into Chromium by Puppeteer; keep it self-contained. */
function browserMathOrigins(html = null) {
  const root = html === null
    ? document
    : new DOMParser().parseFromString(html, "text/html")
  const markerUnits = [...root.querySelectorAll(".arithmatex")]
    .filter((node) => !node.parentElement?.closest(".arithmatex")).length
  const currencyPattern = /^\$(?:\d[\d,]*(?:\.\d+)?)(?:\s?(?:USD|CNY|RMB|[KMBT]))?\b/i
  const ignoredSelector = [
    "code",
    "pre",
    "script",
    "noscript",
    "style",
    "textarea",
    "annotation",
    "annotation-xml",
    ".arithmatex",
    "mjx-container",
    ".no-mathjax",
    ".no-math",
  ].join(", ")

  function escaped(text, index) {
    let slashes = 0
    for (let cursor = index - 1; cursor >= 0 && text[cursor] === "\\"; cursor -= 1) {
      slashes += 1
    }
    return slashes % 2 === 1
  }

  let rawDollarDelimiters = 0
  let rawDollarUnits = 0
  let rawTexDelimiters = 0
  let rawTexUnits = 0

  function texDelimiters(text, open, close) {
    let units = 0
    let delimiters = 0
    let pendingOpen = false
    let cursor = 0
    while (cursor < text.length) {
      const openAt = text.indexOf(open, cursor)
      const closeAt = text.indexOf(close, cursor)
      const candidates = [openAt, closeAt].filter((index) => index >= 0)
      if (!candidates.length) break
      const start = Math.min(...candidates)
      const token = start === openAt ? open : close
      if (escaped(text, start)) {
        cursor = start + token.length
        continue
      }
      delimiters += 1
      if (token === open) {
        pendingOpen = true
      } else if (pendingOpen) {
        units += 1
        pendingOpen = false
      }
      cursor = start + token.length
    }
    return { delimiters, units }
  }

  const walker = root.createTreeWalker(root.body || root, NodeFilter.SHOW_TEXT)
  while (walker.nextNode()) {
    const node = walker.currentNode
    if (node.parentElement?.closest(ignoredSelector)) continue
    const text = node.nodeValue || ""
    const singleDollars = []
    const displayDollars = []
    for (let index = 0; index < text.length;) {
      if (text[index] !== "$" || escaped(text, index)) {
        index += 1
        continue
      }
      if (text[index + 1] === "$") {
        displayDollars.push(index)
        index += 2
        continue
      }
      singleDollars.push(index)
      index += 1
    }
    const singlePairs = Math.floor(singleDollars.length / 2)
    const displayPairs = Math.floor(displayDollars.length / 2)
    rawDollarUnits += singlePairs + displayPairs
    rawDollarDelimiters += (singlePairs + displayPairs) * 2
    if (singleDollars.length % 2 === 1) {
      const unmatched = singleDollars.at(-1)
      if (!currencyPattern.test(text.slice(unmatched))) rawDollarDelimiters += 1
    }
    if (displayDollars.length % 2 === 1) rawDollarDelimiters += 1

    for (const [open, close] of [["\\(", "\\)"], ["\\[", "\\]"]]) {
      const tex = texDelimiters(text, open, close)
      rawTexDelimiters += tex.delimiters
      rawTexUnits += tex.units
    }
  }
  return {
    markerUnits,
    rawDollarDelimiters,
    rawDollarUnits,
    rawTexDelimiters,
    rawTexUnits,
    units: markerUnits + rawDollarUnits + rawTexUnits,
  }
}

export async function htmlMathOrigins(page, html, sourcePath = "<memory>") {
  try {
    return await page.evaluate(browserMathOrigins, html)
  } catch (error) {
    throw new Error(`${sourcePath}: cannot inspect generated HTML`, { cause: error })
  }
}

export async function corpusPages({
  analyzer,
  repoRoot = defaultRepoRoot,
  siteRoot = path.join(repoRoot, "site"),
  manifest = path.join(repoRoot, ".mkdocs-cache", "routes.json"),
} = {}) {
  assert.ok(analyzer, "corpusPages requires an inert browser DOM analyzer")
  const routes = JSON.parse(await readFile(manifest, "utf8"))
  const pages = []
  for (const route of routes) {
    const output = path.join(siteRoot, ...route.output.split("/"))
    const html = await readFile(output, "utf8")
    const sourcePath = `wiki/${route.source}`
    const origins = await htmlMathOrigins(analyzer, html, sourcePath)
    if (origins.units || origins.rawDollarDelimiters || origins.rawTexDelimiters) {
      pages.push({ sourcePath, output: route.output, ...origins })
    }
  }
  return { inspectedPages: routes.length, pages }
}

async function renderPages(
  browser,
  origin,
  items,
  externalRequests,
  failedRequests,
  pageErrors,
  emptySearchIndex,
) {
  const page = await browser.newPage()
  let activeSourcePath = "<navigation>"
  await page.setRequestInterception(true)
  page.on("request", (request) => {
    const url = new URL(request.url())
    if (["http:", "https:"].includes(url.protocol) && url.origin !== origin) {
      externalRequests.push(request.url())
    }
    if (url.origin === origin && url.pathname === searchIndexPath) {
      request.respond({
        status: 200,
        contentType: "application/json",
        body: emptySearchIndex,
      }).catch(() => undefined)
      return
    }
    request.continue().catch(() => undefined)
  })
  page.on("response", (response) => {
    if (response.status() >= 400) {
      failedRequests.push(`${response.status()} ${response.url()}`)
    }
  })
  page.on("requestfailed", (request) => {
    failedRequests.push(`${request.url()}: ${request.failure()?.errorText || "failed"}`)
  })
  page.on("pageerror", (error) => {
    // Malformed Mermaid fixtures are exercised by the diagram contract. Keep
    // this gate focused on MathJax and unrelated page/worker failures.
    if (String(error.stack || error.message).includes("/assets/vendor/mermaid/")) return
    pageErrors.push(`${activeSourcePath}: ${error.message}`)
  })
  try {
    const results = []
    for (const item of items) {
      activeSourcePath = item.sourcePath
      const response = await page.goto(
        `${origin}${projectBase}${item.output}`,
        { waitUntil: "domcontentloaded", timeout: 60_000 },
      )
      assert.equal(response?.status(), 200, item.sourcePath)
      const rendered = await page.evaluate(async () => {
        await window.MathJax.startup.promise
        await document.fonts?.ready
        return {
          containers: document.querySelectorAll("mjx-container").length,
          unprocessed: [...document.querySelectorAll(".arithmatex")]
            .filter((node) => !node.querySelector("mjx-container"))
            .map((node) => node.textContent.trim().slice(0, 240)),
          failures: [...document.querySelectorAll("mjx-merror")].map((node) => ({
            error: node.getAttribute("data-mjx-error") || node.textContent.trim(),
            latex: node.closest("mjx-math")?.getAttribute("data-latex") || "",
          })),
        }
      })
      const remaining = await page.evaluate(browserMathOrigins, null)
      results.push({ ...item, ...rendered, remaining })
    }
    return results
  } finally {
    await page.close()
  }
}

export async function runMathJaxCorpus({ repoRoot = defaultRepoRoot } = {}) {
  const siteRoot = path.join(repoRoot, "site")
  const routeManifest = path.join(repoRoot, ".mkdocs-cache", "routes.json")
  const searchIndex = path.join(siteRoot, "search", "search_index.json")
  await Promise.all([
    access(path.join(siteRoot, "index.html")),
    access(routeManifest),
    access(searchIndex),
    access(path.join(siteRoot, "assets", "mathjax.js")),
    access(path.join(siteRoot, "assets", "vendor", "mathjax", "tex-chtml.js")),
    access(puppeteerPackage),
  ])

  const puppeteer = createRequire(import.meta.url)(
    path.join(nodeModules, "puppeteer-core"),
  )
  let browser
  let server
  try {
    browser = await puppeteer.launch({
      executablePath: await findBrowserExecutable(),
      headless: true,
      args: ["--disable-background-networking", "--no-sandbox"],
    })
    const analyzer = await browser.newPage()
    const { inspectedPages, pages } = await corpusPages({
      analyzer,
      repoRoot,
      siteRoot,
      manifest: routeManifest,
    })
    await analyzer.close()
    assert.ok(pages.length > 0, "generated site contains no MathJax-bearing pages")

    const started = await startStaticSite(siteRoot, projectBase)
    server = started.server
    const { origin } = started
    // Search behavior is covered by smoke.mjs. Preserve the generated schema here,
    // but omit its large document corpus so rapid formula-page traversal cannot
    // cancel unrelated search-index requests and report false asset failures.
    const generatedSearchIndex = JSON.parse(await readFile(searchIndex, "utf8"))
    const emptySearchIndex = JSON.stringify({ ...generatedSearchIndex, docs: [] })
    const externalRequests = []
    const failedRequests = []
    const pageErrors = []
    const workerCount = Math.min(4, pages.length)
    const assignments = Array.from({ length: workerCount }, () => [])
    pages.forEach((item, index) => assignments[index % workerCount].push(item))
    const results = (await Promise.all(assignments.map((items) => renderPages(
      browser,
      origin,
      items,
      externalRequests,
      failedRequests,
      pageErrors,
      emptySearchIndex,
    )))).flat()
    const formulas = results.reduce((total, result) => total + result.containers, 0)
    const mismatches = results.filter((result) => result.containers !== result.units)
    const failures = results.flatMap((result) => result.failures.map((failure) => ({
      sourcePath: result.sourcePath,
      ...failure,
    })))
    const unprocessed = results.flatMap((result) => result.unprocessed.map((source) => ({
      sourcePath: result.sourcePath,
      source,
    })))
    const rawRemainders = results.filter((result) => (
      result.remaining.rawDollarDelimiters > 0
      || result.remaining.rawDollarUnits > 0
      || result.remaining.rawTexDelimiters > 0
      || result.remaining.rawTexUnits > 0
    ))

    assert.ok(formulas > 0, "generated corpus contains no formulas")
    assert.deepEqual(
      mismatches,
      [],
      mismatches.map((item) => (
        `${item.sourcePath}: expected ${item.units} HTML math units `
        + `(${item.markerUnits} arithmatex + ${item.rawDollarUnits} raw dollar), `
        + `rendered ${item.containers} MathJax containers`
      )).join("\n"),
    )
    assert.deepEqual(externalRequests, [], "MathJax corpus requested external runtime assets")
    assert.deepEqual(failedRequests, [], "MathJax corpus emitted failed asset requests")
    assert.deepEqual(pageErrors, [], "MathJax corpus emitted uncaught page errors")
    assert.deepEqual(
      unprocessed,
      [],
      unprocessed.map((item) => `${item.sourcePath}: ${item.source}`).join("\n"),
    )
    assert.deepEqual(
      rawRemainders,
      [],
      rawRemainders.map((item) => (
        `${item.sourcePath}: ${item.remaining.rawDollarDelimiters} raw dollar delimiters `
        + `and ${item.remaining.rawTexDelimiters} raw TeX delimiters remain after MathJax`
      )).join("\n"),
    )
    assert.deepEqual(
      failures,
      [],
      failures.map((failure) => (
        `${failure.sourcePath}: ${failure.error}\n${failure.latex}`
      )).join("\n\n"),
    )
    console.log(
      `[docs:mkdocs:math] PASS: ${formulas} formulas across ${pages.length} pages `
      + `(${inspectedPages} manifest pages inspected)`,
    )
  } finally {
    await browser?.close().catch(() => undefined)
    if (server) await closeServer(server)
  }
}

function parseArguments(args) {
  let repoRoot = defaultRepoRoot
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] !== "--repo-root" || !args[index + 1]) {
      throw new Error(`unknown or incomplete argument: ${args[index]}`)
    }
    repoRoot = path.resolve(args[index + 1])
    index += 1
  }
  return { repoRoot }
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(scriptFile)) {
  Promise.resolve()
    .then(() => runMathJaxCorpus(parseArguments(process.argv.slice(2))))
    .catch((error) => {
      console.error(`[docs:mkdocs:math] ${error.stack || error.message}`)
      process.exitCode = 1
    })
}
