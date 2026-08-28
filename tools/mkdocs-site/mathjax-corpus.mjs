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

function maskInlineCode(line) {
  const chars = [...line]
  let index = 0
  while (index < line.length) {
    if (line[index] !== "`" || (index > 0 && line[index - 1] === "\\")) {
      index += 1
      continue
    }
    let runEnd = index
    while (runEnd < line.length && line[runEnd] === "`") runEnd += 1
    const marker = line.slice(index, runEnd)
    const closing = line.indexOf(marker, runEnd)
    if (closing < 0) {
      index = runEnd
      continue
    }
    chars.fill(" ", index, closing + marker.length)
    index = closing + marker.length
  }
  return chars.join("")
}

function maskHtmlComments(line, state) {
  const chars = [...line]
  let index = 0
  while (index < line.length) {
    if (state.inComment) {
      const close = line.indexOf("-->", index)
      if (close < 0) {
        chars.fill(" ", index)
        return chars.join("")
      }
      chars.fill(" ", index, close + 3)
      state.inComment = false
      index = close + 3
      continue
    }
    const open = line.indexOf("<!--", index)
    if (open < 0) break
    const close = line.indexOf("-->", open + 4)
    if (close < 0) {
      chars.fill(" ", open)
      state.inComment = true
      break
    }
    chars.fill(" ", open, close + 3)
    index = close + 3
  }
  return chars.join("")
}

function stripBlockquotePrefix(line) {
  return line.replace(/^ {0,3}(?:>\s?)+/, "")
}

function openingFence(line) {
  const match = line.match(/^ {0,3}(`{3,}|~{3,})(.*)$/)
  if (!match || (match[1][0] === "`" && match[2].includes("`"))) return null
  return { character: match[1][0], length: match[1].length }
}

function closesFence(line, fence) {
  const match = line.match(/^ {0,3}(`{3,}|~{3,})[ \t]*$/)
  return Boolean(
    match
    && match[1][0] === fence.character
    && match[1].length >= fence.length,
  )
}

function isIndentedCode(line) {
  return /^(?: {4}|\t)/.test(line)
}

function unescapedDoubleDollars(line) {
  const positions = []
  let index = 0
  while (index < line.length - 1) {
    if (line.slice(index, index + 2) === "$$" && (index === 0 || line[index - 1] !== "\\")) {
      positions.push(index)
      index += 2
    } else {
      index += 1
    }
  }
  return positions
}

function nextSingleDollar(text, start) {
  let index = start
  while (index < text.length) {
    if (text[index] !== "$" || (index > 0 && text[index - 1] === "\\")) {
      index += 1
      continue
    }
    if (text[index + 1] === "$") {
      index += 2
      continue
    }
    return index
  }
  return -1
}

const currencyPattern = /^\$(?:\d[\d,]*(?:\.\d+)?)(?:\s?(?:USD|CNY|RMB|[KMBT]))?\b/i
const currencyTailPattern = /^[\s|,.;:)\]，。、；：）]*$/

function isCurrencyAmount(text, start, end) {
  if (text[end] === "$") return false
  const closer = nextSingleDollar(text, end)
  if (closer < 0) return true
  if (currencyPattern.test(text.slice(closer))) return true
  return currencyTailPattern.test(text.slice(end, closer))
}

function singleDollarPositions(text) {
  const positions = []
  let index = 0
  while (index < text.length) {
    if (text[index] !== "$" || (index > 0 && text[index - 1] === "\\")) {
      index += 1
      continue
    }
    if (text[index + 1] === "$") {
      index += 2
      continue
    }
    const currency = text.slice(index).match(currencyPattern)
    if (currency && isCurrencyAmount(text, index, index + currency[0].length)) {
      index += currency[0].length
      continue
    }
    positions.push(index)
    index += 1
  }
  return positions
}

function inlineMathInputs(text, sourcePath, line) {
  const positions = singleDollarPositions(text)
  assert.equal(
    positions.length % 2,
    0,
    `${sourcePath}:${line} has an unpaired inline '$' delimiter`,
  )
  return positions.flatMap((start, index) => (
    index % 2 === 0
      ? [{
          sourcePath,
          line,
          kind: "inline",
          latex: text.slice(start + 1, positions[index + 1]),
        }]
      : []
  ))
}

export function markdownMathInputs(markdown, sourcePath = "<memory>") {
  const inputs = []
  const lines = markdown.split(/\r?\n/)
  let frontmatterEnd = -1
  if (lines[0] === "---") {
    frontmatterEnd = lines.findIndex((line, index) => (
      index > 0 && ["---", "..."].includes(line)
    ))
  }

  let fence = null
  let inDisplay = false
  let displayLine = 0
  let displayParts = []
  const htmlCommentState = { inComment: false }
  for (let index = 0; index < lines.length; index += 1) {
    const lineNumber = index + 1
    if (frontmatterEnd >= 0 && index <= frontmatterEnd) continue
    const rawLine = lines[index]
    if (fence) {
      if (closesFence(stripBlockquotePrefix(rawLine), fence)) fence = null
      continue
    }

    const withoutInlineCode = maskInlineCode(rawLine)
    const withoutComments = maskHtmlComments(withoutInlineCode, htmlCommentState)
    const line = stripBlockquotePrefix(withoutComments)
    if (!inDisplay) {
      const openedFence = openingFence(line)
      if (openedFence) {
        fence = openedFence
        continue
      }
      if (isIndentedCode(line)) continue
    }

    const doublePositions = unescapedDoubleDollars(line)
    let cursor = 0
    for (const position of doublePositions) {
      const fragment = line.slice(cursor, position)
      if (inDisplay) {
        displayParts.push(fragment)
        inputs.push({
          sourcePath,
          line: displayLine,
          kind: "display",
          latex: displayParts.join("\n"),
        })
        displayParts = []
        inDisplay = false
      } else {
        inputs.push(...inlineMathInputs(fragment, sourcePath, lineNumber))
        inDisplay = true
        displayLine = lineNumber
      }
      cursor = position + 2
    }
    const tail = line.slice(cursor)
    if (inDisplay) displayParts.push(tail)
    else inputs.push(...inlineMathInputs(tail, sourcePath, lineNumber))
  }
  assert.equal(inDisplay, false, `${sourcePath}:${displayLine} has an unclosed '$$' delimiter`)
  return inputs
}

export async function corpusPages({
  repoRoot = defaultRepoRoot,
  siteRoot = path.join(repoRoot, "site"),
  wikiRoot = path.join(repoRoot, "wiki"),
  manifest = path.join(repoRoot, ".mkdocs-cache", "routes.json"),
} = {}) {
  const routes = JSON.parse(await readFile(manifest, "utf8"))
  const pages = []
  for (const route of routes) {
    const output = path.join(siteRoot, ...route.output.split("/"))
    const html = await readFile(output, "utf8")
    const sourcePath = `wiki/${route.source}`
    const markdown = await readFile(path.join(wikiRoot, ...route.source.split("/")), "utf8")
    const inputs = markdownMathInputs(markdown, sourcePath)
    if (inputs.length) {
      pages.push({
        sourcePath,
        output: route.output,
        sourceInputs: inputs.length,
        generatedMarkers: html.match(/\bclass="arithmatex"/g)?.length || 0,
      })
    }
  }
  return pages
}

async function renderPages(browser, origin, items, externalRequests, failedRequests) {
  const page = await browser.newPage()
  page.on("request", (request) => {
    const url = new URL(request.url())
    if (["http:", "https:"].includes(url.protocol) && url.origin !== origin) {
      externalRequests.push(request.url())
    }
  })
  page.on("response", (response) => {
    if (response.status() >= 400) {
      failedRequests.push(`${response.status()} ${response.url()}`)
    }
  })
  page.on("requestfailed", (request) => {
    failedRequests.push(`${request.url()}: ${request.failure()?.errorText || "failed"}`)
  })
  try {
    const results = []
    for (const item of items) {
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
      results.push({ ...item, ...rendered })
    }
    return results
  } finally {
    await page.close()
  }
}

export async function runMathJaxCorpus({ repoRoot = defaultRepoRoot } = {}) {
  const siteRoot = path.join(repoRoot, "site")
  const wikiRoot = path.join(repoRoot, "wiki")
  const routeManifest = path.join(repoRoot, ".mkdocs-cache", "routes.json")
  await Promise.all([
    access(path.join(siteRoot, "index.html")),
    access(routeManifest),
    access(path.join(siteRoot, "assets", "mathjax.js")),
    access(path.join(siteRoot, "assets", "vendor", "mathjax", "tex-chtml.js")),
    access(puppeteerPackage),
  ])
  const pages = await corpusPages({ repoRoot, siteRoot, wikiRoot, manifest: routeManifest })
  assert.ok(pages.length > 0, "generated site contains no MathJax-bearing pages")

  const puppeteer = createRequire(import.meta.url)(
    path.join(nodeModules, "puppeteer-core"),
  )
  const { server, origin } = await startStaticSite(siteRoot, projectBase)
  let browser
  try {
    browser = await puppeteer.launch({
      executablePath: await findBrowserExecutable(),
      headless: true,
      args: ["--disable-background-networking", "--no-sandbox"],
    })
    const externalRequests = []
    const failedRequests = []

    const workerCount = Math.min(4, pages.length)
    const assignments = Array.from({ length: workerCount }, () => [])
    pages.forEach((item, index) => assignments[index % workerCount].push(item))
    const results = (await Promise.all(assignments.map((items) => renderPages(
      browser,
      origin,
      items,
      externalRequests,
      failedRequests,
    )))).flat()
    const formulas = results.reduce((total, result) => total + result.containers, 0)
    const skipped = results.filter((result) => result.containers === 0)
    const failures = results.flatMap((result) => result.failures.map((failure) => ({
      sourcePath: result.sourcePath,
      ...failure,
    })))
    const unprocessed = results.flatMap((result) => result.unprocessed.map((source) => ({
      sourcePath: result.sourcePath,
      source,
    })))

    assert.ok(formulas > 0, "generated corpus contains no formulas")
    assert.deepEqual(
      skipped,
      [],
      skipped.map((item) => (
        `${item.sourcePath}: no MathJax containers for ${item.sourceInputs} source inputs `
        + `(${item.generatedMarkers} generated arithmatex markers)`
      )).join("\n"),
    )
    assert.deepEqual(externalRequests, [], "MathJax corpus requested external runtime assets")
    assert.deepEqual(failedRequests, [], "MathJax corpus emitted failed asset requests")
    assert.deepEqual(
      unprocessed,
      [],
      unprocessed.map((item) => `${item.sourcePath}: ${item.source}`).join("\n"),
    )
    assert.deepEqual(
      failures,
      [],
      failures.map((failure) => (
        `${failure.sourcePath}: ${failure.error}\n${failure.latex}`
      )).join("\n\n"),
    )
    console.log(
      `[docs:mkdocs:math] PASS: ${formulas} formulas across ${pages.length} pages`,
    )
  } finally {
    await browser?.close().catch(() => undefined)
    await closeServer(server)
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
