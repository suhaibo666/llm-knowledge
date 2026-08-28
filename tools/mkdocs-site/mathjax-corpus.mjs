import assert from "node:assert/strict"
import { createServer } from "node:http"
import { access, readFile } from "node:fs/promises"
import { createRequire } from "node:module"
import path from "node:path"
import { fileURLToPath } from "node:url"

import { findBrowserExecutable } from "../docs-site/listeners.mjs"

const scriptFile = fileURLToPath(import.meta.url)
const toolDir = path.dirname(scriptFile)
const repoRoot = path.resolve(toolDir, "..", "..")
const siteRoot = path.join(repoRoot, "site")
const routeManifest = path.join(repoRoot, ".mkdocs-cache", "routes.json")
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

function safeSitePath(url) {
  const parsed = new URL(url, "http://127.0.0.1")
  if (!parsed.pathname.startsWith(projectBase)) return null
  const relative = decodeURIComponent(parsed.pathname.slice(projectBase.length))
  const candidate = path.resolve(siteRoot, relative || "index.html")
  const prefix = `${path.resolve(siteRoot)}${path.sep}`
  if (candidate !== path.resolve(siteRoot) && !candidate.startsWith(prefix)) return null
  return candidate
}

async function startStaticSite() {
  const server = createServer(async (request, response) => {
    try {
      const candidate = safeSitePath(request.url || "/")
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

async function corpusPages() {
  const routes = JSON.parse(await readFile(routeManifest, "utf8"))
  const pages = []
  for (const route of routes) {
    const output = path.join(siteRoot, ...route.output.split("/"))
    const source = await readFile(output, "utf8")
    const formulas = source.match(/\bclass="arithmatex"/g)?.length || 0
    if (formulas) {
      pages.push({
        sourcePath: `wiki/${route.source}`,
        output: route.output,
        formulas,
      })
    }
  }
  return pages
}

async function renderPages(browser, origin, items, externalRequests, failedRequests) {
  const page = await browser.newPage()
  page.on("request", (request) => {
    const url = new URL(request.url())
    if (["http:", "https:"].includes(url.protocol) && url.hostname !== "127.0.0.1") {
      externalRequests.push(request.url())
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

export async function runMathJaxCorpus() {
  await Promise.all([
    access(path.join(siteRoot, "index.html")),
    access(routeManifest),
    access(path.join(siteRoot, "assets", "mathjax.js")),
    access(path.join(siteRoot, "assets", "vendor", "mathjax", "tex-chtml.js")),
    access(puppeteerPackage),
  ])
  const pages = await corpusPages()
  assert.ok(pages.length > 0, "generated site contains no MathJax-bearing pages")

  const puppeteer = createRequire(import.meta.url)(
    path.join(nodeModules, "puppeteer-core"),
  )
  const { server, origin } = await startStaticSite()
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
    const formulas = results.reduce((total, result) => total + result.formulas, 0)
    const containers = results.reduce((total, result) => total + result.containers, 0)
    const failures = results.flatMap((result) => result.failures.map((failure) => ({
      sourcePath: result.sourcePath,
      ...failure,
    })))
    const unprocessed = results.flatMap((result) => result.unprocessed.map((source) => ({
      sourcePath: result.sourcePath,
      source,
    })))

    assert.ok(formulas > 0, "generated corpus contains no formulas")
    assert.ok(
      containers >= formulas,
      `MathJax skipped generated formulas (${containers} containers for ${formulas} inputs)`,
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

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(scriptFile)) {
  runMathJaxCorpus().catch((error) => {
    console.error(`[docs:mkdocs:math] ${error.stack || error.message}`)
    process.exitCode = 1
  })
}
