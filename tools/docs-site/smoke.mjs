import assert from "node:assert/strict"
import { spawn } from "node:child_process"
import { access } from "node:fs/promises"
import path from "node:path"
import { setTimeout as delay } from "node:timers/promises"
import { fileURLToPath } from "node:url"
import { createRequire } from "node:module"

import { DOCS_STARTUP_TIMEOUT_MS, main as docsMain } from "./cli.mjs"
import { assertPortAvailable, startQuartz, waitForHttp } from "./processes.mjs"
import { assertLoopbackListeners, findBrowserExecutable } from "./listeners.mjs"
import { buildNpmInvocation } from "./runtime.mjs"

const scriptFile = fileURLToPath(import.meta.url)
const toolDir = path.dirname(scriptFile)
const repoRoot = path.resolve(toolDir, "..", "..")
const html2mdDir = path.join(repoRoot, "tools", "html2md")
const puppeteerPackage = path.join(
  html2mdDir,
  "node_modules",
  "puppeteer-core",
  "package.json",
)

function runInherited(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env ?? process.env,
      shell: options.shell ?? false,
      stdio: "inherit",
      windowsHide: true,
    })
    child.once("error", reject)
    child.once("close", (code) => {
      if (code === 0) resolve()
      else reject(new Error(`${command} exited with status ${code ?? "unknown"}`))
    })
  })
}

async function ensurePuppeteer() {
  try {
    await access(puppeteerPackage)
  } catch {
    console.log("[docs:test] Installing the repository-local browser test dependencies...")
    const npm = buildNpmInvocation(["ci", "--prefix", html2mdDir])
    await runInherited(npm.command, npm.args, {
      cwd: repoRoot,
      shell: npm.shell,
      env: {
        ...process.env,
        npm_config_cache: path.join(repoRoot, ".cache", "llm-knowledge-docs", "npm-cache"),
      },
    })
  }

  const require = createRequire(import.meta.url)
  return require(path.join(html2mdDir, "node_modules", "puppeteer-core"))
}

async function findPortPair() {
  const first = 18_000 + ((process.pid * 37) % 20_000)
  for (let offset = 0; offset < 200; offset += 2) {
    const port = first + offset
    if (port > 65_534) break
    try {
      await assertPortAvailable(port)
      await assertPortAvailable(port + 1)
      return port
    } catch {
      // Try the next deterministic pair.
    }
  }
  throw new Error("Unable to find two free loopback ports for the documentation smoke test")
}

async function waitForValue(readValue, timeoutMs, message) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const value = readValue()
    if (value) return value
    await delay(25)
  }
  throw new Error(message)
}

function withTimeout(promise, timeoutMs) {
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve({ kind: "timeout" }), timeoutMs)
    promise.then((value) => {
      clearTimeout(timer)
      resolve({ kind: "value", value })
    })
  })
}

export function headingIdFromHash(hash) {
  const fragment = hash.startsWith("#") ? hash.slice(1) : hash
  return decodeURIComponent(fragment)
}

async function goto(page, url) {
  const response = await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 })
  assert.ok(response, `Navigation returned no response for ${url}`)
  assert.ok(response.status() >= 200 && response.status() < 400, `${url} returned HTTP ${response.status()}`)
  return response
}

async function assertFetchableLinks(page, selectors) {
  for (const [label, selector] of selectors) {
    const result = await page.evaluate(async ({ selector: linkSelector }) => {
      const link = document.querySelector(linkSelector)
      if (!(link instanceof HTMLAnchorElement)) return { missing: true }
      const response = await fetch(link.href)
      return { href: link.href, status: response.status }
    }, { selector })
    assert.equal(result.missing, undefined, `${label} link is missing (${selector})`)
    assert.ok(
      result.status >= 200 && result.status < 400,
      `${label} link ${result.href} returned HTTP ${result.status}`,
    )
  }
}

async function assertImage(page, selector, label) {
  await page.waitForSelector(selector, { timeout: 30_000 })
  const state = await page.$eval(selector, (image) => ({
    complete: image.complete,
    width: image.naturalWidth,
    height: image.naturalHeight,
    src: image.currentSrc || image.src,
  }))
  assert.equal(state.complete, true, `${label} did not finish loading: ${state.src}`)
  assert.ok(state.width > 0 && state.height > 0, `${label} has zero dimensions: ${state.src}`)
}

async function readLayoutMetrics(page) {
  return page.evaluate(() => {
    const shell = document.querySelector(".page")
    const body = document.querySelector(".page > #quartz-body")
    if (!(shell instanceof HTMLElement) || !(body instanceof HTMLElement)) {
      throw new Error("Documentation layout shell is missing")
    }

    const shellRect = shell.getBoundingClientRect()
    const gridColumns = getComputedStyle(body).gridTemplateColumns
      .split(/\s+/)
      .map((value) => Number.parseFloat(value))

    return {
      viewportWidth: window.innerWidth,
      shellLeft: shellRect.left,
      shellRight: window.innerWidth - shellRect.right,
      shellWidth: shellRect.width,
      gridColumns,
      scrollWidth: document.documentElement.scrollWidth,
    }
  })
}

async function assertResponsiveLayout(page, baseUrl) {
  const desktopCases = [
    { viewportWidth: 1280, columns: [256, 738.8, 224] },
    { viewportWidth: 1920, columns: [364.8, 1148.4, 320] },
    { viewportWidth: 2560, columns: [384, 1743.6, 320] },
  ]

  for (const expected of desktopCases) {
    await page.setViewport({ width: expected.viewportWidth, height: 1000, deviceScaleFactor: 1 })
    await goto(page, baseUrl)
    const actual = await readLayoutMetrics(page)

    assert.ok(
      Math.abs(actual.shellWidth - expected.viewportWidth * 0.96) <= 1,
      `${expected.viewportWidth}px viewport should use a 96vw page shell; got ${actual.shellWidth}px`,
    )
    assert.ok(
      Math.abs(actual.shellLeft - actual.shellRight) <= 1,
      `${expected.viewportWidth}px viewport has asymmetric outer gutters`,
    )
    assert.equal(actual.gridColumns.length, 3, "Desktop layout should have three columns")
    for (let index = 0; index < expected.columns.length; index += 1) {
      assert.ok(
        Math.abs(actual.gridColumns[index] - expected.columns[index]) <= 1,
        `${expected.viewportWidth}px viewport column ${index + 1} should be ${expected.columns[index]}px; got ${actual.gridColumns[index]}px`,
      )
    }
    assert.ok(
      actual.scrollWidth <= actual.viewportWidth,
      `${expected.viewportWidth}px viewport overflows horizontally to ${actual.scrollWidth}px`,
    )
  }

  for (const expected of [
    { viewportWidth: 1024, columnCount: 2 },
    { viewportWidth: 768, columnCount: 1 },
  ]) {
    await page.setViewport({ width: expected.viewportWidth, height: 1000, deviceScaleFactor: 1 })
    await goto(page, baseUrl)
    const actual = await readLayoutMetrics(page)

    assert.ok(
      actual.shellWidth >= expected.viewportWidth - 1,
      `${expected.viewportWidth}px viewport should retain the existing full-width shell`,
    )
    assert.equal(
      actual.gridColumns.length,
      expected.columnCount,
      `${expected.viewportWidth}px viewport should retain its existing grid mode`,
    )
    assert.ok(
      actual.scrollWidth <= actual.viewportWidth,
      `${expected.viewportWidth}px viewport overflows horizontally to ${actual.scrollWidth}px`,
    )
  }
}

export function localNetworkViolations(requestUrls) {
  return requestUrls.filter((rawUrl) => {
    const parsed = new URL(rawUrl)
    if (parsed.protocol === "data:") return false
    return !(["http:", "ws:"].includes(parsed.protocol) && parsed.hostname === "127.0.0.1")
  })
}

function assertLocalNetwork(requestUrls) {
  const violations = localNetworkViolations(requestUrls)
  assert.deepEqual(
    violations,
    [],
    `Non-loopback browser requests were observed: ${violations.join(", ")}`,
  )
}

async function runBrowserAssertions(page, baseUrl) {
  await goto(page, baseUrl)
  // frontmatter 的 title 现在决定 <title>；站点名仍由 .page-title 侧栏承载
  assert.equal(await page.title(), "LLM Knowledge Wiki — 知识库总索引")
  const siteTitle = await page.$eval(".page-title", (element) => element.textContent.trim())
  assert.match(siteTitle, /LLM Knowledge Wiki/)
  for (const selector of [".explorer", ".search-button", ".darkmode", "article"]) {
    assert.ok(await page.$(selector), `Homepage is missing ${selector}`)
  }

  // ---- 回归：子目录 index.md 必须渲染正文（曾因缺 note-properties 而全站空白）----
  const folderIndex = `${baseUrl}02_engineering/03_infer_frameworks/`
  await goto(page, folderIndex)
  assert.equal(
    await page.title(),
    "推理框架 —— 目录索引",
    "Folder index page must take its title from frontmatter, not the file stem",
  )
  const folderBody = await page.$eval("article", (el) => el.textContent.trim())
  assert.ok(
    folderBody.length > 500,
    `Folder index page rendered an empty body (${folderBody.length} chars) - the frontmatter transformer is probably missing`,
  )
  assert.ok(
    await page.$(".page-listing"),
    "Folder index page is missing the auto-generated page listing",
  )
  const folderHeadings = await page.$$eval("article h1", (nodes) => nodes.length)
  assert.equal(folderHeadings, 1, "Folder index page must render exactly one h1")

  const quickstart =
    `${baseUrl}02_engineering/04_posttrain_frameworks/verl/02_verl_quickstart_guide`
  await goto(page, quickstart)
  for (const selector of [
    "article",
    ".breadcrumb-container",
    ".toc",
    ".callout.note",
    ".table-container table",
    "pre code",
  ]) {
    assert.ok(await page.$(selector), `Quickstart page is missing ${selector}`)
  }
  await page.waitForSelector(".mermaid svg", { timeout: 30_000 })
  const mermaidState = await page.evaluate(() => ({
    count: document.querySelectorAll(".mermaid svg").length,
    visible: [...document.querySelectorAll(".mermaid svg")].every(
      (svg) => svg.getBoundingClientRect().width > 0 && svg.getBoundingClientRect().height > 0,
    ),
    htmlLabels: document.querySelectorAll(".mermaid svg foreignObject, .mermaid svg br").length,
  }))
  assert.ok(mermaidState.count >= 2, "Expected both real Mermaid diagrams to render")
  assert.equal(mermaidState.visible, true, "A Mermaid SVG has zero rendered dimensions")
  assert.ok(mermaidState.htmlLabels > 0, "Mermaid inline HTML labels were not preserved")

  await assertFetchableLinks(page, [
    [
      "Obsidian path-suffix",
      'a[data-slug="02_engineering/04_posttrain_frameworks/verl/index"]',
    ],
    [
      "bare wikilink",
      'a[data-slug="02_engineering/04_posttrain_frameworks/verl/10_verl_end_to_end_iteration_analysis"]',
    ],
    [
      "vault-rooted wikilink",
      'a[data-slug="02_engineering/04_posttrain_frameworks/index"]',
    ],
  ])
  const fragmentSelector = 'a[href="#1-适用场景与前置"]'
  assert.ok(await page.$(fragmentSelector), "Quickstart heading-fragment link is missing")
  await page.click(fragmentSelector)
  const fragmentId = headingIdFromHash("#1-适用场景与前置")
  await page.waitForFunction(
    (expectedId) => decodeURIComponent(location.hash.slice(1)) === expectedId,
    {},
    fragmentId,
  )
  assert.equal(
    await page.evaluate((id) => document.getElementById(id) !== null, fragmentId),
    true,
    "Heading-fragment target is missing",
  )

  await goto(
    page,
    `${baseUrl}01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis`,
  )
  const alias = await page.$eval("a.alias", (link) => ({
    text: link.textContent.trim(),
    href: link.href,
  }))
  assert.ok(alias.text.length > 0, "Escaped wikilink alias has no visible label")
  const aliasResponse = await page.evaluate(async (href) => (await fetch(href)).status, alias.href)
  assert.ok(aliasResponse >= 200 && aliasResponse < 400, `Alias link returned HTTP ${aliasResponse}`)

  await goto(page, `${baseUrl}01_theory/06_distributed_parallelism/10_collectives_analysis`)
  const mathState = await page.evaluate(() => {
    const containers = [...document.querySelectorAll('mjx-container[jax="SVG"]')]
    return {
      total: containers.filter((container) => container.querySelector("svg")).length,
      inline: containers.some((container) => container.parentElement?.childNodes.length > 1),
      block: containers.some((container) => container.parentElement?.childNodes.length === 1),
    }
  })
  assert.ok(mathState.total >= 2, "MathJax SVG output is missing")
  assert.equal(mathState.inline, true, "No inline MathJax expression was rendered")
  assert.equal(mathState.block, true, "No block MathJax expression was rendered")

  await goto(
    page,
    `${baseUrl}02_engineering/07_training_reliability/10_determinism_and_numerical_reliability_analysis`,
  )
  await assertImage(page, 'article img[src$=".png"]', "Representative PNG")

  await goto(
    page,
    `${baseUrl}01_theory/01_models/moonshot_kimi/20_gdn_kda_linear_attention_analysis`,
  )
  await assertImage(page, 'article img[src$=".svg"]', "Representative SVG")

  await page.waitForSelector(".search-layout .results-container", { timeout: 60_000 })
  await page.click(".search-button")
  await page.waitForSelector(".search-container.active", { timeout: 10_000 })
  await page.$eval(
    ".search-bar",
    (input, query) => {
      input.value = query
      input.dispatchEvent(new Event("input", { bubbles: true }))
    },
    "确定性",
  )
  await page.waitForSelector(".result-card:not(.no-match)", { timeout: 30_000 })
  const searchResult = await page.$eval(".result-card:not(.no-match)", (link) => ({
    href: link.href,
    text: link.textContent.trim(),
  }))
  assert.ok(searchResult.text.length > 0, "Search result has no visible text")
  const searchStatus = await page.evaluate(
    async (href) => (await fetch(href)).status,
    searchResult.href,
  )
  assert.ok(searchStatus >= 200 && searchStatus < 400, `Search result returned HTTP ${searchStatus}`)
}

export async function runSmoke() {
  const puppeteer = await ensurePuppeteer()
  const browserExecutable = await findBrowserExecutable()
  const port = await findPortPair()
  const baseUrl = `http://127.0.0.1:${port}/`
  let quartzService
  let browser
  let intentionallyStopped = false
  const requestUrls = []
  const browserErrors = []
  const failedRequests = []
  const failedResponses = []

  const servePromise = docsMain(
    // Pin the smoke run to loopback: the default is 0.0.0.0, and an
    // end-to-end test should not publish a port on the local network.
    ["serve", "--port", String(port), "--host", "127.0.0.1", "--no-open"],
    {
      repoRoot,
      startQuartz(options) {
        quartzService = startQuartz(options)
        return quartzService
      },
    },
  )
  const serveOutcome = servePromise.then(
    (code) => ({ kind: "exit", code }),
    (error) => ({ kind: "error", error }),
  )

  try {
    await waitForValue(
      () => quartzService,
      30_000,
      "The docs launcher did not start Quartz within 30 seconds",
    )
    await waitForHttp(baseUrl, { timeoutMs: DOCS_STARTUP_TIMEOUT_MS + 15_000 })
    const earlyOutcome = await withTimeout(serveOutcome, 0)
    if (earlyOutcome.kind === "value") {
      if (earlyOutcome.value.kind === "error") throw earlyOutcome.value.error
      throw new Error(`Documentation service exited early with status ${earlyOutcome.value.code}`)
    }

    await assertLoopbackListeners([port, port + 1], {
      expectedPid: quartzService.pid,
      expectedHost: "127.0.0.1",
    })

    browser = await puppeteer.launch({
      executablePath: browserExecutable,
      headless: true,
      args: ["--disable-background-networking", "--disable-component-update"],
    })
    const page = await browser.newPage()
    await page.setViewport({ width: 1440, height: 1000, deviceScaleFactor: 1 })
    page.on("pageerror", (error) => browserErrors.push(error.message))
    page.on("console", (message) => {
      if (message.type() === "error") browserErrors.push(message.text())
    })
    page.on("request", (request) => requestUrls.push(request.url()))
    page.on("requestfailed", (request) => {
      failedRequests.push(`${request.url()}: ${request.failure()?.errorText ?? "unknown failure"}`)
    })
    page.on("response", (response) => {
      if (response.status() >= 400) failedResponses.push(`${response.status()} ${response.url()}`)
    })
    const cdp = await page.createCDPSession()
    await cdp.send("Network.enable")
    cdp.on("Network.webSocketCreated", ({ url }) => requestUrls.push(url))

    await assertResponsiveLayout(page, baseUrl)
    await page.setViewport({ width: 1440, height: 1000, deviceScaleFactor: 1 })
    await runBrowserAssertions(page, baseUrl)
    assertLocalNetwork(requestUrls)
    assert.deepEqual(failedRequests, [], `Failed browser requests:\n${failedRequests.join("\n")}`)
    assert.deepEqual(failedResponses, [], `HTTP error responses:\n${failedResponses.join("\n")}`)
    assert.deepEqual(browserErrors, [], `Browser errors:\n${browserErrors.join("\n")}`)
    console.log(
      `[docs:test] PASS: browser rendering, search, links, assets, and loopback-only network (${requestUrls.length} requests)`,
    )
  } catch (error) {
    const diagnostics = [
      browserErrors.length ? `Browser errors:\n${browserErrors.join("\n")}` : "",
      failedRequests.length ? `Failed requests:\n${failedRequests.join("\n")}` : "",
      failedResponses.length ? `Failed responses:\n${failedResponses.join("\n")}` : "",
    ].filter(Boolean).join("\n")
    if (diagnostics) error.message += `\n${diagnostics}`
    throw error
  } finally {
    await browser?.close().catch(() => undefined)
    if (quartzService) {
      intentionallyStopped = true
      quartzService.terminate("SIGINT")
      let stopped = await withTimeout(serveOutcome, 5_000)
      if (stopped.kind === "timeout") {
        quartzService.forceKill?.()
        stopped = await withTimeout(serveOutcome, 1_000)
      }
      if (stopped.kind === "timeout") {
        console.warn(`[docs:test] Quartz process ${quartzService.pid} did not report exit`)
      }
    }
  }

  const finalOutcome = await serveOutcome
  if (finalOutcome.kind === "error") throw finalOutcome.error
  const expectedStopCodes = intentionallyStopped ? new Set([0, 130, 143]) : new Set([0])
  if (!expectedStopCodes.has(finalOutcome.code)) {
    throw new Error(`Documentation service exited with status ${finalOutcome.code}`)
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(scriptFile)) {
  runSmoke().catch((error) => {
    console.error(`[docs:test] ${error.stack ?? error.message}`)
    process.exitCode = 1
  })
}
