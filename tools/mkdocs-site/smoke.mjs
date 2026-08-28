import assert from "node:assert/strict"
import { spawn } from "node:child_process"
import { access, mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises"
import { createRequire } from "node:module"
import net from "node:net"
import { tmpdir } from "node:os"
import path from "node:path"
import { setTimeout as delay } from "node:timers/promises"
import { fileURLToPath } from "node:url"

import { findBrowserExecutable } from "../docs-site/listeners.mjs"

const scriptFile = fileURLToPath(import.meta.url)
const toolDir = path.dirname(scriptFile)
const repoRoot = path.resolve(toolDir, "..", "..")
const nodeModules = path.join(toolDir, "node_modules")
const puppeteerPackage = path.join(nodeModules, "puppeteer-core", "package.json")
const articlePath = "/02_engineering/02_train_frameworks/megatron-lm/13_megatron_cp_analysis.html"
const articleTitle = "Megatron-LM 上下文并行(Context Parallelism)深度解析"

export function rootMermaidSvgs(block) {
  return [...block.querySelectorAll("svg")].filter((svg) => {
    let parent = svg.parentElement
    while (parent && parent !== block) {
      if (String(parent.tagName).toLowerCase() === "svg") return false
      parent = parent.parentElement
    }
    return parent === block
  })
}

export function searchResultMatches(result, query, targetSuffix) {
  const url = new URL(result.href)
  return url.pathname.endsWith(targetSuffix)
    && (url.searchParams.get("h") === query || result.text.includes(query))
}

export function diagnosticSearchIndexUrl(origin) {
  return `${origin.replace(/\/$/, "")}/search/search_index.json`
}

function errorChain(error) {
  const details = []
  let current = error
  for (let depth = 0; current && depth < 4; depth += 1) {
    details.push(Object.fromEntries(
      ["name", "message", "code", "errno", "syscall", "address", "port"]
        .flatMap((key) => current[key] === undefined ? [] : [[key, current[key]]]),
    ))
    current = current.cause
  }
  return details
}

export async function fetchWithContext(url, options = {}, implementation = fetch) {
  try {
    return await implementation(url, options)
  } catch (error) {
    const method = options.method || "GET"
    throw new Error(
      `fetch ${method} ${url} failed: ${JSON.stringify(errorChain(error))}`,
      { cause: error },
    )
  }
}

export function mermaidRootContract(diagram) {
  return !diagram.error
    && diagram.roots === 1
    && diagram.role.includes("graphics-document")
    && Boolean(diagram.description)
    && Boolean(diagram.viewBox)
    && diagram.children > 0
    && diagram.rootWidth > 0
    && diagram.rootHeight > 0
    && Boolean(diagram.display)
    && diagram.display !== "none"
    && diagram.visibility === "visible"
}

async function loadPuppeteer() {
  try {
    await access(puppeteerPackage)
  } catch {
    throw new Error(
      `puppeteer-core is not installed at ${puppeteerPackage}. `
      + "Run npm ci --prefix tools/mkdocs-site before browser smoke tests.",
    )
  }
  return createRequire(import.meta.url)(path.join(nodeModules, "puppeteer-core"))
}

function closeServer(server, timeoutMs = 1_000) {
  return new Promise((resolve) => {
    let settled = false
    const finish = () => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolve()
    }
    const timer = setTimeout(() => {
      server.closeAllConnections?.()
      finish()
    }, timeoutMs)
    timer.unref?.()
    server.close(finish)
    server.closeIdleConnections?.()
  })
}

async function freeLoopbackPort() {
  const server = net.createServer()
  await new Promise((resolve, reject) => {
    server.once("error", reject)
    server.listen(0, "127.0.0.1", resolve)
  })
  const address = server.address()
  assert.ok(address && typeof address !== "string", "free-port listener has no TCP address")
  const port = address.port
  await closeServer(server)
  return port
}

async function hideBuiltSite() {
  const site = path.join(repoRoot, "site")
  const cache = path.join(repoRoot, ".mkdocs-cache")
  await mkdir(cache, { recursive: true })
  const backup = path.join(cache, `.smoke-site-${process.pid}-${Date.now()}`)
  try {
    await rename(site, backup)
    return { site, backup }
  } catch (error) {
    if (error.code === "ENOENT") return { site, backup: null }
    throw error
  }
}

async function restoreBuiltSite(saved) {
  await rm(saved.site, { recursive: true, force: true })
  if (saved.backup) await rename(saved.backup, saved.site)
}

async function assertPortReleased(port) {
  const server = net.createServer()
  await new Promise((resolve, reject) => {
    server.once("error", reject)
    server.listen({ port, host: "127.0.0.1", exclusive: true }, resolve)
  })
  await closeServer(server)
}

function boundedOutcome(promise, timeoutMs) {
  return Promise.race([
    promise.then(
      (value) => ({ kind: "value", value }),
      (error) => ({ kind: "error", error }),
    ),
    delay(timeoutMs).then(() => ({ kind: "timeout" })),
  ])
}

function startPythonService(args) {
  const output = []
  const python = process.env.PYTHON?.trim() || "python"
  const child = spawn(python, args, {
    cwd: repoRoot,
    detached: true,
    env: {
      ...process.env,
      PYTHONIOENCODING: "utf-8",
      PYTHONUNBUFFERED: "1",
    },
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  })
  for (const stream of [child.stdout, child.stderr]) {
    stream.setEncoding("utf8")
    stream.on("data", (chunk) => {
      output.push(chunk)
      while (output.join("").length > 100_000) output.shift()
    })
  }
  const exit = new Promise((resolve, reject) => {
    child.once("error", reject)
    child.once("close", (code, signal) => resolve({ code, signal }))
  })
  return { child, exit, output }
}

function startPreview(port) {
  return startPythonService([
    "-m",
    "tools.mkdocs_site.cli",
    "serve",
    "--port",
    String(port),
    "--no-open",
  ])
}

function startSearchFixture(repo, port) {
  return startPythonService([
    "-m",
    "tools.mkdocs_site.tests.serve_search_fixture",
    "--repo",
    repo,
    "--port",
    String(port),
  ])
}

async function runTaskkill(pid) {
  const killer = spawn("taskkill.exe", ["/PID", String(pid), "/T", "/F"], {
    stdio: "ignore",
    windowsHide: true,
  })
  const closed = new Promise((resolve) => killer.once("close", resolve))
  const result = await boundedOutcome(closed, 5_000)
  if (result.kind === "timeout") killer.kill()
}

async function stopPreview(service) {
  if (service.child.exitCode !== null || service.child.signalCode !== null) return
  if (process.platform === "win32") {
    await runTaskkill(service.child.pid)
  } else {
    try {
      process.kill(-service.child.pid, "SIGINT")
    } catch {
      service.child.kill("SIGINT")
    }
  }
  let stopped = await boundedOutcome(service.exit, 5_000)
  if (stopped.kind !== "timeout") return
  if (process.platform === "win32") {
    await runTaskkill(service.child.pid)
  } else {
    try {
      process.kill(-service.child.pid, "SIGKILL")
    } catch {
      service.child.kill("SIGKILL")
    }
  }
  stopped = await boundedOutcome(service.exit, 1_000)
  if (stopped.kind === "timeout") {
    throw new Error(`preview process ${service.child.pid} did not exit after forced cleanup`)
  }
}

async function waitForHttp(url, service, timeoutMs = 120_000) {
  const deadline = Date.now() + timeoutMs
  let lastFailure = "no response"
  while (Date.now() < deadline) {
    if (service.child.exitCode !== null || service.child.signalCode !== null) {
      const outcome = await boundedOutcome(service.exit, 0)
      throw new Error(`preview exited before readiness: ${JSON.stringify(outcome)}`)
    }
    try {
      const response = await fetchWithContext(url, {
        cache: "no-store",
        redirect: "manual",
        signal: AbortSignal.timeout(2_000),
      })
      if (response.status >= 200 && response.status < 400) return
      lastFailure = `HTTP ${response.status}`
    } catch (error) {
      lastFailure = error.message
    }
    await delay(100)
  }
  throw new Error(`timed out waiting for ${url} (${lastFailure})`)
}

async function waitForBody(url, expected, service, timeoutMs = 60_000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (service.child.exitCode !== null || service.child.signalCode !== null) {
      throw new Error(`preview exited while waiting for refreshed body: ${url}`)
    }
    try {
      const response = await fetch(url, {
        cache: "no-store",
        signal: AbortSignal.timeout(2_000),
      })
      if (response.ok && (await response.text()).includes(expected)) return
    } catch {
      // A controlled restart briefly closes the listener; keep polling.
    }
    await delay(100)
  }
  throw new Error(`timed out waiting for ${expected} at ${url}`)
}

async function goto(page, url) {
  const response = await page.goto(url, {
    waitUntil: "domcontentloaded",
    timeout: 60_000,
  })
  assert.ok(response, `navigation returned no response for ${url}`)
  assert.ok(
    response.status() >= 200 && response.status() < 400,
    `${url} returned HTTP ${response.status()}`,
  )
  await page.evaluate(async () => {
    await document.fonts?.ready
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))
  })
}

async function assertNoOverflow(page, width, label) {
  const metrics = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    innerWidth: window.innerWidth,
  }))
  assert.ok(
    metrics.scrollWidth <= metrics.innerWidth,
    `${label} overflows at ${width}px (${metrics.scrollWidth} > ${metrics.innerWidth})`,
  )
}

async function openFreshSearch(page, origin, targetSuffix) {
  await goto(page, `${origin}/`)
  await page.$eval('label[for="__search"]', (label) => label.click())
  await page.waitForSelector("[data-md-component='search-query']", { visible: true })
  const initialMeta = await page.$eval(
    ".md-search-result__meta",
    (element) => element.textContent.trim(),
  )
  if (initialMeta === "正在初始化搜索引擎") {
    await page.waitForFunction(
      (before) => document.querySelector(".md-search-result__meta")
        ?.textContent.trim() !== before,
      { timeout: 60_000 },
      initialMeta,
    )
  }
  const staleTarget = await page.evaluate((suffix) =>
    [...document.querySelectorAll("a.md-search-result__link")]
      .some((link) => new URL(link.href).pathname.endsWith(suffix)), targetSuffix)
  assert.equal(staleTarget, false, "fresh search page retained a previous target")
}

async function searchFor(page, origin, query, targetSuffix) {
  await openFreshSearch(page, origin, targetSuffix)
  const input = await page.$("[data-md-component='search-query']")
  assert.ok(input, "search input is missing")
  await input.evaluate((element, value) => {
    element.value = value
    element.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true }))
  }, query)
  try {
    await page.waitForFunction(
      (value, suffix) => [...document.querySelectorAll("a.md-search-result__link")]
        .some((link) => {
          const url = new URL(link.href)
          return url.pathname.endsWith(suffix)
            && (url.searchParams.get("h") === value || link.textContent.includes(value))
        }),
      { timeout: 60_000 },
      query,
      targetSuffix,
    )
  } catch (error) {
    const state = await page.evaluate(async (searchIndexUrl) => {
      const index = await fetch(searchIndexUrl).then((response) => response.json())
      return {
        query: document.querySelector("[data-md-component='search-query']")?.value,
        meta: document.querySelector(".md-search-result__meta")?.textContent.trim(),
        results: [...document.querySelectorAll("a.md-search-result__link")]
          .map((link) => ({ href: link.href, text: link.textContent.trim() })),
        target: index.docs.find((document) =>
          document.location.includes("13_megatron_cp_analysis")),
        config: index.config,
      }
    }, diagnosticSearchIndexUrl(origin))
    error.message += `\nsearch state: ${JSON.stringify(state)}`
    throw error
  }
  const candidates = await page.evaluate(() =>
    [...document.querySelectorAll("a.md-search-result__link")]
      .map((link) => ({ href: link.href, text: link.textContent.trim() })))
  const match = candidates.find((result) =>
    searchResultMatches(result, query, targetSuffix))
  assert.ok(match?.text, `search for ${query} returned no visible matching result`)
}

async function assertSearchMiss(page, origin, query, targetSuffix) {
  await openFreshSearch(page, origin, targetSuffix)
  const initialMeta = await page.$eval(
    ".md-search-result__meta",
    (element) => element.textContent.trim(),
  )
  await page.$eval("[data-md-component='search-query']", (element, value) => {
    element.value = value
    element.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true }))
  }, query)
  await page.waitForFunction(
    (value, before) => {
      const input = document.querySelector("[data-md-component='search-query']")
      const meta = document.querySelector(".md-search-result__meta")
      return input?.value === value && meta?.textContent.trim() !== before
    },
    { timeout: 60_000 },
    query,
    initialMeta,
  )
  const targetVisible = await page.evaluate((suffix) =>
    [...document.querySelectorAll("a.md-search-result__link")]
      .some((link) => new URL(link.href).pathname.endsWith(suffix)), targetSuffix)
  assert.equal(targetVisible, false, `impossible search ${query} retained the target`)
}

export async function assertSearchSurvivesRefresh(puppeteer, executablePath) {
  const fixture = await mkdtemp(path.join(tmpdir(), "mkdocs-search-refresh-"))
  const wiki = path.join(fixture, "wiki")
  const domain = path.join(wiki, "domain")
  const target = path.join(domain, "13_megatron_cp_analysis.md")
  const targetSuffix = "/domain/13_megatron_cp_analysis.html"
  const query = "13_megatron_cp_analysis"
  const port = await freeLoopbackPort()
  const origin = `http://127.0.0.1:${port}`
  let browser
  let service
  try {
    await mkdir(domain, { recursive: true })
    await writeFile(path.join(wiki, "index.md"), "# Fixture Home\n")
    await writeFile(path.join(domain, "index.md"), "# Fixture Domain\n")
    await writeFile(
      target,
      "---\ntitle: 上下文并行测试\n---\n# 上下文并行测试\ninitial body\n",
    )
    await assert.rejects(
      access(path.join(fixture, "site")),
      (error) => error.code === "ENOENT",
      "fixture site must be absent before cli serve",
    )
    service = startSearchFixture(fixture, port)
    await waitForHttp(`${origin}/`, service)
    browser = await puppeteer.launch({ executablePath, headless: true })
    const page = await browser.newPage()
    await searchFor(page, origin, query, targetSuffix)

    await writeFile(
      target,
      "---\ntitle: 上下文并行测试\n---\n# 上下文并行测试\nrefresh marker round two\n",
    )
    await waitForBody(
      `${origin}${targetSuffix}`,
      "refresh marker round two",
      service,
    )
    await searchFor(page, origin, query, targetSuffix)
  } catch (error) {
    error.message += `\n${service?.output.join("").slice(-20_000) ?? ""}`
    throw error
  } finally {
    await closeBrowser(browser).catch(() => undefined)
    try {
      if (service) await stopPreview(service)
      await assertPortReleased(port)
    } finally {
      await rm(fixture, { recursive: true, force: true })
    }
  }
}

async function assertHead(url, label) {
  const target = url.split("#", 1)[0]
  const response = await fetchWithContext(target, {
    method: "HEAD",
    redirect: "manual",
    signal: AbortSignal.timeout(5_000),
  })
  assert.ok(
    response.status >= 200 && response.status < 400,
    `${label} target ${url} returned HTTP ${response.status}`,
  )
}

async function assertArticleContracts(page, origin) {
  await goto(page, `${origin}${articlePath}`)
  const state = await page.evaluate(() => {
    const active = document.querySelector(
      'a[data-nav-title="13_megatron_cp_analysis"]',
    )
    const siblingTitles = ["12_megatron_tp_analysis", "14_megatron_ep_analysis"]
    const siblings = siblingTitles.map((title) => {
      const link = document.querySelector(`a[data-nav-title="${title}"]`)
      return link ? { title, href: link.href } : null
    })
    const theory = document.querySelector(
      'a[data-nav-title="理论研究 — 知识地图"]',
    )
    const pathLabels = [...document.querySelectorAll(".md-nav__item.kb-active-path")]
      .map((item) => item.textContent.trim())
    return {
      h1: document.querySelector("article h1")?.textContent.trim(),
      activeText: active?.textContent.trim(),
      activePruned: active?.closest(".md-nav__item")?.classList.contains("md-nav__item--pruned"),
      siblings,
      theoryPruned: theory?.closest(".md-nav__item")?.classList.contains("md-nav__item--pruned"),
      pathLabels,
    }
  })
  assert.equal(state.h1, articleTitle, "article H1 lost its Chinese title")
  assert.equal(state.activeText, "13_megatron_cp_analysis", "Megatron CP nav label drifted")
  assert.equal(state.activePruned, false, "current Megatron branch is pruned")
  assert.ok(state.siblings.every(Boolean), "representative Megatron siblings are missing")
  assert.equal(state.theoryPruned, true, "unrelated theory branch is not pruned")
  assert.ok(
    state.pathLabels.some((label) => label.includes("Megatron-LM")),
    "current navigation path did not render",
  )

  const links = await page.evaluate(() => {
    const currentOrigin = location.origin
    const internal = (selector) => [...document.querySelectorAll(selector)]
      .filter((link) => link instanceof HTMLAnchorElement && link.href)
      .map((link) => link.href)
      .filter((href) => new URL(href).origin === currentOrigin)
    const body = internal("article a[href]").filter((href) => new URL(href).pathname !== location.pathname)
    const breadcrumbs = internal(".md-path a[href]")
    const previous = internal("a.md-footer__link--prev")
    const next = internal("a.md-footer__link--next")
    const source = document.querySelector(
      'a.md-content__button[title="查看本页的源代码"]',
    )
    return {
      body: [...new Set(body)].slice(0, 12),
      breadcrumbs: [...new Set(breadcrumbs)],
      previous,
      next,
      source: source?.href,
    }
  })
  assert.ok(links.body.length > 0, "article exposes no representative internal links")
  assert.ok(links.breadcrumbs.length >= 3, "article breadcrumbs are missing")
  assert.equal(links.previous.length, 1, "article previous-page link is missing")
  assert.equal(links.next.length, 1, "article next-page link is missing")
  assert.equal(
    links.source,
    "https://github.com/suhaibo666/llm-knowledge/raw/main/wiki/"
      + "02_engineering/02_train_frameworks/megatron-lm/13_megatron_cp_analysis.md",
    "GitHub source target drifted",
  )
  const representatives = [...new Set([
    ...links.body,
    ...links.breadcrumbs,
    ...links.previous,
    ...links.next,
  ])]
  await Promise.all(representatives.map((url) => assertHead(url, "internal article link")))
}

async function assertSearch(page, origin) {
  await searchFor(page, origin, "跨多节点超长序列", articlePath)
  await assertSearchMiss(page, origin, "__kb_no_matching_document_7f9c__", articlePath)
  await searchFor(page, origin, "13_megatron_cp_analysis", articlePath)
}

async function assertRenderers(page, origin) {
  await goto(
    page,
    `${origin}/01_theory/06_distributed_parallelism/10_collectives_analysis.html`,
  )
  await page.waitForSelector("mjx-container", { timeout: 30_000 })
  const math = await page.evaluate(() => ({
    count: document.querySelectorAll("mjx-container").length,
    errors: document.querySelectorAll("mjx-merror").length,
  }))
  assert.ok(math.count > 0, "MathJax produced no containers")
  assert.equal(math.errors, 0, "MathJax produced an error node")

  await goto(
    page,
    `${origin}/02_engineering/04_posttrain_frameworks/verl/02_verl_quickstart_guide.html`,
  )
  await page.waitForFunction(
    () => [...document.querySelectorAll(".mermaid")]
      .filter((block) => block.querySelector("svg")).length >= 2,
    { timeout: 30_000 },
  )
  const mermaid = await page.evaluate(() => {
    // Keep this browser-side traversal aligned with rootMermaidSvgs(): nested
    // SVG symbols/labels are not independent diagrams and can have zero boxes.
    const rootsFor = (block) => [...block.querySelectorAll("svg")].filter((svg) => {
      let parent = svg.parentElement
      while (parent && parent !== block) {
        if (parent.tagName.toLowerCase() === "svg") return false
        parent = parent.parentElement
      }
      return parent === block
    })
    return [...document.querySelectorAll(".mermaid")].map((block) => {
      const roots = rootsFor(block)
      const svg = roots[0]
      const blockBox = block.getBoundingClientRect()
      const rootBox = svg?.getBoundingClientRect()
      const style = svg ? getComputedStyle(svg) : null
      return {
        error: block.dataset.kbMermaidError ?? null,
        roots: roots.length,
        role: svg?.getAttribute("role") ?? "",
        description: svg?.getAttribute("aria-roledescription") ?? "",
        viewBox: svg?.getAttribute("viewBox") ?? "",
        children: svg?.childElementCount ?? 0,
        rootWidth: rootBox?.width ?? 0,
        rootHeight: rootBox?.height ?? 0,
        display: style?.display ?? "",
        visibility: style?.visibility ?? "",
        blockWidth: blockBox.width,
        blockHeight: blockBox.height,
      }
    })
  })
  assert.ok(mermaid.length >= 2, "expected two representative Mermaid blocks")
  assert.ok(
    mermaid.every(mermaidRootContract),
    `Mermaid root contract failed: ${JSON.stringify(mermaid)}`,
  )
}

async function assertPaletteAndDrawer(page, origin) {
  await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 })
  await goto(page, `${origin}/`)
  await page.$eval('label[for="__palette_1"]', (label) => label.click())
  await page.waitForFunction(() => document.body.dataset.mdColorScheme === "slate")
  await page.$eval('label[for="__palette_0"]', (label) => label.click())
  await page.waitForFunction(() => document.body.dataset.mdColorScheme === "default")

  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 1 })
  await goto(page, `${origin}/`)
  const initial = await page.$eval("#__drawer", (input) => input.checked)
  assert.equal(initial, false, "mobile navigation drawer begins open")
  await page.$eval('label.md-header__button[for="__drawer"]', (label) => label.click())
  await page.waitForFunction(() => document.querySelector("#__drawer")?.checked === true)
  const drawer = await page.evaluate(() => {
    const sidebar = document.querySelector(".md-sidebar--primary")
    return {
      checked: document.querySelector("#__drawer")?.checked,
      width: sidebar?.getBoundingClientRect().width ?? 0,
    }
  })
  assert.equal(drawer.checked, true, "mobile navigation drawer did not open")
  assert.ok(drawer.width > 0, "mobile navigation drawer has no rendered width")
}

async function assertResponsivePages(page, origin) {
  for (const width of [390, 768, 1280, 1600]) {
    await page.setViewport({ width, height: 900, deviceScaleFactor: 1 })
    await goto(page, `${origin}/`)
    assert.ok(await page.$(".kb-source-atlas"), "homepage is missing Source Atlas")
    assert.equal(
      await page.$(".md-sidebar--secondary"),
      null,
      "homepage rendered a secondary table of contents",
    )
    await assertNoOverflow(page, width, "homepage")

    await goto(page, `${origin}${articlePath}`)
    await assertNoOverflow(page, width, "Megatron article")
  }
}

async function closeBrowser(browser) {
  if (!browser) return
  const outcome = await boundedOutcome(browser.close(), 5_000)
  if (outcome.kind === "timeout") {
    browser.process()?.kill("SIGKILL")
  } else if (outcome.kind === "error") {
    throw outcome.error
  }
}

export async function runSmoke() {
  const puppeteer = await loadPuppeteer()
  const executablePath = await findBrowserExecutable()
  await assertSearchSurvivesRefresh(puppeteer, executablePath)
  const port = await freeLoopbackPort()
  const origin = `http://127.0.0.1:${port}`
  const baseUrl = `${origin}/llm-knowledge`
  const savedSite = await hideBuiltSite()
  const service = startPreview(port)
  let browser
  const consoleMessages = []
  const pageErrors = []
  const failedRequests = []
  const failedResponses = []
  const externalRequests = []
  const requestUrls = []
  try {
    await waitForHttp(`${baseUrl}/`, service)
    browser = await puppeteer.launch({
      executablePath,
      headless: true,
      args: [
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-default-apps",
        "--disable-sync",
        "--no-first-run",
      ],
    })
    const page = await browser.newPage()
    await page.setCacheEnabled(false)
    await page.setRequestInterception(true)
    page.on("request", (request) => {
      const rawUrl = request.url()
      requestUrls.push(rawUrl)
      const url = new URL(rawUrl)
      if (
        ["http:", "https:", "ws:", "wss:"].includes(url.protocol)
        && url.hostname !== "127.0.0.1"
      ) {
        externalRequests.push(rawUrl)
        request.abort("blockedbyclient").catch(() => undefined)
      } else {
        request.continue().catch(() => undefined)
      }
    })
    page.on("requestfailed", (request) => {
      const url = new URL(request.url())
      const errorText = request.failure()?.errorText ?? "unknown failure"
      if (
        url.hostname === "127.0.0.1"
        && url.pathname.startsWith("/livereload/")
        && errorText === "net::ERR_ABORTED"
      ) return
      failedRequests.push(
        `${request.url()}: ${errorText}`,
      )
    })
    page.on("response", (response) => {
      if (response.status() >= 400 && new URL(response.url()).hostname === "127.0.0.1") {
        failedResponses.push(`${response.status()} ${response.url()}`)
      }
    })
    page.on("console", (message) => {
      if (["error", "warn", "warning"].includes(message.type())) {
        consoleMessages.push(`${message.type()}: ${message.text()}`)
      }
    })
    page.on("pageerror", (error) => pageErrors.push(error.message))

    await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 })
    await goto(page, `${baseUrl}/02_engineering/index.html`)
    assert.equal(
      await page.$eval("article h1", (heading) => heading.textContent.trim()),
      "工程实现 — 知识地图",
      "/02_engineering/ did not preserve its Chinese index title",
    )
    await assertArticleContracts(page, baseUrl)
    await assertSearch(page, baseUrl)
    await assertRenderers(page, baseUrl)
    await assertPaletteAndDrawer(page, baseUrl)
    await assertResponsivePages(page, baseUrl)

    assert.deepEqual(externalRequests, [], "browser attempted external runtime requests")
    assert.deepEqual(failedRequests, [], "browser emitted failed requests")
    assert.deepEqual(failedResponses, [], "browser received internal HTTP errors")
    assert.deepEqual(consoleMessages, [], "browser emitted console errors or warnings")
    assert.deepEqual(pageErrors, [], "browser emitted page errors")
    const localRendererRequests = requestUrls.filter((rawUrl) => {
      const url = new URL(rawUrl)
      return url.hostname === "127.0.0.1"
        && (url.pathname.includes("/assets/vendor/mathjax/")
          || url.pathname.includes("/assets/vendor/mermaid/"))
    })
    assert.ok(localRendererRequests.length > 0, "no local MathJax/Mermaid runtime was requested")
    console.log(
      `[docs:mkdocs:test] PASS: refresh, 390/768/1280/1600, search, links, renderers, themes, drawer (${requestUrls.length} requests)`,
    )
  } catch (error) {
    error.smokeDiagnostics = JSON.stringify({
      consoleMessages,
      pageErrors,
      failedRequests,
      failedResponses,
      externalRequests,
      previewProcess: {
        pid: service.child.pid,
        exitCode: service.child.exitCode,
        signalCode: service.child.signalCode,
        killed: service.child.killed,
      },
      previewOutput: service.output.join("").slice(-20_000),
    }, null, 2)
    throw error
  } finally {
    await closeBrowser(browser).catch(() => undefined)
    try {
      await stopPreview(service)
      await assertPortReleased(port)
    } finally {
      await restoreBuiltSite(savedSite)
    }
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(scriptFile)) {
  runSmoke().catch((error) => {
    console.error(
      `[docs:mkdocs:test] ${error.stack ?? error.message}`
      + (error.smokeDiagnostics ? `\nsmoke diagnostics:\n${error.smokeDiagnostics}` : ""),
    )
    process.exitCode = 1
  })
}
