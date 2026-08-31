import assert from "node:assert/strict"
import { access, readFile, stat } from "node:fs/promises"
import { createServer } from "node:http"
import { createRequire } from "node:module"
import path from "node:path"
import { setTimeout as delay } from "node:timers/promises"
import { fileURLToPath } from "node:url"

import { findBrowserExecutable } from "../../docs-site/listeners.mjs"

const scriptFile = fileURLToPath(import.meta.url)
const repoRoot = path.resolve(path.dirname(scriptFile), "..", "..", "..")
const mkdocsSiteDir = path.join(repoRoot, "tools", "mkdocs-site")
const rendererNodeModules = path.join(mkdocsSiteDir, "node_modules")
const puppeteerPackage = path.join(
  rendererNodeModules,
  "puppeteer-core",
  "package.json",
)

async function loadPuppeteer() {
  try {
    await access(puppeteerPackage)
  } catch {
    throw new Error(
      `puppeteer-core is not installed at ${puppeteerPackage}. `
      + "Run npm ci --prefix tools/mkdocs-site before renderer contract tests.",
    )
  }

  const require = createRequire(import.meta.url)
  return require(path.join(rendererNodeModules, "puppeteer-core"))
}

const contentTypes = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".png", "image/png"],
  [".svg", "image/svg+xml"],
  [".woff2", "font/woff2"],
  [".xml", "application/xml; charset=utf-8"],
])

function fixtureFile(siteRoot, requestUrl) {
  const url = new URL(requestUrl, "http://127.0.0.1")
  let pathname = decodeURIComponent(url.pathname)
  const projectPath = "/llm-knowledge"
  if (pathname === projectPath || pathname.startsWith(`${projectPath}/`)) {
    pathname = pathname.slice(projectPath.length) || "/"
  }
  if (pathname.endsWith("/")) pathname += "index.html"
  const candidate = path.resolve(siteRoot, pathname.replace(/^\/+/, ""))
  const relative = path.relative(siteRoot, candidate)
  if (relative.startsWith("..") || path.isAbsolute(relative)) return undefined
  return candidate
}

function startFixtureServer(siteRoot, servedResponses) {
  const server = createServer(async (request, response) => {
    const requestPath = new URL(
      request.url ?? "/",
      "http://127.0.0.1",
    ).pathname
    try {
      const candidate = fixtureFile(siteRoot, request.url ?? "/")
      if (!candidate || !(await stat(candidate)).isFile()) {
        servedResponses.set(requestPath, 404)
        response.writeHead(404, { "content-type": "text/plain; charset=utf-8" })
        response.end("Not found")
        return
      }
      const body = await readFile(candidate)
      servedResponses.set(requestPath, 200)
      response.writeHead(200, {
        "content-type": contentTypes.get(path.extname(candidate)) ?? "application/octet-stream",
        "content-security-policy": [
          "default-src 'self' data: blob:",
          "script-src 'self' 'unsafe-inline'",
          "style-src 'self' 'unsafe-inline'",
          "img-src 'self' data: blob:",
          "font-src 'self'",
          "worker-src 'self' blob:",
          "connect-src 'self'",
        ].join("; "),
      })
      response.end(body)
    } catch (error) {
      if (error?.code === "ENOENT") {
        servedResponses.set(requestPath, 404)
        response.writeHead(404, { "content-type": "text/plain; charset=utf-8" })
        response.end("Not found")
        return
      }
      servedResponses.set(requestPath, 500)
      response.writeHead(500, { "content-type": "text/plain; charset=utf-8" })
      response.end(error?.stack ?? String(error))
    }
  })
  return new Promise((resolve, reject) => {
    server.once("error", reject)
    server.listen(0, "127.0.0.1", () => resolve(server))
  })
}

function closeServer(server) {
  return new Promise((resolve) => {
    let settled = false
    const finish = () => {
      if (settled) return
      settled = true
      resolve()
    }
    server.close(finish)
    server.closeIdleConnections?.()
    setTimeout(() => {
      server.closeAllConnections?.()
      finish()
    }, 1_000).unref()
  })
}

function isRendererUrl(rawUrl) {
  const pathname = new URL(rawUrl).pathname
  return (
    pathname.endsWith("/assets/mathjax.js") ||
    pathname.endsWith("/assets/diagram.js") ||
    pathname.includes("/assets/vendor/mathjax/") ||
    pathname.includes("/assets/vendor/mathjax-newcm/") ||
    pathname.includes("/assets/vendor/mermaid/")
  )
}

const requiredRendererSuffixes = [
  "/assets/mathjax.js",
  "/assets/diagram.js",
  "/assets/vendor/mathjax/tex-chtml.js",
  "/assets/vendor/mathjax/input/tex/extensions/boldsymbol.js",
  "/assets/vendor/mathjax-newcm/chtml/dynamic/double-struck.js",
  "/assets/vendor/mathjax/sre/speech-worker.js",
  "/assets/vendor/mathjax/sre/mathmaps/base.json",
  "/assets/vendor/mathjax/sre/mathmaps/en.json",
  "/assets/vendor/mathjax/sre/mathmaps/nemeth.json",
  "/assets/vendor/mermaid/mermaid.min.js",
]

function recordedStatus(rawUrl, responses, servedResponses) {
  return responses.get(rawUrl) ?? servedResponses.get(new URL(rawUrl).pathname)
}

function rendererActivityReady(requests, responses, servedResponses) {
  const rendererRequests = [...new Set(requests.filter(isRendererUrl))]
  const rendererPaths = rendererRequests.map((rawUrl) => new URL(rawUrl).pathname)
  return (
    requiredRendererSuffixes.every((required) =>
      rendererPaths.some((pathname) => pathname.endsWith(required)),
    ) &&
    rendererPaths.some((pathname) =>
      pathname.includes("/assets/vendor/mathjax-newcm/chtml/woff2/"),
    ) &&
    rendererPaths.some((pathname) =>
      pathname.includes("/assets/vendor/mathjax/sre/mathmaps/"),
    ) &&
    rendererRequests.every(
      (rawUrl) => recordedStatus(rawUrl, responses, servedResponses) !== undefined,
    )
  )
}

async function waitForRendererActivity(
  requests,
  responses,
  servedResponses,
  timeoutMs = 10_000,
) {
  const deadline = Date.now() + timeoutMs
  let signature = ""
  let stableSince = 0
  while (Date.now() < deadline) {
    const currentSignature = [...new Set(requests.filter(isRendererUrl))]
      .sort()
      .join("\n")
    if (
      rendererActivityReady(requests, responses, servedResponses) &&
      currentSignature === signature
    ) {
      if (Date.now() - stableSince >= 1_000) return
    } else {
      signature = currentSignature
      stableSince = Date.now()
    }
    await delay(50)
  }
}

function isExpectedMermaidError(entry) {
  return (
    entry.type === "error" &&
    entry.text.startsWith("Mermaid render failed: Parse error")
  )
}

function isExpectedMermaidPageError(message) {
  return message.startsWith("Parse error on line") && message.includes("got 'EOF'")
}

async function assertMermaidViewerLifecycle(page) {
  const triggerSelector = (
    ".mermaid:not([data-kb-mermaid-error]) .kb-mermaid-zoom-trigger"
  )
  const triggerCount = await page.$$eval(
    ".kb-mermaid-zoom-trigger",
    (items) => items.length,
  )
  assert.equal(triggerCount, 1, "only rendered Mermaid diagrams get a zoom trigger")

  const triggerLabel = await page.$eval(
    triggerSelector,
    (trigger) => trigger.getAttribute("aria-label"),
  )
  assert.equal(triggerLabel, "查看大图", "Mermaid zoom trigger has no accessible label")

  await page.setViewport({ width: 390, height: 780, deviceScaleFactor: 1 })
  await page.click(triggerSelector)
  await page.waitForFunction(() => document.querySelector("dialog.kb-mermaid-viewer")?.open)
  const opened = await page.evaluate(() => {
    const viewer = document.querySelector("dialog.kb-mermaid-viewer")
    return {
      viewers: document.querySelectorAll("dialog.kb-mermaid-viewer").length,
      modal: viewer?.getAttribute("aria-modal"),
      label: viewer?.getAttribute("aria-label"),
      svg: viewer?.querySelectorAll(".kb-mermaid-viewer__canvas > svg").length ?? 0,
      bodyLocked: document.body.classList.contains("kb-mermaid-viewer-open"),
    }
  })
  assert.deepEqual(opened, {
    viewers: 1,
    modal: "true",
    label: "Mermaid 图表查看器",
    svg: 1,
    bodyLocked: true,
  })
  const layout = await page.evaluate((selector) => {
    const viewer = document.querySelector("dialog.kb-mermaid-viewer")
    const stage = viewer.querySelector(".kb-mermaid-viewer__stage")
    const toolbar = viewer.querySelector(".kb-mermaid-viewer__toolbar")
    const trigger = document.querySelector(selector)
    const viewerBox = viewer.getBoundingClientRect()
    const stageBox = stage.getBoundingClientRect()
    const triggerBox = trigger.getBoundingClientRect()
    const stageStyle = getComputedStyle(stage)
    return {
      viewerCoversViewport:
        Math.abs(viewerBox.left) <= 1
        && Math.abs(viewerBox.top) <= 1
        && viewerBox.width >= window.innerWidth - 1
        && viewerBox.height >= window.innerHeight - 1,
      stageUsable: stageBox.height >= window.innerHeight * 0.55,
      stageOverflow: stageStyle.overflow,
      stageTouchAction: stageStyle.touchAction,
      toolbarFits: toolbar.scrollWidth <= toolbar.clientWidth,
      triggerUsable: triggerBox.width >= 32 && triggerBox.height >= 32,
      bodyOverflow: getComputedStyle(document.body).overflow,
    }
  }, triggerSelector)
  assert.deepEqual(layout, {
    viewerCoversViewport: true,
    stageUsable: true,
    stageOverflow: "hidden",
    stageTouchAction: "none",
    toolbarFits: true,
    triggerUsable: true,
    bodyOverflow: "hidden",
  })

  const actions = await page.$$eval(
    "[data-kb-mermaid-action]",
    (buttons) => buttons.map((button) => button.dataset.kbMermaidAction),
  )
  assert.deepEqual(actions, ["zoom-out", "zoom-in", "fit", "reset", "close"])
  const initialTransform = await page.$eval(".kb-mermaid-viewer", (viewer) => ({
    mode: viewer.dataset.kbMode,
    scale: Number(viewer.dataset.kbScale),
    x: Number(viewer.dataset.kbX),
    y: Number(viewer.dataset.kbY),
  }))
  assert.equal(initialTransform.mode, "fit", "viewer did not fit on open")
  assert.ok(initialTransform.scale > 0, "viewer has no initial scale")

  await page.click('[data-kb-mermaid-action="zoom-in"]')
  await page.waitForFunction((before) => (
    Number(document.querySelector(".kb-mermaid-viewer")?.dataset.kbScale) > before
  ), { timeout: 5_000 }, initialTransform.scale)
  const buttonScale = await page.$eval(
    ".kb-mermaid-viewer",
    (viewer) => Number(viewer.dataset.kbScale),
  )

  const stage = await page.$(".kb-mermaid-viewer__stage")
  const stageBox = await stage.boundingBox()
  assert.ok(stageBox && stageBox.width > 0 && stageBox.height > 0, "viewer stage is hidden")
  await page.mouse.move(stageBox.x + stageBox.width / 2, stageBox.y + stageBox.height / 2)
  await page.mouse.wheel({ deltaY: -180 })
  await page.waitForFunction((before) => (
    Number(document.querySelector(".kb-mermaid-viewer")?.dataset.kbScale) > before
  ), { timeout: 5_000 }, buttonScale)

  const beforePan = await page.$eval(".kb-mermaid-viewer", (viewer) => ({
    x: Number(viewer.dataset.kbX),
    y: Number(viewer.dataset.kbY),
  }))
  await page.mouse.down()
  await page.mouse.move(
    stageBox.x + stageBox.width / 2 + 48,
    stageBox.y + stageBox.height / 2 + 32,
    { steps: 4 },
  )
  await page.mouse.up()
  await page.waitForFunction((before) => {
    const viewer = document.querySelector(".kb-mermaid-viewer")
    return Number(viewer?.dataset.kbX) !== before.x
      || Number(viewer?.dataset.kbY) !== before.y
  }, { timeout: 5_000 }, beforePan)

  await page.click('[data-kb-mermaid-action="reset"]')
  const reset = await page.$eval(".kb-mermaid-viewer", (viewer) => ({
    mode: viewer.dataset.kbMode,
    scale: Number(viewer.dataset.kbScale),
  }))
  assert.deepEqual(reset, { mode: "actual", scale: 1 })

  await page.click('[data-kb-mermaid-action="fit"]')
  await page.waitForFunction(() => (
    document.querySelector(".kb-mermaid-viewer")?.dataset.kbMode === "fit"
  ))
  const beforePinch = await page.$eval(
    ".kb-mermaid-viewer",
    (viewer) => Number(viewer.dataset.kbScale),
  )
  const touch = await page.createCDPSession()
  await touch.send("Emulation.setTouchEmulationEnabled", {
    enabled: true,
    maxTouchPoints: 2,
  })
  const centerX = stageBox.x + stageBox.width / 2
  const centerY = stageBox.y + stageBox.height / 2
  await touch.send("Input.dispatchTouchEvent", {
    type: "touchStart",
    touchPoints: [
      { id: 1, x: centerX - 36, y: centerY, radiusX: 4, radiusY: 4, force: 1 },
      { id: 2, x: centerX + 36, y: centerY, radiusX: 4, radiusY: 4, force: 1 },
    ],
  })
  await touch.send("Input.dispatchTouchEvent", {
    type: "touchMove",
    touchPoints: [
      { id: 1, x: centerX - 82, y: centerY, radiusX: 4, radiusY: 4, force: 1 },
      { id: 2, x: centerX + 82, y: centerY, radiusX: 4, radiusY: 4, force: 1 },
    ],
  })
  await touch.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] })
  await page.waitForFunction((before) => (
    Number(document.querySelector(".kb-mermaid-viewer")?.dataset.kbScale) > before
  ), { timeout: 5_000 }, beforePinch)
  await touch.detach()

  await page.keyboard.press("Escape")
  await page.waitForFunction(() => !document.querySelector("dialog.kb-mermaid-viewer")?.open)
  const closed = await page.evaluate((selector) => ({
    bodyLocked: document.body.classList.contains("kb-mermaid-viewer-open"),
    focusRestored: document.activeElement === document.querySelector(selector),
  }), triggerSelector)
  assert.deepEqual(closed, { bodyLocked: false, focusRestored: true })

  await page.click(triggerSelector)
  await page.waitForFunction(() => document.querySelector("dialog.kb-mermaid-viewer")?.open)
  await page.click('[data-kb-mermaid-action="close"]')
  await page.waitForFunction(() => !document.querySelector("dialog.kb-mermaid-viewer")?.open)
}

async function runCase(browser, origin, basePath, servedResponses) {
  const page = await browser.newPage()
  const requests = []
  const responses = new Map()
  const blockedExternal = []
  const failedRequests = []
  const failedResponses = []
  const browserMessages = []
  const pageErrors = []
  const workerUrls = []

  try {
    await page.setCacheEnabled(false)
    await page.setRequestInterception(true)
    page.on("request", (request) => {
      const rawUrl = request.url()
      requests.push(rawUrl)
      const url = new URL(rawUrl)
      if (
        ["http:", "https:", "ws:", "wss:"].includes(url.protocol) &&
        url.hostname !== "127.0.0.1"
      ) {
        blockedExternal.push(rawUrl)
        request.abort("blockedbyclient").catch(() => undefined)
      } else {
        request.continue().catch(() => undefined)
      }
    })
    page.on("response", (response) => {
      responses.set(response.url(), response.status())
      if (response.status() >= 400) {
        failedResponses.push(`${response.status()} ${response.url()}`)
      }
    })
    page.on("requestfailed", (request) => {
      failedRequests.push(
        `${request.url()}: ${request.failure()?.errorText ?? "unknown failure"}`,
      )
    })
    page.on("console", (message) => {
      if (["error", "warn", "warning"].includes(message.type())) {
        browserMessages.push({ type: message.type(), text: message.text() })
      }
    })
    page.on("pageerror", (error) => pageErrors.push(error.message))
    page.on("workercreated", (worker) => workerUrls.push(worker.url()))

    const response = await page.goto(
      `${origin}${basePath}domain/10_article.html`,
      { waitUntil: "domcontentloaded", timeout: 60_000 },
    )
    assert.equal(response?.status(), 200, `${basePath} fixture page did not return 200`)
    await page.waitForFunction(
      () =>
        document.querySelectorAll("mjx-container").length > 0 &&
        document.querySelectorAll(".mermaid").length === 2 &&
        document.querySelector(".mermaid[data-kb-mermaid-error='true']") &&
        document.querySelector(".mermaid:not([data-kb-mermaid-error]) svg"),
      { timeout: 30_000 },
    )
    await waitForRendererActivity(requests, responses, servedResponses)

    const metrics = await page.evaluate(() => {
      const blocks = [...document.querySelectorAll(".mermaid")]
      const malformed = blocks.find(
        (block) => block.dataset.kbMermaidError === "true",
      )
      const secure = blocks.find(
        (block) =>
          block.dataset.kbMermaidError !== "true" && block.querySelector("svg"),
      )
      const hasEventHandler = secure
        ? [...secure.querySelectorAll("*")].some((node) =>
            [...node.attributes].some((attribute) =>
              attribute.name.toLowerCase().startsWith("on"),
            ),
          )
        : false
      return {
        math: document.querySelectorAll("mjx-container").length,
        mathErrors: document.querySelectorAll("mjx-merror").length,
        malformedMarked: Boolean(malformed),
        malformedSource: malformed?.textContent?.trim() ?? "",
        orphanArtifacts: document.querySelectorAll(
          'body > div[id^="dkb-mermaid-"]',
        ).length,
        orphanErrorGraphics: document.querySelectorAll(
          'body > div[id^="dkb-mermaid-"] svg, ' +
          '.mermaid[data-kb-mermaid-error="true"] svg .error-icon',
        ).length,
        securitySvg: secure?.querySelectorAll("svg").length ?? 0,
        dangerousDom:
          (secure?.querySelectorAll(
            'script, img, [href^="javascript:"], [src^="javascript:"]',
          ).length ?? 0) + Number(hasEventHandler),
        securityProbeExecuted: Boolean(window.__kbSecurityProbe),
      }
    })

    assert.ok(metrics.math > 0, `${basePath} rendered no MathJax containers`)
    assert.equal(metrics.mathErrors, 0, `${basePath} rendered an mjx-merror`)
    assert.equal(metrics.malformedMarked, true, `${basePath} missed Mermaid error marker`)
    assert.match(metrics.malformedSource, /A -->$/, `${basePath} lost malformed source`)
    assert.equal(metrics.orphanArtifacts, 0, `${basePath} leaked Mermaid render artifacts`)
    assert.equal(metrics.orphanErrorGraphics, 0, `${basePath} leaked Mermaid error graphics`)
    assert.ok(metrics.securitySvg > 0, `${basePath} security diagram did not render`)
    assert.equal(metrics.dangerousDom, 0, `${basePath} Mermaid produced dangerous DOM`)
    assert.equal(
      metrics.securityProbeExecuted,
      false,
      `${basePath} Mermaid security probe executed`,
    )

    await assertMermaidViewerLifecycle(page)

    assert.deepEqual(blockedExternal, [], `${basePath} attempted external requests`)
    const securityProbeRequests = requests.filter((rawUrl) => {
      const requested = new URL(rawUrl)
      return requested.protocol === "javascript:"
        || requested.pathname.endsWith("/domain/x")
    })
    assert.deepEqual(
      securityProbeRequests,
      [],
      `${basePath} made security probe requests`,
    )
    assert.deepEqual(failedRequests, [], `${basePath} had failed requests`)
    assert.deepEqual(failedResponses, [], `${basePath} had HTTP error responses`)
    assert.equal(
      pageErrors.filter(isExpectedMermaidPageError).length,
      1,
      `${basePath} did not emit exactly one expected Mermaid parse pageerror`,
    )
    assert.deepEqual(
      pageErrors.filter((message) => !isExpectedMermaidPageError(message)),
      [],
      `${basePath} emitted unexpected page errors`,
    )
    const unexpectedMessages = browserMessages.filter(
      (entry) => !isExpectedMermaidError(entry),
    )
    assert.equal(
      browserMessages.filter(isExpectedMermaidError).length,
      1,
      `${basePath} did not emit exactly one expected Mermaid parse error`,
    )
    assert.deepEqual(
      unexpectedMessages,
      [],
      `${basePath} emitted unexpected console warnings/errors`,
    )

    for (const rawUrl of requests) {
      const url = new URL(rawUrl)
      assert.ok(
        ["data:", "blob:"].includes(url.protocol) ||
          (["http:", "ws:"].includes(url.protocol) &&
            url.hostname === "127.0.0.1"),
        `${basePath} made a non-loopback request: ${rawUrl}`,
      )
    }
    for (const rawUrl of workerUrls) {
      const url = new URL(rawUrl)
      assert.ok(
        url.protocol === "blob:" ||
          (url.protocol === "http:" && url.hostname === "127.0.0.1"),
        `${basePath} created a non-local worker: ${rawUrl}`,
      )
    }

    const rendererRequests = [...new Set(requests.filter(isRendererUrl))]
    assert.ok(rendererRequests.length > 0, `${basePath} made no renderer requests`)
    for (const rawUrl of rendererRequests) {
      assert.equal(
        recordedStatus(rawUrl, responses, servedResponses),
        200,
        `${basePath} renderer request did not return 200: ${rawUrl}`,
      )
    }
    const rendererPaths = rendererRequests.map((rawUrl) => new URL(rawUrl).pathname)
    for (const required of requiredRendererSuffixes) {
      assert.ok(
        rendererPaths.some((pathname) => pathname.endsWith(required)),
        `${basePath} did not request ${required}`,
      )
    }
    assert.ok(
      rendererPaths.some((pathname) =>
        pathname.includes("/assets/vendor/mathjax-newcm/chtml/woff2/"),
      ),
      `${basePath} did not request a local NewCM woff2 font`,
    )
    assert.ok(
      rendererPaths.some((pathname) =>
        pathname.includes("/assets/vendor/mathjax/sre/mathmaps/"),
      ),
      `${basePath} did not request local SRE mathmaps`,
    )

    return {
      basePath,
      math: metrics.math,
      rendererRequests: rendererRequests.length,
      workers: workerUrls.length,
    }
  } catch (error) {
    const state = await page.evaluate(() => ({
      math: document.querySelectorAll("mjx-container").length,
      mermaid: [...document.querySelectorAll(".mermaid")].map((block) => ({
        error: block.dataset.kbMermaidError ?? null,
        source: block.textContent?.trim() ?? "",
        svg: block.querySelectorAll("svg").length,
      })),
    })).catch((diagnosticError) => ({ diagnosticError: diagnosticError.message }))
    error.message += `\n${JSON.stringify({
      basePath,
      state,
      blockedExternal,
      failedRequests,
      failedResponses,
      browserMessages,
      pageErrors,
      workerUrls,
    })}`
    throw error
  } finally {
    await page.close().catch(() => undefined)
  }
}

async function main() {
  const siteRoot = path.resolve(process.argv[2] ?? "")
  assert.ok(process.argv[2], "Usage: node renderer_contract.mjs <fixture-site>")
  assert.ok((await stat(siteRoot)).isDirectory(), `${siteRoot} is not a directory`)

  const puppeteer = await loadPuppeteer()
  const executablePath = await findBrowserExecutable()
  const servedResponses = new Map()
  const server = await startFixtureServer(siteRoot, servedResponses)
  const address = server.address()
  assert.ok(address && typeof address !== "string", "Fixture server has no TCP address")
  const origin = `http://127.0.0.1:${address.port}`
  let browser
  try {
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
    const cases = []
    for (const basePath of ["/", "/llm-knowledge/"]) {
      cases.push(await runCase(browser, origin, basePath, servedResponses))
    }
    console.log(JSON.stringify({ cases }))
  } finally {
    await browser?.close().catch(() => undefined)
    await closeServer(server)
  }
}

main().catch((error) => {
  console.error(error?.stack ?? String(error))
  process.exitCode = 1
})
