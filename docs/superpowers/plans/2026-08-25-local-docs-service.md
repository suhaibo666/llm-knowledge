# Local Markdown Documentation Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-command, loopback-only Quartz 5 documentation service that renders `wiki/` in a browser without changing any Markdown source.

**Architecture:** A small Node.js launcher owns a repository-specific disposable Quartz runtime under `.cache/llm-knowledge-docs/`. It provisions exact pinned core and plugin revisions once, applies strict version-scoped local-only patches, vendors Mermaid, then invokes Quartz directly against `wiki/`; browser rendering, navigation, indexing, and watching remain Quartz responsibilities.

**Tech Stack:** Node.js ES modules and `node:test`, npm, Git, Quartz 5.0.0, Quartz community plugins, Mermaid 11.4.0, Puppeteer Core, PowerShell/system socket tools for listener verification

**Spec:** `docs/superpowers/specs/2026-08-25-local-docs-service-design.md`

## Global Constraints

- The only content root is `wiki/`; implementation and tests must not modify, copy, or normalize any `wiki/**/*.md` file.
- Pin Quartz to tag `v5.0.0` and commit `ab346fa66a895e12d63a308e70ce330ba795822a`.
- Require Node.js 22 or newer, npm 10.9.2 or newer, and Git before provisioning.
- Pin the Obsidian Flavored Markdown plugin to `eda74e79eac746445b7d6941f90133bf97483cd7` and the LaTeX plugin to `2544fd13fa132493647480d8ce75f4432a6f1750`; every other enabled plugin uses its tracked lockfile commit.
- Pin Mermaid to `11.4.0`, copy its complete browser `dist` tree into the Quartz runtime, and load it only from `/static/vendor/mermaid/mermaid.esm.min.mjs`.
- Bind both HTTP and hot-reload WebSocket listeners explicitly to `127.0.0.1`; there is no public or LAN host option.
- Use MathJax SVG rendering and system fonts; generated pages must not request CDN or other non-loopback resources.
- Keep Quartz runtime, output, staging directories, and Quartz-installed dependencies below ignored `.cache/llm-knowledge-docs/`; the browser smoke test may reuse the already-ignored `tools/html2md/node_modules/puppeteer-core`, and no dependency may enter `wiki/` or a global cache.
- Ordinary serve/build runs are offline after the first successful provision; dependency download is confined to provision and explicit repair.
- Destructive operations are restricted to resolved descendants of `.cache/llm-knowledge-docs/`, and repair must stage a complete replacement before switching runtimes.
- Root commands are `npm run docs`, `npm run docs:build`, `npm run docs:test`, and `npm run docs:repair`; `npm run docs -- --port 8081` overrides the default HTTP port 8080 and reserves the next port for WebSocket.
- Browser-open failure is non-fatal; build failure, version drift, patch drift, missing vendor assets, and port collision are fatal with actionable messages.

## File Map

- `package.json`: dependency-free root command surface.
- `tools/docs-site/contracts.mjs`: pure version, argument, port, path, and Quartz command contracts.
- `tools/docs-site/contracts.test.mjs`: unit tests for the pure contracts.
- `tools/docs-site/runtime.mjs`: pinned provisioning, strict patching, integrity validation, and atomic repair.
- `tools/docs-site/runtime.test.mjs`: filesystem-level tests using temporary fake runtimes.
- `tools/docs-site/config.test.mjs`: tracked Quartz configuration and plugin-lock invariant tests.
- `tools/docs-site/processes.mjs`: process execution, port probing, HTTP readiness, browser opening, and signal forwarding.
- `tools/docs-site/processes.test.mjs`: injected-process and local-socket lifecycle tests.
- `tools/docs-site/cli.mjs`: thin `serve`, `build`, and `repair` orchestration entry point.
- `tools/docs-site/cli.test.mjs`: injected orchestration tests without provisioning or browser side effects.
- `tools/docs-site/smoke.mjs`: end-to-end browser, network, rendering, navigation, search, and listener assertions.
- `tools/docs-site/quartz.config.yaml`: trimmed docs-style Quartz configuration.
- `tools/docs-site/quartz.lock.json`: lock entries only for enabled community plugins, copied from Quartz `v5.0.0`.
- `tools/docs-site/runtime-manifest.json`: core/plugin/Mermaid pins and the exact expected runtime mutations.
- `tools/docs-site/patches/quartz-v5-core-local-only.json`: exact-once replacements for HTTP, WebSocket, and `Head.tsx` preconnect behavior.
- `tools/docs-site/patches/quartz-v5-ofm-local-mermaid.json`: exact-once replacement for the pinned OFM Mermaid import.
- `tools/docs-site/patches/quartz-v5-crawl-links-obsidian-paths.json`: exact-once support for unique Obsidian path-suffix links and relative media paths.
- `tools/docs-site/patches/quartz-v5-breadcrumbs-no-frontmatter.json`: exact-once breadcrumb support for pages without YAML frontmatter.
- `README.md`: user-facing local site commands and first-run behavior.
- `tools/README.md`: launcher ownership, cache boundary, and upgrade workflow.

---

### Task 1: Stable CLI and Path Contracts

**Files:**
- Create: `package.json`
- Create: `tools/docs-site/contracts.mjs`
- Create: `tools/docs-site/contracts.test.mjs`

**Interfaces:**
- Consumes: `process.argv`, `process.versions.node`, repository root path.
- Produces: `parseCliArgs(argv): { command: "serve" | "build" | "repair", port: number, wsPort: number, openBrowser: boolean }`; `assertMinimumVersion(name, actual, minimum): void`; `resolveProjectPaths(repoRoot): ProjectPaths`; `assertPathInside(parent, candidate): void`; `buildQuartzArgs(mode, paths, port): string[]`.

- [ ] **Step 1: Write failing contract tests**

```js
import test from "node:test"
import assert from "node:assert/strict"
import path from "node:path"
import {
  assertMinimumVersion,
  assertPathInside,
  buildQuartzArgs,
  parseCliArgs,
  resolveProjectPaths,
} from "./contracts.mjs"

test("serve defaults to loopback ports and browser open", () => {
  assert.deepEqual(parseCliArgs(["serve"]), {
    command: "serve", port: 8080, wsPort: 8081, openBrowser: true,
  })
})

test("port override reserves the following WebSocket port", () => {
  assert.deepEqual(parseCliArgs(["serve", "--port", "9090", "--no-open"]), {
    command: "serve", port: 9090, wsPort: 9091, openBrowser: false,
  })
})

test("invalid commands and port boundaries fail", () => {
  assert.throws(() => parseCliArgs(["publish"]), /serve, build, or repair/)
  assert.throws(() => parseCliArgs(["serve", "--port", "65535"]), /1 through 65534/)
})

test("semantic version floors compare numeric components", () => {
  assert.doesNotThrow(() => assertMinimumVersion("Node.js", "22.0.0", "22.0.0"))
  assert.doesNotThrow(() => assertMinimumVersion("npm", "11.0.0", "10.9.2"))
  assert.throws(() => assertMinimumVersion("npm", "10.9.1", "10.9.2"), /requires npm >= 10.9.2/)
})

test("cache paths remain under the dedicated cache root", () => {
  const root = path.resolve("repo")
  const paths = resolveProjectPaths(root)
  assert.equal(paths.wikiDir, path.join(root, "wiki"))
  assert.equal(paths.runtimeDir, path.join(root, ".cache", "llm-knowledge-docs", "quartz"))
  assert.doesNotThrow(() => assertPathInside(paths.cacheRoot, paths.runtimeDir))
  assert.throws(() => assertPathInside(paths.cacheRoot, root), /outside dedicated cache root/)
})

test("Quartz arguments always use the wiki and cache output paths", () => {
  const paths = resolveProjectPaths(path.resolve("repo"))
  assert.deepEqual(buildQuartzArgs("build", paths, 8080), [
    "quartz/bootstrap-cli.mjs", "build", "--directory", paths.wikiDir,
    "--output", paths.outputDir,
  ])
  assert.deepEqual(buildQuartzArgs("serve", paths, 8080), [
    "quartz/bootstrap-cli.mjs", "build", "--serve", "--directory", paths.wikiDir,
    "--output", paths.outputDir, "--port", "8080", "--ws-port", "8081",
  ])
})
```

- [ ] **Step 2: Run tests and confirm the module is missing**

Run: `node --test tools/docs-site/contracts.test.mjs`

Expected: FAIL with `ERR_MODULE_NOT_FOUND` for `contracts.mjs`.

- [ ] **Step 3: Implement pure contracts and root scripts**

Use `path.resolve` for every root, compare path-relative results instead of prefix strings, reject duplicate/unknown flags, and keep build/repair browser-free. The root package is exactly:

```json
{
  "name": "llm-knowledge",
  "private": true,
  "type": "module",
  "scripts": {
    "docs": "node tools/docs-site/cli.mjs serve",
    "docs:build": "node tools/docs-site/cli.mjs build",
    "docs:test:unit": "node --test tools/docs-site/*.test.mjs",
    "docs:test": "npm run docs:test:unit && node tools/docs-site/smoke.mjs",
    "docs:repair": "node tools/docs-site/cli.mjs repair"
  }
}
```

The path object contains `repoRoot`, `wikiDir`, `cacheRoot`, `runtimeDir`, `outputDir`, `stagingRoot`, `configSource`, `lockSource`, `manifestSource`, and `markerFile`. `buildQuartzArgs` invokes the checked-out bootstrap file with absolute content/output arguments and never emits a host argument.

- [ ] **Step 4: Run the contract tests**

Run: `node --test tools/docs-site/contracts.test.mjs`

Expected: all tests PASS.

- [ ] **Step 5: Commit the contract**

```bash
git add package.json tools/docs-site/contracts.mjs tools/docs-site/contracts.test.mjs
git commit -m "feat: add local docs command contracts"
```

---

### Task 2: Pinned, Strict, Recoverable Quartz Runtime

**Files:**
- Create: `tools/docs-site/runtime.mjs`
- Create: `tools/docs-site/runtime.test.mjs`
- Create: `tools/docs-site/runtime-manifest.json`
- Create: `tools/docs-site/quartz.lock.json`
- Create: `tools/docs-site/patches/quartz-v5-core-local-only.json`
- Create: `tools/docs-site/patches/quartz-v5-ofm-local-mermaid.json`

**Interfaces:**
- Consumes: `ProjectPaths` from Task 1, tracked manifest/lock/patch files, and an injectable `{ run, now, randomUUID }` dependency object.
- Produces: `applyPatchSpec(root, spec): Promise<"applied" | "already-applied">`; `inspectRuntime(paths, deps): Promise<{ kind: "missing" | "ready" | "invalid", reason?: string }>`; `ensureRuntime(paths, deps): Promise<void>`; `repairRuntime(paths, deps): Promise<void>`; `syncRuntimeConfig(paths): Promise<void>`.

- [ ] **Step 1: Write failing tests for strict patching and state validation**

```js
import test from "node:test"
import assert from "node:assert/strict"
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"
import { resolveProjectPaths } from "./contracts.mjs"
import { applyPatchSpec, inspectRuntime } from "./runtime.mjs"

test("exact-once patches apply once and recognize the patched state", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "docs-patch-"))
  await writeFile(path.join(root, "target.js"), "server.listen(argv.port)\n")
  const spec = { replacements: [{
    file: "target.js",
    before: "server.listen(argv.port)",
    after: 'server.listen(argv.port, "127.0.0.1")',
  }] }
  assert.equal(await applyPatchSpec(root, spec), "applied")
  assert.equal(await applyPatchSpec(root, spec), "already-applied")
  assert.match(await readFile(path.join(root, "target.js"), "utf8"), /127\.0\.0\.1/)
})

test("patch context drift and duplicate context are rejected", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "docs-patch-"))
  const spec = { replacements: [{ file: "target.js", before: "old", after: "new" }] }
  await writeFile(path.join(root, "target.js"), "different")
  await assert.rejects(applyPatchSpec(root, spec), /expected exactly one unpatched or patched context/)
  await writeFile(path.join(root, "target.js"), "old old")
  await assert.rejects(applyPatchSpec(root, spec), /expected exactly one/)
})

test("a marker alone cannot make an incomplete runtime ready", async () => {
  const repoRoot = await mkdtemp(path.join(tmpdir(), "docs-runtime-"))
  const paths = resolveProjectPaths(repoRoot)
  await mkdir(paths.runtimeDir, { recursive: true })
  await writeFile(paths.markerFile, JSON.stringify({ schemaVersion: 1 }))
  const state = await inspectRuntime(paths, { run: async () => ({ code: 0, stdout: "", stderr: "" }) })
  assert.equal(state.kind, "invalid")
  assert.match(state.reason, /manifest|Quartz|vendor|marker/)
})
```

Add cases proving that a wrong core commit, a wrong plugin commit in the runtime lock, a missing Mermaid entry/chunk, an altered tracked input hash, and any tracked runtime diff beyond the approved version-scoped mutations all produce `kind: "invalid"`.

- [ ] **Step 2: Run runtime tests and confirm missing exports**

Run: `node --test tools/docs-site/runtime.test.mjs`

Expected: FAIL with `ERR_MODULE_NOT_FOUND` or missing named exports.

- [ ] **Step 3: Add pins and exact patch specifications**

`runtime-manifest.json` records schema version 1, the Quartz repository/tag/commit, minimum tools, Mermaid version, enabled plugin commits, and allowed tracked modifications. Each manifest plugin entry includes its lock commit and installed package path so inspection validates both the copied lock and installed plugin metadata. The core patch contains these exact replacements:

```json
{
  "id": "quartz-v5-core-local-only",
  "replacements": [
    {
      "file": "quartz/cli/handlers.js",
      "before": "server.listen(argv.port)",
      "after": "server.listen(argv.port, \"127.0.0.1\")"
    },
    {
      "file": "quartz/cli/handlers.js",
      "before": "new WebSocketServer({ port: argv.wsPort })",
      "after": "new WebSocketServer({ port: argv.wsPort, host: \"127.0.0.1\" })"
    },
    {
      "file": "quartz/components/Head.tsx",
      "before": "        <link rel=\"preconnect\" href=\"https://cdnjs.cloudflare.com\" crossOrigin=\"anonymous\" />\n",
      "after": ""
    }
  ]
}
```

The OFM patch targets `.quartz/plugins/obsidian-flavored-markdown/dist/index.js` and replaces exactly one `https://cdnjs.cloudflare.com/ajax/libs/mermaid/11.4.0/mermaid.esm.min.mjs` with `/static/vendor/mermaid/mermaid.esm.min.mjs`. Copy from Quartz `v5.0.0` only the lock entries used by `quartz.config.yaml`, preserving each exact commit.

- [ ] **Step 4: Implement strict state inspection and provisioning**

`applyPatchSpec` first validates each relative file stays below its root. For every replacement, exactly one `before` and zero `after` means apply; zero `before` and exactly one `after` means already applied; every other count fails. Mixed applied/unapplied replacements in one spec are allowed so interrupted staging can be diagnosed, but a completed runtime marker is written only after all replacements validate as applied.

Provision in a unique descendant of `stagingRoot`:

```text
git clone --depth 1 --branch v5.0.0 https://github.com/jackyzha0/quartz.git <stage>
git -C <stage> rev-parse HEAD
npm ci
copy tracked quartz.config.yaml and quartz.lock.json
node quartz/bootstrap-cli.mjs plugin install
npm install --no-save --package-lock=false mermaid@11.4.0
copy node_modules/mermaid/dist recursively to quartz/static/vendor/mermaid
apply every core and plugin patch specification listed in the runtime manifest
validate commit, locks, patches, vendor entry/chunks, and allowed git diff
write .llm-knowledge-docs-runtime.json with SHA-256 hashes of all tracked runtime inputs
rename the complete stage to runtimeDir
```

Run npm commands with `npm_config_cache` set to `.cache/llm-knowledge-docs/npm-cache`, keeping download state repository-local. `ensureRuntime` provisions only when state is `missing`; it rejects `invalid` with the exact `npm run docs:repair` recovery command. `repairRuntime` builds a new stage, validates the resolved dedicated runtime and quarantine paths against `cacheRoot`, renames an existing ready or invalid dedicated runtime to a unique quarantine path, installs the new runtime, and then removes only that validated quarantine path. A failed new provision leaves the old runtime untouched.

- [ ] **Step 5: Run runtime tests**

Run: `node --test tools/docs-site/runtime.test.mjs`

Expected: all tests PASS, including repeated patch and safety-boundary cases.

- [ ] **Step 6: Commit runtime management**

```bash
git add tools/docs-site/runtime.mjs tools/docs-site/runtime.test.mjs tools/docs-site/runtime-manifest.json tools/docs-site/quartz.lock.json tools/docs-site/patches
git commit -m "feat: provision pinned local Quartz runtime"
```

---

### Task 3: Offline Quartz Configuration and Full Wiki Build

**Files:**
- Create: `tools/docs-site/quartz.config.yaml`
- Create: `tools/docs-site/config.test.mjs`
- Create: `tools/docs-site/cli.mjs`
- Create: `tools/docs-site/cli.test.mjs`

**Interfaces:**
- Consumes: Task 1 arguments and Task 2 runtime operations.
- Produces: `main(argv, deps): Promise<number>` for `build` and `repair`; a configured Quartz build reading `wiki/` directly.

- [ ] **Step 1: Add failing configuration invariant tests**

Read the YAML as text and parse the runtime manifest/lock as JSON. Assert that it contains `pageTitle: LLM Knowledge Wiki`, `locale: zh-CN`, `fontOrigin: local`, `cdnCaching: false`, `markdownLinkResolution: shortest`, `renderEngine: mathjax`, and enabled explorer/search/TOC/breadcrumbs/darkmode/content-page/folder-page entries. Assert that remote analytics, Graph, Backlinks, Popover, Reader Mode, Recent Notes, Comments, RSS, sitemap, OG image, CNAME, Canvas, Bases, encrypted pages, remote fonts, footer links, and CDN URLs are absent or explicitly disabled.

Add an injected CLI test:

```js
function fakeDeps(calls) {
  return {
    preflight: async () => calls.push({ kind: "preflight" }),
    ensureRuntime: async () => calls.push({ kind: "ensureRuntime" }),
    syncRuntimeConfig: async () => calls.push({ kind: "syncRuntimeConfig" }),
    runQuartz: async (args) => {
      calls.push({ kind: "runQuartz", args })
      return 0
    },
    resolveRepoRoot: () => path.resolve("repo"),
  }
}

test("build ensures the runtime, syncs config, and invokes Quartz once", async () => {
  const calls = []
  const code = await main(["build"], fakeDeps(calls))
  assert.equal(code, 0)
  assert.deepEqual(calls.map((call) => call.kind), [
    "preflight", "ensureRuntime", "syncRuntimeConfig", "runQuartz",
  ])
  assert.equal(calls.at(-1).args.includes("--serve"), false)
})
```

- [ ] **Step 2: Run tests and confirm missing config/orchestration**

Run: `node --test tools/docs-site/runtime.test.mjs tools/docs-site/cli.test.mjs`

Expected: FAIL because the config and CLI orchestration do not exist yet.

- [ ] **Step 3: Add the trimmed local-only Quartz config**

Start from the `v5.0.0` default YAML and retain only these enabled transformer/emitter/component sources: `syntax-highlighting`, `obsidian-flavored-markdown`, `github-flavored-markdown`, `table-of-contents`, `crawl-links`, `description`, `latex`, `remove-draft`, `alias-redirects`, `content-index`, `favicon`, `content-page`, `folder-page`, `explorer`, `search`, `article-title`, `page-title`, `darkmode`, and `breadcrumbs`. Configure OFM with `mermaid: true`, LaTeX with `renderEngine: mathjax`, content-index with both feed options false, and the docs layout as left explorer/toolbar, body breadcrumbs/title, and right TOC. Use local/system typography names and no footer external links.

- [ ] **Step 4: Implement build and repair orchestration**

`main` validates Node/npm/Git and `wiki/index.md` before calling runtime code. It synchronizes configuration only after runtime integrity passes, runs `process.execPath` with `buildQuartzArgs`, uses `cwd: runtimeDir`, and returns the Quartz exit code. The module executes `process.exitCode = await main(process.argv.slice(2))` only when invoked as the entry script, so tests can import it without side effects.

- [ ] **Step 5: Run unit tests and a first real full build**

Run: `npm run docs:test:unit`

Expected: all unit tests PASS.

Run: `npm run docs:build`

Expected: first run provisions the pinned runtime, builds all `wiki/` Markdown into `.cache/llm-knowledge-docs/output`, emits no CDN URL in generated HTML/JS/CSS, and exits 0. Check with:

```powershell
rg -n "https?://" .cache/llm-knowledge-docs/output -g "*.html" -g "*.js" -g "*.css"
```

Expected: no remote resource reference; content text containing source URLs may be present only inside article bodies and must not occur in `src=`, `href=` for runtime assets, dynamic imports, or CSS `url()`.

- [ ] **Step 6: Commit the configured build**

```bash
git add tools/docs-site/quartz.config.yaml tools/docs-site/config.test.mjs tools/docs-site/cli.mjs tools/docs-site/cli.test.mjs
git commit -m "feat: build wiki with offline Quartz config"
```

---

### Task 4: Loopback Serve Lifecycle

**Files:**
- Create: `tools/docs-site/processes.mjs`
- Create: `tools/docs-site/processes.test.mjs`
- Modify: `tools/docs-site/cli.mjs`
- Modify: `tools/docs-site/cli.test.mjs`

**Interfaces:**
- Consumes: serve arguments and the ready runtime.
- Produces: `assertPortAvailable(port, host): Promise<void>`; `waitForHttp(url, options): Promise<void>`; `openBrowser(url, deps): Promise<boolean>`; `startQuartzService(args, deps): { exitCode: Promise<number>, terminate(signal): void }`.

- [ ] **Step 1: Write failing lifecycle tests**

Use real temporary loopback servers for port checks and injected children for signals:

```js
function listen(server, port, host) {
  return new Promise((resolve, reject) => {
    server.once("error", reject)
    server.listen(port, host, resolve)
  })
}

test("occupied loopback ports fail with override guidance", async () => {
  const server = net.createServer()
  await listen(server, 0, "127.0.0.1")
  const port = server.address().port
  await assert.rejects(assertPortAvailable(port, "127.0.0.1"), /--port/)
  server.close()
})

test("browser launch failure returns false without terminating service", async () => {
  assert.equal(await openBrowser("http://127.0.0.1:8080", {
    platform: "win32",
    spawnDetached: () => { throw new Error("no browser") },
  }), false)
})

test("serve checks both ports, waits for health, then opens browser", async () => {
  const calls = []
  const code = await main(["serve", "--port", "8090"], {
    ...fakeDeps(calls),
    assertPortAvailable: async (port) => calls.push({ kind: `port:${port}` }),
    startQuartz: async (args) => {
      calls.push({ kind: "startQuartz", args })
      return { exitCode: Promise.resolve(0), terminate: () => undefined }
    },
    waitForHttp: async () => undefined,
    openBrowser: async () => {
      calls.push({ kind: "openBrowser" })
      return true
    },
  })
  assert.equal(code, 0)
  assert.deepEqual(calls.slice(-4).map((call) => call.kind), [
    "port:8090", "port:8091", "startQuartz", "openBrowser",
  ])
})
```

Add tests for health timeout, child non-zero exit propagation, and one-time forwarding of `SIGINT`/`SIGTERM` to the Quartz child.

- [ ] **Step 2: Run lifecycle tests and confirm missing behavior**

Run: `node --test tools/docs-site/processes.test.mjs tools/docs-site/cli.test.mjs`

Expected: FAIL on missing lifecycle exports and serve calls.

- [ ] **Step 3: Implement service lifecycle**

Probe `127.0.0.1` for both `port` and `wsPort` before spawning. Start Quartz with inherited stdio, begin a bounded HTTP readiness loop against `http://127.0.0.1:<port>/`, and race readiness against early child exit. Open the browser only after an HTTP 2xx/3xx response. Use `LLM_KNOWLEDGE_BROWSER` when set; otherwise use `cmd.exe /d /s /c start "" <url>` on Windows, `open <url>` on macOS, and `xdg-open <url>` on Linux. Catch launch errors, print the URL, and leave Quartz running. Remove installed signal handlers after child exit.

- [ ] **Step 4: Run lifecycle and full unit tests**

Run: `npm run docs:test:unit`

Expected: all tests PASS and no test process remains listening.

- [ ] **Step 5: Manually start on a non-default port and stop cleanly**

Run: `npm run docs -- --port 8088 --no-open`

Expected: service reports `http://127.0.0.1:8088`, hot reload uses 8089, the homepage returns successfully, and one Ctrl+C ends both listeners without a background Node process.

- [ ] **Step 6: Commit lifecycle support**

```bash
git add tools/docs-site/processes.mjs tools/docs-site/processes.test.mjs tools/docs-site/cli.mjs tools/docs-site/cli.test.mjs
git commit -m "feat: serve docs on loopback with clean lifecycle"
```

---

### Task 5: Browser and Listener Acceptance Harness

**Files:**
- Create: `tools/docs-site/smoke.mjs`
- Create: `tools/docs-site/listeners.mjs`
- Create: `tools/docs-site/listeners.test.mjs`

**Interfaces:**
- Consumes: `tools/html2md` Puppeteer Core installation, local Edge/Chrome, the normal CLI service, and the system listener table.
- Produces: `findBrowserExecutable(env, platform): Promise<string>`; `assertListenerRecords(ports, records): void`; `assertLoopbackListeners(ports, deps): Promise<void>`; a zero/non-zero end-to-end smoke process.

- [ ] **Step 1: Write failing browser discovery and listener parser tests**

Test `LLM_KNOWLEDGE_BROWSER` precedence, Windows Edge/Chrome candidates, macOS/Linux candidates, missing-browser guidance, and listener records. A record for either port with `0.0.0.0`, `::`, or a non-loopback local address must fail; records for exactly `127.0.0.1` pass.

```js
test("listener verification rejects wildcard exposure", () => {
  assert.throws(() => assertListenerRecords([8080, 8081], [
    { port: 8080, address: "127.0.0.1" },
    { port: 8081, address: "0.0.0.0" },
  ]), /8081.*0\.0\.0\.0/)
})
```

- [ ] **Step 2: Run tests and confirm missing harness exports**

Run: `node --test tools/docs-site/listeners.test.mjs`

Expected: FAIL with missing module or exports.

- [ ] **Step 3: Implement cross-platform listener inspection**

On Windows, invoke PowerShell `Get-NetTCPConnection -State Listen` and serialize only `LocalAddress`, `LocalPort`, and `OwningProcess` as JSON. On Linux use `ss -ltnp`; on macOS use `lsof -nP -iTCP -sTCP:LISTEN`. Parse into `{ port, address, pid? }`, require both expected ports, and reject wildcard/non-loopback addresses. The spawned service PID must own or parent the listeners when ownership data is available.

- [ ] **Step 4: Implement the browser smoke flow**

If `tools/html2md/node_modules/puppeteer-core` is absent, run `npm ci --prefix tools/html2md` before launching. Start `node tools/docs-site/cli.mjs serve --port <freePort> --no-open`, wait for health, and launch Puppeteer with the detected browser. Capture `pageerror`, `requestfailed`, response failures, and every request URL.

Assertions cover:

1. `/` has page title `LLM Knowledge Wiki`, explorer, search trigger, dark-mode control, breadcrumbs/article region, and a right-side table of contents on a representative page.
2. The representative page `wiki/02_engineering/04_posttrain_frameworks/verl/02_verl_quickstart_guide.md` renders at its Quartz route and contains navigable bare, rooted-path, and heading-fragment links selected from its real source.
3. A real callout becomes a structured callout container.
4. A real Mermaid block becomes an SVG and no CDN request occurs.
5. Inline and block formulas become MathJax SVG DOM.
6. The determinism/reliability representative page loads at least one PNG and one SVG with non-zero natural dimensions.
7. A real Markdown table, escaped wikilink alias, fenced code block, and supported inline HTML fragment remain visible with their intended structure.
8. Opening search and entering a stable Chinese term returns at least one navigable result.
9. Every network request uses `http:` or `ws:` with hostname `127.0.0.1`; no browser errors or failed local assets remain.
10. `assertLoopbackListeners([port, port + 1])` succeeds while the service is running.

Always close the browser, send one interrupt to the child, wait for exit, and force-kill only after a bounded grace period. Preserve collected diagnostics on failure.

- [ ] **Step 5: Run the full browser smoke test**

Run: `node tools/docs-site/smoke.mjs`

Expected: PASS for UI, links, callout, Mermaid SVG, MathJax SVG, images, search, local-only network, zero browser errors, and both loopback listeners; the service exits afterward.

- [ ] **Step 6: Commit the acceptance harness**

```bash
git add tools/docs-site/smoke.mjs tools/docs-site/listeners.mjs tools/docs-site/listeners.test.mjs
git commit -m "test: verify local docs in browser"
```

---

### Task 6: User Documentation and Final Acceptance

**Files:**
- Modify: `README.md`
- Modify: `tools/README.md`

**Interfaces:**
- Consumes: the stable commands and tested behavior from Tasks 1–5.
- Produces: a concise operator guide and evidence that the existing knowledge base is unchanged.

- [ ] **Step 1: Add user-facing documentation**

In `README.md`, document Node/npm/Git requirements, `npm run docs`, default `http://127.0.0.1:8080`, `--port`, first-run network requirement, subsequent offline behavior, cache location, `npm run docs:build`, `npm run docs:test`, and explicit `npm run docs:repair`. State that only `wiki/` is displayed and Markdown is never rewritten.

In `tools/README.md`, describe `tools/docs-site` as the pinned runtime launcher, list the three local-only adaptations, explain that qmd remains the Agent/MCP search path, and require lock/patch/smoke updates for an explicit Quartz upgrade.

- [ ] **Step 2: Verify repository link integrity**

Run: `python tools/check_links.py --strict`

Expected: exit 0 with no broken or ambiguous wiki links.

- [ ] **Step 3: Verify unit tests, clean full build, and browser behavior**

Run: `npm run docs:test:unit`

Expected: all tests PASS.

Run: `npm run docs:build`

Expected: exit 0 and a complete output tree under `.cache/llm-knowledge-docs/output`.

Run: `npm run docs:test`

Expected: unit and browser smoke tests PASS, including system listener and zero-external-request checks.

- [ ] **Step 4: Prove offline reuse**

With the ready runtime present, run build with Git/npm proxy variables set to an unreachable loopback port and with package-registry access unavailable to the child process.

Run: `npm run docs:build`

Expected: build still exits 0 because normal runtime validation and Quartz build perform no clone, install, or remote import.

- [ ] **Step 5: Verify source immutability and repository cleanliness**

Run:

```powershell
git diff --check
git status --short
git diff --name-only -- wiki
git ls-files --others --exclude-standard .cache/llm-knowledge-docs
```

Expected: `git diff --check` exits 0; `git diff --name-only -- wiki` is empty; the dedicated cache produces no tracked/unignored files; status contains only the intended implementation and documentation changes since the task baseline.

- [ ] **Step 6: Commit documentation and acceptance results**

```bash
git add README.md tools/README.md
git commit -m "docs: document local knowledge site"
```

- [ ] **Step 7: Review the completed implementation against the spec**

Read the design and this plan again, inspect all commits, and confirm every global constraint has direct test or command evidence. Record any environment-specific limitation in the final handoff instead of weakening a test or silently expanding network exposure.
