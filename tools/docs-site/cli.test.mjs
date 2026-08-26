import assert from "node:assert/strict"
import { mkdtemp, mkdir, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"
import test from "node:test"

import { main, preflight } from "./cli.mjs"
import { resolveProjectPaths } from "./contracts.mjs"

async function temporaryRepo() {
  return mkdtemp(path.join(tmpdir(), "docs-cli-"))
}

function buildDependencies(repoRoot, calls, exitCode = 0) {
  return {
    repoRoot,
    preflight: async () => calls.push("preflight"),
    ensureRuntime: async () => calls.push("ensureRuntime"),
    syncRuntimeConfig: async () => calls.push("syncRuntimeConfig"),
    repairRuntime: async () => calls.push("repairRuntime"),
    runQuartz: async (options) => {
      const { cwd, args, env } = options
      calls.push({ runQuartz: { cwd, args, bindHost: env?.DOCS_BIND_HOST } })
      return exitCode
    },
  }
}

test("build validates, prepares, and invokes Quartz against wiki", async () => {
  const repoRoot = await temporaryRepo()
  const calls = []

  const code = await main(["build"], buildDependencies(repoRoot, calls))

  assert.equal(code, 0)
  assert.deepEqual(calls.slice(0, 3), ["preflight", "ensureRuntime", "syncRuntimeConfig"])
  assert.deepEqual(calls[3].runQuartz, {
    cwd: path.join(repoRoot, ".cache", "llm-knowledge-docs", "quartz"),
    args: [
      "quartz/bootstrap-cli.mjs",
      "build",
      "--directory",
      path.join(repoRoot, "wiki"),
      "--output",
      path.join(repoRoot, ".cache", "llm-knowledge-docs", "output"),
    ],
    bindHost: "0.0.0.0",
  })
})

test("build returns the Quartz failure code", async () => {
  const repoRoot = await temporaryRepo()
  const code = await main(["build"], buildDependencies(repoRoot, [], 7))
  assert.equal(code, 7)
})

test("repair rebuilds only the runtime and does not invoke Quartz", async () => {
  const repoRoot = await temporaryRepo()
  const calls = []

  const code = await main(["repair"], buildDependencies(repoRoot, calls))

  assert.equal(code, 0)
  assert.deepEqual(calls, ["preflight", "repairRuntime"])
})

test("serve checks paired ports, waits for health, then opens the browser", async () => {
  const repoRoot = await temporaryRepo()
  const calls = []
  let resolveExit
  const exitCode = new Promise((resolve) => {
    resolveExit = resolve
  })
  const dependencies = {
    ...buildDependencies(repoRoot, calls),
    assertPortAvailable: async (port, opts) =>
      calls.push(`port:${port}@${opts?.host ?? "unset"}`),
    startQuartz: (options) => {
      const { cwd, args, env } = options
      calls.push({ startQuartz: { cwd, args, bindHost: env?.DOCS_BIND_HOST } })
      return { pid: 44, exitCode, terminate: () => undefined }
    },
    waitForHttp: async (url, options) => calls.push({
      health: url,
      timeoutMs: options.timeoutMs,
    }),
    openBrowser: async (url) => {
      calls.push(`browser:${url}`)
      resolveExit(0)
      return true
    },
    installSignalForwarding: () => {
      calls.push("signals:on")
      return () => calls.push("signals:off")
    },
  }

  const code = await main(["serve", "--port", "8090"], dependencies)

  assert.equal(code, 0)
  assert.deepEqual(calls.slice(0, 3), ["preflight", "ensureRuntime", "syncRuntimeConfig"])
  assert.deepEqual(calls.slice(3), [
    "port:8090@0.0.0.0",
    "port:8091@0.0.0.0",
    {
      startQuartz: {
        cwd: path.join(repoRoot, ".cache", "llm-knowledge-docs", "quartz"),
        args: [
          "quartz/bootstrap-cli.mjs",
          "build",
          "--serve",
          "--directory",
          path.join(repoRoot, "wiki"),
          "--output",
          path.join(repoRoot, ".cache", "llm-knowledge-docs", "output"),
          "--port",
          "8090",
          "--ws-port",
          "8091",
        ],
        bindHost: "0.0.0.0",
      },
    },
    "signals:on",
    {
      health: "http://127.0.0.1:8090/",
      timeoutMs: 180_000,
    },
    "browser:http://127.0.0.1:8090/",
    "signals:off",
  ])
})

test("serve propagates an early Quartz failure without opening a browser", async () => {
  const repoRoot = await temporaryRepo()
  let browserOpened = false
  const dependencies = {
    ...buildDependencies(repoRoot, []),
    assertPortAvailable: async () => undefined,
    startQuartz: () => ({
      pid: 55,
      exitCode: Promise.resolve(9),
      terminate: () => undefined,
    }),
    waitForHttp: () => new Promise(() => {}),
    openBrowser: async () => {
      browserOpened = true
      return true
    },
    installSignalForwarding: () => () => undefined,
  }

  assert.equal(await main(["serve", "--no-open"], dependencies), 9)
  assert.equal(browserOpened, false)
})

test("serve terminates Quartz when readiness times out", async () => {
  const repoRoot = await temporaryRepo()
  const terminated = []
  let resolveExit
  const exitCode = new Promise((resolve) => {
    resolveExit = resolve
  })
  const dependencies = {
    ...buildDependencies(repoRoot, []),
    assertPortAvailable: async () => undefined,
    startQuartz: () => ({
      pid: 66,
      exitCode,
      terminate: (signal) => {
        terminated.push(signal)
        resolveExit(1)
      },
    }),
    waitForHttp: async () => {
      throw new Error("health timeout")
    },
    openBrowser: async () => true,
    installSignalForwarding: () => () => undefined,
  }

  await assert.rejects(main(["serve", "--no-open"], dependencies), /health timeout/)
  assert.deepEqual(terminated, ["SIGTERM"])
})

test("preflight rejects an injected Node version below the pinned floor", async () => {
  const repoRoot = await temporaryRepo()
  const paths = resolveProjectPaths(repoRoot)
  await mkdir(paths.toolDir, { recursive: true })
  await mkdir(paths.wikiDir, { recursive: true })
  await writeFile(path.join(paths.wikiDir, "index.md"), "# Home\n")
  await writeFile(
    paths.manifestSource,
    JSON.stringify({ minimumTools: { node: "22.0.0", npm: "10.9.2", git: "2.0.0" } }),
  )

  await assert.rejects(
    preflight(paths, {
      nodeVersion: "21.9.0",
      runCaptured: async (command) => ({
        code: 0,
        stdout: command === "git" ? "git version 2.50.0\n" : "11.0.0\n",
        stderr: "",
      }),
    }),
    /requires Node\.js >= 22\.0\.0; detected 21\.9\.0/,
  )
})
