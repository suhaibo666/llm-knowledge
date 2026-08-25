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
      calls.push({ runQuartz: options })
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
