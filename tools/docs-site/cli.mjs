import { spawn } from "node:child_process"
import { access, readFile } from "node:fs/promises"
import path from "node:path"
import { fileURLToPath } from "node:url"

import {
  assertMinimumVersion,
  buildQuartzArgs,
  parseCliArgs,
  resolveProjectPaths,
} from "./contracts.mjs"
import {
  buildNpmInvocation,
  ensureRuntime as ensurePinnedRuntime,
  repairRuntime as repairPinnedRuntime,
  syncRuntimeConfig as syncPinnedRuntimeConfig,
} from "./runtime.mjs"

const scriptFile = fileURLToPath(import.meta.url)
const defaultRepoRoot = path.resolve(path.dirname(scriptFile), "..", "..")

async function runCaptured(command, args, options = {}) {
  return new Promise((resolve) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env,
      shell: options.shell ?? false,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    })
    let stdout = ""
    let stderr = ""
    child.stdout.setEncoding("utf8")
    child.stderr.setEncoding("utf8")
    child.stdout.on("data", (chunk) => {
      stdout += chunk
    })
    child.stderr.on("data", (chunk) => {
      stderr += chunk
    })
    child.on("error", (error) => {
      resolve({ code: -1, stdout, stderr: `${stderr}${error.message}` })
    })
    child.on("close", (code) => resolve({ code: code ?? -1, stdout, stderr }))
  })
}

function versionFromOutput(name, output) {
  const match = output.match(/\d+(?:\.\d+){1,}/)
  if (!match) throw new Error(`Unable to determine ${name} version from: ${output.trim()}`)
  return match[0]
}

export async function preflight(paths, deps = {}) {
  const run = deps.runCaptured ?? runCaptured
  const manifest = JSON.parse(await readFile(paths.manifestSource, "utf8"))
  assertMinimumVersion(
    "Node.js",
    deps.nodeVersion ?? process.versions.node,
    manifest.minimumTools.node,
  )
  await access(path.join(paths.wikiDir, "index.md"))

  const npm = buildNpmInvocation(["--version"])
  const npmResult = await run(npm.command, npm.args, { shell: npm.shell })
  if (npmResult.code !== 0) {
    throw new Error(`npm is required: ${npmResult.stderr.trim() || "command failed"}`)
  }
  assertMinimumVersion(
    "npm",
    versionFromOutput("npm", npmResult.stdout),
    manifest.minimumTools.npm,
  )

  const gitResult = await run("git", ["--version"])
  if (gitResult.code !== 0) {
    throw new Error(`Git is required: ${gitResult.stderr.trim() || "command failed"}`)
  }
  assertMinimumVersion(
    "Git",
    versionFromOutput("Git", gitResult.stdout),
    manifest.minimumTools.git,
  )
}

async function runQuartz({ cwd, args }) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, {
      cwd,
      env: process.env,
      stdio: "inherit",
      windowsHide: true,
    })
    child.on("error", reject)
    child.on("close", (code) => resolve(code ?? 1))
  })
}

export async function main(argv, dependencies = {}) {
  const options = parseCliArgs(argv)
  const paths = resolveProjectPaths(dependencies.repoRoot ?? defaultRepoRoot)
  const operations = {
    preflight,
    ensureRuntime: ensurePinnedRuntime,
    repairRuntime: repairPinnedRuntime,
    syncRuntimeConfig: syncPinnedRuntimeConfig,
    runQuartz,
    ...dependencies,
  }

  await operations.preflight(paths, operations)
  if (options.command === "repair") {
    await operations.repairRuntime(paths, operations)
    return 0
  }

  await operations.ensureRuntime(paths, operations)
  await operations.syncRuntimeConfig(paths, operations)
  return operations.runQuartz({
    cwd: paths.runtimeDir,
    args: buildQuartzArgs(options.command, paths, options.port),
  })
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(scriptFile)) {
  main(process.argv.slice(2))
    .then((code) => {
      process.exitCode = code
    })
    .catch((error) => {
      console.error(`[docs] ${error.message}`)
      process.exitCode = 1
    })
}
