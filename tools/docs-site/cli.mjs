import { spawn } from "node:child_process"
import { access, readFile } from "node:fs/promises"
import { networkInterfaces } from "node:os"
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
import {
  assertPortAvailable as assertLoopbackPortAvailable,
  installSignalForwarding as installQuartzSignalForwarding,
  openBrowser as openSystemBrowser,
  startQuartz as startQuartzService,
  stopQuartz as stopQuartzService,
  waitForHttp as waitForQuartzHttp,
} from "./processes.mjs"

function localAddresses() {
  const found = []
  for (const entries of Object.values(networkInterfaces())) {
    for (const entry of entries ?? []) {
      if (entry.family === "IPv4" && !entry.internal) found.push(entry.address)
    }
  }
  return found
}

const scriptFile = fileURLToPath(import.meta.url)
const defaultRepoRoot = path.resolve(path.dirname(scriptFile), "..", "..")
export const DOCS_STARTUP_TIMEOUT_MS = 180_000

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
    assertPortAvailable: assertLoopbackPortAvailable,
    startQuartz: startQuartzService,
    waitForHttp: waitForQuartzHttp,
    openBrowser: openSystemBrowser,
    installSignalForwarding: installQuartzSignalForwarding,
    stopQuartz: stopQuartzService,
    ...dependencies,
  }

  await operations.preflight(paths, operations)
  if (options.command === "repair") {
    await operations.repairRuntime(paths, operations)
    return 0
  }

  await operations.ensureRuntime(paths, operations)
  await operations.syncRuntimeConfig(paths, operations)
  const quartzOptions = {
    cwd: paths.runtimeDir,
    args: buildQuartzArgs(options.command, paths, options.port),
    // The patched Quartz reads this for both the HTTP and hot-reload sockets.
    env: { ...process.env, DOCS_BIND_HOST: options.host },
  }

  if (options.command === "build") {
    return operations.runQuartz(quartzOptions)
  }

  await operations.assertPortAvailable(options.port, { host: options.host })
  await operations.assertPortAvailable(options.wsPort, { host: options.host })
  const service = operations.startQuartz(quartzOptions)
  const removeSignalForwarding = operations.installSignalForwarding(service)
  const controller = new AbortController()
  // Probe over loopback even when bound to 0.0.0.0: 0.0.0.0 is not a
  // connectable destination, it only means "every interface".
  const url = `http://127.0.0.1:${options.port}/`

  try {
    const readiness = Promise.resolve(
      operations.waitForHttp(url, {
        signal: controller.signal,
        timeoutMs: DOCS_STARTUP_TIMEOUT_MS,
      }),
    ).then(
      () => ({ kind: "ready" }),
      (error) => ({ kind: "readiness-error", error }),
    )
    const earlyExit = service.exitCode.then(
      (code) => ({ kind: "exit", code }),
      (error) => ({ kind: "child-error", error }),
    )
    const outcome = await Promise.race([readiness, earlyExit])

    if (outcome.kind === "exit") {
      controller.abort()
      if (outcome.code !== 0) return outcome.code
      throw new Error(`Quartz exited before the documentation service became ready at ${url}`)
    }
    if (outcome.kind === "child-error") {
      controller.abort()
      throw outcome.error
    }
    if (outcome.kind === "readiness-error") {
      controller.abort()
      await operations.stopQuartz(service)
      throw outcome.error
    }

    console.log(`[docs] Documentation is available at ${url}`)
    if (options.host === "0.0.0.0" || options.host === "::") {
      for (const address of localAddresses()) {
        console.log(`[docs] Reachable from this network at http://${address}:${options.port}/`)
      }
    } else if (options.host !== "127.0.0.1" && options.host !== "localhost") {
      console.log(`[docs] Bound to ${options.host}:${options.port}`)
    }
    if (options.openBrowser) {
      let opened = false
      try {
        opened = await operations.openBrowser(url)
      } catch (error) {
        console.warn(`[docs] Could not open a browser: ${error.message}`)
      }
      if (!opened) {
        console.warn(`[docs] Open ${url} in your browser.`)
      }
    }
    return await service.exitCode
  } finally {
    controller.abort()
    removeSignalForwarding()
  }
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
