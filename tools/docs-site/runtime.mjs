import { createHash, randomUUID } from "node:crypto"
import { spawn } from "node:child_process"
import {
  access,
  cp,
  copyFile,
  mkdtemp,
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises"
import path from "node:path"

import { assertPathInside } from "./contracts.mjs"

function countOccurrences(text, value) {
  if (!value) throw new Error("Patch contexts must be non-empty")
  let count = 0
  let offset = 0
  while (true) {
    const index = text.indexOf(value, offset)
    if (index === -1) return count
    count += 1
    offset = index + value.length
  }
}

export async function applyPatchSpec(root, spec) {
  const files = new Map()
  let changed = false

  for (const replacement of spec.replacements ?? []) {
    const target = path.resolve(root, replacement.file)
    assertPathInside(root, target)
    if (!replacement.before || !replacement.after || replacement.before === replacement.after) {
      throw new Error(`Invalid patch contexts for ${replacement.file}`)
    }

    const source = files.has(target) ? files.get(target) : await readFile(target, "utf8")
    const beforeCount = countOccurrences(source, replacement.before)
    const afterCount = countOccurrences(source, replacement.after)

    if (beforeCount === 1 && afterCount === 0) {
      files.set(target, source.replace(replacement.before, replacement.after))
      changed = true
      continue
    }

    if (beforeCount === 0 && afterCount === 1) continue

    if (beforeCount > 1 && afterCount === 0) {
      throw new Error(
        `Patch ${replacement.file} expected exactly one unpatched context; found ${beforeCount}`,
      )
    }

    throw new Error(
      `Patch ${replacement.file} expected exactly one unpatched or patched context; found ${beforeCount} unpatched and ${afterCount} patched`,
    )
  }

  if (changed) {
    await Promise.all([...files].map(([target, contents]) => writeFile(target, contents)))
  }

  return changed ? "applied" : "already-applied"
}

async function pathExists(candidate) {
  try {
    await access(candidate)
    return true
  } catch {
    return false
  }
}

function sha256(contents) {
  return createHash("sha256").update(contents).digest("hex")
}

async function sha256File(file) {
  return sha256(await readFile(file))
}

async function listFiles(root, current = root) {
  const entries = await readdir(current, { withFileTypes: true })
  const files = []
  for (const entry of entries) {
    const candidate = path.join(current, entry.name)
    if (entry.isDirectory()) {
      files.push(...(await listFiles(root, candidate)))
    } else if (entry.isFile()) {
      files.push(path.relative(root, candidate).split(path.sep).join("/"))
    }
  }
  return files.sort()
}

async function hashDirectory(root) {
  const files = await listFiles(root)
  const hash = createHash("sha256")
  for (const relative of files) {
    hash.update(relative)
    hash.update("\0")
    hash.update(await readFile(path.join(root, ...relative.split("/"))))
    hash.update("\0")
  }
  return { files, hash: hash.digest("hex") }
}

async function defaultRun(command, args, options = {}) {
  return new Promise((resolve) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
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

async function runChecked(run, command, args, options, label) {
  const result = await run(command, args, options)
  if (result.code !== 0) {
    const detail = result.stderr.trim() || result.stdout.trim() || `exit code ${result.code}`
    throw new Error(`${label} failed: ${detail}`)
  }
  return result.stdout.trim()
}

function trackedInputPaths(paths, manifest) {
  return new Map([
    ["runtime-manifest.json", paths.manifestSource],
    ["quartz.lock.json", paths.lockSource],
    ...manifest.patches.map((relative) => {
      const source = path.resolve(paths.toolDir, relative)
      assertPathInside(paths.toolDir, source)
      return [relative.split(path.sep).join("/"), source]
    }),
  ])
}

async function assertMarkerInputs(paths, manifest, marker) {
  if (marker.quartzCommit !== manifest.quartz.commit) {
    throw new Error("Runtime marker Quartz commit does not match the manifest")
  }
  const inputs = trackedInputPaths(paths, manifest)
  const actualKeys = Object.keys(marker.inputHashes ?? {}).sort()
  const expectedKeys = [...inputs.keys()].sort()
  if (JSON.stringify(actualKeys) !== JSON.stringify(expectedKeys)) {
    throw new Error("Runtime marker input hash set does not match tracked inputs")
  }
  for (const [name, source] of inputs) {
    if ((await sha256File(source)) !== marker.inputHashes[name]) {
      throw new Error(`Runtime input hash changed: ${name}`)
    }
  }
}

async function createRuntimeMarker(paths, manifest) {
  const inputHashes = {}
  for (const [name, source] of trackedInputPaths(paths, manifest)) {
    inputHashes[name] = await sha256File(source)
  }
  return {
    schemaVersion: 1,
    quartzCommit: manifest.quartz.commit,
    inputHashes,
  }
}

async function assertPatchApplied(root, spec) {
  for (const replacement of spec.replacements ?? []) {
    const target = path.resolve(root, replacement.file)
    assertPathInside(root, target)
    const source = await readFile(target, "utf8")
    const beforeCount = countOccurrences(source, replacement.before)
    const afterCount = countOccurrences(source, replacement.after)
    if (beforeCount !== 0 || afterCount !== 1) {
      throw new Error(`Runtime patch is not applied exactly once: ${replacement.file}`)
    }
  }
}

async function validateRuntimeContents(paths, manifest, deps) {
  const run = deps.run ?? defaultRun
  await access(path.join(paths.runtimeDir, "quartz", "bootstrap-cli.mjs"))

  const coreCommit = await runChecked(
    run,
    "git",
    ["rev-parse", "HEAD"],
    { cwd: paths.runtimeDir },
    "Quartz commit inspection",
  )
  if (coreCommit !== manifest.quartz.commit) {
    throw new Error(
      `Quartz commit drift: expected ${manifest.quartz.commit}, detected ${coreCommit}`,
    )
  }

  const sourceLockText = await readFile(paths.lockSource, "utf8")
  const runtimeLockText = await readFile(path.join(paths.runtimeDir, "quartz.lock.json"), "utf8")
  if (runtimeLockText !== sourceLockText) {
    throw new Error("Runtime plugin lock differs from the tracked plugin lock")
  }
  const lock = JSON.parse(sourceLockText)

  for (const plugin of manifest.plugins) {
    const lockEntry = lock.plugins?.[plugin.name]
    if (lockEntry?.commit !== plugin.commit) {
      throw new Error(`Plugin lock commit drift for ${plugin.name}`)
    }
    const pluginRoot = path.resolve(paths.runtimeDir, plugin.runtimePath)
    assertPathInside(paths.runtimeDir, pluginRoot)
    const commit = await runChecked(
      run,
      "git",
      ["rev-parse", "HEAD"],
      { cwd: pluginRoot },
      `Plugin commit inspection for ${plugin.name}`,
    )
    if (commit !== plugin.commit) {
      throw new Error(
        `Plugin commit drift for ${plugin.name}: expected ${plugin.commit}, detected ${commit}`,
      )
    }
  }

  for (const relative of manifest.patches) {
    const patchFile = path.resolve(paths.toolDir, relative)
    assertPathInside(paths.toolDir, patchFile)
    await assertPatchApplied(paths.runtimeDir, JSON.parse(await readFile(patchFile, "utf8")))
  }

  const packageFile = path.resolve(paths.runtimeDir, manifest.mermaid.packagePath)
  const mermaidPackage = JSON.parse(await readFile(packageFile, "utf8"))
  if (mermaidPackage.version !== manifest.mermaid.version) {
    throw new Error(
      `Mermaid version drift: expected ${manifest.mermaid.version}, detected ${mermaidPackage.version}`,
    )
  }
  await access(path.resolve(paths.runtimeDir, manifest.mermaid.entryPath))
  const sourceTree = await hashDirectory(path.resolve(paths.runtimeDir, manifest.mermaid.distPath))
  const vendorTree = await hashDirectory(path.resolve(paths.runtimeDir, manifest.mermaid.vendorPath))
  if (
    sourceTree.files.length < 2 ||
    sourceTree.hash !== vendorTree.hash ||
    JSON.stringify(sourceTree.files) !== JSON.stringify(vendorTree.files)
  ) {
    throw new Error("Mermaid vendor tree is not a complete copy of the pinned dist")
  }

  const changedOutput = await runChecked(
    run,
    "git",
    ["diff", "--name-only"],
    { cwd: paths.runtimeDir },
    "Quartz tracked-change inspection",
  )
  const changed = changedOutput
    .split(/\r?\n/)
    .map((item) => item.trim().replaceAll("\\", "/"))
    .filter(Boolean)
    .sort()
  const allowed = [...manifest.allowedTrackedChanges].sort()
  const unexpected = changed.filter((item) => !allowed.includes(item))
  const missing = allowed.filter((item) => !changed.includes(item))
  if (unexpected.length > 0) {
    throw new Error(`Unexpected tracked Quartz change: ${unexpected.join(", ")}`)
  }
  if (missing.length > 0) {
    throw new Error(`Expected patched Quartz change is missing: ${missing.join(", ")}`)
  }
}

export async function inspectRuntime(paths, deps = {}) {
  if (!(await pathExists(paths.runtimeDir))) return { kind: "missing" }

  try {
    const runtimeStat = await stat(paths.runtimeDir)
    if (!runtimeStat.isDirectory()) throw new Error("Quartz runtime path is not a directory")

    const manifest = JSON.parse(await readFile(paths.manifestSource, "utf8"))
    const marker = JSON.parse(await readFile(paths.markerFile, "utf8"))
    if (manifest.schemaVersion !== 1 || marker.schemaVersion !== 1) {
      throw new Error("Runtime manifest or marker schema is not supported")
    }
    await assertMarkerInputs(paths, manifest, marker)
    await validateRuntimeContents(paths, manifest, deps)
    return { kind: "ready" }
  } catch (error) {
    return { kind: "invalid", reason: error.message }
  }
}

export async function ensureRuntime(paths, deps = {}) {
  const state = await inspectRuntime(paths, deps)
  if (state.kind === "ready") return
  if (state.kind === "invalid") {
    throw new Error(
      `The local Quartz runtime is invalid: ${state.reason}. Run npm run docs:repair to rebuild it.`,
    )
  }

  await (deps.provisionRuntime ?? provisionRuntime)(paths, deps)
}

export async function replaceRuntimeWithStage(paths, stagedRuntime, deps = {}) {
  const resolvedStage = path.resolve(stagedRuntime)
  assertPathInside(paths.stagingRoot, resolvedStage)
  assertPathInside(paths.cacheRoot, paths.runtimeDir)
  await access(resolvedStage)
  await mkdir(paths.cacheRoot, { recursive: true })

  const quarantine = path.join(
    paths.cacheRoot,
    `quartz-quarantine-${(deps.randomUUID ?? randomUUID)()}`,
  )
  assertPathInside(paths.cacheRoot, quarantine)
  const hadRuntime = await pathExists(paths.runtimeDir)
  if (hadRuntime) await rename(paths.runtimeDir, quarantine)

  try {
    await rename(resolvedStage, paths.runtimeDir)
  } catch (error) {
    if (hadRuntime && (await pathExists(quarantine))) {
      await rename(quarantine, paths.runtimeDir)
    }
    throw error
  }

  if (hadRuntime) {
    await rm(quarantine, { recursive: true, force: true })
  }
}

export async function syncRuntimeConfig(paths) {
  assertPathInside(paths.cacheRoot, paths.runtimeDir)
  await copyFile(paths.configSource, path.join(paths.runtimeDir, "quartz.config.yaml"))
}

async function buildStagedRuntime(paths, deps = {}) {
  const run = deps.run ?? defaultRun
  const log = deps.log ?? console.log
  await mkdir(paths.stagingRoot, { recursive: true })
  await mkdir(paths.npmCacheDir, { recursive: true })
  const stage = await mkdtemp(path.join(paths.stagingRoot, "quartz-"))
  assertPathInside(paths.stagingRoot, stage)

  try {
    const manifest = JSON.parse(await readFile(paths.manifestSource, "utf8"))
    const commandEnvironment = {
      ...process.env,
      npm_config_cache: paths.npmCacheDir,
      npm_config_update_notifier: "false",
    }
    const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm"

    log(`[docs] Cloning Quartz ${manifest.quartz.tag} into the repository cache...`)
    await runChecked(
      run,
      "git",
      [
        "clone",
        "--depth",
        "1",
        "--branch",
        manifest.quartz.tag,
        manifest.quartz.repository,
        stage,
      ],
      { cwd: paths.repoRoot, env: commandEnvironment },
      "Quartz clone",
    )
    const checkoutCommit = await runChecked(
      run,
      "git",
      ["rev-parse", "HEAD"],
      { cwd: stage, env: commandEnvironment },
      "Quartz checkout inspection",
    )
    if (checkoutCommit !== manifest.quartz.commit) {
      throw new Error(
        `Quartz tag drift: expected ${manifest.quartz.commit}, detected ${checkoutCommit}`,
      )
    }

    log("[docs] Installing the pinned Quartz dependency tree...")
    await runChecked(
      run,
      npmCommand,
      ["ci"],
      { cwd: stage, env: commandEnvironment },
      "Quartz npm install",
    )
    await copyFile(paths.configSource, path.join(stage, "quartz.config.yaml"))
    await copyFile(paths.lockSource, path.join(stage, "quartz.lock.json"))

    log("[docs] Installing the pinned Quartz community plugins...")
    await runChecked(
      run,
      process.execPath,
      ["quartz/bootstrap-cli.mjs", "plugin", "install"],
      { cwd: stage, env: commandEnvironment },
      "Quartz plugin install",
    )

    log(`[docs] Vendoring Mermaid ${manifest.mermaid.version} for offline rendering...`)
    await runChecked(
      run,
      npmCommand,
      [
        "install",
        "--no-save",
        "--package-lock=false",
        `mermaid@${manifest.mermaid.version}`,
      ],
      { cwd: stage, env: commandEnvironment },
      "Mermaid install",
    )
    const mermaidSource = path.resolve(stage, manifest.mermaid.distPath)
    const mermaidVendor = path.resolve(stage, manifest.mermaid.vendorPath)
    assertPathInside(stage, mermaidSource)
    assertPathInside(stage, mermaidVendor)
    await rm(mermaidVendor, { recursive: true, force: true })
    await cp(mermaidSource, mermaidVendor, { recursive: true })

    for (const relative of manifest.patches) {
      const patchFile = path.resolve(paths.toolDir, relative)
      assertPathInside(paths.toolDir, patchFile)
      await applyPatchSpec(stage, JSON.parse(await readFile(patchFile, "utf8")))
    }

    const stagedPaths = {
      ...paths,
      runtimeDir: stage,
      markerFile: path.join(stage, ".llm-knowledge-docs-runtime.json"),
    }
    await validateRuntimeContents(stagedPaths, manifest, deps)
    await writeFile(
      stagedPaths.markerFile,
      `${JSON.stringify(await createRuntimeMarker(paths, manifest), null, 2)}\n`,
    )
    return stage
  } catch (error) {
    await rm(stage, { recursive: true, force: true })
    throw error
  }
}

export async function provisionRuntime(paths, deps = {}) {
  const stage = await buildStagedRuntime(paths, deps)
  try {
    await replaceRuntimeWithStage(paths, stage, deps)
  } catch (error) {
    if (await pathExists(stage)) await rm(stage, { recursive: true, force: true })
    throw error
  }
}

export async function repairRuntime(paths, deps = {}) {
  await provisionRuntime(paths, deps)
  ;(deps.log ?? console.log)("[docs] Replaced the repository-local Quartz runtime.")
}
