import path from "node:path"

const COMMANDS = new Set(["serve", "build", "repair"])
const DEFAULT_PORT = 8080

export function parseCliArgs(argv) {
  const [command, ...rest] = argv
  if (!COMMANDS.has(command)) {
    throw new Error("The documentation command must be serve, build, or repair")
  }

  let port = DEFAULT_PORT
  let sawPort = false
  let sawNoOpen = false

  for (let index = 0; index < rest.length; index += 1) {
    const argument = rest[index]
    if (argument === "--port") {
      if (sawPort) {
        throw new Error("--port may be provided only once")
      }
      sawPort = true
      const rawPort = rest[index + 1]
      index += 1
      port = Number(rawPort)
      if (!Number.isInteger(port) || port < 1 || port > 65534) {
        throw new Error("The HTTP port must be an integer from 1 through 65534")
      }
      continue
    }

    if (argument === "--no-open") {
      if (sawNoOpen) {
        throw new Error("--no-open may be provided only once")
      }
      sawNoOpen = true
      continue
    }

    throw new Error(`Unknown argument: ${argument}`)
  }

  return {
    command,
    port,
    wsPort: port + 1,
    openBrowser: command === "serve" && !sawNoOpen,
  }
}

function parseVersion(version) {
  const match = String(version).match(/^(\d+)(?:\.(\d+))?(?:\.(\d+))?/)
  if (!match) {
    throw new Error(`Cannot parse version: ${version}`)
  }
  return match.slice(1, 4).map((part) => Number(part ?? 0))
}

export function assertMinimumVersion(name, actual, minimum) {
  const actualParts = parseVersion(actual)
  const minimumParts = parseVersion(minimum)
  for (let index = 0; index < minimumParts.length; index += 1) {
    if (actualParts[index] > minimumParts[index]) return
    if (actualParts[index] < minimumParts[index]) {
      throw new Error(`${name} requires ${name} >= ${minimum}; detected ${actual}`)
    }
  }
}

export function assertPathInside(parent, candidate) {
  const parentPath = path.resolve(parent)
  const candidatePath = path.resolve(candidate)
  const relative = path.relative(parentPath, candidatePath)
  if (!relative || relative.startsWith(`..${path.sep}`) || relative === ".." || path.isAbsolute(relative)) {
    throw new Error(`${candidatePath} is outside dedicated cache root ${parentPath}`)
  }
}

export function resolveProjectPaths(repoRoot) {
  const resolvedRoot = path.resolve(repoRoot)
  const toolDir = path.join(resolvedRoot, "tools", "docs-site")
  const cacheRoot = path.join(resolvedRoot, ".cache", "llm-knowledge-docs")
  const runtimeDir = path.join(cacheRoot, "quartz")

  return {
    repoRoot: resolvedRoot,
    wikiDir: path.join(resolvedRoot, "wiki"),
    cacheRoot,
    runtimeDir,
    outputDir: path.join(cacheRoot, "output"),
    stagingRoot: path.join(cacheRoot, "staging"),
    npmCacheDir: path.join(cacheRoot, "npm-cache"),
    toolDir,
    configSource: path.join(toolDir, "quartz.config.yaml"),
    lockSource: path.join(toolDir, "quartz.lock.json"),
    manifestSource: path.join(toolDir, "runtime-manifest.json"),
    patchesDir: path.join(toolDir, "patches"),
    markerFile: path.join(runtimeDir, ".llm-knowledge-docs-runtime.json"),
  }
}

export function buildQuartzArgs(mode, paths, port) {
  const common = [
    "quartz/bootstrap-cli.mjs",
    "build",
    "--directory",
    paths.wikiDir,
    "--output",
    paths.outputDir,
  ]

  if (mode === "build") return common
  if (mode !== "serve") {
    throw new Error(`Unsupported Quartz mode: ${mode}`)
  }

  return [
    "quartz/bootstrap-cli.mjs",
    "build",
    "--serve",
    "--directory",
    paths.wikiDir,
    "--output",
    paths.outputDir,
    "--port",
    String(port),
    "--ws-port",
    String(port + 1),
  ]
}
