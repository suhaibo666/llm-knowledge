import { spawn } from "node:child_process"
import { access } from "node:fs/promises"
import path from "node:path"

function unique(values) {
  return [...new Set(values.filter(Boolean))]
}

function browserCandidates(env, platform) {
  if (platform === "win32") {
    const programFilesX86 = env["ProgramFiles(x86)"] ?? env.PROGRAMFILES_X86
    const programFiles = env.ProgramFiles ?? env.PROGRAMFILES
    const localAppData = env.LOCALAPPDATA
    const roots = unique([programFilesX86, programFiles, localAppData])
    return roots.flatMap((root) => [
      path.win32.join(root, "Microsoft", "Edge", "Application", "msedge.exe"),
      path.win32.join(root, "Google", "Chrome", "Application", "chrome.exe"),
    ])
  }
  if (platform === "darwin") {
    return [
      "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
      "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
  }
  return [
    "/usr/bin/microsoft-edge",
    "/usr/bin/microsoft-edge-stable",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/var/lib/flatpak/exports/bin/com.google.Chrome",
  ]
}

export async function findBrowserExecutable(
  env = process.env,
  platform = process.platform,
  dependencies = {},
) {
  const accessImpl = dependencies.accessImpl ?? access
  const candidates = unique([
    env.LLM_KNOWLEDGE_BROWSER?.trim(),
    ...browserCandidates(env, platform),
  ])

  for (const candidate of candidates) {
    try {
      await accessImpl(candidate)
      return candidate
    } catch {
      // Continue through deterministic local browser candidates.
    }
  }

  throw new Error(
    "No supported Edge, Chrome, or Chromium executable was found. " +
      "Set LLM_KNOWLEDGE_BROWSER to the browser executable path.",
  )
}

export function parsePowerShellListeners(output) {
  const trimmed = output.trim()
  if (!trimmed) return []
  const parsed = JSON.parse(trimmed)
  if (parsed === null) return []
  return (Array.isArray(parsed) ? parsed : [parsed]).map((record) => ({
    address: String(record.LocalAddress),
    port: Number(record.LocalPort),
    ...(record.OwningProcess === undefined
      ? {}
      : { pid: Number(record.OwningProcess) }),
  }))
}

function parseEndpoint(endpoint) {
  const match = endpoint.match(/^\[?(.+?)\]?:(\d+)$/)
  if (!match) return undefined
  return { address: match[1], port: Number(match[2]) }
}

export function parseSsListeners(output) {
  const records = []
  for (const line of output.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    const fields = trimmed.split(/\s+/)
    const endpoint = parseEndpoint(fields[3] ?? "")
    if (!endpoint) continue
    const pidMatch = trimmed.match(/pid=(\d+)/)
    records.push({
      ...endpoint,
      ...(pidMatch ? { pid: Number(pidMatch[1]) } : {}),
    })
  }
  return records
}

export function parseLsofListeners(output) {
  const records = []
  for (const line of output.split(/\r?\n/).slice(1)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    const fields = trimmed.split(/\s+/)
    const tcpIndex = fields.indexOf("TCP")
    if (tcpIndex < 0) continue
    const endpoint = parseEndpoint(fields[tcpIndex + 1] ?? "")
    if (!endpoint) continue
    const pid = Number(fields[1])
    records.push({
      ...endpoint,
      ...(Number.isInteger(pid) ? { pid } : {}),
    })
  }
  return records
}

async function runCaptured(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env ?? process.env,
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
    child.once("error", reject)
    child.once("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr })
      } else {
        reject(
          new Error(
            `Listener inspection failed (${command}, exit ${code ?? "unknown"}): ${stderr.trim()}`,
          ),
        )
      }
    })
  })
}

function parsePowerShellProcesses(output) {
  const trimmed = output.trim()
  if (!trimmed) return []
  const parsed = JSON.parse(trimmed)
  if (parsed === null) return []
  return (Array.isArray(parsed) ? parsed : [parsed]).map((record) => ({
    pid: Number(record.ProcessId),
    parentPid: Number(record.ParentProcessId),
  }))
}

function parsePsProcesses(output) {
  const records = []
  for (const line of output.split(/\r?\n/)) {
    const match = line.trim().match(/^(\d+)\s+(\d+)$/)
    if (!match) continue
    records.push({ pid: Number(match[1]), parentPid: Number(match[2]) })
  }
  return records
}

export function descendantPids(rootPid, records) {
  const found = new Set([Number(rootPid)])
  let changed = true
  while (changed) {
    changed = false
    for (const record of records) {
      if (found.has(record.parentPid) && !found.has(record.pid)) {
        found.add(record.pid)
        changed = true
      }
    }
  }
  return [...found].sort((left, right) => left - right)
}

export async function findDescendantPids(rootPid, dependencies = {}) {
  const platform = dependencies.platform ?? process.platform
  const run = dependencies.run ?? runCaptured
  let records

  if (platform === "win32") {
    const result = await run("powershell.exe", [
      "-NoLogo",
      "-NoProfile",
      "-NonInteractive",
      "-Command",
      "@(Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId) | ConvertTo-Json -Compress",
    ])
    records = parsePowerShellProcesses(result.stdout)
  } else {
    const result = await run("ps", ["-eo", "pid=,ppid="])
    records = parsePsProcesses(result.stdout)
  }

  return descendantPids(rootPid, records)
}

export async function inspectSystemListeners(ports, dependencies = {}) {
  const platform = dependencies.platform ?? process.platform
  const run = dependencies.run ?? runCaptured

  if (platform === "win32") {
    const portList = ports.join(",")
    const script =
      `$wanted = @(${portList}); ` +
      "@(Get-NetTCPConnection -State Listen -ErrorAction Stop | " +
      "Where-Object { $wanted -contains $_.LocalPort } | " +
      "Select-Object LocalAddress, LocalPort, OwningProcess) | " +
      "ConvertTo-Json -Compress"
    const result = await run("powershell.exe", [
      "-NoLogo",
      "-NoProfile",
      "-NonInteractive",
      "-Command",
      script,
    ])
    return parsePowerShellListeners(result.stdout)
  }

  if (platform === "darwin") {
    const result = await run("lsof", ["-nP", "-iTCP", "-sTCP:LISTEN"])
    return parseLsofListeners(result.stdout)
  }

  const result = await run("ss", ["-ltnpH"])
  return parseSsListeners(result.stdout)
}

export function assertListenerRecords(ports, records, options = {}) {
  const allowedPids = options.allowedPids
    ? new Set(options.allowedPids.map(Number))
    : undefined
  // Binding 0.0.0.0 means "every interface"; the OS may report the wildcard
  // itself or a concrete address, so accept both.
  const expectedHost = options.expectedHost ?? "127.0.0.1"
  const wildcard = expectedHost === "0.0.0.0" || expectedHost === "::"

  for (const port of ports) {
    const matches = records.filter((record) => record.port === port)
    if (matches.length === 0) {
      throw new Error(`No listening socket was found for port ${port}`)
    }

    for (const record of matches) {
      if (!wildcard && record.address !== expectedHost) {
        throw new Error(
          `Port ${port} is listening on ${record.address}; expected only ${expectedHost}`,
        )
      }
      if (allowedPids && record.pid !== undefined && !allowedPids.has(Number(record.pid))) {
        throw new Error(
          `Port ${port} belongs to process ${record.pid}, not the docs service`,
        )
      }
    }
  }
}

export async function assertLoopbackListeners(ports, dependencies = {}) {
  const inspectListeners = dependencies.inspectListeners ?? inspectSystemListeners
  const records = await inspectListeners(ports, dependencies)
  let allowedPids = dependencies.allowedPids
  if (allowedPids === undefined && dependencies.expectedPid !== undefined) {
    const getDescendantPids = dependencies.getDescendantPids ?? findDescendantPids
    allowedPids = await getDescendantPids(dependencies.expectedPid, dependencies)
  }
  assertListenerRecords(ports, records, {
    allowedPids,
    expectedHost: dependencies.expectedHost,
  })
  return records
}
