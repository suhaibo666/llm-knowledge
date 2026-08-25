import { spawn } from "node:child_process"
import net from "node:net"
import { setTimeout as delay } from "node:timers/promises"

const LOOPBACK_HOST = "127.0.0.1"

export async function assertPortAvailable(port, dependencies = {}) {
  const createServer = dependencies.createServer ?? (() => net.createServer())
  const host = dependencies.host ?? LOOPBACK_HOST
  const server = createServer()

  await new Promise((resolve, reject) => {
    const onError = (error) => {
      server.removeListener("listening", onListening)
      if (error.code === "EADDRINUSE" || error.code === "EACCES") {
        reject(new Error(`Loopback port ${port} is already in use or unavailable`))
        return
      }
      reject(error)
    }
    const onListening = () => {
      server.removeListener("error", onError)
      server.close((error) => (error ? reject(error) : resolve()))
    }

    server.once("error", onError)
    server.once("listening", onListening)
    server.listen({ host, port, exclusive: true })
  })
}

export async function waitForHttp(url, dependencies = {}) {
  const fetchImpl = dependencies.fetchImpl ?? globalThis.fetch
  const now = dependencies.now ?? Date.now
  const sleep = dependencies.sleep ?? ((milliseconds) => delay(milliseconds))
  const timeoutMs = dependencies.timeoutMs ?? 60_000
  const pollIntervalMs = dependencies.pollIntervalMs ?? 250
  const signal = dependencies.signal
  const deadline = now() + timeoutMs
  let lastFailure = "no response"

  while (now() < deadline) {
    if (signal?.aborted) {
      throw signal.reason ?? new Error(`Waiting for ${url} was cancelled`)
    }

    try {
      const remaining = Math.max(1, deadline - now())
      const requestTimeout = AbortSignal.timeout(Math.min(2_000, remaining))
      const requestSignal = signal
        ? AbortSignal.any([signal, requestTimeout])
        : requestTimeout
      const response = await fetchImpl(url, {
        cache: "no-store",
        redirect: "manual",
        signal: requestSignal,
      })
      if (response.status >= 200 && response.status < 400) return response
      lastFailure = `HTTP ${response.status}`
    } catch (error) {
      if (signal?.aborted) {
        throw signal.reason ?? error
      }
      lastFailure = error.message
    }

    const remaining = deadline - now()
    if (remaining <= 0) break
    await sleep(Math.min(pollIntervalMs, remaining))
  }

  throw new Error(`Timed out waiting for ${url} (${lastFailure})`)
}

export function browserInvocation(url, options = {}) {
  const env = options.env ?? process.env
  const platform = options.platform ?? process.platform
  const explicitBrowser = env.LLM_KNOWLEDGE_BROWSER?.trim()

  if (explicitBrowser) {
    return { command: explicitBrowser, args: [url], detached: true }
  }
  if (platform === "win32") {
    return {
      command: "cmd.exe",
      args: ["/d", "/s", "/c", `start "" "${url}"`],
      detached: false,
    }
  }
  if (platform === "darwin") {
    return { command: "open", args: [url], detached: false }
  }
  return { command: "xdg-open", args: [url], detached: false }
}

export async function openBrowser(url, dependencies = {}) {
  const spawnImpl = dependencies.spawnImpl ?? spawn
  const invocation = browserInvocation(url, dependencies)

  return new Promise((resolve) => {
    let child
    try {
      child = spawnImpl(invocation.command, invocation.args, {
        detached: invocation.detached,
        stdio: "ignore",
        windowsHide: true,
      })
    } catch {
      resolve(false)
      return
    }

    let settled = false
    const finish = (result) => {
      if (settled) return
      settled = true
      resolve(result)
    }
    child.once("error", () => finish(false))
    child.once("spawn", () => {
      child.unref?.()
      finish(true)
    })
  })
}

function signalExitCode(signal) {
  if (signal === "SIGINT") return 130
  if (signal === "SIGTERM") return 143
  return 1
}

export function startQuartz(options, dependencies = {}) {
  const spawnImpl = dependencies.spawnImpl ?? spawn
  const execPath = dependencies.execPath ?? process.execPath
  const child = spawnImpl(execPath, options.args, {
    cwd: options.cwd,
    env: options.env ?? process.env,
    stdio: "inherit",
    windowsHide: true,
  })
  let settled = false
  let terminationRequested = false

  const exitCode = new Promise((resolve, reject) => {
    child.once("error", (error) => {
      settled = true
      reject(error)
    })
    child.once("close", (code, signal) => {
      settled = true
      resolve(code ?? signalExitCode(signal))
    })
  })

  return {
    pid: child.pid,
    exitCode,
    terminate(signal = "SIGTERM") {
      if (settled || terminationRequested) return false
      terminationRequested = true
      return child.kill(signal)
    },
    forceKill() {
      if (settled) return false
      return child.kill("SIGKILL")
    },
  }
}

export function installSignalForwarding(service, dependencies = {}) {
  const processLike = dependencies.processLike ?? process
  let forwarded = false
  const handlers = new Map()

  for (const signal of ["SIGINT", "SIGTERM"]) {
    const handler = () => {
      if (forwarded) return
      forwarded = true
      service.terminate(signal)
    }
    handlers.set(signal, handler)
    processLike.on(signal, handler)
  }

  return () => {
    for (const [signal, handler] of handlers) {
      processLike.removeListener(signal, handler)
    }
  }
}

export async function stopQuartz(service, dependencies = {}) {
  const timeoutMs = dependencies.timeoutMs ?? 5_000
  const setTimer = dependencies.setTimeoutImpl ?? setTimeout
  const clearTimer = dependencies.clearTimeoutImpl ?? clearTimeout
  const waitForExit = (milliseconds) => new Promise((resolve) => {
    const timer = setTimer(() => resolve(false), milliseconds)
    service.exitCode.then(
      () => {
        clearTimer(timer)
        resolve(true)
      },
      () => {
        clearTimer(timer)
        resolve(true)
      },
    )
  })
  service.terminate("SIGTERM")

  if (await waitForExit(timeoutMs)) return

  service.forceKill?.()
  await waitForExit(Math.min(timeoutMs, 1_000))
}
