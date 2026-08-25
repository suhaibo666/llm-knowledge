import assert from "node:assert/strict"
import { EventEmitter } from "node:events"
import net from "node:net"
import test from "node:test"

import {
  assertPortAvailable,
  browserInvocation,
  installSignalForwarding,
  openBrowser,
  startQuartz,
  waitForHttp,
} from "./processes.mjs"

test("port probing accepts a free loopback port and rejects an occupied one", async () => {
  const holder = net.createServer()
  await new Promise((resolve, reject) => {
    holder.once("error", reject)
    holder.listen({ host: "127.0.0.1", port: 0 }, resolve)
  })
  const occupiedPort = holder.address().port

  try {
    await assert.rejects(assertPortAvailable(occupiedPort), /already in use/)
  } finally {
    await new Promise((resolve) => holder.close(resolve))
  }

  await assert.doesNotReject(assertPortAvailable(occupiedPort))
})

test("HTTP readiness retries transient failures and accepts redirects", async () => {
  let attempts = 0
  let now = 0
  await waitForHttp("http://127.0.0.1:8080/", {
    fetchImpl: async () => {
      attempts += 1
      if (attempts === 1) throw new Error("not listening")
      return { status: 302 }
    },
    now: () => now,
    sleep: async (milliseconds) => {
      now += milliseconds
    },
    timeoutMs: 1_000,
    pollIntervalMs: 25,
  })
  assert.equal(attempts, 2)
})

test("HTTP readiness fails with an actionable bounded timeout", async () => {
  let now = 0
  await assert.rejects(
    waitForHttp("http://127.0.0.1:8080/", {
      fetchImpl: async () => {
        throw new Error("connection refused")
      },
      now: () => now,
      sleep: async (milliseconds) => {
        now += milliseconds
      },
      timeoutMs: 50,
      pollIntervalMs: 25,
    }),
    /Timed out waiting for http:\/\/127\.0\.0\.1:8080\//,
  )
})

test("browser invocation honors the explicit browser before platform defaults", () => {
  assert.deepEqual(
    browserInvocation("http://127.0.0.1:8080/", {
      env: { LLM_KNOWLEDGE_BROWSER: "C:\\Browser\\browser.exe" },
      platform: "win32",
    }),
    {
      command: "C:\\Browser\\browser.exe",
      args: ["http://127.0.0.1:8080/"],
      detached: true,
    },
  )
  assert.deepEqual(
    browserInvocation("http://127.0.0.1:8080/", { env: {}, platform: "darwin" }),
    { command: "open", args: ["http://127.0.0.1:8080/"], detached: false },
  )
  assert.deepEqual(
    browserInvocation("http://127.0.0.1:8080/", { env: {}, platform: "linux" }),
    { command: "xdg-open", args: ["http://127.0.0.1:8080/"], detached: false },
  )
  assert.deepEqual(
    browserInvocation("http://127.0.0.1:8080/", { env: {}, platform: "win32" }),
    {
      command: "cmd.exe",
      args: ["/d", "/s", "/c", 'start "" "http://127.0.0.1:8080/"'],
      detached: false,
    },
  )
})

test("browser opening reports launch success and failure without throwing", async () => {
  const successChild = new EventEmitter()
  successChild.unref = () => undefined
  const success = openBrowser("http://127.0.0.1:8080/", {
    env: { LLM_KNOWLEDGE_BROWSER: "browser" },
    platform: "linux",
    spawnImpl: () => {
      queueMicrotask(() => successChild.emit("spawn"))
      return successChild
    },
  })
  assert.equal(await success, true)

  const failedChild = new EventEmitter()
  const failure = openBrowser("http://127.0.0.1:8080/", {
    env: {},
    platform: "linux",
    spawnImpl: () => {
      queueMicrotask(() => failedChild.emit("error", new Error("missing opener")))
      return failedChild
    },
  })
  assert.equal(await failure, false)
})

test("signal forwarding sends only the first termination signal and cleans up", () => {
  const processLike = new EventEmitter()
  const signals = []
  const cleanup = installSignalForwarding(
    { terminate: (signal) => signals.push(signal) },
    { processLike },
  )

  processLike.emit("SIGINT")
  processLike.emit("SIGINT")
  processLike.emit("SIGTERM")
  assert.deepEqual(signals, ["SIGINT"])

  cleanup()
  assert.equal(processLike.listenerCount("SIGINT"), 0)
  assert.equal(processLike.listenerCount("SIGTERM"), 0)
})

test("Quartz child wrapper exposes exit status and one termination method", async () => {
  const child = new EventEmitter()
  child.pid = 1234
  const killed = []
  child.kill = (signal) => {
    killed.push(signal)
    return true
  }
  const service = startQuartz(
    { cwd: "repo", args: ["quartz/bootstrap-cli.mjs"] },
    { spawnImpl: () => child, execPath: "node" },
  )

  assert.equal(service.pid, 1234)
  service.terminate("SIGTERM")
  service.terminate("SIGINT")
  assert.deepEqual(killed, ["SIGTERM"])

  child.emit("close", 7, null)
  assert.equal(await service.exitCode, 7)
})
