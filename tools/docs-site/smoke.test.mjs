import assert from "node:assert/strict"
import { createServer, get } from "node:http"
import test from "node:test"
import { setTimeout as delay } from "node:timers/promises"

import * as smoke from "./smoke.mjs"

const { headingIdFromHash, localNetworkViolations } = smoke

test("heading fragments decode to DOM ids without becoming CSS selectors", () => {
  assert.equal(headingIdFromHash("#1-%E9%80%82%E7%94%A8%E5%9C%BA%E6%99%AF"), "1-适用场景")
  assert.equal(headingIdFromHash("#1-适用场景与前置"), "1-适用场景与前置")
})

test("network auditing allows inline data but requires exact loopback for HTTP and WebSocket", () => {
  assert.deepEqual(
    localNetworkViolations([
      "http://127.0.0.1:8080/index.css",
      "ws://127.0.0.1:8081/",
      "data:image/svg+xml,%3Csvg%3E%3C/svg%3E",
    ]),
    [],
  )
  assert.deepEqual(
    localNetworkViolations([
      "ws://localhost:8081/",
      "https://example.com/font.woff2",
    ]),
    ["ws://localhost:8081/", "https://example.com/font.woff2"],
  )
})

test("proxy shutdown is bounded when a response remains open", async () => {
  assert.equal(typeof smoke.closeServer, "function")

  let acceptRequest
  const requestAccepted = new Promise((resolve) => {
    acceptRequest = resolve
  })
  const server = createServer(() => acceptRequest())
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve))
  const address = server.address()
  assert.ok(address && typeof address !== "string")

  const client = get(`http://127.0.0.1:${address.port}/`)
  client.on("error", () => undefined)
  await requestAccepted

  const closePromise = smoke.closeServer(server, 25)
  const outcome = await Promise.race([
    closePromise.then(() => "closed"),
    delay(500).then(() => "timed-out"),
  ])
  if (outcome === "timed-out") {
    server.closeAllConnections()
    await closePromise
  }
  client.destroy()

  assert.equal(outcome, "closed")
  assert.equal(server.listening, false)
})
