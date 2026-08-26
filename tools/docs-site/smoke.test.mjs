import assert from "node:assert/strict"
import test from "node:test"

import { headingIdFromHash, localNetworkViolations } from "./smoke.mjs"

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
