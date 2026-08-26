import assert from "node:assert/strict"
import test from "node:test"

import {
  assertListenerRecords,
  assertLoopbackListeners,
  descendantPids,
  findBrowserExecutable,
  parseLsofListeners,
  parsePowerShellListeners,
  parseSsListeners,
} from "./listeners.mjs"

test("browser discovery honors LLM_KNOWLEDGE_BROWSER before platform candidates", async () => {
  const checked = []
  const explicit = "D:\\Portable\\browser.exe"
  const result = await findBrowserExecutable(
    { LLM_KNOWLEDGE_BROWSER: explicit, ProgramFiles: "C:\\Program Files" },
    "win32",
    {
      accessImpl: async (candidate) => {
        checked.push(candidate)
        if (candidate !== explicit) throw new Error("missing")
      },
    },
  )

  assert.equal(result, explicit)
  assert.deepEqual(checked, [explicit])
})

test("browser discovery checks stable Edge and Chrome locations", async () => {
  const edge = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"
  const checked = []
  const result = await findBrowserExecutable(
    {
      "ProgramFiles(x86)": "C:\\Program Files (x86)",
      ProgramFiles: "C:\\Program Files",
      LOCALAPPDATA: "C:\\Users\\me\\AppData\\Local",
    },
    "win32",
    {
      accessImpl: async (candidate) => {
        checked.push(candidate)
        if (candidate !== edge) throw new Error("missing")
      },
    },
  )

  assert.equal(result, edge)
  assert.equal(checked[0], edge)
})

test("missing browser error explains the environment override", async () => {
  await assert.rejects(
    findBrowserExecutable({}, "linux", {
      accessImpl: async () => {
        throw new Error("missing")
      },
    }),
    /LLM_KNOWLEDGE_BROWSER/,
  )
})

test("PowerShell listener JSON normalizes a singleton and an array", () => {
  assert.deepEqual(
    parsePowerShellListeners(
      '[{"LocalAddress":"127.0.0.1","LocalPort":8080,"OwningProcess":42}]',
    ),
    [{ address: "127.0.0.1", port: 8080, pid: 42 }],
  )
  assert.deepEqual(
    parsePowerShellListeners(
      '{"LocalAddress":"0.0.0.0","LocalPort":8081,"OwningProcess":43}',
    ),
    [{ address: "0.0.0.0", port: 8081, pid: 43 }],
  )
})

test("ss and lsof listener output expose address, port, and PID", () => {
  assert.deepEqual(
    parseSsListeners(
      'LISTEN 0 511 127.0.0.1:8080 0.0.0.0:* users:(("node",pid=123,fd=20))\n',
    ),
    [{ address: "127.0.0.1", port: 8080, pid: 123 }],
  )
  assert.deepEqual(
    parseLsofListeners(
      "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n" +
        "node 456 user 20u IPv4 0x0 0t0 TCP 127.0.0.1:8081 (LISTEN)\n",
    ),
    [{ address: "127.0.0.1", port: 8081, pid: 456 }],
  )
})

test("listener verification rejects wildcard and non-loopback exposure", () => {
  assert.throws(
    () =>
      assertListenerRecords([8080, 8081], [
        { port: 8080, address: "127.0.0.1" },
        { port: 8081, address: "0.0.0.0" },
      ]),
    /8081.*0\.0\.0\.0/,
  )
  assert.throws(
    () =>
      assertListenerRecords([8080], [{ port: 8080, address: "192.168.1.20" }]),
    /8080.*192\.168\.1\.20/,
  )
})

test("listener verification requires both exact IPv4 loopback listeners", () => {
  assert.doesNotThrow(() =>
    assertListenerRecords(
      [8080, 8081],
      [
        { port: 8080, address: "127.0.0.1", pid: 77 },
        { port: 8081, address: "127.0.0.1", pid: 77 },
      ],
      { allowedPids: [77] },
    ),
  )
  assert.throws(
    () => assertListenerRecords([8080, 8081], [{ port: 8080, address: "127.0.0.1" }]),
    /No listening socket was found for port 8081/,
  )
  assert.throws(
    () =>
      assertListenerRecords(
        [8080],
        [{ port: 8080, address: "127.0.0.1", pid: 88 }],
        { allowedPids: [77] },
      ),
    /port 8080.*process 88/i,
  )
})

test("system listener assertion delegates inspection then validates ownership", async () => {
  const calls = []
  await assertLoopbackListeners([8080, 8081], {
    expectedPid: 99,
    inspectListeners: async (ports) => {
      calls.push(ports)
      return ports.map((port) => ({ port, address: "127.0.0.1", pid: 99 }))
    },
    getDescendantPids: async (pid) => [pid],
  })
  assert.deepEqual(calls, [[8080, 8081]])
})

test("listener ownership accepts the service process and recursive children only", () => {
  assert.deepEqual(
    descendantPids(10, [
      { pid: 10, parentPid: 1 },
      { pid: 11, parentPid: 10 },
      { pid: 12, parentPid: 10 },
      { pid: 13, parentPid: 11 },
      { pid: 20, parentPid: 1 },
    ]),
    [10, 11, 12, 13],
  )
})
