import assert from "node:assert/strict"
import path from "node:path"
import test from "node:test"

import {
  assertMinimumVersion,
  assertPathInside,
  buildQuartzArgs,
  parseCliArgs,
  resolveProjectPaths,
} from "./contracts.mjs"

test("serve defaults to the paired documentation ports", () => {
  assert.deepEqual(parseCliArgs(["serve"]), {
    command: "serve",
    port: 8080,
    wsPort: 8081,
    openBrowser: true,
  })
})

test("serve accepts a port and disables browser opening", () => {
  assert.deepEqual(parseCliArgs(["serve", "--port", "9090", "--no-open"]), {
    command: "serve",
    port: 9090,
    wsPort: 9091,
    openBrowser: false,
  })
})

test("build and repair never open a browser", () => {
  assert.deepEqual(parseCliArgs(["build"]), {
    command: "build",
    port: 8080,
    wsPort: 8081,
    openBrowser: false,
  })
  assert.deepEqual(parseCliArgs(["repair"]), {
    command: "repair",
    port: 8080,
    wsPort: 8081,
    openBrowser: false,
  })
})

test("invalid commands, flags, and port boundaries fail", () => {
  assert.throws(() => parseCliArgs(["publish"]), /serve, build, or repair/)
  assert.throws(() => parseCliArgs(["serve", "--wat"]), /Unknown argument/)
  assert.throws(() => parseCliArgs(["serve", "--port", "0"]), /1 through 65534/)
  assert.throws(() => parseCliArgs(["serve", "--port", "65535"]), /1 through 65534/)
  assert.throws(() => parseCliArgs(["serve", "--port", "abc"]), /1 through 65534/)
})

test("single-use flags reject ambiguous duplicates", () => {
  assert.throws(
    () => parseCliArgs(["serve", "--port", "8080", "--port", "8082"]),
    /--port may be provided only once/,
  )
  assert.throws(
    () => parseCliArgs(["serve", "--no-open", "--no-open"]),
    /--no-open may be provided only once/,
  )
})

test("semantic version floors compare numeric components", () => {
  assert.doesNotThrow(() => assertMinimumVersion("Node.js", "22.0.0", "22.0.0"))
  assert.doesNotThrow(() => assertMinimumVersion("npm", "11.0.0", "10.9.2"))
  assert.throws(
    () => assertMinimumVersion("npm", "10.9.1", "10.9.2"),
    /requires npm >= 10.9.2; detected 10.9.1/,
  )
})

test("project paths keep mutable state in the dedicated cache", () => {
  const repoRoot = path.resolve("repo")
  const paths = resolveProjectPaths(repoRoot)

  assert.equal(paths.wikiDir, path.join(repoRoot, "wiki"))
  assert.equal(
    paths.runtimeDir,
    path.join(repoRoot, ".cache", "llm-knowledge-docs", "quartz"),
  )
  assert.equal(
    paths.markerFile,
    path.join(paths.runtimeDir, ".llm-knowledge-docs-runtime.json"),
  )
  assert.doesNotThrow(() => assertPathInside(paths.cacheRoot, paths.runtimeDir))
  assert.throws(() => assertPathInside(paths.cacheRoot, repoRoot), /outside dedicated cache root/)
})

test("Quartz build and serve arguments use wiki and cache output", () => {
  const paths = resolveProjectPaths(path.resolve("repo"))

  assert.deepEqual(buildQuartzArgs("build", paths, 8080), [
    "quartz/bootstrap-cli.mjs",
    "build",
    "--directory",
    paths.wikiDir,
    "--output",
    paths.outputDir,
  ])
  assert.deepEqual(buildQuartzArgs("serve", paths, 8080), [
    "quartz/bootstrap-cli.mjs",
    "build",
    "--serve",
    "--directory",
    paths.wikiDir,
    "--output",
    paths.outputDir,
    "--port",
    "8080",
    "--ws-port",
    "8081",
  ])
})
