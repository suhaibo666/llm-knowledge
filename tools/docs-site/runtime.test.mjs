import assert from "node:assert/strict"
import { createHash } from "node:crypto"
import { cp, mkdtemp, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"
import test from "node:test"

import { resolveProjectPaths } from "./contracts.mjs"
import {
  applyPatchSpec,
  buildNpmInvocation,
  ensureRuntime,
  inspectRuntime,
  provisionRuntime,
  repairRuntime,
  replaceRuntimeWithStage,
  syncRuntimeConfig,
} from "./runtime.mjs"

async function temporaryDirectory(prefix) {
  return mkdtemp(path.join(tmpdir(), prefix))
}

test("npm commands use the active npm CLI instead of spawning npm.cmd", () => {
  assert.deepEqual(
    buildNpmInvocation(
      ["ci"],
      { npm_execpath: "C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npm-cli.js" },
      "win32",
      "C:\\Program Files\\nodejs\\node.exe",
    ),
    {
      command: "C:\\Program Files\\nodejs\\node.exe",
      args: [
        "C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npm-cli.js",
        "ci",
      ],
      shell: false,
    },
  )
})

function sha256(text) {
  return createHash("sha256").update(text).digest("hex")
}

async function writeJson(file, value) {
  const text = `${JSON.stringify(value, null, 2)}\n`
  await mkdir(path.dirname(file), { recursive: true })
  await writeFile(file, text)
  return text
}

async function createReadyRuntimeFixture() {
  const repoRoot = await temporaryDirectory("docs-runtime-")
  const paths = resolveProjectPaths(repoRoot)
  const coreCommit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  const pluginCommit = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  const pluginName = "obsidian-flavored-markdown"
  const pluginRoot = `.quartz/plugins/${pluginName}`
  const manifest = {
    schemaVersion: 1,
    quartz: {
      repository: "https://example.invalid/quartz.git",
      tag: "v5.0.0",
      commit: coreCommit,
    },
    corePackages: [
      {
        name: "@quartz-community/types",
        version: "0.2.1",
        packagePath: "node_modules/@quartz-community/types/package.json",
        entryPath: "node_modules/@quartz-community/types/dist/index.js",
      },
      {
        name: "@quartz-community/utils",
        version: "0.1.0",
        packagePath: "node_modules/@quartz-community/utils/package.json",
        entryPath: "node_modules/@quartz-community/utils/dist/index.js",
      },
    ],
    mermaid: {
      version: "11.4.0",
      installRoot: ".llm-knowledge-docs-vendor",
      packagePath: ".llm-knowledge-docs-vendor/node_modules/mermaid/package.json",
      distPath: ".llm-knowledge-docs-vendor/node_modules/mermaid/dist",
      vendorPath: "quartz/static/vendor/mermaid",
      entryPath: "quartz/static/vendor/mermaid/mermaid.esm.min.mjs",
    },
    plugins: [
      { name: pluginName, commit: pluginCommit, runtimePath: pluginRoot },
    ],
    patches: ["patches/core.json", "patches/ofm.json"],
    allowedTrackedChanges: ["quartz/cli/handlers.js"],
  }
  const lock = {
    version: "1.0.0",
    plugins: {
      [pluginName]: {
        source: `github:quartz-community/${pluginName}`,
        resolved: `https://example.invalid/${pluginName}.git`,
        commit: pluginCommit,
      },
    },
  }
  const corePatch = {
    id: "core",
    replacements: [
      {
        file: "quartz/cli/handlers.js",
        before: "server.listen(argv.port)",
        after: 'server.listen(argv.port, "127.0.0.1")',
      },
    ],
  }
  const pluginPatch = {
    id: "ofm",
    replacements: [
      {
        file: `${pluginRoot}/dist/index.js`,
        before: "https://cdn.invalid/mermaid.mjs",
        after: "/static/vendor/mermaid/mermaid.esm.min.mjs",
      },
    ],
  }

  const manifestText = await writeJson(paths.manifestSource, manifest)
  const lockText = await writeJson(paths.lockSource, lock)
  await mkdir(path.dirname(paths.configSource), { recursive: true })
  await writeFile(paths.configSource, "configuration:\n  pageTitle: Test Wiki\n")
  const corePatchPath = path.join(paths.toolDir, "patches", "core.json")
  const pluginPatchPath = path.join(paths.toolDir, "patches", "ofm.json")
  const corePatchText = await writeJson(corePatchPath, corePatch)
  const pluginPatchText = await writeJson(pluginPatchPath, pluginPatch)

  await mkdir(path.join(paths.runtimeDir, "quartz", "cli"), { recursive: true })
  await mkdir(path.join(paths.runtimeDir, pluginRoot, "dist"), { recursive: true })
  await mkdir(
    path.join(
      paths.runtimeDir,
      ".llm-knowledge-docs-vendor",
      "node_modules",
      "mermaid",
      "dist",
      "chunks",
    ),
    { recursive: true },
  )
  await mkdir(
    path.join(paths.runtimeDir, "quartz", "static", "vendor", "mermaid", "chunks"),
    { recursive: true },
  )
  await writeFile(path.join(paths.runtimeDir, "quartz", "bootstrap-cli.mjs"), "export {}\n")
  await writeFile(
    path.join(paths.runtimeDir, "quartz", "cli", "handlers.js"),
    'server.listen(argv.port, "127.0.0.1")\n',
  )
  await writeFile(
    path.join(paths.runtimeDir, pluginRoot, "dist", "index.js"),
    'import("/static/vendor/mermaid/mermaid.esm.min.mjs")\n',
  )
  await writeFile(
    path.join(
      paths.runtimeDir,
      ".llm-knowledge-docs-vendor",
      "node_modules",
      "mermaid",
      "package.json",
    ),
    JSON.stringify({ name: "mermaid", version: "11.4.0" }),
  )
  for (const corePackage of manifest.corePackages) {
    const packageFile = path.join(paths.runtimeDir, ...corePackage.packagePath.split("/"))
    const entryFile = path.join(paths.runtimeDir, ...corePackage.entryPath.split("/"))
    await mkdir(path.dirname(entryFile), { recursive: true })
    await writeFile(
      packageFile,
      JSON.stringify({ name: corePackage.name, version: corePackage.version }),
    )
    await writeFile(entryFile, "export {}\n")
  }
  const mermaidFiles = new Map([
    ["mermaid.esm.min.mjs", "export default {}\n"],
    [path.join("chunks", "chunk.mjs"), "export const chunk = true\n"],
  ])
  for (const [relative, contents] of mermaidFiles) {
    await writeFile(
      path.join(
        paths.runtimeDir,
        ".llm-knowledge-docs-vendor",
        "node_modules",
        "mermaid",
        "dist",
        relative,
      ),
      contents,
    )
    await writeFile(
      path.join(paths.runtimeDir, "quartz", "static", "vendor", "mermaid", relative),
      contents,
    )
  }
  await writeFile(path.join(paths.runtimeDir, "quartz.lock.json"), lockText)
  await writeJson(paths.markerFile, {
    schemaVersion: 1,
    quartzCommit: coreCommit,
    inputHashes: {
      "runtime-manifest.json": sha256(manifestText),
      "quartz.lock.json": sha256(lockText),
      "patches/core.json": sha256(corePatchText),
      "patches/ofm.json": sha256(pluginPatchText),
    },
  })

  const run = async (command, args, options = {}) => {
    assert.equal(command, "git")
    if (args[0] === "rev-parse") {
      return {
        code: 0,
        stdout: options.cwd.includes(pluginRoot.replaceAll("/", path.sep))
          ? `${pluginCommit}\n`
          : `${coreCommit}\n`,
        stderr: "",
      }
    }
    if (args[0] === "diff") {
      return { code: 0, stdout: "quartz/cli/handlers.js\n", stderr: "" }
    }
    throw new Error(`Unexpected git command: ${args.join(" ")}`)
  }

  return { paths, run, manifest, corePatchPath }
}

test("exact-once patches apply once and recognize the patched state", async () => {
  const root = await temporaryDirectory("docs-patch-")
  await writeFile(path.join(root, "target.js"), "server.listen(argv.port)\n")
  const spec = {
    replacements: [
      {
        file: "target.js",
        before: "server.listen(argv.port)",
        after: 'server.listen(argv.port, "127.0.0.1")',
      },
    ],
  }

  assert.equal(await applyPatchSpec(root, spec), "applied")
  assert.equal(await applyPatchSpec(root, spec), "already-applied")
  assert.equal(
    await readFile(path.join(root, "target.js"), "utf8"),
    'server.listen(argv.port, "127.0.0.1")\n',
  )
})

test("patch context drift and duplicate context are rejected", async () => {
  const root = await temporaryDirectory("docs-patch-")
  const target = path.join(root, "target.js")
  const spec = {
    replacements: [{ file: "target.js", before: "old-value", after: "new-value" }],
  }

  await writeFile(target, "different-value")
  await assert.rejects(
    applyPatchSpec(root, spec),
    /expected exactly one unpatched or patched context/,
  )

  await writeFile(target, "old-value old-value")
  await assert.rejects(applyPatchSpec(root, spec), /expected exactly one unpatched context/)
})

test("patch output cannot retain its complete unpatched context", async () => {
  const root = await temporaryDirectory("docs-patch-")
  await writeFile(path.join(root, "target.js"), "const oldValue = true\n")

  await assert.rejects(
    applyPatchSpec(root, {
      replacements: [
        {
          file: "target.js",
          before: "const oldValue = true",
          after: "const oldValue = true\nconst newValue = true",
        },
      ],
    }),
    /after context cannot contain its complete before context/,
  )
})

test("exact patching accepts an after context nested inside its before block", async () => {
  const root = await temporaryDirectory("docs-patch-")
  const target = path.join(root, "target.js")
  await writeFile(
    target,
    "  if (footer) {\n    defaultLayout.footer = footer\n  }\n",
  )
  const spec = {
    replacements: [
      {
        file: "target.js",
        before: "  if (footer) {\n    defaultLayout.footer = footer\n  }",
        after: "  defaultLayout.footer = footer",
      },
    ],
  }

  assert.equal(await applyPatchSpec(root, spec), "applied")
  assert.equal(await applyPatchSpec(root, spec), "already-applied")
  assert.equal(await readFile(target, "utf8"), "  defaultLayout.footer = footer\n")
})

test("patch targets cannot escape their declared root", async () => {
  const root = await temporaryDirectory("docs-patch-")
  const outside = path.join(root, "..", "outside.js")
  await writeFile(outside, "old-value")

  await assert.rejects(
    applyPatchSpec(root, {
      replacements: [{ file: "../outside.js", before: "old-value", after: "new-value" }],
    }),
    /outside dedicated cache root/,
  )
})

test("an absent runtime is reported as missing", async () => {
  const repoRoot = await temporaryDirectory("docs-runtime-")
  const state = await inspectRuntime(resolveProjectPaths(repoRoot))
  assert.deepEqual(state, { kind: "missing" })
})

test("a marker alone cannot make an incomplete runtime ready", async () => {
  const repoRoot = await temporaryDirectory("docs-runtime-")
  const paths = resolveProjectPaths(repoRoot)
  await mkdir(paths.runtimeDir, { recursive: true })
  await writeFile(paths.markerFile, JSON.stringify({ schemaVersion: 1 }))

  const state = await inspectRuntime(paths)

  assert.equal(state.kind, "invalid")
  assert.match(state.reason, /manifest|marker|Quartz|runtime/i)
})

test("a pinned patched runtime with a complete Mermaid copy is ready", async () => {
  const { paths, run } = await createReadyRuntimeFixture()
  assert.deepEqual(await inspectRuntime(paths, { run }), { kind: "ready" })
})

test("runtime inspection rejects core and plugin commit drift", async () => {
  const coreFixture = await createReadyRuntimeFixture()
  const wrongCoreRun = async (command, args, options) => {
    const result = await coreFixture.run(command, args, options)
    if (args[0] === "rev-parse" && options.cwd === coreFixture.paths.runtimeDir) {
      return { ...result, stdout: `${"c".repeat(40)}\n` }
    }
    return result
  }
  const coreState = await inspectRuntime(coreFixture.paths, { run: wrongCoreRun })
  assert.equal(coreState.kind, "invalid")
  assert.match(coreState.reason, /Quartz commit/i)

  const pluginFixture = await createReadyRuntimeFixture()
  const wrongPluginRun = async (command, args, options) => {
    const result = await pluginFixture.run(command, args, options)
    if (args[0] === "rev-parse" && options.cwd.includes("obsidian-flavored-markdown")) {
      return { ...result, stdout: `${"d".repeat(40)}\n` }
    }
    return result
  }
  const pluginState = await inspectRuntime(pluginFixture.paths, { run: wrongPluginRun })
  assert.equal(pluginState.kind, "invalid")
  assert.match(pluginState.reason, /plugin.*commit/i)
})

test("runtime inspection rejects tracked input and lock drift", async () => {
  const inputFixture = await createReadyRuntimeFixture()
  await writeFile(inputFixture.corePatchPath, "{}\n")
  const inputState = await inspectRuntime(inputFixture.paths, { run: inputFixture.run })
  assert.equal(inputState.kind, "invalid")
  assert.match(inputState.reason, /input hash/i)

  const lockFixture = await createReadyRuntimeFixture()
  await writeFile(lockFixture.paths.lockSource, '{"version":"1.0.0","plugins":{}}\n')
  const lockState = await inspectRuntime(lockFixture.paths, { run: lockFixture.run })
  assert.equal(lockState.kind, "invalid")
  assert.match(lockState.reason, /input hash|plugin lock/i)
})

test("runtime inspection rejects incomplete Mermaid vendor trees", async () => {
  const fixture = await createReadyRuntimeFixture()
  await rm(
    path.join(
      fixture.paths.runtimeDir,
      "quartz",
      "static",
      "vendor",
      "mermaid",
      "chunks",
      "chunk.mjs",
    ),
  )

  const state = await inspectRuntime(fixture.paths, { run: fixture.run })
  assert.equal(state.kind, "invalid")
  assert.match(state.reason, /Mermaid.*complete|vendor/i)
})

test("runtime inspection rejects missing pinned Quartz support package builds", async () => {
  const fixture = await createReadyRuntimeFixture()
  await rm(
    path.join(
      fixture.paths.runtimeDir,
      "node_modules",
      "@quartz-community",
      "utils",
      "dist",
      "index.js",
    ),
  )

  const state = await inspectRuntime(fixture.paths, { run: fixture.run })
  assert.equal(state.kind, "invalid")
  assert.match(state.reason, /Quartz support package.*utils|utils.*entry/i)
})

test("runtime inspection rejects unexpected tracked Quartz changes", async () => {
  const fixture = await createReadyRuntimeFixture()
  const runWithExtraDiff = async (command, args, options) => {
    const result = await fixture.run(command, args, options)
    if (args[0] === "diff") {
      return {
        ...result,
        stdout: "quartz/cli/handlers.js\nquartz/build.ts\n",
      }
    }
    return result
  }

  const state = await inspectRuntime(fixture.paths, { run: runWithExtraDiff })
  assert.equal(state.kind, "invalid")
  assert.match(state.reason, /unexpected tracked Quartz change.*quartz\/build\.ts/i)
})

test("ensureRuntime provisions a missing runtime but preserves an invalid one", async () => {
  const missingRoot = await temporaryDirectory("docs-runtime-")
  const missingPaths = resolveProjectPaths(missingRoot)
  await ensureRuntime(missingPaths, {
    provisionRuntime: async () => {
      await mkdir(missingPaths.runtimeDir, { recursive: true })
      await writeFile(path.join(missingPaths.runtimeDir, "provisioned"), "yes\n")
    },
  })
  assert.equal(
    await readFile(path.join(missingPaths.runtimeDir, "provisioned"), "utf8"),
    "yes\n",
  )

  const invalidRoot = await temporaryDirectory("docs-runtime-")
  const invalidPaths = resolveProjectPaths(invalidRoot)
  await mkdir(invalidPaths.runtimeDir, { recursive: true })
  const sentinel = path.join(invalidPaths.runtimeDir, "user-sentinel")
  await writeFile(sentinel, "preserve\n")
  await assert.rejects(ensureRuntime(invalidPaths), /npm run docs:repair/)
  assert.equal(await readFile(sentinel, "utf8"), "preserve\n")
})

test("runtime replacement accepts only staging descendants and installs the staged tree", async () => {
  const repoRoot = await temporaryDirectory("docs-runtime-")
  const paths = resolveProjectPaths(repoRoot)
  const stage = path.join(paths.stagingRoot, "quartz-stage")
  await mkdir(paths.runtimeDir, { recursive: true })
  await mkdir(stage, { recursive: true })
  await writeFile(path.join(paths.runtimeDir, "version"), "old\n")
  await writeFile(path.join(stage, "version"), "new\n")

  await replaceRuntimeWithStage(paths, stage)

  assert.equal(await readFile(path.join(paths.runtimeDir, "version"), "utf8"), "new\n")
  await assert.rejects(
    replaceRuntimeWithStage(paths, path.join(repoRoot, "outside-stage")),
    /outside dedicated cache root|staging root/,
  )
})

test("configuration sync copies the tracked YAML into the runtime", async () => {
  const repoRoot = await temporaryDirectory("docs-runtime-")
  const paths = resolveProjectPaths(repoRoot)
  await mkdir(paths.runtimeDir, { recursive: true })
  await mkdir(path.dirname(paths.configSource), { recursive: true })
  await writeFile(paths.configSource, "configuration:\n  pageTitle: Test Wiki\n")

  await syncRuntimeConfig(paths)

  assert.equal(
    await readFile(path.join(paths.runtimeDir, "quartz.config.yaml"), "utf8"),
    "configuration:\n  pageTitle: Test Wiki\n",
  )
})

test("provisionRuntime installs a staged runtime that passes full inspection", async () => {
  const fixture = await createReadyRuntimeFixture()
  const template = path.join(fixture.paths.repoRoot, "quartz-template")
  await rename(fixture.paths.runtimeDir, template)

  let mermaidInstallDirectory
  let pluginSubcommand
  const run = async (command, args, options = {}) => {
    if (args[0] === "quartz/bootstrap-cli.mjs" && args[1] === "plugin") {
      pluginSubcommand = args[2]
    }
    if (command === "git" && args[0] === "clone") {
      await cp(template, args.at(-1), { recursive: true })
      return { code: 0, stdout: "", stderr: "" }
    }
    if (command === "git") {
      if (args[0] === "rev-parse") {
        return {
          code: 0,
          stdout: options.cwd.includes("obsidian-flavored-markdown")
            ? `${fixture.manifest.plugins[0].commit}\n`
            : `${fixture.manifest.quartz.commit}\n`,
          stderr: "",
        }
      }
      if (args[0] === "diff") {
        return { code: 0, stdout: "quartz/cli/handlers.js\n", stderr: "" }
      }
    }
    if (command === process.execPath || command === "npm" || command === "npm.cmd") {
      if (args.some((argument) => argument === "mermaid@11.4.0")) {
        mermaidInstallDirectory = options.cwd
        await cp(path.join(template, ".llm-knowledge-docs-vendor"), options.cwd, {
          recursive: true,
        })
      }
      return { code: 0, stdout: "", stderr: "" }
    }
    throw new Error(`Unexpected provision command: ${command} ${args.join(" ")}`)
  }

  await provisionRuntime(fixture.paths, {
    run,
    log: () => undefined,
    randomUUID: () => "provision-test",
  })

  assert.deepEqual(await inspectRuntime(fixture.paths, { run }), { kind: "ready" })
  assert.equal(
    await readFile(path.join(fixture.paths.runtimeDir, "quartz.config.yaml"), "utf8"),
    "configuration:\n  pageTitle: Test Wiki\n",
  )
  assert.equal(path.basename(mermaidInstallDirectory), ".llm-knowledge-docs-vendor")
  // `plugin install` follows each plugin's branch tip, so one upstream push breaks
  // provisioning against the pinned commits. Only `restore` honours the lockfile.
  assert.equal(pluginSubcommand, "restore")
})

test("repairRuntime leaves the old runtime untouched when staging fails", async () => {
  const repoRoot = await temporaryDirectory("docs-runtime-")
  const paths = resolveProjectPaths(repoRoot)
  await mkdir(paths.runtimeDir, { recursive: true })
  const sentinel = path.join(paths.runtimeDir, "sentinel")
  await writeFile(sentinel, "old-runtime\n")
  await mkdir(path.dirname(paths.manifestSource), { recursive: true })
  await writeJson(paths.manifestSource, {
    schemaVersion: 1,
    quartz: {
      repository: "https://example.invalid/quartz.git",
      tag: "v5.0.0",
      commit: "a".repeat(40),
    },
    mermaid: { version: "11.4.0" },
    plugins: [],
    patches: [],
    allowedTrackedChanges: [],
  })

  await assert.rejects(
    repairRuntime(paths, {
      run: async () => ({ code: 1, stdout: "", stderr: "network unavailable" }),
      log: () => undefined,
      randomUUID: () => "failed-repair",
    }),
    /network unavailable/,
  )

  assert.equal(await readFile(sentinel, "utf8"), "old-runtime\n")
})
