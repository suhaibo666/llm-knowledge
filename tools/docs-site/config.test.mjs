import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import path from "node:path"
import test from "node:test"
import { fileURLToPath } from "node:url"

const toolDir = path.dirname(fileURLToPath(import.meta.url))

test("Quartz configuration enables the local docs rendering contract", async () => {
  const config = await readFile(path.join(toolDir, "quartz.config.yaml"), "utf8")

  for (const required of [
    "pageTitle: LLM Knowledge Wiki",
    "locale: zh-CN",
    "fontOrigin: local",
    "cdnCaching: false",
    "markdownLinkResolution: shortest",
    "renderEngine: mathjax",
    "mermaid: true",
    "enableSiteMap: false",
    "enableRSS: false",
  ]) {
    assert.ok(config.includes(required), `missing Quartz setting: ${required}`)
  }

  for (const plugin of [
    "syntax-highlighting",
    "obsidian-flavored-markdown",
    "github-flavored-markdown",
    "table-of-contents",
    "crawl-links",
    "description",
    "latex",
    "remove-draft",
    "alias-redirects",
    "content-index",
    "favicon",
    "content-page",
    "folder-page",
    "explorer",
    "search",
    "article-title",
    "page-title",
    "darkmode",
    "breadcrumbs",
  ]) {
    assert.match(config, new RegExp(`source: github:quartz-community/${plugin}\\s+enabled: true`))
  }
})

test("Quartz configuration excludes remote and non-goal features", async () => {
  const config = await readFile(path.join(toolDir, "quartz.config.yaml"), "utf8")

  assert.doesNotMatch(config, /https?:\/\//)
  for (const plugin of [
    "graph",
    "backlinks",
    "reader-mode",
    "recent-notes",
    "comments",
    "footer",
    "og-image",
    "cname",
    "canvas-page",
    "bases-page",
    "encrypted-pages",
    "stacked-pages",
  ]) {
    assert.doesNotMatch(config, new RegExp(`source: github:quartz-community/${plugin}(?:\\s|$)`))
  }
  assert.match(config, /enablePopovers: false/)
  assert.match(config, /enableYouTubeEmbed: false/)
  assert.match(config, /enableTweetEmbed: false/)
  assert.match(config, /enableVideoEmbed: false/)
})

test("runtime manifest and lock agree on every enabled plugin commit", async () => {
  const manifest = JSON.parse(await readFile(path.join(toolDir, "runtime-manifest.json"), "utf8"))
  const lock = JSON.parse(await readFile(path.join(toolDir, "quartz.lock.json"), "utf8"))
  const manifestPins = Object.fromEntries(
    manifest.plugins.map((plugin) => [plugin.name, plugin.commit]),
  )
  const lockPins = Object.fromEntries(
    Object.entries(lock.plugins).map(([name, plugin]) => [name, plugin.commit]),
  )

  assert.deepEqual(manifestPins, lockPins)
  assert.equal(manifest.quartz.commit, "ab346fa66a895e12d63a308e70ce330ba795822a")
  assert.equal(manifest.mermaid.version, "11.4.0")
  assert.equal(manifest.mermaid.installRoot, ".llm-knowledge-docs-vendor")
  assert.equal(
    manifest.mermaid.packagePath,
    ".llm-knowledge-docs-vendor/node_modules/mermaid/package.json",
  )
  assert.deepEqual(
    Object.fromEntries(manifest.corePackages.map((item) => [item.name, item.version])),
    {
      "@quartz-community/types": "0.2.1",
      "@quartz-community/utils": "0.1.0",
    },
  )
  assert.deepEqual(manifest.patches, [
    "patches/quartz-v5-core-local-only.json",
    "patches/quartz-v5-ofm-local-mermaid.json",
    "patches/quartz-v5-crawl-links-obsidian-paths.json",
    "patches/quartz-v5-breadcrumbs-no-frontmatter.json",
    "patches/quartz-v5-responsive-layout.json",
  ])
  assert.deepEqual([...manifest.allowedTrackedChanges].sort(), [
    "quartz.lock.json",
    "quartz/cli/handlers.js",
    "quartz/components/Head.tsx",
    "quartz/plugins/index.ts",
    "quartz/plugins/loader/config-loader.ts",
    "quartz/styles/custom.scss",
  ])
})

test("the core patch removes Quartz's implicit OG plugin dependency", async () => {
  const patch = JSON.parse(
    await readFile(
      path.join(toolDir, "patches", "quartz-v5-core-local-only.json"),
      "utf8",
    ),
  )
  const replacements = patch.replacements.map(({ before, after }) => ({ before, after }))

  assert.ok(
    replacements.some(
      ({ before, after }) =>
        before.includes('import { CustomOgImagesEmitterName } from "../../.quartz/plugins"') &&
        !after.includes("CustomOgImagesEmitterName"),
    ),
  )
  assert.ok(
    replacements.some(
      ({ before, after }) =>
        before.includes("i18n(cfg.locale).propertyDefaults.title") &&
        after.includes("fileData.frontmatter?.title ?? cfg.pageTitle"),
    ),
  )
  assert.ok(
    replacements.some(
      ({ before, after }) =>
        before.includes("ctx.cfg.plugins.emitters.some") &&
        after.includes("const usesCustomOgImage = true") &&
        after.includes('const ogImageDefaultPath = ""'),
    ),
  )
})

test("the core patch supplies a structural no-op footer without a plugin", async () => {
  const patch = JSON.parse(
    await readFile(
      path.join(toolDir, "patches", "quartz-v5-core-local-only.json"),
      "utf8",
    ),
  )
  const footerReplacements = patch.replacements.filter(
    (replacement) => replacement.file === "quartz/plugins/loader/config-loader.ts",
  )

  assert.equal(footerReplacements.length, 3)
  assert.ok(
    footerReplacements.some(
      ({ after }) =>
        after.includes("const EmptyFooter: QuartzComponent = () => null") &&
        after.includes("let footer: QuartzComponent = EmptyFooter"),
    ),
  )
  assert.ok(
    footerReplacements.some(({ after }) => after.trim() === "defaultLayout.footer = footer"),
  )
  assert.ok(
    footerReplacements.some(({ after }) => after.includes("if (!pt.footer) pt.footer = footer")),
  )
})

test("the core patch uses Node's supported gray style name in serve logs", async () => {
  const patch = JSON.parse(
    await readFile(
      path.join(toolDir, "patches", "quartz-v5-core-local-only.json"),
      "utf8",
    ),
  )
  const styleReplacements = patch.replacements.filter(
    ({ file, before }) =>
      file === "quartz/cli/handlers.js" && before.includes('styleText("grey"'),
  )

  assert.equal(styleReplacements.length, 4)
  for (const replacement of styleReplacements) {
    assert.doesNotMatch(replacement.after, /styleText\("grey"/)
    assert.match(replacement.after, /styleText\("gray"/)
  }
})

test("the core patch uses exact IPv4 loopback for listeners and browser endpoints", async () => {
  const patch = JSON.parse(
    await readFile(
      path.join(toolDir, "patches", "quartz-v5-core-local-only.json"),
      "utf8",
    ),
  )
  const combined = patch.replacements.map(({ after }) => after).join("\n")

  assert.match(combined, /server\.listen\(argv\.port, "127\.0\.0\.1"\)/)
  assert.match(combined, /new WebSocketServer\(\{ port: argv\.wsPort, host: "127\.0\.0\.1" \}\)/)
  assert.match(combined, /`ws:\/\/127\.0\.0\.1:\$\{ctx\.argv\.wsPort\}`/)
  assert.match(combined, /http:\/\/127\.0\.0\.1:\$\{argv\.port\}/)
})

test("the crawl-links patch resolves unique Obsidian path suffixes", async () => {
  const patch = JSON.parse(
    await readFile(
      path.join(toolDir, "patches", "quartz-v5-crawl-links-obsidian-paths.json"),
      "utf8",
    ),
  )
  const combined = patch.replacements.map(({ after }) => after).join("\n")

  assert.equal(patch.id, "quartz-v5-crawl-links-obsidian-paths")
  assert.match(combined, /slug2\.endsWith\(`\/\$\{normalizedTarget\}`\)/)
  assert.match(combined, /suffixMatches\.length === 1/)
  assert.match(combined, /resolveRelative\(src, suffixMatches\[0\]\)/)
  assert.match(combined, /transformObsidianLink\(fileSlug, dest, transformOptions\)/)
  assert.ok(
    patch.replacements.some(
      ({ before, after }) =>
        before.includes("node.properties.src = transformLink") &&
        after.includes("node.properties.src = transformObsidianLink"),
    ),
  )
  for (const replacement of patch.replacements) {
    assert.equal(
      replacement.after.includes(replacement.before),
      false,
      `patched output must not retain its full before context: ${replacement.file}`,
    )
  }
})

test("the breadcrumbs patch includes Markdown files without frontmatter", async () => {
  const patch = JSON.parse(
    await readFile(
      path.join(toolDir, "patches", "quartz-v5-breadcrumbs-no-frontmatter.json"),
      "utf8",
    ),
  )
  const combined = patch.replacements.map(({ after }) => after).join("\n")

  assert.equal(patch.id, "quartz-v5-breadcrumbs-no-frontmatter")
  assert.deepEqual(
    [...new Set(patch.replacements.map(({ file }) => file))].sort(),
    [
      ".quartz/plugins/breadcrumbs/dist/components/index.js",
      ".quartz/plugins/breadcrumbs/dist/index.js",
    ],
  )
  assert.match(combined, /file\.slug && file\.filePath/)
  assert.match(combined, /file\.frontmatter\?\.title \?\? file\.filePath\.split/)
  assert.equal(
    patch.replacements.filter(({ after }) => after.includes("typedCtx.breadcrumbTrie ??=")).length,
    2,
  )
  assert.doesNotMatch(
    patch.replacements.map(({ after }) => after).join("\n"),
    /typedCtx\.trie \?\?=/,
  )
  assert.doesNotMatch(combined, /if \(file\.frontmatter\)/)
})
