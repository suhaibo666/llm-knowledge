import assert from "node:assert/strict"
import { access, readFile, readdir } from "node:fs/promises"
import { createRequire } from "node:module"
import path from "node:path"
import test from "node:test"
import { fileURLToPath } from "node:url"

import { findBrowserExecutable } from "../docs-site/listeners.mjs"

const scriptFile = fileURLToPath(import.meta.url)
const toolDir = path.dirname(scriptFile)
const repoRoot = path.resolve(toolDir, "..", "..")
const wikiRoot = path.join(repoRoot, "wiki")
const nodeModules = path.join(toolDir, "node_modules")
const mermaidScript = path.join(nodeModules, "mermaid", "dist", "mermaid.min.js")
const puppeteerPackage = path.join(nodeModules, "puppeteer-core", "package.json")

async function markdownFiles(root) {
  const files = []
  async function visit(directory) {
    const entries = await readdir(directory, { withFileTypes: true })
    entries.sort((left, right) => left.name.localeCompare(right.name, "en"))
    for (const entry of entries) {
      const candidate = path.join(directory, entry.name)
      if (entry.isDirectory()) await visit(candidate)
      else if (entry.isFile() && entry.name.toLowerCase().endsWith(".md")) {
        files.push(candidate)
      }
    }
  }
  await visit(root)
  return files
}

function mermaidBlocks(markdown, relativePath) {
  const blocks = []
  let fence = null
  const lines = markdown.split(/\r?\n/)

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    if (fence) {
      const close = line.match(/^ {0,3}(`{3,}|~{3,})[ \t]*$/)
      if (close && close[1][0] === fence.character && close[1].length >= fence.length) {
        if (fence.isMermaid) {
          blocks.push({
            sourcePath: relativePath,
            line: fence.line,
            source: fence.lines.join("\n"),
          })
        }
        fence = null
      } else if (fence.isMermaid) {
        fence.lines.push(line)
      }
      continue
    }

    const open = line.match(/^ {0,3}(`{3,}|~{3,})[ \t]*([^\s`]*)[^\r\n]*$/)
    if (!open) continue
    fence = {
      character: open[1][0],
      length: open[1].length,
      isMermaid: open[2].toLowerCase() === "mermaid",
      line: index + 1,
      lines: [],
    }
  }

  if (fence?.isMermaid) {
    assert.fail(`${relativePath}:${fence.line} has an unclosed Mermaid fence`)
  }
  return blocks
}

async function corpusDiagrams() {
  const diagrams = []
  for (const file of await markdownFiles(wikiRoot)) {
    const relativePath = path.relative(repoRoot, file).split(path.sep).join("/")
    diagrams.push(...mermaidBlocks(await readFile(file, "utf8"), relativePath))
  }
  return diagrams
}

async function loadPuppeteer() {
  await access(mermaidScript)
  await access(puppeteerPackage)
  return createRequire(import.meta.url)(path.join(nodeModules, "puppeteer-core"))
}

test("every wiki Mermaid block parses with the shipped browser runtime", { timeout: 120_000 }, async (context) => {
  const diagrams = await corpusDiagrams()
  assert.ok(diagrams.length > 0, "wiki corpus contains no Mermaid diagrams")
  context.diagnostic(`parsing ${diagrams.length} Mermaid blocks from wiki/`)

  const puppeteer = await loadPuppeteer()
  const browser = await puppeteer.launch({
    executablePath: await findBrowserExecutable(),
    headless: true,
    args: ["--disable-gpu", "--no-sandbox"],
  })

  try {
    const page = await browser.newPage()
    await page.setContent("<!doctype html><html><body></body></html>")
    await page.addScriptTag({ path: mermaidScript })
    const results = await page.evaluate(async (items) => {
      window.mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict",
        theme: "default",
        htmlLabels: false,
        flowchart: { htmlLabels: false },
      })
      const parsed = []
      for (const item of items) {
        try {
          await window.mermaid.parse(item.source)
          parsed.push({ ...item, error: null })
        } catch (error) {
          parsed.push({
            ...item,
            error: error && (error.str || error.message) || String(error),
          })
        }
      }
      return parsed
    }, diagrams)

    assert.equal(results.length, diagrams.length, "parser skipped a Mermaid block")
    const failures = results.filter((result) => result.error)
    assert.deepEqual(
      failures,
      [],
      failures.map((failure) =>
        `${failure.sourcePath}:${failure.line}\n${failure.error}`,
      ).join("\n\n"),
    )
  } finally {
    await browser.close()
  }
})
