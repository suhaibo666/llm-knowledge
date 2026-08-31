import assert from "node:assert/strict"
import test from "node:test"

import {
  diagnosticSearchIndexUrl,
  fetchWithContext,
  mermaidRootContract,
  rootMermaidSvgs,
  searchResultMatches,
} from "./smoke.mjs"


function element(tagName, parentElement = null) {
  return { tagName, parentElement }
}


test("rootMermaidSvgs excludes SVG descendants of the diagram root", () => {
  const block = element("DIV")
  const wrapper = element("DIV", block)
  const root = element("svg", wrapper)
  const group = element("g", root)
  const nested = element("svg", group)
  block.querySelectorAll = () => [root, nested]

  assert.deepEqual(rootMermaidSvgs(block), [root])
})


test("rootMermaidSvgs rejects SVG nodes outside the supplied block", () => {
  const block = element("DIV")
  const other = element("DIV")
  const detached = element("svg", other)
  block.querySelectorAll = () => [detached]

  assert.deepEqual(rootMermaidSvgs(block), [])
})


test("searchResultMatches rejects a target left by a previous query", () => {
  const stale = {
    href: "http://127.0.0.1:8000/target.html?h=上下文并行",
    text: "Megatron-LM 上下文并行",
  }

  assert.equal(
    searchResultMatches(stale, "13_megatron_cp_analysis", "/target.html"),
    false,
  )
})


test("mermaidRootContract rejects a zero-size root inside a positive block", () => {
  assert.equal(mermaidRootContract({
    error: null,
    roots: 1,
    role: "graphics-document document",
    description: "flowchart-v2",
    viewBox: "0 0 100 100",
    children: 2,
    visible: true,
    rootWidth: 0,
    rootHeight: 0,
    display: "inline",
    visibility: "visible",
    blockWidth: 752,
    blockHeight: 941,
  }), false)
})


test("search diagnostics preserve root and project base paths", () => {
  assert.equal(
    diagnosticSearchIndexUrl("http://127.0.0.1:8000"),
    "http://127.0.0.1:8000/search/search_index.json",
  )
  assert.equal(
    diagnosticSearchIndexUrl("http://127.0.0.1:8000/llm-knowledge"),
    "http://127.0.0.1:8000/llm-knowledge/search/search_index.json",
  )
})


test("direct fetch failures report method, URL, and nested socket cause", async () => {
  const cause = Object.assign(new Error("socket reset by peer"), {
    code: "ECONNRESET",
    syscall: "read",
    address: "127.0.0.1",
    port: 8123,
  })
  const failure = new TypeError("fetch failed", { cause })

  await assert.rejects(
    fetchWithContext(
      "http://127.0.0.1:8123/article.html",
      { method: "HEAD" },
      async () => { throw failure },
    ),
    (error) => {
      assert.match(error.stack, /fetchWithContext/)
      assert.match(error.message, /fetch HEAD http:\/\/127\.0\.0\.1:8123\/article\.html failed/)
      assert.match(error.message, /ECONNRESET/)
      assert.match(error.message, /socket reset by peer/)
      return true
    },
  )
})
