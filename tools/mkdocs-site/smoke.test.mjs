import assert from "node:assert/strict"
import test from "node:test"

import { rootMermaidSvgs } from "./smoke.mjs"


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
