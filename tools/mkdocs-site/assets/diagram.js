(function () {
  "use strict";
  const renderedBlocks = new WeakSet();
  const diagramSources = new WeakMap();
  let diagramSequence = 0;

  function sourceFor(node, savedSource) {
    if (diagramSources.has(node)) return diagramSources.get(node);
    node.classList.add("no-mathjax");
    const code = node.children.length === 1 && node.firstElementChild.tagName === "CODE"
      ? node.firstElementChild
      : null;
    const source = savedSource || (code ? code.textContent : node.textContent);
    diagramSources.set(node, source);
    return source;
  }

  function captureSources(root) {
    const saved = window.__kbMermaidSourceList || [];
    (root || document).querySelectorAll(".mermaid").forEach(function (node, index) {
      sourceFor(node, saved[index]);
    });
  }

  function currentScheme() {
    return document.body.dataset.mdColorScheme === "slate" ? "dark" : "default";
  }

  async function renderDiagrams(root) {
    if (!window.mermaid) return;
    captureSources(root);
    const nodes = Array.from((root || document).querySelectorAll(".mermaid"))
      .filter(function (node) { return !renderedBlocks.has(node); });
    if (!nodes.length) return;

    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: currentScheme()
    });
    const work = nodes.map(function (node) {
      renderedBlocks.add(node);
      node.dataset.kbMermaidRendered = "true";
      return { node: node, source: sourceFor(node) };
    });

    for (const item of work) {
      try {
        const id = "kb-mermaid-" + (++diagramSequence);
        const result = await window.mermaid.render(id, item.source);
        item.node.innerHTML = result.svg;
        if (result.bindFunctions) result.bindFunctions(item.node);
      } catch (error) {
        item.node.dataset.kbMermaidError = "true";
        console.error(
          "Mermaid render failed: " + (error && (error.str || error.message) || String(error))
        );
      }
    }
  }

  function scheduleRender() {
    renderDiagrams(document).catch(function (error) {
      console.error("Mermaid render failed", error);
    });
  }

  captureSources(document);
  window.kbRenderDiagrams = renderDiagrams;
  document.addEventListener("DOMContentLoaded", scheduleRender);
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(scheduleRender);
  }
}());
