(function () {
  "use strict";
  const renderedBlocks = new WeakSet();
  const diagramSources = new WeakMap();
  let diagramSequence = 0;
  let viewer;
  let viewerCanvas;
  let activeDiagram;
  let activeSvg;
  let activeNextSibling;
  let activeTrigger;
  let viewerStage;
  let viewerStatus;
  let viewState = { scale: 1, x: 0, y: 0, mode: "actual" };
  const minScale = 0.08;
  const maxScale = 8;

  function clampScale(scale) {
    return Math.min(maxScale, Math.max(minScale, scale));
  }

  function applyViewState(next) {
    viewState = next;
    viewer.dataset.kbScale = String(next.scale);
    viewer.dataset.kbX = String(next.x);
    viewer.dataset.kbY = String(next.y);
    viewer.dataset.kbMode = next.mode;
    viewerCanvas.style.transform = (
      "translate(" + next.x + "px, " + next.y + "px) scale(" + next.scale + ")"
    );
    if (viewerStatus) viewerStatus.textContent = Math.round(next.scale * 100) + "%";
  }

  function activeDiagramSize() {
    if (!activeSvg) return { width: 1, height: 1 };
    const viewBox = activeSvg.viewBox && activeSvg.viewBox.baseVal;
    if (viewBox && viewBox.width > 0 && viewBox.height > 0) {
      return { width: viewBox.width, height: viewBox.height };
    }
    const box = activeSvg.getBBox();
    return {
      width: Math.max(1, box.width),
      height: Math.max(1, box.height),
    };
  }

  function centeredState(scale, mode) {
    const stageBox = viewerStage.getBoundingClientRect();
    const diagram = activeDiagramSize();
    return {
      scale: scale,
      x: (stageBox.width - diagram.width * scale) / 2,
      y: (stageBox.height - diagram.height * scale) / 2,
      mode: mode,
    };
  }

  function fitViewer() {
    if (!activeSvg) return;
    const stageBox = viewerStage.getBoundingClientRect();
    const diagram = activeDiagramSize();
    const padding = Math.min(48, Math.max(20, stageBox.width * 0.06));
    const scale = clampScale(Math.min(
      Math.max(1, stageBox.width - padding * 2) / diagram.width,
      Math.max(1, stageBox.height - padding * 2) / diagram.height,
    ));
    applyViewState(centeredState(scale, "fit"));
  }

  function resetViewer() {
    if (!activeSvg) return;
    applyViewState(centeredState(1, "actual"));
  }

  function zoomViewer(factor, clientX, clientY) {
    if (!activeSvg) return;
    const stageBox = viewerStage.getBoundingClientRect();
    const anchorX = (clientX === undefined ? stageBox.left + stageBox.width / 2 : clientX)
      - stageBox.left;
    const anchorY = (clientY === undefined ? stageBox.top + stageBox.height / 2 : clientY)
      - stageBox.top;
    const scale = clampScale(viewState.scale * factor);
    if (scale === viewState.scale) return;
    applyViewState({
      scale: scale,
      x: anchorX - ((anchorX - viewState.x) / viewState.scale) * scale,
      y: anchorY - ((anchorY - viewState.y) / viewState.scale) * scale,
      mode: "manual",
    });
  }

  function restoreActiveDiagram() {
    if (activeDiagram && activeSvg) {
      const sibling = activeNextSibling && activeNextSibling.parentElement === activeDiagram
        ? activeNextSibling
        : null;
      activeDiagram.insertBefore(activeSvg, sibling);
    }
    if (viewerCanvas) viewerCanvas.replaceChildren();
    if (viewerCanvas) {
      viewerCanvas.style.width = "";
      viewerCanvas.style.height = "";
      viewerCanvas.style.transform = "";
    }
    activeDiagram = null;
    activeSvg = null;
    activeNextSibling = null;
    const trigger = activeTrigger;
    activeTrigger = null;
    document.body.classList.remove("kb-mermaid-viewer-open");
    if (trigger && trigger.isConnected) trigger.focus();
  }

  function closeViewer() {
    if (viewer && viewer.open) {
      document.body.classList.remove("kb-mermaid-viewer-open");
      viewer.close();
    }
  }

  function ensureViewer() {
    if (viewer && viewer.isConnected) return viewer;
    viewer = document.createElement("dialog");
    viewer.className = "kb-mermaid-viewer";
    viewer.setAttribute("aria-modal", "true");
    viewer.setAttribute("aria-label", "Mermaid 图表查看器");
    viewer.innerHTML = [
      '<div class="kb-mermaid-viewer__surface">',
      '  <div class="kb-mermaid-viewer__toolbar">',
      '    <strong>图表查看器</strong>',
      '    <div class="kb-mermaid-viewer__actions">',
      '      <button type="button" data-kb-mermaid-action="zoom-out" aria-label="缩小图表">−</button>',
      '      <button type="button" data-kb-mermaid-action="zoom-in" aria-label="放大图表">+</button>',
      '      <button type="button" data-kb-mermaid-action="fit">适应窗口</button>',
      '      <button type="button" data-kb-mermaid-action="reset">100%</button>',
      '      <output class="kb-mermaid-viewer__status" aria-live="polite">100%</output>',
      "    </div>",
      '    <button type="button" data-kb-mermaid-action="close" aria-label="关闭大图">×</button>',
      "  </div>",
      '  <div class="kb-mermaid-viewer__stage" tabindex="0">',
      '    <div class="kb-mermaid-viewer__canvas"></div>',
      "  </div>",
      "</div>",
    ].join("");
    viewerCanvas = viewer.querySelector(".kb-mermaid-viewer__canvas");
    viewerStage = viewer.querySelector(".kb-mermaid-viewer__stage");
    viewerStatus = viewer.querySelector(".kb-mermaid-viewer__status");
    viewer.querySelector(".kb-mermaid-viewer__toolbar").addEventListener(
      "click",
      function (event) {
        const button = event.target.closest("[data-kb-mermaid-action]");
        if (!button) return;
        const action = button.dataset.kbMermaidAction;
        if (action === "zoom-out") zoomViewer(1 / 1.2);
        if (action === "zoom-in") zoomViewer(1.2);
        if (action === "fit") fitViewer();
        if (action === "reset") resetViewer();
        if (action === "close") closeViewer();
      },
    );
    viewerStage.addEventListener("wheel", function (event) {
      event.preventDefault();
      zoomViewer(event.deltaY < 0 ? 1.18 : 1 / 1.18, event.clientX, event.clientY);
    }, { passive: false });
    const pointers = new Map();
    let gesture;

    function pointerPoint(event) {
      return { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
    }

    function beginGesture() {
      const points = Array.from(pointers.values());
      if (points.length >= 2) {
        const first = points[0];
        const second = points[1];
        const stageBox = viewerStage.getBoundingClientRect();
        gesture = {
          type: "pinch",
          distance: Math.max(1, Math.hypot(second.x - first.x, second.y - first.y)),
          centerX: (first.x + second.x) / 2 - stageBox.left,
          centerY: (first.y + second.y) / 2 - stageBox.top,
          scale: viewState.scale,
          x: viewState.x,
          y: viewState.y,
        };
        return;
      }
      if (points.length === 1) {
        gesture = {
          type: "pan",
          pointerId: points[0].pointerId,
          clientX: points[0].x,
          clientY: points[0].y,
          x: viewState.x,
          y: viewState.y,
        };
        return;
      }
      gesture = null;
    }

    viewerStage.addEventListener("pointerdown", function (event) {
      if (event.pointerType === "mouse" && event.button !== 0) return;
      event.preventDefault();
      pointers.set(event.pointerId, pointerPoint(event));
      try {
        viewerStage.setPointerCapture(event.pointerId);
      } catch {
        // Synthetic and interrupted touch streams may not expose capture.
      }
      beginGesture();
      viewerStage.classList.add("is-dragging");
    });
    viewerStage.addEventListener("pointermove", function (event) {
      if (!pointers.has(event.pointerId)) return;
      event.preventDefault();
      pointers.set(event.pointerId, pointerPoint(event));
      const points = Array.from(pointers.values());
      if (points.length >= 2) {
        if (!gesture || gesture.type !== "pinch") beginGesture();
        const first = points[0];
        const second = points[1];
        const stageBox = viewerStage.getBoundingClientRect();
        const centerX = (first.x + second.x) / 2 - stageBox.left;
        const centerY = (first.y + second.y) / 2 - stageBox.top;
        const distance = Math.max(1, Math.hypot(second.x - first.x, second.y - first.y));
        const scale = clampScale(gesture.scale * distance / gesture.distance);
        const localX = (gesture.centerX - gesture.x) / gesture.scale;
        const localY = (gesture.centerY - gesture.y) / gesture.scale;
        applyViewState({
          scale: scale,
          x: centerX - localX * scale,
          y: centerY - localY * scale,
          mode: "manual",
        });
        return;
      }
      if (!gesture || gesture.type !== "pan" || gesture.pointerId !== event.pointerId) {
        beginGesture();
      }
      applyViewState({
        scale: viewState.scale,
        x: gesture.x + event.clientX - gesture.clientX,
        y: gesture.y + event.clientY - gesture.clientY,
        mode: "manual",
      });
    });
    function endPointer(event) {
      if (!pointers.has(event.pointerId)) return;
      pointers.delete(event.pointerId);
      beginGesture();
      if (!pointers.size) viewerStage.classList.remove("is-dragging");
    }
    viewerStage.addEventListener("pointerup", endPointer);
    viewerStage.addEventListener("pointercancel", endPointer);
    viewer.addEventListener("cancel", function (event) {
      event.preventDefault();
      closeViewer();
    });
    viewer.addEventListener("close", restoreActiveDiagram);
    viewer.addEventListener("click", function (event) {
      if (event.target === viewer) closeViewer();
    });
    document.body.appendChild(viewer);
    return viewer;
  }

  function openViewer(diagram, trigger) {
    const svg = Array.from(diagram.children).find(function (child) {
      return child.tagName.toLowerCase() === "svg";
    });
    if (!svg) return;
    closeViewer();
    const dialog = ensureViewer();
    activeDiagram = diagram;
    activeSvg = svg;
    activeNextSibling = svg.nextSibling;
    activeTrigger = trigger;
    const diagramSize = activeDiagramSize();
    viewerCanvas.style.width = diagramSize.width + "px";
    viewerCanvas.style.height = diagramSize.height + "px";
    viewerCanvas.replaceChildren(svg);
    document.body.classList.add("kb-mermaid-viewer-open");
    dialog.showModal();
    fitViewer();
  }

  function enhanceDiagram(diagram) {
    if (diagram.dataset.kbMermaidViewerReady === "true") return;
    if (diagram.dataset.kbMermaidError === "true") return;
    if (!Array.from(diagram.children).some(function (child) {
      return child.tagName.toLowerCase() === "svg";
    })) return;
    diagram.dataset.kbMermaidViewerReady = "true";
    diagram.classList.add("kb-mermaid-interactive");
    const trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "kb-mermaid-zoom-trigger";
    trigger.setAttribute("aria-label", "查看大图");
    trigger.title = "查看大图";
    trigger.innerHTML = '<span aria-hidden="true">⛶</span>';
    trigger.addEventListener("click", function () {
      openViewer(diagram, trigger);
    });
    diagram.appendChild(trigger);
  }

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

  function removeRenderArtifact(id) {
    const artifact = document.getElementById("d" + id);
    if (artifact) artifact.remove();
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
      theme: currentScheme(),
      htmlLabels: false,
      flowchart: {
        htmlLabels: false
      }
    });
    const work = nodes.map(function (node) {
      renderedBlocks.add(node);
      node.dataset.kbMermaidRendered = "true";
      return { node: node, source: sourceFor(node) };
    });

    for (const item of work) {
      const id = "kb-mermaid-" + (++diagramSequence);
      try {
        const result = await window.mermaid.render(id, item.source);
        const replacement = item.node.cloneNode(false);
        replacement.innerHTML = result.svg;
        item.node.replaceWith(replacement);
        item.node = replacement;
        renderedBlocks.add(replacement);
        diagramSources.set(replacement, item.source);
        if (result.bindFunctions) result.bindFunctions(item.node);
        enhanceDiagram(replacement);
      } catch (error) {
        removeRenderArtifact(id);
        item.node.textContent = item.source;
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
