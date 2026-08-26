(function exposeFlowDiagram(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  root.DeepseekFlowDiagram = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function makeApi() {
  "use strict";

  function clampScale(value, minScale, maxScale) {
    return Math.min(maxScale, Math.max(minScale, value));
  }

  function zoomAround(transform, pointer, requestedScale, minScale, maxScale) {
    const nextScale = clampScale(requestedScale, minScale, maxScale);
    const worldX = (pointer.x - transform.x) / transform.scale;
    const worldY = (pointer.y - transform.y) / transform.scale;
    return {
      x: pointer.x - worldX * nextScale,
      y: pointer.y - worldY * nextScale,
      scale: nextScale,
    };
  }

  function panBy(transform, dx, dy) {
    return {
      x: transform.x + dx,
      y: transform.y + dy,
      scale: transform.scale,
    };
  }

  function fitViewBox(viewport, content, padding, minScale, maxScale) {
    const availableWidth = Math.max(1, viewport.width - padding * 2);
    const availableHeight = Math.max(1, viewport.height - padding * 2);
    const requestedScale = Math.min(
      availableWidth / content.width,
      availableHeight / content.height,
    );
    const scale = clampScale(requestedScale, minScale, maxScale);
    return {
      x: (viewport.width - content.width * scale) / 2,
      y: (viewport.height - content.height * scale) / 2,
      scale,
    };
  }

  function createViewportState(initialTransform, limits) {
    let transform = { ...initialTransform };
    return {
      get() {
        return { ...transform };
      },
      pan(dx, dy) {
        transform = panBy(transform, dx, dy);
        return { ...transform };
      },
      reset(nextTransform) {
        transform = { ...nextTransform };
        return { ...transform };
      },
      zoomAt(pointer, factor) {
        transform = zoomAround(
          transform,
          pointer,
          transform.scale * factor,
          limits.minScale,
          limits.maxScale,
        );
        return { ...transform };
      },
    };
  }

  function setViewVisibility(views, activeName) {
    for (const view of views) {
      view.toggleAttribute("hidden", view.dataset.view !== activeName);
    }
  }

  function toTransform(transform) {
    return `translate(${transform.x.toFixed(2)} ${transform.y.toFixed(2)}) scale(${transform.scale.toFixed(4)})`;
  }

  return {
    clampScale,
    createViewportState,
    fitViewBox,
    panBy,
    setViewVisibility,
    toTransform,
    zoomAround,
  };
});
