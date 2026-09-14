(function () {
  "use strict";
  const assetRoot = new URL(".", document.currentScript.src);
  const runtimeUrl = new URL("vendor/mathjax/tex-chtml.js", assetRoot).href;
  const newcmRoot = new URL("vendor/mathjax-newcm", assetRoot)
    .href.replace(/\/$/, "");
  const renderedContent = new WeakMap();
  let runtimePromise;

  window.MathJax = {
    loader: {
      paths: {
        "mathjax-newcm": newcmRoot
      }
    },
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: "(^| )(no-mathjax|no-math)( |$)",
    processHtmlClass: "arithmatex"
  },
  chtml: {
    displayAlign: "left"
  },
  startup: {
    typeset: false
  }
  };

  function loadRuntime() {
    if (typeof window.MathJax.typesetPromise === "function") {
      return Promise.resolve(window.MathJax);
    }
    if (runtimePromise) return runtimePromise;
    runtimePromise = new Promise(function (resolve, reject) {
      const script = document.createElement("script");
      script.src = runtimeUrl;
      script.async = true;
      script.addEventListener("load", function () {
        if (typeof window.MathJax.typesetPromise === "function") {
          resolve(window.MathJax);
        } else {
          reject(new Error("MathJax runtime loaded without typesetPromise"));
        }
      }, { once: true });
      script.addEventListener("error", function () {
        reject(new Error("MathJax runtime failed to load: " + runtimeUrl));
      }, { once: true });
      document.head.appendChild(script);
    });
    return runtimePromise;
  }

  function renderMath() {
    const content = document.querySelector("[data-md-component='content']");
    if (!content || !content.querySelector(".arithmatex")) return Promise.resolve();
    if (renderedContent.has(content)) return renderedContent.get(content);
    const rendered = loadRuntime().then(async function (mathjax) {
      if (mathjax.startup && mathjax.startup.promise) {
        await mathjax.startup.promise;
      }
      if (content.isConnected) await mathjax.typesetPromise([document.body]);
    });
    renderedContent.set(content, rendered);
    return rendered;
  }

  function scheduleRender() {
    renderMath().catch(function (error) {
      console.error("MathJax render failed", error);
    });
  }

  document.querySelectorAll(".kb-mermaid").forEach(function (block) {
    block.classList.add("no-mathjax");
  });
  window.kbRenderMath = renderMath;
  scheduleRender();
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(scheduleRender);
  }
}());
