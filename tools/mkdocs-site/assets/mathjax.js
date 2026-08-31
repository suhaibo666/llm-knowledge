(function () {
  "use strict";
  const assetRoot = new URL(".", document.currentScript.src);
  const newcmRoot = new URL("vendor/mathjax-newcm", assetRoot)
    .href.replace(/\/$/, "");

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
  }
  };
}());

document.querySelectorAll(".mermaid").forEach(function (block) {
  block.classList.add("no-mathjax");
});
