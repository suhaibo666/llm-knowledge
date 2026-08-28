window.MathJax = {
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

document.querySelectorAll(".mermaid").forEach(function (block) {
  block.classList.add("no-mathjax");
});
