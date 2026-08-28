(function () {
  "use strict";

  function label(selector, text) {
    document.querySelectorAll(selector).forEach(function (element) {
      element.setAttribute("aria-label", text);
    });
  }

  function enhancePage() {
    document.body.classList.toggle(
      "kb-page-index",
      Boolean(document.querySelector("[data-kb-page-kind='index']"))
    );

    document.querySelectorAll(".md-nav__link[data-nav-title]").forEach(function (link) {
      link.title = link.dataset.navTitle;
      if (!link.getAttribute("aria-label")) link.setAttribute("aria-label", link.dataset.navTitle);
    });
    document.querySelectorAll(".md-nav__item--active").forEach(function (item) {
      let current = item;
      while (current) {
        if (current.classList && current.classList.contains("md-nav__item")) {
          current.classList.add("kb-active-path");
        }
        current = current.parentElement && current.parentElement.closest(".md-nav__item");
      }
    });

    label("[data-md-type='navigation'] nav", "知识导航");
    label("[data-md-type='toc'] nav", "本页目录");
    label("input[data-md-component='search-query']", "搜索知识库");
    label("label[for='__drawer']", "打开知识导航");
    label("label[for='__search']", "打开搜索");

    document.querySelectorAll("[data-kb-search-trigger]").forEach(function (trigger) {
      if (trigger.dataset.kbKeyboardReady === "true") return;
      trigger.dataset.kbKeyboardReady = "true";
      trigger.addEventListener("keydown", function (event) {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        trigger.click();
      });
    });

    if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
      const content = document.querySelector("[data-md-component='content']");
      if (content) window.MathJax.typesetPromise([content]);
    }
  }

  document.addEventListener("DOMContentLoaded", enhancePage);
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(enhancePage);
  }
}());
