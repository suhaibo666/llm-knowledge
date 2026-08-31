from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml
from bs4 import BeautifulSoup

from tools.mkdocs_site.config import write_generated_config
from tools.mkdocs_site.inventory import scan_inventory
from tools.mkdocs_site.models import BuildPaths
from tools.mkdocs_site.staging import stage_wiki


REPO = Path(__file__).resolve().parents[3]
RENDERER_CONTRACT_BODY = r"""

> [!note] Blockquoted display math must stay in the corpus gate
> $$
> \boldsymbol{\theta} \in \mathbb{R}^{d \times k}, \qquad
> \mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^{n} \left\lVert x_i - \boldsymbol{\theta} \right\rVert_2^2
> $$

```mermaid
flowchart LR
  A["<img src=x onerror=window.__kbSecurityProbe=true>"] --> B[安全]
  click A "javascript:window.__kbSecurityProbe=true"
```

```mermaid
flowchart TD
  A -->
```
"""


def test_material_palette_cycles_system_light_dark_with_chinese_labels() -> None:
    config = yaml.safe_load((REPO / "mkdocs.yml").read_text(encoding="utf-8"))

    assert config["theme"]["palette"] == [
        {
            "media": "(prefers-color-scheme)",
            "toggle": {
                "icon": "material/brightness-auto",
                "name": "切换至浅色模式",
            },
        },
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "toggle": {
                "icon": "material/weather-night",
                "name": "切换至深色模式",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "toggle": {
                "icon": "material/brightness-auto",
                "name": "跟随系统主题",
            },
        },
    ]


def _copy_theme_inputs(repo: Path) -> None:
    source_root = REPO / "tools/mkdocs-site"
    target_root = repo / "tools/mkdocs-site"
    for directory in ("assets", "overrides"):
        source = source_root / directory
        if source.is_dir():
            shutil.copytree(source, target_root / directory)
    for relative in (
        Path("node_modules/mathjax/tex-chtml.js"),
        Path("node_modules/mermaid/dist/mermaid.min.js"),
    ):
        source = source_root / relative
        if source.is_file():
            target = target_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for relative in (
        Path("node_modules/mathjax/input/tex/extensions"),
        Path("node_modules/mathjax/sre"),
        Path("node_modules/@mathjax/mathjax-newcm-font/chtml/dynamic"),
        Path("node_modules/@mathjax/mathjax-newcm-font/chtml/woff2"),
    ):
        source = source_root / relative
        if source.is_dir():
            shutil.copytree(source, target_root / relative)


def build_fixture_site(
    tmp_path: Path,
    fixture_wiki: Path,
    *,
    renderer_contract: bool = False,
) -> tuple[Path, object]:
    repo = tmp_path / "repo"
    shutil.copytree(fixture_wiki, repo / "wiki")
    shutil.copy2(REPO / "mkdocs.yml", repo / "mkdocs.yml")
    if renderer_contract:
        article = repo / "wiki/domain/10_article.md"
        article.write_text(
            article.read_text(encoding="utf-8") + RENDERER_CONTRACT_BODY,
            encoding="utf-8",
        )
        base_config = yaml.safe_load((repo / "mkdocs.yml").read_text(encoding="utf-8"))
        base_config.pop("repo_url", None)
        base_config.pop("edit_uri", None)
        (repo / "mkdocs.yml").write_text(
            yaml.safe_dump(base_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    _copy_theme_inputs(repo)
    paths = BuildPaths.from_repo(repo)
    inventory = scan_inventory(paths.wiki)
    staged = stage_wiki(paths, inventory)
    generated = write_generated_config(paths, inventory, staged)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "-f",
            str(generated),
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return paths.site, inventory


def test_rendered_theme_separates_navigation_document_and_index_titles(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    site, _ = build_fixture_site(tmp_path, fixture_wiki)
    article = BeautifulSoup(
        (site / "domain/10_article.html").read_text(encoding="utf-8"),
        "html.parser",
    )
    home = BeautifulSoup(
        (site / "index.html").read_text(encoding="utf-8"), "html.parser"
    )

    assert article.select_one(".md-nav__link[data-nav-title='10_article']")
    assert article.select_one("h1").get_text(strip=True) == "上下文并行"
    assert article.select_one(".kb-file-path").get_text(strip=True).endswith(
        "10_article.md"
    )
    assert article.title.get_text(strip=True).startswith("上下文并行")
    assert home.select_one(".kb-source-atlas")
    assert not home.select_one("body.kb-page-index .md-sidebar--secondary")


def test_search_ui_keeps_ctrl_k_hint_without_unsafe_share_href(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    site, _ = build_fixture_site(tmp_path, fixture_wiki)
    home = BeautifulSoup(
        (site / "index.html").read_text(encoding="utf-8"), "html.parser"
    )

    assert home.select_one("input[data-md-component='search-query']")
    assert home.select_one("[data-kb-search-trigger] kbd").get_text(" ", strip=True) == "Ctrl K"
    assert not home.select("[href^='javascript:']")


def test_native_search_index_keeps_chinese_title_and_filename_tag(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    site, _ = build_fixture_site(tmp_path, fixture_wiki)
    payload = json.loads(
        (site / "search/search_index.json").read_text(encoding="utf-8")
    )
    article = next(
        document
        for document in payload["docs"]
        if document["location"] == "domain/10_article.html"
    )

    assert payload["config"]["pipeline"] == []
    assert article["title"] == "上下文并行"
    assert article["tags"] == ["10_article"]


def test_search_rewrite_keeps_chinese_title_and_indexes_filename_once(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    site, inventory = build_fixture_site(tmp_path, fixture_wiki)
    from tools.mkdocs_site.search_index import rewrite_search_index

    rewrite_search_index(site, inventory)

    payload = json.loads(
        (site / "search/search_index.json").read_text(encoding="utf-8")
    )
    article = next(
        document
        for document in payload["docs"]
        if document["location"].split("#", 1)[0] == "domain/10_article.html"
    )
    assert article["title"] == "上下文并行"
    assert article["text"].split().count("10_article") == 1


def test_search_rewrite_rejects_locations_outside_the_route_manifest(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    site = tmp_path / "site"
    search = site / "search"
    search.mkdir(parents=True)
    (search / "search_index.json").write_text(
        json.dumps(
            {
                "config": {},
                "docs": [
                    {"location": "missing.html", "title": "missing", "text": "body"}
                ],
            }
        ),
        encoding="utf-8",
    )
    from tools.mkdocs_site.search_index import SearchIndexError, rewrite_search_index

    with pytest.raises(SearchIndexError, match=r"unknown search location: missing\.html"):
        rewrite_search_index(site, scan_inventory(fixture_wiki))


def test_local_renderer_runtime_assets_are_staged(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    site, _ = build_fixture_site(tmp_path, fixture_wiki)

    required_runtime_files = (
        "assets/vendor/mathjax/input/tex/extensions/boldsymbol.js",
        "assets/vendor/mathjax/sre/speech-worker.js",
        "assets/vendor/mathjax/sre/mathmaps/en.json",
        "assets/vendor/mathjax-newcm/chtml/dynamic/double-struck.js",
        "assets/vendor/mathjax-newcm/chtml/woff2/mjx-ncm-ab.woff2",
    )
    for relative in required_runtime_files:
        assert (site / relative).is_file(), relative


def _run_renderer_contract(
    site: Path,
    *,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "node",
            str(REPO / "tools/mkdocs_site/tests/renderer_contract.mjs"),
            str(site),
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _run_mathjax_corpus(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "node",
            str(REPO / "tools/mkdocs-site/mathjax-corpus.mjs"),
            "--repo-root",
            str(repo),
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )


class _QuietHttpHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *args: object) -> None:
        pass


def _serve_directory(directory: Path) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(_QuietHttpHandler, directory=str(directory)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _inject_article_script(site: Path, source: str) -> None:
    article = site / "domain/10_article.html"
    html = article.read_text(encoding="utf-8")
    assert "</body>" in html
    article.write_text(
        html.replace("</body>", f'<script src="{source}"></script></body>', 1),
        encoding="utf-8",
    )


def test_renderer_contract_fails_fast_without_owned_puppeteer(tmp_path: Path) -> None:
    isolated_repo = tmp_path / "isolated-repo"
    isolated_tests = isolated_repo / "tools/mkdocs_site/tests"
    isolated_docs_site = isolated_repo / "tools/docs-site"
    isolated_tests.mkdir(parents=True)
    isolated_docs_site.mkdir(parents=True)
    isolated_script = isolated_tests / "renderer_contract.mjs"
    shutil.copy2(
        REPO / "tools/mkdocs_site/tests/renderer_contract.mjs",
        isolated_script,
    )
    shutil.copy2(
        REPO / "tools/docs-site/listeners.mjs",
        isolated_docs_site / "listeners.mjs",
    )
    site = tmp_path / "site"
    site.mkdir()
    before = tuple(isolated_repo.rglob("*"))

    started = time.monotonic()
    completed = subprocess.run(
        ["node", str(isolated_script), str(site)],
        cwd=isolated_repo,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    elapsed = time.monotonic() - started

    assert completed.returncode != 0
    assert elapsed < 5
    assert "puppeteer-core is not installed" in completed.stderr
    assert "npm ci --prefix tools/mkdocs-site" in completed.stderr
    assert tuple(isolated_repo.rglob("*")) == before


def test_renderer_contract_rejects_internal_http_error(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    site, _ = build_fixture_site(tmp_path, fixture_wiki, renderer_contract=True)
    article = site / "domain/10_article.html"
    article.write_text(
        article.read_text(encoding="utf-8").replace(
            "</body>",
            '<script>fetch("contract-missing.json")</script></body>',
        ),
        encoding="utf-8",
    )

    completed = _run_renderer_contract(site)

    assert completed.returncode != 0
    assert "HTTP error responses" in completed.stderr
    assert "404" in completed.stderr


def test_renderer_runtime_in_browser_at_root_and_project_subpath(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    site, _ = build_fixture_site(
        tmp_path, fixture_wiki, renderer_contract=True
    )
    completed = _run_renderer_contract(site)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert [case["basePath"] for case in result["cases"]] == [
        "/",
        "/llm-knowledge/",
    ]
    assert [case["math"] for case in result["cases"]] == [1, 1]


def test_mathjax_corpus_discovers_and_renders_blockquoted_display_math(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    site, _ = build_fixture_site(
        tmp_path, fixture_wiki, renderer_contract=True
    )
    article = BeautifulSoup(
        (site / "domain/10_article.html").read_text(encoding="utf-8"),
        "html.parser",
    )
    assert not article.select(".arithmatex")
    assert "$$" in article.get_text()

    completed = _run_mathjax_corpus(site.parent)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS: 1 formulas across 1 pages" in completed.stdout


def test_mathjax_corpus_rejects_runtime_from_alternate_loopback_origin(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    site, _ = build_fixture_site(
        tmp_path, fixture_wiki, renderer_contract=True
    )
    runtime = site / "assets/alternate-runtime.js"
    runtime.write_text("window.__alternateRuntimeLoaded = true;\n", encoding="utf-8")
    server, thread = _serve_directory(site)
    try:
        port = server.server_address[1]
        _inject_article_script(
            site,
            f"http://127.0.0.1:{port}/assets/alternate-runtime.js",
        )
        completed = _run_mathjax_corpus(site.parent)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert completed.returncode != 0
    assert "requested external runtime assets" in completed.stderr


def test_mathjax_corpus_rejects_missing_runtime_asset(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    site, _ = build_fixture_site(
        tmp_path, fixture_wiki, renderer_contract=True
    )
    _inject_article_script(site, "/llm-knowledge/assets/missing-runtime.js")

    completed = _run_mathjax_corpus(site.parent)

    assert completed.returncode != 0
    assert "failed asset requests" in completed.stderr
    assert "404" in completed.stderr


def test_mathjax_corpus_rejects_visible_mjx_merror(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    site, _ = build_fixture_site(
        tmp_path, fixture_wiki, renderer_contract=True
    )
    article = site / "domain/10_article.html"
    html = article.read_text(encoding="utf-8")
    assert r"\boldsymbol{\theta}" in html
    article.write_text(
        html.replace(
            r"\boldsymbol{\theta}",
            r"\begin{aligned}\tag{bad}x&=y\end{aligned}",
            1,
        ),
        encoding="utf-8",
    )

    completed = _run_mathjax_corpus(site.parent)

    assert completed.returncode != 0
    assert "tag" in completed.stderr.lower()


def test_mkdocs_aggregate_runs_mathjax_corpus_gate() -> None:
    package = json.loads((REPO / "package.json").read_text(encoding="utf-8"))

    assert "tools/mkdocs-site/mathjax-corpus.mjs" in package["scripts"]["docs:mkdocs:test"]
