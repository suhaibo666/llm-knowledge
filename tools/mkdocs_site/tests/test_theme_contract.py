from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from tools.mkdocs_site.config import write_generated_config
from tools.mkdocs_site.inventory import scan_inventory
from tools.mkdocs_site.models import BuildPaths
from tools.mkdocs_site.staging import stage_wiki


REPO = Path(__file__).resolve().parents[3]


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


def build_fixture_site(tmp_path: Path, fixture_wiki: Path) -> tuple[Path, object]:
    repo = tmp_path / "repo"
    shutil.copytree(fixture_wiki, repo / "wiki")
    shutil.copy2(REPO / "mkdocs.yml", repo / "mkdocs.yml")
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
