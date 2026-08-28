import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from tools.mkdocs_site.inventory import scan_inventory
from tools.mkdocs_site.models import BuildPaths
from tools.mkdocs_site.staging import stage_wiki
from tools.mkdocs_site.wikilinks import LinkResolutionError


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    ):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_stage_wiki_is_clean_reproducible_and_source_read_only(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(fixture_wiki, repo / "wiki")
    paths = BuildPaths.from_repo(repo)
    before = tree_digest(paths.wiki)
    inventory = scan_inventory(paths.wiki)
    first = stage_wiki(paths, inventory)
    stale = paths.staging / "stale.md"
    stale.write_text("stale", encoding="utf-8")
    second = stage_wiki(paths, inventory)
    assert not stale.exists()
    assert tree_digest(paths.wiki) == before
    assert first.page_count == second.page_count == 3
    assert (paths.staging / "domain/figure.svg").is_file()


def test_stage_wiki_preserves_frontmatter_and_writes_routes(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(fixture_wiki, repo / "wiki")
    paths = BuildPaths.from_repo(repo)

    result = stage_wiki(paths, scan_inventory(paths.wiki))

    staged = (paths.staging / "domain/10_article.md").read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(staged.split("---", 2)[1])
    assert frontmatter == {
        "title": "上下文并行",
        "mkdocs_preview": {
            "source_path": "domain/10_article.md",
            "nav_title": "10_article",
            "is_index": False,
        },
    }
    assert result.asset_count == 1
    assert result.route_manifest == paths.cache / "routes.json"
    assert len(json.loads(result.route_manifest.read_text(encoding="utf-8"))) == 3


def test_failed_conversion_keeps_last_good_stage_and_reports_source_line(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(fixture_wiki, repo / "wiki")
    paths = BuildPaths.from_repo(repo)
    stage_wiki(paths, scan_inventory(paths.wiki))
    before_stage = tree_digest(paths.staging)
    before_routes = (paths.cache / "routes.json").read_bytes()
    article = paths.wiki / "domain/10_article.md"
    article.write_text(
        article.read_text(encoding="utf-8") + "\n[[missing]]\n", encoding="utf-8"
    )

    with pytest.raises(
        LinkResolutionError,
        match=r"domain/10_article\.md:11: \[\[missing\]\]: broken target",
    ):
        stage_wiki(paths, scan_inventory(paths.wiki))

    assert tree_digest(paths.staging) == before_stage
    assert (paths.cache / "routes.json").read_bytes() == before_routes
