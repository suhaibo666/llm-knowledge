import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable

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
    root = (paths.staging / "index.md").read_text(encoding="utf-8")
    root_frontmatter = yaml.safe_load(root.split("---", 2)[1])
    assert root_frontmatter["template"] == "home.html"
    assert result.asset_count == 1
    assert result.route_manifest == paths.cache / "routes.json"
    assert len(json.loads(result.route_manifest.read_text(encoding="utf-8"))) == 3


def test_stage_wiki_copies_complete_local_renderer_runtime(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(fixture_wiki, repo / "wiki")
    tooling = repo / "tools/mkdocs-site"
    renderer_files = {
        "assets/extra.js": "theme",
        "node_modules/mathjax/tex-chtml.js": "mathjax",
        "node_modules/mathjax/input/tex/extensions/boldsymbol.js": "extension",
        "node_modules/mathjax/sre/speech-worker.js": "worker",
        "node_modules/mathjax/sre/mathmaps/en.json": "{}",
        "node_modules/@mathjax/mathjax-newcm-font/chtml/dynamic/double-struck.js": "font",
        "node_modules/@mathjax/mathjax-newcm-font/chtml/woff2/mjx-ncm-ab.woff2": "font",
        "node_modules/mermaid/dist/mermaid.min.js": "mermaid",
    }
    for relative, content in renderer_files.items():
        destination = tooling / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")

    paths = BuildPaths.from_repo(repo)
    stage_wiki(paths, scan_inventory(paths.wiki))

    expected = (
        "assets/vendor/mathjax/input/tex/extensions/boldsymbol.js",
        "assets/vendor/mathjax/sre/speech-worker.js",
        "assets/vendor/mathjax/sre/mathmaps/en.json",
        "assets/vendor/mathjax-newcm/chtml/dynamic/double-struck.js",
        "assets/vendor/mathjax-newcm/chtml/woff2/mjx-ncm-ab.woff2",
    )
    for relative in expected:
        assert (paths.staging / relative).is_file(), relative


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


@pytest.mark.parametrize(
    "failed_operation",
    [
        "backup_stage",
        "backup_manifest",
        "activate_stage",
        "activate_manifest",
    ],
)
def test_replace_failure_preserves_last_good_outputs_and_cleans_transaction(
    failed_operation: str,
    tmp_path: Path,
    fixture_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(fixture_wiki, repo / "wiki")
    paths = BuildPaths.from_repo(repo)
    inventory = scan_inventory(paths.wiki)
    stage_wiki(paths, inventory)
    route_manifest = paths.cache / "routes.json"
    before_stage = tree_digest(paths.staging)
    before_routes = route_manifest.read_bytes()
    real_replace = Path.replace
    failed = False

    def is_selected(source: Path, destination: Path) -> bool:
        checks: dict[str, Callable[[], bool]] = {
            "backup_stage": lambda: source == paths.staging
            and destination.name.endswith("-old-stage"),
            "backup_manifest": lambda: source == route_manifest
            and destination.name.endswith("-old-manifest"),
            "activate_stage": lambda: source.name.startswith(".docs-")
            and not source.name.endswith("-old-stage")
            and destination == paths.staging,
            "activate_manifest": lambda: source.name.startswith(".routes-")
            and not source.name.endswith("-old-manifest")
            and destination == route_manifest,
        }
        return checks[failed_operation]()

    def fail_selected_replace(source: Path, destination: Path) -> Path:
        nonlocal failed
        destination = Path(destination)
        if not failed and is_selected(source, destination):
            failed = True
            raise OSError(f"injected {failed_operation} failure")
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_selected_replace)

    with pytest.raises(OSError, match=failed_operation):
        stage_wiki(paths, inventory)

    assert failed
    assert tree_digest(paths.staging) == before_stage
    assert route_manifest.read_bytes() == before_routes
    assert {path.name for path in paths.cache.iterdir()} == {"docs", "routes.json"}


def test_stage_rejects_user_frontmatter_reserved_key(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(fixture_wiki, repo / "wiki")
    paths = BuildPaths.from_repo(repo)
    article = paths.wiki / "domain/10_article.md"
    article.write_text(
        article.read_text(encoding="utf-8").replace(
            "title: 上下文并行\n",
            "title: 上下文并行\nmkdocs_preview:\n  owner: user\n",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match=r"domain/10_article\.md.*mkdocs_preview.*reserved"
    ):
        stage_wiki(paths, scan_inventory(paths.wiki))

    assert not paths.staging.exists()
