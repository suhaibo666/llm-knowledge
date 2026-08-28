from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from .models import BuildPaths, Inventory, PageRecord
from .routes import build_route_manifest, write_route_manifest
from .wikilinks import rewrite_wikilinks


@dataclass(frozen=True)
class StageResult:
    page_count: int
    asset_count: int
    route_manifest: Path


def _is_strict_descendant(candidate: Path, root: Path) -> bool:
    if candidate == root:
        return False
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _frontmatter_and_body(
    markdown: str, source: Path
) -> tuple[dict[str, object], str, int]:
    lines = markdown.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return {}, markdown, 0
    for index in range(1, len(lines)):
        if lines[index].rstrip("\r\n") in {"---", "..."}:
            try:
                loaded = yaml.safe_load("".join(lines[1:index]))
            except yaml.YAMLError as error:
                raise ValueError(f"{source.as_posix()}: invalid frontmatter") from error
            if loaded is None:
                return {}, "".join(lines[index + 1 :]), index + 1
            if not isinstance(loaded, Mapping):
                raise ValueError(f"{source.as_posix()}: frontmatter must be a mapping")
            return dict(loaded), "".join(lines[index + 1 :]), index + 1
    return {}, markdown, 0


def _render_page(markdown: str, page: PageRecord, inventory: Inventory) -> str:
    frontmatter, body, body_line_offset = _frontmatter_and_body(
        markdown, Path(page.relative.as_posix())
    )
    if "mkdocs_preview" in frontmatter:
        raise ValueError(
            f"{page.relative.as_posix()}: frontmatter key mkdocs_preview is reserved"
        )
    frontmatter["mkdocs_preview"] = {
        "source_path": page.relative.as_posix(),
        "nav_title": page.nav_title,
        "is_index": page.is_index,
    }
    if page.relative.as_posix() == "index.md":
        frontmatter["template"] = "home.html"
    rendered_frontmatter = yaml.safe_dump(
        frontmatter, allow_unicode=True, sort_keys=False
    )
    line_prefix = "\n" * body_line_offset
    rewritten = rewrite_wikilinks(line_prefix + body, page, inventory)[
        len(line_prefix) :
    ]
    return (
        f"---\n{rendered_frontmatter}---\n"
        f"{rewritten}"
    )


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _copy_theme_assets(paths: BuildPaths, staging: Path) -> int:
    tooling = paths.repo / "tools/mkdocs-site"
    if not tooling.exists():
        return 0
    source_assets = tooling / "assets"
    if not source_assets.is_dir():
        raise FileNotFoundError(f"missing MkDocs theme assets: {source_assets}")
    target_assets = staging / "assets"
    shutil.copytree(source_assets, target_assets, dirs_exist_ok=True)
    copied = sum(1 for path in source_assets.rglob("*") if path.is_file())
    vendor_files = (
        (
            tooling / "node_modules/mathjax/tex-chtml.js",
            target_assets / "vendor/mathjax/tex-chtml.js",
        ),
        (
            tooling / "node_modules/mermaid/dist/mermaid.min.js",
            target_assets / "vendor/mermaid/mermaid.min.js",
        ),
    )
    for source, destination in vendor_files:
        if not source.is_file():
            raise FileNotFoundError(f"missing pinned renderer asset: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    vendor_trees = (
        (
            tooling / "node_modules/mathjax/input/tex/extensions",
            target_assets / "vendor/mathjax/input/tex/extensions",
        ),
        (
            tooling / "node_modules/mathjax/sre",
            target_assets / "vendor/mathjax/sre",
        ),
        (
            tooling / "node_modules/@mathjax/mathjax-newcm-font/chtml/dynamic",
            target_assets / "vendor/mathjax-newcm/chtml/dynamic",
        ),
        (
            tooling / "node_modules/@mathjax/mathjax-newcm-font/chtml/woff2",
            target_assets / "vendor/mathjax-newcm/chtml/woff2",
        ),
    )
    for source, destination in vendor_trees:
        if not source.is_dir():
            raise FileNotFoundError(f"missing pinned renderer asset tree: {source}")
        shutil.copytree(source, destination, dirs_exist_ok=True)
        copied += sum(1 for path in source.rglob("*") if path.is_file())
    return copied


def _commit_stage(
    temporary_stage: Path,
    staging: Path,
    temporary_manifest: Path,
    route_manifest: Path,
) -> None:
    backup_stage = temporary_stage.with_name(f"{temporary_stage.name}-old-stage")
    backup_manifest = temporary_manifest.with_name(
        f"{temporary_manifest.name}-old-manifest"
    )
    moved_stage = False
    moved_manifest = False
    activated_stage = False
    activated_manifest = False
    try:
        if staging.exists():
            staging.replace(backup_stage)
            moved_stage = True
        if route_manifest.exists():
            route_manifest.replace(backup_manifest)
            moved_manifest = True
        temporary_stage.replace(staging)
        activated_stage = True
        temporary_manifest.replace(route_manifest)
        activated_manifest = True
    except BaseException:
        if activated_stage:
            _remove_tree(staging)
        if activated_manifest:
            route_manifest.unlink()
        if moved_stage and backup_stage.exists():
            backup_stage.replace(staging)
        if moved_manifest and backup_manifest.exists():
            backup_manifest.replace(route_manifest)
        raise
    else:
        _remove_tree(backup_stage)
        if backup_manifest.exists():
            backup_manifest.unlink()


def stage_wiki(paths: BuildPaths, inventory: Inventory) -> StageResult:
    """Build and atomically replace a disposable Markdown staging tree."""
    resolved_cache = paths.cache.resolve()
    resolved_staging = paths.staging.resolve()
    if not _is_strict_descendant(resolved_staging, resolved_cache):
        raise ValueError(f"staging path escapes cache: {resolved_staging}")

    paths.cache.mkdir(parents=True, exist_ok=True)
    temporary_stage = Path(tempfile.mkdtemp(prefix=".docs-", dir=paths.cache))
    route_manifest = paths.generated_config.with_name("routes.json")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".routes-", suffix=".json", dir=paths.cache
    )
    os.close(file_descriptor)
    temporary_manifest = Path(temporary_name)
    asset_count = 0
    try:
        for page in inventory.pages:
            markdown = page.source.read_text(encoding="utf-8")
            destination = temporary_stage / Path(page.relative.as_posix())
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                _render_page(markdown, page, inventory), encoding="utf-8"
            )

        for source in sorted(
            (
                source
                for source in paths.wiki.rglob("*")
                if source.is_file() and source.suffix.lower() != ".md"
            ),
            key=lambda source: source.relative_to(paths.wiki).as_posix().casefold(),
        ):
            relative = source.relative_to(paths.wiki)
            destination = temporary_stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            asset_count += 1

        asset_count += _copy_theme_assets(paths, temporary_stage)

        write_route_manifest(build_route_manifest(inventory), temporary_manifest)
        _commit_stage(
            temporary_stage,
            paths.staging,
            temporary_manifest,
            route_manifest,
        )
    finally:
        _remove_tree(temporary_stage)
        if temporary_manifest.exists():
            temporary_manifest.unlink()

    return StageResult(len(inventory.pages), asset_count, route_manifest)
