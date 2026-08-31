from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping


def _is_strict_descendant(candidate: Path, root: Path) -> bool:
    if candidate == root:
        return False
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class PageRecord:
    source: Path
    relative: PurePosixPath
    title: str
    nav_title: str
    is_index: bool
    headings: tuple[str, ...]


@dataclass(frozen=True)
class RouteRecord:
    source: PurePosixPath
    output: PurePosixPath
    public_urls: tuple[str, ...]


@dataclass(frozen=True)
class Inventory:
    pages: tuple[PageRecord, ...]
    by_relative: Mapping[PurePosixPath, PageRecord]
    by_stem: Mapping[str, tuple[PageRecord, ...]]

    @classmethod
    def from_pages(cls, pages: tuple[PageRecord, ...]) -> "Inventory":
        ordered = tuple(sorted(pages, key=lambda page: page.relative.as_posix().casefold()))
        by_relative: dict[PurePosixPath, PageRecord] = {}
        by_stem_mutable: dict[str, list[PageRecord]] = defaultdict(list)
        for page in ordered:
            key = page.relative.with_suffix("")
            if key in by_relative:
                raise ValueError(f"duplicate page path: {key}")
            by_relative[key] = page
            by_stem_mutable[page.source.stem].append(page)
        by_stem = {key: tuple(value) for key, value in by_stem_mutable.items()}
        return cls(ordered, MappingProxyType(by_relative), MappingProxyType(by_stem))


@dataclass(frozen=True)
class BuildPaths:
    repo: Path
    wiki: Path
    cache: Path
    staging: Path
    generated_config: Path
    site: Path

    @classmethod
    def from_repo(cls, repo: Path) -> "BuildPaths":
        resolved = repo.resolve()
        cache = resolved / ".mkdocs-cache"
        staging = cache / "docs"
        generated = cache / "mkdocs.generated.yml"
        site = resolved / "site"
        resolved_cache = cache.resolve()
        resolved_staging = staging.resolve()
        resolved_generated = generated.resolve()
        resolved_site = site.resolve()
        if not _is_strict_descendant(resolved_cache, resolved):
            raise ValueError(f"cache path escapes repository: {resolved_cache}")
        if not _is_strict_descendant(resolved_staging, resolved_cache):
            raise ValueError(f"staging path escapes cache: {resolved_staging}")
        if not _is_strict_descendant(resolved_generated, resolved_cache):
            raise ValueError(f"generated config escapes cache: {resolved_generated}")
        if not _is_strict_descendant(resolved_site, resolved):
            raise ValueError(f"site path escapes repository: {resolved_site}")
        return cls(resolved, resolved / "wiki", cache, staging, generated, site)
