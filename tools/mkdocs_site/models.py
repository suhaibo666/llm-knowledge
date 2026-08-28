from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping


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
        staging.resolve().relative_to(cache.resolve())
        site.resolve().relative_to(resolved)
        return cls(resolved, resolved / "wiki", cache, staging, generated, site)
