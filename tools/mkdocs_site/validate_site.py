from __future__ import annotations

import html
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup


@dataclass(frozen=True)
class ValidationReport:
    pages: int
    broken_links: tuple[str, ...]
    missing_anchors: tuple[str, ...]
    missing_assets: tuple[str, ...]
    missing_legacy_routes: tuple[str, ...]
    orphans: tuple[str, ...]


class SiteValidationError(RuntimeError):
    """Raised after a complete report finds invalid generated-site references."""


@dataclass(frozen=True)
class _Route:
    source: PurePosixPath
    output: PurePosixPath
    public_urls: tuple[str, ...]


@dataclass(frozen=True)
class _Target:
    display: str
    candidates: tuple[PurePosixPath, ...]
    fragment: str | None
    path: PurePosixPath
    unsafe: bool = False
    ignored: bool = False


_IGNORED_SCHEMES = frozenset({"data", "http", "https", "mailto", "tel"})
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9a-fA-F]{2})")
_ASSET_LINK_RELS = frozenset(
    {
        "apple-touch-icon",
        "icon",
        "manifest",
        "modulepreload",
        "preload",
        "stylesheet",
    }
)
_PAGE_SUFFIXES = frozenset({"", ".htm", ".html", ".md"})
_AUXILIARY_HTML = frozenset({PurePosixPath("404.html")})
_REPORT_FIELDS = (
    "broken_links",
    "missing_anchors",
    "missing_assets",
    "missing_legacy_routes",
    "orphans",
)


def _manifest_path(value: object, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"route {field} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or any(part in {"", ".."} for part in path.parts):
        raise ValueError(f"route {field} must stay inside the site root: {value}")
    return path


def _load_routes(route_manifest: Path) -> tuple[_Route, ...]:
    try:
        payload = json.loads(route_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read route manifest: {route_manifest}") from error
    if not isinstance(payload, list):
        raise ValueError("route manifest must contain a list")

    routes: list[_Route] = []
    outputs: set[PurePosixPath] = set()
    public_urls: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("route manifest entries must be objects")
        source = _manifest_path(item.get("source"), "source")
        output = _manifest_path(item.get("output"), "output")
        raw_urls = item.get("public_urls")
        if not isinstance(raw_urls, list) or not all(
            isinstance(url, str) and url for url in raw_urls
        ):
            raise ValueError(f"route public_urls must be non-empty strings: {source}")
        if output in outputs:
            raise ValueError(f"duplicate route output: {output.as_posix()}")
        for url in raw_urls:
            if url in public_urls:
                raise ValueError(f"duplicate public URL: {url}")
            public_urls.add(url)
        outputs.add(output)
        routes.append(_Route(source, output, tuple(raw_urls)))
    return tuple(sorted(routes, key=lambda route: route.source.as_posix()))


def _project_path(path: str, project_prefix: str) -> tuple[str, bool]:
    absolute = path.startswith("/")
    if not absolute:
        return path, False
    normalized_prefix = "/" + project_prefix.strip("/")
    if path == normalized_prefix:
        return "", True
    prefix = normalized_prefix + "/"
    if path.startswith(prefix):
        return path[len(prefix) :], True
    return path.lstrip("/"), True


def _normalized_path(
    path: str, source: PurePosixPath, absolute: bool
) -> tuple[PurePosixPath, bool]:
    parts = [] if absolute else list(source.parent.parts)
    for part in path.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                return PurePosixPath("..") / PurePosixPath(path), False
            parts.pop()
            continue
        parts.append(part)
    return PurePosixPath(*parts) if parts else PurePosixPath("."), True


def _candidate_paths(path: PurePosixPath, trailing_slash: bool) -> tuple[PurePosixPath, ...]:
    if path == PurePosixPath("."):
        return (PurePosixPath("index.html"),)
    if trailing_slash:
        return (path / "index.html",)
    if path.suffix:
        return (path,)
    return (path, path.with_suffix(".html"), path / "index.html")


def _target_for(
    raw: str,
    source: PurePosixPath,
    project_prefix: str,
) -> _Target:
    value = html.unescape(raw.strip())
    if value.startswith("//"):
        return _Target(value, (), None, PurePosixPath("."), ignored=True)
    try:
        parsed = urlsplit(value)
    except ValueError:
        return _Target(value, (), None, PurePosixPath("."), unsafe=True)
    scheme = parsed.scheme.casefold()
    if scheme in _IGNORED_SCHEMES:
        return _Target(value, (), None, PurePosixPath("."), ignored=True)
    if scheme or parsed.netloc:
        return _Target(value, (), None, PurePosixPath("."), unsafe=True)
    if _INVALID_PERCENT_ESCAPE.search(parsed.path):
        return _Target(value, (), None, PurePosixPath("."), unsafe=True)

    decoded_path = unquote(parsed.path)
    decoded_fragment = unquote(html.unescape(parsed.fragment)) if parsed.fragment else None
    if "\\" in decoded_path or any(ord(character) < 32 for character in decoded_path):
        return _Target(value, (), decoded_fragment, PurePosixPath("."), unsafe=True)
    local_path, absolute = _project_path(decoded_path, project_prefix)
    if not decoded_path:
        normalized, safe = source, True
    else:
        normalized, safe = _normalized_path(local_path, source, absolute)
    query = f"?{parsed.query}" if parsed.query else ""
    fragment = f"#{decoded_fragment}" if decoded_fragment is not None else ""
    if not safe:
        display = f"{unquote(parsed.path)}{query}{fragment}"
        return _Target(display, (), decoded_fragment, normalized, unsafe=True)

    trailing_slash = bool(decoded_path) and decoded_path.endswith("/")
    if not decoded_path:
        display_path = source.as_posix()
    elif normalized == PurePosixPath("."):
        display_path = ""
    else:
        display_path = normalized.as_posix() + ("/" if trailing_slash else "")
    display = f"{display_path}{query}{fragment}"
    return _Target(
        display,
        _candidate_paths(normalized, trailing_slash),
        decoded_fragment,
        normalized,
    )


def _inside_site(site: Path, candidate: PurePosixPath) -> tuple[Path, bool]:
    root = site.resolve()
    resolved = (site / Path(candidate.as_posix())).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return resolved, False
    return resolved, True


def _existing_candidate(site: Path, target: _Target) -> PurePosixPath | None:
    for candidate in target.candidates:
        resolved, safe = _inside_site(site, candidate)
        if safe and resolved.is_file():
            return candidate
    return None


def _is_asset_target(target: _Target, existing: PurePosixPath | None) -> bool:
    if existing is not None:
        return True
    return "assets" in target.path.parts or target.path.suffix.casefold() not in _PAGE_SUFFIXES


def _srcset_urls(value: str) -> tuple[str, ...]:
    urls: list[str] = []
    cursor = 0
    while cursor < len(value):
        while cursor < len(value) and (value[cursor].isspace() or value[cursor] == ","):
            cursor += 1
        start = cursor
        while cursor < len(value) and not value[cursor].isspace():
            cursor += 1
        url = value[start:cursor]
        trailing_commas = len(url) - len(url.rstrip(","))
        url = url.rstrip(",")
        if url:
            urls.append(url)
        if trailing_commas:
            continue
        while cursor < len(value) and value[cursor] != ",":
            cursor += 1
        if cursor < len(value):
            cursor += 1
    return tuple(urls)


def _references(soup: BeautifulSoup) -> tuple[tuple[str, bool], ...]:
    references: list[tuple[str, bool]] = []
    for tag in soup.find_all(True):
        href = tag.get("href")
        if isinstance(href, str):
            rel = tag.get("rel") or ()
            asset = tag.name == "link" and bool(
                {str(item).casefold() for item in rel} & _ASSET_LINK_RELS
            )
            references.append((href, asset))
        for attribute in ("src", "poster", "data"):
            value = tag.get(attribute)
            if isinstance(value, str):
                references.append((value, True))
        for attribute in ("action", "formaction"):
            value = tag.get(attribute)
            if isinstance(value, str):
                references.append((value, False))
        srcset = tag.get("srcset")
        if isinstance(srcset, str):
            references.extend((url, True) for url in _srcset_urls(srcset))
    return tuple(references)


def _anchors(path: Path) -> set[str]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    anchors: set[str] = set()
    for tag in soup.find_all(True):
        identifier = tag.get("id")
        legacy_name = tag.get("name")
        if isinstance(identifier, str):
            anchors.add(html.unescape(identifier))
        if isinstance(legacy_name, str):
            anchors.add(html.unescape(legacy_name))
    return anchors


def validate_site(
    site: Path,
    *,
    route_manifest: Path,
    project_prefix: str = "/llm-knowledge/",
) -> ValidationReport:
    """Validate the built site against every route owned by the manifest.

    ``pages`` counts manifest-owned documents. ``orphans`` contains only generated
    HTML that is neither a route output, the MkDocs ``404.html`` auxiliary page,
    nor an HTML asset reached from a scanned internal reference.
    """
    routes = _load_routes(route_manifest)
    route_outputs = frozenset(route.output for route in routes)
    broken_links: set[str] = set()
    missing_anchors: set[str] = set()
    missing_assets: set[str] = set()
    missing_legacy_routes: set[str] = set()
    referenced_assets: set[PurePosixPath] = set()
    generated_html = {
        PurePosixPath(path.relative_to(site).as_posix())
        for path in site.rglob("*.html")
        if path.is_file()
    }
    source_asset_html = {
        path for path in generated_html if "assets" in path.parts
    }

    for route in routes:
        output_path, output_safe = _inside_site(site, route.output)
        output_exists = output_safe and output_path.is_file()
        for public_url in route.public_urls:
            target = _target_for(public_url, route.output, project_prefix)
            maps_to_output = not target.unsafe and route.output in target.candidates
            if not output_exists or not maps_to_output:
                missing_legacy_routes.add(
                    f"{route.source.as_posix()} -> {target.display}"
                )

    queued = deque(
        sorted(
            (
                output
                for output in (*route_outputs, *_AUXILIARY_HTML, *source_asset_html)
                if _inside_site(site, output)[0].is_file()
            ),
            key=lambda path: path.as_posix(),
        )
    )
    scanned: set[PurePosixPath] = set()
    anchor_cache: dict[PurePosixPath, set[str]] = {}
    while queued:
        source = queued.popleft()
        if source in scanned:
            continue
        scanned.add(source)
        source_path, source_safe = _inside_site(site, source)
        if not source_safe or not source_path.is_file():
            continue
        soup = BeautifulSoup(source_path.read_text(encoding="utf-8"), "html.parser")
        page_anchors: set[str] = set()
        for tag in soup.find_all(True):
            identifier = tag.get("id")
            legacy_name = tag.get("name")
            if isinstance(identifier, str):
                page_anchors.add(html.unescape(identifier))
            if isinstance(legacy_name, str):
                page_anchors.add(html.unescape(legacy_name))
        anchor_cache[source] = page_anchors
        for raw_reference, asset_context in _references(soup):
            target = _target_for(raw_reference, source, project_prefix)
            item = f"{source.as_posix()} -> {target.display}"
            if target.ignored:
                continue
            if target.unsafe:
                (missing_assets if asset_context else broken_links).add(item)
                continue

            route_target = next(
                (candidate for candidate in target.candidates if candidate in route_outputs),
                None,
            )
            existing = _existing_candidate(site, target)
            if route_target is not None:
                resolved, safe = _inside_site(site, route_target)
                if not safe or not resolved.is_file():
                    broken_links.add(item)
                    continue
                concrete = route_target
            elif asset_context or _is_asset_target(target, existing):
                if existing is None:
                    missing_assets.add(item)
                    continue
                concrete = existing
                referenced_assets.add(existing)
                if existing.suffix.casefold() in {".htm", ".html"}:
                    queued.append(existing)
            else:
                broken_links.add(item)
                continue

            if target.fragment is not None and concrete.suffix.casefold() in {".htm", ".html"}:
                anchors = anchor_cache.get(concrete)
                if anchors is None:
                    anchors = _anchors(_inside_site(site, concrete)[0])
                    anchor_cache[concrete] = anchors
                if target.fragment not in anchors:
                    missing_anchors.add(item)

    orphans = (
        generated_html
        - route_outputs
        - _AUXILIARY_HTML
        - source_asset_html
        - referenced_assets
    )
    return ValidationReport(
        pages=len(routes),
        broken_links=tuple(sorted(broken_links)),
        missing_anchors=tuple(sorted(missing_anchors)),
        missing_assets=tuple(sorted(missing_assets)),
        missing_legacy_routes=tuple(sorted(missing_legacy_routes)),
        orphans=tuple(sorted(path.as_posix() for path in orphans)),
    )


def raise_for_errors(report: ValidationReport) -> None:
    """Print a stable grouped report and raise when any validation category fails."""
    print(f"pages: {report.pages}")
    has_errors = False
    for field in _REPORT_FIELDS:
        items = tuple(sorted(getattr(report, field)))
        print(f"{field}: {len(items)}")
        for item in items:
            print(f"  {item}")
        has_errors = has_errors or bool(items)
    if has_errors:
        raise SiteValidationError("generated site validation failed")
