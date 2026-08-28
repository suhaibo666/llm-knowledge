import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import yaml

from .models import Inventory, PageRecord


class InventoryError(ValueError):
    """Raised when a Markdown source page cannot be inventoried."""


_FENCE_START = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_HEADING = re.compile(r"^(#{1,3})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
_ROOT_NAV_TITLE = "LLM Knowledge Wiki"


def _frontmatter_and_lines(text: str, source: PurePosixPath) -> tuple[Mapping[str, object], list[str]]:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return {}, lines
    for index in range(1, len(lines)):
        if lines[index] in {"---", "..."}:
            try:
                loaded = yaml.safe_load("\n".join(lines[1:index]))
            except yaml.YAMLError as error:
                raise InventoryError(f"{source.as_posix()}: invalid frontmatter") from error
            if loaded is None:
                return {}, lines[index + 1 :]
            if not isinstance(loaded, Mapping):
                raise InventoryError(f"{source.as_posix()}: frontmatter must be a mapping")
            return loaded, lines[index + 1 :]
    return {}, lines


def _headings(lines: list[str]) -> tuple[str | None, tuple[str, ...]]:
    fence: str | None = None
    first_h1: str | None = None
    headings: list[str] = []
    for line in lines:
        fence_match = _FENCE_START.match(line)
        if fence is not None:
            if (
                fence_match is not None
                and fence_match.group(1)[0] == fence[0]
                and len(fence_match.group(1)) >= len(fence)
                and not line[fence_match.end() :].strip()
            ):
                fence = None
            continue
        if fence_match is not None:
            fence = fence_match.group(1)
            continue
        match = _HEADING.match(line)
        if match is None:
            continue
        level = len(match.group(1))
        text = match.group(2).strip()
        if level == 1 and first_h1 is None:
            first_h1 = text
        elif level in {2, 3}:
            headings.append(text)
    return first_h1, tuple(headings)


def _title(frontmatter: Mapping[str, object], first_h1: str | None) -> str | None:
    title = frontmatter.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return first_h1


def scan_inventory(wiki: Path) -> Inventory:
    """Return a deterministic, read-only inventory of Markdown pages in *wiki*."""
    root = wiki.resolve()
    sources = sorted(
        root.rglob("*.md"),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    pages: list[PageRecord] = []
    for source in sources:
        relative = PurePosixPath(source.relative_to(root).as_posix())
        try:
            resolved_source = source.resolve(strict=True)
        except OSError as error:
            raise InventoryError(
                f"{relative.as_posix()}: cannot resolve source inside wiki root"
            ) from error
        try:
            resolved_source.relative_to(root)
        except ValueError as error:
            raise InventoryError(
                f"{relative.as_posix()}: resolves outside wiki root: {resolved_source}"
            ) from error
        if not resolved_source.is_file():
            continue
        try:
            text = resolved_source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise InventoryError(f"{relative.as_posix()}: cannot read UTF-8") from error
        frontmatter, body = _frontmatter_and_lines(text, relative)
        first_h1, headings = _headings(body)
        title = _title(frontmatter, first_h1)
        if title is None:
            raise InventoryError(f"{relative.as_posix()}: missing title")
        pages.append(
            PageRecord(
                source=source,
                relative=relative,
                title=title,
                nav_title=(
                    _ROOT_NAV_TITLE
                    if relative == PurePosixPath("index.md")
                    else title if source.name == "index.md" else source.stem
                ),
                is_index=source.name == "index.md",
                headings=headings,
            )
        )
    try:
        return Inventory.from_pages(tuple(pages))
    except ValueError as error:
        raise InventoryError(str(error)) from error
