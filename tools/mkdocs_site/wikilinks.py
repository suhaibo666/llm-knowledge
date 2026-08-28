from __future__ import annotations

import posixpath
import re
from pathlib import PurePosixPath

from markdown.extensions.toc import slugify_unicode

from .models import Inventory, PageRecord


_FENCE_START = re.compile(r"^\s*(`{3,}|~{3,})")
_WIKILINK = re.compile(r"!?\[\[([^\[\]\n]+?)\]\]")


class LinkResolutionError(ValueError):
    """Raised when a source Wikilink cannot be converted without guessing."""

    def __init__(
        self,
        page: PageRecord,
        line: int,
        raw_link: str,
        reason: str,
        candidates: tuple[str, ...] = (),
    ) -> None:
        self.source = page.relative
        self.line = line
        self.raw_link = raw_link
        self.reason = reason
        self.candidates = candidates
        candidate_text = ", ".join(candidates) if candidates else "none"
        super().__init__(
            f"{page.relative.as_posix()}:{line}: {raw_link}: {reason}; "
            f"candidates=[{candidate_text}]"
        )


def _normalized_target(raw: str) -> tuple[str, str | None, str | None]:
    normalized = raw.replace(r"\|", "|")
    target_and_anchor, separator, alias = normalized.partition("|")
    target, anchor_separator, anchor = target_and_anchor.partition("#")
    target = target.strip().replace("\\", "/")
    if target.lower().endswith(".md"):
        target = target[:-3]
    return target, anchor.strip() if anchor_separator else None, alias if separator else None


def _normalize_relative(value: str) -> PurePosixPath:
    return PurePosixPath(posixpath.normpath(value))


def _resolve_target(
    target: str,
    page: PageRecord,
    inventory: Inventory,
    line: int,
    raw_link: str,
) -> PageRecord:
    if target.endswith("/"):
        raise LinkResolutionError(page, line, raw_link, "broken target")
    if "/" not in target:
        matches = inventory.by_stem.get(target, ())
    else:
        root_candidate = _normalize_relative(target)
        source_candidate = _normalize_relative(
            (page.relative.parent / target).as_posix()
        )
        for candidate in (root_candidate, source_candidate):
            match = inventory.by_relative.get(candidate)
            if match is not None:
                return match
        suffix = target
        matches = tuple(
            candidate
            for candidate in inventory.pages
            if candidate.relative.with_suffix("").as_posix().endswith(f"/{suffix}")
        )

    candidates = tuple(match.relative.as_posix() for match in matches)
    if not matches:
        raise LinkResolutionError(page, line, raw_link, "broken target", candidates)
    if len(matches) > 1:
        raise LinkResolutionError(page, line, raw_link, "ambiguous target", candidates)
    return matches[0]


def _relative_markdown_path(source: PageRecord, target: PageRecord) -> str:
    return posixpath.relpath(
        target.relative.as_posix(), start=source.relative.parent.as_posix()
    )


def _rewrite_token(
    match: re.Match[str], page: PageRecord, inventory: Inventory, line: int
) -> str:
    raw_link = match.group(0)
    if raw_link.startswith("!"):
        raise LinkResolutionError(page, line, raw_link, "unsupported embed")
    target_text, anchor, alias = _normalized_target(match.group(1))
    if "^" in target_text or (anchor is not None and "^" in anchor):
        raise LinkResolutionError(page, line, raw_link, "unsupported block reference")
    target = _resolve_target(target_text, page, inventory, line, raw_link)
    destination = _relative_markdown_path(page, target)
    if anchor is not None:
        anchor_slug = slugify_unicode(anchor, "-")
        heading_slugs = tuple(slugify_unicode(heading, "-") for heading in target.headings)
        if anchor_slug not in heading_slugs:
            raise LinkResolutionError(
                page,
                line,
                raw_link,
                "missing target anchor",
                target.headings,
            )
        destination = f"{destination}#{anchor_slug}"
    label = alias if alias is not None else target_text
    return f"[{label}]({destination})"


def _rewrite_text_segment(
    text: str, page: PageRecord, inventory: Inventory, line: int
) -> str:
    output: list[str] = []
    cursor = 0
    for match in _WIKILINK.finditer(text):
        if match.start() > 0 and text[match.start() - 1] == "\\":
            continue
        output.append(text[cursor : match.start()])
        output.append(_rewrite_token(match, page, inventory, line))
        cursor = match.end()
    output.append(text[cursor:])
    return "".join(output)


def _rewrite_inline_code_aware(
    line_text: str, page: PageRecord, inventory: Inventory, line: int
) -> str:
    output: list[str] = []
    cursor = 0
    while cursor < len(line_text):
        tick_start = line_text.find("`", cursor)
        if tick_start < 0:
            output.append(_rewrite_text_segment(line_text[cursor:], page, inventory, line))
            break
        output.append(
            _rewrite_text_segment(line_text[cursor:tick_start], page, inventory, line)
        )
        tick_end = tick_start
        while tick_end < len(line_text) and line_text[tick_end] == "`":
            tick_end += 1
        delimiter = line_text[tick_start:tick_end]
        closing = line_text.find(delimiter, tick_end)
        if closing < 0:
            output.append(_rewrite_text_segment(delimiter, page, inventory, line))
            cursor = tick_end
            continue
        closing += len(delimiter)
        output.append(line_text[tick_start:closing])
        cursor = closing
    return "".join(output)


def rewrite_wikilinks(markdown: str, page: PageRecord, inventory: Inventory) -> str:
    """Convert supported Obsidian Wikilinks while preserving literal code."""
    output: list[str] = []
    fence: str | None = None
    for line_number, line in enumerate(markdown.splitlines(keepends=True), start=1):
        fence_match = _FENCE_START.match(line)
        if fence is not None:
            output.append(line)
            if (
                fence_match is not None
                and fence_match.group(1)[0] == fence[0]
                and len(fence_match.group(1)) >= len(fence)
            ):
                fence = None
            continue
        if fence_match is not None:
            fence = fence_match.group(1)
            output.append(line)
            continue
        output.append(_rewrite_inline_code_aware(line, page, inventory, line_number))
    return "".join(output)
