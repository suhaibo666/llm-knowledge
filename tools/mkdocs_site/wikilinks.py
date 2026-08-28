from __future__ import annotations

import posixpath
import re
from pathlib import PurePosixPath

from markdown.extensions.toc import slugify_unicode, unique

from .models import Inventory, PageRecord


_FENCE_START = re.compile(r"^\s*(`{3,}|~{3,})")
_ATX_HEADING = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+|$)")
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


def _closes_fence(line: str, fence: str) -> bool:
    candidate = line.rstrip("\r\n")
    match = _FENCE_START.match(candidate)
    return (
        match is not None
        and match.group(1)[0] == fence[0]
        and len(match.group(1)) >= len(fence)
        and not candidate[match.end() :].strip()
    )


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


def _heading_anchors(page: PageRecord) -> tuple[str, ...]:
    used: set[str] = set()
    return tuple(
        unique(slugify_unicode(heading, "-"), used) for heading in page.headings
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
        heading_anchors = _heading_anchors(target)
        if anchor_slug not in heading_anchors:
            raise LinkResolutionError(
                page,
                line,
                raw_link,
                "missing target anchor",
                heading_anchors,
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
        match_line = line + text.count("\n", 0, match.start())
        output.append(_rewrite_token(match, page, inventory, match_line))
        cursor = match.end()
    output.append(text[cursor:])
    return "".join(output)


def _backtick_run_end(text: str, start: int) -> int:
    end = start
    while end < len(text) and text[end] == "`":
        end += 1
    return end


def _matching_backtick_run(text: str, start: int, length: int) -> int | None:
    cursor = start
    while True:
        candidate = text.find("`", cursor)
        if candidate < 0:
            return None
        end = _backtick_run_end(text, candidate)
        if end - candidate == length:
            return candidate
        cursor = end


def _rewrite_inline_code_aware(
    text: str, page: PageRecord, inventory: Inventory, line: int
) -> str:
    output: list[str] = []
    cursor = 0
    while cursor < len(text):
        tick_start = text.find("`", cursor)
        segment_line = line + text.count("\n", 0, cursor)
        if tick_start < 0:
            output.append(_rewrite_text_segment(text[cursor:], page, inventory, segment_line))
            break
        output.append(
            _rewrite_text_segment(text[cursor:tick_start], page, inventory, segment_line)
        )
        tick_end = _backtick_run_end(text, tick_start)
        closing = _matching_backtick_run(text, tick_end, tick_end - tick_start)
        if closing is None:
            output.append(text[tick_start:tick_end])
            cursor = tick_end
            continue
        closing_end = closing + (tick_end - tick_start)
        output.append(text[tick_start:closing_end])
        cursor = closing_end
    return "".join(output)


def rewrite_wikilinks(markdown: str, page: PageRecord, inventory: Inventory) -> str:
    """Convert supported Obsidian Wikilinks while preserving literal code."""
    output: list[str] = []
    visible_lines: list[str] = []
    visible_start_line = 1
    fence: str | None = None

    def flush_visible() -> None:
        if visible_lines:
            output.append(
                _rewrite_inline_code_aware(
                    "".join(visible_lines), page, inventory, visible_start_line
                )
            )
            visible_lines.clear()

    for line_number, line in enumerate(markdown.splitlines(keepends=True), start=1):
        fence_match = _FENCE_START.match(line)
        if fence is not None:
            output.append(line)
            if _closes_fence(line, fence):
                fence = None
                visible_start_line = line_number + 1
            continue
        if fence_match is not None:
            flush_visible()
            fence = fence_match.group(1)
            output.append(line)
            continue
        if not line.strip():
            flush_visible()
            output.append(line)
            visible_start_line = line_number + 1
            continue
        if _ATX_HEADING.match(line):
            flush_visible()
            output.append(_rewrite_inline_code_aware(line, page, inventory, line_number))
            visible_start_line = line_number + 1
            continue
        if not visible_lines:
            visible_start_line = line_number
        visible_lines.append(line)
    flush_visible()
    return "".join(output)
