from __future__ import annotations

import html
import posixpath
import re
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from markdown.extensions.toc import slugify, slugify_unicode, unique

from .models import Inventory, PageRecord


_FENCE_START = re.compile(r"^\s*(`{3,}|~{3,})")
_ATX_HEADING = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+|$)")
_ATX_HEADING_TEXT = re.compile(
    r"^ {0,3}(?P<marks>#{1,6})[ \t]+(?P<text>.*?)(?:[ \t]+#+)?[ \t]*(?:\r?\n)?$"
)
_WIKILINK = re.compile(r"!?\[\[([^\[\]\n]+?)\]\]")
_REPOSITORY_LINK = re.compile(
    r"(?P<open>\]\()(?P<target>(?:tools/|Megatron-LM/)[^)\s]+)(?P<close>\))"
)
_LOCAL_FILE_LINK = re.compile(
    r"(?<!\\)!?\[(?P<label>[^\]\n]+)\]\(\s*file:[^)\n]*\)",
    re.IGNORECASE,
)
_LOCAL_CODE_LOCATOR_LINK = re.compile(
    r"(?<!\\)!?\[(?P<label>[^\]\n]+)\]\(\s*`[^`\n]*\.[A-Za-z0-9_]+:\d+(?:-\d+)?`\s*\)"
)
_EXISTING_HTML_ANCHOR = re.compile(
    r"\b(?:id|name)\s*=\s*['\"](?P<anchor>[^'\"]+)['\"]",
    re.IGNORECASE,
)
_SAME_PAGE_FRAGMENT_LINK = re.compile(
    r"(?<!\\)!?\[(?P<label>[^\]\n]+)\]\(\s*#(?P<fragment>[^)\s]+)\s*\)"
)
_MEGATRON_BASELINE_DECLARATION = re.compile(
    r"^>\s+\*\*源码基线\*\*.*?NVIDIA/Megatron-LM@"
    r"([0-9a-fA-F]{40})(?![0-9a-fA-F])",
    re.MULTILINE,
)
_SECOND_LEVEL_HEADING = re.compile(r"^ {0,3}##(?:[ \t]+|$)", re.MULTILINE)


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


def _megatron_header_baselines(markdown: str) -> tuple[str, ...]:
    baselines: dict[str, None] = {}
    fence: str | None = None
    for line in markdown.splitlines(keepends=True):
        if fence is not None:
            if _closes_fence(line, fence):
                fence = None
            continue
        fence_match = _FENCE_START.match(line)
        if fence_match is not None:
            prefix = line[: fence_match.start(1)]
            if _indentation_columns(prefix) <= 3 and prefix == " " * len(prefix):
                fence = fence_match.group(1)
                continue
        if _SECOND_LEVEL_HEADING.match(line):
            break
        declaration = _MEGATRON_BASELINE_DECLARATION.match(line)
        if declaration is not None:
            baselines.setdefault(declaration.group(1).lower(), None)
    return tuple(baselines)


def _indentation_columns(line: str) -> int:
    columns = 0
    for character in line:
        if character == " ":
            columns += 1
        elif character == "\t":
            columns += 4 - (columns % 4)
        else:
            break
    return columns


def _loose_anchor_key(value: str) -> str:
    return "".join(
        character.casefold()
        for character in html.unescape(value)
        if character.isalnum()
    )


def _wiki_root(page: PageRecord) -> Path:
    root = page.source.resolve()
    for _ in page.relative.parts:
        root = root.parent
    return root


def _is_wiki_local_standard_target(
    target: str, page: PageRecord, inventory: Inventory
) -> bool:
    path_text = target.split("#", 1)[0].split("?", 1)[0].replace("\\", "/")
    root_relative = _normalize_relative(path_text)
    source_relative = _normalize_relative(
        (page.relative.parent / path_text).as_posix()
    )
    for candidate in (root_relative, source_relative):
        inventory_key = (
            candidate.with_suffix("") if candidate.suffix == ".md" else candidate
        )
        if inventory_key in inventory.by_relative:
            return True

    wiki_root = _wiki_root(page)
    for candidate in (wiki_root / path_text, page.source.parent / path_text):
        resolved = candidate.resolve()
        if resolved.is_relative_to(wiki_root) and resolved.is_file():
            return True
    return False


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
    text: str,
    page: PageRecord,
    inventory: Inventory,
    line: int,
    megatron_baselines: tuple[str, ...],
) -> str:
    def replace_local_file_link(match: re.Match[str]) -> str:
        return f"{match.group('label')} *(local source)*"

    def replace_repository_link(match: re.Match[str]) -> str:
        target = match.group("target")
        if _is_wiki_local_standard_target(target, page, inventory):
            return match.group(0)
        if target.startswith("tools/"):
            destination = (
                "https://github.com/suhaibo666/llm-knowledge/blob/main/" + target
            )
        else:
            match_line = line + text.count("\n", 0, match.start())
            if not megatron_baselines:
                raise LinkResolutionError(
                    page,
                    match_line,
                    match.group(0),
                    "Megatron-LM link requires a page source baseline",
                )
            if len(megatron_baselines) > 1:
                raise LinkResolutionError(
                    page,
                    match_line,
                    match.group(0),
                    "conflicting Megatron-LM source baseline declarations",
                    megatron_baselines,
                )
            destination = (
                "https://github.com/NVIDIA/Megatron-LM/blob/"
                f"{megatron_baselines[0]}/{target.removeprefix('Megatron-LM/')}"
            )
        return f"{match.group('open')}{destination}{match.group('close')}"

    text = _LOCAL_FILE_LINK.sub(replace_local_file_link, text)
    text = _LOCAL_CODE_LOCATOR_LINK.sub(replace_local_file_link, text)
    text = _REPOSITORY_LINK.sub(replace_repository_link, text)
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


def _without_inline_code(text: str) -> str:
    output: list[str] = []
    cursor = 0
    while cursor < len(text):
        start = text.find("`", cursor)
        if start < 0:
            output.append(text[cursor:])
            break
        output.append(text[cursor:start])
        end = _backtick_run_end(text, start)
        closing = _matching_backtick_run(text, end, end - start)
        if closing is None:
            output.append(text[start:end])
            cursor = end
            continue
        closing_end = closing + (end - start)
        output.append(" " * (closing_end - start))
        cursor = closing_end
    return "".join(output)


def _visible_markdown(markdown: str) -> str:
    output: list[str] = []
    fence: str | None = None
    for line in markdown.splitlines(keepends=True):
        fence_match = _FENCE_START.match(line)
        if fence is not None:
            if _closes_fence(line, fence):
                fence = None
            continue
        if _indentation_columns(line) >= 4:
            continue
        if fence_match is not None:
            fence = fence_match.group(1)
            continue
        output.append(_without_inline_code(line))
    return "".join(output)


def _manual_heading_aliases(markdown: str) -> dict[int, tuple[str, ...]]:
    visible = _visible_markdown(markdown)
    heading_texts = tuple(
        match.group("text").strip()
        for line in visible.splitlines(keepends=True)
        if (match := _ATX_HEADING_TEXT.match(line)) is not None
        and len(match.group("marks")) in {2, 3}
    )
    heading_keys = tuple(_loose_anchor_key(heading) for heading in heading_texts)
    aliases: dict[int, list[str]] = {}
    for match in _SAME_PAGE_FRAGMENT_LINK.finditer(visible):
        label_key = _loose_anchor_key(match.group("label"))
        fragment = unquote(html.unescape(match.group("fragment")))
        fragment_key = _loose_anchor_key(fragment)
        candidates = {
            index
            for index, heading_key in enumerate(heading_keys)
            if fragment_key == heading_key
            or label_key == heading_key
            or (len(label_key) >= 4 and heading_key.endswith(label_key))
        }
        if len(candidates) == 1:
            index = candidates.pop()
            aliases.setdefault(index, []).append(fragment)
    return {
        index: tuple(dict.fromkeys(values)) for index, values in aliases.items()
    }


def _rewrite_inline_code_aware(
    text: str,
    page: PageRecord,
    inventory: Inventory,
    line: int,
    megatron_baselines: tuple[str, ...],
) -> str:
    output: list[str] = []
    cursor = 0
    while cursor < len(text):
        tick_start = text.find("`", cursor)
        locator = _LOCAL_CODE_LOCATOR_LINK.search(text, cursor)
        segment_line = line + text.count("\n", 0, cursor)
        if locator is not None and (tick_start < 0 or locator.start() < tick_start):
            output.append(
                _rewrite_text_segment(
                    text[cursor : locator.start()],
                    page,
                    inventory,
                    segment_line,
                    megatron_baselines,
                )
            )
            output.append(f"{locator.group('label')} *(local source)*")
            cursor = locator.end()
            continue
        if tick_start < 0:
            output.append(
                _rewrite_text_segment(
                    text[cursor:], page, inventory, segment_line, megatron_baselines
                )
            )
            break
        output.append(
            _rewrite_text_segment(
                text[cursor:tick_start],
                page,
                inventory,
                segment_line,
                megatron_baselines,
            )
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
    """Convert supported wiki and repository links while preserving literal code."""
    megatron_baselines = _megatron_header_baselines(markdown)
    visible_markdown = _visible_markdown(markdown)
    existing_anchors = {
        html.unescape(match.group("anchor"))
        for match in _EXISTING_HTML_ANCHOR.finditer(visible_markdown)
    }
    manual_heading_aliases = _manual_heading_aliases(markdown)
    used_ascii_anchors: set[str] = set()
    used_unicode_anchors: set[str] = set()
    heading_index = 0
    output: list[str] = []
    visible_lines: list[str] = []
    visible_start_line = 1
    fence: str | None = None

    def flush_visible() -> None:
        if visible_lines:
            output.append(
                _rewrite_inline_code_aware(
                    "".join(visible_lines),
                    page,
                    inventory,
                    visible_start_line,
                    megatron_baselines,
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
        if _indentation_columns(line) >= 4:
            flush_visible()
            output.append(line)
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
            heading_match = _ATX_HEADING_TEXT.match(line)
            if heading_match is not None and len(heading_match.group("marks")) in {2, 3}:
                heading = heading_match.group("text").strip()
                ascii_anchor = unique(slugify(heading, "-"), used_ascii_anchors)
                unicode_anchor = unique(
                    slugify_unicode(heading, "-"), used_unicode_anchors
                )
                aliases: list[str] = []
                if unicode_anchor != ascii_anchor:
                    aliases.append(unicode_anchor)
                aliases.extend(manual_heading_aliases.get(heading_index, ()))
                heading_index += 1
                for alias in dict.fromkeys(aliases):
                    if alias not in existing_anchors:
                        output.append(
                            f'<a name="{html.escape(alias, quote=True)}"></a>\n'
                        )
            output.append(
                _rewrite_inline_code_aware(
                    line, page, inventory, line_number, megatron_baselines
                )
            )
            visible_start_line = line_number + 1
            continue
        if not visible_lines:
            visible_start_line = line_number
        visible_lines.append(line)
    flush_visible()
    return "".join(output)
