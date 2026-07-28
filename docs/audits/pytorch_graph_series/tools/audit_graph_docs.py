from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SOURCE_SUFFIXES = "py|pyi|cpp|cc|cxx|c|h|hpp|cuh|md|rst"
FULL_LOCATOR_RE = re.compile(
    rf"(?P<path>(?:[A-Za-z0-9_.-]+[/\\])+[A-Za-z0-9_.-]+\.(?:{SOURCE_SUFFIXES}))"
    r":L?(?P<start>[0-9]+)(?:-L?(?P<end>[0-9]+))?"
)
SHORT_LOCATOR_RE = re.compile(
    rf"(?<![/\\A-Za-z0-9_.-])(?P<path>[A-Za-z0-9_.-]+\.(?:{SOURCE_SUFFIXES}))"
    r":L?(?P<start>[0-9]+)(?:-L?(?P<end>[0-9]+))?"
)
HEADING_RE = re.compile(r"^(?P<marks>#{2,3})[ \t]+(?P<title>.+?)[ \t]*$")
FENCE_RE = re.compile(
    r"^[ ]{0,3}(?P<marker>`{3,}|~{3,})(?P<language>.*)$"
)
WIKILINK_RE = re.compile(
    r"\[\[(?P<target>[^\]|#]+)(?:#(?P<anchor>[^\]|]+))?(?:\|(?P<label>[^\]]+))?\]\]"
)
INLINE_IMAGE_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^\s)]+)(?:\s+(?P<title>[^)]*))?\)"
)
REFERENCE_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\[(?P<reference>[^\]]+)\]")
REFERENCE_DEFINITION_RE = re.compile(
    r"^\[(?P<reference>[^\]]+)\]:\s*(?P<target>\S+)(?:\s+(?P<title>.+))?$"
)
TABLE_DELIMITER_CELL_RE = re.compile(r"^:?-{3,}:?$")
THEMATIC_BREAK_RE = re.compile(
    r"^[ \t]{0,3}(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})$"
)
EXPERIMENT_HEADING_RE = re.compile(r"\b(?:lab|experiment)\b|实验", re.IGNORECASE)
LIST_ITEM_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>[-+*]|[0-9]+[.)])[ \t]+(?P<body>.*)$"
)
NAVIGATION_LINK = r"(?:\[\[[^\]]+\]\]|\[[^\]]+\]\([^)]+\))"
NAVIGATION_BODY_RE = re.compile(
    rf"^{NAVIGATION_LINK}(?:[ \t]*[,，、;；|/·—-][ \t]*{NAVIGATION_LINK})*[。.]?$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_locators(text: str) -> list[dict[str, object]]:
    """Extract locator occurrences in textual order without deduplicating them."""

    matches: list[tuple[int, int, re.Match[str]]] = []
    occupied: list[tuple[int, int]] = []
    for match in FULL_LOCATOR_RE.finditer(text):
        matches.append((match.start(), match.end(), match))
        occupied.append((match.start(), match.end()))

    for match in SHORT_LOCATOR_RE.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        matches.append((match.start(), match.end(), match))

    result: list[dict[str, object]] = []
    for source_start, source_end, match in sorted(matches, key=lambda item: item[0]):
        start_line = int(match.group("start"))
        end_text = match.group("end")
        target_end_line = int(end_text) if end_text else start_line
        result.append(
            {
                "raw": match.group(0),
                "path": match.group("path").replace("\\", "/"),
                "start_line": start_line,
                "end_line": target_end_line,
                "target_start_line": start_line,
                "target_end_line": target_end_line,
                "source_start_column": source_start + 1,
                "source_end_column": source_end,
            }
        )
    return result


def _extract_wikilinks(text: str) -> list[dict[str, object]]:
    links: list[dict[str, object]] = []
    for match in WIKILINK_RE.finditer(text):
        links.append(
            {
                "raw": match.group(0),
                "target": match.group("target"),
                "anchor": match.group("anchor"),
                "label": match.group("label"),
            }
        )
    return links


def _split_markdown_table_row(line: str) -> list[str] | None:
    """Split a Markdown table row on unescaped pipe separators."""

    if "|" not in line:
        return None
    stripped = line.strip()
    cells: list[str] = []
    current: list[str] = []
    separator_positions: list[int] = []
    escaped = False
    for position, character in enumerate(stripped):
        if character == "|" and not escaped:
            cells.append("".join(current).strip())
            current = []
            separator_positions.append(position)
        else:
            current.append(character)
        escaped = character == "\\" and not escaped
        if character != "\\":
            escaped = False
    cells.append("".join(current).strip())
    if not separator_positions:
        return None
    if separator_positions[0] == 0:
        cells = cells[1:]
    if separator_positions[-1] == len(stripped) - 1:
        cells = cells[:-1]
    return cells


def _is_closing_fence(line: str, opening_marker: str) -> bool:
    stripped = line.lstrip(" ")
    indentation = len(line) - len(stripped)
    if indentation > 3:
        return False
    marker_character = re.escape(opening_marker[0])
    return (
        re.fullmatch(
            rf"{marker_character}{{{len(opening_marker)},}}[ \t]*",
            stripped,
        )
        is not None
    )


def _is_navigation_list_item(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) != 1:
        return False
    match = LIST_ITEM_RE.match(lines[0])
    return bool(match and NAVIGATION_BODY_RE.fullmatch(match.group("body").strip()))


def _is_markdown_table_delimiter(cells: list[str] | None, column_count: int) -> bool:
    return bool(
        cells
        and len(cells) == column_count
        and all(TABLE_DELIMITER_CELL_RE.fullmatch(cell) for cell in cells)
    )


def parse_markdown(path: Path) -> list[dict[str, object]]:
    """Inventory structural units and references from one Markdown document."""

    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, object]] = []
    current_h2: str | None = None
    current_h3: str | None = None
    open_fence: dict[str, Any] | None = None
    open_experiment: dict[str, object] | None = None
    paragraph_lines: list[str] = []
    paragraph_start = 0
    paragraph_heading_path: list[str] = []
    list_item_lines: list[str] = []
    list_item_start = 0
    list_item_end = 0
    list_item_heading_path: list[str] = []
    reference_targets: dict[str, str] = {}
    pending_reference_images: list[dict[str, object]] = []

    def heading_path() -> list[str]:
        return [item for item in (current_h2, current_h3) if item]

    def append_references(line: str, line_number: int) -> None:
        for locator in extract_locators(line):
            records.append(
                {
                    "kind": "locator",
                    "heading_path": heading_path(),
                    "source_line": line_number,
                    **locator,
                }
            )
        for link in _extract_wikilinks(line):
            records.append(
                {
                    "kind": "wikilink",
                    "heading_path": heading_path(),
                    "source_line": line_number,
                    **link,
                }
            )

    def flush_paragraph() -> None:
        nonlocal paragraph_lines, paragraph_start, paragraph_heading_path
        if paragraph_lines:
            records.append(
                {
                    "kind": "claim_candidate",
                    "heading_path": paragraph_heading_path,
                    "start_line": paragraph_start,
                    "end_line": paragraph_start + len(paragraph_lines) - 1,
                    "text": "\n".join(paragraph_lines),
                }
            )
        paragraph_lines = []
        paragraph_start = 0
        paragraph_heading_path = []

    def flush_list_item() -> None:
        nonlocal list_item_lines, list_item_start, list_item_end
        nonlocal list_item_heading_path
        if list_item_lines:
            text = "\n".join(list_item_lines)
            records.append(
                {
                    "kind": (
                        "navigation"
                        if _is_navigation_list_item(text)
                        else "claim_candidate"
                    ),
                    "heading_path": list_item_heading_path,
                    "start_line": list_item_start,
                    "end_line": list_item_end,
                    "text": text,
                }
            )
        list_item_lines = []
        list_item_start = 0
        list_item_end = 0
        list_item_heading_path = []

    def append_images(line: str, line_number: int) -> list[re.Match[str]]:
        matches: list[re.Match[str]] = [
            *INLINE_IMAGE_RE.finditer(line),
            *REFERENCE_IMAGE_RE.finditer(line),
        ]
        for match in INLINE_IMAGE_RE.finditer(line):
            records.append(
                {
                    "kind": "image",
                    "syntax": "inline",
                    "alt": match.group("alt"),
                    "target": match.group("target"),
                    "title": match.group("title"),
                    "heading_path": heading_path(),
                    "source_line": line_number,
                    "source_start_column": match.start() + 1,
                    "source_end_column": match.end(),
                }
            )
        for match in REFERENCE_IMAGE_RE.finditer(line):
            image = {
                "kind": "image",
                "syntax": "reference",
                "alt": match.group("alt"),
                "target": "",
                "reference": match.group("reference"),
                "heading_path": heading_path(),
                "source_line": line_number,
                "source_start_column": match.start() + 1,
                "source_end_column": match.end(),
            }
            records.append(image)
            pending_reference_images.append(image)
        return matches

    def has_prose_outside_images(
        line: str, image_matches: Sequence[re.Match[str]]
    ) -> bool:
        prose_line = line
        for match in sorted(image_matches, key=lambda item: item.start(), reverse=True):
            prose_line = prose_line[: match.start()] + prose_line[match.end() :]
        return bool(re.sub(r"[ \t]+", " ", prose_line).strip())

    index = 0
    while index < len(lines):
        line = lines[index]
        line_number = index + 1
        fence = FENCE_RE.match(line)

        if open_fence is not None:
            if _is_closing_fence(line, str(open_fence["marker"])):
                records.append(
                    {
                        **open_fence,
                        "end_line": line_number,
                        "content": "\n".join(open_fence["content_lines"]),
                        "balanced": True,
                    }
                )
                records[-1].pop("content_lines")
                open_fence = None
            else:
                open_fence["content_lines"].append(line)
            index += 1
            continue

        if fence is not None:
            flush_paragraph()
            flush_list_item()
            language = fence.group("language").strip()
            if fence.group("marker").startswith("`") and "`" in language:
                if not paragraph_lines:
                    paragraph_start = line_number
                    paragraph_heading_path = heading_path()
                paragraph_lines.append(line)
                index += 1
                continue
            open_fence = {
                "kind": "code_fence",
                "language": language,
                "figure_classification": "mermaid" if language == "mermaid" else None,
                "marker": fence.group("marker"),
                "heading_path": heading_path(),
                "start_line": line_number,
                "content_lines": [],
            }
            index += 1
            continue

        heading = HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            flush_list_item()
            level = len(heading.group("marks"))
            title = heading.group("title").strip()
            if open_experiment is not None and level <= int(open_experiment["level"]):
                records.append(
                    {
                        "kind": "experiment",
                        "heading": open_experiment["heading"],
                        "heading_path": open_experiment["heading_path"],
                        "start_line": open_experiment["start_line"],
                        "end_line": line_number - 1,
                    }
                )
                open_experiment = None
            if level == 2:
                current_h2 = title
                current_h3 = None
            else:
                current_h3 = title
            current_path = heading_path()
            records.append(
                {
                    "kind": "heading",
                    "level": level,
                    "title": title,
                    "heading_path": current_path,
                    "source_line": line_number,
                }
            )
            if EXPERIMENT_HEADING_RE.search(title):
                open_experiment = {
                    "level": level,
                    "heading": title,
                    "heading_path": current_path,
                    "start_line": line_number,
                }
            index += 1
            continue

        if re.match(r"^#{1,6}[ \t]+", line):
            flush_paragraph()
            flush_list_item()
            append_references(line, line_number)
            index += 1
            continue

        if THEMATIC_BREAK_RE.fullmatch(line):
            flush_paragraph()
            flush_list_item()
            index += 1
            continue

        header_cells = _split_markdown_table_row(line)
        delimiter_cells = (
            _split_markdown_table_row(lines[index + 1])
            if index + 1 < len(lines)
            else None
        )
        if _is_markdown_table_delimiter(delimiter_cells, len(header_cells or [])):
            flush_paragraph()
            flush_list_item()
            data_rows: list[list[str]] = []
            end_line = line_number + 1
            row_index = index + 2
            while row_index < len(lines):
                data_cells = _split_markdown_table_row(lines[row_index])
                if data_cells is None:
                    break
                column_count = len(header_cells or [])
                if len(data_cells) < column_count:
                    data_cells = [*data_cells, *([""] * (column_count - len(data_cells)))]
                elif len(data_cells) > column_count:
                    data_cells = data_cells[:column_count]
                append_references(lines[row_index], row_index + 1)
                data_rows.append(data_cells)
                end_line = row_index + 1
                row_index += 1
            append_references(line, line_number)
            append_references(lines[index + 1], line_number + 1)
            records.append(
                {
                    "kind": "markdown_table",
                    "heading_path": heading_path(),
                    "start_line": line_number,
                    "end_line": end_line,
                    "header_cells": header_cells,
                    "delimiter_cells": delimiter_cells,
                    "data_rows": data_rows,
                    "raw": "\n".join(lines[index:row_index]),
                    "column_count": len(header_cells or []),
                    "data_row_count": len(data_rows),
                }
            )
            index = row_index
            continue

        append_references(line, line_number)
        definition = REFERENCE_DEFINITION_RE.match(line)
        if definition:
            flush_paragraph()
            flush_list_item()
            reference_targets[definition.group("reference").casefold()] = definition.group(
                "target"
            )
            index += 1
            continue

        list_match = LIST_ITEM_RE.match(line)
        if list_match is not None:
            flush_paragraph()
            flush_list_item()
            append_images(line, line_number)
            list_item_lines = [line]
            list_item_start = line_number
            list_item_end = line_number
            list_item_heading_path = heading_path()
            index += 1
            continue

        image_matches = append_images(line, line_number)
        if list_item_lines:
            if not line.strip():
                flush_list_item()
            else:
                list_item_lines.append(line)
                list_item_end = line_number
            index += 1
            continue

        if not line.strip() or not has_prose_outside_images(line, image_matches):
            flush_paragraph()
        else:
            if not paragraph_lines:
                paragraph_start = line_number
                paragraph_heading_path = heading_path()
            paragraph_lines.append(line)
        index += 1

    flush_paragraph()
    flush_list_item()

    if open_fence is not None:
        records.append(
            {
                **open_fence,
                "end_line": len(lines),
                "content": "\n".join(open_fence["content_lines"]),
                "balanced": False,
            }
        )
        records[-1].pop("content_lines")

    if open_experiment is not None:
        records.append(
            {
                "kind": "experiment",
                "heading": open_experiment["heading"],
                "heading_path": open_experiment["heading_path"],
                "start_line": open_experiment["start_line"],
                "end_line": len(lines),
            }
        )
    for image in pending_reference_images:
        image["target"] = reference_targets.get(str(image["reference"]).casefold(), "")

    return records


def resolve_locator(
    locator: Mapping[str, object], source_root: Path
) -> dict[str, object]:
    """Resolve only explicit repository-relative paths; never guess shorthand paths."""

    result = dict(locator)
    path_text = str(locator["path"]).replace("\\", "/")
    if "/" not in path_text:
        result["resolution"] = "needs_manual_resolution"
        result["resolved_path"] = None
        return result

    candidate = source_root.joinpath(*path_text.split("/"))
    result["resolved_path"] = candidate.as_posix()
    if not candidate.is_file():
        result["resolution"] = "path_missing"
        return result

    with candidate.open("r", encoding="utf-8", errors="replace") as handle:
        line_count = sum(1 for _ in handle)
    result["source_line_count"] = line_count
    start_line = int(locator.get("target_start_line", locator["start_line"]))
    end_line = int(locator.get("target_end_line", locator["end_line"]))
    result["target_start_line"] = start_line
    result["target_end_line"] = end_line
    if start_line < 1 or end_line < start_line or end_line > line_count:
        result["resolution"] = "line_out_of_bounds"
    else:
        result["resolution"] = "path_and_line_valid"
    return result


def audit_manifest(
    repo_root: Path,
    source_root: Path,
    manifest_path: Path,
) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not all(
        isinstance(item, str) for item in manifest
    ):
        raise ValueError("manifest must be a JSON list of repository-relative paths")

    records: list[dict[str, object]] = []
    for relative_path in manifest:
        page_path = repo_root / relative_path
        if not page_path.is_file():
            raise FileNotFoundError(f"manifest page does not exist: {relative_path}")
        line_count = len(page_path.read_text(encoding="utf-8").splitlines())
        page_hash = _sha256(page_path)
        records.append(
            {
                "kind": "page",
                "page": relative_path,
                "source_line": 1,
                "line_count": line_count,
                "sha256": page_hash,
            }
        )
        for record in parse_markdown(page_path):
            enriched = {"page": relative_path, "page_sha256": page_hash, **record}
            if record["kind"] == "locator":
                enriched = resolve_locator(enriched, source_root)
            records.append(enriched)
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_summary(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    by_page: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        page = str(record["page"])
        kind = str(record["kind"])
        by_page[page][kind] += 1
        if kind == "heading":
            by_page[page][f"h{record['level']}"] += 1
        elif kind == "code_fence":
            if record.get("language") == "mermaid":
                by_page[page]["mermaid"] += 1
            if not record.get("balanced"):
                by_page[page]["unbalanced_fence"] += 1
        elif kind == "locator":
            by_page[page][f"locator_{record.get('resolution')}"] += 1

    columns = [
        "Page",
        "Lines",
        "H2",
        "H3",
        "Code",
        "Mermaid",
        "Tables",
        "Images",
        "Experiments",
        "Claims",
        "Navigation",
        "Locators",
        "Links",
        "Unbalanced",
    ]
    output = [
        "# PyTorch Graph Series Mechanical Inventory",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    page_records = {
        str(record["page"]): record for record in records if record["kind"] == "page"
    }
    for page in sorted(by_page):
        counts = by_page[page]
        output.append(
            "| "
            + " | ".join(
                [
                    f"`{page}`",
                    str(page_records[page]["line_count"]),
                    str(counts["h2"]),
                    str(counts["h3"]),
                    str(counts["code_fence"]),
                    str(counts["mermaid"]),
                    str(counts["markdown_table"]),
                    str(counts["image"]),
                    str(counts["experiment"]),
                    str(counts["claim_candidate"]),
                    str(counts["navigation"]),
                    str(counts["locator"]),
                    str(counts["wikilink"]),
                    str(counts["unbalanced_fence"]),
                ]
            )
            + " |"
        )

    locator_counts = Counter(
        str(record.get("resolution"))
        for record in records
        if record["kind"] == "locator"
    )
    output.extend(
        [
            "",
            "## Locator resolution",
            "",
            *[
                f"- `{status}`: {count}"
                for status, count in sorted(locator_counts.items())
            ],
            "",
            "Mechanical path/line resolution does not imply semantic verification.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", "<br/>").replace("|", "\\|")


VALID_AUDIT_STATUSES = {
    "verified-current",
    "verified-historical",
    "corrected",
    "unresolved",
}
VALID_DESTINATION_ROLES = {
    "unassigned",
    "authoritative_import",
    "candidate_only",
    "retain_quarantined",
}
CORRECTION_ID_RE = re.compile(r"\b[FAIP]-[0-9]{3}\b")
CORRECTION_HEADING_RE = re.compile(
    r"^###\s+(?P<correction_id>[FAIP]-[0-9]{3})"
    r"\s+(?:—|-)\s+(?P<title>.+?)\s*$"
)
LOCAL_CORRECTION_CALLOUT_RE = re.compile(
    r"^[ \t]*>[ \t]*\[!(?P<callout_type>correction|contradiction)\]",
    re.IGNORECASE,
)


def _repository_relative_path(path: Path, repo_root: Path) -> str:
    root = repo_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"path is outside repository root: {resolved}"
        ) from error


def load_correction_catalog(
    paths: Sequence[Path], repo_root: Path
) -> dict[str, dict[str, object]]:
    """Index exact correction-report H3 headings by their provenance ID."""

    catalog: dict[str, dict[str, object]] = {}
    for path in paths:
        resolved = path.resolve()
        relative = _repository_relative_path(resolved, repo_root)
        report_hash = _sha256(resolved)
        heading_occurrences: Counter[str] = Counter()
        for line_number, line in enumerate(
            resolved.read_text(encoding="utf-8").splitlines(), 1
        ):
            heading = HEADING_RE.match(line)
            if heading is not None:
                heading_occurrences[heading.group("title").strip()] += 1
            match = CORRECTION_HEADING_RE.match(line)
            if match is None:
                continue
            correction_id = match.group("correction_id")
            if correction_id in catalog:
                raise ValueError(
                    f"duplicate correction ID {correction_id}: {relative}"
                )
            anchor_text = line.removeprefix("###").strip()
            catalog[correction_id] = {
                "correction_id": correction_id,
                "path": relative,
                "anchor_text": anchor_text,
                "anchor_occurrence": heading_occurrences[anchor_text],
                "source_line": line_number,
                "report_sha256": report_hash,
            }
    return catalog


def load_destination_aliases(path: Path) -> dict[str, object]:
    """Load and validate the one-time legacy destination alias table."""

    parsed = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"destination aliases must be a JSON object: {path}")
    if parsed.get("schema_version") != 1:
        raise ValueError(f"unsupported destination alias schema: {path}")
    aliases = parsed.get("aliases")
    manual = parsed.get("manual_split_required", {})
    if (
        not isinstance(aliases, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in aliases.items()
        )
        or not isinstance(manual, dict)
        or not all(isinstance(key, str) for key in manual)
    ):
        raise ValueError(f"invalid destination alias table: {path}")
    overlap = sorted(set(aliases).intersection(manual))
    if overlap:
        raise ValueError(
            f"destination aliases cannot also require manual split: {overlap}"
        )
    return parsed


def resolve_destination_alias(
    token: str, aliases: Mapping[str, object]
) -> str:
    """Resolve one legacy filename, refusing aliases that require semantic splitting."""

    manual = aliases.get("manual_split_required", {})
    if isinstance(manual, Mapping) and token in manual:
        raise ValueError(
            f"manual_split_required destination alias: {token}"
        )
    mapping = aliases.get("aliases")
    if not isinstance(mapping, Mapping) or token not in mapping:
        raise ValueError(f"unknown destination alias: {token}")
    resolved = mapping[token]
    if not isinstance(resolved, str) or not resolved:
        raise ValueError(f"invalid destination alias target: {token}")
    return resolved


def migrate_legacy_destination_decisions(
    decisions: Sequence[Mapping[str, object]],
    aliases: Mapping[str, object],
    repo_root: Path,
) -> list[dict[str, object]]:
    """Safely migrate only unambiguous legacy destination strings.

    This helper deliberately refuses to guess among multiple H2/H3 headings.
    Decisions that already use the canonical schema pass through unchanged,
    making repeated migration runs idempotent.
    """

    root = repo_root.resolve()
    migrated: list[dict[str, object]] = []
    for decision_index, decision in enumerate(decisions):
        if "destination" not in decision:
            migrated.append(
                {
                    key: (
                        [dict(item) for item in value]
                        if key == "destinations"
                        and isinstance(value, list)
                        else value
                    )
                    for key, value in decision.items()
                }
            )
            continue
        if "destinations" in decision:
            raise ValueError(
                "legacy decision cannot contain both destination and "
                f"destinations: index {decision_index}"
            )
        legacy_value = decision["destination"]
        if not isinstance(legacy_value, str):
            raise ValueError(
                f"legacy destination must be a string: index {decision_index}"
            )
        tokens = [token.strip() for token in legacy_value.split(",")]
        if not tokens or any(not token for token in tokens):
            raise ValueError(
                f"legacy destination has an empty token: index {decision_index}"
            )
        destinations: list[dict[str, object]] = []
        for token in tokens:
            relative_text = resolve_destination_alias(token, aliases)
            relative = Path(relative_text)
            target = (root / relative).resolve()
            try:
                target.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"invalid legacy destination path: {token}"
                ) from error
            if not target.is_file():
                raise ValueError(
                    f"missing legacy destination path: {relative_text}"
                )
            headings = [
                (anchor_text, line_number)
                for anchor_text, line_numbers in _destination_heading_index(
                    target
                ).items()
                for line_number in line_numbers
            ]
            if len(headings) != 1:
                raise ValueError(
                    "ambiguous legacy destination "
                    f"{token!r}: target has {len(headings)} H2/H3 headings; "
                    "supply an explicit semantic destination"
                )
            anchor_text, _ = headings[0]
            destinations.append(
                {
                    "path": relative_text.replace("\\", "/"),
                    "anchor_text": anchor_text,
                    "anchor_occurrence": 1,
                }
            )
        converted: dict[str, object] = {}
        for key, value in decision.items():
            if key == "destination":
                converted["destinations"] = destinations
                converted["destination_role"] = (
                    "candidate_only"
                    if decision.get("status") == "unresolved"
                    else "authoritative_import"
                )
            else:
                converted[key] = value
        migrated.append(converted)
    return migrated


def _destination_heading_index(path: Path) -> dict[str, list[int]]:
    """Return exact H2/H3 text to source-line occurrences for a Markdown page."""

    headings: defaultdict[str, list[int]] = defaultdict(list)
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        match = HEADING_RE.match(line)
        if match is not None:
            headings[match.group("title").strip()].append(line_number)
    return dict(headings)


def validate_destinations(
    rows: Sequence[Mapping[str, object]], repo_root: Path
) -> list[dict[str, object]]:
    """Validate structured destinations against exact repository H2/H3 headings."""

    root = repo_root.resolve()
    errors: list[dict[str, object]] = []
    heading_cache: dict[Path, dict[str, list[int]]] = {}

    def add_error(
        code: str,
        row_index: int,
        *,
        destination_index: int | None = None,
        **details: object,
    ) -> None:
        error: dict[str, object] = {
            "code": code,
            "row_index": row_index,
        }
        if destination_index is not None:
            error["destination_index"] = destination_index
        error.update(details)
        errors.append(error)

    for row_index, row in enumerate(rows):
        if "destination" in row:
            add_error(
                "bare_destination_string",
                row_index,
                value=row.get("destination"),
            )
        destinations = row.get("destinations")
        if not isinstance(destinations, list):
            add_error(
                "invalid_destinations",
                row_index,
                message="destinations must be an array",
            )
            continue
        role = row.get("destination_role")
        if role not in VALID_DESTINATION_ROLES:
            add_error(
                "invalid_destination_role",
                row_index,
                value=role,
            )
        if (
            str(row.get("status")) == "unresolved"
            and role == "authoritative_import"
            and destinations
        ):
            add_error(
                "unresolved_authoritative_target",
                row_index,
                message=(
                    "unresolved evidence cannot be imported as authoritative"
                ),
            )
        for destination_index, destination in enumerate(destinations):
            if not isinstance(destination, Mapping):
                add_error(
                    "invalid_destination_object",
                    row_index,
                    destination_index=destination_index,
                    message="destination must be an object",
                )
                continue
            missing = [
                field
                for field in ("path", "anchor_text")
                if not isinstance(destination.get(field), str)
                or not str(destination.get(field)).strip()
            ]
            if missing:
                add_error(
                    "missing_destination_field",
                    row_index,
                    destination_index=destination_index,
                    fields=missing,
                )
                continue
            relative_text = str(destination["path"]).replace("\\", "/")
            relative = Path(relative_text)
            if (
                relative.is_absolute()
                or re.match(r"^[A-Za-z]:", relative_text) is not None
                or ".." in relative.parts
            ):
                add_error(
                    "invalid_destination_path",
                    row_index,
                    destination_index=destination_index,
                    path=relative_text,
                )
                continue
            target = (root / relative).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                add_error(
                    "invalid_destination_path",
                    row_index,
                    destination_index=destination_index,
                    path=relative_text,
                )
                continue
            if not target.is_file():
                add_error(
                    "missing_destination_path",
                    row_index,
                    destination_index=destination_index,
                    path=relative_text,
                )
                continue
            if target.suffix.casefold() != ".md":
                add_error(
                    "invalid_destination_path",
                    row_index,
                    destination_index=destination_index,
                    path=relative_text,
                    message="destination must be Markdown",
                )
                continue
            if target not in heading_cache:
                heading_cache[target] = _destination_heading_index(target)
            anchor_text = str(destination["anchor_text"])
            occurrences = heading_cache[target].get(anchor_text, [])
            if not occurrences:
                add_error(
                    "missing_destination_anchor",
                    row_index,
                    destination_index=destination_index,
                    path=relative_text,
                    anchor_text=anchor_text,
                )
                continue
            raw_occurrence = destination.get("anchor_occurrence")
            if raw_occurrence is None:
                if len(occurrences) > 1:
                    add_error(
                        "ambiguous_destination_anchor",
                        row_index,
                        destination_index=destination_index,
                        path=relative_text,
                        anchor_text=anchor_text,
                        occurrences=len(occurrences),
                    )
                else:
                    add_error(
                        "missing_anchor_occurrence",
                        row_index,
                        destination_index=destination_index,
                        path=relative_text,
                        anchor_text=anchor_text,
                    )
                continue
            if (
                isinstance(raw_occurrence, bool)
                or not isinstance(raw_occurrence, int)
                or raw_occurrence < 1
            ):
                add_error(
                    "invalid_anchor_occurrence",
                    row_index,
                    destination_index=destination_index,
                    path=relative_text,
                    anchor_text=anchor_text,
                    value=raw_occurrence,
                )
                continue
            if raw_occurrence > len(occurrences):
                add_error(
                    "anchor_occurrence_out_of_range",
                    row_index,
                    destination_index=destination_index,
                    path=relative_text,
                    anchor_text=anchor_text,
                    anchor_occurrence=raw_occurrence,
                    occurrences=len(occurrences),
                )
    return errors


def load_semantic_decisions(
    paths: Sequence[Path],
) -> list[Mapping[str, object]]:
    """Load JSON arrays or JSONL objects in argument order."""

    decisions: list[Mapping[str, object]] = []
    for path in paths:
        text = path.resolve().read_text(encoding="utf-8")
        if path.suffix.casefold() == ".jsonl":
            parsed: object = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
        else:
            parsed = json.loads(text)
        if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
            raise ValueError(
                f"decisions file must contain a JSON array or JSONL objects: {path}"
            )
        decisions.extend(parsed)
    return decisions


def _claim_text_sha256(row: Mapping[str, object]) -> str:
    return hashlib.sha256(str(row.get("text", "")).encode("utf-8")).hexdigest()


def _local_correction_callout(
    row: Mapping[str, object], repo_root: Path
) -> dict[str, object] | None:
    """Return the exact local callout beginning at a claim's first source line."""

    relative = str(row["page"])
    page = (repo_root.resolve() / relative).resolve()
    try:
        page.relative_to(repo_root.resolve())
    except ValueError:
        return None
    if not page.is_file() or _sha256(page) != row.get("page_sha256"):
        return None
    lines = page.read_text(encoding="utf-8").splitlines()
    start_line = int(row["source_start_line"])
    if start_line < 1 or start_line > len(lines):
        return None
    match = LOCAL_CORRECTION_CALLOUT_RE.match(lines[start_line - 1])
    if match is None:
        return None
    end_line = start_line
    while end_line < len(lines) and lines[end_line].lstrip().startswith(">"):
        end_line += 1
    callout_text = "\n".join(lines[start_line - 1 : end_line])
    correction_ids = sorted(set(CORRECTION_ID_RE.findall(callout_text)))
    return {
        "kind": "local_callout",
        "path": relative,
        "page_sha256": str(row["page_sha256"]),
        "start_line": start_line,
        "end_line": end_line,
        "callout_type": match.group("callout_type").casefold(),
        "correction_ids": correction_ids,
    }


def _validate_exact_file_evidence(
    evidence: Mapping[str, object],
    repo_root: Path,
) -> str | None:
    relative_text = evidence.get("path")
    if not isinstance(relative_text, str) or not relative_text:
        return "evidence path is missing"
    relative = Path(relative_text.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        return "evidence path is not repository-relative"
    path = (repo_root.resolve() / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError:
        return "evidence path escapes repository root"
    if not path.is_file():
        return "evidence path does not exist"
    try:
        start_line = int(evidence["start_line"])
        end_line = int(evidence["end_line"])
    except (KeyError, TypeError, ValueError):
        return "evidence source span is missing or invalid"
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    if start_line < 1 or end_line < start_line or end_line > line_count:
        return "evidence source span is out of range"
    if not str(evidence.get("baseline", "")).strip():
        return "evidence baseline is missing"
    if evidence.get("sha256") != _sha256(path):
        return "evidence file hash does not match"
    if not str(evidence.get("supports", "")).strip():
        return "evidence does not explain exact claim support"
    return None


def validate_claim_decisions(
    claim_rows: Sequence[Mapping[str, object]],
    claim_decisions: Sequence[Mapping[str, object]],
    repo_root: Path,
    correction_catalog: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Validate exact one-to-one legacy claim decisions and evidence boundaries."""

    rows_by_id = {
        str(row["id"]): row
        for row in claim_rows
        if row.get("kind") == "claim_candidate"
    }
    errors: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(
        code: str,
        *,
        claim_id: object | None = None,
        decision_index: int | None = None,
        **details: object,
    ) -> None:
        error: dict[str, object] = {"code": code}
        if claim_id is not None:
            error["claim_id"] = claim_id
        if decision_index is not None:
            error["decision_index"] = decision_index
        error.update(details)
        errors.append(error)

    for decision_index, decision in enumerate(claim_decisions):
        claim_id = decision.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id:
            add(
                "missing_claim_id",
                decision_index=decision_index,
            )
            continue
        if claim_id in seen:
            add(
                "duplicate_claim_decision",
                claim_id=claim_id,
                decision_index=decision_index,
            )
            continue
        seen.add(claim_id)
        row = rows_by_id.get(claim_id)
        if row is None:
            add(
                "orphan_claim_decision",
                claim_id=claim_id,
                decision_index=decision_index,
            )
            continue

        identity_checks = (
            ("page", str(row["page"]), "claim_page_mismatch"),
            (
                "page_sha256",
                str(row["page_sha256"]),
                "claim_page_hash_mismatch",
            ),
            (
                "start_line",
                int(row["source_start_line"]),
                "claim_span_mismatch",
            ),
            (
                "end_line",
                int(row["source_end_line"]),
                "claim_span_mismatch",
            ),
            (
                "source_start_column",
                row["source_start_column"],
                "claim_column_span_mismatch",
            ),
            (
                "source_end_column",
                row["source_end_column"],
                "claim_column_span_mismatch",
            ),
            (
                "claim_text_sha256",
                _claim_text_sha256(row),
                "claim_text_hash_mismatch",
            ),
        )
        emitted_identity_codes: set[str] = set()
        for field, expected, code in identity_checks:
            if decision.get(field) != expected and code not in emitted_identity_codes:
                add(
                    code,
                    claim_id=claim_id,
                    decision_index=decision_index,
                    field=field,
                    expected=expected,
                    actual=decision.get(field),
                )
                emitted_identity_codes.add(code)

        status = str(decision.get("status", ""))
        action = str(decision.get("action", ""))
        role = decision.get("destination_role")
        blocker = str(decision.get("blocker", "")).strip()
        evidence = decision.get("evidence")
        if not isinstance(evidence, list) or not all(
            isinstance(item, Mapping) for item in evidence
        ):
            add(
                "invalid_claim_evidence",
                claim_id=claim_id,
                decision_index=decision_index,
            )
            evidence_items: list[Mapping[str, object]] = []
        else:
            evidence_items = list(evidence)
        raw_correction_ids = decision.get("correction_ids")
        if not isinstance(raw_correction_ids, list) or not all(
            isinstance(item, str) for item in raw_correction_ids
        ):
            add(
                "invalid_correction_ids",
                claim_id=claim_id,
                decision_index=decision_index,
            )
            correction_ids: list[str] = []
        else:
            correction_ids = list(raw_correction_ids)

        if status == "unresolved":
            if role == "authoritative_import":
                add(
                    "unresolved_claim_imported",
                    claim_id=claim_id,
                    decision_index=decision_index,
                )
            if action != "retain-quarantined":
                add(
                    "invalid_unresolved_claim_action",
                    claim_id=claim_id,
                    decision_index=decision_index,
                    action=action,
                )
            if not blocker:
                add(
                    "missing_claim_blocker",
                    claim_id=claim_id,
                    decision_index=decision_index,
                )
        elif status == "corrected":
            if role == "authoritative_import":
                add(
                    "corrected_claim_imported",
                    claim_id=claim_id,
                    decision_index=decision_index,
                )
            if action != "retain-with-correction":
                add(
                    "invalid_corrected_claim_action",
                    claim_id=claim_id,
                    decision_index=decision_index,
                    action=action,
                )
            if not correction_ids:
                add(
                    "missing_correction_id",
                    claim_id=claim_id,
                    decision_index=decision_index,
                )
            invalid_ids = sorted(
                {
                    correction_id
                    for correction_id in correction_ids
                    if correction_id not in correction_catalog
                }
            )
            for correction_id in invalid_ids:
                add(
                    "invalid_correction_id",
                    claim_id=claim_id,
                    decision_index=decision_index,
                    correction_id=correction_id,
                )
            local_evidence = [
                item
                for item in evidence_items
                if item.get("kind") == "local_callout"
            ]
            actual_callout = _local_correction_callout(row, repo_root)
            if (
                actual_callout is None
                or not local_evidence
                or not any(dict(item) == actual_callout for item in local_evidence)
            ):
                add(
                    "corrected_without_local_callout",
                    claim_id=claim_id,
                    decision_index=decision_index,
                )
            actual_correction_ids = (
                list(actual_callout["correction_ids"])
                if actual_callout is not None
                and isinstance(
                    actual_callout.get("correction_ids"), list
                )
                else []
            )
            if sorted(correction_ids) != sorted(actual_correction_ids):
                add(
                    "correction_id_callout_mismatch",
                    claim_id=claim_id,
                    decision_index=decision_index,
                    expected=actual_correction_ids,
                    actual=correction_ids,
                )
            known_ids = [
                correction_id
                for correction_id in correction_ids
                if correction_id in correction_catalog
            ]
            for correction_id in known_ids:
                expected = correction_catalog[correction_id]
                expected_report = {
                    "kind": "correction_report",
                    "correction_id": correction_id,
                    "path": expected["path"],
                    "anchor_text": expected["anchor_text"],
                    "anchor_occurrence": expected["anchor_occurrence"],
                    "source_line": expected["source_line"],
                    "report_sha256": expected["report_sha256"],
                }
                if not any(
                    dict(item) == expected_report
                    for item in evidence_items
                    if item.get("kind") == "correction_report"
                ):
                    add(
                        "missing_correction_report_evidence",
                        claim_id=claim_id,
                        decision_index=decision_index,
                        correction_id=correction_id,
                    )
        elif status in {"verified-current", "verified-historical"}:
            if role != "authoritative_import":
                add(
                    "verified_claim_not_authoritative",
                    claim_id=claim_id,
                    decision_index=decision_index,
                )
            if action != "migrate":
                add(
                    "invalid_verified_claim_action",
                    claim_id=claim_id,
                    decision_index=decision_index,
                    action=action,
                )
            if not evidence_items:
                add(
                    "missing_verified_claim_evidence",
                    claim_id=claim_id,
                    decision_index=decision_index,
                )
            for item in evidence_items:
                kind = item.get("kind")
                if kind in {"range_decision", "source_locator"}:
                    add(
                        "non_claim_specific_verified_evidence",
                        claim_id=claim_id,
                        decision_index=decision_index,
                        evidence_kind=kind,
                    )
                    continue
                if kind not in {"source", "historical", "runtime"}:
                    add(
                        "invalid_verified_claim_evidence",
                        claim_id=claim_id,
                        decision_index=decision_index,
                        evidence_kind=kind,
                    )
                    continue
                if kind in {"source", "historical"}:
                    evidence_error = _validate_exact_file_evidence(
                        item, repo_root
                    )
                    if evidence_error is not None:
                        add(
                            "invalid_verified_claim_evidence",
                            claim_id=claim_id,
                            decision_index=decision_index,
                            evidence_kind=kind,
                            message=evidence_error,
                        )
                else:
                    artifact = item.get("artifact_path")
                    artifact_hash = item.get("artifact_sha256")
                    if (
                        not isinstance(artifact, str)
                        or not artifact
                        or not isinstance(artifact_hash, str)
                        or not artifact_hash
                        or not str(item.get("baseline", "")).strip()
                    ):
                        add(
                            "invalid_verified_claim_evidence",
                            claim_id=claim_id,
                            decision_index=decision_index,
                            evidence_kind=kind,
                        )
        else:
            add(
                "invalid_claim_status",
                claim_id=claim_id,
                decision_index=decision_index,
                status=status,
            )

    for claim_id in rows_by_id:
        if claim_id not in seen:
            add("missing_claim_decision", claim_id=claim_id)
    return errors


def load_correction_dispositions(path: Path) -> dict[str, object]:
    """Load the catalog-wide semantic disposition document."""

    parsed = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(
            f"correction dispositions must be a JSON object: {path}"
        )
    return parsed


def validate_correction_dispositions(
    document: Mapping[str, object],
    correction_catalog: Mapping[str, Mapping[str, object]],
    claim_rows: Sequence[Mapping[str, object]],
    claim_decisions: Sequence[Mapping[str, object]],
    repo_root: Path,
) -> list[dict[str, object]]:
    """Validate one semantic disposition for every correction catalog ID."""

    errors: list[dict[str, object]] = []

    def add(
        code: str,
        *,
        correction_id: object | None = None,
        disposition_index: int | None = None,
        **details: object,
    ) -> None:
        error: dict[str, object] = {"code": code}
        if correction_id is not None:
            error["correction_id"] = correction_id
        if disposition_index is not None:
            error["disposition_index"] = disposition_index
        error.update(details)
        errors.append(error)

    if document.get("schema_version") != 1:
        add(
            "invalid_correction_disposition_schema",
            actual=document.get("schema_version"),
        )
    raw_dispositions = document.get("dispositions")
    if not isinstance(raw_dispositions, list) or not all(
        isinstance(item, Mapping) for item in raw_dispositions
    ):
        add("invalid_correction_dispositions")
        raw_dispositions = []

    rows_by_id = {
        str(row["id"]): row
        for row in claim_rows
        if row.get("kind") == "claim_candidate"
    }
    decisions_by_id = {
        str(decision["claim_id"]): decision
        for decision in claim_decisions
        if isinstance(decision.get("claim_id"), str)
    }
    local_claims_by_correction: defaultdict[str, set[str]] = defaultdict(
        set
    )
    for decision in claim_decisions:
        claim_id = decision.get("claim_id")
        correction_ids = decision.get("correction_ids")
        if not isinstance(claim_id, str) or not isinstance(
            correction_ids, list
        ):
            continue
        for correction_id in correction_ids:
            if isinstance(correction_id, str):
                local_claims_by_correction[correction_id].add(claim_id)

    seen: set[str] = set()
    for disposition_index, raw in enumerate(raw_dispositions):
        disposition = dict(raw)
        correction_id = disposition.get("correction_id")
        if not isinstance(correction_id, str) or not correction_id:
            add(
                "missing_correction_disposition_id",
                disposition_index=disposition_index,
            )
            continue
        if correction_id in seen:
            add(
                "duplicate_correction_disposition",
                correction_id=correction_id,
                disposition_index=disposition_index,
            )
            continue
        seen.add(correction_id)
        if correction_id not in correction_catalog:
            add(
                "orphan_correction_disposition",
                correction_id=correction_id,
                disposition_index=disposition_index,
            )
            continue

        kind = disposition.get("disposition")
        if kind == "local-claim-corrected":
            raw_refs = disposition.get("claim_refs")
            if not isinstance(raw_refs, list) or not raw_refs or not all(
                isinstance(item, Mapping) for item in raw_refs
            ):
                add(
                    "local_disposition_missing_claim_ref",
                    correction_id=correction_id,
                    disposition_index=disposition_index,
                )
                continue
            referenced_claim_ids: set[str] = set()
            for ref_index, raw_ref in enumerate(raw_refs):
                ref = dict(raw_ref)
                claim_id = ref.get("claim_id")
                if not isinstance(claim_id, str) or not claim_id:
                    add(
                        "local_disposition_invalid_claim_ref",
                        correction_id=correction_id,
                        disposition_index=disposition_index,
                        ref_index=ref_index,
                    )
                    continue
                referenced_claim_ids.add(claim_id)
                row = rows_by_id.get(claim_id)
                decision = decisions_by_id.get(claim_id)
                if row is None or decision is None:
                    add(
                        "local_disposition_orphan_claim_ref",
                        correction_id=correction_id,
                        disposition_index=disposition_index,
                        ref_index=ref_index,
                        claim_id=claim_id,
                    )
                    continue
                expected_identity = {
                    "claim_id": claim_id,
                    "page": str(row["page"]),
                    "page_sha256": str(row["page_sha256"]),
                    "start_line": int(row["source_start_line"]),
                    "end_line": int(row["source_end_line"]),
                    "source_start_column": row["source_start_column"],
                    "source_end_column": row["source_end_column"],
                    "claim_text_sha256": _claim_text_sha256(row),
                }
                if any(
                    ref.get(field) != expected
                    for field, expected in expected_identity.items()
                ):
                    add(
                        "local_disposition_claim_identity_mismatch",
                        correction_id=correction_id,
                        disposition_index=disposition_index,
                        ref_index=ref_index,
                        claim_id=claim_id,
                    )
                    continue
                if decision.get("status") != "corrected":
                    add(
                        "disposition_claim_not_corrected",
                        correction_id=correction_id,
                        disposition_index=disposition_index,
                        ref_index=ref_index,
                        claim_id=claim_id,
                    )
                    continue
                decision_ids = decision.get("correction_ids")
                if not isinstance(decision_ids, list) or (
                    correction_id not in decision_ids
                ):
                    add(
                        "disposition_claim_correction_id_mismatch",
                        correction_id=correction_id,
                        disposition_index=disposition_index,
                        ref_index=ref_index,
                        claim_id=claim_id,
                    )
                    continue
                actual_callout = _local_correction_callout(row, repo_root)
                local_evidence = [
                    item
                    for item in decision.get("evidence", [])
                    if isinstance(item, Mapping)
                    and item.get("kind") == "local_callout"
                ]
                if (
                    actual_callout is None
                    or correction_id
                    not in actual_callout.get("correction_ids", [])
                    or not any(
                        dict(item) == actual_callout
                        for item in local_evidence
                    )
                ):
                    add(
                        "disposition_claim_missing_exact_callout",
                        correction_id=correction_id,
                        disposition_index=disposition_index,
                        ref_index=ref_index,
                        claim_id=claim_id,
                    )
            missing_refs = sorted(
                local_claims_by_correction.get(correction_id, set())
                - referenced_claim_ids
            )
            for claim_id in missing_refs:
                add(
                    "disposition_missing_local_claim_ref",
                    correction_id=correction_id,
                    disposition_index=disposition_index,
                    claim_id=claim_id,
                )
        elif kind == "catalog-only/no-local-target":
            if not str(disposition.get("reason", "")).strip():
                add(
                    "catalog_only_missing_reason",
                    correction_id=correction_id,
                    disposition_index=disposition_index,
                )
            if disposition.get("false_assertion_audit") != "none-found":
                add(
                    "catalog_only_false_assertion_not_cleared",
                    correction_id=correction_id,
                    disposition_index=disposition_index,
                )
            accurate_spans = disposition.get("accurate_spans")
            if not isinstance(accurate_spans, list) or not accurate_spans:
                add(
                    "catalog_only_missing_accurate_span",
                    correction_id=correction_id,
                    disposition_index=disposition_index,
                )
            else:
                for span_index, span in enumerate(accurate_spans):
                    if not isinstance(span, Mapping):
                        add(
                            "catalog_only_invalid_accurate_span",
                            correction_id=correction_id,
                            disposition_index=disposition_index,
                            span_index=span_index,
                        )
                        continue
                    evidence_error = _validate_exact_file_evidence(
                        span, repo_root
                    )
                    if evidence_error is not None:
                        add(
                            "catalog_only_invalid_accurate_span",
                            correction_id=correction_id,
                            disposition_index=disposition_index,
                            span_index=span_index,
                            message=evidence_error,
                        )
            if local_claims_by_correction.get(correction_id):
                add(
                    "catalog_only_has_local_claim",
                    correction_id=correction_id,
                    disposition_index=disposition_index,
                )
        else:
            add(
                "invalid_correction_disposition",
                correction_id=correction_id,
                disposition_index=disposition_index,
                actual=kind,
            )

    for correction_id in sorted(set(correction_catalog).difference(seen)):
        add(
            "missing_correction_disposition",
            correction_id=correction_id,
        )
    return errors


def _nearest_heading_destination(
    row: Mapping[str, object], repo_root: Path
) -> dict[str, object]:
    page = (repo_root.resolve() / str(row["page"])).resolve()
    start_line = int(row["source_start_line"])
    headings: list[tuple[int, str, int]] = []
    occurrences: Counter[str] = Counter()
    for line_number, line in enumerate(
        page.read_text(encoding="utf-8").splitlines(), 1
    ):
        match = HEADING_RE.match(line)
        if match is None:
            continue
        title = match.group("title").strip()
        occurrences[title] += 1
        headings.append((line_number, title, occurrences[title]))
    if not headings:
        raise ValueError(f"legacy page has no H2/H3 destination: {row['page']}")
    eligible = [heading for heading in headings if heading[0] <= start_line]
    _, anchor_text, anchor_occurrence = (
        eligible[-1] if eligible else headings[0]
    )
    return {
        "path": str(row["page"]),
        "anchor_text": anchor_text,
        "anchor_occurrence": anchor_occurrence,
    }


def _deduplicate_destinations(
    destinations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[tuple[object, object, object]] = set()
    for destination in destinations:
        key = (
            destination.get("path"),
            destination.get("anchor_text"),
            destination.get("anchor_occurrence"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(destination))
    return result


def generate_legacy_claim_decisions(
    claim_rows: Sequence[Mapping[str, object]],
    range_decisions: Sequence[Mapping[str, object]],
    correction_catalog: Mapping[str, Mapping[str, object]],
    repo_root: Path,
) -> list[dict[str, object]]:
    """Generate conservative exact decisions without laundering range evidence."""

    correction_destinations: defaultdict[
        str, list[Mapping[str, object]]
    ] = defaultdict(list)
    for decision in range_decisions:
        correction_ids = set(
            CORRECTION_ID_RE.findall(str(decision.get("notes", "")))
        )
        destinations = decision.get("destinations")
        if not isinstance(destinations, list):
            continue
        for correction_id in correction_ids:
            correction_destinations[correction_id].extend(
                destination
                for destination in destinations
                if isinstance(destination, Mapping)
            )

    generated: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for row in claim_rows:
        if row.get("kind") != "claim_candidate":
            continue
        claim_id = str(row["id"])
        if claim_id in seen_ids:
            raise ValueError(f"duplicate claim row: {claim_id}")
        seen_ids.add(claim_id)
        matches = [
            decision
            for decision in range_decisions
            if decision.get("page") == row.get("page")
            and int(decision["start_line"])
            <= int(row["source_start_line"])
            <= int(decision["end_line"])
        ]
        if len(matches) > 1:
            raise ValueError(
                f"overlapping range context for claim {claim_id}"
            )
        range_context = [
            {
                "page": str(decision["page"]),
                "start_line": int(decision["start_line"]),
                "end_line": int(decision["end_line"]),
                "status": str(decision["status"]),
                "correction_ids": sorted(
                    set(
                        CORRECTION_ID_RE.findall(
                            str(decision.get("notes", ""))
                        )
                    )
                ),
            }
            for decision in matches
        ]
        callout = _local_correction_callout(row, repo_root)
        correction_ids = (
            list(callout["correction_ids"])
            if callout is not None
            and isinstance(callout.get("correction_ids"), list)
            else []
        )
        known_correction_ids = [
            correction_id
            for correction_id in correction_ids
            if correction_id in correction_catalog
        ]
        is_corrected = bool(
            callout is not None
            and correction_ids
            and len(known_correction_ids) == len(correction_ids)
        )

        matched_destinations = [
            destination
            for decision in matches
            for destination in decision.get("destinations", [])
            if isinstance(destination, Mapping)
        ]
        if is_corrected and not matched_destinations:
            matched_destinations = [
                destination
                for correction_id in known_correction_ids
                for destination in correction_destinations.get(
                    correction_id, []
                )
            ]
        if matched_destinations:
            destinations = _deduplicate_destinations(matched_destinations)
            destination_role = "candidate_only"
        else:
            destinations = [
                _nearest_heading_destination(row, repo_root)
            ]
            destination_role = "retain_quarantined"

        decision: dict[str, object] = {
            "claim_id": claim_id,
            "page": str(row["page"]),
            "page_sha256": str(row["page_sha256"]),
            "start_line": int(row["source_start_line"]),
            "end_line": int(row["source_end_line"]),
            "source_start_column": row["source_start_column"],
            "source_end_column": row["source_end_column"],
            "claim_text_sha256": _claim_text_sha256(row),
            "claimed_baseline": "unknown",
            "destinations": destinations,
            "destination_role": destination_role,
            "range_context": range_context,
        }
        if is_corrected:
            report_evidence = [
                {
                    "kind": "correction_report",
                    "correction_id": correction_id,
                    "path": correction_catalog[correction_id]["path"],
                    "anchor_text": correction_catalog[correction_id][
                        "anchor_text"
                    ],
                    "anchor_occurrence": correction_catalog[correction_id][
                        "anchor_occurrence"
                    ],
                    "source_line": correction_catalog[correction_id][
                        "source_line"
                    ],
                    "report_sha256": correction_catalog[correction_id][
                        "report_sha256"
                    ],
                }
                for correction_id in known_correction_ids
            ]
            decision.update(
                {
                    "status": "corrected",
                    "current_result": (
                        "local_correction_callout_and_report"
                    ),
                    "action": "retain-with-correction",
                    "blocker": "",
                    "evidence": [callout, *report_evidence],
                    "correction_ids": known_correction_ids,
                    "notes": (
                        "The exact local callout points to correction "
                        "evidence; the legacy wording is not imported as "
                        "authoritative course content."
                    ),
                }
            )
        else:
            blocker = (
                "A broad range decision overlaps this exact span, but "
                "broad range status and nearby locators are not "
                "claim-specific evidence; the original claim remains "
                "quarantined."
                if matches
                else (
                    "No claim-specific source, runtime, or historical "
                    "evidence is recorded for this exact span."
                )
            )
            decision.update(
                {
                    "status": "unresolved",
                    "current_result": "claim_specific_evidence_missing",
                    "action": "retain-quarantined",
                    "blocker": blocker,
                    "evidence": [],
                    "correction_ids": [],
                    "notes": (
                        "The decision is exact to this claim ID, page hash "
                        "and source span; retention does not imply semantic "
                        "verification."
                    ),
                }
            )
        generated.append(decision)
    return generated


def generate_legacy_unit_decisions(
    rows: Sequence[Mapping[str, object]],
    repo_root: Path,
) -> list[dict[str, object]]:
    """Rebuild exact structural-unit decisions from frozen stable IDs."""

    generated: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for row in rows:
        if (
            row.get("kind") == "claim_candidate"
            or bool(row.get("decision_applied"))
        ):
            continue
        unit_id = str(row["id"])
        if unit_id in seen_ids:
            raise ValueError(f"duplicate structural row: {unit_id}")
        seen_ids.add(unit_id)
        preface = row.get("section") == "(page preface)"
        generated.append(
            {
                "unit_id": unit_id,
                "status": "unresolved",
                "destinations": [
                    _nearest_heading_destination(row, repo_root)
                ],
                "destination_role": "retain_quarantined",
                "action": "retain-quarantined",
                "notes": (
                    "Exact preface structural unit retained at the nearest "
                    "auditable heading; this locator does not imply section "
                    "membership or semantic verification."
                    if preface
                    else (
                        "Exact nonclaim structural unit retained in its "
                        "audited legacy page; unresolved status does not "
                        "authorize semantic import."
                    )
                ),
            }
        )
    return generated


def _range_decision_indices(
    record: Mapping[str, object],
    source_line: int,
    decisions: Sequence[Mapping[str, object]],
) -> list[int]:
    return [
        index
        for index, decision in enumerate(decisions)
        if decision.get("page") == record.get("page")
        and int(decision["start_line"]) <= source_line <= int(decision["end_line"])
    ]


def _record_source_span(record: Mapping[str, object]) -> tuple[int, int]:
    if "source_line" in record:
        source_line = int(record["source_line"])
        return source_line, source_line
    start = int(record.get("start_line", record.get("source_line", 1)))
    end = int(record.get("end_line", start))
    return start, end


def _ledger_kind(record: Mapping[str, object]) -> str | None:
    kind = str(record["kind"])
    if kind == "heading":
        return f"heading_h{record['level']}"
    if kind == "code_fence":
        return f"code_{str(record.get('language') or 'plain')}"
    if kind == "locator":
        return "source_locator"
    if kind in {"markdown_table", "experiment", "claim_candidate"}:
        return kind
    if kind == "image":
        return f"image_{str(record.get('syntax') or 'unknown')}"
    return None


def _stable_ledger_id(
    page: str,
    page_hash: str,
    kind: str,
    start_line: int,
    end_line: int,
    source_start_column: int | None,
    source_end_column: int | None,
) -> str:
    material = "\0".join(
        (
            page,
            page_hash,
            kind,
            str(start_line),
            str(end_line),
            "" if source_start_column is None else str(source_start_column),
            "" if source_end_column is None else str(source_end_column),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _record_payload(record: Mapping[str, object]) -> dict[str, object]:
    fields_by_kind = {
        "heading": ("level", "title"),
        "code_fence": (
            "language",
            "marker",
            "figure_classification",
            "content",
            "balanced",
        ),
        "markdown_table": (
            "header_cells",
            "delimiter_cells",
            "data_rows",
            "raw",
            "column_count",
            "data_row_count",
        ),
        "image": ("syntax", "alt", "target", "title", "reference"),
        "experiment": ("heading",),
        "locator": (
            "raw",
            "path",
            "target_start_line",
            "target_end_line",
            "resolution",
            "resolved_path",
            "source_line_count",
        ),
        "claim_candidate": ("text",),
    }
    return {
        field: record[field]
        for field in fields_by_kind.get(str(record["kind"]), ())
        if field in record
    }


def _validate_decision(
    decision: Mapping[str, object], category: str
) -> None:
    if "destination" in decision:
        raise ValueError(
            "bare destination strings are not canonical; use destinations[]"
        )
    status = str(decision.get("status", ""))
    if status not in VALID_AUDIT_STATUSES:
        raise ValueError(f"invalid audit status: {status!r}")
    if "destinations" in decision:
        destinations = decision["destinations"]
        if not isinstance(destinations, list) or not all(
            isinstance(destination, Mapping)
            for destination in destinations
        ):
            raise ValueError("destinations must be an array of objects")
        role = decision.get("destination_role")
        if role not in VALID_DESTINATION_ROLES.difference({"unassigned"}):
            raise ValueError(
                "a decision with destinations requires a non-unassigned "
                "destination_role"
            )
    elif (
        "destination_role" in decision
        and decision["destination_role"] != "unassigned"
    ):
        raise ValueError(
            "a non-unassigned destination_role requires destinations[]"
        )
    if "start_line" in decision and "end_line" in decision:
        if int(decision["start_line"]) > int(decision["end_line"]):
            raise ValueError(f"invalid audit decision line range: {decision!r}")
    if category == "range":
        required = {"page", "start_line", "end_line"}
    elif category == "claim":
        required = (
            set()
            if "claim_id" in decision
            else {"page", "start_line", "end_line"}
        )
    elif category == "unit":
        required = (
            set()
            if "unit_id" in decision
            else {"page", "start_line", "end_line", "kind"}
        )
    else:
        raise ValueError(f"unknown decision category: {category}")
    missing = sorted(required.difference(decision))
    if missing:
        raise ValueError(
            f"invalid {category} decision selector; missing {missing}: {decision!r}"
        )


def _apply_decision(
    row: dict[str, object], decision: Mapping[str, object]
) -> None:
    for field in (
        "claimed_baseline",
        "current_result",
        "status",
        "action",
    ):
        if field in decision:
            row[field] = str(decision[field])
    if "destinations" in decision:
        row["destinations"] = [
            dict(destination)
            for destination in decision["destinations"]  # type: ignore[union-attr]
        ]
        row["destination_role"] = str(decision["destination_role"])
    for field in ("evidence", "range_context"):
        if field in decision:
            row[field] = [
                dict(item)
                for item in decision[field]  # type: ignore[union-attr]
            ]
    if "correction_ids" in decision:
        row["correction_ids"] = [
            str(item)
            for item in decision["correction_ids"]  # type: ignore[union-attr]
        ]
    if "blocker" in decision:
        row["blocker"] = str(decision["blocker"])
    if "claim_text_sha256" in decision:
        row["claim_text_sha256"] = str(decision["claim_text_sha256"])
    decision_notes = str(decision.get("notes", "")).strip()
    if decision_notes:
        row["notes"] = (
            f"{row['notes']}; {decision_notes}" if row["notes"] else decision_notes
        )


def _exact_decision_indices(
    row: Mapping[str, object],
    decisions: Sequence[Mapping[str, object]],
    *,
    id_field: str,
    require_kind: bool,
) -> list[int]:
    matches: list[int] = []
    for index, decision in enumerate(decisions):
        if id_field in decision:
            if decision[id_field] == row["id"]:
                matches.append(index)
            continue
        if (
            decision.get("page") != row["page"]
            or int(decision.get("start_line", -1)) != row["source_start_line"]
            or int(decision.get("end_line", -1)) != row["source_end_line"]
        ):
            continue
        if require_kind and decision.get("kind") != row["kind"]:
            continue
        if (
            "source_start_column" in decision
            and decision["source_start_column"] != row["source_start_column"]
        ):
            continue
        if (
            "source_end_column" in decision
            and decision["source_end_column"] != row["source_end_column"]
        ):
            continue
        matches.append(index)
    return matches


def _decision_selector(
    decision: Mapping[str, object], *, id_field: str | None = None
) -> str:
    if id_field is not None and id_field in decision:
        return f"{id_field}:{decision[id_field]}"
    selector = (
        f"{decision.get('page')}:{decision.get('start_line')}-"
        f"{decision.get('end_line')}"
    )
    if "kind" in decision:
        selector += f":{decision['kind']}"
    if "source_start_column" in decision:
        selector += (
            f":{decision['source_start_column']}-"
            f"{decision.get('source_end_column')}"
        )
    return selector


def build_ledger_rows(
    records: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]] = (),
    claim_decisions: Sequence[Mapping[str, object]] = (),
    unit_decisions: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    """Build canonical audit rows; only these rows may determine status totals."""

    for decision in decisions:
        _validate_decision(decision, "range")
    for decision in claim_decisions:
        _validate_decision(decision, "claim")
    for decision in unit_decisions:
        _validate_decision(decision, "unit")

    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_range_decisions: set[int] = set()
    claim_decision_matches = [0] * len(claim_decisions)
    unit_decision_matches = [0] * len(unit_decisions)
    for record in records:
        kind = _ledger_kind(record)
        if kind is None:
            continue
        page = str(record["page"])
        page_hash = str(
            record.get("page_sha256", record.get("sha256", hashlib.sha256(page.encode()).hexdigest()))
        )
        start_line, end_line = _record_source_span(record)
        source_start_column = (
            int(record["source_start_column"])
            if "source_start_column" in record
            else None
        )
        source_end_column = (
            int(record["source_end_column"])
            if "source_end_column" in record
            else None
        )
        row_id = _stable_ledger_id(
            page,
            page_hash,
            kind,
            start_line,
            end_line,
            source_start_column,
            source_end_column,
        )
        if row_id in seen_ids:
            raise ValueError(f"duplicate stable ledger ID: {row_id}")
        seen_ids.add(row_id)

        heading_path = record.get("heading_path") or []
        section = " > ".join(str(item) for item in heading_path) or "(page preface)"
        if record["kind"] == "code_fence":
            current_result = "not_executed"
            notes = (
                f"lines {start_line}-{end_line}; "
                f"balanced={str(bool(record.get('balanced'))).lower()}"
            )
        elif record["kind"] == "locator":
            current_result = str(record.get("resolution", "unresolved"))
            notes = ""
        else:
            current_result = "not_semantically_audited"
            notes = ""
        row: dict[str, object] = {
            "id": row_id,
            "page": page,
            "page_sha256": page_hash,
            "section": section,
            "source_start_line": start_line,
            "source_end_line": end_line,
            "source_line": start_line,
            "source_start_column": source_start_column,
            "source_end_column": source_end_column,
            "kind": kind,
            "locator": (
                str(record.get("raw", "")) if record["kind"] == "locator" else ""
            ),
            "text": str(record.get("text", "")),
            "payload": _record_payload(record),
            "claimed_baseline": "unknown",
            "current_result": current_result,
            "status": "unresolved",
            "destinations": [],
            "destination_role": "unassigned",
            "action": "audit",
            "notes": notes,
            "decision_applied": False,
            "decision_category": None,
            "decision_selector": None,
            "claim_text_sha256": (
                _claim_text_sha256(
                    {"text": str(record.get("text", ""))}
                )
                if record["kind"] == "claim_candidate"
                else ""
            ),
            "evidence": [],
            "correction_ids": [],
            "blocker": "",
            "range_context": [],
        }
        range_indices = _range_decision_indices(record, start_line, decisions)
        seen_range_decisions.update(range_indices)
        selected: tuple[str, int, Mapping[str, object], str] | None = None
        if record["kind"] == "claim_candidate":
            claim_indices = _exact_decision_indices(
                row,
                claim_decisions,
                id_field="claim_id",
                require_kind=False,
            )
            if len(claim_indices) > 1:
                raise ValueError(f"overlapping claim decisions for {row['id']}")
            if claim_indices:
                index = claim_indices[0]
                claim_decision_matches[index] += 1
                decision = claim_decisions[index]
                selected = (
                    "claim",
                    index,
                    decision,
                    _decision_selector(decision, id_field="claim_id"),
                )
        else:
            unit_indices = _exact_decision_indices(
                row,
                unit_decisions,
                id_field="unit_id",
                require_kind=True,
            )
            if len(unit_indices) > 1:
                raise ValueError(f"overlapping unit decisions for {row['id']}")
            semantic_range_indices = [
                index
                for index in range_indices
                if kind in {"heading_h2", "heading_h3"}
                and decisions[index].get("status") != "unresolved"
            ]
            if len(semantic_range_indices) > 1:
                raise ValueError(
                    f"overlapping audit decisions for {row['page']}:{start_line}"
                )
            if unit_indices and semantic_range_indices:
                raise ValueError(
                    f"overlapping range and unit decisions for {row['id']}"
                )
            if unit_indices:
                index = unit_indices[0]
                unit_decision_matches[index] += 1
                decision = unit_decisions[index]
                selected = (
                    "unit",
                    index,
                    decision,
                    _decision_selector(decision, id_field="unit_id"),
                )
            elif semantic_range_indices:
                index = semantic_range_indices[0]
                decision = decisions[index]
                selected = (
                    "range",
                    index,
                    decision,
                    _decision_selector(decision),
                )
        if selected is not None:
            category, _, decision, selector = selected
            _apply_decision(row, decision)
            row["decision_applied"] = True
            row["decision_category"] = category
            row["decision_selector"] = selector
        rows.append(row)

    unmatched_ranges = [
        index for index in range(len(decisions)) if index not in seen_range_decisions
    ]
    if unmatched_ranges:
        raise ValueError(f"unmatched range decision indices: {unmatched_ranges}")
    unmatched_claims = [
        index for index, count in enumerate(claim_decision_matches) if count == 0
    ]
    if unmatched_claims:
        raise ValueError(f"unmatched claim decision indices: {unmatched_claims}")
    ambiguous_claims = [
        index for index, count in enumerate(claim_decision_matches) if count > 1
    ]
    if ambiguous_claims:
        raise ValueError(f"claim decisions matched multiple rows: {ambiguous_claims}")
    unmatched_units = [
        index for index, count in enumerate(unit_decision_matches) if count == 0
    ]
    if unmatched_units:
        raise ValueError(f"unmatched unit decision indices: {unmatched_units}")
    ambiguous_units = [
        index for index, count in enumerate(unit_decision_matches) if count > 1
    ]
    if ambiguous_units:
        raise ValueError(f"unit decisions matched multiple rows: {ambiguous_units}")
    validate_ledger_rows(rows)
    return rows


def validate_ledger_rows(rows: Sequence[Mapping[str, object]]) -> None:
    """Validate the canonical schema before it can participate in arithmetic."""

    required = {
        "id",
        "page",
        "page_sha256",
        "source_start_line",
        "source_end_line",
        "source_start_column",
        "source_end_column",
        "kind",
        "status",
        "destinations",
        "destination_role",
        "text",
        "payload",
        "decision_applied",
        "decision_category",
        "decision_selector",
        "claim_text_sha256",
        "evidence",
        "correction_ids",
        "blocker",
        "range_context",
    }
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        try:
            missing = sorted(required.difference(row))
            if missing:
                raise ValueError(f"missing fields {missing}")
            page = str(row["page"])
            page_hash = str(row["page_sha256"])
            kind = str(row["kind"])
            row_id = str(row["id"])
            if not page or not kind:
                raise ValueError("page and kind must be non-empty")
            if re.fullmatch(r"[0-9a-f]{64}", page_hash) is None:
                raise ValueError("page_sha256 must be a lowercase SHA-256 digest")
            if str(row["status"]) not in VALID_AUDIT_STATUSES:
                raise ValueError(f"invalid status {row['status']!r}")
            if "destination" in row:
                raise ValueError(
                    "bare destination strings are not canonical"
                )
            destinations = row["destinations"]
            if not isinstance(destinations, list) or not all(
                isinstance(destination, Mapping)
                for destination in destinations
            ):
                raise ValueError(
                    "destinations must be an array of objects"
                )
            destination_role = row["destination_role"]
            if destination_role not in VALID_DESTINATION_ROLES:
                raise ValueError(
                    f"invalid destination_role {destination_role!r}"
                )
            if destinations and destination_role == "unassigned":
                raise ValueError(
                    "rows with destinations cannot be unassigned"
                )
            if not destinations and destination_role != "unassigned":
                raise ValueError(
                    "rows without destinations must be unassigned"
                )
            start_line = int(row["source_start_line"])
            end_line = int(row["source_end_line"])
            if start_line < 1 or end_line < start_line:
                raise ValueError("invalid source line span")
            start_column = row["source_start_column"]
            end_column = row["source_end_column"]
            if (start_column is None) != (end_column is None):
                raise ValueError("source columns must both be present or absent")
            if start_column is not None:
                start_column = int(start_column)
                end_column = int(end_column)
                if start_column < 1 or end_column < start_column:
                    raise ValueError("invalid source column span")
            if not isinstance(row["payload"], Mapping):
                raise ValueError("payload must be an object")
            claim_text_hash = row["claim_text_sha256"]
            if kind == "claim_candidate":
                if claim_text_hash != _claim_text_sha256(row):
                    raise ValueError(
                        "claim_text_sha256 does not match claim text"
                    )
            elif claim_text_hash != "":
                raise ValueError(
                    "non-claim row carries a claim_text_sha256"
                )
            if not isinstance(row["evidence"], list) or not all(
                isinstance(item, Mapping) for item in row["evidence"]
            ):
                raise ValueError("evidence must be an array of objects")
            if not isinstance(row["correction_ids"], list) or not all(
                isinstance(item, str) for item in row["correction_ids"]
            ):
                raise ValueError("correction_ids must be an array of strings")
            if not isinstance(row["blocker"], str):
                raise ValueError("blocker must be a string")
            if not isinstance(row["range_context"], list) or not all(
                isinstance(item, Mapping) for item in row["range_context"]
            ):
                raise ValueError("range_context must be an array of objects")
            if not isinstance(row["decision_applied"], bool):
                raise ValueError("decision_applied must be boolean")
            category = row["decision_category"]
            selector = row["decision_selector"]
            if row["decision_applied"]:
                if category not in {"range", "claim", "unit"}:
                    raise ValueError("assigned row has invalid decision_category")
                if not isinstance(selector, str) or not selector:
                    raise ValueError("assigned row has no decision_selector")
            elif category is not None or selector is not None:
                raise ValueError("unassigned row carries decision provenance")
            expected_id = _stable_ledger_id(
                page,
                page_hash,
                kind,
                start_line,
                end_line,
                start_column,
                end_column,
            )
            if row_id != expected_id:
                raise ValueError("id does not match canonical stable-ID material")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid canonical ledger row {index}: {error}"
            ) from error
        if row_id in seen_ids:
            raise ValueError(f"duplicate stable ledger ID: {row_id}")
        seen_ids.add(row_id)


def write_ledger_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Persist canonical ledger rows without passing through Markdown rendering."""

    validate_ledger_rows(rows)
    _write_jsonl(path, rows)


def read_ledger_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"ledger JSONL row {line_number} must be an object")
        rows.append(row)
    validate_ledger_rows(rows)
    return rows


def summarize_ledger(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    validate_ledger_rows(rows)
    return {
        "rows": len(rows),
        "kind_totals": dict(sorted(Counter(str(row["kind"]) for row in rows).items())),
        "status_totals": dict(
            sorted(Counter(str(row["status"]) for row in rows).items())
        ),
        "decision_totals": {
            "assigned": sum(bool(row["decision_applied"]) for row in rows),
            "unassigned": sum(not bool(row["decision_applied"]) for row in rows),
        },
    }


def summarize_destination_closure(
    rows: Sequence[Mapping[str, object]],
    destination_errors: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    """Report structural closure separately from pending semantic claim work."""

    error_counts = Counter(
        str(error.get("code", "unknown")) for error in destination_errors
    )
    return {
        "bare_destination_strings": error_counts["bare_destination_string"],
        "missing_destination_paths": error_counts["missing_destination_path"],
        "missing_destination_anchors": error_counts[
            "missing_destination_anchor"
        ],
        "ambiguous_destination_anchors": error_counts[
            "ambiguous_destination_anchor"
        ],
        "invalid_destination_entries": sum(
            count
            for code, count in error_counts.items()
            if code
            not in {
                "bare_destination_string",
                "missing_destination_path",
                "missing_destination_anchor",
                "ambiguous_destination_anchor",
            }
        ),
        "unassigned_structural_units": sum(
            str(row.get("kind")) != "claim_candidate"
            and (
                not bool(row.get("decision_applied"))
                or not bool(row.get("destinations"))
                or row.get("destination_role") == "unassigned"
            )
            for row in rows
        ),
        "pending_claim_candidates": sum(
            str(row.get("kind")) == "claim_candidate"
            and not bool(row.get("decision_applied"))
            for row in rows
        ),
    }


def summarize_claim_closure(
    rows: Sequence[Mapping[str, object]],
    claim_decisions: Sequence[Mapping[str, object]],
    claim_errors: Sequence[Mapping[str, object]],
    destination_errors: Sequence[Mapping[str, object]],
    correction_catalog: Mapping[str, Mapping[str, object]],
    correction_dispositions: Mapping[str, object] | None = None,
    disposition_errors: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Summarize decision closure without conflating it with semantic truth."""

    claim_rows = [
        row for row in rows if row.get("kind") == "claim_candidate"
    ]
    error_counts = Counter(
        str(error.get("code", "unknown")) for error in claim_errors
    )
    referenced_corrections = sorted(
        {
            str(correction_id)
            for decision in claim_decisions
            for correction_id in decision.get("correction_ids", [])
        }
    )
    catalog_ids = sorted(correction_catalog)
    raw_dispositions = (
        correction_dispositions.get("dispositions", [])
        if correction_dispositions is not None
        else []
    )
    dispositions = [
        item for item in raw_dispositions if isinstance(item, Mapping)
    ]
    disposition_ids = [
        str(item["correction_id"])
        for item in dispositions
        if isinstance(item.get("correction_id"), str)
    ]
    return {
        "manifest_pages": len(
            {str(row["page"]) for row in claim_rows}
        ),
        "claim_candidates": len(claim_rows),
        "claim_decisions": len(claim_decisions),
        "unique_claim_decisions": len(
            {
                str(decision.get("claim_id"))
                for decision in claim_decisions
                if decision.get("claim_id") is not None
            }
        ),
        "unaudited_claim_candidates": sum(
            not bool(row.get("decision_applied")) for row in claim_rows
        ),
        "unresolved_claims_imported_by_new_series": sum(
            str(row.get("status")) == "unresolved"
            and row.get("destination_role") == "authoritative_import"
            for row in claim_rows
        ),
        "corrected_without_local_callout": error_counts[
            "corrected_without_local_callout"
        ],
        "claim_identity_errors": sum(
            count
            for code, count in error_counts.items()
            if code
            in {
                "missing_claim_id",
                "duplicate_claim_decision",
                "orphan_claim_decision",
                "missing_claim_decision",
                "claim_page_mismatch",
                "claim_page_hash_mismatch",
                "claim_span_mismatch",
                "claim_column_span_mismatch",
                "claim_text_hash_mismatch",
            }
        ),
        "claim_validation_errors": len(claim_errors),
        "destination_validation_errors": len(destination_errors),
        "correction_catalog_ids": len(catalog_ids),
        "correction_ids_referenced_by_claims": len(
            referenced_corrections
        ),
        "correction_ids_without_local_claim": sorted(
            set(catalog_ids).difference(referenced_corrections)
        ),
        "correction_dispositions": len(dispositions),
        "unique_correction_dispositions": len(set(disposition_ids)),
        "correction_disposition_validation_errors": len(
            disposition_errors
        ),
        "correction_ids_without_disposition": sorted(
            set(catalog_ids).difference(disposition_ids)
        ),
        "correction_disposition_totals": dict(
            sorted(
                Counter(
                    str(item.get("disposition", ""))
                    for item in dispositions
                ).items()
            )
        ),
        "status_totals": dict(
            sorted(
                Counter(
                    str(row["status"]) for row in claim_rows
                ).items()
            )
        ),
        "action_totals": dict(
            sorted(
                Counter(
                    str(row.get("action", ""))
                    for row in claim_rows
                ).items()
            )
        ),
        "destination_role_totals": dict(
            sorted(
                Counter(
                    str(row.get("destination_role", ""))
                    for row in claim_rows
                ).items()
            )
        ),
    }


def _claim_or_unit(row: Mapping[str, object]) -> str:
    text = str(row.get("text", ""))
    if text:
        return text
    payload = row.get("payload")
    if not isinstance(payload, Mapping):
        return ""
    kind = str(row["kind"])
    if kind.startswith("heading_"):
        return str(payload.get("title", ""))
    if kind.startswith("code_"):
        return str(payload.get("content", ""))
    if kind == "markdown_table":
        return str(payload.get("raw", ""))
    if kind.startswith("image_"):
        return (
            f"image alt={payload.get('alt', '')!s}; "
            f"target={payload.get('target', '')!s}"
        )
    if kind == "experiment":
        return str(payload.get("heading", ""))
    if kind == "source_locator":
        return str(payload.get("raw", ""))
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def write_coverage_ledger(
    path: Path,
    records: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]] = (),
    claim_decisions: Sequence[Mapping[str, object]] = (),
    unit_decisions: Sequence[Mapping[str, object]] = (),
) -> None:
    """Render Markdown from canonical rows, retaining the legacy calling convention."""

    rows = (
        [dict(record) for record in records]
        if all("id" in record and "status" in record for record in records)
        else build_ledger_rows(records, decisions, claim_decisions, unit_decisions)
    )

    columns = [
        "id",
        "legacy_page",
        "section",
        "source_line",
        "content_kind",
        "source_span",
        "claim_or_unit",
        "locator",
        "claimed_baseline",
        "current_result",
        "status",
        "destinations",
        "destination_role",
        "action",
        "notes",
        "decision_applied",
        "decision_category",
        "decision_selector",
    ]
    output = [
        "# PyTorch Graph Series Coverage Ledger",
        "",
        "> Mechanical inventory only. `path_and_line_valid` proves that a locator is in bounds; it does not verify the surrounding claim.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        source_span = (
            f"{row['source_start_line']}-{row['source_end_line']}"
        )
        if row["source_start_column"] is not None:
            source_span += (
                f":{row['source_start_column']}-{row['source_end_column']}"
            )
        cells = [
            row["id"],
            f"`{row['page']}`",
            row["section"],
            row["source_line"],
            row["kind"],
            source_span,
            _claim_or_unit(row),
            f"`{row['locator']}`" if row["locator"] else "",
            row["claimed_baseline"],
            row["current_result"],
            row["status"],
            json.dumps(
                row["destinations"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            row["destination_role"],
            row["action"],
            row["notes"],
            row["decision_applied"],
            row["decision_category"] or "",
            row["decision_selector"] or "",
        ]
        output.append("| " + " | ".join(_markdown_cell(cell) for cell in cells) + " |")
    summary = summarize_ledger(rows)

    output.extend(
        [
            "",
            "## Mechanical totals",
            "",
            f"- Auditable rows: {summary['rows']}",
            *[
                f"- `{kind}`: {count}"
                for kind, count in summary["kind_totals"].items()
            ],
            "",
            "A row may leave `unresolved` only after a manual audit explains why the evidence is insufficient.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory graph-related Markdown and source locators."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        required=True,
        help="Legacy-compatible mechanical inventory JSONL output.",
    )
    parser.add_argument(
        "--ledger-jsonl-output",
        type=Path,
        help="Canonical audit-ledger JSONL output.",
    )
    parser.add_argument(
        "--claim-ledger-jsonl-output",
        type=Path,
        help="Optional canonical JSONL containing only claim rows.",
    )
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--ledger-output", type=Path, required=True)
    parser.add_argument(
        "--decisions",
        type=Path,
        nargs="+",
        help="One or more JSON decision files, concatenated in argument order.",
    )
    parser.add_argument(
        "--claim-decisions",
        type=Path,
        nargs="+",
        help="One or more claim-decision JSON or JSONL files.",
    )
    parser.add_argument(
        "--unit-decisions",
        type=Path,
        nargs="+",
        help="One or more exact structural-unit decision JSON or JSONL files.",
    )
    parser.add_argument(
        "--destination-errors-output",
        type=Path,
        help="Optional JSON output for structured destination validation errors.",
    )
    parser.add_argument(
        "--correction-reports",
        type=Path,
        nargs="+",
        help="Detailed correction reports whose exact H3 IDs form the catalog.",
    )
    parser.add_argument(
        "--correction-dispositions",
        type=Path,
        help=(
            "Catalog-wide semantic disposition JSON; required for a "
            "complete correction closure."
        ),
    )
    parser.add_argument(
        "--claim-errors-output",
        type=Path,
        help="Optional JSON output for exact claim-decision validation errors.",
    )
    parser.add_argument(
        "--claim-closure-output",
        type=Path,
        help="Optional JSON output for claim audit closure metrics.",
    )
    return parser.parse_args()


def _parse_destination_migration_args(
    argv: Sequence[str],
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "docs.audits.pytorch_graph_series.tools.audit_graph_docs "
            "migrate-destinations"
        ),
        description=(
            "Migrate only mechanically unambiguous legacy destination "
            "strings; semantic ambiguity is a hard error."
        ),
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(list(argv))


def _run_destination_migration(argv: Sequence[str]) -> int:
    args = _parse_destination_migration_args(argv)
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        print(
            json.dumps(
                {"error": "migration input and output must be different"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        aliases = load_destination_aliases(args.aliases)
        decisions = load_semantic_decisions([input_path])
        migrated = migrate_legacy_destination_decisions(
            decisions,
            aliases,
            args.repo_root.resolve(),
        )
        errors = validate_destinations(
            migrated, args.repo_root.resolve()
        )
        if errors:
            print(
                json.dumps(
                    {"destination_errors": errors},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {"error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.casefold() == ".jsonl":
        output_path.write_text(
            "".join(
                json.dumps(decision, ensure_ascii=False) + "\n"
                for decision in migrated
            ),
            encoding="utf-8",
        )
    else:
        output_path.write_text(
            json.dumps(
                migrated,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "decisions": len(migrated),
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


def _parse_legacy_claim_generation_args(
    argv: Sequence[str],
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "docs.audits.pytorch_graph_series.tools.audit_graph_docs "
            "generate-legacy-claims"
        ),
        description=(
            "Generate exact, conservative per-claim decisions from a "
            "frozen legacy manifest. Broad ranges are context only."
        ),
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--decisions", type=Path, nargs="+", required=True
    )
    parser.add_argument(
        "--correction-reports", type=Path, nargs="+", required=True
    )
    parser.add_argument(
        "--correction-dispositions", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--claim-ledger-output", type=Path, required=True
    )
    parser.add_argument("--closure-output", type=Path, required=True)
    parser.add_argument("--errors-output", type=Path, required=True)
    return parser.parse_args(list(argv))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_claim_decisions_by_page(
    output_dir: Path,
    decisions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(
        list
    )
    for decision in decisions:
        grouped[str(decision["page"])].append(decision)
    filename_to_page: dict[str, str] = {}
    for page in grouped:
        filename = f"{Path(page).stem}.jsonl"
        existing = filename_to_page.get(filename)
        if existing is not None and existing != page:
            raise ValueError(
                "claim decision filename collision: "
                f"{existing!r} and {page!r}"
            )
        filename_to_page[filename] = page
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for filename, page in sorted(filename_to_page.items()):
        path = output_dir / filename
        _write_jsonl(path, grouped[page])
        manifest.append(
            {
                "page": page,
                "path": path.as_posix(),
                "claims": len(grouped[page]),
                "sha256": _sha256(path),
            }
        )
    return manifest


def _run_legacy_claim_generation(argv: Sequence[str]) -> int:
    args = _parse_legacy_claim_generation_args(argv)
    try:
        repo_root = args.repo_root.resolve()
        records = audit_manifest(
            repo_root,
            args.source_root.resolve(),
            args.manifest.resolve(),
        )
        range_decisions = load_semantic_decisions(args.decisions)
        catalog = load_correction_catalog(
            args.correction_reports, repo_root
        )
        dispositions = load_correction_dispositions(
            args.correction_dispositions
        )
        base_rows = build_ledger_rows(records, range_decisions)
        claim_rows = [
            row
            for row in base_rows
            if row["kind"] == "claim_candidate"
        ]
        claim_decisions = generate_legacy_claim_decisions(
            claim_rows,
            range_decisions,
            catalog,
            repo_root,
        )
        claim_errors = validate_claim_decisions(
            claim_rows,
            claim_decisions,
            repo_root,
            catalog,
        )
        disposition_errors = validate_correction_dispositions(
            dispositions,
            catalog,
            claim_rows,
            claim_decisions,
            repo_root,
        )
        destination_errors = validate_destinations(
            claim_decisions, repo_root
        )
        combined_errors = [
            *claim_errors,
            *disposition_errors,
            *[
                {"code": "destination_validation_error", **error}
                for error in destination_errors
            ],
        ]
        if combined_errors:
            _write_json(args.errors_output, combined_errors)
            print(
                json.dumps(
                    {
                        "claim_validation_errors": len(claim_errors),
                        "correction_disposition_validation_errors": len(
                            disposition_errors
                        ),
                        "destination_validation_errors": len(
                            destination_errors
                        ),
                    },
                    sort_keys=True,
                )
            )
            return 2
        rows = build_ledger_rows(
            records,
            range_decisions,
            claim_decisions=claim_decisions,
        )
        canonical_claim_rows = [
            row for row in rows if row["kind"] == "claim_candidate"
        ]
        write_ledger_jsonl(
            args.claim_ledger_output, canonical_claim_rows
        )
        decision_files = _write_claim_decisions_by_page(
            args.output_dir, claim_decisions
        )
        closure = summarize_claim_closure(
            rows,
            claim_decisions,
            claim_errors,
            destination_errors,
            catalog,
            dispositions,
            disposition_errors,
        )
        closure["decision_files"] = decision_files
        closure["frozen_page_hashes"] = {
            str(record["page"]): str(record["sha256"])
            for record in records
            if record["kind"] == "page"
        }
        _write_json(args.closure_output, closure)
        _write_json(args.errors_output, [])
        print(json.dumps(closure, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        _write_json(
            args.errors_output,
            [{"code": "generation_error", "message": str(error)}],
        )
        print(
            json.dumps(
                {"error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "migrate-destinations":
        return _run_destination_migration(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "generate-legacy-claims":
        return _run_legacy_claim_generation(sys.argv[2:])
    args = _parse_args()
    records = audit_manifest(
        args.repo_root.resolve(),
        args.source_root.resolve(),
        args.manifest.resolve(),
    )
    _write_jsonl(args.jsonl_output, records)
    _write_summary(args.summary_output, records)
    decisions: Sequence[Mapping[str, object]] = ()
    if args.decisions is not None:
        decisions = load_semantic_decisions(args.decisions)
    claim_decisions: Sequence[Mapping[str, object]] = ()
    if args.claim_decisions is not None:
        claim_decisions = load_semantic_decisions(args.claim_decisions)
    unit_decisions: Sequence[Mapping[str, object]] = ()
    if args.unit_decisions is not None:
        unit_decisions = load_semantic_decisions(args.unit_decisions)
    correction_catalog: Mapping[str, Mapping[str, object]] = {}
    if args.correction_reports is not None:
        correction_catalog = load_correction_catalog(
            args.correction_reports, args.repo_root.resolve()
        )
    correction_dispositions: Mapping[str, object] | None = None
    if args.correction_dispositions is not None:
        correction_dispositions = load_correction_dispositions(
            args.correction_dispositions
        )
    base_rows = build_ledger_rows(records, decisions)
    base_claim_rows = [
        row for row in base_rows if row["kind"] == "claim_candidate"
    ]
    claim_errors: list[dict[str, object]] = []
    if claim_decisions:
        claim_errors = validate_claim_decisions(
            base_claim_rows,
            claim_decisions,
            args.repo_root.resolve(),
            correction_catalog,
        )
    disposition_errors: list[dict[str, object]] = []
    if correction_dispositions is not None:
        disposition_errors = validate_correction_dispositions(
            correction_dispositions,
            correction_catalog,
            base_claim_rows,
            claim_decisions,
            args.repo_root.resolve(),
        )
    rows = build_ledger_rows(
        records,
        decisions,
        (() if claim_errors else claim_decisions),
        unit_decisions,
    )
    decision_destination_rows = [
        decision
        for decision in (
            *decisions,
            *claim_decisions,
            *unit_decisions,
        )
        if any(
            field in decision
            for field in (
                "destination",
                "destinations",
                "destination_role",
            )
        )
    ]
    destination_errors = validate_destinations(
        (
            decision_destination_rows
            if decision_destination_rows
            else rows
        ),
        args.repo_root.resolve(),
    )
    closure = summarize_destination_closure(rows, destination_errors)
    claim_closure = summarize_claim_closure(
        rows,
        claim_decisions,
        claim_errors,
        destination_errors,
        correction_catalog,
        correction_dispositions,
        disposition_errors,
    )
    if args.destination_errors_output is not None:
        args.destination_errors_output.parent.mkdir(
            parents=True, exist_ok=True
        )
        args.destination_errors_output.write_text(
            json.dumps(
                destination_errors,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.claim_errors_output is not None:
        _write_json(
            args.claim_errors_output,
            [*claim_errors, *disposition_errors],
        )
    if args.claim_closure_output is not None:
        _write_json(args.claim_closure_output, claim_closure)
    if args.ledger_jsonl_output is not None:
        write_ledger_jsonl(args.ledger_jsonl_output, rows)
    if args.claim_ledger_jsonl_output is not None:
        write_ledger_jsonl(
            args.claim_ledger_jsonl_output,
            [
                row
                for row in rows
                if row["kind"] == "claim_candidate"
            ],
        )
    write_coverage_ledger(args.ledger_output, rows)
    unbalanced = [
        record
        for record in records
        if record["kind"] == "code_fence" and not record.get("balanced")
    ]
    print(
        json.dumps(
            {
                "pages": sum(record["kind"] == "page" for record in records),
                "records": len(records),
                "unbalanced_fences": len(unbalanced),
                **summarize_ledger(rows),
                "destination_closure": closure,
                "destination_validation_errors": len(
                    destination_errors
                ),
                "claim_closure": claim_closure,
                "claim_validation_errors": len(claim_errors),
                "correction_disposition_validation_errors": len(
                    disposition_errors
                ),
            },
            sort_keys=True,
        )
    )
    return (
        2
        if unbalanced
        or destination_errors
        or claim_errors
        or disposition_errors
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
