from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import unquote


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
FENCE_WITH_INFO_RE = re.compile(r"^\s*(`{3,}|~{3,})\s*([A-Za-z0-9_-]*)")
CORRECTION_ID_RE = re.compile(r"\b[A-Z]-\d{3}\b")
LOCAL_CORRECTION_PHRASE = "本区段按固定基线纠错"
MERMAID_DIRECTIVE_RE = re.compile(
    r"^(?:"
    r"flowchart|graph|sequenceDiagram|classDiagram(?:-v2)?|"
    r"stateDiagram(?:-v2)?|erDiagram|journey|gantt|pie|mindmap|timeline|"
    r"gitGraph|quadrantChart|xychart-beta|block-beta|packet-beta|"
    r"architecture-beta|sankey-beta|requirementDiagram|"
    r"C4Context|C4Container|C4Component|C4Dynamic|C4Deployment"
    r")\b"
)
MARKDOWN_LINK_RE = re.compile(
    r"!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
)


def _error(code: str, path: object, detail: str) -> dict[str, object]:
    return {"code": code, "path": str(path), "detail": detail}


def _load_manifest(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"manifest must be a JSON string array: {path}")
    return value


def _h2_titles(text: str) -> list[str]:
    return [
        match.group(2).strip()
        for line in text.splitlines()
        if (match := HEADING_RE.match(line)) and len(match.group(1)) == 2
    ]


def validate_related_pages(paths: Sequence[Path]) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            errors.append(_error("related_pages_file_missing", path, "file is missing"))
            continue
        h2_titles = _h2_titles(path.read_text(encoding="utf-8"))
        if not h2_titles:
            errors.append(
                _error("related_pages_missing", path, "page has no H2 headings")
            )
        elif h2_titles[-1] != "Related Pages":
            code = (
                "related_pages_not_final"
                if "Related Pages" in h2_titles
                else "related_pages_missing"
            )
            errors.append(
                _error(code, path, "final H2 must be exactly 'Related Pages'")
            )
    return errors


def validate_markdown_fences_and_mermaid(
    paths: Sequence[Path],
) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            errors.append(_error("markdown_source_missing", path, "file is missing"))
            continue
        marker_char: str | None = None
        marker_length = 0
        language = ""
        start_line = 0
        body: list[str] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = FENCE_WITH_INFO_RE.match(line)
            if marker_char is None:
                if match is None:
                    continue
                marker = match.group(1)
                marker_char = marker[0]
                marker_length = len(marker)
                language = match.group(2).lower()
                start_line = line_number
                body = []
                continue

            closing = re.match(
                rf"^\s*{re.escape(marker_char)}{{{marker_length},}}\s*$",
                line,
            )
            if closing is None:
                body.append(line)
                continue
            if language == "mermaid":
                meaningful = [
                    item.strip()
                    for item in body
                    if item.strip() and not item.lstrip().startswith("%%")
                ]
                if not meaningful:
                    errors.append(
                        _error(
                            "mermaid_empty",
                            path,
                            f"line {start_line}: Mermaid fence has no diagram",
                        )
                    )
                elif MERMAID_DIRECTIVE_RE.match(meaningful[0]) is None:
                    errors.append(
                        _error(
                            "mermaid_unknown_directive",
                            path,
                            f"line {start_line}: {meaningful[0]}",
                        )
                    )
            marker_char = None
            marker_length = 0
            language = ""
            start_line = 0
            body = []
        if marker_char is not None:
            errors.append(
                _error(
                    "unbalanced_code_fence",
                    path,
                    f"opening fence at line {start_line} is not closed",
                )
            )
    return errors


def validate_numbered_course(
    repo_root: Path, manifest_path: Path
) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    manifest = _load_manifest(manifest_path)
    if len(manifest) != 22:
        errors.append(
            _error(
                "course_file_count",
                manifest_path,
                f"expected 22 numbered files, found {len(manifest)}",
            )
        )
    paths = [repo_root / item for item in manifest]
    numbers: list[int] = []
    stems: dict[int, str] = {}
    for relative, path in zip(manifest, paths):
        match = re.match(r"^(\d{2})_(.+)\.md$", path.name)
        if match is None:
            errors.append(
                _error(
                    "course_filename_not_numbered",
                    relative,
                    "filename must start with a two-digit number",
                )
            )
            continue
        number = int(match.group(1))
        numbers.append(number)
        stems[number] = path.stem
        if not path.is_file():
            errors.append(_error("course_file_missing", relative, "file is missing"))
            continue
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        if not first_line.startswith(f"# {number:02d} · "):
            errors.append(
                _error(
                    "course_h1_number_mismatch",
                    relative,
                    f"H1 must start with '# {number:02d} · '",
                )
            )
    if sorted(numbers) != list(range(22)):
        errors.append(
            _error(
                "course_number_set",
                manifest_path,
                f"numbers must be exactly 00-21; found {sorted(numbers)}",
            )
        )
    if stems.get(0) != "00_pytorch_graph_series_index":
        errors.append(
            _error(
                "course_index_filename",
                manifest_path,
                "00 must be 00_pytorch_graph_series_index.md",
            )
        )
    for number in range(1, 22):
        if number not in stems:
            continue
        path = next(
            path
            for path in paths
            if path.name.startswith(f"{number:02d}_")
        )
        text = path.read_text(encoding="utf-8")
        if "## 学习顺序" not in text:
            errors.append(
                _error(
                    "course_learning_order_missing",
                    path,
                    "numbered course page lacks ## 学习顺序",
                )
            )
            continue
        previous = stems.get(number - 1)
        if previous and f"[[{previous}]]" not in text:
            errors.append(
                _error(
                    "course_previous_link_missing",
                    path,
                    f"missing previous link to {previous}",
                )
            )
        expected_next = stems.get(number + 1) if number < 21 else stems.get(0)
        if expected_next and f"[[{expected_next}]]" not in text:
            errors.append(
                _error(
                    "course_next_link_missing",
                    path,
                    f"missing next/return link to {expected_next}",
                )
            )
    errors.extend(validate_related_pages([path for path in paths if path.is_file()]))
    return errors


def _wikilinks_in_line(line: str) -> Iterable[str]:
    """Yield wikilink bodies while ignoring links inside inline code.

    Inline code is allowed *inside* a wikilink heading, and headings may also
    contain ordinary bracket pairs. A regular expression that strips inline
    code first loses part of links such as
    ``[[page#8. `torch.compile` dynamic=[True]]]``.
    """
    index = 0
    inline_code_ticks: int | None = None
    while index < len(line):
        if line[index] == "`":
            end = index + 1
            while end < len(line) and line[end] == "`":
                end += 1
            tick_count = end - index
            if inline_code_ticks is None:
                inline_code_ticks = tick_count
            elif tick_count == inline_code_ticks:
                inline_code_ticks = None
            index = end
            continue
        if inline_code_ticks is not None:
            index += 1
            continue
        if not line.startswith("[[", index):
            index += 1
            continue

        body_start = index + 2
        cursor = body_start
        bracket_depth = 0
        while cursor < len(line):
            if line[cursor] == "[":
                bracket_depth += 1
                cursor += 1
                continue
            if line[cursor] == "]":
                if bracket_depth:
                    bracket_depth -= 1
                    cursor += 1
                    continue
                if line.startswith("]]", cursor):
                    yield line[body_start:cursor]
                    index = cursor + 2
                    break
            cursor += 1
        else:
            return


def _mask_inline_code(line: str) -> str:
    characters = list(line)
    index = 0
    inline_code_ticks: int | None = None
    while index < len(line):
        if line[index] != "`":
            if inline_code_ticks is not None:
                characters[index] = " "
            index += 1
            continue
        end = index + 1
        while end < len(line) and line[end] == "`":
            end += 1
        tick_count = end - index
        for position in range(index, end):
            characters[position] = " "
        if inline_code_ticks is None:
            inline_code_ticks = tick_count
        elif tick_count == inline_code_ticks:
            inline_code_ticks = None
        index = end
    return "".join(characters)


def _iter_wikilinks(path: Path) -> Iterable[tuple[int, str]]:
    fence_marker: str | None = None
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)[0]
            if fence_marker is None:
                fence_marker = marker
            elif marker == fence_marker:
                fence_marker = None
            continue
        if fence_marker is not None:
            continue
        for body in _wikilinks_in_line(line):
            yield line_number, body


def _iter_markdown_links(path: Path) -> Iterable[tuple[int, str]]:
    fence_marker: str | None = None
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)[0]
            if fence_marker is None:
                fence_marker = marker
            elif marker == fence_marker:
                fence_marker = None
            continue
        if fence_marker is not None:
            continue
        prose = _mask_inline_code(line)
        for match in MARKDOWN_LINK_RE.finditer(prose):
            yield line_number, match.group(1) or match.group(2)


def _markdown_index(repo_root: Path) -> tuple[Path, dict[str, list[Path]]]:
    wiki_root = (repo_root / "wiki").resolve()
    index: dict[str, list[Path]] = {}
    for path in repo_root.rglob("*.md"):
        relative = path.relative_to(repo_root)
        if relative.parts and relative.parts[0] in {"raw", ".git"}:
            continue
        key = relative.with_suffix("").as_posix()
        index.setdefault(key, []).append(path)
        try:
            wiki_key = path.relative_to(wiki_root).with_suffix("").as_posix()
        except ValueError:
            pass
        else:
            index.setdefault(wiki_key, []).append(path)
        index.setdefault(path.stem, []).append(path)
    return wiki_root, index


def _heading_titles(path: Path) -> set[str]:
    return {
        match.group(2).strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := HEADING_RE.match(line))
    }


def _heading_slugs(path: Path) -> set[str]:
    slugs: set[str] = set()
    for title in _heading_titles(path):
        slug = re.sub(r"[^\w\s-]", "", title.lower(), flags=re.UNICODE)
        slug = re.sub(r"[\s-]+", "-", slug).strip("-")
        if slug:
            slugs.add(slug)
    return slugs


def validate_markdown_links(
    repo_root: Path, paths: Sequence[Path]
) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    resolved_root = repo_root.resolve()
    for source in paths:
        if not source.is_file():
            errors.append(
                _error("markdown_link_source_missing", source, "file is missing")
            )
            continue
        for line_number, raw in _iter_markdown_links(source):
            target = unquote(raw.strip())
            if (
                re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I)
                or target.startswith("//")
            ):
                continue
            path_text, separator, anchor = target.partition("#")
            path_text = path_text.partition("?")[0]
            candidate = (
                source
                if not path_text
                else (source.parent / path_text.replace("/", str(Path("/")))).resolve()
            )
            try:
                candidate.relative_to(resolved_root)
            except ValueError:
                candidate = Path()
            if not candidate.is_file():
                errors.append(
                    _error(
                        "markdown_link_target_missing",
                        source,
                        f"line {line_number}: {raw}",
                    )
                )
                continue
            if separator and anchor and candidate.suffix.lower() == ".md":
                if (
                    anchor not in _heading_titles(candidate)
                    and anchor.lower() not in _heading_slugs(candidate)
                ):
                    errors.append(
                        _error(
                            "markdown_link_anchor_missing",
                            source,
                            f"line {line_number}: {raw}",
                        )
                    )
    return errors


def validate_wikilinks(
    repo_root: Path, paths: Sequence[Path]
) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    wiki_root, index = _markdown_index(repo_root)
    for source in paths:
        if not source.is_file():
            errors.append(_error("wikilink_source_missing", source, "file is missing"))
            continue
        for line_number, raw in _iter_wikilinks(source):
            target = raw.split("|", maxsplit=1)[0].strip()
            page_part, separator, anchor = target.partition("#")
            if not page_part:
                candidates = [source]
            else:
                normalized = page_part.replace("\\", "/").removesuffix(".md")
                normalized = normalized.removeprefix("wiki/").lstrip("/")
                if re.match(r"^[a-z]+://", normalized, re.I):
                    continue
                candidates = list(index.get(normalized, ()))
                if not candidates and "/" in normalized:
                    candidates = [
                        candidate
                        for key, values in index.items()
                        if "/" in key and key.endswith("/" + normalized)
                        for candidate in values
                    ]
                if not candidates:
                    relative = (source.parent / normalized).with_suffix(".md").resolve()
                    try:
                        relative.relative_to(wiki_root)
                    except ValueError:
                        relative = Path()
                    if relative.is_file():
                        candidates = [relative]
            candidates = list(dict.fromkeys(candidates))
            if not candidates:
                errors.append(
                    _error(
                        "wikilink_target_missing",
                        source,
                        f"line {line_number}: {raw}",
                    )
                )
                continue
            if separator and anchor and not anchor.startswith("^"):
                if not any(anchor in _heading_titles(candidate) for candidate in candidates):
                    errors.append(
                        _error(
                            "wikilink_anchor_missing",
                            source,
                            f"line {line_number}: {raw}",
                        )
                    )
    return errors


def validate_local_corrections(
    repo_root: Path, decision_paths: Sequence[Path]
) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    for decision_path in decision_paths:
        decisions = json.loads(decision_path.read_text(encoding="utf-8"))
        if not isinstance(decisions, list):
            raise ValueError(f"semantic decisions must be an array: {decision_path}")
        for decision in decisions:
            if not isinstance(decision, Mapping) or decision.get("status") != "corrected":
                continue
            page = repo_root / str(decision["page"])
            if not page.is_file():
                errors.append(
                    _error(
                        "corrected_page_missing",
                        decision.get("page"),
                        "corrected decision page is missing",
                    )
                )
                continue
            identifiers = list(
                dict.fromkeys(
                    CORRECTION_ID_RE.findall(str(decision.get("notes", "")))
                )
            )
            if not identifiers:
                errors.append(
                    _error(
                        "corrected_range_without_id",
                        decision.get("page"),
                        f"{decision['start_line']}-{decision['end_line']}",
                    )
                )
                continue
            lines = page.read_text(encoding="utf-8").splitlines()
            start = int(decision["start_line"])
            end = min(len(lines), int(decision["end_line"]) + 1)
            matching = [
                line
                for line in lines[start - 1 : end]
                if line.startswith("> [!correction]")
                and LOCAL_CORRECTION_PHRASE in line
                and "[[correction_report]]" in line
                and all(identifier in line for identifier in identifiers)
            ]
            if len(matching) != 1:
                errors.append(
                    _error(
                        "corrected_range_without_local_callout",
                        decision.get("page"),
                        f"{start}-{decision['end_line']}: found {len(matching)}",
                    )
                )
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible structural delivery gates for the graph series."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--course-manifest", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--semantic-decisions", type=Path, nargs="+", required=True)
    parser.add_argument("--scope", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    course_paths = [
        repo_root / item for item in _load_manifest(args.course_manifest.resolve())
    ]
    legacy_paths = [
        repo_root / item for item in _load_manifest(args.legacy_manifest.resolve())
    ]
    scope = [path.resolve() for path in args.scope]
    errors = [
        *validate_numbered_course(repo_root, args.course_manifest.resolve()),
        *validate_related_pages(legacy_paths),
        *validate_markdown_fences_and_mermaid(
            [*course_paths, *legacy_paths, *scope]
        ),
        *validate_markdown_links(repo_root, [*course_paths, *legacy_paths, *scope]),
        *validate_wikilinks(repo_root, [*course_paths, *legacy_paths, *scope]),
        *validate_local_corrections(
            repo_root, [path.resolve() for path in args.semantic_decisions]
        ),
    ]
    counts: dict[str, int] = {}
    for error in errors:
        code = str(error["code"])
        counts[code] = counts.get(code, 0) + 1
    result = {
        "course_files": len(course_paths),
        "legacy_files": len(legacy_paths),
        "scope_files": len(scope),
        "errors": errors,
        "error_counts": counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**result, "errors": len(errors)}, sort_keys=True))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
