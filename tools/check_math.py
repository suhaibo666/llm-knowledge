#!/usr/bin/env python3
"""Check Obsidian Markdown math delimiters and common notation mistakes."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Iterable


LEGACY_DELIMITER_RE = re.compile(r"\\[()\[\]]")
FENCE_RE = re.compile(r"^\s{0,3}(?:>\s*)?(?P<fence>`{3,}|~{3,})")
ALLOWED_TEXT_COMMAND_RE = re.compile(
    r"\\(?:text|texttt|mathrm|operatorname)\s*\{"
)
CURRENCY_RE = re.compile(
    r"\$(?:\d[\d,]*(?:\.\d+)?)(?:\s?(?:USD|CNY|RMB|[KMBT]))?\b",
    re.IGNORECASE,
)
SUBSCRIPT_RE = re.compile(r"_\{([^{}]*)\}")


@dataclass(frozen=True)
class Diagnostic:
    """One stable, line-addressable checker finding."""

    severity: str
    code: str
    path: str
    line: int
    message: str

    def render(self) -> str:
        return (
            f"{self.path}:{self.line} "
            f"[{self.severity.upper()} {self.code}] {self.message}"
        )


@dataclass(frozen=True)
class MathSegment:
    content: str
    line: int
    kind: str
    source_line: str


def _mask_inline_code(line: str) -> str:
    """Replace complete backtick code spans with spaces, preserving columns."""

    chars = list(line)
    index = 0
    while index < len(line):
        if line[index] != "`" or (index > 0 and line[index - 1] == "\\"):
            index += 1
            continue
        run_end = index
        while run_end < len(line) and line[run_end] == "`":
            run_end += 1
        marker = line[index:run_end]
        closing = line.find(marker, run_end)
        if closing < 0:
            index = run_end
            continue
        for pos in range(index, closing + len(marker)):
            chars[pos] = " "
        index = closing + len(marker)
    return "".join(chars)


def _unescaped_double_dollars(line: str) -> list[int]:
    positions: list[int] = []
    index = 0
    while index < len(line) - 1:
        if (
            line[index : index + 2] == "$$"
            and (index == 0 or line[index - 1] != "\\")
        ):
            positions.append(index)
            index += 2
        else:
            index += 1
    return positions


def _is_standalone_display_delimiter(line: str) -> bool:
    return re.fullmatch(r"\s*(?:>\s*)?\$\$\s*", line) is not None


def _looks_like_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2


CURRENCY_TAIL_RE = re.compile(r"[\s|,.;:)\]，。、；：）]*\Z")


def _next_single_dollar(text: str, start: int) -> int:
    """Offset of the next unescaped single ``$`` at or after ``start``, else -1."""

    index = start
    while index < len(text):
        if text[index] != "$" or (index > 0 and text[index - 1] == "\\"):
            index += 1
            continue
        if index + 1 < len(text) and text[index + 1] == "$":
            index += 2
            continue
        return index
    return -1


def _is_currency_amount(text: str, start: int, end: int) -> bool:
    """Decide whether ``text[start:end]`` is money rather than inline math.

    ``$5.576M`` in a cost table is currency.  ``$4.2 \times 10^{-4}$`` is inline
    math that merely begins with a number: it is closed by a later ``$`` and the
    text in between carries mathematical content, not just separators.
    """

    if end < len(text) and text[end] == "$":
        return False
    closer = _next_single_dollar(text, end)
    if closer < 0:
        return True
    if CURRENCY_RE.match(text, closer):
        # a second amount on the same row (e.g. a cost table): both are money
        return True
    return CURRENCY_TAIL_RE.match(text, end, closer) is not None


def _legacy_delimiter_matches(text: str) -> list[tuple[int, str]]:
    r"""Return (offset, token) for real legacy delimiters.

    A backslash that is itself escaped (``\\``) does not start a delimiter, so
    LaTeX row spacing such as ``\\[2pt]`` inside ``aligned``/``cases`` is not a
    legacy ``\[`` display opener.
    """

    found: list[tuple[int, str]] = []
    index = 0
    while index < len(text):
        if text[index] != "\\":
            index += 1
            continue
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if nxt == "\\":
            index += 2
            continue
        if nxt in "()[]":
            found.append((index, "\\" + nxt))
        index += 2
    return found


def _single_dollar_positions(text: str) -> list[int]:
    positions: list[int] = []
    index = 0
    while index < len(text):
        if text[index] != "$" or (index > 0 and text[index - 1] == "\\"):
            index += 1
            continue
        if index + 1 < len(text) and text[index + 1] == "$":
            index += 2
            continue

        currency = CURRENCY_RE.match(text, index)
        if currency and _is_currency_amount(text, index, currency.end()):
            index = currency.end()
            continue

        positions.append(index)
        index += 1
    return positions


def _inline_segments(
    text: str,
    path: str,
    line_number: int,
    source_line: str,
    diagnostics: list[Diagnostic],
) -> list[MathSegment]:
    positions = _single_dollar_positions(text)
    if len(positions) % 2:
        diagnostics.append(
            Diagnostic(
                "error",
                "MATH003",
                path,
                line_number,
                "inline '$' delimiter is not paired on the same line",
            )
        )
        return []

    return [
        MathSegment(text[start + 1 : end], line_number, "inline", source_line)
        for start, end in zip(positions[::2], positions[1::2])
    ]


def _balanced_braces(content: str) -> bool:
    depth = 0
    for index, char in enumerate(content):
        if char not in "{}" or (index > 0 and content[index - 1] == "\\"):
            continue
        if char == "{":
            depth += 1
        else:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _mask_allowed_text_commands(content: str) -> str:
    """Mask text-like command arguments so their escaped underscores are valid."""

    chars = list(content)
    search_from = 0
    while True:
        match = ALLOWED_TEXT_COMMAND_RE.search(content, search_from)
        if not match:
            break
        depth = 1
        index = match.end()
        while index < len(content) and depth:
            if content[index] in "{}" and content[index - 1] != "\\":
                depth += 1 if content[index] == "{" else -1
            index += 1
        if depth:
            break
        for pos in range(match.start(), index):
            chars[pos] = " "
        search_from = index
    return "".join(chars)


def _has_raw_pipe(content: str) -> bool:
    return re.search(r"(?<!\\)\|", content) is not None


def _check_segment(
    segment: MathSegment,
    path: str,
    diagnostics: list[Diagnostic],
) -> None:
    content = segment.content
    if not _balanced_braces(content):
        diagnostics.append(
            Diagnostic(
                "error",
                "MATH004",
                path,
                segment.line,
                "math expression has unbalanced braces",
            )
        )

    masked_text = _mask_allowed_text_commands(content)
    if re.search(r"(?<!\\)\\_", masked_text):
        diagnostics.append(
            Diagnostic(
                "warning",
                "MATH102",
                path,
                segment.line,
                "escaped underscore appears outside a text-like command",
            )
        )

    for match in SUBSCRIPT_RE.finditer(content):
        subscript = match.group(1)
        if re.search(r"\\(?:text|texttt|mathrm|operatorname)\s*\{", subscript):
            continue
        without_commands = re.sub(r"\\[A-Za-z]+", "", subscript)
        if re.search(r"[A-Za-z]{3,}|[-/]", without_commands):
            diagnostics.append(
                Diagnostic(
                    "warning",
                    "MATH103",
                    path,
                    segment.line,
                    "semantic subscript should use \\mathrm{...} or \\text{...}",
                )
            )
            break

    if _has_raw_pipe(content):
        diagnostics.append(
            Diagnostic(
                "warning",
                "MATH104",
                path,
                segment.line,
                "use \\mid for conditioning or \\lvert...\\rvert for absolute value",
            )
        )
        if _looks_like_table_row(segment.source_line):
            diagnostics.append(
                Diagnostic(
                    "error",
                    "MATH005",
                    path,
                    segment.line,
                    "raw '|' inside table math conflicts with Markdown columns",
                )
            )

    if (
        segment.kind == "display"
        and any(len(line.strip()) > 120 for line in content.splitlines())
        and "\\begin{aligned}" not in content
    ):
        diagnostics.append(
            Diagnostic(
                "warning",
                "MATH105",
                path,
                segment.line,
                "long display equation should use an aligned environment",
            )
        )


def check_text(text: str, path: str = "<memory>") -> list[Diagnostic]:
    """Return deterministic diagnostics for one Markdown document."""

    diagnostics: list[Diagnostic] = []
    segments: list[MathSegment] = []
    fence_char: str | None = None
    fence_length = 0
    in_display = False
    display_start = 0
    display_parts: list[str] = []
    display_source = ""

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        fence = FENCE_RE.match(raw_line)
        if fence:
            marker = fence.group("fence")
            if fence_char is None:
                fence_char = marker[0]
                fence_length = len(marker)
                continue
            if marker[0] == fence_char and len(marker) >= fence_length:
                fence_char = None
                fence_length = 0
            continue
        if fence_char is not None:
            continue

        line = _mask_inline_code(raw_line)
        for _offset, token in _legacy_delimiter_matches(line):
            diagnostics.append(
                Diagnostic(
                    "error",
                    "MATH001",
                    path,
                    line_number,
                    f"use Obsidian dollar delimiters instead of '{token}'",
                )
            )

        double_positions = _unescaped_double_dollars(line)
        if double_positions and not _is_standalone_display_delimiter(line):
            diagnostics.append(
                Diagnostic(
                    "warning",
                    "MATH101",
                    path,
                    line_number,
                    "display '$$' delimiters must each be on their own line",
                )
            )
        if double_positions and _looks_like_table_row(line):
            diagnostics.append(
                Diagnostic(
                    "error",
                    "MATH005",
                    path,
                    line_number,
                    "display math is not allowed inside a Markdown table",
                )
            )

        cursor = 0
        for position in double_positions:
            fragment = line[cursor:position]
            if in_display:
                display_parts.append(fragment)
                segments.append(
                    MathSegment(
                        "\n".join(display_parts),
                        display_start,
                        "display",
                        display_source,
                    )
                )
                display_parts = []
                in_display = False
            else:
                segments.extend(
                    _inline_segments(
                        fragment, path, line_number, raw_line, diagnostics
                    )
                )
                in_display = True
                display_start = line_number
                display_source = raw_line
            cursor = position + 2

        tail = line[cursor:]
        if in_display:
            display_parts.append(tail)
        else:
            segments.extend(
                _inline_segments(tail, path, line_number, raw_line, diagnostics)
            )

    if in_display:
        diagnostics.append(
            Diagnostic(
                "error",
                "MATH002",
                path,
                display_start,
                "display '$$' delimiter is not closed",
            )
        )

    for segment in segments:
        _check_segment(segment, path, diagnostics)

    return sorted(
        diagnostics,
        key=lambda item: (item.path, item.line, item.code, item.message),
    )


def check_file(path: Path) -> list[Diagnostic]:
    return check_text(path.read_text(encoding="utf-8"), str(path))


def _diagnostic_fingerprint(
    diagnostic: Diagnostic, text: str
) -> tuple[str, str, str, tuple[str, ...]]:
    """Identify the same finding across unrelated line-number shifts."""

    lines = text.splitlines()
    index = min(max(diagnostic.line - 1, 0), max(len(lines) - 1, 0))
    context = tuple(line.strip() for line in lines[index : index + 3])
    return (
        diagnostic.severity,
        diagnostic.code,
        diagnostic.message,
        context,
    )


def new_diagnostics_since(
    base_text: str, current_text: str, path: str
) -> list[Diagnostic]:
    """Return findings introduced since base_text, ignoring shifted legacy debt."""

    base_counts = Counter(
        _diagnostic_fingerprint(item, base_text)
        for item in check_text(base_text, path)
    )
    introduced: list[Diagnostic] = []
    for item in check_text(current_text, path):
        fingerprint = _diagnostic_fingerprint(item, current_text)
        if base_counts[fingerprint]:
            base_counts[fingerprint] -= 1
        else:
            introduced.append(item)
    return introduced


def _git_paths(repo_root: Path, arguments: list[str]) -> list[Path]:
    result = subprocess.run(
        ["git", *arguments, "-z"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    names = result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    return [repo_root / name for name in names if name]


def collect_changed_markdown(repo_root: Path) -> list[Path]:
    """Collect modified, staged, and untracked Markdown paths from Git."""

    candidates: set[Path] = set()
    commands = [
        ["diff", "--name-only", "--diff-filter=ACMR"],
        ["diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        ["ls-files", "--others", "--exclude-standard"],
    ]
    for command in commands:
        candidates.update(_git_paths(repo_root, command))
    return sorted(
        path.resolve()
        for path in candidates
        if path.suffix.lower() == ".md" and path.is_file()
    )


def _read_head_text(repo_root: Path, path: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return None
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        return None
    return result.stdout.decode("utf-8", errors="surrogateescape")


def _collect_explicit_paths(values: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for value in values:
        candidate = Path(value)
        if candidate.is_dir():
            paths.update(path.resolve() for path in candidate.rglob("*.md"))
        elif candidate.is_file() and candidate.suffix.lower() == ".md":
            paths.add(candidate.resolve())
    return sorted(paths)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check Obsidian Markdown math syntax and notation."
    )
    parser.add_argument("paths", nargs="*", help="Markdown files or directories")
    parser.add_argument(
        "--changed",
        action="store_true",
        help="also check modified, staged, and untracked Markdown files",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero for warnings as well as errors",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    explicit_files = set(_collect_explicit_paths(args.paths))
    changed_files: set[Path] = set()
    if args.changed or not args.paths:
        try:
            changed_files.update(collect_changed_markdown(repo_root))
        except subprocess.CalledProcessError as error:
            print(error.stderr.decode("utf-8", errors="replace").strip())
            return 2

    diagnostics: list[Diagnostic] = [
        diagnostic
        for path in sorted(explicit_files)
        for diagnostic in check_file(path)
    ]
    for path in sorted(changed_files - explicit_files):
        current_text = path.read_text(encoding="utf-8")
        base_text = _read_head_text(repo_root, path)
        if base_text is None:
            diagnostics.extend(check_text(current_text, str(path)))
        else:
            diagnostics.extend(
                new_diagnostics_since(base_text, current_text, str(path))
            )

    files = explicit_files | changed_files
    for diagnostic in diagnostics:
        print(diagnostic.render())

    errors = sum(item.severity == "error" for item in diagnostics)
    warnings = sum(item.severity == "warning" for item in diagnostics)
    print(
        f"Checked {len(files)} Markdown file(s): "
        f"{errors} error(s), {warnings} warning(s)."
    )
    return int(errors > 0 or (args.strict and warnings > 0))


if __name__ == "__main__":
    raise SystemExit(main())
