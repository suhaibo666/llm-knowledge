#!/usr/bin/env python3
"""Check that local asset references in Markdown resolve to real files.

Asset existence is a pure filesystem judgement, so it does not need the built
site: this checker runs at the source layer, in the second or so that scanning
``wiki/**/*.md`` costs, instead of waiting for a full MkDocs build.

Covered references (fenced blocks and inline code excluded):
  - Markdown images ``![alt](path/to/x.png)``
  - HTML ``<img src=...>`` and ``<source srcset=...>``
  - relative links whose target is a file rather than a page (``.png``,
    ``.pdf``, ...)

Deliberately not covered: external URLs, ``mailto:``, bare ``#anchor``,
site-absolute ``/path`` (the source layer does not know the site root),
``file:`` links (this repository's intentional "local source" annotation, which
the site rewrites to *(local source)*), and Obsidian ``[[wikilinks]]`` — those
belong to ``check_links.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterator
from urllib.parse import unquote

from check_math import Diagnostic, _collect_explicit_paths, collect_changed_markdown


FENCE_RE = re.compile(r"^\s{0,3}(?:>\s*)?(?P<fence>`{3,}|~{3,})")
SCHEME_RE = re.compile(r"\A[A-Za-z][A-Za-z0-9+.-]*:")
HTML_ASSET_TAG_RE = re.compile(r"<(?:img|source)\b[^>]*>", re.IGNORECASE)
HTML_SRC_RE = re.compile(
    r"""\bsrc\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""", re.IGNORECASE
)
HTML_SRCSET_RE = re.compile(
    r"""\bsrcset\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""", re.IGNORECASE
)
ASSET_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".avif",
        ".bmp",
        ".ico",
        ".tif",
        ".tiff",
        ".pdf",
        ".mp4",
        ".webm",
    }
)


@dataclass(frozen=True)
class Reference:
    """One local asset reference as the author spelled it."""

    target: str
    line: int
    kind: str


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


def _destination(raw: str) -> str:
    """Strip an inline destination down to its path.

    ``<a b.png>`` keeps its spaces, ``x.png "title"`` drops the title.
    """

    text = raw.strip()
    if text.startswith("<"):
        closing = text.find(">")
        return text[1:closing] if closing > 0 else text[1:]
    return text.split(maxsplit=1)[0] if text else ""


def _inline_destinations(line: str) -> Iterator[tuple[str, str]]:
    """Yield ``(destination, kind)`` for inline images and links on one line.

    The label is closed by the first ``]`` that a ``(`` follows: alt text in this
    wiki routinely carries unbalanced brackets (``[a,b)`` intervals) that
    depth counting would swallow, while ``[[wikilink]]`` — never followed by
    ``(`` — stays untouched.
    """

    index = 0
    length = len(line)
    while index < length:
        if line[index] != "[":
            index += 1
            continue
        is_image = index > 0 and line[index - 1] == "!"
        close = line.find("]", index + 1)
        while close >= 0 and (close + 1 >= length or line[close + 1] != "("):
            close = line.find("]", close + 1)
        if close < 0:
            index += 1
            continue

        depth = 0
        cursor = close + 1
        while cursor < length:
            if line[cursor] == "(":
                depth += 1
            elif line[cursor] == ")":
                depth -= 1
                if not depth:
                    break
            cursor += 1
        if cursor >= length:
            index = close + 1
            continue

        yield _destination(line[close + 2 : cursor]), "image" if is_image else "link"
        index = cursor + 1


def _attribute_value(raw: str) -> str:
    return raw[1:-1] if raw[:1] in {'"', "'"} else raw


def _html_destinations(line: str) -> Iterator[tuple[str, str]]:
    """Yield ``(destination, kind)`` for ``<img src>`` / ``<source srcset>``."""

    for tag in HTML_ASSET_TAG_RE.finditer(line):
        for match in HTML_SRC_RE.finditer(tag.group(0)):
            yield _attribute_value(match.group(1)).strip(), "img-src"
        for match in HTML_SRCSET_RE.finditer(tag.group(0)):
            value = _attribute_value(match.group(1))
            if "data:" in value:
                # commas separate candidates *and* live inside data URIs
                continue
            for candidate in value.split(","):
                parts = candidate.split()
                if parts:
                    yield parts[0], "srcset"


def _is_local_asset(destination: str, kind: str) -> bool:
    if not destination or destination.startswith(("#", "/")):
        return False
    if SCHEME_RE.match(destination):
        return False
    if kind != "link":
        return True
    path = destination.split("#", 1)[0].split("?", 1)[0]
    return os.path.splitext(path)[1].lower() in ASSET_SUFFIXES


def iter_references(text: str) -> Iterator[Reference]:
    """Yield every local asset reference outside fenced blocks and code spans."""

    fence_char: str | None = None
    fence_length = 0

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        fence = FENCE_RE.match(raw_line)
        if fence:
            marker = fence.group("fence")
            if fence_char is None:
                fence_char = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                fence_char = None
                fence_length = 0
            continue
        if fence_char is not None:
            continue

        line = _mask_inline_code(raw_line)
        for destination, kind in (
            *_inline_destinations(line),
            *_html_destinations(line),
        ):
            if _is_local_asset(destination, kind):
                yield Reference(destination, line_number, kind)


def _directory_names(directory: Path, cache: dict[Path, frozenset[str]]) -> frozenset[str]:
    names = cache.get(directory)
    if names is None:
        try:
            names = frozenset(os.listdir(directory))
        except OSError:
            names = frozenset()
        cache[directory] = names
    return names


def _walk(
    base: Path, reference: str, cache: dict[Path, frozenset[str]]
) -> tuple[Path, str, str] | None:
    """Resolve ``reference`` under ``base``, reporting the on-disk spelling.

    Each component is matched against the real directory listing rather than by
    ``Path.exists()``: on Windows and macOS a case-wrong path opens happily and
    then 404s on the Linux site build, so the mismatch has to be observed, not
    inherited from the filesystem's own case folding.
    """

    parts = [
        part
        for part in reference.replace("\\", "/").split("/")
        if part not in ("", ".")
    ]
    if not parts:
        return None

    current = base
    spelled: list[str] = []
    for index, part in enumerate(parts):
        if part == "..":
            current = current.parent
            spelled.append(part)
            continue
        names = _directory_names(current, cache)
        if part in names:
            actual = part
        else:
            folded = part.casefold()
            matches = [name for name in names if name.casefold() == folded]
            if len(matches) != 1:
                return None
            actual = matches[0]
        spelled.append(actual)
        current = current / actual
        if index < len(parts) - 1 and not current.is_dir():
            return None

    if not current.is_file():
        return None
    return current, "/".join(spelled), "/".join(parts)


def _locate(
    base: Path, target: str, cache: dict[Path, frozenset[str]]
) -> tuple[Path, str] | None:
    """Return ``(path, on-disk spelling)`` for an existing target, else ``None``.

    A percent-encoded path is retried decoded: the browser decodes it, so
    ``one%20fig.png`` really does reach ``one fig.png``.
    """

    cleaned = target.split("#", 1)[0].split("?", 1)[0]
    candidates = [cleaned]
    decoded = unquote(cleaned)
    if decoded != cleaned:
        candidates.append(decoded)
    for candidate in candidates:
        found = _walk(base, candidate, cache)
        if found is not None:
            path, spelled, requested = found
            return path, "" if spelled == requested else spelled
    return None


def check_text(text: str, path: str = "<memory>", base: Path | None = None) -> list[Diagnostic]:
    """Return deterministic asset diagnostics for one Markdown document."""

    directory = base if base is not None else Path(path).parent
    cache: dict[Path, frozenset[str]] = {}
    diagnostics: list[Diagnostic] = []

    for reference in iter_references(text):
        located = _locate(directory, reference.target, cache)
        if located is None:
            diagnostics.append(
                Diagnostic(
                    "error",
                    "ASSET001",
                    path,
                    reference.line,
                    f"missing local asset: '{reference.target}'",
                )
            )
            continue
        _, mismatch = located
        if mismatch:
            diagnostics.append(
                Diagnostic(
                    "warning",
                    "ASSET101",
                    path,
                    reference.line,
                    "asset path case does not match on disk: "
                    f"'{reference.target}' (found '{mismatch}')",
                )
            )

    return sorted(
        diagnostics,
        key=lambda item: (item.path, item.line, item.code, item.message),
    )


def check_file(path: Path) -> list[Diagnostic]:
    return check_text(
        path.read_text(encoding="utf-8", errors="replace"), str(path), path.parent
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check that local asset references resolve to real files."
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
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    args = _parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    files = set(_collect_explicit_paths(args.paths))
    if args.changed or not args.paths:
        try:
            files.update(collect_changed_markdown(repo_root))
        except subprocess.CalledProcessError as error:
            print(error.stderr.decode("utf-8", errors="replace").strip())
            return 2

    diagnostics = [
        diagnostic for path in sorted(files) for diagnostic in check_file(path)
    ]
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
