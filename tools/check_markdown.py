"""check_markdown —— Markdown 渲染陷阱检查器。

三条规则原本写死在 `tools/labs_torch_compile/test_volume_demo_contract.py`
的 `CourseMarkdownContractTest` 里，只作用于四个历史遗留目录（那段代码的注释
本身就在流水账式记录 kb-reorg 把哪些卷搬到了哪）。规则本身对全仓 446 页都成立，
被限定在四个目录纯属历史包袱，故升格为独立检查器：覆盖面从 4 个目录扩到全仓，
成本从一次 38 秒的 pytest 降到毫秒级，并支持按改动增量运行。

检查项:
  MD001  无序列表标记 `-` 后缺少空格 —— CommonMark 不渲染成列表
  MD002  有序列表标记 `N.` 后缺少空格 —— 同上
  MD003  mermaid 标签内嵌引号 —— mermaid 解析器会炸

用法:
  python tools/check_markdown.py                 # 检查改动的 Markdown
  python tools/check_markdown.py wiki            # 全量
  python tools/check_markdown.py --changed --strict
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from check_math import (
    _collect_explicit_paths,
    _read_head_text,
    collect_changed_markdown,
)

# `-` 之后必须是空格/制表符才是列表；排除 `-` 自身，放过 `--`、`---`(分隔线与 frontmatter)。
UNORDERED_MARKER = re.compile(r"^-[^ \t-]")
# 排除 `2.8T`(小数)与 `2605.14220`(论文编号):真正的列表标记后面不会紧跟数字。
ORDERED_MARKER = re.compile(r"^\s*\d+\.(?![ \t\d])")
# mermaid 的边标签一旦内嵌引号，解析器会在该行中断整张图。
MERMAID_PIPE_LABEL = re.compile(r"\|[\"'][^|]*[\"']\|")
MERMAID_INLINE_LABEL = re.compile(
    r"(?:--|==|-\.)(?:\s+)[\"'][^\"']+[\"'](?:\s+)(?:-->|==>|\.->)"
)

FENCE_MARKERS = {"```", "~~~"}
MERMAID_OPENER = "```mermaid"
# `$$` 块里 `-\infty`、`-\rho_t A_t` 这类行首负号是 LaTeX，不是列表标记。
DISPLAY_MATH = "$$"

MESSAGES = {
    "MD001": "无序列表标记 `-` 后缺少空格，CommonMark 不会渲染成列表",
    "MD002": "有序列表标记 `N.` 后缺少空格，CommonMark 不会渲染成列表",
    "MD003": "mermaid 标签内嵌引号，个别渲染器版本会解析失败",
}

# MD001/MD002 是确定性的渲染错误。MD003 不是:
# `skills/writing-mermaid-diagrams` 把带引号的管道标签定性为「渲染器相关，求稳一律
# 避免」——多数 mermaid 版本能渲，并明说「已能渲的旧图不必为此大改，但新图按此写」。
# 报 warning 正好落在这个位置:新图靠 --strict 拦住，存量不当作缺陷。
SEVERITIES = {"MD001": "error", "MD002": "error", "MD003": "warning"}


@dataclass
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


def _finding(code: str, path: str, line: int) -> Diagnostic:
    return Diagnostic(SEVERITIES[code], code, path, line, MESSAGES[code])


def check_text(text: str, path: str) -> list[Diagnostic]:
    """Report render traps in one Markdown document."""

    diagnostics: list[Diagnostic] = []
    fence: str | None = None
    in_mermaid = False
    in_math = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        marker = line.lstrip()[:3]
        if marker in FENCE_MARKERS:
            if fence is None:
                fence = marker
                in_mermaid = stripped == MERMAID_OPENER
            elif marker == fence:
                fence = None
                in_mermaid = False
            continue
        if in_mermaid:
            if MERMAID_PIPE_LABEL.search(line) or MERMAID_INLINE_LABEL.search(line):
                diagnostics.append(_finding("MD003", path, line_number))
            continue
        if fence is not None:
            continue
        if stripped == DISPLAY_MATH:
            in_math = not in_math
            continue
        if in_math:
            continue
        if UNORDERED_MARKER.match(line):
            diagnostics.append(_finding("MD001", path, line_number))
        if ORDERED_MARKER.match(line):
            diagnostics.append(_finding("MD002", path, line_number))
    return diagnostics


def check_file(path: Path) -> list[Diagnostic]:
    return check_text(path.read_text(encoding="utf-8"), str(path))


def _fingerprint(diagnostic: Diagnostic, text: str) -> tuple[str, str]:
    """按「规则 + 出问题的那行内容」定身份，行号漂移不算新问题。"""

    lines = text.splitlines()
    offending = lines[diagnostic.line - 1] if diagnostic.line <= len(lines) else ""
    return diagnostic.code, offending.strip()


def new_diagnostics_since(
    base_text: str, current_text: str, path: str
) -> list[Diagnostic]:
    """Return findings introduced since base_text, ignoring shifted legacy debt."""

    base_counts = Counter(
        _fingerprint(item, base_text) for item in check_text(base_text, path)
    )
    introduced: list[Diagnostic] = []
    for item in check_text(current_text, path):
        fingerprint = _fingerprint(item, current_text)
        if base_counts[fingerprint]:
            base_counts[fingerprint] -= 1
        else:
            introduced.append(item)
    return introduced


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check Markdown for list and mermaid render traps."
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
    # 显式给路径 = 要全量结论；--changed = 只对本次改动负责，存量按 HEAD 扣掉。
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
    sys.exit(main())
