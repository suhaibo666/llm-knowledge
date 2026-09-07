"""Markdown 数学公式检查器的行为测试。"""

from pathlib import Path

import check_math


def codes(markdown: str) -> set[str]:
    return {item.code for item in check_math.check_text(markdown, "note.md")}


def test_accepts_canonical_inline_and_display_math():
    markdown = """行内公式 $p_\\theta(x)$。

$$
\\begin{aligned}
x &= y + 1, \\\\
z &= x^2.
\\end{aligned}
$$
"""
    assert check_math.check_text(markdown, "good.md") == []


def test_reports_legacy_math_delimiters():
    diagnostics = check_math.check_text(
        "行内 \\(x+y\\)。\n\n\\[\nx+y\n\\]\n", "legacy.md"
    )
    legacy = [item for item in diagnostics if item.code == "MATH001"]
    assert len(legacy) == 4
    assert {item.line for item in legacy} == {1, 3, 5}
    assert all(item.severity == "error" for item in legacy)


def test_reports_unclosed_display_and_inline_math():
    assert "MATH002" in codes("$$\nx+y\n")
    assert "MATH003" in codes("这里有未闭合公式 $x+y。\n")


def test_ignores_fenced_code_inline_code_and_currency():
    markdown = r"""价格约为 $5.5M，示例 `\(x+y\)` 不检查。

```markdown
\[
accept\_rate
$$
```
"""
    assert check_math.check_text(markdown, "examples.md") == []


def test_reports_brace_imbalance_inside_math():
    assert "MATH004" in codes("$x_{i$\n")


def test_escaped_underscore_is_only_allowed_in_text_commands():
    allowed = r"指标名 $\texttt{accept\_rate}$。"
    warned = r"错误变量 $accept\_rate$。"
    assert "MATH102" not in codes(allowed)
    diagnostics = check_math.check_text(warned, "identifier.md")
    assert any(
        item.code == "MATH102" and item.severity == "warning"
        for item in diagnostics
    )


def test_reports_raw_pipe_in_table_math():
    markdown = "| 条件分布 | 含义 |\n|---|---|\n| $p(x|y)$ | 示例 |\n"
    diagnostics = check_math.check_text(markdown, "table.md")
    assert any(
        item.code == "MATH005" and item.severity == "error"
        for item in diagnostics
    )


def test_warns_when_display_delimiters_share_formula_line():
    diagnostics = check_math.check_text("$$x+y$$\n", "compact.md")
    assert any(
        item.code == "MATH101" and item.severity == "warning"
        for item in diagnostics
    )


def test_cli_strict_turns_warnings_into_failure(tmp_path: Path, monkeypatch):
    note = tmp_path / "note.md"
    note.write_text("$$x+y$$\n", encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["check_math.py", str(note)])
    assert check_math.main() == 0

    monkeypatch.setattr("sys.argv", ["check_math.py", "--strict", str(note)])
    assert check_math.main() == 1


def test_delta_ignores_preexisting_diagnostic_after_line_shift():
    base = "旧记录。\n历史公式 \\(x+y\\)。\n下一行。\n"
    current = "新增的干净记录。\n" + base

    assert check_math.new_diagnostics_since(base, current, "changelog.md") == []


def test_delta_reports_new_issue_and_deleted_display_closer():
    clean_base = "旧记录。\n"
    new_legacy = clean_base + "新增公式 \\(z\\)。\n"
    assert {
        item.code
        for item in check_math.new_diagnostics_since(
            clean_base, new_legacy, "changelog.md"
        )
    } == {"MATH001"}

    closed_display = "$$\nx+y\n$$\n"
    unclosed_display = "$$\nx+y\n"
    assert "MATH002" in {
        item.code
        for item in check_math.new_diagnostics_since(
            closed_display, unclosed_display, "note.md"
        )
    }


def test_index_arithmetic_subscripts_stay_italic():
    """索引下标保持斜体：MATH103 只针对词语/角色/配置标签。

    `h_{t-1}` 是索引表达式，不是语义标签；把它强制改成直立体是错的。
    """

    index_only = r'行内 $h_{t-1} = W x_{k-1} + b_{s+r-1}$。' + chr(10)
    assert "MATH103" not in codes(index_only)

    semantic = r'行内 $C_{low} + p_{train-old}$。' + chr(10)
    assert "MATH103" in codes(semantic)


def test_display_block_needs_blank_lines_around_it():
    """`$$` 紧贴正文时 Python-Markdown 不会交给 arithmatex。

    Obsidian 不要求空行，站点这条链要求。缺空行时 `$$` 会以字面文本落进
    `<p>`，LaTeX 改走行内解析器，`\\\\` 与 `\\{` 会被吃掉——公式在页面上
    直接算错，而不只是晚一步渲染。
    """

    canonical = "文字：\n\n$$\ny = W x\n$$\n\n后续。\n"
    assert check_math.check_text(canonical, "good.md") == []

    missing_before = check_math.check_text(
        "文字：\n$$\ny = W x\n$$\n\n后续。\n", "note.md"
    )
    assert [(item.code, item.line) for item in missing_before] == [("MATH006", 2)]

    missing_after = check_math.check_text(
        "文字：\n\n$$\ny = W x\n$$\n后续。\n", "note.md"
    )
    assert [(item.code, item.line) for item in missing_after] == [("MATH006", 5)]

    assert check_math.check_text("$$\ny = W x\n$$\n", "note.md") == []


def test_display_block_blank_line_rule_counts_quote_markers_as_empty():
    """callout 里的分隔行是 `>`，空行会截断引用块。"""

    quoted = "> 文字：\n>\n> $$\n> y = W x\n> $$\n>\n> 后续。\n"
    assert check_math.check_text(quoted, "quote.md") == []

    assert {
        item.code
        for item in check_math.check_text(
            "> 文字：\n> $$\n> y = W x\n> $$\n> 后续。\n", "quote.md"
        )
    } == {"MATH006"}


def test_display_block_under_a_list_item_needs_four_space_indent():
    """CommonMark 用 list marker 宽度做续行缩进，Python-Markdown 固定要 4。"""

    assert [
        (item.code, item.line)
        for item in check_math.check_text(
            "1. 步骤\n\n   $$\n   S_t = D_t\n   $$\n\n2. 下一步\n", "list.md"
        )
    ] == [("MATH007", 3)]

    four_spaces = "1. 步骤\n\n    $$\n    S_t = D_t\n    $$\n\n2. 下一步\n"
    assert check_math.check_text(four_spaces, "list.md") == []


def test_display_block_rules_ignore_fenced_examples():
    fenced = "```markdown\n文字：\n$$\ny = W x\n$$\n```\n"
    assert check_math.check_text(fenced, "fence.md") == []
