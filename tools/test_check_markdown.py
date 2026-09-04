"""Markdown 渲染陷阱检查器的行为测试。

这些规则原本只作用于 `tools/labs_torch_compile/test_volume_demo_contract.py`
里写死的四个目录；它们本身对全仓成立，故升格为独立检查器。测试一律用内联
fixture 构造，不绑定 `wiki/` 下的任何真实页面。
"""

import check_markdown


def codes(markdown: str) -> list[str]:
    return [item.code for item in check_markdown.check_text(markdown, "note.md")]


def test_accepts_well_formed_lists_and_mermaid():
    markdown = """---
title: 示例
---

- 正确的无序项
- 另一项

1. 正确的有序项
2. 另一项

分隔线：

---

```mermaid
flowchart TD
    A -->|label| B
    C -- plain --> D
```
"""
    assert check_markdown.check_text(markdown, "good.md") == []


def test_reports_unordered_marker_without_space():
    diagnostics = check_markdown.check_text("-没有空格\n", "bad.md")

    assert [item.code for item in diagnostics] == ["MD001"]
    assert diagnostics[0].line == 1
    assert diagnostics[0].severity == "error"


def test_accepts_horizontal_rule_and_frontmatter_fence():
    assert codes("---\n") == []
    assert codes("--\n") == []


def test_reports_ordered_marker_without_space():
    diagnostics = check_markdown.check_text("1.没有空格\n", "bad.md")

    assert [item.code for item in diagnostics] == ["MD002"]
    assert diagnostics[0].line == 1


def test_reports_indented_ordered_marker_without_space():
    assert codes("正文\n\n    12.缩进的有序项\n") == ["MD002"]


def test_ignores_list_markers_inside_fenced_code():
    markdown = """```python
-not_a_list
1.also_not
```

~~~text
-still_not
~~~
"""
    assert check_markdown.check_text(markdown, "fenced.md") == []


def test_reports_quoted_mermaid_pipe_label():
    markdown = """```mermaid
flowchart TD
    A -->|"quoted"| B
```
"""
    diagnostics = check_markdown.check_text(markdown, "diagram.md")

    assert [item.code for item in diagnostics] == ["MD003"]
    assert diagnostics[0].line == 3


def test_reports_quoted_mermaid_inline_label():
    markdown = """```mermaid
flowchart TD
    A -- "quoted" --> B
```
"""
    assert codes(markdown) == ["MD003"]


def test_ignores_mermaid_shaped_text_outside_a_mermaid_fence():
    markdown = """正文里写 `A -->|"quoted"| B` 只是示例。

```text
A -->|"quoted"| B
```
"""
    assert check_markdown.check_text(markdown, "prose.md") == []


def test_reports_every_offending_line_with_its_own_number():
    markdown = "-一\n\n正常段落\n\n2.二\n"

    diagnostics = check_markdown.check_text(markdown, "multi.md")

    assert [(item.code, item.line) for item in diagnostics] == [
        ("MD001", 1),
        ("MD002", 5),
    ]


def test_render_is_line_addressable():
    diagnostic = check_markdown.check_text("-x\n", "wiki/page.md")[0]

    assert diagnostic.render() == (
        "wiki/page.md:1 [ERROR MD001] "
        "无序列表标记 `-` 后缺少空格，CommonMark 不会渲染成列表"
    )


def test_ignores_latex_lines_inside_display_math():
    """行首负号在 $$ 块里是 LaTeX，不是列表标记。"""

    markdown = (
        "公式：\n\n"
        "$$\n"
        r"f(x) = \begin{cases}" "\n"
        r"-\infty, & \text{otherwise}" "\n"
        r"\end{cases}" "\n"
        "$$\n"
    )
    assert check_markdown.check_text(markdown, "math.md") == []


def test_ignores_decimal_and_paper_numbers_at_line_start():
    """`2.8T`、`2605.14220` 是数字与论文编号，不是有序列表。"""

    assert codes("2.8T 总参数可以理解为规模化后的结果。\n") == []
    assert codes("2605.14220 归因清单的第一类，原文：\n") == []


def test_mermaid_quoted_label_is_a_warning_not_an_error():
    """技能文档定性为『渲染器相关，求稳避免』，且明说旧图不必大改。"""

    diagnostic = check_markdown.check_text(
        '```mermaid\nflowchart TD\n    A -->|"是"| B\n```\n', "diagram.md"
    )[0]

    assert diagnostic.code == "MD003"
    assert diagnostic.severity == "warning"


def test_changed_mode_ignores_findings_already_in_the_base():
    """存量 MD003 随行号漂移也不算新问题——技能文档明说旧图不必重写。"""
    base = '```mermaid\nflowchart TD\n    A -->|"是"| B\n```\n'
    current = '新增一段正文。\n\n' + base

    assert check_markdown.new_diagnostics_since(base, current, "page.md") == []


def test_changed_mode_reports_newly_introduced_findings():
    base = "```mermaid\nflowchart TD\n    A -->|是| B\n```\n"
    current = '```mermaid\nflowchart TD\n    A -->|"是"| B\n```\n'

    introduced = check_markdown.new_diagnostics_since(base, current, "page.md")

    assert [item.code for item in introduced] == ["MD003"]


def test_changed_mode_counts_an_added_duplicate_of_an_existing_finding():
    base = '```mermaid\nflowchart TD\n    A -->|"是"| B\n```\n'
    current = '```mermaid\nflowchart TD\n    A -->|"是"| B\n    C -->|"是"| D\n```\n'

    assert len(check_markdown.new_diagnostics_since(base, current, "page.md")) == 1
