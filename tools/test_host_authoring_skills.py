"""Lock the host-side authoring contracts that `source-faithful-analysis` delegates to.

The analysis skill is deliberately portable: it owns semantic order, not heading templates or
rendering pipelines. That portability only works if the host skills actually carry the pieces it
delegates. Two delegations are load-bearing and were, at one point, carried only by an exemplar
page rather than by any skill:

* `drawing-wiki-figures` is named a REQUIRED SUB-SKILL by the algorithmic-figure gate, so it must
  carry the rules that make a figure trustworthy — including the drift test that actually reads the
  page, and the fact that canvas-bounds checking is not an overlap check.
* `maintaining-llm-knowledge` owns this wiki's page shape. With it unwritten, reproducing a page
  required reading a sibling page, and exemplars drift.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES = REPO_ROOT / "skills/drawing-wiki-figures/SKILL.md"
MAINTAIN = REPO_ROOT / "skills/maintaining-llm-knowledge/SKILL.md"


def _text(path):
    return path.read_text(encoding="utf-8")


def _squash(text):
    return " ".join(text.split())


def test_solver_figure_test_must_read_the_page_not_only_the_svg():
    """Hardcoding a number in the test and asserting it appears in the SVG locks one side only:
    edit the prose alone and the test stays green. Two of three generators shipped that way."""
    figures = _squash(_text(FIGURES))
    assert "测试必须真的去读那个 `.md` 文件" in figures
    assert "readFile" in figures
    # The self-check that distinguishes a real drift test from a one-sided one.
    assert "只改正文、不改图，这个测试会不会红" in figures


def test_canvas_bounds_is_not_an_overlap_check():
    """A chip overlapping a neighbouring column clipped glyphs while staying inside the viewBox."""
    figures = _squash(_text(FIGURES))
    assert "越界不等于不重叠" in figures
    assert "包围盒重叠断言" in figures


def test_house_page_shape_is_written_down_not_carried_by_an_exemplar():
    maintain = _text(MAINTAIN)
    assert "### 分析页的房子形状" in maintain
    squashed = _squash(maintain)
    for field in ("**源码基线**", "**核心源码**", "**中心结论**", "**适用范围**", "**最近更新**"):
        assert field in squashed, f"house header lost {field}"
    # The three recurring table shapes that prose alone kept getting wrong.
    for column in ("必付成本或边界", "破坏后的行为", "字段 \\| 类型 \\| 默认 \\| 契约"):
        assert column in squashed, f"house table shape lost {column}"


def test_house_shape_records_the_coverage_gate_blind_spot():
    """`check_coverage` C2 re-verifies manual owners only; `auto: true` rows skip the mention
    check, so a rewrite can drop an owned field out of the wiki with every gate green."""
    maintain = _squash(_text(MAINTAIN))
    assert "auto: true" in maintain
    assert "门禁挡不住" in maintain


def test_house_shape_defers_method_to_the_portable_analysis_skill():
    """The split matters: if house style leaks into the analysis profile, the profile stops being
    portable to non-wiki deliverables."""
    maintain = _squash(_text(MAINTAIN))
    assert "source-faithful-analysis" in maintain
    assert "只管**语义顺序**、不管标题模板" in maintain
