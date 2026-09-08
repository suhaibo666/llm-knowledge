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
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES = REPO_ROOT / "skills/drawing-wiki-figures/SKILL.md"
MAINTAIN = REPO_ROOT / "skills/maintaining-llm-knowledge/SKILL.md"


def test_shared_skill_prose_is_english_with_exact_output_literals_preserved():
    """Catch untranslated instructions without changing the Chinese wiki header protocol.

    This is a CJK regression check, not a general-purpose natural-language detector.
    Exceptions are exact output labels in the two skills that specify that protocol;
    arbitrary prose or fenced examples are not blanket exemptions.
    """
    literal_paths = {
        "maintaining-llm-knowledge/SKILL.md",
        "feature-tree-analysis/SKILL.md",
    }
    allowed_labels = {"源码基线", "主题", "适用范围", "最近更新"}
    failures = []
    root = REPO_ROOT / "skills"
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".md", ".json", ".yaml", ".yml"}:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if path.relative_to(root).as_posix() in literal_paths:
                for label in allowed_labels:
                    line = line.replace(f"**{label}**", "")
            if re.search(r"[\u3400-\u9fff]", line):
                failures.append(f"{path.relative_to(root)}:{lineno}")
    assert not failures, "Untranslated skill prose: " + ", ".join(failures)


def _text(path):
    return path.read_text(encoding="utf-8")


def _squash(text):
    return " ".join(text.split())


def test_solver_figure_test_must_read_the_page_not_only_the_svg():
    """Hardcoding a number in the test and asserting it appears in the SVG locks one side only:
    edit the prose alone and the test stays green. Two of three generators shipped that way."""
    figures = _squash(_text(FIGURES))
    assert "The test must actually read the `.md` file" in figures
    assert "readFile" in figures
    # The self-check that distinguishes a real drift test from a one-sided one.
    assert "If only the prose changed, would this test fail?" in figures


def test_canvas_bounds_is_not_an_overlap_check():
    """A chip overlapping a neighbouring column clipped glyphs while staying inside the viewBox."""
    figures = _squash(_text(FIGURES))
    assert "Canvas bounds do not prove non-overlap" in figures
    assert "element-to-element bounding-box overlaps" in figures


def test_house_page_shape_is_written_down_not_carried_by_an_exemplar():
    maintain = _text(MAINTAIN)
    assert "### House Page Shape for Analysis Pages" in maintain
    squashed = _squash(maintain)
    for field in ("**源码基线**", "**主题**", "**适用范围**", "**最近更新**"):
        assert field in squashed, f"house header lost {field}"
    # The header states the page subject, never previews its thesis: the old 中心结论 line
    # made readers digest exceptions and qualifiers before reading a single section.
    assert "does not preview the argument" in squashed
    # The three recurring table shapes that prose alone kept getting wrong.
    for column in ("Required cost or boundary", "Behavior when violated", "Field \\| Type \\| Default \\| Contract"):
        assert column in squashed, f"house table shape lost {column}"


def test_house_shape_records_the_coverage_gate_blind_spot():
    """`check_coverage` C2 re-verifies manual owners only; `auto: true` rows skip the mention
    check, so a rewrite can drop an owned field out of the wiki with every gate green."""
    maintain = _squash(_text(MAINTAIN))
    assert "auto: true" in maintain
    assert "manual/process review beyond the automated gate" in maintain


def test_house_shape_defers_method_to_the_portable_analysis_skill():
    """The split matters: if house style leaks into the analysis profile, the profile stops being
    portable to non-wiki deliverables."""
    maintain = _squash(_text(MAINTAIN))
    assert "source-faithful-analysis" in maintain
    assert "defines **semantic order**, not a heading template" in maintain
