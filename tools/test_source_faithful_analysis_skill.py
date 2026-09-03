import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = REPO_ROOT / "skills/source-faithful-analysis"
CORE = SKILL_ROOT / "SKILL.md"
CODEBASE = SKILL_ROOT / "references/codebase.md"
PAPER = SKILL_ROOT / "references/paper.md"
GENERAL = SKILL_ROOT / "references/general.md"
PARALLEL = SKILL_ROOT / "references/parallel-agent-contract.md"
PAGE_REVIEW = SKILL_ROOT / "references/page-review-rubric.md"
EVALS = SKILL_ROOT / "evals/evals.json"
CONSTITUTION = REPO_ROOT / "CLAUDE.md"
FEATURE_TREE = REPO_ROOT / "skills/feature-tree-analysis/SKILL.md"
FEATURE_TEMPLATE = REPO_ROOT / "skills/feature-tree-analysis/references/feature-point-template.md"


def _text(path):
    return path.read_text(encoding="utf-8")


def _squash(text):
    return " ".join(text.split())


def test_core_and_codebase_pack_stay_lean_after_deduplication():
    core_lines = _text(CORE).splitlines()
    codebase_lines = _text(CODEBASE).splitlines()

    assert len(core_lines) <= 220
    assert len(codebase_lines) <= 170
    assert len(core_lines) + len(codebase_lines) <= 370


def test_generic_rules_have_one_owner_in_the_core():
    core = _text(CORE)
    codebase = _text(CODEBASE)

    assert core.count("## Canonical analysis contract") == 1
    assert core.count("## Workflow") == 1
    assert "## Output: the doc template" not in core
    assert "append a changelog entry" not in core
    assert "Author each diagram as HTML/SVG" not in core

    assert "## Code mechanism contract" in codebase
    assert "## Essence checklist" not in codebase
    assert "## Doc structure" not in codebase


def test_core_preserves_the_load_bearing_source_and_reasoning_contract():
    core = _squash(_text(CORE)).lower()
    required = (
        "verified source evidence",
        "frozen baseline",
        "source fact",
        "analyst inference",
        "central thesis",
        "obvious alternative",
        "mechanism and evidence",
        "constraints and failure boundary",
        "planning-codebase-analysis",
    )
    for phrase in required:
        assert phrase in core, f"core contract lost {phrase!r}"


def test_code_pages_default_to_stable_symbol_anchors_not_line_number_citations():
    core = _squash(_text(CORE)).lower()
    codebase = _squash(_text(CODEBASE)).lower()
    parallel = _squash(_text(PARALLEL)).lower()
    review = _squash(_text(PAGE_REVIEW)).lower()

    assert "stable code anchor" in core
    assert "path + qualified symbol" in codebase
    assert "line numbers are optional" in codebase
    assert "compact source-reading route" in codebase
    assert "every non-trivial claim is tied to a verified exact locator" not in core
    assert "verified `file:line` for each semantic hop" not in codebase
    assert "do not attach `file:line` to every claim or semantic hop" in parallel
    assert "line-number citations are not required" in review


def test_legacy_line_checker_is_conditional_and_feature_specs_use_symbol_anchors():
    constitution = _text(CONSTITUTION)
    always_on_gates = constitution.split("```bash", 1)[1].split("```", 1)[0]
    feature_tree = _squash(_text(FEATURE_TREE)).lower()
    template = _squash(_text(FEATURE_TEMPLATE)).lower()

    assert "python tools/check_locators.py" not in always_on_gates
    assert "conditional legacy gate" in constitution
    assert "stable source anchors" in feature_tree
    assert "path::qualified.symbol" in template
    assert "each with a locator" not in template


def test_codebase_pack_requires_traceable_execution_semantics():
    pack = _squash(_text(CODEBASE)).lower()
    required = (
        "actual call path",
        "caller → callee",
        "crossing object",
        "pre-state",
        "post-state",
        "state owner",
        "local/remote",
        "sync/async",
        "blocking/non-blocking",
        "completion signal",
        "externally visible",
        "partial side effects",
        "rollback",
        "collapse pure forwarding helpers",
    )
    for phrase in required:
        assert phrase in pack, f"execution trace contract lost {phrase!r}"


def test_codebase_depth_profiles_are_conditional_not_page_templates():
    pack = _squash(_text(CODEBASE)).lower()
    required = (
        "multiple live implementations",
        "data representation",
        "asynchronous, concurrent",
        "runtime verification",
        "only when the trigger is present",
    )
    for phrase in required:
        assert phrase in pack, f"conditional depth routing lost {phrase!r}"


def test_source_analysis_has_four_regression_evals():
    payload = json.loads(EVALS.read_text(encoding="utf-8"))
    assert payload["skill_name"] == "source-faithful-analysis"
    assert {item["id"] for item in payload["evals"]} == {1, 2, 3, 4}
    assert all(item["prompt"] and item["expected_output"] for item in payload["evals"])

    scenarios = "\n".join(
        f'{item["prompt"]}\n{item["expected_output"]}' for item in payload["evals"]
    )
    for phrase in (
        "是什么、怎么做、为什么",
        "提交、执行完成和下游可见",
        "数据生命周期",
        "静态责任图和动态生命周期",
    ):
        assert phrase in scenarios, f"missing regression scenario for {phrase!r}"


def test_source_type_packs_instantiate_instead_of_redefining_the_core():
    paper = _text(PAPER)
    general = _text(GENERAL)

    assert len(paper.splitlines()) <= 115
    assert len(general.splitlines()) <= 50
    for text in (paper, general):
        assert "## Essence checklist" not in text
        assert "## Doc structure" not in text

    for phrase in (
        "arXiv id + VERSION + date",
        "released artifact",
        "机制 ↔ 源码",
        "✅ 官方已发布",
        "❌ 无任何实现",
    ):
        assert phrase in paper, f"paper pack lost unique contract {phrase!r}"

    for phrase in (
        "Spec / RFC / standard / contract",
        "Dataset / schema / table",
        "Running system / incident / logs / metrics",
        "Business / financial / market report",
    ):
        assert phrase in general, f"general pack lost source kind {phrase!r}"


def test_parallel_contract_is_compact_and_delegates_house_figure_rules():
    parallel = _text(PARALLEL)
    assert len(parallel.splitlines()) <= 90
    assert "SVG→PNG" not in parallel
    assert "<div class=\"diagram\"" not in parallel
    assert "active house figure skill" in parallel
    assert "completion or visibility" in parallel


def test_all_runtime_instruction_files_fit_one_progressive_disclosure_budget():
    instruction_files = (CORE, CODEBASE, PAPER, GENERAL, PARALLEL)
    assert sum(len(_text(path).splitlines()) for path in instruction_files) <= 550
