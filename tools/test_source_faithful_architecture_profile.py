import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "skills/source-faithful-analysis"
SOURCE_CORE = SOURCE_ROOT / "SKILL.md"
SOURCE_PACK = SOURCE_ROOT / "references/codebase.md"
ARCHITECTURE = SOURCE_ROOT / "references/document-types/software-architecture.md"
REVIEW = SOURCE_ROOT / "references/reviews/software-architecture.md"
EVALS = SOURCE_ROOT / "evals/software-architecture.json"
OLD_SKILL_ROOT = REPO_ROOT / "skills/analyzing-software-architecture"
PLANNER_ROOT = REPO_ROOT / "skills/planning-codebase-analysis"
PLANNER = PLANNER_ROOT / "SKILL.md"
PLANNER_SOURCE_REFERENCE = "../source-faithful-analysis/SKILL.md"
FEATURE_TREE = REPO_ROOT / "skills/feature-tree-analysis/SKILL.md"
ROUTE_DOCS = (
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "CLAUDE.md",
    REPO_ROOT / "skills/README.md",
)


def _text(path):
    return path.read_text(encoding="utf-8")


def _squash(text):
    return " ".join(text.split())


def _route_row(path, skill_name):
    return next(
        line
        for line in _text(path).splitlines()
        if line.startswith("|") and f"`{skill_name}`" in line
    )


def test_architecture_is_a_profile_not_a_second_top_level_skill():
    assert ARCHITECTURE.is_file()
    assert REVIEW.is_file()
    assert EVALS.is_file()
    assert not OLD_SKILL_ROOT.exists()

    router = _text(SOURCE_CORE)
    assert "`references/document-types/software-architecture.md`" in router
    assert "`references/reviews/software-architecture.md`" in router


def test_architecture_routes_resolve_through_source_faithful_analysis():
    for path in ROUTE_DOCS:
        row = _route_row(path, "source-faithful-analysis")
        if path.name == "README.md":
            for phrase in ("软件架构", "特性", "机制", "按需"):
                assert phrase in row
        else:
            for phrase in ("software architecture", "feature", "mechanism", "on demand"):
                assert phrase in row.lower()
        assert "analyzing-software-architecture" not in _text(path)

    planner = _text(PLANNER)
    assert f"`{PLANNER_SOURCE_REFERENCE}`" in planner
    assert (PLANNER_ROOT / PLANNER_SOURCE_REFERENCE).resolve().is_file()
    assert "software-architecture" in planner
    assert "analyzing-software-architecture" not in planner

    feature_tree = _squash(_text(FEATURE_TREE))
    assert "architecture page → `source-faithful-analysis` / `software-architecture`" in feature_tree
    assert "mechanism page → `source-faithful-analysis` / `mechanism-analysis`" in feature_tree


def test_architecture_profile_keeps_the_six_reader_questions_in_order():
    contract = _text(ARCHITECTURE)
    ordered_sections = (
        "## 1. Establish the software boundary",
        "## 2. Build architecture before motion",
        "## 3. Carry the partition into module design",
        "## 4. Make source call relationships exact",
        "## 5. Map architecture to physical code",
        "## 6. Specify every usage scenario",
        "## Sizing and ownership gate",
        "## Completion gate",
    )
    positions = [contract.index(heading) for heading in ordered_sections]
    assert positions == sorted(positions)

    boundary = contract.split(ordered_sections[0], 1)[1].split(ordered_sections[1], 1)[0]
    for phrase in ("design background", "design principles", "capability range", "unsupported"):
        assert phrase in boundary

    views = contract.split(ordered_sections[1], 1)[1].split(ordered_sections[2], 1)[0]
    assert "### Static capability view" in views
    assert "### Dynamic implementation view" in views
    assert "Every peer box must be the same semantic kind" in views

    modules = contract.split(ordered_sections[2], 1)[1].split(ordered_sections[3], 1)[0]
    for slot in ("Responsibility and contract", "Design pressure and choice", "Internal implementation", "Constraints and limits"):
        assert slot in modules

    code_map = contract.split(ordered_sections[4], 1)[1].split(ordered_sections[5], 1)[0]
    assert "explicitly many-to-many" in code_map
    assert "Load-bearing symbols" in code_map

    scenarios = contract.split(ordered_sections[5], 1)[1].split(ordered_sections[6], 1)[0]
    normalized = _squash(scenarios).lower()
    fields = re.findall(r"^\d+\. \*\*(.+?):\*\*", scenarios, flags=re.MULTILINE)
    assert fields == ["Execution entry", "Command template", "Function calls", "Software logic", "Completion and limits"]
    for phrase in ("fenced `text` ascii caller tree", "verify the script", "verify every flag", "documentation and implementation disagree", "known failure path"):
        assert phrase in normalized


def test_architecture_presentation_delegates_generic_execution_semantics():
    contract = _text(ARCHITECTURE)
    calls = _squash(
        contract.split("## 4. Make source call relationships exact", 1)[1].split(
            "## 5. Map architecture to physical code", 1
        )[0]
    )
    source_reference = "../codebase.md"

    assert f"`{source_reference}`" in calls
    assert (ARCHITECTURE.parent / source_reference).resolve().is_file()
    for presentation_rule in ("fenced `text` ASCII caller tree", "sequential siblings", "back under the caller", "owning architecture module"):
        assert presentation_rule in calls
    for generic_rule in ("wrappers, registries, schedulers", "transitive/elided", "local/remote", "sync/async", "settled future"):
        assert generic_rule not in calls

    source_pack = _text(SOURCE_PACK)
    for generic_rule in ("wrapper", "transitive/elided", "local/remote", "sync/async", "Completion"):
        assert generic_rule in source_pack

    review_reference = "../reviews/software-architecture.md"
    completion = contract.split("## Completion gate", 1)[1]
    assert f"`{review_reference}`" in completion
    assert (ARCHITECTURE.parent / review_reference).resolve().is_file()


def test_architecture_review_has_eight_independent_checks_and_base_review():
    review = _text(REVIEW)
    base_reference = "../page-review-rubric.md"
    assert "reviewer is never the writer" in review
    assert f"`{base_reference}`" in review
    assert (REVIEW.parent / base_reference).resolve().is_file()
    assert len(re.findall(r"^\d+\. ", review, flags=re.MULTILINE)) == 8
    flat = _squash(review)
    for phrase in (
        "design background", "goals and non-goals", "current capability boundary", "classification axis",
        "qualified symbols", "many-to-many", "usage-scenario inventory", "every flag",
        "exact-parentage hop-walk", "completion boundary", "reader handoff",
        "what the reader now knows", "authoritative next pages",
    ):
        assert phrase in flat


def test_architecture_profile_has_three_cross_domain_regression_evals():
    payload = json.loads(_text(EVALS))
    assert payload["skill_name"] == "source-faithful-analysis/software-architecture"
    assert {item["id"] for item in payload["evals"]} == {1, 2, 3}
    assert all(item["prompt"] and item["expected_output"] for item in payload["evals"])

    scenarios = "\n".join(f'{item["prompt"]}\n{item["expected_output"]}' for item in payload["evals"])
    for phrase in ("静态责任图和动态生命周期", "ASCII 函数调用树", "实际执行命令", "异步服务", "writer 不能自行拆页"):
        assert phrase in scenarios
