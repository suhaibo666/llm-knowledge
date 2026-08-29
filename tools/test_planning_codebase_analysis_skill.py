import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER = REPO_ROOT / "skills/planning-codebase-analysis"


def test_planner_entrypoint_declares_scope_and_hard_gate():
    text = (PLANNER / "SKILL.md").read_text(encoding="utf-8")
    assert "name: planning-codebase-analysis" in text
    assert "Use when" in text
    for phrase in (
        "whole codebase",
        "capability map",
        "static architecture",
        "dynamic lifecycle",
        "coverage matrix",
        "user approval",
        "source-faithful-analysis",
    ):
        assert phrase in text, f"planner contract missing {phrase!r}"


def test_planner_has_six_routing_and_behavior_evals():
    payload = json.loads((PLANNER / "evals/evals.json").read_text(encoding="utf-8"))
    assert payload["skill_name"] == "planning-codebase-analysis"
    assert {item["id"] for item in payload["evals"]} == {1, 2, 3, 4, 5, 6}
    assert all(item["prompt"] and item["expected_output"] for item in payload["evals"])


def test_planner_rejects_fixed_source_code_ratio_quotas():
    text = (PLANNER / "SKILL.md").read_text(encoding="utf-8")
    assert (
        "Do not use a fixed source-code or code-to-explanation ratio as a quality "
        "or completion constraint."
    ) in text


ROUTE_DOCS = (
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "CLAUDE.md",
    REPO_ROOT / "skills/README.md",
)
SOURCE_SKILL = REPO_ROOT / "skills/source-faithful-analysis/SKILL.md"
CODEBASE_PACK = REPO_ROOT / "skills/source-faithful-analysis/references/codebase.md"


def test_new_codebase_planner_is_discoverable_from_every_route_table():
    for path in ROUTE_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "planning-codebase-analysis" in text, f"planner is not routed from {path}"
        assert "source-faithful-analysis" in text, f"page-level writer disappeared from {path}"


def test_source_analysis_delegates_unplanned_whole_codebases():
    core = SOURCE_SKILL.read_text(encoding="utf-8")
    pack = CODEBASE_PACK.read_text(encoding="utf-8")
    assert "planning-codebase-analysis" in core
    assert "approved blueprint" in core
    assert "one analysis unit/page" in pack
    assert "not one source file" in pack
