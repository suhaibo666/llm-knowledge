import json
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER = REPO_ROOT / "skills/planning-codebase-analysis"


def _squash(text):
    return " ".join(text.split())


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

    frontmatter = text.split("---", 2)[1]
    assert "codebase-domain-level replanning" in frontmatter
    assert "knowledge-domain-level replanning" not in frontmatter


def test_planner_blocks_every_persisted_mutation_before_approval():
    text = (PLANNER / "SKILL.md").read_text(encoding="utf-8")
    hard_gate = _squash(text.split("## Hard gate", 1)[1].split("## Workflow", 1)[0])
    assert "Read-only repository and existing-Wiki inspection is allowed" in hard_gate
    for mutation in (
        "page body",
        "stub",
        "frontmatter",
        "create",
        "rename",
        "rewrite",
        "directory",
        "index",
        "changelog",
        "persisted plan",
    ):
        assert mutation in hard_gate, f"pre-approval gate does not cover {mutation!r}"
    assert "Workflow persistence begins only after approval" in hard_gate


def test_planner_has_six_routing_and_behavior_evals():
    payload = json.loads((PLANNER / "evals/evals.json").read_text(encoding="utf-8"))
    assert payload["skill_name"] == "planning-codebase-analysis"
    assert {item["id"] for item in payload["evals"]} == {1, 2, 3, 4, 5, 6}
    assert all(item["prompt"] and item["expected_output"] for item in payload["evals"])

    joined = [f'{item["prompt"]}\n{item["expected_output"]}' for item in payload["evals"]]
    scenario_categories = {
        "unplanned whole codebase": ("新框架仓库", "能力地图", "覆盖矩阵"),
        "focused mechanism": ("scheduler", "不重新规划整个代码库", "跨文件"),
        "approved page contract": ("已经批准的蓝图", "不修改目录", "不包含内容"),
        "directory-mirroring pressure": ("五个一级目录", "拒绝", "跨目录"),
        "material ownership drift": ("不可拆分", "具体修订页面边界", "普通措辞"),
        "scale and ratio pressure": ("数千个文件", "1:1", "固定源码/讲解比例"),
    }
    for category, fragments in scenario_categories.items():
        assert any(all(fragment in scenario for fragment in fragments) for scenario in joined), (
            f"missing eval scenario category: {category}"
        )


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


def _route_row(path, skill_name):
    return next(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("|") and f"`{skill_name}`" in line
    )


def test_route_rows_preserve_codebase_planner_and_all_scale_source_meanings():
    for path in ROUTE_DOCS:
        planner_row = _route_row(path, "planning-codebase-analysis")
        source_row = _route_row(path, "source-faithful-analysis")
        if path.name == "README.md":
            planner_fragments = ("代码库", "多页", "蓝图", "确认")
            source_fragments = (
                "已批准/聚焦代码库",
                "论文",
                "规范",
                "数据集",
                "事故",
                "报告",
                "其他非代码制品",
                "任意规模",
            )
        else:
            planner_fragments = ("codebase", "multi-page", "blueprint", "approval")
            source_fragments = (
                "approved/focused codebase",
                "paper",
                "spec",
                "dataset",
                "incident",
                "report",
                "other non-code artifact",
                "any scale",
            )
        for fragment in planner_fragments:
            assert fragment in planner_row, f"planner route in {path} is missing {fragment!r}"
        for fragment in source_fragments:
            assert fragment in source_row, f"source route in {path} is missing {fragment!r}"


def test_source_analysis_delegates_unplanned_whole_codebases():
    core = SOURCE_SKILL.read_text(encoding="utf-8")
    pack = CODEBASE_PACK.read_text(encoding="utf-8")
    assert "planning-codebase-analysis" in core
    assert "approved blueprint" in core
    assert "one analysis unit/page" in pack
    assert "not one source file" in pack


def test_approved_codebase_pages_inherit_and_freeze_planner_baseline():
    planner = (PLANNER / "SKILL.md").read_text(encoding="utf-8")
    core = _squash(SOURCE_SKILL.read_text(encoding="utf-8"))
    pack = _squash(CODEBASE_PACK.read_text(encoding="utf-8")).lower()
    assert "approved repository commit" in planner
    assert "inherit the approved repository commit" in core
    assert "verify that the checkout is at that exact commit" in core
    assert "inherit the approved repository commit" in pack
    for operation in ("fetch", "pull", "fast-forward", "switch", "checkout", "reset", "move"):
        assert operation in pack, f"approved baseline movement rule omits {operation!r}"
    assert "do not fetch, pull, fast-forward, switch, checkout, reset, or move it" in pack
    assert "return to `planning-codebase-analysis`" in pack
    assert "focused code analysis with no approved blueprint" in pack


def test_approved_page_may_organize_sections_but_not_decompose_documents():
    core = SOURCE_SKILL.read_text(encoding="utf-8")
    assert core.count("### Phase 1") == 1
    after_phase_one = core.split("### Phase 1", 1)[1]
    assert "### Phase 2" in after_phase_one
    phase_one = _squash(after_phase_one.split("### Phase 2", 1)[0])
    assert "non-code sources or unplanned focused analysis" in phase_one
    assert "approved codebase page may organize sections" in phase_one
    assert "must not rename, split, or reassign pages locally" in phase_one
    assert "required page split is material drift" in phase_one
    assert "return it to `planning-codebase-analysis`" in phase_one


def test_replanning_is_limited_to_authoritative_boundary_or_coverage_changes():
    text = (PLANNER / "SKILL.md").read_text(encoding="utf-8")
    replanning = _squash(text.split("## Replanning gate", 1)[1].split("## Completion gate", 1)[0])
    assert "authoritative/core concept ownership" in replanning
    assert "materially alters page boundaries or coverage" in replanning
    assert "Ordinary wording and link-only edits remain local" in replanning


def test_planner_names_only_tracked_routed_downstream_skills():
    text = (PLANNER / "SKILL.md").read_text(encoding="utf-8")
    workflow = text.split("## Workflow", 1)[1].split("## Blueprint contract", 1)[0]
    named_skills = set(re.findall(r"`([a-z][a-z0-9-]+)`", workflow))
    assert named_skills == {
        "source-faithful-analysis",
        "maintaining-llm-knowledge",
        "writing-obsidian-math",
        "writing-mermaid-diagrams",
    }

    for skill_name in named_skills:
        skill_path = f"skills/{skill_name}/SKILL.md"
        assert (REPO_ROOT / skill_path).is_file(), f"missing downstream skill: {skill_path}"
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", skill_path],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert tracked.returncode == 0, f"untracked downstream skill: {skill_path}"
        for route_doc in ROUTE_DOCS:
            assert f"`{skill_name}`" in route_doc.read_text(encoding="utf-8"), (
                f"{skill_name} is not discoverable from {route_doc}"
            )
