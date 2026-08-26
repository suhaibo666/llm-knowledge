"""Repository integration tests for the Obsidian math writing skill."""

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_SKILL = REPO_ROOT / ".agents/skills/writing-obsidian-math"
CLAUDE_SKILL = REPO_ROOT / ".claude/skills/writing-obsidian-math"


def test_agent_and_claude_skill_files_are_identical():
    for relative in (Path("SKILL.md"), Path("evals/evals.json")):
        assert (AGENT_SKILL / relative).read_bytes() == (
            CLAUDE_SKILL / relative
        ).read_bytes()


def test_skill_contains_canonical_syntax_and_mandatory_check():
    text = (AGENT_SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "name: writing-obsidian-math" in text
    assert "Use when writing" in text
    assert "python tools/check_math.py --strict" in text
    assert "python tools/check_math.py --changed --strict" in text
    assert "\\mid" in text
    assert "\\begin{aligned}" in text


def test_skill_has_three_representative_evals():
    payload = json.loads(
        (AGENT_SKILL / "evals/evals.json").read_text(encoding="utf-8")
    )
    assert payload["skill_name"] == "writing-obsidian-math"
    assert len(payload["evals"]) == 3
    assert {item["id"] for item in payload["evals"]} == {1, 2, 3}
    assert all(item["prompt"] and item["expected_output"] for item in payload["evals"])
