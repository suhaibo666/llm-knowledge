"""Repository integration tests for the Obsidian math writing skill.

The skill used to be duplicated under `.agents/skills/` and `.claude/skills/`, with a test
whose only job was to keep the two byte-identical. There is now a single shared copy in
`skills/`, so that sync test is gone; what remains is a check that the skill still states
the rules the checker enforces.
"""

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = REPO_ROOT / "skills"
MATH_SKILL = SKILLS_ROOT / "writing-obsidian-math"


def test_skills_live_in_exactly_one_shared_place():
    assert MATH_SKILL.is_dir()
    # the per-agent duplicates must not come back
    assert not (REPO_ROOT / ".agents/skills").exists()
    assert not (REPO_ROOT / ".claude/skills").exists()


def test_skill_contains_canonical_syntax_and_mandatory_check():
    text = (MATH_SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "name: writing-obsidian-math" in text
    assert "Use when writing" in text
    assert "python tools/check_math.py --strict" in text
    assert "python tools/check_math.py --changed --strict" in text
    assert "\\mid" in text
    assert "\\begin{aligned}" in text


def test_skill_documents_every_checker_code():
    """The rule table must cover each code check_math can emit."""

    text = (MATH_SKILL / "SKILL.md").read_text(encoding="utf-8")
    checker = (REPO_ROOT / "tools/check_math.py").read_text(encoding="utf-8")
    emitted = {
        line.split('"')[1]
        for line in checker.splitlines()
        if line.strip().startswith('"MATH') and line.strip().endswith('",')
    }
    assert emitted, "no diagnostic codes found in check_math.py"
    missing = sorted(code for code in emitted if code not in text)
    assert not missing, "skill does not document: %s" % missing


def test_skill_warns_about_the_known_false_positive_traps():
    text = (MATH_SKILL / "SKILL.md").read_text(encoding="utf-8")
    for phrase in ("List\\[str\\]", "row spacing", "money", "index"):
        assert phrase in text, "missing guidance about %r" % phrase


def test_skill_has_three_representative_evals():
    payload = json.loads((MATH_SKILL / "evals/evals.json").read_text(encoding="utf-8"))
    assert payload["skill_name"] == "writing-obsidian-math"
    assert len(payload["evals"]) == 3
    assert {item["id"] for item in payload["evals"]} == {1, 2, 3}
    assert all(item["prompt"] and item["expected_output"] for item in payload["evals"])
