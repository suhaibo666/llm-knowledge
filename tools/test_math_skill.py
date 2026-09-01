"""Repository integration tests for the Obsidian math writing skill.

The skill used to be duplicated under `.agents/skills/` and `.claude/skills/`, with a test
whose only job was to keep the two byte-identical. There is now a single shared copy in
`skills/`, with `.claude/skills` symlinked to it for Claude Code's native discovery.
(No `.codex/skills`: Codex has no project-level skill discovery, so it would be inert.)
What is tested here is that exactly one PHYSICAL copy exists, and that the skill still
states the rules the checker enforces.
"""

import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = REPO_ROOT / "skills"
MATH_SKILL = SKILLS_ROOT / "writing-obsidian-math"


def test_skills_live_in_exactly_one_shared_place():
    """Agent directories may point at skills/, but must never hold a second copy."""

    assert MATH_SKILL.is_dir()
    assert not (REPO_ROOT / ".agents/skills").exists(), "the .agents duplicate came back"

    # every agent-visible path must resolve to the one shared tree
    for agent_path in (REPO_ROOT / ".claude/skills", REPO_ROOT / ".codex/skills"):
        if not agent_path.exists():
            continue
        assert agent_path.resolve() == SKILLS_ROOT.resolve(), (
            "%s is a real directory, not a pointer to skills/" % agent_path
        )

    # and there is exactly one physical SKILL.md per skill in the whole repo
    seen = {}
    for path in REPO_ROOT.rglob("SKILL.md"):
        # .worktrees/ 下是 git worktree —— 同一仓库的另一份合法检出，
        # 不是"技能被复制了一份"。排除它，否则任何人开个 worktree 都会让本用例变红。
        if {".cache", ".git", ".worktrees", "worktrees"} & set(path.parts):
            continue
        seen.setdefault(os.path.realpath(path), []).append(path)
    names = [p.parent.name for real, ps in seen.items() for p in ps[:1]]
    assert len(names) == len(set(names)), "a skill has more than one physical copy: %s" % names


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
