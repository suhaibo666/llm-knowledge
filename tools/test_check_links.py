"""check_links.py 的行为测试。用临时目录构造微型 wiki。"""
from pathlib import Path

from check_links import scan, target_of


def make(tmp_path: Path, files: dict) -> Path:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def test_target_of_variants():
    assert target_of("alpha|显示名") == "alpha"
    assert target_of("alpha#章节") == "alpha"
    assert target_of("dir/alpha.md") == "dir/alpha"
    # Obsidian 表格内的转义别名 [[a\|label]] 等价于 [[a|label]]
    assert target_of("alpha\\|标签") == "alpha"
    # 真正的孤立末尾反斜杠仍是畸形链接(解析后含 /,会报 broken)
    assert "/" in target_of("alpha\\")


def test_broken_and_ok(tmp_path):
    wiki = make(tmp_path, {
        "alpha.md": "see [[bravo]] and [[missing_page]]",
        "bravo.md": "back [[alpha]]",
        "index.md": "[[alpha]] [[bravo]]",
    })
    r, n = scan(wiki)
    assert n == 3
    assert len(r["broken"]) == 1 and "missing_page" in r["broken"][0]


def test_alias_anchor_and_path(tmp_path):
    wiki = make(tmp_path, {
        "sub/alpha.md": "[[bravo|B]] [[bravo#x]] [[../charlie]]",
        "sub/bravo.md": "",
        "charlie.md": "",
        "index.md": "[[sub/alpha]] [[sub/bravo]] [[charlie]]",
    })
    r, _ = scan(wiki)
    assert r["broken"] == []


def test_code_blocks_ignored(tmp_path):
    wiki = make(tmp_path, {
        "alpha.md": "```\n[[ghost]]\n```\n还有 `[[ghost2]]` 行内\n[[bravo]]",
        "bravo.md": "",
        "index.md": "[[alpha]] [[bravo]]",
    })
    r, _ = scan(wiki)
    assert r["broken"] == []


def test_bare_index_and_ambiguous(tmp_path):
    wiki = make(tmp_path, {
        "x/index.md": "[[index]]",       # 裸 index:既 bare 又 ambiguous
        "y/index.md": "",
        "index.md": "[[x/index]] [[y/index]]",   # 路径限定:合法
    })
    r, _ = scan(wiki)
    assert len(r["bare_index"]) == 1
    assert len(r["ambiguous"]) == 1
    assert r["broken"] == []


def test_orphan(tmp_path):
    wiki = make(tmp_path, {
        "alpha.md": "[[bravo]]",
        "bravo.md": "",
        "charlie.md": "",                 # 无入链且不在 index → 孤儿
        "index.md": "[[alpha]]",
    })
    r, _ = scan(wiki)
    assert r["orphans"] == ["charlie.md"]


def test_main_exit_codes(tmp_path, monkeypatch):
    import check_links
    bad = make(tmp_path / "w1", {"alpha.md": "[[missing]]", "index.md": "[[alpha]]"})
    monkeypatch.setattr("sys.argv", ["check_links.py", "--wiki", str(bad), "--strict"])
    assert check_links.main() == 1
    good = make(tmp_path / "w2", {"alpha.md": "", "index.md": "[[alpha]]"})
    monkeypatch.setattr("sys.argv", ["check_links.py", "--wiki", str(good), "--strict"])
    assert check_links.main() == 0
    monkeypatch.setattr("sys.argv", ["check_links.py", "--wiki", str(tmp_path / "nope"), "--strict"])
    assert check_links.main() == 2


def test_orphan_rescued_by_index(tmp_path):
    wiki = make(tmp_path, {
        "alpha.md": "",
        "charlie_page.md": "",
        "index.md": "- charlie_page 相关内容(提及但未链接)",
    })
    r, _ = scan(wiki)
    assert "charlie_page.md" not in r["orphans"]
    assert "alpha.md" in r["orphans"]


# ── stale_section:[[页面]] 紧跟 §N 时校验目标页确有该顶层节 ──────────────────

def test_stale_section_flags_out_of_range(tmp_path):
    wiki = make(tmp_path, {
        "alpha.md": "见 [[bravo]] §7 的说明",
        "bravo.md": "## 1. 一\n## 2. 二\n",
        "index.md": "[[alpha]] [[bravo]]",
    })
    r, _ = scan(wiki)
    assert len(r["stale_section"]) == 1
    assert "§7" in r["stale_section"][0]


def test_stale_section_accepts_existing(tmp_path):
    wiki = make(tmp_path, {
        "alpha.md": "见 [[bravo]] §2 与 [[bravo]] §1.3",
        "bravo.md": "## 1. 一\n## 2. 二\n",
        "index.md": "[[alpha]] [[bravo]]",
    })
    r, _ = scan(wiki)
    assert r["stale_section"] == []


def test_stale_section_handles_chinese_numerals(tmp_path):
    """本域两种编号风格并存:## 一、 与 ## 1. —— 中文序号也要能校验。"""
    wiki = make(tmp_path, {
        "alpha.md": "见 [[bravo]] §七",
        "bravo.md": "## 一、甲\n## 二、乙\n",
        "index.md": "[[alpha]] [[bravo]]",
    })
    r, _ = scan(wiki)
    assert len(r["stale_section"]) == 1
    wiki2 = make(tmp_path / "ok", {
        "alpha.md": "见 [[bravo]] §二",
        "bravo.md": "## 一、甲\n## 二、乙\n",
        "index.md": "[[alpha]] [[bravo]]",
    })
    assert scan(wiki2)[0]["stale_section"] == []


def test_stale_section_only_when_adjacent(tmp_path):
    """§ 必须紧跟在链接后。隔着一句话的 §N 往往指本页自己,不该误报。"""
    wiki = make(tmp_path, {
        "alpha.md": "[[bravo]] —— 某机制是本页 §9 模式的工程应用",
        "bravo.md": "## 1. 一\n",
        "index.md": "[[alpha]] [[bravo]]",
    })
    r, _ = scan(wiki)
    assert r["stale_section"] == []


def test_stale_section_does_not_span_lines(tmp_path):
    """窗口止于行尾:下一行的 §N 不属于上一行的链接。"""
    wiki = make(tmp_path, {
        "alpha.md": "见 [[bravo]]\n\n本页 §9 另有说明\n",
        "bravo.md": "## 1. 一\n",
        "index.md": "[[alpha]] [[bravo]]",
    })
    r, _ = scan(wiki)
    assert r["stale_section"] == []


def test_stale_section_skips_unnumbered_target(tmp_path):
    """目标页没有编号小节时不评判,避免对 index 之类页面误报。"""
    wiki = make(tmp_path, {
        "alpha.md": "见 [[bravo]] §3",
        "bravo.md": "## Overview\n## Related Pages\n",
        "index.md": "[[alpha]] [[bravo]]",
    })
    r, _ = scan(wiki)
    assert r["stale_section"] == []


def test_stale_section_counts_toward_strict(tmp_path, monkeypatch):
    import check_links
    wiki = make(tmp_path, {
        "alpha.md": "见 [[bravo]] §9",
        "bravo.md": "## 1. 一\n",
        "index.md": "[[alpha]] [[bravo]]",
    })
    monkeypatch.setattr("sys.argv", ["check_links.py", "--wiki", str(wiki), "--strict"])
    assert check_links.main() == 1
