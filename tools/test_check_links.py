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
    # 末尾反斜杠是本库实际存在的坏链形态,规范化后仍应无法解析(报 broken)
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
