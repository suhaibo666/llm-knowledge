from pathlib import Path

import pytest

from tools.mkdocs_site.inventory import scan_inventory
from tools.mkdocs_site.navigation import NavigationError, build_navigation


def test_navigation_uses_folder_titles_and_leaf_stems(fixture_wiki: Path) -> None:
    nav = build_navigation(scan_inventory(fixture_wiki))

    assert nav == [
        {"LLM Knowledge Wiki": "index.md"},
        {
            "工程实现 — 知识地图": [
                {"工程实现 — 知识地图": "domain/index.md"},
                {"10_article": "domain/10_article.md"},
            ]
        },
    ]


def test_navigation_rejects_directories_without_an_index(tmp_path: Path) -> None:
    missing_index = tmp_path / "missing"
    missing_index.mkdir()
    (tmp_path / "index.md").write_text("# Root\n", encoding="utf-8")
    (missing_index / "page.md").write_text("# Page\n", encoding="utf-8")

    with pytest.raises(NavigationError, match=r"missing.*index\.md"):
        build_navigation(scan_inventory(tmp_path))


def test_navigation_supports_constitutional_indexless_collections(
    tmp_path: Path,
) -> None:
    (tmp_path / "index.md").write_text("# Root\n", encoding="utf-8")
    (tmp_path / "changelog").mkdir()
    (tmp_path / "changelog" / "2026_q2.md").write_text(
        "# Archived changes\n", encoding="utf-8"
    )
    (tmp_path / "courses").mkdir()
    (tmp_path / "courses" / "reading_path.md").write_text(
        "# Reading path\n", encoding="utf-8"
    )

    assert build_navigation(scan_inventory(tmp_path)) == [
        {"LLM Knowledge Wiki": "index.md"},
        {"changelog": [{"2026_q2": "changelog/2026_q2.md"}]},
        {"courses": [{"reading_path": "courses/reading_path.md"}]},
    ]
