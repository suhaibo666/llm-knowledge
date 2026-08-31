from pathlib import Path, PurePosixPath

import pytest

from tools.mkdocs_site.inventory import InventoryError, scan_inventory


def make_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable on this platform: {error}")


def test_scan_inventory_uses_filename_only_for_leaf_navigation(
    fixture_wiki: Path,
) -> None:
    inventory = scan_inventory(fixture_wiki)
    leaf = inventory.by_relative[PurePosixPath("domain/10_article")]
    folder = inventory.by_relative[PurePosixPath("domain/index")]
    assert (leaf.title, leaf.nav_title) == ("上下文并行", "10_article")
    assert (folder.title, folder.nav_title) == ("工程实现 — 知识地图", "工程实现 — 知识地图")
    assert leaf.headings == ("0. 总览", "1. 机制")


def test_scan_inventory_uses_fixed_navigation_title_for_root_index(
    fixture_wiki: Path,
) -> None:
    root = scan_inventory(fixture_wiki).by_relative[PurePosixPath("index")]

    assert (root.title, root.nav_title) == (
        "LLM Knowledge Wiki — 知识库总索引",
        "LLM Knowledge Wiki",
    )


def test_scan_inventory_ignores_headings_in_mismatched_or_shorter_fences(
    tmp_path: Path,
) -> None:
    (tmp_path / "fences.md").write_text(
        """---
title: Fences
---
````markdown
```
~~~
## hidden in a four-backtick fence
````
~~~markdown
```
### hidden in a tilde fence
~~~
## visible
""",
        encoding="utf-8",
    )

    page = scan_inventory(tmp_path).by_relative[PurePosixPath("fences")]

    assert page.headings == ("visible",)


def test_scan_inventory_fails_when_a_page_has_no_title(tmp_path: Path) -> None:
    (tmp_path / "untitled.md").write_text("plain text", encoding="utf-8")
    with pytest.raises(InventoryError, match="untitled.md.*title"):
        scan_inventory(tmp_path)


def test_scan_inventory_rejects_markdown_symlink_outside_wiki(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("# External\n\nSECRET=leaked\n", encoding="utf-8")
    make_symlink(wiki / "leak.md", outside)

    with pytest.raises(
        InventoryError,
        match=r"leak\.md.*resolves outside wiki root.*outside\.md",
    ):
        scan_inventory(wiki)


def test_scan_inventory_treats_four_space_backticks_as_literal_text(
    tmp_path: Path,
) -> None:
    (tmp_path / "literal.md").write_text(
        "    ```\n# Valid title\n```\n",
        encoding="utf-8",
    )

    page = scan_inventory(tmp_path).by_relative[PurePosixPath("literal")]

    assert page.title == "Valid title"


def test_scan_inventory_requires_whitespace_after_fence_closer(
    tmp_path: Path,
) -> None:
    (tmp_path / "closer.md").write_text(
        "```text\n# hidden\n```not-a-close\n# still hidden\n```\n# Valid title\n",
        encoding="utf-8",
    )

    page = scan_inventory(tmp_path).by_relative[PurePosixPath("closer")]

    assert page.title == "Valid title"
