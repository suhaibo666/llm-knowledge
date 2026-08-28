from pathlib import Path, PurePosixPath

import pytest

from tools.mkdocs_site.inventory import InventoryError, scan_inventory


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
