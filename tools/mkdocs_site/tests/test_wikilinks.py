from pathlib import Path, PurePosixPath

import pytest

from tools.mkdocs_site.inventory import scan_inventory
from tools.mkdocs_site.models import Inventory, PageRecord
from tools.mkdocs_site.wikilinks import LinkResolutionError, rewrite_wikilinks


@pytest.fixture
def resolver_fixture(tmp_path: Path) -> tuple[PageRecord, Inventory]:
    (tmp_path / "section").mkdir()
    (tmp_path / "section" / "current.md").write_text("# Current\n", encoding="utf-8")
    (tmp_path / "target.md").write_text(
        "# Target\n\n## 二、机制\n", encoding="utf-8"
    )
    (tmp_path / "section" / "index.md").write_text("# Section\n", encoding="utf-8")
    inventory = scan_inventory(tmp_path)
    return inventory.by_relative[PurePosixPath("section/current")], inventory


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("[[target]]", "[target](../target.md)"),
        ("[[target|中文标签]]", "[中文标签](../target.md)"),
        (r"[[target\|表格标签]]", "[表格标签](../target.md)"),
        ("[[target#二、机制]]", "[target](../target.md#二机制)"),
        ("[[./index|本目录]]", "[本目录](index.md)"),
    ],
)
def test_rewrite_supported_wikilinks(
    source: str, expected: str, resolver_fixture: tuple[PageRecord, Inventory]
) -> None:
    page, inventory = resolver_fixture
    assert rewrite_wikilinks(source, page, inventory) == expected


def test_rewriter_skips_fenced_and_inline_code(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = "`[[target]]`\n```text\n[[target]]\n```\n[[target]]"
    assert rewrite_wikilinks(markdown, page, inventory).endswith(
        "[target](../target.md)"
    )
    assert rewrite_wikilinks(markdown, page, inventory).count("[[target]]") == 2


def test_rewriter_rejects_embed_and_block_reference(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    with pytest.raises(LinkResolutionError, match="embed"):
        rewrite_wikilinks("![[target]]", page, inventory)
    with pytest.raises(LinkResolutionError, match="block reference"):
        rewrite_wikilinks("[[target^block]]", page, inventory)


def test_resolution_error_reports_location_raw_link_reason_and_candidates(
    tmp_path: Path,
) -> None:
    (tmp_path / "section").mkdir()
    (tmp_path / "section" / "current.md").write_text("# Current\n", encoding="utf-8")
    (tmp_path / "one" / "duplicate.md").parent.mkdir()
    (tmp_path / "one" / "duplicate.md").write_text("# One\n", encoding="utf-8")
    (tmp_path / "two" / "duplicate.md").parent.mkdir()
    (tmp_path / "two" / "duplicate.md").write_text("# Two\n", encoding="utf-8")
    inventory = scan_inventory(tmp_path)
    page = inventory.by_relative[PurePosixPath("section/current")]

    with pytest.raises(LinkResolutionError) as caught:
        rewrite_wikilinks("first line\n[[duplicate]]", page, inventory)

    error = caught.value
    assert (error.source, error.line, error.raw_link, error.reason) == (
        PurePosixPath("section/current.md"),
        2,
        "[[duplicate]]",
        "ambiguous target",
    )
    assert error.candidates == ("one/duplicate.md", "two/duplicate.md")


def test_rewriter_rejects_missing_anchor_and_preserves_escaped_literal(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    assert rewrite_wikilinks(r"\[[target]]", page, inventory) == r"\[[target]]"
    with pytest.raises(LinkResolutionError, match="missing target anchor"):
        rewrite_wikilinks("[[target#不存在]]", page, inventory)


def test_rewriter_rejects_trailing_backslash_like_source_checker(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    with pytest.raises(LinkResolutionError, match="broken target"):
        rewrite_wikilinks("[[target\\]]", page, inventory)
