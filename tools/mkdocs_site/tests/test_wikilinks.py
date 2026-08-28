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


def test_rewriter_externalizes_repository_root_and_pinned_source_links(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    baseline = "71092579522a12522d9f323ae180c9825d01928a"
    markdown = (
        f"> **源码基线**：`NVIDIA/Megatron-LM@{baseline}`\n\n"
        "[lab](tools/labs_torch_compile/README.md)\n"
        "[source](Megatron-LM/megatron/core/model_parallel_config.py)\n"
        "`[literal](tools/labs_torch_compile/README.md)`"
    )

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert (
        "[lab](https://github.com/suhaibo666/llm-knowledge/blob/main/"
        "tools/labs_torch_compile/README.md)"
    ) in rewritten
    assert (
        f"[source](https://github.com/NVIDIA/Megatron-LM/blob/{baseline}/"
        "megatron/core/model_parallel_config.py)"
    ) in rewritten
    assert "`[literal](tools/labs_torch_compile/README.md)`" in rewritten


def test_rewriter_uses_only_explicit_header_source_baseline(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    historical = "1111111111111111111111111111111111111111"
    declared = "2222222222222222222222222222222222222222"
    markdown = (
        "# Page\n\n"
        f"> Historical pin: `NVIDIA/Megatron-LM@{historical}`.\n"
        f"> **源码基线**：`NVIDIA/Megatron-LM@{declared}`\n\n"
        "## Body\n\n"
        "[source](Megatron-LM/megatron/core/model_parallel_config.py)\n"
        f"Previous baseline: `NVIDIA/Megatron-LM@{historical}`."
    )

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert f"https://github.com/NVIDIA/Megatron-LM/blob/{declared}/" in rewritten
    assert f"https://github.com/NVIDIA/Megatron-LM/blob/{historical}/" not in rewritten


@pytest.mark.parametrize(
    ("header", "reason"),
    [
        (
            "# Page\n\n## History\n\n"
            "NVIDIA/Megatron-LM@1111111111111111111111111111111111111111\n",
            "source baseline",
        ),
        (
            "# Page\n\n"
            "> **源码基线**：`NVIDIA/Megatron-LM@"
            "1111111111111111111111111111111111111111`\n"
            "> **源码基线**：`NVIDIA/Megatron-LM@"
            "2222222222222222222222222222222222222222`\n\n"
            "## Body\n",
            "conflicting",
        ),
    ],
)
def test_rewriter_rejects_missing_or_conflicting_header_source_baseline(
    header: str,
    reason: str,
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture

    with pytest.raises(LinkResolutionError, match=reason):
        rewrite_wikilinks(
            header + "[source](Megatron-LM/megatron/core/model_parallel_config.py)",
            page,
            inventory,
        )


@pytest.mark.parametrize(
    ("target", "content"),
    [
        ("tools/local.md", "# Local page\n"),
        ("Megatron-LM/diagram.svg", "<svg/>"),
    ],
)
def test_rewriter_preserves_wiki_local_repository_like_targets(
    target: str, content: str, tmp_path: Path
) -> None:
    (tmp_path / "section").mkdir()
    (tmp_path / "section/current.md").write_text("# Current\n", encoding="utf-8")
    local = tmp_path / Path(target)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(content, encoding="utf-8")
    inventory = scan_inventory(tmp_path)
    page = inventory.by_relative[PurePosixPath("section/current")]
    markdown = f"[local]({target})"

    assert rewrite_wikilinks(markdown, page, inventory) == markdown


def test_rewriter_skips_repository_like_links_in_indented_code(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = (
        "    [lab](tools/labs_torch_compile/README.md)\n"
        "\t[source](Megatron-LM/megatron/core/model_parallel_config.py)\n"
    )

    assert rewrite_wikilinks(markdown, page, inventory) == markdown


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_fence_run_with_trailing_text_does_not_close_block(
    fence: str, resolver_fixture: tuple[PageRecord, Inventory]
) -> None:
    page, inventory = resolver_fixture
    markdown = (
        f"{fence}text\n"
        "[[target]]\n"
        f"{fence}not-a-close\n"
        "[[target]]\n"
        f"{fence}   \n"
        "[[target]]"
    )

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert rewritten.count("[[target]]") == 2
    assert rewritten.endswith("[target](../target.md)")


def test_multiline_inline_code_span_preserves_contained_wikilink(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = "`code starts\n[[target]]\ncode ends`\n[[target]]"

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert rewritten.count("[[target]]") == 1
    assert rewritten.endswith("[target](../target.md)")


def test_blank_line_prevents_inline_code_pairing_across_paragraphs(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = "`start\n\n[[target]]\nend`"

    assert rewrite_wikilinks(markdown, page, inventory) == (
        "`start\n\n[target](../target.md)\nend`"
    )


def test_atx_heading_prevents_inline_code_pairing_across_blocks(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = "`start\n## Heading\n[[target]]\nend`"

    assert rewrite_wikilinks(markdown, page, inventory) == (
        "`start\n## Heading\n[target](../target.md)\nend`"
    )


@pytest.mark.parametrize(
    ("delimiter", "unequal_run"),
    [("``", "```"), ("```", "``")],
)
def test_inline_code_closes_only_on_equal_backtick_run(
    delimiter: str,
    unequal_run: str,
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = (
        f"prefix {delimiter}code {unequal_run} still code [[target]]\n"
        f"code ends{delimiter}\n"
        "[[target]]"
    )

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert rewritten.count("[[target]]") == 1
    assert rewritten.endswith("[target](../target.md)")


@pytest.mark.parametrize("suffix", ["_1", "_2"])
def test_duplicate_heading_anchor_uses_python_markdown_unique_suffix(
    suffix: str, tmp_path: Path
) -> None:
    (tmp_path / "section").mkdir()
    (tmp_path / "section/current.md").write_text("# Current\n", encoding="utf-8")
    (tmp_path / "target.md").write_text(
        "# Target\n\n## 重复\n\n## 重复\n\n## 重复\n", encoding="utf-8"
    )
    inventory = scan_inventory(tmp_path)
    page = inventory.by_relative[PurePosixPath("section/current")]

    assert rewrite_wikilinks(
        f"[[target#重复{suffix}]]", page, inventory
    ) == f"[target](../target.md#重复{suffix})"
