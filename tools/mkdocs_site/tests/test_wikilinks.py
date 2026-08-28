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


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"\[[target]]", r"\[[target]]"),
        (r"\\[[target]]", r"\\[target](../target.md)"),
        (r"\\\[[target]]", r"\\\[[target]]"),
        (r"\\\\[[target]]", r"\\\\[target](../target.md)"),
    ],
)
def test_rewriter_uses_consecutive_backslash_parity_for_escapes(
    source: str,
    expected: str,
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture

    assert rewrite_wikilinks(source, page, inventory) == expected


def test_rewriter_preserves_angle_context_while_classifying_wikilink(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture

    assert rewrite_wikilinks("<[[target]]>", page, inventory) == (
        "<[target](../target.md)>"
    )


def test_rewriter_preserves_table_alias_shape_during_classification(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = "| Link |\n|---|\n| [[target\\|表格标签]] |"

    assert rewrite_wikilinks(markdown, page, inventory).endswith(
        "| [表格标签](../target.md) |"
    )


def test_rewriter_generates_collision_free_classification_tokens(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    collision = "LLMKNOWLEDGEWIKILINKSENTINEL00000000END"

    rewritten = rewrite_wikilinks(f"{collision} [[target]]", page, inventory)

    assert rewritten == f"{collision} [target](../target.md)"


def test_rewriter_rejects_wikilink_nested_in_markdown_link_label(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture

    with pytest.raises(
        LinkResolutionError,
        match="unsupported nested Markdown link label",
    ):
        rewrite_wikilinks("[see [[target]]](https://example.com)", page, inventory)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("[[target]](说明)", "[target](../target.md) (说明)"),
        ("[[target]]（说明）", "[target](../target.md)（说明）"),
        (
            "[[target#二、机制]](章节)",
            "[target](../target.md#二机制) (章节)",
        ),
        ("[[target|别名]](说明)", "[别名](../target.md) (说明)"),
    ],
)
def test_rewriter_disambiguates_immediate_wikilink_annotations_in_stage(
    source: str,
    expected: str,
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture

    assert rewrite_wikilinks(source, page, inventory) == expected


def test_rewriter_disambiguates_table_alias_annotation_without_source_change(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = "| Link |\n|---|\n| [[target\\|表格标签]](说明) |"

    assert rewrite_wikilinks(markdown, page, inventory).endswith(
        "| [表格标签](../target.md) (说明) |"
    )


def test_rewriter_skips_fenced_and_inline_code(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = "`[[target]]`\n```text\n[[target]]\n```\n[[target]]"
    assert rewrite_wikilinks(markdown, page, inventory).endswith(
        "[target](../target.md)"
    )
    assert rewrite_wikilinks(markdown, page, inventory).count("[[target]]") == 2


def test_rewriter_neutralizes_visible_local_file_links_but_preserves_code(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    local_link = "[source.py:17](file:///E:\\private\\source.py#L17)"
    markdown = (
        f"{local_link}\n"
        f"`{local_link}`\n"
        "```markdown\n"
        f"{local_link}\n"
        "```\n"
        f"    {local_link}\n"
    )

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert rewritten.startswith("source.py:17 *(local source)*\n")
    assert rewritten.count("file:///") == 3


def test_rewriter_neutralizes_backtick_wrapped_file_url(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture

    rewritten = rewrite_wikilinks(
        "[source](`file:///E:/private/source.py:17`)", page, inventory
    )

    assert rewritten == "source *(local source)*"


def test_real_npu_page_has_no_publishable_local_file_links() -> None:
    repo = Path(__file__).resolve().parents[3]
    wiki = repo / "wiki"
    relative = PurePosixPath(
        "02_engineering/01_pytorch/02_compile_stack/04_inductor/npu/"
        "10_npu_inductor_backend_analysis.md"
    )
    inventory = scan_inventory(wiki)
    page = inventory.by_relative[relative.with_suffix("")]
    markdown = page.source.read_text(encoding="utf-8")
    local_link_count = markdown.casefold().count("](file:")
    assert local_link_count > 0

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert "](file:" not in rewritten.casefold()
    assert rewritten.count("*(local source)*") == local_link_count
    assert "codegen/npu_combined_scheduling.py:17 *(local source)*" in rewritten


def test_rewriter_adds_legacy_unicode_heading_aliases_for_local_and_cross_page_links(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    current = "[same](#二机制)\n\n## 二、机制\n\n## 二、机制"

    rewritten_current = rewrite_wikilinks(current, page, inventory)
    rewritten_target = rewrite_wikilinks(
        page.source.parents[1].joinpath("target.md").read_text(encoding="utf-8"),
        inventory.by_relative[PurePosixPath("target")],
        inventory,
    )
    rewritten_cross_page = rewrite_wikilinks("[[target#二、机制]]", page, inventory)

    assert '<a name="二机制"></a>\n' in rewritten_current
    assert '<a name="二机制_1"></a>\n' in rewritten_current
    assert '<a name="二机制"></a>\n' in rewritten_target
    assert rewritten_cross_page == "[target](../target.md#二机制)"
    assert "## 二、机制" in rewritten_current


def test_rewriter_does_not_duplicate_an_existing_legacy_heading_alias(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = '<a name="二机制"></a>\n\n## 二、机制'

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert rewritten.count('name="二机制"') == 1


def test_rewriter_places_declared_manual_fragment_alias_at_matching_heading(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    persistent = "四ub-便笺--无跨核协作--塞满-ub只做-persistent-规约"
    historical_typo = "41-vllmdisaggregated-prefilldecodle"
    markdown = (
        f"[四](#{persistent})\n"
        f"[vLLM：Disaggregated Prefill/Decode](#{historical_typo})\n\n"
        "## 四、UB 便笺 + 无跨核协作 ⇒ 塞满 UB，只做 persistent 规约\n\n"
        "### 4.1 vLLM：Disaggregated Prefill/Decode\n"
    )

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert f'<a name="{persistent}"></a>' in rewritten
    assert rewritten.index(f'<a name="{persistent}"></a>') < rewritten.index("## 四、")
    assert f'<a name="{historical_typo}"></a>\n### 4.1' in rewritten


def test_rewriter_matches_exact_fragment_to_heading_with_inline_code(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    fragment = "10-2025-2026-进展从-guard_size_oblivious"
    markdown = (
        f"[section](#{fragment})\n\n"
        "## 10. 2025-2026 进展从 `guard_size_oblivious`\n"
    )

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert f'<a name="{fragment}"></a>' in rewritten


def test_rewriter_does_not_guess_ambiguous_manual_fragment_alias(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = (
        "[mechanism](#historical-typo)\n\n"
        "## One mechanism\n\n"
        "## Two mechanism\n"
    )

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert 'name="historical-typo"' not in rewritten


def test_rewriter_does_not_guess_manual_alias_from_heading_suffix(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = "[Design](#invented)\n\n## Architecture Design\n"

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert 'name="invented"' not in rewritten


def test_rewriter_drops_manual_alias_claimed_by_two_headings(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = "[Alpha](#shared)\n[Beta](#shared)\n\n## Alpha\n\n## Beta\n"

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert 'name="shared"' not in rewritten


def test_rewriter_strips_heading_attr_list_without_id_for_exact_label_match(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = (
        '[Alpha](#legacy)\n\n## Alpha {.feature key="value"}\n'
    )

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert '<a name="legacy"></a>' in rewritten


@pytest.mark.parametrize(
    "literal",
    [
        "`{#custom}`",
        "{#custom} ordinary text",
    ],
)
def test_rewriter_does_not_treat_inline_or_nonterminal_text_as_heading_attr_list(
    literal: str,
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = f"[Beta](#custom)\n\n## Alpha {literal}\n\n## Beta\n"

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert rewritten.count('name="custom"') == 1
    assert rewritten.index('name="custom"') < rewritten.index("## Beta")


def test_rewriter_neutralizes_backtick_wrapped_local_code_locator(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture

    rewritten = rewrite_wikilinks(
        "[DispatchTable computation](`OperatorEntry.cpp:353-385`)",
        page,
        inventory,
    )

    assert rewritten == "DispatchTable computation *(local source)*"


@pytest.mark.parametrize(
    "target",
    [
        "https://example.com/source.py:12",
        "mailto:owner@example.com:12",
        "//cdn.example.com/source.py:12",
    ],
)
def test_rewriter_preserves_backtick_wrapped_nonlocal_urls(
    target: str,
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = f"[source](`{target}`)"

    assert rewrite_wikilinks(markdown, page, inventory) == markdown


def test_rewriter_neutralizes_only_visible_local_code_locators(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    locator = "[source](`src/runtime.py:12-14`)"
    markdown = (
        f"{locator}\n"
        f"``{locator}``\n"
        "```markdown\n"
        f"{locator}\n"
        "```\n"
        f"    {locator}\n"
    )

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert rewritten.startswith("source *(local source)*\n")
    assert rewritten.count(locator) == 3


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


@pytest.mark.parametrize("fence_indent", ["", " ", "  ", "   "])
def test_rewriter_ignores_fenced_header_source_baseline_declarations(
    fence_indent: str,
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    fake = "1111111111111111111111111111111111111111"
    declared = "2222222222222222222222222222222222222222"
    markdown = (
        "# Page\n\n"
        f"{fence_indent}````text\n"
        f"> **源码基线**：`NVIDIA/Megatron-LM@{fake}`\n"
        f"{fence_indent}```\n"
        f"{fence_indent}````\n"
        f"> **源码基线**：`NVIDIA/Megatron-LM@{declared}`\n\n"
        "## Body\n\n"
        "[source](Megatron-LM/megatron/core/model_parallel_config.py)"
    )

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert f"https://github.com/NVIDIA/Megatron-LM/blob/{declared}/" in rewritten
    assert f"https://github.com/NVIDIA/Megatron-LM/blob/{fake}/" not in rewritten


@pytest.mark.parametrize("code_indent", ["    ", "\t", " \t", "  \t", "   \t"])
def test_indented_backtick_run_does_not_hide_visible_header_source_baseline(
    code_indent: str,
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    declared = "2222222222222222222222222222222222222222"
    markdown = (
        "# Page\n\n"
        f"{code_indent}````text\n"
        f"> **源码基线**：`NVIDIA/Megatron-LM@{declared}`\n\n"
        "## Body\n\n"
        "[source](Megatron-LM/megatron/core/model_parallel_config.py)"
    )

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert f"https://github.com/NVIDIA/Megatron-LM/blob/{declared}/" in rewritten


def test_rewriter_rejects_fenced_only_header_source_baseline(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = (
        "# Page\n\n"
        "~~~text\n"
        "> **源码基线**：`NVIDIA/Megatron-LM@"
        "1111111111111111111111111111111111111111`\n"
        "~~~   \n\n"
        "## Body\n\n"
        "[source](Megatron-LM/megatron/core/model_parallel_config.py)"
    )

    with pytest.raises(LinkResolutionError, match="source baseline"):
        rewrite_wikilinks(markdown, page, inventory)


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


@pytest.mark.parametrize(
    ("indent", "target"),
    [
        ("    ", "tools/labs_torch_compile/README.md"),
        ("\t", "Megatron-LM/megatron/core/model_parallel_config.py"),
        (" \t", "tools/labs_torch_compile/README.md"),
        ("  \t", "tools/labs_torch_compile/README.md"),
        ("   \t", "tools/labs_torch_compile/README.md"),
    ],
)
def test_rewriter_skips_repository_like_links_at_commonmark_code_indent(
    indent: str,
    target: str,
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = f"{indent}[link]({target})\n"

    assert rewrite_wikilinks(markdown, page, inventory) == markdown


@pytest.mark.parametrize("indent", ["", " ", "  ", "   "])
def test_rewriter_rewrites_repository_links_below_commonmark_code_indent(
    indent: str,
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = f"{indent}[lab](tools/labs_torch_compile/README.md)"

    assert rewrite_wikilinks(markdown, page, inventory) == (
        f"{indent}[lab](https://github.com/suhaibo666/llm-knowledge/blob/main/"
        "tools/labs_torch_compile/README.md)"
    )


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


def test_list_indented_visible_wikilink_is_rewritten_by_rendered_classification(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = "1. item\n\n    [[target]]"

    assert rewrite_wikilinks(markdown, page, inventory) == (
        "1. item\n\n    [target](../target.md)"
    )


def test_blockquoted_fenced_wikilink_remains_literal(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = "> ```markdown\n> [[target]]\n> ```\n\n[[target]]"

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert rewritten.count("[[target]]") == 1
    assert rewritten.endswith("[target](../target.md)")


def test_mixed_tab_space_code_wikilink_remains_literal(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = "  \t[[target]]\n\n[[target]]"

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert rewritten.count("[[target]]") == 1
    assert rewritten.endswith("[target](../target.md)")


def test_html_comment_wikilink_remains_literal(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = "<!--\n[[target]]\n-->\n\n[[target]]"

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert rewritten.count("[[target]]") == 1
    assert rewritten.endswith("[target](../target.md)")


def test_same_line_active_and_code_wikilinks_are_classified_independently(
    resolver_fixture: tuple[PageRecord, Inventory],
) -> None:
    page, inventory = resolver_fixture
    markdown = "[[target]] and `[[target]]`"

    assert rewrite_wikilinks(markdown, page, inventory) == (
        "[target](../target.md) and `[[target]]`"
    )


def test_historical_changelog_callout_example_remains_literal() -> None:
    repo = Path(__file__).resolve().parents[3]
    wiki = repo / "wiki"
    inventory = scan_inventory(wiki)
    page = inventory.by_relative[PurePosixPath("changelog")]
    markdown = page.source.read_text(encoding="utf-8")

    rewritten = rewrite_wikilinks(markdown, page, inventory)

    assert "[[verl_end_to_end_iteration_analysis]]" in rewritten


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
