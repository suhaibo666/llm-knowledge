from pathlib import Path, PurePosixPath

from tools.mkdocs_site.models import BuildPaths, Inventory, PageRecord, RouteRecord


def test_page_record_separates_nav_and_document_titles(tmp_path: Path) -> None:
    page = PageRecord(
        source=tmp_path / "13_megatron_cp_analysis.md",
        relative=PurePosixPath("engine/megatron/13_megatron_cp_analysis.md"),
        title="Megatron-LM 上下文并行深度解析",
        nav_title="13_megatron_cp_analysis",
        is_index=False,
        headings=("0. 总览",),
    )
    assert page.title == "Megatron-LM 上下文并行深度解析"
    assert page.nav_title == "13_megatron_cp_analysis"


def test_build_paths_keep_destructive_outputs_inside_cache(tmp_path: Path) -> None:
    paths = BuildPaths.from_repo(tmp_path)
    assert paths.staging.is_relative_to(paths.cache)
    assert paths.generated_config.is_relative_to(paths.cache)
    assert paths.site == tmp_path / "site"


def test_inventory_indexes_pages_by_relative_path_and_stem(tmp_path: Path) -> None:
    page = PageRecord(
        source=tmp_path / "a.md",
        relative=PurePosixPath("domain/a.md"),
        title="A",
        nav_title="a",
        is_index=False,
        headings=(),
    )
    inventory = Inventory.from_pages((page,))
    assert inventory.by_relative[PurePosixPath("domain/a")] is page
    assert inventory.by_stem["a"] == (page,)
