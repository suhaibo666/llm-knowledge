# MkDocs Material Local Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully validated local MkDocs Material preview from `wiki/` without modifying the Obsidian sources or the current GitHub Pages deployment.

**Architecture:** A Python staging tool inventories `wiki/`, resolves Obsidian Wikilinks with the repository's existing semantics, and writes a disposable `.mkdocs-cache/docs/` tree plus generated MkDocs configuration. MkDocs Material renders that tree; a static-site crawler and Puppeteer smoke suite prove routes, anchors, assets, navigation, search, and responsive layout before the preview is opened.

**Tech Stack:** Python 3.11+, MkDocs 1.6.1, Material for MkDocs 9.7.7, PyYAML, Python-Markdown/Pymdown Extensions, Beautiful Soup 4.15.0, pytest 8.4.2, Node.js 22+, puppeteer-core from `tools/html2md`.

**Spec:** `docs/superpowers/specs/2026-08-28-mkdocs-material-migration-design.md`

## Global Constraints

- Work only in `.worktrees/mkdocs-preview` on branch `codex/mkdocs-preview`; do not touch the dirty primary worktree.
- `wiki/` is the only authoritative content tree. No task may write into `wiki/`.
- `.mkdocs-cache/` and `site/` are disposable, ignored outputs; clean builds recreate them from scratch.
- Ordinary Markdown navigation labels are file stems; `index.md` navigation labels are their Chinese titles.
- The source page title remains the Chinese frontmatter/H1 title even when the navigation label is an English filename.
- Existing Wikilink semantics are authoritative. Broken, ambiguous, unsupported, or missing-anchor internal links fail the build.
- Ordinary output files use `.html` (`use_directory_urls: false`) so existing extensionless Quartz URLs continue to resolve on GitHub Pages; index files remain `index.html`.
- All local and generated URLs must honor the eventual `/llm-knowledge/` project base.
- External HTTP(S) availability is not a blocking build gate.
- Preview work must not modify `.github/workflows/pages.yml` or publish to `main`.
- Retain the existing Quartz toolchain and continue running its tests during parallel preview development.
- Use TDD for every behavioral task and commit after each independently passing task.

---

## Planned File Structure

```text
requirements-docs.txt                         # exact top-level preview/test dependencies
mkdocs.yml                                    # stable human-readable Material base config
.gitignore                                    # disposable MkDocs outputs
package.json                                  # parallel MkDocs scripts; Quartz scripts unchanged
tools/mkdocs_site/__init__.py                 # package marker
tools/mkdocs_site/models.py                   # immutable PageRecord/Inventory/RouteRecord types
tools/mkdocs_site/inventory.py                # source scan, frontmatter/H1/headings/assets
tools/mkdocs_site/navigation.py               # derived nav tree and neighbor ordering
tools/mkdocs_site/wikilinks.py                # parse, resolve, rewrite visible Wikilinks
tools/mkdocs_site/staging.py                  # clean staging tree and generated config
tools/mkdocs_site/routes.py                    # source/output/public URL manifest
tools/mkdocs_site/config.py                    # merge base config with generated nav/atlas
tools/mkdocs_site/search_index.py              # Chinese result titles and filename terms
tools/mkdocs_site/validate_site.py             # static HTML/anchor/asset/legacy-route crawler
tools/mkdocs_site/watch.py                     # deterministic source snapshot polling
tools/mkdocs_site/cli.py                       # stage/build/serve orchestration
tools/mkdocs_site/tests/                       # focused pytest suite and fixtures
tools/mkdocs-site/overrides/main.html          # filename metadata and theme shell hooks
tools/mkdocs-site/overrides/home.html          # Source Atlas homepage
tools/mkdocs-site/assets/extra.css             # responsive Source Atlas visual system
tools/mkdocs-site/assets/extra.js              # nav tooltip/current-path/accessibility behavior
tools/mkdocs-site/assets/mathjax.js            # local MathJax configuration
tools/mkdocs-site/assets/diagram.js            # local Mermaid initialization
tools/mkdocs-site/package.json                 # exact browser renderer dependencies
tools/mkdocs-site/package-lock.json            # reproducible renderer dependency lock
tools/mkdocs-site/smoke.mjs                    # real browser smoke test
.mkdocs-cache/                                 # ignored staging/config/runtime data
site/                                          # ignored local static build
```

Each Python file owns one responsibility. Cross-module contracts are frozen in Task 1 and reused verbatim by later tasks.

---

### Task 1: Freeze the Preview Tool Contracts and Dependencies

**Files:**
- Create: `requirements-docs.txt`
- Create: `tools/mkdocs_site/__init__.py`
- Create: `tools/mkdocs_site/models.py`
- Create: `tools/mkdocs_site/tests/__init__.py`
- Create: `tools/mkdocs_site/tests/test_models.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `PageRecord`, `RouteRecord`, `Inventory`, and `BuildPaths` dataclasses used by all later tasks.
- Produces: exact dependency pins and dedicated ignored cache paths.

- [ ] **Step 1: Write the failing model contract tests**

```python
# tools/mkdocs_site/tests/test_models.py
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
```

- [ ] **Step 2: Run the contract tests and verify the package does not exist**

Run: `python -m pytest tools/mkdocs_site/tests/test_models.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.mkdocs_site'`.

- [ ] **Step 3: Add exact dependencies and immutable dataclasses**

`requirements-docs.txt`:

```text
mkdocs==1.6.1
mkdocs-material==9.7.7
beautifulsoup4==4.15.0
pytest==8.4.2
```

`models.py` must define these exact public shapes:

```python
@dataclass(frozen=True)
class PageRecord:
    source: Path
    relative: PurePosixPath
    title: str
    nav_title: str
    is_index: bool
    headings: tuple[str, ...]

@dataclass(frozen=True)
class RouteRecord:
    source: PurePosixPath
    output: PurePosixPath
    public_urls: tuple[str, ...]

@dataclass(frozen=True)
class Inventory:
    pages: tuple[PageRecord, ...]
    by_relative: Mapping[PurePosixPath, PageRecord]
    by_stem: Mapping[str, tuple[PageRecord, ...]]

    @classmethod
    def from_pages(cls, pages: tuple[PageRecord, ...]) -> "Inventory":
        ordered = tuple(sorted(pages, key=lambda page: page.relative.as_posix().casefold()))
        by_relative: dict[PurePosixPath, PageRecord] = {}
        by_stem_mutable: dict[str, list[PageRecord]] = defaultdict(list)
        for page in ordered:
            key = page.relative.with_suffix("")
            if key in by_relative:
                raise ValueError(f"duplicate page path: {key}")
            by_relative[key] = page
            by_stem_mutable[page.source.stem].append(page)
        by_stem = {key: tuple(value) for key, value in by_stem_mutable.items()}
        return cls(ordered, MappingProxyType(by_relative), MappingProxyType(by_stem))

@dataclass(frozen=True)
class BuildPaths:
    repo: Path
    wiki: Path
    cache: Path
    staging: Path
    generated_config: Path
    site: Path

    @classmethod
    def from_repo(cls, repo: Path) -> "BuildPaths":
        resolved = repo.resolve()
        cache = resolved / ".mkdocs-cache"
        staging = cache / "docs"
        generated = cache / "mkdocs.generated.yml"
        site = resolved / "site"
        staging.resolve().relative_to(cache.resolve())
        site.resolve().relative_to(resolved)
        return cls(resolved, resolved / "wiki", cache, staging, generated, site)
```

The implementation imports `defaultdict` from `collections` and `MappingProxyType` from `types`; tests assert the returned indexes cannot be mutated.

`BuildPaths.from_repo()` resolves all paths and raises `ValueError` unless both destructive targets are strict descendants of the expected repository/cache roots.

- [ ] **Step 4: Ignore only the dedicated generated paths**

Append:

```gitignore
# MkDocs Material local preview outputs
.mkdocs-cache/
/site/
```

- [ ] **Step 5: Run the focused tests**

Run: `python -m pytest tools/mkdocs_site/tests/test_models.py -q`

Expected: `3 passed`.

- [ ] **Step 6: Commit the contracts**

```bash
git add .gitignore requirements-docs.txt tools/mkdocs_site
git commit -m "feat(mkdocs): define preview build contracts"
```

---

### Task 2: Inventory Pages, Titles, Headings, Assets, and Routes

**Files:**
- Create: `tools/mkdocs_site/inventory.py`
- Create: `tools/mkdocs_site/routes.py`
- Create: `tools/mkdocs_site/tests/test_inventory.py`
- Create: `tools/mkdocs_site/tests/test_routes.py`
- Create: `tools/mkdocs_site/tests/conftest.py`
- Create: `tools/mkdocs_site/tests/fixtures/wiki/index.md`
- Create: `tools/mkdocs_site/tests/fixtures/wiki/domain/index.md`
- Create: `tools/mkdocs_site/tests/fixtures/wiki/domain/10_article.md`
- Create: `tools/mkdocs_site/tests/fixtures/wiki/domain/figure.svg`

**Interfaces:**
- Consumes: dataclasses from Task 1.
- Produces: `scan_inventory(wiki: Path) -> Inventory`.
- Produces: `build_route_manifest(inventory: Inventory) -> tuple[RouteRecord, ...]`.
- Produces: `write_route_manifest(records, destination: Path) -> None`.

- [ ] **Step 1: Add fixture pages and failing inventory tests**

The root fixture has frontmatter title `LLM Knowledge Wiki — 知识库总索引`. The domain index has title `工程实现 — 知识地图`. The ordinary page has title `上下文并行` and headings `0. 总览`, `1. 机制`.

`conftest.py` defines `fixture_wiki(tmp_path)` by copying `tools/mkdocs_site/tests/fixtures/wiki` into the test's temporary directory and returning that path. Tests never mutate the repository fixture in place.

```python
def test_scan_inventory_uses_filename_only_for_leaf_navigation(fixture_wiki: Path) -> None:
    inventory = scan_inventory(fixture_wiki)
    leaf = inventory.by_relative[PurePosixPath("domain/10_article")]
    folder = inventory.by_relative[PurePosixPath("domain/index")]
    assert (leaf.title, leaf.nav_title) == ("上下文并行", "10_article")
    assert (folder.title, folder.nav_title) == ("工程实现 — 知识地图", "工程实现 — 知识地图")
    assert leaf.headings == ("0. 总览", "1. 机制")


def test_scan_inventory_fails_when_a_page_has_no_title(tmp_path: Path) -> None:
    (tmp_path / "untitled.md").write_text("plain text", encoding="utf-8")
    with pytest.raises(InventoryError, match="untitled.md.*title"):
        scan_inventory(tmp_path)
```

- [ ] **Step 2: Run inventory tests and verify failure**

Run: `python -m pytest tools/mkdocs_site/tests/test_inventory.py -q`

Expected: FAIL because `scan_inventory` and `InventoryError` are undefined.

- [ ] **Step 3: Implement deterministic inventory scanning**

`inventory.py` must:

- read YAML frontmatter with `yaml.safe_load` when present;
- fall back to the first H1 only when frontmatter has no title;
- collect H2/H3 heading text outside fenced code;
- assign `nav_title=source.stem` for leaves and `nav_title=title` for `index.md`;
- sort paths by case-folded POSIX relative path;
- fail on duplicate relative paths, unreadable UTF-8, or missing titles;
- never write to the source tree.

- [ ] **Step 4: Add failing route tests**

```python
def test_route_manifest_matches_quartz_file_shape(fixture_wiki: Path) -> None:
    records = build_route_manifest(scan_inventory(fixture_wiki))
    by_source = {record.source.as_posix(): record for record in records}
    assert by_source["domain/10_article.md"].output.as_posix() == "domain/10_article.html"
    assert by_source["domain/10_article.md"].public_urls == (
        "/domain/10_article",
        "/domain/10_article.html",
    )
    assert by_source["domain/index.md"].output.as_posix() == "domain/index.html"
    assert "/domain/" in by_source["domain/index.md"].public_urls
```

- [ ] **Step 5: Implement route manifest generation**

`routes.py` writes stable, UTF-8 JSON sorted by source path. Root `index.md` maps to `/`; nested index pages map to both `/path/` and `/path/index.html`; leaves map to extensionless and `.html` URLs. Reject collisions between output paths or public URLs.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tools/mkdocs_site/tests/test_inventory.py tools/mkdocs_site/tests/test_routes.py -q`

Expected: all tests PASS.

- [ ] **Step 7: Verify the real Wiki baseline without writing output**

Run:

```powershell
python -c "from pathlib import Path; from tools.mkdocs_site.inventory import scan_inventory; print(len(scan_inventory(Path('wiki')).pages))"
```

Expected on design baseline: `426`.

- [ ] **Step 8: Commit inventory and routing**

```bash
git add tools/mkdocs_site
git commit -m "feat(mkdocs): inventory wiki pages and legacy routes"
```

---

### Task 3: Resolve and Rewrite Obsidian Wikilinks in a Disposable Stage

**Files:**
- Create: `tools/mkdocs_site/wikilinks.py`
- Create: `tools/mkdocs_site/staging.py`
- Create: `tools/mkdocs_site/tests/test_wikilinks.py`
- Create: `tools/mkdocs_site/tests/test_staging.py`

**Interfaces:**
- Consumes: `Inventory` and `PageRecord` from Tasks 1–2.
- Produces: `rewrite_wikilinks(markdown: str, page: PageRecord, inventory: Inventory) -> str`.
- Produces: `stage_wiki(paths: BuildPaths, inventory: Inventory) -> StageResult`.
- Produces: `LinkResolutionError` containing source-relative path, 1-based line, raw link, and reason.

`test_wikilinks.py` defines `resolver_fixture` with `section/current.md`, root `target.md`, and `section/index.md`. `test_staging.py` defines a local `tree_digest(root)` helper that hashes sorted relative paths and bytes; neither helper is production code.

- [ ] **Step 1: Write failing link-resolution tests**

```python
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
def test_rewrite_supported_wikilinks(source, expected, resolver_fixture) -> None:
    page, inventory = resolver_fixture
    assert rewrite_wikilinks(source, page, inventory) == expected


def test_rewriter_skips_fenced_and_inline_code(resolver_fixture) -> None:
    page, inventory = resolver_fixture
    markdown = "`[[target]]`\n```text\n[[target]]\n```\n[[target]]"
    assert rewrite_wikilinks(markdown, page, inventory).endswith("[target](../target.md)")
    assert rewrite_wikilinks(markdown, page, inventory).count("[[target]]") == 2


def test_rewriter_rejects_embed_and_block_reference(resolver_fixture) -> None:
    page, inventory = resolver_fixture
    with pytest.raises(LinkResolutionError, match="embed"):
        rewrite_wikilinks("![[target]]", page, inventory)
    with pytest.raises(LinkResolutionError, match="block reference"):
        rewrite_wikilinks("[[target^block]]", page, inventory)
```

- [ ] **Step 2: Run the resolver tests and verify failure**

Run: `python -m pytest tools/mkdocs_site/tests/test_wikilinks.py -q`

Expected: FAIL because the resolver is undefined.

- [ ] **Step 3: Implement source-faithful resolution**

Resolution order must match `tools/check_links.py`:

1. normalize `\|`, optional `.md`, Windows separators, alias and anchor;
2. for path targets, try Wiki-root relative, then source-directory relative, then a unique path suffix;
3. for bare names, require exactly one stem match;
4. report zero matches as broken and multiple matches as ambiguous;
5. use the same Unicode slugifier configured for MkDocs headings;
6. produce POSIX relative Markdown paths from the staged source page to the staged target page.

Implement a line scanner that preserves fenced blocks and splits inline-code spans before applying the Wikilink matcher. Error messages include `page.relative`, line number, raw token, and candidate list.

- [ ] **Step 4: Add failing staging tests**

```python
def test_stage_wiki_is_clean_reproducible_and_source_read_only(tmp_path: Path, fixture_wiki: Path) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(fixture_wiki, repo / "wiki")
    paths = BuildPaths.from_repo(repo)
    before = tree_digest(paths.wiki)
    inventory = scan_inventory(paths.wiki)
    first = stage_wiki(paths, inventory)
    stale = paths.staging / "stale.md"
    stale.write_text("stale", encoding="utf-8")
    second = stage_wiki(paths, inventory)
    assert not stale.exists()
    assert tree_digest(paths.wiki) == before
    assert first.page_count == second.page_count == 3
    assert (paths.staging / "domain/figure.svg").is_file()
```

- [ ] **Step 5: Implement safe staging**

`stage_wiki()` verifies `paths.staging.resolve()` is inside `paths.cache.resolve()` before any replacement. It builds a sibling temporary tree inside `paths.cache`, copies non-Markdown resources, rewrites Markdown links, preserves user frontmatter, and adds generated keys under a single `mkdocs_preview` mapping:

```yaml
mkdocs_preview:
  source_path: domain/10_article.md
  nav_title: 10_article
  is_index: false
```

Only after the whole temporary tree and route manifest succeed does it replace the previous staging tree, so a conversion error leaves the last working preview intact. It writes `routes.json` beside the generated config and returns:

```python
@dataclass(frozen=True)
class StageResult:
    page_count: int
    asset_count: int
    route_manifest: Path
```

- [ ] **Step 6: Run source and staging tests**

Run: `python -m pytest tools/mkdocs_site/tests/test_wikilinks.py tools/mkdocs_site/tests/test_staging.py -q`

Expected: all tests PASS.

- [ ] **Step 7: Cross-check the real source checker**

Run: `python tools/check_links.py --strict`

Expected: `broken=0`, `ambiguous=0`, `bare_index=0`.

- [ ] **Step 8: Commit Wikilink staging**

```bash
git add tools/mkdocs_site
git commit -m "feat(mkdocs): stage Obsidian wiki without source rewrites"
```

---

### Task 4: Generate Navigation and Build a Minimal Material Site

**Files:**
- Create: `mkdocs.yml`
- Create: `tools/mkdocs_site/navigation.py`
- Create: `tools/mkdocs_site/config.py`
- Create: `tools/mkdocs_site/cli.py`
- Create: `tools/mkdocs_site/tests/test_navigation.py`
- Create: `tools/mkdocs_site/tests/test_config.py`
- Create: `tools/mkdocs_site/tests/test_cli.py`
- Modify: `package.json`

**Interfaces:**
- Consumes: inventory/staging contracts.
- Produces: `build_navigation(inventory: Inventory) -> list[dict[str, object] | str]`.
- Produces: `write_generated_config(paths, inventory, stage_result) -> Path`.
- Produces: CLI commands `stage`, `build`, `serve` via `python -m tools.mkdocs_site.cli`.

- [ ] **Step 1: Write failing navigation tests**

```python
def test_navigation_uses_folder_titles_and_leaf_stems(fixture_wiki: Path) -> None:
    nav = build_navigation(scan_inventory(fixture_wiki))
    assert nav == [
        {"LLM Knowledge Wiki — 知识库总索引": "index.md"},
        {
            "工程实现 — 知识地图": [
                {"工程实现 — 知识地图": "domain/index.md"},
                {"10_article": "domain/10_article.md"},
            ]
        },
    ]
```

- [ ] **Step 2: Run navigation tests and verify failure**

Run: `python -m pytest tools/mkdocs_site/tests/test_navigation.py -q`

Expected: FAIL because `build_navigation` is undefined.

- [ ] **Step 3: Implement derived navigation**

Build a directory trie from `PageRecord.relative`. Every directory with `index.md` uses that page's title as the section label and lists the index first. Directories without an index are rejected because the repository constitution requires local index ownership. Remaining children sort case-insensitively by filename. The result is generated only in `.mkdocs-cache/mkdocs.generated.yml`; no deep nav is committed.

- [ ] **Step 4: Write the base Material configuration**

`mkdocs.yml` must include:

```yaml
site_name: LLM Knowledge Wiki
site_url: https://suhaibo666.github.io/llm-knowledge/
docs_dir: .mkdocs-cache/docs
site_dir: site
use_directory_urls: false
theme:
  name: material
  language: zh
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.tabs
    - navigation.sections
    - navigation.indexes
    - navigation.prune
    - navigation.top
    - toc.follow
    - search.suggest
    - search.highlight
    - search.share
    - content.code.copy
    - content.action.view
plugins:
  - search:
      lang:
        - zh
        - en
markdown_extensions:
  - admonition
  - attr_list
  - md_in_html
  - pymdownx.details
    - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
  - pymdownx.arithmatex:
      generic: true
  - toc:
      permalink: true
      toc_depth: 3
repo_url: https://github.com/suhaibo666/llm-knowledge
edit_uri: blob/main/wiki/
```

- [ ] **Step 5: Write failing generated-config and CLI tests**

Tests assert that generated config inherits the base values, injects derived `nav`, targets only the cache staging directory, and that `cli build` calls these operations in order: inventory, stage, config, MkDocs subprocess. Dependencies are injected into `main(argv, operations=None)` so tests never launch a real server.

- [ ] **Step 6: Implement config generation and CLI build**

`config.py` uses `yaml.safe_load/safe_dump(allow_unicode=True, sort_keys=False)`, verifies that the base file keeps the declared `docs_dir`, `site_dir`, `site_url`, and `use_directory_urls`, then writes absolute `docs_dir` and `site_dir` values from `BuildPaths` into the generated config. Absolute generated paths are required because `.mkdocs-cache/mkdocs.generated.yml` has a different parent than `mkdocs.yml`. `cli.py` resolves the repository from its own file path, invokes `sys.executable -m mkdocs build --strict -f <generated-config>`, and returns the subprocess exit code.

- [ ] **Step 7: Add parallel npm scripts without changing Quartz scripts**

```json
"docs:mkdocs": "python -m tools.mkdocs_site.cli serve",
"docs:mkdocs:build": "python -m tools.mkdocs_site.cli build",
"docs:mkdocs:test:unit": "python -m pytest tools/mkdocs_site/tests -q",
"docs:mkdocs:test": "npm run docs:mkdocs:test:unit && npm run docs:mkdocs:build && node tools/mkdocs-site/smoke.mjs"
```

- [ ] **Step 8: Install pinned dependencies and run unit tests**

Run:

```powershell
python -m pip install -r requirements-docs.txt
python -m pytest tools/mkdocs_site/tests -q
```

Expected: all tests PASS.

- [ ] **Step 9: Run the first full strict build**

Run: `npm run docs:mkdocs:build`

Expected: exit 0, exactly the inventory page count is emitted, and `site/index.html` exists. Any corpus conversion failure is fixed in the resolver with a regression test before retrying.

- [ ] **Step 10: Commit the minimal Material build**

```bash
git add mkdocs.yml package.json tools/mkdocs_site
git commit -m "feat(mkdocs): generate navigation and strict Material build"
```

---

### Task 5: Build the Source Atlas Theme and Preserve Search Titles

**Files:**
- Modify: `mkdocs.yml`
- Create: `tools/mkdocs-site/overrides/main.html`
- Create: `tools/mkdocs-site/overrides/home.html`
- Create: `tools/mkdocs-site/assets/extra.css`
- Create: `tools/mkdocs-site/assets/extra.js`
- Create: `tools/mkdocs-site/assets/mathjax.js`
- Create: `tools/mkdocs-site/assets/diagram.js`
- Create: `tools/mkdocs-site/package.json`
- Create: `tools/mkdocs-site/package-lock.json`
- Create: `tools/mkdocs_site/search_index.py`
- Create: `tools/mkdocs_site/tests/test_theme_contract.py`
- Modify: `tools/mkdocs_site/staging.py`
- Modify: `tools/mkdocs_site/config.py`

**Interfaces:**
- Consumes: `mkdocs_preview.nav_title`, inventory domain data, generated navigation.
- Produces: filename nav labels, Chinese document/head/search titles, Source Atlas homepage, and responsive layout.
- Produces: `rewrite_search_index(site: Path, inventory: Inventory) -> None`.

- [ ] **Step 1: Write failing theme-contract tests**

Tests inspect rendered fixture HTML and assert:

```python
assert soup.select_one(".md-nav__link[data-nav-title='10_article']")
assert soup.select_one("h1").get_text(strip=True) == "上下文并行"
assert soup.select_one(".kb-file-path").get_text(strip=True).endswith("10_article.md")
assert soup.title.get_text(strip=True).startswith("上下文并行")
assert soup.select_one(".kb-source-atlas")
assert not soup.select_one("body.kb-page-index .md-sidebar--secondary")
```

Each assertion obtains `soup = BeautifulSoup((site / <route>).read_text(encoding="utf-8"), "html.parser")` from the appropriate rendered fixture route. Also read `site/search/search_index.json` and assert one document has Chinese `title` while its searchable text contains `10_article`. The test builds the three-page fixture through the real MkDocs command; a test-local `build_fixture_site()` helper owns subprocess setup and cleanup.

- [ ] **Step 2: Run the theme tests and verify failure**

Run: `python -m pytest tools/mkdocs_site/tests/test_theme_contract.py -q`

Expected: FAIL because the overrides and metadata are absent.

- [ ] **Step 3: Implement title separation and homepage data**

The generated config retains filename labels in `nav`. Staging keeps the Chinese frontmatter `title` and adds the source path. `main.html` uses frontmatter title for `<title>` and renders `.kb-file-path`; navigation links receive the generated filename as text and tooltip without replacing the document H1. The root page receives `template: home.html`; `config.extra.source_atlas` is generated from the root domain inventory, not a hand-written list.

`rewrite_search_index()` maps each search document location back through the route manifest, replaces its title with `PageRecord.title`, and prepends `PageRecord.nav_title` to the searchable text exactly once. `cli build` runs this rewrite after MkDocs and before static validation. Unknown search locations are rejected instead of guessed.

Add the theme and local renderer paths to `mkdocs.yml` only in this task:

```yaml
theme:
  custom_dir: tools/mkdocs-site/overrides
extra_css:
  - assets/extra.css
extra_javascript:
  - assets/mathjax.js
  - assets/vendor/mathjax/tex-chtml.js
  - assets/diagram.js
  - assets/vendor/mermaid/mermaid.min.js
  - assets/extra.js
```

The renderer dependency manifest is exact:

```json
{
  "name": "llm-knowledge-mkdocs-assets",
  "private": true,
  "version": "1.0.0",
  "dependencies": {
    "mathjax": "4.1.3",
    "mermaid": "11.17.2"
  }
}
```

Create the lock with `npm install --package-lock-only --prefix tools/mkdocs-site`, install with `npm ci --prefix tools/mkdocs-site`, and make staging copy `tools/mkdocs-site/assets/`, `node_modules/mathjax/tex-chtml.js`, and `node_modules/mermaid/dist/mermaid.min.js` into staged `assets/`. No CDN URL is allowed in generated HTML.

Because the generated config lives below `.mkdocs-cache/`, `config.py` also rewrites `theme.custom_dir` to the absolute worktree path after verifying that the base value is exactly `tools/mkdocs-site/overrides`.

- [ ] **Step 4: Implement the Source Atlas templates**

`home.html` extends `main.html`, renders one search-first hero and three data-backed entry groups: theory, engineering, and courses. It then renders the original `wiki/index.md` content below the atlas. No domain description is duplicated in the template.

- [ ] **Step 5: Implement responsive, accessible CSS**

`extra.css` defines named color variables for light/dark schemes, a 280–320px left rail, 760–840px article measure, two-line leaf labels, active-link rail, quieter inline code, provenance blockquotes, visible focus rings, and breakpoints at 1200px and 900px. Add:

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
    animation-duration: 0.01ms !important;
  }
}
```

At 390px, the document root must have no horizontal overflow. Index/home pages hide the secondary TOC. Article TOCs display only H2/H3 and follow the active heading.

- [ ] **Step 6: Add minimal progressive enhancement**

`extra.js` adds full filename tooltips, marks the active path, and sets Chinese accessible labels after Material instant navigation events. All content and links remain usable when JavaScript is disabled. `mathjax.js` configures local MathJax before the local runtime script executes.

`diagram.js` calls `mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: currentScheme })`, renders `.mermaid` blocks, and reruns on Material's `document$` observable without rendering the same block twice.

- [ ] **Step 7: Run theme and full unit tests**

Run:

```powershell
python -m pytest tools/mkdocs_site/tests/test_theme_contract.py -q
python -m pytest tools/mkdocs_site/tests -q
```

Expected: all tests PASS.

- [ ] **Step 8: Build and inspect representative output**

Run: `npm run docs:mkdocs:build`

Confirm these files exist and contain no CDN URL:

```text
site/index.html
site/02_engineering/index.html
site/02_engineering/02_train_frameworks/megatron-lm/13_megatron_cp_analysis.html
site/search/search_index.json
```

- [ ] **Step 9: Commit the theme**

```bash
git add tools/mkdocs-site tools/mkdocs_site
git commit -m "feat(mkdocs): add Source Atlas documentation theme"
```

---

### Task 6: Enforce Whole-Site Link, Anchor, Asset, and Route Integrity

**Files:**
- Create: `tools/mkdocs_site/validate_site.py`
- Create: `tools/mkdocs_site/tests/test_validate_site.py`
- Modify: `tools/mkdocs_site/cli.py`

**Interfaces:**
- Consumes: built `site/` and `.mkdocs-cache/routes.json`.
- Produces: `ValidationReport` with `pages`, `broken_links`, `missing_anchors`, `missing_assets`, `missing_legacy_routes`, `orphans`.
- Produces: CLI command `validate`; `build` invokes it automatically after MkDocs succeeds.

```python
@dataclass(frozen=True)
class ValidationReport:
    pages: int
    broken_links: tuple[str, ...]
    missing_anchors: tuple[str, ...]
    missing_assets: tuple[str, ...]
    missing_legacy_routes: tuple[str, ...]
    orphans: tuple[str, ...]
```

- [ ] **Step 1: Write failing validator tests with a synthetic site**

```python
def test_validator_reports_each_internal_failure_kind(tmp_path: Path) -> None:
    site = make_site(
        tmp_path,
        index_links=["missing.html", "page.html#missing", "assets/missing.png"],
        pages={"page.html": "<h2 id='present'>Present</h2>"},
    )
    report = validate_site(site, route_manifest=write_routes(tmp_path))
    assert report.broken_links == ("index.html -> missing.html",)
    assert report.missing_anchors == ("index.html -> page.html#missing",)
    assert report.missing_assets == ("index.html -> assets/missing.png",)


def test_validator_walks_manifest_pages_even_when_not_linked(tmp_path: Path) -> None:
    site = make_site(tmp_path, pages={"orphan.html": "<h1>orphan</h1>"})
    report = validate_site(site, route_manifest=write_routes(tmp_path, "orphan.html"))
    assert report.pages == 2
    assert report.missing_legacy_routes == ()
```

`test_validate_site.py` owns `make_site()` and `write_routes()` helpers. They write only tiny deterministic HTML/JSON fixtures beneath `tmp_path`; they do not invoke MkDocs or production staging code.

- [ ] **Step 2: Run validator tests and verify failure**

Run: `python -m pytest tools/mkdocs_site/tests/test_validate_site.py -q`

Expected: FAIL because `validate_site` is undefined.

- [ ] **Step 3: Implement canonical static resolution**

Use Beautiful Soup to parse each HTML file. For every internal `href`, `src`, `srcset`, stylesheet and script URL:

- strip the configured `/llm-knowledge/` prefix before local resolution;
- ignore `mailto:`, `tel:`, `data:`, and external HTTP(S);
- map `/path`, `/path.html`, `/path/`, relative paths, query strings and fragments to concrete output files;
- verify fragment IDs after HTML unescaping;
- read every manifest route even if no rendered page links to it;
- report deterministic POSIX-relative source/target pairs.

The validator returns data; `raise_for_errors(report)` prints grouped counts and exits nonzero if any internal category is non-empty.

- [ ] **Step 4: Integrate validation into build**

`cli build` runs validator only after MkDocs exits 0. A failed validation returns 1 and leaves `site/` intact for inspection. `cli validate` validates an existing output without rebuilding.

- [ ] **Step 5: Run unit and real-site validation**

Run:

```powershell
python -m pytest tools/mkdocs_site/tests/test_validate_site.py -q
npm run docs:mkdocs:build
python -m tools.mkdocs_site.cli validate
```

Expected real report: all internal error categories equal 0 and page count equals inventory count.

- [ ] **Step 6: Commit the validator**

```bash
git add tools/mkdocs_site
git commit -m "test(mkdocs): enforce whole-site route integrity"
```

---

### Task 7: Add Live Source Synchronization and Browser Smoke Tests

**Files:**
- Create: `tools/mkdocs_site/watch.py`
- Create: `tools/mkdocs_site/tests/test_watch.py`
- Create: `tools/mkdocs-site/smoke.mjs`
- Modify: `tools/mkdocs_site/cli.py`
- Modify: `package.json`

**Interfaces:**
- Consumes: the staging/build/validation pipeline.
- Produces: source snapshot polling that restages on create/modify/move/delete.
- Produces: browser smoke command used by `npm run docs:mkdocs:test`.

```python
@dataclass(frozen=True)
class TreeSnapshot:
    digest: str
    files: tuple[PurePosixPath, ...]
```

- [ ] **Step 1: Write failing source-snapshot tests**

```python
def test_snapshot_changes_for_create_modify_move_and_delete(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    first = snapshot_tree(wiki)
    page = wiki / "a.md"
    page.write_text("# A", encoding="utf-8")
    second = snapshot_tree(wiki)
    page.rename(wiki / "b.md")
    third = snapshot_tree(wiki)
    (wiki / "b.md").unlink()
    fourth = snapshot_tree(wiki)
    assert len({first.digest, second.digest, third.digest, fourth.digest}) == 4
```

- [ ] **Step 2: Implement deterministic polling without a new runtime dependency**

`snapshot_tree()` hashes sorted relative paths, sizes and nanosecond mtimes for Markdown and assets. `watch_changes()` polls every 500ms, debounces for 250ms, and invokes one injected callback per stable change. The callback stops the MkDocs child, rescans inventory, atomically restages, rewrites generated config, and relaunches MkDocs so navigation additions, moves, and deletions take effect immediately. If conversion fails, it prints the error and relaunches against the previous working stage/config; the next source change retries.

- [ ] **Step 3: Implement `cli serve` process lifecycle**

`serve` stages once, verifies the requested port is available, launches:

```text
python -m mkdocs serve --dirtyreload --dev-addr 127.0.0.1:<port> -f .mkdocs-cache/mkdocs.generated.yml
```

It starts the watcher, waits for HTTP readiness after the initial launch and every controlled restart, opens the browser unless `--no-open` is present, forwards Ctrl+C, terminates the child, and never binds a public interface by default.

- [ ] **Step 4: Write the browser smoke script**

Reuse the repository-local `puppeteer-core` installation and executable discovery pattern from `tools/docs-site/smoke.mjs`. The script starts `cli serve --no-open` on a free loopback port and checks 390, 768, 1280 and 1600px viewports.

Required assertions:

- homepage contains `.kb-source-atlas` and no secondary TOC;
- the current domain path and representative sibling links render correctly while unrelated branches remain pruned;
- the Megatron CP nav label is exactly `13_megatron_cp_analysis`;
- `/02_engineering/` displays `工程实现 — 知识地图`;
- article H1 remains the Chinese title;
- search finds the article by both Chinese title and English stem;
- article links, breadcrumbs, next/previous and GitHub source link return valid targets;
- `document.documentElement.scrollWidth <= window.innerWidth` at every viewport;
- browser console errors and failed internal requests are empty.

- [ ] **Step 5: Complete the parallel npm test script**

Ensure `npm run docs:mkdocs:test` runs unit tests, clean build/validator, and browser smoke while existing `npm run docs:test` still runs Quartz.

- [ ] **Step 6: Run the complete MkDocs suite**

Run: `npm run docs:mkdocs:test`

Expected: unit, build, static validation, search, responsive and browser checks all PASS.

- [ ] **Step 7: Prove local automatic refresh**

In a temporary fixture repository, start `serve`, create a page, verify its URL appears, rename it, verify the old URL disappears and new URL appears, then delete it. Do not modify the real `wiki/` for this test.

- [ ] **Step 8: Commit live preview and smoke coverage**

```bash
git add package.json tools/mkdocs_site tools/mkdocs-site/smoke.mjs
git commit -m "test(mkdocs): verify live preview and responsive docs UI"
```

---

### Task 8: Run Full Quality Gates and Open the Local Preview

**Files:**
- Modify only files already introduced by this plan when a regression test proves a correction is required.
- Do not modify: `wiki/**`, `.github/workflows/pages.yml`, or current Quartz deployment configuration.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: a clean feature branch, a running local preview, and an evidence report for user review.

- [ ] **Step 1: Run formatting and focused MkDocs tests**

Run:

```powershell
git diff --check
python -m pytest tools/mkdocs_site/tests -q
```

Expected: no whitespace errors and all MkDocs tests PASS.

- [ ] **Step 2: Run the repository source gates**

Run:

```powershell
python tools/check_links.py --strict
python tools/check_math.py --changed --strict
python -m pytest tools/
```

Expected: broken/ambiguous/bare index/orphans remain 0, math reports no introduced findings, and the full Python suite passes.

- [ ] **Step 3: Prove the current Quartz site still builds**

Run: `npm run docs:test`

Expected: existing Quartz unit and browser smoke checks PASS.

- [ ] **Step 4: Prove the complete MkDocs preview**

Run: `npm run docs:mkdocs:test`

Expected: 426 baseline pages (or the current synchronized inventory count), zero internal link/anchor/asset/route errors, and all browser assertions PASS.

- [ ] **Step 5: Check source immutability and branch scope**

Run:

```powershell
git status --short
git diff origin/main -- wiki .github/workflows/pages.yml
```

Expected: no migration changes under `wiki/` or the Pages workflow. Only the design, plan, MkDocs toolchain, theme, tests, ignore entries and parallel scripts differ.

- [ ] **Step 6: Commit any final regression-only corrections**

If Step 1–5 exposed a defect, add a failing regression test, make the smallest correction, rerun the affected gate, then commit:

```bash
git add <tested-preview-files>
git commit -m "fix(mkdocs): close full-preview regression"
```

If no correction was needed, do not create an empty commit.

- [ ] **Step 7: Start and open the review server**

Run:

```powershell
npm run docs:mkdocs -- --port 8081
```

Open `http://127.0.0.1:8081/` in the in-app browser and leave the server running for user review. Inspect the homepage, one folder index, the Megatron CP article, search, light/dark themes and mobile emulation before reporting readiness.

- [ ] **Step 8: Hand off without publishing**

Report the local URL, branch name, commit list, page/link/test counts, known visual trade-offs, and the explicit statement that the online Quartz site remains unchanged. Wait for the user's visual approval before writing the separate production cutover plan.

---

## Plan Self-Review Result

- **Spec coverage:** source authority, Wikilinks, dynamic navigation, title separation, search, Source Atlas UI, URL compatibility, assets, local refresh, strict validation, worktree isolation, preview-first rollout and rollback boundary all map to explicit tasks.
- **Deliberate split:** GitHub Pages workflow replacement is excluded from this plan because the spec requires local visual approval first. It will be a separate production-cutover plan.
- **Type consistency:** `PageRecord`, `Inventory`, `RouteRecord`, `BuildPaths`, `StageResult` and `ValidationReport` have one owning task and stable names across consumers.
- **Placeholder scan:** no deferred implementation markers or unspecified validation steps remain.
