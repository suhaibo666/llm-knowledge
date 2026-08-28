from __future__ import annotations

import json
from pathlib import Path

import pytest
from markdown import markdown as render_markdown

from tools.mkdocs_site.inventory import scan_inventory
from tools.mkdocs_site.wikilinks import rewrite_wikilinks
from tools.mkdocs_site.validate_site import (
    SiteValidationError,
    ValidationReport,
    raise_for_errors,
    validate_site,
)


def make_site(
    tmp_path: Path,
    *,
    index_links: list[str] | None = None,
    index_body: str = "",
    pages: dict[str, str] | None = None,
) -> Path:
    site = tmp_path / "site"
    site.mkdir()
    links = "".join(f'<a href="{target}">{target}</a>' for target in index_links or [])
    (site / "index.html").write_text(
        f"<!doctype html><html><body>{links}{index_body}</body></html>",
        encoding="utf-8",
    )
    for relative, body in (pages or {}).items():
        destination = site / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith(".html") and "<html" not in body:
            body = f"<!doctype html><html><body>{body}</body></html>"
        destination.write_text(body, encoding="utf-8")
    return site


def write_routes(
    tmp_path: Path,
    *outputs: str,
    public_urls: dict[str, list[str]] | None = None,
) -> Path:
    routes = []
    for output in ("index.html", *outputs):
        source = "index.md" if output == "index.html" else output.removesuffix(".html") + ".md"
        if public_urls is not None and output in public_urls:
            urls = public_urls[output]
        elif output == "index.html":
            urls = ["/", "/index.html"]
        elif output.endswith("/index.html"):
            parent = output.removesuffix("index.html")
            urls = [f"/{parent}", f"/{output}"]
        else:
            urls = [f"/{output.removesuffix('.html')}", f"/{output}"]
        routes.append({"source": source, "output": output, "public_urls": urls})
    manifest = tmp_path / "routes.json"
    manifest.write_text(json.dumps(routes), encoding="utf-8")
    return manifest


def test_validation_report_owns_only_the_public_contract_fields() -> None:
    assert tuple(ValidationReport.__dataclass_fields__) == (
        "pages",
        "broken_links",
        "missing_anchors",
        "missing_assets",
        "missing_legacy_routes",
        "orphans",
    )


def test_validator_reports_each_internal_failure_kind(tmp_path: Path) -> None:
    site = make_site(
        tmp_path,
        index_links=["missing.html", "page.html#missing", "assets/missing.png"],
        pages={"page.html": "<h2 id='present'>Present</h2>"},
    )

    report = validate_site(site, route_manifest=write_routes(tmp_path, "page.html"))

    assert report.broken_links == ("index.html -> missing.html",)
    assert report.missing_anchors == ("index.html -> page.html#missing",)
    assert report.missing_assets == ("index.html -> assets/missing.png",)


def test_validator_walks_manifest_pages_even_when_not_linked(tmp_path: Path) -> None:
    site = make_site(
        tmp_path,
        pages={
            "orphan.html": '<a href="nested/missing.html">missing</a>',
        },
    )

    report = validate_site(site, route_manifest=write_routes(tmp_path, "orphan.html"))

    assert report.pages == 2
    assert report.broken_links == ("orphan.html -> nested/missing.html",)
    assert report.missing_legacy_routes == ()
    assert report.orphans == ()


def test_validator_resolves_supported_page_urls_and_decoded_anchors(
    tmp_path: Path,
) -> None:
    site = make_site(
        tmp_path,
        index_body="""
        <a href="/llm-knowledge/guide?mode=full#caf%C3%A9">prefixed</a>
        <a href="guide.html#legacy&amp;name">html</a>
        <a href="guide/#caf%C3%A9">directory</a>
        <a href="nested/../guide/index.html">relative</a>
        """,
        pages={
            "guide.html": '<h2 id="café">Cafe</h2><a name="legacy&amp;name"></a>',
            "guide/index.html": '<h2 id="café">Cafe</h2>',
        },
    )
    manifest = write_routes(
        tmp_path,
        "guide.html",
        "guide/index.html",
        public_urls={
            "index.html": ["/", "/index.html"],
            "guide.html": ["/guide", "/guide.html"],
            "guide/index.html": ["/guide/", "/guide/index.html"],
        },
    )

    report = validate_site(site, route_manifest=manifest)

    assert report.broken_links == ()
    assert report.missing_anchors == ()
    assert report.missing_assets == ()


def test_validator_resolves_fragment_only_urls_against_the_current_page(
    tmp_path: Path,
) -> None:
    site = make_site(
        tmp_path,
        pages={
            "nested/page.html": '<h2 id="present">Present</h2><a href="#present">same page</a>',
        },
    )

    report = validate_site(
        site,
        route_manifest=write_routes(tmp_path, "nested/page.html"),
    )

    assert report.missing_anchors == ()
    assert report.broken_links == ()


def test_validator_checks_src_srcset_stylesheets_scripts_and_html_assets(
    tmp_path: Path,
) -> None:
    site = make_site(
        tmp_path,
        index_body="""
        <img src="assets/present.png" srcset="assets/present.png 1x, assets/missing-2x.png 2x">
        <script src="/llm-knowledge/assets/missing.js"></script>
        <link rel="stylesheet" href="assets/missing.css">
        <video poster="assets/missing-poster.jpg"></video>
        <object data="assets/missing.bin"></object>
        <iframe src="assets/interactive.html"></iframe>
        """,
        pages={
            "assets/present.png": "png",
            "assets/interactive.html": '<script src="nested.js"></script>',
            "assets/nested.js": "js",
            "x/assets/provenance.html": '<script src="missing-runtime.js"></script>',
            "unlinked.html": "orphan",
            "404.html": "auxiliary",
        },
    )

    report = validate_site(site, route_manifest=write_routes(tmp_path))

    assert report.missing_assets == (
        "index.html -> assets/missing-2x.png",
        "index.html -> assets/missing-poster.jpg",
        "index.html -> assets/missing.bin",
        "index.html -> assets/missing.css",
        "index.html -> assets/missing.js",
        "x/assets/provenance.html -> x/assets/missing-runtime.js",
    )
    assert report.orphans == ("unlinked.html",)
    assert report.pages == 1


def test_validator_ignores_only_declared_external_reference_kinds(
    tmp_path: Path,
) -> None:
    site = make_site(
        tmp_path,
        index_body="""
        <a href="https://example.com/missing">https</a>
        <a href="http://example.com/missing">http</a>
        <a href="//cdn.example.com/missing.js">cdn</a>
        <a href="mailto:docs@example.com">mail</a>
        <a href="tel:+86123">phone</a>
        <img src="data:image/png;base64,AAAA">
        <img srcset="data:image/png;base64,AAAA 1x">
        """,
    )

    report = validate_site(site, route_manifest=write_routes(tmp_path))

    assert report.broken_links == ()
    assert report.missing_assets == ()


@pytest.mark.parametrize(
    ("body", "category", "target"),
    [
        ('<a href="javascript:alert(1)">unsafe</a>', "broken_links", "javascript:alert(1)"),
        ('<a href="file:///C:/private.txt">unsafe</a>', "broken_links", "file:///C:/private.txt"),
        ('<a href="%2e%2e/outside.html">escape</a>', "broken_links", "../outside.html"),
        ('<img src="..%2foutside.png">', "missing_assets", "../outside.png"),
    ],
)
def test_validator_reports_unsafe_schemes_and_encoded_traversal(
    tmp_path: Path, body: str, category: str, target: str
) -> None:
    site = make_site(tmp_path, index_body=body)

    report = validate_site(site, route_manifest=write_routes(tmp_path))

    assert getattr(report, category) == (f"index.html -> {target}",)


def test_rendered_suffix_guess_remains_a_missing_anchor(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    source = wiki / "index.md"
    source.write_text(
        "# Test page\n\n[Design](#invented)\n\n## Architecture Design\n",
        encoding="utf-8",
    )
    inventory = scan_inventory(wiki)
    page = inventory.by_relative[next(iter(inventory.by_relative))]
    rendered = render_markdown(
        rewrite_wikilinks(source.read_text(encoding="utf-8"), page, inventory),
        extensions=["toc"],
    )
    site = make_site(tmp_path, index_body=rendered)

    report = validate_site(site, route_manifest=write_routes(tmp_path))

    assert report.missing_anchors == ("index.html -> index.html#invented",)


def test_validator_requires_exact_case_for_pages_and_assets(tmp_path: Path) -> None:
    site = make_site(
        tmp_path,
        index_links=["Guide.html"],
        index_body='<img src="assets/logo.png">',
        pages={
            "guide.html": "guide",
            "assets/Logo.png": "logo",
        },
    )
    manifest = write_routes(tmp_path, "guide.html")

    report = validate_site(site, route_manifest=manifest)

    assert report.broken_links == ("index.html -> Guide.html",)
    assert report.missing_assets == ("index.html -> assets/logo.png",)


def test_validator_requires_exact_case_for_manifest_output(tmp_path: Path) -> None:
    site = make_site(tmp_path, pages={"guide.html": "guide"})
    manifest = write_routes(
        tmp_path,
        "Guide.html",
        public_urls={"Guide.html": ["/Guide.html"]},
    )

    report = validate_site(site, route_manifest=manifest)

    assert report.missing_legacy_routes == ("Guide.md -> Guide.html",)


def test_validator_checks_every_declared_public_url_maps_to_its_output(
    tmp_path: Path,
) -> None:
    site = make_site(tmp_path, pages={"guide.html": "guide"})
    manifest = write_routes(
        tmp_path,
        "guide.html",
        public_urls={
            "index.html": ["/", "/index.html"],
            "guide.html": ["/wrong", "/llm-knowledge/guide.html"],
        },
    )

    report = validate_site(site, route_manifest=manifest)

    assert report.missing_legacy_routes == ("guide.md -> wrong",)


def test_validator_reports_missing_route_output_independently_of_public_urls(
    tmp_path: Path,
) -> None:
    site = make_site(tmp_path)
    manifest = write_routes(
        tmp_path,
        "nested/missing.html",
        public_urls={"nested/missing.html": ["/nested/missing"]},
    )

    report = validate_site(site, route_manifest=manifest)

    assert report.pages == 2
    assert report.missing_legacy_routes == (
        "nested/missing.md -> nested/missing.html",
    )


def test_validator_rejects_route_without_public_urls(tmp_path: Path) -> None:
    site = make_site(tmp_path)
    manifest = write_routes(
        tmp_path,
        "missing.html",
        public_urls={"missing.html": []},
    )

    with pytest.raises(ValueError, match="public_urls"):
        validate_site(site, route_manifest=manifest)


def test_report_output_is_grouped_sorted_and_fails_only_for_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    failing = ValidationReport(
        pages=2,
        broken_links=("z.html -> z", "a.html -> a"),
        missing_anchors=(),
        missing_assets=(),
        missing_legacy_routes=(),
        orphans=(),
    )

    with pytest.raises(SiteValidationError):
        raise_for_errors(failing)

    output = capsys.readouterr().out
    assert "pages: 2" in output
    assert "broken_links: 2" in output
    assert output.index("a.html -> a") < output.index("z.html -> z")

    raise_for_errors(
        ValidationReport(1, (), (), (), (), ()),
    )
    assert "orphans: 0" in capsys.readouterr().out
