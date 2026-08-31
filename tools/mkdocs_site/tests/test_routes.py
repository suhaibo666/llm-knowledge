import json
from pathlib import Path

from tools.mkdocs_site.inventory import scan_inventory
from tools.mkdocs_site.routes import build_route_manifest, write_route_manifest


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


def test_write_route_manifest_serializes_records_by_source(
    fixture_wiki: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "routes.json"
    write_route_manifest(build_route_manifest(scan_inventory(fixture_wiki)), destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == [
        {
            "source": "domain/10_article.md",
            "output": "domain/10_article.html",
            "public_urls": ["/domain/10_article", "/domain/10_article.html"],
        },
        {
            "source": "domain/index.md",
            "output": "domain/index.html",
            "public_urls": ["/domain/", "/domain/index.html"],
        },
        {
            "source": "index.md",
            "output": "index.html",
            "public_urls": ["/", "/index.html"],
        },
    ]
