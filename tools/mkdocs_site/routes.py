import json
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from .models import Inventory, RouteRecord


class RouteError(ValueError):
    """Raised when source pages would expose conflicting legacy routes."""


def _urls_for(source: PurePosixPath) -> tuple[str, ...]:
    html = source.with_suffix(".html")
    if source.name == "index.md":
        if source.parent == PurePosixPath("."):
            return ("/", "/index.html")
        return (f"/{source.parent.as_posix()}/", f"/{html.as_posix()}")
    return (f"/{source.with_suffix('').as_posix()}", f"/{html.as_posix()}")


def build_route_manifest(inventory: Inventory) -> tuple[RouteRecord, ...]:
    """Build the stable public URL manifest for the supplied page inventory."""
    records = tuple(
        RouteRecord(
            source=page.relative,
            output=page.relative.with_suffix(".html"),
            public_urls=_urls_for(page.relative),
        )
        for page in inventory.pages
    )
    seen_outputs: set[PurePosixPath] = set()
    seen_urls: set[str] = set()
    for record in records:
        if record.output in seen_outputs:
            raise RouteError(f"duplicate output path: {record.output.as_posix()}")
        seen_outputs.add(record.output)
        for url in record.public_urls:
            if url in seen_urls:
                raise RouteError(f"duplicate public URL: {url}")
            seen_urls.add(url)
    return records


def write_route_manifest(records: Iterable[RouteRecord], destination: Path) -> None:
    """Write *records* as deterministic UTF-8 JSON at *destination*."""
    ordered = sorted(records, key=lambda record: record.source.as_posix().casefold())
    payload = [
        {
            "source": record.source.as_posix(),
            "output": record.output.as_posix(),
            "public_urls": list(record.public_urls),
        }
        for record in ordered
    ]
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
