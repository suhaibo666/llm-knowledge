from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .models import Inventory, PageRecord
from .routes import build_route_manifest


class SearchIndexError(ValueError):
    """Raised when the generated search index violates the route contract."""


def _route_lookup(inventory: Inventory) -> dict[str, PageRecord]:
    lookup: dict[str, PageRecord] = {}
    for route in build_route_manifest(inventory):
        page = inventory.by_relative[route.source.with_suffix("")]
        locations = {route.output.as_posix()}
        locations.update(url.lstrip("/") for url in route.public_urls)
        for location in locations:
            existing = lookup.get(location)
            if existing is not None and existing != page:
                raise SearchIndexError(f"ambiguous route manifest location: {location}")
            lookup[location] = page
    return lookup


def _page_for_location(
    location: str, lookup: Mapping[str, PageRecord]
) -> PageRecord:
    route = location.split("#", 1)[0]
    page = lookup.get(route)
    if page is None:
        raise SearchIndexError(f"unknown search location: {location}")
    return page


def rewrite_search_index(site: Path, inventory: Inventory) -> None:
    """Restore document titles and filename terms in MkDocs' search index."""
    destination = site / "search/search_index.json"
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SearchIndexError(f"cannot read generated search index: {destination}") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("docs"), list):
        raise SearchIndexError("generated search index must contain a docs list")

    lookup = _route_lookup(inventory)
    for document in payload["docs"]:
        if not isinstance(document, dict):
            raise SearchIndexError("search document must be an object")
        location = document.get("location")
        text = document.get("text")
        if not isinstance(location, str) or not isinstance(text, str):
            raise SearchIndexError("search document location and text must be strings")
        page = _page_for_location(location, lookup)
        document["title"] = page.title
        prefix = f"{page.nav_title} "
        if text != page.nav_title and not text.startswith(prefix):
            document["text"] = prefix + text

    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(destination)
