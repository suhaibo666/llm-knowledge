from pathlib import PurePosixPath

from .models import Inventory, PageRecord


class NavigationError(ValueError):
    """Raised when the page tree cannot produce valid navigation."""


NavigationItem = dict[str, object] | str
_INDEXLESS_COLLECTIONS = {
    PurePosixPath("changelog"),
    PurePosixPath("courses"),
}


def _directory_navigation(
    directory: PurePosixPath, inventory: Inventory
) -> list[NavigationItem]:
    direct_pages = [
        page for page in inventory.pages if page.relative.parent == directory
    ]
    index = next((page for page in direct_pages if page.is_index), None)
    if index is None and directory not in _INDEXLESS_COLLECTIONS:
        display = directory.as_posix() if directory.parts else "."
        raise NavigationError(f"{display}: directory is missing index.md")

    items: list[NavigationItem] = []
    if index is not None:
        items.append({index.nav_title: index.relative.as_posix()})
    children: dict[str, PageRecord | PurePosixPath] = {
        page.relative.name: page for page in direct_pages if not page.is_index
    }
    for page in inventory.pages:
        try:
            relative = page.relative.relative_to(directory)
        except ValueError:
            continue
        if len(relative.parts) > 1:
            child = directory / relative.parts[0]
            children.setdefault(relative.parts[0], child)

    for name in sorted(children, key=str.casefold):
        child = children[name]
        if isinstance(child, PageRecord):
            items.append({child.nav_title: child.relative.as_posix()})
            continue
        child_items = _directory_navigation(child, inventory)
        if child in _INDEXLESS_COLLECTIONS:
            label = child.name
        else:
            label = inventory.by_relative[child / "index"].nav_title
        items.append({label: child_items})
    return items


def build_navigation(inventory: Inventory) -> list[NavigationItem]:
    """Derive deterministic MkDocs navigation from the inventoried page tree."""
    return _directory_navigation(PurePosixPath(), inventory)
