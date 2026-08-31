from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import yaml

from .models import BuildPaths, Inventory, PageRecord
from .navigation import build_navigation
from .staging import StageResult


class ConfigError(ValueError):
    """Raised when the stable base configuration breaks its path contract."""


_BASE_CONTRACT: dict[str, object] = {
    "docs_dir": ".mkdocs-cache/docs",
    "site_dir": "site",
    "site_url": "https://suhaibo666.github.io/llm-knowledge/",
    "use_directory_urls": False,
}
_THEME_OVERRIDE = "tools/mkdocs-site/overrides"


def _load_base_config(path: Path) -> dict[str, object]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot load base configuration: {path}") from error
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"base configuration must be a mapping: {path}")
    config = dict(loaded)
    for key, expected in _BASE_CONTRACT.items():
        if config.get(key) != expected:
            raise ConfigError(f"{key} must remain {expected!r} in {path.name}")
    theme = config.get("theme")
    if not isinstance(theme, Mapping):
        raise ConfigError(f"theme must be a mapping in {path.name}")
    if theme.get("custom_dir") != _THEME_OVERRIDE:
        raise ConfigError(
            f"theme.custom_dir must remain {_THEME_OVERRIDE!r} in {path.name}"
        )
    return config


def _entry(page: PageRecord, inventory: Inventory) -> dict[str, object]:
    relative = page.relative
    directory = relative.parent if page.is_index else relative.with_suffix("")
    page_count = sum(
        1
        for candidate in inventory.pages
        if candidate.relative == relative
        or candidate.relative.is_relative_to(directory)
    )
    return {
        "title": page.title,
        "source_label": directory.name,
        "href": relative.with_suffix(".html").as_posix(),
        "page_count": page_count,
    }


def _domain_entries(root: PurePosixPath, inventory: Inventory) -> list[dict[str, object]]:
    pages = (
        page
        for page in inventory.pages
        if page.is_index and page.relative.parent.parent == root
    )
    return [_entry(page, inventory) for page in pages]


def _source_atlas(inventory: Inventory) -> dict[str, list[dict[str, object]]]:
    courses = (
        page
        for page in inventory.pages
        if page.relative.parent == PurePosixPath("courses") and not page.is_index
    )
    return {
        "theory": _domain_entries(PurePosixPath("01_theory"), inventory),
        "engineering": _domain_entries(PurePosixPath("02_engineering"), inventory),
        "courses": [_entry(page, inventory) for page in courses],
    }


def write_generated_config(
    paths: BuildPaths, inventory: Inventory, stage_result: StageResult
) -> Path:
    """Write a generated MkDocs config for the latest staged inventory."""
    del stage_result
    config = _load_base_config(paths.repo / "mkdocs.yml")
    config["docs_dir"] = str(paths.staging.resolve())
    config["site_dir"] = str(paths.site.resolve())
    theme = dict(config["theme"])
    theme["custom_dir"] = str((paths.repo / _THEME_OVERRIDE).resolve())
    config["theme"] = theme
    config["nav"] = build_navigation(inventory)
    base_extra = config.get("extra")
    if base_extra is not None and not isinstance(base_extra, Mapping):
        raise ConfigError("extra must be a mapping")
    extra = dict(base_extra or {})
    extra["source_atlas"] = _source_atlas(inventory)
    config["extra"] = extra
    paths.generated_config.parent.mkdir(parents=True, exist_ok=True)
    paths.generated_config.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return paths.generated_config
