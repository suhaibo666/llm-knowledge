from collections.abc import Mapping
from pathlib import Path

import yaml

from .models import BuildPaths, Inventory
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
    return config


def write_generated_config(
    paths: BuildPaths, inventory: Inventory, stage_result: StageResult
) -> Path:
    """Write a generated MkDocs config for the latest staged inventory."""
    del stage_result
    config = _load_base_config(paths.repo / "mkdocs.yml")
    config["docs_dir"] = str(paths.staging.resolve())
    config["site_dir"] = str(paths.site.resolve())
    config["nav"] = build_navigation(inventory)
    paths.generated_config.parent.mkdir(parents=True, exist_ok=True)
    paths.generated_config.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return paths.generated_config
