from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tools.mkdocs_site import cli
from tools.mkdocs_site.cli import Operations, _default_serve_runtime, main
from tools.mkdocs_site.inventory import scan_inventory
from tools.mkdocs_site.models import BuildPaths, Inventory
from tools.mkdocs_site.navigation import build_navigation
from tools.mkdocs_site.staging import StageResult, stage_wiki
from tools.mkdocs_site.validate_site import ValidationReport


def _stage_fixture(paths: BuildPaths, inventory: Inventory) -> StageResult:
    result = stage_wiki(paths, inventory)
    home = paths.staging / "index.md"
    home.write_text(
        home.read_text(encoding="utf-8").replace("template: home.html\n", "", 1),
        encoding="utf-8",
    )
    return result


def _write_fixture_config(
    paths: BuildPaths, inventory: Inventory, _stage: StageResult
) -> Path:
    paths.generated_config.write_text(
        yaml.safe_dump(
            {
                "site_name": "Search Refresh Fixture",
                "docs_dir": str(paths.staging),
                "site_dir": str(paths.site),
                "use_directory_urls": False,
                "theme": {
                    "name": "material",
                    "language": "zh",
                    "font": False,
                    "features": ["search.suggest", "search.highlight"],
                },
                "plugins": [
                    {"search": {"lang": ["zh", "en"], "pipeline": []}}
                ],
                "nav": build_navigation(inventory),
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return paths.generated_config


def run(repo: Path, port: int) -> int:
    fake_cli = repo / "tools/mkdocs_site/cli.py"
    fake_cli.parent.mkdir(parents=True, exist_ok=True)
    cli.__file__ = str(fake_cli)
    operations = Operations(
        scan_inventory,
        _stage_fixture,
        _write_fixture_config,
        lambda _command, _cwd: 0,
        lambda _site, _inventory: None,
        lambda _site, _manifest, _sources=None: ValidationReport(0, (), (), (), (), ()),
    )
    return main(
        ["serve", "--port", str(port), "--no-open"],
        operations,
        _default_serve_runtime(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    raise SystemExit(run(args.repo.resolve(), args.port))
