from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import write_generated_config
from .inventory import scan_inventory
from .models import BuildPaths, Inventory
from .search_index import rewrite_search_index
from .staging import StageResult, stage_wiki


@dataclass(frozen=True)
class Operations:
    inventory: Callable[[Path], Inventory]
    stage: Callable[[BuildPaths, Inventory], StageResult]
    config: Callable[[BuildPaths, Inventory, StageResult], Path]
    mkdocs: Callable[[list[str], Path], int]
    rewrite_search: Callable[[Path, Inventory], None]


def _run_mkdocs(command: list[str], cwd: Path) -> int:
    return subprocess.run(command, cwd=cwd, check=False).returncode


def _default_operations() -> Operations:
    return Operations(
        scan_inventory,
        stage_wiki,
        write_generated_config,
        _run_mkdocs,
        rewrite_search_index,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the local MkDocs preview")
    parser.add_argument("command", choices=("stage", "build", "serve"))
    return parser


def main(
    argv: Sequence[str] | None = None, operations: Operations | None = None
) -> int:
    args = _parser().parse_args(argv)
    active = operations or _default_operations()
    repo = Path(__file__).resolve().parents[2]
    paths = BuildPaths.from_repo(repo)
    inventory = active.inventory(paths.wiki)
    stage_result = active.stage(paths, inventory)
    if args.command == "stage":
        return 0

    generated_config = active.config(paths, inventory, stage_result)
    if args.command == "build":
        mkdocs_args = ["build", "--strict"]
    else:
        mkdocs_args = ["serve", "--dev-addr", "127.0.0.1:8000"]
    command = [
        sys.executable,
        "-m",
        "mkdocs",
        *mkdocs_args,
        "-f",
        str(generated_config),
    ]
    result = active.mkdocs(command, paths.repo)
    if args.command == "build" and result == 0:
        active.rewrite_search(paths.site, inventory)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
