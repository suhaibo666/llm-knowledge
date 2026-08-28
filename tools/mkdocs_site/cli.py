from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import write_generated_config
from .inventory import scan_inventory
from .models import BuildPaths, Inventory
from .search_index import rewrite_search_index
from .staging import StageResult, stage_wiki
from .validate_site import (
    SiteValidationError,
    ValidationReport,
    raise_for_errors,
    validate_site,
)
from .watch import watch_changes


@dataclass(frozen=True)
class Operations:
    inventory: Callable[[Path], Inventory]
    stage: Callable[[BuildPaths, Inventory], StageResult]
    config: Callable[[BuildPaths, Inventory, StageResult], Path]
    mkdocs: Callable[[list[str], Path], int]
    rewrite_search: Callable[[Path, Inventory], None]
    validate: Callable[[Path, Path], ValidationReport]


@dataclass(frozen=True)
class ServeRuntime:
    check_port: Callable[[int], None]
    start: Callable[[list[str], Path], Any]
    wait_ready: Callable[[str, Any], None]
    open_browser: Callable[[str], object]
    watch: Callable[[Path, Callable[[], None], Callable[[], bool]], None]
    stop: Callable[[Any], None]


@dataclass
class _ServeState:
    child: Any


@dataclass(frozen=True)
class _WorkingBackup:
    root: Path
    moved: tuple[tuple[Path, Path], ...]


def _run_mkdocs(command: list[str], cwd: Path) -> int:
    return subprocess.run(command, cwd=cwd, check=False).returncode


def _default_operations() -> Operations:
    return Operations(
        scan_inventory,
        stage_wiki,
        write_generated_config,
        _run_mkdocs,
        rewrite_search_index,
        lambda site, manifest: validate_site(site, route_manifest=manifest),
    )


def _assert_loopback_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", port))


def _start_mkdocs(command: list[str], cwd: Path) -> subprocess.Popen[bytes]:
    options: dict[str, object] = {
        "cwd": cwd,
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    return subprocess.Popen(command, **options)  # type: ignore[arg-type]


def _wait_for_http(
    url: str,
    child: subprocess.Popen[bytes],
    *,
    timeout: float = 60.0,
) -> None:
    deadline = time.monotonic() + timeout
    last_failure = "no response"
    while time.monotonic() < deadline:
        return_code = child.poll()
        if return_code is not None:
            raise RuntimeError(
                f"MkDocs exited before HTTP readiness with status {return_code}"
            )
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if 200 <= response.status < 400:
                    return
                last_failure = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as error:
            last_failure = str(error)
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {url} ({last_failure})")


def _stop_mkdocs(child: subprocess.Popen[bytes]) -> None:
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        child.kill()
    try:
        child.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        print(f"MkDocs process {child.pid} did not exit after kill", file=sys.stderr)


def _default_serve_runtime() -> ServeRuntime:
    return ServeRuntime(
        _assert_loopback_port_available,
        _start_mkdocs,
        _wait_for_http,
        webbrowser.open,
        lambda wiki, callback, stop: watch_changes(
            wiki, callback, stop_requested=stop
        ),
        _stop_mkdocs,
    )


def _serve_command(generated_config: Path, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mkdocs",
        "serve",
        "--dirtyreload",
        "--dev-addr",
        f"127.0.0.1:{port}",
        "-f",
        str(generated_config),
    ]


def _working_outputs(paths: BuildPaths) -> tuple[Path, ...]:
    return (
        paths.staging,
        paths.cache / "routes.json",
        paths.generated_config,
    )


def _stash_working_outputs(paths: BuildPaths) -> _WorkingBackup:
    paths.cache.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=".serve-backup-", dir=paths.cache))
    moved: list[tuple[Path, Path]] = []
    try:
        for index, source in enumerate(_working_outputs(paths)):
            if not source.exists():
                continue
            destination = root / f"{index}-{source.name}"
            source.replace(destination)
            moved.append((source, destination))
    except BaseException:
        for source, destination in reversed(moved):
            if destination.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                destination.replace(source)
        shutil.rmtree(root, ignore_errors=True)
        raise
    return _WorkingBackup(root, tuple(moved))


def _discard_working_outputs(paths: BuildPaths) -> None:
    for target in _working_outputs(paths):
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _restore_working_outputs(paths: BuildPaths, backup: _WorkingBackup) -> None:
    _discard_working_outputs(paths)
    for source, destination in backup.moved:
        source.parent.mkdir(parents=True, exist_ok=True)
        destination.replace(source)
    shutil.rmtree(backup.root, ignore_errors=True)


def _refresh_preview(
    paths: BuildPaths,
    operations: Operations,
    runtime: ServeRuntime,
    state: _ServeState,
    port: int,
) -> None:
    """Replace the served snapshot, restoring it if any refresh step fails."""
    url = f"http://127.0.0.1:{port}/"
    runtime.stop(state.child)
    backup = _stash_working_outputs(paths)
    candidate = None
    try:
        inventory = operations.inventory(paths.wiki)
        stage_result = operations.stage(paths, inventory)
        generated_config = operations.config(paths, inventory, stage_result)
        candidate = runtime.start(_serve_command(generated_config, port), paths.repo)
        runtime.wait_ready(url, candidate)
    except BaseException as refresh_error:
        if candidate is not None:
            runtime.stop(candidate)
        _restore_working_outputs(paths, backup)
        try:
            recovery = runtime.start(
                _serve_command(paths.generated_config, port), paths.repo
            )
            runtime.wait_ready(url, recovery)
            state.child = recovery
        except BaseException as recovery_error:
            raise RuntimeError(
                f"preview refresh failed and the previous server could not restart: "
                f"{recovery_error}"
            ) from refresh_error
        raise
    else:
        shutil.rmtree(backup.root, ignore_errors=True)
        state.child = candidate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the local MkDocs preview")
    parser.add_argument("command", choices=("stage", "build", "serve", "validate"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true")
    return parser


def _validation_result(report: ValidationReport) -> int:
    try:
        raise_for_errors(report)
    except SiteValidationError:
        return 1
    return 0


def main(
    argv: Sequence[str] | None = None,
    operations: Operations | None = None,
    serve_runtime: ServeRuntime | None = None,
) -> int:
    args = _parser().parse_args(argv)
    active = operations or _default_operations()
    repo = Path(__file__).resolve().parents[2]
    paths = BuildPaths.from_repo(repo)
    if args.command == "validate":
        report = active.validate(paths.site, paths.cache / "routes.json")
        return _validation_result(report)

    if args.command == "serve":
        runtime = serve_runtime or _default_serve_runtime()
        runtime.check_port(args.port)

    inventory = active.inventory(paths.wiki)
    stage_result = active.stage(paths, inventory)
    if args.command == "stage":
        return 0

    generated_config = active.config(paths, inventory, stage_result)
    if args.command == "build":
        command = [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "-f",
            str(generated_config),
        ]
    else:
        command = _serve_command(generated_config, args.port)
    if args.command == "serve":
        url = f"http://127.0.0.1:{args.port}/"
        state = _ServeState(runtime.start(command, paths.repo))
        try:
            try:
                runtime.wait_ready(url, state.child)
                if not args.no_open:
                    runtime.open_browser(url)

                def refresh() -> None:
                    _refresh_preview(paths, active, runtime, state, args.port)

                runtime.watch(
                    paths.wiki,
                    refresh,
                    lambda: state.child.poll() is not None,
                )
                return_code = state.child.poll()
                return return_code if return_code is not None else 0
            except KeyboardInterrupt:
                return 130
        finally:
            runtime.stop(state.child)

    result = active.mkdocs(command, paths.repo)
    if args.command == "build" and result == 0:
        active.rewrite_search(paths.site, inventory)
        return _validation_result(
            active.validate(paths.site, stage_result.route_manifest)
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
