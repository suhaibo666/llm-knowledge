import json
import shutil
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from tools.mkdocs_site import cli
from tools.mkdocs_site.cli import Operations, _default_serve_runtime, main
from tools.mkdocs_site.inventory import scan_inventory
from tools.mkdocs_site.models import BuildPaths, Inventory
from tools.mkdocs_site.navigation import build_navigation
from tools.mkdocs_site.staging import StageResult
from tools.mkdocs_site.validate_site import ValidationReport
from tools.mkdocs_site.watch import watch_changes


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_status(url: str, expected: int, timeout: float = 10.0) -> str:
    deadline = time.monotonic() + timeout
    last_status = 0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                last_status = response.status
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            last_status = error.code
            body = error.read().decode("utf-8", errors="replace")
        except OSError:
            time.sleep(0.05)
            continue
        if last_status == expected:
            return body
        time.sleep(0.05)
    raise AssertionError(f"{url} stayed at HTTP {last_status}, expected {expected}")


def _fixture_operations() -> Operations:
    def stage(paths: BuildPaths, inventory: Inventory) -> StageResult:
        paths.cache.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".fixture-docs-", dir=paths.cache))
        for page in inventory.pages:
            destination = temporary / Path(page.relative.as_posix())
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(page.source, destination)
        if paths.staging.exists():
            shutil.rmtree(paths.staging)
        temporary.replace(paths.staging)
        manifest = paths.cache / "routes.json"
        manifest.write_text(json.dumps([]), encoding="utf-8")
        return StageResult(len(inventory.pages), 0, manifest)

    def config(
        paths: BuildPaths, inventory: Inventory, _stage: StageResult
    ) -> Path:
        paths.generated_config.write_text(
            yaml.safe_dump(
                {
                    "site_name": "Refresh Fixture",
                    "docs_dir": str(paths.staging),
                    "site_dir": str(paths.site),
                    "use_directory_urls": False,
                    "theme": {
                        "name": "material",
                        "features": [
                            "navigation.sections",
                            "navigation.indexes",
                            "navigation.prune",
                        ],
                    },
                    "nav": build_navigation(inventory),
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return paths.generated_config

    return Operations(
        scan_inventory,
        stage,
        config,
        lambda _command, _cwd: 0,
        lambda _site, _inventory: None,
        lambda _site, _manifest: ValidationReport(0, (), (), (), (), ()),
    )


def test_serve_refreshes_create_rename_delete_and_navigation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "fixture-repo"
    wiki = repo / "wiki"
    domain = wiki / "domain"
    domain.mkdir(parents=True)
    (wiki / "index.md").write_text("# Fixture Home\n", encoding="utf-8")
    (domain / "index.md").write_text("# Fixture Domain\n", encoding="utf-8")
    fake_cli = repo / "tools/mkdocs_site/cli.py"
    fake_cli.parent.mkdir(parents=True)
    monkeypatch.setattr(cli, "__file__", str(fake_cli))
    port = _free_port()
    origin = f"http://127.0.0.1:{port}"
    evidence: list[str] = []

    def drive_refreshes(inputs: tuple[Path, ...], callback, _stop_requested) -> None:
        assert inputs == (
            wiki,
            repo / "mkdocs.yml",
            repo / "tools/mkdocs-site/overrides",
            repo / "tools/mkdocs-site/assets",
        )
        created = domain / "20_created.md"
        created.write_text("# Created Page\n", encoding="utf-8")
        callback()
        created_html = _wait_status(f"{origin}/domain/20_created.html", 200)
        assert "20_created" in created_html
        evidence.append("create")

        renamed = domain / "21_renamed.md"
        created.rename(renamed)
        callback()
        _wait_status(f"{origin}/domain/20_created.html", 404)
        renamed_html = _wait_status(f"{origin}/domain/21_renamed.html", 200)
        assert "21_renamed" in renamed_html
        assert "20_created" not in renamed_html
        evidence.append("rename")

        renamed.unlink()
        callback()
        _wait_status(f"{origin}/domain/21_renamed.html", 404)
        domain_html = _wait_status(f"{origin}/domain/index.html", 200)
        assert "21_renamed" not in domain_html
        evidence.append("delete")

    runtime = replace(
        _default_serve_runtime(),
        watch=drive_refreshes,
        open_browser=lambda _url: pytest.fail("--no-open tried to open a browser"),
    )

    result = main(
        ["serve", "--port", str(port), "--no-open"],
        _fixture_operations(),
        runtime,
    )

    assert result == 0
    assert evidence == ["create", "rename", "delete"]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", port))


def test_serve_retries_after_candidate_and_recovery_readiness_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "fixture-repo"
    wiki = repo / "wiki"
    domain = wiki / "domain"
    domain.mkdir(parents=True)
    (wiki / "index.md").write_text("# Fixture Home\n", encoding="utf-8")
    (domain / "index.md").write_text("# Fixture Domain\n", encoding="utf-8")
    fake_cli = repo / "tools/mkdocs_site/cli.py"
    fake_cli.parent.mkdir(parents=True)
    monkeypatch.setattr(cli, "__file__", str(fake_cli))
    port = _free_port()
    origin = f"http://127.0.0.1:{port}/"
    base_operations = _fixture_operations()
    inventory_calls = 0
    children = []
    readiness_calls = 0
    sleep_calls = 0
    evidence: list[str] = []

    def inventory(root: Path) -> Inventory:
        nonlocal inventory_calls
        inventory_calls += 1
        if inventory_calls == 2:
            raise RuntimeError("candidate conversion failed")
        return base_operations.inventory(root)

    operations = replace(base_operations, inventory=inventory)

    def start(_command: list[str], cwd: Path):
        child = cli._start_mkdocs(
            [
                sys.executable,
                "-m",
                "http.server",
                str(port),
                "--bind",
                "127.0.0.1",
            ],
            cwd,
        )
        children.append(child)
        return child

    def wait_ready(url: str, child) -> None:
        nonlocal readiness_calls
        readiness_calls += 1
        cli._wait_for_http(url, child, timeout=10.0)
        if readiness_calls == 2:
            raise TimeoutError("recovery readiness failed")

    def watch(root: Path, callback, stop_requested) -> None:
        nonlocal sleep_calls

        def drive_change(_seconds: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls == 1:
                (domain / "20_first.md").write_text(
                    "# First failed refresh\n", encoding="utf-8"
                )
                evidence.append("first-change")
            elif sleep_calls == 3:
                assert all(child.poll() is not None for child in children)
                assert len(list((repo / ".mkdocs-cache").glob(".serve-backup-*"))) == 1
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                    listener.bind(("127.0.0.1", port))
                (domain / "21_second.md").write_text(
                    "# Second healthy refresh\n", encoding="utf-8"
                )
                evidence.append("watch-survived")
            elif sleep_calls == 5:
                assert children[-1].poll() is None
                assert not list((repo / ".mkdocs-cache").glob(".serve-backup-*"))
                evidence.append("healthy-retry")
                raise KeyboardInterrupt

        watch_changes(
            root,
            callback,
            stop_requested=stop_requested,
            poll_interval=0,
            stable_debounce=0,
            sleep=drive_change,
        )

    runtime = replace(
        _default_serve_runtime(),
        start=start,
        wait_ready=wait_ready,
        watch=watch,
        open_browser=lambda _url: pytest.fail("--no-open tried to open a browser"),
    )

    try:
        result = main(
            ["serve", "--port", str(port), "--no-open"],
            operations,
            runtime,
        )

        assert result == 130
        assert evidence == ["first-change", "watch-survived", "healthy-retry"]
        assert inventory_calls == 3
        assert readiness_calls == 3
        assert len(children) == 3
        assert all(child.poll() is not None for child in children)
        assert not list((repo / ".mkdocs-cache").glob(".serve-backup-*"))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", port))
    finally:
        for child in children:
            cli._stop_mkdocs(child)
