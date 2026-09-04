import sys
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.mkdocs_site import cli as cli_module
from tools.mkdocs_site.cli import (
    Operations,
    ServeRuntime,
    _ServeState,
    _refresh_preview,
    _start_mkdocs,
    _stop_mkdocs,
    _wait_for_http,
    main,
)
from tools.mkdocs_site.models import BuildPaths
from tools.mkdocs_site.staging import StageResult
from tools.mkdocs_site.validate_site import ValidationReport


def recording_operations(
    events: list[tuple[object, ...]],
    generated_config: Path,
    exit_code: int = 0,
    report: ValidationReport | None = None,
    rewrite_error: Exception | None = None,
) -> Operations:
    inventory = object()
    stage_result = SimpleNamespace(route_manifest=generated_config.with_name("routes.json"))

    def scan(wiki: Path) -> object:
        events.append(("inventory", wiki))
        return inventory

    def stage(paths: object, received_inventory: object) -> object:
        assert received_inventory is inventory
        events.append(("stage", paths))
        return stage_result

    def config(
        paths: object, received_inventory: object, received_stage_result: object
    ) -> Path:
        assert received_inventory is inventory
        assert received_stage_result is stage_result
        events.append(("config", paths))
        return generated_config

    def run(command: list[str], cwd: Path) -> int:
        events.append(("mkdocs", command, cwd))
        return exit_code

    def rewrite(site: Path, received_inventory: object) -> None:
        assert received_inventory is inventory
        events.append(("rewrite", site))
        if rewrite_error is not None:
            raise rewrite_error

    def validate(
        site: Path, route_manifest: Path, sources: set[str] | None = None
    ) -> ValidationReport:
        events.append(("validate", site, route_manifest, sources))
        return report or ValidationReport(3, (), (), (), (), ())

    return Operations(scan, stage, config, run, rewrite, validate)


def recording_serve_runtime(
    events: list[tuple[object, ...]],
    *,
    child_exit: int | None = None,
) -> ServeRuntime:
    child = SimpleNamespace(pid=1234, returncode=None)
    child.poll = lambda: child.returncode

    def check_port(port: int) -> None:
        events.append(("port", port))

    def start(command: list[str], cwd: Path) -> object:
        events.append(("start", command, cwd))
        return child

    def ready(url: str, received_child: object) -> None:
        assert received_child is child
        events.append(("ready", url))

    def open_browser(url: str) -> None:
        events.append(("open", url))

    def watch(wiki: Path, callback, stop_requested) -> None:
        del callback
        events.append(("watch", wiki))
        if child_exit is not None:
            assert child.poll() is None
            child.returncode = child_exit
            assert stop_requested()

    def stop(received_child: object) -> None:
        assert received_child is child
        events.append(("stop", child.pid))

    return ServeRuntime(check_port, start, ready, open_browser, watch, stop)


def test_build_stages_writes_config_then_runs_strict_mkdocs(tmp_path: Path) -> None:
    events: list[tuple[object, ...]] = []
    generated_config = tmp_path / "mkdocs.generated.yml"

    result = main(["build"], recording_operations(events, generated_config, 17))

    assert result == 17
    assert [event[0] for event in events] == [
        "inventory",
        "stage",
        "config",
        "mkdocs",
    ]
    command = events[-1][1]
    assert command == [
        sys.executable,
        "-m",
        "mkdocs",
        "build",
        "--strict",
        "-f",
        str(generated_config),
    ]
    assert events[-1][2] == events[1][1].repo


def test_serve_stages_writes_config_then_binds_mkdocs_to_loopback(
    tmp_path: Path,
) -> None:
    events: list[tuple[object, ...]] = []
    generated_config = tmp_path / "mkdocs.generated.yml"

    result = main(
        ["serve"],
        recording_operations(events, generated_config),
        recording_serve_runtime(events),
    )

    assert result == 0
    assert [event[0] for event in events] == [
        "port",
        "inventory",
        "stage",
        "config",
        "start",
        "ready",
        "open",
        "watch",
        "stop",
    ]
    assert events[0] == ("port", 8000)
    assert events[4][1] == [
        sys.executable,
        "-m",
        "mkdocs",
        "serve",
        "--dev-addr",
        "127.0.0.1:8000",
        "-f",
        str(generated_config),
    ]
    assert events[5] == ("ready", "http://127.0.0.1:8000/")
    assert events[6] == ("open", "http://127.0.0.1:8000/")
    repo = Path(cli_module.__file__).resolve().parents[2]
    assert events[7] == (
        "watch",
        (
            repo / "wiki",
            repo / "mkdocs.yml",
            repo / "tools/mkdocs-site/overrides",
            repo / "tools/mkdocs-site/assets",
        ),
    )


def test_serve_accepts_port_and_no_open(tmp_path: Path) -> None:
    events: list[tuple[object, ...]] = []
    generated_config = tmp_path / "mkdocs.generated.yml"

    result = main(
        ["serve", "--port", "8123", "--no-open"],
        recording_operations(events, generated_config),
        recording_serve_runtime(events),
    )

    assert result == 0
    assert events[0] == ("port", 8123)
    assert not any(event[0] == "open" for event in events)
    assert "127.0.0.1:8123" in events[4][1]


def test_serve_returns_an_unexpected_child_exit_code(tmp_path: Path) -> None:
    events: list[tuple[object, ...]] = []

    result = main(
        ["serve", "--no-open"],
        recording_operations(events, tmp_path / "generated.yml"),
        recording_serve_runtime(events, child_exit=19),
    )

    assert result == 19
    assert events[-1] == ("stop", 1234)


def test_serve_ctrl_c_returns_130_and_stops_child(tmp_path: Path) -> None:
    events: list[tuple[object, ...]] = []
    runtime = recording_serve_runtime(events)

    def interrupt(_wiki: Path, _callback, _stop_requested) -> None:
        raise KeyboardInterrupt

    runtime = ServeRuntime(
        runtime.check_port,
        runtime.start,
        runtime.wait_ready,
        runtime.open_browser,
        interrupt,
        runtime.stop,
    )

    result = main(
        ["serve", "--no-open"],
        recording_operations(events, tmp_path / "generated.yml"),
        runtime,
    )

    assert result == 130
    assert events[-1] == ("stop", 1234)


def test_serve_checks_port_before_scanning_or_staging(tmp_path: Path) -> None:
    events: list[tuple[object, ...]] = []
    runtime = recording_serve_runtime(events)

    def unavailable(port: int) -> None:
        events.append(("port", port))
        raise OSError("port occupied")

    runtime = ServeRuntime(
        unavailable,
        runtime.start,
        runtime.wait_ready,
        runtime.open_browser,
        runtime.watch,
        runtime.stop,
    )

    with pytest.raises(OSError, match="port occupied"):
        main(
            ["serve", "--port", "8123"],
            recording_operations(events, tmp_path / "generated.yml"),
            runtime,
        )

    assert events == [("port", 8123)]


def test_serve_opens_browser_only_for_initial_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.mkdocs_site import cli

    fake_cli = tmp_path / "repo/tools/mkdocs_site/cli.py"
    fake_cli.parent.mkdir(parents=True)
    (tmp_path / "repo/wiki").mkdir()
    monkeypatch.setattr(cli, "__file__", str(fake_cli))
    events: list[tuple[object, ...]] = []
    children = [
        SimpleNamespace(pid=1, returncode=None),
        SimpleNamespace(pid=2, returncode=None),
    ]
    for child in children:
        child.poll = lambda child=child: child.returncode

    def watch(_wiki: Path, callback, _stop_requested) -> None:
        callback()

    runtime = ServeRuntime(
        lambda port: events.append(("port", port)),
        lambda command, _cwd: events.append(("start", command)) or children.pop(0),
        lambda url, _child: events.append(("ready", url)),
        lambda url: events.append(("open", url)),
        watch,
        lambda child: events.append(("stop", child.pid)),
    )

    result = main(
        ["serve"],
        recording_operations(events, tmp_path / "generated.yml"),
        runtime,
    )

    assert result == 0
    assert [event[0] for event in events].count("open") == 1
    assert [event[0] for event in events].count("start") == 2
    assert [event[0] for event in events].count("ready") == 2


@pytest.mark.parametrize("failure_position", [1, 2, 3])
def test_refresh_stash_copy_failure_keeps_all_outputs_and_restarts_previous_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_position: int,
) -> None:
    paths = BuildPaths.from_repo(tmp_path / "repo")
    paths.wiki.mkdir(parents=True)
    paths.staging.mkdir(parents=True)
    (paths.staging / "old.md").write_text("old stage", encoding="utf-8")
    routes = paths.cache / "routes.json"
    routes.write_text("old routes", encoding="utf-8")
    paths.generated_config.write_text("old config", encoding="utf-8")
    old_child = SimpleNamespace(pid=1, returncode=None)
    old_child.poll = lambda: old_child.returncode
    recovery = SimpleNamespace(pid=2, returncode=None)
    recovery.poll = lambda: recovery.returncode
    state = _ServeState(old_child)
    events: list[tuple[object, ...]] = []
    copy_calls = 0
    real_copytree = cli_module.shutil.copytree
    real_copy2 = cli_module.shutil.copy2

    def fail_at_position(copy_function):
        def wrapped(source, destination, *args, **kwargs):
            nonlocal copy_calls
            copy_calls += 1
            if copy_calls == failure_position:
                raise OSError(f"backup copy {failure_position} failed")
            return copy_function(source, destination, *args, **kwargs)

        return wrapped

    monkeypatch.setattr(
        cli_module.shutil, "copytree", fail_at_position(real_copytree)
    )
    monkeypatch.setattr(cli_module.shutil, "copy2", fail_at_position(real_copy2))
    operations = Operations(
        lambda _wiki: (_ for _ in ()).throw(
            AssertionError("inventory must not run after backup failure")
        ),
        lambda _paths, _inventory: (_ for _ in ()).throw(AssertionError()),
        lambda _paths, _inventory, _stage: (_ for _ in ()).throw(AssertionError()),
        lambda _command, _cwd: 0,
        lambda _site, _inventory: None,
        lambda _site, _manifest, _sources=None: ValidationReport(0, (), (), (), (), ()),
    )
    runtime = ServeRuntime(
        lambda _port: None,
        lambda command, _cwd: events.append(("start", command)) or recovery,
        lambda url, child: events.append(("ready", url, child.pid)),
        lambda _url: None,
        lambda _wiki, _callback, _stop: None,
        lambda child: events.append(("stop", child.pid)),
    )

    with pytest.raises(
        OSError, match=f"backup copy {failure_position} failed"
    ):
        _refresh_preview(paths, operations, runtime, state, 8123)

    assert copy_calls == failure_position
    assert state.child is recovery
    assert (paths.staging / "old.md").read_text(encoding="utf-8") == "old stage"
    assert routes.read_text(encoding="utf-8") == "old routes"
    assert paths.generated_config.read_text(encoding="utf-8") == "old config"
    assert [event[0] for event in events] == ["stop", "start", "ready"]
    assert not list(paths.cache.glob(".serve-backup-*"))


def test_refresh_config_failure_restores_working_outputs_and_restarts(
    tmp_path: Path,
) -> None:
    paths = BuildPaths.from_repo(tmp_path / "repo")
    paths.wiki.mkdir(parents=True)
    paths.staging.mkdir(parents=True)
    (paths.staging / "old.md").write_text("old stage", encoding="utf-8")
    routes = paths.cache / "routes.json"
    routes.write_text("old routes", encoding="utf-8")
    paths.generated_config.write_text("old config", encoding="utf-8")
    old_child = SimpleNamespace(pid=1, returncode=None)
    old_child.poll = lambda: old_child.returncode
    recovery = SimpleNamespace(pid=2, returncode=None)
    recovery.poll = lambda: recovery.returncode
    state = _ServeState(old_child)
    events: list[tuple[object, ...]] = []

    def stage(_paths: BuildPaths, _inventory: object) -> StageResult:
        cli_module.shutil.rmtree(paths.staging)
        paths.staging.mkdir(parents=True)
        (paths.staging / "new.md").write_text("new stage", encoding="utf-8")
        routes.write_text("new routes", encoding="utf-8")
        return StageResult(1, 0, routes)

    def config(*_args: object) -> Path:
        paths.generated_config.write_text("partial config", encoding="utf-8")
        raise RuntimeError("invalid generated config")

    operations = Operations(
        lambda _wiki: object(),
        stage,
        config,
        lambda _command, _cwd: 0,
        lambda _site, _inventory: None,
        lambda _site, _manifest, _sources=None: ValidationReport(0, (), (), (), (), ()),
    )
    runtime = ServeRuntime(
        lambda _port: None,
        lambda command, _cwd: events.append(("start", command)) or recovery,
        lambda url, child: events.append(("ready", url, child.pid)),
        lambda _url: None,
        lambda _wiki, _callback, _stop: None,
        lambda child: events.append(("stop", child.pid)),
    )

    with pytest.raises(RuntimeError, match="invalid generated config"):
        _refresh_preview(paths, operations, runtime, state, 8123)

    assert state.child is recovery
    assert (paths.staging / "old.md").read_text(encoding="utf-8") == "old stage"
    assert not (paths.staging / "new.md").exists()
    assert routes.read_text(encoding="utf-8") == "old routes"
    assert paths.generated_config.read_text(encoding="utf-8") == "old config"
    assert [event[0] for event in events] == ["stop", "start", "ready"]
    assert not list(paths.cache.glob(".serve-backup-*"))


def test_refresh_stops_unready_recovery_and_keeps_backup_for_diagnostics(
    tmp_path: Path,
) -> None:
    paths = BuildPaths.from_repo(tmp_path / "repo")
    paths.wiki.mkdir(parents=True)
    paths.staging.mkdir(parents=True)
    (paths.staging / "old.md").write_text("old stage", encoding="utf-8")
    routes = paths.cache / "routes.json"
    routes.write_text("old routes", encoding="utf-8")
    paths.generated_config.write_text("old config", encoding="utf-8")
    old_child = SimpleNamespace(pid=1, returncode=None)
    old_child.poll = lambda: old_child.returncode
    state = _ServeState(old_child)
    recovery_children = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])

    def start_recovery(_command: list[str], cwd: Path):
        child = _start_mkdocs(
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
        recovery_children.append(child)
        return child

    def fail_readiness(url: str, child) -> None:
        _wait_for_http(url, child, timeout=10.0)
        raise TimeoutError("recovery readiness failed")

    operations = Operations(
        lambda _wiki: (_ for _ in ()).throw(RuntimeError("conversion failed")),
        lambda _paths, _inventory: (_ for _ in ()).throw(AssertionError()),
        lambda _paths, _inventory, _stage: (_ for _ in ()).throw(AssertionError()),
        lambda _command, _cwd: 0,
        lambda _site, _inventory: None,
        lambda _site, _manifest, _sources=None: ValidationReport(0, (), (), (), (), ()),
    )

    def stop(child) -> None:
        if child is not old_child:
            _stop_mkdocs(child)

    runtime = ServeRuntime(
        lambda _port: None,
        start_recovery,
        fail_readiness,
        lambda _url: None,
        lambda _wiki, _callback, _stop: None,
        stop,
    )

    try:
        with pytest.raises(
            RuntimeError, match="previous server could not restart"
        ):
            _refresh_preview(paths, operations, runtime, state, port)

        recovery = recovery_children[0]
        stopped = recovery.poll() is not None
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("127.0.0.1", port))
            port_rebound = True
        except OSError:
            port_rebound = False
        assert (stopped, port_rebound) == (True, True)
        assert state.child is None
        assert state.backup is not None
        assert state.backup.root.is_dir()
        assert (paths.staging / "old.md").read_text(encoding="utf-8") == "old stage"
        assert routes.read_text(encoding="utf-8") == "old routes"
        assert paths.generated_config.read_text(encoding="utf-8") == "old config"
        assert len(list(paths.cache.glob(".serve-backup-*"))) == 1
    finally:
        for child in recovery_children:
            _stop_mkdocs(child)


def test_refresh_success_activates_new_outputs_before_readiness(tmp_path: Path) -> None:
    paths = BuildPaths.from_repo(tmp_path / "repo")
    paths.wiki.mkdir(parents=True)
    paths.staging.mkdir(parents=True)
    (paths.staging / "old.md").write_text("old", encoding="utf-8")
    routes = paths.cache / "routes.json"
    routes.write_text("old", encoding="utf-8")
    paths.generated_config.write_text("old", encoding="utf-8")
    old_child = SimpleNamespace(pid=1, returncode=None)
    new_child = SimpleNamespace(pid=2, returncode=None)
    old_child.poll = lambda: old_child.returncode
    new_child.poll = lambda: new_child.returncode
    state = _ServeState(old_child)
    events: list[str] = []

    def stage(_paths: BuildPaths, _inventory: object) -> StageResult:
        events.append("stage")
        cli_module.shutil.rmtree(paths.staging)
        paths.staging.mkdir(parents=True)
        (paths.staging / "new.md").write_text("new", encoding="utf-8")
        routes.write_text("new", encoding="utf-8")
        return StageResult(1, 0, routes)

    def config(*_args: object) -> Path:
        events.append("config")
        paths.generated_config.write_text("new", encoding="utf-8")
        return paths.generated_config

    operations = Operations(
        lambda _wiki: events.append("inventory") or object(),
        stage,
        config,
        lambda _command, _cwd: 0,
        lambda _site, _inventory: None,
        lambda _site, _manifest, _sources=None: ValidationReport(0, (), (), (), (), ()),
    )
    runtime = ServeRuntime(
        lambda _port: None,
        lambda _command, _cwd: events.append("start") or new_child,
        lambda _url, _child: events.append("ready"),
        lambda _url: None,
        lambda _wiki, _callback, _stop: None,
        lambda _child: events.append("stop"),
    )

    _refresh_preview(paths, operations, runtime, state, 8123)

    assert state.child is new_child
    assert state.backup is None
    assert (paths.staging / "new.md").read_text(encoding="utf-8") == "new"
    assert not (paths.staging / "old.md").exists()
    assert routes.read_text(encoding="utf-8") == "new"
    assert paths.generated_config.read_text(encoding="utf-8") == "new"
    assert events == ["stop", "inventory", "stage", "config", "start", "ready"]
    assert not list(paths.cache.glob(".serve-backup-*"))


def test_stage_stops_after_inventory_and_staging() -> None:
    events: list[tuple[object, ...]] = []

    result = main(["stage"], recording_operations(events, Path("unused")))

    assert result == 0
    assert [event[0] for event in events] == ["inventory", "stage"]


def test_successful_build_rewrites_search_after_mkdocs() -> None:
    events: list[tuple[object, ...]] = []

    result = main(["build"], recording_operations(events, Path("generated.yml")))

    assert result == 0
    assert [event[0] for event in events] == [
        "inventory",
        "stage",
        "config",
        "mkdocs",
        "rewrite",
        "validate",
    ]
    assert events[-2][1] == events[1][1].site
    assert events[-1][1:] == (
        events[1][1].site,
        Path("generated.yml").with_name("routes.json"),
        None,
    )


def test_changed_build_hands_validate_a_scope_instead_of_none() -> None:
    """不带 --changed 时必须是 None（全量语义）；带上就得是集合，哪怕是空集。"""
    events: list[tuple[object, ...]] = []

    result = main(["build", "--changed"], recording_operations(events, Path("generated.yml")))

    assert result == 0
    assert events[-1][0] == "validate"
    assert isinstance(events[-1][3], set)


def test_rewrite_failure_prevents_validation() -> None:
    events: list[tuple[object, ...]] = []

    with pytest.raises(RuntimeError, match="rewrite failed"):
        main(
            ["build"],
            recording_operations(
                events,
                Path("generated.yml"),
                rewrite_error=RuntimeError("rewrite failed"),
            ),
        )

    assert [event[0] for event in events] == [
        "inventory",
        "stage",
        "config",
        "mkdocs",
        "rewrite",
    ]


def test_validate_checks_existing_output_without_staging() -> None:
    events: list[tuple[object, ...]] = []

    result = main(["validate"], recording_operations(events, Path("unused.yml")))

    assert result == 0
    assert [event[0] for event in events] == ["validate"]
    assert events[0][2].name == "routes.json"


def test_failed_build_validation_returns_nonzero_and_preserves_site(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.mkdocs_site import cli

    fake_cli = tmp_path / "repo/tools/mkdocs_site/cli.py"
    fake_cli.parent.mkdir(parents=True)
    monkeypatch.setattr(cli, "__file__", str(fake_cli))
    site = tmp_path / "repo/site"
    site.mkdir()
    marker = site / "keep.html"
    marker.write_text("inspect me", encoding="utf-8")
    events: list[tuple[object, ...]] = []
    report = ValidationReport(
        3,
        ("index.html -> missing.html",),
        (),
        (),
        (),
        (),
    )

    result = main(
        ["build"],
        recording_operations(events, tmp_path / "generated.yml", report=report),
    )

    assert result == 1
    assert marker.read_text(encoding="utf-8") == "inspect me"
    assert [event[0] for event in events][-1] == "validate"


# ---------- --changed 的作用域换算 ----------

def test_scope_from_changed_converts_to_manifest_spelling(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    (wiki / "domain").mkdir(parents=True)
    page = wiki / "domain" / "a.md"
    page.write_text("x", encoding="utf-8")

    scope = cli_module.scope_from_changed([page], wiki, {"domain/a.md", "other.md"})

    assert scope == {"domain/a.md"}


def test_scope_from_changed_drops_paths_outside_the_wiki(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    outside = tmp_path / "tools" / "notes.md"
    outside.parent.mkdir()
    outside.write_text("x", encoding="utf-8")

    assert cli_module.scope_from_changed([outside], wiki, {"notes.md"}) == set()


def test_scope_from_changed_drops_pages_that_own_no_route(tmp_path: Path) -> None:
    """--changed 是推导出来的，不该因为某个 wiki 文件没有路由就整体报错。"""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    unrouted = wiki / "draft.md"
    unrouted.write_text("x", encoding="utf-8")

    assert cli_module.scope_from_changed([unrouted], wiki, {"a.md"}) == set()
