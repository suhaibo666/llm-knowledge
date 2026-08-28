import sys
from pathlib import Path

from tools.mkdocs_site.cli import Operations, main


def recording_operations(
    events: list[tuple[object, ...]], generated_config: Path, exit_code: int = 0
) -> Operations:
    inventory = object()
    stage_result = object()

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

    return Operations(scan, stage, config, run)


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

    result = main(["serve"], recording_operations(events, generated_config))

    assert result == 0
    assert [event[0] for event in events] == [
        "inventory",
        "stage",
        "config",
        "mkdocs",
    ]
    assert events[-1][1] == [
        sys.executable,
        "-m",
        "mkdocs",
        "serve",
        "--dev-addr",
        "127.0.0.1:8000",
        "-f",
        str(generated_config),
    ]


def test_stage_stops_after_inventory_and_staging() -> None:
    events: list[tuple[object, ...]] = []

    result = main(["stage"], recording_operations(events, Path("unused")))

    assert result == 0
    assert [event[0] for event in events] == ["inventory", "stage"]
