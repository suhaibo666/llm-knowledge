import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.mkdocs_site.cli import Operations, main
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

    def validate(site: Path, route_manifest: Path) -> ValidationReport:
        events.append(("validate", site, route_manifest))
        return report or ValidationReport(3, (), (), (), (), ())

    return Operations(scan, stage, config, run, rewrite, validate)


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
    )


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
