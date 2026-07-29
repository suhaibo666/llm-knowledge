"""Shared CLI and evidence contract for the A-F volume teaching demos."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence


SCHEMA_VERSION = "torch-compile-volume-demo/v1"
VALID_STATUSES = {"PASS", "BLOCKED", "FAIL"}


class DemoArgumentParser(argparse.ArgumentParser):
    """Argparse variant with the demo contract's dedicated CLI-error code."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(4, f"{self.prog}: error: {message}\n")


@dataclass(frozen=True)
class CapabilitySnapshot:
    torch_available: bool
    cuda_available: bool
    cuda_device_count: int
    distributed_available: bool
    triton_available: bool
    native_compiler_available: bool
    linux: bool
    details: Mapping[str, object] = field(default_factory=dict)

    def supports(self, requirement: str) -> bool:
        checks = {
            "torch": self.torch_available,
            "cuda": self.cuda_available,
            "cuda_multi_gpu": (
                self.cuda_available and self.cuda_device_count >= 2
            ),
            "distributed": self.distributed_available,
            "triton": self.triton_available,
            "native_compiler": self.native_compiler_available,
            "linux": self.linux,
        }
        if requirement not in checks:
            raise ValueError(f"unknown capability requirement: {requirement}")
        return bool(checks[requirement])

    def to_dict(self) -> dict[str, object]:
        return {
            "torch_available": self.torch_available,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "distributed_available": self.distributed_available,
            "triton_available": self.triton_available,
            "native_compiler_available": self.native_compiler_available,
            "linux": self.linux,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class DemoContext:
    volume: str
    device: str
    output_dir: Path
    seed: int
    capabilities: CapabilitySnapshot


CaseBody = Callable[[DemoContext], Mapping[str, object]]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    title: str
    pages: tuple[str, ...]
    requirements: tuple[str, ...]
    run: CaseBody
    description: str = ""


@dataclass
class CaseResult:
    volume: str
    case_id: str
    title: str
    pages: tuple[str, ...]
    requirements: tuple[str, ...]
    status: str
    environment: Mapping[str, object]
    observations: Mapping[str, object] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    missing_requirements: tuple[str, ...] = ()
    error: Mapping[str, object] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid result status: {self.status}")
        return {
            "schema_version": SCHEMA_VERSION,
            "volume": self.volume,
            "case_id": self.case_id,
            "title": self.title,
            "status": self.status,
            "pages": list(self.pages),
            "requirements": list(self.requirements),
            "environment": dict(self.environment),
            "observations": dict(self.observations),
            "artifacts": list(self.artifacts),
            "limitations": list(self.limitations),
            "missing_requirements": list(self.missing_requirements),
            "error": dict(self.error),
            "duration_ms": round(self.duration_ms, 3),
        }


def detect_capabilities() -> CapabilitySnapshot:
    details: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch
    except Exception as error:  # pragma: no cover - PyTorch is required here.
        details["torch_import_error"] = f"{type(error).__name__}: {error}"
        return CapabilitySnapshot(
            torch_available=False,
            cuda_available=False,
            cuda_device_count=0,
            distributed_available=False,
            triton_available=False,
            native_compiler_available=False,
            linux=sys.platform.startswith("linux"),
            details=details,
        )

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    details.update(
        {
            "torch": torch.__version__,
            "torch_git": getattr(torch.version, "git_version", None),
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "cudnn": (
                torch.backends.cudnn.version()
                if cuda_available and torch.backends.cudnn.is_available()
                else None
            ),
        }
    )
    if cuda_available:
        details["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(cuda_device_count)
        ]

    compiler_candidates = {
        name: shutil.which(name) for name in ("cl", "c++", "g++", "clang++", "nvcc")
    }
    details["compiler_candidates"] = compiler_candidates
    return CapabilitySnapshot(
        torch_available=True,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        distributed_available=bool(
            hasattr(torch, "distributed") and torch.distributed.is_available()
        ),
        triton_available=importlib.util.find_spec("triton") is not None,
        native_compiler_available=any(compiler_candidates.values()),
        linux=sys.platform.startswith("linux"),
        details=details,
    )


def _effective_requirements(
    spec: CaseSpec, context: DemoContext
) -> tuple[str, ...]:
    requirements = list(spec.requirements)
    if context.device == "cuda" and "cuda" not in requirements:
        requirements.append("cuda")
    return tuple(requirements)


def _relative_artifacts(case_dir: Path) -> tuple[str, ...]:
    return tuple(
        path.relative_to(case_dir).as_posix()
        for path in sorted(case_dir.rglob("*"))
        if path.is_file() and path.name != "result.json"
    )


def _write_case_result(case_dir: Path, result: CaseResult) -> Path:
    target = case_dir / "result.json"
    target.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return target


def execute_case(spec: CaseSpec, context: DemoContext) -> CaseResult:
    started = time.perf_counter()
    case_dir = context.output_dir / spec.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    case_context = dataclasses.replace(context, output_dir=case_dir)
    effective_requirements = _effective_requirements(spec, context)
    missing = tuple(
        requirement
        for requirement in effective_requirements
        if not context.capabilities.supports(requirement)
    )
    environment = context.capabilities.to_dict()

    if missing:
        result = CaseResult(
            volume=context.volume,
            case_id=spec.case_id,
            title=spec.title,
            pages=spec.pages,
            requirements=effective_requirements,
            status="BLOCKED",
            environment=environment,
            missing_requirements=missing,
            limitations=(
                "Required runtime capability is unavailable; the case body was not executed.",
            ),
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        _write_case_result(case_dir, result)
        return result

    try:
        observations = spec.run(case_context)
        if not isinstance(observations, Mapping):
            raise TypeError(
                f"case {spec.case_id} must return a mapping, got "
                f"{type(observations).__name__}"
            )
        result = CaseResult(
            volume=context.volume,
            case_id=spec.case_id,
            title=spec.title,
            pages=spec.pages,
            requirements=effective_requirements,
            status="PASS",
            environment=environment,
            observations=dict(observations),
            artifacts=_relative_artifacts(case_dir),
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as error:
        result = CaseResult(
            volume=context.volume,
            case_id=spec.case_id,
            title=spec.title,
            pages=spec.pages,
            requirements=effective_requirements,
            status="FAIL",
            environment=environment,
            artifacts=_relative_artifacts(case_dir),
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(
                    traceback.format_exception(error)
                ).splitlines()[-20:],
            },
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    _write_case_result(case_dir, result)
    return result


def exit_code_for(statuses: Sequence[str]) -> int:
    unknown = set(statuses) - VALID_STATUSES
    if unknown:
        raise ValueError(f"unknown result statuses: {sorted(unknown)}")
    if "FAIL" in statuses:
        return 2
    if "BLOCKED" in statuses:
        return 3
    return 0


def write_run_summary(
    context: DemoContext, results: Sequence[CaseResult]
) -> Path:
    context.output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(result.status for result in results)
    document = {
        "schema_version": SCHEMA_VERSION,
        "volume": context.volume,
        "device": context.device,
        "seed": context.seed,
        "environment": context.capabilities.to_dict(),
        "status_counts": dict(sorted(counts.items())),
        "cases": [result.to_dict() for result in results],
    }
    target = context.output_dir / "summary.json"
    target.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _validate_cases(cases: Sequence[CaseSpec]) -> None:
    identifiers = [case.case_id for case in cases]
    duplicates = sorted(
        identifier
        for identifier, count in Counter(identifiers).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate case identifiers: {duplicates}")
    for case in cases:
        if not case.case_id or not case.pages:
            raise ValueError("every case requires an id and at least one page")
        for requirement in case.requirements:
            CapabilitySnapshot(
                True, False, 0, False, False, False, False
            ).supports(requirement)


def _list_cases(volume: str, cases: Sequence[CaseSpec], as_json: bool) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "volume": volume,
        "cases": [
            {
                "case_id": case.case_id,
                "title": case.title,
                "pages": list(case.pages),
                "requirements": list(case.requirements),
                "description": case.description,
            }
            for case in cases
        ],
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(f"Volume {volume} cases:")
    for case in cases:
        requirements = ",".join(case.requirements) or "device"
        print(
            f"- {case.case_id}: {case.title} "
            f"[pages={','.join(case.pages)}; requires={requirements}]"
        )


def run_volume_cli(
    volume: str,
    cases: Sequence[CaseSpec],
    argv: Sequence[str] | None = None,
) -> int:
    _validate_cases(cases)
    parser = DemoArgumentParser(
        description=f"torch.compile volume {volume} teaching demo"
    )
    parser.add_argument("--list", action="store_true", dest="list_cases")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "artifacts"
            / "volume_demos"
            / volume.lower()
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.list_cases:
        _list_cases(volume, cases, arguments.as_json)
        return 0

    requested = arguments.case_ids or ["all"]
    by_id = {case.case_id: case for case in cases}
    if "all" in requested and len(requested) != 1:
        parser.error("--case all cannot be combined with another --case")
    selected = list(cases) if requested == ["all"] else []
    if not selected:
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            parser.error(f"unknown case ids: {', '.join(unknown)}")
        selected = [by_id[case_id] for case_id in dict.fromkeys(requested)]

    os.environ.setdefault("PYTHONHASHSEED", str(arguments.seed))
    capabilities = detect_capabilities()
    if capabilities.torch_available:
        import torch

        torch.manual_seed(arguments.seed)
        if capabilities.cuda_available:
            torch.cuda.manual_seed_all(arguments.seed)

    context = DemoContext(
        volume=volume,
        device=arguments.device,
        output_dir=arguments.output_dir.resolve(),
        seed=arguments.seed,
        capabilities=capabilities,
    )
    results = [execute_case(case, context) for case in selected]
    summary = write_run_summary(context, results)
    for result in results:
        suffix = (
            f" missing={','.join(result.missing_requirements)}"
            if result.missing_requirements
            else ""
        )
        print(f"{result.case_id}={result.status}{suffix}")
    print(f"summary={summary}")
    return exit_code_for([result.status for result in results])
