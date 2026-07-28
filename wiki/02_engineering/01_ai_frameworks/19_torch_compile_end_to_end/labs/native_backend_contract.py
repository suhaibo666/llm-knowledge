"""Capability probe and acceptance validator for native backend evidence.

The local probe records what the current host can attempt.  It deliberately
does not turn capability discovery, code generation, or a blocked run into an
acceptance result.  A PASS requires a separately produced, complete native
execution evidence bundle that satisfies this module and its JSON Schema.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import json
import math
import os
import pickle
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator


SCHEMA_VERSION = "native-backend-evidence/v1"
SCHEMA_PATH = Path(__file__).with_name("native_backend_environment.schema.json")
_STATUS_NOT_ACCEPTED = {"BLOCKED", "FAIL", "NOT_RUN"}


def _compiler_probe() -> dict[str, Any]:
    candidates = (
        ("cl.exe", [])
        if platform.system() == "Windows"
        else (("c++", ["--version"]), ("g++", ["--version"]), ("clang++", ["--version"]))
    )
    if platform.system() == "Windows":
        candidate_items = [candidates]
    else:
        candidate_items = list(candidates)

    for executable, version_args in candidate_items:
        compiler_path = shutil.which(executable)
        if not compiler_path:
            continue
        try:
            command = [compiler_path, *version_args]
            working_directory = None
            temporary_probe: tempfile.TemporaryDirectory[str] | None = None
            if executable == "cl.exe":
                temporary_probe = tempfile.TemporaryDirectory()
                probe_root = Path(temporary_probe.name)
                probe_source = probe_root / "compiler_probe.cpp"
                probe_object = probe_root / "compiler_probe.obj"
                probe_source.write_text("int main() { return 0; }\n", encoding="utf-8")
                command = [
                    compiler_path,
                    "/c",
                    str(probe_source),
                    f"/Fo{probe_object}",
                ]
                working_directory = probe_root
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
                cwd=working_directory,
            )
            if temporary_probe is not None:
                temporary_probe.cleanup()
            output = "\n".join(
                part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
            )
            first_line = next(
                (line.strip() for line in output.splitlines() if line.strip()),
                f"{executable}: version output unavailable",
            )
            if completed.returncode != 0:
                return {
                    "available": False,
                    "compiler_path": str(Path(compiler_path).resolve()),
                    "compiler_version": first_line,
                    "detail": (
                        f"{executable} version probe failed with return code "
                        f"{completed.returncode}"
                    ),
                }
            return {
                "available": True,
                "compiler_path": str(Path(compiler_path).resolve()),
                "compiler_version": first_line,
                "detail": f"{executable} was found on PATH",
            }
        except (OSError, subprocess.SubprocessError) as error:
            return {
                "available": False,
                "compiler_path": str(Path(compiler_path).resolve()),
                "compiler_version": None,
                "detail": f"{executable} was found but its version probe failed: {error}",
            }

    expected = "cl.exe" if platform.system() == "Windows" else "c++, g++, or clang++"
    return {
        "available": False,
        "compiler_path": None,
        "compiler_version": None,
        "detail": f"no supported native C++ compiler ({expected}) was found on PATH",
    }


def _torch_and_cuda_probe() -> tuple[dict[str, Any], dict[str, Any]]:
    runtime: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "torch_version": None,
        "torch_git_version": None,
    }
    cuda = {
        "available": False,
        "device_count": 0,
        "detail": "PyTorch could not be imported",
    }
    try:
        import torch
    except (ImportError, OSError) as error:
        cuda["detail"] = f"PyTorch import failed: {error}"
        return runtime, cuda

    runtime["torch_version"] = str(torch.__version__)
    runtime["torch_git_version"] = getattr(torch.version, "git_version", None)
    try:
        available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if available else 0
        detail = (
            f"{device_count} CUDA device(s) available; "
            f"torch.version.cuda={torch.version.cuda}"
            if available
            else f"torch.cuda.is_available() is false; torch.version.cuda={torch.version.cuda}"
        )
        cuda = {
            "available": available,
            "device_count": device_count,
            "detail": detail,
        }
    except (AttributeError, RuntimeError) as error:
        cuda = {
            "available": False,
            "device_count": 0,
            "detail": f"CUDA capability probe failed: {error}",
        }
    return runtime, cuda


def _triton_probe() -> dict[str, Any]:
    try:
        specification = importlib.util.find_spec("triton")
    except (ImportError, ValueError) as error:
        return {
            "available": False,
            "version": None,
            "detail": f"Triton discovery failed: {error}",
        }
    if specification is None:
        return {
            "available": False,
            "version": None,
            "detail": "the triton Python package is not installed",
        }
    try:
        triton = importlib.import_module("triton")
    except Exception as error:
        return {
            "available": False,
            "version": None,
            "detail": f"Triton import failed: {error}",
        }
    version = getattr(triton, "__version__", None)
    if not version:
        try:
            version = metadata.version("triton")
        except metadata.PackageNotFoundError:
            return {
                "available": False,
                "version": None,
                "detail": "Triton imported but its version could not be determined",
            }
    return {
        "available": True,
        "version": version,
        "detail": "the triton Python package is importable",
    }


def probe_local_capabilities() -> dict[str, Any]:
    """Return an honest capability diagnostic; never return a PASS status."""

    cpu_cpp = _compiler_probe()
    runtime, cuda = _torch_and_cuda_probe()
    triton = _triton_probe()
    cpu_status = "NOT_RUN" if cpu_cpp["available"] else "BLOCKED"
    cuda_status = (
        "NOT_RUN" if cuda["available"] and triton["available"] else "BLOCKED"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "local_capability_diagnostic",
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "host": {
            "hostname": socket.gethostname(),
            "os": platform.system() or "unknown",
            "os_version": platform.version() or "unknown",
            "architecture": platform.machine() or "unknown",
        },
        "runtime": runtime,
        "capabilities": {
            "cpu_cpp": cpu_cpp,
            "cuda": cuda,
            "triton": triton,
        },
        "statuses": {
            "cpu_native": cpu_status,
            "cuda_triton": cuda_status,
        },
    }


def _format_schema_error(error: Any) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    return f"schema {path or '<root>'}: {error.message}"


def _timing_errors(timing: Any, iterations: Any, path: str) -> list[str]:
    if not isinstance(timing, dict):
        return []
    samples = timing.get("samples")
    if not isinstance(samples, list) or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in samples
    ):
        return []
    errors: list[str] = []
    if isinstance(iterations, int) and len(samples) != iterations:
        errors.append(
            f"{path}.iterations={iterations} but timings_ms.samples has {len(samples)} values"
        )
    if not samples:
        return errors
    non_finite = [
        index for index, value in enumerate(samples) if not math.isfinite(float(value))
    ]
    if non_finite:
        errors.append(
            f"{path}.timings_ms.samples must contain only finite values; "
            f"invalid indices={non_finite}"
        )
        return errors

    expected = {
        "min": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "p95": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
        "max": max(samples),
    }
    for key, value in expected.items():
        recorded = timing.get(key)
        if isinstance(recorded, (int, float)) and not isinstance(recorded, bool):
            if not math.isfinite(float(recorded)):
                errors.append(f"{path}.timings_ms.{key} must be finite")
                continue
            if math.isclose(
                float(recorded), float(value), rel_tol=1e-9, abs_tol=1e-12
            ):
                continue
            errors.append(
                f"{path}.timings_ms.{key}={recorded} does not match samples ({value})"
            )
    return errors


def _numerical_errors(comparison: Any, path: str) -> list[str]:
    if not isinstance(comparison, dict):
        return []
    errors: list[str] = []
    numeric_keys = (
        "rtol",
        "atol",
        "max_abs_error",
        "max_rel_error",
        "reference_max_abs",
    )
    values: dict[str, float] = {}
    for key in numeric_keys:
        value = comparison.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if not math.isfinite(numeric):
                errors.append(f"{path}.{key} must be finite")
            else:
                values[key] = numeric
    if all(key in values for key in ("rtol", "atol", "max_abs_error", "reference_max_abs")):
        allowed = values["atol"] + values["rtol"] * values["reference_max_abs"]
        if values["max_abs_error"] > allowed + 1e-15:
            errors.append(
                f"{path} numerical PASS is inconsistent: max_abs_error "
                f"{values['max_abs_error']} exceeds atol + rtol * reference_max_abs "
                f"({allowed})"
            )
    if comparison.get("allclose") is not True:
        errors.append(f"{path}.allclose must be true for numerical PASS")
    return errors


def _tensor_summary_errors(summary: Any, path: str) -> list[str]:
    if not isinstance(summary, dict):
        return []
    shape = summary.get("shape")
    numel = summary.get("numel")
    if (
        isinstance(shape, list)
        and all(isinstance(value, int) and value >= 0 for value in shape)
        and isinstance(numel, int)
    ):
        expected = math.prod(shape)
        if expected != numel:
            return [f"{path}.numel={numel} does not match shape product {expected}"]
    return []


def _producer_errors(document: dict[str, Any]) -> list[str]:
    producer = document.get("producer")
    if not isinstance(producer, dict):
        return ["native acceptance result requires producer provenance"]
    errors: list[str] = []
    if producer.get("id") != Path(__file__).name:
        errors.append("producer.id must identify native_backend_contract.py")
    if producer.get("contract_version") != SCHEMA_VERSION:
        errors.append("producer.contract_version must match schema_version")
    expected_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if producer.get("module_sha256") != expected_hash:
        errors.append("producer.module_sha256 does not match the validating producer")
    command = producer.get("command")
    target = producer.get("target")
    if not isinstance(command, list) or len(command) < 9:
        errors.append("producer.command must record the complete produce invocation")
    elif command[2:5] != ["produce", "--target", target]:
        errors.append(
            "producer.command must bind the produce subcommand to producer.target"
        )
    else:
        for option in ("--artifact-root", "--output"):
            if option not in command or command.index(option) == len(command) - 1:
                errors.append(f"producer.command must record {option}")
    return errors


def _semantic_errors(document: Any) -> list[str]:
    if not isinstance(document, dict):
        return []
    errors: list[str] = []
    artifact_type = document.get("artifact_type")
    statuses = document.get("statuses")
    if artifact_type == "local_capability_diagnostic" and isinstance(statuses, dict):
        if "acceptance_results" in document:
            errors.append("local capability diagnostic cannot contain acceptance_results")
        if "producer" in document:
            errors.append("local capability diagnostic cannot contain producer provenance")
        for target, status in statuses.items():
            if status == "PASS":
                errors.append(
                    f"diagnostic artifact cannot claim PASS for statuses.{target}"
                )

    if artifact_type == "native_acceptance_result":
        errors.extend(_producer_errors(document))

    results = document.get("acceptance_results")
    if isinstance(statuses, dict):
        for status_key in ("cpu_native", "cuda_triton"):
            has_result = isinstance(results, dict) and status_key in results
            is_pass = statuses.get(status_key) == "PASS"
            if has_result != is_pass:
                errors.append(
                    f"acceptance_results.{status_key} presence must match "
                    f"statuses.{status_key} PASS state"
                )
    if not isinstance(results, dict):
        return errors
    host = document.get("host") if isinstance(document.get("host"), dict) else {}
    runtime = (
        document.get("runtime") if isinstance(document.get("runtime"), dict) else {}
    )
    capabilities = (
        document.get("capabilities")
        if isinstance(document.get("capabilities"), dict)
        else {}
    )

    def require_equal(left: Any, right: Any, label: str) -> None:
        if left is not None and right is not None and left != right:
            errors.append(f"{label} values disagree: {left!r} != {right!r}")

    cpu = results.get("cpu_native")
    if isinstance(cpu, dict):
        producer = document.get("producer")
        if isinstance(producer, dict) and producer.get("target") != "cpu":
            errors.append("CPU result requires producer.target='cpu'")
        compiler = cpu.get("compiler") if isinstance(cpu.get("compiler"), dict) else {}
        cpu_capability = (
            capabilities.get("cpu_cpp")
            if isinstance(capabilities.get("cpu_cpp"), dict)
            else {}
        )
        require_equal(
            compiler.get("path"),
            cpu_capability.get("compiler_path"),
            "CPU compiler path provenance",
        )
        require_equal(
            compiler.get("version"),
            cpu_capability.get("compiler_version"),
            "CPU compiler version provenance",
        )
        compile_record = (
            cpu.get("compile") if isinstance(cpu.get("compile"), dict) else {}
        )
        command = compile_record.get("command")
        compiler_path = compiler.get("path")
        source_argument = compile_record.get("source_argument")
        output_argument = compile_record.get("output_argument")
        if isinstance(command, list):
            if not command or command[0] != compiler_path:
                errors.append(
                    "CPU compile.command[0] must equal the recorded compiler path"
                )
            if source_argument not in command:
                errors.append(
                    "CPU compile.source_argument must appear verbatim in compile.command"
                )
            if not any(
                isinstance(argument, str)
                and isinstance(output_argument, str)
                and (
                    argument == output_argument or argument.endswith(output_argument)
                )
                for argument in command
            ):
                errors.append(
                    "CPU compile.output_argument must be bound to compile.command"
                )
        generated_source = (
            cpu.get("generated_source")
            if isinstance(cpu.get("generated_source"), dict)
            else {}
        )
        if generated_source.get("language") != "c++":
            errors.append("CPU generated_source.language must be 'c++'")
        if isinstance(source_argument, str):
            source_path = generated_source.get("path")
            if isinstance(source_path, str) and not source_argument.replace(
                "\\", "/"
            ).endswith(source_path):
                errors.append(
                    "CPU compile.source_argument does not name generated_source.path"
                )
        output_artifact = (
            compile_record.get("output_artifact")
            if isinstance(compile_record.get("output_artifact"), dict)
            else {}
        )
        if isinstance(output_argument, str):
            output_path = output_artifact.get("path")
            if isinstance(output_path, str) and not output_argument.replace(
                "\\", "/"
            ).endswith(output_path):
                errors.append(
                    "CPU compile.output_argument does not name output_artifact.path"
                )
        load = cpu.get("load") if isinstance(cpu.get("load"), dict) else {}
        if load.get("artifact_path") != output_artifact.get("path"):
            errors.append("CPU load.artifact_path must equal compiled output path")
        errors.extend(
            _numerical_errors(
                cpu.get("numerical_comparison"),
                "acceptance_results.cpu_native.numerical_comparison",
            )
        )
        environment = (
            cpu.get("environment") if isinstance(cpu.get("environment"), dict) else {}
        )
        for result_key, top_value, label in (
            ("os", host.get("os"), "CPU environment OS provenance"),
            (
                "architecture",
                host.get("architecture"),
                "CPU environment architecture provenance",
            ),
            (
                "python_version",
                runtime.get("python_version"),
                "CPU environment Python provenance",
            ),
            (
                "torch_version",
                runtime.get("torch_version"),
                "CPU environment PyTorch provenance",
            ),
            (
                "torch_git_version",
                runtime.get("torch_git_version"),
                "CPU environment PyTorch git provenance",
            ),
        ):
            require_equal(environment.get(result_key), top_value, label)
        workloads = cpu.get("workloads")
        if isinstance(workloads, dict):
            required_calls = 0
            for workload_name in ("pointwise", "reduction"):
                workload = workloads.get(workload_name)
                if isinstance(workload, dict):
                    warmup = workload.get("warmup")
                    iterations = workload.get("iterations")
                    if isinstance(warmup, int) and isinstance(iterations, int):
                        required_calls += warmup + iterations
                    entry_point = workload.get("entry_point")
                    entry_points = load.get("entry_points")
                    if (
                        isinstance(entry_points, list)
                        and entry_point not in entry_points
                    ):
                        errors.append(
                            "acceptance_results.cpu_native.workloads."
                            f"{workload_name}.entry_point was not loaded"
                        )
                    if workload.get("kind") != workload_name:
                        errors.append(
                            "acceptance_results.cpu_native.workloads."
                            f"{workload_name}.kind must equal {workload_name!r}"
                        )
                    errors.extend(
                        _timing_errors(
                            workload.get("timings_ms"),
                            workload.get("iterations"),
                            f"acceptance_results.cpu_native.workloads.{workload_name}",
                        )
                    )
                    errors.extend(
                        _numerical_errors(
                            workload.get("numerical_comparison"),
                            "acceptance_results.cpu_native.workloads."
                            f"{workload_name}.numerical_comparison",
                        )
                    )
                    for summary_key in (
                        "input_summary",
                        "eager_output_summary",
                        "native_output_summary",
                    ):
                        errors.extend(
                            _tensor_summary_errors(
                                workload.get(summary_key),
                                "acceptance_results.cpu_native.workloads."
                                f"{workload_name}.{summary_key}",
                            )
                        )
            execute = (
                cpu.get("execute") if isinstance(cpu.get("execute"), dict) else {}
            )
            calls = execute.get("calls")
            if isinstance(calls, int) and calls < required_calls:
                errors.append(
                    "CPU execute.calls is smaller than the recorded warmup and "
                    "timed workload calls"
                )

    cuda = results.get("cuda_triton")
    if isinstance(cuda, dict):
        producer = document.get("producer")
        if isinstance(producer, dict) and producer.get("target") != "cuda":
            errors.append("CUDA result requires producer.target='cuda'")
        device = cuda.get("device") if isinstance(cuda.get("device"), dict) else {}
        environment = (
            cuda.get("environment")
            if isinstance(cuda.get("environment"), dict)
            else {}
        )
        triton_capability = (
            capabilities.get("triton")
            if isinstance(capabilities.get("triton"), dict)
            else {}
        )
        for result_key, top_value, label in (
            ("os", host.get("os"), "CUDA environment OS provenance"),
            (
                "architecture",
                host.get("architecture"),
                "CUDA environment architecture provenance",
            ),
            (
                "python_version",
                runtime.get("python_version"),
                "CUDA environment Python provenance",
            ),
            (
                "torch_version",
                runtime.get("torch_version"),
                "CUDA environment PyTorch provenance",
            ),
            (
                "torch_git_version",
                runtime.get("torch_git_version"),
                "CUDA environment PyTorch git provenance",
            ),
            (
                "triton_version",
                triton_capability.get("version"),
                "CUDA environment Triton capability provenance",
            ),
        ):
            require_equal(environment.get(result_key), top_value, label)
        for version_key, label in (
            ("driver_version", "CUDA driver provenance"),
            ("cuda_runtime_version", "CUDA runtime provenance"),
            ("triton_version", "CUDA Triton provenance"),
        ):
            require_equal(device.get(version_key), environment.get(version_key), label)
        device_index = device.get("index")
        cuda_capability = (
            capabilities.get("cuda")
            if isinstance(capabilities.get("cuda"), dict)
            else {}
        )
        device_count = cuda_capability.get("device_count")
        if (
            isinstance(device_index, int)
            and isinstance(device_count, int)
            and not 0 <= device_index < device_count
        ):
            errors.append(
                "CUDA device.index must satisfy 0 <= index < capabilities.cuda.device_count"
            )
        errors.extend(
            _numerical_errors(
                cuda.get("numerical_comparison"),
                "acceptance_results.cuda_triton.numerical_comparison",
            )
        )
        generated_source = (
            cuda.get("generated_source")
            if isinstance(cuda.get("generated_source"), dict)
            else {}
        )
        if generated_source.get("language") != "triton":
            errors.append("CUDA generated_source.language must be 'triton'")
        for summary_key in (
            "input_summary",
            "eager_output_summary",
            "native_output_summary",
        ):
            errors.extend(
                _tensor_summary_errors(
                    cuda.get(summary_key),
                    f"acceptance_results.cuda_triton.{summary_key}",
                )
            )
        benchmark = cuda.get("benchmark")
        if isinstance(benchmark, dict):
            iterations = benchmark.get("iterations")
            candidates = benchmark.get("candidates")
            candidate_medians: dict[str, float] = {}
            candidate_parameters: dict[str, int] = {}
            if isinstance(candidates, list):
                for index, candidate in enumerate(candidates):
                    if not isinstance(candidate, dict):
                        continue
                    errors.extend(
                        _timing_errors(
                            candidate.get("timings_ms"),
                            iterations,
                            f"acceptance_results.cuda_triton.benchmark.candidates[{index}]",
                        )
                    )
                    name = candidate.get("name")
                    parameters = candidate.get("parameters")
                    if isinstance(parameters, dict):
                        fingerprint = json.dumps(
                            parameters, sort_keys=True, separators=(",", ":")
                        )
                        if fingerprint in candidate_parameters:
                            errors.append(
                                "acceptance_results.cuda_triton.benchmark candidate "
                                "parameters must be unique"
                            )
                        candidate_parameters[fingerprint] = index
                    timing = candidate.get("timings_ms")
                    if (
                        isinstance(name, str)
                        and isinstance(timing, dict)
                        and isinstance(timing.get("median"), (int, float))
                    ):
                        if name in candidate_medians:
                            errors.append(
                                "acceptance_results.cuda_triton.benchmark candidate "
                                f"name {name!r} is duplicated"
                            )
                        candidate_medians[name] = float(timing["median"])

            winner = benchmark.get("winner")
            if isinstance(winner, dict) and candidate_medians:
                winner_name = winner.get("candidate_name")
                if winner_name not in candidate_medians:
                    errors.append(
                        "acceptance_results.cuda_triton.benchmark.winner does not "
                        "name a measured candidate"
                    )
                else:
                    recorded = winner.get("timing_ms")
                    expected = candidate_medians[winner_name]
                    if isinstance(recorded, (int, float)) and not math.isclose(
                        float(recorded), expected, rel_tol=1e-9, abs_tol=1e-12
                    ):
                        errors.append(
                            "acceptance_results.cuda_triton.benchmark.winner timing "
                            "does not match the candidate median"
                        )
                    fastest = min(candidate_medians, key=candidate_medians.get)
                    if winner_name != fastest:
                        errors.append(
                            "acceptance_results.cuda_triton.benchmark.winner is not "
                            "the lowest measured median"
                        )
            execution = (
                cuda.get("execution")
                if isinstance(cuda.get("execution"), dict)
                else {}
            )
            calls = execution.get("calls")
            if (
                isinstance(calls, int)
                and isinstance(iterations, int)
                and isinstance(benchmark.get("warmup"), int)
                and isinstance(candidates, list)
            ):
                required_calls = len(candidates) * (
                    iterations + benchmark["warmup"]
                ) + 1
                if calls < required_calls:
                    errors.append(
                        "CUDA execution.calls is smaller than candidate warmup, "
                        "timing, and winner execution calls"
                    )

        allocator = cuda.get("allocator")
        if isinstance(allocator, dict):
            allocated = allocator.get("max_memory_allocated_bytes")
            reserved = allocator.get("max_memory_reserved_bytes")
            if (
                isinstance(allocated, int)
                and isinstance(reserved, int)
                and reserved < allocated
            ):
                errors.append(
                    "acceptance_results.cuda_triton.allocator max reserved bytes "
                    "cannot be less than max allocated bytes"
                )
    return errors


def validate_document(document: Any) -> list[str]:
    """Validate structure and cross-field measurement semantics."""

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    schema_errors = sorted(
        validator.iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    return [_format_schema_error(error) for error in schema_errors] + _semantic_errors(
        document
    )


def _artifact_references(document: dict[str, Any], target: str) -> list[dict[str, str]]:
    results = document.get("acceptance_results", {})
    if target == "cpu":
        cpu = results.get("cpu_native", {})
        compile_record = cpu.get("compile", {})
        return [
            cpu.get("generated_source", {}),
            compile_record.get("stdout_log", {}),
            compile_record.get("stderr_log", {}),
            compile_record.get("output_artifact", {}),
        ]
    cuda = results.get("cuda_triton", {})
    allocator = cuda.get("allocator", {})
    return [
        cuda.get("generated_source", {}),
        cuda.get("cache_artifact", {}),
        allocator.get("memory_snapshot", {}),
        allocator.get("memory_trace", {}),
    ]


def _verify_artifact_reference(
    reference: Any, artifact_root: Path, label: str
) -> list[str]:
    if not isinstance(reference, dict):
        return [f"{label} must be an artifact reference object"]
    relative = reference.get("path")
    expected_hash = reference.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        return []
    if "\\" in relative:
        return [f"{label}.path must use forward slashes inside the evidence bundle"]
    normalized = PurePosixPath(relative)
    if normalized.is_absolute() or ".." in normalized.parts:
        return [f"{label}.path must be a safe bundle-relative path"]
    try:
        resolved_root = artifact_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        return [f"artifact root cannot be resolved: {error}"]
    try:
        path = resolved_root.joinpath(*normalized.parts)
        resolved_path = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        return [f"{label}.path cannot be resolved under artifact root: {error}"]
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return [f"{label}.path resolves outside the artifact root: {relative}"]
    if not resolved_path.is_file():
        return [f"{label}.path does not exist under artifact root: {relative}"]
    digest = hashlib.sha256()
    with resolved_path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_hash = digest.hexdigest()
    if actual_hash != expected_hash:
        return [
            f"{label}.sha256 mismatch: expected {expected_hash}, observed {actual_hash}"
        ]
    return []


def validate_acceptance(
    document: Any,
    target: str,
    artifact_root: Path | str | None = None,
    *,
    verify_files: bool = True,
) -> list[str]:
    """Validate that *target* is real PASS evidence, not a diagnostic or skip."""

    if target not in {"cpu", "cuda"}:
        raise ValueError("target must be 'cpu' or 'cuda'")
    errors = validate_document(document)
    if not isinstance(document, dict):
        return errors
    if document.get("artifact_type") != "native_acceptance_result":
        errors.append(
            "local capability diagnostic is not native acceptance evidence"
        )
    status_key = "cpu_native" if target == "cpu" else "cuda_triton"
    statuses = document.get("statuses")
    status = statuses.get(status_key) if isinstance(statuses, dict) else None
    if status != "PASS":
        errors.append(
            f"statuses.{status_key} must be PASS for {target} acceptance; got {status!r}"
        )
    results = document.get("acceptance_results")
    result = results.get(status_key) if isinstance(results, dict) else None
    if not isinstance(result, dict) or result.get("status") != "PASS":
        errors.append(
            f"acceptance_results.{status_key}.status must be PASS for acceptance"
        )
    references: list[dict[str, str]] = []
    if isinstance(result, dict):
        references = _artifact_references(document, target)
        paths = [
            reference.get("path")
            for reference in references
            if isinstance(reference, dict)
        ]
        if len(paths) != len(set(paths)):
            errors.append(
                f"acceptance_results.{status_key} artifact roles must use distinct paths"
            )
        if target == "cuda":
            hashes = [
                reference.get("sha256")
                for reference in references
                if isinstance(reference, dict)
            ]
            if len(hashes) != len(set(hashes)):
                errors.append(
                    "acceptance_results.cuda_triton artifact roles must have "
                    "distinct content hashes"
                )
    if errors:
        return errors
    if verify_files and isinstance(result, dict):
        root = Path.cwd() if artifact_root is None else Path(artifact_root)
        for index, reference in enumerate(references):
            errors.extend(
                _verify_artifact_reference(
                    reference,
                    root,
                    f"acceptance_results.{status_key}.artifact[{index}]",
                )
            )
    return errors


def write_document(document: dict[str, Any], output: Path | str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_reference(path: Path, artifact_root: Path) -> dict[str, str]:
    relative = path.resolve(strict=True).relative_to(
        artifact_root.resolve(strict=True)
    )
    return {"path": relative.as_posix(), "sha256": _sha256_file(path)}


def _timing_stats(samples: list[float]) -> dict[str, Any]:
    if not samples or not all(math.isfinite(value) and value > 0 for value in samples):
        raise RuntimeError("native timing samples must be finite and positive")
    ordered = sorted(samples)
    return {
        "samples": samples,
        "min": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max": max(samples),
    }


def _tensor_summary(tensor: Any) -> dict[str, Any]:
    contiguous = tensor.detach().cpu().contiguous()
    payload = contiguous.numpy().tobytes()
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "numel": int(contiguous.numel()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _comparison_record(eager: Any, native: Any) -> dict[str, Any]:
    import torch

    eager64 = eager.detach().cpu().to(torch.float64)
    native64 = native.detach().cpu().to(torch.float64)
    difference = (native64 - eager64).abs()
    reference_abs = eager64.abs()
    rtol = 1e-5
    atol = 1e-6
    allclose = bool(torch.allclose(native64, eager64, rtol=rtol, atol=atol))
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    denominator = reference_abs.clamp_min(torch.finfo(torch.float64).tiny)
    max_rel = (
        float((difference / denominator).max().item())
        if difference.numel()
        else 0.0
    )
    reference_max = (
        float(reference_abs.max().item()) if reference_abs.numel() else 0.0
    )
    values = (max_abs, max_rel, reference_max)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("numerical comparison produced a non-finite metric")
    if not allclose:
        raise RuntimeError(
            f"native output differs from eager: max_abs={max_abs}, max_rel={max_rel}"
        )
    return {
        "status": "PASS",
        "reference": "torch.eager",
        "allclose": True,
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "reference_max_abs": reference_max,
    }


def _producer_record(target: str, artifact_root: Path, output: Path) -> dict[str, Any]:
    return {
        "id": Path(__file__).name,
        "contract_version": SCHEMA_VERSION,
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": [
            str(Path(sys.executable).resolve()),
            str(Path(__file__).resolve()),
            "produce",
            "--target",
            target,
            "--artifact-root",
            str(artifact_root.resolve()),
            "--output",
            str(output.resolve()),
        ],
        "target": target,
    }


def _acceptance_base(
    diagnostic: dict[str, Any],
    target: str,
    artifact_root: Path,
    output: Path,
) -> dict[str, Any]:
    status_key = "cpu_native" if target == "cpu" else "cuda_triton"
    statuses = dict(diagnostic["statuses"])
    statuses[status_key] = "PASS"
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "native_acceptance_result",
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "host": diagnostic["host"],
        "runtime": diagnostic["runtime"],
        "capabilities": diagnostic["capabilities"],
        "statuses": statuses,
        "producer": _producer_record(target, artifact_root, output),
    }


_CPU_SOURCE = r"""
#include <cstddef>
#if defined(_WIN32)
#define CONTRACT_EXPORT extern "C" __declspec(dllexport)
#else
#define CONTRACT_EXPORT extern "C" __attribute__((visibility("default")))
#endif

CONTRACT_EXPORT void pointwise_kernel(
    const float* input, float* output, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) {
    output[index] = input[index] * 2.0f + 1.0f;
  }
}

CONTRACT_EXPORT double reduction_kernel(
    const float* input, std::size_t count) {
  double total = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    total += static_cast<double>(input[index]);
  }
  return total;
}
""".lstrip()


def _produce_cpu_acceptance(
    diagnostic: dict[str, Any], artifact_root: Path, output: Path
) -> dict[str, Any]:
    import torch

    cpu_root = artifact_root / "cpu"
    cpu_root.mkdir(parents=True, exist_ok=True)
    source = cpu_root / "generated.cpp"
    source.write_text(_CPU_SOURCE, encoding="utf-8")
    suffix = ".dll" if platform.system() == "Windows" else (
        ".dylib" if platform.system() == "Darwin" else ".so"
    )
    binary = cpu_root / f"kernel{suffix}"
    stdout_log = cpu_root / "compile.stdout.txt"
    stderr_log = cpu_root / "compile.stderr.txt"
    compiler = diagnostic["capabilities"]["cpu_cpp"]
    compiler_path = str(Path(compiler["compiler_path"]).resolve())
    source_argument = str(source.resolve())
    output_argument = str(binary.resolve())
    if platform.system() == "Windows":
        command = [
            compiler_path,
            "/nologo",
            "/LD",
            "/O2",
            "/std:c++17",
            source_argument,
            f"/Fe{output_argument}",
        ]
    else:
        command = [
            compiler_path,
            "-shared",
            "-fPIC",
            "-O3",
            "-std=c++17",
            source_argument,
            "-o",
            output_argument,
        ]
    completed = subprocess.run(
        command,
        cwd=artifact_root,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    stdout_log.write_text(completed.stdout, encoding="utf-8")
    stderr_log.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not binary.is_file():
        raise RuntimeError(
            f"native CPU compilation failed with return code {completed.returncode}"
        )

    library = ctypes.CDLL(str(binary.resolve()))
    pointwise = library.pointwise_kernel
    pointwise.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    pointwise.restype = None
    reduction = library.reduction_kernel
    reduction.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    reduction.restype = ctypes.c_double

    values = torch.linspace(-1.0, 1.0, steps=4096, dtype=torch.float32)
    pointwise_native = torch.empty_like(values)
    pointwise(
        ctypes.c_void_p(values.data_ptr()),
        ctypes.c_void_p(pointwise_native.data_ptr()),
        values.numel(),
    )
    pointwise_eager = values * 2.0 + 1.0
    pointwise_comparison = _comparison_record(pointwise_eager, pointwise_native)
    reduction_native = torch.tensor(
        reduction(ctypes.c_void_p(values.data_ptr()), values.numel()),
        dtype=torch.float64,
    )
    reduction_eager = values.to(torch.float64).sum()
    reduction_comparison = _comparison_record(reduction_eager, reduction_native)

    warmup = 3
    iterations = 10
    for _ in range(warmup):
        pointwise(
            ctypes.c_void_p(values.data_ptr()),
            ctypes.c_void_p(pointwise_native.data_ptr()),
            values.numel(),
        )
    pointwise_samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        pointwise(
            ctypes.c_void_p(values.data_ptr()),
            ctypes.c_void_p(pointwise_native.data_ptr()),
            values.numel(),
        )
        pointwise_samples.append(max((time.perf_counter_ns() - started) / 1e6, 1e-9))
    for _ in range(warmup):
        reduction(ctypes.c_void_p(values.data_ptr()), values.numel())
    reduction_samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        reduction(ctypes.c_void_p(values.data_ptr()), values.numel())
        reduction_samples.append(max((time.perf_counter_ns() - started) / 1e6, 1e-9))

    overall = {
        "status": "PASS",
        "reference": "torch.eager",
        "allclose": True,
        "rtol": max(
            pointwise_comparison["rtol"], reduction_comparison["rtol"]
        ),
        "atol": max(
            pointwise_comparison["atol"], reduction_comparison["atol"]
        ),
        "max_abs_error": max(
            pointwise_comparison["max_abs_error"],
            reduction_comparison["max_abs_error"],
        ),
        "max_rel_error": max(
            pointwise_comparison["max_rel_error"],
            reduction_comparison["max_rel_error"],
        ),
        "reference_max_abs": max(
            pointwise_comparison["reference_max_abs"],
            reduction_comparison["reference_max_abs"],
        ),
    }
    document = _acceptance_base(diagnostic, "cpu", artifact_root, output)
    document["acceptance_results"] = {
        "cpu_native": {
            "status": "PASS",
            "compiler": {
                "path": compiler_path,
                "version": compiler["compiler_version"],
            },
            "generated_source": {
                **_artifact_reference(source, artifact_root),
                "language": "c++",
            },
            "compile": {
                "status": "PASS",
                "command": command,
                "returncode": completed.returncode,
                "source_argument": source_argument,
                "output_argument": output_argument,
                "stdout_log": _artifact_reference(stdout_log, artifact_root),
                "stderr_log": _artifact_reference(stderr_log, artifact_root),
                "output_artifact": _artifact_reference(binary, artifact_root),
            },
            "load": {
                "status": "PASS",
                "method": "ctypes.CDLL",
                "artifact_path": _artifact_reference(binary, artifact_root)["path"],
                "entry_points": ["pointwise_kernel", "reduction_kernel"],
            },
            "execute": {
                "status": "PASS",
                "returncode": 0,
                "calls": 2 * (1 + warmup + iterations),
            },
            "numerical_comparison": overall,
            "workloads": {
                "pointwise": {
                    "status": "PASS",
                    "kind": "pointwise",
                    "entry_point": "pointwise_kernel",
                    "warmup": warmup,
                    "iterations": iterations,
                    "timings_ms": _timing_stats(pointwise_samples),
                    "input_summary": _tensor_summary(values),
                    "eager_output_summary": _tensor_summary(pointwise_eager),
                    "native_output_summary": _tensor_summary(pointwise_native),
                    "numerical_comparison": pointwise_comparison,
                },
                "reduction": {
                    "status": "PASS",
                    "kind": "reduction",
                    "entry_point": "reduction_kernel",
                    "warmup": warmup,
                    "iterations": iterations,
                    "timings_ms": _timing_stats(reduction_samples),
                    "input_summary": _tensor_summary(values),
                    "eager_output_summary": _tensor_summary(reduction_eager),
                    "native_output_summary": _tensor_summary(reduction_native),
                    "numerical_comparison": reduction_comparison,
                },
            },
            "environment": {
                "os": diagnostic["host"]["os"],
                "architecture": diagnostic["host"]["architecture"],
                "python_version": diagnostic["runtime"]["python_version"],
                "torch_version": diagnostic["runtime"]["torch_version"],
                "torch_git_version": diagnostic["runtime"]["torch_git_version"],
                "backend": "contract_native_cpp",
            },
        }
    }
    return document


_TRITON_SOURCE = r"""
import triton
import triton.language as tl


@triton.jit
def pointwise_kernel(
    input_pointer,
    output_pointer,
    element_count,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < element_count
    values = tl.load(input_pointer + offsets, mask=mask)
    tl.store(output_pointer + offsets, values * 2.0 + 1.0, mask=mask)
""".lstrip()


def _cuda_driver_version() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
        "--id=0",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi could not report the CUDA driver version")
    version = next(
        (line.strip() for line in completed.stdout.splitlines() if line.strip()), ""
    )
    if not version:
        raise RuntimeError("CUDA driver version output was empty")
    return version


def _produce_cuda_acceptance(
    diagnostic: dict[str, Any], artifact_root: Path, output: Path
) -> dict[str, Any]:
    import torch
    import triton

    cuda_root = artifact_root / "cuda"
    cache_root = cuda_root / "triton_cache"
    cuda_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    source = cuda_root / "generated.py"
    source.write_text(_TRITON_SOURCE, encoding="utf-8")
    module_name = f"_native_backend_contract_triton_{os.getpid()}"
    specification = importlib.util.spec_from_file_location(module_name, source)
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load generated Triton source")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    previous_cache = os.environ.get("TRITON_CACHE_DIR")
    os.environ["TRITON_CACHE_DIR"] = str(cache_root.resolve())
    history_started = False
    try:
        values = torch.linspace(
            -1.0, 1.0, steps=1 << 20, device="cuda", dtype=torch.float32
        )
        output_tensor = torch.empty_like(values)
        eager = values * 2.0 + 1.0
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        try:
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks="all",
                max_entries=100000,
            )
            history_started = True
        except Exception as error:
            raise RuntimeError(
                f"CUDA allocator history could not be enabled: {error}"
            ) from error

        warmup = 3
        iterations = 10
        configurations = (
            {"BLOCK_SIZE": 128, "num_warps": 4},
            {"BLOCK_SIZE": 256, "num_warps": 8},
        )
        candidates: list[dict[str, Any]] = []
        grid = lambda metadata: (triton.cdiv(values.numel(), metadata["BLOCK_SIZE"]),)
        for configuration in configurations:
            for _ in range(warmup):
                module.pointwise_kernel[grid](
                    values,
                    output_tensor,
                    values.numel(),
                    **configuration,
                )
            torch.cuda.synchronize()
            samples: list[float] = []
            for _ in range(iterations):
                started = time.perf_counter_ns()
                module.pointwise_kernel[grid](
                    values,
                    output_tensor,
                    values.numel(),
                    **configuration,
                )
                torch.cuda.synchronize()
                samples.append(max((time.perf_counter_ns() - started) / 1e6, 1e-9))
            candidates.append(
                {
                    "name": (
                        f"block_{configuration['BLOCK_SIZE']}_"
                        f"warps_{configuration['num_warps']}"
                    ),
                    "parameters": dict(configuration),
                    "timings_ms": _timing_stats(samples),
                }
            )
        winner = min(candidates, key=lambda item: item["timings_ms"]["median"])
        winner_parameters = dict(winner["parameters"])
        module.pointwise_kernel[grid](
            values,
            output_tensor,
            values.numel(),
            **winner_parameters,
        )
        torch.cuda.synchronize()
        comparison = _comparison_record(eager, output_tensor)
        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
        snapshot = torch.cuda.memory._snapshot()
        snapshot_path = cuda_root / "memory_snapshot.pickle"
        trace_path = cuda_root / "memory_trace.json"
        with snapshot_path.open("wb") as snapshot_file:
            pickle.dump(snapshot, snapshot_file, protocol=pickle.HIGHEST_PROTOCOL)
        trace_path.write_text(
            json.dumps(
                {
                    "device_traces": snapshot.get("device_traces", []),
                    "segment_count": len(snapshot.get("segments", [])),
                    "producer": Path(__file__).name,
                },
                ensure_ascii=False,
                indent=2,
                default=repr,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        if history_started:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass
        if previous_cache is None:
            os.environ.pop("TRITON_CACHE_DIR", None)
        else:
            os.environ["TRITON_CACHE_DIR"] = previous_cache
        sys.modules.pop(module_name, None)

    cache_files = sorted(path for path in cache_root.rglob("*") if path.is_file())
    if not cache_files:
        raise RuntimeError("Triton execution did not produce a cache artifact")
    cache_archive = cuda_root / "triton_cache.zip"
    with zipfile.ZipFile(
        cache_archive, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for cache_file in cache_files:
            archive.write(cache_file, cache_file.relative_to(cache_root).as_posix())

    device_index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device_index)
    driver_version = _cuda_driver_version()
    cuda_runtime = str(torch.version.cuda)
    triton_version = str(triton.__version__)
    document = _acceptance_base(diagnostic, "cuda", artifact_root, output)
    document["acceptance_results"] = {
        "cuda_triton": {
            "status": "PASS",
            "device": {
                "name": str(properties.name),
                "index": device_index,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "driver_version": driver_version,
                "cuda_runtime_version": cuda_runtime,
                "triton_version": triton_version,
            },
            "generated_source": {
                **_artifact_reference(source, artifact_root),
                "language": "triton",
            },
            "benchmark": {
                "warmup": warmup,
                "iterations": iterations,
                "synchronization": "torch.cuda.synchronize",
                "candidates": candidates,
                "winner": {
                    "candidate_name": winner["name"],
                    "metric": "median_ms",
                    "timing_ms": winner["timings_ms"]["median"],
                },
            },
            "cache_artifact": _artifact_reference(cache_archive, artifact_root),
            "execution": {
                "status": "PASS",
                "returncode": 0,
                "calls": len(configurations) * (warmup + iterations) + 1,
                "launcher": "triton.JITFunction.run",
            },
            "input_summary": _tensor_summary(values),
            "eager_output_summary": _tensor_summary(eager),
            "native_output_summary": _tensor_summary(output_tensor),
            "numerical_comparison": comparison,
            "allocator": {
                "max_memory_allocated_bytes": allocated,
                "max_memory_reserved_bytes": reserved,
                "reset_peak_api": "torch.cuda.reset_peak_memory_stats",
                "history_enabled": True,
                "memory_snapshot": _artifact_reference(
                    snapshot_path, artifact_root
                ),
                "memory_trace": _artifact_reference(trace_path, artifact_root),
            },
            "environment": {
                "os": diagnostic["host"]["os"],
                "architecture": diagnostic["host"]["architecture"],
                "python_version": diagnostic["runtime"]["python_version"],
                "torch_version": diagnostic["runtime"]["torch_version"],
                "torch_git_version": diagnostic["runtime"]["torch_git_version"],
                "cuda_runtime_version": cuda_runtime,
                "driver_version": driver_version,
                "triton_version": triton_version,
                "backend": "contract_triton",
            },
        }
    }
    return document


def _command_probe(arguments: argparse.Namespace) -> int:
    document = probe_local_capabilities()
    errors = validate_document(document)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    write_document(document, arguments.output)
    print(json.dumps(document["statuses"], sort_keys=True))
    print(f"diagnostic={Path(arguments.output).resolve()}")
    return 0


def _command_produce(arguments: argparse.Namespace) -> int:
    artifact_root = Path(arguments.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    output = Path(arguments.output)
    diagnostic = probe_local_capabilities()
    status_key = "cpu_native" if arguments.target == "cpu" else "cuda_triton"
    if diagnostic["statuses"][status_key] != "NOT_RUN":
        write_document(diagnostic, output)
        print(
            f"{arguments.target} native producer: BLOCKED; "
            f"diagnostic={output.resolve()}",
            file=sys.stderr,
        )
        return 3
    try:
        if arguments.target == "cpu":
            document = _produce_cpu_acceptance(diagnostic, artifact_root, output)
        else:
            document = _produce_cuda_acceptance(diagnostic, artifact_root, output)
        errors = validate_acceptance(
            document, arguments.target, artifact_root, verify_files=True
        )
        if errors:
            raise RuntimeError("; ".join(errors))
    except Exception as error:
        diagnostic["statuses"][status_key] = "FAIL"
        capability_key = "cpu_cpp" if arguments.target == "cpu" else "cuda"
        diagnostic["capabilities"][capability_key]["detail"] += (
            f"; producer failed: {error}"
        )
        write_document(diagnostic, output)
        print(
            f"{arguments.target} native producer: FAIL: {error}",
            file=sys.stderr,
        )
        return 4
    write_document(document, output)
    print(
        f"{arguments.target} native producer: PASS; result={output.resolve()}"
    )
    return 0


def _command_validate(arguments: argparse.Namespace) -> int:
    try:
        document = json.loads(Path(arguments.input).read_text(encoding="utf-8"))
        errors = validate_acceptance(
            document,
            arguments.target,
            arguments.artifact_root,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"acceptance input error: {error}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 2
    print(f"{arguments.target} native acceptance evidence: PASS")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="write a local capability diagnostic")
    probe.add_argument("--output", type=Path, required=True)
    probe.set_defaults(handler=_command_probe)

    produce = subparsers.add_parser(
        "produce",
        help="run a real environment-gated CPU or CUDA native evidence producer",
    )
    produce.add_argument("--target", choices=("cpu", "cuda"), required=True)
    produce.add_argument("--output", type=Path, required=True)
    produce.add_argument("--artifact-root", type=Path, required=True)
    produce.set_defaults(handler=_command_produce)

    validate = subparsers.add_parser(
        "validate", help="validate a native acceptance evidence bundle"
    )
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--target", choices=("cpu", "cuda"), required=True)
    validate.add_argument("--artifact-root", type=Path, default=Path.cwd())
    validate.set_defaults(handler=_command_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
