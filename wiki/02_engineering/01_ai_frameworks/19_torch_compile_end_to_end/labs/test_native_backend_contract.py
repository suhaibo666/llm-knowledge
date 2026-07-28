from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from jsonschema import Draft202012Validator

import native_backend_contract as contract


HASH_A = "a" * 64
HASH_B = "b" * 64


def _host_runtime() -> dict[str, object]:
    return {
        "schema_version": "native-backend-evidence/v1",
        "artifact_type": "native_acceptance_result",
        "generated_at_utc": "2026-07-27T01:02:03Z",
        "host": {
            "hostname": "evidence-host",
            "os": "Windows",
            "os_version": "11",
            "architecture": "AMD64",
        },
        "runtime": {
            "python_version": "3.13.5",
            "python_executable": "C:/Python313/python.exe",
            "torch_version": "2.9.1+cpu",
            "torch_git_version": "5811a8d7da873dd699ff6687092c225caffcf1bb",
        },
        "capabilities": {
            "cpu_cpp": {
                "available": True,
                "compiler_path": "C:/BuildTools/cl.exe",
                "compiler_version": "Microsoft C/C++ 19.44",
                "detail": "compiler executable discovered",
            },
            "cuda": {
                "available": False,
                "device_count": 0,
                "detail": "torch.cuda.is_available() is false",
            },
            "triton": {
                "available": False,
                "version": None,
                "detail": "Triton is not installed",
            },
        },
        "producer": {
            "id": "native_backend_contract.py",
            "contract_version": "native-backend-evidence/v1",
            "module_sha256": hashlib.sha256(
                Path(contract.__file__).read_bytes()
            ).hexdigest(),
            "command": [
                "python",
                "native_backend_contract.py",
                "produce",
                "--target",
                "cpu",
                "--artifact-root",
                "C:/bundle",
                "--output",
                "C:/bundle/result.json",
            ],
            "target": "cpu",
        },
    }


def _timings(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    return {
        "samples": samples,
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "mean": sum(samples) / len(samples),
        "p95": ordered[-1],
        "max": ordered[-1],
    }


def _comparison() -> dict[str, object]:
    return {
        "status": "PASS",
        "reference": "torch.eager",
        "allclose": True,
        "rtol": 1e-5,
        "atol": 1e-6,
        "max_abs_error": 2e-7,
        "max_rel_error": 3e-7,
        "reference_max_abs": 10.0,
    }


def _cpu_document() -> dict[str, object]:
    document = _host_runtime()
    document["statuses"] = {"cpu_native": "PASS", "cuda_triton": "BLOCKED"}
    document["acceptance_results"] = {
        "cpu_native": {
            "status": "PASS",
            "compiler": {
                "path": "C:/BuildTools/cl.exe",
                "version": "Microsoft C/C++ 19.44",
            },
            "generated_source": {
                "path": "cpu/generated.cpp",
                "sha256": HASH_A,
                "language": "c++",
            },
            "compile": {
                "status": "PASS",
                "command": [
                    "C:/BuildTools/cl.exe",
                    "/LD",
                    "C:/bundle/cpu/generated.cpp",
                    "/Fe:C:/bundle/cpu/kernel.pyd",
                ],
                "returncode": 0,
                "source_argument": "C:/bundle/cpu/generated.cpp",
                "output_argument": "C:/bundle/cpu/kernel.pyd",
                "stdout_log": {
                    "path": "cpu/compile.stdout.txt",
                    "sha256": "d" * 64,
                },
                "stderr_log": {
                    "path": "cpu/compile.stderr.txt",
                    "sha256": "e" * 64,
                },
                "output_artifact": {
                    "path": "cpu/kernel.pyd",
                    "sha256": HASH_B,
                },
            },
            "load": {
                "status": "PASS",
                "method": "ctypes.CDLL",
                "artifact_path": "cpu/kernel.pyd",
                "entry_points": ["pointwise_kernel", "reduction_kernel"],
            },
            "execute": {"status": "PASS", "returncode": 0, "calls": 10},
            "numerical_comparison": _comparison(),
            "workloads": {
                "pointwise": {
                    "status": "PASS",
                    "kind": "pointwise",
                    "entry_point": "pointwise_kernel",
                    "warmup": 2,
                    "iterations": 3,
                    "timings_ms": _timings([1.0, 2.0, 3.0]),
                    "input_summary": {
                        "dtype": "torch.float32",
                        "shape": [1024],
                        "numel": 1024,
                        "sha256": "1" * 64,
                    },
                    "eager_output_summary": {
                        "dtype": "torch.float32",
                        "shape": [1024],
                        "numel": 1024,
                        "sha256": "2" * 64,
                    },
                    "native_output_summary": {
                        "dtype": "torch.float32",
                        "shape": [1024],
                        "numel": 1024,
                        "sha256": "2" * 64,
                    },
                    "numerical_comparison": _comparison(),
                },
                "reduction": {
                    "status": "PASS",
                    "kind": "reduction",
                    "entry_point": "reduction_kernel",
                    "warmup": 2,
                    "iterations": 3,
                    "timings_ms": _timings([4.0, 5.0, 6.0]),
                    "input_summary": {
                        "dtype": "torch.float32",
                        "shape": [1024],
                        "numel": 1024,
                        "sha256": "3" * 64,
                    },
                    "eager_output_summary": {
                        "dtype": "torch.float64",
                        "shape": [],
                        "numel": 1,
                        "sha256": "4" * 64,
                    },
                    "native_output_summary": {
                        "dtype": "torch.float64",
                        "shape": [],
                        "numel": 1,
                        "sha256": "4" * 64,
                    },
                    "numerical_comparison": _comparison(),
                },
            },
            "environment": {
                "os": "Windows",
                "architecture": "AMD64",
                "python_version": "3.13.5",
                "torch_version": "2.9.1+cpu",
                "torch_git_version": "5811a8d7da873dd699ff6687092c225caffcf1bb",
                "backend": "contract_native_cpp",
            },
        }
    }
    return document


def _cuda_document() -> dict[str, object]:
    document = _host_runtime()
    document["producer"]["target"] = "cuda"
    target_index = document["producer"]["command"].index("--target") + 1
    document["producer"]["command"][target_index] = "cuda"
    document["host"].update(
        {"os": "Linux", "os_version": "6.8.0", "architecture": "x86_64"}
    )
    document["runtime"]["torch_version"] = "2.9.1+cu128"
    document["capabilities"] = {
        "cpu_cpp": {
            "available": False,
            "compiler_path": None,
            "compiler_version": None,
            "detail": "CPU compiler not required for this result",
        },
        "cuda": {
            "available": True,
            "device_count": 1,
            "detail": "CUDA device 0 is available",
        },
        "triton": {
            "available": True,
            "version": "3.4.0",
            "detail": "Triton imported",
        },
    }
    document["statuses"] = {"cpu_native": "BLOCKED", "cuda_triton": "PASS"}
    document["acceptance_results"] = {
        "cuda_triton": {
            "status": "PASS",
            "device": {
                "name": "NVIDIA H100",
                "index": 0,
                "compute_capability": "9.0",
                "driver_version": "575.57",
                "cuda_runtime_version": "12.8",
                "triton_version": "3.4.0",
            },
            "generated_source": {
                "path": "cuda/generated.py",
                "sha256": HASH_A,
                "language": "triton",
            },
            "benchmark": {
                "warmup": 2,
                "iterations": 3,
                "synchronization": "torch.cuda.synchronize",
                "candidates": [
                    {
                        "name": "block_64",
                        "parameters": {"BLOCK_SIZE": 64, "num_warps": 4},
                        "timings_ms": _timings([3.0, 3.2, 3.1]),
                    },
                    {
                        "name": "block_128",
                        "parameters": {"BLOCK_SIZE": 128, "num_warps": 8},
                        "timings_ms": _timings([2.0, 2.2, 2.1]),
                    },
                ],
                "winner": {
                    "candidate_name": "block_128",
                    "metric": "median_ms",
                    "timing_ms": 2.1,
                },
            },
            "cache_artifact": {
                "path": "cuda/cache.bin",
                "sha256": HASH_B,
            },
            "execution": {
                "status": "PASS",
                "returncode": 0,
                "calls": 11,
                "launcher": "triton.JITFunction.run",
            },
            "input_summary": {
                "dtype": "torch.float32",
                "shape": [1048576],
                "numel": 1048576,
                "sha256": "5" * 64,
            },
            "eager_output_summary": {
                "dtype": "torch.float32",
                "shape": [1048576],
                "numel": 1048576,
                "sha256": "6" * 64,
            },
            "native_output_summary": {
                "dtype": "torch.float32",
                "shape": [1048576],
                "numel": 1048576,
                "sha256": "6" * 64,
            },
            "numerical_comparison": _comparison(),
            "allocator": {
                "max_memory_allocated_bytes": 1048576,
                "max_memory_reserved_bytes": 2097152,
                "reset_peak_api": "torch.cuda.reset_peak_memory_stats",
                "history_enabled": True,
                "memory_snapshot": {
                    "path": "cuda/memory_snapshot.pickle",
                    "sha256": "7" * 64,
                },
                "memory_trace": {
                    "path": "cuda/memory_trace.json",
                    "sha256": "8" * 64,
                },
            },
            "environment": {
                "os": "Linux",
                "architecture": "x86_64",
                "python_version": "3.13.5",
                "torch_version": "2.9.1+cu128",
                "torch_git_version": "5811a8d7da873dd699ff6687092c225caffcf1bb",
                "cuda_runtime_version": "12.8",
                "driver_version": "575.57",
                "triton_version": "3.4.0",
                "backend": "contract_triton",
            },
        }
    }
    return document


def _legacy_cpu_document() -> dict[str, object]:
    document = copy.deepcopy(_cpu_document())
    document.pop("producer")
    cpu = document["acceptance_results"]["cpu_native"]
    compile_record = cpu["compile"]
    compile_record["command"] = ["cl.exe", "/LD", "generated.cpp"]
    for key in (
        "returncode",
        "source_argument",
        "output_argument",
        "stdout_log",
        "stderr_log",
    ):
        compile_record.pop(key)
    cpu["load"] = {"status": "PASS"}
    cpu["execute"] = {"status": "PASS"}
    for workload in cpu["workloads"].values():
        for key in (
            "kind",
            "entry_point",
            "input_summary",
            "eager_output_summary",
            "native_output_summary",
            "numerical_comparison",
        ):
            workload.pop(key)
    return document


def _legacy_cuda_document() -> dict[str, object]:
    document = copy.deepcopy(_cuda_document())
    document.pop("producer")
    cuda = document["acceptance_results"]["cuda_triton"]
    cuda["benchmark"].pop("synchronization")
    for key in (
        "execution",
        "input_summary",
        "eager_output_summary",
        "native_output_summary",
    ):
        cuda.pop(key)
    cuda["allocator"].pop("reset_peak_api")
    cuda["allocator"].pop("history_enabled")
    return document


def _schema_validator() -> Draft202012Validator:
    schema = json.loads(contract.SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


class NativeBackendContractTests(unittest.TestCase):
    def test_complete_cpu_native_result_is_accepted(self) -> None:
        document = _cpu_document()

        self.assertEqual([], list(_schema_validator().iter_errors(document)))
        self.assertEqual([], contract.validate_document(document))
        self.assertEqual([], contract.validate_acceptance(document, "cpu", verify_files=False))

    def test_complete_cuda_triton_result_is_accepted(self) -> None:
        document = _cuda_document()

        self.assertEqual([], list(_schema_validator().iter_errors(document)))
        self.assertEqual([], contract.validate_document(document))
        self.assertEqual(
            [], contract.validate_acceptance(document, "cuda", verify_files=False)
        )

    def test_cuda_autotune_requires_multiple_measured_candidates(self) -> None:
        document = _cuda_document()
        document["acceptance_results"]["cuda_triton"]["benchmark"]["candidates"] = [
            document["acceptance_results"]["cuda_triton"]["benchmark"]["candidates"][1]
        ]

        self.assertTrue(list(_schema_validator().iter_errors(document)))
        self.assertTrue(contract.validate_document(document))

    def test_missing_critical_fields_are_rejected_by_schema_and_validator(self) -> None:
        cases = [
            ("cpu compiler version", _cpu_document(), ("acceptance_results", "cpu_native", "compiler", "version")),
            ("cpu reduction", _cpu_document(), ("acceptance_results", "cpu_native", "workloads", "reduction")),
            ("cuda candidates", _cuda_document(), ("acceptance_results", "cuda_triton", "benchmark", "candidates")),
            ("cuda memory trace", _cuda_document(), ("acceptance_results", "cuda_triton", "allocator", "memory_trace")),
        ]

        for label, document, path in cases:
            with self.subTest(label=label):
                cursor = document
                for key in path[:-1]:
                    cursor = cursor[key]
                del cursor[path[-1]]
                self.assertTrue(list(_schema_validator().iter_errors(document)))
                self.assertTrue(contract.validate_document(document))

    def test_acceptance_requires_concrete_top_level_runtime_versions(self) -> None:
        document = _cpu_document()
        document["runtime"]["torch_git_version"] = None

        self.assertTrue(list(_schema_validator().iter_errors(document)))
        self.assertTrue(contract.validate_document(document))

    def test_diagnostic_cannot_claim_blocked_target_as_pass(self) -> None:
        document = contract.probe_local_capabilities()
        document["artifact_type"] = "local_capability_diagnostic"
        document["statuses"]["cpu_native"] = "PASS"

        self.assertTrue(list(_schema_validator().iter_errors(document)))
        errors = contract.validate_acceptance(document, "cpu", verify_files=False)
        self.assertTrue(errors)
        self.assertTrue(any("diagnostic" in error.lower() for error in errors))

    def test_acceptance_validator_returns_errors_for_malformed_container_types(self) -> None:
        document = _cpu_document()
        document["statuses"] = ["PASS"]
        document["acceptance_results"] = "not-an-object"

        errors = contract.validate_acceptance(document, "cpu", verify_files=False)

        self.assertTrue(errors)
        self.assertTrue(any("schema" in error for error in errors))

    def test_semantics_reject_timing_or_winner_claims_not_backed_by_samples(self) -> None:
        cpu = _cpu_document()
        cpu["acceptance_results"]["cpu_native"]["workloads"]["pointwise"][
            "iterations"
        ] = 4
        cuda = _cuda_document()
        cuda["acceptance_results"]["cuda_triton"]["benchmark"]["winner"][
            "candidate_name"
        ] = "block_64"

        self.assertTrue(
            any("iterations" in error for error in contract.validate_document(cpu))
        )
        self.assertTrue(
            any("winner" in error.lower() for error in contract.validate_document(cuda))
        )

    def test_semantics_reject_p95_not_derived_from_samples(self) -> None:
        document = _cpu_document()
        document["acceptance_results"]["cpu_native"]["workloads"]["pointwise"][
            "timings_ms"
        ]["p95"] = 2.5

        errors = contract.validate_document(document)

        self.assertTrue(any("p95" in error for error in errors))

    def test_acceptance_rejects_environment_or_capability_provenance_mismatch(self) -> None:
        cpu = _cpu_document()
        cpu["acceptance_results"]["cpu_native"]["compiler"][
            "version"
        ] = "different compiler"
        cuda = _cuda_document()
        cuda["acceptance_results"]["cuda_triton"]["environment"][
            "driver_version"
        ] = "different driver"

        cpu_errors = contract.validate_acceptance(cpu, "cpu", verify_files=False)
        cuda_errors = contract.validate_acceptance(cuda, "cuda", verify_files=False)

        self.assertTrue(any("compiler" in error.lower() for error in cpu_errors))
        self.assertTrue(any("driver" in error.lower() for error in cuda_errors))

    def test_acceptance_verifies_referenced_artifact_hashes(self) -> None:
        document = _cpu_document()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "cpu").mkdir()
            source = root / "cpu" / "generated.cpp"
            binary = root / "cpu" / "kernel.pyd"
            stdout_log = root / "cpu" / "compile.stdout.txt"
            stderr_log = root / "cpu" / "compile.stderr.txt"
            source.write_bytes(b"void kernel() {}\n")
            binary.write_bytes(b"native-binary")
            stdout_log.write_bytes(b"compiler stdout\n")
            stderr_log.write_bytes(b"compiler stderr\n")
            document["acceptance_results"]["cpu_native"]["generated_source"][
                "sha256"
            ] = hashlib.sha256(source.read_bytes()).hexdigest()
            document["acceptance_results"]["cpu_native"]["compile"][
                "output_artifact"
            ]["sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
            document["acceptance_results"]["cpu_native"]["compile"]["stdout_log"][
                "sha256"
            ] = hashlib.sha256(stdout_log.read_bytes()).hexdigest()
            document["acceptance_results"]["cpu_native"]["compile"]["stderr_log"][
                "sha256"
            ] = hashlib.sha256(stderr_log.read_bytes()).hexdigest()

            self.assertEqual([], contract.validate_acceptance(document, "cpu", root))
            source.write_bytes(b"tampered")
            errors = contract.validate_acceptance(document, "cpu", root)
            self.assertTrue(any("sha256" in error.lower() for error in errors))

    def test_schema_rejects_artifact_path_traversal_in_both_separator_styles(self) -> None:
        for unsafe_path in ("../outside.cpp", "..\\outside.cpp"):
            with self.subTest(path=unsafe_path):
                document = _cpu_document()
                document["acceptance_results"]["cpu_native"]["generated_source"][
                    "path"
                ] = unsafe_path

                self.assertTrue(list(_schema_validator().iter_errors(document)))
                self.assertTrue(contract.validate_document(document))

    def test_cli_rejects_file_verification_bypass_for_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = Path(temporary_directory) / "result.json"
            result.write_text(json.dumps(_cpu_document()), encoding="utf-8")

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    contract.main(
                        [
                            "validate",
                            "--input",
                            str(result),
                            "--target",
                            "cpu",
                            "--artifact-root",
                            temporary_directory,
                            "--no-verify-files",
                        ]
                    )

            self.assertEqual(2, raised.exception.code)

    def test_cpu_false_pass_bundle_is_rejected(self) -> None:
        document = _cpu_document()
        cpu = document["acceptance_results"]["cpu_native"]
        cpu["compile"]["command"] = ["echo", "not-a-compile"]
        cpu["compile"]["source_argument"] = "not-the-source"
        cpu["compile"]["output_argument"] = "not-the-output"
        cpu["numerical_comparison"].update(
            {
                "rtol": 0.0,
                "atol": 0.0,
                "max_abs_error": 1e300,
                "max_rel_error": 1e300,
            }
        )
        cpu["workloads"]["pointwise"]["numerical_comparison"].update(
            {
                "rtol": 0.0,
                "atol": 0.0,
                "max_abs_error": 1e300,
                "max_rel_error": 1e300,
            }
        )
        errors = contract.validate_acceptance(document, "cpu", verify_files=False)

        self.assertTrue(errors)
        self.assertTrue(
            any(
                token in error.lower()
                for error in errors
                for token in ("compiler", "source_argument", "numerical")
            )
        )

    def test_cuda_false_pass_bundle_is_rejected(self) -> None:
        document = _cuda_document()
        cuda = document["acceptance_results"]["cuda_triton"]
        cuda["benchmark"]["candidates"][1]["parameters"] = copy.deepcopy(
            cuda["benchmark"]["candidates"][0]["parameters"]
        )
        cuda["device"]["index"] = 1
        cuda["numerical_comparison"].update(
            {
                "rtol": 0.0,
                "atol": 0.0,
                "max_abs_error": 1e300,
                "max_rel_error": 1e300,
            }
        )
        repeated_hash = "f" * 64
        cuda["generated_source"]["sha256"] = repeated_hash
        cuda["cache_artifact"]["sha256"] = repeated_hash
        cuda["allocator"]["memory_snapshot"]["sha256"] = repeated_hash
        cuda["allocator"]["memory_trace"]["sha256"] = repeated_hash

        errors = contract.validate_acceptance(document, "cuda", verify_files=False)

        self.assertTrue(errors)
        self.assertTrue(any("candidate" in error.lower() for error in errors))
        self.assertTrue(any("device" in error.lower() for error in errors))
        self.assertTrue(any("numerical" in error.lower() for error in errors))

    def test_non_finite_measurements_are_rejected(self) -> None:
        cpu = _cpu_document()
        cpu_result = cpu["acceptance_results"]["cpu_native"]
        cpu_result["workloads"]["pointwise"]["timings_ms"] = {
            "samples": [float("inf")],
            "min": float("inf"),
            "median": float("inf"),
            "mean": float("inf"),
            "p95": float("inf"),
            "max": float("inf"),
        }
        cpu_result["workloads"]["pointwise"]["iterations"] = 1
        cpu_result["workloads"]["pointwise"]["numerical_comparison"][
            "max_abs_error"
        ] = float("nan")

        errors = contract.validate_acceptance(cpu, "cpu", verify_files=False)

        self.assertTrue(any("finite" in error.lower() for error in errors))

    def test_artifact_symlink_cannot_escape_bundle_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            root = base / "bundle"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            artifact = outside / "artifact.bin"
            artifact.write_bytes(b"outside")
            try:
                os.symlink(outside, root / "link", target_is_directory=True)
            except OSError as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            reference = {
                "path": "link/artifact.bin",
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }

            errors = contract._verify_artifact_reference(reference, root, "artifact")

            self.assertTrue(errors)
            self.assertTrue(any("root" in error.lower() for error in errors))

    def test_malformed_acceptance_returns_exit_two_without_traceback(self) -> None:
        document = _legacy_cpu_document()
        document["acceptance_results"]["cpu_native"][
            "generated_source"
        ] = "not-an-object"
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = Path(temporary_directory) / "result.json"
            result.write_text(json.dumps(document), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(Path(contract.__file__).resolve()),
                    "validate",
                    "--input",
                    str(result),
                    "--target",
                    "cpu",
                    "--artifact-root",
                    temporary_directory,
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(2, completed.returncode)
            self.assertNotIn("Traceback", completed.stderr)

    def test_diagnostic_forbids_acceptance_results(self) -> None:
        document = contract.probe_local_capabilities()
        document["acceptance_results"] = {
            "cpu_native": _legacy_cpu_document()["acceptance_results"]["cpu_native"]
        }

        errors = contract.validate_document(document)

        self.assertTrue(errors)

    def test_result_presence_must_match_top_level_status(self) -> None:
        document = _legacy_cpu_document()
        document["statuses"]["cpu_native"] = "BLOCKED"

        errors = contract.validate_document(document)

        self.assertTrue(
            any("cpu_native" in error and "status" in error.lower() for error in errors)
        )

    def test_broken_triton_install_is_not_reported_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package = Path(temporary_directory) / "triton"
            package.mkdir()
            (package / "__init__.py").write_text(
                "raise RuntimeError('broken triton install')\n", encoding="utf-8"
            )
            sys.path.insert(0, temporary_directory)
            try:
                sys.modules.pop("triton", None)
                result = contract._triton_probe()
            finally:
                sys.modules.pop("triton", None)
                sys.path.remove(temporary_directory)

        self.assertFalse(result["available"])
        self.assertIn("import", result["detail"].lower())

    def test_nonzero_compiler_version_probe_is_not_available(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["cl.exe"], returncode=1, stdout="", stderr="broken compiler"
        )
        with mock.patch.object(contract.platform, "system", return_value="Windows"):
            with mock.patch.object(
                contract.shutil, "which", return_value="C:/Tools/cl.exe"
            ):
                with mock.patch.object(
                    contract.subprocess, "run", return_value=completed
                ):
                    result = contract._compiler_probe()

        self.assertFalse(result["available"])
        self.assertIn("failed", result["detail"].lower())

    def test_blocked_producer_writes_diagnostic_and_returns_three(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "result.json"
            with mock.patch.object(
                contract,
                "probe_local_capabilities",
                return_value={
                    **contract.probe_local_capabilities(),
                    "statuses": {
                        "cpu_native": "BLOCKED",
                        "cuda_triton": "BLOCKED",
                    },
                },
            ):
                try:
                    exit_code = contract.main(
                        [
                            "produce",
                            "--target",
                            "cpu",
                            "--output",
                            str(output),
                            "--artifact-root",
                            temporary_directory,
                        ]
                    )
                except SystemExit as error:
                    exit_code = int(error.code)
            document = (
                json.loads(output.read_text(encoding="utf-8"))
                if output.is_file()
                else None
            )

        self.assertEqual(3, exit_code)
        self.assertIsNotNone(document)
        self.assertEqual("local_capability_diagnostic", document["artifact_type"])
        self.assertNotIn("PASS", document["statuses"].values())

    def test_local_probe_is_a_valid_non_acceptance_diagnostic(self) -> None:
        document = contract.probe_local_capabilities()

        self.assertEqual("local_capability_diagnostic", document["artifact_type"])
        self.assertEqual([], contract.validate_document(document))
        self.assertNotIn("PASS", document["statuses"].values())
        self.assertIn("cpu_cpp", document["capabilities"])
        self.assertIn("cuda", document["capabilities"])
        self.assertIn("triton", document["capabilities"])
        self.assertTrue(document["host"]["hostname"])
        self.assertTrue(document["runtime"]["python_version"])

    def test_write_document_round_trips_probe_without_mutation(self) -> None:
        document = contract.probe_local_capabilities()
        expected = copy.deepcopy(document)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "probe.json"

            contract.write_document(document, output)

            self.assertEqual(expected, json.loads(output.read_text(encoding="utf-8")))
            self.assertEqual(expected, document)


if __name__ == "__main__":
    unittest.main()
