import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch


LAB_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_ROOT = LAB_ROOT.parents[4]
DEFAULT_OUTPUT_PATH = LAB_ROOT / "artifacts" / "course_runtime_observations.json"


@dataclass(frozen=True)
class ScriptSpec:
    script: str
    output_subdir: str | None = None


COURSE_SCRIPT_SPECS = (
    ScriptSpec("part1_graph_taxonomy.py"),
    ScriptSpec("part1_fx_core.py"),
    ScriptSpec("part1_values_signatures.py"),
    ScriptSpec("part1_symbolic_shapes.py"),
    ScriptSpec("part1_effects_alias.py"),
    ScriptSpec("part1_structured_hop.py"),
    ScriptSpec("part2_capture_frontends.py"),
    ScriptSpec("part2_normalization.py"),
    ScriptSpec("part2_aot_graphs.py"),
    ScriptSpec("part2_aot_recompute_analysis.py", "part2_recompute"),
    ScriptSpec("part2_activation_peak.py", "part2_activation_peak"),
    ScriptSpec("part2_continuous_aot_inductor.py", "part2_continuous_aot_inductor"),
    ScriptSpec("part3_passes.py"),
    ScriptSpec("part3_pattern.py", "part3_pattern"),
    ScriptSpec("part3_legality.py"),
    ScriptSpec("part3_end_to_end_pass.py", "part3"),
    ScriptSpec("part3_real_stage_hooks.py", "part3_real_stage_hooks"),
    ScriptSpec("part4_inductor.py"),
    ScriptSpec("part4_ir_scheduler_analysis.py", "part4_ir"),
    ScriptSpec("part4_artifact_bundle.py", "part4"),
    ScriptSpec("series_artifact_bundle.py", "end_to_end"),
)


def validate_script_specs(specs: Sequence[ScriptSpec], lab_root: Path) -> None:
    scripts = [spec.script for spec in specs]
    if len(scripts) != len(set(scripts)):
        raise ValueError("duplicate script declaration")
    for script in scripts:
        path = lab_root / script
        if not path.is_file():
            raise ValueError(f"declared script does not exist: {script}")


def parse_key_value_stdout(stdout: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in stdout.splitlines():
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
        if match is not None:
            parsed[match.group(1)] = match.group(2)
    return parsed


def collect_runtime_environment() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cuda_available": torch.cuda.is_available(),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_and_write(
    specs: Sequence[ScriptSpec],
    *,
    lab_root: Path,
    working_directory: Path,
    output_path: Path,
) -> int:
    validate_script_specs(specs, lab_root)
    observations: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="course_runtime_observations_") as temp_dir:
        temporary_output_root = Path(temp_dir)
        for spec in specs:
            script_path = (lab_root / spec.script).resolve()
            command = [sys.executable, "-B", str(script_path)]
            if spec.output_subdir is not None:
                command.extend(
                    [
                        "--output-dir",
                        str(temporary_output_root / spec.output_subdir),
                    ]
                )
            completed = subprocess.run(
                command,
                cwd=working_directory,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            observations.append(
                {
                    "script": spec.script,
                    "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
                    "command": command,
                    "working_directory": str(working_directory.resolve()),
                    "exit_code": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "stdout_key_values": parse_key_value_stdout(completed.stdout),
                }
            )

    all_scripts_passed = all(
        observation["exit_code"] == 0 for observation in observations
    )
    failed_scripts = [
        str(observation["script"])
        for observation in observations
        if observation["exit_code"] != 0
    ]
    payload = {
        "schema_version": "course-runtime-observations/v1",
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "passed" if all_scripts_passed else "failed",
        "all_scripts_passed": all_scripts_passed,
        "script_count": len(specs),
        "completed_script_count": len(observations),
        "failed_script_count": len(failed_scripts),
        "failed_scripts": failed_scripts,
        "runtime_environment": collect_runtime_environment(),
        "observations": observations,
    }
    write_json_atomic(output_path, payload)
    return 0 if all_scripts_passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the 21 course Labs and aggregate their runtime observations."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path of the single aggregated JSON artifact.",
    )
    args = parser.parse_args(argv)
    exit_code = run_and_write(
        COURSE_SCRIPT_SPECS,
        lab_root=LAB_ROOT,
        working_directory=KNOWLEDGE_ROOT,
        output_path=args.output,
    )
    print(f"course_runtime_artifact={args.output.resolve()}")
    print(f"course_runtime_all_scripts_passed={exit_code == 0}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
