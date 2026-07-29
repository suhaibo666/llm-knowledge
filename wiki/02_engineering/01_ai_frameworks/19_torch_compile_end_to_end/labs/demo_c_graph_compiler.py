"""Volume C entry point: orchestrate the existing graph-compiler evidence labs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from demo_harness import CaseSpec, DemoContext, run_volume_cli


LAB_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ChildScript:
    name: str
    requires_output_dir: bool = False


IR_FX_SCRIPTS = (
    ChildScript("part1_graph_taxonomy.py"),
    ChildScript("part1_fx_core.py"),
    ChildScript("part1_values_signatures.py"),
    ChildScript("part1_symbolic_shapes.py"),
    ChildScript("part1_effects_alias.py"),
    ChildScript("part1_structured_hop.py"),
)

CAPTURE_NORMALIZE_SCRIPTS = (
    ChildScript("part2_capture_frontends.py"),
    ChildScript("part2_normalization.py"),
)

AOT_RECOMPUTE_SCRIPTS = (
    ChildScript("part2_aot_graphs.py"),
    ChildScript("part2_aot_recompute_analysis.py", requires_output_dir=True),
    ChildScript("part2_activation_peak.py", requires_output_dir=True),
    ChildScript("part2_continuous_aot_inductor.py", requires_output_dir=True),
)

PATTERN_REWRITE_SCRIPTS = (
    ChildScript("part3_end_to_end_pass.py", requires_output_dir=True),
    ChildScript("part3_passes.py"),
    ChildScript("part3_real_stage_hooks.py", requires_output_dir=True),
    ChildScript("part3_legality.py"),
    ChildScript("part3_pattern.py", requires_output_dir=True),
)

INDUCTOR_IR_SCRIPTS = (
    ChildScript("part4_inductor.py"),
    ChildScript("part4_ir_scheduler_analysis.py", requires_output_dir=True),
    ChildScript("part4_artifact_bundle.py", requires_output_dir=True),
)

FULL_BUNDLE_SCRIPTS = (
    ChildScript("series_artifact_bundle.py", requires_output_dir=True),
)


def _artifact_names(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return [
        path.relative_to(output_dir).as_posix()
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    ]


def run_child_scripts(
    context: DemoContext,
    scripts: Sequence[ChildScript],
    *,
    script_root: Path = LAB_ROOT,
    python_executable: str = sys.executable,
) -> dict[str, object]:
    """Run child labs sequentially and retain process-level evidence.

    A subprocess boundary is intentional: several existing labs mutate private
    compiler configuration or install tracing hooks. Process isolation prevents
    one teaching case from leaking that state into the next case.
    """

    context.output_dir.mkdir(parents=True, exist_ok=True)
    child_root = context.output_dir / "children"
    child_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(context.seed)
    environment["PYTHONIOENCODING"] = "utf-8"
    records: list[dict[str, object]] = []

    for child in scripts:
        script_path = (script_root / child.name).resolve()
        if not script_path.is_file():
            raise FileNotFoundError(f"child lab does not exist: {script_path}")
        output_dir = child_root / script_path.stem
        command = [python_executable, "-B", str(script_path)]
        if child.requires_output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            command.extend(("--output-dir", str(output_dir)))

        completed = subprocess.run(
            command,
            cwd=script_root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        record = {
            "script": child.name,
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "artifacts": (
                _artifact_names(output_dir)
                if child.requires_output_dir
                else []
            ),
        }
        records.append(record)
        manifest_path = context.output_dir / "child_processes.json"
        manifest_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise ChildProcessError(
                f"{child.name} exited with code {completed.returncode}; "
                f"see {manifest_path}"
            )

    return {
        "child_count": len(records),
        "all_exit_codes_zero": all(
            record["exit_code"] == 0 for record in records
        ),
        "children": records,
    }


def _runner(scripts: Sequence[ChildScript]):
    return lambda context: run_child_scripts(context, scripts)


CASES = (
    CaseSpec(
        case_id="ir_fx",
        title="IR taxonomy, FX graph storage, symbolic values and effects",
        pages=tuple(f"c{index:02d}" for index in range(1, 7)),
        requirements=("torch",),
        run=_runner(IR_FX_SCRIPTS),
        description="Runs the six Part I graph/FX evidence labs.",
    ),
    CaseSpec(
        case_id="capture_normalize",
        title="Capture frontends and functional normalization",
        pages=("c07", "c08"),
        requirements=("torch",),
        run=_runner(CAPTURE_NORMALIZE_SCRIPTS),
        description="Compares capture boundaries and normalization semantics.",
    ),
    CaseSpec(
        case_id="aot_recompute",
        title="AOTAutograd forward/backward partitioning and recompute",
        pages=("c09", "c10", "c11"),
        requirements=("torch",),
        run=_runner(AOT_RECOMPUTE_SCRIPTS),
        description="Emits joint/fw/bw, saved-value and recompute evidence.",
    ),
    CaseSpec(
        case_id="pattern_rewrite",
        title="Pattern matching, graph rewrites, legality and pass placement",
        pages=tuple(f"c{index:02d}" for index in range(12, 17)),
        requirements=("torch",),
        run=_runner(PATTERN_REWRITE_SCRIPTS),
        description="Exercises real matchers, rewrites and failure boundaries.",
    ),
    CaseSpec(
        case_id="inductor_ir",
        title="Inductor IR, scheduling, fusion and generated artifacts",
        pages=tuple(f"c{index:02d}" for index in range(17, 22)),
        requirements=("torch",),
        run=_runner(INDUCTOR_IR_SCRIPTS),
        description="Captures lowering and scheduler evidence without overstating execution.",
    ),
    CaseSpec(
        case_id="full_bundle",
        title="End-to-end graph compiler evidence bundle",
        pages=tuple(f"c{index:02d}" for index in range(1, 22)),
        requirements=("torch",),
        run=_runner(FULL_BUNDLE_SCRIPTS),
        description="Builds the cross-stage provenance bundle for the full C volume.",
    ),
)


if __name__ == "__main__":
    raise SystemExit(run_volume_cli("C", CASES))
