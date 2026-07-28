import argparse
import contextlib
import hashlib
import json
import platform
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.async_compile import AsyncCompile
import torch._inductor.cpp_builder as cpp_builder
import torch._inductor.lowering as lowering


SOURCE_BASELINE = "e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52"
TRACE_FILES = (
    "fx_graph_readable.py",
    "fx_graph_runnable.py",
    "fx_graph_transformed.py",
    "ir_pre_fusion.txt",
    "ir_post_fusion.txt",
    "inductor_provenance_tracking_node_mappings.json",
    "output_code.py",
)


def pointwise_reduction(x):
    return torch.sin(x.sum(dim=1))


def matmul_model(x, weight):
    return x @ weight


def fallback_eigvals_model(x):
    return torch.linalg.eigvals(x)


def _find_trace_directory(trace_root: Path) -> Path:
    candidates = sorted(trace_root.rglob("ir_pre_fusion.txt"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one Inductor trace below {trace_root}, found {len(candidates)}"
        )
    return candidates[0].parent


def _copy_trace(trace_root: Path, output_dir: Path) -> dict[str, str]:
    source_dir = _find_trace_directory(trace_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for name in TRACE_FILES:
        source = source_dir / name
        if source.is_file():
            destination = output_dir / name
            shutil.copy2(source, destination)
            copied[name] = destination.as_posix()
    return copied


def _run_codegen_only(
    model,
    example_inputs,
    output_dir: Path,
    *,
    max_fusion_size: int,
) -> dict[str, object]:
    captured_sources: list[str] = []
    scheduler_groups: list[dict[str, object]] = []

    def fake_cpp_pybinding(self, argtypes, source_code):
        captured_sources.append(source_code)

        def no_op_kernel(*args):
            return None

        return no_op_kernel

    def capture_post_fusion(nodes):
        for group in nodes:
            subnodes = []
            for subnode in group.get_nodes():
                ir_node = getattr(subnode, "node", None)
                if ir_node is None:
                    continue
                origins = []
                for origin in ir_node.get_origins():
                    origins.append(
                        {
                            "name": str(origin.name),
                            "op": str(origin.op),
                            "target": str(origin.target),
                            "stack_trace": origin.stack_trace,
                        }
                    )
                subnodes.append(
                    {
                        "scheduler_node": subnode.get_name(),
                        "ir_type": type(ir_node).__name__,
                        "origins": origins,
                        "stack_traces": sorted(ir_node.get_stack_traces()),
                    }
                )
            scheduler_groups.append(
                {
                    "group": group.get_name(),
                    "type": type(group).__name__,
                    "operation_names": sorted(
                        str(name) for name in group.get_operation_names()
                    ),
                    "subnodes": subnodes,
                }
            )
        return nodes

    with tempfile.TemporaryDirectory(prefix="graph_series_codegen_") as temp:
        trace_root = Path(temp) / "trace"
        torch._dynamo.reset()
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    cpp_builder,
                    "check_compiler_exist_windows",
                    lambda compiler: None,
                )
            )
            stack.enter_context(
                patch.object(
                    cpp_builder,
                    "get_compiler_version_info",
                    lambda compiler: "mock-msvc-codegen-only",
                )
            )
            stack.enter_context(
                patch.object(cpp_builder, "get_cpp_compiler", lambda: "cl")
            )
            stack.enter_context(
                patch.object(AsyncCompile, "cpp_pybinding", fake_cpp_pybinding)
            )
            stack.enter_context(
                config.patch(
                    {
                        "force_disable_caches": True,
                        "max_fusion_size": max_fusion_size,
                        "trace.debug_dir": str(trace_root),
                        "trace.enabled": True,
                        "trace.provenance_tracking_level": 1,
                        "_post_fusion_custom_pass": capture_post_fusion,
                    }
                )
            )

            compiled = torch.compile(model, backend="inductor", fullgraph=True)
            compiled(*example_inputs)

        copied = _copy_trace(trace_root, output_dir)

    if not captured_sources:
        raise RuntimeError("Inductor codegen did not emit a C++ source")
    kernel_path = output_dir / "captured_cpp_kernel.cpp"
    source_text = "\n\n".join(
            f"// captured translation unit {index}\n{source}"
            for index, source in enumerate(captured_sources)
        )
    kernel_path.write_text(
        source_text,
        encoding="utf-8",
    )
    post_fusion = (output_dir / "ir_post_fusion.txt").read_text(encoding="utf-8")
    provenance_path = (
        output_dir / "inductor_provenance_tracking_node_mappings.json"
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    origin_records: dict[str, list[dict[str, object]]] = {}
    for group in scheduler_groups:
        for subnode in group["subnodes"]:
            for origin in subnode["origins"]:
                origin_records.setdefault(str(origin["name"]), []).append(
                    {
                        "scheduler_group": group["group"],
                        "scheduler_node": subnode["scheduler_node"],
                        "stack_trace": origin["stack_trace"],
                    }
                )
    joined_chains = []
    for cpp_code, post_nodes in provenance.get("cppCodeToPost", {}).items():
        for post_node in post_nodes:
            for origin in origin_records.get(str(post_node), []):
                joined_chains.append(
                    {
                        "cpp_code": cpp_code,
                        "scheduler_group": origin["scheduler_group"],
                        "scheduler_node": origin["scheduler_node"],
                        "post_fx_node": post_node,
                        "pre_fx_nodes": provenance.get("postToPre", {}).get(
                            post_node, []
                        ),
                        "source_stack": origin["stack_trace"],
                    }
                )
    provenance_chain = {
        "scheduler_groups": scheduler_groups,
        "raw_mapping_schema_keys": sorted(provenance),
        "joined_chains": joined_chains,
    }
    (output_dir / "provenance_chain.json").write_text(
        json.dumps(provenance_chain, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scheduler_to_fx = any(origin_records.values())
    kernel_to_fx = bool(provenance.get("cppCodeToPost"))
    fx_to_python = any(
        bool(chain["source_stack"]) for chain in joined_chains
    )
    scheduler_kernel_source_chain = bool(joined_chains) and all(
        chain["pre_fx_nodes"] and chain["source_stack"]
        for chain in joined_chains
    )
    return {
        "captured_cpp_translation_units": len(captured_sources),
        "cpp_entrypoint_count": len(captured_sources),
        "cpp_loop_count": source_text.count("for("),
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "copied_trace_files": copied,
        "has_fused_scheduler": "FusedSchedulerNode" in post_fusion,
        "scheduler_to_fx_provenance_observed": scheduler_to_fx,
        "kernel_to_fx_provenance_observed": kernel_to_fx,
        "fx_to_python_source_observed": fx_to_python,
        "scheduler_kernel_source_chain_observed": scheduler_kernel_source_chain,
        "evidence_boundary": {
            "cpp_compiler_mocked": True,
            "wrapper_executed_to_trigger_codegen": True,
            "mock_noop_kernel_executed": True,
            "generated_cpp_source_compiled": False,
            "generated_cpp_kernel_executed": False,
            "numerical_correctness_checked": False,
            "native_kernel_performance_tested": False,
        },
    }


def _register_custom_scale_lowering():
    @torch.library.custom_op("graph_series_lab::scale", mutates_args=())
    def scale(x: torch.Tensor, factor: float) -> torch.Tensor:
        return x * factor

    @scale.register_fake
    def scale_fake(x, factor):
        return torch.empty_like(x)

    @lowering.register_lowering(torch.ops.graph_series_lab.scale.default)
    def scale_lowering(x, factor):
        return lowering.mul(x, factor)

    return torch.ops.graph_series_lab.scale.default


def _real_pointwise_compile_status() -> str:
    torch._dynamo.reset()
    try:
        compiled = torch.compile(pointwise_reduction, backend="inductor", fullgraph=True)
        x = torch.randn(8, 16)
        actual = compiled(x)
        if not torch.allclose(actual, pointwise_reduction(x)):
            raise AssertionError("compiled pointwise output differs from eager")
        return "compiled"
    except Exception as exc:
        if "Compiler: cl is not found" not in str(exc):
            raise
        return "blocked_missing_msvc_cl"


def _run_real_trace(model, example_inputs, output_dir: Path):
    with tempfile.TemporaryDirectory(prefix="graph_series_real_trace_") as temp:
        trace_root = Path(temp) / "trace"
        torch._dynamo.reset()
        with config.patch(
            {
                "force_disable_caches": True,
                "trace.debug_dir": str(trace_root),
                "trace.enabled": True,
                "trace.provenance_tracking_level": 1,
            }
        ):
            compiled = torch.compile(model, backend="inductor", fullgraph=True)
            actual = compiled(*example_inputs)
        copied = _copy_trace(trace_root, output_dir)
    return actual, copied


def _real_external_paths(output_dir: Path) -> tuple[bool, bool, dict[str, object]]:
    x = torch.randn(8, 16)
    weight = torch.randn(16, 32)
    actual_matmul, matmul_trace = _run_real_trace(
        matmul_model,
        (x, weight),
        output_dir / "external_matmul",
    )
    matmul_ok = torch.allclose(actual_matmul, matmul_model(x, weight))

    matrix = torch.randn(4, 4)
    actual, fallback_trace = _run_real_trace(
        fallback_eigvals_model,
        (matrix,),
        output_dir / "fallback_eigvals",
    )
    expected = fallback_eigvals_model(matrix)
    eigvals_ok = torch.allclose(actual, expected, equal_nan=True)
    fallback_ir = (
        output_dir / "fallback_eigvals" / "ir_pre_fusion.txt"
    ).read_text(encoding="utf-8")
    fallback_wrapper = (
        output_dir / "fallback_eigvals" / "output_code.py"
    ).read_text(encoding="utf-8")
    fallback_trace_captured = (
        bool(fallback_trace)
        and "FallbackKernel" in fallback_ir
        and (output_dir / "fallback_eigvals" / "output_code.py").is_file()
    )
    fallback_wrapper_observed = (
        "torch.ops.aten.linalg_eig.default" in fallback_wrapper
        and "cpp_pybinding" not in fallback_wrapper
    )
    return matmul_ok, eigvals_ok, {
        "external_matmul_trace": matmul_trace,
        "fallback_eigvals_trace": fallback_trace,
        "fallback_trace_captured": fallback_trace_captured,
        "fallback_wrapper_observed": fallback_wrapper_observed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    real_pointwise_status = _real_pointwise_compile_status()
    matmul_ok, eigvals_ok, external_trace = _real_external_paths(output_dir)
    if not matmul_ok or not eigvals_ok:
        raise AssertionError("real extern/fallback execution did not match eager")
    fallback_trace_captured = bool(external_trace["fallback_trace_captured"])
    if not fallback_trace_captured:
        raise AssertionError("expected an inspectable FallbackKernel trace")
    fallback_wrapper_observed = bool(external_trace["fallback_wrapper_observed"])
    if not fallback_wrapper_observed:
        raise AssertionError("expected the real fallback wrapper call")

    fusion_input = torch.randn(8, 16)
    fusion_enabled = _run_codegen_only(
        pointwise_reduction,
        (fusion_input,),
        output_dir / "fusion_enabled",
        max_fusion_size=64,
    )
    fusion_limited = _run_codegen_only(
        pointwise_reduction,
        (fusion_input,),
        output_dir / "fusion_limited",
        max_fusion_size=1,
    )
    if (
        not fusion_enabled["has_fused_scheduler"]
        or fusion_limited["has_fused_scheduler"]
    ):
        raise AssertionError("fusion size toggle did not change Scheduler groups")
    fusion_codegen_structure_changed = (
        fusion_enabled["source_sha256"] != fusion_limited["source_sha256"]
        and fusion_enabled["cpp_loop_count"] < fusion_limited["cpp_loop_count"]
        and fusion_enabled["cpp_entrypoint_count"] == 1
        and fusion_limited["cpp_entrypoint_count"] == 1
    )
    if not fusion_codegen_structure_changed:
        raise AssertionError("fusion toggle did not change captured C++ loop structure")
    scheduler_kernel_source_chain_observed = bool(
        fusion_enabled["scheduler_kernel_source_chain_observed"]
    )
    if not scheduler_kernel_source_chain_observed:
        raise AssertionError("scheduler/kernel/source provenance chain was not joined")

    custom_scale = _register_custom_scale_lowering()

    def custom_lowering_model(x):
        return custom_scale(x, 2.0).sum(dim=1)

    custom_result = _run_codegen_only(
        custom_lowering_model,
        (torch.randn(8, 16),),
        output_dir / "custom_lowering",
        max_fusion_size=64,
    )
    custom_ir = (output_dir / "custom_lowering" / "ir_pre_fusion.txt").read_text(
        encoding="utf-8"
    )
    custom_lowering_reached_ir = (
        "FallbackKernel" not in custom_ir
        and "ExternKernel" not in custom_ir
        and "graph_series_lab" not in custom_ir
        and "ComputedBuffer" in custom_ir
        and "ops.mul" in custom_ir
        and "ops.reduction" in custom_ir
    )
    if not custom_lowering_reached_ir:
        raise AssertionError("custom lowering did not produce the expected loop IR")

    environment = {
        "source_locator_baseline": SOURCE_BASELINE,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "runtime_torch_git": torch.version.git_version,
        "runtime_matches_source_baseline": torch.version.git_version
        == SOURCE_BASELINE,
        "cuda_available": torch.cuda.is_available(),
        "msvc_cl": shutil.which("cl"),
    }
    summary = {
        "external_matmul_execution": matmul_ok,
        "fallback_eigvals_execution": eigvals_ok,
        "fallback_trace_captured": fallback_trace_captured,
        "fallback_wrapper_observed": fallback_wrapper_observed,
        "external_trace": external_trace,
        "codegen_only_status": "generated_not_executed",
        "real_pointwise_compile_status": real_pointwise_status,
        "fusion_enabled_has_fused_scheduler": fusion_enabled[
            "has_fused_scheduler"
        ],
        "fusion_limited_has_fused_scheduler": fusion_limited[
            "has_fused_scheduler"
        ],
        "fusion_codegen_structure_changed": fusion_codegen_structure_changed,
        "scheduler_to_fx_provenance_observed": fusion_enabled[
            "scheduler_to_fx_provenance_observed"
        ],
        "kernel_to_fx_provenance_observed": fusion_enabled[
            "kernel_to_fx_provenance_observed"
        ],
        "fx_to_python_source_observed": fusion_enabled[
            "fx_to_python_source_observed"
        ],
        "scheduler_kernel_source_chain_observed": (
            scheduler_kernel_source_chain_observed
        ),
        "custom_lowering_reached_ir": custom_lowering_reached_ir,
        "triton_autotune_tested": False,
        "fusion_enabled": fusion_enabled,
        "fusion_limited": fusion_limited,
        "custom_lowering": custom_result,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"external_matmul_execution={matmul_ok}")
    print(f"fallback_eigvals_execution={eigvals_ok}")
    print(f"fallback_trace_captured={fallback_trace_captured}")
    print(f"fallback_wrapper_observed={fallback_wrapper_observed}")
    print("codegen_only_status=generated_not_executed")
    print(f"real_pointwise_compile_status={real_pointwise_status}")
    print(
        "fusion_enabled_has_fused_scheduler="
        + str(fusion_enabled["has_fused_scheduler"])
    )
    print(
        "fusion_limited_has_fused_scheduler="
        + str(fusion_limited["has_fused_scheduler"])
    )
    print(f"fusion_codegen_structure_changed={fusion_codegen_structure_changed}")
    print(
        "scheduler_to_fx_provenance_observed="
        + str(fusion_enabled["scheduler_to_fx_provenance_observed"])
    )
    print(
        "kernel_to_fx_provenance_observed="
        + str(fusion_enabled["kernel_to_fx_provenance_observed"])
    )
    print(
        "fx_to_python_source_observed="
        + str(fusion_enabled["fx_to_python_source_observed"])
    )
    print(
        "scheduler_kernel_source_chain_observed="
        + str(scheduler_kernel_source_chain_observed)
    )
    print(f"custom_lowering_reached_ir={custom_lowering_reached_ir}")
    print("triton_autotune_tested=False")
    print(f"artifact_dir={output_dir.as_posix()}")


if __name__ == "__main__":
    main()
