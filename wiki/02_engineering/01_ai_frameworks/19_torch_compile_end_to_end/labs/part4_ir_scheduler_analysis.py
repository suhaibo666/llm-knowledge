import argparse
import json
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch._inductor import config, memory
from torch._inductor.debug import DebugContext
from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import Scheduler
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx


SOURCE_BASELINE = "e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52"
ARTIFACT_NAMES = (
    "environment.json",
    "manifest.json",
    "ir_matrix.json",
    "scheduler_dependencies.json",
    "fusion_comparison.json",
    "reorder_comparison.json",
    "liveness_peak.json",
)


@dataclass
class LoweredCase:
    graph_module: torch.fx.GraphModule
    graph: GraphLowering
    scheduler: Scheduler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _json_value(value: Any) -> int | float | bool | str | None:
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    if bool(getattr(value, "is_number", False)):
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    return str(value)


def _json_sequence(values: Any) -> list[int | float | bool | str | None]:
    return [_json_value(value) for value in values]


def _optional_attribute(value: Any, name: str, default: Any) -> Any:
    try:
        return getattr(value, name)
    except (AttributeError, NotImplementedError):
        return default


def _lower(
    model,
    example_inputs: tuple[torch.Tensor, ...],
    *,
    max_fusion_size: int = 64,
    reorder_for_peak_memory: bool = True,
) -> LoweredCase:
    graph_module = make_fx(model)(*example_inputs)
    graph = GraphLowering(
        graph_module,
        example_inputs=example_inputs,
        is_inference=True,
    )
    with (
        config.patch(
            {
                "max_fusion_size": max_fusion_size,
                "reorder_for_peak_memory": reorder_for_peak_memory,
                "trace.enabled": False,
            }
        ),
        DebugContext(),
        V.set_graph_handler(graph),
        V.set_extern_kernel_nodes([]),
    ):
        graph.run(*example_inputs)
        scheduler = Scheduler(graph.operations)
    return LoweredCase(graph_module, graph, scheduler)


def _layout_record(value: Any) -> dict[str, Any] | None:
    try:
        layout = value.get_layout()
    except (AttributeError, NotImplementedError):
        return None
    record: dict[str, Any] = {"type": type(layout).__name__}
    for field in ("size", "stride"):
        data = getattr(layout, field, None)
        if data is not None:
            record[field] = _json_sequence(data)
    record["device"] = str(getattr(layout, "device", "unknown"))
    record["dtype"] = str(getattr(layout, "dtype", "unknown"))
    record["offset"] = _json_value(getattr(layout, "offset", None))
    return record


def _dependency_record(dependency: Any) -> dict[str, Any]:
    return {
        "type": type(dependency).__name__,
        "name": _optional_attribute(dependency, "name", None),
        "index": _json_value(_optional_attribute(dependency, "index", None)),
        "var_names": _json_sequence(
            _optional_attribute(dependency, "var_names", ())
        ),
        "size": _json_sequence(_optional_attribute(dependency, "size", ())),
        "mode": _optional_attribute(dependency, "mode", None),
    }


def _scheduler_node_record(node: Any) -> dict[str, Any]:
    read_writes = node.read_writes
    return {
        "name": node.get_name(),
        "type": type(node).__name__,
        "operation_names": sorted(str(name) for name in node.get_operation_names()),
        "reads": [_dependency_record(dep) for dep in read_writes.reads],
        "writes": [_dependency_record(dep) for dep in read_writes.writes],
        "last_usage": sorted(str(name) for name in node.last_usage),
    }


def _case_record(case: LoweredCase) -> dict[str, Any]:
    buffers = []
    for buffer in case.graph.buffers:
        data = getattr(buffer, "data", None)
        buffers.append(
            {
                "name": buffer.get_name(),
                "type": type(buffer).__name__,
                "data_type": type(data).__name__ if data is not None else None,
                "layout": _layout_record(buffer),
            }
        )
    outputs = [
        {
            "type": type(output).__name__,
            "name": output.get_name(),
            "layout": _layout_record(output),
        }
        for output in case.graph.graph_outputs
    ]
    return {
        "fx_graph": str(case.graph_module.graph),
        "operation_types": [type(op).__name__ for op in case.graph.operations],
        "buffers": buffers,
        "outputs": outputs,
        "scheduler_nodes": [
            _scheduler_node_record(node) for node in case.scheduler.nodes
        ],
    }


def _dependency_graph(case: LoweredCase) -> dict[str, Any]:
    nodes = [_scheduler_node_record(node) for node in case.scheduler.nodes]
    writer_by_buffer: dict[str, str] = {}
    for node in nodes:
        for write in node["writes"]:
            name = write["name"]
            if name is not None:
                writer_by_buffer[str(name)] = str(node["name"])

    graph_inputs = set(str(name) for name in case.graph.graph_input_names)
    edges = []
    for node in nodes:
        for read in node["reads"]:
            buffer_name = read["name"]
            if buffer_name is None:
                continue
            name = str(buffer_name)
            producer = writer_by_buffer.get(name)
            if producer is None and name in graph_inputs:
                producer = f"graph_input:{name}"
            edges.append(
                {
                    "producer": producer or "external_or_unresolved",
                    "consumer": node["name"],
                    "buffer": name,
                    "dependency": read,
                }
            )
    return {
        "nodes": nodes,
        "edges": edges,
        "graph_inputs": sorted(graph_inputs),
        "graph_outputs": sorted(str(name) for name in case.graph.get_output_names()),
    }


def _fusion_record(case: LoweredCase, max_fusion_size: int) -> dict[str, Any]:
    nodes = [_scheduler_node_record(node) for node in case.scheduler.nodes]
    return {
        "max_fusion_size": max_fusion_size,
        "node_count": len(nodes),
        "has_fused_scheduler_node": any(
            "FusedSchedulerNode" in str(node["type"]) for node in nodes
        ),
        "nodes": nodes,
    }


def _schedule_is_topological(case: LoweredCase) -> bool:
    dependency_graph = _dependency_graph(case)
    positions = {
        node.get_name(): index
        for index, node in enumerate(case.scheduler.nodes)
    }
    for edge in dependency_graph["edges"]:
        producer = edge["producer"]
        consumer = edge["consumer"]
        if (
            producer in positions
            and consumer in positions
            and positions[producer] >= positions[consumer]
        ):
            return False
    return True


def _liveness_record(case: LoweredCase) -> dict[str, Any]:
    with V.set_graph_handler(case.graph):
        scheduled_order_estimated_peak_bytes, freeable_inputs = (
            memory.prepare_planning_info(
                case.scheduler.nodes,
                case.scheduler.name_to_buf,
                case.scheduler.name_to_fused_node,
                case.graph.graph_input_names,
                case.graph.get_output_names(),
            )
        )
        estimated_peak_bytes, timeline = memory.estimate_peak_memory(
            case.scheduler.nodes,
            freeable_inputs,
            case.graph.get_output_names(),
        )

    buffers = []
    for name, scheduler_buffer in case.scheduler.name_to_buf.items():
        planning = scheduler_buffer.mpi_buffer
        buffers.append(
            {
                "name": name,
                "size_alloc_bytes": int(planning.size_alloc),
                "size_free_bytes": int(planning.size_free),
                "successor_nodes": sorted(
                    node.get_name() for node in planning.succ_nodes
                ),
            }
        )
    nodes = []
    for node in case.scheduler.nodes:
        planning = node.mpi_node
        nodes.append(
            {
                **_scheduler_node_record(node),
                "planning_index": int(planning.index),
                "output_allocation_bytes": int(planning.size),
                "predecessor_nodes": sorted(
                    predecessor.get_name() for predecessor in planning.pred_nodes
                ),
                "successor_nodes": sorted(
                    successor.get_name() for successor in planning.succ_nodes
                ),
            }
        )
    return {
        "estimator": "torch._inductor.memory.estimate_peak_memory",
        "estimator_kind": "static_scheduler_estimate_not_physical_allocator_peak",
        "scheduled_order_estimated_peak_bytes": int(
            scheduled_order_estimated_peak_bytes
        ),
        "estimated_peak_bytes": int(estimated_peak_bytes),
        "timeline_bytes": [int(value) for value in timeline],
        "buffers": buffers,
        "nodes": nodes,
        "native_kernel_performance_tested": False,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    elementwise = _lower(lambda x: torch.sin(x + 1), (torch.randn(3, 4),))
    broadcast = _lower(
        lambda x, y: x + y,
        (torch.randn(3, 1), torch.randn(1, 4)),
    )
    transpose_view = _lower(
        lambda x: torch.ops.aten.permute.default(x, (1, 0)),
        (torch.randn(3, 4),),
    )
    contiguous_copy = _lower(
        lambda x: torch.ops.aten.clone.default(
            torch.ops.aten.permute.default(x, (1, 0)),
            memory_format=torch.contiguous_format,
        ),
        (torch.randn(3, 4),),
    )
    reduction = _lower(lambda x: x.sum(dim=1), (torch.randn(8, 16),))
    matmul = _lower(
        lambda x, weight: x @ weight,
        (torch.randn(3, 4), torch.randn(4, 5)),
    )

    case_records = {
        "elementwise": _case_record(elementwise),
        "broadcast": _case_record(broadcast),
        "transpose_view": _case_record(transpose_view),
        "contiguous_copy": _case_record(contiguous_copy),
        "reduction": _case_record(reduction),
        "matmul": _case_record(matmul),
    }
    broadcast_reads = case_records["broadcast"]["scheduler_nodes"][0]["reads"]
    observations = {
        "elementwise_ir_observed": any(
            buffer["data_type"] == "Pointwise"
            for buffer in case_records["elementwise"]["buffers"]
        ),
        "broadcast_index_observed": (
            len(broadcast_reads) == 2
            and len({read["index"] for read in broadcast_reads}) == 2
        ),
        "transpose_view_observed": (
            case_records["transpose_view"]["outputs"][0]["type"]
            == "ReinterpretView"
            and not case_records["transpose_view"]["scheduler_nodes"]
        ),
        "copy_buffer_observed": any(
            buffer["type"] == "ComputedBuffer"
            for buffer in case_records["contiguous_copy"]["buffers"]
        ),
        "reduction_ir_observed": any(
            buffer["data_type"] == "Reduction"
            for buffer in case_records["reduction"]["buffers"]
        ),
        "matmul_extern_observed": any(
            operation.startswith("ExternKernel")
            for operation in case_records["matmul"]["operation_types"]
        ),
    }
    if not all(observations.values()):
        raise AssertionError(f"IR matrix observation failed: {observations}")

    dependency_case = _lower(
        lambda x: torch.sin(x.sum(dim=1)),
        (torch.randn(8, 16),),
        max_fusion_size=1,
    )
    dependency_graph = _dependency_graph(dependency_case)
    dependency_edges_recorded = any(
        edge["producer"].startswith("op")
        for edge in dependency_graph["edges"]
    )
    if not dependency_edges_recorded:
        raise AssertionError("expected an internal Scheduler dependency edge")

    fusion_model = lambda x: (torch.sin(x + 1), torch.cos(x + 2))
    fusion_input = torch.randn(8, 16)
    fusion_enabled = _lower(
        fusion_model,
        (fusion_input,),
        max_fusion_size=64,
    )
    fusion_limited = _lower(
        fusion_model,
        (fusion_input,),
        max_fusion_size=1,
    )
    enabled_record = _fusion_record(fusion_enabled, 64)
    limited_record = _fusion_record(fusion_limited, 1)
    fusion_toggle_observed = (
        enabled_record["has_fused_scheduler_node"]
        and not limited_record["has_fused_scheduler_node"]
        and enabled_record["node_count"] < limited_record["node_count"]
    )
    if not fusion_toggle_observed:
        raise AssertionError("fusion size toggle did not change Scheduler groups")

    reorder_model = lambda x, weight, y: x @ weight + y.sum()
    reorder_inputs = (
        torch.randn(16, 16),
        torch.randn(16, 16),
        torch.randn(256, 256),
    )
    reorder_enabled = _lower(
        reorder_model,
        reorder_inputs,
        max_fusion_size=1,
        reorder_for_peak_memory=True,
    )
    reorder_disabled = _lower(
        reorder_model,
        reorder_inputs,
        max_fusion_size=1,
        reorder_for_peak_memory=False,
    )
    reorder_enabled_liveness = _liveness_record(reorder_enabled)
    reorder_disabled_liveness = _liveness_record(reorder_disabled)
    reorder_enabled_order = [
        node.get_name() for node in reorder_enabled.scheduler.nodes
    ]
    reorder_disabled_order = [
        node.get_name() for node in reorder_disabled.scheduler.nodes
    ]
    reorder_enabled_topological = _schedule_is_topological(reorder_enabled)
    reorder_disabled_topological = _schedule_is_topological(reorder_disabled)
    reorder_comparison_recorded = (
        len(reorder_enabled_order) == len(reorder_disabled_order)
        and len(reorder_enabled_order) > 0
        and reorder_enabled_liveness["estimated_peak_bytes"] > 0
        and reorder_disabled_liveness["estimated_peak_bytes"] > 0
        and reorder_enabled_topological
        and reorder_disabled_topological
    )
    if not reorder_comparison_recorded:
        raise AssertionError("reorder on/off comparison was not recorded")
    reorder_effect_observed = (
        reorder_enabled_order != reorder_disabled_order
        and reorder_enabled_liveness["estimated_peak_bytes"]
        < reorder_disabled_liveness["estimated_peak_bytes"]
    )
    if not reorder_effect_observed:
        raise AssertionError("peak-memory reorder did not improve the test case")

    liveness_peak = _liveness_record(dependency_case)
    static_peak_estimate_recorded = (
        liveness_peak["estimated_peak_bytes"] > 0
        and len(liveness_peak["timeline_bytes"]) == len(dependency_case.scheduler.nodes) + 1
    )
    if not static_peak_estimate_recorded:
        raise AssertionError("static peak estimator did not produce a timeline")

    evidence_boundary = {
        "graph_lowering_executed": True,
        "scheduler_executed": True,
        "static_memory_estimator_executed": True,
        "native_codegen_requested": False,
        "native_kernel_compiled": False,
        "native_kernel_executed": False,
        "native_kernel_performance_tested": False,
        "physical_allocator_peak_measured": False,
    }
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
        "native_kernel_performance_tested": False,
        "evidence_boundary": evidence_boundary,
    }
    manifest = {
        "schema_version": 1,
        "command": (
            "python part4_ir_scheduler_analysis.py "
            "--output-dir <artifact-directory>"
        ),
        "artifacts": list(ARTIFACT_NAMES),
        "analysis_path": "make_fx -> GraphLowering.run -> Scheduler",
        "source_locator_baseline": SOURCE_BASELINE,
        "runtime_torch_git": torch.version.git_version,
        "runtime_matches_source_baseline": torch.version.git_version
        == SOURCE_BASELINE,
        "graph_lowering_execution": "runtime_observed",
        "scheduler_execution": "runtime_observed",
        "native_kernel_execution": "not_run",
        "native_kernel_performance_tested": False,
        "runtime_observations": [
            "IR classes and layouts",
            "Scheduler read/write dependencies and last usage",
            "Scheduler fusion groups under max_fusion_size",
            "Scheduler order and static peak with peak-memory reorder on/off",
            "static Scheduler peak-memory estimate",
        ],
        "evidence_boundary": evidence_boundary,
    }
    ir_matrix = {
        "evidence_kind": "runtime_observed_internal_graphlowering",
        "cases": case_records,
        "observations": observations,
        "native_kernel_performance_tested": False,
    }
    scheduler_dependencies = {
        "evidence_kind": "runtime_observed_scheduler",
        **dependency_graph,
        "dependency_edges_recorded": dependency_edges_recorded,
        "native_kernel_performance_tested": False,
    }
    fusion_comparison = {
        "evidence_kind": "runtime_observed_scheduler_no_codegen",
        "fusion_enabled": enabled_record,
        "fusion_limited": limited_record,
        "fusion_toggle_observed": fusion_toggle_observed,
        "native_kernel_count": None,
        "native_kernel_performance_tested": False,
    }
    reorder_comparison = {
        "evidence_kind": "runtime_observed_scheduler_no_codegen",
        "configuration": "torch._inductor.config.reorder_for_peak_memory",
        "reorder_enabled": {
            "node_order": reorder_enabled_order,
            "liveness": reorder_enabled_liveness,
        },
        "reorder_disabled": {
            "node_order": reorder_disabled_order,
            "liveness": reorder_disabled_liveness,
        },
        "schedule_changed": reorder_enabled_order != reorder_disabled_order,
        "estimated_peak_changed": (
            reorder_enabled_liveness["estimated_peak_bytes"]
            != reorder_disabled_liveness["estimated_peak_bytes"]
        ),
        "comparison_recorded": reorder_comparison_recorded,
        "effect_observed": reorder_effect_observed,
        "reorder_enabled_topological": reorder_enabled_topological,
        "reorder_disabled_topological": reorder_disabled_topological,
        "native_kernel_count": None,
        "native_kernel_performance_tested": False,
    }

    _write_json(output_dir / "environment.json", environment)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "ir_matrix.json", ir_matrix)
    _write_json(
        output_dir / "scheduler_dependencies.json",
        scheduler_dependencies,
    )
    _write_json(output_dir / "fusion_comparison.json", fusion_comparison)
    _write_json(output_dir / "reorder_comparison.json", reorder_comparison)
    _write_json(output_dir / "liveness_peak.json", liveness_peak)

    statuses = {
        **observations,
        "dependency_edges_recorded": dependency_edges_recorded,
        "fusion_toggle_observed": fusion_toggle_observed,
        "reorder_comparison_recorded": reorder_comparison_recorded,
        "reorder_effect_observed": reorder_effect_observed,
        "static_peak_estimate_recorded": static_peak_estimate_recorded,
        "native_kernel_performance_tested": False,
    }
    for key, value in statuses.items():
        print(f"{key}={value}")
    print(f"artifact_dir={output_dir.as_posix()}")


if __name__ == "__main__":
    main()
