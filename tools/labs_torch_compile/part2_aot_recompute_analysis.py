import argparse
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

import torch
import torch._functorch.config as functorch_config
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_function
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch.utils import _pytree as pytree


SOURCE_BASELINE = "e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52"
HIGH_BUDGET = 1.0
LOW_BUDGET = 0.0
USER_OUTPUT_COUNT = 1


def model(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mm(torch.cos(x), weight).sum()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _graph_text(graph_module: torch.fx.GraphModule) -> str:
    return f"{graph_module.graph}\n\n# Generated Python\n{graph_module.code}"


def _write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def _call_targets(graph_module: torch.fx.GraphModule) -> list[str]:
    return [
        str(node.target)
        for node in graph_module.graph.nodes
        if node.op == "call_function"
    ]


def _saved_value_records(
    forward_graph: torch.fx.GraphModule,
) -> list[dict[str, object]]:
    output_node = next(
        node for node in forward_graph.graph.nodes if node.op == "output"
    )
    output_leaves = pytree.tree_leaves(output_node.args[0])
    saved_leaves = output_leaves[USER_OUTPUT_COUNT:]
    records: list[dict[str, object]] = []

    for slot, leaf in enumerate(saved_leaves):
        assert isinstance(leaf, torch.fx.Node), (
            f"saved output slot {slot} is not an FX Node: {leaf!r}"
        )
        tensor_meta = leaf.meta.get("tensor_meta")
        assert tensor_meta is not None, (
            f"saved output slot {slot} has no tensor_meta: {leaf.name}"
        )
        shape = [int(dimension) for dimension in tensor_meta.shape]
        logical_bytes = math.prod(shape) * torch.empty(
            (), dtype=tensor_meta.dtype
        ).element_size()
        records.append(
            {
                "slot": slot,
                "name": leaf.name,
                "op": leaf.op,
                "target": str(leaf.target),
                "shape": shape,
                "dtype": str(tensor_meta.dtype),
                "logical_bytes": logical_bytes,
            }
        )
    return records


def _cross_graph_node_refs(
    graph_module: torch.fx.GraphModule, graph_name: str
) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for node in graph_module.graph.nodes:
        for value in pytree.tree_leaves((node.args, node.kwargs)):
            if (
                isinstance(value, torch.fx.Node)
                and value.graph is not graph_module.graph
            ):
                refs.append(
                    {
                        "graph": graph_name,
                        "consumer": node.name,
                        "producer": value.name,
                    }
                )
    return refs


def _capture_partition(
    budget: float,
    base_x: torch.Tensor,
    base_weight: torch.Tensor,
) -> dict[str, object]:
    joint_graphs: list[torch.fx.GraphModule] = []
    forward_graphs: list[torch.fx.GraphModule] = []
    backward_graphs: list[torch.fx.GraphModule] = []

    def partition(joint_module, *partition_args, **partition_kwargs):
        joint_graphs.append(joint_module)
        return min_cut_rematerialization_partition(
            joint_module, *partition_args, **partition_kwargs
        )

    def forward_compiler(graph_module, _example_inputs):
        forward_graphs.append(graph_module)
        return make_boxed_func(graph_module.forward)

    def backward_compiler(graph_module, _example_inputs):
        backward_graphs.append(graph_module)
        return make_boxed_func(graph_module.forward)

    x = base_x.detach().clone().requires_grad_(True)
    weight = base_weight.detach().clone().requires_grad_(True)
    with functorch_config.patch(activation_memory_budget=budget):
        compiled = aot_function(
            model,
            fw_compiler=forward_compiler,
            bw_compiler=backward_compiler,
            partition_fn=partition,
        )
        output = compiled(x, weight)
        output.backward()

    assert len(joint_graphs) == 1
    assert len(forward_graphs) == 1
    assert len(backward_graphs) == 1
    assert x.grad is not None
    assert weight.grad is not None

    forward_graph = forward_graphs[0]
    backward_graph = backward_graphs[0]
    saved_values = _saved_value_records(forward_graph)
    cross_graph_refs = [
        *_cross_graph_node_refs(forward_graph, "forward"),
        *_cross_graph_node_refs(backward_graph, "backward"),
    ]
    return {
        "budget": budget,
        "forward_graph": forward_graph,
        "backward_graph": backward_graph,
        "saved_values": saved_values,
        "saved_slots": len(saved_values),
        "saved_bytes": sum(
            int(record["logical_bytes"]) for record in saved_values
        ),
        "backward_call_targets": _call_targets(backward_graph),
        "cross_graph_node_refs": cross_graph_refs,
        "output": output.detach(),
        "x_grad": x.grad.detach().clone(),
        "weight_grad": weight.grad.detach().clone(),
    }


def _json_partition(partition: dict[str, object]) -> dict[str, object]:
    return {
        "activation_memory_budget": partition["budget"],
        "forward_saved_value_slots": partition["saved_values"],
        "forward_saved_slot_count": partition["saved_slots"],
        "forward_saved_logical_bytes": partition["saved_bytes"],
        "backward_call_targets": partition["backward_call_targets"],
        "cross_graph_node_refs": partition["cross_graph_node_refs"],
    }


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    torch.manual_seed(0)

    base_x = torch.randn(8, 8)
    base_weight = torch.randn(8, 8)
    eager_x = base_x.detach().clone().requires_grad_(True)
    eager_weight = base_weight.detach().clone().requires_grad_(True)
    eager_output = model(eager_x, eager_weight)
    eager_output.backward()

    high = _capture_partition(HIGH_BUDGET, base_x, base_weight)
    low = _capture_partition(LOW_BUDGET, base_x, base_weight)

    low_extra_targets = sorted(
        (
            Counter(low["backward_call_targets"])
            - Counter(high["backward_call_targets"])
        ).elements()
    )
    gradient_matches = all(
        torch.allclose(actual, expected)
        for partition in (high, low)
        for actual, expected in (
            (partition["output"], eager_output.detach()),
            (partition["x_grad"], eager_x.grad),
            (partition["weight_grad"], eager_weight.grad),
        )
    )
    cross_graph_refs = [
        *high["cross_graph_node_refs"],
        *low["cross_graph_node_refs"],
    ]
    physical_allocator_peak_measured = False

    assert high["saved_bytes"] > low["saved_bytes"], (
        f"expected high budget to save more logical bytes: "
        f"high={high['saved_bytes']}, low={low['saved_bytes']}"
    )
    assert high["saved_slots"] > low["saved_slots"], (
        f"expected high budget to save more slots: "
        f"high={high['saved_slots']}, low={low['saved_slots']}"
    )
    assert low_extra_targets, "low-budget backward exposed no recompute targets"
    assert gradient_matches, "AOTAutograd gradients differ from eager"
    assert not cross_graph_refs, f"cross-Graph Node references: {cross_graph_refs}"
    assert physical_allocator_peak_measured is False

    checks = {
        "budget_high_saved_bytes_gt_low": True,
        "budget_low_recompute_targets_observed": True,
        "gradient_matches": True,
        "cross_graph_node_refs": 0,
        "physical_allocator_peak_measured": False,
    }
    comparison = {
        "budget_high": _json_partition(high),
        "budget_low": _json_partition(low),
        "budget_low_recompute_targets": low_extra_targets,
        "recompute_target_definition": (
            "Positive multiset difference between low- and high-budget "
            "backward call_function targets."
        ),
        "checks": checks,
        "measurement_scope": {
            "saved_bytes": (
                "Sum of logical tensor bytes in forward outputs after the "
                "single user-output slot; aliases/views are separate ABI slots."
            ),
            "physical_allocator_peak_measured": False,
            "disclaimer": (
                "This Lab inspects partition graphs and does not measure a "
                "physical CPU/CUDA allocator peak."
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text(
        output_dir / "budget_high_forward.py",
        _graph_text(high["forward_graph"]),
    )
    _write_text(
        output_dir / "budget_high_backward.py",
        _graph_text(high["backward_graph"]),
    )
    _write_text(
        output_dir / "budget_low_forward.py",
        _graph_text(low["forward_graph"]),
    )
    _write_text(
        output_dir / "budget_low_backward.py",
        _graph_text(low["backward_graph"]),
    )
    _write_json(output_dir / "partition_comparison.json", comparison)

    environment = {
        "audit_source_baseline": SOURCE_BASELINE,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda_available": torch.cuda.is_available(),
        "seed": 0,
    }
    _write_json(output_dir / "environment.json", environment)

    artifact_names = [
        "environment.json",
        "manifest.json",
        "budget_high_forward.py",
        "budget_high_backward.py",
        "budget_low_forward.py",
        "budget_low_backward.py",
        "partition_comparison.json",
    ]
    manifest = {
        "command": [sys.executable, *sys.argv],
        "model": "sum(mm(cos(x), weight))",
        "partitioner": "min_cut_rematerialization_partition",
        "activation_memory_budgets": {
            "high": HIGH_BUDGET,
            "low": LOW_BUDGET,
        },
        "artifacts": artifact_names,
        "checks": checks,
        "physical_allocator_peak_measured": False,
    }
    _write_json(output_dir / "manifest.json", manifest)

    for key, value in checks.items():
        print(f"{key}={value}")
    print(f"budget_high_saved_slots={high['saved_slots']}")
    print(f"budget_low_saved_slots={low['saved_slots']}")
    print(f"budget_high_saved_bytes={high['saved_bytes']}")
    print(f"budget_low_saved_bytes={low['saved_bytes']}")
    print(f"budget_low_recompute_targets={','.join(low_extra_targets)}")
    print(f"artifact_dir={output_dir.as_posix()}")


if __name__ == "__main__":
    main()
