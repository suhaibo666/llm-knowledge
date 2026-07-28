import argparse
import copy
import inspect
import json
import platform
import shutil
from pathlib import Path

import torch
from functorch.compile import make_boxed_func
from torch import nn
from torch._functorch.aot_autograd import aot_module
from torch._functorch.partitioners import default_partition
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from part4_artifact_bundle import SOURCE_BASELINE, _run_codegen_only


AOT_JOINT_ORIGIN_META_KEY = "graph_series_joint_origin_id"


class UnifiedGraphModel(nn.Module):
    """Stable prefix used across the series; later stages progressively lower it."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.register_buffer("offset", torch.linspace(-0.2, 0.2, 4))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        activation, summary = backend_core(x, self.weight, self.offset)
        return {"activation": activation, "summary": summary}


class HigherOrderBranch(nn.Module):
    def forward(
        self, x: torch.Tensor, predicate: torch.Tensor
    ) -> torch.Tensor:
        return torch.cond(
            predicate,
            lambda value: torch.sin(value),
            lambda value: torch.cos(value),
            (x,),
        )


class InvalidHigherOrderBranch(nn.Module):
    def forward(
        self, x: torch.Tensor, predicate: torch.Tensor
    ) -> torch.Tensor:
        return torch.cond(
            predicate,
            lambda value: value.clone(),
            lambda value: value.sum(),
            (x,),
        )


def backend_core(
    x: torch.Tensor,
    weight: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected = x.view(-1, 4) @ weight
    biased = projected + offset
    scratch = biased.clone()
    scratch.view(-1).add_(0.25)
    activation = torch.sin(scratch)
    return activation, activation.sum(dim=1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _graph_text(graph_module: torch.fx.GraphModule) -> str:
    return f"{graph_module.graph}\n\n# Generated Python\n{graph_module.code}"


def _node_records(graph_module: torch.fx.GraphModule) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for node in graph_module.graph.nodes:
        records.append(
            {
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "users": [user.name for user in node.users],
                "meta_keys": sorted(str(key) for key in node.meta),
            }
        )
    return records


def _cross_graph_node_refs(graph_module: torch.fx.GraphModule) -> list[dict[str, str]]:
    cross_refs: list[dict[str, str]] = []
    for node in graph_module.graph.nodes:
        for value in pytree.tree_leaves((node.args, node.kwargs)):
            if (
                isinstance(value, torch.fx.Node)
                and value.graph is not graph_module.graph
            ):
                cross_refs.append({"consumer": node.name, "producer": value.name})
    return cross_refs


def _tag_joint_nodes(
    joint_module: torch.fx.GraphModule,
) -> dict[str, torch.fx.Node]:
    tagged: dict[str, torch.fx.Node] = {}
    for ordinal, node in enumerate(joint_module.graph.nodes):
        origin_id = f"joint:{ordinal}:{node.name}"
        if AOT_JOINT_ORIGIN_META_KEY in node.meta:
            raise AssertionError(
                f"unexpected pre-existing audit metadata on joint node {node.name}"
            )
        node.meta[AOT_JOINT_ORIGIN_META_KEY] = origin_id
        tagged[origin_id] = node
    return tagged


def _joint_partition_mapping(
    joint_module: torch.fx.GraphModule,
    forward_module: torch.fx.GraphModule,
    backward_module: torch.fx.GraphModule,
    tagged_joint_nodes: dict[str, torch.fx.Node],
) -> dict[str, object]:
    stage_modules = {
        "partition_forward": forward_module,
        "partition_backward": backward_module,
    }
    destinations: dict[str, list[tuple[str, torch.fx.Node]]] = {}
    unknown_origin_ids: list[dict[str, str]] = []
    for stage, module in stage_modules.items():
        seen_in_stage: set[str] = set()
        for node in module.graph.nodes:
            origin_id = node.meta.get(AOT_JOINT_ORIGIN_META_KEY)
            if origin_id is None:
                continue
            if not isinstance(origin_id, str) or origin_id not in tagged_joint_nodes:
                unknown_origin_ids.append(
                    {
                        "stage": stage,
                        "node": node.name,
                        "origin_id": repr(origin_id),
                    }
                )
                continue
            if origin_id in seen_in_stage:
                raise AssertionError(
                    f"origin {origin_id} appeared twice in {stage}"
                )
            seen_in_stage.add(origin_id)
            destinations.setdefault(origin_id, []).append((stage, node))

    entries: list[dict[str, object]] = []
    for origin_id, source in tagged_joint_nodes.items():
        mapped_nodes = destinations.get(origin_id, [])
        destination_records = [
            {
                "stage": stage,
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "origin_token_observed": node.meta.get(
                    AOT_JOINT_ORIGIN_META_KEY
                ),
                "python_object_identity_preserved": node is source,
                "owner_graph_preserved": node.graph is source.graph,
            }
            for stage, node in mapped_nodes
        ]
        destination_stages = {stage for stage, _ in mapped_nodes}
        if mapped_nodes:
            mapping_mechanism = (
                "audit origin token propagated while partition extraction "
                "created a fresh placeholder or copied the node"
            )
        elif source.op == "output":
            mapping_mechanism = (
                "the extracted graph creates a fresh output node from mapped "
                "output values; the joint output node itself is not node_copy'd"
            )
        else:
            mapping_mechanism = (
                "the joint node was not selected into either extracted graph "
                "or was removed by extraction-time DCE"
            )
        entries.append(
            {
                "origin_id": origin_id,
                "joint_node": {
                    "name": source.name,
                    "op": source.op,
                    "target": str(source.target),
                },
                "destinations": destination_records,
                "present_in_forward": "partition_forward" in destination_stages,
                "present_in_backward": "partition_backward" in destination_stages,
                "mapping_mechanism": mapping_mechanism,
            }
        )

    mapped_entries = [entry for entry in entries if entry["destinations"]]
    mapped_to_both = [
        entry
        for entry in mapped_entries
        if entry["present_in_forward"] and entry["present_in_backward"]
    ]
    joint_output_entries = [
        entry for entry in entries if entry["joint_node"]["op"] == "output"
    ]
    unmapped_non_output_entries = [
        entry
        for entry in entries
        if not entry["destinations"] and entry["joint_node"]["op"] != "output"
    ]
    all_destination_records = [
        destination
        for entry in mapped_entries
        for destination in entry["destinations"]
    ]
    exact_mapping_checks = {
        "no_unknown_origin_tokens": not unknown_origin_ids,
        "all_mapped_destination_tokens_match_joint_sources": all(
            destination["origin_token_observed"] == entry["origin_id"]
            for entry in mapped_entries
            for destination in entry["destinations"]
        ),
        "all_mapped_destination_objects_are_fresh": all(
            not destination["python_object_identity_preserved"]
            for destination in all_destination_records
        ),
        "all_mapped_destination_graph_owners_are_fresh": all(
            not destination["owner_graph_preserved"]
            for destination in all_destination_records
        ),
        "joint_has_exactly_one_structurally_rebuilt_output": (
            len(joint_output_entries) == 1
            and not joint_output_entries[0]["destinations"]
        ),
    }
    return {
        "mapping_kind": (
            "instrumented exact old-to-new mapping for one default_partition call"
        ),
        "mapping_scope": (
            "joint GraphModule to the fresh forward/backward GraphModules returned "
            "by the same partition_fn invocation, before compiler callbacks"
        ),
        "instrumentation": {
            "meta_key": AOT_JOINT_ORIGIN_META_KEY,
            "injected_before_default_partition": True,
            "collected_immediately_after_default_partition": True,
            "production_schema_key": False,
            "does_not_match_by_node_name": True,
        },
        "source_semantics": (
            "Graph.node_copy copies node.meta and partition-created placeholders "
            "inherit source meta; the audit token observes that extraction path."
        ),
        "entries": entries,
        "summary": {
            "joint_node_count": len(entries),
            "mapped_joint_node_count": len(mapped_entries),
            "unmapped_or_rebuilt_joint_node_count": len(entries)
            - len(mapped_entries),
            "joint_output_node_count": len(joint_output_entries),
            "unmapped_non_output_joint_node_count": len(
                unmapped_non_output_entries
            ),
            "mapped_to_forward_count": sum(
                bool(entry["present_in_forward"]) for entry in mapped_entries
            ),
            "mapped_to_backward_count": sum(
                bool(entry["present_in_backward"]) for entry in mapped_entries
            ),
            "mapped_to_both_graphs_count": len(mapped_to_both),
            "forward_destination_node_count": sum(
                destination["stage"] == "partition_forward"
                for destination in all_destination_records
            ),
            "backward_destination_node_count": sum(
                destination["stage"] == "partition_backward"
                for destination in all_destination_records
            ),
        },
        "unknown_origin_tokens": unknown_origin_ids,
        "exact_mapping_checks": exact_mapping_checks,
        "limitations": [
            (
                "The token is lab-only instrumentation and is not a supported "
                "PyTorch provenance schema."
            ),
            (
                "The fresh output Node is structural glue and is intentionally "
                "not represented as a node_copy destination."
            ),
            (
                "This run does not exercise control_deps see-through/remapping "
                "special cases in the extraction helper."
            ),
        ],
    }


def _origin_token_set(graph_module: torch.fx.GraphModule) -> set[str]:
    return {
        origin_id
        for node in graph_module.graph.nodes
        if isinstance(
            origin_id := node.meta.get(AOT_JOINT_ORIGIN_META_KEY), str
        )
    }


def _attach_compiler_callback_mapping(
    mapping: dict[str, object],
    joint_module: torch.fx.GraphModule,
    partition_forward: torch.fx.GraphModule,
    partition_backward: torch.fx.GraphModule,
    compiler_forward: torch.fx.GraphModule,
    compiler_backward: torch.fx.GraphModule,
) -> dict[str, bool]:
    source_by_origin = {
        origin_id: node
        for node in joint_module.graph.nodes
        if isinstance(
            origin_id := node.meta.get(AOT_JOINT_ORIGIN_META_KEY), str
        )
    }
    partition_by_stage_and_origin: dict[tuple[str, str], torch.fx.Node] = {}
    for stage, module in {
        "forward": partition_forward,
        "backward": partition_backward,
    }.items():
        for node in module.graph.nodes:
            origin_id = node.meta.get(AOT_JOINT_ORIGIN_META_KEY)
            if isinstance(origin_id, str):
                partition_by_stage_and_origin[(stage, origin_id)] = node

    entry_by_origin = {
        entry["origin_id"]: entry for entry in mapping["entries"]
    }
    unknown_origin_ids: list[dict[str, str]] = []
    callback_records: list[dict[str, object]] = []
    for stage, module in {
        "forward_compiler_callback": compiler_forward,
        "backward_compiler_callback": compiler_backward,
    }.items():
        partition_stage = stage.removesuffix("_compiler_callback")
        seen_in_stage: set[str] = set()
        for node in module.graph.nodes:
            origin_id = node.meta.get(AOT_JOINT_ORIGIN_META_KEY)
            if origin_id is None:
                continue
            if not isinstance(origin_id, str) or origin_id not in source_by_origin:
                unknown_origin_ids.append(
                    {
                        "stage": stage,
                        "node": node.name,
                        "origin_id": repr(origin_id),
                    }
                )
                continue
            if origin_id in seen_in_stage:
                raise AssertionError(
                    f"origin {origin_id} appeared twice in {stage}"
                )
            seen_in_stage.add(origin_id)
            source = source_by_origin[origin_id]
            partition_node = partition_by_stage_and_origin.get(
                (partition_stage, origin_id)
            )
            record = {
                "stage": stage,
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "origin_token_observed": origin_id,
                "python_object_identity_preserved_from_joint": node is source,
                "owner_graph_preserved_from_joint": node.graph is source.graph,
                "same_python_object_as_partition_destination": (
                    node is partition_node
                ),
            }
            callback_records.append(record)
            entry_by_origin[origin_id].setdefault(
                "compiler_callback_destinations", []
            ).append(record)

    for entry in mapping["entries"]:
        entry.setdefault("compiler_callback_destinations", [])
    checks = {
        "no_unknown_origin_tokens_at_compiler_callbacks": (
            not unknown_origin_ids
        ),
        "all_compiler_callback_nodes_are_fresh_from_joint": all(
            not record["python_object_identity_preserved_from_joint"]
            for record in callback_records
        ),
        "all_compiler_callback_graph_owners_are_fresh_from_joint": all(
            not record["owner_graph_preserved_from_joint"]
            for record in callback_records
        ),
    }
    mapping["compiler_callback_unknown_origin_tokens"] = unknown_origin_ids
    mapping["compiler_callback_mapping_checks"] = checks
    mapping["summary"]["forward_compiler_callback_node_count"] = sum(
        record["stage"] == "forward_compiler_callback"
        for record in callback_records
    )
    mapping["summary"]["backward_compiler_callback_node_count"] = sum(
        record["stage"] == "backward_compiler_callback"
        for record in callback_records
    )
    return checks


def _tree_allclose(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> bool:
    return actual.keys() == expected.keys() and all(
        torch.allclose(actual[key], expected[key]) for key in actual
    )


def _loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return outputs["activation"].sum() + outputs["summary"].sum()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    model = UnifiedGraphModel().eval()
    example = torch.randn(3, 4)

    model_source = (
        "import torch\n"
        "from torch import nn\n\n"
        + inspect.getsource(UnifiedGraphModel)
        + "\n\n"
        + inspect.getsource(HigherOrderBranch)
        + "\n\n"
        + inspect.getsource(InvalidHigherOrderBranch)
        + "\n\n"
        + inspect.getsource(backend_core)
    )
    _write_text(output_dir / "model_source.py", model_source)
    model_contract = {
        "stable_prefix": [
            "view",
            "matmul",
            "parameter_plus_buffer",
            "clone",
            "view_mutation",
            "sin",
            "reduction",
        ],
        "features": [
            "parameter",
            "buffer",
            "view",
            "mutation",
            "dynamic_shape",
            "structured_output",
            "matmul",
            "pointwise",
            "reduction",
            "higher_order_branch",
        ],
        "higher_order_branch_is_separate_export_variant": True,
        "reason": (
            "The HOP variant keeps the same tensor domain but is exported separately "
            "because nested branch graphs have their own ownership boundary."
        ),
    }
    _write_text(
        output_dir / "model_contract.json",
        json.dumps(model_contract, ensure_ascii=False, indent=2),
    )

    symbolic = symbolic_trace(model)
    _write_text(output_dir / "symbolic_fx.py", _graph_text(symbolic))

    dynamo_graphs: list[torch.fx.GraphModule] = []

    def capture_backend(gm, example_inputs):
        dynamo_graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    dynamo_model = torch.compile(
        copy.deepcopy(model),
        backend=capture_backend,
        fullgraph=True,
        dynamic=True,
    )
    dynamo_output = dynamo_model(example)
    eager_output = model(example)
    if len(dynamo_graphs) != 1:
        raise RuntimeError(f"expected one Dynamo graph, got {len(dynamo_graphs)}")
    dynamo_graph = dynamo_graphs[0]
    _write_text(output_dir / "dynamo_fx.py", _graph_text(dynamo_graph))

    torch._dynamo.reset()
    explanation = torch._dynamo.explain(copy.deepcopy(model))(example)
    _write_text(output_dir / "dynamo_guards.txt", str(explanation))

    batch = torch.export.Dim("batch", min=1, max=8)
    exported = torch.export.export(
        copy.deepcopy(model),
        (example,),
        dynamic_shapes={"x": {0: batch}},
    )
    _write_text(
        output_dir / "exported_program.py",
        _graph_text(exported.graph_module),
    )
    export_signature = {
        "input_specs": [
            {
                "kind": spec.kind.name,
                "arg": repr(spec.arg),
                "target": str(spec.target),
            }
            for spec in exported.graph_signature.input_specs
        ],
        "output_specs": [
            {
                "kind": spec.kind.name,
                "arg": repr(spec.arg),
                "target": str(spec.target),
            }
            for spec in exported.graph_signature.output_specs
        ],
        "range_constraints": {
            str(symbol): str(value)
            for symbol, value in exported.range_constraints.items()
        },
    }
    _write_text(
        output_dir / "export_graph_signature.json",
        json.dumps(export_signature, ensure_ascii=False, indent=2),
    )
    try:
        exported.module()(torch.randn(9, 4))
    except AssertionError as exc:
        export_out_of_range_rejected = "x.size()[0] <= 8" in str(exc)
    else:
        export_out_of_range_rejected = False

    functionalized = torch.func.functionalize(
        copy.deepcopy(model), remove="mutations_and_views"
    )
    functional_aten = make_fx(functionalized)(example)
    _write_text(
        output_dir / "functional_aten.py",
        _graph_text(functional_aten),
    )

    joint_graphs: list[torch.fx.GraphModule] = []
    partition_forward_graphs: list[torch.fx.GraphModule] = []
    partition_backward_graphs: list[torch.fx.GraphModule] = []
    joint_partition_mappings: list[dict[str, object]] = []
    forward_graphs: list[torch.fx.GraphModule] = []
    backward_graphs: list[torch.fx.GraphModule] = []

    def recording_partition(joint_module, *partition_args, **partition_kwargs):
        tagged_joint_nodes = _tag_joint_nodes(joint_module)
        joint_graphs.append(joint_module)
        forward_module, backward_module = default_partition(
            joint_module, *partition_args, **partition_kwargs
        )
        partition_forward_graphs.append(forward_module)
        partition_backward_graphs.append(backward_module)
        joint_partition_mappings.append(
            _joint_partition_mapping(
                joint_module,
                forward_module,
                backward_module,
                tagged_joint_nodes,
            )
        )
        return forward_module, backward_module

    def forward_compiler(gm, example_inputs):
        forward_graphs.append(gm)
        return make_boxed_func(gm.forward)

    def backward_compiler(gm, example_inputs):
        backward_graphs.append(gm)
        return make_boxed_func(gm.forward)

    aot_model = copy.deepcopy(model)
    aot_compiled = aot_module(
        aot_model,
        fw_compiler=forward_compiler,
        bw_compiler=backward_compiler,
        partition_fn=recording_partition,
    )
    aot_input = example.detach().clone().requires_grad_(True)
    aot_outputs = aot_compiled(aot_input)
    _loss(aot_outputs).backward()

    eager_model_for_grad = copy.deepcopy(model)
    eager_input = example.detach().clone().requires_grad_(True)
    eager_outputs_for_grad = eager_model_for_grad(eager_input)
    _loss(eager_outputs_for_grad).backward()

    if not (
        len(joint_graphs)
        == len(partition_forward_graphs)
        == len(partition_backward_graphs)
        == len(joint_partition_mappings)
        == len(forward_graphs)
        == len(backward_graphs)
        == 1
    ):
        raise RuntimeError(
            "AOT capture did not produce exactly one joint, partition forward/"
            "backward pair, and forward/backward compiler input"
        )
    joint_graph = joint_graphs[0]
    partition_forward_graph = partition_forward_graphs[0]
    partition_backward_graph = partition_backward_graphs[0]
    forward_graph = forward_graphs[0]
    backward_graph = backward_graphs[0]
    joint_partition_mapping = joint_partition_mappings[0]
    forward_partition_tokens = _origin_token_set(partition_forward_graph)
    backward_partition_tokens = _origin_token_set(partition_backward_graph)
    forward_compiler_tokens = _origin_token_set(forward_graph)
    backward_compiler_tokens = _origin_token_set(backward_graph)
    compiler_callback_continuity = {
        "same_aot_module_invocation": True,
        "forward_partition_module_is_compiler_callback_module": (
            partition_forward_graph is forward_graph
        ),
        "backward_partition_module_is_compiler_callback_module": (
            partition_backward_graph is backward_graph
        ),
        "forward_partition_graph_is_compiler_callback_graph": (
            partition_forward_graph.graph is forward_graph.graph
        ),
        "backward_partition_graph_is_compiler_callback_graph": (
            partition_backward_graph.graph is backward_graph.graph
        ),
        "forward_origin_tokens_preserved_to_compiler_callback": (
            forward_partition_tokens == forward_compiler_tokens
        ),
        "backward_origin_tokens_preserved_to_compiler_callback": (
            backward_partition_tokens == backward_compiler_tokens
        ),
    }
    compiler_callback_mapping_checks = _attach_compiler_callback_mapping(
        joint_partition_mapping,
        joint_graph,
        partition_forward_graph,
        partition_backward_graph,
        forward_graph,
        backward_graph,
    )
    compiler_callback_continuity_verified = (
        compiler_callback_continuity["same_aot_module_invocation"]
        and compiler_callback_continuity[
            "forward_origin_tokens_preserved_to_compiler_callback"
        ]
        and compiler_callback_continuity[
            "backward_origin_tokens_preserved_to_compiler_callback"
        ]
        and all(compiler_callback_mapping_checks.values())
    )
    compiler_callback_continuity["continuity_verified"] = (
        compiler_callback_continuity_verified
    )
    joint_partition_mapping["compiler_callback_continuity"] = (
        compiler_callback_continuity
    )
    exact_joint_partition_mapping = all(
        joint_partition_mapping["exact_mapping_checks"].values()
    )
    if not exact_joint_partition_mapping:
        raise AssertionError(
            "joint-to-fw/bw audit token mapping failed: "
            f"{joint_partition_mapping['exact_mapping_checks']}"
        )
    _write_text(
        output_dir / "aot_joint_to_fw_bw_node_mapping.json",
        json.dumps(joint_partition_mapping, ensure_ascii=False, indent=2),
    )
    _write_text(output_dir / "aot_joint.py", _graph_text(joint_graph))
    _write_text(output_dir / "aot_forward.py", _graph_text(forward_graph))
    _write_text(output_dir / "aot_backward.py", _graph_text(backward_graph))

    forward_output = next(
        node for node in forward_graph.graph.nodes if node.op == "output"
    )
    forward_output_leaves = pytree.tree_leaves(forward_output.args[0])
    backward_placeholders = [
        node
        for node in backward_graph.graph.nodes
        if node.op == "placeholder"
    ]
    cross_graph_refs = _cross_graph_node_refs(backward_graph)
    saved_value_leaves = forward_output_leaves[2:]
    if len(backward_placeholders) < len(saved_value_leaves):
        raise AssertionError(
            "backward signature has fewer placeholders than saved forward slots"
        )
    saved_slot_bindings = [
        {
            "saved_slot_index": saved_index,
            "forward_output_index": saved_index + 2,
            "forward_value_node": (
                leaf.name if isinstance(leaf, torch.fx.Node) else repr(leaf)
            ),
            "forward_joint_origin_token": (
                leaf.meta.get(AOT_JOINT_ORIGIN_META_KEY)
                if isinstance(leaf, torch.fx.Node)
                else None
            ),
            "backward_placeholder_index": saved_index,
            "backward_placeholder_node": backward_placeholders[saved_index].name,
            "backward_joint_origin_token": backward_placeholders[
                saved_index
            ].meta.get(AOT_JOINT_ORIGIN_META_KEY),
            "same_joint_origin": (
                isinstance(leaf, torch.fx.Node)
                and isinstance(
                    leaf.meta.get(AOT_JOINT_ORIGIN_META_KEY), str
                )
                and leaf.meta.get(AOT_JOINT_ORIGIN_META_KEY)
                == backward_placeholders[saved_index].meta.get(
                    AOT_JOINT_ORIGIN_META_KEY
                )
            ),
            "transport": (
                "runtime value slot from forward output/context to a fresh "
                "backward placeholder; not a cross-Graph Node edge"
            ),
        }
        for saved_index, leaf in enumerate(saved_value_leaves)
    ]
    saved_slot_bindings_have_same_joint_origins = all(
        binding["same_joint_origin"] for binding in saved_slot_bindings
    )
    if not saved_slot_bindings_have_same_joint_origins:
        raise AssertionError(
            "saved forward slots and positional backward placeholders do not "
            "carry the same joint audit origins"
        )
    aot_abi = {
        "forward_user_output_count": 2,
        "forward_output_slots": [
            leaf.name if isinstance(leaf, torch.fx.Node) else repr(leaf)
            for leaf in forward_output_leaves
        ],
        "saved_value_slots": [
            leaf.name if isinstance(leaf, torch.fx.Node) else repr(leaf)
            for leaf in saved_value_leaves
        ],
        "backward_placeholders": [
            node.name for node in backward_placeholders
        ],
        "saved_slot_bindings": saved_slot_bindings,
        "saved_slot_binding_checks": {
            "all_bindings_have_same_joint_origins": (
                saved_slot_bindings_have_same_joint_origins
            ),
            "binding_uses_positional_runtime_abi": True,
            "binding_is_cross_graph_node_edge": False,
        },
        "cross_graph_node_refs": cross_graph_refs,
        "joint_to_fw_bw_mapping_artifact": (
            "aot_joint_to_fw_bw_node_mapping.json"
        ),
        "explanation": (
            "Forward outputs and backward placeholders form the runtime ABI; "
            "there are no Node references whose owner is the other Graph."
        ),
    }
    _write_text(
        output_dir / "aot_partition_abi.json",
        json.dumps(aot_abi, ensure_ascii=False, indent=2),
    )

    hop_export = torch.export.export(
        HigherOrderBranch(),
        (example, torch.tensor(True)),
    )
    _write_text(
        output_dir / "hop_exported_program.py",
        _graph_text(hop_export.graph_module),
    )
    hop_branch_captured = "higher_order.cond" in str(hop_export.graph)
    try:
        torch.export.export(
            InvalidHigherOrderBranch(),
            (example, torch.tensor(True)),
        )
    except Exception as exc:
        hop_invalid_branch_rejected = (
            "expected same dim" in str(exc)
            or "same number of outputs" in str(exc)
            or "metadata" in str(exc)
        )
    else:
        hop_invalid_branch_rejected = False

    backend_result = _run_codegen_only(
        backend_core,
        (example, model.weight.detach(), model.offset.detach()),
        output_dir / "backend",
        max_fusion_size=64,
    )

    stage_mapping = {
        "mapping_kind": (
            "semantic stage records plus separately referenced exact mappings"
        ),
        "overall_stage_continuity": "partial",
        "continuity_definition": (
            "continuous means the downstream artifact was produced from the "
            "upstream artifact inside the same capture/compile invocation; "
            "running multiple frontends on the same Python computation is an "
            "independent recapture, not a continuous transition"
        ),
        "exact_mapping_artifacts": {
            "aot_joint_to_partition_forward_backward": (
                "aot_joint_to_fw_bw_node_mapping.json"
            ),
            "post_grad_fx_to_generated_code": (
                "backend/inductor_provenance_tracking_node_mappings.json"
            ),
        },
        "stage_run_ids": {
            "symbolic_fx": "symbolic_trace_capture",
            "dynamo_fx": "dynamo_capture_for_eager_backend",
            "exported_program": "export_capture",
            "functional_aten": "functional_make_fx_capture",
            "aot_joint": "aot_autograd_capture_and_partition",
            "aot_forward": "aot_autograd_capture_and_partition",
            "aot_backward": "aot_autograd_capture_and_partition",
            "inductor_ir/scheduler/kernel": "inductor_backend_compile",
        },
        "stages": {
            "symbolic_fx": _node_records(symbolic),
            "dynamo_fx": _node_records(dynamo_graph),
            "exported_program": _node_records(exported.graph_module),
            "functional_aten": _node_records(functional_aten),
            "aot_joint": _node_records(joint_graph),
            "aot_forward": _node_records(forward_graph),
            "aot_backward": _node_records(backward_graph),
        },
        "relationships": [
            {
                "from": "symbolic_fx",
                "to": "functional_aten",
                "relationship": (
                    "independent captures of the same UnifiedGraphModel "
                    "computation"
                ),
                "continuous_transition": False,
                "mechanism": (
                    "functionalization plus a separate make_fx invocation; "
                    "functional_aten was not produced by consuming symbolic_fx"
                ),
                "identity_preserved": False,
            },
            {
                "from": "functional_aten",
                "to": "aot_joint",
                "relationship": (
                    "independent captures of the same UnifiedGraphModel "
                    "computation"
                ),
                "continuous_transition": False,
                "mechanism": (
                    "a separate aot_module invocation recaptured the model and "
                    "synthesized backward; it did not consume functional_aten"
                ),
                "identity_preserved": False,
            },
            {
                "from": "aot_joint",
                "to": "aot_forward/aot_backward",
                "mechanism": "partition copies required nodes into fresh Graphs",
                "continuous_transition": True,
                "identity_preserved": False,
                "evidence": "aot_joint_to_fw_bw_node_mapping.json",
            },
            {
                "from": "aot_forward",
                "to": "backend/fx_graph_readable.py",
                "relationship": (
                    "the backend_core computation is shared, but the backend "
                    "run has explicit state inputs and independently recaptures it"
                ),
                "continuous_transition": False,
                "mechanism": (
                    "_run_codegen_only invokes torch.compile on backend_core; "
                    "it does not consume the AOT forward GraphModule above"
                ),
                "identity_preserved": False,
            },
            {
                "from": "backend/fx_graph_readable.py",
                "to": "inductor_ir/scheduler/kernel",
                "mechanism": (
                    "one _run_codegen_only torch.compile invocation performs "
                    "GraphLowering, scheduling, fusion, and code generation"
                ),
                "continuous_transition": True,
                "identity_preserved": False,
                "evidence": "backend/inductor_provenance_tracking_node_mappings.json",
            },
        ],
        "backend": {
            "codegen_status": "generated_not_executed",
            "captured_cpp_translation_units": backend_result[
                "captured_cpp_translation_units"
            ],
            "trace_files": backend_result["copied_trace_files"],
        },
    }
    _write_text(
        output_dir / "stage_node_mapping.json",
        json.dumps(stage_mapping, ensure_ascii=False, indent=2),
    )

    artifact_manifest = {
        "schema_version": 1,
        "bundle_kind": (
            "multi-run evidence bundle for one stable user computation prefix"
        ),
        "overall_stage_continuity": "partial",
        "claims": {
            "manifested_artifacts_generated_or_refreshed_in_one_script_invocation": (
                True
            ),
            "all_frontend_artifacts_form_one_continuous_compile": False,
            "independent_stable_prefix_frontends_capture_same_user_computation": (
                True
            ),
            "aot_joint_partition_fw_bw_is_one_continuous_run": True,
            "inductor_trace_to_ir_scheduler_codegen_is_one_continuous_run": True,
            "aot_forward_is_direct_input_to_recorded_inductor_run": False,
            "native_generated_kernel_executed": False,
        },
        "continuous_segments": [
            {
                "run_id": "aot_autograd_capture_and_partition",
                "artifacts": [
                    "aot_joint.py",
                    "aot_forward.py",
                    "aot_backward.py",
                    "aot_joint_to_fw_bw_node_mapping.json",
                    "aot_partition_abi.json",
                ],
                "evidence": {
                    "exact_old_to_new_mapping": (
                        "aot_joint_to_fw_bw_node_mapping.json"
                    ),
                    "runtime_slot_abi": "aot_partition_abi.json",
                },
            },
            {
                "run_id": "inductor_backend_compile",
                "input_scope": (
                    "backend_core with parameter and buffer values passed as "
                    "explicit tensor inputs"
                ),
                "artifacts": [
                    "backend/fx_graph_readable.py",
                    "backend/fx_graph_transformed.py",
                    "backend/ir_pre_fusion.txt",
                    "backend/ir_post_fusion.txt",
                    "backend/output_code.py",
                    "backend/captured_cpp_kernel.cpp",
                    (
                        "backend/"
                        "inductor_provenance_tracking_node_mappings.json"
                    ),
                ],
                "evidence_boundary": backend_result.get("evidence_boundary", {}),
            },
        ],
        "independent_capture_runs": [
            {
                "run_id": "symbolic_trace_capture",
                "artifact": "symbolic_fx.py",
            },
            {
                "run_id": "dynamo_capture_for_eager_backend",
                "artifacts": ["dynamo_fx.py", "dynamo_guards.txt"],
            },
            {
                "run_id": "export_capture",
                "artifacts": [
                    "exported_program.py",
                    "export_graph_signature.json",
                ],
            },
            {
                "run_id": "functional_make_fx_capture",
                "artifact": "functional_aten.py",
            },
            {
                "run_id": "hop_export_capture",
                "artifact": "hop_exported_program.py",
            },
        ],
        "missing_continuous_edges": [
            {
                "from": "symbolic_fx",
                "to": "functional_aten",
                "reason": "functional_aten is a separate make_fx recapture",
            },
            {
                "from": "functional_aten",
                "to": "aot_joint",
                "reason": "aot_module performs a separate capture",
            },
            {
                "from": "aot_forward",
                "to": "backend/fx_graph_readable.py",
                "reason": (
                    "the recorded backend run compiles backend_core directly "
                    "instead of consuming the captured AOT forward GraphModule"
                ),
            },
        ],
        "index_artifacts": {
            "semantic_stage_records": "stage_node_mapping.json",
            "environment": "environment.json",
            "model_contract": "model_contract.json",
        },
    }
    _write_text(
        output_dir / "artifact_manifest.json",
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2),
    )

    forward_matches = _tree_allclose(dynamo_output, eager_output)
    gradient_matches = (
        torch.allclose(aot_input.grad, eager_input.grad)
        and torch.allclose(aot_model.weight.grad, eager_model_for_grad.weight.grad)
    )
    contract_checks = {
        "forward_matches": forward_matches,
        "gradient_matches": gradient_matches,
        "dynamic_export_has_range_constraints": bool(
            exported.range_constraints
        ),
        "export_out_of_range_rejected": export_out_of_range_rejected,
        "aot_has_joint_forward_backward": True,
        "aot_cross_graph_node_refs": len(cross_graph_refs) == 0,
        "aot_joint_partition_mapping_exact": exact_joint_partition_mapping,
        "aot_partition_to_compiler_callback_continuity": (
            compiler_callback_continuity_verified
        ),
        "aot_saved_slot_binding_origins_match": (
            saved_slot_bindings_have_same_joint_origins
        ),
        "backend_codegen_status_generated_not_executed": True,
        "hop_branch_captured": hop_branch_captured,
        "hop_invalid_branch_rejected": hop_invalid_branch_rejected,
    }
    if not all(contract_checks.values()):
        raise AssertionError(f"unified artifact contract failed: {contract_checks}")
    environment = {
        "audit_source_baseline": SOURCE_BASELINE,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda_available": torch.cuda.is_available(),
        "msvc_cl": shutil.which("cl"),
        "backend_codegen_status": "generated_not_executed",
    }
    _write_text(
        output_dir / "environment.json",
        json.dumps(environment, ensure_ascii=False, indent=2),
    )

    print(f"forward_matches={forward_matches}")
    print(f"gradient_matches={gradient_matches}")
    print(
        "dynamic_export_has_range_constraints="
        + str(bool(exported.range_constraints))
    )
    print(f"export_out_of_range_rejected={export_out_of_range_rejected}")
    print("aot_has_joint_forward_backward=True")
    print(f"aot_cross_graph_node_refs={len(cross_graph_refs)}")
    print(
        "aot_joint_partition_mapping_exact="
        + str(exact_joint_partition_mapping)
    )
    print(
        "aot_partition_to_compiler_callback_continuity="
        + str(compiler_callback_continuity_verified)
    )
    print(
        "aot_saved_slot_binding_origins_match="
        + str(saved_slot_bindings_have_same_joint_origins)
    )
    print("artifact_bundle_continuity=partial")
    print("backend_codegen_status=generated_not_executed")
    print(f"hop_branch_captured={hop_branch_captured}")
    print(f"hop_invalid_branch_rejected={hop_invalid_branch_rejected}")
    print(f"artifact_dir={output_dir.as_posix()}")


if __name__ == "__main__":
    main()
