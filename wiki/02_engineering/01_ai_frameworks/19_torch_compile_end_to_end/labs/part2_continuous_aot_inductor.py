import argparse
import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch._inductor import config
from torch._inductor.custom_graph_pass import CustomPartitionerFn
from torch._inductor.virtualized import V


SOURCE_BASELINE = "e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52"
RUN_ID = "continuous-aot-inductor-extern-mm-seed-0"
TOKEN_KEY = "graph_series_continuity_token"


def model(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mm(x, weight)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def _graph_text(graph_module: torch.fx.GraphModule) -> str:
    return (
        f"# module_id={id(graph_module)}\n"
        f"# graph_id={id(graph_module.graph)}\n"
        f"# owner_id={id(graph_module.graph.owning_module)}\n\n"
        f"{graph_module.graph}\n\n# Generated Python\n{graph_module.code}"
    )


def _tokens(graph_module: torch.fx.GraphModule) -> list[str]:
    return sorted(
        str(node.meta[TOKEN_KEY])
        for node in graph_module.graph.nodes
        if TOKEN_KEY in node.meta
    )


def _graph_record(
    graph_module: torch.fx.GraphModule,
    *,
    role: str,
) -> dict[str, object]:
    owner = graph_module.graph.owning_module
    return {
        "role": role,
        "module_id": id(graph_module),
        "graph_id": id(graph_module.graph),
        "owner_id": id(owner),
        "owner_is_graph_module": owner is graph_module,
        "tokens": _tokens(graph_module),
    }


class RecordingPartitioner(CustomPartitionerFn):
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = events
        self.forward_module: torch.fx.GraphModule | None = None
        self.backward_module: torch.fx.GraphModule | None = None
        self.forward_record: dict[str, object] | None = None
        self.backward_record: dict[str, object] | None = None

    def uuid(self) -> None:
        return None

    def __call__(
        self,
        joint_module: torch.fx.GraphModule,
        joint_inputs: list[object],
        **kwargs: Any,
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        for index, node in enumerate(joint_module.graph.nodes):
            node.meta[TOKEN_KEY] = f"{RUN_ID}:joint:{index}:{node.name}"
        forward_module, backward_module = min_cut_rematerialization_partition(
            joint_module,
            joint_inputs,
            **kwargs,
        )
        for role, graph_module in (
            ("forward", forward_module),
            ("backward", backward_module),
        ):
            for index, node in enumerate(graph_module.graph.nodes):
                node.meta.setdefault(
                    TOKEN_KEY,
                    f"{RUN_ID}:{role}:structural:{index}:{node.name}",
                )
        self.forward_module = forward_module
        self.backward_module = backward_module
        self.forward_record = _graph_record(forward_module, role="forward")
        self.backward_record = _graph_record(backward_module, role="backward")
        self.events.append(
            {
                "run_id": RUN_ID,
                "phase": "partition_return",
                "forward": self.forward_record,
                "backward": self.backward_record,
            }
        )
        return forward_module, backward_module


def _artifact_records(output_dir: Path, names: list[str]) -> list[dict[str, str]]:
    return [
        {
            "path": name,
            "sha256": hashlib.sha256((output_dir / name).read_bytes()).hexdigest(),
        }
        for name in names
    ]


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    compile_fx_module = importlib.import_module("torch._inductor.compile_fx")
    scheduler_module = importlib.import_module("torch._inductor.scheduler")
    events: list[dict[str, object]] = []
    compiler_records: dict[str, dict[str, object]] = {}
    compiler_modules: dict[str, torch.fx.GraphModule] = {}
    lowering_records: dict[str, dict[str, object]] = {}
    scheduler_records: dict[str, list[dict[str, object]]] = {}
    partitioner = RecordingPartitioner(events)

    def recording_inner_compile(
        graph_module: torch.fx.GraphModule,
        example_inputs: list[object],
        **kwargs: Any,
    ):
        role = "backward" if kwargs.get("is_backward", False) else "forward"
        record = _graph_record(graph_module, role=role)
        compiler_records[role] = record
        compiler_modules[role] = graph_module
        events.append(
            {
                "run_id": RUN_ID,
                "phase": "compiler_callback_inner_compile_entry",
                **record,
            }
        )
        return compile_fx_module.compile_fx_inner(
            graph_module,
            example_inputs,
            **kwargs,
        )

    def inductor_backend(
        graph_module: torch.fx.GraphModule,
        example_inputs: list[object],
    ):
        events.append(
            {
                "run_id": RUN_ID,
                "phase": "dynamo_backend_compile_fx_entry",
                "graph_id": id(graph_module.graph),
            }
        )
        return compile_fx_module.compile_fx(
            graph_module,
            example_inputs,
            inner_compile=recording_inner_compile,
        )

    original_graph_lowering_init = compile_fx_module.GraphLowering.__init__

    def recording_graph_lowering_init(
        lowering_self,
        graph_module: torch.fx.GraphModule,
        *init_args: object,
        **init_kwargs: object,
    ) -> None:
        role = "backward" if init_kwargs.get("is_backward", False) else "forward"
        record = _graph_record(graph_module, role=role)
        lowering_records[role] = record
        events.append(
            {
                "run_id": RUN_ID,
                "phase": "graph_lowering_init",
                **record,
            }
        )
        original_graph_lowering_init(
            lowering_self,
            graph_module,
            *init_args,
            **init_kwargs,
        )

    original_scheduler_init = scheduler_module.Scheduler.__init__

    def recording_scheduler_init(
        scheduler_self,
        nodes: list[object],
    ) -> None:
        graph_module = V.graph.module
        assert isinstance(graph_module, torch.fx.GraphModule)
        role = "backward" if V.graph.is_backward else "forward"
        record = _graph_record(graph_module, role=role)
        origin_tokens = sorted(
            {
                str(origin.meta[TOKEN_KEY])
                for operation in nodes
                for origin in operation.get_origins()
                if isinstance(origin, torch.fx.Node)
                and TOKEN_KEY in origin.meta
            }
        )
        record.update(
            {
                "graph_lowering_copy_module_id": id(V.graph.orig_gm),
                "owner_is_graph_lowering_copy": (
                    graph_module.graph.owning_module is V.graph.orig_gm
                ),
                "input_operation_count": len(nodes),
                "origin_tokens": origin_tokens,
            }
        )
        original_scheduler_init(scheduler_self, nodes)
        record.update(
            {
                "scheduler_node_count": len(scheduler_self.nodes),
                "read_dependency_count": sum(
                    len(node.read_writes.reads)
                    for node in scheduler_self.nodes
                ),
                "write_dependency_count": sum(
                    len(node.read_writes.writes)
                    for node in scheduler_self.nodes
                ),
                "unmet_dependency_count": sum(
                    len(node.unmet_dependencies)
                    for node in scheduler_self.nodes
                ),
            }
        )
        scheduler_records.setdefault(role, []).append(record)
        events.append(
            {
                "run_id": RUN_ID,
                "phase": "scheduler_dependency_construction_complete",
                **record,
            }
        )

    base_x = torch.randn(4, 8)
    base_weight = torch.randn(8, 6)
    eager_x = base_x.detach().clone().requires_grad_(True)
    eager_weight = base_weight.detach().clone().requires_grad_(True)
    expected = model(eager_x, eager_weight)
    expected.backward(torch.ones_like(expected))

    original_partitioner = config.custom_partitioner_fn
    original_force_disable_caches = config.force_disable_caches
    torch._dynamo.reset()
    x = base_x.detach().clone().requires_grad_(True)
    weight = base_weight.detach().clone().requires_grad_(True)
    compile_fx_module.GraphLowering.__init__ = recording_graph_lowering_init
    scheduler_module.Scheduler.__init__ = recording_scheduler_init
    try:
        with config.patch(
            {
                "custom_partitioner_fn": partitioner,
                "force_disable_caches": True,
            }
        ):
            compiled = torch.compile(
                model,
                backend=inductor_backend,
                fullgraph=True,
            )
            actual = compiled(x, weight)
            actual.backward(torch.ones_like(actual))
    finally:
        compile_fx_module.GraphLowering.__init__ = original_graph_lowering_init
        scheduler_module.Scheduler.__init__ = original_scheduler_init

    assert partitioner.forward_module is not None
    assert partitioner.backward_module is not None
    assert partitioner.forward_record is not None
    assert partitioner.backward_record is not None
    partition_forward = partitioner.forward_record
    partition_backward = partitioner.backward_record
    compiler_forward = compiler_records["forward"]
    compiler_backward = compiler_records["backward"]
    compiler_forward_module = compiler_modules["forward"]
    compiler_backward_module = compiler_modules["backward"]
    lowering_forward = lowering_records["forward"]
    lowering_backward = lowering_records["backward"]
    assert set(scheduler_records) == {"forward", "backward"}
    assert len(scheduler_records["forward"]) == 1
    assert len(scheduler_records["backward"]) == 1
    scheduler_forward = scheduler_records["forward"][0]
    scheduler_backward = scheduler_records["backward"][0]

    forward_tokens_preserved = (
        partition_forward["tokens"]
        == compiler_forward["tokens"]
        == lowering_forward["tokens"]
        == scheduler_forward["tokens"]
    )
    backward_tokens_preserved = (
        partition_backward["tokens"]
        == compiler_backward["tokens"]
        == lowering_backward["tokens"]
        == scheduler_backward["tokens"]
    )
    forward_graph_identity_preserved = (
        partition_forward["graph_id"]
        == compiler_forward["graph_id"]
        == lowering_forward["graph_id"]
    )
    forward_owner_identity_preserved = (
        partition_forward["module_id"]
        == compiler_forward["module_id"]
        == lowering_forward["module_id"]
        and partition_forward["owner_id"]
        == compiler_forward["owner_id"]
        == lowering_forward["owner_id"]
        and partition_forward["owner_is_graph_module"]
        and compiler_forward["owner_is_graph_module"]
        and lowering_forward["owner_is_graph_module"]
    )
    backward_partition_to_callback_identity_transition = (
        partition_backward["module_id"] != compiler_backward["module_id"]
        and partition_backward["graph_id"] != compiler_backward["graph_id"]
        and partition_backward["owner_id"] != compiler_backward["owner_id"]
    )
    backward_callback_to_lowering_identity_preserved = (
        compiler_backward["module_id"] == lowering_backward["module_id"]
        and compiler_backward["graph_id"] == lowering_backward["graph_id"]
        and compiler_backward["owner_id"] == lowering_backward["owner_id"]
        and compiler_backward["owner_is_graph_module"]
        and lowering_backward["owner_is_graph_module"]
    )
    forward_callback_to_scheduler_module_graph_preserved = (
        compiler_forward["module_id"] == scheduler_forward["module_id"]
        and compiler_forward["graph_id"] == scheduler_forward["graph_id"]
    )
    backward_callback_to_scheduler_module_graph_preserved = (
        compiler_backward["module_id"] == scheduler_backward["module_id"]
        and compiler_backward["graph_id"] == scheduler_backward["graph_id"]
    )
    scheduler_owner_transition_recorded = all(
        scheduler_record["owner_id"] != compiler_record["owner_id"]
        and scheduler_record["owner_id"]
        == scheduler_record["graph_lowering_copy_module_id"]
        and scheduler_record["owner_is_graph_lowering_copy"]
        and not scheduler_record["owner_is_graph_module"]
        for scheduler_record, compiler_record in (
            (scheduler_forward, compiler_forward),
            (scheduler_backward, compiler_backward),
        )
    )
    scheduler_origin_tokens_recorded = all(
        bool(scheduler_record["origin_tokens"])
        and set(scheduler_record["origin_tokens"]).issubset(
            scheduler_record["tokens"]
        )
        for scheduler_record in (scheduler_forward, scheduler_backward)
    )
    scheduler_dependency_construction_recorded = all(
        scheduler_record["input_operation_count"] > 0
        and scheduler_record["scheduler_node_count"] > 0
        and scheduler_record["read_dependency_count"] > 0
        and scheduler_record["write_dependency_count"] > 0
        for scheduler_record in (scheduler_forward, scheduler_backward)
    )
    compiler_callback_recorded = set(compiler_records) == {"forward", "backward"}
    graph_lowering_recorded = set(lowering_records) == {"forward", "backward"}
    scheduler_recorded = set(scheduler_records) == {"forward", "backward"}
    forward_matches = torch.allclose(actual, expected.detach())
    gradient_matches = (
        torch.allclose(x.grad, eager_x.grad)
        and torch.allclose(weight.grad, eager_weight.grad)
    )
    graph_lowering_wrapper_restored = (
        compile_fx_module.GraphLowering.__init__ is original_graph_lowering_init
    )
    scheduler_wrapper_restored = (
        scheduler_module.Scheduler.__init__ is original_scheduler_init
    )
    config_restored = (
        config.custom_partitioner_fn is original_partitioner
        and config.force_disable_caches is original_force_disable_caches
    )
    continuous = (
        forward_tokens_preserved
        and forward_graph_identity_preserved
        and forward_owner_identity_preserved
        and forward_callback_to_scheduler_module_graph_preserved
        and scheduler_owner_transition_recorded
        and scheduler_origin_tokens_recorded
        and scheduler_dependency_construction_recorded
        and compiler_callback_recorded
        and graph_lowering_recorded
        and scheduler_recorded
    )

    checks = {
        "continuous_aot_forward_to_inductor": continuous,
        "partition_forward_tokens_preserved": forward_tokens_preserved,
        "partition_backward_tokens_preserved": backward_tokens_preserved,
        "forward_graph_identity_preserved": forward_graph_identity_preserved,
        "forward_owner_identity_preserved": forward_owner_identity_preserved,
        "backward_partition_to_callback_identity_transition": (
            backward_partition_to_callback_identity_transition
        ),
        "backward_callback_to_lowering_identity_preserved": (
            backward_callback_to_lowering_identity_preserved
        ),
        "forward_callback_to_scheduler_module_graph_preserved": (
            forward_callback_to_scheduler_module_graph_preserved
        ),
        "backward_callback_to_scheduler_module_graph_preserved": (
            backward_callback_to_scheduler_module_graph_preserved
        ),
        "scheduler_owner_transition_recorded": (
            scheduler_owner_transition_recorded
        ),
        "scheduler_origin_tokens_recorded": scheduler_origin_tokens_recorded,
        "scheduler_dependency_construction_recorded": (
            scheduler_dependency_construction_recorded
        ),
        "compiler_callback_recorded": compiler_callback_recorded,
        "graph_lowering_recorded": graph_lowering_recorded,
        "scheduler_recorded": scheduler_recorded,
        "forward_matches": bool(forward_matches),
        "gradient_matches": bool(gradient_matches),
        "native_kernel_executed": False,
        "evidence_scope": "extern_matmul_only",
    }
    assert checks == {
        "continuous_aot_forward_to_inductor": True,
        "partition_forward_tokens_preserved": True,
        "partition_backward_tokens_preserved": True,
        "forward_graph_identity_preserved": True,
        "forward_owner_identity_preserved": True,
        "backward_partition_to_callback_identity_transition": True,
        "backward_callback_to_lowering_identity_preserved": True,
        "forward_callback_to_scheduler_module_graph_preserved": True,
        "backward_callback_to_scheduler_module_graph_preserved": True,
        "scheduler_owner_transition_recorded": True,
        "scheduler_origin_tokens_recorded": True,
        "scheduler_dependency_construction_recorded": True,
        "compiler_callback_recorded": True,
        "graph_lowering_recorded": True,
        "scheduler_recorded": True,
        "forward_matches": True,
        "gradient_matches": True,
        "native_kernel_executed": False,
        "evidence_scope": "extern_matmul_only",
    }, checks
    assert graph_lowering_wrapper_restored
    assert scheduler_wrapper_restored
    assert config_restored

    forward = {
        "partition_module_id": partition_forward["module_id"],
        "partition_graph_id": partition_forward["graph_id"],
        "partition_owner_id": partition_forward["owner_id"],
        "partition_tokens": partition_forward["tokens"],
        "compiler_callback_module_id": compiler_forward["module_id"],
        "compiler_callback_graph_id": compiler_forward["graph_id"],
        "compiler_callback_owner_id": compiler_forward["owner_id"],
        "compiler_callback_tokens": compiler_forward["tokens"],
        "graph_lowering_module_id": lowering_forward["module_id"],
        "graph_lowering_graph_id": lowering_forward["graph_id"],
        "graph_lowering_owner_id": lowering_forward["owner_id"],
        "graph_lowering_tokens": lowering_forward["tokens"],
        "scheduler_module_id": scheduler_forward["module_id"],
        "scheduler_graph_id": scheduler_forward["graph_id"],
        "scheduler_owner_id": scheduler_forward["owner_id"],
        "scheduler_graph_lowering_copy_module_id": scheduler_forward[
            "graph_lowering_copy_module_id"
        ],
        "scheduler_owner_is_graph_lowering_copy": scheduler_forward[
            "owner_is_graph_lowering_copy"
        ],
        "scheduler_module_tokens": scheduler_forward["tokens"],
        "scheduler_origin_tokens": scheduler_forward["origin_tokens"],
        "scheduler_input_operation_count": scheduler_forward[
            "input_operation_count"
        ],
        "scheduler_node_count": scheduler_forward["scheduler_node_count"],
        "scheduler_read_dependency_count": scheduler_forward[
            "read_dependency_count"
        ],
        "scheduler_write_dependency_count": scheduler_forward[
            "write_dependency_count"
        ],
        "scheduler_unmet_dependency_count": scheduler_forward[
            "unmet_dependency_count"
        ],
    }
    backward = {
        "partition_module_id": partition_backward["module_id"],
        "partition_graph_id": partition_backward["graph_id"],
        "partition_owner_id": partition_backward["owner_id"],
        "partition_tokens": partition_backward["tokens"],
        "compiler_callback_module_id": compiler_backward["module_id"],
        "compiler_callback_graph_id": compiler_backward["graph_id"],
        "compiler_callback_owner_id": compiler_backward["owner_id"],
        "compiler_callback_tokens": compiler_backward["tokens"],
        "graph_lowering_module_id": lowering_backward["module_id"],
        "graph_lowering_graph_id": lowering_backward["graph_id"],
        "graph_lowering_owner_id": lowering_backward["owner_id"],
        "graph_lowering_tokens": lowering_backward["tokens"],
        "scheduler_module_id": scheduler_backward["module_id"],
        "scheduler_graph_id": scheduler_backward["graph_id"],
        "scheduler_owner_id": scheduler_backward["owner_id"],
        "scheduler_graph_lowering_copy_module_id": scheduler_backward[
            "graph_lowering_copy_module_id"
        ],
        "scheduler_owner_is_graph_lowering_copy": scheduler_backward[
            "owner_is_graph_lowering_copy"
        ],
        "scheduler_module_tokens": scheduler_backward["tokens"],
        "scheduler_origin_tokens": scheduler_backward["origin_tokens"],
        "scheduler_input_operation_count": scheduler_backward[
            "input_operation_count"
        ],
        "scheduler_node_count": scheduler_backward["scheduler_node_count"],
        "scheduler_read_dependency_count": scheduler_backward[
            "read_dependency_count"
        ],
        "scheduler_write_dependency_count": scheduler_backward[
            "write_dependency_count"
        ],
        "scheduler_unmet_dependency_count": scheduler_backward[
            "unmet_dependency_count"
        ],
        "identity_note": (
            "This run rebuilt the backward module/Graph before the compiler "
            "callback. GraphLowering then shallow-copied that module, retaining "
            "the Graph while transferring graph.owning_module to orig_gm before "
            "Scheduler dependency construction."
        ),
    }
    continuity = {
        "run_id": RUN_ID,
        "evidence_scope": "extern_matmul_only",
        "forward": forward,
        "backward": backward,
        "events": events,
        "checks": checks,
        "mock_compiler_used": False,
        "native_kernel_executed": False,
        "native_kernel_boundary": (
            "The observed path executes real Inductor around ATen extern mm calls. "
            "It does not compile or execute an Inductor-generated C++ kernel."
        ),
        "graph_lowering_wrapper_restored": graph_lowering_wrapper_restored,
        "scheduler_wrapper_restored": scheduler_wrapper_restored,
        "config_restored": config_restored,
    }
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
    _write_text(
        output_dir / "partition_forward.py",
        _graph_text(partitioner.forward_module),
    )
    _write_text(
        output_dir / "partition_backward.py",
        _graph_text(partitioner.backward_module),
    )
    _write_text(
        output_dir / "compiler_forward.py",
        _graph_text(compiler_forward_module),
    )
    _write_text(
        output_dir / "compiler_backward.py",
        _graph_text(compiler_backward_module),
    )
    _write_json(output_dir / "continuity.json", continuity)
    artifact_names = [
        "environment.json",
        "partition_forward.py",
        "partition_backward.py",
        "compiler_forward.py",
        "compiler_backward.py",
        "continuity.json",
    ]
    manifest = {
        "entrypoint": Path(__file__).resolve().as_posix(),
        "command": [sys.executable, *sys.argv],
        "run_id": RUN_ID,
        "evidence_scope": "extern_matmul_only",
        "artifacts": _artifact_records(output_dir, artifact_names),
        "checks": checks,
    }
    _write_json(output_dir / "manifest.json", manifest)

    for key, value in checks.items():
        print(f"{key}={value}")
    print(f"artifact_dir={output_dir.as_posix()}")


if __name__ == "__main__":
    main()
