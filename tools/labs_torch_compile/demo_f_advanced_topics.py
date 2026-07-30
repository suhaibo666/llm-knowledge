"""Volume F demos: compiled autograd, distributed graphs and extension paths."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from demo_harness import CaseSpec, DemoContext, run_volume_cli


def _device(context: DemoContext) -> torch.device:
    return torch.device(context.device)


def compiled_autograd(context: DemoContext) -> dict[str, object]:
    from torch._dynamo import compiled_autograd as compiled_autograd_api

    captured_graphs: list[torch.fx.GraphModule] = []

    def backend(
        gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ):
        captured_graphs.append(gm)
        return gm.forward

    autograd_compiler = torch.compile(backend=backend, fullgraph=True)
    device = _device(context)
    x = torch.randn(8, device=device, requires_grad=True)
    eager_x = x.detach().clone().requires_grad_(True)

    def loss_fn(value: torch.Tensor) -> torch.Tensor:
        first = torch.sin(value)
        second = torch.cos(first)
        return (first * second).sum()

    with compiled_autograd_api._enable(autograd_compiler, dynamic=False):
        loss_fn(x).backward()
    loss_fn(eager_x).backward()
    torch.testing.assert_close(x.grad, eager_x.grad)
    if not captured_graphs:
        raise AssertionError("compiled autograd did not submit an FX graph")
    (context.output_dir / "compiled_autograd_graph.py").write_text(
        captured_graphs[0].code, encoding="utf-8"
    )
    return {
        "captured_backward_graph_count": len(captured_graphs),
        "captured_node_count": len(list(captured_graphs[0].graph.nodes)),
        "gradient_matches": True,
        "scope": "autograd engine backward execution",
    }


def checkpoint_recompute(context: DemoContext) -> dict[str, object]:
    device = _device(context)
    baseline_calls = 0
    checkpoint_calls = 0

    def baseline_block(x: torch.Tensor) -> torch.Tensor:
        nonlocal baseline_calls
        baseline_calls += 1
        return torch.sin(x) * torch.cos(x)

    def checkpoint_block(x: torch.Tensor) -> torch.Tensor:
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        return torch.sin(x) * torch.cos(x)

    baseline_x = torch.randn(32, device=device, requires_grad=True)
    checkpoint_x = baseline_x.detach().clone().requires_grad_(True)
    baseline_block(baseline_x).sum().backward()
    checkpoint(
        checkpoint_block,
        checkpoint_x,
        use_reentrant=False,
    ).sum().backward()
    torch.testing.assert_close(baseline_x.grad, checkpoint_x.grad)

    def compiled_fn(x: torch.Tensor) -> torch.Tensor:
        return checkpoint(
            lambda value: torch.sin(value) * torch.cos(value),
            x,
            use_reentrant=False,
        )

    compiled_x = baseline_x.detach().clone().requires_grad_(True)
    compiled = torch.compile(compiled_fn, backend="eager", fullgraph=True)
    compiled(compiled_x).sum().backward()
    torch.testing.assert_close(baseline_x.grad, compiled_x.grad)
    return {
        "baseline_forward_calls": baseline_calls,
        "checkpoint_forward_calls": checkpoint_calls,
        "recompute_observed": checkpoint_calls > baseline_calls,
        "gradient_matches": True,
        "compiled_checkpoint_gradient_matches": True,
        "use_reentrant": False,
    }


def ddp_compile(context: DemoContext) -> dict[str, object]:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    initialized_here = False
    if not dist.is_initialized():
        rendezvous = context.output_dir / "ddp_rendezvous"
        dist.init_process_group(
            backend="nccl" if context.device == "cuda" else "gloo",
            init_method=rendezvous.resolve().as_uri(),
            rank=0,
            world_size=1,
        )
        initialized_here = True
    graphs: list[torch.fx.GraphModule] = []

    def backend(
        gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ):
        graphs.append(gm)
        return gm.forward

    try:
        device = _device(context)
        if device.type == "cuda":
            torch.cuda.set_device(0)
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        ).to(device)
        ddp = DistributedDataParallel(
            model,
            device_ids=[0] if device.type == "cuda" else None,
        )
        compiled = torch.compile(ddp, backend=backend)
        x = torch.randn(4, 8, device=device)
        loss = compiled(x).square().mean()
        loss.backward()
        gradients_present = all(
            parameter.grad is not None for parameter in model.parameters()
        )
        return {
            "world_size": dist.get_world_size(),
            "backend": dist.get_backend(),
            "captured_graph_count": len(graphs),
            "gradients_present": gradients_present,
            "compile_boundary": "DistributedDataParallel module",
        }
    finally:
        if initialized_here:
            dist.destroy_process_group()


def _fsdp_dtensor_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    output_dir: str,
) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cuda", (world_size,))
        model = nn.Linear(8, 8, bias=False).cuda(rank)
        fully_shard(model, mesh=mesh)
        compiled = torch.compile(model, backend="eager")
        x = torch.randn(4, 8, device=f"cuda:{rank}")
        compiled(x).sum().backward()
        parameter = next(model.parameters())
        placement_names = [
            type(placement).__name__ for placement in parameter.placements
        ]
        Path(output_dir, f"rank_{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "parameter_type": type(parameter).__name__,
                    "placements": placement_names,
                    "local_shape": list(parameter.to_local().shape),
                    "gradient_present": parameter.grad is not None,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


def fsdp_dtensor(context: DemoContext) -> dict[str, object]:
    import torch.multiprocessing as mp

    world_size = 2
    output_dir = context.output_dir / "ranks"
    output_dir.mkdir(parents=True, exist_ok=True)
    rendezvous_file = context.output_dir / "fsdp_rendezvous"
    rendezvous = rendezvous_file.resolve().as_uri()
    mp.spawn(
        _fsdp_dtensor_worker,
        args=(world_size, rendezvous, str(output_dir)),
        nprocs=world_size,
        join=True,
    )
    rank_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_dir.glob("rank_*.json"))
    ]
    if len(rank_records) != world_size:
        raise AssertionError("not every FSDP rank emitted evidence")
    return {
        "world_size": world_size,
        "rank_records": rank_records,
        "all_parameters_are_dtensor": all(
            record["parameter_type"] == "DTensor" for record in rank_records
        ),
        "all_gradients_present": all(
            record["gradient_present"] for record in rank_records
        ),
    }


_COURSE_CUSTOM_OP = None


def _course_custom_op():
    global _COURSE_CUSTOM_OP
    if _COURSE_CUSTOM_OP is not None:
        return _COURSE_CUSTOM_OP

    @torch.library.custom_op(
        "torch_compile_course::scaled_sin",
        mutates_args=(),
    )
    def scaled_sin(x: torch.Tensor, scale: float) -> torch.Tensor:
        return torch.sin(x) * scale

    @scaled_sin.register_fake
    def scaled_sin_fake(
        x: torch.Tensor, scale: float
    ) -> torch.Tensor:
        return torch.empty_like(x)

    def setup_context(ctx, inputs, output) -> None:
        x, scale = inputs
        ctx.save_for_backward(x)
        ctx.scale = scale

    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        return grad_output * torch.cos(x) * ctx.scale, None

    scaled_sin.register_autograd(backward, setup_context=setup_context)
    _COURSE_CUSTOM_OP = scaled_sin
    return scaled_sin


def custom_op_contract(context: DemoContext) -> dict[str, object]:
    operation = _course_custom_op()
    device = _device(context)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return operation(x, 2.5).sum()

    x = torch.randn(8, device=device, requires_grad=True)
    eager_x = x.detach().clone().requires_grad_(True)
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    actual = compiled(x)
    expected = fn(eager_x)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, eager_x.grad)
    fake_x = torch.empty(8, device="meta")
    fake_output = operation(fake_x, 2.5)
    return {
        "qualified_name": "torch_compile_course::scaled_sin",
        "fake_kernel_shape": list(fake_output.shape),
        "fake_kernel_device": str(fake_output.device),
        "compiled_forward_matches": True,
        "registered_autograd_gradient_matches": True,
    }


def custom_backend(context: DemoContext) -> dict[str, object]:
    compile_records: list[dict[str, object]] = []

    def backend(
        gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ):
        call_targets = [
            str(node.target)
            for node in gm.graph.nodes
            if node.op == "call_function"
        ]
        compile_records.append(
            {
                "node_count": len(list(gm.graph.nodes)),
                "call_targets": call_targets,
                "example_shapes": [
                    list(value.shape)
                    for value in example_inputs
                    if isinstance(value, torch.Tensor)
                ],
            }
        )
        return gm.forward

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.sin(x) + 1)

    torch._dynamo.reset()
    device = _device(context)
    x = torch.randn(8, device=device)
    compiled = torch.compile(fn, backend=backend, fullgraph=True)
    first = compiled(x)
    second = compiled(x + 1)
    torch.testing.assert_close(first, fn(x))
    torch.testing.assert_close(second, fn(x + 1))
    (context.output_dir / "backend_contract.json").write_text(
        json.dumps(compile_records, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return {
        "compile_count": len(compile_records),
        "output_matches": True,
        "second_call_reused": len(compile_records) == 1,
        "backend_return_contract": "callable",
    }


def aotinductor_package(context: DemoContext) -> dict[str, object]:
    class Model(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(torch.sin(x) + 1)

    device = _device(context)
    model = Model().eval().to(device)
    x = torch.randn(8, device=device)
    exported = torch.export.export(model, (x,))
    package_path = context.output_dir / "model.pt2"
    actual_package = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=package_path,
    )
    loaded = torch._inductor.aoti_load_package(actual_package)
    actual = loaded(x)
    expected = model(x)
    torch.testing.assert_close(actual, expected)
    return {
        "export_node_count": len(list(exported.graph.nodes)),
        "package_path": str(actual_package),
        "package_size_bytes": Path(actual_package).stat().st_size,
        "loaded_type": type(loaded).__name__,
        "output_matches": True,
    }


def inference_freezing_cudagraph(context: DemoContext) -> dict[str, object]:
    from torch._inductor import config

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(128, 128)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(x))

    device = _device(context)
    model = Model().eval().to(device)
    x = torch.randn(64, 128, device=device)
    torch._dynamo.reset()
    with torch.no_grad(), config.patch(
        {"freezing": True, "triton.cudagraphs": True}
    ):
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        torch.compiler.cudagraph_mark_step_begin()
        warm = compiled(x)
        torch.cuda.synchronize()
        outputs = []
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            outputs.append(compiled(x))
        torch.cuda.synchronize()
    expected = model(x)
    for output in outputs:
        torch.testing.assert_close(output, expected)
    return {
        "freezing_enabled_during_compile": True,
        "cudagraphs_enabled_during_compile": True,
        "warm_output_ptr": warm.untyped_storage().data_ptr(),
        "replay_output_ptrs": [
            output.untyped_storage().data_ptr() for output in outputs
        ],
        "output_matches": True,
        "model_training": model.training,
    }


CASES = (
    CaseSpec(
        "compiled_autograd",
        "Compiled Autograd graph capture",
        ("f01",),
        ("torch",),
        compiled_autograd,
    ),
    CaseSpec(
        "checkpoint_recompute",
        "Activation checkpoint recompute under compile",
        ("f02",),
        ("torch",),
        checkpoint_recompute,
    ),
    CaseSpec(
        "ddp_compile",
        "DDP compile boundary and gradient synchronization",
        ("f03",),
        ("torch", "distributed", "linux"),
        ddp_compile,
    ),
    CaseSpec(
        "fsdp_dtensor",
        "FSDP2 parameter DTensor sharding across CUDA ranks",
        ("f04",),
        ("torch", "distributed", "cuda_multi_gpu"),
        fsdp_dtensor,
    ),
    CaseSpec(
        "custom_op_contract",
        "Custom operator fake kernel and autograd contract",
        ("f05",),
        ("torch",),
        custom_op_contract,
    ),
    CaseSpec(
        "custom_backend",
        "Custom torch.compile backend callable contract",
        ("f06",),
        ("torch",),
        custom_backend,
    ),
    CaseSpec(
        "aotinductor_package",
        "AOTInductor export, package and load",
        ("f07",),
        ("torch", "native_compiler", "triton"),
        aotinductor_package,
    ),
    CaseSpec(
        "inference_freezing_cudagraph",
        "Inference freezing with CUDA Graph replay",
        ("f08",),
        ("torch", "cuda", "triton"),
        inference_freezing_cudagraph,
    ),
)


if __name__ == "__main__":
    raise SystemExit(run_volume_cli("F", CASES))
