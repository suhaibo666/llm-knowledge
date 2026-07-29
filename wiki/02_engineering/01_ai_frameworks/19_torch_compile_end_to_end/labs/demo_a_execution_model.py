"""Volume A demo: execution-model foundations used by torch.compile."""

from __future__ import annotations

import dis
import inspect
import time
from typing import Any

import torch

from demo_harness import CaseSpec, DemoContext, run_volume_cli


def _device(context: DemoContext) -> torch.device:
    return torch.device(context.device)


def tensor_storage_layout(context: DemoContext) -> dict[str, object]:
    base = torch.arange(24, dtype=torch.float32, device=_device(context)).reshape(
        4, 6
    )
    view = base[:, 1:5:2]
    clone = view.clone()
    transposed = base.transpose(0, 1)

    base_storage = base.untyped_storage().data_ptr()
    view_storage = view.untyped_storage().data_ptr()
    clone_storage = clone.untyped_storage().data_ptr()
    observations = {
        "base_shape": list(base.shape),
        "base_stride": list(base.stride()),
        "view_shape": list(view.shape),
        "view_stride": list(view.stride()),
        "transpose_stride": list(transposed.stride()),
        "view_shares_storage": view_storage == base_storage,
        "clone_shares_storage": clone_storage == base_storage,
        "view_storage_offset": int(view.storage_offset()),
        "view_is_contiguous": view.is_contiguous(),
        "clone_is_contiguous": clone.is_contiguous(),
    }
    assert observations["view_shares_storage"]
    assert not observations["clone_shares_storage"]
    assert observations["view_storage_offset"] == 1
    return observations


def dispatcher_autograd(context: DemoContext) -> dict[str, object]:
    from torch.utils._python_dispatch import TorchDispatchMode

    dispatched: list[str] = []

    class RecordingMode(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func: Any,
            types: tuple[type, ...],
            args: tuple[object, ...] = (),
            kwargs: dict[str, object] | None = None,
        ) -> object:
            dispatched.append(str(func))
            return func(*args, **(kwargs or {}))

    x = torch.linspace(
        -0.75,
        0.75,
        8,
        device=_device(context),
        requires_grad=True,
    )
    with RecordingMode():
        loss = (torch.sin(x) * x).sum()
        loss.backward()
    expected = torch.sin(x.detach()) + x.detach() * torch.cos(x.detach())
    gradient_matches = bool(torch.allclose(x.grad, expected))
    observations = {
        "dispatch_observed": bool(dispatched),
        "dispatch_count": len(dispatched),
        "first_dispatches": dispatched[:8],
        "autograd_grad_fn": type(loss.grad_fn).__name__,
        "gradient_matches": gradient_matches,
    }
    assert observations["dispatch_observed"]
    assert gradient_matches
    return observations


def python_frame_bytecode(context: DemoContext) -> dict[str, object]:
    del context

    def branchy(value: int, bias: int = 3) -> int:
        shifted = value + bias
        if shifted > 5:
            return shifted * 2
        return shifted - 2

    instructions = list(dis.get_instructions(branchy))
    frame = inspect.currentframe()
    observations = {
        "function_name": branchy.__code__.co_name,
        "positional_arguments": branchy.__code__.co_argcount,
        "local_names": list(branchy.__code__.co_varnames),
        "opnames": [instruction.opname for instruction in instructions],
        "has_conditional_jump": any(
            "JUMP" in instruction.opname for instruction in instructions
        ),
        "same_code_object_across_calls": branchy.__code__ is branchy.__code__,
        "high_branch_result": branchy(4),
        "low_branch_result": branchy(1),
        "caller_code_name": frame.f_code.co_name if frame is not None else None,
    }
    assert observations["has_conditional_jump"]
    assert observations["high_branch_result"] == 14
    assert observations["low_branch_result"] == 2
    return observations


def proxy_fake_tensor(context: DemoContext) -> dict[str, object]:
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    def function(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ weight).view(x.shape[0], -1)

    device = _device(context)
    x = torch.randn(3, 4, device=device)
    weight = torch.randn(4, 5, device=device)
    graph_module = make_fx(function, tracing_mode="fake")(x, weight)
    call_targets = [
        str(node.target)
        for node in graph_module.graph.nodes
        if node.op == "call_function"
    ]

    fake_mode = FakeTensorMode()
    fake_x = fake_mode.from_tensor(torch.randn(2, 4))
    fake_weight = fake_mode.from_tensor(torch.randn(4, 5))
    with fake_mode:
        fake_output = function(fake_x, fake_weight)
    observations = {
        "make_fx_node_count": len(list(graph_module.graph.nodes)),
        "call_targets": call_targets,
        "fake_output_is_fake_tensor": isinstance(fake_output, FakeTensor),
        "fake_output_shape": list(fake_output.shape),
        "fake_output_device": str(fake_output.device),
    }
    assert observations["fake_output_is_fake_tensor"]
    assert observations["fake_output_shape"] == [2, 5]
    assert any("mm" in target for target in call_targets)
    return observations


def eager_compile_cost(context: DemoContext) -> dict[str, object]:
    device = _device(context)

    def function(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ weight + 0.25)

    x = torch.randn(1024, 1024, device=device)
    weight = torch.randn(1024, 1024, device=device)
    compiled = torch.compile(function)

    torch.cuda.synchronize()
    first_started = time.perf_counter()
    first = compiled(x, weight)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - first_started) * 1000

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    steady_samples: list[float] = []
    for _ in range(10):
        start.record()
        current = compiled(x, weight)
        end.record()
        end.synchronize()
        steady_samples.append(float(start.elapsed_time(end)))
    eager = function(x, weight)
    observations = {
        "first_call_wall_ms": first_call_ms,
        "steady_cuda_event_ms": steady_samples,
        "steady_median_ms": sorted(steady_samples)[len(steady_samples) // 2],
        "first_matches_eager": bool(torch.allclose(first, eager)),
        "last_matches_eager": bool(torch.allclose(current, eager)),
        "timing_scope": "single CUDA stream; fixed input; compile cache warm after first call",
    }
    assert observations["first_matches_eager"]
    assert observations["last_matches_eager"]
    return observations


CASES = (
    CaseSpec(
        "tensor_storage_layout",
        "Tensor storage, stride, view and clone",
        ("a01",),
        ("torch",),
        tensor_storage_layout,
    ),
    CaseSpec(
        "dispatcher_autograd",
        "Dispatcher observation and autograd",
        ("a02",),
        ("torch",),
        dispatcher_autograd,
    ),
    CaseSpec(
        "python_frame_bytecode",
        "Python code objects and bytecode",
        ("a03",),
        ("torch",),
        python_frame_bytecode,
    ),
    CaseSpec(
        "proxy_fake_tensor",
        "ProxyTensor, FakeTensor and make_fx",
        ("a04",),
        ("torch",),
        proxy_fake_tensor,
    ),
    CaseSpec(
        "eager_compile_cost",
        "First-call compile cost versus CUDA steady state",
        ("a05",),
        ("torch", "cuda"),
        eager_compile_cost,
    ),
)


if __name__ == "__main__":
    raise SystemExit(run_volume_cli("A", CASES))
