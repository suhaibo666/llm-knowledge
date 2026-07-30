"""Volume D demos: backend orchestration, artifacts, caches and runtime replay."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_function

from demo_harness import CaseSpec, DemoContext, run_volume_cli


def _device(context: DemoContext) -> torch.device:
    return torch.device(context.device)


def compile_fx_orchestration(context: DemoContext) -> dict[str, object]:
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._inductor.compile_fx import compile_fx

    device = _device(context)

    def fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ weight + 0.25)

    x = torch.randn(32, 32, device=device)
    weight = torch.randn(32, 32, device=device)
    graph_module = make_fx(fn)(x, weight)
    graph_path = context.output_dir / "input_fx.py"
    graph_path.write_text(graph_module.code, encoding="utf-8")
    compiled = compile_fx(graph_module, [x, weight])
    actual = compiled([x, weight])
    if isinstance(actual, (tuple, list)):
        actual = actual[0]
    expected = fn(x, weight)
    torch.testing.assert_close(actual, expected)
    return {
        "fx_node_count": len(list(graph_module.graph.nodes)),
        "callable_type": type(compiled).__name__,
        "output_matches": True,
        "direct_entry": "torch._inductor.compile_fx.compile_fx",
    }


def aot_wrappers_lazy_backward(context: DemoContext) -> dict[str, object]:
    forward_graphs: list[torch.fx.GraphModule] = []
    backward_graphs: list[torch.fx.GraphModule] = []

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.cos(torch.sin(x)) * x

    def fw_compiler(
        gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ):
        forward_graphs.append(gm)
        return make_boxed_func(gm.forward)

    def bw_compiler(
        gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ):
        backward_graphs.append(gm)
        return make_boxed_func(gm.forward)

    compiled = aot_function(
        fn,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
    )
    device = _device(context)
    x = torch.randn(16, device=device, requires_grad=True)
    eager_x = x.detach().clone().requires_grad_(True)
    output = compiled(x)
    fw_after_forward = len(forward_graphs)
    bw_after_forward = len(backward_graphs)
    output.sum().backward()
    fn(eager_x).sum().backward()
    torch.testing.assert_close(x.grad, eager_x.grad)

    (context.output_dir / "forward_graph.py").write_text(
        forward_graphs[0].code, encoding="utf-8"
    )
    (context.output_dir / "backward_graph.py").write_text(
        backward_graphs[0].code, encoding="utf-8"
    )
    return {
        "fw_compile_after_forward": fw_after_forward,
        "bw_compile_after_forward": bw_after_forward,
        "bw_compile_after_backward": len(backward_graphs),
        "forward_backward_distinct_graphs": (
            forward_graphs[0].graph is not backward_graphs[0].graph
        ),
        "gradient_matches": True,
    }


def async_compile_loading(context: DemoContext) -> dict[str, object]:
    from torch._inductor import config, metrics
    from torch._inductor.codecache import PyCodeCache

    device = _device(context)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) * torch.cos(x + 1)

    torch._dynamo.reset()
    metrics.reset()
    cache_before = len(PyCodeCache.cache)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)
    x = torch.randn(4096, device=device)
    actual = compiled(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    torch.testing.assert_close(actual, fn(x))
    cache_after = len(PyCodeCache.cache)
    return {
        "compile_threads": int(config.compile_threads),
        "python_module_cache_before": cache_before,
        "python_module_cache_after": cache_after,
        "generated_kernel_count": int(metrics.generated_kernel_count),
        "loaded_callable_type": type(compiled).__name__,
        "output_matches": True,
    }


def cache_keys_invalidation(context: DemoContext) -> dict[str, object]:
    compile_records: list[dict[str, object]] = []

    def backend(
        gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ):
        compile_records.append(
            {
                "graph": gm.code,
                "inputs": [
                    {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                    for value in example_inputs
                    if isinstance(value, torch.Tensor)
                ],
            }
        )
        return gm.forward

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + 1

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend=backend, dynamic=False)
    device = _device(context)
    compiled(torch.randn(4, device=device))
    after_first = len(compile_records)
    compiled(torch.randn(4, device=device))
    after_cache_hit = len(compile_records)
    compiled(torch.randn(8, device=device))
    after_shape_change = len(compile_records)
    compiled(torch.randn(8, device=device, dtype=torch.float64))
    after_dtype_change = len(compile_records)
    torch._dynamo.reset()
    compiled(torch.randn(4, device=device))
    after_explicit_reset = len(compile_records)
    (context.output_dir / "compile_records.json").write_text(
        json.dumps(compile_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "after_first": after_first,
        "after_cache_hit": after_cache_hit,
        "after_shape_change": after_shape_change,
        "after_dtype_change": after_dtype_change,
        "after_explicit_reset": after_explicit_reset,
        "same_signature_reused": after_cache_hit == after_first,
        "shape_invalidated": after_shape_change > after_cache_hit,
        "dtype_invalidated": after_dtype_change > after_shape_change,
        "reset_invalidated": after_explicit_reset > after_dtype_change,
    }


def wrapper_memory_reuse(context: DemoContext) -> dict[str, object]:
    from torch._inductor import config

    device = _device(context)

    def fn(x: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = torch.cos(a)
        c = torch.relu(b + a)
        return c * c

    x = torch.randn(2048, 2048, device=device)

    def measure(memory_planning: bool) -> dict[str, object]:
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        with config.patch(memory_planning=memory_planning):
            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            compiled(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            outputs = [compiled(x) for _ in range(3)]
            torch.cuda.synchronize()
            return {
                "memory_planning": memory_planning,
                "allocated_before_replay": before,
                "peak_allocated_during_replay": torch.cuda.max_memory_allocated(),
                "output_storage_ptrs": [
                    value.untyped_storage().data_ptr() for value in outputs
                ],
                "output_matches": all(
                    torch.allclose(value, fn(x)) for value in outputs
                ),
            }

    disabled = measure(False)
    enabled = measure(True)
    return {
        "without_memory_planning": disabled,
        "with_memory_planning": enabled,
        "interpretation": (
            "Peak allocator measurements include caching effects; output pointers "
            "do not expose internal temporary-buffer reuse."
        ),
    }


def cudagraph_replay(context: DemoContext) -> dict[str, object]:
    device = _device(context)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.sin(x) * 2)

    torch._dynamo.reset()
    compiled = torch.compile(
        fn,
        backend="inductor",
        mode="reduce-overhead",
        fullgraph=True,
    )
    x = torch.randn(1024, 1024, device=device)
    torch.compiler.cudagraph_mark_step_begin()
    warm = compiled(x)
    torch.cuda.synchronize()
    replay_times_ms: list[float] = []
    outputs: list[torch.Tensor] = []
    for _ in range(3):
        torch.compiler.cudagraph_mark_step_begin()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs.append(compiled(x))
        end.record()
        end.synchronize()
        replay_times_ms.append(float(start.elapsed_time(end)))
    expected = fn(x)
    for output in outputs:
        torch.testing.assert_close(output, expected)
    return {
        "warm_output_ptr": warm.untyped_storage().data_ptr(),
        "replay_output_ptrs": [
            output.untyped_storage().data_ptr() for output in outputs
        ],
        "replay_times_ms": replay_times_ms,
        "output_matches": True,
        "step_boundary_api": "torch.compiler.cudagraph_mark_step_begin",
    }


def artifact_lifecycle_failure(context: DemoContext) -> dict[str, object]:
    graphs: list[torch.fx.GraphModule] = []
    runtime_calls = 0

    def backend(
        gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ):
        graphs.append(gm)
        inner = gm.forward

        def checked_runtime(*args):
            nonlocal runtime_calls
            runtime_calls += 1
            tensor = next(value for value in args if isinstance(value, torch.Tensor))
            if tensor.shape[-1] == 3:
                raise RuntimeError("injected compiled-callable runtime failure")
            return inner(*args)

        return checked_runtime

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + 2

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend=backend, dynamic=True)
    device = _device(context)
    good = torch.randn(2, 4, device=device)
    actual = compiled(good)
    torch.testing.assert_close(actual, fn(good))
    failure_type = ""
    failure_message = ""
    bad = torch.randn(2, 3, device=device)
    try:
        compiled(bad)
    except RuntimeError as error:
        failure_type = type(error).__name__
        failure_message = str(error)
    else:
        raise AssertionError("injected runtime failure was not observed")
    eager_fallback = fn(bad)
    torch.testing.assert_close(eager_fallback, torch.sin(bad) + 2)
    (context.output_dir / "captured_graph.py").write_text(
        graphs[0].code, encoding="utf-8"
    )
    return {
        "capture_count": len(graphs),
        "runtime_calls": runtime_calls,
        "valid_output_matches": True,
        "failure_stage": "compiled_callable_runtime",
        "failure_type": failure_type,
        "failure_message": failure_message,
        "eager_fallback_matches": True,
    }


CASES = (
    CaseSpec(
        "compile_fx_orchestration",
        "Direct compile_fx orchestration",
        ("d01",),
        ("torch", "native_compiler", "triton"),
        compile_fx_orchestration,
    ),
    CaseSpec(
        "aot_wrappers_lazy_backward",
        "AOT runtime wrappers and lazy backward compilation",
        ("d02",),
        ("torch",),
        aot_wrappers_lazy_backward,
    ),
    CaseSpec(
        "async_compile_loading",
        "Inductor async compilation and generated-module loading",
        ("d03",),
        ("torch", "native_compiler", "triton"),
        async_compile_loading,
    ),
    CaseSpec(
        "cache_keys_invalidation",
        "Compile cache reuse and invalidation boundaries",
        ("d04",),
        ("torch",),
        cache_keys_invalidation,
    ),
    CaseSpec(
        "wrapper_memory_reuse",
        "Wrapper allocation and memory-planning observations",
        ("d05",),
        ("torch", "cuda", "triton"),
        wrapper_memory_reuse,
    ),
    CaseSpec(
        "cudagraph_replay",
        "CUDA Graph warmup, record and replay",
        ("d06",),
        ("torch", "cuda", "triton"),
        cudagraph_replay,
    ),
    CaseSpec(
        "artifact_lifecycle_failure",
        "Captured artifact lifecycle and runtime failure boundary",
        ("d07",),
        ("torch",),
        artifact_lifecycle_failure,
    ),
)


if __name__ == "__main__":
    raise SystemExit(run_volume_cli("D", CASES))
