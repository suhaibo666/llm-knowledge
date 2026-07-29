"""Volume E demos: diagnostics, reproducibility, validation and rollout."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter

import torch
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_function

from demo_harness import CaseSpec, DemoContext, run_volume_cli


def _device(context: DemoContext) -> torch.device:
    return torch.device(context.device)


def logs_artifact_map(context: DemoContext) -> dict[str, object]:
    from torch._dynamo.utils import counters

    graphs: list[torch.fx.GraphModule] = []

    def backend(
        gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ):
        graphs.append(gm)
        return gm.forward

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.sin(x) + 1)

    torch._dynamo.reset()
    counters.clear()
    x = torch.randn(8, device=_device(context))
    compiled = torch.compile(fn, backend=backend)
    actual = compiled(x)
    torch.testing.assert_close(actual, fn(x))
    graph_path = context.output_dir / "dynamo_graph.py"
    graph_path.write_text(graphs[0].code, encoding="utf-8")
    counter_snapshot = {
        group: dict(values)
        for group, values in sorted(counters.items())
        if values
    }
    map_document = {
        "stages": [
            {
                "stage": "dynamo_capture",
                "artifact": graph_path.name,
                "node_count": len(list(graphs[0].graph.nodes)),
            },
            {
                "stage": "runtime",
                "artifact": None,
                "output_shape": list(actual.shape),
            },
        ],
        "counters": counter_snapshot,
    }
    (context.output_dir / "artifact_map.json").write_text(
        json.dumps(map_document, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return {
        "captured_graph_count": len(graphs),
        "counter_groups": sorted(counter_snapshot),
        "artifact_stage_count": len(map_document["stages"]),
        "output_matches": True,
    }


def dynamo_explain(context: DemoContext) -> dict[str, object]:
    def fn(x: torch.Tensor) -> torch.Tensor:
        first = torch.sin(x)
        torch._dynamo.graph_break()
        return torch.cos(first) + 1

    x = torch.randn(8, device=_device(context))
    explanation = torch._dynamo.explain(fn)(x)
    break_reasons = [
        {
            "reason": str(reason.reason),
            "user_stack": [str(frame) for frame in reason.user_stack],
        }
        for reason in explanation.break_reasons
    ]
    (context.output_dir / "explain.json").write_text(
        json.dumps(
            {
                "graph_count": explanation.graph_count,
                "graph_break_count": explanation.graph_break_count,
                "op_count": explanation.op_count,
                "break_reasons": break_reasons,
                "out_guards": [str(guard) for guard in explanation.out_guards],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "graph_count": int(explanation.graph_count),
        "graph_break_count": int(explanation.graph_break_count),
        "op_count": int(explanation.op_count),
        "break_reason_count": len(break_reasons),
        "guard_count": len(explanation.out_guards),
    }


def guard_failure(context: DemoContext) -> dict[str, object]:
    records: list[dict[str, object]] = []

    def backend(
        gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ):
        tensors = [
            value for value in example_inputs if isinstance(value, torch.Tensor)
        ]
        records.append(
            {
                "shapes": [list(value.shape) for value in tensors],
                "dtypes": [str(value.dtype) for value in tensors],
            }
        )
        return gm.forward

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) * 2

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend=backend, dynamic=False)
    device = _device(context)
    transitions: list[dict[str, object]] = []
    for label, value in (
        ("first", torch.randn(4, device=device)),
        ("same_contract", torch.randn(4, device=device)),
        ("shape_change", torch.randn(7, device=device)),
        ("dtype_change", torch.randn(7, device=device, dtype=torch.float64)),
    ):
        compiled(value)
        transitions.append({"label": label, "compile_count": len(records)})
    (context.output_dir / "guard_transitions.json").write_text(
        json.dumps(
            {"transitions": transitions, "compile_records": records},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "compile_counts": {
            item["label"]: item["compile_count"] for item in transitions
        },
        "same_contract_reused": (
            transitions[1]["compile_count"] == transitions[0]["compile_count"]
        ),
        "shape_recompiled": (
            transitions[2]["compile_count"] > transitions[1]["compile_count"]
        ),
        "dtype_recompiled": (
            transitions[3]["compile_count"] > transitions[2]["compile_count"]
        ),
    }


def stage_failure_localization(context: DemoContext) -> dict[str, object]:
    device = _device(context)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.cos(torch.sin(x))

    x = torch.randn(8, device=device, requires_grad=True)
    exported = torch._dynamo.export(fn)(x)
    capture_nodes = len(list(exported.graph_module.graph.nodes))

    def broken_backend(
        gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ):
        raise RuntimeError("injected backend compile failure")

    torch._dynamo.reset()
    backend_failure = ""
    try:
        torch.compile(fn, backend=broken_backend, fullgraph=True)(x)
    except Exception as error:
        backend_failure = type(error).__name__
    else:
        raise AssertionError("backend compile failure was not observed")

    backward_compiles = 0

    def fw_compiler(
        gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ):
        return make_boxed_func(gm.forward)

    def broken_bw_compiler(
        gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ):
        nonlocal backward_compiles
        backward_compiles += 1
        raise RuntimeError("injected AOT backward compile failure")

    compiled_aot = aot_function(
        fn,
        fw_compiler=fw_compiler,
        bw_compiler=broken_bw_compiler,
    )
    aot_output = compiled_aot(x)
    backward_compiles_after_forward = backward_compiles
    aot_failure = ""
    try:
        aot_output.sum().backward()
    except RuntimeError as error:
        aot_failure = type(error).__name__
    else:
        raise AssertionError("AOT backward compile failure was not observed")

    eager_output = fn(x.detach())
    return {
        "capture_succeeded_before_backend_failure": capture_nodes > 0,
        "capture_node_count": capture_nodes,
        "backend_failure_type": backend_failure,
        "backward_compiles_after_forward": backward_compiles_after_forward,
        "backward_compiles_after_backward": backward_compiles,
        "aot_backward_failure_type": aot_failure,
        "eager_fallback_is_finite": bool(torch.isfinite(eager_output).all()),
    }


def minifier_repro(context: DemoContext) -> dict[str, object]:
    from torch._dynamo.repro.after_dynamo import (
        generate_dynamo_fx_repro_string,
    )

    device = _device(context)
    captured: list[
        tuple[torch.fx.GraphModule, list[torch.Tensor]]
    ] = []

    def fn(x: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = torch.cos(a)
        c = b + a
        return c.sum()

    def recording_backend(
        gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ):
        captured.append((gm, list(example_inputs)))
        return gm.forward

    x = torch.randn(4, device=device)
    torch._dynamo.reset()
    torch.compile(fn, backend=recording_backend, fullgraph=True)(x)
    if len(captured) != 1:
        raise AssertionError(
            f"expected one Dynamo graph, captured {len(captured)}"
        )
    graph_module, example_inputs = captured[0]
    source = generate_dynamo_fx_repro_string(
        graph_module,
        example_inputs,
        "eager",
        stable_output=True,
        save_dir=str(context.output_dir),
    )
    repro_path = context.output_dir / "repro.py"
    repro_path.write_text(source, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-B", str(repro_path), "run"],
        cwd=context.output_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    (context.output_dir / "repro_process.json").write_text(
        json.dumps(
            {
                "command": completed.args,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise ChildProcessError(
            f"generated after-Dynamo repro exited {completed.returncode}"
        )
    return {
        "original_node_count": len(list(graph_module.graph.nodes)),
        "repro_generated_by": (
            "torch._dynamo.repro.after_dynamo.generate_dynamo_fx_repro_string"
        ),
        "standalone_exit_code": completed.returncode,
        "contains_run_repro": "run_repro" in source,
    }


def correctness_validation(context: DemoContext) -> dict[str, object]:
    device = _device(context)

    def fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x @ weight).square().mean()

    x = torch.randn(4, 4, device=device, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(
        4, 4, device=device, dtype=torch.float64, requires_grad=True
    )
    eager_x = x.detach().clone().requires_grad_(True)
    eager_weight = weight.detach().clone().requires_grad_(True)
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    actual = compiled(x, weight)
    expected = fn(eager_x, eager_weight)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(x.grad, eager_x.grad, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(
        weight.grad, eager_weight.grad, rtol=1e-7, atol=1e-9
    )
    gradcheck_passed = torch.autograd.gradcheck(
        compiled,
        (
            x.detach().clone().requires_grad_(True),
            weight.detach().clone().requires_grad_(True),
        ),
        fast_mode=True,
    )
    return {
        "forward_matches": True,
        "input_gradient_matches": True,
        "parameter_gradient_matches": True,
        "gradcheck_passed": bool(gradcheck_passed),
        "dtype": str(x.dtype),
    }


def cold_warm_steady(context: DemoContext) -> dict[str, object]:
    device = _device(context)

    def fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x @ weight)

    x = torch.randn(1024, 1024, device=device)
    weight = torch.randn(1024, 1024, device=device)
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)

    def timed_call() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        compiled(x, weight)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    wall_start = time.perf_counter()
    cold_ms = timed_call()
    cold_wall_ms = (time.perf_counter() - wall_start) * 1000
    warm_ms = timed_call()
    steady_ms = [timed_call() for _ in range(10)]
    return {
        "cold_cuda_event_ms": cold_ms,
        "cold_wall_ms_includes_host_compile": cold_wall_ms,
        "warm_ms": warm_ms,
        "steady_samples_ms": steady_ms,
        "steady_median_ms": sorted(steady_ms)[len(steady_ms) // 2],
        "synchronized": True,
    }


def fusion_memory_profiler(context: DemoContext) -> dict[str, object]:
    from torch._inductor import metrics
    from torch.profiler import ProfilerActivity, profile

    device = _device(context)

    def fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        projected = x @ weight
        return torch.nn.functional.gelu(projected + 0.1) * 2

    x = torch.randn(1024, 1024, device=device)
    weight = torch.randn(1024, 1024, device=device)
    metrics.reset()
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)
    compiled(x, weight)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    trace_path = context.output_dir / "cuda_trace.json"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as profiler:
        compiled(x, weight)
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    event_counts = Counter(event.key for event in profiler.key_averages())
    return {
        "generated_kernel_count": int(metrics.generated_kernel_count),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "profile_event_kinds": len(event_counts),
        "cuda_event_seen": any(
            event.self_cuda_time_total > 0
            for event in profiler.key_averages()
        ),
        "output_matches": bool(
            torch.allclose(compiled(x, weight), fn(x, weight))
        ),
    }


def rollout_fallback(context: DemoContext) -> dict[str, object]:
    device = _device(context)
    metrics = {"compiled_successes": 0, "eager_fallbacks": 0}

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + 1

    def backend(
        gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ):
        inner = gm.forward

        def guarded(*args):
            tensor = next(value for value in args if isinstance(value, torch.Tensor))
            if tensor.numel() == 3:
                raise RuntimeError("simulated compiled runtime incompatibility")
            return inner(*args)

        return guarded

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend=backend, dynamic=True)

    def production_call(x: torch.Tensor) -> torch.Tensor:
        try:
            output = compiled(x)
            metrics["compiled_successes"] += 1
            return output
        except RuntimeError:
            metrics["eager_fallbacks"] += 1
            return fn(x)

    normal = torch.randn(4, device=device)
    fallback = torch.randn(3, device=device)
    normal_output = production_call(normal)
    fallback_output = production_call(fallback)
    torch.testing.assert_close(normal_output, fn(normal))
    torch.testing.assert_close(fallback_output, fn(fallback))
    return {
        **metrics,
        "normal_matches_eager": True,
        "fallback_matches_eager": True,
        "fallback_scope": "per-call",
        "failure_was_not_silently_counted_as_compile_success": True,
    }


CASES = (
    CaseSpec(
        "logs_artifact_map",
        "Logs, counters and artifact-stage map",
        ("e01",),
        ("torch",),
        logs_artifact_map,
    ),
    CaseSpec(
        "dynamo_explain",
        "Dynamo explain and graph-break diagnosis",
        ("e02",),
        ("torch",),
        dynamo_explain,
    ),
    CaseSpec(
        "guard_failure",
        "Guard miss and recompilation diagnosis",
        ("e03",),
        ("torch",),
        guard_failure,
    ),
    CaseSpec(
        "stage_failure_localization",
        "Capture, backend and lazy-backward failure localization",
        ("e04",),
        ("torch",),
        stage_failure_localization,
    ),
    CaseSpec(
        "minifier_repro",
        "After-Dynamo standalone repro generation",
        ("e05",),
        ("torch",),
        minifier_repro,
    ),
    CaseSpec(
        "correctness_validation",
        "Compiled forward and gradient correctness validation",
        ("e06",),
        ("torch",),
        correctness_validation,
    ),
    CaseSpec(
        "cold_warm_steady",
        "Cold compile, warm call and steady-state CUDA timing",
        ("e07",),
        ("torch", "cuda", "triton"),
        cold_warm_steady,
    ),
    CaseSpec(
        "fusion_memory_profiler",
        "Kernel, fusion and CUDA memory profiler evidence",
        ("e08",),
        ("torch", "cuda", "triton"),
        fusion_memory_profiler,
    ),
    CaseSpec(
        "rollout_fallback",
        "Production-style compiled call fallback and counters",
        ("e09",),
        ("torch",),
        rollout_fallback,
    ),
)


if __name__ == "__main__":
    raise SystemExit(run_volume_cli("E", CASES))
