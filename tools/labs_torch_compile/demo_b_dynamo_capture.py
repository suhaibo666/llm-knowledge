"""Volume B demo: TorchDynamo capture, guards, breaks and backends."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Iterator

import torch

from demo_harness import CaseSpec, DemoContext, run_volume_cli


def _device(context: DemoContext) -> torch.device:
    return torch.device(context.device)


@contextlib.contextmanager
def _fresh_dynamo() -> Iterator[None]:
    torch._dynamo.reset()
    try:
        yield
    finally:
        torch._dynamo.reset()


@dataclass
class BackendRecorder:
    compile_count: int = 0
    graph_codes: list[str] | None = None
    input_summaries: list[list[str]] | None = None

    def __post_init__(self) -> None:
        self.graph_codes = []
        self.input_summaries = []

    def __call__(
        self, graph_module: torch.fx.GraphModule, example_inputs: list[object]
    ) -> Callable[..., object]:
        self.compile_count += 1
        assert self.graph_codes is not None
        assert self.input_summaries is not None
        self.graph_codes.append(graph_module.code)
        self.input_summaries.append(
            [
                (
                    f"Tensor(shape={tuple(value.shape)},dtype={value.dtype})"
                    if isinstance(value, torch.Tensor)
                    else type(value).__name__
                )
                for value in example_inputs
            ]
        )
        return graph_module.forward


def compile_lifecycle(context: DemoContext) -> dict[str, object]:
    recorder = BackendRecorder()
    device = _device(context)

    def function(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + x.square()

    with _fresh_dynamo():
        compiled = torch.compile(function, backend=recorder)
        wrapper_compile_count = recorder.compile_count
        x = torch.randn(8, device=device)
        first = compiled(x)
        first_count = recorder.compile_count
        second = compiled(x + 1)
        second_count = recorder.compile_count
    observations = {
        "wrapper_creation_compile_count": wrapper_compile_count,
        "first_call_compile_count": first_count,
        "second_call_compile_count": second_count,
        "graph_count": len(recorder.graph_codes or []),
        "first_matches_eager": bool(torch.allclose(first, function(x))),
        "second_matches_eager": bool(torch.allclose(second, function(x + 1))),
    }
    assert wrapper_compile_count == 0
    assert first_count == 1
    assert second_count == 1
    assert observations["first_matches_eager"]
    assert observations["second_matches_eager"]
    return observations


def backend_modes_fullgraph(context: DemoContext) -> dict[str, object]:
    device = _device(context)
    normal_recorder = BackendRecorder()
    fullgraph_error = ""

    def with_break(x: torch.Tensor) -> torch.Tensor:
        left = x.sin()
        torch._dynamo.graph_break()
        return left.cos()

    with _fresh_dynamo():
        normal = torch.compile(with_break, backend=normal_recorder)
        x = torch.randn(4, device=device)
        normal_output = normal(x)
        try:
            torch.compile(
                with_break,
                backend=BackendRecorder(),
                fullgraph=True,
            )(x)
        except Exception as error:
            fullgraph_error = type(error).__name__
    observations = {
        "partial_graph_backend_calls": normal_recorder.compile_count,
        "fullgraph_rejected_explicit_break": bool(fullgraph_error),
        "fullgraph_error_type": fullgraph_error,
        "partial_matches_eager": bool(torch.allclose(normal_output, with_break(x))),
    }
    assert observations["partial_graph_backend_calls"] == 2
    assert observations["fullgraph_rejected_explicit_break"]
    assert observations["partial_matches_eager"]
    return observations


def eval_frame_cache(context: DemoContext) -> dict[str, object]:
    recorder = BackendRecorder()
    device = _device(context)

    def function(x: torch.Tensor) -> torch.Tensor:
        return x.cos() * 3

    code_id = id(function.__code__)
    with _fresh_dynamo():
        compiled = torch.compile(function, backend=recorder)
        x = torch.randn(6, device=device)
        compiled(x)
        after_first = recorder.compile_count
        compiled(x + 1)
        after_second = recorder.compile_count
    observations = {
        "code_object_id": code_id,
        "same_code_object": id(function.__code__) == code_id,
        "after_first_call": after_first,
        "after_second_call": after_second,
        "cache_reused": after_first == after_second == 1,
    }
    assert observations["cache_reused"]
    return observations


def bytecode_state_machine(context: DemoContext) -> dict[str, object]:
    recorder = BackendRecorder()
    device = _device(context)

    def function(x: torch.Tensor, choose_left: bool) -> torch.Tensor:
        local = x + 1
        if choose_left:
            local = local.sin()
        else:
            local = local.cos()
        return local * 2

    with _fresh_dynamo():
        compiled = torch.compile(function, backend=recorder)
        x = torch.randn(5, device=device)
        left = compiled(x, True)
        left_count = recorder.compile_count
        right = compiled(x, False)
        right_count = recorder.compile_count
    observations = {
        "left_specialization_count": left_count,
        "right_specialization_count": right_count,
        "boolean_branch_created_new_specialization": right_count > left_count,
        "left_matches": bool(torch.allclose(left, function(x, True))),
        "right_matches": bool(torch.allclose(right, function(x, False))),
    }
    assert observations["boolean_branch_created_new_specialization"]
    assert observations["left_matches"] and observations["right_matches"]
    return observations


def variable_source_guards(context: DemoContext) -> dict[str, object]:
    recorder = BackendRecorder()
    device = _device(context)

    class Scaler:
        def __init__(self) -> None:
            self.scale = 2.0

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.scale

    model = Scaler()
    with _fresh_dynamo():
        compiled = torch.compile(model, backend=recorder)
        x = torch.randn(4, device=device)
        first = compiled(x)
        first_count = recorder.compile_count
        model.scale = 3.0
        second = compiled(x)
        second_count = recorder.compile_count
    observations = {
        "first_compile_count": first_count,
        "after_attribute_change_compile_count": second_count,
        "attribute_change_invalidated_guard": second_count > first_count,
        "first_matches": bool(torch.allclose(first, x * 2)),
        "second_matches": bool(torch.allclose(second, x * 3)),
    }
    assert observations["attribute_change_invalidated_guard"]
    assert observations["first_matches"] and observations["second_matches"]
    return observations


def output_graph_side_effects(context: DemoContext) -> dict[str, object]:
    recorder = BackendRecorder()
    device = _device(context)

    def mutating(x: torch.Tensor) -> torch.Tensor:
        x.add_(1)
        return x * 2

    with _fresh_dynamo():
        compiled = torch.compile(mutating, backend=recorder)
        x = torch.zeros(4, device=device)
        output = compiled(x)
    observations = {
        "backend_compile_count": recorder.compile_count,
        "input_mutated": bool(torch.equal(x, torch.ones_like(x))),
        "output_matches": bool(torch.equal(output, torch.full_like(output, 2))),
        "graph_contains_inplace_add": any(
            "add_" in code for code in (recorder.graph_codes or [])
        ),
    }
    assert observations["input_mutated"]
    assert observations["output_matches"]
    return observations


def guards_recompile(context: DemoContext) -> dict[str, object]:
    recorder = BackendRecorder()
    device = _device(context)

    def function(x: torch.Tensor) -> torch.Tensor:
        return x * 2 + 1

    with _fresh_dynamo():
        compiled = torch.compile(function, backend=recorder)
        compiled(torch.randn(4, device=device, dtype=torch.float32))
        after_first = recorder.compile_count
        compiled(torch.randn(4, device=device, dtype=torch.float32))
        after_cache_hit = recorder.compile_count
        compiled(torch.randn(4, device=device, dtype=torch.float64))
        final_count = recorder.compile_count
    observations = {
        "after_first_call": after_first,
        "after_same_contract": after_cache_hit,
        "final_compile_count": final_count,
        "same_contract_cache_hit": after_cache_hit == after_first,
        "dtype_guard_caused_new_compile": final_count > after_cache_hit,
    }
    assert observations["same_contract_cache_hit"]
    assert observations["dtype_guard_caused_new_compile"]
    return observations


def graph_break_resume(context: DemoContext) -> dict[str, object]:
    recorder = BackendRecorder()
    device = _device(context)

    def function(x: torch.Tensor) -> torch.Tensor:
        first = x.sin()
        torch._dynamo.graph_break()
        return first.square() + 1

    with _fresh_dynamo():
        compiled = torch.compile(function, backend=recorder)
        x = torch.randn(7, device=device)
        output = compiled(x)
    observations = {
        "backend_graph_count": recorder.compile_count,
        "resume_created_second_graph": recorder.compile_count == 2,
        "output_matches": bool(torch.allclose(output, function(x))),
    }
    assert observations["resume_created_second_graph"]
    assert observations["output_matches"]
    return observations


def dynamic_shapes(context: DemoContext) -> dict[str, object]:
    recorder = BackendRecorder()
    device = _device(context)

    def function(x: torch.Tensor) -> torch.Tensor:
        return x.sin().sum(dim=-1)

    with _fresh_dynamo():
        compiled = torch.compile(function, backend=recorder, dynamic=True)
        first = compiled(torch.randn(2, 4, device=device))
        after_first = recorder.compile_count
        second = compiled(torch.randn(5, 4, device=device))
        after_second = recorder.compile_count
        third = compiled(torch.randn(5, 4, 1, device=device))
        after_rank_change = recorder.compile_count
    observations = {
        "after_first_shape": after_first,
        "after_second_shape": after_second,
        "after_rank_change": after_rank_change,
        "same_rank_reused_dynamic_graph": after_second == after_first,
        "rank_change_new_specialization": after_rank_change > after_second,
        "first_output_shape": list(first.shape),
        "second_output_shape": list(second.shape),
        "third_output_shape": list(third.shape),
    }
    assert observations["same_rank_reused_dynamic_graph"]
    assert observations["rank_change_new_specialization"]
    return observations


def custom_backend_contract(context: DemoContext) -> dict[str, object]:
    recorder = BackendRecorder()
    device = _device(context)

    def function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ y)

    with _fresh_dynamo():
        compiled = torch.compile(function, backend=recorder, fullgraph=True)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(4, 5, device=device)
        output = compiled(x, y)
    graph_code = (recorder.graph_codes or [""])[0]
    observations = {
        "backend_compile_count": recorder.compile_count,
        "received_graph_module": bool(graph_code),
        "example_input_count": len((recorder.input_summaries or [[]])[0]),
        "graph_has_matmul": "matmul" in graph_code,
        "output_matches": bool(torch.allclose(output, function(x, y))),
    }
    assert observations["backend_compile_count"] == 1
    assert observations["received_graph_module"]
    assert observations["example_input_count"] == 2
    assert observations["output_matches"]
    return observations


CASES = (
    CaseSpec("compile_lifecycle", "torch.compile first-call lifecycle", ("b01",), ("torch",), compile_lifecycle),
    CaseSpec("backend_modes_fullgraph", "Partial graphs versus fullgraph", ("b02",), ("torch",), backend_modes_fullgraph),
    CaseSpec("eval_frame_cache", "Code-object cache reuse", ("b03",), ("torch",), eval_frame_cache),
    CaseSpec("bytecode_state_machine", "Bytecode branch specialization", ("b04",), ("torch",), bytecode_state_machine),
    CaseSpec("variable_source_guards", "Object sources and guards", ("b05",), ("torch",), variable_source_guards),
    CaseSpec("output_graph_side_effects", "Output graph and side effects", ("b06",), ("torch",), output_graph_side_effects),
    CaseSpec("guards_recompile", "Guard hit and recompilation", ("b07",), ("torch",), guards_recompile),
    CaseSpec("graph_break_resume", "Graph break and resume graphs", ("b08",), ("torch",), graph_break_resume),
    CaseSpec("dynamic_shapes", "Dynamic shape reuse and rank specialization", ("b09",), ("torch",), dynamic_shapes),
    CaseSpec("custom_backend_contract", "Custom backend callable contract", ("b10",), ("torch",), custom_backend_contract),
)


if __name__ == "__main__":
    raise SystemExit(run_volume_cli("B", CASES))
