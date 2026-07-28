import argparse
import json
import operator
from pathlib import Path

import torch
from torch._inductor import lowering
from torch._inductor.debug import DebugContext
from torch._inductor.graph import GraphLowering
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    FailedMatch,
    GraphPatternEntry,
    Ignored,
    KeywordArg,
    LoweringPatternEntry,
    MULTIPLE,
    MultiOutputPattern,
    PatternMatcherPass,
    ReplacementPatternEntry,
    fwd_only,
    register_graph_pattern,
    register_lowering_pattern,
    register_replacement,
)
from torch._inductor.virtualized import V
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx


SOURCE_BASELINE = "e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52"


def _only_entry(pattern_pass, expected_type):
    entries = [
        entry
        for bucket in pattern_pass.patterns.values()
        for entry in bucket
    ]
    if len(entries) != 1 or not isinstance(entries[0], expected_type):
        raise AssertionError(
            f"expected one {expected_type.__name__}, got "
            f"{[type(entry).__name__ for entry in entries]}"
        )
    return entries[0]


def run_expression_cases():
    def unary(x):
        return -x

    unary_gm = symbolic_trace(unary)
    unary_node = next(
        node for node in unary_gm.graph.nodes if node.op == "call_function"
    )
    unary_match = CallFunction(operator.neg, Arg()).match(unary_node)
    unary_pattern_matched = not isinstance(unary_match, FailedMatch)
    unary_arg_captured = unary_pattern_matched and unary_match.args == [
        next(node for node in unary_gm.graph.nodes if node.op == "placeholder")
    ]

    def shared(x, y):
        return (x + x) * y

    shared_gm = symbolic_trace(shared)
    mul = next(
        node
        for node in shared_gm.graph.nodes
        if node.op == "call_function" and node.target is operator.mul
    )

    shared_x = KeywordArg("x")
    shared_pattern = CallFunction(
        operator.mul,
        CallFunction(operator.add, shared_x, shared_x),
        Arg(),
    )
    shared_match = shared_pattern.match(mul)
    shared_pattern_matched = not isinstance(shared_match, FailedMatch)
    if not shared_pattern_matched:
        raise AssertionError(f"shared pattern did not match: {shared_match}")

    ignored_pattern = CallFunction(
        operator.mul,
        CallFunction(
            operator.add,
            KeywordArg("left"),
            KeywordArg("right"),
        ),
        Ignored(),
    )
    ignored_match = ignored_pattern.match(mul)
    if isinstance(ignored_match, FailedMatch):
        raise AssertionError(f"ignored pattern did not match: {ignored_match}")

    def kwargs_case(x):
        return torch.clamp(x, min=-0.5, max=0.5)

    kwargs_gm = symbolic_trace(kwargs_case)
    clamp = next(
        node
        for node in kwargs_gm.graph.nodes
        if node.op == "call_function"
    )
    kwargs_pattern = CallFunction(
        torch.clamp,
        KeywordArg("input"),
        min=-0.5,
        max=KeywordArg("max_value"),
    )
    kwargs_match = kwargs_pattern.match(clamp)
    kwargs_pattern_matched = not isinstance(kwargs_match, FailedMatch)
    kwargs_constant_captured = (
        kwargs_pattern_matched
        and kwargs_match.kwargs["max_value"] == 0.5
        and isinstance(kwargs_match.kwargs["input"], torch.fx.Node)
    )

    def not_shared(x, y, z):
        return (x + y) * z

    other = symbolic_trace(not_shared)
    other_mul = next(
        node
        for node in other.graph.nodes
        if node.op == "call_function" and node.target is operator.mul
    )
    failed = shared_pattern.match(other_mul)
    failed_sharing_pattern = isinstance(failed, FailedMatch)

    def multiple_outputs(x):
        shared_value = x + x
        return shared_value * 2, shared_value - 3

    multi_gm = symbolic_trace(multiple_outputs)
    multi_mul = next(
        node
        for node in multi_gm.graph.nodes
        if node.op == "call_function" and node.target is operator.mul
    )
    multi_x = KeywordArg("multi_x")
    multi_add = CallFunction(
        operator.add,
        multi_x,
        multi_x,
        _users=MULTIPLE,
    )
    multi_pattern = MultiOutputPattern(
        [
            CallFunction(operator.mul, multi_add, Ignored()),
            CallFunction(operator.sub, multi_add, Ignored()),
        ]
    )
    multi_match = multi_pattern.match(multi_mul)
    multi_output_pattern_matched = not isinstance(multi_match, FailedMatch)

    checks = {
        "unary_pattern_matched": unary_pattern_matched,
        "unary_arg_captured": unary_arg_captured,
        "shared_pattern_matched": shared_pattern_matched,
        "failed_sharing_pattern": failed_sharing_pattern,
        "kwargs_pattern_matched": kwargs_pattern_matched,
        "kwargs_constant_captured": kwargs_constant_captured,
        "multi_output_pattern_matched": multi_output_pattern_matched,
        "ignored_not_positional": len(ignored_match.args) == 0,
    }
    if not all(checks.values()):
        raise AssertionError(f"PatternExpr contract failed: {checks}")

    print(f"unary_pattern_matched={unary_pattern_matched}")
    print(f"unary_arg_captured={unary_arg_captured}")
    print(f"shared_pattern_matched={shared_pattern_matched}")
    print(
        "positional_captures="
        + ",".join(node.name for node in shared_match.args)
    )
    print(
        "keyword_captures="
        + ",".join(
            f"{key}:{value.name}"
            for key, value in shared_match.kwargs.items()
        )
    )
    print(f"ignored_positional_count={len(ignored_match.args)}")
    print(f"ignored_keyword_count={len(ignored_match.kwargs)}")
    print(f"kwargs_pattern_matched={kwargs_pattern_matched}")
    print(f"kwargs_constant_captured={kwargs_constant_captured}")
    print(f"failed_sharing_pattern={failed_sharing_pattern}")
    print(f"multi_output_pattern_matched={multi_output_pattern_matched}")
    return checks


def run_graph_pattern_entry(output_dir):
    def source(x, y):
        return (x + x) * y

    graph_module = symbolic_trace(source)
    original_compute_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
    ]
    graph_pass = PatternMatcherPass(pass_name="graph_series_graph_entry")
    graph_handler_calls = 0
    shared_x = KeywordArg("x")
    graph_pattern = CallFunction(
        operator.mul,
        CallFunction(operator.add, shared_x, shared_x),
        KeywordArg("y"),
    )

    @register_graph_pattern(
        graph_pattern,
        extra_check=lambda match: not match.output_node().meta.get(
            "graph_series_pattern_rewritten",
            False,
        ),
        pass_dict=graph_pass,
    )
    def replace_graph(match, *, x, y):
        nonlocal graph_handler_calls
        graph_handler_calls += 1
        old_output = match.output_node()
        doubled = match.graph.call_function(operator.add, (x, x))
        replacement = match.graph.call_function(operator.mul, (doubled, y))
        replacement.meta.update(old_output.meta)
        replacement.meta["graph_series_pattern_rewritten"] = True
        old_output.replace_all_uses_with(replacement)
        match.erase_nodes()

    _only_entry(graph_pass, GraphPatternEntry)
    pattern_matcher_pass_apply_count = graph_pass.apply(graph_module)
    graph_module.graph.lint()
    graph_module.recompile()

    sample_x = torch.tensor([2.0, -1.0])
    sample_y = torch.tensor([0.5, 3.0])
    graph_pattern_value_matches = torch.equal(
        graph_module(sample_x, sample_y),
        source(sample_x, sample_y),
    )
    graph_pattern_old_nodes_erased = all(
        node._erased for node in original_compute_nodes
    )
    graph_pattern_second_apply_count = graph_pass.apply(graph_module)
    if output_dir is not None:
        (output_dir / "graph_pattern_after.txt").write_text(
            str(graph_module.graph) + "\n",
            encoding="utf-8",
        )

    checks = {
        "pattern_matcher_pass_apply_count": (
            pattern_matcher_pass_apply_count == 1
        ),
        "pattern_matcher_handler_calls": graph_handler_calls == 1,
        "graph_pattern_value_matches": graph_pattern_value_matches,
        "graph_pattern_old_nodes_erased": graph_pattern_old_nodes_erased,
        "graph_pattern_second_apply_count": (
            graph_pattern_second_apply_count == 0
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"GraphPatternEntry contract failed: {checks}")

    print(f"pattern_matcher_pass_apply_count={pattern_matcher_pass_apply_count}")
    print(f"pattern_matcher_handler_calls={graph_handler_calls}")
    print("graph_pattern_entry_type=GraphPatternEntry")
    print(f"graph_pattern_value_matches={graph_pattern_value_matches}")
    print(
        "graph_pattern_old_nodes_erased="
        + str(graph_pattern_old_nodes_erased)
    )
    print(
        "graph_pattern_second_apply_count="
        + str(graph_pattern_second_apply_count)
    )
    return {
        "entry_type": "GraphPatternEntry",
        "apply_count": pattern_matcher_pass_apply_count,
        "handler_calls": graph_handler_calls,
        "value_matches": graph_pattern_value_matches,
        "old_nodes_erased": graph_pattern_old_nodes_erased,
        "second_apply_count": graph_pattern_second_apply_count,
    }


def run_replacement_pattern_entry(output_dir):
    def search(x):
        return torch.ops.aten.add.Tensor(x, x)

    def replacement(x):
        return x * 2

    example = torch.tensor([2.0, -3.0, 0.5])
    replacement_pass = PatternMatcherPass(
        pass_name="graph_series_replacement_entry"
    )
    register_replacement(
        search,
        replacement,
        [example],
        fwd_only,
        replacement_pass,
    )
    _only_entry(replacement_pass, ReplacementPatternEntry)

    graph_module = fwd_only(search, [example])
    replacement_apply_count = replacement_pass.apply(graph_module)
    graph_module.graph.eliminate_dead_code()
    graph_module.graph.lint()
    graph_module.recompile()
    replacement_value_matches = torch.equal(
        graph_module(example),
        replacement(example),
    )
    replacement_has_mul = any(
        node.op == "call_function"
        and node.target is torch.ops.aten.mul.Tensor
        for node in graph_module.graph.nodes
    )
    replacement_has_no_add = not any(
        node.op == "call_function"
        and node.target is torch.ops.aten.add.Tensor
        for node in graph_module.graph.nodes
    )
    replacement_second_apply_count = replacement_pass.apply(graph_module)
    if output_dir is not None:
        (output_dir / "replacement_after.txt").write_text(
            str(graph_module.graph) + "\n",
            encoding="utf-8",
        )

    rejecting_pass = PatternMatcherPass(
        pass_name="graph_series_replacement_rejected"
    )
    register_replacement(
        search,
        replacement,
        [example],
        fwd_only,
        rejecting_pass,
        extra_check=lambda match: False,
    )
    rejected_graph_module = fwd_only(search, [example])
    rejected_before = str(rejected_graph_module.graph)
    rejected_count = rejecting_pass.apply(rejected_graph_module)
    replacement_extra_check_rejected = (
        rejected_count == 0
        and str(rejected_graph_module.graph) == rejected_before
    )

    checks = {
        "replacement_apply_count": replacement_apply_count == 1,
        "replacement_value_matches": replacement_value_matches,
        "replacement_has_mul": replacement_has_mul,
        "replacement_has_no_add": replacement_has_no_add,
        "replacement_second_apply_count": replacement_second_apply_count == 0,
        "replacement_extra_check_rejected": (
            replacement_extra_check_rejected
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"ReplacementPatternEntry contract failed: {checks}")

    print("replacement_entry_type=ReplacementPatternEntry")
    print(f"replacement_apply_count={replacement_apply_count}")
    print(f"replacement_value_matches={replacement_value_matches}")
    print(f"replacement_has_mul={replacement_has_mul}")
    print(f"replacement_has_no_add={replacement_has_no_add}")
    print(
        "replacement_second_apply_count="
        + str(replacement_second_apply_count)
    )
    print(
        "replacement_extra_check_rejected="
        + str(replacement_extra_check_rejected)
    )
    return {
        "entry_type": "ReplacementPatternEntry",
        "apply_count": replacement_apply_count,
        "value_matches": replacement_value_matches,
        "has_mul": replacement_has_mul,
        "has_no_add": replacement_has_no_add,
        "second_apply_count": replacement_second_apply_count,
        "extra_check_rejected": replacement_extra_check_rejected,
    }


def run_lowering_pattern_entry(output_dir):
    def source(x, y):
        return torch.ops.aten.add.Tensor(x, y)

    example_x = torch.randn(4)
    example_y = torch.randn(4)
    graph_module = make_fx(source)(example_x, example_y)
    lowering_pass = PatternMatcherPass(
        pass_name="graph_series_lowering_entry"
    )
    lowering_handler_calls = 0

    @register_lowering_pattern(
        CallFunction(
            torch.ops.aten.add.Tensor,
            KeywordArg("x"),
            KeywordArg("y"),
        ),
        pass_dict=lowering_pass,
    )
    def lower_add_direct(match, *, x, y):
        nonlocal lowering_handler_calls
        lowering_handler_calls += 1
        return lowering.add(x, y)

    _only_entry(lowering_pass, LoweringPatternEntry)
    lowering_apply_count = lowering_pass.apply(graph_module)
    graph_module.graph.lint()
    graph_module.recompile()

    lowering_call = next(
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
    )
    lowering_handler_marked = bool(
        getattr(
            lowering_call.target,
            "_inductor_lowering_function",
            False,
        )
    )
    lowering_handler_deferred_until_graph_lowering = (
        lowering_handler_calls == 0
    )

    graph_lowering = GraphLowering(
        graph_module,
        example_inputs=(example_x, example_y),
        is_inference=True,
    )
    with (
        DebugContext(),
        V.set_graph_handler(graph_lowering),
        V.set_extern_kernel_nodes([]),
    ):
        graph_lowering.run(example_x, example_y)

    lowering_pattern_reached_inductor_ir = (
        lowering_handler_calls == 1
        and any(
            type(operation).__name__ == "ComputedBuffer"
            and type(operation.data).__name__ == "Pointwise"
            for operation in graph_lowering.operations
        )
    )
    lowering_original_add_erased = not any(
        node.op == "call_function"
        and node.target is torch.ops.aten.add.Tensor
        for node in graph_module.graph.nodes
    )
    if output_dir is not None:
        (output_dir / "lowering_pattern_after.txt").write_text(
            str(graph_module.graph) + "\n",
            encoding="utf-8",
        )
        (output_dir / "lowering_ir.txt").write_text(
            "\n".join(
                "operation="
                + type(operation).__name__
                + " data="
                + type(operation.data).__name__
                + " layout="
                + type(operation.get_layout()).__name__
                for operation in graph_lowering.operations
            )
            + "\n",
            encoding="utf-8",
        )

    checks = {
        "lowering_apply_count": lowering_apply_count == 1,
        "lowering_handler_marked": lowering_handler_marked,
        "lowering_handler_deferred_until_graph_lowering": (
            lowering_handler_deferred_until_graph_lowering
        ),
        "lowering_pattern_reached_inductor_ir": (
            lowering_pattern_reached_inductor_ir
        ),
        "lowering_original_add_erased": lowering_original_add_erased,
    }
    if not all(checks.values()):
        raise AssertionError(f"LoweringPatternEntry contract failed: {checks}")

    print("lowering_entry_type=LoweringPatternEntry")
    print(f"lowering_apply_count={lowering_apply_count}")
    print(f"lowering_handler_marked={lowering_handler_marked}")
    print(
        "lowering_handler_deferred_until_graph_lowering="
        + str(lowering_handler_deferred_until_graph_lowering)
    )
    print(
        "lowering_pattern_reached_inductor_ir="
        + str(lowering_pattern_reached_inductor_ir)
    )
    print(f"lowering_original_add_erased={lowering_original_add_erased}")
    print("lowering_native_kernel_executed=False")
    return {
        "entry_type": "LoweringPatternEntry",
        "apply_count": lowering_apply_count,
        "handler_marked": lowering_handler_marked,
        "handler_deferred_until_graph_lowering": (
            lowering_handler_deferred_until_graph_lowering
        ),
        "reached_inductor_ir": lowering_pattern_reached_inductor_ir,
        "original_add_erased": lowering_original_add_erased,
        "native_kernel_executed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    print(f"source_locator_baseline={SOURCE_BASELINE}")
    print(f"runtime_torch_version={torch.__version__}")
    print(f"runtime_torch_git={torch.version.git_version}")
    print(
        "runtime_matches_source_baseline="
        + str(torch.version.git_version == SOURCE_BASELINE)
    )
    summary = {
        "source_locator_baseline": SOURCE_BASELINE,
        "runtime_torch_version": torch.__version__,
        "runtime_torch_git": torch.version.git_version,
        "runtime_matches_source_baseline": (
            torch.version.git_version == SOURCE_BASELINE
        ),
        "expression": run_expression_cases(),
        "graph_pattern_entry": run_graph_pattern_entry(args.output_dir),
        "replacement_pattern_entry": run_replacement_pattern_entry(
            args.output_dir
        ),
        "lowering_pattern_entry": run_lowering_pattern_entry(args.output_dir),
        "evidence_boundary": {
            "fx_pattern_matching_and_rewrite": "runtime_observed",
            "traced_replacement_execution": "runtime_observed",
            "inductor_graph_lowering": "runtime_observed",
            "native_kernel_execution": "not_run",
        },
    }
    if args.output_dir is not None:
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"artifact_dir={args.output_dir.as_posix()}")


if __name__ == "__main__":
    main()
