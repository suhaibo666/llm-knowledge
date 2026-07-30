import argparse
import hashlib
import json
import operator
import platform
import sys
from pathlib import Path

import torch
from torch._inductor import config


SOURCE_BASELINE = "e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52"


def model(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return x @ weight + bias


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def _pattern_nodes(
    graph: torch.fx.Graph,
    add_target: object,
    matmul_target: object,
) -> list[tuple[torch.fx.Node, torch.fx.Node, object]]:
    matches = []
    for add in graph.nodes:
        if add.op != "call_function" or add.target != add_target:
            continue
        lhs, rhs = add.args[:2]
        matmul = next(
            (
                value
                for value in (lhs, rhs)
                if isinstance(value, torch.fx.Node)
                and value.op == "call_function"
                and value.target == matmul_target
            ),
            None,
        )
        if matmul is None or len(matmul.users) != 1:
            continue
        bias = rhs if matmul is lhs else lhs
        matches.append((add, matmul, bias))
    return matches


def _rewrite_add_matmul(
    graph: torch.fx.Graph,
    *,
    add_target: object,
    matmul_target: object,
    addmm_target: object,
) -> int:
    matches = _pattern_nodes(graph, add_target, matmul_target)
    for add, matmul, bias in matches:
        with graph.inserting_before(add):
            replacement = graph.call_function(
                addmm_target,
                args=(bias, matmul.args[0], matmul.args[1]),
            )
        replacement.meta = dict(add.meta)
        add.replace_all_uses_with(replacement)
        graph.erase_node(add)
        graph.erase_node(matmul)
    graph.lint()
    return len(matches)


def _run_stage(
    stage: str,
    base_x: torch.Tensor,
    base_weight: torch.Tensor,
    bias: torch.Tensor,
    expected_output: torch.Tensor,
    expected_x_grad: torch.Tensor,
    expected_weight_grad: torch.Tensor,
) -> dict[str, object]:
    state: dict[str, object] = {
        "hook_calls": 0,
        "correct_hits": 0,
        "wrong_hits": 0,
        "pass_idempotence_checks": 0,
        "pass_second_run_rewrite_count": 0,
        "pass_second_run_graph_unchanged": True,
        "graph_text": "",
    }

    def stage_pass(graph: torch.fx.Graph) -> None:
        state["hook_calls"] = int(state["hook_calls"]) + 1
        if stage == "pre":
            correct = _pattern_nodes(graph, operator.add, operator.matmul)
            wrong = _pattern_nodes(
                graph,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.mm.default,
            )
            state["correct_hits"] = int(state["correct_hits"]) + len(correct)
            state["wrong_hits"] = int(state["wrong_hits"]) + len(wrong)
            rewrite_count = _rewrite_add_matmul(
                graph,
                add_target=operator.add,
                matmul_target=operator.matmul,
                addmm_target=torch.addmm,
            )
        else:
            correct = _pattern_nodes(
                graph,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.mm.default,
            )
            wrong = _pattern_nodes(graph, operator.add, operator.matmul)
            state["correct_hits"] = int(state["correct_hits"]) + len(correct)
            state["wrong_hits"] = int(state["wrong_hits"]) + len(wrong)
            rewrite_count = _rewrite_add_matmul(
                graph,
                add_target=torch.ops.aten.add.Tensor,
                matmul_target=torch.ops.aten.mm.default,
                addmm_target=torch.ops.aten.addmm.default,
            )
        if rewrite_count:
            first_rewrite_graph = graph.python_code("self").src
            if stage == "pre":
                second_rewrite_count = _rewrite_add_matmul(
                    graph,
                    add_target=operator.add,
                    matmul_target=operator.matmul,
                    addmm_target=torch.addmm,
                )
            else:
                second_rewrite_count = _rewrite_add_matmul(
                    graph,
                    add_target=torch.ops.aten.add.Tensor,
                    matmul_target=torch.ops.aten.mm.default,
                    addmm_target=torch.ops.aten.addmm.default,
                )
            second_rewrite_graph = graph.python_code("self").src
            state["pass_idempotence_checks"] = (
                int(state["pass_idempotence_checks"]) + 1
            )
            state["pass_second_run_rewrite_count"] = (
                int(state["pass_second_run_rewrite_count"])
                + second_rewrite_count
            )
            state["pass_second_run_graph_unchanged"] = bool(
                state["pass_second_run_graph_unchanged"]
            ) and first_rewrite_graph == second_rewrite_graph
            state["graph_text"] = second_rewrite_graph

    patch = {
        "force_disable_caches": True,
        (
            "pre_grad_custom_pass"
            if stage == "pre"
            else "post_grad_custom_pre_pass"
        ): stage_pass,
    }
    torch._dynamo.reset()
    x = base_x.detach().clone().requires_grad_(True)
    weight = base_weight.detach().clone().requires_grad_(True)
    with config.patch(patch):
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        actual = compiled(x, weight, bias)
        actual.backward(torch.ones_like(actual))
        first_hook_calls = int(state["hook_calls"])
        first_hits = int(state["correct_hits"])

        second_x = base_x.detach().clone().requires_grad_(True)
        second_weight = base_weight.detach().clone().requires_grad_(True)
        second_actual = compiled(second_x, second_weight, bias)
        second_actual.backward(torch.ones_like(second_actual))

    return {
        "matched_targets": (
            ["operator.matmul", "operator.add"]
            if stage == "pre"
            else ["aten.mm.default", "aten.add.Tensor"]
        ),
        "correct_stage_hits": int(state["correct_hits"]),
        "wrong_stage_hits": int(state["wrong_hits"]),
        "hook_calls": int(state["hook_calls"]),
        "compile_count": 1 if first_hits == 1 else 0,
        "forward_matches": bool(
            torch.allclose(actual, expected_output)
            and torch.allclose(second_actual, expected_output)
        ),
        "gradient_matches": bool(
            torch.allclose(x.grad, expected_x_grad)
            and torch.allclose(weight.grad, expected_weight_grad)
            and torch.allclose(second_x.grad, expected_x_grad)
            and torch.allclose(second_weight.grad, expected_weight_grad)
        ),
        "pass_idempotence_checks": int(state["pass_idempotence_checks"]),
        "pass_second_run_rewrite_count": int(
            state["pass_second_run_rewrite_count"]
        ),
        "pass_second_run_graph_unchanged": bool(
            state["pass_second_run_graph_unchanged"]
        ),
        "second_call_no_recompile": int(state["hook_calls"]) == first_hook_calls,
        "second_call_no_additional_rewrite": (
            int(state["correct_hits"]) == first_hits
        ),
        "graph_text": str(state["graph_text"]),
    }


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

    base_x = torch.randn(4, 8)
    base_weight = torch.randn(8, 6)
    bias = torch.randn(6)
    eager_x = base_x.detach().clone().requires_grad_(True)
    eager_weight = base_weight.detach().clone().requires_grad_(True)
    expected_output = model(eager_x, eager_weight, bias)
    expected_output.backward(torch.ones_like(expected_output))

    original_config = {
        "force_disable_caches": config.force_disable_caches,
        "pre_grad_custom_pass": config.pre_grad_custom_pass,
        "post_grad_custom_pre_pass": config.post_grad_custom_pre_pass,
    }
    pre = _run_stage(
        "pre",
        base_x,
        base_weight,
        bias,
        expected_output.detach(),
        eager_x.grad,
        eager_weight.grad,
    )
    post = _run_stage(
        "post",
        base_x,
        base_weight,
        bias,
        expected_output.detach(),
        eager_x.grad,
        eager_weight.grad,
    )
    config_restored = all(
        getattr(config, name) is value
        for name, value in original_config.items()
    )

    checks = {
        "actual_torch_compile_stage_hook_executed": (
            pre["hook_calls"] >= 1 and post["hook_calls"] >= 1
        ),
        "pre_correct_stage_hits": pre["correct_stage_hits"],
        "pre_wrong_stage_hits": pre["wrong_stage_hits"],
        "post_correct_stage_hits": post["correct_stage_hits"],
        "post_wrong_stage_hits": post["wrong_stage_hits"],
        "pre_forward_matches": pre["forward_matches"],
        "pre_gradient_matches": pre["gradient_matches"],
        "post_forward_matches": post["forward_matches"],
        "post_gradient_matches": post["gradient_matches"],
        "pass_second_run_zero_rewrites": (
            pre["pass_idempotence_checks"] == 1
            and post["pass_idempotence_checks"] == 1
            and pre["pass_second_run_rewrite_count"] == 0
            and post["pass_second_run_rewrite_count"] == 0
        ),
        "pass_second_run_graph_unchanged": (
            pre["pass_second_run_graph_unchanged"]
            and post["pass_second_run_graph_unchanged"]
        ),
        "second_call_no_recompile": (
            pre["second_call_no_recompile"]
            and post["second_call_no_recompile"]
        ),
        "second_call_no_additional_rewrite": (
            pre["second_call_no_additional_rewrite"]
            and post["second_call_no_additional_rewrite"]
        ),
        "config_restored": config_restored,
        "native_kernel_executed": False,
    }
    assert checks == {
        "actual_torch_compile_stage_hook_executed": True,
        "pre_correct_stage_hits": 1,
        "pre_wrong_stage_hits": 0,
        "post_correct_stage_hits": 1,
        "post_wrong_stage_hits": 0,
        "pre_forward_matches": True,
        "pre_gradient_matches": True,
        "post_forward_matches": True,
        "post_gradient_matches": True,
        "pass_second_run_zero_rewrites": True,
        "pass_second_run_graph_unchanged": True,
        "second_call_no_recompile": True,
        "second_call_no_additional_rewrite": True,
        "config_restored": True,
        "native_kernel_executed": False,
    }, checks

    results = {
        "pre": {key: value for key, value in pre.items() if key != "graph_text"},
        "post": {key: value for key, value in post.items() if key != "graph_text"},
        "config_restored": config_restored,
        "mock_compiler_used": False,
        "native_kernel_executed": False,
        "native_kernel_boundary": (
            "The test executes real Inductor extern addmm/mm paths; it does not "
            "claim an Inductor-generated C++ kernel."
        ),
        "checks": checks,
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
    _write_json(output_dir / "results.json", results)
    _write_text(output_dir / "pre_stage_graph.py", str(pre["graph_text"]))
    _write_text(output_dir / "post_stage_graph.py", str(post["graph_text"]))
    artifact_names = [
        "environment.json",
        "results.json",
        "pre_stage_graph.py",
        "post_stage_graph.py",
    ]
    manifest = {
        "entrypoint": Path(__file__).resolve().as_posix(),
        "command": [sys.executable, *sys.argv],
        "evidence_scope": "real_torch_compile_inductor_extern_addmm",
        "artifacts": _artifact_records(output_dir, artifact_names),
        "checks": checks,
    }
    _write_json(output_dir / "manifest.json", manifest)

    for key, value in checks.items():
        print(f"{key}={value}")
    print(f"artifact_dir={output_dir.as_posix()}")


if __name__ == "__main__":
    main()
