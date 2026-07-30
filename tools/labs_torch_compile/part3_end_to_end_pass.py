import argparse
import copy
import json
import operator
import platform
import shutil
import sys
from pathlib import Path

import torch
from torch.fx import GraphModule, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


SOURCE_BASELINE = "e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52"
ADD_TARGETS = {operator.add, torch.add}
MATMUL_TARGETS = {operator.matmul, torch.matmul}


def add_matmul_model(x, weight, bias):
    return torch.add(torch.matmul(x, weight), bias)


def _tensor_meta(node):
    return node.meta.get("tensor_meta")


def rewrite_add_matmul(gm: GraphModule, example_inputs) -> bool:
    """Rewrite exact-shape add(matmul(x, w), bias) to addmm(bias, x, w)."""

    # Analyze and edit a detached candidate. A rejected match or an exception
    # before commit cannot leave structural or metadata changes in the caller's gm.
    candidate = copy.deepcopy(gm)
    ShapeProp(candidate).propagate(*example_inputs)
    modified = False

    for add_node in list(candidate.graph.nodes):
        if add_node.op != "call_function" or add_node.target not in ADD_TARGETS:
            continue
        if len(add_node.args) != 2:
            continue

        left, right = add_node.args
        if (
            isinstance(left, torch.fx.Node)
            and left.op == "call_function"
            and left.target in MATMUL_TARGETS
        ):
            matmul_node, bias_node = left, right
        elif (
            isinstance(right, torch.fx.Node)
            and right.op == "call_function"
            and right.target in MATMUL_TARGETS
        ):
            matmul_node, bias_node = right, left
        else:
            continue

        if not isinstance(bias_node, torch.fx.Node) or len(matmul_node.args) != 2:
            continue
        x_node, weight_node = matmul_node.args
        if not isinstance(x_node, torch.fx.Node) or not isinstance(
            weight_node, torch.fx.Node
        ):
            continue

        add_meta = _tensor_meta(add_node)
        matmul_meta = _tensor_meta(matmul_node)
        bias_meta = _tensor_meta(bias_node)
        x_meta = _tensor_meta(x_node)
        weight_meta = _tensor_meta(weight_node)
        metas = (add_meta, matmul_meta, bias_meta, x_meta, weight_meta)
        if any(meta is None for meta in metas):
            continue

        exact_shape = tuple(matmul_meta.shape) == tuple(bias_meta.shape)
        matrix_contract = (
            len(x_meta.shape) == 2
            and len(weight_meta.shape) == 2
            and len(matmul_meta.shape) == 2
        )
        same_dtype = len({meta.dtype for meta in metas}) == 1
        if not (exact_shape and matrix_contract and same_dtype):
            continue

        with candidate.graph.inserting_before(add_node):
            replacement = candidate.graph.call_function(
                torch.addmm, args=(bias_node, x_node, weight_node)
            )
            replacement.meta = copy.copy(add_node.meta)
        add_node.replace_all_uses_with(replacement)
        candidate.graph.erase_node(add_node)
        if not matmul_node.users:
            candidate.graph.erase_node(matmul_node)
        modified = True

    if modified:
        candidate.graph.eliminate_dead_code()
        candidate.graph.lint()
        candidate.recompile()
        gm.graph = candidate.graph
        gm.recompile()
    return modified


def _clone_leaf(value):
    return value.detach().clone().requires_grad_(value.requires_grad)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _meta_snapshot(gm: GraphModule) -> list[dict[str, object]]:
    return [
        {
            "name": node.name,
            "keys": sorted(str(key) for key in node.meta),
            "values": {str(key): repr(value) for key, value in node.meta.items()},
        }
        for node in gm.graph.nodes
    ]


def _write_artifacts(
    output_dir: Path,
    *,
    results: dict[str, bool],
    legal_before: str,
    legal_after: str,
    illegal_before: str,
    illegal_after: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    text_artifacts = {
        "legal_before.py": legal_before,
        "legal_after.py": legal_after,
        "illegal_before.py": illegal_before,
        "illegal_after.py": illegal_after,
    }
    for name, content in text_artifacts.items():
        (output_dir / name).write_text(content.rstrip() + "\n", encoding="utf-8")

    environment = {
        "audit_source_baseline": SOURCE_BASELINE,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda_available": torch.cuda.is_available(),
        "msvc_cl": shutil.which("cl"),
        "seed": 0,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "command": [sys.executable, *sys.argv],
        "rewrite": "add(matmul(x, weight), bias) -> addmm(bias, x, weight)",
        "legal_domain": {
            "x_rank": 2,
            "weight_rank": 2,
            "bias_shape": "exactly equal to matmul output",
            "dtype": "all tensor metadata dtypes equal",
        },
        "rejection_case": "broadcast bias shape (5,) is intentionally unsupported",
        "artifacts": sorted(
            [*text_artifacts, "environment.json", "results.json", "manifest.json"]
        ),
        "checks": results,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    torch.manual_seed(0)

    base = torch.randn(3, 4)
    x = base.t().detach().requires_grad_(True)
    weight = torch.randn(3, 5, requires_grad=True)
    bias = torch.randn(4, 5, requires_grad=True)

    legal_gm = symbolic_trace(add_matmul_model)
    legal_code_before = legal_gm.code
    legal_modified = rewrite_add_matmul(legal_gm, (x, weight, bias))
    legal_has_addmm = any(
        node.op == "call_function" and node.target is torch.addmm
        for node in legal_gm.graph.nodes
    )

    actual_inputs = tuple(_clone_leaf(value) for value in (x, weight, bias))
    expected_inputs = tuple(_clone_leaf(value) for value in (x, weight, bias))
    actual_output = legal_gm(*actual_inputs)
    expected_output = add_matmul_model(*expected_inputs)
    actual_output.square().sum().backward()
    expected_output.square().sum().backward()

    forward_matches = torch.allclose(actual_output, expected_output)
    gradient_matches = all(
        torch.allclose(actual.grad, expected.grad)
        for actual, expected in zip(actual_inputs, expected_inputs)
    )
    shape_matches = actual_output.shape == expected_output.shape
    alias_contract_matches = tuple(
        torch._C._is_alias_of(actual_output, value) for value in actual_inputs
    ) == tuple(
        torch._C._is_alias_of(expected_output, value)
        for value in expected_inputs
    )

    actual_mutation_inputs = tuple(
        _clone_leaf(value) for value in (x, weight, bias)
    )
    expected_mutation_inputs = tuple(
        _clone_leaf(value) for value in (x, weight, bias)
    )
    actual_snapshots = tuple(
        value.detach().clone() for value in actual_mutation_inputs
    )
    expected_snapshots = tuple(
        value.detach().clone() for value in expected_mutation_inputs
    )
    legal_gm(*actual_mutation_inputs)
    add_matmul_model(*expected_mutation_inputs)
    actual_mutations = tuple(
        not torch.equal(value.detach(), snapshot)
        for value, snapshot in zip(actual_mutation_inputs, actual_snapshots)
    )
    expected_mutations = tuple(
        not torch.equal(value.detach(), snapshot)
        for value, snapshot in zip(expected_mutation_inputs, expected_snapshots)
    )
    mutation_contract_matches = actual_mutations == expected_mutations

    gradcheck_inputs = (
        torch.randn(2, 2, dtype=torch.float64, requires_grad=True),
        torch.randn(2, 3, dtype=torch.float64, requires_grad=True),
        torch.randn(2, 3, dtype=torch.float64, requires_grad=True),
    )
    gradcheck_matches = torch.autograd.gradcheck(
        legal_gm, gradcheck_inputs, fast_mode=True
    )

    code_before_second_run = legal_gm.code
    second_run_modified = rewrite_add_matmul(
        legal_gm, tuple(value.detach() for value in actual_inputs)
    )
    second_run_code_unchanged = legal_gm.code == code_before_second_run

    broadcast_bias = torch.randn(5)
    illegal_gm = symbolic_trace(add_matmul_model)
    illegal_code_before = illegal_gm.code
    illegal_meta_before = _meta_snapshot(illegal_gm)
    illegal_modified = rewrite_add_matmul(
        illegal_gm, (x.detach(), weight.detach(), broadcast_bias)
    )
    illegal_graph_unchanged = illegal_gm.code == illegal_code_before
    failure_atomicity_matches = (
        illegal_graph_unchanged
        and _meta_snapshot(illegal_gm) == illegal_meta_before
    )

    results = {
        "legal_rewrite_applied": legal_modified,
        "legal_has_addmm": legal_has_addmm,
        "illegal_broadcast_rewrite_applied": illegal_modified,
        "illegal_graph_unchanged": illegal_graph_unchanged,
        "failure_atomicity_matches": failure_atomicity_matches,
        "forward_matches": forward_matches,
        "gradient_matches": gradient_matches,
        "gradcheck_matches": gradcheck_matches,
        "shape_matches": shape_matches,
        "alias_contract_matches": alias_contract_matches,
        "mutation_contract_matches": mutation_contract_matches,
        "second_run_modified": second_run_modified,
        "second_run_code_unchanged": second_run_code_unchanged,
    }
    expected_results = {
        **{key: True for key in results if key != "second_run_modified"},
        "illegal_broadcast_rewrite_applied": False,
        "second_run_modified": False,
    }
    if results != expected_results:
        raise AssertionError(
            f"rewrite contract failed: expected={expected_results}, actual={results}"
        )

    if args.output_dir is not None:
        _write_artifacts(
            args.output_dir.resolve(),
            results=results,
            legal_before=legal_code_before,
            legal_after=legal_gm.code,
            illegal_before=illegal_code_before,
            illegal_after=illegal_gm.code,
        )

    for key, value in results.items():
        print(f"{key}={value}")
    if args.output_dir is not None:
        print(f"artifact_dir={args.output_dir.resolve().as_posix()}")


if __name__ == "__main__":
    main()
