import operator

import torch
from torch.fx import Graph, GraphModule, symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.infra.pass_manager import PassManager
from torch._inductor.pattern_matcher import stable_topological_sort


def f(x):
    return (x + 1) * 2


source_gm = symbolic_trace(f)
node_copy_graph = Graph()
node_copy_env = {}
for source_node in source_gm.graph.nodes:
    if source_node.op == "output":
        node_copy_graph.output(
            torch.fx.node.map_arg(
                source_node.args[0], lambda node: node_copy_env[node]
            )
        )
    else:
        node_copy_env[source_node] = node_copy_graph.node_copy(
            source_node, lambda node: node_copy_env[node]
        )
node_copy_gm = GraphModule(source_gm, node_copy_graph)
node_copy_gm.graph.lint()

graph_copy_graph = Graph()
graph_copy_output = graph_copy_graph.graph_copy(source_gm.graph, {})
graph_copy_graph.output(graph_copy_output)
graph_copy_gm = GraphModule(source_gm, graph_copy_graph)
graph_copy_gm.graph.lint()

copy_input = torch.tensor(3.0)
expected_copy_value = source_gm(copy_input)
node_copy_value_matches = torch.equal(
    node_copy_gm(copy_input), expected_copy_value
)
graph_copy_value_matches = torch.equal(
    graph_copy_gm(copy_input), expected_copy_value
)
print(f"node_copy_value_matches={node_copy_value_matches}")
print(f"graph_copy_value_matches={graph_copy_value_matches}")


gm = symbolic_trace(f)
add = next(
    n for n in gm.graph.nodes if n.op == "call_function" and n.target is operator.add
)
mul = next(
    n for n in gm.graph.nodes if n.op == "call_function" and n.target is operator.mul
)
mul.append(add)
try:
    gm.graph.lint()
    lint_failed = False
except RuntimeError:
    lint_failed = True
print(f"lint_failed_before_sort={lint_failed}")
stable_topological_sort(gm.graph)
gm.graph.lint()
gm.recompile()
print(f"topology_repaired_value={gm(torch.tensor(3.0)).item()}")


class CountToThree(PassBase):
    def call(self, graph_module):
        count = graph_module.meta.get("count_to_three", 0) + 1
        graph_module.meta["count_to_three"] = count
        return PassResult(graph_module, count < 3)


one_step_gm = symbolic_trace(f)
PassManager(passes=[CountToThree()], steps=1)(one_step_gm)
print(f"pass_manager_steps_1_count={one_step_gm.meta['count_to_three']}")

four_step_gm = symbolic_trace(f)
PassManager(passes=[CountToThree()], steps=4)(four_step_gm)
print(f"pass_manager_steps_4_count={four_step_gm.meta['count_to_three']}")


def stage_model(x, weight, bias):
    return torch.add(torch.matmul(x, weight), bias)


def target_spelling(target):
    if hasattr(target, "_schema"):
        return str(target)
    return getattr(target, "__name__", str(target))


def call_target_spellings(graph_module):
    return [
        target_spelling(node.target)
        for node in graph_module.graph.nodes
        if node.op == "call_function"
    ]


def apply_add_matmul_fusion(
    graph_module,
    *,
    add_target,
    matmul_target,
    addmm_target,
):
    edits = 0
    for add_node in list(graph_module.graph.nodes):
        if (
            add_node.op != "call_function"
            or add_node.target is not add_target
            or add_node.kwargs
            or len(add_node.args) != 2
        ):
            continue

        lhs, rhs = add_node.args
        matmul_node = None
        bias = None
        for candidate, other in ((lhs, rhs), (rhs, lhs)):
            if (
                isinstance(candidate, torch.fx.Node)
                and candidate.op == "call_function"
                and candidate.target is matmul_target
                and not candidate.kwargs
                and len(candidate.args) == 2
            ):
                matmul_node = candidate
                bias = other
                break
        if matmul_node is None:
            continue

        with graph_module.graph.inserting_before(add_node):
            fused = graph_module.graph.call_function(
                addmm_target,
                args=(bias, matmul_node.args[0], matmul_node.args[1]),
            )
        fused.meta.update(add_node.meta)
        add_node.replace_all_uses_with(fused)
        graph_module.graph.erase_node(add_node)
        if not matmul_node.users:
            graph_module.graph.erase_node(matmul_node)
        edits += 1

    if edits:
        graph_module.graph.lint()
        graph_module.recompile()
    return edits


high_level_contract = {
    "add_target": torch.add,
    "matmul_target": torch.matmul,
    "addmm_target": torch.addmm,
}
functional_aten_contract = {
    "add_target": torch.ops.aten.add.Tensor,
    "matmul_target": torch.ops.aten.mm.default,
    "addmm_target": torch.ops.aten.addmm.default,
}

stage_x = torch.randn(2, 3)
stage_weight = torch.randn(3, 4)
stage_bias = torch.randn(4)
stage_expected = stage_model(stage_x, stage_weight, stage_bias)

front_end_like_gm = symbolic_trace(stage_model)
functional_aten_like_gm = make_fx(stage_model)(
    stage_x,
    stage_weight,
    stage_bias,
)
front_end_before_targets = call_target_spellings(front_end_like_gm)
functional_aten_before_targets = call_target_spellings(functional_aten_like_gm)

front_end_wrong_contract_gm = symbolic_trace(stage_model)
functional_aten_wrong_contract_gm = make_fx(stage_model)(
    stage_x,
    stage_weight,
    stage_bias,
)
front_end_wrong_contract_edits = apply_add_matmul_fusion(
    front_end_wrong_contract_gm,
    **functional_aten_contract,
)
functional_aten_wrong_contract_edits = apply_add_matmul_fusion(
    functional_aten_wrong_contract_gm,
    **high_level_contract,
)

front_end_edits = apply_add_matmul_fusion(
    front_end_like_gm,
    **high_level_contract,
)
functional_aten_edits = apply_add_matmul_fusion(
    functional_aten_like_gm,
    **functional_aten_contract,
)
front_end_second_edits = apply_add_matmul_fusion(
    front_end_like_gm,
    **high_level_contract,
)
functional_aten_second_edits = apply_add_matmul_fusion(
    functional_aten_like_gm,
    **functional_aten_contract,
)

front_end_after_targets = call_target_spellings(front_end_like_gm)
functional_aten_after_targets = call_target_spellings(functional_aten_like_gm)
stage_target_spelling_differs = (
    front_end_before_targets == ["matmul", "add"]
    and functional_aten_before_targets == [
        "aten.mm.default",
        "aten.add.Tensor",
    ]
    and front_end_after_targets == ["addmm"]
    and functional_aten_after_targets == ["aten.addmm.default"]
)
stage_correct_contract_rewrites = (
    front_end_edits == 1 and functional_aten_edits == 1
)
stage_wrong_contract_rejected = (
    front_end_wrong_contract_edits == 0
    and functional_aten_wrong_contract_edits == 0
    and call_target_spellings(front_end_wrong_contract_gm)
    == front_end_before_targets
    and call_target_spellings(functional_aten_wrong_contract_gm)
    == functional_aten_before_targets
    and torch.allclose(
        front_end_wrong_contract_gm(stage_x, stage_weight, stage_bias),
        stage_expected,
    )
    and torch.allclose(
        functional_aten_wrong_contract_gm(
            stage_x,
            stage_weight,
            stage_bias,
        ),
        stage_expected,
    )
)
stage_rewrite_idempotent = (
    front_end_second_edits == 0 and functional_aten_second_edits == 0
)
stage_outputs_match = torch.allclose(
    front_end_like_gm(stage_x, stage_weight, stage_bias),
    stage_expected,
) and torch.allclose(
    functional_aten_like_gm(stage_x, stage_weight, stage_bias),
    stage_expected,
)

print(
    "front_end_like_before_targets="
    + ",".join(front_end_before_targets)
)
print(
    "functional_aten_like_before_targets="
    + ",".join(functional_aten_before_targets)
)
print(
    "front_end_like_after_targets="
    + ",".join(front_end_after_targets)
)
print(
    "functional_aten_like_after_targets="
    + ",".join(functional_aten_after_targets)
)
print(f"stage_target_spelling_differs={stage_target_spelling_differs}")
print(f"stage_correct_contract_rewrites={stage_correct_contract_rewrites}")
print(f"stage_wrong_contract_rejected={stage_wrong_contract_rejected}")
print(f"stage_rewrite_idempotent={stage_rewrite_idempotent}")
print(f"stage_outputs_match={stage_outputs_match}")
print("stage_contract_kind=simulated_frontend_like_vs_functional_aten")
print("actual_torch_compile_stage_hook_executed=False")


class AddToSub(PassBase):
    def call(self, graph_module):
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target is operator.add:
                node.target = operator.sub
                graph_module.meta["oscillation_edits"] = (
                    graph_module.meta.get("oscillation_edits", 0) + 1
                )
                return PassResult(graph_module, True)
        return PassResult(graph_module, False)


class SubToAdd(PassBase):
    def call(self, graph_module):
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target is operator.sub:
                node.target = operator.add
                graph_module.meta["oscillation_edits"] = (
                    graph_module.meta.get("oscillation_edits", 0) + 1
                )
                return PassResult(graph_module, True)
        return PassResult(graph_module, False)


oscillating_gm = symbolic_trace(f)
PassManager(passes=[AddToSub(), SubToAdd()], steps=4)(oscillating_gm)
oscillation_bounded_at_four = oscillating_gm.meta["oscillation_edits"] == 8
oscillation_final_target_is_add = any(
    node.op == "call_function" and node.target is operator.add
    for node in oscillating_gm.graph.nodes
)
print(f"oscillation_bounded_at_four={oscillation_bounded_at_four}")
print(
    "oscillation_final_target_is_add="
    + str(oscillation_final_target_is_add)
)

checks = {
    "node_copy_value_matches": node_copy_value_matches,
    "graph_copy_value_matches": graph_copy_value_matches,
    "lint_failed_before_sort": lint_failed,
    "topology_repaired_value": gm(torch.tensor(3.0)).item() == 8.0,
    "pass_manager_steps_1_count": one_step_gm.meta["count_to_three"] == 1,
    "pass_manager_steps_4_count": four_step_gm.meta["count_to_three"] == 3,
    "stage_target_spelling_differs": stage_target_spelling_differs,
    "stage_correct_contract_rewrites": stage_correct_contract_rewrites,
    "stage_wrong_contract_rejected": stage_wrong_contract_rejected,
    "stage_rewrite_idempotent": stage_rewrite_idempotent,
    "stage_outputs_match": stage_outputs_match,
    "oscillation_bounded_at_four": oscillation_bounded_at_four,
    "oscillation_final_target_is_add": oscillation_final_target_is_add,
}
if not all(checks.values()):
    raise AssertionError(f"editing/pass-manager contract failed: {checks}")
