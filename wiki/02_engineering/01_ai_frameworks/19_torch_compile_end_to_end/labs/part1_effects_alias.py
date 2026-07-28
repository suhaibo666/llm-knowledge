import operator

import torch
from torch.fx import Graph, GraphModule


graph = Graph()
x = graph.placeholder("x")
y = graph.placeholder("y")
graph.call_function(operator.add, args=(x, y))
graph.call_function(torch.ops.aten.copy_.default, args=(x, y))
graph.output(x)
gm = GraphModule({}, graph)

before = sum(node.op == "call_function" for node in graph.nodes)
changed = graph.eliminate_dead_code()
gm.recompile()
after_targets = [
    str(node.target).replace("aten.", "")
    for node in graph.nodes
    if node.op == "call_function"
]

print(f"before_dce_call_functions={before}")
print(f"dce_changed={changed}")
print("after_dce_targets=" + ",".join(after_targets))

value = torch.tensor(1.0)
alias = value.view(())
result = gm(value, torch.tensor(7.0))
print(f"mutation_result={result.item()}")
print(f"alias_observes_mutation={alias.item()}")

pure_dead_removed = all("add" not in target for target in after_targets)
impure_copy_retained = any("copy_" in target for target in after_targets)
alias_observes_mutation = alias.item() == 7.0


dead_graph = Graph()
dead_x = dead_graph.placeholder("x")
dead_add = dead_graph.call_function(operator.add, args=(dead_x, 1))
dead_graph.call_function(operator.mul, args=(dead_add, 2))
dead_graph.output(dead_x)
dead_graph.eliminate_dead_code()
dead_chain_removed = not any(
    node.op == "call_function" for node in dead_graph.nodes
)


child_graph = Graph()
child_x = child_graph.placeholder("x")
child_graph.call_function(operator.add, args=(child_x, 1))
child_graph.output(child_x)
child_gm = GraphModule({}, child_graph)
outer_root = torch.nn.Module()
outer_root.add_module("child", child_gm)
outer_graph = Graph()
outer_x = outer_graph.placeholder("x")
child_ref = outer_graph.get_attr("child")
outer_graph.output((outer_x, child_ref))
outer_gm = GraphModule(outer_root, outer_graph)
outer_gm.graph.eliminate_dead_code()
nested_child_dead_removed = not any(
    node.op == "call_function" for node in child_gm.graph.nodes
)


effect_graph = Graph()
effect_x = effect_graph.placeholder("x")
first_value = effect_graph.placeholder("first_value")
second_value = effect_graph.placeholder("second_value")
first_copy = effect_graph.call_function(
    torch.ops.aten.copy_.default, args=(effect_x, first_value)
)
second_copy = effect_graph.call_function(
    torch.ops.aten.copy_.default, args=(effect_x, second_value)
)
effect_graph.output(effect_x)
effect_gm = GraphModule({}, effect_graph)
effect_graph.lint()
original_order = effect_gm(
    torch.tensor(0.0), torch.tensor(1.0), torch.tensor(2.0)
).item()

# Both copies use the same input but neither consumes the other copy's return.
# Moving one after the other preserves FX data-topology while changing effects.
second_copy.append(first_copy)
effect_graph.lint()
effect_gm.recompile()
reordered = effect_gm(
    torch.tensor(0.0), torch.tensor(1.0), torch.tensor(2.0)
).item()
both_effect_orders_lint = True
effect_reorder_changes_result = original_order != reordered

checks = {
    "pure_dead_removed": pure_dead_removed,
    "dead_chain_removed": dead_chain_removed,
    "nested_child_dead_removed": nested_child_dead_removed,
    "impure_copy_retained": impure_copy_retained,
    "alias_observes_mutation": alias_observes_mutation,
    "both_effect_orders_lint": both_effect_orders_lint,
    "effect_reorder_changes_result": effect_reorder_changes_result,
}
if not all(checks.values()):
    raise AssertionError(f"effect/alias contract failed: {checks}")

for key, check in checks.items():
    print(f"{key}={check}")
print(f"effect_original_order_result={original_order}")
print(f"effect_reordered_result={reordered}")
