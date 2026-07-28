import operator

import torch
from torch.fx import Graph, symbolic_trace
from torch.utils import _pytree as pytree


def f(x):
    y = x + x
    return y * 2


gm = symbolic_trace(f)
nodes = list(gm.graph.nodes)
x = next(node for node in nodes if node.op == "placeholder")
add = next(
    node
    for node in nodes
    if node.op == "call_function" and node.target is operator.add
)
mul = next(
    node
    for node in nodes
    if node.op == "call_function" and node.target is operator.mul
)

print(f"x_distinct_users={len(x.users)}")
uses = sum(leaf is x for leaf in pytree.tree_leaves(add.args))
print(f"x_uses_in_add={uses}")
print(f"before_replace={gm(torch.tensor(2.0)).item()}")

try:
    gm.graph.erase_node(add)
except RuntimeError as exc:
    print(f"erase_live_node={type(exc).__name__}")

with gm.graph.inserting_before(add):
    replacement = gm.graph.call_function(operator.add, args=(x, 1))
add.replace_all_uses_with(replacement)
gm.graph.erase_node(add)
gm.graph.lint()
gm.recompile()
print(f"after_replace={gm(torch.tensor(2.0)).item()}")

foreign = Graph()
foreign_x = foreign.placeholder("foreign_x")
mul.args = (foreign_x, 2)
try:
    gm.graph.lint()
except RuntimeError as exc:
    print(f"lint_cross_graph={type(exc).__name__}")

