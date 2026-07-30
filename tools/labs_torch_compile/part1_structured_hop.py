import operator

import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx


def structured(x):
    values, indices = torch.max(x, dim=0)
    return {"value": values, "metadata": (indices, x.shape[0])}


tuple_gm = make_fx(structured)(torch.tensor([1.0, 4.0, 2.0]))
has_getitem = any(
    node.op == "call_function" and node.target is operator.getitem
    for node in tuple_gm.graph.nodes
)
output_node = next(node for node in tuple_gm.graph.nodes if node.op == "output")
print(f"tuple_graph_has_getitem={has_getitem}")
print(f"output_is_nested_dict={isinstance(output_node.args[0], dict)}")


class CondModule(nn.Module):
    def forward(self, x):
        pred = x.sum() > 0

        def true_fn(value):
            return value * 2

        def false_fn(value):
            return value - 1

        return torch.cond(pred, true_fn, false_fn, (x,))


ep = torch.export.export(CondModule(), (torch.tensor([1.0, 2.0]),))
outer_cond = next(
    node
    for node in ep.graph.nodes
    if node.op == "call_function" and "cond" in str(node.target)
)
children = {
    name: child
    for name, child in ep.graph_module.named_children()
    if isinstance(child, torch.fx.GraphModule)
}
for child in children.values():
    child.graph.lint()

module = ep.module()
positive = module(torch.tensor([1.0, 2.0])).sum().item()
negative = module(torch.tensor([-1.0, -2.0])).sum().item()
child_names = ",".join(children)
output_specs = [
    type(next(n for n in child.graph.nodes if n.op == "output").args[0]).__name__
    for child in children.values()
]

print(f"cond_outer_target={outer_cond.target}")
print("cond_subgraphs=" + child_names)
print(f"cond_positive={positive}")
print(f"cond_negative={negative}")
print(f"branch_output_specs_equal={len(set(output_specs)) == 1}")

