import operator

import torch
from torch.fx import symbolic_trace


def f(x):
    return (x + 1).relu().sum()


x = torch.tensor([-1.0, 2.0], requires_grad=True)
y = f(x)
print(f"eager_grad_fn={type(y.grad_fn).__name__}")
print(f"eager_next={type(y.grad_fn.next_functions[0][0]).__name__}")

gm = symbolic_trace(f)
print("fx_ops=" + ",".join(node.op for node in gm.graph.nodes))

targets = []
for node in gm.graph.nodes:
    if node.op == "call_function" and node.target in (operator.add, torch.add):
        targets.append("add")
    elif node.op == "call_method" and node.target in ("relu", "sum"):
        targets.append(str(node.target))
print("fx_call_targets=" + ",".join(targets))

