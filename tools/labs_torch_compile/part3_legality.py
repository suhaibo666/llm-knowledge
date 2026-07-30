import operator

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


def exact_shape_add(x, y):
    if x.shape != y.shape:
        raise RuntimeError("exact_shape_add does not support broadcasting")
    return torch.add(x, y)


def add_fn(x, y):
    return x + y


def try_rewrite(x, y):
    gm = symbolic_trace(add_fn)
    ShapeProp(gm).propagate(x, y)
    add = next(
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target is operator.add
    )
    structural_match = True
    shape_legal = tuple(x.shape) == tuple(y.shape)
    if shape_legal:
        add.target = exact_shape_add
        gm.graph.lint()
        gm.recompile()
    return gm, structural_match, shape_legal


broadcast_x = torch.randn(2, 3)
broadcast_y = torch.randn(3)
broadcast_gm, matched, legal = try_rewrite(broadcast_x, broadcast_y)
print(f"broadcast_structural_match={matched}")
print(f"broadcast_rewrite_legal={legal}")
print(
    "broadcast_original_preserved="
    + str(torch.allclose(broadcast_gm(broadcast_x, broadcast_y), broadcast_x + broadcast_y))
)

base = torch.randn(3, 2)
exact_x = base.t().detach().requires_grad_(True)
exact_y = torch.randn(2, 3, requires_grad=True)
exact_gm, _, exact_legal = try_rewrite(exact_x, exact_y)
actual = exact_gm(exact_x, exact_y).sum()
actual.backward()
eager_x = exact_x.detach().clone().requires_grad_(True)
eager_y = exact_y.detach().clone().requires_grad_(True)
expected = add_fn(eager_x, eager_y).sum()
expected.backward()
print(f"transposed_exact_rewrite_legal={exact_legal}")
print(f"forward_matches={torch.allclose(actual.detach(), expected.detach())}")
print(f"gradient_matches={torch.allclose(exact_x.grad, eager_x.grad) and torch.allclose(exact_y.grad, eager_y.grad)}")

