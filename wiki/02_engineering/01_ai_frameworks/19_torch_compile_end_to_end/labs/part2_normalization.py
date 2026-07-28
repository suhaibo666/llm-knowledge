import torch
from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx


def mutating_view(x):
    view = x.view(-1)
    view.add_(1)
    return x * 2


example = torch.randn(2, 3)
original = make_fx(mutating_view)(example.clone())
functional_fn = torch.func.functionalize(mutating_view, remove="mutations_and_views")
functional = make_fx(functional_fn)(example.clone())

original_targets = [str(n.target) for n in original.graph.nodes if n.op == "call_function"]
functional_targets = [
    str(n.target) for n in functional.graph.nodes if n.op == "call_function"
]
print(f"original_has_inplace={any('add_' in target for target in original_targets)}")
print(
    "functional_has_outplace_add="
    + str(any("add.Tensor" in target for target in functional_targets))
)
expected_input = example.clone()
actual_input = example.clone()
expected = mutating_view(expected_input)
actual = functional_fn(actual_input)
print(f"functional_output_matches={torch.allclose(expected, actual)}")
print(f"functional_input_semantics_match={torch.allclose(expected_input, actual_input)}")


def silu_fn(x):
    return torch.ops.aten.silu.default(x)


plain = make_fx(silu_fn)(example)
decompositions = get_decompositions([torch.ops.aten.silu.default])
decomposed = make_fx(silu_fn, decomposition_table=decompositions)(example)
plain_targets = [str(n.target) for n in plain.graph.nodes if n.op == "call_function"]
decomposed_targets = [
    str(n.target) for n in decomposed.graph.nodes if n.op == "call_function"
]
print(f"plain_has_silu={any('silu' in target for target in plain_targets)}")
print(f"decomposed_has_silu={any('silu' in target for target in decomposed_targets)}")

