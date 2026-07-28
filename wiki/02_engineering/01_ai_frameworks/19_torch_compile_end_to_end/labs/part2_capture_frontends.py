import torch
from torch import nn
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x).relu()


model = Model().eval()
example = torch.randn(2, 4)

symbolic_gm = symbolic_trace(model)
make_fx_gm = make_fx(model)(example)
print(
    "symbolic_has_call_module="
    + str(any(node.op == "call_module" for node in symbolic_gm.graph.nodes))
)
print(
    "make_fx_has_call_module="
    + str(any(node.op == "call_module" for node in make_fx_gm.graph.nodes))
)

compile_count = 0
captured_dynamo_graphs = []


def backend(gm, example_inputs):
    global compile_count
    compile_count += 1
    captured_dynamo_graphs.append(gm)
    return gm.forward


torch._dynamo.reset()
compiled = torch.compile(model, backend=backend)
compiled(example)
print(f"dynamo_backend_graphs={compile_count}")
dynamo_example_value_meta_recorded = any(
    "example_value" in node.meta
    for gm in captured_dynamo_graphs
    for node in gm.graph.nodes
)

torch._dynamo.reset()
explanation = torch._dynamo.explain(model)(example)
dynamo_guards_recorded = len(explanation.out_guards) > 0


def explicit_break(x):
    first = torch.sin(x)
    torch._dynamo.graph_break()
    return torch.cos(first)


break_graphs = []


def break_backend(gm, example_inputs):
    break_graphs.append(gm)
    return gm.forward


torch._dynamo.reset()
compiled_break = torch.compile(explicit_break, backend=break_backend)
break_input = torch.randn(2, 4)
break_actual = compiled_break(break_input)
break_expected = explicit_break(break_input)
explicit_graph_break_backend_graphs = len(break_graphs)

ep = torch.export.export(model, (example,))
export_input_kinds = ",".join(
    spec.kind.name for spec in ep.graph_signature.input_specs
)
print(
    "export_input_kinds="
    + export_input_kinds
)
print(f"export_range_constraints={len(ep.range_constraints)}")
print(f"dynamo_guards_recorded={dynamo_guards_recorded}")
print(
    "dynamo_example_value_meta_recorded="
    + str(dynamo_example_value_meta_recorded)
)
print(
    "explicit_graph_break_backend_graphs="
    + str(explicit_graph_break_backend_graphs)
)

checks = {
    "symbolic_has_call_module": any(
        node.op == "call_module" for node in symbolic_gm.graph.nodes
    ),
    "make_fx_has_call_module": not any(
        node.op == "call_module" for node in make_fx_gm.graph.nodes
    ),
    "dynamo_backend_graphs": compile_count == 1,
    "dynamo_guards_recorded": dynamo_guards_recorded,
    "dynamo_example_value_meta_recorded": dynamo_example_value_meta_recorded,
    "explicit_graph_break_backend_graphs": explicit_graph_break_backend_graphs
    == 2,
    "explicit_graph_break_output_matches": torch.allclose(
        break_actual, break_expected
    ),
    "export_input_kinds": export_input_kinds
    == "PARAMETER,PARAMETER,USER_INPUT",
}
if not all(checks.values()):
    raise AssertionError(f"capture frontend contract failed: {checks}")
