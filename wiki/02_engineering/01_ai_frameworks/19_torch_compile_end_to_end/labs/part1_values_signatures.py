import torch
from torch import nn
from torch.fx import symbolic_trace


class ReadState(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.register_buffer("bias", torch.tensor(1.0))

    def forward(self, x):
        return x * self.weight + self.bias


symbolic_gm = symbolic_trace(ReadState())
print("symbolic_fx_ops=" + ",".join(n.op for n in symbolic_gm.graph.nodes))
print(
    "symbolic_parameter_access="
    + next(n.op for n in symbolic_gm.graph.nodes if n.target == "weight")
)

ep = torch.export.export(ReadState(), (torch.ones(2),))
try:
    ep(torch.ones(2))
    directly_callable = True
except RuntimeError:
    directly_callable = False
print(f"exported_program_callable={directly_callable}")
print(
    "export_input_kinds="
    + ",".join(spec.kind.name for spec in ep.graph_signature.input_specs)
)
print(
    "export_output_kinds="
    + ",".join(spec.kind.name for spec in ep.graph_signature.output_specs)
)
print(
    "export_non_output_has_val_meta="
    + str(all("val" in n.meta for n in ep.graph.nodes if n.op != "output"))
)
