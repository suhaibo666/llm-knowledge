import torch
from torch import nn


def run_compile_mode(dynamic):
    compile_count = 0
    saw_symint = False

    def backend(gm, example_inputs):
        nonlocal compile_count, saw_symint
        compile_count += 1
        saw_symint |= any(isinstance(value, torch.SymInt) for value in example_inputs)
        return gm.forward

    def f(x):
        return x.sin() + x.shape[0]

    torch._dynamo.reset()
    compiled = torch.compile(f, backend=backend, dynamic=dynamic)
    for batch in (3, 4, 5):
        compiled(torch.randn(batch, 8))
    return compile_count, saw_symint


static_count, _ = run_compile_mode(False)
auto_count, auto_has_symint = run_compile_mode(None)
print(f"dynamic_false_compiles={static_count}")
print(f"dynamic_none_compiles={auto_count}")
print(f"dynamic_none_second_graph_has_symint={auto_has_symint}")


class DynamicModule(nn.Module):
    def forward(self, x):
        return x.cos() * 2


batch = torch.export.Dim("batch", min=2, max=8)
ep = torch.export.export(
    DynamicModule(),
    (torch.randn(3, 8),),
    dynamic_shapes={"x": {0: batch}},
)
print(f"export_range_constraints={len(ep.range_constraints)}")

