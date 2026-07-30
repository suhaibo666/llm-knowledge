class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 16]"):
         # File: E:\97-codes\torch_parallel\llm-knowledge\wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part4_artifact_bundle.py:31 in pointwise_reduction, code: return torch.sin(x.sum(dim=1))
        sum_1: "f32[8]" = torch.ops.aten.sum.dim_IntList(arg0_1, [1]);  arg0_1 = None
        sin: "f32[8]" = torch.ops.aten.sin.default(sum_1);  sum_1 = None
        return (sin,)
        