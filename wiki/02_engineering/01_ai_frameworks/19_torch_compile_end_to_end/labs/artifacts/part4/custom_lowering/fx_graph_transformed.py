class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 16]"):
         # File: E:\97-codes\torch_parallel\llm-knowledge\wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part4_artifact_bundle.py:393 in custom_lowering_model, code: return custom_scale(x, 2.0).sum(dim=1)
        scale: "f32[8, 16]" = torch.ops.graph_series_lab.scale.default(arg0_1, 2.0);  arg0_1 = None
        sum_1: "f32[8]" = torch.ops.aten.sum.dim_IntList(scale, [1]);  scale = None
        return (sum_1,)
        