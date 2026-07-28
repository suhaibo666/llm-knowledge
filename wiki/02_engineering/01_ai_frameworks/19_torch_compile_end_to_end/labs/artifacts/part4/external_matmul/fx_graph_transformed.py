class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 16]", arg1_1: "f32[16, 32]"):
         # File: E:\97-codes\torch_parallel\llm-knowledge\wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part4_artifact_bundle.py:35 in matmul_model, code: return x @ weight
        mm: "f32[8, 32]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        return (mm,)
        