class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[4, 4]", arg2_1: "f32[4]"):
         # File: E:\97-codes\torch_parallel\llm-knowledge\wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\series_artifact_bundle.py:66 in backend_core, code: projected = x.view(-1, 4) @ weight
        mm: "f32[3, 4]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        
         # File: E:\97-codes\torch_parallel\llm-knowledge\wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\series_artifact_bundle.py:67 in backend_core, code: biased = projected + offset
        add: "f32[3, 4]" = torch.ops.aten.add.Tensor(mm, arg2_1);  mm = arg2_1 = None
        
         # File: E:\97-codes\torch_parallel\llm-knowledge\wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\series_artifact_bundle.py:69 in backend_core, code: scratch.view(-1).add_(0.25)
        view_1: "f32[12]" = torch.ops.aten.view.default(add, [-1]);  add = None
        add_1: "f32[12]" = torch.ops.aten.add.Tensor(view_1, 0.25);  view_1 = None
        view_2: "f32[3, 4]" = torch.ops.aten.view.default(add_1, [3, 4]);  add_1 = None
        
         # File: E:\97-codes\torch_parallel\llm-knowledge\wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\series_artifact_bundle.py:70 in backend_core, code: activation = torch.sin(scratch)
        sin: "f32[3, 4]" = torch.ops.aten.sin.default(view_2);  view_2 = None
        
         # File: E:\97-codes\torch_parallel\llm-knowledge\wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\series_artifact_bundle.py:71 in backend_core, code: return activation, activation.sum(dim=1)
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(sin, [1])
        return (sin, sum_1)
        