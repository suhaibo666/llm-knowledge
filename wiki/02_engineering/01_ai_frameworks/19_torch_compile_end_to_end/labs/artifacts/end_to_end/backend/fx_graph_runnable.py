
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = 'C:/Users/suhaibo/AppData/Local/Temp/torchinductor_suhaibo/tmptr4puytb'
os.environ['TRITON_CACHE_DIR'] = 'C:/Users/suhaibo/AppData/Local/Temp/torchinductor_suhaibo/tmptr4puytb/triton'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.recompile_limit = 8
torch._dynamo.config.accumulated_recompile_limit = 256
torch._dynamo.config.specialize_int = False
torch._dynamo.config.specialize_float = False
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = True
torch._dynamo.config.capture_scalar_outputs = False
torch._dynamo.config.capture_dynamic_output_shape_ops = False
torch._dynamo.config.prefer_deferred_runtime_asserts_over_guards = False
torch._dynamo.config.do_not_emit_runtime_asserts = False
torch._dynamo.config.dont_skip_tracing = False
torch._dynamo.config.allow_rnn = False
torch._dynamo.config.allow_empty_graphs = False
import part4_artifact_bundle
torch._inductor.config._post_fusion_custom_pass = part4_artifact_bundle.capture_post_fusion
torch._inductor.config.max_fusion_size = 64
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.trace.debug_dir = 'C:\\Users\\suhaibo\\AppData\\Local\\Temp\\graph_series_codegen_hd0hi_60\\trace'
torch._inductor.config.trace.provenance_tracking_level = 1
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.9.1+cpu
# torch cuda version: None
# torch git version: 5811a8d7da873dd699ff6687092c225caffcf1bb


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        mm = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        add = torch.ops.aten.add.Tensor(mm, arg2_1);  mm = arg2_1 = None
        view_1 = torch.ops.aten.view.default(add, [-1]);  add = None
        add_1 = torch.ops.aten.add.Tensor(view_1, 0.25);  view_1 = None
        view_2 = torch.ops.aten.view.default(add_1, [3, 4]);  add_1 = None
        sin = torch.ops.aten.sin.default(view_2);  view_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(sin, [1])
        return (sin, sum_1)
        
def load_args(reader):
    buf0 = reader.storage(None, 48)
    reader.tensor(buf0, (3, 4), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 64)
    reader.tensor(buf1, (4, 4), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 16)
    reader.tensor(buf2, (4,), is_leaf=True)  # arg2_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)