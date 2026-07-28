
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = 'C:/Users/suhaibo/AppData/Local/Temp/torchinductor_suhaibo/tmpdn3s_c3m'
os.environ['TRITON_CACHE_DIR'] = 'C:/Users/suhaibo/AppData/Local/Temp/torchinductor_suhaibo/tmpdn3s_c3m/triton'

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

torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.trace.debug_dir = 'C:\\Users\\suhaibo\\AppData\\Local\\Temp\\graph_series_real_trace_bn3u7l09\\trace'
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

    
    
    def forward(self, arg0_1, arg1_1):
        mm = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        return (mm,)
        
def load_args(reader):
    buf0 = reader.storage(None, 512)
    reader.tensor(buf0, (8, 16), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 2048)
    reader.tensor(buf1, (16, 32), is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)