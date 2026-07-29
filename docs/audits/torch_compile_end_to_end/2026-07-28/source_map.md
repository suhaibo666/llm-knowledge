# `torch.compile` End-to-End Source Map

> Fixed checkout: `E:/97-codes/torch_parallel/p`  
> Commit: `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> Navigation rule: all locators in this course resolve against this detached checkout. The sibling
> `E:/97-codes/torch_parallel/pytorch` is a separate working copy with uncommitted work and must not
> be used for line-number navigation or forcibly switched to the audit commit.
> Purpose: map each volume to its load-bearing source entry points before prose is written.

## Volume A

| Topic | Entry points |
|---|---|
| Tensor/storage/layout/version | `c10/core/TensorImpl.h:2888-2914`; `c10/core/TensorImpl.h:2919-2932`; `c10/core/StorageImpl.h:55-75`; `c10/core/StorageImpl.h:76-97`; `c10/core/TensorImpl.h:324-350`; `c10/core/TensorImpl.h:351-380`; `c10/core/TensorImpl.h:381-410`; `c10/core/TensorImpl.h:411-416` |
| Differentiable views | `torch/csrc/autograd/variable.h:605-634`; `torch/csrc/autograd/variable.h:635-664`; `torch/csrc/autograd/variable.h:665-670`; `torch/csrc/autograd/variable.h:721-750`; `torch/csrc/autograd/variable.h:751-780`; `torch/csrc/autograd/variable.h:781-810`; `torch/csrc/autograd/variable.h:811-837` |
| Python operator object | `torch/_ops.py:837-866`; `torch/_ops.py:867-896`; `torch/_ops.py:897-925` |
| Dispatcher | `aten/src/ATen/core/dispatch/Dispatcher.h:861-881`; `aten/src/ATen/core/dispatch/Dispatcher.h:882-902` |
| Autograd edges/nodes/engine | `torch/csrc/autograd/edge.h:14-31`; `torch/csrc/autograd/node.h:112-131`; `torch/csrc/autograd/node.h:132-150`; `torch/csrc/autograd/engine.h:130-159`; `torch/csrc/autograd/engine.h:160-189`; `torch/csrc/autograd/engine.h:190-210` |
| eval-frame boundary | `torch/csrc/dynamo/eval_frame.c:226-245`; `torch/csrc/dynamo/eval_frame.c:246-264`; `torch/csrc/dynamo/eval_frame.c:616-639`; `torch/csrc/dynamo/eval_frame.c:640-663` |
| dispatch modes | `torch/utils/_python_dispatch.py:72-101`; `torch/utils/_python_dispatch.py:102-131`; `torch/utils/_python_dispatch.py:132-151` |
| FakeTensor | `torch/_subclasses/fake_tensor.py:834-853`; `torch/_subclasses/fake_tensor.py:854-872`; `torch/_subclasses/fake_tensor.py:1526-1550`; `torch/_subclasses/fake_tensor.py:1551-1575` |
| ProxyTensor/make_fx | `torch/fx/experimental/proxy_tensor.py:934-957`; `torch/fx/experimental/proxy_tensor.py:958-981`; `torch/fx/experimental/proxy_tensor.py:2102-2117`; `torch/fx/experimental/proxy_tensor.py:2118-2133`; `torch/fx/experimental/proxy_tensor.py:2134-2149`; `torch/fx/experimental/proxy_tensor.py:2150-2165` |
| public compile entry | `torch/__init__.py:3134-3153`; `torch/__init__.py:3154-3173`; `torch/__init__.py:3361-3378` |

## Volume B

| Topic | Entry points |
|---|---|
| compile context | `torch/_dynamo/eval_frame.py:1350-1379`; `torch/_dynamo/eval_frame.py:1380-1395`; `torch/_dynamo/eval_frame.py:1396-1410` |
| frame conversion | `torch/_dynamo/convert_frame.py:2295-2320`; `torch/_dynamo/convert_frame.py:2321-2345` |
| bytecode state | `torch/_dynamo/symbolic_convert.py:1458-1481`; `torch/_dynamo/symbolic_convert.py:1482-1505`; `torch/_dynamo/symbolic_convert.py:5493-5519`; `torch/_dynamo/symbolic_convert.py:5520-5545` |
| Python values/sources | `torch/_dynamo/variables/base.py`; `torch/_dynamo/source.py`; `torch/_dynamo/variables/builder.py` |
| OutputGraph | `torch/_dynamo/output_graph.py:741-765`; `torch/_dynamo/output_graph.py:766-790`; `torch/_dynamo/output_graph.py:1490-1517`; `torch/_dynamo/output_graph.py:1518-1547`; `torch/_dynamo/output_graph.py:1548-1570` |
| guards | `torch/_dynamo/guards.py:1267-1291`; `torch/_dynamo/guards.py:1292-1315`; `torch/_dynamo/guards.py:3150-3179`; `torch/_dynamo/guards.py:3180-3209`; `torch/_dynamo/guards.py:3210-3230` |
| cache limits | `torch/_dynamo/cache_size.py:72-89`; `torch/_dynamo/cache_size.py:90-106`; `torch/_dynamo/cache_size.py:142-160` |
| resume bytecode | `torch/_dynamo/resume_execution.py:328-351`; `torch/_dynamo/resume_execution.py:352-375` |
| backend registry | `torch/_dynamo/backends/registry.py:87-108`; `torch/_dynamo/backends/registry.py:109-130` |

## Volume C

Canonical source map and evidence live in:

- `docs/audits/pytorch_graph_series/2026-07-27/course_claim_ledger.jsonl`
- `wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/00_pytorch_graph_series_index.md`

## Volume D

| Topic | Entry points |
|---|---|
| Inductor orchestration | `torch/_inductor/compile_fx.py:2889-2914`; `torch/_inductor/compile_fx.py:2915-2940` |
| AOT runtime wrappers | `torch/_functorch/_aot_autograd/runtime_wrappers.py:189-218`; `torch/_functorch/_aot_autograd/runtime_wrappers.py:3882-3911` |
| async compile | `torch/_inductor/async_compile.py` |
| code/module cache | `torch/_inductor/codecache.py` |
| FX graph cache | `torch/_inductor/codecache.py`; `torch/_inductor/remote_cache.py` |
| wrapper execution | `torch/_inductor/codegen/wrapper.py` |
| CUDAGraph Trees | `torch/_inductor/cudagraph_trees.py:2243-2265`; `torch/_inductor/cudagraph_trees.py:2266-2288` |

## Volume E

| Topic | Entry points |
|---|---|
| logging registration | `torch/_logging/_registrations.py` |
| explain | `torch/_dynamo/eval_frame.py:1869-1884`; `torch/_dynamo/eval_frame.py:1885-1899`; `torch/_dynamo/eval_frame.py:1900-1915`; `torch/_dynamo/eval_frame.py:1916-1930` |
| graph-break reasons | `torch/_dynamo/exc.py`; `torch/_dynamo/graph_break_hints.py` |
| repro/minifier | `torch/_dynamo/repro/after_dynamo.py`; `torch/_dynamo/repro/after_aot.py`; `torch/_functorch/fx_minifier.py:51-80` |
| compiler bisector | `torch/_inductor/compiler_bisector.py` |
| correctness | `torch/_dynamo/backends/debugging.py`; `torch/_inductor/fx_passes/numeric_utils.py` |
| metrics | `torch/_dynamo/utils.py`; `torch/_inductor/metrics.py` |

## Volume F

| Topic | Entry points |
|---|---|
| Compiled Autograd | `torch/_dynamo/compiled_autograd.py:1634-1663`; `torch/_dynamo/compiled_autograd.py:1664-1693`; `torch/_dynamo/compiled_autograd.py:1694-1705`; `torch/csrc/dynamo/compiled_autograd.cpp`; `torch/csrc/dynamo/compiled_autograd.h` |
| activation checkpoint | `torch/utils/checkpoint.py`; `torch/_functorch/partitioners.py` |
| DDP optimization | `torch/_dynamo/backends/distributed.py`; `torch/nn/parallel/distributed.py` |
| FSDP/DTensor | `torch/distributed/fsdp`; `torch/distributed/_composable/fsdp`; `torch/distributed/tensor` |
| custom operators | `torch/_library/custom_ops.py`; `torch/library.py`; `torch/_library/fake_impl.py` |
| backend/device integration | `torch/_dynamo/backends/registry.py`; `torch/_dynamo/device_interface.py`; `torch/_inductor/codegen/common.py` |
| AOTInductor | `torch/_inductor/__init__.py:64-93`; `torch/_inductor/__init__.py:108-127`; `torch/_inductor/__init__.py:151-166`; `torch/_inductor/package/package.py:83-100`; `torch/_inductor/package/package.py:103-123`; `torch/csrc/inductor/aoti_runner/model_container_runner.cpp:84-99`; `torch/csrc/inductor/aoti_runner/model_container_runner.cpp:100-115` |
| freezing/inference | `torch/_inductor/freezing.py:28-57`; `torch/_inductor/freezing.py:58-87`; `torch/_inductor/freezing.py:98-127`; `torch/_inductor/freezing.py:130-144`; `torch/_inductor/cudagraph_utils.py:305-333`; `torch/_inductor/cudagraph_utils.py:343-368` |
