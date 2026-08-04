# 01 · Eager 运行时地基 — 目录索引

> PyTorch 默认即时执行（eager mode）所依赖的数据模型与机制：张量表达、算子分发、算子定义与执行、反向自动微分、`torch.nn` 模块体系，以及运行时横切设施（缓存分配器/AMP/Profiler）。torch.compile 编译栈（见 [[02_compile_stack/index]]）建立在这套地基之上。

## 子目录

| 目录 | 核心主题 |
|------|---------|
| [[01_tensor_and_storage/index]] | 张量表达机制:`Tensor=intrusive_ptr<TensorImpl>`、Storage/视图别名、sizes/strides/dtype、DispatchKeySet |
| [[02_dispatcher_and_device/index]] | 算子分发(Dispatcher)、PrivateUse1 设备接入 |
| [[03_op_registration/index]] | 算子接入供给侧:`TORCH_LIBRARY`/torchgen 通用注册机制、NPU op-plugin |
| [[04_aten_op_execution/index]] | ATen 算子定义与执行:`native_functions.yaml`、torchgen 代码生成、结构化 kernel、boxing |
| [[05_autograd_engine/index]] | eager 反向自动微分引擎:Node/Edge DAG、多线程 Engine、AccumulateGrad、SavedVariable |
| [[06_nn_module_system/index]] | `torch.nn` 模块体系:Module/Parameter/Buffer 注册、state_dict、hooks、容器、lazy、Optimizer |
| [[07_memory_amp_profiler/index]] | 横切运行时:缓存内存分配器、AMP/autocast + GradScaler、Kineto Profiler |

## Related Pages

- [[01_ai_frameworks/index]] — 本域总索引（5 层架构导航）
- [[02_compile_stack/index]] — torch.compile 编译栈（建立在本层之上）
