# 01 · 算子分发与设备接入 — 目录索引

> PyTorch 运行时基石:ATen Dispatcher 的 DispatchKey/KeySet 优先级与 redispatch;以及 PrivateUse1 out-of-tree 设备接入机制(NPU 经由它注册为加速器)。
> 知识分层:overview(本索引)→ quick start → deep dive(约定见 [[01_ai_frameworks/index]])。
> 最后更新: 2026-07-15

---

## 页面列表(按层次)

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[device_integration_quickstart]] | **quick start** | 最小 PrivateUse1 接入步骤(`rename_privateuse1_backend`/`generate_methods_for_privateuse1_backend` + C++ DeviceGuardImpl/`TORCH_LIBRARY_IMPL(...,PrivateUse1,...)`,基于 torch_openreg);9 接入点速查;dispatch 排查命令(`_dispatch_dump`/`_dispatch_has_kernel_for_dispatch_key`/`__torch_dispatch__`) |
| [[pytorch_dispatcher_analysis]] | deep dive | DispatchKey/KeySet 优先级、redispatch 洋葱、boxed/unboxed、torchgen 代码生成、`__torch_dispatch__` 自定义分发 |
| [[privateuse1_device_integration_analysis]] | deep dive | PrivateUse1 设备键、9 个接入点(device/guard/hooks/operators/amp/autoload/profiler/distributed/ci)、torch_openreg 参考实现 |
| [[torch_npu_upstream_adaptation_analysis]] | overview / comparison | torch_npu 相对 upstream 的三层边界：标准插件面、Ascend 硬件实现面、Dynamo/Inductor/Graph/Distributed 兼容补丁面；当前差异分类与收敛优先级 |

---

## 关联域

- [[01_eager_runtime/03_op_registration/index]] — 算子如何注册进 dispatcher(NPU op-plugin 供给侧)
- [[02_compile_stack/01_dynamo/index]] — torch.compile 前端
- [[01_ai_frameworks/index]] — 本域总索引
