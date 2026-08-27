---
title: "02 · 算子分发与设备接入 — 目录索引"
---

# 02 · 算子分发与设备接入 — 目录索引

> PyTorch 运行时基石:ATen Dispatcher 的 DispatchKey/KeySet 优先级与 redispatch;以及 PrivateUse1 out-of-tree 设备接入机制(NPU 经由它注册为加速器)。
> 知识分层:overview(本索引)→ quick start → deep dive(约定见 [[01_pytorch/index]])。
> 最后更新: 2026-07-15

---

## 页面列表(按层次)

> **段位与阅读顺序**(kb-reorg P4 Task 9.5,2026-07-30):段 0(01-09)入门;段 1(10-19)核心机制——dispatcher 基础机制(10)先于 PrivateUse1 完整接入面(11);段 2(20-29)深潜专题——custom backend/torch.compile 契约(20)、torch_npu 实例(21)按依赖 11 的顺序排列。

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[01_device_integration_quickstart]] | **quick start**(段 0) | 最小 PrivateUse1 接入步骤(`rename_privateuse1_backend`/`generate_methods_for_privateuse1_backend` + C++ DeviceGuardImpl/`TORCH_LIBRARY_IMPL(...,PrivateUse1,...)`,基于 torch_openreg);9 接入点速查;dispatch 排查命令(`_dispatch_dump`/`_dispatch_has_kernel_for_dispatch_key`/`__torch_dispatch__`) |
| [[10_pytorch_dispatcher_analysis]] | deep dive(段 1) | DispatchKey/KeySet 优先级、redispatch 洋葱、boxed/unboxed、torchgen 代码生成、`__torch_dispatch__` 自定义分发 |
| [[11_privateuse1_device_integration_analysis]] | deep dive(段 1) | PrivateUse1 设备键、9 个接入点(device/guard/hooks/operators/amp/autoload/profiler/distributed/ci)、torch_openreg 参考实现 |
| [[20_custom_backends_and_device_integration_analysis]] | deep dive(专题,段 2) | 设备接入 `torch.compile` 的另一层契约:Dynamo backend / Inductor device backend(含 `DeviceInterface`)/ dispatcher-custom-op backend 三者怎样分层组合;2026-07-30 迁入,与本目录 [[11_privateuse1_device_integration_analysis]]、`04_inductor` 的 [[34_codegen_extension_guide]] 三方划界(见页头 note) |
| [[21_torch_npu_upstream_adaptation_analysis]] | overview / comparison(段 2) | torch_npu 相对 upstream 的三层边界：标准插件面、Ascend 硬件实现面、Dynamo/Inductor/Graph/Distributed 兼容补丁面；当前差异分类与收敛优先级 |

---

## 关联域

- [[01_eager_runtime/03_op_registration/index]] — 算子如何注册进 dispatcher(NPU op-plugin 供给侧)
- [[02_compile_stack/01_dynamo/index]] — torch.compile 前端
- [[01_pytorch/index]] — 本域总索引
