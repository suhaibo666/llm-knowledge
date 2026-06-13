# 01 · 算子分发与设备接入 — 目录索引

> PyTorch 运行时基石:ATen Dispatcher 的 DispatchKey/KeySet 优先级与 redispatch;以及 PrivateUse1 out-of-tree 设备接入机制(NPU 经由它注册为加速器)。
> 最后更新: 2026-06-13

---

## 页面列表

| 页面 | 类型 | 核心主题 |
|------|------|---------|
| [[pytorch_dispatcher_analysis]] | Analysis | DispatchKey/KeySet 优先级、redispatch 洋葱、boxed/unboxed、torchgen 代码生成、`__torch_dispatch__` 自定义分发 |
| [[privateuse1_device_integration_analysis]] | Analysis | PrivateUse1 设备键、9 个接入点(device/guard/hooks/operators/amp/autoload/profiler/distributed/ci)、torch_openreg 参考实现 |

---

## 关联域

- [[07_op_registration/index]] — 算子如何注册进 dispatcher(NPU op-plugin 供给侧)
- [[02_dynamo/index]] — torch.compile 前端
- [[01_ai_frameworks/index]] — 本域总索引
