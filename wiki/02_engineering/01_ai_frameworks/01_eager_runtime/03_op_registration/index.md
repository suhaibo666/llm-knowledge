# 03 · 算子接入(算子供给侧)— 目录索引

> 算子如何从定义进入 PyTorch dispatcher 并最终「入图」。通用注册机制(`TORCH_LIBRARY`/torchgen)见 [[pytorch_dispatcher_analysis]];**NPU 的算子供给侧(op-plugin)** 见 [[01_eager_runtime/03_op_registration/npu/index]]。
> 最后更新: 2026-06-13

---

## 硬件子目录

| 目录 | 核心主题 |
|------|---------|
| [[01_eager_runtime/03_op_registration/npu/index]] | op-plugin:config 配置与分类、yaml→codegen→dispatcher 注册链路与生效时机、算子入图判别 |

---

## 关联域

- [[01_eager_runtime/02_dispatcher_and_device/index]] — Dispatcher 机制(注册的目标)
- [[02_compile_stack/04_inductor/npu/index]] — Inductor lowering/fallback(入图第二关)
- [[03_runtime_graphs/npu/index]] — ACLGraph(入图第三关)
- [[01_ai_frameworks/index]] — 本域总索引
