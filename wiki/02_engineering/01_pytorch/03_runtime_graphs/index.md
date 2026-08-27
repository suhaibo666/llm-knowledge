---
title: "03 · 运行时图捕获（CUDA / NPU Graphs）— 目录索引"
---

# 03 · 运行时图捕获（CUDA / NPU Graphs）— 目录索引

> 捕获并回放运行时计算图，以消除 host 侧的算子下发开销。CUDA Graphs 与 NPU Graphs（ACLGraph）按硬件分列。本层建立在 [[02_compile_stack/index]] 之上（`mode="reduce-overhead"` 经 cudagraph trees 集成）；硬件子目录不再二级细分，直接作为架构层。
> 最后更新：2026-07-30

---

## 硬件子目录

| 目录 | 核心主题 |
|------|---------|
| [[03_runtime_graphs/cuda/index]] | CUDA Graphs:完整使用指南、捕获/回放时序、最佳实践 |
| [[03_runtime_graphs/npu/index]] | NPU Graphs / ACLGraph:编译集成、内存管理/重用、make_graphed_callables、与 CUDA 对比 |

---

## 关联域

- [[02_compile_stack/04_inductor/index]] — Inductor（`mode="reduce-overhead"` 经 cudagraph trees 集成）
- [[02_compile_stack/04_inductor/npu/index]] — NPU Inductor（ACLGraph 为其路径之一）
- [[02_train_frameworks/megatron-lm/index]] — Megatron-LM（CUDA Graphs 使用场景）
- [[01_pytorch/index]] — 本域总索引
