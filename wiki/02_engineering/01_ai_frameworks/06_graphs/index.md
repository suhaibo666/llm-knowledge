# 06 · 运行时图捕获(CUDA / NPU Graphs)— 目录索引

> 运行时计算图的捕获与回放,消除 host 侧算子下发开销。CUDA Graphs 与 NPU Graphs(ACLGraph)按硬件分列。
> 最后更新: 2026-06-13

---

## 硬件子目录

| 目录 | 核心主题 |
|------|---------|
| [[06_graphs/cuda/index]] | CUDA Graphs:完整使用指南、捕获/回放时序、最佳实践 |
| [[06_graphs/npu/index]] | NPU Graphs / ACLGraph:编译集成、内存管理/重用、make_graphed_callables、与 CUDA 对比 |

---

## 关联域

- [[04_inductor/index]] — Inductor(`mode="reduce-overhead"` 经 cudagraph trees 集成)
- [[04_inductor/npu/index]] — NPU Inductor(ACLGraph 为其路径之一)
- [[02_train_frameworks/megatron-lm/index]] — Megatron-LM(CUDA Graphs 使用场景)
- [[01_ai_frameworks/index]] — 本域总索引
