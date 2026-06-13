# 08 · 算子调优 — 目录索引

> 跨硬件的算子级性能优化:Roofline、Memory/Compute Bound、融合策略(GPU/NPU),以及 Tile 级 DSL(TileLang)。
> 最后更新: 2026-06-13

---

## 页面列表

| 页面 | 类型 | 核心主题 |
|------|------|---------|
| [[operator_optimization_guide]] | Guide | GPU/NPU 算子调优体系:Roofline、Memory/Compute Bound 优化、AscendC、融合策略;Host CPU fallback 与 AICPU 辨析 |
| [[tilelang_analysis]] | Analysis | TileLang:Tile-Level IR、Host Codegen、Z3 SMT 验证、通算 wave 绑定(概念级,基于公开资料) |

---

## 关联域

- [[04_inductor/index]] — Inductor(codegen 自动调优)
- [[05_codegen_backends/mlir/index]] — MLIR codegen
- [[05_gpu_kernel/index]] — GPU Kernel 开发(执行层级、Tensor Core/MMA)
- [[01_ai_frameworks/index]] — 本域总索引
