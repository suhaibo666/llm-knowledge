# 05 · MLIR 编译栈 — 目录索引

> MLIR 核心概念、torch-mlir Pass 管线、Triton vs MLIR 选型对比。**NPU MLIR 后端**深度分析见 [[05_codegen_backends/mlir/npu/index]]。
> 注:torch-mlir / 上游 MLIR 内容多源自社区项目,本地代码库不含其源码(标注「上游,本地不可验证」)。
> 最后更新: 2026-06-13

---

## MLIR 基础与通用

| 页面 | 核心主题 |
|------|---------|
| [[mlir_core_concepts]] | MLIR 基础:Dialect、Pass、IR 注册、递降原理;Mesh Dialect、IREE、StableHLO、Triton 3.x |
| [[torch_mlir_pass_pipeline_analysis]] | torch-mlir Pass 管线:按执行顺序的 Pass 完整分析(源自 torch-mlir 上游) |
| [[triton_vs_mlir_backend_analysis]] | Triton vs Torch-MLIR:六阶段概念对等映射、优劣势、选型指南 |

---

## 硬件子目录

| 目录 | 核心主题 |
|------|---------|
| [[05_codegen_backends/mlir/npu/index]] | NPU MLIR 后端:IR 回溯、Bisheng 编译器、六阶段适配、Scheduler monkey-patch、AKG 集成 |

---

## 关联域

- [[04_inductor/index]] — TorchInductor(MLIR 作为 Inductor 后端路径之一)
- [[04_inductor/npu/index]] — NPU Inductor(Triton/MLIR 路径对比)
- [[05_codegen_backends/index]] — Codegen 后端总索引
- [[01_ai_frameworks/index]] — 本域总索引
