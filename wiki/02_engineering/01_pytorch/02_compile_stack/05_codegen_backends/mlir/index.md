---
title: "05 · MLIR 编译栈 — 目录索引"
---

# 05 · MLIR 编译栈 — 目录索引

> MLIR 核心概念、torch-mlir Pass 管线、Triton vs MLIR 选型对比。**NPU MLIR 后端**深度分析见 [[02_compile_stack/05_codegen_backends/mlir/npu/index]]。
> 知识分层:quickstart→deep dive(约定见 [[01_pytorch/index]])。通用篇先从 [[01_torch_mlir_quickstart]] 快速上手,再按「概念→Pass 管线→选型对比」deep dive 递进。
> 注:torch-mlir / 上游 MLIR 内容多源自社区项目,本地代码库不含其源码(标注「上游,本地不可验证」)。
> 最后更新: 2026-06-15

---

## MLIR 基础与通用(按层次)

> **段位与阅读顺序**(kb-reorg P4 Task 9.5,2026-07-30):段 0(01-09)入门;段 1(10-19)核心机制——MLIR 基础概念先于按执行顺序的 pass 管线;段 3(30-39)对照——选型指南收尾。本目录内容页共 4 篇,不占段 2。

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[01_torch_mlir_quickstart]] | quick start(段 0) | torch-mlir 上手:生态定位(与 Inductor 关系)、何时用、最小 backend 骨架、output_type 选择 |
| [[10_mlir_core_concepts]] | overview/基础(段 1) | MLIR 基础:Dialect、Pass、IR 注册、递降原理;Mesh Dialect、IREE、StableHLO、Triton 3.x |
| [[11_torch_mlir_pass_pipeline_analysis]] | deep dive(段 1) | torch-mlir Pass 管线:按执行顺序的 Pass 完整分析(源自 torch-mlir 上游) |
| [[30_triton_vs_mlir_backend_analysis]] | deep dive(段 3,对照) | Triton vs Torch-MLIR:六阶段概念对等映射、优劣势、选型指南 |

---

## 硬件子目录

| 目录 | 核心主题 |
|------|---------|
| [[02_compile_stack/05_codegen_backends/mlir/npu/index]] | NPU MLIR 后端:IR 回溯、Bisheng 编译器、六阶段适配、Scheduler monkey-patch、AKG 集成 |

---

## 关联域

- [[02_compile_stack/04_inductor/index]] — TorchInductor(MLIR 作为 Inductor 后端路径之一)
- [[02_compile_stack/04_inductor/npu/index]] — NPU Inductor(Triton/MLIR 路径对比)
- [[02_compile_stack/05_codegen_backends/index]] — Codegen 后端总索引
- [[01_pytorch/index]] — 本域总索引
