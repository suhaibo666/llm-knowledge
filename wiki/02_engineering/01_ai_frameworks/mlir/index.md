# MLIR 编译栈 — 目录索引

> 覆盖 MLIR 核心概念、Torch-MLIR Pass 管线、Triton vs MLIR 对比、NPU MLIR 后端深度分析
> 最后更新: 2026-05-15

---

## 页面列表

### MLIR 基础与通用

| 页面 | 核心主题 |
|------|---------|
| [[mlir_core_concepts]] | MLIR 基础: Dialect、Pass、IR 注册、递降原理; Mesh Dialect、IREE、StableHLO、Triton 3.x |
| [[torch_mlir_pass_pipeline_analysis]] | torch-mlir Pass 管线: 按执行顺序的 34 个 Pass 完整分析 |
| [[triton_vs_mlir_backend_analysis]] | Triton vs Torch-MLIR: 六阶段概念对等映射, 优劣势分析, 选型指南 |

### NPU MLIR 后端

| 页面 | 核心主题 |
|------|---------|
| [[npu_mlir_backend_deep_analysis]] | **MLIR 路径深度分析**: IR 回溯机制、Bisheng 编译器、Scheduler monkey-patch、auto_fallback |
| [[npu_mlir_pipeline_analysis]] | **NPU MLIR 六阶段适配全景**: GPU vs NPU 逐阶段对比、三层 Pass、15 个 Monkey Patch |
| [[NPU_MLIR_Backend_Technical_Analysis]] | 基于 MLIR 的 NPU 后端: TracedGraph 机制、编译模式状态机、AKG 集成 |

---

## 关联域

- [[../inductor/index]] — TorchInductor（MLIR 作为 Inductor 后端路径之一）
- [[../cudagraphs/npugraphs/index]] — NPU Graphs
- [[../index]] — AI 框架总索引
- [[../../01_theory/index]] — 理论研究
