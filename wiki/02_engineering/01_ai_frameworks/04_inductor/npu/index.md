# 04 · NPU Inductor 后端 — 目录索引

> torch_npu 的 Inductor 后端:三条 torch.compile 路径(Triton / ACLGraph / MLIR)、NPU 特定 lowering 与 fallback、35+ monkey patch、按硬件特性组织的优化思想。
> 核验基准:torch_npu **v2.7.1.post5**(`torch_npu/_inductor/`)
> 最后更新: 2026-06-13

---

## 路径总览与编译流程

| 页面 | 核心主题 |
|------|---------|
| [[npu_compile_paths_overview]] | **NPU compile 路径全景**:三条路径(Triton/ACLGraph/MLIR)差异、收益、演进;含 GPU vs NPU Dynamic Shape 难易度对比 |
| [[npu_compile]] | NPU 编译工作流、Autotune、精度校验 |

## 后端集成与机制

| 页面 | 核心主题 |
|------|---------|
| [[NPU_Inductor_Backend_Analysis]] | NPU Inductor 后端集成架构分析 |
| [[NPU_Inductor_Backend_Mechanism]] | NPU 后端内部实现机制 |
| [[npu_triton_backend_deep_analysis]] | **Triton/default 路径深度分析**:golden_var_list、CATLASS/CK GEMM、35+ monkey patch、NPUIndexTritonKernel |

## Lowering 与优化

| 页面 | 核心主题 |
|------|---------|
| [[npu_lowering_guide]] | NPU 特定 lowering 步骤与算子映射;FALLBACK_LIST 黑名单策略 |
| [[npu_inductor_optimization_analysis]] | **优化思想全景(why)**:硬件特性→优化思想→实际案例,跨 Triton/MLIR/DVM 三后端 |

---

## 关联域

- [[04_inductor/index]] — 通用 Inductor(本目录的硬件无关母域)
- [[05_codegen_backends/mlir/npu/index]] — NPU MLIR 后端(路径之一)
- [[06_graphs/npu/index]] — NPU Graphs / ACLGraph(路径之一)
- [[07_op_registration/npu/index]] — 算子供给侧(lowering/fallback 的算子来源)
- [[01_ai_frameworks/index]] — 本域总索引
