# 05 · NPU MLIR 后端 — 目录索引

> 基于 MLIR 的 NPU torch.compile 路径深度分析:IR 回溯机制、Bisheng 编译器、六阶段适配、Scheduler monkey-patch、AKG 集成。
> 核验基准:torch_npu **v2.7.1.post5**(`torch_npu/_inductor/ascend_npu_ir/`)
> 最后更新: 2026-06-13

---

## 页面列表

| 页面 | 核心主题 |
|------|---------|
| [[npu_mlir_backend_deep_analysis]] | **MLIR 路径深度分析**:IR 回溯机制、Bisheng 编译器、Scheduler monkey-patch、auto_fallback |
| [[npu_mlir_pipeline_analysis]] | **NPU MLIR 六阶段适配全景**:GPU vs NPU 逐阶段对比、三层 Pass、Monkey Patch |
| [[NPU_MLIR_Backend_Technical_Analysis]] | 基于 MLIR 的 NPU 后端:TracedGraph 机制、编译模式状态机、AKG 集成 |

---

## 关联域

- [[05_codegen_backends/mlir/index]] — 通用 MLIR(本目录的母域)
- [[04_inductor/npu/index]] — NPU Inductor(MLIR 为其三路径之一)
- [[06_graphs/npu/index]] — NPU Graphs(另一条路径)
- [[01_ai_frameworks/index]] — 本域总索引
