# 05 · NPU MLIR 后端 — 目录索引

> 基于 MLIR 的 NPU torch.compile 路径。原三篇(技术架构 / 六阶段管线 / 深度分析)重叠 65-75%,已**合并为单篇综合 deepdive**,信息无损。
> 知识分层:overview(本索引)→ quick start → deep dive(约定见 [[01_ai_frameworks/index]])。
> 核验基准:torch_npu **v2.7.1.post5**(`torch_npu/_inductor/ascend_npu_ir/`)
> 最后更新: 2026-06-13

---

## 页面列表(按层次)

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[npu_mlir_quickstart]] | **quick start** | 启用 MLIR 后端(`TORCHINDUCTOR_NPU_BACKEND=mlir` / `config.npu_backend` / `options`)、`compile_mode`/`enable_graph_trace`/autotune/精度校验等真实开关、bishengir-compile、如何确认走了 MLIR 路径 |
| [[npu_mlir_backend_technical_analysis]] | deep dive | NPU MLIR 后端综合分析:架构与组件、TracedGraph 机制、融合规则、毕昇编译与 60 维 autotune、编译模式状态机;六阶段适配主线、三层 Pass、15 Monkey Patch 分组、Fallback 双通道、与社区遵循/打破、演进建议。Triton 门控已订正(`patch_has_triton` 对 NPU 返回 True,非强制禁用) |

> deep dive 由原 `npu_mlir_backend_technical_analysis` + `npu_mlir_pipeline_analysis`(六阶段)+ `npu_mlir_backend_deep_analysis`(社区对齐/演进)三篇合并而成。

---

## 关联域

- [[02_compile_stack/05_codegen_backends/mlir/index]] — 通用 MLIR(本目录的母域)
- [[02_compile_stack/04_inductor/npu/index]] — NPU Inductor(MLIR 为其三路径之一)
- [[03_runtime_graphs/npu/index]] — NPU Graphs(另一条路径)
- [[01_ai_frameworks/index]] — 本域总索引
