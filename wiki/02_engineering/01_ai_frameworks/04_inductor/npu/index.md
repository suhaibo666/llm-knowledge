# 04 · NPU Inductor 后端 — 目录索引

> torch_npu 的 Inductor 后端:三条 torch.compile 路径(Triton / ACLGraph / MLIR)、NPU 特定 lowering 与 fallback、monkey patch、按硬件特性组织的优化思想、NPU 调试。
> 阅读路径 **overview → quick start → deep dive**(约定见 [[01_ai_frameworks/index]])。upstream 通用 Inductor 见 [[04_inductor/index]]。
> 核验基准:torch_npu **v2.7.1.post5**(`torch_npu/_inductor/`)
> 最后更新: 2026-06-15

---

## overview(概览)

| 页面 | 核心主题 |
|------|---------|
| [[npu_compile_paths_overview]] | NPU compile 路径全景:三条路径(Triton/ACLGraph/MLIR)差异、收益、演进;GPU vs NPU Dynamic Shape 难易度对比 |

## quick start(上手)

| 页面 | 核心主题 |
|------|---------|
| [[npu_compile]] | NPU 编译工作流、三种编译模式、毕昇编译器接口、Autotune、在线精度校验 |
| [[npu_debug_guide]] | NPU torch.compile 调试:环境变量、API/脚本、NPU vs CUDA 调试差异、常见问题速修(原 upstream 调试页 §11 迁出) |

---

## deep dive

| 页面 | 核心主题 |
|------|---------|
| [[NPU_Inductor_Backend_Analysis]] | NPU Inductor 后端集成架构;5 后端融合规则与性能对比;后端混合使用机制(MultiTemplateBuffer、Prologue/Epilogue Fusion、4 实战场景);**NPU 适配补充**(后端注册/初始化/RNG patch/特定配置,来自 upstream 技术分析迁入) |
| [[npu_lowering_guide]] | NPU 特定 lowering 与算子映射;FALLBACK_LIST 黑名单策略;§9 当前源码复核 |
| [[npu_triton_backend_deep_analysis]] | Triton/default 路径深度:golden_var_list、CATLASS/CK GEMM、monkey patch、NPUIndexTritonKernel |
| [[npu_inductor_optimization_analysis]] | 优化思想全景(why):硬件特性→优化思想→实际案例,跨 Triton/MLIR/DVM 三后端 |

---

## 关联域

- [[04_inductor/index]] — 通用 Inductor(本目录的硬件无关母域)
- [[05_codegen_backends/mlir/npu/index]] — NPU MLIR 后端(路径之一)
- [[06_graphs/npu/index]] — NPU Graphs / ACLGraph(路径之一)
- [[07_op_registration/npu/index]] — 算子供给侧(lowering/fallback 的算子来源)
- [[01_ai_frameworks/index]] — 本域总索引
