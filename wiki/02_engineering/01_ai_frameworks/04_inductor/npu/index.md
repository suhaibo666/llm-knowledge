# 04 · NPU Inductor 后端 — 目录索引

> torch_npu 的 Inductor 后端:三条 torch.compile 路径(Triton / ACLGraph / MLIR)、NPU 特定 lowering 与 fallback、monkey patch、按硬件特性组织的优化思想、NPU 调试。
> 阅读路径 **overview → quick start → deep dive**(约定见 [[01_ai_frameworks/index]])。upstream 通用 Inductor 见 [[04_inductor/index]]。
> 核验基准:torch_npu **v2.7.1.post5**(`torch_npu/_inductor/`)
> 收录：除 torch_npu 内置后端外，另含**独立实验性** Linearize 后端 `npu_inductor_2.9.0`（PyTorch 2.9.0，见 [[npu_inductor_linearize_backend_analysis]]）。
> 最后更新: 2026-06-17

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
| [[npu_inductor_splittiling_backend_analysis]] | **内置 default（Split-Tiling 方案）**：Triton/default 路径深度——golden_var_list、CATLASS/CK GEMM、monkey patch、NPUIndexTritonKernel（对照实验 [[npu_inductor_linearize_backend_analysis]]） |
| [[npu_inductor_optimization_analysis]] | 优化思想全景(why):硬件特性→优化思想→实际案例,跨 Triton/MLIR/DVM 三后端 |
| [[npu_inductor_linearize_backend_analysis]] | **实验性 `npu_inductor_2.9.0`**（≠ 内置后端）：Linearize + 40-CU group dispatch、索引线性化、编译一次动态 shape、`NPU_MAX_FUSED_READS` 融合门控、r 轴 rsplit、与内置后端对比、可优化点 |
| [[npu_inductor_linearize_dynamic_shape_analysis]] | 实验后端动态 shape：编译一次、签名只传动态 numel/divisor、header 三件套与三情形 A/B/C（含 inner-loop）、permute 产物 |
| [[npu_inductor_linearize_vs_builtin_comparison]] | 三方 output code 逐行对比（GPU / 内置 Split-Tiling / 本后端 Linearize）+ §0 实测对标（torchbench 34 模型 / 京东 OneRec / test_all 算子 case） |

---

## 关联域

- [[04_inductor/index]] — 通用 Inductor(本目录的硬件无关母域)
- [[05_codegen_backends/mlir/npu/index]] — NPU MLIR 后端(路径之一)
- [[06_graphs/npu/index]] — NPU Graphs / ACLGraph(路径之一)
- [[07_op_registration/npu/index]] — 算子供给侧(lowering/fallback 的算子来源)
- [[01_ai_frameworks/index]] — 本域总索引
