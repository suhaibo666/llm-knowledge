# TorchInductor — 目录索引

> 覆盖 PyTorch Inductor 后端全流程：IR lowering、代码生成、调度、NPU 适配
> 最后更新: 2026-05-11

---

## 页面列表

### 架构与流程

| 页面 | 核心主题 |
|------|---------|
| [[inductor_compiler_pipeline_analysis]] | **端到端编译管线全景**：Eager→Dynamo→AOT Autograd→Decomposition→FX Passes→Lowering→Scheduler→CodeGen，逐阶段源码级分析 |
| [[PyTorch_Inductor_Technical_Analysis]] | Inductor 总体架构, IR 设计, 后端代码生成 |
| [[torch_compile_architecture]] | torch.compile 端到端流水线 |
| [[torch_compile_source_analysis]] | 源码结构与模块组织 |

### 编译阶段

| 页面 | 核心主题 |
|------|---------|
| [[aotautograd_analysis]] | AOT Autograd 前向/反向图分解 |
| [[lowering_analysis]] | FX → Inductor IR lowering |
| [[inductor_codegen_analysis]] | 代码生成策略, kernel 融合 |
| [[dynamic_shapes_full_analysis]] | **Dynamic Shape 全链路**: 静态特化→符号化→Guard→渐进动态化, ShapeEnv 源码分析 |
| [[inductor_codegen_dynamic_shape_analysis]] | 代码生成中动态形状处理 |
| [[scheduler_analysis]] | 算子调度器, 融合决策 |
| [[scheduler_fusion_strategies]] | 调度器融合策略与自定义 Pass |
| [[pre_grad_passes_guide]] | 预梯度优化 passes |
| [[post_grad_passes_guide]] | 后梯度优化 passes |
| [[joint_graph_passes_guide]] | 联合图优化 passes |

### NPU 后端

| 页面 | 核心主题 |
|------|---------|
| [[npu_lowering_guide]] | NPU 特定 lowering 步骤与算子映射 |
| [[npu_compile]] | NPU 编译工作流、Autotune、精度校验 |
| [[npu_mlir_pipeline_analysis]] | NPU MLIR 六阶段适配全景: 三层 Pass、15 个 Monkey Patch |
| [[NPU_Inductor_Backend_Analysis]] | NPU Inductor 后端集成架构分析 |
| [[NPU_Inductor_Backend_Mechanism]] | NPU 后端内部实现机制 |
| [[NPU_MLIR_Backend_Technical_Analysis]] | 基于 MLIR 的 NPU 后端技术分析 |
| [[npu_mlir_backend_deep_analysis]] | **MLIR 路径深度分析**: IR 回溯、Bisheng 编译器、Scheduler patch、auto_fallback |
| [[npu_triton_backend_deep_analysis]] | **Triton/Inductor default 路径深度分析**: golden_var_list、CATLASS/CK GEMM、35+ monkey patches、NPUIndexTritonKernel |
| [[triton_vs_mlir_backend_analysis]] | Triton vs Torch-MLIR 后端: 六阶段概念对等映射 |
| [[mlir_core_concepts]] | MLIR 基础: Dialect、Pass、IR 注册、递降原理 |

### 调试与诊断

| 页面 | 核心主题 |
|------|---------|
| [[Pytorch_Compile_Debug_Analysis]] | torch.compile 调试方法与日志解读 |

---

## 关联域

- [[cudagraphs/index]] — CUDA/NPU Graphs
- [[index]] — torch.compile 总索引
- [[llm/06_infra/megatron-lm/index]] — Megatron-LM（CUDA Graphs 使用场景）
