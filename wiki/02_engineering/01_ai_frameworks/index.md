# PyTorch Compilation Stack — 目录索引

> 覆盖 PyTorch 编译流水线 (`torch.compile`): Dynamo 图捕获, AOT Autograd, Inductor 代码生成, CUDA/NPU Graphs
> 最后更新: 2026-05-07

---

## 架构总览

```
User Code → @torch.compile
              ↓
         TorchDynamo (frame evaluation hook 图捕获)
              ↓
         AOT Autograd (前向/反向分解)
              ↓
         TorchInductor (IR lowering → codegen)
              ↓
         优化 Kernel 执行
```

详见 [[torch_compile_architecture]] 完整流水线分析。

---

## 子目录

| 目录 | 核心主题 |
|------|---------|
| [[inductor/index]] | Inductor IR, lowering, codegen, scheduling, NPU 后端 (17 篇) |
| [[cudagraphs/index]] | CUDA Graphs 使用指南, NPU Graphs 对比 (10 篇) |
| [[cudagraphs/npugraphs/index]] | NPU Graphs 深度分析 (8 篇) |

---

## 页面列表

### 编译架构

| 页面 | 核心主题 |
|------|---------|
| [[torch_compile_source_analysis]] | 源码结构, 模块组织 |
| [[torch_compile_architecture]] | 端到端流水线: Dynamo → AOT Autograd → Inductor |
| [[triton_vs_mlir_backend_analysis]] | Triton vs Torch-MLIR: 六阶段概念对等映射, 优劣势分析 |
| [[mlir_core_concepts]] | MLIR 基础: Dialect、Pass、IR 注册、递降原理 |
| [[torch_mlir_pass_pipeline_analysis]] | torch-mlir Pass 管线: 按执行顺序的 34 个 Pass 完整分析 |
| [[PyTorch_Dynamo_Technical_Analysis]] | 帧评估 API, 字节码符号执行, guard 生成 |
| [[PyTorch_Inductor_Technical_Analysis]] | Inductor IR, 调度, 代码生成后端 |
| [[Pytorch_Compile_Debug_Analysis]] | 调试技巧, 日志解读 |

### 编译优化

| 页面 | 核心主题 |
|------|---------|
| [[aotautograd_analysis]] | AOT Autograd 图分解, 联合图 passes |
| [[lowering_analysis]] | FX → Inductor IR lowering |
| [[inductor_codegen_analysis]] | 代码生成策略, kernel 融合 |
| [[inductor_codegen_dynamic_shape_analysis]] | 代码生成中的动态形状 |
| [[scheduler_analysis]] | 算子调度, 融合决策 |
| [[pre_grad_passes_guide]] | 预梯度优化 passes |
| [[post_grad_passes_guide]] | 后梯度优化 passes |
| [[joint_graph_passes_guide]] | 联合图优化 passes |

### NPU 后端

| 页面 | 核心主题 |
|------|---------|
| [[npu_lowering_guide]] | NPU 特定 lowering |
| [[npu_compile]] | NPU 编译工作流 |
| [[NPU_Inductor_Backend_Analysis]] | NPU 后端集成架构 |
| [[NPU_Inductor_Backend_Mechanism]] | NPU 后端内部机制 |
| [[NPU_MLIR_Backend_Technical_Analysis]] | MLIR 基 NPU 后端 |
| [[npu_mlir_pipeline_analysis]] | NPU MLIR 六阶段适配全景: GPU vs NPU 逐阶段对比 |
| [[mlir_core_concepts]] | MLIR 基础: Dialect、Pass、IR 注册 |

### CUDA/NPU Graphs

| 页面 | 核心主题 |
|------|---------|
| [[SUMMARY]] | CUDA/NPU Graphs 文档索引 |
| [[PyTorch_CUDA_Graphs_Complete_Guide]] | CUDA Graphs 完整指南 |
| [[CUDA_Graphs_Timing_Diagrams]] | Graph 时序图 |
| [[torch_compile_npugraphs_deep_dive]] | NPU Graphs + torch.compile |
| [[npugraphs_make_graphed_callables_deep_dive]] | make_graphed_callables API |
| [[npugraphs_memory_management_analysis]] | 内存管理 |
| [[npugraphs_memory_reuse_analysis]] | 内存重用 |
| [[torch_compile_mode_reduce_overhead_vs_backend_npugraphs]] | reduce_overhead vs npugraphs |
| [[aclgraph]] | ACL Graph 集成 |
| [[comparison]] | CUDA vs NPU Graphs 对比 |

---

## 知识空白

- **TorchDynamo guard 失败调试** — 常见但未记录
- **Inductor autotuning** — Triton kernel autotuning 策略
- **动态形状支持完整性** — 已知限制未编目
- **Multi-backend dispatch** — Inductor 在 CUDA/NPU 间选择逻辑

---

## 关联域

- [[llm/06_infra/megatron-lm/index]] — Megatron-LM (CUDA Graphs 使用场景)
- [[llm/02_training/low_precision_training_analysis]] — 低精度训练 + CUDA Graphs
- [[llm/06_infra/llm_parallelism_analysis]] — 计算通信重叠
