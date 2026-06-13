# 04 · TorchInductor — 目录索引

> PyTorch 编译后端核心:FX passes → lowering(FX→Inductor IR)→ 调度/融合 → codegen。含动态形状全链路。**NPU Inductor 后端**见 [[04_inductor/npu/index]]。
> 最后更新: 2026-06-13

---

## 架构与流程

| 页面 | 核心主题 |
|------|---------|
| [[torch_compile_architecture]] | torch.compile 端到端流水线(Dynamo→AOT→Inductor) |
| [[inductor_compiler_pipeline_analysis]] | 端到端编译管线全景,逐阶段源码级分析 |
| [[PyTorch_Inductor_Technical_Analysis]] | Inductor 总体架构、IR 设计、后端代码生成 |
| [[torch_compile_source_analysis]] | 源码结构与模块组织 |

## 编译阶段

| 页面 | 核心主题 |
|------|---------|
| [[lowering_analysis]] | FX → Inductor IR lowering |
| [[inductor_codegen_analysis]] | 代码生成策略、kernel 融合 |
| [[scheduler_analysis]] | 算子调度器、融合决策 |
| [[scheduler_fusion_strategies]] | 调度器融合策略与自定义 Pass |

## FX Passes

| 页面 | 核心主题 |
|------|---------|
| [[pre_grad_passes_guide]] | 预梯度优化 passes(`fx_passes/pre_grad.py`) |
| [[joint_graph_passes_guide]] | 联合图优化 passes(`fx_passes/joint_graph.py`) |
| [[post_grad_passes_guide]] | 后梯度优化 passes(`fx_passes/post_grad.py`) |

## 动态形状

| 页面 | 核心主题 |
|------|---------|
| [[dynamic_shapes_full_analysis]] | Dynamic Shape 全链路:静态特化→符号化→Guard→渐进动态化,ShapeEnv 源码分析 |
| [[inductor_codegen_dynamic_shape_analysis]] | 代码生成中的动态形状,含 XBLOCK 选择模式与性能代价 |
| [[unbacked_symint_analysis]] | Unbacked SymInt:数据相关 shape、deferred_runtime_asserts、GuardOnDataDependentSymNode、torch._check() |

## 特性与调试

| 页面 | 核心主题 |
|------|---------|
| [[flex_attention_analysis]] | FlexAttention:可组合注意力融合、BlockMask、score_mod、语义驱动 codegen |
| [[Pytorch_Compile_Debug_Analysis]] | torch.compile 调试方法与日志解读 |

---

## 硬件子目录

| 目录 | 核心主题 |
|------|---------|
| [[04_inductor/npu/index]] | NPU Inductor 后端:三条 compile 路径、lowering/fallback、35+ monkey patch、优化思想 |

---

## 关联域

- [[02_dynamo/index]] — 上游:图捕获
- [[03_aot_autograd/index]] — 上游:前/反向分解
- [[05_codegen_backends/index]] — codegen 后端(MLIR/Triton)
- [[06_graphs/index]] — 运行时图捕获
- [[01_ai_frameworks/index]] — 本域总索引
