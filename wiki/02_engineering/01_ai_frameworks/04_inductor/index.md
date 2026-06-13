# 04 · TorchInductor — 目录索引

> PyTorch 编译后端核心:FX passes → lowering(FX→Inductor IR)→ 调度/融合 → codegen。含动态形状全链路。**NPU Inductor 后端**见 [[04_inductor/npu/index]]。
> 知识分层见各页「层次」标注(overview→quick start→deep dive,约定见 [[01_ai_frameworks/index]])。
> 最后更新: 2026-06-13

---

## 架构与流程

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[torch_compile_architecture]] | overview | torch.compile 端到端流水线(Dynamo→AOT→Inductor),高层全景 |
| [[inductor_compiler_pipeline_analysis]] | deep dive | 端到端编译管线全景,逐阶段源码级分析 |
| [[PyTorch_Inductor_Technical_Analysis]] | deep dive | Inductor 总体架构、IR 设计、后端代码生成(综合参考) |
| [[torch_compile_source_analysis]] | deep dive | 源码结构与模块组织、torch.compile 入口 |

## 上手

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[inductor_quickstart]] | **quick start** | 最小 fwd+bwd 示例、`torch.compile` 参数速查(mode/dynamic/fullgraph/options)、关键 `torch._inductor.config` 与环境变量、mode 选型、看生成代码与调试入口 |

## 编译阶段

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[lowering_analysis]] | deep dive | FX → Inductor IR lowering(注册/API/优化) |
| [[inductor_codegen_analysis]] | deep dive | 代码生成策略、kernel 融合 |
| [[scheduler_analysis]] | deep dive | 算子调度器、融合决策;**含自定义融合 Pass 与排查指南** |

## FX Passes

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[pre_grad_passes_guide]] | deep dive | 预梯度优化 passes(`fx_passes/pre_grad.py`) |
| [[joint_graph_passes_guide]] | deep dive | 联合图优化 passes(`fx_passes/joint_graph.py`) |
| [[post_grad_passes_guide]] | deep dive | 后梯度优化 passes(`fx_passes/post_grad.py`) |

## 动态形状

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[dynamic_shapes_full_analysis]] | deep dive | Dynamic Shape 全链路:静态特化→符号化→Guard→渐进动态化,ShapeEnv |
| [[inductor_codegen_dynamic_shape_analysis]] | deep dive | 代码生成中的动态形状,XBLOCK 选择与性能代价 |
| [[unbacked_symint_analysis]] | deep dive | Unbacked SymInt:数据相关 shape、deferred_runtime_asserts、torch._check() |

## 特性与调试

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[Pytorch_Compile_Debug_Analysis]] | deep dive(调试) | torch.compile 调试方法、`TORCH_LOGS`/`TORCH_COMPILE_DEBUG`、日志解读 |
| [[flex_attention_analysis]] | deep dive | FlexAttention:可组合注意力融合、BlockMask、score_mod、语义驱动 codegen |

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
