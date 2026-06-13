# 06 · NPU Graphs / ACLGraph — 目录索引

> NPU Graphs 的编译集成、内存管理与重用、make_graphed_callables 适配、ACLGraph 深度分析,以及与 CUDA Graphs 的对比。
> 核验基准:torch_npu **v2.7.1.post5**
> 最后更新: 2026-06-13

---

## 集成与使用

| 页面 | 核心主题 |
|------|---------|
| [[torch_compile_npugraphs_deep_dive]] | NPU Graphs 与 torch.compile 集成深度分析 |
| [[npugraphs_make_graphed_callables_deep_dive]] | make_graphed_callables API 深入分析 |
| [[torch_compile_mode_reduce_overhead_vs_backend_npugraphs]] | reduce_overhead 模式 vs backend npugraphs 对比 |

## 内存

| 页面 | 核心主题 |
|------|---------|
| [[npugraphs_memory_management_analysis]] | NPU Graphs 内存管理机制 |
| [[npugraphs_memory_reuse_analysis]] | 内存重用策略分析 |

## ACLGraph 与对比

| 页面 | 核心主题 |
|------|---------|
| [[aclgraph]] | ACL Graph(昇腾计算语言)集成 |
| [[aclgraph_deep_analysis]] | **ACLGraph 深度分析**:图捕获/重放、Super Kernel、NpuGraphOpHandler、StaticKernelCompiler、与社区 CUDAGraph 差异及演进 |
| [[comparison]] | CUDA Graphs vs NPU Graphs 特性对比 |

> 代码示例:`npugraphs_usage_guide.py`;原始说明见 [[06_graphs/npu/README]]。

---

## 关联域

- [[06_graphs/cuda/index]] — CUDA Graphs(对比基准)
- [[04_inductor/npu/index]] — NPU Inductor(ACLGraph 为其路径之一)
- [[07_op_registration/npu/index]] — 入图判别第三关(aclgraph aclnn-only)
- [[01_ai_frameworks/index]] — 本域总索引
