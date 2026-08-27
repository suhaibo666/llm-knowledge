---
title: "06 · CUDA Graphs — 目录索引"
---

# 06 · CUDA Graphs — 目录索引

> CUDA Graphs 使用指南、捕获/回放时序、API 与最佳实践。
> 知识分层:overview→deep dive(约定见 [[01_pytorch/index]])。
> 核验基准:PyTorch 上游 `torch/cuda/graphs.py`、`torch/_inductor/cudagraph_trees.py`
> 最后更新: 2026-07-31(kb-reorg P7 Task 7:命名统一,`PyTorch_CUDA_Graphs_Complete_Guide` 改小写并从段 0 移入段 1——其内容是四种用法+实现原理的核心机制主线而非纯入门页;`cudagraph_trees` 深挖顺延为 11 号)

---

## 页面列表(按层次)

> **段位与阅读顺序**(kb-reorg P4 Task 9.5 定稿,2026-07-31 kb-reorg P7 Task 7 重定段位):本目录无独立 overview 页,段 1(10-19)核心机制主线——Complete Guide 涵盖四种用法+综合比较,是唯一的机制主线兼参考页;CUDAGraph Trees 源码级深挖顺接其后;段 2(20-29)深潜/专题——training/inference/freezing 组合边界(F08,按 F 卷惯例落在专题段,与其余目录 F0x 页处置一致)。3 篇分居两段,无同段竞争。

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[10_pytorch_cuda_graphs_complete_guide]] | **机制主线(段 1)** | CUDA Graphs 完整指南:四种用法(`backend="cudagraphs"`/`inductor + reduce-overhead`/`torch.cuda.graph()`/`make_graphed_callables`)、实现原理、各方式 Mermaid 时序图(原 Timing_Diagrams 页已并入对应小节)、综合比较(功能/硬件要求/使用场景/性能)、最佳实践(已并入原 README 的速览与差异内容) |
| [[11_cudagraph_trees_warmup_record_and_replay_analysis]] | **deep dive(源码级,段 1)** | `torch/_inductor/cudagraph_trees.py` 机制深挖:`CUDAGraphNode`/`CUDAGraphTreeManager`、按静态输入地址与动态整数 key 分流的多份 recording、warmup→record→replay→fallback 状态机、invariant 检查;与本页方式2/综合比较判重后保留为专题页(独有内容 >50%) |
| [[20_training_inference_cudagraph_and_freezing_analysis]] | **deep dive(源码级,专题,段 2)** | training/inference/freezing/CUDA Graph 四轴组合边界:`freezing.py` 变换链与所有权后果、`cudagraphify` 地址不变式、组合矩阵与失败/回退边界;本页 Guide 未覆盖,保留为专题页 |

> 代码示例:`cudagraphs_usage_guide.py`、`run_cudagraphs_examples.py`(同目录)。

---

## 关联域

- [[03_runtime_graphs/npu/index]] — NPU Graphs(对比)
- [[03_runtime_graphs/index]] — 运行时图捕获总索引
- [[01_pytorch/index]] — 本域总索引
