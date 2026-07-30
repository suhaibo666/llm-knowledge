# 06 · CUDA Graphs — 目录索引

> CUDA Graphs 使用指南、捕获/回放时序、API 与最佳实践。
> 知识分层:overview→deep dive(约定见 [[01_ai_frameworks/index]])。
> 核验基准:PyTorch 上游 `torch/cuda/graphs.py`、`torch/_inductor/cudagraph_trees.py`
> 最后更新: 2026-07-30(kb-reorg P4 Task 6:D06/F08 随 D/F 卷分发迁入,判重 vs 本页 Guide 方式2/综合比较节后保留为专题页)

---

## 页面列表(按层次)

> **段位与阅读顺序**(kb-reorg P4 Task 9.5,2026-07-30):段 0(01-09)入门/overview——Complete Guide 本身即涵盖四种用法+综合比较,充当本目录唯一的入门兼参考页;段 1(10-19)核心机制——CUDAGraph Trees 源码级深挖,是本目录仅有的一篇核心机制专题;段 2(20-29)深潜/专题——training/inference/freezing 组合边界(F08,按 F 卷惯例落在专题段,与其余目录 F0x 页处置一致)。3 篇分居三段,无同段竞争。

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[01_PyTorch_CUDA_Graphs_Complete_Guide]] | **overview → deep dive**(段 0) | CUDA Graphs 完整指南:四种用法(`backend="cudagraphs"`/`inductor + reduce-overhead`/`torch.cuda.graph()`/`make_graphed_callables`)、实现原理、各方式 Mermaid 时序图(原 Timing_Diagrams 页已并入对应小节)、综合比较(功能/硬件要求/使用场景/性能)、最佳实践(已并入原 README 的速览与差异内容) |
| [[10_cudagraph_trees_warmup_record_and_replay_analysis]] | **deep dive(源码级,段 1)** | `torch/_inductor/cudagraph_trees.py` 机制深挖:`CUDAGraphNode`/`CUDAGraphTreeManager`、按静态输入地址与动态整数 key 分流的多份 recording、warmup→record→replay→fallback 状态机、invariant 检查;与本页方式2/综合比较判重后保留为专题页(独有内容 >50%) |
| [[20_training_inference_cudagraph_and_freezing_analysis]] | **deep dive(源码级,专题,段 2)** | training/inference/freezing/CUDA Graph 四轴组合边界:`freezing.py` 变换链与所有权后果、`cudagraphify` 地址不变式、组合矩阵与失败/回退边界;本页 Guide 未覆盖,保留为专题页 |

> 代码示例:`cudagraphs_usage_guide.py`、`run_cudagraphs_examples.py`(同目录)。

---

## 关联域

- [[03_runtime_graphs/npu/index]] — NPU Graphs(对比)
- [[03_runtime_graphs/index]] — 运行时图捕获总索引
- [[01_ai_frameworks/index]] — 本域总索引
