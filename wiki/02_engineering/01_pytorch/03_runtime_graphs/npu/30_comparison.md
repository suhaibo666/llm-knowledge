---
title: "CUDA Graphs vs NPU Graphs 差异对比"
---

# CUDA Graphs vs NPU Graphs 差异对比

本文档只保留 CUDA Graphs 与 torch_npu NPU Graphs 之间**可验证的真实差异**（API 映射表、组件对照、捕获/执行行为差异结论）。两侧各自的机制原理、实现细节与可运行代码示例不在本页复述，请查阅权威页：CUDA 侧 [[10_pytorch_cuda_graphs_complete_guide]]；NPU 侧 [[01_aclgraph]] / [[10_aclgraph_deep_analysis]] / [[11_torch_compile_npugraphs_deepdive]] / [[20_npugraphs_make_graphed_callables_deepdive]]。

---

## 目录

1. [概述](#概述)
2. [API 对比](#api-对比)
3. [实现原理对比](#实现原理对比)
4. [捕获/执行时序差异](#捕获执行时序差异)
5. [代码示例对比](#代码示例对比)
6. [使用建议](#使用建议)

---

## 概述

CUDA Graphs 是 NVIDIA 提供的图捕获/重放优化技术，机制与完整 API 详见 [[10_pytorch_cuda_graphs_complete_guide]]。NPU Graphs 是 torch_npu 基于 ACL Graph API 的对应实现，机制与完整 API 详见 [[01_aclgraph]]。

### 核心概念对比

| 概念 | CUDA Graphs | NPU Graphs |
|------|------------|-------------|
| 设备 | NVIDIA GPU | 华为昇腾 NPU |
| 库 | PyTorch (torch.cuda) | torch_npu (torch_npu.npu) |
| 底层 API | CUDA Runtime API | ACL (Ascend Computing Language) |
| 捕获 | cudaGraphCaptureBegin/End | aclmdlRICaptureBegin/End |
| 执行 | cudaGraphLaunch | aclmdlRIExecuteAsync |

---

## API 对比

三种集成方式——上下文管理器（`torch.cuda.graph()` / `torch_npu.npu.graph()`）、`make_graphed_callables()`、`torch.compile(backend=...)`——在两侧的调用签名、参数与调用顺序一一对应，仅命名空间和后端字符串不同；可运行示例见下文「代码示例对比」，机制原理见各权威页。

### API 完整对比表

| 功能 | CUDA Graphs | NPU Graphs | 说明 |
|------|------------|-------------|------|
| 图对象 | `torch.cuda.CUDAGraph()` | `torch_npu.npu.NPUGraph()` | 管理图的生命周期 |
| 上下文管理器 | `torch.cuda.graph()` | `torch_npu.npu.graph()` | 捕获和重放 |
| Stream | `torch.cuda.Stream()` | `torch_npu.npu.Stream()` | 异步执行流 |
| 高级 API | `make_graphed_callables()` | `make_graphed_callables()` | 自动管理 |
| 编译后端 | `backend="cudagraphs"` | `backend="npugraphs"` | torch.compile |
| 捕获开始 | `cudaGraphCaptureBegin()` | `aclmdlRICaptureBegin()` | 底层 API |
| 捕获结束 | `cudaGraphCaptureEnd()` | `aclmdlRICaptureEnd()` | 底层 API |
| 执行图 | `cudaGraphLaunch()` | `aclmdlRIExecuteAsync()` | 底层 API |

---

## 实现原理对比

CUDA 侧的调用层次（`torch.cuda.graph()` → Python 上下文管理器 → `CUDAGraph`(C++) → CUDA Runtime API）见 [[10_pytorch_cuda_graphs_complete_guide]]「方式3: torch.cuda.graph() 上下文管理器」§实现原理。NPU 侧的调用层次（`torch_npu.npu.graph()` → Python 上下文管理器 → `NPUGraph`(C++) → ACL Graph API）见 [[01_aclgraph]] §3.2 调用流程图。

### 核心组件对比

| 组件 | CUDA Graphs | NPU Graphs | CUDA 文件位置 | NPU 文件位置 |
|------|------------|-------------|----------|----------|
| Python 包装 | `torch.cuda.graph()` | `torch_npu.npu.graph()` | `torch/cuda/graph.py` | `torch_npu/npu/graphs.py` |
| C++ 实现 | `CUDAGraph` | `NPUGraph` | `torch/csrc/cuda/CUDAGraph.cpp` | `torch_npu/csrc/core/npu/NPUGraph.cpp` |
| 头文件 | `CUDAGraph.h` | `NPUGraph.h` | `torch/csrc/cuda/CUDAGraph.h` | `torch_npu/csrc/core/npu/NPUGraph.h` |
| 内存管理 | `CUDACachingAllocator` | `NPUCachingAllocator` | `torch/csrc/cuda/CUDACachingAllocator.h` | `torch_npu/csrc/core/npu/NPUCachingAllocator.h` |

---

## 捕获/执行时序差异

两侧的捕获→重放模型结构一致：捕获期只记录不执行，重放期一次性提交整个图。完整时序图：CUDA 侧见 [[10_pytorch_cuda_graphs_complete_guide]]「方式3」的「代码调用流程时序图」（初始化→捕获→执行全流程）；NPU 侧见 [[01_aclgraph]] §3.3「详细调用时序图」。

两侧在时序细节上的真实行为差异（源码核验，详见 [[10_aclgraph_deep_analysis]]）：

| 环节 | CUDA Graphs | NPU Graphs |
|---|---|---|
| API 级数 | 四级：`cudaStreamBeginCapture` → `cudaGraphGetRootNode` → `cudaGraphInstantiate` → `cudaGraphLaunch` | 三级：`AclmdlRICaptureBegin` → `AclmdlRICaptureEnd` → `AclmdlRIExecuteAsync`（[[10_aclgraph_deep_analysis]] 差异 2） |
| 图实例化时机 | 捕获结束后显式调用 `cudaGraphInstantiate()` 分配 graphExec 资源 | `model_ri` 在 `capture_begin()` 时即创建（`NPUGraph.cpp:235-252`），无独立 instantiate 步骤 |
| 捕获期算子门禁 | 无算子类型级门禁 | CUDA 完全没有的门禁：仅 aclnn 算子可入图，aclop 在捕获期被 `assertNotCapturingAclop` 直接拦截（`OpCommand.cpp:139`，根因是 aclop 走主机侧 JIT 编译）；详见 [[10_aclgraph_deep_analysis]] 差异 8 |

---

## 代码示例对比

三种集成方式的完整可运行示例（模型定义、捕获、重放全流程）不在本页重复：

| 方式 | CUDA 示例 | NPU 示例 |
|---|---|---|
| 上下文管理器 | [[10_pytorch_cuda_graphs_complete_guide]] 方式3 | [[01_aclgraph]] §3（含 §3.3 时序图） |
| `make_graphed_callables` | [[10_pytorch_cuda_graphs_complete_guide]] 方式4 | [[20_npugraphs_make_graphed_callables_deepdive]]（六阶段实现流程） |
| `torch.compile(backend=...)` | [[10_pytorch_cuda_graphs_complete_guide]] 方式1/方式2 | [[11_torch_compile_npugraphs_deepdive]] |

---

## 使用建议

### 选择标准

| 场景 | 推荐使用 | 原因 |
|------|---------|------|
| NVIDIA GPU | CUDA Graphs | 原生支持 |
| 华为昇腾 NPU | NPU Graphs | 原生支持 |
| 需要跨平台 | 按 `device.type` 分支调用对应 API | 两侧 API 签名一一对应，可用薄封装切换 |
| 性能调优 | 对应平台原生 API | 避开抽象层开销，便于对照权威页排查 |

> [!todo] 待核验：原页曾声称 NPU Graphs 需 Ascend 910/910B、ACL 20.0+。全库未找到源码或官方文档佐证（aclgraph 系权威页只有带引文的 CANN 8.5.0+ 特性门槛），暂存此处待核实后转正或删除。

---

## 总结

### 相似性

1. **API 设计**：两者 API 设计高度相似，除命名空间和后端字符串外几乎一一对应
2. **捕获/重放模型**：都是"捕获期只记录、重放期一次性提交"
3. **优化机制**：都通过消除 kernel launch 开销、减少 CPU-设备交互、静态内存分配来降低总执行时间；量化数字（30-70%）未附实测来源，见 [[10_pytorch_cuda_graphs_complete_guide]] 概述

### 差异性

1. **硬件平台**：NVIDIA GPU vs 华为昇腾 NPU
2. **底层 API 级数与实例化时机**：CUDA 四级 API + 显式 instantiate；CANN 三级 API，`model_ri` 随 `capture_begin` 创建（见上文「捕获/执行时序差异」）
3. **捕获期算子门禁**：NPU 独有的 aclop/aclnn 门禁，CUDA 无对应机制
4. **Python 库**：torch.cuda vs torch_npu.npu

---

## 参考资料

### CUDA Graphs

- [PyTorch CUDA Graphs 文档](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html)
- [NVIDIA CUDA Graphs 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [PyTorch 2.0 性能优化指南](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- 完整时序图：已并入 [[10_pytorch_cuda_graphs_complete_guide]] 各"方式"小节的"代码调用流程时序图"

### NPU Graphs

- [华为 ACL 文档](https://www.hiascend.com/document)
- 完整时序图：已并入 [[01_aclgraph]] §3.3"详细调用时序图"

## Related Pages

- [[02_engineering/01_pytorch/index]]
- [[10_pytorch_cuda_graphs_complete_guide]]
- [[01_aclgraph]]
- [[10_aclgraph_deep_analysis]]
