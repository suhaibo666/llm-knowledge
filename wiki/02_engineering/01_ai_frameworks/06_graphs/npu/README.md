# NPU Graphs (npugraphs) 使用指南

NPU Graphs 是 torch_npu 对应 CUDA Graphs 的实现，用于华为昇腾 NPU 设备的性能优化。

## 概述

NPU Graphs 与 CUDA Graphs 的概念类似，都是通过预先记录计算图来减少 CPU-NPU 交互开销。

### NPU Graphs vs CUDA Graphs

| 特性 | CUDA Graphs | NPU Graphs |
|------|------------|-------------|
| 设备 | NVIDIA GPU | 华为昇腾 NPU |
| 库 | PyTorch | torch_npu |
| 捕获 API | `torch.cuda.graph()` | `torch_npu.npu.graph()` |
| 后端 | `backend="cudagraphs"` | `backend="npugraphs"` |

## 使用方式

torch_npu 提供了与 PyTorch CUDA Graphs 类似的 API：

### 1. `torch_npu.npu.graph()` 上下文管理器

```python
import torch
import torch_npu

# 创建静态内存池
static_input = torch.empty_like(input_tensor)
static_output = torch.empty_like(output_tensor)

# 创建 NPU Graph
graph = torch_npu.npu.NPUGraph()
stream = torch_npu.npu.Stream()

with torch_npu.npu.graph(graph, stream=stream):
    # 记录操作
    temp = static_input
    temp = model.linear1(temp)
    temp = model.relu(temp)
    temp = model.linear2(temp)
    static_output.copy_(temp)

# Replay
static_input.copy_(new_input)
graph.replay()
result = static_output.clone()
```

### 2. `torch_npu.npu.make_graphed_callables`

```python
import torch
import torch_npu

model = MyModel().npu()
sample_input = torch.randn(32, 1024).npu()

# 创建 graphed callable
graphed_model = torch_npu.npu.make_graphed_callables(model, [sample_input])

# 使用
output = graphed_model(input_tensor)
```

### 3. `torch.compile(backend="npugraphs")`

```python
import torch
import torch_npu

model = MyModel().npu()
compiled_model = torch.compile(model, backend="npugraphs")

output = compiled_model(input_tensor)
```

## 实现原理

### NPU Graphs 核心组件

1. **NPUGraph 类** (`torch_npu/csrc/core/npu/NPUGraph.h`)
   - 管理 NPU 图的捕获和重放
   - 提供静态内存池管理

2. **ACL Graph API** (`third_party/acl/`)
   - 使用华为 ACL (Ascend Computing Language) 图 API
   - `aclmdlRICaptureBegin()` - 开始捕获
   - `aclmdlRICaptureEnd()` - 结束捕获
   - `aclmdlRIExecuteAsync()` - 异步执行

3. **内存管理** (`torch_npu/csrc/core/npu/NPUCachingAllocator.h`)
   - 静态内存池
   - 缓存分配器

## 与 CUDA Graphs 的对应关系

| CUDA Graphs | NPU Graphs | 说明 |
|------------|-------------|------|
| `torch.cuda.graph()` | `torch_npu.npu.graph()` | 上下文管理器 |
| `torch.cuda.CUDAGraph()` | `torch_npu.npu.NPUGraph()` | 图对象 |
| `torch.cuda.make_graphed_callables()` | `torch_npu.npu.make_graphed_callables()` | 高级 API |
| `cudaGraphCaptureBegin()` | `aclmdlRICaptureBegin()` | 开始捕获 |
| `cudaGraphCaptureEnd()` | `aclmdlRICaptureEnd()` | 结束捕获 |
| `cudaGraphLaunch()` | `aclmdlRIExecuteAsync()` | 执行图 |

## 注意事项

1. **设备要求**: 需要华为昇腾 NPU 设备
2. **torch_npu 版本**: 需要支持 NPU Graphs 的 torch_npu 版本
3. **ACL 版本**: 需要支持图捕获的 ACL 版本
4. **静态形状**: 输入形状必须固定
5. **无控制流**: 不支持动态控制流

## 参考资料

- [torch_npu 文档](../torch_npu/README.md)
- [华为 ACL 文档](https://www.hiascend.com/document)
- [CUDA Graphs 指南](../cudagraphs_usage_guide.py)

## Related Pages

- [[02_engineering/01_ai_frameworks/index]]
- [[aclgraph]]
- [[comparison]]
