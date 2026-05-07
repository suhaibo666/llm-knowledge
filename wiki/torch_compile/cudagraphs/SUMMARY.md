# PyTorch CUDA Graphs 完整指南 - 总结

本文档总结了 PyTorch CUDA Graphs 的所有使用方式，以及与 torch_npu NPU Graphs 的对比。

---

## 📁 目录结构

```
cudagraphs/
├── README.md                          # 本文件，快速参考
├── SUMMARY.md                         # 总结文档
├── cudagraphs_usage_guide.py          # CUDA Graphs 使用指南（含 Mermaid 时序图）
├── run_cudagraphs_examples.py         # 可运行的完整示例脚本
├── PyTorch_CUDA_Graphs_Complete_Guide.md  # 完整技术文档
├── CUDA_Graphs_Timing_Diagrams.md        # 时序图汇总（Mermaid 格式）
└── npugraphs/                        # NPU Graphs 对比目录
    ├── README.md                       # NPU Graphs 概述
    ├── npugraphs_usage_guide.py        # NPU vs CUDA 对比示例
    └── comparison.md                   # 详细对比文档
```

---

## 📚 文档说明

### 1. [README.md](README.md)
**快速参考指南**

包含：
- ✅ 目录结构说明
- ✅ 快速开始指南
- ✅ 系统要求
- ✅ 使用场景推荐

---

### 2. [cudagraphs_usage_guide.py](cudagraphs_usage_guide.py)
**CUDA Graphs 使用指南（含 Mermaid 时序图）**

包含：
- ✅ 所有5种使用方式的完整示例代码
- ✅ 详细的实现原理说明
- ✅ Mermaid 格式的代码调用流程时序图
- ✅ 优化机制和限制说明
- ✅ 综合比较表格

**运行方式：**
```bash
python cudagraphs/cudagraphs_usage_guide.py
```

---

### 3. [run_cudagraphs_examples.py](run_cudagraphs_examples.py)
**可运行的完整示例脚本**

包含：
- ✅ 所有5种 CUDA Graphs 使用方式的完整示例代码
- ✅ 性能基准测试
- ✅ 错误处理和兼容性检查
- ✅ 详细的输出信息

**运行方式：**
```bash
python cudagraphs/run_cudagraphs_examples.py
```

---

### 4. [PyTorch_CUDA_Graphs_Complete_Guide.md](PyTorch_CUDA_Graphs_Complete_Guide.md)
**完整的技术文档**

包含：
- ✅ 每种使用方式的详细说明
- ✅ 深度实现原理解析
- ✅ 完整的代码调用流程时序图（Mermaid 格式）
- ✅ 高级用法和最佳实践
- ✅ 性能优化建议
- ✅ 错误处理和调试技巧

---

### 5. [CUDA_Graphs_Timing_Diagrams.md](CUDA_Graphs_Timing_Diagrams.md)
**时序图汇总（Mermaid 格式）**

包含：
- ✅ 所有5种使用方式的完整 Mermaid 时序图
- ✅ 详细的执行流程说明
- ✅ 性能优化层级分析
- ✅ 关键特性总结

**如何查看 Mermaid 时序图：**
- 在 VS Code 中安装 **Mermaid Preview** 扩展
- 打开 `.md` 文件
- 右键点击 → "Open Preview" 或按 `Ctrl+Shift+V`

---

### 6. [npugraphs/](npugraphs/)
**NPU Graphs 对比目录**

#### [npugraphs/README.md](npugraphs/README.md)
NPU Graphs 概述和基本使用

#### [npugraphs/npugraphs_usage_guide.py](npugraphs/npugraphs_usage_guide.py)
NPU vs CUDA Graphs 对比示例代码

#### [npugraphs/comparison.md](npugraphs/comparison.md)
详细的对比文档，包含：
- API 对比
- 实现原理对比
- 时序图对比
- 代码示例对比
- 性能优化对比

---

## 🎯 CUDA Graphs 使用方式概览

### 方式1: `torch.compile(backend="cudagraphs")`

**特点：**
- 最简单易用
- 自动捕获 CUDA Graph
- 适合固定形状的推理场景

**示例：**
```python
import torch
import torch.nn as nn

model = MyModel().cuda()
compiled_model = torch.compile(model, backend="cudagraphs")

output = compiled_model(input_tensor)
```

**实现原理：**
- cudagraphs backend 专用后端
- 自动捕获计算图并转换为 CUDA Graph
- 消除 CPU-GPU 交互开销
- 预先分配所有需要的内存

**时序图：**
```
首次调用: warmup → 捕获图 → 记录所有操作 → 创建 CUDA Graph
后续调用: 复制输入数据 → replay CUDA Graph → 获取输出
```

---

### 方式2: `torch.compile(backend="inductor", mode="reduce-overhead")`

**特点：**
- PyTorch 2.0 推荐
- 多级优化（kernel fusion、Triton、CUDA Graphs）
- 生产环境就绪

**示例：**
```python
import torch
import torch.nn as nn

model = MyModel().cuda()
compiled_model = torch.compile(
    model,
    backend="inductor",
    mode="reduce-overhead"
)

output = compiled_model(input_tensor)
```

**实现原理：**
- Inductor 是 PyTorch 2.0 的默认编译后端
- "reduce-overhead" 模式专门优化推理性能
- 自动使用 CUDA Graphs（当条件满足时）
- 生成优化的 Triton kernels

**优化层级：**
```
Level 1: Kernel Fusion (多个操作融合为一个 kernel)
Level 2: Triton Kernels (自动优化 tiling、向量化)
Level 3: CUDA Graphs (消除 kernel launch 开销)
Level 4: 内存优化 (预先分配内存池)
```

---

### 方式3: `torch.cuda.graph()` 上下文管理器

**特点：**
- 最低级别的 CUDA Graphs API
- 最大灵活性
- 需要手动管理内存

**示例：**
```python
import torch
import torch.nn as nn

model = MyModel().cuda()

# 创建静态内存池
static_input = torch.empty_like(input_tensor)
static_output = torch.empty_like(output_tensor)

# 捕获 CUDA Graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    temp = static_input
    temp = model(temp)
    static_output.copy_(temp)

# Replay 图
static_input.copy_(new_input)
graph.replay()
result = static_output.clone()
```

**实现原理：**
- 手动管理输入/输出的静态内存池
- 在 graph() 上下文中执行一次，记录所有操作
- 后续调用时，复制输入 → replay → 复制输出

---

### 方式4: `torch.cuda.make_graphed_callables`

**特点：**
- 高级 API
- 自动管理内存
- 支持多个函数

**示例：**
```python
import torch
import torch.nn as nn

model = MyModel().cuda()
sample_input = torch.randn(32, 1024).cuda()

# 创建 graphed callable
graphed_model = torch.cuda.make_graphed_callables(
    model,
    [sample_input]
)

# 使用
output = graphed_model(input_tensor)
```

**实现原理：**
- 自动创建和管理静态内存池
- 支持多个独立的 callable
- 自动处理输入/输出的复制

---

### 方式5: `experimental` 参数

**特点：**
- PyTorch 2.1+ 实验性功能
- 细粒度控制
- 智能子图捕获

**示例：**
```python
import torch
import torch.nn as nn

model = MyModel().cuda()
compiled_model = torch.compile(
    model,
    backend="inductor",
    mode="reduce-overhead",
    experimental={"enable_cuda_graph": True}
)

output = compiled_model(input_tensor)
```

**实现原理：**
- 与 inductor backend 集成
- 自动识别可捕获的子图
- 提供更细粒度的控制

---

## 🔀 NPU Graphs (npugraphs) 对比

torch_npu 提供了对应的 NPU Graphs 实现，用于华为昇腾 NPU 设备。

### 主要区别

| 特性 | CUDA Graphs | NPU Graphs |
|------|------------|-------------|
| 设备 | NVIDIA GPU | 华为昇腾 NPU |
| 库 | PyTorch | torch_npu |
| 捕获 API | `torch.cuda.graph()` | `torch_npu.npu.graph()` |
| 后端 | `backend="cudagraphs"` | `backend="npugraphs"` |

### API 对应关系

| CUDA Graphs | NPU Graphs | 说明 |
|------------|-------------|------|
| `torch.cuda.graph()` | `torch_npu.npu.graph()` | 上下文管理器 |
| `torch.cuda.CUDAGraph()` | `torch_npu.npu.NPUGraph()` | 图对象 |
| `make_graphed_callables()` | `make_graphed_callables()` | 高级 API |
| `cudaGraphCaptureBegin()` | `aclmdlRICaptureBegin()` | 开始捕获 |
| `cudaGraphCaptureEnd()` | `aclmdlRICaptureEnd()` | 结束捕获 |
| `cudaGraphLaunch()` | `aclmdlRIExecuteAsync()` | 执行图 |

### NPU Graphs 示例

```python
import torch
import torch_npu

model = MyModel().npu()

# 使用 NPU Graphs
graph = torch_npu.npu.NPUGraph()
stream = torch_npu.npu.Stream()

with torch_npu.npu.graph(graph, stream=stream):
    # 记录操作
    output = model(input)

# Replay
graph.replay()
```

详细对比请查看 [npugraphs/](npugraphs/) 目录。

---

## 🎯 使用场景推荐

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 快速原型 | `backend="cudagraphs"` | 最简单，自动处理大部分细节 |
| 生产环境/推理优化 | `backend="inductor" + mode="reduce-overhead"` | PyTorch 2.0 推荐，自动选择最优策略 |
| 高级用户/精细控制 | `torch.cuda.graph()` | 完全控制内存和执行 |
| 多函数优化 | `make_graphed_callables` | 同时优化多个函数，自动内存管理 |
| 实验性功能 | `experimental` 参数 | 探索最新优化，细粒度控制 |

---

## ⚡ 系统要求

### CUDA Graphs
- **PyTorch**: 1.10+ (某些功能需要 2.0+)
- **CUDA**: 10.0+
- **GPU**: 支持 CUDA Graphs 的 NVIDIA GPU
  - Volta (V100) 及更新架构
  - 推荐使用 Ampere (A100) 或更新架构

### NPU Graphs
- **torch_npu**: 支持图捕获的版本
- **ACL**: 支持图捕获的 ACL 版本
- **NPU**: 华为昇腾 910B/910C 及更新

---

## 🚀 快速开始

### 1. 运行完整示例

```bash
# CUDA Graphs 示例
python cudagraphs/run_cudagraphs_examples.py

# NPU vs CUDA 对比
python cudagraphs/npugraphs/npugraphs_usage_guide.py
```

### 2. 查看详细指南

打开以下文件查看详细文档：
- [PyTorch_CUDA_Graphs_Complete_Guide.md](PyTorch_CUDA_Graphs_Complete_Guide.md)
- [CUDA_Graphs_Timing_Diagrams.md](CUDA_Graphs_Timing_Diagrams.md)
- [npugraphs/comparison.md](npugraphs/comparison.md)

### 3. 查看 Mermaid 时序图

在支持 Mermaid 的编辑器中打开 `.md` 文件，时序图会自动渲染。

---

## 📊 性能对比

### 理论性能提升

**传统执行 (每次调用):**
```
CPU → Launch Kernel 1 → GPU → CPU → Launch Kernel 2 → GPU → CPU → Launch Kernel 3 → GPU
     (开销)              (等待)    (开销)              (等待)    (开销)              (等待)
```

**CUDA/NPU Graphs 执行 (后续调用):**
```
CPU → Replay CUDA/NPU Graph → GPU/NPU (执行所有操作)
     (一次开销)                  (连续执行)
```

### 优化效果

- ✓ 消除多次 CPU-GPU/NPU 交互
- ✓ 消除多次 kernel launch 开销
- ✓ 消除多次驱动程序调用
- ✓ 提高 GPU/NPU 利用率
- ✓ 减少总执行时间 30-70%

---

## ⚠️ 注意事项

### CUDA Graphs

1. **固定形状**: 大多数方式要求输入形状固定
2. **静态内存**: 某些方式需要预先分配所有内存
3. **无控制流**: 不支持动态控制流（if/else、循环等）
4. **首次开销**: 首次运行会有额外的编译/捕获开销
5. **推理优化**: 主要用于推理场景，训练场景使用较少

### NPU Graphs

1. **设备要求**: 需要华为昇腾 NPU 设备
2. **torch_npu 版本**: 需要支持 NPU Graphs 的 torch_npu 版本
3. **ACL 版本**: 需要支持图捕获的 ACL 版本
4. **静态形状**: 输入形状必须固定
5. **无控制流**: 不支持动态控制流

---

## 📚 参考资源

### CUDA Graphs

- [PyTorch CUDA Graphs 文档](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html)
- [NVIDIA CUDA Graphs 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [PyTorch 2.0 性能优化指南](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

### NPU Graphs

- [torch_npu 文档](../torch_npu/README.md)
- [华为 ACL 文档](https://www.hiascend.com/document)
- [torch_npu graphs.py](../torch_npu/torch_npu/npu/graphs.py)

---

## 📝 许可证

本指南仅供学习和参考使用。

---

## 🤝 贡献

欢迎提出问题和改进建议！

## Related Pages

- [[torch_compile/index]]
- [[PyTorch_CUDA_Graphs_Complete_Guide]]
- [[README]]
