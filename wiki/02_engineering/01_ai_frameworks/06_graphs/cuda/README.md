# PyTorch CUDA Graphs 完整使用指南

本目录包含 PyTorch CUDA Graphs 的完整使用指南：原理、四种用法、性能与注意事项。

## 📁 目录结构

```
cuda/
├── PyTorch_CUDA_Graphs_Complete_Guide.md   # 完整技术文档
├── CUDA_Graphs_Timing_Diagrams.md          # 时序图汇总（Mermaid 格式）
├── cudagraphs_usage_guide.py               # 使用指南（含 Mermaid 时序图）
├── run_cudagraphs_examples.py              # 可运行的完整示例脚本
└── README.md                               # 本文件
```

---

## 📚 文档说明

- [cudagraphs_usage_guide.py](cudagraphs_usage_guide.py)：四种用法的示例代码与实现原理（含 Mermaid 时序图），可直接 `python` 运行。
- [run_cudagraphs_examples.py](run_cudagraphs_examples.py)：可运行的完整示例，含性能基准与兼容性检查。
- [PyTorch_CUDA_Graphs_Complete_Guide.md](PyTorch_CUDA_Graphs_Complete_Guide.md)：完整技术文档，覆盖每种用法、实现原理、高级用法与调试技巧。
- [CUDA_Graphs_Timing_Diagrams.md](CUDA_Graphs_Timing_Diagrams.md)：四种用法的 Mermaid 时序图与执行流程说明。

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
首次调用: warmup → 捕获图 → 记录操作 → 创建 CUDA Graph
后续调用: 复制输入 → replay 图 → 获取输出
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

## 🎯 使用场景推荐

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 快速原型 | `backend="cudagraphs"` | 最简单，自动处理 |
| 生产推理 | `backend="inductor" + mode="reduce-overhead"` | PyTorch 2.0 推荐 |
| 精细控制 | `torch.cuda.graph()` | 最大灵活性 |
| 多函数优化 | `make_graphed_callables` | 自动内存管理 |

---

## ⚡ 系统要求

### CUDA Graphs
- **PyTorch**: 1.10+ (某些功能需要 2.0+)
- **CUDA**: 10.0+
- **GPU**: 支持 CUDA Graphs 的 NVIDIA GPU
  - Volta (V100) 及更新架构
  - 推荐使用 Ampere (A100) 或更新架构

---

## 🚀 快速开始

### 1. 运行完整示例

```bash
# CUDA Graphs 示例
python run_cudagraphs_examples.py
```

### 2. 查看详细指南

打开以下文件查看详细文档：
- [PyTorch_CUDA_Graphs_Complete_Guide.md](PyTorch_CUDA_Graphs_Complete_Guide.md)
- [CUDA_Graphs_Timing_Diagrams.md](CUDA_Graphs_Timing_Diagrams.md)

### 3. 查看 Mermaid 时序图

在支持 Mermaid 的编辑器中打开 `.md` 文件，时序图会自动渲染。

**如何查看 Mermaid 时序图（合并自 SUMMARY）：**
- 在 VS Code 中安装 **Mermaid Preview** 扩展
- 打开 `.md` 文件
- 右键点击 → "Open Preview" 或按 `Ctrl+Shift+V`

---

## 📊 性能对比

### 理论性能提升

**传统执行:**
```
CPU → Launch Kernel 1 → GPU → CPU → Launch Kernel 2 → GPU → CPU → Launch Kernel 3 → GPU
     (开销)              (等待)    (开销)              (等待)    (开销)              (等待)
```

**CUDA Graphs 执行:**
```
CPU → Replay Graph → GPU (执行所有操作)
     (一次开销)        (连续执行)
```

**优化效果:**
- ✓ 消除多次 CPU-GPU 交互
- ✓ 消除多次 kernel launch 开销
- ✓ 消除多次驱动程序调用
- ✓ 提高 GPU 利用率
- ✓ 减少总执行时间 30-70%

### 实际性能因素

实际性能取决于：
- 模型复杂度
- 输入形状
- GPU 型号
- CUDA 版本
- 批量大小

---

## ⚠️ 注意事项

### CUDA Graphs

1. **固定形状**: 大多数方式要求输入形状固定
2. **静态内存**: 某些方式需要预先分配所有内存
3. **无控制流**: 不支持动态控制流（if/else、循环等）
4. **首次开销**: 首次运行会有额外的编译/捕获开销
5. **推理优化**: 主要用于推理场景，训练场景使用较少

---

## 📚 参考资源

### CUDA Graphs

- [PyTorch CUDA Graphs 文档](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html)
- [NVIDIA CUDA Graphs 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [PyTorch 2.0 性能优化指南](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- CUDA vs NPU Graphs 对比见 [[06_graphs/npu/comparison]]

---

## 📝 许可证

本指南仅供学习和参考使用。

---

## 🤝 贡献

欢迎提出问题和改进建议！

## Related Pages

- [[02_engineering/01_ai_frameworks/index]]
- [[PyTorch_CUDA_Graphs_Complete_Guide]]
- [[06_graphs/cuda/README]]
