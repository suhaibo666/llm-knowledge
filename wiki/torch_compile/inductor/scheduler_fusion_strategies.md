# Inductor Scheduler 融合策略与自定义 Pass 指南

本文档详细说明了 PyTorch Inductor 后端在 Scheduler 阶段的算子融合机制，包括默认策略、自定义 Pass 编写方法以及常见问题排查。

## 1. Scheduler 融合机制总览

在 `torch.compile` 的编译流程中，Scheduler 阶段负责将 Lowering 后的计算图转换为可执行的 Kernel 序列。其核心任务是**算子融合 (Operator Fusion)**，旨在减少 Kernel Launch 开销和显存读写（Memory Traffic）。

融合过程主要发生在 `torch/_inductor/scheduler.py` 的 `Scheduler.fuse_nodes()` 方法中。

## 2. 自定义融合 Pass (`pre_fusion_custom_pass`)

Inductor 允许用户通过 `pre_fusion_custom_pass` 在默认融合逻辑执行前介入，修改调度图（Scheduling Graph）。

### 2.1 定义与签名

```python
# 类型签名
Callable[[torch._inductor.graph.GraphLowering], torch._inductor.graph.GraphLowering]
```

- **执行时机**：在 `Scheduler` 初始化后，`fuse_nodes()` 被调用前。
- **输入/输出**：接收 `GraphLowering` 实例，必须返回修改后的实例。
- **作用域**：此时节点已转换为 `SchedulerNode`，但尚未进行融合分组。

### 2.2 编写示例

以下示例展示了如何编写一个自定义 Pass，用于防止大 Tensor 融合导致 OOM，以及强制融合特定模式。

```python
import torch
import torch._inductor.config as config
from torch._inductor.graph import GraphLowering

def my_pre_fusion_pass(graph: GraphLowering) -> GraphLowering:
    """
    自定义调度器融合前 Pass
    """
    for node in graph.nodes:
        fx_node = getattr(node, 'node', None)
        if fx_node is None:
            continue

        target = getattr(fx_node, 'target', None)
        if target is None:
            continue

        # 场景 1：防止大输出 Tensor 被融合 (OOM 防护)
        # 检查节点输出大小，超过 64MB 则禁止融合
        if hasattr(node, 'get_outputs'):
            outputs = node.get_outputs()
            if outputs and hasattr(outputs[0], 'numel'):
                # 假设 FP16，64MB ≈ 32M elements
                if outputs[0].numel() > 32 * 1024 * 1024:
                    node.fusable = False  # 关键：关闭融合标记
                    continue

        # 场景 2：强制融合特定模式 (add -> relu)
        if 'aten.add.Tensor' in str(target):
            for user in node.users:
                user_target = getattr(getattr(user, 'node', None), 'target', '')
                if 'aten.relu' in str(user_target):
                    node.fusable = True
                    user.fusable = True
                    break

    return graph

# 注册到 Inductor 配置
config.pre_fusion_custom_pass = my_pre_fusion_pass
```

### 2.3 核心 API 说明

| 对象/属性 | 说明 | 常用操作 |
| :--- | :--- | :--- |
| `graph.nodes` | 待调度的节点列表 (`SchedulerNode`) | 遍历、过滤、重排 |
| `node.node` | 底层 FX Node | 获取 `op`, `target`, `args` |
| `node.fusable` | 融合开关 (bool) | `True` 允许融合，`False` 强制隔离 |
| `node.users` | 数据依赖关系 | 检查消费者，构建自定义融合组 |

## 3. 默认融合策略详解

如果没有定义自定义 Pass，Inductor 会执行以下默认融合策略：

### 3.1 Vertical Fusion (垂直融合)
- **机制**：消费者融合生产者。将计算图中的上下游节点（如 `add` -> `relu`）融合到一个 Kernel 中。
- **目的**：消除中间显存读写。
- **代码位置**：`Scheduler.can_fuse_vertical()`

### 3.2 Horizontal Fusion (水平融合)
- **机制**：兄弟节点融合。将具有相同输入或计算模式的并行节点融合。
- **代码位置**：`Scheduler.can_fuse_horizontal()`

### 3.3 Reduction Fusion (归约融合)
- **场景**：当一个**逐元素操作 (Pointwise)** 的输出紧接着被一个**归约操作 (Reduction)** 消费时。
- **示例**：`torch.sum(x + 1.0)`
- **融合前**：
  1. Kernel A: 计算 `x + 1.0`，写回显存。
  2. Kernel B: 读取结果，计算 `sum`。
- **融合后**：
  - Fused Kernel: 读取 `x`，在寄存器中计算 `x + 1.0`，直接累加，**只输出最终标量**。
- **收益**：完全消除中间显存读写，对 Memory Bound 场景提升巨大。

### 3.4 Template Fusion (模板融合)
- **场景**：当计算图中出现**特定的算子组合模式**，且 Inductor 内置了针对该模式的手写高性能模板时。
- **示例**：`scaled_dot_product_attention` (FlashAttention)
- **融合前**：Inductor 尝试生成通用的 MatMul, Softmax 等 Kernel 序列。
- **融合后**：
  1. **模式匹配**：Scheduler 命中内置的 `AttentionTemplate`。
  2. **替换**：将子图替换为 `TemplateNode`。
  3. **生成**：调用预编译好的、高度优化的 FlashAttention Kernel。
- **收益**：算法升级，利用硬件特定指令（如 MFMA），通常带来数倍性能提升。

## 4. 融合问题排查指南

### 4.1 开启调试模式
```bash
export TORCH_COMPILE_DEBUG=1
export TORCH_LOGS="+inductor"
```
生成的 `torch_compile_debug` 目录包含：
- `fx_graph_readable.html`: 原始 FX 图。
- `post_grad_graph_*.txt`: **融合后的图**（查看 `FusedSchedulerNode`）。
- `triton_kernel_*.py`: 生成的 Triton 代码。

### 4.2 编译报错 (Compilation Errors)
- **常见原因**：
  - 动态 Shape 问题（未处理的 `SymInt`）。
  - 不支持的算子（Fallback 到 Eager）。
- **排查**：检查日志中的 `FALLBACK` 警告，查看生成的 Triton 代码是否有语法错误。

### 4.3 内存 OOM 问题
`reorder_for_peak_memory` 旨在解决 OOM，若仍失败：
1. **检查融合是否过度**：
   - 设置 `config.max_fused_size = 1` 禁用融合。若 OOM 消失，说明是融合策略问题。
2. **检查 Fusion Groups**：
   - 查看 Debug 日志中 `FusedSchedulerNode` 的大小。
3. **手动干预**：
   - 检查长生命周期的大 Tensor。
   - 在模型代码中显式 `del tensor`。

### 4.4 性能回退
- 对比 `config.pre_fusion_custom_pass = None` 的基准性能。
- 确认 Template Fusion 是否命中（如 Attention 未命中会导致性能大幅下降）。
