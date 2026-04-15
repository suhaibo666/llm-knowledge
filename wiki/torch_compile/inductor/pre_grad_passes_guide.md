# PyTorch Inductor Pre-Grad Passes 深度解析

## 目录
1. [概述](#1-概述)
2. [Pass 详解](#2-pass-详解)
3. [执行顺序与依赖关系](#3-执行顺序与依赖关系)
4. [自定义开发 Pass 指南](#4-自定义开发-pass-指南)

---

## 1. 概述

### 1.1 什么是 Pre-Grad Graph

Pre-Grad Graph 是 PyTorch Inductor 在**梯度计算之前**的 FX 计算图表示阶段。它位于 AOT Autograd 之前，直接处理用户编写的 PyTorch 代码转换后的 FX Graph。

```python
# 用户代码
def model(x, w):
    y = torch.nn.functional.linear(x, w)
    z = torch.cat([y, y], dim=1)
    return z

# Pre-Grad Graph（Torch IR 级别）
def pre_grad_graph(x, w):
    y = torch.nn.functional.linear(x, w)  # [B, M] @ [N, M].T -> [B, N]
    z = torch.cat([y, y], dim=1)          # [B, N] + [B, N] -> [B, 2N]
    return z
```

与 Post-Grad Graph（ATen IR）不同，Pre-Grad Graph：
- 使用 **Torch IR**（`torch.nn.functional` 级别）
- **尚未功能化/规范化** - 需要处理别名和变异
- 保留更多**高层语义信息**（如 `linear`、`layer_norm` 等）

### 1.2 Pre-Grad Passes 的作用

在功能化（Functionalization）和 AOT Autograd 之前，对图进行早期优化：

- **算子融合**：linear + permute、conv + bn 等
- **批量化优化**：将独立的小操作合并为批量操作
- **冗余消除**：split + cat 相互抵消、无意义操作移除
- **规范化**：统一操作调用形式，简化后续处理
- **内存优化**：sink cat、消除中间张量

### 1.3 Pre-Grad vs Post-Grad

| 特性 | Pre-Grad | Post-Grad |
|------|----------|-----------|
| IR 级别 | Torch IR (functional) | ATen IR (decomposed) |
| 执行时机 | AOT Autograd 之前 | AOT Autograd 之后 |
| 优化重点 | 高层语义融合、批量化 | 底层算子融合、内存规划 |
| 处理复杂度 | 高（需处理各种参数变体） | 低（已规范化） |

**WARNING**: Pre-Grad IR 不是功能化的，编写 pass 时需要小心处理别名和变异。

---

## 2. Pass 详解

### Pass 1: fuse_fx

**执行顺序**：在 `pre_grad_passes` 早期执行（如果不在 predispatch 模式）

**包含的子 Passes**：

#### 1.1 sink_cat_after_pointwise

**作用**：将 `cat` 操作下沉到逐点操作之后

```python
# 优化前
cat_result = torch.cat([a, b], dim=0)  # [2B, ...]
relu_result = torch.relu(cat_result)   # 对大张量执行 relu

# 优化后
relu_a = torch.relu(a)                 # 对小张量执行 relu
relu_b = torch.relu(b)
cat_result = torch.cat([relu_a, relu_b], dim=0)
```

**为什么**：
- 逐点操作在小张量上执行，可能更好的缓存局部性
- 为后续 split-cat 消除创造条件

#### 1.2 Permute 融合系列（仅限非 CPU 设备）

##### linear_permute_fusion
```python
# 优化前
y = F.linear(x, w, b)      # y = x @ w.T + b
z = y.permute(0, 2, 1)     # 最后两维转置

# 优化后
z = linear_transpose(x, w, b)
# 实现: torch.matmul(w, x.transpose(-1, -2)) + bias.unsqueeze(-1)
```

##### permute_linear_fusion
```python
# 优化前
x_t = x.permute(0, 2, 1)   # [B, M, K] -> [B, K, M]
y = F.linear(x_t, w, b)    # y = x_t @ w.T

# 优化后
y = transpose_linear(x, w, b)
# 实现: torch.matmul(x.transpose(-1, -2), w.T) + bias
```

##### permute_matmul_fusion
```python
# 优化前
a_t = a.permute(0, 2, 1)
b_t = b.permute(0, 2, 1)
c = torch.matmul(a_t, b_t)

# 优化后
c = transpose_matmul(a, b, Atrans=True, Btrans=True)
# 实现: a.transpose(-1, -2) @ b.transpose(-1, -2)
```

**为什么**：
- 消除显式 permute 的内存拷贝
- 利用矩阵乘法的转置性质：`A @ B = (B.T @ A.T).T`
- GPU 上批量矩阵运算更高效

**关键检查**：`check_permute()` 只处理**最后两维交换**的 permute

#### 1.3 CPU 推理优化（仅在 freezing 模式）

##### remove_identity
```python
# 消除 nn.Identity 层
y = nn.Identity()(x)  ->  y = x
```

##### fuse_conv_bn
**作用**：将 Conv + BatchNorm 融合为单个 Conv（推理模式）

```python
# 数学原理
# Conv: y = x * W + b
# BN:   z = (y - mean) / sqrt(var + eps) * gamma + beta

# 融合后: z = x * W_fused + b_fused
W_fused = W * gamma / sqrt(var + eps)
b_fused = beta + gamma / sqrt(var + eps) * (b - mean)
```

**限制条件**：
- 仅在 `config.freezing` 且 `torch.no_grad()` 时启用
- BN 必须处于 eval 模式且有 running stats

---

### Pass 2: normalization_pass

**作用**：将各种变体的操作统一成标准化的形式

**为什么必须第一位**：后续所有 pattern matching 都依赖统一的调用形式

**具体规则**：

| 规则 | 匹配模式 | 标准化结果 |
|------|---------|-----------|
| `normalize_split_default` | `split(x, 32, dim=0)` / `split(x, [16,16], dim=0)` | 统一为 `split(x, sections_list, dim)` |
| `normalize_unbind_default` | `unbind(x)` / `unbind(x, dim=0)` | 明确 `dim=0` |
| `normalize_cat_default` | `cat([a,b])` / `cat([a,b], dim=0)` | 明确 `dim=0` |
| `normalize_stack_default` | `stack([a,b])` | 明确 `dim=0` |
| `normalize_squeeze_default` | `squeeze(x)` / `squeeze(x, dim=1)` | 统一参数形式 |
| `normalize_reshape_default` | `reshape(x, shape)` | 确保元数据有效性 |
| `normalize_clamp_default` | `clamp(x, min, max)` | 简化参数形式 |
| `normalize_detach_default` | `detach(x)` | 适当情况内联消除 |

**示例**：
```python
# 原始（多种形式）
torch.split(x, 32, dim=1)
x.split([32, 32], 1)
torch.split(x, 32)  # 默认 dim=0

# 标准化后（统一形式）
torch.split(x, [32, 32], dim=1)  # 总是 (tensor, sections_list, dim)
```

---

### Pass 3: group_batch_fusion_passes

**核心机制**：将多个独立的相似操作合并为批量操作，减少 kernel 启动开销

#### 3.1 batch_linear（Pre-Grad）

**作用**：融合多个相同输入维度的 `F.linear` 为 `baddbmm`

```python
# 优化前
y1 = F.linear(x1, w1, b1)  # [M, K] @ [N1, K].T
y2 = F.linear(x2, w2, b2)  # [M, K] @ [N2, K].T
y3 = F.linear(x3, w3, b3)  # [M, K] @ [N3, K].T

# 优化后
X = torch.stack([x1, x2, x3])      # [3, M, K]
W = torch.stack([w1, w2, w3])      # [3, N, K]
W_t = W.transpose(1, 2)            # [3, K, N]
B = torch.stack([b1, b2, b3]).unsqueeze(1)  # [3, 1, N]
Y = torch.baddbmm(B, X, W_t)       # [3, M, N]
y1, y2, y3 = torch.unbind(Y, dim=0)
```

**收益**：
- 减少 kernel 启动开销（3次 → 1次）
- 更好的 GPU 并行利用率

#### 3.2 batch_linear_lhs

**特殊场景**：共享输入的多个 Linear

```python
# 优化前
q = F.linear(x, w_q)  # 共享输入 x
k = F.linear(x, w_k)
v = F.linear(x, w_v)

# 优化后：权重横向拼接，单次矩阵乘法
W_cat = torch.cat([w_q, w_k, w_v], dim=0).transpose(0, 1)
result = torch.mm(x, W_cat)
q, k, v = torch.split(result, [N_q, N_k, N_v], dim=1)
```

#### 3.3 batch_layernorm

```python
# 优化前
y1 = F.layer_norm(x1, x1.shape[-1:])
y2 = F.layer_norm(x2, x2.shape[-1:])

# 优化后
X = torch.stack([x1, x2], dim=-1)  # 在最后维 stack
Y = F.layer_norm(X, normalized_shape)
y1, y2 = torch.unbind(Y, dim=-1)
```

#### 3.4 逐点操作批量化

```python
# batch_tanh / batch_sigmoid / batch_relu / batch_detach / batch_nan_to_num / batch_clamp

# 优化前
y1 = torch.tanh(x1)
y2 = torch.tanh(x2)

# 优化后
X = torch.stack([x1, x2])
Y = torch.tanh(X)
y1, y2 = torch.unbind(Y, dim=0)
```

**限制参数**：
```python
MIN_FUSE_SET_SIZE = 5      # 最少融合 5 个节点
MAX_FUSE_SET_SIZE = 300    # 最多融合 300 个节点
MAX_FUSE_SEARCH_DEPTH = 5  # BFS 搜索深度
```

---

### Pass 4: Split/Cat 优化 Passes

#### 4.1 remove_split_with_size_one_pass

**作用**：将 `split + squeeze` 转换为更高效的 `unbind`

```python
# 优化前
parts = torch.split(x, [1,1,1], dim=1)  # 3个 [N,1,H] 张量
result = [p.squeeze(dim=1) for p in parts]  # 逐个 squeeze

# 优化后
result = torch.unbind(x, dim=1)  # 直接得到 [N,H] 张量列表
```

#### 4.2 merge_getitem_cat_pass

**作用**：合并 split 后部分 getitem 再 cat 的操作

```python
# 优化前
x1, x2, x3, x4 = torch.split(input, [a,b,c,d], dim=1)
y = torch.cat([x1, x2, x3], dim=1)  # 只取前3个

# 优化后
# 调整 split 分段，将前3段合并
x_fused, x4 = torch.split(input, [a+b+c, d], dim=1)
# 或者直接使用 slice
```

#### 4.3 merge_splits_pass

**作用**：合并嵌套的 split 操作

```python
# 优化前
parts = torch.split(input, [a,b,c], dim=0)
inner = torch.split(parts[1], [d,e], dim=0)  # 对第2段再split

# 优化后
result = torch.split(input, [a, d, e, c], dim=0)  # 扁平化为一次split
```

#### 4.4 mutate_cat_pass

**作用**：消除 split + getitem + cat 链条

```python
# Case 1: cat 使用所有 getitems
x1, x2, x3 = torch.split(input, sections, dim=1)
y = torch.cat([x1, x2, x3], dim=1)
# -> y = input  (完全消除)

# Case 2: cat 使用部分连续的 getitems
y = torch.cat([x2, x3], dim=1)
# -> y = input[:, a:a+b+c]  (转换为 slice)
```

#### 4.5 split_cat_pass

**作用**：优化 `split + squeeze` 模式

```python
# 优化前
parts = torch.split(x, [1,1,1], dim=1)
squeezed = [p.squeeze(dim=1) for p in parts]

# 优化后
squeezed = torch.unbind(x, dim=1)
```

#### 4.6 split_cat_to_slices_pass

**作用**：将 split + getitem + cat 转换为 slice 操作

```python
# 优化前
x1, x2, x3, x4 = torch.split(input, sections, dim=1)
y = torch.cat([x2, x3], dim=1)

# 优化后
y = input[:, sections[0]:sections[0]+sections[1]+sections[2]]
```

#### 4.7 unbind_cat_to_view_pass

**作用**：将 `unbind → cat` 转换为 view/reshape

```python
# 优化前
parts = torch.unbind(x, dim=0)  # 沿 batch 拆
y = torch.cat(parts, dim=1)     # 沿特征维合并

# 优化后
y = x.reshape(...)  # 重新解释布局，无数据移动
```

#### 4.8 split_stack_to_cats_pass

**作用**：将 split + stack 转换为 cat + reshape

```python
# 优化前
parts = torch.split(x, sections, dim=1)
y = torch.stack(parts, dim=0)

# 优化后
y = torch.cat([...], dim=1).reshape(...)
```

#### 4.9 move_reshape_out_of_split_stack_pass

**作用**：将 reshape 从 split/stack 链条中移出

```python
# 优化前
parts = torch.split(x, sections, dim=1)
reshaped = [p.reshape(...) for p in parts]
y = torch.stack(reshaped, dim=0)

# 优化后
y = torch.cat([...], dim=1).reshape(...)  # 推迟 reshape
```

---

### Pass 5: 特殊优化 Passes

#### 5.1 efficient_conv_bn_eval_pass

**作用**：高效的 ConvBN 执行方式（不合并权重，运行时动态调整）

**基于论文**："Efficient ConvBN Blocks for Transfer Learning and Beyond" (https://arxiv.org/abs/2305.11624)

```python
# 数学原理：利用结合律
# normalize(weight * conv(x)) = (normalize weight) * conv(x)

# 运行时调整 Conv 权重
W_eff = W * gamma / sqrt(var + eps)
b_eff = beta + coeff * (b - mean)

# 单次卷积，无需保存中间特征图
output = conv(x, W_eff, b_eff)
```

**vs 传统 fuse_conv_bn**：
- 传统：编译时合并权重，永久改变模型
- Efficient：运行时调整，可用于训练（如果 bn.training=False）

**收益**：
- 减少内存占用（不保存 conv 输出）
- 减少计算量（省去 BN 操作）

#### 5.2 apply_gumbel_max_trick_pass

**作用**：优化 Gumbel-Softmax 采样中的计算

```python
# 原始实现（常见模式）
softmax_probs = F.softmax(logits, dim=-1)
gumbel_noise = -torch.log(-torch.log(torch.rand(shape)))
scaled = softmax_probs / torch.exp(gumbel_noise)
sample = torch.argmax(scaled, dim=-1)

# 优化后（Gumbel-Max Trick）
gumbel_noise = -torch.log(exponential_random)
sample = torch.argmax(logits + gumbel_noise, dim=-1)
```

**数学等价变换**：
```
argmax(softmax(x) / exp(g)) 
= argmax(exp(x) / sum(exp(x)) / exp(g))
= argmax(exp(x - g)) 
= argmax(x - g)
```

**收益**：
- 跳过 softmax 计算
- 更数值稳定
- 减少指数运算次数

---

## 3. 执行顺序与依赖关系

```
Stage 1: 基础 Fusion 与规范化
─────────────────────────────────
fuse_fx                         ──> linear/permute 融合、ConvBN 融合
│   ├─ sink_cat_after_pointwise
│   ├─ linear_permute_fusion
│   ├─ permute_linear_fusion
│   ├─ permute_matmul_fusion
│   ├─ remove_identity
│   └─ fuse_conv_bn
│
numpy_compat_normalization      ──> NumPy 兼容性处理
│
normalization_pass              ──> 统一操作调用形式
│   ├─ normalize_split_default
│   ├─ normalize_unbind_default
│   ├─ normalize_cat_default
│   └─ ... (其他 normalize)
│

Stage 2: 批量化与分组融合
─────────────────────────────────
group_batch_fusion_passes       ──> 批量化独立操作
│   ├─ batch_linear
│   ├─ batch_linear_lhs
│   ├─ batch_layernorm
│   └─ batch_*(tanh/sigmoid/relu...)
│

Stage 3: Split/Cat 结构优化
─────────────────────────────────
PRE_GRAD_PATTERNS               ──> 按配置顺序执行
│   ├─ remove_split_with_size_one_pass
│   ├─ merge_getitem_cat_pass
│   ├─ merge_splits_pass
│   ├─ mutate_cat_pass
│   ├─ split_cat_pass
│   ├─ unbind_stack_pass
│   ├─ split_cat_to_slices_pass
│   ├─ unbind_cat_to_view_pass
│   └─ ... (其他 split-cat 优化)
│

Stage 4: 特殊优化
─────────────────────────────────
efficient_conv_bn_eval_pass     ──> 高效 ConvBN 执行
apply_gumbel_max_trick_pass     ──> Gumbel 采样优化
│

Stage 5: 后处理
─────────────────────────────────
stable_topological_sort         ──> 稳定拓扑排序
quant_lift_up                   ──> 量化相关处理
lint + recompile                ──> 验证与编译
```

### 关键依赖关系

1. **normalization_pass → 所有 Pattern Matching**
   - 必须先统一调用形式，后续 pattern 才能正确匹配

2. **fuse_fx permute 融合 → group_batch_fusion**
   - permute 融合后的 linear 可能更适合批量化

3. **group_batch_fusion → split_cat_passes**
   - batch fusion 产生的 stack/unbind 链条需要后续 split-cat 优化消除

4. **split_cat_to_slices_pass → move_reshape_out_of_split_stack_pass**
   - slice 优化后，reshape 移动才能正确识别模式

---

## 4. 自定义开发 Pass 指南

### 4.1 Pre-Grad Pass 的特点

**与 Post-Grad 的主要区别**：

```python
# Pre-Grad: 处理 Torch IR
torch.nn.functional.linear(x, w, b)  # 高层 op
torch.cat([a, b], dim=0)             # 带 dim 参数

# Post-Grad: 处理 ATen IR  
torch.ops.aten.mm.default(x, w.T)    # 底层 op
torch.ops.aten.cat.default([a, b], 0)  # 已规范化
```

**注意事项**：
- IR 未规范化，需要处理各种参数变体
- 需要小心处理别名和变异
- 保留 `example_value` 元数据用于形状推断

### 4.2 注册新 Pattern 的方法

#### 方式一：使用 PatternMatcherPass（推荐）

```python
from torch._inductor.fx_passes.split_cat import PRE_GRAD_PATTERNS
from torch._inductor.pattern_matcher import (
    register_graph_pattern,
    CallFunction,
    KeywordArg,
    Match,
)
import torch

# 注册到 Pre-Grad patterns
my_pass = PatternMatcherPass(pass_name="my_custom_pass")

@register_graph_pattern(
    CallFunction(
        torch.nn.functional.linear,
        KeywordArg("input"),
        KeywordArg("weight"),
    ),
    pass_dict=my_pass,
)
def optimize_my_linear(match: Match, input, weight):
    """自定义 linear 优化"""
    node = match.output_node()
    graph = match.graph
    
    # 检查条件
    if not is_node_meta_valid(input):
        return
    
    # 创建替换节点
    with graph.inserting_before(node):
        new_node = graph.call_function(
            torch.ops.aten.mm,
            args=(input, weight.T),
        )
        new_node.meta.update(node.meta)
    
    node.replace_all_uses_with(new_node)
    graph.erase_node(node)

# 添加到 PRE_GRAD_PATTERNS
PRE_GRAD_PATTERNS["my_custom_pass"] = my_pass
```

#### 方式二：Graph Pass（全图遍历）

```python
def my_graph_pass(graph: torch.fx.GraphModule):
    """遍历整个 graph 进行复杂变换"""
    for node in graph.graph.nodes:
        if should_optimize(node):
            # 复杂的多节点变换
            pass
    
    graph.graph.lint()
    graph.recompile()

# 在 pre_grad_passes 中调用
# 或配置为 custom pass
```

### 4.3 常见 Pre-Grad 优化模式

#### A. 高层算子融合

```python
# 模式：融合常见序列
# linear + activation -> linear_activation fused
# conv + bn + relu -> conv_bn_relu fused

@register_graph_pattern(
    CallFunction(
        torch.relu,
        CallFunction(
            torch.nn.functional.linear,
            KeywordArg("input"),
            KeywordArg("weight"),
            KeywordArg("bias"),
        ),
    ),
    pass_dict=PRE_GRAD_PATTERNS["linear_relu_fusion"],
)
def fuse_linear_relu(match, input, weight, bias):
    # 替换为 fused linear_relu
    pass
```

#### B. 批量化优化

```python
# 模式：收集相似操作并批量化
# 参考 BatchFusion 基类的实现

class MyBatchFusion(BatchFusion):
    def match(self, node):
        # 返回 group key，相同 key 的节点会被批量处理
        if is_target_op(node):
            shape = node.meta["example_value"].shape
            return ("my_batch_key", str(shape))
        return None
    
    def fuse(self, graph, subset):
        # 批量融合逻辑
        # 1. stack 输入
        # 2. 批量操作
        # 3. unbind 结果
        pass
```

#### C. 结构消除

```python
# 模式：消除相互抵消的操作
# split + cat（相同维度）-> 原始张量或 slice
# stack + unbind -> 原始张量（如果维度匹配）

@register_graph_pattern(
    CallFunction(
        torch.cat,
        getitem_split,  # List of getitem from split
        dim=Ignored(),
    ),
    pass_dict=PRE_GRAD_PATTERNS["split_cat_elimination"],
)
def eliminate_split_cat(match, ...):
    # 检查是否使用所有 split 结果且维度匹配
    # 替换为原始输入或 slice
    pass
```

### 4.4 开发流程

```python
# Step 1: 识别优化机会
# 通过查看 FX Graph 发现冗余模式
# torch._inductor.config.joint_custom_pre_pass = debug_graph_pass

def debug_graph_pass(graph):
    print(graph.graph)
    return graph

# Step 2: 编写 Pattern
@register_graph_pattern(...)
def my_optimization(match, ...):
    # 实现替换逻辑
    pass

# Step 3: 添加测试
class TestMyOptimization(TestCase):
    def test_my_opt(self):
        def fn(x):
            # 测试 pattern
            return optimize_result
        
        # 编译并运行
        compiled = torch.compile(fn)
        # 验证正确性和性能

# Step 4: 验证
# - 单测通过
# - 数值精度无损
# - 实际模型性能提升
```

### 4.5 最佳实践

**1. 始终检查元数据有效性**
```python
def my_match(node):
    if not is_node_meta_valid(node):
        return None
    shape = node.meta["example_value"].shape
    # ...
```

**2. 保留元数据信息**
```python
def my_fuse(graph, subset):
    with graph.inserting_before(node):
        new_node = graph.call_function(...)
        # 必须复制元数据！
        new_node.meta.update(node.meta)
        # 特别是 example_value
        new_node.meta["example_value"] = compute_output(...)
```

**3. 处理动态形状**
```python
from torch.fx.experimental.symbolic_shapes import free_symbols

if free_symbols(node.meta["example_value"].shape):
    # 动态形状需要特殊处理
    return
```

**4. 使用配置控制**
```python
if not torch._inductor.config.my_optimization:
    return

# 或在配置中添加选项
config.pre_grad_fusion_options["my_pass"] = {"enabled": True}
```

### 4.6 调试技巧

```python
# 打印匹配信息
@register_graph_pattern(...)
def my_pattern(match, ...):
    print(f"Matched: {match.nodes}")
    print(f"Args: {...}")

# 导出优化前后的 graph
torch._inductor.config.joint_custom_pre_pass = \
    lambda g: (print("Before:", g.graph), g)[1]
torch._inductor.config.joint_custom_post_pass = \
    lambda g: (print("After:", g.graph), g)[1]

# 使用 counters 统计优化次数
from torch._dynamo.utils import counters
counters["inductor"]["my_pass"] += 1
```

---

## 总结

Pre-Grad Passes 是 PyTorch Inductor 编译流程中的**早期优化阶段**，主要特点：

1. **高层语义优化**：处理 `F.linear`、`F.layer_norm` 等高级 API
2. **批量化机会**：将独立小操作合并为批量操作
3. **结构消除**：消除 split/cat、stack/unbind 等相互抵消的操作
4. **为后续阶段做准备**：规范化操作形式，简化 Post-Grad 优化

自定义开发 Pre-Grad Pass 的关键：
- 理解 Torch IR 与 ATen IR 的区别
- 正确处理未规范化的参数变体
- 维护 `example_value` 元数据
- 注意别名和变异的安全性
