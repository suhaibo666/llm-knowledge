# PyTorch Inductor Joint Graph Passes 完全解析

## 目录
1. [概述](#1-概述)
2. [Pass 详解](#2-pass-详解)
3. [执行顺序与依赖关系](#3-执行顺序与依赖关系)
4. [自定义开发 Pass 指南](#4-自定义开发-pass-指南)

---

## 1. 概述

### 1.1 什么是 Joint Graph

Joint Graph 是 PyTorch 2.0 编译器（Inductor）在训练场景下的核心数据流图，它包含：
- **Forward 计算**：模型的前向传播
- **Backward 计算**：自动微分生成的反向传播

```python
# 用户代码
def model(x, w):
    y = x @ w
    loss = y.sum()
    return loss

# AOT Autograd 生成的 Joint Graph
def joint_graph(x, w, grad_output):
    # Forward
    y = x @ w
    loss = y.sum()
    
    # Backward (自动插入)
    grad_y = grad_output
    grad_x = grad_y @ w.T
    grad_w = x.T @ grad_y
    
    return loss, grad_x, grad_w
```

### 1.2 Joint Graph Passes 的作用

在代码生成之前，对 Joint Graph 进行**图级优化**，主要包括：
- **消除冗余操作**：clone、类型转换、无操作算术
- **算子融合**：将多个小算子合并成大算子
- **内存优化**：常量折叠、分块计算
- **设备优化**：修复设备不匹配问题

---

## 2. Pass 详解

### Pass 1: canonicalize_aten_ir_passes

**执行顺序**：第1个（必须在最前）

**核心代码**：
```python
def canonicalize_aten_ir_passes(gm: torch.fx.GraphModule):
    canonicalize_quant_mapping(gm)
```

**作用**：
将 AOT Autograd 产生的非标准表示统一成标准 ATen IR。

**关键问题**：
Pattern Matcher 基于 `(op_type, target)` 元组索引。如果同一操作有不同表示，只能匹配到其中一种。

**示例**：
```python
# 输入（非标准表示）
invoke_quant_packed(subgraph, 'quant_invoke_0_0', (arg0, arg1))

# 输出（标准表示）
invoke_quant(subgraph, arg0, arg1, scheme='nf4')
```

**为什么必须在第一位**：
后续所有 pattern matching 都依赖统一的标准表示。没有规范化，部分节点会"隐形"，导致优化失效。

---

### Pass 2 & 9: 自定义 Hooks

**配置项**：
```python
config.joint_custom_pre_pass   # 在其他 passes 之前执行
config.joint_custom_post_pass  # 在其他 passes 之后执行
```

**用途**：
- 插入调试代码（打印中间图状态）
- 实验性优化
- 领域特定优化

**示例**：
```python
def my_custom_pass(graph: torch.fx.GraphModule):
    # 自定义图变换
    for node in graph.graph.nodes:
        if node.target == torch.ops.aten.exp:
            # 替换为更快的近似实现
            pass

# 配置
torch._inductor.config.joint_custom_pre_pass = my_custom_pass
```

---

### Pass 3: remove_noop_ops

**核心机制**：
```python
noop_registry = {
    aten.clone.default: (lambda **kwargs: True, 0),
    aten.alias.default: (lambda **kwargs: True, 0),
    aten.add.Tensor: (check_add_nop, 0),      # add(x, 0) -> x
    aten.sub.Tensor: (check_sub_nop, 0),      # sub(x, 0) -> x
    aten.mul.Tensor: (check_mul_nop, 1),      # mul(x, 1) -> x
    aten.div.Tensor: (check_div_nop, 0),      # div(x, 1) -> x
    aten.view.default: (check_view_nop, 0),   # view 不变 shape
    aten.slice.Tensor: (check_slice_nop, 0),  # 完整切片 [0:全长:1]
    aten.slice_scatter: (check_slice_scatter_nop, 1),
    aten.repeat: (check_repeat_nop, 0),       # repeat 全为 1
    aten.constant_pad_nd: (check_pad_nop, 0), # padding 全为 0
    aten.cat: (check_cat_nop, lambda args: args[0][0]),  # cat 单输入
    prims.convert_element_type: (check_convert_nop, 0),  # dtype 相同
    prims.device_put: (check_device_put_nop, 0),         # device 相同
    aten.pow: (check_pow_nop, 0),             # pow(x, 1) -> x
    aten.ceil/floor/round/trunc: (check_int_nop, 0),  # 整数输入
}
```

**关键检查：别名安全**
```python
# 不能引入新的输入-输出别名关系
if (
    not node_is_view
    and node_storage in output_storages
    and (src_storage in input_storages or src_storage in output_storages)
):
    continue  # 跳过优化，避免破坏语义
```

**示例**：
```python
# 原始（Autograd 插入的 clone）
y = x.clone()
z = y + 1

# 优化后（如果安全）
z = x + 1  # 直接复用 x，消除 clone
```

**风险示例**（没有别名检查会出错）：
```python
def forward(x):
    y = x.clone()  # 意图：y 和 x 不共享存储
    y.add_(1)      # 修改 y，不应该影响 x
    return y, x    # 两个都输出

# 错误优化：移除 clone
def forward_broken(x):
    y = x          # 现在 y 和 x 共享存储！
    y.add_(1)      # 这也修改了 x！
    return y, x    # 两个都被修改了，错误！
```

---

### Pass 4: constant_fold_uniform_value

**核心类**：`UniformValueConstantFolder`

**解决的问题**：
标准常量折叠会将大张量内联到 graph 中，导致代码膨胀。这个 pass 只折叠可以用 `aten.full` 构造的 uniform value 张量。

**优化流程**：
```python
# 原始 Graph
def forward():
    x = torch.zeros(1024, 1024)  # 全 0
    y = torch.ones(1024, 1024)   # 全 1
    z = x + y                    # 全 1
    return z

# 常量折叠后
def forward():
    # 所有 uniform 张量变成 full 构造器
    x = aten.full([1024, 1024], 0)
    y = aten.full([1024, 1024], 1)
    z = aten.full([1024, 1024], 1)  # x+y 被折叠
    
    # 进一步调用 remove_no_ops：全 0/全 1 标记用于后续优化
    return z
```

**支持的传播规则**：
```python
# 1. 单元素属性
get_attr(tensor) with tensor.numel() == 1  -> 折叠

# 2. 构造函数
aten.full([...], value)  -> 折叠为 [1] 表示

# 3. View 操作传播 uniform value
view/uniform/permute/expand 输入是 uniform -> 输出也是 uniform

# 4. Pointwise 操作
add/mul/sub 等：实际执行计算，看结果是否 uniform
```

---

### Pass 5: early_patterns

**包含的 Patterns**：

#### A. 冗余 View 消除
```python
# view(x, x.shape) -> x
@register_graph_pattern(
    CallFunction(aten.view.default, KeywordArg("arg"), KeywordArg("size")),
    pass_dict=early_patterns,
)
def pointless_view(match, arg, size):
    if definitely_equal(arg.shape, size):
        node.replace_all_uses_with(arg)

# view(view(x, s1), s2) -> view(x, s2) 如果 s2 == x.shape
@register_graph_pattern(
    CallFunction(aten.view.default,
        CallFunction(aten.view.default, KeywordArg("arg"), KeywordArg("size1")),
        KeywordArg("size2")),
    pass_dict=early_patterns,
)
def pointless_view_pair(match, arg, size1, size2):
    if definitely_equal(arg.shape, size2):
        node.replace_all_uses_with(arg)
```

#### B. 冗余 Permute 消除
```python
# permute(permute(x, p1), p2) -> x 如果 p1 和 p2 互为逆排列
@register_graph_pattern(
    CallFunction(aten.permute.default,
        CallFunction(aten.permute.default, KeywordArg("arg"), KeywordArg("perm1")),
        KeywordArg("perm2")),
    pass_dict=early_patterns,
)
def pointless_permute_pair(match, arg, perm1, perm2):
    # 检查 perm1[perm2[i]] == i 对所有 i 成立
    if is_inverse_permutation(perm1, perm2):
        node.replace_all_uses_with(arg)
```

**为什么要在 AutoChunker 之前**：
这些 view/permute 如果不清理，会干扰分块边界的识别，使 AutoChunker 无法找到最优分块策略。

---

### Pass 6: auto_chunker

**触发条件**：
```python
if config.auto_chunker.enable:
    from .auto_chunker import CantChunk, chunk
    graph = chunk(graph)
```

**核心思想**：
将大矩阵操作分块，逐块计算 forward + backward，减少激活内存峰值。

**适用场景**：
- Joint Graph（有 tangent nodes，即需要 backward）
- 存在 "amplifier node"（尺寸放大的操作如 bmm）
- 静态形状（暂不支持动态 shape）

**Amplifier Node 定义**：
```python
def find_amplifier_node(graph):
    # 找输出尺寸远大于输入的操作
    for node in graph.nodes:
        if node.target in (aten.mm, aten.bmm, aten.addmm):
            input_size = prod(node.args[0].meta["val"].shape)
            output_size = prod(node.meta["val"].shape)
            if output_size / input_size > threshold:
                return node
```

**分块流程**：
```python
# 1. 找到放大器节点
amplifier_node = find_amplifier_node(graph)

# 2. 传播分块信息到前后节点
propagate(amplifier_node)

# 3. 检查是否能分块（tangent 有 chunking meta）
if not tangent_has_chunking_meta(gm):
    raise CantChunk

# 4. 应用分块
num_chunks = config.auto_chunker.num_chunk or decide_num_chunks(gm)
out_gm = ChunkingApplier(gm, num_chunks).apply()
```

**示例**：
```python
# 原始（峰值内存：保存所有中间结果）
def joint_graph(a, b, w):
    # Forward
    c = torch.bmm(a, b)      # [B, M, N] - 大激活！
    d = c @ w
    loss = d.sum()
    
    # Backward（需要 c 保留到这里）
    grad_c = ...  # 使用 c
    return loss

# 分块后（峰值降低）
def joint_graph_chunked(a, b, w, num_chunks=8):
    for i in range(num_chunks):
        a_c = a[i*B//8:(i+1)*B//8]
        b_c = b[i*B//8:(i+1)*B//8]
        
        # Forward chunk
        c_c = torch.bmm(a_c, b_c)  # 小激活 [B/8, M, N]
        d_c = c_c @ w
        
        # 立即 Backward chunk
        grad_c_c = ...  # 使用完立即释放 c_c
        
    return total_loss
```

**关键机制：invoke_subgraph**
```python
# 实际实现使用 higher order op
chunked_result = torch.ops.higher_order.invoke_subgraph(
    subgraph,           # 包含 chunk forward + backward 的子图
    "chunked_bmm",
    (a_chunk, b_chunk)
)
```

---

### Pass 7: pass_patterns（主优化）

**初始化**：
```python
pass_patterns = [patterns, PatternMatcherPass()]

def lazy_init():
    from .fuse_attention import _sfdp_init      # SDPA 融合
    from .misc_patterns import _misc_patterns_init
    from .pad_mm import _pad_mm_init           # 矩阵乘法 padding
    
    _pad_mm_init()
    _sfdp_init()
    _misc_patterns_init()
```

#### A. 设备修复（fix_iota_device）

```python
# 问题：CPU arange 索引 CUDA tensor 导致隐式拷贝
indices = torch.arange(10, device='cpu')  # CPU tensor
result = cuda_tensor[indices]             # 隐式 H2D 拷贝！

# 优化：将 arange 移到 CUDA
indices = torch.arange(10, device='cuda')  # CUDA tensor
result = cuda_tensor[indices]              # 无拷贝
```

#### B. 类型转换链优化（pointless_convert）

```python
# AMP 常见模式
x_f32 = input
x_f16 = x_f32.to(torch.float16)   # 内层转换
x_f32_again = x_f16.to(torch.float32)  # 外层转换

# 优化：f32 -> f16 -> f32 直接变成 f32
x_f32_again = input  # 完全消除转换链
```

#### C. Batch Matmul 优化（bmm_to_mm）

```python
# bmm([1, M, K], [1, K, N]) -> mm([M, K], [K, N]).unsqueeze(0)
# 当 batch=1 时，用更高效的 mm 代替 bmm
```

#### D. Softmax 数值稳定性

```python
# 问题：scale(x) - scale(x).amax() 可能数值不稳定
# 因为 scale(x) 计算了两次

# 优化：scale(x - x.amax()) 
# 数学等价，但只计算一次 scaling
```

#### E. Scatter 优化（scatter_upon_const_tensor）

```python
# 模式：在 full 张量上做 scatter
result = torch.full((B, C), 0.01)
result.scatter_(1, indices, 0.9)  # 随机写入，需要 atomic

# 优化：转换为 pointwise where 操作
mask = (arange(C) == indices.unsqueeze(1))  # [B, C]
result = torch.where(mask, 0.9, 0.01)       # 全向量化，无 atomic
```

---

### Pass 8: replace_random_passes

**功能**：
1. 替换随机数相关操作
2. 融合种子创建

**种子融合**：
```python
# 优化前：每个随机 op 单独创建 seed
seed1 = inductor_seed(cuda)
seed2 = inductor_seed(cuda)
seed3 = inductor_seed(cuda)  # 3 次 kernel launch

# 优化后：批量创建
seeds = inductor_seeds(3, cuda)     # 1 次 kernel launch
seed1 = inductor_lookup_seed(seeds, 0)
seed2 = inductor_lookup_seed(seeds, 1)
seed3 = inductor_lookup_seed(seeds, 2)
```

**可复现性说明**：
```python
# 注释说明为什么不进 bisector
# "decomps may have already affected rng reproducibility"
# 如果 config.fallback_random=True，完全回退到 eager 随机实现
```

---

## 3. 执行顺序与依赖关系

```
Stage 1: 规范化 & 用户前置处理
─────────────────────────────────
1. canonicalize_aten_ir_passes  ──> 统一 IR 表示（必须在最前）
2. joint_custom_pre_pass        ──> 用户自定义（可选）
                              
Stage 2: 基础清理与常量优化
─────────────────────────────────
3. remove_noop_ops              ──> 消除克隆/别名/noop
4. constant_fold_uniform_value  ──> 折叠 uniform 常量
                              
Stage 3: 结构简化（为分块做准备）
─────────────────────────────────
5. early_patterns               ──> 清理冗余 view/permute
                                   （必须在 auto_chunker 之前）
                              
Stage 4: 大粒度策略优化
─────────────────────────────────
6. auto_chunker                 ──> 自动分块（改变 graph 结构）
                              
Stage 5: 主要计算优化
─────────────────────────────────
7. pass_patterns                ──> 算子融合、设备优化等
8. replace_random_passes        ──> 随机数优化
                              
Stage 6: 后处理
─────────────────────────────────
9. joint_custom_post_pass       ──> 用户自定义（可选）
                                  
最后: stable_topological_sort + lint + recompile
```

**关键依赖关系**：
- `canonicalize` -> 所有 pattern matching（标准表示）
- `early_patterns` -> `auto_chunker`（简化 view 边界）
- `auto_chunker` -> `pad_mm`（在 padding 前分块）

---

## 4. 自定义开发 Pass 指南

### 4.1 注册新 Pattern 的方法

#### 方式一：Graph Pattern（推荐）

```python
from torch._inductor.pattern_matcher import (
    register_graph_pattern,
    CallFunction,
    KeywordArg,
    Match,
    pass_patterns,
)

# 在 joint_graph.py 或新建文件
@register_graph_pattern(
    CallFunction(
        torch.ops.aten.your_target,
        KeywordArg("input"),
        KeywordArg("param"),
    ),
    pass_dict=pass_patterns[0],  # 或 patterns, early_patterns
)
def optimize_your_op(match: Match, input, param):
    """Pattern 的文档说明"""
    # 获取匹配到的节点
    node = match.output_node()
    
    # 创建替换节点
    with match.graph.inserting_before(node):
        replacement = match.graph.call_function(
            torch.ops.aten.better_op,
            args=(input, param),
        )
        replacement.meta.update(node.meta)
        
    # 替换所有使用
    node.replace_all_uses_with(replacement)
    match.erase_nodes()
```

#### 方式二：使用 Match.replace_by_example

```python
def optimize_pattern(match: Match, x, y):
    """用示例函数生成替换"""
    
    def replacement_example(x, y):
        # 编写理想的计算方式
        return torch.nn.functional.silu(x) * y
    
    # 自动将示例函数编译成 FX subgraph 替换
    match.replace_by_example(replacement_example, [x, y])
```

#### 方式三：Graph Pass（全图遍历）

```python
def my_graph_pass(graph: torch.fx.GraphModule):
    """遍历整个 graph 进行复杂变换"""
    for node in graph.graph.nodes:
        if should_optimize(node):
            # 复杂的多节点变换
            pass
    
    graph.graph.lint()
    graph.recompile()

# 注册到配置
torch._inductor.config.joint_custom_post_pass = my_graph_pass
```

---

### 4.2 常见优化模式分类

#### A. 消除冗余操作（Algebraic Optimization）

```python
# 模式：op(op(x)) -> x
# permute(permute(x, p1), p2) -> x

# 模式：op(x, identity) -> x  
# add(x, 0), sub(x, 0), mul(x, 1), div(x, 1) -> x

# 模式：合并连续操作
# view(view(x, s1), s2) -> view(x, s2) if s2 == original_shape
```

#### B. 类型/设备优化

```python
# 模式：冗余类型转换
# convert(convert(x, f16), f32) -> convert(x, f32)

# 模式：修复设备不匹配
# arange(cpu) 索引 cuda_tensor -> arange(cuda)
```

#### C. 内存布局优化

```python
# 模式：连续的转置链
# transpose(transpose(x, 0, 1), 0, 1) -> x

# 模式：无意义的 to(contiguous)
# contiguous(contiguous(x)) -> contiguous(x)
```

#### D. 算子融合

```python
# 模式：逐点操作融合
# add(mul(x, s), b) -> addmm(b, x, diag(s))

# 模式：归一化融合
# div(sub(x, mean), std) -> layer_norm(x)
```

#### E. 特定领域优化

```python
# 注意力融合
# scale_dot_product_attention 替换手动实现的 sdp

# 矩阵乘法分块
# 大 bmm 分块成多个小 bmm

# 嵌入层优化
# embedding + dropout + scale 融合
```

---

### 4.3 寻找优化点的方法

#### 方法 1：性能分析（Triton/Inductor Profiler）

```python
# 启用详细日志
import torch._logging
torch._logging.set_logs(inductor=True)
torch._inductor.config.trace.enabled = True

# 运行模型，查看生成的代码
with torch.no_grad():
    compiled_model = torch.compile(model)
    output = compiled_model(input)

# 查看生成的 Triton kernel
# 位置：/tmp/torchinductor_xxx/
```

#### 方法 2：FX Graph 可视化

```python
# 导出 graph 检查
def debug_graph_pass(graph):
    print(graph.graph)
    # 或使用 graph.print_readable()
    return graph

torch._inductor.config.joint_custom_post_pass = debug_graph_pass
```

#### 方法 3：寻找常见反模式

**检查列表**：
1. **重复计算**：同一个子图被计算多次
2. **冗余转换**：连续的 dtype 或 device 转换
3. **低效实现**：手动实现的 op 可以用更高效的 fused op 替代
4. **内存带宽瓶颈**：多个小 kernel 可以合并
5. **不规则访问**：scatter/gather 可以转换为 pointwise

```python
# 示例：检查重复子图
from collections import defaultdict

def find_duplicate_subgraphs(graph):
    subgraph_hashes = defaultdict(list)
    
    for node in graph.nodes:
        # 计算子图指纹（简化示例）
        fingerprint = (node.target, tuple(id(a) for a in node.args))
        subgraph_hashes[fingerprint].append(node)
    
    # 报告重复
    for fingerprint, nodes in subgraph_hashes.items():
        if len(nodes) > 1:
            print(f"Potential duplicate: {fingerprint}, count={len(nodes)}")
```

#### 方法 4：参考其他框架

- **XLA**：AllReduce 融合、集体通信优化
- **TensorRT**：Layer fusion、Kernel auto-tuning
- **ONNX Runtime**：Constant folding、Operator fusion

---

### 4.4 性能验证工具

#### 基准测试

```python
import time
import torch.utils.benchmark as benchmark

def benchmark_pattern(fn, args, num_runs=100):
    # Warmup
    for _ in range(10):
        fn(*args)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        fn(*args)
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / num_runs

# 对比优化前后
time_before = benchmark_pattern(original_fn, args)
time_after = benchmark_pattern(optimized_fn, args)
print(f"Speedup: {time_before / time_after:.2f}x")
```

#### 内存分析

```python
import torch

def profile_memory(fn, args):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    result = fn(*args)
    torch.cuda.synchronize()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    return peak_memory

mem_before = profile_memory(original_fn, args)
mem_after = profile_memory(optimized_fn, args)
print(f"Memory saved: {mem_before - mem_after:.2f} GB")
```

#### 正确性验证

```python
def verify_correctness(original_fn, optimized_fn, args):
    with torch.no_grad():
        expected = original_fn(*args)
        actual = optimized_fn(*args)
    
    # 数值检查
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    print("Correctness verified!")

# 梯度检查
def verify_gradients(model, inputs):
    inputs_copy = [x.clone().detach().requires_grad_(True) for x in inputs]
    
    output = model(*inputs)
    loss = output.sum()
    loss.backward()
    
    # 检查梯度是否存在
    for i, inp in enumerate(inputs_copy):
        assert inp.grad is not None, f"Input {i} has no gradient"
```

---

### 4.5 开发流程与最佳实践

#### 标准开发流程

```python
# Step 1: 识别优化机会
# - 通过 profiler 发现热点
# - 分析 graph 发现冗余

# Step 2: 编写 Pattern
@register_graph_pattern(
    CallFunction(target, ...),
    pass_dict=patterns,
    extra_check=optional_check,  # 额外条件检查
)
def my_optimization(match, ...):
    # 实现替换逻辑
    pass

# Step 3: 添加测试
class TestMyOptimization(TestCase):
    @torch._dynamo.config.patch("..."):
    def test_my_opt(self):
        # 测试正确性
        # 测试性能提升
        
# Step 4: 验证
# - 单测通过
# - 模型精度无损
# - 性能有提升

# Step 5: 注册到正确的 pass_dict
# - 早期优化 -> early_patterns
# - 主要优化 -> patterns / pass_patterns[0]
# - 后期优化 -> pass_patterns[1]
```

#### 最佳实践

**1. 安全检查优先**
```python
def safe_optimization(match, x):
    node = match.output_node()
    
    # 检查 dtype
    if node.meta["val"].dtype != torch.float32:
        return  # 跳过不安全的情况
    
    # 检查 shape
    if not statically_known_true(x.shape[0] > 0):
        return  # 无法确定时跳过
    
    # 执行优化...
```

**2. 使用 statically_known_true/false**
```python
from torch.fx.experimental.symbolic_shapes import statically_known_true

# 对于动态 shape，使用 guard 函数
if statically_known_true(x.shape[0] == y.shape[0]):
    # 确定相等时优化
    pass
```

**3. 保持元数据**
```python
def optimize(match, ...):
    with match.graph.inserting_before(node):
        new_node = match.graph.call_function(...)
        
        # 复制元数据！
        new_node.meta.update(node.meta)
        
        # 特别是 fake tensor 信息
        new_node.meta["val"] = compute_output_fake_tensor(...)
```

**4. 渐进式启用**
```python
# 使用 config 控制
if not config.my_new_optimization:
    return

# 或者使用环境变量
import os
if os.environ.get("ENABLE_MY_OPT") != "1":
    return
```

#### 调试技巧

```python
# 打印匹配信息
@register_graph_pattern(...)
def my_pattern(match, ...):
    print(f"Matched node: {match.nodes}")
    print(f"Args: {...}")
    # ...

# 条件断点
if node.meta["val"].shape[0] > 1000:
    import pdb; pdb.set_trace()

# Graph 可视化
def visualize_graph(graph, filename):
    with open(filename, 'w') as f:
        f.write(str(graph.graph))
        
# 比较优化前后
torch._inductor.config.joint_custom_pre_pass = lambda g: visualize_graph(g, "before.py")
torch._inductor.config.joint_custom_post_pass = lambda g: visualize_graph(g, "after.py")
```

---

### 4.6 推荐入手方向

| 优先级 | 方向 | 说明 | 难度 |
|-------|-----|------|-----|
| 1 | **View/Permute 清理** | 消除冗余的 shape 操作 | 1 |
| 2 | **Pointwise 融合** | 合并连续的 element-wise 操作 | 2 |
| 3 | **类型转换优化** | 清理 AMP 产生的冗余转换 | 2 |
| 4 | **特定算子融合** | 如 LayerNorm 分解后重新融合 | 3 |
| 5 | **内存布局优化** | 优化 stride/contiguous | 3 |
| 6 | **自定义融合 Kernel** | 编写 Triton 融合 kernel | 4 |

#### 快速开始示例

```python
# 文件: my_custom_passes.py
from torch._inductor.fx_passes.joint_graph import patterns
from torch._inductor.pattern_matcher import (
    register_graph_pattern, CallFunction, KeywordArg, Match
)
import torch

# 示例：优化 mul(pow(x, 2), y) -> square(x) * y（如果有 square 算子）
@register_graph_pattern(
    CallFunction(
        torch.ops.aten.mul.Tensor,
        CallFunction(
            torch.ops.aten.pow.Tensor_Scalar,
            KeywordArg("x"),
            2,  # 指数为 2
        ),
        KeywordArg("y"),
    ),
    pass_dict=patterns,
)
def optimize_pow2_mul(match: Match, x, y):
    """pow(x, 2) * y -> x * x * y（避免 pow 函数调用开销）"""
    node = match.output_node()
    
    with match.graph.inserting_before(node):
        # x * x 代替 pow(x, 2)
        x_squared = match.graph.call_function(
            torch.ops.aten.mul.Tensor,
            args=(x, x),
        )
        # 再乘 y
        result = match.graph.call_function(
            torch.ops.aten.mul.Tensor,
            args=(x_squared, y),
        )
        result.meta.update(node.meta)
    
    node.replace_all_uses_with(result)
    match.erase_nodes()

# 使用
import torch._inductor.config
torch._inductor.config.joint_custom_post_pass = lambda g: None  # 确保加载
```

---

## 总结

Joint Graph Passes 是 PyTorch Inductor 优化的核心阶段，通过系统化的图变换实现：
1. **正确性保证**：别名检查、元数据维护
2. **性能提升**：算子融合、冗余消除、分块计算
3. **可扩展性**：用户自定义 pass 机制

自定义开发 pass 的关键是：
- 理解 pattern matching 的工作原理
- 识别常见反模式和优化机会
- 严格验证正确性和性能收益
- 渐进式部署和测试
