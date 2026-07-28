# PyTorch Inductor Post-Grad Passes 完全解析

> [!correction] 页面角色、审计状态与集中纠错（见 [[correction_report]]）
> **页面角色**：post-grad阶段目录、注册API与尾部不变量。
> **原始基线**：见下方`9922478dffa`；**当前审计基线**：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`。
> **课程分工**：本页保留阶段开发参考；当前改图不变量、pass管线与合法性见 [[19_torch_compile_end_to_end/12_fx_graph_editing_primitives_and_invariants]]、[[19_torch_compile_end_to_end/15_graph_pass_pipeline_ordering_and_fixpoint]] 和 [[19_torch_compile_end_to_end/16_graph_rewrite_legality_validation_and_complexity]]。

> **Updated**: 2026-07-22

> **Source baseline**: PyTorch `9922478dffa`，重点核验 `torch/_inductor/fx_passes/post_grad.py:84-89,144-446`、`torch/_inductor/custom_graph_pass.py:73-76,118-145`。
>
> **阶段结论**：Post-Grad 的输入已经 normalized/functionalized，并且前向、反向图已分开。这是后端 ATen 图融合的默认落点；但 layout/read-write 已成为核心条件的优化应放 Lowering/Scheduler，而不是继续堆 FX rewrite。

## 目录
1. [概述](#1-概述)
2. [Pass 详解](#2-pass-详解)
3. [执行顺序与依赖关系](#3-执行顺序与依赖关系)
4. [关键优化模式](#4-关键优化模式)
5. [自定义开发指南](#5-自定义开发指南)

---

## 1. 概述
> [!correction] P-017：本区段按固定基线纠错；现行结论见 [[19_torch_compile_end_to_end/11_graph_stage_boundaries_identity_and_provenance#2. 阶段地图]]，逐项说明见 [[correction_report]]。
### 1.1 什么是 Post-Grad Passes

Post-Grad Passes 是 PyTorch Inductor 编译器在**梯度计算之后**执行的一系列图级优化。与 Joint Graph Passes 不同，Post-Grad Passes 分别作用于：
- **前向图（Forward Graph）**：模型推理/前向传播计算
- **反向图（Backward Graph）**：自动微分生成的梯度计算

```python
# 执行时机
@torch.compile
def model(x, w):
    y = x @ w
    loss = y.sum()
    return loss

# AOT Autograd 流程：
# 1. 捕获前向图 + 反向图 -> Joint Graph
# 2. Joint Graph Passes 优化
# 3. 拆分为独立的前向图和反向图
# 4. 【本文重点】Post-Grad Passes 分别优化前向/反向图
# 5. Lowering -> Triton/C++ Kernel 代码生成
```

### 1.2 为什么需要 Post-Grad Passes

| 特性 | Joint Graph | Post-Grad |
|------|-------------|-----------|
| **处理对象** | 合并的前向+反向 | 独立的前向/反向图 |
| **IR 状态** | 原始 ATen 操作 | 已函数化（Functionalized）|
| **主要目标** | 跨前向/反向的优化 | 设备优化、内存优化、通信优化 |
| **关键优化** | 自动分块、算子融合 | 分布式通信、inplace 优化、后端特化 |

**核心作用**：
- **设备与内存优化**：将构造器移至 GPU、重新 inplace 化
- **分布式训练优化**：DDP/FSDP 通信融合、分桶
- **后端特化优化**：MKLDNN、自定义后端 passes
- **图清理与规范化**：DCE、noop 消除、拓扑排序

### 1.3 为什么在这里做，为什么不放相邻阶段

- **适合这里**：精确 ATen overload pattern、inference/backward 专用变换、设备/量化/通信图改写、把子图换成已有后端 op。
- **不放 Joint**：Joint 还不能把 inference-only、forward-only、backward-only 条件安全分开。
- **不放 Lowering**：若产物仍是 ATen 图，Post-Grad 更容易用 PatternMatcher 表达和测试；Lowering 应负责 ATen → Inductor IR。
- **应放 Scheduler/Codegen**：优化依赖完整 buffer 读写、融合组、stream、target ISA 或 wrapper ABI 时，FX 图信息不够。

### 1.4 固定基线的主流程与“为什么”

| 区段 | 主要 Pass | 为什么这样排序 |
|---|---|---|
| 前置清理 | DCE；inference locality reorder；`post_grad_custom_pre_pass` | 先缩图并给用户一个内建 pattern 前的插点 |
| CPU 专项 | grouped GEMM、WOQ int4 concat-linear | 只在 MKLDNN/对应配置满足时运行 |
| Pattern 核心 | profiler-op 清理；group-batch；noop/assert；`pass_patterns[0..2]`；配置化 `POST_GRAD_PATTERNS`；B2B GEMM | 先移除阻塞匹配的节点，再按三轮依赖逐步改写 |
| 并行专项 | micro-pipeline TP、DDP communication fusion | 需要函数化图和已稳定的 ATen 结构 |
| 用户后置 | `post_grad_custom_post_pass` | 第三方后端通常在内建 FX 融合后接入 |
| 设备/通信 | stable sort、constructor-to-GPU、设备 custom backend pass、collective decomposition/bucketing/overlap | 通信 bucketing 可能要求再次拓扑排序；all-gather bucketing 会引入 mutation，故靠后 |
| 最终 mutation | `reinplace_inplaceable_ops` 与 functional-wrapper 分解 | 源码明确要求会引入 mutation 的步骤保持最后 |

`post_grad_passes(gm, is_inference)` 在前向和反向各调用一次；`CustomInferenceAwareGraphPass` 能收到 `is_inference`，普通 `CustomGraphPass` 则只收到 `torch.fx.Graph`。

---

## 2. Pass 详解

> **说明**：以下 Pass 编号为示意，实际执行顺序以 `post_grad.py` 源码为准。

### Pass 1: FSDP2 参数引用清理

> [!deprecated]
> 该 FSDP2 专用 pass（`remove_fsdp2_unsharded_param_graph_input_usage`）不在固定基线 `9922478dffa` 的 `post_grad_passes()` 中。以下历史说明保留用于版本考古，不应加入当前执行顺序；当前 FSDP/collective 路径以 §1.4 的 dedup、bucketing、overlap 为准。

**代码位置**：`remove_fsdp2_unsharded_param_graph_input_usage`

**触发条件**：
```python
if not torch._dynamo.config.skip_fsdp_hooks:
    remove_fsdp2_unsharded_param_graph_input_usage(gm.graph)
```

**作用**：
清理 FSDP2（Fully Sharded Data Parallel 2）中对 **unsharded parameter** 的冗余图输入引用。

**FSDP2 背景**：
- **Sharded Param**：分片到各 GPU 的参数（每个 GPU 存一部分）
- **Unsharded Param**：通过 `All-Gather` 收集的完整参数
- 前向/反向传播后，这些 unsharded 参数引用可能残留在图中

**优化效果**：
- 减少内存占用
- 避免不必要的张量保持

---

### Pass 2: 死代码消除（DCE）
> [!correction] P-012、P-013、P-014：本区段按固定基线纠错；“`copy_` 因没有 users 就会被当前 FX DCE 误删”的例子不准确，当前 mutable-schema `OpOverload` 会被 impurity 检测保留；现行规则见 [[19_torch_compile_end_to_end/14_dead_code_topology_and_effect_order#2. FX DCE]] 与 [[19_torch_compile_end_to_end/14_dead_code_topology_and_effect_order#3. Pure + no users]]，逐项说明见 [[correction_report]]。
**代码位置**：`gm.graph.eliminate_dead_code()`

**触发条件**：
```python
if config.dce:
    # has some issues with mutation in inference mode
    gm.graph.eliminate_dead_code()
```

**作用**：
删除对输出无贡献的节点（无副作用且不被使用的节点）。

**注意事项**：
> "has some issues with mutation in inference mode"

在推理模式下，如果图包含 **in-place 操作**（如 `copy_`、`slice_scatter_`），DCE 可能误删：
```python
# DCE 可能错误删除这个节点
x.copy_(y)  # in-place 操作，x 之后没被使用但 mutation 已发生
```

---

### Pass 3: 局部性重排序（推理模式）
> [!correction] P-017：本区段按固定基线纠错；现行结论见 [[19_torch_compile_end_to_end/14_dead_code_topology_and_effect_order#8. 拓扑正确不等于effect正确]]，逐项说明见 [[correction_report]]。
**代码位置**：`reorder_for_locality`

**触发条件**：
```python
if is_inference and config.reorder_for_locality:
    reorder_for_locality(gm.graph)
```

**作用**：
将张量的**生产者节点**尽量移动到**消费者节点附近**，提高缓存局部性。

**实现机制**：
```python
def reorder_for_locality(graph):
    seen_nodes = set()
    for node in reversed(graph.nodes):
        seen_nodes.add(node)
        # 如果 node's producer 的所有用户都已被访问
        # 将 producer 移动到 node 之前
        torch.fx.map_arg((node.args, node.kwargs), visit)
```

**示例**：
```python
# 优化前
x = compute_a()  # 生产者
y = compute_b()
z = x + y        # 消费者
w = compute_c()  # 无关操作
result = z * 2   # 另一个消费者

# 优化后（compute_a 靠近消费者）
y = compute_b()
x = compute_a()  # 移动到使用位置附近
z = x + y
w = compute_c()
result = z * 2
```

**限制**：
- 不会跨越 `copy_` 操作（标记为 mutation epilogue）
- 不会重排序分布式通信操作（可能导致 hang）

---

### Pass 4: 用户自定义 Pre-Pass

**配置项**：`config.post_grad_custom_pre_pass`

**用途**：
- 插入调试代码
- 实验性优化
- 领域特定优化

---

### Pass 5: MKLDNN 融合优化

**触发条件**：`torch._C._has_mkldnn`

**包含优化**：

#### A. Grouped GEMM
```python
if (config.cpp.enable_grouped_gemm_template and 
    config.max_autotune and 
    "CPP" in config.max_autotune_gemm_backends):
    grouped_gemm_pass(gm.graph)
```

#### B. Concat Linear（Int4 权重量化）
```python
if config.cpp.enable_concat_linear:
    concat_linear_woq_int4(gm)
```

---

### Pass 6: Profiler 算子移除

**代码位置**：`_remove_profiler_ops`

**作用**：
移除 `record_function` 等 profiler 算子，防止它们阻塞算子融合。

```python
profiler_ops = [
    torch.ops.profiler._record_function_enter.default,
    torch.ops.profiler._record_function_enter_new.default,
    torch.ops.profiler._record_function_exit._RecordFunction,
]
```

**原因**：
- Profiler 算子是**有副作用**的
- 它们会阻止原本可以融合的操作被融合
- 在最终代码生成前移除，不影响性能分析功能

---

### Pass 7-10: 模式匹配优化（核心）
> [!correction] P-017、P-021：本区段按固定基线纠错；现行结论见 [[19_torch_compile_end_to_end/15_graph_pass_pipeline_ordering_and_fixpoint#4. Registration order]]，逐项说明见 [[correction_report]]。
#### A. 批处理融合
```python
group_batch_fusion_passes(gm.graph, pre_grad=False)
```

#### B. Noop 操作移除
```python
remove_noop_ops(gm.graph)
```

**处理的操作**：
| 操作 | 条件 | 替换为 |
|------|------|--------|
| `clone` / `alias` | 元数据相同 | 输入本身 |
| `slice` | 全范围切片 `[0:size:1]` | 输入本身 |
| `view` | shape 不变 | 输入本身 |
| `repeat` | 所有重复次数为 1 | 输入本身 |
| `cat` | 单输入 | 该输入 |
| `convert_element_type` | dtype 相同 | 输入本身 |
| `pow(x, 1)` | 指数为 1 | x |

**别名安全检查**（关键！）：
```python
# 不能引入新的输入-输出别名关系
if (not node_is_view and 
    node_storage in output_storages and 
    (src_storage in input_storages or src_storage in output_storages)):
    continue  # 跳过，避免破坏语义
```

#### C. Assert 算子移除
```python
remove_assert_ops(gm.graph)
```

移除 `aten._assert_tensor_metadata.default`，因为它：
1. 会被 lowering 为 no-op
2. 可能阻塞融合（如 `unfuse_bias_add_to_pointwise`）

#### D. 三轮 Pattern Matching
```python
pass_patterns = [PatternMatcherPass(), PatternMatcherPass(), PatternMatcherPass()]

for i, patterns in enumerate(pass_patterns):
    patterns.apply(gm.graph)
```

**分三轮的原因**：
1. **Pass 0**：基础模式（简单替换）
2. **Pass 1**：依赖基础清理后的模式（如 online softmax）
3. **Pass 2**：复杂融合（如 `mm_plus_mm`、`addmm` 融合判断）

---

### Pass 11: Scatter 分区优化

**触发条件**：`config.partitioned_scatter_enabled`

**代码位置**：`partitioned_scatter_optimization_pass`

**作用**：
将大规模 scatter 操作分区，减少 atomic 竞争。

---

### Pass 12: Back-to-Back GEMM

**触发条件**：`config.b2b_gemm_pass`

**代码位置**：`B2B_GEMM_PASS`

**作用**：
优化连续的矩阵乘法模式。

---

### Pass 13: 微管道 Tensor Parallel

**触发条件**：`config._micro_pipeline_tp`

**代码位置**：`micro_pipeline_tp_pass`

**作用**：
在张量并行场景中重叠通信与计算。

---

### Pass 14: DDP 通信融合

**触发条件**：`config._fuse_ddp_communication`

**代码位置**：`fuse_ddp_communication`

**作用**：
融合 DDP（DistributedDataParallel）中的梯度通信操作。

---

### Pass 15: 用户自定义 Post-Pass

**配置项**：`config.post_grad_custom_post_pass`

---

### Pass 16-17: 图规范化
> [!correction] P-015、P-017：本区段按固定基线纠错；现行结论见 [[19_torch_compile_end_to_end/14_dead_code_topology_and_effect_order#8. 拓扑正确不等于effect正确]]，逐项说明见 [[correction_report]]。
#### A. 稳定拓扑排序
```python
stable_topological_sort(gm.graph)
```

**作用**：
确保节点按确定性顺序排列，避免非确定性编译结果。

#### B. 构造器移至 GPU
```python
move_constructors_to_gpu(gm.graph)
```

**作用**：
将 CPU 上创建的构造器（`zeros`、`ones`、`full` 等）移至 GPU。

**场景**：
```python
# 原始代码
x = torch.zeros(1024, 1024, device='cpu')  # CPU 创建
y = x.cuda()                               # H2D 拷贝
z = y + cuda_tensor                        # GPU 计算

# 优化后
x = torch.zeros(1024, 1024, device='cuda') # 直接在 GPU 创建
z = x + cuda_tensor                        # 无 H2D 拷贝
```

**触发条件**：
```python
allow_inputs_outputs = bool(
    torch._inductor.config.triton.cudagraphs and 
    torch._inductor.config.graph_partition
)
```

---

### Pass 18: 自定义后端 Passes

**代码位置**：`custom_backend_passes`

**作用**：
支持第三方后端（非 CUDA/XPU）的自定义优化。

---

### Pass 19-21: 集合通信分桶（Bucketing）

**包含三个 passes**：
- `bucket_reduce_scatters`
- `bucket_all_reduces`
- `bucket_all_gathers`

**作用**：
将多个小的集合通信操作合并成大的桶操作，减少同步开销。

**示例**：
```python
# 优化前（3 次小 All-Gather）
ag1 = all_gather(param1)
ag2 = all_gather(param2)
ag3 = all_gather(param3)

# 优化后（1 次大 All-Gather）
bucketed_ag = all_gather_bucket([param1, param2, param3])
ag1, ag2, ag3 = split_bucket(bucketed_ag)
```

**注意事项**：
- `bucket_all_gathers` 放在最后，因为它引入 mutation
- 分桶后需要重新拓扑排序

---

### Pass 22: 重叠调度

**触发条件**：`config.aten_distributed_optimizations.enable_overlap_scheduling`

**作用**：
在分布式训练中调度通信与计算的重叠执行。

---

### Pass 23-27: Mutation 引入（最后阶段）

> "Keep these last, since they introduce mutation"

这些 passes 会引入 **in-place 操作**，破坏函数化不变性，因此放在最后：

#### A. 重新 Inplace 化
```python
reinplace_inplaceable_ops(gm.graph)
```

**作用**：
将函数化的操作重新转换为 in-place 形式，节省内存。

**示例**：
```python
# 函数化表示
new_x = aten.add(x, y)  # 创建新张量

# 重新 inplace 化（如果安全）
x.add_(y)  # 原地修改，无新分配
```

#### B. Triton Kernel 分解
```python
decompose_triton_kernel_wrapper_functional(gm.graph)
```

#### C. Auto-Functionalized 分解
```python
decompose_auto_functionalized(gm.graph)
```

#### D. FSDP All-Gather 重新 Inplace 化
```python
reinplace_fsdp_all_gather(gm.graph)
```

#### E. 高阶算子分解
```python
decompose_scan_to_while_loop(gm.graph)
decompose_map_to_while_loop(gm.graph)
```

---

## 3. 执行顺序与依赖关系
> [!correction] P-017：本区段按固定基线纠错；现行结论见 [[19_torch_compile_end_to_end/15_graph_pass_pipeline_ordering_and_fixpoint#2. 三个Inductor FX driver不同]]，逐项说明见 [[correction_report]]。
```
Stage 1: FSDP & 基础清理
─────────────────────────────────
1. remove_fsdp2_unsharded_param_graph_input_usage
2. eliminate_dead_code (if config.dce)

Stage 2: 推理优化
─────────────────────────────────
3. reorder_for_locality (仅推理模式)

Stage 3: 用户自定义 & 后端特化
─────────────────────────────────
4. post_grad_custom_pre_pass
5. MKLDNN passes (grouped_gemm, concat_linear)

Stage 4: 图清理 & 模式匹配
─────────────────────────────────
6. remove_profiler_ops
7. group_batch_fusion_passes
8. remove_noop_ops
9. remove_assert_ops
10. pass_patterns[0] - 基础模式
11. partitioned_scatter_optimization (if enabled)
12. post_grad_fusion_options - 扩展模式
13. B2B_GEMM_PASS (if enabled)

Stage 5: 分布式优化
─────────────────────────────────
14. micro_pipeline_tp_pass
15. fuse_ddp_communication
16. post_grad_custom_post_pass

Stage 6: 图规范化
─────────────────────────────────
17. stable_topological_sort
18. move_constructors_to_gpu
19. custom_backend_passes

Stage 7: 集合通信分桶
─────────────────────────────────
20. bucket_reduce_scatters
21. bucket_all_reduces
22. bucket_all_gathers
23. stable_topological_sort (if collectives_bucketing)
24. overlap_scheduling (if enabled)

Stage 8: Mutation 引入（必须在最后）
─────────────────────────────────
25. reinplace_inplaceable_ops
26. decompose_triton_kernel_wrapper_functional
27. decompose_auto_functionalized
28. reinplace_fsdp_all_gather
29. decompose_scan_to_while_loop
30. decompose_map_to_while_loop

最后: recompile + lint
```

**关键依赖关系**：
- `remove_profiler_ops` -> `pass_patterns`（profiler 阻塞融合）
- `stable_topological_sort` -> `bucketing` 之后（分桶可能破坏拓扑序）
- `reinplace_*` passes -> 必须在最后（引入 mutation）

---

## 4. 关键优化模式

### 4.1 GEMM 融合模式

#### A. MM + MM -> MM_Plus_MM
```python
# 模式
add(mm(mat1, mat2), mm(mat3, mat4))

# 条件
# - 所有矩阵维度匹配
# - max_autotune 启用

# 替换为
mm_plus_mm(mat1, mat2, mat3, mat4)  # 融合 kernel
```

#### B. Add + MM -> Addmm（或反向优化）

**融合方向**（默认）：
```python
# 模式
add(mm(mat1, mat2), inp)

# 条件
# - 形状可广播
# - dtype 相同
# - 输出用户不是 pointwise（否则反向优化）

# 替换为
addmm(inp, mat1, mat2)
```

**反向展开**（特定条件下更优）：
```python
# 当输出用户全是 pointwise 操作时
addmm(inp, mat1, mat2)
# 展开为
mm_result = mat1 @ mat2
if alpha != 1: mm_result = alpha * mm_result
if beta != 1: inp = beta * inp
return inp + mm_result
```

### 4.2 Softmax 数值稳定性

```python
# 原始模式（数值不稳定）
xmax = x.amax(dim=dim, keepdim=True)
xsub = x - xmax
xexp = xsub.exp()
xsum = xexp.sum(dim=dim, keepdim=True)

# 优化（使用 online softmax）
xmax, xsum = prepare_softmax_online(x, dim)
xsub = x - xmax
```

### 4.3 冗余 View/Permute 消除

```python
# view(view(x, s1), s2) -> view(x, s2) 如果 s2 == x.shape
# permute(permute(x, p1), p2) -> x 如果 p1 和 p2 互为逆排列
```

### 4.4 Scatter 优化

```python
# 在 full 张量上的 scatter
result = torch.full((B, C), 0.01)
result.scatter_(1, indices, 0.9)

# 优化为 where（向量化，无 atomic）
mask = (arange(C) == indices.unsqueeze(1))
result = torch.where(mask, 0.9, 0.01)
```

---

## 5. 自定义开发指南

### 5.0 关键 API 速查

| API/对象 | 作用 | 适用位置 |
|---|---|---|
| `config.post_grad_custom_pre_pass` | 内建 group/noop/pattern 前插入 | 给内建规则铺路、观测初始切分图 |
| `config.post_grad_custom_post_pass` | 内建 FX pattern 与通信前段后插入 | 第三方后端融合的常用落点 |
| `CustomInferenceAwareGraphPass` | `Graph, is_inference -> None` | 明确区分 inference 与训练图 |
| `pass_patterns[0..2]` | 三轮主 PatternMatcher | 用轮次表达 pattern 依赖 |
| `POST_GRAD_PATTERNS` + `post_grad_fusion_options` | 名字化的配置 pattern | 内部注册/配置表 |
| `register_graph_pattern` | ATen 子图 → ATen 图改写 | 仍由后续 lowering 处理 |
| `post_grad.register_lowering_pattern` | ATen pattern → Inductor IR handler | lowering-time fusion，默认 `pass_number=1` |
| `FakeTensorUpdater` | Pass 后增量维护 fake tensor meta | constructor 移动、reinplace 等依赖 |
| `stable_topological_sort` | 确定性拓扑与通信改写后修序 | 不等同于语义正确性证明 |
| `reinplace_inplaceable_ops` | 最终恢复安全的 inplace | 会引入 mutation，必须靠后 |

### 5.1 注册新 Pattern

#### 方式一：Lowering Pattern（生成 Inductor IR）

```python
from torch._inductor.fx_passes.post_grad import register_lowering_pattern
from torch._inductor.pattern_matcher import CallFunction, KeywordArg, Match

@register_lowering_pattern(
    CallFunction(
        torch.ops.aten.add,
        CallFunction(torch.ops.aten.mm, KeywordArg("mat1"), KeywordArg("mat2")),
        CallFunction(torch.ops.aten.mm, KeywordArg("mat3"), KeywordArg("mat4")),
    ),
    extra_check=is_valid_mm_plus_mm,  # 额外条件检查
)
def mm_plus_mm_replacement(match: Match, mat1, mat2, mat3, mat4):
    """融合两个矩阵乘法"""
    from torch._inductor.kernel.mm_plus_mm import tuned_mm_plus_mm
    return tuned_mm_plus_mm(mat1, mat2, mat3, mat4)
```

#### 方式二：Graph Pattern（图变换）

```python
from torch._inductor.pattern_matcher import (
    register_graph_pattern, CallFunction, KeywordArg, Match
)
from torch._inductor.fx_passes.post_grad import pass_patterns

@register_graph_pattern(
    CallFunction(
        torch.ops.aten.view.default,
        KeywordArg("arg"),
        KeywordArg("size"),
    ),
    pass_dict=pass_patterns[0],  # 指定在哪一轮执行
)
def remove_pointless_view(match: Match, arg, size):
    """如果 view 不改变 shape，直接替换为输入"""
    node = match.output_node()
    arg_val = arg.meta.get("val")
    
    if arg_val is None:
        return
    
    from torch.fx.experimental.symbolic_shapes import statically_known_true
    
    # 检查 shape 是否相同
    if statically_known_true(sym_eq(arg_val.shape, tuple(size))):
        node.replace_all_uses_with(arg)
        match.erase_nodes()
```

### 5.2 编写条件检查函数

```python
def is_valid_my_pattern(match: Match) -> bool:
    """检查 pattern 是否可以安全应用"""
    # 1. 检查设备类型
    device = match.kwargs["x"].meta["val"].device.type
    if device not in ["cuda", "xpu"]:
        return False
    
    # 2. 检查 dtype
    dtype = match.kwargs["x"].meta["val"].dtype
    if dtype != torch.float32:
        return False
    
    # 3. 检查 shape（支持动态 shape）
    shape = match.kwargs["x"].meta["val"].shape
    from torch.fx.experimental.symbolic_shapes import statically_known_true
    if not statically_known_true(shape[0] >= 128):
        return False  # 太小不值得优化
    
    return True
```

### 5.3 添加自定义 Graph Pass

```python
import torch
from torch._inductor import config
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass

def my_custom_pass(graph: torch.fx.Graph) -> None:
    """普通 hook：接收 Graph，原地修改，返回 None。"""
    for node in graph.nodes:
        if should_optimize(node):
            # 复杂的多节点变换
            pass
    graph.lint()

with config.patch(post_grad_custom_post_pass=my_custom_pass):
    compiled = torch.compile(model)

class InferenceOnlyPass(CustomInferenceAwareGraphPass):
    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if is_inference:
            graph.lint()  # 在这里执行 inference-only rewrite

    def uuid(self):
        return b"inference-only-pass-v1"

with config.patch(post_grad_custom_post_pass=InferenceOnlyPass()):
    compiled = torch.compile(model)
```

选择 pre 还是 post hook 的方法：规则要先给内建 PatternMatcher 铺路时用 `post_grad_custom_pre_pass`；规则依赖内建规范化/融合结果、或要把最终 ATen 子图换成后端 op 时用 `post_grad_custom_post_pass`。

### 5.4 调试技巧

```python
# 打印图状态
from torch._logging import trace_structured

trace_structured(
    "artifact",
    metadata_fn=lambda: {
        "name": "post_grad_after_my_pass",
        "encoding": "string",
    },
    payload_fn=lambda: gm.print_readable(
        print_output=False, include_stride=True, include_device=True
    ),
)

# 条件断点
if node.meta["val"].shape[0] > 1000:
    import pdb; pdb.set_trace()

# 导出图对比
torch._inductor.config.post_grad_custom_pre_pass = \
    lambda g: open("before.py", "w").write(str(g))
torch._inductor.config.post_grad_custom_post_pass = \
    lambda g: open("after.py", "w").write(str(g))
```

### 5.5 最佳实践

**1. 安全第一**
```python
# 不确定时跳过，不要破坏正确性
if not can_prove_safe(node):
    return  # 跳过此优化
```

**2. 保留元数据**
```python
with match.graph.inserting_before(node):
    new_node = match.graph.call_function(...)
    new_node.meta.update(node.meta)  # 复制元数据！
```

**3. 使用 statically_known_true**
```python
# 对于动态 shape，使用 guard 函数
from torch.fx.experimental.symbolic_shapes import statically_known_true

if statically_known_true(x.shape[0] == y.shape[0]):
    # 确定相等时优化
    pass
```

**4. 渐进式启用**
```python
# 使用 config 控制新优化
if not config.my_new_optimization:
    return
```

---

## 总结

Post-Grad Passes 是 PyTorch Inductor 编译器的关键阶段，专注于：

1. **设备与内存优化**：构造器移动、inplace 化
2. **分布式训练优化**：DDP/FSDP 通信融合、分桶
3. **算子融合**：GEMM 融合、Pointwise 融合
4. **图清理**：DCE、noop 消除、拓扑排序

与 Joint Graph Passes 的核心区别：
- **Post-Grad** 关注设备后端、内存布局、分布式通信
- **Joint Graph** 关注前向/反向联合优化、自动分块

两者配合，形成完整的 PyTorch 2.0 编译优化流水线。

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]] — 当前固定基线的图编译系统化课程入口
- [[02_engineering/01_ai_frameworks/index]]
- [[fx_pass_optimization_methodology]] — 八阶段放置方法论
- [[lowering_analysis]] — Post-Grad 之后的 ATen → IR 边界
- [[pre_grad_passes_guide]]
- [[joint_graph_passes_guide]]
- [[inductor_compiler_pipeline_analysis]] — 端到端编译管线全景（本文 §4.3 Post-grad Passes）
