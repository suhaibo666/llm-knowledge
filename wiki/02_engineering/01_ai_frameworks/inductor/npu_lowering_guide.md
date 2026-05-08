# torch_npu Inductor Lowering 完全解析

## 目录
1. [概述](#1-概述)
2. [与社区 Lowering 的架构对比](#2-与社区-lowering-的架构对比)
3. [核心机制详解](#3-核心机制详解)
4. [Op 分流策略](#4-op-分流策略)
5. [NPU 专有 Lowering 实现](#5-npu-专有-lowering-实现)
6. [NPU 专有 IR 节点](#6-npu-专有-ir-节点)
7. [配置体系](#7-配置体系)
8. [调试与扩展指南](#8-调试与扩展指南)

---

## 1. 概述

### 1.1 什么是 Lowering

Lowering（降级）是 TorchInductor 编译器中将 **ATen 算子**翻译为 **Inductor IR 节点**的过程。它是从 FX Graph 到代码生成之间的核心桥梁：

```
torch.compile / torch._dynamo
  ↓ (FX Graph with ATen ops)
Decomposition → Pre-grad passes
  ↓ (Simplified ATen ops)
GraphLowering.call_function()
  ↓
┌────────────────────────────────────┐
│  lowerings[target](*args, **kwargs)│ ← Lowering 核心
│  将 ATen op → IR Node              │
└────────────────────────────────────┘
  ↓ (IR Nodes: Pointwise, Reduction, ExternKernel, FallbackKernel...)
Post-grad passes → Scheduler → Codegen
```

### 1.2 torch_npu Lowering 的特殊性

torch_npu 不是重新实现一套 lowering 系统，而是**基于社区 lowering 做 monkey-patch 式的覆盖和扩展**。核心策略是：

1. **导入社区全部 lowerings**（继承社区注册表）
2. **大面积 fallback**（大部分 op 回退到 AclNN 算子库）
3. **选择性覆盖**（关键 op 用 NPU 专有实现替换）
4. **新增 NPU 专属算子**（如 `npu.npu_dtype_cast`）

```python
# torch_npu/_inductor/lowering.py 核心思路
from torch._inductor.lowering import lowerings, register_lowering, ...  # 继承社区

# Step 1: 大面积 fallback — 不在白名单的 op 全部走 AclNN
_register_npu_inductor_fallbacks()

# Step 2: 删除社区的部分 lowering
for op in LOWERING_OVERLOAD_OP:
    del lowerings[op]

# Step 3: 用 NPU 版本重新注册
@register_lowering(aten.mean)
def mean(x, axis=None, keepdim=False, *, dtype=None):
    ...  # NPU 专有实现
```

---

## 2. 与社区 Lowering 的架构对比

### 2.1 整体策略差异

| 维度 | 社区 (GPU/Triton) | torch_npu (NPU) |
|------|-------------------|-----------------|
| **默认行为** | 尽量走 codegen 生成 Triton kernel | 大部分走 fallback (AclNN) |
| **白名单** | 无（默认全部 codegen） | `GENERATE_LIST` ~94 个 op |
| **黑名单** | 极少数 fallback | `FALLBACK_LIST` ~1100+ 个 op |
| **覆盖机制** | 无需覆盖 | `LOWERING_OVERLOAD_OP` 删除后重注册 |
| **芯片感知** | 不区分 GPU 型号 | 根据 `get_soc_version()` 动态策略 |
| **间接内存** | 统一的 `indirect_indexing` | 多种模式（SIMT template/SIMT only/SIMD-SIMT mix） |
| **Welford 算法** | 支持（大 reduction） | 不支持（全部走 sum-based 两步法） |
| **cat 操作** | `ir.ConcatKernel`（统一） | NPU 自定义 `ConcatKernel` + 多种策略 |

### 2.2 数据流对比

```
社区 GPU 路径：
ATen op → lowerings[op] → Pointwise/Reduction/ExternKernel → Triton codegen

NPU 路径（白名单 op）：
ATen op → lowerings[op] → Pointwise/Reduction → NPU Triton codegen

NPU 路径（fallback op）：
ATen op → lowerings[op] → FallbackKernel → AclNN 算子库调用

NPU 路径（覆盖 op）：
ATen op → del lowerings[op] → NPU 重新注册 → NPU 专有 IR → NPU codegen
```

---

## 3. 核心机制详解

### 3.1 入口函数 `_register_npu_inductor_fallbacks`

这是 NPU lowering 的核心分流逻辑，决定了每个 op 走 codegen 还是 fallback。

```python
# lowering.py:L176-L218
def _register_npu_inductor_fallbacks():
    enable_fallback_list = os.environ.get('ENABLE_FALLBACK_LIST', '1')

    if get_soc_version() >= Ascend910_9391 and enable_fallback_list:
        # 策略 A（新芯片 910_9391+）：黑名单模式
        # FALLBACK_LIST 中的 op 走 fallback，其余允许 codegen
        for op in lowering.lowerings:
            if op in FALLBACK_LIST:
                make_fallback(op)
    else:
        # 策略 B（旧芯片）：白名单模式
        # 只有 GENERATE_LIST 中的 op 允许 codegen，其余全部 fallback
        for op in lowerings:
            if op not in decompositions and op not in gen_set:
                make_fallback(op)

    # 删除需要覆盖的 op
    for op in overload_op_set:
        if op in lowerings:
            del lowerings[op]
```

**关键理解**：`make_fallback(op)` 的作用是用 `fallback_handler` **替换** `lowerings[op]` 的值，使得该 op 不走 codegen，而是创建 `ir.FallbackKernel` 调用 AclNN 库。

### 3.2 NPU 自定义 `make_fallback`

```python
# lowering.py:L74-L95
def npu_make_fallback(op, layout_constraint=None, warn=True, override_decomp=False):
    if op in decompositions and not override_decomp:
        raise RuntimeError(f"both a fallback and a decomp for same op: {op}")

    def register_fallback(op_overload):
        add_needs_realized_inputs(op_overload)
        if layout_constraint is not None:
            add_layout_constraint(op_overload, layout_constraint)
        return register_lowering(op_overload, type_promotion_kind=None)(
            fallback_handler(op_overload)
        )

    if isinstance(op, torch._ops.OpOverloadPacket):
        for ol in op.overloads():
            register_fallback(getattr(op, ol))
    elif isinstance(op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
        register_fallback(op)

make_fallback = npu_make_fallback  # 替换社区的 make_fallback
```

与社区的区别：NPU 版本支持 `override_decomp` 参数，允许在有 decomposition 的情况下仍然 fallback（社区直接报错）。

### 3.3 Reduction 覆盖 — 附加元数据

```python
# lowering.py:L115-L159
def make_reduction(reduction_type: str, override_return_dtype=None):
    def inner(x, axis=None, keepdims=False, *, dtype=None):
        kwargs = _make_reduction_inner(x, axis=axis, keepdims=keepdims, ...)
        result = Reduction.create(reduction_type=reduction_type, input_node=x, **kwargs)

        if isinstance(result.data.data, Reduction):
            # ★ NPU 特有：附加 kept_idx 和 reduced_idx
            object.__setattr__(result.data.data, "kept_idx", kept_idx)
            object.__setattr__(result.data.data, "reduced_idx", reduced_idx)
            result.realize()
        return result

    return inner

lowering.make_reduction = make_reduction  # 替换社区的 make_reduction
```

**为什么需要这些额外属性**：
NPU 的 reduction kernel codegen 需要显式知道哪些维度被 reduce、哪些被保留，以正确生成 Ascend 向量/Cube 指令。社区的 Triton codegen 不需要这个信息，因为 Triton 内部自行处理维度映射。

---

## 4. Op 分流策略

### 4.1 三个核心列表

#### A. GENERATE_LIST（白名单 ~94 个 op）

允许 codegen 的 op，涵盖最常用的计算：

```python
# lowering_op_list.py
GENERATE_LIST = [
    # 基础 pointwise
    aten.add, aten.sub, aten.mul, aten.div, aten.neg,
    aten.exp, aten.log, aten.sqrt, aten.rsqrt, aten.abs,
    aten.cos, aten.sin, aten.tanh, aten.sigmoid, aten.relu,
    aten.pow, aten.reciprocal, aten.sign, aten.erf, aten.gelu,

    # 比较
    aten.lt, aten.gt, aten.le, aten.ge, aten.eq, aten.ne,

    # 逻辑
    aten.logical_and, aten.logical_or, aten.logical_not,

    # reduction
    aten.sum, aten.amax, aten.mean, aten.min, aten.max, aten.argmax, aten.argmin,

    # shape 操作
    aten.select, aten.unsqueeze, aten.squeeze, aten.reshape, aten.permute,
    aten.expand, aten.slice, aten.cat, aten.repeat, aten.clone, aten.copy, aten.copy_,

    # 特殊
    aten.where, aten.clamp, aten.clamp_min, aten.clamp_max,
    aten.arange, aten.full, aten.scalar_tensor,
    aten.isnan, aten.bitwise_and, aten.bitwise_xor,
    aten.select_scatter, aten.slice_scatter,

    # matmul（走 ExternKernel，非 codegen）
    aten.mm, aten.bmm, aten.addmm,

    # NPU 专有
    npu.npu_dtype_cast, npu._npu_dtype_cast, npu.npu_grouped_matmul,

    # higher order
    aten.native_layer_norm,
    triton_kernel_wrapper_mutation,
    torch.ops.higher_order.flex_attention,
    torch.ops.higher_order.flex_attention_backward,
    torch.ops.higher_order.invoke_subgraph,
]
```

#### B. FALLBACK_LIST（黑名单 ~1100+ 个 op）

强制走 AclNN fallback 的 op，包括：

| 类别 | 示例 | 原因 |
|------|------|------|
| 复杂数学函数 | `acos`, `acosh`, `asin`, `atan`, `cosh`, `sinh` | NPU codegen 暂不支持这些 libm 函数 |
| 池化类 | `avg_pool2d`, `max_pool2d`, `adaptive_avg_pool2d` | 需要专门的算子实现 |
| 卷积 | `convolution`, `convolution_backward` | 需要 Cube 单元专用算子 |
| 线性代数 | `linalg_*`, `cholesky`, `triangular_solve` | NPU 不支持 codegen |
| 随机数 | `rand`, `randn`, `bernoulli_` | 走 `config.fallback_random = True` |
| 排序/topk | `sort`, `topk`, `kthvalue` | 需要专用比较网络 |
| 量化 | `quantized_decomposed.*` | 走 AclNN 量化实现 |
| prims 基础 | `prims.add`, `prims.mul`, `prims.sin` 等全部 | prims 全部 fallback |
| 通信 | `_c10d_functional.*` | 走 HCCL 通信库 |
| inplace | `add_`, `mul_`, `relu_`, `sigmoid_` | inplace 全部 fallback |
| view 类 | `aten.view`, `aten.as_strided`, `aten.diagonal` | 部分 view 走 fallback |
| scatter/split | `aten.split`, `aten.unbind`, `aten.unfold` | 走 AclNN |

#### C. LOWERING_OVERLOAD_OP（需要覆盖的 op）

先从 `lowerings` 字典中删除社区注册，再用 NPU 版本重新注册：

```python
LOWERING_OVERLOAD_OP = [
    aten.cumsum,     # NPU 特殊 dtype 处理
    aten.mean,       # NPU 版 mean（无 Welford）
    aten.max, aten.min, aten.amin, aten.amax,  # NPU 版 reduction（附加元数据）
    aten.argmax, aten.argmin,
    aten.var_mean, aten.var,   # NPU 版 variance（只用 sum-based）
    aten.cat,        # NPU 版 concat（自定义 ConcatKernel）
    aten.mm, aten.bmm, aten.addmm,  # NPU matmul（走 catlass 模板）
    torch.ops.higher_order.flex_attention,
    torch.ops.higher_order.flex_attention_backward,
]
```

### 4.2 间接内存访问 op 的条件注册

根据配置 `inductor_indirect_memory_mode`，额外的 op 可以加入 codegen 白名单：

```python
# lowering_op_list.py:L124-L155
INDIRECT_MEM_GENERATE_LIST = [
    aten.embedding, aten.gather,
    aten.index_put_, aten.index_put, aten._unsafe_index_put,
    aten.scatter, aten.scatter_, aten.scatter_reduce, aten.scatter_reduce_,
    aten.index, aten._unsafe_index,
]

if inductor_indirect_memory_mode:
    GENERATE_LIST += INDIRECT_MEM_GENERATE_LIST
    if inductor_indirect_memory_mode == "simt_template":
        LOWERING_OVERLOAD_OP += INDIRECT_MEM_OVERLOAD_LIST
```

**四种间接内存模式**：

| 模式 | 环境变量值 | 行为 |
|------|----------|------|
| **关闭** | `None` / `fallback` | 间接内存 op 全部走 AclNN fallback |
| **simt_template** | `simt_template` | 使用 NPU SIMT Template kernel（最优） |
| **simt_only** | `simt_only` | 只使用 SIMT 核，cat 用 store 模式 |
| **simd_simt_mix** | 其他值 | SIMD+SIMT 混合模式 |

### 4.3 分流总结图

```
                          ATen op 进入
                              │
                    ┌─────────┴──────────┐
                    │  在 decompositions? │
                    └────┬──────────┬────┘
                         │Yes       │No
                         ↓          ↓
                    已被分解      ┌──────────────┐
                    不进 lowering │ LOWERING_     │
                                 │ OVERLOAD_OP?  │
                                 └──┬─────────┬──┘
                                    │Yes      │No
                                    ↓         ↓
                             删除社区 lowering  ┌──────────────┐
                             NPU 重新注册       │ 新芯片 9391+?│
                                                └──┬────────┬──┘
                                                   │Yes     │No
                                                   ↓        ↓
                                            ┌──────────┐  ┌──────────┐
                                            │FALLBACK  │  │GENERATE  │
                                            │_LIST 中? │  │_LIST 中? │
                                            └─┬─────┬──┘  └──┬────┬──┘
                                              │Yes  │No      │Yes │No
                                              ↓     ↓        ↓    ↓
                                          fallback codegen codegen fallback
                                          (AclNN)  (NPU)  (NPU)  (AclNN)
```

---

## 5. NPU 专有 Lowering 实现

### 5.1 mean — 与社区几乎相同

```python
# lowering.py:L250-L264
@register_lowering(aten.mean)
def mean(x, axis=None, keepdim=False, *, dtype=None):
    if dtype is not None:
        x = to_dtype(x, dtype)
    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    output_dtype = x.get_dtype()
    if output_dtype in (torch.float16, torch.bfloat16):
        x = to_dtype(x, torch.float)        # 同社区：低精度升到 fp32
    sum_result = sum_(x, axis, keepdim)
    denom = sympy_product(size[i] for i in axis)
    denom = ir.IndexingConstant(index=denom, dtype=x.get_dtype(), device=x.get_device())
    denom = ExpandView.create(denom, list(sum_result.get_size()))
    return to_dtype(div(sum_result, denom), output_dtype)
```

### 5.2 cumsum — 芯片版本感知

```python
# lowering.py:L266-L276
@register_lowering(aten.cumsum)
def cumsum(x, axis=None, dtype=None):
    if (is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())) and dtype is None:
        # ★ 旧芯片只支持 int32，新芯片支持 int64
        dtype = torch.int64 if get_soc_version() >= 250 else torch.int32
    if len(x.get_size()) == 0:
        # 标量 tensor 特殊处理
        if axis not in [0, -1]:
            raise ValueError("axis must be 0 or -1")
        dtype = dtype or x.get_dtype()
        return to_dtype(x, dtype, copy=True)
    return fallback_cumsum(x, dim=axis, dtype=dtype)
```

### 5.3 var_mean — 去掉 Welford，只用 sum-based

```python
# lowering.py:L796-L847
def var_mean_sum_(x, axis, correction, keepdim, return_mean):
    # 两步法：先算 mean，再算 (x - mean)^2 的 sum
    x_mean = mean(x, axis, keepdim=True)
    if return_mean:
        x_mean.realize()
    diffs = square(sub(x, x_mean))
    sum_result = sum_(diffs, axis, keepdim)
    # ...省略 correction 处理
    return x_var, x_mean

def var_mean_helper_(x, *, axis, correction, keepdim, return_mean):
    compute_dtype = get_computation_dtype(out_dtype)
    x = to_dtype(x, compute_dtype, copy=False)
    output = var_mean_sum_(**kwargs)  # ★ 永远走 sum 路径，不走 Welford
    output = tuple(to_dtype(x, out_dtype, copy=False) for x in output)
    return output[0] if not return_mean else output
```

**与社区的关键差异**：
| 维度 | 社区 GPU | torch_npu |
|------|---------|-----------|
| 小 reduction | `var_mean_sum_`（两步法） | `var_mean_sum_`（两步法） |
| 大 reduction | `var_mean_welford_`（单 pass） | `var_mean_sum_`（两步法） ← **差异** |
| 原因 | Triton 支持 Welford reduction | NPU Welford codegen 不成熟 |

### 5.4 native_layer_norm — 手动分解

社区通过 decomposition 在 lowering 之前分解 `native_layer_norm`。NPU 在 lowering 层直接实现：

```python
# lowering.py:L890-L945
@register_lowering(aten.native_layer_norm)
def native_layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    # ★ bf16/fp16 在新芯片上走 fallback（AclNN 性能更好）
    if get_soc_version() >= 250 and (x.dtype == torch.bfloat16 or x.dtype == torch.float16):
        return fallback_handler(aten.native_layer_norm.default)(x, normalized_shape, weight, bias, eps)

    # 手动分解为基础 op
    reduce_dims = list(range(len(input_shape) - normalized_ndim, len(input_shape)))
    var, mean = var_mean_helper_(x=x, axis=reduce_dims, correction=0, keepdim=True, return_mean=True)

    x_normalized = sub(x, mean)
    eps_tensor = ir.IndexingConstant(index=eps, ...)
    var_eps = add(var, eps_tensor)
    inv_std = rsqrt(var_eps)
    normalized = mul(x_normalized, inv_std)

    if weight is not None:
        normalized = mul(normalized, weight)
    if bias is not None:
        normalized = add(normalized, bias)

    return normalized, mean, inv_std
```

**好处**：NPU 控制整个计算流程，确保中间 op 都在白名单内（sub, mul, rsqrt, add），避免 decomposition 产生不适合 NPU 的中间节点。

### 5.5 embedding — SIMT Template 加速

```python
# lowering.py:L328-L355
@register_lowering(aten.embedding, type_promotion_kind=None)
def embedding(weight, indices, padding_idx=-1, ...):
    # 跳过标记
    if node.meta.get("skip_lowering", False):
        return fallback_handler(aten.embedding.default)(...)

    # 非 SIMT template 模式，走社区路径
    if inductor_indirect_memory_mode != str(NPUKernelType.SIMT_TEMPLATE):
        return lowering.embedding(weight, indices)

    # 判断是否适合用 template
    def should_use_template():
        if 1 in weight.get_size():      return False  # 退化情况
        if isinstance(weight.data, ir.BaseView): return False  # view 输入
        return True

    if should_use_template():
        return lowering_index_select(weight, 0, indices, 'embedding', ...)
    return lowering.embedding(weight, indices)
```

`lowering_index_select` 内部使用了 NPU 专有的 `ops.index_select()` 操作，生成 SIMT 模板 kernel，比通用的 `indirect_indexing` 在 NPU 上更高效。

### 5.6 cat — NPU 自定义 ConcatKernel

```python
# lowering.py:L357-L388
@register_lowering(aten.cat)
def cat(inputs, dim=0):
    if len(inputs) == 1:
        return clone(inputs[0])

    if lowering_cat_with_concat_kernel:  # 910_9391+ 默认开启
        # 检查是否有 reindex view 输入
        for inp in inputs:
            if is_reindex_view(inp):
                return TensorBox(npu_ir.ConcatKernel.create(inputs, dim, True))

        # 非最后一维 → 使用 NPU ConcatKernel
        if input_dims > 1 and (dim == -1 or dim == input_dims - 1):
            return TensorBox(npu_ir.ConcatKernel.create(inputs, dim, False))
        else:
            return fallback_handler(aten.cat.default)(inputs, dim)  # 其他情况 fallback
    else:
        # 旧芯片走社区路径
        return TensorBox(ir.ConcatKernel.create(inputs, dim))
```

**NPU ConcatKernel vs 社区 ConcatKernel**：
| 维度 | 社区 | NPU |
|------|------|-----|
| 实现 | `ir.ConcatKernel`（copy 每个 input 到 output slice） | `npu_ir.ConcatKernel`（分块 copy + insert_slice） |
| 分块 | 无 | `max_cat_size_in_per_kernel` 控制每个 kernel 处理的最大数据量 |
| 特殊模式 | 无 | `cat_store` / `cat_insert_slice` 两种策略 |
| reindex view | 统一处理 | 专门的 `single_realize_into` 路径 |

### 5.7 index_put — IndexputTemplate

```python
# lowering.py:L549-L602（简化）
def should_use_template():
    if accumulate: return False          # atomic 操作不走 template
    if x_ndim == 1 or 1 in x_size: return False   # 退化情况
    if len(valid_indices) != 1: return False        # 只支持单索引
    if isinstance(self.data, ir.BaseView): return False
    return True

if should_use_template():
    scatter = IndexputTemplate(
        inner_fn=values.make_loader(),
        output_indexer=inner_fn,
        boundary=boundary          # ★ NPU 特有：传入索引上界
    )
else:
    scatter = ir.Scatter(
        inner_fn=values.make_loader(),
        output_indexer=inner_fn,
        scatter_mode="atomic_add" if accumulate else None,
    )
```

`IndexputTemplate` 会生成 `ops.indexput_template()` 调用，NPU codegen 可以据此产生更高效的 SIMT scatter 指令。

---

## 6. NPU 专有 IR 节点

### 6.1 ConcatKernel

```python
# ir.py:L82-L199
class ConcatKernel(NopKernel):
    """NPU 专有的 cat 实现，支持分块 copy 和 insert_slice"""

    @classmethod
    def create(cls, inputs, dim, is_reindex):
        # 计算输出 size 和每个输入的 offset
        # ...
        if is_reindex:
            # 逐个 input realize 到 output 的 slice 中
            for i, inp in enumerate(inputs):
                input_buffer = cls.single_realize_into(inp, SliceView.create(kernel, dim, ...))
        else:
            # 分块 realize（max_cat_size_in_per_kernel 控制）
            for i, inp in enumerate(inputs):
                if offset > max_numel_in_per_kernel:
                    input_buffer = cls.realize_into(input_sub, SliceView.create(kernel, dim, ...))
```

### 6.2 IndexputTemplate

```python
# ir.py:L210-L236
class IndexputTemplate(Scatter):
    boundary: Optional[int] = None  # 索引上界，用于 bounds checking

    def store_output(self, output_name, indexer, store_vars):
        # 调用 NPU 专有的 ops.indexput_template
        return ops.indexput_template(
            output_name,
            indexer(output_indexer),
            loader(store_vars),
            indirect_indexer,
            self.boundary
        )
```

### 6.3 ScatterTemplate

```python
# ir.py:L239-L261
class ScatterTemplate(Scatter):
    def store_output(self, output_name, indexer, store_vars):
        # 调用 NPU 专有的 ops.scatter_template
        return ops.scatter_template(
            output_name,
            indexer(output_indexer),
            loader(store_vars),
            indirect_indexer,
            int(boundary),
        )
```

### 6.4 ConcatOutputKernel

```python
# ir.py:L202-L207
class ConcatOutputKernel(Pointwise):
    """自定义 store 逻辑的 Pointwise，用于 cat 的 store 模式"""
    def store_output(self, output_name, indexer, store_vars):
        loader = self.make_loader()
        loader(store_vars)  # 直接调用 loader，不走标准的 store 路径
        return None
```

---

## 7. 配置体系

### 7.1 芯片相关配置

```python
# config.py
Ascend910B1 = 220    # 910B1
Ascend310B1 = 240    # 310B1
Ascend910_9391 = 250 # 910_9391（最新）

# UB 大小（Unified Buffer）
ub_size = 192 * 1024                        # 默认 192KB
if get_soc_version() >= Ascend910_9391:
    ub_size = 256 * 1024                    # 新芯片 256KB

# Core 数量
num_cube_core = prop["num_aicore"]
num_vector_core = num_cube_core             # 默认 1:1
if Ascend910B1 <= soc < Ascend310B1 or soc >= Ascend910_9391:
    num_vector_core = num_cube_core * 2     # 这些芯片 vector core 是 cube 的 2 倍
```

### 7.2 Lowering 相关配置

| 配置 | 默认值 | 作用 |
|------|--------|------|
| `ENABLE_FALLBACK_LIST` | `'1'` | 启用黑名单模式（新芯片） |
| `inductor_indirect_memory_mode` | `None` (旧芯片) / 可配 (新芯片) | 间接内存 op 的 codegen 策略 |
| `lowering_cat_with_concat_kernel` | 新芯片 `True` | cat 使用 NPU ConcatKernel |
| `config.fallback_random` | `True` | 随机数 op 全部回退 eager |
| `config.allow_buffer_reuse` | `False` | 禁止 buffer 复用（避免 NPU 问题） |
| `enable_inplace_buffers` | `True`（可关闭） | inplace buffer 复用开关 |
| `dump_fx_graph` | `False`（可开启） | 调试模式：dump 每个 op 的 traced graph |
| `use_store_in_cat` | `False`（SIMT 模式下 `True`） | cat 使用 store 模式 |
| `max_cat_size_in_per_kernel` | `4096` (默认) / `1024` (SIMT) | 单个 cat kernel 最大数据量 |

### 7.3 环境变量速查

```bash
# Op 分流控制
ENABLE_FALLBACK_LIST=1          # 启用 FALLBACK_LIST 黑名单模式
INDUCTOR_INDIRECT_MEMORY_MODE=simt_template  # 间接内存模式

# 调试
INDUCTOR_ASCEND_DUMP_FX_GRAPH=1  # dump lowering 的 FX graph
INDUCTOR_ASCEND_CHECK_ACCURACY=1 # 精度检查模式
INDUCTOR_ASCEND_LOG_LEVEL=DEBUG  # 日志级别

# 性能
FASTAUTOTUNE=1                  # 快速 autotune
INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE=1  # 激进 autotune
USE_STORE_IN_CAT=1              # cat 使用 store 模式
TORCHINDUCTOR_NDDMA=1           # NDDMA 开关

# matmul
TORCHINDUCTOR_NPU_CATLASS_DIR=...  # CATLASS 库路径
TORCHINDUCTOR_CATLASS_ENABLED_OPS=mm,addmm,bmm  # 使用 CATLASS 的 op
```

---

## 8. 调试与扩展指南

### 8.1 查看 Op 分流结果

```python
import torch_npu
from torch._inductor.lowering import lowerings, fallbacks

# 查看有多少 op 在 lowerings 中
print(f"Total lowerings: {len(lowerings)}")

# 查看有多少 op 是 fallback
print(f"Total fallbacks: {len(fallbacks)}")

# 检查某个 op 的 lowering 是否是 fallback
op = torch.ops.aten.add.Tensor
handler = lowerings.get(op)
if handler and hasattr(handler, '_is_fallback_handler'):
    print(f"{op} → FallbackKernel (AclNN)")
else:
    print(f"{op} → Codegen")
```

### 8.2 添加新 Op 到 Codegen

```python
# Step 1: 在 lowering_op_list.py 的 GENERATE_LIST 中添加
GENERATE_LIST = [
    ...
    aten.my_new_op,  # 添加到白名单
]

# Step 2: 如果需要 NPU 特殊实现，在 lowering.py 中注册
@register_lowering(aten.my_new_op, type_promotion_kind=None)
def my_new_op(x, param):
    # NPU 专有的 lowering 实现
    def inner_fn(idx):
        return ops.my_custom_op(x.make_loader()(idx), param)

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=x.get_size(),
    )

# Step 3: 如果需要从覆盖列表中删除社区实现
LOWERING_OVERLOAD_OP = [
    ...
    aten.my_new_op,  # 删除社区 lowering 后用自己的
]
```

### 8.3 将 Op 从 Codegen 改为 Fallback

```python
# 方式 1：添加到 FALLBACK_LIST
FALLBACK_LIST = [
    ...
    aten.problematic_op,
]

# 方式 2：从 GENERATE_LIST 中移除
# （旧芯片白名单模式下，不在白名单的自动 fallback）

# 方式 3：运行时强制 fallback
# 在 lowering.py 中：
make_fallback(aten.problematic_op, override_decomp=True)
```

### 8.4 dump_fx_graph 调试模式

```bash
# 开启 FX Graph dump
export INDUCTOR_ASCEND_DUMP_FX_GRAPH=1

# 运行后，每个 lowering 的 op 都会记录对应的 traced graph
# 可以用于：
# 1. 分析某个 op 的 lowering 路径是否正确
# 2. 对比 codegen 和 fallback 的等价性
# 3. 检查 fusion 边界是否合理
```

开启后，`make_reduction` 和各 lowering 函数会额外记录 `traced_graph` 和 `node_name`：

```python
if npu_config.dump_fx_graph:
    result = Reduction.create(
        reduction_type=reduction_type,
        input_node=x,
        node_name=node_name,          # ★ 额外参数
        traced_graph=new_graph,        # ★ 额外参数
        **kwargs
    )
```

### 8.5 精度问题排查

```python
# 1. 开启精度检查
import torch_npu._inductor.config as npu_config
npu_config.check_accuracy = True

# 2. 默认精度容忍度
npu_config.acc_comp_tol = {
    torch.float32: {'rtol': 1.3e-6, 'atol': 1e-5},
    torch.float16: {'rtol': 1e-3,   'atol': 1e-5},
    torch.bfloat16: {'rtol': 1.6e-2, 'atol': 1e-5},
}

# 3. 强制某个 kernel fallback 到 FX Graph 执行
npu_config.force_fallback_kernel_id = [1, 2, 10]  # 指定 kernel id
# 或
npu_config.force_fallback_kernel_id = 'all'        # 全部 fallback
```

---

## 总结

### 核心设计哲学

torch_npu 的 lowering 策略可以概括为**"保守 codegen，激进 fallback"**：

1. **正确性第一**：不确定能正确 codegen 的 op，一律 fallback 到 AclNN
2. **逐步扩展**：通过白名单/黑名单控制 codegen 范围，新验证的 op 逐步加入
3. **芯片感知**：不同芯片型号走不同策略，充分利用硬件能力
4. **关键 op 深度优化**：对 embedding、gather、scatter、cat、layer_norm 等高频 op 做 NPU 专有的 template kernel

### 与社区的核心差距

| 方面 | 影响 | 改进方向 |
|------|------|---------|
| 大量 op fallback | 融合机会损失，kernel launch 多 | 逐步扩展 codegen 白名单 |
| 无 Welford 算法 | 大 reduction 的 variance 需要两遍读数据 | 实现 NPU Welford reduction codegen |
| inplace op 全 fallback | 优化器 step 性能受限 | 支持 inplace lowering |
| prims 全 fallback | decomposition 后的 op 无法 codegen | 逐步支持 prims codegen |
| view/as_strided fallback | 部分 zero-copy 机会损失 | 扩展 view 类 op 支持 |

## Related Pages

- [[02_engineering/01_ai_frameworks/index]]
- [[NPU_Inductor_Backend_Analysis]]
- [[lowering_analysis]]
