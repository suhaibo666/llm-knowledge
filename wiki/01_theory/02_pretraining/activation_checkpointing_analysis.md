# Activation Checkpointing（重计算）完整分析

> [!note] 页面角色与审计状态
> **页面角色**：用户侧 activation checkpoint、Megatron-LM 重入式 checkpoint 与训练策略专题；它保留框架/API、RNG 和分布式重计算细节，不等同于 AOTAutograd partitioner 自动选择的 min-cut recompute。
> **原始基线**：baseline-unknown（原页未固定 PyTorch 与 Megatron commit）；**当前审计基线**：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`。
> **审计状态**：已纳入历史 manifest，但 PyTorch/Megatron 两侧 locator、代码块和版本边界尚未逐结构单元闭合，不能据页名推定为当前实现全量验证。AOT saved-value ABI 与 partition-time recompute 见 [[19_torch_compile_end_to_end/10_saved_tensors_recompute_and_runtime_abi]]；预训练领域入口见 [[01_theory/02_pretraining/index]]。

> 从 PyTorch autograd 保存机制 → Megatron-LM CheckpointFunction 源码实现 → 选择性重计算策略 → view/cast/slice 的 ctx 特性 → 理论显存评估的完整链路分析。

---

## 1. 核心概念

**重计算（Activation Checkpointing / Recomputation）** 的本质：前向传播时不保存中间激活值，反向传播时从最近的 checkpoint 重新执行前向来生成所需激活。

```
无 Checkpointing:
  [Layer0 fwd] → [Layer1 fwd] → ... → [LayerN fwd] → [Loss]
   ↑存所有激活                                  ↑存所有激活
                                                 ↓
                                     [bwd] 所有激活仍在内存

有 Checkpointing (逐层 Full):
  [Layer0 fwd] → [Layer1 fwd] → ... → [LayerN fwd] → [Loss]
   ↑只存 input_0  ↑只存 input_1    ↑只存 input_N
                                      ↓
  [重跑 L0 fwd] ← [bwd 开始]
   临时激活仅 L0 层存在，用完释放
```

**时间换空间**：激活显存从 $O(L)$ 降到 $O(1)$（per layer），代价是 forward 多跑一次，约增加 1/3 计算量。

---

## 2. PyTorch Autograd 的激活值保存机制

### 2.1 `ctx.save_for_backward()` 是入口

```python
# 以 Linear 为例（简化自 PyTorch 源码）
class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        return input @ weight.T + bias

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_output @ weight          # 需要 weight
        grad_weight = grad_output.T @ input        # 需要 input ← 这就是"激活值"
        grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias
```

`ctx.save_for_backward()` 是反向需要前向数据的原因。**哪些 tensor 被保存决定了激活显存的大小。**

### 2.2 `torch.utils.checkpoint` 如何干预

```python
# torch.utils.checkpoint 的核心思想（简化）
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function
        with torch.no_grad():           # ← 关键：禁用 grad 跟踪
            outputs = run_function(*args)  # 内部 save_for_backward 全部失效
        ctx.save_for_backward(*args)       # 只保存入口输入
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        with torch.enable_grad():          # 重开 grad 跟踪
            outputs = ctx.run_function(*inputs)  # 重跑 forward
        torch.autograd.backward(outputs, grad_output)
        return (None,) + tuple(inp.grad for inp in inputs)
```

`torch.no_grad()` 使 `run_function` 内部的所有 `save_for_backward` 被跳过——中间激活不会被 autograd 图持有。**只保留入口输入**，反向时重跑。

### 2.3 ctx 保存的两种信息：Tensor vs 元信息

| 类型 | 存储方式 | 内存开销 | 示例 |
|------|---------|---------|------|
| **Tensor 激活值** | `ctx.save_for_backward(tensor)` | 大（与 batch/size 正比） | Linear 的 input、Softmax 的 output |
| **元信息** | `ctx.attribute = value`（Python 属性） | 可忽略（几十字节） | shape、dtype、stride、dims、slice 范围 |

**重计算只消除第一类（tensor 激活值）**。第二类元信息恒定可忽略，checkpoint 与否不影响。

---

## 3. View / Cast / Slice 算子的 ctx 存储机制

### 3.1 View 类算子：仅保存 shape 元信息

View 类（`view`、`reshape`、`transpose`、`permute`、`squeeze`、`unsqueeze`、`expand`）共享底层 storage，只改 strides/offset。**backward 仅需 shape/dims 元信息，不存储任何 tensor 数据。**

| 算子 | Backward | ctx 保存 | 存 tensor? |
|------|----------|---------|-----------|
| `view` | `grad.view(input_shape)` | `input_shape: tuple` | 否 |
| `transpose` | `grad.transpose(dim0, dim1)` | `(dim0, dim1)` | 否 |
| `permute` | `grad.permute(inverse_dims)` | `dims: tuple` | 否 |
| `squeeze`/`unsqueeze` | 逆操作 | dim 索引 | 否 |
| `expand` | `grad.sum(expanded_dims)` | `input_shape: tuple` | 否 |
| `chunk`/`unbind` | `cat`/`stack` | dim + sizes | 否 |

### 3.2 Cast 类算子：仅保存 dtype

`float()` / `half()` / `bfloat16()` 等类型转换的 backward 只需要 `input_dtype`——一个 Python 枚举值。

```python
# 伪代码
class CastBackward:
    def backward(grad_output, ctx):
        return grad_output.to(ctx.input_dtype)  # 仅依赖 dtype 元信息
```

AMP `autocast` 中的 cast 在 CUDA 上更是被融合为 dispatcher cast，不产生 autograd 节点。

### 3.3 Slice 类算子

- **简单整数 slice**（`x[2:10]`）：仅保存 `(input_shape, start, end, step)`——纯元信息。
- **高级 tensor 索引**（`x[index_tensor]`）：index_tensor 如果本身是 tensor，则需要保存，但通常比 input 小几个数量级。

### 3.4 View Chain 问题与 `make_viewless_tensor`

View 算子自身不存激活值，但有**间接内存引用问题**：

```
large_tensor (1GB storage)
  → x.view(-1)[0:100] → small_view (100MB 那部分, 但 _base → large_tensor)
    → Linear forward
      → ctx.save_for_backward(small_view)
        结果：large_tensor 的 1GB storage 全部保持存活！
```

**Megatron 的解决方案**——`megatron/core/utils.py:655-666`：

```python
def _kernel_make_viewless_tensor(inp, requires_grad):
    out = torch.empty((1,), dtype=inp.dtype, device=inp.device, requires_grad=requires_grad)
    out.data = inp.data  # 指向相同 storage，但 out 没有 _base 引用
    return out           # 切断 view chain：不再引用原始 large_tensor
```

在 `TransformerBlock.forward()` 每一层入口调用（`transformer_block.py:795`）：

```python
hidden_states = make_viewless_tensor(
    inp=hidden_states, requires_grad=True, keep_graph=True
)
```

### 3.5 对选择性重计算的影响

**view/cast/slice 算子不存储 tensor 激活值**，因此：
- checkpoint 它们 = 零收益（没有 tensor 数据可释放）
- Megatron 的 selective recomputation 配置项（`layernorm` / `core_attn` / `mlp` / `moe_act` / `mhc` / `mla_up_proj`）全部 targeting 真正存储 tensor 的算子

---

## 4. Megatron-LM 重计算实现：三层架构

```
┌─────────────────────────────────────────────────────────────┐
│  TransformerBlock._checkpointed_forward()                   │
│  调度层：uniform vs block 策略，决定哪些层用 checkpoint       │
│  文件：megatron/core/transformer/transformer_block.py        │
├─────────────────────────────────────────────────────────────┤
│  checkpoint() / te_checkpoint()                             │
│  入口层：选择合适的 checkpoint 实现（Megatron 自研 vs TE）     │
│  文件：random.py、transformer_engine.py                       │
├─────────────────────────────────────────────────────────────┤
│  CheckpointFunction / CheckpointWithoutOutput /             │
│  CheckpointWithoutOutputFunction                            │
│  执行层：autograd Function，真正实现重计算逻辑                 │
│  文件：megatron/core/tensor_parallel/random.py               │
└─────────────────────────────────────────────────────────────┘
```

---

### 4.1 第一层：`CheckpointFunction` — 核心 autograd 重计算

文件：`megatron/core/tensor_parallel/random.py:555`

从 `torch.utils.checkpoint` 改造，两个关键增强：

#### (a) RNG 状态管理扩展

```python
def _get_all_rng_states():
    return (torch.get_rng_state(),           # CPU RNG
            _get_cuda_rng_state(...),         # CUDA RNG
            get_cuda_rng_tracker().get_states())  # TP RNG Tracker ← 新增

def _set_all_rng_states(cpu, cuda, tp_tracker):
    torch.set_rng_state(cpu)
    _set_cuda_rng_state(cuda)
    get_cuda_rng_tracker().set_states(tp_tracker)
```

TP 场景下 dropout 等随机操作需要确定性——每个 rank 分别追踪自己的 RNG，重计算时精确恢复。

#### (b) 分布式激活值存储（`distribute_saved_activations`）

```python
# forward: 按 TP 维度切分 hidden_states，每 rank 只保存 1/tp_size
if distribute_saved_activations:
    ctx.input_0_shape = args[0].data.shape
    # 只有第 tp_rank 个分片被保留
    args[0] = split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True)

# backward: 重计算前 all-gather 回来
if ctx.distribute_saved_activations:
    inputs[0] = gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape)
```

**好处**：TP 切分下 checkpoint 保存的激活显存从 `N` 降到 `N / tp_size`。与之配合的是 `assert_viewless_tensor` 检查——确保在替换 `.data` 前 tensor 没有 view 引用。

#### (c) 全局标志 `IS_CHECKPOINTING`

`random.py:530-548`。Megatron 内部模块（如 fused kernels）检测到此标志后跳过不必要的中间结果保存——"已经有 checkpoint 在帮我省显存了，我自己不需要再存。"

#### (d) Forward/Backward 完整流程

```
Forward:
  1. _set_checkpointing()           ← 全局标记
  2. 保存 RNG 状态到 ctx
  3. with torch.no_grad():          ← 内部 save_for_backward 失效
        outputs = run_function(*args)
  4. 如果 distribute_saved: 切分 args[0] 为 1/tp_size
  5. ctx.save_for_backward(*args)   ← 只存入口输入
  6. _unset_checkpointing()
  7. return outputs                 ← 输出保有 grad 的 backward hook

Backward:
  1. 从 ctx 取出入口输入
  2. 如果 distribute_saved: all-gather 恢复完整 tensor
  3. 恢复 forward 前保存的 RNG 状态
  4. torch.enable_grad() 下重跑 run_function（激活临时生成）
  5. torch.autograd.backward(outputs, grad_args)
  6. 返回入口输入的各参数梯度
```

---

### 4.2 第二层：`CheckpointWithoutOutput` — 连输出也丢掉

文件：`megatron/core/tensor_parallel/random.py:800`

普通 `CheckpointFunction` 保留 checkpoint 函数的**输出**。但 selective recomputation 在 layer 内部 checkpoint 子模块时，子模块的输出会被后续模块也保存——如果不丢掉子模块自己的输出副本，相当于存了两份。

```python
class CheckpointWithoutOutput:
    def checkpoint(self, run_function, *args):
        # forward 时走 CheckpointWithoutOutputFunction
        outputs = CheckpointWithoutOutputFunction.apply(run_function, self, *args)
        self.outputs = outputs
        return outputs

    def discard_output_and_register_recompute(self, hook_tensor):
        # 物理丢弃 outputs 的 storage（resize_(0)）
        for output in self.outputs:
            output.untyped_storage().resize_(0)
        # 在 hook_tensor 的 grad hook 中注册 _recompute
        if hook_tensor.requires_grad:
            hook_tensor.register_hook(self._recompute)
```

**关键优化——Zero-copy storage sharing**（`random.py:903-910`）：

```python
# _recompute 末尾：用 C++ 扩展让丢弃的 output 重新指向重计算结果
for output, recomputation_output in zip(self.outputs, outputs):
    share_storage(output, recomputation_output)
```

不拷贝数据，不增 tensor version counter。后续模块反向时透明拿到重算后的正确数据——它们眼中的 output 从未被丢弃过。

#### `CheckpointWithoutOutputFunction`（`random.py:699`）

```python
@staticmethod
def forward(ctx, run_function, checkpoint_without_output_obj, *args):
    with torch.no_grad():  # 同普通 checkpoint，跳过内部 save_for_backward
        outputs = run_function(*args)
    _save_args_to_ctx(ctx, args)  # 支持 tensor + non-tensor 混合参数
    checkpoint_without_output_obj.ctx = ctx  # 桥接：让 CheckpointWithoutOutput 可访问 ctx
    return outputs
```

`_save_args_to_ctx` / `_load_args_from_ctx`（`random.py:647-696`）：支持 tensor 和 Python 非 tensor（int、float、None、bool 等）混合的参数列表，tensor 走 `save_for_backward`，非 tensor 存 `ctx._non_tensor_entries`。

#### `CheckpointManager`（`random.py:754`）

管理多个 `CheckpointWithoutOutput` 的链式重计算——主要用于 mHC 场景：

```python
# Forward 末尾统一操作
ckpt_manager.discard_all_outputs_and_register_unified_recompute(final_output)
# 只注册一个 backward hook，反向时串行 _recompute 所有被管理的子模块
```

---

### 4.3 第三层：调度层 — `TransformerBlock._checkpointed_forward`

文件：`megatron/core/transformer/transformer_block.py:445`

#### checkpoint_handler 的两条路径（line 521-547）

```python
def checkpoint_handler(forward_func):
    if self.config.fp8 or self.config.fp4:
        return te_checkpoint(forward_func, ...)        # Transformer Engine 实现
    else:
        return tensor_parallel.checkpoint(forward_func, ...)  # Megatron 自身实现
```

`te_checkpoint`（`transformer_engine.py:2595`）委托给 `transformer_engine.pytorch.distributed.checkpoint`——TE 针对 FP8 有专门的激活重计算优化。

#### Uniform 策略（`recompute_method='uniform'`）

```python
layer_idx = 0
while layer_idx < num_layers:
    chunk_end = min(layer_idx + recompute_num_layers, num_layers)
    hidden_states = checkpoint_handler(custom(layer_idx, chunk_end))
    layer_idx += recompute_num_layers
```

以 `recompute_num_layers` 为粒度做 checkpoint。每 N 层一个块，块内中间激活全部丢弃，只保留块的输入。反向时重算整个块。

```
[Layer0 + Layer1] → checkpoint → [Layer2 + Layer3] → checkpoint → ...
    ↑ 保留 input_0                ↑ 保留 input_2
    块内激活不保存                 块内激活不保存
```

#### Block 策略（`recompute_method='block'`）

```python
for layer_idx in range(num_layers):
    if layer_idx < recompute_num_layers:
        hidden_states = checkpoint_handler(custom(layer_idx, layer_idx + 1))  # 重计算
    else:
        hidden_states = custom(layer_idx, layer_idx + 1)(...)                 # 正常
```

只对前 `recompute_num_layers` 层做 checkpoint。利用了 PP first stage 剩余显存更多的特点。

---

### 4.4 Selective 子模块级别

文件：`megatron/core/transformer/transformer_layer.py:421-490`

Selective 模式下，**在 layer 内部**选择性 checkpoint 特定子模块：

| 配置项 | 目标 | 实现 | 选择依据 |
|--------|------|------|---------|
| `"layernorm"` | `input_layernorm` + `pre_mlp_layernorm` | `CheckpointWithoutOutput` | 重算 $O(b·s·h)$ 极轻，输出被 attention/MLP 保存 |
| `"core_attn"` | Attention（QKV + softmax + output proj） | `CheckpointFunction`（Block 级别通过标志控制） | 激活随 $s^2$ 增长，但 kernel 是 memory-bound |
| `"mlp"` | FFN（非 MoE） | `te_checkpoint` / `checkpoint` | gate/up 的中间层 $b·s·4h$ 占用大 |
| `"mla_up_proj"` | MLA 上投影 | `CheckpointWithoutOutput` | DeepSeek MLA 特有，重算成本低 |
| `"mhc"` | Hyper-Connection 多流 | `CheckpointManager` + `CheckpointWithoutOutput` | 多流扩展后中间状态 ×n 倍 |
| `"moe_act"` | MoE 专家激活函数 | `CheckpointWithoutOutput` | 每 token 只激活少数专家，但激活值 shape 仍可观 |

**以 `input_layernorm` 为例的具体流程**（`transformer_layer.py:587-638`）：

```python
# 前向：CheckpointWithoutOutput 包住 layernorm
if self.recompute_input_layernorm:
    input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
        apply_module(self.input_layernorm), hidden_states
    )
    # ... attention 正常执行 ...
    # 丢弃 layernorm 输出，注册重计算 hook 到 attention 输出上
    self.input_layernorm_checkpoint.discard_output_and_register_recompute(
        attention_output_with_bias[0]
    )
```

**为什么 target layernorm**：layernorm 计算 = 均值 + 方差 + normalize，纯 memory-bound，重算代价 $O(b·s·h)$ 可忽略。但它的输出作为 attention/MLP 的输入被 `ctx.save_for_backward` 保存——省掉这一份输入即省 $b·s·h$ 的显存。

---

## 5. Full vs Selective vs Offloading：理论对比

### 5.1 对比表

| | Full Recomputation | Selective Recomputation | Fine-Grained Offloading |
|---|---|---|---|
| **粒度** | 整个 Transformer Layer | Layer 内特定子模块 | 子模块级别 |
| **保留的 tensor** | 每层只保留 input hidden states | 部分子模块的输入保留 | 正常执行 + 异步搬到 CPU |
| **重算内容** | Layer 内所有中间激活 | 仅 layernorm/attention 等 | 反向时从 CPU 异步拉回 |
| **计算开销** | ~+33% | <10%（重算的都是 memory-bound kernel） | 几乎无（异步 overlap） |
| **显存节省** | 最大 | 中等（attention 约占 layer 40-60%） | 灵活（可按子模块粒度控制） |

### 5.2 选择性重计算的选择依据

Megatron 选择重计算某个模块的原则：

1. **显存占用大**：该模块的核心中间数据（如 attention 的 softmax 输出 $b·heads·s·s$）占显存多
2. **重计算开销低**：该模块的计算是 **memory-bound**（如 softmax、layernorm、dropout），重算不显著占 compute 时间
3. **不是 compute-bound**：FFN 的大矩阵乘法不重算（除非整层 full checkpoint），因为重算代价高

### 5.3 MFU 下降但显存不降的诊断

（`Exam.md` Q12 考点）：

1. **granularity 不合理**：`recompute_num_layers` 过小，forward bubble 无法被 compute overlap
2. **checkpoint 保留过多**：保留了不该保留的中间 buffer
3. **kernel 未 fuse**：用了 `torch.utils.checkpoint` 而非 Megatron 自定义的 `checkpoint`，产生 CPU launch overhead
4. **batch size 未调大**：显存省了但没扩大 batch，效率没提升

---

## 6. Decoder 模型的激活值依赖全景

一个标准 Decoder Layer 中哪些算子需要保存激活值，哪些不需要：

```
Input (hidden_states) [b, s, h]
  │
  ├─→ RMSNorm ──→ [ctx: input, 用于反向统计量梯度]  ← 需保存
  │
  ├─→ Q/K/V Proj ──→ [ctx: norm输出 = input]          ← 需保存
  │     │
  │     └─→ Attention (Q·K^T → softmax → ×V)
  │           ├─ softmax 输出 [ctx: attn_weights]      ← 需保存, [b, h, s, s] 大
  │           ├─ Q, K [FlashAttention 中片上融合不显式保存]
  │           └─→ Output Proj [ctx: V_out]             ← 需保存
  │
  ├─→ Residual Add ──→ [等价的 view 操作, 不存]       ← 不需保存
  │
  ├─→ RMSNorm ──→ [ctx: input]                        ← 需保存
  │
  └─→ FFN (Gate + Up → Activation → ×Down)
        ├─ Gate/Up 投影 [ctx: norm输出]               ← 需保存
        ├─ GELU/SiLU [ctx: input 或 output]           ← 需保存, [b, s, 4h]
        └─→ Down Proj [ctx: 激活输出]                  ← 需保存
```

### 显存占用排序（per layer，从大到小）

| 激活值 | 形状 | 大小占比 | 是否适合重计算 |
|--------|------|---------|--------------|
| FFN 中间层（Gate+Up 后 × 激活） | $b \times s \times 8h$ | ~8× | 视情况（compute-bound） |
| Attention softmax 输出 | $b \times heads \times s \times s$ | 取决于 $s^2$ | **是**（memory-bound） |
| Attention QKV 中间值 | $b \times s \times 3h$ | ~3× | **是**（memory-bound） |
| 各投影层输入 | $b \times s \times h$ | ~1× each | **是**（如果做逐层 full） |
| RMSNorm 输入 | $b \times s \times h$ | ~1× | **是**（重算极轻） |
| View/Cast/Slice | — | 0 | 不需要 |

---

## 7. 理论激活值开销估算

### 7.1 无 Checkpointing 的完整激活值

单层 Decoder 的激活值（不含参数和优化器状态）：

$$M_{act} \approx b \times s \times h \times L \times k$$

其中 $k$ 是每层激活值与 hidden size 的比例系数，Decoder 层典型值为 $k \approx 8\text{-}12$（attention 3× + FFN 8× + norms 1-2× ≈ 12-14，考虑 FlashAttention 中某些不显式保存，实际约 8-12）。

**以 7B 模型为例**（$b=4, s=4096, h=4096, L=32, k=10$）：

$$M_{act} \approx 4 \times 4096 \times 4096 \times 32 \times 10 \times 2\text{ bytes (bf16)} \approx 43\text{ GB}$$

### 7.2 Full Checkpointing 后

只保留每层输入（$k \approx 1$）：

$$M_{act} \approx 4 \times 4096 \times 4096 \times 32 \times 1 \times 2\text{ bytes} \approx 4.3\text{ GB}$$

约 **10× 降低**。

### 7.3 Selective 后（只重算 attention + norms）

$k \approx 6\text{-}7$（保留了 FFN 激活）：

$$M_{act} \approx \text{约 } 20\text{-}28\text{ GB}$$

---

## 8. 逐层 Checkpoint vs 整 Model Checkpoint

**必须逐层 checkpoint**，不能用整个 model 包一层 checkpoint：

```python
# ✅ 正确：逐层 checkpoint（Megatron 做法）
for layer in self.layers:
    hidden_states = checkpoint(layer, hidden_states)

# ❌ 错误：整 model 一层 checkpoint
output = checkpoint(self, input)  # backward 时重跑整个 model
```

整 model 一层的情况：backward 到第一层时重跑整个 forward → **所有层的中间激活同时存在于内存中** → 显存峰值 = 无 checkpointing 的峰值 + checkpoint 本身的开销 → **完全没有节省**。

逐层 checkpoint：backward 到第 N 层，只重算第 N 层的 forward，只有第 N 层的临时激活存在。显存分布：

```
Time →
  Fwd:  [L0] → [L1] → ... → [LN] → Loss
         ↓保存   ↓保存         ↓保存
        input0  input1       inputN
                                  ↓
  Bwd:  [重跑L0 fwd] ← [L0 bwd] ← [重跑L1 fwd] ← [L1 bwd] ← ...
        临时激活仅L0层          临时激活仅L1层
```

---

## 9. 完整对比：PyTorch 原生 vs Megatron 自研

| 特性 | `torch.utils.checkpoint` | Megatron `CheckpointFunction` | Megatron `CheckpointWithoutOutput` |
|------|-------------------------|-------------------------------|-----------------------------------|
| RNG 管理 | CPU + CUDA | CPU + CUDA + TP RNG Tracker | 同上 |
| TP 感知 | 无 | `distribute_saved_activations` 切分/聚合 | 间接使用 |
| 输出处理 | 保留 checkpoint 函数输出 | 保留输出 | **丢弃输出**（zero-copy storage sharing 恢复） |
| 非 tensor 参数 | 不支持 | `_save_args_to_ctx` 支持混合参数 | 同上 |
| 链式管理 | 无 | 无 | `CheckpointManager` 统一管理多个 checkpoint |
| FP8 兼容 | 不兼容 | 不兼容（此时用 `te_checkpoint`） | 支持 |

---

## Related Pages

- [[02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/00_pytorch_graph_series_index]] — 当前固定基线的图编译系统化课程入口
- [[01_theory/02_pretraining/index]] — 预训练领域索引
- [[19_torch_compile_end_to_end/10_saved_tensors_recompute_and_runtime_abi]] — AOTAutograd saved-value、min-cut partition 与 backward recompute 边界
- [[torchtitan_ac_analysis]] — torchtitan/PyTorch 工程侧:非重入 `checkpoint_wrapper` 票据机制、SAC dispatch 缓存回放、显存预估(与本文 Megatron `CheckpointFunction` 重入路径互补)
- [[megatron_recompute_analysis]] — Selective Recomputation、Fine-Grained Offloading、Checkpoint Resharding（原 Exam Q12/Q13/Q30 内容已并入此页）
- [[mHC]] — mHC 的选择性重计算实现与 CheckpointManager 的应用
- [[aotautograd_analysis]] — §10.1 激活检查点与重计算、§10.2 视图重放优化
- [[deepseek_v3_analysis]] — V3 的 RMSNorm + MLA up-projection 重计算
- [[deepseek_v4_analysis]] — V4 的 selective recomputation 与 SWA KV 策略
- [[deepseek_v4_cp_analysis]] — CP 场景的尾部 token 重计算策略
- [[torch_compile_npugraphs_deep_dive]] — NPU 侧 Joint Graph 中的重计算
- [[npugraphs_memory_reuse_analysis]] — NPU 内存池的 Checkpoint/Restore（同名异义，指 allocator 状态快照）
- [[llm_initiliaze_analysis]] — 内存管理与 meta device 初始化
