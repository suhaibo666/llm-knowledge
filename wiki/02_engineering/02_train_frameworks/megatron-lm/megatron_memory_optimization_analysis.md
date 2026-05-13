# Megatron-LM 显存优化 全景分析

**Date**: 2026-05-12
**Status**: Complete
**Source**: `megatron/core/nccl_allocator.py`, `moe/paged_stash.py`, `cpu_offloading/`, `pipeline_parallel/fine_grained_activation_offload.py`, `fp8_utils.py`, `fp4_utils.py`

## 1. 显存瓶颈全景

大规模 MoE 训练中，显存消耗可分为五类：

| 类别 | 占用比例（典型） | 优化手段 |
|------|-----------------|---------|
| **模型参数** | 1x（BF16） | FP8/FP4 量化、Sequence Parallelism |
| **梯度** | 1x（BF16） | 分布式优化器（Reduce-Scatter 替代 All-Reduce）、Buffer 复用 |
| **优化器状态** | 2-3x（FP32） | 分布式优化器分片、CPU Offloading、FP8 状态 |
| **激活值** | 5-10x（与 seq_len² 相关） | Activation Checkpointing、Activation Offloading、Paged Stash（MoE） |
| **临时 Buffer** | 1-3x | NCCL Memory Pool、CUDA Graph Buffer 复用、融合算子 |

Megatron-LM 提供了覆盖所有五类的完整优化工具箱。

## 2. 各显存优化技术详解

### 2.1 NCCL Memory Pool (`nccl_allocator.py`)

**解决什么问题**：通信 buffer 的显存碎片化和 NCCL 对称内存注册开销。

**机制**（`nccl_allocator.py:51-58`）：使用 `ncclMemAlloc` / `ncclMemFree` 替代 PyTorch 默认分配器：

```python
nccl_mem = torch.cuda.MemPool(ncclMemAlloc, ncclMemFree)
```

**关键优化**：
- `MultiGroupMemPoolAllocator`（`:276`）：同一 pool 被 FSDP 和 EP 多组共享，避免为不同通信组分配重复 buffer
- `NCCL_NVLS_ENABLE=1`（`:154`）：启用 NVLink SHARP 硬件 offload，减少通信延迟
- Warmup barrier 预先建立 NCCL 通信 buffer，避免首次通信时分配

**适用条件**：多机训练（NCCL 通信密集）、FSDP/EP 混合使用

### 2.2 MoE Paged Stash (`moe/paged_stash.py`)

**解决什么问题**：MoE 专家层的激活值显存（每个 expert 需要独立存储激活用于 backward）。

**机制**：基于分页的激活管理，类似 OS 的虚拟内存：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PagedStash    │────→│   CUDA Buffer   │ (Tier 1: GPU pages)
│     Manager     │     │   1.10x peak    │
└────────┬────────┘     └─────────────────┘
         │              ┌─────────────────┐
         └──────────────→│ CPU Buffer      │ (Tier 2: Pinned host)
                         │   optional      │
                         └─────────────────┘
                         ┌─────────────────┐
                         │   Overflow      │ (Tier 3: Rerun)
                         │   fallback      │
                         └─────────────────┘
```

**生命周期**（`:587`）：
1. **begin**：第 1 次迭代，idle 观察
2. **capture**：第 2 次迭代，记录 max tokens per (dtype, hidden_size)，建立 page schedule
3. **captured**：使用 CUDA Graph 加速 stash/reload

**三级溢出处理**（Triton kernel `:129`）：
- Tier 1（快速路径）：页在 CUDA stash buffer 内
- Tier 2（CPU 溢出）：CUDA buffer 满但 pinned host buffer 可用
- Tier 3（重运行）：都满了 → 触发 `PagedStashRunner.rerun()` 回退到无 stash 重新执行

**配置参数**：
```python
moe_paged_stash_page_size: int = 64           # 页面大小
moe_paged_stash_buffer_size_factor_cuda: float = 1.10  # GPU buffer 比例
moe_paged_stash_buffer_size_factor_cpu: float = 0.0    # CPU buffer 比例
```

**适用条件**：MoE 训练（expert 数量 ≥ 8，capacity factor 大时收益最高）。与 `cpu_offloading` 不兼容。

### 2.3 Fine-Grained Activation Offloading (`fine_grained_activation_offload.py`)

**解决什么问题**：一般 Transformer 层的激活值显存（非 MoE 特定）。

**机制**（`:384`）：`PipelineOffloadManager` 在 sub-layer 粒度选择性卸载：

```
支持的 offload 模块（`:1141`）：attn_norm, qkv_linear, core_attn, attn_proj, 
                          mlp_norm, expert_fc1, moe_act
```

**关键设计**：

1. **OffloadTensorPool**（`:99`）：基于 deque 的 pinned CPU memory pool
   - Match by (shape, dtype) 复用，避免重复分配 CPU 内存
   
2. **选择性卸载**：
   - `min_offloaded_tensor_size`（`:928`）：跳过小 tensor（默认 ≥1M elements）
   - `activation_offload_fraction`（`:667`）：只卸载部分层
   - `offload_margin`（`:557-573`）：最后 N 层不卸载（避免 reload 阻塞 backward 计算）

3. **CUDA Graph 兼容**（`:1247-1267`）：支持延迟卸载（`delay_offload_until_cuda_graph`）

**性能开销**：D2H/H2D 异步传输（pinned memory + 独立 CUDA stream），与计算重叠

**适用条件**：大模型（≥70B）、长序列、PP 并行。不能与 `cpu_offloading` 组合。

### 2.4 Optimizer State Offloading (`optimizer_state_offloader.py`)

**解决什么问题**：优化器状态（exp_avg, exp_avg_sq）和 FP32 主权重的 GPU 显存。

**机制**（`:253`）：
```
GPU: optimizer.step()
  → offload(): D2H 异步拷贝 state → CPU (pinned)
    → release_gpu_memory(): GPU storage.resize_(0)
      ... [大量 forward/backward 运算] ...
        → reload(): H2D 异步拷回 state → GPU
          → sync_before_step(): 等待 H2D 完成
            → optimizer.step()
```

**显存节省**：Adam 状态下节省 2x param（exp_avg + exp_avg_sq）+ 可选 1x (master weights)

**适用条件**：100B+ 参数、优化器状态成为显存瓶颈。不兼容 `overlap_param_gather`（MXFP8 模式）。

### 2.5 Parameter And Gradient Buffer 复用

`param_and_grad_buffer.py` 中的多重复用：

**MXFP8 共享 Buffer**（`:1097-1113`）：
```
param all-gather ─────┐
                      ├── shared_buffer（同一物理显存）
grad reduce-scatter ──┘
```
节省一份完整的 buffer（1x param size）。

**NVFP4 双 Buffer 布局**（`:946-963`）：
- 参数 Buffer: N/2 bytes（每字节 2 个 FP4 值）
- 梯度 Buffer: N×2 bytes（BF16 全精度）
- 比全精度方案节省 75% 参数 buffer

**Grad Buffer 复用为 Param All-Gather Buffer**（`:357`）：
```python
reuse_buf = bucket.grad_data.view(param_dtype)  # recycle
```
梯度 reduce-scatter 完成后，buffer 被清零并重用于参数 all-gather。

### 2.6 FP8/FP4 参数精度显存节省

| 方案 | 参数显存节省 | 硬件要求 |
|------|------------|---------|
| BF16（基准） | 0% | 任何 GPU |
| FP8 E4M3 | 50%（2 bytes → 1 byte） | H100+ |
| MXFP8 | 50% + block scaling | GH200, Blackwell |
| NVFP4 | 75%（2 bytes → 0.5 byte） | Blackwell |

**首尾层保护**（`fp8_utils.py:594`）：`first_last_layers_bf16` 保持首尾 N 层为 BF16，防止精度损失在输入/输出端被放大。

### 2.7 CUDA Graph Buffer 复用 (`cuda_graphs.py`)

`TensorReusePool`（`:161`）复用 CUDA Graph 的输入/输出 buffer：
- 按 (shape, dtype, device) 匹配复用
- 跨 pipeline stage 共享（后续 stage 的输入 = 前序 stage 的输出）
- TE weak references（`:384-392`）允许更激进的 buffer 回收

### 2.8 Rerun State Machine (`rerun_state_machine.py`)

间接显存优化：允许使用更快但可能不稳定的计算路径（如特殊数值模式）。检测到结果异常（NaN, spiky loss）时，通过重运行恢复。

### 2.9 Sequence Parallelism 的隐性显存收益

按 `tp_size` 切分序列维度，每个 rank 的 Attention 激活值仅为 `1/tp_size`。对于 128K 序列、TP=8 的场景，Attention 激活从 ~8GB 降至 ~1GB/rank。

### 2.10 Resharding (`resharding/`)

`resharding/nvshmem_copy_service/memory/double_buffer_manager.py` 实现双缓冲，在并行策略动态切换时重叠通信与计算，减小过渡期的临时 buffer 显存。

## 3. 显存优化组合拳

```
对于 671B MoE @ 128 GPUs 的典型优化层次：

Layer 1（无开销）：FP8 参数 + 分布式优化器 + Sequence Parallelism
  → 参数显存: 1x → 0.5x | 状态显存: 3x → 3x/128

Layer 2（少量开销）：Grad Buffer 复用 + NCCL Memory Pool + FP8 Input Store
  → 临时 Buffer: ~50% 减少 | 激活: ~40% 减少

Layer 3（可控开销）：Selective Activation Checkpointing + Paged Stash（MoE）
  → 激活: 5-10x → 1-2x | MoE 激活: ~90% 减少

Layer 4（显著开销，降速 10-30%）：
  Optimizer State CPU Offloading + Fine-Grained Activation Offloading
  → 状态显存: +额外 2x 节省 | 激活: +额外 ~30% 节省
```

## 4. 配置决策表

| 模型规模 | GPU 配置 | 推荐显存优化组合 |
|---------|---------|----------------|
| 10B MoE | 8×A100 | 分布式优化器 + Sequence Parallelism |
| 50B MoE | 16×H100 | + FP8 参数 + Activation Checkpointing |
| 200B MoE | 64×H100 | + Grad Buffer 复用 + Paged Stash + FP8 Input Store |
| 671B MoE | 256×H100 | + NCCL Pool + Fine-Grained Activation Offloading |
| 1.xT MoE | 512×Blackwell | + NVFP4 + Optimizer State Offloading + 全量 Overlap |

## Related Pages

- [[megatron_distributed_optimizer_analysis]]
- [[megatron_fusion_operators_analysis]]
- [[megatron_comm_overlap_analysis]]
- [[activation_checkpointing_analysis]]
- [[low_precision_training_analysis]]
- [[Megatron-LM_Distributed_Parallel_Exam]]
