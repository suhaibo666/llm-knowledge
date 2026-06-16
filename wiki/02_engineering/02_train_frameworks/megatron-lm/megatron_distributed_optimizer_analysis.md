# Megatron-LM 分布式优化器 深度分析

**Date**: 2026-05-12
**Status**: Complete
**Source**: `megatron/core/optimizer/distrib_optimizer.py`, `param_and_grad_buffer.py`, `cpu_offloading/`

## 1. 优化点是什么？

分布式优化器在数据并行（DP）维度上分片优化器状态（Adam 的 momentum、variance）和 FP32 主权重，将显存压力从每个 GPU 分摊到整个 DP 组。核心思想是：每个 DP rank 只"拥有"参数的一部分分片，只为自己拥有的分片维护优化器状态。

这与 ZeRO-1（优化器状态分片）等价。配合 Grad Buffer 的 Reduce-Scatter 替代 All-Reduce，额外实现了 ZeRO-2（梯度分片）的效果。

**显存节省**：对于 Adam 优化器，状态占用为参数的 2x（m + v）+ 1x（FP32 主权重）= 3x。分片到 N 个 DP rank 后，每个 rank 只存 `3P/N`。

## 2. 为什么有效？

### 2.1 核心机制：连续 Buffer 均匀分片

`param_and_grad_buffer.py:838-852` 将全部参数展平为连续的大 Buffer，切分为 `dp_world_size` 个等大块：

```
Bucket: [████████████████████████████████]
          ← dp_rank 0 → ← dp_rank 1 → ← dp_rank 2 → ← dp_rank 3 →
```

每个 rank "拥有"对应块上的参数子集，负责：
1. Reduce-Scatter：将梯度归约到自己拥有的分片
2. 优化器状态存储：仅为自己的分片保留 Adam state
3. All-Gather：更新后将参数广播到所有 rank

### 2.2 关键设计：分片不按参数边界

`distrib_optimizer.py:155-182` 核心映射 `_build_model_gbuf_param_range_map`：

```python
# 一个参数可能被多个 DP rank 分片持有
param_local_start = max(0, param_world_start - gbuf_world_range.start)
param_local_end = min(gbuf_world_range.size, param_world_end - gbuf_world_range.start)
```

分片边界可能"切"在参数的中间。一个参数的不同部分由不同 DP rank 维护优化器状态。这意味着每个 rank 上的参数只是一个 view/shard，不是完整的参数。

### 2.3 通信组与通信量分析

#### 2.3.1 涉及的通信组

| 通信组 | 获取函数 | 组成员 | 通信操作 |
|--------|---------|--------|---------|
| **DP Group** | `get_data_parallel_group()` | 所有数据并行的 rank | AllReduce（标准DP）/ ReduceScatter + AllGather（DistOpt） |
| **Intra-Instance DP** | DP Group 的子组 | `num_distributed_optimizer_instances` 划分的子组 | ReduceScatter（梯度分片） |
| **Inter-Instance DP** | DP Group / Intra-Instance | 跨 instance 的 rank | AllReduce（梯度去重, HSDP only） |
| **EP DP Group** | `get_expert_data_parallel_group()` | Expert 数据并行的 rank | AllReduce（expert 梯度） |

**通信组定义**（`parallel_state.py:1482, 2000`）：
```python
# DP Group: 跨所有 TP/PP/EP 的相同位置 rank 组成
get_data_parallel_group(with_context_parallel=True)

# EP DP Group: Expert 的 DP 组（EP 隔离专家参数后形成）
get_expert_data_parallel_group()
```

#### 2.3.2 通信量与通信原语

设模型参数总量为 **P**（BF16/FP8 元素个数），DP 世界大小为 **D**：

**标准 AllReduce 方案（无分片）**：
```
Forward:  无通信（参数已同步）
Backward: 1× AllReduce(grad) = 2P/D × (D-1)  ← 通信量 2P bytes
          总: 2P bytes (per step)
```

**分布式优化器方案（梯度+状态分片）**：
```
Forward:  1× AllGather(param) = P bytes (所有 rank 获得完整参数)
Backward: 1× ReduceScatter(grad) = P bytes (每个 rank 只保留 1/D 的梯度)
          总: 2P bytes (per step)

通信量相同，但：
  - AllGather: 每 rank 输出 P 字节（gather 后取完整参数）
  - ReduceScatter: 每 rank 输出 P/D 字节（只保留分片部分）
```

**HSDP 多层方案**（`num_distributed_optimizer_instances = K`）：
```
Intra-instance (组内 D/K 个 rank):
  Forward:  1× AllGather(param) = P/K bytes (组内共享完整参数)
  Backward: 1× ReduceScatter(grad) = P/K bytes (组内梯度分片)
Inter-instance (跨 K 个 instance):
  Backward: 1× AllReduce = 2P/D × (K-1) bytes (组间去重)
  总: 2P/K + 2P/D×(K-1) bytes
```

#### 2.3.3 正反向通信时序

```
═══════════════════════════════════════════════════════════════════
                   标准 AllReduce（无分片）
═══════════════════════════════════════════════════════════════════
Forward:  [Layer 1] [Layer 2] ... [Layer L]     ← 无通信
Backward: [∂L/∂W ← AllReduce ← ∂L/∂W] × N buckets
          每个 bucket: 2P_bucket × (D-1)/D bytes
═══════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════
                   分布式优化器（ReduceScatter + AllGather）
═══════════════════════════════════════════════════════════════════
Forward:  Wait(Bucket1_AG) → [Layer 1,2] → Wait(Bucket2_AG) → [Layer 3,4] ...
          (Param AllGather overlap with compute via _make_forward_pre_hook)

Backward: [Layer L] → Trigger(Bucket1_RS async) → [Layer L-1] → Trigger(Bucket2_RS) ...
          (Grad ReduceScatter overlap with compute via start_grad_sync)

Optimizer.Step:
  _copy_model_grads_to_main_grads()   ← 分片 grad → FP32 main grad
  optimizer.step()                    ← 仅更新使用的分片
  _copy_main_params_to_model_params() ← FP32 → BF16/FP8

Post-Step:
  start_param_sync() ← AllGather（不 overlap 时）或
                     ← 下一个 Forward 的 _forward_pre_hook 中处理（overlap 时）
═══════════════════════════════════════════════════════════════════
```

#### 2.3.4 FP8/FP4 参数对通信量的影响

| 参数精度 | AllGather 通信量 | ReduceScatter 通信量 | 总通信节省 |
|---------|-----------------|---------------------|-----------|
| BF16 (基准) | P × 2 bytes | P × 2 bytes | 0% |
| FP8 E4M3 (`fp8_param_gather=True`) | P × 1 byte | P × 2 bytes (grad 仍为 BF16) | 25% |
| MXFP8 (`fp8_param_gather` + `reuse_grad_buf`) | 共享 buffer → 0 | P × 2 bytes (仅 grad) | ~33% |
| NVFP4 (`fp4_param_gather=True`) | P × 0.5 byte | P × 2 bytes (grad 仍为 BF16) | 37.5% |

> 注：梯度 ReduceScatter 始终在 BF16/FP32 精度下执行，因为梯度累积需要全精度。

#### 2.3.5 Bucket 策略与通信粒度

`param_and_grad_buffer.py:838-852` 定义了 Bucket 划分策略：

```python
# 默认 bucket_size = max(40000000, 1000000 * dp_size) 参数
# 每个 bucket 独立触发 ReduceScatter/AllGather
# padding: lcm(dp_size, 128, 65536) → 确保均匀分片 + NCCL 高带宽
```

Bucket 数量 = `total_params / bucket_size`。更多 bucket = 更细粒度的 overlap（backward 中每完成一个 bucket 的梯度就触发 ReduceScatter），但更小的消息可能导致 NCCL 延迟成为瓶颈。

**NCCL 带宽与 Bucket 大小的关系**：
- Bucket < 1MB → latency-bound（延迟主导）
- Bucket > 10MB → bandwidth-bound（带宽主导）
- 默认 bucket_size 精心选择以在两者间取得平衡

`param_and_grad_buffer.py:848`：
```python
bucket_size_divisor = math.lcm(self.data_parallel_world_size, 128, 2**16)
```

Padding 到 `lcm(dp_size, 128, 65536)` 确保每个 bucket 可以被均匀分片，且对齐到 NCCL 最优传输粒度，最大化 NVLink/IB 带宽利用率。

> [!update] 2026-06-16 · dev@232c478d4 — bucket 对齐/尺寸的两处更正
> **① 65536 对齐是条件性的,不是恒定的**:bucket 末端对齐 divisor 现集中在 `param_layout.py:29-33` `bucket_end_divisor()`,只有 `pad_buckets_for_high_nccl_busbw=True` 时才是 `lcm(dp, 128, 2**16)`;否则只对齐到 `lcm(dp, 128)`。即上文的 `65536` 项是"高 NCCL busbw"开关下的产物,默认未必启用。
> **② 默认 `bucket_size` 公式改由 pg_collection 计算**(#5006,`dist_utils.py:249`):`max(40000000, 1000000 * pg_collection.dp_cp.size())` —— 数值口径与原 `1000000 * dp_size` 一致,但来源从全局 `mpu.get_data_parallel_world_size()` 换成显式 `pg_collection.dp_cp.size()`(同时 `pp_rank`、`expert_data_parallel_world_size` 等也都改走 pg_collection)。`pg_collection` 现在对 Megatron-FSDP 与 DistributedOptimizer **两条路径都会传入**(原仅 FSDP 传)。

## 3. 源码关键实现

### 3.1 类继承结构

```
MegatronOptimizer           [optimizer.py:100]
  └─ MixedPrecisionOptimizer [optimizer.py:465]
       └─ DistributedOptimizer [distrib_optimizer.py:98]
```

### 3.2 DP 分组与多实例

支持 HSDP（Hybrid Sharded Data Parallel）模式：
- `num_distributed_optimizer_instances > 1` 时，DP 域划分为组内（intra-instance）和组间（inter-instance）
- 组内做 Reduce-Scatter 分片 + 组间做 All-Reduce 去重
- 参见 `param_and_grad_buffer.py:596-650`

### 3.3 精度处理

**标准混合精度**：模型 BF16 → 主权重 FP32 → 梯度从 BF16 转 FP32
```python
# distrib_optimizer.py:2570
shard_main_param.grad = shard_model_grad.float()
```

**精度感知优化器** (`use_precision_aware_optimizer: True`)：
- 主权重、exp_avg、exp_avg_sq 可采用不同的低精度格式
- 使用 `.decoupled_grad` 解耦模型参数 dtype 和优化器 state dtype
```python
# distrib_optimizer.py:2568
shard_main_param.decoupled_grad = shard_model_grad
```

### 3.4 FP8/FP4 量化参数

**FP8 参数 All-Gather**（`distrib_optimizer.py:2609-2632`）：
- 参数在 FP8 格式下做 All-Gather，传输量减半
- FP32 主权重通过 `quantize_param_shard()` 量化回 FP8

**NVFP4 双 Buffer 布局**（`param_and_grad_buffer.py:946-963`）：
- 参数 Buffer：每字节 2 个 FP4 值（numel/2）
- 梯度 Buffer：全精度 BF16（numel）
- 需要两套索引映射

**MXFP8 共享 Buffer**（`param_and_grad_buffer.py:1097-1113`）：
- 参数 All-Gather 和梯度 Reduce-Scatter 共享同一块显存
- 通过 `reuse_grad_buf_for_mxfp8_param_ag` 控制

> [!update] 2026-06-16 · dev@232c478d4 — 精度/MXFP8 相关行号与门控
> - **行号漂移**(§3.3):解耦梯度赋值 `shard_main_param.decoupled_grad = shard_model_grad` 现在 `distrib_optimizer.py:2728`(原 `:2568`);`shard_main_param.grad = shard_model_grad.float()` 现 `:2730`(原 `:2570`)。语义未变。
> - **MXFP8 共享 buffer 的 all-gather 后处理被抽函数**(#4771,`distributed_data_parallel.py:492` `_start_bucket_group_param_sync`):原内联在 `start_param_sync` 里的"把 all-gather 出的 MXFP8 参数从共享 buffer 拷回 `param.data` 并清零 buffer 供梯度累加"逻辑被抽成单 bucket-group 方法,便于 LayerWise+DistOpt 链式各自只同步自己的 bucket。
> - **ChainedOptimizer 的 MXFP8 defer-sync 改为探测 DDP config**(#4982,`optimizer.py:1456`):`_should_defer_mxfp8_param_sync` 不再信 `OptimizerConfig.overlap_param_gather`,而是逐个探测子 `DistributedOptimizer.ddp_config.overlap_param_gather`(详见 [[megatron_optimizer_internals_analysis]] §1.1 的 2026-06-16 更新)。

### 3.5 完整训练迭代流程

```
Forward（参数 All-Gather 可与计算 overlap）
  → Backward（梯度 Reduce-Scatter 可与计算 overlap）
    → _copy_model_grads_to_main_grads()  [model grad → FP32 main grad]
      → optimizer.step()（仅更新本 rank 的分片）
        → _copy_main_params_to_model_params() [FP32 → BF16/FP8]
          → start_param_sync() [All-Gather 参数]
            → 下一轮迭代
```

## 4. CPU Offloading 机制

### 4.1 HybridDeviceOptimizer

`cpu_offloading/hybrid_optimizer.py:14` — 将参数按比例拆分到 GPU 和 CPU：

- `offload_fraction`（默认 0.5）：控制多少参数放在 CPU
- 双流 Overlap：`_d2h_stream` 传梯度到 CPU，`_h2d_stream` 传参数回 GPU
- 支持 `param_update_in_fp32`：CPU 上做 FP32 更新
- 通过 step hooks 自动化参数回拷

### 4.2 OptimizerStateOffloader

`cpu_offloading/optimizer_state_offloader.py` — 在 optimizer.step() 完成后将状态暂存 CPU：
- offload：D2H 异步拷贝 exp_avg, exp_avg_sq, master weights
- release：GPU 显存 resize_(0) 释放
- reload：两阶段——先分配 GPU 显存，再 H2D 异步拷回
- 下次 step 前调用 `sync_before_step()` 等待 H2D 完成

## 5. 配置决策树

| 场景 | 推荐配置 |
|------|---------|
| 单机 8 GPU，<10B 模型 | `use_distributed_optimizer=True`, 无需特殊设置 |
| 多机 32 GPU，70B 模型 | `use_distributed_optimizer=True`, `overlap_param_gather=True` |
| 多机 128 GPU，200B+ MoE | `use_distributed_optimizer=True`, `overlap_param_gather=True`, `overlap_grad_reduce=True`, 考虑 `fp8_param=True` |
| 极端规模（H100/Blackwell） | 全开 overlap, `fp4`/`fp8_param`, 复用 `reuse_grad_buf_for_mxfp8_param_ag` |
| 显存仍然不足 | 启用 `optimizer_cpu_offload` 或 `offload_optimizer_states` |

## 6. 何时适用

- ✓ 任何使用 Adam/AdamW 的场景
- ✓ 模型越大收益越高（优化器状态占比随参数量线性增长）
- ✓ DP 世界大小 ≥ 2
- ✓ 可与 TP、PP、EP 叠加（正交优化）
- ✗ DP=1 时无收益（无分片对象）
- ✗ SGD 无状态优化器收益有限（SGD 只有 momentum buffer）

## Related Pages

- [[Megatron-LM_Distributed_Parallel_Exam]]
- [[megatron_memory_optimization_analysis]]
- [[low_precision_training_analysis]]
- [[megatron_ep_analysis]]
- [[../distributed_optimizer_deep_dive|distributed_optimizer_deep_dive]] — FSDP2/ZeRO/MindSpeed 三方对比, 梯度累积通信量分析, Adam vs Muon

---

## 附录 A：Megatron 的三种梯度/参数分片策略对比

Megatron-LM 现有 **三套** 并行梯度/参数分片方案，适用于不同场景：

### A.1 三套方案概览

| 维度 | DistributedOptimizer | TorchFullyShardedDataParallel | MegatronFSDP |
|------|---------------------|------------------------------|-------------|
| **文件** | `distrib_optimizer.py:98` | `torch_fully_sharded_data_parallel.py:28` | `fsdp/src/megatron_fsdp/megatron_fsdp.py:105` |
| **分片粒度** | 参数级别（连续 buffer 切分） | Module 级别（FSDP Unit） | Module 级别（FSDP Unit） |
| **分片策略** | ZeRO-1/2（状态+梯度分片） | PyTorch FSDP2（参数+梯度+状态） | ZeRO-1/2/3 可配置 |
| **依赖** | 无外部依赖 | PyTorch >= 2.4, DTensor | 自研（不依赖 PyTorch FSDP） |
| **与 EP 协同** | 通过 `expert_parallel_buffers` 隔离 | 通过 `_check_module_parameter_types` | 通过 `has_expert_parameters` 自动检测 |
| **通信 Overlap** | `overlap_param_gather` + `overlap_grad_reduce` | PyTorch FSDP2 自动 | `overlap_param_gather` + `overlap_grad_reduce` 默认开启 |
| **CUDA Graph** | 兼容 | 不兼容（PyTorch FSDP2 限制） | 兼容 |
| **NCCL UB** | 不支持 | 不支持 | 支持（`nccl_ub` 减少 SM 占用） |

### A.2 MegatronFSDP 详细分析 (`fsdp/src/megatron_fsdp/megatron_fsdp.py`)

MegatronFSDP 是 NVIDIA 自研的 FSDP 实现，提供从 ZeRO-1 到 ZeRO-3 的完整分片谱系：

**分片策略**（`megatron_fsdp.py:112-120`）：
```python
# data_parallel_sharding_strategy 控制:
'no_shard'             # 传统 DP（无分片）
'optim'                # ZeRO-1: 仅优化器状态分片（+ FP32 主权重）
'optim_grads'          # ZeRO-2: 梯度 + 优化器状态分片
'optim_grads_params'   # ZeRO-3: 参数 + 梯度 + 优化器状态全分片
```

**四种训练状态**（`megatron_fsdp.py:62-74`）：
```python
class TrainingState(Enum):
    FORWARD = auto()       # Forward: 参数需 unshard
    PRE_BACKWARD = auto()  # Pre-backward: 参数需 unshard
    POST_BACKWARD = auto() # Post-backward: 梯度需 re-shard
    IDLE = auto()          # 空闲：无 un/sharding 活动
```

**FSDP Unit 概念**（`megatron_fsdp.py:141-143`）：
FSDP Unit 是最小可释放模型单元。参数按 Unit 分组——在 Forward 进入 Unit 时 AllGather 参数，离开时释放；Backward 进入时重新 AllGather。默认 Unit = `TransformerLayer`。

**与 Activation Checkpointing 的协同**（`megatron_fsdp.py:127-130`）：
> 重算整个 Transformer Layer 时，参数只需 Gather 一次，同时服务于重算和 Backward 计算。

**Delayed Wgrad Overlap**（`megatron_fsdp.py:77-103`）：
当启用 `overlap_dispatch_backward_with_experts_wgrad` 时，expert 参数的梯度 reduce-scatter 延迟到 MoE dispatch backward 完成后再执行，最大化 EP 通信与 DP 梯度同步的重叠。

**NCCL UserBuffer**（`megatron_fsdp.py:164-168`）：
```python
nccl_ub=True → 使用 NCCL UserBuffer 进行 FSDP 通信
  - 减少 SM 占用（通信操作占用更少计算资源）
  - 自动设置 fsdp_double_buffer=True（使用额外 GPU 显存换性能）
```

**HSDP 分层分片**（`megatron_fsdp.py:247-252`）：
```python
data_parallel_sharding_strategy="optim_grads_params"  # 组内全分片
outer_dp_sharding_strategy="no_shard"                 # 组间无分片（复制）
```
组内做 ZeRO-3 全分片，组间做 AllReduce 去重。

> [!update] 2026-06-16 · dev@232c478d4 — Megatron-FSDP 内部一组修复(FSDP-internal)
> **① `no_shard`(ZeRO-0)收敛性修复**(#3835/#3754,`megatron_fsdp.py:1234`、`optimizer/__init__.py:1060`):`no_shard` 下参数本就在各 DP rank 复制,故 ① `start_param_sync` 对 `no_shard` 直接 return(无需 all-gather);② 梯度统计/范数只能在 **TP/PP(model_parallel_group)** 上规约,**不能**再在 DP 维度规约(梯度已是 all-reduce 后的复制值,再 reduce 会**虚高 grad norm 致不收敛**)—— 通过 `effective_intra_dist_opt_group = mp_group if no_shard else intra_dist_opt_group` 实现。另禁止 `no_shard` 配 meta-device 初始化。详见 [[megatron_ddp_optimizer_analysis]] 阶段① 的 2026-06-16 更新。
> **② grouped expert 权重减少 padding**(#5013,`fsdp/.../param_and_grad_buffer.py:1404`):当 ≥3D 的 grouped-expert 张量与异构 chunk-size-factor 混在同一 bucket 时,LCM 对齐会**放大 padding**;修复把这类 grouped-expert 张量拆到独立 bucket,避免 LCM 膨胀。利好大规模 MoE(见 §A.5)。
> **③ 跨 AllGatherPipeline reset 保留非-FSDP-unit bucket**(#4717,`fsdp/.../param_and_grad_buffer.py`):pipeline reset 时不再误清非 FSDP-unit 的 bucket。
> **④ A2A Overlap**(#3797):把 MoE 的 All-to-All dispatch/combine 与 FSDP 的参数 all-gather / 梯度 reduce-scatter 重叠,详见 [[megatron_ddp_optimizer_analysis]] 阶段④ 与 [[megatron_comm_overlap_analysis]]。

### A.3 TorchFullyShardedDataParallel 详细分析 (`torch_fully_sharded_data_parallel.py`)

对接 PyTorch FSDP2 `torch.distributed.fsdp.fully_shard` API：

**Sub-module 级别的 `fully_shard` 包装**（`torch_fully_sharded_data_parallel.py:60-64, 126-134`）：
```python
sub_modules_to_wrap = {
    TransformerLayer,           # 所有 Transformer 层
    LanguageModelEmbedding,     # 初始嵌入层
    RotaryEmbedding,            # 初始 RoPE
    ColumnParallelLinear,       # 最终输出层
}
# 每个 sub-module 独立 fully_shard → 逐层 AllGather/释放参数
fully_shard(sub_module, mesh=device_mesh, reshard_after_forward=True)
```

**FP8 权重转置缓存处理**（`torch_fully_sharded_data_parallel.py:93-98`）：
PyTorch FSDP2 无法感知 micro-batch 边界，会缓存 FP8 权重的转置版本。Megatron 通过 `save_custom_attrs` / `restore_custom_attrs` 机制在每次 `fully_shard` 前后保存/恢复参数的 FP8 属性，避免不必要的显存占用。

**与 Activation Checkpointing 的 Backward Prefetch 协调**（`torch_fully_sharded_data_parallel.py:136-141`）：
```python
if config.recompute_granularity is not None:
    sub_module.set_modules_to_backward_prefetch(
        [prev_module] if prev_module else []
    )
```
显式设置 backward prefetch schedule，防止 Activation Checkpointing 的重复计算破坏 FSDP2 自动生成的 prefetch 顺序。

### A.4 三套方案选型矩阵

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 标准训练（无特殊需求） | `DistributedOptimizer` | 最成熟、与所有 MCore 特性深度集成 |
| PyTorch >= 2.4 + 新项目 | `TorchFullyShardedDataParallel` | 对接 PyTorch 原生 API，社区支持好 |
| 需要 CUDA Graph + FSDP | `MegatronFSDP` | TorchFSDP2 不支持 CUDA Graph |
| 需要 NCCL UB 优化 | `MegatronFSDP` | 独有的 NCCL UserBuffer 支持 |
| 需要 ZeRO-3 全分片 | `MegatronFSDP` | `optim_grads_params` 策略 |
| MoE（与 EP 深度耦合） | `MegatronFSDP` | 自动检测 EP 参数 + Delayed Wgrad Overlap |
| HSDP 分层分片 | `MegatronFSDP` | 组内 ZeRO-3 + 组间复制的分层策略 |

### A.5 为什么 FSDP2 在 MoE 训练中重要

1. **ZeRO-3 对 Expert 参数的全分片**：MoE 的 expert 数量增长（64→256+），在 EP 不能无限扩展时（通信开销），FSDP 可在 DP 维度进一步分片 expert 参数。`MegatronFSDP` 的 `optim_grads_params` 支持参数级的分片释放，对 1.xT MoE 至关重要。

2. **与 EP 的层级协同**：EP 处理跨 expert 的分布（通信模式：All-to-All），FSDP 处理跨 DP 的参数分片（通信模式：AllGather/ReduceScatter），两者正交叠加。

3. **FSDP Unit 粒度控制**：MoE 中的 Expert 可以作为独立的 FSDP Unit，在不需要时释放参数，需要的 token 到达时再 AllGather。

### A.6 FSDP 与并行拓扑的关系

**FSDP 的 shard 在哪个进程组维度上执行？**

FSDP 的 shard 在 **DP 维度**上执行（即 `world_size / (TP × CP × PP × EP)`）。FSDP 的 shard group 不应与 TP/EP/CP/PP group 重叠，因为那些 group 已经做了参数分片或 activation 分片，FSDP 只负责 DP 维度的冗余消除。

```
在 TP=4, EP=8 配置下的 FSDP 分组：
  总 rank = 256
  TP group:    4 个 rank（同 PP stage, 同 EP rank）
  EP group:    8 个 rank（跨 expert）
  PP group:    P 个 rank（跨层）
  剩余维度 = 256 / (4 × 8 × P) = DP size
  FSDP shard:  在 DP group 上执行（只在此 group 内 reduce-scatter/all-gather）
  
  关键：FSDP group ∩ TP group = ∅
        FSDP group ∩ EP group = ∅
        FSDP group ∩ PP group = ∅
```

### A.7 Layer-Wise Distributed Optimizer（Q17 核心要点）

`Layer-Wise Distributed Optimizer`（`--layer-wise-distributed-optimizer`）将参数按**层**分配到 DP rank，而非按扁平的参数列表：

**解决的问题**：
- 支持**多个优化器组合**（如 Muon 处理 ≥2D 矩阵参数，AdamW 处理 vector/bias 参数），普通 distributed optimizer 难以优雅支持 per-parameter optimizer 切换
- 更细粒度的 all-gather overlap：可在计算第 L 层 forward 的同时，异步 all-gather 第 L+1 层的参数

**ChainedOptimizer 分配规则**：
- 通过 `param_group` 的 `optimizer_name` 或 `foreach` 映射规则路由不同参数到不同底层优化器
- 例如：所有 `weight` 矩阵参数（≥2D）→ `MuonOptimizer`，所有 `bias`、`norm`、`embedding` 参数 → `AdamWOptimizer`

**选择场景**：使用混合优化器（如 Muon + AdamW）或超大模型需要极致 per-layer overlap 时。

> [!deprecated] 2026-06-16:**触发方式更正**。不存在 `--layer-wise-distributed-optimizer` 这个 flag。Layer-wise 分布式优化器通过 **`--optimizer muon`(或其它 emerging 优化器)+ `--use-distributed-optimizer`** 触发:`arguments.py:1811-1823` 在 optimizer 非 `sgd`/`adam` 且开了 distributed optimizer 时,把 `use_layer_wise_distributed_optimizer` 置 True、并关掉普通 `use_distributed_optimizer`。`--optimizer dist_muon` 是旧写法,已弃用。

> [!update] 2026-06-16 · dev@232c478d4 — LayerWise 与 DDP buffer 基建整合 + 非-Muon 参数改走真正的 DistributedOptimizer(#4509 / #4771,`layer_wise_optimizer.py`、`optimizer/__init__.py:796-960`、`distrib_optimizer.py:3041`)
>
> 这组 PR 实质性改写了 layer-wise 的实现,并**修正了上文"普通 distributed optimizer 难以优雅支持 per-parameter optimizer 切换"的暗示** —— 现在两者是**链式协作**,而非二选一:
>
> **① LayerWise 不再用独立 ping-pong 路径,而是建在 DDP 的 grad/param buffer 之上**(#4509)。它预计算一个 shard-aligned 的 `FullParamLayout`/`PerBufferParamLayout`(`param_layout.py`),把参数按 backprop 顺序装进**对齐到 shard 边界**的 bucket,使任何参数都不跨 shard 边界,从而能直接复用 DDP 的 reduce-scatter(`use_distributed_optimizer=True` 时)/ all-gather 通信与 `overlap_grad_reduce`/`overlap_param_gather` 重叠语义。装箱算法在 #4771 中从"同尺寸配对(size-matching)"换成 **LPT 贪心装箱**(按 numel 降序塞进当前负载最小的 shard),在保证 bucket 连续 backprop 区间的同时让各 shard 尽量均衡。
>
> **② 非-Muon 参数改由独立的 `DistributedOptimizer` 按字节级分片管理**(#4771)。新增 `is_managed_by_layer_wise_optimizer(param)`(`layer_wise_optimizer.py:37`):2D 矩阵权重且非 embedding/output → Muon/LayerWise 接管;embedding、bias、LayerNorm 等 → **路由到一个独立的 `DistributedOptimizer`**(真正的 ZeRO 字节级分片)。`BufferKey` 增加 `is_managed_by_layer_wise_optimizer` 维度(`param_and_grad_buffer.py:863`),让两类参数落进不同 buffer;`DistributedOptimizer.start_param_sync_for_bucket_group_subset()` 只同步自己那批 bucket group,避免与 sibling LayerWise 重复 all-gather。最终 `LayerWiseDistributedOptimizer`(Muon)+ `DistributedOptimizer`(Adam)由 `ChainedOptimizer` 串成一个。
>
> **结论(对上文 Muon/ZeRO 框架的修正)**:Muon 现在**可以与 ZeRO 分片共存**。Muon 管的矩阵权重经 LayerWise 走 shard-aligned 的 reduce-scatter/all-gather(等效 ZeRO-1/2 沿 DP 分片优化器状态与梯度),非-Muon 参数走标准 `DistributedOptimizer`。早期"Muon 对 ZeRO 切分的根本性挑战"(见 [[../distributed_optimizer_deep_dive]])指的是 Newton-Schulz 正交化需要**整块矩阵**、无法像 Adam 那样按字节随意切;LayerWise 的解法正是 **shard-aligned bucket + 按层/按整参数分配**,让每个矩阵整体落在某个 shard 内,从而既正交化又分片。
>
> **限制**:此 split 路径要求 `use_layer_wise_param_layout=True`(默认开;`--no-use-layer-wise-param-layout` 回退到 legacy ping-pong)、`num_distributed_optimizer_instances == 1`、且不支持 expert-parallel 的非-Muon 参数组与 `overlap_param_gather_with_optimizer_step`(`optimizer/__init__.py:761` 断言)。

> [!update] 2026-06-16 · dev@232c478d4 — MTP-stage word_embeddings 必须打 `is_embedding_or_output_parameter` 标签(#5034,`language_module.py:205-213`)
> `is_embedding_or_output_parameter` 标签决定参数被 Muon/LayerWise 接管还是路由给 Adam/DistOpt(见上)。MTP(Multi-Token Prediction)阶段的 `word_embeddings.weight` 是 pre_process embedding 的**副本**(靠跨 stage all-reduce 同步),原来漏打此标签 → 被 LayerWise 当作 2D 矩阵接管、且因 `shared_embedding=True` 在 `_emit_bucket` 里把整个 `(vocab × hidden)` 张量**复制到全部 `dp_size` 个 shard**,使该 chunk 的 buffer 膨胀约 8×。修复:`pre_process` 或 `mtp_process` 任一为真就打标签,让 MTP embedding 正确归 Adam/DistOpt 管理。

4. **PyTorch FSDP2 (`fully_shard`) 的原生支持**：`TorchFullyShardedDataParallel` 让 Megatron 可以跟随 PyTorch 上游的 FSDP2 优化（如 per-param FSDP、DTensor 集成），降低维护成本。
