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

> [!update] 2026-06-16 · dev@232c478d4
> **NCCL UB 内存池：正确反注册 + 降显存（#4492，e35d4e50c）**
> 1. **退出时反注册 NCCL user-buffer 池**（`megatron/training/training.py:3862-3871`）：训练满足退出条件时，先 `torch.distributed.barrier()`，再遍历每个 `DDP` 的 `buffers + expert_parallel_buffers`，对带 `nccl_mem_pool` 的 buffer 调用 `nccl_allocator.deregister_mem_pool(...)`。不这样做的话，`ProcessGroupNCCL` 析构会走 `abort()` → 对 `ncclCommWindowRegister` 注册过的 handle 调 `ncclCommDeregister`，报 `NCCL WARN Deregister: Could not find handle` 并崩溃。
> 2. **每个 buffer 记录自己的池**（`megatron/core/distributed/param_and_grad_buffer.py:1012,1020`）：`_ParamAndGradBuffer.nccl_mem_pool` 字段保存 `create_nccl_mem_pool(...)` 返回的池句柄，供上面反注册时取用。
> 3. **torch≥2.11 放宽 `expandable_segments` 限制**（`megatron/core/distributed/distributed_data_parallel_config.py:223`）：`nccl_ub` 与 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 互斥的断言改为 `if self.nccl_ub and not is_torch_min_version("2.11.0a0")` —— 新版 torch 上两者可共存，降低显存碎片。

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

> [!deprecated] 2026-06-16：上面"三级溢出"的画法把**两个不同层级的机制**混为一谈，按 #4247 实测应拆成 **"2 级缓冲 + 1 级 runner 级重跑"**（详见下方 [!update]，源码 `ops/paged_stash.py:13`、`paged_stash.py:968`）。具体更正：
> - **Tier 2 的"host spill"不是溢出，而是成功的回退**。Triton kernel 注释原话："`host_spill` = 1 if any successful host spill (**not set on overflow path**)"（`megatron/core/transformer/moe/ops/paged_stash.py:25`）。CUDA 页用尽时正常往 pinned host 页写，只置 `host_spill` 标志（信息性日志，提示你调大 `factor_cuda`），**不触发重跑**。
> - **真正的 overflow = CUDA 页与 host 页都满** → kernel 置 `overflow` 标志（`ops/paged_stash.py:98`）。这才是触发重跑的条件之一。
> - **"重跑"不是 kernel 里的 Tier 3，而是 runner 级（`PagedStashRunner`）的整步 forward-backward 重跑**，且条件是 **stash overflow `或` HybridEP 容量 over-budget（任一 rank 命中）**，并非"都满了"。重跑会**同时**清掉 `moe_expert_rank_capacity_factor` 容量 padding **和** `moe_paged_stash`，最多重跑 1 次（共 2 次尝试）。

**配置参数**：
```python
moe_paged_stash_page_size: int = 64           # 页面大小
moe_paged_stash_buffer_size_factor_cuda: float = 1.10  # GPU buffer 比例
moe_paged_stash_buffer_size_factor_cpu: float = 0.0    # CPU buffer 比例
```

**适用条件**：MoE 训练（expert 数量 ≥ 8，capacity factor 大时收益最高）。与 `cpu_offloading` 不兼容。

> [!update] 2026-06-16 · dev@232c478d4
> **Paged Stashing 正式合入（#4247，f007db77b）+ Triton kernel 搬迁（#5003，b8fef119c）—— 对上文的权威订正与补全**
>
> 官方文档 `docs/user-guide/features/paged_stash.md` 把这个特性定义为 **"sync-free 专家执行 + paged stashing"** 两部分，比上文"只是 MoE 激活分页"更完整：
>
> **(1) 前置依赖（不是可选项）**：`moe_paged_stash` 要求先开 **sync-free**——`--moe-flex-dispatcher-backend hybridep` + `--use-transformer-engine-op-fuser` + `--moe-expert-rank-capacity-factor <float>`（`transformer_config.py:1789-1797` 校验）。`moe_paged_stash` 自身校验（`transformer_config.py:2044-2052`）：必须设容量因子、不能与 `cpu_offloading` 共存、`offload_modules` 不能含 `expert_fc1`/`moe_act`。容量因子 = "相对完美均衡情况的 buffer 倍率"，留 headroom 吸收路由 skew。
>
> **(2) 缓冲是 2 级而非 3 级**：`PagedStashBuffer`（`paged_stash.py:24`）维护两条 free list：`[0]=CUDA 页`、`[1]=pinned host 页`。host 页仅当 `moe_paged_stash_buffer_size_factor_cpu>0` 时分配（`num_tokens_host>0`）。pack/reload 由融合 Triton kernel `paged_stash_copy_kernel`（**已由 #5003 搬到 `megatron/core/transformer/moe/ops/paged_stash.py:13`**，原 `paged_stash.py:129` 引用失效）一次 launch 完成，无 host 同步、CUDA-graph 安全。
>
> **(3) overflow → runner 级整步重跑**：`PagedStashRunner.__call__`（`paged_stash.py:968,1149`）包裹 `forward_backward_func`：每趟跑完用 `check_moe_overflow()`（`:1066`）把 **stash_overflow / over_budget / host_spill** 三个标志 `all_reduce(SUM)` 跨全 rank 汇总；只要 stash overflow 或 over-budget 在**任一 rank** > 0 → `prepare_for_rerun()`（`:1087`）：清容量因子（转为 dropless 动态派发，靠 CPU sync 定尺寸）、关 paged stash、`zero_grad`、重置 FullCudaGraph、释放 stash buffer，然后**重跑整步**；`num_tries < 2` 断言保证最多 2 次。host_spill>0 只打日志（建议调大 `factor_cuda`），不重跑。
>
> **(4) 生命周期三态精确化**（`paged_stash.py:906-915` 在 `paged_stash_reset` 里推进）：`begin → capture → captured`。
> - `begin`：初始态，**首次 reset 即转 `capture`**（并无完整一迭代停在 begin）；
> - `capture`：本迭代**逐 (dtype, hidden_size) 累计 max tokens、并构建 `_pp_schedule`**（`:711-742`），此阶段**显式不使用 CUDA graph**（源码注释 `:744`，故可按实际 token 数截断保存张量）；下次 reset 时 `capture → captured` 并 `allocate_stash_buffers`；
> - `captured`：稳态，按 `_pp_schedule` 真正 stash/reload，**与 full CUDA graph 兼容**。注意上文"captured：使用 CUDA Graph 加速 stash/reload"措辞不准——CUDA graph 不是用来"加速 stash/reload"，而是 paged stash 被设计成 CUDA-graph-safe（freelist 原地 reset、无新分配）；rerun fallback 后 buffer 会被释放，再次进入 captured 时按 capture 期 maxima 重新分配（`:916-929`）。
>
> **(5) 配置项当前行号**（`transformer_config.py`）：`moe_expert_rank_capacity_factor:903`、`moe_paged_stash:1312`、`moe_paged_stash_page_size:1315`、`moe_paged_stash_buffer_size_factor_cuda:1318`、`moe_paged_stash_buffer_size_factor_cpu:1324`。

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

> [!update] 2026-06-16 · dev@232c478d4
> **新增"最大在途 offload"节流旋钮（#4692，9a7cd17fd）**：`fine_grained_offloading_max_inflight_offloads: Optional[int]`（`transformer_config.py:1304`）。语义——**按 offload group 名字**分别限制"已发起 D2H、但主流尚未 `wait_event` join 的 offload 个数"（每个名字一条独立 FIFO，同一上限）。`0` = 每次 offload 后主流立即等待该名字；`1` = 该名字最多 1 个未 join 的在途；`None`（默认）= 不插入这些 join。
> 实现：`ChunkOffloadHandler.__init__` 增 `max_inflight_offloads` 参数与 `_offload_pending_by_name`（`fine_grained_activation_offload.py:864`），commit offload 时入队事件并调 `_drain_offload_pending(name)`（`:1066`）让主流等待超额的旧事件。
> **动机**：在 **full-iteration CUDA graph** 捕获下，主流可能不会自动等 D2H 事件 → 在途 offload 无界堆积、pinned host buffer 与 D2H 队列爆显存/带宽；此旋钮给 CG 场景一个回压闸门。

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

### 2.11 dev 增量显存项（2026-05~06）

> [!update] 2026-06-16 · dev@232c478d4
> 一批小而实的显存相关改动：
>
> **(a) 提前 `del` 输出张量（#4742，f1b5516b2）**：`forward_backward_no_pipelining` 在每个 microbatch `backward_step` 后立即 `del output_tensor`（`pipeline_parallel/schedules.py:751,774`）。不删的话上一个 microbatch 的 `output_tensor` 会一直活到下次迭代重新绑定变量，把 autograd 节点的析构推迟到下一次 forward 的 dispatch 路径上，触发 PyTorch "AccumulateGrad node's stream does not match" 警告（issue #4124）；提前删让 autograd 计算图头部及时释放。
>
> **(b) 删除 checkpoint 期的 GPU cache 回收 workaround（#5170，3a183e235）**：`save_checkpoint_and_time` 里原先为给异步 checkpoint worker 腾 D2H 显存而做的 `free_overlap_buffers()` + `torch.cuda.empty_cache()`（`training/training.py:~2855` 附近）已被**移除**（7 行）。该 workaround 不再需要。
>
> **(c) 权重/优化器显存估算正确计入 EP（#4687，a7c9e8c44）**：`theoretical_memory_usage.py` 重写 `compute_weight_and_optimizer_memory`，把 **routed expert 参数**单独按 `expert_tensor_parallel_size × expert_model_parallel_size` 分片、按 `expert_data_parallel_size = world_size /(etp×ep×pp)` 计 optimizer 字节，并区分 shared_expert / router / shared_expert_gate / active vs total 专家参数。修复了此前 MoE+EP 下理论显存被高/低估的问题（纯估算工具，不改运行时显存）。
>
> **(d) 修复 FSDP double-buffer 的 CUDA IMA（#4810，d199bb9e9）**：`custom_fsdp/.../param_and_grad_buffer.py:727` 把 `FixedPoolAllocator.backup_allocator` 从 `TemporaryBucketAllocator()` 改为 `StorageResizeBasedBucketAllocator()`。当某个 FSDP unit 的 bucket 装不进固定池、回退到 backup 分配器时，旧实现会越界访问（IMA）；改用基于 storage resize 的分配器修正。

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
- [[recompute_analysis]]
- [[low_precision_training_analysis]]
- [[Megatron-LM_Distributed_Parallel_Exam]]
