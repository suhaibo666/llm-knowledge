---
title: "Megatron-LM 显存优化 全景分析"
---

# Megatron-LM 显存优化 全景分析

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）。
> **重定基线**：2026-09-01 由 `71092579`（2026-08-27）推进，跨 7 个提交；本页落在本轮改动文件上的引用已按 difflib 逐行对齐重定位（含裸续引 `:NNN`），指向历史基线（`ee3f1ff` / `232c478d4`）的引用按原样冻结、未参与重定位。
> **重定基线**：2026-08-28 由 `ee3f1ffa2acd18131ab67cabab4cec45283512ab`（2026-05-19）推进，跨 578 个提交；本页全部 `path:line` 形式的引用已在新基线下逐条重核;**代码块内被点名的符号与不带行号的裸路径不在该次扫描口径内**,已知漏网处已于 2026-08-28 单独更正。
> **叙事顺序**：本页按五拍组织——背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势。
> **最近更新**：2026-08-28。按五拍重排章节顺序；机制正文与既有引用未改。

**Date**: 2026-05-12
**Status**: Complete
**Source**: `megatron/core/nccl_allocator.py`, `megatron/core/transformer/moe/paged_stash.py`, `megatron/core/optimizer/cpu_offloading/`, `megatron/core/pipeline_parallel/fine_grained_activation_offload.py`, `megatron/core/fp8_utils.py`, `megatron/core/fp4_utils.py`

## 1. 背景：五类显存的瓶颈全景

大规模 MoE 训练中，显存消耗可分为五类：

| 类别 | 占用比例（典型） | 优化手段 |
|------|-----------------|---------|
| **模型参数** | 1x（BF16） | FP8/FP4 量化、Sequence Parallelism |
| **梯度** | 1x（BF16） | 分布式优化器（Reduce-Scatter 替代 All-Reduce）、Buffer 复用 |
| **优化器状态** | 2-3x（FP32） | 分布式优化器分片、CPU Offloading、FP8 状态 |
| **激活值** | 5-10x（与 seq_len² 相关） | Activation Checkpointing、Activation Offloading、Paged Stash（MoE） |
| **临时 Buffer** | 1-3x | NCCL Memory Pool、CUDA Graph Buffer 复用、融合算子 |

Megatron-LM 提供了覆盖所有五类的完整优化工具箱。

## 2. 为什么这么设计：五类显存，五种最便宜的省法

§1 的表格是本页的骨架：**没有一个通用的省显存开关**，五类显存各自的最便宜省法互不相同，所以工具箱是五套独立机制而非一个总预算器。下面五条决断的判据都能落到源码或官方文档上，不需要推断。

| 决断 | 选中的路线 | 被否掉的替代 | 源码/文档给出的判据 |
|---|---|---|---|
| 通信 buffer 从哪来 | `ncclMemAlloc` 分配并注册成 **NCCL user buffer** | PyTorch caching allocator 的普通显存 | `megatron/core/distributed/distributed_data_parallel_config.py:131-145`：`nccl_ub` 的目的写得很直白——"enables SM efficient nccl algorithm"，docstring 还附了一张 AG/RS 的 SM 占用表（NVL 4/5；NVL+IB 16/16，配 SHARP 降到 6/6；IB 1/4→1/1）。**第一判据是省 SM（把 SM 还给计算），显存收益是附带的** |
| MoE 专家激活怎么存 | 容量因子**预定尺寸** + 分页存放 | 每步按实际 token 数动态查询/重分配 | `docs/user-guide/features/paged_stash.md:16`：sync-free 的目的是从用户给定的容量 "pre-size dispatch and fused grouped expert buffers"，从而 "avoiding a **per-step device query / realloc loop** for buffer sizing"（即消除 CPU-GPU 同步）；分页本身则是 "to avoid memory waste due to **fragmentation**"（`docs/user-guide/features/paged_stash.md:34`，配置示例里的注释原话） |
| 容量估错了怎么办 | runner 级**整步重跑一次** | 丢 token，或按最坏 skew 常驻预留 | `docs/user-guide/features/paged_stash.md:20`：容量因子一旦设定，runner 就包住 forward-backward；overflow 或 over-budget 在**任一 rank** 命中即 "reruns once without capacity padding and without paged stashing"。判据：把**罕见**的路由尖峰用一次重跑吃掉，而不是为它长期占显存 |
| 激活换出的粒度 | **子模块级**（`--offload-modules`） | 层级 offloading | `docs/user-guide/features/fine_grained_activation_offloading.md:14`："**Unlike layer-level offloading**, it allows precise control over which activations to offload, enabling a tradeoff between memory savings and **PCIe bandwidth overhead**"——判据是让用户能在显存与 PCIe 之间取点，而不是一刀切 |
| 优化器状态怎么搬 | **分块**（chunk）恢复/预取 | 整体 offload → 整体 reload | `megatron/core/optimizer/cpu_offloading/README.md:17-21` 与 `:30-33`：正的 chunk size "bounds that tensor-state window"，而 `0` "**does not bound** the temporary tensor-state peak"。判据是**临时峰值有界**，不是搬得最多 |

**贯穿五条的判据是同一条**：显存优化的价值不在"省了多少"，而在"省的那部分是否换掉了更稀缺的资源"——SM、CPU-GPU 同步点、PCIe 带宽、峰值 headroom。这也是为什么 §4 的组合拳按"开销递增"分四层，而不是按"节省量递减"。

> [!note] 推断
> 以下两点是本页依据源码结构重建的，**源码并未陈述**：
> - **为什么不做统一的显存预算器**：这些机制的开关彼此独立、互斥关系散落在 `TransformerConfig.__post_init__` 的十几处 assert 里（见 §6），说明它们是分别演进而来的，而非从一个总设计切分下来。源码没有解释为何不收敛成统一入口。
> - **§5 配置决策表的分档依据**：那张按模型规模推荐组合的表是本页早期的归纳，源码与官方文档都没有给出按参数量分档的建议；`docs/user-guide/features/paged_stash.md:53-59` 给的是"按实测 routing skew 分布选容量因子"这种**基于 profile 的**方法，与按规模查表不是一回事。

## 3. 各显存优化技术详解

### 3.1 NCCL Memory Pool (`megatron/core/nccl_allocator.py`)

**解决什么问题**：通信 buffer 的显存碎片化和 NCCL 对称内存注册开销。

**机制**（`megatron/core/nccl_allocator.py:51-58`）：使用 `ncclMemAlloc` / `ncclMemFree` 替代 PyTorch 默认分配器：

```python
nccl_mem = torch.cuda.MemPool(ncclMemAlloc, ncclMemFree)
```

**关键优化**：
- `MultiGroupMemPoolAllocator`（`:276`）：同一 pool 被 FSDP 和 EP 多组共享，避免为不同通信组分配重复 buffer
- `NCCL_NVLS_ENABLE=1`（`:154`）：启用 NVLink SHARP 硬件 offload，减少通信延迟
- Warmup barrier 预先建立 NCCL 通信 buffer，避免首次通信时分配

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> **NCCL UB 内存池：正确反注册 + 降显存（#4492，e35d4e50c）**
> 1. **退出时反注册 NCCL user-buffer 池**（`megatron/training/training.py:5004-5017`）：训练满足退出条件时，先 `torch.distributed.barrier()`，再遍历每个 `DDP` 的 `buffers + expert_parallel_buffers`，对带 `nccl_mem_pool` 的 buffer 调用 `nccl_allocator.deregister_mem_pool(...)`。不这样做的话，`ProcessGroupNCCL` 析构会走 `abort()` → 对 `ncclCommWindowRegister` 注册过的 handle 调 `ncclCommDeregister`，报 `NCCL WARN Deregister: Could not find handle` 并崩溃。
> 2. **每个 buffer 记录自己的池**（`megatron/core/distributed/param_and_grad_buffer.py:1230,1257`）：`_ParamAndGradBuffer.nccl_mem_pool` 字段保存 `create_nccl_mem_pool(...)` 返回的池句柄，供上面反注册时取用。
> 3. **torch≥2.11 放宽 `expandable_segments` 限制**（`megatron/core/distributed/distributed_data_parallel_config.py:300`）：`nccl_ub` 与 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 互斥的断言改为 `if self.nccl_ub and not is_torch_min_version("2.11.0a0")` —— 新版 torch 上两者可共存，降低显存碎片。

**适用条件**：多机训练（NCCL 通信密集）、FSDP/EP 混合使用

### 3.2 MoE Paged Stash (`megatron/core/transformer/moe/paged_stash.py`)

> [!warning] 读序提示：本节正文里的**"三级溢出"画法（Tier 1/2/3）在基线 `71092579` 下已被证伪** —— 真实结构是 **2 级缓冲（CUDA 页 / pinned host 页）+ 1 级 runner 级整步重跑**，且 host spill 是成功的回退、不触发重跑。请先读本节下方的 [!deprecated] 与 [!update]，再把下面这张图与"三级溢出处理"当作旧基线的记录看。

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

**生命周期**（状态字段 `:413`，状态机推进 `:906-909`）：
1. **begin**：第 1 次迭代，idle 观察
2. **capture**：第 2 次迭代，记录 max tokens per (dtype, hidden_size)，建立 page schedule
3. **captured**：使用 CUDA Graph 加速 stash/reload

**三级溢出处理**（原 Triton kernel `megatron/core/transformer/moe/paged_stash.py:130` 已迁走，现为 `megatron/core/transformer/moe/ops/paged_stash.py:13`）：
- Tier 1（快速路径）：页在 CUDA stash buffer 内
- Tier 2（CPU 溢出）：CUDA buffer 满但 pinned host buffer 可用
- Tier 3（重运行）：都满了 → 触发 `PagedStashRunner.rerun()` 回退到无 stash 重新执行

> [!deprecated] 2026-06-16：上面"三级溢出"的画法把**两个不同层级的机制**混为一谈，按 #4247 实测应拆成 **"2 级缓冲 + 1 级 runner 级重跑"**（详见下方 [!update]，源码 `megatron/core/transformer/moe/ops/paged_stash.py:13`、`megatron/core/transformer/moe/paged_stash.py:1190`）。**2026-08-28 重核：`paged_stash.py` 与 `ops/paged_stash.py` 在 `232c478d4` 与 `71092579` 之间逐字节相同（`git diff` 空），本条及下方 [!update] 里这两个文件的全部行号原样成立。**具体更正：
> - **Tier 2 的"host spill"不是溢出，而是成功的回退**。Triton kernel 注释原话："`host_spill` = 1 if any successful host spill (**not set on overflow path**)"（`megatron/core/transformer/moe/ops/paged_stash.py:25`）。CUDA 页用尽时正常往 pinned host 页写，只置 `host_spill` 标志（信息性日志，提示你调大 `factor_cuda`），**不触发重跑**。
> - **真正的 overflow = CUDA 页与 host 页都满** → kernel 置 `overflow` 标志（`megatron/core/transformer/moe/ops/paged_stash.py:98`）。这才是触发重跑的条件之一。
> - **"重跑"不是 kernel 里的 Tier 3，而是 runner 级（`PagedStashRunner`）的整步 forward-backward 重跑**，且条件是 **stash overflow `或` HybridEP 容量 over-budget（任一 rank 命中）**，并非"都满了"。重跑会**同时**清掉 `moe_expert_rank_capacity_factor` 容量 padding **和** `moe_paged_stash`，最多重跑 1 次（共 2 次尝试）。

**配置参数**：
```python
moe_paged_stash_page_size: int = 64           # 页面大小
moe_paged_stash_buffer_size_factor_cuda: float = 1.10  # GPU buffer 比例
moe_paged_stash_buffer_size_factor_cpu: float = 0.0    # CPU buffer 比例
```

**适用条件**：MoE 训练（expert 数量 ≥ 8，capacity factor 大时收益最高）。与 `cpu_offloading` 不兼容。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> **Paged Stashing 正式合入（#4247，f007db77b）+ Triton kernel 搬迁（#5003，b8fef119c）—— 对上文的权威订正与补全**
>
> 官方文档 `docs/user-guide/features/paged_stash.md` 把这个特性定义为 **"sync-free 专家执行 + paged stashing"** 两部分，比上文"只是 MoE 激活分页"更完整：
>
> **(1) 前置依赖（不是可选项）**：`moe_paged_stash` 要求先开 **sync-free**——`--moe-flex-dispatcher-backend hybridep` + `--use-transformer-engine-op-fuser` + `--moe-expert-rank-capacity-factor <float>`（`megatron/core/transformer/transformer_config.py:2265-2279` 校验，且后端取值已扩为 `"hybridep"` 或 `"ncclep"`）。`moe_paged_stash` 自身校验（`megatron/core/transformer/transformer_config.py:2574-2588`，冲突模块集合已扩为 `expert_fc1`/`moe_act`/`fused_group_mlp`）：必须设容量因子、不能与 `cpu_offloading` 共存、`offload_modules` 不能含 `expert_fc1`/`moe_act`。容量因子 = "相对完美均衡情况的 buffer 倍率"，留 headroom 吸收路由 skew。
>
> **(2) 缓冲是 2 级而非 3 级**：`PagedStashBuffer`（`megatron/core/transformer/moe/paged_stash.py:25`）维护两条 free list：`[0]=CUDA 页`、`[1]=pinned host 页`。host 页仅当 `moe_paged_stash_buffer_size_factor_cpu>0` 时分配（`num_tokens_host>0`）。pack/reload 由融合 Triton kernel `paged_stash_copy_kernel`（**已由 #5003 搬到 `megatron/core/transformer/moe/ops/paged_stash.py:13`**，原 `megatron/core/transformer/moe/paged_stash.py:130` 引用失效）一次 launch 完成，无 host 同步、CUDA-graph 安全。
>
> **(3) overflow → runner 级整步重跑**：`PagedStashRunner.__call__`（`megatron/core/transformer/moe/paged_stash.py:1190,1149`）包裹 `forward_backward_func`：每趟跑完用 `check_moe_overflow()`（`:1295`）把 **stash_overflow / over_budget / host_spill** 三个标志 `all_reduce(SUM)` 跨全 rank 汇总；只要 stash overflow 或 over-budget 在**任一 rank** > 0 → `prepare_for_rerun()`（`:1357`）：清容量因子（转为 dropless 动态派发，靠 CPU sync 定尺寸）、关 paged stash、`zero_grad`、重置 FullCudaGraph、释放 stash buffer，然后**重跑整步**；`num_tries < 2` 断言保证最多 2 次。host_spill>0 只打日志（建议调大 `factor_cuda`），不重跑。
>
> **(4) 生命周期三态精确化**（`megatron/core/transformer/moe/paged_stash.py:1162-1170` 在 `paged_stash_reset` 里推进）：`begin → capture → captured`。
> 
> [!update] 2026-09-01（基线 `85902ef59`）：这段不是纯行号漂移，结构被 #6022 拆开了。旧基线上「转成 `captured`」与「按 cuda/cpu 两个 factor 分配 stash buffer」写在同一个 `elif` 分支里；新基线把**状态推进**（`megatron/core/transformer/moe/paged_stash.py:1162-1165`）与**缓冲准备**（`:1167-1170` 的独立 `if status == 'captured'` 判定）分开，后者改调新增方法 `prepare_stash_buffers(config)`，两个 size factor 的读取下沉进该方法。**三态语义本身未变**，变的是「何时分配缓冲」从状态迁移的副作用变成了一个可被单独调用的步骤——这正是 TE whole-MoE CUDA graph 捕获流程所需要的。
> - `begin`：初始态，**首次 reset 即转 `capture`**（并无完整一迭代停在 begin）；
> - `capture`：本迭代**逐 (dtype, hidden_size) 累计 max tokens、并构建 `_pp_schedule`**（`:711-742`），此阶段**显式不使用 CUDA graph**（源码注释 `:744`，故可按实际 token 数截断保存张量）；下次 reset 时 `capture → captured` 并 `allocate_stash_buffers`；
> - `captured`：稳态，按 `_pp_schedule` 真正 stash/reload，**与 full CUDA graph 兼容**。注意上文"captured：使用 CUDA Graph 加速 stash/reload"措辞不准——CUDA graph 不是用来"加速 stash/reload"，而是 paged stash 被设计成 CUDA-graph-safe（freelist 原地 reset、无新分配）；rerun fallback 后 buffer 会被释放，再次进入 captured 时按 capture 期 maxima 重新分配（`:916-929`）。
>
> **(5) 配置项当前行号**（`megatron/core/transformer/transformer_config.py`，已重核至 `71092579`）：`moe_expert_rank_capacity_factor:997`、`moe_paged_stash:1495`、`moe_paged_stash_page_size:1498`、`moe_paged_stash_buffer_size_factor_cuda:1501`、`moe_paged_stash_buffer_size_factor_cpu:1507`。

### 3.3 Fine-Grained Activation Offloading (`megatron/core/pipeline_parallel/fine_grained_activation_offload.py`)

**解决什么问题**：一般 Transformer 层的激活值显存（非 MoE 特定）。

**机制**（`:400`）：`PipelineOffloadManager` 在 sub-layer 粒度选择性卸载：

```
支持的 offload 模块（`megatron/core/transformer/transformer_config.py:1464-1476`）：attn_norm, qkv_linear, core_attn, attn_proj, 
                          mlp_norm, expert_fc1, moe_act, fused_group_mlp
```

**Autograd 挂钩与执行流程**（`:400-440` 类初始化 + `:960-1059`）：`PipelineOffloadManager` 是单例管理器，通过 `saved_tensors_hooks` 拦截 autograd 的 save/retrieve 操作，在两条独立 CUDA stream 上做异步搬运：

```python
# megatron/core/pipeline_parallel/fine_grained_activation_offload.py:400-440
class PipelineOffloadManager:
    def __init__(self):
        self._d2h_stream = torch.cuda.Stream()  # GPU→CPU
        self._h2d_stream = torch.cuda.Stream()  # CPU→GPU
        self._cpu_tensor_pool = OffloadTensorPool()  # 复用 pinned CPU buffer
        self._is_warmup = True
        self._offload_margin = 0  # 保留在 GPU 上的 group 数
```

1. **Forward 阶段**：`on_save_for_backward(tensor)` 将 tensor 推入当前 forward chunk；当 layer group 完成时，`bulk_offload_group` 在 `_d2h_stream` 上执行异步 D2H 拷贝。
2. **Backward 阶段**：`on_get_saved_tensor(saved_state)` 从当前 backward chunk 弹出 tensor；如果已被 offload，`bulk_reload_group` 在 `_h2d_stream` 上执行异步 H2D 拷贝。

```python
# megatron/core/pipeline_parallel/fine_grained_activation_offload.py:960-1059
def tensor_push(self, tensor):
    tag = self._next_tag
    self._current_group.append((tag, tensor))
    return tag

def tensor_pop(self, tensor_tag):
    _, saved = self._groups[tensor_tag].pop(0)
    if isinstance(saved, tuple):  # offloaded
        return self.reload(saved)
    return saved

def bulk_offload_group(self, group_to_offload):
    for tag, tensor in group_to_offload:
        cpu_tensor = self._cpu_tensor_pool.allocate(tensor.shape, tensor.dtype)
        cpu_tensor.copy_(tensor, non_blocking=True)
        ...

def bulk_reload_group(self):
    for tag, (cpu_tensor, event) in self._pending_reload.items():
        gpu_tensor = torch.empty_like(cpu_tensor, device="cuda")
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)
        ...
```

**`OffloadTensorGroup` —— 一组激活 + 双 CUDA event 同步**：把一个模块的激活打包成 group，持有 `offload_event` / `reload_event` 两个 CUDA event。前向 D2H 拷完 `record_offload_event`；反向用之前 `wait_reload_event` 确保 H2D 拷回完成才用 —— 这是 D2H/H2D 传输与计算重叠、又不产生数据竞争的同步机制。

**Warmup 后的自适应调优**（`post_warmup_callback`，`:566-647`）：第一个 iteration（warmup）完成后，根据实际观测到的 tensor 大小和 group 分布计算最优 offload 策略：

```python
# megatron/core/pipeline_parallel/fine_grained_activation_offload.py:566-647
def post_warmup_callback(self):
    self._is_warmup = False
    # 计算 _offload_margin：保留在 GPU 上的 group 数，避免 reload 阻塞 compute stream
    self._offload_margin = max_deduplicated_groups
    # 平衡各 PP rank 的 offload bytes
    keep_on_gpu_bytes = total_bytes * activation_offload_fraction
    # 禁用最后几个同名 group 的 offloading（防止 reload 阻塞）
```

下方「关键设计」第 2 条的 `offload_margin` 正是由此在 warmup 后自适应算出，而非静态配置常量。

**关键设计**：

1. **OffloadTensorPool**（`:115`）：基于 deque 的 pinned CPU memory pool
   - Match by (shape, dtype) 复用，避免重复分配 CPU 内存
   
2. **选择性卸载**：
   - `min_offloaded_tensor_size`（判定 `:993`，默认值 `:691` 与 `megatron/core/transformer/transformer_config.py:1477`）：跳过小 tensor（默认 ≥1M elements）
   - `activation_offload_fraction`（`:693`，生效于 `:617`）：只卸载部分层
   - `offload_margin`（`:580-596`）：最后 N 层不卸载（避免 reload 阻塞 backward 计算）

3. **CUDA Graph 兼容**（`:1226-1234` 的 `delay_offload` 分支 + `:1295-1298` 的 flush 入口）：支持延迟卸载（`delay_offload_until_cuda_graph`，`megatron/core/transformer/transformer_config.py:1480`）

**性能开销**：D2H/H2D 异步传输（pinned memory + 独立 CUDA stream），与计算重叠

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> **勘误（2026-08-28）**："新增于 B" 的归属有误——该旋钮在**旧基线 `ee3f1ff` 下就已存在**（`megatron/core/transformer/transformer_config.py:1246`、`megatron/core/pipeline_parallel/fine_grained_activation_offload.py:668`），首次落地是 #4514（`994f5c9d6`，为 `ee3f1ff` 的祖先）；#4692（`9a7cd17fd`）是 dev 侧的**再次落地**，不是首次新增。机制描述本身不受影响。
> **"最大在途 offload"节流旋钮**：`fine_grained_offloading_max_inflight_offloads: Optional[int]`（`megatron/core/transformer/transformer_config.py:1498`，校验 `:2561-2564`）。语义——**按 offload group 名字**分别限制"已发起 D2H、但主流尚未 `wait_event` join 的 offload 个数"（每个名字一条独立 FIFO，同一上限）。`0` = 每次 offload 后主流立即等待该名字；`1` = 该名字最多 1 个未 join 的在途；`None`（默认）= 不插入这些 join。
> 实现：`ChunkOffloadHandler`（`megatron/core/pipeline_parallel/fine_grained_activation_offload.py:825`）的 `__init__` 增 `max_inflight_offloads` 参数（`:867`，落到 `:892`）与 `_offload_pending_by_name`（`:894`），commit offload 时入队事件并调 `_drain_offload_pending(name)`（定义 `:1112`，调用点 `:1025-1028`）让主流等待超额的旧事件。
> **动机**：在 **full-iteration CUDA graph** 捕获下，主流可能不会自动等 D2H 事件 → 在途 offload 无界堆积、pinned host buffer 与 D2H 队列爆显存/带宽；此旋钮给 CG 场景一个回压闸门。

**适用条件**：大模型（≥70B）、长序列、PP 并行。不能与 `cpu_offloading` 组合。

### 3.4 Optimizer State Offloading (`megatron/core/optimizer/cpu_offloading/optimizer_state_offloader.py`)

> [!deprecated] 2026-08-28：本节引用的 `megatron/core/optimizer/cpu_offloading/optimizer_state_offloader.py`（原 `:253` 的 `offload()`）在基线 `71092579` 下**已被整体删除**，替换实现是 #6244（`9050d4c5f`，"[dev] Add chunked optimizer-state and master-weight offload"）引入的 `megatron/core/optimizer/cpu_offloading/chunked_optimizer_state_offload.py`：`ChunkedOptimizerStateOffloader`（`:57`）把状态切成 `OptimizerStateChunk`（`:31`）分块搬运，接口从"整体 offload/reload/release"改成 `offload_for_forward()`（`:808`）+ `prefetch_for_step()`（`:785`）/ `prefetch_master_for_step()`（`:794`）的分块预取，并保留双传输流（`transfer_streams`，`:290`）。目录内 `hybrid_optimizer.py`（`HybridDeviceOptimizer`，`:14`）仍在。**以下时序图与结论对应旧基线 `ee3f1ffa2acd18131ab67cabab4cec45283512ab`**（节省量级与适用条件在新实现下不变，但函数名/行号不再可核验）。

**解决什么问题**：优化器状态（exp_avg, exp_avg_sq）和 FP32 主权重的 GPU 显存。

**机制**（旧基线 `:253`）：
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

### 3.5 Parameter And Gradient Buffer 复用

`megatron/core/distributed/param_and_grad_buffer.py` 中的多重复用：

**MXFP8 共享 Buffer**（`:1302-1321`）：
```
param all-gather ─────┐
                      ├── shared_buffer（同一物理显存）
grad reduce-scatter ──┘
```
节省一份完整的 buffer（1x param size）。

**NVFP4 双 Buffer 布局**（`:1159-1184`）：
- 参数 Buffer: N/2 bytes（每字节 2 个 FP4 值）
- 梯度 Buffer: N×2 bytes（BF16 全精度）
- 比全精度方案节省 75% 参数 buffer

**Grad Buffer 复用为 Param All-Gather Buffer**（`:534-546`）：
```python
reuse_buf = bucket.grad_data.view(param_dtype)  # recycle
```
梯度 reduce-scatter 完成后，buffer 被清零并重用于参数 all-gather。

### 3.6 FP8/FP4 参数精度显存节省

| 方案 | 参数显存节省 | 硬件要求 |
|------|------------|---------|
| BF16（基准） | 0% | 任何 GPU |
| FP8 E4M3 | 50%（2 bytes → 1 byte） | H100+ |
| MXFP8 | 50% + block scaling | GH200, Blackwell |
| NVFP4 | 75%（2 bytes → 0.5 byte） | Blackwell |

**首尾层保护**（`megatron/core/fp8_utils.py:698-714`）：`first_last_layers_bf16` 保持首尾 N 层为 BF16，防止精度损失在输入/输出端被放大。

### 3.7 CUDA Graph Buffer 复用 (`megatron/core/transformer/cuda_graphs.py`)

> [!deprecated] 2026-08-28：`TensorReusePool` 这个类在基线 `71092579` 下**已不存在**——`git grep TensorReusePool 71092579 -- megatron/` 无任何命中；它是被 #5451（`5e4fe9b3c`，"Optimize memory usage of partial CUDA graphs"，`cuda_graphs.py` 净删 369→改写）移除的。以下"按 (shape, dtype, device) 匹配复用 + 跨 stage 共享"的池式描述对应旧基线 `ee3f1ffa2acd18131ab67cabab4cec45283512ab`（`:161`）。当前实现改成**引用计数式的 buffer 复用**：`cg_buffer_metadata` 上的 `cudagraph_reuse_ref_count` / `capture_reuse_count`（`megatron/core/transformer/cuda_graphs.py:153-161`）配合全局 `fwd_buffer_reuse_ref_count` / `bwd_buffer_reuse_ref_count`（`:345-346`），在捕获期决定某个 buffer 能否被下一张图直接复用或写入（`:870-911`）。

旧基线下的 `TensorReusePool`（`:161`）复用 CUDA Graph 的输入/输出 buffer：
- 按 (shape, dtype, device) 匹配复用
- 跨 pipeline stage 共享（后续 stage 的输入 = 前序 stage 的输出）
- TE weak references（现 `:405-414` 的 `HAVE_TE_GRAPHS` 警告块，旧基线 `:384-392`）允许更激进的 buffer 回收

### 3.8 Rerun State Machine (`megatron/core/rerun_state_machine.py`)

间接显存优化：允许使用更快但可能不稳定的计算路径（如特殊数值模式）。检测到结果异常（NaN, spiky loss）时，通过重运行恢复。

### 3.9 Sequence Parallelism 的隐性显存收益

按 `tp_size` 切分序列维度，每个 rank 的 Attention 激活值仅为 `1/tp_size`。对于 128K 序列、TP=8 的场景，Attention 激活从 ~8GB 降至 ~1GB/rank。

### 3.10 Resharding (`resharding/`)

`megatron/core/resharding/nvshmem_copy_service/memory/double_buffer_manager.py` 实现双缓冲，在并行策略动态切换时重叠通信与计算，减小过渡期的临时 buffer 显存。

### 3.11 dev 增量显存项（2026-05~06）

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> 一批小而实的显存相关改动：
>
> **(a) 提前 `del` 输出张量（#4742，f1b5516b2）**：`forward_backward_no_pipelining` 在每个 microbatch `backward_step` 后立即 `del output_tensor`（`megatron/core/pipeline_parallel/schedules.py:787,810`）。不删的话上一个 microbatch 的 `output_tensor` 会一直活到下次迭代重新绑定变量，把 autograd 节点的析构推迟到下一次 forward 的 dispatch 路径上，触发 PyTorch "AccumulateGrad node's stream does not match" 警告（issue #4124）；提前删让 autograd 计算图头部及时释放。
>
> **(b) 删除 checkpoint 期的 GPU cache 回收 workaround（#5170，3a183e235）**：`save_checkpoint_and_time` 里原先为给异步 checkpoint worker 腾 D2H 显存而做的 `free_overlap_buffers()` + `torch.cuda.empty_cache()`（`megatron/training/training.py:2957-2963` 附近）已被**移除**（7 行）。该 workaround 不再需要。
>
> [!contradiction] 2026-08-28：(b) 在基线 `71092579` 下**已不成立——该 workaround 又被加回来了**。#5366（`be82829c3`，标题即 `Revert "Remove checkpoint-time GPU cache reclaim workaround (#5170)"`）把它整段还原：现在 `save_checkpoint_and_time`（`megatron/training/training.py:3882`）在 `timers('interval-time').stop()` 之后仍然遍历 `model` 调 `free_overlap_buffers()` 并 `torch.cuda.empty_cache()`（`:3906-3912`，注释原文 "Free overlap param-gather buffers and release cached GPU memory so that the async checkpoint worker process has enough GPU headroom for D2H tensor transfers."）。即 #5170 的"不再需要"判断被上游自己推翻了。
>
> **(c) 权重/优化器显存估算正确计入 EP（#4687，a7c9e8c44）**：`megatron/training/theoretical_memory_usage.py:13` 重写 `compute_weight_and_optimizer_memory`，把 **routed expert 参数**单独按 `expert_tensor_parallel_size × expert_model_parallel_size` 分片、按 `expert_data_parallel_size = world_size /(etp×ep×pp)`（`:223`，使用于 `:304`）计 optimizer 字节，并区分 shared_expert / router / shared_expert_gate / active vs total 专家参数。修复了此前 MoE+EP 下理论显存被高/低估的问题（纯估算工具，不改运行时显存）。
>
> **(d) 修复 FSDP double-buffer 的 CUDA IMA（#4810，d199bb9e9）**：`megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py:745`（另一处同类分配器在 `:1001`）把 `FixedPoolAllocator.backup_allocator` 从 `TemporaryBucketAllocator()` 改为 `StorageResizeBasedBucketAllocator()`。当某个 FSDP unit 的 bucket 装不进固定池、回退到 backup 分配器时，旧实现会越界访问（IMA）；改用基于 storage resize 的分配器修正。

## 4. 显存优化组合拳

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

## 5. 配置决策表

| 模型规模 | GPU 配置 | 推荐显存优化组合 |
|---------|---------|----------------|
| 10B MoE | 8×A100 | 分布式优化器 + Sequence Parallelism |
| 50B MoE | 16×H100 | + FP8 参数 + Activation Checkpointing |
| 200B MoE | 64×H100 | + Grad Buffer 复用 + Paged Stash + FP8 Input Store |
| 671B MoE | 256×H100 | + NCCL Pool + Fine-Grained Activation Offloading |
| 1.xT MoE | 512×Blackwell | + NVFP4 + Optimizer State Offloading + 全量 Overlap |

## 6. 约束

每一项都要付代价，且大多数组合会在构造期被 assert 挡住。以下全部按基线 `71092579` 实读。

### 6.1 Paged Stash：前置依赖是硬的，不是可选项

- 必须先具备 **sync-free** 三件套：`--moe-flex-dispatcher-backend hybridep`（或 `ncclep`）+ `--use-transformer-engine-op-fuser` + `--moe-expert-rank-capacity-factor`（`megatron/core/transformer/transformer_config.py:2265-2279`）。
- `moe_paged_stash` 自身：必须设容量因子（`megatron/core/transformer/transformer_config.py:2576-2579`，报错文案"there is no need to use paged stashing without it"）、**不能与 `cpu_offloading` 共存**（`:2574-2575`）、`offload_modules` 不能含 `expert_fc1`/`moe_act`/`fused_group_mlp`（`:2580-2588`，理由是 paged stash 已经覆盖了这些激活）。
- 重跑上限是 **1 次**（共 2 次尝试，见 §3.2 的 [!update] (3)）。容量因子给小了不是"慢一点"，是**每步都多跑一遍 forward-backward**。

> [!update] 2026-09-01（基线 `85902ef59`，#6022）：当 paged stash 被用来支撑 **TE whole-MoE CUDA Graph** 时，前置依赖再加一层，且**失败方式从降级变成硬报错**。
>
> - 判定"是否在捕获整个 MoE 模块"由 `is_whole_moe_cuda_graph_scope()` 给出（`megatron/core/transformer/cuda_graph_config.py:25`）：空 scope（默认整层捕获）或显式 `CudaGraphModule.moe` 都算。
> - 成立时 `validate_moe_cuda_graph_support()`（`:35`）断言六项同时满足（`:49-55`），其中就包含 `moe_paged_stash`——**paged stash 由"省显存的可选项"变成 whole-MoE 图的必要条件**。
> - `paged_stash.py` 为 TE 捕获期新增了一整套调度切换：`start_te_graph_capture` / `finish_te_graph_capture`、`_build_te_graph_capture_schedule(order)`（把 TE 的 chunk 级顺序展开成 capture-only 的分页层条目）、上下文管理器 `paged_stash_te_graph_capture(...)`、`mark_te_graph_captured(num_microbatches)`。捕获期与运行期用的是**两份不同的调度**，捕获结束后再切回全局批调度。
> - 溢出不再回退：`_raise_if_te_whole_moe_graph_overflow(...)` 在已捕获的图上溢出静态缓冲时直接抛错（文案 "Dynamic fallback is not supported for an already captured TE whole-MoE graph."）。**本节开头"容量因子给小了只是多跑一遍 forward-backward"这条，在 whole-MoE 图打开时不成立——那时它是直接失败。**
>
> 机制侧详见 [[23_megatron_precision_cudagraph_fusion_analysis]] §8.5。

### 6.2 Fine-Grained Activation Offloading：模块之间有依赖，不能随便挑

- 只支持 `--transformer-impl transformer_engine`（`megatron/training/arguments.py:2132-2135`）。
- **`attn_proj` 不能单独选**：它的输入是 `core_attn` 的输出，而 `core_attn.backward()` 需要这个张量，所以必须与 `core_attn` 同选（`megatron/core/transformer/transformer_config.py:2535-2540`）。
- `fused_group_mlp` 要求 `use_transformer_engine_op_fuser`，且**不能**与 `expert_fc1`/`moe_act` 组合——它已经整块换出了融合后的 grouped MLP（`megatron/core/transformer/transformer_config.py:2565-2573`）。
- 数值域：`min_offloaded_tensor_size ≥ 0`、`activation_offload_fraction ∈ [0,1]`、`delta_offload_bytes_across_pp_ranks ≥ 0`、`fine_grained_offloading_max_inflight_offloads ≥ 0`（`megatron/core/transformer/transformer_config.py:2552-2564`）。
- 与整层 `moe` 重计算互斥（`megatron/core/transformer/transformer_config.py:2541-2551`）；与 `cpu_offloading` 不能组合（见 §3.3 末）。

### 6.3 NCCL User Buffer：省 SM 的代价是常驻显存

- UB 注册要求**持久缓冲池**：`fsdp_double_buffer` 在 `nccl_ub=True` 时被自动打开，docstring 直说 "Persistent buffers add memory overhead but are **required for** NCCL user-buffer registration"（`megatron/core/distributed/distributed_data_parallel_config.py:147-153`；自动开启点 `megatron/training/arguments.py:1274-1280`，同时强制 `fsdp_manual_registration=True`）。
- 缓冲个数默认 2；**combined 1F1B overlap 可能需要 3**——因为"一个反向/重算单元、当前前向单元、以及被前向预取的后继单元"可能同时在世（`megatron/core/distributed/distributed_data_parallel_config.py:155-161`）。这是通信重叠反噬显存的一个具体面。
- `nccl_ub` 与 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 在 **torch < 2.11** 上互斥（`megatron/core/distributed/distributed_data_parallel_config.py:300-304`）——即"降碎片"与"省 SM"在旧 torch 上二选一。
- 退出时必须显式反注册，否则 `ProcessGroupNCCL` 析构会崩（见 §3.1 的 [!update] 第 1 条）。

### 6.4 优化器状态 offload：分块换来的是峰值有界，不是全都能用

按 `megatron/core/optimizer/cpu_offloading/README.md`：
- **fraction 非 0 时，full-iteration CUDA graph 与 optimizer CUDA graph 都不支持**（`:38-39`）。这条直接和 §3.7 / [[23_megatron_precision_cudagraph_fusion_analysis]] 的 CUDA Graph 路线冲突。
- 保存优化器状态时**拒绝异步分布式 checkpoint 保存**（`:39-40`）；必须用分布式 checkpoint，legacy torch checkpoint 路径"can reconstruct the full state on CUDA and is **not supported**"（`:86-96`）。
- fraction 是**近似**字节比例：一个参数和它的全部优化器状态是**原子 bundle**，不能拆（`:33-36`）。
- chunk size = `0` 时**不限制**临时张量峰值，argument 校验只发 warning（`:30-32`）；且任何 chunk size 都**不**限制被选中的 master weight 那一整个窗口（`:93-95`）。

### 6.5 全局互斥面

- `cpu_offloading` 与 **PP > 1** 互斥（`megatron/core/transformer/transformer_config.py:2288-2291`），与**任何**激活重计算互斥（`:2293-2296`，见 [[18_megatron_recompute_analysis]] §7.3）。
- §5 那张按模型规模的推荐表里，"Optimizer State Offloading + 全量 Overlap"这一格需要先过 §6.4 的 CUDA Graph 限制；表本身不含这些前置校验。

## 7. 发展趋势

每条都锚定基线 `71092579` 上实读到的痕迹（revert、改名、`TODO`、版本门槛）。**"方向"是本页的推断，源码只陈述锚点本身。**

| 锚点（实读） | locator | 推断的方向 |
|---|---|---|
| **上游自己推翻了自己**：#5170 删掉的 checkpoint 期显存回收 workaround 被 #5366（`be82829c3`，标题即 `Revert "Remove checkpoint-time GPU cache reclaim workaround (#5170)"`）整段还原 | `megatron/training/training.py:3882` / `:3906-3912`（详见 §3.11 的 [!contradiction]） | 异步 checkpoint worker 的 D2H headroom 问题**没有被真正解决**，只是把 workaround 放回去了；后续大概率还会有一次结构性修复（例如把 D2H 缓冲纳入统一的显存预算），而不是继续 `empty_cache()` |
| 优化器状态 offload 整体重写：旧 `optimizer_state_offloader.py` 删除，换成 `chunked_optimizer_state_offload.py` 的分块预取（#6244） | `megatron/core/optimizer/cpu_offloading/chunked_optimizer_state_offload.py:31`/`:57`/`:785`/`:794`/`:808`（详见 §3.4 的 [!deprecated]） | 从"整体搬运"转向"**峰值有界的分块搬运**"；这条路线的下一个约束是 §6.4 那条"CUDA graph 不兼容" |
| 旧旋钮 `--offload-optimizer-states` 仅保留为 parser 兼容拼写，校验期发 `FutureWarning` 并改写成新模式 | `megatron/core/optimizer/cpu_offloading/README.md:98-101` | 旧接口在一个可见的弃用窗口内，迁移完成后会移除 |
| `TensorReusePool` 被 #5451 删除，图间缓冲复用改为**引用计数** | `megatron/core/transformer/cuda_graphs.py:150`/`:160-161`/`:345-346`（详见 §3.7 的 [!deprecated]） | 从"池 + (shape,dtype,device) 匹配"转向"按图的生命周期精确判定"，方向是让部分图（partial CUDA graph）的显存开销随捕获范围线性缩小 |
| 细粒度 offload 管理器里留着一处未验证的终止条件：`# TODO: check if this is correct` —— `ChunkOffloadHandler.finish_all_groups` 在"无待 reload、无待 offload 且 `_offloaded_group_index > 0`"时直接判定完成 | `megatron/core/pipeline_parallel/fine_grained_activation_offload.py:925-932` | 这段判定与 §3.3 的 `offload_margin` / 在途节流是同一套状态机；作者自己标注不确定，说明该状态机的收敛条件仍在打磨 |
| `nccl_ub` 与 `expandable_segments` 的互斥被改成**版本门槛** `torch >= 2.11.0a0` | `megatron/core/distributed/distributed_data_parallel_config.py:300-304`（详见 §3.1 的 [!update] 第 3 条） | "降碎片"与"省 SM"从二选一变成可共存；随 torch 版本推进，这条 assert 最终会消失 |
| Paged stash 把 MoE 的 dispatch 形状变成静态，官方 CUDA Graph 文档因此给出了 **MoE + full-iteration CUDA graph** 的组合配方 | `docs/user-guide/features/cuda_graph.md:170-181` | 显存优化（paged stash）与吞吐优化（整步图）在此合流：容量因子预定尺寸这一步，同时买到了"省显存"和"可图化"两个收益，见 [[23_megatron_precision_cudagraph_fusion_analysis]] §4.3 |

> [!note] 推断
> 右栏是本页依据锚点做的外推，不是上游的路线声明。引用时请回到左栏 locator。

## Related Pages

- [[16_megatron_distributed_optimizer_analysis]]
- [[21_megatron_fusion_operators_analysis]]
- [[20_megatron_comm_overlap_analysis]]
- [[12_activation_checkpointing_analysis]]
- [[18_megatron_recompute_analysis]]
- [[13_low_precision_training_analysis]]
- [[17_megatron_parallelism_orchestration_analysis]]
