# 通算融合（Compute-Communication Co-Fusion）完全指南

> 从手动调优到编译器自动化：WaveEP、DeepEP 与通算自动编译的演进路线
> 最后更新: 2026-05-12（2026-07-31 补边界声明）

> **与"计算通信掩盖"的边界**：本页讲的是把通信与计算编进**同一个 kernel**的**融合**（如 WaveEP 的 wave-tile 绑定、DeepEP `FusedDispatch`、MC2 的 `npu_all_gather_base_mm`）——源码层已不存在独立的两次算子调用。仅通过调度/多 stream 让**各自独立**的通信与计算并发执行、互相隐藏延迟的**掩盖**手段，见 [[30_comm_compute_overlap_analysis]]。

---

## 1. 核心问题：通信在关键路径上

大规模分布式训练/推理中，通信（AllReduce、AllGather、All-to-All）往往占据 **30%~50%** 的 step 时间。

**MoE 模型的通信密度尤为突出**：

```
每个 MoE layer 的执行流（串行）：

  ┌─────────────────────────────────────────────────────────────┐
  │ Dispatch All-to-All  (80-150μs, IB 网络)                    │
  ├─────────────────────────────────────────────────────────────┤
  │ Expert GEMM Compute  (150-300μs, GPU 计算)                  │
  ├─────────────────────────────────────────────────────────────┤
  │ Combine All-to-All   (80-150μs, IB 网络)                    │
  └─────────────────────────────────────────────────────────────┘
  总耗时: ~400-600μs  ←  其中通信占 ~40-50%，全部在关键路径上
```

**目标**：将通信从关键路径上移除，使其与计算并行执行。

---

## 2. 通算融合的四个层次

```
层次 0：手动设计（今天生产主流）
层次 1：半自动 Pass（今天研究前沿）
层次 2：框架感知自动融合（近期方向）
层次 3：全自动通算编译（中期目标）
```

### 层次 0：手动设计

**代表技术**：
- Megatron-LM `--tp-comm-overlap`（TP AllGather/ReduceScatter 与 GEMM 重叠）
- DeepSeek V3 **DualPipe**（PP 维度 1F1B 调度，bubble ratio 降至 1/PP × 1.x）
- DeepSeek V4 **WaveEP**（EP 维度 wave-based 细粒度调度）

**特点**：针对特定并行拓扑和硬件手动调优，换硬件或调整并行策略需重新设计。

### 层次 1：半自动 Pass

**代表技术**：
- `torch.compile` `micro_pipeline_tp_pass`（TP 通算重叠自动插入）
- `torch.compile` `fuse_ddp_communication`（DDP AllReduce 融合）
- `torch.compile` `bucket_all_gathers`（集合通信分桶）

**半自动的含义**：编译器在已知并行策略（TP/DP）的情况下，自动在图中插入 async 通信并安排与计算的重叠，但仍需用户标注并行维度。

### 层次 2：框架感知自动融合

**代表技术**：
- **Alpa（2022）**：ILP 自动搜索算子分区策略 + 通信插入
- **XLA GSPMD**：给 tensor 标注 sharding 注解 → 自动插入 AllGather/ReduceScatter
- **torch DTensor + torch.compile**（演进中）

**工作原理（GSPMD 示例）**：

```python
# 用户只需标注 sharding 策略
x = jax.device_put(x, NamedSharding(mesh, P("data", "model")))

# 编译器自动：
# 1. 分析哪些算子需要通信
# 2. 插入 async AllGather（提前启动, 不阻塞后续计算）
# 3. 安排通信与相邻层计算的重叠
```

### 层次 3：全自动通算编译（研究中）

目标：**无需任何并行注解**，编译器从模型描述自动生成最优通算调度。

当前最大挑战：
- 通信延迟是 non-deterministic（受网络拥塞影响）
- MoE 路由是动态的（token 分配 runtime 才知道）
- 3D/4D 并行（TP×DP×PP×EP）的联合优化是 NP-hard 问题

---

## 3. WaveEP：DeepSeek V4 的细粒度 EP 通算融合

### 3.1 核心思路

标准 MoE EP（串行）：

```
Wave 0: [Dispatch A2A] → [Expert GEMM] → [Combine A2A]
        ←────────────────────────────────────────────→ 全部完成才处理 Wave 1
```

WaveEP（流水线）：

```
将 tokens 分成 W 个 wave，每 wave 独立处理，相邻 wave 流水线重叠：

时间轴 →
Wave 0: [Dispatch A2A₀]──[Expert GEMM₀]──[Combine A2A₀]
Wave 1:                 [Dispatch A2A₁]──[Expert GEMM₁]──[Combine A2A₁]
Wave 2:                                 [Dispatch A2A₂]──[Expert GEMM₂]──...

稳态（W足够大）：
  Expert GEMM₀ 运行时，Dispatch A2A₁ 同步在后台进行
  → A2A 通信被完全隐藏在 Expert GEMM 后面
```

### 3.2 实测性能（DeepSeek V4）

| 场景 | 加速比 |
|------|--------|
| 一般推理 | **1.50~1.73×** |
| RL rollout（延迟敏感）| **高达 1.96×** |

RL rollout 加速更高的原因：rollout 的 token 批量较小，通信延迟占比更高，隐藏通信的收益更大。

### 3.3 工程实现要点

**CUDA Stream 架构**：

```
Stream 0 (Compute Stream):
  Expert GEMM Wave 0 → Expert GEMM Wave 1 → Expert GEMM Wave 2 → ...

Stream 1 (Comm Dispatch Stream):
  A2A Dispatch Wave 0 → A2A Dispatch Wave 1 → A2A Dispatch Wave 2 → ...

Stream 2 (Comm Combine Stream):
  A2A Combine Wave 0 → A2A Combine Wave 1 → A2A Combine Wave 2 → ...

CUDA Event 同步：
  Dispatch A2A₀ 完成 → event signal → Compute Stream 开始 GEMM₀
  GEMM₀ 完成 → event signal → Combine A2A₀ 可以执行
  Dispatch A2A₁ 在 GEMM₀ 运行时已经在 Stream 1 上进行
```

**Wave 粒度的权衡**：

```
Wave 太大（接近全量 tokens）：
  → 退化为串行，A2A 无法被 GEMM 隐藏
  → 通信占比高时几乎无收益

Wave 太小（如 8 tokens）：
  → GEMM 矩阵太小，GPU 利用率下降
  → A100/H100 的 GEMM peak 需要至少 512+ tokens 才能打满
  → Host launch overhead（若无 TileLang）成为瓶颈

最优 wave 粒度 ≈ 使 Expert GEMM 保持 >80% GPU 利用率的最小 token 数
                 通常在 256~512 tokens/wave 范围内
                 具体取决于 hidden_dim 和 expert_dim
```

**DeepEP 自定义 A2A kernel**：

DeepSeek V4 使用自研 **DeepEP** 而非标准 NCCL，原因：

| 问题 | NCCL AlltoAll | DeepEP |
|------|--------------|--------|
| SM 分配 | 固定，不可调 | fine-grained SM control（可调分配给 comm 的 SM 数） |
| 与 GEMM 的 SM 竞争 | 存在（NCCL 抢占 SM）| 可精确控制，避免竞争 |
| Async 控制 | 有限 | 完全异步 + 精确 event 控制 |
| Dispatch+Permute 融合 | 分离 | Permute + A2A + Unpermute 一体融合 |

### 3.4 WaveEP 的编译化路径（未来方向）

当前 WaveEP 是手动实现，针对 H800 集群调优。将其编译化需要：

```
Step 1: Wave 调度的 IR 表示
  在图 IR 中引入 wave 概念：
    moe_dispatch(tokens[wave_i])  depends_on  route(tokens[wave_i])
    moe_gemm(tokens[wave_i])      depends_on  moe_dispatch(tokens[wave_i])
    moe_combine(tokens[wave_i])   depends_on  moe_gemm(tokens[wave_i])
  
  相邻 wave 之间：
    moe_dispatch(wave_i+1) 可以与 moe_gemm(wave_i) 并行

Step 2: Cost Model
  latency(A2A, chunk_size, EP_size, IB_bw) → 延迟预测
  throughput(expert_GEMM, token_count, hidden_dim) → 吞吐预测
  → 求解：最优 wave_size = argmin(critical_path_length)

Step 3: TileLang 绑定
  将 wave_size 与 GEMM tile 大小绑定
  Expert GEMM tile = wave_size × head_dim / num_tiles_per_expert

Step 4: 与 DTensor 集成
  用户标注 EP sharding → 编译器自动生成 WaveEP 调度
```

---

## 4. DeepEP：MoE A2A 通信的专用融合

DeepEP（Deep Expert Parallelism）是 Megatron-LM 集成的 EP 专用通信库：

### 4.1 FusedDispatch

```python
# 标准 MoE dispatch（三步）：
permuted = permute(tokens, dispatch_idx)    # CPU/GPU 重排
a2a_result = all_to_all(permuted, ep_group) # All-to-All 通信
unpermuted = unpermute(a2a_result, ...)     # 重排为 expert-local

# DeepEP FusedDispatch（一步）：
a2a_result = FusedDispatch(tokens, dispatch_idx, ep_group)
# 内部: Layout 计算 + Permute + All-to-All + Unpermute 全部在 GPU 上融合
```

### 4.2 HybridEP 模式

在节点内（NVLink）和节点间（IB）使用不同策略：

```
HybridEP:
  节点内（NVLink, ~600GB/s）:
    使用 NVLink 直连，直接 peer-to-peer copy
    无需经过 NCCL 协议栈
    
  节点间（InfiniBand, ~25-100GB/s）:
    RDMA 直接写
    精确控制每个 rank 发送的 token 数量（避免 padding 浪费）
```

---

## 5. 其他维度的通算融合

### 5.1 TP 通算重叠（Tensor Parallel）

```
TP AllGather（前向）与 GEMM 的 Pipelined 重叠：

传统（串行）：
  AllGather weight → GEMM(all_weight) → ReduceScatter

Pipelined（torch.compile micro_pipeline_tp_pass / Megatron TE overlap）：
  AllGather weight[0:tile_k] ──→ GEMM(weight[0:tile_k])
        AllGather weight[tile_k:2*tile_k] ──→ GEMM(weight[tile_k:...])
                ...
  在 GEMM 处理第 i 个 tile 时，AllGather 在后台取第 i+1 个 tile
```

**实现依赖**：Transformer Engine（TE）的 User Buffer，预分配静态内存避免 AllGather 结果的额外拷贝。

### 5.2 PP 通算重叠（Pipeline Parallel）

**DeepSeek V3 DualPipe**（1F1B 的改进版）：

```
传统 1F1B（存在 bubble）：
  Stage 0: [F0][F1][F2][B2][B1][B0]
  Stage 1:    [F0][F1][F2][B2][B1][B0]
  bubble ←───↑                ↑───→ bubble

DualPipe（减少 bubble）：
  同时从管道两端注入 microbatch
  Stage 0 处理正向时，Stage N 同时处理反向
  bubble ratio ≈ 1/PP × (1 + V/M)
  V=stages_per_pipeline, M=microbatches
```

### 5.3 CP 通算重叠（Context Parallel）

```
Ring Attention（CP 的核心）：

rank i 处理本地 Q_i 时：
  ① 用本地 KV_i 计算 partial attention（无需等待）
  ② 同时将 KV_i 通过 Ring P2P 传给 rank i+1
  ③ 收到 rank i-1 的 KV_{i-1}
  ④ 继续计算 Q_i × KV_{i-1}（使用 online softmax 合并结果）

关键：Step ② 和 Step ④ 与 Step ① 并行
KV AllGather（双缓冲）：
  Buffer A: 当前用于计算的 KV
  Buffer B: 后台正在接收下一个 KV block
```

### 5.4 DP 梯度通算重叠

```
DDP 反向过程中的 Bucket 重叠：

Layer N backward → Layer N-1 backward → ... → Layer 1 backward
                ↘ bucket AllReduce for Layer N 参数梯度
                              ↘ bucket AllReduce for Layer N-1 参数梯度
                                            ↘ ...

后层的 AllReduce 与前层的 backward 计算并行进行
bucket 大小决定 overlap 粒度（太小 → overhead 高，太大 → 等待时间长）
```

---

## 6. MLIR Mesh Dialect：通信进入 IR（未来基础）

当前通信是"框架级黑盒"：torch.compile 看不到 NCCL 调用。未来的方向是将通信作为 IR 一等公民：

```mlir
// MLIR Mesh Dialect（正在开发）
mesh.mesh @data_mesh<["data"=8]>
mesh.mesh @model_mesh<["model"=4]>

// AllGather 作为 IR 节点，可被 Pass 感知和重排
%ag_result, %token = mesh.all_gather %local_shard 
    on @model_mesh[<"model">] 
    async : tensor<256x512xf32> -> (tensor<1024x512xf32>, !mesh.token)

// 依赖分析：expert_compute 只需要 %ag_result 的第 0~256 行
// Pass 可以将 all_gather 分解为多个 chunk，与 expert compute 流水线
%chunk0 = mesh.wait_chunk %token, 0 : tensor<256x512xf32>
%result0 = linalg.matmul %chunk0, %expert_weight_0 ...
// 同时 chunk1 的 all_gather 在后台进行...
```

**与 WaveEP 的关系**：Mesh Dialect 提供了 WaveEP 所需的 IR 表示——通信的 chunk 粒度和异步 token 可以在编译时分析，自动生成 wave-based schedule。

---

## 7. 现状与未来总结

```
通算融合成熟度矩阵:

维度      | 今天状态          | 近期（1-2年）        | 中期（3-5年）
─────────────────────────────────────────────────────────────────
TP        | 半自动（TE+MG）   | torch.compile 自动化 | 无需 flag
DP        | 自动（DDP 分桶）  | 成熟                 | 成熟
PP        | 手动（DualPipe）  | 半自动（sched Pass） | 自动化
EP (MoE)  | 手动（WaveEP）   | DeepEP+框架集成      | wave 粒度自动搜索
CP        | 半自动（Ring AG） | FlexAttention 融合   | 序列维度通算自动化
全自动    | ❌ 无生产实现     | 特定场景可行         | 通用场景（理论）
```

**核心结论**：

1. **WaveEP 是 EP 通算融合的里程碑**，验证了 wave-based pipeline 在万亿参数 MoE 生产系统中的可行性
2. **DeepEP 的 fine-grained SM control 是 GPU/通信协同优化的关键**，未来编译器需要类似的通信 SM 预算感知
3. **MLIR Mesh Dialect + DTensor** 是通算编译器化的两条并行路线，最终可能在 torch.export AOT 场景下汇合
4. **TileLang 的 wave-tile 绑定**是实现 WaveEP 编译化的关键 Missing Link

---

## Related Pages

- [[tilelang_analysis]] — WaveEP 的 tile-level IR 实现机制
- [[26_flex_attention_analysis]] — Context Parallel 中的 FlexAttention + WaveEP 集成
- [[10_mlir_core_concepts]] — MLIR Mesh Dialect 基础
- [[32_post_grad_passes_guide]] — `micro_pipeline_tp_pass`、`fuse_ddp_communication`、`bucket_*` passes
- [[20_megatron_comm_overlap_analysis]] — Megatron-LM 各并行维度通算重叠详细实现
- [[01_megatron_moe_training_optimization_report]] — DeepEP/HybridEP 在 Megatron-LM 中的集成
- [[12_deepseek_v3_analysis]] — DualPipe PP 通算重叠设计
- [[13_deepseek_v4_analysis]] — WaveEP 细粒度 EP 重叠（wave-based expert scheduling）
