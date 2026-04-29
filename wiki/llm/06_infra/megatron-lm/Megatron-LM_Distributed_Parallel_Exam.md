# Megatron-LM 分布式并行训练考核题（Dev 最新特性版）

> 覆盖版本：Megatron-LM dev branch (~0.12.0–0.17.0)，结合 TransformerEngine、MoE、Context Parallelism、CUDA Graphs 等最新特性。

---

## 评分参考

| 层次 | 题号 | 能力要求 |
|------|------|----------|
| 基础概念 | Q1–Q5 | 能准确区分各并行维度的作用范围与通信特征 |
| 原理与通信分析 | Q6–Q11 | 能定量推导通信量，理解通信与计算的重叠机制 |
| 显存优化与计算优化 | Q12–Q17 | 理解 activation checkpointing、offloading、CUDA Graphs、FP8 的底层机制与 trade-off |
| MoE 与新型并行 | Q18–Q22 | 掌握 MoE 的 EP/ETP、dispatch 机制、MoE Folding、MLA、MTP 等前沿特性 |
| 代码实现与调试 | Q23–Q26 | 能阅读并修改核心源码（layers.py、schedule.py、parallel_state.py） |
| 实战调优与排障 | Q27–Q30 | 能基于 profiling 和日志定位性能瓶颈、显存 OOM、通信 hang 等真实问题 |

---

## 第一层：基础概念

### Q1. 3D 并行与 4D/5D 并行
**考点**：并行维度的定义、组合原则、各自解决的问题。

Megatron-LM 传统上提 "3D 并行"，随着新特性加入，实际训练已经支持 5D 甚至更高维度的混合并行。请说明：
1. 传统 3D 并行指的是哪三个维度？
2. 在引入 Sequence Parallelism (SP) 和 Context Parallelism (CP) 后，这五个维度（DP, TP, PP, SP, CP）各自切分的是哪个张量维度？
3. Expert Parallelism (EP) 和 Expert Tensor Parallelism (ETP) 与上述维度的关系是什么？

<details>
<summary>参考答案</summary>

1. **传统 3D**：Data Parallelism (DP)、Tensor Parallelism (TP)、Pipeline Parallelism (PP)。
2. **五维切分维度**：
   - **DP**：在 `batch` 维度切分数据，每张卡持有完整参数的一个副本（或 shard）。
   - **TP**：在 `hidden` 维度（线性层权重/激活）切分，属于 intra-layer parallelism。
   - **PP**：在 `layer` 维度切分模型深度，属于 inter-layer parallelism。
   - **SP**：在 `sequence` 维度切分 LayerNorm/Dropout 的激活，是对 TP 的补充。
   - **CP**：在 `sequence` 维度切分 Attention 的 Q/K/V，使长序列能跨卡分布。
3. **EP/ETP**：
   - EP 在 `expert` 维度将不同的 MoE expert FFN 分布到不同卡上。
   - ETP 在 `hidden` 维度切分单个 expert 内部权重，与 attention 的 TP 解耦，避免小 GEMM 效率低下。

</details>

---

### Q2. Sequence Parallelism 与 Tensor Parallelism 的边界
**考点**：SP 的引入动机、通信位置、与 TP 的协作方式。

在标准的 Transformer Layer 中，Megatron-LM 推荐 "TP 必开 SP"。
1. 如果只用 TP 不用 SP，LayerNorm 和 Dropout 的显存为什么无法下降？
2. 开启 SP 后，从 ColumnParallelLinear 输出到 LayerNorm，再到 RowParallelLinear 输入之间，通信原语发生了怎样的变化？请写出具体的通信算子。

<details>
<summary>参考答案</summary>

1. **原因**：LayerNorm 和 Dropout 是 element-wise 算子，计算与参数都不涉及 `hidden` 维度，只与 `sequence` 维度相关。TP 切分的是 `hidden` 维度，因此这些算子的输入激活在每个 TP rank 上仍然是完整序列长度的副本，显存无法减少。
2. **通信变化**：
   - 无 SP 时：ColumnParallelLinear 输出后需进行一次 `AllReduce`（或 `AllGather`），将完整激活传给 LayerNorm。
   - 有 SP 时：ColumnParallelLinear 输出后改为 `ReduceScatter`，将序列维度切分，使每个 rank 只持有 `seq/tp` 的激活；LayerNorm/Dropout 后在进入 RowParallelLinear 前做一次 `AllGather`（或等价通信），恢复 `hidden` 维度的完整切片。
   - **结论**：通信量不变，但显存峰值下降，因为中间激活被 shard 了。

</details>

---

### Q3. Context Parallelism 与 Sequence Parallelism 的本质区别
**考点**：CP 的切分对象、通信模式、与 SP 的互补关系。

请从以下三个角度对比 CP 和 SP：
1. 切分的模块范围（哪些层被切分）
2. 通信模式（AllReduce / AllGather / Ring P2P）
3. 为什么 CP 可以在不增加 TP group 大小的前提下支持更长的序列？

<details>
<summary>参考答案</summary>

| 维度 | SP | CP |
|------|----|----|
| **切分模块** | 仅 LayerNorm、Dropout 等 seq-dim element-wise 算子 | Attention 的 Q/K/V 以及对应的 FlashAttention/FA3 计算 |
| **通信模式** | `ReduceScatter` + `AllGather`（或 `AllReduce`） | Ring P2P（`cp_comm_type=p2p`）或 `AllGather`/`a2a`/`a2a+p2p` |
| **长序列支持** | 只减少 LN/Dropout 显存，Attention 仍需完整序列 | 将 Attention 的序列长度本身 shard 到 CP group 上，直接降低单卡 KV 激活和 Attention 计算量 |

CP 支持更长序列的核心：Attention 的复杂度是 $O(seq^2)$，CP 将 $seq$ 分到多个卡上，单卡计算量和激活显存都与 $(seq/cp)^2$ 成正比（或线性 attention 下与 $seq/cp$ 成正比），因此可以在 TP=1 的情况下将序列扩展到多张卡。

</details>

---

### Q4. Pipeline Parallelism 中的 micro-batch
**考点**：micro-batch 的定义、作用、与 pipeline bubble 的关系。

1. 什么是 micro-batch？它与 global batch size 的关系是什么？
2. 在 Pipeline Parallelism 中，如果只用一个 macro-batch（即 micro-batch = 1），会出现什么问题？
3. 1F1B（One-Forward-One-Backward）调度相比 GPipe，在显存和 bubble 上有何差异？

<details>
<summary>参考答案</summary>

1. **micro-batch** 是将 global batch 进一步切分后的最小计算单元。关系：$\text{global batch} = \text{micro batch size} \times \text{num micro batches} \times \text{DP size}$。
2. **问题**：单 micro-batch 时，PP 各 stage 串行执行，除了当前计算的 stage 外其余全部空闲，pipeline bubble 达到 100%，硬件利用率极低。
3. **差异**：
   - **显存**：1F1B 在 steady state 阶段先 forward 一个 micro-batch 后立即 backward 前一个 micro-batch，可以尽快释放前向激活；GPipe 需缓存全部 micro-batch 的前向激活到最后一个 stage 才开始 backward，峰值显存更高。
   - **Bubble**：两者理论 bubble ratio 相同，都是 $(p-1)/(m \cdot v)$（$v=1$ 时即 $(p-1)/m$），其中 $p$ 为 physical stage 数，$m$ 为 micro-batch 数，$v$ 为 virtual stage 数。但实际中 1F1B 的显存更低，允许使用更大的 $m$。

</details>

---

### Q5. Pipeline Parallelism 的进程组拓扑
**考点**：Megatron rank ordering、进程组的嵌套关系。

在一个 32 卡集群上，配置为 `TP=2, PP=4, DP=4`。
1. 按照 Megatron 默认的 rank ordering（先 TP，再 CP，再 PP，最后 DP），写出 GPU 0 所在的所有进程组成员。
2. 如果在此基础上加入 `CP=2`，GPU 0 的进程组又该如何变化？

<details>
<summary>参考答案</summary>

**TP=2, PP=4, DP=4**：
- TP group: `[0, 1]`
- PP group: `[0, 2, 4, 6]`（同一 DP rank 内，沿 TP group 取第一个 rank 构成 PP）
- DP group: `[0, 8, 16, 24]`（每个 TP×PP tile 取对应位置 rank）

**加入 CP=2 后**：
- CP group: `[0, 1]`（注意：CP 在 TP 之后、PP 之前）
- TP group: `[0]`（若 TP=1）或 `[0, 1]`（若 TP=2，需重算 rank layout）

严格来说，加入 CP=2 后总并行度 = 2×2×4×4=64 > 32，32 卡配置需调整。若改为 `TP=2, CP=2, PP=2, DP=4`（2×2×2×4=32）：
- TP: `[0, 1]`
- CP: `[0, 2]`（同 TP group 的相邻 tile）
- PP: `[0, 4]`
- DP: `[0, 8, 16, 24]`

</details>

---

## 第二层：原理与通信分析

### Q6. 张量并行的通信量定量分析
**考点**：AllReduce 的通信量公式、TP 的前向/反向通信次数。

对于一个标准的 Transformer Decoder Layer（包含 Self-Attention + FFN），在 TP 度为 $t$ 时：
1. 前向传播需要多少次 AllReduce（或等价的 ReduceScatter + AllGather）？分别发生在哪些位置？
2. 反向传播需要多少次 AllReduce？与正向如何对偶？
3. 如果模型参数量为 $N$，数据并行度为 $d$，梯度同步的总通信量是多少（以字节数表示，考虑 FP32 梯度）？

<details>
<summary>参考答案</summary>

1. **前向**：2 次 AllReduce（或等价通信）：
   - Attention 的 `WO`（RowParallelLinear）输出后
   - FFN 的第二个线性层（RowParallelLinear）输出后
   注意：若开启 SP，这两次前向 AllReduce 被拆分为 `ReduceScatter` + `AllGather`（LayerNorm 前后），通信量不变，但显存峰值降低。

2. **反向**：2 次 AllReduce（对偶位置）：
   - `WO` 反向时，对输入梯度做 AllReduce（对应 ColumnParallelLinear 的反向）
   - FFN 第二个线性层反向时，同样对输入梯度做 AllReduce
   注意：RowParallelLinear 的反向本身不需要通信（AllReduce 在前向已完成）。

3. **梯度同步通信量**：DP 梯度同步是标准的 Ring AllReduce，通信量为 $2N$（每个 rank 需要发送和接收约 $N$ 的数据）。若梯度为 FP32，则实际传输字节数为 $2N \times 4 = 8N$ 字节。SP/TP 不改变 DP 的梯度同步通信量（它们各自 rank 持有不同分片的参数，各自产生独立梯度）。

</details>

---

### Q7. Pipeline 并行的 stage 间通信
**考点**：PP 的通信内容、通信量计算、与 TP 通信的对比。

1. 在标准 Transformer 中，PP 相邻 stage 间传递的数据是什么？请用 `hidden_size (h)`、`sequence_length (s)`、`micro_batch_size (b)` 表示一次 forward P2P 的通信量。
2. 为什么 PP 的 stage 间通信通常比 TP 的 AllReduce "更便宜"？请从通信频率、通信量、能否被 overlap 三个角度分析。

<details>
<summary>参考答案</summary>

1. **传递内容**：相邻 stage 间传递的是前向激活（hidden states）和反向梯度。一次 forward P2P 通信量为 $b \times s \times h \times \text{sizeof(dtype)}$。若考虑 activation checkpointing 保留 input/output，则还需要考虑发送/接收的边界 tensor 数量。

2. **PP 通信更便宜的原因**：
   - **通信频率**：TP 的 AllReduce 发生在**每层内部**（Attention + FFN 共 2 次/层），频率高；PP 的 P2P 只在**层边界**发生，频率低。
   - **通信量**：TP AllReduce 的通信量与权重矩阵的 hidden size 成正比，通常是 $2 \times h \times h / t$（每次 AllReduce 的数据量），虽然绝对值不一定比 PP 大，但发生次数多得多。
   - **Overlap 能力**：PP 的 P2P 可以和下一个 micro-batch 的计算高度重叠（利用 micro-batch pipeline），而 TP 的 AllReduce 通常位于矩阵乘法的 critical path 上（虽然 modern implementation 可以做 AG/RS overlap，但难度更高）。

</details>

---

### Q8. AllReduce vs ReduceScatter + AllGather
**考点**：集合通信的 latency/bandwidth 模型、TP overlap 的实现。

在 Tensor Parallelism 中，Megatron 早期直接用 `AllReduce`，而最新实现（特别是开启 `tp_comm_overlap` 后）倾向于将 `AllReduce` 拆分为 `ReduceScatter (RS)` + `AllGather (AG)`。
1. 从通信量角度看，这两种方式的通信量是否相同？请给出定量说明。
2. 为什么 `RS + AG` 更有利于与 GEMM 计算 overlap？请描述 `tp_comm_overlap` 的时间线调度原理。

<details>
<summary>参考答案</summary>

1. **通信量相同**：对于 $t$ 个 rank、数据量为 $D$ 的张量：
   - Ring AllReduce: 分为 ReduceScatter + AllGather 两阶段，总通信量 = $2(t-1)/t \cdot D \approx 2D$。
   - 显式 RS + AG: 总通信量 = $(t-1)/t \cdot D + (t-1)/t \cdot D \approx 2D$。
   因此总通信量等价。

2. **Overlap 优势**：
   - `AllReduce` 要求所有 rank 持有完整输出，通信和后续计算（如下一个 LayerNorm 或 Attention）的依赖关系是"全有或全无"，难以细粒度 overlap。
   - `RS + AG` 将过程拆成两阶段：
     - `ReduceScatter` 阶段：每个 rank 先拿到自己负责分片的 reduced 结果，可以**立即开始下一部分的局部计算**。
     - `AllGather` 阶段：在局部计算的同时或之后，异步 gather 其他分片。
   - `tp_comm_overlap`（如 TransformerEngine 的实现）会将 GEMM 的输出切片按 chunk 流水线化：计算完一个 chunk 就启动该 chunk 的 AG/RS，使得通信与后续 GEMM 的其余 chunk 计算重叠。

</details>

---

### Q9. Context Parallelism 的 Ring Attention 通信分析
**考点**：CP 的 ring P2P 通信量、与 Attention 计算复杂度的关系。

假设使用 `cp_comm_type=p2p` 的 Ring Attention 实现，序列长度 $S$，hidden size $H$，CP 度为 $c$。
1. 每个 CP rank 上 Q/K/V 的本地序列长度是多少？
2. 在 ring 过程中，每个 rank 需要向邻居发送/接收多少次 KV block？每次的通信量是多少？
3. 若改用 `cp_comm_type=all_gather`，通信模式有何变化？显存和计算上各有什么 trade-off？

<details>
<summary>参考答案</summary>

1. 本地序列长度 = $S / c$。

2. Ring P2P 过程：
   - 每个 rank 需要将本地 KV 沿 CP ring 发送给其余 $c-1$ 个 rank，并接收来自它们的 KV。
   - 共 $c-1$ 轮，每轮每个 rank 发送/接收一个 KV block。
   - 每次通信量 = $(S/c) \times H \times \text{sizeof(dtype)}$（KV 各一份，若按 fused KV 算则需乘 2）。
   - 总通信量 ≈ $(c-1) \times (S/c) \times H \times 2 \approx 2SH$（与 $c$ 无关，仅与总序列长度相关）。

3. `all_gather` 模式：
   - **通信模式**：在 Attention 前一次性 AllGather 完整的 K 和 V，然后每个 rank 用本地 Q 和全局 K/V 做 Attention。
   - **显存 trade-off**：AllGather 后每个 rank 需要暂存完整的 K 和 V（显存占用 $2 \times S \times H$），而 Ring P2P 只需同时存两个 block（$2 \times S/c \times H$），显存更优。
   - **计算 trade-off**：AllGather 可以使用标准的 FlashAttention（无需 custom ring kernel），实现更简单；Ring P2P 需要 custom kernel（如 TransformerEngine 的 `te_attn` with CP）来支持逐块 softmax 归约，但显存和某些场景下的性能更好。

</details>

---

### Q10. Dynamic Context Parallelism 的调度原理
**考点**：Dynamic CP 的负载均衡、packed sequence、scheduler 的作用。

Megatron-LM 引入了 `Dynamic Context Parallelism` 来优化 packed sequence（如 SFT 数据）的训练效率。
1. 标准的 CP 在处理 packed sequence 时会遇到什么问题？
2. `dp_balanced` scheduler 和 `default_dynamic_cp` scheduler 各自解决什么问题？
3. Dynamic CP 是如何在不改变模型并行拓扑的前提下，动态调整 CP group 大小的？

<details>
<summary>参考答案</summary>

1. **标准 CP 的问题**：packed sequence 中每个 sample 的序列长度差异很大。标准 CP 按固定 $c$ 将 batch 内的序列均匀切分，导致某些 CP rank 分配到的有效 token 数远少于其他 rank，出现严重的负载不均衡（straggler effect），GPU 利用率下降。

2. **Scheduler 区别**：
   - `dp_balanced`：按原始顺序将序列打包到 `max_seqlen` 以内，保证 DP rank 间负载均衡，但不解决 CP rank 内的不均衡。
   - `default_dynamic_cp`：专门为 Dynamic CP 设计，在打包的同时考虑 CP 组内的 token 分布，使每个动态 CP group 内的有效计算量更均衡。

3. **Dynamic CP 原理**：
   - 训练开始前，从 DP×CP 的 rank pool 中**动态地**构建大小不等的 CP groups。
   - 对于较长的序列，使用更大的 CP group（更多 rank 分担计算）；对于较短的序列，使用更小的 CP group（减少通信开销）。
   - 模型并行拓扑（TP/PP）保持不变，仅 CP group 的划分在 step 间或 batch 间动态调整。
   - 效果：相比固定 CP，Dynamic CP 在 packed sequence SFT 场景下可提升高达 1.48x 的速度。

</details>

---

### Q11. Expert Parallelism 的通信与拓扑
**考点**：EP 的 all-to-all 通信、MoE Folding、通信 overlap。

在一个 MoE 模型中，配置为 `ETP=2, EP=8, DP=4`，总卡数 = 64。
1. 请写出该配置下一个 MoE Layer 的前向通信流程（从 token 路由到 expert 计算再到输出聚合）。
2. 什么是 MoE Parallel Folding？为什么它可以打破 `EP ≤ DP` 的传统限制？
3. `overlap_moe_expert_parallel_comm` 和 `delay_wgrad_compute` 是如何配合实现 EP all-to-all 与计算 overlap 的？

<details>
<summary>参考答案</summary>

1. **前向通信流程**：
   - **路由阶段**：所有 token 经过 Router，决定每个 token 要发送到哪些 top-k expert。
   - **Dispatch（all-to-all）**：沿 EP group 进行 all-to-all，将 token 从 source rank 分发到目标 expert 所在的 rank。
   - **Expert 计算**：每个 rank 只计算分配给本地 expert 的 token（此时 expert 内部可再开 ETP=2 做张量并行）。
   - **Combine（all-to-all）**：沿 EP group 进行反向 all-to-all，将计算结果返回给 source rank，并做加权求和。

2. **MoE Parallel Folding**：
   - 传统实现中，attention 和 MoE 共享同一套并行拓扑，因此 EP 受限于 DP 大小（$EP \leq DP$）。
   - **Folding** 将 CP 和 EP 维度"折叠"组合，使得 MoE 层可以拥有独立于 attention 层的并行配置。例如 attention 用 (TP=4, CP=1, DP=8)，MoE 可以用 (ETP=1, EP=32, EDP=1)。
   - 这样 EP 可以远大于 attention 的 DP，从而支持更多 expert 和更大的 EP 规模。

3. **Overlap 机制**：
   - `overlap_moe_expert_parallel_comm`：将当前 micro-batch 的 MoE dispatch/combine all-to-all 与 expert GEMM 计算在 batch 级别做 overlap（例如当第一批 token 在 expert 中计算时，第二批 token 的 dispatch 通信已启动）。
   - `delay_wgrad_compute`：延迟 weight gradient 的计算，为 all-to-all 通信腾出更多时间和带宽，避免通信与 wgrad GEMM 争用 SM/stream。

</details>

---

## 第三层：显存优化与计算优化

### Q12. Activation Checkpointing 的显存与计算 trade-off
**考点**：Full vs Selective Recomputation 的显存收益、计算开销、工程选择。

Megatron-LM 支持 `recompute_granularity='full'` 和 `'selective'`。
1. Full Recomputation 每层的显存收益是什么？计算开销增加了多少？
2. Selective Recomputation 默认重计算哪些模块？选择这些模块的依据是什么？
3. 如果开启 selective recomputation 后，发现 MFU 下降了 15% 但显存下降不明显，可能是什么原因？

<details>
<summary>参考答案</summary>

1. **Full Recomputation**：
   - 只保留 transformer layer 的输入（和可能的 attention mask），丢弃中间所有激活。
   - 显存收益：单层激活从 $O(bs \cdot seq \cdot h \cdot \text{num layers})$ 降低到只保存 $O(bs \cdot seq \cdot h)$ 的输入。
   - 计算开销：增加约 1/3 的总计算量（因为 forward 多算了一次，用于 backward 中的 recompute）。

2. **Selective Recomputation**：
   - 默认重计算 activation 占显存大、重计算计算开销小的模块，典型如 `core_attn`（attention 的 softmax 和输出投影）。
   - 选择依据：
     - 显存占用高（activation 随 $seq^2$ 或 $bs \cdot seq \cdot h$ 增长）。
     - 重计算开销低（主要是矩阵乘法和 memory-bound kernel，重算一次不会显著增加计算时间）。
   - 可选模块还包括 `moe_act`、`layernorm`、`mla_up_proj`、`mlp`、`moe`、`shared_experts`、`mhc`。

3. **MFU 下降但显存下降不明显的原因**：
   - 重计算的 granularity 设置不合理（例如 `num_layers` 过小，导致 recompute 的 forward bubble 无法被计算 overlap）。
   - checkpoint 保留的 tensor 过多（如保留了 attention input 但 forgot 了一些中间 buffer），导致 recompute 并没有释放预期显存。
   - 使用了未 fuse 的 recompute kernel（如 `torch.utils.checkpoint` 而非 Megatron 自定义的 `checkpoint`），导致额外的 CPU launch overhead 或 kernel gap。
   - 显存虽然下降了一点，但 batch size 没有相应调大，所以效率没有提升。

> [!note] [[activation_checkpointing_analysis]] 从 autograd ctx 保存机制到 Megatron CheckpointFunction 源码的完整分析——包括 view/cast/slice 为何不需要重计算、`make_viewless_tensor`、`distribute_saved_activations`、CheckpointWithoutOutput 等。

</details>

---

### Q13. Fine-Grained Activation Offloading
**考点**：Offloading 的粒度、异步传输、与 CUDA Graphs / PP 的兼容性。

Megatron-LM 支持 `fine_grained_activation_offloading`。
1. 与 Full Activation Checkpointing 相比，Fine-Grained Activation Offloading 的显存释放粒度有何不同？
2. 它使用什么 CUDA stream 机制来实现计算与传输 overlap？
3. 为什么 Fine-Grained Activation Offloading 需要特别注意与 CUDA Graphs 的兼容？Megatron 是如何解决这个问题的？

<details>
<summary>参考答案</summary>

1. **粒度差异**：
   - Full Checkpointing：以整个 Transformer Layer 为粒度，要么全部保留输入，要么全部重算。
   - Fine-Grained Offloading：以**子模块**为粒度（如 `attention`、`mlp`、`moe_experts`），只将特定子模块的输出 activation 异步 offload 到 CPU，backward 时再异步 prefetch 回 GPU。

2. **异步传输机制**：
   - 使用独立的 CUDA stream（D2H stream 和 H2D stream）执行 `cudaMemcpyAsync`。
   - Forward 时，主计算流完成子模块计算后，触发 D2H stream 的拷贝；backward 时，在需要该 activation 之前提前触发 H2D stream 的 prefetch。
   - 通过事件同步（`cudaEventRecord` / `cudaStreamWaitEvent`）保证数据依赖。

3. **CUDA Graphs 兼容性**：
   - CUDA Graph 会 capture 一段固定执行路径中的所有 kernel 和 memory operation。如果 offloading 的 D2H/H2D 拷贝发生在 graph capture 范围内，graph 会尝试 capture 这些跨设备操作，可能导致 capture 失败或运行时错误。
   - **解决方案**：Megatron 将 offloading 的 `commit` 操作延迟到 CUDA graph 的边界之外（如 layer 之间或 iteration 边界），确保 graph 内部只包含纯 GPU 计算 kernel。

</details>

---

### Q14. CUDA Graphs 的选型与 Scope 配置
**考点**：CUDA Graphs 的两种实现、scope 选择、与 MoE 的兼容。

Megatron-LM 支持 `--cuda-graph-impl local` 和 `transformer_engine`。
1. 这两种实现的主要区别是什么？各适合什么场景？
2. `--cuda-graph-scope` 支持哪些 fine-grained scope（至少列出 4 个）？在什么情况下你应该选择 `full_iteration` 而不是 `attn`？
3. MoE 模型使用 CUDA Graphs 时，需要解决哪些特殊问题？

<details>
<summary>参考答案</summary>

1. **两种实现区别**：
   - `local`：MCore 内部的 CUDA graph manager，支持 partial layer graph 或 full iteration graph，灵活性更高，适合需要自定义 graph 边界或调试的场景。
   - `transformer_engine`：使用 TE 的 `make_graphed_callables()`，TE 会自动处理内部 kernel 的 graph capture，通常与 TE 的 FP8/BF16 kernel 配合更好，适合生产环境。

2. **Fine-grained scopes**：
   - `attn`、`mlp`、`moe`、`moe_router`、`moe_preprocess`、`mamba`、`full_iteration`。
   - 选择 `full_iteration` 的场景：
     - 当模型中存在大量 tiny kernel（如 frequent elementwise ops、custom optimizer states update），CPU launch overhead 成为主要瓶颈时。
     - 当 partial scope 无法覆盖足够多的 kernel，导致 graph 的收益不明显时。
   - 注意：`full_iteration` 要求整个 iteration 的执行路径完全静态（包括 tensor shape、conditional branch 等），因此不适合 dynamic shape 或 packed sequence 训练。

3. **MoE + CUDA Graphs 的特殊问题**：
   - **动态形状**：MoE 的 dispatch 后每个 expert 处理的 token 数随输入变化，导致 GEMM 的 M 维度不固定，graph capture 要求静态 shape。
   - **解决方案**：通过 `graph-safe padding`（将 dispatch 后的 tensor pad 到固定 shape，计算后再 unpad）或 `skip routed expert padding` 来稳定 shape；或者只在 non-MoE 部分做 graph（选择 `moe` 以外的 scope）。

</details>

---

### Q15. FP8 / MXFP8 / FP4 混合精度训练
**考点**：FP8 的 recipe 选择、blockwise scaling、primary weights。

Megatron-LM 支持多种 FP8 训练策略。
1. `--fp8-recipe` 支持 `tensorwise`、`delayed`、`blockwise`、`mxfp8`、`custom`，请简述 `blockwise` 和 `mxfp8` 的核心区别及适用硬件。
2. 什么是 `fp8_param_gather`（FP8 Primary Weights）？它为什么能节省显存？
3. `first_last_layers_bf16` 这个 flag 的作用是什么？为什么需要它？

<details>
<summary>参考答案</summary>

1. **Blockwise vs MXFP8**：
   - **Blockwise FP8**：在更小的 block（如 1×128 或 128×128）级别做 scale，比 tensorwise 或 delayed scaling 更细粒度，能更好地保留数值精度。已在 Hopper（H100）上生产验证（DeepSeek-V3、Minimax-M2）。
   - **MXFP8**：基于 Microscaling 格式，是 Blackwell/GB200 的原生硬件支持格式，无需软件模拟 block scaling，硬件直接处理。适合 B200/GB200 及以后架构。

2. **FP8 Primary Weights**：
   - 传统混合精度保留 FP32 master weights + BF16/FP16 前向副本，显存占用 = $4N + 2N = 6N$ 字节。
   - FP8 Primary Weights 直接将 master weights 以 FP8 格式保存，前向时直接使用 FP8 weight（或配合更高精度 buffer），省去 BF16 副本。
   - 显存节省：从 $6N$ 降至约 $5N$（FP32 optimizer states 仍需保留）。

3. **first_last_layers_bf16**：
   - 将模型的前 N 层和后 N 层强制保留为 BF16（而非 FP8）。
   - **原因**：模型的首层和末层通常对数值精度更敏感（首层处理原始输入分布，末层决定 logits 和 loss），FP8 的量化误差可能导致训练不稳定或 loss 发散。保留 BF16 可以在精度和效率之间取得平衡。

</details>

---

### Q16. Megatron FSDP 与 PyTorch FSDP2 的选择
**考点**：FSDP 的 sharding strategy、与 TP/EP/CP 的兼容性、性能差异。

Megatron-LM 同时提供 `--use-megatron-fsdp` 和 `--use-torch-fsdp2`。
1. Megatron FSDP 支持哪些 sharding strategy？各对应 ZeRO 的哪个阶段？
2. 为什么 Megatron FSDP 声称比 PyTorch FSDP2 快约 15%？请从通信 overlap 和 tensor layout 角度分析。
3. 在 `TP=4, EP=8` 的配置下，能否开启 FSDP？如果可以，FSDP 的 shard 应该在哪个进程组维度上执行？

<details>
<summary>参考答案</summary>

1. **Sharding strategies**：
   - `optim`：仅 shard optimizer states（ZeRO-1）。
   - `optim_grads`：shard optimizer states + gradients（ZeRO-2）。
   - `optim_grads_params`：shard optimizer states + gradients + parameters（ZeRO-3）。

2. **性能优势原因**：
   - **通信 overlap**：Megatron FSDP 针对 TP/EP/CP 的 group 拓扑做了深度优化，`param_gather` 和 `grad_reduce` 可以与 forward/backward 计算更紧密地 overlap；PyTorch FSDP2 的 overlap 策略更通用，在某些 Megatron-specific 的 layout 下不够极致。
   - **Tensor layout**：Megatron FSDP 使用与 TP 兼容的 sharded tensor layout，all-gather 和 reduce-scatter 的数据排列更符合 Megatron 的 column/row parallel 格式，减少了额外的 permute/copy 开销。

3. **TP=4, EP=8 下能否开 FSDP**：
   - **可以**。FSDP 的 shard 应在 **DP 维度**上执行（即 `world_size / (TP × CP × PP × EP)`）。
   - 注意：FSDP 的 shard group 不应与 TP/EP/CP/PP group 重叠，因为那些 group 已经做了参数分片或 activation 分片，FSDP 只负责 DP 维度的冗余消除。

</details>

---

### Q17. Distributed Optimizer 与 Layer-Wise Distributed Optimizer
**考点**：Optimizer state sharding、checkpointing 优势、多优化器组合。

1. `--distributed-optimizer` 相比标准的 AdamW，在显存和 checkpoint 性能上各有什么优势？
2. `Layer-Wise Distributed Optimizer`（`--layer-wise-distributed-optimizer`）解决了什么问题？什么场景下你会选择它而不是普通 distributed optimizer？
3. 当使用 `Muon` + `AdamW` 组合优化器时，Layer-Wise 的 `ChainedOptimizer` 是如何分配不同参数到不同优化器的？

<details>
<summary>参考答案</summary>

1. **Distributed Optimizer 优势**：
   - **显存**：将 optimizer states（如 Adam 的 $m$ 和 $v$）shard 到 DP rank 上，每个 rank 只保存 $1/d$ 的 optimizer states，显著降低显存。
   - **Checkpoint**：保存 checkpoint 时，每个 rank 只需保存自己 shard 的优化器状态，写带宽和 checkpoint 体积都大幅降低；加载时也只需读取对应 shard。

2. **Layer-Wise Distributed Optimizer**：
   - 将参数按 **层** 分配到 DP rank 上，而不是按扁平化的参数列表分配。
   - **解决的问题**：
     - 支持 **多个优化器组合**（如 Muon 处理矩阵参数，AdamW 处理 vector/bias 参数），普通 distributed optimizer 难以优雅支持 per-parameter optimizer。
     - 更细粒度的 all-gather overlap：可以在计算第 $L$ 层 forward 的同时，异步 all-gather 第 $L+1$ 层的参数。
   - **选择场景**：使用混合优化器（如 Muon + AdamW）或超大模型需要极致 overlap 时。

3. **ChainedOptimizer 分配**：
   - 通过 `param_group` 的 `optimizer_name` 或 `foreach` 映射规则，将不同的参数子集路由到不同的底层优化器。
   - 例如：所有 `weight` 矩阵参数（≥2D）分配给 `MuonOptimizer`，所有 `bias`、`norm`、`embedding` 参数分配给 `AdamWOptimizer`。

</details>

---

## 第四层：MoE 与新型并行

### Q18. MoE 的 Router 设计与负载均衡
**考点**：Top-k routing、aux loss、aux-loss-free load balancing。

1. 标准的 MoE Top-k Router 中，aux loss 的作用是什么？它的梯度会更新哪些参数？
2. Megatron-LM 支持 `dynamic_expert_bias` 实现 aux-loss-free 负载均衡，请简述其原理。
3. `moe_router_num_groups` + `moe_router_group_topk` 这组配置是什么含义？它对应 DeepSeek 的哪种路由策略？

<details>
<summary>参考答案</summary>

1. **Aux loss**：
   - 作用是惩罚路由不均衡，鼓励 router 将 token 均匀分配到各个 expert 上。
   - 梯度只更新 **router 的权重矩阵**（即 gate 网络），不影响 expert 参数或下游层。

2. **Dynamic Expert Bias（aux-loss-free）**：
   - 为每个 expert 维护一个可学习的 bias 项。
   - 在训练过程中，根据每个 expert 的负载动态更新 bias：过载的 expert bias 减小，欠载的 expert bias 增大。
   - 这样在 inference 时可以直接丢弃 bias（或固定为最终值），避免 aux loss 对训练目标的干扰。

3. **Group-Limited Routing**：
   - `moe_router_num_groups`：将 expert 分成若干组。
   - `moe_router_group_topk`：每个 token 先从组级别选择 top-k 组，再在组内选择 top-1 expert。
   - 对应 **DeepSeek-V2/V3 的 Device-Limited Routing / Node-Limited Routing**，限制 token 只访问部分设备/节点上的 expert，减少 all-to-all 的通信跳数和延迟。

</details>

---

### Q19. Grouped GEMM 与 Router Fusion
**考点**：MoE 的计算优化、kernel fusion、吞吐提升。

1. 在标准的 MoE 实现中，每个 expert 是一个独立的 `F.linear` 调用，这在大量 expert 时有什么问题？Megatron 的 `Grouped GEMM` 如何解决？
2. `Router Fusion` 具体融合了哪些操作？它的性能收益主要来自减少什么开销？

<details>
<summary>参考答案</summary>

1. **独立 F.linear 的问题**：
   - 每个 expert 的 GEMM 形状（M 维度）由分配给它的 token 数决定，通常很小（如 M=几十到几百）。
   - 大量小 GEMM 导致：**kernel launch overhead 剧增**、**GPU 利用率低**（SM 占用率低）、**无法充分利用 Tensor Core**。

2. **Grouped GEMM 解决方式**：
   - 将多个 expert 的小 GEMM 合并为一个 `grouped GEMM` kernel launch（通过 cuBLAS 的 grouped GEMM API 或 custom fused kernel）。
   - 一次 launch 处理所有 expert 的计算，消除多次 kernel 调度开销，提高 SM 占用率和 Tensor Core 效率。
   - 支持 FP8、MXFP8、BF16 等精度。

3. **Router Fusion**：
   - 融合的操作：Router 的线性投影（`gate_proj`）→ Top-k 选择 → Softmax/SqrtSoftplus → Aux loss 计算。
   - 性能收益：减少了多个小 kernel 之间的 global memory 读写和 CPU launch 延迟，使路由决策成为一个紧凑的 compute-bound 阶段。

</details>

---

### Q20. Multi-Latent Attention (MLA)
**考点**：MLA 的显存优势、与 TP/CP 的兼容性、吸收化优化。

1. 标准 MHA 的 KV cache 显存是 $O(2 \cdot n_{layers} \cdot n_{heads} \cdot d_{head} \cdot seq)$，MLA 如何降低这一开销？
2. Megatron 中的 "Absorbed MLA" 是什么意思？它通过什么方式减少了 inference 时的额外矩阵乘法？
3. MLA 在训练时能否与 Context Parallelism 一起使用？CP 切分的是哪个投影后的张量？

<details>
<summary>参考答案</summary>

1. **MLA 的显存优势**：
   - MLA 将 Key 和 Value 投影到一个低维的 **latent space**（即 compressed KV），而不是直接存储每个 head 的完整 KV。
   - KV cache 大小从 $O(n_{heads} \cdot d_{head})$ 降低到 $O(d_{c})$，其中 $d_c$ 是压缩后的 latent dimension，通常 $d_c \ll n_{heads} \cdot d_{head}$。

2. **Absorbed MLA**：
   - 在 inference 时，将 MLA 中的 up-projection 矩阵"吸收"到 attention 的 Q projection 或 output projection 中。
   - 这样不需要在每次 decode step 都显式做 KV 的 up-projection，减少了一个矩阵乘法，降低了 latency。
   - Megatron 的实现通过 fused down-projection GEMM 进一步提升了训练吞吐。

3. **MLA + CP**：
   - **可以**一起使用。CP 切分的是经过压缩后的 KV latent 张量（或 Q 投影后的张量），而不是原始完整的 per-head KV。
   - 因为 MLA 的 attention 计算仍需要完整的 softmax 归约，CP 的 ring attention / all-gather 机制同样适用，只是通信量因压缩而更小。

</details>

---

### Q21. Multi-Token Prediction (MTP)
**考点**：MTP 的 loss 计算、pipeline 布局、与 speculative decoding 的关系。

1. MTP 模块在模型结构中是如何连接的？它的 loss 如何与主 loss 加权求和？
2. Megatron 的 `Custom Pipeline Layout` 中，`m` 符号代表什么？为什么 MTP 层需要特殊的 pipeline 布局支持？
3. MTP 在 inference 时如何支持 speculative decoding？

<details>
<summary>参考答案</summary>

1. **MTP 结构**：
   - 在主 decoder 层之上，添加额外的 MTP head 层。每个 MTP head 预测下一个 token（head 1 预测 t+1，head 2 预测 t+2，依此类推）。
   - Loss：$\mathcal{L}_{total} = \mathcal{L}_{main} + \lambda \sum_{i=1}^{n_{mtp}} \mathcal{L}_{mtp_i}$，其中 $\lambda$ 是 MTP loss 的权重系数（通常较小，如 0.3）。

2. **Pipeline Layout 中的 `m`**：
   - `m` 符号代表 MTP layer。
   - 由于 MTP 层共享主 decoder 的 hidden states但又有独立的预测 head，它可能无法与主 decoder 层均匀切分到 PP stage 上（例如 MTP 层数远少于 decoder 层数）。
   - Custom pipeline layout 允许显式指定 `"Et*3|(tt|)*29,m|L"` 这样的字符串，将 MTP 层 (`m`) 和 loss 层 (`L`) 灵活放置到特定的 PP stage，避免 stage 间参数量失衡。

3. **MTP + Speculative Decoding**：
   - 在 inference 时，MTP 的多个 head 可以一次性生成多个未来 token 的 draft。
   - 这些 draft token 可以作为 speculative decoding 的 candidate，由主模型（或一个更小的 verifier）并行验证，从而加速 decoding。

</details>

---

### Q22. Hyper-Connections (mHC)
**考点**：mHC 的 residual stream 机制、fused kernel。

1. Manifold Hyper-Connection (mHC) 在 Transformer Layer 中引入了什么新结构？它与标准残差连接有何不同？
2. mHC 中的 `Sinkhorn iterations` 和 `gating factors` 分别起什么作用？
3. Megatron 为 mHC 提供了哪些 fused kernel 优化？

<details>
<summary>参考答案</summary>

1. **mHC 结构**：
   - 在标准残差连接之外，引入多个并行的 **residual streams**（残差流）。
   - 每个子层（如 attention 或 mlp）的输出不是简单地加回单一残差流，而是通过可学习的门控加权，分配到多个残差流中。
   - 这增加了模型表达能力和梯度流动性，但也增加了显存和计算开销。

2. **Sinkhorn iterations 和 gating factors**：
   - **Gating factors**：每个残差流和每个子层之间的连接强度，可学习。
   - **Sinkhorn iterations**：对 gating matrix 进行 Sinkhorn 归一化（行列双随机归一化），保证每个子层的输出总强度和每个残差流接收的总强度都有约束，防止某些流过度主导或消失。

3. **Fused kernel**：
   - `cuTile` kernel：将多个 residual stream 的加权和融合到单个 CUDA kernel 中，减少 memory-bound 的 element-wise 操作。
   - mHC 也支持 selective recomputation block，以控制显存峰值。

</details>

---

## 第五层：代码实现与调试

### Q23. ColumnParallelLinear 与 RowParallelLinear 的通信技巧
**考点**：自定义 autograd function、前向/反向的通信注入、工程原因。

阅读以下 Megatron-LM 风格的伪代码：

```python
class ColumnParallelLinear(nn.Module):
    def forward(self, x):
        x = f(x)           # 前向: identity, 反向: AllReduce
        y = F.linear(x, self.weight)
        return y

class RowParallelLinear(nn.Module):
    def forward(self, x):
        y = F.linear(x, self.weight)
        y = g(y)           # 前向: AllReduce, 反向: identity
        return y
```

1. `f` 和 `g` 具体是什么（从 autograd.Function 角度）？请写出它们的前向和反向逻辑。
2. 为什么 ColumnParallelLinear 的 AllReduce 要放在**反向**而不是前向？
3. 如果去掉 `f` 和 `g` 这两个自定义算子，直接在 `forward` 里调用 `torch.distributed.all_reduce`，会导致什么问题？

<details>
<summary>参考答案</summary>

1. **`f` 和 `g` 的实现**：
   - `f` 是 `identity` 在前向、`all_reduce` 在反向的自定义 `autograd.Function`。
     ```python
     class F(torch.autograd.Function):
         @staticmethod
         def forward(ctx, x):
             return x
         @staticmethod
         def backward(ctx, grad_output):
             torch.distributed.all_reduce(grad_output)
             return grad_output
     ```
   - `g` 是 `all_reduce` 在前向、`identity` 在反向的自定义 `autograd.Function`。
     ```python
     class G(torch.autograd.Function):
         @staticmethod
         def forward(ctx, x):
             torch.distributed.all_reduce(x)
             return x
         @staticmethod
         def backward(ctx, grad_output):
             return grad_output
     ```

2. **ColumnParallel 的 AllReduce 放在反向的原因**：
   - ColumnParallelLinear 的权重按输出维度切分。前向时每个 rank 计算自己的输出切片，无需通信（输出天然是分片的）。
   - 反向时，每个 rank 会计算输入 $X$ 的局部梯度。由于下一层（RowParallel）的输入梯度是完整的，因此需要对 $X$ 的梯度在 TP group 上做 AllReduce，才能得到正确的输入梯度。

3. **直接调用 `all_reduce` 的问题**：
   - 如果在前向里直接调用 `torch.distributed.all_reduce(x)`，这个操作不会被 autograd 追踪，反向传播图里不会包含它。
   - 结果是：梯度不会自动经过 AllReduce，导致 TP 各 rank 的梯度不一致，模型发散。
   - 另外，直接调用 `all_reduce` 无法与计算做 fine-grained overlap（自定义 Function 可以在 backward hook 中调度通信）。

</details>

---

### Q24. Pipeline Schedule 的 Warmup / Steady / Cooldown 阶段
**考点**：1F1B 调度的实现逻辑、last stage 的反向触发、VPP 的影响。

在 Megatron-LM 的 `schedule.py` 中，1F1B 调度分为 warmup、steady state、cooldown 三个阶段。
1. 对于中间 stage（非首非尾），warmup 阶段连续执行多少个 forward micro-batch 后才能进入 steady state？
2. Last stage 为什么能最早进入 steady state？它触发第一个 backward 的条件是什么？
3. 如果开启 VPP（virtual pipeline），warmup 阶段的 forward 次数如何变化？

<details>
<summary>参考答案</summary>

1. **中间 stage 的 warmup**：
   - 需要连续执行 $p-1$ 个 forward micro-batch（$p$ = physical stage 数），直到第一个 micro-batch 的数据从前面所有 stage 传递到达本 stage 并输出到 next stage。
   - 更准确地说，stage $i$ 需要等待 $i$ 个 micro-batch "填充"前面的 pipeline。

2. **Last stage 最早进入 steady state**：
   - Last stage 收到第一个 micro-batch 的 forward 输入后，由于它后面没有 stage 了，可以**立即**开始该 micro-batch 的 backward。
   - 因此 last stage 在 warmup 阶段只执行了 1 个 forward 后就进入 1F1B 交替模式。

3. **VPP 的影响**：
   - 开启 VPP 后，每个 physical stage 被拆分为 $v$ 个 virtual stages。
   - Warmup 阶段的 forward 次数增加，因为要填充更长的 virtual pipeline。
   - 具体公式：warmup 阶段的 forward 次数 = $v \times p - 1$（或取决于具体 schedule 变体）。
   - 虽然 warmup 变长，但 steady state 的 bubble ratio 降低到 $(p-1)/(m \cdot v)$，整体吞吐提升。

</details>

---

### Q25. Parallel State 初始化与进程组查询
**考点**：Megatron 的 rank ordering、进程组创建逻辑、代码级调试。

假设配置为 `TP=2, CP=2, PP=4, DP=4`，共 64 卡。
1. 请写出 `parallel_state.py` 中，GPU 0 所在的 `tensor_model_parallel_group`、`context_parallel_group`、`pipeline_model_parallel_group`、`data_parallel_group` 的成员列表。
2. 如果运行时 `torch.distributed.get_rank() == 0` 发现自己不在期望的 TP group 中，你会如何排查？请列出至少 3 个检查点。
3. 在代码中，`get_tensor_model_parallel_rank()` 和 `get_data_parallel_rank()` 分别在哪些场景下被调用？

<details>
<summary>参考答案</summary>

1. **GPU 0 的进程组（Megatron 默认 ordering: TP → CP → PP → DP）**：
   - `tensor_model_parallel_group`: `[0, 1]`
   - `context_parallel_group`: `[0, 2]`（注意：CP 在 TP 之后，同 TP tile 的下一个 CP slice）
   - `pipeline_model_parallel_group`: `[0, 4, 8, 12]`（PP 在 CP 之后，每个 PP group 跨越 TP×CP=4 个 rank）
   - `data_parallel_group`: `[0, 16, 32, 48]`（DP 在最后，每个 DP group 跨越 TP×CP×PP=16 个 rank）

2. **排查步骤**：
   - 检查 launcher 的 `rank` 和 `world_size` 是否正确（`torchrun` / `mpirun` 的 `--nnodes` / `--nproc_per_node`）。
   - 检查 `CUDA_VISIBLE_DEVICES` 和 `LOCAL_RANK` 的映射是否正确，是否有其他框架（如 DeepSpeed）抢先初始化了进程组。
   - 打印 `parallel_state.py` 中 `_TENSOR_MODEL_PARALLEL_GROUP` 的成员，与 `torch.distributed.get_process_group_ranks()` 交叉验证。
   - 检查是否在初始化 parallel state 之前就已经调用了 `torch.distributed.init_process_group()`，导致 backend 冲突。

3. **API 调用场景**：
   - `get_tensor_model_parallel_rank()`：在 `ColumnParallelLinear` / `RowParallelLinear` 中用于确定当前 rank 持有权重的哪个分片；在 checkpoint save/load 中用于确定保存哪个 TP shard。
   - `get_data_parallel_rank()`：在 distributed optimizer 中用于 shard optimizer states；在数据加载（dataloader）中用于确定当前 rank 应读取哪个数据分片；在 DP 梯度同步中用于确定 AllReduce 的 group 成员。

</details>

---

### Q26. 自定义 Pipeline Layout 的解析
**考点**：Custom pipeline layout 的字符串语法、模型切分逻辑。

Megatron-LM 支持 `--pipeline-model-parallel-layout "Et*3|(tt|)*29,m|L"` 这样的自定义布局字符串。
1. 请解析这个字符串，说明 `"E"`、`"t"`、`"m"`、`"L"`、`*3`、`|` 各自的含义。
2. 如果模型有 32 层 decoder + 1 层 embedding + 1 层 loss + 1 层 MTP，PP=4，这个布局应该如何调整才能尽量均衡各 stage 的参数量？
3. 在什么情况下必须使用 custom layout，而不能依赖默认的均匀切分？

<details>
<summary>参考答案</summary>

1. **字符串解析**：
   - `E`：Embedding 层
   - `t`：一个标准的 Transformer decoder 层（包含 attention + mlp/moe）
   - `m`：MTP (Multi-Token Prediction) 层
   - `L`：Loss 计算层
   - `*3`：重复前面的模块 3 次
   - `|`：Pipeline stage 的分隔符
   - `(...)`：分组，用于与 `*` 结合表示重复一个 block
   - 例：`Et*3|(tt|)*29,m|L` 表示：
     - Stage 0: E + t + t + t
     - Stage 1-29: 每个 stage 是 (tt|)，即 2 层 decoder 然后切分（注意这里的括号语法可能因版本略有不同，核心是支持灵活分组）
     - Stage 30: m
     - Stage 31: L

2. **PP=4 的均衡调整**：
   - 总层数相关模块：1 (E) + 32 (t) + 1 (m) + 1 (L) = 35 个逻辑单元。
   - PP=4 无法完全均分（35/4=8.75），需要让某些 stage 多一层。
   - 一个可能的 layout：`"Etttttttt|tttttttt|tttttttt|ttttmmmmL"`（stage 0-2 各 8 层 t，stage 3 8层 t + m + L，具体需根据 embedding 和 loss 的参数量微调）。

3. **必须使用 custom layout 的场景**：
   - 模型结构不平衡（如 embedding 层参数量巨大，或某几层是 MoE 层而其他层是 Dense 层）。
   - 包含非标准模块（如 vision encoder → LLM 的多模态模型，或 MTP 层数远少于主 decoder）。
   - 使用 Multi-Module Pipeline Communicator 连接不同并行拓扑的子模块时。

</details>

---

## 第六层：实战调优与排障

### Q27. GPU Util 高但 MFU 低的诊断
**考点**：CUDA kernel profiling、ncu/nsys、communication bubble、launch overhead。

你有一个 175B 模型在 256 张 A100 上训练，`nvidia-smi` 显示 GPU 利用率 98%，但实测 MFU 只有 18%。请给出系统化的诊断流程，并列出至少 4 种可能的原因及验证方法。

<details>
<summary>参考答案</summary>

**诊断流程**：
1. 使用 `nsys profile` 或 `ncu` 抓取一个 iteration 的 timeline。
2. 在 timeline 中观察：
   - 是否存在大块空闲（gap）？
   - 空闲期间 CUDA stream 上是否有 kernel 运行？
   - NCCL kernel（如 `ncclAllReduceRingLLKernel_sum_f32`）是否占据了大量时间？

**可能原因及验证**：

1. **大量 tiny kernel 导致 CPU launch bottleneck**：
   - 现象：timeline 上有很多极短的 kernel（<10μs），之间有较大的 launch gap。
   - 验证：检查是否有未 fuse 的 element-wise op（如多次 `add` + `mul` + `silu`）。可用 `torch.profiler` 的 memory view + trace view 确认。
   - 解决：开启 `fused_bias_gelu`、`fused_layer_norm`、`CUDA Graphs`。

2. **TP AllReduce 或 CP communication 未充分 overlap**：
   - 现象：大块 NCCL kernel 之间计算 kernel 很少或没有。
   - 验证：用 `nsys` 查看 TP group 的 AllReduce 是否和 GEMM 重叠；检查 `tp_comm_overlap` 是否开启。
   - 解决：开启 `tp_comm_overlap`，或改用 `reduce_scatter + all_gather` 的细粒度 overlap。

3. **Pipeline bubble 过大（micro-batch 数不足）**：
   - 现象：部分 GPU 长时间空闲，等待 P2P 数据。
   - 验证：计算理论 bubble ratio = $(p-1)/(m \cdot v)$，对比实际 timeline 的空闲比例。
   - 解决：增加 micro-batch 数（在显存允许范围内），或开启 VPP。

4. **Data loading / CPU preprocessing bottleneck**：
   - 现象：每个 iteration 开头有固定的 CPU 等待时间（`cudaStreamSynchronize` 或 `Host To Device` 拷贝间隙）。
   - 验证：检查 dataloader 的 `num_workers`、`prefetch_factor`，以及是否使用了 `torch.utils.data.DataLoader` 的默认 pinned memory。
   - 解决：增加 data workers、使用 `Megatron Energon` 等优化数据加载的框架。

5. **CUDA Graph 未开启导致的 dispatcher overhead**：
   - 现象：相同 shape 的 iteration，但每次 launch 的 kernel 顺序略有不同，或存在 PyTorch dispatcher 的额外开销。
   - 验证：对比开启/关闭 CUDA Graph 的 iteration 时间。
   - 解决：开启 `--cuda-graph-impl transformer_engine`（前提是输入 shape 固定）。

</details>

---

### Q28. OOM 的系统化排查
**考点**：显存占用分解、activation 峰值、fragmentation、NUMA 拓扑。

某训练任务在运行约 50 个 step 后突然 OOM，报错 `CUDA out of memory`。相同配置在其他节点上能正常运行。
1. 请给出一个显存占用的分解公式（Parameters + Optimizer States + Activations + Temp Buffers + Fragmentation）。
2. 如果 OOM 发生在特定 step（如 loss scale 骤降或序列长度突增），可能是什么原因？
3. 如果问题只出现在 `CUDA_VISIBLE_DEVICES=4,5,6,7` 的节点上，而与 `0,1,2,3` 无关，最可能是什么底层原因？如何验证？

<details>
<summary>参考答案</summary>

1. **显存分解公式**：
   - **Parameters**：模型参数总量。FP32 master + BF16/FP8 副本。
   - **Optimizer States**：Adam 需要 $2 \times$ 参数量（momentum + variance）；若用 distributed optimizer 则除以 DP size。
   - **Activations**：前向激活峰值。与 batch size、seq length、hidden size、checkpointing 策略相关。对于 Transformer，峰值约为 $O(bs \cdot seq \cdot h \cdot \text{layers} \cdot \text{factor})$。
   - **Temp Buffers**：梯度 buffer、all-reduce buffer、TP/CP 的通信 buffer、MoE dispatch 的临时 buffer。
   - **Fragmentation**：长期分配/释放导致的显存碎片。PyTorch caching allocator 的 `memory_stats` 中可查看 `allocated` vs `reserved` 的 gap。

2. **特定 step OOM 的原因**：
   - **Loss scale 骤降**：当 GradScaler 检测到 inf/nan 并降低 loss scale 时，如果代码路径中存在为特定 scale 预分配的 buffer（较少见），或数据中的异常 long sequence 导致 activation 突增。
   - **序列长度突增**：packed sequence 训练中，某个 batch 的 max sequence length 远大于平均值（如 padding 到 max length 时），导致 attention 的 $O(seq^2)$ 激活瞬间爆炸。
   - **验证方法**：在 `forward` 前后打印 `torch.cuda.memory_allocated()` 和 `max_memory_allocated()`；开启 `TORCH_CUDA_MEMORY_SUMMARY` 查看分配栈。

3. **CUDA_VISIBLE_DEVICES=4,5,6,7 导致 OOM 的原因**：
   - **最可能原因：NUMA / PCIe topology 不匹配导致 P2P 通信走系统内存或慢速路径，或 NCCL 环构建错误导致 fallback 到非 P2P 路径，激活了额外的 copy buffer。**
   - 更直接的原因：某些 GPU（如 4,5,6,7）不在同一个 NVLink domain 或 PCIe switch 下，而 Megatron 默认的 topology-aware 调度假设 rank 0-3 在同一 domain。如果进程绑定错误， intra-node 通信可能走 IB/RDMA 甚至 sysmem，产生大量临时 buffer。
   - **验证方法**：
     - 运行 `nvidia-smi topo -m` 查看 GPU 4-7 的互联拓扑。
     - 设置 `NCCL_DEBUG=INFO` 查看 NCCL 选择的 transport（NVLink / P2P / SHM / NET）。
     - 检查 `LOCAL_RANK` 与物理 GPU 的映射：确保 rank 0 绑定到 GPU 4 时，该 rank 的 TP group 伙伴（rank 1）绑定到 GPU 5 且两者有 NVLink/P2P 连接。
   - **修复**：使用 `mpirun` 的 `--bind-to` 或 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 确保 rank 到 GPU 的映射与物理拓扑一致。

</details>

---

### Q29. NCCL Timeout / Hang 的系统化排查
**考点**：分布式通信 hang 的定位、GDB、monitored_barrier、P2P 死锁。

训练过程中某节点报告 `NCCL operation timed out` 或进程完全 hang 住。
1. 如何第一步判断这是 **compute hang** 还是 **communication hang**？
2. 如果确认是 communication hang，如何定位是哪一个 process group（TP/PP/DP/CP/EP）出现了问题？
3. PP 的 P2P Send-Recv 最容易出现哪种死锁？代码中应该如何避免？

<details>
<summary>参考答案</summary>

1. **判断 compute hang 还是 communication hang**：
   - **方法 A**：所有 rank 设置 `NCCL_DEBUG=INFO`，观察 timeout 前最后一个 NCCL kernel 的输出。如果所有 rank 都在等待同一个 `ncclAllReduce` 或 `ncclSendRecv`，则是 communication hang。如果某些 rank 没有输出任何 NCCL 日志（即根本没进入通信），可能是 compute hang。
   - **方法 B**：使用 `torch.distributed.monitored_barrier()` 插入到可疑位置。如果 barrier 超时，说明至少有一个 rank 没到达该点（compute hang 或 communication hang 的源头）。
   - **方法 C**：用 `gdb -p <pid>` attach 到各 rank 进程，查看 backtrace。如果所有 rank 都卡在 `ncclKernel` 或 `cudaStreamSynchronize`，则是 communication hang；如果某些 rank 卡在 Python 层面的计算循环中，则是 compute hang。

2. **定位具体 process group**：
   - 在 `gdb` backtrace 中查看调用栈，找到 `torch.distributed.all_reduce` / `p2p_communication` 的调用位置。
   - 检查该位置的 group 参数（通过打印 `group.name` 或 `group.ranks()`）。
   - 如果 backtrace 显示在 `pipeline_parallel/schedules.py` 的 `send_forward` / `recv_forward` 中，则是 **PP group** 的 P2P hang。
   - 如果在 `tensor_parallel/layers.py` 的 `all_reduce` 中，则是 **TP group**。
   - 如果在 `distributed_optimizer` 或 `fsdp` 的 `reduce_scatter` 中，则是 **DP group**。
   - 对于 MoE，如果在 `flex_dispatcher` 或 `all_to_all` 中，则是 **EP group**。

3. **PP P2P 的死锁及避免**：
   - **最容易出现的死锁**：双向阻塞。例如 stage A 先执行 `send_forward` 再执行 `recv_backward`，而 stage B 先执行 `recv_forward` 再执行 `send_backward`。如果系统没有为 send 和 recv 分配独立的 buffer 或 stream，两个 stage 可能互相等待对方释放资源。
   - **避免方法**：
     - 确保 send 和 recv 使用 **配对顺序**（Megatron 的 `p2p_communication.py` 中严格规定了 send/recv 的顺序）。
     - 使用独立的 CUDA stream 或 **asynchronous P2P**（`overlap_p2p_comm`）来解耦发送和接收的同步关系。
     - 在 VPP 场景下，确保 virtual stage 间的 P2P buffer 分配不会互相覆盖。

</details>

---

### Q30. Checkpoint Resharding：从 TP=4/PP=4 加载到 TP=8/PP=2
**考点**：Checkpoint 转换、TP merge/split、PP 层重映射、QKV 的 permute。

你需要将一个已训练好的 Megatron checkpoint（TP=4, PP=4）迁移到新的集群上继续训练（TP=8, PP=2）。
1. 在 `tools/checkpoint` 的转换逻辑中，TP 维度的权重需要如何处理？对于 `ColumnParallelLinear` 和 `RowParallelLinear`，处理方向是否相同？
2. 如果原始模型使用了 `transformer_engine` 的 `fused_qkv_params`（即 QKV 权重是融合存储的），在 TP 维度变化时除了简单的 cat/split 还需要做什么额外操作？
3. PP 维度从 4 变到 2，意味着某些 stage 的层数翻倍。在 `load_checkpoint` 中，`pipeline_model_parallel_rank` 是如何影响读取哪个 checkpoint 文件的？

<details>
<summary>参考答案</summary>

1. **TP 维度的处理**：
   - **ColumnParallelLinear**（权重按输出维度切分）：新 TP=8 的每个 rank 需要持有更小的输出切片。因此需要将原始 4 个 TP shard **沿输出维度 split 成 8 份**。
   - **RowParallelLinear**（权重按输入维度切分）：新 TP=8 的每个 rank 需要持有更小的输入切片。因此需要将原始 4 个 TP shard **沿输入维度 split 成 8 份**。
   - 注意：split 方向相反。同时 bias 只存在于 ColumnParallelLinear，需随输出维度一起 split。

2. **Fused QKV 的额外操作**：
   - Fused QKV 权重在 TP shard 中通常不是简单的 `[Q_shard, K_shard, V_shard]` 拼接，而是经过特定的 **permutation/reordering** 以匹配 attention kernel 的 layout（如 `[q0, k0, v0, q1, k1, v1, ...]` 的 interleaved 格式）。
   - 因此不能直接 `torch.cat` 或 `torch.chunk`，而需要：
     1. 先将每个 TP shard **unfuse** 成独立的 Q/K/V 矩阵。
     2. 按新的 TP 度重新分片。
     3. 对每个新 shard 重新执行 **permute/fuse** 以匹配新 TP group 下的 kernel 期望格式。
   - Megatron 的 `split_query_key_value` 和 `merge_query_key_value` 工具函数就是为此设计的。

3. **PP 维度的 checkpoint 读取**：
   - Megatron 的 checkpoint 文件名通常包含 `mp_rank_0{tp_rank}_0{pp_rank}` 这样的后缀。
   - 原始 PP=4 时有 4 个 pipeline stage 的文件。新 PP=2 时，每个新 stage 需要加载原始 **两个 stage** 的层。
   - 例如，新 `pp_rank=0` 需要加载原始的 `pp_rank=0` 和 `pp_rank=1` 的 checkpoint 文件中的层参数。
   - 在代码中，通过设置 `mpu.set_pipeline_model_parallel_rank()` 和 `mpu.set_tensor_model_parallel_rank()`，让当前进程知道它对应哪个逻辑 rank，然后从正确的源 checkpoint 文件中提取匹配的层。

</details>

---

## 附录：快速参考

### 通信量公式
- **AllReduce**: $2(t-1)/t \cdot D \approx 2D$
- **AllGather**: $(t-1)/t \cdot D \approx D$
- **ReduceScatter**: $(t-1)/t \cdot D \approx D$
- **PP P2P (forward)**: $b \times s \times h \times \text{dtype_bytes}$

### Bubble Ratio
- **标准 1F1B**: $(p - 1) / m$
- **VPP 1F1B**: $(p - 1) / (m \cdot v)$

### 显存估算（每卡）
- **Parameters**: $N \times \text{dtype_bytes}$ (BF16=2, FP8=1)
- **Master Weights**: $4N$ (FP32)
- **Optimizer States**: $8N$ (Adam FP32) 或 $4N$ (BF16 optimizer states)
- **Activations**: 与 $bs \cdot seq \cdot h \cdot layers$ 成正比，checkpointing 可大幅降低

## Related Pages

- [[llm/06_infra/megatron-lm/overview]]
- [[Megatron-LM_MoE_Zero_Redundancy_Analysis]]
- [[Megatron_LM_TFLOPS_Analysis]]
- [[activation_checkpointing_analysis]]
