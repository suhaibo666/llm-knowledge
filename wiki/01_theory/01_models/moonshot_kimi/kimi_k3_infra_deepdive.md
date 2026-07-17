# Kimi K3 训推基础设施深析：结构、训练与推理如何共同支撑 2.8T + 1M 上下文

> **来源基线**：
> - K3 官方 Tech Blog 快照 `raw/01_theory/01_models/moonshot_kimi/Kimi_K3_blog_2026-07-16.txt`（下称“博客”）。基础设施相关原文集中在 §Architecture and Infrastructure 与 §Availability，即本地快照 `:207-227`。
> - K2 训练基础设施基线：arXiv 2507.20534（本库 `raw/.../Kimi_K2-2507.20534.pdf`，见 [[kimi_k2_analysis]]）；K2 Thinking INT4：Hugging Face 上的 `moonshotai/Kimi-K2-Thinking` 模型卡。
> - Mooncake：`kvcache-ai/Mooncake` README（FAST'25 Best Paper，见 [[mooncake_analysis]]）；vLLM：issue #26201 与 PR #27654、#42406、#44539、#44848、#43833（通过 GitHub API 核实合入状态）；FlashKDA：`MoonshotAI/FlashKDA@d2ff19a` 与 MarkTechPost 2026-04-30 报道。
> - OCP MX 格式：OCP Microscaling Formats v1.0 规范（经解读页逐条核对）。
> **标记**：`[官方]` 表示第一手材料，`[三方]` 表示第三方来源，`[推断]` 表示基于已核实事实的推理。K3 完整技术报告尚未发布，所有推断均需后续回填核对。
> **更新**：2026-07-17，依据官方 Tech Blog 优化中文叙述与公式排版；同日新增 §四「负载建模」——推理 decode/prefill 逐部件 roofline 记账 + 训练侧 8K→1M 逐模块 bound 推演（参数化公式，报告发布后代入真实维度即可刷新）。

---

## 一、主线

K3 的核心基础设施逻辑是：**每一项结构选择都会产生一项新的系统约束。** KDA 使 1M 上下文具备服务可行性，却迫使 prefix cache 从“缓存逐 token KV”升级为“缓存线性状态快照”；高稀疏 MoE 提高了参数效率，却需要静态 shape 的全平衡专家并行和更大的高速互联域；MXFP4 权重降低了 2.8T 模型的驻留成本，却要求从 SFT 阶段开始进行 QAT。官方博客在同一段中依次给出了这些“结构选择 → 系统配套”的关系（博客 `:207-210`）。

## 二、训练侧

### 2.1 Per-Head Muon：从按头修补升级为按头优化

- **K2 基线。** MuonClip 将 Muon 与 QK-Clip 结合：每次参数更新后，按注意力头检查最大 logit；若第 `h` 个头超过阈值 `τ`，就用系数 `γ_h` 同时缩放该头的 Q/K 投影（[[kimi_k2_analysis]]；arXiv 2507.20534）：

$$
\gamma_h = \min\!\left(1, \frac{\tau}{S_{\max}^{(h)}}\right).
$$

K2 报告称，这套机制在 15.5T tokens 训练中实现了零 loss spike。此时 per-head 粒度主要用于**事后稳定性修补**。

- **K3 的官方变化。** 博客称 Per-Head Muon “通过独立优化每个注意力头，使大规模训练更具适应性”（博客 `:207-209`）。这意味着 per-head 粒度从 clip 阶段前移到了优化器更新本身。
- **机制边界。** Muon 的 Newton–Schulz 正交化以矩阵为单位，因此一种自然实现是把 Q/K 投影按头切分后分别更新，而不是把整个投影矩阵视为一个整体。这样可以适应不同头的谱结构和更新尺度，但 K3 报告尚未披露切分维度、学习率共享方式及其与 QK-Clip 的关系；这些都仍是 **[推断]**。

### 2.2 高稀疏 MoE 的训练配套：Quantile Balancing + 全平衡 EP

官方明确给出的目标是：避免大规模专家并行中的负载不均拖低吞吐。为此，K3 引入了“静态 shape、关键路径无 host synchronization”的全平衡专家并行训练方法（博客 `:208-209`）。

- **算法侧：Quantile Balancing。** 博客称它直接依据 router score 的分位数确定专家分配，从而去掉启发式更新和敏感的均衡超参数。当前公开材料没有公式，详见 [[kimi_k3_architecture_deepdive]] §5。
- **系统侧：全平衡 EP。** “静态 shape”意味着 kernel 不需要在每一步应对变化的专家 batch；“关键路径无 host synchronization”意味着容量统计或溢出判断不能通过 device→host 回读阻断 all-to-all 流水。把这两点合起来，可以把 MoE 训练变成更适合静态编译和流水执行的确定性负载。
- **两者是否严格耦合仍待确认。** 分位数分配可能天然导出固定专家配额，从而与静态 shape 相互配合；但博客没有说明是否做到“每个专家恰好接收固定数量 token”，也没有说明这是否就是 “Stable” 的来源，因此这里只标为 **[推断]**。

### 2.3 量化：从 INT4 weight-only 转向 MXFP4 权重 + MXFP8 激活

![K2 到 K3 的量化路线：K2 在后训练阶段使用 INT4 weight-only QAT；K3 从 SFT 开始采用 MXFP4 权重与 MXFP8 激活，并以 OCP MX 块格式面向多类硬件。](assets/kimi_k3_fig_mxfp_qat.png)

- **K2 Thinking 的做法。** HF 模型卡披露的是 INT4 weight-only，只量化 MoE 组件，并在 post-training 阶段进行 QAT；官方宣称低延迟模式可获得“无损 2× 加速”。工程师 AMA 的第三方转述称，选择 INT4 而不是 FP4 是为了兼容非 Blackwell 硬件，并改善 thinking 模型长解码的低利用率。
- **K3 的升级。** 博客明确写道：K3 从 SFT 阶段开始进行 QAT，采用 MXFP4 权重和 MXFP8 激活，以获得更广泛的硬件兼容性（博客 `:208-209`）。相较 K2，这包含三项变化：QAT 时间点提前到整个后训练链路；格式切换到 OCP MX 开放标准；激活也进入低精度。至于 RL 是否全程保持相同伪量化配置、训练本身是否使用 MX GEMM，官方尚未披露。

仅按 4 bit 权重计算，2.8T 参数的理论下限为：

$$
\frac{2.8 \times 10^{12} \times 4\ \mathrm{bit}}{8}
= 1.4 \times 10^{12}\ \mathrm{bytes}
\approx 1.4\ \mathrm{TB}.
$$

再计入每 32 个元素共享的尺度等元数据，本文估算约为 **1.49 TB**。这是容量账，不是官方公布的模型文件大小；它解释了 §3.4 为何建议 64 卡以上超节点部署。

### 2.4 对照：K2 是“约 2.5× 效率”口径的基线

K2 报告（arXiv 2507.20534）披露的训练系统是：H800 集群，每节点 8 卡和 2 TB RAM，节点间使用 8 × 400 Gbps RoCE；并行策略为 16-way PP（含虚拟 stage）+ 16-way EP + ZeRO-1 DP，不使用 TP。显存优化包括选择性重计算、FP8-E4M3 激活压缩和与计算重叠的 CPU offload；官方报告称 15.5T tokens 训练期间没有 loss spike。

K3 的“约 2.5×”指 overall scaling efficiency，即计算投入到模型能力的总体转化效率，不是上述训练系统吞吐的直接倍率。组件论文可核验的数字只有 KDA 约 1.16× 与 AttnRes 约 1.25×，简单相乘约为 `1.45×`。其余收益归因于 LatentMoE 或数据配方只是 **[推断]**。K3 的训练集群规模、卡型和成本均未披露；TechCrunch 发布稿也没有可靠成本数字。

## 三、推理侧

### 3.1 Mooncake：K3 官方 API 的分离式推理底座

![Mooncake 分离式推理示意：KVCache 感知调度器按缓存位置分派请求；Prefill 集群负责算力密集的上下文处理，Decode 集群负责带宽密集的逐 token 解码，两者经 Transfer Engine 迁移缓存；Mooncake Store 将 CPU、DRAM 和 SSD 池化为多级缓存。K3 官方 API 在 coding 负载下的缓存命中率超过 90%。](assets/kimi_k3_fig_mooncake.png)

- **定位。** Mooncake 官方将其定义为“以 KVCache 为中心的分离式 LLM Serving 架构”，并明确说明 Mooncake 是 Kimi 的 serving platform。相关论文获得 FAST'25 Best Paper，机制详见 [[mooncake_analysis]]。
- **机制。** Prefill 与 Decode 使用不同集群；闲置 CPU、DRAM 和 SSD 被池化为分离式 KVCache；Transfer Engine 负责跨层级、跨节点迁移缓存。Mooncake README 报告，真实负载下在相同 SLO 约束内可多承载 75% 请求，并曾支持 128 × H200 的 Prefill/Decode 分离部署。
- **K3 的落点。** 官方博客称，Mooncake 支撑的 Kimi API 在 coding 负载下缓存命中率超过 90%（博客 `:223-227`）。按恰好 90% 命中估算：

$$
\begin{aligned}
C_{\text{input}}
&= 0.9 \times 0.30 + 0.1 \times 3.00 \\
&= 0.57\ \mathrm{USD/MTok}.
\end{aligned}
$$

这是根据官方价格计算出的上界估算，不是官方直接公布的平均账单。OpenRouter 观测到的 77.7% 是不同流量构成下的第三方数字，与“官方 API + coding 负载”的限定口径不能直接比较。

### 3.2 KDA prefix caching：线性注意力带来的新缓存形态

![KDA 前缀缓存示意：全注意力层复用 KV block；KDA 层则在 cache block 边界保存状态快照，命中共享前缀后恢复最近快照并重算尾段。](assets/kimi_k3_fig_kda_prefix_cache.png)

- **为什么传统前缀缓存失效。** 全注意力的 KV 按 token 追加，因此可以对缓存块做哈希并直接复用共享前缀。KDA 只维护一个固定大小、随 token 持续覆写的递归状态，并不保留完整的逐 token 历史；传统 Automatic Prefix Caching（APC）因而不能原样套用。这一机制解释来自 vLLM issue #26201，映射到 K3 的判断为 **[推断]**。
- **如何恢复可复用性。** vLLM 的设计是在 prefill 过程中，于 cache block 边界保存 KDA 状态快照。新请求命中共享前缀后，系统先恢复“不超过前缀长度的最近快照”，再重算快照之后的短尾段。`all` 模式保存所有候选快照；`align` 模式只保留对齐位置，并用 GPU kernel 完成后处理，以免引入 CPU–GPU 同步。
- **社区落地链。** GitHub API 显示：PR #27654 于 2025-10-28 将 KDA 纳入 vLLM，支持 Kimi Linear day-0；#42406 为 hybrid 模型加入 `align` 前缀缓存并进入 v0.25.0；随后 #44539 统一 KDA convolution state 的双状态布局，#44848 打通 Kimi Linear 经 NIXL 的 Prefill/Decode 分离。FlashKDA prefill backend 对应 #43833，截至 2026-07-17 尚未合入。
- **K3 官方结论。** 博客指出，KDA 给传统 prefix caching 带来了新问题，Moonshot 已向 vLLM 社区贡献相应实现，并计划随模型发布；具备 prefill cache 后，K3 才能在 2.8T 规模和 1M 上下文下维持有竞争力的 token 价格（博客 `:208-210`）。博客没有点名具体 PR，因此与上述 PR 链的对应关系仍是 **[推断]**。
- **与 Mooncake 如何组合。** 在 hybrid 模型中，MLA 层复用的是 KV block，KDA 层复用的是状态快照；Mooncake 一类缓存系统需要统一调度这两种生命周期不同的工件。这解释了 K3 为何把 KDA prefix cache 视为服务成本的关键环节，但具体存储布局尚未公开，属于 **[推断]**。

### 3.3 FlashKDA：面向 prefill 的第二代 kernel

- **定位。** FlashKDA README 将它定义为基于 CUTLASS 手写的推理专用 forward kernel，要求 SM90+、CUDA 12.9+，并固定 `K=V=128`。安装后，FLA 0.5.0 及以上版本可在 `chunk_kda` 中自动分派到该后端（`fla/ops/kda/backends/flash_kda.py:35,115,139`）；训练反向仍由 FLA 的 Triton 实现承担。
- **为什么选择 `CHUNK=16`。** 仓库设计文档给出的组合条件是 gate 下界为 −5。该 chunk 大小使 `exp(cumsum(g))` 可落在 bf16 可表示范围内，不必再做第二级 rescale；同时，16 × 16 的逆矩阵可以直接展开为 Neumann 级数。
- **为什么拆成两个 kernel。** K1 沿 token 维并行，负责门激活、L2 归一化、矩阵构造和求逆；K2 沿 head 维并行，负责递推。官方文档报告，这一拆分相较单 kernel 方案带来至少 15% 的端到端提升。实现还组合了 bf16 片上状态与 fp32 FMA、fp16 的 16 × 16 求逆、近似 `tanh`/`ex2`，并用寄存器内转置减少 shared memory 往返。
- **服务侧适配。** `cu_seqlens` 将不等长序列打包为 varlen batch，使该 kernel 可以直接进入 continuous batching 的 prefill 路径，而不必把所有请求 padding 到同一长度。
- **性能口径。** 仓库 BENCHMARK 在 `T=8192、D=128` 下，相比 FLA Triton `chunk_kda` 报告 H20 上 **1.85×–2.31×**、GB200 上 **1.70×–3.27×**。H20 结果说明实现者至少把境内常见算力纳入了优化和验证范围；更强的产品路线判断则属于 **[推断]**。

### 3.4 部署门槛：64+ 加速卡超节点

官方博客没有指定机型，但明确建议在 **64 个或更多加速器组成的超节点**上部署 K3，因为更大的高带宽通信域也能提高推理效率（博客 `:208-210`）。这里使用的是 “accelerators” 而非 “GPUs”，不应把建议绑定到单一厂商。

这一门槛可以从三个系统约束理解，但以下因果拆解均为 **[推断]**：

1. **权重驻留。** MXFP4 下的权重容量估算约为 1.49 TB，已超过 8 × H200 单节点约 1.13 TB 的总显存；扩展到 64 卡后，才更容易同时为运行时缓冲区、缓存与 1M 上下文留出空间。
2. **专家并行通信。** 896 个专家、每 token 激活 16 个专家，会在每层引入两次 all-to-all。若以 64 卡承载专家，平均约为每卡 14 个专家；通信最好留在 NVLink 或 Unified Bus 一类 scale-up 域内，而不是频繁跨越节点间 RoCE。
3. **稀疏模型的摊销效率。** 更大的 batch 和 EP 域可以摊薄专家权重读取成本。作为旁证，LMSYS 对 K2 128 × H200 部署的第三方分析给出了约 224k token/s 的 prefill、288k token/s 的 decode，以及约 0.21 USD/MTok 的输出成本；这些数字不能直接外推为 K3 的实测性能。

现实中的超节点形态包括 NVIDIA GB200 NVL72（72 张 GPU 处于统一 NVLink 域，原生支持 FP4）和华为 CloudMatrix384（384 张 NPU 通过 Unified Bus 互联，并面向大规模 MoE 专家并行和分布式 KV cache 访问优化，见 arXiv 2506.12708）。K3 选择开放的 MXFP4 标准，并采用厂商中性的 “accelerators” 表述，说明它在接口层面没有排除这两类平台；但具体完成了哪些平台适配，仍需等待权重、部署文档和完整技术报告确认。

## 四、负载建模：引入 KDA 后，每个模块的瓶颈是什么 bound

> **方法**：roofline + 每 token 字节/FLOP 记账。**K3 的隐藏维度、层数、激活参数均未公开**，本节用 Kimi-Linear-48B 的已核实 config 做精确模板，对 K3 只做量级估算，全部标注 **[推断]**。
> **模型假设 [推断]**：d=8192；64 层 = 48 KDA + 16 MLA；64 头；KDA 状态 128×128/头；MLA 压缩 KV 512+64；MoE 每层 896 专家（44M/专家，latent 维 d_lat=2048），激活 16+1 → 每层激活 ≈750M，全模型激活 ≈50B。
> **硬件锚点**（公开规格，凭记忆，量级可靠）：H200 ≈ 989 TFLOPS bf16 / 4.8 TB/s，knee ≈ 206 FLOPs/B；H20 ≈ 148 TFLOPS / 4.0 TB/s，knee ≈ 37 FLOPs/B；H800（K2 实际训练卡）bf16 峰值 989 TFLOPS，按 45% MFU 记 **450 TFLOPS 有效**，HBM 3.35TB/s，节点内 NVLink 400GB/s，节点间 8×400Gbps ≈ **50GB/s/卡**。
> 技术报告发布后，把真实维度代入本节的参数化公式即可整体刷新结论；公式列不随维度变。

### 4.1 总图：瓶颈发生了"迁移"而不是"消失"

```
纯 MLA 时代 decode 瓶颈:  attention KV 扫描(随 L 线性爆炸) —— 一家独大
K3(KDA 3:1)decode 瓶颈:  ① 剩余 1/4 MLA 层的 KV 带宽扫描(长上下文时)
                          ② MoE 专家权重带宽扫描(几乎任何 batch 下)
                          ③ EP all-to-all 同步延迟(每层两次,纯关键路径)
KDA 层自己:              降到"噪声级"(定长状态,~MB 量级/层/序列)
```

官方 infra 三件套与三个 bound 一一对应：MXFP4 打②（权重字节减半），64+ 卡超节点打③（a2a 留在 scale-up 域），KDA prefix cache + Mooncake 打的是 prefill 侧的算力 bound。

### 4.2 推理 decode 侧：逐部件记账（每 token、每序列）

**（1）KDA 层：带宽 bound，但绝对量是"噪声级"。** 每步每层做一次状态读改写。Kimi-Linear 模板（精确）：32 头 × 128×128 状态 × bf16 = **1MB/层**，读+写 2MB；20 层 KDA 合计 **~40MB/token/序列，与 L 无关**。FLOPs ≈ 4×128×128×32 ≈ 4 MFLOPs/层 → 算术强度 ≈ **2 FLOPs/B**，离 knee 两个数量级——**纯 HBM 带宽 bound**。但因为绝对量小，它不构成系统瓶颈；它的代价形态变了：**随 batch 线性增长**（每序列独立状态），且需要 prefix cache 快照管理（§3.2）。K3 若为 64 头/~48 KDA 层 [推断]，也就 ~200MB/token/序列，仍是零头。

**（2）Gated MLA 层：长上下文时的 decode 主 bound（带宽 + 容量）。** MLA 吸收态 decode 是 MQA：每层每序列读 `L×(512+64)×2B` 的压缩 KV。**1M 上下文 = 1.15GB/层/序列/步**。Kimi-Linear 7 层 MLA → 8GB/token；在 ~4.8TB/s 上仅 KV 扫描就要 **~1.7ms——与论文实测 TPOT 1.84ms@1M 几乎重合**（本节最硬的交叉验证：hybrid 模型 1M 解码的 TPOT ≈ 纯粹的 MLA 层 KV 带宽扫描时间，KDA 层 40MB 完全可忽略；论文测量所用 GPU 未披露，量级自洽）。算术强度上 MQA 让所有头共享一次 KV 读，≈ 4×n_heads/2B ≈ 128 FLOPs/B，已经不算惨，但仍在 knee 之下 → **带宽 bound**；容量上 `1M×576×2B×n_mla` ≈ 8GB（7 层）~18GB（16 层 [推断]）/序列 → **容量 bound 决定并发数**。3:1 混层的本质就是把这两项同时砍 4×。

**（3）LatentMoE 专家：decode 的"权重扫描" bound——MXFP4 正打在这里。** 单 token 激活 16+1 个专家，按 `2.8T×(16+1)/896` 估激活 MoE 参数 ~50B [推断] → MXFP4 下 **~28GB 权重字节/token（无 batch 摊销时）**——单序列解码是天方夜谭，必须靠 batch 共享专家权重读取。粗账：B 个 token/步时，专家权重流量 ≈ `min(B×28GB, 全量 1.49TB)`；B≳200 后基本"全专家命中"，流量封顶为**每步全量权重扫一遍：1.49TB ÷ 64 卡 ≈ 23GB/卡 ≈ 4.8ms@4.8TB/s**——这就是大 batch decode 的 TPOT 地板。要翻越到 compute bound，需要每专家每步 ≳ 数百 token（FP4 tensor core 的 knee 极高），折算全域 ≳ 3 万 token 在飞 [推断]——实际达不到，所以 **MoE decode 长期停留在权重带宽 bound**。这解释了两件事：MXFP4 相对 FP8 直接把地板砍半（字节减半，FLOPs 无所谓）；Quantile Balancing + 静态 shape 保证 64 卡扫描均匀，否则 bound 变成"最慢那张卡"（straggler bound）。LatentMoE 的小专家让同样字节预算下路由粒度更细，也是在这个 bound 里抠效率。

**（4）EP all-to-all：延迟 bound（不是带宽 bound）。** 每个 MoE 层 dispatch+combine 两次同步 a2a，~60 层 [推断] 就是每步 ~120 次同步点。字节量很小：`16×d_latent×MXFP8`，估 30-60KB/token/层、全模型 2-4MB/token [推断]——对 NVL72 的 130TB/s 或 CloudMatrix 的 UB 都是九牛一毛，**卡的是往返延迟 × 120 次的关键路径**（每次哪怕 10µs 也是 >1ms/步）。这就是"64+ 卡超节点"的真正含义：把 a2a 从跨节点 RoCE（几十 µs 级、有拥塞尾延迟）搬进 scale-up 域（µs 级、确定性），否则④会反超③成为 TPOT 主项。

**（5）AttnRes 与稠密投影：不构成 bound。** AttnRes decode 每子层残差 I/O 从 3d→5.5d（论文 Table 1），端到端 <2%，深度方向"KV"有界（N≈8 块）——噪声级。稠密投影/共享专家权重随 batch 摊销，并入（3）的权重扫描账里。

### 4.3 推理 prefill 侧：bound 换了两次主角

1. **无缓存 prefill = 算力 bound，且 MLA 的 O(L²) 只剩 1/4 权重。** 1M 序列的量级账 [推断]：MLA 二次项 ≈ 4.4 PFLOPs/层 × 16 层 ≈ 70 PFLOPs；MoE/稠密 GEMM ≈ `2×50B×1M` ≈ 100 PFLOPs——**1M 冷 prefill ≈ 170 PFLOPs ≈ 单 H200 三分钟**。KDA 层是线性的（chunkwise），在这本账里可忽略。所以 PD 分离（Mooncake）把这坨算力密集负载隔离到 prefill 集群是必然选择。
2. **KDA prefill kernel 自己的 bound 很特别：既不是带宽也不是大矩阵算力，而是"小矩阵效率"**——chunkwise 把递推转成 16×16/64×64 的小 matmul 链，tensor core 利用率和指令发射才是限制。这正是 FlashKDA 全部设计的靶子（CHUNK=16 塞进 bf16 数值域、双 kernel 拆分解决 K1 被 K2 低并行度拖死、寄存器内转置消 shared memory 往返，见 §3.3）。也解释了 H20 上收益最大（1.85–2.31×）：H20 knee 只有 ~37 FLOPs/B，低强度 kernel 在它上面天然不吃亏，优化空间全在调度。
3. **缓存命中 >90% 后，prefill 的 bound 再迁移一次：从算力变成"恢复带宽"。** 命中时的工作 = 从 Mooncake Store 拉回 MLA KV block + KDA 状态快照 + 重算尾段。拉回一个 1M 前缀 ≈ 8–18GB [推断]，Transfer Engine 标称 87–190GB/s → **50–200ms**，对比冷算三分钟是 100× 的差距——这就是"KDA with prefill cache 让 K3 能按这个价卖"的物理内涵：**>90% 的 prefill 负载被从算力 roofline 上搬到了存储/网络 roofline 上**，后者便宜得多。

### 4.4 推理侧汇总表

| 部件 | decode bound | prefill bound | 随什么增长 | K3 的对策 |
|---|---|---|---|---|
| KDA 层 | HBM 带宽(状态 R/W,~2 FLOPs/B),**绝对量噪声级** | 小矩阵效率/指令发射 | batch(不随 L) | FlashKDA;状态快照缓存 |
| Gated MLA 层(1/4) | **KV 带宽扫描 + 显存容量**(1M≈1.15GB/层/序列) | O(L²) 算力 | B×L | 3:1 混层砍 4×;MQA 共享读 |
| LatentMoE 专家 | **权重带宽扫描**(全量 ~1.49TB/步,≈4.8ms 地板/64 卡) | 算力(GEMM 大、强度高) | 封顶后不随 B | MXFP4 砍半;大 batch 摊销;Quantile Balancing 防 straggler |
| EP a2a | **同步延迟**(~120 次/步) | 带宽(token 多,可流水) | 层数×2 | 64+ 卡超节点;LatentMoE 缩字节 |
| AttnRes | 无(<2%) | 容量压力(块表示,TP 分片解决) | N 固定 | 论文 §4 工程即够 |
| 缓存系统 | — | 命中后:**恢复带宽**(Transfer Engine 87–190GB/s) | 前缀长度 | Mooncake 分级池化 |

**一句话：KDA 把"注意力"从 decode roofline 上抹掉之后，K3 的 decode 是一台"带宽机器"（MoE 权重 + 残余 MLA KV），TPOT 地板 ≈ max(权重扫描, KV 扫描, a2a 延迟)；prefill 在高命中率下是一台"搬运机器"。** 这也预示技术报告最值得盯的 infra 数字：MLA 层数与 KV 精度（决定①）、d_latent 与激活参数（决定②③）、以及是否做了 a2a 与计算的双 micro-batch 重叠（决定③能否被藏住）。

### 4.5 训练侧：随 L（8K→64K→256K→1M）逐模块记账

训练的关键变量变了——**哪些成本项随 L 增长、哪些是常数**，以及固定 token 预算下"序列变长 = 序列变少"对并行拓扑的连锁反应。系统假设：每卡每 micro-batch **8192 token**；EP=16（K2 拓扑）；fwd GEMM = 2NP，fwd+bwd = 6NP；每层时间都折算到"每卡 8K token 的一个 µbatch"。

**账本总纲：随 L 增长的只有一项。** GEMM（MoE+稠密+投影）fwd+bwd ≈ **300 GFLOPs/token（常数）**；MLA 注意力 fwd+bwd ≈ `16 层 × L × 64头 × (192+128) × 3` ≈ **10⁶ × L FLOPs/token（∝L）**；KDA chunkwise 每 token 只看本 chunk（C=64）+ 状态递推，**~2-3 GFLOPs（常数）**；AttnRes 计算常数、可忽略。

> **记账单位说明（防误读）**：本节所有"常数 / ∝L"均指**每 token** 成本。GEMM 的计算形态是 `[N_tok×d]×[d×d']`——每个 token 独立过同样的权重矩阵、token 之间无交互，故每 token 恒定 300 GF；**按序列算**的 GEMM 总量当然 ∝L（一条 1M 序列 = 1M×300GF = 3×10¹⁷ FLOPs，是 8K 序列的 128 倍），但训练预算按 **token 数**定：固定每卡每步 8192 token 时，这些 token 来自"一条 8K 序列"还是"一条 1M 序列的 1/128 段"，GEMM 工作量一模一样，变的只有注意力（每个 token 的历史从 8K 变成 1M）。注意力是全模型唯一让 token **两两交互**的模块，因此每 token 成本 ∝L、每序列 ∝L²。两个二阶效应不改结论：(1) L 增大只影响 GEMM 的**利用率**而非 FLOPs（单序列贡献的 M 维变大，但 8192 token/µbatch 下 M 维早已饱和）；(2) MoE 的 compute-bound 判据里每专家 token 数 t ∝ 每步 token 数，与 L 无关。稳健性检查：若改用"固定序列条数"口径，GEMM 与注意力的总量同乘序列数，**占比表各列数字不变**，只有绝对时间变。

| 项 | 随 L | 8K | 64K | 256K | 1M |
|---|---|---|---|---|---|
| GEMM(MoE/稠密) | 常数 | 300G | 300G | 300G | 300G |
| MLA 注意力(16 层) | **∝L** | 8G | 64G | 256G | ~1050G |
| KDA(48 层) | 常数 | ~3G | ~3G | ~3G | ~3G |
| **MLA 注意力占比** | — | **~3%** | **~18%** | **~46%** | **~78%** |

对照反事实：若 64 层全是 MLA，1M 时注意力 ≈ 4200 GFLOPs/token、占比 93%，每 token 总量 4500G vs K3 的 1350G——**KDA 3:1 对训练的意义与推理同构：把唯一随 L 爆炸的项压到 1/4，1M 训练每 token 代价 ≈ 8K 的 4.3×（纯 MLA 会是 ~15×）** [量级推断]。这就是 1M 原生上下文在经济上可训的结构前提。

#### 模块 1：MoE 专家 GEMM —— compute bound（与推理相反！）

**判据**：专家 GEMM 的算术强度 ≈ 每专家每步路由到的 token 数 t（FLOPs/权重字节 ≈ 2t/2B = t）。H800 knee ≈ 295 FLOPs/B → **t ≥ ~300 即 compute bound**。**代入**：EP 组内 16 卡 × 8192 = 131K token/步，每 token 17 个专家 → 分配数 = 2.2M，摊到 896 专家 → **t ≈ 2500/专家** ≫ 300 ✓。**时间**：每卡每层 fwd = 8192 token × 750M × 2 = **12.3 TFLOPs → 27ms**；fwd+bwd ≈ 82ms/层；64 层 fwd 合计 ≈ 1.7s/µbatch。**结论**：训练时 batch 巨大，推理侧的"权重扫描 bound"消失，专家 GEMM 稳定 compute bound——它是分母，别的模块都在跟它比"能不能藏进去"。这也解释了为什么 t≈2500 必须**均匀**：Quantile Balancing 若失效，t 的方差直接变成各卡步时方差（straggler bound）。

#### 模块 2：EP all-to-all —— 边缘 bound，LatentMoE 是它的救命稻草

**字节账**（每卡每层 fwd）：dispatch = `8192 token × 16 目标 × d_lat 2048 × 1B(MXFP8)` = **268MB**；combine 回程 bf16 = 537MB → **~0.8GB/层/卡**。EP=16 跨 2 个节点 → 走 IB 50GB/s → **16ms/层**。**对比计算**：16ms vs 专家 fwd 27ms → **占比 ~60%，只能靠通信-计算重叠藏住**，且余量很小（bwd 同比例）。反事实：若专家吃全宽 d_model=8192（无 LatentMoE），字节 ×4 → 64ms > 27ms，**a2a 直接成为主 bound，重叠也救不了**。**结论**：a2a 在训练里是"贴着计算跑的影子"——这就是官方把 LatentMoE（缩字节）、全平衡 EP（静态 shape，不打断重叠流水）、supernode（把 16ms 的 IB 换成 NVLink 域的 ~2ms）三件事绑在一起说的原因。**随 L 增大，每步 token 数不变而注意力计算暴涨 → a2a 占比自动下降，8K 阶段是它最危险的时刻。**

#### 模块 3：MLA 注意力 —— 唯一 ∝L 的项，256K 交叉、1M 统治

**公式**：每 token 每层 fwd ≈ `L × h × (d_qk+d_v)` = `L × 64 × 320` = **2×10⁴ × L FLOPs**；fwd+bwd（FA 反向自带重算）≈ ×3.5。**每层时间**（每卡 8K token µbatch，fwd）：

| L | 每 token FLOPs | 每层 fwd 时间 | vs MoE 层(27ms) |
|---|---|---|---|
| 8K | 0.16 GF | 3ms | 0.11× |
| 64K | 1.3 GF | 24ms | 0.9× |
| 256K | 5.1 GF | **93ms** | 3.4× |
| 1M | 20.5 GF | **373ms** | 13.8× |

16 层 MLA vs 64 层 MoE 的全模型份额：8K 时 3%、64K 18%、**256K 46%（交叉）**、1M 78%。**bound 性质**：FA 是分块大矩阵乘，intensity 数百 FLOPs/B，**贴峰值的 compute bound**——是所有 bound 里"最能用钱解决"的。CP 下的 ring 通信：每层轮转的 KV = `L×576×2B` ≈ 1.15GB@1M，而对应计算是秒级/层 → **通信占比 <1%，完全可重叠**（MLA 的 512 维压缩在这里第三次赚钱：GQA 全宽 KV 的 ring 字节会大 4-8×）。

#### 模块 4：KDA chunkwise —— 常数、噪声级，但反向有"状态检查点"内存账

**公式**：chunkwise 等效于窗口 ~2C 的注意力：每 token 每层 fwd ≈ `4×C×d_head×h` = `4×64×128×64` ≈ **2M FLOPs**——相当于 L≈100 的 MLA，**与 L 无关**。每层每 µbatch ≈ 41 GFLOPs → 即便按 30% 利用率（16×16 小矩阵，tensor core 吃不满，正是 FlashKDA 解决的"小矩阵效率"问题）也只有 **~0.3ms/层**，48 层合计 15ms ≈ 一层 MoE 的一半。**真正要算的是反向的状态内存**：bwd 需要各 chunk 边界状态，1M/64 = 16K 个边界 × 2MB（64头）= **32GB/层/序列**——不可能全存，fla 的做法是隔段检查点 + 重算，属于内存-重算折衷，不构成时间 bound 但吃显存预算。**CP 交互**：状态沿序列递推 → 跨段两遍扫描（先各段本地算段末状态、段间串行传 2MB、再带初值重算），字节可忽略、代价是每层一次串行同步 [机制推断]。

#### 模块 5：AttnRes —— 计算为零，1M 时是显存税

计算：每子层每 token 对 ≤9 个深度源做点积 = 9×8192 ≈ **74K FLOPs，四舍五入等于不存在**。内存：块表示 `8×L_local×d×2B`；1M/CP=16 → L_local=64K → **8.6GB/卡**，叠加在本已紧张的激活预算上；论文的序列维分片+融入 all-reduce 方案在 128K 实测把 15GB 压到 1.9GB/卡，1M 需同款手段放大一档。论文的 "<4% PP 开销"实测上限是 128K，**1M 下是否仍成立是技术报告要盯的点**。

#### 模块 6：优化器 —— Muon NS 的隐藏大账，Per-Head 恰好砍掉它

这是训练侧最有意思的一笔：**Newton-Schulz 正交化的代价 ∝ min(m,n)² × max(m,n)**（每次迭代 ~4m²n，5 次迭代 ≈ 20m²n）。**代入**（每卡分片 ~11B 参数，以 8192×8192 的注意力投影为例）：

- **整矩阵 Muon**：`20×8192²×8192` ≈ 11 TFLOPs/矩阵——每卡每步全部矩阵合计可达 **数百 TFLOPs ≈ 1s 量级**，与一个 µbatch 的计算同阶，不能忽视；
- **Per-Head Muon**（按头切成 64 片 128×8192）：每片 `20×128²×8192` ≈ 2.7 GFLOPs，64 片合计 **0.17 TFLOPs——64× 便宜**。

即：Per-Head Muon 除了官方说的"每头自适应学习"，还有一笔硬邦邦的算力账——**NS 成本被 min 维度平方压制，按头切片把注意力投影的正交化开销砍掉近两个数量级** [推断，算术本身是确定的]。ZeRO-1 侧：梯度 RS + 参数 AG ≈ 43GB/卡/步 @50GB/s ≈ 0.9s，必须与 bwd 重叠（标准操作），否则是 ~30% 的税。

#### 模块 7：PP 气泡 —— 随 L 恶化最快的"隐形 bound"

**公式**：气泡率 ≈ `(p−1)/m`，m = 每步每流水线的 µbatch 数。固定 token 预算下 **m ∝ 1/L**：8K 时序列多、m 充裕；1M 时一条序列 ≈ 一个（甚至跨多个）µbatch，16PP 想把气泡压到 20% 需要 m≥64——**1M 下凑不出来**。出路只有：PP 维度让位给 CP（注意力反正需要它）、加深虚拟流水交错、或 DualPipe 式双向调度。**这是"训练拓扑必须随 L 阶段重排"的定量原因**：K2 的 16PP×16EP×ZeRO-1 是 8K 拓扑，1M 阶段大概率变成 小PP × CP16+ × EP × ZeRO-1 [推断]。

#### 模块 8：激活显存 —— 决定 CP 下限的硬约束

一条 1M 序列的 hidden states = `L×d×64层×2B` = **1TB**（未计 AttnRes 8.6GB/卡与 KDA 检查点）。CP=16 分完还有 64GB/卡 → 必须叠加：选择性重算（把 78% 的注意力再算一遍，实际把注意力份额推向 ~85%）+ FP8 激活存储（K2 三板斧之一）+ 卸载。**显存不是性能 bound，但它决定 CP 的最小并行度，而 CP 又反过来决定模块 3/4/7 的通信与调度形态。**

### 4.6 训练侧四档序列长度的 binding constraint

**8K（预训练主体段）：瓶颈根本不在计算，在通信与调度。** 注意力仅 3%，FLOPs 全是 MoE GEMM——但 GEMM 是 roofline 上最好办的。真正 bind 的是：① **EP a2a**（每 MoE 层两次，能否被专家计算重叠决定 MFU 上限）；② **PP 气泡**（K2 基线 16PP×16EP×ZeRO-1，气泡率 ∝ p/m）；③ ZeRO-1 的梯度 reduce-scatter/参数 all-gather。短序列阶段 token 多、序列多、micro-batch 充足——**这一段是"全平衡 EP + 静态 shape + 关键路径零 host 同步"的主战场**：负载不均或一次 device→host 回读，打断的就是 a2a-计算重叠这根最紧的弦。

**64K（长上下文扩展第一档）：注意力开始显形（~18%），显存压力先于算力到来。** 单序列 hidden states ≈ `64K×8K×64×2B` ≈ 64GB [推断]——选择性重计算 + FP8 激活压缩（K2 三板斧）还能兜住，但单卡放整条序列已经紧张；典型对策是激活重计算加深 + 小规模 CP 试水。注意力 kernel（FA 系）此长度仍高效，bound 仍偏 ①②，但"每 micro-batch 时长"开始拉长，PP 调度粒度变粗。

**256K：交叉点，拓扑必须重排。** 本节估算的 MLA 注意力 ≈ GEMM 交叉恰在 L* ≈ 260K 附近。两件事同时发生：(a) 注意力计算升为并列主项；(b) 固定 token 预算下序列数骤减，**PP 的 micro-batch 数不够摊气泡**。出路是把并行维度从 PP/DP 挪给 **CP**：MLA 层走 ring attention 类方案，KV 块（每 token 仅 576 维，MLA 的压缩此处又赚一次）在 CP 组内轮转、与计算重叠。此档 bound = **注意力算力 + CP 通信重叠质量 + 拓扑重排本身**（PP↓、CP↑、DP↓）[推断]。

**1M：训练退化为"1/4 层的二次注意力工厂"，bound 反而"变好"了。** 注意力占 ~78%（重算后 ~85%），且它是 FA 式大矩阵乘——tensor core 友好、ring 可重叠，是 roofline 上**最容易用钱（FLOPs）解决的 bound**。真正的工程难点是三个次生问题：① KDA × CP 的串行 scan（两遍扫描，见模块 4）；② AttnRes 块表示显存税（见模块 5）；③ 重计算放大——选择性重计算把 O(L²) 注意力在 bwd 再算一遍，1M 时这笔最贵，预计对 MLA 层单独调策略（保留 softmax 统计量、只重算 AV 之类）[推断]。

**汇总矩阵**：

| L | 主导 FLOPs | 第一 bound | 第二 bound | 结构红利在哪 |
|---|---|---|---|---|
| **8K** | MoE GEMM(97%) | **EP a2a 重叠余量**(16ms vs 27ms/层) | PP 气泡、ZeRO/Muon 重叠 | LatentMoE 缩 a2a 4×;Per-Head Muon 砍 NS 64× |
| **64K** | MoE(~80%) | 同上,余量变大 | 激活显存起步(64GB/序列级) | KDA 48 层零增长 |
| **256K** | MoE≈MLA(各半) | **MLA 注意力算力**(93ms vs 27ms/层) | 拓扑重排:m 崩塌→PP 让位 CP | MLA 576 维 KV 使 ring 通信 <1% |
| **1M** | MLA 注意力(78%→重算后 ~85%) | **纯注意力算力**(373ms/层) | AttnRes 显存税、KDA 跨段 scan、气泡 | 3:1 混层:每 token 1350 GF,纯 MLA 要 4500 GF(**3.3×**)|

### 4.7 两个反直觉结论 + 一句话总结

- **MoE 的 a2a 压力随 L 反而变轻**：a2a 字节量 ∝ 每步 token 数（常数），而可用来遮它的计算随 L 涨——8K 时 a2a 是最紧的弦，1M 时它几乎免费。所以"全平衡 EP/静态 shape"这套是**为预训练主体段（短序列）设计的**，长上下文段的矛盾自动转移。
- **训练和推理的瓶颈迁移方向相反**：推理引入 KDA 后瓶颈从注意力迁到 MoE 权重带宽；训练里 MoE 恒定、注意力随 L 回来——**1M 训练的主项恰是那 1/4 没被 KDA 替掉的 MLA 层**。换句话说，3:1 这个比例同时钉住了两头：再稀（15:1）长文质量掉（Kimi-Linear Table 1），再密（1:1）1M 训推两侧的账都翻倍。
- **一句话**：训练侧的 bound 随 L 从"通信/调度问题"（8K，a2a 贴着专家计算跑）单调迁移到"纯算力问题"（1M，1/4 的 MLA 层吃掉八成 FLOPs）；KDA 的作用不是消灭这个迁移，而是把终点的账单除以 3.3，把 1M 原生预训练从"不经济"拉回"贵但可付"。

## 五、事实边界与待报告确认清单

| 项 | 现状 | 待确认 |
|---|---|---|
| Per-Head Muon 精确定义 | 博客一句 [官方] | 按头切分粒度、与 QK-Clip 的关系、消融 |
| Quantile Balancing 公式 | 博客一句 [官方] | 分位数如何定义/更新、与 aux-loss-free 的对比数据 |
| 全平衡 EP 训练 | 博客一句 [官方] | token drop 与否、静态配额的精度代价 |
| MXFP4/MXFP8 QAT 细节 | 博客一句 [官方] | 伪量化算子位置、RL 阶段是否同精度、训练硬件 |
| K3 训练集群/成本 | 无任何可靠数字(TechCrunch 亦无)| 卡型、规模、token 数 |
| vLLM 贡献的具体 PR | "随模型发布" [官方] | 权重发布时的配套 PR/分支 |
| 2.5× scaling 效率的度量 | "overall scaling efficiency" [官方] | 是 loss-matched compute 比还是别的口径 |
| §四负载建模的模型维度 | 全部 [推断](d=8192、64 层=48 KDA+16 MLA、64 头、d_lat=2048、激活 ≈50B) | 真实 hidden/层数/头数/d_latent/激活参数——代入 §四 公式即可整体刷新 |
| MLA 层数与 KV 精度 | [推断] 16 层、bf16 | 直接决定 decode 主 bound①(KV 扫描)与容量账 |
| 长上下文训练的 CP 方案 | 未披露 | ring vs 双遍扫描;KDA 段间 scan 实现;各 L 阶段的 PP/CP/EP/DP 拓扑(对照 K2 16PP×16EP×ZeRO-1) |
| AttnRes 在 1M 训练的实测开销 | 论文 "<4%" 只测到 128K [官方论文] | 1M 下块表示分片与开销是否仍 <4% |
| a2a 与计算的重叠机制 | 未披露 | 是否有 DualPipe 式双 micro-batch 重叠;decode 侧 a2a 延迟怎么藏 |
| 长上下文各阶段课程 | 未披露 | 8K→1M 的 L 课程与 token 配比;RL rollout 与训练算力配比 |

## Related Pages

- [[kimi_k3_analysis]] — K3 发布总结
- [[kimi_k3_architecture_deepdive]] — 结构变化点(本页多数 infra 选择的结构侧上半场)
- [[mooncake_analysis]] — Mooncake 论文级分析(FAST'25)
- [[kimi_k2_analysis]] — K2 的 MuonClip 与训练系统基线
- [[kimi_k2.5_analysis]] — K2.5(INT4 沿用、Agent Swarm)
- [[kimi_linear_analysis]] — KDA 的效率证据与 vLLM day-0 集成
- [[muon_analysis]] — Muon 优化器原理(Per-Head Muon 的基座)
- [[moonshot_kimi/index]] — Kimi/Moonshot 技术路线总览
