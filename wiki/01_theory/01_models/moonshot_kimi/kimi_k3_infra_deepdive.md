# Kimi K3 训推 Infra 深析 — 结构、训练、推理三层联动:为 2.8T + 1M 上下文把每一层的账都算平

> **来源基线**:
> - K3 官方博客快照 `raw/01_theory/01_models/moonshot_kimi/Kimi_K3_blog_2026-07-16.txt`(下称"博客",infra 原句集中在 §Architecture and Infrastructure 与 §Availability)
> - K2 训练 infra 基线:arXiv 2507.20534(本库 `raw/.../Kimi_K2-2507.20534.pdf`,[[kimi_k2_analysis]]);K2 Thinking INT4:HF moonshotai/Kimi-K2-Thinking 模型卡
> - Mooncake:github.com/kvcache-ai/Mooncake README(FAST'25 Best Paper,[[mooncake_analysis]]);vLLM:issue #26201 与 PR #27654/#42406/#44539/#44848/#43833(GitHub API 核实合入状态);FlashKDA:MoonshotAI/FlashKDA @ `d2ff19a` + MarkTechPost 2026-04-30
> - OCP MX 格式:OCP Microscaling Formats v1.0 规范(经解读页逐条核对)
> **标记**:[官方]=第一手页面;[三方]=第三方报道;[推断]=基于已核实事实的推理。K3 技术报告未发布,凡 [推断] 均待回填核对。
> **更新**: 2026-07-17 初版

---

## 一、主线

K3 的 infra 叙事一句话:**结构选择(KDA/LatentMoE/MXFP4)每一项都同时是 infra 选择**——线性注意力换来 1M 上下文可服务,但逼着重做 prefix caching;高稀疏 MoE 换来参数效率,但逼出全平衡 EP 训练与 64+ 卡超节点;MXFP4 权重换来 2.8T 可部署,但要求 QAT 从 SFT 就开始。博客把这条链写得很直白(§Architecture and Infrastructure 一段内依次给出四组"选择→配套")。

## 二、训练侧

### 2.1 Per-Head Muon:从"per-head 修补"到"per-head 优化"

- **K2 基线**([[kimi_k2_analysis]];arXiv 2507.20534):MuonClip = Muon + QK-Clip——每步更新后按头检查最大注意力 logit $S^h_{max}$,超阈值 τ 则以 $\gamma_h=\min(1,\tau/S^h_{max})$ 重缩放该头的 Q/K 投影,15.5T tokens **零 loss spike**。per-head 粒度在 K2 里只用于**稳定性修补**。
- **K3**:"Per-Head Muon extends Muon by optimizing attention heads independently for more adaptive learning at scale"(博客 §Architecture [官方])——per-head 粒度升级为**优化本身**:每个注意力头独立做 Muon 的矩阵正交化更新,而非整个投影矩阵一起 [推断:Muon 的 NS 正交化以矩阵为单位,按头切分即每头 128×d 子矩阵独立正交化;精确定义待报告]。
- **为什么**:头与头的谱结构/更新尺度差异大(QK-Clip 的存在本身就是证据——总有个别头 logit 爆),按头独立优化让学习率/正交化适配每头的谱,是 MuonClip 经验的自然下一步 [推断]。

### 2.2 高稀疏 MoE 的训练配套:Quantile Balancing + 全平衡 EP

博客原句 [官方]:"To prevent expert imbalance from degrading throughput at large expert-parallel scales, we introduce a **fully balanced expert-parallel training method with static shapes and no host synchronization on the critical path**."

- **Quantile Balancing**(算法侧,详见 [[kimi_k3_architecture_deepdive]] §5):专家配额直接取 router 分数分位数,去掉 bias 启发式更新与敏感均衡超参。
- **全平衡 EP**(系统侧)[机制为推断,措辞为官方]:"static shapes" ⇒ 每步每卡的专家批大小固定,无动态 padding/重分配——kernel 可预编译、无 shape 抖动;"no host synchronization on the critical path" ⇒ 关键路径上没有 GPU→CPU 回读(容量统计、溢出判断之类),all-to-all 流水不被打断。两者合起来 = **把 MoE 训练做成静态图友好的确定性负载**,这正是 896 专家 / 大 EP 规模下吞吐不掉的前提。
- 与 Quantile Balancing 的耦合 [推断]:分位数配额天然产出"每专家恰好 top-q 个 token"的**恒定形状分配**,算法与系统在"静态 shape"上互为因果——这大概率就是"Stable LatentMoE"里 "Stable" 的一部分含义。

### 2.3 量化:INT4(K2 Thinking)→ MXFP4 权重 + MXFP8 激活(K3)

![量化路线演进图:上行为 K2 Thinking/K2.5——BF16 预训练+SFT 后在 post-training 阶段对 MoE 组件做 INT4 weight-only QAT,得原生 INT4(W4A16)推理、低延迟模式无损 2× 提速,选 INT4 的官方理由是兼容非 Blackwell 硬件并加速 RL;下行为 K3——预训练后自 SFT 起全程 QAT(MXFP4 权重+MXFP8 激活,贯穿 SFT→RL),部署即 MXFP4/MXFP8,2.8T 权重约 1.49TB;下方为 OCP MX 块格式示意(每 32 元素共享一个 E8M0 尺度因子,MXFP4 元素为 E2M1 即有效约 4.25bit/权重),原生硬件为 NVIDIA Blackwell 与 AMD MI355X,选开放 MX 标准而非私有 NVFP4 即"broad hardware compatibility"](assets/kimi_k3_fig_mxfp_qat.png)

- **K2 Thinking 的做法**(HF 模型卡 [官方]):INT4 weight-only、仅 MoE 组件、QAT 在 post-training 阶段;"lossless 2× speed-up in low-latency mode"。选 INT4 不选 FP4 的官方理由(工程师 AMA [三方转述]):兼容非 Blackwell 硬件 + 消解 thinking 模型长解码的 long-tail 低效、**加速 RL**。
- **K3 的升级**(博客 [官方]):"applies quantization-aware training **from the SFT stage onward**, using **MXFP4 weights with MXFP8 activations** for broad hardware compatibility"。三个跳变:(1) 时点提前——伪量化贯穿 SFT→RL 全后训练链路,权重分布在对齐阶段就适配 4-bit 网格,部署零 PTQ 损失 [推断];(2) 格式换轨——INT4→OCP MX 开放标准(块 k=32、E8M0 共享尺度、元素 E2M1),Blackwell/AMD CDNA4 原生支持;(3) 激活也进低精度(MXFP8),若训练硬件支持 MX GEMM 可顺带提速训练 [推断]。
- **体量账** [推断,算术]:2.8T @MXFP4 ≈ 1.4TB + 尺度开销 ≈ **~1.49TB 权重**——这直接决定了 §3.4 的部署门槛。

### 2.4 对照:K2 的训练系统基线(2.5× 效率宣称的分母)

K2 报告(arXiv 2507.20534 [官方]):H800 集群(节点 8 卡 + 2TB RAM,节点间 8×400Gbps RoCE);**16-way PP(虚拟 stage)+ 16-way EP + ZeRO-1 DP,无 TP**;显存三板斧=选择性重计算(LayerNorm/SwiGLU/MLA 上投影/MoE 下投影)、激活 FP8-E4M3 压缩存储、CPU offload 重叠;15.5T tokens 零 spike。K3 的 2.5× 是"overall scaling efficiency"(compute→intelligence 的转化率),分解到组件:KDA 混合 1.16×(Kimi Linear Table 2)× AttnRes 1.25×(AttnRes §5.1)≈1.45×,余量归 LatentMoE 稀疏度与配方 [推断];K3 训练集群规模/卡型未披露,TechCrunch 发布稿亦无成本数字 [三方,已核实原文无]。

## 三、推理侧

### 3.1 Mooncake:KVCache 中心的分离式架构(K3 官方 API 的底座)

![Mooncake 分离式推理示意:KVCache 感知调度器按缓存命中位置分派请求;Prefill 集群(算力密集,K3 时 KDA 层产状态快照、MLA 层产 KV)与 Decode 集群(带宽密集,64+ 加速卡超节点、大 EP 摊 896 专家)分离,之间经 Transfer Engine(RDMA,87–190GB/s)迁移 KVCache;底部为分离式 KVCache 池(Mooncake Store),把集群闲置 CPU/DRAM/SSD 池化成多级缓存以存储换重算,真实负载下同 SLO 多扛 75% 请求;K3 官方 API coding 负载缓存命中率 >90%,有效输入价约 $0.57/MTok](assets/kimi_k3_fig_mooncake.png)

- 定位 [官方 README]:"KVCache-centric Disaggregated Architecture for LLM Serving","Mooncake is the serving platform for Kimi"——FAST'25 **Best Paper**(论文 arXiv 2407.00079,详见 [[mooncake_analysis]])。
- 机制:Prefill/Decode 集群分离;闲置 CPU/DRAM/SSD 池化成分离式 KVCache 池;Transfer Engine 多协议 KV 传输(标称 87/190 GB/s 两档);"真实负载下同 SLO 多扛 75% 请求";K2 时代已支撑 128×H200 PD 分离部署,并可在数千卡训练集群 ~20s 完成权重更新分发(README [官方])。
- **K3 落点**(博客 §Availability [官方]):"Powered by Mooncake's disaggregated inference architecture, the official Kimi API achieves a **cache hit rate above 90% in coding workloads**" ⇒ 有效输入单价 ≈ 0.9×$0.30+0.1×$3.00 = **$0.57/MTok**。旁证:OpenRouter 流量观测 K3 命中率 77.7% [三方]——官方数字限定"official API + coding 负载",流量构成不同,不矛盾 [推断]。

### 3.2 KDA prefix caching:线性注意力逼出来的新基建

![KDA 前缀缓存问题与解法:左侧说明全注意力层 KV 逐 token 追加、天然支持按块哈希复用共享前缀,而 KDA 层只有一个定长状态逐 token 覆写、没有逐 token 历史可复用,传统 APC 对线性层失效;右侧为解法——prefill 时在每个 cache block 边界额外落一份 SSM/conv 状态快照,新请求命中共享前缀时 MLA 层直接复用 KV block、KDA 层恢复不超过前缀长度的最近快照并重算余段,有 all/align 两种模式;下方列出 vLLM 落地链:#27654 KDA 进 vLLM(2025-10)、#42406 hybrid align 前缀缓存(v0.25.0)、#44539 KDA conv state 统一布局、#44848 NIXL PD 分离、#43833 FlashKDA prefill](assets/kimi_k3_fig_kda_prefix_cache.png)

- **问题**:全注意力的 KV 逐 token 追加,按块哈希即可复用任意共享前缀;KDA 只维护**固定大小、逐 token 覆写**的 RNN 状态——没有"逐 token 历史"可复用,传统 APC 失效([推断],机制框架来自 vLLM issue #26201 [官方 vLLM])。
- **解法**(vLLM issue #26201):prefill 时在 cache block 边界 checkpoint 状态;命中共享前缀 ⇒ 恢复"≤前缀长度的最近快照"再重算余段;"all"/"align" 两模式(align 需 GPU kernel 后处理以避免 CPU-GPU 同步)。
- **落地链**(GitHub API 核实):PR **#27654**(KDA 进 vLLM,2025-10-28,Kimi Linear day-0)→ **#42406**(hybrid 模型 align 前缀缓存,进 v0.25.0)→ #44539(KDA conv state 统一 2-state 布局)→ #44848(KimiLinear 经 NIXL 的 PD 分离)、#43833(FlashKDA prefill backend,截至 2026-07-17 未合入)。
- **K3 官方表态**(博客 [官方]):"as KDA poses new challenges for conventional prefix caching, we have contributed a corresponding implementation to the vLLM community, to be released alongside the model. **KDA with prefill cache allows us to serve Kimi K3 at a highly competitive token price despite its scale and long context.**"(具体 PR 未点名;对应到上述工作线为 [推断]。)
- **与 Mooncake 的组合** [推断]:>90% 命中率的分母里,MLA 层复用的是 KV block、KDA 层复用的是状态快照——KVCache 池要同时管理两种工件;这解释了为何 K3 强调"KDA prefix cache"是服务成本的关键一环。

### 3.3 FlashKDA:prefill kernel 的第二代实现

- 定位(FlashKDA README [官方]):CUTLASS 手写**推理专用 forward kernel**(SM90+/CUDA≥12.9,K=V=128,`torch.inference_mode()`);装上后被 fla≥0.5.0 的 `chunk_kda` 自动分派(`fla/ops/kda/backends/flash_kda.py:35,115,139`),训练反向仍走 fla Triton。
- 设计要点(仓库 docs/20260420-flashkda-v1-deep-dive.md [官方]):**CHUNK=16**(配 gate lower_bound=−5,`exp(cumsum(g))` 塞进 bf16 免二级 rescale;16×16 求逆直接 Neumann 展开);**两 kernel 切分**(K1 token 并行做门激活/L2/构造+求逆,K2 head 并行做递推,拆分带来 ≥15% 端到端提速);数值技巧(片上状态 bf16 + fp32 FMA、16×16 逆用 fp16、`tanh.approx`/`ex2.approx`、寄存器内转置消 shared memory 往返);**varlen batching**(`cu_seqlens` 打包不等长序列,直接服务 continuous batching)。
- 数字(仓库 BENCHMARK [官方]):vs fla Triton `chunk_kda`,**H20 1.85×–2.31×**、GB200 1.70×–3.27×(T=8192, D=128);在 H20(对华出口版 Hopper)上给基准的信号是面向境内算力优化 KDA prefill [推断]。

### 3.4 部署门槛:64+ 加速卡超节点

- 官方原话 [官方]:"Since inference efficiency likewise benefits from larger high-bandwidth communication domains, we recommend deploying Kimi K3 on **supernode configurations with 64 or more accelerators**."(未点名机型;通篇用 "accelerators" 不用 "GPUs"。)
- **为什么需要** [推断,基于已核实数字]:(1) 权重驻留——MXFP4 下 ~1.49TB,单节点 8×H200(1.13TB)放不下,64 卡才有余量给 KV cache 与 1M 上下文;(2) EP 通信——896 选 16 的大 EP(如 64 卡 ≈ 每卡 14 专家)每层两次 all-to-all,必须落在 NVLink/UB 级 scale-up 域内而非跨节点 RoCE;(3) MoE 稀疏性要求大 batch/大 EP 以摊薄专家权重读带宽(LMSYS 对 K2 128×H200 部署的分析 [三方],prefill ~224k tok/s、decode ~288k tok/s、约 $0.21/1M 输出 token)。
- "supernode" 的现实对应 [官方各家]:NVIDIA GB200 NVL72(72 GPU 统一 NVLink 域、130TB/s、原生 FP4)与华为 CloudMatrix384(384 NPU Unified Bus 全互联,明言为 "large-scale MoE expert parallelism and distributed KV cache access" 优化,arXiv 2506.12708)。K3 同时选 MXFP4(开放标准)+ "accelerators" 措辞,兼容 NVL72 与国产超节点两条部署路线 [推断]。

## 四、事实边界与待报告确认清单

| 项 | 现状 | 待确认 |
|---|---|---|
| Per-Head Muon 精确定义 | 博客一句 [官方] | 按头切分粒度、与 QK-Clip 的关系、消融 |
| Quantile Balancing 公式 | 博客一句 [官方] | 分位数如何定义/更新、与 aux-loss-free 的对比数据 |
| 全平衡 EP 训练 | 博客一句 [官方] | token drop 与否、静态配额的精度代价 |
| MXFP4/MXFP8 QAT 细节 | 博客一句 [官方] | 伪量化算子位置、RL 阶段是否同精度、训练硬件 |
| K3 训练集群/成本 | 无任何可靠数字(TechCrunch 亦无)| 卡型、规模、token 数 |
| vLLM 贡献的具体 PR | "随模型发布" [官方] | 权重发布时的配套 PR/分支 |
| 2.5× scaling 效率的度量 | "overall scaling efficiency" [官方] | 是 loss-matched compute 比还是别的口径 |

## Related Pages

- [[kimi_k3_analysis]] — K3 发布总结
- [[kimi_k3_architecture_deepdive]] — 结构变化点(本页多数 infra 选择的结构侧上半场)
- [[mooncake_analysis]] — Mooncake 论文级分析(FAST'25)
- [[kimi_k2_analysis]] — K2 的 MuonClip 与训练系统基线
- [[kimi_k2.5_analysis]] — K2.5(INT4 沿用、Agent Swarm)
- [[kimi_linear_analysis]] — KDA 的效率证据与 vLLM day-0 集成
- [[muon_analysis]] — Muon 优化器原理(Per-Head Muon 的基座)
- [[moonshot_kimi/index]] — Kimi/Moonshot 技术路线总览
