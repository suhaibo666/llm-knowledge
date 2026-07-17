# Kimi K3 结构变化点深析 — 六处改动,一条主线:让信息在「序列长度」与「网络深度」两个轴上都流得更省、更准

> **来源基线**(所有 file:line / §页码均已实际打开核对):
> - K3 官方博客快照 `raw/01_theory/01_models/moonshot_kimi/Kimi_K3_blog_2026-07-16.txt`(下称"博客");官方内嵌架构图原件 `assets/kimi_k3_official_arch.svg`
> - KDA:arXiv **2510.26692v2**(本库 `raw/.../Kimi_Linear_Attention-2510.26692.pdf`);仓库 MoonshotAI/Kimi-Linear @ `8c1d85e`;kernel 在 fla-org/flash-linear-attention @ `b328e7c`(`fla/ops/kda/*`);HF **moonshotai/Kimi-Linear-48B-A3B-Instruct** @ `e1df551a`(`modeling_kimi.py` / `config.json`)
> - AttnRes:arXiv **2603.15031v1**(2026-03-16);仓库 MoonshotAI/Attention-Residuals @ `85e2231`(仅 README + 论文 PDF,**无 .py 实现**,伪代码 README.md:52-91 = 论文 Fig.2)
> - FlashKDA:MoonshotAI/FlashKDA @ `d2ff19a`;LatentMoE:arXiv **2601.18089v1**(NVIDIA)
> **事实边界**:K3 权重/技术报告 2026-07-27 前才发布。对 K3 本体的论断以博客为准;机制细节来自官方声明 K3 所基于的组件论文与源码;凡推断均标 **[推断]**。
> **更新**: 2026-07-17 初版

---

## 一、总览:变化点清单与"论文→产品"节奏

| # | 部件 | K2 / K2.5(基线) | K3 | 一句话动机 | 出处 |
|---|---|---|---|---|---|
| 1 | 注意力主干 | 纯 MLA(61 层) | **KDA : Gated MLA = 3:1 混合** | 1M 上下文的 KV cache 与解码吞吐 | 博客架构图 3×/1×;Kimi Linear §4 |
| 2 | 全局注意力层 | MLA + RoPE | **Gated MLA**(输出门;预计 NoPE)| 注意力选择性、长度外推 | 博客 §Architecture;Kimi Linear §5.2 |
| 3 | 跨层连接 | 标准 pre-norm 残差 | **AttnRes**(深度方向 softmax 注意力) | 深层贡献稀释、梯度失衡 | 博客 §Architecture;AttnRes 论文 |
| 4 | MoE | 384 选 8 + 1 共享(专家全宽) | **Stable LatentMoE 896 选 16** + Quantile Balancing | 同算力更高稀疏度、更低 EP 通信 | 博客 §Architecture;LatentMoE 论文 |
| 5 | 激活函数 | SwiGLU 系 | **SiTU(Sigmoid Tanh Unit)** | 激活幅值控制(配 MXFP8)[推断] | 博客 §Architecture(仅一句) |
| 6 | 规模/上下文/模态 | 1.04T-A32B / 256K / 视觉 | **2.8T / 1M / +视频** | 结构效率红利再投入规模 | 博客开篇 |

**"论文→产品"节奏**:KDA(2025-10 论文,48B 验证,[[kimi_linear_analysis]])→ AttnRes(2026-03 论文,**在同一个 Kimi-Linear 48B 骨架 + 1.4T tokens 上验证**)→ K3(2026-07,两者合体上 2.8T)。Moonshot 把结构创新先以独立论文+开源仓库发布、再在下一代旗舰合体——**K3 的结构源码今天就躺在 Kimi-Linear 的 HF modeling 文件和 fla 库里**,这是本页能做源码级分析的原因。

官方合并口径:三项结构改动 + 配方 ⇒ 相对 K2 **约 2.5× 整体 scaling 效率**(博客 §An Open 3T-Class Model)。可交叉的组件级数字:KDA 混合架构 1.16×(Kimi Linear Table 2)、AttnRes 1.25×(AttnRes §5.1),1.16×1.25≈1.45,余下缺口归 LatentMoE 稀疏度提升与数据/配方 [推断]。

---

## 二、变化点 1:注意力主干 —— 纯 MLA → KDA:MLA = 3:1

### 2.1 动机

K2.5 是 61 层纯 MLA、256K 上下文。要到 1M:KV cache 随长度线性涨、解码每步扫全量 KV。Kimi Linear 实测:**1M 上下文 TPOT,纯 MLA 11.48ms vs 3:1 混合 1.84ms(6.3×)**(论文 Fig.1b 图注);同时 agentic/RL 负载重心移向长解码——同配方 Math RLVR 下 Kimi Linear 训练/测试精度全程高于 MLA 且差距扩大(§5.5 Fig.6)。

### 2.2 机制

![KDA 机制图:x_t 分三路投影(q/k 走 Linear+ShortConv4+SiLU+L2Norm,v 走 Linear+ShortConv4+SiLU),外加低秩细粒度遗忘门 α_t(每 key 通道一个 decay,2304→128→4096)与每头标量 β_t;核心为 delta rule 状态更新 S_t=(I−β_t k_t k_tᵀ)Diag(α_t)S_{t−1}+β_t k_t v_tᵀ,o_t=S_tᵀq_t;输出经低秩 sigmoid 门 ⊙ 分头 RMSNorm 后投影输出](assets/kimi_k3_fig_kda_block.png)

**状态更新**(论文 Eq.1;参考实现 `fla/ops/kda/naive.py:59-63` 逐行对应):

$$S_t = (I - \beta_t k_t k_t^\top)\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top,\qquad o_t = S_t^\top q_t$$

每头一个 **128×128 定长矩阵状态**,解码成本与序列长度无关。四个关键设计:

1. **细粒度遗忘门 α_t ∈ (0,1)^{d_k}**:GDN 是每头标量门(`fla/ops/gated_delta_rule/naive.py:31,54`,整个状态乘同一标量);KDA 细化为每 key 通道一个(`fla/ops/kda/naive.py:30-31` docstring "Per-dimension decay gates (log-space)")——GLA 式细粒度门移植到 delta rule(论文 §1 p.2、§3 p.4)。参数化:低秩 2304→128→4096(`modeling_kimi.py:490-491`)+ `g=−exp(A_log)·softplus(g_raw+dt_bias)`(`fla/ops/kda/gate.py:35,53`)。
2. **β_t 写入强度门**(每头 sigmoid 标量,`modeling_kimi.py:496,561`):delta rule"先擦除 k 方向旧值、再写入新键值对"的力度——**定向记忆管理**,区别于 Mamba2 只衰减不擦写。
3. **低秩 sigmoid 输出门 + 分头 RMSNorm**(论文 Eq.10;`modeling_kimi.py:498-502,596-598`):缓解 attention sink、稳定梯度。
4. q/k/v 过 ShortConv(4)+SiLU,q/k 再 L2 归一(kernel 内完成,`modeling_kimi.py:471-485,568-577`)——控制谱半径,保训练稳定(§4 p.5)。

**硬件效率来源**:KDA = 广义 DPLR 转移矩阵中 **a、b 都绑定到 k** 的特化,二级 chunk 矩阵计算 4 次→2 次、再消 3 个 matmul,chunkwise kernel 比通用 DPLR 快约 100%(§3.2 + Fig.2)。生产入口:`chunk_kda`(`fla/ops/kda/chunk.py:178`),解码 `fused_recurrent_kda`(`fla/ops/kda/fused_recurrent.py:336`;q_len≤64 自动切,`modeling_kimi.py:523-525`)。

### 2.3 混合排布:只有 1/4 的层还留"真 KV"

Kimi-Linear-48B 实际排布(`config.json:20-52`):27 层中 `full_attn_layers=[4,8,12,16,20,24,27]`,即 **20 KDA : 7 MLA ≈ 2.86:1**(论文口径"uniform 3:1",前 24 层严格 3+1×6)。KV cache 只在 MLA 层存在:`KimiDynamicCache` 给 KDA 层只存 conv states + 定长 recurrent states(`modeling_kimi.py:118-150`)——**"KV cache ↓75%"的直接来源**。K3 官方架构图沿用同款 3×/1× 标注(博客架构图)。

### 2.4 证据(全部 1.4T tokens 同配方公平对比)

层比消融(论文 Table 1):

| 配置 | Train PPL | Val PPL |
|---|---|---|
| 0:1(纯 MLA) | 9.45 | 5.77 |
| 1:1 | 9.29 | 5.66 |
| **3:1(采用)** | **9.23** | **5.65** |
| 7:1 | 9.23 | 5.70 |
| 15:1 | 9.34 | 5.82 |
| 3:1 去输出门 | 9.25 | 5.67 |
| 3:1 输出门换 swish | 9.43 | 5.81 |

三方对比(MLA / GDN-H / Kimi Linear,§5.4):MMLU-Pro(base)47.2 / 47.9 / **51.0**;GPQA-D(SFT 后)57.1 / 58.6 / **62.1**;RULER@128k 81.3 / 80.5 / **84.3**;MRCR@128k 22.6 / 23.9 / **29.6**。

**为什么不是替代方案**:纯 MLA——质量还略输、效率差一个量级;Mamba2——无 delta rule,合成任务(Palindrome/MQAR/Stack)全败(Fig.4);GDN——粗粒度标量门**短文本略胜 MLA、长文本反转落后**,只有细粒度门两头赢(p.12)。Scaling law:对 MLA 1.16× 计算效率(Table 2)。

> 更完整的 Kimi Linear 论文分析(含 chunkwise 并行推导)见 [[kimi_linear_analysis]];本节聚焦 K3 采用视角与源码定位。

---

## 三、变化点 2:全局层 MLA → Gated MLA(+ 预计 NoPE)

- **改了什么**:给 1/4 的全局 MLA 层加输出门——"Gated MLA improve … attention selectivity"(博客 §Architecture)。
- **为什么(一次"补作业")**:Kimi Linear 论文里 MLA 层**故意不加门**以保证与标准 MLA 严格可比,并明说计划将来加上;KDA 侧消融已证 sigmoid 输出门有效(去门 5.67 vs 5.65;swish 恶化到 5.81,Table 1),且能缓解 attention sink(§4 p.6)。K3 把同款门补到 MLA 上,是论文遗留项的兑现。
- **NoPE 大概率随之而来 [推断]**:Kimi-Linear 发布模型的 MLA 全部 NoPE——`config.json:56` `"mla_use_nope": true`,代码 `modeling_kimi.py:378` 直接 `assert self.use_nope`,整个 modeling 文件无任何旋转操作(q_rot/k_rot 64 维只 concat 不旋转、MQA 广播,`modeling_kimi.py:392-410`)。理由:位置信息完全交给 KDA(gated delta rule ≈ 数据依赖的乘性位置编码,§6.1 Eq.11-12);消融 RoPE 版 RULER 78.8 vs NoPE 84.3(Table 5)——显式短程位置信号伤长上下文外推,且 NoPE 免调 RoPE 频率基/YaRN(p.7)。K3 是否同样 NoPE 待报告。
- **旁证**:博客 Kernel Optimization 案例任务集含"512-head-dimension 的 MLA kernel"(博客 §Kernel Optimization)——与 MLA 吸收态解码的 kv_lora_rank=512(`config.json:19`,DeepSeek-V3 同款)一致,暗示 K3 的 MLA 压缩维沿用 512 [推断]。

---

## 四、变化点 3:残差流 → Attention Residuals(深度方向的注意力)

### 4.1 动机:Pre-Norm 残差在"深度轴"上是个无门控的均匀累加器

标准残差展开 $h_l = h_1 + \sum_{i<l} f_i(h_i)$——所有前层输出恒权求和(AttnRes §2.1 p.3)。三个后果:‖h_l‖ 随深度 O(L) 增长、单层贡献被稀释;attention/MLP 拿同一坨聚合态、无法选择性访问早层;被淹没的信息不可找回。48B/1T tokens 实测:baseline 输出幅值随深度单调涨、梯度不成比例集中在最浅层(Fig.5 p.10)。**该问题随深度加剧——这是 K3 在 2.8T 上采纳它的规模逻辑 [推断]**。论文的统一视角:标准残差=全 1 下三角混合矩阵;Highway=深度方向 stick-breaking;Hyper-Connections/mHC=深度方向 linear attention(m-semiseparable);**AttnRes=深度方向 softmax attention(dense rank-L)**——把序列维上"linear→softmax"的跃迁在深度维复刻(§6.2 + Table 5 p.14-16)。

### 4.2 机制

![AttnRes 机制对比图:左为标准 Pre-Norm 残差(所有前层输出恒权 ×1 累加,导致范数 O(L) 增长与浅层梯度集中);右为 AttnRes(每子层一个零初始化的 d 维 pseudo-query w_l,对历史块输出 b_0..b_{n} 的 RMSNorm 做点积、沿深度 softmax 得 α,加权取回;块内仍普通残差,深度方向的 KV 数量固定为 N≈8;训练开销 <4%、推理 <2%)](assets/kimi_k3_fig_attnres.png)

**Full 版**(论文 Eq.1-4):$h_l=\sum_{i=0}^{l-1}\alpha_{i\to l}v_i$,$\alpha_{i\to l}=\mathrm{softmax}_i(w_l^\top\mathrm{RMSNorm}(v_i))$;v₀=embedding,vᵢ=各子层原始输出;**query 是每子层一个 d 维参数 w_l,与输入无关,必须零初始化**(§5 p.8);RMSNorm 只作用于 key,防大幅值层霸屏。

**Block 版(实用形态,K3 官方图即此)**:L 层切 N 块,**块内普通残差、块间 softmax attention**(Eq.5-6);残差流从单个 hidden_states 变成 `(blocks, partial_block)` 二元状态;伪代码全文即仓库唯一"实现"(README.md:52-91 = 论文 Fig.2):每子层新增 `attn_res_proj`(即 w_l)+`attn_res_norm`,attention 前、MLP 前各聚合一次,块边界封块。官方架构图上每子层旁的 (α,w) 与 Block n−1/n−2/n−3、Embedding 正是这套记号;官方 SVG 的 aria-label 就叫 "Block Attention Residuals architecture diagram"。

**开销**:每子层 1 个 d 维向量 + 1 个 RMSNorm;PP 下端到端训练开销 **<4%**(块表示 O(Nd) + cross-stage caching,§4.1);推理 two-phase + online softmax,每 token 每层残差 I/O 3d→**≈5.5d**(mHC 是 34d),端到端延迟 **<2%**(Table 1 + §4.2);N 固定 ⇒ 深度方向"KV"有界。

### 4.3 证据

Scaling(5 规模 MoE,超参刻意偏袒 baseline,Table 2):

| 激活参数/tokens | Baseline | Block AttnRes(N=8) | Full AttnRes |
|---|---|---|---|
| 194M / 38.7B | 1.931 | 1.909 | 1.899 |
| 436M / 87.9B | 1.766 | 1.746 | 1.737 |
| 528M / 119B | 1.719 | 1.693 | 1.692 |

截距整体下移、斜率持平 ⇒ **等效 1.25× 算力**(§5.1)。下游(Kimi Linear 48B + 1.4T tokens,Table 3):**GPQA-Diamond 36.9→44.4(+7.5)**、Math 53.5→57.1、HumanEval 59.1→62.2,全部 ≥ baseline。**这份"AttnRes 装在 Kimi-Linear 骨架上"的实验,基本就是 K3 主干的前身。**

### 4.4 为什么不是替代方案(Table 4,436M)

| 方案 | Val loss | 结论 |
|---|---|---|
| baseline | 1.766 | — |
| DenseFormer(静态标量) | 1.767 | 无效 ⇒ **输入依赖的权重才是关键** |
| mHC / Hyper-Connections | 1.747 | 弱于 softmax 版,且推理 I/O 34d |
| sigmoid 代 softmax | 1.741 | softmax 竞争归一化更优 |
| 滑窗只看近 8 层 | 1.764 | **访问远层比多看近层重要** |
| **Full AttnRes** | **1.737** | 采用(Block 版为工程形态) |
| input-dependent query | 1.731 | 更好 0.006,但每层多 d×d 投影+解码顺序访存 ⇒ **被弃用** |

学到的模式:对角占优(局部主通路)+ embedding 持续保留权重 + 学出跨层 skip + 深度方向 attention sink(Fig.8 + §6.2)。

> 与 [[deepseek_v4_analysis]] 中 DeepSeek V4 采用的 mHC(Hyper-Connections 系)形成路线对照:AttnRes 论文正面消融了 mHC(1.747 vs 1.737)并给出 I/O 账(34d vs 5.5d)。

---

## 五、变化点 4:MoE —— 全宽专家 384 选 8 → Stable LatentMoE 896 选 16 + Quantile Balancing

![Stable LatentMoE 对比图:左为常规 MoE(每个专家 up/down 投影对接全宽 d_model,专家数与 EP 通信量互相顶死,K2 稀疏度 48);右为 LatentMoE(Linear 降维到 latent → Router → 896 个 latent 维小专家选 16 + Shared Expert → Linear 升维,同参数量下专家数/稀疏度更高、EP all-to-all 传 latent 向量通信量下降,K3 稀疏度 56;配套 Quantile Balancing 以 router 分数分位数直接定专家分配)](assets/kimi_k3_fig_latentmoe.png)

- **改了什么**:"Kimi K3 uses Stable LatentMoE, effectively activating 16 of 896 experts. At this level of sparsity, routing and optimization become first-order challenges."(博客 §Architecture)。稀疏度 48→**56**,总参上到 2.8T。
- **LatentMoE 机制**([推断] 为其基座;命名与官方图 Router 前后两个 Linear 吻合):NVIDIA arXiv 2601.18089——token 先降维入 latent,**路由与专家计算都在 latent 空间**,聚合后升维;≤95B、>1T tokens 设计空间探索中每 FLOP/每参数精度一致优于标准 MoE,已被 Nemotron-3 Super/Ultra 采用(论文 abstract)。收益:专家投影不对接全宽 d_model ⇒ 同参数预算养**更多更小的专家**;EP dispatch/combine 传 latent 向量 ⇒ **all-to-all 通信量下降**(对 2.8T 的 64+ 卡 EP 部署是刚需,见 [[kimi_k3_infra_deepdive]] §3.4)。"Stable" 前缀的具体所指待报告。
- **Quantile Balancing**:"derives expert allocation directly from router-score quantiles, **eliminating heuristic updates and a sensitive balancing hyperparameter**"(博客原句)。[推断] 矛头指向 DeepSeek-V3 系(K2 沿用)的 loss-free balancing:每专家 bias 按负载启发式 ±γ,γ 即"敏感超参";改按 router 分数分位数切配额 = 把"控制器调参"换成"统计量定档",无需调速率、自适应分布漂移。
- **为什么稀疏度往上推**:固定激活、加大总专家数 → loss 单调降的方向已由 MoE scaling law 确立;真正约束是 infra(路由崩塌、EP 通信、负载不均)。K3 的答案 = 结构(LatentMoE)+ 算法(Quantile Balancing)+ 系统(全平衡 EP 训练,见 infra 页)三管齐下 [推断,依据博客并列陈述]。

---

## 六、变化点 5:SiTU(Sigmoid Tanh Unit)

博客全部信息一句:"Sigmoid Tanh Unit (SiTU) and Gated MLA improve activation control and attention selectivity respectively"(博客 §Architecture)。无论文、无代码、无检索可得的先行工作(2026-07-17 时点)。字面拆解 [推断]:门控激活家族(SiLU/GLU 系)引入 tanh 的有界化变体;"activation control"的直接受益者是 **MXFP8 激活量化**——tanh 压值域到 [−1,1]、消灭 outlier,恰是低精度 QAT 最头疼的东西(博客同段落即 QAT 表述)。**机制与位置必须等技术报告,不做进一步展开。**

---

## 七、变化点 6:1M 原生上下文 + 原生视觉/视频

- **1M 不是外推出来的**:Kimi-Linear 发布模型 `model_max_length: 1048576`(`config.json:57`),长上下文正是 3:1 混合 + NoPE 的设计目标;K3 直接继承为原生规格(博客开篇)。BrowseComp 用 1M 窗口、不做 context 管理仍 90.4(博客 §Footnotes)。
- **原生多模态**:K2.5 已用 MoonViT(400M)在 ~15T 图文混合 tokens 继续预训练([[kimi_k2.5_analysis]]);K3 升级为"text, images, and video within the same model"(博客 §Video Editing),视觉塔细节待报告。

---

## 八、汇总:每处改动的"为什么"

| 变化点 | 解决什么 | 拒绝了什么(证据) | 代价 |
|---|---|---|---|
| KDA 3:1 | 1M KV cache/解码带宽;RL 长轨迹 | 纯 MLA(6.3× TPOT);Mamba2(合成任务全败);GDN(长文反转) | 全局层只剩 1/4;线性层状态管理复杂化(prefix cache 重做,见 infra 页) |
| Gated MLA | attention sink、选择性 | 无门(+0.02 PPL);swish 门(+0.16) | 低秩门投影(可忽略) |
| AttnRes | 深度轴信息稀释/梯度失衡 | DenseFormer(无效);mHC(弱且 I/O 6×);input-dependent query(好 0.006 但推理不友好) | 训 <4%、推 <2%;残差流变二元状态(PP/重计算要适配) |
| Stable LatentMoE + 896/16 | 同算力更高稀疏度;EP 通信 | 全宽专家继续堆;bias 启发式均衡(敏感超参) | 路由/优化难度上升("first-order challenges") |
| SiTU | 激活幅值控制(配 MXFP8)[推断] | 待报告 | 待报告 |
| 2.8T/1M/视频 | 效率红利变现 | — | 部署门槛 64+ 卡超节点(infra 页) |

## Related Pages

- [[kimi_k3_analysis]] — K3 发布总结(基准全表、限制、定位)
- [[kimi_k3_infra_deepdive]] — 训推 infra(本页多处结构选择的系统侧下半场)
- [[kimi_linear_analysis]] — KDA/3:1 混合的原始论文分析
- [[kimi_k2.5_analysis]] / [[kimi_k2_analysis]] — 直接前代与 2.5× 效率基线
- [[moba_analysis]] — Moonshot 前一代长上下文注意力(块稀疏路线)
- [[deepseek_v4_analysis]] — mHC(Hyper-Connections)路线对照
- [[01_theory/06_distributed_parallelism/index]] — EP/PP 并行背景
