# Kimi K3 结构变化深析：同时优化序列轴与深度轴的信息流

> **来源基线**（所有 `file:line` 与论文页码均已打开核对）：
> - K3 官方 Tech Blog 快照 `raw/01_theory/01_models/moonshot_kimi/Kimi_K3_blog_2026-07-16.txt`（下称“博客”）；官方内嵌架构图原件 `assets/kimi_k3_official_arch.svg`。博客确认 K3 基于 KDA、AttnRes 与 Stable LatentMoE，但更多技术细节仍待完整报告披露（博客 `:207-210`）。
> - KDA：arXiv **2510.26692v2**（本库 `raw/.../Kimi_Linear_Attention-2510.26692.pdf`）；`MoonshotAI/Kimi-Linear@8c1d85e`；kernel 来自 `fla-org/flash-linear-attention@b328e7c` 的 `fla/ops/kda/*`；发布模型为 `moonshotai/Kimi-Linear-48B-A3B-Instruct@e1df551a`（`modeling_kimi.py`、`config.json`）。
> - AttnRes：arXiv **2603.15031v1**（2026-03-16）；`MoonshotAI/Attention-Residuals@85e2231`。仓库只有 README 和论文 PDF，**没有 `.py` 实现**；`README.md:52-91` 的伪代码对应论文 Fig. 2。
> - FlashKDA：`MoonshotAI/FlashKDA@d2ff19a`；LatentMoE：NVIDIA arXiv **2601.18089v1**。
> **事实边界**：K3 权重最迟于 2026-07-27 发布；完整技术报告尚无明确发布日期。对 K3 本体的判断以博客为准；机制细节来自官方明确点名的组件论文与源码；所有外推均标记为 **[推断]**。
> **更新**：2026-07-17，依据官方 Tech Blog 优化中文叙述和公式排版。

---

## 一、总览：六处变化如何汇成一条主线

| # | 部件 | K2 / K2.5(基线) | K3 | 一句话动机 | 出处 |
|---|---|---|---|---|---|
| 1 | 注意力主干 | 纯 MLA(61 层) | **KDA : Gated MLA = 3:1 混合** | 1M 上下文的 KV cache 与解码吞吐 | 博客架构图 3×/1×;Kimi Linear §4 |
| 2 | 全局注意力层 | MLA + RoPE | **Gated MLA**(输出门;预计 NoPE)| 注意力选择性、长度外推 | 博客 §Architecture;Kimi Linear §5.2 |
| 3 | 跨层连接 | 标准 pre-norm 残差 | **AttnRes**(深度方向 softmax 注意力) | 深层贡献稀释、梯度失衡 | 博客 §Architecture;AttnRes 论文 |
| 4 | MoE | 384 选 8 + 1 共享(专家全宽) | **Stable LatentMoE 896 选 16** + Quantile Balancing | 同算力更高稀疏度、更低 EP 通信 | 博客 §Architecture;LatentMoE 论文 |
| 5 | 激活函数 | SwiGLU 系 | **SiTU(Sigmoid Tanh Unit)** | 激活幅值控制(配 MXFP8)[推断] | 博客 §Architecture(仅一句) |
| 6 | 规模/上下文/模态 | 1.04T-A32B / 256K / 视觉 | **2.8T / 1M / +视频** | 结构效率红利再投入规模 | 博客开篇 |

Moonshot 的推进节奏是“先验证组件，再合入旗舰”：KDA 于 2025 年 10 月在 48B 模型上验证；AttnRes 于 2026 年 3 月继续使用同一套 Kimi-Linear 48B 骨架，并训练 1.4T tokens；到 2026 年 7 月，K3 才把两条路线合并并扩展到 2.8T 参数。KDA 和 AttnRes 的独立论文与开源实现，使得本文能够分析其机制；但这些组件实现仍不能替代尚未发布的 K3 本体代码。

官方只给出了合并口径：结构改动与训练/数据配方共同带来相对 K2 **约 2.5× 的整体 scaling efficiency**（博客 `:104-105`）。组件论文分别报告 KDA 混合架构约 1.16×、AttnRes 约 1.25× 的计算效率；两者简单相乘约为 `1.16 × 1.25 ≈ 1.45`。剩余差额是否来自 LatentMoE、数据或其他配方，目前只能作为 **[推断]**，不能视为官方分解。

---

## 二、变化点 1：注意力主干从纯 MLA 转向 KDA:MLA = 3:1

### 2.1 动机

K2.5 使用 61 层纯 MLA，原生上下文为 256K。若直接扩展到 1M，KV cache 会随序列长度线性增长，解码时每一步都要扫描更长的 KV。Kimi Linear 的实验显示，在 1M 上下文下，纯 MLA 的 TPOT 为 11.48 ms，而 3:1 混合架构只需 1.84 ms，相差约 6.3×（论文 Fig. 1b）。与此同时，agent 和 RL 负载越来越依赖长轨迹；在相同 Math RLVR 配方下，Kimi Linear 的训练与测试精度始终高于 MLA，且差距随训练扩大（§5.5 Fig. 6）。

### 2.2 机制

![KDA 模块：输入经过 q、k、v 投影和短卷积，由按 key 通道的遗忘门与按 head 的写入门更新定长状态，最后经过输出门和分头 RMSNorm。](assets/kimi_k3_fig_kda_block.png)

**状态更新**（论文 Eq. 1；参考实现 `fla/ops/kda/naive.py:59-63` 与之逐行对应）：

$$
\begin{aligned}
\widetilde{S}_{t-1} &= \operatorname{Diag}(\alpha_t) S_{t-1}, \\
S_t &= \left(I - \beta_t k_t k_t^{\top}\right)\widetilde{S}_{t-1}
      + \beta_t k_t v_t^{\top}, \\
o_t &= S_t^{\top}q_t.
\end{aligned}
$$

其中，`S_t` 是每个注意力头维护的 **128 × 128 定长状态矩阵**；`α_t` 控制按通道遗忘，`β_t` 控制当前键值对的写入强度。状态大小不随历史长度增长，因此解码阶段的状态访问成本与序列长度无关。

四个关键设计分别解决遗忘粒度、定向写入、输出稳定性和输入谱控制：

1. **细粒度遗忘门 `α_t`。** GDN 为每个头使用一个标量门，整个状态统一衰减（`fla/ops/gated_delta_rule/naive.py:31,54`）；KDA 则为每个 key 通道生成独立衰减率（`fla/ops/kda/naive.py:30-31`），把 GLA 式细粒度门引入 delta rule。其参数化采用低秩投影 2304→128→4096（`modeling_kimi.py:490-491`），门值计算见 `fla/ops/kda/gate.py:35,53`。
2. **写入强度门 `β_t`。** 它是每个头一个 sigmoid 标量（`modeling_kimi.py:496,561`），控制“先擦除 `k_t` 方向的旧信息，再写入当前键值对”的力度。与只做整体衰减的状态模型相比，这是定向的记忆更新。
3. **低秩 sigmoid 输出门与分头 RMSNorm。** 两者共同缓解 attention sink，并稳定梯度（论文 Eq. 10；`modeling_kimi.py:498-502,596-598`）。
4. **卷积和归一化预处理。** `q/k/v` 先经过核大小为 4 的 ShortConv 与 SiLU，`q/k` 再做 L2 归一化（`modeling_kimi.py:471-485,568-577`），用于约束状态更新的数值尺度。

**硬件效率来自对广义 DPLR 转移的特化。** KDA 把 DPLR 中的两个向量都绑定到 `k`，使二级 chunk 计算由 4 次减为 2 次，并进一步消去 3 个矩阵乘。论文报告其 chunkwise kernel 相比通用 DPLR 快约 100%（§3.2、Fig. 2）。训练和 prefill 的生产入口是 `chunk_kda`（`fla/ops/kda/chunk.py:178`）；短序列解码切到 `fused_recurrent_kda`（`fla/ops/kda/fused_recurrent.py:336`，`q_len ≤ 64` 的选择逻辑见 `modeling_kimi.py:523-525`）。

### 2.3 混合排布：只有四分之一的层保留完整 KV

Kimi-Linear-48B 的 `config.json:20-52` 给出了实际排布：27 层中，`full_attn_layers=[4,8,12,16,20,24,27]`，即 20 个 KDA 层和 7 个 MLA 层，比例约为 2.86:1。论文称其为 uniform 3:1；前 24 层确实严格重复六次“3 个 KDA + 1 个 MLA”。

KV cache 只存在于 MLA 层。`KimiDynamicCache` 为 KDA 层保存 convolution state 和定长 recurrent state，而不保存逐 token KV（`modeling_kimi.py:118-150`）。因此，相比所有层均使用 MLA，3:1 混合架构可直接减少约 75% 的 KV cache。K3 官方架构图继续使用 3× KDA、1× Gated MLA 的标注。

### 2.4 证据：1.4T tokens、相同配方下的对照

论文 Table 1 的层比消融如下：

| 配置 | Train PPL | Val PPL |
|---|---|---|
| 0:1(纯 MLA) | 9.45 | 5.77 |
| 1:1 | 9.29 | 5.66 |
| **3:1(采用)** | **9.23** | **5.65** |
| 7:1 | 9.23 | 5.70 |
| 15:1 | 9.34 | 5.82 |
| 3:1 去输出门 | 9.25 | 5.67 |
| 3:1 输出门换 swish | 9.43 | 5.81 |

在 MLA、GDN-H 与 Kimi Linear 的三方对比中，Kimi Linear 在四类任务上均领先（§5.4）：MMLU-Pro base 分别为 47.2、47.9、**51.0**；SFT 后的 GPQA-Diamond 为 57.1、58.6、**62.1**；RULER@128K 为 81.3、80.5、**84.3**；MRCR@128K 为 22.6、23.9、**29.6**。

**为什么不选择其他方案。** 纯 MLA 的质量略低，在 1M 上下文下 TPOT 又高出约 6.3×；Mamba2 没有 delta rule，在 Palindrome、MQAR 和 Stack 等合成任务上全面落后（Fig. 4）；GDN 的粗粒度标量门在短文本上略优于 MLA，但到长文本时反而落后，只有 KDA 的细粒度门在两端都占优（p.12）。Scaling law 拟合进一步给出：KDA 混合架构相对 MLA 约有 1.16× 的计算效率（Table 2）。

> 更完整的 Kimi Linear 论文分析(含 chunkwise 并行推导)见 [[kimi_linear_analysis]];本节聚焦 K3 采用视角与源码定位。

---

## 三、变化点 2：全局层由 MLA 升级为 Gated MLA

- **官方确认的变化。** K3 为四分之一的全局 MLA 层增加输出门；博客将其作用概括为提升 attention selectivity（博客 `:207-209`）。
- **为什么要加门。** Kimi Linear 为了与标准 MLA 做严格对照，实验模型中的 MLA 层刻意没有门控，并把“未来补门”列为后续方向。KDA 侧的消融已经显示 sigmoid 输出门有效：去掉输出门时 Val PPL 从 5.65 升至 5.67，换成 swish 门则恶化到 5.81（Table 1）。K3 的 Gated MLA 可以看作这项遗留设计的产品化。
- **NoPE 仍是推断，不是 K3 已披露规格。** Kimi-Linear 发布模型设置了 `"mla_use_nope": true`（`config.json:56`），并在 `modeling_kimi.py:378` 强制断言；其 RULER 消融中，NoPE 为 84.3，RoPE 为 78.8（Table 5）。这些证据说明 NoPE 与 KDA 混合架构相容，但 K3 是否沿用必须等待完整技术报告。
- **512 维 MLA 只能作为旁证。** 博客的 Kernel Optimization 案例包含“512-head-dimension MLA kernel”（博客 `:164-166`），与 Kimi-Linear 的 `kv_lora_rank=512`（`config.json:19`）一致；据此推测 K3 沿用 512 维压缩态是合理的，但仍属于 **[推断]**。

---

## 四、变化点 3：用 Attention Residuals 重写深度方向的信息聚合

### 4.1 动机：标准残差会无差别累加所有历史层

标准 Pre-Norm 残差可以展开为：

$$
h_l = h_1 + \sum_{i<l} f_i(h_i).
$$

这意味着第 `l` 层接收到的是所有历史子层输出的等权累加（AttnRes §2.1，p.3），会带来三个问题：

1. 隐状态范数随深度近似按 `O(L)` 增长，单个子层的相对贡献被持续稀释；
2. attention 与 MLP 只能读取同一个聚合态，无法按需选择早期表征；
3. 一旦某层信息被后续累加淹没，标准残差没有显式机制把它重新取回。

在 48B、1T tokens 的实验中，baseline 的输出幅值随深度单调增长，梯度也不成比例地集中在浅层（Fig. 5，p.10）。AttnRes 将标准残差视为“深度轴上的固定全 1 混合”，再把它替换成深度轴上的 softmax attention。**K3 在 2.8T 规模采用这套机制，很可能正是因为上述问题会随深度放大；这是基于组件论文的 [推断]，不是博客原文。**

### 4.2 机制

![标准残差与 AttnRes 对比：前者等权累加历史子层，后者使用可学习 pseudo-query 在深度方向对历史块做 softmax 加权取回。](assets/kimi_k3_fig_attnres.png)

**Full AttnRes**（论文 Eq. 1–4）对所有历史子层做深度注意力：

$$
\begin{aligned}
e_{i \rightarrow l} &= w_l^{\top}\operatorname{RMSNorm}(v_i), \\
\alpha_{i \rightarrow l} &=
\operatorname{softmax}_{i<l}\!\left(e_{i \rightarrow l}\right), \\
h_l &= \sum_{i=0}^{l-1}\alpha_{i \rightarrow l}v_i.
\end{aligned}
$$

- `v_0` 是 embedding，`v_i` 是第 `i` 个子层的原始输出；
- `w_l` 是第 `l` 个子层独有的 `d` 维 pseudo-query，与当前 token 内容无关，并且必须零初始化（§5，p.8）；
- RMSNorm 只作用于被检索的历史表示，避免大幅值层仅凭范数占据过高权重。

**Block AttnRes 是工程化版本，也是 K3 官方图展示的形态。** 它把 `L` 层划分为 `N` 个块：块内继续使用普通残差，只在块间执行 softmax attention（Eq. 5–6）。残差状态因此从单个 `hidden_states` 变成 `(blocks, partial_block)`：前者保存已完成块的表示，后者保存当前块内的普通残差。

官方仓库没有完整 Python 实现，唯一可执行描述是 `README.md:52-91` 的伪代码，与论文 Fig. 2 相同。每个子层增加 `attn_res_proj`（即 `w_l`）和 `attn_res_norm`，在 attention 前和 MLP 前各聚合一次，并在块边界封存当前块。K3 官方图中的 `(α, w)`、`Block n−1/n−2/n−3` 和 `Embedding` 正是这套记号；SVG 的 aria-label 也明确写着 “Block Attention Residuals architecture diagram”。

**额外开销受控。** 每个子层只新增一个 `d` 维向量和一个 RMSNorm。论文报告：在 pipeline parallel 训练下，端到端开销低于 4%；推理通过 two-phase 计算与 online softmax，把每 token、每层的残差 I/O 从 `3d` 提高到约 `5.5d`，仍远低于 mHC 的 `34d`，端到端延迟增幅低于 2%（Table 1、§4.2）。由于块数 `N` 固定，深度方向的“KV”规模也是有界的。

### 4.3 证据

论文使用五种规模的 MoE 做 scaling 对照，并刻意把超参数设置得更有利于 baseline（Table 2）。其中三组代表性结果如下：

| 激活参数/tokens | Baseline | Block AttnRes(N=8) | Full AttnRes |
|---|---|---|---|
| 194M / 38.7B | 1.931 | 1.909 | 1.899 |
| 436M / 87.9B | 1.766 | 1.746 | 1.737 |
| 528M / 119B | 1.719 | 1.693 | 1.692 |

拟合曲线的截距整体下移而斜率基本不变，对应 **约 1.25× 的等效计算效率**（§5.1）。在 Kimi Linear 48B、1.4T tokens 的下游实验中，AttnRes 将 GPQA-Diamond 从 36.9 提高到 44.4，Math 从 53.5 提高到 57.1，HumanEval 从 59.1 提高到 62.2（Table 3）。这组实验直接证明 AttnRes 可以装入 Kimi-Linear 骨架；把它称为 K3 主干的前身是基于发布时间与结构对应关系的 **[推断]**。

### 4.4 为什么不采用其他替代方案（Table 4，436M）

| 方案 | Val loss | 结论 |
|---|---|---|
| baseline | 1.766 | — |
| DenseFormer(静态标量) | 1.767 | 无效 ⇒ **输入依赖的权重才是关键** |
| mHC / Hyper-Connections | 1.747 | 弱于 softmax 版,且推理 I/O 34d |
| sigmoid 代 softmax | 1.741 | softmax 竞争归一化更优 |
| 滑窗只看近 8 层 | 1.764 | **访问远层比多看近层重要** |
| **Full AttnRes** | **1.737** | 采用(Block 版为工程形态) |
| input-dependent query | 1.731 | 更好 0.006,但每层多 d×d 投影+解码顺序访存 ⇒ **被弃用** |

可视化结果显示，AttnRes 学到的混合矩阵整体对角占优，说明局部层仍是主通路；与此同时，embedding 持续获得非零权重，模型也会主动学习跨层 skip，并在深度方向形成 attention sink（Fig. 8、§6.2）。

> 这与 [[deepseek_v4_analysis]] 中 DeepSeek V4 采用的 mHC（Hyper-Connections 系）形成路线对照。AttnRes 论文直接消融了 mHC：验证损失为 1.747，对比 Full AttnRes 的 1.737；推理 I/O 则分别为 `34d` 与约 `5.5d`。

---

## 五、变化点 4：从全宽 MoE 转向 Stable LatentMoE 896 选 16

![标准 MoE 与 Stable LatentMoE 对比：后者先降到 latent 空间，再从 896 个小专家中激活 16 个，聚合后升回模型宽度，并配合 Quantile Balancing。](assets/kimi_k3_fig_latentmoe.png)

- **官方披露的变化。** K3 使用 Stable LatentMoE，每个 token 从 896 个专家中激活 16 个；总专家数与激活专家数之比为 56，高于 K2.5 的 48。博客同时强调，在这一稀疏度下，路由和优化已经成为一等问题（博客 `:207-209`）。
- **LatentMoE 的可能机制。** K3 官方图在 router 前后各有一个 Linear，与 NVIDIA LatentMoE 论文中“先降到 latent 空间，在 latent 空间路由和执行专家，再升回模型宽度”的结构一致。它的直接收益是：专家不再对接完整的 `d_model`，同一参数预算可以容纳更多、更小的专家；EP dispatch/combine 传输的也是 latent 向量，all-to-all 通信量随之下降。由于 K3 完整报告尚未发布，这一对应关系及 “Stable” 的具体含义仍标为 **[推断]**。
- **Quantile Balancing 的官方边界。** 博客只说明它根据 router score 的分位数直接确定专家分配，从而去掉启发式更新和敏感的均衡超参数。它是否针对 K2 沿用的 loss-free balancing、分位数如何更新，以及是否保证每个专家固定 token 数，都要等待报告确认。
- **为何继续提高稀疏度。** 增加总专家数、保持每 token 激活量相对有限，可以提高参数容量而不同比例增加计算量；真正的约束转移到路由稳定性、专家并行通信和负载均衡。K3 将 LatentMoE、Quantile Balancing 与全平衡 EP 训练并列介绍，表明它试图从结构、算法和系统三个层面同时解除这些约束；三者之间的精确耦合仍属于 **[推断]**。

---

## 六、变化点 5：SiTU（Sigmoid Tanh Unit）

博客对 SiTU 只给出一句定义：SiTU 用于改善 activation control，而 Gated MLA 用于改善 attention selectivity（博客 `:207-209`）。截至 2026-07-17，没有公开论文、代码或可核验的公式说明 SiTU 位于何处、如何参数化。

从名称推测，它可能是在 SiLU/GLU 类门控激活中引入 tanh 的有界化设计；同一段落又紧接 MXFP8 激活量化，因此它也可能用于压制 activation outlier。**这两点都只是 [推断]。** 在完整技术报告披露前，本文不把 `tanh(x) ∈ [-1,1]` 这一数学性质进一步包装成 K3 的既定实现。

---

## 七、变化点 6：1M 原生上下文与原生视觉、视频

- **1M 是原生规格，不是事后外推。** Kimi-Linear 发布模型已设置 `model_max_length: 1048576`（`config.json:57`），长上下文本就是 3:1 混合与 NoPE 的设计目标；K3 延续为官方原生规格。BrowseComp 脚注进一步显示：使用完整 1M 窗口且不做 context management 时，模型仍能取得 90.4 分（博客 `:538-539`）。
- **多模态范围从图文扩展到视频。** K2.5 已使用 400M 参数的 MoonViT，在约 15T 图文混合 tokens 上继续预训练（[[kimi_k2.5_analysis]]）；K3 则明确在同一模型中原生处理文本、图像和视频（博客 `:76-77`、§Video Editing）。视觉塔及视频编码细节仍待完整报告披露。

---

## 八、汇总：每处改动解决了什么问题

| 变化点 | 解决什么 | 拒绝了什么(证据) | 代价 |
|---|---|---|---|
| KDA 3:1 | 1M KV cache/解码带宽;RL 长轨迹 | 纯 MLA(6.3× TPOT);Mamba2(合成任务全败);GDN(长文反转) | 全局层只剩 1/4;线性层状态管理复杂化(prefix cache 重做,见 infra 页) |
| Gated MLA | attention sink、选择性 | 无门(+0.02 PPL);swish 门(+0.16) | 低秩门投影(可忽略) |
| AttnRes | 深度轴信息稀释/梯度失衡 | DenseFormer(无效);mHC(弱且 I/O 6×);input-dependent query(好 0.006 但推理不友好) | 训 <4%、推 <2%;残差流变二元状态(PP/重计算要适配) |
| Stable LatentMoE + 896/16 | 同算力更高稀疏度;EP 通信 | 全宽专家继续堆;bias 启发式均衡(敏感超参) | 路由/优化难度上升("first-order challenges") |
| SiTU | 激活幅值控制(配 MXFP8)[推断] | 待报告 | 待报告 |
| 2.8T/1M/视频 | 效率红利变现 | — | 部署门槛 64+ 卡超节点(infra 页) |

## Related Pages

- [[kimi_k3_analysis]] — K3 发布总结、完整基准、限制与官方定位
- [[kimi_k3_infra_deepdive]] — 本页各项结构选择在训练与推理系统中的配套实现
- [[kimi_linear_analysis]] — KDA/3:1 混合的原始论文分析
- [[kimi_k2.5_analysis]] / [[kimi_k2_analysis]] — 直接前代与 2.5× 效率基线
- [[moba_analysis]] — Moonshot 前一代长上下文注意力(块稀疏路线)
- [[deepseek_v4_analysis]] — mHC(Hyper-Connections)路线对照
- [[01_theory/06_distributed_parallelism/index]] — EP/PP 并行背景
