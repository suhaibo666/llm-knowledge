# DeepSeek-V4 深度解析

> **核对基线**: arXiv:**2606.19348v1** (DeepSeek-AI, **2026-04-26**) ＝ `raw/01_theory/01_models/deepseek/DeepSeek_V4.pdf`  
> **作者**: DeepSeek-AI　**初稿摄入**: 2026-04-24（基于预发布 PDF）　**对正式版复核/订正**: 2026-06-25  
>
> [!note] 本页超参表与基准数字已逐项对正式发表版核对（[[30_deepseek_v4_audit_analysis]]），结论：**数字全部一致**；
> 已订正预发布残留（FP4 归属、章节口径），并对正文混入的 1 处臆造小节加 `> [!contradiction]` 标注。

---

## 概述

DeepSeek-V4 引入了**混合注意力架构**，将**压缩稀疏注意力（CSA）**与**重度压缩注意力（HCA）**相结合，实现了高效的百万 token 长上下文处理。该系列包含两个模型：**V4-Pro**（总参数 1.6T / 每 token 激活 49B）和 **V4-Flash**（总参数 284B / 每 token 激活 13B），均原生支持 100 万 token 上下文。

相比 V3 的核心架构升级：
1. **CSA + HCA 混合注意力** —— 在序列维度压缩 KV，结合稀疏选择
2. **mHC（流形约束超连接）** —— 通过双随机矩阵约束增强残差连接稳定性
3. **Muon 优化器** —— 更快收敛、更好的训练稳定性

在 100 万 token 场景下，V4-Pro 的单 token 推理 FLOPs 仅为 V3.2 的 **27%**，KV Cache 仅为 **10%**。

---

## 模型规格

| 属性 | V4-Pro | V4-Flash |
|------|--------|----------|
| 总参数量 | 1.6T | 284B |
| 每 token 激活参数量 | 49B | 13B |
| Transformer 层数 | 61 | 43 |
| 隐藏维度 | 7168 | 4096 |
| CSA 压缩率（$m$） | 4 | 4 |
| HCA 压缩率（$m'$） | 128 | 128 |
| Query 头数（$n_h$） | 128 | 64 |
| 头维度（$c$） | 512 | 512 |
| Query 压缩维度（$d_c$） | 1536 | 1024 |
| 输出投影分组数（$g$） | 16 | 8 |
| SWA 窗口大小（$n_{\text{win}}$） | 128 | 128 |
| 路由专家数 | 384 | 256 |
| 每 token 激活专家数 | 6 | 6 |
| mHC 扩展因子（$n_{hc}$） | 4 | 4 |
| 上下文长度 | 100 万 token | 100 万 token |

---

## 混合注意力：CSA + HCA

### 1. 压缩稀疏注意力（CSA）

CSA 将 **KV 压缩** 与 **DeepSeek 稀疏注意力（DSA）** 相结合，实现极端长上下文下的高效计算。

#### 1.1 重叠窗口 KV 压缩

对于输入隐状态 $H \in \mathbb{R}^{n \times d}$：

$$
C_a = H \cdot W_a^{KV}, \quad C_b = H \cdot W_b^{KV}
$$

$$
Z_a = H \cdot W_a^{Z}, \quad Z_b = H \cdot W_b^{Z}
$$

采用**重叠窗口**进行压缩（每块使用 $2m$ 个 token）：

$$
[S_a; S_b] = \text{Softmax}_{\text{row}}([Z_a + B_a; Z_b + B_b])
$$

$$
C_i^{\text{Comp}} = \sum_{j=mi}^{m(i+1)-1} S_j^a \odot C_j^a + \sum_{j=m(i-1)}^{mi-1} S_j^b \odot C_j^b
$$

**关键特性**：序列长度被压缩到 $\frac{1}{m}$，但通过重叠窗口，每个压缩 entry 融入了相邻块的信息，缓解了块边界的信息损失。

**数值示例**（$n=16, m=4, c=512$）：

```
Token:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        ├── Comp[0] ──┤  (tokens 0-3 via C_a)
           ├── Comp[1] ──┤  (tokens 4-7 via C_a + tokens 0-3 via C_b, 重叠!)
              ├── Comp[2] ──┤  (tokens 8-11 via C_a + tokens 4-7 via C_b)
                 ├── Comp[3] ──┤  (tokens 12-15 via C_a + tokens 8-11 via C_b)

C_Comp[1] = Σ_{j=4}^{7}  S_a[j]⊙C_a[j] + Σ_{j=0}^{3}  S_b[j]⊙C_b[j]
            ↑ 当前块 (C_a)           ↑ 重叠块 (C_b,与 Comp[0] 共享)
```

重叠压缩使每个压缩 entry 融入了 $2m$ 个原始 token 的信息（$C_a$ 当前块 + $C_b$ 前一块），避免了硬边界的信息丢失。16 个 token 最终压缩为 $\lfloor 16/4 \rfloor = 4$ 个 entry，KV 内存减少 75%。

> 完整压缩算法伪代码与 CSA/HCA 对比表见 `raw/05_model_families/deepseek/DeepSeek_V4_Compressed_KV_Entries.md`。

#### 1.2 Lightning Indexer（DSA 稀疏选择）

压缩后，CSA 使用低秩索引器为每个 query token 选择 top-$k$ 个压缩 KV entry：

$$
c_t^Q = h_t \cdot W^{DQ}, \quad q_t^I = c_t^Q \cdot W^{IUQ}
$$

索引分数计算：

$$
w_t^I = h_t \cdot W_w
$$

$$
I_{t,s} = \sum_{h=1}^{n_h^I} w_{t,h}^I \cdot \text{ReLU}(q_{t,h}^I \cdot K_s^{I\text{Comp}})
$$

Top-$k$ 选择：

$$
C_t^{\text{SprsComp}} = \{C_s^{\text{Comp}} \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}
$$

#### 1.3 共享 KV-MQA 与分组输出投影

被选中的压缩 entry 以 **MQA（Multi-Query Attention）** 方式同时作为 key 和 value。为降低输出投影的计算开销：

- 将 $n_h$ 个输出分成 $g$ 组
- 每组先投影到中间维度 $d_g$（$d_g < c \cdot \frac{n_h}{g}$）
- 最后将 $g$ 组中间输出合并投影到 $d$ 维

### 2. 重度压缩注意力（HCA）

HCA 采用**更激进的压缩策略，但不使用稀疏选择**：

| 特性 | CSA | HCA |
|------|-----|-----|
| 压缩率 | $m$（如 4） | $m'$（如 128） |
| 稀疏选择 | DSA top-k | 无（对所有压缩 KV 做 dense attention） |
| Attention 类型 | 稀疏 | Dense |

压缩公式（无重叠）：

$$
S_{m'i:m'(i+1)-1} = \text{Softmax}_{\text{row}}(Z_{m'i:m'(i+1)-1} + B)
$$

$$
C_i^{\text{Comp}} = \sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j
$$

同样使用共享 KV-MQA 和分组输出投影。

### 3. 混合配置

- **V4-Pro**：前 2 层使用 HCA；后续层 CSA 与 HCA 交错
- **V4-Flash**：前 2 层使用纯 SWA；后续层 CSA 与 HCA 交错

### 4. 辅助机制

#### 滑动窗口注意力（SWA）
- 每个 query token 额外关注最近 $n_{\text{win}}=128$ 个未压缩的 KV entry
- 补偿因果性约束（query 无法访问同一块内的其他 token）

#### 部分旋转位置编码（Partial RoPE）
- 对 query、KV entry 和 attention output 的最后 64 维应用 RoPE
- 对 attention output 额外应用位置 $-i$ 的 RoPE，注入相对位置信息

#### Attention Sink
- 每个头设置可学习的 sink logits $z'_h$，加到 attention 分母上
- 允许每头的 attention 总分偏离 1，甚至可以接近 0
- 缓解极端长上下文下的 attention 分配问题

---

## 与 MLA 和 DSA 的对比

### 与 MLA（Multi-head Latent Attention，V2/V3）对比

| 维度 | MLA（V2/V3） | CSA/HCA（V4） |
|------|-------------|--------------|
| 压缩对象 | 模型维度（$d_c \ll n_h d_h$） | 序列维度（$\frac{n}{m}$ 个 block） |
| KV Cache 增长 | $O(l)$ 线性增长 | $O(\frac{l}{m})$ 亚线性增长 |
| 100 万上下文 KV Cache | 较大（128K 已优化） | 约为 BF16 GQA-8 基线的 2% |
| 稀疏选择 | 无（dense） | CSA: DSA top-k；HCA: 压缩后 dense |
| Query 处理 | 低秩压缩（训练时） | 仅 indexer 使用低秩压缩 |
| 局部性机制 | 无 | SWA（128 token 窗口） |
| 位置编码 | 解耦 RoPE | Partial RoPE + 输出修正 |

**核心范式转移**：从"压缩每 token 的 KV 表示"（MLA）转向"在序列维度合并 token"（CSA/HCA）。在百万 token 级别，序列维度的压缩收益远超模型维度的压缩。

### 与 DSA（DeepSeek Sparse Attention）对比

DSA 是 CSA 的**组成部分**，在 V4 中并非独立的 attention 机制：

| 方面 | DSA（概念） | CSA（V4 实现） |
|------|------------|---------------|
| KV 状态 | 原始未压缩 KV | 压缩后的 KV entry（长度为 $\frac{1}{m}$） |
| 选择空间 | 原始 token/block 粒度 | 压缩后的 block 粒度 |
| 选择机制 | 粗粒度索引 | Lightning Indexer（低秩 score） |
| 复杂度 | $O(n \cdot k)$ | $O(\frac{n}{m} \cdot k)$ |

DSA 提供稀疏选择的理念；CSA 在其下层增加了压缩层。

---

## mHC 集成

V4 引入 [[25_mhc_analysis]]（流形约束超连接）来增强残差连接：

- 扩展因子 $n_{hc} = 4$
- 残差映射矩阵 $B_l$ 通过 Sinkhorn-Knopp 算法（20 次迭代）投影到双随机矩阵
- 动态 + 静态参数化，输入/输出映射通过 Sigmoid 约束为有界非负值

**系统开销**：在 overlapped 1F1B 流水线中，wall-time 开销仅 **6.7%**，通过以下方式实现：
1. TileLang 融合 kernel（映射计算 + Sinkhorn + 残差合并）
2. 选择性重计算（保存层间输入，重算 mHC 但不重算重型层函数）
3. DualPipe 调度调整以适配增加的流水线通信

详见 [[25_mhc_analysis]] 获取完整数学推导和稳定性分析。

---

## Muon 优化器

V4 对大部分模块采用 [[11_muon_analysis]]：

- **AdamW** 保留用于：embedding、prediction head、mHC 静态偏置/门控、RMSNorm 权重
- **Muon** 用于其余所有模块
- 混合 Newton-Schulz：前 8 步使用快速收敛系数 + 后 2 步使用稳定化系数
- 无需 QK-Clip，因为 CSA/HCA 直接对 query 和 KV entry 应用 RMSNorm

详见 [[11_muon_analysis]] 了解分布式训练策略（ZeRO 混合分配、MoE 参数展平、BF16 梯度同步）。

---

## 训练阶段

### 长上下文扩展策略
- 从 4K → 16K → 64K → 100 万 token 渐进扩展
- 前 1T token 使用**标准 dense attention** 热身
- 在 64K 长度分两步引入稀疏 attention：
  1. 先热身 Lightning Indexer
  2. 再启用完整 CSA 稀疏 attention

### 面向 CSA/HCA 的上下文并行（Contextual Parallelism）

由于训练数据使用 **packed sequences**（多条文档拼接），各文档独立压缩（尾部不足 $c$ 个 token 直接丢弃），标准 CP 面临三个矛盾：跨 rank 文档切断、压缩窗口跨 CP 边界、各 rank 压缩输出长度不可预测。

两阶段通信方案解决：

1. **Stage 1（P2P 邻接交换）**：Rank $i$ 将最后 $c$ 个未压缩 KV entry 发送给 rank $i+1$。Rank $i+1$ 将收到的 entry 与本地前 $c$ 个 entry 合并压缩为 $(c+1)$ 个固定长度 compressed block。通信量 $O(c \cdot n_{kv} \cdot d_h)$——与序列长度 $S$ 无关。
2. **Stage 2（All-Gather）**：收集所有本地压缩 KV，通过 fused select-and-pad 算子重组为全局压缩 KV（总长 $P \times c$）。

通信节省：All-gather 传输的是**压缩后 KV**，相比标准 CP（传输未压缩 KV），通信量随压缩率 $c$ 线性缩减。

> [!note] 「CSA ~51×、HCA ~2048×」是 [[23_deepseek_v4_cp_analysis]] 据通信量公式（$\approx 1/(P\cdot c)$，取 $P{=}8$）的**本页推导/估算**，正式版论文未直接给出该数字。

> 完整分析见 [[23_deepseek_v4_cp_analysis]]，涵盖 packed sequences 数据格式、三层可见性控制（sample-level mask → block causal → precomputed rules/Top-K selector）、训练/推理尾部 token 处理差异、通信量公式推导和数值示例。

### 训练稳定性

两种关键技术防止万亿参数 MoE 训练中的 loss spike：

1. **预期路由（Anticipatory Routing）**：用当前参数 $\theta_t$ 计算特征，但使用历史参数 $\theta_{t-\Delta t}$ 计算路由索引。在基础设施层面预计算，wall-time 开销约 20%；仅在 loss spike 时动态触发。

2. **SwiGLU 截断（SwiGLU Clamping）**：线性分量截断到 $[-10, 10]$，gate 分量上限截断到 10。有效消除异常值且不损失性能。

---

## 推理阶段

### 异构 KV Cache 管理

V4 的混合注意力因 cache 策略多样和对齐约束，无法直接使用 PagedAttention。自定义布局：

**经典 KV Cache**
- 存储 CSA/HCA 的压缩 entry
- 每 block 覆盖 $\text{lcm}(m, m')$ 个原始 token
- 同时产出 $k_1 = \frac{\text{lcm}}{m}$ 个 CSA entry 和 $k_2 = \frac{\text{lcm}}{m'}$ 个 HCA entry

**状态缓存（State Cache）**
- 每个 request 预分配固定大小的缓存池
- SWA 段：存储最近 $n_{\text{win}}$ 个 token 的未压缩 KV
- CSA/HCA 段：存储待压缩的尾部状态（累积 token 数 $< m$ 时暂不能压缩）

### 磁盘 KV Cache 存储

消除共享前缀的重复 prefill：

| 组件 | 存储策略 |
|------|---------|
| CSA/HCA 压缩 KV | 完整存储；前缀命中时直接复用 |
| SWA KV | 三种可选策略：完整缓存（零重计算、写入密集）/ 周期性检查点（可调参数 $p$）/ 零缓存（重算最后 $n_{\text{win}} \cdot L$ 个 token） |

### 推理效率

在 100 万 token 上下文下（相对 V3.2）：

| 指标 | V4-Pro | V4-Flash |
|------|--------|----------|
| 推理 FLOPs | 27% | 10% |
| KV Cache 大小 | 10% | 7% |

### DualPath 推理框架

> [!contradiction] **此小节与正式发表版不符（疑为预发布期臆造）。** 正式版 arXiv:2606.19348v1 全文**无** “DualPath”
> （计数 0）、无 SNIC/CNIC 双网卡路由、无 1.87×/1.96× 吞吐数字。论文真正的推理框架是 **§3.5**：KV-Cache 结构与管理
> (§3.5.1) + On-Disk KV 存储 (§3.5.2)（见上文「异构 KV Cache 管理」「磁盘 KV Cache 存储」两节，已与论文一致）。
> 以下原文保留以备查，**请勿引用**。详见 [[30_deepseek_v4_audit_analysis]]。

V4 引入 DualPath 解决大规模推理中的**存储带宽瓶颈**——所有 KV Cache 都通过 Prefill 节点的存储网卡（SNIC）加载，Decode 节点的网卡（CNIC）闲置。

```
传统架构:  存储 → Prefill SNIC → GPU → Decode (Decode SNIC 闲置)
DualPath:  路径A: 存储 → Prefill SNIC → GPU → Decode
           路径B: 存储 → Decode SNIC → RDMA → Prefill GPU (利用闲置带宽)
           动态调度器: 根据实时负载在两路径间负载均衡
```

**性能提升**：
- 离线推理吞吐量提升 **1.87×**
- 在线服务吞吐量提升 **1.96×**

### 推理模式（Think Modes）

V4 支持三级推理强度：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| Non-Think | 快速响应，跳过反思链 | 简单查询、翻译 |
| Think High | 中等推理强度 | 复杂任务、分析 |
| Think Max | 最大推理强度（多轮反思+验证） | Agent 场景、高难度推理 |

---

## MoE 架构

继承自 DeepSeek-V3，有少量调整：

- **激活函数**：Sigmoid → $\sqrt{\text{Softplus}(\cdot)}$
- **负载均衡**：无辅助损失策略 + 序列级平衡损失（权重 0.0001）
- **路由约束**：移除目标节点数量限制；重新设计并行策略
- **哈希路由（Hash Routing）**：前 3 层 MoE 使用基于 token ID 的预定义哈希函数
- V4-Pro：1 个共享专家 + 384 个路由专家（中间维度 3072）
- V4-Flash：1 个共享专家 + 256 个路由专家（中间维度 2048）
- 两者：每 token 激活 6 个专家

---

## 关键基础设施

1. **细粒度 EP 重叠**：基于 wave 的专家调度，一般推理加速 1.50~1.73 倍（RL rollout 等延迟敏感场景高达 1.96 倍）
2. **TileLang 内核**：融合算子替代数百个 ATen 算子；Host Codegen 将 CPU 开销降至 <1μs；Z3 SMT 求解器辅助整数分析
3. **批次不变性与确定性内核**：双内核 attention 策略、DeepGEMM 替代 cuBLAS、MoE 反向传播隔离累加缓冲区
4. **FP4 量化感知训练（QAT）**——⚠️**这是后训练技术**（正式版 §5.2.1「Post-Training Infrastructures」，非预训练/训练框架）：FP4 仅作用于**两个组件**（MoE 专家权重、CSA indexer QK 路径）；**索引分数 I 另走 FP32→BF16**（top-k 选择器 2× 加速 / 99.7% 召回）。FP4→FP8 反量化**无损**（E4M3 比 FP4 多 2 个指数位）。详见 [[24_deepseek_v4_fp4_qat_analysis]]

---

## 评测亮点

### 基座模型（Table 1）

| 基准测试 | V3.2-Base | V4-Flash-Base | V4-Pro-Base |
|---------|-----------|---------------|-------------|
| MMLU | 87.8 | 88.7 | 90.1 |
| MMLU-Pro | 65.5 | 68.3 | 73.5 |
| HumanEval | 62.8 | 69.5 | 76.8 |
| MATH | 60.5 | 57.4 | 64.5 |
| LongBench-V2 | 40.2 | 44.7 | 51.5 |

V4-Flash-Base 尽管激活参数更小（13B vs 37B），仍在大多数基准上超越 V3.2-Base。V4-Pro-Base 实现近乎全面的领先。

### V4-Pro-Max（后训练旗舰，Think Max 模式）

> 说明列已按正式版 **Table 6 (p38)** 校准（原稿「顶尖」措辞被 Table 6 反证，此处改为有基线的相对表述）。
> 列内对手：Opus-4.6 / GPT-5.4 / Gemini-3.1-Pro / K2.6 / GLM-5.1 / DS-V4-Pro-Max。

| 基准测试 | DS-V4-Pro-Max | 相对位置（Table 6） |
|---------|------|------|
| LiveCodeBench (Pass@1) | 93.5 | **所列最高**（Gemini 91.7 / K2.6 89.6 / Opus 88.8） |
| Codeforces (Rating) | 3206 | **最高**（> GPT-5.4 3168 > Gemini 3052；Opus 未报） |
| IMO-AnswerBench (Pass@1) | 89.8 | 开源领先；低于 GPT-5.4 91.4 |
| HMMT 2026 Feb (Pass@1) | 95.2 | 开源领先；低于 GPT-5.4 97.7 / Opus 96.2 |
| SWE Verified (Resolved) | 80.6 | 与 Gemini 并列；略低于 Opus 80.8 |
| MRCR 1M (MMR) | 83.5 | 胜 Gemini-3.1-Pro 76.3；负 Opus-4.6 92.9 |

### 后训练

两阶段流水线：
1. **专家训练（Specialist Training）**：针对数学/代码/智能体/指令等域的 SFT + GRPO RL
2. **在线策略蒸馏（OPD）**：从多个专家模型蒸馏统一模型

**生成式奖励模型（GRM）**：演员网络本身兼任奖励模型，实现生成能力与评估能力的联合优化。

---

## 相关页面

- [[23_deepseek_v4_cp_analysis]] — Context Parallelism 深度分析（packed sequences 适配、通信量、可见性控制）
- [[24_deepseek_v4_fp4_qat_analysis]] — FP4 量化感知训练分析
- [[12_deepseek_v3_analysis]] — 前代模型，MLA、FP8 训练、DualPipe
- [[11_deepseek_v2_analysis]] — MLA 起源、DeepSeekMoE 引入
- [[25_mhc_analysis]] — 流形约束超连接深度解析
- [[11_muon_analysis]] — Muon 优化器原理与分布式实现
- [[14_deepseek_r1_analysis]] — GRPO 与推理流水线
- [[20_deepseek_moe_analysis]] — MoE 路由与负载均衡
- [[28_deepseek_v4_architecture_analysis]] — V4 架构结构图（补充参考）
- [[27_deepseek_v4_implementation_deepdive]] — V4 技术实现伪代码（补充参考）
- [[26_deepseek_v4_technical_deepdive]] — CSA/HCA/DSA/MLA 对比深度解析（补充参考）
- [[attention_is_all_you_need_analysis]] — 原始 Transformer 注意力
- [[dspark_analysis]] — DSpark：挂在 V4 checkpoint 上的投机解码草稿模块（生产相对 MTP-1 提速 60%–85% / 57%–78%）
- [[deepspec_codebase_analysis]] — DSpark 开源训练/评测仓 DeepSpec 源码分析
