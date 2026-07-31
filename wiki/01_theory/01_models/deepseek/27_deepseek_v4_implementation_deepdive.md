# DeepSeek-V4 实现要点：核心组件伪代码 + 数据流

> **核对基线**: arXiv:**2606.19348v1**「DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence」（DeepSeek-AI, **2026-04-26**）＝ `raw/01_theory/01_models/deepseek/DeepSeek_V4.pdf`
> **维度**: 实现级 Deep Dive —— V4 五大核心组件的**源忠实伪代码 + 数据流**，每段代码逐步标注其实现的论文 §/Eq/page。
> **整页重写**: 2026-06-25。旧版为预发布期 AI 臆造的通用伪代码（路由专家写成 128、HCA 写成 `compression_ratio=0.1`、Sinkhorn `max_iter=100`、"Muon" 实为 Adam、量化写成 INT8、臆造 DualPath/PCA-KV/task_classifier 动态-k），已整页废弃；审计见 [[30_deepseek_v4_audit_analysis]]。
>
> **本页定位**: V4 把"百万 token 上下文效率"拆进四类机制——CSA/HCA 沿**序列维**把 KV 压成 $1/m$、$1/m'$（再叠 DSA top-k）；mHC 用**双随机矩阵**约束残差混合保证深堆叠稳定；Muon 用**混合 Newton-Schulz** 正交化更新；DeepSeekMoE 把亲和度激活换成 $\sqrt{\mathrm{Softplus}(\cdot)}$ 并对前 3 层 MoE 用哈希路由。本页给每个机制的逐步伪代码并锚定到论文方程，是 [[13_deepseek_v4_analysis]] 的**实现级补充**；架构综述见 [[13_deepseek_v4_analysis]]，对比与图示见 [[26_deepseek_v4_technical_deepdive]] / [[28_deepseek_v4_architecture_analysis]]，mHC/Muon 数学细节见 [[25_mhc_analysis]] / [[muon_analysis]]。

> [!note] 伪代码的可信度约定
> 下列伪代码是对论文方程的**忠实重构**（reconstruction）：控制流/张量重排为**演示性骨架**，但每一行的算子、张量形状、常量都映射到论文已核验的 §/Eq/page，并在行内注释标出。常量取自 **§4.2.1 Model Setups（p24–25）**，凡两规格不同处给出 `Flash | Pro` 双值。论文同时开源了推理实现（HuggingFace `DeepSeek-V4-Pro/inference`，§2.3 脚注），用于消歧极细节——本页对这些「未在正文给出公式」之处会显式标注 *(实现细节/演示)*。

---

## 1. 概述

### 1.1 组件 → 论文位置 → 关键方程

| 组件 | 论文位置 | 关键方程 | 本页节 |
|---|---|---|---|
| CSA 重叠窗口压缩器 | §2.3.1, Figure 3 (p9–10) | Eq 9–12 | §2.1 |
| CSA Lightning Indexer + Top-k 选择 | §2.3.1 (p10) | Eq 13–17 | §2.2 |
| CSA 共享 KV-MQA | §2.3.1 (p11) | Eq 18–19 | §2.3 |
| 分组输出投影 | §2.3.1 (p11) | 文字描述（无独立编号） | §2.3 |
| HCA 重度压缩 + MQA | §2.3.2, Figure 4 (p11–12) | Eq 20–26 | §3 |
| RMSNorm / Partial RoPE / SWA / Attention Sink | §2.3.3 (p12–13) | Eq 27 | §4 |
| mHC 流形约束残差混合 | §2.2 (p7–8) | Eq 1–8 | §5 |
| Muon 优化器 + 混合 Newton-Schulz | §2.4, Algorithm 1 (p14) | Eq 28 | §6 |
| DeepSeekMoE 路由 | §2.1 (p7) | 文字描述 | §7 |
| 推理异构 KV Cache | §3.5.1 / §3.5.2, Figure 6 (p21–23) | — | §8 |

### 1.2 关键超参（§4.2.1, p24–25，逐项已核验）

| 超参 | V4-Flash | V4-Pro |
|---|---|---|
| Transformer 层数 $L$ | 43 | 61 |
| 隐藏维 $d$ | 4096 | 7168 |
| 前 2 层注意力 | 纯 SWA | HCA |
| CSA 压缩率 $m$ | 4 | 4 |
| HCA 压缩率 $m'$ | 128 | 128 |
| Indexer query 头数 $n_h^I$ | 64 | 64 |
| Indexer 头维 $c^I$ | 128 | 128 |
| 注意力 top-k（DSA 选中的压缩 KV 数） | 512 | 1024 |
| Query 头数 $n_h$ | 64 | 128 |
| 头维 $c$ | 512 | 512 |
| Query 压缩维 $d_c$ | 1024 | 1536 |
| 输出投影分组 $g$ | 8 | 16 |
| 每组中间输出维 $d_g$ | 1024 | 1024 |
| SWA 窗口 $n_\text{win}$ | 128 | 128 |
| 共享 / 路由专家 | 1 / 256 | 1 / 384 |
| 专家中间维 | 2048 | 3072 |
| 每 token 激活专家 | 6 | 6 |
| 哈希路由层 | 前 3 个 MoE 层 | 前 3 个 MoE 层 |
| MTP depth | 1 | 1 |
| mHC 扩展因子 $n_{hc}$ | 4 | 4 |
| Sinkhorn-Knopp 迭代 $t_\text{max}$ | 20 | 20 |
| 词表 | 128K | 128K |
| 总参 / 激活参 | 284B / 13B | 1.6T / 49B |

> 整体计算图（Figure 2, p6）：`Input → Embedding → [Transformer Block ×L：mHC(Residual/Pre/Post Mixing) 包裹 (CSA 或 HCA) + DeepSeekMoE] → Prediction Head + MTP(depth 1) → LM Loss + MTP Loss`。Transformer 主干与 MTP 沿用 V3；新增 mHC、混合 CSA/HCA、Muon。

---

## 2. CSA —— 压缩稀疏注意力（§2.3.1, Figure 3, p9–11）

CSA 的口径（§2.3, p9 原文）："先把每 $m$ 个 token 的 KV 压成一个 entry，再对压缩后的 entry 施加 DeepSeek Sparse Attention（DSA），每个 query 只关注 $k$ 个压缩 KV entry"。三步：① 重叠窗口压缩 → ② Lightning Indexer top-k 选择 → ③ 共享 KV-MQA + 分组输出投影。

### 2.1 重叠窗口压缩器（Eq 9–12）

```python
# CSA 压缩器：H ∈ R^{n×d} → C_Comp ∈ R^{(n/m)×c}
# 常量: m=4, c=512, d=4096|7168   (§4.2.1, p24-25)
def csa_compress(H, W_a_KV, W_b_KV, W_a_Z, W_b_Z, B_a, B_b, m):
    # Eq 9: 两路 KV entry。W_*^{KV} ∈ R^{d×c}
    C_a = H @ W_a_KV                       # R^{n×c}
    C_b = H @ W_b_KV                       # R^{n×c}
    # Eq 10: 两路压缩权重 logits。W_*^{Z} ∈ R^{d×c}
    Z_a = H @ W_a_Z                        # R^{n×c}
    Z_b = H @ W_b_Z                        # R^{n×c}

    n = H.shape[0]
    C_Comp = []                            # 目标长度 n/m
    for i in range(n // m):
        # 当前块 [mi, m(i+1)-1] 走 C_a；前一块 [m(i-1), mi-1] 走 C_b（重叠！）
        za = Z_a[m*i : m*(i+1)] + B_a      # B_a,B_b ∈ R^{m×c} 可学习位置偏置
        if i == 0:
            # i=0: C_b 的 Z 用 -inf 填充、C_b 用 0 填充 (Eq 11/12 下方说明, p10)
            zb = full((m, c), -inf)
            Cb_blk = zeros((m, c))
        else:
            zb = Z_b[m*(i-1) : m*i] + B_b
            Cb_blk = C_b[m*(i-1) : m*i]
        # Eq 11: Softmax_row 在 [za; zb] 上对 2m 个元素整体归一化
        S = softmax_row(concat([za, zb], axis=0))     # R^{2m×c}
        Sa, Sb = S[:m], S[m:]
        # Eq 12: 当前块(C_a) + 重叠前块(C_b) 的加权和，⊙ 为 Hadamard 积
        Ci = (Sa * C_a[m*i:m*(i+1)]).sum(0) + (Sb * Cb_blk).sum(0)   # R^c
        C_Comp.append(Ci)
    return stack(C_Comp)                   # R^{(n/m)×c}
```

**数据流 / 为什么是 $1/m$**：每个 $C_i^\text{Comp}$ 由 $2m$ 个 KV entry 派生，但 $C_i^\text{Comp}$ 用到的 $C^b$ 索引段与 $C_{i-1}^\text{Comp}$ 用到的 $C^a$ 索引段**重叠**——因此净压缩率恰为 $1/m$（§2.3.1 原文，p10）。重叠让每个压缩 entry 跨越 $2m$ 个原始 token，缓解硬块边界的信息损失。Softmax 在 $2m$ 个元素上整体归一化（不是分别归一化两块）。

> 同一压缩算子还会被复用一次，得到压缩后的 **indexer keys** $K^{IComp}\in\mathbb R^{(n/m)\times c^I}$（§2.3.1, p10；$c^I=128$）。

### 2.2 Lightning Indexer + Top-k 选择（Eq 13–17）

```python
# 对 query token t，在 n/m 个压缩块上选 top-k 个送入 core attention
# 常量: d_c=1024|1536, n_h_I=64, c_I=128, top_k=512|1024  (§4.2.1)
def lightning_indexer(h_t, K_IComp, W_DQ, W_IUQ, W_w, n_h_I, top_k):
    # Eq 13: 低秩 query 隐向量。W_DQ ∈ R^{d×d_c}
    c_Q_t = h_t @ W_DQ                     # R^{d_c}  —— 与下游 MQA 共享 (Eq 18)
    # Eq 14: 升投影到 indexer 多头 query。W_IUQ ∈ R^{d_c × c_I·n_h_I}
    q_I_t = (c_Q_t @ W_IUQ).reshape(n_h_I, c_I)        # {q_I[t,1..n_h_I]}
    # Eq 15: 每个 indexer 头的权重。W_w ∈ R^{d×n_h_I}
    w_I_t = h_t @ W_w                      # R^{n_h_I}

    scores = []
    for s in range(K_IComp.shape[0]):      # s 为压缩块索引
        if s >= floor(t / m):              # Eq 16 因果条件: s < floor(t/m)
            scores.append(-inf); continue
        # Eq 16: 索引分数 = Σ_h w_I[t,h] · ReLU(q_I[t,h] · K_IComp[s])
        I_ts = sum(w_I_t[h] * relu(dot(q_I_t[h], K_IComp[s])) for h in range(n_h_I))
        scores.append(I_ts)
    # Eq 17: 保留 top-k 个压缩 KV entry 作为 C^SprsComp_t
    topk_idx = argtopk(scores, top_k)
    return topk_idx                        # 选中的压缩块下标集合
```

**数据流**：indexer 用与主压缩相同的算子压 keys，使 top-k **在 $n/m$ 个块上选**（而非 V3.2 的逐 token 选）——这是 V4 "压缩 → 稀疏" 叠加的关键。分数用 `ReLU` + 多头加权和（非 softmax）。$c_t^Q$ 同时供 indexer 与下游 MQA query，避免重复的 query 投影。

### 2.3 共享 KV-MQA + 分组输出投影（Eq 18–19 + p11 文字）

```python
# 常量: n_h=64|128, c=512, g=8|16, d_g=1024  (§4.2.1)
def csa_mqa_and_output(c_Q_t, sprs_comp_kv, W_UQ, W_O_group, W_O_final, n_h, g):
    # Eq 18: 由共享隐向量升投影出 n_h 个 query 头。W_UQ ∈ R^{d_c × c·n_h}
    q_t = (c_Q_t @ W_UQ).reshape(n_h, c)              # {q[t,1..n_h]}
    # Eq 19: MQA —— 每个压缩 entry 同时作 key 和 value（共享 KV）
    o = []
    for i in range(n_h):
        o.append(core_attn(query=q_t[i], key=sprs_comp_kv, value=sprs_comp_kv))  # o[t,i] ∈ R^c
    o = stack(o)                                       # R^{n_h×c}, 即 o_t ∈ R^{c·n_h}

    # 分组输出投影 (p11): c·n_h 很大，直接投到 d 代价高 → 先分 g 组降维
    groups = o.reshape(g, (n_h // g) * c)              # 每组 o^G ∈ R^{c·n_h/g}
    inter = [grp @ W_O_group[j] for j, grp in enumerate(groups)]   # → d_g, 且 d_g < c·n_h/g
    o_hat = concat(inter) @ W_O_final                  # R^{g·d_g} → R^d, 最终 ô_t ∈ R^d
    return o_hat
```

**为什么分组**：$c\cdot n_h$（如 Pro 的 $512\times128=65536$）远大于 $d$，单步投影代价巨大；先按 $g$ 组各降到 $d_g=1024$（$d_g<c\,n_h/g$），再拼接投到 $d$，显著降本（§2.3.1, p11，无独立方程编号）。

---

## 3. HCA —— 重度压缩注意力（§2.3.2, Figure 4, p11–12）

HCA 口径（§2.3.2 原文）："以更重的方式压缩 KV，但**不使用稀疏注意力**"。与 CSA 三点不同：① 压缩率 $m'\gg m$（$m'=128$）；② **无重叠窗口**；③ **无 indexer / 无 top-k**——每个 query 关注它之前的**所有**压缩块。

```python
# 常量: m_prime=128, c=512, n_h=64|128, d_c=1024|1536  (§4.2.1)
def hca_attention(H, W_KV, W_Z, B, W_DQ, W_UQ, m_prime, n_h):
    # Eq 20-21: 单路 KV entry 与压缩 logits。W_KV, W_Z ∈ R^{d×c}
    C = H @ W_KV                           # R^{n×c}
    Z = H @ W_Z                            # R^{n×c}

    n = H.shape[0]
    C_Comp = []
    for i in range(n // m_prime):          # 无重叠：每 m' 个 token → 1 entry
        # Eq 22: Softmax_row 仅在本块 m' 个元素上归一化。B ∈ R^{m'×c}
        S = softmax_row(Z[m_prime*i : m_prime*(i+1)] + B)     # R^{m'×c}
        # Eq 23: 块内加权和 → 1/m' 压缩
        Ci = (S * C[m_prime*i : m_prime*(i+1)]).sum(0)        # R^c
        C_Comp.append(Ci)
    C_Comp = stack(C_Comp)                 # R^{(n/m')×c}

    # Eq 24-25: 与 CSA 相同的低秩 query。W_DQ ∈ R^{d×d_c}, W_UQ ∈ R^{d_c × c·n_h}
    out = []
    for t in range(n):
        c_Q_t = H[t] @ W_DQ
        q_t = (c_Q_t @ W_UQ).reshape(n_h, c)
        # Eq 26: 对所有合法压缩块做 MQA（无 top-k 筛选）
        visible = C_Comp[: floor(t / m_prime)]               # 因果可见范围
        o = [core_attn(q_t[i], key=visible, value=visible) for i in range(n_h)]
        out.append(grouped_output_projection(stack(o)))      # 同 §2.3 (CSA) 分组投影
    return stack(out)
```

**数据流**：HCA 把 $128$ 个 token 压成 1 个 entry，省到极致，但因不可稀疏跳过，靠"对所有压缩块 dense"维持全局可见。Pro 前 2 层用 HCA、Flash 前 2 层用纯 SWA（§4.2.1，非对称设计）；其余层 CSA 与 HCA **交错**（§2.3, p9）。

---

## 4. CSA / HCA 共用的辅助技术（§2.3.3, p12–13；§2.3.4, p13）

```python
# 这些技术在 §2.1-2.3 主线中被略去，§2.3.3 补充（论文称细节以开源实现为准）

# (a) Query/KV RMSNorm —— core attention 之前，对每个 query 头与唯一的压缩 KV 头各做一次
def pre_attn_norm(q_head, kv_head):
    return rmsnorm(q_head), rmsnorm(kv_head)   # 防 attention logit 爆炸 / 稳训练 (§2.3.3, p12)

# (b) Partial RoPE —— 仅作用于 query 与 KV entry 的最后 64 维 (§2.3.3, p13)
def partial_rope(vec, pos):
    vec[-64:] = rope(vec[-64:], pos)           # 64 为论文给定值
    return vec
def output_relative_rope(o_ti, i):
    # KV 同时作 key 与 value → 朴素输出携带绝对位置；对 o[t,i] 最后 64 维施加位置 -i 的 RoPE
    o_ti[-64:] = rope(o_ti[-64:], pos=-i)      # 使输出携带"相对"位置 (§2.3.3, p13)
    return o_ti

# (c) SWA 辅助分支 —— 每个 query 额外关注最近 n_win 个未压缩 KV entry
def swa_branch(query, recent_kv, n_win=128):   # 补偿: query 看不到自身压缩块内的 token
    return core_attn(query, key=recent_kv[-n_win:], value=recent_kv[-n_win:])

# (d) Attention Sink (Eq 27) —— 可学习 sink logit z'_h 加进分母，使每头注意力总和可 ≠ 1
def attn_with_sink(logits_h, z_sink_h):        # logits_h: 第 h 头对各 key/块的 logit
    denom = exp(logits_h).sum() + exp(z_sink_h)
    return exp(logits_h) / denom               # s_{h,i,j}; 允许总注意力近 0
```

**效率（§2.3.4, p13，paper-stated）**：KV 混合存储——RoPE 维用 **BF16**、其余维用 **FP8**，相比纯 BF16 省约 **½**；Lightning Indexer 的注意力以 **FP4** 计算；top-k 比 V3.2 更小。综合下 1M 上下文的 KV cache 约为 BF16 GQA8（头维 128）基线的 **~2%**。

---

## 5. mHC —— 流形约束超连接（§2.2, Eq 1–8, p7–8）

把残差流从 $\mathbb R^{d}$ 扩展为 $\mathbb R^{n_{hc}\times d}$（$n_{hc}=4$）。核心创新：把残差混合矩阵 $B_l$ 约束到**双随机矩阵流形（Birkhoff polytope）**，使谱范数 $\lVert B_l\rVert_2\le 1$（非扩张），且该集合对乘法封闭 → 深堆叠稳定（§2.2 原文，p8）。详尽数学与稳定性分析见 [[25_mhc_analysis]]。

```python
# 残差状态 X_l ∈ R^{n_hc×d}, n_hc=4 ; F_l 为第 l 层 (CSA/HCA 或 MoE), 输入输出均 R^d
# 注意: B_l 是 n_hc×n_hc（4×4），不是 hidden×hidden —— 旧版臆造点
def mhc_block(X_l, F_l, W_pre, W_res, W_post, S_pre, S_res, S_post,
              a_pre, a_res, a_post, t_max=20):
    n_hc, d = X_l.shape
    # 动态参数化 (Eq 3-5): 先 flatten+RMSNorm
    X_hat = rmsnorm(X_l.reshape(-1))                 # vec(X_l) ∈ R^{n_hc·d}, 然后 R^{1×n_hc·d}
    # W_pre,W_post ∈ R^{n_hc·d × n_hc}; W_res ∈ R^{n_hc·d × n_hc^2}; a_* 为可学习门控(初始很小)
    A_tilde = a_pre  * (X_hat @ W_pre)        + S_pre              # Eq 3, R^{1×n_hc}
    B_tilde = a_res  * mat(X_hat @ W_res)     + S_res              # Eq 4, R^{n_hc×n_hc}
    C_tilde = a_post * (X_hat @ W_post).T     + S_post             # Eq 5, R^{n_hc×1}

    # 约束 (Eq 6-8)
    A_l = sigmoid(A_tilde)                           # Eq 6: 非负有界
    C_l = 2 * sigmoid(C_tilde)                       # Eq 7: 非负有界 (上界 2)
    B_l = sinkhorn_knopp(B_tilde, t_max)             # Eq 8 → 双随机矩阵

    # Eq 1: 残差更新。A_l X_l ∈ R^d 即真正喂给层的输入
    X_next = B_l @ X_l + C_l @ F_l(A_l @ X_l)         # R^{n_hc×d}
    return X_next

def sinkhorn_knopp(B_tilde, t_max):
    M = exp(B_tilde)                                 # Eq 8: M^(0)=exp(B̃_l) 保正
    for _ in range(t_max):                           # t_max=20 (§4.2.1)
        M = row_norm(col_norm(M))                    # M^(t)=T_r(T_c(M^(t-1)))
    return M                                          # ≈ B_l ∈ Birkhoff polytope (Eq 2)
```

**为什么双随机 + Sinkhorn**：朴素 HC（Zhu et al. 2025）在多层堆叠时频繁数值不稳；约束 $B_l$ 到双随机矩阵保证 $\lVert B_l\rVert_2\le1$（前向与反向都非扩张），$A_l,C_l$ 经 Sigmoid 有界非负避免信号相消（§2.2, p8）。`t_max=20` 是论文给定的实用值（**非旧版臆造的 100**）。

**对照 Figure 2（p6）的三个 Mixing 框**：$A_l$（$1\times n_{hc}$）= **Pre-Block Mixing**——把 $n_{hc}$ 路残差混成喂给 CSA/HCA/MoE 的单路输入 $A_l X_l\in\mathbb R^d$；$C_l$（$n_{hc}\times1$）= **Post-Block Mixing**——把层输出广播回 $n_{hc}$ 路；$B_l$（$n_{hc}\times n_{hc}$）= **Residual Mixing**——残差流自身的跨路混合（即被约束为双随机的那一项）。三者都按 Eq 3–5 动态生成（输入相关分量 $\alpha\cdot\hat X_l W$ + 静态偏置 $S$），其中**静态偏置与门控 $\alpha$ 由 AdamW 更新、其余由 Muon 更新**（§2.4, p14）。

---

## 6. Muon 优化器（§2.4, Algorithm 1, Eq 28, p14）

```python
# 参数分配 (§2.4, p14):
#   AdamW: embedding、prediction head、mHC 静态偏置 & 门控因子、所有 RMSNorm 权重
#   Muon : 其余全部模块
# Muon 超参 (§4.2.2): momentum μ=0.95, weight_decay λ=0.1, update-RMS rescale γ=0.18
def muon_step(W, M_prev, grad, lr_eta, mu=0.95, lam=0.1, gamma=0.18):
    n, m = W.shape
    G_t = grad                                  # Alg1-3: G_t = ∇_W L
    M_t = mu * M_prev + G_t                      # Alg1-4: 动量累积
    # Alg1-5: Nesterov(μM_t+G_t) 后做混合 Newton-Schulz 正交化
    O_prime = hybrid_newton_schulz(mu * M_t + G_t)
    O_t = O_prime * sqrt(max(n, m)) * gamma      # Alg1-6: 重标定 update 的 RMS（复用 AdamW LR）
    W_new = W * (1 - lr_eta * lam) - lr_eta * O_t  # Alg1-7: 权重衰减 + 更新
    return W_new, M_t

def hybrid_newton_schulz(X):
    # Eq 28: 把 X 近似正交化为 U V^T（SVD: X=UΣV^T）
    M = X / frobenius_norm(X)                    # 先归一化，保证最大奇异值 ≤ 1
    # 10 次迭代、2 阶段
    for k in range(10):
        if k < 8:
            a, b, c = 3.4445, -4.7750, 2.0315    # 前 8 步: 快速把奇异值驱近 1
        else:
            a, b, c = 2.0, -1.5, 0.5             # 后 2 步: 把奇异值稳定精确在 1
        MMT = M @ M.T
        M = a * M + b * (MMT @ M) + c * (MMT @ MMT @ M)   # Eq 28
    return M
```

**要点**：① `HybridNewtonSchulz` 是**真实的多项式迭代正交化**（旧版把 Muon 写成带 bias-correction 的 Adam，完全错误）；② 沿用 Liu et al. (2025) 的 Nesterov + update-RMS 重标定以复用 AdamW 学习率，但**改用 hybrid（两段系数）NS 迭代**；③ **不使用 QK-Clip**——因为 CSA/HCA 已直接对 Q/KV 做 RMSNorm，足以防 logit 爆炸（§2.4, p14）。AdamW 超参（§4.2.2）：$\beta_1=0.9,\beta_2=0.95,\varepsilon=10^{-20},\text{wd}=0.1$。更多分布式实现见 [[muon_analysis]]。

---

## 7. DeepSeekMoE 路由（§2.1, p7）

沿用 DeepSeekMoE 范式（细粒度路由专家 + 共享专家），相对 V3 仅小改：

```python
# 常量: 1 shared + 256|384 routed, 6 activated, 专家中间维 2048|3072 (§4.2.1)
def deepseek_moe_route(h_t, W_gate, n_routed, n_activated=6):
    # §2.1: 亲和度激活由 Sigmoid 改为 Sqrt(Softplus(·))  —— 旧版写 Softmax 错误
    logits = h_t @ W_gate                        # R^{n_routed}
    affinity = sqrt(softplus(logits))            # softplus(x)=log(1+e^x)
    # aux-loss-free 负载均衡: 给亲和度加一组动态偏置 b（不进入加权）后再选 top-k
    biased = affinity + balance_bias             # bias 更新速度 0.001 (§4.2.2)
    idx = argtopk(biased, n_activated)           # 每 token 激活 6 个路由专家
    weight = affinity[idx]                       # 加权用未加偏置的亲和度
    y = shared_expert(h_t) + sum(weight[j] * routed_expert[idx[j]](h_t) for j in range(n_activated))
    return y

def hash_route(token_id, n_routed):
    # §2.1: 前 3 个 MoE 层用哈希路由替代 dense FFN —— 按 token-ID 的预定义哈希定专家
    return predefined_hash(token_id) % n_routed
```

**改动清单（§2.1, p7；§4.2.2）**：① 亲和度 $\text{Sigmoid}\to\sqrt{\text{Softplus}(\cdot)}$；② aux-loss-free + **序列级**平衡损失（权重 $10^{-4}$，防单序列内极端不均）；③ **前 3 个 MoE 层用哈希路由**（token-ID 哈希）替代 V3 的 dense FFN；④ 移除"路由目标节点数"约束并重设并行策略；⑤ MTP depth 仍为 1。路由原理见 [[20_deepseek_moe_analysis]]。

---

## 8. 推理：异构 KV Cache（§3.5.1 / §3.5.2, Figure 6, p21–23）—— 简述

混合注意力产生多种 KV，破坏 PagedAttention 假设（cache 策略多样 + kernel 对齐约束），故 V4 自定义两段式布局。深度分析见 [[23_deepseek_v4_cp_analysis]]。

```python
# (1) Classical KV Cache —— 存 CSA/HCA 压缩 entry (§3.5.1, p22)
block = lcm(m, m_prime)                 # 每 block 覆盖 lcm(m,m')=lcm(4,128)=128 个原始 token
k1 = block // m                         # 该 block 产出的 CSA 压缩 entry 数
k2 = block // m_prime                   # 该 block 产出的 HCA 压缩 entry 数
# kernel 协同设计: 每 block 原始 token 数可取 lcm(m,m') 的任意倍数 (§3.5.1, p23)

# (2) State Cache —— SWA 最近 n_win + 未达压缩长度的尾部，按 SSM 处理 (§3.5.1, p22-23)
#     每 request 预分配固定大小池: SWA 段(最近 n_win token) + CSA/HCA 段(待压缩尾状态)

# (3) On-Disk KV (§3.5.2, p23) —— 消除共享前缀重复 prefill
#     CSA/HCA: 存全部压缩块; 命中前缀复用到"最后一个完整压缩块", 尾部不完整块需重算
#     SWA (体积约 8× 于压缩块): 三策略
#       - Full SWA Caching      : 全存, 零重算 (写入密集, SSD 不友好)
#       - Periodic Checkpointing: 每 p token 存一次最近 n_win, 命中后重算尾部 (p 可调)
#       - Zero SWA Caching       : 不存, 命中需重算最后 n_win·L 个 token (L 层模型)
```

> 说明：以上 `lcm`、`k1/k2`、三种 SWA 策略均为 §3.5 paper-stated（图 6 标注 $k_1=\mathrm{lcm}(m,m')/m$、$k_2=\mathrm{lcm}(m,m')/m'$）。

---

## Related / Cross-references

- [[13_deepseek_v4_analysis]] —— V4 整体架构、规格、评测综述（本页的上层）
- [[26_deepseek_v4_technical_deepdive]] —— CSA/HCA/DSA/MLA 机制对比深解析
- [[28_deepseek_v4_architecture_analysis]] —— V4 结构图与时间线图示
- [[23_deepseek_v4_cp_analysis]] —— Contextual Parallelism + 推理异构 KV 的深度分析（§8 的展开）
- [[24_deepseek_v4_fp4_qat_analysis]] —— FP4/MXFP4 量化感知训练（§5.2.1，后训练技术）
- [[30_deepseek_v4_audit_analysis]] —— 本页旧伪代码的逐项审计与订正依据
- [[25_mhc_analysis]] —— 流形约束超连接的完整数学与稳定性分析（§5 的展开）
- [[muon_analysis]] —— Muon 原理与分布式实现（§6 的展开）
- [[20_deepseek_moe_analysis]] —— DeepSeekMoE 路由与负载均衡（§7 的展开）
- [[12_deepseek_v3_analysis]] —— 前代架构（MLA、FP8 训练、DualPipe），V4 的对照基线
