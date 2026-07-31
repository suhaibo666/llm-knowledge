# DeepSeek-V4 模型结构图（源忠实重画）

> **基线**：arXiv:2606.19348v1 "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"，DeepSeek-AI，2026-04-26。
> **维度**：Overview / 结构图。本页对照正式版 Figure 2（p6）/ Figure 3（p9–11）/ Figure 4（p11–12）/ §4.2.1（p24–25）逐图核对、整页重画（2026-06-25），替换预发布期 AI 臆造的旧图。
> **主旨**：V4 在 V3 的 Transformer + MTP 骨架上做三处结构升级——用 **mHC**（流形约束超连接）取代朴素残差、用 **CSA/HCA 逐层交错的混合注意力**取代 MLA、用 **Muon** 取代 AdamW（多数参数）；MoE 仍是 DeepSeekMoE（仅微调）。配套深读见 [[13_deepseek_v4_analysis]]。
>
> 本页所有维度/计数均来自 GROUND_TRUTH（§B）与已开页面的 §/Figure；图内不放 `[[wiki link]]`（会泄漏成字面方括号），链接一律放正文。

---

## 1. 概述

DeepSeek-V4 是 DeepSeek-V3 的**效率重构**：在 1M token 上下文下，V4-Pro 仅用 V3.2 的 **27% 单 token FLOPs（FP8 等效）与 10% KV cache**，V4-Flash 进一步降到 **10% FLOPs 与 7% KV cache**（§1 / Figure 1，p5）。这一效率来自三个互相独立的结构改动（§2，p6–7）：

1. **mHC**（§2.2）——把残差流从 `R^d` 扩展到 `R^{n_hc×d}`（`n_hc=4`），并把残差混合矩阵 `B_l` 约束到双随机矩阵流形（Birkhoff polytope），用 **Sinkhorn-Knopp（t_max=20）**投影，保证 `‖B_l‖₂≤1` 非膨胀、深堆叠稳定。
2. **混合注意力**（§2.3）—— **CSA**（压缩 + DSA 稀疏 top-k）与 **HCA**（更狠的压缩、无稀疏）**逐层交错**，取代 V3 系的 MLA；首 2 层非对称（Flash=纯 SWA，Pro=HCA）。
3. **Muon 优化器**（§2.4）——多数参数用 Muon，仅 embedding / 预测头 / mHC 静态偏置与门控 / 所有 RMSNorm 权重保留 AdamW。

MoE 沿用 DeepSeekMoE，affinity 激活由 `Sigmoid` 改为 `Sqrt(Softplus(·))`，前 3 个 MoE 层改用 Hash 路由（§2.1）。MTP 配置与 V3 完全一致（深度 1）。

V4 是**两个不同规模的模型**（不是一个模型按任务切换激活）：Flash 284B/13B（激活），Pro 1.6T/49B（激活）。完整配置见下表（§4.2.1，p24–25，全部值取自 GROUND_TRUTH §B）。

| 参数 | 符号 | V4-Flash | V4-Pro |
|---|---|---|---|
| Transformer 层数 | L | **43** | **61** |
| 隐藏维度 | d | 4096 | 7168 |
| 首 2 层注意力 | — | **纯滑窗注意力 SWA** | **HCA** |
| 后续层注意力 | — | CSA / HCA 交错 | CSA / HCA 交错 |
| CSA 压缩率 | m | 4 | 4 |
| indexer query 头数 | n_h^I | 64 | 64 |
| indexer 头维度 | c^I | 128 | 128 |
| 注意力 top-k（DSA 选中 KV 条目） | k | **512** | **1024** |
| HCA 压缩率 | m′ | 128 | 128 |
| query 头数 | n_h | 64 | 128 |
| 头维度 | c | 512 | 512 |
| query 压缩维度 | d_c | 1024 | 1536 |
| 输出投影分组数 | g | 8 | 16 |
| 每组中间输出维度 | d_g | 1024 | 1024 |
| SWA 窗口 | n_win | 128 | 128 |
| 共享专家 | — | 1 | 1 |
| 路由专家 | N | **256** | **384** |
| 专家中间维度 | — | 2048 | 3072 |
| 每 token 激活专家 | K | **6** | **6** |
| Hash 路由 | — | 前 3 个 MoE 层 | 前 3 个 MoE 层 |
| MTP 深度 | — | 1 | 1 |
| mHC 扩展因子 | n_hc | 4 | 4 |
| Sinkhorn-Knopp 迭代 | t_max | 20 | 20 |
| 词表 | — | 128K | 128K |
| 总参数 / 激活参数 | — | **284B / 13B** | **1.6T / 49B** |
| 预训练 token | — | 32T | 33T |

---

## 2. 整体层栈（Figure 2，p6）

Figure 2 给出端到端结构：`Input Tokens → Embedding → [Transformer Block ×L] → Prediction Head + MTP Modules → LM Loss + MTP Loss`。每个 Transformer Block 含**两个 mHC 包裹的子层**：注意力子层（**CSA 或 HCA**）与 **DeepSeekMoE** FFN 子层；每个子层都被 mHC 的三个混合算子包裹——**Pre-Block Mixing**（输入映射 A_l）、子层本体、**Post-Block Mixing**（输出映射 C_l），外加作用在扩展残差流上的 **Residual Mixing**（B_l）。

```
                          Input Tokens
                               │
                          Embedding
                               │
        ┌──────────────────────┴──────────────────────┐
        │  扩展残差流  X ∈ R^{n_hc×d}   (n_hc = 4)      │
        └──────────────────────┬──────────────────────┘
                               │
   ┌───────────────────────────▼───────────────────────────┐
   │  Transformer Block ×L                                  │
   │                                                        │
   │   ── 注意力子层（CSA 或 HCA，逐层交错）──              │
   │      Residual Mixing (B_l) ─────────────┐              │
   │      Pre-Block Mixing (A_l) ──► [ CSA / HCA ] ──►      │
   │                              Post-Block Mixing (C_l)   │
   │                                          │ (+残差)      │
   │   ── FFN 子层（DeepSeekMoE）──            ▼             │
   │      Residual Mixing (B_l) ─────────────┐              │
   │      Pre-Block Mixing (A_l) ──► [ DeepSeekMoE ] ──►    │
   │                              Post-Block Mixing (C_l)   │
   └───────────────────────────┬───────────────────────────┘
                               │  (堆叠 L 层)
                               ▼
                        Prediction Head ──────────► LM Loss
                               │
                        MTP Modules (depth = 1) ──► MTP Loss

  逐层注意力分配（§4.2.1）：
    Flash (L=43)：层 1–2 = 纯 SWA   │ 层 3..43 = CSA / HCA 交错
    Pro   (L=61)：层 1–2 = HCA      │ 层 3..61 = CSA / HCA 交错
```

*图注*：复刻 Figure 2（p6）骨架 + §4.2.1 的首 2 层非对称。三处 Mixing（Residual / Pre-Block / Post-Block）即 mHC 的 `B_l / A_l / C_l`（§2.2，Eq 1：`X_{l+1}=B_l X_l + C_l F_l(A_l X_l)`，p7）。承载维度：`n_hc=4`；Flash `L=43`、Pro `L=61`；MTP 深度 `1`。**MTP Modules 与 Prediction Head 并列**输出，分别产生 MTP Loss 与 LM Loss（旧图遗漏了 MTP）。注意力按层在 CSA / HCA 间交错，论文未给出确切的奇偶排布，此处不臆造具体序列。

---

## 3. CSA 内部结构（Figure 3，§2.3.1，p9–11）

CSA = **逐 m token 压缩 KV**（1/m）+ **DSA lightning indexer top-k 选择** + **shared-KV MQA** + **分组输出投影**，并叠加一条 **SWA 分支**做局部细粒度补偿。query 的低秩 latent `c^Q_t` 在 indexer 与 MQA 之间**共享**。

```
   Hidden States of KV Tokens                 Hidden State of Query Token
            │                                          │
   ┌────────┴────────┐                        c^Q_t = h_t · W^DQ  (Eq 13)
   │ Token-Level     │                          │            │
   │ Compressor      │── Compressed KV Entries  │            │
   │ (每 m token→1,  │      (C^Comp, 长度 n/m)  │            │
   │  重叠窗口, 1/m) │           │              │            │
   └────────┬────────┘           │     Indexer Queries   Queries
            │                     │     q^I_t = c^Q·W^IUQ  q_t = c^Q·W^UQ
   ┌────────┴────────┐           │     (Eq 14)           (Eq 18)
   │ Token-Level     │           │            │            │
   │ Compressor      │── Compressed Indexer    │            │
   │ (同压缩操作)    │   Keys (K^IComp)        │            │
   └─────────────────┘           │            │            │
                                 │   ┌────────▼────────┐    │
                                 │   │ Lightning       │    │
                                 │   │ Indexer         │    │
                                 │   │ I_{t,s}=Σ w·ReLU│    │
                                 │   │ (q^I·K^IComp)   │    │
                                 │   └────────┬────────┘    │
                                 │       Index Scores       │
                                 │            │             │
                                 │   ┌────────▼────────┐    │
                                 └──►│ Top-k Selector  │    │
                                     │ (保留 k 条)     │    │
                                     └────────┬────────┘    │
                                  Selected Compressed       │
                                     KV Entries             │
                                              │             │
   Sliding Window KV Entries ─── Concatenation ◄────────────┤
   (recent n_win, 未压缩)               │                   │
                                ┌───────▼───────────────────▼───┐
                                │ Shared Key-Value Multi-Query   │
                                │ Attention (MQA, K=V=选中条目)  │
                                └───────────────┬───────────────┘
                                                │  {o_{t,i}}, i=1..n_h
                                ┌───────────────▼───────────────┐
                                │ Grouped Output Projection      │
                                │ n_h→g 组→各组 d_g→concat→ R^d  │
                                └───────────────┬───────────────┘
                                          attention output  ô_t ∈ R^d
```

*图注*：复刻 Figure 3（p9）的数据流 + Eq 9–19。**两路 Token-Level Compressor** 用同一压缩操作（Eq 9–12，重叠窗口、softmax 跨 2m 元素归一，净压缩恰为 **1/m**）分别产出压缩 KV 条目 `C^Comp` 与压缩 indexer keys `K^IComp`。lightning indexer 算 index score `I_{t,s}`（Eq 15–16，causal `s<⌊t/m⌋`），top-k 选 `k` 条（Eq 17）。MQA 用共享 latent `c^Q_t` 升投影出 query（Eq 18），对选中条目 + SWA 条目做 core attention（Eq 19）。承载维度（Flash / Pro）：`m=4`；indexer `n_h^I=64, c^I=128`；**top-k `k=512 / 1024`**；`n_h=64 / 128, c=512, d_c=1024 / 1536`；输出投影 `g=8 / 16, d_g=1024`；`n_win=128`。补充技巧（§2.3.3，p12–13）：core attention 前对 query 与 KV 头做 **RMSNorm**；**Partial RoPE** 只作用于 query/KV 的**最后 64 维**，并对 `o_{t,i}` 末 64 维施加位置 `−i` 的 RoPE 使输出携带**相对**位置；**Attention Sink**（Eq 27）在分母加 `Exp(z'_h)`，允许某头总注意力 ≠1。

---

## 4. HCA 内部结构（Figure 4，§2.3.2，p11–12）

HCA = **更狠的逐 m′ token 压缩**（1/m′，`m′≫m`，**无重叠窗口**）+ **shared-KV MQA** + **分组输出投影** + SWA 分支。相对 CSA，HCA **没有 lightning indexer、没有 top-k**——每个 query 直接对所有在前的压缩块做注意力。

```
   Hidden States of KV Tokens                 Hidden State of Query Token
            │                                          │
   ┌────────┴────────┐                        c^Q_t = h_t · W^DQ  (Eq 24)
   │ Token-Level     │                                 │
   │ Compressor      │── Heavily Compressed       Queries  q_t = c^Q·W^UQ
   │ (每 m′ token→1, │   KV Entries                     │   (Eq 25)
   │  无重叠, 1/m′)  │   (C^Comp, 长度 n/m′)            │
   └────────┬────────┘           │                     │
            │                     │                     │
   Sliding Window KV Entries ─ Concatenation ◄──────────┤
   (recent n_win, 未压缩)         │                     │
                          ┌───────▼─────────────────────▼───┐
                          │ Shared Key-Value Multi-Query     │
                          │ Attention (MQA, K=V=压缩条目)    │
                          │   —— 无 indexer / 无 top-k ——    │
                          └───────────────┬─────────────────┘
                                          │  {o_{t,i}}, i=1..n_h
                          ┌───────────────▼─────────────────┐
                          │ Grouped Output Projection        │
                          │ n_h→g 组→各组 d_g→concat→ R^d    │
                          └───────────────┬─────────────────┘
                                    attention output  ô_t ∈ R^d
```

*图注*：复刻 Figure 4（p11）+ Eq 20–26。压缩与 CSA 同理但用更大率 `m′` 且**不重叠**（Eq 22 的 softmax 只跨 m′ 个元素），序列长压到 **1/m′**。MQA / 分组输出投影与 CSA 共用（Eq 24–26），同样叠加 SWA 分支、RMSNorm、Partial RoPE、Attention Sink（§2.3.3）。承载维度：`m′=128`（两模型同）；`n_h=64 / 128, c=512, d_c=1024 / 1536`；`g=8 / 16, d_g=1024`；`n_win=128`。**关键区别**：HCA 无稀疏选择，是为极致压缩；CSA 才挂 DSA top-k。

---

## 5. DeepSeekMoE FFN 块（§2.1，p7；§4.2.1，p24–25）

每个 MoE 层 = **1 个共享专家**（始终激活）+ **N 个细粒度路由专家**（Flash 256 / Pro 384），每 token 经 router 选 **top-6** 路由专家激活。affinity 分数激活由 V3 的 `Sigmoid` 改为 **`Sqrt(Softplus(·))`**；负载均衡用**无辅助损失**策略 + 轻量**序列级 balance loss**；**前 3 个 MoE 层用 Hash 路由**（按 token ID 哈希定专家，替换 dense FFN）；并移除了 V3 的路由目标节点数约束。

```
                         token 隐藏态 h_t
                               │
            ┌──────────────────┼───────────────────────────┐
            │                  │                            │
            ▼                  ▼                            ▼
    ┌──────────────┐   ┌───────────────┐         ┌─────────────────────┐
    │ Shared Expert│   │ Router        │         │ Routed Expert 池     │
    │ (1 个, 恒激活)│   │ affinity =    │         │ N = 256(Flash)       │
    └──────┬───────┘   │ Sqrt(Softplus)│         │   / 384(Pro)         │
           │           │ → Top-6 选择  │         │ E_1 E_2 … E_N        │
           │           └───────┬───────┘         │ (中间维 2048 / 3072) │
           │                   │  选中 6 个       └──────────┬──────────┘
           │                   └──────────────┬─────────────┘
           │                                  │  仅 6 个路由专家前向
           ▼                                  ▼
        ┌────────────────────────────────────────────┐
        │  加权求和 (共享专家 + 6 个路由专家输出)      │
        └────────────────────┬───────────────────────┘
                             ▼
                        FFN 输出 ∈ R^d

  注：前 3 个 MoE 层改用 Hash 路由 (按 token-ID 哈希定专家, 替换 dense FFN)。
      专家不按领域命名; K 固定为 6 (非 8–12); N 为 256/384 (非 128)。
```

*图注*：复刻 §2.1（p7）描述 + §4.2.1（p24–25）配置。承载计数（Flash / Pro）：共享 `1`，路由 `256 / 384`，专家中间维 `2048 / 3072`，每 token 激活 `6`，前 `3` 个 MoE 层 Hash 路由。V4 在**所有** Transformer block 都用 MoE（不再有 dense-FFN 前缀层，仅前 3 层换成 Hash 路由的 MoE）。

---

## 6. 与旧版臆造图的差异（审计纠正点）

本页整页替换了预发布期 AI 生成的旧图，纠正以下臆造（详见 [[30_deepseek_v4_audit_analysis]]）：

- 路由专家 **256 / 384**（旧图误作 128）；激活 **K=6 固定**（旧图误作 8–12，且臆造「按任务 5%/35% 激活」——13B/49B 实为 Flash/Pro 两个**不同模型**的激活量）。
- 专家**不按领域命名**（旧图臆造「代码/数学/推理/对话」专家）。
- **无 "O(n log n)" 复杂度**之说（论文未给该断言）。
- **无 "DualPath / SNIC / CNIC" 推理框架**（论文 §3.5 推理框架为异构 KV cache + on-disk KV，见 [[23_deepseek_v4_cp_analysis]]）。
- CSA 与 HCA 是**逐层独立变体并交错**，非同一块内并存；首 2 层非对称（Flash=纯 SWA，Pro=HCA）。
- 必含 **MTP 模块**（Figure 2 中 Prediction Head 与 MTP Modules 并列）。

---

## Related / Cross-references

- [[13_deepseek_v4_analysis]] — V4 整体架构深读（本页结构图的文字底座）
- [[26_deepseek_v4_technical_deepdive]] — CSA / HCA / DSA / MLA 机制级对比
- [[27_deepseek_v4_implementation_deepdive]] — 配置与实现要点
- [[23_deepseek_v4_cp_analysis]] — Contextual Parallelism 与推理框架（异构 KV cache / on-disk KV）
- [[24_deepseek_v4_fp4_qat_analysis]] — FP4 QAT（后训练；MoE 权重 + CSA indexer QK 路径）
- [[25_mhc_analysis]] — 流形约束超连接（B_l 双随机约束 + Sinkhorn-Knopp）
- [[30_deepseek_v4_audit_analysis]] — 本页所依据的正式版审计报告
- [[20_deepseek_moe_analysis]] — DeepSeekMoE（共享 + 细粒度路由专家）
- [[12_deepseek_v3_analysis]] — V3 架构（MLA / FP8 / DualPipe），V4 的对照基线
