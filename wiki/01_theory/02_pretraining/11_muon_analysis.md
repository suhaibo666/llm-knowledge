---
title: "Muon 优化器：正交化原理与分片冲突"
---

# Muon 优化器：正交化原理与分片冲突

> **主题**：Muon 把梯度当作矩阵、用 Newton–Schulz 迭代正交化更新方向；本页只讲这套数学，以及它与「按元素切分优化器状态」（ZeRO 系）为什么在原理上冲突、论文给出的修法是什么。
> **来源**：Muon Is Scalable for LLM Training（arXiv:2502.16982）与 Jordan et al. 的原始 Muon 表述。本页不分析代码，因此不钉源码基线。
> **适用范围**：算法与分片冲突原理。各训练框架的工程落地、各家模型的 Muon 配方（per-head / per-expert / Muon Split / MuonClip 等）都不在本页，路由见 §3。
> **最近更新**：2026-09-08。§3 收缩为工程落地转指，本页回到纯算法职责。

---

## 1. Muon 优化器核心原理

Muon (Momentum Orthogonalized) 是一种专为神经网络（特别是大规模 Transformer）设计的新型优化器。其核心思想是对参数更新量进行**正交化 (Orthogonalization)** 处理。

### 1.1 核心算法：Newton-Schulz 迭代

不同于 AdamW 这种逐元素 (Element-wise) 更新的优化器，Muon 将梯度视为矩阵，并对其进行正交化变换。
Muon 使用 **Newton-Schulz (NS) 迭代** 来逼近矩阵的奇异值分解 (SVD) 或正交化结果，从而使得更新量的谱范数 (Spectral Norm) 受到约束。

**算法步骤 (简化版)**:
1.  **输入**: 梯度矩阵 $G$。
2.  **迭代**: 运行 $K$ 次 Newton-Schulz 迭代（通常 $K=5$）：

    $$
    X_{k+1} = a X_k + b X_k X_k^T X_k + c X_k (X_k^T X_k)^2
    $$

    其中 $a, b, c$ 是特定系数。
3.  **缩放**: 对输出矩阵进行缩放（Scaling），以匹配期望的 RMS 或谱范数。
4.  **更新**: 使用处理后的矩阵 $U$ 更新参数 $W$。

---

## 2. 分布式训练中的挑战：Muon vs ZeRO

在分布式训练（如 ZeRO-1/2/3）中，模型参数和优化器状态通常被切分到不同的 GPU (Rank) 上。这对 Muon 提出了挑战。

### 2.1 为什么 Muon 不兼容原生 ZeRO-1？

*   **ZeRO-1 机制**：将完整的参数矩阵 $W$ 切分为 $N$ 个碎片 (Shard)。每个 Rank $i$ 只持有并维护第 $i$ 个碎片 $w_i$ 及其对应的梯度 $g_i$。
*   **Muon 的冲突**：Muon 的 Newton-Schulz 算法是**整体矩阵运算**。
    *   $NS(G) \neq Concat(NS(g_1), NS(g_2), ...)$
    *   如果在切分后的 $g_i$ 上直接运行 Muon，数学上完全不等价，且失去了正交化的物理意义（变成了一种 Block-Diagonal 近似，效果可能很差）。

### 2.2 论文提出的解决方案：Distributed Muon

论文中提出的算法试图在 ZeRO-1 框架下修复这个问题：
1.  **Reduce-Scatter**: 得到分片梯度 $g_i$。
2.  **DP Gather (关键步骤)**: 每个 Rank 临时收集所有其他 Rank 的 $g_j$，拼凑出完整的 $G$。
3.  **Local Compute**: 在完整 $G$ 上跑 Newton-Schulz，得到完整 $U$。
4.  **Discard**: 丢弃不属于自己的部分，只保留 $u_i$。
5.  **Update**: 更新本地 $w_i$。

---

## 3. 工程落地：本页不拥有的部分

Muon 的分布式实现变化很快（Megatron 的实现已经整体外移到独立的 `emerging-optimizers` 包），
把它写在这张理论页上必然过时。**本页只保留算法与冲突原理，落地一律转指 owner 页。**

> [!deprecated]
> 本页旧版在这一节写过一份 Megatron 实现分析（基于 Merge Request 4106，页头未钉基线）。
> 它的若干结论已被当前基线证伪——例如把实现归给 `megatron/core/optimizer/muon.py`
> （该文件现在只是 28 行兼容 shim）、以及描述为「显式禁用 `DistributedOptimizer`」
> （实际是 `ChainedOptimizer` 把 LayerWise 与 `DistributedOptimizer` 串联，两者并存）。
> 该节已删除，以 [[26_megatron_optimizer_step_internals_deepdive]] 为准。

### 3.1 框架侧实现

| 关心什么 | 去哪一页 | 该页拥有的内容 |
|---|---|---|
| Megatron-LM 怎样接 Muon | [[26_megatron_optimizer_step_internals_deepdive]] §4.4 | 触发组合（`--optimizer muon` + `--use-distributed-optimizer`，**没有** `--layer-wise-distributed-optimizer` 这个 flag）、`is_managed_by_layer_wise_optimizer` 路由、`ChainedOptimizer` 串联、QKV 切分形状、`qk_clip`、硬约束与失败边界 |
| 为什么 range 切法容不下 Muon | [[16_megatron_distributed_optimizer_analysis]] §2.5 | 两条 LayerWise whole-parameter 数据面；$k>1$ 会让 Muon 梯度跨 DP 域欠规约的原因 |
| TorchTitan / Kimi 的 DistMuon | [[26_torchtitan_flex_shard_dist_muon_analysis]] | 存储所有权与优化器计算所有权分离、per-expert layout、all-to-all 计划与双槽运行时 |
| Ascend / MindSpeed 的反向移植 | [[13_mindspeed_ascend_affinity_analysis]] | `--muon-num-ns-steps` / `--muon-scale-mode` / `--muon-tp-mode` 等旋钮，NS 迭代落到 Cube 核 |
| Adam vs Muon 的显存与系统性影响 | [[32_distributed_optimizer_deepdive]] §5–6 | 各 ZeRO stage 的 bytes/param、非 element-wise 优化器对 overlap 与分片体系的冲击 |

### 3.2 正交化粒度：per-head / per-expert 与更细的拆分

原始 Muon 把**整个参数矩阵**当作 NS 的操作单元。这对无语义子结构的 FFN 是对的，
对注意力投影与 MoE 专家则不是——这一支扩展由下面两页拥有，本页不复述：

| 关心什么 | 去哪一页 |
|---|---|
| 粒度应与语义独立单元对齐的设计原则、per-head / per-expert / 整矩阵三分法，以及 FSDP、TP+FSDP 下的分头 NS 流水线（含 $W_o$ 与 FFN $W_{\mathrm{down}}$ 的关键区别） | [[22_muon_sharded_hsdp_analysis]] §08–§10 |
| 融合权重（qkv、SwiGLU fc1）**直接正交化是错的**：迭代会混合不相关子块，且缩放因子用了拼接后的形状；以及 Canzona 的 $\alpha$-均衡分区与跨 TP 融合 All-to-All | [[21_qwen3_8_flash_next_optimization_deepdive]] §2.3–§2.4 |

> **框架现状（2026-09-08 在 `NVIDIA/Megatron-LM@85902ef5` 上核过）**：Megatron 基线**不提供 per-head 粒度**。
> `muon_split_qkv` 默认开，但切出的是 Q / K / V 三个**完整投影矩阵**，group 与 head 都不是独立的正交化单元；
> 它修掉的是缩放因子用错形状，没修掉跨 head 混合奇异方向。per-expert 则无需开关——专家本就是逐个的 2D 参数。
> 证据与边界见 [[26_megatron_optimizer_step_internals_deepdive]] §4.4。

### 3.3 模型侧配方

| 模型 | 变体 | owner 页 |
|---|---|---|
| Kimi K2 | MuonClip（Muon + QK-Clip） | [[11_kimi_k2_analysis]] |
| Kimi K3 | Per-Head Muon + 保留 K2 weight-clipping + QB | [[25_kimi_k3_stability_analysis]] |
| GLM-5 | Muon Split（配合它，attention logits 全程稳定、无需 clipping） | [[20_glm5_architecture_deepdive]]；分布式实现见 [[22_glm5_training_infra_deepdive]] |
| DeepSeek-V4 | Hybrid Newton–Schulz；**不用** QK-Clip（Q/KV RMSNorm 已足够） | [[27_deepseek_v4_implementation_deepdive]] |
| Qwen3.5 Flash-Next | Muon + Gated Residual；路由器与低秩投影走 AdamW | [[21_qwen3_8_flash_next_optimization_deepdive]] |
| LongCat-2.0 | 异构 ASIC 上把 Muon 跑到 1.6T 规模 | [[longcat_2_analysis]] |

---

## 4. 总结

1. **算法**：Muon 用 Newton–Schulz 迭代把动量矩阵正交化，约束更新的谱范数，使每步更新在谱范数球面上做梯度下降。
2. **分片冲突的根源**：NS 是整体矩阵运算，$NS(G) \neq Concat(NS(g_1), NS(g_2), \dots)$，所以按元素切分参数的 ZeRO-1 无法直接承载 Muon——这是所有分布式 Muon 方案共同要解的一件事。
3. **两类解法**：论文路线是**临时 gather 出完整矩阵、算完丢弃**（通信换正确性）；工程路线是**让每个矩阵整体落在某个 shard 内**（Megatron 的 layer-wise 分布式优化器），从而无需额外 gather。
4. **一个正交的扩展方向**：正交化粒度不必是整个矩阵。按 head / 按 expert 拆分既更贴合架构语义，又顺带把 NS 的 $\min^2\times\max$ 成本大幅压低——见 §3.2。

## Related Pages

- [[26_megatron_optimizer_step_internals_deepdive]] — Muon 在 Megatron-LM 里的完整集成（触发、路由、约束），本页 §3.1 的主要 owner。
- [[22_muon_sharded_hsdp_analysis]] — 分片 Muon、双网格 HSDP 与 per-head / per-expert 正交化粒度，本页 §3.2 的主要 owner。
- [[32_distributed_optimizer_deepdive]] — Adam 与 Muon 在各 ZeRO stage 下的显存与通信对照。
- [[25_kimi_k3_stability_analysis]] — Per-Head Muon 的模型侧证据与稳定性论证。
- [[21_qwen3_8_flash_next_optimization_deepdive]] — 融合权重必须按语义算子边界拆分的论证，以及 NS 步数选择的稳定性理由。
- [[01_theory/02_pretraining/index|LLM 训练技术]] — 本页所属目录。
