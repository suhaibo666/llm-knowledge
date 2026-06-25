# DeepSeek-V4 Wiki 审计报告 — 以正式发表版核对既有 V4 页面

> **核对基线（Source baseline）**: arXiv:**2606.19348v1** 「DeepSeek-V4: Towards Highly Efficient
> Million-Token Context Intelligence」, DeepSeek-AI, 提交于 **2026-04-26**。
> **审计日期**: 2026-06-25　**维度**: 审计 / 核对 / 订正（Reconciliation）
> 本页回答：wiki 中既有的 ~7 篇 DeepSeek-V4 页面，哪些断言/数字/出处与**正式发表版**一致，哪些
> 是**预发布残留**或**未源于论文的臆造**。它是对 [[deepseek_v4_analysis]] 及其专题页的一次源忠实复核。

---

## 0. 一句话结论（主线）

既有 V4 页面是**论文正式上 arXiv 前两天（2026-04-24，基于一份无编号预发布 PDF / AI 合成笔记）**写成的。
复核结论分两类：**真正逐节读论文写的页面，其超参与基准数字全部正确**，只需订正「章节号位移」「FP4 归属」
「基线头」三类**结构性过时**；但**有三篇页面并非源于论文**，而是 AI 生成的臆造内容，含大量论文中**根本
不存在**的机制与数字（最典型：贯穿三页的 “DualPath 推理框架” 及其 1.87×/1.96× 吞吐——`DualPath` 在论文
全文出现 **0 次**）。

---

## 1. 逐页裁决（Verdict）

| 页面 | 是否真源于论文 | 裁决 | 主要问题 |
|---|---|---|---|
| [[deepseek_v4_analysis]] | ✅ 是 | **基本可信，做定点订正** | 基线头过时；FP4 归属应为后训练 §5.2.1；混入臆造的「DualPath 推理框架」小节；个别「顶尖」措辞被 Table 6 反证 |
| [[deepseek_v4_cp_analysis]] | ✅ 是（含真实论文引文） | **可信，仅订正章节号** | §3.5.3→§3.4.3、§3.6/§3.6.1/§3.6.2→§3.5/§3.5.1/§3.5.2 等约 13 处位移；基线头 |
| [[deepseek_v4_fp4_qat_analysis]] | ✅ 是 | **可信，定点订正** | 出处 §3.7 错误（应 §5.2.1）；旧路径 `raw/05_model_families/`；「三个组件 FP4 量化」应为「两组件 FP4 + 索引分数 BF16」；效果表数字多为未标注推断 |
| [[mHC]] | ✅ 是（源自 mHC 论文 arXiv:2512.24880v2） | **基本无需改动** | 其消融表（27B，59.0/63.0/63.4）来自 **mHC 论文自身实验**，并非 V4 论文，故「V4 论文查无此数」属正常；与 V4 §2.2 公式一致 |
| [[deepseek_v4_technical_deep_dive]] | 原 ❌（AI 臆造）→ 已重写 | ✅ **已据正式版整页重写**（2026-06-25） | 旧稿：DSA=CSA+HCA 倒置；Highly/Heavily；HCA「10%」应 m′=128；臆造分层调度/MoE 任务路由/DualPath。新稿逐方程引 §2.3 Eq 9–27 |
| [[deepseek_v4_implementation_details]] | 原 ❌（AI 臆造）→ 已重写 | ✅ **已据正式版整页重写**（2026-06-25） | 旧稿：专家 128/激活 8；HCA 0.1；Sinkhorn 100；Muon 实为 Adam；INT8；臆造 PCA/DualPath。新稿伪代码逐行锚定 Eq 1–28/Alg 1，常量取自 §4.2.1 |
| [[deepseek_v4_architecture_diagrams]] | 原 ❌（AI 臆造）→ 已重写 | ✅ **已据正式版整页重画**（2026-06-25） | 旧稿：128 专家/K=8–12/5%·35% 任务自适应/领域命名/O(n log n)/DualPath/缺 MTP。新稿复刻 Figure 2/3/4 + §4.2.1 配置 |

> [!important] 三篇 ❌ 页面原以「`来源: raw/...DeepSeek_V4_*.md（AI 辅助分析生成）`」为头、**未引用论文任何
> §/Table/Eq**，是预发布期 AI 臆造产物。**已于 2026-06-25 据正式版整页重写完成**：由并行 writer-agent 锚定核验过的
> 方程/超参事实表逐方程重写，协调者抽查定位符 + 机械校验（横幅已除、臆造词清零、跨链无悬挂、图内无链接泄漏）。

---

## 2. 核对通过的事实（正式版 = 唯一权威）

以下逐项开正式 PDF 对应位置读过，**与既有可信页面一致**：

### 2.1 超参（§4.2.1 Model Setups, p24–25）— 全部一致
| 参数 | V4-Flash | V4-Pro |
|---|---|---|
| Transformer 层数 | 43 | 61 |
| 隐藏维 d | 4096 | 7168 |
| 前 2 层 | **纯滑窗 SWA** | **HCA**（不对称！） |
| CSA 压缩率 m | 4 | 4 |
| HCA 压缩率 m′ | 128 | 128 |
| indexer query heads n_h^I / head dim c^I | 64 / 128 | 64 / 128 |
| 稀疏 attention top-k | 512 | 1024 |
| query heads n_h / head dim c | 64 / 512 | 128 / 512 |
| query 压缩维 d_c / 输出分组 g / 每组维 d_g | 1024 / 8 / 1024 | 1536 / 16 / 1024 |
| SWA 窗口 n_win | 128 | 128 |
| 共享专家 / 路由专家 / 专家中间维 | 1 / 256 / 2048 | 1 / 384 / 3072 |
| 激活专家/token | 6 | 6 |
| Hash 路由 | 前 3 个 MoE 层 | 前 3 个 MoE 层 |
| MTP depth / mHC n_hc / Sinkhorn t_max | 1 / 4 / 20 | 1 / 4 / 20 |
| 词表 / 上下文 | 128K / 1M | 128K / 1M |
| 总参 / 激活参 | 284B / 13B | 1.6T / 49B |
| 预训练 token | 32T | 33T |

### 2.2 头条效率（§1 / Fig.1, p5）— 一致
1M 上下文、对比 DeepSeek-V3.2：**V4-Pro = 单 token FLOPs 27%（FP8 等效）、KV cache 10%**；
**V4-Flash = FLOPs 10%、KV cache 7%**。路由专家用 FP4（当前硬件 FP4×FP8 峰值 FLOPs 同 FP8×FP8）。

### 2.3 基准数字 — 一致
- **Base（Table 1, p28）** 列 = V3.2 / Flash / Pro（激活 37B/13B/49B，总 671B/284B/1.6T）：
  MMLU-Pro(EM,5s) **65.5 / 68.3 / 73.5**；MMLU 87.8/88.7/90.1；HumanEval 62.8/69.5/76.8；
  MATH 60.5/57.4/64.5；LongBench-V2 40.2/44.7/51.5。
- **后训练（Table 6, p38）** DS-V4-Pro-Max 列：LiveCodeBench(Pass@1) **93.5**（所列开源最高）；
  MRCR 1M(MMR) **83.5**（Opus-4.6=92.9 领先、Gemini-3.1-Pro=76.3 落后 → V4-Pro 胜 Gemini、负 Opus）。

### 2.4 关键 V3→V4 架构变化（§2, p6–8）— 论文确证
mHC 替代朴素残差（B_l 约束到**双随机矩阵 / Birkhoff polytope**，Sinkhorn-Knopp t_max=20，‖B_l‖₂≤1）；
hybrid 注意力 = **CSA + HCA**（逐层交错），其中 **DSA（DeepSeek 稀疏注意力）是 CSA 内部的 top-k 选择步骤，
HCA 不做稀疏**；Muon 优化器；MoE 亲和度激活由 **Sigmoid → Sqrt(Softplus(·))**；前 3 个 MoE 层改用 **Hash 路由**；
取消路由目标节点数约束；MTP 配置同 V3（depth 1）。

---

## 3. 必须订正的「结构性过时」

### 3.1 章节号位移（根因：FP4-QAT 从 §3.4 移入后训练 §5.2.1，导致 §3 其后整体下移一位）
| 主题 | 预发布 §（wiki 现引） | 正式版 §（正确） |
|---|---|---|
| FP4 量化感知训练 | §3.4（及错误的 wiki「§3.7」） | **§5.2.1**（§5.2「Post-Training Infrastructures」） |
| 训练框架（总） | §3.5 | **§3.4** |
| 高效 Muon | §3.5.1 | **§3.4.1** |
| mHC 高效实现 | §3.5.2 | **§3.4.2** |
| 长上下文 Contextual Parallelism | §3.5.3 | **§3.4.3** |
| 扩展自动微分 / 检查点 | §3.5.4 | **§3.4.4** |
| 推理框架（总） | §3.6 | **§3.5** |
| KV Cache 结构与管理 | §3.6.1 | **§3.5.1** |
| On-Disk KV Cache | §3.6.2 | **§3.5.2** |

> 架构 §2、预训练 §4、后训练流水线 §5.1 的节号**未变**。

### 3.2 FP4 归属与口径
正式版把 FP4-QAT 明确放在**后训练**（§5.2.1）。FP4 仅作用于**两个组件**：(1) MoE 专家权重；(2) CSA indexer 的
**QK path**。**索引分数 I 是 FP32→BF16（不是 FP4）**，带来 top-k 选择器 **2× 加速 / 99.7% KV 召回**。MoE 权重路径
FP32 master → FP4 → **无损反量化到 FP8** 计算，反向 STE 回 FP32，复用既有 FP8 框架；推理/RL rollout 用**原生 FP4**。

---

## 4. 论文中**不存在**的内容（既有页面的臆造，按出处反证）

| 臆造断言 | 出现页面 | 反证（正式版全文计数 / 实情） |
|---|---|---|
| 「**DualPath** 推理框架」/ SNIC·CNIC / 路径 A·B / 吞吐 **1.87×·1.96×** | analysis、technical_deep_dive、implementation_details、architecture_diagrams | `DualPath`=0、`1.87`=0、`SNIC`/`CNIC`=0。真正的推理框架是 §3.5：KV-Cache 结构(§3.5.1) + On-Disk 存储(§3.5.2) |
| 「**DSA = CSA + HCA**」 | technical_deep_dive | 倒置。§2.3：hybrid=CSA+HCA；**DSA 是 CSA 内的 top-k 步骤**，HCA「does not employ sparse attention」 |
| 「Highly Compressed Attention」 | technical_deep_dive、implementation_details | 名称错。`Highly Compressed`=0，正名「**Heavily** Compressed」=7 |
| 「HCA 压缩比 10%（仅留 10% token）」 | technical_deep_dive、implementation_details | 与头条「KV 10%」混淆。HCA m′=**128**（≈0.78%） |
| 分层压缩调度 0.25/0.15/0.10（按层深） | technical_deep_dive、implementation_details、architecture_diagrams | 论文为 CSA/HCA 逐层**交错**、m/m′ **固定**，无层深调度 |
| MoE「任务类型动态路由」/ 5%·35% / k 随简单·复杂变 / `task_classifier` | technical_deep_dive、implementation_details、architecture_diagrams | `task_classifier`=0。固定**激活 6 专家**；13B/49B 是 **Flash/Pro 两个不同模型**的激活参，非一模型两档 |
| 路由专家 **128 个**、激活 **8 / 8–12 个** | implementation_details、architecture_diagrams | 应 **256(Flash)/384(Pro)** 路由 + 1 共享，**固定激活 6** |
| 专家按领域命名（代码/数学/推理/对话） | implementation_details、architecture_diagrams | 论文无语义专家分配（细粒度路由 + 亲和度分数） |
| Sinkhorn-Knopp `max_iter=100` | technical_deep_dive、implementation_details | 应 **t_max=20**（§2.2 Eq.8 / §4.2.1） |
| 「Muon」实现实为 Adam（有 bias-correction、无 Newton-Schulz） | implementation_details | Muon 核心是 Hybrid Newton-Schulz 正交化 + RMS rescale（§2.4 Alg.1） |
| 量化为 **INT8** / PCA·随机投影压 KV | implementation_details | 应 **FP4(MXFP4) QAT**；KV 压缩来自 CSA/HCA 架构 + On-Disk，非 PCA |
| 「O(n log n) 复杂度」 | technical_deep_dive、architecture_diagrams | `n log n`=0。DSA 是固定 top-k（512/1024）选择 |
| MLA「DeepSeek-V3 引入」 | technical_deep_dive | MLA 源自 **V2**；且 V4 用 CSA/HCA，不用 MLA |

> [!note] **不是臆造、但需标注为「本页推导」**：[[deepseek_v4_cp_analysis]] 的「CSA ~51× / HCA ~2048× 通信节省」
> 与 [[deepseek_v4_fp4_qat_analysis]] 的「~75% 内存 / ~2×」是页面**自身的公式推导/估算**，论文未给出该数字；保留但应明确标为推断。
> [[mHC]] 的「~3000× / ~1.6× 信号放大、6.7% 开销」来自 **mHC 论文实验**，非 V4 论文。

---

## 5. 处置与后续

1. ✅ Tier-A（analysis / cp_analysis / fp4_qat_analysis）：已就地订正（基线头、章节号、FP4 口径），并对臆造小节加
   `> [!contradiction]` 标注（遵守 wiki「不删除、仅标注」规则）。
2. ✅ Tier-B（technical_deep_dive / implementation_details / architecture_diagrams）：**已据正式版整页重写完成**（2026-06-25）。
   并行 writer-agent 锚定核验过的 `MECH_BRIEF`（CSA/HCA/DSA Eq 9–27、Muon Alg 1/Eq 28、mHC Eq 1–8、§4.2.1 配置）逐方程重写，
   每条断言带 §/Eq/page 定位符；协调者抽查定位符 + 机械校验（警示横幅已除、臆造词清零、跨链无悬挂、图内无 `[[…]]` 泄漏）。
3. ✅ mHC：保留，已补注「消融数据来自 mHC 论文 arXiv:2512.24880v2，非 V4」。

---

## Related / Cross-references

- [[deepseek_v4_analysis]] — V4 整体架构（Tier-A，已订正）
- [[deepseek_v4_cp_analysis]] · [[deepseek_v4_fp4_qat_analysis]] · [[mHC]] — 专题（Tier-A / 旁证）
- [[deepseek_v4_technical_deep_dive]] · [[deepseek_v4_implementation_details]] · [[deepseek_v4_architecture_diagrams]] — 已据正式版整页重写（原 Tier-B）
- [[deepseek_v3_analysis]] — V3 基线（MLA / FP8 / DualPipe / MTP）
- [[deepseek_moe_analysis]] — DeepSeekMoE（细粒度专家 + 共享专家）
- [[muon_analysis]] — Muon 优化器
