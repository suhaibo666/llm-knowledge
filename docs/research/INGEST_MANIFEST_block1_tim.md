# 第 1 块摄入清单：训推一致性（Training-Inference Mismatch, TIM）

> 生成日期：2026-07-25
> 用途：本文件是 **raw/ 摄入的施工单**，不是 wiki 内容页。所有条目均经实际打开页面核实；未核实项显式标注，不做静默省略。
> 路线：按「先补齐 raw 再动笔」的严格路线执行（对齐 `16_graph_compiler_foundations` 那批页的标准，而非 `07_training_reliability` 的二手综述路线）。

---

## 0. 目标页与本清单的关系

第 1 块的产出**不是**新起一个大簇，而是补上仓库里断掉的那一环：

```
kernel 非确定性  →  logprob 偏差  →  重要性比重尾  →  训练崩溃
   ▲已有                 ▲缺口            ▲缺口           ▲部分已有
determinism_and_          ← 本次 raw 主要服务的两环 →    training_dynamics_
numerical_reliability                                    stability_analysis
batch_invariance_guide
vllm/ 13 篇
```

计划产出：
- **新页 1 篇**：`wiki/01_theory/04_posttraining/tim_causal_chain_analysis.md` —— 四环因果链 + 病因/症状/修法/代价对照表
- **新页 1 篇**：`wiki/01_theory/04_posttraining/moe_routing_replay_analysis.md` —— R3 / PR² / RSPO 谱系与 GSPO 的立场冲突
- **增补 2 处**：`determinism_and_numerical_reliability_analysis.md`（问题 2 补算法侧全谱）、`RL_Training_Inference_Precision_Analysis.md`（补 2025-2026 一代）

---

## 1. 已核实来源（可脚本下载）

### A. `raw/01_theory/04_posttraining/tim/` —— TIM 诊断与算法侧修正

| # | 目标文件名 | arXiv | 标题 / 机构 | 在因果链上的位置 | 备注 |
|---|---|---|---|---|---|
| 1 | `Diagnosing_Training_Inference_Mismatch-2605.14220.pdf` | 2605.14220 | Diagnosing Training Inference Mismatch in LLM RL — ByteDance + Univ. of Virginia，2026-05-14 | **全链主骨架** | ★★★ 本块第一优先。构造 VeXact 零 mismatch 基线，证明纯 token 级数值不一致即可独立致崩；并消融 TIS / RS |
| 2 | `TIS_On_Rollout_Training_Mismatch-openreview_8MHqvb4lK9.pdf` | 无 arXiv | On the Rollout-Training Mismatch in Modern RL Systems — Yao/Liu et al., MSR + UCSD, NeurIPS 2025 WS | 修法（TIS 出处） | **不要给它编 arXiv ID**。只有 OpenReview + Notion 博客两个形态 |
| 3 | `TRM_Trust_Region_Masking_SeqMIS-2512.23075.pdf` | 2512.23075 | Trust Region Masking for Long-Horizon LLM RL — v5 2026-06-26 | 修法（Seq-MIS / RS 出处） | 「拒绝采样修正 TIM」实际就是这一篇；verl 实现为 Seq-MIS / Geo-RS / K3-RS |
| 4 | `ALP_Adaptive_Layerwise_Perturbation-2603.19470.pdf` | 2603.19470 | Adaptive Layerwise Perturbation — UIUC + Amazon | 修法（统一 staleness + TIM） | 把两类 off-policy 源统一进单个 ratio，避免多阈值调参 |
| 5 | `MIPU_Mirage_of_Optimizing_Training_Policies-2606.29526.pdf` | 2606.29526 | The Mirage of Optimizing Training Policies — 天津大学 + 阿里巴巴 | 目标重定义 | 主张真正该单调改进的是推理策略 µ 而非训练策略 π |
| 6 | `FP16_Defeating_TIM-2510.26788.pdf` | 2510.26788 | Defeating the Training-Inference Mismatch via FP16 — Sea AI Lab + NUS，2025-10-30 | 病因（精度）+ 长度累积 | ★★★ KL 7.64 (BF16) vs 0.32 (FP16)；序列 mismatch-长度斜率 −1.01 vs −0.07 |
| 7 | `Beyond_Precision_TIM_is_Optimization-2602.01826.pdf` | 2602.01826 | Beyond Precision: TIM is an Optimization Problem — NUS/复旦/港中深/中科大 | **反方立场** | 反驳「必须换 FP16」：Theorem 3.1 梯度误差界 ~ C·T²；LR 调度即可缓解 |
| 8 | `QaRL_Rollout_Aligned_QAT_RL-2604.07853.pdf` | 2604.07853 | QaRL — 港科大 + 港城大 + 浙大，2026-04-09 | 量化 rollout 放大 TIM | 量化把 TIM 从 kernel 级微扰放大成分布漂移 |
| 9 | `FP8_RL_Low_Precision_Stack-2601.18150.pdf` | 2601.18150 | FP8-RL — NVIDIA 北京 | 阈值性观测 | MoE FP8 rollout-only 在 step ~700 KL 冲过 5 后崩溃 |
| 10 | `AIS_Adaptive_Importance_Sampling_Quantized_RL-2605.13907.pdf` | 2605.13907 | AIS — 华为 + 港大 | 诊断指标（ESS） | ESS ratio 0.95 → 0.70 的衰减曲线 |
| 11 | `M2PO_Prosperity_before_Collapse-2510.01161.pdf` | 2510.01161 | Prosperity before Collapse — CMU + Meta AI | **重尾→崩溃这一环最直接** | 约束重要性权重二阶矩而非裁剪；被裁 token 1.22% → 0.06% |
| 12 | `VCPO_Stable_Asynchrony_ESS-2602.17616.pdf` | 2602.17616 | Stable Asynchrony (VCPO) — MIT + NVIDIA | 前兆指标 | ESS ratio 退化**先于** KL 爆炸；TIS run 梯度尖峰后迅速崩溃 |
| 13 | `Stabilizing_RL_with_LLMs_Qwen-2512.01374.pdf` | 2512.01374 | Stabilizing RL with LLMs: Formulation and Practices — Qwen Team | 理论统一 | 一阶近似解释 IS/clipping/Routing Replay 为何有效 |
| 14 | `Jackpot_Budgeted_Rejection_Sampling-2602.06107.pdf` | 2602.06107 | Jackpot — CMU | **相邻但不同问题** | 修的是「小 rollout 模型 vs 大 target 模型」，**不是**数值级 TIM。入库需显式标注区别 |
| 15 | `RGPO_Rejection_Gated_Policy_Optimization-2604.14895.pdf` | 2604.14895 | Beyond Importance Sampling: RGPO | ⚠️ **质量存疑** | 作者仅列 QQ/gmail 邮箱、无机构；实证在 MuJoCo 而非 LLM。**只引其幂律尾理论命题（α ≤ 2 ⇒ 方差发散），实证数字不要用** |

### B. `raw/01_theory/04_posttraining/moe_rl/` —— MoE 路由漂移与 RL

| # | 目标文件名 | arXiv | 标题 / 机构 | 备注 |
|---|---|---|---|---|
| 16 | `R3_Rollout_Routing_Replay-2510.11370.pdf` | 2510.11370 | Stabilizing MoE RL by Aligning Training and Inference Routers — 北大 + 小米 LLM-Core，ICLR 2026 | ★★★ 训推 KL 1.535e-3 (MoE) vs 6.4e-4 (Dense)，R3 后 7.5e-4；~10% router 选到不同专家、94% token 至少一层不一致 |
| 17 | `PR2_Predictive_Routing_Replay-2606.00395.pdf` | 2606.00395 | PR²: Predictive Routing Replay — Rutgers / AMD / MBZUAI | 指出 R3 式冻结回放导致 **router staleness**，加 evolution predictor |
| 18 | `RSPO_Router_Shift_MoE_RL-2510.23027.pdf` | 2510.23027 | Towards Stable and Effective RL for MoE — MSR + 北大 | ★ 「router shift → 重要性比剧烈波动 → 突发 clipping」，该模式**稳定地先于崩溃出现** |
| 19 | `CompassMax_Router_Replay_100B_MoE-2512.07710.pdf` | 2512.07710 | Each Prompt Matters — Shopee LLM Team | 工业验证：训推 log-prob 差异**在 router 层之后显著放大**，10⁻³ → 10⁻⁴ |

> 已在库，无需下载：`raw/01_theory/04_posttraining/GSPO_Group_Sequence_Policy_Optimization-2507.18071.md`
> —— 其 **§5.3** 是本主题的关键对立面（已核实原文：「~10% 专家在更新后不同」「GSPO eliminates the dependency on Routing Replay」）

### C. `raw/01_theory/04_posttraining/collapse_diagnosis/` —— 崩溃诊断指标（同时服务第 4 块）

| # | 目标文件名 | arXiv | 标题 / 机构 | 备注 |
|---|---|---|---|---|
| 20 | `AVSPO_Advantage_Collapse_ACR-2605.21125.pdf` | 2605.21125 | Advantage Collapse in GRPO: Diagnosis and Mitigation — 国防科大 + 港中深，ICML 2026 | ★ ACR = Advantage Collapse Rate，AVSPO = 缓解方法。**Final Accuracy = 51.4 − 29.6 × ACR₁₀₀，R² = 0.617**（早期指标预测最终性能） |
| 21 | `OPEFO_Entropy_Collapse_Entropy_Flow-2605.11491.pdf` | 2605.11491 | Understanding and Preventing Entropy Collapse in RLVR — 南洋理工 + 上交 + VinUni，ACL 2026 Findings | ★ token 级熵流分解 S⁺/S⁻，失衡条件 Σ_{S⁻}\|ΔH\| > Σ_{S⁺}ΔH |

### D. `raw/02_engineering/04_posttrain_frameworks/determinism/` —— 系统 / kernel 侧

| # | 目标文件名 | arXiv | 标题 / 机构 | 备注 |
|---|---|---|---|---|
| 22 | `TBIK_Deterministic_Inference_Across_TP-2511.17826.pdf` | 2511.17826 | Deterministic Inference across Tensor Parallel Sizes — v1 2025-11-21 匿名投稿 / **v2 2026-05-29 去匿名** | ★★ TBIK = **Tree-Based Invariant Kernels**。TP∈{1,2,4,8}×batch∈{8,16,32} 全配置 unique output 恒为 1；GSM8K RL rollout-training **KL = 0**；开销 22%–63%。代码 github.com/nanomaoli/llm_reproducibility。⚠️ **作者机构无法核实，不要在 wiki 里填** |
| 23 | `LLM42_Determinism_Verified_Speculation-2601.17768.pdf` | 2601.17768 | LLM-42 — Microsoft Research + UW，2026-01-25 | 「固定 tiling 代价」的量化来源：batch-invariant Triton GEMM 194 vs cuBLAS 527 TFLOPS（**降速 63%**）；改用 decode-verify-rollback 选择性保证 |
| 24 | `BitExact_Inference_Verification-2606.00279.pdf` | 2606.00279 | Bit-Exact AI Inference Verification Without Performance Tradeoffs — Naci Cankaya，2026-06-05 | **反方立场**：区分「真非确定性」与「非不变性」，主张消除 atomics 后全速即可 bitwise 复现。⚠️ 页面未列机构 |

> 已在库，无需下载：`raw/01_theory/01_models/deepseek/DeepSeek_V4.md`（= arXiv 2606.19348）
> —— 关键章节 **§3.3 High-Performance Batch-Invariant and Deterministic Kernel Libraries**（§3.3.1 Batch Invariance / §3.3.2 Determinism）
> —— ⚠️ **重要语义修正**：dual-kernel strategy 原文是 "dual-kernel strategy for **batch-invariant decoding**"，抵消的是**放弃 split-KV 后 decoding attention 的性能损失**，**不是**「固定 tiling 的 matmul 损失」。matmul 侧另有做法：split-k 必须用时「output each split part separately and perform a deterministic reduction in a subsequent kernel」。落库前请按 PDF 原文再核一次 §3.3

### E. `raw/02_engineering/04_posttrain_frameworks/moe_scheduling/` —— MoE 系统侧（顺手收，第 2 块用）

| # | 目标文件名 | arXiv | 标题 / 机构 | 备注 |
|---|---|---|---|---|
| 25 | `ReLibra_Routing_Replay_Load_Balancing-2605.08639.pdf` | 2605.08639 | ReLibra — 北大，2026-05-09 | 把 routing replay 的「路由训练前已知」性质转成负载均衡先验，较 Megatron 最高 1.6× |
| 26 | `ForeMoE_Routing_Foresight_Microstep_LB-2606.11867.pdf` | 2606.11867 | Harnessing Routing Foresight for Micro-step-level MoE Load Balancing — 北大/上交/腾讯，2026-06-10 | 微步级专家负载均衡，64 GPU 最高 1.45×。**注意定位**：把 router replay 当既定约束而非研究对象 |

### F. 文档类（脚本可下载）

| 目标文件名 | 来源 |
|---|---|
| `vllm_docs_batch_invariance.md` | `raw.githubusercontent.com/vllm-project/vllm/main/docs/features/batch_invariance.md` — 开关 `VLLM_BATCH_INVARIANT=1`，要求 CC ≥ 8.0，状态 beta |
| `sglang_docs_deterministic_inference.md` | `raw.githubusercontent.com/sgl-project/sglang/main/docs/advanced_features/deterministic_inference.md` — `--enable-deterministic-inference`，FA3/Triton 全功能，FlashInfer 不支持 radix cache |
| `miles_sglang_tim_all_in_one_blog.md` | `raw.githubusercontent.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/main/rlhf/slime/mismatch/blog-en.md` — K3 KL 监控指标；**TIS clip [0.5, 2.0] 会崩、[0.5, 1.5] 不崩** |

---

## 2. 需手动存档（脚本抓不到：需 JS 渲染或原生 HTML）

| 内容 | URL | 建议存法 |
|---|---|---|
| Defeating Nondeterminism in LLM Inference（Horace He / Thinking Machines，2025-09-10） | https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ | 浏览器另存单页 HTML。★ 温度 0 采样 1000 次得 80 个不同 completion，前 102 token 一致、第 103 首次发散 |
| No More Train-Inference Mismatch（vLLM + TorchTitan，2025-11-10） | https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference | ★ 首个开源 bitwise 一致 on-policy RL run；**bitwise run 比非 bitwise 慢 2.4×** |
| Native RL APIs in vLLM（2026-05-28） | https://vllm.ai/blog/2026-05-28-native-rl-apis | 四阶段权重传输 API（init → start → update → finish） |
| Towards Deterministic Inference in SGLang（LMSYS，2025-09-22） | https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/ | ★ 确定性模式平均降速 **34.35%**（对比 TML 方案 61.5%） |
| When Speed Kills Stability（Seq-MIS/Seq-TIS 术语出处） | https://richardli.xyz/post/rl-collapse-part1/ 与 /rl-collapse-part3/ | ⚠️ Notion 镜像需 JS，本次**未能核实正文数字**。请用浏览器打开存档，不要凭二手转述填数 |
| Your Efficient RL Framework Secretly Brings You Off-Policy RL（TIS 原始博客） | https://fengyao.notion.site/off-policy-rl | ⚠️ 需 JS。正式版用清单第 2 项的 OpenReview PDF |
| vLLM 2026 Q2 RL Roadmap | github.com/vllm-project/vllm/issues/41733 | issue 正文 + 关键子项 #31848（权重同步生命周期 RFC）、#39451（稀疏原地权重更新，O(numel) → O(nnz)）、#39701（routing replay 换 CUDA-graph 兼容 device cache） |

---

## 3. 必须打 `> [!contradiction]` 的对立主张

| 议题 | 立场 A | 立场 B | 说明 |
|---|---|---|---|
| Routing Replay 是否必要 | **GSPO §5.3**（Qwen, 2507.18071）：序列级比值「eliminates the dependency on Routing Replay」 | **R3**（2510.11370）实验结论「R3 优于 GSPO」；**PR²**、**CompassMax** 均保留 | 同一 Qwen 作者组（Chujie Zheng 一作）在 **2512.01374 §3** 自我修正为「off-policy 时 clipping + Routing Replay 组合成为必需」。三方并列，链条清晰 |
| TIM 的根因是精度还是优化 | **FP16 论文**（2510.26788）：根因是 BF16 舍入误差，换 FP16 即可 | **Beyond Precision**（2602.01826）：根因是优化问题（响应变长 + 策略变尖锐），LR 调度即可 | 两者都有实验支撑，且互相点名。这是第 1 块最值得写透的一处分歧 |
| batch-invariant kernel 是否必须付性能代价 | **TML 博客** ~20% / **LLM-42** 实测降速 63% / **SGLang** 34.35% / **TBIK** 22–63% | **Bit-Exact**（2606.00279）：消除 atomics 后全速即可 bitwise 复现，定制 kernel 非必需 | 注意 B 方限定条件是「固定 SKU/版本/并行拓扑」，与 TBIK 追求的「跨 TP size 不变」不是同一目标 |

---

## 4. 确认无一手来源的缺口（写进 index.md 的 Knowledge Gaps，不要硬凑）

1. **崩溃的校准阈值**：没有任何一篇给出「δ > X 或 KL > Y ⇒ 必崩」的通用阈值。最接近的三条都是特定配置下的经验观测（FP8-RL 的 KL>5、Miles 的 TIS clip 2.0 vs 1.5、M2PO 的 τ_M2 = 0.04 且后者是超参而非实测崩溃点）。
2. **逐位置 TIM 增长曲线**：没有在 16k/32k 长 CoT 上按 token 位置画 |δ| 增长的实证。只能用 FP16 的**序列级**斜率（−1.01 vs −0.07）+ 2602.01826 的 O(T²) 理论界拼，需在页面里注明这是拼接而非单一来源。
3. **真实 mismatch 数据上的 IS 尾部统计**：多篇用「重尾」这个词，但没有一篇在 vLLM/SGLang vs Megatron/FSDP 真实数据上给出尾指数 α、ESS 绝对值或 ratio 分位数表。可用替代量化：R3 的「τ>2 极端 token 比例差一个数量级」、M2PO 的裁剪比、AIS 的 ESS ratio。
4. **重尾 → 熵坍塌的直接因果**：OPEFO 完全没提 TIM；TIM 那批也没一篇连到熵动力学。目前唯一桥梁是 M2PO Fig.4(b)「重要性比偏离越大的 token 熵越高」。**页面必须标注这是跨论文推断**。
5. **RL 阶段的 MoE 路由坍塌**：常规意义的 routing collapse（token 挤向少数专家、专精度丧失）在 RL 阶段尚未形成独立研究主题。现有最接近的是 2510.23027 的 router shift → reward collapse，语义不同。
6. **vLLM logprobs/logits 语义一致性 RFC**：roadmap #41733 链到的 #37737 实际标题是「Missing logprobs for `<tool_call>` in streaming chat completions」，已 closed as not planned。**该子条目无法核实为独立 RFC**，落库时应写「roadmap 以 #37737 为占位，尚无专门 RFC」。
7. **「降低 batch-invariant kernel 开销」的专题论文**：不存在单篇。缓解手段散落在 LLM-42、Bit-Exact、SGLang 博客、TBIK、DeepSeek-V4 §3.3 里。Thinking Machines 官方博客在该篇之后没有性能续作。

---

## 5. 原地图中未能核实 / 需修正的表述

| 原表述 | 核实结果 |
|---|---|
| 「SIS」作为独立方法 | **不是独立工作的缩写**。在 verl / slime / ms-swift 文档中一律以 Seq-TIS / Seq-MIS 形式出现，出处指向 2512.23075 与 richardli.xyz 博客 |
| 「TIS 有论文」 | **无 arXiv 版**，只有 OpenReview（NeurIPS 2025 WS / OPT 2025）+ Notion 博客 |
| 「独立的拒绝采样修 TIM 的工作」 | **不存在独立论文**。Diagnosing TIM 引的 "RS (Li et al., 2026)" 就是 2512.23075 |
| 「DeepSeek-V4 双 kernel 抵消固定 tiling/规约顺序的性能损失」 | 术语对（原文即 dual-kernel），但**抵消对象是 decoding attention 放弃 split-KV 的损失，不是 matmul 的固定 tiling 损失**。见 §1.D 备注 |
| 「TBIK 用树形规约解决跨 TP 尺寸的累加顺序不一致」 | **完全属实**，且 TBIK 就是 Tree-Based Invariant Kernels 的缩写，论文原文用 "fixed full binary-tree reduction topology" |
| 「AVSPO 及其 ACR 诊断指标」 | **完全属实**，名称、语义全部对上 |
| 「OPEFO 从 token 级熵流视角解释：熵减 token 持续压倒熵增 token」 | **完全属实**，原文即 "entropy-decreasing tokens consistently outweigh entropy-increasing ones" |
| DeepSeek-V4 §3.3 的开销数字 | 论文**未披露**任何百分比，只有定性 "negligible" |

---

## 6. 本次搜索捞出的、原地图没有的条目

这几条建议一并纳入第 1 块，它们填的正是原提纲里最薄的两环：

- **2605.14220 Diagnosing TIM** —— 全链最强的一手证据，原地图只泛指「有工作系统分析了…」
- **2510.01161 M2PO** —— 「重尾 → 崩溃」这一环唯一直接的实证 + 方法工作
- **2602.01826 Beyond Precision** —— 与 FP16 路线正面对立，原地图完全没有这条反方
- **2510.23027 RSPO** —— router shift → 重要性比波动的机理分析，比 R3/PR² 更偏「为什么」
- **2512.01374 Qwen** —— 把 IS / clipping / Routing Replay 统一到一阶近似框架下
- **2601.17768 LLM-42** + **2606.00279 Bit-Exact** —— 确定性代价的两种绕法，原地图只有「付代价」一侧
