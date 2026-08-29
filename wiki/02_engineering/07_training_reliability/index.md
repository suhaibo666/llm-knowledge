---
title: "万卡级训练：确定性与可靠性问题域 — 目录索引"
---

# 万卡级训练：确定性与可靠性问题域 — 目录索引

> 覆盖「万卡级 LLM 训练」中**确定性 / 数值可靠性 / 故障容错 / 训练动力学稳定性**四大问题域，以**问题为纲**（背景→影响→如何发现→解决方案→代码实现）逐个讲透机理。
> 最后更新: 2026-07-31(kb-reorg P7 Task 7:目录内分段编号——段 1(10-12)=下方「三张内容页」,按问题 1-4→5-8→9 顺序;段 2(20)=第四篇 batch 不变性算子实现,是问题 2 的算子级细化专题)

---

## 模块定位：做什么 · 提供什么能力 · 边界在哪

**一句话**：本域研究的是**万卡规模下"训练能不能跑完、跑出来的数字能不能信"**——它不是性能问题，是**可用性与正确性**问题。

**为什么必须独立成一层**：这些问题在单卡和百卡上要么不存在、要么可以忽略；到万卡规模全部变成一等工程问题：

- 硬件 MTBF 摊到万卡后掉到**小时级**，"跑完一次训练"本身成了需要设计的能力；
- 浮点加法不满足结合律 + 并行归约顺序不固定 ⇒ **重跑一遍对不上**成为常态，于是无法用"重放比对"区分随机噪声与坏硬件；
- 低精度（FP8/FP4）把长链累加的误差裕量压薄，同时也压薄了 **SDC 单比特翻转**的检测窗口；
- loss spike / NaN 在万亿 token 规模下几乎必然出现，回滚一次的成本以天计。

关键在于：**这些问题不属于任何一个框架**——Megatron、torchtitan、MindSpeed 都会遇到，而解法往往落在框架之外（硬件 RAS、网络工程、运维体系）。所以本域按**问题**组织，不按框架组织。

### 本域的来源构成与证据分级

按问题组织的代价是：单页里常常并列多家的做法，而它们的证据强度差别很大。如实标注：

| 来源类别 | 具体是谁 | 证据强度 | 在本域中承担什么 |
|---|---|---|---|
| **本地可核验源码** | `NVIDIA/Megatron-LM@232c478d4`（`--deterministic-mode`、`dist_checkpointing/`）、`torch_npu` | **可按 `file:line` 核验** | 确定性开关与 checkpoint 体系的具体落点 |
| **厂商技术报告 / 论文** | Gemini、Llama 3、MegaScale(NSDI'24)、ByteRobust(SOSP'25)、Aegis(NSDI'25)、C4(HPCA'25)、DeepSeek-V3(ISCA'25) | 一手，但**只披露作者选择披露的部分** | 故障统计、goodput/ETTR 口径、恢复链路的工业实证 |
| **工程博客 / 公开分析** | Thinking Machines「Defeating Nondeterminism」、LongCat-2.0 博客、华为 CloudMatrix、Anthropic 事故 postmortem | 一手但非同行评议 | batch 不变性、确定性算子、链路切流、真实事故复盘 |
| **二手综述稿** | `docs/research/wanka_determinism_reliability_deep_analysis.md`（用户提供，2026-07 摄入） | **二手**——本簇三篇内容页的骨架来自它 | 把上述来源编织成 9 个问题域的坐标系 |
| **本库一手页交叉** | [[longcat_2_analysis]]、[[13_low_precision_training_analysis]]、[[20_batch_invariance_guide]] 等 | 见各页 | 与本域结论互为印证或补充细节 |

> 两点必须说清：① **本簇三篇主内容页是对一份二手综述的结构化摄入**，机制/数字/命令忠实于原文，但不等于逐条读过原始论文；② **公开信息本身有样本偏差**——Google/Meta/字节/阿里/华为/美团有成体系披露，OpenAI 与 Anthropic 训练侧基本不发表，所以 spike/NaN 治理的前沿实证几乎全部来自中国实验室的开源报告 + Google 的间接披露。下结论时不要把"公开得多"读成"做得好"。

### 本域提供的能力


| 能力 | 具体提供什么 | 锚点 | 详见 |
|---|---|---|---|
| **问题坐标系** | 9 个问题域 × 三条主线（确定性 / 容错 / 训练动力学）的归类，及它们之间的依赖关系 | — | 下方「问题地图」 |
| **判据与度量** | goodput / ETTR 的定义与五级恢复坐标系（Job/Pod/Node/进程/Step）；"确定性税"的量级（5%–35%，取决于层次） | — | [[11_fault_tolerance_and_recovery_analysis]] |
| **确定性的获取手段** | 确定性算子、二叉树/pairwise 分段累加、固定顺序归约、框架级开关 | `NVIDIA/Megatron-LM@232c478d4`：`megatron/training/arguments.py:1786`（`--deterministic-mode`） | [[10_determinism_and_numerical_reliability_analysis]] |
| **batch 不变性的算子级实现** | 双内核 attention（单 SM 一序列 vs 多 SM 协作+固定顺序归约）、DeepGEMM 1D1D 布局替代 cuBLAS split-k、MoE 反向 per-SM 独立缓冲 | DeepSeek V4 报告 §3.3 + DeepGEMM 源码 | [[20_batch_invariance_guide]] |
| **故障的发现与定界** | NCCL Flight Recorder、栈聚类识别 hang、straggler 打分、SDC 的四层检测（含"重放不复现 ⇒ 坏硬件"这条定界法） | — | [[11_fault_tolerance_and_recovery_analysis]] |
| **恢复链路** | 异步分布式 checkpoint、本地 ckpt、临终保存、进程内重启、弹性缩容 | `Megatron-LM@232c478d4`：`megatron/core/dist_checkpointing/` | 同上 |
| **网络侧的可靠性** | PFC 风暴、ECMP hash 冲突、链路级快恢与无感切流、流量工程 | — | 同上 |
| **训练动力学的稳定化** | loss spike/NaN 的四类根因、前兆指标与排查决策树、四层防线（架构 QK-Norm/z-loss、优化器 MuonClip/自适应 clip、数据、运维） | — | [[12_training_dynamics_stability_analysis]] |

### 不属于本模块的

- 怎么把训练**跑得更快**（并行策略、通信掩盖、算子融合）→ [[02_engineering/02_train_frameworks/index|训练框架]] 与 [[02_engineering/05_gpu_kernel/index|GPU Kernel]]；本域只在"快"与"可信/可恢复"冲突时给出取舍依据；
- 具体框架的容错**实现走查**（Megatron NVRx / MindSpeed MindIO ARF / MindFormers 委托 MindSpore）→ [[02_engineering/02_train_frameworks/33_fault_recovery_relink_comparison|跨框架快恢对比]]；本域讲问题与判据，那页讲某三家怎么接线；
- 训推一致性在 **RL 训练中的算法后果**（importance ratio 失真如何影响梯度）→ [[01_theory/04_posttraining/index|后训练理论]] 与 [[02_engineering/04_posttrain_frameworks/index|后训练框架]]；本域只负责"两侧必须逐位对齐"这个数值前提。

### 与兄弟域的关系
本域是横切层——它对 `01_pytorch`（归约顺序、低精度累加）、`02_train_frameworks`（checkpoint 与恢复）、`04_posttrain_frameworks`（训推一致）、`05_gpu_kernel`（确定性算子的代价）**同时提出约束**，但不拥有其中任何一处实现。读本域是为了知道"该向那些域要什么保证"。

---

## 来源与定位

- **本簇原始素材**：一份多来源综述深度分析文档 `docs/research/wanka_determinism_reliability_deep_analysis.md`（用户提供，2026-07 摄入）。它综合了 **Gemini 1.0/2.5、Llama 3、ByteRobust(SOSP'25)、MegaScale(NSDI'24)、Aegis(NSDI'25)、C4(HPCA'25)、DeepSeek-V3(ISCA'25)、Thinking Machines「Defeating Nondeterminism」、Anthropic 事故 postmortem、华为 CloudMatrix、美团 LongCat-2.0 博客**，以及 Megatron-LM / NVRx / torch_npu 等代码库的公开实现。
- **保真度说明**：本簇是对上述**二手综述**的结构化摄入——机制、数字、命令、代码均**忠实于原文**；原文引用的一手来源见其[附录](#附主要一手来源)。与本库已有一手页（LongCat / 低精度 / RL 精度）冲突或互补处已交叉链接。
- **为什么摄入到这里**：由「LongCat 的稳定性训练」延伸——LongCat（确定性算子、二叉树分段累加、bit-flip 检测、链路无感切流）是全篇的一条运行主线，但问题域本身是引擎无关的通用工程知识，故独立成簇，与 [[longcat_2_analysis]]/[[longcat_flash_analysis]] 互链。

---

## 问题地图（9 个问题 × 两条主线）

| # | 问题 | 主线 | 代表性工作 | 详见 |
|---|------|------|-----------|------|
| 1 | 训练比特级不可复现（浮点非确定性） | 确定性 | Google 全确定性栈、LongCat 确定性算子、Megatron `--deterministic-mode` | [[10_determinism_and_numerical_reliability_analysis]] |
| 2 | 训推数值不一致 / batch 不变性 | 确定性 | Thinking Machines、SGLang/vLLM 确定性推理、Anthropic top-k 事故 | [[10_determinism_and_numerical_reliability_analysis]] · [[20_batch_invariance_guide]] |
| 3 | 低精度长链累加误差 | 数值可靠性 | DeepSeek FP8 两级累加、LongCat 二叉树规约、Megatron FP32 main_grad | [[10_determinism_and_numerical_reliability_analysis]] |
| 4 | 静默数据损坏 SDC / 比特翻转 | 数值可靠性 | Gemini 确定性重放+checksum、ABFT、LongCat 比特翻转检测 | [[10_determinism_and_numerical_reliability_analysis]] |
| 5 | 显式故障高频化与恢复链路开销 | 容错 | Llama 3 故障统计、NVRx in-process restart、Gemini slice 弹性 | [[11_fault_tolerance_and_recovery_analysis]] |
| 6 | 隐式故障：hang / 慢节点 / 性能劣化 | 容错 | ByteRobust 栈聚类、NCCL Flight Recorder、NVRx straggler | [[11_fault_tolerance_and_recovery_analysis]] |
| 7 | Checkpoint 保存开销与恢复窗口 | 容错 | Megatron async dist-ckpt、NVRx local ckpt、MindIO 临终保存 | [[11_fault_tolerance_and_recovery_analysis]] |
| 8 | 网络链路故障的识别、切流与定界 | 容错 | LongCat 无感切流、华为 1-3-10、Alibaba C4/Aegis | [[11_fault_tolerance_and_recovery_analysis]] |
| 9 | 训练动力学：loss spike / NaN / 发散 | 训练动力学 | PaLM 回滚跳批、GLM-130B EGS、Kimi K2 MuonClip、DeepSeek-V3 零 spike | [[12_training_dynamics_stability_analysis]] |

---

## 一条贯穿全篇的主线：确定性是地基

两条主线并非独立：**确定性是故障定界的地基**——没有 bitwise 可复现，就无法用「重放 + 比对」区分随机噪声与硬件故障（Gemini 报告与 LongCat 博客共同的底层逻辑）。它同时是三件事的前提：

```
                    ┌─ SDC 重放定界（问题 4 第四层）：重放不复现 → 坏硬件
全栈确定性（问题 1） ─┼─ 训推一致的 RL（问题 2）：log-prob 逐位一致 → importance ratio=1
                    ├─ 跨硬件精度验收（问题 3）：与高精度基线逐位对照 → 误差可归因
                    └─ spike 鉴别（问题 9）：重放复现 → 优化/数据问题；不复现 → SDC
```

## 三张内容页

- **[[10_determinism_and_numerical_reliability_analysis]]** — 第一部分（问题 1-4）：浮点非确定性五层来源、batch 不变性与 RL 确定性、低精度长链累加（树形/pairwise、FP32 main_grad、DeepSeek FP8 两级累加、Kahan）、SDC/比特翻转四层检测体系。
- **[[11_fault_tolerance_and_recovery_analysis]]** — 第二部分（问题 5-8）：goodput/ETTR 与五级恢复坐标系（Job/Pod/Node/进程/Step）、hang/straggler 的发现与定界（flight recorder / 栈聚类 / straggler 打分）、Checkpoint 体系（异步/本地/临终/数据回放）、网络链路故障（PFC 风暴、ECMP hash、链路级快恢、流量工程）。
- **[[12_training_dynamics_stability_analysis]]** — 第三部分（问题 9）：loss spike/NaN 的四类根因、分层监控+前兆指标、排查决策树、四层防线（架构 QK-Norm/z-loss、优化器 MuonClip/自适应 clip、数据、运维），及 2026 前沿一代（Muon 路线、DeepSeek-V4 Anticipatory Routing、Kimi K2.5 / GLM-5 RL 稳定性）。

## 第四篇：batch 不变性算子实现（kb-reorg P5 归位）

- **[[20_batch_invariance_guide]]**（2026-07-31 从 `04_posttrain_frameworks/` 迁入）— 源自 DeepSeek V4 报告 §3.3 + DeepGEMM 源码分析，独立于上述 wanka 综述素材，与 `10_determinism_and_numerical_reliability_analysis` 问题 2 互为算子级实现细化：双内核 Attention（单 SM 一条序列 vs 多 SM 协作 + 固定顺序归约）、DeepGEMM 1D1D 布局替代 cuBLAS split-k、MoE 反向的 per-SM 独立缓冲区 + 确定性全局求和。

---

## 趋势与开放问题（原文第四部分）

1. **确定性从调试工具升格为体系能力**：同时是 SDC 重放定界、训推一致 RL、跨硬件精度验收三者的地基。「确定性税」（当前 5%–35%，取决于层次）随专用 kernel 优化下降，运维杠杆在放大。
2. **恢复粒度持续细化，度量收敛到 goodput/ETTR**：任务级重启 → 弹性缩容（Gemini 97% 吞吐）→ 进程内重启（NVRx 秒级）→ step 级确定性重放（Gemini 0.25% 重放率）。
3. **硬件-软件责任边界在重新谈判**：软件用 ABFT / 二叉树规约 / DP hash 校验补硬件的课；DeepSeek 呼吁把 checksum 做进硬件；华为 RAS 是国产侧体系化回应。FP4 压缩误差裕量后，纯软件检测窗口更窄。
4. **尚无公认解法的三题**：① 超节点故障域（NVL72 / CloudMatrix 384 机架级爆炸半径）；② 大 EP 的容错语义（专家状态无天然副本，弹性缩容与负载均衡交互无成熟方案）；③ 低精度与 SDC 检测的结构性冲突（FP4 量化噪声可能淹没单比特翻转信号）。
5. **公开信息不对称**：Google/Meta/字节/阿里/华为/美团有成体系披露；OpenAI 与 Anthropic 训练侧基本不发表（样本偏差——spike/NaN 治理的前沿实证几乎全部来自中国实验室开源报告 + Google 间接披露）。

---

## 与本库已有页的交叉

| 本簇问题 | 关联的一手/深挖页 |
|---------|------------------|
| 问题 1（确定性）· 问题 3-4 | [[longcat_2_analysis]] §6-7（确定性算子/二叉树累加/bit-flip）· [[longcat_flash_analysis]] §三（SDC 检测） |
| 问题 2（训推一致） | [[20_rl_training_inference_precision_analysis]] · [[10_rl_ppo_loss_and_grpo_analysis]] · [[20_batch_invariance_guide]] |
| 问题 3（低精度） | [[13_low_precision_training_analysis]] · [[12_deepseek_v3_analysis]]（FP8 两级累加/DeepGEMM） |
| 问题 1 第 3 层（通信规约树） | [[10_collectives_analysis]]（ring/tree allreduce）· [[14_expert_parallel_analysis]]（MoE all-to-all） |
| 问题 9（spike / Muon 系） | [[11_muon_analysis]] · [[11_kimi_k2_analysis]]（MuonClip）· [[13_deepseek_v4_analysis]] · [[01_glm_5_analysis]] |
| 容错 / 分布式框架 | [[02_engineering/02_train_frameworks/megatron-lm/index]] |

## 附：主要一手来源

见 raw 文档末节「主要参考」：Gemini（SDC/确定性重放/热备）、Llama 3（54 天故障统计/flight recorder）、ByteRobust(SOSP'25)、MegaScale(NSDI'24)、Aegis(NSDI'25)、C4(HPCA'25)、DeepSeek-V3(ISCA'25)、Thinking Machines「Defeating Nondeterminism」、Anthropic postmortem、华为 CloudMatrix、LongCat-2.0 博客、SDC 研究（arXiv 2604.00726 / 2604.10390 / ACL 2025）、训练动力学（PaLM 2204.02311 / GLM-130B 2210.02414 / ViT-22B 2302.05442 / Molybog 2304.09871 / Spike No More 2312.16903 / ST-MoE 2202.08906 / AdaGC 2502.11034 / ZClip 2504.02507 / Kimi K2 2507.20534 / DeepSeek-V4 2606.19348 等）、NVRx / Megatron-LM / PyTorch 文档。

---

## Related Pages

- [[02_engineering/index]] — 工程实现总索引
- [[10_determinism_and_numerical_reliability_analysis]] · [[11_fault_tolerance_and_recovery_analysis]] · [[12_training_dynamics_stability_analysis]] — 本簇三页
- [[longcat_2_analysis]] · [[longcat_flash_analysis]] — 引出本簇的 LongCat 稳定性主线
- [[01_theory/06_distributed_parallelism/index]] — 分布式并行原理（通信原语/EP 与本簇的规约树/容错语义相接）
