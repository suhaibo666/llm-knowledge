---
title: "训练框架 — 目录索引"
---

# 训练框架 — 目录索引

> 覆盖分布式训练框架、并行策略、通信优化
> 最后更新: 2026-08-27（TorchTitan 基线刷新至 `a3168782c`；补齐 full config、Grain、checkpoint recovery、TorchFT、Transformers backend 与 Forge，并按新版源码分析机制重写主要页面）

---

## 模块定位：做什么 · 提供什么能力 · 边界在哪

**一句话**：训练框架回答的是**怎么把一个装不进单卡的模型，切开摊到成千上万张卡上，还能跑满算力**。它不发明并行方式（那是理论域的事），也不提供分片原语（那是 PyTorch 的事）——它的工作是**把多个并行维度组合起来，并把由此产生的通信藏进计算的空隙里**。

**为什么必须独立成一层**：单看任何一个并行维度都不难，难的是它们**互相耦合**：

- TP 大了单步通信量涨、PP 大了气泡涨、EP 大了 all-to-all 涨、CP 大了 attention 的环形通信涨——四者在固定卡数下此消彼长；
- 每一维都会改变**显存账本**（参数/梯度/优化器状态/激活各自被谁分片），而 OOM 是硬约束，不是性能问题；
- 通信一旦不能与计算重叠，前面所有切分收益都会被吐回去，所以**通信掩盖不是优化项而是设计前提**。

这三件事没法在 `torch.distributed` 原语层解决（那层只提供 DTensor/FSDP/PP 的**语义**），也没法留给用户在训练脚本里手写。于是有了这一层：**并行策略的组合、通信与计算的编排、以及万卡尺度下的 checkpoint 与恢复**。

### 本域覆盖的框架与各自定位

本域是本库**产品最多**的域，四个框架分属两个硬件生态、三条技术路线，覆盖度不均衡，如实标注：

| 框架 | 在本域中的定位 | 本库覆盖 | 基线 |
|---|---|---|---|
| **Megatron-LM** | NVIDIA 出品，**手工并行的事实标准**；5D 并行、MoE、通信掩盖、分布式优化器的原始参考实现 | 33 篇 + index（**系统性源码覆盖 + 代码仓功能树对账**） | `NVIDIA/Megatron-LM@85902ef59`（`dev`，2026-09-01） |
| **TorchTitan** | **PyTorch-native 路线**：用 full configuration、DeviceMesh/SPMD Types、FSDP2 与编译器实验组合训练系统；本库另一条系统性覆盖主线 | 23 篇 + index（**系统性源码覆盖**） | `pytorch/torchtitan@a3168782c`（`main`，commit date 2026-08-26） |
| **MindSpeed / MindSpeed-LLM** | 华为昇腾侧：**猴补丁式**加速栈，在 Megatron 之上叠 ~70 个特性；代表"在既有框架上做厂商适配"这条路 | 5 篇 + index（按并行/通信掩盖/内存/昇腾亲和四类 + CP 专题的**机制级深挖，非全量特性走查**） | MindSpeed `master@1432cb09`（基于 Megatron `core_r0.17.0`）· MindSpeed-LLM `master@0c16322d` |
| **MindFormers** | 华为昇腾侧的另一条路：**MindSpore 生态**，与前三者不共享 PyTorch 底座 | 2 篇 + index（**单点切入**：仅 MoE 专家并行的 PyNative/Graph 两条 token dispatch 路径） | `master@01e71622`（2026-06-18） |
| **跨框架专题** | **不属于任何单一框架**的机制与对比：异步集合张量、Muon 分片、通信计算重叠与融合、分布式优化器、快恢对比 | 6 篇 | 见各页头（多数标注多仓基线） |

> 三点引用注意：① **Megatron 与 TorchTitan 是仅有的两条系统性覆盖主线**，跨框架结论应以这两者为主证据；② **MindSpeed 是机制级深挖而非特性全量走查**，"MindSpeed 支持 X 吗"这类问题本库答不全；③ **MindFormers 只覆盖了 MoE EP 一个切面**，不足以支撑对 MindSpore 训练栈的整体判断——[[33_fault_recovery_relink_comparison]] 里它的容错部分明确标了"闭源边界，本仓不可见"。

### 本域提供的能力

下表按**能力**组织；"样本与锚点"列说明本库是拿谁的实现讲的。源码锚点按侧车 checkout 核对路径存在：

| 能力 | 具体提供什么 | 样本与源码锚点 | 详见 |
|---|---|---|---|
| **多维并行的组合** | TP/PP/CP/EP/DP 各维如何接线、如何共存、切分顺序对通信量的影响 | Megatron `megatron/core/transformer/` · TorchTitan `torchtitan/distributed/` | [[megatron-lm/index\|Megatron-LM]] · [[torchtitan/index\|TorchTitan]] |
| **声明式并行表达** | 用 DTensor/DeviceMesh 描述布局，而非手写 all-gather/reduce-scatter | TorchTitan（SPMD Types、storage/compute layout 分离） | [[torchtitan/index\|TorchTitan]] |
| **数据并行与分片** | ZeRO/FSDP/HSDP 的分片粒度、prefetch 时机与显存-通信取舍 | Megatron `megatron/core/distributed/` · TorchTitan FSDP/SimpleFSDP | [[32_distributed_optimizer_deepdive]] |
| **流水并行调度** | 微批次编排、1F1B 与 zero-bubble 变体、stage 适配层 | Megatron PP schedulers · TorchTitan PP | [[megatron-lm/index\|Megatron-LM]] |
| **MoE 的专家并行** | token dispatch、all-to-all、去冗余与重叠、通信量账本 | Megatron MoE · MindSpeed CP/EP · MindFormers（MindSpore 侧对照） | [[mindformers/index\|MindFormers]] |
| **通信与计算重叠** | 把集合通信藏进计算空隙的手段，及其正确性前提 | 跨框架专题 | [[30_comm_compute_overlap_analysis]] · [[21_async_collective_tensor_deepdive]] |
| **通算融合** | 比"重叠"更进一步：把通信下沉进 kernel/图 | 跨框架专题 | [[31_comm_compute_fusion_guide]] |
| **昇腾亲和优化** | NPU 上与 GPU 不同的内存/通信/算子取舍 | MindSpeed（猴补丁式接入） | [[mindspeed/index\|MindSpeed]] |
| **故障恢复与重新建链** | 节点更换后如何重建通信域：进程内重启 vs 原地重链 vs 委托 runtime | Megatron+NVRx · MindSpeed+MindIO · MindFormers→MindSpore（**含闭源边界标注**） | [[33_fault_recovery_relink_comparison]] |

### 不属于本模块的

- 各并行维度**本身的原理**（TP 为什么这么切、ZeRO 三阶段各省什么、α-β 通信代价模型）→ [[01_theory/06_distributed_parallelism/index|分布式并行原理]]；本域讲工程接线，那里讲为什么成立；
- `DTensor` / `DeviceMesh` / `FSDP2` / `pipelining` 这些**原语本身**的语义 → [[02_engineering/01_pytorch/index|PyTorch]]；本域是它们的**消费方**；
- **自动**求解并行策略 → [[02_engineering/06_auto_parallel/index|自动并行]]；本域的策略是人配的，那里研究怎么让机器配；
- 万卡尺度的**确定性、SDC、loss spike** → [[02_engineering/07_training_reliability/index|训练可靠性]]；本域只覆盖 checkpoint 与快恢的框架侧实现；
- RL 后训练的 rollout 编排 → [[02_engineering/04_posttrain_frameworks/index|后训练框架]]，它把本域的训练引擎接进 RL 循环。

### 与兄弟域的关系

本域向下**完全建立在** `01_pytorch` 的分布式原语之上（TorchTitan 尤其明显——它几乎不自建抽象）；向上是 `04_posttrain_frameworks` 的训练侧引擎；与 `06_auto_parallel` 互为对照组（手工 vs 自动）和执行后端；与 `07_training_reliability` 的分界是"跑得快"与"跑得完/跑得准"。

---

## 子目录

| 目录 | 核心主题 |
|------|---------|
| [[megatron-lm/index]] | NVIDIA Megatron-LM, 5D 并行, MoE, TFLOPS, 通信掩盖;源码级系统分析 + 段 4 代码仓功能树(600 个 `.py` 双向对账) 共 33 篇内容页 + index(`dev` 85902ef59, 2026-09-01 全域重定基线) |
| [[torchtitan/index]] | PyTorch-native 训练框架；full config、Grain/checkpoint/multimodal contract、核心 Trainer + DP/TP/CP/EP/PP + SPMD Types + 低精度/LoRA/structured trace + FlexShard/GraphTrainer + TorchFT/HF backend/Forge，源码级分析 23 篇内容页 + index（`main` `a3168782c`） |
| [[mindformers/index]] | 华为 MindFormers;MoE 专家并行(EP)源码级分析,PyNative 与 Graph 两条路径的 token dispatch、去冗余/零冗余/重叠与通信量(`master` 01e71622)2 篇内容页 + index |
| [[mindspeed/index]] | 华为昇腾 MindSpeed × MindSpeed-LLM;猴补丁式 Megatron 加速栈,~70 个特性按并行/通信掩盖/内存优化/昇腾亲和四类 + CP 专题的机制级深挖(`master` 1432cb09)5 篇内容页 + index |

## 页面列表

> **段位**(kb-reorg P7 Task 7,2026-07-31):子目录索引不编号;段 2(20-29)特定框架/组件的机制深挖;段 3(30-39)跨框架对比矩阵与方法论指南。
> **20 号编号空出**(2026-08-01,spec §3.4 补执行):`20_megatron_pp_parallelism_analysis.md` 已并入 `megatron-lm/15_megatron_pp_schedulers_analysis.md`(§1.5/§8/其余增量)并删除,`megatron-lm/26_megatron_pp_supplements_analysis.md` 同批一并删除;`20` 号不重新分配,详见 `wiki/changelog.md`。

| 页面 | 层次 | 来源 | 核心主题 |
|------|------|------|---------|
| [[megatron-lm/index]] | 子目录 | Megatron-LM 源码 | 分布式并行、通信优化、MoE |
| [[torchtitan/index]] | 子目录 | torchtitan 源码 | Trainer 生命周期与观测、双平面 mesh/SPMD 布局协议、DP/TP/CP/EP/PP、低精度/LoRA 与通信融合、FlexShard/DistMuon、GraphTrainer/GraphPP |
| [[21_async_collective_tensor_deepdive]] | 深潜(段 2) | PyTorch 源码 (_functional_collectives.py) | ACT 三层机制（wrapper subclass / __torch_dispatch__ 分流 / wait_tensor→WorkRegistry→stream block）、掩盖成立的三个前提，以及 torchtitan 默认后端下该窗口为何已不存在 |
| [[22_muon_sharded_hsdp_analysis]] | 深潜(段 2) | Cursor Composer 2.5 博客 | 分片 Muon + 双网格 HSDP: all-to-all N-S、EP/CP 解耦、异步流水线、非专家分工优化 |
| [[30_comm_compute_overlap_analysis]] | 方法论(段 3) | Megatron-LM / torchtitan 源码 | 计算通信掩盖: combined_1f1b vs ZBV/DualPipe, sub-layer 级调度, DeepEP/HybridEP |
| [[31_comm_compute_fusion_guide]] | 方法论(段 3) | 综合深度分析 | 通算融合: WaveEP、DeepEP、TP/DP/PP/CP 各维度重叠, 自动化路线图 |
| [[32_distributed_optimizer_deepdive]] | 方法论(段 3) | 综合深度分析 | FSDP2/ZeRO/MindSpeed 对比, 梯度累积, Adam vs Muon |
| [[33_fault_recovery_relink_comparison]] | 方法论(段 3) | Megatron/MindSpeed/MindFormers + torch_npu 源码 | 跨框架快恢与「重新建链」对比: Megatron NVRx 进程内重启(abort NCCL→destroy→PrefixStore 重 init)、MindSpeed MindIO ARF 空中加油(`reinit_process_group(rebuild_link=True)`→`abort_hccl_comm` 原地重建 + replica 拷态)、MindFormers 委托 MindSpore runtime;含闭源边界标注 |

> MindFormers MoE 专家并行(PyNative + Graph 两路径,共 2 篇)已收入子目录 [[mindformers/index]]。

---

## 原始素材

`raw/02_engineering/02_train_frameworks/`:

| 文件 | 主题 |
|------|------|
| `megatron.eddx` | Megatron 训练框架架构图 |
| `mindformers.eddx` | MindFormers 训练框架架构图 |

---

## 关联域

- [[../01_pytorch/index]] — AI框架 (PyTorch 编译栈)
- [[../03_infer_frameworks/index]] — 推理框架
- [[../../01_theory/02_pretraining/index]] — 预训练技术
- [[../../01_theory/04_posttraining/index]] — 后训练算法
