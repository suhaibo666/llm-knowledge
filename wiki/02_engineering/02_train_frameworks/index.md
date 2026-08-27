# 训练框架 — 目录索引

> 覆盖分布式训练框架、并行策略、通信优化
> 最后更新: 2026-08-27（TorchTitan 基线刷新至 `a3168782c`；新增 Trainer quickstart、SPMD Types、FlexShard/DistMuon、GraphTrainer 四条主线）

---

## 子目录

| 目录 | 核心主题 |
|------|---------|
| [[megatron-lm/index]] | NVIDIA Megatron-LM, 5D 并行, MoE, TFLOPS, 通信掩盖;源码级系统分析 18 篇(`dev` 232c478d4, 2026-06 刷新) |
| [[torchtitan/index]] | PyTorch-native 训练框架；核心 Trainer + DP/TP/CP/EP/PP + SPMD Types + 低精度/融合/通信重叠 + FlexShard/DistMuon + GraphTrainer/GraphPP，源码级分析 16 篇内容页 + index（`main` `a3168782c`） |
| [[mindformers/index]] | 华为 MindFormers;MoE 专家并行(EP)源码级分析,PyNative 与 Graph 两条路径的 token dispatch、去冗余/零冗余/重叠与通信量(`master` 01e71622)2 篇 |
| [[mindspeed/index]] | 华为昇腾 MindSpeed × MindSpeed-LLM;猴补丁式 Megatron 加速栈,~70 个特性按并行/通信掩盖/内存优化/昇腾亲和四类 + CP 专题的机制级深挖(`master` 1432cb09)5 篇 + index |

## 页面列表

> **段位**(kb-reorg P7 Task 7,2026-07-31):子目录索引不编号;段 2(20-29)特定框架/组件的机制深挖;段 3(30-39)跨框架对比矩阵与方法论指南。
> **20 号编号空出**(2026-08-01,spec §3.4 补执行):`20_megatron_pp_parallelism_analysis.md` 已并入 `megatron-lm/15_megatron_pp_schedulers_analysis.md`(§0.4/§6/其余增量)并删除,`megatron-lm/26_megatron_pp_supplements_analysis.md` 同批一并删除;`20` 号不重新分配,详见 `wiki/changelog.md`。

| 页面 | 层次 | 来源 | 核心主题 |
|------|------|------|---------|
| [[megatron-lm/index]] | 子目录 | Megatron-LM 源码 | 分布式并行、通信优化、MoE |
| [[torchtitan/index]] | 子目录 | torchtitan 源码 | Trainer 生命周期、双平面 mesh/SPMD 布局协议、DP/TP/CP/EP/PP、低精度与通信融合、FlexShard/DistMuon、GraphTrainer/GraphPP |
| [[21_async_collective_tensor_deepdive]] | 深潜(段 2) | PyTorch 源码 (_functional_collectives.py) | ACT 源码追踪: __torch_dispatch__, wait_tensor, stream 级执行过程, 与 Megatron 手动 stream 对比 |
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

- [[../01_ai_frameworks/index]] — AI框架 (PyTorch 编译栈)
- [[../03_infer_frameworks/index]] — 推理框架
- [[../../01_theory/02_pretraining/index]] — 预训练技术
- [[../../01_theory/04_posttraining/index]] — 后训练算法
