# Megatron-LM — Knowledge Map

This domain covers NVIDIA Megatron-LM distributed training framework, including parallelism strategies, MoE implementation, performance measurement, and integration with inference engines.

> 最后更新:2026-07-31(kb-reorg P7 Task 7:目录内分段编号,27 篇按 spec §5 段位约定统一加两位前缀;下方各小节的既有分组不变,仅补充下表作段位速查)

## 段位速查(kb-reorg P7 Task 7)

> 段 0(01)导览/capstone;段 1(10-19)16 篇源码级系统分析系列中最贴近核心流水线的 10 篇——模型结构→数据→TP/CP/EP/PP 四并行轴→分布式优化器→编排→重计算→存档;段 2(20-29)系列剩余的深挖/补遗/系统专题共 9 篇,以及 Core Topics 的专题深挖(除 audit 外);段 3(30-35)RL/推理集成、度量方法论与两篇 DeepSeek-V4 案例研究。与下文按主题分组的表格是同一组页面的两种视图。
> **26 号编号空出**(2026-08-01,spec §3.4 补执行):`26_megatron_pp_supplements_analysis.md` 已并入 [[15_megatron_pp_schedulers_analysis]](§0.4 进程组拓扑、§6 混合 CP 动态调度/多模块流水线、§0.3/⑤.6/②.2 等增量)并删除;父目录 `20_megatron_pp_parallelism_analysis.md` 同批一并删除。`26` 号不重新分配,详见 `wiki/changelog.md`。

| 段 | 编号 | 页面 |
|---|---|---|
| 0 | 01 | [[01_megatron_moe_training_optimization_analysis]] |
| 1 | 10-19 | [[10_megatron_model_structure_analysis]] · [[11_megatron_dataset_analysis]] · [[12_megatron_tp_analysis]] · [[13_megatron_cp_analysis]] · [[14_megatron_ep_analysis]] · [[15_megatron_pp_schedulers_analysis]] · [[16_megatron_distributed_optimizer_analysis]] · [[17_megatron_parallelism_orchestration_analysis]] · [[18_megatron_recompute_analysis]] · [[19_megatron_dist_checkpointing_analysis]] |
| 2 | 20-25,27-29(26 空出) | [[20_megatron_comm_overlap_analysis]] · [[21_megatron_fusion_operators_analysis]] · [[22_megatron_memory_optimization_analysis]] · [[23_megatron_precision_cudagraph_fusion_analysis]] · [[24_megatron_linear_cross_entropy_analysis]] · [[25_megatron_nonuniform_tp_analysis]] · [[27_megatron_tp_fsdp_resharding_supplements_analysis]] · [[28_megatron_training_stability_observability_analysis]] · [[29_megatron_packed_dataset_dynamic_cp_analysis]] |
| 3 | 30-35 | [[30_megatron_rl_posttraining_consistency_analysis]] · [[31_megatron_inference_engine_analysis]] · [[32_megatron_tflops_analysis]] · [[33_megatron_vllm_weight_sync_analysis]] · [[34_deepseek_v4_tensor_parallel_analysis]] · [[35_deepseek_v4_context_parallel_analysis]] |

## Core Topics（系列外的全景报告与专题深挖）

> 本节列出 16 篇「源码级系统分析系列」(见下文)**之外**的全景报告与跨切面专题深挖。并行轴(TP/PP/EP/CP/DP)、重计算、训练稳定性、RL、模型结构等逐维度深挖请直接看下文的系列。
>
> 命名约定:本目录所有页统一为 `megatron_<topic>_analysis`(小写 snake_case,对齐 torchtitan 的 `torchtitan_<topic>_analysis` 风格)。

### 全景报告

| Page | Key Concepts |
|------|-------------|
| [[01_megatron_moe_training_optimization_analysis]] | 7 维 MoE 训练优化全景(TP/PP/EP/CP、分布式优化器、重计算、低精度 FP8/FP4、通信 Overlap、显存、融合算子);跨 ~13 篇深挖的导航 capstone,覆盖 50B → 1.xT MoE 规模 |

### 专题深挖(系列外)

| Page | Key Concepts |
|------|-------------|
| [[16_megatron_distributed_optimizer_analysis]] | **合一页**(2026-07-31 由原「DDP/ZeRO 分片」+「优化器内部」+ 本页三篇合并):DP/DDP/ZeRO 0-3 四阶段(§0-§5)、优化器类层次与 step 内部(fp32 master/loss scaling/梯度裁剪/LR 调度,§6-§11)、CPU offload(§12)、三种 FSDP 实现对比(§13)、Layer-Wise/Muon 集成(§14) |
| [[../32_distributed_optimizer_deepdive\|32_distributed_optimizer_deepdive]] | (跨框架,父目录)FSDP2/ZeRO/MindSpeed 三方对比, 梯度累积通信量 (K×P), Adam vs Muon 内存估算 (18→14 bytes/param) |
| [[22_megatron_memory_optimization_analysis]] | 显存手段全survey:NCCL memory pool、MoE Paged Stash、细粒度激活 offload、param/grad buffer 复用(MXFP8/NVFP4)、FP8/FP4 参数精度、CUDA graph buffer 复用、resharding(与 [[18_megatron_recompute_analysis]] 互补) |
| [[21_megatron_fusion_operators_analysis]] | 融合算子全目录:Bias+激活(GEGLU/SwiGLU/GELU)、fused LayerNorm/Softmax、MoE 融合、fused Cross-Entropy、fused All-to-All(DeepEP/HybridEP)、FP8 input store、Triton/CUTLASS/cuTile —— 系列的 [[23_megatron_precision_cudagraph_fusion_analysis]] §3 是其精简版 |
| [[24_megatron_linear_cross_entropy_analysis]] | **融合线性交叉熵("chunk loss")源码级深挖**:`cross_entropy_fusion_impl='linear'` 把 LM-head matmul 融进 CE 核、logits 从不物化(`save_for_backward` 只存 max+sum-exp、反向按 vocab 块重算);Blackwell-only CuTe 核 + TP `all_reduce(MAX/SUM)`;对照 `native`/`te`(仍物化 logits)与 MindSpeed 序列分块 `chunk_loss` |
| [[20_megatron_comm_overlap_analysis]] | 6 维通信-计算重叠综合(TP/DP/PP/EP/CP)、bulk & pipelined overlap、delay-wgrad、DeepEP/HybridEP;TP overlap 深挖 + 跨维收益/配置速查表为其独有 |
| [[25_megatron_nonuniform_tp_analysis]] | 混合尺寸 TP 组容错(spare→core→extra 梯度 all-to-all 重共享)、进程组重配、冷重启 —— 系列的 [[27_megatron_tp_fsdp_resharding_supplements_analysis]] §2 是其精简版 |
| [[32_megatron_tflops_analysis]] | 理论 FLOPS 估算、前/反向 FLOP 计数、MoE dropless vs droptoken、吞吐公式 |
| [[33_megatron_vllm_weight_sync_analysis]] | verl 框架 Megatron→vLLM 权重同步、Gather-Broadcast-Load、colocation、HF 格式重组(与 [[30_megatron_rl_posttraining_consistency_analysis]] 的内部 Refit 互补) |
| [[mooncake_analysis]] | (跨域,推理框架目录)Mooncake KVCache 中心化分离式服务架构 |
| [[34_deepseek_v4_tensor_parallel_analysis]] | **DeepSeek-V4 TP 切分实现**:DSv4 Hybrid Attention 强制 `tp==1` 的架构动因、Compressor/Indexer duplicated、mHC 非 TP-aware 梯度同步、MoE Shared/Routed expert TP 约束、通信量修正(2026-06-25 自父目录移入)。模型侧架构见 [[../../../01_theory/01_models/deepseek/13_deepseek_v4_analysis\|13_deepseek_v4_analysis]] |
| [[35_deepseek_v4_context_parallel_analysis]] | **DeepSeek-V4 CP 实现**:MLA 对 CP 通信量降低 ~128 倍、CSA/HCA 压缩注意力与 CP 的论文↔代码 gap 审计、RoPE 的 CP 感知、TE CP 的 cp_stream 双缓冲、Dynamic CP 对 MLA 的不支持(2026-06-25 自父目录移入)。CP 通用机制见 [[../../../01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis\|20_ring_attention_and_context_parallel_analysis]];论文级 CP 算法见 [[../../../01_theory/01_models/deepseek/23_deepseek_v4_cp_analysis\|23_deepseek_v4_cp_analysis]] |

## 源码级系统分析系列(Megatron-LM `dev` @ `232c478d4`, 2026-06 刷新)

> 一套 16 篇源码级系统分析,基于 Megatron-LM `dev` 分支,逐层覆盖并行体系、性能基建、训练稳定性、数据、推理与 RL、模型结构与存档。各篇互为 `[[wiki link]]` 交叉引用,自成体系。
>
> **基线**:初版基于 commit `ee3f1ff`(2026-05-19);**2026-06-16 已对照 `dev@232c478d4` 全量刷新**,逐页核实并增补 `ee3f1ff..232c478d4`(298 commits)的增量(每处增补带 `> [!update] 2026-06-16` 标注 + `path:line` + `(#PR)`)。

> [!note] 2026-06-16 · 去重与命名整合
> - **命名统一**:本目录所有页重命名为 `megatron_<topic>_analysis`(小写 snake_case,对齐 [[torchtitan/index|torchtitan]] 的 `torchtitan_<topic>_analysis` 风格);如 `ep_analysis`→`14_megatron_ep_analysis`、`Megatron_LM_TFLOPS_Analysis`→`32_megatron_tflops_analysis`。
> - **去重删除**:`Megatron-LM_MoE_Zero_Redundancy_Analysis` 的零冗余 AllToAll 知识已被 [[14_megatron_ep_analysis]] 完全涵盖(深度更高),其独有的「EP=4 逐 token 数值走查」并入 [[14_megatron_ep_analysis]] §②.3.1 后,该页**已删除**。
> - **审计结论(保留)**:`21_megatron_fusion_operators_analysis`(融合深版)、`25_megatron_nonuniform_tp_analysis`(NTP 深版)、`22_megatron_memory_optimization_analysis`(显存 survey)均为**深版**,系列内对应页(precision §3 / tp_fsdp_resharding §2)是其精简digest——二者**互补非重复**,故保留。`16_megatron_distributed_optimizer_analysis` 2026-07-31 起已合并原「DDP/ZeRO 分片」与「优化器内部」两篇系列页,不再是"深版 vs 精简版"关系,详见 Core Topics 行。
> - 索引已收敛:并行轴/重计算/RL/模型结构等逐维深挖见下文系列;上方「Core Topics」仅列系列外的全景报告与专题深版。

> [!update] 2026-06-23 · DDP/分布式优化器 bucketing 与 overlap 机制深挖
> [[16_megatron_distributed_optimizer_analysis|分布式优化器]] §2.7「bucketing 算法与 overlap 调度」(原属已并入的 `megatron_ddp_optimizer_analysis.md`):逆序贪心分桶(`param_and_grad_buffer.py:891-939`)、bucket_size 默认 `max(40M,1M·dp)` 与 ring 报文 `bucket_size/dp` 调参(`distributed_data_parallel_config.py:49-61`)、反向 `register_grad_ready`(就绪**计数器**,非填数据;填数据是 `main_grad.add_` 原地累加,main_grad 为 buffer 视图)集齐 golden-count 才触发 RS(`:802`)、前向 forward-pre-hook → `finish_param_sync` wait + 预取下一桶(`:496/:531`、DDP`:413`)。基线 dev@232c478d4。

> [!update] 2026-06-23 · DeepEP 通信量图解(配 DeepEP 源码核实)
> [[14_megatron_ep_analysis|专家并行(EP)]] 新增 §③.3.5「通信量图解」三图(SVG→PNG):①按专家 vs 按节点发、②两级通信量分解 + 逐 token 公式、③2node×2GPU 数值走查 + IB 加速比。配图源码基线 **DeepEP @ `af9a040`**(legacy v1 `Buffer` 内核),并据 `internode.cu`(`notify_dispatch` :314/:313、`SourceMeta` :22、`kRDMAAndNVLForwarder` :971、:826 落地卡同号)对 §③.3.2 的「−1 免费落地卡」做了上界纠正。

> [!update] 2026-06-16 · ee3f1ff→232c478d4 增量刷新(298 commits,7 维 + 模型结构)
>
> 9 个并行 agent 逐页对照当前源码核实,**纯增不删**(既有内容仅以 `[!update]`/`[!deprecated]`/`[!contradiction]` 标注)。各维度要点(详见各页 2026-06-16 更新块):
>
> - **并行优化**:1F1B `mtp_post_process` 重排序 + combined-1F1B 释放 loss-node 输入存储(#4695/#4909);HyperCommGrid 命名视图支持异构并行(#5148)、bridge 跨网格 P2P 专用进程组(#5234);训练循环全面迁移 `pg_collection` 注入(#5259/#5250/#5006);动态 CP per-microbatch CP 度 + TE CP-group 还原修复(#4226/#5215)。见 [[15_megatron_pp_schedulers_analysis]] [[17_megatron_parallelism_orchestration_analysis]] [[13_megatron_cp_analysis]] [[12_megatron_tp_analysis]]
> - **通信优化**:**DeepEP v2 flex dispatcher**(`deepepv2` 后端,ElasticBuffer,#4793)、THD 序列打包下支持 deepep/hybridep(#4816)、高优先级 A2A 流 + HybridEP 预处理 SM(#4694)、dispatch 时排空前驱 reduce-scatter(#4940)、A2A-Overlap for Megatron-FSDP(#3797)。见 [[14_megatron_ep_analysis]] [[20_megatron_comm_overlap_analysis]]
> - **显存优化**:**Paged Stashing 正式落地**(#4247,2 层 PagedStashBuffer + runner 级整步重跑)、NCCL UB 内存池正确反注册(#4492)、细粒度 offload in-flight 节流(#4692)、显存估算计入 EP 切分(#4687)。见 [[22_megatron_memory_optimization_analysis]] [[18_megatron_recompute_analysis]]
> - **计算优化**:TE op-fuser 路径(GroupedMLP FC1→act→FC2 融合,#4636)、TEFusedDenseMLP(Dense+Grouped GEMM,SM100+,#4318)、ScaledSReLU / ClampedSwiGLU 融合(#4859/#5130)、DSv4 Hybrid Attention 融合 kernel(#4894)。见 [[21_megatron_fusion_operators_analysis]]
> - **低精度**:MXFP8/NVFP4 param-gather 一组修复(#4994/#4800/#4358/#4852)、opt-in MXFP8 LM-head 输出投影(#4825)、CUDA Graph API 拆解(impl/modules/inference-scope,#4292)。见 [[23_megatron_precision_cudagraph_fusion_analysis]]
> - **训练稳定性**:**grad-norm 超阈值跳过整步**(`grad_norm_skip_threshold`,#3460)、MoE aux/z-loss 在 TP>1 的梯度缩放修复(#5047)、DSA indexer loss 跨 micro-batch 平均(#4070)、**MTP 稳定性套件**(detach-heads / 独立损失缩放 / 独立裁剪组,#3456/#3459/#4116/#5080)。见 [[28_megatron_training_stability_observability_analysis]]
> - **RL / 训推一致**:`--rl-inference-parsers` 接入 MRL(#4768)、Refit 重构(统一 CopyService,#4762)、policy-epoch 权重时效追踪(#4533);推理侧高层 `MegatronLLM`/`MegatronAsyncLLM` API(#4697)、进程级 `InferenceMode` 标志取代基于 `training` 的判定(#4617)。见 [[30_megatron_rl_posttraining_consistency_analysis]] [[31_megatron_inference_engine_analysis]]
> - **模型结构(最大新增)**:**DeepSeek-V4 hybrid**(DSA 学习索引器 top-k 稀疏 + CSA/HCA 压缩注意力,#5042;TP=1、暂无推理路径)、GDN 序列打包(#2645)、mHC 支持 HybridModel(#4949)、Step-3.5-Flash 逐头注意力门控(#4841)。见 [[10_megatron_model_structure_analysis]]
>
> **本次纠正的知识库错误(4 处实质性)**:
> 1. **Muon/ZeRO 框架**:既有"普通 distributed optimizer 难以优雅支持 per-parameter 优化器切换"已过时 —— Muon 现经 `LayerWiseDistributedOptimizer` + 独立 `DistributedOptimizer` 经 `ChainedOptimizer` 串联,**与 ZeRO 切分共存**(#4509/#4771);且 `--layer-wise-distributed-optimizer` 这一 flag **不存在**(由 `--optimizer muon --use-distributed-optimizer` 触发)。`muon.py` 是 28 行兼容 shim,真实现在 `emerging_optimizers.py`。
> 2. **`mtp_isolated_loss` 已移除**:#5080 引入后被 #5223 合并进 `mtp_detach_heads` 并删除,HEAD 上不存在该配置。
> 3. **moe_layer `train()` 重写已删除**:dispatcher 不再按 train/eval 模式切换,改由 `InferenceMode.is_active()` 在 `MoELayer.forward` 判定(#4617)。
> 4. **GDN 统一 A2A(#4913)未在当前源码**:被后续 dev↔main 合并回退,GDN 前向仍用 per-section A2A 循环(已标 `[!contradiction]`)。

### 并行轴(4)

| Page | Key Concepts |
|------|-------------|
| [[15_megatron_pp_schedulers_analysis]] | 流水线并行 5 调度器:无流水线 / 1F1B / 交错 VPP / P2P-overlap / combined-1F1B;气泡公式推导、流水线模拟图(附调度模拟器 `_pp_sim.py`);2026-08-01 吸收 PP 进程组拓扑、P2P 内部实现、混合 CP 动态调度、多模块/多模态流水线(原 20/26 号页,已删除) |
| [[14_megatron_ep_analysis]] | 专家并行:AllGather / AllToAll / Flex(DeepEP/HybridEP)三种 token dispatcher;MoE Parallel Folding;通信量与负载均衡 |
| [[12_megatron_tp_analysis]] | 张量并行:ColumnParallel/RowParallel 共轭算子 f/g、MLP/Attention 切分;Sequence Parallelism |
| [[13_megatron_cp_analysis]] | 上下文并行:`cp_comm_type` 四选一(p2p/all_gather/a2a/a2a+p2p)配置接口、TE 透传架构、选型决策树;通用机制见 [[../../../01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis\|20_ring_attention_and_context_parallel_analysis]] |

> 数据并行 + 分布式优化器(ZeRO-0/1/2/3 四阶段、梯度分桶重叠、HSDP)2026-07-31 起并入 [[16_megatron_distributed_optimizer_analysis]](见下方「专题深挖」),不再单列本区。

### 编排与补遗(2)

| Page | Key Concepts |
|------|-------------|
| [[17_megatron_parallelism_orchestration_analysis]] | 进程组编排 capstone:RankGenerator、order 字符串、正交分组数学、双 RankGenerator(MoE Folding)、parallel_state/ProcessGroupCollection/HyperCommGrid 三层抽象 |
| [[27_megatron_tp_fsdp_resharding_supplements_analysis]] | Megatron-FSDP 内部(ZeRO-2/3 流水线)、Nonuniform TP 容错、Resharding/Refit |

> PP 补遗(P2P 通信内部、混合 CP 动态调度、多模块/多模态流水线)2026-08-01 起并入 [[15_megatron_pp_schedulers_analysis|流水线并行调度器]] §0.4/§6,不再单列本区;激活换出见 [[22_megatron_memory_optimization_analysis]] §2.3。

### 性能基建(2)

| Page | Key Concepts |
|------|-------------|
| [[18_megatron_recompute_analysis]] | 激活重计算:full(uniform/block)vs selective;标准 vs 输出丢弃 checkpointing |
| [[23_megatron_precision_cudagraph_fusion_analysis]] | FP8/FP4 四 recipe、CUDA Graph 三 impl、算子融合 |

> 优化器内部(混合精度 fp32 master、step 五步、Loss Scaling、梯度裁剪、LR 调度)2026-07-31 起并入 [[16_megatron_distributed_optimizer_analysis|分布式优化器]] §6-§11,不再单列本区。

### 系统专题(3)

| Page | Key Concepts |
|------|-------------|
| [[28_megatron_training_stability_observability_analysis]] | 训练稳定性与可观测:loss scaling / 梯度裁剪 / RerunStateMachine(SDC 归因)/ QK-clip;Timer、MoE 逐层指标、指标目录 |
| [[30_megatron_rl_posttraining_consistency_analysis]] | RL 后训练适配与训推一致性:Refit、布局/格式保真、inference_optimized、重算 logprob、importance sampling |
| [[31_megatron_inference_engine_analysis]] | 推理引擎:Static/Dynamic 引擎、连续批处理、块级 KV cache(paged)、prefix caching、chunked prefill |

### 数据 / 模型 / 存档(4)

| Page | Key Concepts |
|------|-------------|
| [[11_megatron_dataset_analysis]] | GPT 数据集:IndexedDataset(.bin/.idx)、三级索引、隐式打包 vs 显式打包 |
| [[29_megatron_packed_dataset_dynamic_cp_analysis]] | 序列打包与动态 CP 统一流水线:BasePackingScheduler→DpBalanced→DefaultDynamicCP 继承链 |
| [[10_megatron_model_structure_analysis]] | 模型结构:Spec 系统、TransformerLayer、注意力家族(MHA/GQA/MLA)、MoE Router 算法、MTP、SSM/Mamba |
| [[19_megatron_dist_checkpointing_analysis]] | 分布式 checkpoint:ShardedTensor(local↔global 映射)、并行无关存档、sharded_state_dict、fully-parallel/async 策略 |

## Cross-Domain Links

- Distributed parallelism concepts connect to [[11_muon_analysis]] in the LLM domain (Muon's ZeRO incompatibility)
- MoE zero-redundancy dispatch relates to [[10_llm_initiliaze_analysis]] (MoE expert initialization)
- CUDA Graphs usage in Megatron connects to [[10_pytorch_cuda_graphs_complete_guide]] in the CUDA Graphs sub-domain
- Weight sync patterns relate to inference optimization topics in the torch_compile domain
- FP8/low-precision training connects to [[13_low_precision_training_analysis]] and [[14_transformer_engine_analysis]] in the training domain
- Distributed optimizer state sharding relates to ZeRO strategies in [[16_megatron_distributed_optimizer_analysis]]
- Memory optimization techniques coalesce around [[22_megatron_memory_optimization_analysis]]
- Fusion operators complement communication overlap patterns in [[20_megatron_comm_overlap_analysis]] and [[21_megatron_fusion_operators_analysis]]
- NTP fault tolerance connects to TP/DP/CP topology concepts in [[25_megatron_nonuniform_tp_analysis]]

## Knowledge Gaps

These topics are referenced but lack dedicated wiki pages:

- ~~**Context Parallelism deep dive**~~ → 专文 [[13_megatron_cp_analysis]](4 种 cp_comm_type + zigzag 负载均衡);另见 [[15_megatron_pp_schedulers_analysis]]
- **TransformerEngine integration** — ~~referenced but not documented~~ → addressed by [[14_transformer_engine_analysis]]
- **Low-precision training** — ~~FP8/FP4 scattered across exam and V4 pages~~ → consolidated in [[13_low_precision_training_analysis]];另见 [[23_megatron_precision_cudagraph_fusion_analysis]]
- ~~**Megatron-LM checkpoint format**~~ → 专文 [[19_megatron_dist_checkpointing_analysis]](ShardedTensor、并行无关存档)
- ~~**Distributed optimizer** — no standalone analysis~~ → addressed by [[16_megatron_distributed_optimizer_analysis]](2026-07-31 合一页)
- ~~**Memory optimization panorama** — scattered across multiple pages~~ → addressed by [[22_megatron_memory_optimization_analysis]];另见 [[18_megatron_recompute_analysis]]
- ~~**Fusion operators** — no dedicated page~~ → addressed by [[21_megatron_fusion_operators_analysis]]
- ~~**Sequence Parallelism implementation details**~~ → 专门章节见 [[12_megatron_tp_analysis]] §4;另见 [[15_megatron_pp_schedulers_analysis]]、[[20_megatron_comm_overlap_analysis]]
- ~~**Fault tolerance / GPU failure recovery** — not covered~~ → addressed by [[25_megatron_nonuniform_tp_analysis]];另见 [[27_megatron_tp_fsdp_resharding_supplements_analysis]] §2
- **NTP checkpoint conversion tooling** — gradient resharing is implemented, but parameter/optimizer state resharding for checkpoint restore is not yet in the codebase

## Related Pages

- [[15_megatron_pp_schedulers_analysis]] · [[14_megatron_ep_analysis]] · [[12_megatron_tp_analysis]] · [[13_megatron_cp_analysis]] · [[16_megatron_distributed_optimizer_analysis]]
- [[17_megatron_parallelism_orchestration_analysis]] · [[10_megatron_model_structure_analysis]] · [[25_megatron_nonuniform_tp_analysis]]
- [[02_engineering/01_ai_frameworks/index]]
