---
title: "TorchTitan — 训练运行时、多维并行与编译器实验知识地图"
---

# TorchTitan — 训练运行时、多维并行与编译器实验知识地图

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（commit date：2026-08-26；核验：2026-08-27）
> **演进跨度**：上一知识库基线 `61c010fcb` → 当前基线，320 commits
> **内容规模**：23 篇内容页 + 本索引
>
> **主线结论**：TorchTitan 已从“训练脚本 + 模型手写并行 plan”演进成三层协议系统：full Python configuration/`Configurable` 决定组件所有权，`ParallelDims` 决定同一组 rank 的 storage/runtime mesh 视图，`ShardingConfig + SpmdType` 决定参数、输入、局部 kernel 与输出布局。核心 Trainer 负责完整优化步、数据游标与 checkpoint 提交顺序；TorchFT、Transformers backend、Forge、GraphTrainer 则是能力和成熟度不同的独立实验路径，不能被当成 core Trainer 的开关。

---

## 0. 先从哪里读

| 目标 | 推荐页面 | 读完应得到什么 |
|---|---|---|
| 跑通当前训练入口 | [[01_torchtitan_trainer_quickstart]] → [[04_torchtitan_config_model_protocol_analysis]] | Python full recipe、组件所有权、Trainer 初始化与一次完整优化步 |
| 理解数据、恢复与多模态契约 | [[02_torchtitan_data_pipeline_grain_analysis]] → [[03_torchtitan_checkpoint_state_recovery_analysis]] → [[05_torchtitan_multimodal_data_model_contract_analysis]] | Grain iterator graph、packing/token budget、数据游标恢复，以及 placeholder/patch/vision scatter 如何保持一一对应 |
| 理解所有并行的共同底座 | [[10_torchtitan_parallel_dims_analysis]] → [[16_torchtitan_spmd_types_analysis]] | storage mesh 与类型 mesh 为什么分开，模块如何声明布局 |
| 理解传统五维并行 | [[11_torchtitan_fsdp_analysis]]、[[12_torchtitan_tp_analysis]]、[[13_torchtitan_cp_analysis]]、[[14_torchtitan_pp_analysis]]、[[15_torchtitan_ep_analysis]] | 参数/激活切分、collective、组合约束与失败边界 |
| 调吞吐和显存 | [[22_torchtitan_ac_analysis]]、[[23_torchtitan_compute_memory_optimizations_analysis]]、[[24_torchtitan_comm_optimizations_overlap_analysis]] | 重计算、低精度/LoRA/融合、compile、通信资源与重叠所有权 |
| 看编译/optimizer 实验 | [[26_torchtitan_flex_shard_dist_muon_analysis]]、[[27_torchtitan_graph_trainer_compiler_runtime_analysis]] | optimizer compute layout 重分布与 joint FX graph 训练 |
| 看独立实验路径 | [[28_torchtitan_torchft_fault_tolerance_analysis]]、[[29_torchtitan_transformers_modeling_backend_analysis]]、[[30_torchtitan_forge_engine_analysis]] | 动态副本容错、HF 模型协议适配，以及 Forge 的构造 seam/当前协议漂移 |

## 1. 本轮审计发现：哪些知识已经过时

这次不是只更新 commit。以下旧叙述会直接误导当前代码阅读：

| 旧知识 | 当前事实 | 处置 |
|---|---|---|
| `full_dtensor` 是当前 SPMD 后端之一 | `full_dtensor` 已删除；默认是 `spmd_types`，`partial_dtensor` 仍是兼容路径 | 从通信专题的 live backend 列表移除 |
| Llama TP 有手写 `ParallelStyle` plan 与配置式路径两条主线 | 模型统一递归消费 `ShardingConfig`;默认后端用 plain local tensor + `SpmdType` + 显式 collective | 重写 [[12_torchtitan_tp_analysis]] |
| CP 通过 `apply_cp_to_forward`，语言模型有 SDPA Ring 与 Flex all-gather 两条 wrapper | wrapper 和 partial-DTensor CP 已删除；当前由输入 sharder、attention `ShardingConfig` 和 `[dp,cp,tp]` current mesh 接线 | 重写 [[13_torchtitan_cp_analysis]] |
| Async TP 是 `ParallelismConfig` 初始化开关 | 核心 Trainer 中开关移入 `CompileConfig`，在 `apply_compile()` 内注册 symmetric memory 并启用 Inductor micro-pipeline | 更新 TP/通信专题 |
| EP 是模型专属 `ExpertParallel`，shared expert 可与 combine 重叠 | 当前稳定边界是 `RoutedExperts = token_dispatcher + inner_experts`;shared expert 在 routed combine 后顺序执行 | 重写 [[15_torchtitan_ep_analysis]] |
| MinimalAsyncEP 只能单独使用，TP=CP=PP=1 | 当前 H100 recipe 覆盖 FSDP+CP+TP+EP；但 full recompute、固定 buffer 与 PP 测试边界仍需遵守 | 更新 EP/通信专题 |
| GraphTrainer 主要等于 SimpleFSDP | 已扩展为 joint FX pass pipeline、EP overlap、GraphPP、precompile、AutoParallel 等实验控制面 | 新增 [[27_torchtitan_graph_trainer_compiler_runtime_analysis]] |
| TOML/巨型 `JobConfig` 是配置权威 | 当前入口是返回完整 typed config 的 Python recipe；CLI override 是兼容面，组件 Config 自己 build owner | 新增 [[04_torchtitan_config_model_protocol_analysis]] |
| dataloader 只需保存 batch index | 当前 Grain pipeline 的 exact resume 依赖 source、mix/packing 与 iterator graph 状态；训练 token budget 已折成 `[T]` | 新增 [[02_torchtitan_data_pipeline_grain_analysis]] |
| checkpoint 等于一个 DCP `save/load` 调用 | manager/协议、storage、PP FQN flatten、async staging、HF/native/final export 是不同边界；TorchCheckpointing hooks 仍未闭环 | 新增 [[03_torchtitan_checkpoint_state_recovery_analysis]] |
| 多模态只是普通文本 batch 外挂一个 vision tensor | 正确性边界是 resize 后 placeholder runs、按 document/media 顺序打包的 patches，以及 encoder 输出 scatter 三者一一对应；视觉输入还是 DP-local 可变长对象，当前多模态模型不支持 CP | 新增 [[05_torchtitan_multimodal_data_model_contract_analysis]] |
| LoRA 是 build 后挂 adapter/过滤 optimizer，已有 adapter-only export | 当前在 Config 树上继承量化后的 Linear owner、推导 TP placement 并冻结所有非目标节点；普通 DCP 仍保存完整 model state | 更新 [[23_torchtitan_compute_memory_optimizations_analysis]] |
| 可观测性只有 metrics 与短窗口 profiler | core/RL 共享按 rank/source/task 的 structured span/scalar/instant；compile 内主动 no-op，离线可转 Perfetto trace | 更新 [[01_torchtitan_trainer_quickstart]] |
| HF backend 的 MoE 仍靠原 HF forward hooks，且不能加载 HF 权重 | HEAD 已先替换 native Titan MoE；dense/SFT 已有 HF adapter/load recipe，但 MoE+PP 等组合仍缺 | 新增 [[29_torchtitan_transformers_modeling_backend_analysis]] |
| Forge 是与 Trainer 等价、只缺量化/容错的轻量入口 | 设计上它只拥有构造核心；HEAD 还保留已不存在的 `ModelSpec.loss` 消费点，标准 example 当前发生协议漂移 | 新增 [[30_torchtitan_forge_engine_analysis]] |

删除与替代的源码证据分别在 `torchtitan/config/configs.py:168-180`、`torchtitan/distributed/context_parallel/api.py:28-37`、`torchtitan/distributed/compile.py:39-96`、`torchtitan/models/common/token_dispatcher.py:358-423`。新默认不是“把 DTensor 藏起来”，而是把布局语义提升成可以检查的模块契约。

## 2. 当前系统的两张平面

### 2.1 storage plane：训练状态长期放在哪里

`ParallelDims` 从 world mesh 派生两类持久 storage 视图：

```text
dense storage:  [pp, dp_replicate, dp_shard, cp, tp]
sparse storage: [pp, dp_replicate, efsdp, ep]
```

稠密参数由 FSDP 在 `dp_shard(+cp)` 上分片、在 `dp_replicate` 上复制；稀疏专家参数由 EFSDP 与 EP 分工。两者覆盖同一批 rank，满足 `dp_shard * cp * tp == efsdp * ep`（`torchtitan/distributed/parallel_dims.py:216-295`,`torchtitan/config/configs.py:284-291`）。

### 2.2 forward/backward type plane：当前值有什么布局

模型 forward/backward 使用 dense `[dp,cp,tp]` 或 sparse `[dp_replicate,efsdp,ep]` current mesh。参数/输入/输出布局写成 `SpmdType`;模块边界只在 source 与 destination 不同时发对应单轴 collective（`torchtitan/protocols/sharding.py:59-159`,`torchtitan/protocols/module.py:244-290,559-725`）。

为什么分两张平面：同一个 CP rank 既可以在激活 token 维上持有 shard，也可以作为 FSDP 参数 storage shard 轴；同名硬件分组承担的是两种不同状态语义。把它们压成一张 placements 表会让“计算值布局”和“持久所有权”互相污染。

完整机制见 [[10_torchtitan_parallel_dims_analysis]] 与 [[16_torchtitan_spmd_types_analysis]]。

## 3. 核心 Trainer 的当前施加顺序

推荐入口是 Python recipe/registry 经 `run_train.sh` 启动 `torchtitan.train`，最后构造 `Trainer`（`run_train.sh:21-45`,`torchtitan/train.py:17-68`）。对 decoder 的主链可概括为：

```text
配置解析与 structured logger 初始化
  → 模型 Config.update_from_config
  → 对 full config tree 应用 override，并复验组合 guards
  → meta device 构模
  → [若 PP] 按 module FQN 切 stage
  → model.parallelize(): TP/SP/CP/EP 的 ShardingConfig 与 state shard
  → activation checkpoint policy
  → per-block compile（含可选 Async TP）
  → FSDP2：dense/MoE 共享 apply_fsdp_to_decoder
  → materialize + initialize
  → optimizer / LR scheduler / Grain dataloader / checkpointer
  → [可选] 捕获 forward-backward body 的 CUDA Graph
```

Llama 当前按这个顺序调用 `model.parallelize → AC → compile → FSDP`（`torchtitan/models/llama3/parallelize.py:23-78`）。FSDP 被放在最外层，仍可管理被 TP/EP 切过的参数；PP 则必须在各 stage 施加上述链之前完成模型切分。

一次优化步的状态顺序不是“forward/backward/step”三个词这么简单：Trainer 先规约全局有效 token 数，再执行每个 accumulation group，随后做 grad clip 与 finite 检查；后台 checkpoint staging 完成后才允许 optimizer 修改状态（`torchtitan/trainer.py:774-940`）。详见 [[01_torchtitan_trainer_quickstart]]。

## 4. 运行时、数据与控制面（段 0）

| 页面 | 当前核心机制 |
|---|---|
| [[01_torchtitan_trainer_quickstart]] | full recipe 到运行时状态图；meta→parallelize→materialize；完整 optimizer-step 提交、有限性闸门与 structured trace |
| [[02_torchtitan_data_pipeline_grain_analysis]] | Grain random-access/stream source、mix、packing、`[T]` token budget 与 exact iterator resume |
| [[03_torchtitan_checkpoint_state_recovery_analysis]] | manager/ABC/storage seam、PP/FQN state、async staging、load precedence、HF/native/final export |
| [[04_torchtitan_config_model_protocol_analysis]] | full configuration、`Configurable` owner/build、override traversal、`ModelSpec`/`Module` 协议与当前 TODO |
| [[05_torchtitan_multimodal_data_model_contract_analysis]] | resize 后 placeholder、document/media 保序 packing、token/patch 双预算、DP-local vision join 与 scatter 一致性 |

## 5. 基础并行系列（段 1）

| 页面 | 当前核心机制 |
|---|---|
| [[10_torchtitan_parallel_dims_analysis]] | `ParallelDims`、dense/sparse storage mesh、forward/backward current mesh、等积约束 |
| [[11_torchtitan_fsdp_analysis]] | dense/sparse 参数 storage plane、FSDP unit/per-param mesh、mixed precision/offload、全局 token 归一化与 EP 预取 |
| [[12_torchtitan_tp_analysis]] | `ShardingConfig` 的 colwise/rowwise/SP/Loss Parallel；compile Async TP 与显式 dist-GEMM 两条重叠路径 |
| [[13_torchtitan_cp_analysis]] | 输入与 mask sharding、K/V `CP shard→replicate`、decoder Flex 与 Flux SDPA 的边界 |
| [[14_torchtitan_pp_analysis]] | stage split、P2P、PyTorch schedule、microbatch 与 FSDP reshard 策略 |
| [[15_torchtitan_ep_analysis]] | dense/sparse mesh、统一 dispatcher API、aux-loss-free bias 状态机、确定性 scatter combine、AllToAll/DeepEP v2/HybridEP/MinimalAsyncEP |
| [[16_torchtitan_spmd_types_analysis]] | `SpmdType`/`PartitionSpec`、state sharding、local region、类型断言与当前限制 |

### 五个常用维度速览

| 维度 | 主要切分对象 | 当前框架边界 | 主要通信 |
|---|---|---|---|
| DP/FSDP | 参数、梯度、优化器状态 | `fully_shard` + storage mesh | all-gather + reduce-scatter（HSDP 另有 all-reduce） |
| TP/SP | weight feature/head、activation token/feature | `ShardingConfig` + SpmdType | all-gather / reduce-scatter / all-reduce，或 dist-GEMM fused path |
| CP | activation token/sequence | input sharder + attention local map | K/V shard→replicate；输入/梯度由 CP 布局约束 |
| PP | 模型 module/layer depth | `torch.distributed.pipelining` schedule | stage P2P |
| EP | routed expert weights、tokens | sparse mesh + token dispatcher | dispatch/combine all-to-all 或专用 kernel |

## 6. 机制深挖、性能与实验系列（段 2）

| 页面 | 边界与用途 |
|---|---|
| [[20_torchtitan_fsdp_prefetch_overlap_memory_analysis]] | FSDP 通信组、dense 隐式/EP 显式预取、reshard/symmetric-memory 与上游 buffer 生命周期边界 |
| [[21_torchtitan_hsdp_backward_overlap_analysis]] | HSDP replicate/shard 所有权、逐 block overlap 窗口、全局 token 缩放与上游 FSDP2 证据边界 |
| [[22_torchtitan_ac_analysis]] | Full/Selective 策略对象、编译器 memory budget、non-reentrant/SAC 固定基线；GraphTrainer 图内策略另见 27 |
| [[23_torchtitan_compute_memory_optimizations_analysis]] | Float8/MXFP8/NVFP4、LoRA 冻结/TP 布局、融合模块、regional compile/CUDA Graph、Chunked Loss、optimizer state 与 finite gate |
| [[24_torchtitan_comm_optimizations_overlap_analysis]] | process-group 控制、symmetric memory、compiler/module/FSDP/EP 的通信所有权、host-sync 与 capture 边界 |
| [[25_torchtitan_simple_fsdp_analysis]] | SimpleFSDP 如何把 unshard/reduce collective 表达进 joint graph |
| [[26_torchtitan_flex_shard_dist_muon_analysis]] | storage layout 与 optimizer compute layout 的双向 packed A2A 重分布 |
| [[27_torchtitan_graph_trainer_compiler_runtime_analysis]] | joint fwd/loss/bwd FX、ordered pass pipeline、EP overlap、GraphPP、precompile 与 AutoParallel；集成定义仍禁用，optimizer 仍在图外 |
| [[28_torchtitan_torchft_fault_tolerance_analysis]] | replica-group 故障域、quorum optimizer、FSDP all-reduce hook 与全局/每副本双通道 checkpoint |
| [[29_torchtitan_transformers_modeling_backend_analysis]] | HF config/Module/sharding/state-dict 协议适配、native Titan MoE replacement 与兼容矩阵 |
| [[30_torchtitan_forge_engine_analysis]] | 可嵌入构造内核、下游 loop 所有权，以及 HEAD `ModelSpec.loss` 协议漂移审计 |

## 7. 新增能力地图

本轮 320 commits 中，最值得单独建立概念所有权的不是新增模型名，而是以下跨模型子系统：

1. **Full configuration + model/component protocols**：配置从 TOML/单体对象转成 typed owner/build tree，模型扩展面从集中 registry 变成 `ModelSpec`/`Module`/preprocess 协议。
2. **Grain + state recovery**：数据不再是 Trainer 外的黑盒；mix/packing/iterator graph 与 checkpoint load precedence 共同决定是否 exact resume。
3. **SPMD Types**：默认布局协议；解决 plain local tensor 如何保留可检查的全局分布语义。
4. **多模态数据—模型契约**：placeholder 数量、packed patch 顺序与 vision scatter 是同一个端到端不变量；只分析 processor 或 vision encoder 都无法证明正确性。
5. **LoRA + structured observability**：前者把冻结范围和 TP adapter placement 纳入 Config 协议，后者用共享事件 schema 对齐同步 Trainer 与异步 actor；两者原先均被知识库漏掉。
6. **dist-GEMM / symmetric memory**：把 TP all-gather/reduce-scatter 直接折进 GEMM；与 compile Async TP 是不同入口。
7. **FlexShard / DistMuon**：在 optimizer step 内临时重排计算所有权，而不改变 FSDP/EP 持久 storage。
8. **GraphTrainer / GraphPP / AutoParallel**：让 fwd/loss/bwd 通信、重算、offload、Inductor、stage action 与 placement solver 进入统一实验控制面；生产 optimizer 仍在图外，AutoParallel 集成定义仍禁用。
9. **TorchFT / Transformers backend / Forge**：分别探索动态副本容错、外部模型生态适配与可嵌入训练构造；三者各自有独立成熟度和失败边界。

此外，核心 Trainer 已增加独立 CUDA Graph 捕获和全程 structured trace；checkpoint/data/optimizer 也从单文件演进为组件包。数据与 checkpoint 已各自建立独立概念页，避免继续把 exact resume、HF bridge 和 storage backend 混进并行维度页面。

## 8. 组合与网络映射原则

仍可用的工程启发是把高频/大流量维度放在快互联上，但当前代码的合法性首先由 mesh 与配置验证决定：

```text
节点内高速互联：TP/SP、EP token exchange、dist-GEMM symmetric memory
节点内或少量节点：FSDP shard group
跨节点复制：HSDP dp_replicate
较慢链路：PP（stage 间主要传 activation/grad）
```

不要仅凭这张映射判断组合可用。例如 decoder CP 当前只支持 `spmd_types + FlexAttention`;GraphTrainer CP 在当前测试矩阵仍禁用；DistMuon 的 Kimi recipe 又显式拒绝 TP>1。各专题页的“失败边界”优先于一般映射原则。

## Cross-Domain Links

- [[../megatron-lm/index|Megatron-LM 知识地图]] —— CUDA/Megatron 专用并行栈的对照系
- [[../21_async_collective_tensor_deepdive|AsyncCollectiveTensor 深潜]] —— ACT 机制本体（已于 2026-08-28 重定基线并逐条重核）；注意它当初是为「集合通信可被 dynamo/FX 追踪」而引入，掩盖是副产品
- [[../30_comm_compute_overlap_analysis|跨框架通算重叠]] —— PP/TP/EP/FSDP 的横向比较
- [[../31_comm_compute_fusion_guide|通算融合指南]] —— 从独立 collective 到 fused/distributed GEMM 的方法论
- [[../32_distributed_optimizer_deepdive|分布式优化器对比]] —— FSDP2、ZeRO 与其他实现
- [[../../06_auto_parallel/index|自动并行]] —— GraphTrainer AutoParallel 的上位概念域

## Related Pages

- [[../index|训练框架目录索引]]
- [[01_torchtitan_trainer_quickstart]] · [[10_torchtitan_parallel_dims_analysis]] · [[16_torchtitan_spmd_types_analysis]]
- [[11_torchtitan_fsdp_analysis]] · [[12_torchtitan_tp_analysis]] · [[13_torchtitan_cp_analysis]] · [[14_torchtitan_pp_analysis]] · [[15_torchtitan_ep_analysis]]
- [[23_torchtitan_compute_memory_optimizations_analysis]] · [[24_torchtitan_comm_optimizations_overlap_analysis]]
- [[26_torchtitan_flex_shard_dist_muon_analysis]] · [[27_torchtitan_graph_trainer_compiler_runtime_analysis]]
- [[02_torchtitan_data_pipeline_grain_analysis]] · [[03_torchtitan_checkpoint_state_recovery_analysis]] · [[04_torchtitan_config_model_protocol_analysis]] · [[05_torchtitan_multimodal_data_model_contract_analysis]]
- [[28_torchtitan_torchft_fault_tolerance_analysis]] · [[29_torchtitan_transformers_modeling_backend_analysis]] · [[30_torchtitan_forge_engine_analysis]]
