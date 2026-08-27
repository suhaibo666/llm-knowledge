---
title: "TorchTitan — 训练运行时、多维并行与编译器实验知识地图"
---

# TorchTitan — 训练运行时、多维并行与编译器实验知识地图

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-27）
> **演进跨度**：上一知识库基线 `61c010fcb` → 当前基线，320 commits
> **内容规模**：16 篇内容页 + 本索引
>
> **主线结论**：TorchTitan 已从“模型适配层手写每一种并行 plan”演进为两层协议：`ParallelDims` 负责同一组 rank 的 storage/forward mesh 视图，`ShardingConfig + SpmdType` 负责参数、输入、局部 kernel 与输出的模块契约。核心 Trainer 保留 PyTorch-native eager 生命周期，同时把 CUDA Graph、dist-GEMM、四类 EP dispatcher、FlexShard/DistMuon 接入生产路径；GraphTrainer 则把 forward+loss+backward 暴露为可变换 FX 图。

---

## 0. 先从哪里读

| 目标 | 推荐页面 | 读完应得到什么 |
|---|---|---|
| 跑通当前训练入口 | [[01_torchtitan_trainer_quickstart]] | Python recipe、Trainer 初始化、一次 train step、checkpoint/CUDA Graph 边界 |
| 理解所有并行的共同底座 | [[10_torchtitan_parallel_dims_analysis]] → [[16_torchtitan_spmd_types_analysis]] | storage mesh 与类型 mesh 为什么分开，模块如何声明布局 |
| 理解传统五维并行 | [[11_torchtitan_fsdp_analysis]]、[[12_torchtitan_tp_analysis]]、[[13_torchtitan_cp_analysis]]、[[14_torchtitan_pp_analysis]]、[[15_torchtitan_ep_analysis]] | 参数/激活切分、collective、组合约束与失败边界 |
| 调吞吐和显存 | [[22_torchtitan_ac_analysis]]、[[23_torchtitan_compute_memory_optimizations_analysis]]、[[24_torchtitan_comm_optimizations_overlap_analysis]] | 重计算、低精度/融合、compile/对称内存/通信重叠 |
| 看新演进 | [[26_torchtitan_flex_shard_dist_muon_analysis]]、[[27_torchtitan_graph_trainer_compiler_runtime_analysis]] | optimizer compute layout 重分布与 joint FX graph 训练 |

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
配置解析与模型 Config.update_from_config
  → meta device 构模
  → [若 PP] 按 module FQN 切 stage
  → model.parallelize(): TP/SP/CP/EP 的 ShardingConfig 与 state shard
  → activation checkpoint policy
  → per-block compile（含可选 Async TP）
  → FSDP2：dense/MoE 共享 apply_fsdp_to_decoder
  → materialize + initialize
  → optimizer / LR scheduler / dataloader / checkpointer
  → [可选] 捕获 forward-backward body 的 CUDA Graph
```

Llama 当前按这个顺序调用 `model.parallelize → AC → compile → FSDP`（`torchtitan/models/llama3/parallelize.py:23-78`）。FSDP 被放在最外层，仍可管理被 TP/EP 切过的参数；PP 则必须在各 stage 施加上述链之前完成模型切分。

一次优化步的状态顺序不是“forward/backward/step”三个词这么简单：Trainer 先规约全局有效 token 数，再执行每个 accumulation group，随后做 grad clip 与 finite 检查；后台 checkpoint staging 完成后才允许 optimizer 修改状态（`torchtitan/trainer.py:774-940`）。详见 [[01_torchtitan_trainer_quickstart]]。

## 4. 基础并行系列（段 1）

| 页面 | 当前核心机制 |
|---|---|
| [[10_torchtitan_parallel_dims_analysis]] | `ParallelDims`、dense/sparse storage mesh、forward/backward current mesh、等积约束 |
| [[11_torchtitan_fsdp_analysis]] | FSDP2 逐参数状态机、all-gather/reduce-scatter 多流、`DataParallelMeshDims` 与统一 dense/MoE 接线 |
| [[12_torchtitan_tp_analysis]] | `ShardingConfig` 的 colwise/rowwise/SP/Loss Parallel；compile Async TP 与显式 dist-GEMM 两条重叠路径 |
| [[13_torchtitan_cp_analysis]] | 输入与 mask sharding、K/V `CP shard→replicate`、decoder Flex 与 Flux SDPA 的边界 |
| [[14_torchtitan_pp_analysis]] | stage split、P2P、PyTorch schedule、microbatch 与 FSDP reshard 策略 |
| [[15_torchtitan_ep_analysis]] | dense/sparse mesh、统一 dispatcher API、AllToAll/DeepEP v2/HybridEP/MinimalAsyncEP |
| [[16_torchtitan_spmd_types_analysis]] | `SpmdType`/`PartitionSpec`、state sharding、local region、类型断言与当前限制 |

### 五个常用维度速览

| 维度 | 主要切分对象 | 当前框架边界 | 主要通信 |
|---|---|---|---|
| DP/FSDP | 参数、梯度、优化器状态 | `fully_shard` + storage mesh | all-gather + reduce-scatter（HSDP 另有 all-reduce） |
| TP/SP | weight feature/head、activation token/feature | `ShardingConfig` + SpmdType | all-gather / reduce-scatter / all-reduce，或 dist-GEMM fused path |
| CP | activation token/sequence | input sharder + attention local map | K/V shard→replicate；输入/梯度由 CP 布局约束 |
| PP | 模型 module/layer depth | `torch.distributed.pipelining` schedule | stage P2P |
| EP | routed expert weights、tokens | sparse mesh + token dispatcher | dispatch/combine all-to-all 或专用 kernel |

## 5. 机制深挖与性能系列（段 2）

| 页面 | 边界与用途 |
|---|---|
| [[20_torchtitan_fsdp_prefetch_overlap_memory_analysis]] | FSDP all-gather 预取、copy-in 双缓冲与完整参数显存生命周期 |
| [[21_torchtitan_hsdp_backward_overlap_analysis]] | HSDP reduce-scatter / all-reduce 双流时序与显存峰值 |
| [[22_torchtitan_ac_analysis]] | Full/Selective 策略对象、编译器 memory budget、non-reentrant/SAC 固定基线；GraphTrainer 图内策略另见 27 |
| [[23_torchtitan_compute_memory_optimizations_analysis]] | Float8/MXFP8/NVFP4、融合算子、dist-GEMM、compile/CUDA Graph、optimizer 路线 |
| [[24_torchtitan_comm_optimizations_overlap_analysis]] | FSDP/TP/EP/PP/CP 跨维度重叠、symmetric memory 与当前组合边界 |
| [[25_torchtitan_simple_fsdp_analysis]] | SimpleFSDP 如何把 unshard/reduce collective 表达进 joint graph |
| [[26_torchtitan_flex_shard_dist_muon_analysis]] | storage layout 与 optimizer compute layout 的双向 packed A2A 重分布 |
| [[27_torchtitan_graph_trainer_compiler_runtime_analysis]] | full train-step FX、pass pipeline、EP overlap、GraphPP 与 precompile |

## 6. 新增能力地图

本轮 320 commits 中，最值得单独建立概念所有权的不是新增模型名，而是四个跨模型子系统：

1. **SPMD Types**：默认布局协议；解决 plain local tensor 如何保留可检查的全局分布语义。
2. **dist-GEMM / symmetric memory**：把 TP all-gather/reduce-scatter 直接折进 GEMM；与 compile Async TP 是不同入口。
3. **FlexShard / DistMuon**：在 optimizer step 内临时重排计算所有权，而不改变 FSDP/EP 持久 storage。
4. **GraphTrainer / GraphPP**：让通信、重算、offload、Inductor 与 stage action 进入统一 joint graph/pass 控制面。

此外，核心 Trainer 已增加独立 CUDA Graph 捕获；checkpoint/data/optimizer 也从单文件演进为组件包，并新增 Torch checkpointing backend 与 Grain data path。这些组件生命周期已纳入 [[01_torchtitan_trainer_quickstart]]，若后续要分析格式兼容与 dataloader 恢复语义，应再建立独立组件页，而不是塞入并行维度页面。

## 7. 组合与网络映射原则

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
- [[../21_async_collective_tensor_deepdive|AsyncCollectiveTensor 深潜]] —— 旧/通用异步 collective 语义；不可再直接当作当前 CP wrapper 证据
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
