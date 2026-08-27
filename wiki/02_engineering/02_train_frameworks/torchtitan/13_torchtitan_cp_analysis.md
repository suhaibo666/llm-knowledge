# 上下文并行 CP：从 forward wrapper 到 SPMD 布局边界

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-27）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **主线**：当前 CP 不再由 `apply_cp_to_forward` 给 attention 动态套 Ring/AllGather wrapper，而是拆成两个显式边界：模型输入先按声明的 CP `SpmdType` 做物理切分；attention 的 `ShardingConfig` 再把 Q 保持为 CP token shard、把 K/V 从 CP shard 重分布为 replicate，并把输出恢复为 Q 的布局。换言之，CP 已从“识别 kernel 类型后改写 forward”迁到“输入布局 + 模块边界布局 + 运行时 mesh”的声明式接线。
>
> 本页只分析 TorchTitan 的框架接线、布局契约、组合验证与失败边界。Ring Attention、HeadTail/PTRR 算法与通信量等通用机制统一见 [[01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis|Ring Attention 与上下文并行理论页]]。
>
> 主要源文件：`torchtitan/distributed/context_parallel/api.py`、`torchtitan/models/common/decoder_sharding.py`、`torchtitan/protocols/module.py`、`torchtitan/distributed/spmd_types.py`。

---

## 1. 先纠正旧版知识：路径迁移与 forward wrapper 删除

旧页基于 `cf3c4312`，其两个入口是单文件 `torchtitan/distributed/context_parallel.py` 中的 `cp_shard` 与 `apply_cp_to_forward`。当前基线已经经历两次决定性演进：

1. 提交 `547b0b481` 把旧单文件迁为 `torchtitan/distributed/context_parallel/{__init__.py,api.py}`。当前包只导出 `cp_shard`、`prepare_context_parallel_input` 与 `validate_cp_backend`（`torchtitan/distributed/context_parallel/__init__.py:7-24`）。
2. 提交 `5dd944e62` 删除 `apply_cp_to_forward` 及 `partial_dtensor` CP 路径。当前验证器在 `cp > 1` 时只接受 `spmd_backend="spmd_types"`（`torchtitan/distributed/context_parallel/api.py:28-37`）。

> [!deprecated] 旧版“双 forward wrapper”与 Ring dispatcher 描述已失效
> 当前 TorchTitan 源码没有 `apply_cp_to_forward`，也不再从 CP 模块启用 `_enable_context_parallel_dispatcher` 或调用 `flex_cp_allgather`。因此旧页“SDPA → DTensor CP dispatcher/Ring、FlexAttention → custom-op K/V all-gather”的**框架接线**不是现状；现状是配置阶段写入 `ShardingConfig`，`Module.parallelize()` 统一安装输入/输出 redistribution 与 local region（`torchtitan/protocols/module.py:244-290`）。

> [!deprecated] 旧版 `AsyncCollectiveTensor` 时序不能继续写成 TorchTitan 当前 CP 机制
> 那段描述依赖已删除的 SDPA Ring rotater 路径。当前 TorchTitan 可直接核验的是 `spmd_redistribute_per_axis()` 在布局变化时调用 `spmd.redistribute`，并没有在 CP 接线层暴露或检查 `AsyncCollectiveTensor`（`torchtitan/distributed/spmd_types.py:398-437`）。是否以及怎样异步由当前 `spmd_types`/collective 实现决定，不能沿用旧页的 Ring-ACT 时序作现状断言。

但 `_context_parallel_shard` **没有消失**。它仍从 PyTorch 的 `torch.distributed.tensor.experimental._attention` 导入，与 `_HeadTailLoadBalancer`、`_PTRRLoadBalancer` 一起负责输入与 `BlockMask` 的物理切分（`torchtitan/distributed/context_parallel/api.py:9-18`、`torchtitan/distributed/context_parallel/api.py:229-269`）。失效的是旧 TorchTitan 文件路径和 forward wrapper，不是输入 sharder 本身。

---

## 2. 当前 CP 的三层职责

| 层 | 当前职责 | 关键证据 |
|---|---|---|
| 输入层 | 根据模型声明的 `SpmdType` 推导每个输入的 CP shard 维；用同一个 balancer 重排并切分 tensor 与 mask | `torchtitan/distributed/context_parallel/api.py:40-51`、`torchtitan/distributed/context_parallel/api.py:54-122` |
| 模块边界 | attention 声明 Q/K/V 的 source/destination、输出及输入梯度布局；框架据此插入 redistribution 与 local region | `torchtitan/models/common/decoder_sharding.py:275-314` |
| 运行时 | 在 dense `[dp, cp, tp]` mesh 上解释布局并执行单轴 collective；进入 sparse expert 区域时另切 sparse mesh | `torchtitan/distributed/parallel_dims.py:229-253`、`torchtitan/distributed/spmd_types.py:108-176` |

这三层刻意分开：输入切分发生在模型 forward 之前，attention 内部只消费已经带有类型语义的本地 tensor，而参数存储仍由另一张 FSDP mesh 管理。这样 CP 不需要在每个 attention kernel 外手写一份专用 wrapper。

---

## 3. 从配置到一次 forward 的真实调用链

### 3.1 配置与建模期

1. `ParallelismConfig.spmd_backend` 默认是 `spmd_types`；CP 度数、load balancer 与 PTRR mask key 都是公共配置（`torchtitan/config/configs.py:168-180`、`torchtitan/config/configs.py:241-258`）。
2. Trainer 先构造 `ParallelDims`，随后设置全局 SPMD 后端，再调用模型配置的 `update_from_config()`（`torchtitan/trainer.py:290-309`、`torchtitan/trainer.py:334-340`）。
3. 以 Llama 3 为例，`update_from_config()` 先执行 Decoder 的 CP/backend 校验，再给所有子模块写入 sharding config（`torchtitan/models/llama3/model.py:69-83`）；每层 attention 都安装共同的 inner-attention local-map 契约（`torchtitan/models/llama3/sharding.py:57-67`）。
4. `parallelize_llama()` 在默认后端下无条件调用 `model.parallelize(parallel_dims)`（`torchtitan/models/llama3/parallelize.py:40-42`）。递归 parallelize 对每个有 `ShardingConfig` 的模块执行“分布状态 → 输入重分布 → 可选 local region → forward → 输出重分布”（`torchtitan/protocols/module.py:244-290`）。

### 3.2 每个 microbatch

```text
Trainer.forward_backward_step
  -> model.preprocess_inputs
       -> 构造 Flex/Varlen mask
       -> prepare_context_parallel_input
            -> 从 input SpmdType 推导 CP shard dim
            -> cp_shard(tensors, masks, configured balancer)
       -> annotate_input_spmd_types
  -> train_context 激活 dense [dp, cp, tp] mesh
  -> model forward
       -> attention boundary: K/V CP S -> R，Q 保持 CP S
       -> local attention kernel
       -> output 保持 Q 的 CP shard
```

Decoder 的实际预处理顺序是“先建 mask，再 CP shard，最后注入 SPMD 类型”（`torchtitan/models/common/decoder.py:351-386`）。Trainer 在非 PP 路径先调用这个入口，再在 `train_context` 中执行模型 forward（`torchtitan/trainer.py:688-711`）；`get_spmd_context()` 注册 dense/sparse mesh，并把 dense mesh 压入当前运行时栈（`torchtitan/distributed/utils.py:397-423`）。

---

## 4. 输入布局：由 `SpmdType` 决定“切哪一维”

### 4.1 折叠后的语言模型输入

当前 decoder 输入不是旧页假设的统一 `[B,S]`。默认 token ID 是一维 `(tokens,)`，其 `SpmdType` 把 DP 与 CP 都放在 token dim 0；labels 使用同一 `PartitionSpec((DP, CP))`，但在 TP 上标为 input-only（`torchtitan/models/common/decoder_sharding.py:64-73`、`torchtitan/models/common/decoder_sharding.py:110-119`）。

`prepare_context_parallel_input()` 不再硬编码 `inputs/labels/positions` 的同一个维度：

- `_cp_shard_dims()` 解析每个布局的 CP 轴，只收集 CP 为 `Shard` 的输入；CP 为 replicate/partial 的输入保持不动（`torchtitan/distributed/context_parallel/api.py:40-51`）。
- 入口只切实际存在且为 tensor 的命名字段，再把结果写回原 dict；无可切字段时直接返回（`torchtitan/distributed/context_parallel/api.py:93-122`）。
- 类型注入要求每个顶层 tensor 都有布局，缺项会显式报错；容器内 tensor 需要在构造处单独注解（`torchtitan/distributed/spmd_types.py:195-228`）。

### 4.2 `BlockMask` 是独立的 Q 维切分

所有输入 tensor 共用一次选定的 balancer；`BlockMask` 随后使用同一 balancer，但固定切 `[B,H,Q,KV]` 的 Q dim 2，KV 维不切（`torchtitan/distributed/context_parallel/api.py:229-261`）。如果 mask 是 dict，函数保持原 key 顺序并逐一重建 sharded dict（`torchtitan/distributed/context_parallel/api.py:242-269`）。

### 4.3 Flux 仍保留专用入口

包注释明确：`prepare_context_parallel_input` 是主 API，而 `cp_shard` 仍供输入形态不同的 Flux 使用（`torchtitan/distributed/context_parallel/__init__.py:7-16`）。Flux 的序列 tensor 是 `[B,L,...]`，训练和验证都显式用 `input_seq_dims=1`，没有 attention mask，并把 load balancer 设为 `None`（`torchtitan/models/flux/trainer.py:228-248`、`torchtitan/models/flux/validate.py:272-288`）。切分后，Flux 再把这些序列输入注解为 DP shard dim 0、CP shard dim 1（`torchtitan/models/flux/sharding.py:81-106`）。

这就是两个输入入口尚未统一的原因：语言模型依据命名 `SpmdType` 处理折叠 token 与 `BlockMask`，Flux 则同时处理多条 `[B,L,...]` 图像/文本序列且没有 mask。源码也把“未来让主 API 覆盖 Flux”留作 TODO（`torchtitan/distributed/context_parallel/__init__.py:10-13`）。

---

## 5. 当前 attention 边界：Q 保持 shard，K/V 在 kernel 前 replicate

### 5.1 语言模型 FlexAttention

decoder inner attention 的局部张量布局为 `(T,N,H)`：DP/CP 共同切 T，TP 切 N（`torchtitan/models/common/decoder_sharding.py:76-95`）。`set_gqa_inner_attention_local_map()` 声明：

- Q source/destination 都是 CP token shard；
- K/V source 是 CP token shard，destination 是 CP replicate；
- 输出 source 与 Q 相同，继续保持 CP token shard；
- 配置还把 local region 的 K/V input-gradient placement 记为 CP partial（`torchtitan/models/common/decoder_sharding.py:275-314`）。

在 `spmd_types` 后端，模块 wrapper 先检查 source layout，再对 source/destination 不同的轴调用单轴 `spmd.redistribute`（`torchtitan/protocols/module.py:591-621`）。因此当前 Flex CP 的可核验通信边界是 **K/V 的 CP shard → replicate**；Q 与输出不做 CP all-gather。需要注意，`spmd_types` 的 local-region wrapper只消费 in/out types，并不读取 `LocalMapConfig.in_grad_placements`；不能把旧 DTensor local-map 的显式 gradient placement 接线直接套到现状（`torchtitan/protocols/module.py:539-557`）。Flex kernel 只在 local region 内把 `(T,N,H)` 适配为 `[1,N,T,H]`，输出再回到 `(T,N,H)`（`torchtitan/models/common/attention.py:326-377`）。

### 5.2 Flux SDPA

Flux 仍直接构造 `ScaledDotProductAttention`，并以 `is_causal=False` 调用（`torchtitan/models/flux/model/layers.py:145-170`）。它的 local-map 契约与 decoder 同构，但维度不同：Q/K/V 是 `[B,L,N,H]`，CP 切 L(dim 1)；K/V destination 为 CP replicate，输出保持 Q layout，配置同样保留 K/V input-gradient 的 CP-partial placement（`torchtitan/models/flux/sharding.py:28-61`）。SDPA 在 kernel 边界把 `[B,L,N,H]` 转成 `[B,N,L,H]`，且只接受 causal bool、不接受 attention mask（`torchtitan/models/common/attention.py:380-441`）。

### 5.3 “SDPA + HeadTail / FlexAttention + PTRR”不再是自动分支

配置文档仍把 `headtail` 标成 SDPA 用、`ptrr` 标成 FlexAttention 用（`torchtitan/config/configs.py:241-258`），但 `cp_shard()` 实际只按字符串选择 balancer，并不检查 attention 实例（`torchtitan/distributed/context_parallel/api.py:180-227`）。当前真实接线是：

| 路径 | 输入布局 | mask | 实际 balancer 接线 | attention 边界输出 |
|---|---|---|---|---|
| decoder + FlexAttention | 折叠 `(T,)` 输入在 dim 0 切 CP；Q/K/V 为 `(T,N,H)` | `BlockMask` 或 `dict[str, BlockMask]`，Q dim 2 切分 | 公共配置，默认仍是 `headtail`；需要基于稀疏 mask 计量时显式选 `ptrr` | `(T,N,H)`，保持 Q 的 CP token shard |
| Flux + SDPA | `[B,L,...]` 输入在 dim 1 切 CP；Q/K/V 为 `[B,L,N,H]` | 无 | 训练/验证显式 `None`，当前并未接 HeadTail | `[B,L,N,H]`，保持 Q 的 CP sequence shard |

标准语言模型已经拒绝 `sdpa` backend，因为它不能表达按 document positions 生成的 packed mask；它只留给直接构造 SDPA 的 Flux（`torchtitan/models/common/config_utils.py:71-103`）。Decoder 的 CP 校验还会拒绝 `ScaledDotProductAttention` 与 `VarlenAttention`，只允许 FlexAttention CP（`torchtitan/models/common/decoder.py:178-188`）。所以旧页的“SDPA Ring 与 Flex all-gather 两个并列语言模型分支”必须删除；当前 SDPA CP 是 Flux 的无 mask、无 balancer 路径。

### 5.4 PTRR 的 dict mask key

`_PTRRLoadBalancer` 只能从一个 `BlockMask` 构造。若 `attention_masks` 是 dict，调用者必须提供 `context_parallel_ptrr_mask_key`；缺 key、key 不存在、值不是 `BlockMask` 都会在切分前报错。选中的一个 mask 只负责构造 balancer，该 balancer随后应用到所有输入和 dict 中的每个 mask（`torchtitan/distributed/context_parallel/api.py:188-227`）。

这不是空配置：GPT-OSS 的 Flex 路径会生成 `basic_mask` 与可选的 `sliding_window_mask`，每层按自己的 key 取 mask（`torchtitan/models/gpt_oss/model.py:143-182`、`torchtitan/models/gpt_oss/model.py:235-268`）；集成 recipe 显式把 PTRR 基准 mask 选为 `basic_mask`（`torchtitan_recipes/tests/models.py:254-266`）。这里无法把多个 mask 各自独立优化，因为输入只能采用一套共同重排；key 的作用正是选定这套共同顺序的计量基准。

---

## 6. `[dp, cp, tp]` dense mesh：CP 同时参与值布局与参数存储

默认 `spmd_types` 后端为同一批 rank 建两张 dense 视图：FSDP storage mesh 是 `[pp,dp_replicate,dp_shard,cp,tp]`，前后向类型 mesh 是 `[dp,cp,tp]`，其中逻辑 `dp = dp_replicate * dp_shard`（`torchtitan/distributed/parallel_dims.py:229-253`）。

CP 在两张视图中扮演不同角色：

- **值布局平面**：decoder 激活用 `PartitionSpec((DP,CP),TP,...)`，表示 DP 与 CP 依次切 token，TP 切 feature/head（`torchtitan/models/common/decoder_sharding.py:39-61`、`torchtitan/models/common/decoder_sharding.py:76-95`）。
- **参数存储平面**：dense 参数的逻辑 CP type 是 replicate，但 FSDP storage resolver 会把启用的 CP 与 `dp_shard` 一起作为 FSDP shard axes；`dp_replicate` 仍是 replicate axis（`torchtitan/models/common/decoder_sharding.py:24-36`、`torchtitan/distributed/fsdp.py:28-62`）。

这不是矛盾：前者回答“forward 值在逻辑 CP 轴上怎样分布”，后者回答“参数状态由哪些 rank 分片保存”。把两者混成一张 mesh，会让 CP activation shard 与 FSDP parameter shard 难以分别表达。

---

## 7. 与 FSDP、TP、PP、EP/MinimalAsyncEP 的组合证据

### 7.1 组合不是靠隐式猜测，而是配置验证 + 集成矩阵

- **FSDP/HSDP + CP**：测试矩阵覆盖 FSDP+CP、无 `dp_shard` 的 DDP/HSDP+CP、带 `dp_shard` 的 HSDP+CP；相应 recipe 都显式使用 `spmd_types` 与 CP2（`torchtitan_recipes/tests/features.py:292-318`、`tests/integration_tests/features.py:208-230`）。
- **TP + CP**：同一矩阵覆盖 FSDP+TP+CP；布局上 TP 切 head/feature、CP 切 token，二者在 `PartitionSpec((DP,CP),TP,...)` 中是不同维（`torchtitan/models/common/decoder_sharding.py:76-95`、`tests/integration_tests/models.py:20-29`）。
- **PP + CP**：PP 路径对每个 microbatch 先执行 `preprocess_inputs()`，再把 sharded args/kwargs 交给 schedule（`torchtitan/trainer.py:737-766`）；验证 recipe 覆盖 TP2+CP2+PP2（`torchtitan_recipes/tests/features.py:357-366`）。
- **EP + CP**：dense region 满足 `dp_shard * cp * tp == efsdp * ep`，EP 不是额外乘在 world-size 上，而是对同一组 rank 重切 sparse mesh（`torchtitan/config/configs.py:284-291`）。集成矩阵覆盖 DeepSeek V3 的 FSDP+CP+PP+EP（`torchtitan_recipes/tests/models.py:91-102`、`tests/integration_tests/models.py:57-66`）。
- **MinimalAsyncEP + CP**：H100 recipe 显式配置 FSDP2+CP2+TP2+EP8、`spmd_types` 与 full activation checkpoint；测试条目命名为 FSDP+CP+TP+MinimalAsyncEP（`torchtitan_recipes/tests/h100.py:63-77`、`tests/integration_tests/h100.py:57-62`）。MinimalAsyncEP 自身还要求 EP>1、expert 数可被 EP 整除并启用 full recompute（`torchtitan/distributed/minimal_async_ep/api.py:84-140`）。

这些条目证明仓库把相应组合作为持续验证目标，但不意味着任意 attention/backend/compile 组合都被支持；下面的失败边界仍然优先。

---

## 8. 限制与失败边界

| 边界 | 当前行为 | 源码 |
|---|---|---|
| `partial_dtensor + CP` | 配置更新阶段直接拒绝；`cp=1` 时 partial backend 仍合法 | `tests/unit_tests/cpu/test_context_parallel_validation.py:15-31` |
| decoder SDPA/Varlen + CP | 显式 `NotImplementedError`；FlexAttention 是当前 decoder CP 路径 | `torchtitan/models/common/decoder.py:178-188` |
| PTRR 无 mask / dict 无 key / key 无效 | 在构造 balancer 前 `ValueError` | `torchtitan/distributed/context_parallel/api.py:188-227` |
| mask dict 含非 `BlockMask` | 切分阶段拒绝 | `torchtitan/distributed/context_parallel/api.py:239-255` |
| token 数整除 | 通用 Trainer 要求每个 PP microbatch token 数可被启用的 TP-SP 与 `2*cp` 乘积整除 | `torchtitan/trainer.py:292-305` |
| SPMD 多轴重分布 | 当前 helper 只允许一次单轴布局变化；多轴变化与 shard-order 重排会拒绝 | `torchtitan/distributed/spmd_types.py:256-269`、`torchtitan/distributed/spmd_types.py:293-380` |
| graph_trainer CP | 集成 CP 全局禁用，等待采用 `spmd_types`；Flex regional compile 另有 load-balancer trace 问题 | `torchtitan/experiments/graph_trainer/tests/integration_tests.py:20-28` |
| CooR precompile + CP | 明确抛 `NotImplementedError`，因为尚未复用输入 CP shard 路径 | `torchtitan/experiments/graph_trainer/precompile_main.py:243-250` |

`2*cp` 整除检查当前不读取 `context_parallel_load_balancer`。因此即便调用者选择 PTRR 或 `None`，通用 Trainer 仍施加这一保守约束；不能把它解释成“只有 HeadTail 才会检查”。

---

## 9. 小结

- 当前 CP 的核心抽象是 `SpmdType + ShardingConfig + dense runtime mesh`，不是 attention 类型驱动的 forward monkey patch。
- `_context_parallel_shard` 仍负责输入与 mask 的物理切分；旧路径失效的是单文件位置、`apply_cp_to_forward`、SDPA Ring dispatcher 与基于它的 ACT 时序。
- decoder/Flex 与 Flux/SDPA 在 attention 边界采用同一种 Q-shard、KV-replicate、output-shard 契约，但输入形状、mask 能力和 balancer 接线不同，所以仍保留两个输入适配入口。
- `headtail`/`ptrr` 是用户选择的输入重排策略，不再按 attention 类型自动分派；当前标准语言模型 CP 只支持 Flex，而 Flux SDPA 显式关闭 load balancing。
- CP 同时进入 `[dp,cp,tp]` 前后向布局平面与 FSDP storage shard axes；与 TP/PP/EP/MinimalAsyncEP 的组合由 mesh 约束、配置检查和集成 recipe 共同限定。

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 本系列入口、页面边界与统一源码基线。
- [[10_torchtitan_parallel_dims_analysis|ParallelDims 与双平面 DeviceMesh]] —— 解释 `[dp,cp,tp]` 前后向 mesh 与 FSDP storage mesh 的关系。
- [[12_torchtitan_tp_analysis|TorchTitan TP 分析]] —— CP token shard 与 TP head/feature shard 如何在同一 `PartitionSpec` 中组合。
- [[15_torchtitan_ep_analysis|TorchTitan EP 分析]] —— dense/sparse mesh 重切分及 CP 与 EP 的等积约束。
- [[16_torchtitan_spmd_types_analysis|TorchTitan SPMD Types 分析]] —— `SpmdType`、单轴 redistribution、local region 与运行时类型检查的完整机制。
- [[24_torchtitan_comm_optimizations_overlap_analysis|TorchTitan 通信优化与重叠]] —— 当前 collective/compile 优化边界以及不应沿用旧 ACT 叙事的背景。
- [[01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis|Ring Attention 与上下文并行理论页]] —— HeadTail、PTRR、Ring、online softmax 与通信量的通用理论权威页。
