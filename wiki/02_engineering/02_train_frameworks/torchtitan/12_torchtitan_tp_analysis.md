# 张量并行 TP：ShardingConfig 驱动的布局协议与两类重叠路径

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-27）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **主线**：当前 TP 的权威入口不是一张手写 `ParallelStyle` plan，而是一套由 `ShardingConfig` 声明、以 `SpmdType` 表达的参数与激活布局协议。协议同时驱动默认的“本地 tensor + 类型 + 显式 collective”路径和兼容的 DTensor 路径；Sequence Parallel、Loss Parallel、compile Async TP 与 dist-GEMM 都是在这套布局边界上选择“何时物化哪一次 collective”。
>
> 主要源文件：`torchtitan/protocols/{sharding,module}.py`、`torchtitan/distributed/{spmd_types,compile,linear}.py`、`torchtitan/models/common/{decoder_sharding,dist_gemm}.py`、`torchtitan/components/loss.py`。

---

## 1. 先纠正旧基线

旧页基于 `cf3c4312`，其中六组主张已经被当前源码替代：

> [!deprecated] “手写 `parallelize_module` 与配置式 TP 是两条并存主线”已失效
> 当前模型目录已把 `model.parallelize(parallel_dims)` 定义为自动递归的声明式入口；Llama 只要使用 `spmd_types`，或确实启用了 TP，就走这一入口（`torchtitan/models/README.md:34`；`torchtitan/models/llama3/parallelize.py:23-44`）。`torchtitan/distributed/tensor_parallel.py` 当前只剩给特殊复制模块使用的 `NoParallel`，不再承载 Llama 的 colwise/rowwise 主计划（`torchtitan/distributed/tensor_parallel.py:19-30`）。

> [!deprecated] “两条路径底层都把激活变成 DTensor 再 `DTensor.redistribute`”已失效
> 默认 `spmd_types` 路径分片后仍注册普通 `torch.Tensor`，用 `spmd.assert_type` 绑定布局；边界通信直接调用 `spmd.redistribute`。只有 `partial_dtensor` 路径用 `distribute_tensor`、`DTensor.from_local` 与 `DTensor.redistribute`（`torchtitan/protocols/module.py:292-325`、`398-420`、`597-653`）。

> [!deprecated] 旧版 SP 的 `[B,S,D]`/`Shard(1)` 图已失效
> 当前 dense decoder 已折叠 batch/sequence 为 token 维：SP 激活是 `(tokens, hidden)`，TP 加入 token 维的 `PartitionSpec((DP, CP, TP), None)`，即在 TP 上切第 0 维（`torchtitan/models/common/decoder_sharding.py:98-107`）。

> [!deprecated] Async TP 不再由 `ParallelismConfig.enable_async_tensor_parallel` 和 `maybe_enable_async_tp()` 接线
> 提交 `737594746` 将入口移到 `CompileConfig.enable_async_tensor_parallel`；当前合法条件是同时开启 compile 且 `compile.components` 包含 `model`，真正配置发生在 `apply_compile()` 内（`torchtitan/config/configs.py:295-315`；`torchtitan/distributed/compile.py:39-52`）。

> [!deprecated] “Loss Parallel 通过 PyTorch `loss_parallel()` context manager”已失效
> 当前 TorchTitan 自己的 `_LossParallelCrossEntropy` 是显式、可注册进 `spmd_types` 的 autograd Function；默认路径直接在 plain local vocab shard 上运行，不依赖 DTensor context manager（`torchtitan/components/loss.py:32-79`）。

旧页的 `ColwiseParallelWithGradPlacement` 与 `ShardingConfig.local_input_grad_placements` 也不再是当前协议。现在只有 `LocalMapConfig.in_grad_placements` 用来声明进入本地 kernel 区域的输入梯度布局（`torchtitan/protocols/sharding.py:33-53`），一般输入/输出边界则由 `spmd.redistribute` 或 DTensor redistribution 负责。

---

## 2. 一套声明，两种运行时表示

### 2.1 配置先声明布局，模型实例再物化

Llama 的真实接线是：

```text
Llama3Model.Config.update_from_config
  -> set_llama3_sharding_config(enable_sp)
     root: embedding / final norm / lm_head
     layer: norms / GQA / inner attention / FFN
  -> parallelize_llama
     -> model.parallelize(parallel_dims)
        -> 递归处理每个 Module 的 ShardingConfig
        -> 分片 state
        -> forward = redistribute inputs -> [local region] -> compute
                     -> redistribute outputs
```

配置更新在构模前把 sharding 写入每个子 Config（`torchtitan/models/llama3/model.py:69-83`；`torchtitan/models/llama3/sharding.py:25-41`），`parallelize_llama` 再调用实例的递归入口（`torchtitan/models/llama3/parallelize.py:23-55`）。`Module.parallelize()` 先递归子模块、校验 redistribution、分片 state，最后包裹 forward；FSDP 的 `__call__` hooks 位于这个 wrapped forward 之外（`torchtitan/protocols/module.py:244-290`）。

### 2.2 `ShardingConfig` 是协议，不是某个后端的参数表

所有布局都统一写成按命名 mesh 轴索引的 `SpmdType`。它同时描述：

| 协议字段 | 负责编码的边界 |
|---|---|
| `state_shardings` | 参数/缓冲区在 DP、CP、TP 等轴上的布局 |
| `in_src_shardings -> in_dst_shardings` | 模块入口已有什么布局、计算前要变成什么布局 |
| `out_src_shardings -> out_dst_shardings` | 本地计算产生什么布局、交给下游前要变成什么布局 |
| `local_map` | 哪一段关闭分布式算子传播、按本地 tensor kernel 计算 |

字段的公共契约与示例集中在 `torchtitan/protocols/sharding.py:59-113`。显式写出 src 并非重复：默认后端的 plain tensor 本身没有 DTensor placements，必须先断言“它现在是什么”才能安全通信（`torchtitan/protocols/module.py:597-620`）。

### 2.3 默认 `spmd_types` 与兼容 `partial_dtensor`

配置只保留两个后端且默认 `spmd_types`（`torchtitan/config/configs.py:168-180`）：

| 阶段 | `spmd_types`（默认） | `partial_dtensor`（兼容） |
|---|---|---|
| 参数物化 | `spmd.shard` 逐轴切本地 tensor；参数注册后仍是 plain tensor | `distribute_tensor` 产生 DTensor |
| 激活布局 | `SpmdType` + `spmd.assert_type`，值本身是 plain tensor | DTensor 的 mesh + placements |
| 边界通信 | `spmd.redistribute(x, process_group, src, dst)` | `DTensor.redistribute(..., async_op=True)` |
| 可见逻辑轴 | DP/CP/TP/EP 都在类型平面内 | 只把 TP/EP 留在 DTensor 带内 |

默认路径的 state 分片会按 `PartitionSpec` 顺序逐轴调用 `spmd.shard`（`torchtitan/distributed/spmd_types.py:440-489`）；兼容路径则解析具体 placements 后调用 `distribute_tensor`（`torchtitan/protocols/module.py:398-420`）。两者选择的 mesh 轴集合也不同：`resolve_mesh()` 对 `spmd_types` 保留 DP/CP/TP/EP，对 `partial_dtensor` 只保留 TP/EP（`torchtitan/distributed/parallel_dims.py:461-482`）。因此“两后端”是**同一声明协议的两种运行时表示**，不是两套模型 TP 计划。

### 2.4 默认 redistribution 目前刻意只做单轴变化

`spmd_validate_redistributions()` 会拒绝一次边界改变多个 mesh 轴、涉及 `V` 语义的变化，以及改变同一 tensor 维上 shard 顺序的重排（`torchtitan/distributed/spmd_types.py:256-269`、`293-380`）。通过校验后，helper 才逐轴比较 src/dst，对真正变化且 size > 1 的轴调用一次 `spmd.redistribute`；其目标是让每个配置边界至多对应一个 all-gather、reduce-scatter 或 all-reduce（`torchtitan/distributed/spmd_types.py:398-437`）。

这样做牺牲了任意 DTensor-style 重排的表达力，却让模块边界的通信选择保持可审计；更复杂的多轴语义必须写成显式 collective，而不是依赖隐式顺序。

---

## 3. 参数布局：colwise/rowwise 配对仍是核心，但存储形态变了

`dense_param_placement()` 固定 DP/CP 为复制，调用者只决定 TP 类型（`torchtitan/models/common/decoder_sharding.py:24-36`）。在线性层的实际存储形状 `weight=[out_features,in_features]` 上：

| 角色 | 参数布局 | 激活布局与结果 | 当前配置 |
|---|---|---|---|
| colwise | weight/bias 在 TP 上 `S(0)`，切输出特征 | 输入完整；输出沿最后特征维分片 | `torchtitan/models/common/decoder_sharding.py:122-130` |
| rowwise | weight 在 TP 上 `S(1)`，bias 复制 | 输入特征分片；本地 GEMM 先产生 `P` 部分和 | `torchtitan/models/common/decoder_sharding.py:133-150` |

GQA 的 `wqkv`（或 `wq`/`wkv`）用 colwise，`wo` 用 rowwise（`torchtitan/models/common/decoder_sharding.py:194-205`、`257-272`）；FFN 的 `w1/w3` 用 colwise，`w2` 用 rowwise（`torchtitan/models/common/decoder_sharding.py:317-350`）。inner attention 的 `(tokens, heads, head_dim)` 布局让 TP 切 heads 维而不是 head_dim（`torchtitan/models/common/decoder_sharding.py:76-95`、`275-314`）。这保持了经典配对不变量：colwise 的 feature shard 正好是 rowwise 的输入 shard，中间的 attention/SiLU/乘法不需要先拼回完整 hidden。

根层同样遵守该协议：token embedding weight 沿 vocab 维 `S(0)`，其查表局部结果是 partial；`lm_head` weight 与 logits 也沿 vocab/output 维 `S(0)`/`S(-1)`（`torchtitan/models/common/decoder_sharding.py:353-388`）。差别只在物理表示：默认后端把这些参数物化为本地 shard，兼容后端把它们物化为 DTensor。

---

## 4. Sequence Parallel：collective 由模块边界的 src/dst 决定

SP 当前默认开启（`torchtitan/config/configs.py:168-172`）。在折叠后的二维激活 `(tokens, hidden)` 上，普通 TP 区域与 SP 区域的布局是：

```text
非 SP：token 维只按 DP/CP 切；TP 上为 I/R，rank 持有完整 token 行
SP：   token 维按 DP -> CP -> TP 依次切；TP 上是 S(0)
```

具体 `PartitionSpec` 位于 `dense_activation_placement()` 与 `dense_sequence_parallel_placement()`（`torchtitan/models/common/decoder_sharding.py:39-61`、`98-107`）。一个 stock Transformer block 的真实前向边界是：

```text
x: TP S(0)                         # SP token shard
  -> attention.in  S(0) -> R       # all-gather token 行
  -> qkv colwise                   # 输出 feature/head shard
  -> inner attention local region
  -> wo rowwise: P -> S(0)         # reduce-scatter，完成部分和并恢复 SP
  -> residual                      # 两边都是 S(0)，无 TP 通信
  -> FFN.in S(0) -> R              # all-gather
  -> w1/w3 colwise -> SwiGLU -> w2 rowwise
  -> P -> S(0)                     # reduce-scatter
```

attention 的入口 all-gather 与 `wo` 输出 reduce-scatter 由 `set_gqa_attention_sharding()` 明确声明（`torchtitan/models/common/decoder_sharding.py:234-272`）；FFN 的同构边界在 `set_dense_ffn_sharding()`（`torchtitan/models/common/decoder_sharding.py:317-350`）。如果关闭 SP，rowwise 的目标从 sequence shard 改成 `I`，于是 `P -> I` 是 all-reduce（`torchtitan/models/common/decoder_sharding.py:133-150`；`torchtitan/models/llama3/sharding.py:44-74`）。

final norm 是另一个易漏边界：block 可持续输出 SP shard，但 `pre_lm_head_norm_config()` 会在 norm 后把它恢复为 TP-replicated hidden，再交给模型输出或 chunked lm-head/loss（`torchtitan/models/common/decoder_sharding.py:171-191`）。另外，运行配置必须让序列长度可被 TP degree 整除；若同时启用默认负载均衡 CP，则总 divisor 是 `tp * (2 * cp)`（`torchtitan/distributed/parallel_dims.py:601-609`）。

---

## 5. Loss Parallel：不 gather 完整 vocab logits

`lm_head` 的输出保持 vocab 维分片（`torchtitan/models/common/decoder_sharding.py:382-388`）。`cross_entropy_loss()` 随后按后端分流：兼容路径识别 `DTensor + Shard(1)` 后取 local shard；默认路径在 TP mesh size > 1 时直接把 plain local logits、当前 TP group 与全局 vocab size 交给同一个 `_LossParallelCrossEntropy`（`torchtitan/components/loss.py:32-62`）。

这个 autograd Function 的前向没有物化 `[tokens, global_vocab]`：

1. 本地 vocab shard 求每个 token 的 max，再 TP all-reduce(MAX)。
2. 用全局 max 做稳定化，本地求 sum-exp，再 TP all-reduce(SUM)。
3. 只在拥有 target id 的 rank gather 对应 log-prob，其余 rank 置零，再 TP all-reduce(SUM)。

实现与 label 有效域检查位于 `torchtitan/components/loss.py:127-180`。反向直接在本地 `log_probs` 上融合 NLL 与 log-softmax 梯度，没有 collective（`torchtitan/components/loss.py:186-224`）。所以当前 Loss Parallel 的关键不再是“进入某个 context”，而是**保持 vocab shard 到 CE 内部，只规约 `[tokens,1]`/`[tokens]` 级中间量**。

一个重要边界是：CE 内核本身按 ceil-chunk 计算 shard 范围，能够处理最后一片较短的 uneven vocab（`torchtitan/components/loss.py:127-143`）；但默认 `spmd_types` 的参数物化层会在 parallelize 时拒绝任何不能被 TP/EP degree 整除的参数维（`torchtitan/protocols/module.py:327-364`）。因此“loss kernel 支持 uneven”不等于“标准 `lm_head` 参数现在允许 uneven TP”。

---

## 6. compile Async TP：入口已经属于 compile

当前启用方式是 `CompileConfig.enable_async_tensor_parallel=True`，并且必须同时满足：

- `compile.enable=True`；
- `"model" in compile.components`。

配置构造期就会拒绝不满足前提的组合（`torchtitan/config/configs.py:295-315`）。Llama 在 activation checkpoint wrapper 之后、FSDP 之前调用 `apply_compile()`（`torchtitan/models/llama3/parallelize.py:42-68`），该函数先从 `ParallelDims.get_dense_tp_mesh()` 取得**当前后端真正用于 dense forward/backward 的 TP group**，而不是自行猜 group（`torchtitan/distributed/compile.py:39-52`；`torchtitan/distributed/parallel_dims.py:425-441`）。

`_maybe_enable_async_tp()` 随后：

1. 取该 TP group 的 `group_name`；
2. 为这个精确 group 注册 symmetric memory；
3. 打开 `torch._inductor.config._micro_pipeline_tp`；
4. 对每个 TransformerBlock 执行 fullgraph compile。

调用点在 `torchtitan/distributed/compile.py:75-96`，逐 block compile 在 `torchtitan/distributed/compile.py:66-72`。因此旧的“parallelism 开关在 TP 初始化阶段打开 Inductor pass”已经不成立：**只有模型 compile 路径拥有并配置这项优化**。源码把它定义为“让 TP collective 与矩阵乘流水化”，但具体哪些 pattern 被当前 PyTorch/Inductor 版本接住，不应再从旧 PyTorch 2.9.1 行号外推。

---

## 7. dist-GEMM：显式把 collective 折进 GEMM 的新增机制

dist-GEMM 是另一条、与 compiler graph pass 不同的重叠入口。模型构造时把 `tp_gemm_backend="dist_gemm"` 传给 GQA/FFN config 工厂，工厂直接把 stock projection 类替换为 `AllGatherFusedQKVLinear`、`RowParallelLinear` 和 `DistGEMMFeedForward`；attention 还强制要求 fused QKV（`torchtitan/models/common/config_utils.py:190-229`、`281-306`）。

它的三段机制是：

| 投影 | 显式融合路径 | 结果布局 |
|---|---|---|
| QKV colwise | `AllGatherLinear`: sequence all-gather + QKV GEMM | full tokens、feature/head shard |
| attention `wo` / FFN `w2` rowwise | `LinearReduceScatter`: local partial GEMM + reduce-scatter | sequence shard、完整 hidden |
| FFN `w1/w3` | `AllGatherLinearMulti`:一次 all-gather 同时喂两个 GEMM | 两个 feature shard |

模型层的调用位于 `torchtitan/models/common/dist_gemm.py:113-175`、`178-231`。底层 autograd Functions 直接调用 symmetric-memory fused ops：`AllGatherLinear` 的 forward 与 dgrad 分别使用 fused all-gather-matmul / matmul-reduce-scatter（`torchtitan/distributed/linear.py:110-159`）；`LinearReduceScatter` 的 forward 与 dgrad 做镜像变换（`torchtitan/distributed/linear.py:305-375`、`386-418`）。FFN 两个上投影共享一次 gather，而 backward 把两个 dgrad 拼成一次 product + reduce-scatter（`torchtitan/distributed/linear.py:185-206`、`243-280`）。

这要求 ShardingConfig 不再在模块外重复发通信：dist-GEMM attention 不声明入口 all-gather，rowwise 模块也只保留参数布局，因为 fused op 已直接返回最终 `S(0)`，不会先暴露 `Partial`（`torchtitan/models/common/decoder_sharding.py:226-272`、`329-350`）。这说明它是**新的执行机制**，而不是给旧 `redistribute(async_op=True)` 换一个名字。

当前前提与边界也很具体：

- 只支持 `spmd_types` 且必须开启 SP；校验代码没有 compile 前提（`torchtitan/models/common/dist_gemm.py:88-110`）。
- TP group 在每次 forward 从 current SPMD mesh 解析；TP=1 时保留可运行 fallback，但会警告实际没有 fusion（`torchtitan/models/common/dist_gemm.py:53-85`）。
- symmetric-memory op 是 CUDA-only；同一 group 的 workspace 从 offset 0 复用，所以当前只保证顺序 module forward/autograd 安全，不支持把两个此类 op 人为放到不同 stream 并发（`torchtitan/distributed/linear.py:14-25`）。
- `DistGEMMFeedForward` 不支持 `w1/w3` bias（`torchtitan/models/common/dist_gemm.py:178-208`）。

fused SwiGLU override 没有破坏这条接线。对 `DistGEMMFeedForward.Config` 应用 `dist_gemm_fused_swiglu` 会生成 `DistGEMMFusedSwiGLU`：单个 interleaved `w13` 通过 `AllGatherLinear` 消费 sequence shard，激活后仍用 `LinearReduceScatter` 运行 `w2`（`torchtitan/overrides/fused_swiglu.py:507-558`、`589-597`）。它节省的是 gate/up 的结构与 dgrad cats，不会再省一轮 all-gather，因为未融合版的 `AllGatherLinearMulti` 本来就只 gather 一次。

compile Async TP 与 dist-GEMM 因而应理解为两种调度方式：前者让 Inductor 从编译图中识别并流水化 stock 边界；后者在模块和 autograd Function 中显式调用 fused symmetric-memory ops。源码没有禁止两开，但 dist-GEMM 已移除 stock 边界 collective，因此二者不应被当成可叠加两次的收益；这是从两套当前接线得出的实现推论。

---

## 8. 当前限制与失败边界

| 限制 | 当前行为 | 证据 |
|---|---|---|
| GQA/MHA head 数 | TP>1 时，`n_heads` 和实际 `n_kv_heads` 都必须被 TP degree 整除 | `torchtitan/models/common/decoder.py:190-204` |
| 默认后端参数 uneven shard | 任一 TP/EP 参数维不能整除对应 degree，parallelize 立即报错 | `torchtitan/protocols/module.py:327-364` |
| 配置式 SPMD redistribution | 一次只允许一个轴变化；不能以 `V` 为 src/dst，也不能重排 shard 次序 | `torchtitan/distributed/spmd_types.py:293-380` |
| 输出结构 | 默认后端的通用 output redistribution 目前只支持单 tensor 输出 | `torchtitan/protocols/module.py:660-703` |
| Loss Parallel | global vocab shard 不能为空；label 必须是 ignore index 或合法全局 token id | `torchtitan/components/loss.py:127-151` |
| SP 序列长度 | sequence length 必须满足 TP divisor；与默认 CP balancing 组合时还含 `2*CP` | `torchtitan/distributed/parallel_dims.py:601-609` |
| dist-GEMM | `spmd_types` + SP + CUDA symmetric memory；attention 还要求 fused QKV | `torchtitan/models/common/dist_gemm.py:88-110`；`torchtitan/models/common/config_utils.py:208-229` |
| compile Async TP | 必须 compile model；只开 loss compile 不合法 | `torchtitan/config/configs.py:295-315` |

---

## 9. 小结

- 当前 TP 是一套 `ShardingConfig + SpmdType` 布局协议，不是手写 plan 与配置 plan 并存。
- 默认 `spmd_types` 使用 plain local tensor 与显式 collective；`partial_dtensor` 仍消费同一声明，但把参数/激活表示成 DTensor。
- colwise 输出 shard 与 rowwise 输入 shard 的配对仍是算子内 TP 核心；SP 现在沿折叠 token 维 `S(0)`，在模块边界把 all-reduce 改写为 all-gather + reduce-scatter。
- Loss Parallel 已内建为 TorchTitan 自己的分布式 CE autograd Function；它支持 uneven local logits，但默认参数物化仍拒绝 uneven TP/EP shard。
- Async TP 的配置和 group 注册属于 compile；dist-GEMM 则以显式模块/autograd Function 把 collective 折进 symmetric-memory GEMM，二者是不同调度入口。

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 本系列入口、页面边界与统一源码基线。
- [[10_torchtitan_parallel_dims_analysis]] —— TP 所消费的 dense storage mesh 与 SPMD forward/backward mesh 从何而来。
- [[11_torchtitan_fsdp_analysis]] —— TP 参数 shard 如何与 FSDP 的参数存储/混合精度平面叠加。
- [[13_torchtitan_cp_analysis]] —— token 维同时被 DP/CP/TP 切分时，CP 的布局与 attention 边界如何接续。
- [[15_torchtitan_ep_analysis]] —— 默认后端对 TP/EP 参数 uneven shard 的共同限制，以及 sparse mesh 切换。
- [[16_torchtitan_spmd_types_analysis]] —— `SpmdType`、`PartitionSpec`、`V/I/R/P/S` 与显式 collective 的协议细节。
- [[24_torchtitan_comm_optimizations_overlap_analysis]] —— compile Async TP、dist-GEMM 与其他通信重叠机制的横向对照。
