---
title: "Context Parallel：把序列所有权、attention 边界与参数存储拆开"
---

# Context Parallel：把序列所有权、attention 边界与参数存储拆开

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页论点**：当前 TorchTitan CP 的核心不是在 attention forward 外安装一个 Ring dispatcher，而是把三种所有权拆成三个显式边界：输入预处理决定每个 rank 拥有哪些 query token；attention 的 `ShardingConfig` 让 Q 与输出保持 CP token shard、只把 K/V 重分布为 CP replicate；FSDP storage mesh 则独立决定参数长期存在哪些 rank。这个拆分以 K/V 全序列物化换取普通 FlexAttention/SDPA 内核复用，并以 Q/输出不复制避免全序列冗余计算。
>
> 本页回答：配置与序列除数如何进入输入 sharder；HeadTail/PTRR 是否仍由 attention 类型自动选择；`BlockMask`、Q/K/V 与输出在边界上怎样分布；为什么 CP-only 仍触发 FSDP；以及 CP 与 TP、PP、EP、MinimalAsyncEP 的当前组合和失败矩阵。`ParallelDims` 的 rank/mesh 通则见 [[10_torchtitan_parallel_dims_analysis]]，FSDP/TP/PP/EP/AC 各自内部机制由 [[11_torchtitan_fsdp_analysis]]、[[12_torchtitan_tp_analysis]]、[[14_torchtitan_pp_analysis]]、[[15_torchtitan_ep_analysis]]、[[22_torchtitan_ac_analysis]] 负责。
>
> Ring Attention、online softmax、HeadTail/PTRR 算法本身与通信量属于 [[01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis|Ring Attention 与上下文并行理论页]]；本页只分析 TorchTitan 当前接线。

---

## 1. Overview

长序列训练不能只把 batch 切给更多 rank：一个样本内部的 token 仍会让 attention 激活与计算增长。CP 要把同一序列的 query token 分给不同 rank，同时保证每个 query 仍能访问所需的全局 key/value。旧 TorchTitan 用 attention 类型驱动的 forward wrapper 分派不同 CP 内核；提交 `5dd944e62` 说明维护多条 CP 路径成本过高，因此删除 `apply_cp_to_forward` 与 partial-DTensor CP，只保留 config-based CP。

当前主线是“边界布局而非 kernel dispatcher”：输入先被物理切分并注入 `SpmdType`；模块 wrapper 根据 `ShardingConfig` 只在必要的轴上 redistribution；普通 attention kernel 在 local region 内运行。当前 backend guard 要求 `cp>1` 时必须使用 `spmd_types`（`torchtitan/distributed/context_parallel/api.py:28`、`torchtitan/distributed/context_parallel/api.py:31`）。

| 概念 | 当前所有权/布局 | 不应再采用的旧心智模型 |
|---|---|---|
| 输入 token | DP 与 CP 依次切 token dim；切哪一维来自每个输入的 `SpmdType` | 所有输入都由 Trainer 硬编码切同一维 |
| `BlockMask` | 只切 Q dim 2，KV dim 保持全局 | mask 与 Q/K/V 一起任意切 |
| Q | CP token shard，进入内核前不 all-gather | 每个 rank 重建完整 Q |
| K/V | source 是 CP token shard，inner-attention 前变为 CP replicate | 当前仍由 Ring 内核逐步轮转 KV |
| attention 输出 | 保持 Q 的 CP token shard | kernel 输出完整序列再重新切分 |
| 参数 | 逻辑 CP 类型是 replicate；持久 storage 可由 FSDP 沿 CP 分片 | CP rank 必然各存一份完整参数 |
| load balancer | `cp_shard()` 按配置字符串选择；不检查 attention 类型 | SDPA 自动 HeadTail、Flex 自动 PTRR |

```text
完整 microbatch（每 DP rank）
        |
        | model.preprocess_inputs：建 mask
        v
prepare_context_parallel_input
        |  SpmdType -> 每个输入的 CP shard dim
        |  同一个 balancer 重排并切 inputs + BlockMask.Q
        v
本 rank 的 query/token shard
        |
        | annotate_input_spmd_types
        v
dense current mesh [dp, cp, tp]
        |
        | inner-attention boundary
        | Q: S -> S     K/V: S -> R
        v
普通 FlexAttention / Flux SDPA local kernel
        |
        | output 与 Q 同布局
        v
本 rank 的 output token shard

参数走另一平面：dense storage mesh [dp_replicate, dp_shard, cp, tp]
                    -> FSDP shard axes = dp_shard + cp
```

### Quick Start：最小入口与阅读顺序

对标准 decoder/FlexAttention，最小配置是：

```python
config.parallelism.context_parallel_degree = 2
config.parallelism.spmd_backend = "spmd_types"
config.parallelism.context_parallel_load_balancer = "ptrr"
# 只有 dict[str, BlockMask]（如 GPT-OSS）才需要：
config.parallelism.context_parallel_ptrr_mask_key = "basic_mask"
```

公共配置仍默认 `headtail`，并把 `headtail`/`ptrr` 文档化为 SDPA/Flex 的建议配对；但当前实际选择只取决于 `context_parallel_load_balancer` 字符串（`torchtitan/config/configs.py:241`、`torchtitan/config/configs.py:244`）。选择 `ptrr` 时必须有 `BlockMask`；dict mask 还必须提供有效 key（`torchtitan/distributed/context_parallel/api.py:188`、`torchtitan/distributed/context_parallel/api.py:195`）。

一次非 PP 训练步的可追踪调用链是：

```text
Trainer.__init__
  -> ParallelDims.from_config
  -> 校验 num_tokens_per_pp_microbatch % (TP_if_SP * 2*CP) == 0
  -> model_config.update_from_config
       -> validate_cp_backend
       -> 安装各模块 ShardingConfig
  -> model.parallelize
       -> state distribution + forward redistribution wrapper

Trainer.forward_backward_step
  -> Decoder.preprocess_inputs
       -> get_attention_masks
       -> prepare_context_parallel_input
       -> annotate_input_spmd_types
  -> train_context 激活 dense [dp,cp,tp] mesh
  -> model forward
       -> inner attention: K/V S@CP -> R@CP
       -> local kernel
       -> output S@CP
```

Trainer 构造 `ParallelDims` 后校验实际 PP microbatch token 数，再调用模型配置更新（`torchtitan/trainer.py:289`、`torchtitan/trainer.py:292`、`torchtitan/trainer.py:334`）。非 PP 路径先做 `preprocess_inputs()`，随后才进入 `train_context` 执行 forward/backward（`torchtitan/trainer.py:688`、`torchtitan/trainer.py:703`）。

---

## 2. 路径演进：为什么从 forward wrapper 收敛到布局边界

### ① 背景/问题

CP 最初需要同时适配 SDPA 和 FlexAttention。提交 `1e8f9acd1` 引入 `apply_cp()`：它用 PyTorch `_ContextParallel` 计划包装 attention forward，并为 SDPA 启用 dispatcher；输入另由 `_context_parallel_shard` 与 HeadTail 切分。紧随其后的 `0a2107f98` 又为 Llama 3 增加 FlexCP 与 PTRR。这个结构形成“输入 sharder + attention-specific forward wrapper”的双重接线。

随着 config-based `ShardingConfig` 和 `spmd_types` 成为默认，继续保留 wrapper、partial-DTensor CP 与声明式 CP 意味着同一正确性边界要实现多次。提交 `5dd944e62` 的正文直接给出维护原因：不同 CP paths 太多，只保留 config-based CP，并暂时禁用 GraphTrainer+CP 与 partial-DTensor+CP。

### ② 为什么这么设计

**选中的路线**是把 CP 放在输入类型与模块 src/dst layout 上；**被否掉的替代方案**是继续按 attention 类型 monkey-patch forward。决策准则由删除提交明确给出：减少重复 CP 机制。布局路线还让 module 的通用 wrapper统一执行“输入 redistribution → local region → 输出 redistribution”，而无需 CP 子系统知道每种模型类（`torchtitan/protocols/module.py:244`、`torchtitan/protocols/module.py:279`）。

源码没有宣称新路线在通信量上优于旧 Ring dispatcher。**知识库推断**：它选择的是可维护性与普通 kernel 复用，而不是保证最低 K/V 通信；证据是当前边界明确把 K/V 变成 replicate，而不是保留 shard 交给 kernel 内 CP。

### ③ 实现思路与细节

提交 `547b0b481` 先把单文件 `torchtitan/distributed/context_parallel.py` 迁成 package，提交说明当时是纯重构。当前 package 只导出 `cp_shard`、`prepare_context_parallel_input` 与 `validate_cp_backend`，并把前者标为 Flux 专用、后者标为主 API（`torchtitan/distributed/context_parallel/__init__.py:7`、`torchtitan/distributed/context_parallel/__init__.py:19`）。

旧 wrapper 删除后，当前 CP 的两个执行点是：

1. `prepare_context_parallel_input()` 在 forward 前调用 PyTorch `_context_parallel_shard`，物理切 tensor 与 mask（`torchtitan/distributed/context_parallel/api.py:54`、`torchtitan/distributed/context_parallel/api.py:109`）。
2. `Module.parallelize()` 为带 `ShardingConfig` 的模块安装通用 wrapper，输入 source/destination 不同时调用 `spmd_redistribute_per_axis()`（`torchtitan/protocols/module.py:285`、`torchtitan/protocols/module.py:591`）。

`_context_parallel_shard` 本身没有消失：它与 `_HeadTailLoadBalancer`、`_PTRRLoadBalancer` 仍从 PyTorch experimental attention API 导入（`torchtitan/distributed/context_parallel/api.py:13`）。消失的是 TorchTitan 的 attention forward dispatcher，不是输入物理切分能力。

### ④ 约束/边界

- 当前 `validate_cp_backend()` 在 `cp>1` 且 backend 不是 `spmd_types` 时直接失败；`partial_dtensor` 只在 CP=1 时合法（`torchtitan/distributed/context_parallel/api.py:28`；`tests/unit_tests/cpu/test_context_parallel_validation.py:22`）。
- 不应把旧 dispatcher 中的 `AsyncCollectiveTensor` 或 Ring 时序写成当前 TorchTitan 机制。当前可核验路径只在布局变化时调用单轴 `spmd.redistribute`（`torchtitan/distributed/spmd_types.py:398`、`torchtitan/distributed/spmd_types.py:430`）。
- package commit 曾列出 Ulysses、Varlen all-gather、SelectiveGather 等计划，但当前源码没有这些可选实现；历史计划不能升级为支持矩阵。

### ⑤ 发展趋势（有锚点的推断）

当前 `spmd_redistribute_per_axis()` 的 TODO 要求未来由 `spmd_types` 提供更通用的 DTensor-style redistribute，或在 partial-DTensor 删除后改写为显式 collective（`torchtitan/distributed/spmd_types.py:256`、`torchtitan/distributed/spmd_types.py:265`）。据此只能推断边界 API 仍在收敛；源码没有承诺恢复旧 forward wrapper。

---

## 3. 配置与序列除数：先保证可均分，再谈 kernel

### ① 背景/问题

CP rank 必须得到等长 query shard；默认 HeadTail 还把序列首尾配对，因此不仅需要序列可被 CP 整除，还需要每个 rank 能取得成对分块。TP Sequence Parallel 也会切 token 轴。若等到 attention 内部才发现长度不整除，错误会远离配置入口，并可能在不同 PP microbatch 上表现不同。

### ② 为什么这么设计

**选中的路线**是在 Trainer 初始化期用“实际每个 PP microbatch 的 token 数”做统一整除检查；**替代方案**是只检查配置的最大上下文长度，或让每个 kernel 自行 padding。决策准则是 fail fast，并让 TP-SP 与 CP 对同一 token 轴的切分一次验证。源码注释把 CP 除数写为默认 load balancing 所需的 `2*CP`（`torchtitan/distributed/parallel_dims.py:601`、`torchtitan/distributed/parallel_dims.py:606`）。

### ③ 实现思路与细节

`ParallelDims.from_config()` 把 `context_parallel_degree` 写入 `cp`，dense world 预算要求：

```text
dp_replicate * dp_shard * cp * tp * pp == world_size
```

若 `dp_shard=-1`，先用其他轴反推剩余 degree；随后验证完整乘积。EP 不增加 world-size 乘数，而要求 `ep` 整除 `dp_shard * cp * tp`（`torchtitan/distributed/parallel_dims.py:84`、`torchtitan/distributed/parallel_dims.py:114`、`torchtitan/distributed/parallel_dims.py:118`、`torchtitan/distributed/parallel_dims.py:123`）。

Trainer 的实际 token guard 是：启用 Sequence Parallel 时乘 TP，否则 TP 因子为 1；CP>1 时再乘 `2*CP`。不整除就抛 `ValueError`，错误信息同时标出 sequence/context parallelism（`torchtitan/trainer.py:292`、`torchtitan/trainer.py:296`、`torchtitan/trainer.py:299`）。`ParallelDims.seq_len_divisor` 属性则固定返回 `tp * (2*cp)`，单元测试用 TP4×CP2 固定为 16（`torchtitan/distributed/parallel_dims.py:602`；`tests/unit_tests/cpu/test_parallel_dims.py:221`）。

### ④ 约束/边界

- Trainer guard 不读取 `context_parallel_load_balancer`。即使显式选择 PTRR 或 `None`，只要 CP>1 仍施加 `2*CP`；这是当前保守约束，不能解释成“仅 HeadTail 检查”（`torchtitan/trainer.py:296`）。
- 配置字段只拒绝空字符串；非法非空 balancer 要到 `cp_shard()` 的 match default 才失败（`torchtitan/config/configs.py:261`、`torchtitan/distributed/context_parallel/api.py:223`）。
- 对 EP dispatcher，CP 与 TP/SP 已提前切 token，因此容量按 `CP*TP` 折算；token 数不能被该乘积整除会在 dispatcher 配置阶段失败（`torchtitan/models/common/token_dispatcher.py:1206`、`torchtitan/models/common/token_dispatcher.py:1215`）。

---

## 4. 输入与 load balancer：物理切分来自类型，不来自参数名猜测

### ① 背景/问题

折叠后的 decoder token 是一维 `(tokens,)`，Flux 序列是 `[B,L,...]`，VLM 又同时携带不应沿 CP 切分的图像 tensor。把 `input/labels/positions` 永远按 dim 0 切分，无法覆盖这些形态；如果 tensor 与 `BlockMask` 使用不同重排顺序，query token 还会与自己的 mask 行错位。

### ② 为什么这么设计

**选中的路线**是用每个输入的 `SpmdType` 推导 CP shard dim，再让同一个 balancer 同时处理所有被选 tensor 与 mask；**替代方案**是按字段名硬编码或让每个输入单独平衡。决策准则是保持 token、position、label、mask 的共同全局顺序，并允许未声明 CP shard 的输入保持不动。提交 `e8e39abc6` 的说明明确把“模型返回 input sharding → CP shard → SPMD annotate”定义为新的 post-dataloading 顺序。

### ③ 实现思路与细节

Decoder 默认输入布局把 `input/positions/labels` 的 token dim 声明为 `PartitionSpec((DP,CP))`；labels 在 TP 上是 input-only，而 token/position 在 TP 上 replicate（`torchtitan/models/common/decoder_sharding.py:64`、`torchtitan/models/common/decoder_sharding.py:110`）。`_cp_shard_dims()` 只挑 CP 轴为 `Shard` 的条目，并记录其 tensor dim；CP replicate/partial 或没有 CP 轴的输入被省略（`torchtitan/distributed/context_parallel/api.py:40`、`torchtitan/distributed/context_parallel/api.py:47`）。

`Decoder.preprocess_inputs()` 的顺序不可交换：先由完整 positions 构造 packed-document mask，再 CP shard batch，最后注入 SPMD type（`torchtitan/models/common/decoder.py:351`、`torchtitan/models/common/decoder.py:365`、`torchtitan/models/common/decoder.py:372`）。`annotate_input_spmd_types()` 要求每个顶层 tensor 都有 layout；容器内 tensor 必须在构造处另行注解（`torchtitan/distributed/spmd_types.py:195`、`torchtitan/distributed/spmd_types.py:205`、`torchtitan/distributed/spmd_types.py:221`）。

`cp_shard()` 先根据配置字符串构造 balancer，然后把相同对象传给 inputs 与 masks 的两次 `_context_parallel_shard`。`BlockMask` 形状语义是 `[B,H,Q,KV]`，只切 Q dim 2，不切 KV dim（`torchtitan/distributed/context_parallel/api.py:180`、`torchtitan/distributed/context_parallel/api.py:229`、`torchtitan/distributed/context_parallel/api.py:239`）。dict mask 保留 key 顺序并逐 mask 重建结果（`torchtitan/distributed/context_parallel/api.py:242`、`torchtitan/distributed/context_parallel/api.py:262`）。

PTRR 只能从一个 `BlockMask` 建 cost ordering。对 dict，`context_parallel_ptrr_mask_key` 选择基准 mask；缺 key、key 不存在或值非 `BlockMask` 都显式报错，但建出的同一个 balancer 会应用于所有输入和每个 mask（`torchtitan/distributed/context_parallel/api.py:188`、`torchtitan/distributed/context_parallel/api.py:200`、`torchtitan/distributed/context_parallel/api.py:217`）。GPT-OSS 正好同时生成 `basic_mask` 与可选 `sliding_window_mask`，每层按自身 key 取用；当前集成 recipe 用 `basic_mask` 构造 PTRR（`torchtitan/models/gpt_oss/model.py:235`、`torchtitan/models/gpt_oss/model.py:250`；`torchtitan_recipes/tests/models.py:254`）。

### ④ 约束/边界

- `prepare_context_parallel_input()` 没有任何可切的命名 tensor 时直接返回，mask 也不会单独切；调用者必须保证至少有一个被声明并存在的 tensor（`torchtitan/distributed/context_parallel/api.py:98`、`torchtitan/distributed/context_parallel/api.py:102`）。
- mask dict 中任一值不是 `BlockMask` 都会失败；dense tensor mask 不属于这个入口（`torchtitan/distributed/context_parallel/api.py:242`、`torchtitan/distributed/context_parallel/api.py:250`）。
- VLM vision inputs 的布局只有 DP/TP、没有 CP，所以通用 sharder不会切图像 tensor；Qwen3.5 还为二维 MRoPE positions 单独声明 token dim 0、component dim replicate（`torchtitan/models/common/vision_encoder_sharding.py:26`、`torchtitan/models/qwen3_5/model.py:386`、`torchtitan/models/qwen3_5/model.py:395`）。
- Flux 仍直接调用 `cp_shard(input_seq_dims=1)`，无 mask 且显式关闭 balancer；package 留有 TODO，希望未来让主 API 覆盖 Flux（`torchtitan/models/flux/trainer.py:232`、`torchtitan/models/flux/trainer.py:242`；`torchtitan/distributed/context_parallel/__init__.py:10`）。

### ⑤ 发展趋势（有锚点的推断）

package 的 Flux TODO 与 `e8e39abc6` 的 per-input sharding 重构共同指向一个方向：让所有模型从声明布局推导输入切分。**推断**：Flux 专用入口可能被统一，但源码没有给出迁移时间，也没有承诺删除 `cp_shard()`。

---

## 5. Attention 边界：Q shard + K/V replicate，而不是全序列重建或 kernel 内 CP

### ① 背景/问题

每个 rank 只持有一部分 query，却必须为这些 query 访问全局 key/value。明显替代方案有三种：把 Q/K/V 全部 all-gather 后冗余计算完整输出；保持三者 shard 并让 kernel 内部做 Ring/online softmax；或只复制 K/V，让普通 kernel 对 local Q × global KV 计算。当前 TorchTitan 选择第三种。

### ② 为什么这么设计

源码明确给出的正确性准则是：`BlockMask` 的 KV 维保持全局，所以 local-map 边界必须让 K/V 看到 full-length keys；Q 与输出仍按 token shard（`torchtitan/models/common/decoder_sharding.py:275`、`torchtitan/models/common/decoder_sharding.py:287`）。源码没有给出完整性能比较。

**知识库推断**：相对“全序列重建”，该布局避免复制 Q、输出以及对应 query 计算；相对“kernel 内 CP”，它牺牲 K/V 临时内存和 all-gather 带宽，换取普通 FlexAttention/SDPA kernel 与统一模块 wrapper。这个推断只由当前布局与已删除 wrapper支撑，不应写成上游 benchmark 结论。

### ③ 实现思路与细节

Decoder attention 激活是 `(T,N,H)`：DP/CP 依次切 T，TP 切 N（`torchtitan/models/common/decoder_sharding.py:76`、`torchtitan/models/common/decoder_sharding.py:87`）。`set_gqa_inner_attention_local_map()` 声明：

| 值 | attention 前 CP src | attention 前 CP dst | kernel 后布局 |
|---|---|---|---|
| Q | shard token dim 0 | shard token dim 0 | — |
| K | shard token dim 0 | replicate | — |
| V | shard token dim 0 | replicate | — |
| output | — | — | 与 Q 相同，shard token dim 0 |

对应 src/dst/out 配置位于 `torchtitan/models/common/decoder_sharding.py:293`、`torchtitan/models/common/decoder_sharding.py:299`、`torchtitan/models/common/decoder_sharding.py:305`。通用 module wrapper 先 assert source type，再仅对 source/destination 不同的输入调用 `spmd_redistribute_per_axis()`；Q 不变，K/V 在 CP group 上执行 shard→replicate collective（`torchtitan/protocols/module.py:597`、`torchtitan/protocols/module.py:606`、`torchtitan/protocols/module.py:615`）。

FlexAttention local kernel 把 `(T,N,H)` 转成 `[1,N,T,H]`，消费已经按 Q dim 切好的 `BlockMask`，最后恢复 `(T,N,H)`；output type 还被检查为与 Q 相同（`torchtitan/models/common/attention.py:314`、`torchtitan/models/common/attention.py:326`、`torchtitan/models/common/attention.py:346`、`torchtitan/models/common/attention.py:373`）。因此 kernel 内没有当前 TorchTitan CP dispatcher。

Flux SDPA 使用同一边界思想但形状为 `[B,L,N,H]`：CP 切 L(dim 1)，K/V destination 为 replicate，输出保持 Q layout（`torchtitan/models/flux/sharding.py:28`、`torchtitan/models/flux/sharding.py:40`、`torchtitan/models/flux/sharding.py:46`）。SDPA 再转成 `[B,N,L,H]` 调 `scaled_dot_product_attention` 并转回；它拒绝任何 attention mask，只接受 `is_causal` bool（`torchtitan/models/common/attention.py:410`、`torchtitan/models/common/attention.py:422`、`torchtitan/models/common/attention.py:427`）。

### ④ 约束/边界

- 标准 decoder CP 当前只允许 FlexAttention；配置更新会拒绝 `ScaledDotProductAttention` 与 `VarlenAttention`（`torchtitan/models/common/decoder.py:178`、`torchtitan/models/common/decoder.py:181`）。语言模型 backend 本身也已拒绝 SDPA，因为它不能表达 per-document positions 的 packed mask（`torchtitan/models/common/config_utils.py:71`、`torchtitan/models/common/config_utils.py:97`）。
- Qwen3/Muse Glimmer 子类还保留“CP supports SDPA and Flex”的旧错误文字，但基类校验先执行并已经拒绝 SDPA；不能从子类 message 反推实际支持（`torchtitan/models/qwen3/model.py:81`、`torchtitan/models/qwen3/model.py:87`、`torchtitan/models/qwen3/model.py:90`）。
- `LocalMapConfig` 仍记录 K/V input-grad 的 CP partial intent，但当前 spmd-types local region只把 input/output local type交给 `spmd.no_typecheck`，没有消费 `in_grad_placements`；不要把旧 DTensor local-map 的显式 grad placement接线写成当前事实（`torchtitan/models/common/decoder_sharding.py:296`、`torchtitan/models/common/decoder_sharding.py:311`；`torchtitan/protocols/module.py:539`、`torchtitan/protocols/module.py:554`）。
- 数值验证脚本会让全局 index 与输入一起过 `cp_shard`，再 all-gather logits 并按 index scatter 恢复全序列；dense Flex+CP 阈值是相对误差 `1e-4`，MoE 为 `2e-3`（`torchtitan/experiments/transformers_modeling_backend/tests/test_flex_cp_numerical.py:138`、`torchtitan/experiments/transformers_modeling_backend/tests/test_flex_cp_numerical.py:174`、`torchtitan/experiments/transformers_modeling_backend/tests/test_flex_cp_numerical.py:183`）。

---

## 6. FSDP storage mesh：为什么 CP-only 仍然 fully_shard

### ① 背景/问题

“CP 只切激活，为什么还要 FSDP？”这个问题混合了两种所有权：forward 中参数在逻辑 CP 轴上应当 replicate，才能让每个 query shard运行相同层；但若每个 CP rank长期保存完整参数，长序列扩展设备数时参数、梯度与 optimizer state会随 CP 度重复。单纯不启用 FSDP 还会绕过 TorchTitan 通过 `fully_shard` 安装的 mixed-precision policy。

### ② 为什么这么设计

**选中的路线**是逻辑值布局与持久参数存储分平面：`SpmdType` 把 dense parameter 声明为 R@CP，FSDP storage resolver 却把 CP 加入 shard axes；**替代方案**是 CP rank永久复制参数，或让算子类型直接暴露 FSDP shard。决策准则是 forward 类型语义不泄漏参数存储细节，同时降低持久状态重复。源码注释明确区分给 `fully_shard()` 的 full dense storage mesh 与给 SPMD typechecker 的 `[dp,cp,tp]` mesh（`torchtitan/distributed/parallel_dims.py:230`、`torchtitan/distributed/parallel_dims.py:233`、`torchtitan/distributed/parallel_dims.py:238`）。

### ③ 实现思路与细节

`dense_param_placement()` 对 DP/CP 都声明 replicate，只把 TP placement交给调用者（`torchtitan/models/common/decoder_sharding.py:24`、`torchtitan/models/common/decoder_sharding.py:30`）。mesh 构造却建立：

```text
dense storage : [pp, dp_replicate, dp_shard, cp, tp]
dense fwd/bwd : [dp, cp, tp]
```

两张 view 来自同一 world rank；具体构造位于 `torchtitan/distributed/parallel_dims.py:243` 与 `torchtitan/distributed/parallel_dims.py:248`。`resolve_fsdp_mesh()` 总包含 `dp_shard`，CP 启用时把 shard axes 从 `dp_shard` 扩为 `(dp_shard,cp)`，只有 `dp_replicate` 是 replicate axis（`torchtitan/distributed/fsdp.py:32`、`torchtitan/distributed/fsdp.py:44`、`torchtitan/distributed/fsdp.py:54`）。

Llama3 这条标准 decoder 并行化路径无条件调用 `apply_fsdp_to_decoder()`：degree-1 时 collective 是 no-op，但仍安装 mixed precision；CP-only 时 storage mesh size>1，FSDP 则沿 CP 参与的扁平 shard submesh保存参数（`torchtitan/models/llama3/parallelize.py:57`、`torchtitan/models/llama3/parallelize.py:68`）。配置文档也明确：`mixed_precision_param` 在 `data_parallel_shard_degree>1` **或** `context_parallel_degree>1` 时通过 `fully_shard` 生效（`torchtitan/config/configs.py:96`、`torchtitan/config/configs.py:98`）。

**知识库推断**：FSDP all-gather在计算前恢复逻辑 R@CP 参数，从而同时满足“类型上 replicate”与“存储上 shard”；代价是每个 FSDP unit 的参数通信。页面只据 storage axes、logical layout 与 `fully_shard` 接线作此机制推断，不把它冒充上游性能结论。

### ④ 约束/边界

- CP 降低的是持久参数重复与 Q/output 激活；K/V 在每个 attention 边界仍临时 replicate，不能把总激活显存简单写成除以 CP。
- CP-only 不是“没有数据并行”：`dp_shard=1` 仍作为 storage axis保留，CP 与它一起组成 FSDP shard submesh；`_mesh_exist` 特意保留 size-1 `dp_shard` 供 FSDP 识别（`torchtitan/distributed/parallel_dims.py:130`、`torchtitan/distributed/parallel_dims.py:135`）。
- PP 下默认 FSDP 不在 forward 后 reshard，以避免每 microbatch 的非重叠 all-gather；这属于 FSDP/PP 代价，不是 CP 的独立策略（`torchtitan/distributed/fsdp.py:124`、`torchtitan/distributed/fsdp.py:129`）。
- `spmd_redistribute_per_axis()` 每个 src/dst pair只允许一条 mesh 轴变化，并拒绝 shard-order 重排；复杂多轴变换必须拆边界或写显式 collective（`torchtitan/distributed/spmd_types.py:256`、`torchtitan/distributed/spmd_types.py:293`、`torchtitan/distributed/spmd_types.py:315`）。

---

## 7. 组合与模型矩阵：支持来自 guard + 布局 + 测试，而不是名称相乘

### ① 背景/问题

CP 可以与 TP、PP、EP、FSDP 共用 rank，但“mesh 能构造”不等于某个 attention、load balancer、compile backend也能运行。支持必须同时满足：rank 等积、token 等分、模型输入/attention layout、dispatcher容量和集成测试。只列一张并行度乘法表会掩盖模型专属边界。

### ② 为什么这么设计

**选中的路线**是让 CP 与 TP 在 dense `[dp,cp,tp]` value mesh 上表达不同 tensor dim，让 PP 在每个 microbatch进入同一预处理入口，让 EP 在 sparse region重切同一批 rank；**替代方案**是为每种组合写独立模型分支。决策准则是组合正交到布局/region，而不是正交到 rank 乘数。EP 的公共配置明确要求 `dp_shard*cp*tp == efsdp*ep`（`torchtitan/config/configs.py:284`、`torchtitan/config/configs.py:288`）。

### ③ 实现思路与细节

| 组合/模型 | 当前接线 | 持续验证或边界证据 |
|---|---|---|
| CP-only | input/Q/output token shard；参数 storage 沿 CP FSDP shard | feature suite `llama3_debugmodel_cp4` / 4 GPU（`tests/integration_tests/features.py:196`） |
| FSDP/HSDP + CP | `dp_shard` 与 CP 同属 dense storage shard axes；可再有 `dp_replicate` | FSDP+CP、无 dp_shard 的 HSDP+CP、有 dp_shard 的 HSDP+CP（`tests/integration_tests/features.py:208`、`tests/integration_tests/features.py:214`、`tests/integration_tests/features.py:220`） |
| TP + CP | CP 切 token T，TP 切 attention head N；SP activation还可沿 `(DP,CP,TP)` 连续切 token | `attention_activation_placement`（`torchtitan/models/common/decoder_sharding.py:76`）与 FSDP+TP+CP 测试（`tests/integration_tests/features.py:226`） |
| PP + CP | 每个 PP microbatch分别执行 `preprocess_inputs`，schedule接收已切 args/kwargs | `torchtitan/trainer.py:737`、`torchtitan/trainer.py:740`、`torchtitan/trainer.py:759` |
| EP + CP | dense region 的 CP/TP ranks重切成 `efsdp*ep`；dispatcher容量除以 `CP*TP` | DeepSeek V3 FSDP+CP+PP+EP golden test（`tests/integration_tests/models.py:57`） |
| MinimalAsyncEP + CP | 同一 sparse重切；额外要求 EP>1、expert整除与 full recompute | 8-GPU FSDP+CP+TP+MinimalAsyncEP（`tests/integration_tests/h100.py:57`）与 config guards（`torchtitan/distributed/minimal_async_ep/api.py:111`、`torchtitan/distributed/minimal_async_ep/api.py:131`） |
| GPT-OSS | Flex dict masks；PTRR key显式选 `basic_mask` | FSDP2+CP2+PP2+EP4 recipe（`torchtitan_recipes/tests/models.py:254`） |
| Flux | SDPA、`[B,L,...]` dim 1 CP、无 mask、无 balancer | `torchtitan/models/flux/trainer.py:232`、`torchtitan/models/flux/sharding.py:40` |
| VLM decoder | text token走 CP；vision input没有 CP轴，不被通用 sharder切分 | `torchtitan/models/common/vision_encoder_sharding.py:26`、`torchtitan/models/qwen3_5/model.py:412` |

PP 组合的关键不是 pipeline kernel懂 CP，而是第一 stage 和各 stage model part 都复用模型的 `preprocess_inputs()`，使 mask、position与局部 token在进入 schedule 前已经一致（`torchtitan/trainer.py:730`、`torchtitan/trainer.py:741`）。SPMD runtime再注册 dense/sparse mesh，并在训练 context中激活 dense mesh（`torchtitan/distributed/utils.py:397`、`torchtitan/distributed/utils.py:414`）。

### ④ 约束/边界

| 失败边界 | 当前行为 | 源码/测试 |
|---|---|---|
| partial-DTensor + CP | 配置更新期拒绝 | `tests/unit_tests/cpu/test_context_parallel_validation.py:50` |
| decoder SDPA/Varlen + CP | `NotImplementedError`；当前 decoder CP 是 Flex | `tests/unit_tests/cpu/test_context_parallel_validation.py:59`、`tests/unit_tests/cpu/test_context_parallel_validation.py:63` |
| PTRR 无 mask/dict 无 key/非法 key | 切分前 `ValueError`，不猜测基准 mask | `torchtitan/distributed/context_parallel/api.py:195`、`torchtitan/distributed/context_parallel/api.py:200`、`torchtitan/distributed/context_parallel/api.py:208` |
| HeadTail/PTRR 自动分支 | **不存在**；generic API 只 match 配置字符串 | `torchtitan/distributed/context_parallel/api.py:180`、`torchtitan/distributed/context_parallel/api.py:182` |
| Transformers modeling backend + Flex | 额外拒绝 HeadTail，要求 PTRR 或 None；其 recipe显式 PTRR | `torchtitan/experiments/transformers_modeling_backend/trainer.py:32`、`torchtitan_recipes/tests/transformers_modeling_backend.py:19` |
| core Decoder + Flex | 公共默认仍是 HeadTail，core CP feature recipes未覆写；说明 balancer policy尚非全局统一 | `torchtitan/config/configs.py:244`、`torchtitan_recipes/tests/features.py:279` |
| GraphTrainer + CP | 集成矩阵全局禁用，等待采用 `spmd_types`；Flex regional compile另有 index-rearrange trace问题 | `torchtitan/experiments/graph_trainer/tests/integration_tests.py:22`、`torchtitan/experiments/graph_trainer/tests/integration_tests.py:26` |
| CooR precompile + CP | 明确 `NotImplementedError`，因为 dummy input尚未复用 CP sharder | `torchtitan/experiments/graph_trainer/precompile_main.py:243`、`torchtitan/experiments/graph_trainer/precompile_main.py:246` |

这里最重要的纠偏是：配置注释仍写“HeadTail for SDPA / PTRR for Flex”，但当前 runtime没有读取 attention 类型。Flux SDPA显式用 None；core Decoder Flex可让默认 HeadTail流入；Transformers backend Flex却额外拒绝 HeadTail。支持策略是模型路径专属，不是一条全局自动分派规则。

### ⑤ 发展趋势（有锚点的推断）

GraphTrainer 测试与 CooR precompile都把缺口写成“尚未复用 `spmd_types`/输入 CP shard 路径”（`torchtitan/experiments/graph_trainer/tests/integration_tests.py:22`；`torchtitan/experiments/graph_trainer/precompile_main.py:243`）。**推断**：下一步更可能是让实验编译路径接入现有输入与布局边界，而不是恢复已删除的 `apply_cp_to_forward`；但源码没有给出完成日期。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 本系列入口、统一源码基线与页面职责。
- [[10_torchtitan_parallel_dims_analysis]] —— CP 如何进入 rank 预算、dense storage/value 双平面 mesh 与 sparse region重切。
- [[11_torchtitan_fsdp_analysis]] —— CP storage axis如何被 FSDP展平，以及参数 all-gather/reduce-scatter 生命周期。
- [[12_torchtitan_tp_analysis]] —— CP token shard与 TP head/feature shard、Sequence Parallel如何组合。
- [[14_torchtitan_pp_analysis]] —— PP microbatch、stage I/O 与 CP预处理边界的完整机制。
- [[15_torchtitan_ep_analysis]] —— `dp_shard*cp*tp = efsdp*ep`、dispatcher容量与 MinimalAsyncEP约束。
- [[16_torchtitan_spmd_types_analysis]] —— `SpmdType`、`PartitionSpec`、单轴 redistribution 与 current-mesh runtime。
