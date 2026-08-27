---
title: "张量并行 TP：让布局协议决定通信，让编译器或 GEMM 决定等待位置"
---

# 张量并行 TP：让布局协议决定通信，让编译器或 GEMM 决定等待位置

> **论点式副标题**：当前 TorchTitan TP 的核心不是一张 `ParallelStyle` 计划表，而是模块配置上的 `ShardingConfig + SpmdType` 布局协议；默认路径在模块边界发显式 collective，compile Async TP 与 dist-GEMM 则分别把“等待”下沉到编译图或 GEMM 调度中。
>
> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页回答**：TP 参数与激活怎样布局；colwise/rowwise、Sequence Parallel 与 Loss Parallel 分别选择什么 collective；compile Async TP 和 eager dist-GEMM 怎样移动通信等待；哪些组合会回退或失败。
>
> **Sibling 边界**：mesh 构造归 [[10_torchtitan_parallel_dims_analysis|ParallelDims]]；SPMD 类型代数归 [[16_torchtitan_spmd_types_analysis|SPMD Types]]；FSDP/CP/EP 各归其专页；跨机制的通信重叠比较归 [[24_torchtitan_comm_optimizations_overlap_analysis|通信重叠]]；编译与图运行时全景归 [[27_torchtitan_graph_trainer_compiler_runtime_analysis|编译器与图运行时]]。

---

## 1. Overview

### 1.1 背景与问题

经典 TP 叙事常从“某个 Linear 是 colwise 还是 rowwise”开始，但这不足以解释当前 TorchTitan：同一组参数布局既可由 `spmd_types` 的 plain local tensor 承载，也可由 `partial_dtensor` 的 DTensor 承载；同一次 all-gather / reduce-scatter 又可能发生在模块边界、被 Inductor 重排，或直接融合进 GEMM。配置只保留这两个 SPMD 后端并默认选择 `spmd_types`（`torchtitan/config/configs.py:168-180`）。

因此真正要追踪的是三件事：

1. **状态**：参数和激活在每个命名 mesh 轴上是什么布局；
2. **边界**：布局从 src 变到 dst 时，哪一次 collective 被物化；
3. **调度**：collective 与 GEMM 是分立、由编译器流水化，还是由显式 fused op 合成。

### 1.2 Thesis

当前 TP 的单一控制面是挂在子模块 Config 上的 `ShardingConfig`。Llama 在构模配置更新时写入根层和各 block 的布局，实例化后由 `Module.parallelize()` 自动递归、分片 state，并把 forward 包成“输入 redistribution → 可选本地区域 → 计算 → 输出 redistribution”（`torchtitan/models/llama3/model.py:69-83`；`torchtitan/models/llama3/sharding.py:25-41`；`torchtitan/protocols/module.py:244-290`）。

这否定了两个旧心智模型：

> [!deprecated] **旧：Llama TP 的权威入口是 `tensor_parallel.py` 里的手写 plan。**
> **现：**该文件当前只保留给复制计算模块使用的 `NoParallel`；它负责把特殊模块参数放到同一 TP mesh，而不是定义 Llama 的 colwise/rowwise 主计划（`torchtitan/distributed/tensor_parallel.py:19-30`、`80-95`）。

> [!deprecated] **旧：TP 通信等价于 `DTensor.redistribute(async_op=True)`。**
> **现：**默认 `spmd_types` 直接在 plain tensor 上调用 `spmd.redistribute`；DTensor 的 `from_local/redistribute` 只属于 `partial_dtensor` 分支（`torchtitan/protocols/module.py:597-653`）。

### 1.3 概念表

| 概念 | 本页中的精确定义 | 不要混同为 |
|---|---|---|
| `ShardingConfig` | state、输入 src/dst、输出 src/dst 与 local region 的模块协议 | 单一后端的参数表 |
| Colwise | 存储为 `[out,in]` 的 weight 在 TP 上 `S(0)` | “切矩阵的第二维” |
| Rowwise | weight 在 TP 上 `S(1)`，局部输出是 partial | 输出天然已完整 |
| Sequence Parallel | 折叠后 `(tokens, hidden)` 的 token 维在 TP 上 `S(0)` | 旧 `[B,S,D]` 的固定 `Shard(1)` |
| Loss Parallel | vocab shard 保持到 CE 内部，只规约 softmax/NLL 中间量 | gather 全量 logits 后做普通 CE |
| compile Async TP | Inductor micro-pipeline 对已编译 model block 的 TP 通信/GEMM 调度 | eager TP 开关 |
| dist-GEMM | 模型模块显式调用 symmetric-memory collective+GEMM autograd Function | compile pass 的别名 |

### 1.4 关键图：同一布局协议，三种等待位置

```text
Model Config.update_from_config
  -> 写入各 Module.Config.sharding_config
  -> build model
  -> model.parallelize(parallel_dims)
       -> shard states
       -> wrap module boundaries

默认 stock 路径：  boundary AG -> ordinary GEMM -> ordinary GEMM -> boundary RS/AR
compile Async TP： same stock graph -------- Inductor micro-pipeline -----------^
dist-GEMM 路径：   fused(AG + GEMM) -------- fused(GEMM + RS) ------------------^
```

上图的事实边界是：stock attention/FFN 在 `ShardingConfig` 声明入口 gather 和 rowwise 输出归约，而 dist-GEMM 分支删除这些边界声明，因为 fused projection 已产生最终布局（`torchtitan/models/common/decoder_sharding.py:226-272`、`317-350`）。compile 路径则仍编译每个 stock TransformerBlock，并启用 Inductor 的 `_micro_pipeline_tp`（`torchtitan/distributed/compile.py:39-72`、`75-96`）。

### 1.5 Quick Start

仓库自己的 H100 recipe 给出了两条最短可追踪配置：

```python
# compile Async TP
config = llama3_debugmodel()
config.compile.enable = True
config.parallelism.tensor_parallel_degree = 2
config.compile.enable_async_tensor_parallel = True

# eager dist-GEMM
config = llama3_debugmodel_dist_gemm()
config.parallelism.tensor_parallel_degree = 2
```

这两段来自 `torchtitan_recipes/tests/h100.py:21-32`。前者必须编译 model；后者的 registry 把 `tp_gemm_backend="dist_gemm"` 写入 model spec，并固定 `spmd_backend="spmd_types"`（`torchtitan/models/llama3/config_registry.py:90-103`）。

### 1.6 可追踪调用链

```text
Trainer config
  -> Llama3Model.Config.update_from_config
  -> set_llama3_sharding_config(enable_sp)
  -> model build
  -> parallelize_llama
       -> model.parallelize(parallel_dims)
       -> [AC]
       -> apply_compile(...)        # 仅 model compile 开启时
       -> apply_fsdp_to_decoder(...)
```

Llama 的执行顺序明确是 TP protocol → AC → compile → FSDP（`torchtitan/models/llama3/parallelize.py:40-78`）。这条顺序解释了为何 Async TP 在 `apply_compile()` 内取 dense TP mesh，而不是在某个早期 parallelism 初始化函数中猜 process group。

---

## 2. 模块边界协议：从声明到真实 collective

### ① 背景 / 问题

plain tensor 不携带 placements。若只写“目标是 Replicate”，运行时无法判断当前值究竟是 Shard、Partial 还是已经 Replicate；若每个算子自己手写通信，又会把模型布局、不变量与 process-group 操作散落在 forward 中。

### ② 为什么这样设计

TorchTitan 选择在**模块边界同时声明 src 与 dst**，把“计算本身”和“进入/离开计算所需的布局变化”分开。明显替代方案是只写 dst 并让 DTensor 自行推断 src，但这只适用于 placements 带内的 DTensor；协议注释明确说明，显式 src 是为了让 erased-type 系统也能表达完整 redistribute pair（`torchtitan/protocols/sharding.py:59-75`）。

历史上，提交 `1786292db0ca10146a032d68ea89fd994bfffd22` 的正文把目标写成：以附着在 `Module.Config` 的 sharding spec 取代传给 `parallelize_module()` 的字符串 plan，并由 `Module.parallelize(mesh)` 自动递归。当前源码已经兑现这一控制面；这里引用提交正文解释选择原因，而非替代当前机制 locator。

### ③ 实现思路与细节

`ShardingConfig` 的五组状态是：

| 字段 | 真实职责 |
|---|---|
| `state_shardings` | 参数/缓冲区布局 |
| `in_src_shardings` | 模块收到输入时的布局 |
| `in_dst_shardings` | 计算前需要的输入布局 |
| `out_src_shardings` | 本地计算产出的布局 |
| `out_dst_shardings` | 交给下游前需要的布局 |

`local_map` 另行定义进入本地 kernel 区域时的输入梯度布局（`torchtitan/protocols/sharding.py:33-53`、`77-113`）。

默认 `spmd_types` 的真实调用链是：

```text
Module.parallelize
  -> spmd_validate_redistributions(config)
  -> spmd_distribute_tensor(state)
  -> forward wrapper
       -> assert src type（仅 typechecking）
       -> spmd_redistribute_per_axis(src, dst)
       -> local compute
       -> assert out_src type（仅 typechecking）
       -> spmd_redistribute_per_axis(out_src, out_dst)
```

state 分片会按 `PartitionSpec` 记录的轴顺序逐次 `spmd.shard`，返回值仍是 `torch.Tensor`（`torchtitan/distributed/spmd_types.py:440-489`）；注册后再用当前 mesh context 做类型断言（`torchtitan/protocols/module.py:292-325`）。边界 helper 对真正变化且 size 大于 1 的轴调用一次 `spmd.redistribute`；具体可能是 all-reduce、reduce-scatter 或 all-gather（`torchtitan/distributed/spmd_types.py:398-437`）。

`partial_dtensor` 消费同一声明，但把状态用 `distribute_tensor` 物化为 DTensor；输入可以先 `DTensor.from_local`，再以 `async_op=True` redistribute（`torchtitan/protocols/module.py:398-420`、`624-653`）。因此两者是**一套布局协议的两种运行时表示**，不是两套模型计划。

### ④ 约束与边界

默认 helper 当前刻意只允许一次 src/dst pair 改变一个 mesh 轴；它拒绝轴集合不一致、多个轴同时变化、以 `V` 为变化端点，以及同一 tensor 维上 shard 顺序重排（`torchtitan/distributed/spmd_types.py:293-380`）。

这一限制给出清晰决策准则：能表达为单轴 AG/RS/AR 的变化放模块边界；多轴重排或值语义不清的 `V` 转换写显式 collective。通用输出 redistribution 也只处理单 tensor，tuple src 会报错，非 tensor 输出直接原样返回（`torchtitan/protocols/module.py:660-703`）。

### ⑤ 发展趋势（有源码锚点）

源码把单轴 helper 标为过渡设计：未来要么让 `spmd_types` 提供更通用的 DTensor-style redistribute，要么在移除 `partial_dtensor` 后改成 collective-based 声明（`torchtitan/distributed/spmd_types.py:256-269`、`410-412`）。**推断**：这意味着当前 src/dst 协议的稳定部分是“边界布局契约”，不一定是现有 per-axis helper 的永久 API。

---

## 3. Colwise / Rowwise 与 Sequence Parallel：布局怎样选 collective

### ① 背景 / 问题

切权重可降低每 rank GEMM 与参数量，但如果每个 projection 前后都恢复完整 hidden，通信会吞掉收益。TP 需要让相邻算子的 shard 自然衔接；SP 还要把 norm、residual 等逐 token 算子的激活内存分摊到 TP ranks。

### ② 为什么这样设计

TorchTitan 保留经典配对：colwise 切输出特征，rowwise 紧接着消费该 feature shard；只有在进入需要完整 hidden 的区域或离开 partial-sum 区域时才通信。明显替代方案是每个 Linear 都接收并返回 replicated tensor，但那会在 colwise 后立刻 all-gather，又在 rowwise 前重新切分，破坏中间 attention/SiLU/乘法可直接在 feature shard 上运行的性质。

SP 选择折叠后的 `(tokens, hidden)`，在 TP 上切 token 维而非引入旧 `[B,S,D]` 语义。这样 rowwise 的 partial output 可以 reduce-scatter 到 token shard，residual 两支都保持同一 `S(0)` 布局（`torchtitan/models/common/decoder_sharding.py:98-107`、`133-150`）。

### ③ 实现思路与细节

真实存储布局如下：

| 角色 | weight `[out,in]` | 输入/局部输出 | 边界结果 |
|---|---|---|---|
| colwise | TP `S(0)` | 输入 hidden 完整 | 输出最后特征维 shard |
| rowwise + SP | TP `S(1)` | 输入特征 shard；局部 GEMM 为 `P` | `P -> S(0)`，reduce-scatter |
| rowwise - SP | TP `S(1)` | 输入特征 shard；局部 GEMM 为 `P` | `P -> I`，all-reduce |

配置事实位于 `colwise_config()` 与 `rowwise_config()`（`torchtitan/models/common/decoder_sharding.py:122-150`）。Llama block 的接线是：Q/K/V projection colwise，`wo` rowwise；`w1/w3` colwise，`w2` rowwise（`torchtitan/models/common/decoder_sharding.py:194-205`、`257-272`、`317-350`）。

开启 SP 后，一层 stock block 的布局流为：

```text
x: TP S(0)                         # token shard
  -> attention boundary S(0) -> R  # all-gather tokens
  -> QKV colwise -> head shard
  -> local inner attention
  -> wo rowwise P -> S(0)          # reduce-scatter
  -> residual stays S(0)
  -> FFN boundary S(0) -> R        # all-gather tokens
  -> w1/w3 colwise -> SwiGLU -> w2 rowwise
  -> P -> S(0)                     # reduce-scatter
```

inner attention 用 `(tokens, heads, head_dim)`，TP 切 heads 维；local-map 边界把 q/k/v 交给本地 kernel（`torchtitan/models/common/decoder_sharding.py:76-95`、`275-314`）。root embedding 的局部 lookup 先产生 partial，再归约到所选激活布局；`lm_head` 的 weight 和 logits 沿 vocab/output 维切分（`torchtitan/models/common/decoder_sharding.py:353-388`）。

最终 norm 是容易漏掉的恢复点：block 可以持续输出 SP shard，但 `pre_lm_head_norm_config()` 在 norm 后将 hidden 恢复为 TP-replicated，再交给普通 model forward 或 chunked lm-head/loss（`torchtitan/models/common/decoder_sharding.py:171-191`）。

### ④ 约束与边界

`enable_sequence_parallel` 默认是 `True`，但它和 TP degree 是独立配置项（`torchtitan/config/configs.py:168-172`）；Llama 的 sharding 声明无条件写入 Config，实际 mesh 决定哪些轴生效（`torchtitan/models/llama3/sharding.py:25-38`）。

TP>1 时，模型配置会要求 `n_heads` 与实际 `n_kv_heads` 都可被 TP degree 整除；失败在构模配置更新阶段显式报错，而不是运行到 reshape 才暴露（`torchtitan/models/common/decoder.py:190-204`）。此外，`spmd_types` 会拒绝任一 TP/EP 参数 shard 的目标维不能整除 mesh size（`torchtitan/protocols/module.py:327-364`）。

### ⑤ 发展趋势

本节没有针对 colwise/rowwise/SP 布局本身的当前 TODO 或弃用锚点，因此不从源码外推新的布局方向。可以确认的演进只是旧 `[B,S,D]/Shard(1)` 叙事已被当前 `(tokens,hidden)/S(0)` 实现替代。

---

## 4. Loss Parallel：让 vocab shard 活到交叉熵内部

### ① 背景 / 问题

`lm_head` 的 vocab logits 往往是训练中最大的瞬时激活之一。若 colwise `lm_head` 后立即 all-gather 完整 vocab，每 rank 都付出全量 logits 内存与通信，等于在 loss 前放弃最后一段 TP。

### ② 为什么这样设计

当前实现选择分布式 softmax/NLL，只规约每 token 的 max、sum-exp 和 target log-prob。明显替代方案是 PyTorch `loss_parallel()` DTensor context；TorchTitan 的源码注释明确说明，自定义 autograd Function 是为了让 SPMD 路径直接使用 local tensor 与 process group，而不依赖 DTensor context manager（`torchtitan/components/loss.py:65-80`）。

提交 `fec0c175d00bd196a53b0d7820bf61b2796ffe6a` 的正文把这一选择记录为：替换 DTensor-based context manager，并要求支持 last-rank uneven shard 与 `IGNORE_INDEX`；当前 tests 继续以 PyTorch `loss_parallel()` 作为 bitwise ground truth（`tests/unit_tests/cpu/test_loss.py:354-376`、`426-479`）。

### ③ 实现思路与细节

`lm_head` 输出保持 vocab 维 shard（`torchtitan/models/common/decoder_sharding.py:382-388`）。`cross_entropy_loss()` 随后分流：

- `partial_dtensor`：识别 `DTensor + Shard(1)`，取 local logits 与 TP group；
- `spmd_types` 且 TP>1：直接传 plain local logits、current TP group 和 global vocab size；
- 其余情况：普通 `F.cross_entropy`。

分支在 `torchtitan/components/loss.py:32-62`。`CrossEntropyLoss.Config` 保存全局 vocab size，调用时透传给分布式 CE；Llama registry 从 model spec 填入该值（`torchtitan/components/loss.py:285-317`；`torchtitan/models/llama3/config_registry.py:38-46`）。

前向真实通信是三次 TP all-reduce：

1. local max → `MAX`，得到稳定 softmax 的全局基准；
2. shifted local sum-exp → `SUM`，得到全局分母；
3. 只有 target 所属 rank 保留 local log-prob → `SUM`，选出目标值。

这三步在 `torchtitan/components/loss.py:127-180`。反向直接由保存的 local `log_probs` 构造本地 vocab shard 梯度，不再发 collective（`torchtitan/components/loss.py:186-224`）。

### ④ 约束与边界

CE 内核以 ceil-chunk 计算每 rank vocab 区间，因此支持最后一片较短；但它明确拒绝空 vocab shard，并异步断言 label 只能是 `IGNORE_INDEX` 或合法全局 id（`torchtitan/components/loss.py:127-151`）。

必须区分两个层级：**loss kernel 支持 uneven vocab**，不代表标准 `lm_head.weight` 可以 uneven TP shard。默认参数物化会先执行通用 even-sharding guard，因此常规模型仍会在 parallelize 阶段拒绝 vocab 维不能整除 TP degree（`torchtitan/protocols/module.py:327-364`）。

### ⑤ 发展趋势

当前主训练路径已经使用手写 CE；GraphTrainer 的 precompile 路径仍留有迁移到 manual loss-parallel implementation 的 TODO（`torchtitan/experiments/graph_trainer/precompile_main.py:252-257`）。该例外由 sibling 编译器页负责，本页不把它误写成主 Trainer 现状。

---

## 5. Compile Async TP：让编译器移动 stock 边界的等待

### ① 背景 / 问题

stock TP 在模块边界发通信并保证下一个普通 GEMM 获得正确布局，但 eager 执行通常会在依赖点等待。若编译器能看见完整 block 图，就可能把 collective 切片并与矩阵乘流水化，而无需给每个模型 projection 写新模块。

### ② 为什么这样设计

提交 `737594746fda65a6d94dc9482ef07863a80c8588` 的正文说明了当前选择：Async TP setup 要跨 SPMD 后端取得**精确 dense TP process group**，因此把开关从 parallelism config 移到 compile config，并在 `apply_compile()` 中配置。明显替代方案是在 TP 初始化时全局打开 pass，但那既不能保证拿到正确后端 mesh，也把编译器优化伪装成 eager parallelism 能力。

### ③ 实现思路与细节

当前配置入口是 `CompileConfig.enable_async_tensor_parallel`。构造期 guard 要求 `compile.enable=True` 且 `"model" in compile.components`（`torchtitan/config/configs.py:295-315`）。Llama 只有 model compile 开启时才调用 `apply_compile()`，且调用发生在 AC 之后、FSDP 之前（`torchtitan/models/llama3/parallelize.py:42-68`）。

`apply_compile()` 的真实链路是：

```text
parallel_dims.tp_enabled ? get_dense_tp_mesh() : None
  -> _maybe_enable_async_tp
       -> exact TP group_name
       -> enable_symm_mem_for_group(group_name)
       -> torch._inductor.config._micro_pipeline_tp = True
  -> for each TransformerBlock:
       block.compile(backend=..., fullgraph=True)
```

调用与 per-block compile 位于 `torchtitan/distributed/compile.py:39-72`；symmetric-memory group 注册和 Inductor flag 位于 `torchtitan/distributed/compile.py:75-96`。`get_dense_tp_mesh()` 对 `spmd_types` 从 dense SPMD mesh 取 `tp`，对兼容后端则取普通 `tp` mesh（`torchtitan/distributed/parallel_dims.py:425-441`）。

### ④ 约束与边界

没有 TP mesh 时 helper 直接返回；因此配置合法不等于 TP=1 会获得流水化收益（`torchtitan/distributed/compile.py:75-83`）。该路径还以 `fullgraph=True` 编译每个 block，而不是编译整个模型（`torchtitan/distributed/compile.py:66-72`）。

单测固定了两个失败配置：只开 Async TP 不开 compile，以及只 compile loss；并验证正确配置会为精确 group 启用 symmetric memory 且打开 `_micro_pipeline_tp`（`tests/unit_tests/gpu/test_compile_moe.py:45-91`）。源码没有在 TorchTitan 层枚举 Inductor 能识别的全部 GEMM pattern，因此不应从旧 PyTorch 版本实现行号外推覆盖率。

### ⑤ 发展趋势（有源码锚点）

当前仍手动调用已标 deprecated 的 `enable_symm_mem_for_group`；TODO 指向 PyTorch 自动为 Async TP 使用的 process group 注册 symmetric memory（`torchtitan/distributed/compile.py:83-95`）。**推断**：未来可能删除显式注册，但 Async TP 仍属于 compile graph scheduling，而不会因此变成 eager TP 开关。

---

## 6. dist-GEMM：在 eager 模块里显式合并 collective 与 GEMM

### ① 背景 / 问题

compile Async TP 把重叠能力绑定到编译器能否捕获并改写 stock 图。希望在 eager Trainer 中获得同类调度时，需要另一条路线：模型 projection 自己调用 collective+GEMM fused primitive，并让布局协议停止在模块外重复通信。

### ② 为什么这样设计

提交 `9a711521ac2973fe230a3f38efc6aedfc7d1f9c6` 的正文明确给出动机：把“AG/RS 折进相邻 GEMM”从特定 compiler/graph pass 解耦，变成 `tp_gemm_backend` 选择的模型属性。明显替代方案仍是“GEMM 完成后再单独 collective”；它保持通用性，却不能在 GEMM tile 级消费/产生通信数据。

当前 config 工厂也将两者区分为：`default` 让 framework 在 ordinary GEMM 两侧处理 collective；`dist_gemm` 通过 symmetric memory 把 collective 折入相邻 GEMM（`torchtitan/models/common/config_utils.py:187-212`）。

### ③ 实现思路与细节

选择 `tp_gemm_backend="dist_gemm"` 后：

| 模块 | fused primitive | 输入 → 输出布局 |
|---|---|---|
| QKV colwise | `AllGatherLinear` | token shard → full tokens / feature shard |
| attention `wo` | `LinearReduceScatter` | feature shard → token shard / full hidden |
| FFN `w1/w3` | `AllGatherLinearMulti` | 一次 AG 同时喂两个 GEMM |
| FFN `w2` | `LinearReduceScatter` | partial local GEMM → RS 后 token shard |

GQA 工厂要求 fused QKV，并替换 QKV/`wo` 配置类；FFN 工厂替换为 `DistGEMMFeedForward`（`torchtitan/models/common/config_utils.py:218-278`、`281-306`）。模块在 forward 时从 current SPMD mesh 解析 TP group；无 group 或 TP=1 则警告后调用 stock projection（`torchtitan/models/common/dist_gemm.py:53-85`、`113-175`、`178-230`）。

底层并非“先 collective、返回、再 GEMM”的 Python 组合：`AllGatherLinear.forward` 直接调用 `torch.ops.symm_mem.fused_all_gather_matmul`，dgrad 用 `fused_matmul_reduce_scatter`；为避免保存 full gathered activation，forward 只保存一个 K shard，backward 再沿 K gather（`torchtitan/distributed/linear.py:47-78`、`110-180`）。镜像的 `LinearReduceScatter.forward` 直接调用 fused matmul-RS，backward 的一次 fused AG-matmul 同时提供 full dy 与 dgrad，再用 full dy 做本地 wgrad（`torchtitan/distributed/linear.py:305-325`、`356-418`）。

SwiGLU 的两个上投影共享输入，因此 `AllGatherLinearMulti` 用一次 AG 喂两次 GEMM，并在 dgrad 通过 concat 表达两路求和，避免第二次 RS（`torchtitan/distributed/linear.py:185-206`）。若再叠加 fused SwiGLU override，`DistGEMMFusedSwiGLU` 用 interleaved `w13` 的单次 `AllGatherLinear` 和 `w2` 的 `LinearReduceScatter`；它减少结构/cat，但不再减少 collective 次数（`torchtitan/overrides/fused_swiglu.py:507-558`、`589-597`）。

布局协议必须同步改变：dist-GEMM attention 不再声明 block 入口 AG，rowwise projection 也只保留 state sharding，因为 fused op 已直接返回最终 sequence shard，而不是暴露 `Partial` 给 framework 再 RS（`torchtitan/models/common/decoder_sharding.py:226-272`、`329-350`）。

### ④ 约束与边界

硬前提是 `spmd_backend="spmd_types"` 且 SP 开启；否则 sharding setup 直接报错，因为 fused GEMM 消费/返回 plain local tensor，而且其 AG/RS 正是 SP collective（`torchtitan/models/common/dist_gemm.py:88-110`）。attention 还要求 `fuse_qkv=True`；`DistGEMMFeedForward` 拒绝 `w1/w3` bias（`torchtitan/models/common/config_utils.py:208-229`；`torchtitan/models/common/dist_gemm.py:178-208`）。

symmetric-memory op 是 CUDA-only。一个 process group 的 workspace 从 offset 0 复用；顺序 module forward/autograd 是安全假设，但人为把两个此类 op 放到不同 stream 并发会别名同一 workspace（`torchtitan/distributed/linear.py:14-25`）。CPU/gloo 测试只验证 config 与 sharding contract，CUDA guard 下另测 stock-vs-fused FFN 与 fused-SwiGLU composition；真实 H100 集成项以 TP2+FSDP2 运行（`tests/unit_tests/cpu/test_dist_gemm.py:190-249`、`tests/unit_tests/cpu/test_dist_gemm.py:319-389`；`tests/integration_tests/h100.py:64-71`）。

compile Async TP 与 dist-GEMM 没有源码 guard 禁止同时开启；但 dist-GEMM 已删除 stock 边界 collective。**推断**：两开不等于同一 AG/RS 获得两次重叠收益，应按“哪条路径实际拥有 collective”检查编译图。

### ⑤ 发展趋势（有源码锚点）

`decoder_sharding.py` 把 dist-GEMM 的特殊分支标为 transitional：当 redistribution collectives 普遍移入模块、边界 src→dst 机制消失时，该分支可折叠（`torchtitan/models/common/decoder_sharding.py:261-271`、`329-333`）。**推断**：dist-GEMM 正在展示一种可能的未来模块契约——模块只声明 state，通信成为算子实现的一部分；这不是当前所有 TP 模块已经完成的迁移。

---

## 7. 组合决策、失败分支与被替代的旧断言

### ① 背景 / 问题

TP、SP、Loss Parallel、compile 和 dist-GEMM 共享同一 mesh，却不共享相同前提。只看开关名称容易把“布局语义”“后端表示”和“调度优化”混为一谈。

### ② 为什么这样设计

当前系统把 correctness 放在布局协议与早期 guard，把 performance choice 放在 compile/model backend。明显替代方案是一个总开关同时决定 TP 布局和重叠实现，但那会使 eager、compile、DTensor/local tensor 的能力边界无法独立验证。

### ③ 实现思路与细节

| 目标 | 推荐入口 | 必要条件 | collective 所有者 |
|---|---|---|---|
| 普通 TP/SP | `tensor_parallel_degree` + `enable_sequence_parallel` | model 支持对应 sharding config | 模块边界 redistribution |
| vocab-sharded CE | colwise `lm_head` + `CrossEntropyLoss` | TP group；正确 global vocab | `_LossParallelCrossEntropy` |
| 编译器流水化 | `compile.enable_async_tensor_parallel` | compile model；有 TP mesh | stock boundary + Inductor |
| eager collective/GEMM fusion | `tp_gemm_backend="dist_gemm"` | `spmd_types` + SP + CUDA；attention fused QKV | projection autograd Function |

### ④ 约束与边界

以下旧断言应从当前知识模型中删除：

| 旧断言 | 当前结论 | 当前证据 |
|---|---|---|
| TP 主计划在 `tensor_parallel.py` | 主线是 config sharding + recursive `Module.parallelize`；该文件只剩 `NoParallel` | `torchtitan/distributed/tensor_parallel.py:19-30`；`torchtitan/protocols/module.py:244-290` |
| 两后端都把激活变为 DTensor | 默认后端保留 plain tensor + SPMD type；兼容后端才用 DTensor | `torchtitan/protocols/module.py:597-653` |
| SP 是 `[B,S,D]` 的 `Shard(1)` | dense decoder 使用 `(tokens,hidden)` 的 TP `S(0)` | `torchtitan/models/common/decoder_sharding.py:98-107` |
| Loss Parallel 依赖 PyTorch context | 主 Trainer 使用自定义 `_LossParallelCrossEntropy` | `torchtitan/components/loss.py:32-80` |
| Async TP 属于 `ParallelismConfig` | 已迁至 `CompileConfig`，且要求 compile model | `torchtitan/config/configs.py:295-315` |
| `async_op=True` 就等于通信/GEMM重叠 | DTensor async redistribute、Inductor pipeline、dist-GEMM fused op 是三种不同机制 | `torchtitan/protocols/module.py:648-653`；`torchtitan/distributed/compile.py:75-96`；`torchtitan/distributed/linear.py:110-159` |
| loss kernel 支持 uneven 即模型支持 uneven vocab 参数 | CE 支持 last-rank uneven；state guard 仍拒绝 uneven TP/EP 参数 | `torchtitan/components/loss.py:127-143`；`torchtitan/protocols/module.py:327-364` |

测试边界也必须写清：CPU/gloo 可以验证布局 contract，不会运行 CUDA-only symmetric-memory kernel；`test_distributed_linear.py` 因此用 CUDA guard 测三个 fused primitive 的前后向数值（`tests/unit_tests/cpu/test_distributed_linear.py:34-48`、`120-161`）。

### ⑤ 发展趋势

本页可锚定的方向只有两条：边界 redistribution helper 仍是过渡实现，以及 dist-GEMM 的特殊 sharding 分支期待 collective 普遍内移。除此之外，源码没有承诺默认后端何时移除或哪条 Async TP 路径将统一，因此不作时间表推断。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/10_torchtitan_parallel_dims_analysis|ParallelDims：多平行维度约束与双层网格]] —— 解释 dense TP mesh、SPMD mesh 与 degree 校验从何而来。
- [[02_engineering/02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis|FSDP：参数存储、重分片与混合精度]] —— 解释 TP state shard 如何与 FSDP 存储轴组合。
- [[02_engineering/02_train_frameworks/torchtitan/13_torchtitan_cp_analysis|Context Parallel：上下文切分与注意力通信]] —— 解释 token 维同时被 CP/TP 分片时的 attention 边界。
- [[02_engineering/02_train_frameworks/torchtitan/15_torchtitan_ep_analysis|Expert Parallel：稀疏网格、路由与 token dispatch]] —— 对照 dense TP 与 sparse EP 的 mesh/uneven 限制。
- [[02_engineering/02_train_frameworks/torchtitan/16_torchtitan_spmd_types_analysis|SPMD Types：布局代数与显式 collective]] —— 深挖 `V/I/R/P/S`、`PartitionSpec` 与 typechecking。
- [[02_engineering/02_train_frameworks/torchtitan/24_torchtitan_comm_optimizations_overlap_analysis|通信优化与重叠：从 bucket 到 dist-GEMM]] —— 横向比较 Async TP、dist-GEMM 与其他 overlap 机制。
- [[02_engineering/02_train_frameworks/torchtitan/27_torchtitan_graph_trainer_compiler_runtime_analysis|GraphTrainer、编译器与图运行时]] —— 解释 compile pass、CUDA Graph 与图级 memory/communication policy。
