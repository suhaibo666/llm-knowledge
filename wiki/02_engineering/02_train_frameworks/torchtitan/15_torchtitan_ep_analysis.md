---
title: "专家并行 EP：统一 dispatcher、稀疏网格与四条通信路径"
---

# 专家并行 EP：统一 dispatcher、稀疏网格与四条通信路径

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-27）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **主线**：当前 EP 的稳定边界不是某个模型专属的 `ExpertParallel`，而是 `RoutedExperts = token_dispatcher + inner_experts`：dispatcher 统一产出“按本地专家连续的激活、每专家 token 数、可逆 metadata”，GroupedExperts 只消费这份协议。参数在 `EP × EFSDP` 稀疏存储平面上切分，激活在 dense/sparse SPMD 运行时网格之间切换；AllToAll、DeepEP v2、HybridEP、MinimalAsyncEP 只是同一协议下不同的通信/静态形状取舍。
>
> 主要源文件：`torchtitan/models/common/{moe,moe_sharding,token_dispatcher}.py`、`torchtitan/distributed/{parallel_dims,spmd_types,fsdp}.py`、`torchtitan/distributed/{deepep,minimal_async_ep}/`。

---

## 1. 先纠正旧版知识：对象边界与调用链都已迁移

提交 `4a93ee4e4` 统一了 expert token dispatcher API。当前 `RoutedExperts` 明确把 `inner_experts` 与 `token_dispatcher` 建成兄弟节点，前向固定为 `dispatch → inner_experts → combine`，并由 `parallelize()` 将 EP mesh 接到 dispatcher；dispatcher 不再藏在 GroupedExperts 内部（`torchtitan/models/common/moe.py:123-180`）。模型构造也统一先建 `GroupedExperts.Config`，再按 `comm_backend` 填入 dispatcher config（`torchtitan/models/common/config_utils.py:426-455`）。

> [!deprecated] 旧 `ExpertParallel` / llama4 装配路径已失效
> 旧页引用的 `torchtitan/distributed/expert_parallel.py`、`torchtitan/models/llama4/parallelize.py` 与 `ExpertParallel._partition_fn` 已不在当前源码树。现行权威入口是 common `moe_sharding.set_moe_sharding_config()`；例如 DeepSeek V3 把 grouped-expert 的三种 TP placement 交给该公共函数（`torchtitan/models/deepseek_v3/sharding.py:34-39`、`torchtitan/models/deepseek_v3/sharding.py:143-150`）。

统一后的 backend 选择是显式的四分支：`standard`、`deepep`、`hybridep`、`minimal_async_ep` 分别构造对应 Config，未知字符串直接报错（`torchtitan/models/common/config_utils.py:358-423`）。因此这四条路径都是 **live**；legacy 的是旧的模型专属 API，以及 DeepEP v1 分立的 HT/LL API，而不是其中某一条当前 dispatcher。

---

## 2. EP 不是额外乘数：dense ranks 被重解释为 `EFSDP × EP`

### 2.1 设备集合不变，坐标系改变

全局 world-size 仍满足：

```text
dp_replicate * dp_shard * cp * tp * pp == world_size
```

EP 不进入该乘积；它必须整除 `dp_shard * cp * tp`（`torchtitan/distributed/parallel_dims.py:102-128`）。构网时：

```text
fsdp  = dp_shard * cp
efsdp = fsdp * tp / ep

dense storage  = [pp, dp_replicate, dp_shard, cp, tp]
sparse storage = [pp, dp_replicate, efsdp, ep]
```

计算与两张 mesh 的构造见 `torchtitan/distributed/parallel_dims.py:216-278`。配置文档进一步明确 `pp` 与 `dp_replicate` 是外层轴，EP 借用的是 FSDP/CP/TP 所覆盖的 dense region（`torchtitan/config/configs.py:284-292`）。这意味着 PP 只复制“每个 stage 内的一套 EP group”，而不会进入 EP/EFSDP 的等积约束。

### 2.2 专家参数不是简单的“每卡若干完整专家”

公共 sharding 先把 routed-expert 参数描述为：`DP_REPLICATE=Replicate`、`EFSDP=Replicate`、`EP=Shard(0)`；`dim=0` 是 grouped weight 的专家维（`torchtitan/models/common/moe_sharding.py:30-48`）。这一步表达的是模型并行布局，而非最终静态存储形态。

随后 FSDP 为稀疏参数选择 `[dp_replicate, efsdp, ep]` storage mesh，并把 `efsdp` 指定成 FSDP shard axis（`torchtitan/distributed/fsdp.py:65-82`）。对一个 MoE block，FSDP 的 per-param placement 把专家参数送往 sparse mesh、其余参数送往 dense mesh；当 `efsdp * ep` 大于专家数时，专家参数会改沿 `Shard(1)` 切，以避免在专家维 padding（`torchtitan/distributed/fsdp.py:267-359`）。所以旧页“每个本地专家权重始终完整未切”只描述了纯 EP 的逻辑视图，漏掉了当前 EFSDP 存储分片。

### 2.3 激活从 dense region 临时进入 sparse region

默认 `spmd_types` 后端注册 dense `[dp, cp, tp]` 与 sparse `[dp_replicate, efsdp, ep]` 两张前后向 mesh（`torchtitan/config/configs.py:168-180`、`torchtitan/distributed/parallel_dims.py:229-279`）；线程局部 stack 保存当前运行区域（`torchtitan/distributed/spmd_types.py:108-181`）。`maybe_set_sparse_mesh()` 只在 EP sparse mesh 存在时切换，退出上下文自动恢复 dense mesh（`torchtitan/distributed/spmd_types.py:184-192`）。

真实前向中有三次关键边界：

1. `RoutedExperts` 的 `local_map` 把 dense DTensor 输入变成本地 tensor，并规定返回 dense sequence-sharded 或 Partial 布局（`torchtitan/models/common/moe_sharding.py:200-269`）。
2. AllToAll dispatch/combine 在 `maybe_set_sparse_mesh()` 内把计数与 token buffer 重新解释到 sparse mesh，并只沿 `ep` group 通信（`torchtitan/models/common/token_dispatcher.py:421-464`、`torchtitan/models/common/token_dispatcher.py:581-602`）。
3. grouped expert GEMM 本身也运行在 sparse context；完成 combine 后，`local_map` 再把结果包装回 MoE 边界所需的 dense 布局（`torchtitan/models/common/moe.py:137-169`）。

因此“mesh 切换”不是迁移参数，也不是另做一次 all-gather；它是在同一批 rank 上改变当前 tensor type/collective 应按哪组逻辑轴解释。

---

## 3. 统一 dispatcher 协议：通信后端与专家 kernel 解耦

所有 dispatcher 都实现同一组三段式数据契约：

```text
dispatch(x_TD, scores_TK, expert_ids_TK, local_counts_E)
  -> routed_input_RD, counts_per_local_expert_e, metadata

inner_experts(routed_input_RD, counts_e)
  -> routed_output_RD

combine(routed_output_RD, metadata, original_x_TD)
  -> out_TD
```

抽象 EP 基类统一 EP mesh wiring 与方法签名（`torchtitan/models/common/token_dispatcher.py:172-225`）；实际 orchestrator 不分 backend，始终沿该协议调用（`torchtitan/models/common/moe.py:137-169`）。这比让每个模型/通信库各自包住整个 MoE 更可组合：通信实现只需保证 `RD + counts_e`，专家实现只需保证 expert-major grouped compute。

普通 `GroupedExperts` 用 `counts_e.cumsum()` 生成 grouped-mm offsets，然后执行 gate/up/down 三次 grouped GEMM（`torchtitan/models/common/moe.py:55-120`）。因此动态 AllToAll、DeepEP compact、HybridEP padding、MinimalAsyncEP 固定 receive capacity 都可以复用同一计算接口；各 dispatcher 的差异被 metadata 与 counts 隔离在计算边界之外。

### 3.1 FusedSwiGLU 与 offset-aware 尾部保护

`FusedGroupedExperts` 把 w1/w3 合成 `w13(E,F,2,D)`，一次 grouped GEMM 同时产出 gate/up，再把相同 offsets 传给 fused SiLU-mul 与 down projection（`torchtitan/overrides/fused_swiglu.py:600-658`）。fused kernel 从最后一个 offset 读取有效 row 上界，forward/backward 都跳过 capacity-padding 尾部（`torchtitan/overrides/fused_swiglu.py:96-124`、`torchtitan/overrides/fused_swiglu.py:144-179`）。这正是“dispatcher 可以返回静态大 buffer、专家 kernel 只计算有效前缀”的接口闭环。

量化 grouped GEMM 还有更严格的 group alignment：`TorchAOTokenDispatcher` 是 standard AllToAll 的 live 特化，会把每个 expert group pad 到 FP8/MXFP8 所需倍数（`torchtitan/models/common/token_dispatcher.py:621-686`）。当前量化转换只接受 TorchAO dispatcher 或 HybridEP padding；DeepEP/MinimalAsyncEP config 会被拒绝（`torchtitan/components/quantization/utils.py:33-64`）。这是一条真实的组合边界，而不是所有 dispatcher 与所有 expert kernel 自动互换。

---

## 4. 四条 live 通信路径

| Dispatcher | 状态与适用面 | dispatch / combine 的关键取舍 |
|---|---|---|
| `AllToAllTokenDispatcher` | live，`standard` 分支；无外部 EP 库 | 通用变长 all-to-all；精确、易读，但计数 wait 与 D2H split 暴露在 CPU critical path |
| `DeepEPTokenDispatcher` | live；只接受 DeepEP v2 `ElasticBuffer` | training compact 有 host sync 与完整 autograd；no-grad inference 可选静态 expand、无 host sync；combine 发起异步后立即显式同步 |
| `HybridEPTokenDispatcher` | live；面向 GB200/NVLink72 的 TMA 优化 | fused dispatch-with-permute；blocking 模式精确无丢 token，non-blocking 模式静态容量、可 CUDA graph，但容量不足会静默丢 token |
| `MinimalAsyncEPTokenDispatcher` | live；CUDA symmetric memory + Triton，无 DeepEP 依赖 | 固定对称 buffer、直接写最终 expert-major row；custom-op 边界同步等待，不提供 microbatch 通信重叠；需 full recompute |

### 4.1 Standard AllToAll：两次交换与立即闭合的异步窗口

dispatch 的真实链是：本地稳定排序 → 小 all-to-all 交换每专家计数 → 显式 `wait_tensor` → D2H 物化 `input_splits/output_splits` → token all-to-all → rank-major 到 expert-major `_permute`（`torchtitan/models/common/token_dispatcher.py:245-314`、`torchtitan/models/common/token_dispatcher.py:412-493`）。`output_splits` 必须同步拷到 CPU 后马上 `.tolist()`，所以 D2H 仍是明确的 host stall（`torchtitan/models/common/token_dispatcher.py:296-308`）。

combine 则 `_unpermute` 后用交换过的 split 做反向 all-to-all，再乘 routing score 与 scatter-add 回原 token（`torchtitan/models/common/token_dispatcher.py:549-618`）。当前源码仍注明 token all-to-all 返回 `AsyncCollectiveTensor`，但 dispatch 结果立即被索引 `_permute`，combine 结果立即被 dtype cast/乘法读取；这两个数据依赖都很快关闭异步窗口（`torchtitan/models/common/token_dispatcher.py:459-483`、`torchtitan/models/common/token_dispatcher.py:585-617`）。

### 4.2 DeepEP v2：训练 compact 与推理 expand 共用一个 buffer

旧 v1 的 HT/LL 两套 API 已是 legacy。当前模块要求 DeepEP `>=2.0.0`，以一个 `ElasticBuffer`、一对 dispatch/combine custom op 和一种 `DispatchState` 覆盖训练/推理（`torchtitan/distributed/deepep/deepep.py:7-18`、`torchtitan/distributed/deepep/deepep.py:44-50`）。

- grad-enabled training 强制 compact：deduplicated receive rows 经 `_permute_tokens` 展成 expert-major；精确 per-expert counts 来自 host sync，支持 dispatch/combine 对偶的手写 autograd（`torchtitan/distributed/deepep/deepep.py:20-35`、`torchtitan/distributed/deepep/deepep.py:384-419`）。
- no-grad inference 且 `cudagraphable=True` 才走 expand：静态 expert slot layout、device-side counts、无 host sync；该布局明确不支持 backward（`torchtitan/distributed/deepep/deepep.py:454-500`）。
- combine 先在普通 PyTorch 中应用 routing score，再调用纯 reduction custom op（`torchtitan/distributed/deepep/deepep.py:560-605`）。底层记录异步完成 event，但 dispatcher 在返回前立刻 `sync_combine()`（`torchtitan/models/common/token_dispatcher.py:865-878`、`torchtitan/distributed/deepep/deepep.py:315-327`）。

### 4.3 HybridEP：fused permute 的显存—同步—丢 token 三角

HybridEP 的 custom op 直接调用 `dispatch_with_permute`，把 all-to-all 与 expert-major 重排融合（`torchtitan/distributed/deepep/hybridep.py:115-190`）。两种模式不能混为一谈：

- blocking 默认模式让 DeepEP `cudaStreamSynchronize`，从 pinned CPU counts 算精确输出行数，无 token dropping；
- non-blocking 模式按 `num_tokens * ep_size * min(local_experts, top_k) * capacity_factor` 预分配输出，避免 D2H；factor `1.0` 是无丢失 worst-case 容量，小于 `1.0` 节省显存但溢出 token 只在 GPU flag 中记录、调用侧不会同步检查（`torchtitan/distributed/deepep/hybridep.py:80-109`、`torchtitan/distributed/deepep/hybridep.py:151-183`）。

combine 在 expert 输出上应用 permuted scores，再用 opaque handle 调 `combine_with_unpermute`；API 没有像 DeepEP v2 那样暴露 deferred `sync_combine` 窗口（`torchtitan/distributed/deepep/hybridep.py:455-540`）。该路径依赖 DeepEP 的 `hybrid-ep` 分支，且当前显式拒绝 FP8 dispatch（`torchtitan/distributed/deepep/hybridep.py:394-419`）。

### 4.4 MinimalAsyncEP：名字是 async EP，实现边界却是同步可读

MinimalAsyncEP 预先分配两套 hidden receive buffer 与一套 counts buffer，通过 CUDA symmetric memory rendezvous 获得 peer buffer/pointer（`torchtitan/distributed/minimal_async_ep/api.py:174-295`）。dispatch 先交换完整 counts，直接计算最终 expert-major 目标 row，再由 Triton kernel 写 peer receive buffer；它省掉 standard 路径的“先 rank-major all-to-all、再 `_permute`”（`torchtitan/distributed/minimal_async_ep/api.py:520-555`、`torchtitan/distributed/minimal_async_ep/api.py:563-645`）。

但 `_copy_rows_to_peers_and_wait_cuda()` 在返回前做 EP-group barrier，源码明确把 custom-op 定义为 synchronous readable boundary，并明确说不提供 microbatch communication overlap（`torchtitan/distributed/minimal_async_ep/api.py:298-365`）。combine 对称地把专家输出写回 origin rank，然后用 FP32 top-k reduction kernel 合并；反向也显式注册 dispatch/combine 对偶 custom op（`torchtitan/distributed/minimal_async_ep/api.py:689-739`、`torchtitan/distributed/minimal_async_ep/api.py:902-1015`）。

---

## 5. 旧 `AsyncCollectiveTensor` / shared-experts overlap 结论现已不成立

> [!deprecated] 旧版“combine all-to-all 与 shared experts FFN 重叠”已失效
> 当前 `MoE.forward()` 先完整调用 `routed_experts`，等其 `combine()` 返回后才计算 `shared_experts(x)` 并相加（`torchtitan/models/common/moe.py:396-453`）。standard combine 在返回前已经消费 ACT 做 score/scatter，DeepEP dispatcher 更在返回前显式 `sync_combine()`；所以旧页画出的“通信 stream 跑 combine，同时 compute stream 跑 shared experts”不是当前公共 eager 调用链。

旧页对 PyTorch 2.9.1 `AsyncCollectiveTensor.__torch_dispatch__` 的逐行解释也不应继续冒充 TorchTitan 当前机制基线。本页只保留当前 TorchTitan 可验证的事实：standard token exchange 仍可能返回 ACT，但立即出现数据依赖；真正仍然存在的 EP/FSDP overlap 补偿，是 EP 开启后无条件安装显式 forward/backward FSDP prefetch，以降低 EP 中 D2H sync 对隐式预取的干扰（`torchtitan/distributed/fsdp.py:384-424`）。

这也解释了不同 backend 的“通信窗口”差异：standard 的窗口被 `_permute`/score 很快关闭；DeepEP v2 的 event 被 dispatcher 立即等待；HybridEP 不暴露 deferred wait；MinimalAsyncEP 明确同步返回。若要获得更强的跨算子调度，应把它视为 graph scheduling/overlap pass 的职责，而不是继续依赖旧 shared-experts 代码位置。

---

## 6. MinimalAsyncEP 与 CP / TP / PP：支持范围大于当前验证范围

提交 `15db18b9c` 删除了 MinimalAsyncEP 对 CP、TP、PP 的旧禁用检查。当前机制不是新增三套 backend-specific 通信算法，而是复用公共边界：CP/TP 在进入 `RoutedExperts` 前把 token 轴切成本地 shard；MinimalAsyncEP 只在 stage-local EP group 内搬运当前 `x_TD`，PP 则位于 sparse mesh 外层（`torchtitan/distributed/minimal_async_ep/api.py:7-20`、`torchtitan/models/common/token_dispatcher.py:1009-1014`）。

持久 buffer 容量按当前 microbatch token 数除以 `CP * TP` 推导；必须整除，用户显式容量不能小于推导值。PP 不进入容量公式，因为它切层而不切 token（`torchtitan/models/common/token_dispatcher.py:1186-1252`）。仍有三条硬限制：EP 必须大于 1、专家数必须被 EP 整除、训练必须 full activation recompute 或 graph trainer full memory policy（`torchtitan/distributed/minimal_async_ep/api.py:99-143`）；底层还只接受 symmetric-memory CUDA backend（`torchtitan/distributed/minimal_async_ep/api.py:204-225`）。

验证边界要保守解读：当前仓库可直接定位的 H100 integration recipe 覆盖 `FSDP2 + CP2 + TP2 + EP8 + spmd_types + FullAC`（`torchtitan_recipes/tests/h100.py:63-77`），测试清单以 8 GPU 运行它（`tests/integration_tests/h100.py:57-63`）。这个 committed case 的 `pipeline_parallel_degree` 仍沿用基配置的 1（`torchtitan/models/deepseek_v3/config_registry.py:122-137`）；因此“PP 可组合”有源码/mesh 契约支撑，但本 baseline 的这项 integration evidence 并未同时验证 PP，也不能从一个 case 外推任意 CP×TP×PP 拓扑都已覆盖。

---

## 7. 如何选择 dispatcher

1. **先要可移植、精确与 reference 行为**：选 standard AllToAll。代价是 count exchange、host split materialization 与额外 permute。
2. **训练与 CUDA-graph 推理共用 DeepEP v2 基础设施**：选 DeepEP；训练接受 compact host sync，推理才用 no-grad expand。不要把 expand 当成可反传路径。
3. **GB200/NVLink72 且希望融合 dispatch+permute**：选 HybridEP；需要在 blocking dropless 与 non-blocking 静态容量之间显式做内存/正确性选择。
4. **希望无 DeepEP 依赖、CUDA graph 可接受同步 custom-op 边界**：选 MinimalAsyncEP；接受 full recompute、CUDA symmetric memory 与无 microbatch overlap。
5. **需要 FP8/MXFP8 expert group padding**：当前只能从 standard 切到 TorchAO specialization，或使用 HybridEP 的 `pad_multiple`；不能把任意 dispatcher 视为量化 kernel 的即插即用替代（`torchtitan/components/quantization/utils.py:33-64`）。

CUDA graph 配置也把选择边界写死：EP 只接受 non-blocking HybridEP 或 MinimalAsyncEP；其它 dispatcher 因 captured region 内 host sync 被拒绝，PP CUDA graphs 仍不支持（`torchtitan/config/configs.py:77-86`、`torchtitan/trainer.py:165-196`）。

---

## 8. 小结

- EP 的现行抽象是 `RoutedExperts` 协议，而不是旧 `ExpertParallel` 类；模型、dispatcher、expert kernel 与 sharding 因此可以分别替换。
- 参数平面是 `EP` 切专家逻辑维、`EFSDP` 管专家参数存储；激活平面则在 dense local-map 边界与 sparse EP collective 上下文之间切换。
- 四种 dispatcher 都是 live：standard 提供通用精确基线，DeepEP v2 区分 compact training/expand inference，HybridEP 融合 permute 并暴露容量取舍，MinimalAsyncEP 用 symmetric memory 换取静态 buffer 与同步边界。
- 旧 shared-experts overlap 结论已失效；当前 common eager forward 在 routed combine 完成后才计算 shared experts。
- MinimalAsyncEP 已在 API/mesh 层组合 CP、TP、PP，但当前 committed H100 integration 只覆盖 FSDP+CP+TP+EP，不应把 PP 支持误写成同等 CI 覆盖。

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 本系列入口、基线与阅读顺序。
- [[10_torchtitan_parallel_dims_analysis]] —— `dp_shard * cp * tp = efsdp * ep` 与 dense/sparse 双平面构网。
- [[12_torchtitan_tp_analysis]] —— TP/SP 如何先切 token 轴，再把本地 shard 交给统一 RoutedExperts 边界。
- [[13_torchtitan_cp_analysis]] —— CP 的序列切分为何会进入 persistent dispatcher 的 per-rank capacity 计算。
- [[16_torchtitan_spmd_types_analysis]] —— `local_map`、current mesh stack 与 dense/sparse type reinterpret 的机制。
- [[23_torchtitan_compute_memory_optimizations_analysis]] —— FusedSwiGLU、量化 grouped GEMM 与 full recompute 的计算/显存代价。
- [[24_torchtitan_comm_optimizations_overlap_analysis]] —— 超出 common eager EP 调用链的图调度与通信重叠优化。
