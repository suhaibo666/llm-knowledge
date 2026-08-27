---
title: "通信优化与计算-通信掩盖：编译期 Async-TP、FSDP 对称内存与 EP dispatcher"
---

# 通信优化与计算-通信掩盖：编译期 Async-TP、FSDP 对称内存与 EP dispatcher

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-27）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **主线**：TorchTitan 当前没有一个统一的“通信重叠后端”。Async-TP 是 `CompileConfig` 驱动的 Inductor 微流水入口；FSDP symmetric memory 是 `fully_shard` 后逐模块切换通信实现；EP 则在统一的 token-dispatcher 接口下选择标准 all-to-all、DeepEP v2、HybridEP 或 MinimalAsyncEP。三者共享的不是同一条执行流，而是“在准确的 mesh/process group 上，把通信边界交给专用运行时”的设计原则。
>
> 本页只分析 TorchTitan 能直接核验的接线、同步边界、组合关系与失败条件。FSDP/TP/EP 各自的算法与通用重叠原理见 Related Pages；PyTorch Inductor 与 symmetric-memory 内核内部不属于本次源码基准。
>
> 主要源文件：`torchtitan/distributed/compile.py`、`torchtitan/distributed/fsdp.py`、`torchtitan/models/common/token_dispatcher.py`、`torchtitan/distributed/minimal_async_ep/api.py`、`torchtitan/distributed/deepep/deepep.py`。

---

## 1. 先纠正旧版知识

旧页基于 `61c010fcb`，其中四条结论已经失效。

### 1.1 `full_dtensor` 已删除，不是当前通信后端

提交 `601cf4d23` 删除了 `torchtitan/distributed/full_dtensor.py`。当前 `ParallelismConfig.spmd_backend` 只接受 `partial_dtensor` 与 `spmd_types`，默认是 `spmd_types`；前者仍是“只让模型并行轴使用 DTensor”的兼容路径（`torchtitan/config/configs.py:174-180`、`torchtitan/config/configs.py:261-266`）。

> [!deprecated] 旧版 `full_dtensor` 章节已失效
> 当前 dense mesh 的两条分支是：`spmd_types` 同时建立 FSDP storage mesh 与 `[dp, cp, tp]` 前后向 mesh；`partial_dtensor` 则把 `dp_shard × cp` 折为 `fsdp` 轴（`torchtitan/distributed/parallel_dims.py:229-260`）。这是一项布局/执行后端选择，不应继续列成与 Async-TP、symmetric memory 并列的通信优化。

### 1.2 Async-TP 已迁到 compile 子系统

提交 `737594746` 将 Async-TP 修复并迁入 compile。当前开关属于 `CompileConfig.enable_async_tensor_parallel`，并在配置构造时强制要求 `compile.enable=True` 且 `components` 包含 `model`（`torchtitan/config/configs.py:295-315`）；实际接线在 `torchtitan/distributed/compile.py`，不再是旧页所引的 `distributed/tensor_parallel.py`（`torchtitan/distributed/compile.py:39-52`、`torchtitan/distributed/compile.py:75-96`）。

### 1.3 MinimalAsyncEP 不再禁止 TP/CP/PP

当前 MinimalAsyncEP 配置验证只要求 EP>1、`num_experts % EP == 0`、存在 training runtime config，并启用 full recompute；这里没有 `TP=CP=PP=1` 约束（`torchtitan/distributed/minimal_async_ep/api.py:84-140`）。H100 配方与集成测试已经实际覆盖 FSDP2+TP2+CP2+EP8+MinimalAsyncEP（`torchtitan_recipes/tests/h100.py:63-77`、`tests/integration_tests/h100.py:57-62`）。PP 的边界需要单独讨论，见 §5.5。

### 1.4 当前 DeepEP combine 没有与 shared experts 重叠

DeepEP 底层 combine 仍返回异步结果并要求调用者 `sync_combine()`（`torchtitan/distributed/deepep/deepep.py:560-568`），但当前 `DeepEPTokenDispatcher.combine()` 立即执行该同步后才返回（`torchtitan/models/common/token_dispatcher.py:865-878`）。上层 MoE 又是在 routed experts 整体返回后才计算 shared experts（`torchtitan/models/common/moe.py:440-452`），所以旧页“DeepEP combine 与 `shared_experts(x)` 重叠”不是当前调用链。

---

## 2. 当前通信优化分层

| 层 | 当前入口 | TorchTitan 实际负责什么 | 不应混淆成什么 |
|---|---|---|---|
| 编译期算子微流水 | `compile.enable_async_tensor_parallel` | 选择 TP process group、注册 symmetric memory、打开 Inductor `_micro_pipeline_tp` | 普通 TP eager `async_op` |
| FSDP 通信实现 | `parallelism.enable_fsdp_symm_mem` | 对所有 `FSDPModule` 打开 sum-reduction 与 symmetric-memory comm | 通用 compile pass |
| FSDP 调度 | reshard policy、EP explicit prefetch | 决定参数何时 reshard、下一个模块何时预取 | symmetric-memory 本身 |
| EP 通信后端 | token dispatcher config | 在统一 dispatch→experts→combine 边界选择标准、DeepEP v2、HybridEP、MinimalAsyncEP | 一个统一“异步 EP”语义 |
| 实验编译调度 | `experiments/graph_trainer` passes | 对 SimpleFSDP/EP/TP 图做 bucketing、重排与 process-group 隔离 | 主 Trainer 默认路径 |

`spmd_types` 是上述机制的默认布局底座：它把前后向值放到 `[dp,cp,tp]` 逻辑 mesh 上，而 FSDP 仍取得带 `dp_replicate/dp_shard/cp/tp` 的 storage mesh（`torchtitan/distributed/parallel_dims.py:230-253`）。EP 则使用 `[pp,dp_replicate,efsdp,ep]` sparse mesh，满足 `dp_shard × cp × tp == efsdp × ep`，PP 与 DP-replicate 是外层维度（`torchtitan/config/configs.py:284-291`、`torchtitan/distributed/parallel_dims.py:262-279`）。

---

## 3. Async Tensor Parallel：当前真实调用链

### 3.1 接线顺序

以 Llama 3 为例，模型先安装 SPMD/TP sharding，再应用 activation checkpointing，随后在 FSDP 之前逐 TransformerBlock compile（`torchtitan/models/llama3/parallelize.py:40-55`）：

```text
ParallelismConfig / CompileConfig
  -> model.parallelize(parallel_dims)
  -> activation checkpointing
  -> apply_compile(model, compile_config, parallel_dims)
       -> get_dense_tp_mesh()
       -> _maybe_enable_async_tp()
       -> compile each TransformerBlock(fullgraph=True)
  -> apply_fsdp_to_decoder(...)
```

`apply_compile()` 只有在 TP 已启用时才传入 dense TP mesh；`_maybe_enable_async_tp()` 用该 mesh 的 process-group name 调用 `enable_symm_mem_for_group()`，然后设置 `torch._inductor.config._micro_pipeline_tp=True`（`torchtitan/distributed/compile.py:49-52`、`torchtitan/distributed/compile.py:75-96`）。单测同时断言了 process-group 注册和 Inductor flag（`tests/unit_tests/gpu/test_compile_moe.py:64-91`）。

### 3.2 “Async” 在本仓可证明到哪一层

TorchTitan 的配置文档把该开关定义为“让 TP collectives 与矩阵乘流水”（`torchtitan/config/configs.py:300-301`），但本仓的主路径只负责 symmetric-memory 注册和 Inductor flag。all-gather+matmul、matmul+reduce-scatter 如何匹配、切块和排程属于所依赖 PyTorch 版本的 Inductor 实现，不能从 TorchTitan 基准继续推导固定内核细节。

这也解释了两个边界：

1. 开关打开但 TP=1 时，`tp_mesh=None` 会让 `_maybe_enable_async_tp()` 直接返回；配置校验只保证 compile 条件，不保证 TP 度数大于 1（`torchtitan/distributed/compile.py:75-81`）。
2. Async-TP 不是只能单独使用。当前 H100 配方把 FSDP2、TP2、PP2、Float8、compile 与 Async-TP 放在同一配置中，并显式关闭 CUDA Graphs（`torchtitan_recipes/tests/h100.py:41-51`）。

### 3.3 graph_trainer 是另一条编译接线

实验性 graph_trainer 直接在 FX graph 中寻找 all-gather/reduce-scatter process group，为每个 group 注册 symmetric memory，再调用 PyTorch `micro_pipeline_tp_pass`（`torchtitan/experiments/graph_trainer/passes.py:95-130`）；该 pass 仅在 graph-trainer compile 配置打开 Async-TP 时加入 pass list（`torchtitan/experiments/graph_trainer/passes.py:337-338`）。它与主 Trainer 的“设全局 Inductor flag后编译各 block”不是同一个接线点。

---

## 4. FSDP symmetric memory：模块通信实现，不是编译微流水

### 4.1 当前接线

`parallelism.enable_fsdp_symm_mem` 默认关闭（`torchtitan/config/configs.py:162-166`）。模型 parallelize 函数把它透传给 `apply_fsdp_to_decoder()`；例如 Llama 3 在解析 FSDP mesh 后传入该参数（`torchtitan/models/llama3/parallelize.py:57-78`）。

decoder 的全部 block 与 root model 完成 `fully_shard()` 后，开关才触发 `enable_fsdp_symm_mem(model)`（`torchtitan/distributed/fsdp.py:355-371`）。该函数遍历所有 `FSDPModule`，对每个模块调用：

- `set_force_sum_reduction_for_comms(True)`；
- `set_symm_mem_for_comm()`（`torchtitan/distributed/fsdp.py:102-109`）。

这就是 TorchTitan 当前能做出的完整机制断言。具体 all-gather/reduce-scatter 内核与传输协议由当前 PyTorch FSDP 实现决定；旧页基于本机某个 PyTorch 版本猜测 `set_symm_mem_for_comm()` 是否存在、是否使用 multimem，不应继续保留。

### 4.2 硬件与组合边界

配置层会拒绝无 CUDA 的环境；对 NVIDIA 还要求 compute capability 至少 9.0，HIP 路径不走该 capability 判断（`torchtitan/config/configs.py:272-281`）。当前 H100 配方提供单独的 FSDP symmetric-memory 覆盖入口（`torchtitan_recipes/tests/h100.py:35-38`）。源码没有检查 NVLink 拓扑，因此不能把“必须 NVLink”写成 TorchTitan 已验证约束。

FSDP 的重叠调度仍是另一层逻辑：

- PP 下默认不在 forward 后 reshard，目的是避免每个 microbatch 都产生昂贵且不可掩盖的 all-gather（`torchtitan/distributed/fsdp.py:112-132`）。
- EP 打开时，代码显式设置前向与反向 prefetch，因为 EP 的 device-to-host 同步可能干扰 FSDP 隐式预取（`torchtitan/distributed/fsdp.py:384-424`）。

因此 symmetric memory 只改变 FSDP module 的通信实现；reshard policy 与 prefetch 决定通信何时发生，二者不能合并成一个开关解释。

---

## 5. EP token dispatcher：四条通信路径，一套计算边界

### 5.1 统一边界

`RoutedExperts.forward()` 的固定顺序是 dispatcher `dispatch()` → local experts → dispatcher `combine()`（`torchtitan/models/common/moe.py:145-169`）。`make_token_dispatcher_config()` 当前接受 `standard`、`deepep`、`hybridep`、`minimal_async_ep` 四种 backend（`torchtitan/models/common/config_utils.py:358-422`）。因此 dispatcher 只拥有 token 重排、跨 EP rank 传输、combine 及其元数据；专家 GEMM 和 shared experts 不属于 dispatcher 的通信实现。

### 5.2 standard：可见的同步成本

标准路径先交换各 expert token count，再做可变长 token all-to-all。count 结果必须显式 `wait_tensor()`，远端 output splits 又同步拷回 CPU 后才能启动数据 all-to-all（`torchtitan/models/common/token_dispatcher.py:245-308`）。数据 dispatch/combine 在 compile 或 `partial_dtensor` 下调用 functional `all_to_all_single`，默认 `spmd_types` eager 路径调用 `spmd.all_to_all`（`torchtitan/models/common/token_dispatcher.py:316-370`）。

所以标准 dispatcher 的瓶颈不能概括成“AsyncCollectiveTensor 自动延迟 wait”：当前源码明确存在 count wait 与 D2H split materialization。

### 5.3 DeepEP v2：统一 ElasticBuffer，但当前调用者同步 combine

当前 DeepEP 模块要求 DeepEP v2 `ElasticBuffer`，把旧 v1 的 high-throughput/low-latency 两套 API 合并为一对 dispatch/combine；training 走 compact、host-synced、可反向路径，no-grad inference 在 `cudagraphable=True` 时走静态 expand、无 host sync 路径（`torchtitan/distributed/deepep/deepep.py:7-36`）。buffer 根据 group、hidden、top-k 与每 rank 最大 token 数预分配，group 变化或容量不足才重建（`torchtitan/distributed/deepep/deepep.py:335-381`）。

通信层的 combine 本可返回 pending event，但 dispatcher 立即 `sync_combine()`（`torchtitan/models/common/token_dispatcher.py:865-878`）。因此当前页面只把 DeepEP v2 视为 EP 通信后端与静态/动态布局选择，不再声称它在主 Trainer 中与 shared-expert 计算重叠。

### 5.4 HybridEP：blocking 与 non-blocking 是容量/同步权衡

HybridEP 默认 `non_blocking_capacity_factor=None`：dispatch 后同步 CUDA stream，把每 expert token 数复制到 pinned CPU 并在 host 计算精确输出尺寸，不丢 token；设为 `(0,1]` 才进入 CPU-free non-blocking 模式，但容量因子小于 1 时，超出 fused-permute 容量的 token 会被丢弃并在 GPU 上设置 overflow flag（`torchtitan/models/common/token_dispatcher.py:888-920`）。dispatcher 把这个因子连同 padding 配置传给 HybridEP `dispatch_tokens()`（`torchtitan/models/common/token_dispatcher.py:955-984`）。这也是 CUDA Graph validator 只接受设置了 non-blocking capacity 的 HybridEP，而不接受默认 blocking 模式的原因（`torchtitan/trainer.py:178-195`）。

### 5.5 MinimalAsyncEP：同步 custom-op 边界，组合约束已改变

MinimalAsyncEP 为 EP group 建立进程内唯一的 fixed-capacity symmetric-memory buffer；后端必须是 CUDA symmetric memory，buffer 经 rendezvous 获取所有 peer 的可寻址视图（`torchtitan/distributed/minimal_async_ep/api.py:174-225`、`torchtitan/distributed/minimal_async_ep/api.py:227-295`）。dispatch custom op 直接把 token 行复制到目标 rank 的 expert-major receive rows，对应标准 dispatcher 的 token all-to-all（`torchtitan/distributed/minimal_async_ep/api.py:558-634`）。

但这个实现的 custom-op 边界是同步的：copy 完成后 barrier/wait，再把可读 buffer 返回；源码明确说明它不提供 microbatch communication overlap（`torchtitan/distributed/minimal_async_ep/api.py:300-342`）。当前 barrier 用单个 fused barrier 同时 signal/poll 全部 peer，取代旧的逐 peer Python kernel 循环（`torchtitan/distributed/minimal_async_ep/api.py:345-365`）。因此名字里的 “Async” 不能被解释成“与 experts/shared experts 自动重叠”。

当前组合边界应拆成三层：

1. **MinimalAsyncEP 自身硬约束**：EP>1、expert 数可被 EP 整除、存在 training config、full activation checkpoint 或 graph-trainer `memory_policy="full"`（`torchtitan/distributed/minimal_async_ep/api.py:84-140`）。
2. **TP/CP 已有集成证据**：dispatcher 文档明确把 CP/TP token sharding 交给公共 MoE sharding路径；H100 测试配方覆盖 TP2+CP2（`torchtitan/models/common/token_dispatcher.py:1009-1014`、`torchtitan_recipes/tests/h100.py:63-77`）。
3. **PP 是条件式组合**：sparse mesh 把 PP 放在 EP group 外层，因此每个 pipeline coordinate 有自己的 EP group（`torchtitan/distributed/parallel_dims.py:262-287`）；但 Trainer 在 CUDA Graphs 开启时无条件拒绝 PP>1，所以 PP+MinimalAsyncEP 必须关闭 CUDA Graphs（`torchtitan/trainer.py:165-173`）。当 PP=1 时，MinimalAsyncEP 是当前被 CUDA Graph validator 明确允许的 CPU-free EP dispatcher 之一（`torchtitan/trainer.py:175-195`）。

当前仓库有 TP/CP+MinimalAsyncEP 的集成用例，但没有同页可定位的 PP+MinimalAsyncEP 集成配方；因此这里把 PP 写成源码结构允许、受 CUDA Graph 条件限制，而不是写成已端到端验证。

---

## 6. 当前能力矩阵

| 路径 | compile 必需 | 当前同步/重叠边界 | 主要失败条件 | 已核验组合 |
|---|---:|---|---|---|
| Async-TP | 是 | TorchTitan 注册 TP group symmetric memory并打开 Inductor 微流水；内部排程在 PyTorch | compile/model component 缺失；TP=1 时无操作 | FSDP+TP+PP+Float8 配方 |
| FSDP symmetric memory | 否 | `fully_shard` 后逐 `FSDPModule` 切换通信方法；prefetch/reshard 另行控制 | 无 CUDA；NVIDIA capability <9.0 | H100 FSDP 配方 |
| standard EP | 否 | count exchange 显式 wait，output splits D2H 同步后再 all-to-all | 可变 splits 的 host sync | `spmd_types` 与 `partial_dtensor` 均有分支 |
| DeepEP v2 | 否 | compact training host-sync；static expand inference 可无 host sync；dispatcher 返回前同步 combine | DeepEP v2 依赖、预分配容量 | 统一 ElasticBuffer |
| HybridEP | 否 | 默认 blocking 获取精确容量；non-blocking 用固定容量换 CPU-free dispatch | 外部依赖；capacity<1 可能丢 token | non-blocking 模式可配 CUDA Graphs |
| MinimalAsyncEP | 否 | symmetric-memory 直写 + fused barrier；custom-op 返回时数据可读，不做 microbatch overlap | CUDA symm mem、EP/整除、full recompute、固定 buffer config | FSDP+TP+CP+EP；PP 需关 CUDA Graphs |

`spmd_types`/`partial_dtensor` 不出现在“重叠方式”列，因为它们决定布局表达和 collective 接口，不等同于某一种通信优化。默认 `spmd_types` 与兼容 `partial_dtensor` 的状态以当前配置枚举为准（`torchtitan/config/configs.py:174-180`）。

---

## 7. 结论与失败边界

1. **Async-TP 的 ownership 已从 parallelism 移到 compile**：配置验证、TP group symmetric-memory 注册和 Inductor flag 都在 compile 子系统；具体融合规则属于 PyTorch 依赖，不应继续引用旧 TorchTitan `tensor_parallel.py`（`torchtitan/config/configs.py:295-315`、`torchtitan/distributed/compile.py:75-96`）。
2. **FSDP symmetric memory 与 Async-TP 共享技术底座但不共享开关**：前者调用 FSDP module API，后者注册 TP process group；配置与硬件验证也彼此独立（`torchtitan/distributed/fsdp.py:102-109`、`torchtitan/distributed/compile.py:83-95`）。
3. **EP 后端的差异首先是同步边界和布局**：standard 有 count/D2H 同步，DeepEP v2 区分 compact/expand，MinimalAsyncEP 直写 fixed-capacity receive buffer但同步返回；不能把四条路径概括成同一种 ACT/stream 重叠（`torchtitan/models/common/token_dispatcher.py:275-308`、`torchtitan/distributed/deepep/deepep.py:20-31`、`torchtitan/distributed/minimal_async_ep/api.py:309-315`）。
4. **组合判断必须分层**：MinimalAsyncEP 已明确覆盖 TP+CP；PP group 结构可组合，但 CUDA Graphs 当前拒绝 PP，且仓库缺少 PP+MinimalAsyncEP 的端到端配方（`tests/integration_tests/h100.py:57-62`、`torchtitan/trainer.py:165-173`）。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/12_torchtitan_tp_analysis|TorchTitan TP 分析]] —— TP/SP 布局、collective 与 Async-TP 上下文
- [[02_engineering/02_train_frameworks/torchtitan/15_torchtitan_ep_analysis|TorchTitan EP 分析]] —— EP mesh、专家布局与 dispatcher 全流程
- [[02_engineering/02_train_frameworks/torchtitan/16_torchtitan_spmd_types_analysis|TorchTitan SPMD Types 分析]] —— 默认 `spmd_types` 后端与 `partial_dtensor` 对照
- [[02_engineering/02_train_frameworks/torchtitan/20_torchtitan_fsdp_prefetch_overlap_memory_analysis|TorchTitan FSDP 预取、重叠与内存]] —— FSDP 通信调度与生命周期
- [[02_engineering/02_train_frameworks/torchtitan/25_torchtitan_simple_fsdp_analysis|TorchTitan SimpleFSDP 分析]] —— graph_trainer 可追踪 FSDP 与编译 pass
- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 系列入口与功能树定位
