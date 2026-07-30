# F04 · FSDP、DTensor 与 Distributed Graphs

> 卷别：F · 训练、分布式、扩展与部署  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[ddp_compile_boundaries_and_optimizer_analysis]]  
> 后续：[[custom_operators_fake_kernels_and_decompositions_analysis]]  
> 最后更新：2026-07-30(kb-reorg P4 Task 9 迁入本目录,与 [[c10d_ddp_fsdp_dtensor_analysis]] 互指划界)

> [!note] 与 [[c10d_ddp_fsdp_dtensor_analysis]] 的分工
> 该页讲 FSDP/DTensor **原语本身**——`FlatParameter` 怎样 shard/unshard/reshard、DTensor 的 placement 怎样传播与插入通信,不涉及 `torch.compile`。本页讲这些原语与**编译器相遇时**新增的一层问题:为什么 FSDP1 编译要求 `use_orig_params=True`、Dynamo 为什么跳过 FSDP wrapper frame、每 rank local graph 与 collective 顺序如何在编译期保持一致。理解顺序同 F03:先读该页知道原语本身怎样工作,再读本页知道编译器如何处理它。

## 1. 分布式图不是“一张全局 FX 图”

典型SPMD训练中，每个rank运行自己的Python进程和local FX graph。跨rank依赖由collective及其
顺序表达，而不是一张Graph的普通Node边：

```text
rank 0 local graph ─┐
rank 1 local graph ─┼─ collective sequence / process group ─ global semantics
rank N local graph ─┘
```

因此正确性需要同时满足：

- 每rank local dataflow正确；
- placement/shard metadata正确；
- collective参与集合、顺序、shape、dtype一致；
- reshard/unshard时机与liveness正确。

## 2. FSDP 的状态机

FSDP围绕parameter经历：

```text
local shard
→ all-gather / unshard
→ compute 使用完整参数
→ reshard / free full parameter
→ backward prefetch/unshard
→ grad reduce-scatter
→ sharded grad / optimizer
```

这些是runtime通信和storage转换；即使FSDP内部frames被Dynamo跳过，边界前后的module参数
身份、view和hooks仍决定被捕获子图。

## 3. 为什么 FSDP1 编译要求 `use_orig_params=True`

FSDP公开说明该模式保留原Parameter对象，但其`.data`在unsharded和sharded FlatParameter
view之间切换；某rank可能只拥有部分或空数据。当前实现要求它才能使用`torch.compile`
（`torch/distributed/fsdp/fully_sharded_data_parallel.py:361-378`）。

Dynamo annotation说明：

- FSDP frames被跳过，以避免trace复杂hooks并在通信边界切图；
- wrapped submodule被标成FSDP-managed和UnspecializedNNModule；
- 每轮新建的orig-param views必须作为参数输入使用，避免首次view被固化在图中，从而破坏
  backward compute/communication交错；
- Dynamo只支持FSDP `use_orig_params=True`。

见 `torch/distributed/fsdp/_dynamo_utils.py:4-27` 与
`torch/distributed/fsdp/_dynamo_utils.py:29-43`。

## 4. 为什么 FSDP 内部通常不是全被捕获

跳过复杂FSDP wrapper frame、捕获wrapped module compute，是一种边界设计：

- 通信和parameter lifecycle留在明确runtime层；
- 纯计算子模块交给Dynamo/AOT/Inductor；
- unshard/reshard之间形成可审查边界；
- 避免Python hook/control state全部进入一张图。

代价是图间Python/runtime overhead，以及跨边界fusion受限。是否把更多collective函数化入图
取决于具体FSDP版本和配置，不能泛化为“FSDP所有通信都在FX图里”。

## 5. Reshard、释放与 compile 的差异

`_reshard`会恢复sharded参数并按策略释放unsharded flat parameter。非compile路径在
`limit_all_gathers`下还会记录free event queue；compile时当前实现不运行这套event queue
（`torch/distributed/fsdp/_runtime_utils.py:309-330`）。

这说明eager与compile在host-side memory scheduling上可能不同。验收必须看真实峰值和
prefetch，而不是只比较collective算子数量。

## 6. Composable FSDP2 的接入

`fully_shard`初始化mesh、managed modules、parameter group、post-forward reshard和mixed
precision/offload策略，然后给managed module写入Dynamo annotations
（`torch/distributed/fsdp/_fully_shard/_fully_shard.py:250-267`、
`torch/distributed/fsdp/_fully_shard/_fully_shard.py:268-285` 与
`torch/distributed/fsdp/_fully_shard/_fully_shard.py:287-296`）。

相较FSDP1，参数常以DTensor表达shard placement，module通过mixin改变运行状态。学习时仍要
分开：

- Python module变换；
- parameter global/local表示；
- collective执行；
- Dynamo capture boundary；
- AOT fw/bw与saved tensors；
- optimizer看到的local state。

## 7. DTensor 的数据模型

DTensor是Tensor subclass，核心持有：

- `_local_tensor`：当前rank真实Tensor；
- `_spec`：`DeviceMesh + placements + global metadata`。

placement包括Shard、Replicate和Partial；operator dispatch会传播placement并在必要时发起
通信
（`torch/distributed/tensor/_api.py:356-380`）。

其PT2 flatten协议把local tensor和device mesh暴露给trace，并把placements/shape/stride
作为context
（`torch/distributed/tensor/_api.py:403-430`）。

因此图中可能同时看到local Tensor计算、DTensor相关HOP/redistribution和collective；具体
形态取决于trace/decomposition边界。

## 8. Sharding propagation 如何决定通信

每个operator根据输入`DTensorSpec`选择输出placement/strategy。若实际输入spec与建议spec
不同，dispatch在local args层插入redistribution，并可记录隐式reshard
（`torch/distributed/tensor/_dispatch.py:575-603` 与
`torch/distributed/tensor/_dispatch.py:604-629`）。

这意味着“用户代码没有显式collective”不代表没有通信：placement不兼容会在operator
dispatch中引入reshard。

ShardingPropagator维护op→rule、op→strategy和带schema信息的cache；shape/stride参数还需按
global/local语义调整
（`torch/distributed/tensor/_sharding_prop.py:362-391` 与
`torch/distributed/tensor/_sharding_prop.py:392-421`）。

## 9. `from_local` / `to_local` 是 autograd 边界

`from_local`的输入必须是真实Tensor；`run_check=False`时由用户保证各rank local shard/
replica正确。它通过autograd Function创建DTensor，让梯度流回local tensor
（`torch/distributed/tensor/_api.py:569-598` 与
`torch/distributed/tensor/_api.py:600-621` 与
`torch/distributed/tensor/_api.py:622-643`）。

`to_local`也可微，并允许指定未来gradient placement
（`torch/distributed/tensor/_api.py:645-662` 与
`torch/distributed/tensor/_api.py:663-680`）。

所以它们不是无语义的包装/拆包，错误的grad placement会改变backward通信。

## 10. `redistribute` 的图语义

placement转换对应：

- Shard→Replicate：all-gather；
- Shard(dim A)→Shard(dim B)：all-to-all；
- Replicate→Shard：local chunk；
- Partial→Replicate：all-reduce；
- Partial→Shard：reduce-scatter。

见 `torch/distributed/tensor/_api.py:710-736`。该API是out-of-place且autograd-aware，并可指定
forward/backward collective dtype
（`torch/distributed/tensor/_api.py:738-753` 与
`torch/distributed/tensor/_api.py:754-769`）。

编译器若改变redistribution位置，必须保持collective/effect顺序、dtype和placement语义。

## 11. Rank 一致性与 guards

以下状态应被部署契约固定或跨rank一致：

- world size、mesh维度与rank坐标；
- placements和shard dim；
- global/local shape与stride；
- parameter shard ownership；
- collective process group；
- train/eval、requires-grad和unused parameter路径；
- activation checkpoint/partition选择；
- Dynamo specialization与graph break路径。

某rank guard miss并独立重编译不一定立即错误，但若因此改变collective序列，可能hang或数据
错误。监控必须比较各rank compile id和collective timeline。

## 12. 复杂度与性能

以all-gather/reduce-scatter为例，通信成本粗略为延迟项加带宽项：

\[
T_{\text{collective}}\approx \alpha\cdot f(P)+\beta\cdot \text{bytes}\cdot g(P)
\]

FSDP的峰值还包含相邻unsharded parameter、prefetch和activation。图编译可减少local compute
与launch overhead，但不能消除必要通信量；真正目标是减少隐式reshard、优化bucket并与
compute重叠。

## 13. 常见误解

- **“分布式只有一张全局FX图。”** 通常是每rank local graph加collective协议。
- **“FSDP wrapper没进图就与编译无关。”** parameter views与边界决定捕获和backward顺序。
- **“DTensor只是local Tensor加标签。”** placement传播会主动插入collective。
- **“from_local/to_local只是cast。”** 它们定义autograd gradient placement。
- **“kernel融合更好就一定step更快。”** reshard、collective和最慢rank可能主导。

## 配套 Demo

本页对应卷级入口 `tools/labs_torch_compile/demo_f_advanced_topics.py` 的 `fsdp_dtensor` 用例。默认以 CUDA 为验收设备：

```powershell
python -B tools\labs_torch_compile\demo_f_advanced_topics.py `
  --case fsdp_dtensor --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\f04
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `fsdp_dtensor/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[ddp_compile_boundaries_and_optimizer_analysis]]
- [[custom_operators_fake_kernels_and_decompositions_analysis]]
- [[04_export_and_distributed/02_distributed_primitives/index]]
- [[12_graph_effects_alias_mutation_and_order_analysis]]
- [[19_production_rollout_fallback_and_monitoring_analysis]]
