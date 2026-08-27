---
title: "FlexShard 与 DistMuon：把参数存储布局和优化器计算布局解耦"
---

# FlexShard 与 DistMuon：把参数存储布局和优化器计算布局解耦

> **代码基准**：torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`
> **最后更新**：2026-08-27 · **系列**：torchtitan 源码演进分析（见 [[torchtitan/index]]）
>
> **中心结论**：FSDP/EP/TP 决定参数长期怎样存，Muon 却要求完整矩阵或完整矩阵批次才能做正交化。FlexShard 在 `optimizer.step()` 内临时执行“storage layout → compute layout → storage layout”，并用打包 all-to-all 与双槽流水隐藏重分布开销。它不是另一套模型并行，也不会改变持久参数的 DTensor 布局。

---

## 1. 它解决的不是“再切一次参数”

FSDP 的目标是让参数、梯度和优化器状态长期以省显存的 DTensor 分片保存；Muon 的核心计算却以二维矩阵为单位。一个普通 `Shard(0)` 可能从矩阵中间切开行，一个专家权重又可能先按 EP 分专家、再按 EFSDP 切专家内部，两者的最佳布局并不一致。

FlexShard 因而显式区分两种布局：

| 平面 | 生命周期 | 权威来源 | 目标 |
|---|---|---|---|
| **storage layout** | 整个训练期持久存在 | 参数 DTensor 的 mesh + placements | 与 FSDP/EP/TP 组合，节省训练状态显存 |
| **compute layout** | 一次 optimizer bucket 内临时存在 | 每个 FQN 的 `ComputeLayout` | 给优化器完整矩阵、完整 block 或合适的矩阵批次 |

包内 README 直接把目标定义为“在不同于持久 DTensor 存储布局的布局上运行优化器计算”，并把 DistMuon 标成第一个消费者（`torchtitan/distributed/flex_shard/README.md:3-6`）。依赖边界也有意保持单向：TorchTitan 组件/模型 → FlexShard → PyTorch；FlexShard 不得反向依赖模型或 Trainer（`torchtitan/distributed/flex_shard/README.md:68-80`）。

这比把 Muon 硬塞进 FSDP placement 更合适：FSDP placement 是模型前后向和 checkpoint 的长期所有权契约，而 Muon layout 只是优化器瞬时的计算需求。把两者绑死会让一种优化器的矩阵边界污染整个训练布局。

## 2. `ComputeLayout`：按具名 mesh 轴声明临时所有权

公开配置面只有五个核心名字：`ComputeLayout`、`Owned`、`BlockShard`、PyTorch 的 `Replicate`/`Shard`，以及负责排程的 `BucketConfig`（`torchtitan/distributed/flex_shard/__init__.py:9-18`，`torchtitan/distributed/flex_shard/optimizer_reshard.py:18-59`）。

### 2.1 四种 compute sharding

| 声明 | compute 阶段含义 | 典型 Muon 用途 |
|---|---|---|
| `Owned()` | 一个动态选出的 rank 拿到 subgroup-local 完整逻辑张量 | 单个完整二维矩阵 |
| `Replicate()` | 每个参与 rank 都拿到完整逻辑张量 | 小张量或无需分工的计算 |
| `Shard(dim)` | 沿普通 tensor 维切分 | 原生 `[M,R,C]` 矩阵 batch 按 `M` 切 |
| `BlockShard(dim,block_size)` | 只在完整定长 block 之间切，绝不切开 block | 扁平 `[M*R,C]` 中每连续 `R` 行是一张矩阵 |

`Owned` 只描述 compute 阶段的动态 owner，不改变持久 DTensor 所有权；多个 `Owned` 轴会在轴的笛卡尔积上选一个 owner（`torchtitan/distributed/flex_shard/optimizer_reshard.py:21-31,110-125`）。`BlockShard` 则保持张量 rank 和全局 shape，不偷偷 reshape，只保证 block 不跨参与者（`torchtitan/distributed/flex_shard/optimizer_reshard.py:34-56`）。

`ComputeLayout.shardings_by_mesh_axis` 用**轴名**而不是轴下标声明布局。未出现的轴保留 storage placement；额外声明可以服务同一配置的其他 mesh 变体，但解析后至少要有一个声明真正生效（`torchtitan/distributed/flex_shard/optimizer_reshard.py:110-132`）。这让 Kimi 的同一套配置能面对 dense `[dp_shard,...]` 和 sparse `[efsdp,ep]` 两类 storage mesh。

### 2.2 同一 tensor 维被多轴切时，顺序也是语义

当 `efsdp` 和 `ep` 都对 tensor dim 0 使用 `Shard(0)`，仅知道 placements 还不够：先按 EP 切专家、再在 EP-local 专家域内按 EFSDP 切，与反过来不是同一分区。

`shard_order_by_tensor_dim={0:("ep","efsdp")}` 把顺序声明为“外层 EP，内层 EFSDP”。实现验证每个列出的轴确实对同一 tensor 维声明 `Shard`，拒绝重复轴、缺失轴和只含一个轴的伪顺序（`torchtitan/distributed/flex_shard/optimizer_reshard.py:127-145,196-244`）。这正是 FlexShard 能保留 EP 专家所有权、只重排 EFSDP-local 计算域的关键。

## 3. 从配置到传输计划：先证明区域映射，再打包通信

构造 `DistMuon` 时，每个本 stage 的参数 FQN 都必须匹配一个 `ComputeLayout`;其他 PP stage 的多余配置可忽略。参数组、FQN、compute layout 和 bucket 配置在构造后冻结，因为优化器状态和集合通信顺序都依赖它们（`torchtitan/distributed/flex_shard/dist_muon.py:170-205,252-269`）。

计划阶段可拆成四步：

```text
参数 DTensor storage regions
  → 解析 compute participants/regions
  → 求 source region 与 destination region 的路由
  → 按 bucket 合并所有参数的 spans
  → 生成双向 packed all-to-all schedule
```

`BucketConfig` 用大小写敏感的 FQN glob 选择参数；每个 FQN 必须恰好匹配一个 bucket，配置顺序就是执行顺序（`torchtitan/distributed/flex_shard/optimizer_reshard.py:251-287`）。完全 compute-ready 的 bucket 走 local plan，不创建通信组；其他 bucket 必须提供 1D communication mesh，并解析为 transport-neutral redistribution plan（`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:1055-1124`）。

打包器遍历每个参数的 region route，把同一 source→destination 的片段塞进连续 input/output buffer，同时计算 `input_split_sizes` 与 `output_split_sizes`;因此一个 bucket 的许多小参数不会各发一次 collective（`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:775-884`）。执行前还会对 redistribution plan 的稳定描述做 SHA-256 摘要并跨 rank all-gather；哈希不一致就拒绝进入热路径，避免不同 rank 用不同集合通信顺序死锁（`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:1211-1248`）。

## 4. `step()` 的双向流水：A2A 不是裸露在计算前后

`DistMuon.step()` 本身很薄：preflight 后把 bucket plans 与 `prepare/compute/finalize` 三个回调交给 `_BucketedRedistributionRuntime.run()`（`torchtitan/distributed/flex_shard/dist_muon.py:307-329`）。运行时维护两个 rolling buffer slot：一个计算当前 bucket，另一个预取下一 redistributed bucket；缓冲与 stream event 都在规划后预留，热路径只复用资源（`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:295-312,324-417`）。

稳态时序是：

```text
transfer stream:  storage→compute A2A(bucket N+1) ───── compute→storage A2A(bucket N-1)
compute stream :                  Muon compute(bucket N)
                                      ↑ event 依赖，不做 host 同步
```

具体依赖为：

1. transfer stream 把 storage regions 打包，执行 storage→compute packed all-to-all，记录 `compute_input_ready`（`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:558-586`）。
2. compute stream 先处理 bucket 中无需重分布的参数，再等待该 event，执行重分布参数的 Muon 计算，记录 `compute_done`（`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:589-618`）。
3. transfer stream 等 `compute_done`，反向执行 compute→storage packed all-to-all，把 update 写回原 storage region，记录 `done`（`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:620-637`）。
4. 主循环在计算当前 bucket 后尽量预取下一个，轮换两个 slot；local-only bucket 是明确的 prefetch barrier（`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:419-547`）。

所以“临时重排两次”不等于“每步先全局搬完、算完再搬回”。设计价值来自 bucket 化、双向打包和跨 bucket 重叠；单个 bucket 仍必须等自己的 storage→compute 通信完成才能计算。

## 5. DistMuon 在 compute tensor 上做什么

DistMuon 的更新链是：

```text
DTensor grad
  → momentum lerp / 可选 Nesterov
  → BF16 Newton-Schulz 近似极分解方向
  → decoupled weight decay
  → 按矩阵形状修正学习率后更新
  → 写回原 DTensor storage region
```

动量准备使用 `momentum_buffer.lerp_(gradient,1-momentum)`，Nesterov 时再对 gradient 与 momentum 做一次 `lerp`（`torchtitan/distributed/flex_shard/dist_muon.py:1828-1847`）。正交化先转 BF16、按 Frobenius norm 归一化，再做配置次数的 Newton-Schulz 迭代；2D 走 `addmm`，矩阵 batch 走 `baddbmm`（`torchtitan/distributed/flex_shard/dist_muon.py:1850-1866,1889-1918`）。最终更新把 decoupled weight decay 与按矩阵形状调整后的学习率分开应用（`torchtitan/distributed/flex_shard/dist_muon.py:1810-1825,1869-1886`）。

对扁平的 `[M*R,C]` 参数，`BlockShard(dim=0,block_size=R)` 保证每个 compute rank 收到整数张矩阵，随后 DistMuon 才做零拷贝 `[M_local,R,C]` view；原生 `[M,R,C]` 则用 `Shard(0)`。两者都避免把一张矩阵拆给两个 rank（`torchtitan/distributed/flex_shard/dist_muon.py:58-75,121-167`）。

## 6. TorchTitan 接线与当前边界

通用 optimizer container 已把 `DistMuon` 注册为与 Adam/AdamW 并列的 factory，实例级的 layout 和 bucket 对象通过 `optimizer_factory_kwargs_by_name` 传入，而不是混进每个 param group 的超参数（`torchtitan/components/optimizer/optimizer.py:127-150`）。

Kimi 配置是第一处完整接线：

- 普通完整矩阵用 `Owned(dp_shard)`；Q/KV head 拼接矩阵用不同 block size 的 `BlockShard`;专家再依据 EP/EFSDP 组合生成 per-expert layout（`torchtitan/models/kimi_k2_7/config_registry.py:287-330`）。
- attention、dense MLP、routed/shared experts 与 router 的矩阵交给 DistMuon，其余 embedding、norm、bias、LM head、vision tower 继续用 AdamW（`torchtitan/distributed/flex_shard/README.md:52-66`,`torchtitan/models/kimi_k2_7/config_registry.py:445-460`）。

当前边界必须显式记住：

| 边界 | 后果 | 代码证据 |
|---|---|---|
| DistMuon 只接受恰好一个 param group | 不同 Muon 超参数组尚不能混用 | `torchtitan/distributed/flex_shard/dist_muon.py:337-357` |
| 参数必须是同一进程单 CUDA device 上的 DTensor | CPU/plain Tensor 不走这条实现 | `torchtitan/distributed/flex_shard/dist_muon.py:359-369` |
| 每个配置参数在 `step()` 前必须有 DTensor gradient，布局不可变 | 条件不满足会在发起 bucket communication 前失败 | `torchtitan/distributed/flex_shard/dist_muon.py:486-543` |
| compute sharding 不写入 optimizer state dict | restore 必须用相同 layout/bucket 重新构造 | `torchtitan/distributed/flex_shard/README.md:48-50` |
| Kimi DistMuon 当前拒绝 TP>1 | TP 可产生尚未支持的 `_StridedShard` storage layout | `torchtitan/models/kimi_k2_7/config_registry.py:519-527` |
| PP 可用，但每 stage 必须至少有一个 Muon 参数 | 只含 norm/lm_head 的 stage 会匹配不到 Muon param group | `torchtitan/models/kimi_k2_7/config_registry.py:528-532` |

这些限制说明 FlexShard 已经是可运行的优化器重分布子系统，但 API 和覆盖面仍在成熟；README 也明确写着未来可能拆成独立包（`torchtitan/distributed/flex_shard/README.md:68-71`）。

## 7. 与 FSDP、SPMD Types 的关系

- **FSDP** 管 storage state machine：何时 all-gather 完整参数、何时 reduce-scatter 梯度；FlexShard 只在 optimizer step 内改变计算所有权，结束后仍回到同一 storage DTensor。
- **SPMD Types** 给每个参数稳定的具名 storage mesh 与 placements；FlexShard 的 `ComputeLayout` 正是用这些轴名定位“哪一轴需要临时重排”。
- **DistMuon** 是第一个 consumer，不是 FlexShard 的同义词。公共 `ComputeLayout`/planner/runtime 有意保持 optimizer-agnostic，未来其他需要特殊计算布局的优化器可以复用。

## Related Pages

- [[16_torchtitan_spmd_types_analysis]] —— storage mesh、placement 与模块级布局契约
- [[11_torchtitan_fsdp_analysis]] —— 持久参数分片、梯度规约与 optimizer storage
- [[15_torchtitan_ep_analysis]] —— EP/EFSDP 的专家存储布局与 token dispatch
- [[22_muon_sharded_hsdp_analysis]] —— 另一条分片 Muon/HSDP 设计路线，可对照所有权与通信策略
- [[23_torchtitan_compute_memory_optimizations_analysis]] —— 低精度、融合算子和 optimizer 计算优化全景
- [[torchtitan/index]] —— TorchTitan 知识地图与本轮代码演进审计
