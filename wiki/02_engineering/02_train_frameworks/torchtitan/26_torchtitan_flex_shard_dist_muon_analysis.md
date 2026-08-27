---
title: "FlexShard 与 DistMuon：存储所有权不等于优化器计算所有权"
---

# FlexShard 与 DistMuon：存储所有权不等于优化器计算所有权

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页论点**：FlexShard 解决的是“参数长期怎样存”与“优化器临时需要谁来算”不一致，而不是再发明一种模型并行。它在 `optimizer.step()` 内把持久 DTensor storage layout 映射到临时 compute layout，以矩形 region 证明双向路由，用 bucket、packed all-to-all 和两个复用槽重叠搬运与计算，最后把更新送回原 storage layout；DistMuon 只是当前第一个 consumer，不是 FlexShard 的同义词。
>
> 本页回答 storage/compute ownership 如何分离、为什么 QKV 选择“先重分布再 view”、EP/EFSDP 为什么必须显式声明 shard order、计划与双槽运行时如何保证正确性，以及 Kimi 的 DistMuon 接线与失败边界。FSDP 的参数生命周期、EP 的 token/expert 语义、SPMD Types 的前后向值布局分别由 sibling 页负责。

---

## 1. Overview：优化器矩阵边界不能反向支配参数存储

FSDP、EP 与 TP 面向训练全生命周期安排参数、梯度和优化器状态；Muon 的正交化却以完整二维矩阵或完整矩阵批次为计算单位。一个普通 `Shard(0)` 可以从一张 head matrix 中间切开，专家参数又可能在 `[efsdp, ep]` storage mesh 上分别沿矩阵内部和 expert 维分片。若直接把 storage placement 当作 Muon 的矩阵所有权，前后向节省显存的布局就会与 optimizer kernel 的边界冲突；当前 README 也明确区分持久 storage placement 与 Muon matrix boundary（`torchtitan/distributed/flex_shard/README.md:23`、`torchtitan/distributed/flex_shard/README.md:35`）。

TorchTitan 因而把 FlexShard 定义为“在不同于持久 DTensor storage layout 的布局上执行优化器计算”，并明确说 DistMuon 是第一个 consumer（`torchtitan/distributed/flex_shard/README.md:3`、`torchtitan/distributed/flex_shard/README.md:6`）。公共包同时导出 optimizer-agnostic 的 `ComputeLayout`、`Owned`、`BlockShard`、`BucketConfig` 和 DistMuon builder；这组 API 边界本身就否定了“FlexShard 就是 DistMuon”的旧心智模型（`torchtitan/distributed/flex_shard/__init__.py:7`、`torchtitan/distributed/flex_shard/__init__.py:9`）。

| 概念 | 生命周期 | 权威来源 | 它不负责什么 |
|---|---|---|---|
| storage layout | 跨 step 持久存在 | 参数 DTensor 的 named mesh、placements 与 local storage | 不定义 Muon 的矩阵边界 |
| compute layout | 一个 optimizer bucket 内临时存在 | FQN 对应的 `ComputeLayout` | 不改 checkpoint 中的参数所有权 |
| region plan | optimizer 构造或重载后冻结 | storage/compute partitions 与双向 routes | 不在热路径重新猜路由 |
| bucket schedule | 多参数共享的通信与执行顺序 | `BucketConfig` 与 resolved transport group | 不允许一个 bucket 混用异构通信组 |
| DistMuon | FlexShard 上的 Muon 算法 consumer | momentum、Newton-Schulz、weight update | 不拥有通用 redistribution API |

这张表的生命周期边界由两处当前契约固定：compute sharding 在 optimizer 构造时验证并冻结、但不写入 state dict；runtime callback 则只能在槽拥有期间消费临时 tensor（`torchtitan/distributed/flex_shard/README.md:48`、`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:298`、`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:303`）。

关键状态流不是“reshard 后留下新参数”，而是：

```text
persistent DTensor parameter / gradient / momentum
                 |
                 | prepare：在 storage ownership 上更新 momentum
                 v
        packed storage regions
                 |
                 | storage -> compute all-to-all
                 v
    temporary compute tensor ownership
                 |
                 | optional zero-copy matrix-batch view
                 | Newton-Schulz direction
                 v
       packed compute regions
                 |
                 | compute -> storage all-to-all
                 v
update original local DTensor storage in place
```

### Quick Start：从 Kimi 配置追到一次 `step()`

Kimi 不是直接调用内部 planner，而是给 optimizer container 两份实例级元数据：每个 FQN 的 `ComputeLayout` 与有序 `BucketConfig`。这些对象通过 `optimizer_factory_kwargs_by_name["DistMuon"]` 进入 builder（`torchtitan/models/kimi_k2_7/config_registry.py:440`、`torchtitan/models/kimi_k2_7/config_registry.py:456`）。

最短调用链如下：

```text
_dist_muon_optimizer()
  -> OptimizersContainer._build_param_groups()
  -> build_dist_muon()
  -> _initialize_dist_muon()
  -> resolve storage-to-compute transition
  -> build/validate bucket plans + reserve two slots
  -> DistMuon.step()
  -> runtime.run(prepare, compute, finalize)
```

container 以正则 first-match-wins 分配参数，并把 canonical FQN 与参数对齐；同名 optimizer 的组被聚合后调用注册的 `build_dist_muon`（`torchtitan/components/optimizer/optimizer.py:168`、`torchtitan/components/optimizer/optimizer.py:190`、`torchtitan/components/optimizer/optimizer.py:225`）。builder 随后校验 FQN/layout，建立计划、跨 rank 验证摘要并预留运行时 buffer（`torchtitan/distributed/flex_shard/dist_muon.py:170`、`torchtitan/distributed/flex_shard/dist_muon.py:228`、`torchtitan/distributed/flex_shard/dist_muon.py:240`）。

---

## 2. Compute layout 与 view 边界：为什么先重分布，再解释矩阵

### ① 背景/问题

Q/K/V projection 常把多个 head 的矩阵按行拼成一个二维参数 `[M * R, C]`。FSDP `Shard(0)` 的均匀行切分可能落在某个 `R` 行矩阵内部；Muon 若直接把每个 local shard 当成一张矩阵，算出的就不是逐 head 的正交方向。与此同时，真正把持久参数改存成三维 `[M, R, C]` 会把 optimizer 的局部需要泄漏到模型参数格式、FSDP 和 checkpoint。

### ② 为什么这么设计

**选中的路线**是 `storage sharding -> compute sharding -> view`；**明显替代方案**是先把 storage tensor view 成矩阵 batch，再对 view 做 compute sharding。提交 `858b9e24825d97565ec483b1a0972b9445635bdb` 的正文明确比较了两条路线，并选择前者：view 被视为 Muon core 的一部分，`BlockShard(block_size=head_dim)` 先恢复完整 head ownership，之后才做零拷贝三维 view。决定性标准是能显式处理 FSDP2 `Shard(0)` oversharded QKV，而不让 view 成为 FlexShard 之外的参数预处理协议。

该提交还记录了四卡 sanity check：10 步 loss 与 grad norm bitwise identical，CUDA kernel/通信/memcpy/memset 顺序一致，峰值显存和 allocator history 一致。这是历史验证证据，不等价于对所有模型和拓扑的普遍性能承诺。

### ③ 实现思路与细节

`ComputeLayout` 按 storage mesh **轴名**声明 `Owned`、`Replicate`、`Shard` 或 `BlockShard`；未声明轴保留 storage placement，配置对象在构造时冻结并校验类型（`torchtitan/distributed/flex_shard/optimizer_reshard.py:110`、`torchtitan/distributed/flex_shard/optimizer_reshard.py:116`、`torchtitan/distributed/flex_shard/optimizer_reshard.py:168`）。其中：

- `Owned()` 表示 compute 阶段从通信子网动态挑一个 owner 持有完整 subgroup-local 逻辑张量，不改变持久 ownership；多个 `Owned` 轴共同选择笛卡尔积中的一个 rank（`torchtitan/distributed/flex_shard/optimizer_reshard.py:21`、`torchtitan/distributed/flex_shard/optimizer_reshard.py:25`）。
- `BlockShard(dim=0, block_size=R)` 只在完整定长 row block 之间切分，保持原 tensor rank 和 global shape，本身绝不执行 view（`torchtitan/distributed/flex_shard/optimizer_reshard.py:34`、`torchtitan/distributed/flex_shard/optimizer_reshard.py:38`）。
- 原生三维 `[M, R, C]` 矩阵 batch 用 `Shard(0)` 分配完整矩阵；单个二维矩阵则用 `Owned`。builder 的公共契约把三者明确区分（`torchtitan/distributed/flex_shard/dist_muon.py:65`、`torchtitan/distributed/flex_shard/dist_muon.py:71`）。

对于 flat QKV，初始化先从 `BlockShard` 推导 `_MatrixBatchView` 和 global compute shape，但不触碰参数 storage（`torchtitan/distributed/flex_shard/dist_muon.py:121`、`torchtitan/distributed/flex_shard/dist_muon.py:199`）。planner 仍以二维 storage region 建路由，让每个 destination 收到整数个完整矩阵的连续行（`torchtitan/distributed/flex_shard/dist_muon.py:1134`、`torchtitan/distributed/flex_shard/dist_muon.py:1191`）。只有 compute buffer 已到位时，`_compute_update()` 才调用 `unflatten` 得到零拷贝 batch view，再运行 Muon（`torchtitan/distributed/flex_shard/dist_muon.py:694`、`torchtitan/distributed/flex_shard/dist_muon.py:610`）。

Kimi 将 `wq_b`/无 LoRA 的 `wq` 配成 per-query-head `BlockShard`，将 `wkv_b` 配成另一种 block size，而 `wq_a`、`wkv_a`、`wo` 用 `Owned(dp_shard)`；block size 直接来自 attention head dimensions（`torchtitan/models/kimi_k2_7/config_registry.py:295`、`torchtitan/models/kimi_k2_7/config_registry.py:300`、`torchtitan/models/kimi_k2_7/config_registry.py:308`、`torchtitan/models/kimi_k2_7/config_registry.py:317`）。

### ④ 约束/边界

- `BlockShard` 路径只接受 contiguous local storage 的二维 `[M * R, C]`，第一维必须非零且可被 `R` 整除；不满足时在建 optimizer 阶段失败（`torchtitan/distributed/flex_shard/dist_muon.py:144`、`torchtitan/distributed/flex_shard/dist_muon.py:650`）。
- matrix-batch storage 的所有非单例轴只能是 exact `Shard` 或 `Replicate`，且 shard 必须落在 tensor dim 0（`torchtitan/distributed/flex_shard/dist_muon.py:700`、`torchtitan/distributed/flex_shard/dist_muon.py:712`）。
- 当前只允许一个非单例 mesh 轴承载 active `BlockShard`，其他非单例 storage 轴必须 replicated；多 active block axes 明确抛 `NotImplementedError`（`torchtitan/distributed/flex_shard/dist_muon.py:795`、`torchtitan/distributed/flex_shard/dist_muon.py:816`）。
- `ComputeLayout` 类型里虽然允许 `Replicate`，DistMuon 当前 resolver 并未实现显式 replicated compute；这再次说明 FlexShard 公共表达能力与 DistMuon consumer 覆盖面不能画等号（`torchtitan/distributed/flex_shard/optimizer_reshard.py:116`、`torchtitan/distributed/flex_shard/dist_muon.py:1471`）。

### ⑤ 发展趋势（有源码锚点的推断）

提交 `858b9e248` 把 attention-specific view 从配置对象移入 Muon core，当前公共 `BlockShard` 文档也坚持“不 reshape”。据此可推断，FlexShard 的方向是让通用层只描述 ownership 与路由，让 consumer 自己解释 compute tensor 的算法语义；源码没有承诺所有未来 optimizer 都采用 matrix-batch view。

---

## 3. EP/EFSDP shard order 与 region planner：placement 相同不代表所有权相同

### ① 背景/问题

在 sparse storage mesh `[efsdp, ep]` 上，routed expert 参数可持久存为 `Shard(1)` on EFSDP、`Shard(0)` on EP：EP 先分 expert，EFSDP 再切每个 expert 的矩阵内部。Muon compute 希望两个轴都沿 expert batch 的 dim 0 分工，但若按 storage-mesh 默认顺序解释两个 `Shard(0)`，就会先按 EFSDP 切 global experts，再按 EP 切 local domain，破坏原有 EP ownership。

### ② 为什么这么设计

**选中的路线**是保留 EP 作为外层 shard，再在 EP-local expert domain 内用 EFSDP 重分配；**明显替代方案**是接受 storage-mesh 顺序 `[efsdp, ep]` 的默认同维切分。提交 `6ee6b10b934730df00cbcd0cab5c93d04ae2e38a` 的正文把契约写成 `shard_order_by_tensor_dim={0: ("ep", "efsdp")}`，并把 routed experts 与 dense 参数分到不同 bucket/mesh。决定性标准是重分布只改变 EFSDP 方向的 compute ownership，同时保留 exact EP-first storage ownership。

### ③ 实现思路与细节

`shard_order_by_tensor_dim` 以“外层轴在前”声明同一 tensor dim 的嵌套顺序；构造时拒绝重复轴、缺失轴、少于两个轴，以及没有在同一 dim 上声明 `Shard` 的轴（`torchtitan/distributed/flex_shard/optimizer_reshard.py:196`、`torchtitan/distributed/flex_shard/optimizer_reshard.py:217`、`torchtitan/distributed/flex_shard/optimizer_reshard.py:229`）。

DTensor 默认按 storage-mesh 顺序解释同维 shards，因此 resolver 将“应晚于右侧轴应用”的 axis 降成 `_StridedShard`；split factor 由它之前但位于 storage mesh 右侧的轴大小乘积推导，而不是让 Kimi 手写 rank 数（`torchtitan/distributed/flex_shard/dist_muon.py:1277`、`torchtitan/distributed/flex_shard/dist_muon.py:1285`、`torchtitan/distributed/flex_shard/dist_muon.py:1299`）。Kimi 的实际 per-expert layout 正是 EFSDP/EP 都 `Shard(0)`，order 为 `("ep", "efsdp")`（`torchtitan/models/kimi_k2_7/config_registry.py:260`、`torchtitan/models/kimi_k2_7/config_registry.py:274`、`torchtitan/models/kimi_k2_7/config_registry.py:279`）。

planner 不直接拼 rank-to-rank copy 列表，而是建立三层几何对象：participant 的 logical partition、storage/compute endpoint 中的 tensor region、两端 numel 相等的 logical route（`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:113`、`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:133`、`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:157`）。计划构造时验证：

1. storage 与 compute partitions 的 participant 顺序一致；
2. 每个 partition 的 regions 完整覆盖且互不越界；
3. 每条 route 的 source/destination region 元素数相等；
4. 正向 routes 自动取逆得到 compute-to-storage routes。

这些 invariant 在 `_RedistributionPlan.__post_init__()` 中一次性执行，而不是推迟到 collective 已启动后（`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:183`、`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:194`、`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:229`）。对于 EP+EFSDP，storage domain 允许 transport axis 之外恰好一个**正交** exact `Shard`，并把它收缩为 subgroup-local logical shape（`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:988`、`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:997`、`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:1011`）。

`Owned` 的 owner 也不是固定 rank 0。resolver 用 Newton-Schulz 估算量作为 primary load、参数 bytes 作为 secondary load，以稳定 FQN 做 tie-break，运行确定性的 LPT heuristic，并在有序 buckets 之间累计负载（`torchtitan/distributed/flex_shard/dist_muon.py:954`、`torchtitan/distributed/flex_shard/dist_muon.py:992`、`torchtitan/distributed/flex_shard/dist_muon.py:1029`、`torchtitan/distributed/flex_shard/dist_muon.py:1039`）。这是 compute ownership 的动态负载均衡，不是 storage ownership 迁移。

### ④ 约束/边界

- 当前一次参数转换至多使用一个 transport mesh axis；多个变化轴或多个 active `Owned`/reshard 轴会报 multi-axis transport 未实现（`torchtitan/distributed/flex_shard/dist_muon.py:1523`、`torchtitan/distributed/flex_shard/dist_muon.py:1531`）。
- 同维 shard reorder 只支持“一个重分布轴越过恰好一个右侧 preserved shard axis”的形态；更多层级的 reorder 会被 validator 拒绝（`torchtitan/distributed/flex_shard/dist_muon.py:1312`、`torchtitan/distributed/flex_shard/dist_muon.py:1356`）。
- bucket 中所有需要重分布的参数必须解析到同名、同 rank 集合的 transport group；否则要求用户拆分 `BucketConfig.patterns`（`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:66`、`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:79`、`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:85`）。
- GPU 单测固定了错误 order 必须失败、正确 EP-first order 得到预期 local expert slice、更新后参数仍保持原 `(Shard(1), Shard(0))` storage placements（`tests/unit_tests/gpu/flex_shard/test_dist_muon.py:277`、`tests/unit_tests/gpu/flex_shard/test_dist_muon.py:303`、`tests/unit_tests/gpu/flex_shard/test_dist_muon.py:363`、`tests/unit_tests/gpu/flex_shard/test_dist_muon.py:374`）。

### ⑤ 发展趋势（有源码锚点的推断）

Kimi 注释明确暂不在 EP-local experts 少于 EFSDP size 时增加另一层 balanced rank assignment，只有 benchmark 证明固定非空坐标形成热点才考虑（`torchtitan/models/kimi_k2_7/config_registry.py:271`）。因此当前优先级是保持 EP ownership 可解释、计划确定，而非先追求所有极端 expert/rank 比例上的完美负载；这是源码注释支持的局部趋势，不是通用路线图。

---

## 4. Bucket、哈希与 packed 双槽：为什么通信计划在热路径之前冻结

### ① 背景/问题

若每个 parameter region 独立发 collective，启动开销会淹没小矩阵；若每步动态重新排 bucket，又可能让不同 rank 以不同 collective 顺序进入 NCCL/RCCL，产生 hang。另一方面，完全串行地“所有参数先 storage-to-compute、全部算完、再全部返回”会扩大临时内存，也失去按 layer 生产消费的重叠机会。

### ② 为什么这么设计

**选中的路线**是构造期冻结有序 bucket 与精确 region routes，bucket 内打包为一对 variable-split all-to-all schedule，运行时只保留两个 rolling slots；**明显替代方案**是 per-parameter collective 或全模型单一大 buffer。决定性标准是同时约束 collective 次数、峰值临时内存和跨 bucket overlap，而不是单独最小化其中一个指标。Kimi 又把 layer 0 的大 dense MLP 单列，其余 MoE layers 两层一组，以在 collective 启动摊销与流水粒度之间取折中（`torchtitan/models/kimi_k2_7/config_registry.py:388`、`torchtitan/models/kimi_k2_7/config_registry.py:392`）。

### ③ 实现思路与细节

`BucketConfig` 用大小写敏感 fnmatch patterns 选 FQN，每个 optimizer parameter 必须恰好命中一个 bucket；配置顺序就是执行顺序（`torchtitan/distributed/flex_shard/optimizer_reshard.py:251`、`torchtitan/distributed/flex_shard/optimizer_reshard.py:274`、`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:93`）。完全 compute-ready 的 bucket 绑定为 mesh-free local plan；其余 bucket 才解析通信 mesh（`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:33`、`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:57`）。

packed schedule 逐 parameter 解析 routes，把同一 source/destination 的 region span 放入连续 input/output buffers，并计算 `input_split_sizes` 与 `output_split_sizes`；正反两个方向各生成一个 schedule（`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:775`、`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:812`、`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:867`、`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:1181`）。若没有 remote transfer，执行器退化为 local copy；否则调用一次 `all_to_all_single`（`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:103`、`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:109`）。

进入热路径前，每个 redistribution plan 的 partitions、routes、FQN、shape、dtype、layout 与 optimizer group signature 被稳定序列化，取 SHA-256 前 7 bytes 形成 int64 hash，再在对应 process group 上 all-gather；任何 rank 不一致都立即报错（`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:1211`、`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:1227`、`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:1238`）。hash 是 rank-stability guard，不是安全认证。

运行时固定两个 `_PipelineSlot`；每个槽持有 storage/compute exchange buffers、两类 scratch buffer，以及 `compute_input_ready`、`compute_done`、`done` 三个跨 stream event（`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:41`、`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:234`、`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:255`）。`reserve_buffers()` 按分配到每个滚动槽的 bucket 最大需求提前扩容，热路径不再为每个 bucket 新建 buffer（`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:324`、`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:350`、`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:402`）。

稳态调用链是：

```text
transfer stream: pack + S->C A2A(bucket N+1)       C->S A2A(bucket N-1) + finalize
caller stream:                    local work + wait + compute(bucket N)
                                    ^ events connect ownership hand-offs ^
```

storage-to-compute 在 transfer stream 打包并记录 input-ready event；caller stream 先算 bucket 中 local items，再等 event 计算 redistributed items；transfer stream 等 compute-done 后反向 all-to-all 并 finalize（`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:559`、`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:589`、`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:620`）。主循环计算当前 bucket 后预取下一 redistributed bucket并轮换槽（`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:419`、`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:518`、`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:533`）。

### ④ 约束/边界

- local-only bucket 是明确的 prefetch barrier；运行时不会跨过它提前调用后续 redistributed bucket 的 `prepare`（`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:303`、`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:449`）。
- callback 操作的是 runtime-owned scratch，不得保留 tensor、同步 stream 或自行 `record_stream()`；否则会破坏槽复用协议（`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:303`、`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:305`）。
- 运行中异常被定义为 fatal：部分参数/状态可能已更新，collective 可能仍在 flight，不能复用 optimizer（`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:310`）。
- 当前打包前后的多个 region copy 用 `_foreach_copy_` 合成一次 launch；空列表和单 region 特判绕过 foreach，CPU 单测覆盖 mixed sizes、noncontiguous views、dtype cast 和 shape mismatch（`torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py:784`、`tests/unit_tests/cpu/flex_shard/test_optimizer_reshard_runtime.py:20`、`tests/unit_tests/cpu/flex_shard/test_optimizer_reshard_runtime.py:30`、`tests/unit_tests/cpu/flex_shard/test_optimizer_reshard_runtime.py:47`）。

### ⑤ 发展趋势（提交锚点与收益上限）

提交 `10cc9ffbb93b99318dd33c4e618d9ac67441c2c3` 把逐 region `Tensor.copy_()` 合并为 batched copies。其四张 AMD MI250、ROCm、Kimi K2.7 debug、FSDP4+EP2 benchmark 显示：每 rank async copies 从 493 降至 423（-14.2%），non-RCCL launches 从 1474 降至 1421.5（-3.6%），但 DistMuon GPU span 仅从 93.12 ms 到 92.54 ms（-0.6%），CPU span 反而在噪声内从 92.29 ms 到 92.64 ms，RCCL launches 仍为 8。正确结论是“packing copy 启动开销可测地下降，但该工作负载的 optimizer 端到端延迟基本不变”，不能把局部 launch 降幅写成同幅度吞吐收益。

---

## 5. DistMuon consumer：在临时 ownership 上算方向，在持久 ownership 上存状态

### ① 背景/问题

即使 FlexShard 已经把完整矩阵送到 compute rank，优化器仍要回答两个不同问题：momentum state 应长期放在哪里，Newton-Schulz 产生的 direction 又应在哪里计算。如果把 momentum 也随 compute owner 持久迁移，动态 owner 与 bucket 重规划就会变成 checkpoint/state ownership 的一部分；如果在 storage shard 上直接做正交化，又回到矩阵被切碎的问题。

### ② 为什么这么设计

**选中的路线**是 momentum 与 parameter 一直采用 storage DTensor layout，只把已融合 momentum/Nesterov 的 optimizer input 临时送到 compute layout；**明显替代方案**是让 compute owner 长期持有 optimizer state。决定性标准是 checkpoint 和跨 step 状态必须稳定，而矩阵完整性只在一次方向计算期间需要。当前实现用 `zeros_like(grad, memory_format=preserve_format)` 创建 momentum DTensor，并在每个 step 前验证它仍匹配参数 storage layout（`torchtitan/distributed/flex_shard/dist_muon.py:558`、`torchtitan/distributed/flex_shard/dist_muon.py:569`）。

### ③ 实现思路与细节

`DistMuon.step()` 先做 preflight，再把 `prepare`、`compute`、`finalize` 三个 callback 交给通用 bucket runtime（`torchtitan/distributed/flex_shard/dist_muon.py:315`、`torchtitan/distributed/flex_shard/dist_muon.py:322`）。三段职责严格分开：

1. `_prepare_local()` 在 storage layout 上读取 gradient/momentum，以 in-place `lerp_` 更新 momentum；Nesterov 模式再把 gradient 与 momentum lerp 到 scratch（`torchtitan/distributed/flex_shard/dist_muon.py:595`、`torchtitan/distributed/flex_shard/dist_muon.py:1828`）。
2. `_compute_update()` 必要时把 flat tensor zero-copy view 成 matrix batch，再调用 Newton-Schulz 生成 direction（`torchtitan/distributed/flex_shard/dist_muon.py:610`、`torchtitan/distributed/flex_shard/dist_muon.py:614`）。
3. `_apply_update()` 已回到 storage-local shape，先做 decoupled weight decay，再按 matrix shape 调整后的 learning rate 加 direction，最终递增 parameter version（`torchtitan/distributed/flex_shard/dist_muon.py:624`、`torchtitan/distributed/flex_shard/dist_muon.py:1869`）。

Newton-Schulz 内核先复制为 BF16，必要时转置使最后两维短边在前，再用 Frobenius norm 归一化；单矩阵循环使用 `addmm`，batch 使用 `baddbmm`，最后恢复原方向（`torchtitan/distributed/flex_shard/dist_muon.py:1889`、`torchtitan/distributed/flex_shard/dist_muon.py:1898`、`torchtitan/distributed/flex_shard/dist_muon.py:1904`、`torchtitan/distributed/flex_shard/dist_muon.py:1909`）。Kimi 将 `adjust_lr_fn` 设成 `match_rms_adamw`，对应 `0.2 * sqrt(max(rows, columns))` 的 shape scaling（`torchtitan/models/kimi_k2_7/config_registry.py:332`、`torchtitan/distributed/flex_shard/dist_muon.py:1810`）。

### ④ 约束/边界

- DistMuon 当前恰好接受一个 parameter group，并拒绝 fused/foreach 模式、非法超参数和不支持的 LR adjustment 名称（`torchtitan/distributed/flex_shard/dist_muon.py:337`、`torchtitan/distributed/flex_shard/dist_muon.py:343`）。
- 参数必须全是 DTensor，且本进程只使用一个 CUDA device；plain Tensor、CPU 或多 local devices 不走这条实现（`torchtitan/distributed/flex_shard/dist_muon.py:359`）。
- 每个 configured parameter 在 `step()` 前必须有 DTensor gradient，gradient、parameter local storage 与已有 momentum layout 不得改变；preflight 在 bucket communication 前集中检查，避免确定性输入错误只更新早期 bucket（`torchtitan/distributed/flex_shard/dist_muon.py:486`、`torchtitan/distributed/flex_shard/dist_muon.py:517`、`torchtitan/distributed/flex_shard/dist_muon.py:537`）。
- batched BF16 Newton-Schulz 与逐矩阵调用只承诺相同数学更新，不承诺 bitwise equality；GPU 单测以较宽 absolute tolerance 对照 `torch.optim.Muon`，同时要求参数写回与 checkpoint resume 正确（`torchtitan/distributed/flex_shard/dist_muon.py:260`、`tests/unit_tests/gpu/flex_shard/test_dist_muon.py:130`、`tests/unit_tests/gpu/flex_shard/test_dist_muon.py:199`、`tests/unit_tests/gpu/flex_shard/test_dist_muon.py:224`）。

### ⑤ 发展趋势（有源码锚点的推断）

optimizer state restore 后，hook 会重新校验 group、重建 bucket plans、跨 rank 验证并重新预留 buffers，因为恢复的 `ns_steps` 会改变 compute cost 与 buffer planning（`torchtitan/distributed/flex_shard/dist_muon.py:1923`）。据此可推断，当前 API 把 layout 配置视为“重建 optimizer 时必须再次提供的构造契约”，而不是 checkpoint payload；README 也明确说 compute sharding 不写入 state dict（`torchtitan/distributed/flex_shard/README.md:48`）。

---

## 6. Kimi 接线与组合边界：FlexShard 能表达，不代表该 recipe 已支持

### ① 背景/问题

一个通用 redistribution layer 能描述多种 layout，并不自动证明某个 model recipe、optimizer parameter selection 和 parallel composition 已经闭环。Kimi 还要决定哪些矩阵用 Muon、哪些留给 AdamW，如何按 dense/routed experts 拆 transport group，以及 CLI 覆盖 EP 后如何修正构造期 layout。

### ② 为什么这么设计

**选中的路线**是 `OptimizersContainer` 以有序正则把矩阵组交给 DistMuon、其余参数交给 AdamW，并把 layout/bucket 作为 factory-level 元数据；**明显替代方案**是让 DistMuon 接管所有 trainable tensors 或把 layout 塞进每个 param group。决定性标准是 Muon 只适合矩阵参数，而 per-FQN layout 与通信顺序属于整个 optimizer instance，不是可随 group 独立变化的普通超参数。container 的配置注释也明确区分 instance-wide metadata 与 group hyperparameters（`torchtitan/components/optimizer/optimizer.py:127`、`torchtitan/components/optimizer/optimizer.py:132`）。

### ③ 实现思路与细节

Kimi 把 attention projections、dense MLP、routed/shared experts 与 router gate 的矩阵匹配给 DistMuon；embedding、norm、bias、LM head 和 vision tower 落入后续 catch-all AdamW group（`torchtitan/models/kimi_k2_7/config_registry.py:349`、`torchtitan/models/kimi_k2_7/config_registry.py:425`、`torchtitan/models/kimi_k2_7/config_registry.py:440`、`torchtitan/models/kimi_k2_7/config_registry.py:448`）。每层生成 FQN layout，layer 0 dense bucket 独立，后续 MoE layers 两两分组，并把 routed-expert FQNs 再拆成独立 bucket；这些参数绑定 `[efsdp, ep]` storage mesh，但 planner 只取真正发生重分布的 1D axis submesh 作为 transport group（`torchtitan/models/kimi_k2_7/config_registry.py:380`、`torchtitan/models/kimi_k2_7/config_registry.py:398`、`torchtitan/models/kimi_k2_7/config_registry.py:405`、`torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py:66`）。

recipe 构造后仍允许 CLI 覆盖 `expert_parallel_degree`，所以 `_KimiTrainerConfig.__post_init__()` 会根据最终 parallelism 重建 routed-expert compute layouts；源码 TODO 说明等 CLI 不再允许这种覆盖后可删除该补偿层（`torchtitan/models/kimi_k2_7/config_registry.py:465`、`torchtitan/models/kimi_k2_7/config_registry.py:477`、`torchtitan/models/kimi_k2_7/config_registry.py:511`）。

当前组合证据不能只看 guard：integration catalog 实际列出四卡 Kimi DistMuon `PP+FSDP+EP` 和八卡 `FSDP+EP` cases（`tests/integration_tests/models.py:172`、`tests/integration_tests/models.py:181`）。提交 `63dc89cfea34f84d40b61edc16591c43573d5263` 又说明 PP 曾因 non-empty param-group 约束被 gate，现已重新开启；条件是每个 PP stage 至少拥有一层、从而至少匹配一个 Muon 参数。

### ④ 约束/边界

| 组合/输入 | 当前结论 | 失败位置或证据 |
|---|---|---|
| FSDP + EP | 已有 Kimi integration case；routed experts 使用 EP-first shard order | `tests/integration_tests/models.py:181`；`torchtitan/models/kimi_k2_7/config_registry.py:274` |
| PP + FSDP + EP | 可用；每个 stage 必须至少有一个 Muon matrix | `tests/integration_tests/models.py:173`；`torchtitan/models/kimi_k2_7/config_registry.py:528` |
| TP > 1 | Kimi DistMuon recipe 在 config parsing 时显式拒绝 | `torchtitan/models/kimi_k2_7/config_registry.py:519`、`torchtitan/models/kimi_k2_7/config_registry.py:523` |
| `_StridedShard` storage | 只支持 EP/EFSDP 那个受控 reorder 形态；不能据此泛化到 TP-produced layouts | `torchtitan/distributed/flex_shard/dist_muon.py:1627`、`torchtitan/models/kimi_k2_7/config_registry.py:519` |
| PP stage 无 transformer layer | Muon regex 匹配为空，container 报 `matched no parameters` | `torchtitan/components/optimizer/optimizer.py:200`；`torchtitan/models/kimi_k2_7/config_registry.py:528` |
| 参数 FQN/layout/bucket 在构造后改变 | 不支持；parameter groups 与计划冻结，需重建 optimizer | `torchtitan/distributed/flex_shard/dist_muon.py:176`、`torchtitan/distributed/flex_shard/dist_muon.py:332` |

这里最容易产生的错误归纳是：“FlexShard 支持某 placement，所以 Kimi DistMuon 支持任意产生该 placement 的 TP/EP 组合。”实际 resolver 对 `_StridedShard` 的来源、轴顺序和 preserved shard 数量有结构性限制，Kimi 又有更早的 recipe guard；公共表示、consumer 能力、模型接线必须分三层判断。

### ⑤ 发展趋势（有源码锚点的推断）

Kimi 的 TP guard 带 `TODO(#3353)`，所以“支持 TP-produced `_StridedShard`”是有锚点但尚未完成的方向，不能写成现状（`torchtitan/models/kimi_k2_7/config_registry.py:519`）。FlexShard README 另称该目录在 API 成熟前暂留 TorchTitan，并计划迁到独立 package/repository；同时要求依赖只从 TorchTitan 指向 FlexShard、FlexShard 只依赖 PyTorch（`torchtitan/distributed/flex_shard/README.md:68`、`torchtitan/distributed/flex_shard/README.md:73`）。这是明确意图，不保证迁移时间表。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis|FSDP 分析]] —— parameter、gradient 与 optimizer state 的持久 storage ownership；本页只处理 optimizer step 内的临时 compute ownership。
- [[02_engineering/02_train_frameworks/torchtitan/15_torchtitan_ep_analysis|EP 分析]] —— EP/EFSDP mesh、expert 参数 storage 与 token dispatch；本页只解释 DistMuon 的 expert compute re-layout。
- [[02_engineering/02_train_frameworks/torchtitan/14_torchtitan_pp_analysis|PP 分析]] —— ModelPart/stage 所有权及 schedule；本页只记录 stage-local optimizer 的 non-empty 参数前提。
- [[02_engineering/02_train_frameworks/torchtitan/16_torchtitan_spmd_types_analysis|SPMD Types 分析]] —— 前后向 value layout 与 named mesh；不要把它与 FlexShard 的 optimizer compute layout 混用。
- [[02_engineering/02_train_frameworks/torchtitan/24_torchtitan_comm_optimizations_overlap_analysis|通信与 overlap 分析]] —— 从全局视角比较 collective、symmetric memory 与 optimizer 通信流水。
- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— sibling 页边界、阅读顺序与源码基准。
