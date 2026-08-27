# 数据并行 DP —— FSDP2 机制级深度分析

> **代码基准**:torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e` · PyTorch `2.9.1`(FSDP2 内核 `torch/distributed/fsdp/_fully_shard/`)
> **最后更新**:2026-08-27 · **系列**:torchtitan 多维并行源码级分析(见 [[torchtitan/index]])
>
> 本文是机制级分析的**标杆篇**。它回答四个问题:**参数怎么切?切完怎么预取回来?哪些通信能掩盖?异步通信怎么实现?**
>
> **四页分工**(2026-07-31 补):本页是 FSDP2(PyTorch 原生 eager 路径)的标杆机制篇,覆盖切分/预取/掩盖/异步四问的完整轮廓;预取时序、掩盖窗口与显存生命周期的源码级深挖见 [[20_torchtitan_fsdp_prefetch_overlap_memory_analysis]](深挖伴篇);HSDP 反向 reduce-scatter→all-reduce 双流掩盖的展开见 [[21_torchtitan_hsdp_backward_overlap_analysis]](本页 §4.4/§6.2 的展开篇);编译器友好、把集合通信表达进计算图的替代方案见 [[25_torchtitan_simple_fsdp_analysis]]。
>
> torchtitan 当前把所有 decoder 的 FSDP 接线收敛到 `torchtitan/distributed/fsdp.py:168-424`;真正的参数生命周期与多流通信仍在 PyTorch FSDP2。本文行号约定:
> - torchtitan:`torchtitan/...`,以 `torchtitan/` 为根。
> - PyTorch FSDP2(2.9.1):`[pt]` 前缀,根目录 `torch/distributed/fsdp/_fully_shard/`。

---

## 1. 功能范围与定位

**DP(数据并行)** 在 torchtitan 里通过 PyTorch **FSDP2**(`fully_shard`,逐参数分片 API)实现。一套代码,三种模式:

| 模式 | 触发条件 | mesh | 行为 |
|------|---------|------|------|
| **FSDP**(ZeRO-3) | `dp_shard>1`,`dp_replicate=1` | 稠密 storage mesh,`shard=dp_shard(+cp)` | 参数/梯度/优化器状态全部分片 |
| **HSDP** | `dp_replicate>1` 且分片轴乘积 `>1` | 同上,另有 `replicate=dp_replicate` | 组内分片,组间复制 |
| **复制/单卡退化** | 分片轴乘积为 1 | size-1 mesh 或 replicate 轴 | 仍安装 mixed-precision policy,集合通信退化为空操作 |

> FSDP2 vs FSDP1:FSDP1 是 `FlatParameter`——把一组参数拍平成一个大张量再切,用户无法访问单个参数。**FSDP2 是逐参数(per-parameter)分片**:每个 `nn.Parameter` 单独切成一个 `DTensor`,语义清晰、与 TP/EP 的 DTensor 天然组合。这是 torchtitan 能做多维并行的前提。

**当前 torchtitan 入口**是共享的 `apply_fsdp_to_decoder`:模型适配层先解析 storage mesh,再把统一 FSDP 配置交给共享实现(`torchtitan/models/llama3/parallelize.py:57-78`,`torchtitan/distributed/fsdp.py:168-236`)。核心仍是三件事:

```python
# torchtitan/distributed/fsdp.py:267-374（简化）
for layer_id, transformer_block in model.layers.items():
    fully_shard(transformer_block, **fsdp_config, reshard_after_forward=...)  # 每个 block 一个 FSDP 单元
fully_shard(model, **fsdp_config)                                            # 最后包根模块
disable_fsdp_gradient_division(model)                                        # 关掉 FSDP 自带梯度除法
```

`spmd_types` 默认后端不再把 FSDP 只交给一个预先 flatten 的 1D `fsdp` mesh。`resolve_fsdp_mesh()` 保留 `[dp_replicate,dp_shard,cp,tp]` storage mesh,再显式传 `DataParallelMeshDims(shard=(dp_shard,cp),replicate=dp_replicate)` 给 `fully_shard`;这样参数的存储 mesh 与前向/反向的类型化布局可来自同一份 SPMD 契约(`torchtitan/distributed/fsdp.py:28-62`)。这也是本页 PyTorch 机制层之上的最大接线变化,详见 [[16_torchtitan_spmd_types_analysis]]。

---

## 2. 参数切分:从完整 weight 到 sharded DTensor

### 2.1 分组(FSDPParamGroup):通信的基本单位

`fully_shard(module)` 每调用一次,就为 `module` 构造**一个 `FSDPParamGroup`**(`[pt]_fully_shard.py:225`),它收纳 `module.parameters()` 里**尚未被子模块的 `fully_shard` 认领**的参数。

> **一个 FSDPParamGroup = 一次 all-gather 集合通信 + 一次 reduce-scatter 集合通信。** 把多个张量打包进一次集合通信对通信效率至关重要(`[pt]_fully_shard.py:120-127` docstring)。

因此 `fully_shard` 必须**自底向上(bottom-up)**调用:先包各叶子 FSDP 单元,最后包根 `model`。根模块的 group 只剩下尚未被子模块认领的参数。torchtitan 当前依次处理 embedding、`[norm,lm_head]`、各 TransformerBlock,最后处理根模块(`torchtitan/distributed/fsdp.py:238-368`)。

> **为什么逐层分组而不是整模型一组?** 一个 group 的参数必须**同时驻留**在显存里(all-gather 后)。若整个模型一组,前向时要把全部参数 all-gather 回来——失去 FSDP 的意义。逐层分组 → 峰值显存只需"1 层完整参数 + 其余分片",且为通信/计算重叠创造了机会(见 §4)。

### 2.2 单个参数怎么切:`FSDPParam._init_sharded_param`

每个参数由一个 `FSDPParam` 对象(`[pt]_fsdp_param.py:192`)管理。切分逻辑在 `_init_sharded_param`(`[pt]_fsdp_param.py:257`)。

**第一步:确定分片维度**

```python
# [pt]_fsdp_param.py:272-279
fsdp_placement = shard_placement_fn(param) if shard_placement_fn else None
if fsdp_placement is None:
    fsdp_placement = Shard(0)        # 默认沿 dim-0 切
shard_dim = fsdp_placement.dim
```

默认 `Shard(0)`——沿参数的第 0 维切。`shard_placement_fn` 可改成 `Shard(1)` 等(torchtitan 的 MoE 用到,见 §8.4)。

**第二步:`torch.chunk` 切块 + 预填充(pre-pad)**

```python
# [pt]_fsdp_param.py:362-380
chunks = _chunk_with_empty(param_data, shard_world_size, dim=shard_dim)
sharded_param = chunks[shard_rank]                       # 本 rank 的那一块
padded_sharded_size = chunks[0].size()                   # chunk 0 永远是最大(可能含 padding)
padded_sharded_param = param_data.new_zeros(padded_sharded_size)
padded_sharded_param.narrow(shard_dim, 0, sharded_param.size(shard_dim)).copy_(sharded_param)
...
self._sharded_param_data = padded_sharded_param.view(-1)  # 1D 扁平!
```

要点:
- `torch.chunk` 把完整参数沿 `shard_dim` 切成 `shard_world_size` 块,本 rank 取第 `shard_rank` 块。
- **预填充**:若参数第 0 维不能被 world_size 整除,各块大小不一。FSDP **提前**把本 rank 的分片 pad 到统一大小(`chunks[0]` 的大小),`new_zeros` 补零。这样 all-gather 时**不需要再 padding**——all-gather 要求各 rank 等大。
- `self._sharded_param_data` 是 **1D 扁平张量**——这就是后续 all-gather 的输入(input)。这块就是该参数在本 rank 上**真正占用的显存**。

**第三步:包装成 DTensor**

```python
# [pt]_fsdp_param.py:386
self.sharded_param = nn.Parameter(self.to_sharded_dtensor(sharded_param))
```

注册到模块上的 `sharded_param` 是一个 **`DTensor`**(ND 形状,带 sharding spec)。它对外报告全局逻辑形状,对内 `_local_tensor` 只是本 rank 的分片。优化器 step 直接作用在这个分片 DTensor 上——所以**优化器状态也天然只占 1/N**。

### 2.3 三种参数情况

`_init_sharded_param` 对参数的来源分情况(`[pt]_fsdp_param.py:284-342`):

| 情况 | 条件 | sharding spec |
|------|------|---------------|
| **纯 FSDP** | 普通 `Tensor` 参数,1D dp mesh | `(Shard(0),)` |
| **HSDP** | 普通参数,2D dp mesh | `(Replicate(), Shard(0))` |
| **FSDP+TP/EP** | 参数已是 `DTensor`(被 TP/EP 先切过) | `(_StridedShard 或 Shard, *tp_placements)` |

第三种最关键:当 TP 已把参数切成 DTensor(见 [[12_torchtitan_tp_analysis]]),FSDP 不能再"重切",而是构造一个**组合 mesh**(`_spmd_mesh`,DP 轴 + TP 轴)和**组合 placement**。当 FSDP 分片维与 TP 分片维相同时,用 `_StridedShard`(带 `split_factor`)表达"先按 TP 切、再按 FSDP 切"的嵌套顺序(`[pt]_fsdp_param.py:312-319`)。这就是 FSDP 与 TP/EP 在同一参数上叠加的底层接口——**FSDP 要求 DP 与 TP/EP 的 mesh 有共同父 mesh**(`[pt]_fsdp_param.py:290-296`),这正是 [[10_torchtitan_parallel_dims_analysis|torchtitan 并行基座]] 里三张 mesh 出自同一 world mesh 的原因。

### 2.4 三种分片状态:`ShardedState`

`[pt]_fsdp_param.py:146` 定义参数的三态:

```
SHARDED               ── 只有分片参数注册在模块上(常态,省显存)
SHARDED_POST_FORWARD  ── reshard 到一个更小的 world size(reshard_after_forward=int 时)
UNSHARDED             ── 完整参数注册在模块上(前向/反向计算时)
```

FSDP 运行时就是在 `SHARDED ↔ UNSHARDED` 之间来回切换:用之前 all-gather 成 `UNSHARDED`,用完 reshard 回 `SHARDED`。

---

## 3. 参数怎么预取回来:all-gather 全流程

### 3.1 钩子链:谁触发了 all-gather

`fully_shard` 给模块注册了 forward 前后钩子(`[pt]_fsdp_state.py:103-109`):

```
nn.Module.__call__
   │
   ├─ forward_pre_hook  → FSDPState._pre_forward    [pt]_fsdp_state.py:228
   │                         └→ FSDPParamGroup.pre_forward   [pt]_fsdp_param_group.py:431
   │                               ├→ unshard()        ← 发起 all-gather
   │                               └→ wait_for_unshard() ← 等 all-gather + copy-out
   │
   ├─ module.forward(...)   ← 此时参数已是完整的 UNSHARDED
   │
   └─ forward_hook      → FSDPState._post_forward   [pt]_fsdp_state.py:254
                             └→ FSDPParamGroup.post_forward
                                   └→ reshard()       ← 释放完整参数,回到 SHARDED
```

### 3.2 `unshard()`:发起 all-gather

`FSDPParamGroup.unshard`(`[pt]_fsdp_param_group.py:299`)调 `foreach_all_gather`(`[pt]_fsdp_collectives.py:236`)。两个阶段,**跑在两条不同的 stream 上**:

```python
# [pt]_fsdp_collectives.py:247-281(简化)
with device_handle.stream(all_gather_copy_in_stream):
    # 阶段 A:copy-in —— 把各参数的 1D 分片(可能带 dtype 转换)拷进一块连续缓冲
    all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(...)

all_gather_stream.wait_stream(all_gather_copy_in_stream)   # all-gather 等 copy-in 完成

with device_handle.stream(all_gather_stream):
    # 阶段 B:真正的集合通信
    all_gather_work = all_gather_comm(output_tensor=all_gather_output,
                                      input_tensor=all_gather_input, group=group, ...)
    all_gather_event = all_gather_stream.record_event()
    return AllGatherResult(all_gather_output, all_gather_event, all_gather_work, ...)
```

- **阶段 A(copy-in)**:`all_gather_copy_in`(`[pt]_fsdp_collectives.py:175`)把 group 内所有参数的分片(`_sharded_param_data`)用 `torch._foreach_copy_` 一次性拷进一块连续的 `all_gather_input` 缓冲(它是 `all_gather_output` 里属于本 rank 的那一段的 view)。混合精度时在这里做 bf16 转换。
- **阶段 B(all-gather)**:`dist.all_gather_into_tensor` 把各 rank 的 `all_gather_input` 收集进 `all_gather_output`(大小 = `input_numel × world_size`)。
- 返回 `AllGatherResult`,带一个 **CUDA event**(`all_gather_event`)。**注意此时 all-gather 只是"已入队",未必完成**。

### 3.3 `wait_for_unshard()`:等待 + copy-out + 注册完整参数

`wait_for_unshard`(`[pt]_fsdp_param_group.py:340`):

```python
foreach_all_gather_copy_out(self._all_gather_result, self.fsdp_params, group)  # copy-out
for fsdp_param in self.fsdp_params:
    fsdp_param.init_unsharded_param()      # 构造完整参数
self._to_unsharded()                       # 把完整参数注册到模块上
```

- `foreach_all_gather_copy_out`(`[pt]_fsdp_collectives.py:344`):先 `current_stream().wait_event(all_gather_event)` **等 all-gather 完成**,再用 `split_with_sizes_copy` 把扁平的 `all_gather_output` 拆回每个参数的完整张量。
- `init_unsharded_param`(`[pt]_fsdp_param.py:450`):用 `torch.as_strided` 把 all-gather 输出重新解释成参数的原始 ND 形状。
- `_to_unsharded`(`[pt]_fsdp_param_group.py:644`)→ `FSDPParam.to_unsharded`:把完整参数 `setattr` 到模块上,替换掉分片参数。

### 3.4 `reshard()`:用完即释放

前向结束,`post_forward` → `reshard()`(`[pt]_fsdp_param_group.py:419`)→ `FSDPParam.to_sharded`:把分片参数重新注册回模块,并 `free_unsharded_param()` —— 用 `untyped_storage().resize_(0)` 把完整参数的显存**真正释放**(`[pt]_fsdp_param.py:872`)。

> **关键技巧**:FSDP 用**存储缩放(storage resizing)**而非重新分配来释放/分配完整参数(`[pt]_fsdp_param.py:58-66` 的 Note)。因为 autograd 可能存了完整参数的引用/view,直接释放对象会破坏 aliasing。`resize_(0)` 保留张量对象、只释放底层存储,再次需要时 `resize_(full)`。

### 3.5 预取(prefetch):overlap 的关键

如果只是"用时 all-gather、用完 reshard",每层的 all-gather 会**阻塞**该层计算——零重叠。**预取**就是把第 N+1 层的 all-gather **提前**到第 N 层计算时发起。

**隐式预取(implicit)** —— FSDP2 的默认行为,机制是「独立 stream + CPU 跑在 GPU 前面」:
1. all-gather 在 `all_gather_stream` 上执行,与默认计算流独立。
2. CPU 线程发射完第 N 层的计算 kernel(入队默认流)后,**不等 GPU 执行**,立刻继续 Python 代码,进入第 N+1 层的 `pre_forward` → 发起第 N+1 层的 all-gather(入队 `all_gather_stream`)。
3. GPU 上,默认流跑第 N 层计算、`all_gather_stream` 跑第 N+1 层 all-gather —— **两条流并行**。

**显式预取(explicit)** —— `set_modules_to_forward_prefetch`(`[pt]_fully_shard.py:418`):在第 N 层的 `pre_forward` 里**主动**发起指定模块(如第 N+1、N+2 层)的 all-gather(`[pt]_fsdp_state.py:249-251`)。比隐式更早、更激进,代价是更多显存(更多完整参数同时驻留)。torchtitan 的 MoE 模型用它(见 §8.5)。

**反向预取** —— `_backward_prefetch`(`[pt]_fsdp_param_group.py:599`):反向时按**前向记录的逆序**(`post_forward_order`)预取上一个 group。`pre_backward` 里 `default_prefetch = (没有显式设置 backward 预取)` 时启用(`[pt]_fsdp_state.py:286`)。

```
前向记录顺序: emb → blk0 → blk1 → ... → blkN → norm+head
反向预取顺序: norm+head → blkN → ... → blk1 → blk0   (reverse post-forward order)
```

---

## 4. 哪些通信能掩盖,如何掩盖

FSDP 有四类可重叠的通信,全部靠**多 CUDA stream + 预取**实现:

### 4.1 前向:all-gather(N+1)↔ 计算(N)

如 §3.5,第 N+1 层 all-gather 跑在 `all_gather_stream`,与第 N 层计算(默认流)重叠。这是 FSDP 最主要的重叠。

```
默认流(compute):   [== fwd blk0 ==][== fwd blk1 ==][== fwd blk2 ==]
all_gather_stream:  [AG blk1][AG blk2][AG blk3]
                           ↑ blk1 的 all-gather 与 blk0 的计算重叠
```

### 4.2 前向:copy-in(N+1)↔ all-gather(N)

copy-in 单独占一条 `all_gather_copy_in_stream`(`[pt]_fsdp_param_group.py:67`)。下一个 group 的 copy-in 能与当前 group 的 all-gather 重叠。这就是文件头 `[Note: Overlapping all-gather copy-in and all-gather]`(`[pt]_fsdp_param_group.py:43`)讲的事:为此必须让下一次 copy-in 写进**不同的显存**,所以 FSDP 保存当前 all-gather 输出的引用(`AllGatherState`),由下一个 group 在 copy-in 后释放。

### 4.3 反向:reduce-scatter ↔ 反向计算

梯度的 reduce-scatter 跑在 `reduce_scatter_stream`(`[pt]_fsdp_param_group.py:75`)。第 N 层算完梯度发起 reduce-scatter 后,默认流可以继续算第 N-1 层的反向,二者重叠。

### 4.4 HSDP:all-reduce ↔ reduce-scatter

HSDP 下梯度先 reduce-scatter(组内,分片轴)、再 all-reduce(组间,复制轴)。all-reduce 单独占 `all_reduce_stream`(`[pt]_fsdp_param_group.py:79`)。注释说明(`[pt]_fsdp_param_group.py:76-78`):

> all-reduce 与 all-gather/reduce-scatter 用**不同的网络资源**(典型场景:分片在节点内 NVLink、复制跨节点),所以能并发。

第 N 层的 all-reduce 能与第 N-1 层的 reduce-scatter 重叠。

### 4.5 `reshard_after_forward` 策略

`reshard_after_forward` 控制前向后要不要释放完整参数(`torchtitan/distributed/fsdp.py:112-136`):

| 取值 | 行为 |
|------|------|
| `True` | 前向后释放,反向重新 all-gather(省显存,多一次通信) |
| `False` | 前向后保留,反向直接用(省通信,费显存) |
| `"default"` | **`not pp_enabled`** —— 不开 PP 时 reshard,开 PP 时不 reshard |

为什么 PP 启用时默认不 reshard?PP 下每个 microbatch 都要前向,若每次都 reshard 就要反复 all-gather,开销大且难重叠(`torchtitan/distributed/fsdp.py:124-132`)。torchtitan 还有个细节:最后的 `[norm,lm_head]` 默认不 reshard,因为 FSDP 反向会立刻预取它们(`torchtitan/distributed/fsdp.py:258-265`)。

---

## 5. 异步通信怎么实现的 —— 五条 stream 的编排

这是 FSDP 性能的核心。FSDP 不依赖 `async_op=True`(那只是可选项),而是**用多条 CUDA stream + CUDA event 手工编排**。

### 5.1 五条 stream

`FSDPCommContext.lazy_init`(`[pt]_fsdp_param_group.py:58-79`)创建:

| stream | 用途 | 优先级 |
|--------|------|--------|
| (默认流) | 前向/反向**计算** | 普通 |
| `all_gather_copy_in_stream` | all-gather 前的 copy-in(dtype 转换 + 拷入连续缓冲) | 高(-1) |
| `all_gather_stream` | all-gather 集合通信本身 | 高(-1) |
| `reduce_scatter_stream` | reduce-scatter + 梯度除法 | 高(-1) |
| `all_reduce_stream` | HSDP 的组间 all-reduce | 普通 |

通信流设高优先级,避免其 copy 被计算挤掉而阻塞(`[pt]_fsdp_param_group.py:62-64`)。

### 5.2 stream 同步的两种原语

CUDA stream 之间靠两种操作建立顺序依赖(GPU 上是异步的,不阻塞 CPU):

- **`record_event()`**:在某条 stream 的当前位置插一个 event(标记"执行到这里")。
- **`wait_event(event)` / `wait_stream(other)`**:让一条 stream 等待某 event / 另一条 stream —— 后续 kernel 必须等 event 触发才执行。`wait_stream` ≈ 在 other 上 record 再 wait。

FSDP 的 stream 编排全是这两者的组合。例如 `foreach_all_gather` 里:

```python
all_gather_stream.wait_stream(all_gather_copy_in_stream)   # all-gather 等 copy-in
# ... all-gather ...
all_gather_event = all_gather_stream.record_event()        # 标记 all-gather 完成
# wait_for_unshard 里:
current_stream().wait_event(all_gather_event)              # 计算流等 all-gather
```

### 5.3 跨流的张量生命周期:`AllGatherState` / `ReduceScatterState`

一个张量在 A 流产生、在 B 流使用时,**不能让 A 流提前覆盖它的显存**。FSDP 用 `NamedTuple` + event 保活:

- `AllGatherState`(`[pt]_fsdp_param_group.py:102`)= `(all_gather_result, copy_out_event)`。前向时把当前 all-gather 输出存进 `comm_ctx.all_gather_state`,**推迟释放**,直到**下一个** group 的 copy-out 完成后才放手(`[pt]_fsdp_param_group.py:352-355`)——这样当前 copy-out 能与下一次 all-gather 重叠(§4.2)。
- `ReduceScatterState`(`[pt]_fsdp_param_group.py:107`)= `(reduce_scatter_input, event)`。下一个 group 做 reduce-scatter 前 `wait_event` 上一个的 event(`[pt]_fsdp_param_group.py:516-522`),保证 reduce-scatter 输入缓冲不被提前复用。

### 5.4 与优化器的同步:`post_optim_event`

`_root_pre_forward`(`[pt]_fsdp_state.py:120`)在每次迭代开头,让 all-gather 流**等待上一步优化器完成**:默认 `all_gather_stream.wait_stream(current_stream)`;若用户调了 `set_post_optim_event`,则 `wait_event` 那个 event(`[pt]_fsdp_state.py:131-138`)。保证不会用还没更新完的参数去 all-gather。

### 5.5 `async_op` 路径(可选)

`unshard_async_op`(`[pt]_fsdp_param_group.py:195`)默认 `False`。设 `True` 时 all-gather 用 `async_op=True` 返回 `dist.Work`,在默认流分配缓冲(避免跨流显存碎片),但 copy-in 等就无法与计算重叠了(`[pt]_fully_shard.py:599-610`)。torchtitan 默认不开这条路径,走 §5.1 的多 stream 方案。

### 5.6 异步全景图(前向 3 层)

```
CPU 线程:  发blk0计算 │发blk1预取│发blk1计算│发blk2预取│发blk2计算│...
                      ╲          ╲         ╲
默认流:        [AG0 copy-out][== compute blk0 ==][== compute blk1 ==][== compute blk2 ==]
                              ╱(wait AG0 event)   ╱(wait AG1)         ╱(wait AG2)
all_gather流:  [== AG blk0 ==][== AG blk1 ==][== AG blk2 ==][== AG blk3 ==]
copy_in流:     [copy-in0][copy-in1][copy-in2][copy-in3]
                          └ copy-in1 与 AG0 重叠 ┘

要点:AG(blk N+1) 与 compute(blk N) 在不同流上并行;
      计算流靠 wait_event(all_gather_event) 在真正用参数前才同步。
```

---

## 6. 梯度规约:reduce-scatter(+ HSDP all-reduce)

### 6.1 触发:`post_backward`

某个 group 的所有完整参数的梯度算完后,`post_backward`(`[pt]_fsdp_param_group.py:478`)被触发(触发机制见 §7):

```python
# [pt]_fsdp_param_group.py:496-560(简化)
for fsdp_param in self.fsdp_params:           # 收集 autograd 算出的完整梯度
    unsharded_grads.append(fsdp_param.unsharded_grad_data)
if self.reshard_after_backward:
    self.reshard()                            # 先 reshard 释放完整参数(省显存)
foreach_reduce(fsdp_params_with_grad, unsharded_grads, reduce_scatter_group,
               reduce_scatter_stream, ..., all_reduce_group if hsdp else None, ...)
```

注意顺序:**先 reshard 释放完整参数,再 reduce-scatter**——这样规约梯度时显存里没有完整参数(`[pt]_fsdp_param_group.py:494-495` 注释)。

### 6.2 `foreach_reduce`:reduce-scatter 全流程

`foreach_reduce`(`[pt]_fsdp_collectives.py:446`):

```
1. copy-in:把各参数的完整梯度 chunk_cat 进一块连续的 reduce_scatter_input
2. reduce_scatter_stream.wait_stream(current_stream)   ← RS 等反向计算产出梯度
3. 在 reduce_scatter_stream 上:
   - predivide(可选的预除)
   - dist.reduce_scatter_tensor(reduce_output, reduce_scatter_input, op=...)
   - record event
4. [HSDP] all_reduce_stream.wait_stream(reduce_scatter_stream)
         在 all_reduce_stream 上:dist.all_reduce(reduce_output, ...)
5. postdivide + 转回原 dtype
6. view-out:把规约结果切片写回各参数的 sharded_param.grad(一个分片 DTensor)
```

reduce-scatter 的语义:**规约 + 散射合一**——N 个 rank 各有完整梯度,reduce-scatter 后每个 rank 拿到"求和后梯度"的 1/N 分片。正好匹配分片参数。

### 6.3 梯度除法因子

`_get_gradient_divide_factors`(`[pt]_fsdp_collectives.py:672`):梯度规约要除以数据并行规模 N。FSDP 的优化:
- fp32/bf16(无溢出风险):直接用 NCCL 内置的 `ReduceOp.AVG`,**省一个独立的除法 kernel**。
- fp16(有溢出风险):拆成预除 + 后除,各除 ~√N,避免规约中间值溢出。

> **torchtitan 关掉了它**:`disable_fsdp_gradient_division`(`torchtitan/distributed/fsdp.py:85-99`)对所有 FSDP 模块设 `set_gradient_divide_factor(1.0)`。因为 Trainer 自己按全局有效 token 数缩放梯度(`torchtitan/trainer.py:821-839`),不让 FSDP 再按 world size 除。

### 6.4 非 dim-0 分片的处理

若参数沿非 0 维分片(`Shard(1)` 等),`foreach_reduce` 先把梯度 chunk 后重排成 dim-0 连续(`[pt]_fsdp_collectives.py:495-502`),因为 reduce-scatter 在 dim-0 上做。

---

## 7. 反向传播与钩子机制

FSDP 的反向不靠 `loss.backward()` 之后的显式调用,而是**挂在 autograd 图里**,由 autograd 引擎在恰当时机回调。

### 7.1 三个钩子

| 钩子 | 注册点 | 触发时机 | 作用 |
|------|--------|---------|------|
| **pre-backward** | `register_hook` 挂在前向**输出**张量上(`[pt]_fsdp_state.py:336`) | 该输出的梯度即将计算前 | `unshard()` 把参数 all-gather 回来 + 反向预取 |
| **post-backward** | `RegisterPostBackwardFunction` autograd Function 挂在前向**输入**上(`[pt]_fsdp_param_group.py:672`) | 该 group 所有梯度算完后 | reduce-scatter 规约梯度 + reshard |
| **root final callback** | `Variable._execution_engine.queue_callback`(`[pt]_fsdp_state.py:345`) | 整个 backward 结束 | 收尾、等待所有 reduce-scatter |

### 7.2 pre-backward:为什么是挂在"输出"上

前向输出张量参与反向时,它的梯度最先被算。所以在它上面挂 `register_hook`,就能在"该层反向计算开始前"被回调 → 此时 `unshard()` 把参数 all-gather 回来(反向需要完整参数)。`pre_backward`(`[pt]_fsdp_param_group.py:459`)还会触发 `_backward_prefetch` 预取上一层。

### 7.3 post-backward:为什么是 autograd Function

`RegisterPostBackwardFunction`(`[pt]_fsdp_param_group.py:830`)是个 `torch.autograd.Function`,在 `pre_forward` 时把它 `apply` 到前向**输入**上(`[pt]_fsdp_param_group.py:692`)。它的 `forward` 是恒等,`backward` 调 `post_backward()`。

> 原理:autograd 反向是拓扑序的。挂在"输入"上的 Function,其 `backward` 会在"该 group 所有参数的梯度都算完(因为输入梯度依赖所有参数)"之后才触发——正是发起 reduce-scatter 的最佳时机。

### 7.4 root final callback

`queue_callback` 注册的回调在整个 backward 引擎跑完后执行一次(`_root_post_backward_final_callback`,`[pt]_fsdp_state.py:293`):对没跑过 `post_backward` 的 group 补跑、等待所有 reduce-scatter event、清理 backward 预取数据结构。

### 7.5 反向全景

```
loss.backward()
   │
   ├─ [norm+head 输出的梯度hook] → pre_backward → unshard(norm+head) + 预取(blkN)
   │     └─ 算 norm+head 梯度
   │     └─ [RegisterPostBackwardFunction.backward] → post_backward → reshard + reduce-scatter
   │
   ├─ [blkN] pre_backward → unshard(blkN,多半已预取好) + 预取(blkN-1)
   │     └─ 算 blkN 梯度 → post_backward → reshard + reduce-scatter
   │  ... 逐层 ...
   │
   └─ root_post_backward_final_callback:收尾,等所有 reduce-scatter 完成
```

---

## 8. torchtitan 的接入策略

### 8.1 分片粒度:每个 TransformerBlock 一个单元

`apply_fsdp_to_decoder` 对 embedding、`[norm,lm_head]`、每个 block、根模块分别调用 `fully_shard`(`torchtitan/distributed/fsdp.py:238-267,267-368`)。粒度选择是显存与通信的权衡:太粗(整模型一组)峰值显存爆;太细(每个 Linear 一组)集合通信次数多、每次太小。**一个 TransformerBlock** 是主要计算体的折中粒度。

### 8.2 混合精度

`MixedPrecisionPolicy(param_dtype, reduce_dtype,cast_forward_inputs=False)` 由共享入口构造(`torchtitan/distributed/fsdp.py:223-232`):`param_dtype` 决定 all-gather 后用于计算的参数 dtype(默认 bf16),`reduce_dtype` 决定梯度规约 dtype(当前配置只允许 fp32;`torchtitan/config/configs.py:96-108`)。分片参数始终以训练 dtype 保存——优化器直接更新它,**FSDP 不需要再维护一份额外 master 参数**(`[pt]_fsdp_api.py:22-26`)。

### 8.3 权重绑定

`enable_weight_tying` 时,`tok_embeddings`/`norm`/`lm_head` 被打包进**同一个 FSDP 单元**,避免共享参数重复 all-gather(`torchtitan/distributed/fsdp.py:238-250`)。

### 8.4 MoE:`shard_placement_fn` 按参数分流

共享入口把 dense 与 MoE 统一为一条路径。MoE block 先识别 grouped-GEMM 子模块的专家参数;EP>1 时把**专家参数**路由到 sparse `edp_mesh`,非专家参数留在 dense `dp_mesh`,并依据 `efsdp×ep` 与专家数关系选择 `Shard(0)` 或 `Shard(1)`以减少 padding(`torchtitan/distributed/fsdp.py:267-360`)。`resolve_sparse_fsdp_mesh()` 则把 sparse storage mesh 的 FSDP 轴显式声明为 `efsdp`(`torchtitan/distributed/fsdp.py:65-82`)。详见 [[15_torchtitan_ep_analysis]]。

### 8.5 MoE:显式预取

EP 开启时,共享入口末尾用 `set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch` 显式串联 embedding、各 block 与输出层(`torchtitan/distributed/fsdp.py:384-424`)。原因仍是 EP token dispatch 中的 D2H 同步可能阻塞 CPU 线程,打断依赖 CPU 跑在 GPU 前面的隐式预取;显式依赖把下一层 all-gather 更早发起。详见 [[15_torchtitan_ep_analysis]]。

### 8.6 当前新增开关:对称内存通信

`parallelism.enable_fsdp_symm_mem` 会遍历全部 `FSDPModule`,强制 sum reduction 并调用 `set_symm_mem_for_comm()`(`torchtitan/distributed/fsdp.py:102-109,368-374`)。配置层在 NVIDIA 上要求 compute capability ≥9.0(`torchtitan/config/configs.py:272-281`)。它改变的是 FSDP 集合通信实现,不改变本页描述的分片状态机与预取依赖;通信路径细节见 [[24_torchtitan_comm_optimizations_overlap_analysis]]。

---

## 9. 完整时间线

```
═══ 一次训练迭代的 FSDP 时间线 ═══

[迭代开始] root_pre_forward: all_gather 流 wait 上一步 optimizer

前向(reshard_after_forward=True):
  blk0: pre_forward → unshard(AG blk0) → wait → compute → post_forward → reshard(释放)
  blk1: (AG 已在 blk0 计算时预取) → wait → compute → reshard
  ...                                              ↑ AG(blkN) ‖ compute(blkN-1) 重叠

反向:
  blkN: pre_backward(梯度hook) → unshard(AG blkN,反向重新取) → compute 梯度
        → post_backward → reshard → reduce-scatter(RS 流)
  blkN-1: (AG 已预取) → compute 梯度 → reduce-scatter
  ...                                  ↑ RS(blkN) ‖ compute(blkN-1) 重叠
  [HSDP] all-reduce(AR 流)‖ reduce-scatter 重叠

[backward 结束] root_final_callback:等所有 RS/AR 完成
optimizer.step():在分片参数(DTensor)上更新,优化器状态只占 1/N
```

---

## 10. 小结

- **参数切分**:FSDP2 逐参数切。`FSDPParam._init_sharded_param` 用 `torch.chunk` 沿 dim-0 切 + **预填充**对齐,本 rank 的分片存成 1D 扁平张量 `_sharded_param_data`(即 all-gather 输入),注册到模块上的是 ND 的分片 `DTensor`。一次 `fully_shard` = 一个 `FSDPParamGroup` = 一次 all-gather + 一次 reduce-scatter,必须自底向上分层调用。
- **怎么预取回来**:forward 钩子链 `pre_forward → unshard → wait_for_unshard`。`foreach_all_gather` 两阶段(copy-in / all-gather)→ `foreach_all_gather_copy_out` 拆分 → 注册完整参数。预取 = 把下一层 all-gather 提前发起:隐式靠"独立 stream + CPU 跑在 GPU 前",显式靠 `set_modules_to_forward_prefetch`,反向靠 reverse post-forward 顺序。
- **哪些通信能掩盖**:all-gather(N+1)↔ 计算(N);copy-in(N+1)↔ all-gather(N);reduce-scatter ↔ 反向计算;HSDP 的 all-reduce ↔ reduce-scatter。
- **异步怎么实现**:**五条 CUDA stream**(计算 / copy-in / all-gather / reduce-scatter / all-reduce)+ `record_event` / `wait_event` / `wait_stream` 编排顺序依赖;`AllGatherState` / `ReduceScatterState` 用 event 跨流保活张量。不靠 `async_op`,靠多流。
- **梯度规约**:`post_backward` 触发 `foreach_reduce` —— 先 reshard 再 reduce-scatter;HSDP 追加组间 all-reduce(独立流,与 RS 重叠)。torchtitan 关掉 FSDP 自带梯度除法,自己按 token 数缩放。
- **反向钩子**:pre-backward 挂在前向输出上(unshard + 预取);post-backward 是挂在前向输入上的 autograd Function(reduce-scatter);root callback 收尾。
- **2026-08 接线变化**:dense/MoE 共用 `apply_fsdp_to_decoder`;默认 `spmd_types` 通过 storage mesh + `DataParallelMeshDims` 显式声明 DP/CP 分片轴;新增 FSDP symmetric-memory 开关。PyTorch FSDP2 的逐参数状态机与多流机制未因此改变。

## Related Pages

- [[20_torchtitan_fsdp_prefetch_overlap_memory_analysis]] —— **深挖伴篇**(配图):预取/掩盖时序、copy-in 三步与唯一跨流同步点、flat 双缓冲、完整参数 ≤2 份证明
- [[torchtitan/index]] · [[10_torchtitan_parallel_dims_analysis]] · [[16_torchtitan_spmd_types_analysis]] —— 知识地图、双平面 mesh 与 SPMD 类型契约
- [[12_torchtitan_tp_analysis]] —— FSDP 与 TP 在同一参数上叠加(DTensor 嵌套、`_StridedShard`)
- [[15_torchtitan_ep_analysis]] —— MoE 专家的 `edp_mesh` FSDP 与显式预取
- [[16_megatron_distributed_optimizer_analysis]] —— Megatron-LM 数据并行 + 分布式优化器(ZeRO-0/1/2/3 四阶段、Reduce-Scatter + All-Gather)
- [[32_distributed_optimizer_deepdive]] —— FSDP2 / ZeRO / MindSpeed 三方对比
