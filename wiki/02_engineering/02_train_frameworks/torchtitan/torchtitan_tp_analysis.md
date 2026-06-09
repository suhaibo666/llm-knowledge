# 张量并行 TP —— 机制级深度分析

> **代码基准**:torchtitan `main` @ `cf3c4312` · PyTorch `2.9.1`(DTensor / `_inductor` 内核)
> **最后更新**:2026-05-22 · **系列**:torchtitan 多维并行源码级分析(见 [[torchtitan/index]])
>
> 本文按统一结构回答:**参数怎么切?切完怎么通信?哪些通信能掩盖?异步怎么实现?** 涵盖基础 TP、Sequence Parallel(SP)、Async TP、Loss Parallel。
>
> 行号约定:torchtitan 以 `torchtitan/` 为根;PyTorch 2.9.1 以 `[pt]` 前缀,根目录 `torch/`。

---

## 1. 功能范围与定位

**TP(张量并行)** 把**单个算子(矩阵乘)**切到多张卡上——既降低单卡权重/激活显存,又把 matmul 并行化。它服务于"单层太大、单卡 matmul 太慢"的场景,典型用在节点内 NVLink。

torchtitan 当前有**两条 TP 实现并存**,写代码时务必区分:

| 路径 | 入口 | 切分声明方式 |
|------|------|-------------|
| **旧路径** | `parallelize_module(model, plan)` | 手写 `{FQN: ParallelStyle}` 字典(`ColwiseParallel`/`RowwiseParallel`/`SequenceParallel`) |
| **新路径(配置式)** | `model.parallelize(parallel_dims)` | 每个子模块 Config 上挂一个 `ShardingConfig` |

**两条路径底层机制完全相同**——都归结到 `distribute_tensor` 切权重、`DTensor.redistribute` 做通信。新路径只是把"输入/输出在边界处该 redistribute 成什么 placement"从硬编码的 `ParallelStyle` 改成声明式的 `ShardingConfig` 字段。llama3 已迁移到新路径(`torchtitan/models/llama3/parallelize.py:67/76` 调 `model.parallelize`),`torchtitan/protocols/module.py:207` 的 TODO 说明迁移尚未全部完成。

```
torchtitan 配置层                          PyTorch DTensor 层
ShardingConfig (sharding.py)
  state_shardings   ──────────────►  distribute_tensor()        切权重→DTensor
  in_src/in_dst_shardings ────────►  DTensor.from_local / redistribute  输入对齐
  out_dst_shardings ──────────────►  DTensor.redistribute()     输出对齐
  local_*_grad_placements ────────►  from_local/to_local 的 grad_placements
  local_map (LocalMapConfig) ─────►  local_map()  DTensor↔本地张量
                                          │
Module.parallelize() (module.py:176)      ▼
  _shard_states / forward_with_redistribution
                                     redistribute_local_tensor()  ([pt] _redistribute.py:157)
                                       按 _TransformInfo 逐 mesh 轴选通信:
                                       Shard→Replicate   = all_gather
                                       Partial→Replicate = all_reduce
                                       Partial→Shard     = reduce_scatter
```

---

## 2. 参数切分:`distribute_tensor` 怎么把权重切成 DTensor

### 2.1 入口:两条路径都汇聚到 `distribute_tensor`

**新路径** `Module._shard_states`(`torchtitan/protocols/module.py:222`)逐参数取出 `ShardingConfig.state_shardings` 声明的 placement,然后:

```python
# torchtitan/protocols/module.py:255
self.register_parameter(name, nn.Parameter(distribute_tensor(param, mesh, list(placements))))
```

**旧路径** `ColwiseParallel._partition_linear_fn`(`[pt] torch/distributed/tensor/parallel/style.py:123`)同样 `distribute_tensor(param, device_mesh, [Shard(0)])`。

### 2.2 `distribute_tensor` 内部:逐 mesh 轴切

`distribute_tensor`(`[pt] torch/distributed/tensor/_api.py:652`)对 placements 逐 mesh 轴处理:`Shard(d)` 调 `Shard._shard_tensor`,`Replicate()` 调 `Replicate._replicate_tensor`。

`Shard._shard_tensor`(`[pt] placement_types.py:148`)核心:

```python
scatter_list, pad_sizes = self._split_tensor(tensor, num_chunks, with_padding=True, contiguous=True)
output = torch.empty_like(scatter_list[mesh_dim_local_rank])
mesh_scatter(output, scatter_list, mesh, mesh_dim=mesh_dim, group_src=src_data_rank)
```

- `_split_tensor` 用 `torch.chunk(tensor, num_chunks, dim=self.dim)` 沿张量 `dim` 维切 `num_chunks` 块(= TP 轴 rank 数)。
- `src_data_rank` 默认 0:做一次 `mesh_scatter`(底层 `torch.distributed.scatter`),把 rank 0 上完整权重的各分片散发出去——保证"单设备语义"。若传 `src_data_rank=None` 则**跳过通信**,每个 rank 直接 chunk 自己已有的本地数据。

### 2.3 DTensor 的物理布局

```python
# [pt] _api.py:782
spec = DTensorSpec(mesh, placements, tensor_meta=TensorMeta(shape=tensor.size(), ...))
return DTensor(local_tensor, spec, ...)
```

- `DTensor` 是 `torch.Tensor` 子类,只有两个 slot:`_local_tensor`(本 rank 真实持有的分片,普通连续张量)和 `_spec`(sharding 元信息)。
- `DTensor.shape` 报告**全局逻辑形状**,`_local_tensor.shape` 是**物理分片形状**。对外像完整张量,对内只占 `1/N` 显存。

### 2.4 列并行 `Shard(0)` vs 行并行 `Shard(1)` —— TP 的核心

`nn.Linear` 算 `y = x @ weight.T`,`weight` 物理形状 `[out_features, in_features]`。

```
完整 Linear:  y[B,O] = x[B,I] @ Wᵀ[I,O]      (W 物理形状 [O,I])

列并行 ColwiseParallel —— W: Shard(0) 切 O 维
  rank0: W0=[O/2, I]   rank1: W1=[O/2, I]
  x 必须 Replicate  ──►  y0=x@W0ᵀ=[B,O/2]   y1=x@W1ᵀ=[B,O/2]
  输出 Shard(-1):每 rank 持有输出的一半通道,算完无需通信

行并行 RowwiseParallel —— W: Shard(1) 切 I 维
  rank0: W0=[O, I/2]   rank1: W1=[O, I/2]
  x 必须 Shard(-1)  ──►  x0=[B,I/2]  x1=[B,I/2]
  y_part0 = x0@W0ᵀ=[B,O]   y_part1 = x1@W1ᵀ=[B,O]
  二者都是 [B,O] 但各是"部分和" => Partial,需 all-reduce 求真值
```

要点:
- **列并行**(`colwise_config()`,`torchtitan/models/common/decoder_sharding.py:57`):weight `Shard(0)`。切 `weight` 第 0 维 = `out_features`。从 `weight.T` 看是切列,故名"列并行"。输入要 `Replicate`,输出是 `Shard(-1)`。
- **行并行**(`rowwise_config()`,`decoder_sharding.py:68`):weight `Shard(1)`。切 `in_features`。输入要 `Shard(-1)`,输出是 `Partial`(部分和)。bias `Replicate`(行并行不切输出维)。
- llama3 用法:attention 的 `wq/wk/wv` 和 FFN 的 `w1/w3` 用 colwise;`wo`、`w2` 用 rowwise——经典 Megatron "列并行→行并行"配对。

> **一个隐藏归一化**:`resolve_placements`(`torchtitan/protocols/sharding.py:174`)发现某 mesh 轴 size==1 时,把 `Shard(d)` 悄悄换成 `Replicate()`。两者在单 rank 轴上等价,但 DTensor 算子规则会把它们当不同 placement 而报错。

---

## 3. 通信原语:`redistribute` 怎么选集合通信

### 3.1 调用栈与变换计划

`DTensor.redistribute`(`[pt] _api.py:477`)→ `Redistribute.apply`(autograd Function,`[pt] _redistribute.py:281`)→ `redistribute_local_tensor`(`[pt] _redistribute.py:157`)。

`redistribute_local_tensor` 先调 `_gen_transform_infos` 生成一个 `_TransformInfo` 列表——每个元素描述"在某 mesh 轴上,placement A→B"。源/目标 placement 相同就直接复用本地张量(**零通信短路**)。

### 3.2 通信原语选择表

| 源 placement | 目标 placement | 集合通信 | 代码 |
|---|---|---|------|
| `Shard(d)` | `Replicate()` | **all_gather** | `[pt] _redistribute.py:209` |
| `Partial` | `Replicate()` | **all_reduce** | `[pt] _redistribute.py:204` |
| `Partial` | `Shard(d)` | **reduce_scatter** | `[pt] _redistribute.py:221` |
| `Replicate()` | `Shard(d)` | 本地 `torch.chunk`(**无通信**) | `[pt] _redistribute.py:226` |
| `Shard(d1)` | `Shard(d2)` | **all_to_all** | `[pt] _redistribute.py:236` |
| `Replicate()` | `Partial` | 无通信(仅前向除以 N) | `[pt] _redistribute.py:245` |

底层(`[pt] placement_types.py`):`Shard._to_replicate_tensor` → `funcol.all_gather_tensor`;`Partial._reduce_value` → `funcol.all_reduce`;`Shard._reduce_shard_tensor` → `funcol.reduce_scatter_tensor`。`funcol` = `torch.distributed._functional_collectives`,返回 `AsyncCollectiveTensor`(异步,见 §5)。

### 3.3 列并行→行并行配对:为何整对只需一次通信

这是 TP 设计的精髓。看 llama3 FFN(`w1` colwise → SiLU → `w2` rowwise):

```
x: Replicate [B,S,D]
  │ w1 (Colwise, W1:Shard(0))   本地算 x@W1ᵀ          ← 无通信(x 已 Replicate)
h1: Shard(-1) [B,S,4D/N]                              ← 输出特征维天然被切
  │ SiLU 逐元素                                        ← 无通信(逐元素算子在 Shard 上直接算)
h1': Shard(-1)
  │ w2 (Rowwise, W2:Shard(1))   w2 要求输入 Shard(-1) ← h1' 已是 Shard(-1),无通信
y_partial: Partial [B,S,D]                            ← 每 rank 一个部分和
  │ redistribute Partial→Replicate                    ◄── all_reduce(整个 FFN 唯一一次通信)
y: Replicate [B,S,D]
```

**关键**:列并行的输出 `Shard(-1)` **恰好就是**行并行所需的输入 `Shard(-1)`,中间不需要任何 redistribute(`_redistribute_inputs` 发现 placement 已匹配就跳过)。激活函数是逐元素算子,在 `Shard` 上直接算。所以一对 Colwise+Rowwise 从完整输入到完整输出,**全程只在最后做一次 all-reduce**。这就是 Megatron 风格 TP 总把 Colwise→Rowwise 成对出现的原因——不配对的话中间要 all-gather 拼回再切,通信翻倍。Attention 同理:`qkv` colwise(头维度被切)、`wo` rowwise,最后一次 all-reduce。

---

## 4. Sequence Parallel:通信如何从 all-reduce 变成 all-gather + reduce-scatter

**SP 动机**:普通 TP 下,非 TP 区域(LayerNorm/RMSNorm、dropout、residual)的激活是 `Replicate()`,每个 rank 存完整的 `[B,S,D]`,显存浪费。SP 让这些区域的激活沿**序列维 `Shard(1)`** 切分。

torchtitan 用 `enable_sp` 开关(与 TP 解耦,`torchtitan/models/llama3/sharding.py`)。`enable_sp=True` 时:norm 的输入/输出激活变 `Shard(1)`;`rowwise_config(output_sp=True)` 让 `w2/wo` 输出从 `Replicate()` 改成 `Shard(1)`。

SP 下一个 TransformerBlock 的数据流:

```
x: Shard(1) [B,S/N,D]            ← SP 区域,序列维被切,每 rank 只存 1/N 激活
  │ attention_norm (RMSNorm 沿 D 维归一,与 S 无关)  ← Shard(1) 上直接算,无通信
  │ 进 attention,in_dst 要求 Replicate
  │   redistribute Shard(1)→Replicate              ◄── all_gather(序列维)
  │ qkv colwise → 本地 attention → wo rowwise(算完 Partial)
  │ wo 的 out_dst = Shard(1)
  │   redistribute Partial→Shard(1)                ◄── reduce_scatter(序列维)
attn_out: Shard(1) [B,S/N,D]
  │ residual h = x + attn_out  两者都 Shard(1)      ← 无通信
  (FFN 同理:all_gather → w1/w2 → reduce_scatter)
```

| | 普通 TP | Sequence Parallel |
|---|---|---|
| 非 TP 区域激活 | `Replicate` 全量 `[B,S,D]` | `Shard(1)` 切分 `[B,S/N,D]` |
| 进 TP 区域 | 无 | `Shard(1)→Replicate` = **all_gather** |
| 出 TP 区域 | `Partial→Replicate` = **all_reduce** | `Partial→Shard(1)` = **reduce_scatter** |

**通信量**:按 NCCL ring 模型,1× all_reduce ≈ 2× all_gather ≈ 2× reduce_scatter(`[pt] _collective_utils.py` 的代价模型)。所以 SP 的 `all_gather + reduce_scatter` ≈ 1× all_reduce——**通信总量不变**,额外收益是非 TP 区域激活显存从 `O(S·D)` 降到 `O(S·D/N)`。这正是论文《Reducing Activation Recomputation in Large Transformer Models》的核心。SP 还有个隐性好处见 §5.3。

---

## 5. 通信掩盖与异步实现

TP 的通信与计算有**强数据依赖**,默认基本无法重叠——列并行必须先 all-gather 完整输入才能 matmul,行并行必须 matmul 算完才有部分和才能 all-reduce。这是 TP 区别于 FSDP 的本质难点。torchtitan/PyTorch 有三个层次的掩盖机制。

### 5.1 层次一:`async_op=True`(机会主义重叠)

`redistribute(async_op=True)` 让集合通信返回 `AsyncCollectiveTensor`(ACT):通信 kernel 已入队但不立即 `wait()`。`redistribute_local_tensor` 结尾(`[pt] _redistribute.py:275`):

```python
if not async_op and isinstance(new_local_tensor, funcol.AsyncCollectiveTensor):
    new_local_tensor = new_local_tensor.wait()
```

`async_op=False` 立即 wait;`async_op=True` 不 wait,把 ACT 包进返回的 DTensor,**ACT 在第一次被实际算子用到时隐式触发 wait**。torchtitan 几乎所有 redistribute 都传 `async_op=True`(`torchtitan/protocols/module.py:422/452`)。但这只是机会主义——若 wait 前恰好有不依赖该结果的算子才能重叠;对全是依赖的纯 TP 链收益有限。

### 5.2 层次二:Async TP(`_micro_pipeline_tp`)—— 真重叠

这是 TP 通信掩盖的**主力**。`maybe_enable_async_tp`(`torchtitan/distributed/tensor_parallel.py:192`)只做一件事:

```python
torch._inductor.config._micro_pipeline_tp = True   # 且强制要求 torch.compile
```

它打开一个 inductor 的 FX 图 pass。pass 挂在 **post-grad** 阶段(`[pt] torch/_inductor/fx_passes/post_grad.py:168`),所以对前向和反向的集合通信都生效。

`micro_pipeline_tp_pass`(`[pt] micro_pipeline_tp.py:1052`)做的事:把集合通信与相邻 matmul **拆成分块流水线**。

- `fuse_all_gather_matmul`:匹配 `A = all_gather(A_shard); C = A @ B` 模式(列并行 / SP 输入侧),融合成 `torch.ops.symm_mem.fused_all_gather_matmul`。
- `fuse_matmul_reduce_scatter`:匹配 `reduce_scatter(A @ B)` 模式(行并行 / SP 输出侧),融合成 `fused_matmul_reduce_scatter`。

融合 op 的核心实现在 `[pt] torch/distributed/_symmetric_memory/__init__.py`:用 **symmetric memory** + **两条 CUDA 流**,把一个大集合通信拆成 `N` 个微块,块 `i` 的 P2P 拷贝与块 `i-1` 的 matmul 重叠:

```
普通 TP(行并行输出):
  [============ matmul ============][===== reduce_scatter =====]   通信完全暴露

Async TP(fused_matmul_reduce_scatter):
  stream0: [mm块0][mm块1][mm块2][mm块3]
  stream1:       [p2p0][p2p1][p2p2][p2p3][local reduce]            通信被 matmul 掩盖
```

### 5.3 哪些 TP 通信能被掩盖

| 通信 | 能否被 Async TP 掩盖 |
|------|---------------------|
| 列并行 / SP 输入侧的 **all-gather** | 能(`fuse_all_gather_matmul`) |
| 行并行 / SP 输出侧的 **reduce-scatter** | 能(`fuse_matmul_reduce_scatter`) |
| 行并行输出 `Partial→Replicate` 的 **all-reduce** | **不能**——Async TP 的 pattern matcher 只匹配 `all_gather_into_tensor` 和 `reduce_scatter_tensor` |

这给出一个重要工程结论:**Async TP 几乎总要和 SP 一起用**。因为 SP 把"出 TP 区域"的 all-reduce 换成了可流水化的 reduce-scatter——纯 TP 的 all-reduce 无法被 Async TP 掩盖,SP 才让它变得可掩盖。

---

## 6. 反向传播:TP 在 backward 的通信

### 6.1 总原则:`redistribute` 是 autograd Function,反向通信是前向的"共轭"

`Redistribute` 是 `torch.autograd.Function`(`[pt] _redistribute.py:281`)。`Redistribute.backward`(`[pt] _redistribute.py:333`)把源/目标 spec 角色对调再调一次 `redistribute_local_tensor`。所以前向 `A→B`,反向自动 `B→A`:

| 前向 placement 变换 | 前向通信 | 反向通信 |
|---|---|---|
| `Shard→Replicate`(列并行输入 all-gather) | all_gather | **reduce_scatter** |
| `Partial→Replicate`(行并行输出 all-reduce) | all_reduce | **无通信**(特判,见下) |
| `Partial→Shard`(SP 输出 reduce_scatter) | reduce_scatter | **all_gather** |

两个关键特判:
- **`Replicate→Partial` 在 backward 是 no-op**(`[pt] _redistribute.py:245-258`):前向 `Partial→Replicate` 的严格共轭应是 `Replicate→Partial`,但把已 replicate 的梯度转回 partial 没意义(迟早要 reduce),直接当 replicate 传。
- **`Shard→Partial` 仅 backward 合法**:前向被禁(代价 `inf`),反向出现时做 all_gather。

### 6.2 `ColwiseParallelWithGradPlacement`:精确控制反向通信

问题根源:`DTensor.from_local(x, mesh, Replicate())` 的反向会把梯度强制 redistribute 回 `Replicate`。若上游 d_x 是 `Partial`,这就强制一次 all-reduce——有时是多余的。

torchtitan 的扩展 `ColwiseParallelWithGradPlacement`(`torchtitan/distributed/tensor_parallel.py:109`)给 `from_local` 多传一个 `grad_placements`:

```python
# torchtitan/distributed/tensor_parallel.py:152
input_tensor = DTensor.from_local(input_tensor, device_mesh, input_layouts,
                                  run_check=False, grad_placements=local_input_grad_placements)
```

若声明 `grad_placements={TP: Partial()}`,反向就把 d_x 当 `Partial` 接收,**跳过那次 all-reduce**,把 reduce 推迟给下游。它还强制:输入是 plain tensor 时必须显式指定 `local_input_grad_placements`(`tensor_parallel.py:148`)——呼应 `torchtitan/.claude/rules/distributed.md` 的硬性规则"任何 `to_local` 都要显式声明 `grad_placements`"。新路径的等价物是 `ShardingConfig.local_input_grad_placements` / `local_output_grad_placements` 字段。

---

## 7. Loss Parallel:词表并行下交叉熵如何避免 gather 完整 logits

### 7.1 问题

llama3 的 `lm_head` 输出 logits 沿**词表维 `Shard(-1)`**(`decoder_sharding.py:236`,`loss_parallel` 开启时)。词表常达 128256,`[B,S,vocab]` 的完整 logits 极大。直接算交叉熵需先 all-gather 把 logits 拼完整——巨大通信 + 显存。Loss Parallel 让交叉熵**直接在 `Shard(vocab)` 的 logits 上算**。

### 7.2 机制:`loss_parallel()` 替换 aten 算子的 DTensor handler

`loss_parallel()`(`[pt] torch/distributed/tensor/parallel/loss.py:30`)是 context manager,进入时往 `DTensor._op_dispatcher._custom_op_handlers` 注册 6 个自定义 handler(`_log_softmax`、`nll_loss_forward`、各自的 backward 等)。`F.cross_entropy` 内部分解成 `log_softmax + nll_loss`,在 context 内这些 aten 算子作用于 DTensor 时走分布式实现。

### 7.3 分布式 `log_softmax`:用微张量 all-reduce 替代 gather 大 logits

`_log_softmax`(`[pt] loss.py:128`):标准 log_softmax 需要全词表的 `max` 和 `sum(exp)`。分布式版:

```python
x_max = torch.amax(x, dim, keepdim=True)                              # 本地 vocab 分片的 max
x_max = funcol.all_reduce(x_max, reduceOp=MAX, group=...)             # ◄ all_reduce(MAX),形状 [B,S,1]
shifted_sumexp = torch.sum(torch.exp(x - x_max), dim, keepdim=True)
shifted_sumexp = funcol.all_reduce(shifted_sumexp, reduceOp=SUM, ...) # ◄ all_reduce(SUM),形状 [B,S,1]
result = (x - x_max) - torch.log(shifted_sumexp)
```

关键:两次 all-reduce 传的是 `keepdim` 后的 `[B,S,1]` **微张量**,而不是 `[B,S,vocab]`。

### 7.4 分布式 `nll_loss`:`_MaskPartial` 处理跨分片 gather

NLL loss 要 `-logp[target]`,但 `target` 是全局词表索引,当前 rank 只有 `vocab/N` 分片。`_MaskPartial`(`[pt] torch/distributed/tensor/_ops/_embedding_ops.py:69`)的做法:造一个 mask 标记"target 不在本 rank 分片内"的位置,本地 `gather` 后把不命中的位置清零,再 `all_reduce(SUM)`——每个有效 target 恰好只在一个 rank 命中,求和即得真值。这次 all-reduce 传 `[B,S]`,又是微张量。

### 7.5 反向:融合掉一次 all-gather

`_nll_loss_and_log_softmax_backward`(`[pt] loss.py:346`)把 nll + log_softmax 两步反向**融合**。普通做法 log_softmax 反向需要全词表信息会引入一次 all-gather;融合后用 `(grad_input + exp(x)) * grad_output`(`exp(x)` 即 softmax 概率,本地可得)在本地分片上算完,**省掉这次 all-gather**。

```
朴素:logits Shard(vocab) ──all_gather──► Replicate[B,S,128256]  ◄ 巨量通信+显存
Loss Parallel:logits 不动 → 前向 3 次 [B,S,1]/[B,S] 微 all-reduce → 反向 0 通信
              完整 logits 从不物化
```

约束:`loss_parallel` 当前只支持 1D mesh。

---

## 8. torchtitan 的接入:配置式 sharding

新路径的核心是 `Module.parallelize()`(`torchtitan/protocols/module.py:176`):递归遍历模块树,对每个带 `sharding_config` 的模块:

1. `_shard_states`:用 `distribute_tensor` 把参数变成 DTensor。每个参数独立 `resolve_mesh(axes)`——**非 full_dtensor 路径下只保留 `tp`/`ep` 轴为"带内"**(DP/CP 属带外,交给 FSDP/CP),见 [[torchtitan_parallel_dims_analysis]] §7。
2. 把 `forward` 包成 `redistribute 输入 → [local_map] → forward → redistribute 输出`。

`ShardingConfig`(`torchtitan/protocols/sharding.py`)的字段:`state_shardings`(参数切法)、`in_src/in_dst_shardings`(输入 redistribute)、`out_src/out_dst_shardings`(输出 redistribute)、`local_map`(attention 内核用,把 DTensor 转本地张量再跑 SDPA/Flex)。`decoder_sharding.py` 提供 `colwise_config()`/`rowwise_config()`/`norm_config()` 等工厂,`llama3/sharding.py` 把它们填到各层 Config 上。

---

## 9. 完整流程图

```
═══ 建模期 ═══
set_llama3_sharding_config()  在各层 Config 挂 ShardingConfig
        │
model.parallelize(parallel_dims)            torchtitan/protocols/module.py:176
   ├─ _shard_states:distribute_tensor 切权重 → DTensor(Shard(0)/Shard(1))
   └─ 包装 forward:redistribute 输入 → forward → redistribute 输出
        │
maybe_enable_async_tp() → torch._inductor.config._micro_pipeline_tp = True

═══ 前向期(一个 TransformerBlock,SP 开启) ═══
x:Shard(1) ─allgather─► Replicate ─[qkv colwise→attn→wo rowwise]─► Partial
           ─reduce_scatter─► Shard(1)  (FFN 同构)
   若开 Async TP + torch.compile:上述 allgather/reduce_scatter 被 inductor
   pass 拆块,与 matmul 在 symmetric memory 双流上流水重叠

═══ 反向期 ═══
Redistribute.backward 自动做共轭通信:
   allgather ↔ reduce_scatter,Partial→Replicate 反向为 no-op
ColwiseParallelWithGradPlacement 通过 grad_placements 控制 d_x 是否做 all-reduce

═══ Loss ═══
loss_parallel():logits 保持 Shard(vocab),交叉熵用 3 次微 all-reduce 算完
```

---

## 10. 小结

- **参数切分**:两条路径(`parallelize_module` 旧路径 / `ShardingConfig` 新路径)都汇聚到 `distribute_tensor`,用 `torch.chunk` + `mesh_scatter` 把权重切成 `DTensor`。列并行 `Shard(0)` 切输出维、行并行 `Shard(1)` 切输入维。
- **通信原语**:`redistribute` 按源/目标 placement 选通信——`Shard→Replicate`=all_gather、`Partial→Replicate`=all_reduce、`Partial→Shard`=reduce_scatter。**列并行→行并行配对**让一对 linear 全程只需一次集合通信。
- **Sequence Parallel**:把非 TP 区域激活沿序列维 `Shard(1)`,出入 TP 区域时的 all-reduce 被换成 all-gather + reduce-scatter(通信量等价,激活显存降 N 倍)。
- **通信掩盖**:TP 通信与计算强依赖,默认难重叠。`async_op=True` 是机会主义重叠;**Async TP**(`_micro_pipeline_tp` inductor pass)才是真重叠——用 symmetric memory 双流把 all-gather/reduce-scatter 与 matmul 拆块流水。Async TP **不能掩盖 all-reduce**,故几乎总与 SP 绑定。
- **异步实现**:底层是 `funcol` 返回 `AsyncCollectiveTensor` 延迟 wait;Async TP 的双流流水在 `fused_all_gather_matmul`/`fused_matmul_reduce_scatter` 里实现。
- **反向**:`Redistribute` 是 autograd Function,反向通信是前向的共轭;`ColwiseParallelWithGradPlacement` / `grad_placements` 精确控制 d_x 是否做 all-reduce。
- **Loss Parallel**:词表并行交叉熵,logits 永不物化,前向 3 次微张量 all-reduce、反向融合掉 all-gather。

## Related Pages

- [[torchtitan/index]] · [[torchtitan_parallel_dims_analysis]] —— 知识地图与并行基座
- [[torchtitan_fsdp_analysis]] —— FSDP 与 TP 的 DTensor 叠加
- [[torchtitan_ep_analysis]] —— MoE 专家的 TP 切法(`Shard(1)/Shard(2)`)对比
- [[tp_analysis]] —— Megatron-LM 张量并行(`ColumnParallel`/`RowParallel` 共轭算子 f/g)
- [[async_collective_tensor_deep_dive]] —— `AsyncCollectiveTensor` 源码追踪(`__torch_dispatch__`、`wait_tensor`)
- [[comm_compute_overlap_analysis]] —— 计算通信掩盖对比分析
- [[llm_parallelism_analysis]] —— TP/SP 通信依赖与正反向 DAG
