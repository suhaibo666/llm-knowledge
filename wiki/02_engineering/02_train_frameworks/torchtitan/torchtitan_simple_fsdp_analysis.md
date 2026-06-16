# SimpleFSDP —— 编译器友好的 FSDP:分片即可追踪的 DTensor 集合通信(源码级)

> **代码基准**:torchtitan `main` @ `61c010fcb`(`experiments/graph_trainer/`)· PyTorch nightly(实验件)
> **最后更新**:2026-06-16 · **系列**:torchtitan 多维并行源码级分析(见 [[torchtitan/index]])
>
> 对象是 torchtitan `experiments/graph_trainer/` 的 SimpleFSDP(论文 arXiv:2411.00284),FSDP2 的编译器友好替代:把 all-gather/reduce-scatter 表达成 DTensor `redistribute` 进图,交编译器分桶重叠。对照 [[torchtitan_fsdp_analysis]] / [[torchtitan_hsdp_backward_overlap_analysis]]。
> 行号约定:实验代码以 `graph_trainer/`(= `torchtitan/torchtitan/experiments/graph_trainer/`)为根;PyTorch 以 `[pt]` 前缀。

---

## 0. 一句话定位

```
FSDP2(02/07):  eager + FSDPParamGroup + 5 条 CUDA stream + 预取钩子 + event 手写编排
                通信藏在 module hook 里,编译器看不见

SimpleFSDP:     把"分片参数 -> 完整参数"表达成一个 DTensor.redistribute(Replicate)
                = 前向 all-gather;其反向的 Partial(sum) -> Shard = reduce-scatter
                通信变成图里的 collective 节点 -> 编译器 pass 做分桶/重排/重叠
                "Sharding as parameterized collectives within the computation graph"(README:9)
```

核心权衡:**把通信编排从"运行时手写多流"挪到"编译期图变换"**。代价是必须走编译路径(`aot_fx_trace`,nightly);收益是通信节点对编译器可见,可与 AC/CUDAGraph/CPU-offload/Async-TP **作为同质的图 pass 自由组合**(MANIFESTO:42-46),并号称比 eager FSDP2 有显存与吞吐改进(README:9)。

---

## 1. 核心思想:参数化(parametrization)+ 可追踪集合通信

SimpleFSDP 用 PyTorch 的 **parametrization** 机制:注册到模块上的参数是**分片 DTensor**,但每次 `forward` 通过一个参数化模块 `ReplicateComputation` 把它"动态还原"成完整参数。这个还原就是一次 `redistribute`,在 traced 图里就是一个 all-gather collective。

- MANIFESTO:48-51:"expresses all-gather and reduce-scatter as **traceable DTensor operations**. The collectives show up as **nodes in the graph**, so they can be reordered, fused, and overlapped by passes — not hidden behind opaque module hooks."

于是"参数怎么取回来 / 梯度怎么规约"不再是 FSDP2 那套 `unshard()/wait_for_unshard()/post_backward` 钩子链,而是**纯 DTensor 语义 + 编译器 pass**。

---

## 2. 参数切分:`data_parallel` + `distribute_tensor`

入口 `data_parallel(model, device_mesh, mode, mp_policy, shard_dim, full_dtensor)`(`simple_fsdp.py:261`)。对每个 module 的直接参数:

1. **切成分片 DTensor**:`distribute_tensor(p, device_mesh, param_sharding)`(`:294-302`)。若参数已是 DTensor(被 TP/EP 先切过),改用 `_distribute_dtensor` 组合(§4)。
2. **注册参数化** `ReplicateComputation`(`:318-328`)。
3. 已经是 SimpleFSDP 包过的模块跳过(`if "SimpleFSDP" in mod.__class__.__name__: continue`,`:289-290`),避免重复套。

三种模式 → 三种 placement(`:270-281`):

| mode | param_sharding | 等价 |
|---|---|---|
| `replicate` | `(Replicate(),)` | DDP(参数全复制) |
| `fully_shard` | `(Shard(shard_dim),)` | FSDP / ZeRO-3(默认 dim 0) |
| `hybrid_shard` | `(Replicate(), Shard(shard_dim))` | HSDP(需 2D mesh,`:277-279`) |

> 与 FSDP2 一样,分片参数注册在模块上 → **优化器状态天然 1/N**;混合精度由 §3 的 redistribute dtype 控制。

---

## 3. 运行时:`ReplicateComputation` 的 redistribute

参数化模块 `ReplicateComputation`(`simple_fsdp.py:167`)的 `forward(x)` 调 `replicate_compute(x)`(`:189`)。核心(无 TP/EP 的 `non_dp_mesh_dims==0` 分支,`:231-239`):

```python
output = x.redistribute(
    placements=self.compute_placements,    # [Replicate()] * ndim  -> 前向 all-gather
    forward_dtype=self.param_dtype,        # bf16:all-gather 前转 dtype
    backward_dtype=self.reduce_dtype,      # fp32:反向规约 dtype
)
if not self.full_dtensor:
    output = output.to_local(grad_placements=self.grad_placements)  # [Partial(sum)] * ndim
```

- `compute_placements = [Replicate()] * mesh.ndim`(`:180`):**前向把分片参数 redistribute 成 Replicate = all-gather**,得到完整参数做本地计算。
- `grad_placements = [Partial(reduce_op="sum")] * mesh.ndim`(`:181-183`):告诉 DTensor"本地张量的梯度是 Partial(sum)"。**反向时 DTensor 自动把 Partial(sum) 规约回原 placement**:
  - `fully_shard`:Partial(sum) → Shard = **reduce-scatter**;
  - `replicate`(DDP):Partial(sum) → Replicate = **all-reduce**;
  - `hybrid_shard`(HSDP):两者混合(组内 RS + 组间 AR)。
  - 代码注释明示(`:190-193`):"gradients are partial tensors that need reduction (DDP: allreduce, FSDP: reduce_scatter, HSDP: mix of both)"。

> 对照 [[torchtitan_hsdp_backward_overlap_analysis]]:FSDP2 用 `foreach_reduce` 在 RS/AR 两条流上手写 HSDP 反向;**SimpleFSDP 只声明 `grad_placements=Partial(sum)`,RS/AR 由 DTensor 反向 + 编译器 pass 生成**。语义等价,实现路径完全不同。

- `forward_dtype/backward_dtype`(`:212-213`):把混合精度(`param=bf16 / reduce=fp32`)**编进 redistribute**,无需 FSDP2 的 `MixedPrecisionPolicy` 钩子。
- `full_dtensor=True` 时输出保持 DTensor(不 `to_local`,`:238-239`),用于整网 DTensor/autoparallel 路径;nD 并行下未实现(`:197-200`)。

### 3.1 动态子类 + 缓存(为 compile 复用)

`_register_parametrization`(`:135`)不是用 `nn.utils.parametrize`,而是**动态造一个子类** `SimpleFSDP{ClassName}`,把参数名变成 `property` getter,getter 调 `parametrization(self._parameters[pn])`(`:145-164`)。

- **按 `(原类, 参数名集合)` 缓存**(`_wrap_class_cache`,`:130-132/151-163`):同类型模块共享同一个 SimpleFSDP 子类 → `torch.compile` 只需编译一次、复用到每层(与 [[torchtitan_compute_memory_optimizations_analysis]] §4 逐 block compile 复用同理)。
- 子类注入 `sys.modules` 让 pickle/GraphPickler 能解析(为 precompile 序列化,`:160-162`)。
- 为何不用官方 `parametrize`:为兼容 DCP(分布式 checkpoint),`state_dict` 直接从 `self._parameters` 取分片参数、不触发 getter(`:138-143`)。

### 3.2 `disable_active_parametrization`:初始化/调试旁路

全局开关 `_active_parametrization`(`:29`)+ 上下文管理器(`:32-39`)。`forward` 里若关闭则**直接返回 x**(`:254-255`),不做 all-gather。用于模型初始化/状态检查:`with disable_active_parametrization(): model.init_states()`(`:251-253`)。各模型 `model.py` import 它(如 `llama3/model.py:13`)。

---

## 4. 与 TP/EP 组合:`_distribute_dtensor`

当参数已被 TP/EP 切成 DTensor,DP 不能再 `distribute_tensor` 重切,而是组合 **outer=DP mesh / inner=TP/EP mesh**(`_distribute_dtensor`,`simple_fsdp.py:48`):

- 用 `DeviceMesh._concatenate([outer, inner])` 拼出跨 mesh(`:61`),placement = `(DP_placement,) + inner_spec.placements`。
- DP 分片维与 inner 分片有嵌套时用 `_StridedShard(shard_dim, split_factor=inner.num_shards_map[shard_dim])` 表达"先 TP 切、再 FSDP 切"的顺序(`:70-94`)——与 FSDP2 在 [[torchtitan_fsdp_analysis]] §2.3 用 `_StridedShard` 叠 TP 是同一思想。
- 支持 FSDP/DDP/HSDP × EP/TP/EP+TP(`:54-94`)。

运行时若有 TP/EP 轴(`non_dp_mesh_dims>0`,`:194-230`):先把 2D DTensor **重包成 dp_mesh 上的 1D DTensor** 做高效 all-gather,redistribute 后再重包回 non_dp_mesh(注释 TODO:DTensor 应原生支持这种 mesh 折叠,`:216-217`)。

---

## 5. 编译器接管掩盖:collectives 成图节点后的 pass

SimpleFSDP 自己**不写多流编排**;all-gather/reduce-scatter 进图后,由 `fsdp_passes.py` 的图 pass 做重叠(它们对无 FSDP collective 的图是 no-op,`fsdp_passes.py:10-12`):

- **`reassign_collective_pgs_pass`**(`fsdp_passes.py:129`):把 all-gather 改派到一个**额外的 NCCL 进程组**(同 ranks、独立 CUDA stream)。注释直说原因(`:133-140`):"Each PG runs on its own CUDA stream, so moving a collective to an extra PG lets it **overlap with the collectives left on the original PG — e.g. all-gathers overlapping reduce-scatters in backward**"。这正是 FSDP2 用"独立 AG 流 / RS 流"换来的反向 AG∥RS 重叠([[torchtitan_hsdp_backward_overlap_analysis]] §2),这里改由 pass 加 PG 实现。
- **`autobucketing_reordering_pass`**(`:167`):`schedule_overlap_bucketing(collective_bucketing=True)` 自动分桶 + 重排,优化 comm/compute 重叠。
- **`transformer_block_bucketing_reordering_pass`**(`:181`):按 TransformerBlock 手动分桶(`manual_overlap_bucketing`)。
- **任意前移预取**:`is_wait_tensor_from_fsdp`(`:52-64`)识别"输入可一路回溯到 graph input"的 all-gather——这种 AG 可被**任意提前**(等价 FSDP2 的预取),供调度 pass 前移到计算影子里。

> 对照 graph_trainer CLAUDE.md / README:bucketing、async-TP、CUDAGraph、CPU-offload、AC 全是**同一张图上的 pass**,自由组合(README:11)。Async-TP 微流水见 [[torchtitan_comm_optimizations_overlap_analysis]] §2。

---

## 6. 与 FSDP2 的对比

| 维度 | FSDP2(`fully_shard`,[[torchtitan_fsdp_analysis]]/[[torchtitan_hsdp_backward_overlap_analysis]]) | SimpleFSDP(本篇) |
|---|---|---|
| 通信单位 | `FSDPParamGroup`(自底向上分组) | 逐参数 DTensor + per-module 参数化 |
| 取回参数 | `unshard()` 钩子发 all-gather | 前向 `redistribute(Replicate)` 进图 |
| 梯度规约 | `post_backward` → `foreach_reduce`(RS/AR) | 声明 `grad_placements=Partial(sum)`,DTensor 反向自动 RS/AR |
| 掩盖编排 | 5 条 CUDA stream + 预取钩子 + event(运行时) | 编译器图 pass:额外 PG + 分桶 + 重排(编译期) |
| 混合精度 | `MixedPrecisionPolicy` | redistribute 的 `forward_dtype/backward_dtype` |
| 对编译器 | 通信藏在 module hook,tracer 看不见 | 通信是显式图节点,可重排/融合/重叠 |
| 运行环境 | eager(主路径) | `torch.compile aot_fx_trace`,需 nightly(实验) |
| HSDP | 第 5 条 all_reduce 流手写 | `(Replicate, Shard)` placement,反向自动出 RS+AR |

**本质差异**:FSDP2 把"何时通信、藏在哪"写死在运行时;SimpleFSDP 把它**留给编译器在图上决定**——这也是 graph_trainer 整体哲学(MANIFESTO:42-46 "Every optimization is a graph pass")。

---

## 7. 接入、组合与约束

- **接入**:`apply_simple_fsdp(model, parallel_dims, training)`(`common_utils.py:217`)按 `parallel_dims` 选 mesh 与 mode(`dp_replicate+dp_shard/cp` → `hybrid_shard`;仅 `dp_replicate` → `replicate`;否则 `fully_shard`,`:228-237`),对模型整体 `data_parallel`(`:273`);MoE 的 `moe.experts` 在 EP 开启时单独用 `edp_mesh` 包(`:265-271`)——与 FSDP2 的 `edp_mesh` 思想一致([[torchtitan_ep_analysis]] §7)。
- **组合现状**(README:200-218 composability 表,`aot_fx_trace`):Meta init / AC / 混合精度 / TP / CP / EP / DCP / CUDAGraph 已 ✅;Float8/MXFP8、EP+AC、EP+PP、图级 PP、microbatch overlap、precompile 仍 🚧。
- **AutoParallel**:可用 `--compile.enable_autoparallel` 让 AutoParallel 解 SPMD placement,再走同一 `aot_fx_trace` 流水(README:87-126);numerics 与 eager 紧密一致但不要求逐位相同。
- **约束**:需 PyTorch nightly(README:26);走编译路径;`full_dtensor` 的 nD 并行未实现(`simple_fsdp.py:197-200`)。

---

## 8. 源码复核小结

| 断言 | 位置 | 结果 |
|---|---|---|
| 分片表达成图内可追踪集合通信 | `MANIFESTO.md:48-51`、`README.md:9` | OK |
| `data_parallel` 逐 module distribute_tensor + 注册参数化 | `simple_fsdp.py:261-329` | OK |
| 三模式 placement(replicate/fully_shard/hybrid_shard) | `simple_fsdp.py:270-281` | OK |
| 前向 redistribute→Replicate = all-gather | `simple_fsdp.py:232-236` | OK |
| 反向 grad_placements=Partial(sum) 触发 RS/AR | `simple_fsdp.py:181-183/190-193` | OK |
| 混合精度 = redistribute forward/backward_dtype | `simple_fsdp.py:212-213` | OK |
| 动态子类 + 按 (类,参数名) 缓存供 compile 复用 | `simple_fsdp.py:130-164` | OK |
| `disable_active_parametrization` 旁路(init/调试) | `simple_fsdp.py:32-39/254-255` | OK |
| TP/EP 组合用 `_distribute_dtensor` + `_StridedShard` | `simple_fsdp.py:48-127` | OK |
| 额外 NCCL PG → 独立流 → 反向 AG∥RS 重叠 | `fsdp_passes.py:129-164` | OK |
| 自动/手动分桶重排 pass | `fsdp_passes.py:167-194` | OK |
| 可任意前移的 all-gather 识别(预取) | `fsdp_passes.py:52-64` | OK |
| 接入 `apply_simple_fsdp`,MoE 专家走 edp_mesh | `common_utils.py:217-282` | OK |
| 实验件,需 nightly,Float8/PP/microbatch overlap 仍 🚧 | `README.md:24-29/200-218` | OK |

---

## 9. 小结

- **是什么**:SimpleFSDP = 把 DDP/FSDP/HSDP 表达成"分片 DTensor + 参数化的 `redistribute`",前向 redistribute→Replicate 即 all-gather,反向 Partial(sum)→Shard/Replicate 即 reduce-scatter/all-reduce。通信成为**图里的显式节点**。
- **怎么掩盖**:不写多流,改由编译器图 pass——额外 NCCL PG(独立流)实现反向 AG∥RS、`schedule_overlap_bucketing` 分桶重排、可前移 AG 实现预取。
- **怎么组合**:与 TP/EP 用 `_distribute_dtensor`+`_StridedShard` 嵌套;混合精度编进 redistribute dtype;AC/CUDAGraph/CPU-offload/Async-TP 都是同一张图上的 pass。
- **与 FSDP2 的关系**:**语义等价、实现哲学相反**——FSDP2 运行时手写编排(主路径、eager),SimpleFSDP 编译期图变换(实验、需 nightly)。它是 graph_trainer "Every optimization is a graph pass" 哲学在数据并行上的落地,也是 [[torchtitan_comm_optimizations_overlap_analysis]] §5 `full_dtensor`/autoparallel 方向的近亲。

---

## Related Pages

- [[torchtitan_fsdp_analysis]] —— FSDP2 eager 实现(SimpleFSDP 的对照对象)
- [[torchtitan_hsdp_backward_overlap_analysis]] —— HSDP 反向多流(SimpleFSDP 用 placement 表达)
- [[torchtitan_comm_optimizations_overlap_analysis]] —— 编译器通信优化 / full_dtensor 近亲
- [[torchtitan/index]] —— torchtitan 多维并行知识地图
