# 专家并行 EP —— 机制级深度分析

> **代码基准**:torchtitan `main` @ `cf3c4312` · PyTorch `2.9.1`(`_functional_collectives` 内核)
> **最后更新**:2026-05-22 · **系列**:torchtitan 多维并行源码级分析(见 [[torchtitan/index]])
>
> 本文按统一结构回答:**专家怎么切?token 怎么路由通信?哪些通信能掩盖?异步怎么实现?** 重点是 token all-to-all 与 `AsyncCollectiveTensor`。
>
> 行号约定:torchtitan 以 `torchtitan/` 为根;PyTorch 2.9.1 以 `[pt]` 前缀。

---

## 1. 功能范围与定位

**EP(专家并行)** 服务于 MoE 模型(llama4、deepseek_v3、gpt_oss)。MoE 有几十~几百个专家(expert),全部复制到每张卡显存会爆。EP 把**专家分散到不同卡**,每卡只持有一部分专家。

如 [[10_torchtitan_parallel_dims_analysis]] 所述,EP 不占 `world_size` 乘积——它是对 `dp_shard × cp × tp` 子网格的重新切分,专家参数活在 `sparse_mesh` 上。

**一条贯穿全文的主线 —— EP 与 TP 的切分对偶**:

| | 切什么维 | 每卡持有 | 通信发生在 |
|---|---|---|---|
| **EP** | 专家维(`dim=0`) | 全部专家的一个**子集**,每个专家权重**完整** | **token** 上(all-to-all) |
| **TP** | GEMM 矩阵维(`dim=1/2`) | **全部**专家,但每个专家权重被**切碎** | **激活** 上(all-reduce/reduce-scatter) |

MoE 层骨架(`torchtitan/models/common/moe.py`):`MoE.forward` → router 算专家分配 → `GroupedExperts.forward` 三步 `dispatch → _experts_forward → combine`,全委托给 `token_dispatcher` 策略对象。

---

## 2. 专家权重切分:`ExpertParallel` 的 `Shard(0)`

### 2.1 专家权重的原始形状

`GroupedExperts`(`torchtitan/models/common/moe.py`)的三个权重,**第 0 维都是专家数**:

```python
w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))   # (E, H, D)  gate
w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))   # (E, D, H)  down
w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))   # (E, H, D)  up
```

### 2.2 EP 开启:`ExpertParallel._partition_fn`

`torchtitan/distributed/expert_parallel.py:50`:

```python
class ExpertParallel(ParallelStyle):
    def _partition_fn(self, name, mod, device_mesh):
        for param_name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(param_name, dist_param)
        mod.token_dispatcher.ep_mesh = device_mesh         # 把 1D EP mesh 绑上 dispatcher
```

- w1/w2/w3 无差别套 `Shard(0)`——沿专家维切到 EP mesh 各 rank。
- **每张卡持有哪些专家**:EP degree = `ep_size`,每 rank 持有 `num_experts/ep_size` 个**连续编号**的专家。`num_experts=8, ep_size=4` → rank0 拿专家 {0,1}、rank1 {2,3}、rank2 {4,5}、rank3 {6,7}。每个本地专家的权重是**完整未切**的矩阵。
- 同时把 1D EP `DeviceMesh` 写进 `token_dispatcher.ep_mesh`——把"权重切分"和"通信原语"绑在同一个 mesh 上。

### 2.3 EP 关闭、TP 开启:`TensorParallel` 的 `Shard(1)/Shard(2)`

对比 `expert_parallel.py:23` `TensorParallel`:w1/w3 切 `Shard(1)`(沿 hidden,column-wise)、w2 切 `Shard(2)`(沿 hidden,row-wise)——经典 Megatron MLP 切法,只是多了不切的专家维 `dim=0`。

| 权重 | EP 切法 | TP 切法 |
|---|---|---|
| w1 (gate) | `Shard(0)` 沿专家 | `Shard(1)` column-wise |
| w2 (down) | `Shard(0)` 沿专家 | `Shard(2)` row-wise |
| w3 (up) | `Shard(0)` 沿专家 | `Shard(1)` column-wise |

装配入口 `apply_moe_ep_tp`(`torchtitan/models/llama4/parallelize.py:367`):`ep_mesh is None` 时用 `TensorParallel()`,否则用 `ExpertParallel()`。

### 2.4 专家计算消费切片

`_experts_forward`(`moe.py`)在权重是 DTensor 时先 `.to_local()` 转回普通 Tensor(EP 下输入是动态形状——每卡收到的 token 数运行时才知,不便表达成 DTensor),再用 `torch._grouped_mm` 配合 `offsets` 做分组 GEMM:一次 kernel 算完本地所有专家,每个专家吃 `offsets` 划出的那段连续 token。

---

## 3. 通信原语:token all-to-all dispatch / combine

EP 下专家分散在各卡,router 在每张卡上对本卡 token 算出 top-k 专家分配,但这些 token 要去的专家可能在别的卡。**dispatch** 把 token 按目标专家所在 rank 重新分发,计算完 **combine** 送回来。核心是 `AllToAllTokenDispatcher`(`torchtitan/models/common/token_dispatcher.py:156`)。

### 3.1 dispatch 四阶段

**① 本地按专家重排**(`token_dispatcher.py:254`):

```python
num_tokens_per_expert = torch.histc(selected_experts_indices.view(-1), bins=num_experts, ...)
token_indices_experts_sorted = torch.argsort(selected_experts_indices.view(-1), stable=True)
routed_input = x[token_indices_experts_sorted]   # (N*top_k, dim),本卡内按专家 0,1,2... 排好
```

**② 交换计数**(`token_dispatcher.py:280`)——**为何要先做一次小 all-to-all**:

token all-to-all 是**变长**的——本卡发给 rank j 多少 token、从 rank j 收多少,都依赖路由结果,运行时才知道。`all_to_all_single` 的 `input_splits/output_splits` 参数必须是具体数字,所以**先用一次小 all-to-all 交换 `num_tokens_per_expert`**(长度 `num_experts` 的计数张量):

```python
num_tokens_per_expert_group = all_to_all_single(num_tokens_per_expert, None, None, group=ep_mesh)
num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(...)   # 显式 wait
input_splits  = num_tokens_per_expert.view(ep_size,-1).sum(1).to("cpu")     # 发给每个 rank 的 token 数
output_splits = num_tokens_per_expert_group.view(ep_size,-1).sum(1).to("cpu", non_blocking=False)
```

`.to("cpu")` 是必要的 **D2H 同步**(split 参数必须是 CPU int list)。`output_splits` 强制 `non_blocking=False`——后面紧接 `.tolist()` 要读它,异步拷贝会读到旧值。**这些 D2H 同步会阻塞 CPU 线程**,是后面 §7 显式预取的根因。

**③ token all-to-all**(`token_dispatcher.py:312`):

```python
routed_input = all_to_all_single_autograd(routed_input, output_splits, input_splits, ep_mesh)
```

真正的 token 分发,用 **autograd 版**(token 数据要反传梯度)。返回 `AsyncCollectiveTensor`——通信此时只是 enqueue,未 wait(见 §4)。

**④ rank-major → expert-major 重排**(`_permute`,`token_dispatcher.py:349`):all-to-all 收回的布局是 rank-major `[(e0,r0),(e1,r0),(e0,r1),...]`(先按发送方 rank 分块);但本卡的分组 GEMM 要求 token 按专家连续(expert-major)。`_permute` 做这个转置,并把 `permuted_indices` 存进 metadata 供 combine 逆转。

```
                       === DISPATCH ===
每卡 router 算出 top-k 专家分配
   │ ① histc + argsort:本卡内按专家排序 routed_input
   │ ② all_to_all_single(计数) → 算 input_splits/output_splits (含 D2H 同步)
   │ ③ all_to_all_single_autograd(token) → 返回 ACT,未 wait
   │      rank0 ─┐ 专家{0,1}的token ┌─► rank0
   │      rank1 ─┤   ╳ 交叉分发      ├─► rank1
   │      rank2 ─┤                  ├─► rank2
   │      rank3 ─┘                  └─► rank3
   │ ④ _permute:rank-major → expert-major
   ▼ routed_input (R, dim) → _grouped_mm 分组 GEMM
```

### 3.2 combine

`combine`(`token_dispatcher.py:399`)是 dispatch 的逆:`_unpermute`(expert-major→rank-major)→ `all_to_all_single_autograd`(input/output splits **互换**)→ 应用路由权重 → `deterministic_scatter_add` 把结果散回原 token 位置。

---

## 4. 异步实现:`AsyncCollectiveTensor` 怎么"延迟 wait"

这是 EP 通信掩盖的机制基础。

### 4.1 两条返回路径

`all_to_all_single_autograd`(`[pt] torch/distributed/_functional_collectives.py:493`)调底层 op 后,eager 下经 `_maybe_wrap_tensor` 返回 **`AsyncCollectiveTensor`(ACT)**。底层 op 调用本身就是**异步发起** NCCL 通信,返回的是"通信尚未完成"的 tensor。

### 4.2 ACT 的延迟 wait 机制

`AsyncCollectiveTensor`(`[pt] _functional_collectives.py:561`)是 `torch.Tensor` 的 wrapper 子类,核心在 `__torch_dispatch__`(`_functional_collectives.py:631`):

```python
@classmethod
def __torch_dispatch__(cls, func, types, args, kwargs):
    is_view_op = _is_view_op(func)
    def unwrap(e: AsyncCollectiveTensor):
        if not is_view_op:
            return e.trigger_wait()   # 非 view op → 真正触发 wait_tensor!
        return e.elem                  # view op → 不 wait
    ...
```

**机制**:任何 op 作用到 ACT 上都进 `__torch_dispatch__`:
- **view 类 op**(reshape/view/transpose/slice):**不 wait**,只对 `.elem` 操作——对在途 all-to-all 结果做 reshape 不会强制同步。
- **非 view op**(任何真正读取/计算数据的 op:加法、乘法、`scatter_add`、matmul):`trigger_wait()` → 真正触发 `wait_tensor`,通信在此刻同步。`completed` 标志保证 wait 只做一次。

> 这就是"延迟 wait"的精髓:**reshape 不同步,计算才同步**。它让"发起通信"和"等待通信完成"之间能塞进别的计算。

dispatch 里 all-to-all 后紧接的 `_permute` 里 `routed_input[permuted_indices]`(index_select 是非 view op)→ 在此触发 wait。所以 dispatch 的 token a2a 进专家 GEMM 前必然已同步。

---

## 5. 通信掩盖:`combine()` 的 shared_experts 重叠

### 5.1 标准 all-to-all 路径

`combine`(`token_dispatcher.py:428`)关键几行:

```python
routed_output = _unpermute(routed_output, ...)
# all-to-all combine:返回 ACT,a2a 跑在 NCCL stream,访问前不阻塞
routed_output = all_to_all_single_autograd(routed_output, input_splits, output_splits, ep_mesh)
# shared_experts 与异步 a2a 重叠
out = shared_experts(x) if shared_experts is not None else torch.zeros_like(x)
...
out = deterministic_scatter_add(out, token_indices, routed_output)   # ← 强制 a2a 同步
```

**重叠机制**:
1. `all_to_all_single_autograd` 把 combine 的 token a2a **enqueue 到 NCCL stream**,立即返回 ACT(`routed_output`),通信在 GPU 后台跑。
2. 下一行 `shared_experts(x)` —— `x` 是普通 tensor(原始输入)**不是 ACT**,所以 shared experts 的 FFN 计算 enqueue 到默认 compute stream,**不会触发 `routed_output` 的 wait**。
3. 于是:**NCCL stream 跑 combine all-to-all 通信 ‖ compute stream 跑 shared experts FFN**——通信被计算掩盖。
4. `deterministic_scatter_add` 的 `src` 是 `routed_output`(ACT),`scatter_add` 读取它的实际数据(非 view op)→ `__torch_dispatch__` 自动 `trigger_wait` → **强制 wait combine 的 all-to-all 完成**。此时 shared_experts 已算完,掩盖窗口刚好闭合。

> 这就是为什么 `combine` 接收 `shared_experts` 作为参数而不在 `MoE.forward` 里单独算——为了把它塞进"a2a enqueue 和 a2a 结果首次使用"之间的窗口。

```
combine:
   all_to_all_single_autograd ──► NCCL stream: [==== combine a2a ====]
                                  compute stream: [== shared_experts(x) ==]
                                                          ↓
   deterministic_scatter_add(src=ACT) ──► trigger_wait ──► 等 a2a 完成,窗口闭合
```

### 5.2 DeepEP 路径的显式 `sync_combine()`

DeepEP 不用 ACT,而是 DeepEP 库自己的 event 机制。`DeepEPTokenDispatcher.combine`:`combine_tokens` 内部 `async_finish=True` 把 event 存进进程级全局变量、**不立即同步**;`shared_experts(x)` 重叠;再**显式**调 `sync_combine()` —— 它在当前 CUDA stream 上插入对 combine 通信 stream 上 event 的等待(等价 `cudaStreamWaitEvent`)。

> 两套同步哲学:标准路径靠 PyTorch ACT 的 `__torch_dispatch__` **自动**插 `wait_tensor`;DeepEP 路径靠**手工** CUDA event(因为 DeepEP 输出是普通 tensor,不带 ACT 的自动 wait 语义)。

---

## 6. DeepEP / HybridEP:专用内核优化

标准 `AllToAllTokenDispatcher` 一次 dispatch 要做:histc → argsort → index_select 重排 → 计数 a2a + wait + **2 次 D2H 同步** → token a2a → `_permute`(再一次 index_select)。低效点:多次独立 kernel、D2H 同步打断流水线、通用 NCCL all-to-all 不感知 MoE 结构。

**DeepEP**(`torchtitan/distributed/deepep/deepep.py`,H100/NVLink Switch):
- `get_dispatch_layout` 在 GPU 上一次性算出 token 路由布局,**无需 Python 层 histc/sum/D2H**。
- **NVLink + RDMA 两级感知**:节点内走 NVLink P2P、节点间走 RDMA。
- 进程级常驻 `Buffer` 复用,省去每步通信缓冲分配。

**HybridEP**(`hybridep.py`,GB200/NVLink72):
- **fused dispatch-with-permute**:一个 kernel 内同时做 all-to-all 通信 + expert-major permute(标准路径的"a2a 然后 `_permute`"两步融合)。
- **non-blocking 模式**:用 `non_blocking_capacity_factor` 预先估算 token 容量,**完全跳过 D2H 同步**;代价是 `cf<1.0` 时超容量 token 被静默丢弃。

一句话:标准 all-to-all 是"通用 NCCL collective + Python 层手工算 split + 多次 D2H 同步";DeepEP/HybridEP 是"MoE 专用、NVLink/RDMA 两级感知、dispatch+permute 融合、可选 CPU-free 的常驻 buffer 内核"。

---

## 7. EP 与 FSDP 组合:`edp_mesh` + 显式预取

### 7.1 为何专家参数要单独的 `edp_mesh` 做 FSDP

非专家参数(attention、norm、router gate、shared experts)在 `dp_mesh` 上做 FSDP。但专家参数已被 `ExpertParallel` 沿 `Shard(0)` 切到 `ep` mesh——它的 FSDP 必须在"扣掉 EP 后剩余的 DP 维"上做。

[[10_torchtitan_parallel_dims_analysis]] 给出这个维度:`efsdp = dp_shard × cp × tp / ep`。`sparse_mesh = [pp, dp_replicate, efsdp, ep]` 专门给专家参数。`tp` 进了 `efsdp`——因为 EP 开启时 TP 不切专家权重,TP 维对专家就是额外可用的 FSDP 维。

### 7.2 `shard_placement_fn` 分流

`apply_fsdp`(`torchtitan/models/llama4/parallelize.py:143`)对每个 MoE block 调 `fully_shard` 时,用 `shard_placement_fn` 给同一个 block 内的不同参数派发不同 mesh:

```python
def _shard_placement_fn(param, ...):
    if param in expert_params:
        return ShardPlacementResult(placement=expert_placement, mesh_info=edp_mesh_info)  # 专家 → edp_mesh
    else:
        return ShardPlacementResult(placement=Shard(0), mesh_info=dp_mesh_info)           # 其它 → dp_mesh
```

即:**同一个 `fully_shard(transformer_block)` 调用里,专家参数在 `edp_mesh` 上分片、其余参数在 `dp_mesh` 上分片**(`expert_placement` 按 FSDP 度数与专家数的关系选 `Shard(0)` 或 `Shard(1)` 避免 padding)。详见 [[11_torchtitan_fsdp_analysis]] §8.4。

### 7.3 为何 EP 开启要显式预取

`apply_fsdp` 末尾(`llama4/parallelize.py:319`):`if ep_degree == 1: return`,即只有 EP>1 才做下面的显式预取。

**根本原因**:FSDP2 的隐式预取依赖 **CPU 跑在 GPU 前面**——CPU 提前发射下一层的 all-gather(见 [[11_torchtitan_fsdp_analysis]] §3.5)。但 EP 的 dispatch 里有**多次 D2H 同步**(§3.1 ② 的 `input_splits/output_splits`),D2H 同步会**阻塞 CPU 线程**——CPU 卡住就没法及时发射下一层 FSDP 的 all-gather,隐式预取被打断,通信暴露。

**解决**:用 `set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch` **手工显式预取**,把下一层 all-gather 的发起**提前到 D2H 同步之前**。

```python
# llama4/parallelize.py:324(简化)
model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])
for block, next_block in zip(transformer_blocks, next_transformer_blocks):
    block.set_modules_to_forward_prefetch([next_block])
# backward 对称:lm_head → 末层 → ... → 首层 → tok_embeddings
```

> 这是"为什么 EP 的 FSDP 代码比普通 FSDP 复杂得多"的根因。

---

## 8. 反向传播:all-to-all 的 backward

### 8.1 标准 all-to-all 的 autograd

`all_to_all_single_autograd` 的反向语义很对称:

> **all-to-all 的 backward 仍是一个 all-to-all,把 `input_splits` 和 `output_splits` 互换**,再做一次 `wait_tensor`。

直觉:前向 dispatch 把 token 从 rank A 送到 rank B,反向就要把 B 上算出的梯度送回 A。所以 a2a 的反向是"split 参数颠倒的 a2a"。

而 `combine` 前向本身就用了与 `dispatch` **对调**的 split 参数(combine = dispatch 的逆)。于是:**combine 的 backward 自动等价于 dispatch 的 forward,dispatch 的 backward 等价于 combine 的 forward**——整个 MoE 的反向自动闭合,无需手写(标准路径)。

### 8.2 反向中的隐式同步

反向梯度也是 `AsyncCollectiveTensor`,在 autograd 引擎需要 tangent 时(`__coerce_same_metadata_as_tangent__`)自动 `trigger_wait`。所以反向 a2a 的同步同样是"梯度首次被非 view op 使用时"自动触发,机制与前向一致。

### 8.3 DeepEP 路径

DeepEP **手写反向**:`dispatch` 的反向 = "对梯度做 combine",`combine` 的反向 = "对梯度做 dispatch",通过 `torch.library.register_autograd` 注册。这与 8.1 的"a2a 反向是 split 颠倒的 a2a"是同一对称性,只是用 DeepEP 库的 dispatch/combine kernel 对实现。

### 8.4 梯度 placement 的全局正确性

`MoE.forward` 把输入 DTensor `to_local(grad_placements=(Partial(),))`——无论 TP-only / TP+EP 各种配置,`grad(x)` 在 tp_mesh 上都是 `Partial`。这保证梯度归约(`Partial → Shard(1)` 的 reduce-scatter)在 MoE 边界**只做一次**。

---

## 9. 完整流程图

```
═══ 建模期 ═══
apply_moe_ep_tp()                            llama4/parallelize.py:367
  ├─ ep_mesh 存在 → ExpertParallel():w1/w2/w3 distribute_tensor(Shard(0))
  │                  + token_dispatcher.ep_mesh = ep_mesh
  └─ ep_mesh 为 None → TensorParallel():w1/w3 Shard(1)、w2 Shard(2)
apply_fsdp()  EP>1 时:
  ├─ shard_placement_fn:专家参数→edp_mesh,其余→dp_mesh
  └─ set_modules_to_forward/backward_prefetch:显式预取(绕过 D2H 同步阻塞)

═══ 前向期(一个 MoE 层) ═══
MoE.forward → router 算 top-k 专家分配
   │ GroupedExperts.forward:dispatch → _experts_forward → combine
   ▼
dispatch: ①histc+argsort 本地重排 ②计数 a2a(+D2H 同步算 splits)
          ③token a2a_autograd(返回 ACT)④_permute rank→expert-major
   │  (ACT 在 _permute 的 index_select 处 trigger_wait)
   ▼
_experts_forward: torch._grouped_mm 分组 GEMM(本地专家)
   │
   ▼
combine: _unpermute → token a2a_autograd(splits 互换,返回 ACT)
   │     ┌─ NCCL stream:  [==== combine a2a ====]      ┐ 通信掩盖
   │     └─ compute stream:[== shared_experts(x) ==]    ┘
   │  scatter_add(src=ACT) → trigger_wait → 等 a2a,窗口闭合
   ▼ out = shared + routed

═══ 反向期 ═══
a2a 的 backward = split 参数互换的 a2a;combine 反向 ≡ dispatch 正向
→ 整个 MoE 反向自动闭合(标准路径);DeepEP 路径手写对称反向
```

---

## 10. 小结

- **专家切分**:`ExpertParallel._partition_fn` 用 `Shard(0)` 把 `(num_experts, ...)` 的 w1/w2/w3 沿专家维切——每卡持有 `num_experts/ep_size` 个**完整**专家。对偶:EP 切专家维(通信在 token 上),TP 切矩阵维(通信在激活上)。
- **通信原语**:token **all-to-all** dispatch/combine。因为是**变长**通信,dispatch 先做一次小 all-to-all 交换计数算出 `input/output_splits`(含 D2H 同步),再做 token all-to-all,最后 `_permute` 把 rank-major 重排成 expert-major。
- **异步实现**:`all_to_all_single_autograd` 返回 `AsyncCollectiveTensor`。ACT 的 `__torch_dispatch__` 对 **view op 不 wait、非 view op 才 `trigger_wait`**——这就是"延迟 wait"。
- **通信掩盖**:`combine()` 把 `shared_experts(x)` 塞进"a2a enqueue 到结果首次使用"的窗口——NCCL stream 跑 combine 通信 ‖ compute stream 跑 shared experts,`scatter_add` 读 ACT 时自动同步、窗口闭合。
- **DeepEP/HybridEP**:MoE 专用内核,NVLink/RDMA 两级感知、dispatch+permute 融合、可选 CPU-free non-blocking;同步靠手工 `sync_combine()` 的 CUDA event。
- **EP 与 FSDP**:专家参数走单独的 `edp_mesh`(`efsdp = dp_shard·cp·tp/ep`),`shard_placement_fn` 在同一 `fully_shard` 调用里给专家/非专家参数分流不同 mesh;EP 的 D2H 同步会打断 FSDP 隐式预取,故必须 `set_modules_to_*_prefetch` 显式预取。
- **反向**:all-to-all 的 backward = split 参数互换的 all-to-all;combine 正向是 dispatch 的逆,故 MoE 反向自动闭合。

## Related Pages

- [[torchtitan/index]] · [[10_torchtitan_parallel_dims_analysis]] —— 知识地图与并行基座
- [[11_torchtitan_fsdp_analysis]] —— MoE 专家的 `edp_mesh` FSDP 与显式预取
- [[14_torchtitan_pp_analysis]] —— 相邻并行维度
- [[14_megatron_ep_analysis]] —— Megatron-LM 专家并行(AllGather/AllToAll/Flex 三种 token dispatcher)
- [[14_megatron_ep_analysis]] —— MoE 零冗余通信、AlltoAll token dispatch
- [[21_async_collective_tensor_deepdive]] —— `AsyncCollectiveTensor` 源码追踪
- [[31_comm_compute_fusion_guide]] —— 通算融合:WaveEP、DeepEP、各维度重叠
- [[30_comm_compute_overlap_analysis]] —— 跨框架(Megatron/torchtitan/MindSpeed)通算掩盖对比矩阵,本页是其 torchtitan EP 掩盖机制来源(§4-5)
