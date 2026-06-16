# 上下文并行 CP —— 机制级深度分析

> **代码基准**:torchtitan `main` @ `cf3c4312` · PyTorch CP 内核(版本差异见下方"版本说明")
> **最后更新**:2026-05-22 · **系列**:torchtitan 多维并行源码级分析(见 [[torchtitan/index]])
>
> 本文按统一结构回答:**序列怎么切?切完怎么通信?哪些通信能掩盖?异步怎么实现?** 重点是 Ring Attention。
>
> 行号约定:torchtitan 以 `torchtitan/` 为根;PyTorch CP 实现以 `[pt]` 前缀。

> **⚠️ 版本说明(重要)**:本机安装的是 PyTorch `2.9.1+cpu`,其 CP 实现是单文件 `torch/distributed/tensor/experimental/_attention.py`。torchtitan 当前版本(`context_parallel.py:15-23`)导入的 `_context_parallel_shard`、`_HeadTailLoadBalancer`、`_PTRRLoadBalancer`、`flex_cp_allgather` 等符号来自更新的 `torch/distributed/tensor/experimental/_context_parallel/` **包**(2.9 之后的重构)。本文机制以新版包为准;旧版单文件的 ring 核心逻辑(`_templated_ring_attention` 等)与新版几乎逐行相同,差异只在切分/负载均衡的封装层。`[pt]` 引用对新版包标 `_context_parallel/...`。

---

## 1. 功能范围与定位

**CP(上下文并行)** 把**序列维度**切到多张卡上,让超长序列(128k+ token)的 attention 能放下——attention 的显存/计算随序列长度增长,长序列单卡放不下。CP 切的是 seq 维,与 TP(切 feature 维)、DP(切 batch 维)正交。

CP 在 torchtitan 里只做**两件事**(`torchtitan/distributed/context_parallel.py`):

1. **数据切分**(每个训练 step,在 trainer 里):`prepare_context_parallel_input` → `cp_shard` → PyTorch `_context_parallel_shard`。把 `inputs / labels / positions / attention_masks` 沿序列维切到 CP ranks。
2. **forward 包装**(建模时,在 parallelize 里):`apply_cp_to_forward`。根据 inner attention 类型给 `attention.inner_attention.forward` 套一层 CP wrapper。

调用点:`trainer.py:618` 切数据,`models/llama3/parallelize.py:69` 包 forward。`context_parallel.py:46-50` 的注释明说这是**临时方案**,将来要改成基于 `ShardingConfig` 的声明式 CP。`VarlenAttention` 暂不支持 CP(`context_parallel.py:97` `raise NotImplementedError`)。

---

## 2. 数据切分:`_context_parallel_shard` 怎么把序列切到 CP ranks

### 2.1 torchtitan 入口 `cp_shard`

`cp_shard`(`torchtitan/distributed/context_parallel.py:155`):

```python
seq_len = inputs[0].size(input_seq_dim)         # input_seq_dim 默认 1,即 [B, seq_len]
load_balancer = _HeadTailLoadBalancer(...) | _PTRRLoadBalancer(...) | None   # 见 §3
inputs = _context_parallel_shard(mesh=cp_mesh, buffers=inputs,
                                 seq_dims=(1,1,1), load_balancer=load_balancer)
# attention_masks(BlockMask)单独切,seq_dim 用 2(BlockMask 形状 [B,H,Q,KV],只能切 Q 维)
masks = _context_parallel_shard(mesh=cp_mesh, buffers=masks, seq_dims=(2,)*len(masks), ...)
```

### 2.2 PyTorch `_context_parallel_shard` 内部

核心 `_context_parallel_buffers`:

```python
load_balance_indices = load_balancer._generate_indices() if load_balancer else None
for buffer, seq_dim in zip(buffers, buffer_seq_dims):
    if load_balance_indices is not None:
        buffer = torch.gather(buffer, dim=seq_dim, index=indices)   # ① 用索引重排序列
    sharded_buffer = distribute_tensor(buffer, mesh, [Shard(seq_dim)], src_data_rank=None).to_local()  # ②
```

**每个 rank 持有哪一段?** 由 `distribute_tensor(buffer, mesh, [Shard(seq_dim)])` 决定——把张量沿 `seq_dim` 均匀切成 `cp_size` 段,rank `r` 拿第 `r` 段:

- **不开负载均衡**:rank `r` 直接拿 `seq[r·S/cp : (r+1)·S/cp]`——序列的第 r 个**连续**段。
- **开负载均衡**:先 `torch.gather` 按重排索引把序列**打乱**(头尾交错),再均匀切。rank `r` 物理上拿到的是"原序列的某个头段 + 某个尾段"组合(见 §3)。

> `_context_parallel_shard` 本身**不做通信**(`src_data_rank=None` 时 `distribute_tensor` 等价于本地 slice)。真正的跨 rank 通信发生在 attention 计算阶段(§5)。

---

## 3. 负载均衡:因果掩码的三角形问题

### 3.1 为什么需要负载均衡

因果掩码下 `mask[i,j]=1 ⟺ q_idx ≥ kv_idx`,计算量正比于矩阵里 1 的个数。`seq_len=8, cp=2` 朴素均分:

```
            KV_index
   [1,0,0,0,0,0,0,0]
   [1,1,0,0,0,0,0,0]
   [1,1,1,0,0,0,0,0]   rank0 → 1+2+3+4 = 10 次计算
   [1,1,1,1,0,0,0,0]
 ──────────────────────
   [1,1,1,1,1,0,0,0]
   [1,1,1,1,1,1,0,0]   rank1 → 5+6+7+8 = 26 次计算
   [1,1,1,1,1,1,1,0]
   [1,1,1,1,1,1,1,1]
```

**后段 rank 的 Q 行更靠下,要 attend 的 KV 更多**——rank1 工作量是 rank0 的 2.6 倍。在 ring attention 里所有 rank 每步同步,最慢的 rank 成为 straggler 拖垮整组。

### 3.2 `_HeadTailLoadBalancer`(SDPA 路径默认)

算法核心是**头尾配对**:把序列切成 `2·cp` 个等长 chunk,**rank `r` 领取 chunk `r`(头)和 chunk `2cp-1-r`(尾)**。

`seq_len=8, cp=2`:4 个 chunk(每个长 2),`head_idx=[0,1]`、`tail_idx=[3,2]`,重排索引 `[0,7,1,6,2,5,3,4]`。重排后:

```
   rank0:chunk0(行0,1)+chunk3(行6,7) → 1+8+2+7 = 18
   rank1:chunk1(行2,3)+chunk2(行4,5) → 3+6+4+5 = 18    ← 完全均衡
```

直觉:第 r 个头 chunk(轻)配第 r 个尾 chunk(重),头重之和对每个 rank 恒定。

> **这就是 `seq_len` 必须被 `2·cp` 整除的原因**([[torchtitan_parallel_dims_analysis]] 的 `seq_len_divisor`):头尾配对要把序列切成 `2·cp` 个等长 chunk;且每个 rank 拿"1 头 + 1 尾"= 2 个 chunk,在 ring attention 里被当作 2 个 round-robin 子块处理。

### 3.3 `_PTRRLoadBalancer`(FlexAttention 路径)

PTRR = Processing-Time based Round-Robin。HeadTail 假定掩码是标准因果三角(纯几何规则);PTRR 面对**任意稀疏 `BlockMask`**(滑动窗口、文档掩码等),无法用固定几何,必须**真去数每个 Q-block 的实际计算量**:

1. 从 `BlockMask` 取 `non_sparse_kv_num_blocks`(每个 Q-block 实际要算的 KV-block 数)作为"处理时间"。
2. `ptrr_scheduling`:按处理时间降序排,每 `cp_size` 个一组做"蛇形(serpentine)正逆交替"分配——把最大的和次小的配在一起摊平,经典 LPT 多机调度近似。

PTRR 返回 `(B, seq_len)` 索引(每个样本 BlockMask 不同,重排逐样本不同),HeadTail 返回 `(1, seq_len)`(纯因果与样本内容无关,全 batch 共用)。

> **负载均衡只改"数据怎么切"**——切分前一次 `gather` 重排,完全不碰 ring 通信本身。它与 ring 算法、与通信掩盖都正交。

---

## 4. 两条 attention 路径

`apply_cp_to_forward`(`context_parallel.py:35`)按 inner attention 类型分两条路径。

### 4.1 SDPA 路径:Ring Attention

```python
# context_parallel.py:74
elif isinstance(first, ScaledDotProductAttention):
    _enable_context_parallel_dispatcher()        # 装上 DTensor 的 CP 派发器
    def cp_forward(q, k, v, **kwargs):
        q = DTensor.from_local(q, mesh, [Shard(2)], run_check=False)   # 包成 CP-sharded DTensor
        k = DTensor.from_local(k, mesh, [Shard(2)], run_check=False)
        v = DTensor.from_local(v, mesh, [Shard(2)], run_check=False)
        return orig_fn(q, k, v, **kwargs).to_local()
```

`ScaledDotProductAttention.forward` 先把布局 transpose 成 `[B,H,S,D]`,所以序列维是 **dim 2** → `Shard(2)`。`_enable_context_parallel_dispatcher` 把 6 个 aten SDPA op(flash/efficient/cudnn 的 fwd+bwd)映射到 `_sdpa_handler`,后者把 `Shard(2)` 的 SDPA 路由到 `_templated_ring_attention`。

### 4.2 FlexAttention 路径:all-gather K/V

```python
# context_parallel.py:57
if isinstance(first, FlexAttention):
    def cp_forward(q, k, v, **kwargs):
        global_k, global_v = flex_cp_allgather(k, v, 1, pg_name)   # 直接 all-gather 全量 K/V
        return orig_fn(q, global_k, global_v, **kwargs)            # Q 是本地分片,K/V 是全局
```

`seq_dim` 传 `1`:`apply_cp_to_forward` 包的是 `FlexAttention.forward` 外层,此时 q/k/v 还是 `[B,S,H,D]` 布局。`flex_cp_allgather` 是注册过的 custom op(为了支持 `torch.compile` 与 autograd)。

### 4.3 两条路径的取舍

| 维度 | FlexAttention(all-gather) | SDPA(ring) |
|---|---|---|
| K/V 通信 | **一次** all-gather 拿全局 K/V | `cp_size` 步,K/V 环形轮转 |
| K/V 显存 | 每 rank 持有**全量** K/V `O(S)` —— 抵消了 CP 省显存初衷 | 每 rank 任意时刻只持 1~2 份分片 `O(S/cp)` |
| 通信/计算重叠 | 几乎没有(就一次 all-gather) | ring 主循环天然重叠 |
| 负载均衡 | `_PTRRLoadBalancer`(数 BlockMask 实际计算量) | `_HeadTailLoadBalancer`(因果三角几何配对) |
| 掩码 | 任意稀疏 `BlockMask` | 只支持 `is_causal` 布尔 |

**一句话**:Flex 路径用"K/V 全量复制 + 一次 all-gather"换实现简单和任意稀疏掩码,代价是 K/V 显存退化成 `O(S)`;SDPA ring 路径用"K/V 分片 + 多步轮转 + 在线 softmax"保住 `O(S/cp)` 显存并天然重叠通信。**CP 省显存的核心收益在 SDPA ring 路径上才完整。** 下面 §5-§7 主讲 ring 路径。

---

## 5. 通信原语:Ring Attention 的 K/V 环形轮转

### 5.1 `_templated_ring_attention` 主循环

`[pt] _context_parallel/_attention.py:_templated_ring_attention`:

```python
rank = dist.get_rank(group);  size = dist.get_world_size(group)
sdpa_merger = _SDPAMerger(convert_to_f32=True, seq_dim=seq_dim)
rotater = _create_rotater(group, 2)                       # 默认 _AllGatherRotater

for i in range(size):                                     # size = cp_world_size 步
    if i > 0:
        next_kv = rotater.next_buffer()                   # (A) 取上一步发起的传输结果
        key, value = 从 next_kv 切出
    if i < size - 1:
        next_kv = torch.cat([key.flatten(), value.flatten()])
        next_kv = rotater.exchange_buffers(next_kv)       # (B) 发起下一步要用的 K/V 传输(异步)
    is_causal_behavior = _is_causal_behavior(rank, size, i, is_causal)
    if is_causal_behavior == _CausalBehavior.SKIP:
        continue
    ... 选 q/k/v 子块 ...
    out, lse, *rest = op(q, k, v, is_causal=..., **kwargs) # (C) 当前步 SDPA 计算
    sdpa_merger.step(out, lse, partial)                   # (D) 在线 softmax 合并
return *sdpa_merger.results(), *rest
```

**核心思想**:Q 不动(每个 rank 永远只算自己那段 Q),K/V 在 CP 组内**环形传递**。第 `i` 步 rank `r` 手里的 K/V 来自 rank `(r-i) mod size`。跑满 `size` 步后,每个 rank 的 Q 都和**全部** K/V 算过一遍,在线 softmax 把 `size` 次局部结果合并成最终输出。

### 5.2 K/V 环形轮转图(cp=4)

```
初始(step 0):每个 rank 用自己的 K/V
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│rank0 │  │rank1 │  │rank2 │  │rank3 │
│Q0 KV0│  │Q1 KV1│  │Q2 KV2│  │Q3 KV3│
└──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘
   │ KV0     │ KV1     │ KV2     │ KV3      每步把当前 KV 发给右邻
   └───►─────┴───►─────┴───►─────┴──►─(绕回 rank0)

step 1:KV 环转一格,rank_r 收到 KV[(r-1) mod 4]
   rank0: Q0×KV3   rank1: Q1×KV0   rank2: Q2×KV1   rank3: Q3×KV2
step 2:rank0: Q0×KV2  rank1: Q1×KV3  rank2: Q2×KV0  rank3: Q3×KV1
step 3:rank0: Q0×KV1  rank1: Q1×KV2  rank2: Q2×KV3  rank3: Q3×KV0

N=4 步后每个 Q_r 都与 KV0..KV3 全部相乘过;每步的 (out,lse) 由 _SDPAMerger 在线合并。
```

### 5.3 在线 softmax 合并 `_SDPAMerger`

每步 SDPA 产出局部 `(out, lse)`(lse = log-sum-exp),`_SDPAMerger.step` 用数值稳定公式增量合并:

```python
out = out - sigmoid(block_lse - lse) * (out - block_out)
lse = lse - logsigmoid(lse - block_lse)
```

`convert_to_f32=True` 全程 fp32 累加避免误差。这就是 ring attention 不需要一次性持有完整 K/V 的关键——用在线 softmax 把"分 `size` 步、每步一段 K/V"的局部结果正确合并成"全量 K/V"的结果。

### 5.4 两种轮转器(rotater)

`_create_rotater` 按 `rotate_method` 选(torchtitan config `context_parallel_rotate_method`,默认 `"allgather"`):

| | `_AllToAllRotater`(`"alltoall"`) | `_AllGatherRotater`(`"allgather"`,默认) |
|---|---|---|
| 原语 | 每步一次 `permute_tensor`(P2P 式置换,送右邻) | 第一步一次 `all_gather_tensor` 收齐全部 K/V |
| 每步开销 | 传 1 份 K/V(共 `size-1` 次) | 一次性传 `size` 份,后续 `next_buffer` 只是本地 `chunk()` |
| 重叠 | 真正逐步 P2P 重叠 | 仅第一次 all-gather 与第 0 步 SDPA 重叠 |

### 5.5 `_is_causal_behavior`:每步选掩码

第 `i` 步该用什么掩码:`i==0` 用标准因果(本地 Q×本地 KV);`i>0` 且传来的 KV 全在 Q 之前 → 不掩码算满;`i>0` 不开负载均衡且 KV 全在 Q 之后 → `SKIP`(因果下全被 mask)。**开了 HeadTail 后永远不会 SKIP**(每个 rank 的本地块含头+尾,任何一步都有非空计算)——这就是负载均衡的运行期体现。

---

## 6. 通信掩盖:下一步 P2P 与当前步 SDPA 重叠

这是 ring attention 性能的命门。看 §5.1 循环体的**指令顺序**:

```
for i in range(size):
    (A) next_buffer()        ← 收割上一步发起的传输(这里才 wait)
    (B) exchange_buffers()   ← 发起"下一步要用的 K/V"传输 —— 异步,不阻塞
    (C) op(q,k,v)            ← 当前步 SDPA 计算
    (D) sdpa_merger.step()   ← 合并
```

关键:**(B) 在 (C) 之前发起,且是异步的**。`exchange_buffers` 返回不等待,所以 (B) 发起的集合通信与 (C) 的 SDPA 计算**在硬件上并行**。到下一轮的 (A) `next_buffer()` 才真正 `wait()`。

即:**第 `i` 步发起第 `i+1` 步要用的 K/V 传输,然后立刻算第 `i` 步的 attention,传输与计算重叠;第 `i+1` 步开头才收割传输结果。** 这一步 SDPA 的耗时把上一步发起的 K/V 传输延迟"藏"了进去。

```
            step i              step i+1            step i+2
计算流:    [SDPA(i)        ]   [SDPA(i+1)      ]   [SDPA(i+2)      ]
通信流:    [exchange KV→i+1]   [exchange KV→i+2]   [exchange KV→i+3]
                ↑ 第 i 步发起的传输,与第 i 步 SDPA 并行,第 i+1 步开头收割
```

---

## 7. 异步实现:靠 functional collective,不是手写 stream

> **关键澄清**:torchtitan / PyTorch 的 CP **没有手写 CUDA stream**。重叠完全靠 `torch.distributed._functional_collectives` 的异步语义——容易被误以为是"显式多 stream"。

机制:

- `_AllToAllRotater.exchange_buffers` 调 `ft_c.permute_tensor(...)`(`ft_c` = functional collectives),底层走 `torch.ops._c10d_functional.*`,**返回一个 `AsyncCollectiveTensor`(ACT)**。通信在 c10d 的通信 stream 上发起,**不阻塞当前计算 stream**。
- `next_buffer` 里 `_maybe_wait(tensor)`:若是 ACT 就 `.wait()`。`ACT.wait()` 触发 `wait_tensor` op,在通信 stream 与计算 stream 之间插入跨 stream 同步。

所以重叠的本质:**(B) 在通信 stream 上发起 collective 并立即返回 ACT;(C) 的 SDPA kernel 同时在计算 stream 上跑;下一轮 (A) 的 `wait()` 才在两条 stream 间同步**。这与 [[torchtitan_fsdp_analysis]] FSDP 的"手写多 stream"是不同的实现风格——CP 复用了 functional collective 内建的通信 stream + ACT 延迟 wait 机制。

`_AllGatherRotater` 同理:第一步 `all_gather_tensor` 返回 ACT,只有第 0 步那次 all-gather 能和第 0 步 SDPA 重叠。

---

## 8. 反向传播:CP 在 backward 的通信

### 8.1 SDPA ring 的反向 `_templated_ring_attention_backward`

它**同时维护两个环**:

```python
kv_rotater  = _create_rotater(group, 2)                        # K/V 前向轮转(同 fwd)
dkv_rotater = _create_rotater(group, 2, method=ALL_TO_ALL)     # K/V 梯度轮转,强制 all-to-all
grad_query/grad_key/grad_value = zeros(fp32)
```

每步 `i` 做两件事:

1. **K/V 本身环形轮转**(同 forward):重演 forward 时那对 `(Q_local, KV_from_rank_(r-i))` 才能算梯度。
2. **K/V 梯度的环形规约**:某段 K/V 在 forward 里被**每个** rank 的 Q attend 过,其梯度是所有 rank 局部贡献之和——DTensor 意义上的 `Partial` 规约。ring backward 把 dK/dV 跟着 K/V 一起绕环,每经过一个 rank 就把该 rank 的局部贡献累加进去。绕满 `size` 步后得到完整的 `grad_key/grad_value`。

`grad_query` 不需要环规约——Q 是每个 rank 独占的分片,直接 `+=` 累加。

> **为什么 `dkv_rotater` 强制 all-to-all 而不能用 all-gather**:梯度必须**逐 rank 顺序累加**(每步 = 上一 rank 的部分和 + 本 rank 贡献),all-gather 一次性收齐就没法做这个增量累加。

### 8.2 FlexAttention all-gather 路径的反向

`flex_cp_allgather` custom op 注册了 autograd:forward 是 `all_gather`(本地 K/V 分片 → 全局 K/V),所以 backward 必须是 **`reduce_scatter`**(全局 K/V 梯度 → 规约求和 + 散回各 rank 本地分片)。这正是 all-gather 的标准转置算子,也是 K/V 梯度 `Partial → Shard(seq)` 规约的体现。

---

## 9. 完整流程图

```
═══ 每个训练 step:数据切分 ═══
trainer.py:618  prepare_context_parallel_input → cp_shard
   │ load_balancer 生成重排索引(HeadTail 头尾配对 / PTRR 处理时间均衡)
   │ torch.gather 按索引重排序列  →  distribute_tensor(Shard(seq)) 均匀切
   ▼ 每个 CP rank 拿到序列的一段(开 LB 时是"头段+尾段"组合)

═══ 建模期:forward 包装 ═══
apply_cp_to_forward(inner_attention 列表, cp_mesh)
   ├─ FlexAttention → cp_forward 里 flex_cp_allgather 全量 K/V
   └─ SDPA → _enable_context_parallel_dispatcher,q/k/v 包成 Shard(2) DTensor
              → _sdpa_handler 路由到 _templated_ring_attention

═══ 前向期:Ring Attention(SDPA 路径) ═══
for i in range(cp_size):
   next_buffer()            ← 收割上一步的 K/V 传输(wait)
   exchange_buffers()       ← 异步发起下一步 K/V 传输 ┐
   op(q, k, v) SDPA 计算    ← 与上面的传输并行          ┘ 通信掩盖
   sdpa_merger.step()       ← 在线 softmax 合并
跑满 cp_size 步 → 每个 Q 段见过全部 K/V

═══ 反向期 ═══
_templated_ring_attention_backward:两个环并行
   kv_rotater  轮转 K/V(重演 forward)
   dkv_rotater 轮转 + 累加 K/V 梯度(Partial 规约,强制 all-to-all)
```

---

## 10. 小结

- **序列切分**:`cp_shard` → `_context_parallel_shard`,用 `distribute_tensor(Shard(seq_dim))` 把序列均匀切到 CP ranks。开负载均衡时先 `torch.gather` 按索引重排再切。本身不通信。
- **负载均衡**:解决因果掩码三角形导致的后段 rank 计算量大的问题。`_HeadTailLoadBalancer` 用"头尾配对"(几何规则,故 `seq_len` 须被 `2·cp` 整除);`_PTRRLoadBalancer` 数 `BlockMask` 实际计算量做处理时间均衡。负载均衡只改数据切法,与 ring 通信正交。
- **通信原语**:SDPA 路径用 **Ring Attention**——Q 不动,K/V 在 CP 组内环形轮转 `cp_size` 步,在线 softmax 合并各步局部结果。两种轮转器:all-to-all(逐步 P2P)/ all-gather(一次性)。
- **通信掩盖**:ring 主循环里"**先异步发起下一步 K/V 传输,再算当前步 SDPA**",传输与计算重叠,下一步开头才收割。
- **异步实现**:**不是手写 CUDA stream**——靠 `functional_collectives` 返回 `AsyncCollectiveTensor`,通信在 c10d 通信 stream 上发起,`wait()` 延迟到下一步。
- **两条路径**:FlexAttention all-gather(K/V 全量复制,`O(S)` 显存,支持任意稀疏掩码)vs SDPA ring(K/V 分片,`O(S/cp)` 显存,只支持因果)。CP 省显存的收益在 ring 路径上才完整。
- **反向**:ring backward 同时维护"K/V 轮转环"和"K/V 梯度规约环"(后者强制 all-to-all 做增量累加)。

## Related Pages

- [[torchtitan/index]] · [[torchtitan_parallel_dims_analysis]] —— 知识地图与并行基座
- [[torchtitan_tp_analysis]] · [[torchtitan_pp_analysis]] —— 相邻并行维度
- [[megatron_cp_analysis]] —— Megatron-LM 上下文并行(4 种 `cp_comm_type` + 因果 zigzag 负载均衡)
- [[deepseek_v4_context_parallel_analysis]] —— DeepSeek-V4 CP 实现、Native/TE CP、Dynamic CP
