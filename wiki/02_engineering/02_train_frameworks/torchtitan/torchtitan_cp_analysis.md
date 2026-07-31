# 上下文并行 CP —— 机制级深度分析

> **代码基准**:torchtitan `main` @ `cf3c4312` · PyTorch CP 内核(版本差异见下方"版本说明")
> **最后更新**:2026-07-31 · **系列**:torchtitan 多维并行源码级分析(见 [[torchtitan/index]])
>
> **划界声明**:CP 通用机制(为什么要切序列、折叠/头尾负载均衡的数学证明、因果块裁剪、Ring 主循环 + online-softmax、通信掩盖原理、通信量代数)已归一到 [[../../../01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis|20_ring_attention_and_context_parallel_analysis]]——事实上,本文的 Ring 主循环伪代码、负载均衡量化算例、通信掩盖时序图正是该理论页对应章节的骨架来源。**本页只保留 torchtitan/PyTorch CP 的框架实现差异**:trainer/parallelize 接入点、SDPA-ring 与 FlexAttention-allgather 两条路径的取舍(torchtitan 独有的双路径架构)、DTensor dispatcher 接线、以及"不手写 CUDA stream、靠 functional collectives 实现异步"这一 PyTorch 特有的工程选择。
>
> 行号约定:torchtitan 以 `torchtitan/` 为根;PyTorch CP 实现以 `[pt]` 前缀。

> **⚠️ 版本说明(重要)**:本机安装的是 PyTorch `2.9.1+cpu`,其 CP 实现是单文件 `torch/distributed/tensor/experimental/_attention.py`。torchtitan 当前版本(`context_parallel.py:15-23`)导入的 `_context_parallel_shard`、`_HeadTailLoadBalancer`、`_PTRRLoadBalancer`、`flex_cp_allgather` 等符号来自更新的 `torch/distributed/tensor/experimental/_context_parallel/` **包**(2.9 之后的重构)。本文机制以新版包为准;旧版单文件的 ring 核心逻辑(`_templated_ring_attention` 等)与新版几乎逐行相同,差异只在切分/负载均衡的封装层。`[pt]` 引用对新版包标 `_context_parallel/...`。

---

## 1. 功能范围与定位

**CP(上下文并行)** 把**序列维度**切到多张卡上,让超长序列(128k+ token)的 attention 能放下——通用动机见理论页 §1。

CP 在 torchtitan 里只做**两件事**(`torchtitan/distributed/context_parallel.py`):

1. **数据切分**(每个训练 step,在 trainer 里):`prepare_context_parallel_input` → `cp_shard` → PyTorch `_context_parallel_shard`。把 `inputs / labels / positions / attention_masks` 沿序列维切到 CP ranks。
2. **forward 包装**(建模时,在 parallelize 里):`apply_cp_to_forward`。根据 inner attention 类型给 `attention.inner_attention.forward` 套一层 CP wrapper。

调用点:`trainer.py:618` 切数据,`models/llama3/parallelize.py:69` 包 forward。`context_parallel.py:46-50` 的注释明说这是**临时方案**,将来要改成基于 `ShardingConfig` 的声明式 CP。`VarlenAttention` 暂不支持 CP(`context_parallel.py:97` `raise NotImplementedError`)。

---

## 2. 数据切分入口:`cp_shard`

> 序列如何均匀切分(`distribute_tensor(Shard(seq_dim))`,不产生通信)是通用机制,见理论页 §3.1。本节只记 torchtitan 的调用链与它对 `BlockMask` 的特殊处理。

`cp_shard`(`torchtitan/distributed/context_parallel.py:155`):

```python
seq_len = inputs[0].size(input_seq_dim)         # input_seq_dim 默认 1,即 [B, seq_len]
load_balancer = _HeadTailLoadBalancer(...) | _PTRRLoadBalancer(...) | None   # 见 §3
inputs = _context_parallel_shard(mesh=cp_mesh, buffers=inputs,
                                 seq_dims=(1,1,1), load_balancer=load_balancer)
# attention_masks(BlockMask)单独切,seq_dim 用 2(BlockMask 形状 [B,H,Q,KV],只能切 Q 维)
masks = _context_parallel_shard(mesh=cp_mesh, buffers=masks, seq_dims=(2,)*len(masks), ...)
```

**注意 `attention_masks` 走独立的 `seq_dim=2`**——`BlockMask` 的形状是 `[B,H,Q,KV]`,只能切 Q 维,不能像 `inputs/labels/positions` 那样统一按 `seq_dim=1` 切,这是 torchtitan 数据切分入口里唯一的框架特殊处理。

---

## 3. 负载均衡:torchtitan 的两个 Balancer 类

> 折叠/头尾配对的数学证明、量化算例、PTRR(任意稀疏掩码处理时间均衡)算法均已归一到理论页 §3.2-§3.4——本文的算例正是理论页 §3.2/§3.4 的骨架来源。本节只记 torchtitan 侧的 API/配置入口。

torchtitan 提供两个 `load_balancer` 实现,由 `apply_cp_to_forward` 按 attention 类型选择:

- **`_HeadTailLoadBalancer`**(SDPA 路径默认):机制见理论页 §3.3。
- **`_PTRRLoadBalancer`**(FlexAttention 路径):机制见理论页 §3.4。

> **`seq_len` 必须被 `2·cp` 整除**([[torchtitan_parallel_dims_analysis]] 的 `seq_len_divisor`):头尾配对要把序列切成 `2·cp` 个等长 chunk,这条约束在 torchtitan 里由 `ParallelDims` 校验层强制检查——这是理论页未展开的 torchtitan 特有配置校验入口。

---

## 4. 两条 attention 路径(torchtitan 独有架构)

`apply_cp_to_forward`(`context_parallel.py:35`)按 inner attention 类型分两条路径——**这是 torchtitan/PyTorch CP 特有的设计**:Megatron/MindSpeed/DeepSeek-V4 都没有按"attention 实现类型"分派两条独立 CP 路径的架构。

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

`ScaledDotProductAttention.forward` 先把布局 transpose 成 `[B,H,S,D]`,所以序列维是 **dim 2** → `Shard(2)`。`_enable_context_parallel_dispatcher` 把 6 个 aten SDPA op(flash/efficient/cudnn 的 fwd+bwd)映射到 `_sdpa_handler`,后者把 `Shard(2)` 的 SDPA 路由到 `_templated_ring_attention`(主循环机制见理论页 §5)。

### 4.2 FlexAttention 路径:all-gather K/V

```python
# context_parallel.py:57
if isinstance(first, FlexAttention):
    def cp_forward(q, k, v, **kwargs):
        global_k, global_v = flex_cp_allgather(k, v, 1, pg_name)   # 直接 all-gather 全量 K/V
        return orig_fn(q, global_k, global_v, **kwargs)            # Q 是本地分片,K/V 是全局
```

`seq_dim` 传 `1`:`apply_cp_to_forward` 包的是 `FlexAttention.forward` 外层,此时 q/k/v 还是 `[B,S,H,D]` 布局。`flex_cp_allgather` 是注册过的 custom op(为了支持 `torch.compile` 与 autograd)——与理论页 §6 描述的 Megatron 原生 all-gather CP 是**同一大类机制**(先收齐 KV 再单次计算),但 torchtitan 用 custom op 而非 `torch.autograd.Function` 实现,这是 PyTorch 生态里的实现差异。

### 4.3 两条路径的取舍

| 维度 | FlexAttention(all-gather) | SDPA(ring) |
|---|---|---|
| K/V 通信 | **一次** all-gather 拿全局 K/V | `cp_size` 步,K/V 环形轮转 |
| K/V 显存 | 每 rank 持有**全量** K/V `O(S)` —— 抵消了 CP 省显存初衷 | 每 rank 任意时刻只持 1~2 份分片 `O(S/cp)` |
| 通信/计算重叠 | 几乎没有(就一次 all-gather) | ring 主循环天然重叠 |
| 负载均衡 | `_PTRRLoadBalancer`(数 BlockMask 实际计算量) | `_HeadTailLoadBalancer`(因果三角几何配对) |
| 掩码 | 任意稀疏 `BlockMask` | 只支持 `is_causal` 布尔 |

**一句话**:Flex 路径用"K/V 全量复制 + 一次 all-gather"换实现简单和任意稀疏掩码,代价是 K/V 显存退化成 `O(S)`;SDPA ring 路径用"K/V 分片 + 多步轮转 + 在线 softmax"保住 `O(S/cp)` 显存并天然重叠通信。**CP 省显存的核心收益在 SDPA ring 路径上才完整。**

---

## 5. Ring 通信调度:torchtitan 的接线

> Ring 主循环(`_templated_ring_attention`)、在线 softmax 合并、演算图均已归一到理论页 §5.1-§5.2(本文正是骨架来源)。本节只记 §4.1 之外 torchtitan 特有的调度器接线细节。

`_create_rotater(group, 2)` 按 `rotate_method` 选择两种轮转器(`_AllToAllRotater` / `_AllGatherRotater`,默认后者),两者的取舍见理论页 §5.4;torchtitan 侧的配置项是 `context_parallel_rotate_method`。`_is_causal_behavior` 每步选掩码(含 SKIP 分支)的机制见理论页 §4.2。

---

## 6. 通信掩盖

机制(为什么下一步传输能与当前步计算重叠、指令顺序 (A)(B)(C)(D))已归一到理论页 §5.3——本文正是该节骨架来源,不再重复。torchtitan 侧唯一需要补充的是它**用什么实现这份异步**,见 §7。

---

## 7. 异步实现:靠 functional collective,不是手写 stream(torchtitan/PyTorch 特有)

> **关键澄清**:torchtitan / PyTorch 的 CP **没有手写 CUDA stream**。重叠完全靠 `torch.distributed._functional_collectives` 的异步语义——容易被误以为是"显式多 stream"。这是 PyTorch 生态独有的工程选择,Megatron/TE 用独立 `cp_stream` + `cudaEvent`、MindSpeed 用 `isend`/`irecv` 实现同一条通用原则(见理论页 §5.3 末尾的三框架对照),三者互不相同,不构成重复。

机制:

- `_AllToAllRotater.exchange_buffers` 调 `ft_c.permute_tensor(...)`(`ft_c` = functional collectives),底层走 `torch.ops._c10d_functional.*`,**返回一个 `AsyncCollectiveTensor`(ACT)**。通信在 c10d 的通信 stream 上发起,**不阻塞当前计算 stream**。
- `next_buffer` 里 `_maybe_wait(tensor)`:若是 ACT 就 `.wait()`。`ACT.wait()` 触发 `wait_tensor` op,在通信 stream 与计算 stream 之间插入跨 stream 同步。

所以重叠的本质:**(B) 在通信 stream 上发起 collective 并立即返回 ACT;(C) 的 SDPA kernel 同时在计算 stream 上跑;下一轮 (A) 的 `wait()` 才在两条 stream 间同步**。这与 [[torchtitan_fsdp_analysis]] FSDP 的"手写多 stream"是不同的实现风格——CP 复用了 functional collective 内建的通信 stream + ACT 延迟 wait 机制。

`_AllGatherRotater` 同理:第一步 `all_gather_tensor` 返回 ACT,只有第 0 步那次 all-gather 能和第 0 步 SDPA 重叠。

---

## 8. 反向传播

### 8.1 SDPA ring 的反向

机制(两个环、为什么 dKV 环强制 all-to-all)已归一到理论页 §5.5——本文正是该节骨架来源,不再重复。

### 8.2 FlexAttention all-gather 路径的反向(torchtitan 特有实现)

`flex_cp_allgather` custom op 注册了 autograd:forward 是 `all_gather`(本地 K/V 分片 → 全局 K/V),所以 backward 必须是 **`reduce_scatter`**(全局 K/V 梯度 → 规约求和 + 散回各 rank 本地分片)。这与理论页 §6.3 描述的 Megatron 原生 all-gather CP 反向是**同一种 adjoint 模式**(all-gather 的转置算子是 reduce-scatter),但 torchtitan 用**注册了 autograd 的 custom op**(而非 `torch.autograd.Function` 子类)实现——这是为了让 `flex_cp_allgather` 能被 `torch.compile` 追踪,是 torchtitan/PyTorch 特有的工程约束。

---

## 9. 完整流程图(torchtitan 自身的调用链)

```
═══ 每个训练 step:数据切分 ═══
trainer.py:618  prepare_context_parallel_input → cp_shard
   │ load_balancer 生成重排索引(§3)
   │ torch.gather 按索引重排序列  →  distribute_tensor(Shard(seq)) 均匀切
   ▼ 每个 CP rank 拿到序列的一段

═══ 建模期:forward 包装 ═══
apply_cp_to_forward(inner_attention 列表, cp_mesh)
   ├─ FlexAttention → cp_forward 里 flex_cp_allgather 全量 K/V(§4.2)
   └─ SDPA → _enable_context_parallel_dispatcher,q/k/v 包成 Shard(2) DTensor
              → _sdpa_handler 路由到 _templated_ring_attention(主循环见理论页 §5)

═══ 前向 / 反向期 ═══
Ring 主循环、online-softmax、通信掩盖、反向双环 —— 见理论页 §5
```

---

## 10. torchtitan 特有小结

- **两个数据入口**:`cp_shard`(trainer 侧)+ `apply_cp_to_forward`(parallelize 侧),`context_parallel.py` 注释自陈是过渡到 `ShardingConfig` 前的临时方案(§1)。
- **`BlockMask` 独立切分**:`attention_masks` 走 `seq_dim=2`,与 `inputs/labels/positions` 的 `seq_dim=1` 不同(§2)。
- **torchtitan 独有的双路径架构**:SDPA→ring(`Shard(2)` DTensor + `_templated_ring_attention`)与 FlexAttention→all-gather(`flex_cp_allgather` custom op)是两条完全独立的实现,按 inner attention 类型分派,四框架中仅此一家(§4)。
- **异步靠 functional collectives,不手写 stream**:`AsyncCollectiveTensor` 延迟 `wait()` 是 PyTorch 生态特有的重叠实现路径,与 Megatron/TE 的独立 `cp_stream`、MindSpeed 的 `isend`/`irecv` 并列为三种不同的异步落地方式(§7)。
- **通用机制**(为什么切序列、折叠/头尾负载均衡、因果裁剪、Ring 主循环+online-softmax、通信掩盖原理、反向双环)见 [[../../../01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis|20_ring_attention_and_context_parallel_analysis]]。

## Related Pages

- [[../../../01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis|20_ring_attention_and_context_parallel_analysis]] —— CP/Ring Attention 通用机制(本页多节的骨架来源页)
- [[torchtitan/index]] · [[torchtitan_parallel_dims_analysis]] —— 知识地图与并行基座
- [[torchtitan_tp_analysis]] · [[torchtitan_pp_analysis]] —— 相邻并行维度
- [[13_megatron_cp_analysis]] —— Megatron-LM 上下文并行实现差异(`cp_comm_type` 四选一 + TE 透传)
- [[35_deepseek_v4_context_parallel_analysis]] —— DeepSeek-V4 CP 实现、Native/TE CP、Dynamic CP
- [[mindspeed_context_parallel_analysis]] —— MindSpeed 上下文并行实现差异(五算法运行期分派)
