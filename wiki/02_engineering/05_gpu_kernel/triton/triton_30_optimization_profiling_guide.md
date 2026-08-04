# L4 · 会优化 — Profiling 驱动的 Kernel 优化：从 roofline 到 FlashAttention

> **源基线**: `triton main @ 70e0929`，v3.8.0 ｜ 锚定 `python/tutorials/06-fused-attention.py`（FlashAttention v2）、`05-layer-norm.py`（反向）、`09/10` 的 proton profiler
> **维度**: 学习路线 L4（会优化）｜ 能力：**会优化**
> 前四页让你**写对**（L1）、**调好**（L2）、**调通**（L3）。本页回答最后一问：**已经能跑了，怎么让它更快？** 主线是「先 profile 定位瓶颈，再用对的杠杆」。标杆案例是 FlashAttention——融合到底、把 S 留在 SRAM、把 HBM 流量从 $O(N^2)$ 压到 $O(N\cdot d)$。前置：[[triton_01_gpu_essentials_guide]]（roofline）、[[triton_13_autotune_guide]]（`num_warps`/`num_stages`）。

---

## 1. 前提：先判 Roofline，再选杠杆

Roofline 判据（memory-bound / compute-bound 怎么判）、公式与硬件 Ridge Point 参数见 Roofline 权威页 [[11_operator_optimization_guide]] §2；「profile → 判 bound → 选对杠杆 → 复测」的通用优化闭环见该页 §8。**不要凭直觉改 kernel，「优化错误的瓶颈」是性能工程第一大浪费。** 本页把这个闭环落到 Triton 实战的五个具体杠杆（下表），逐个杠杆的源码级证据见 §3。

### 优化杠杆速查表

| 杠杆 | 治什么 bound | 机制 | 源出处（本仓 tutorial） |
|---|---|---|---|
| **① fusion** | memory | 多个 op 合一个 kernel，中间结果留 SRAM 不落 HBM | `02-fused-softmax.py`、`06-fused-attention.py:103` |
| **② `num_stages`** | compute（隐藏访存延迟） | 软件流水线：`cp.async` 多缓冲，算当前块时预取下一块 | `02-fused-softmax.py:90`、`03/06` configs |
| **③ `num_warps`** | compute（占用率） | 每 program 的 warp 数，权衡并行度 vs 每线程寄存器 | `02-fused-softmax.py:131-171` |
| **④ 分块 & L2** | memory | 块大小 + `GROUP_SIZE_M` 重排，提复用与 L2 命中 | `03-matrix-multiplication.py:166-198` |
| **⑤ Tensor Core `tl.dot`** | compute | 把矩阵乘映射到 MMA 单元，峰值算力的唯一入口 | `06-fused-attention.py:73,103` |

> 第 3 节逐个展开每个杠杆并给源证据；第 4 节用 FlashAttention 把 ①②③⑤ 串成一个标杆。

---

## 2. 先 profile 再优化

### 2.1 最轻量：`do_bench`（量时间/带宽）

每个 tutorial 都用 `triton.testing.do_bench` 测单 kernel 墙钟时间，再换算成 GB/s 或 TFLOPS。LayerNorm 反向就把它包成带宽曲线（`05-layer-norm.py:368`，`gbps = 3*x.numel()*elem*1e-9/(ms*1e-3)`，反向读 dy/x、写 dx 共 3 趟 → 系数 3）。这是判 memory-bound 最快的手段：**实测带宽接近卡的 HBM 峰值 → 已是 memory-bound，再加算力无用**。

`do_bench` 还支持 `quantiles=[0.5, 0.2, 0.8]`（`05-layer-norm.py:347,369`）返回中位数与上下分位——**报性能用中位数而非均值**（抗偶发抖动），分位差大则提示 launch/时钟不稳，profile 结论要打折扣。`do_bench` 内部自带 warmup（`python/triton/testing.py:263,310-315`）并在**每次计时前清 L2 cache**（`:325-326` 的 `clear_cache`），避免「热缓存假象」把带宽测虚高。

### 2.2 In-tree profiler：proton（真实 API，已核验到行号）

Triton **自带** profiler，源在 `third_party/proton/proton/`，公开入口 `import triton.profiler as proton`（`09-persistent-matmul.py:28`、`10-block-scaled-matmul.py:125`）。核心 API（签名见 `proton/profile.py`）：

| API | 签名/用法 | 源出处 |
|---|---|---|
| `proton.start(name, *, hook=...)` | 开一个 profiling session；`hook="triton"` 记录每次 kernel launch 元数据 | `profile.py:52-60`；`09:753`、`10:743` |
| `proton.scope(name, metrics_dict)` | **context manager**，给一段代码打标签并附带 `{"bytes":…, "flops16":…}` 等指标 | `09:618-619`、`10:485-486` |
| `proton.activate(session=None)` / `deactivate(...)` | 临时开/关录制（跳过 warmup 与参数构造） | `profile.py:133,155`；`09:636-641,754`、`10:710-713` |
| `proton.finalize()` | 收尾，写出 `<name>.hatchet` 结果文件 | `profile.py:178`；`09:757`、`10:751` |
| `triton.profiler.viewer.parse / print_tree` | 解析 `.hatchet`、按指标打印调用树 | `09:719-728`、`10:492-499` |

09 里把指标名设成 `["time/ms"]`、`["tflop16/s", "time/ms"]`（`09:721-725`）——proton 用 `scope` 里登记的 `bytes`/`flops*` 字段自动算出吞吐。这正是 profiler 的价值：**同一张表里把 Triton kernel、cuBLAS、torch.matmul 排在一起比**（`09:618-632`，三个 `proton.scope` 分别标 BLAS/torch/各 Triton 变体）。

> ⚠️ 注意：`proton.start()` 在脚本**从命令行经 proton 前端启动**或 profiling 被 knob 关闭时会直接返回 `None`、不重复开 session（`profile.py:106-108`）。env / knob 名本页不臆造，以源 `profile.py:106` 的 `triton.knobs.proton.disable` 为准。

### 2.3 外部替代：Nsight（概念级，非本仓代码）

NVIDIA 的 **Nsight Systems**（`nsys`，时间线/CPU-GPU 重叠/kernel 间隙）与 **Nsight Compute**（`ncu`，单 kernel 的占用率、Tensor Core 利用率、访存吞吐、stall 原因）是 proton 之外的工业标准。它们不在 Triton 源树内，本页只作概念指引、不杜撰具体 flag。**经验法则**：先用 `do_bench`/proton 找到「哪个 kernel 慢、是哪种 bound」，再用 `ncu` 钻进那一个 kernel 看微观瓶颈。

**只优化瓶颈**：profile 的第一产出永远是「时间花在哪」。占比 60% 的 kernel 提速 2× 比占比 5% 的提速 10× 更值——这是 Amdahl 定律的直接推论。

---

## 3. 五个优化杠杆（逐个带证据）

### ① Fusion —— 把中间结果留在 SRAM（治 memory-bound）

最有效的省带宽手段：把本要分多个 kernel、靠 HBM 传递中间结果的计算，**合并成一个 kernel**，中间量只在 SRAM 周转。`02-fused-softmax.py` 把 `max→sub→exp→sum→div` 五步融合成一遍读、一遍写（详见 [[triton_11_fused_softmax_guide]]）；FlashAttention 更激进——把整个 `QK^T→softmax→·V` 融合（`06:73,82,103`），连 $N\times N$ 的注意力矩阵都不落 HBM（第 4 节详解）。**为什么有效**：memory-bound kernel 的时间≈HBM 字节数/带宽，融合直接砍掉中间张量的写+读两趟流量。

### ② `num_stages` —— 软件流水线隐藏访存延迟（治 compute-bound 的隐藏延迟）

`num_stages` 控制 Triton 为循环开几级缓冲：算第 $i$ 块时，用 `cp.async` 异步把第 $i{+}1$ 块从 HBM 预取进 SRAM，让访存延迟被计算掩盖（多缓冲/double-buffering 的推广）。源里两种写法都有：
- 显式：`02-fused-softmax.py:90` 的 `tl.range(row_start, n_rows, row_step, num_stages=num_stages)`，`num_stages` 是 kernel 的 `constexpr` 参数（`:86`），host 端按 SMEM 大小选 `num_stages = 4 if SIZE_SMEM > 200000 else 2`（`:137`）。
- autotune 搜：`03-matrix-multiplication.py:166-198` 的 configs 里 `num_stages` 取 3/4/5；`06-fused-attention.py:131-141` 的 `NUM_STAGES_OPTIONS=[2,3,4]` 进 autotune 空间。

**权衡**：级数越多预取越深、越能掩盖延迟，但每多一级就多占一份 SRAM 缓冲——SMEM 不够会降占用率甚至编不出来。所以 `02` 才按 `SIZE_SMEM` 动态选，`06` 才交给 autotune。流水线在编译器侧如何下降到 `cp.async`/MMA，见 [[30_triton_vs_mlir_backend_analysis]]。

### ③ `num_warps` —— 占用率的旋钮（治 compute-bound 的并行度）

`num_warps` = 每个 program 用多少 warp。`02-fused-softmax.py:131-171` 给了教科书式的占用率计算：先 `warmup` 编一次 kernel 拿到寄存器用量 `n_regs`（`:146`），再算

```text
occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)   # 02:167
occupancy = min(occupancy, SIZE_SMEM // size_smem)         # :168 受 SMEM 再压一道
num_programs = NUM_SM * occupancy                          # :169
```

**权衡**：warp 越多，能并行掩盖延迟的线程越多（占用率↑）；但寄存器是定额资源，`num_warps↑` 会摊薄每线程寄存器，多了反而 spill。`06-fused-attention.py:140` 把 `num_warps∈{4,8}` 交给 autotune 搜——因为最优值随 `BLOCK_M/BLOCK_N`/卡而变，没有静态最优。承接 [[triton_13_autotune_guide]]。

### ④ 分块大小 & L2 复用（治 memory-bound）

块越大，每个 program 复用进 SRAM 的数据越多、对 HBM 的有效访存越少；但块太大挤占 SRAM/寄存器、降占用率——又是权衡。matmul 还有一招超越单块的复用：`GROUP_SIZE_M` 把 program 的执行顺序按「L2 友好」重排，让相邻 program 复用同一批已在 L2 的 A/B 分块（`03-matrix-multiplication.py:166-198` 每个 config 都带 `GROUP_SIZE_M`，原理见 [[triton_12_matmul_guide]] 的 L2 grouping 一节）。

### ⑤ Tensor Core：`tl.dot`（治 compute-bound）

矩阵乘想摸到峰值算力，**唯一入口是 `tl.dot`**——它会被编译器映射到 Tensor Core 的 MMA 指令；用标量循环手写乘加只能跑 CUDA core，差一个数量级。FlashAttention 两处 `tl.dot`：`06:73` 算 $S=QK^\top$、`06:103` 算 $acc \mathrel{+}= P\,V$（第三参数 `acc` 让它做「乘加进累加器」）。`tl.dot→MMA` 的下降细节见 [[30_triton_vs_mlir_backend_analysis]]。配套技巧：累加器用 fp32（`06:218` `acc = tl.zeros(..., tl.float32)`）保精度，输入在喂给 MMA 前转半精度（`06:101` `p = p.to(dtype)`）。

---

## 4. 标杆案例：FlashAttention（锚定 `06-fused-attention.py`）

`06-fused-attention.py` 头部即标明出处：Tri Dao 的 **Flash Attention v2**（`:5`）与原始 FA 论文 **arXiv:2205.14135**（`:11`），OpenAI kernel team 实现（`:7`）。它是「五个杠杆同时拉满」的集大成者，核心创新只有一句话——

> **永远不把 $N\times N$ 的注意力分数矩阵 $S=QK^\top$ 写进 HBM。** 用 online softmax 沿 K/V 维分块流式计算，全程只在 SRAM 里维护三个小状态：running max `m_i`、running sum `l_i`、累加器 `acc`。

下图是这套流式方案的全貌：一个 program 持有一块 $Q$（`:222` 注释「stay in SRAM throughout」），沿 K/V 维一块块滑过，每滑一块就地更新三个常驻状态，**分数块 $S_j$ 用完即弃、永不落 HBM**：

```text
       SRAM 常驻（一个 program 的私有状态）
       ┌───────────────────────────────────────────┐
  Q_i ─┤ q (BLOCK_M×d)   m_i(-inf)  l_i(1)  acc(0)   │
       └───────────────────────────────────────────┘
              │            ▲   ▲   ▲
   HBM 流式读   │  每块: S_j=q·k → 更新 m_i → p=exp2 → α重标定 acc,l_i
        ┌──────┴──────┬──────────┬──────────┬─────────┐
   K/V: │ blk 0       │ blk 1    │ blk 2    │ ...      │   (各 BLOCK_N×d)
        └─────────────┴──────────┴──────────┴─────────┘
        S_0 ✗丢弃      S_1 ✗丢弃   S_2 ✗丢弃   ...   ← N×N 永不物化
                                                  │
                              收尾一次: O_i = acc / l_i  ──► HBM 写 O (N×d)
```

### 4.1 online softmax 内层循环：逐步骤拆解

前向把每个 query 块（`BLOCK_M` 行）交给一个 program，沿 key/value 维分块迭代（`_attn_fwd_inner`，`:47-110`）。初始化（`:216-218`）：

$$m_i = -\infty,\qquad \ell_i = 1,\qquad acc = \mathbf{0}\in\mathbb{R}^{\text{BLOCK\_M}\times d}$$

内层循环 `for start_n in tl.range(lo, hi, BLOCK_N, ...)`（`:69`）对第 $j$ 个 K/V 块做 6 步——这正是 online softmax 的精髓：

| 步 | 代码 | 行 | 数学 |
|---|---|---|---|
| 1. 算分数块 | `k = desc_k.load(...).T; qk = tl.dot(q, k)` | `:72-73` | $S_j = Q K_j^\top$（**只在 SRAM**） |
| 2. 更新 running max | `m_ij = tl.maximum(m_i, tl.max(qk,1)*qk_scale)` | `:80`(`:77` causal) | $m_i^{\text{new}}=\max(m_i,\ \max_n S_{j})$ |
| 3. 稳定指数 | `qk = qk*qk_scale - m_ij[:,None]; p = tl.math.exp2(qk)` | `:81-82` | $P_j=\exp(S_j-m_i^{\text{new}})$ |
| 4. **修正因子** | `alpha = tl.math.exp2(m_i - m_ij)` | `:84` | $\alpha=\exp(m_i^{\text{old}}-m_i^{\text{new}})$ |
| 5. **重标定累加器** | `acc = acc * alpha[:,None]; acc = tl.dot(p, v, acc)` | `:95,103` | $acc \leftarrow \alpha\cdot acc + P_j V_j$ |
| 6. 更新 running sum | `l_i = l_i * alpha + l_ij; m_i = m_ij` | `:106-107` | $\ell_i\leftarrow \alpha\,\ell_i+\textstyle\sum_n P_j$ |

收尾（epilogue，`:243-244`）做唯一一次归一化：

$$O_i = \frac{acc}{\ell_i},\qquad \text{并存 logsumexp } m_i \mathrel{+}= \log_2 \ell_i\ \text{供反向用}$$

**关键 why——为什么要 `alpha` 重标定**：softmax 必须减去整行最大值才数值稳定，但流式计算时「整行最大值」要到看完最后一块才知道。第 $j$ 块发现了更大的 max 时，之前所有块都是按**旧的、偏小的** $m_i$ 归一化的，已偏大。乘 $\alpha=\exp(m_i^{\text{old}}-m_i^{\text{new}})\le 1$ 把历史的 `acc` 和 `l_i` **同步降权到新基准**，等价于「假装一开始就用新 max 算」。于是无需回看任何已丢弃的块，就得到与一次性 softmax 数值等价的结果。源把 `m_i/l_i` 的更新刻意放在循环末尾「以降低寄存器压力」（`:104-105` 注释）。

> 实现细节：用 `tl.math.exp2`（base-2）而非 `exp`，因为 GPU 有更快的 `exp2` 指令；`qk_scale` 在 `:221` 预乘 `1.44269504 = 1/ln2`，把 $e^x$ 换成 $2^{x/\ln2}$，等价但更快。

### 4.2 IO 复杂度：$O(N^2)\to O(N\cdot d)$（含推导边界）

**源事实**（可核验，`:72-103`）：分数块 $S_j$ 由 `tl.dot` 产出后，当场被 `exp2`→`P`→`tl.dot(p,v,acc)` 消费，**从不 `store` 回 HBM**；跨迭代只有 `m_i`(BLOCK_M)、`l_i`(BLOCK_M)、`acc`(BLOCK_M×d) 三个小状态；全 kernel 唯一的大写出是 `desc_o.store(...)` 的 $O$（$N\times d$，`:247`）。

**推导**（本页推导，基于上述源事实）：

| | 标准 attention | FlashAttention |
|---|---|---|
| 是否物化 $S=QK^\top$ 到 HBM | 是：写 $N{\times}N$ 再读回 | **否**：S 只在 SRAM 生灭 |
| HBM 流量主项 | $\Theta(N^2)$（被 $S$ 支配） | $\Theta(N\cdot d)$（读 Q,K,V + 写 O） |

直觉上 FlashAttention 把 HBM 流量从「被 $N^2$ 的分数矩阵支配」降到「只与 Q/K/V/O 的 $N\cdot d$ 同阶」。**严格表述**（FA 论文 arXiv:2205.14135，`:11`）：设片上 SRAM 大小为 $M$，FA 的 HBM 访问量是 $\Theta(N^2 d^2 / M)$，而标准实现是 $\Theta(Nd + N^2)$——因为 K/V 会被各个 query 块重复读，所以并非字面 $O(Nd)$，但当 $d^2\ll M$ 时 FA 的 $N^2$ 系数被 $d^2/M$ 显著压小。这正是 attention 从 memory-bound 受益于融合的根本原因。

> 与 [[01_gpu_kernel_guide]] §08 的关系：那页已给出 FlashAttention 的硬件层级映射表（Grid/Block/SRAM/Warp/Register/Tile 各落到哪）。本页**不重复**那张表，只补「Triton 实现视角」——上面的 `m_i/l_i/acc` 三状态与 `alpha` 重标定，就是该表里「SRAM 常驻状态」一行的代码级真相。attention 的其它变体见 [[26_flex_attention_analysis]]。

### 4.3 configs 解读：把杠杆②③④交给 autotune

`_attn_fwd` 用 `@triton.autotune` 包裹（`:176-177`），搜索空间由 `:135-141` 笛卡尔积生成：

```python
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, ...)
    for BM in [64, 128] for BN in [32, 64, 128]
    for s in NUM_STAGES_OPTIONS   # = [2, 3, 4]  (:131,133)
    for w in [4, 8]               # num_warps
]
```

这一个空间同时拉动**④分块**（`BLOCK_M×BLOCK_N`）、**②流水线**（`num_stages`）、**③占用率**（`num_warps`）三个杠杆；`key=["N_CTX","HEAD_DIM",...]`（`:176`）让不同序列长/头维各自缓存最优 config；`prune_invalid_configs`（`:156-165`）剪掉 `BLOCK_M>N_CTX` 等非法组合，省搜索时间。这就是 L2「会调」在真实 kernel 上的样子——优化杠杆的最优档位由 autotune 实测决定，而非拍脑袋。

---

## 5. 反向 kernel 优化：LayerNorm backward 的并发累加（`05-layer-norm.py`）

> **反向比前向难优化**，典型卡点：多个 program 要把梯度累加到**同一块**输出上，产生写-写冲突。LayerNorm 的 $\nabla_w,\nabla_b$ 是教科书例子——同一个 $w,b$ 被一个 batch 的所有行共享，每行都贡献一份 $\nabla_w=\nabla_y\odot\hat x$、$\nabla_b=\nabla_y$，必须求和（源公式 `:115`）。

朴素做法「每行 `tl.atomic_add` 到全局 $\nabla_w$」会让成百上千行抢同一地址，原子串行化 → 灾难性慢。源用**两阶段并行归约**（策略说明 `:117-129`）：

**Stage 1 — `_layer_norm_bwd_dx_fused`（`:132-194`）：分桶 + 锁**
把行哈希进 `GROUP_SIZE_M` 个独立缓冲（`lock_id = row % GROUP_SIZE_M`，`:153`），让冲突只发生在桶内、且每桶配一把锁。累加用自旋锁保护临界区：

```python
while tl.atomic_cas(Lock, 0, 1) == 1:   # :177 自旋抢锁
    pass
count = tl.load(Count)                   # :179
if count == 0:                           # :181 本桶首次写：直接存，不累加
    tl.atomic_xchg(Count, 1)
else:                                     # :183 非首次：读旧值 + 本次偏导
    partial_dw += tl.load(DW, mask=mask)
    partial_db += tl.load(DB, mask=mask)
tl.store(DW, partial_dw, mask=mask)      # :186
tl.store(DB, partial_db, mask=mask)
tl.debug_barrier()                       # :191 放锁前栅栏，确保写完成
tl.atomic_xchg(Lock, 0)                  # :194 放锁
```

这些 `GROUP_SIZE_M×N` 的部分和缓冲**驻留 L2**（源 `:120`），所以桶内累加快。

**Stage 2 — `_layer_norm_bwd_dwdb`（`:197-221`）：跨桶最终归约**
另起一个 kernel 把 `GROUP_SIZE_M` 个部分和按列纵向加总成最终 $\nabla_w,\nabla_b$（`:211-219` 的 `for i in range(0, M, BLOCK_SIZE_M)` 循环 + `tl.sum(dw, axis=0)`）。host 端按特征维 `N` 启发式选桶数（`:267-270`，`N≤1024` 用 256 桶，`N` 大则减到 64），并预分配锁与缓冲（`:272-274`，`locks = zeros(2*GROUP_SIZE_M)`，前半 Lock、后半 Count）。

**为什么这样比纯原子快**：把「全局 $M$-路冲突」降级为「桶内少量冲突 + 一次干净的二阶段树形归约」，用一点 L2 缓冲空间换掉了原子风暴。对照 FlashAttention 的反向（`_attn_bwd*`，`:250-502`）同样要拆 dK/dV 与 dQ 两个方向、靠分块复用 K/V 常驻 SRAM（`:413-415` 注释）——**反向的难点永远是「梯度要往多个方向累加」，解法永远是「重排成可并行的分块归约」**。

---

## 6. 动手验证（必做）

```bash
cd triton/python/tutorials

# ① FlashAttention：打印 TFLOPS 表 + 存曲线（源 :773-775 的入口）
python 06-fused-attention.py
#   会跑 fwd/bwd × causal × HEAD_DIM 多组，红线 Triton[FP16]，N_CTX=1k..16k（源 :713）
#   TFLOPS 怎么算（源 :764-769）：两个 matmul 各 2·B·H·N²·d FLOP → ×2；
#   causal 三角只算一半 → ×0.5；反向 ×2.5（2.0 反传 + 0.5 重算前向）

# ② LayerNorm 反向：打印 GB/s 带宽曲线（源 :374-375）
python 05-layer-norm.py
```

**最小 proton profiling 片段**（只用第 2.2 节核验过的 API）：

```python
import torch, triton
import triton.profiler as proton          # 09:28

proton.start("my_run", hook="triton")     # profile.py:52；hook 记录每次 launch 元数据
for _ in range(100):
    # scope 给这段打标签并登记字节/算力，proton 据此算吞吐（09:618-619）
    with proton.scope("my_kernel", {"bytes": nbytes, "flops16": 2.0 * M * N * K}):
        my_kernel[grid](...)
proton.finalize()                          # 09:757，写出 my_run.hatchet

# 查看结果（09:719-728 的 viewer）
import triton.profiler.viewer as proton_viewer
tree, metrics = proton_viewer.parse(["tflop16/s", "time/ms"], "my_run.hatchet")
proton_viewer.print_tree(tree, metrics)
```

> 想跳过 warmup 不计入：用 `proton.activate()` / `proton.deactivate()` 把录制只圈在正式测量循环（`09:636-641` 的 `proton_context` 范式）。

**优化练习（巩固「会优化」）**：
1. 给 `06-fused-attention.py` 临时把 `NUM_STAGES_OPTIONS` 砍成 `[2]`（`:131-133`），复跑看 TFLOPS 掉多少——量化 `num_stages` 流水线的收益。
2. 给 `05-layer-norm.py` 的反向把 `GROUP_SIZE_M` 固定成 1（`:267-270`），观察带宽暴跌——亲手验证「原子风暴」。
3. 用上面的 proton 片段，把你自己的 kernel 和 `torch` 等价算子放进同一份 `.hatchet` 比 TFLOPS。

---

## 7. 「会优化」能力清单

- [ ] 默写优化闭环：**profile → 判 memory/compute-bound → 选对杠杆 → 复测**，且坚持「只优化瓶颈」
- [ ] 会用 `do_bench` 量带宽/TFLOPS；会用 proton 的 `start/scope/activate/deactivate/finalize` + viewer 出调用树
- [ ] 说得清五个杠杆各治哪种 bound：fusion / `num_stages` 流水线 / `num_warps` 占用率 / 分块&L2 / `tl.dot` Tensor Core
- [ ] 讲得清 FlashAttention「为什么不物化 S」：online softmax 用 `m_i/l_i/acc` + `alpha` 重标定，HBM 流量 $O(N^2)\to O(N\cdot d)$
- [ ] 看得懂 `06` 的 autotune configs 如何同时拉动分块/流水线/占用率三个杠杆
- [ ] 懂反向 kernel 的核心难点（梯度并发累加）与解法（分桶+锁的两阶段并行归约）

至此 L0→L4 闭环完成。回到 [[triton_31_knowledge_guide]] 做四种能力的自测与查漏。

---

## 相关页面

- [[02_engineering/05_gpu_kernel/triton/index|Triton 学习路线]] — Triton 学习路线总索引
- [[triton_01_gpu_essentials_guide]] — 前置：roofline / memory vs compute bound 的判据
- [[11_operator_optimization_guide]] — **Roofline 权威页**：公式、Ridge Point 参数表、通用优化闭环（§2/§8）
- [[triton_11_fused_softmax_guide]] — 杠杆①fusion 的最小案例
- [[triton_12_matmul_guide]] — 杠杆④分块&L2 grouping、`tl.dot` 入门
- [[triton_13_autotune_guide]] — 杠杆②③：`num_stages`/`num_warps` 的自动搜索
- [[triton_14_debug_guide]] — 优化引入 bug 时回这里（interpreter / assert）
- [[triton_31_knowledge_guide]] — 四种能力总纲与自测
- [[01_gpu_kernel_guide]] — FlashAttention 硬件层级映射表（§08）、Tensor Core 硬件视角（与本页互补）
- [[26_flex_attention_analysis]] — attention 变体与 mask/score 修改
- [[30_triton_vs_mlir_backend_analysis]] — `tl.dot→MMA`、`num_stages` 流水线在编译器侧的下降
