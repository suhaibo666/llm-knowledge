---
title: "Megatron-LM 张量并行(Tensor Parallelism)深度解析"
---

# Megatron-LM 张量并行(Tensor Parallelism)深度解析

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **重定基线**：2026-09-01 由 `71092579`（2026-08-27）推进，跨 7 个提交；该增量只触及 20 个 `megatron/` 文件，本页 `path:line` 引用所涉源文件均不在其中，故无行号漂移，无需逐条重核。
> **重定基线**：2026-08-28 由 `ee3f1ffa…`（2026-05-19）推进，跨 578 个提交；本页全部 `path:line` 形式的引用已在新基线下逐条重核;**代码块内被点名的符号与不带行号的裸路径不在该次扫描口径内**,已知漏网处已于 2026-08-28 单独更正。
> 核心文件:`megatron/core/tensor_parallel/layers.py`(1408 行)、`megatron/core/tensor_parallel/mappings.py`(714 行)
> 配套阅读:`15_megatron_pp_schedulers_analysis.md`、`14_megatron_ep_analysis.md`
> 适用读者:已了解 transformer 训练与 DP/PP,想吃透 Megatron 张量并行实现的工程师。
> **叙事顺序**：本页按五拍组织——背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势。
> **最近更新**：2026-08-28。按五拍重排章节顺序；机制正文与既有引用未改。

---

## 1. 背景：单层的四个大矩阵乘放不下,而 PP 切层、DP 切批都动不了它

### 1.1 要解决的问题

一个 transformer 层的权重主要是 4 个大矩阵:attention 的 QKV 投影、输出投影,FFN 的 fc1、fc2。一层约 `12h²` 参数;大模型 `h` 上万、层数上百 → 单层权重 + 激活就可能超单卡显存。

**PP 按层切、DP 不省权重**;当**单层本身就放不下**,或想在层内并行加速时,就需要 TP —— 把单个矩阵乘切开。

### 1.2 TP 是什么

**张量并行(Tensor Parallelism,TP)**:把 transformer 层里**单个权重矩阵的矩阵乘**沿某个维度切开,分到 `tp` 张卡上并行算,算完用一次集合通信(all-reduce / reduce-scatter)拼回。它在**单层内部**做并行,所以与 PP(层间切分)、DP(批次切分)、EP(专家切分)正交。

TP 的两个基本算子是一对共轭(conjugate)的线性层:
- **`ColumnParallelLinear`**(列并行):权重按**输出维**切。
- **`RowParallelLinear`**(行并行):权重按**输入维**切。

一个 transformer 层把它们成对使用:**Column → (逐元素算子)→ Row**,使中间不需要通信,只在 Row 之后做一次规约。

### 1.3 TP 在并行体系中的位置

| 并行轴 | 切什么 | 峰值激活 | 权重 | 优化器 | 通信特征 |
|--------|--------|---------|------|--------|---------|
| **TP** | 单层权重矩阵 | 1/tp(开 SP) | 1/tp | 1/tp | **高频、量大、在关键路径** → 只宜机内 NVLink |
| EP | 专家(按个数) | ~1 | MoE 层 1/N | 1/N | 中 |
| PP | 层(按深度) | 1(VPP>1) | 1/N | 1/N | 中,点对点 |
| CP | 序列 | 1/N | 1 | 1/N | 中 |
| DP | 批次 | 1 | 1 | 1/N(分布式优化器) | 低 |

TP 的通信是**每层 4 次** all-reduce、量为 `O(s·b·h)`、且卡在计算关键路径上 —— 这决定了 **TP 必须留在单机 NVLink 域内**(典型 `tp ≤ 8`),跨机做 TP 会被网络打死。

### 1.4 TP 的收益

- **权重显存 `÷tp`**:每卡只存权重的 `1/tp`。
- **优化器状态 `÷tp`**:相应地 Adam 状态也 `÷tp`。
- **激活显存 `÷tp`**(需配合 SP,见 §7)。
- **算力聚合**:单层矩阵乘由 `tp` 卡并行,降低单层延迟。
### 1.5 记号约定

| 符号 | 含义 |
|------|------|
| `tp` | TP 度(`--tensor-model-parallel-size`) |
| `s` / `b` / `h` | 序列长度 / micro-batch / hidden size |
| `h_ffn` | FFN 中间维(通常 `4h` 或 `8h/3`(SwiGLU)) |
| `a` | attention 头数;`tp` 须整除 `a` |
| `f` / `g` | 一对共轭通信算子(见 §3) |
| SP | 序列并行(Sequence Parallelism),TP 的配套扩展 |

---

## 2. 为什么这么设计：把通信塞进反向的 GEMM 空档,而不是"算完再同步"

朴素做法很直接:每个并行线性层用 `torch.nn.Linear` 算本地 GEMM,前向后挂一次 all-reduce,反向让 autograd 正常产出 dgrad/wgrad,再对 dgrad 做一次 all-reduce。Megatron 没有走这条路 —— 它给 TP 线性层写了一个专用的 `torch.autograd.Function`。源码陈述了其中三条理由;第四条源码沉默,由本页重建并标为推断。

**① 反向里通信必须与 wgrad 同时在跑,所以必须自己写 backward。**
`linear_with_grad_accumulation_and_async_allreduce` 的 docstring 明写:dgrad 的 TP all-reduce「can be done asynchronously with the calculation of the weight gradients」;开 SP 时对应的 reduce-scatter「is done asynchronously with the calculation of the weight gradients」(`megatron/core/tensor_parallel/layers.py:691-697`)。要做到这点,backward 内部必须手工排成「先发起异步通信 → 再算 wgrad → 最后 `wait()`」:`LinearWithGradAccumulationAndAsyncCommunication`(`megatron/core/tensor_parallel/layers.py:470`)的反向先 `torch.distributed.all_reduce(grad_input, ..., async_op=True)`(`:571`)或 `reduce_scatter(..., async_op=True)`(`:583`),中间做 wgrad GEMM,把 `handle.wait()` 一路推到函数末尾(`:660-667`)。`nn.Linear` 的反向是一个不可拆的整体,没有这个插缝的位置。

**② 顺序靠什么保证:先试过插 noop 硬拖,最后改成一条环境变量约定。**
异步通信要真和 wgrad 重叠,前提是通信 kernel **先**被下发。源码把这个前提写死成 `CUDA_DEVICE_MAX_CONNECTIONS=1`,并在三处注释标出「Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the … is scheduled before the … computation」(`megatron/core/tensor_parallel/layers.py:553`、`:572`、`:585`);docstring 进一步说明这些集合通信「should be scheduled before compute kernels to overlap the communication with the computation, which is **necessary for a speedup but not for correctness** so that ordering isn't imposed by the scheduler」(`:699-706`)。
**被否掉的替代就写在历史里**:提交 `bdd973128`(2022-10-14,commit message 即「Remove noop used to try to force scheduling and check for environment variable instead」)之前,同样这三处写的是「Delay the start of … computation shortly (3us) to have gather scheduled first and have GPU resources allocated」,后面跟一行 `_ = torch.empty(1, device=grad_output.device) + 1` —— 用一个空 kernel 硬拖约 3 微秒去抢调度顺序。这条路被整段删除,换成「要求用户设环境变量,没设就只 `warnings.warn`」(`:761-775`)。
→ 决定取舍的判据是**把正确性和性能解耦**:重叠失败只掉吞吐、不出错,所以宁可把 kernel 顺序托付给一个全局环境变量,也不在热路径里留运行时 hack。

**③ wgrad 直接累加进已有梯度缓冲,而不是"先算完再加一次"。**
同一段 docstring 的开头就说明:这个实现「has the option to accumulate the result of backprop calculation into an existing gradient buffer, **preventing the need to do an additional addition kernel** after the gradient calculation」(`megatron/core/tensor_parallel/layers.py:686-689`)。这解释了 `gradient_accumulation_fusion` 为什么被做成 TP 线性层的一等形参而非事后优化 —— 缺少 APEX 的 `fused_weight_gradient_mlp_cuda` 扩展时构造期直接 `RuntimeError`(`:970-979`)。

**④ 为什么 Column→Row 成对,而不是每层 gather 回完整张量。**
`ColumnParallelLinear` 确实提供了 `gather_output` 开关(「If true, call all-gather on output and make Y available to all GPUs, otherwise, every GPU will have its output which is Y_i = XA_i」,`megatron/core/tensor_parallel/layers.py:797-799`),但**默认 `False`**(`:841`);`RowParallelLinear` 对称地提供 `input_is_parallel`(「If true, we assume that the input is already split across the GPUs and we do not split again」,`:1161-1163`)。两个默认值合起来,就是"列并行的切片输出直接喂给行并行、中间不通信"这条路径。

> [!note] 推断
> 源码只陈述了这两个开关的语义与默认值,**没有**写"我们成对使用是为了省掉中间那次 all-gather"。"`gather_output=True` 是被否掉的替代"这层意思由本页承担:打开它会在每个 MLP / attention 内部多一次 `O(s·b·h)` 的 all-gather,而 TP 的通信全部压在计算关键路径上(§8.3)。要引用这条判断,请回到 `megatron/core/tensor_parallel/layers.py:797-799`、`:841`、`:1161-1163` 这三个 locator,不要引用本段推断。

---

## 3. 核心机制:共轭算子 `f` / `g`

TP 的正确性建立在两个共轭的 autograd 算子上(`megatron/core/tensor_parallel/mappings.py`)。

**`f` 算子 —— `_CopyToModelParallelRegion`**(`megatron/core/tensor_parallel/mappings.py:201`):
```python
class _CopyToModelParallelRegion(torch.autograd.Function):
    def forward(ctx, input_, group):   return input_                       # 前向:恒等
    def backward(ctx, grad_output):    return _reduce(grad_output, ...)     # 反向:all-reduce
```

**`g` 算子 —— `_ReduceFromModelParallelRegion`**(`megatron/core/tensor_parallel/mappings.py:221`):
```python
class _ReduceFromModelParallelRegion(torch.autograd.Function):
    def forward(ctx, input_, group):   return _reduce(input_, group)       # 前向:all-reduce
    def backward(ctx, grad_output):    return grad_output                  # 反向:恒等
```

二者互为共轭:`f` 前向恒等、反向 all-reduce;`g` 前向 all-reduce、反向恒等。一个列并行区域用 `f` 开头、`g` 结尾,就能保证**前向和反向各恰好一次** all-reduce,且梯度数学上正确。

`megatron/core/tensor_parallel/mappings.py` 还提供 SP 版本(§7 用):`_ScatterToSequenceParallelRegion`、`_GatherFromSequenceParallelRegion`(反向 reduce-scatter)、`_ReduceScatterToSequenceParallelRegion`。

---

## 4. 算子① — ColumnParallelLinear(列并行)

`megatron/core/tensor_parallel/layers.py:784`。

### ①.1 动机与切分方式

权重 `A`(形状 `[in, out]`)**按输出维切** `A = [A₁ | A₂ | … | Aₜ]`,每卡持 `Aᵢ` 形状 `[in, out/tp]`。输入 `X` 在每卡上是**完整副本**,各卡算 `Y_i = X·Aᵢ`,得到输出的一个切片 `[*, out/tp]`。

动机:列并行的**输出天然是切开的**,正好喂给下一个行并行层,中间无需通信 —— 这是 Column→Row 配对的根基。

### ①.2 源码与流程

```python
def forward(self, input_, ...):                                # megatron/core/tensor_parallel/layers.py:1000
    if self.sequence_parallel or self.allreduce_dgrad or ...:
        input_parallel = input_                                # SP / 已并行:直接用
    else:
        input_parallel = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)  # f 算子
    output_parallel = self._forward_impl(input_parallel, weight, ...)   # 本地 GEMM:X · Aᵢ
    if gather_output:
        output = gather_from_tensor_model_parallel_region(output_parallel, ...)  # 可选 all-gather
    else:
        output = output_parallel                               # 默认:输出保持切开
    return output, output_bias
```

```
输入 X(完整副本)
   │  f 算子(前向恒等;反向 all-reduce 汇总各卡 dgrad)
   ▼
本地 GEMM:Yᵢ = X · Aᵢ           ← tp 卡并行
   │
   ▼
输出 Yᵢ(切片 [*, out/tp])—— 默认不 gather,直接交给下游 RowParallel
```

### ①.3 适用位置

- **FFN 的 fc1**:`[h → h_ffn]`,列并行后输出 `[*, h_ffn/tp]`。
- **attention 的 QKV 投影**:按 head 切,每卡得 `a/tp` 个头。
- **输出层 embedding**(`VocabParallelEmbedding`,词表维切分)。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> **非融合 vocab-parallel 交叉熵现在显式接收 TP 组**(#5128,`megatron/core/tensor_parallel/cross_entropy.py:213`、`megatron/core/models/common/language_module/language_module.py:184`)。
> 词表并行(输出投影按词表维切到各 TP rank)后,损失计算 `vocab_parallel_cross_entropy` 要在 TP 组内做 **3 次 all-reduce**(`logits_max` 取 MAX、`predicted_logits`/`sum_exp_logits` 取 SUM)以跨分片拼出全词表 softmax 的分母。原实现把通信组写死为全局 `get_tensor_model_parallel_group()`;此 PR 给 `vocab_parallel_cross_entropy(..., tp_group=None)` 增加可选 `tp_group` 形参(`megatron/core/tensor_parallel/cross_entropy.py:217`),并改用 `get_pg_rank/get_pg_size(tp_group)` 取 rank/world_size(`:135-136`,替换原 `get_tensor_model_parallel_rank/world_size`)。`LanguageModule.compute_language_model_loss` 现传入 `self.tp_group`(`megatron/core/models/common/language_module/language_module.py:184`)。
> **意义**:与非均匀/异构 TP(每层 TP 组可不同,见 [[25_megatron_nonuniform_tp_analysis]])解耦 —— CE 的 all-reduce 不再强制走全局 TP 组,而跟随调用方实际的 TP 子组;`tp_group=None` 时回退全局组,旧行为不变。融合实现(`fused_vocab_parallel_cross_entropy`)早已通过 `self.pg_collection.tp` 传组,本 PR 把非融合路径对齐。

---

## 5. 算子② — RowParallelLinear(行并行)

`megatron/core/tensor_parallel/layers.py:1148`。

### ②.1 动机与切分方式

权重 `B`(形状 `[in, out]`)**按输入维切** `B = [B₁; B₂; …; Bₜ]`,每卡持 `Bᵢ` 形状 `[in/tp, out]`。输入 `X` 本身已经是切开的 `X = [X₁ | X₂ | … | Xₜ]`(来自上游列并行),各卡算 `Yᵢ = Xᵢ·Bᵢ`,然后 **all-reduce 求和** `Y = ΣYᵢ`。

动机:行并行的**输入正好接列并行的切片输出**,且只需在最后做一次规约。

### ②.2 源码与流程

```python
def forward(self, input_):                                      # megatron/core/tensor_parallel/layers.py:1314
    input_parallel = input_                                     # 输入已是并行切片
    output_parallel = self._forward_impl(input_parallel, self.weight, ...)  # 本地 GEMM:Xᵢ · Bᵢ
    if self.sequence_parallel:
        output_ = reduce_scatter_to_sequence_parallel_region(output_parallel, ...)  # SP:reduce-scatter
    else:
        output_ = reduce_from_tensor_model_parallel_region(output_parallel, ...)    # g 算子:all-reduce
    return output_ + bias, output_bias
```

```
输入 Xᵢ(切片,来自上游 ColumnParallel)
   │
   ▼
本地 GEMM:Yᵢ = Xᵢ · Bᵢ           ← tp 卡并行
   │  g 算子(前向 all-reduce 求和 ΣYᵢ;反向恒等)
   ▼
输出 Y(完整副本)
```

### ②.3 适用位置

- **FFN 的 fc2**:`[h_ffn → h]`,输入是 fc1 的切片输出。
- **attention 的输出投影**:输入是各卡 `a/tp` 头的 attention 结果。

---

## 6. 组合应用:一个 transformer 层的 TP

### 6.1 MLP 的 TP(Column → 激活 → Row)

```
        X [s,b,h]  (完整副本)
          │  f
          ▼
   ┌─────────────────┐
   │ fc1 = Column     │  权重 [h, h_ffn] 切成 [h, h_ffn/tp]
   │  Yᵢ = X·A1ᵢ      │
   └─────────────────┘
          │  输出 [s,b,h_ffn/tp](切片)
          ▼
   GeLU / SwiGLU      ← 逐元素算子,在切片上直接做,无需通信 ✅
          │  [s,b,h_ffn/tp]
          ▼
   ┌─────────────────┐
   │ fc2 = Row        │  权重 [h_ffn, h] 切成 [h_ffn/tp, h]
   │  Zᵢ = Yᵢ·A2ᵢ     │
   └─────────────────┘
          │  g(all-reduce 求和)
          ▼
        Z [s,b,h]  (完整副本)
```

关键:fc1 列并行 → 激活函数在切片上逐元素算 → fc2 行并行。**整个 MLP 前向只有 1 次 all-reduce(fc2 的 `g`),反向只有 1 次(fc1 的 `f`)**。

### 6.2 Attention 的 TP(按 head 切)

QKV 投影 = ColumnParallel,按 attention head 切 → 每卡 `a/tp` 个头;每卡独立算这 `a/tp` 个头的完整 self-attention(头之间本就独立);输出投影 = RowParallel,all-reduce 汇总。同样**前向 1 次、反向 1 次** all-reduce。

### 6.3 一层的通信账

| 部位 | 前向 all-reduce | 反向 all-reduce |
|------|----------------|----------------|
| Attention | 1(输出投影 `g`) | 1(QKV `f`) |
| MLP | 1(fc2 `g`) | 1(fc1 `f`) |
| **合计/层** | **2** | **2** |

每次 all-reduce 量为 `O(s·b·h)`。`L` 层模型一步前向+反向共 `4L` 次 all-reduce —— **高频、在关键路径**,这就是 TP 必须 NVLink 的根本原因。

---

## 7. Sequence Parallelism(SP)—— TP 的配套扩展

### 7.1 动机:TP 没省的那块激活

TP 把 4 个大矩阵乘的权重与中间激活切成 `1/tp`,但 transformer 层里还有**不含矩阵乘的部分**:LayerNorm、Dropout、残差加。这些区域的激活在朴素 TP 下是**每卡完整副本**(因为 `f` 把输入复制到了每卡)→ 这部分激活显存没被 TP 省下。

**SP 的动机**:把这些区域的激活**沿序列维 `s` 切开**,分到 `tp` 卡,每卡只存 `s/tp`。于是**整层激活都 `÷tp`**。

### 7.2 实现:把 all-reduce 换成 reduce-scatter + all-gather

SP 区域(序列切分)与 TP 区域(权重切分)交界处,通信算子改变:
- 进入 TP 列并行区:`f`(复制)→ **all-gather**(把序列维拼全)。`_GatherFromSequenceParallelRegion`(`megatron/core/tensor_parallel/mappings.py:300`,反向是 reduce-scatter)。
- 离开 TP 行并行区:`g`(all-reduce)→ **reduce-scatter**(求和的同时把序列维散开)。`reduce_scatter_to_sequence_parallel_region`(RowParallel `forward` 的 `sequence_parallel` 分支,`megatron/core/tensor_parallel/layers.py:1359`)。

**通信量不变**:数学上 `all-reduce = reduce-scatter + all-gather`,SP 只是把一次 all-reduce 拆成两半,分别放在区域两端。所以 **SP 省显存但不增加通信量** —— 几乎是"免费"的,生产配置默认开(`--sequence-parallel`)。

```
朴素 TP:            LayerNorm/Dropout/残差 区域激活 = 每卡完整 [s,b,h]
开 SP:              同区域激活 = 每卡 [s/tp,b,h]                ← 激活 ÷tp
通信:              all-reduce  ──拆成──→  all-gather + reduce-scatter(总量相同)
```

### 7.3 TP Communication Overlap

`--tp-comm-overlap`(TE 特性,需 SP):把 all-gather / reduce-scatter 与相邻 GEMM 重叠 —— 无依赖的部分**批量重叠**(bulk),有依赖的**流水重叠**(pipelined)。把 TP 通信尽量从关键路径上藏掉。要求 `tp ≥ 2` 且开 SP。

---

## 8. 约束与开销

记单层权重 `W_layer ≈ 12h²`,激活 `A_layer ∝ s·b·h`。

### 8.1 显存

| 项 | 朴素 TP | TP + SP |
|----|---------|---------|
| 权重 / 卡 | `W_layer / tp` | `W_layer / tp` |
| 优化器状态 / 卡 | `÷tp` | `÷tp` |
| 矩阵乘中间激活 | `÷tp` | `÷tp` |
| LN/Dropout/残差 区激活 | **×1(未省)** | **`÷tp`** ✅ |

结论:**TP 把权重/优化器/矩阵乘激活都 `÷tp`;但只有再加 SP,才能把整层激活都 `÷tp`**。

### 8.2 通信量

- 每层 **4 次** 集合通信(前向 2 + 反向 2),每次 `O(s·b·h)`。
- 一次 all-reduce 的总线流量 ≈ `2(tp-1)/tp · s·b·h`。
- SP 不改变总量(all-reduce = RS + AG)。
- **关键:这些通信在计算关键路径上**,无法像 PP/DP 那样靠流水线隐藏(只能靠 `--tp-comm-overlap` 部分重叠)。

### 8.3 "气泡" —— TP 没有流水线气泡,但有通信暴露

TP 不像 PP 有流水线气泡。它的低效来自:**每层 4 次通信的延迟暴露在关键路径上**。设单层计算 `T_comp`、通信 `T_comm`,则一层壁钟 ≈ `T_comp + T_comm`(未重叠)。`tp` 越大,`T_comp` 越小但 `T_comm` 不降反升(`2(tp-1)/tp` 随 `tp` 升)→ **TP 的强扩展性有上限**,`tp` 过大时通信占比失控。这也是"`tp ≤ 8`、留在 NVLink 域"的量化原因。

### 8.4 整除性约束

- `tp` 必须整除 attention 头数 `a`(按 head 切)。
- `tp` 必须整除 FFN 中间维、词表大小等。
- SP 要求序列长度能被 `tp` 整除。

### 8.5 为什么 TP 不能跨机

TP 把一次矩阵乘拆成 `tp` 份,每份算完必须立刻通信拼回(否则下一步算不了)。这次通信:
- **在关键路径上**:计算流必须等它,无法靠流水线隐藏(不像 PP/DP)。
- **频率高**:每个 transformer 层 4 次(attn 2 次 + MLP 2 次)。
- **量大**:每次 `O(s·b·h)`。

NVLink(数百 GB/s)能扛;跨机 IB(数十 GB/s)扛不住。所以工业配置永远是 **TP 在机内、PP 跨机**。

### 8.6 前提、代价与故意不做的事

上面的开销表只在下列前提成立时才是这个形状。每条都能落到一个 `file:line`,越出前提就不再适用:

| # | 前提 / 不变量 | 源码落点 | 破坏后的表现 |
|---|---|---|---|
| 1 | `CUDA_DEVICE_MAX_CONNECTIONS=1` | `megatron/core/tensor_parallel/layers.py:761-775` | 不报错,只 `warnings.warn`;通信与 wgrad 不再重叠,§8.2 那 4 次通信重新完整暴露在关键路径上 |
| 2 | `sequence_parallel` 与 `allreduce_dgrad` 互斥 | 构造期 `RuntimeError`(`:982-985`),反向里再 `assert not ctx.allreduce_dgrad`(`:576`) | 同开直接抛错 —— SP 与朴素 TP 是二选一,不是叠加 |
| 3 | 开 SP 时下游行并行层必须 `input_is_parallel=True` | `RuntimeError`(`:1215-1216`) | 否则 SP 建不起"切片进 / 切片出"的链路 |
| 4 | `tp == 1` 时 SP 被静默关闭 | `:959-964`,`warnings.warn` 后置 `self.sequence_parallel = False` | 单卡跑通不等于 SP 路径被覆盖到 |
| 5 | 切分维度必须整除 | `divide()` → `ensure_divisibility` 的 `assert`(`megatron/core/utils.py:557-566`);调用点 `megatron/core/tensor_parallel/layers.py:876`(输出维)、`:1227`(输入维) | 不整除直接 assert 失败,没有自动 padding 兜底 |
| 6 | `gradient_accumulation_fusion` 依赖 APEX 扩展 | `RuntimeError`(`:970-979`) | 缺扩展时必须显式关掉,§2③ 那条"省一次加法 kernel"的收益随之消失 |

**代价**:§2① 的重叠是拿"手工排 backward"换来的 —— TP 线性层的反向不再是 autograd 默认路径,任何改动都要同时照顾 `sequence_parallel` / `allreduce_dgrad` / `gradient_accumulation_fusion` / `wgrad_deferral_limit` 四条支路(形参清单见 `megatron/core/tensor_parallel/layers.py:672-682`)。

**故意不做**:权重被冻结(`weight.requires_grad` 为假)时,`_forward_impl` 整个绕开这套机制,改走 `linear_with_frozen_weight`(`megatron/core/tensor_parallel/layers.py:994-998`)。也就是说 **TP"把通信藏进 wgrad"的红利只对可训练权重成立**;冻结主干 + 只训适配器一类的场景,反向里根本没有 wgrad 可用来掩盖 dgrad 通信。

---

## 9. 适用场景及原因

| 场景 | 是否用 TP | 原因 |
|------|----------|------|
| 单层能放进单卡 | ❌ 不用 | TP 引入高频关键路径通信,纯亏;用 DP |
| 单层放不下 / 想降单层延迟 | ✅ 用 TP | 唯一能在层内并行的手段 |
| 跨机扩展 | ❌ TP 不跨机 | 通信扛不住 IB;改用 PP 跨机 |
| MoE 专家层 | ⚠️ 优先 EP | 细粒度专家被 TP 切碎,GEMM 效率低(见 `14_megatron_ep_analysis.md`) |

**经验法则(见 MoE README Guidelines)**:
- `TP × (EP) ≤ 单机卡数`,吃满 NVLink。
- TP 尽量小:`tp` 越大通信占比越高;先把 `tp` 压到"刚好装下"。
- **永远配合 SP**(`--sequence-parallel`):免费省激活。
- 跨机用 PP,不要扩 TP。
- 典型大模型配置:`TP(机内 NVLink)× PP(跨机 IB)× DP(最外层)`,MoE 再加 `EP`。

### 9.1 一句话总结

- **TP 的本质**:把单层的 4 个大矩阵乘切开,用一对共轭算子 `f/g` 保证前向/反向各一次 all-reduce。
- **Column→Row 配对**:列并行输出天然切开,正好喂行并行,中间逐元素算子无需通信。
- **SP**:把 TP 没省到的 LN/Dropout/残差区激活也沿序列切开,把 all-reduce 拆成 RS+AG,**免费**把整层激活 `÷tp`。
- **代价**:每层 4 次关键路径通信 → TP 只能机内 NVLink,且 `tp` 不宜过大。

---

## 10. 发展趋势

> [!note] 推断:锚点是基线 `71092579` 下的源码事实(废弃标记、TODO、新模块),方向判断由本页承担,不是源码的自陈计划。

**一、TP 通信重叠的"粒度旋钮"正在从 Megatron 侧退场。** `tp_comm_split_ag` / `tp_comm_atomic_ag` / `tp_comm_split_rs` / `tp_comm_atomic_rs` 四个字段的 docstring 第一行都是「Deprecated from TransformerEngine v1.6.0」(`megatron/core/model_parallel_config.py:287`、`:293`、`:299`、`:305`),而 `tp_comm_overlap` 本体(`:255`)与 `tp_comm_overlap_disable_qkv` / `tp_comm_overlap_disable_fc1`(`:325`、`:330`)这类"整体开关 + 逐 GEMM 关闭"的旋钮仍然在用。**由此可推断**:§7.3 里 bulk / pipelined 的粒度选择正在整体沉进 TE,Megatron 侧只保留"要不要重叠、哪几个 GEMM 不重叠"这一层;调参时不应再依赖 split/atomic 这组字段。

**二、`f`/`g` 的集合通信在推理路径上正被 NVLS 单边原语替换。** 基线下新增 `megatron/core/tensor_parallel/inference_layers.py`,其中 `inference_all_gather_from_tensor_model_parallel_region`(`:498`)与 `inference_reduce_scatter_to_sequence_parallel_region`(`:533`)的 docstring 直接写明它们「Replaces `reduce_scatter_to_sequence_parallel_region` in inference paths where autograd is not needed and NVLS symmetric-memory is available」(`:536-539`),底层换成 `multimem_all_gather` / `multimem_reduce_scatter`,还有更激进的 `fused_multimem_rs_add_norm_ag`(`:15`),NCCL 只作 fallback;`InferenceRowParallelLinear`(`:351`)是它的使用方。该文件里还挂着一条未完成项:「TODO(ksanthanam): Refactor InferenceRowParallelLinear._matmul_reduce_scatter to use this function for its non-fused NVLS reduce-scatter path」(`:541`)。**由此可推断**:§3 那对共轭算子的形态在训练侧仍然成立,但"TP 的 AG/RS 就是 NCCL 集合通信"这个前提已经在推理侧被 symmetric memory 的单边 multimem 打破;训练侧是否跟进,源码没有表态。

**三、TP 组正在从"全局唯一"变成"传进来的参数"。** §①.3 那条 `[!update]` 记录的 #5128 把 `vocab_parallel_cross_entropy` 的通信组从写死的 `get_tensor_model_parallel_group()` 改成可选形参(`megatron/core/tensor_parallel/cross_entropy.py:217`),并改用 `get_pg_rank/get_pg_size(tp_group)` 取 rank/world size(`:135-136`)。同一方向上,`ColumnParallelLinear` / `RowParallelLinear` 也都已接受 `tp_group` 形参并用 `get_tensor_model_parallel_group_if_none` 兜底(`megatron/core/tensor_parallel/layers.py:871-873`、`:1220-1222`)。**由此可推断**:"一个进程只属于一个 TP 组"这个隐含假设正在被逐点拆除(另见 [[25_megatron_nonuniform_tp_analysis]]),后续读到引用 TP 组的代码时,应默认它是**调用方传入**的而不是全局的。

---

*生成依据:`Megatron-LM` `dev` 分支 `85902ef599ea4eb06ada7567a479c524b605767a`(2026-09-01;由 `71092579` 重定基线而来,更早一次为 2026-08-28 由 `ee3f1ff` 推进)。源码行号以该 commit 为准。配套文档:`15_megatron_pp_schedulers_analysis.md`、`14_megatron_ep_analysis.md`、`13_megatron_cp_analysis.md`。*

---

## 配置契约：embedding 侧的一个补充字段

本节补一个此前零提及、与序列并行下 embedding 输出处理相关的字段。



### `TransformerConfig`（`megatron/core/transformer/transformer_config.py`，1 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `clone_scatter_output_in_embedding` | `bool` | `True` | When set to True, clone the output of scatter_to_sequence_parallel_region in embedding layer to facilitate garbage collection of input. | `:1320` |

> 该类共 266 个字段，本表收 1 项；其余 265 项已在别处归属：主要归 [[10_megatron_model_structure_analysis]] 92 项、[[14_megatron_ep_analysis]] 38 项、[[23_megatron_precision_cudagraph_fusion_analysis]] 38 项、[[21_megatron_fusion_operators_analysis]] 26 项，另散见 19 页（完整归属见 `docs/coverage/megatron-lm.yaml`）。

## Related Pages

- [[15_megatron_pp_schedulers_analysis]] · [[14_megatron_ep_analysis]] · [[13_megatron_cp_analysis]] · [[16_megatron_distributed_optimizer_analysis]] · [[17_megatron_parallelism_orchestration_analysis]]
- [[25_megatron_nonuniform_tp_analysis]] · [[20_megatron_comm_overlap_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]


