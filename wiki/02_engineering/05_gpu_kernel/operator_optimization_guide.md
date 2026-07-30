# 算子调优体系指南

> **文档说明**：本文档系统整理 GPU（NVIDIA）与昇腾 NPU 两个平台的算子开发与性能优化方法，所有核心结论均注明参考来源。适用对象：深度学习框架算子开发工程师、模型推理/训练优化工程师。

## 文档结构与阅读路径

> **新增导航**：本文是 GPU/NPU 对比型指南，章节按"通用理论 → 分平台深入 → 跨平台通用方法 → 工作流对比"组织，便于按平台或按主题检索。

**① 结构**：

- **§1-2 通用优化理论**：算子编程体系概览与 Roofline 模型（架构概览、性能分析框架），GPU/NPU 通用。
- **§3 深入 GPU 路径**：Memory Bound / Compute Bound 的 GPU 具体优化手段。
- **§6 深入 NPU（昇腾）路径**：Da Vinci 架构、AscendC 编程模型与 GPU 经验的适配分析。
- **§4-5 两平台通用方法**：融合算子的识别与设计、等价替换的寻找方法，GPU/NPU 均适用。
- **§8 完整工作流对比**：GPU 与 NPU 的端到端优化工作流对照。

**② 阅读建议**：

- **GPU 用户**：读 §1-5 + §3 + §8.1。
- **NPU 用户**：读 §1-2、§4-6 + §8.2。
- **框架设计者**：建议按序通读全文。

**③ 一句话**：本文既给出通用的性能分析框架（Roofline、Memory/Compute 分类、融合与等价替换），也分述 GPU 与 NPU（昇腾）的具体优化路径，便于跨平台迁移经验。

---

## 目录

1. [算子编程体系概览](#1-算子编程体系概览)
2. [性能分析基础：Roofline 模型](#2-性能分析基础roofline-模型)
3. [GPU 算子优化路径](#3-gpu-算子优化路径)
   - 3.1 Memory Bound 优化
   - 3.2 Compute Bound 优化
4. [融合算子的识别与设计](#4-融合算子的识别与设计)
5. [等价替换的寻找方法](#5-等价替换的寻找方法)
6. [昇腾 NPU 算子优化路径](#6-昇腾-npu-算子优化路径)
   - 6.1 硬件架构差异
   - 6.2 AscendC 编程模型
   - 6.3 GPU 经验的适配分析
7. [与 torch.compile 的关系](#7-与-torchcompile-的关系)
8. [完整优化工作流](#8-完整优化工作流)
9. [参考文献](#9-参考文献)

---

## 1. 算子编程体系概览

### 1.1 GPU 算子编程层次

GPU 算子开发形成了从底层到高层的完整层次结构：

| 层次 | 代表框架/工具 | 抽象粒度 | 典型使用场景 |
|------|------------|---------|------------|
| 汇编层 | PTX / SASS | 指令级 | 极端性能优化，极少直接使用 |
| CUDA C++ | CUDA Runtime | Thread / Warp / Block | 通用 kernel 开发 |
| 模板抽象层 | CUTLASS | Tile（Block Tile → Warp Tile → Thread Tile） | 矩阵乘、卷积等高性能计算 |
| DSL 层 | Triton | Block（SRAM 粒度） | Attention、Softmax、自定义 reduction |
| DSL 层 | TileLang | Tile + 硬件 primitive 注入 | 介于 Triton 和 CUTLASS 之间 |
| 编译器层 | TVM / XLA | 计算图 IR | 跨硬件自动调优 |
| 算子库 | cuBLAS / cuDNN | 黑盒函数调用 | 标准 GEMM / Conv |

> **出处**：CUTLASS 架构设计见 NVIDIA CUTLASS 官方文档（github.com/NVIDIA/cutlass）；Triton 编程模型见 Tillet et al., *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*, MLSys 2019；TileLang 见 Microsoft Research 开源仓库（github.com/microsoft/TileLang）。

### 1.2 昇腾 NPU 算子编程路径

| 路径 | 工具 | 状态 | 说明 |
|-----|------|------|------|
| 主推路线 | AscendC | 当前主推 | C++ 算子编程框架，显式管理存储与流水线 |
| 旧路线 | TBE（Tensor Boost Engine） | 维护模式 | 基于 TVM 改造的 Python DSL |
| 预置库 | CANN 算子库 + torch_npu | 开箱即用 | 不支持自定义，扩展性差 |
| 研究阶段 | Triton for Ascend | 探索中 | 华为适配中，尚未生产就绪 |

> **出处**：华为 CANN AscendC 编程指南（docs.hiascend.com）；华为开发者论坛，AscendC 技术博客，2023。

---

## 2. 性能分析基础：Roofline 模型

Roofline 模型是算子优化的首要分析工具，用于判断瓶颈类型，指导优化方向。

> **出处**：Williams, Patterson, Waterman, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, Communications of the ACM, 2009.

### 2.1 核心公式

```
实际性能上界 = min( Peak_FLOPS,  AI × Peak_BW )

其中：
  AI（算术强度）= kernel 实际执行的 FLOPs / 实际 HBM 访存 Bytes
  Ridge Point  = Peak_FLOPS / Peak_BW  ←  Memory Bound / Compute Bound 分界线
```

**Ridge Point 的物理含义**：单位带宽对应的最大算力。AI 低于 Ridge Point，性能受带宽限制；AI 高于 Ridge Point，性能受算力限制。

### 2.2 主流硬件参考参数

（以下对比 GPU（NVIDIA）与 NPU（昇腾）两类加速器）

| 硬件 | Peak FLOPS（fp16） | Peak HBM BW | Ridge Point |
|------|-------------------|-------------|-------------|
| NVIDIA A100 SXM4 | 312 TFLOPS | ~2 TB/s | ~156 FLOPs/Byte |
| NVIDIA H100 SXM5 | 989 TFLOPS | ~3.35 TB/s | ~295 FLOPs/Byte |
| 昇腾 910B | ~256–320 TFLOPS | ~900 GB/s | ~333 FLOPs/Byte |

> **出处**：NVIDIA A100 规格：NVIDIA A100 Tensor Core GPU Architecture Whitepaper, 2020；H100 规格：NVIDIA H100 Tensor Core GPU Architecture Whitepaper, 2022；昇腾 910B 规格：华为昇腾官方产品规格（hiascend.com/hardware/ascend-910B）。

### 2.3 常见算子的理论算术强度

| 算子 | 理论 AI 估算（fp16） | A100 判断 | 昇腾 910B 判断 |
|-----|-------------------|----------|--------------|
| Elementwise（Add、ReLU） | 0.5–1 FLOPs/Byte | Memory Bound | Memory Bound |
| Softmax（N 元素） | 3–5 FLOPs/Byte | Memory Bound | Memory Bound |
| LayerNorm | 4–8 FLOPs/Byte | Memory Bound | Memory Bound |
| GEMM（M=N=K=4096, fp16） | ~136 FLOPs/Byte | 接近 Memory Bound | Memory Bound |
| GEMM（M=N=K=8192, fp16） | ~273 FLOPs/Byte | Compute Bound | Memory Bound |
| Large Conv（large batch） | 100+ FLOPs/Byte | Compute Bound | 视尺寸而定 |

> **GEMM AI 手算方法**（出处：Roofline 模型原论文）：
> ```
> FLOPs = 2 × M × N × K
> Bytes = 2 × (M×K + K×N + M×N) × sizeof(dtype)
> AI = FLOPs / Bytes
> ```

**重要观察**：昇腾 910B 的 Ridge Point（~333）远高于 A100（~156），意味着在昇腾上更多算子落在 Memory Bound 区间，算子融合的优化收益更为显著。

### 2.4 Profiling 工具与关键指标

#### GPU（NVIDIA Nsight Compute）

```bash
# 采集命令
ncu --set full \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    -o profile_output \
    python your_script.py
```

**Memory Bound 时关注的指标：**

| 指标名 | 含义 | 目标值 |
|-------|------|--------|
| `Memory Throughput %` | 实测 HBM BW / 峰值 HBM BW | **≥ 80%** 说明接近最优 |
| `dram__bytes_read.sum + dram__bytes_write.sum` | 实际 HBM 读写总字节数 | 对比理论下界（见下） |
| `l2_global_hit_rate` | L2 命中率 | 低说明数据局部性差 |
| `sectors_not_aligned` | 非对齐访存次数 | 趋近 0 |

**Compute Bound 时关注的指标：**

| 指标名 | 含义 | 目标值 |
|-------|------|--------|
| `Compute Throughput %` | 实测算力 / 峰值算力 | **≥ 70%** |
| `sm__inst_executed_pipe_tensor_op_hmma.sum` | Tensor Core 指令数 | 占计算指令比例 ≥ 90% |
| `Warp Occupancy` | 活跃 warp / SM 最大 warp | **≥ 50%** |
| `stall_mio_throttle %` | 因访存等待导致的停顿 | **< 20%** |
| `shared_load_transactions_per_request` | SRAM bank conflict | 趋近 1.0 |

> **出处**：NVIDIA Nsight Compute User Guide, Counter Reference 章节（docs.nvidia.com/nsight-compute）。

**理论访存下界计算方法：**

```
以 Softmax（N 个 fp16 元素）为例：
  理论下界 = 1次读 + 1次写 = 2 × N × 2 Bytes
  若实测 dram__bytes = 6×N×2（3 pass），说明仍有 3× 优化空间
```

#### 昇腾（MindStudio + msprof）

```bash
# 命令行采集
msprof --output=/path/to/output \
       --application="python train.py" \
       --aic-metrics=ALL

# Python 插桩
import torch_npu
torch_npu.npu.profiler.start()
# ... your code ...
torch_npu.npu.profiler.stop()
```

| 昇腾指标 | 含义 | 对应 GPU 指标 | 目标值 |
|---------|------|-------------|--------|
| `AI Core 利用率` | AI Core 工作时间 / 总时间 | `Compute Throughput %` | ≥ 70% |
| `Cube 利用率` | Cube Unit 工作时间占比 | `tensor_op_hmma 利用率` | GEMM 类 ≥ 70% |
| `Vector 利用率` | Vector Unit 工作时间占比 | `FFMA 利用率` | Elementwise 类 ≥ 70% |
| `MTE 搬运时间占比` | DMA 搬运时间 / 总时间 | `stall_mio_throttle %` | 应与计算重叠 |
| `HBM 带宽利用率` | 实测 BW / 峰值 BW | `Memory Throughput %` | Memory Bound 时 ≥ 80% |
| `AI CPU 占比` | 退化到 CPU 执行的算子比例 | 无对应 | **趋近 0（优先级最高）** |

> **出处**：华为 MindStudio 性能分析工具用户指南（hiascend.com/document/detail/zh/mindstudio）。

---

## 3. GPU 算子优化路径

### 3.1 Memory Bound 优化

#### 策略一：算子融合（最高优先级）

**原理**：相邻 Memory Bound 算子的中间结果写回 HBM 再读取，产生冗余 HBM 流量。融合后中间结果留在寄存器或 SRAM，HBM 读写次数从 O(kN) 降至 O(N)。

```
朴素实现（3 个 kernel，HBM 读写 6 次）：
  x → [Add+bias] → HBM → [LayerNorm] → HBM → [GeLU] → HBM

融合实现（1 个 kernel，HBM 读写 2 次）：
  x → [Add+bias+LayerNorm+GeLU] → HBM（寄存器传递中间值）
```

**验证方法**：融合前后对比 `dram__bytes.sum`，应下降 k 倍（k = 融合算子数）。

**实现工具**：
- **Triton**：手写融合 kernel，寄存器传递中间结果
- **torch.compile + Inductor**：自动进行 pointwise fusion
- **CUTLASS Epilogue Fusion**：在矩阵乘输出寄存器上直接接 bias add、激活函数

> **出处**：Inductor 自动融合机制见 PyTorch 官方博客 *Introducing TorchInductor*, Meta AI, 2022；CUTLASS epilogue fusion 见 NVIDIA CUTLASS 官方文档，Epilogue 章节。

#### 策略二：减少访存 Pass 数（算法变形）

**Online Softmax**（3 pass → 1 pass）：

利用递推关系，在一遍扫描中同时维护 `running_max` 和 `running_sum`，消除额外的 HBM 读取：

```
朴素实现（3 遍扫描，每元素读 3 次）：
  pass1: max_x = max(x)
  pass2: exp_x = exp(x - max_x)
  pass3: out   = exp_x / sum(exp_x)

Online Softmax（1 遍扫描，每元素读 1 次）：
  for x_i in x:
      m_new = max(m, x_i)
      d = d * exp(m - m_new) + exp(x_i - m_new)
      m = m_new
```

FlashAttention（Dao et al., NeurIPS 2022）正是将 Online Softmax 与分块 Tiling 结合，将整个 Attention 的 HBM 访问从 O(N²) 降至 O(N)。

**Welford 在线方差**（LayerNorm 用，2 pass → 1 pass）：

一遍扫描同时计算 mean 和 variance，避免两次 HBM 读取。

> **出处**：Online Softmax：Milakov & Gimelshein, *Online normalizer calculation for softmax*, arXiv:1805.02867, 2018；FlashAttention：Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022；Welford 算法：Welford, *Note on a method for calculating corrected sums of squares and products*, Technometrics, 1962。

#### 策略三：访存向量化（128-bit Load）

**原理**：GPU 的 L2 cache line 为 128 Bytes。使用 `LDG.128`（float4，一次 load 16 Bytes）比 `LDG.32` 减少 L2 transaction 次数 4 倍，降低 warp 访存等待停顿。

**Triton 中的触发条件**：将 `BLOCK_SIZE` 设置为 128 的整数倍，且数据首地址 16-byte 对齐，编译器自动生成向量化 load。

**验证方法**：Nsight 中 `sectors_not_aligned` 趋近 0，`Memory Throughput %` 提升。

> **出处**：Tillet et al., MLSys 2019，Section 3（Block-level vectorization 描述）；CUDA Best Practices Guide，Memory Optimizations 章节（docs.nvidia.com）。

#### Memory Bound 成功标准

```
✓ Memory Throughput %  ≥ 80% × BW_peak
✓ dram__bytes          接近理论下界（= 算法必须读写的最小字节数）
✓ kernel 启动次数       减少（融合后）
✓ Wall time            缩短比例与 DRAM bytes 减少比例基本一致
```

---

### 3.2 Compute Bound 优化

#### 策略一：Tiling 参数调优

**原理**：GEMM 的 Tile 尺寸（BM × BN × BK）决定每块 A/B tile 在 SRAM 中的复用次数，进而决定算术强度。Tile 越大，复用越多，但寄存器和 SRAM 占用越多，Occupancy 越低，需要找 Pareto 最优点。

```
A100 SRAM 容量：192 KB / SM
典型 fp16 GEMM tile 配置：
  BM=128, BN=128, BK=32 → SRAM 占用 ≈ 16 KB（常用基准）
  BM=128, BN=128, BK=64 → SRAM 占用 ≈ 32 KB（更高复用，寄存器压力增大）

算术强度近似公式：AI ≈ BM × BN / (BM + BN)
```

**AutoTuning 工具**：
- **Triton** `@triton.autotune`：枚举 (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps) 组合
- **TVM MetaSchedule**：搜索 tiling / unroll / vectorize 策略空间
- **CUTLASS Profiler**：列出所有等价 GEMM 实现并 benchmark

> **出处**：Tillet et al., MLSys 2019；Chen et al., *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning*, OSDI 2018。

#### 策略二：软件流水线（Double/Triple Buffer）

**原理**：HBM → SRAM 的 `cp.async`（异步复制）延迟约 300–500 个 cycle（Ampere 架构）。软件流水线让当前 tile 的计算与下一 tile 的加载同时进行，隐藏访存延迟。

```
无流水线（串行，Tensor Core 大量空闲）：
  [Load T0] → [Compute T0] → [Load T1] → [Compute T1] → ...

双缓冲（num_stages=2，Load/Compute 完全重叠）：
  [Load T0]
            [Load T1] [Compute T0]
                      [Load T2] [Compute T1]
                                ...
```

**工具对应**：
- **CUTLASS**：`num_stages` 参数控制缓冲级数（通常 2–4）
- **Triton**：`num_stages` 参数 + `tl.load` 的 async hint

**验证方法**：`stall_mio_throttle %` 显著下降，`Compute Throughput %` 提升。

> **出处**：NVIDIA CUTLASS 官方文档，Software Pipelining 章节；NVIDIA Hopper 架构白皮书（异步 cp.async 描述），2022。

#### 策略三：确保 Tensor Core（HMMA 指令）被充分利用

Tensor Core 比 CUDA Core（scalar FFMA）在 fp16 下快 8–16 倍。如果 tile 尺寸不满足 Tensor Core 的最小 fragment 要求（Ampere：16×16），编译器会 fallback 到 FFMA。

**验证方法**：Nsight 中 `sm__inst_executed_pipe_tensor_op_hmma.sum` 占计算指令比例应接近 100%。若 `sm__inst_executed_pipe_fma.sum`（scalar FFMA）占比高，说明 Tensor Core 未充分利用。

**CUTLASS/cute 显式指定方式**：使用 `cute::MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>` 明确指定 MMA fragment 形状。

> **出处**：NVIDIA Ampere 架构白皮书，Tensor Core 章节，2020；CUTLASS cute MMA 文档（github.com/NVIDIA/cutlass/tree/main/media/docs/cute）。

#### Compute Bound 成功标准

```
✓ Compute Throughput %        ≥ 70% × Peak FLOPS
✓ tensor_op_hmma 占比         ≥ 90%
✓ Warp Occupancy              ≥ 50%
✓ stall_mio_throttle %        < 20%
✓ 对比 cuBLAS 同 shape GEMM   差距 < 15%（大矩阵时 cuBLAS 约 85–90% MFU）
```

---

## 4. 融合算子的识别与设计

### 4.1 识别机会的核心规则

**基本判断准则**：若两个相邻算子的输出/输入不被其他分支消费，且均为 Memory Bound，则融合收益高。

```
可融合（中间结果只有一个消费者）：
  x = linear(x)    # Compute Bound
  x = x + bias     # Memory Bound ← 可融入 linear epilogue
  x = gelu(x)      # Memory Bound ← 继续融合

不易融合（中间结果被多个消费者使用）：
  x = linear(x)
  y = x.sum(-1)    # x 被两个分支消费
  z = x * 2        # 融合会导致 x 被重复计算
```

### 4.2 决策矩阵

| 算子 A 类型 | 算子 B 类型 | 融合建议 | 收益级别 |
|-----------|-----------|---------|---------|
| Memory Bound | Memory Bound | 强烈推荐 | 高（消除 k-1 次 HBM 往返） |
| Compute Bound | Memory Bound | 推荐（Epilogue Fusion） | 中（无额外 HBM 读，直接从输出寄存器计算） |
| Memory Bound | Compute Bound | 需评估 | 低（通常不值得） |
| Compute Bound | Compute Bound | 通常不融合 | 各自需要完整硬件资源 |

### 4.3 常见可融合 Pattern

| Pattern 类型 | 典型例子 | 推荐实现方式 |
|------------|---------|------------|
| Pointwise 链 | `bias + dropout + residual` | Triton / Inductor 自动融合 |
| GEMM + Epilogue | `linear + bias + activation` | CUTLASS epilogue fusion |
| Reduction + Pointwise | `layernorm = (x-mean)/std` | Triton / Liger Kernel |
| Attention 完整流程 | `QKᵀ → scale → mask → softmax → V` | FlashAttention |
| Norm + 激活 | `RMSNorm + SiLU`（LLaMA FFN） | Liger Kernel |
| 分布式：计算 + 通信 | grad 通信 + 下一层前向 | ZeRO-Infinity / Megatron |

### 4.4 通过 FX Graph 做自定义算子替换

```python
import torch
from torch.fx import symbolic_trace
from torch.fx.subgraph_rewriter import replace_pattern

def pattern(x, weight, bias):        # 原始写法
    x = torch.nn.functional.linear(x, weight)
    x = x + bias
    x = torch.nn.functional.gelu(x)
    return x

def replacement(x, weight, bias):    # 高效融合实现
    return fused_linear_bias_gelu(x, weight, bias)

traced = symbolic_trace(model)
replace_pattern(traced, pattern, replacement)
```

> **出处**：PyTorch FX 官方文档（pytorch.org/docs/stable/fx.html）；torch.fx.subgraph_rewriter API 文档。

---

## 5. 等价替换的寻找方法

### 5.1 数学等价变形

寻找同一计算的不同数学表达，选择对硬件更友好的形式：

| 原始写法 | 等价替换 | 收益 |
|---------|---------|------|
| `(A @ B).T` | `B.T @ A.T` | 可选择更优的内存布局 |
| 朴素 LayerNorm（2 pass） | Welford 在线算法（1 pass） | HBM 读次数减半 |
| 朴素 Softmax（3 pass） | Online Softmax（1 pass） | HBM 读次数减少 3× |
| `sigmoid(x) * x`（SiLU） | 融合实现 | 减少中间写回 |

### 5.2 算法层等价替换

| 原始算法 | 等价替换 | 适用场景 |
|---------|---------|---------|
| 标准 Attention O(N²) 显存 | FlashAttention（IO-optimal） | 训练/推理通用 |
| 标准 LayerNorm | RMSNorm（去均值近似） | 推理，模型支持时 |
| 标准 Dropout + Residual Add | Fused Dropout + Residual | 训练 |
| 稠密 GEMM | SpMM（稀疏矩阵乘） | 剪枝后模型 |

> **出处**：FlashAttention：Dao et al., NeurIPS 2022；RMSNorm：Zhang & Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019；SpMM：Gale et al., *SparseGPT*, ICML 2023。

### 5.3 AutoTuning 搜索等价实现

```
1. CUTLASS Profiler
   列出所有等价 GEMM 实现（不同 tile size、warp 布局、epilogue），
   自动 benchmark 选最快

2. Triton autotuner
   @triton.autotune(configs=[
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'num_stages': 3}),
       triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'num_stages': 4}),
   ])
   搜索最优 tiling 配置

3. TVM MetaSchedule
   系统搜索 tiling / unroll / vectorize 策略空间，
   支持 beam search 和随机演化

4. cuDNN FindAlgorithm
   自动搜索卷积的最优算法（Winograd / FFT / Implicit GEMM）
```

> **出处**：CUTLASS Profiler 文档（github.com/NVIDIA/cutlass）；Triton autotuner：Tillet et al., MLSys 2019；TVM MetaSchedule：Shao et al., *Tensor Program Optimization with Probabilistic Programs*, NeurIPS 2022；cuDNN：Chetlur et al., *cuDNN: Efficient Primitives for Deep Learning*, arXiv 2014。

---

## 6. 昇腾 NPU 算子优化路径

> **导航**：前文 §2-5 的通用方法——Roofline 模型、Memory Bound / Compute Bound 分类、融合算子设计与等价替换思想——同样适用于昇腾 NPU。
> 本节不重复这些通用原理，而是聚焦昇腾硬件特有的设计（Cube / Vector / MTE 等计算与搬运单元）及其约束。
> GPU 优化经验向 NPU 迁移的适配边界与可迁移性分析，详见 §6.3。

### 6.1 硬件架构差异

> **出处**：本节所有架构描述来自 Liao et al., *Ascend: A Scalable and Unified Architecture for Ubiquitous Deep Neural Network Computing*, HPCA 2021（昇腾 Da Vinci 架构唯一主要学术论文）。

#### AI Core 内部结构（Da Vinci 架构）

```
┌─────────────────────────────────────────────────────────┐
│                        AI Core                          │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Cube Unit   │  │ Vector Unit  │  │ Scalar Unit  │  │
│  │  (矩阵乘法)  │  │ (向量/逐元素)│  │   (控制流)   │  │
│  │ 16×16 MMAD  │  │  128-lane    │  │              │  │
│  │  per cycle  │  │     SIMD     │  │              │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘  │
│         └─────────────────┘                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │         MTE（Memory Transfer Engine）              │  │
│  │         专用 DMA 引擎，负责各级存储数据搬运         │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**与 GPU 的根本区别**（HPCA 2021, Section 3）：Cube Unit 和 Vector Unit **不能在同一时刻并发执行**，必须通过 Pipeline 交替工作。GPU 的 Tensor Core 和 CUDA Core 可以被不同 warp 同时调度，而昇腾没有这个能力。

#### 存储层次（必须全部手动管理）

```
HBM（外部）
  ↓ DataCopy（MTE1）
L2 Buffer（芯片级共享，约数 MB）       ← 类比 GPU L2 Cache（但需手动搬运）
  ↓ DataCopy（MTE2）
L1 Buffer（每 AI Core，约 1 MB）       ← Cube 输入暂存
  ↓ DataCopy（MTE2）
L0A / L0B（Cube 输入缓冲，约 16 KB）   ← Cube Unit 直接读取
L0C（Cube 输出/累加缓冲，约 256 KB）   ← Cube Unit 写入
  ↓ FixPipe（可融合激活函数）
UB（Unified Buffer，约 256 KB）        ← Vector Unit 使用
  ↓ DataCopy（MTE3）
HBM（写回）
```

**关键差异**：GPU 的 L1/L2 Cache 对程序员透明，只有 Shared Memory 需要手动管理。昇腾的**所有存储层次**都需要显式 `DataCopy`，没有任何自动 Cache。

> **出处**：HPCA 2021, Figure 4 & Section 3.2；CANN AscendC 编程指南，"存储模型"章节（docs.hiascend.com）。

#### FixPipe 机制

L0C → UB 阶段的 `FixPipe` 搬运可以**免费融合激活函数**（ReLU、Sigmoid 等）：

```
正常路径：L0C → UB → 单独 Vector kernel 计算激活 → HBM
FixPipe  ：L0C → (激活融合) → UB → HBM   ← 节省一次 Vector Unit 调用
```

这是昇腾的 Epilogue Fusion 机制，与 CUTLASS 的 epilogue 设计理念一致，但在架构层面强制支持。

> **出处**：HPCA 2021, Section 3.3（FixPipe 描述）；CANN AscendC 编程指南，"Matmul 算子 FixPipe 配置"章节。

---

### 6.2 AscendC 编程模型

#### 核心抽象：三段 Pipeline 流水线

AscendC 要求将算子计算组织为 **CopyIn → Compute → CopyOut** 三段流水，这是强制编程模式，不是可选优化：

```cpp
class KernelAdd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, ...) {
        // 初始化 HBM 上的全局 Tensor 视图
        xGm.SetGlobalBuffer((__gm__ half*)x);
        yGm.SetGlobalBuffer((__gm__ half*)y);
        zGm.SetGlobalBuffer((__gm__ half*)z);

        // 在 UB 中分配局部缓冲，BUFFER_NUM=2 即双缓冲
        pipe.InitBuffer(inQueueX, /*BUFFER_NUM=*/2, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(inQueueY, /*BUFFER_NUM=*/2, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueZ,/*BUFFER_NUM=*/2, TILE_LENGTH * sizeof(half));
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);   // MTE DMA: HBM → UB
            Compute(i);  // Vector Unit 计算
            CopyOut(i);  // MTE DMA: UB → HBM
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        // DataCopy 是显式 DMA 调用，不会自动发生
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);  // 入队，通知 Compute 阶段数据已就绪
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        Add(zLocal, xLocal, yLocal, TILE_LENGTH);  // Vector Unit 执行
        inQueueX.FreeTensor(xLocal);
        outQueueZ.EnQue(zLocal);
    }
};
```

> **出处**：华为 CANN AscendC 编程指南，"算子开发流程"及"双缓冲优化"章节（docs.hiascend.com）。

#### Cube 算子的额外存储路径

涉及矩阵乘时，需要额外管理 L1 → L0A/L0B 的搬运：

```
HBM → L1（DataCopy，MTE1 负责）
L1 → L0A、L0B（DataCopy，MTE2 负责，为 Cube Unit 喂数据）
Cube Unit 计算 → 结果写入 L0C
L0C → UB（FixPipe，可融合 ReLU 等激活）
UB → HBM（DataCopy，MTE3 负责）
```

**Tiling 约束**（硬约束，不满足时编译失败）：

```
昇腾 Cube Unit 最小粒度（fp16）：
  一次 MMAD：16 × 16 × 16
  因此 BM、BN、BK 必须是 16 的整数倍

L0A 容量（约 16 KB）限制：
  L0A 最多容纳 16KB / (16×16×2B) = 32 个 16×16 block
  → 单次 M 维最大 512

对比 GPU CUTLASS：Ampere 也要求 16×16 fragment，约束类似
```

> **出处**：HPCA 2021, Section 3.2；CANN AscendC 编程指南，"Matmul 类算子 Tiling 策略"章节。

#### DataCopy 对齐要求

```
对齐要求：每次 DataCopy 的数据量须是 32 Bytes 的整数倍
地址对齐：首地址须 32-byte 对齐
Tile 尽量大：减少 DataCopy 调用次数（减少 MTE 启动 overhead）
```

> **出处**：CANN AscendC 编程指南，"DataCopy 接口说明"章节。

---

### 6.3 GPU 经验的适配分析

#### 完全适用的部分

| GPU 经验 | 昇腾适用性 | 说明 |
|---------|----------|------|
| Roofline 模型判断瓶颈 | ✅ 完全适用 | 替换硬件参数即可；昇腾 Ridge Point 更高，Memory Bound 算子更多 |
| 算子融合减少 HBM 访存 | ✅ 完全适用 | 收益通常比 GPU 更大（昇腾带宽相对算力更低） |
| Online Softmax / Welford | ✅ 完全适用 | 算法层优化，与硬件无关 |
| 减少理论访存下界 | ✅ 完全适用 | 分析方法完全相同 |

#### 需要调整才能适用的部分

| GPU 经验 | 昇腾调整点 |
|---------|----------|
| 软件流水线（双缓冲）隐藏延迟 | ✅ 适用，但昇腾中是**强制编程模式**，必须显式用 `TQue + BUFFER_NUM=2` 实现 |
| Tiling 参数调优 | ✅ 适用，但有**强约束**：BM/BN/BK 必须是 16 的倍数，L0 容量固定 |
| 128-bit 向量化访存 | ✅ 适用，对应为 DataCopy 的块要足够大且 32-byte 对齐 |
| Epilogue Fusion | ✅ 适用，对应 FixPipe 机制，但只在 L0C→UB 阶段有效 |

#### 不能迁移的部分（昇腾特有约束）

| GPU 经验 | 昇腾注意事项 | 原因 |
|---------|-----------|------|
| Tensor Core 和 CUDA Core 可同时调度 | ❌ 不适用 | Cube Unit 和 Vector Unit **互斥**，同一 AI Core 同一时刻只有一个工作 |
| L1/L2 Cache 自动管理 | ❌ 不适用 | 昇腾所有存储层次全部需要显式 DataCopy |
| Occupancy 调优（warp 多路 hide latency） | ⚠️ 部分适用 | 昇腾的并发模型不同，以 AI Core 并发代替 warp 并发 |

**昇腾特有问题：AICPU 与 host CPU fallback（需区分两个概念）**

- **昇腾 AICPU**：片上 ARM 核执行的**一类 CANN 算子**（真实存在，常用于动态 shape、控制流密集等不适合 AI Core 的算子）。它仍运行在 NPU 设备上、无需主机拷贝，但比 AI Core 慢。MindStudio 时间线里的 "AI CPU" 指标统计的就是这类算子，占比高的 kernel 需优先用 AscendC 重写。
- **PyTorch host CPU fallback**：当某算子在 NPU 侧**完全没有实现**（CANN 与 op-plugin 都未适配）时，PyTorch 分发机制会把它**回退到主机 CPU** 执行，需在 NPU↔host 间来回拷贝张量，性能可能下降 100× 以上，需补齐 NPU 实现来消除。

辨析：torch_npu 遇到"不支持的算子"走的是 **host CPU fallback**（主机 CPU），而非 AICPU；AICPU 是 CANN 内置、在片上 ARM 核执行的算子实现，二者一个在主机、一个在 NPU 上。

> **出处**：torch_npu 官方文档，"算子支持情况"（gitee.com/ascend/pytorch）；HPCA 2021, Section 3（Cube/Vector 互斥描述）。

---

## 7. 与 torch.compile 的关系

### 7.1 编译栈整体结构

```
用户 Python 代码（nn.Module）
         ↓
    torch.compile()
         ↓
   TorchDynamo（Graph Capture）
   → 通过 trace 捕获 FX Graph，处理 Python 控制流
         ↓
   TorchInductor（默认 Backend，IR 优化 + Codegen）
   → 做 pointwise fusion、layout 优化、常量折叠
         ↓
      ┌──────────────────┬─────────────────┐
      │    GPU 路径       │    CPU 路径      │
      │  Triton Codegen  │  C++ / OpenMP   │
      └──────────────────┴─────────────────┘
```

> **出处**：PyTorch 官方博客 *TorchDynamo: Towards PyTorch Compiler*, Meta AI, 2022；*TorchInductor: PyTorch Native Compiler*, Meta AI, 2022。

### 7.2 各框架与 torch.compile 的具体接入方式

| 框架 | 与 torch.compile 的关系 | 接入方式 |
|-----|----------------------|---------|
| **Triton** | Inductor 的**默认 GPU codegen 后端** | 自动生成；自定义 Triton kernel 通过 `torch.library` 注册 |
| **CUTLASS / TileLang** | **专家级 kernel 替换**（非默认路径） | 注册为 `torch.ops.custom_op`，inductor 跳过 codegen 直接 dispatch |
| **TVM** | 可作为 Inductor 的**替换后端** | `torch.compile(backend="tvm")` |
| **XLA** | 独立编译栈 | JAX / TF 路线，或 `torch_xla` |
| **昇腾（torch_npu）** | 提供昇腾版 Inductor backend | `torch.compile` 后由 torch_npu 接管 dispatch |

### 7.3 自定义高性能算子的注册方式

```python
import torch

# 方式1：通过 torch.library 注册 Triton kernel
@torch.library.custom_op("mylib::fused_layernorm_gelu", mutates_args=())
def fused_layernorm_gelu(x: torch.Tensor) -> torch.Tensor:
    return _triton_fused_layernorm_gelu(x)

# 方式2：通过 FX subgraph rewriter 替换子图
from torch.fx.subgraph_rewriter import replace_pattern
replace_pattern(traced_model, pattern_fn, replacement_fn)

# 验证 torch.compile 正确调用
compiled = torch.compile(model, backend="inductor")
```

---

## 8. 完整优化工作流

### 8.1 GPU 优化工作流

```
Step 1  Profile 采集
        ncu --set full python script.py
        → 识别热点 kernel（时间占比 > 5%）

Step 2  Roofline 判断
        → 手算算子理论 AI
        → 对比 Ridge Point（A100: ~156, H100: ~295 FLOPs/Byte）
        → 读 Memory Throughput % 和 Compute Throughput %

Step 3a  Memory Bound 路径
        → 识别可融合的 elementwise 链（中间结果无多分支消费）
        → 用 Triton 手写融合 kernel 或依赖 torch.compile 自动融合
        → 检查访存对齐（BLOCK_SIZE 为 128 倍数，地址 16-byte 对齐）
        → 考虑算法变形（Online Softmax / Welford / FlashAttention）
        → 目标：Memory Throughput ≥ 80%，DRAM bytes 接近理论下界

Step 3b  Compute Bound 路径
        → AutoTuning Tiling 参数（BM/BN/BK/num_stages/num_warps）
        → 确认 Tensor Core 被充分利用（hmma 占比 ≥ 90%）
        → 开启软件流水线（num_stages=2-4）隐藏 SRAM load 延迟
        → 目标：Compute Throughput ≥ 70%，stall < 20%

Step 4  注册为 Custom Op
        → torch.library.custom_op 注册
        → 验证 torch.compile 下正确 dispatch

Step 5  验证正确性
        → 与 PyTorch 参考实现对比，数值误差 < 1e-3（fp16）

Step 6  回归 Profile
        → 确认优化收益，排除引入的新瓶颈
```

### 8.2 昇腾 NPU 优化工作流

```
Step 1  Profile 采集
        msprof --aic-metrics=ALL python script.py
        → MindStudio 可视化分析时间线
        → 优先检查 AI CPU 占比（AI CPU 高 = 算子未适配，优先级最高）

Step 2  Roofline 判断
        → 手算算子理论 AI
        → 对比昇腾 Ridge Point（910B: ~333 FLOPs/Byte）
        → 读 AI Core 利用率、HBM 带宽利用率

Step 3a  Memory Bound 路径
        → 识别可融合的 elementwise 链
        → AscendC 手写融合 kernel（替代 Triton 的角色）
        → 检查 DataCopy 是否流水（BUFFER_NUM=2，MTE 与 Vector 重叠）
        → 检查 DataCopy 对齐（32-byte 对齐，Tile 足够大）
        → 考虑算法变形（Online Softmax / Welford，与 GPU 相同）
        → 目标：HBM 带宽利用率 ≥ 80%，MTE 与 Compute 时间线重叠

Step 3b  Compute Bound 路径
        → 检查 Tiling 满足 16 的倍数约束（硬约束）
        → 确认 Cube + Vector 通过 Pipeline 交替（不能期望自动并发）
        → 利用 FixPipe 融合激活函数（L0C→UB 阶段）
        → 检查 L1 Buffer 是否超限（超限导致 tile 被强制拆小）
        → 目标：Cube 利用率 ≥ 70%，MTE 搬运时间被计算覆盖

Step 4  注册到 PyTorch 生态
        → torch_npu 自定义算子接口（AscendC 算子注册）
        → 验证 torch.compile + torch_npu backend 正确调度

Step 5  验证正确性
        → 与 CPU 或 GPU 参考实现对比，fp16 误差 < 1e-3
        → 注意昇腾 fp16 的 NaN / Inf 行为与 GPU 有差异

Step 6  回归 Profile
        → 确认 AI CPU 占比未增加，整体端到端性能提升
```

---

## 9. 参考文献

### 核心论文

| 论文 | 内容 | 出处 |
|-----|------|------|
| Williams et al. | Roofline 模型定义与分析方法 | *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, CACM, 2009 |
| Tillet et al. | Triton 编程模型，block-level 向量化，autotuner | *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*, MLSys, 2019 |
| Dao et al. | FlashAttention，IO-complexity 分析，Online Softmax 融合 | *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS, 2022 |
| Milakov & Gimelshein | Online Softmax 算法（3 pass → 1 pass） | *Online normalizer calculation for softmax*, arXiv:1805.02867, 2018 |
| Welford | 在线均值方差计算（Welford 算法） | *Note on a method for calculating corrected sums of squares and products*, Technometrics, 1962 |
| Liao et al. | 昇腾 Da Vinci 架构（Cube/Vector/MTE/存储层次） | *Ascend: A Scalable and Unified Architecture for Ubiquitous Deep Neural Network Computing*, HPCA, 2021 |
| Chen et al. | TVM 编译器，自动调度，Schedule 原语 | *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning*, OSDI, 2018 |
| Shao et al. | TVM MetaSchedule，概率调度搜索 | *Tensor Program Optimization with Probabilistic Programs*, NeurIPS, 2022 |
| Chetlur et al. | cuDNN，ConvNet 高性能原语 | *cuDNN: Efficient Primitives for Deep Learning*, arXiv:1410.0759, 2014 |
| Zhang & Sennrich | RMSNorm，去均值近似等价 | *Root Mean Square Layer Normalization*, NeurIPS, 2019 |
| Gale et al. | 稀疏 GEMM，剪枝模型推理 | *SparseGPT: Massive Language Models Can be Accurately Pruned in One Shot*, ICML, 2023 |

### 官方文档与白皮书

| 文档 | 内容 |
|-----|------|
| NVIDIA A100 Tensor Core GPU Architecture Whitepaper, 2020 | A100 硬件规格，Tensor Core 描述 |
| NVIDIA H100 Tensor Core GPU Architecture Whitepaper, 2022 | H100 硬件规格，Hopper 架构 |
| NVIDIA Nsight Compute User Guide, Counter Reference | Nsight 指标定义（docs.nvidia.com/nsight-compute） |
| NVIDIA CUTLASS 官方文档 | Tiling 层次，Epilogue Fusion，Software Pipelining（github.com/NVIDIA/cutlass） |
| CUDA Best Practices Guide | 访存对齐，向量化 load，occupancy 调优（docs.nvidia.com） |
| PyTorch 官方博客：*TorchDynamo* & *TorchInductor*, Meta AI, 2022 | torch.compile 编译栈架构 |
| PyTorch FX 官方文档 | subgraph_rewriter，pattern matching（pytorch.org/docs/stable/fx.html） |
| 华为 CANN AscendC 编程指南 | AscendC API，存储模型，Pipeline 流水，DataCopy（docs.hiascend.com） |
| 华为 MindStudio 性能分析工具用户指南 | 昇腾 Profiling 指标定义（hiascend.com/document/detail/zh/mindstudio） |
| 华为昇腾 910B 官方规格 | 算力、HBM BW 参数（hiascend.com/hardware/ascend-910B） |
| torch_npu 官方文档 | 算子支持列表，AI CPU 退化机制（gitee.com/ascend/pytorch） |

---

*文档版本：v1.0 | 最后更新：2026-05*

---

## Related Pages

- [[02_torch_compile_architecture]] — torch.compile 端到端编译流水线（Dynamo → Inductor）
- [[26_flex_attention_analysis]] — FlexAttention 可组合注意力融合机制
- [[tilelang_analysis]] — TileLang Tile-Level IR 与 Host Codegen
- [[01_npu_compile_paths_overview]] — NPU torch.compile 三条路径（Triton/ACLGraph/MLIR）
- [[NPU_MLIR_Backend_Technical_Analysis]] — NPU MLIR 六阶段编译管线
- [[12_npu_compile]] — NPU 编译工作流与 Autotune
- [[20_npu_lowering_guide]] — NPU 特定算子 Lowering 指南
- [[01_PyTorch_CUDA_Graphs_Complete_Guide]] — CUDA Graphs 图捕获/重放机制
- [[11_torch_compile_npugraphs_deep_dive]] — NPU Graphs + torch.compile 深度分析
- [[10_aclgraph_deep_analysis]] — ACLGraph 图捕获与 Super Kernel
- [[ascend_kernel_execution_model_analysis]] — DaVinci AI Core、L1/L0/UB 缓冲链、TQue 与 FixPipe 的执行模型深挖
- [[01_ai_frameworks/index]] — AI 框架领域索引
- [[02_engineering/index]] — 工程实现领域索引
- [[changelog]] — 变更日志
