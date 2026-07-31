# L0 · GPU 编程要素 — 在写第一行 Triton 前必须建立的三个直觉

> **源基线**: `triton main @ 70e0929`，v3.8.0 ｜ 锚定 `python/tutorials/01..03-*.py` 的官方 benchmark 公式
> **维度**: 学习路线 L0（地基认知）｜ 能力：全部能力的前置
> 本页回答：GPU 为什么是这个样子？哪三个「数」决定 kernel 快慢？这些要素在 Triton 里以什么形式出现？通用硬件细节见姊妹页 [[01_gpu_kernel_guide]]，本页聚焦**和 Triton 编程直接相关、且可由官方源验证**的部分。

---

## 1. 一句话地基

> **GPU 是「吞吐机器」：几千个弱核 + 超量并行，用「随时切换到下一批活」来掩盖访存的几百周期延迟。所以 GPU 编程 90% 是在管「数据怎么搬」，而不是「算什么」——绝大多数 kernel 受内存带宽限制（memory-bound），不是算力。**

记住这条主线，后面所有优化（fusion、tiling、流水线）都是它的推论。

---

## 2. 三个你必须建立的直觉

### 直觉一：执行层级（你在 Triton 里看到的是「program」）

CUDA 的物理层级是 `GPU → SM → warp(32 线程) → thread`（**这条逻辑↔物理映射链的概念→深入讲解见 [[10_cuda_execution_model_guide]]**，卡概念务必先读它；硬件数值细节见 [[01_gpu_kernel_guide]] §01）。

**Triton 把这层抽象掉了**：你写的 kernel 是一个 **program（程序实例）**，对应 CUDA 的「一个 block 处理的一块数据」。你不碰 `threadIdx`，只用：

| Triton 概念 | 含义 | 源 |
|---|---|---|
| `tl.program_id(axis)` | 我是第几个 program（≈ `blockIdx`） | `01-vector-add.py:39` |
| `tl.num_programs(axis)` | 一共有多少个 program | `02-fused-softmax.py:89` |
| grid | 启动多少个 program（SPMD） | `01-vector-add.py:70` |
| `num_warps` | 编译器用多少 warp 并行**一个** program 的块内计算 | `02-fused-softmax.py:134` |

> 关键认知：**block 内的 32×N 个线程怎么分工、怎么合并访存——Triton 编译器替你决定**。你只通过 `num_warps` 给个并行度提示。

### 直觉二：内存层级（决定性能的全部）

从快到慢：`寄存器 → Shared Memory(SRAM) → L2 → HBM(显存)`，相邻层延迟/带宽差**一个数量级以上**（具体数值见 [[01_gpu_kernel_guide]] §02）。所有优化都在做同一件事：**把数据从 HBM 搬进 SRAM/寄存器后尽量多复用，少回 HBM**。

Triton 里这层的体现：
- `tl.load(ptr + offsets, mask=...)` —— 从 HBM 读一个块进片上（编译器自动合并访问）。源 `01-vector-add.py:50`
- 块内的中间变量（`tl.exp`、`tl.dot` 的累加器）—— 活在寄存器/SRAM，**不回 HBM**。这正是 fusion 省带宽的来源（见 [[triton_11_fused_softmax_guide]]）。

### 直觉三：Roofline —— 优化的指挥棒

每个 kernel 落在两种状态之一，由**算术强度**（Arithmetic Intensity, AI = FLOPs/Bytes）决定：**AI 低 → memory-bound**（瓶颈带宽，靠减少 HBM 访问/fusion）；**AI 高 → compute-bound**（瓶颈算力，靠 Tensor Core `tl.dot`）；交叉点 = 硬件 Ridge Point（A100 ≈ 156 FLOP/Byte）。

> **永远不要优化不是瓶颈的东西。** 先判断在 roofline 哪一侧，再动手。

Roofline 公式、Ridge Point 硬件参数表、GPU/NPU 双平台 profiling 指标，以及用 Triton 官方 `01/02/03-*.py` 三个 tutorial 的 benchmark 公式逐项手算 AI 的完整推导（向量加法极度 memory-bound、融合 softmax 省带宽、矩阵乘 compute-bound——「官方拿什么单位 benchmark 就告诉了你它在 roofline 哪一侧」），见 Roofline 权威页 [[11_operator_optimization_guide]] §2（§2.5 是本页这组 demo 的完整版）。

---

## 3. 这些要素在 Triton 里的「分工表」

| GPU 编程要素 | 谁负责 | 你在 Triton 里的动作 |
|---|---|---|
| 合并访问 (coalescing) | **编译器** | 让 `offsets = base + tl.arange(0, B)` 连续即可（`01-vector-add.py:45`） |
| Shared memory / tiling | 你描述形状，**编译器排布** | 用 `tl.dot` + 块循环（`03:298-304`），编译器分配 SRAM |
| Bank conflict | **编译器**（swizzling） | 一般不用手管 |
| 寄存器累加 | 你声明 | `acc = tl.zeros((BM,BN), tl.float32)`（`03:297`） |
| warp 划分 | **编译器** | 给 `num_warps`（`02:134`） |
| 软件流水线 (cp.async) | **编译器** | 给 `num_stages`（`02:90`、`03:166`），见 [[triton_30_optimization_profiling_guide]] |
| 算法结构 / 数据复用 | **你** ← 真正决定性能 | 切块方式、循环顺序、L2 grouping（`03:256-264`） |

**这张表是「以 Triton 为切入点学 GPU」的核心理由**：左列是 CUDA 里最难最易错的活，Triton 把大半交给了编译器，你能把精力集中在最右那一行——**算法与复用**，这恰恰是性能的真正来源。

---

## 4. 动手验证

```bash
cd triton/python/tutorials
python 01-vector-add.py     # 看 GB/s 曲线：达到的带宽 vs 你显卡的峰值带宽，差距=优化空间
python 03-matrix-multiplication.py   # 看 TFLOPS 曲线：Triton vs cuBLAS
```

- 把 `01` 的 GB/s 峰值除以你显卡标称带宽（如 A100 ≈ 2 TB/s），得到带宽利用率——这是 memory-bound kernel 的「成绩单」。
- 思考题：为什么 `01` 再怎么优化也到不了 `03` 的 TFLOPS？（答案：AI 差三个数量级，它俩根本不在 roofline 同一侧。）

---

## 相关页面

- [[index]] — Triton 学习路线总索引
- [[10_cuda_execution_model_guide]] — **前置地基**：Grid·Block·Warp·Thread·SM 执行模型（概念→深入）
- [[01_gpu_kernel_guide]] — GPU/NPU Kernel 工程总览（执行/内存层级的硬件细节、Tensor Core、NPU 差异、CUDA 手写视角）
- [[11_operator_optimization_guide]] — Roofline 权威页（公式、Ridge Point 参数表、GPU/NPU profiling 指标、§2.5 是本页 Roofline demo 的完整推导）
- [[triton_10_programming_model_guide]] — 下一步：用 `program_id`/`mask`/`load` 写第一个 kernel
- [[triton_11_fused_softmax_guide]] — fusion 如何把 memory-bound kernel 的访存量砍半
- [[triton_12_matmul_guide]] — compute-bound kernel 如何用 `tl.dot` 逼近峰值
- [[triton_31_knowledge_map]] — 全部知识点清单
