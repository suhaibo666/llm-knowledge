---
title: "GPU Kernel 开发 — 目录索引"
---

# GPU Kernel 开发 — 目录索引

> 覆盖 GPU/NPU Kernel 的执行模型、内存优化、Tensor Core、torch.compile、FlashAttention 与架构差异；以及以 **Triton 为切入点**、从小白到「会写·会调·会优化·会debug」全能专家的系统学习路线
> 最后更新: 2026-08-27(补模块定位、技术栈分级与能力清单;原「最后更新 2026-07-31 kb-reorg P7 Task 7:目录内分段编号」)

---

## 模块定位：做什么 · 提供什么能力 · 边界在哪

**一句话**：本域回答的是**一段计算在硬件上为什么是这个速度、还能快多少**。它是全栈唯一**不谈编排、只谈单个 kernel 内部**的层：执行层级如何映射到 SM、数据如何在存储层级间搬运、Tensor Core 如何被喂满。

**为什么必须独立成一层**：上面所有域最终都落到 kernel 上，但它们关心的是"**调哪个** kernel"（[[02_engineering/03_infer_frameworks/index|推理框架]] 按 batch 形态派发 attention 后端、[[02_engineering/01_pytorch/index|PyTorch]] 的 Inductor 生成 Triton 源码），本域关心的是"**这个 kernel 内部为什么这么写**"。两者的判断依据完全不同：上层看吞吐与 SLO，本域看 **roofline**——一个算子是 memory-bound 还是 compute-bound，决定了优化方向是访存合并与融合，还是 tile 切分与异步流水。方向搞反，优化做得再精细也没有收益。

### 本域覆盖的硬件/编程栈与各自定位

本域横跨两种硬件架构与两套编程范式，**来源性质差异很大**，如实标注：

| 技术栈 | 在本域中的定位 | 本库覆盖 | 来源与基线 |
|---|---|---|---|
| **CUDA / NVIDIA GPU** | 全域的**概念基准**：执行层级、roofline 判据、GEMM 与非 GEMM 两类 kernel 的构造法 | 4 篇 | CUDA C++ Programming Guide v12.9.1（执行模型）+ **本地 HTML 快照**（SM80/A100 的 GEMM 与非 GEMM 分析）——**非源码级** |
| **Triton** | tile 级 DSL；本域**唯一有本地源码基线**的实现，从写第一个 kernel 到 autotune/debug/profiling 的完整学习路线 | 8 篇 + index（**源码级，可核验 `file:line`**） | `triton-lang/triton@70e0929`（v3.8.0，2026-06-25），浅克隆于侧车目录 |
| **Ascend / DaVinci NPU** | 另一种硬件架构：Cube/Vector/Scalar/MTE 分工与显式缓冲链，及 CUDA↔Ascend 概念映射 | 1 篇 | **本地 HTML 快照**（910B-class）——**非源码级** |
| **TileLang** | tile-level IR 的另一条路线，填补图 Pass 与 kernel 代码之间的空白 | 1 篇 | **公开论文/官方文档的概念分析，本地无实现源码** |
| **跨栈综览** | 上述四者的入门总览与诊断清单 | 1 篇 | 综合 |

> 这个差异很重要：**只有 Triton 那 8 篇能按 `file:line` 核验**。CUDA GEMM/非 GEMM、Ascend 三篇来自 HTML 快照的深度分析，TileLang 一篇是概念分析——它们的结论可靠但**不是从源码读出来的**，引用时不要写成"读 XX 源码可见"。各页头部都标了自己的来源性质。

### 本域提供的能力

| 能力 | 具体提供什么 | 技术栈与来源锚点 | 详见 |
|---|---|---|---|
| **执行模型的逻辑↔物理映射** | Grid/Block/Warp/Thread 与 SM 的对应关系，及由此推出的三个后果：分支发散、合并访存、占用率 | CUDA 官方文档 v12.9.1 | [[10_cuda_execution_model_guide]] |
| **性能判据（roofline）** | 算术强度与 ridge point 的算法、常见算子的理论 AI、GPU/NPU profiling 指标对照与优化路径 | 论文 + 官方文档 | [[11_operator_optimization_guide]] |
| **GEMM 类 kernel 的构造法** | CTA/Warp/MMA 四层切分、M/N 空间 tile 与 K 时间归约、`cp.async` 多级流水、寄存器账本、shared-memory epilogue | CUDA / 本地 HTML 快照（SM80） | [[20_cuda_gemm_kernel_analysis]] |
| **非 GEMM kernel 的优化轴** | 五类数据依赖（elementwise / reduction / scan / stencil / gather-scatter）各自的瓶颈与可行变换 | CUDA / 本地 HTML 快照 | [[21_cuda_nonmatmul_kernels_analysis]] |
| **tile 级 DSL 的编写与调优** | Triton 编程模型、fused softmax 与 matmul 的写法、autotune 如何选出 winner、调试与 profiling 手段 | Triton `@70e0929`：`python/triton/language/` · `python/triton/runtime/autotuner.py` | [[triton/index\|Triton 学习路线]] |
| **tile 级 DSL 的编译内部** | layout 在 IR 中的表示、多后端下降路径 | Triton `@70e0929`：`lib/Dialect/TritonGPU/` · `third_party/{nvidia,amd}/` | 同上 |
| **NPU 侧的执行模型差异** | Cube/Vector/Scalar/MTE 分工、GM→L1→L0→UB 显式缓冲链、Queue 双缓冲与 FixPipe、训练三条优化主线 | 本地 HTML 快照（DaVinci） | [[22_ascend_kernel_execution_model_analysis]] |
| **IR 层次的另一种取舍** | TileLang 的 IR 设计与融合 kernel 表达，与 Triton 的 tile 概念关系 | 公开资料（概念分析） | [[23_tilelang_analysis]] |

### 不属于本模块的

- kernel 由谁在什么时机被调用、图怎么切、缓存怎么命中 → [[02_engineering/01_pytorch/index|PyTorch]] 的编译栈；本域到"生成什么样的 Triton/CUDA 源码"为止，不管上游的 lowering 决策；
- 单卡之外的通信 kernel 与并行策略 → [[02_engineering/02_train_frameworks/index|训练框架]] 与 [[01_theory/06_distributed_parallelism/index|分布式并行原理]]；
- 算子的**数值**行为（累加顺序、确定性、低精度误差）→ [[02_engineering/07_training_reliability/index|训练可靠性]]；本域看速度，那边看正确性——同一个归约 kernel 在两个域里被问的是不同的问题。

### 与兄弟域的关系

本域是全栈的**性能解释层**。当上层观察到"这个配置比那个慢 30%"，能给出机理解释的通常只有这一层；反过来，本域的结论要通过 `01_pytorch` 的 codegen 或推理框架的后端选择才能落到真实负载上。

---

## 页面列表

> **段位**(kb-reorg P7 Task 7,2026-07-31):段 0(01)综合概览,先建立执行层级/内存/Tensor Core/torch.compile/NPU 差异的全貌;段 1(10-19)两篇"权威页"——执行模型与 Roofline,P6 归一后的核心机制主线;段 2(20-29)四篇专题深挖(生产级 GEMM、非 GEMM kernel 轴、Ascend 执行模型、TileLang)。

| 页面 | 层次 | 来源 | 核心主题 |
|------|------|------|---------|
| [[01_gpu_kernel_guide]] | 概览(段 0) | 综合深度分析 | 执行层级(Grid/Block/Warp/Thread/SM/Tile)、内存优化(Coalesced/Shared/Tiling/Bank Conflict)、Occupancy、Warp Divergence、异步Pipeline、Tensor Core/MMA、torch.compile 编译路径、FlashAttention 完整链路、NPU 架构差异、诊断清单 |
| [[10_cuda_execution_model_guide]] | 核心机制(段 1) | CUDA C++ Programming Guide v12.9.1 | **执行模型权威页**：Grid·Block·Warp·Thread·SM 逻辑↔物理映射（概念→深入），warp=32 锁步、Block→SM 驻留、分支发散/合并访问/占用率三推论、`__syncthreads` 与 Block 独立性、映射到 Triton；3 个可运行 demo |
| [[11_operator_optimization_guide]] | 核心机制(段 1) | 论文+官方文档综合整理 | **Roofline 权威页**：公式/Ridge Point 硬件参数表/常见算子理论 AI/GPU+NPU profiling 指标（§2）；GPU Memory/Compute Bound 优化路径（§3）；融合算子识别与等价替换（§4-5）；昇腾 NPU 优化路径速查+GPU 经验迁移 checklist（§6）；与 torch.compile 关系、完整工作流对比（§7-8） |
| [[20_cuda_gemm_kernel_analysis]] | 深潜(段 2) | 本地 HTML 快照（SM80 / A100） | **生产级 CUDA GEMM**：Grid/CTA/Warp/MMA 四层切分、M/N 空间 tile 与 K 时间归约、`cp.async` 多级流水、每线程寄存器账本、shared-memory epilogue、完整教学 kernel |
| [[21_cuda_nonmatmul_kernels_analysis]] | 深潜(段 2) | 本地 HTML 快照（CUDA） | **非 GEMM 优化轴**：roofline + 五类数据依赖；elementwise、reduction、norm、FlashAttention、stencil、scan、gather/scatter/sort 的结构、瓶颈与边界 |
| [[22_ascend_kernel_execution_model_analysis]] | 深潜(段 2) | 本地 HTML 快照（DaVinci / 910B-class） | **Ascend 执行模型**：AI Core 的 Cube/Vector/Scalar/MTE、GM→L1→L0→UB 显式缓冲链、Queue 双缓冲、FixPipe、CUDA↔Ascend 映射与训练三条优化主线 |
| [[23_tilelang_analysis]] | 深潜(段 2) | TileLang 源码 | TileLang IR、DSL 融合 kernel(DeepSeek-V4 mHC/MoE 应用) |

---

## Triton 学习路线（`triton/` 子目录）

> 以 **Triton（block-level GPU 编程）** 为切入点的手把手系列，锚定官方 `triton-lang/triton main @ 70e0929`（v3.8.0）的真实 tutorial 源码，每页配可运行 demo。入口：[[02_engineering/05_gpu_kernel/triton/index|Triton 学习路线]]（本子目录的 `triton/index.md`）。
>
> **段位**(kb-reorg P7 Task 7,2026-07-31):原课程序号 `triton_00`–`triton_06` 规范化为段位前缀(文件名 `triton_` 前缀保留)——00 入门→段 0(01);01-05 会写/会调/会debug 主线→段 1(10-14);06 会优化(profiling 方法论)+ 总纲→段 3(30-31,方法论/参考)。

| 页面 | 层次 | 阶段·能力 | 核心 demo（锚定源） |
|------|------|-----------|--------|
| [[triton_01_gpu_essentials_guide]] | 段 0 | L0 地基 | roofline 手算 + `do_bench` 量带宽 |
| [[triton_10_programming_model_guide]] | 段 1 | L1 会写① | 向量加法（`01-vector-add.py`） |
| [[triton_11_fused_softmax_guide]] | 段 1 | L1 会写② | 融合 softmax（`02-fused-softmax.py`） |
| [[triton_12_matmul_guide]] | 段 1 | L1 会写③+优化 | 分块矩阵乘（`03-matrix-multiplication.py`） |
| [[triton_13_autotune_guide]] | 段 1 | L2 会调 | autotune（`@triton.autotune`） |
| [[triton_14_debug_guide]] | 段 1 | L3 会debug | 解释器抓越界（`TRITON_INTERPRET=1`） |
| [[triton_30_optimization_profiling_guide]] | 段 3 | L4 会优化 | proton + FlashAttention（`06-fused-attention.py`） |
| [[triton_31_knowledge_guide]] | 段 3 | 总纲 | 四能力知识点清单 + 自测 + 进阶 |

---

## 关联域

- [[../01_pytorch/index]] — AI 框架 (PyTorch 编译栈；Inductor 自动生成 Triton kernel)
- [[../02_train_frameworks/index]] — 训练框架 (Kernel 是上层掩盖的基础)
- [[../03_infer_frameworks/index]] — 推理框架 (FlashAttention 等 kernel 是推理性能关键)
- [[30_triton_vs_mlir_backend_analysis]] — Triton 作为 torch.compile 后端的编译流水线
