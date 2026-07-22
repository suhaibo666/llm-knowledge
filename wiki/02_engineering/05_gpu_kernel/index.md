# GPU Kernel 开发 — 目录索引

> 覆盖 GPU/NPU Kernel 的执行模型、内存优化、Tensor Core、torch.compile、FlashAttention 与架构差异；以及以 **Triton 为切入点**、从小白到「会写·会调·会优化·会debug」全能专家的系统学习路线
> 最后更新: 2026-07-22

---

## 页面列表

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[cuda_execution_model_guide]] | CUDA C++ Programming Guide v12.9.1 | **执行模型地基**：Grid·Block·Warp·Thread·SM 逻辑↔物理映射（概念→深入），warp=32 锁步、Block→SM 驻留、分支发散/合并访问/占用率三推论、`__syncthreads` 与 Block 独立性、映射到 Triton；3 个可运行 demo |
| [[cuda_gemm_kernel_analysis]] | 本地 HTML 快照（SM80 / A100） | **生产级 CUDA GEMM**：Grid/CTA/Warp/MMA 四层切分、M/N 空间 tile 与 K 时间归约、`cp.async` 多级流水、每线程寄存器账本、shared-memory epilogue、完整教学 kernel |
| [[cuda_nonmatmul_kernels_analysis]] | 本地 HTML 快照（CUDA） | **非 GEMM 优化轴**：roofline + 五类数据依赖；elementwise、reduction、norm、FlashAttention、stencil、scan、gather/scatter/sort 的结构、瓶颈与边界 |
| [[ascend_kernel_execution_model_analysis]] | 本地 HTML 快照（DaVinci / 910B-class） | **Ascend 执行模型**：AI Core 的 Cube/Vector/Scalar/MTE、GM→L1→L0→UB 显式缓冲链、Queue 双缓冲、FixPipe、CUDA↔Ascend 映射与训练三条优化主线 |
| [[gpu_kernel_guide]] | 综合深度分析 | 执行层级(Grid/Block/Warp/Thread/SM/Tile)、内存优化(Coalesced/Shared/Tiling/Bank Conflict)、Occupancy、Warp Divergence、异步Pipeline、Tensor Core/MMA、torch.compile 编译路径、FlashAttention 完整链路、NPU 架构差异、诊断清单 |

---

## Triton 学习路线（`triton/` 子目录）

> 以 **Triton（block-level GPU 编程）** 为切入点的手把手系列，锚定官方 `triton-lang/triton main @ 70e0929`（v3.8.0）的真实 tutorial 源码，每页配可运行 demo。入口：[[index]]（本子目录的 `triton/index.md`）。

| 页面 | 阶段·能力 | 核心 demo（锚定源） |
|------|-----------|--------|
| [[triton_00_gpu_essentials_guide]] | L0 地基 | roofline 手算 + `do_bench` 量带宽 |
| [[triton_01_programming_model_guide]] | L1 会写① | 向量加法（`01-vector-add.py`） |
| [[triton_02_fused_softmax_guide]] | L1 会写② | 融合 softmax（`02-fused-softmax.py`） |
| [[triton_03_matmul_guide]] | L1 会写③+优化 | 分块矩阵乘（`03-matrix-multiplication.py`） |
| [[triton_04_autotune_guide]] | L2 会调 | autotune（`@triton.autotune`） |
| [[triton_05_debug_guide]] | L3 会debug | 解释器抓越界（`TRITON_INTERPRET=1`） |
| [[triton_06_optimization_profiling_guide]] | L4 会优化 | proton + FlashAttention（`06-fused-attention.py`） |
| [[triton_knowledge_map]] | 总纲 | 四能力知识点清单 + 自测 + 进阶 |

---

## 关联域

- [[../01_ai_frameworks/index]] — AI 框架 (PyTorch 编译栈；Inductor 自动生成 Triton kernel)
- [[../02_train_frameworks/index]] — 训练框架 (Kernel 是上层掩盖的基础)
- [[../03_infer_frameworks/index]] — 推理框架 (FlashAttention 等 kernel 是推理性能关键)
- [[triton_vs_mlir_backend_analysis]] — Triton 作为 torch.compile 后端的编译流水线
