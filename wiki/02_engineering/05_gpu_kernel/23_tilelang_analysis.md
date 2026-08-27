---
title: "TileLang：Tile-Level IR 与新一代 Kernel 编程范式"
---

# TileLang：Tile-Level IR 与新一代 Kernel 编程范式

> 填补图 Pass（层次过高）与 Kernel 代码（层次过低）之间的空白
> 最后更新: 2026-05-12
>
> 注：本文为基于公开资料（论文/官方文档）的概念分析，本地代码库不含对应实现源码，具体以上游为准。

---

## 1. 背景：当前编译栈的抽象层次缺口

```
图 Pass 层（FX Graph / ATen IR）
  优势: 看到全局结构, 可以做跨算子优化
  局限: 不知道 tile/warp/shared memory 细节
        ↑                      ↑
        Gap：没有 tile-level 的全局优化
        ↓                      ↓
Kernel 层（Triton DSL / CUDA C++）
  优势: 精确控制 tile 大小、pipeline depth、SM 分配
  局限: 只看到单个 kernel 内部, 不知道全局图结构
```

**具体问题**：

- FlashAttention 的 tile 大小（BLOCK_M, BLOCK_K）与 A2A 的 chunk 粒度分别独立确定
- 理想情况下，WaveEP 的 wave 粒度（tokens per wave）应与 Expert GEMM 的 tile 大小联合优化
- 图 Pass 决定"融合哪些算子"，但不能决定"如何 tile 这个融合 kernel"
- Triton kernel 知道 tile 大小，却不了解自身在全局通算调度中的位置

---

## 2. TileLang 是什么

**TileLang** 是 DeepSeek V4 引入的 tile-level kernel 编程语言（DSL），在 Triton DSL 的上方增加了一层抽象：

```
位置：图 Pass（ATen IR）
           ↓
      TileLang IR          ← 新层次
      (tile-based ops,     
       pipeline desc,      
       A2A chunk binding)  
           ↓
      Triton / CUDA code   ← 代码生成
```

**核心能力**：

| 特性 | 描述 |
|------|------|
| **Tile-based 算子描述** | 以 tile（块矩阵）为单位描述计算，而非单个元素 |
| **Pipeline 声明** | 显式声明 compute-pipeline depth（prefetch 层数）|
| **Host Codegen** | 自动生成 CPU 侧调度代码，kernel launch 开销 <1μs |
| **Z3 SMT 整数分析** | 自动验证 tile 大小整除性、边界合法性 |
| **A2A Chunk Binding** | 将通信 chunk 粒度与 tile 粒度绑定，实现 WaveEP |

---

## 3. TileLang 在 DeepSeek V4 的应用

### 3.1 mHC 融合 Kernel

mHC（流形约束超连接）需要将多个高内存带宽操作融合：

```
mHC 原始计算链：
  RMSNorm(H) → Linear(W_map) → Scale+Bias → Sigmoid → 
  Sinkhorn-Knopp(20 iterations) → Apply(H_post, H_res, residual)

内存带宽瓶颈（假设 n=4 条残差流, hidden_dim=C）：
  读取: (3n+1)C 元素（每条残差流 3 次读取）
  写入: 3nC 元素

TileLang 融合后：
  整条链在片上（shared memory）完成
  读取: (n+1)C → 降低 3× 带宽
  写入: nC → 降低 3× 带宽
```

**Sinkhorn-Knopp 20 次迭代的片上处理**：

```python
# TileLang 伪代码（概念示意）
@tilelang.kernel
def mhc_fused_kernel(H, W_map, B_l, output):
    # 1. Load H tile to shared memory
    H_tile = tl.load(H[tile_idx])  # 一次 HBM 读取
    
    # 2. RMSNorm（片上）
    H_norm = rms_norm_onchip(H_tile)
    
    # 3. Linear projection（片上 matmul）
    H_proj = matmul_onchip(H_norm, W_map_tile)
    
    # 4. Sinkhorn-Knopp 20 iterations（全部在寄存器/shared mem）
    B_doubly_stochastic = sinkhorn_onchip(B_l_tile, iters=20)
    
    # 5. Apply and merge residuals（片上）
    result = apply_mhc_onchip(H_tile, H_proj, B_doubly_stochastic)
    
    # 6. 一次 HBM 写入
    tl.store(output[tile_idx], result)
```

### 3.2 Host Codegen：<1μs Launch 开销

传统 kernel launch：

```
Python → torch dispatch → CUDA driver API → kernel launch
                                              ↑
                                    通常 5-20μs CPU overhead
```

TileLang Host Codegen：

```
TileLang 编译时 → 生成 C++ launch 代码（预计算所有 grid/block/smem）
                    ↓
Runtime: 直接调用 C++ launch，<1μs
```

**意义**：在 WaveEP 这类需要频繁触发小 wave 的场景中，launch overhead 会成为瓶颈。将开销降至 <1μs 后，细粒度 wave（如 64 tokens/wave）才具备可行性。

### 3.3 Z3 SMT 整数分析

TileLang 在编译时使用 Z3 SMT 求解器验证：

```python
# TileLang 自动检查（无需手动断言）：
# - seq_len 能被 tile_size 整除？
# - wave_size × wave_num = total_tokens？
# - GEMM 的 K 维度与 tile_k 的整除关系？
# - A2A chunk_size 与 expert_tile_size 的公因数？

# 如果不满足，编译时报错，而非运行时 silent error
```

这样可以避免在 Triton kernel 中大量手写 `assert seq_len % BLOCK_SIZE == 0`。

---

## 4. TileLang vs Triton DSL

| 维度 | Triton DSL | TileLang |
|------|-----------|---------|
| 抽象层次 | Tile 级，每次处理 1 个 tile | Tile 级，支持多 tile pipeline |
| Pipeline 描述 | 手动（`tl.multiple_d`） | 声明式 pipeline descriptor |
| Host 调度 | Python wrapper（高 overhead）| 自动生成 C++ launcher |
| 整数约束 | 手动 assert | Z3 SMT 自动验证 |
| 通信绑定 | 无 | A2A chunk 直接绑定 tile |
| 适用场景 | 单 kernel 优化 | 通算融合 kernel |

> [!note]
> TileLang 和 Triton 不是替代关系。TileLang 可以将 Triton 作为 codegen 后端，
> 等同于"在 Triton 之上加了一层声明式编程模型"。

---

## 5. Tile-Level IR 的通用意义

TileLang 是 **tile-level IR** 这个更广泛概念的一个实现。类似思路在多个项目中出现：

| 项目 | Tile-Level 概念 | 场景 |
|------|----------------|------|
| **TileLang（DeepSeek V4）** | TileLang IR | MoE/mHC 融合 kernel |
| **FlexAttention（PyTorch）** | BlockMask + block 粒度 | 稀疏 Attention |
| **MLIR Linalg Tiling** | 结构化 tiling 变换 | 通用 matmul/conv |
| **Triton 3.x TMA** | Tensor Memory Accelerator descriptor | H100 async copy |
| **CUTLASS 3.x CuTe** | CuTe layout algebra | GEMM tile 描述 |

**收敛趋势**：不同项目都在向"tile-level 声明式描述 + 自动代码生成"收敛，区别只在于 tile 的语义范围（单 kernel vs 跨 kernel 通算）。

---

## 6. 对图优化 Pass 体系的影响

```
未来的编译流程（有 tile-level IR 时）：

图 Pass（ATen IR 层）：
  决定"哪些算子融合" → 输出融合算子组

Tile-Level IR（新增层）：
  决定"如何 tile 这个融合算子组"
  决定"tile 粒度与通信 chunk 的绑定"
  决定"pipeline depth（prefetch 几个 wave）"
  ↓ 输出：TileLang IR

Kernel CodeGen（Triton/CUDA）：
  输入 TileLang IR → 生成高效 kernel + C++ launcher
```

**关键好处**：
1. 图 Pass 和 Kernel 通过 Tile-Level IR 解耦，各自可以独立优化
2. WaveEP 的 wave 粒度可以在 Tile-Level IR 中统一描述，而非分散在 graph scheduler 和 Triton kernel 中
3. 跨 kernel 的 tile 对齐（如 GEMM tile size = A2A chunk size）可以在 Tile-Level IR 层统一约束

---

## 7. 关键挑战

```
挑战 1：动态形状下的 Z3 验证困难
  - seq_len 是运行时动态值时，静态 SMT 求解失效
  - 需要 symbolic shape 支持（类似 torch 的 SymInt）

挑战 2：Tile-Level IR 的标准化
  - TileLang（DeepSeek）、Linalg Tiling（MLIR）、CuTe（NVIDIA）各自为政
  - 缺乏跨框架的 tile-level IR 标准

挑战 3：自动 tile 大小搜索
  - 最优 tile 大小依赖硬件特性（shared memory 大小、register 数量）
  - 需要与 Triton autotuning 集成
  - WaveEP 场景下还需考虑 A2A 带宽
```

---

## Related Pages

- [[31_comm_compute_fusion_guide]] — WaveEP 中 TileLang 的通算融合应用
- [[26_flex_attention_analysis]] — FlexAttention 的 BlockMask 与 tile-level 概念
- [[10_mlir_core_concepts]] — MLIR Linalg Tiling（tile-level IR 的 MLIR 实现）
- [[30_triton_vs_mlir_backend_analysis]] — Triton DSL 与 TileLang 的定位对比
- [[npu_mlir_backend_technical_analysis]] — NPU 端 tile-level 优化的不同路径
