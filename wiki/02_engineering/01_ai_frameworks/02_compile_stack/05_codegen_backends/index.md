# 05 · Codegen 后端 — 目录索引

> Inductor IR/FX 之后的代码生成后端。本域聚焦 **MLIR** 路径(核心概念、torch-mlir Pass 管线、NPU MLIR 后端)及其与 **Triton** 的对比选型。
> Triton 在 NPU 上的具体适配见 [[02_compile_stack/04_inductor/npu/index]] 的 [[11_npu_inductor_splittiling_backend_analysis]];CUDA/C++ codegen 见 [[02_compile_stack/04_inductor/index]]。
> 最后更新: 2026-06-13

---

## 子目录

| 目录 | 核心主题 |
|------|---------|
| [[02_compile_stack/05_codegen_backends/mlir/index]] | MLIR 核心概念、torch-mlir Pass 管线、Triton vs MLIR;NPU MLIR 后端(`mlir/npu/`) |

---

## 关联域

- [[02_compile_stack/04_inductor/index]] — Inductor(MLIR/Triton 作为其后端路径)
- [[03_runtime_graphs/index]] — 运行时图捕获
- [[01_ai_frameworks/index]] — 本域总索引
