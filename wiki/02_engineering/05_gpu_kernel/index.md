# GPU Kernel 开发 — 目录索引

> 覆盖 GPU/NPU Kernel 的执行模型、内存优化、Tensor Core、torch.compile、FlashAttention 与架构差异
> 最后更新: 2026-06-09

---

## 页面列表

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[gpu_kernel_guide]] | 综合深度分析 | 执行层级(Grid/Block/Warp/Thread/SM/Tile)、内存优化(Coalesced/Shared/Tiling/Bank Conflict)、Occupancy、Warp Divergence、异步Pipeline、Tensor Core/MMA、torch.compile 编译路径、FlashAttention 完整链路、NPU 架构差异、诊断清单 |

---

## 关联域

- [[../01_ai_frameworks/index]] — AI 框架 (PyTorch 编译栈)
- [[../02_train_frameworks/index]] — 训练框架 (Kernel 是上层掩盖的基础)
- [[../03_infer_frameworks/index]] — 推理框架 (FlashAttention 等 kernel 是推理性能关键)
