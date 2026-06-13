# PyTorch 编译与运行时架构 — 知识地图

> 本域(`01_ai_frameworks`)按 **PyTorch 编译/运行时架构**组织。每个功能目录下,**硬件无关的通用机制**置于该目录层级,**硬件特定实现**下沉到 `npu/`、`cuda/` 等硬件子目录。
> 最后更新: 2026-06-13

---

## 架构总览(torch.compile 流水线 + 运行时)

```
  算子供给侧                                                运行时分发
(07_op_registration)                                    (01_dispatcher_and_device)
      │                                                        ▲
      ▼              @torch.compile                            │ aten 算子分发
  User Code ──► TorchDynamo ──► AOTAutograd ──► TorchInductor ──► Kernel 执行
                  (02)            (03)             (04)             │
                图捕获/Guard    前/反向分解     lowering→调度→codegen │
                                                     │              ▼
                                            codegen 后端(05)   图捕获/回放(06)
                                            Triton / MLIR      CUDA Graphs / NPU Graphs
```

详见 [[torch_compile_architecture]] 端到端流水线分析。

---

## 功能目录(按架构顺序)

| 目录 | 功能 | 硬件子目录 |
|------|------|-----------|
| [[01_dispatcher_and_device/index]] | 运行时:算子分发(Dispatcher)、PrivateUse1 设备接入 | 通用 |
| [[02_dynamo/index]] | torch.compile 前端:帧评估图捕获、Guard | 通用 |
| [[03_aot_autograd/index]] | 前/反向图分解、partition、functionalization | 通用 |
| [[04_inductor/index]] | 编译后端核心:lowering、调度、codegen、FX passes、动态形状 | `npu/` NPU Inductor 后端 |
| [[05_codegen_backends/index]] | codegen 后端:MLIR(及 Triton 对比) | `mlir/` + `mlir/npu/` |
| [[06_graphs/index]] | 运行时图捕获:CUDA Graphs / NPU Graphs(ACLGraph) | `cuda/` + `npu/` |
| [[07_op_registration/index]] | 算子接入供给侧:op-plugin 配置/注册/入图判别 | `npu/` |
| [[08_kernel_optimization/index]] | 算子调优(GPU/NPU Roofline、融合)、TileLang | 通用 |
| [[09_other_frameworks/index]] | 非 PyTorch 框架对照:MindSpore 编译器 | — |

---

## 硬件分层约定

- **通用页**(无硬件后缀)位于功能目录层级,描述 PyTorch 上游通用机制。
- **NPU 特定页**位于 `<功能>/npu/`,基于 `torch_npu` / `op-plugin`(当前核验基准 = **v2.7.1.post5**)。
- **CUDA 特定页**位于 `<功能>/cuda/`。
- 跨硬件对比页(如 NPU vs CUDA Graphs)放硬件目录内并双向 backlink。

---

## 知识分层约定(overview → quick start → deep dive)

每个(子)模块内的知识按「深入浅出」分三层,索引页用 **层次** 列标注:

- **overview**(浅):该模块是什么 / 为什么 / 全景图 / 如何选型,5 分钟通读;通常即各目录的 `index.md` 或显式 overview 页。
- **quick start**(用):最小可用路径、关键 API、可跑示例、常见开关(并非每模块都有,按需)。
- **deep dive**(深):源码级深度分析(多为既有大文),按主题拆分,一概念一页。

阅读建议:先 overview 建立全局 → 需要动手看 quick start → 钻研细节看 deep dive。

---

## 知识空白

- **TorchDynamo guard 失败调试** — 常见但未系统记录
- **Inductor autotuning** — Triton kernel autotuning 策略
- **NPU Monkey Patch 演进追踪** — v2.7.1 → v2.9.0 → master,每次 PyTorch 升级需人工对齐内部接口
- **CATLASS/CK GEMM 模板库生态** — 社区 CUTLASS 与 NPU CATLASS 差异
- **IR 回溯机制通用性** — MLIR 路径 FX Graph 重建的泛化方案
- **Multi-backend dispatch** — Inductor 在 CUDA/NPU 间选择逻辑(部分覆盖,见 [[npu_compile_paths_overview]])
- **IREE 实际 Pass 细节** — Flow/Stream Dialect 具体 Pass 列表
- **TileLang 源码分析** — 实现未开源,当前分析基于论文
- **Triton 3.x MLIR 迁移进度** — TMA 以外特性支持状态

---

## 关联域

- [[02_train_frameworks/megatron-lm/index]] — Megatron-LM(CUDA Graphs 使用场景)
- [[05_gpu_kernel/index]] — GPU Kernel 开发(执行层级、内存优化、NPU 差异)
- [[04_posttrain_frameworks/batch_invariance_guide]] — 批不变性与 torch.compile
- [[01_theory/index]] — 理论研究
