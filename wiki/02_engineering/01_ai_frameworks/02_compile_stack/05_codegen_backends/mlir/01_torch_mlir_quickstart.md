# torch-mlir 快速上手 (Quickstart)

> 层次：quick start（浅、实用）——torch-mlir 在 PyTorch 生态的定位、何时用、最小 backend 骨架、output_type 选择
> 范围：上游通用 `llvm/torch-mlir`（社区项目，本地代码库不含其源码）。下文 API 名（`FxImporter` / `import_stateless_graph` / `output_type` / `OutputType`）按上游公开约定书写，**以上游为准**，不保证跨版本签名稳定。
> 最后更新：2026-06-15

---

## 1. torch-mlir 是什么 / 在生态中的定位

**torch-mlir 是 PyTorch 生态 → MLIR 生态的「转接层」。** 它把 PyTorch 的计算图（来自 TorchDynamo / `torch.fx` / `torch.export` 等多种前端）翻译成 MLIR 的 **Torch Dialect**，再经一串 Pass 逐层递降到 Linalg / TOSA / StableHLO 等标准 dialect，交给下游 MLIR 工具链（如 IREE、上游 LLVM 后端）继续编译到具体硬件。

它定位为**可组合的编译组件**，而不是一个端到端的生产级编译器——你通常把它接到某条更大的工具链里，而不是单独用它跑模型。

### 与 Inductor 的关系

`torch.compile` 默认后端是 **Inductor**：FX Graph → Inductor IR → Triton kernel（GPU）/ C++ kernel（CPU）。torch-mlir 是**另一条平行的 backend 路线**，两者都通过 `torch.compile(..., backend=...)` 的 backend 机制挂载，但走完全不同的 IR 栈：

```
默认路线 (Inductor，2 层 IR):
  FX Graph → Inductor IR → Triton / C++ → PTX / .so

torch-mlir 路线 (5+ 层 IR):
  FX Graph → Torch Dialect → Linalg/TOSA/StableHLO → ... → 目标硬件
```

简言之：Inductor 偏「FX 直接 lower 到 Triton，少层、快」；torch-mlir 偏「多层标准 IR 递降，可被多种 MLIR 后端复用」。二者不是替代关系，按目标工具链取舍。逐阶段的概念对等映射见 [[30_triton_vs_mlir_backend_analysis]]。

---

## 2. 何时用 torch-mlir

**适合：**

- **多层 IR 递降需求**：你的下游需要标准的 MLIR dialect（Linalg / TOSA / StableHLO），以便复用 MLIR 社区的 tiling / fusion / bufferization / lowering Pass，而不是直接生成 Triton。
- **跨硬件 / 非 NVIDIA 后端**：目标是 IREE、自研 MLIR 编译器、或其它 MLIR-based 工具链（昇腾 NPU 的 MLIR 路径即属此类，但 NPU 有独立适配，见 §6 导航）。
- **需要把 PyTorch 模型「导出成 MLIR 文本」做离线分析 / 二次编译**（`output_type` 直接拿到某一层 IR）。

**不适合 / 不必用：**

- 只想在 NVIDIA GPU 上把现成模型跑快 → 直接用默认 Inductor + Triton，无需引入 MLIR 栈。
- 追求开箱即用的端到端编译器 → torch-mlir 是组件，需自己接 JIT / runtime。

---

## 3. 最小用法骨架

torch-mlir 接入 `torch.compile` 的标准姿势：写一个自定义 backend，把 Dynamo 捕获的 `gm`（`torch.fx.GraphModule`）导入为 MLIR，运行 Pass 管线降级，再返回可调用对象。

### 3.1 底层 API：`FxImporter.import_stateless_graph`

```python
import torch
from torch_mlir.extras.fx_importer import FxImporter   # 导入路径以上游为准

def my_mlir_backend(gm: torch.fx.GraphModule, example_inputs):
    # Step 1: FX Graph → MLIR Torch Dialect
    importer = FxImporter()
    importer.import_stateless_graph(gm.graph)   # 入参为 fx.Graph；方法名/签名以上游为准
    module = importer.module                     # 得到 MLIR Module（Torch Dialect）

    # Step 2: 运行 MLIR Pass 管线，降级到目标后端（Linalg / TOSA / StableHLO）
    #         见 [[11_torch_mlir_pass_pipeline_analysis]]

    # Step 3: 把降级后的 MLIR 交给下游（IREE / 自研 runtime）JIT 编译

    # Step 4: 返回一个 callable，签名与原模型一致
    return compiled_callable

compiled = torch.compile(model, backend=my_mlir_backend)
out = compiled(*example_inputs)
```

要点：

- `import_stateless_graph` 接收的是 `gm.graph`（`torch.fx.Graph`），做的是 **1:1 语法翻译**（`aten.add` → `torch.aten.add`），不涉及优化。所有 shape/dtype 已由 Dynamo 的 fake tensor 在 trace 时确定。
- 「无状态」(stateless) 指它面向 Dynamo 在线捕获的图，**不经过** `torch.export` 的离线导出；二者最终汇入同一条 Pass 管线。

### 3.2 便捷封装：`stateless_fx_import`

上游通常提供一个把「导入 + 降级」打包好的便捷函数，直接用 `output_type` 一步拿到目标 IR：

```python
from torch_mlir.fx import stateless_fx_import   # 模块/函数名以上游为准

def my_mlir_backend(gm, example_inputs):
    module = stateless_fx_import(gm, output_type="linalg-on-tensors")
    # module 已是降级到 Linalg 的 MLIR，再交下游编译
    ...
```

`stateless_fx_import(gm, output_type=...)` 内部即：`FxImporter.import_stateless_graph(gm.graph)` → 运行 Pass 管线 → 后端降级。需要细粒度控制（自定义 Pass、分步调试）时用 §3.1 的底层 API；只想拿某层 IR 时用本封装。

> 离线场景另有 `export_and_import(model, *args, output_type=...)`（内部走 `torch.export`），与上面在线路径殊途同归。具体函数名/参数**以上游为准**。

---

## 4. `output_type` 选项怎么选

`output_type`（取值常以 `OutputType` 枚举表达）决定 torch-mlir 把 IR 降到**哪一层就停下并返回**。常见取值（以上游为准）：

| output_type | 产出 IR | 何时选 |
|-------------|---------|--------|
| `RAW` | 刚导入、未跑任何 Pass 的 Torch Dialect | 想看「原样翻译」结果、自己接管全部 Pass |
| `TORCH` | 满足 backend contract 的规范化 Torch Dialect | 下游只认 Torch Dialect，或想在 Torch 层做分析 |
| `LINALG_ON_TENSORS` | 纯 Linalg / Tensor / Arith / SCF（无 Torch 残留） | **最通用**：复用 MLIR 社区 Linalg→硬件 lowering（IREE、GPU 路径常用） |
| `TOSA` | TOSA dialect | 目标后端吃 TOSA（部分推理 / 边缘编译器、移动端） |
| `STABLEHLO` | StableHLO dialect | 对接 XLA / StableHLO 生态（IREE、跨框架互通） |

经验法则：**不确定就选 `LINALG_ON_TENSORS`**——它消除了全部 Torch 痕迹、被最多下游复用；只有当下游明确要求 TOSA / StableHLO 时才改。`RAW` / `TORCH` 主要用于调试与自定义流水线。各层之间的具体 Pass 序列见 [[11_torch_mlir_pass_pipeline_analysis]]。

---

## 5. 社区资源

- **主仓库**：`llvm/torch-mlir`（LLVM 孵化器项目，活跃维护；FX/ONNX 导入器、TorchToLinalg/TosaStablehlo lowering 都在此）。
- **下游集成**：`iree-org/iree-turbine`（torch-mlir + IREE 的端到端路径）。
- 历史上的 `nod-ai/SHARK-Turbine` 代码已迁出至上述两仓，现为 CI staging 空壳，不再活跃——评估生态以 `llvm/torch-mlir` 与 `iree-org/iree-turbine` 为准。

---

## 6. 深入阅读导航

按「概念 → Pass 管线 → 选型对比」递进：

1. [[10_mlir_core_concepts]] — MLIR 基础：Dialect、Pass、IR 注册、递降原理（先打底）
2. [[11_torch_mlir_pass_pipeline_analysis]] — torch-mlir Pass 管线：本页骨架里「运行 Pass 管线」那一步的完整执行序拆解
3. [[30_triton_vs_mlir_backend_analysis]] — Triton vs Torch-MLIR：六阶段概念对等映射与选型指南

> 昇腾 NPU 的 MLIR 后端是另一条带硬件适配的路线（毕昇编译器、monkey-patch），上手见 [[npu_mlir_quickstart]]，深度分析见 [[npu_mlir_backend_technical_analysis]]。

---

## Related Pages

- [[10_mlir_core_concepts]] — MLIR 核心概念
- [[11_torch_mlir_pass_pipeline_analysis]] — torch-mlir Pass 管线完整执行序
- [[30_triton_vs_mlir_backend_analysis]] — Triton vs Torch-MLIR 后端对比
- [[01_ai_frameworks/index]] — AI 框架领域入口
