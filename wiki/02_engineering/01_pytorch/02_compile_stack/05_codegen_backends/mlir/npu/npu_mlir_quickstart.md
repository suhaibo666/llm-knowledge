---
title: "NPU MLIR 后端 Quick Start"
---

# NPU MLIR 后端 Quick Start

> 层次：quick start（浅、实用）  
> 核验基准：torch_npu **v2.7.1.post5**（所有 env/config/接口均对照源码核实，下文括注 path）  
> 最后更新：2026-06-13

---

## 1. 这是什么 / 何时用

在 Ascend NPU 上，`torch.compile` 有三条后端路径，由环境变量 `TORCHINDUCTOR_NPU_BACKEND` 选择（核验：`torch_npu/_inductor/__init__.py:34,47,57`）：

| 取值 | 路径 | codegen |
|------|------|---------|
| `default`（缺省） | Triton 后端 | `NPUCombinedScheduling` |
| `mlir` | **MLIR / ascend_npu_ir 本页主角** | `NpuMlirScheduling` → 毕昇编译器 `bishengir-compile` |
| `dvm` | DVM 融合 | `NpuMlirScheduling` + dvm |

MLIR 路径把 FX 子图经 torch-mlir 转成 MLIR，再交毕昇编译器生成融合算子（`.o`/`.so`），适合「想要算子级深度融合 + 自动 tiling 调优、且已装好 torch-mlir 与毕昇编译器」的场景。三条路径的取舍详见 [[01_npu_compile_paths_overview]]（含 ACLGraph 图执行路径对比）。

> 前置：MLIR 路径依赖 `torch-mlir`（`torch_npu/_inductor/__init__.py:48-52` 未装会直接 `ImportError`），以及毕昇编译器 `bishengir-compile`（由 env `BISHENG_INSTALL_PATH` 定位，见 §4）。

---

## 2. 如何启用 MLIR 后端

三种方式任选其一（与 `torch_npu/_inductor/ascend_npu_ir/README.md` 一致，且经 `torch_npu/utils/_dynamo.py:207-222` 核实三者最终都落到 env `TORCHINDUCTOR_NPU_BACKEND="mlir"`）：

```python
# 方式 1：环境变量（必须在 import torch_npu / 首次 compile 之前）
import os
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "mlir"
import torch, torch_npu

# 方式 2：inductor config（compile 之前）
torch._inductor.config.npu_backend = "mlir"

# 方式 3：compile options
compiled = torch.compile(fn, options={"npu_backend": "mlir"})
```

后端注册发生在 import 期：`register_backend_for_device("npu", NpuMlirScheduling, NpuMlirWrapperCodeGen)`（`.../npu/npu_inductor_plugin.py:96`）。若 env `TORCHINDUCTOR_USE_AKG=1` 则优先用 AKG，失败再回落到 `NpuMlirScheduling`（同文件 `:87-96`）。

**两个关键开关（在 anir config 里，均为模块属性而非 env）**——核验：`torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/config.py`：

- `compile_mode`（`:146`，默认 **`"auto_fallback"`**）：
  - `"auto_fallback"`：单算子 MLIR 编译失败时自动回落到 FX eager（默认，最稳）。
  - `"complete_fallback"`：完全不做 MLIR 编译，整图走 FX eager，**仅用于调试**。
  - `"default"`：全量 MLIR 编译，**源码注释明确「尚未完全支持」**，能力成熟后才设为缺省。
- `enable_graph_trace`（`:22`，默认 **`True`**）：启用后 NPU 自定义 scheduler 融合/last-usage 逻辑（`npu_inductor_plugin.py:391-396` 据此 patch `Scheduler`）。

修改方式：
```python
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir import config as anir_config
anir_config.compile_mode = "complete_fallback"   # 调试时整图退回 eager
```

---

## 3. 关键 env / config 速查表（仅列源码确认存在的）

均核验自 anir config（`torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/config.py`，简称 anir_config）与 `.../npu/mlir_compiler.py`。

### Autotune / 编译

| 名称 | 类型 | 默认 | 作用 / 核验 path |
|------|------|------|------------------|
| `AUTOTUNE` | env | `"1"`（开） | tiling 自动调优总开关（anir_config `:94`），开启时给 `bishengir-compile` 加 `-enable-tuning-mode=true`（mlir_compiler.py `:109-110`） |
| `DISABLE_MP_COMPILE` | env | `"0"` | 置 `"1"` 关多进程编译；`multiprocess_compile = autotune and DISABLE_MP_COMPILE=="0"`（anir_config `:95`） |
| `ANIR_MODE` | env | `"O1"` | 仅允许 `O0`/`O1`，否则报错（anir_config `:97-99`） |
| `BISHENG_INSTALL_PATH` | env | `""` | 毕昇编译器目录，拼 `bishengir-compile` 路径（mlir_compiler.py `:88-89`） |
| `anir_config.block_dim` | attr | `48` | 传 `bishengir-compile -block-dim=`（anir_config `:158` / mlir_compiler.py `:94`） |
| `anir_config.extra_command` | attr(list) | `[]` | 追加 `bishengir-compile` 参数（如 `-mlir-print-ir-after-all`）（anir_config `:113`） |
| `anir_config.always_compile` | attr | `False` | 跳过 kernel cache 强制重编（anir_config `:21`） |

**autotune 搜索维度**（mlir_compiler.py `get_autotune_config` `:311-319`）：`ops_reorder∈{True,False}` × `auto_db(multi-buffer)∈{True,False}` × `tiling_size∈range(-10,20,2)`（15 档）= **60 组配置**，逐一编译并 bench 取最快。

### Online accuracy check（精度对比）

| 名称 | 类型 | 默认 | 作用 / 核验 path |
|------|------|------|------------------|
| `INDUCTOR_ASCEND_CHECK_ACCURACY` | env | `"0"` | 开启在线精度对比 `online_acc_comp`（anir_config `:40`）；开启后**强制关闭** `fx_graph_cache`（npu_inductor_plugin.py `:81-82`） |
| `INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL` | env | 空 | 自定义阈值，格式 `rtol=1e-6,atol=1e-5`（anir_config `:79-84`） |
| `ANIR_ACC_CHECK_DURING_TUNE` | env | `"0"` | autotune 阶段对每个 config 与 FX fallback 输出 `torch.allclose` 比对，仅保留通过的（anir_config `:35`，mlir_compiler.py `:400-449`） |

默认阈值（anir_config `:41-45`）：`f32 rtol=1.3e-6`、`f16 rtol=1e-3`、`bf16 rtol=1.6e-2`、`atol=1e-5`。对比实现 `accuracy_pass` 用 `torch.allclose(..., equal_nan=True)`（mlir_compiler.py `:379-393`）。

### Fallback / 调试 dump

| 名称 | 类型 | 默认 | 作用 / 核验 path |
|------|------|------|------------------|
| `NPU_INDUCTOR_FALLBACK_LIST` | env | 空 | 设为 `allfallback` 时令进入 lowering 的算子全部走 fallback（anir_config `:30,168-169`，`fallback_to_aten_mode="all"`） |
| `anir_config.fallback_to_aten_mode` | attr | `"exclude"` | `off`/`include`/`exclude`（注释见 anir_config `:160-166`）：`exclude`=不在 `GENERATE_LIST` 的 aten IR 回落 aten；`include`=`FALLBACK_LIST` 内回落 |
| `ANIR_FALLBACK_WARNING` | env | `"0"` | kernel 回落 FX 时打印性能下降告警（anir_config `:116`，mlir_compiler.py `:212-214`） |
| `ANIR_FALLBACK_DUMP` | env | `"0"` | 回落时 dump FX 子图（anir_config `:37`） |
| `ANIR_RUNTIME_ERROR_DUMP` | env | `"0"` | 运行期出错 dump（anir_config `:36`） |
| `ANIR_DEBUG` / `ANIR_DEBUG_DIR` | env | `"0"` / `$PWD/anir_debug` | 调试 dump 总开关与目录（anir_config `:115,129-130`） |
| `FX_SUBGRAPH_DUMP_PATH` | env | `None` | dump named-op MLIR / kernel 产物的目录（anir_config `:136`） |
| `anir_config.force_fallback_kernel_names` / `_paths` | attr(set) | `{}` | 按 kernel 名/产物路径强制跳过该 kernel（anir_config `:118-126`，mlir_compiler.py `:435-438`） |

> 注：`torch_npu/_inductor/config.py`（Triton 路径配置）里也有同名 `INDUCTOR_ASCEND_CHECK_ACCURACY` 等开关，但 MLIR 路径以上表的 **anir config** 为准。

---

## 4. 如何确认确实走了 MLIR 路径

1. **生成代码**：`run_and_get_code` 抓到的 wrapper 里出现 `async_compile.mlir_auto_fallback(...)`（auto_fallback 模式）或 `async_compile.mlir(...)`（default 模式），kernel 名前缀为 `mlir_`、调用形如 `mlir_fused_mul_0.run(...)`。核验：`.../npu/codegen/mlir.py` `_get_kernel_prefix()->"mlir"`（`:70-71`）、`_get_compile_api()->"mlir_auto_fallback"`（`:67-68`）；README 示例同。
2. **毕昇编译器调用日志**：`anir` logger（前缀 `[INFO] ANIR ...`）打印
   `Start to compile, command is: [<BISHENG_INSTALL_PATH>/bishengir-compile -enable-hfusion-compile=true --enable-bin-relocation=0 -block-dim=48 ...]`
   成功后 `[bisheng-compile success]`。核验：mlir_compiler.py `:122-125`；编译 worker 内 logger 被设为 INFO（`.../npu/utils.py:171`）。
3. **磁盘产物**（cache 目录，`TORCHINDUCTOR_CACHE_DIR` 下）：`<kernel>_named_op.mlir`、静态 shape 的 `<kernel>_<tiling>_<reorder>_<db>.o`、动态 shape 的 `lib<...>.so`，以及 launcher `<kernel>.so`。核验：mlir_compiler.py `:138-171,233-262`。

---

## 5. 常见问题

- **算子 fallback（白/黑名单）**：anir config 维护 `GENERATE_LIST`（`POINTWISE_OPS + NON_POINTWISE_OPS`，会被 lower 到 MLIR）与 `FALLBACK_LIST`（如 `aten.mm/bmm/addmm/convolution/embedding/gather/scatter` 等回落 aten）。由 `fallback_to_aten_mode` 控制策略，整图强制回落用 `NPU_INDUCTOR_FALLBACK_LIST=allfallback`。核验：anir_config `:171-279`。单 kernel 编译失败时按 `compile_mode="auto_fallback"` 自动退回 FX eager（mlir_compiler.py `register_fx_fallback` 路径）。
- **动态 shape**：动态分支会给 `bishengir-compile` 加 `--enable-static-bare-ptr=false --enable-symbol-analysis=true`，并走 tiling host 函数 + `lib*.so`（mlir_compiler.py `:112-114,173-197`）。细节与限制见下方 deep dive。
- **精度怀疑**：先开 `INDUCTOR_ASCEND_CHECK_ACCURACY=1`（在线对比），或 `ANIR_ACC_CHECK_DURING_TUNE=1`（调优期就过滤掉不达标 config）。
- **想定位坏 kernel**：用 `anir_config.force_fallback_kernel_names={...}` 或 `force_fallback_kernel_paths={...}` 跳过指定 kernel。

---

## 6. 深入导航

- [[npu_mlir_backend_technical_analysis]] — MLIR 后端架构、TracedGraph、融合规则、毕昇编译、monkey patch 全解（deep dive）
- [[12_npu_compile]] — NPU Inductor 编译工作流
- [[02_compile_stack/04_inductor/npu/index]] — NPU Inductor 文档入口

---

## Related Pages

- [[01_npu_compile_paths_overview]] — torch.compile 三路径全景（Triton / MLIR / ACLGraph）
- [[npu_mlir_backend_technical_analysis]] — MLIR 后端深度分析
- [[30_triton_vs_mlir_backend_analysis]] — Triton vs MLIR 后端对比
- [[12_npu_compile]] — Inductor 编译工作流
- [[01_pytorch/index]] — AI 框架领域入口
