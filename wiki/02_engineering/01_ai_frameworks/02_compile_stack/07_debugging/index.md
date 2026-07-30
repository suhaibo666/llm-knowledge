# 07 · 调试与诊断 — 目录索引

> 卷别：`torch.compile` 调试、正确性与性能（原课程卷 E，2026-07-30 随 P4 两级重组物理迁入本目录并去 `eNN_` 前缀）
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> 最后更新：2026-07-30

`torch.compile` 一次失败或一次变慢，可能落在 Dynamo 捕获、AOTAutograd、Inductor 编译、
native/Triton 编译、module load 或 runtime 六个阶段中的任意一个。本目录九篇按证据层级、
失败分层定位、修复策略与生产上线组织，从"如何读懂一条日志"到"如何在生产环境安全回滚"。

## 九篇一览

| 页面 | 一句话定位 |
|---|---|
| [[observability_logs_counters_and_artifact_map_analysis]] | 建立证据层级：log、artifact、counter 分别能证明什么、不能证明什么 |
| [[dynamo_explain_and_graph_break_diagnosis_analysis]] | 用 `explain` 与 `graph_breaks` 定位捕获失败与切图原因 |
| [[guard_failure_and_recompile_diagnosis_analysis]] | 定位 recompile storm：cache entry 选择失败的根因分类与修复 |
| [[aotautograd_and_inductor_failure_localization_analysis]] | 用 backend 阶梯（eager→aot_eager→decomp/partition→inductor）做失败分层二分 |
| [[minifier_repro_and_compiler_bisector_analysis]] | Repro、Minifier 与 Compiler Bisector 三个正交工具的选用边界 |
| [[compiled_correctness_validation_methodology_analysis]] | 值、梯度、mutation、alias、effect 六维正确性验证方法论 |
| [[compile_latency_cache_and_steady_state_performance_analysis]] | 冷启动、cache hit、稳态三类场景分开测量与 break-even 分析 |
| [[kernel_fusion_memory_and_hardware_performance_analysis]] | 从图结构到 fusion、内存生命周期、硬件计数器的四层性能归因 |
| [[production_rollout_fallback_and_monitoring_analysis]] | 分阶段上线、fallback 分级、SLO 与自动回退、回滚演练 |

**建议阅读顺序**：observability → explain/graph break → guard failure → failure
localization → minifier/bisector → correctness → latency/cache → kernel/硬件性能 →
production rollout。九篇按此顺序互为前置/后续（见各页头部"前置/后续"行），先建立证据
层级和分层定位方法，再进入具体的正确性/性能验收和上线策略。

## 附录：分布式 + torch.compile 全链路排查包

> 本附录内容迁移自已删除的 `Pytorch_Compile_Debug_Analysis.md`（558 行，原是本卷九篇的
> 压缩前身）。该页逐节判重后，机制性内容（Dynamo/FX/Guards/AOT/Inductor 各阶段的原理与
> 定位方法）已被上表九篇更严谨的源码级分析取代——尤其 [[dynamo_explain_and_graph_break_diagnosis_analysis]]、
> [[guard_failure_and_recompile_diagnosis_analysis]]、
> [[aotautograd_and_inductor_failure_localization_analysis]] §8 的 backend 阶梯决策树。
> 但该页提供的**可直接运行的排查脚本**、**分布式专属决策分支**与**kernel/CUDA 层崩溃诊断**
> 九篇均未覆盖，逐字保留于此，作为九篇分析之外的操作性附件。

### 快速前置（必读）

* **务必在 `import torch` 之前设置环境变量。** 在 Notebook 中请在第一个 cell 设置并重启 kernel。
* 推荐在排查时使用 `tee` 将 stdout/stderr 打到文件（每个 rank 一个日志文件）。
* 运行结束后，记得把 `/tmp/torchinductor_*` 目录以及日志归档到 `artifacts/` 下。

### 环境变量一键启动脚本 — `run_debug.sh`

将下面脚本保存为 `run_debug.sh`，赋可执行权限后，按说明运行（可修改 `NUM_GPUS`、`ENTRY_SCRIPT`、`RUNDIR`）。

```bash
#!/usr/bin/env bash
set -euo pipefail

# 配置区（按需修改）
NUM_GPUS=${NUM_GPUS:-8}
ENTRY_SCRIPT=${ENTRY_SCRIPT:-train.py}
RUNDIR=${RUNDIR:-./debug_run}

mkdir -p "$RUNDIR"

# 1) 环境变量（针对 PyTorch 2.7-2.10）
# 基础日志配置
export TORCH_LOGS="+dynamo,guards,graph_breaks,recompiles,aot,inductor,schedule,codegen"
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR=./torch_compile_debug
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# CUDA 特有环境变量（仅在使用 CUDA 时设置）
if [ "${DEVICE_TYPE:-cuda}" = "cuda" ]; then
    export TORCHINDUCTOR_DUMP=1
    export TORCHINDUCTOR_VERBOSE=1
    export TORCHDYNAMO_VERBOSE=1
    export TORCH_NCCL_DEBUG=INFO
fi

# 2) 启动（示例使用 torchrun）
# 注意：如果你的训练脚本需要额外 args，请把它们追加在末尾

RANK=0
# 使用 torchrun 启动，会为每个进程产生 stdout/stderr；我们用 tee 保存
# 在多节点场景下，你可以把此脚本作为各节点的启动脚本
torchrun --nproc_per_node=${NUM_GPUS} ${ENTRY_SCRIPT} 2>&1 | tee "${RUNDIR}/compile_debug_rank${RANK}.log"

# 3) 归档 / 拷贝调试产物
mkdir -p "${RUNDIR}/artifacts"

# CUDA: 归档 inductor dump
if [ "${DEVICE_TYPE:-cuda}" = "cuda" ]; then
    cp -r /tmp/torchinductor_* "${RUNDIR}/artifacts/" 2>/dev/null || true
fi

echo "Debug run complete. Artifacts collected under ${RUNDIR}/artifacts"
```

**说明**：

* 对于真实多卡（每个进程有独立 stdout），你应该在 launcher 层让每个 rank 的 stdout 分别写入 `compile_debug_rank${RANK}.log`。如何在你的集群环境中实现按 rank 日志分离，取决于 launcher（例如一些集群默认会将每个 rank 的 stdout 写到不同文件）。

### Export + Backend 捕获脚本

**目的**：用 API 方式保存可复用的 FX `GraphModule`（用于离线对比），并在 `torch.compile` 时把 backend 接收到的最终 `GraphModule` 写盘。两部分代码分别为 `export_capture.py` 和 `capture_backend.py`。这是公开 `export()` API + 自定义 backend 的轻量方案，与 [[minifier_repro_and_compiler_bisector_analysis]] 描述的官方 `after_dynamo`/`after_aot` repro 生成器是两条正交路径：官方生成器在编译失败时自动触发，这里的脚本用于主动、无需失败即可捕获的离线基线对比。

#### `export_capture.py`（最小可运行）

```python
# export_capture.py
import os
import time
import json
import torch
from torch._dynamo import export

# ---- 配置区 ----
MODEL_FN = 'build_model'  # 模块内构造模型的函数名
MODULE_PY = 'your_model_module'  # 把你的模型放到一个可 import 的模块里
EXAMPLE_INPUT = (torch.randn(1,3,224,224),)
OUTDIR = './debug_dumps/export'
# -----------------

os.makedirs(OUTDIR, exist_ok=True)

# import user module dynamically
m = __import__(MODULE_PY, fromlist=[MODEL_FN])
model = getattr(m, MODEL_FN)().eval()

# export（会捕获 control flow）
gm, guards = export(model, EXAMPLE_INPUT)

ts = int(time.time())
prefix = os.path.join(OUTDIR, f'gm_{ts}')
os.makedirs(prefix, exist_ok=True)

# 保存源码
with open(os.path.join(prefix, 'graph_code.py'), 'w') as f:
    f.write(gm.code)

# 保存 graph repr
with open(os.path.join(prefix, 'graph_repr.txt'), 'w') as f:
    f.write(str(gm.graph))

# 保存可 load 的 GraphModule（state_dict）
torch.save({'gm_code': gm.code, 'state_dict': gm.state_dict()}, os.path.join(prefix, 'gm_snapshot.pt'))

# 保存 guards
with open(os.path.join(prefix, 'guards.json'), 'w') as f:
    json.dump([str(g) for g in guards], f, indent=2)

print('Exported gm to', prefix)
```

**使用**：先把模型放进 `your_model_module.py` 并实现 `build_model()`，然后运行 `python export_capture.py`。

#### `capture_backend.py`（供 `torch.compile` 使用的 backend 截获器）

```python
# capture_backend.py
import os
import time
import torch
from torch.fx import GraphModule

OUTDIR = os.environ.get('BACKEND_CAPTURE_DIR', './debug_dumps/backend')
os.makedirs(OUTDIR, exist_ok=True)


def capture_backend_factory():
    def backend(gm: GraphModule, example_inputs):
        ts = int(time.time()*1000)
        dirname = os.path.join(OUTDIR, f'gm_capture_{ts}')
        os.makedirs(dirname, exist_ok=True)
        # gm.code
        with open(os.path.join(dirname, 'gm_code.py'), 'w') as f:
            f.write(gm.code)
        with open(os.path.join(dirname, 'gm_graph.txt'), 'w') as f:
            f.write(str(gm.graph))
        try:
            torch.save({'state_dict': gm.state_dict()}, os.path.join(dirname, 'state_dict.pt'))
        except Exception:
            # gm may not be a Module with state_dict; ignore
            pass
        print(f'[capture_backend] saved gm to {dirname}')
        # Return a callable: use gm.forward (no further compilation)
        return gm.forward
    return backend

# Usage example (save as module and import in your train script):
# from capture_backend import capture_backend_factory
# opt_model = torch.compile(model, backend=capture_backend_factory())
```

**说明**：把 `capture_backend.py` 放到你的代码仓里，在训练脚本中把 `torch.compile(model, backend=capture_backend_factory())` 临时替换或注入（可以在调试时做），这样就能在 backend 层拿到实际传入 backend 的 `GraphModule` 并保存。

### 灾难恢复 / 归档脚本 — `collect_artifacts.sh`

用于把 `/tmp/torchinductor_*` 及日志统一收集到一个可上传/分析的 `artifacts` 目录。

```bash
#!/usr/bin/env bash
set -euo pipefail
OUTDIR=${1:-./artifacts_$(date +%s)}
mkdir -p "$OUTDIR"

# 拷贝 inductor dump
cp -r /tmp/torchinductor_* "$OUTDIR/" 2>/dev/null || true

# 拷贝 compile 日志
cp -r compile_debug_rank*.log "$OUTDIR/" 2>/dev/null || true

# 打包
tar -czvf "${OUTDIR}.tgz" -C "$(dirname "$OUTDIR")" "$(basename "$OUTDIR")"

echo "Artifacts archived to ${OUTDIR}.tgz"
```

### 日志对比脚本 — `diff_rank_logs.sh`

快速对比每个 rank 的关键日志字段（graph_breaks / guards / recompiles / aot / inductor）。九篇均未提供多 rank 日志对比工具，此脚本是分布式排查专属留存。

```bash
#!/usr/bin/env bash
set -euo pipefail

LOGS=(compile_debug_rank*.log)
if [ ${#LOGS[@]} -eq 0 ]; then
  echo "No compile debug logs found"
  exit 1
fi

for f in "${LOGS[@]}"; do
  echo "===== $f ====="
  grep -nE "GRAPH BREAK|GUARD|GUARD FAIL|recompil|aot|inductor|falling back" "$f" || true
  echo
done

# 简单 diff（可按需改进）
if [ ${#LOGS[@]} -gt 1 ]; then
  echo "===== DIFF rank0 vs rank1 (raw) ====="
  diff -u "${LOGS[0]}" "${LOGS[1]}" | sed -n '1,200p'
fi
```

### 决策树第 5 分支：仅在多卡/分布式场景复现

原页决策树共 5 步：GRAPH BREAK 多→Dynamo 问题、recompiles/GUARD FAIL 频繁→guard 问题、
forward 正常但 backward 异常→AOT 问题、数值/性能异常→Inductor 问题、**仅在多卡/分布式
出现→分布式问题**。前 4 步已被 [[dynamo_explain_and_graph_break_diagnosis_analysis]]、
[[guard_failure_and_recompile_diagnosis_analysis]]、
[[aotautograd_and_inductor_failure_localization_analysis]] §8 的 backend 阶梯决策树取代
且更严谨，故不再重复；第 5 步九篇未覆盖，逐字保留：

> 5. **若问题仅在多卡/分布式出现**
>
>    * 分布式问题（数据 split / padding / seed / DDP wrap order / NCCL）→ 单卡/两卡复现后对比 logs

### Kernel / Triton / CUDA 层崩溃诊断（最终执行）

九篇中 [[kernel_fusion_memory_and_hardware_performance_analysis]] 只覆盖 kernel/fusion 的
**性能**归因（roofline、fusion score、occupancy），[[aotautograd_and_inductor_failure_localization_analysis]]
§5 的 "Native compile/load failure"、"Runtime failure" 只有两句话概述；两者都未覆盖以下
**崩溃类**故障（segfault、OOM、launch failed、arch mismatch）的具体关键词与修复建议，
故逐字保留。

**要捕获的产物**

* 在 `/tmp/torchinductor_*` 下的 kernel 源（triton/cpp/cu/ptx 等）、编译输出、任何 `stderr` 的编译器信息。
* 运行时的 CUDA 错误信息（`cuda error`、`invalid device function`、`segfault`）。

**如何识别问题**

* 关键词：`cuda error`、`segfault`、`out of memory`、`invalid configuration`、`launch failed`、`triton`。

  ```bash
  grep -nE "cuda error|segfault|out of memory|launch failed|triton" compile_debug_rank*.log
  ```
* 若有 core dump / segfault，获取 core 栈或把程序运行在 gdb 中复现（通常需要开发环境支持）。

**解释与快速定位**

* kernel 层错误通常是：资源超配（shared mem / regs / block dims）、triton 编译器 bug、或者生成的 kernel 不适合当前 GPU 架构（arch mismatch）。
* 当出现数值差异时，也应确认 kernel 的浮点逐步计算顺序（fused reductions 或 reorder）是否不同。

**快速修复建议**

1. 缩小问题：把输入 size 缩小到能在单卡上重现。
2. 强制单线程/单 block 运行（若可能）或禁用 fusion（把问题回退到 aten）以确定是否 kernel 引起问题。
3. 若怀疑 triton bug：将 `output_code.py` 与 kernel 源和运行脚本打包提交给 upstream（附上最小 reproducer）。
4. 尝试在另一台 GPU（或另一 driver 版本）上重现以排除环境问题。

### 使用流程（工程化）

1. **单卡快速复现**：先在单 GPU 上运行 `run_debug.sh` 修改 `NUM_GPUS=1`，排除分布式问题。
2. **导出 FX（可复用）**：在单卡上运行 `export_capture.py` 保存 `gm`，用于离线比对。
3. **后台捕获 backend GM**：在模型上临时使用 `capture_backend_factory()` 得到 backend 层的 GM。
4. **运行多卡（2卡→全卡）**：按 `run_debug.sh` 启动，收集每个 rank 的日志与 `/tmp/torchinductor_*`。
5. **归档并对比**：用 `collect_artifacts.sh` 和 `diff_rank_logs.sh` 做初步对比。
6. **若是 Inductor 问题**：提取单 kernel reproducer，向 upstream 提交 issue 或在本地调试 kernel/triton 代码。

### 附：小技巧与注意事项

* Notebook: 在第一个 cell 设置 `os.environ[...]` 并重启 kernel。
* 若你需要把 `/tmp/torchinductor_*` 固定到其他路径：部分版本提供 `TORCHINDUCTOR_CACHE_DIR` 或类似变量，若不支持请手动归档。
* 对比 `gm.text`/`code` 建议用 `git diff --no-index` 或 `diff -u` 做文本 diff。
* 对于大模型，导出和保存要控制样本大小（抽样）以免生成过大文件。
* 报 issue 时务必附上最小可复现脚本、完整 log、`/tmp/torchinductor_*` 相关文件、以及 export 的 gm（参见下方"提交问题给 upstream / 报 issue 时要附的最小清单"）。

### 附：提交问题给 upstream / 报 issue 时要附的最小清单

[[minifier_repro_and_compiler_bisector_analysis]] §11 已给出通用的"交付一个有效 repro"标准
（干净进程可复现、无私有数据、版本明确、predicate 稳定等），偏重 minifier/bisector 场景。
以下是原页面向 CUDA + 分布式场景给出的具体附件清单，两者互补，不重复列出通用项：

当你把问题上报给 PyTorch / Inductor / Triton 团队时，尽量打包并提供以下内容（便于快速定位）：

1. **最小可复现脚本**（`run_repro.py`）与 `requirements.txt`（包含 PyTorch + CUDA + triton 版本）
2. `compile_debug_rank0.log`（完整日志）
3. `/tmp/torchinductor_*` 下相关 kernel 源与 `output_code.py`（压缩包）
4. `export` 导出的 `gm_code.py` / `gm_graph.txt` / `guards.json` / `gm_snapshot.pt`
5. 复现步骤（命令），说明是否单卡/多卡复现、GPU 型号、driver 以及 NCCL/OSOS 信息
6. 若存在数值不一致，给出对比数据（eager vs compiled）的小样本

## Related Pages

- [[00_torch_compile_end_to_end_index]] — 端到端课程总索引
- [[02_compile_stack/index]] — torch.compile 编译栈领域索引
- [[02_compile_stack/04_inductor/index]] — Inductor 领域索引（本卷 CUDA 排查与 §5 越界排查互补）
- [[02_compile_stack/04_inductor/npu/npu_debug_guide]] — NPU 侧调试（本附录是纯 upstream/CUDA 视角）
- [[d07_compiled_artifact_lifecycle_and_runtime_failures_analysis]] — 编译产物生命周期与 runtime failure（本卷前置卷 D）
