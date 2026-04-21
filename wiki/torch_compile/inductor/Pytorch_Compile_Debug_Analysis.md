# 分布式 + torch.compile 全链路排查包（PyTorch 2.7–2.10）

> 这个包包含：
>
> * 一份详尽的 Markdown 指南（本文件）
> * 可直接运行/修改的脚本：`run_debug.sh`, `export_capture.py`, `capture_backend.py`, `collect_artifacts.sh`, `diff_rank_logs.sh`
>
> 目的：通过最小入侵或纯环境变量方式，在单卡/多卡场景下收集 `torch.compile` 各阶段（Dynamo、FX、AOT、Inductor、kernel）产物与日志，帮助定位问题归因并给出修复建议。
>
> **支持平台**：CUDA（NVIDIA GPU）、NPU（华为昇腾）

---

## 目录

1. 快速前置（必读）
2. 环境变量一键启动脚本（`run_debug.sh`）
3. Export + Backend 捕获脚本（`export_capture.py`, `capture_backend.py`）
4. 灾难恢复 / 归档脚本（`collect_artifacts.sh`）
5. 日志对比脚本（`diff_rank_logs.sh`）
6. 决策树（精炼版）—— 何时判定 Dynamo / AOT / Inductor / 分布式 问题
7. **每个阶段的捕获要点与日志分析指南（Dynamo / FX / AOT / Inductor / kernel）**
8. 常见问题与快速修复清单
9. 使用流程（工程化）
10. 附：小技巧与注意事项
11. **NPU 特有调试手段（华为昇腾）** ←（新增，支持昇腾 NPU 平台）

---

## 1) 快速前置（必读）

* **务必在 `import torch` 之前设置环境变量。** 在 Notebook 中请在第一个 cell 设置并重启 kernel。
* 推荐在排查时使用 `tee` 将 stdout/stderr 打到文件（每个 rank 一个日志文件）。
* 运行结束后，记得把 `/tmp/torchinductor_*` 目录以及日志归档到 `artifacts/` 下。

---

## 2) 环境变量一键启动脚本 — `run_debug.sh`

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
# 基础日志配置（适用于 CUDA 和 NPU）
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

# NPU 特有环境变量（仅在使用 NPU 时设置）
if [ "${DEVICE_TYPE:-cuda}" = "npu" ]; then
    export ASCEND_GLOBAL_LOG_LEVEL=0  # 0:debug, 1:info, 2:warning, 3:error
    export HCCL_DESYNC_DEBUG=1  # 通信超时分析
    export HCCL_ASYNC_ERROR_HANDLING=1  # 异步错误处理
    export OOM_SNAPSHOT_ENABLE=1  # OOM 时保存内存快照
    export OOM_SNAPSHOT_PATH=./oom_snapshot
    export ACL_OP_COMPILER_CACHE_DIR=./op_cache  # 算子编译缓存目录
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

# NPU: 归档算子编译缓存和 OOM 快照
if [ "${DEVICE_TYPE:-cuda}" = "npu" ]; then
    cp -r ./op_cache "${RUNDIR}/artifacts/" 2>/dev/null || true
    cp -r ./oom_snapshot "${RUNDIR}/artifacts/" 2>/dev/null || true
    cp -r ./torch_compile_debug "${RUNDIR}/artifacts/" 2>/dev/null || true
fi

echo "Debug run complete. Artifacts collected under ${RUNDIR}/artifacts"
```

**说明**：

* 对于真实多卡（每个进程有独立 stdout），你应该在 launcher 层让每个 rank 的 stdout 分别写入 `compile_debug_rank${RANK}.log`。如何在你的集群环境中实现按 rank 日志分离，取决于 launcher（例如一些集群默认会将每个 rank 的 stdout 写到不同文件）。

---

## 3) Export + Backend 捕获脚本

**目的**：用 API 方式保存可复用的 FX `GraphModule`（用于离线对比），并在 `torch.compile` 时把 backend 接收到的最终 `GraphModule` 写盘。两部分代码分别为 `export_capture.py` 和 `capture_backend.py`。

### `export_capture.py`（最小可运行）

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

### `capture_backend.py`（供 `torch.compile` 使用的 backend 截获器）

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

---

## 4) 灾难恢复 / 归档脚本 — `collect_artifacts.sh`

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

---

## 5) 日志对比脚本 — `diff_rank_logs.sh`

快速对比每个 rank 的关键日志字段（graph_breaks / guards / recompiles / aot / inductor）。

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

---

## 6) 决策树（精炼版）

> 使用本决策树按顺序排查，逐步把问题归因到 Dynamo / AOT / Inductor / 分布式。

1. **先看 GRAPH BREAK**（grep 日志）

   * 如果很多：Dynamo 问题（Python 控制流 / 不支持 op / numpy）→ 优先改写（tensor ops / export 来定位）
2. **看 recompiles / GUARD FAIL**

   * 如果频繁重编译：Guard 问题（动态 shape / stride / device）→ 固定输入 shape 或改造为动态 shape 支持
3. **若 forward 正常但 backward 异常或 large backward graph**

   * AOT 问题（in-place / mutation / custom autograd）→ 避免 in-place，使用 `aot_eager` 测试
4. **若数值差异/性能异常且以上都 OK**

   * Inductor 问题（fusion / kernel / codegen）→ 看 `/tmp/torchinductor_*` 下的 `output_code.py` 与 triton kernel
5. **若问题仅在多卡/分布式出现**

   * 分布式问题（数据 split / padding / seed / DDP wrap order / NCCL）→ 单卡/两卡复现后对比 logs

---

## 7) 新增：每个阶段的捕获要点与日志分析指南（Dynamo / FX / AOT / Inductor / kernel）

下面把上面“各阶段产物与日志”的说明补充成一份**可执行的、按阶段排查清单**。目标是：**告诉你该捕获什么、从哪些日志行判断问题、如何快速定位与修复**。把此节放在原文的第 6 节之后，方便直接参考。

> 说明（通用）
>
> * 以下术语：**rank** 指训练进程（单卡场景 rank=0）。
> * 所有命令假设在 `RUNDIR` 下有 `compile_debug_rank${RANK}.log`。
> * 不同版本 log/文件名会有细微差别；我尽量使用通用的 grep 模式并注明可变点。
> * 如果某条建议与当前版本不一致，请把具体日志片段贴出来，我会直接指出差异并调整建议。

---

### 结构总览（快速参考表）

| 阶段                        |                                                                要捕获的内容 | 日志/信号                                      | 首要定位命令                                             | 常见原因            | 直接尝试的快速修复                         |                                                  |                                                     |
| ------------------------- | --------------------------------------------------------------------: | ------------------------------------------ | -------------------------------------------------- | --------------- | --------------------------------- | ------------------------------------------------ | --------------------------------------------------- |
| Dynamo (frontend)         |      export 的 `GraphModule`（`gm.code`）、`export()` 输出、python traceback | `GRAPH BREAK`、`falling back`、`unsupported` | `grep -nE "GRAPH BREAK                             | falling back    | unsupported" *.log`               | Python control flow、numpy/py ops、动态 control flow | 用 `export()` 捕获，改写为 tensor ops / 使用 `torch.where` 等 |
| FX (graph)                |                          `gm.graph`、`graph_repr.txt`、`gm_snapshot.pt` | `graph generated`、`graph_break` 上下文        | 打开 `graph_repr.txt`，`git diff --no-index`          | 无效/冗余分支、未展开的子模块 | 对比 eager 与 exported graph，合并/重写分支 |                                                  |                                                     |
| AOT (autograd + decomps)  |                                           `aot_eager` traces、aot 相关日志 | `aot`、`aot_eager`、`mutation detected`      | `grep -nE "aot                                     | aot_eager       | mutation detected" *.log`         | in-place / mutation / custom autograd            | 关闭 in-place、用 `aot_eager` 调试                        |
| Inductor (codegen/fusion) | `/tmp/torchinductor_*` 下 `output_code.py`、kernel dumps、triton kernels | `inductor`、`falling back to aten`、`emit`   | `ls -R /tmp/torchinductor_*` + `grep -nE "inductor | triton          | falling back" *.log`              | fusion 被切碎、不支持 op、triton bug                     | 查看 `output_code.py`，用 capture backend 导出局部 gm       |
| Kernel (triton/CUDA)      |                            生成的 kernel 源、编译错误、PTX/ASM、runtime error 堆栈 | `triton`、`cuda error`、`segfault`           | `grep -nE "triton                                  | cuda error      | segfault" *.log`                  | 编译参数、内核资源不足（shared/mem/threads）                  | 降低并行度，缩小 problem size，使用 CPU/Eager 验证数值             |

> 注：上表为快速参考，下面给出更细化的可执行步骤与命令示例。

---

### A) Dynamo（Python 前端 / 控制流）

**要捕获的产物**

* `export_capture.py` 生成的 `gm_code.py`、`gm_graph.txt`、`guards.json`。
* compile 日志中 `GRAPH BREAK` 与 `guard` 上下文片段（每个 rank）。

**如何识别问题**

* 常见 log 关键词：`GRAPH BREAK`、`falling back to eager`、`unsupported`、`unsupported opcode`、`PythonOp`。
* 命令（按 rank）：

  ```bash
  grep -nE "GRAPH BREAK|falling back|unsupported|PythonOp" compile_debug_rank*.log
  ```
* 统计 GRAPH BREAK 次数（帮助判断问题严重度）：

  ```bash
  grep -c "GRAPH BREAK" compile_debug_rank0.log
  ```

**解释与快速定位**

* `GRAPH BREAK` 通常意味着某段 Python 控制流或 PyObject 操作没法被 Dynamo 捕获（例如 `if` 里有 numpy 操作、list 变长、使用了大量 Python 索引等）。
* 首先用 `export()`（见第 3 节的 `export_capture.py`）来定位确切的节点/源码行。`guards.json` 会告诉哪些 guard 导致脱编译。

**快速修复建议**

1. 用纯 tensor 操作替换 Python control flow，或把可变控制流改成 `torch.where` / `torch.where`-style 向量化。
2. 如果是少量不支持的 op，考虑用 `torch.fx` 或自定义 decompositions 替代。
3. 若不可改写，用 `export()` 导出 gm，在线下复现并手工修补（例如把对应子函数替换为 aten 原语序列）。

---

### B) FX（Graph 内部表示）

**要捕获的产物**

* `gm.code`（源码形式）与 `gm.graph`（repr 文本），`gm_snapshot.pt`（可选 state_dict）。

**如何识别问题**

* 关注 `gm.graph` 中的节点：`call_function` / `call_method` / `get_attr` 的异常模式；查找 `PythonOp` 或 `proxy` 标记。
* 命令：

  ```bash
  sed -n '1,200p' debug_dumps/export/gm_<ts>/graph_repr.txt
  ```
* 对比 eager trace 与 exported graph 的差异：

  ```bash
  git diff --no-index eager_trace.txt debug_dumps/export/gm_<ts>/graph_code.py
  ```

**解释与快速定位**

* FX graph 能展示被捕获的计算图，缺少某些算子或多出大量 `call_function`/`PythonOp`，意味着 Dynamo 前端没能完全替代 Python 控制流或解析到原子算子。
* 查看 `guards.json`，判断 guard 是 shape/stride/device 还是值相关（value dependent）。

**快速修复建议**

1. 若 graph 中有大量 `aten` 被分割成小节点，检查是否存在 `tensor.shape` / `tensor.size()` 在控制流中导致分裂。
2. 使用 `torch.fx` 手工重写关键子图或编写 decomposition，或把复杂 Python 逻辑抽成可编译的子模块。

---

### C) Guards / Recompiles（动态性问题）

**要捕获的产物**

* compile 日志中 `GUARD` / `GUARD FAIL` / `recompil` 行，`guards.json`。

**如何识别问题**

* 关键词：`GUARD`、`GUARD FAIL`、`recompile`、`recompil`。

  ```bash
  grep -nE "GUARD|GUARD FAIL|recompil|recompile" compile_debug_rank*.log
  ```
* 统计每种 guard fail 次数（帮助判断是否为多卡输入不同导致）：

  ```bash
  grep -n "GUARD FAIL" compile_debug_rank*.log | sed -E 's/.*GUARD FAIL: (.*)/\1/' | sort | uniq -c | sort -nr
  ```

**解释与快速定位**

* Guard 失败常见原因：输入 shape、device、stride 或 tensor.storage 发生变化；不同 rank 间输入 batch/seq 不一致也会导致 guard 失效并重编译。
* 若重编译仅在某些 rank 出现，说明数据划分不一致或 preprocessing 有差异。

**快速修复建议**

1. 固定 batch size / seq length 或使用 `torch._dynamo.config.suppress_errors=True`（仅临时调试）。
2. 检查并统一所有 rank 的数据预处理 pipeline（shuffle / padding / bucketing 等）。
3. 考虑使用 `torch.compile(..., dynamic=True)` 或显式声明动态轴（视版本支持情况）。

---

### D) AOT（Autograd / aot_dispatch / aot_eager）

**要捕获的产物**

* aot 相关的 trace、`aot_eager` 运行输出，如果出错保存完整 traceback。

**如何识别问题**

* 关键词：`aot`、`aot_eager`、`mutation detected`、`backward graph`。

  ```bash
  grep -nE "aot|aot_eager|mutation detected|backward" compile_debug_rank*.log
  ```
* 若 backward 异常：从训练日志抓取完整 traceback（早期的堆栈信息最有价值）。

**解释与快速定位**

* AOT 错误多与 in-place 操作、mutation 或自定义 autograd 实现冲突有关。大模型的 backward graph 很大时，AOT 的分割/clone 也可能出问题。
* 用 `aot_eager` 运行可以快速判断是否是 AOT 本身的问题（`aot_eager` 会使用 eager autograd 进行模拟）。

**快速修复建议**

1. 禁止/替换 in-place ops（`.add_()` 等），保证 forward/backward 的纯函数语义。
2. 若使用 custom autograd，确保 `ctx.save_for_backward` / `ctx.mark_dirty` 等用法正确。
3. 使用 `aot_eager` 做最小重现并逐步缩减输入规模复现。

---

### E) Inductor（Fusion / codegen / triton 入口层）

**要捕获的产物**

* `/tmp/torchinductor_*` 目录下所有文件（`output_code.py`、`fused_kernel*.py`/`.triton`/`.cu`、`kernel_*`、`compile_log` 等）。
* compile 日志中 `inductor` 相关行（`TORCHINDUCTOR_VERBOSE` 有助于更多输出）。

**如何识别问题**

* 关键词：`inductor`、`triton`、`falling back to aten`、`emit`、`fusion`。

  ```bash
  grep -nE "inductor|triton|falling back|fall back|fusion|emit" compile_debug_rank*.log
  ls -R /tmp/torchinductor_* | sed -n '1,200p'
  ```
* 阅读 `output_code.py`，查看对应 subgraph 的 Python->fusion->kernel 的映射；在 `output_code.py` 中查找 `# kernel` 注释或相关 triton 调用。

**解释与快速定位**

* Inductor 会将一组算子 fuse 为 kernel。若 kernel 太大或包含不支持的 op，系统可能：

  * 切碎 fusion（产生很多小 kernel，性能退化）；
  * 或 `fall back to aten`（回退到 CPU/aten 实现）；
  * 或生成 triton kernel 导致编译/运行时错误。
* `output_code.py` 是最直接的查看入口，能看到 Python-level 为 kernel 生成的代码和 metadata（例如 grid/block 大小、使用的 buffers）。

**快速修复建议**

1. 若发现 `fall back to aten`：检查该子图内的 op 是否在 Inductor 支持列表之外，必要时替换或拆分子图（给 compiler 更友好的形态）。
2. 若 kernel 出现性能问题（太多小 kernel）：尝试改变图形结构让更多算子能被 fuse（例如合并连续的 elementwise，避免间插大量 view/reshape/permute）。
3. 使用 `capture_backend` 导出目标 gm，并在单卡上对该 gm 做 further-debug（方便离线重现）。

---

### F) Kernel / Triton / CUDA 层（最终执行）

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

---

### 常用命令与快速脚本片段（直接可用）

**1) 统计每个 rank 的 GRAPH BREAK / GUARD FAIL / INDUCTOR 行数**

```bash
for f in compile_debug_rank*.log; do
  echo "=== $f ==="
  echo "GRAPH BREAK: $(grep -c "GRAPH BREAK" "$f")"
  echo "GUARD FAIL:  $(grep -c "GUARD FAIL" "$f")"
  echo "RECOMPILE:   $(grep -c "recompil" "$f")"
  echo "INDUCTOR:    $(grep -c "inductor" "$f")"
  echo
done
```

**2) 从 guards json 提取最常见的 guard（示例）**

```bash
jq -r '.[]' debug_dumps/export/gm_<ts>/guards.json | sort | uniq -c | sort -nr | head -n 50
```

（如果没有 `jq`，可用 `sed`/`grep` 简化）

**3) 把某个 gm 的 code 提取成最小 repro 模块（人工步骤）**

* 从 `gm_code.py` 把 `GraphModule` 的 `forward` 与需要的 `__init__`（attributes）复制到新的 `repro_model.py`，写一个小的 `run_repro.py` 给定同样的 `example_inputs` 去 `torch.compile(repro_model, backend=...)` 运行。

---

## 8) 常见问题与快速修复清单

* **大量 GRAPH BREAK** → 改写 Python 控制流为 tensor ops，或用 `export()` 定位并修复具体行。
* **频繁 RECOMPILES** → 确认 batch/seq 长度一致，避免不同 rank 输入 shape。
* **AOT: mutation detected / backward graph 过大** → 移除 in-place ops，确保自定义 autograd 正确。
* **Inductor: 多小 kernel / 性能退化 / 数值差异** → 检查 fusion 被切碎的原因，使用 `TORCHINDUCTOR_DUMP` 导出并分析 kernel。
* **分布式 only：输出不一致** → 检查数据划分、随机种子、模型 wrapping 顺序、NCCL 报警。

---

## 9) 使用流程（工程化）

1. **单卡快速复现**：先在单 GPU 上运行 `run_debug.sh` 修改 `NUM_GPUS=1`，排除分布式问题。
2. **导出 FX（可复用）**：在单卡上运行 `export_capture.py` 保存 `gm`，用于离线比对。
3. **后台捕获 backend GM**：在模型上临时使用 `capture_backend_factory()` 得到 backend 层的 GM。
4. **运行多卡（2卡→全卡）**：按 `run_debug.sh` 启动，收集每个 rank 的日志与 `/tmp/torchinductor_*`。
5. **归档并对比**：用 `collect_artifacts.sh` 和 `diff_rank_logs.sh` 做初步对比。
6. **若是 Inductor 问题**：提取单 kernel reproducer，向 upstream 提交 issue 或在本地调试 kernel/triton 代码。

---

## 10) 附：小技巧与注意事项

* Notebook: 在第一个 cell 设置 `os.environ[...]` 并重启 kernel。
* 若你需要把 `/tmp/torchinductor_*` 固定到其他路径：部分版本提供 `TORCHINDUCTOR_CACHE_DIR` 或类似变量，若不支持请手动归档。
* 对比 `gm.text`/`code` 建议用 `git diff --no-index` 或 `diff -u` 做文本 diff。
* 对于大模型，导出和保存要控制样本大小（抽样）以免生成过大文件。
* 报 issue 时务必附上最小可复现脚本、完整 log、`/tmp/torchinductor_*` 相关文件、以及 export 的 gm（参见下文“提交问题给 upstream / 报 issue 时要附的最小清单”）。

---

## 附：提交问题给 upstream / 报 issue 时要附的最小清单

当你把问题上报给 PyTorch / Inductor / Triton 团队时，尽量打包并提供以下内容（便于快速定位）：

1. **最小可复现脚本**（`run_repro.py`）与 `requirements.txt`（包含 PyTorch + CUDA + triton 版本）
2. `compile_debug_rank0.log`（完整日志）
3. `/tmp/torchinductor_*` 下相关 kernel 源与 `output_code.py`（压缩包）
4. `export` 导出的 `gm_code.py` / `gm_graph.txt` / `guards.json` / `gm_snapshot.pt`
5. 复现步骤（命令），说明是否单卡/多卡复现、GPU 型号、driver 以及 NCCL/OSOS 信息
6. 若存在数值不一致，给出对比数据（eager vs compiled）的小样本

---

## 11) NPU 特有调试手段（华为昇腾）

> 本节专门针对使用 `torch_npu`（华为昇腾 NPU）时的调试方法，包括 NPU 特有的环境变量、API 和工具。

### 11.1 NPU 环境变量配置

#### 基础日志环境变量

```bash
# 全局日志级别（0:debug, 1:info, 2:warning, 3:error, 4:null）
export ASCEND_GLOBAL_LOG_LEVEL=0

# 禁用 NPU 告警信息（可选）
export TORCH_NPU_DISABLED_WARNING=1

# 精简错误输出（将 CANN 内部调用栈转移到 plog）
export TORCH_NPU_COMPACT_ERROR_OUTPUT=1
```

#### 算子编译与执行调试

```bash
# 算子编译磁盘缓存目录
export ACL_OP_COMPILER_CACHE_DIR=./op_cache

# 算子编译缓存模式（0:不使用缓存, 1:仅使用缓存, 2:使用缓存并生成, 3:不使用缓存但生成）
export ACL_OP_COMPILER_CACHE_MODE=2

# 同步模式（算子执行时是否启动同步模式，用于调试）
export ASCEND_LAUNCH_BLOCKING=1

# 任务队列优化（0:关闭, 1:level 1, 2:level 2）
export TASK_QUEUE_ENABLE=2

# 每个流一个 task_queue
export PER_STREAM_QUEUE=1
```

#### 内存管理调试

```bash
# OOM 时保存内存快照（0:关闭, 1:保存当前和历史, 2:仅保存当前）
export OOM_SNAPSHOT_ENABLE=1
export OOM_SNAPSHOT_PATH=./oom_snapshot

# 关闭内存复用机制（用于排查内存问题）
export PYTORCH_NO_NPU_MEMORY_CACHING=1

# 内存分配器配置
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:128"

# 多流内存复用
export MULTI_STREAM_MEMORY_REUSE=1
```

#### 集合通信（HCCL）调试

```bash
# 异步错误处理（0:不开启, 1:开启）
export HCCL_ASYNC_ERROR_HANDLING=1

# 通信超时分析（0:不开启, 1:开启）
export HCCL_DESYNC_DEBUG=1

# 通信超时时间（毫秒）
export HCCL_EVENT_TIMEOUT=1800000

# 点对点通信独立通信域
export P2P_HCCL_BUFFSIZE=1

# 零拷贝功能（0:关闭, 1:开启）
export TORCH_HCCL_ZERO_COPY=1
```

#### 特征值检测（ASD）

```bash
# 特征值检测（0:关闭, 1:开启仅打印, 2:开启并告警, 3:开启告警并记录过程数据）
export NPU_ASD_ENABLE=2

# 特征值检测绝对阈值
export NPU_ASD_UPPER_THRESH=100000

# 特征值检测相对阈值
export NPU_ASD_SIGMA_THRESH=4.0
```

#### Profiling 与性能分析

```bash
# Profiling 配置文件路径
export PROF_CONFIG_PATH=./profiler_config.json

# 使用 msMonitor nputrace 方式
export KINETO_USE_DAEMON=1

# GE Profiling 输出到标准输出
export GE_PROFILING_TO_STD_OUT=1
```

#### 设备管理

```bash
# Stream pool 最大流数
export STREAMS_PER_DEVICE=32

# NPU 设备能力返回值
export TORCH_NPU_DEVICE_CAPABILITY=9.0
```

### 11.2 NPU API 调试方法

#### Dump 功能（算子数据 dump）

```python
import torch
import torch_npu

# 初始化 dump
torch_npu.npu_config.init_dump()

# 设置 dump 配置文件
torch_npu.npu_config.set_dump("dump_config.json")

# 运行模型
# ...

# 结束 dump
torch_npu.npu_config.finalize_dump()
```

**dump_config.json 示例**：

```json
{
    "dump_enable": 1,
    "dump_path": "./dump_data",
    "dump_step": "all",
    "dump_mode": "input",
    "dump_op_list": ["MatMul", "Add"],
    "dump_op_switch": {
        "MatMul": {
            "dump_enable": 1,
            "dump_mode": "input"
        }
    }
}
```

#### Profiling 功能

```python
import torch
import torch_npu

# 使用 Profile 类
with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    output_dir="./npu_profiling",
) as prof:
    # 运行模型
    output = model(input)

# 打印结果
print(prof.key_averages())
```

#### Dynamic Profile（动态 profiling）

```python
from torch_npu.profiler import dynamic_profile as dp

# 初始化
dp.init("profiler_config_path")

# 训练循环
for step in range(num_steps):
    train_one_step()
    dp.step()
```

#### 同步调试模式

```python
import torch_npu

# 设置同步调试模式（0:default, 1:warn, 2:error）
torch_npu.npu.set_sync_debug_mode(2)

# 获取当前模式
mode = torch_npu.npu.get_sync_debug_mode()
```

#### 算子编译缓存清理

```python
import torch_npu

# 清理算子编译缓存
torch_npu.npu.clear_op_cache()
```

### 11.3 NPU 特有调试脚本

#### `run_npu_debug.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# 配置区
NUM_NPUS=${NUM_NPUS:-8}
ENTRY_SCRIPT=${ENTRY_SCRIPT:-train.py}
RUNDIR=${RUNDIR:-./npu_debug_run}

mkdir -p "$RUNDIR"

# NPU 环境变量
export ASCEND_GLOBAL_LOG_LEVEL=0
export HCCL_ASYNC_ERROR_HANDLING=1
export HCCL_DESYNC_DEBUG=1
export OOM_SNAPSHOT_ENABLE=1
export OOM_SNAPSHOT_PATH="${RUNDIR}/oom_snapshot"
export ACL_OP_COMPILER_CACHE_DIR="${RUNDIR}/op_cache"
export ACL_OP_COMPILER_CACHE_MODE=2
export TASK_QUEUE_ENABLE=2
export TORCH_NPU_COMPACT_ERROR_OUTPUT=1

# PyTorch 基础环境变量
export TORCH_LOGS="+dynamo,guards,graph_breaks,recompiles,aot,inductor"
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR="${RUNDIR}/torch_compile_debug"
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 启动
RANK=0
torchrun --nproc_per_node=${NUM_NPUS} ${ENTRY_SCRIPT} 2>&1 | tee "${RUNDIR}/compile_debug_rank${RANK}.log"

# 归档
mkdir -p "${RUNDIR}/artifacts"
cp -r "${RUNDIR}/op_cache" "${RUNDIR}/artifacts/" 2>/dev/null || true
cp -r "${RUNDIR}/oom_snapshot" "${RUNDIR}/artifacts/" 2>/dev/null || true
cp -r "${RUNDIR}/torch_compile_debug" "${RUNDIR}/artifacts/" 2>/dev/null || true

echo "NPU debug run complete. Artifacts collected under ${RUNDIR}/artifacts"
```

#### `collect_npu_artifacts.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

OUTDIR=${1:-./npu_artifacts_$(date +%s)}
mkdir -p "$OUTDIR"

# 拷贝算子编译缓存
cp -r ./op_cache "$OUTDIR/" 2>/dev/null || true

# 拷贝 OOM 快照
cp -r ./oom_snapshot "$OUTDIR/" 2>/dev/null || true

# 拷贝 torch_compile_debug
cp -r ./torch_compile_debug "$OUTDIR/" 2>/dev/null || true

# 拷贝 dump 数据
cp -r ./dump_data "$OUTDIR/" 2>/dev/null || true

# 拷贝 profiling 数据
cp -r ./npu_profiling "$OUTDIR/" 2>/dev/null || true

# 拷贝日志
cp -r compile_debug_rank*.log "$OUTDIR/" 2>/dev/null || true

# 打包
tar -czvf "${OUTDIR}.tgz" -C "$(dirname "$OUTDIR")" "$(basename "$OUTDIR")"

echo "NPU artifacts archived to ${OUTDIR}.tgz"
```

### 11.4 NPU 常见问题与快速修复

| 问题类型 | 可能原因 | 快速修复方法 |
|---------|---------|------------|
| **算子编译失败** | 算子不支持 / 参数不匹配 | 检查 `ACL_OP_COMPILER_CACHE_DIR`，使用 `ASCEND_LAUNCH_BLOCKING=1` 同步调试 |
| **OOM 错误** | 显存不足 / 内存泄漏 | 设置 `OOM_SNAPSHOT_ENABLE=1` 分析快照，使用 `PYTORCH_NPU_ALLOC_CONF` 调整分配策略 |
| **通信超时** | HCCL 配置问题 / 网络问题 | 设置 `HCCL_DESYNC_DEBUG=1` 和 `HCCL_ASYNC_ERROR_HANDLING=1` 分析 |
| **数值精度问题** | 精度模式 / 算子实现差异 | 检查 `INF_NAN_MODE_ENABLE`，使用 `NPU_ASD_ENABLE` 检测异常值 |
| **性能问题** | 算子未融合 / 内存拷贝多 | 使用 Profiling 分析，开启 `TASK_QUEUE_ENABLE` 和 `TORCH_HCCL_ZERO_COPY` |
| **图捕获失败** | 不支持的算子 / 控制流问题 | 使用 `torch_npu.npu_config.init_dump()` 分析算子调用，参考 Dynamo 调试方法 |

### 11.5 NPU 与 CUDA 调试差异对比

| 调试项 | CUDA | NPU |
|-------|------|-----|
| **编译产物目录** | `/tmp/torchinductor_*` | `./torch_compile_debug` + `./op_cache` |
| **通信后端** | NCCL | HCCL |
| **通信调试变量** | `TORCH_NCCL_*` | `HCCL_*` |
| **内存快照** | CUDA memory snapshot | `OOM_SNAPSHOT_*` |
| **算子 dump** | N/A | `torch_npu.npu_config.init_dump()` |
| **Profiling** | Nsight Systems / PyTorch Profiler | torch_npu.profiler / dynamic_profile |
| **特征值检测** | N/A | `NPU_ASD_ENABLE` |
| **同步调试** | `CUDA_LAUNCH_BLOCKING` | `ASCEND_LAUNCH_BLOCKING` |

### 11.6 NPU 调试最佳实践

1. **分层调试**：先使用 `ASCEND_GLOBAL_LOG_LEVEL=0` 获取详细日志，再根据问题类型设置特定环境变量
2. **启用同步模式**：在调试算子问题时，设置 `ASCEND_LAUNCH_BLOCKING=1` 便于定位错误
3. **保存调试产物**：使用 `collect_npu_artifacts.sh` 统一收集所有调试信息
4. **使用 Profiling**：性能问题优先使用 `torch_npu.profiler` 分析瓶颈
5. **内存分析**：OOM 问题启用 `OOM_SNAPSHOT_ENABLE=1` 保存快照进行分析
6. **通信调试**：分布式问题启用 `HCCL_DESYNC_DEBUG=1` 和 `HCCL_ASYNC_ERROR_HANDLING=1`
7. **算子 dump**：特定算子问题使用 `torch_npu.npu_config.init_dump()` 精准定位

---

## Related Pages

- [[torch_compile/overview]]
- [[PyTorch_Dynamo_Technical_Analysis]]
- [[PyTorch_Inductor_Technical_Analysis]]
