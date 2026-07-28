# Native Backend 外部验收 Runbook

> 更新日期：2026-07-27  
> 合同版本：`native-backend-evidence/v1`

这份 Runbook 解决一个容易造成错误结论的问题：**“能生成 IR/源代码”不等于“已完成原生编译、加载、执行、数值验证和性能测量”**。本机能力探测与外部验收证据是两种不同的 artifact；前者即使成功执行，也不能升级为后者。

## 1. 信任边界与状态语义

| artifact | 能回答的问题 | 允许的状态 | 能否作为课程原生后端 PASS |
|---|---|---|---|
| `local_capability_diagnostic` | 当前进程是否看得到编译器、CUDA 和 Triton | `BLOCKED`、`NOT_RUN`、`FAIL` | 否 |
| `native_acceptance_result` | 原生 kernel 是否真的编译、加载、执行、对齐数值并完成测量 | 只有目标完整验收后才可写 `PASS` | 是 |

状态含义：

- `BLOCKED`：缺少执行所需的编译器、设备或运行时。
- `NOT_RUN`：能力存在，但本次只是探测，没有运行验收 workload。
- `FAIL`：尝试了验收动作，但编译、加载、执行、数值比较或测量失败。
- `PASS`：目标的所有必填证据均存在，通过 JSON Schema、跨字段语义检查和 bundle 文件哈希检查。
- `SKIP` 不是合法状态。不得把 `BLOCKED`、`NOT_RUN`、`FAIL` 或 `SKIP` 汇总为 PASS。

`probe` 返回码为 0 只表示“诊断文件成功生成”；它不表示 CPU 或 CUDA 验收通过。`produce` 只有真实执行并自校验通过才返回 0；能力不足时写出 diagnostic 并返回 3，执行失败时写出 `FAIL` diagnostic 并返回 4。`validate` 只有在目标为真实 PASS 且 bundle 校验完整时才返回 0，任何验收拒绝统一返回 2。

## 2. 当前 Windows 本机诊断

从知识库根目录执行：

```powershell
cd E:\97-codes\torch_parallel\llm-knowledge
$Lab = "wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs"
$Diagnostic = "$Lab\artifacts\native_backend\local_capability_diagnostic.json"

python -B "$Lab\native_backend_contract.py" probe --output "$Diagnostic"
$LASTEXITCODE
```

2026-07-27 的实际结果保存在 `artifacts/native_backend/local_capability_diagnostic.json`：

| 能力 | 实测结果 | 验收状态 |
|---|---|---|
| Windows 原生 CPU C++ | `cl.exe` 不在 `PATH` | `BLOCKED` |
| CUDA | `torch.cuda.is_available() == False`，`torch.version.cuda == None` | `BLOCKED` |
| Triton | Python package 不存在 | `BLOCKED` |
| Runtime | Python `3.13.5`，PyTorch `2.9.1+cpu`，git `5811a8d…` | 仅环境事实 |

下面的命令必须失败，并返回 2；它证明诊断文件不能冒充 CPU 验收结果：

```powershell
python -B "$Lab\native_backend_contract.py" validate `
  --input "$Diagnostic" `
  --target cpu `
  --artifact-root "$Lab\artifacts\native_backend"
$LASTEXITCODE
```

CUDA 目标同理：

```powershell
python -B "$Lab\native_backend_contract.py" validate `
  --input "$Diagnostic" `
  --target cuda `
  --artifact-root "$Lab\artifacts\native_backend"
$LASTEXITCODE
```

## 3. 验收 bundle 的固定布局

证据 JSON 中的 artifact 路径必须是相对于 bundle 根目录的安全相对路径，并统一使用 `/` 分隔；绝对路径、反斜杠和包含 `..` 的路径都会被拒绝。建议布局如下：

```text
native_backend_run/
├── result.json
├── cpu/
│   ├── generated.cpp
│   ├── compile.stdout.txt
│   ├── compile.stderr.txt
│   └── kernel.dll-or-so
└── cuda/
    ├── generated.py
    ├── triton_cache.zip
    ├── memory_snapshot.pickle
    └── memory_trace.json
```

`result.json` 中每个文件都带小写 64 位 SHA-256。validator 会重新读取 bundle 文件并计算哈希；文件缺失、路径越界或哈希不一致都会阻止 PASS。

## 4. CPU 原生验收

### 4.1 环境下限

Windows 使用 **x64 Native Tools PowerShell for Visual Studio 2022**，并在同一个 shell 中完成编译和验证。Linux 使用当前 shell 可发现的 `c++`、`g++` 或 `clang++`。Python 与 PyTorch 必须是实际执行 workload 的同一环境，不能从另一台机器抄写版本。

Windows 初始化与预检命令。`$Bundle` 是本次独立证据目录，不要指向知识库根目录：

```powershell
$Repo = "E:\97-codes\torch_parallel\llm-knowledge"
Set-Location -LiteralPath $Repo
$Lab = Join-Path $Repo "wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs"
$Bundle = Join-Path $env:TEMP "native_backend_cpu_run"
New-Item -ItemType Directory -Force -Path $Bundle | Out-Null

where.exe cl
cl.exe 2>&1 | Select-Object -First 3
python --version
python -c "import importlib.metadata; print(importlib.metadata.version('jsonschema'))"
python -c "import platform, torch; print(platform.platform()); print(torch.__version__); print(torch.version.git_version)"
python -B "$Lab\native_backend_contract.py" probe `
  --output "$Bundle\cpu_capability_diagnostic.json"
```

Linux 初始化与预检命令；如果仓库不在示例位置，只修改 `REPO`：

```bash
set -u
REPO=/path/to/llm-knowledge
LAB="$REPO/wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/labs"
BUNDLE="${TMPDIR:-/tmp}/native_backend_cpu_run"
mkdir -p "$BUNDLE"
cd "$REPO"

command -v c++ || command -v g++ || command -v clang++
c++ --version | head -n 3
python --version
python -c 'import importlib.metadata; print(importlib.metadata.version("jsonschema"))'
python -c 'import platform, torch; print(platform.platform()); print(torch.__version__); print(torch.version.git_version)'
python -B "$LAB/native_backend_contract.py" probe \
  --output "$BUNDLE/cpu_capability_diagnostic.json"
```

能力探测在编译器存在时仍写 `NOT_RUN`，因为探测没有执行 native workload。

### 4.2 Producer 必须真实执行的动作

CPU `PASS` producer 必须对 pointwise 和 reduction 两类 workload 分别完成：

1. 由合同 producer 生成同时包含 pointwise 与 reduction entry point 的真实 C++ 源文件并保存原文。该 producer 验证系统 native toolchain，不得把它误写成 Inductor codegen 证据；若要证明 Inductor，还需要单独的 Inductor 来源链。
2. 调用记录在 `compile.command` 中的真实编译命令，保存动态库或 Python extension。
3. 在当前进程中加载已编译 artifact。
4. 调用 native entry point，而不是重新调用 eager 函数。
5. 与 `torch.eager` reference 做数值比较，记录 `rtol`、`atol`、最大绝对误差和最大相对误差。
6. 至少一次 warmup 和一次独立计时迭代；`timings_ms.samples` 数量必须等于 `iterations`。
7. 记录 source、编译 stdout/stderr、二进制 artifact 的独立路径和 SHA-256，以及 producer hash、compiler path/version、Python/PyTorch/OS/architecture/backend。

仓库现有 `part4_artifact_bundle.py` 的 codegen-only 分支会绕过编译器检查并明确标记 `generated_cpp_source_compiled=False`。它可用于 IR 教学，**不得**用它填充本合同的 CPU PASS。

### 4.3 CPU 验收命令

先运行 producer。返回 3 表示环境仍然 `BLOCKED`，此时 `$Bundle\result.json` 是 diagnostic，不能继续宣称 PASS：

```powershell
$Repo = "E:\97-codes\torch_parallel\llm-knowledge"
$Lab = Join-Path $Repo "wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs"
$Bundle = Join-Path $env:TEMP "native_backend_cpu_run"

python -B "$Lab\native_backend_contract.py" produce `
  --target cpu `
  --output "$Bundle\result.json" `
  --artifact-root "$Bundle"
$ProducerExit = $LASTEXITCODE
if ($ProducerExit -eq 3) { throw "CPU native producer BLOCKED; inspect result.json" }
if ($ProducerExit -ne 0) { throw "CPU native producer failed: exit $ProducerExit" }

python -B "$Lab\native_backend_contract.py" validate `
  --input "$Bundle\result.json" `
  --target cpu `
  --artifact-root "$Bundle"
if ($LASTEXITCODE -ne 0) { throw "CPU native acceptance failed" }
```

Linux：

```bash
REPO=/path/to/llm-knowledge
LAB="$REPO/wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/labs"
BUNDLE="${TMPDIR:-/tmp}/native_backend_cpu_run"

python -B "$LAB/native_backend_contract.py" produce \
  --target cpu \
  --output "$BUNDLE/result.json" \
  --artifact-root "$BUNDLE"
producer_exit=$?
if [ "$producer_exit" -ne 0 ]; then
  echo "CPU native producer did not pass; inspect $BUNDLE/result.json" >&2
  exit "$producer_exit"
fi

python -B "$LAB/native_backend_contract.py" validate \
  --input "$BUNDLE/result.json" \
  --target cpu \
  --artifact-root "$BUNDLE"
test $? -eq 0
```

## 5. CUDA/Triton 原生验收

### 5.1 环境下限与预检

需要 NVIDIA CUDA device、与设备兼容的 driver、CUDA-enabled PyTorch 和可导入的 Triton。版本不靠期望值推断，必须从实际执行环境记录。

```bash
set -u
REPO=/path/to/llm-knowledge
LAB="$REPO/wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/labs"
BUNDLE="${TMPDIR:-/tmp}/native_backend_cuda_run"
mkdir -p "$BUNDLE"
cd "$REPO"

nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv
python --version
python - <<'PY'
import torch
import triton
print("torch", torch.__version__, torch.version.git_version)
print("cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("triton", triton.__version__)
PY
python -B "$LAB/native_backend_contract.py" probe \
  --output "$BUNDLE/cuda_capability_diagnostic.json"
```

只有 CUDA 与 Triton 同时可用时，探测状态才会是 `NOT_RUN`；它仍不是 PASS。

### 5.2 Producer 必须真实执行的动作

CUDA/Triton `PASS` producer 必须：

这里的 producer 直接执行合同生成的 Triton kernel 和两组唯一配置的实测候选，不得把结果扩大解释为 Inductor 自动调优证据；Inductor 来源链仍需由对应课程 Lab 单独证明。

1. 保存实际生成的 Triton source、PTX 或等价可审计 source，并计算 SHA-256。
2. 保存至少两个实际 benchmark candidate 的配置。每个 candidate 都要有真实 `samples`，样本数等于 `iterations`。
3. 按 `median_ms` 选择实测中位数最低的 candidate；`winner.timing_ms` 必须等于该 candidate 的中位数。
4. 保存实际 autotune/cache artifact 和 SHA-256，不能只记录 cache 路径字符串。
5. 调用 winner kernel，与 eager reference 做数值比较。
6. 在计时前 warmup，在同步边界后采样；不能用异步 launch 时间冒充 kernel 执行时间。
7. 在被测区间前调用 `torch.cuda.reset_peak_memory_stats()`；在被测区间后记录
   `torch.cuda.max_memory_allocated()` 和 `torch.cuda.max_memory_reserved()`。
8. 启用 allocator memory history，保存 snapshot 与可审计 trace。两个文件都必须纳入 SHA-256 校验。
9. 记录 device name/index/compute capability、driver、CUDA runtime、Triton、Python、PyTorch 与 backend。

静态 Scheduler peak 估计、逻辑 saved-tensor bytes、CPU 进程 RSS 或生成但未执行的 Triton source，都不能替代 CUDA allocator peak。

### 5.3 CUDA 验收命令

```bash
REPO=/path/to/llm-knowledge
LAB="$REPO/wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/labs"
BUNDLE="${TMPDIR:-/tmp}/native_backend_cuda_run"

python -B "$LAB/native_backend_contract.py" produce \
  --target cuda \
  --output "$BUNDLE/result.json" \
  --artifact-root "$BUNDLE"
producer_exit=$?
if [ "$producer_exit" -ne 0 ]; then
  echo "CUDA native producer did not pass; inspect $BUNDLE/result.json" >&2
  exit "$producer_exit"
fi

python -B "$LAB/native_backend_contract.py" validate \
  --input "$BUNDLE/result.json" \
  --target cuda \
  --artifact-root "$BUNDLE"
test $? -eq 0
```

## 6. JSON 合同的完整字段

顶层固定包含：

- `schema_version`、UTC timestamp、`artifact_type`；
- `host`：hostname、OS/version、architecture；
- `runtime`：Python executable/version、PyTorch version/git version；
- `capabilities`：CPU compiler、CUDA 和 Triton 的可用性及诊断细节；
- `statuses`：`cpu_native` 与 `cuda_triton`。
- `producer`：固定脚本标识、合同版本、脚本 SHA-256、完整 producer 命令与目标。

CPU PASS 额外包含：

- compiler path/version；
- generated source path/hash/language；
- compile command/return code、stdout/stderr 日志和二进制 artifact path/hash；
- load method、entry points、execute return code/call count；
- pointwise 与 reduction 各自的输入、eager/native 输出摘要、数值比较、warmup、iterations、全部 timing samples 和统计量；
- 完整环境版本。

CUDA/Triton PASS 额外包含：

- device、driver、CUDA runtime、Triton；
- generated source path/hash；
- candidate list、参数、真实 samples、winner；
- cache artifact path/hash；
- execution return code/call count、输入与 eager/native 输出摘要、numerical comparison；
- allocator max allocated/reserved、memory snapshot/trace 及各自哈希；
- warmup、iterations 和完整环境版本。

Schema 负责必填字段、类型、枚举、合法状态和 PASS 条件；Python validator 在同一 Schema 之上增加以下语义检查：

- 样本数必须等于 iterations；
- 所有 timing、tolerance 与 error 都必须是有限数；
- min/median/mean/p95/max 必须由 samples 推出；p95 使用 nearest-rank 定义；
- producer 脚本 hash/命令/目标必须一致；CPU compiler/source/output 必须绑定到真实 compile command；
- CPU 每个 workload 必须有独立数值比较、输入输出摘要和已加载 entry point；
- CUDA candidate 名称与参数配置必须唯一，device index 必须落在实测 device count 内；
- CUDA winner 必须是最低实测 median，且 winner timing 与 candidate 一致；
- max reserved 不得小于 max allocated；
- result 中的 compiler/device/environment 版本必须与顶层 capability、host、runtime 记录一致；
- bundle 文件必须存在、真实路径不能通过 symlink/junction 越出 bundle，且 SHA-256 一致；CUDA source/cache/snapshot/trace 必须使用不同角色路径与内容哈希；
- diagnostic、BLOCKED、NOT_RUN、FAIL 不能作为 acceptance。

公开 CLI 故意不提供关闭文件校验的选项；带 `--no-verify-files` 的命令会被参数解析器拒绝。单元测试可以直接调用 Python API 的结构检查路径，但该路径不构成外部验收。

## 7. 当前结论与外部闭环条件

当前本机只能交付可信的 `BLOCKED` capability diagnostic，不能交付 CPU native 或 CUDA/Triton PASS。要关闭这两个 blocker，必须分别从满足第 4 节和第 5 节环境要求的执行机返回完整 bundle，并通过对应 `validate` 命令。

外部执行尚未发生时，报告应写“合同、探测和复现命令已就绪；原生执行证据 BLOCKED”，不能写“测试已跳过但整体 PASS”。

## Related Pages

- [[21_codegen_kernel_mapping_autotuning_and_provenance]]
- [[11_graph_stage_boundaries_identity_and_provenance]]
- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
