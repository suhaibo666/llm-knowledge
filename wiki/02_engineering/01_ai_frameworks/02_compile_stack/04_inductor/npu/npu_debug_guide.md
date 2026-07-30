# NPU torch.compile 调试指南

> 层次：quick start / 核验基准：torch_npu v2.7.1.post5 / 最后更新 2026-06-15

> 本节专门针对使用 `torch_npu`（华为昇腾 NPU）时的调试方法，包括 NPU 特有的环境变量、API 和工具。

## NPU 环境变量配置

### 基础日志环境变量

```bash
# 全局日志级别（0:debug, 1:info, 2:warning, 3:error, 4:null）
export ASCEND_GLOBAL_LOG_LEVEL=0

# 禁用 NPU 告警信息（可选）
export TORCH_NPU_DISABLED_WARNING=1

# 精简错误输出（将 CANN 内部调用栈转移到 plog）
export TORCH_NPU_COMPACT_ERROR_OUTPUT=1
```

### 算子编译与执行调试

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

### 内存管理调试

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

### 集合通信（HCCL）调试

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

### 特征值检测（ASD）

```bash
# 特征值检测（0:关闭, 1:开启仅打印, 2:开启并告警, 3:开启告警并记录过程数据）
export NPU_ASD_ENABLE=2

# 特征值检测绝对阈值
export NPU_ASD_UPPER_THRESH=100000

# 特征值检测相对阈值
export NPU_ASD_SIGMA_THRESH=4.0
```

### Profiling 与性能分析

```bash
# Profiling 配置文件路径
export PROF_CONFIG_PATH=./profiler_config.json

# 使用 msMonitor nputrace 方式
export KINETO_USE_DAEMON=1

# GE Profiling 输出到标准输出
export GE_PROFILING_TO_STD_OUT=1
```

### 设备管理

```bash
# Stream pool 最大流数
export STREAMS_PER_DEVICE=32

# NPU 设备能力返回值
export TORCH_NPU_DEVICE_CAPABILITY=9.0
```

## NPU API 调试方法

### Dump 功能（算子数据 dump）

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

### Profiling 功能

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

### Dynamic Profile（动态 profiling）

```python
from torch_npu.profiler import dynamic_profile as dp

# 初始化
dp.init("profiler_config_path")

# 训练循环
for step in range(num_steps):
    train_one_step()
    dp.step()
```

### 同步调试模式

```python
import torch_npu

# 设置同步调试模式（0:default, 1:warn, 2:error）
torch_npu.npu.set_sync_debug_mode(2)

# 获取当前模式
mode = torch_npu.npu.get_sync_debug_mode()
```

### 算子编译缓存清理

```python
import torch_npu

# 清理算子编译缓存
torch_npu.npu.clear_op_cache()
```

## NPU 特有调试脚本

### `run_npu_debug.sh`

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

### `collect_npu_artifacts.sh`

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

## NPU 常见问题与快速修复

| 问题类型 | 可能原因 | 快速修复方法 |
|---------|---------|------------|
| **算子编译失败** | 算子不支持 / 参数不匹配 | 检查 `ACL_OP_COMPILER_CACHE_DIR`，使用 `ASCEND_LAUNCH_BLOCKING=1` 同步调试 |
| **OOM 错误** | 显存不足 / 内存泄漏 | 设置 `OOM_SNAPSHOT_ENABLE=1` 分析快照，使用 `PYTORCH_NPU_ALLOC_CONF` 调整分配策略 |
| **通信超时** | HCCL 配置问题 / 网络问题 | 设置 `HCCL_DESYNC_DEBUG=1` 和 `HCCL_ASYNC_ERROR_HANDLING=1` 分析 |
| **数值精度问题** | 精度模式 / 算子实现差异 | 检查 `INF_NAN_MODE_ENABLE`，使用 `NPU_ASD_ENABLE` 检测异常值 |
| **性能问题** | 算子未融合 / 内存拷贝多 | 使用 Profiling 分析，开启 `TASK_QUEUE_ENABLE` 和 `TORCH_HCCL_ZERO_COPY` |
| **图捕获失败** | 不支持的算子 / 控制流问题 | 使用 `torch_npu.npu_config.init_dump()` 分析算子调用，参考 Dynamo 调试方法 |

## NPU 与 CUDA 调试差异对比

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

## NPU 调试最佳实践

1. **分层调试**：先使用 `ASCEND_GLOBAL_LOG_LEVEL=0` 获取详细日志，再根据问题类型设置特定环境变量
2. **启用同步模式**：在调试算子问题时，设置 `ASCEND_LAUNCH_BLOCKING=1` 便于定位错误
3. **保存调试产物**：使用 `collect_npu_artifacts.sh` 统一收集所有调试信息
4. **使用 Profiling**：性能问题优先使用 `torch_npu.profiler` 分析瓶颈
5. **内存分析**：OOM 问题启用 `OOM_SNAPSHOT_ENABLE=1` 保存快照进行分析
6. **通信调试**：分布式问题启用 `HCCL_DESYNC_DEBUG=1` 和 `HCCL_ASYNC_ERROR_HANDLING=1`
7. **算子 dump**：特定算子问题使用 `torch_npu.npu_config.init_dump()` 精准定位

## Related Pages

- [[Pytorch_Compile_Debug_Analysis]]（upstream 调试）
- [[npu_compile]]
- [[02_compile_stack/04_inductor/npu/index]]
- [[01_ai_frameworks/index]]
