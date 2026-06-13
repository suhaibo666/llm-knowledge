# NPU 编译工作流: 毕昇编译器、Autotune、精度校验

> 覆盖 NPU MLIR 后端的编译执行全链路：三种编译模式、60 维 Autotune、在线精度对比、缓存管理
> 最后更新: 2026-05-09

---

## 编译执行链路

```
Scheduler 融合完成后的 SchedulerNode (含 traced_graph)
  → NpuMlirScheduling.codegen_node()
    → create_fx_from_snodes_by_traced_graph()   // 重建 FX Graph
    → FxImporter.import_stateless_graph()        // FX → MLIR Torch dialect
    → bishengir-compile                          // MLIR → NPU kernel (.so)
    → torch_npu._C.mlir.load()                   // 加载编译产物
    → 执行
```

详见 [[NPU_MLIR_Backend_Technical_Analysis]] 六阶段全景分析。

---

## 三种编译模式

NPU MLIR 路径在运行时有一个 GPU 没有的决策树——MLIR 编译可能失败，需要回退策略。

```
每个 kernel 编译时:

  compile_mode = "auto_fallback" (默认)
    ├── num_call_functions ≤ 1? → complete_fallback (自动降级)
    ├── MLIR 编译 → 成功? → 执行 MLIR kernel
    └── MLIR 编译 → 失败? → 自动回退 FX Graph eager 执行

  compile_mode = "default"
    ├── MLIR 编译 → 成功? → 执行 MLIR kernel
    └── MLIR 编译 → 失败? → 抛异常 (无回退)

  compile_mode = "complete_fallback"
    └── 跳过 MLIR, 直接 FX Graph eager 执行
```

### 自动降级规则

```python
# codegen/mlir.py:273-392
num_call_functions = get_num_call_functions(mlir_kernel._gm)
if num_call_functions <= 1:
    mode = "complete_fallback"
    # 原因: 单算子 kernel 的 MLIR 编译开销大于收益

if kernel_name in anir_config.force_fallback_kernel_names:
    mode = "complete_fallback"
    # 原因: 用户手动指定的回退 kernel
```

### 回退数据保存

`auto_fallback` 和 `complete_fallback` 模式下，FX Graph 被序列化到磁盘：

```python
dump_path = os.path.join(os.getenv("TORCHINDUCTOR_CACHE_DIR"),
                          anir_config.traced_graph_cache, ...)
to_folder(mlir_kernel._gm, dump_path, ...)
```

用于后续加载 FX Graph eager 执行和精度对比。

---

## 毕昇编译器接口

毕昇编译器 (`bishengir-compile`) 通过 subprocess 调用，入口在 `mlir_compiler.py:103-151`：

```python
def bisheng_compile(self, input_path, output_path,
                    auto_db=True, ops_reorder=False,
                    tiling_size=None, extra_command=None):
    command = [
        bisheng_ir_compile_path,
        "-enable-hfusion-compile=true",     # 始终: 水平融合
        "--enable-bin-relocation=0",        # 始终: 禁用二进制重定位
        f"-block-dim={anir_config.block_dim}",  # block 维度 (默认48)
    ]

    if auto_db:
        command.append("--enable-auto-multi-buffer=true")    # 多缓冲
    else:
        command.append("--enable-auto-multi-buffer=false")

    if ops_reorder:
        command.append("--enable-ops-reorder=true")          # 算子重排
    else:
        command.append("--enable-ops-reorder=false")

    if anir_config.autotune:
        command.append("-enable-tuning-mode=true")            # 调优模式

    if self.dynamic:
        command.append("--enable-static-bare-ptr=false")
        command.append("--enable-symbol-analysis=true")      # 动态形状符号分析

    command += [input_path, "-o", output_path]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, timeout=600)
```

| 编译选项 | 默认 | 作用 |
|---------|------|------|
| `-enable-hfusion-compile` | `true` (始终) | 水平算子融合 — Ascend Vector/Cube 单元并行 |
| `--enable-bin-relocation` | `0` (始终) | 禁用二进制重定位 — NPU 不需要 |
| `-block-dim` | `48` (可配) | AI Core block 调度粒度 |
| `--enable-auto-multi-buffer` | `true`/`false` (autotune 决定) | 双/多缓冲 — 隐藏内存延迟，UB 大小固定 |
| `--enable-ops-reorder` | `true`/`false` (autotune 决定) | 指令重排 — 优化 NPU 流水线利用率 |
| `--enable-symbol-analysis` | 动态形状时 `true` | 符号维度分析 |
| `-enable-tuning-mode` | autotune 时 `true` | 启用调优模式 |

### 与 GPU Triton JIT 的核心区别

| | GPU Triton JIT | NPU 毕昇编译器 |
|---|---|---|
| 调用方式 | Python API (`@triton.jit` 装饰器) | subprocess 命令行 |
| 编译时机 | 首次执行时 JIT | compile_fx 阶段 AOT |
| 编译产物 | cubin (内存中) | .so 文件 (磁盘) |
| 融合能力 | 无 (Scheduler 已完成融合) | hfusion 水平融合 |
| 失败处理 | 抛异常 | auto_fallback 自动回退 |

---

## Autotune: 60 种编译配置

NPU 的 autotune 不同于 GPU Triton 的运行时 benchmark 搜索。它是**编译时参数搜索**：不同的编译选项组合产生不同的 kernel binary，首次执行时 benchmark 选最优。

```python
# mlir_compiler.py:335-343
def get_autotune_config(self):
    def get_tiling_range():
        return [i for i in range(-10, 20, 2)]   # 15 个值

    compile_args = []
    for ops_reorder in [True, False]:            # 2 种
        for auto_db in [True, False]:            # 2 种
            for tiling_size in get_tiling_range(): # 15 种
                compile_args.append((tiling_size, ops_reorder, auto_db))
    return compile_args  # 15 × 2 × 2 = 60 种组合
```

### 首次执行流程

```
NpuMlirCompiler.run() 首次调用
  → 注册 FX fallback 基准 (用于精度对比)
  → 遍历 60 种编译配置
    → 每种配置调用 bishengir-compile 编译
    → benchmark 执行时间
  → accuracy_pass 精度校验
    → 过滤掉精度不达标的配置
  → 选择最快且精度通过的配置
  → cache 最优配置供后续使用
```

### autotune=False 时

仅使用默认参数 `(tiling_size=None, ops_reorder=True, auto_db=True)`，单次编译，不做 benchmark。

---

## 在线精度对比

环境变量 `ANIR_ONLINE_ACC_COMP=1` 启用。原理：MLIR kernel 输出 vs FX reference 输出逐元素比较。

```python
# mlir_compiler.py:520-596
def acc_compare_and_dump(self, *args, **kwargs):
    # 1. 注册 FX 参考实现
    self.register_fx_fallback(self.kernel_meta)

    # 2. FX 参考执行 (bfloat16 提升为 float32 提高基准精度)
    fx_outputs = [x.to(torch.float32) if x.dtype == torch.bfloat16
                  else x for x in args[-num_outputs:]]
    launcher_fx(*(fx_inputs + fx_outputs), **kwargs)

    # 3. MLIR kernel 执行
    output = launcher(*args_new, **kwargs)

    # 4. 逐输出对比
    for idx, (actual, expected) in enumerate(zip(outputs, fx_outputs)):
        matches = torch.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
        if not matches.all():
            has_acc_error = True
            args[idx + num_inputs].copy_(expected)  # 用 FX 结果修正

    # 5. dump 精度失败子图
    if has_acc_error:
        self.fx_subgraph_dump('acc_failed')
```

### 精度容差配置

| dtype | rtol | atol |
|-------|------|------|
| float32 | 1.3e-6 | 1e-5 |
| float16 | 1e-3 | 1e-5 |
| bfloat16 | 1.6e-2 | 1e-5 |

---

## 芯片感知

NPU 编译在不同芯片型号上有不同策略：

| 芯片 | soc_version | UB 大小 | 策略差异 |
|------|------------|---------|---------|
| 910B1 | 220 | 192 KB | 白名单模式，白名单外全 fallback |
| 310B1 | 240 | — | vector core : cube core = 1:1 |
| 910_9391 | 250 | 256 KB | 黑名单模式，黑名单外允许 codegen |

```python
# config.py
# UB 大小
ub_size = 192 * 1024
if get_soc_version() >= Ascend910_9391:
    ub_size = 256 * 1024

# Core 数量比率
num_vector_core = num_cube_core
if Ascend910B1 <= soc < Ascend310B1 or soc >= Ascend910_9391:
    num_vector_core = num_cube_core * 2  # vector core 是 cube 的 2 倍
```

芯片差异影响：
- **UB 大小** → 影响 auto_multi_buffer 的可行性
- **vector/cube 比率** → 影响 ops_reorder 的策略
- **白名单/黑名单模式** → 影响 op 分流策略

---

## 缓存管理

NPU 编译产物缓存通过环境变量控制：

| 变量 | 作用 |
|------|------|
| `TORCHINDUCTOR_CACHE_DIR` | 编译产物根目录 |
| `anir_config.traced_graph_cache` | FX Graph 缓存（回退用） |

编译后的 `.so` 文件和 FX Graph dump 均存入缓存目录。相同输入（相同 traced_graph hash）的 kernel 直接使用缓存。

---

## 调试入口

```bash
# 启用精度对比
ANIR_ONLINE_ACC_COMP=1 python script.py

# dump FX Graph (每个 op 的 traced graph)
INDUCTOR_ASCEND_DUMP_FX_GRAPH=1 python script.py

# 强制某个 kernel fallback
# 在代码中: anir_config.force_fallback_kernel_id = [1, 2, 10]
# 或全部 fallback: anir_config.force_fallback_kernel_id = 'all'

# 详细日志
INDUCTOR_ASCEND_LOG_LEVEL=DEBUG python script.py
```

---

## Related Pages

- [[NPU_MLIR_Backend_Technical_Analysis]] — TracedGraph 机制、Monkey Patch 清单、编译模式状态机、六阶段适配全景
- [[npu_lowering_guide]] — NPU lowering op 分流策略、NPU 专有 IR 节点
- [[NPU_Inductor_Backend_Analysis]] — 多后端选择与混合使用
- [[NPU_Inductor_Backend_Analysis]] — MultiTemplateBuffer、Epilogue/Prologue Fusion
- [[triton_vs_mlir_backend_analysis]] — Triton vs Torch-MLIR 通用对比
- [[02_engineering/01_ai_frameworks/index]]
