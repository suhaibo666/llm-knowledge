# torch.compile 卷级教学 Demo 设计

> 日期：2026-07-29  
> 状态：已批准实施  
> 用户确认：采用“每卷一个总 Demo，卷内多个用例”；以 CUDA 为正式验收主线；当前没有可用 CUDA 环境，先完成代码与无 GPU 证据边界。  
> 源码审计基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 当前可运行预检环境：Windows、Python 3.13、PyTorch `2.9.1+cpu`

## 1. 目标

为 `19_torch_compile_end_to_end` A–F 六卷建立统一、可发现、可分用例执行的教学 Demo：

1. 每卷只有一个主入口，避免初学者在几十个脚本间寻找入口。
2. 每个主入口内部按机制拆成独立 `case`，可以单跑、列举和组合运行。
3. A/B/D/E/F 的 39 篇正文都链接到一个主 case；C01–C21 继续保留既有逐篇 Lab，并由 C 卷入口统一编排。
4. 每个 case 产生机器可读结果，区分 `PASS`、`BLOCKED` 和 `FAIL`。
5. CUDA 是正式验收主线；CPU 只用于设备无关机制和合同预检。
6. 没有实际 GPU receipt 时，不把生成代码、FakeTensor、mock、CPU 或源码推断写成 CUDA 已验证事实。

## 2. 非目标

- 不重新实现 C 卷已有的 FX/AOT/PatternMatcher/Inductor Labs。
- 不承诺当前机器完成 CUDA、Triton、多卡、CUDAGraph 或 AOTInductor 原生验收。
- 不把 microbenchmark 结果写成跨设备、跨版本的性能结论。
- 不要求一个 Python 进程同时演示所有机制；卷级入口可以为隔离状态启动子进程。
- 不引入 pytest、click、pydantic 等新依赖，继续使用标准库和 PyTorch。

## 3. 总体结构

```text
labs/
├── demo_harness.py
├── demo_manifest.json
├── demo_a_execution_model.py
├── demo_b_dynamo_capture.py
├── demo_c_graph_compiler.py
├── demo_d_artifact_runtime.py
├── demo_e_diagnostics.py
├── demo_f_advanced_topics.py
├── test_volume_demo_contract.py
└── artifacts/volume_demos/<volume>/<run-id>/
```

`demo_harness.py` 负责 CLI、能力探测、结果状态、环境快照和 JSON 写入。六个卷文件只负责定义 case 及其机制代码。`demo_manifest.json` 是“教学页 → 卷入口 → case”的唯一映射源。

## 4. CLI 合同

所有卷入口共享：

```powershell
python -B <volume-demo>.py --list
python -B <volume-demo>.py --case <case-id> --device cuda --output-dir <dir>
python -B <volume-demo>.py --case all --device cuda --output-dir <dir>
```

参数：

- `--list`：只列 case、关联页面和能力要求，不执行模型。
- `--case`：可重复；`all` 表示本卷所有 case。
- `--device`：`cuda` 或 `cpu`，默认 `cuda`。
- `--output-dir`：结果根目录；每个 case 写独立 JSON。
- `--seed`：默认 0。

退出码：

- `0`：所选 case 全部 `PASS`。
- `2`：至少一个 case `FAIL`。
- `3`：无 `FAIL`，但至少一个 case 因能力缺失而 `BLOCKED`。
- `4`：CLI 或 manifest 合同错误。

## 5. 结果合同

每个 case 结果至少包含：

```json
{
  "schema_version": "torch-compile-volume-demo/v1",
  "volume": "B",
  "case_id": "guards_recompile",
  "status": "PASS",
  "pages": ["b07"],
  "requirements": ["torch"],
  "environment": {
    "python": "...",
    "torch": "...",
    "torch_git": "...",
    "cuda_available": false,
    "cuda_runtime": null,
    "device_count": 0
  },
  "observations": {},
  "artifacts": [],
  "limitations": []
}
```

`BLOCKED` 必须给出缺失能力及探测值。`FAIL` 必须保留异常类型和简短 traceback。任何 case 都不能通过吞掉异常返回 `PASS`。

## 6. 卷与用例

### A：执行模型前置基础

入口：`demo_a_execution_model.py`

| case | 主页面 | 机制 |
|---|---|---|
| `tensor_storage_layout` | A01 | storage、stride、view、clone、alias |
| `dispatcher_autograd` | A02 | dispatcher 观察、autograd graph 与梯度 |
| `python_frame_bytecode` | A03 | frame/code object/bytecode 与 Python 执行边界 |
| `proxy_fake_tensor` | A04 | TorchDispatchMode、FakeTensor、make_fx |
| `eager_compile_cost` | A05 | 首次编译、缓存重放和 CUDA event 稳态测量 |

### B：TorchDynamo 捕获

入口：`demo_b_dynamo_capture.py`

| case | 主页面 | 机制 |
|---|---|---|
| `compile_lifecycle` | B01 | wrapper 创建、首次 frame 命中、backend 回调、复用 |
| `backend_modes_fullgraph` | B02 | backend/mode/fullgraph 的合同差异 |
| `eval_frame_cache` | B03 | code object、backend compile count 和 cache reuse |
| `bytecode_state_machine` | B04 | 分支、stack/local 状态和图捕获边界 |
| `variable_source_guards` | B05 | Source 派生 guard 与对象属性变化 |
| `output_graph_side_effects` | B06 | 输出图、mutation/side effect 和提交边界 |
| `guards_recompile` | B07 | guard 命中、失败和重编译 |
| `graph_break_resume` | B08 | graph break、多个 backend graph 和恢复执行 |
| `dynamic_shapes` | B09 | static、automatic dynamic、越界与 fallback |
| `custom_backend_contract` | B10 | GraphModule/example_inputs/backend callable 合同 |

### C：图编译核心

入口：`demo_c_graph_compiler.py`

| case | 页面范围 | 机制 |
|---|---|---|
| `ir_fx` | C01–C06 | 调用既有 Part I 脚本 |
| `capture_normalize` | C07–C08 | 调用既有捕获/规范化脚本 |
| `aot_recompute` | C09–C11 | 调用既有 AOT、recompute、activation 脚本 |
| `pattern_rewrite` | C12–C16 | 调用既有 PatternMatcher/pass/legality 脚本 |
| `inductor_ir` | C17–C21 | 调用既有 IR/Scheduler/artifact 脚本 |
| `full_bundle` | C01–C21 | 调用 `series_artifact_bundle.py` |

C 入口保存子进程命令、退出码、stdout/stderr 和已有 artifact 路径，不改变原脚本的证据等级。

### D：编译产物、缓存与运行时

入口：`demo_d_artifact_runtime.py`

| case | 主页面 | 机制 |
|---|---|---|
| `compile_fx_orchestration` | D01 | Dynamo backend → AOT/Inductor 编排观察 |
| `aot_wrappers_lazy_backward` | D02 | forward wrapper 与首次 backward 编译 |
| `async_compile_loading` | D03 | 编译任务、等待点、模块装载 artifact |
| `cache_keys_invalidation` | D04 | 冷/热 cache、输入/配置变化与失效 |
| `wrapper_memory_reuse` | D05 | wrapper 输出、地址复用与 allocator 统计 |
| `cudagraph_replay` | D06 | warmup、record、replay、step marker |
| `artifact_lifecycle_failure` | D07 | created/loaded/executed/failed/fallback 状态 |

### E：调试、正确性与性能

入口：`demo_e_diagnostics.py`

| case | 主页面 | 机制 |
|---|---|---|
| `logs_artifact_map` | E01 | 日志类别、计数器和 artifact 对阶段的映射 |
| `dynamo_explain` | E02 | graph、break reason、guards、compile count |
| `guard_failure` | E03 | guard failure 与 recompile 诊断 |
| `stage_failure_localization` | E04 | capture/AOT/lowering/runtime 故障注入 |
| `minifier_repro` | E05 | 最小复现输入与失败保真边界 |
| `correctness_validation` | E06 | eager/compiled 输出、梯度、alias/mutation 对照 |
| `cold_warm_steady` | E07 | 冷启动、热缓存、稳态 CUDA event 测量 |
| `fusion_memory_profiler` | E08 | profiler kernel、memory 和 fusion 对照 |
| `rollout_fallback` | E09 | 分批启用、错误预算、回退与结果记录 |

### F：训练、分布式、扩展与部署

入口：`demo_f_advanced_topics.py`

| case | 主页面 | 机制 |
|---|---|---|
| `compiled_autograd` | F01 | backward capture 与 AOTAutograd 区别 |
| `checkpoint_recompute` | F02 | activation checkpoint 与 compile/recompute |
| `ddp_compile` | F03 | rank-local graph、DDP 边界和 optimizer step |
| `fsdp_dtensor` | F04 | 多卡 FSDP/DTensor placement 与 collective 边界 |
| `custom_op_contract` | F05 | custom op、fake kernel、autograd/decomposition |
| `custom_backend` | F06 | backend callable、能力声明和失败边界 |
| `aotinductor_package` | F07 | export、package、load、运行 ABI |
| `inference_freezing_cudagraph` | F08 | inference/freezing/CUDAGraph 正交组合 |

## 7. 能力探测

能力键至少包含：

- `torch`
- `cuda`
- `cuda_multi_gpu`
- `distributed`
- `triton`
- `native_compiler`
- `linux`

能力探测只回答“环境是否具备运行前提”，不能回答 case 是否正确。GPU-only case 在进入任何 CUDA API 之前完成探测。

## 8. 教学 Markdown 关联

1. A/B/D/E/F 每篇正文在最终 `## Related Pages` 前增加 `## 配套 Demo`。
2. 每个区块包含入口、case、命令、预期观察字段和无能力时的状态。
3. C01–C21 保留原 `已验证 Lab`；C 卷入口由课程总索引和 Labs README 统一链接。
4. `demo_manifest.json` 覆盖全部 60 篇正文，测试保证页面映射无遗漏、case 真正存在。
5. 总索引移除“当前阶段不新增 demo”的旧说明，增加六卷运行顺序。

## 9. 测试策略

- 测试先于实现。
- Harness 单元测试覆盖选择、能力阻塞、异常失败、JSON 输出和退出码。
- 六个入口均以 subprocess 验证 `--list`。
- 每卷至少一个设备无关 case 在 CPU 运行并断言真实行为。
- 每卷至少一个 CUDA-only case 在当前环境断言产生 `BLOCKED`，不是 `PASS`。
- C 入口验证对子脚本退出码和 stdout/stderr 的保真传播。
- manifest 测试覆盖 60 篇正文、case 存在性及 Markdown 回链。
- 保留并重跑原 90 项审计测试和 42 项课程合同测试。

## 10. 验收边界

当前机器可以交付：

- 六个卷级入口和完整 case registry；
- CPU 可运行机制的真实预检；
- CUDA/multi-GPU/Triton/AOTI 用例的代码、能力门和 `BLOCKED` artifact；
- 教学页关联、合同测试和审计账本闭合。

当前机器不能交付：

- CUDA case 的 `PASS` receipt；
- GPU kernel 数、显存峰值、加速比或 autotune winner；
- 多卡 collective/FSDP/DTensor 的真实执行结论；
- AOTInductor 二进制 package 的可装载性结论。

这些项目必须等实际 CUDA 环境运行后才能升级状态。

