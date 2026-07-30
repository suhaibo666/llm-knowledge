# TorchInductor Quick Start

> 层次:quick start · 核验基准:PyTorch 上游(`E:\97-codes\pytorch\pytorch`) · 最后更新 2026-06-13
>
> 本页所有 API / 参数 / 配置 / 环境变量均对照真实源码核实,标注 `path:line`。深入原理见文末导航。

## 1. Inductor 是什么 / 何时用

TorchInductor 是 `torch.compile` 默认的**后端代码生成器**:把 Dynamo 捕获、AOTAutograd 处理过的 FX 图,经调度(scheduler)与降级(lowering)生成 Triton(GPU/NPU)或 C++/OpenMP(CPU)kernel,实现算子融合、内存复用与 autotuning。

- **何时用**:训练/推理热点循环里希望"少改代码拿性能"——直接套一层 `torch.compile`,前向+反向都会被编译。
- **何时别用**:一次性脚本、控制流极度动态导致频繁重编译、或需要逐算子精确调试时。
- 默认 `backend="inductor"`(`torch/__init__.py:2579`,`None` 时解析为默认后端 `torch/__init__.py:2750-2753`)。

## 2. 最小可跑示例(前向 + 反向)

```python
import torch

@torch.compile  # 等价 torch.compile(model, backend="inductor", mode="default")
def f(x, w):
    return torch.relu(x @ w)

x = torch.randn(64, 128, device="cuda", requires_grad=True)
w = torch.randn(128, 256, device="cuda", requires_grad=True)

out = f(x, w)        # 首次调用触发编译(前向图)
out.sum().backward() # 反向图同样被 Inductor 编译(AOTAutograd 联合图)
print(x.grad.shape)  # torch.Size([64, 128])
```

要点:`torch.compile` 是**惰性、按帧编译**——首次执行某 frame 才编译并缓存到 code object;输入形状/类型变化可能触发重编译(guard failure),用 `TORCH_LOGS=guards` 调试(`torch/__init__.py:2628-2629`)。

## 3. 关键参数速查(`torch.compile`)

签名见 `torch/__init__.py:2603-2619`。仅列**真实存在**的参数:

| 参数 | 类型 / 默认 | 说明 | 来源 |
|------|------------|------|------|
| `fullgraph` | `bool=False` | True 要求整函数捕获为单图,有 graph break 即报错;并默认开启 unbacked 语义 | `torch/__init__.py:2606`,`2638-2642` |
| `dynamic` | `bool\|None=None` | True 尽量生成动态形状 kernel;False 永远特化;None 自动检测后重编译为更动态 | `torch/__init__.py:2607`,`2643-2649` |
| `backend` | `str\|Callable\|None` | 后端,默认 `"inductor"`;`torch._dynamo.list_backends()` 看可选项 | `torch/__init__.py:2608`,`2650-2659` |
| `mode` | `str\|None=None` | `"default"`/`"reduce-overhead"`/`"max-autotune"`/`"max-autotune-no-cudagraphs"` | `torch/__init__.py:2609`,`2660-2678` |
| `options` | `dict\|None=None` | 直传后端的 config(见 §4);**与 `mode` 互斥** | `torch/__init__.py:2610`,`2784-2787` |
| `disable` | `bool=False` | 变 no-op,用于测试 | `torch/__init__.py:2612`,`2707` |
| `recompile_limit` | `int\|None=None` | 本次 compile 的重编译上限;None 用全局 `torch._dynamo.config.recompile_limit`(8) | `torch/__init__.py:2613`,`2708-2711` |
| `isolate_recompiles` | `bool=False` | True 时本 compile 独立计重编译次数 | `torch/__init__.py:2614`,`2712-2726` |
| `name` | `str\|None=None` | 编译区域可选标识,出现在调试元数据 | `torch/__init__.py:2611`,`2705-2706` |
| `shapes_spec` | `Any=None` | 显式形状规格(ParamsSpec/ShapesSpec) | `torch/__init__.py:2615`,`2756-2760` |

> `mode` 与 `options` 不能同时给,否则 `RuntimeError`(`torch/__init__.py:2784-2787`);两者都为 None 时回落到 `"default"`(`torch/__init__.py:2788-2789`)。各 mode 具体开了哪些 config,运行 `torch._inductor.list_mode_options()`(`torch/__init__.py:2678`)。

常用 `options` 键(均为 Inductor config,逐项见 §4):`max_autotune`、`epilogue_fusion`、`triton.cudagraphs`、`shape_padding`、`fallback_random`、`trace.enabled`(`torch/__init__.py:2680-2704`)。

## 4. 常用 config 与环境变量

### Inductor config(`torch._inductor.config`,文件 `torch/_inductor/config.py`)

可两种方式设:`torch._inductor.config.max_autotune = True`,或 `torch.compile(..., options={"max_autotune": True})`。完整列表 `torch._inductor.list_options()`。

| config | 默认 | 对应环境变量 | 来源 |
|--------|------|-------------|------|
| `max_autotune` | `False` | `TORCHINDUCTOR_MAX_AUTOTUNE=1` | `config.py:514` |
| `max_autotune_gemm` | `False` | `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1` | `config.py:520` |
| `max_autotune_pointwise` | `False` | `TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1` | `config.py:517` |
| `coordinate_descent_tuning` | `False` | `TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1` | `config.py:756-758` |
| `epilogue_fusion` | `True` | — | `config.py:274` |
| `shape_padding` | `True` | `TORCHINDUCTOR_SHAPE_PADDING`(默认 `"1"`) | `config.py:1465` |
| `cpp_wrapper` | `False` | `TORCHINDUCTOR_CPP_WRAPPER=1` | `config.py:196` |
| `freezing` | `False` | `TORCHINDUCTOR_FREEZING=1` | `config.py:1561` |
| `fallback_random` | `False` | — | `config.py:888` |
| `benchmark_kernel` | `False` | `TORCHINDUCTOR_BENCHMARK_KERNEL=1` | `config.py:1020` |
| `compile_threads` | 自动(`None`→自动决定) | — | `config.py:1373` |
| `debug` | `False` | — | `config.py:90` |
| `triton.cudagraphs` | `False` | `TORCHINDUCTOR_CUDAGRAPHS=1` | `config.py:1804`(class),`1810` |
| `trace.enabled` | `False` | `TORCH_COMPILE_DEBUG=1` | `config.py:2720`(class),`2724` |

### FX Graph Cache(磁盘缓存,加速重复编译)

| config | 默认 | 环境变量 | 来源 |
|--------|------|---------|------|
| `fx_graph_cache` | `True` | `TORCHINDUCTOR_FX_GRAPH_CACHE`(强制)/ `TORCHINDUCTOR_FX_GRAPH_CACHE_DEFAULT` | `config.py:109-114` |
| `fx_graph_remote_cache` | `None`(OSS 关) | — | `config.py:122` |

### 缓存目录

- `TORCHINDUCTOR_CACHE_DIR`:Inductor 产物缓存根目录(`runtime/cache_dir_utils.py:14-21`)。
- 未设时默认 `<tempdir>/torchinductor_<username>`(`runtime/cache_dir_utils.py:24-34`)。

### Dynamo config(`torch._dynamo.config`,文件 `torch/_dynamo/config.py`)

| config | 默认 | 说明 | 来源 |
|--------|------|------|------|
| `recompile_limit` | `8` | 单 frame 重编译上限,超过回落 eager | `config.py:121` |
| `accumulated_recompile_limit` | `256` | 全局累计重编译上限 | `config.py:124` |
| `cache_size_limit` | (别名→`recompile_limit`) | 旧名,等价 `recompile_limit` | `config.py:137` |
| `fail_on_recompile_limit_hit` | `False` | 超限时报错而非回落 | `config.py:135` |

## 5. 何时选哪个 mode

| mode | 适用 | 代价 | 来源 |
|------|------|------|------|
| `default` | 通用,性能/编译开销平衡 | 编译快 | `torch/__init__.py:2662-2663` |
| `reduce-overhead` | 小 batch / 启动开销大;启用 CUDA graphs | 更高显存(缓存 workspace),仅对不改输入的纯 CUDA 图生效 | `torch/__init__.py:2664-2670` |
| `max-autotune` | 追极致吞吐;Triton/模板 matmul,GPU 上默认开 CUDA graphs | 编译慢(profiling 选型) | `torch/__init__.py:2672-2674` |
| `max-autotune-no-cudagraphs` | 同上但不要 CUDA graphs | 编译慢 | `torch/__init__.py:2676` |

经验:开发期用 `default`;部署/benchmark 再试 `max-autotune`;推理小 batch 延迟敏感选 `reduce-overhead`。

## 6. 看生成代码 / 调试

```bash
# 打印 Inductor 生成的 Triton / C++ 代码
TORCH_LOGS=output_code python train.py
# 多类日志组合
TORCH_LOGS="inductor,output_code,schedule" python train.py
# 全量调试落盘(graph、fusion、kernel 等到调试目录)
TORCH_COMPILE_DEBUG=1 python train.py
```

`TORCH_LOGS` 取值(`torch/_logging/_registrations.py`):

| 取值 | 含义 | 来源 |
|------|------|------|
| `inductor` | Inductor 全量日志 | `_registrations.py:29` |
| `dynamo` | Dynamo 日志 | `_registrations.py:25` |
| `dynamic` | 动态形状 / overspecialization | `_registrations.py:36` |
| `output_code` | 打印生成的 Triton/C++(默认关) | `_registrations.py:172-177` |
| `kernel_code` | 按 kernel 粒度打印代码 | `_registrations.py:178-183` |
| `schedule` | 调度器信息 | `_registrations.py:184-188` |
| `fusion` | 融合决策(比 schedule 更细) | `_registrations.py:192-196` |
| `perf_hints` | 性能提示(如 CUDA graph 为何不生效) | `_registrations.py:189` |
| `graph_breaks` | graph break 触发点 | `_registrations.py:158-162` |
| `recompiles_verbose` | 重编译时所有失败 guard | `_registrations.py:150-157` |

`TORCH_COMPILE_DEBUG=1` 等价 `torch._inductor.config.trace.enabled=True`(`config.py:2724`),把 FX 图、IR、融合、kernel 等落到调试目录。完整全链路排查(单卡/多卡、CUDA/NPU、配套脚本)见 [[02_compile_stack/07_debugging/index]]。

## 7. 深入阅读导航

- 总览:[[torch_compile_architecture]] — Dynamo → AOTAutograd → Inductor 全景
- compile_fx 编排:[[inductor_compile_fx_orchestration_analysis]] — 为什么先调 AOTAutograd、fw/bw compiler 分工、§0 全链路全景图
- 调度器:[[scheduler_dependency_graph_fusion_and_ordering_analysis]] — 融合算法与调度
- 降级:[[fx_lowering_to_inductor_ir_analysis]] — FX 算子 → Inductor IR
- 动态形状:[[symbolic_shapes_guards_and_graph_reuse_analysis]]
- 调试:[[02_compile_stack/07_debugging/index]]
- NPU 后端:[[02_compile_stack/04_inductor/npu/index]]

## Related Pages

- [[01_ai_frameworks/index]]
- [[torch_compile_architecture]]
- [[inductor_compile_fx_orchestration_analysis]]
- [[scheduler_dependency_graph_fusion_and_ordering_analysis]]
- [[fx_lowering_to_inductor_ir_analysis]]
- [[symbolic_shapes_guards_and_graph_reuse_analysis]]
- [[02_compile_stack/07_debugging/index]]
- [[02_compile_stack/04_inductor/npu/index]]
