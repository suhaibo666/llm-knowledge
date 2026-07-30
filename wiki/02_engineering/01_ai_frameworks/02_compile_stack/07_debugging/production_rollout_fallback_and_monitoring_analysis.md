# E09 · Production Rollout、Fallback 与 Monitoring

> 卷别：E · 调试、正确性与性能  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[kernel_fusion_memory_and_hardware_performance_analysis]]  
> 后续：[[f01_compiled_autograd_analysis]]  
> 最后更新：2026-07-28

## 1. 上线的目标不是“尽量不报错”

生产契约必须同时定义：

- 正确性：值、grad、mutation、alias、effect；
- 可用性：编译/加载/runtime失败如何降级；
- 延迟：cold、new specialization和steady SLO；
- 资源：编译CPU、disk cache、host/device memory；
- 状态空间：允许多少shape/config specialization；
- 可观测性：能定位到frame、compile id、阶段和artifact；
- 回滚：不重启或最小影响地切回eager/旧版本。

“打开 `suppress_errors` 后请求成功”只满足部分可用性，不证明编译路径正确或有收益。

## 2. Fallback 不是单一动作

| 位置 | 可能动作 | 仍保留什么 |
|---|---|---|
| graph break | 当前边界Python执行，后续可继续捕获 | 其他compiled regions |
| guard miss | 选其他entry或新编译 | 已有cache entries |
| recompile limit | 当前frame转run-only/eager | 其他frame/region |
| Dynamo内部错误抑制 | eager fallback | 上层程序继续 |
| CUDAGraph不适用 | 普通compiled callable | Inductor kernels |
| artifact load失败 | miss/rebuild或上层失败 | 可用的其他cache |
| feature flag/stance | force eager | 原compiled wrapper仍可保留 |

监控必须区分这些层级，不能只用一个“fallback count”。

## 3. `suppress_errors`的真实语义

配置说明将其定义为：Dynamo内部错误时强制fallback到eager，可能失去优化机会，并明确要求
开发者调查benchmark failure
（`torch/_dynamo/config.py:264-273`）。

因此：

- 它是可用性策略，不是错误修复；
- 不能掩盖eager本身错误；
- 不保证AOT/Inductor所有runtime错误都能安全转换；
- silent wrong answer不能靠fallback自动检测；
- 上线仍需记录被抑制的异常与流量比例。

## 4. Stance 是运行时控制面

公开接口支持：

- `default`；
- `force_eager`；
- `eager_on_recompile`；
- `fail_on_recompile`；
- `eager_then_compile`；
- `aot_eager_then_compile`。

其公开语义见 `torch/compiler/__init__.py:389-418`、
`torch/compiler/__init__.py:419-421` 与
`torch/compiler/__init__.py:426-441`。

eval-frame侧的实际选择是：

- `force_eager`返回`None`禁用callback；
- `eager_on_recompile`返回`False`进入run mode；
- fail stance包装callback并在检测到recompile时抛错；
- then-compile使用延迟编译callback。

见 `torch/_dynamo/eval_frame.py:279-303`。

这些能力适合构建kill switch、warmup策略和CI防回归，但需验证线程/请求作用域与业务调用
方式，不应假设所有全局状态都天然租户隔离。

## 5. 不安全 guard 优化为何不应作为常规上线手段

`skip_guard_eval_unsafe`允许只运行区分entry的guards以降低guard overhead，但公开文档明确
要求已经用足够多输入warmup、此后不再需要recompile；若假设不成立，可能静默产生错误结果
（`torch/compiler/__init__.py:444-457`）。

这不是普通性能开关，而是把一部分正确性证明责任转移给部署系统。没有形式化输入契约、
流量封闭性和强回归验证时不应开启。

## 6. 分阶段 rollout

### Stage 0：离线契约

- backend阶梯正确性；
- 输入类/shape/alias矩阵；
- cold/hit/steady性能；
- 失败注入与fallback；
- artifact可复现。

### Stage 1：shadow

eager返回真实结果，compiled旁路运行并比较；限制资源与采样率。关注silent wrong answer和
编译资源，而非用户延迟收益。

### Stage 2：小流量 canary

按模型版本、device、shape bucket、worker/rank隔离。设置错误率、重编译、cold latency、
memory和正确性自动阈值。

### Stage 3：分批扩大

逐步扩大流量与shape范围，观察长尾输入是否产生specialization explosion或cache压力。

### Stage 4：稳态与升级

固定artifact/cache版本策略；升级PyTorch、driver、Triton或config时重新执行canary，不把
旧cache hit视为兼容性证明。

## 7. 必须监控的指标

### Capture/cache

- compile attempts/success/failure；
- graph breaks按reason；
- recompile按guard category；
- 每code object entry数与limit hit；
- Dynamo/AOT/FX/Triton cache hit/miss；
- artifact load/rebuild。

### Latency

- cold compile；
- new specialization；
- persistent hit load；
- first forward/first backward；
- steady p50/p95/p99；
- runtime autotune/CUDAGraph时间。

### Correctness

- shadow mismatch；
- NaN/Inf；
- gradient/parameter drift；
- exception divergence；
- fallback后结果。

### Resources

- 编译CPU/worker queue；
- cache disk/IO；
- host RSS；
- device allocated/reserved/CUDAGraph pools；
- kernel/communication utilization。

`CompilationMetrics`已有fail type/reason、guard/cache、forward/backward/runtime及多个阶段耗时
字段，可作为事件骨架
（`torch/_dynamo/utils.py:1573-1602`、
`torch/_dynamo/utils.py:1603-1605`、
`torch/_dynamo/utils.py:1613-1642` 与
`torch/_dynamo/utils.py:1643-1649`）。

## 8. SLO 与自动回退

建议按“错误预算”而不是单次异常设计：

- correctness mismatch：通常立即切断compiled路径；
- compile failure rate：超过阈值降级该模型/shape bucket；
- recompile storm：转eager_on_recompile或封禁长尾shape；
- cold latency：预热、预编译、持久化cache或异步隔离；
- memory pressure：禁用高内存mode/CUDAGraph或缩小并发；
- load/ABI failure：隔离artifact版本并重建。

回退粒度越小越好：请求、shape bucket、模型版本、worker、device，而不是整个集群无条件关闭。

## 9. Cache 与发布

发布键至少应包含：

- PyTorch/compiler commit；
- backend与配置；
- Python/ABI；
- device capability、driver/toolchain；
- 模型/常量版本；
- shape/dtype/layout specialization；
- 安全/租户域。

`torch.compile`文档说明cache关联code object，guard miss会产生多个compiled result，超过
recompile limit后fallback
（`torch/__init__.py:3152-3166`）。`isolate_recompiles`还能为不同compile调用维护独立region
entries，但累计上限仍是全局安全cap
（`torch/__init__.py:3237-3257`）。

发布系统不能只用“模型文件hash”作为所有cache层的完整兼容键。

## 10. 回滚演练

必须在上线前验证：

- 进程内切换force eager；
- 新请求不再触发compile；
- 已在执行的请求语义明确；
- 清理/保留cache不会误加载不兼容artifact；
- distributed所有rank一致切换；
- lazy backward/async compile future不会遗留资源；
- 回滚后指标能区分eager与compiled流量；
- 恢复compiled路径需要重新warmup还是可复用cache。

## 11. 容量与复杂度

设模型版本 \(M\)、设备类 \(D\)、specialization \(S\)、部署副本 \(R\)：

\[
|\text{artifact states}|=O(MDS)
\]

若每副本独立持有runtime module/CUDAGraph pool，运行时内存近似再乘 \(R\)；remote cache可减少
重复编译与disk内容，但不能共享进程内live handles。

监控label也要控制基数；不要直接把完整guard字符串、shape或stack作为无限基数label，应作为
采样artifact保存。

## 12. 常见误解

- **“suppress_errors保证任何失败都回eager。”** 它不是所有runtime failure的总兜底。
- **“fallback率低就说明上线成功。”** silent wrong answer和无收益仍可能存在。
- **“预热一组输入后可安全跳过guards。”** 只有封闭输入契约才能承担该风险。
- **“remote cache能共享CUDAGraph和loaded module。”** live runtime状态属于进程/device。
- **“回滚只需改一个flag。”** async、rank一致性、cache和正在执行的调用都需演练。

## 配套 Demo

本页对应卷级入口 `tools/labs_torch_compile/demo_e_diagnostics.py` 的 `rollout_fallback` 用例。默认以 CUDA 为验收设备：

```powershell
python -B tools\labs_torch_compile\demo_e_diagnostics.py `
  --case rollout_fallback --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\e09
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `rollout_fallback/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[guard_failure_and_recompile_diagnosis_analysis]]
- [[compile_latency_cache_and_steady_state_performance_analysis]]
- [[kernel_fusion_memory_and_hardware_performance_analysis]]
- [[compiled_artifact_lifecycle_and_runtime_failures_analysis]]
- [[f01_compiled_autograd_analysis]]
- [[f08_training_inference_cudagraph_and_freezing_analysis]]
