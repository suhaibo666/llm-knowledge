# E03 · Guard Failure 与 Recompile 诊断

> 卷别：E · 调试、正确性与性能  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[11_dynamo_explain_and_graph_break_diagnosis_analysis]]  
> 后续：[[13_aotautograd_and_inductor_failure_localization_analysis]]  
> 最后更新：2026-07-28

## 1. 为什么重编译不是“缓存随机失效”

一个 Dynamo cache entry包含 transformed code、guard manager、compile id、backend等。
新frame到来时：

1. 沿同一code object的entry链检查；
2. 某entry全部guards通过则复用；
3. 未命中时为新状态捕获并编译；
4. 新entry加入缓存；
5. 达到限制后按策略失败或转run-only/eager。

因此recompile的直接原因通常是“当前frame不满足既有specialization的契约”，而不是FX
GraphModule自己变了。

## 2. Guard failure reason 是怎样得到的

`get_guard_fail_reason_helper`使用当前`f_locals`、guard manager保存的global/closure scope
执行verbose check。正常日志只报告第一个失败检查
（`torch/_dynamo/guards.py:5355-5383`）。

对于Tensor match或符号shape的组合guard，它会处理verbose code parts；如果打开
`recompiles_verbose`，会继续收集多个失败项，否则找到第一项即停止
（`torch/_dynamo/guards.py:5388-5406` 与
`torch/_dynamo/guards.py:5428-5454`）。

所以普通`recompiles`中的“第一个原因”是entry选择失败的充分证据，但不代表该entry只有
一个guard不满足。

## 3. Reasons 与 cache entries 的对应关系

系统对每个旧cache entry调用失败原因分析，记录到全局`guard_failures`，并可触发用户
`guard_fail_fn` callback
（`torch/_dynamo/guards.py:5467-5494`）。

随后将所有entry的reason组合为recompile日志；verbose模式按entry展开多个失败检查，并
产生结构化 `recompile_reasons` artifact
（`torch/_dynamo/guards.py:5497-5526`、
`torch/_dynamo/guards.py:5527-5527`、
`torch/_dynamo/guards.py:5529-5558` 与
`torch/_dynamo/guards.py:5559-5559`）。

这解释了为什么一次新调用可能打印多条reason：它正在说明“为什么每个历史specialization
都不能用”，不是发生了多次新的重编译。

## 4. 常见 guard 类别与设计原因

| 类别 | 保护的不变量 | 高频根因 |
|---|---|---|
| Tensor metadata | dtype/device/rank/shape/stride/requires-grad | batch、layout、dtype漂移 |
| Symbolic shape | size关系、范围、整除或分支条件 | 未泛化、约束过强 |
| Python value | int/bool/string/enum/容器长度 | 配置或循环状态改变 |
| Identity | object/function/module实例身份 | 每次构造新对象/closure |
| Global/module state | training、hook、属性、全局函数 | 运行中切mode或monkey patch |
| Alias | 输入间是否为同一Tensor | batch组装方式改变 |
| Dispatch state | mode、grad、autocast、determinism | 上下文跨调用漂移 |
| Backend match | backend callable身份 | 每次创建新partial/lambda |

backend不一致也会作为失败原因报告；源码会描述缓存和新backend callable，并建议复用
callable或稳定的partial（`torch/_dynamo/guards.py:5407-5415`）。

## 5. 诊断 recompile storm 的顺序

### 5.1 建立调用序列

记录每次调用的：

- 输入shape/stride/dtype/device；
- requires-grad与alias；
- module training/eval；
- autocast、grad、dispatch mode；
- 标量/配置值；
- code object与compile id。

没有时间序列，只能知道“某guard失败”，不能知道状态从何时开始漂移。

### 5.2 开普通 recompiles

先用第一失败原因分类：

- shape mismatch；
- value/identity变化；
- global/module状态变化；
- backend变化；
- alias变化。

### 5.3 必要时开 verbose

`recompiles_verbose`会收集更完整的失败检查，但注册说明明确指出：运行时entry检查通常在
首个失败处停止，verbose输出中并非每个失败检查都由真实热路径执行。该语义见
`torch/_logging/_registrations.py:146-156`。

### 5.4 使用 fail-fast

`fail_on_recompile` stance在本应命中cache时直接构造错误，读取已有entry的reason并附到
异常中（`torch/_dynamo/eval_frame.py:300-329`、
`torch/_dynamo/eval_frame.py:330-330` 与
`torch/_dynamo/eval_frame.py:331-343`）。

它适合CI不变量，不适合需要自然适配多shape的生产流量。

## 6. 动态shape不是“关闭shape guards”

动态化的目标是用符号size和约束覆盖一组输入，而不是不检查shape。若控制流、stride、
rank、dtype或某个shape关系不同，仍可能需要新specialization。

处理shape重编译要区分：

- 同一范围应泛化但没有泛化；
- 不同rank/layout本来就需要不同图；
- Python从Tensor读取scalar导致值specialization；
- shape触发不同控制流/算法；
- 用户手工mark dynamic与实际范围矛盾。

盲目把所有维度设为dynamic可能扩大编译和kernel泛化成本。

## 7. Cache limit 的双层语义

计数结构同时保存：

- 当前region的entry数；
- 具有相同ID_MATCH对象的entry数；
- 该code object所有region累计entry数。

见 `torch/_dynamo/cache_size.py:72-101` 与
`torch/_dynamo/cache_size.py:102-106`。

`compute_cache_size`从当前region entries计算前两者，累计值由调用方传入
（`torch/_dynamo/cache_size.py:142-160`）。因此：

- `recompile_limit`面向某region/相同identity上下文；
- `accumulated_recompile_limit`是跨region全局安全上限；
- 限制是防止无界编译成本，不是优化目标。

达到限制时，convert_frame记录最后原因和限制类型；`fullgraph`或fail配置会硬失败，否则
把frame执行策略改为RUN_ONLY
（`torch/_dynamo/convert_frame.py:2000-2029`、
`torch/_dynamo/convert_frame.py:2030-2053`）。

仅提高上限会延迟症状，不能修复状态空间爆炸。

## 8. 修复策略

| 根因 | 优先修复 |
|---|---|
| 有限稳定shape集合 | 允许少量specialization并预热 |
| 本应连续的batch维 | 修正dynamic range/shape依赖控制流 |
| 每次新Python对象 | 将稳定配置移出热路径或复用对象 |
| backend callable每次变化 | 复用backend实例/partial |
| train/eval频繁切换 | 分离compiled callable或稳定调用阶段 |
| autocast/grad模式漂移 | 在明确边界建立不同specialization |
| 输入alias模式变化 | 规范调用契约，或接受独立specialization |
| 不受控用户流量shape | 分桶、上限、eager fallback与监控 |

## 9. 复杂度

若一个code object有 $S$ 个entry，朴素地对entry逐个guard检查，miss诊断至少与访问的
entry数和guard工作量相关。记entry $i$ 的检查成本为 $g_i$：

$$
T_{\text{lookup-miss}} = O\left(\sum_{i=1}^{S} g_i\right)
\ T_{\text{capture+compile}}
$$

verbose诊断会重新获取失败细节，成本高于正常first-failure路径。若输入状态形成笛卡尔积：

$$
S \lesssim
\lvert \text{shape classes}\rvert\cdot\lvert \text{dtype/device}\rvert
\cdot\lvert \text{Python states}\rvert\cdot\lvert \text{dispatch states}\rvert
$$

真正目标是缩小有效specialization空间，而不是只优化一次guard执行。

## 10. 验收不变量

- warmup完成后关键code object不再产生意外compile id；
- 输入契约允许的shape/stride范围命中有限entry；
- 单请求编译时间和累计entry有上限；
- 失败原因按类别聚合，而不是只统计recompile总数；
- limit hit可观测并有明确fallback；
- 不同rank/worker不会因初始化差异产生无界specialization；
- 修复后比较正确性和steady-state，不只看日志消失。

## 11. 常见误解

- **“一次日志有十条reason就是重编译十次。”** 通常是十个旧entry分别未命中。
- **“动态shape没有shape guard。”** 动态图仍有符号约束。
- **“提高recompile limit能修性能。”** 可能只允许更多昂贵编译。
- **“guard失败是Inductor问题。”** 它发生在Dynamo cache entry选择阶段。
- **“普通recompiles列出了全部失败guard。”** 默认只取每entry首个关键失败。

## 配套 Demo

本页对应卷级入口 `tools/labs_torch_compile/demo_e_diagnostics.py` 的 `guard_failure` 用例。默认以 CUDA 为验收设备：

```powershell
python -B tools\labs_torch_compile\demo_e_diagnostics.py `
  --case guard_failure --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\e03
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `guard_failure/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[courses/torch_compile_end_to_end]]
- [[15_guards_cache_lookup_and_recompilation_analysis]]
- [[17_dynamic_shapes_generalization_and_fallback_analysis]]
- [[10_observability_logs_counters_and_artifact_map_analysis]]
- [[11_dynamo_explain_and_graph_break_diagnosis_analysis]]
- [[17_compile_latency_cache_and_steady_state_performance_analysis]]
- [[19_production_rollout_fallback_and_monitoring_analysis]]
