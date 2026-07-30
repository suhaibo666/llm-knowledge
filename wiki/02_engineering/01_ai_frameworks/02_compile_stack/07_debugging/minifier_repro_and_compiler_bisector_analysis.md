# E05 · Minifier、Repro 与 Compiler Bisector

> 卷别：E · 调试、正确性与性能  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[aotautograd_and_inductor_failure_localization_analysis]]  
> 后续：[[compiled_correctness_validation_methodology_analysis]]  
> 最后更新：2026-07-28

## 1. 三个工具解决的不是同一问题

- **Repro**：把失败状态序列化成可独立运行的程序；
- **Minifier**：在保持失败predicate为真的前提下缩小FX图和输入；
- **Compiler bisector**：保持程序不变，逐层切换backend/subsystem或二分某子系统的第 \(k\) 次
  应用。

它们对应三个正交轴：

```text
环境可重放性 × 程序/图规模 × 编译器功能开关/应用序号
```

先有稳定repro，再做minify或bisect；否则任何“缩小成功”都可能只是偶然状态变化。

## 2. Repro 必须保存哪些状态

一个高保真repro至少需要：

- GraphModule code与必要submodule/constants；
- 输入shape、stride、dtype、device、requires-grad和alias；
- 随机状态与determinism；
- compiler/backend/config和环境变量；
- forward-only或forward+backward；
- 异常或accuracy predicate；
- 分布式world/rank/collective条件；
- 复现依赖的custom op注册与toolchain。

after-AOT生成器会构造独立FX脚本，加入环境、配置、Tensor构造和必要的distributed/Triton
imports（`torch/_dynamo/repro/after_aot.py:528-557` 与
`torch/_dynamo/repro/after_aot.py:559-575`）。

生成脚本不等于自动包含所有外部状态；文件、网络、全局单例和自定义extension仍需显式处理。

## 3. 为什么 after-Dynamo 与 after-AOT 都存在

### after-Dynamo

保存backend收到的 Dynamo FX，适合：

- Dynamo FX本身错误；
- backend入口即失败；
- 需要保留Dynamo region边界。

`dump_to_minify_after_dynamo`把图、参数、backend名和accuracy模式写入launcher
（`torch/_dynamo/repro/after_dynamo.py:300-316`）。

### after-AOT

保存lift parameters后的fw或bw图，适合：

- Inductor compiler失败；
- 单独最小化forward/backward；
- 保存更规则的Tensor/SymInt输入ABI。

`dump_compiler_graph_state`按node数建立checkpoint并复制最新`repro.py`
（`torch/_dynamo/repro/after_aot.py:917-943`）；`dump_to_minify`则生成带`minify`命令的launcher
（`torch/_dynamo/repro/after_aot.py:951-960`）。

选择错误层级会丢失关键变换或携带大量无关上游状态。

## 4. Minifier 的契约

核心函数接收：

- failing `GraphModule`；
- 与placeholder对齐的inputs；
- `module_fails(gm, inputs) -> bool`；
- 每次缩减后的dump callback。

它先做concrete propagation和sanity check，确保原始图确实满足失败predicate
（`torch/_functorch/fx_minifier.py:195-224`、
`torch/_functorch/fx_minifier.py:225-225` 与
`torch/_functorch/fx_minifier.py:228-250`）。

最重要的不变量是：

\[
\operatorname{predicate}(G_i, I_i)=\text{True}
\]

图更小但失败类型改变，不是有效最小化。

## 5. Minifier 如何缩图

源码概括两大主策略：

1. 截断suffix并选择新output；
2. delta debugging，把部分node替换成新的placeholder。

定义见 `torch/_functorch/fx_minifier.py:207-220`。

实际循环还会：

- 移除outputs；
- 消除dead code；
- 移除未使用inputs；
- 整理placeholder到图首；
- 尝试不同granularity。

未使用input和DCE策略见 `torch/_functorch/fx_minifier.py:382-411` 与
`torch/_functorch/fx_minifier.py:412-422`；delta debugging把候选node
变placeholder、DCE、整理输入后再次验证失败
（`torch/_functorch/fx_minifier.py:454-478`）；策略顺序见
`torch/_functorch/fx_minifier.py:491-517`。

## 6. 为什么替换为 placeholder 而不是直接删除

直接删除一个中间node会让其users失去定义。把它变成placeholder相当于：

- 切断上游子图；
- 把该中间值物化为新输入；
- 保留下游失败路径；
- 让DCE删除不再可达的上游。

这会缩小图，但也可能降低保真度：新输入跳过了producer的alias、layout、mutation或device
行为。因此每次候选都必须重跑predicate。

## 7. Compiler Bisector 的设计

bisector先在backend阶梯（如eager、aot_eager、inductor）定位首个失败系统，再：

- 完全禁用某subsystem判断问题是否消失；
- 对于重复应用的subsystem，寻找最大界限；
- 二分到第几次pass/lowering应用触发问题。

类的设计目标和CLI见 `torch/_inductor/compiler_bisector.py:98-123`。

`disable_subsystem`读取backend、subsystem和run state；`test_disable`先全关，
`find_max_bounds`建立范围，`bisect`按midpoint选择是否禁用后续应用
（`torch/_inductor/compiler_bisector.py:378-407` 与
`torch/_inductor/compiler_bisector.py:411-428`）。

最终循环先定位backend，再推进subsystem并返回backend、subsystem、bisect number和debug info
（`torch/_inductor/compiler_bisector.py:614-643`、
`torch/_inductor/compiler_bisector.py:644-644`、
`torch/_inductor/compiler_bisector.py:646-675` 与
`torch/_inductor/compiler_bisector.py:676-677`）。

## 8. 何时用 minifier，何时用 bisector

| 症状 | 优先 |
|---|---|
| 大图中某少量op组合失败 | minifier |
| 新配置/某pass引入回归 | bisector |
| 不知是Dynamo/AOT/Inductor | backend阶梯/bisector |
| 失败依赖输入layout或alias | 高保真repro后谨慎minify |
| 只在第N个lowering/pass触发 | bisector |
| 分布式时序/collective hang | 先做多进程repro；普通FX minifier可能破坏时序 |

两者可组合：bisect到subsystem后，再对该配置下的图minify。

## 9. 非确定性与失败漂移

以下情况会让predicate不稳定：

- 未固定随机源；
- 异步设备错误在后续同步点才抛出；
- autotune选择变化；
- cache冷热变化；
- allocator地址或CUDAGraph状态；
- 数据竞争/分布式时序；
- 浮点误差刚好跨容差；
- 缩图后换成另一个更早异常。

可采用重复 \(r\) 次、要求至少 \(k\) 次失败的统计predicate，但查询成本近似乘以 \(r\)。
对hang需要外部timeout与进程隔离，不能让minifier本身永久阻塞。

## 10. 复杂度

设图node数为 \(V\)，单次predicate成本为 \(T_p\)：

- 理想二分式缩减查询数接近 \(O(\log V)\)，但策略回退和依赖约束可显著增加；
- 每次要复制、lint、DCE并运行图，总成本约 \(Q(T_p+O(V))\)；
- bisector对可排序的 \(M\) 次应用，二分部分约 \(O(\log M)\) 次程序运行；
- 跨多个backend/subsystem还要加线性探测成本。

最大成本通常来自编译/运行predicate，而不是图数据结构操作。

## 11. 交付一个有效 repro 的标准

- 干净进程可复现；
- 无私有数据和不必要依赖；
- 明确版本、device、命令；
- 明确预期与实际；
- 异常类型/消息或accuracy predicate稳定；
- 说明forward/backward和cache状态；
- 最小输入仍保留shape/stride/alias；
- 若已bisect，附backend/subsystem/第N次应用；
- 原始和最小repro都保留，防止最小化失真。

## 12. 常见误解

- **“repro.py必然完全自包含。”** 外部custom op、toolchain和分布式状态可能仍缺失。
- **“node最少就是最好repro。”** 失去真实layout/alias后可能不再代表原问题。
- **“minifier找到的任何异常都算成功。”** predicate必须锁定原始失败。
- **“bisector在Git提交间二分。”** 这里主要二分编译backend/subsystem及其应用序号。
- **“accuracy bug可用异常predicate最小化。”** 两类predicate语义不同。

## 配套 Demo

本页对应卷级入口 `tools/labs_torch_compile/demo_e_diagnostics.py` 的 `minifier_repro` 用例。默认以 CUDA 为验收设备：

```powershell
python -B tools\labs_torch_compile\demo_e_diagnostics.py `
  --case minifier_repro --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\e05
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `minifier_repro/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[aotautograd_and_inductor_failure_localization_analysis]]
- [[compiled_correctness_validation_methodology_analysis]]
- [[kernel_fusion_memory_and_hardware_performance_analysis]]
- [[graph_rewrite_legality_validation_and_complexity_analysis]]
- [[backend_modes_options_stances_and_fullgraph_analysis]] — §13:`torch.compile()` API 入口如何调用 `CompilerBisector.get_backend()` 覆盖 backend
