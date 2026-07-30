# E06 · Compiled Correctness 验证方法论

> 卷别：E · 调试、正确性与性能  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[e05_minifier_repro_and_compiler_bisector_analysis]]  
> 后续：[[e07_compile_latency_cache_and_steady_state_performance_analysis]]  
> 最后更新：2026-07-28

## 1. “输出 allclose”为什么不够

编译正确性至少有六个维度：

1. forward值与结构；
2. backward梯度；
3. 输入、参数、buffer和全局状态的mutation；
4. view/alias/storage identity；
5. exception、RNG、hook、collective等effect和顺序；
6. 跨调用、不同specialization和cache状态的稳定性。

只比较一次forward输出，会漏掉训练状态更新、in-place语义、saved tensor、lazy backward以及
第二次调用才暴露的复用问题。

## 2. Reference 不是“同一对象再跑一次”

eager和compiled两路必须从等价但独立的状态开始：

- 深拷贝可复制的module/optimizer state；
- 克隆输入并保留requires-grad；
- 明确共享alias关系，不能把本应alias的输入独立clone；
- 重置RNG；
- 清grad；
- 控制train/eval、autocast、grad、determinism；
- 同步异步device后再读取结果；
- 避免第一路mutation污染第二路。

源码的debug helper会复制GraphModule、克隆输入并恢复gradness；若输出需要backward，则把
输出归约为scalar loss再收集结果
（`torch/_dynamo/debug_utils.py:586-615` 与
`torch/_dynamo/debug_utils.py:617-628`）。

## 3. 比较对象矩阵

| 对象 | 比较内容 |
|---|---|
| 输出 pytree | 结构、Tensor metadata、数值、非Tensor值 |
| 输入 | mutation后的值、metadata、storage/alias |
| Parameter | 值、grad、requires-grad |
| Buffer | 值与更新次数 |
| optimizer | state dict、step、参数更新 |
| exception | 类型、发生阶段、必要消息语义 |
| RNG | 后续随机序列或显式state |
| hooks/effects | 次数、顺序、payload |
| distributed | 每rank值、collective序列、全局结果 |

比较alias时，应构建“输出/输入对之间是否共享storage、offset/stride关系”的矩阵，而不只比较
Tensor值。

## 4. 容差如何设计

对浮点Tensor通常使用：

\[
|x-y| \le \text{atol} + \text{rtol}\cdot |y|
\]

但容差应按dtype、算子数值性质和业务误差预算定义：

- fp64/fp32/fp16/bf16不能使用同一容差；
- reduction顺序改变会产生可预期误差；
- NaN相等是否允许要显式决定；
- 整数、布尔、shape和索引通常要求精确；
- 容差不能随模型规模任意放大到掩盖错误。

`same_two_models`先运行eager reference，并可生成fp64 reference辅助判断低精度误差，再用
配置的tolerance比较（`torch/_dynamo/debug_utils.py:631-658` 与
`torch/_dynamo/debug_utils.py:666-686`）。

fp64 reference是辅助基准，不代表真实低精度程序必须逐位等于fp64。

## 5. Forward、Backward 与 Optimizer 的顺序

训练step应至少比较：

```text
initial state
→ forward outputs/loss
→ backward input/parameter grads
→ optimizer state transition
→ updated parameters/buffers
→ next iteration output
```

下一迭代很重要：某些错误只在mutation后的state、lazy backward compile或cache复用时出现。

Inductor pass numeric check会对pass前后GraphModule重复运行，比较parameters、forward
outputs和gradients
（`torch/_inductor/fx_passes/numeric_utils.py:43-72`、
`torch/_inductor/fx_passes/numeric_utils.py:73-74`、
`torch/_inductor/fx_passes/numeric_utils.py:77-104` 与
`torch/_inductor/fx_passes/numeric_utils.py:107-130`）。

它还可执行optimizer step后比较parameter
（`torch/_inductor/fx_passes/numeric_utils.py:133-159` 与
`torch/_inductor/fx_passes/numeric_utils.py:160-180`）。

## 6. Mutation 与 Alias 的专门验收

需要为每个可变对象记录：

- 调用前值和version；
- 调用后值和version；
- 是否仍是view；
- base/storage identity；
- storage offset、size、stride；
- 相互alias是否保持；
- mutation发生次数和可见顺序。

常见错误包括：

- functionalization返回更新值但wrapper未写回；
- 输出值正确但view被materialize成copy；
- 输入alias模式特殊化错误；
- buffer mutation被DCE或重排；
- saved tensor在反向前被不当覆盖。

这类错误不一定造成首轮数值差异。

## 7. Effect 与异常语义

对有effect的程序，比较：

- print/log/callback/hook是否发生；
- 发生次数与相对顺序；
- RNG消耗；
- collective调用顺序；
- 异常是在调用前、部分mutation后还是调用后抛出；
- 异常后对象是否保持一致状态。

编译器通常不保证所有错误消息逐字符一致，但异常类别和用户可观察状态不能被无意改变。
生产验收应提前定义哪些effect在compile region内允许。

## 8. 多输入与跨状态验证

测试集合应覆盖：

- 最小/典型/最大shape；
- 动态范围边界和越界；
- contiguous与代表性stride；
- 不同dtype/autocast；
- requires-grad组合；
- alias与非alias；
- train/eval；
- 空Tensor、零维、极端值、NaN/Inf；
- 首次编译、cache hit、recompile后的新entry；
- forward-only与forward+backward；
- 单进程与多rank。

不能用一次示例输入证明guard覆盖范围内所有输入正确；至少要按约束边界设计代表类。

## 源码跟读：PyTorch 的调试比较器到底比较了什么

### 1. reference run为什么必须克隆 GraphModule 与输入

`clone_inputs_retaining_gradness`先克隆输入，再逐 Tensor 恢复原
`requires_grad`（`torch/_dynamo/debug_utils.py:586-596`）。`run_fwd_maybe_bwd`又
deepcopy GraphModule，默认克隆参数，清零 module grads，运行 forward；输出需要 backward
时把结构归约为 scalar loss 后调用 backward，最后收集模型、输出与输入相关结果
（`torch/_dynamo/debug_utils.py:599-628`）。

这套所有权隔离防止 reference 先执行时的 mutation、grad accumulation 或 module state
直接污染 optimized run。`disable_clone=True`只在追求更高复现 fidelity、并接受输入可能被
破坏时使用，不能成为常规正确性基线。

### 2. `same_two_models`不是简单的 output allclose

调用链是：

```mermaid
flowchart LR
    GM["reference GraphModule"] --> RefRun["clone + forward + optional backward"]
    GM --> FP64["optional fp64 deepcopy/reference"]
    Opt["optimized GraphModule"] --> OptRun["独立 clone + forward + optional backward"]
    RefRun --> Compare["utils.same<br/>nested outputs / grads / tolerance"]
    FP64 --> Compare
    OptRun --> Compare
```

函数先运行 reference；若开启配置则尝试把 model 与 inputs 转为 fp64 再得到第三份 reference
（`torch/_dynamo/debug_utils.py:631-658`）。随后独立运行 optimized model，把 ref、result
与可选 fp64_ref 交给 `same`，使用 repro tolerance、`equal_nan=True`等策略
（`torch/_dynamo/debug_utils.py:666-686`）。

这里有一个容易误用的机制：optimized run 抛异常时，`same_two_models`在 accuracy minifier
语境中记录异常并返回 `True`，意思是“这不是当前要保留的 accuracy failure”，不是“运行
正确”。`backend_accuracy_fails`同样在 compiler/比较异常时返回 `False`
（`torch/_dynamo/debug_utils.py:739-768`）。所以这些返回值是**minifier predicate**，不能
直接当生产验收布尔值；异常必须由外层单独记录为 FAIL。

### 3. FX pass numeric checker覆盖的状态比名字看起来更多，也更有限

numeric checker固定 Python、NumPy、Torch seed并打开 deterministic algorithms
（`torch/_inductor/fx_passes/numeric_utils.py:19-35`）。它分别比较：

- 字典 key 与非 `None`参数/梯度 Tensor
  （`torch/_inductor/fx_passes/numeric_utils.py:43-61`;
  `torch/_inductor/fx_passes/numeric_utils.py:62-74`）；
- tuple output长度与逐项 Tensor
  （`torch/_inductor/fx_passes/numeric_utils.py:77-104`）；
- named parameters、forward output 与 parameter gradients
  （`torch/_inductor/fx_passes/numeric_utils.py:107-130`）。

`run_model`在每轮为 reference/control分别重置随机状态，比较 forward 前后的 parameters 与
outputs，再尝试 backward 与 gradient 比较
（`torch/_inductor/fx_passes/numeric_utils.py:133-159`）；配置要求时才额外执行 SGD step 并
比较参数（`torch/_inductor/fx_passes/numeric_utils.py:160-180`）。

这解释了它的边界：它适合 pass 前后数值回归，不自动证明 alias identity、storage offset、
异常类型、I/O/RNG次数、optimizer state dict 或分布式 collective 语义。那些必须按本篇的
mutation/alias/effect/optimizer矩阵另建断言。

### 4. 正确性 harness的失败语义必须比比较器更严格

一个可交付 harness 至少需要把三种结果分开：

1. reference 或 setup 本身失败：测试无效，标记 harness/setup FAIL；
2. compiled callable 抛异常：runtime FAIL；
3. 两边都完成但值、grad、alias/effect不一致：correctness FAIL。

不能沿用 minifier 为缩图服务的“异常不属于 accuracy predicate，所以返回非失败”语义。
同一底层 helper在不同 caller 中有不同返回值含义，正是这里需要源码跟读而不能只看函数名的
原因。

## 9. 分层验证与定位

```mermaid
flowchart LR
    E["Eager"] --> D["Dynamo FX eager"]
    D --> A["AOT eager"]
    A --> I["Inductor"]
    I --> C["CUDAGraph / cache replay"]
```

每相邻两层比较，可以把差异收敛到：

- capture/side effects；
- AOT functionalization/partition/runtime；
- Inductor passes/codegen；
- runtime wrapper、cache或CUDAGraph。

`backend_accuracy_fails`会复制图与输入、编译候选并调用`same_two_models`；若候选出现不同
runtime异常，则不把它判成原accuracy failure
（`torch/_dynamo/debug_utils.py:739-768`）。

## 10. 确定性与重复性

numeric check helper设置torch/Python/NumPy seed，并请求deterministic algorithms
（`torch/_inductor/fx_passes/numeric_utils.py:19-35`）。

但确定性测试仍需考虑：

- 某些算子没有deterministic实现；
- 并行reduction次序变化；
- device异步错误；
- 分布式通信时序；
- allocator地址影响；
- autotune与CUDAGraph warmup。

建议把“严格确定性验证”和“统计误差分布验证”分开，不用一次测试兼任二者。

## 11. 复杂度与预算

若有 \(M\) 个输入类、\(S\) 个状态组合、重复 \(R\) 次，每次含forward/backward/optimizer
成本 \(T\)：

\[
T_{\text{validation}}=O(MSRT)
\]

完整笛卡尔积通常不可行，应基于风险做pairwise/边界覆盖，并为alias、mutation、distributed
等高风险机制保留专门用例。CI保留短而稳定的契约集，离线回归运行更广矩阵。

## 12. 验收记录

每个结果应记录：

- reference与candidate层级；
- 版本/config/device；
- 输入类和状态；
- 比较字段与容差；
- 是否发生graph break/recompile/cache hit；
- cold/warm/steady调用序号；
- 失败artifact和最小repro；
- 已知数值差异的批准依据。

## 13. 常见误解

- **“allclose就是正确。”** alias、mutation、grad和effect可能仍错误。
- **“同一module先eager再compiled即可。”** eager可能已修改state。
- **“fp64一定是真理。”** 它是辅助reference，也可能不可构造或改变算法路径。
- **“无异常等于正确。”** silent wrong answer正是编译器最危险的故障。
- **“单次forward通过即可上线训练。”** backward、optimizer和下一step必须验证。

## 配套 Demo

本页对应卷级入口 `tools/labs_torch_compile/demo_e_diagnostics.py` 的 `correctness_validation` 用例。默认以 CUDA 为验收设备：

```powershell
python -B tools\labs_torch_compile\demo_e_diagnostics.py `
  --case correctness_validation --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\e06
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `correctness_validation/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[e04_aotautograd_and_inductor_failure_localization_analysis]]
- [[e05_minifier_repro_and_compiler_bisector_analysis]]
- [[e07_compile_latency_cache_and_steady_state_performance_analysis]]
- [[19_torch_compile_end_to_end/05_graph_effects_alias_mutation_and_order]]
- [[19_torch_compile_end_to_end/16_graph_rewrite_legality_validation_and_complexity]]
