# F01 · Compiled Autograd：运行时反向图捕获

> 卷别：F · 训练、分布式、扩展与部署  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[e09_production_rollout_fallback_and_monitoring_analysis]]  
> 后续：[[f02_activation_checkpoint_recompute_and_compile_analysis]]  
> 最后更新：2026-07-28

## 1. 它为什么不是 AOTAutograd 的别名

AOTAutograd从一个forward callable和示例输入出发，在调用forward的编译阶段构造joint
fw/bw图并分区。Compiled Autograd则在真实backward由C++ autograd engine调度时，捕获更大
范围的autograd node、hooks和grad accumulation，再把这段运行时反向工作转为FX图。

所以二者的时间与边界不同：

| 机制 | 何时建图 | 起点 | 主要边界 |
|---|---|---|---|
| AOTAutograd | forward编译期间 | 可微forward | 单个Dynamo region的fw/bw |
| Compiled Autograd | backward首次运行/缓存miss | autograd engine任务 | 实际engine调度的反向图、hooks、accumulate-grad |

Compiled Autograd还可把AOT产生的lazy backward图复制到更大的CA图中；这不是两套反向图
互不相关，而是局部AOT backward成为整体runtime backward的一部分。

## 2. C++ engine 与 Python tracer 的握手

C++入口负责：

-从autograd engine收集inputs、sizes、scalars、hooks和节点拓扑；
-查Compiled Autograd cache；
-cache miss时调用Python compiler instance的`begin_capture`；
-逐个代理autograd node/hook/accumulate-grad；
-调用`end_capture`取得 `(runtime_wrapper, compiled_fn)`；
-缓存并执行。

C++实现的capture入口与Python callback位置见
`torch/csrc/dynamo/python_compiled_autograd.cpp:878-907`、
`torch/csrc/dynamo/python_compiled_autograd.cpp:974-1003`；end-capture返回值契约见
`torch/csrc/dynamo/python_compiled_autograd.cpp:1135-1157`。

这解释了为什么CA graph不是由Python正向源码直接symbolic trace得到：driver是autograd
engine。

## 3. `AutogradCompilerInstance`持有什么

每个实例拥有：

-compiler callback；
-独立`ShapeEnv`；
-允许fallback/non-fake input的`FakeTensorMode`；
-`PythonKeyTracer`与ProxyTorchDispatchMode；
-context stack、hook proxy和compile context。

构造见 `torch/_dynamo/compiled_autograd.py:333-346`。

`begin_capture`建立固定的顶层placeholder：

```text
inputs, sizes, scalars, hooks, packed_data
```

随后把真实Tensor转换为FakeTensor，把sizes转为动态SymInt，并绑定对象与proxy。入口和
placeholder创建见 `torch/_dynamo/compiled_autograd.py:357-375` 与
`torch/_dynamo/compiled_autograd.py:376-392`；FakeTensor与SymInt准备见
`torch/_dynamo/compiled_autograd.py:394-423`。

## 4. 图内有哪些特殊对象

CA图不仅有ATen计算，还需要表达：

-autograd Node apply；
-tensor/node pre/post hooks；
-saved tensor pack/unpack hooks；
-AccumulateGrad与`.grad`更新；
-final callbacks；
-动态sizes/scalars；
-可能的AOT backward子图；
-错误/NaN检查；
-输出grad与`.grad()`返回值。

因此effect与顺序约束比纯函数forward更强，DCE和重排不能只看Tensor users。

## 5. 为什么结束捕获后还要重排

AOT或trace过程可能把accumulate-grad、hook等节点推到图末，而eager autograd engine会在
依赖满足时尽早调度。CA在end-capture中依次重排：

-unpack hooks；
-tensor/node pre-hooks；
-accumulate-grad；
-post-acc-grad hooks；
-post-hooks。

调用序列见 `torch/_dynamo/compiled_autograd.py:1192-1208` 与
`torch/_dynamo/compiled_autograd.py:1209-1223`。

例如`reorder_accumulate_grad_nodes`把更新移动到最后一个参数依赖之后，尽量模拟eager调度
（`torch/_dynamo/compiled_autograd.py:1308-1337`）。这会影响hook顺序、通信重叠和grad可见
时间，是语义机制，不是单纯性能pass。

## 6. DCE 为什么需要专用 impurity

CA graph中的placeholder unpack和特殊effect target即使无普通Tensor users也不能删除。
`dce()`先收集placeholder的unpack users，并把它们及 `_impure_targets`强制视为impure，再
调用FX DCE
（`torch/_dynamo/compiled_autograd.py:1090-1114`）。

这与普通“users为空即dead”的错误理解相反：在反向图里hook、grad mutation和callback是
用户可观察effect。

## 7. `end_capture`的产物

结束时：

1. 插入final callback stub和output；
2. 关闭trace contexts；
3. 可选处理CUDAGraph相关runtime inputs；
4. 清理dummy tensor metadata；
5. 记录重排前artifact；
6. hook/accumulate-grad重排；
7. DCE、移除未使用size；
8. 形成`CompiledAutograd{id}` GraphModule；
9. 返回runtime wrapper和`compiler_fn(graph)`。

前半流程见 `torch/_dynamo/compiled_autograd.py:1166-1190` 与
`torch/_dynamo/compiled_autograd.py:1192-1209` 与
`torch/_dynamo/compiled_autograd.py:1210-1228`；最终GraphModule与日志见
`torch/_dynamo/compiled_autograd.py:1228-1242`。

runtime wrapper过滤实际使用的sizes、必要时移动输入，并在 `_disable()`下调用compiled
function，防止递归capture
（`torch/_dynamo/compiled_autograd.py:1244-1259`、
`torch/_dynamo/compiled_autograd.py:1260-1275` 与
`torch/_dynamo/compiled_autograd.py:1276-1290`）。

## 8. Cache 与 specialization

CA cache必须覆盖的不只是Tensor metadata，还包括autograd graph结构、hooks、是否
accumulate-grad及其他runtime状态。改变以下任一项可能产生新图：

-loss到leaf的反向拓扑；
-hook注册/顺序；
-`backward()`与`autograd.grad()`输出契约；
-create_graph/retain_graph相关行为；
-动态sizes/scalars；
-AOT backward子图；
-参数是否需要grad。

因此“forward graph cache hit”不推出“Compiled Autograd cache hit”。

## 9. Enable/disable 生命周期

Python侧维护enabled、force-eager、in-region、disable-context和depth状态
（`torch/_dynamo/compiled_autograd.py:1621-1632`）。

`_enable`接收消费CA GraphModule的compiler callback和独立dynamic策略；注释建议通过配置
开启，并把对应forward也放在同一上下文
（`torch/_dynamo/compiled_autograd.py:1634-1658`）。

退出/重置时需要恢复C++ autograd compiler callback和logger/cache；不能在CA region内部
reset。该状态的全局性也意味着多线程/嵌套用法需遵守入口管理。

## 10. 与 DDP/通信的关系

Compiled Autograd看到AccumulateGrad与更完整的backward调度，因此理论上能优化跨region的
反向、hooks和通信边界。但：

-collective是effect，不能任意DCE/重排；
-DDP reducer对grad ready顺序有假设；
-所有rank必须得到兼容图和collective序列；
-graph/cache差异可能导致rank间不一致。

不能因为CA“图更大”就自动推断通信重叠更好。

## 11. 复杂度与成本

若runtime autograd图有 \(V_b\) 个node、\(E_b\) 条依赖：

-capture与FX建立至少 \(O(V_b+E_b)\)；
-专用重排/DCE通常线性或与users遍历相关；
-后续Dynamo/AOT/Inductor编译成本随图规模增长；
-cache lookup依赖CA cache entries与specialization；
-稳态收益需要摊销首次backward capture/compile。

CA扩大优化范围，也可能显著增加编译延迟和失败面。

## 12. 常见误解

- **“AOTAutograd已经编译反向，所以CA没作用。”** AOT边界通常较局部，CA从engine捕获更大
  runtime backward。
- **“CA图只是ATen backward op列表。”** 还包含hooks、grad accumulation和callbacks。
- **“forward命中cache意味着backward不会编译。”** CA与lazy AOT backward有独立时点。
- **“hook无Tensor输出即可DCE。”** hook是effect。
- **“更大的backward图一定更快。”** 编译成本、effect和distributed约束同样增加。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[d02_aot_runtime_wrappers_and_lazy_backward_compile_analysis]]
- [[e04_aotautograd_and_inductor_failure_localization_analysis]]
- [[f02_activation_checkpoint_recompute_and_compile_analysis]]
- [[f03_ddp_compile_boundaries_and_optimizer_analysis]]
- [[19_torch_compile_end_to_end/09_aotautograd_joint_forward_backward_graphs]]
