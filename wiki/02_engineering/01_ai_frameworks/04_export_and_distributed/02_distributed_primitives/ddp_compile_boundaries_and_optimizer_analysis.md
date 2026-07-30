# F03 · DDP、Compile Boundaries 与 Optimizer

> 卷别：F · 训练、分布式、扩展与部署  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[activation_checkpoint_recompute_and_compile_analysis]]  
> 后续：[[fsdp_dtensor_and_distributed_graphs_analysis]]  
> 最后更新：2026-07-30(kb-reorg P4 Task 9 迁入本目录,与 [[c10d_ddp_fsdp_dtensor_analysis]] 互指划界)

> [!note] 与 [[c10d_ddp_fsdp_dtensor_analysis]] 的分工
> 该页讲 DDP **原语本身**——C++ `Reducer` 怎样分桶、`all_reduce` 怎样与反向重叠,不涉及 `torch.compile`。本页讲 DDP 与**编译器相遇时**新增的一层问题:`DDPOptimizer` 为什么要按 DDP bucket 边界反向切分 Dynamo 捕获的 forward FX 图、bucket 与编译器 split 何时不对齐、以及 optimizer 是否入图的取舍。理解顺序:先读该页知道 DDP 本身怎样分桶通信,再读本页知道编译器如何据此切图。

## 1. DDP 为什么会反过来影响图切分

eager DDP通过每个parameter的AccumulateGrad hook获知梯度ready，并把多个grad组成bucket做
all-reduce。理想状态是：

```text
后层梯度ready → 启动后层bucket通信
              ↘ 同时继续算前层backward
```

如果AOTAutograd/Inductor把整个backward融合为一个对autograd engine“原子”的callable，
engine直到整段返回才看到所有grad ready，通信与计算重叠会消失。

DDPOptimizer的源码动机正是：让Dynamo forward graph按DDP bucket边界拆成多个可独立编译的
子图，使对应backward也分段返回并更早触发hooks
（`torch/_dynamo/backends/distributed.py:389-405`）。

## 2. DDPOptimizer 的输入与输出

输入：

- Dynamo捕获的forward `GraphModule`；
- example inputs；
- 与DDP一致的bucket byte cap/first bucket cap；
- 实际backend compiler。

输出不是多张独立顶层模型，而是：

- 一个split outer GraphModule；
- 每个child submodule单独编译；
- outer graph按原数据依赖调用children；
- 整体返回一个callable。

算法说明见 `torch/_dynamo/backends/distributed.py:407-427`。

## 3. 为什么逆序遍历 forward graph

典型网络的backward执行顺序与forward相反。DDPOptimizer逆序扫描FX nodes：

- 识别node参数；
- 累计参数storage bytes；
- 达到first/normal bucket cap时尝试开新bucket；
- 所有node即使没有参数也映射到某bucket；
- 若当前bucket没有可作为外部输出的值，会继续扩展，确保split子图有合法输出。

入口与阈值判断见 `torch/_dynamo/backends/distributed.py:506-535`；参数识别与node归桶见
`torch/_dynamo/backends/distributed.py:541-559` 与
`torch/_dynamo/backends/distributed.py:560-577`。

这是一种从forward数据流近似gradient-ready顺序的工程算法，不是读取真实autograd
execution trace。

## 4. Bucket 与 FX subgraph 的构造

收集bucket后：

1. 创建 `node -> bucket index` partition map；
2. 调用FX `split_module`；
3. 传播Dynamo source和metadata；
4. 记录outer与child图；
5. 用FakeTensorMode运行`SubmodCompiler`编译各child；
6. 重新compile outer GraphModule。

见 `torch/_dynamo/backends/distributed.py:594-619` 与
`torch/_dynamo/backends/distributed.py:621-647`。

若只有一个bucket，直接调用原backend，不做split
（`torch/_dynamo/backends/distributed.py:579-592`）。

## 5. 为什么 bucket 大小是多目标权衡

DDP本身说明小bucket意味着更频繁的小all-reduce，大bucket意味着更少、更大的collective；
first bucket可更小以降低首个通信启动延迟
（`torch/nn/parallel/distributed.py:36-62`）。

过小：

- 更多subgraphs与AOT/Inductor compile；
- 更多runtime边界和guards；
- 小collective效率低；
- fusion范围缩小。

过大：

- 通信启动晚；
- compute/communication overlap变差；
- 峰值grad/bucket生命周期可能增加。

优化目标是step critical path，而非最少图或最大fusion。

## 6. DDP bucket 与编译器 bucket 不一定完全一致

源码明确承认：

- DDP可能在运行中观察并重建bucket顺序；
- Dynamo若已有graph break，编译器split可能与DDP bucket不对齐；
- 被DDP忽略的parameter marker可能经其他transform丢失；
- 单个child backend失败可eager执行，其余child仍compiled。

见 `torch/_dynamo/backends/distributed.py:415-430`。

因此需要同时观察`ddp_graphs`、DDP logging data、collective timeline与每child compiled状态，
不能只看编译器打印的bucket表。

## 7. `static_graph`、unused parameters 与 hooks

DDP `find_unused_parameters`会从forward outputs遍历autograd graph，提前把不会获得grad的参数
标ready；`gradient_as_bucket_view`使`.grad`成为communication bucket的view；`static_graph`
声明used/unused集合和控制流在整个训练不变
（`torch/nn/parallel/distributed.py:713-728`、
`torch/nn/parallel/distributed.py:729-743` 与
`torch/nn/parallel/distributed.py:743-762`）。

这些选项改变：

- grad storage alias；
- hook触发与ready顺序；
- bucket rebuild；
- 可否跨迭代复用图；
- optimizer读取`.grad`的方式。

它们是编译正确性契约的一部分，不只是DDP调参。

## 8. Optimizer 是否入图

常见边界有三种：

### 只编译 model forward

backward由AOT/CA处理，optimizer在Python eager。边界清晰，optimizer Python overhead仍在。

### 编译完整 train step

loss、backward/grad和optimizer可能跨多个机制捕获；需要支持mutation、state dict、step
counter、foreach/fused op和AMP scaler。

### 独立编译 optimizer

forward/backward与optimizer分别有cache和guards。parameter/grad identity、`None` grad和state
初始化会决定specialization。

无论哪种方式，所有rank必须对是否step、skip step、grad scale溢出等控制流达成一致，否则
collective序列或parameter会分叉。

## 9. 与 Compiled Autograd 的关系

DDPOptimizer通过 **forward graph split** 间接让AOT backward分段，恢复autograd hook的较早
可见性。Compiled Autograd则直接捕获更大runtime backward，包括AccumulateGrad和hooks。

二者都可能改变grad-ready时间，但不是同一机制：

- DDPOptimizer：预先按bucket拆region；
- Compiled Autograd：运行时捕获engine backward；
- Python reducer/其他DDP模式：可能采用不同hook与collective表达。

组合启用时必须用timeline验证，而不能把单一机制的预期收益相加。

## 10. Correctness 与性能验收

每个rank验证：

- 相同参数初值、loss和grad；
- used/unused参数一致；
- collective数量、类型、顺序一致；
- bucket mapping与rebuild稳定；
- grad alias符合`gradient_as_bucket_view`；
- optimizer state与parameter更新一致；
- overflow/skip step跨rank一致；
- graph break/recompile不会造成某rank独立路径。

性能同时看backward compute、collective、overlap和最慢rank；单rank kernel时间不足以评价DDP。

## 11. 复杂度

DDPOptimizer逆序扫描graph和参数，主体近似 \(O(V+P)\)。FX split与编译成本取决于
bucket数 \(B\)：

\[
T_{\text{compile}} \approx
\sum_{b=1}^{B}T_{\text{backend}}(G_b)+T_{\text{outer}}
\]

通信近似：

\[
T_{\text{comm}} \approx
\sum_b(\alpha+\beta\cdot \text{bytes}_b)
\]

其中 \(\alpha\) 是collective启动成本，\(\beta\)是带宽项；有效step时间还取决于它与backward
critical path的重叠。

## 12. 常见误解

- **“DDP只是复制模型，不影响compile graph。”** bucket hook时机决定合理图边界。
- **“图越大通信重叠越好。”** 原子大backward可能让hooks全部延后。
- **“bucket越小越早通信所以最好。”** 启动成本、子图数和fusion会恶化。
- **“所有rank各自编译成功即可。”** collective序列和specialization必须兼容。
- **“optimizer不在forward就与编译无关。”** grad identity、alias、state mutation仍连接两者。

## 配套 Demo

本页对应卷级入口 `tools/labs_torch_compile/demo_f_advanced_topics.py` 的 `ddp_compile` 用例。默认以 CUDA 为验收设备：

```powershell
python -B tools\labs_torch_compile\demo_f_advanced_topics.py `
  --case ddp_compile --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\f03
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `ddp_compile/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[compiled_autograd_analysis]]
- [[activation_checkpoint_recompute_and_compile_analysis]]
- [[fsdp_dtensor_and_distributed_graphs_analysis]]
- [[04_export_and_distributed/02_distributed_primitives/index]]
- [[production_rollout_fallback_and_monitoring_analysis]]
