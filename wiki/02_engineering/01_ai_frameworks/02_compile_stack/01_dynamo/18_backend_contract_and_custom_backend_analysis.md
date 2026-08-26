# B10 · Backend Contract 与自定义 Backend 边界

> 卷别：B · TorchDynamo 捕获  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[17_dynamic_shapes_generalization_and_fallback_analysis]]  
> 后续：[[courses/torch_compile_end_to_end]]  
> 最后更新：2026-07-30(与 [[30_dynamo_pass_methodology]] 互加回链;该页收窄为本页未覆盖的开发决策线,契约机制以本页为准)

## 1. 为什么 Dynamo需要极小的 backend契约

如果Dynamo直接依赖某个特定codegen，捕获层就无法：

- 用eager backend隔离frontend问题；
- 接入AOTAutograd/Inductor、ONNX或第三方compiler；
- 测试GraphModule本身；
- 组合debug wrapper和compiler bisector；
- 让out-of-tree backend独立演进。

因此它把边界压缩为：

```python
def backend(gm: torch.fx.GraphModule, example_inputs: list[Tensor]):
    return compiled_callable
```

**核心结论**：Dynamo承诺交付“当前region的FX GraphModule与用于编译的example inputs”；
backend承诺返回行为等价的callable。两边并不共享任意Python frame内部状态。

## 2. Registry负责名字到callable

`register_backend`接受一个读取FX graph和fake inputs的compiler callable，保存名字和tags
（`torch/_dynamo/backends/registry.py:87-115`）。

`lookup_backend`：

- string未加载时触发lazy import；
- 可从entry point惰性load；
- 最终返回callable；
- 传入callable时直接返回。

见 `torch/_dynamo/backends/registry.py:124-142`。

registry是发现/命名层，不是编译cache，也不定义backend内部IR。

## 3. Backend实际收到的 `gm`

它是一个Dynamo region，不一定是：

- 用户完整函数；
- whole program；
- forward与backward联合图；
- functionalized ATen图；
- 已经完成decomposition的图；
- 只有call_function nodes的纯图。

取决于backend包装层：

- 直接自定义backend收到Dynamo FX region；
- 默认Inductor wrapper内部调用 `compile_fx`，后者再编排AOTAutograd；
- AOTAutograd的fw/bw compiler收到更低层、functionalized/partitioned图。

所以“backend graph长什么样”必须说明是哪一个backend边界。

## 4. Example inputs为什么通常是FakeTensor

Dynamo构图时用FakeTensor传播metadata，避免真实执行大Tensor计算。backend inputs用于：

- 读取shape/dtype/device/layout；
- 运行meta/decomposition；
- 建立AOT/Inductor tracing context；
- specialization/codegen决策。

backend不应假设它们可以任意做真实device计算，也不应把example input对象identity当作
runtime input identity。

OutputGraph在调用前从graph inputs收集example inputs并执行lint
（`torch/_dynamo/output_graph.py:3037-3052`）。

## 5. 返回值必须是callable

OutputGraph最终调用 `compiler_fn(gm, example_inputs)`并检查返回值
（`torch/_dynamo/output_graph.py:3286-3293`）。

这个callable之后被嵌入transformed bytecode。其runtime契约包括：

- 接收与graph placeholders对应的真实值；
- 输出结构与graph output约定一致；
- 保持dtype/device/layout/alias/mutation语义；
- 抛出与用户可观察语义兼容的异常；
- 不依赖仅在compile-time存在的fake对象。

“成功返回callable”只是结构检查，不代表语义正确；正确性验证需要E卷的方法论。

## 6. Backend wrapper如何传 mode/options

默认Inductor wrapper把mode/options转成Inductor config patches，再调用
`compile_fx`（`torch/__init__.py:2984-2999`）。

其他backend由 `_TorchCompileWrapper`解析名字并保存kwargs；实际调用为：

```python
self.compiler_fn(model_, inputs_, **self.kwargs)
```

见 `torch/__init__.py:3057-3080` 与 `torch/__init__.py:3090-3095`。

因此最小backend只需两个位置参数；若用户会传非默认mode/options，backend还需明确接收或
拒绝这些kwargs。

## 7. Backend可选上下文和配置接口

Dynamo还会查询：

- `backend_ctx_ctor`：编译wrapper执行时进入额外context；
- `get_compiler_config`：暴露compiler config用于记录/缓存/调试；
- `reset`：`torch._dynamo.reset`时清后端状态；
- compiler name/equality：backend identity和cache dispatch。

`_optimize`读取 `backend_ctx_ctor`和可选compiler config
（`torch/_dynamo/eval_frame.py:1826-1829` 与
`torch/_dynamo/eval_frame.py:1850-1860`）。

这些是增强接口，不改变核心二入一出契约。

## 8. Backend failure在哪一层出现

```text
Dynamo capture成功
→ gm.graph.lint成功
→ backend(gm, example_inputs)
   ├── lowering失败
   ├── decomposition/fake support失败
   ├── native compiler失败
   ├── 生成callable不是callable
   └── callable运行时失败/算错
```

前四类通常在first-call compile阶段暴露；最后一类可能只在特定runtime input上暴露。
Dynamo会把一般backend异常包装为backend compiler failure，但一些异常类型原样传播。

## 9. 最小自定义backend的用途

### 返回 `gm.forward`

用于验证Dynamo capture和GraphModule语义，隔离native codegen。

### 打印/保存graph后返回forward

用于观察region边界和placeholder metadata，但必须避免在生产路径泄露大对象或改变graph。

### 包装另一个backend

用于计时、校验、图变换或fallback。若修改graph，应：

- 保持topological legality；
- 维护node metadata；
- 处理副作用/alias；
- 执行lint和必要DCE；
- 验证输出结构；
- 明确cache key是否覆盖自己的配置。

## 10. 自定义backend与FX pass的责任边界

一个backend可以在调用下游compiler前应用FX passes，但pass必须以**它实际收到的IR方言**
为前提。Dynamo graph可能含：

- `call_function`、`call_method`、`call_module`；
- Python operator与ATen target混合；
- higher-order ops；
- symbolic shapes；
- node metadata和Dynamo Source。

不能把Inductor post-grad pattern直接假设为适用于原始Dynamo graph。PatternExpr定义的是
“在某一FX graph方言中寻找怎样的局部子图”，不是任意阶段通用语法。

## 11. Backend与AOTAutograd的组合

训练时，常见组合为：

```text
Dynamo backend wrapper
→ AOTAutograd functionalization/decomposition/joint graph
→ partition fw/bw
→ fw_compiler(fw_gm, fw_inputs)
→ bw_compiler(bw_gm, bw_inputs)
→ runtime wrapper
```

此时fw/bw compiler也是backend-like callable，但其输入图和runtime契约不同于最外层
Dynamo backend。两张图通过显式saved values/output-placeholder接口连接，而不是共享
GraphNode或FX edge。

## 12. 复杂度

最小registry lookup：

- 已解析callable：近似 $O(1)$；
- string首次lazy import：取决于entry point/module加载；
- backend wrapper配置处理：$O(P)$，$P$为options数量。

backend本身复杂度记为 $K(G, S, H)$，受graph规模 $G$、shape约束 $S$和硬件/候选
$H$影响。对自定义backend不能用Dynamo的线性capture复杂度替代 $K$。

## 13. 正确性清单

- 输入placeholder顺序和runtime args一致；
- 输出pytree、alias与mutation一致；
- 不对FakeTensor做不受支持的真实计算；
- graph变换后lint；
- 对random、collective、mutation等effect保序；
- backend配置进入cache key或明确禁用错误复用；
- compile-time和runtime device context清晰；
- failure不会静默返回错误结果；
- 与dynamic shapes契约一致；
- 多线程/多进程cache和global state有明确所有权。

## 14. 常见误解

- **“backend就是Inductor。”** Inductor只是默认实现。
- **“自定义backend收到原Python函数。”** 核心输入是Dynamo GraphModule。
- **“example inputs一定是真实Tensor。”** 通常是FakeTensor或带符号metadata的输入。
- **“GraphModule能跑就等价。”** alias、mutation、异常、dtype精度和输出结构都要验证。
- **“backend pass可以匹配任何阶段的graph。”** pattern依赖目标IR方言和规范化阶段。

## 配套 Demo

本页对应卷级入口 `tools/labs_torch_compile/demo_b_dynamo_capture.py` 的 `custom_backend_contract` 用例。默认以 CUDA 为验收设备：

```powershell
python -B tools\labs_torch_compile\demo_b_dynamo_capture.py `
  --case custom_backend_contract --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\b10
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `custom_backend_contract/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[courses/torch_compile_end_to_end]]
- [[17_dynamic_shapes_generalization_and_fallback_analysis]]
- [[22_pattern_expression_and_matcher_engine_analysis]]
- [[20_custom_backends_and_device_integration_analysis]]
- [[16_compiled_correctness_validation_methodology_analysis]]
- [[30_dynamo_pass_methodology]] — 开发决策线:何时该/不该在 Dynamo 做、可运行注册代码、改图排错流程、何时该离开 Dynamo
