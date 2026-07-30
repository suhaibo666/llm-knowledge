# F05 · Custom Operator、Fake Kernel 与 Decomposition

> 卷别：F · 训练、分布式、扩展与部署  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[fsdp_dtensor_and_distributed_graphs_analysis]]  
> 后续：[[20_custom_backends_and_device_integration_analysis]]  
> 最后更新：2026-07-30(kb-reorg P4 Task 9 迁入本目录,判重 vs [[fx_graph_export_and_custom_ops_analysis]] §7)

> [!note] 判重结论:与 [[fx_graph_export_and_custom_ops_analysis]] §7 的关系
> 该页 §7「torch.library / custom_op」是 FX/export/functorch 全景 deep dive 中的一节(约 40 行),概述 `custom_op` 定义、`register_kernel`/`register_fake`/`register_autograd`/`Library` 句柄的**公开重导出入口**(`torch/library.py` 行号)。本页是专讲"custom op 作为编译器边界契约"的 13 节深度分析,覆盖 fake kernel 正确性要求(§5-§6)、mutation/version 与 ADInplaceOrView(§8)、decomposition/lowering/fallback 的选择与 Inductor 内部机制(§9)、失败定位分层(§10)、测试矩阵与复杂度模型(§11-§12)——这些在该页完全没有展开,且本页引用的是**底层实现**(`torch/_library/custom_ops.py`)行号而非公开重导出入口,两者互补不重复。独有内容占比远超一半,故保留为独立页,不并入;仅在该页 §7 末尾补一条指向本页的深度链接(重叠段无需收缩,因二者角度不同、无逐句重复)。

## 1. Custom op 定义的是“编译器边界契约”

一个Python/C++/第三方kernel要稳定进入PT2，编译器需要知道：

- 稳定operator name和schema；
- 输入输出类型与pytree边界；
- mutation、alias和view语义；
- 每device真实kernel；
- 无数据执行时的FakeTensor metadata；
- autograd formula与saved state；
- vmap/autocast等transform行为；
- 后端是直接lower、decompose还是fallback。

`custom_op`的用途之一就是阻止`torch.compile`/export/FX深入函数体，把它当稳定operator
（`torch/_library/custom_ops.py:67-93`）。

因此custom op不是“让任意Python函数自动可编译”的装饰器，而是要求作者补齐各子系统契约。

## 2. Schema 是所有层的共同事实

schema描述：

- 参数与返回类型；
- 默认值和keyword-only；
- alias set；
- 哪些输入被写；
- view/in-place/out标签。

`mutates_args`必须准确；若设为`"unknown"`会悲观假设所有输入都变异。错误声明属于undefined
behavior
（`torch/_library/custom_ops.py:94-113`）。

若用户同时提供schema与`mutates_args`，创建时会检查schema alias-write annotations与声明
一致（`torch/_library/custom_ops.py:235-263`）。

为什么如此严格：Dynamo guards、functionalization、AOT saved tensors、DCE、memory reuse和
autograd version counter都依赖同一mutation/alias事实。

## 3. `CustomOpDef`保存哪些注册槽位

对象持有：

- namespace/name/schema/tags；
- 各device backend kernels；
- abstract/fake function；
- autograd setup/backward；
- TorchDispatch/vmap/autocast；
- in-place/out metadata；
- dispatcher library和OpOverload。

见 `torch/_library/custom_ops.py:272-292` 与
`torch/_library/custom_ops.py:293-313`。

稳定identity来自dispatcher operator name，不来自初始Python function对象；FX node target会是
对应`OpOverload`。

## 4. 真实 kernel 与 dispatcher

`register_kernel`按device type注册实现：

- 无device type时走CompositeExplicitAutograd默认实现；
- 指定device时注册对应dispatch key；
- in-place/out实现会校验返回对象identity；
- 非view functional op检查alias约束；
- 无Tensor输入但device-specific时必须有`device: torch.device`参数。

见 `torch/_library/custom_ops.py:433-448`、
`torch/_library/custom_ops.py:449-463`、
`torch/_library/custom_ops.py:464-479`、
`torch/_library/custom_ops.py:480-494` 与
`torch/_library/custom_ops.py:496-516`。

这层解决“eager真实数据如何执行”，还没有解决FakeTensor或Inductor如何优化。

## 5. Fake kernel 为什么是编译必需品

Dynamo/AOT/Inductor经常只持有FakeTensor，需要在不读数据的情况下推导：

- 输出shape/stride；
- dtype/device；
- storage offset；
- 动态维度；
- 可能的alias关系。

`register_fake`的公开说明明确称其为让custom op高效工作于`torch.compile`所必需，并定义为
无数据Tensor上的metadata行为
（`torch/_library/custom_ops.py:523-534`）。

数据依赖输出shape应通过fake context创建unbacked dynamic size，而不是读取输入data；
示例语义见 `torch/_library/custom_ops.py:568-585`。

无fake impl且无法自动生成trivial fake时，dispatcher会直接报错，并说明compile/export/FX
需要fake impl（`torch/_library/custom_ops.py:772-787`）。

## 6. Fake kernel 必须满足的正确性

对每个代表输入，真实kernel与fake kernel的：

- 输出数量/pytree；
- shape/dtype/device/stride；
- alias/view；
- dynamic shape约束；
- 异常前置条件

必须一致。Fake kernel“只返回同shape empty Tensor”并不总正确：transpose、slice、view、
channels-last、data-dependent shape都会要求更精确metadata。

Fake registration最终作为Meta kernel加入dispatcher；holder还管理覆盖与注销
（`torch/_library/fake_impl.py:38-61` 与
`torch/_library/fake_impl.py:63-80` 与
`torch/_library/fake_impl.py:81-97`）。

## 7. Autograd formula

注册需要：

- `setup_context(ctx, inputs, output)`保存反向所需值；
- `backward(ctx, *grads)`返回各输入梯度；
- 两者自身必须可trace，不得直接读取data pointer、依赖或修改global state。

契约见 `torch/_library/custom_ops.py:639-655` 与
`torch/_library/custom_ops.py:656-672`。

当前API拒绝给out variant或非functional schema直接注册autograd formula，要求创建functional
operator
（`torch/_library/custom_ops.py:724-739`）。

这使AOTAutograd可以对functional forward/backward继续建图；若backward含不可trace实现，应
把它再封装为独立custom op。

## 8. Mutation version 与 ADInplaceOrView

mutable/view custom op需要dispatcher的ADInplaceOrView行为。实现会：

- 为mutable/view注册fallback；
- 对schema声明的mutated positional/keyword Tensor递增version；
- 保证autograd能发现in-place修改。

见 `torch/_library/custom_ops.py:789-806` 与
`torch/_library/custom_ops.py:807-823`。

漏报mutation不仅影响梯度报错，也会让编译器错误复用旧值、DCE或memory planning。

## 9. Decomposition、Lowering 与 Fallback 的选择

### Decomposition

把custom/high-level op展开为已有operator子图。优点是复用autograd、fake和Inductor
lowerings并获得fusion；缺点是失去opaque kernel边界，可能增加图或改变数值/性能。

### Direct lowering

保留op并在Inductor将其映射到自定义IR/kernel。适合真正专用硬件或高性能kernel，但需处理
layout、symbolic shape、codegen和runtime。

### Fallback

Inductor生成一个`FallbackKernel`调用dispatcher op。`fallback_handler`把IR args包装后创建
fallback IR，并标记该handler
（`torch/_inductor/lowering.py:2714-2729`）。

fallback通常保正确性/覆盖面，但可能阻断fusion、增加realization与launch。

Inductor防止同一op同时无理由拥有decomposition和fallback；优先建议decomposition
（`torch/_inductor/lowering.py:2832-2849` 与
`torch/_inductor/lowering.py:2850-2867`）。

## 10. 编译链中的检查点

```text
Python call
→ dispatcher schema/real kernel
→ Dynamo FX OpOverload
→ FakeTensor metadata
→ functionalization/AOT autograd
→ decomposition or preserved op
→ Inductor lowering/fallback
→ generated wrapper/runtime dispatcher
```

失败定位：

- eager失败：schema/device kernel；
- FakeTensor失败：fake impl；
- backward失败：autograd formula；
- functionalization失败：mutation/alias/view schema；
- Inductor失败：decomposition/lowering/layout；
- compiled accuracy错误：逐层比real、fake、autograd与lowering。

## 11. 测试矩阵

- 每device与dtype；
- shape/stride/layout；
- dynamic output shape；
- empty/zero-dim；
- mutation/version；
- alias/view；
- forward/backward/gradgrad；
- autocast/vmap；
- FakeTensor与`torch.compile(fullgraph=True)`；
- export/AOTInductor若要部署；
- fallback与direct lowering结果；
- 多rank/collective若为distributed op。

## 12. 复杂度与性能

opaque custom op把内部 \(V\) 个算子压成一个FX node，降低capture/pass图规模，但编译器失去
跨边界优化信息。decomposition展开为 \(V\) 个node，增加编译成本，却可能通过fusion减少
runtime物化。选择应比较：

\[
T_{\text{total}}
=T_{\text{compile}}+N\cdot T_{\text{runtime}}
\]

而不是只按FX node数决定。

## 13. 常见误解

- **“有CPU/CUDA kernel就能compile。”** 还需要fake、schema和必要autograd契约。
- **“fake kernel可运行真实数据。”** 它只描述metadata，不能读数据。
- **“mutates_args只是文档。”** 它驱动alias、functionalization和version语义。
- **“decomposition总比fallback快。”** 展开后的图与目标硬件需实测。
- **“opaque custom op更安全。”** 错误schema会让编译器在边界外做错误优化。

## 配套 Demo

本页对应卷级入口 `tools/labs_torch_compile/demo_f_advanced_topics.py` 的 `custom_op_contract` 用例。默认以 CUDA 为验收设备：

```powershell
python -B tools\labs_torch_compile\demo_f_advanced_topics.py `
  --case custom_op_contract --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\f05
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `custom_op_contract/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[04_export_and_distributed/01_fx_export_extensibility/index]] — 本模块 overview
- [[fx_graph_export_and_custom_ops_analysis]] — §7 是 custom_op 注册机制的公开入口概述(本页的浅层对照),见页头判重结论
- [[10_pytorch_dispatcher_analysis]] — schema/dispatcher/ADInplaceOrView 分层
- [[dispatch_modes_proxytensor_faketensor_analysis]] — ProxyTensor/FakeTensor 两套抽象执行状态（2026-07-30 起独立成页，取代原 `aotautograd_analysis` §13 引用）
- [[fsdp_dtensor_and_distributed_graphs_analysis]]
- [[20_custom_backends_and_device_integration_analysis]]
