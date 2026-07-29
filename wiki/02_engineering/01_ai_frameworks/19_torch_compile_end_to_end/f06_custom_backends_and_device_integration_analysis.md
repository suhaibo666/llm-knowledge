# F06 · Custom Backend 与 Device Integration

> 卷别：F · 训练、分布式、扩展与部署  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[f05_custom_operators_fake_kernels_and_decompositions_analysis]]  
> 后续：[[f07_aotinductor_packaging_and_deployment_analysis]]  
> 最后更新：2026-07-28

## 1. 三种“backend”不要混为一层

### Dynamo backend

接收FX `GraphModule + example inputs`，返回可调用对象。可以完全绕过Inductor。

### Inductor device backend

继续使用Inductor GraphLowering/IR/Scheduler，但提供目标device的scheduling、kernel codegen、
wrapper codegen和runtime overrides。

### Dispatcher/custom op backend

为operator提供device kernel。它解决单op执行，不自动构成整图compiler backend。

三者可组合，也可独立存在。

## 2. Dynamo backend 的最小契约

类型定义是：

```text
CompilerFn(GraphModule, list[Tensor]) -> CompiledFn
CompiledFn(*runtime Tensor) -> tuple[Tensor, ...]
```

源码见 `torch/_dynamo/backends/registry.py:75-84`。

注册只提供string shorthand；外部项目也可直接传callable。registry保存name、tags和compiler
function，并拒绝重复name
（`torch/_dynamo/backends/registry.py:87-115`）。

lookup按需加载entry point、给无效name提供建议，最终返回稳定compiler callable
（`torch/_dynamo/backends/registry.py:124-142`）。

真正生产backend还需处理：

-FakeTensor/example input，不可读取真实数据；
-dynamic SymInt；
-boxed calling convention；
-mutation/alias输出ABI；
-forward/backward compiler；
-config/cache key；
-线程、device和exception；
-返回callable lifetime。

## 3. 为什么 backend callable identity 要稳定

Dynamo cache entry保存backend identity；同一逻辑backend若每次由新lambda/closure创建，可能
触发`BACKEND_MATCH` guard failure。注册名称解决发现，不自动解决内部config/version
identity。

建议backend对象具有：

-稳定名称与版本；
-可序列化、可hash配置；
-确定性compile output；
-清晰capability；
-显式artifact/cache compatibility key。

## 4. DeviceInterface 解决 runtime 抽象

Inductor需要设备无关地查询：

-device context与数量；
-Event/Stream；
-current/exchange/set device；
-raw stream与synchronize；
-device properties/compute capability；
-multiprocessing worker安全查询。

抽象定义见 `torch/_dynamo/device_interface.py:40-68`、
`torch/_dynamo/device_interface.py:70-85`、
`torch/_dynamo/device_interface.py:86-100`、
`torch/_dynamo/device_interface.py:102-119` 与
`torch/_dynamo/device_interface.py:120-136`。

Event/Stream必须继承PyTorch基类才能被Dynamo捕获。Worker API之所以独立，是因为forked
compile worker不能随意初始化GPU runtime。

接口按device string注册和lazy lookup；内置注册包括cuda、xpu、mtia、cpu、mps、tpu
（`torch/_dynamo/device_interface.py:620-641` 与
`torch/_dynamo/device_interface.py:644-660`）。

## 5. Inductor backend 为什么需要 scheduler 与 wrapper

generated code有两部分：

-kernel code；
-桥接kernel、allocation、stream和输出的wrapper。

源码说明新backend需提供自定义Scheduling以生成kernel，并从PythonWrapperCodegen继承/覆盖
目标逻辑；`register_backend_for_device`登记：

-device scheduling；
-Python/C++/FX wrapper constructors；
-custom graph pass；
-custom config。

见 `torch/_inductor/codegen/common.py:389-418` 与
`torch/_inductor/codegen/common.py:419-434`。

只实现kernel emitter而没有wrapper、stream、allocation与call ABI，不能形成可运行整图。

## 6. Device op overrides

wrapper代码需要目标device专属表达：

-set/synchronize device；
-device/stream guard；
-raw/current stream；
-kernel header/driver/type；
-device pointer；
-AOTI stream guard；
-scratch/TMA helpers。

抽象方法见 `torch/_inductor/codegen/common.py:321-350` 与
`torch/_inductor/codegen/common.py:352-380`。

override按device注册并lazy初始化
（`torch/_inductor/codegen/common.py:650-679`）。

这是code string/runtime ABI层，与`DeviceInterface`的Python runtime query层不同。

## 7. Operator coverage：decomposition、lowering、fallback

Inductor lowering把ATen target映射到IR构造函数，并统一处理broadcast、type promotion和IR
validation
（`torch/_inductor/lowering.py:481-510` 与
`torch/_inductor/lowering.py:511-532`）。

device backend需要回答每个可能op：

-先decompose成已有op；
-在共享/自定义lowering生成IR；
-生成external/fallback call；
-明确不支持并让上层失败/回退。

缺少一个路径会表现为lowering failure；错误fallback layout则可能导致runtime错或不必要
realization。

## 8. Capability 与 feature gating

不同device可支持不同：

-foreach；
-in-place buffers；
-scan/sort；
-tuple reduction；
-Triton templates；
-single-element reduction；
-loop order偏好。

`BackendFeature`定义与按device查询入口见
`torch/_inductor/codegen/common.py:437-455`。

pass/scheduler应先查询capability，而不是生成代码后再让compiler报错。feature集合还必须进入
cache兼容性。

## 9. 完整接入链

```mermaid
flowchart LR
    O["Operator kernels/schema/fake"] --> D["Dynamo backend"]
    D --> G["AOT/FX graph"]
    G --> L["Decomposition + lowering"]
    L --> I["Inductor IR"]
    I --> S["Device scheduling"]
    S --> K["Kernel codegen"]
    S --> W["Wrapper codegen"]
    W --> R["DeviceInterface/runtime"]
    K --> R
    R --> A["Artifact/cache/AOTI"]
```

每层都有独立单元测试和失败分类。用一个“device支持torch.compile”的布尔值无法描述能力。

## 10. Cache 与 ABI

cache key至少包含：

-backend版本；
-device architecture/capability；
-driver/runtime/compiler；
-feature flags；
-lowering/decomposition版本；
-wrapper/AOTI ABI；
-shape/dtype/layout specialization；
-custom op library version。

远端cache不能假设所有worker硬件等价。load前应验证兼容性，失败时rebuild或安全fallback。

## 11. 测试梯度

1. dispatcher op eager；
2. FakeTensor/meta；
3. Dynamo backend identity/contract；
4. dynamic shapes与guards；
5. AOT forward/backward；
6. lowering IR；
7. scheduler legality；
8. wrapper/kernel compile；
9. stream/event/device guard；
10. correctness与memory；
11. cache serialize/load；
12. AOTI/package；
13. multi-device/distributed。

CPU-only或codegen-only测试不能声称目标accelerator runtime已验证。

## 12. 复杂度与维护成本

接入成本大致是：

\[
O(|\text{ops}|+|\text{IR features}|+|\text{runtime features}|
+|\text{artifact ABIs}|)
\]

decomposition可降低直接lowering数量，但会增加图规模；fallback提高覆盖面但损失fusion；
自定义scheduler提高性能但扩大正确性与维护面。应先建立可用的最小纵向链，再按profiling扩展。

## 13. 常见误解

- **“注册Dynamo backend就等于接入Inductor新设备。”** 前者可完全独立。
- **“有custom op kernel就能编译整图。”** 还缺fake、AOT、lowering和wrapper。
- **“DeviceInterface就是codegen。”** 它是runtime抽象；kernel/wrapper有另一组接口。
- **“fallback意味着回到整个eager模型。”** 可能只在compiled graph内调用单个external op。
- **“能生成源码就是设备支持完成。”** native compile、load、execute、stream和ABI仍需验证。

## 配套 Demo

本页对应卷级入口 `labs/demo_f_advanced_topics.py` 的 `custom_backend` 用例。默认以 CUDA 为验收设备：

```powershell
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\demo_f_advanced_topics.py `
  --case custom_backend --device cuda `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\volume_demos\f06
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `custom_backend/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[b10_backend_contract_and_custom_backend_analysis]]
- [[f05_custom_operators_fake_kernels_and_decompositions_analysis]]
- [[f07_aotinductor_packaging_and_deployment_analysis]]
- [[d03_async_compile_workers_and_module_loading_analysis]]
- [[19_torch_compile_end_to_end/17_fx_lowering_to_inductor_ir]]
