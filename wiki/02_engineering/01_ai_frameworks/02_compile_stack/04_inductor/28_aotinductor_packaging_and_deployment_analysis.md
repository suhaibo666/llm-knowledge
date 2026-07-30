# F07 · AOTInductor 打包、ABI 与部署

> 卷别：F · 训练、分布式、扩展与部署  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[20_custom_backends_and_device_integration_analysis]]  
> 后续：[[training_inference_cudagraph_and_freezing_analysis]]  
> 最后更新：2026-07-30(kb-reorg P4 Task 9 迁入本目录,纯平移;解答 [[22_backend_modes_options_stances_and_fullgraph_analysis]] §14.2 的 `use_aoti` todo)

## 1. 为什么还需要 AOTInductor

`torch.compile` 的默认使用方式是“Python 进程遇到输入后捕获、编译、缓存并执行”。这很适合
开发和服务内即时编译，却没有天然解决以下部署问题：

- 构建机与运行机分离；
- 运行环境不希望携带 Python 编译器栈；
- 模型要作为可审计、可发布、可回滚的文件；
- 输入输出的 Python 容器结构需要跨编译边界保留；
- 多模型、常量权重、设备选择和 runtime ABI 需要显式管理。

AOTInductor 的设计目标不是另一套图优化器，而是把已经导出的程序沿 Inductor lowering
编译成可装载产物，并补齐 archive、call spec、C ABI 和 runner。

## 2. JIT cache 与 AOT package 不是同一种产物

| 维度 | `torch.compile` JIT/cache | AOTInductor package |
|---|---|---|
| 捕获时机 | 运行中，由实际 frame/guards 驱动 | 部署前，由 `ExportedProgram` 驱动 |
| 选择正确版本 | Dynamo guard/cache lookup | export 约束与 package/runtime 契约 |
| 主要消费者 | 原 Python 进程中的 compiled callable | PT2 loader 与 AOTI runner |
| 文件语义 | 编译加速缓存，可失效重建 | 发布工件，需要版本、兼容性和回滚治理 |
| Python 结构 | wrapper/调用现场仍在 Python 内 | 通过序列化 pytree call spec 重建 |
| 失败策略 | recompile、graph break、fallback | 拒绝加载、重新构建或切换已发布版本 |

所以不能把 Inductor cache 目录当作稳定部署格式。cache key 解决“能否复用一次编译”，
package ABI 解决“另一个运行环境能否正确装载并调用”。

## 3. 入口为什么要求 ExportedProgram

公开入口 `aoti_compile_and_package` 明确只接受 `ExportedProgram`，并要求其中存在
`example_inputs`；它还把 `aot_inductor.package` 强制设为真
（`torch/_inductor/__init__.py:108-127`、`torch/_inductor/__init__.py:151-166`）。

这个约束表达了两层设计：

1. 部署边界前必须完成可导出程序的捕获与约束表达，而不是把任意 Python 行为推迟到运行机；
2. 编译仍需要样例输入提供 dtype、device、shape/layout 等专门化信息。

它并不意味着所有动态输入都被固定。可支持的动态维度由 export 约束和后端能力决定；
样例输入只是编译实例，不能代替完整输入契约。

## 4. 从 ExportedProgram 到 PT2 的状态链

```mermaid
flowchart LR
    E["ExportedProgram"] --> V["校验 example inputs 与 package 配置"]
    V --> G["GraphModule 与扁平输入"]
    G --> C["compile_fx_aot / AOTInductor"]
    C --> F["C++、object、shared library、metadata、weights"]
    F --> P["package_aoti"]
    P --> Z["PT2 archive"]
    Z --> L["load_pt2 / package loader"]
    L --> R["AOTI model runner"]
    R --> O["unflatten 后的用户输出"]
```

内部函数先调用 `aot_compile`，再根据生成文件推导默认 `.pt2` 路径并调用
`package_aoti`；用于 minifier 时还能装载、运行或做准确性对比
（`torch/_inductor/__init__.py:199-214`、
`torch/_inductor/__init__.py:215-229`、`torch/_inductor/__init__.py:232-254`）。

更底层的 `aot_compile` 会：

- 去掉目前 AOTI 不支持的 guards 子模块；
- 扁平化样例输入并生成 call-spec 配置；
- 在 compiling-state context 中进入 `compile_fx_aot`。

对应入口见 `torch/_inductor/__init__.py:290-305`、
`torch/_inductor/__init__.py:306-320` 与
`torch/_inductor/__init__.py:322-332`。

## 5. Package 装的不是“一个 Python 函数”

PT2 archive 保存的是 AOTInductor 生成文件及其元数据。`package_aoti` 接受单模型文件集合，
也接受“模型名 → 文件集合”的映射
（`torch/_inductor/package/package.py:83-100`）。

公开接口也明确展示了把多个命名模型放入一个 `.pt2`，再按名字加载的流程
（`torch/_inductor/__init__.py:87-106`）。

这意味着部署标识至少有两层：

- archive version：哪一个发布包；
- model name：包中的哪一个 compiled model。

模型版本、权重版本、运行时版本和硬件目标不应被压缩成一个模糊文件名。

## 6. 编译产物与权重布局

CPU package 路径会读取编译与链接 flags，将生成的 C++ 和 constants object 链接成 shared
library；若存在 serialized weights，还会按页对齐后追加到 `.so`
（`torch/_inductor/package/package.py:24-50`、
`torch/_inductor/package/package.py:52-80`）。

这是为何 package 兼容性不能只看 FX graph：

- generated source 依赖编译器和 flags；
- object/shared library 依赖目标平台 ABI；
- kernel 依赖设备架构与 runtime；
- serialized weights 依赖常量布局和生成代码；
- metadata/call spec 依赖对应 loader 的解释方式。

同一张图在不同 target 上可以得到不同的、不可互换的二进制工件。

## 7. Call spec：Python 容器与扁平 Tensor ABI 的桥

编译前 `_aoti_flatten_inputs` 提取输入/输出 pytree spec，将其序列化进 AOTI 配置；同时
把 `(args, kwargs)` 扁平化，并验证调用结构和 exported input spec 一致
（`torch/_inductor/compile_fx.py:3390-3417`、
`torch/_inductor/compile_fx.py:3419-3437`、
`torch/_inductor/compile_fx.py:3454-3474`）。

运行时 `AOTICompiledModel.__call__` 的顺序是：

1. 从 loader 获取 input/output call spec；
2. 按 input spec 重排 kwargs；
3. flatten 为 Tensor 列表；
4. 调用 `boxed_run`；
5. 按 output spec unflatten。

实现见 `torch/export/pt2_archive/_package.py:731-746`。C++ runner 通过 ABI 函数取得两段
序列化 spec（`torch/csrc/inductor/aoti_runner/model_container_runner.cpp:480-485`）。

因此 ABI 不是只有 Tensor 指针；参数树的叶子顺序、kwargs 顺序、输出嵌套结构同样是
调用契约。

## 8. C ABI runner 为什么采用动态符号

runner 用 `DynamicLibrary` 打开模型 `.so`，再按名称解析创建、删除、运行、输出数、
常量信息、call spec 等函数
（`torch/csrc/inductor/aoti_runner/model_container_runner.cpp:84-99` 与
`torch/csrc/inductor/aoti_runner/model_container_runner.cpp:100-115`）。

部分新功能使用可选符号：缺失时警告，只有真正需要该功能才失败；但单线程执行等被请求的
关键能力缺失时会立即拒绝
（`torch/csrc/inductor/aoti_runner/model_container_runner.cpp:119-138`、
`torch/csrc/inductor/aoti_runner/model_container_runner.cpp:140-165`）。

这种设计的原因是：

- shared library 边界不应传递不稳定的 C++ STL 对象；
- C symbol 比 C++ name mangling 更适合作为跨编译单元 ABI；
- required/optional symbol 区分允许 runtime 与旧产物做有限兼容；
- ABI 检查必须在 load 阶段暴露，而不是执行到某条 kernel 才随机失败。

旁路算子还可以通过与 `.so` 同名的 JSON 创建 proxy executor
（`torch/csrc/inductor/aoti_runner/model_container_runner.cpp:167-178`）。

## 9. 常量与所有权

外部常量以“名称 + `AtenTensorHandle`”形式跨 C ABI 传递。runner 明确只借用调用方 Tensor，
调用方保留所有权且 Tensor 必须比 runner 活得更久
（`torch/csrc/inductor/aoti_runner/model_container_runner.cpp:52-81`）。

runner 还解析常量名称/FQN、dtype、更新 active/inactive buffer、constant folding 和 buffer
swap 等符号
（`torch/csrc/inductor/aoti_runner/model_container_runner.cpp:93-115`）。

所以“权重在包里”不是唯一模式。部署系统必须记录：

- 权重内嵌还是外部托管；
- 外部 Tensor 的 owner 和 lifetime；
- 更新是全量还是增量；
- update 后是否需要 constant folding；
- 并发请求正在读哪个 constant buffer；
- 权重 schema 是否与 compiled model 的 FQN、dtype、shape 匹配。

错误管理 lifetime 会成为悬空 handle；错误热更新可能成为跨请求的数据竞争。

## 10. Loader 与兼容路径

`load_package` 优先使用新的 `load_pt2`，检查指定模型名；设备不兼容错误会直接向上传播，
其他 runtime error 才进入旧 PT2 loader 的兼容路径并发出“重新生成 package”的警告
（`torch/_inductor/package/package.py:103-123`、
`torch/_inductor/package/package.py:125-141`）。

因此“能打开 zip/archive”远不等于可执行。加载检查应包括：

- archive/schema version；
- model name；
- OS、architecture、libc/C++ ABI；
- PyTorch/AOTI runtime ABI；
- CUDA/ROCm/XPU runtime 与目标架构；
- required symbols；
- custom op/proxy executor library；
- constant schema；
- call spec；
- 安全来源与完整性。

兼容失败的正确动作通常是选择匹配工件或重新构建，而不是吞掉错误继续执行。

## 11. 动态 shape 与部署版本

AOT 并不取消 shape 约束。部署前要区分：

- export 接受的输入范围；
- generated kernel 的动态能力；
- runtime assert；
- 某些 shape 是否触发不同 artifact；
- 超出范围时是拒绝、fallback 还是路由到另一版本。

在线 JIT 可以为新输入重新编译；纯 AOT 环境通常没有这个逃生口。因此动态范围必须进入
接口测试和容量规划，而不是只用一组 example inputs 验收。

## 12. 失败定位按阶段分层

| 阶段 | 典型失败 | 证据 |
|---|---|---|
| export | Python 行为、约束、custom op fake 不可导出 | ExportedProgram/constraint 诊断 |
| AOT compile | lowering、codegen、native compile 失败 | generated source、compiler command |
| package | 文件缺失、命名或 archive schema 错 | package manifest |
| load | device/ABI/symbol/custom library 不匹配 | loader 与 `dlsym` 错误 |
| call adaptation | pytree/call spec 不一致 | in/out spec 与 flat leaves |
| execute | kernel、stream、内存、常量错误 | runtime logs、correctness oracle |

公开入口本身由 AOTInductor minifier wrapper 包裹
（`torch/_inductor/__init__.py:116-127`、
`torch/_inductor/__init__.py:160-166`），但最小复现不能替代部署环境的 ABI 记录。

## 13. 发布与回滚机制

推荐把 package 当成不可变工件，manifest 至少记录：

- 源 PyTorch commit 与构建配置；
- export 输入约束；
- target/device architecture；
- compiler、driver、runtime 版本；
- custom libraries；
- model names、call specs、constant schema；
- correctness/performance 测试结果；
- 工件 hash、签名与生成时间；
- 可回滚的上一版本。

上线顺序应是 load-only 检查、离线 golden correctness、目标机 smoke、shadow/canary、
逐级放量。重新编译应产生新 artifact id，不应原地覆盖已运行版本。

## 14. 复杂度与容量

设 FX/IR 节点数为 \(N\)，生成 kernel 数为 \(K\)，生成源码总大小为 \(S\)，权重大小为
\(W\)，package 内模型数为 \(M\)：

- 图 lowering 通常至少 \(O(N)\)，具体优化可能高于线性；
- native compile 主要随 \(K\)、\(S\) 与编译器优化增长；
- package 写入/读取至少 \(O(S+W)\)；
- model-name lookup 取决于 archive 索引，不能掩盖解压、动态装载和权重映射成本；
- 常量热更新至少与被更新字节数线性相关；
- 同时保留多个 runner/constant buffer 会近似放大相应内存。

容量评估必须分开 archive 磁盘、load 峰值内存、常驻权重、workspace、首次调用初始化与
稳态执行。

## 15. 常见误解

- **“AOTInductor 等于提前跑一次 `torch.compile`。”** 它还建立 export、package 与 runtime ABI。
- **“PT2 是跨任意平台的通用模型格式。”** 其中可包含 target-specific 二进制和权重布局。
- **“一个 example input 就定义了全部输入域。”** 输入域由 export/shape 约束定义。
- **“Python kwargs 不属于 ABI。”** 它们的 pytree 顺序由 call spec 编码。
- **“package 能加载就证明结果正确。”** load、call-adaptation、execute 和 numerical correctness 是不同验证层。
- **“外部常量交给 runner 后由 runner 拥有。”** 该 C++ 路径明确采用借用所有权。
- **“旧包 fallback loader 会自动修复兼容性。”** 它只是有限兼容路径，并会要求重新生成。

## 配套 Demo

本页对应卷级入口 `tools/labs_torch_compile/demo_f_advanced_topics.py` 的 `aotinductor_package` 用例。默认以 CUDA 为验收设备：

```powershell
python -B tools\labs_torch_compile\demo_f_advanced_topics.py `
  --case aotinductor_package --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\f07
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `aotinductor_package/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[20_custom_backends_and_device_integration_analysis]]
- [[training_inference_cudagraph_and_freezing_analysis]]
- [[02_compile_stack/06_compile_cache/index]]
- [[14_compiled_artifact_lifecycle_and_runtime_failures_analysis]]
- [[19_production_rollout_fallback_and_monitoring_analysis]]
- [[22_backend_modes_options_stances_and_fullgraph_analysis]] — §14.2:`torch.compile(..., options={"use_aoti": True})` 的 JIT 入口路径,与本页 §2-§3 的 export 驱动打包路径汇合于同一套 `compile_fx`/`CompiledAOTI` 机制,但 `enable_autograd_for_aot` 门控使 runner 就绪时机不对称;差异还在捕获来源(Dynamo 运行时捕获 vs `ExportedProgram`)与是否打包成 `.pt2`(关系已于该页 §14.2 note 核实)
