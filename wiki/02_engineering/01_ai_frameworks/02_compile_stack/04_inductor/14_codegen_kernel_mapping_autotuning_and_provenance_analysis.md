# 21 · Codegen、Kernel 映射、Autotuning 与 Provenance

> 前置：[[13_scheduler_dependency_graph_fusion_and_ordering_analysis]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. Codegen的双层产物

```text
kernel side:
  fused loop/template/extern → backend kernel/source/call

wrapper side:
  guards、allocation/reuse、device/stream context、
  kernel launch、outputs、subgraph calls
```

`DeviceCodegen`注册scheduling与Python/C++/FX wrapper constructors
（`torch/_inductor/codegen/common.py:309-318`）。

这两个产物必须分层，是因为 kernel body 只知道一次计算怎样执行，而整个 compiled graph
还要维护跨 kernel 的运行协议。源码把它们分别放进 `DeviceCodegen.scheduling` 与
`wrapper_codegen/cpp_wrapper_codegen/fx_wrapper_codegen`
（`torch/_inductor/codegen/common.py:309-318`）；backend registration 同时接收这两类
constructor，而不是只注册一个 kernel printer
（`torch/_inductor/codegen/common.py:410-434`）。

## 2. Backend contract

kernel code由Scheduling实现；host code由Wrapper实现。`register_backend_for_device`还支持
optional custom graph pass/config
（`torch/_inductor/codegen/common.py:389-434`）。

新增backend不是只实现一个printer；至少要处理scheduling+wrapper contract。

## 3. 当前built-in dispatch

- CPU：config选择Cpp/Halide/Triton/Pallas；
- CUDA Triton choice：`CUDACombinedScheduling`；
- XPU：`XPUCombinedScheduling`；
- MPS：`MetalScheduling`；
- MTIA：`TritonScheduling`
  （`torch/_inductor/codegen/common.py:515-613`）。

因此“所有GPU都直接TritonScheduling”不准确。

## 4. GraphLowering.codegen

顺序：

1. init wrapper；
2. update/create Scheduler；
3. draw original FX/debug；
4. Scheduler.codegen；
5. wrapper.generate
  （`torch/_inductor/graph.py:2991-3008`）。

随后 `compile_to_module`才把wrapper source编译/加载为module
（入口见`torch/_inductor/graph.py:3050-3067`；写入、cache load与module装载见
`torch/_inductor/graph.py:3091-3155`）。

codegen、source compile、runtime execute是三个边界。

### 4.1 从最终 Scheduler 到可调用 module 的源码链

```text
GraphLowering.codegen()
  → init_wrapper_code()
  → _update_scheduler()
  → Scheduler.codegen()          # 各 group 生成 kernel/call 与 wrapper lines
  → wrapper_code.generate()      # 生成 host wrapper source

GraphLowering.compile_to_module()
  → codegen() / codegen_with_cpp_wrapper()
  → _compile_to_module_lines()
  → PyCodeCache.write()
  → PyCodeCache.load_by_key_path()
  → CompiledModule
```

第一段调用链见 `torch/_inductor/graph.py:2991-3008`；第二段入口见
`torch/_inductor/graph.py:3050-3073`，Python wrapper 的写入与模块装载见
`torch/_inductor/graph.py:3116-3145`及
`torch/_inductor/graph.py:3146-3155`。

因此：

- `output_code.py`存在，只证明 wrapper source 已生成；
- `PyCodeCache`模块已加载，不自动证明其中每个 native kernel 已执行；
- runtime 调用成功也可能只走 extern/fallback，而不是 generated kernel。

这三个边界必须在实验与报告中分别标注。

## 5. Scheduler group到kernel

Scheduler `_codegen`遍历最终nodes，管理：

- device/context flush；
- stream/mempool；
- alignment copy；
- last usage/free；
- template/extern/foreach/nested/mix/fused普通node dispatch；
- backend flush与wrapper lines。

入口：`torch/_inductor/scheduler.py:9749-9764`与
`torch/_inductor/scheduler.py:9920-9949`、
`torch/_inductor/scheduler.py:9950-9974`、
`torch/_inductor/scheduler.py:9976-10005`、
`torch/_inductor/scheduler.py:10006-10034`、
`torch/_inductor/scheduler.py:10036-10064`、
`torch/_inductor/scheduler.py:10065-10087`及
`torch/_inductor/scheduler.py:10089-10117`。

Scheduler group是一次backend codegen decision的输入，不存在“一group必然对应一个
native kernel”的不变量。extern可变成library call，foreach/combo/partition/nested
reduction与backend自身也会改变entry point、translation unit和launch基数。

## 6. Loop codegen

backend将loop IR转换为：

- loop order；
- tiling/block sizes；
- program id→offset；
- index expressions；
- masks；
- loads/stores；
- reduction strategy。

Triton与C++共享上游IR/Scheduler，但具体loop/kernel classes与heuristics不同。

## 7. Template与Extern分支

- generated pointwise/reduction：backend scheduling生成；
- template：专用matmul/conv/flex等template；
- extern：ATen/vendor/custom library call；
- fallback：extern family；
- user-defined Triton：可有特殊fusion/codegen。

不要承诺matmul一定是Triton或一定单独kernel；algorithm/config决定。

## 8. 两层autotuning

### AlgorithmSelectorCache

在GEMM/conv-like operation层选择extern/template implementations，负责persistent cache、
precompile、benchmark
（`torch/_inductor/select_algorithm.py:3949-3978`、
`torch/_inductor/select_algorithm.py:3979-4004`、
`torch/_inductor/select_algorithm.py:4021-4040`、
`torch/_inductor/select_algorithm.py:4041-4057`、
`torch/_inductor/select_algorithm.py:4077-4106`、
`torch/_inductor/select_algorithm.py:4113-4142`、
`torch/_inductor/select_algorithm.py:4154-4183`、
`torch/_inductor/select_algorithm.py:4190-4208`、
`torch/_inductor/select_algorithm.py:4238-4255`、
`torch/_inductor/select_algorithm.py:4391-4408`、
`torch/_inductor/select_algorithm.py:4410-4424`、
`torch/_inductor/select_algorithm.py:4458-4486`、
`torch/_inductor/select_algorithm.py:4487-4507`、
`torch/_inductor/select_algorithm.py:4511-4537`、
`torch/_inductor/select_algorithm.py:4541-4561`、
`torch/_inductor/select_algorithm.py:4563-4578`、
`torch/_inductor/select_algorithm.py:4580-4594`、
`torch/_inductor/select_algorithm.py:4613-4624`、
`torch/_inductor/select_algorithm.py:4638-4665`、
`torch/_inductor/select_algorithm.py:4673-4690`、
`torch/_inductor/select_algorithm.py:4705-4718`）。

可返回 `MultiTemplateBuffer`，把winner延迟到Scheduler benchmark fusion
（`torch/_inductor/select_algorithm.py:4190-4208`）。

源码注释明确其 key 取决于输入 sizes、strides、dtypes 等，而不依赖 output layout，并且
同一对象还维护 precompile 与 prescreening cache
（`torch/_inductor/select_algorithm.py:3949-3988`）。调用入口先预处理 choices，并可按
CPU/custom input 等条件关闭 deferred multi-template
（`torch/_inductor/select_algorithm.py:4021-4057`）。

### CachingAutotuner

对一个generated Triton kernel选择launch config：

- precompile configs；
- benchmark launchers；
- select one；
- optional coordinate descent；
- cache steady-state launcher
  （`torch/_inductor/runtime/triton_heuristics.py:531-545`;
  `torch/_inductor/runtime/triton_heuristics.py:546-561`;
  `torch/_inductor/runtime/triton_heuristics.py:576-603`;
  `torch/_inductor/runtime/triton_heuristics.py:628-647`;
  `torch/_inductor/runtime/triton_heuristics.py:648-675`;
  `torch/_inductor/runtime/triton_heuristics.py:719-747`;
  `torch/_inductor/runtime/triton_heuristics.py:763-791`;
  `torch/_inductor/runtime/triton_heuristics.py:793-818`;
  `torch/_inductor/runtime/triton_heuristics.py:1789-1818`;
  `torch/_inductor/runtime/triton_heuristics.py:1819-1848`;
  `torch/_inductor/runtime/triton_heuristics.py:1849-1859`;
  `torch/_inductor/runtime/triton_heuristics.py:2412-2440`;
  `torch/_inductor/runtime/triton_heuristics.py:2441-2470`;
  `torch/_inductor/runtime/triton_heuristics.py:2471-2500`;
  `torch/_inductor/runtime/triton_heuristics.py:2501-2530`;
  `torch/_inductor/runtime/triton_heuristics.py:2531-2550`）。

两者candidate、cache key、timing point不同。

`CachingAutotuner`要求非空 Triton configs，保存设备和 kernel metadata
（`torch/_inductor/runtime/triton_heuristics.py:531-561`）；
`benchmark_all_configs()`逐 launcher 实测并保留当前 winner
（`torch/_inductor/runtime/triton_heuristics.py:1789-1821`）；
steady state 又把唯一 launcher 缓存在 `_cached_launcher`，直接进入 fast path
（`torch/_inductor/runtime/triton_heuristics.py:2412-2439`）。

所以“算法实现 winner”与“同一 Triton kernel 的 launch-config winner”是两次不同选择：
前者改变 IR/template/extern implementation，后者不改变该 kernel 的逻辑语义，只改变如何
启动。

## 9. Candidate list不能写死

pointwise/reduction configs来自device heuristic registry；templates有各自search space。
current runtime grid可依赖symbolic numel，而block config来自compile-time selection
（`torch/_inductor/runtime/triton_heuristics.py:4365-4435`;
`torch/_inductor/runtime/triton_heuristics.py:5005-5032`;
`torch/_inductor/runtime/triton_heuristics.py:5078-5099`;
`torch/_inductor/runtime/triton_heuristics.py:5101-5115`;
`torch/_inductor/runtime/triton_heuristics.py:5118-5134`;
`torch/_inductor/runtime/triton_heuristics.py:5137-5152`）。

固定BLOCK/XBLOCK列表只在绑定具体backend/version/generator时成立。

## 10. Autotune failure与fallback

Algorithm selection若timings empty优先选extern；全部inf也可回退extern
（`torch/_inductor/select_algorithm.py:4210-4255`）。

CachingAutotuner运行时若0 launcher先precompile，>1则benchmark到1，再launch
（`torch/_inductor/runtime/triton_heuristics.py:2473-2519`）。

这再次说明两层fallback不同。

## 11. Wrapper不是“没有Python”

默认JIT路径生成Python wrapper；它减少的是eager每op Python/dispatcher/launch边界，而不是
保证执行栈完全无Python。AOTInductor可用C++ wrapper/共享库，但属于不同产物模式。

## 12. Allocation与launch

wrapper包含：

- input guards/symbol definitions；
- Allocate/Free/Reuse；
- device guard/stream；
- extern/template/generated kernel call；
- output alias/reinterpret；
- runtime asserts；
- benchmark/autotune helpers。

因此生成kernel源码只看计算body，不能解释完整runtime ABI与内存行为。

## 13. Compile cache

当前源码中至少要区分以下缓存域：

| 缓存 | key/命中条件的核心 | 缓存的对象 | 源码锚点 |
|---|---|---|---|
| `FxGraphCache` | graph module、inputs、system settings 的 hash，再验证 symbolic guards | compiled FX graph metadata与底层 artifact 位置 | `torch/_inductor/codecache.py:1993-2018` |
| `PyCodeCache` | generated Python source 写入得到的 key/path | Python module 与 line map | `torch/_inductor/codecache.py:4767-4795` |
| `CppCodeCache` | source + compilation settings 构成的编译缓存键 | 编译后的 C++ library loader/module | `torch/_inductor/codecache.py:3789-3806` |
| `AlgorithmSelectorCache` / `PersistentCache` | op、input characteristics、precision、choice hash | implementation benchmark timings/winner依据 | `torch/_inductor/codecache.py:412-450` |
| `CachingAutotuner` | generated Triton kernel 对应的 configs/cache metadata | 最佳 launcher/config与 steady-state launcher | `torch/_inductor/runtime/triton_heuristics.py:531-554` |

这些缓存的 invalidation 与 guard 不同：FX graph cache hit 仍要验证 guards；Python/C++ code
cache围绕生成 source；algorithm cache围绕 choice timings；Triton autotuner围绕 launch
config。不能用“PyCodeCache缓存一切”或“同一个 graph key 控制所有层”概括。缓存专题见
[[02_compile_stack/06_compile_cache/12_fx_graph_cache_analysis]]。

## 14. Provenance链

```text
Python source
→ FX stack/source meta
→ IR origins
→ Scheduler node/group origins
→ wrapper/kernel line map
```

GraphLowering origin propagation：
`torch/_inductor/graph.py:1960-1989`。

IR origin fields：
`torch/_inductor/ir.py:589-645`。

Scheduler wrapper context选择：
`torch/_inductor/scheduler.py:9053-9069`。

显式pre/post/code映射与debug handle生成见
`torch/_inductor/debug.py:948-977`、
`torch/_inductor/debug.py:978-1007`、
`torch/_inductor/debug.py:1008-1037`、
`torch/_inductor/debug.py:1038-1044`、
`torch/_inductor/debug.py:1047-1076`、
`torch/_inductor/debug.py:1077-1104`、
`torch/_inductor/debug.py:1107-1136`、
`torch/_inductor/debug.py:1137-1142`、
`torch/_inductor/debug.py:1231-1256`、
`torch/_inductor/debug.py:1256-1281`、
`torch/_inductor/debug.py:1282-1311`、
`torch/_inductor/debug.py:1322-1345`、
`torch/_inductor/debug.py:1347-1358`、
`torch/_inductor/codegen/cpp.py:6086-6093`与
`torch/_inductor/codegen/wrapper.py:4262-4270`。

fusion group可对应多个FX origins；一FX Node也可产生多个generated lines/kernels。

## 15. Debug顺序

1. 保存Dynamo/post-grad FX；
2. 查看IR pre-fusion；
3. 查看IR post-fusion；
4. 找Scheduler group/buffer names；
5. 查看output code wrapper/kernel；
6. 对照origin/stack trace；
7. 查看autotune choices/timings/cache hit；
8. runtime profiler确认真正launch。

不要把post-grad FX dump称为fusion graph。

## 16. 复杂度

设最终 Scheduler group 数为 `G`，所有 loop/template/extern codegen 实际访问的 IR/body
规模之和为 `B`，wrapper line 数为 `L`，算法候选数为 `K_alg`，Triton config 数为
`K_cfg`。不含后端编译与 benchmark 时，结构生成骨架近似：

```text
O(G + B + L)
```

完整 wall time 更接近：

```text
T_total =
  O(G + B + L)
  + T_symbolic
  + Σ T_compile(generated source)
  + Σ_{i=1..K_alg} T_benchmark(i)
  + Σ_{j=1..K_cfg} [T_precompile(j) + T_benchmark(j)]
  + T_cache_lookup/load
```

其中 cache hit 会删除一部分 compile/benchmark 项，却不会让所有层都命中；FX graph cache
还可能遍历同一 graph hash 下的多个 guard variants 并逐一验证。wall time常由：

- symbolic/index simplification；
- C++/Triton compilation；
- `K`个algorithm/template/config benchmarks；
- subprocess/async precompile；
- cache miss；
- large generated source

主导。autotune winner 的 `min`选择本身仅 `O(K)`，但实际成本是各 candidate 的
compile+benchmark 之和；只写 `O(K)`会遗漏主导项。

## 17. 已验证 Lab

### 17.1 命令

```powershell
python tools/labs_torch_compile/part4_artifact_bundle.py `
  --output-dir tools/labs_torch_compile/artifacts/part4

python tools/labs_torch_compile/part4_ir_scheduler_analysis.py `
  --output-dir tools/labs_torch_compile/artifacts/part4_ir
```

### 17.2 实际结果

```text
external_matmul_execution=True
fallback_eigvals_execution=True
fallback_trace_captured=True
fallback_wrapper_observed=True
fusion_enabled_has_fused_scheduler=True
fusion_limited_has_fused_scheduler=False
fusion_codegen_structure_changed=True
custom_lowering_reached_ir=True
scheduler_to_fx_provenance_observed=True
kernel_to_fx_provenance_observed=True
fx_to_python_source_observed=True
scheduler_kernel_source_chain_observed=True
codegen_only_status=generated_not_executed
real_pointwise_compile_status=blocked_missing_msvc_cl
triton_autotune_tested=False
```

真实执行边界只有extern matmul与fallback `eigvals`。

当前Windows CPU缺少MSVC `cl`，所以generated pointwise kernel没有真实编译/执行。

为继续检查后续产物，codegen-only路径仅patch compiler探测并用no-op callable截获Inductor
生成的source。它走过真实GraphLowering、Scheduler、wrapper/source generation；wrapper与
no-op callable实际被调用，但generated C++未编译、未执行，返回值不具备计算语义。

原生证据由独立合同工具约束，不能靠手写 `PASS` JSON 补齐：

```powershell
python tools/labs_torch_compile/native_backend_contract.py probe `
  --output tools/labs_torch_compile/artifacts/native_backend/local_capability_diagnostic.json
```

完整 CPU/CUDA producer 与 validator 命令见
[`tools/labs_torch_compile/NATIVE_BACKEND_RUNBOOK.md`](tools/labs_torch_compile/NATIVE_BACKEND_RUNBOOK.md)。合同要求 producer
真实执行 compile/load/kernel、逐 workload 数值对照与 timing；GPU 还必须保存至少两个
参数不同的 candidate、真实 timing、winner、cache 与 allocator snapshot/trace。文件存在、
hash 正确或字符串 `status=PASS` 都不足以通过。

2026-07-27 当前主机实测 `cpu_native=BLOCKED`、`cuda_triton=BLOCKED`：缺 MSVC、CUDA 与
Triton，CPU/CUDA producer 都返回受控退出码 3，而不是伪造结果。

这个合同工具验证的是 `contract_native_cpp` / `contract_triton` 环境与证据格式；即使以后
在外部主机通过，也不能自动证明某个 artifact 来自 Inductor。关闭本页 native
Scheduler→kernel provenance 缺口，还必须把通过的执行结果与本页具体 generated source、
wrapper 和 Scheduler provenance 绑定。

### 17.3 fusion group到产物

`external_matmul`、`fallback_eigvals`、`fusion_enabled`、`fusion_limited`与
`custom_lowering`目录按各自路径保存可产生的：

- `fx_graph_readable.py`、`fx_graph_transformed.py`；
- `ir_pre_fusion.txt`、`ir_post_fusion.txt`；
- `inductor_provenance_tracking_node_mappings.json`；
- `output_code.py`；
- `captured_cpp_kernel.cpp`；
- `provenance_chain.json`。

`external_matmul`与`fallback_eigvals`目录来自真实执行trace，并不人为补造
`captured_cpp_kernel.cpp`。`fallback_eigvals/ir_pre_fusion.txt`中的`FallbackKernel`把
unsupported-op结果从“数值跑通”提升为可检查的lowering产物。

`fusion_enabled`、`fusion_limited`与`custom_lowering`目录中的C++ translation unit是
codegen-only截获，不是native执行产物。

fusion enabled/limited的Scheduler group数是`1/2`、captured C++ loop计数是`2/3`，但
两边`cpp_pybinding` entry point都为`1`。这证明Scheduler group数不能直接等同于生成的
C++ binding entry-point数；source hash与loop结构确实变化，但证据仍停留在codegen-only
边界。

由于source未做native compile/execute，这里没有测量native kernel数。

`part4_ir/scheduler_dependencies.json`和`fusion_comparison.json`补充Scheduler node/group、
operation names与buffer dependencies。

结合不同阶段对象会复制、折叠和融合这一实现事实，这组记录形成many-to-many证据；不能
假定存在一个可跨阶段复用的Node id。

### 17.4 provenance可以证明到哪一步

Lab在post-fusion custom pass中读取同一次编译的Scheduler group/subnode origins与
stack trace，再按FX node name和Inductor provenance JSON连接，生成
`fusion_enabled/provenance_chain.json`。已断言：

```text
Scheduler group/subnode
  → IR origins中的post-grad FX node
  → provenance JSON中的pre-grad FX node与C++ debug handle
  → FX stack_trace中的Python文件/行
```

这证明的是基于FX node name/debug handle的结构化join。

它不是runtime PC/profiler sample→source映射；实验没有运行generated C++，也没有GPU，
因而不能验证Triton launch-config winner、autotune timings或cache hit。

相关章节的autotune机制是`[S]`源码核验。

当前实验明确记录 `triton_autotune_tested=False`，不能把默认配置写成运行时已验证。

## 18. 最终心智模型

```text
Pattern pass决定FX子图形态
Lowering决定IR implementation候选
Realization决定schedulable operations
Scheduler决定dependencies、fusion、order、last use
Backend Scheduling生成kernel
Wrapper生成allocation与launch
Autotuners在不同层选择实现/config
Provenance把generated artifacts映回source
```

## 学习顺序

- 上一篇：[[13_scheduler_dependency_graph_fusion_and_ordering_analysis]]
- 返回总索引：[[00_pytorch_graph_series_index]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[20_graph_stage_boundaries_identity_and_provenance_analysis]]
- [[13_scheduler_dependency_graph_fusion_and_ordering_analysis]]
- [[12_buffer_liveness_memory_planning_and_reuse_analysis]]
- [[20_inductor_codegen_analysis]] — codegen 双层架构与 wrapper 段的源码级完整参考(本页 §1/§4/§12 的纵深版；§7 另有 CPU kernel 类体系专题)
- [[23_inductor_gpu_kernel_dispatch_model]] — 本页 §6"Loop codegen"未展开的 GPU kernel 骨架/IterationRanges/tiling/grid 类型
- [[22_inductor_reduction_codegen_deep_analysis]] — reduction codegen 专题(persistent/looped/split/cooperative)
- [[21_inductor_autotuning_analysis]] — 本页 §8"两层autotuning"的运行时纵深(CachingAutotuner 生命周期、config 启发式、Triton 编译链、§七 CoordescTuner)
- [[02_compile_stack/04_inductor/index]]
