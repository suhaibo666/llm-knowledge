# D04 · 编译缓存层级、Key、Guard 与失效边界

> 卷别：D · 编译产物、缓存与运行时  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[d03_async_compile_workers_and_module_loading_analysis]]  
> 后续：[[d05_wrapper_execution_memory_allocation_and_reuse_analysis]]  
> 最后更新：2026-07-28

## 1. 为什么不存在一个统一的“compile cache”

不同阶段的输入、产物和可复用条件不同：

| 层 | 输入身份 | Value | 典型失效 |
|---|---|---|---|
| Dynamo code cache | code object + runtime guards + backend | transformed code | guard/backend miss |
| AOTAutograd cache | Dynamo graph/AOT config/inputs | fw/bw compile/runtime metadata | graph/config/guard变化 |
| Inductor FXGraphCache | post-grad graph/inputs/system config | serializable `CompiledFxGraph` | key/shape guard/env miss |
| source/module cache | generated source + flags | Python module/shared lib | source/flags/ABI/path变化 |
| Triton kernel cache | kernel source + environment | future/binary/kernel | source/backend/toolchain变化 |
| autotune cache | op/kernel/config/device | best candidate/timing |候选/precision/device/config变化 |
| CUDA Graph runtime cache | function/int key/memory path | recorded graph | address/liveness/invariant变化 |

**核心结论**：一次“cache hit”必须带层名；上层hit可能仍需下层load/runtime包装，上层miss也
可能在下层命中已生成产物。

## 2. Dynamo cache与FXGraphCache的根本差异

Dynamo cache挂在Python code object上，通过可执行guard predicates选择transformed
bytecode；FXGraphCache接收一张已经捕获并规范化的FX graph，计算content/system key并在
磁盘目录中寻找带shape guard的compiled graph。

前者解决“这个Python frame能否复用捕获”；后者解决“这张backend graph能否复用codegen
产物”。

## 3. FXGraphCache key包含什么

策略注释说明：

- 收集GraphModule、graph inputs、system settings等形成 `FxGraphCacheDetails`；
- pickle并hash为key；
- 同key下可保存多个guard版本；
- lookup遍历leaf files并评估symbol guards。

见 `torch/_inductor/codecache.py:1993-2018`。

content hash基础函数把code和extra组合进策略key
（`torch/_inductor/codecache.py:478-507`）。但FX graph key不是只hash最终source；它在
codegen前就必须覆盖影响编译结果的graph/config/environment。

## 4. 为什么key与guard要同时存在

完全把所有动态shape值写进key会导致每个shape都成为不同目录，无法复用动态compiled
graph。完全只用guards又会让无关graphs落入同一候选集。

所以两级策略是：

```text
粗粒度稳定身份 → hash key目录
同一目录内不同symbol约束 → guard expression entries
```

FXGraphCache注释明确允许同graph多个guard版本
（`torch/_inductor/codecache.py:2010-2018`）。

## 5. Lookup怎样处理symbolic shapes

`_lookup_graph`：

1. 获取当前ShapeEnv；
2. 过滤backed SymInts并提取hints；
3. 遍历guarded entries；
4. 评估serialized guard expression；
5. 校验extern libraries key；
6. 命中后重新向当前ShapeEnv加入guards；
7. 执行cache-hit post compile。

见 `torch/_inductor/codecache.py:2185-2209`、
`torch/_inductor/codecache.py:2220-2239` 和
`torch/_inductor/codecache.py:2241-2253`。

重新加入guards很关键：cache hit跳过了原编译，却不能跳过对上层正确性域的约束传播。

## 6. `CompiledFxGraph`为何可序列化但callable不可序列化

`CompiledFxGraph`保存：

- cache/source/linemap；
- device、mutation、constant metadata；
- output strides；
- guards expression和extern library key；
- provenance；
- cudagraph metadata；
- FX compile kwargs。

字段见 `torch/_inductor/output_code.py:516-545`、
`torch/_inductor/output_code.py:546-550` 与
`torch/_inductor/output_code.py:551-572`。

序列化前它清空C++/Triton/Python callable，保留其PyCodeCache磁盘位置和可重建metadata
（`torch/_inductor/output_code.py:999-1016`）。

## 7. Cache hit为什么仍要load和post-compile

反序列化后：

1. 确保generated source存在于预期path；
2. `PyCodeCache.load_by_key_path`重载module；
3. 重新附加constants；
4. 取module `call`；
5. 恢复partition runner；
6. 再做input alignment、output stride和CUDAGraph包装。

callable恢复见 `torch/_inductor/output_code.py:1018-1047`、
`torch/_inductor/output_code.py:1048-1048` 与
`torch/_inductor/output_code.py:1049-1057`。

`post_compile`注释明确说它hit/miss后都会运行，且结果本身不保存在cache中
（`torch/_inductor/output_code.py:842-856`）。

## 8. Artifact被删除时为何退化为miss

FXGraph metadata可能仍存在，但PyCodeCache source或下层artifact被外部清理。
`cache_hit_post_compile`捕获load时的OSError并把它当cache miss重新编译
（`torch/_inductor/codecache.py:2112-2124`）。

这说明“metadata hit”不等于“artifact load hit”。

## 9. Local与remote cache

local cache提供低延迟、与本机toolchain紧密相关的复用；remote cache用于跨进程/机器共享。
remote key仍必须覆盖：

- PyTorch/compiler版本；
- backend/device/toolchain；
- relevant configs；
- graph与constants语义；
- dynamic guard表达式；
- extern libraries。

remote命中后的artifact仍需在本机落盘/load；若环境key或guard不匹配必须miss。

## 10. AOTAutograd cache为什么在FXGraphCache之上

AOT cache输入接近Dynamo graph + AOT config，value包含：

- fw/bw partition结果；
- runtime wrapper metadata；
- deeper Inductor cache references；
- lazy/compiled backward状态。

它可跳过functionalization、joint tracing和partition，但如果深层artifact丢失，仍可能需要
Inductor load/recompile。反之AOT miss产生相同post-grad graph时，FXGraphCache可能命中。

## 11. Invalidation不是一个广播事件

- `torch._dynamo.reset()`影响code cache/backend状态；
- 清in-memory module cache不一定删disk source；
- `PyCodeCache.cache_clear(purge=True)`才尝试删除已跟踪source
  （`torch/_inductor/codecache.py:4838-4853`）；
- worker future cache有自己的clear；
- FXGraphCache目录有自己的clear；
- Triton/driver cache可能由外部系统管理；
- CUDA Graph recording随进程/device/memory path生命周期存在。

因此调试“清缓存”必须列出目标层。

## 12. Cache安全不变量

- 所有影响codegen的配置都应进入key或被明确guard；
- dynamic shape guard在load时评估并回注当前ShapeEnv；
- constant绑定不能被同source module错误共享；
- backend/toolchain/extern libs变化必须miss；
- 序列化内容不能保存进程私有callable/pointer；
- load失败转miss，不能执行半恢复对象；
- unsafe skip guard选项可能破坏正确性，不应作为常规加速；
- 远程内容须视为不可信序列化边界并受部署策略控制。

## 13. 复杂度

设key序列化对象大小 \(H\)，同key候选entries数 \(C\)，artifact大小 \(A\)：

- key构造/hash约为 \(O(H)\)；
- guarded lookup worst case \(O(C \cdot Q)\)，\(Q\)为guard表达式成本；
- local deserialize/load约与entry/source/artifact大小相关；
- remote lookup另加网络延迟与传输 \(O(A)\)；
- hit仍支付module import、constant attach和post-compile；
- cache空间随graph版本、guard版本、toolchain/config组合增长。

## 14. 常见误解

- **“Dynamo hit就不会进入Inductor cache。”** transformed code可能调用已装载callable；新
  进程/新Dynamo捕获则可能查询深层cache。
- **“FXGraphCache key就是FX graph文本。”** 还包括inputs和系统/编译配置。
- **“动态shape不能用hash cache。”** hash选粗粒度身份，guard区分symbol适用域。
- **“metadata文件存在就是完整hit。”** source/binary可能已删除或load失败。
- **“一次reset清所有层。”** 每层有独立所有权和失效API。

## 配套 Demo

本页对应卷级入口 `tools/labs_torch_compile/demo_d_artifact_runtime.py` 的 `cache_keys_invalidation` 用例。默认以 CUDA 为验收设备：

```powershell
python -B tools\labs_torch_compile\demo_d_artifact_runtime.py `
  --case cache_keys_invalidation --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\d04
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `cache_keys_invalidation/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[guards_cache_lookup_and_recompilation_analysis]]
- [[d03_async_compile_workers_and_module_loading_analysis]]
- [[d05_wrapper_execution_memory_allocation_and_reuse_analysis]]
- [[d07_compiled_artifact_lifecycle_and_runtime_failures_analysis]]
- [[02_compile_stack/06_compile_cache/index]]
