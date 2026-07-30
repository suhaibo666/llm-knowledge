# 06 · torch.compile 跨阶段缓存 — 目录索引

> 层次:overview → deep dive
> overview 正文固定源码基线:PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`(2026-07-30 kb-reorg P4 Task 6 由 D04`compile_cache_hierarchy_keys_and_invalidation_analysis`迁入并改写为本页 overview 主体,原索引的四层导航表保留于 §15 起)
> §17 专题页表各自绑定其页头基线;不得把不同版本的 key/schema 混写。
> 最后更新:2026-07-30

`torch.compile`的不同阶段重复成本不同:Dynamo需要决定shape/guard策略,AOTAutograd需要
functionalize并构造/切分前反向图,Inductor需要lower与codegen,Triton还要选择config并
编译kernel。因此cache不是一张覆盖全栈的表,而是多层各自定义:

- 缓存输入是什么;
- key包含哪些语义与环境;
- value是决策画像、图级编译结果还是kernel产物;
- 命中时跳过哪段工作;
- guard、设备、版本和配置变化如何使entry失效。

只说"命中cache"没有诊断意义,必须给出命中的层。

## 1. 为什么不存在一个统一的"compile cache"

不同阶段的输入、产物和可复用条件不同:

| 层 | 输入身份 | Value | 典型失效 |
|---|---|---|---|
| Dynamo code cache | code object + runtime guards + backend | transformed code | guard/backend miss |
| AOTAutograd cache | Dynamo graph/AOT config/inputs | fw/bw compile/runtime metadata | graph/config/guard变化 |
| Inductor FXGraphCache | post-grad graph/inputs/system config | serializable `CompiledFxGraph` | key/shape guard/env miss |
| source/module cache | generated source + flags | Python module/shared lib | source/flags/ABI/path变化 |
| Triton kernel cache | kernel source + environment | future/binary/kernel | source/backend/toolchain变化 |
| autotune cache | op/kernel/config/device | best candidate/timing |候选/precision/device/config变化 |
| CUDA Graph runtime cache | function/int key/memory path | recorded graph | address/liveness/invariant变化 |

**核心结论**:一次"cache hit"必须带层名;上层hit可能仍需下层load/runtime包装,上层miss也
可能在下层命中已生成产物。

## 2. Dynamo cache与FXGraphCache的根本差异

Dynamo cache挂在Python code object上,通过可执行guard predicates选择transformed
bytecode;FXGraphCache接收一张已经捕获并规范化的FX graph,计算content/system key并在
磁盘目录中寻找带shape guard的compiled graph。

前者解决"这个Python frame能否复用捕获";后者解决"这张backend graph能否复用codegen
产物"。

## 3. FXGraphCache key包含什么

策略注释说明:

- 收集GraphModule、graph inputs、system settings等形成 `FxGraphCacheDetails`;
- pickle并hash为key;
- 同key下可保存多个guard版本;
- lookup遍历leaf files并评估symbol guards。

见 `torch/_inductor/codecache.py:1993-2018`。

content hash基础函数把code和extra组合进策略key
(`torch/_inductor/codecache.py:478-507`)。但FX graph key不是只hash最终source;它在
codegen前就必须覆盖影响编译结果的graph/config/environment。

## 4. 为什么key与guard要同时存在

完全把所有动态shape值写进key会导致每个shape都成为不同目录,无法复用动态compiled
graph。完全只用guards又会让无关graphs落入同一候选集。

所以两级策略是:

```text
粗粒度稳定身份 → hash key目录
同一目录内不同symbol约束 → guard expression entries
```

FXGraphCache注释明确允许同graph多个guard版本
(`torch/_inductor/codecache.py:2010-2018`)。

## 5. Lookup怎样处理symbolic shapes

`_lookup_graph`:

1. 获取当前ShapeEnv;
2. 过滤backed SymInts并提取hints;
3. 遍历guarded entries;
4. 评估serialized guard expression;
5. 校验extern libraries key;
6. 命中后重新向当前ShapeEnv加入guards;
7. 执行cache-hit post compile。

见 `torch/_inductor/codecache.py:2185-2209`、
`torch/_inductor/codecache.py:2220-2239` 和
`torch/_inductor/codecache.py:2241-2253`。

重新加入guards很关键:cache hit跳过了原编译,却不能跳过对上层正确性域的约束传播。

## 6. `CompiledFxGraph`为何可序列化但callable不可序列化

`CompiledFxGraph`保存:

- cache/source/linemap;
- device、mutation、constant metadata;
- output strides;
- guards expression和extern library key;
- provenance;
- cudagraph metadata;
- FX compile kwargs。

字段见 `torch/_inductor/output_code.py:516-545`、
`torch/_inductor/output_code.py:546-550` 与
`torch/_inductor/output_code.py:551-572`。

序列化前它清空C++/Triton/Python callable,保留其PyCodeCache磁盘位置和可重建metadata
(`torch/_inductor/output_code.py:999-1016`)。

## 7. Cache hit为什么仍要load和post-compile

反序列化后:

1. 确保generated source存在于预期path;
2. `PyCodeCache.load_by_key_path`重载module;
3. 重新附加constants;
4. 取module `call`;
5. 恢复partition runner;
6. 再做input alignment、output stride和CUDAGraph包装。

callable恢复见 `torch/_inductor/output_code.py:1018-1047`、
`torch/_inductor/output_code.py:1048-1048` 与
`torch/_inductor/output_code.py:1049-1057`。

`post_compile`注释明确说它hit/miss后都会运行,且结果本身不保存在cache中
(`torch/_inductor/output_code.py:842-856`)。

## 8. Artifact被删除时为何退化为miss

FXGraph metadata可能仍存在,但PyCodeCache source或下层artifact被外部清理。
`cache_hit_post_compile`捕获load时的OSError并把它当cache miss重新编译
(`torch/_inductor/codecache.py:2112-2124`)。

这说明"metadata hit"不等于"artifact load hit"。

## 9. Local与remote cache

local cache提供低延迟、与本机toolchain紧密相关的复用;remote cache用于跨进程/机器共享。
remote key仍必须覆盖:

- PyTorch/compiler版本;
- backend/device/toolchain;
- relevant configs;
- graph与constants语义;
- dynamic guard表达式;
- extern libraries。

remote命中后的artifact仍需在本机落盘/load;若环境key或guard不匹配必须miss。

## 10. AOTAutograd cache为什么在FXGraphCache之上

AOT cache输入接近Dynamo graph + AOT config,value包含:

- fw/bw partition结果;
- runtime wrapper metadata;
- deeper Inductor cache references;
- lazy/compiled backward状态。

它可跳过functionalization、joint tracing和partition,但如果深层artifact丢失,仍可能需要
Inductor load/recompile。反之AOT miss产生相同post-grad graph时,FXGraphCache可能命中。

## 11. Invalidation不是一个广播事件

- `torch._dynamo.reset()`影响code cache/backend状态;
- 清in-memory module cache不一定删disk source;
- `PyCodeCache.cache_clear(purge=True)`才尝试删除已跟踪source
  (`torch/_inductor/codecache.py:4838-4853`);
- worker future cache有自己的clear;
- FXGraphCache目录有自己的clear;
- Triton/driver cache可能由外部系统管理;
- CUDA Graph recording随进程/device/memory path生命周期存在。

因此调试"清缓存"必须列出目标层。

## 12. Cache安全不变量

- 所有影响codegen的配置都应进入key或被明确guard;
- dynamic shape guard在load时评估并回注当前ShapeEnv;
- constant绑定不能被同source module错误共享;
- backend/toolchain/extern libs变化必须miss;
- 序列化内容不能保存进程私有callable/pointer;
- load失败转miss,不能执行半恢复对象;
- unsafe skip guard选项可能破坏正确性,不应作为常规加速;
- 远程内容须视为不可信序列化边界并受部署策略控制。

## 13. 复杂度

设key序列化对象大小 \(H\),同key候选entries数 \(C\),artifact大小 \(A\):

- key构造/hash约为 \(O(H)\);
- guarded lookup worst case \(O(C \cdot Q)\),\(Q\)为guard表达式成本;
- local deserialize/load约与entry/source/artifact大小相关;
- remote lookup另加网络延迟与传输 \(O(A)\);
- hit仍支付module import、constant attach和post-compile;
- cache空间随graph版本、guard版本、toolchain/config组合增长。

## 14. 常见误解

- **"Dynamo hit就不会进入Inductor cache。"** transformed code可能调用已装载callable;新
  进程/新Dynamo捕获则可能查询深层cache。
- **"FXGraphCache key就是FX graph文本。"** 还包括inputs和系统/编译配置。
- **"动态shape不能用hash cache。"** hash选粗粒度身份,guard区分symbol适用域。
- **"metadata文件存在就是完整hit。"** source/binary可能已删除或load失败。
- **"一次reset清所有层。"** 每层有独立所有权和失效API。

## 配套 Demo

本页(原 D04)对应卷级入口 `tools/labs_torch_compile/demo_d_artifact_runtime.py` 的 `cache_keys_invalidation` 用例。默认以 CUDA 为验收设备:

```powershell
python -B tools\labs_torch_compile\demo_d_artifact_runtime.py `
  --case cache_keys_invalidation --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\d04
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制;CUDA/Triton/多卡专属用例会返回 `BLOCKED`,且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `cache_keys_invalidation/result.json`:`status` 区分 `PASS/BLOCKED/FAIL`,`environment` 固化运行环境,`observations` 保存本页机制的实测字段,`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过,不外推到其他 PyTorch 版本、shape、dtype 或硬件。

---

## 15. 四层速查(教学视角对照)

上面 §1 的七层是源码级的精确划分;下表是入门视角的四层归并,两者不矛盾——PGO 与
Dynamo code cache同属"捕获前决策"层,只是 §1 未单列 PGO(见 §17 topic page)。

| 层 | 主要缓存对象 | 命中后省掉/改善什么 | 不代表什么 |
|---|---|---|---|
| Dynamo PGO | dynamic-shape行为画像、CodeState相关决策输入 | 冷启动第一次编译就采用更合适的动态策略,减少试错重编译 | 不会直接返回compiled graph,编译仍会发生 |
| AOTAutograd result | Dynamo图对应的前反向编译单元/下游graph key或产物 | 可跳过functionalization、joint trace、partition及下游重复编译 | 不等于所有runtime wrapper状态都可忽略 |
| Inductor FxGraph artifact | post-grad图指纹、guards、config/device与`CompiledFxGraph` | 跳过lowering、Scheduler、codegen及相关Triton编译 | 仍需entry guard选择与runtime wrapper |
| Triton autotune/kernel | kernel候选winner、launch config与compiled binary | 跳过重复benchmark或`triton.compile` | 不等于上层FX/AOT graph cache已命中 |

版本相关的精确key、bypass和remote-cache字段必须进入对应专题页核对,不能仅凭此概览
推断当前实现。

## 16. 按编译生命周期阅读

```text
Python调用
  → Dynamo读取/更新PGO画像
  → AOTAutograd cache查找前反向编译单元
  → Inductor FxGraphCache查找post-grad graph artifact
  → template/Triton autotune cache查winner
  → code/kernel cache装载compiled artifact
  → runtime guards与wrapper执行
```

cache位于既有阶段的入口/出口,不是新的IR阶段。调试时从最上层开始记录hit/miss/bypass,
再向下确认:上层hit可能让下层完全没有新日志;上层miss也不妨碍下层hit。

## 17. 专题页与保留角色

| 页面 | 页面角色 |
|---|---|
| [[dynamo_pgo_cache_analysis]] | PGO画像的状态合并、local/remote/sticky key与"soundly stale"边界 |
| [[aotautograd_cache_analysis]] | AOT级key、entry形态、命中wrapper链与bypass条件 |
| [[fx_graph_cache_analysis]] | post-grad图指纹、GuardedCache、CompiledFxGraph与Triton bundling |
| [[triton_autotune_cache_analysis]] | kernel级autotune winner、remote cache与Triton磁盘cache |

这些是保留的deep dive,不是废弃材料;但其历史行号和schema尚未全部迁移到当前课程基线。
尚缺独立、当前基线核验的Mega-cache/precompile专题,现阶段按明确知识缺口处理,不创建
虚假wikilink。

## 18. 与图编译课程的边界

| 课程页 | cache视角 |
|---|---|
| [[19_torch_compile_end_to_end/04_symbolic_shapes_guards_and_graph_reuse]] | guard定义"同一graph entry何时可复用";PGO影响第一次如何选择dynamic策略 |
| [[19_torch_compile_end_to_end/09_aotautograd_joint_forward_backward_graphs]] | AOT cache hit可能跳过joint/fw/bw的重新构造,但不改变其语义ABI |
| [[19_torch_compile_end_to_end/11_graph_stage_boundaries_identity_and_provenance]] | cache加载会让对象identity、dump存在性与本次编译阶段更加不连续 |
| [[19_torch_compile_end_to_end/21_codegen_kernel_mapping_autotuning_and_provenance]] | autotune是搜索/选择过程,cache是winner/artifact复用;两者不能合并成一个概念 |

## 19. 审计边界

本索引完成的是导航与概念分层,不声明四份历史专题已逐 claim 重核。2026-07-27
复核时,历史材料的 4,155 个结构单元中已有 1,602 个写入精确 destination,但它们全部仍是
`unresolved / retain-quarantined`;另有 2,553 个结构单元尚无决策,其中包含全部 2,160 个
claim candidate。也就是说,旧的 `TBD destination` 统计已被更精确的 destination 取代,
但语义迁移仍未闭环。最终状态原见
`docs/audits/pytorch_graph_series/2026-07-26/design_conformance_review.md`;该目录属审计流水线中间产物,已在 kb-reorg 清理中移出工作区(可经 git 历史追溯,删除前末次提交 `1ebafb5`),当前 checkout 不再包含该路径。

## Related Pages

- [[19_torch_compile_end_to_end/00_torch_compile_end_to_end_index]] — 编号化端到端课程:卷 B 的 guard/cache 与卷 D 的多层 artifact cache
- [[01_ai_frameworks/index]] — PyTorch编译与运行时总地图
- [[02_compile_stack/01_dynamo/index]] — Dynamo捕获、guard与dynamic shape
- [[02_compile_stack/02_aot_autograd/index]] — AOTAutograd构图与partition
- [[02_compile_stack/04_inductor/index]] — Inductor lowering、Scheduler与codegen
- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]] — 图编译系统化课程
- [[19_torch_compile_end_to_end/04_symbolic_shapes_guards_and_graph_reuse]]
- [[19_torch_compile_end_to_end/09_aotautograd_joint_forward_backward_graphs]]
- [[19_torch_compile_end_to_end/11_graph_stage_boundaries_identity_and_provenance]]
- [[19_torch_compile_end_to_end/21_codegen_kernel_mapping_autotuning_and_provenance]]
- [[guards_cache_lookup_and_recompilation_analysis]] — Dynamo guard树与recompile决策(本页§2的Dynamo cache一侧)
