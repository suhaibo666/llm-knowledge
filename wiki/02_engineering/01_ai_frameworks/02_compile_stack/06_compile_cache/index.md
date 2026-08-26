# 06 · torch.compile 跨阶段缓存 — 目录索引

> 层次：overview → deep dive
> overview 正文固定源码基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`（2026-07-30 kb-reorg P4 Task 6 从 D04 `compile_cache_hierarchy_keys_and_invalidation_analysis` 迁入，并改写为本页 overview 主体；原索引的四层导航表保留于 §15 起）
> §17 的各篇专题页分别绑定其页头基线；不得混写不同版本的 key/schema。
> 最后更新：2026-07-30

`torch.compile` 各阶段的重复成本不同：Dynamo 需要确定 shape/guard 策略，AOTAutograd 需要
执行 functionalization 并构造、切分前反向图，Inductor 需要执行 lowering 与 codegen，Triton 还要选择 config 并
编译 kernel。因此，cache 不是一张覆盖全栈的表，而是由各层分别定义：

- 缓存输入是什么；
- key 包含哪些语义与环境信息；
- value 是决策画像、图级编译结果还是 kernel 产物；
- 命中时会跳过哪一段工作；
- guard、设备、版本和配置变化会如何使 entry 失效。

只说“命中 cache”没有诊断意义，必须指出具体命中了哪一层。

## 1. 为什么不存在统一的“compile cache”

不同阶段的输入、产物和复用条件各不相同：

| 层 | 输入身份 | Value | 典型失效 |
|---|---|---|---|
| Dynamo code cache | code object + runtime guards + backend | transformed code | guard/backend miss |
| AOTAutograd cache | Dynamo graph/AOT config/inputs | fw/bw compile/runtime metadata | graph/config/guard变化 |
| Inductor FXGraphCache | post-grad graph/inputs/system config | serializable `CompiledFxGraph` | key/shape guard/env miss |
| source/module cache | generated source + flags | Python module/shared lib | source/flags/ABI/path变化 |
| Triton kernel cache | kernel source + environment | future/binary/kernel | source/backend/toolchain变化 |
| autotune cache | op/kernel/config/device | best candidate/timing |候选/precision/device/config变化 |
| CUDA Graph runtime cache | function/int key/memory path | recorded graph | address/liveness/invariant变化 |

**核心结论**：描述一次“cache hit”时必须指明层级；上层 hit 后可能仍需执行下层 load/runtime 包装，上层 miss 时也
可能在下层命中已生成产物。

## 2. Dynamo cache 与 FXGraphCache 的根本差异

Dynamo cache 挂在 Python code object 上，通过可执行的 guard predicates 选择 transformed
bytecode；FXGraphCache 接收已经捕获并规范化的 FX graph，计算 content/system key，并在
磁盘目录中查找带有 shape guard 的 compiled graph。

前者解决“这个 Python frame 能否复用捕获结果”的问题；后者解决“这张 backend graph 能否复用 codegen
产物”的问题。

## 3. FXGraphCache key 包含什么

策略注释说明：

- 收集 GraphModule、graph inputs、system settings 等信息，形成 `FxGraphCacheDetails`；
- pickle 后计算 hash，得到 key；
- 同一 key 下可保存多个 guard 版本；
- lookup 会遍历 leaf files，并评估 symbol guards。

见 `torch/_inductor/codecache.py:1993-2018`。

content hash 基础函数会把 code 和 extra 组合到策略 key 中
（`torch/_inductor/codecache.py:478-507`）。但 FX graph key 并非只对最终 source 计算 hash；它在
codegen 前就必须覆盖会影响编译结果的 graph/config/environment。

## 4. 为什么 key 与 guard 要同时存在

如果把所有动态 shape 值都写入 key，每种 shape 都会对应不同目录，因而无法复用动态 compiled
graph；如果完全依赖 guards，又会让无关 graphs 落入同一候选集。

因此采用两级策略：

```text
粗粒度稳定身份 → hash key目录
同一目录内不同symbol约束 → guard expression entries
```

FXGraphCache 的注释明确允许同一 graph 存在多个 guard 版本
（`torch/_inductor/codecache.py:2010-2018`）。

## 5. Lookup 如何处理 symbolic shapes

`_lookup_graph`:

1. 获取当前 ShapeEnv；
2. 过滤 backed SymInts 并提取 hints；
3. 遍历 guarded entries；
4. 评估 serialized guard expression；
5. 校验 extern libraries key；
6. 命中后重新向当前 ShapeEnv 加入 guards；
7. 执行 cache-hit post compile。

见 `torch/_inductor/codecache.py:2185-2209`、
`torch/_inductor/codecache.py:2220-2239` 和
`torch/_inductor/codecache.py:2241-2253`。

重新加入 guards 至关重要：cache hit 虽然跳过了原编译过程，却不能跳过向上层传播正确性约束。

## 6. 为什么 `CompiledFxGraph` 可序列化，而 callable 不可序列化

`CompiledFxGraph` 保存：

- cache/source/linemap；
- device、mutation、constant metadata；
- output strides；
- guards expression 和 extern library key；
- provenance；
- cudagraph metadata；
- FX compile kwargs。

字段见 `torch/_inductor/output_code.py:516-545`、
`torch/_inductor/output_code.py:546-550` 与
`torch/_inductor/output_code.py:551-572`。

序列化前，它会清空 C++/Triton/Python callable，只保留 PyCodeCache 磁盘位置和可重建的 metadata
（`torch/_inductor/output_code.py:999-1016`）。

## 7. 为什么 Cache hit 后仍要执行 load 和 post-compile

反序列化后，还要执行以下操作：

1. 确保 generated source 位于预期 path；
2. 通过 `PyCodeCache.load_by_key_path` 重新加载 module；
3. 重新附加 constants；
4. 获取 module 的 `call`；
5. 恢复 partition runner；
6. 再执行 input alignment、output stride 和 CUDAGraph 包装。

callable恢复见 `torch/_inductor/output_code.py:1018-1047`、
`torch/_inductor/output_code.py:1048-1048` 与
`torch/_inductor/output_code.py:1049-1057`。

`post_compile` 的注释明确指出，无论 hit 还是 miss 都会运行该步骤，而且其结果本身不保存在 cache 中
（`torch/_inductor/output_code.py:842-856`）。

## 8. Artifact 被删除时为何会退化为 miss

FXGraph metadata 可能仍然存在，但 PyCodeCache source 或下层 artifact 已被外部清理。
`cache_hit_post_compile` 会捕获 load 时的 OSError，并将其视为 cache miss 后重新编译
（`torch/_inductor/codecache.py:2112-2124`）。

这说明“metadata hit”不等于“artifact load hit”。

## 9. Local cache 与 remote cache

local cache 提供低延迟、与本机 toolchain 紧密相关的复用；remote cache 则用于跨进程或跨机器共享。
remote key 仍必须覆盖：

- PyTorch/compiler 版本；
- backend/device/toolchain；
- relevant configs；
- graph 与 constants 的语义；
- dynamic guard 表达式；
- extern libraries。

remote 命中后的 artifact 仍需在本机落盘并执行 load；若环境 key 或 guard 不匹配，则必须判定为 miss。

## 10. 为什么 AOTAutograd cache 位于 FXGraphCache 之上

AOT cache 的输入接近 Dynamo graph + AOT config，value 包含：

- fw/bw partition 结果；
- runtime wrapper metadata；
- deeper Inductor cache references；
- lazy/compiled backward状态。

它可以跳过 functionalization、joint tracing 和 partition；但如果深层 artifact 丢失，仍可能需要
Inductor load/recompile。反之，AOT miss 产生相同的 post-grad graph 时，FXGraphCache 仍可能命中。

## 11. Invalidation 不是广播事件

- `torch._dynamo.reset()` 会影响 code cache/backend 状态；
- 清理 in-memory module cache 不一定会删除 disk source；
- 只有 `PyCodeCache.cache_clear(purge=True)` 才会尝试删除已跟踪的 source
  （`torch/_inductor/codecache.py:4838-4853`）；
- worker future cache 有自己的 clear；
- FXGraphCache 目录有自己的 clear；
- Triton/driver cache 可能由外部系统管理；
- CUDA Graph recording 随进程/device/memory path 的生命周期存在。

因此，调试“清缓存”问题时必须明确目标层。

## 12. Cache 安全不变量

- 所有影响 codegen 的配置都应进入 key，或由明确的 guard 约束；
- dynamic shape guard 在 load 时评估，并回注当前 ShapeEnv；
- constant 绑定不能在同一 source module 中被错误共享；
- backend/toolchain/extern libs 变化时必须判定为 miss；
- 序列化内容不能保存进程私有的 callable/pointer；
- load 失败时应转为 miss，不能执行仅恢复了一部分的对象；
- unsafe skip guard 选项可能破坏正确性，不应作为常规加速手段；
- 远程内容须视为不可信序列化边界并受部署策略控制。

## 13. 复杂度

设 key 序列化对象大小为 $H$，同一 key 下的候选 entries 数为 $C$，artifact 大小为 $A$：

- key 构造/hash 约为 $O(H)$；
- guarded lookup 的 worst case 为 $O(C \cdot Q)$，其中 $Q$ 是 guard 表达式的计算成本；
- local deserialize/load 的成本约与 entry/source/artifact 大小相关；
- remote lookup 还会增加网络延迟和 $O(A)$ 的传输成本；
- hit 后仍需承担 module import、constant attach 和 post-compile 的成本；
- cache 空间随 graph 版本、guard 版本及 toolchain/config 组合增长。

## 14. 常见误解

- **“Dynamo hit 就不会进入 Inductor cache。”** transformed code 可能调用已装载的 callable；新
  进程或新的 Dynamo 捕获则可能查询深层 cache。
- **“FXGraphCache key 就是 FX graph 文本。”** 它还包括 inputs 和系统/编译配置。
- **“动态 shape 不能使用 hash cache。”** hash 选择粗粒度身份，guard 用于区分 symbol 适用域。
- **“metadata 文件存在就表示完整 hit。”** source/binary 可能已删除，load 也可能失败。
- **“一次 reset 会清理所有层。”** 每层都有独立的所有权和失效 API。

## 配套 Demo

本页（原 D04）对应卷级入口 `tools/labs_torch_compile/demo_d_artifact_runtime.py` 的 `cache_keys_invalidation` 用例，默认以 CUDA 为验收设备：

```powershell
python -B tools\labs_torch_compile\demo_d_artifact_runtime.py `
  --case cache_keys_invalidation --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\d04
```

先使用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可将 `--device` 改为 `cpu`，以探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `cache_keys_invalidation/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不能外推到其他 PyTorch 版本、shape、dtype 或硬件。

---

## 15. 四层速查（教学视角对照）

上面 §1 的七层是源码级精确划分；下表则从入门视角归并为四层，两者并不矛盾。PGO 与
Dynamo code cache 同属“捕获前决策”层，只是 §1 没有单列 PGO（见 §17 topic page）。

| 层 | 主要缓存对象 | 命中后省掉/改善什么 | 不代表什么 |
|---|---|---|---|
| Dynamo PGO | dynamic-shape行为画像、CodeState相关决策输入 | 冷启动第一次编译就采用更合适的动态策略,减少试错重编译 | 不会直接返回compiled graph,编译仍会发生 |
| AOTAutograd result | Dynamo图对应的前反向编译单元/下游graph key或产物 | 可跳过functionalization、joint trace、partition及下游重复编译 | 不等于所有runtime wrapper状态都可忽略 |
| Inductor FxGraph artifact | post-grad图指纹、guards、config/device与`CompiledFxGraph` | 跳过lowering、Scheduler、codegen及相关Triton编译 | 仍需entry guard选择与runtime wrapper |
| Triton autotune/kernel | kernel候选winner、launch config与compiled binary | 跳过重复benchmark或`triton.compile` | 不等于上层FX/AOT graph cache已命中 |

版本相关的精确 key、bypass 和 remote-cache 字段必须进入对应专题页核对，不能仅凭此概览
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

cache 位于既有阶段的入口或出口，并不是新的 IR 阶段。调试时从最上层开始记录 hit/miss/bypass，
再逐层向下确认：上层 hit 可能使下层完全没有新日志；上层 miss 也不妨碍下层 hit。

## 17. 专题页与保留角色

> **段位与阅读顺序**（kb-reorg P4 Task 9.5，2026-07-30）：段 1（10-19）的四篇文章均介绍核心机制，不区分 quickstart、深潜或方法论层级；按照 §16 的编译生命周期排序（捕获前决策 PGO → AOT 结果 → Inductor FX graph artifact → Triton kernel 级），顺序如下表。overview 主体已写入本页（index.md 本身，2026-07-30 从旧 D04 页迁入并改写），不单独占页，因此不编号。

| 页面 | 页面角色 |
|---|---|
| [[10_dynamo_pgo_cache_analysis]] | PGO画像的状态合并、local/remote/sticky key与"soundly stale"边界 |
| [[11_aotautograd_cache_analysis]] | AOT级key、entry形态、命中wrapper链与bypass条件 |
| [[12_fx_graph_cache_analysis]] | post-grad图指纹、GuardedCache、CompiledFxGraph与Triton bundling |
| [[13_triton_autotune_cache_analysis]] | kernel级autotune winner、remote cache与Triton磁盘cache |

这些是保留的 deep dive，并非废弃材料；但其中的历史行号和 schema 尚未全部迁移到当前课程基线。
目前仍缺少基于当前基线独立核验的 Mega-cache/precompile 专题，现阶段将其作为明确的知识缺口处理，不创建
虚假 wikilink。

## 18. 与图编译课程的边界

| 课程页 | cache视角 |
|---|---|
| [[20_symbolic_shapes_guards_and_graph_reuse_analysis]] | guard定义"同一graph entry何时可复用";PGO影响第一次如何选择dynamic策略 |
| [[11_aotautograd_joint_forward_backward_graphs_analysis]] | AOT cache hit可能跳过joint/fw/bw的重新构造,但不改变其语义ABI |
| [[20_graph_stage_boundaries_identity_and_provenance_analysis]] | cache加载会让对象identity、dump存在性与本次编译阶段更加不连续 |
| [[14_codegen_kernel_mapping_autotuning_and_provenance_analysis]] | autotune是搜索/选择过程,cache是winner/artifact复用;两者不能合并成一个概念 |

## 19. 审计边界

本索引只完成导航与概念分层，并不表示四份历史专题已逐条重新核验 claim。2026-07-27
复核时，历史材料的 4,155 个结构单元中已有 1,602 个写入精确 destination，但它们仍全部处于
`unresolved / retain-quarantined` 状态；另有 2,553 个结构单元尚无决策，其中包含全部 2,160 个
claim candidate。也就是说，旧的 `TBD destination` 统计已被更精确的 destination 取代，
但语义迁移仍未闭环。最终状态原见
`docs/audits/pytorch_graph_series/2026-07-26/design_conformance_review.md`；该目录属于审计流水线的中间产物，已在 kb-reorg 清理中移出工作区（可通过 git 历史追溯，删除前的末次提交为 `1ebafb5`），当前 checkout 不再包含该路径。

## Related Pages

- [[courses/torch_compile_end_to_end]] — 编号化端到端课程：卷 B 的 guard/cache、图编译系统化课程与卷 D 的多层 artifact cache
- [[01_ai_frameworks/index]] — PyTorch 编译与运行时总地图
- [[02_compile_stack/01_dynamo/index]] — Dynamo 捕获、guard 与 dynamic shape
- [[02_compile_stack/02_aot_autograd/index]] — AOTAutograd 构图与 partition
- [[02_compile_stack/04_inductor/index]] — Inductor lowering、Scheduler 与 codegen
- [[20_symbolic_shapes_guards_and_graph_reuse_analysis]]
- [[11_aotautograd_joint_forward_backward_graphs_analysis]]
- [[20_graph_stage_boundaries_identity_and_provenance_analysis]]
- [[14_codegen_kernel_mapping_autotuning_and_provenance_analysis]]
- [[15_guards_cache_lookup_and_recompilation_analysis]] — Dynamo guard 树与 recompile 决策（本页 §2 的 Dynamo cache 一侧）
