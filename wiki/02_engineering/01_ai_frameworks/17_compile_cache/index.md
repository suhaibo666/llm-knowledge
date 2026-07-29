# 17 · torch.compile 跨阶段缓存 — 目录索引

> 层次：overview
> 当前课程源码基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> 历史专题页各自绑定其页头基线；不得把不同版本的key/schema混写。
> 最后更新：2026-07-27

## 1. 为什么“编译缓存”不是一层

`torch.compile`的不同阶段重复成本不同：Dynamo需要决定shape/guard策略，AOTAutograd需要
functionalize并构造/切分前反向图，Inductor需要lower与codegen，Triton还要选择config并
编译kernel。因此cache不是一张覆盖全栈的表，而是多层各自定义：

- 缓存输入是什么；
- key包含哪些语义与环境；
- value是决策画像、图级编译结果还是kernel产物；
- 命中时跳过哪段工作；
- guard、设备、版本和配置变化如何使entry失效。

只说“命中cache”没有诊断意义，必须给出命中的层。

## 2. 四层对照

| 层 | 主要缓存对象 | 命中后省掉/改善什么 | 不代表什么 |
|---|---|---|---|
| Dynamo PGO | dynamic-shape行为画像、CodeState相关决策输入 | 冷启动第一次编译就采用更合适的动态策略，减少试错重编译 | 不会直接返回compiled graph，编译仍会发生 |
| AOTAutograd result | Dynamo图对应的前反向编译单元/下游graph key或产物 | 可跳过functionalization、joint trace、partition及下游重复编译 | 不等于所有runtime wrapper状态都可忽略 |
| Inductor FxGraph artifact | post-grad图指纹、guards、config/device与`CompiledFxGraph` | 跳过lowering、Scheduler、codegen及相关Triton编译 | 仍需entry guard选择与runtime wrapper |
| Triton autotune/kernel | kernel候选winner、launch config与compiled binary | 跳过重复benchmark或`triton.compile` | 不等于上层FX/AOT graph cache已命中 |

版本相关的精确key、bypass和remote-cache字段必须进入对应专题页核对，不能仅凭此overview
推断当前实现。

## 3. 按编译生命周期阅读

```text
Python调用
  → Dynamo读取/更新PGO画像
  → AOTAutograd cache查找前反向编译单元
  → Inductor FxGraphCache查找post-grad graph artifact
  → template/Triton autotune cache查winner
  → code/kernel cache装载compiled artifact
  → runtime guards与wrapper执行
```

cache位于既有阶段的入口/出口，不是新的IR阶段。调试时从最上层开始记录hit/miss/bypass，
再向下确认：上层hit可能让下层完全没有新日志；上层miss也不妨碍下层hit。

## 4. 专题页与保留角色

| 页面 | 页面角色 |
|---|---|
| [[dynamo_pgo_cache_analysis]] | PGO画像的状态合并、local/remote/sticky key与“soundly stale”边界 |
| [[aotautograd_cache_analysis]] | AOT级key、entry形态、命中wrapper链与bypass条件 |
| [[fx_graph_cache_analysis]] | post-grad图指纹、GuardedCache、CompiledFxGraph与Triton bundling |
| [[triton_autotune_cache_analysis]] | kernel级autotune winner、remote cache与Triton磁盘cache |

这些是保留的deep dive，不是废弃材料；但其历史行号和schema尚未全部迁移到当前课程基线。
尚缺独立、当前基线核验的Mega-cache/precompile专题，现阶段按明确知识缺口处理，不创建
虚假wikilink。

## 5. 与图编译课程的边界

| 课程页 | cache视角 |
|---|---|
| [[19_torch_compile_end_to_end/04_symbolic_shapes_guards_and_graph_reuse]] | guard定义“同一graph entry何时可复用”；PGO影响第一次如何选择dynamic策略 |
| [[19_torch_compile_end_to_end/09_aotautograd_joint_forward_backward_graphs]] | AOT cache hit可能跳过joint/fw/bw的重新构造，但不改变其语义ABI |
| [[19_torch_compile_end_to_end/11_graph_stage_boundaries_identity_and_provenance]] | cache加载会让对象identity、dump存在性与本次编译阶段更加不连续 |
| [[19_torch_compile_end_to_end/21_codegen_kernel_mapping_autotuning_and_provenance]] | autotune是搜索/选择过程，cache是winner/artifact复用；两者不能合并成一个概念 |

## 6. 审计边界

本索引完成的是导航与概念分层，不声明四份历史专题已逐 claim 重核。2026-07-27
复核时，历史材料的 4,155 个结构单元中已有 1,602 个写入精确 destination，但它们全部仍是
`unresolved / retain-quarantined`；另有 2,553 个结构单元尚无决策，其中包含全部 2,160 个
claim candidate。也就是说，旧的 `TBD destination` 统计已被更精确的 destination 取代，
但语义迁移仍未闭环。最终状态原见
`docs/audits/pytorch_graph_series/2026-07-26/design_conformance_review.md`；该目录属审计流水线中间产物，已在 kb-reorg 清理中移出工作区（可经 git 历史追溯，删除前末次提交 `1ebafb5`），当前 checkout 不再包含该路径。

## Related Pages

- [[19_torch_compile_end_to_end/00_torch_compile_end_to_end_index]] — 编号化端到端课程：卷 B 的 guard/cache 与卷 D 的多层 artifact cache
- [[01_ai_frameworks/index]] — PyTorch编译与运行时总地图
- [[02_dynamo/index]] — Dynamo捕获、guard与dynamic shape
- [[03_aot_autograd/index]] — AOTAutograd构图与partition
- [[04_inductor/index]] — Inductor lowering、Scheduler与codegen
- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]] — 图编译系统化课程
- [[19_torch_compile_end_to_end/04_symbolic_shapes_guards_and_graph_reuse]]
- [[19_torch_compile_end_to_end/09_aotautograd_joint_forward_backward_graphs]]
- [[19_torch_compile_end_to_end/11_graph_stage_boundaries_identity_and_provenance]]
- [[19_torch_compile_end_to_end/21_codegen_kernel_mapping_autotuning_and_provenance]]
