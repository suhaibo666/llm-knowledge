# 图编译系列迁移、历史审计与索引复核

> **复核日期**：2026-07-26  
> **知识库基线**：`llm-knowledge@c884b9d6b5cd702515e5f1b38aff236620af9865`，叠加当前未提交工作树  
> **设计基线**：`docs/superpowers/specs/2026-07-23-pytorch-graph-learning-series-design.md`  
> **复核范围**：设计 §7–§8、§10、§12 中与 coverage ledger、历史页面角色、6 个索引、回链和 changelog 有关的要求  
> **边界**：本报告只给出审计结论和逐文件修订清单，不改 wiki 正文。

## 1. 结论

当前输出在“新系列已建、4 个主索引已接入、Batch 0 已有机械 inventory 和 correction report”这三点上有实质进展，但**尚不满足设计的迁移验收门槛**。不能把现状表述为“无损迁移已完成”。

阻断验收的三个核心问题是：

1. **coverage ledger 还不是可闭环的迁移账本**：1,532 行中有 321 行目的地仍为 `TBD`，所有 1,532 行都没有目标章节锚点；实现阶段又取消了文件名前的数字前缀，导致 ledger 中除 3 个 `index.md` 行外，其余目的地字符串均不对应现存系列文件。
2. **19 个保留页面被统一加成 `[!deprecated] 历史材料`**，与设计 §8.2“保留领域专题、新系列不搬空专题页”冲突。这里需要的是“页面角色 + 基线 + 审计状态 + 课程导航”，不是整页废弃。
3. **设计要求修改的 6 个索引只完成了 4 个**：总索引、AOTAutograd、Inductor、FX/export 已改；runtime memory 索引未接入新系列；`17_compile_cache/index.md` 不存在。当前 changelog 自己也只声明“更新……四个索引”（`wiki/changelog.md:32`）。

### 1.1 按设计条款的状态

| 设计要求 | 当前证据 | 状态 |
|---|---|---|
| §7.1 审计至少覆盖列出的主干页及 runtime memory / activation checkpointing / compile cache 相关页 | manifest 有 22 页，但 compile cache 只含 2/4 页，activation checkpointing 为 0 页，runtime memory 只含 1 个 deep dive | **不通过** |
| §7.2 每个 H2/H3、代码块、图表、实验都有 ledger 去向 | 工具只把 `heading`、`code_fence`、`locator` 写入 ledger（`audit_graph_docs.py:417-419`）；没有 table/image/experiment/claim 行类型 | **不通过** |
| §7.2 `destination` 指向“新文档与章节” | 0/1,532 行含 `#anchor` 或 `§`；321 行为 `TBD`；仅 3 行目的地文件名在现目录直接存在 | **不通过** |
| §7.3 unresolved 不进入权威结论 | 526 行明确为 unresolved，Batch 0 也声明不得迁移（`batch_0_summary.md:48-60`） | **机制存在；后续迁移仍需逐项证明** |
| §7.5 索引、反链、changelog 同步 | 4/6 索引已改；runtime/cache 缺失；cache 页还有指向不存在页面的链接 | **不通过** |
| §8.1 综合报告保留全文并加“系统系列去向”表 | 全文保留；现有 `:547` 是“当前基线位置”源码表，不是旧章节→新章节去向表 | **不通过** |
| §8.1 AOT 全量页增加基线、审计状态、课程导航 | 只有整页 deprecated 提示；原基线仍是 unknown，且没有分角色课程导航 | **不通过** |
| §8.2 领域专题继续承担各自完整角色 | 19 页被统一标成历史/废弃，和索引中继续把它们当 deep dive 的描述互相冲突 | **不通过** |
| §8.3 修改 6 个索引和 changelog | 4 个索引 + changelog 已改；runtime/cache 未完成 | **部分通过** |
| §12.3 每个旧页所有结构单元有 ledger 去向 | manifest、记录种类、目的地和章节锚点均不完整 | **不通过** |
| §12.4 历史错漏有 correction report 和 changelog | 90 个 correction ID 已有报告；旧页本地 correction callout 基本未落地，changelog 只有汇总计数 | **部分通过** |
| §12.10 Related Pages、索引、回链全通过 | 两个已入 manifest 的 cache 页连规范标题 `## Related Pages` 都没有；compile cache index 缺失 | **不通过** |

## 2. Coverage ledger 的实质缺口

### 2.1 manifest 范围不足

设计 §7.1 明确写的是“至少包括”，并把 runtime memory、activation checkpointing 和 compile-cache 相关页面纳入范围（设计 `:528-550`）。现有 manifest 的确记录了 22 页，但尾部只加入：

- `caching_allocator_autocast_profiler_analysis.md`；
- `aotautograd_cache_analysis.md`；
- `fx_graph_cache_analysis.md`。

最小应补的明显同域页面是：

| 当前未入 manifest 的页面 | 为什么属于设计明示范围 |
|---|---|
| `wiki/01_theory/02_pretraining/activation_checkpointing_analysis.md` | 设计明确点名 activation checkpointing；也是 saved tensor / recompute 的历史解释来源 |
| `wiki/02_engineering/01_ai_frameworks/13_runtime_memory_amp_profiler/amp_and_memory_tooling_quickstart.md` | 与已入 manifest 的 runtime memory deep dive 同属一个模块，含实验和命令 |
| `wiki/02_engineering/01_ai_frameworks/17_compile_cache/dynamo_pgo_cache_analysis.md` | compile cache 目录现有四篇之一，解释上游动态形状画像 |
| `wiki/02_engineering/01_ai_frameworks/17_compile_cache/triton_autotune_cache_analysis.md` | compile cache 目录现有四篇之一，且与 Part IV autotune 直接相邻 |

还应显式判定 `04_inductor/inductor_memory_management_analysis.md` 与 `inductor_memory_allocation_guide.md` 是否纳入；它们与新系列 doc19 的编译期 buffer reuse / peak-memory reorder 有直接交集。若不纳入，ledger 或审计总览必须写清楚排除理由，不能让“runtime memory 相关页面”成为未定义集合。

### 2.2 parser 没有实现设计所说的审计单位

当前 parser 会抽取 heading、locator、wikilink 和 code fence（`audit_graph_docs.py:90-150`），但生成 ledger 时主动丢弃 wikilink，并且只接受三类记录：

```text
heading | code_fence | locator
```

证据见 `docs/audits/pytorch_graph_series/tools/audit_graph_docs.py:417-435`。

因此当前“1,532 条可审计记录”准确地说只是**标题 / 代码围栏 / 定位符账本**，不能据此证明以下设计要求已满足：

- Markdown 表格逐项映射；
- 非 Mermaid 图片/图表逐项映射；
- 实验的命令、输入、预期输出和实际状态逐项映射；
- 章节内多条互相独立的非平凡 claim 逐项映射。

修订方式：

1. parser 新增至少 `table`、`image_or_figure`、`experiment` 三种 record；实验不能只靠标题，需记录命令块与输出块的绑定。
2. 对非平凡 claim 增加人工 decision row，或把一个 heading 的 claim 列表展开，不能用一个章节级状态替代章节内所有断言。
3. 单元测试加入“表格、图片、实验命令+输出、同节多 claim”四类样例。
4. 重新生成 inventory / ledger，再对当前页面 hash 和结构计数做一次最终快照。

### 2.3 destination 字段尚未形成可导航映射

对当前 `coverage_ledger.md` 的 1,532 行逐行解析，结果是：

| 检查项 | 结果 |
|---|---:|
| ledger 行数 | 1,532 |
| `destination == TBD` | 321 |
| `status == unresolved` | 526 |
| destination 含章节锚点 `#...` | 0 |
| destination 含章节标记 `§` | 0 |
| destination 文件名直接存在于当前系列目录 | 3 行 |

这是两个独立问题：

1. 设计要求 destination 为“新文档与章节”（设计 `:554-568`），当前没有任何章节级目的地。
2. ledger 使用设计期数字前缀，例如 `09_aotautograd_joint_forward_backward_graphs.md`；实际文件是 `09_aotautograd_joint_forward_backward_graphs.md`。这不是 Obsidian 别名，而是不存在的文件名。

必须做一次 ledger 迁移：

- 去掉 01–21 数字前缀并绑定当前真实文件名；
- `13_patternexpr_and_patternmatcher.md`、`13_pattern_expression_and_matcher_engine.md` 统一到 `13_pattern_expression_and_matcher_engine.md`；
- `12_fx_graph_editing_primitives.md` 统一到 `12_fx_graph_editing_primitives_and_invariants.md`；
- `00_pytorch_graph_series_index.md` 统一到 `index.md`；
- `20_debugging_observability_and_verification_labs.md` 在当前 21 篇中不存在，必须逐行重分配到具体正文和具体 Lab，不能机械改名；
- 321 个 `TBD` 必须变成明确目标或明确写成“保留在旧领域专题，不迁入主线”，并给出旧页章节锚点；
- 每条 destination 至少采用 `actual_file.md#actual-heading-anchor`，随后机械验证文件和标题都存在。

### 2.4 correction report 有了，但旧页本地纠错未闭环

Batch 0 已产出 90 个 correction ID（49 P0、18 P1、4 P2、19 P3，见 `batch_0_summary.md:88`），这是可复用资产。但在 22 个 manifest 页面中：

- 21 页没有任何 `[!correction]`；
- 只有 `inductor_codegen_dynamic_shape_analysis.md` 有 1 个 `[!contradiction]`；
- 19 页的统一 `[!deprecated]` 不能替代具体错误的 correction callout。

设计 §10 要求修改旧页的“导航、审计状态、correction callout 和 Related Pages”（设计 `:707-711`）。正确落地方式是：

1. 页首给出页面级审计摘要：原基线、当前审计基线、verified/corrected/unresolved 状态和 correction ID。
2. 对会误导读者的 P0/P1 错误，在原错误附近加 `[!correction]`，写出当前结论和新系列目标页；不能只在页首说“冲突时看新页”。
3. P2/P3 可在页首 correction table 汇总，但仍需能从 correction ID 回到旧章节和新章节。
4. unresolved 内容必须明确标成“未验证，不可作为当前实现依据”，而不是把整页粗粒度标成 deprecated。

## 3. Blanket deprecated banner 与设计角色冲突

扫描 manifest 页前 15 行，19/22 页出现统一的 `[!deprecated]`；除 `fx_graph_export_and_custom_ops_analysis.md:1` 外，其余均在 `:3`。设计 §8.2 却明确要求保留：

- FX/export/custom-op/functorch 专题；
- advanced dynamic shape；
- decomposition / pre / joint / post 的 pass 目录与注册 API；
- pass methodology；
- lowering / scheduler / codegen / autotuning 完整源码参考；
- runtime memory / compile cache / activation checkpointing 独立专题。

所以应统一把顶部 banner 改成 `[!note] 页面角色与审计状态`，再根据页面角色写具体内容；不要把“不是课程主线”误写成“整页已废弃”。

### 3.1 AOTAutograd 与综合报告：逐文件建议

| 文件 | 当前问题 | 精确修订 |
|---|---|---|
| `03_aot_autograd/aotautograd_analysis.md:3` | 整页标历史材料；没有固定原基线、审计状态、课程分流 | 改成“保留的 AOTAutograd 全量 reference / edge-case 集合”；原基线标 `baseline-unknown`（页面只写 2026-04-09）；当前审计基线写 `e8f97c…`；列出 A-001/002/003/004/005/006/007/010/011/012/013/014/015/017/019；课程导航分别指向 effect、normalization、joint/fw/bw、saved/recompute、stage-boundary 五页；P0/P1 在原章节附近加 correction |
| `03_aot_autograd/aot_autograd_quickstart.md:3` | 可用 API 入门被整页判废弃；代码块未全部按当前环境验证 | 改成“保留的 API quick start，历史代码示例需看审计状态”；列 A-004/006/007/009/010/020；链接 joint/fw/bw 与 saved/recompute 的正式 Lab；未重跑代码块逐块标 `unverified`，不能靠页级 banner 代替 |
| `03_aot_autograd/fx_graph_construction_and_transformation_analysis.md:3` | 设计要求的“综合报告快照”被写成 deprecated；`:547` 的表是源码导航，不是迁移去向 | 改成“2026-07-23 问答形成的综合报告快照”；保留旧基线 `ea5655…`，并声明当前课程基线 `e8f97c…`；新增下述“系统系列去向”表；列出 A-006/007/008/016/019、F-010、P-002/004/005/008/010/011/012/015/016；在 `## Related Pages` 增加系列 index 与六个主目标页 |

综合报告应新增的去向表至少为：

| 原报告章节 | 新系列主目的地 |
|---|---|
| §1 核心结论 | `19_torch_compile_end_to_end/00_pytorch_graph_series_index.md` |
| §2 FX 对象模型、边与图序 | doc02 `02_fx_graph_core_data_model.md`；图序部分同时指向 doc14 |
| §3 joint→fw/bw 与跨图 ABI | doc09 `09_aotautograd_joint_forward_backward_graphs.md`；捕获前端和 provenance 只作为补充交叉引用 |
| §4 saved vs recompute、复制、reorder | doc10 `10_saved_tensors_recompute_and_runtime_abi.md` |
| §5–§6 PatternExpr、候选、匹配与替换 | doc13 `13_pattern_expression_and_matcher_engine.md` |
| §7–§8 dead、DCE、拓扑与 effect order | doc14 `14_dead_code_topology_and_effect_order.md` |
| §9–§10 复杂度、合法性与验证 | doc16 `16_graph_rewrite_legality_validation_and_complexity.md` |
| §11 源码导航 | 上述六页各自的“源码路径”章节 |

### 3.2 FX/export 与 dynamic shape：逐文件建议

| 文件 | 当前 correction IDs | 应保留的页面角色与回链 |
|---|---|---|
| `14_fx_export_and_extensibility/fx_graph_export_and_custom_ops_analysis.md:1` | F-008/009/016/017 | 保留 export、custom-op、functorch 的完整 deep dive；FX core 只指向新主线，不把全页标历史；Related Pages 加 capture、structured/HOP、FX core |
| `14_fx_export_and_extensibility/fx_export_custom_op_quickstart.md:3` | F-008/017/018 | 保留 API quick start；逐块标 current-run / unverified；回链 capture、FX core、editing invariants 和相应 Lab |
| `04_inductor/dynamic_shapes_full_analysis.md:3` | F-001–007、F-019 | 保留 advanced symbolic-shape 专题；新系列 doc04 负责主线概念，本页继续承载 ShapeEnv/guard 纵深；不能标整页 deprecated |
| `04_inductor/inductor_codegen_dynamic_shape_analysis.md:3` | F-006/012/013/014/015 | 保留 codegen dynamic-shape 专题；把已有 contradiction 保留并补 page-level audit summary；回链 doc04、doc18、doc21 |

### 3.3 Pass、Inductor IR、Scheduler、Codegen：逐文件建议

| 文件 | 当前 correction IDs | 应保留的页面角色与回链 |
|---|---|---|
| `04_inductor/decomposition_passes_guide.md:3` | P-019/021 | decomposition 逐 pass 目录和注册 API；回链 normalization + pass pipeline |
| `04_inductor/pre_grad_passes_guide.md:3` | P-006/010/017/021 | pre-grad 阶段目录、注册和扩展点；回链 editing + pipeline + legality |
| `04_inductor/joint_graph_passes_guide.md:3` | P-011/017/021 | joint 阶段目录与注册 API；回链 AOT joint graph + pipeline |
| `04_inductor/post_grad_passes_guide.md:3` | P-012/014/015/017/021 | post-grad 阶段目录、注册 API 与尾部不变量；回链 editing + DCE/order + pipeline + legality |
| `04_inductor/torch_upstream_pass_deepdive.md:3` | P-001/006/009/010/011/015/017/020/021 | 当前上游 pass 全集/目录参考；回链 matcher + pipeline；具体 stale catalog 在原位置纠错 |
| `04_inductor/fx_pass_optimization_methodology.md:3` | P-006/010/011/015/017/019/020/021 | 保留跨项目 pass 工程方法论；新系列提供事实主线，本页提供选层与验证方法 |
| `04_inductor/lowering_analysis.md:3` | I-001/005/006/008 | 保留 lowering 完整源码参考；回链 doc17 + doc18；已有阶段顺序纠错应改成正式 `[!correction]` |
| `04_inductor/scheduler_analysis.md:3` | I-009/010/015/016/017/018/023/024 | 保留 scheduler 完整源码参考；回链 doc19 + doc20 + doc21 |
| `04_inductor/inductor_codegen_analysis.md:3` | I-018/022/024/025 | 保留 codegen 完整源码参考；回链 doc20 + doc21 |
| `04_inductor/inductor_autotuning_analysis.md:3` | I-025/026 | 保留 autotuning 专题；回链 doc21 与 compile-cache index |
| `04_inductor/PyTorch_Inductor_Technical_Analysis.md:3` | I-001/007/013/017/019/021/022/023/024/025/027/028/029/030 | 保留 Inductor 纵向综合参考/模块快照；用角色说明取代历史材料；按 lowering/memory/scheduler/codegen 四路课程导航 |
| `04_inductor/torch_compile_architecture.md:3` | I-001/007/015/018/022/024/029/030 | 保留模块 overview；它与新系列是 overview ↔ curriculum 的关系，不是旧页废弃；回链系列 index 与 compile-cache index |

### 3.4 runtime/cache：逐文件建议

| 文件 | 当前问题 | 精确修订 |
|---|---|---|
| `13_runtime_memory_amp_profiler/caching_allocator_autocast_profiler_analysis.md` | ledger 中 87/87 行均 `unresolved + TBD`；没有新系列回链 | 页首声明独立 runtime allocator/AMP/profiler 专题，不能当作 Inductor compile-time liveness 的证据；Related Pages 加 doc19 和 doc10，并用一句话划清“saved activation / Inductor buffer / allocator block”三层 |
| `17_compile_cache/aotautograd_cache_analysis.md` | 82/82 行均 `unresolved + TBD`；标题是非规范的 `Related / Cross-references`；链接到不存在的 index/overview/megacache 页 | 先加入扩展 manifest 并审计；将 heading 改为 `## Related Pages`；把 `compile_cache_overview` 回链统一到新建的 `17_compile_cache/index`；删除或创建 `megacache_and_precompile_analysis`，不能留 dangling；回链 doc09/doc11/doc21 |
| `17_compile_cache/fx_graph_cache_analysis.md` | 90/90 行均 `unresolved + TBD`；同样有 dangling links | 同上；角色是 Inductor graph artifact cache，不与 Dynamo PGO/AOT result cache 混合；回链 doc04/doc11/doc21 |
| `17_compile_cache/dynamo_pgo_cache_analysis.md` | 未入 manifest；缺规范 `Related Pages`；指向缺失 index/overview/megacache | 纳入审计；角色是 shape decision/profile cache，不是 compiled artifact；回链 doc04 与新 cache index |
| `17_compile_cache/triton_autotune_cache_analysis.md` | 未入 manifest；缺规范 `Related Pages`；指向缺失 index/overview/megacache | 纳入审计；角色是 kernel winner / Triton artifact cache；回链 doc21 与新 cache index |
| `01_theory/02_pretraining/activation_checkpointing_analysis.md` | 设计点名但未入 manifest；没有新课程回链 | 纳入审计；保留通用/训练框架 activation checkpoint 专题；页首划清 eager checkpoint 与 AOT min-cut recompute；回链 doc10 |

## 4. 六个索引的逐文件修订

### 4.1 `01_ai_frameworks/index.md`

当前 `:56` 已加入系列主入口，这是正确的；但 `:44-56` 的编译栈目录没有 `17_compile_cache/index`，而 `:101` 只在规划说明里说“17 为 compile cache”。

修订：

- 在“torch.compile 编译栈 + 运行时分发”表加入 `[[17_compile_cache/index]]`；
- 描述为“跨阶段缓存：Dynamo PGO 决策画像、AOTAutograd result、Inductor FX graph artifact、Triton autotune/kernel”；
- 在阅读顺序中说明 doc21 之后可进入 compile-cache，避免把 cache 当成新 IR 阶段；
- `最后更新` 改为本次最终验证日期。

### 4.2 `03_aot_autograd/index.md`

当前 `:19-23` 已加三篇课程入口，但“旧 deep dive 适合追溯历史解释”的总括会把本应保留的 AOT 全量参考降格成纯历史页。

修订：

- 将三份旧页分别标为“quick start / 全量 reference+edge cases / 问答综合报告快照”；
- 添加“课程主线与专题参考的分工”表，而不是统一“冲突时新页为准”；
- 加 compile-cache index，说明 AOTAutograd cache 跳过的是哪段构图/分图工作；
- 与三份旧页的 page-role callout 双向一致。

### 4.3 `04_inductor/index.md`

当前 `:14` 和 `:92-94` 已接入系列 Part IV，方向正确；问题是索引仍把 lowering/scheduler/codegen 等列为有效 deep dive，而页面顶部又统一写 deprecated，语义冲突。

修订：

- 增加一段“课程主线 vs 保留的子系统源码参考”，明确两者互补；
- 补 doc18 `18_inductor_ir_values_loops_layouts_and_buffers` 和 doc19 `19_buffer_liveness_memory_planning_and_reuse` 的直链；
- 加 `17_compile_cache/index`，并把 autotuning 与 cache 分开：前者讲搜索/选择，后者讲结果与 artifact 复用；
- 页面表中的旧 deep dive 描述与各页新 role callout 保持一致。

### 4.4 `14_fx_export_and_extensibility/index.md`

当前 `:113-128` 同时列出 FX/export 专题和新课程，基本结构正确；但旧专题顶部的 deprecated 与这里的 deep-dive 定位冲突。

修订：

- 把 `fx_graph_export_and_custom_ops_analysis` 明确为“保留的 export/custom-op/functorch 专题”，不是“旧版 FX core 权威页”；
- 把 quickstart 明确为 API 实操入口，并指向当前验证 Lab；
- 系列直链补 structured/HOP 与 symbolic-shape 页面，保持 export 的参数/buffer lifting、nested graph、constraint 主线可导航；
- 反向检查两份旧页的 `Related Pages`。

### 4.5 `13_runtime_memory_amp_profiler/index.md`

当前 `:92-116` 完全没有新系列链接。

修订：

- 页面列表或“关联域”增加 doc19：编译期 buffer liveness / reuse / peak-memory；
- 增加 doc10：AOT saved tensor / recompute；
- 用一个三行边界表区分：
  - AOT activation 是否跨 fw/bw 保存；
  - Inductor buffer 何时 live、何时可复用；
  - runtime caching allocator 如何管理实际 device blocks；
- 给 `caching_allocator_autocast_profiler_analysis` 加对应反链。

### 4.6 新建 `17_compile_cache/index.md`

当前文件不存在，这是设计 §8.3/§10 的硬缺口。建议 index 本身承担 overview，避免继续保留不存在的 `[[compile_cache_overview]]`。

最小结构：

1. 为什么缓存不是一层：cache 的 key、value、命中后跳过阶段不同。
2. 四层比较表：Dynamo PGO / AOTAutograd result / FX graph artifact / Triton autotune+kernel。
3. 按编译生命周期的读路径。
4. 四篇现存页面列表及各自角色。
5. 与新系列 doc04、doc09、doc11、doc21 的边界和回链。
6. `## Related Pages`。

同时把四篇 cache 页中的 `[[compile_cache_overview]]` 改为 `[[17_compile_cache/index]]`。`[[megacache_and_precompile_analysis]]` 当前也不存在；要么在本次范围内补页，要么暂时移除该 link 并把主题作为 index 中的“待建专题”纯文本记录，零 dangling 验收前不能保留不存在链接。

## 5. changelog 修订

`wiki/changelog.md:24-32` 已记录系列、22 页 inventory、90 个 correction 和环境限制，这部分应保留。但要修正四点：

1. `:32` 的“更新四个索引”必须在实际完成六个索引后改成精确六项列表。
2. “为 19 篇旧版主干资料增加历史提示”要改成“增加页面角色、基线、审计状态、targeted correction 和课程回链”；领域专题不是整体 deprecated。
3. `:29` 的 1,532 行应标成“heading/code/locator ledger rows”，不能让读者误以为已覆盖表格、图片、实验和全部 claim。
4. 在 correction 汇总后记录审计 artifact 的仓库路径，并列出最关键的纠错主题；旧页本地 callout 完成前，不能把 §12.4 记为完成。

最终 changelog 还应使用**最终重跑结果**更新 Lab 数、artifact 数和环境状态，避免保留 `:30` 的旧“13 个 Lab”计数。

## 6. 建议实施顺序与复核门槛

1. 先扩展 manifest/parser，补 table/image/experiment/claim 单元并重生成 ledger。
2. 修正所有 destination 到当前真实文件名和真实章节锚点，清空 `TBD`；若内容保留在旧专题，destination 就写旧专题的精确章节并标 `retain-domain-page`。
3. 把 19 个 blanket deprecated banner 改成角色化 note；按 correction ID 给 P0/P1 加原位 callout。
4. 给综合报告补“系统系列去向”表，给 AOT 全量页补基线/状态/课程导航。
5. 修改 6 个索引；新建 compile-cache index；修复 cache 目录 dangling links。
6. 最后更新 changelog，不能先写完成声明再补事实。

只有在下面检查全部为零/全通过后，迁移部分才可判定完成：

```text
manifest 显式范围遗漏                         = 0
H2/H3/code/table/figure/experiment 未入 ledger = 0
destination == TBD                          = 0
destination 文件不存在                      = 0
destination 章节锚点不存在                  = 0
P0/P1 无旧页原位 correction                 = 0
领域专题仍被 blanket deprecated             = 0
6 个索引未更新                              = 0
dangling wikilink                           = 0
非规范或缺失的 ## Related Pages             = 0
git diff --check                            = pass
```

在这些项完成前，设计 §12.3、§12.4、§12.10 应保持 **fail / partial**，不能在总验收中标 green。
