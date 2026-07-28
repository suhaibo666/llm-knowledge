# PyTorch 图编译器学习系列 Parts I–II 源码忠实度审计

> **结论**：**NOT ACCEPTED / 当前不能通过设计验收**  
> **设计规格**：`docs/superpowers/specs/2026-07-23-pytorch-graph-learning-series-design.md`，SHA256 `9EB2AD12ACED753F6E3CF89DEBE5DE00AE81A6A14EC7CC9604552843F9007603`  
> **PyTorch 源码基线**：`E:/97-codes/torch_parallel/p` @ `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`，detached、clean，commit date `2026-07-23`  
> **Wiki 工作树**：`E:/97-codes/torch_parallel/llm-knowledge`，Git HEAD `c884b9d6b5cd702515e5f1b38aff236620af9865c`；本系列在审计时为 untracked working-tree snapshot  
> **审计快照**：2026-07-26 09:48:31 +08:00  
> **Lab 复跑环境**：Python `3.13.5`；PyTorch `2.9.1+cpu`；`torch.version.git_version=5811a8d7da873dd699ff6687092c225caffcf1bb`

## 1. 审计范围与方法

本报告只审计设计规格的 §3.2、§3.5、§5.1–§5.3、§6、§12，并逐行阅读：

- `index.md`；
- Part I 的 01–06；
- Part II 的 07–11；
- `labs/part1_*.py` 与 `labs/part2_*.py`。

核验方法：

1. 将规格中的每个“包含”项和 Lab 目标拆成逐篇检查项；
2. 检查每篇是否形成“动机 → 机制 → 数据结构/不变量 → 真实调用链 → Lab 命令/预期/错误或边界/artifact → failure modes → 复杂度 → 上下游”的闭环；
3. 抽取 11 篇中的源码 locator：共 143 次引用、119 个唯一 locator；
4. 机械检查所有 locator 的文件存在性和范围：`missing_files=0`，`out_of_range=0`；
5. 对关键断言实际打开固定源码范围核验，而不把“文件和行号存在”当成“断言被支持”；
6. 实跑 9 个唯一的 Part I/II Lab；
7. 检查本范围内 128 个 wikilink、代码围栏与 `Related Pages`。

严重度定义：

- **P0**：设计验收阻断项；不修复就不能把系列称为完成或把实验称为端到端证据。
- **P1**：主要内容、实验或源码事实缺口；会让读者形成错误机制模型或无法复现关键结论。
- **P2**：局部完整性、可维护性或定位精度问题；不一定改变主结论，但削弱可验证性。

## 2. 总结

优点是明确的：11 篇大多先讲“为什么”，能区分图类型、值层次、alias/effect、joint/fw/bw 与阶段 identity；9 个现有脚本均能在声明的 Lab 环境退出 0；所有源码路径和行范围都存在；本范围内 wikilink、代码围栏和 `Related Pages` 机械检查通过。

但当前实现没有满足设计的两个中心承诺：

1. **没有统一贯穿模型，也没有端到端 artifact bundle。** 9 个 Part I/II Lab 使用彼此独立的函数、Module 或手工 Graph；第 11 篇明确写着 post-grad、Inductor IR、Scheduler、generated code 等“最终 Lab 还会追加”。
2. **Lab 七要素系统性不完整。** 11 篇均未给出可复制的 `python ...` 执行命令，均未给出 artifact 路径；多个设计指定的正例/边界例没有进入脚本。脚本“能运行”不等于规格中的 Lab 已完成。

此外，存在直接可证的文档/实现不一致：第 03 篇记录的 stdout 与实际运行不符；第 08、11 篇声称脚本输出/展示了脚本实际没有做的内容；第 09 篇把总 fw output leaves 标成 saved-output count；至少一处核心 partition locator 不能支持所述调用链，并且“该 helper 创建 GraphModule”的断言本身不成立。

## 3. P0 / P1 / P2 问题表

| ID | 级别 | 问题 | 证据 | 影响 | 修订方向 |
|---|---|---|---|---|---|
| P0-01 | P0 | §6 的统一贯穿模型不存在，端到端 artifact bundle 未生成 | 规格 §6：`design.md:501-524`；索引只宣称模型：`index.md:97-121`；9 个脚本分别定义不同模型；第 11 篇将后半链路写成未来项：`11_graph_stage_boundaries_identity_and_provenance.md:177-188`；Part I/II 脚本中没有 artifact 写出逻辑 | 无法把同一节点/语义从 Python 追到 joint/fw/bw，更无法证明后续 provenance；§12.7 失败 | 新建唯一 canonical model/fixture；所有 Lab 以 feature flag 或逐步 wrapper 复用；生成有 manifest 的 artifact bundle |
| P0-02 | P0 | Lab 七要素系统性不合格 | 规格 §3.5：`design.md:81-91`；11 篇均无明确 `python ...` 命令，均无 dump/log/artifact 定位；多个页面没有错误/边界例；第 11 篇没有本篇设计 Lab | §12.6 不能成立；读者无法机械复现或知道产物在哪里 | 为每篇增加统一 Lab contract：命令、cwd、环境、输入、expected、negative/boundary、artifact tree、实跑状态 |
| P1-01 | P1 | 文档记录与脚本实际行为不一致 | 第 03 篇写 `symbolic_fx_ops=placeholder,get_attr,call_function,output`（`03_graph_values_metadata_and_signatures.md:236-247`），实跑为 `placeholder,get_attr,call_function,get_attr,call_function,output`；第 08 篇称 `part2_normalization.py` 展示 `symbolic_trace`（`08_graph_normalization_decomposition_and_functionalization.md:184-191`），脚本未导入或调用它；第 11 篇称 `part2_aot_graphs.py` 输出 Node 表和 owner 信息（`11_graph_stage_boundaries_identity_and_provenance.md:177-180`），实际只输出计数、targets 和 `cross_refs` | “已验证”标签失真 | 以脚本 stdout/JSON 为单一真值；修正文档；增加断言和 golden-field 检查 |
| P1-02 | P1 | 第 05 篇缺少设计指定的 post-grad reinplace，Lab 也未覆盖指定反例 | 规格：`design.md:206-221`；正文中 `reinplace` 为 0 次；Lab 只验证 pure add 被删、`copy_` 被保留和 alias 可见（`part1_effects_alias.py:7-32`），没有错误 DCE、错误重排或 functionalization 前后对照 | alias/mutation 主线在 functionalization 后断裂，读者无法判断何时能安全恢复 in-place | 增加 post-grad 调用阶段和 `can_inplace` 合法边界；加入三个设计指定 case |
| P1-03 | P1 | 第 07 篇 Lab 没有验证设计要求的 guards、graph break 和 meta | 规格：`design.md:242-255`；正文声称 Lab 记录 Node opcode/target（`07_graph_capture_frontends_and_tracing.md:157-165`），脚本只打印两个 `call_module` 布尔量、compile count、signature 与空 range constraints（`part2_capture_frontends.py:19-48`） | 四种捕获前端的关键差异只停留在正文，未被实验支持 | 同一模型分别捕获完整 node/target/meta；加入数据分支和 graph-break case；保存 guards/explain 输出 |
| P1-04 | P1 | 第 09 篇没有实现“用 AOT 日志逐节点映射 joint/fw/bw” | 规格：`design.md:275-291`；脚本只保存 GraphModule 并打印图数量、总 output leaf 数和 bw target 列表（`part2_aot_graphs.py:17-85`），没有逐节点 old→new 表或 AOT log artifact | joint→fw/bw identity 断言不能由 Lab 重现 | 输出包含 stage、node name、target、owner id、origin/meta、fw/bw slot 的 JSON/TSV；保存 AOT 日志 |
| P1-05 | P1 | 第 09 篇指标名把“总 fw output leaves”误称为“saved output count” | `part2_aot_graphs.py:47-49,65,79` 对 `output.args[0]` 全部 leaves 计数，其中包含用户输出；第 09 篇记录 `fw_saved_output_count=4`（`09_aotautograd_joint_forward_backward_graphs.md:237-250`） | 会把 user-visible prefix 误算为 saved boundary，破坏 ABI 心智模型 | 分别报告 `fw_user_output_count`、各 saved slice count 与 `fw_total_output_leaves` |
| P1-06 | P1 | 第 10 篇 Lab 没有切换 activation memory budget，也没有测量峰值激活 | 规格：`design.md:293-308`；脚本只切换 `default_partition` 与 `min_cut_rematerialization_partition`（`part2_aot_graphs.py:73-85`）；文档以 output count 代替 peak measurement（`10_saved_tensors_recompute_and_runtime_abi.md:193-207`） | save/recompute 的核心 trade-off 没有量化证据 | 至少运行两个 budget；报告 saved bytes、recompute FLOPs/targets 和明确定义的 peak metric |
| P1-07 | P1 | 第 11 篇设计 Lab 尚未实现 | 规格要求追踪 `matmul+activation` 穿过所有阶段：`design.md:310-323`；正文示例是 `relu(x @ w + b)`（`11_graph_stage_boundaries_identity_and_provenance.md:8-23`），但引用的两个 Lab 分别是 Linear 前端对照与 `sin/cos` AOT；后半 artifact 明示待追加 | 本篇“真实阶段、dump 名称、generated mapping”的核心问题没有实验证据 | 使用统一 matmul 模型，产出 stage manifest 和 many-to-many provenance 表；完成前将标题改为“计划中的 Lab” |
| P1-08 | P1 | 第 01 篇没有单独解释 dataflow graph，Lab 没有编译日志 | 规格：`design.md:144-157`；正文有 FX program graph/use-def，但没有把 dataflow graph 作为独立概念与 program graph 对照；`part1_graph_taxonomy.py` 只打印 eager grad_fn 与 FX graph（`part1_graph_taxonomy.py:7-25`） | 图分类仍可能把 program order 与 dataflow 混同；设计正例未完成 | 增加 program graph vs dataflow projection；加入 custom backend/`TORCH_LOGS` 编译观察面 |
| P1-09 | P1 | 03、05、07、08 缺少复杂度环节 | 四篇没有 `## ...复杂度`；规格闭环要求复杂度：`design.md:49-64` | 这四篇不满足 §3.2 的完整闭环 | 每篇补参数化成本、常见成本和严格边界；不要只写“可能变慢” |
| P1-10 | P1 | `_extract_graph_with_inputs_outputs` 的核心断言和 locator 不准确 | 第 09 篇说 helper “`node_copy`并创建 GraphModule”，引 `torch/_functorch/partitioners.py:514-615`（`09_aotautograd_joint_forward_backward_graphs.md:108-118`）；实际 `node_copy` 在 `:656/:659`，output/return Graph 在 `:694-705`，GraphModule 在 `_extract_fwd_bwd_modules` 的 `:1577-1578` 创建 | 固定 SHA 下的真实调用链被截断且对象类型错误 | 改成“helper 返回 fresh `fx.Graph`；caller 再包装为两个 GraphModule”，使用分段 locator |
| P1-11 | P1 | 第 04 篇 Lab 没有观察真正的 guard 表达式或 symbolic graph nodes | 规格：`design.md:190-204`；脚本只计数 compilation、检查 backend example input 中是否有 `SymInt`、统计 range constraints（`part1_symbolic_shapes.py:5-43`） | `guard miss → recompile → dynamic graph` 的因果链没有 artifact | 保存 guard/explain 日志、每次 backend graph 与 symbolic placeholder/meta；加入 range 越界失败 |
| P2-01 | P2 | 索引缺少“每篇前置依赖与学习成果”，cache 导航也不明确 | 规格：`design.md:128-140`；索引表只有“先解决的问题”（`index.md:40-81`）；阅读路径存在，但没有逐篇 outcomes；Related Pages 无 compile-cache 入口（`index.md:158-165`） | 学习路径可读，但不能作为依赖/outcome ledger | 为 21 篇增加 prerequisites/outcomes 两列；补 runtime/cache 明确入口 |
| P2-02 | P2 | 第 06 篇对 `map` 只列名，未解释机制；invalid branch contract 未进入 Lab | `06_structured_outputs_higher_order_and_nested_graphs.md:79-101` 只在列表中出现 `map`；Lab 只跑合法正/负 predicate（`:239-257`） | “包含 map”形式满足、机制不满足 | 增加 map body/signature/shape 约束；加入 cond TreeSpec 或 metadata 不匹配失败 |
| P2-03 | P2 | 两处 locator/措辞只部分支持断言 | `OutputAliasInfo` 的 `unsafe_view_alias/custom_function_view` 枚举在 `schemas.py:51-80`，正文却只引 `:83-128`；automatic-dynamic exclusion 的 PGO 状态会先记录 excluded value，配置控制的是传入/发射 guard，正文“只有开启时才维护”过强（`04_symbolic_shapes_guards_and_graph_reuse.md:156-159`） | 结论大体正确，但证据落点和状态机不精确 | 扩展枚举 locator；改成“PGO 记录，config 控制传播/guard 发射”，引 builder/symbolic_shapes 实际 gate |
| P2-04 | P2 | 9 个脚本全是 print-only，没有断言 | 机械扫描 `part1_*.py`/`part2_*.py`：每个 `asserts=0` | 行为漂移时脚本仍可退出 0，“实跑成功”不等于结果符合 expected | 让脚本返回结构化结果并断言关键 invariant；或由统一 unittest/pytest contract 校验 |

## 4. 索引 §5.1 审计

| 设计要求 | 状态 | 当前证据 | 缺口 |
|---|---|---|---|
| 图类型和完整编译生命周期总图 | ✅ | `index.md:17-38` | 主线图存在 |
| 四部分知识地图 | ✅ | `index.md:40-81` | 无 |
| 每篇前置依赖与学习成果 | ❌ | 表格只有“先解决的问题”；个别页面页头有 `前置` | 索引没有逐篇 prerequisite/outcome ledger |
| 基础、pass 开发、后端三条路径 | ✅ | `index.md:83-95` | 无 |
| 贯穿模型和 Lab 环境 | △ | `index.md:97-121` 声明模型与 Lab 版本 | 没有共同模型实现、统一命令、artifact root；正文承认链路不完整 |
| 源码基线、验证等级、术语入口 | ✅ | `index.md:3-6,124-156` | Lab/source 基线不同已明确披露 |
| FX/export、AOTAutograd、Inductor、runtime、cache 导航 | △ | `index.md:158-165` 有 FX/export、AOT、Inductor、eager/CUDA Graph 链接 | compile-cache 无明确领域入口 |

## 5. 逐篇闭环缺项矩阵

符号：✅ 足够；△ 有内容但没有形成可验证闭环；❌ 缺失。  
“真实调用链”要求至少有入口、关键 hop、输出对象和固定源码 locator，而不是并列列出 API。

| 篇 | 设计“包含项” | 动机→机制 | 数据结构/不变量 | 真实调用链 | Lab 七要素与设计 case | failure modes | 复杂度 | 上/下游 | 统一模型复用 | 关键缺项 |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 图 IR 动机与分类 | △ | ✅ | △ | ❌ | ❌ | △ | ✅ | ✅ | ❌ | dataflow graph 未单独解释；无 compile log；无命令/边界/artifact |
| 02 FX 核心数据模型 | ✅ | ✅ | ✅ | △ | △ | ✅ | ✅ | ✅ | ❌ | 机制最完整；Lab 有两个错误例，但仍无明确命令和 artifact |
| 03 值、元数据与签名 | ✅ | ✅ | ✅ | △ | △ | ✅ | ❌ | ✅ | ❌ | stdout 记录错误；无复杂度、命令、artifact |
| 04 符号形状与复用 | △ | ✅ | ✅ | △ | △ | ✅ | ✅ | ✅ | ❌ | shape op/FakeTensor propagation 调用链薄；未输出 guards/symbolic nodes；无越界例 |
| 05 effect/alias/mutation | △ | ✅ | ✅ | △ | ❌ | ✅ | ❌ | ✅ | ❌ | reinplace 缺失；mutation region 只被提及；设计指定三个 Lab 对照未做 |
| 06 结构化输出/HOP | △ | ✅ | ✅ | △ | △ | ✅ | ✅ | ✅ | ❌ | `map` 仅列名；无 invalid branch；无 artifact |
| 07 捕获前端 | ✅（正文） | ✅ | △ | △ | ❌ | ✅ | ❌ | ✅ | ❌ | Lab 未覆盖 guards、break、meta，且不输出完整 node/target |
| 08 规范化 | ✅（正文） | ✅ | ✅ | △ | △ | ✅ | ❌ | ✅ | ❌ | 脚本没有正文声称的 symbolic_trace；无复杂度、错误例、artifact |
| 09 AOT joint/fw/bw | ✅（正文） | ✅ | ✅ | ✅（有一处错误 locator） | ❌ | ✅ | ✅ | ✅ | ❌ | 没有逐节点映射/log；总 output leaf 被误标 saved count |
| 10 save/recompute/ABI | ✅（正文） | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | 未切 memory budget、未测 peak、无错误/边界/artifact |
| 11 阶段 identity/provenance | △ | ✅ | ✅ | △ | ❌ | ✅ | ✅ | ✅ | ❌ | debug handle/dump 名称缺真实入口；设计 Lab 和 artifact bundle 尚未实现 |

闭环结论：

- 11 篇中，没有一篇满足规格 §3.5 的全部 Lab 七要素。
- 03、05、07、08 明确缺少复杂度，直接违反 §3.2。
- 01、03、04、05、06、07、08、11 的“调用链”主要是机制片段或阶段表，不是一个真实输入端到端走过源码 hop 的调用链。
- 所有篇都提供了 Related Pages，所以上下游导航是本批次最稳定的闭环部分。

## 6. Lab 真实覆盖矩阵

### 6.1 复跑结果

9 个唯一脚本全部退出 0：

| 脚本 | 退出状态 | 与页面记录一致性 |
|---|---:|---|
| `part1_graph_taxonomy.py` | 0 | 输出与页面 block 一致，但缺设计要求的 compile log |
| `part1_fx_core.py` | 0 | 一致；包含 erase-live 与 cross-graph 两个错误例 |
| `part1_values_signatures.py` | 0 | **不一致**；实际 `symbolic_fx_ops` 多一组 `get_attr,call_function` |
| `part1_symbolic_shapes.py` | 0 | 一致；但没有输出 guard 表达式或 symbolic nodes |
| `part1_effects_alias.py` | 0 | 一致；只覆盖默认 DCE purity 边界 |
| `part1_structured_hop.py` | 0 | 一致；两个合法 cond 分支均运行 |
| `part2_capture_frontends.py` | 0 | 一致；只输出粗粒度布尔/计数 |
| `part2_normalization.py` | 0 | 摘要 block 一致；**未做页面声称的 symbolic_trace 对照** |
| `part2_aot_graphs.py` | 0 | 数值一致；没有页面 11 声称的 Node/owner 表 |

第 03 篇的具体差异：

```text
文档:
symbolic_fx_ops=placeholder,get_attr,call_function,output

实跑:
symbolic_fx_ops=placeholder,get_attr,call_function,get_attr,call_function,output
```

原因不是 PyTorch 随机漂移：脚本中的 `ReadState.forward` 同时读取 `weight` 和 `bias`，并执行乘法与加法（`part1_values_signatures.py:6-23`），因此实际序列与脚本本身一致，文档记录被过度压缩。

### 6.2 逐篇设计正例/边界例

| 篇 | 脚本 | 设计正例覆盖 | 错误/边界覆盖 | 缺失 |
|---|---|---|---|---|
| 01 | `part1_graph_taxonomy.py` | △ eager grad_fn + FX | ❌ | compile logs、错误/边界、artifact |
| 02 | `part1_fx_core.py` | ✅ create/replace/lint/recompile | ✅ erase live、cross-owner | 明确命令、artifact |
| 03 | `part1_values_signatures.py` | ✅ state lifting/signature/meta | ✅ ExportedProgram 直接调用报错 | 正确 expected block、复杂度、artifact |
| 04 | `part1_symbolic_shapes.py` | △ static/auto compile count + range constraint | ❌ 3/4/5 都在合法范围内 | guard 内容、symbolic graph、range 越界 |
| 05 | `part1_effects_alias.py` | △ pure vs `copy_`、alias 可见 | △ purity boundary | 错误 DCE、错误重排、functionalization 对照、reinplace |
| 06 | `part1_structured_hop.py` | ✅ tuple/getitem、cond child GMs | △ 两个合法 predicate 分支 | branch TreeSpec/meta 不匹配失败、map |
| 07 | `part2_capture_frontends.py` | △ 四入口只做粗粒度对照 | ❌ | guards、break、meta、数据分支 |
| 08 | `part2_normalization.py` | △ functionalization + decomposition | ✅ mutating view 是有价值边界 | schema normalization、synthetic-base、页面声称的 symbolic_trace |
| 09 | `part2_aot_graphs.py` | △ joint/fw/bw 数量、owner invariant、gradient | ❌ | 逐节点映射、AOT log、失败例 |
| 10 | `part2_aot_graphs.py` | △ default vs min-cut | ❌ | memory budget、saved bytes/peak、边界例 |
| 11 | 引用 07/09 两脚本 | ❌ 不是本篇的 matmul 全阶段追踪 | ❌ | post-grad/IR/Scheduler/kernel/provenance bundle 全缺 |

### 6.3 §3.5 七要素横向结论

| 要素 | 当前状态 |
|---|---|
| 最小可运行输入 | 9 个脚本都有；但第 11 篇没有对应设计输入 |
| 明确执行命令 | **0/11**；页面只写“运行 `labs/...py`”或仅列脚本名 |
| 预期图结构/输出模式 | 01–10 有摘要；第 11 篇没有本篇目标的 expected |
| 正确 + 错误/边界 | 只有 02、03 明确同时具备；05、06、08 有边界性质但未覆盖设计指定失败 |
| dump/log/artifact 定位 | **0/11**；Part I/II 脚本无 artifact 写出或 logging enable 代码 |
| 源码基线和运行环境 | 11 篇页头都有 |
| 验证结果 | 现有 9 脚本已复跑；但 03 记录错误，08/11 有脚本能力过度声明 |

## 7. §6 统一贯穿模型核验

规格要求一个逐步扩展、同时包含 parameter/buffer、view/mutation、dynamic shape、结构化输出、可微 matmul/pointwise/reduction 和 HOP 分支的模型。

当前实际是下列互不相同的模型：

| 能力 | 当前所在脚本 | 与其他阶段的连接 |
|---|---|---|
| parameter + buffer | `part1_values_signatures.py::ReadState` | 未进入 dynamic、AOT 或 provenance |
| view + mutation | `part1_effects_alias.py` 的手工 Graph；`part2_normalization.py::mutating_view` | 两者也不是同一函数 |
| dynamic shape | `part1_symbolic_shapes.py::f/DynamicModule` | 未进入 AOT partition |
| structured output + cond | `part1_structured_hop.py::structured/CondModule` | 未进入 Part II AOT |
| matmul + pointwise | `part2_capture_frontends.py::Model` 的 Linear+relu | 无 buffer、dynamic、structured output、reduction、cond |
| reduction | `part1_graph_taxonomy.py::f` 的 sum | 与 matmul 模型不同 |
| AOT save/recompute | `part2_aot_graphs.py::fn` 的 sin/cos | 与索引/第 11 篇的 matmul+activation 示例不同 |

不存在：

- 共同的 model/fixture 文件；
- 共同输入和 feature flag；
- 跨 Lab 稳定的 node/source 标识；
- artifact manifest；
- Python → Dynamo/FX → functional ATen → joint/fw/bw 的同一输入证据链。

因此“统一贯穿模型存在”的判断是 **否**。索引中的文字声明不能替代代码和 artifact。

## 8. 已打开核验的源码 locator

机械检查只能证明 locator 可打开；下表记录的是实际阅读源码后的语义核验。表中列出 31 组，超过“至少 10 个”的要求。

| # | Wiki 断言主题 | 实际打开的固定源码范围 | 结果 |
|---:|---|---|---|
| 1 | FX Graph 是 Node 序列并构成 Python 函数；root/insertion state | `torch/fx/graph.py:1397-1465` | ✅ 支持 |
| 2 | 六种 opcode；users 是 distinct-user ordered set，重复 use 只记一次 | `torch/fx/node.py:258-304` | ✅ 支持 |
| 3 | 更新 args/kwargs 同步 `_input_nodes` 与 users | `torch/csrc/fx/node.cpp:307-359` | ✅ 支持 |
| 4 | `create_node` 归一化参数、命名、插入、side table、长度 | `torch/fx/graph.py:1585-1661` | ✅ 支持 |
| 5 | `replace_all_uses_with` snapshot users；`is_impure` 当前分派 | `torch/fx/node.py:717-808` | ✅ 支持 |
| 6 | `lint` 检查 ownership、拓扑、side table、name | `torch/fx/graph.py:2610-2655` | ✅ 支持 |
| 7 | 默认 FX DCE 的 side-effect detection 不 sound | `torch/fx/graph.py:2690-2732` | ✅ 源码 warning 直接支持 |
| 8 | FakeTensor 是 meta tensor 加 logical fake device | `torch/_subclasses/fake_tensor.py:834-845` | ✅ 支持 |
| 9 | SymNode 的 expr/ShapeEnv/type/hint/constant 与 hint 含义 | `torch/fx/experimental/sym_node.py:89-136` | ✅ 支持 |
| 10 | `meta["val"]` 的能力和 autograd 信息限制 | `torch/fx/experimental/proxy_tensor.py:817-835` | ✅ 支持 |
| 11 | Export input/output kinds、state lifting、mutation output | `torch/export/graph_signature.py:81-181` | ✅ 支持 |
| 12 | range 是保守近似；size-like 只在 size-oblivious reasoning 下假设 `>=2` | `torch/fx/experimental/symbolic_shapes.py:4024-4068` | ✅ 支持 |
| 13 | `var_to_val` deprecated，权威名为 `backed_var_to_val` | `torch/fx/experimental/symbolic_shapes.py:5960-5980` | ✅ 支持 |
| 14 | `torch.compile` per-code-object cache、guard failure、dynamic 三态 | `torch/__init__.py:3134-3181` | ✅ 支持 |
| 15 | Export 默认 `strict=False`；strict/non-strict contract | `torch/export/__init__.py:59-69,179-187` | ✅ 支持 |
| 16 | `make_fx` 返回待调用 wrapper；real/fake/symbolic modes | `torch/fx/experimental/proxy_tensor.py:3312-3385` | ✅ 支持 |
| 17 | effect token 通过 value threading 建立顺序 | `torch/_higher_order_ops/effects.py:75-138` | ✅ 支持 |
| 18 | cond 两分支 TreeSpec/metadata 合并与失败 | `torch/_higher_order_ops/cond.py:388-449` | ✅ 支持；正好可用于补 negative Lab |
| 19 | FX DCE 只递归到被 `get_attr` 引用的 child GraphModule | `torch/fx/graph.py:2762-2774` | ✅ 支持 |
| 20 | metadata analysis 没有 tracing | `torch/_functorch/_aot_autograd/collect_metadata_analysis.py:167-242` | ✅ `:237-238` 明确说明 |
| 21 | AOT 准备 primals/tangents、`create_joint`、`make_fx` capture | `torch/_functorch/_aot_autograd/graph_capture.py:92-183,472-536` | ✅ 支持 |
| 22 | `_extract_graph_with_inputs_outputs` 的 fresh graph/env/placeholder/copy/output | `torch/_functorch/partitioners.py:514-705` | △ 文档所引 `:514-615` 只覆盖前半；`node_copy` 在 `:656/:659`，函数返回 `fx.Graph` |
| 23 | fw/bw Graph 包装成两个 GraphModule，ABI input 顺序 | `torch/_functorch/partitioners.py:1473-1592` | ✅ GraphModule 创建在 `:1577-1578` |
| 24 | default partition save/recompute 和 reorder | `torch/_functorch/partitioners.py:1595-1805,1920-1995` | ✅ 支持 |
| 25 | min-cut flow network、capacity、minimum cut、cut nodes | `torch/_functorch/partitioners.py:2641-2765,2888-2890,3052-3072` | ✅ 支持 |
| 26 | saved tensors/SymInt/opaque objects 在 ctx 中分组 | `torch/_functorch/_aot_autograd/runtime_wrappers.py:2615-2683` | ✅ 支持 |
| 27 | backward prologue 参数拼接顺序 | `torch/_functorch/_aot_autograd/runtime_wrappers.py:2982-3089` | ✅ 支持 |
| 28 | GraphLowering 汇聚 origins 并安装 stream/mempool context | `torch/_inductor/graph.py:1925-2000` | ✅ 支持 |
| 29 | IRNode origins/trace/origin_node/stream/mempool | `torch/_inductor/ir.py:589-645` | ✅ 支持 |
| 30 | StorageBox realization 注册 ComputedBuffer/Operation 并传播 provenance | `torch/_inductor/ir.py:10578-10607` | ✅ 支持 |
| 31 | Scheduler wrapper source context 选择 latest origin | `torch/_inductor/scheduler.py:9053-9069` | ✅ 支持，同时源码说明这只是 context 选择 |

### 8.1 需要纠正或补 locator 的具体断言

1. **第 09 篇 Graph/GraphModule 混写。**  
   `_extract_graph_with_inputs_outputs` 返回 `fx.Graph`（`partitioners.py:694-705`）；两个 GraphModule 由 caller 在 `:1577-1578` 包装。应把调用链拆成两个 hop。

2. **第 09 篇 required-node closure/classify 没有 locator。**  
   实际 `classify_nodes` 在 `torch/_functorch/partitioners.py:4005-4088`，其中分别构造 `required_bw_nodes`、`tangents_closure`、`required_fw_nodes` 与 `unclaimed_nodes`。正文 §6 应引用这个范围，而不是只做概念描述。

3. **第 05 篇 OutputAliasInfo 枚举 locator 截掉了枚举。**  
   `unsafe_view_alias` 与 `custom_function_view` 位于 `torch/_functorch/_aot_autograd/schemas.py:51-80`；当前引用从 `:83` 才开始。

4. **第 04 篇 exclusion 状态机表述过强。**  
   PGO 在 `torch/_dynamo/pgo.py:399-420` 维护 transition/excluded state；配置 gate 在 `torch/_dynamo/variables/builder.py:3388-3397` 和 `torch/fx/experimental/symbolic_shapes.py:6851-6855`。应区分“记录 transition”与“将 exclusion 传入并发射 guard”。

5. **第 11 篇 stage/dump 表缺少真实 dump 入口。**  
   它列出了通用名称（`11_graph_stage_boundaries_identity_and_provenance.md:139-152`），但没有固定源码中的 logging/artifact producer、配置开关、真实文件名或一份 Lab 产物。当前只能当概念表，不能当“每个 pass 的真实 dump 名称”证据。

## 9. §12 验收标准状态

| §12 条目 | 当前状态 | 本审计证据 |
|---:|---|---|
| 1. 索引从概念引导至 codegen | ✅ | 总图、四部分地图、三条路径存在 |
| 2. 21 篇各自完整闭环 | ❌ | Parts I/II 已有 11 篇中，Lab 全部缺七要素之一；4 篇无复杂度 |
| 3. 旧页 coverage ledger | ⏸ 本报告不评 | 不在本子任务范围 |
| 4. correction report/changelog | ⏸ 本报告不评 | 不在本子任务范围 |
| 5. 权威实现断言绑定固定基线 | △ | 页头 SHA 和 143 个 locator 均存在；但有错误/截断 locator 和未绑定的 required mechanism |
| 6. 正式 Lab 已运行并记录预期 | ❌ | 9 个脚本已复跑，但第 03 篇记录不符，多个页面过度声明脚本能力，第 11 篇 Lab 未实现 |
| 7. 贯穿模型生成完整 artifact bundle | ❌ | 无统一模型；Part I/II 无 artifact writer；第 11 篇明确待追加 |
| 8. Part III pass 全语义验证 | ⏸ 本报告不评 | 超出 Parts I/II 范围 |
| 9. Part IV kernel 反向定位 | ⏸/❌ | 超出主要范围；但第 11 篇需要的跨阶段 bundle 明确不存在 |
| 10. 链接/Mermaid/围栏/Related/index/回链/diff | △ | 本范围 128 links：0 unresolved、0 ambiguous；12/12 围栏平衡且有 Related Pages；Mermaid 未用 parser 验证；系列为 untracked，`git diff --check` 无法覆盖这些文件 |

## 10. 具体修订建议

### 10.1 先修 P0：建立唯一模型和 artifact contract

建议新增一个唯一模型定义，例如 `labs/series_model.py`：

```text
SeriesModel
├── Parameter weight
├── registered Buffer bias/scale
├── dynamic batch/sequence dimension
├── view path
├── optional controlled mutation path
├── differentiable matmul → pointwise → reduction
├── tuple/dict output
└── torch.cond branch
```

所有 Part I/II Lab 只允许：

- 复用这个 model；
- 通过 wrapper/feature flag 暂时隐藏尚未讲到的特性；
- 使用同一组 deterministic inputs；
- 输出同一个 `run_id`、source hash 和稳定 stage manifest。

建议统一 artifact 目录：

```text
artifacts/graph_series/<run_id>/
├── environment.json
├── model_source.py
├── manifest.json
├── part01/eager_autograd.txt
├── part01/fx_graph.txt
├── part01/compile_log.txt
├── part04/guards.txt
├── part04/dynamo_graphs/
├── part07/frontends/
├── part08/functional_aten.txt
├── part09/joint.json
├── part09/fw.json
├── part09/bw.json
└── provenance.tsv
```

`manifest.json` 至少记录：源码 SHA、Lab torch SHA、Python/OS/device、命令、seed、输入摘要、artifact 相对路径、verified/blocked 状态。

### 10.2 每篇采用相同 Lab 模板

每篇 Lab 章节至少固定为：

1. **环境**：Python、torch version/git SHA、device、cwd；
2. **命令**：从 repo root 可复制的完整命令；
3. **正例输入**；
4. **错误/边界输入**；
5. **expected stdout/结构断言**；
6. **artifact tree 和定位方法**；
7. **实际验证时间与状态**。

建议命令形态：

```powershell
python wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/labs/<script>.py `
  --artifact-dir artifacts/graph_series/<run_id>/<part>
```

脚本应返回 JSON 或写 `summary.json`，由测试断言关键字段；不要只依赖人工比对 print。

### 10.3 逐篇最小修订

| 篇 | 最小必须修订 |
|---|---|
| index | 增加 21 篇 prerequisites/outcomes；指向 canonical model、统一 runner、artifact root；补 cache 导航；把未完成链路显式标 `planned/blocked` |
| 01 | 增加 program graph vs dataflow projection；同一 canonical model 输出 eager tape、FX 和 compile log；加入 graph break 或 unsupported control-flow 边界 |
| 02 | 保留现有错误例；输出 graph/users/lookup snapshot artifact；增加脚本断言 |
| 03 | 修正 stdout；分别报告两个 get_attr 和两个 call_function；补 signature/meta 成本与存储复杂度 |
| 04 | 输出每次 backend GraphModule、guards 和 symbolic meta；运行 export range 内/外输入；修正 exclusion 状态机 locator |
| 05 | 增加 `post_grad.py` 中 reinplace 调用阶段及 `reinplace.py` 合法边界；实现错误 DCE、错误 reorder、functionalization 对照；补复杂度 |
| 06 | 解释 map body/signature；加入 cond TreeSpec 或 metadata mismatch 失败；输出 child graph artifacts |
| 07 | 输出四种路径的完整 node/target/meta/signature；加入 graph break/fullgraph failure 和 guards；补复杂度 |
| 08 | 让脚本真的运行 symbolic_trace 或删除该声明；加入 schema normalization、synthetic-base alias case；补复杂度 |
| 09 | 修正 Graph→GraphModule 调用链和 locator；输出 per-node joint/fw/bw mapping；分开 user outputs 与 saved slices |
| 10 | 参数化 activation memory budget；定义并测量 saved bytes/peak proxy；输出 recompute nodes 和 cost basis |
| 11 | 使用 canonical matmul+activation 生成全阶段 manifest；完成前不得写“已验证 artifact bundle”；每个 dump 名称绑定 producer/flag/path |

### 10.4 Source-faithful 验证门槛

修订后应机械执行：

1. 所有 source locator 文件/范围存在；
2. 每个 required mechanism 至少抽查一个真实 caller→callee→output hop；
3. 所有 Lab summary 与文档 expected 自动比较；
4. 所有页面至少一个 negative/boundary assertion；
5. artifact manifest 中每个路径存在且非空；
6. 统一模型 source hash 在所有 Part I/II artifact 中相同；
7. wikilink、围栏、Mermaid parser、`Related Pages`、index backlink、`git diff --check` 全部通过；
8. series 文件纳入 Git 后再宣称 `git diff --check` 已覆盖。

## 11. 最终判定

Parts I–II 的**概念解释骨架已经成形**，许多页面的 motivation、术语边界和源码密度明显高于普通 API 说明；现有 9 个 Lab 也确实能在声明环境运行。

但按已确认的设计规格，它们仍是“多篇机制说明 + 多个独立 smoke script”，还不是“统一模型驱动、每篇可验证、可沿真实调用链和 artifact 追踪”的学习系列。P0-01 与 P0-02 修复前，§12.2、§12.6、§12.7 均不能通过。
