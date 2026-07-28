# PyTorch Graph Series Part III 源码忠实性与设计验收审计

> 审计日期：2026-07-26
> PyTorch 源码基线：`E:/97-codes/torch_parallel/p` @ `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> 知识库快照：`E:/97-codes/torch_parallel/llm-knowledge`，HEAD `c884b9d6b5cd702515e5f1b38aff236620af9865`；本文定位符针对 2026-07-26 工作树快照
> 实跑环境：Windows，Python `3.13.5`，PyTorch `2.9.1+cpu`
> 审计范围：设计规格 3.2、3.5、5.4、6、验收项 12.6–12.8；正文 12–16；`labs/part3_*.py`，以及正文直接复用的两个 Part I Lab
> 审计约束：只读检查 wiki 与 PyTorch 源码；仅新建本报告
> 并发快照说明：`labs/part3_end_to_end_pass.py`在本次审计初稿后加入工作树；最终结论已重新审计并包含该文件

## 1. 结论

**Part III 当前不可验收。**

五篇正文对当前源码的核心机制解释多数成立：FX 的插入/替换/擦除与反向 uses 更新、PatternExpr 的对象 identity sharing、root bucket 候选快照、reverse DCE、三类 stage driver、以及 PassManager 的有界重复，都能在固定 commit 中找到实证。

阻塞验收的是证据闭环：

1. 新增的 `part3_end_to_end_pass.py`确实实现并运行了受限的 `add(matmul) → addmm` rewrite；本次观察到数值、梯度、shape、non-alias、无输入 mutation 与第二次不改图均为真。
2. 该 pass 没有被 12–16 任一正文引用，也没有进入页面命令、stage Lab 或 artifact bundle；它仍是孤立脚本，不是设计所称的跨五篇贯穿证据。
3. 五篇均把原有 Lab 标成“已验证”，但至少四篇的具体覆盖声明与各自引用脚本不符；新增 pass 不会补真这些页面级声明。
4. 所有相关脚本都可运行且本次复跑 exit 0，但它们只 `print`，没有 assertion，打印 `False` 仍会成功退出。
5. 每篇都缺显式命令和可定位的输出 artifact；不存在设计要求的 Part III artifact bundle。
6. 复杂度没有按“常见 / 期望 / 严格上界”分层；稳定拓扑排序还被错误写成无条件 `O(V+E)`。

因此，“脚本能跑”不能替代“设计验收项已执行”。

## 2. P0 / P1 / P2 发现表

| ID | 级别 | 发现 | 已核验证据 | 影响 / 违反条款 | 可执行修订 |
|---|---|---|---|---|---|
| III-001 | P1 | **贯穿 pass 已存在，但尚未贯穿文档与证据链。** 新增 `part3_end_to_end_pass.py`实现 exact-shape、rank-2、same-dtype 的 `add(matmul) → addmm`，并打印 forward/gradient/shape/alias/mutation/second-run 结果；五篇正文均未引用它，且它没有 assertion、stage placement 或 artifact。 | 设计 `docs/superpowers/specs/2026-07-23-pytorch-graph-learning-series-design.md:501-524,735-748`；脚本 `part3_end_to_end_pass.py:13-91,98-162`；本次实跑见 5.2 | 设计 6 与 12.8 从“缺失”提升为“部分”；12.7 仍失败。孤立脚本不能自动补齐五篇 Lab。 | 把该文件升级为唯一贯穿 harness；五篇分别引用同一命令/manifest，并增加 PatternMatcher、stage、effect 与 legality 分层证据。 |
| III-002 | P0 | **“已验证 Lab”声明与页面所引代码不一致。** 12 声称 `part3_passes.py`有 `node_copy`；13 声称运行 `PatternMatcherPass`；15 声称有互逆 oscillation pair；16 声称用 Fake/meta 并比较 input alias。它们在各页所引脚本中均不存在；新增 end-to-end pass 也不会追溯补真这些声明。16 的 `structural_match=True`还是硬编码，而不是 pattern 命中结果。 | 正文 `12_fx_graph_editing_primitives_and_invariants.md:152-155`、`13_pattern_expression_and_matcher_engine.md:164-175`、`15_graph_pass_pipeline_ordering_and_fixpoint.md:161-178`、`16_graph_rewrite_legality_validation_and_complexity.md:197-209`；脚本 `part3_passes.py:1-47`、`part3_pattern.py:1-56`、`part3_legality.py:18-57` | 违反 3.5、12.6 的“不得把未运行示例写成已验证”；削弱所有后续验收证据。 | 在功能补齐前把各节改名为“部分验证”，逐项列出 `implemented / observed / not covered`；功能补齐后再恢复“已验证”。 |
| III-003 | P1 | **五篇均不满足逐篇 Lab 合同。** 没有任何页面给出实际执行命令；没有一篇给出输出文件或 artifact 定位；页面 12 没有明确预期输出，15 虽给出两个计数，但所声称的错误案例不存在。 | 五页 Lab 节：12 `:152-155`；13 `:164-178`；14 `:140-144`；15 `:161-178`；16 `:197-209`。全系列 Markdown 搜索 `python ... part3_*.py`为零命中。 | 违反 3.5 全部七项，并使 12.6 无可审计记录。 | 每页增加完整命令、输入/seed、精确预期、正例、反例、artifact 相对路径、环境与 exit 标准；命令必须可从仓库根执行。 |
| III-004 | P1 | **脚本不是测试。** 四个 Part III 脚本只有 print，没有 `assert`/`torch.testing.assert_close`；即使 `forward_matches=False`或 `gradient_matches=False`也会 exit 0。 | `part3_passes.py:21-47`、`part3_pattern.py:30-56`、`part3_legality.py:35-57`、`part3_end_to_end_pass.py:98-162` | “真实运行”只有进程级证据，没有判定级证据；12.6 不能据 exit 0 通过。 | 将所有布尔输出先 assertion，再打印；负例断言“拒绝且图未变”，正例断言“应用且全部等价”；统一返回非零失败码。 |
| III-005 | P1 | **PatternExpr 规格与 Lab 均缺关键项。** 正文没有设计要求的 serialized/precompiled pattern cache；Lab没有 `MultiOutputPattern`、`ReplacementPatternEntry`、`LoweringPatternEntry`或 `PatternMatcherPass.apply()`，只直接调用 `pattern.match(node)`。 | 设计 5.4/13 `:345-361`；正文 `13_pattern_expression_and_matcher_engine.md:94-146,164-178`；脚本 imports 与调用 `part3_pattern.py:1-10,24-56`；源码 cache `torch/_inductor/pattern_matcher.py:2095-2224` | 5.4/13 仅部分覆盖；候选 lifecycle、handler ABI 与 replacement 行为未被实验证明。 | 增加四个 assertion case：multi-output anchor、root bucket apply、graph replacement、lowering entry；增加 serialized pattern 生成/导入路径说明及 cache 失效边界。 |
| III-006 | P1 | **DCE / effect Lab 不完整。** 当前只有一个 unused add、一个 `copy_`和一个 producer-after-consumer；没有 dead chain 级联、nested cleanup，也没有“data topo 合法但 effect 次序错误”的反例。 | 设计 5.4/14 `:363-379`；`part1_effects_alias.py:7-32`；`part3_passes.py:10-31`；正文 `14_dead_code_topology_and_effect_order.md:140-144` | 5.4/14 的核心命题“拓扑正确为何仍可能错”没有实验，effect/mutation/RNG 保序仍是纯文字。 | 添加至少四张图：两节点 dead chain、impure/no-user 保留、nested child DCE、两个无 data edge 的 effect 节点被重排的失败例；再用显式 dependency/control_deps 修复。 |
| III-007 | P1 | **stage placement / fixed-point 页面 Lab 与设计不符。** `CountToThree`只是递增 GraphModule metadata 并在第三次返回 stable；没有同一 fusion rule 放到不同 stage，也没有 A→B / B→A 互逆 pass。新增贯穿 pass 单独观察到第二次不改图，但页面 15 未引用。 | 设计 5.4/15 `:381-395`；正文 `15_graph_pass_pipeline_ordering_and_fixpoint.md:161-178`；`part3_passes.py:34-47`；`part3_end_to_end_pass.py:138-142` | 页面 Lab仍没有验证阶段输入不变量、pass conflict 或 non-convergence；idempotence 只有孤立脚本证据。 | 用贯穿 pass 做 pre-grad 与 post-grad 两个真实挂载实验；另建两条互逆 target rewrite，断言 `steps=N`时被上限截断；把第二次 apply=0 写成 assertion 并接入页面 15。 |
| III-008 | P1 | **页面 16 引用的 legality Lab 远低于声明；新增贯穿 pass 只部分补足。** `part3_legality.py`只比较静态 tuple shape，`ShapeProp`结果未读取，非 Fake/meta，也无 alias/mutation。新增 pass 使用 real `ShapeProp.tensor_meta`，测了静态 exact/broadcast、forward、一阶梯度、non-alias、输入未变与第二次不改图；仍无 Fake/dynamic guard、device/layout proof、差分 alias/mutation、failure atomicity、random matrix、gradcheck/gradgradcheck或 determinism。 | `part3_legality.py:18-57`；`part3_end_to_end_pass.py:21-91,98-162`；正文 `16_graph_rewrite_legality_validation_and_complexity.md:24-136,182-209` | 5.4/16 仍未满足；12.8 提升为部分。页面“Fake/meta/input alias 已比较”的原声明仍然错误。 | 将新增 pass 接入页面 16；legality predicate 改为消费 FakeTensor/meta；补齐 eager-vs-rewrite alias/mutation relation、dynamic rejection、异常前后 graph+meta hash、gradcheck。 |
| III-009 | P1 | **复杂度回归且分层不足。** 14 把当前 in-place stable topo 写成 `O(V+E)`，但 waiting node 每次被唤醒都会重新扫描全部 args；高入度下保守上界含 `Σ d(v)^2`。13/15/16也没有定义 workload 分布，因而没有“期望复杂度”；更没有把任意 `extra_check`、handler、trace tensor computation 从图结构上界中分离。 | 正文 13 `:150-162`、14 `:129-138`、15 `:145-159`、16 `:157-180`；源码 `torch/_inductor/pattern_matcher.py:2940-2980` | 违反设计风险控制 `:733`“区分期望、常见和严格上界”；14 的结论本身不正确。 | stable topo 改为：bounded arity 常见近 `O(V+E)`；实现级保守界 `O(V+E+Σd(v)^2)`。Matcher 用 `C,B(v),K(p),A(p,v),Q,U` 参数化；无分布假设时明确“期望值未定义”。 |
| III-010 | P2 | **post-grad locator 不能支撑“mutation-tail ordering”。** 页面只引 `post_grad.py:227-330`，该范围覆盖 main patterns、hooks、stable sort 起点，但 late mutation tail 在 `:449-464`，最终 recompile/lint 在 `:473-474`。 | 正文 `15_graph_pass_pipeline_ordering_and_fixpoint.md:35-38`；源码 `torch/_inductor/fx_passes/post_grad.py:227-303,385-474` | 结论方向正确，但定位证据不完整。 | 把 locator 扩为分段定位：`:165-180`、`:227-303`、`:385-474`，并分别说明 DCE、main passes、collective sort、late mutation tail、final lint。 |
| III-011 | P2 | **若干关键机制缺源码锚点。** 12 的 `node_copy/graph_copy`、placeholder/output ABI；14 的 effect/control dependency；13 的 cache 均没有当前 commit locator。 | 正文 12 `:74-98`、13 全文无 cache、14 `:88-102`；已打开源码见第 7 节 | 不一定是语义错误，但不满足 3.2 的源码执行路径与固定基线可追溯要求。 | 补 `graph.py:1525-1551,2386-2420`、相关 signature/codegen 源码、`control_dependencies.py:1-40,134-217`、`pattern_matcher.py:2095-2224`。 |
| III-012 | P2 | **Lab 环境记录不足以复现内部 API。** 页头只有 `torch 2.9.1+cpu`；没有 Python、OS、安装来源/commit。源码结论绑定 main SHA，Lab 却运行发行版 2.9.1，二者兼容性没有逐页说明。 | 五页 `:3-6`；13 仅在 `:175`单独提示内部 API 版本差异 | 对公开 FX API 影响较小，对 `torch._inductor.pattern_matcher`、stable sort 等内部 API 影响显著。 | manifest 记录 Python、平台、torch version、torch git SHA、device、seed、命令；每个内部 API case 标出“pinned build”或“2.9.1 compatibility-only”。 |

## 3. 设计验收矩阵

状态定义：`通过`表示规格全部有源码与实跑证据；`部分`表示正文解释存在但实验或证据不完整；`失败`表示核心交付不存在或证据与声明冲突。

| 设计条款 | 要求 | 状态 | 证据与判断 |
|---|---|---|---|
| 3.2 | 为什么需要 → 概念 → 数据结构 → 不变量 → 源码路径 → 贯穿示例 → Lab → 失败模式 → 复杂度 → 上下游 | **部分** | 五篇概念与源码主干较完整；新增贯穿 pass 未接入五篇，Lab/失败模式与复杂度仍不闭环。 |
| 3.5 | 每篇含最小输入、命令、预期、正反例、artifact、环境、真实结果 | **失败** | 全部页面无命令、无 artifact locator；多个“已验证”声明超出代码。 |
| 5.4 / 12 | 编辑、复制、不变量、ABI、recompile；插入/替换/复制三个 pass 和破坏案例 | **部分** | 正文覆盖多数概念；Lab没有 `node_copy`/`graph_copy` pass，也没有设计所列三个独立 pass。 |
| 5.4 / 13 | PatternExpr、sharing、users、rollback、multi-output、candidate、entries、traced replacement/cache、边界 | **部分** | 核心 AST/candidate 解释较好；cache 缺失；Lab只验证 direct match 的 sharing/capture。 |
| 5.4 / 14 | dead/unclaimed/unsaved、DCE、stable topo、effect order、nested cleanup | **部分** | 正文解释 DCE/topo；“unclaimed/unsaved”未正面展开；effect 反例与 nested Lab 缺失；复杂度有误。 |
| 5.4 / 15 | stage invariants、ordering、conflict、fixed point、idempotence、candidate lifecycle、debug | **部分** | 正文大体覆盖；Lab没有真实 stage placement、pass conflict 或 oscillation。 |
| 5.4 / 16 | shape/dtype/device/layout/alias/mutation/autograd/dynamic、差分、失败原子性、复杂度 | **部分** | 正文形成 checklist；Lab几乎只测静态 shape 与一次一阶梯度，且页面高估覆盖。 |
| 6 | Part III 实现统一 `add + matmul` rewrite，并作为全系列逐步扩展模型的一段 | **部分** | `part3_end_to_end_pass.py`已实现受限 rewrite，但与 Part I/II 模型、五篇正文和 stage pipeline 均未连接。 |
| 12.6 | 所有正式 Lab 真实运行并记录预期输出 | **失败** | 本次可复跑 exit 0，但仓库没有命令/输出 artifact；脚本无 assertions；声明与实现不一致。 |
| 12.7 | 贯穿模型生成完整阶段 artifact bundle | **失败** | 已有单脚本 add+matmul 模型，但没有跨阶段 dump、manifest 或 bundle。 |
| 12.8 | 贯穿 pass 通过数值、梯度、shape、alias/mutation、重复运行 | **部分** | 本次实际观察六维均得到预期值；但脚本无 assertions/持久化结果，alias/mutation 不是 eager-vs-rewrite 差分，shape 仅静态 exact/broadcast。 |

## 4. 每篇 Lab 合同核对

| 页面 | 明确命令 | 明确预期 | 正例 + 错误/边界 | artifact 定位 | 环境 | 声明与脚本一致性 |
|---|---:|---:|---:|---:|---:|---|
| 12 FX 编辑 | 否 | 否 | 部分：live erase/cross-graph 来自 Part I | 否 | 仅 torch | **不一致**：声称 `part3_passes.py`有 `node_copy` |
| 13 PatternExpr | 否 | 是，正文给出 captures | 部分：shared / not-shared | 否 | 仅 torch | **不一致**：没有 `PatternMatcherPass.apply()`；设计要求的 multi-output/replacement/lowering 缺失 |
| 14 DCE/topo/effect | 否 | 仅行为描述 | 部分：pure/impure 与 topo lint；无 effect reorder | 否 | 仅 torch | 对已有两项一致，但规格覆盖不足 |
| 15 stage/fixpoint | 否 | 是，两个 count | 否：声称 oscillation，脚本只有单调 counter | 否 | 仅 torch | **不一致** |
| 16 legality | 否 | 仅布尔行为描述 | 部分：broadcast reject / exact accept | 否 | 仅 torch | **不一致**：非 Fake/meta，无 alias comparison，structural match 硬编码 |

`part3_end_to_end_pass.py`不属于任何页面当前 Lab 节。它包含贯穿 pass 的正反例与六维打印结果，
但没有页面命令、预期表、artifact locator 或回链，不能作为“每篇 Lab 合同”自动继承。

## 5. 实际执行审计

### 5.1 本次只读复跑命令

为避免更新 `.pyc`，本次使用：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part3_passes.py
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part3_pattern.py
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part3_legality.py
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part3_end_to_end_pass.py
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part1_fx_core.py
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part1_effects_alias.py
```

六个进程均 exit `0`。

### 5.2 实测输出

| 脚本 | 本次观察值 |
|---|---|
| `part3_passes.py` | `lint_failed_before_sort=True`；`topology_repaired_value=8.0`；`steps_1_count=1`；`steps_4_count=3` |
| `part3_pattern.py` | shared 命中；positional=`y`；keyword=`x:x`；Ignored positional count=0；not-shared 失败 |
| `part3_legality.py` | broadcast 结构标志为真但 legality=false，原图数值保留；transposed exact case forward/gradient=true |
| `part3_end_to_end_pass.py` | legal rewrite=true；graph 含 addmm；broadcast rewrite=false 且 code unchanged；forward/gradient/shape/non-alias/input-unmutated=true；second apply=false 且 code unchanged |
| `part1_fx_core.py` | live erase 抛 `RuntimeError`；替换后值 `6.0`；cross-graph lint 抛 `RuntimeError` |
| `part1_effects_alias.py` | unused add 被 DCE；`copy_`保留；alias 观察到值 `7.0` |

这些输出证明“当前发行版环境能执行”，不证明页面声称的未实现检查。

### 5.3 12.8 维度逐项核对

| 维度 | 真正执行的内容 | 是否作用于贯穿 rewrite | 验收 |
|---|---|---:|---|
| 数值 | addmm rewrite 与 eager forward `allclose=True` | 是 | **部分**：单一 seed/case，无 assertion |
| 梯度 | 三个输入的一阶 `.backward()`结果均 `allclose=True` | 是 | **部分**：无 gradcheck/gradgrad |
| shape | exact-shape 接受、vector broadcast 拒绝、output shape 相等 | 是 | **部分**：real/static ShapeProp，无 symbolic/Fake guard |
| alias | 断言式逻辑实际只检查 rewritten output 不 alias 三个输入，观察值为真 | 是 | **部分**：未与 eager alias relation 差分 |
| mutation | 调用 rewritten graph 后三个输入等于各自 snapshot，观察值为真 | 是 | **部分**：未与 eager effect/mutation relation 差分 |
| 重复运行 | 第二次 rewrite 返回 false，generated code 不变 | 是 | **部分**：只比较 code，无 graph/meta hash |

## 6. 五篇正文的源码语义判断

### 6.1 12：FX 编辑原语与不变量

成立：

- `create_node`负责命名、Node 构造、插入、side table 与长度更新。
- `_update_args_kwargs`先删除旧 reverse uses，再建立新的 `_input_nodes/users`。
- `replace_all_uses_with`先 snapshot users；`erase_node`拒绝 live 或跨 graph Node。
- `Node.__setattr__`会在属性变化前后从 lookup table remove/reinsert，因此 Lab 的 `node.target = ...`本身不会破坏当前 main 的 target index。
- `lint`检查 owner、拓扑、side table、name、opcode 和 module target；它不等价于数值/alias proof。

缺口：

- `node_copy/graph_copy`没有源码 locator，且没有 Lab。
- placeholder/output ABI 只有概念列表，未追到 companion signature/codegen。
- “改图事务”没有用一个完整 pass 证明 failure atomicity 与重复运行。

### 6.2 13：PatternExpr 与 matcher

成立：

- PatternExpr 是递归 predicate/capture AST，不是可执行 FX Graph。
- 同一 PatternExpr 对象复用通过 `pattern_to_node` identity 约束 DAG sharing。
- `_TargetArgsExpr`做 target/op、users、normalized args/kwargs、aggregate 与 constant 检查。
- MultiOutput 第一 root 必须是 `_TargetExpr`；后续输出从已绑定节点的 users 找 anchor，并对失败尝试恢复 context。
- root `(op,target)`注册、候选快照、逆序处理、单次 apply 不重访新 root 均正确。
- mutation region、stream/mempool 和 `guard_or_false(extra_check)`是 structural match 后的 safety layer。

缺口：

- 页面没有 serialized/precompiled pattern 路径。
- Lab没有候选 index、entry apply、multi-output 或 traced replacement。
- 复杂度只给近似式，没有 `B(v)`、anchor fan-out 与任意 callback 成本边界。

### 6.3 14：DCE、拓扑与 effect

成立：

- FX DCE 先 lint，再 reverse sweep `pure && no users`，并递归处理被直接 `get_attr`引用的 child GraphModule。
- Scheduler DCE 使用 active output users 与 `has_side_effects()`，不是 FX `Node.users`。
- stable topo 只从 args/kwargs 建 data dependencies，不证明 hidden effect order。

错误 / 缺口：

- stable topo 的无条件 `O(V+E)`不成立；当前实现会重复扫描高入度节点的全部 args。
- effect/order 结论没有用 control dependency 的 current source path 与 Lab 支撑。
- 设计要求的 dead chain、nested graph 和 effect reorder 反例均缺失。

### 6.4 15：stage、ordering 与 fixed point

成立：

- pre-grad 明确声明输入未 functionalize/normalize，且总会走 sort/lint/recompile tail。
- joint 先 canonicalize，main matcher 为 ordered lists；其 cleanup 由局部 `count`条件触发，不是通用 changed bit。
- post-grad 是 functionalized/normalized fw/bw graph，且 main pattern、collective、late mutation tail 有不同顺序。
- PassManager 默认 `steps=1`；`modified`只在 configured bound 内驱动下一轮。

缺口：

- 页面用于 mutation-tail 的 locator 截短。
- Lab没有任何真实 graph rewrite、stage placement、冲突或 oscillation。
- `CountToThree`能说明“有界重复与提前停止”，不能证明 idempotence 或 non-convergence。

### 6.5 16：rewrite legality 与复杂度

正文的 legality checklist 作为 review 框架是合理的，但 Lab 只验证其很小子集。尤其：

- `ShapeProp`被调用，但产生的 metadata 没有参与决定；`shape_legal`直接读取真实 input tuple。
- `structural_match=True`不是 matcher 结果。
- forward 与一阶 gradient 只测 exact add；没有 matmul 或通过 matcher 应用的 replacement。
- transposed input 只证明 non-contiguous input 能运行；没有检查 output/input storage alias。
- failure atomicity 只有“条件为 false 时不 retarget”，没有异常注入或 before/after graph identity。

新增 `part3_end_to_end_pass.py`显著改善了贯穿覆盖，但仍有残余契约缺口：

- matcher 是手写 `list(graph.nodes)`与 target 判断，不是页面 13 所教的 PatternExpr/PatternMatcher；
- legality 依赖真实输入运行 `ShapeProp`，不是 FakeTensor 或 dynamic guard；
- `same_dtype`有检查，但 device 没有显式 proof；layout 仅有一个 transposed x 实例；
- `alias_contract_matches`只检查 rewritten output 不 alias inputs，没有计算 eager 侧 relation 后比较；
- mutation 同样只检查 rewritten inputs 未变，不是 eager-vs-rewrite effect 差分；
- broadcast rejection 只比较 `gm.code`；ShapeProp 已经写入 metadata，所以这不是完整 graph+meta failure atomicity；
- 没有 assertion、randomized shape matrix、gradcheck、stage placement 或 artifact。

## 7. 已实际打开并核验的源码 locator

下表不是从旧文档复制的 locator 清单；本次审计逐一打开了目标范围。

| # | 已核验 locator | 源码事实 | 审计结论 |
|---:|---|---|---|
| 1 | `torch/fx/graph.py:1360-1393` | `_FindNodesLookupTable`按 call_function `(op,target)`建索引 | 支撑 candidate index；也说明 target 变化需重建索引 |
| 2 | `torch/fx/node.py:894-913` | `Node.__setattr__`变化前 remove、变化后 insert | `part3_legality.py`直接 retarget 在当前 main 会同步 side table |
| 3 | `torch/fx/graph.py:1525-1551` | `graph_copy`用 val_map/node_copy 并特殊处理 output | 12 正文方向正确，但缺 locator/Lab |
| 4 | `torch/fx/graph.py:1585-1661` | `create_node`更新 namespace、链表、side table 与长度 | 12 正确 |
| 5 | `torch/fx/graph.py:1675-1713` | `erase_node`拒绝 live/wrong graph，并清理输入 reverse uses | 12 正确 |
| 6 | `torch/fx/graph.py:2386-2420` | `node_copy`通过 arg_transform remap，复制浅 meta | 12 正确但未实验 |
| 7 | `torch/csrc/fx/node.cpp:307-359` | `_update_args_kwargs`删除旧 users、规范化 aggregate、建立新 users | 12 正确 |
| 8 | `torch/fx/node.py:713-757` | replace 先 snapshot users，支持 filter 与受限 meta propagation | 12 正确 |
| 9 | `torch/fx/graph.py:2610-2687` | lint 检查 owner、topo、side table、name、opcode 与 targets | 12/14 正确 |
| 10 | `torch/fx/graph.py:2690-2776` | reverse DCE、impurity predicate、referenced child recursion | 14 正确 |
| 11 | `torch/fx/node.py:760-808` | placeholder/output impure；call_function 委托 unified impurity | 14 正确且需保留“不完备”警告 |
| 12 | `torch/_inductor/pattern_matcher.py:483-523` | identity binding 与 multi-user pattern filter | 13 正确 |
| 13 | `torch/_inductor/pattern_matcher.py:876-1019` | normalized args/kwargs、users、aggregate、constants 递归匹配 | 13 正确 |
| 14 | `torch/_inductor/pattern_matcher.py:1162-1233` | MultiOutput root 约束、anchor search 与局部 rollback | 13 正确 |
| 15 | `torch/_inductor/pattern_matcher.py:1341-1402,1625-1652` | 三类 entry mutation ownership 与 replacement local cleanup | 13 正确；Lab未覆盖 |
| 16 | `torch/_inductor/pattern_matcher.py:1828-2092` | match-specific retrace、extra_check、replacement retrace | 13 正确；Lab未覆盖 |
| 17 | `torch/_inductor/pattern_matcher.py:2095-2224` | serialized/precompiled patterns 的生成与导入路径 | 13 规格缺项 |
| 18 | `torch/_inductor/pattern_matcher.py:2583-2726` | root lookup、candidate snapshot/sort、safety checks、single apply | 12/13/15 正确；Lab未调用 pass |
| 19 | `torch/_inductor/pattern_matcher.py:2946-2980` | waiting node wake-up后重新扫描全部 args | 14 的无条件线性复杂度错误 |
| 20 | `test/inductor/test_pattern_matcher.py:2201-2225` | stable sort 保留独立节点当前相对顺序并修复 misplaced producer | 14 的 stable 语义正确 |
| 21 | `torch/fx/passes/infra/pass_manager.py:154-192,254-317` | 默认 steps=1；在上限内按 modified 重复并提前停止 | 15 正确；现有 Lab只覆盖这一小点 |
| 22 | `torch/_inductor/fx_passes/pre_grad.py:336-433` | 非 functional/normalized 输入、counter、sort/lint/recompile | 15 正确 |
| 23 | `torch/_inductor/fx_passes/joint_graph.py:699-772` | canonicalize、ordered lists、局部 count 与条件 cleanup | 15 正确 |
| 24 | `torch/_inductor/fx_passes/post_grad.py:165-180,227-303,385-474` | post-grad DCE、main passes、collective sort、late mutation tail、final lint | 页面结论正确但原 locator 不完整 |
| 25 | `torch/_inductor/scheduler.py:5055-5098` | Scheduler DCE基于 active buffers、side effects 与 weak deps | 14 正确 |
| 26 | `torch/_inductor/fx_passes/post_grad.py:116-141` | fallback random ops用显式 deps 保持原顺序 | 支撑“topo 不等于 effect” |
| 27 | `torch/_inductor/fx_passes/control_dependencies.py:1-40,134-217` | control_deps 把 ordering-only deps 显式加入 FX | 14 应补的 current mechanism locator |
| 28 | `torch/_inductor/scheduler.py:6395-6462,6830-6884,8610-8675` | fusion 分 candidate、legality、score，部分路径可 benchmark | 16 关于“无通用单一收益公式”的判断成立 |

## 8. 复杂度修订基线

### 8.1 Matcher

定义：

- `V/E`：FX nodes / Node references；
- `C`：当前 matcher pass 物化的 root candidates，`C <= V`；
- `B(v)`：candidate `v`对应 root bucket 的 rule 数；
- `K(p)`：pattern AST 大小；
- `A(p,v)`：multi-output/repeated pattern 的 anchor/user exploration；
- `Q/U`：replacement nodes / rewritten uses。

当前结构引擎应写成：

```text
O(C) + O(C log C)
+ Σv Σp∈B(v) O(K(p) + A(p,v))
+ Σsuccessful O(Q + U + local erase)
```

还必须另列：

- `extra_check`是任意 Python，不能仅由 `V/E`给严格上界；
- traced replacement 会实际 trace/search/replacement function，其 tensor computation 可主导；
- custom graph handler 同样是外生成本。

分层表达：

- **常见**：root buckets、pattern size、arity、anchor fan-out 有界，且 `C << V`，结构匹配近候选线性；
- **期望**：只有声明候选分布、bucket 分布与 fan-out 分布后才有定义；当前文档没有这种模型，应写“未定义”而不是暗示平均线性；
- **严格参数化上界**：使用上式并显式保留 `A`、callback、trace/handler 成本，不能简化成无条件 `O(CP)`。

### 8.2 Stable topological sort

当前实现对等待节点每次 wake-up都重新构造 `waiting_for`，故：

```text
常见 bounded-arity：接近 O(V + E)
保守实现级上界：O(V + E + Σv d(v)^2)
辅助空间：O(V + E)
```

正文 14 与 16 都应使用这一分层；不能把另一个 heap-based helper 的复杂度直接套到
`torch/_inductor/pattern_matcher.py`的 in-place 实现。

### 8.3 Pipeline

不同 stage 的 graph size 不同。应使用：

```text
Σgraph g Σpass q (
  candidate_q,g + match_q,g + rewrite_q,g + cleanup_q,g
)
```

有界重复再乘各自实际 round 上限；不能用一个全局 `V`或把 ordered matcher lists 称为 fixed point。

## 9. 可执行修订方案

### 9.1 先纠正证据标签

在补代码前，立即把五页“已验证 Lab”改为“部分验证”，并加入统一表：

```text
Implemented:
Executed:
Observed:
Not covered:
Artifact:
```

具体删改错误声明：

1. 12 移除“`part3_passes.py`追加 node_copy”。
2. 13 移除“PatternMatcherPass 对 root candidate 运行”。
3. 15 移除“两个互逆 passes 形成 oscillation”。
4. 16 移除“Fake/meta 与 input alias 已比较”。

### 9.2 把新增 pass 升级为正式贯穿 harness

以现有 `labs/part3_end_to_end_pass.py`为唯一贯穿入口。它已经在窄合法域实现：

```text
search:      add(matmul(x, w), b)
replacement: addmm(b, x, w)
```

最小合法域至少约束：

- `x,w`均为 rank-2，inner dimension 相等；
- `b`只接受明确支持的 exact output shape，首版不要声称任意 broadcast；
- dtype/device 一致且 replacement 的 promotion/accumulation 语义相同；
- 无 mutation/effect crossing；
- output alias contract 与原式相同；
- fake/meta 可证明 shape；无法证明时拒绝，而不是读取 hint 后默认合法。

pass 输出必须包含：

- before/after graph；
- first apply count 与 second apply count；
- rejection reason；
- node/user counts；
- stage 与 graph id。

还需把当前打印值全部改为 assertion，并补：

- eager 与 rewritten 两侧的 alias relation 对比；
- eager 与 rewritten 两侧的输入 mutation/effect 对比；
- graph structure、targets、users 与 metadata 的 rejection 前后 hash；
- FakeTensor/symbolic legality path；
- device、layout、dtype/promotion、empty 与 randomized shape cases；
- PatternMatcher 与真实 stage hook，而不只是手写 FX target traversal。

### 9.3 把五篇 Lab 分层绑定到同一 pass

| 页面 | 必做实验 |
|---|---|
| 12 | insertion、replacement、`node_copy/graph_copy`；故意制造 live erase、cross-owner、self-cycle、output ABI mismatch；每个失败后断言原图未变 |
| 13 | direct AST、shared input、kwargs、MultiOutput、root bucket、Graph/Replacement/Lowering entry；验证 handler positional/keyword ABI 与 same-round no-revisit |
| 14 | pass 后 local erase + whole-graph DCE；dead chain；impure node；topo repair；effect reorder 失败与 control_deps 修复 |
| 15 | 同一规则在 pre-grad/post-grad 可见形态差异；specific-before-general；A↔B oscillation；贯穿 pass 第二次 apply=0 |
| 16 | static/dynamic、broadcast reject、transpose、empty/size-1、dtype/device、forward、grad、gradcheck、alias、mutation snapshot、determinism、failure atomicity |

### 9.4 使用 assertion，而不是把布尔值当日志

最低要求：

```text
torch.testing.assert_close(actual, expected)
assert first_apply_count == 1
assert second_apply_count == 0
assert normalized_graph_after_first == normalized_graph_after_second
assert rejected_graph_hash_before == rejected_graph_hash_after
assert input tensors unchanged
assert output/input alias relations equal between eager and rewritten paths
```

梯度至少包含：

- `torch.autograd.grad`逐输入比较；
- float64 `gradcheck`；
- 若页面继续声称 gradgrad，则必须实际运行 `gradgradcheck`，否则删除该覆盖声明。

### 9.5 固化 artifact bundle

建议路径：

```text
wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/
  labs/artifacts/part3/
    manifest.json
    editing_before.fx.txt
    editing_after.fx.txt
    matcher_events.jsonl
    dce_topology_effect.txt
    stage_runs.json
    legality_results.json
```

`manifest.json`至少记录：

```text
source_sha
torch_version
torch_git_version
python_version
platform
device
seed
command
exit_code
artifact sha256
checks and observed values
```

每篇正文给出从仓库根可复制的命令及相对 artifact 路径。

### 9.6 建议验收命令

实现修订后，至少运行：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part3_end_to_end_pass.py --artifact-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\part3
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part3_pattern.py --artifact-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\part3
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part3_passes.py --artifact-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\part3
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part3_legality.py --artifact-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\part3
```

最后机械检查：

```powershell
rg -n -g '*.md' "已验证 Lab|python -B|Artifact|预期|错误|边界" wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end
rg -n -g 'part3_*.py' "matmul|mm|addmm|assert|gradcheck|alias|mutation|second_apply" wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs
git diff --check
```

验收门槛是：命令 exit 0、所有 assertions 生效、manifest 与页面观察值一致、第二次 rewrite 不改图、拒绝路径保持图字节级/规范化结构不变；不能只看到一组 `...=True`打印。
