# PyTorch 图编译器学习系列设计符合性复核

> 最终复核日期：2026-07-28  
> 设计依据：`docs/superpowers/specs/2026-07-23-pytorch-graph-learning-series-design.md`  
> 固定源码基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> Lab 运行版本：PyTorch `2.9.1+cpu`，git `5811a8d7da873dd699ff6687092c225caffcf1bb`  
> 环境：Windows、Python 3.13.5、无 CUDA、无 MSVC `cl`、无 Triton

## 1. 最终结论

本轮重构已经满足“先建立设计背景，再解释对象关系、源码路径、不变量、失败边界与
复杂度”的课程要求。最终学习路径由 `00` 总索引与 `01–21` 正文组成，文件名、H1、
索引表和前后导航使用同一编号。

结论分三层：

| 范围 | 状态 | 结论 |
|---|---|---|
| 00–21 课程结构、原理解读与固定源码核验 | **PASS** | 21 篇均形成完整学习闭环；课程 claim gate 为 3131/3131 decisions、0 errors |
| 历史 28 页的无损保留、逐 claim 审计与纠错处置 | **PASS（含隔离）** | 2190/2190 claims 有决定；91 corrected，2099 unresolved 均 `retain-quarantined`，没有把未核验旧结论导入新主线 |
| 原生 C++ kernel 与 CUDA/Triton 实测 | **BLOCKED** | native probe 明确记录 `cpu_native=BLOCKED`、`cuda_triton=BLOCKED`；不得由 generated source 或 CPU 内部结构外推 |

“PASS（含隔离）”不表示 2099 条历史结论被证明正确，而表示设计要求的无损保留、
去向、处置和禁止导入均已闭合。历史材料仍可查阅，但 unresolved 内容不是新课程的权威
事实来源。

## 2. 文档体系与编号

最终编号为：

```text
00  总索引与源码阅读方法
01–06  图的概念与语义基础
07–11  捕获、规范化、AOT 正反向构图
12–16  改图、PatternMatcher、DCE、保序与验证
17–21  Lowering、Inductor IR、内存、Scheduler 与 Codegen
```

总索引新增“如何阅读本系列的源码部分”，要求每次源码跟读回答六个问题：

1. driver 入口是谁；
2. 读取哪套状态；
3. 写回哪套状态；
4. 不变量在哪里检查；
5. 后续 consumer 是谁；
6. 跨阶段 identity 是否保留。

该方法避免从抽象基类或单个 API 孤立推断完整机制。

## 3. 本轮源码级增强

### 3.1 Part I：图、FX 数据结构与横切语义

- 解释 eager autograd `Edge/Node`、FX use-def、AOT fresh graph、Inductor IR 与 Scheduler
  graph 为何不能共用一种通用 Graph 数据结构。
- 建立完整 FX 构图链：
  `symbolic_trace → Tracer.trace → create_proxy → Graph.create_node → GraphModule`。
- 建立完整 FX 改图链：
  `replace_all_uses_with → _update_args_kwargs → erase_node → lint → recompile`。
- 区分 Node identity、`node.meta["val"]`、Interpreter `env[Node]` 与 graph signature。
- 追踪 real size/source 如何进入 ShapeEnv、SymNode、guard/deferred assert 与 cache reuse。
- 追踪 functionalization、effect token、input mutation writeback 与 nested GraphModule ownership。

### 3.2 Part II：捕获、规范化与正反向构图

- 对比 symbolic FX、make_fx、Dynamo、export 的拦截层、输入值与产物。
- 追踪 decomposition dispatch、functionalized wrapper、duplicate input/synthetic base 与
  post-capture checks。
- 从 `create_joint()`、`autograd.grad()`到 partitioner 的 fresh fw/bw graph extraction。
- 明确 fw/bw 没有跨 Graph Node 边；连接由 fw 额外输出、runtime 保存槽位、bw
  placeholder 与 backward 参数组装共同形成。
- 对 min-cut save/recompute、recompute `node_copy`、backward reorder 和 runtime ABI
  分别定位，避免把“重算”误解为 bw 通过特殊边回到 fw。
- 解释跨阶段 relation 是 lineage/provenance，不是稳定对象 identity。

### 3.3 Part III：改图、匹配、DCE 与 pass

- 将 FX rewrite 写成显式状态机：插入、重连、擦除、复制、graph setter/recompile。
- 完整解释 `PatternExpr`、`MatchContext`、`Arg/KeywordArg/Ignored`、
  `CallFunction`五类 gate、users 约束、`MultiOutputPattern` anchor/rollback。
- 明确 PatternMatcher 只遍历注册 root 的候选并使用 candidate snapshot；不是每个 pattern
  对全图做无索引穷举。
- 区分 FX DCE、nested graph DCE、stable topo 与 Scheduler DCE。
- 追踪 pre-grad、joint、post-grad driver 的真实 pass 顺序、bounded repeat 与 late reinplace。
- 将 matcher mutation/stream/mempool 检查、stage contract、GraphLowering layout constraint、
  失败原子性和外部成本放入统一合法性矩阵。

### 3.4 Part IV：Lowering、IR、内存、Scheduler 与 Codegen

- 建立
  `GraphLowering.run → Interpreter.run → run_node → call_function → lowering/fallback`
  调用链，并区分 `env`、lazy IR 与 registered operation 三套状态。
- 解释 `register_lowering` wrapper 的 broadcast/type promotion/IR validation 职责。
- 区分 realization 的决策位置与执行位置；output、extern、stream/mempool、mutation、
  repeated-read hint 分别有真实调用点。
- 将 Inductor IR 拆成计算关系、存储关系和调度关系，解释为何不能用万能 IRNode。
- 将 dependency 入口追到
  `LoopBody → extract_read_writes → MemoryDep`，而不是只引用 Scheduler 类型分派。
- 解释 alias user 合并、mutation rename、last-use alias closure 和 wrapper reuse key。
- 修正 Scheduler topo 为
  `O(V+E+Σ d_v log d_v)`；fusion candidates 为
  `O(Σ n_g·min(n_g,W))`，当前默认 `W=64`，aggressive fusion 不应被无条件写成二次。
- 建立
  `Scheduler group → backend scheduling → kernel/call + wrapper → PyCodeCache module`
  调用链，并区分 FX graph cache、Python/C++ code cache、algorithm cache 与 Triton
  config cache。

## 4. 课程 claim evidence

正式数据源：

- `docs/audits/pytorch_graph_series/2026-07-27/course_claim_ledger.jsonl`
- `docs/audits/pytorch_graph_series/2026-07-27/course_claim_summary.md`
- `docs/audits/pytorch_graph_series/2026-07-27/course_claim_errors.json`

最终结果：

| Evidence class | 数量 | 含义 |
|---|---:|---|
| `[S]` | 916 | 固定 PyTorch checkout 的窄源码证据 |
| `[R]` | 366 | 当前 CPU 环境真实执行 |
| `[I]` | 1284 | 具有已验证父 claim 与逐父 rationale 的推论 |
| `[M]` | 19 | codegen-only/mock boundary，native kernel 未执行 |
| `[B]` | 41 | 环境阻塞或尚未取得目标能力证据 |
| nonassertive | 505 | 导航、问题、命令、展示代码等非事实单元 |

总计 3131 个 claim/table-row/code 单元，3131 个 decisions，0 个 validation error。
所有 `[S]` source range 不超过 30 行；所有 `[R]/[M]/[B]` 绑定 producer 脚本哈希、
tokenized command、退出码、JSON artifact 哈希与 selector。

## 5. 历史资料审计

正式数据源：

- `legacy_claim_ledger.jsonl`
- `legacy_claim_closure.json`
- `legacy_unit_decisions.jsonl`
- `legacy_correction_dispositions.json`

结果：

| 项目 | 结果 |
|---|---:|
| 冻结历史页 | 28 |
| claim candidates / decisions | 2190 / 2190 |
| corrected | 91 |
| unresolved、保留并隔离 | 2099 |
| structural unit decisions | 1602 |
| correction catalog / dispositions | 94 / 94 |
| corrected without local callout | 0 |
| destination validation errors | 0 |
| unresolved claims imported by new series | 0 |

旧正文没有删除。未取得足够证据的内容保持 `retain-quarantined`；新课程不依赖这些
unresolved claims。

## 6. 设计 §12 验收

| # | 验收项 | 状态 | 证据 |
|---:|---|---|---|
| 1 | 索引从概念引导到 codegen | PASS | 00 总图、四部分地图、三条路径、源码入口表 |
| 2 | 21 篇形成完整学习闭环 | PASS | 动机、对象、源码、边界、复杂度、Lab、导航齐全 |
| 3 | 旧结构单元有 ledger 去向 | PASS（隔离） | 1602 structural decisions；未核验项保留隔离 |
| 4 | 历史错漏有 correction 记录 | PASS | 94/94 dispositions，91 local corrected claims |
| 5 | 权威实现断言绑定固定基线 | PASS | course claim gate 0 errors |
| 6 | 正式 Lab 真实运行 | PASS（能力内） | 21/21 producer receipt 退出码 0 |
| 7 | 贯穿模型生成阶段 artifacts | PARTIAL | AOT→Inductor 连续 CPU/extern 路径已证；native generated kernel 仍 BLOCKED |
| 8 | Part III 语义验证 | PASS | 数值、梯度、shape、alias/mutation、原子性、幂等性合同 |
| 9 | Scheduler/kernel 反查 FX/source | PASS（artifact 级） | provenance join；不是 runtime PC mapping |
| 10 | 链接、围栏、Related、编号等 | PASS | delivery gate 0 errors |

## 7. 验证记录

### 审计工具

```text
Ran 90 tests
OK
```

### Lab 与 native 合同

```text
Ran 42 tests
OK
```

### 课程 claim gate

```json
{"claims": 3131, "decisions": 3131, "errors": 0, "ledger_rows": 3131}
```

### 结构门禁

```json
{"course_files": 22, "legacy_files": 28, "errors": 0}
```

### Runtime receipt

```text
course_runtime_all_scripts_passed=True
completed_script_count=21
```

### Native probe

```json
{"cpu_native": "BLOCKED", "cuda_triton": "BLOCKED"}
```

## 8. 保留的环境阻塞

以下内容不能标记为已完成：

- generated C++ pointwise/reduction kernel 的本机编译、加载、执行与数值对照；
- CUDA/Triton kernel、真实 autotune candidates/timings/winner/cache；
- GPU allocator snapshot/trace；
- native fusion/reorder 性能对照；
- runtime PC/profiler sample 到 generated source 的映射。

这些阻塞不会削弱固定源码机制与当前 CPU 可观察结构的结论，但禁止把 `[M]`或 `[B]`
升级成 `[R]`。

## 9. 交付决定

1. 00–21 可作为固定源码基线下的系统学习主线。
2. 历史 28 页继续保留；unresolved 内容只作历史线索，不作权威来源。
3. native CPU/CUDA 能力保持外部环境 gate。
4. 本轮不创建、不暂存、不提交 Git commit。
